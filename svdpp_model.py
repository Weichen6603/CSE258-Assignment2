import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, ndcg_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.multiprocessing as mp

# Import the updated MovieRatingDataset
from dataset import MovieRatingDataset


class SVDppModel(nn.Module):
    def __init__(self, n_users, n_items, n_factors,
                 n_categorical_features=0, n_numerical_features=0,
                 categorical_dims=None, embed_dim=10):
        super(SVDppModel, self).__init__()
        self.n_factors = n_factors

        # User and item embeddings
        self.user_embeddings = nn.Embedding(n_users, n_factors)
        self.item_embeddings = nn.Embedding(n_items, n_factors)
        self.user_implicit_embeddings = nn.Embedding(n_items, n_factors)

        # Bias terms
        self.user_biases = nn.Embedding(n_users, 1)
        self.item_biases = nn.Embedding(n_items, 1)

        # Global bias
        self.global_bias = nn.Parameter(torch.zeros(1))

        # Categorical feature embeddings
        self.categorical_embeddings = None
        if n_categorical_features > 0 and categorical_dims is not None:
            self.categorical_embeddings = nn.ModuleList([
                nn.Embedding(input_dim, embed_dim) for input_dim in categorical_dims
            ])
            self.categorical_embed_dim = n_categorical_features * embed_dim
        else:
            self.categorical_embed_dim = 0

        # Numerical feature layer
        self.numerical_layer = None
        if n_numerical_features > 0:
            self.numerical_layer = nn.Linear(n_numerical_features, n_factors)

        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(n_factors + self.categorical_embed_dim, n_factors),
            nn.ReLU(),
            nn.Linear(n_factors, 1)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using He initialization"""
        nn.init.normal_(self.user_embeddings.weight, 0, np.sqrt(2.0 / self.n_factors))
        nn.init.normal_(self.item_embeddings.weight, 0, np.sqrt(2.0 / self.n_factors))
        nn.init.normal_(self.user_implicit_embeddings.weight, 0, np.sqrt(2.0 / self.n_factors))
        nn.init.zeros_(self.user_biases.weight)
        nn.init.zeros_(self.item_biases.weight)
        if self.numerical_layer is not None:
            nn.init.xavier_uniform_(self.numerical_layer.weight)
            nn.init.zeros_(self.numerical_layer.bias)
        if self.output_layer is not None:
            for layer in self.output_layer:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def forward(self, user_idx, item_idx, numerical_features=None, categorical_features=None):
        # Base SVD++ computation
        user_embed = self.user_embeddings(user_idx)
        item_embed = self.item_embeddings(item_idx)
        user_bias = self.user_biases(user_idx).squeeze()
        item_bias = self.item_biases(item_idx).squeeze()

        # Implicit feedback
        implicit_feedback = torch.mean(self.user_implicit_embeddings(item_idx), dim=0)

        # Interaction term
        interaction = user_embed * item_embed + implicit_feedback

        # Add numerical features
        if numerical_features is not None and self.numerical_layer is not None:
            numerical_output = self.numerical_layer(numerical_features)
            interaction += numerical_output

        # Add categorical features
        if categorical_features is not None and self.categorical_embeddings is not None:
            categorical_embeds = []
            for i, embed_layer in enumerate(self.categorical_embeddings):
                categorical_embeds.append(embed_layer(categorical_features[:, i]))
            categorical_embeds = torch.cat(categorical_embeds, dim=1)
            interaction = torch.cat([interaction, categorical_embeds], dim=1)

        # Final prediction
        pred = self.output_layer(interaction).squeeze()
        pred = pred + user_bias + item_bias + self.global_bias

        return pred


class GPUSvdppRecommender:
    def __init__(self,
                 n_factors=100,
                 n_epochs=1000,
                 batch_size=2048,
                 learning_rate=0.001,
                 weight_decay=0.02,
                 dropout_rate=0.2,
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 num_workers=4):
        # Initialize parameters
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate
        self.device = device
        self.num_workers = num_workers

        # Set up logging
        self._setup_logger()
        self.logger.info(f"Initializing SVD++ model with {n_factors} factors on {device}")

        # Model related attributes
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.user_mapping = None
        self.item_mapping = None

        # Training history
        self.history = {
            'train_loss': [],
            'val_metrics': []
        }

        # Create model save directory
        self.model_dir = 'model'
        os.makedirs(self.model_dir, exist_ok=True)
        self.logger.info(f"Model directory created/verified at {self.model_dir}")

    def _setup_logger(self):
        # Create logger instance
        self.logger = logging.getLogger('GPUSVDpp')
        self.logger.setLevel(logging.INFO)

        # Create log directory
        log_dir = 'log'
        os.makedirs(log_dir, exist_ok=True)

        if not self.logger.handlers:
            # 1. Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)

            # 2. File handler
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = os.path.join(log_dir, f'training_{timestamp}.log')
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)

            # Log format
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            file_handler.setFormatter(formatter)

            # Add handlers
            self.logger.addHandler(console_handler)
            self.logger.addHandler(file_handler)

    def fit(self, ratings_df, validation_size=0.1, patience=10, min_improvement=0.0001):
        """Enhanced training function with multi-metric early stopping"""
        self.logger.info(f"Training on {self.device}")

        # Create user and item mappings
        self.user_mapping = {user: idx for idx, user
                             in enumerate(ratings_df['userId'].unique())}
        self.item_mapping = {item: idx for idx, item
                             in enumerate(ratings_df['movieId'].unique())}

        # Define numerical and categorical features
        numerical_features = ['user_avg_rating', 'user_rating_count', 'item_avg_rating', 'item_rating_count']
        categorical_features = ['year', 'month', 'day_of_week', 'hour']

        # Compute categorical dimensions
        categorical_dims = []
        for feature in categorical_features:
            categorical_dims.append(ratings_df[feature].nunique())

        # Standardize numerical features
        scaler = StandardScaler()
        ratings_df[numerical_features] = scaler.fit_transform(ratings_df[numerical_features])

        # Split training and validation data
        train_data, val_data = train_test_split(
            ratings_df, test_size=validation_size, random_state=42
        )

        # Create datasets and data loaders
        train_dataset = MovieRatingDataset(
            train_data, self.user_mapping, self.item_mapping,
            categorical_features=categorical_features,
            numerical_features=numerical_features
        )
        val_dataset = MovieRatingDataset(
            val_data, self.user_mapping, self.item_mapping,
            categorical_features=categorical_features,
            numerical_features=numerical_features
        )

        # Specify multiprocessing context
        multiprocessing_context = mp.get_context('spawn')

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            multiprocessing_context=multiprocessing_context
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            multiprocessing_context=multiprocessing_context
        )

        # Initialize model
        self.model = SVDppModel(
            n_users=len(self.user_mapping),
            n_items=len(self.item_mapping),
            n_factors=self.n_factors,
            n_categorical_features=len(categorical_features),
            n_numerical_features=len(numerical_features),
            categorical_dims=categorical_dims
        ).to(self.device)

        # Initialize optimizer and criterion
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        self.criterion = nn.MSELoss()

        # Initialize early stopping variables
        best_metrics = {
            'rmse': float('inf'),
            'mae': float('inf'),
            'ndcg': 0
        }
        patience_counter = 0
        best_epoch = 0

        # Training loop
        for epoch in range(self.n_epochs):
            train_loss = self._train_epoch(train_loader)
            val_metrics = self._validate(val_loader)

            self.history['train_loss'].append(train_loss)
            self.history['val_metrics'].append(val_metrics)

            self.logger.info(
                f'Epoch {epoch + 1}/{self.n_epochs}: '
                f'Train RMSE = {train_loss:.4f}, '
                f'Val RMSE = {val_metrics["rmse"]:.4f}, '
                f'Val MAE = {val_metrics["mae"]:.4f}, '
                f'Val NDCG = {val_metrics["ndcg"]:.4f}'
            )

            # Check for improvement
            improved = False
            if (val_metrics['rmse'] < best_metrics['rmse'] - min_improvement or
                    val_metrics['mae'] < best_metrics['mae'] - min_improvement or
                    val_metrics['ndcg'] > best_metrics['ndcg'] + min_improvement):

                improved = True
                best_metrics = val_metrics.copy()
                best_epoch = epoch
                patience_counter = 0

                # Save best model
                self.save_model('best_model.pt')
                self.logger.info("New best model saved!")
            else:
                patience_counter += 1

            # Early stopping check
            if patience_counter >= patience:
                self.logger.info(
                    f"Early stopping triggered after {epoch + 1} epochs. "
                    f"Best epoch was {best_epoch + 1} with metrics: "
                    f"RMSE = {best_metrics['rmse']:.4f}, "
                    f"MAE = {best_metrics['mae']:.4f}, "
                    f"NDCG = {best_metrics['ndcg']:.4f}"
                )
                break

        # Load best model
        self.load_model('best_model.pt')
        return self

    def _train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_batches = 0

        pbar = tqdm(train_loader, desc='Training', leave=False)
        for batch in pbar:
            self.optimizer.zero_grad()

            user_idx = batch['user_idx'].to(self.device)
            item_idx = batch['item_idx'].to(self.device)
            ratings = batch['rating'].to(self.device)

            numerical_features = batch['numerical_features'].to(self.device)
            categorical_features = batch['categorical_features'].to(self.device)

            predictions = self.model(user_idx, item_idx, numerical_features, categorical_features)
            loss = self.criterion(predictions, ratings)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_batches += 1

            # Update progress bar
            pbar.set_postfix({
                'batch_loss': loss.item(),
                'avg_loss': total_loss / total_batches
            })

        return np.sqrt(total_loss / total_batches)

    def _validate(self, val_loader):
        """Validate and compute multiple metrics"""
        self.model.eval()
        total_loss = 0
        total_batches = 0
        all_predictions = []
        all_ratings = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validating', leave=False):
                user_idx = batch['user_idx'].to(self.device)
                item_idx = batch['item_idx'].to(self.device)
                ratings = batch['rating'].to(self.device)

                numerical_features = batch['numerical_features'].to(self.device)
                categorical_features = batch['categorical_features'].to(self.device)

                predictions = self.model(user_idx, item_idx, numerical_features, categorical_features)
                loss = self.criterion(predictions, ratings)

                total_loss += loss.item()
                total_batches += 1

                all_predictions.extend(predictions.cpu().numpy())
                all_ratings.extend(ratings.cpu().numpy())

        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_ratings = np.array(all_ratings)

        # Compute multiple metrics
        metrics = {
            'rmse': np.sqrt(mean_squared_error(all_ratings, all_predictions)),
            'mae': mean_absolute_error(all_ratings, all_predictions),
            'ndcg': ndcg_score(
                all_ratings.reshape(1, -1),
                all_predictions.reshape(1, -1)
            )
        }

        return metrics

    def get_model_path(self, filename):
        """Get full path for model file in the model directory"""
        return os.path.join(self.model_dir, filename)

    def save_model(self, filename):
        """Save model state, mappings, and training history to model directory"""
        try:
            # Ensure we're saving to the model directory
            path = self.get_model_path(filename)

            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'user_mapping': self.user_mapping,
                'item_mapping': self.item_mapping,
                'history': self.history,
                'model_config': {
                    'n_factors': self.n_factors,
                    'dropout_rate': self.dropout_rate
                }
            }

            torch.save(checkpoint, path)
            self.logger.info(f"Model saved successfully to {path}")

        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            raise

    def load_model(self, filename):
        """Load saved model state and mappings from model directory"""
        try:
            path = self.get_model_path(filename)

            if not os.path.exists(path):
                raise FileNotFoundError(f"Model file not found: {path}")

            checkpoint = torch.load(path, map_location=self.device)

            # Load model configuration and reinitialize if needed
            if self.model is None:
                model_config = checkpoint['model_config']
                self.n_factors = model_config['n_factors']
                self.dropout_rate = model_config['dropout_rate']
                # Model will be reinitialized in fit() method

            # Load state dictionaries
            if self.model is not None:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            if self.optimizer is not None:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # Load mappings and history
            self.user_mapping = checkpoint['user_mapping']
            self.item_mapping = checkpoint['item_mapping']
            self.history = checkpoint['history']

            self.logger.info(f"Model loaded successfully from {path}")

        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise

    def plot_training_history(self):
        """Plot training metrics history"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        # Plot RMSE
        ax1.plot([m['rmse'] for m in self.history['val_metrics']], label='Validation')
        ax1.plot(self.history['train_loss'], label='Train')
        ax1.set_title('RMSE vs. Epoch')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('RMSE')
        ax1.legend()
        ax1.grid(True)

        # Plot MAE
        ax2.plot([m['mae'] for m in self.history['val_metrics']])
        ax2.set_title('MAE vs. Epoch')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.grid(True)

        # Plot NDCG
        ax3.plot([m['ndcg'] for m in self.history['val_metrics']])
        ax3.set_title('NDCG vs. Epoch')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('NDCG')
        ax3.grid(True)

        plt.tight_layout()

        # Save the plot
        plot_path = os.path.join('log', f'training_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.savefig(plot_path)
        plt.close()


def main():
    # Load data
    ratings_df = pd.read_csv('./dataset/ratings_small.csv')
    # ratings_df = pd.read_csv('./dataset/ratings_mid.csv')
    # ratings_df = pd.read_csv('./dataset/ratings.csv')


    # Convert timestamp to datetime
    ratings_df['timestamp'] = pd.to_datetime(ratings_df['timestamp'], unit='s')

    # Extract time features
    ratings_df['year'] = ratings_df['timestamp'].dt.year
    ratings_df['month'] = ratings_df['timestamp'].dt.month
    ratings_df['day'] = ratings_df['timestamp'].dt.day
    ratings_df['day_of_week'] = ratings_df['timestamp'].dt.dayofweek
    ratings_df['hour'] = ratings_df['timestamp'].dt.hour

    # Compute user features
    user_stats = ratings_df.groupby('userId').agg({
        'rating': ['mean', 'count']
    }).reset_index()
    user_stats.columns = ['userId', 'user_avg_rating', 'user_rating_count']

    # Compute item features
    item_stats = ratings_df.groupby('movieId').agg({
        'rating': ['mean', 'count']
    }).reset_index()
    item_stats.columns = ['movieId', 'item_avg_rating', 'item_rating_count']

    # Merge statistical features into ratings_df
    ratings_df = ratings_df.merge(user_stats, on='userId', how='left')
    ratings_df = ratings_df.merge(item_stats, on='movieId', how='left')

    # Initialize recommender
    recommender = GPUSvdppRecommender(
        n_factors=100,
        n_epochs=1000,
        batch_size=1024,
        learning_rate=0.001,
        weight_decay=0.02,
        num_workers=8
    )

    # Train model
    recommender.fit(
        ratings_df,
        validation_size=0.1,
        patience=10,
        min_improvement=0.0001
    )

    # Plot training history
    recommender.plot_training_history()

    # Evaluate final model
    print("\nBest Model Metrics:")
    for metric, value in recommender.history['val_metrics'][-1].items():
        print(f"{metric.upper()}: {value:.4f}")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
