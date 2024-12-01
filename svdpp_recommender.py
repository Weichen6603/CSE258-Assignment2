import os
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, ndcg_score
from sklearn.model_selection import train_test_split
import logging
from datetime import datetime
import matplotlib.pyplot as plt


class SvdppRecommender:
    def __init__(self,
                 n_factors=150,
                 n_epochs=50,
                 learning_rate=0.005,
                 regularization=0.02,
                 early_stopping_rounds=5,
                 min_improvement=0.0001,
                 learning_rate_decay=0.96,
                 random_state=42):
        """
        Initialize improved SVD++ recommender

        Parameters:
            n_factors: number of latent factors
            n_epochs: maximum number of training epochs
            learning_rate: initial learning rate
            regularization: regularization parameter
            early_stopping_rounds: number of rounds for early stopping
            min_improvement: minimum improvement threshold
            learning_rate_decay: learning rate decay factor
            random_state: random seed
        """
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.initial_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.early_stopping_rounds = early_stopping_rounds
        self.min_improvement = min_improvement
        self.learning_rate_decay = learning_rate_decay

        # Set random seed
        np.random.seed(random_state)

        # Initialize logger
        self._setup_logger()

        # Training history
        self.history = {
            'train_rmse': [],
            'val_rmse': [],
            'learning_rates': []
        }

    def _setup_logger(self):
        """Setup logger to save logs in the 'log' folder."""
        # Ensure the log directory exists
        log_dir = 'log'
        os.makedirs(log_dir, exist_ok=True)

        # Set up logger
        self.logger = logging.getLogger('ImprovedSVDpp')
        self.logger.setLevel(logging.INFO)

        # Avoid adding duplicate handlers
        if not self.logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

            # File handler
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = os.path.join(log_dir, f'svdpp_training_{timestamp}.log')
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        self.logger.info("Logger initialized successfully!")

    def save_model(self, file_path):
        """
        Save model to a file using pickle.

        Parameters:
            file_path: path to save the model.
        """
        try:
            with open(file_path, 'wb') as file:
                pickle.dump(self, file)
            self.logger.info(f"Model saved successfully to {file_path}")
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")

    @staticmethod
    def load_model(file_path):
        """
        Load model from a file using pickle.

        Parameters:
            file_path: path to load the model from.

        Returns:
            Loaded SvdppRecommender instance.
        """
        try:
            with open(file_path, 'rb') as file:
                model = pickle.load(file)
            logging.info(f"Model loaded successfully from {file_path}")
            return model
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            return None


    def _init_model_parameters(self, n_users, n_items):
        """Initialize model parameters using He initialization"""
        self.user_factors = np.random.normal(0, np.sqrt(2.0 / self.n_factors),
                                             (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, np.sqrt(2.0 / self.n_factors),
                                             (n_items, self.n_factors))

        self.user_biases = np.zeros(n_users)
        self.item_biases = np.zeros(n_items)

        self.user_implicit_factors = np.random.normal(0, np.sqrt(2.0 / self.n_factors),
                                                      (n_items, self.n_factors))

    def fit(self, ratings_df, validation_size=0.1, verbose=True):
        """
        Train the model

        Parameters:
            ratings_df: ratings DataFrame
            validation_size: proportion of data for validation
            verbose: whether to print progress information
        """
        self.logger.info("Starting improved SVD++ model training...")

        # Create user and item mappings
        self.user_mapping = {user: idx for idx, user in
                             enumerate(ratings_df['userId'].unique())}
        self.item_mapping = {item: idx for idx, item in
                             enumerate(ratings_df['movieId'].unique())}

        n_users = len(self.user_mapping)
        n_items = len(self.item_mapping)

        # Initialize model parameters
        self._init_model_parameters(n_users, n_items)

        # Calculate global mean
        self.global_mean = ratings_df['rating'].mean()

        # Split training and validation data
        train_data, val_data = train_test_split(
            ratings_df,
            test_size=validation_size,
            random_state=42
        )

        # Create user rating history dictionary
        self.user_rated_items = self._create_user_rated_items(train_data)

        # Training loop
        best_val_rmse = float('inf')
        no_improvement_count = 0
        best_epoch = 0

        for epoch in range(self.n_epochs):
            # Train one epoch
            train_rmse = self._train_epoch(train_data)

            # Evaluate on validation set
            val_rmse = self._calculate_validation_rmse(val_data)

            # Record history
            self.history['train_rmse'].append(train_rmse)
            self.history['val_rmse'].append(val_rmse)
            self.history['learning_rates'].append(self.learning_rate)

            if verbose:
                self.logger.info(
                    f"Epoch {epoch + 1}/{self.n_epochs} - "
                    f"Train RMSE: {train_rmse:.4f}, "
                    f"Val RMSE: {val_rmse:.4f}, "
                    f"LR: {self.learning_rate:.6f}"
                )

            # Early stopping check
            if val_rmse < best_val_rmse - self.min_improvement:
                best_val_rmse = val_rmse
                best_epoch = epoch
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            # Learning rate decay
            self.learning_rate *= self.learning_rate_decay

            # Early stopping
            if no_improvement_count >= self.early_stopping_rounds:
                self.logger.info(
                    f"Early stopping triggered at epoch {epoch + 1}. "
                    f"Best epoch was {best_epoch + 1}"
                )
                break

        self.logger.info("Model training completed!")
        return self

    def _create_user_rated_items(self, ratings_df):
        """Create user rating history dictionary"""
        user_rated_items = {}
        for user, group in ratings_df.groupby('userId'):
            if user in self.user_mapping:
                user_idx = self.user_mapping[user]
                user_rated_items[user_idx] = [
                    self.item_mapping[item]
                    for item in group['movieId']
                    if item in self.item_mapping
                ]
        return user_rated_items

    def _train_epoch(self, train_data):
        """Train one epoch"""
        epoch_loss = []

        train_data = train_data.sample(frac=1).reset_index(drop=True)

        for _, row in train_data.iterrows():
            if row['userId'] in self.user_mapping and \
                    row['movieId'] in self.item_mapping:
                user_idx = self.user_mapping[row['userId']]
                item_idx = self.item_mapping[row['movieId']]
                rating = row['rating']

                pred = self._predict_one(user_idx, item_idx)
                error = rating - pred
                epoch_loss.append(error ** 2)

                self._update_parameters(user_idx, item_idx, error)

        return np.sqrt(np.mean(epoch_loss))

    def _predict_one(self, user_idx, item_idx):
        """Predict single rating"""
        baseline = self.global_mean + \
                   self.user_biases[user_idx] + \
                   self.item_biases[item_idx]

        user_item_interaction = np.dot(
            self.user_factors[user_idx],
            self.item_factors[item_idx]
        )

        rated_items = self.user_rated_items.get(user_idx, [])
        if rated_items:
            implicit_feedback = np.sum(self.user_implicit_factors[rated_items], axis=0)
            implicit_feedback /= np.sqrt(len(rated_items))
            implicit_term = np.dot(implicit_feedback, self.item_factors[item_idx])
        else:
            implicit_term = 0

        return baseline + user_item_interaction + implicit_term

    def _update_parameters(self, user_idx, item_idx, error):
        """Update model parameters"""
        # Update biases
        self.user_biases[user_idx] += self.learning_rate * \
                                      (error - self.regularization * self.user_biases[user_idx])
        self.item_biases[item_idx] += self.learning_rate * \
                                      (error - self.regularization * self.item_biases[item_idx])

        # Update factors
        user_factors = self.user_factors[user_idx]
        item_factors = self.item_factors[item_idx]

        user_factors_grad = error * item_factors - \
                            self.regularization * user_factors
        item_factors_grad = error * user_factors - \
                            self.regularization * item_factors

        self.user_factors[user_idx] += self.learning_rate * user_factors_grad
        self.item_factors[item_idx] += self.learning_rate * item_factors_grad

        # Update implicit feedback factors
        rated_items = self.user_rated_items.get(user_idx, [])
        if rated_items:
            sqrt_rated = 1.0 / np.sqrt(len(rated_items))
            for rated_item in rated_items:
                implicit_grad = sqrt_rated * (error * item_factors - \
                                              self.regularization * \
                                              self.user_implicit_factors[rated_item])
                self.user_implicit_factors[rated_item] += \
                    self.learning_rate * implicit_grad

    def _calculate_validation_rmse(self, val_data):
        """Calculate validation RMSE"""
        predictions = []
        actuals = []

        for _, row in val_data.iterrows():
            if row['userId'] in self.user_mapping and \
                    row['movieId'] in self.item_mapping:
                user_idx = self.user_mapping[row['userId']]
                item_idx = self.item_mapping[row['movieId']]
                pred = self._predict_one(user_idx, item_idx)
                predictions.append(pred)
                actuals.append(row['rating'])

        return np.sqrt(mean_squared_error(actuals, predictions))

    def predict(self, user_id, movie_id):
        """Predict rating for specific user and movie"""
        if user_id not in self.user_mapping or \
                movie_id not in self.item_mapping:
            return self.global_mean

        user_idx = self.user_mapping[user_id]
        item_idx = self.item_mapping[movie_id]

        return self._predict_one(user_idx, item_idx)

    def get_recommendations(self, user_id, n=10):
        """Generate recommendations for user"""
        try:
            if user_id not in self.user_mapping:
                self.logger.warning(f"User {user_id} not found in mapping")
                return []

            user_idx = self.user_mapping[user_id]
            predictions = []

            # 获取用户已评分的电影
            rated_movies = self.user_rated_items.get(user_idx, [])

            # 为所有未评分的电影生成预测
            for movie_id, item_idx in self.item_mapping.items():
                if item_idx not in rated_movies:  # 修改这里的判断条件
                    pred_rating = self._predict_one(user_idx, item_idx)
                    predictions.append((movie_id, pred_rating))

            # 排序并返回top-N推荐
            recommendations = sorted(predictions, key=lambda x: x[1], reverse=True)[:n]
            return recommendations

        except Exception as e:
            self.logger.error(f"Error in generating recommendations: {e}")
            return []

    def plot_training_history(self):
        """Plot training history"""
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_rmse'], label='Train RMSE')
        plt.plot(self.history['val_rmse'], label='Validation RMSE')
        plt.xlabel('Epoch')
        plt.ylabel('RMSE')
        plt.title('RMSE vs. Epoch')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.history['learning_rates'])
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Decay')

        plt.tight_layout()
        plt.show()


class SVDppEvaluator:
    def __init__(self):
        self.logger = logging.getLogger('SVDppEvaluator')
        self.logger.setLevel(logging.INFO)

    def evaluate(self, model, test_data):
        """Evaluate model performance"""
        predictions = []
        actuals = []

        for _, row in test_data.iterrows():
            pred = model.predict(row['userId'], row['movieId'])
            if pred is not None:
                predictions.append(pred)
                actuals.append(row['rating'])

        predictions = np.array(predictions)
        actuals = np.array(actuals)

        return {
            'RMSE': np.sqrt(mean_squared_error(actuals, predictions)),
            'MAE': mean_absolute_error(actuals, predictions),
            'NDCG': ndcg_score(actuals.reshape(1, -1), predictions.reshape(1, -1))
        }


def main():
    """Main function for example usage"""
    # Load data
    ratings_df = pd.read_csv('./dataset/ratings_small.csv')

    # Split train and test data
    train_data, test_data = train_test_split(
        ratings_df, test_size=0.2, random_state=42
    )

    # Initialize and train model
    model = SvdppRecommender(
        n_factors=150,
        n_epochs=50,
        learning_rate=0.005,
        regularization=0.02,
        early_stopping_rounds=5,
        min_improvement=0.0001,
        learning_rate_decay=0.96
    )

    # Train model
    model.fit(train_data, validation_size=0.1, verbose=True)

    # Save the trained model
    model.save_model('./model/svdpp_model.pkl')

    # Load the model back
    loaded_model = SvdppRecommender.load_model('svdpp_model.pkl')

    # Verify loaded model
    if loaded_model:
        # Plot training history
        loaded_model.plot_training_history()

        # Evaluate model
        evaluator = SVDppEvaluator()
        metrics = evaluator.evaluate(loaded_model, test_data)

        print("\nModel Evaluation Results:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")

        # Generate recommendations for sample user
        sample_user = ratings_df['userId'].iloc[0]
        recommendations = loaded_model.get_recommendations(sample_user, n=5)

        print(f"\nTop 5 recommendations for user {sample_user}:")
        for movie_id, pred_rating in recommendations:
            print(f"Movie ID: {movie_id}, Predicted Rating: {pred_rating:.2f}")


if __name__ == "__main__":
    main()
