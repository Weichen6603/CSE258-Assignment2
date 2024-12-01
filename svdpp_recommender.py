import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, ndcg_score
from sklearn.model_selection import train_test_split
import logging
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.multiprocessing as mp

from dataset import MovieRatingDataset


class SVDppModel(nn.Module):
    def __init__(self, n_users, n_items, n_factors, n_features=None):
        super(SVDppModel, self).__init__()
        self.n_factors = n_factors

        # 用户和物品嵌入
        self.user_embeddings = nn.Embedding(n_users, n_factors)
        self.item_embeddings = nn.Embedding(n_items, n_factors)
        self.user_implicit_embeddings = nn.Embedding(n_items, n_factors)

        # 偏置项
        self.user_biases = nn.Embedding(n_users, 1)
        self.item_biases = nn.Embedding(n_items, 1)

        # 全局偏置
        self.global_bias = nn.Parameter(torch.zeros(1))

        # 特征层
        if n_features is not None:
            self.feature_layer = nn.Sequential(
                nn.Linear(n_features, n_factors),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(n_factors, n_factors)
            )
        else:
            self.feature_layer = None

        # 初始化参数
        self._init_weights()

    def _init_weights(self):
        """使用He初始化"""
        nn.init.normal_(self.user_embeddings.weight, 0, np.sqrt(2.0 / self.n_factors))
        nn.init.normal_(self.item_embeddings.weight, 0, np.sqrt(2.0 / self.n_factors))
        nn.init.normal_(self.user_implicit_embeddings.weight, 0, np.sqrt(2.0 / self.n_factors))
        nn.init.zeros_(self.user_biases.weight)
        nn.init.zeros_(self.item_biases.weight)

    def forward(self, user_idx, item_idx, features=None):
        # 基础SVD++计算
        user_embed = self.user_embeddings(user_idx)
        item_embed = self.item_embeddings(item_idx)
        user_bias = self.user_biases(user_idx).squeeze()
        item_bias = self.item_biases(item_idx).squeeze()

        # 计算隐式反馈
        implicit_feedback = torch.mean(self.user_implicit_embeddings(item_idx), dim=0)

        # 基础预测
        pred = torch.sum(user_embed * item_embed, dim=1)
        pred = pred + user_bias + item_bias + self.global_bias

        # 如果有特征，添加特征交互
        if features is not None and self.feature_layer is not None:
            feature_vectors = self.feature_layer(features)
            pred = pred + torch.sum(feature_vectors * item_embed, dim=1)

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
        # 初始化参数
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate
        self.device = device
        self.num_workers = num_workers

        # 设置日志
        self._setup_logger()
        self.logger.info(f"Initializing SVD++ model with {n_factors} factors on {device}")

        # 模型相关属性
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.user_mapping = None
        self.item_mapping = None

        # 训练历史
        self.history = {
            'train_loss': [],
            'val_metrics': []
        }

        # 创建模型保存目录
        os.makedirs('model', exist_ok=True)
        self.logger.info("Model save directory created/verified")

        self.model_dir = 'model'
        os.makedirs(self.model_dir, exist_ok=True)
        self.logger.info(f"Model directory created/verified at {self.model_dir}")

    def _setup_logger(self):
        # 创建logger实例
        self.logger = logging.getLogger('GPUSVDpp')
        self.logger.setLevel(logging.INFO)

        # 创建日志目录
        log_dir = 'log'
        os.makedirs(log_dir, exist_ok=True)

        if not self.logger.handlers:
            # 1. 控制台处理器
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)

            # 2. 文件处理器
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = os.path.join(log_dir, f'training_{timestamp}.log')
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)

            # 日志格式
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            file_handler.setFormatter(formatter)

            # 添加处理器
            self.logger.addHandler(console_handler)
            self.logger.addHandler(file_handler)

    def fit(self, ratings_df, validation_size=0.1, features_df=None,
            patience=10, min_improvement=0.0001):
        """增强的训练函数，包含多指标早停"""
        self.logger.info(f"Training on {self.device}")

        # 创建用户和物品映射
        self.user_mapping = {user: idx for idx, user
                             in enumerate(ratings_df['userId'].unique())}
        self.item_mapping = {item: idx for idx, item
                             in enumerate(ratings_df['movieId'].unique())}

        # 划分训练和验证数据
        train_data, val_data = train_test_split(
            ratings_df, test_size=validation_size, random_state=42
        )

        # 创建数据集和数据加载器
        train_dataset = MovieRatingDataset(
            train_data, self.user_mapping, self.item_mapping, features_df
        )
        val_dataset = MovieRatingDataset(
            val_data, self.user_mapping, self.item_mapping, features_df
        )

        # 指定多进程上下文
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

        # 初始化模型
        n_features = features_df.shape[1] if features_df is not None else None
        self.model = SVDppModel(
            len(self.user_mapping),
            len(self.item_mapping),
            self.n_factors,
            n_features
        ).to(self.device)

        # 初始化优化器和损失函数
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        self.criterion = nn.MSELoss()

        # 初始化早停变量
        best_metrics = {
            'rmse': float('inf'),
            'mae': float('inf'),
            'ndcg': 0
        }
        patience_counter = 0
        best_epoch = 0

        # 训练循环
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

            # 检查是否有改进
            improved = False
            if (val_metrics['rmse'] < best_metrics['rmse'] - min_improvement or
                    val_metrics['mae'] < best_metrics['mae'] - min_improvement or
                    val_metrics['ndcg'] > best_metrics['ndcg'] + min_improvement):

                improved = True
                best_metrics = val_metrics.copy()
                best_epoch = epoch
                patience_counter = 0

                # 保存最佳模型
                self.save_model('best_model.pt')
                self.logger.info("New best model saved!")
            else:
                patience_counter += 1

            # 早停检查
            if patience_counter >= patience:
                self.logger.info(
                    f"Early stopping triggered after {epoch + 1} epochs. "
                    f"Best epoch was {best_epoch + 1} with metrics: "
                    f"RMSE = {best_metrics['rmse']:.4f}, "
                    f"MAE = {best_metrics['mae']:.4f}, "
                    f"NDCG = {best_metrics['ndcg']:.4f}"
                )
                break

        # 加载最佳模型
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

            features = None
            if 'features' in batch:
                features = batch['features'].to(self.device)

            predictions = self.model(user_idx, item_idx, features)
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

                features = None
                if 'features' in batch:
                    features = batch['features'].to(self.device)

                predictions = self.model(user_idx, item_idx, features)
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
            'rmse': np.sqrt(total_loss / total_batches),
            'mae': mean_absolute_error(all_ratings, all_predictions),
            'ndcg': ndcg_score(
                all_ratings.reshape(1, -1),
                all_predictions.reshape(1, -1)
            )
        }

        return metrics

    def get_model_path(self, filename):
        """
        Get full path for model file in the model directory
        """
        return os.path.join(self.model_dir, filename)

    def save_model(self, filename):
        """
        Save model state, mappings and training history to model directory

        Args:
            filename: Name of the model file (e.g., 'best_model.pt')
        """
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
        """
        Load saved model state and mappings from model directory

        Args:
            filename: Name of the model file (e.g., 'best_model.pt')
        """
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
    # 加载数据
    # ratings_df = pd.read_csv('./dataset/ratings_small.csv')
    ratings_df = pd.read_csv('./dataset/ratings.csv')

    # 初始化推荐器
    recommender = GPUSvdppRecommender(
        n_factors=100,
        n_epochs=1000,
        batch_size=4096,
        learning_rate=0.001,
        weight_decay=0.02,
        num_workers=2
    )

    # 训练模型
    recommender.fit(
        ratings_df,
        validation_size=0.1,
        patience=10,
        min_improvement=0.0001
    )

    # 绘制训练历史
    recommender.plot_training_history()

    # 评估最终模型
    print("\nBest Model Metrics:")
    for metric, value in recommender.history['val_metrics'][-1].items():
        print(f"{metric.upper()}: {value:.4f}")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()