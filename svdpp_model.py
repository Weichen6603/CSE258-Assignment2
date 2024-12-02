# svdpp_model.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import logging
from datetime import datetime
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.multiprocessing as mp
from sklearn.metrics import mean_squared_error, mean_absolute_error, ndcg_score
from sklearn.model_selection import train_test_split


class SVDppModel(nn.Module):
    def __init__(self, n_users, n_items, n_factors, n_numerical_features):
        super(SVDppModel, self).__init__()
        self.n_factors = n_factors

        # 用户和物品嵌入
        self.user_embeddings = nn.Embedding(n_users, n_factors)
        self.item_embeddings = nn.Embedding(n_items, n_factors)
        self.user_implicit_embeddings = nn.Embedding(n_items, n_factors)

        # 偏置项
        self.user_biases = nn.Embedding(n_users, 1)
        self.item_biases = nn.Embedding(n_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

        # 数值特征处理层
        self.numerical_layer = nn.Sequential(
            nn.Linear(n_numerical_features, n_factors),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(n_factors, n_factors)
        )

        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(n_factors * 2, n_factors),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(n_factors, 1)
        )

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        nn.init.normal_(self.user_embeddings.weight, 0, 0.1)
        nn.init.normal_(self.item_embeddings.weight, 0, 0.1)
        nn.init.normal_(self.user_implicit_embeddings.weight, 0, 0.1)
        nn.init.zeros_(self.user_biases.weight)
        nn.init.zeros_(self.item_biases.weight)

        for layer in self.numerical_layer:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        for layer in self.output_layer:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, user_idx, item_idx, numerical_features=None):
        # 基础SVD++计算
        user_embed = self.user_embeddings(user_idx)
        item_embed = self.item_embeddings(item_idx)
        user_bias = self.user_biases(user_idx).squeeze()
        item_bias = self.item_biases(item_idx).squeeze()

        # 隐式反馈
        implicit_feedback = torch.mean(self.user_implicit_embeddings(item_idx), dim=0)
        if implicit_feedback.dim() == 1:
            implicit_feedback = implicit_feedback.unsqueeze(0)

        # 数值特征处理
        if numerical_features is not None:
            numerical_output = self.numerical_layer(numerical_features)
            combined_features = torch.cat([
                user_embed * item_embed + implicit_feedback,
                numerical_output
            ], dim=1)
        else:
            combined_features = user_embed * item_embed + implicit_feedback

        # 最终预测
        pred = self.output_layer(combined_features).squeeze()
        pred = pred + user_bias + item_bias + self.global_bias

        return pred


class GPUSvdppRecommender:
    def __init__(self,
                 n_factors=100,
                 n_epochs=20,
                 batch_size=1024,
                 learning_rate=0.001,
                 weight_decay=0.02,
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 num_workers=4):

        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device = device
        self.num_workers = num_workers

        # 设置日志
        self._setup_logger()
        self.logger.info(f"初始化SVD++模型: {n_factors}因子, 设备: {device}")

        # 模型相关属性
        self.model = None
        self.optimizer = None
        self.criterion = nn.MSELoss()
        self.user_mapping = None
        self.item_mapping = None

        # 训练历史
        self.history = {
            'train_loss': [],
            'val_metrics': []
        }

        # 创建模型保存目录
        self.model_dir = 'model'
        os.makedirs(self.model_dir, exist_ok=True)

    def _setup_logger(self):
        """设置日志，使用UTF-8编码"""
        self.logger = logging.getLogger('SVDpp')
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            # 控制台处理器
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)

            # 文件处理器
            log_dir = 'log'
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            # 使用utf-8编码
            file_handler = logging.FileHandler(
                os.path.join(log_dir, f'training_{timestamp}.log'),
                encoding='utf-8'
            )
            file_handler.setLevel(logging.INFO)

            # 日志格式
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            file_handler.setFormatter(formatter)

            self.logger.addHandler(console_handler)
            self.logger.addHandler(file_handler)

    def fit(self, ratings_df, validation_size=0.1, patience=5):
        """训练模型"""
        self.logger.info(f"开始训练，使用设备: {self.device}")

        # 创建用户和物品映射
        self.user_mapping = {user: idx for idx, user
                             in enumerate(ratings_df['userId'].unique())}
        self.item_mapping = {item: idx for idx, item
                             in enumerate(ratings_df['movieId'].unique())}

        # 准备数据
        ratings_df['user_idx'] = ratings_df['userId'].map(self.user_mapping)
        ratings_df['item_idx'] = ratings_df['movieId'].map(self.item_mapping)

        # 数值特征列表
        numerical_features = [
            'rating_mean',  # 用户平均评分
            'rating_count',  # 评分数量
            'weighted_rating',  # IMDB加权评分
            'revenue',  # 收入
            'budget',  # 预算
            'popularity',  # 热度
            'roi'  # 投资回报率
        ]

        # 分割训练集和验证集
        train_data, val_data = train_test_split(
            ratings_df, test_size=validation_size, random_state=42
        )

        # 创建数据加载器
        from dataset import MovieRatingDataset
        train_dataset = MovieRatingDataset(
            train_data,
            self.user_mapping,
            self.item_mapping,
            numerical_features=numerical_features
        )
        val_dataset = MovieRatingDataset(
            val_data,
            self.user_mapping,
            self.item_mapping,
            numerical_features=numerical_features
        )

        # 设置多进程上下文
        mp_context = mp.get_context('spawn')

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            multiprocessing_context=mp_context
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            multiprocessing_context=mp_context
        )

        # 初始化模型
        self.model = SVDppModel(
            n_users=len(self.user_mapping),
            n_items=len(self.item_mapping),
            n_factors=self.n_factors,
            n_numerical_features=len(numerical_features)
        ).to(self.device)

        # 初始化优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        # 训练循环
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.n_epochs):
            train_loss = self._train_epoch(train_loader)
            val_metrics = self._validate(val_loader)

            self.history['train_loss'].append(train_loss)
            self.history['val_metrics'].append(val_metrics)

            self.logger.info(
                f'Epoch {epoch + 1}/{self.n_epochs}: '
                f'Train Loss = {train_loss:.4f}, '
                f'Val RMSE = {val_metrics["rmse"]:.4f}'
            )

            # 早停检查
            if val_metrics['rmse'] < best_val_loss:
                best_val_loss = val_metrics['rmse']
                patience_counter = 0
                self.save_model('best_model.pt')
            else:
                patience_counter += 1

            if patience_counter >= patience:
                self.logger.info(f"早停触发，在epoch {epoch + 1}")
                break

        # 加载最佳模型
        self.load_model('best_model.pt')
        return self

    def _train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0

        with tqdm(train_loader, desc='Training') as pbar:
            for batch in pbar:
                self.optimizer.zero_grad()

                user_idx = batch['user_idx'].to(self.device)
                item_idx = batch['item_idx'].to(self.device)
                ratings = batch['rating'].to(self.device)
                numerical_features = batch['numerical_features'].to(self.device)

                predictions = self.model(user_idx, item_idx, numerical_features)
                loss = self.criterion(predictions, ratings)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix({'batch_loss': loss.item()})

        return total_loss / len(train_loader)

    def _validate(self, val_loader):
        """验证模型"""
        self.model.eval()
        predictions = []
        actuals = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validating'):
                user_idx = batch['user_idx'].to(self.device)
                item_idx = batch['item_idx'].to(self.device)
                ratings = batch['rating']
                numerical_features = batch['numerical_features'].to(self.device)

                pred = self.model(user_idx, item_idx, numerical_features)
                predictions.extend(pred.cpu().numpy())
                actuals.extend(ratings.numpy())

        predictions = np.array(predictions)
        actuals = np.array(actuals)

        return {
            'rmse': np.sqrt(mean_squared_error(actuals, predictions)),
            'mae': mean_absolute_error(actuals, predictions),
            'ndcg': ndcg_score(
                actuals.reshape(1, -1),
                predictions.reshape(1, -1)
            )
        }

    def save_model(self, filename):
        """保存模型"""
        path = os.path.join(self.model_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'user_mapping': self.user_mapping,
            'item_mapping': self.item_mapping,
            'history': self.history
        }, path)
        self.logger.info(f"模型保存至 {path}")

    def load_model(self, filename):
        """加载模型"""
        path = os.path.join(self.model_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"模型文件不存在: {path}")

        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.user_mapping = checkpoint['user_mapping']
        self.item_mapping = checkpoint['item_mapping']
        self.history = checkpoint['history']
        self.logger.info(f"模型加载自 {path}")

    def plot_training_history(self):
        """绘制训练历史"""
        plt.figure(figsize=(15, 5))

        # 损失曲线
        plt.subplot(131)
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot([m['rmse'] for m in self.history['val_metrics']], label='Val RMSE')
        plt.title('Loss vs. Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # MAE曲线
        plt.subplot(132)
        plt.plot([m['mae'] for m in self.history['val_metrics']])
        plt.title('MAE vs. Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')

        # NDCG曲线
        plt.subplot(133)
        plt.plot([m['ndcg'] for m in self.history['val_metrics']])
        plt.title('NDCG vs. Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('NDCG')

        plt.tight_layout()
        plt.savefig(os.path.join('log', f'training_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'))
        plt.close()