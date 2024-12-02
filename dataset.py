# dataset.py

import torch
from torch.utils.data import Dataset


class MovieRatingDataset(Dataset):
    def __init__(self, ratings_df, user_mapping, item_mapping, numerical_features=None):
        self.ratings = ratings_df.reset_index(drop=True)
        self.user_mapping = user_mapping
        self.item_mapping = item_mapping
        self.numerical_features = numerical_features or [
            'rating_mean',          # 用户平均评分
            'rating_count',         # 评分数量
            'weighted_rating',      # IMDB加权评分
            'revenue',              # 收入
            'budget',              # 预算
            'popularity',          # 热度
            'roi'                  # 投资回报率
        ]

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        """获取单个数据样本"""
        row = self.ratings.iloc[idx]

        # 获取用户和物品索引
        user_idx = self.user_mapping[row['userId']]
        item_idx = self.item_mapping[row['movieId']]
        rating = row['rating']

        # 获取数值特征
        numerical_features = []
        if self.numerical_features:
            try:
                numerical_features = [
                    float(row[feature]) if feature in row
                    else 0.0
                    for feature in self.numerical_features
                ]
            except Exception as e:
                print(f"处理特征时出错: {e}")
                numerical_features = [0.0] * len(self.numerical_features)

        # 转换为张量
        return {
            'user_idx': torch.tensor(user_idx, dtype=torch.long),
            'item_idx': torch.tensor(item_idx, dtype=torch.long),
            'rating': torch.tensor(rating, dtype=torch.float),
            'numerical_features': torch.tensor(numerical_features, dtype=torch.float)
        }

    def get_feature_dim(self):
        """返回特征维度"""
        return len(self.numerical_features)

    def get_features_info(self):
        """返回特征信息"""
        return {
            'numerical_features': self.numerical_features,
            'n_users': len(self.user_mapping),
            'n_items': len(self.item_mapping)
        }

class MovieDataCollator:
    """数据批处理收集器"""
    def __init__(self, pad_value=0):
        self.pad_value = pad_value

    def __call__(self, batch):
        """将批次数据整理成张量"""
        user_idx = torch.stack([item['user_idx'] for item in batch])
        item_idx = torch.stack([item['item_idx'] for item in batch])
        ratings = torch.stack([item['rating'] for item in batch])
        numerical_features = torch.stack([item['numerical_features'] for item in batch])

        return {
            'user_idx': user_idx,
            'item_idx': item_idx,
            'rating': ratings,
            'numerical_features': numerical_features
        }