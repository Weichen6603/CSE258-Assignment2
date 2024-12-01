# dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset

class MovieRatingDataset(Dataset):
    def __init__(self, ratings_df, user_mapping, item_mapping, categorical_features=None, numerical_features=None):
        self.ratings = ratings_df.reset_index(drop=True)
        self.user_mapping = user_mapping
        self.item_mapping = item_mapping
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features

        # Build categorical feature mappings
        if self.categorical_features is not None:
            self.categorical_mappings = {}
            for feature in self.categorical_features:
                unique_values = self.ratings[feature].unique()
                self.categorical_mappings[feature] = {val: idx for idx, val in enumerate(unique_values)}
                self.ratings[f'{feature}_idx'] = self.ratings[feature].map(self.categorical_mappings[feature])

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        row = self.ratings.iloc[idx]
        user_idx = self.user_mapping[row['userId']]
        item_idx = self.item_mapping[row['movieId']]
        rating = row['rating']

        # Numerical features
        numerical_features = []
        if self.numerical_features is not None:
            numerical_features = row[self.numerical_features].values.astype(np.float32)

        # Categorical features
        categorical_features = []
        if self.categorical_features is not None:
            for feature in self.categorical_features:
                categorical_features.append(row[f'{feature}_idx'])

        return {
            'user_idx': torch.tensor(user_idx, dtype=torch.long),
            'item_idx': torch.tensor(item_idx, dtype=torch.long),
            'rating': torch.tensor(rating, dtype=torch.float),
            'numerical_features': torch.tensor(numerical_features, dtype=torch.float),
            'categorical_features': torch.tensor(categorical_features, dtype=torch.long)
        }
