import torch
from torch.utils.data import Dataset

class MovieRatingDataset(Dataset):
    def __init__(self, ratings_df, user_mapping, item_mapping, features_df=None):
        self.ratings = ratings_df
        self.user_mapping = user_mapping
        self.item_mapping = item_mapping
        self.features_df = features_df

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        row = self.ratings.iloc[idx]
        user_idx = self.user_mapping[row['userId']]
        item_idx = self.item_mapping[row['movieId']]
        rating = row['rating']

        if self.features_df is not None:
            features = self.features_df.iloc[idx].values
            return {
                'user_idx': torch.tensor(user_idx, dtype=torch.long),
                'item_idx': torch.tensor(item_idx, dtype=torch.long),
                'rating': torch.tensor(rating, dtype=torch.float),
                'features': torch.tensor(features, dtype=torch.float)
            }
        else:
            return {
                'user_idx': torch.tensor(user_idx, dtype=torch.long),
                'item_idx': torch.tensor(item_idx, dtype=torch.long),
                'rating': torch.tensor(rating, dtype=torch.float)
            }
