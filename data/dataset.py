import torch
from torch.utils.data import Dataset
import numpy as np


class StockDataset(Dataset):
    """PyTorch Dataset for stock data - returns dict for consistency"""

    def __init__(self, X, y, vix):
        """
        Args:
            X: numpy array [N, seq_len, features]
            y: numpy array [N, 1]
            vix: numpy array [N, 1]
        """
        self.X = X
        self.y = y
        self.vix = vix

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Return dict for compatibility with both trainers
        return {
            'X': torch.FloatTensor(self.X[idx]),
            'y': torch.FloatTensor(self.y[idx]),
            'vix': torch.FloatTensor(self.vix[idx])
        }
