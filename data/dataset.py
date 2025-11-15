"""
Dataset Module

This module defines PyTorch Dataset classes for stock price data.
Handles time series windowing, sequence generation, and batch preparation
for training and evaluation of time series forecasting models.
"""

import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))


class StockDataset(Dataset):
    """
    PyTorch Dataset for stock price time series data.

    Stores input sequences (X), target values (y), and VIX values as tensors.
    Each sample consists of a sequence of features for prediction.

    Args:
        X: Input sequences of shape [N, seq_len, n_features]
        y: Target values of shape [N, 1]
        vix: VIX values at prediction time of shape [N, 1]

    Example:
        >>> dataset = StockDataset(X_train, y_train, vix_train)
        >>> print(len(dataset))
        1000
        >>> sample = dataset[0]
        >>> print(sample['X'].shape, sample['y'].shape, sample['vix'].shape)
        torch.Size([20, 4]) torch.Size([1]) torch.Size([1])
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, vix: np.ndarray):
        """
        Initialize dataset with numpy arrays and convert to tensors.

        Args:
            X: Input sequences [N, seq_len, n_features]
            y: Target values [N, 1]
            vix: VIX values [N, 1]
        """
        # Convert to torch tensors (float32 for neural network compatibility)
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.vix = torch.FloatTensor(vix)

        # Validate shapes
        assert len(self.X) == len(self.y) == len(self.vix), \
            "X, y, and vix must have the same number of samples"

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns:
            Number of samples
        """
        return len(self.X)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset.

        Args:
            idx: Index of the sample to retrieve

        Returns:
            Dictionary containing:
            - 'X': Input sequence tensor [seq_len, n_features]
            - 'y': Target value tensor [1]
            - 'vix': VIX value tensor [1]
        """
        return {
            'X': self.X[idx],      # [seq_len, n_features]
            'y': self.y[idx],      # [1]
            'vix': self.vix[idx]   # [1]
        }


def get_dataloaders(
    train_data: Dict[str, np.ndarray],
    val_data: Dict[str, np.ndarray],
    test_data: Dict[str, np.ndarray],
    batch_size: int = 32,
    shuffle_train: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoader objects for train, validation, and test sets.

    Args:
        train_data: Dictionary with keys 'X', 'y', 'vix' for training data
        val_data: Dictionary with keys 'X', 'y', 'vix' for validation data
        test_data: Dictionary with keys 'X', 'y', 'vix' for test data
        batch_size: Batch size for DataLoaders (default: 32)
        shuffle_train: Whether to shuffle training data (default: True)

    Returns:
        Tuple of (train_loader, val_loader, test_loader)

    Example:
        >>> train_loader, val_loader, test_loader = get_dataloaders(
        ...     train_data, val_data, test_data, batch_size=32
        ... )
        >>> for batch in train_loader:
        ...     X, y, vix = batch['X'], batch['y'], batch['vix']
        ...     print(X.shape)
        ...     break
        torch.Size([32, 20, 4])
    """
    # Create datasets
    train_dataset = StockDataset(
        X=train_data['X'],
        y=train_data['y'],
        vix=train_data['vix']
    )

    val_dataset = StockDataset(
        X=val_data['X'],
        y=val_data['y'],
        vix=val_data['vix']
    )

    test_dataset = StockDataset(
        X=test_data['X'],
        y=test_data['y'],
        vix=test_data['vix']
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        drop_last=False  # Keep all samples
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle validation
        drop_last=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle test
        drop_last=False
    )

    print(f"Created DataLoaders:")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val  : {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"  Test : {len(test_dataset)} samples, {len(test_loader)} batches")

    return train_loader, val_loader, test_loader


def load_processed_data(
    ticker: str,
    data_dir: str = 'data/processed'
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Load processed stock data from .npz file.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL')
        data_dir: Directory containing processed .npz files

    Returns:
        Dictionary with structure:
        {
            'train': {'X': np.ndarray, 'y': np.ndarray, 'vix': np.ndarray},
            'val': {...},
            'test': {...}
        }

    Raises:
        FileNotFoundError: If the .npz file doesn't exist

    Example:
        >>> data = load_processed_data('AAPL')
        >>> print(data['train']['X'].shape)
        (800, 20, 4)
    """
    data_path = Path(data_dir) / f"{ticker}_processed.npz"

    if not data_path.exists():
        raise FileNotFoundError(f"Processed data file not found: {data_path}")

    # Load .npz file
    print(f"Loading data from {data_path}")
    loaded = np.load(data_path)

    # Extract arrays into structured dictionary
    data = {
        'train': {
            'X': loaded['X_train'],
            'y': loaded['y_train'],
            'vix': loaded['vix_train']
        },
        'val': {
            'X': loaded['X_val'],
            'y': loaded['y_val'],
            'vix': loaded['vix_val']
        },
        'test': {
            'X': loaded['X_test'],
            'y': loaded['y_test'],
            'vix': loaded['vix_test']
        }
    }

    # Print loaded data info
    print(f"Loaded data for {ticker}:")
    print(f"  Train: X={data['train']['X'].shape}, y={data['train']['y'].shape}, vix={data['train']['vix'].shape}")
    print(f"  Val  : X={data['val']['X'].shape}, y={data['val']['y'].shape}, vix={data['val']['vix'].shape}")
    print(f"  Test : X={data['test']['X'].shape}, y={data['test']['y'].shape}, vix={data['test']['vix'].shape}")

    return data


def main() -> None:
    """
    Main function for testing the dataset and dataloader functionality.

    Loads one ticker's data, creates datasets and dataloaders, and prints
    information about shapes and sample batches.
    """
    print("\n" + "="*80)
    print("Testing StockDataset and DataLoader")
    print("="*80 + "\n")

    # Load data for one ticker (AAPL as example)
    ticker = 'AAPL'
    try:
        data = load_processed_data(ticker, data_dir='data/processed')
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run data/preprocessor.py first to generate processed data.")
        return

    # Create dataloaders
    print(f"\nCreating dataloaders with batch_size=32...")
    train_loader, val_loader, test_loader = get_dataloaders(
        train_data=data['train'],
        val_data=data['val'],
        test_data=data['test'],
        batch_size=32,
        shuffle_train=True
    )

    # Test loading a batch
    print("\n" + "="*80)
    print("Sample Batch from Training Data")
    print("="*80)

    for batch in train_loader:
        X_batch = batch['X']
        y_batch = batch['y']
        vix_batch = batch['vix']

        print(f"\nBatch shapes:")
        print(f"  X (inputs)  : {X_batch.shape}  # [batch_size, seq_len, n_features]")
        print(f"  y (targets) : {y_batch.shape}  # [batch_size, 1]")
        print(f"  vix (VIX)   : {vix_batch.shape}  # [batch_size, 1]")

        print(f"\nBatch statistics:")
        print(f"  X - min: {X_batch.min():.4f}, max: {X_batch.max():.4f}, mean: {X_batch.mean():.4f}")
        print(f"  y - min: {y_batch.min():.4f}, max: {y_batch.max():.4f}, mean: {y_batch.mean():.4f}")
        print(f"  vix - min: {vix_batch.min():.4f}, max: {vix_batch.max():.4f}, mean: {vix_batch.mean():.4f}")

        print(f"\nData types:")
        print(f"  X: {X_batch.dtype}")
        print(f"  y: {y_batch.dtype}")
        print(f"  vix: {vix_batch.dtype}")

        # Only show first batch
        break

    print("\n" + "="*80)
    print("Dataset and DataLoader test complete!")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
