"""
Data Preprocessor Module

This module contains functions for preprocessing stock price data including:
- Normalization and scaling
- Feature engineering (technical indicators, rolling statistics)
- Handling missing values
- Train/validation/test split generation
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.helpers import create_dir_if_not_exists


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical features from OHLCV data.

    Features computed:
    - normalized_price: Price normalized to first value (Close / Close.iloc[0])
    - log_return: Log returns (np.log(Close / Close.shift(1)))
    - normalized_volume: Volume normalized by 20-day rolling mean
    - volatility: 20-day rolling standard deviation of log returns

    Args:
        df: DataFrame with OHLCV columns (must have 'Close' and 'Volume')

    Returns:
        DataFrame with added feature columns, first 20 rows dropped (NaN from rolling)

    Example:
        >>> df = compute_features(stock_df)
        >>> print(df.columns)
        Index(['Open', 'High', 'Low', 'Close', 'Volume', 'normalized_price',
               'log_return', 'normalized_volume', 'volatility'], dtype='object')
    """
    # Create a copy to avoid modifying original
    df = df.copy()

    # 1. Normalized price (price relative to first value)
    df['normalized_price'] = df['Close'] / df['Close'].iloc[0]

    # 2. Log returns
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))

    # 3. Normalized volume (volume / 20-day rolling mean)
    rolling_volume_mean = df['Volume'].rolling(window=20).mean()
    df['normalized_volume'] = df['Volume'] / rolling_volume_mean

    # 4. Volatility (20-day rolling std of log returns)
    df['volatility'] = df['log_return'].rolling(window=20).std()

    # Drop first 20 rows that contain NaN values from rolling operations
    df = df.iloc[20:].copy()

    # Reset index to ensure clean indexing
    df = df.reset_index(drop=False)

    print(f"  Computed features: {len(df)} rows after dropping first 20")

    return df


def merge_with_vix(stock_df, vix_df):
    """Merge stock data with VIX"""
    stock_df = stock_df.copy()
    vix_df = vix_df.copy()
    
    # Ensure stock_df has Date column
    if 'Date' not in stock_df.columns:
        stock_df = stock_df.reset_index()
    
    # Convert to timezone-naive (handles tz-aware objects)
    stock_df['Date'] = pd.to_datetime(stock_df['Date'], utc=True).dt.tz_localize(None)
    
    # Prepare VIX
    vix_merge = vix_df.copy()
    if vix_merge.index.name == 'Date':
        vix_merge = vix_merge.reset_index()
    vix_merge['Date'] = pd.to_datetime(vix_merge['Date'], utc=True).dt.tz_localize(None)
    
    # Merge
    merged = pd.merge(stock_df, vix_merge[['Date', 'VIX']], on='Date', how='left')
    
    # Fill missing VIX
    merged['VIX'] = merged['VIX'].ffill().bfill()
    
    print(f"  Merged with VIX: {len(merged)} rows, {merged['VIX'].isna().sum()} missing")
    
    return merged


def train_val_test_split(
    df: pd.DataFrame,
    train_end: str = '2021-12-31',
    val_end: str = '2022-12-31'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train, validation, and test sets based on dates.

    Args:
        df: DataFrame with 'Date' column
        train_end: End date for training set (inclusive)
        val_end: End date for validation set (inclusive)

    Returns:
        Tuple of (train_df, val_df, test_df)

    Example:
        >>> train, val, test = train_val_test_split(df, '2021-12-31', '2022-12-31')
        >>> print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    """
    # Ensure Date is datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Split by date
    train_df = df[df['Date'] <= train_end].copy()
    val_df = df[(df['Date'] > train_end) & (df['Date'] <= val_end)].copy()
    test_df = df[df['Date'] > val_end].copy()

    # Print split sizes
    print(f"\n  Data split:")
    print(f"    Train: {len(train_df)} rows ({train_df['Date'].min().strftime('%Y-%m-%d')} to {train_df['Date'].max().strftime('%Y-%m-%d')})")
    print(f"    Val  : {len(val_df)} rows ({val_df['Date'].min().strftime('%Y-%m-%d')} to {val_df['Date'].max().strftime('%Y-%m-%d')})")
    print(f"    Test : {len(test_df)} rows ({test_df['Date'].min().strftime('%Y-%m-%d')} to {test_df['Date'].max().strftime('%Y-%m-%d')})")

    return train_df, val_df, test_df


def create_sequences(
    df: pd.DataFrame,
    seq_len: int = 20,
    feature_cols: List[str] = ['normalized_price', 'log_return', 'normalized_volume', 'volatility'],
    target_col: str = 'log_return'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create sliding window sequences for time series prediction.

    Args:
        df: DataFrame with feature columns
        seq_len: Length of input sequences (lookback window)
        feature_cols: List of feature column names to use as input
        target_col: Column name to use as prediction target

    Returns:
        Tuple of (X, y, vix):
        - X: numpy array of shape [N, seq_len, n_features] - input sequences
        - y: numpy array of shape [N, 1] - next day's target value
        - vix: numpy array of shape [N, 1] - VIX value at prediction time

    Example:
        >>> X, y, vix = create_sequences(df, seq_len=20)
        >>> print(f"X shape: {X.shape}, y shape: {y.shape}")
        X shape: (1000, 20, 4), y shape: (1000, 1)
    """
    # Extract feature values
    feature_data = df[feature_cols].values
    target_data = df[target_col].values
    vix_data = df['VIX'].values

    X_list = []
    y_list = []
    vix_list = []

    # Create sequences
    # We need seq_len past values to predict the next value
    for i in range(len(df) - seq_len):
        # Input: past seq_len timesteps
        X_list.append(feature_data[i:i + seq_len])

        # Target: next timestep's target value
        y_list.append(target_data[i + seq_len])

        # VIX: VIX value at prediction time (i + seq_len)
        vix_list.append(vix_data[i + seq_len])

    # Convert to numpy arrays
    X = np.array(X_list)  # Shape: [N, seq_len, n_features]
    y = np.array(y_list).reshape(-1, 1)  # Shape: [N, 1]
    vix = np.array(vix_list).reshape(-1, 1)  # Shape: [N, 1]

    print(f"  Created sequences: X shape {X.shape}, y shape {y.shape}, vix shape {vix.shape}")

    return X, y, vix


def process_stock_data(
    ticker: str,
    raw_data_dir: str = 'data/raw',
    train_end: str = '2021-12-31',
    val_end: str = '2022-12-31',
    seq_len: int = 20
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Complete preprocessing pipeline for a single stock.

    Steps:
    1. Load stock CSV and VIX CSV
    2. Compute technical features
    3. Merge with VIX data
    4. Split into train/val/test
    5. Create sequences for each split

    Args:
        ticker: Stock ticker symbol
        raw_data_dir: Directory containing raw CSV files
        train_end: End date for training set
        val_end: End date for validation set
        seq_len: Sequence length for sliding windows

    Returns:
        Dictionary with structure:
        {
            'train': {'X': np.ndarray, 'y': np.ndarray, 'vix': np.ndarray},
            'val': {...},
            'test': {...}
        }

    Example:
        >>> data = process_stock_data('AAPL')
        >>> print(data['train']['X'].shape)
        (800, 20, 4)
    """
    print(f"\n{'='*80}")
    print(f"Processing {ticker}")
    print(f"{'='*80}")

    # Load data
    raw_path = Path(raw_data_dir)
    stock_file = raw_path / f"{ticker}.csv"
    vix_file = raw_path / "VIX.csv"

    if not stock_file.exists():
        raise FileNotFoundError(f"Stock data file not found: {stock_file}")
    if not vix_file.exists():
        raise FileNotFoundError(f"VIX data file not found: {vix_file}")

    print(f"  Loading data from {raw_data_dir}")
    stock_df = pd.read_csv(stock_file, index_col='Date', parse_dates=True)
    vix_df = pd.read_csv(vix_file, index_col='Date', parse_dates=True)

    # Remove timezone info if present to avoid merge conflicts
    if hasattr(stock_df.index, 'tz') and stock_df.index.tz is not None:
        stock_df.index = stock_df.index.tz_localize(None)
    if hasattr(vix_df.index, 'tz') and vix_df.index.tz is not None:
        vix_df.index = vix_df.index.tz_localize(None)

    # Compute features
    print(f"  Computing features...")
    stock_df = compute_features(stock_df)

    # Merge with VIX
    print(f"  Merging with VIX...")
    stock_df = merge_with_vix(stock_df, vix_df)

    # Split data
    train_df, val_df, test_df = train_val_test_split(stock_df, train_end, val_end)

    # Create sequences for each split
    feature_cols = ['normalized_price', 'log_return', 'normalized_volume', 'volatility']

    print(f"\n  Creating sequences for train set...")
    X_train, y_train, vix_train = create_sequences(train_df, seq_len, feature_cols)

    print(f"  Creating sequences for validation set...")
    X_val, y_val, vix_val = create_sequences(val_df, seq_len, feature_cols)

    print(f"  Creating sequences for test set...")
    X_test, y_test, vix_test = create_sequences(test_df, seq_len, feature_cols)

    # Return processed data
    processed_data = {
        'train': {'X': X_train, 'y': y_train, 'vix': vix_train},
        'val': {'X': X_val, 'y': y_val, 'vix': vix_val},
        'test': {'X': X_test, 'y': y_test, 'vix': vix_test}
    }

    print(f"\n  Processing complete for {ticker}!")

    return processed_data


def main() -> None:
    """
    Main function to process all tickers and save processed data.

    Loads configuration, processes each ticker, and saves results as .npz files
    in the processed data directory.
    """
    # Import config
    from training.config import Config

    config = Config()

    print("\n" + "="*80)
    print("Data Preprocessing Pipeline")
    print("="*80)
    print(f"Tickers: {config.TICKERS}")
    print(f"Sequence Length: {config.SEQ_LEN}")
    print(f"Train End: {config.TRAIN_END}")
    print(f"Val End: {config.VAL_END}")

    # Create output directory
    output_dir = Path(config.DATA_DIR)
    create_dir_if_not_exists(output_dir)

    # Process each ticker
    all_stats = {}

    for ticker in config.TICKERS:
        try:
            # Process stock data
            processed_data = process_stock_data(
                ticker=ticker,
                raw_data_dir='data/raw',
                train_end=config.TRAIN_END,
                val_end=config.VAL_END,
                seq_len=config.SEQ_LEN
            )

            # Save processed data
            output_file = output_dir / f"{ticker}_processed.npz"
            np.savez(
                output_file,
                X_train=processed_data['train']['X'],
                y_train=processed_data['train']['y'],
                vix_train=processed_data['train']['vix'],
                X_val=processed_data['val']['X'],
                y_val=processed_data['val']['y'],
                vix_val=processed_data['val']['vix'],
                X_test=processed_data['test']['X'],
                y_test=processed_data['test']['y'],
                vix_test=processed_data['test']['vix']
            )
            print(f"  Saved to {output_file}")

            # Collect statistics
            all_stats[ticker] = {
                'train_samples': len(processed_data['train']['X']),
                'val_samples': len(processed_data['val']['X']),
                'test_samples': len(processed_data['test']['X'])
            }

        except Exception as e:
            print(f"\nError processing {ticker}: {str(e)}")
            continue

    # Print summary
    print("\n" + "="*80)
    print("Processing Summary")
    print("="*80)
    for ticker, stats in all_stats.items():
        print(f"\n{ticker}:")
        print(f"  Train samples: {stats['train_samples']}")
        print(f"  Val samples  : {stats['val_samples']}")
        print(f"  Test samples : {stats['test_samples']}")
        print(f"  Total        : {sum(stats.values())}")

    print("\n" + "="*80)
    print("All data processed successfully!")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
