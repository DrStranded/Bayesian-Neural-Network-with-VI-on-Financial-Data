"""
Data Downloader Module

This module handles downloading stock price data from various sources (primarily Yahoo Finance).
It provides functionality to fetch historical stock data, handle API requests, and manage
data caching to minimize redundant downloads.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.helpers import create_dir_if_not_exists


def download_stock_data(
    ticker: str,
    start_date: str,
    end_date: str
) -> Optional[pd.DataFrame]:
    """
    Download OHLCV (Open, High, Low, Close, Volume) stock data using yfinance.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'SPY')
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format

    Returns:
        DataFrame with Date index and OHLCV columns, or None if download fails

    Raises:
        ValueError: If date format is invalid

    Example:
        >>> df = download_stock_data('AAPL', '2020-01-01', '2021-01-01')
        >>> print(df.head())
    """
    try:
        # Download data using yfinance
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)

        if df.empty:
            print(f"Warning: No data returned for {ticker}")
            return None

        # Reset index to make Date a column, then set it back
        # This ensures proper Date index format
        df.index.name = 'Date'

        # Keep only OHLCV columns (remove Dividends, Stock Splits if present)
        standard_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        available_columns = [col for col in standard_columns if col in df.columns]
        df = df[available_columns]

        print(f"Successfully downloaded {len(df)} rows for {ticker}")
        return df

    except Exception as e:
        print(f"Error downloading data for {ticker}: {str(e)}")
        return None


def download_vix(start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """
    Download VIX (Volatility Index) data from Yahoo Finance.

    The VIX is a measure of market volatility and is often called the "fear index".
    This function downloads VIX data and returns only the Close column.

    Args:
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format

    Returns:
        DataFrame with Date index and VIX Close column, or None if download fails

    Example:
        >>> vix_df = download_vix('2020-01-01', '2021-01-01')
        >>> print(vix_df.head())
    """
    try:
        # VIX ticker symbol on Yahoo Finance
        vix_ticker = '^VIX'

        # Download VIX data
        vix = yf.Ticker(vix_ticker)
        df = vix.history(start=start_date, end=end_date)

        if df.empty:
            print(f"Warning: No VIX data returned")
            return None

        # Keep only Close column and rename it to VIX
        df = df[['Close']].copy()
        df.columns = ['VIX']
        df.index.name = 'Date'

        # Handle missing data by forward filling
        if df.isnull().any().any():
            print(f"Warning: VIX data contains missing values, forward filling...")
            df = df.fillna(method='ffill')

            # If still have NaN at the beginning, backward fill
            if df.isnull().any().any():
                df = df.fillna(method='bfill')

        print(f"Successfully downloaded {len(df)} rows for VIX")
        return df

    except Exception as e:
        print(f"Error downloading VIX data: {str(e)}")
        return None


def download_all_data(
    tickers: List[str],
    start_date: str,
    end_date: str,
    save_dir: str = 'data/raw'
) -> Dict[str, pd.DataFrame]:
    """
    Download stock data for multiple tickers plus VIX index and save to CSV files.

    Args:
        tickers: List of stock ticker symbols to download
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        save_dir: Directory to save CSV files (default: 'data/raw')

    Returns:
        Dictionary mapping ticker symbols to their DataFrames
        {ticker: DataFrame, 'VIX': DataFrame}

    Example:
        >>> data = download_all_data(['AAPL', 'SPY'], '2020-01-01', '2021-01-01')
        >>> print(f"Downloaded {len(data)} datasets")
    """
    # Create save directory if it doesn't exist
    save_path = Path(save_dir)
    create_dir_if_not_exists(save_path)

    data_dict = {}

    print(f"\n{'='*80}")
    print(f"Downloading data from {start_date} to {end_date}")
    print(f"Save directory: {save_path}")
    print(f"{'='*80}\n")

    # Download stock data for each ticker with progress bar
    print("Downloading stock data...")
    for ticker in tqdm(tickers, desc="Stocks"):
        df = download_stock_data(ticker, start_date, end_date)

        if df is not None:
            data_dict[ticker] = df

            # Save to CSV
            csv_path = save_path / f"{ticker}.csv"
            df.to_csv(csv_path)
            print(f"  Saved {ticker} to {csv_path}")
        else:
            print(f"  Skipping {ticker} due to download failure")

    # Download VIX data
    print("\nDownloading VIX data...")
    vix_df = download_vix(start_date, end_date)

    if vix_df is not None:
        data_dict['VIX'] = vix_df

        # Save VIX to CSV
        csv_path = save_path / "VIX.csv"
        vix_df.to_csv(csv_path)
        print(f"  Saved VIX to {csv_path}")
    else:
        print(f"  Warning: VIX data not available")

    print(f"\n{'='*80}")
    print(f"Download complete! Successfully downloaded {len(data_dict)} datasets")
    print(f"{'='*80}\n")

    return data_dict


def main() -> None:
    """
    Main function to download all stock data using configuration parameters.

    This function imports the Config class and downloads data for all configured
    tickers plus VIX index, saving them to the raw data directory.
    """
    # Import config
    from training.config import Config

    config = Config()

    print("Using configuration:")
    print(f"  Tickers: {config.TICKERS}")
    print(f"  Date Range: {config.START_DATE} to {config.END_DATE}")
    print(f"  Train End: {config.TRAIN_END}")
    print(f"  Val End: {config.VAL_END}")

    # Download all data
    data = download_all_data(
        tickers=config.TICKERS,
        start_date=config.START_DATE,
        end_date=config.END_DATE,
        save_dir='data/raw'
    )

    # Print summary
    print("\nData Summary:")
    for ticker, df in data.items():
        print(f"  {ticker:6s}: {len(df):5d} rows, "
              f"Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")


if __name__ == '__main__':
    main()
