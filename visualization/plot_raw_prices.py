"""
Plot raw price history for AAPL, SPY and TSLA on a single figure.

Reads:  data/raw/{AAPL,SPY,TSLA}.csv  (Yahoo Finance format)
Saves:  visualization/raw/raw_prices_three_stocks.png
"""

from pathlib import Path
from typing import List

import pandas as pd
import matplotlib.pyplot as plt


def load_price_series(ticker: str, data_dir: str = "data/raw") -> pd.Series:
    """
    Load a single ticker's price series from CSV and return a pandas Series.

    The function prefers the 'Adj Close' column; if not available, it falls
    back to 'Close'. The series index is the date, and the name is the ticker.
    """
    csv_path = Path(data_dir) / f"{ticker}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found for {ticker}: {csv_path}")

    # Most Yahoo Finance CSVs have a 'Date' column
    df = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")

    price_col = None
    for col in ["Adj Close", "Close"]:
        if col in df.columns:
            price_col = col
            break

    if price_col is None:
        raise ValueError(
            f"No 'Adj Close' or 'Close' column found in {csv_path}. "
            f"Available columns: {list(df.columns)}"
        )

    series = df[price_col].rename(ticker)
    return series


def plot_raw_prices(
    tickers: List[str],
    data_dir: str = "data/raw",
    save_dir: str = "visualization/raw",
    filename: str = "raw_prices_three_stocks.png",
) -> None:
    """
    Load raw prices for the given tickers and plot them on a single figure.

    The figure is formatted with small margins to use space efficiently.
    """
    # Load all price series and align on the date index
    series_list = []
    for ticker in tickers:
        s = load_price_series(ticker, data_dir=data_dir)
        series_list.append(s)

    prices = pd.concat(series_list, axis=1).dropna(how="all")
    prices = prices.sort_index()

    # Create output directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 5))

    for ticker in tickers:
        ax.plot(
            prices.index,
            prices[ticker],
            label=ticker,
            linewidth=1.3,
        )

    ax.set_title("Raw Price History: AAPL, SPY, TSLA")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")

    ax.legend(loc="upper left", frameon=False)
    ax.grid(True, alpha=0.2, linewidth=0.5)

    # Use space efficiently: small margins and tight x-axis
    ax.margins(x=0)
    fig.subplots_adjust(left=0.06, right=0.995, top=0.95, bottom=0.12)

    out_file = save_path / filename
    fig.savefig(out_file, dpi=200)
    plt.close(fig)

    print(f"Saved raw price figure to: {out_file}")


def main() -> None:
    tickers = ["AAPL", "SPY", "TSLA"]
    plot_raw_prices(tickers)


if __name__ == "__main__":
    main()
