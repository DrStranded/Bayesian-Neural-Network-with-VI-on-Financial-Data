"""
Visualization for the Two-Stage LSTM + Bayesian Volatility Model.

For each ticker we generate four types of plots:

  1) Prediction plot with 50% and 95% intervals over the full test set
  2) The same prediction plot, zoomed into the first 200 test days
  3) Volatility-only view: predicted sigma_t vs realized volatility and VIX
  4) Price vs predicted volatility: normalized price on the left axis and
     sigma_t on the right axis

Reads:
    results/predictions/all_two_stage.json

Writes:
    results/figures_two_stage/*.png
"""

import sys
from pathlib import Path

# Make project root importable if needed later
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------


def compute_coverage(y_true, y_pred, sigma, z: float) -> float:
    """
    Compute empirical coverage of an interval of the form:

        [mu_t - z * sigma_t, mu_t + z * sigma_t]

    Parameters
    ----------
    y_true : array-like
        True targets.
    y_pred : array-like
        Predicted means mu_t.
    sigma : array-like
        Predicted standard deviations sigma_t.
    z : float
        Z-score (e.g., 0.674 for 50% CI, 1.96 for 95% CI).

    Returns
    -------
    coverage : float
        Fraction of points whose |y_true - y_pred| <= z * sigma.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    sigma = np.asarray(sigma)

    abs_err = np.abs(y_true - y_pred)
    inside = abs_err <= z * sigma
    return float(np.mean(inside))


def plot_prediction_intervals(
    dates,
    y_true,
    y_pred,
    sigma,
    title: str,
    save_path: Path,
):
    """
    Plot prediction intervals for the whole test set.

    Elements:
      - black line: true log returns
      - blue line: predicted mean mu_t
      - dark band: 50% predictive interval (mu_t ± 0.674 * sigma_t)
      - light band: 95% predictive interval (mu_t ± 1.96 * sigma_t)

    This figure is used to visually show that
    intervals widen in volatile periods and contain most observations.
    """
    dates = np.asarray(dates)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    sigma = np.asarray(sigma)

    z50 = 0.674  # ~50% CI
    z95 = 1.96   # ~95% CI

    lower_50 = y_pred - z50 * sigma
    upper_50 = y_pred + z50 * sigma

    lower_95 = y_pred - z95 * sigma
    upper_95 = y_pred + z95 * sigma

    plt.figure(figsize=(12, 4))

    # 95% band (light)
    plt.fill_between(
        dates,
        lower_95,
        upper_95,
        color="C0",
        alpha=0.20,
        label="95% predictive interval",
    )

    # 50% band (darker)
    plt.fill_between(
        dates,
        lower_50,
        upper_50,
        color="C0",
        alpha=0.45,
        label="50% predictive interval",
    )

    # True returns
    plt.plot(dates, y_true, color="black", linewidth=1.0, label="True return")

    # Predicted mean
    plt.plot(dates, y_pred, color="C1", linewidth=1.5, label="Predicted mean")

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Log return")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_prediction_intervals_zoom(
    dates,
    y_true,
    y_pred,
    sigma,
    title: str,
    save_path: Path,
    n_points: int = 200,
):
    """
    Zoomed-in version of the prediction interval plot.

    We keep the same visual style (50% and 95% bands), but
    restrict to the first n_points of the test set.
    """
    n = min(n_points, len(dates))
    plot_prediction_intervals(
        dates[:n],
        y_true[:n],
        y_pred[:n],
        sigma[:n],
        title,
        save_path,
    )


def moving_average(x, window: int):
    """
    Simple centered moving average using convolution.

    Parameters
    ----------
    x : array-like
        Input time series.
    window : int
        Window size. If <= 1, the input is returned unchanged.
    """
    x = np.asarray(x)
    if window <= 1:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")


def plot_volatility_vs_realized(
    dates,
    y_true,
    sigma_ale,
    vix,
    title: str,
    save_path: Path,
    window: int = 5,
):
    """
    Volatility-only view for a single ticker.

    - Blue line: predicted next-day volatility sigma_t (aleatoric)
    - Orange line: moving average of |y_t| over 'window' days
                   as a proxy for realized volatility
    - Green dashed line: VIX, rescaled to a similar magnitude
                         as sigma_t for visual comparison

    This figure is useful to argue that the volatility head
    reacts to market regimes: when realized volatility and VIX spike,
    predicted sigma_t also increases.
    """
    dates = np.asarray(dates)
    y_true = np.asarray(y_true)
    sigma_ale = np.asarray(sigma_ale)
    vix = np.asarray(vix)

    realized = np.abs(y_true)
    realized_smooth = moving_average(realized, window=window)

    # Rescale VIX to roughly match the scale of sigma_ale
    vix_centered = (vix - vix.mean()) / (vix.std() + 1e-8)
    vix_rescaled = vix_centered * sigma_ale.std() + sigma_ale.mean()

    plt.figure(figsize=(12, 4))

    plt.plot(
        dates,
        sigma_ale,
        color="C0",
        linewidth=2,
        label=r"Predicted $\sigma_t$ (aleatoric)",
    )
    plt.plot(
        dates,
        realized_smooth,
        color="C1",
        linewidth=1.5,
        alpha=0.9,
        label=rf"$|y_t|$ {window}-day MA (realized vol)",
    )
    plt.plot(
        dates,
        vix_rescaled,
        "g--",
        linewidth=1.5,
        alpha=0.8,
        label="VIX (rescaled)",
    )

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Volatility scale")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_price_vs_sigma(
    dates,
    y_true,
    sigma_ale,
    ticker: str,
    save_path: Path,
):
    """
    Plot normalized price (reconstructed from log returns) together
    with predicted volatility sigma_t on a secondary y-axis.

    Steps:
      - Reconstruct a price index from log returns:
            P_t = exp( cumulative_sum(y_true) )
        and normalize it to start at 1.
      - Left y-axis: normalized price index
      - Right y-axis: predicted sigma_t

    This plot is especially helpful for discussion:
    large trends or drawdowns in price should coincide with
    spikes in predicted volatility.
    """
    dates = np.asarray(dates)
    y_true = np.asarray(y_true)
    sigma_ale = np.asarray(sigma_ale)

    # Reconstruct normalized price index from log returns
    price_index = np.exp(np.cumsum(y_true))
    price_index = price_index / price_index[0]  # start at 1

    fig, ax1 = plt.subplots(figsize=(12, 4))

    # Left axis: price index
    line1, = ax1.plot(
        dates,
        price_index,
        color="C0",
        linewidth=2,
        label="Normalized price (from returns)",
    )
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Normalized price", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")

    # Right axis: predicted volatility
    ax2 = ax1.twinx()
    line2, = ax2.plot(
        dates,
        sigma_ale,
        color="C1",
        linewidth=1.8,
        alpha=0.9,
        label=r"Predicted $\sigma_t$",
    )
    ax2.set_ylabel("Predicted volatility", color="C1")
    ax2.tick_params(axis="y", labelcolor="C1")

    # Combined legend
    lines = [line1, line2]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper left")

    plt.title(f"{ticker}: Price vs Predicted Volatility")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------


def main():
    pred_dir = Path("results/predictions")
    fig_dir = Path("results/figures_two_stage")
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Load all two-stage results (one entry per ticker)
    with open(pred_dir / "all_two_stage.json", "r") as f:
        results = json.load(f)

    print("\n=== Two-Stage Visualization Summary ===")

    for _, data in results.items():
        ticker = data["ticker"]
        print(f"\nTicker: {ticker}")

        # Extract arrays and flatten to 1D
        y_true = np.array(data["targets"]).reshape(-1)
        y_pred = np.array(data["predictions"]).reshape(-1)
        sigma_ale = np.array(data["sigma_aleatoric"]).reshape(-1)
        sigma_epi = np.array(data["sigma_epistemic"]).reshape(-1)
        vix = np.array(data["vix"]).reshape(-1)

        dates = np.arange(len(y_true))

        # For intervals we rely mostly on aleatoric sigma
        sigma_for_band = sigma_ale

        # Basic metrics
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        mae = float(np.mean(np.abs(y_true - y_pred)))
        cov50 = compute_coverage(y_true, y_pred, sigma_for_band, z=0.674)
        cov95 = compute_coverage(y_true, y_pred, sigma_for_band, z=1.96)

        print(f"  RMSE: {rmse:.6f}")
        print(f"  MAE : {mae:.6f}")
        print(f"  mean sigma_ale: {float(np.mean(sigma_ale)):.6f}")
        print(f"  mean sigma_epi: {float(np.mean(sigma_epi)):.6f}")
        print(f"  50% CI coverage (aleatoric only): {cov50:.3f}")
        print(f"  95% CI coverage (aleatoric only): {cov95:.3f}")

        # -------- Figure 1: Full prediction intervals --------
        plot_prediction_intervals(
            dates,
            y_true,
            y_pred,
            sigma_for_band,
            title=f"{ticker}: Two-Stage Predictions with Uncertainty",
            save_path=fig_dir / f"{ticker}_pred_interval_full.png",
        )

        # -------- Figure 2: Zoomed prediction intervals --------
        plot_prediction_intervals_zoom(
            dates,
            y_true,
            y_pred,
            sigma_for_band,
            title=f"{ticker}: Two-Stage Predictions (First 200 Days)",
            save_path=fig_dir / f"{ticker}_pred_interval_zoom.png",
            n_points=200,
        )

        # -------- Figure 3: Volatility vs realized vol and VIX --------
        plot_volatility_vs_realized(
            dates,
            y_true,
            sigma_ale,
            vix,
            title=f"{ticker}: Predicted Volatility vs Realized Volatility and VIX",
            save_path=fig_dir / f"{ticker}_vol_vs_realized.png",
            window=5,
        )

        # -------- Figure 4: Price vs predicted volatility --------
        plot_price_vs_sigma(
            dates,
            y_true,
            sigma_ale,
            ticker=ticker,
            save_path=fig_dir / f"{ticker}_price_vs_sigma.png",
        )

    print(f"\nAll figures saved to: {fig_dir.resolve()}")


if __name__ == "__main__":
    main()
