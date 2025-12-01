"""
Visualization for the Two-Stage LSTM + Bayesian Volatility Model.

What we want to show:
  1) Prediction plots with uncertainty bands (no need to highlight the mean line)
  2) How the learned volatility behaves vs realized volatility and VIX

Reads:  results/predictions/all_two_stage.json
Saves:  results/figures_two_stage/*.png
"""

import sys
from pathlib import Path

# Make project root importable if needed (for future extensions)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------


def moving_average(x, window: int):
    """
    Simple centered moving average using convolution.

    Parameters
    ----------
    x : array-like
        Input 1D series.
    window : int
        Window size. If <= 1, returns x unchanged.
    """
    x = np.asarray(x)
    if window <= 1:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")


def plot_prediction_interval(
    dates,
    y_true,
    y_pred,
    sigma,
    title: str,
    save_path: Path,
    z: float = 1.96,
):
    """
    Prediction plot focusing on intervals instead of the mean:

    - We draw a shaded band for the predictive interval:
        [mu_t - z * sigma_t, mu_t + z * sigma_t]
    - We plot only the observed returns as a line (or markers).
    - The mean prediction mu_t is used to center the band,
      but we do not show it as a separate line to avoid clutter.

    This makes the story about "intervals and uncertainty",
    not about perfectly matching the mean.
    """
    dates = np.asarray(dates)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    sigma = np.asarray(sigma)

    lower = y_pred - z * sigma
    upper = y_pred + z * sigma

    plt.figure(figsize=(10, 4))
    # Shaded predictive interval
    plt.fill_between(
        dates,
        lower,
        upper,
        color="C0",
        alpha=0.25,
        label=f"{int(round(100 * (2 * (0.5 * (1 + 1)))))}% predictive interval"
        # this is just a label; you can change it to "95% predictive interval"
    )
    # Observed returns
    plt.plot(dates, y_true, color="black", linewidth=1.0, label="Observed return")

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Log return")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_prediction_interval_zoom(
    dates,
    y_true,
    y_pred,
    sigma,
    title: str,
    save_path: Path,
    z: float = 1.96,
    n_points: int = 200,
):
    """
    Zoomed-in version of the prediction-interval plot
    on the first n_points time steps.

    This is often visually cleaner for the presentation,
    especially when the full test set is long.
    """
    n = min(n_points, len(dates))
    plot_prediction_interval(
        dates[:n],
        y_true[:n],
        y_pred[:n],
        sigma[:n],
        title=title,
        save_path=save_path,
        z=z,
    )


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
    Time-series plot for a single ticker:

    - Predicted volatility sigma_t (aleatoric)
    - Realized volatility: moving average of |y_t|
    - VIX, rescaled to roughly match sigma_t scale

    This is used to show that our learned volatility reacts to
    market conditions and roughly tracks realized volatility and VIX.
    """
    dates = np.asarray(dates)
    y_true = np.asarray(y_true)
    sigma_ale = np.asarray(sigma_ale)
    vix = np.asarray(vix)

    # Realized volatility proxy: smoothed absolute returns
    realized = np.abs(y_true)
    realized_smooth = moving_average(realized, window=window)

    # Rescale VIX to the same rough scale as sigma_ale for visual comparison
    vix_centered = (vix - vix.mean()) / (vix.std() + 1e-8)
    vix_rescaled = vix_centered * sigma_ale.std() + sigma_ale.mean()

    plt.figure(figsize=(10, 4))
    plt.plot(
        dates,
        sigma_ale,
        label=r"Predicted $\sigma_t$ (aleatoric)",
        linewidth=2,
    )
    plt.plot(
        dates,
        realized_smooth,
        label=rf"$|y_t|$ {window}-day MA (realized vol)",
        linewidth=1.5,
        alpha=0.8,
    )
    plt.plot(
        dates,
        vix_rescaled,
        "--",
        linewidth=1.5,
        alpha=0.7,
        label="VIX (rescaled)",
    )

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Volatility scale")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# ---------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------


def main():
    pred_dir = Path("results/predictions")
    fig_dir = Path("results/figures_two_stage")
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Load per-ticker predictions and uncertainties
    with open(pred_dir / "all_two_stage.json", "r") as f:
        results = json.load(f)

    for key, data in results.items():
        ticker = data["ticker"]
        print(f"\nProcessing {ticker} (Two-Stage Model)...")

        # Extract arrays and flatten to 1D
        y_true = np.array(data["targets"]).reshape(-1)
        y_pred = np.array(data["predictions"]).reshape(-1)
        sigma_ale = np.array(data["sigma_aleatoric"]).reshape(-1)
        sigma_epi = np.array(data["sigma_epistemic"]).reshape(-1)
        vix = np.array(data["vix"]).reshape(-1)

        dates = np.arange(len(y_true))

        # For intervals we mainly trust aleatoric uncertainty
        # (epistemic is very small in this model), so we use sigma_ale.
        sigma_for_band = sigma_ale

        # ========== Figure 1: Full prediction interval over the test set ==========
        plot_prediction_interval(
            dates,
            y_true,
            y_pred,
            sigma_for_band,
            title=f"{ticker}: Predictive Intervals over Time",
            save_path=fig_dir / f"{ticker}_pred_interval_full.png",
            z=1.96,  # ~95% CI
        )

        # ========== Figure 2: Zoomed prediction interval (first 200 points) ==========
        plot_prediction_interval_zoom(
            dates,
            y_true,
            y_pred,
            sigma_for_band,
            title=f"{ticker}: Predictive Intervals (First 200 Days)",
            save_path=fig_dir / f"{ticker}_pred_interval_zoom.png",
            z=1.96,
            n_points=200,
        )

        # ========== Figure 3: Volatility vs realized volatility vs VIX ==========
        plot_volatility_vs_realized(
            dates,
            y_true,
            sigma_ale,
            vix,
            title=f"{ticker}: Predicted Volatility vs Realized Volatility and VIX",
            save_path=fig_dir / f"{ticker}_vol_vs_realized.png",
            window=5,
        )

        # Print a short numeric summary (optional, for the console)
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        print(
            f"  RMSE={rmse:.5f}, "
            f"mean sigma_ale={float(np.mean(sigma_ale)):.5f}, "
            f"mean sigma_epi={float(np.mean(sigma_epi)):.5f}"
        )

    print(f"\nAll two-stage plots saved to {fig_dir}/")


if __name__ == "__main__":
    main()
