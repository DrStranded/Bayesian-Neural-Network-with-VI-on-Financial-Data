"""
Comprehensive Visualization for Two-Stage Bayesian LSTM

Generate more diagnostic and convincing plots for:
- calibration
- PIT and Q-Q checks
- uncertainty vs error
- volatility regimes
- time-varying volatility
- zoomed prediction windows
- summary tables and cross-ticker comparisons
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

# Global plotting style
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 150


def load_results(ticker, pred_dir="results/predictions"):
    """Load saved prediction results for a given ticker."""
    with open(f"{pred_dir}/{ticker}_two_stage.json", "r") as f:
        data = json.load(f)
    return data


def plot_1_calibration_diagram(tickers, results_dict, save_dir):
    """
    Figure 1: Calibration diagram (most important!)
    Expected coverage vs. actual coverage for multiple confidence levels.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    confidence_levels = np.array(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    )

    colors = {"AAPL": "#1f77b4", "SPY": "#2ca02c", "TSLA": "#d62728"}
    markers = {"AAPL": "o", "SPY": "s", "TSLA": "^"}

    for ticker in tickers:
        data = results_dict[ticker]
        y_true = np.array(data["targets"]).flatten()
        y_pred = np.array(data["predictions"]).flatten()
        # use correct key from JSON
        sigma = np.array(data["sigma_aleatoric"]).flatten()

        actual_coverage = []
        for conf in confidence_levels:
            z = stats.norm.ppf((1 + conf) / 2)
            lower = y_pred - z * sigma
            upper = y_pred + z * sigma
            coverage = np.mean((y_true >= lower) & (y_true <= upper))
            actual_coverage.append(coverage)

        ax.plot(
            confidence_levels,
            actual_coverage,
            marker=markers[ticker],
            markersize=8,
            linewidth=2,
            color=colors[ticker],
            label=ticker,
        )

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", linewidth=2, label="Perfect calibration")

    # ±5% tolerance band
    ax.fill_between(
        [0, 1],
        [0 - 0.05, 1 - 0.05],
        [0 + 0.05, 1 + 0.05],
        color="gray",
        alpha=0.2,
        label="±5% tolerance",
    )

    ax.set_xlabel("Expected coverage (confidence level)")
    ax.set_ylabel("Actual coverage")
    ax.set_title("Calibration diagram: expected vs. actual coverage")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / "calibration_diagram.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ calibration_diagram.png")


def plot_2_pit_histogram(tickers, results_dict, save_dir):
    """
    Figure 2: PIT (Probability Integral Transform) histograms.
    If the model is well calibrated, PIT values should be ~Uniform(0, 1).
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, ticker in enumerate(tickers):
        data = results_dict[ticker]
        y_true = np.array(data["targets"]).flatten()
        y_pred = np.array(data["predictions"]).flatten()
        sigma = np.array(data["sigma_aleatoric"]).flatten()

        # PIT: F(y_true | y_pred, sigma)
        z_scores = (y_true - y_pred) / sigma
        pit_values = stats.norm.cdf(z_scores)

        ax = axes[idx]
        ax.hist(
            pit_values,
            bins=20,
            density=True,
            alpha=0.7,
            color="steelblue",
            edgecolor="white",
        )
        ax.axhline(
            y=1.0,
            color="red",
            linestyle="--",
            linewidth=2,
            label="Uniform (ideal)",
        )

        # KS test for uniformity
        ks_stat, ks_pval = stats.kstest(pit_values, "uniform")

        ax.set_xlabel("PIT value")
        ax.set_ylabel("Density")
        ax.set_title(f"{ticker}\nKS test p-value: {ks_pval:.3f}")
        ax.set_xlim([0, 1])
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Probability Integral Transform (PIT) histograms\nUniform → well-calibrated",
        fontsize=14,
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(save_dir / "pit_histograms.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ pit_histograms.png")


def plot_3_qq_plot(tickers, results_dict, save_dir):
    """
    Figure 3: Q-Q plots of standardized residuals.
    If residuals are Gaussian with correct σ, points should lie on the red line.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, ticker in enumerate(tickers):
        data = results_dict[ticker]
        y_true = np.array(data["targets"]).flatten()
        y_pred = np.array(data["predictions"]).flatten()
        sigma = np.array(data["sigma_aleatoric"]).flatten()

        z_scores = (y_true - y_pred) / sigma

        ax = axes[idx]
        stats.probplot(z_scores, dist="norm", plot=ax)
        ax.set_title(
            f"{ticker}\nz-mean={np.mean(z_scores):.3f}, z-std={np.std(z_scores):.3f}"
        )
        ax.grid(True, alpha=0.3)

        # Make the fitted line red and thicker
        ax.get_lines()[1].set_color("red")
        ax.get_lines()[1].set_linewidth(2)

    fig.suptitle(
        "Q-Q plots of standardized residuals\nIdeal: points on red line",
        fontsize=14,
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(save_dir / "qq_plots.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ qq_plots.png")


def plot_4_uncertainty_vs_error(tickers, results_dict, save_dir):
    """
    Figure 4: Predicted uncertainty vs. actual absolute error.
    Shows whether the model “knows when it does not know”.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, ticker in enumerate(tickers):
        data = results_dict[ticker]
        y_true = np.array(data["targets"]).flatten()
        y_pred = np.array(data["predictions"]).flatten()
        sigma = np.array(data["sigma_aleatoric"]).flatten()

        errors = np.abs(y_true - y_pred)

        ax = axes[idx]
        ax.scatter(sigma, errors, alpha=0.3, s=20, c="steelblue")

        # Binned mean error vs sigma
        n_bins = 10
        bin_edges = np.percentile(sigma, np.linspace(0, 100, n_bins + 1))
        bin_centers = []
        bin_mean_errors = []

        for i in range(n_bins):
            mask = (sigma >= bin_edges[i]) & (sigma < bin_edges[i + 1])
            if np.sum(mask) > 0:
                bin_centers.append(np.mean(sigma[mask]))
                bin_mean_errors.append(np.mean(errors[mask]))

        ax.plot(
            bin_centers,
            bin_mean_errors,
            "r-o",
            linewidth=2,
            markersize=8,
            label="Binned mean error",
        )

        # Ideal proportionality line: error ∝ σ
        sigma_range = np.linspace(sigma.min(), sigma.max(), 100)
        ax.plot(
            sigma_range,
            sigma_range * 0.8,
            "g--",
            linewidth=2,
            alpha=0.7,
            label="Ideal (error ∝ σ)",
        )

        corr = np.corrcoef(sigma, errors)[0, 1]

        ax.set_xlabel("Predicted uncertainty (σ)")
        ax.set_ylabel("Actual absolute error")
        ax.set_title(f"{ticker}\nCorrelation: {corr:.3f}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Uncertainty vs. actual error\nPositive correlation → model knows when it is uncertain",
        fontsize=14,
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(save_dir / "uncertainty_vs_error.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ uncertainty_vs_error.png")


def plot_5_volatility_regimes(tickers, results_dict, save_dir):
    """
    Figure 5: Performance across volatility regimes.
    Split test points into low / medium / high σ and compare coverage.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, ticker in enumerate(tickers):
        data = results_dict[ticker]
        y_true = np.array(data["targets"]).flatten()
        y_pred = np.array(data["predictions"]).flatten()
        sigma = np.array(data["sigma_aleatoric"]).flatten()

        # Split σ into three regimes
        low_thresh = np.percentile(sigma, 33)
        high_thresh = np.percentile(sigma, 67)

        regimes = ["Low σ", "Medium σ", "High σ"]
        masks = [
            sigma <= low_thresh,
            (sigma > low_thresh) & (sigma <= high_thresh),
            sigma > high_thresh,
        ]

        coverages_50 = []
        coverages_95 = []

        for mask in masks:
            y_t = y_true[mask]
            y_p = y_pred[mask]
            s = sigma[mask]

            if len(y_t) == 0:
                coverages_50.append(np.nan)
                coverages_95.append(np.nan)
                continue

            # 50% CI
            z50 = 0.674
            cov50 = np.mean((y_t >= y_p - z50 * s) & (y_t <= y_p + z50 * s))
            coverages_50.append(cov50)

            # 95% CI
            z95 = 1.96
            cov95 = np.mean((y_t >= y_p - z95 * s) & (y_t <= y_p + z95 * s))
            coverages_95.append(cov95)

        ax = axes[idx]
        x = np.arange(3)
        width = 0.25

        ax.bar(
            x - width,
            coverages_50,
            width,
            label="50% CI coverage",
            color="steelblue",
        )
        ax.bar(
            x,
            coverages_95,
            width,
            label="95% CI coverage",
            color="coral",
        )

        # Target lines
        ax.axhline(y=0.5, color="steelblue", linestyle="--", alpha=0.5)
        ax.axhline(y=0.95, color="coral", linestyle="--", alpha=0.5)

        ax.set_xlabel("Volatility regime")
        ax.set_ylabel("Coverage")
        ax.set_title(f"{ticker}")
        ax.set_xticks(x)
        ax.set_xticklabels(regimes)
        ax.legend()
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        "Coverage by volatility regime\nStable coverage → robust uncertainty estimation",
        fontsize=14,
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(save_dir / "volatility_regimes.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ volatility_regimes.png")


def plot_6_rolling_coverage(tickers, results_dict, save_dir):
    """
    Figure 6: Rolling 95% coverage over time.
    Shows temporal stability of calibration.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    window = 50  # rolling window size

    for idx, ticker in enumerate(tickers):
        data = results_dict[ticker]
        y_true = np.array(data["targets"]).flatten()
        y_pred = np.array(data["predictions"]).flatten()
        sigma = np.array(data["sigma_aleatoric"]).flatten()

        z95 = 1.96
        in_ci = (y_true >= y_pred - z95 * sigma) & (y_true <= y_pred + z95 * sigma)

        rolling_cov = np.convolve(
            in_ci.astype(float), np.ones(window) / window, mode="valid"
        )

        ax = axes[idx]
        time = np.arange(len(rolling_cov))

        ax.plot(
            time,
            rolling_cov,
            color="steelblue",
            linewidth=1.5,
            label="Rolling 95% CI coverage",
        )
        ax.axhline(y=0.95, color="red", linestyle="--", linewidth=2, label="Target 0.95")
        ax.fill_between(time, 0.90, 1.0, color="green", alpha=0.1, label="Good range")

        ax.set_ylabel("Coverage")
        ax.set_title(f"{ticker}")
        ax.legend(loc="lower right")
        ax.set_ylim([0.7, 1.05])
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time")
    fig.suptitle(
        f"Rolling 95% CI coverage (window={window})\n"
        "Stable around 0.95 → consistent calibration",
        fontsize=14,
        y=1.01,
    )
    plt.tight_layout()
    plt.savefig(save_dir / "rolling_coverage.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ rolling_coverage.png")


def plot_7_sigma_timeseries(tickers, results_dict, save_dir):
    """
    Figure 7: Time series of predicted volatility vs realized volatility.
    Shows that σ_t tracks volatility regimes.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    for idx, ticker in enumerate(tickers):
        data = results_dict[ticker]
        y_true = np.array(data["targets"]).flatten()
        sigma_ale = np.array(data["sigma_aleatoric"]).flatten()
        sigma_epi = np.array(data["sigma_epistemic"]).flatten()

        # Realized volatility: rolling 20-day std of returns
        window = 20
        realized_vol = np.array(
            [np.std(y_true[max(0, i - window) : i + 1]) for i in range(len(y_true))]
        )

        time = np.arange(len(y_true))

        ax = axes[idx]
        ax.plot(
            time,
            sigma_ale,
            color="steelblue",
            linewidth=1.5,
            label="Predicted σ (aleatoric)",
            alpha=0.8,
        )
        ax.plot(
            time,
            realized_vol,
            color="coral",
            linewidth=1.5,
            label="Realized volatility (20d)",
            alpha=0.8,
        )

        corr = np.corrcoef(sigma_ale[window:], realized_vol[window:])[0, 1]

        ax.set_ylabel("Volatility")
        ax.set_title(f"{ticker} | Corr(pred σ, realized vol) = {corr:.3f}")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time")
    fig.suptitle(
        "Predicted vs. realized volatility\nHigh correlation → good volatility forecasting",
        fontsize=14,
        y=1.01,
    )
    plt.tight_layout()
    plt.savefig(save_dir / "sigma_timeseries.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ sigma_timeseries.png")


def plot_8_prediction_zoomed(tickers, results_dict, save_dir):
    """
    Figure 8: Zoomed prediction windows with uncertainty bands.
    Show first 100 days for each ticker.
    """
    for ticker in tickers:
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        data = results_dict[ticker]
        y_true = np.array(data["targets"]).flatten()
        y_pred = np.array(data["predictions"]).flatten()
        sigma = np.array(data["sigma_aleatoric"]).flatten()

        n = min(100, len(y_true))
        time = np.arange(n)

        # Top: predictive intervals
        ax1 = axes[0]
        ax1.fill_between(
            time,
            y_pred[:n] - 1.96 * sigma[:n],
            y_pred[:n] + 1.96 * sigma[:n],
            color="lightblue",
            alpha=0.5,
            label="95% CI",
        )
        ax1.fill_between(
            time,
            y_pred[:n] - 0.674 * sigma[:n],
            y_pred[:n] + 0.674 * sigma[:n],
            color="steelblue",
            alpha=0.5,
            label="50% CI",
        )
        ax1.plot(time, y_true[:n], "k-", linewidth=1, label="Actual return")
        ax1.axhline(
            y=0,
            color="orange",
            linestyle="-",
            linewidth=2,
            alpha=0.7,
            label="Predicted mean (≈0)",
        )
        ax1.set_ylabel("Log return")
        ax1.set_title(f"{ticker}: predictions with uncertainty (first {n} days)")
        ax1.legend(loc="upper right")
        ax1.grid(True, alpha=0.3)

        # Bottom: volatility over the same window
        ax2 = axes[1]
        ax2.fill_between(time, 0, sigma[:n], color="steelblue", alpha=0.5)
        ax2.plot(time, sigma[:n], "b-", linewidth=1.5, label="Predicted σ")
        ax2.set_ylabel("Predicted volatility (σ)")
        ax2.set_xlabel("Time")
        ax2.legend(loc="upper right")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            save_dir / f"{ticker}_prediction_zoomed.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

    print("  ✓ *_prediction_zoomed.png for all tickers")


def plot_9_summary_table(tickers, results_dict, save_dir):
    """
    Figure 9: Summary table of key metrics for all tickers.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")

    rows = []
    for ticker in tickers:
        data = results_dict[ticker]
        y_true = np.array(data["targets"]).flatten()
        y_pred = np.array(data["predictions"]).flatten()
        sigma_ale = np.array(data["sigma_aleatoric"]).flatten()
        sigma_epi = np.array(data["sigma_epistemic"]).flatten()

        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mae = np.mean(np.abs(y_true - y_pred))

        z95 = 1.96
        cov95 = np.mean(
            (y_true >= y_pred - z95 * sigma_ale)
            & (y_true <= y_pred + z95 * sigma_ale)
        )
        z50 = 0.674
        cov50 = np.mean(
            (y_true >= y_pred - z50 * sigma_ale)
            & (y_true <= y_pred + z50 * sigma_ale)
        )

        z = (y_true - y_pred) / sigma_ale
        z_mean = np.mean(z)
        z_std = np.std(z)

        rows.append(
            [
                ticker,
                f"{rmse:.4f}",
                f"{mae:.4f}",
                f"{np.mean(sigma_ale):.4f}",
                f"{np.mean(sigma_epi):.5f}",
                f"{cov50:.1%}",
                f"{cov95:.1%}",
                f"{z_mean:.3f}",
                f"{z_std:.3f}",
            ]
        )

    columns = [
        "Ticker",
        "RMSE",
        "MAE",
        "Mean σ_ale",
        "Mean σ_epi",
        "50% cov",
        "95% cov",
        "z-mean",
        "z-std",
    ]

    table = ax.table(
        cellText=rows,
        colLabels=columns,
        loc="center",
        cellLoc="center",
        colColours=["lightsteelblue"] * len(columns),
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Highlight “good” cells in green
    for i in range(len(tickers)):
        cov95_val = float(rows[i][6].rstrip("%")) / 100
        if abs(cov95_val - 0.95) < 0.02:
            table[(i + 1, 6)].set_facecolor("lightgreen")

        z_std_val = float(rows[i][8])
        if abs(z_std_val - 1.0) < 0.1:
            table[(i + 1, 8)].set_facecolor("lightgreen")

    ax.set_title(
        "Summary of model performance\n(Green = excellent)",
        fontsize=14,
        pad=20,
    )

    plt.tight_layout()
    plt.savefig(save_dir / "summary_table.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ summary_table.png")


def plot_10_comparison_bars(tickers, results_dict, save_dir):
    """
    Figure 10: Cross-ticker bar charts (RMSE, mean σ, 95% coverage).
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = {"RMSE": [], "Mean σ": [], "95% coverage": []}

    for ticker in tickers:
        data = results_dict[ticker]
        y_true = np.array(data["targets"]).flatten()
        y_pred = np.array(data["predictions"]).flatten()
        sigma = np.array(data["sigma_aleatoric"]).flatten()

        metrics["RMSE"].append(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        metrics["Mean σ"].append(np.mean(sigma))

        z95 = 1.96
        cov = np.mean(
            (y_true >= y_pred - z95 * sigma) & (y_true <= y_pred + z95 * sigma)
        )
        metrics["95% coverage"].append(cov)

    colors = ["#1f77b4", "#2ca02c", "#d62728"]

    # RMSE
    ax = axes[0]
    bars = ax.bar(tickers, metrics["RMSE"], color=colors)
    ax.set_ylabel("RMSE")
    ax.set_title("Prediction error (RMSE)\nLower is better")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, metrics["RMSE"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Mean σ
    ax = axes[1]
    bars = ax.bar(tickers, metrics["Mean σ"], color=colors)
    ax.set_ylabel("Mean σ")
    ax.set_title("Average predicted uncertainty\nShould roughly match RMSE")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, metrics["Mean σ"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # 95% coverage
    ax = axes[2]
    bars = ax.bar(tickers, metrics["95% coverage"], color=colors)
    ax.axhline(
        y=0.95,
        color="red",
        linestyle="--",
        linewidth=2,
        label="Target (0.95)",
    )
    ax.set_ylabel("Coverage")
    ax.set_title("95% CI coverage\nCloser to 0.95 is better")
    ax.set_ylim([0.85, 1.0])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, metrics["95% coverage"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.1%}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.suptitle("Cross-stock comparison", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_dir / "comparison_bars.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ comparison_bars.png")


def plot_11_sharpness_reliability(tickers, results_dict, save_dir):
    """
    Figure 11: Sharpness–reliability diagram.
    Sharpness: average 95% CI width (smaller is better).
    Reliability: 95% coverage (closer to 0.95 is better).
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = {"AAPL": "#1f77b4", "SPY": "#2ca02c", "TSLA": "#d62728"}

    for ticker in tickers:
        data = results_dict[ticker]
        y_true = np.array(data["targets"]).flatten()
        y_pred = np.array(data["predictions"]).flatten()
        sigma = np.array(data["sigma_aleatoric"]).flatten()

        sharpness = np.mean(2 * 1.96 * sigma)

        z95 = 1.96
        coverage = np.mean(
            (y_true >= y_pred - z95 * sigma) & (y_true <= y_pred + z95 * sigma)
        )

        ax.scatter(
            sharpness,
            coverage,
            s=200,
            c=colors[ticker],
            label=ticker,
            edgecolors="black",
            linewidth=2,
        )
        ax.annotate(
            ticker,
            (sharpness, coverage),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=12,
            fontweight="bold",
        )

    ax.axhline(
        y=0.95,
        color="red",
        linestyle="--",
        linewidth=2,
        label="Target coverage (0.95)",
    )

    ax.set_xlabel("Sharpness (mean 95% CI width)\n← narrower is better")
    ax.set_ylabel("Reliability (95% CI coverage)\n↑ closer to 0.95 is better")
    ax.set_title(
        "Sharpness vs. reliability\nIdeal: lower-right near red line",
        fontsize=14,
    )
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        save_dir / "sharpness_reliability.png", dpi=150, bbox_inches="tight"
    )
    plt.close()
    print("  ✓ sharpness_reliability.png")


def main():
    tickers = ["AAPL", "SPY", "TSLA"]
    pred_dir = Path("results/predictions")
    save_dir = Path("results/figures_comprehensive")
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Generating comprehensive visualizations")
    print("=" * 60)

    results_dict = {}
    for ticker in tickers:
        results_dict[ticker] = load_results(ticker, pred_dir)
        print(f"Loaded {ticker} results")

    print("\nGenerating plots...")

    plot_1_calibration_diagram(tickers, results_dict, save_dir)
    plot_2_pit_histogram(tickers, results_dict, save_dir)
    plot_3_qq_plot(tickers, results_dict, save_dir)
    plot_4_uncertainty_vs_error(tickers, results_dict, save_dir)
    plot_5_volatility_regimes(tickers, results_dict, save_dir)
    plot_6_rolling_coverage(tickers, results_dict, save_dir)
    plot_7_sigma_timeseries(tickers, results_dict, save_dir)
    plot_8_prediction_zoomed(tickers, results_dict, save_dir)
    plot_9_summary_table(tickers, results_dict, save_dir)
    plot_10_comparison_bars(tickers, results_dict, save_dir)
    plot_11_sharpness_reliability(tickers, results_dict, save_dir)

    print("\n" + "=" * 60)
    print(f"All plots saved to {save_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
