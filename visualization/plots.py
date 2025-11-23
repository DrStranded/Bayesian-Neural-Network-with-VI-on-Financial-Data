"""
Visualization and Plotting Module

This module provides visualization functions for stock price forecasts and model evaluation.
Includes functions for plotting predictions with uncertainty bands, training curves,
calibration plots, and comparative visualizations between different models.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def plot_predictions_with_uncertainty(dates, y_true, pred_mean, pred_std,
                                     title='Predictions with Uncertainty',
                                     save_path=None):
    """
    Plot predictions with confidence intervals

    Args:
        dates: Date array or indices [N]
        y_true: Ground truth [N, 1]
        pred_mean: Predictions [N, 1]
        pred_std: Uncertainties [N, 1]
        title: Plot title
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # Flatten arrays
    y_true = y_true.flatten()
    pred_mean = pred_mean.flatten()
    pred_std = pred_std.flatten()

    # Plot ground truth
    ax.plot(dates, y_true, 'k-', label='True', linewidth=1.5, alpha=0.7)

    # Plot predictions
    ax.plot(dates, pred_mean, 'b-', label='Prediction', linewidth=1.5)

    # 50% confidence interval
    ax.fill_between(dates,
                    pred_mean - 0.674 * pred_std,
                    pred_mean + 0.674 * pred_std,
                    alpha=0.3, color='blue', label='50% CI')

    # 95% confidence interval
    ax.fill_between(dates,
                    pred_mean - 1.96 * pred_std,
                    pred_mean + 1.96 * pred_std,
                    alpha=0.15, color='blue', label='95% CI')

    ax.set_xlabel('Time')
    ax.set_ylabel('Log Return')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    plt.close()


def plot_calibration_curve(calibration_data, model_names=None, save_path=None):
    """
    Plot calibration curves for multiple models

    Args:
        calibration_data: List of calibration dicts or single dict
        model_names: List of model names
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Ensure list
    if not isinstance(calibration_data, list):
        calibration_data = [calibration_data]
    if model_names is None:
        model_names = [f'Model {i+1}' for i in range(len(calibration_data))]

    colors = ['blue', 'red', 'green', 'orange', 'purple']

    # Plot ideal line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration', alpha=0.7)

    # Plot each model
    for i, (cal_data, name) in enumerate(zip(calibration_data, model_names)):
        expected = cal_data['expected_coverage']
        actual = cal_data['actual_coverage']

        color = colors[i % len(colors)]
        ax.plot(expected, actual, 'o-', color=color, linewidth=2,
                markersize=8, label=name, alpha=0.8)

    ax.set_xlabel('Expected Coverage', fontsize=12)
    ax.set_ylabel('Actual Coverage', fontsize=12)
    ax.set_title('Calibration Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    # Add diagonal helper lines
    for val in [0.5, 0.68, 0.9, 0.95]:
        ax.axvline(val, color='gray', linestyle=':', alpha=0.3, linewidth=0.8)
        ax.axhline(val, color='gray', linestyle=':', alpha=0.3, linewidth=0.8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    plt.close()


def plot_uncertainty_decomposition(dates, epistemic, aleatoric,
                                   title='Uncertainty Decomposition',
                                   save_path=None):
    """
    Stacked area plot of uncertainty components

    Args:
        dates: Date array [N]
        epistemic: Epistemic uncertainty [N, 1]
        aleatoric: Aleatoric uncertainty [N, 1]
        title: Plot title
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    epistemic = epistemic.flatten()
    aleatoric = aleatoric.flatten()

    # Stacked area plot
    ax.fill_between(dates, 0, aleatoric,
                    alpha=0.6, color='orange', label='Aleatoric (Data Noise)')
    ax.fill_between(dates, aleatoric, aleatoric + epistemic,
                    alpha=0.6, color='blue', label='Epistemic (Model Uncertainty)')

    # Total line
    total = epistemic + aleatoric
    ax.plot(dates, total, 'k-', linewidth=1.5, label='Total', alpha=0.7)

    ax.set_xlabel('Time')
    ax.set_ylabel('Uncertainty (Std Dev)')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    plt.close()


def plot_uncertainty_vs_error(pred_std, errors, title='Uncertainty vs Error',
                              save_path=None):
    """
    Scatter plot: predicted uncertainty vs actual error

    Args:
        pred_std: Predicted uncertainties [N, 1]
        errors: Absolute errors [N, 1]
        title: Plot title
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    pred_std = pred_std.flatten()
    errors = errors.flatten()

    # Scatter plot
    ax.scatter(pred_std, errors, alpha=0.5, s=20, color='blue')

    # Regression line
    z = np.polyfit(pred_std, errors, 1)
    p = np.poly1d(z)
    x_line = np.linspace(pred_std.min(), pred_std.max(), 100)
    ax.plot(x_line, p(x_line), 'r--', linewidth=2, label=f'Fit: y={z[0]:.3f}x+{z[1]:.3f}')

    # Correlation
    corr = np.corrcoef(pred_std, errors)[0, 1]
    ax.text(0.05, 0.95, f'Correlation: {corr:.4f}',
            transform=ax.transAxes, fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('Predicted Uncertainty (Std Dev)', fontsize=12)
    ax.set_ylabel('Actual Error (|y_true - y_pred|)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    plt.close()


def plot_vix_vs_prior(dates, vix_values, prior_stds, save_path=None):
    """
    Dual Y-axis: VIX vs Prior Standard Deviation

    Args:
        dates: Date array [N]
        vix_values: VIX index [N, 1]
        prior_stds: Prior standard deviations [N, 1]
        save_path: Path to save figure
    """
    fig, ax1 = plt.subplots(figsize=(14, 6))

    vix_values = vix_values.flatten()
    prior_stds = prior_stds.flatten()

    # VIX on left axis
    color = 'tab:red'
    ax1.set_xlabel('Time')
    ax1.set_ylabel('VIX Index', color=color, fontsize=12)
    ax1.plot(dates, vix_values, color=color, linewidth=2, label='VIX', alpha=0.7)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    # Prior std on right axis
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Prior Standard Deviation', color=color, fontsize=12)
    ax2.plot(dates, prior_stds, color=color, linewidth=2, label='Prior σ', alpha=0.7)
    ax2.tick_params(axis='y', labelcolor=color)

    # Title
    plt.title('VIX-Adaptive Prior Mechanism', fontsize=14, fontweight='bold')

    # Legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    plt.close()


def plot_regime_comparison(regime_data, save_path=None):
    """
    Bar chart: Model performance across VIX regimes

    Args:
        regime_data: List of dicts from stratify_by_vix
        save_path: Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    regimes = [d['regime'] for d in regime_data]
    rmse = [d['rmse'] for d in regime_data]
    uncertainty = [d['mean_uncertainty'] for d in regime_data]

    # RMSE by regime
    ax1.bar(regimes, rmse, color='steelblue', alpha=0.7)
    ax1.set_ylabel('RMSE', fontsize=12)
    ax1.set_title('Prediction Error by Market Regime', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # Uncertainty by regime
    ax2.bar(regimes, uncertainty, color='coral', alpha=0.7)
    ax2.set_ylabel('Mean Uncertainty', fontsize=12)
    ax2.set_title('Model Uncertainty by Market Regime', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    plt.close()


def plot_model_comparison(baseline_results, bayesian_results, save_path=None):
    """
    Comparison table: Baseline vs Bayesian models

    Args:
        baseline_results: Dict with MA and LSTM results
        bayesian_results: Dict with Bayesian results
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')

    # Prepare data
    tickers = list(baseline_results.keys())
    tickers = [t.replace('_ma', '').replace('_standard_lstm', '').replace('_bayesian_lstm', '')
               for t in tickers]
    tickers = sorted(set(tickers))

    table_data = []
    table_data.append(['Ticker', 'Model', 'RMSE', 'MAE', 'Unc-Err Corr'])

    for ticker in tickers:
        # MA
        if f'{ticker}_ma' in baseline_results:
            ma = baseline_results[f'{ticker}_ma']['metrics']
            table_data.append([ticker, 'MA', f"{ma['rmse']:.6f}", f"{ma['mae']:.6f}", 'N/A'])

        # LSTM
        if f'{ticker}_standard_lstm' in baseline_results:
            lstm = baseline_results[f'{ticker}_standard_lstm']['metrics']
            table_data.append(['', 'LSTM', f"{lstm['rmse']:.6f}", f"{lstm['mae']:.6f}", 'N/A'])

        # Bayesian
        if f'{ticker}_bayesian_lstm' in bayesian_results:
            bayes = bayesian_results[f'{ticker}_bayesian_lstm']['metrics']
            table_data.append(['', 'Bayesian',
                             f"{bayes['rmse']:.6f}",
                             f"{bayes['mae']:.6f}",
                             f"{bayes.get('uncertainty_error_corr', 0):.4f}"])

    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=[0.15, 0.2, 0.2, 0.2, 0.25])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header
    for i in range(5):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Style ticker rows
    for i, row in enumerate(table_data[1:], 1):
        if row[0]:  # Ticker name
            table[(i, 0)].set_facecolor('#E7E6E6')
            table[(i, 0)].set_text_props(weight='bold')

    plt.title('Model Performance Comparison', fontsize=14, fontweight='bold', pad=20)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    plt.close()


def main():
    """Test plotting functions"""
    print("="*80)
    print("Testing Plotting Functions")
    print("="*80)

    # Dummy data
    np.random.seed(42)
    n = 200
    dates = np.arange(n)
    y_true = np.random.randn(n, 1) * 0.02
    pred_mean = y_true + np.random.randn(n, 1) * 0.005
    pred_std = np.abs(np.random.randn(n, 1) * 0.01) + 0.005
    epistemic = pred_std * 0.6
    aleatoric = pred_std * 0.4
    errors = np.abs(y_true - pred_mean)
    vix = np.random.uniform(12, 35, n)
    prior_std = 0.1 * (1 + np.tanh((vix - 20) / 10))

    # Create output directory
    output_dir = Path('results/figures/test')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating plots...")

    # Plot 1
    plot_predictions_with_uncertainty(
        dates, y_true, pred_mean, pred_std,
        title='Test: Predictions with Uncertainty',
        save_path=output_dir / 'test_predictions.png'
    )

    # Plot 2
    from evaluation.calibration import compute_calibration_curve
    cal_data = compute_calibration_curve(y_true, pred_mean, pred_std)
    plot_calibration_curve(
        cal_data,
        model_names=['Test Model'],
        save_path=output_dir / 'test_calibration.png'
    )

    # Plot 3
    plot_uncertainty_decomposition(
        dates, epistemic, aleatoric,
        save_path=output_dir / 'test_uncertainty_decomp.png'
    )

    # Plot 4
    plot_uncertainty_vs_error(
        pred_std, errors,
        save_path=output_dir / 'test_unc_vs_error.png'
    )

    # Plot 5
    plot_vix_vs_prior(
        dates, vix.reshape(-1, 1), prior_std.reshape(-1, 1),
        save_path=output_dir / 'test_vix_prior.png'
    )

    print("\n" + "="*80)
    print(f"All test plots saved to: {output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
