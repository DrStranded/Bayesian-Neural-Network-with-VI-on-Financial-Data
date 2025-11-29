"""
Visualization for Linear Baselines (MA + BLR)
Separate from Bayesian LSTM plots
"""

import sys
from pathlib import Path
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from training.config import Config

# Style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 300

def plot_linear_baseline_predictions(ticker, data, save_dir):
    """
    Plot MA and BLR predictions

    Args:
        ticker: stock ticker
        data: dict with ma, blr, true_values
        save_dir: output directory
    """
    true_values = np.array(data['true_values']).flatten()
    ma_pred = np.array(data['ma']['predictions']).flatten()
    blr_pred = np.array(data['blr']['predictions']).flatten()
    blr_std = np.array(data['blr']['uncertainty']).flatten()

    n = len(true_values)
    time = np.arange(n)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Plot 1: MA
    ax1.plot(time, true_values, 'k-', label='True', alpha=0.7, linewidth=1)
    ax1.plot(time, ma_pred, 'b-', label='MA Prediction', alpha=0.8, linewidth=1.5)
    ax1.fill_between(time, true_values, ma_pred, alpha=0.2, color='blue')
    ax1.set_ylabel('Log Return')
    ax1.set_title(f'{ticker}: Moving Average Baseline (window=20)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: BLR with uncertainty
    ax2.plot(time, true_values, 'k-', label='True', alpha=0.7, linewidth=1)
    ax2.plot(time, blr_pred, 'r-', label='BLR Prediction', alpha=0.8, linewidth=1.5)

    # 50% CI
    ax2.fill_between(time,
                     blr_pred - 0.67 * blr_std,
                     blr_pred + 0.67 * blr_std,
                     alpha=0.3, color='red', label='50% CI')

    # 95% CI
    ax2.fill_between(time,
                     blr_pred - 1.96 * blr_std,
                     blr_pred + 1.96 * blr_std,
                     alpha=0.15, color='red', label='95% CI')

    ax2.set_xlabel('Time')
    ax2.set_ylabel('Log Return')
    ax2.set_title(f'{ticker}: Bayesian Linear Regression')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / f'{ticker}_linear_baselines_predictions.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved {ticker} predictions plot")

def plot_blr_coefficients(all_data, save_dir):
    """
    Plot BLR coefficients for all tickers
    """
    config = Config()
    feature_names = config.FEATURES
    tickers = list(all_data.keys())

    # Gather coefficients
    coef_matrix = []
    for ticker in tickers:
        coef_matrix.append(all_data[ticker]['blr']['coefficients'])

    coef_matrix = np.array(coef_matrix)  # [n_tickers, n_features]

    # Heatmap
    fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(coef_matrix, cmap='RdBu_r', aspect='auto', vmin=-0.5, vmax=0.5)

    # Ticks
    ax.set_xticks(np.arange(len(feature_names)))
    ax.set_yticks(np.arange(len(tickers)))
    ax.set_xticklabels(feature_names, rotation=45, ha='right')
    ax.set_yticklabels(tickers)

    # Annotate
    for i in range(len(tickers)):
        for j in range(len(feature_names)):
            text = ax.text(j, i, f'{coef_matrix[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=10)

    ax.set_title('Bayesian Linear Regression Coefficients')
    plt.colorbar(im, ax=ax, label='Coefficient Value')

    plt.tight_layout()
    plt.savefig(save_dir / 'blr_coefficients_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved BLR coefficients heatmap")

def plot_model_comparison_table(all_data, save_dir):
    """
    Comparison table: MA vs BLR vs Bayesian LSTM
    """
    config = Config()
    tickers = config.TICKERS

    # Load Bayesian LSTM results
    bayesian_path = Path('results/predictions/all_bayesian.json')
    if bayesian_path.exists():
        with open(bayesian_path) as f:
            bayesian_data = json.load(f)
    else:
        bayesian_data = None

    # Create table
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')

    # Headers
    headers = ['Ticker', 'Model', 'RMSE', 'MAE', 'Unc-Err Corr']

    # Data rows
    table_data = []

    for ticker in tickers:
        # MA
        ma_metrics = all_data[ticker]['ma']['metrics']
        table_data.append([
            ticker,
            'MA',
            f"{ma_metrics['rmse']:.6f}",
            f"{ma_metrics['mae']:.6f}",
            'N/A'
        ])

        # BLR
        blr_metrics = all_data[ticker]['blr']['metrics']
        table_data.append([
            '',
            'BLR',
            f"{blr_metrics['rmse']:.6f}",
            f"{blr_metrics['mae']:.6f}",
            f"{blr_metrics['uncertainty_error_corr']:.4f}"
        ])

        # Bayesian LSTM (if available)
        if bayesian_data and ticker in bayesian_data:
            b_metrics = bayesian_data[ticker]['metrics']
            table_data.append([
                '',
                'Bayesian LSTM',
                f"{b_metrics['rmse']:.6f}",
                f"{b_metrics['mae']:.6f}",
                f"{b_metrics.get('uncertainty_error_corr', 0):.4f}"
            ])

    # Create table
    table = ax.table(cellText=table_data, colLabels=headers,
                    cellLoc='center', loc='center',
                    colWidths=[0.15, 0.25, 0.2, 0.2, 0.2])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Style alternating rows
    for i in range(1, len(table_data) + 1):
        if (i - 1) // 3 % 2 == 0:
            for j in range(len(headers)):
                table[(i, j)].set_facecolor('#D9E1F2')

    plt.title('Model Performance Comparison', fontsize=14, fontweight='bold', pad=20)

    plt.savefig(save_dir / 'linear_baselines_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved comparison table")

def main():
    """Generate all plots for linear baselines"""
    print('='*80)
    print('GENERATING LINEAR BASELINE PLOTS')
    print('='*80)

    # Load data
    data_path = Path('results/predictions/linear_baselines/all_linear_baselines.json')

    if not data_path.exists():
        print(f"ERROR: {data_path} not found!")
        print("Run experiments/run_linear_baselines.py first")
        return

    with open(data_path) as f:
        all_data = json.load(f)

    # Create output directory
    save_dir = Path('results/figures/linear_baselines')
    save_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    print("\nGenerating individual prediction plots...")
    for ticker in all_data.keys():
        plot_linear_baseline_predictions(ticker, all_data[ticker], save_dir)

    print("\nGenerating BLR coefficients heatmap...")
    plot_blr_coefficients(all_data, save_dir)

    print("\nGenerating comparison table...")
    plot_model_comparison_table(all_data, save_dir)

    print(f"\n{'='*80}")
    print('✓ All linear baseline plots generated')
    print(f'  Output: {save_dir}')
    print('='*80)

if __name__ == '__main__':
    main()
