"""
Visualization for MC Dropout LSTM results
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

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 300


def plot_predictions_with_uncertainty(ticker, data, save_dir):
    """Plot predictions with epistemic and aleatoric uncertainty"""
    true_values = np.array(data['true_values']).flatten()
    predictions = np.array(data['predictions']).flatten()
    epistemic = np.array(data['epistemic']).flatten()
    aleatoric = np.array(data['aleatoric']).flatten()
    total = np.array(data['total_uncertainty']).flatten()

    n = len(true_values)
    time = np.arange(n)

    fig, ax = plt.subplots(figsize=(14, 6))

    # True values
    ax.plot(time, true_values, 'k-', label='True', alpha=0.7, linewidth=1)

    # Predictions
    ax.plot(time, predictions, 'b-', label='MC Dropout Prediction', linewidth=1.5)

    # Uncertainty bands
    ax.fill_between(time,
                     predictions - 0.67 * total,
                     predictions + 0.67 * total,
                     alpha=0.3, color='blue', label='50% CI (Total)')

    ax.fill_between(time,
                     predictions - 1.96 * total,
                     predictions + 1.96 * total,
                     alpha=0.15, color='blue', label='95% CI (Total)')

    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Log Return', fontsize=12)
    ax.set_title(f'{ticker}: MC Dropout LSTM with Uncertainty', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / f'{ticker}_mc_dropout_predictions.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved {ticker} predictions plot")


def plot_uncertainty_decomposition(ticker, data, save_dir):
    """Plot epistemic vs aleatoric uncertainty"""
    epistemic = np.array(data['epistemic']).flatten()
    aleatoric = np.array(data['aleatoric']).flatten()

    n = len(epistemic)
    time = np.arange(n)

    fig, ax = plt.subplots(figsize=(14, 6))

    # Stacked area plot
    ax.fill_between(time, 0, aleatoric, alpha=0.5, color='orange', label='Aleatoric (Data Noise)')
    ax.fill_between(time, aleatoric, aleatoric + epistemic, alpha=0.5, color='blue', label='Epistemic (Model)')

    # Total line
    total = np.sqrt(epistemic**2 + aleatoric**2)
    ax.plot(time, total, 'k-', linewidth=2, label='Total', alpha=0.7)

    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Uncertainty (Std Dev)', fontsize=12)
    ax.set_title(f'{ticker}: Uncertainty Decomposition', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / f'{ticker}_uncertainty_decomp.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved {ticker} uncertainty decomposition plot")


def main():
    print('='*80)
    print('GENERATING MC DROPOUT PLOTS')
    print('='*80)

    data_path = Path('results/predictions/all_mc_dropout.json')

    if not data_path.exists():
        print(f"ERROR: {data_path} not found!")
        print("Run experiments/run_mc_dropout.py first")
        return

    with open(data_path) as f:
        all_data = json.load(f)

    save_dir = Path('results/figures/mc_dropout')
    save_dir.mkdir(parents=True, exist_ok=True)

    for ticker in all_data.keys():
        print(f"\nGenerating plots for {ticker}...")
        plot_predictions_with_uncertainty(ticker, all_data[ticker], save_dir)
        plot_uncertainty_decomposition(ticker, all_data[ticker], save_dir)

    print(f"\n{'='*80}")
    print('✓ All MC Dropout plots generated')
    print(f'  Output: {save_dir}')
    print('='*80)


if __name__ == '__main__':
    main()
