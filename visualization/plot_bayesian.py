"""Plot Bayesian LSTM results"""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import numpy as np
from visualization.plots import (
    plot_predictions_with_uncertainty,
    plot_uncertainty_decomposition,
    plot_uncertainty_vs_error,
    plot_vix_vs_prior
)

def main():
    pred_dir = Path('results/predictions')
    fig_dir = Path('results/figures')
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    with open(pred_dir / 'all_bayesian.json', 'r') as f:
        results = json.load(f)

    # Plot each ticker
    for key, data in results.items():
        ticker = data['ticker']
        print(f"\nPlotting {ticker}...")

        # Extract data
        y_true = np.array(data['targets'])
        y_pred = np.array(data['predictions'])
        total_std = np.array(data['total_std'])
        epistemic = np.array(data['epistemic_std'])
        aleatoric = np.array(data['aleatoric_std'])
        vix = np.array(data['vix'])

        dates = np.arange(len(y_true))
        errors = np.abs(y_true - y_pred)

        # Plot 1: Predictions (full)
        plot_predictions_with_uncertainty(
            dates, y_true, y_pred, total_std,
            title=f'{ticker}: Bayesian LSTM Predictions with Uncertainty',
            save_path=fig_dir / f'{ticker}_predictions.png'
        )

        # Plot 2: Predictions (zoom first 100)
        plot_predictions_with_uncertainty(
            dates[:100], y_true[:100], y_pred[:100], total_std[:100],
            title=f'{ticker}: Predictions (First 100 Days)',
            save_path=fig_dir / f'{ticker}_predictions_zoom.png'
        )

        # Plot 3: Uncertainty decomposition
        plot_uncertainty_decomposition(
            dates, epistemic, aleatoric,
            title=f'{ticker}: Uncertainty Decomposition',
            save_path=fig_dir / f'{ticker}_uncertainty_decomp.png'
        )

        # Plot 4: Uncertainty vs Error
        plot_uncertainty_vs_error(
            total_std, errors,
            title=f'{ticker}: Uncertainty vs Error',
            save_path=fig_dir / f'{ticker}_unc_vs_error.png'
        )

        # Plot 5: VIX vs Prior (compute prior_std from vix)
        from models.prior import AdaptivePrior
        from training.config import Config
        config = Config()
        prior = AdaptivePrior(
            base_std=config.PRIOR_BASE_STD,
            vix_mean=config.VIX_MEAN,
            vix_scale=config.VIX_SCALE,
            sensitivity=config.VIX_SENSITIVITY
        )
        prior_stds = np.array([prior.get_prior_std(v).item() for v in vix.flatten()])

        plot_vix_vs_prior(
            dates, vix, prior_stds.reshape(-1, 1),
            save_path=fig_dir / f'{ticker}_vix_prior.png'
        )

        print(f"  ✓ Saved 5 plots for {ticker}")

    print(f"\nAll plots saved to {fig_dir}/")

if __name__ == '__main__':
    main()
