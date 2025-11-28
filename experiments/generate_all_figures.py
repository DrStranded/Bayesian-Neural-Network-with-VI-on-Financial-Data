import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import numpy as np
import matplotlib.pyplot as plt

from evaluation.metrics import compute_all_metrics
from evaluation.calibration import compute_calibration_curve, compute_calibration_error
from evaluation.uncertainty import stratify_by_vix, stratify_by_uncertainty
from visualization.plots import (
    plot_predictions_with_uncertainty,
    plot_calibration_curve,
    plot_uncertainty_decomposition,
    plot_uncertainty_vs_error,
    plot_vix_vs_prior,
    plot_regime_comparison,
    plot_model_comparison
)
from training.config import Config
from models.prior import AdaptivePrior


def load_results(results_dir):
    """Load all saved results"""
    results_dir = Path(results_dir)

    # Load baseline results
    baseline_path = results_dir / 'all_baselines.json'
    with open(baseline_path, 'r') as f:
        baseline_results = json.load(f)

    # Load Bayesian results
    bayesian_path = results_dir / 'all_bayesian.json'
    with open(bayesian_path, 'r') as f:
        bayesian_results = json.load(f)

    return baseline_results, bayesian_results


def generate_prediction_plots(bayesian_results, config, output_dir):
    """Generate prediction plots with uncertainty for each ticker"""
    print("\nGenerating prediction plots...")

    for ticker in config.TICKERS:
        key = f"{ticker}_bayesian_lstm"
        if key not in bayesian_results:
            continue

        result = bayesian_results[key]

        # Extract data
        predictions = np.array(result['predictions'])
        targets = np.array(result['targets'])
        total_std = np.array(result['total_std'])

        # Create time indices
        dates = np.arange(len(predictions))

        # Plot
        plot_predictions_with_uncertainty(
            dates, targets, predictions, total_std,
            title=f'{ticker}: Bayesian LSTM Predictions with Uncertainty',
            save_path=output_dir / f'{ticker}_predictions.png'
        )

        # Plot a zoomed-in version (first 100 days)
        if len(dates) > 100:
            plot_predictions_with_uncertainty(
                dates[:100], targets[:100], predictions[:100], total_std[:100],
                title=f'{ticker}: Predictions (First 100 Days)',
                save_path=output_dir / f'{ticker}_predictions_zoom.png'
            )


def generate_calibration_plots(bayesian_results, config, output_dir):
    """Generate calibration curves"""
    print("\nGenerating calibration plots...")

    all_calibration_data = []
    model_names = []

    for ticker in config.TICKERS:
        key = f"{ticker}_bayesian_lstm"
        if key not in bayesian_results:
            continue

        result = bayesian_results[key]

        predictions = np.array(result['predictions']).reshape(-1, 1)
        targets = np.array(result['targets']).reshape(-1, 1)
        total_std = np.array(result['total_std']).reshape(-1, 1)

        # Compute calibration
        cal_data = compute_calibration_curve(targets, predictions, total_std)
        all_calibration_data.append(cal_data)
        model_names.append(f'{ticker}')

        # Individual plot
        plot_calibration_curve(
            cal_data,
            model_names=[f'{ticker} Bayesian LSTM'],
            save_path=output_dir / f'{ticker}_calibration.png'
        )

        # Print calibration error
        ece = compute_calibration_error(cal_data)
        print(f"  {ticker} ECE: {ece:.4f}")

    # Combined plot
    if all_calibration_data:
        plot_calibration_curve(
            all_calibration_data,
            model_names=model_names,
            save_path=output_dir / 'all_calibration.png'
        )


def generate_uncertainty_plots(bayesian_results, config, output_dir):
    """Generate uncertainty decomposition and analysis plots"""
    print("\nGenerating uncertainty analysis plots...")

    for ticker in config.TICKERS:
        key = f"{ticker}_bayesian_lstm"
        if key not in bayesian_results:
            continue

        result = bayesian_results[key]

        predictions = np.array(result['predictions']).reshape(-1, 1)
        targets = np.array(result['targets']).reshape(-1, 1)
        total_std = np.array(result['total_std']).reshape(-1, 1)
        epistemic_std = np.array(result['epistemic_std']).reshape(-1, 1)
        aleatoric_std = np.array(result['aleatoric_std']).reshape(-1, 1)

        dates = np.arange(len(predictions))
        errors = np.abs(predictions - targets)

        # Uncertainty decomposition
        plot_uncertainty_decomposition(
            dates, epistemic_std, aleatoric_std,
            title=f'{ticker}: Uncertainty Decomposition',
            save_path=output_dir / f'{ticker}_uncertainty_decomp.png'
        )

        # Uncertainty vs Error
        plot_uncertainty_vs_error(
            total_std, errors,
            title=f'{ticker}: Uncertainty vs Error',
            save_path=output_dir / f'{ticker}_unc_vs_error.png'
        )


def generate_vix_prior_plots(bayesian_results, config, output_dir):
    """Generate VIX vs Prior plots"""
    print("\nGenerating VIX-Prior adaptation plots...")

    # Create prior module
    prior = AdaptivePrior(
        base_std=config.PRIOR_BASE_STD,
        vix_mean=config.VIX_MEAN,
        vix_scale=config.VIX_SCALE,
        sensitivity=config.VIX_SENSITIVITY
    )

    for ticker in config.TICKERS:
        key = f"{ticker}_bayesian_lstm"
        if key not in bayesian_results:
            continue

        result = bayesian_results[key]

        vix_values = np.array(result['vix']).reshape(-1, 1)
        dates = np.arange(len(vix_values))

        # Compute prior std for each VIX value
        prior_stds = np.array([
            prior.get_prior_std(vix).item()
            for vix in vix_values.flatten()
        ]).reshape(-1, 1)

        # Plot
        plot_vix_vs_prior(
            dates, vix_values, prior_stds,
            save_path=output_dir / f'{ticker}_vix_prior.png'
        )


def generate_regime_plots(bayesian_results, config, output_dir):
    """Generate regime-specific performance plots"""
    print("\nGenerating regime analysis plots...")

    for ticker in config.TICKERS:
        key = f"{ticker}_bayesian_lstm"
        if key not in bayesian_results:
            continue

        result = bayesian_results[key]

        predictions = np.array(result['predictions']).reshape(-1, 1)
        targets = np.array(result['targets']).reshape(-1, 1)
        total_std = np.array(result['total_std']).reshape(-1, 1)
        vix_values = np.array(result['vix']).reshape(-1, 1)

        # Stratify by VIX
        regime_data = stratify_by_vix(targets, predictions, total_std, vix_values)

        # Plot
        plot_regime_comparison(
            regime_data,
            save_path=output_dir / f'{ticker}_regime_comparison.png'
        )

        # Print table
        print(f"\n{ticker} - Performance by VIX Regime:")
        print(f"{'Regime':<10} {'VIX Range':<15} {'N':<8} {'RMSE':<12} {'Mean Unc':<12}")
        print("-" * 60)
        for r in regime_data:
            vix_range = f"[{r['vix_range'][0]:.0f}, {r['vix_range'][1]:.0f}]"
            print(f"{r['regime']:<10} {vix_range:<15} {r['n_samples']:<8} "
                  f"{r['rmse']:<12.6f} {r['mean_uncertainty']:<12.6f}")


def generate_comparison_table(baseline_results, bayesian_results, config, output_dir):
    """Generate model comparison table"""
    print("\nGenerating model comparison table...")

    plot_model_comparison(
        baseline_results,
        bayesian_results,
        save_path=output_dir / 'model_comparison.png'
    )


def generate_summary_report(baseline_results, bayesian_results, config, output_dir):
    """Generate text summary report"""
    print("\nGenerating summary report...")

    report_path = output_dir / 'summary_report.txt'

    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("BAYESIAN LSTM STOCK FORECASTING - FINAL RESULTS\n")
        f.write("="*80 + "\n\n")

        # Overall comparison
        f.write("Model Performance Summary\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Ticker':<10} {'Model':<15} {'RMSE':<12} {'MAE':<12} {'Unc-Err Corr':<15}\n")
        f.write("-"*80 + "\n")

        for ticker in config.TICKERS:
            # MA
            ma_key = f"{ticker}_ma"
            if ma_key in baseline_results:
                ma = baseline_results[ma_key]['metrics']
                f.write(f"{ticker:<10} {'MA':<15} {ma['rmse']:<12.6f} {ma['mae']:<12.6f} {'N/A':<15}\n")

            # LSTM
            lstm_key = f"{ticker}_standard_lstm"
            if lstm_key in baseline_results:
                lstm = baseline_results[lstm_key]['metrics']
                f.write(f"{'':10} {'LSTM':<15} {lstm['rmse']:<12.6f} {lstm['mae']:<12.6f} {'N/A':<15}\n")

            # Bayesian
            bayes_key = f"{ticker}_bayesian_lstm"
            if bayes_key in bayesian_results:
                bayes = bayesian_results[bayes_key]['metrics']
                corr = bayes.get('uncertainty_error_corr', 0)
                f.write(f"{'':10} {'Bayesian':<15} {bayes['rmse']:<12.6f} {bayes['mae']:<12.6f} {corr:<15.4f}\n")

            f.write("\n")

        # Uncertainty statistics
        f.write("\n" + "="*80 + "\n")
        f.write("Uncertainty Analysis\n")
        f.write("="*80 + "\n\n")

        for ticker in config.TICKERS:
            bayes_key = f"{ticker}_bayesian_lstm"
            if bayes_key not in bayesian_results:
                continue

            result = bayesian_results[bayes_key]
            metrics = result['metrics']

            f.write(f"{ticker}:\n")
            f.write(f"  Average Epistemic:  {metrics['avg_epistemic']:.6f}\n")
            f.write(f"  Average Aleatoric:  {metrics['avg_aleatoric']:.6f}\n")
            f.write(f"  Average Total:      {metrics['avg_total_std']:.6f}\n")
            f.write(f"  Unc-Error Corr:     {metrics['uncertainty_error_corr']:.4f}\n\n")

        # Configuration
        f.write("\n" + "="*80 + "\n")
        f.write("Model Configuration\n")
        f.write("="*80 + "\n")
        f.write(f"Beta (KL weight):        0.001\n")
        f.write(f"Prior base std:          {config.PRIOR_BASE_STD}\n")
        f.write(f"VIX mean:                {config.VIX_MEAN}\n")
        f.write(f"VIX scale:               {config.VIX_SCALE}\n")
        f.write(f"VIX sensitivity:         {config.VIX_SENSITIVITY}\n")
        f.write(f"Hidden size:             {config.HIDDEN_SIZE}\n")
        f.write(f"Sequence length:         {config.SEQ_LEN}\n")
        f.write(f"Test samples (MC):       {config.N_SAMPLES_TEST}\n")

    print(f"  Summary report saved to: {report_path}")


def main():
    """Generate all figures and reports"""
    config = Config()

    print("="*80)
    print("GENERATING ALL FIGURES AND REPORTS")
    print("="*80)
    print(f"\nTickers: {config.TICKERS}")

    # Create output directory
    output_dir = Path(config.FIG_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load results
    print("\nLoading results...")
    try:
        baseline_results, bayesian_results = load_results(config.PRED_DIR)
        print(f"  Loaded {len(baseline_results)} baseline results")
        print(f"  Loaded {len(bayesian_results)} Bayesian results")
    except FileNotFoundError as e:
        print(f"\nERROR: Could not find results files!")
        print(f"  {e}")
        print("\nPlease run the experiments first:")
        print("  1. python experiments/run_baselines.py")
        print("  2. python experiments/run_bayesian.py")
        return

    # Generate all plots
    try:
        generate_prediction_plots(bayesian_results, config, output_dir)
        generate_calibration_plots(bayesian_results, config, output_dir)
        generate_uncertainty_plots(bayesian_results, config, output_dir)
        generate_vix_prior_plots(bayesian_results, config, output_dir)
        generate_regime_plots(bayesian_results, config, output_dir)
        generate_comparison_table(baseline_results, bayesian_results, config, output_dir)
        generate_summary_report(baseline_results, bayesian_results, config, output_dir)
    except Exception as e:
        print(f"\nERROR: Failed to generate figures!")
        print(f"  {e}")
        import traceback
        traceback.print_exc()
        return

    # Final summary
    print("\n" + "="*80)
    print("ALL FIGURES GENERATED!")
    print("="*80)
    print(f"\nTotal figures saved to: {output_dir}")

    # List all generated files
    figures = sorted(output_dir.glob('*.png'))
    print(f"\nGenerated {len(figures)} figures:")
    for fig in figures:
        print(f"  - {fig.name}")

    # Summary report
    report_path = output_dir / 'summary_report.txt'
    if report_path.exists():
        print(f"\nSummary report: {report_path}")

    print("\n" + "="*80)
    print("READY FOR REPORT AND PRESENTATION!")
    print("="*80)


if __name__ == '__main__':
    main()
