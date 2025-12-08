"""
Uncertainty Quantification Analysis Module

This module provides utilities for analyzing and visualizing uncertainty in predictions.
Includes functions for computing prediction intervals, analyzing epistemic vs aleatoric
uncertainty, and evaluating uncertainty quality through proper scoring rules.
"""

import numpy as np


def decompose_uncertainty(epistemic_std, aleatoric_std):
    """
    Verify uncertainty decomposition

    Total^2 = Epistemic^2 + Aleatoric^2

    Args:
        epistemic_std: Model uncertainty [N, 1]
        aleatoric_std: Data uncertainty [N, 1]

    Returns:
        dict: Decomposed uncertainties with statistics
    """
    total_std = np.sqrt(epistemic_std**2 + aleatoric_std**2)

    # Ratios
    epistemic_ratio = np.mean(epistemic_std) / np.mean(total_std)
    aleatoric_ratio = np.mean(aleatoric_std) / np.mean(total_std)

    return {
        'epistemic_mean': np.mean(epistemic_std),
        'epistemic_std': np.std(epistemic_std),
        'aleatoric_mean': np.mean(aleatoric_std),
        'aleatoric_std': np.std(aleatoric_std),
        'total_mean': np.mean(total_std),
        'total_std': np.std(total_std),
        'epistemic_ratio': epistemic_ratio,
        'aleatoric_ratio': aleatoric_ratio
    }


def stratify_by_uncertainty(y_true, pred_mean, pred_std, n_bins=5):
    """
    Stratify predictions by uncertainty level

    Check if high uncertainty → high error

    Args:
        y_true: Ground truth [N, 1]
        pred_mean: Predictions [N, 1]
        pred_std: Uncertainties [N, 1]
        n_bins: Number of uncertainty bins

    Returns:
        list of dicts: Metrics for each uncertainty bin
    """
    errors = np.abs(y_true - pred_mean)

    # Create bins based on uncertainty quantiles
    bin_edges = np.percentile(pred_std, np.linspace(0, 100, n_bins+1))

    results = []

    for i in range(n_bins):
        # Get samples in this bin
        mask = (pred_std >= bin_edges[i]) & (pred_std < bin_edges[i+1])

        if i == n_bins - 1:  # Include upper edge in last bin
            mask = (pred_std >= bin_edges[i]) & (pred_std <= bin_edges[i+1])

        if np.sum(mask) == 0:
            continue

        bin_data = {
            'bin': i+1,
            'uncertainty_range': (float(bin_edges[i]), float(bin_edges[i+1])),
            'n_samples': int(np.sum(mask)),
            'mean_uncertainty': float(np.mean(pred_std[mask])),
            'mean_error': float(np.mean(errors[mask])),
            'rmse': float(np.sqrt(np.mean(errors[mask]**2)))
        }

        results.append(bin_data)

    return results


def stratify_by_vix(y_true, pred_mean, pred_std, vix_values,
                    vix_bins=[0, 15, 25, 35, 100]):
    """
    Stratify by market volatility (VIX)

    Check if model adapts uncertainty to market conditions

    Args:
        y_true: Ground truth [N, 1]
        pred_mean: Predictions [N, 1]
        pred_std: Uncertainties [N, 1]
        vix_values: VIX at each time [N, 1]
        vix_bins: VIX bin edges (default: Low, Normal, High, Crisis)

    Returns:
        list of dicts: Metrics for each VIX regime
    """
    errors = np.abs(y_true - pred_mean)

    regime_names = ['Low', 'Normal', 'High', 'Crisis']
    results = []

    for i in range(len(vix_bins)-1):
        mask = (vix_values >= vix_bins[i]) & (vix_values < vix_bins[i+1])

        if np.sum(mask) == 0:
            continue

        regime_data = {
            'regime': regime_names[i] if i < len(regime_names) else f'Regime {i+1}',
            'vix_range': (vix_bins[i], vix_bins[i+1]),
            'n_samples': int(np.sum(mask)),
            'mean_vix': float(np.mean(vix_values[mask])),
            'mean_uncertainty': float(np.mean(pred_std[mask])),
            'mean_error': float(np.mean(errors[mask])),
            'rmse': float(np.sqrt(np.mean(errors[mask]**2)))
        }

        results.append(regime_data)

    return results


def main():
    """Test uncertainty analysis"""
    print("="*80)
    print("Testing Uncertainty Analysis")
    print("="*80)

    # Dummy data
    np.random.seed(42)
    n = 500
    y_true = np.random.randn(n, 1) * 0.02
    y_pred = y_true + np.random.randn(n, 1) * 0.01

    epistemic = np.abs(np.random.randn(n, 1) * 0.01)
    aleatoric = np.abs(np.random.randn(n, 1) * 0.015)
    total_std = np.sqrt(epistemic**2 + aleatoric**2)

    vix = np.random.uniform(10, 40, (n, 1))

    # Decomposition
    print("\nUncertainty Decomposition:")
    decomp = decompose_uncertainty(epistemic, aleatoric)
    for key, value in decomp.items():
        print(f"  {key:20s}: {value:.6f}")

    # Stratify by uncertainty
    print("\n" + "="*80)
    print("Stratification by Uncertainty Level")
    print("="*80)
    unc_strat = stratify_by_uncertainty(y_true, y_pred, total_std, n_bins=5)

    print(f"{'Bin':<5} {'Unc Range':<25} {'N':<8} {'Mean Unc':<12} {'Mean Err':<12} {'RMSE':<12}")
    print("-"*80)
    for bin_data in unc_strat:
        unc_range = f"[{bin_data['uncertainty_range'][0]:.4f}, {bin_data['uncertainty_range'][1]:.4f}]"
        print(f"{bin_data['bin']:<5} {unc_range:<25} {bin_data['n_samples']:<8} "
              f"{bin_data['mean_uncertainty']:<12.6f} {bin_data['mean_error']:<12.6f} "
              f"{bin_data['rmse']:<12.6f}")

    # Stratify by VIX
    print("\n" + "="*80)
    print("Stratification by VIX Regime")
    print("="*80)
    vix_strat = stratify_by_vix(y_true, y_pred, total_std, vix)

    print(f"{'Regime':<10} {'VIX Range':<15} {'N':<8} {'Mean Unc':<12} {'Mean Err':<12} {'RMSE':<12}")
    print("-"*80)
    for regime_data in vix_strat:
        vix_range = f"[{regime_data['vix_range'][0]:.0f}, {regime_data['vix_range'][1]:.0f}]"
        print(f"{regime_data['regime']:<10} {vix_range:<15} {regime_data['n_samples']:<8} "
              f"{regime_data['mean_uncertainty']:<12.6f} {regime_data['mean_error']:<12.6f} "
              f"{regime_data['rmse']:<12.6f}")

    print("\n" + "="*80)
    print("Uncertainty analysis complete!")
    print("="*80)


if __name__ == '__main__':
    main()
