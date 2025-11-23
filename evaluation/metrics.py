"""
Evaluation Metrics Module

This module implements various metrics for evaluating forecast performance including:
- Point prediction metrics (MSE, MAE, RMSE, MAPE)
- Directional accuracy metrics
- Financial metrics (Sharpe ratio, returns-based evaluation)
"""

import numpy as np
from scipy import stats


def compute_rmse(y_true, y_pred):
    """Root Mean Squared Error"""
    return np.sqrt(np.mean((y_true - y_pred)**2))


def compute_mae(y_true, y_pred):
    """Mean Absolute Error"""
    return np.mean(np.abs(y_true - y_pred))


def compute_mse(y_true, y_pred):
    """Mean Squared Error"""
    return np.mean((y_true - y_pred)**2)


def compute_crps(y_true, pred_mean, pred_std):
    """
    Continuous Ranked Probability Score

    For Gaussian predictions: N(μ, σ²)
    CRPS = σ * [1/√π - 2φ((y-μ)/σ) - (y-μ)/σ * (2Φ((y-μ)/σ) - 1)]

    where φ is PDF, Φ is CDF of standard normal

    Lower is better (measures both sharpness and calibration)
    """
    z = (y_true - pred_mean) / pred_std  # Standardized error

    phi = stats.norm.pdf(z)  # PDF
    Phi = stats.norm.cdf(z)  # CDF

    crps = pred_std * (1/np.sqrt(np.pi) - 2*phi - z*(2*Phi - 1))

    return np.mean(crps)


def compute_nll(y_true, pred_mean, pred_std):
    """
    Negative Log-Likelihood for Gaussian predictions

    -log N(y | μ, σ²) = 0.5 * ((y-μ)/σ)² + log σ + 0.5*log(2π)
    """
    nll = 0.5 * ((y_true - pred_mean) / pred_std)**2 + \
          np.log(pred_std) + \
          0.5 * np.log(2 * np.pi)

    return np.mean(nll)


def compute_uncertainty_error_correlation(pred_std, errors):
    """
    Correlation between predicted uncertainty and actual errors

    High positive correlation (>0.4) means model knows when it's uncertain

    Args:
        pred_std: Predicted uncertainty [N, 1]
        errors: Absolute errors |y_true - y_pred| [N, 1]

    Returns:
        correlation: Pearson correlation coefficient
    """
    corr = np.corrcoef(pred_std.flatten(), errors.flatten())[0, 1]
    return corr


def compute_all_metrics(y_true, pred_mean, pred_std=None):
    """
    Compute all available metrics

    Args:
        y_true: Ground truth [N, 1]
        pred_mean: Predictions [N, 1]
        pred_std: Uncertainties [N, 1] (optional)

    Returns:
        dict: All computed metrics
    """
    metrics = {
        'rmse': compute_rmse(y_true, pred_mean),
        'mae': compute_mae(y_true, pred_mean),
        'mse': compute_mse(y_true, pred_mean)
    }

    if pred_std is not None:
        errors = np.abs(y_true - pred_mean)

        metrics['crps'] = compute_crps(y_true, pred_mean, pred_std)
        metrics['nll'] = compute_nll(y_true, pred_mean, pred_std)
        metrics['uncertainty_error_corr'] = compute_uncertainty_error_correlation(pred_std, errors)

        # Uncertainty statistics
        metrics['mean_uncertainty'] = np.mean(pred_std)
        metrics['std_uncertainty'] = np.std(pred_std)

    return metrics


def main():
    """Test metrics"""
    print("="*80)
    print("Testing Evaluation Metrics")
    print("="*80)

    # Dummy data
    np.random.seed(42)
    n = 100
    y_true = np.random.randn(n, 1) * 0.02
    y_pred = y_true + np.random.randn(n, 1) * 0.01
    pred_std = np.abs(np.random.randn(n, 1) * 0.015) + 0.005

    # Compute metrics
    metrics = compute_all_metrics(y_true, y_pred, pred_std)

    print("\nMetrics:")
    for key, value in metrics.items():
        print(f"  {key:25s}: {value:.6f}")

    print("\n" + "="*80)
    print("Metrics test complete!")
    print("="*80)


if __name__ == '__main__':
    main()
