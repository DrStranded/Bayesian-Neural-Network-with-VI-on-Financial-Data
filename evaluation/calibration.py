"""
Calibration Analysis Module

This module provides tools for assessing the calibration of probabilistic forecasts.
Implements calibration plots, reliability diagrams, and statistical tests to verify
that predicted uncertainty intervals have proper coverage.
"""

import numpy as np
from scipy import stats


def compute_calibration_curve(y_true, pred_mean, pred_std,
                              confidence_levels=[0.5, 0.68, 0.9, 0.95]):
    """
    Compute calibration curve: expected vs actual coverage

    For a well-calibrated model:
    - 50% confidence interval should contain 50% of true values
    - 90% confidence interval should contain 90% of true values

    Args:
        y_true: Ground truth [N, 1]
        pred_mean: Predictions [N, 1]
        pred_std: Uncertainties [N, 1]
        confidence_levels: List of confidence levels to check

    Returns:
        dict: {
            'confidence_levels': list,
            'expected_coverage': list,
            'actual_coverage': list
        }
    """
    expected = []
    actual = []

    for conf in confidence_levels:
        # Z-score for confidence level (two-tailed)
        z_score = stats.norm.ppf((1 + conf) / 2)

        # Prediction intervals
        lower = pred_mean - z_score * pred_std
        upper = pred_mean + z_score * pred_std

        # Check coverage
        in_interval = (y_true >= lower) & (y_true <= upper)
        actual_coverage = np.mean(in_interval)

        expected.append(conf)
        actual.append(actual_coverage)

    return {
        'confidence_levels': confidence_levels,
        'expected_coverage': expected,
        'actual_coverage': actual
    }


def compute_calibration_error(calibration_data):
    """
    Expected Calibration Error (ECE)

    ECE = mean(|expected - actual|)

    Good models: ECE < 0.05

    Args:
        calibration_data: Output from compute_calibration_curve

    Returns:
        ece: Expected Calibration Error
    """
    expected = np.array(calibration_data['expected_coverage'])
    actual = np.array(calibration_data['actual_coverage'])

    ece = np.mean(np.abs(expected - actual))

    return ece


def compute_sharpness(pred_std, confidence=0.95):
    """
    Sharpness: Average width of prediction intervals

    Lower is better (but must maintain calibration)

    Args:
        pred_std: Predicted uncertainties [N, 1]
        confidence: Confidence level (default 0.95)

    Returns:
        sharpness: Mean interval width
    """
    z_score = stats.norm.ppf((1 + confidence) / 2)
    interval_width = 2 * z_score * pred_std

    sharpness = np.mean(interval_width)

    return sharpness


def compute_prediction_interval_coverage(y_true, pred_mean, pred_std, confidence=0.95):
    """
    Prediction Interval Coverage Probability (PICP)

    Fraction of true values within confidence interval

    Args:
        y_true: Ground truth [N, 1]
        pred_mean: Predictions [N, 1]
        pred_std: Uncertainties [N, 1]
        confidence: Confidence level

    Returns:
        picp: Coverage probability
    """
    z_score = stats.norm.ppf((1 + confidence) / 2)

    lower = pred_mean - z_score * pred_std
    upper = pred_mean + z_score * pred_std

    in_interval = (y_true >= lower) & (y_true <= upper)
    picp = np.mean(in_interval)

    return picp


def compute_interval_score(y_true, pred_mean, pred_std, alpha=0.05):
    """
    Interval Score (Gneiting & Raftery 2007)

    Combines sharpness and calibration
    Lower is better

    Args:
        y_true: Ground truth [N, 1]
        pred_mean: Predictions [N, 1]
        pred_std: Uncertainties [N, 1]
        alpha: Miscoverage level (1-confidence)

    Returns:
        interval_score: Mean interval score
    """
    z = stats.norm.ppf(1 - alpha/2)

    lower = pred_mean - z * pred_std
    upper = pred_mean + z * pred_std

    width = upper - lower

    # Penalty for being outside interval
    penalty_lower = (2/alpha) * (lower - y_true) * (y_true < lower)
    penalty_upper = (2/alpha) * (y_true - upper) * (y_true > upper)

    score = width + penalty_lower + penalty_upper

    return np.mean(score)


def main():
    """Test calibration metrics"""
    print("="*80)
    print("Testing Calibration Metrics")
    print("="*80)

    # Dummy well-calibrated data
    np.random.seed(42)
    n = 1000
    y_true = np.random.randn(n, 1) * 0.02
    y_pred = y_true + np.random.randn(n, 1) * 0.005  # Small error
    pred_std = np.ones((n, 1)) * 0.015  # Constant uncertainty

    # Calibration curve
    print("\nCalibration Curve:")
    cal_data = compute_calibration_curve(y_true, y_pred, pred_std)

    print(f"{'Confidence':<15} {'Expected':<15} {'Actual':<15} {'Diff':<15}")
    print("-"*60)
    for conf, exp, act in zip(cal_data['confidence_levels'],
                              cal_data['expected_coverage'],
                              cal_data['actual_coverage']):
        diff = abs(exp - act)
        print(f"{conf:<15.2f} {exp:<15.2f} {act:<15.4f} {diff:<15.4f}")

    # Calibration error
    ece = compute_calibration_error(cal_data)
    print(f"\nExpected Calibration Error: {ece:.4f}")
    if ece < 0.05:
        print("  → Well calibrated! ✓")
    else:
        print("  → Poor calibration")

    # Sharpness
    sharpness = compute_sharpness(pred_std)
    print(f"\nSharpness (95% interval width): {sharpness:.6f}")

    # PICP
    picp = compute_prediction_interval_coverage(y_true, y_pred, pred_std, confidence=0.95)
    print(f"Prediction Interval Coverage (95%): {picp:.4f}")

    # Interval score
    interval_score = compute_interval_score(y_true, y_pred, pred_std)
    print(f"Interval Score: {interval_score:.6f}")

    print("\n" + "="*80)
    print("Calibration test complete!")
    print("="*80)


if __name__ == '__main__':
    main()
