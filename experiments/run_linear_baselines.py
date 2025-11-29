"""
Linear Baselines: MA and Bayesian Linear Regression
Separate from LSTM models for clean comparison
"""

import sys
from pathlib import Path
import numpy as np
import json
from sklearn.linear_model import BayesianRidge

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from training.config import Config
from evaluation.metrics import compute_rmse, compute_mae

def moving_average_baseline(y_train, y_test, window=20):
    """
    Moving Average baseline: predict mean of last N values

    Args:
        y_train: [N_train, 1] training log returns
        y_test: [N_test, 1] test log returns
        window: window size for MA

    Returns:
        dict with predictions and metrics
    """
    predictions = []

    # Concatenate train and test for rolling window
    all_returns = np.concatenate([y_train, y_test], axis=0).flatten()

    # Start predictions after training set
    start_idx = len(y_train)

    for i in range(len(y_test)):
        # Use past 'window' values
        hist_idx = start_idx + i
        window_data = all_returns[hist_idx - window : hist_idx]
        pred = np.mean(window_data)
        predictions.append(pred)

    predictions = np.array(predictions).reshape(-1, 1)

    # Compute metrics
    rmse = compute_rmse(y_test, predictions)
    mae = compute_mae(y_test, predictions)

    return {
        'predictions': predictions,
        'metrics': {'rmse': rmse, 'mae': mae}
    }

def bayesian_linear_regression(X_train, y_train, X_test, y_test):
    """
    Bayesian Linear Regression baseline

    Args:
        X_train: [N_train, seq_len, n_features] - we'll use last timestep only
        y_train: [N_train, 1]
        X_test: [N_test, seq_len, n_features]
        y_test: [N_test, 1]

    Returns:
        dict with predictions, uncertainty, metrics, and coefficients
    """
    # Extract features from last timestep
    # Shape: [N, seq_len, features] -> [N, features]
    X_train_flat = X_train[:, -1, :]  # Last day's features
    X_test_flat = X_test[:, -1, :]

    y_train_flat = y_train.flatten()
    y_test_flat = y_test.flatten()

    # Train Bayesian Ridge Regression
    model = BayesianRidge(
        alpha_1=1e-6,  # Gamma prior on alpha (precision of weights)
        alpha_2=1e-6,
        lambda_1=1e-6, # Gamma prior on lambda (precision of noise)
        lambda_2=1e-6,
        compute_score=True,
        fit_intercept=True
    )

    model.fit(X_train_flat, y_train_flat)

    # Predict with uncertainty
    y_pred, y_std = model.predict(X_test_flat, return_std=True)

    y_pred = y_pred.reshape(-1, 1)
    y_std = y_std.reshape(-1, 1)

    # Compute metrics
    rmse = compute_rmse(y_test, y_pred)
    mae = compute_mae(y_test, y_pred)

    # Uncertainty-error correlation
    errors = np.abs(y_test.flatten() - y_pred.flatten())
    corr = np.corrcoef(y_std.flatten(), errors)[0, 1]

    return {
        'predictions': y_pred,
        'uncertainty': y_std,
        'coefficients': model.coef_,
        'intercept': model.intercept_,
        'metrics': {
            'rmse': rmse,
            'mae': mae,
            'uncertainty_error_corr': corr
        }
    }

def run_linear_baselines_for_ticker(ticker, config):
    """Run MA and BLR for one ticker"""
    print(f"\n{'='*80}")
    print(f"Running Linear Baselines: {ticker}")
    print('='*80)

    # Load data
    data_path = Path(config.DATA_DIR) / f'{ticker}_processed.npz'
    data = np.load(data_path)

    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    vix_test = data['vix_test']

    print(f"Data loaded: X_train {X_train.shape}, X_test {X_test.shape}")

    # 1. Moving Average
    print("\n1. Moving Average (window=20)")
    ma_results = moving_average_baseline(y_train, y_test, window=20)
    print(f"   RMSE: {ma_results['metrics']['rmse']:.6f}")
    print(f"   MAE:  {ma_results['metrics']['mae']:.6f}")

    # 2. Bayesian Linear Regression
    print("\n2. Bayesian Linear Regression")
    blr_results = bayesian_linear_regression(X_train, y_train, X_test, y_test)
    print(f"   RMSE: {blr_results['metrics']['rmse']:.6f}")
    print(f"   MAE:  {blr_results['metrics']['mae']:.6f}")
    print(f"   Unc-Err Corr: {blr_results['metrics']['uncertainty_error_corr']:.4f}")
    print(f"   Coefficients: {blr_results['coefficients']}")
    print(f"   Intercept: {blr_results['intercept']:.6f}")

    # Save results
    results_dir = Path('results/predictions/linear_baselines')
    results_dir.mkdir(parents=True, exist_ok=True)

    save_data = {
        'ticker': ticker,
        'ma': {
            'predictions': ma_results['predictions'].tolist(),
            'metrics': ma_results['metrics']
        },
        'blr': {
            'predictions': blr_results['predictions'].tolist(),
            'uncertainty': blr_results['uncertainty'].tolist(),
            'coefficients': blr_results['coefficients'].tolist(),
            'intercept': float(blr_results['intercept']),
            'metrics': blr_results['metrics']
        },
        'true_values': y_test.tolist(),
        'vix': vix_test.tolist()
    }

    output_path = results_dir / f'{ticker}_linear_baselines.json'
    with open(output_path, 'w') as f:
        json.dump(save_data, f, indent=2)

    print(f"\n✓ Saved to {output_path}")

    return save_data

def main():
    config = Config()

    print('='*80)
    print('LINEAR BASELINES: MA + Bayesian Linear Regression')
    print('='*80)

    all_results = {}

    for ticker in config.TICKERS:
        results = run_linear_baselines_for_ticker(ticker, config)
        all_results[ticker] = results

    # Save combined results
    combined_path = Path('results/predictions/linear_baselines/all_linear_baselines.json')
    with open(combined_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*80}")
    print('Summary')
    print('='*80)

    for ticker in config.TICKERS:
        print(f"\n{ticker}:")
        print(f"  MA   RMSE: {all_results[ticker]['ma']['metrics']['rmse']:.6f}")
        print(f"  BLR  RMSE: {all_results[ticker]['blr']['metrics']['rmse']:.6f}")
        print(f"  BLR  Corr: {all_results[ticker]['blr']['metrics']['uncertainty_error_corr']:.4f}")

    print(f"\n{'='*80}")
    print('✓ All linear baselines complete')
    print('='*80)

if __name__ == '__main__':
    main()
