import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from pathlib import Path
import json

from data.dataset import load_processed_data, get_dataloaders
from models.bayesian_lstm import BayesianLSTM
from training.config import Config
from training.trainer import Trainer
from utils.helpers import set_seed, get_device

def run_bayesian_lstm(ticker, config, device):
    """Train and evaluate Bayesian LSTM with adaptive prior"""
    print(f"\n{'='*80}")
    print(f"Training Bayesian LSTM for {ticker}")
    print(f"{'='*80}")

    # Load data
    data_dict = load_processed_data(ticker, config.DATA_DIR)

    # Create dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dict['train'],
        data_dict['val'],
        data_dict['test'],
        batch_size=config.BATCH_SIZE
    )

    # Create model
    model = BayesianLSTM(
        input_size=config.INPUT_SIZE,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        prior_base_std=config.PRIOR_BASE_STD,
        vix_mean=config.VIX_MEAN,
        vix_scale=config.VIX_SCALE,
        vix_sensitivity=config.VIX_SENSITIVITY
    )

    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        model_type='bayesian_lstm',
        device=device
    )

    # Train
    save_path = Path(config.MODEL_DIR) / f"{ticker}_bayesian_lstm.pt"
    history = trainer.train(save_path=save_path)

    # Evaluate on test set with uncertainty quantification
    model.eval()
    all_preds = []
    all_total_stds = []
    all_epistemic_stds = []
    all_aleatoric_stds = []
    all_targets = []
    all_vix = []

    print("\nEvaluating on test set with uncertainty quantification...")

    with torch.no_grad():
        for batch in test_loader:
            X = batch['X'].to(device)
            y = batch['y'].to(device)
            vix = batch['vix'].to(device).squeeze()

            # Handle VIX
            if vix.dim() == 0:
                vix = vix.unsqueeze(0)
            vix_mean = vix.mean()

            # Predict with uncertainty
            pred_mean, total_std, epistemic_std, aleatoric_std = model.predict(
                X, vix_mean, n_samples=config.N_SAMPLES_TEST
            )

            all_preds.append(pred_mean.cpu().numpy())
            all_total_stds.append(total_std.cpu().numpy())
            all_epistemic_stds.append(epistemic_std.cpu().numpy())
            all_aleatoric_stds.append(aleatoric_std.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            all_vix.append(vix.cpu().numpy())

    # Concatenate results
    predictions = np.concatenate(all_preds, axis=0)
    total_stds = np.concatenate(all_total_stds, axis=0)
    epistemic_stds = np.concatenate(all_epistemic_stds, axis=0)
    aleatoric_stds = np.concatenate(all_aleatoric_stds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    vix_values = np.concatenate(all_vix, axis=0)

    # Compute metrics
    errors = predictions - targets
    mse = np.mean(errors**2)
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(mse)

    # Uncertainty metrics
    avg_epistemic = np.mean(epistemic_stds)
    avg_aleatoric = np.mean(aleatoric_stds)
    avg_total = np.mean(total_stds)

    # Uncertainty-error correlation
    uncertainty_error_corr = np.corrcoef(
        total_stds.flatten(),
        np.abs(errors).flatten()
    )[0, 1]

    print(f"\nTest Results:")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"\nUncertainty Statistics:")
    print(f"  Avg Epistemic: {avg_epistemic:.6f}")
    print(f"  Avg Aleatoric: {avg_aleatoric:.6f}")
    print(f"  Avg Total: {avg_total:.6f}")
    print(f"  Uncertainty-Error Correlation: {uncertainty_error_corr:.4f}")

    results = {
        'ticker': ticker,
        'model': 'bayesian_lstm',
        'predictions': predictions.tolist(),
        'targets': targets.tolist(),
        'total_std': total_stds.tolist(),
        'epistemic_std': epistemic_stds.tolist(),
        'aleatoric_std': aleatoric_stds.tolist(),
        'vix': vix_values.tolist(),
        'metrics': {
            'rmse': float(rmse),
            'mae': float(mae),
            'mse': float(mse),
            'avg_epistemic': float(avg_epistemic),
            'avg_aleatoric': float(avg_aleatoric),
            'avg_total_std': float(avg_total),
            'uncertainty_error_corr': float(uncertainty_error_corr)
        },
        'history': history
    }

    return results

def main():
    """Run Bayesian LSTM experiments"""
    # Setup
    set_seed(42)
    config = Config()
    device = get_device()

    print("="*80)
    print("Bayesian LSTM Experiment")
    print("="*80)
    print(f"Tickers: {config.TICKERS}")
    print(f"Device: {device}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Epochs: {config.NUM_EPOCHS}")
    print(f"Test samples: {config.N_SAMPLES_TEST}")
    print(f"\nAdaptive Prior Parameters:")
    print(f"  Base std: {config.PRIOR_BASE_STD}")
    print(f"  VIX mean: {config.VIX_MEAN}")
    print(f"  VIX scale: {config.VIX_SCALE}")
    print(f"  VIX sensitivity: {config.VIX_SENSITIVITY}")

    # Create results directory
    results_dir = Path(config.PRED_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # Run experiments for each ticker
    for ticker in config.TICKERS:
        print(f"\n{'#'*80}")
        print(f"# Processing {ticker}")
        print(f"{'#'*80}")

        # Bayesian LSTM
        bayes_results = run_bayesian_lstm(ticker, config, device)
        all_results[f"{ticker}_bayesian_lstm"] = bayes_results

        # Save individual results
        with open(results_dir / f"{ticker}_bayesian.json", 'w') as f:
            json.dump(bayes_results, f, indent=2)

        print(f"\n✓ Results saved for {ticker}")

    # Save combined results
    with open(results_dir / "all_bayesian.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    # Summary
    print("\n" + "="*80)
    print("Bayesian LSTM Experiments Complete!")
    print("="*80)
    print("\nSummary:")
    print("-"*80)
    print(f"{'Ticker':<10} {'RMSE':<12} {'MAE':<12} {'Unc-Err Corr':<15}")
    print("-"*80)

    for ticker in config.TICKERS:
        result = all_results[f"{ticker}_bayesian_lstm"]
        rmse = result['metrics']['rmse']
        mae = result['metrics']['mae']
        corr = result['metrics']['uncertainty_error_corr']

        print(f"{ticker:<10} {rmse:<12.6f} {mae:<12.6f} {corr:<15.4f}")

    print("-"*80)
    print(f"\nResults saved to: {results_dir}")

if __name__ == '__main__':
    main()
