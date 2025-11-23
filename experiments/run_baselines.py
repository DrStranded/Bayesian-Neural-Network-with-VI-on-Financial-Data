import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import json

from data.dataset import load_processed_data, get_dataloaders
from models.baseline_ma import MovingAverageModel
from models.baseline_lstm import StandardLSTM
from training.config import Config
from training.trainer import Trainer
from utils.helpers import set_seed, get_device

def run_ma_baseline(ticker, config):
    """Run Moving Average baseline"""
    print(f"\n{'='*80}")
    print(f"Running Moving Average Baseline for {ticker}")
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
    model = MovingAverageModel(window=config.SEQ_LEN)

    # Evaluate on test set
    all_preds = []
    all_targets = []

    for batch in test_loader:
        X = batch['X'].numpy()
        y = batch['y'].numpy()

        pred, _ = model.predict(X)
        all_preds.append(pred)
        all_targets.append(y)

    predictions = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    # Compute metrics
    mse = np.mean((predictions - targets)**2)
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(mse)

    print(f"\nTest Results:")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE: {mae:.6f}")

    results = {
        'ticker': ticker,
        'model': 'ma',
        'predictions': predictions.tolist(),
        'targets': targets.tolist(),
        'metrics': {
            'rmse': float(rmse),
            'mae': float(mae),
            'mse': float(mse)
        }
    }

    return results

def run_standard_lstm(ticker, config, device):
    """Train and evaluate Standard LSTM"""
    print(f"\n{'='*80}")
    print(f"Training Standard LSTM for {ticker}")
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
    model = StandardLSTM(
        input_size=config.INPUT_SIZE,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT
    )

    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        model_type='standard_lstm',
        device=device
    )

    # Train
    save_path = Path(config.MODEL_DIR) / f"{ticker}_standard_lstm.pt"
    history = trainer.train(save_path=save_path)

    # Evaluate on test set
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            X = batch['X'].to(device)
            y = batch['y'].to(device)

            pred = model(X)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    predictions = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    # Compute metrics
    mse = np.mean((predictions - targets)**2)
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(mse)

    print(f"\nTest Results:")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE: {mae:.6f}")

    results = {
        'ticker': ticker,
        'model': 'standard_lstm',
        'predictions': predictions.tolist(),
        'targets': targets.tolist(),
        'metrics': {
            'rmse': float(rmse),
            'mae': float(mae),
            'mse': float(mse)
        },
        'history': history
    }

    return results

def main():
    """Run all baseline experiments"""
    # Setup
    set_seed(42)
    config = Config()
    device = get_device()

    print("="*80)
    print("Baseline Models Experiment")
    print("="*80)
    print(f"Tickers: {config.TICKERS}")
    print(f"Device: {device}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Epochs: {config.NUM_EPOCHS}")

    # Create results directory
    results_dir = Path(config.PRED_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # Run experiments for each ticker
    for ticker in config.TICKERS:
        print(f"\n{'#'*80}")
        print(f"# Processing {ticker}")
        print(f"{'#'*80}")

        # Moving Average
        ma_results = run_ma_baseline(ticker, config)
        all_results[f"{ticker}_ma"] = ma_results

        # Standard LSTM
        lstm_results = run_standard_lstm(ticker, config, device)
        all_results[f"{ticker}_standard_lstm"] = lstm_results

        # Save individual results
        with open(results_dir / f"{ticker}_baselines.json", 'w') as f:
            json.dump({
                'ma': ma_results,
                'standard_lstm': lstm_results
            }, f, indent=2)

        print(f"\n✓ Results saved for {ticker}")

    # Save combined results
    with open(results_dir / "all_baselines.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    # Summary
    print("\n" + "="*80)
    print("Baseline Experiments Complete!")
    print("="*80)
    print("\nSummary:")
    print("-"*80)
    print(f"{'Ticker':<10} {'Model':<15} {'RMSE':<12} {'MAE':<12}")
    print("-"*80)

    for ticker in config.TICKERS:
        ma_rmse = all_results[f"{ticker}_ma"]['metrics']['rmse']
        ma_mae = all_results[f"{ticker}_ma"]['metrics']['mae']
        lstm_rmse = all_results[f"{ticker}_standard_lstm"]['metrics']['rmse']
        lstm_mae = all_results[f"{ticker}_standard_lstm"]['metrics']['mae']

        print(f"{ticker:<10} {'MA':<15} {ma_rmse:<12.6f} {ma_mae:<12.6f}")
        print(f"{ticker:<10} {'LSTM':<15} {lstm_rmse:<12.6f} {lstm_mae:<12.6f}")

    print("-"*80)
    print(f"\nResults saved to: {results_dir}")

if __name__ == '__main__':
    main()
