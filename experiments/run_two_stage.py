"""
Run Two-Stage Model Experiment

1. Train Stage 1 (mean model)
2. Train Stage 2 (Bayesian volatility head)
3. Evaluate on test set
4. Save results
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import json

from data.dataset import StockDataset
from torch.utils.data import DataLoader
from models.two_stage_model import TwoStageModel
from training.two_stage_trainer import TwoStageTrainer
from training.config import Config


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Get best available device."""
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def run_experiment(ticker, config, device):
    """Run two-stage experiment for one ticker."""
    print(f"\n{'#' * 70}")
    print(f"# {ticker}")
    print(f"{'#' * 70}")

    # Load data
    data_path = Path(config.DATA_DIR) / f'{ticker}_processed.npz'
    data = np.load(data_path)

    # Create datasets
    train_dataset = StockDataset(data['X_train'], data['y_train'], data['vix_train'])
    val_dataset = StockDataset(data['X_val'], data['y_val'], data['vix_val'])
    test_dataset = StockDataset(data['X_test'], data['y_test'], data['vix_test'])

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE)

    print(f"Data loaded: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

    # Create model
    model = TwoStageModel(
        input_size=config.INPUT_SIZE,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        prior_std=config.PRIOR_BASE_STD,
        vix_mean=config.VIX_MEAN,
        vix_scale=config.VIX_SCALE,
        kl_weight=1e-4
    )

    # Trainer
    trainer = TwoStageTrainer(model, train_loader, val_loader, config, device)

    # Stage 1: Train mean model
    save_path1 = Path(config.MODEL_DIR) / f"{ticker}_two_stage_s1.pt"
    history1 = trainer.train_stage1(
        epochs=config.NUM_EPOCHS,
        lr=config.LEARNING_RATE,
        patience=config.EARLY_STOP_PATIENCE,
        save_path=save_path1
    )

    # Stage 2: Train volatility head
    save_path2 = Path(config.MODEL_DIR) / f"{ticker}_two_stage_s2.pt"
    history2 = trainer.train_stage2(
        epochs=config.NUM_EPOCHS,
        lr=config.LEARNING_RATE,
        patience=config.EARLY_STOP_PATIENCE,
        save_path=save_path2
    )

    # Evaluate on test set
    print("\n--- Test Set Evaluation ---")
    model.eval()

    all_mu, all_sigma_ale, all_sigma_epi = [], [], []
    all_y, all_vix = [], []

    with torch.no_grad():
        for batch in test_loader:
            X = batch['X'].to(device)
            y = batch['y'].to(device)
            vix = batch['vix'].to(device).squeeze()

            mu, sigma_ale, sigma_epi, _ = model.predict(X, vix, n_samples=config.N_SAMPLES_TEST)

            all_mu.append(mu.cpu().numpy())
            all_sigma_ale.append(sigma_ale.cpu().numpy())
            all_sigma_epi.append(sigma_epi.cpu().numpy())
            all_y.append(y.cpu().numpy())
            all_vix.append(vix.cpu().numpy().reshape(-1, 1) if vix.dim() > 0 else np.array([[vix.item()]]))

    # Concatenate
    mu = np.concatenate(all_mu)
    sigma_ale = np.concatenate(all_sigma_ale)
    sigma_epi = np.concatenate(all_sigma_epi)
    y_true = np.concatenate(all_y)
    vix = np.concatenate(all_vix)

    # Metrics
    errors = mu - y_true
    rmse = np.sqrt(np.mean(errors ** 2))
    mae = np.mean(np.abs(errors))

    # Coverage
    for conf, z in [(0.5, 0.674), (0.95, 1.96)]:
        lower = mu - z * sigma_ale
        upper = mu + z * sigma_ale
        coverage = np.mean((y_true >= lower) & (y_true <= upper))
        print(f"  {int(conf*100)}% CI Coverage: {coverage:.3f} (target: {conf})")

    print(f"\n  RMSE: {rmse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  Mean σ_aleatoric: {sigma_ale.mean():.4f}")
    print(f"  Mean σ_epistemic: {sigma_epi.mean():.4f}")

    # Save results
    results = {
        'ticker': ticker,
        'predictions': mu.tolist(),
        'targets': y_true.tolist(),
        'sigma_aleatoric': sigma_ale.tolist(),
        'sigma_epistemic': sigma_epi.tolist(),
        'vix': vix.tolist(),
        'metrics': {
            'rmse': float(rmse),
            'mae': float(mae),
            'mean_sigma_ale': float(sigma_ale.mean()),
            'mean_sigma_epi': float(sigma_epi.mean())
        },
        'history': {'stage1': history1, 'stage2': history2}
    }

    return results


def main():
    set_seed(42)
    config = Config()
    device = get_device()

    print("=" * 70)
    print("Two-Stage Model Experiment")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Tickers: {config.TICKERS}")

    results_dir = Path(config.PRED_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for ticker in config.TICKERS:
        results = run_experiment(ticker, config, device)
        all_results[ticker] = results

        # Save individual
        with open(results_dir / f"{ticker}_two_stage.json", 'w') as f:
            json.dump(results, f, indent=2)

    # Save all
    with open(results_dir / "all_two_stage.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"{'Ticker':<8} {'RMSE':<10} {'σ_ale':<10} {'σ_epi':<10}")
    print("-" * 40)
    for ticker, r in all_results.items():
        m = r['metrics']
        print(f"{ticker:<8} {m['rmse']:<10.6f} {m['mean_sigma_ale']:<10.4f} {m['mean_sigma_epi']:<10.4f}")

    print(f"\nResults saved to {results_dir}")


if __name__ == '__main__':
    main()
