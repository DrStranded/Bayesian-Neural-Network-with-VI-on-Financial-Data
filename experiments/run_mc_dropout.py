"""
Train MC Dropout LSTM models
"""

import sys
from pathlib import Path
import numpy as np
import torch
import json
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from training.config import Config
from models.mc_dropout_lstm import create_mc_dropout_lstm
from data.dataset import StockDataset
from evaluation.metrics import compute_rmse, compute_mae
from torch.utils.data import DataLoader


def train_one_epoch(model, dataloader, optimizer, config):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        x = batch['X'].to(config.DEVICE)
        y = batch['y'].to(config.DEVICE)

        optimizer.zero_grad()
        loss = model.loss(x, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, config):
    """Evaluate with uncertainty quantification"""
    model.eval()

    all_mu = []
    all_epistemic = []
    all_aleatoric = []
    all_total = []
    all_y = []

    with torch.no_grad():
        for batch in dataloader:
            x = batch['X'].to(config.DEVICE)
            y = batch['y'].to(config.DEVICE)
            vix_batch = batch['vix'].to(config.DEVICE)

            mu, epistemic, aleatoric, total = model.predict_with_uncertainty(x, n_samples=100)

            all_mu.append(mu.cpu())
            all_epistemic.append(epistemic.cpu())
            all_aleatoric.append(aleatoric.cpu())
            all_total.append(total.cpu())
            all_y.append(y.cpu())

    mu = torch.cat(all_mu).numpy()
    epistemic = torch.cat(all_epistemic).numpy()
    aleatoric = torch.cat(all_aleatoric).numpy()
    total = torch.cat(all_total).numpy()
    y = torch.cat(all_y).numpy()

    rmse = compute_rmse(y, mu)
    mae = compute_mae(y, mu)

    # Uncertainty-error correlation
    errors = np.abs(y - mu).flatten()
    unc_flat = total.flatten()
    corr = np.corrcoef(unc_flat, errors)[0, 1]

    return {
        'mu': mu,
        'epistemic': epistemic,
        'aleatoric': aleatoric,
        'total': total,
        'rmse': rmse,
        'mae': mae,
        'uncertainty_error_corr': corr
    }


def train_mc_dropout_for_ticker(ticker, config):
    """Train MC Dropout LSTM for one ticker"""
    print(f"\n{'='*80}")
    print(f"Training MC Dropout LSTM: {ticker}")
    print('='*80)

    # Load data
    data_path = Path(config.DATA_DIR) / f'{ticker}_processed.npz'
    data = np.load(data_path)

    # Create datasets
    train_dataset = StockDataset(
        data['X_train'], data['y_train'], data['vix_train']
    )
    val_dataset = StockDataset(
        data['X_val'], data['y_val'], data['vix_val']
    )
    test_dataset = StockDataset(
        data['X_test'], data['y_test'], data['vix_test']
    )

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE)

    print(f"Data loaded: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

    # Create model
    model = create_mc_dropout_lstm(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    print(f"\nModel: MC Dropout LSTM")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"  Dropout rate: {model.dropout_rate}")

    # Training loop
    best_val_rmse = float('inf')
    patience_counter = 0

    print("\nTraining...")
    for epoch in range(config.NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, config)

        if (epoch + 1) % 10 == 0:
            val_results = evaluate(model, val_loader, config)
            val_rmse = val_results['rmse']

            print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}: "
                  f"Loss={train_loss:.6f}, Val RMSE={val_rmse:.6f}")

            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                patience_counter = 0

                # Save best model
                model_dir = Path('results/models')
                model_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), model_dir / f'{ticker}_mc_dropout_lstm.pt')
            else:
                patience_counter += 1
                if patience_counter >= config.EARLY_STOP_PATIENCE // 10:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

    # Load best model and evaluate on test set
    model.load_state_dict(torch.load(f'results/models/{ticker}_mc_dropout_lstm.pt'))
    test_results = evaluate(model, test_loader, config)

    print(f"\n{'='*80}")
    print(f"Test Results for {ticker}")
    print('='*80)
    print(f"RMSE: {test_results['rmse']:.6f}")
    print(f"MAE: {test_results['mae']:.6f}")
    print(f"Unc-Err Corr: {test_results['uncertainty_error_corr']:.4f}")
    print(f"Mean Epistemic: {test_results['epistemic'].mean():.6f}")
    print(f"Mean Aleatoric: {test_results['aleatoric'].mean():.6f}")
    print(f"Mean Total: {test_results['total'].mean():.6f}")

    # Save results
    results_dir = Path('results/predictions')
    results_dir.mkdir(parents=True, exist_ok=True)

    save_data = {
        'ticker': ticker,
        'predictions': test_results['mu'].tolist(),
        'epistemic': test_results['epistemic'].tolist(),
        'aleatoric': test_results['aleatoric'].tolist(),
        'total_uncertainty': test_results['total'].tolist(),
        'true_values': data['y_test'].tolist(),
        'vix': data['vix_test'].tolist(),
        'metrics': {
            'rmse': float(test_results['rmse']),
            'mae': float(test_results['mae']),
            'uncertainty_error_corr': float(test_results['uncertainty_error_corr'])
        }
    }

    with open(results_dir / f'{ticker}_mc_dropout.json', 'w') as f:
        json.dump(save_data, f, indent=2)

    return save_data


def main():
    config = Config()

    # Add DEVICE if not present
    if not hasattr(config, 'DEVICE'):
        config.DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    print('='*80)
    print('MC DROPOUT LSTM - Bayesian Uncertainty Quantification')
    print('='*80)
    print(f"Method: Gal & Ghahramani (2016)")
    print(f"Device: {config.DEVICE}")
    print(f"Dropout rate: 0.2")
    print(f"MC samples: 100")

    all_results = {}

    for ticker in config.TICKERS:
        results = train_mc_dropout_for_ticker(ticker, config)
        all_results[ticker] = results

    # Save combined results
    with open('results/predictions/all_mc_dropout.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*80}")
    print('Summary')
    print('='*80)
    for ticker in config.TICKERS:
        metrics = all_results[ticker]['metrics']
        print(f"\n{ticker}:")
        print(f"  RMSE: {metrics['rmse']:.6f}")
        print(f"  MAE: {metrics['mae']:.6f}")
        print(f"  Unc-Err Corr: {metrics['uncertainty_error_corr']:.4f}")

    print(f"\n{'='*80}")
    print('✓ All MC Dropout models trained')
    print('='*80)


if __name__ == '__main__':
    main()
