"""
Comprehensive Diagnostics and Visualization for Two-Stage Model

Run this to:
1. Check if Stage 1 (mean) is learning properly
2. Compare μ with MA baseline
3. Verify VIX → σ relationship
4. Generate publication-quality plots
"""

import sys
sys.path.append('.')

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Style settings for better plots
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'figure.figsize': (14, 5),
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 1.5,
})


def compute_ma_prediction(y_series, window=20):
    """Compute Moving Average prediction (predict t using t-window:t-1)."""
    ma_pred = np.zeros_like(y_series)
    for t in range(len(y_series)):
        if t < window:
            ma_pred[t] = np.mean(y_series[:t+1]) if t > 0 else 0
        else:
            ma_pred[t] = np.mean(y_series[t-window:t])
    return ma_pred


def diagnose_stage1(model, test_loader, device):
    """
    Diagnose Stage 1: Is the mean model learning anything useful?
    """
    print("=" * 70)
    print("DIAGNOSTIC: Stage 1 Mean Model")
    print("=" * 70)
    
    model.eval()
    all_mu = []
    all_y = []
    
    with torch.no_grad():
        for batch in test_loader:
            X = batch['X'].to(device)
            y = batch['y'].to(device)
            
            h = model.get_hidden_state(X)
            mu = model.fc_mean(h)
            
            all_mu.append(mu.cpu().numpy())
            all_y.append(y.cpu().numpy())
    
    mu = np.concatenate(all_mu).flatten()
    y = np.concatenate(all_y).flatten()
    
    # Compute MA baseline
    ma_pred = compute_ma_prediction(y, window=20)
    
    # Metrics
    rmse_lstm = np.sqrt(np.mean((mu - y) ** 2))
    rmse_ma = np.sqrt(np.mean((ma_pred - y) ** 2))
    rmse_zero = np.sqrt(np.mean(y ** 2))  # Predicting 0
    rmse_mean = np.sqrt(np.mean((y - y.mean()) ** 2))  # Predicting mean
    
    mu_std = np.std(mu)
    mu_range = mu.max() - mu.min()
    
    print(f"\n--- Prediction Statistics ---")
    print(f"μ mean: {mu.mean():.6f}")
    print(f"μ std:  {mu_std:.6f}")
    print(f"μ range: [{mu.min():.6f}, {mu.max():.6f}]")
    print(f"y mean: {y.mean():.6f}")
    print(f"y std:  {y.std():.6f}")
    
    print(f"\n--- RMSE Comparison ---")
    print(f"LSTM μ:      {rmse_lstm:.6f}")
    print(f"MA(20):      {rmse_ma:.6f}")
    print(f"Predict 0:   {rmse_zero:.6f}")
    print(f"Predict ȳ:   {rmse_mean:.6f}")
    
    # Diagnosis
    print(f"\n--- Diagnosis ---")
    if mu_std < 0.001:
        print("⚠ WARNING: μ is nearly constant! LSTM not learning temporal patterns.")
        print("  Possible causes:")
        print("  - Stage 1 not trained enough epochs")
        print("  - Learning rate too high (diverged) or too low (stuck)")
        print("  - post_lstm layer may be hurting gradient flow")
    
    if rmse_lstm > rmse_ma:
        print(f"⚠ WARNING: LSTM ({rmse_lstm:.6f}) worse than MA ({rmse_ma:.6f})")
    elif rmse_lstm < rmse_ma * 0.99:
        print(f"✓ LSTM ({rmse_lstm:.6f}) beats MA ({rmse_ma:.6f})")
    else:
        print(f"~ LSTM ≈ MA (both around {rmse_lstm:.6f})")
    
    return {
        'mu': mu, 'y': y, 'ma_pred': ma_pred,
        'rmse_lstm': rmse_lstm, 'rmse_ma': rmse_ma
    }


def diagnose_stage2(model, test_loader, device):
    """
    Diagnose Stage 2: Is σ responding to VIX?
    """
    print("\n" + "=" * 70)
    print("DIAGNOSTIC: Stage 2 Volatility Model")
    print("=" * 70)
    
    model.eval()
    all_sigma = []
    all_vix = []
    all_residual = []
    
    with torch.no_grad():
        for batch in test_loader:
            X = batch['X'].to(device)
            y = batch['y'].to(device)
            vix = batch['vix'].to(device).squeeze()
            
            mu, sigma = model(X, vix)
            residual = (y - mu).abs()
            
            all_sigma.append(sigma.cpu().numpy())
            all_vix.append(vix.cpu().numpy().reshape(-1, 1))
            all_residual.append(residual.cpu().numpy())
    
    sigma = np.concatenate(all_sigma).flatten()
    vix = np.concatenate(all_vix).flatten()
    residual = np.concatenate(all_residual).flatten()
    
    # Correlation
    corr_vix_sigma = np.corrcoef(vix, sigma)[0, 1]
    corr_sigma_error = np.corrcoef(sigma, residual)[0, 1]
    
    print(f"\n--- VIX-σ Relationship ---")
    print(f"Correlation(VIX, σ): {corr_vix_sigma:.4f}")
    print(f"Correlation(σ, |error|): {corr_sigma_error:.4f}")
    
    print(f"\n--- σ Statistics ---")
    print(f"σ mean: {sigma.mean():.4f}")
    print(f"σ std:  {sigma.std():.4f}")
    print(f"σ range: [{sigma.min():.4f}, {sigma.max():.4f}]")
    
    # VIX bins analysis
    print(f"\n--- σ by VIX Regime ---")
    vix_bins = [(0, 15, 'Low'), (15, 20, 'Normal'), (20, 25, 'Elevated'), (25, 100, 'High')]
    for low, high, name in vix_bins:
        mask = (vix >= low) & (vix < high)
        if mask.sum() > 0:
            print(f"  VIX {name:8s} [{low:2d}-{high:2d}]: σ = {sigma[mask].mean():.4f} (n={mask.sum()})")
    
    return {
        'sigma': sigma, 'vix': vix, 'residual': residual,
        'corr_vix_sigma': corr_vix_sigma, 'corr_sigma_error': corr_sigma_error
    }


def plot_comprehensive_results(stage1_data, stage2_data, save_dir='results/figures'):
    """
    Generate publication-quality plots.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    mu = stage1_data['mu']
    y = stage1_data['y']
    ma_pred = stage1_data['ma_pred']
    sigma = stage2_data['sigma']
    vix = stage2_data['vix']
    residual = stage2_data['residual']
    
    T = len(y)
    time = np.arange(T)
    
    # =========================================
    # Plot 1: Mean Prediction Comparison
    # =========================================
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # Top: Full comparison
    ax = axes[0]
    ax.plot(time, y, 'k-', alpha=0.7, linewidth=0.8, label='True Returns')
    ax.plot(time, mu, 'b-', linewidth=1.5, label=f'LSTM μ (RMSE={stage1_data["rmse_lstm"]:.4f})')
    ax.plot(time, ma_pred, 'r--', linewidth=1.2, label=f'MA(20) (RMSE={stage1_data["rmse_ma"]:.4f})')
    ax.set_ylabel('Log Return')
    ax.set_title('Mean Prediction: LSTM vs Moving Average')
    ax.legend(loc='upper right')
    ax.set_ylim([-0.08, 0.08])
    
    # Bottom: Zoom first 100
    ax = axes[1]
    zoom = 100
    ax.plot(time[:zoom], y[:zoom], 'k-', alpha=0.7, linewidth=1, label='True')
    ax.plot(time[:zoom], mu[:zoom], 'b-', linewidth=2, label='LSTM μ')
    ax.plot(time[:zoom], ma_pred[:zoom], 'r--', linewidth=1.5, label='MA(20)')
    ax.set_xlabel('Time')
    ax.set_ylabel('Log Return')
    ax.set_title('Zoomed View (First 100 Days)')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/diagnostic_mean_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_dir}/diagnostic_mean_comparison.png")
    
    # =========================================
    # Plot 2: VIX vs Sigma Scatter + Time Series
    # =========================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Scatter plot
    ax = axes[0]
    scatter = ax.scatter(vix, sigma, c=time, cmap='viridis', alpha=0.6, s=20)
    
    # Fit line
    z = np.polyfit(vix, sigma, 1)
    p = np.poly1d(z)
    vix_line = np.linspace(vix.min(), vix.max(), 100)
    ax.plot(vix_line, p(vix_line), 'r-', linewidth=2, 
            label=f'Fit: σ = {z[0]:.4f}×VIX + {z[1]:.4f}')
    
    ax.set_xlabel('VIX Index')
    ax.set_ylabel('Predicted σ')
    ax.set_title(f'VIX → σ Relationship (Corr = {stage2_data["corr_vix_sigma"]:.3f})')
    ax.legend()
    plt.colorbar(scatter, ax=ax, label='Time')
    
    # Right: Time series overlay
    ax = axes[1]
    ax2 = ax.twinx()
    
    l1, = ax.plot(time, vix, 'r-', alpha=0.7, linewidth=1, label='VIX')
    l2, = ax2.plot(time, sigma, 'b-', linewidth=1.5, label='Predicted σ')
    
    ax.set_xlabel('Time')
    ax.set_ylabel('VIX', color='red')
    ax2.set_ylabel('Predicted σ', color='blue')
    ax.set_title('VIX and σ Over Time')
    ax.legend(handles=[l1, l2], loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/diagnostic_vix_sigma.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_dir}/diagnostic_vix_sigma.png")
    
    # =========================================
    # Plot 3: Prediction with Adaptive CI
    # =========================================
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Confidence intervals
    ci_50 = 0.674 * sigma
    ci_95 = 1.96 * sigma
    
    ax.fill_between(time, mu - ci_95, mu + ci_95, alpha=0.2, color='blue', label='95% CI')
    ax.fill_between(time, mu - ci_50, mu + ci_50, alpha=0.4, color='blue', label='50% CI')
    ax.plot(time, y, 'k-', linewidth=0.8, alpha=0.8, label='True Returns')
    ax.plot(time, mu, 'b-', linewidth=1.5, label='Predicted μ')
    
    # Mark high VIX periods
    high_vix_mask = vix > 25
    if high_vix_mask.any():
        ax.fill_between(time, -0.1, 0.1, where=high_vix_mask, 
                       alpha=0.1, color='red', label='High VIX (>25)')
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Log Return')
    ax.set_title('Two-Stage Model: Predictions with VIX-Adaptive Uncertainty')
    ax.legend(loc='upper right')
    ax.set_ylim([-0.1, 0.1])
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/diagnostic_predictions_adaptive.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_dir}/diagnostic_predictions_adaptive.png")
    
    # =========================================
    # Plot 4: Calibration Check
    # =========================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Coverage by confidence level
    ax = axes[0]
    conf_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    expected = []
    actual = []
    
    for conf in conf_levels:
        from scipy import stats
        z = stats.norm.ppf((1 + conf) / 2)
        in_ci = np.abs(y - mu) <= z * sigma
        expected.append(conf)
        actual.append(in_ci.mean())
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect Calibration')
    ax.plot(expected, actual, 'bo-', markersize=8, linewidth=2, label='Model')
    ax.set_xlabel('Expected Coverage')
    ax.set_ylabel('Actual Coverage')
    ax.set_title('Calibration Curve')
    ax.legend()
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    # Right: Coverage over time (rolling)
    ax = axes[1]
    window = 50
    rolling_coverage = []
    for t in range(window, T):
        in_ci = np.abs(y[t-window:t] - mu[t-window:t]) <= 1.96 * sigma[t-window:t]
        rolling_coverage.append(in_ci.mean())
    
    ax.plot(time[window:], rolling_coverage, 'b-', linewidth=1)
    ax.axhline(y=0.95, color='r', linestyle='--', label='Target (95%)')
    ax.set_xlabel('Time')
    ax.set_ylabel('Rolling 95% CI Coverage')
    ax.set_title(f'Rolling Calibration (window={window})')
    ax.legend()
    ax.set_ylim([0.5, 1.05])
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/diagnostic_calibration.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_dir}/diagnostic_calibration.png")
    
    # =========================================
    # Plot 5: Uncertainty Decomposition (Better)
    # =========================================
    # This requires epistemic uncertainty - need to call model.predict()
    # Skipping for now, will be in full evaluation
    
    print(f"\n✓ All diagnostic plots saved to {save_dir}/")


def run_full_diagnostic(model, test_loader, device, save_dir='results/figures'):
    """Run all diagnostics."""
    stage1_data = diagnose_stage1(model, test_loader, device)
    stage2_data = diagnose_stage2(model, test_loader, device)
    plot_comprehensive_results(stage1_data, stage2_data, save_dir)
    
    return stage1_data, stage2_data


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    from data.dataset import load_processed_data, get_dataloaders
    from models.two_stage_model import TwoStageModel
    from training.config import Config
    from utils.helpers import get_device
    
    config = Config()
    device = get_device()
    
    # Load model
    model = TwoStageModel(
        input_size=config.INPUT_SIZE,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        prior_std=config.PRIOR_BASE_STD,
        vix_mean=config.VIX_MEAN,
        vix_scale=config.VIX_SCALE,
        kl_weight=1e-4
    )
    
    # Load checkpoint if exists
    ckpt_path = Path(config.MODEL_DIR) / "AAPL_two_stage_s2.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"✓ Loaded checkpoint: {ckpt_path}")
    else:
        print(f"⚠ No checkpoint found at {ckpt_path}")
        print("  Running with untrained model for testing...")
    
    model.to(device)
    
    # Load data
    ticker = 'AAPL'
    data = load_processed_data(ticker, config.DATA_DIR)
    _, _, test_loader = get_dataloaders(
        data['train'], data['val'], data['test'],
        batch_size=config.BATCH_SIZE
    )
    
    # Run diagnostics
    run_full_diagnostic(model, test_loader, device)
