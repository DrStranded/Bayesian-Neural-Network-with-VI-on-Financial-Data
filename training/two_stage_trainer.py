"""
Two-Stage Trainer

Stage 1: Train mean model (LSTM + fc_mean) with MSE loss
Stage 2: Freeze mean model, train Bayesian volatility head with ELBO
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm


class TwoStageTrainer:
    """Trainer for the two-stage model."""

    def __init__(self, model, train_loader, val_loader, config, device=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        # Device
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device('mps')
            elif torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device

        self.model.to(self.device)
        print(f"Using device: {self.device}")

        # Optimizers (will be set per stage)
        self.optimizer = None

        # History
        self.history = {
            'stage1': {'train_loss': [], 'val_loss': []},
            'stage2': {'train_loss': [], 'val_loss': [], 'val_nll': [], 'val_kl': []}
        }

    def train_stage1(self, epochs, lr=1e-3, patience=10, save_path=None):
        """
        Train Stage 1: Mean prediction model.
        """
        print("\n" + "=" * 70)
        print("STAGE 1: Training Mean Model (LSTM + fc_mean)")
        print("=" * 70)

        self.model.unfreeze_mean_model()
        self.optimizer = torch.optim.Adam(
            list(self.model.lstm.parameters()) + list(self.model.fc_mean.parameters()),
            lr=lr,
            weight_decay=1e-5
        )

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            # Train
            self.model.train()
            train_losses = []

            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
            for batch in pbar:
                X = batch['X'].to(self.device)
                y = batch['y'].to(self.device)

                self.optimizer.zero_grad()
                loss, _ = self.model.stage1_loss(X, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.optimizer.step()

                train_losses.append(loss.item())
                pbar.set_postfix({'loss': f'{loss.item():.6f}'})

            # Validate
            val_loss = self._validate_stage1()

            train_loss = np.mean(train_losses)
            self.history['stage1']['train_loss'].append(train_loss)
            self.history['stage1']['val_loss'].append(val_loss)

            print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                if save_path:
                    self._save_checkpoint(save_path, stage=1)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        print(f"Stage 1 complete. Best val loss: {best_val_loss:.6f}")
        return self.history['stage1']

    def train_stage2(self, epochs, lr=1e-3, patience=10, save_path=None):
        """
        Train Stage 2: Bayesian volatility head.
        """
        print("\n" + "=" * 70)
        print("STAGE 2: Training Bayesian Volatility Head")
        print("=" * 70)

        self.model.freeze_mean_model()
        self.optimizer = torch.optim.Adam(
            self.model.vol_head.parameters(),
            lr=lr,
            weight_decay=1e-5
        )

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            # Train
            self.model.train()
            train_losses = []

            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
            for batch in pbar:
                X = batch['X'].to(self.device)
                y = batch['y'].to(self.device)
                vix = batch['vix'].to(self.device).squeeze()

                self.optimizer.zero_grad()
                loss, metrics = self.model.stage2_loss(X, y, vix)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.vol_head.parameters(), 5.0)
                self.optimizer.step()

                train_losses.append(loss.item())
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'σ': f'{metrics["mean_sigma"]:.4f}'
                })

            # Validate
            val_loss, val_metrics = self._validate_stage2()

            train_loss = np.mean(train_losses)
            self.history['stage2']['train_loss'].append(train_loss)
            self.history['stage2']['val_loss'].append(val_loss)
            self.history['stage2']['val_nll'].append(val_metrics['nll'])
            self.history['stage2']['val_kl'].append(val_metrics['kl'])

            print(f"Epoch {epoch+1:3d} | Train: {train_loss:.4f} | "
                  f"Val: {val_loss:.4f} (NLL: {val_metrics['nll']:.4f}, σ: {val_metrics['mean_sigma']:.4f})")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                if save_path:
                    self._save_checkpoint(save_path, stage=2)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        print(f"Stage 2 complete. Best val loss: {best_val_loss:.4f}")
        return self.history['stage2']

    def _validate_stage1(self):
        """Validate Stage 1."""
        self.model.eval()
        losses = []
        with torch.no_grad():
            for batch in self.val_loader:
                X = batch['X'].to(self.device)
                y = batch['y'].to(self.device)
                loss, _ = self.model.stage1_loss(X, y)
                losses.append(loss.item())
        return np.mean(losses)

    def _validate_stage2(self):
        """Validate Stage 2."""
        self.model.eval()
        losses = []
        all_metrics = []
        with torch.no_grad():
            for batch in self.val_loader:
                X = batch['X'].to(self.device)
                y = batch['y'].to(self.device)
                vix = batch['vix'].to(self.device).squeeze()
                loss, metrics = self.model.stage2_loss(X, y, vix)
                losses.append(loss.item())
                all_metrics.append(metrics)

        avg_metrics = {
            'nll': np.mean([m['nll'] for m in all_metrics]),
            'kl': np.mean([m['kl'] for m in all_metrics]),
            'mean_sigma': np.mean([m['mean_sigma'] for m in all_metrics])
        }
        return np.mean(losses), avg_metrics

    def _save_checkpoint(self, path, stage):
        """Save model checkpoint."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'stage': stage,
            'model_state_dict': self.model.state_dict(),
            'history': self.history
        }, path)
        print(f"  ✓ Checkpoint saved: {path}")
