"""
Model Trainer Module

This module implements the training loop for both baseline and Bayesian models.
Handles ELBO computation for Bayesian models, gradient descent optimization,
learning rate scheduling, checkpointing, and training monitoring.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json


class Trainer:
    """
    Unified trainer for all model types.

    This class provides a consistent training interface for:
    - Moving Average (no training needed, just validation)
    - Standard LSTM (MSE loss)
    - Bayesian LSTM (ELBO loss with KL divergence)

    Features:
    ---------
    - Early stopping based on validation loss
    - Model checkpointing (saves best model)
    - Training history tracking
    - Gradient clipping for stability
    - Automatic device selection (MPS/CUDA/CPU)
    - Progress bars via tqdm

    Args:
        model: PyTorch model or wrapper (MovingAverage, StandardLSTM, BayesianLSTM)
        train_loader: Training DataLoader from get_dataloaders()
        val_loader: Validation DataLoader
        config: Config object with hyperparameters
        model_type: One of ['ma', 'standard_lstm', 'bayesian_lstm']
        device: torch.device (optional, auto-detects if None)

    Example:
        >>> from training.trainer import Trainer
        >>> trainer = Trainer(model, train_loader, val_loader, config, 'bayesian_lstm')
        >>> history = trainer.train(save_path='results/models/best_model.pt')
    """

    def __init__(self, model, train_loader, val_loader, config, model_type, device=None):
        """
        Initialize the trainer.

        Args:
            model: Model instance
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            config: Config object
            model_type: 'ma', 'standard_lstm', or 'bayesian_lstm'
            device: Device to use (auto-detected if None)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.model_type = model_type

        # Device selection
        # ----------------
        # Priority: MPS (Apple Silicon) > CUDA (NVIDIA) > CPU
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device('mps')
            elif torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device

        # Move model to device (skip for Moving Average)
        if model_type != 'ma':
            self.model.to(self.device)
            print(f"Model moved to device: {self.device}")

        # Optimizer setup (skip for Moving Average)
        # ------------------------------------------
        # Adam optimizer with weight decay for L2 regularization
        if model_type != 'ma':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=config.LEARNING_RATE,
                weight_decay=config.WEIGHT_DECAY
            )
            print(f"Optimizer: Adam (lr={config.LEARNING_RATE}, weight_decay={config.WEIGHT_DECAY})")

        # Training history
        # ----------------
        # Tracks loss and metrics across epochs
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_rmse': [],
            'val_rmse': []
        }

        # Early stopping tracking
        # -----------------------
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def train_epoch_standard(self):
        """
        Train one epoch for Standard LSTM.

        Uses standard MSE loss: L = mean((y - ŷ)²)

        Returns:
            avg_loss: Average training loss
            avg_rmse: Average RMSE (√MSE)
        """
        self.model.train()
        total_loss = 0      
        n_batches = 0
        

        # Progress bar
        pbar = tqdm(self.train_loader, desc="Training", leave=False)

        for batch in pbar:
            # Move data to device
            X = batch['X'].to(self.device)
            y = batch['y'].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            pred = self.model(X)

            # MSE loss
            loss = nn.functional.mse_loss(pred, y)

            # Backward pass
            loss.backward()

            # Gradient clipping (optional, for stability)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

            # Optimizer step
            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            total_mse += loss.item()
            n_batches += 1

            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Compute averages
        avg_loss = total_loss / n_batches
        avg_rmse = np.sqrt(total_mse / n_batches)

        return avg_loss, avg_rmse

    def train_epoch_bayesian(self):
        """
        Train one epoch for Bayesian LSTM.

        Uses ELBO loss: L = NLL + KL/batch_size

        Returns:
            avg_loss: Average ELBO loss
            metrics: Dict with 'nll' and 'kl' components
        """
        self.model.train()
        total_loss = 0
        total_nll = 0
        total_kl = 0
        total_se = 0.0       
        n_batches = 0
        n_points = 0

        # Progress bar
        pbar = tqdm(self.train_loader, desc="Training", leave=False)

        for batch in pbar:
            # Move data to device
            X = batch['X'].to(self.device)
            y = batch['y'].to(self.device)
            vix = batch['vix'].to(self.device).squeeze()

            # Handle VIX dimensions
            # ---------------------
            # Ensure vix is at least 1D
            if vix.dim() == 0:
                vix = vix.unsqueeze(0)

            

            # Forward pass + ELBO loss
            # ------------------------
            # n_samples=1 is efficient and works well in practice
            self.optimizer.zero_grad()
            vix_mean = vix.mean()
            loss, metrics = self.model.elbo_loss(X, y, vix_mean, n_samples=self.config.N_SAMPLES_TRAIN)

            # Backward pass
            loss.backward()

            # Gradient clipping
            # -----------------
            # Important for Bayesian models to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

            # Optimizer step
            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            total_nll += metrics['nll']
            total_kl += metrics['kl']
            n_batches += 1
            
            mean_pred, _, _, _ = self.model.predict(
                X, vix_mean, n_samples=10   # 或者用 config.N_SAMPLES_TRAIN
            )
            se = (mean_pred - y) ** 2
            total_se += se.sum().item()
            n_points += y.numel()

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'nll': f'{metrics["nll"]:.4f}',
                'kl_s': f'{metrics["kl_scaled"]:.4f}'
            })


        # Compute averages
        avg_loss = total_loss / n_batches
        avg_nll  = total_nll / n_batches
        avg_kl   = total_kl  / n_batches
        avg_rmse = np.sqrt(total_se / n_points)

        return avg_loss, {'nll': avg_nll, 'kl': avg_kl, 'rmse': avg_rmse}

    def validate(self):
        """
        Validate the model.

        Routes to appropriate validation method based on model type.

        Returns:
            avg_loss: Average validation loss
            avg_rmse: Average RMSE
        """
        if self.model_type == 'ma':
            return self.validate_ma()
        elif self.model_type == 'standard_lstm':
            return self.validate_standard()
        else:
            return self.validate_bayesian()

    def validate_ma(self):
        """
        Validate Moving Average model.

        No training needed, just compute validation metrics.

        Returns:
            avg_mse: Average MSE
            avg_rmse: Average RMSE
        """
        total_mse = 0
        n_batches = 0

        for batch in self.val_loader:
            # MA works with numpy
            X = batch['X'].numpy()
            y = batch['y'].numpy()

            # Predict (deterministic)
            pred, _ = self.model.predict(X)

            # MSE
            mse = np.mean((pred - y) ** 2)

            total_mse += mse
            n_batches += 1

        avg_mse = total_mse / n_batches
        avg_rmse = np.sqrt(avg_mse)

        return avg_mse, avg_rmse

    def validate_standard(self):
        """
        Validate Standard LSTM.

        Uses MSE loss on validation set.

        Returns:
            avg_mse: Average MSE
            avg_rmse: Average RMSE
        """
        self.model.eval()
        total_mse = 0
        n_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                # Move to device
                X = batch['X'].to(self.device)
                y = batch['y'].to(self.device)

                # Predict
                pred = self.model(X)

                # MSE
                mse = nn.functional.mse_loss(pred, y)

                total_mse += mse.item()
                n_batches += 1

        avg_mse = total_mse / n_batches
        avg_rmse = np.sqrt(avg_mse)

        return avg_mse, avg_rmse

    def validate_bayesian(self):
        """
        Validate Bayesian LSTM.

        Uses ELBO loss on validation set.

        Returns:
            avg_loss: Average ELBO loss
            avg_rmse: True RMSE from predictive means

        """
        self.model.eval()
        total_loss = 0.0
        total_nll = 0.0
        total_kl = 0.0
        total_se = 0.0  # sum of squared errors
        n_batches = 0
        n_points = 0

        with torch.no_grad():
            for batch in self.val_loader:
                X = batch['X'].to(self.device)
                y = batch['y'].to(self.device)
                vix = batch['vix'].to(self.device).squeeze()

                if vix.dim() == 0:
                    vix = vix.unsqueeze(0)
                vix_mean = vix.mean()

                # ELBO loss
                loss, metrics = self.model.elbo_loss(
                    X, y, vix_mean, n_samples=1
                )
                total_loss += loss.item()
                total_nll += metrics['nll']
                total_kl += metrics['kl']
                n_batches += 1

                # True RMSE from predictions
                mean_pred, _, _, _ = self.model.predict(
                    X, vix_mean, n_samples=10
                )
                se = (mean_pred - y) ** 2
                total_se += se.sum().item()
                n_points += y.numel()

        avg_loss = total_loss / n_batches
        avg_nll = total_nll / n_batches
        avg_kl = total_kl / n_batches
        avg_rmse = np.sqrt(total_se / n_points)  # ✅ 正确的RMSE

        return avg_loss, avg_rmse

    def train(self, save_path=None):
        """
        Full training loop with early stopping.

        Process:
        --------
        1. For each epoch:
           - Train on training set
           - Validate on validation set
           - Track metrics
           - Check early stopping
           - Save best model

        2. Early stopping:
           - Monitor validation loss
           - Stop if no improvement for EARLY_STOP_PATIENCE epochs

        Args:
            save_path: Path to save best model checkpoint (optional)

        Returns:
            history: Dictionary with training history
                    {'train_loss', 'val_loss', 'train_rmse', 'val_rmse'}

        Example:
            >>> history = trainer.train(save_path='results/models/best.pt')
            >>> print(f"Best val loss: {min(history['val_loss'])}")
        """
        # Moving Average doesn't need training
        if self.model_type == 'ma':
            print("\nMoving Average - No training needed, running validation...")
            val_loss, val_rmse = self.validate()
            print(f"Moving Average - Val Loss: {val_loss:.6f}, Val RMSE: {val_rmse:.6f}\n")
            return {'val_loss': val_loss, 'val_rmse': val_rmse}

        # Print training info
        print(f"\nTraining {self.model_type} on {self.device}")
        print("=" * 80)
        print(f"Epochs: {self.config.NUM_EPOCHS}")
        print(f"Batch size: {self.config.BATCH_SIZE}")
        print(f"Learning rate: {self.config.LEARNING_RATE}")
        print(f"Early stopping patience: {self.config.EARLY_STOP_PATIENCE}")
        print("=" * 80 + "\n")

        # Training loop
        for epoch in range(self.config.NUM_EPOCHS):
            # Train one epoch
            # ---------------
            if self.model_type == 'standard_lstm':
                train_loss, train_rmse = self.train_epoch_standard()
            else:  # bayesian_lstm
                train_loss, train_metrics = self.train_epoch_bayesian()
                train_rmse = train_metrics['rmse']

            # Validate
            # --------
            val_loss, val_rmse = self.validate()

            # Store history
            # -------------
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_rmse'].append(train_rmse)
            self.history['val_rmse'].append(val_rmse)

            # Print progress
            # --------------
            if self.model_type == 'bayesian_lstm':
                print(f"Epoch {epoch+1:3d}/{self.config.NUM_EPOCHS} | "
                      f"Train Loss: {train_loss:.4f} "
                      f"(NLL: {train_metrics['nll']:.4f}, KL: {train_metrics['kl']:6.2f}) | "
                      f"Val Loss: {val_loss:.4f}, RMSE: {val_rmse:.6f}")
            else:
                print(f"Epoch {epoch+1:3d}/{self.config.NUM_EPOCHS} | "
                      f"Train Loss: {train_loss:.4f}, RMSE: {train_rmse:.6f} | "
                      f"Val Loss: {val_loss:.4f}, RMSE: {val_rmse:.6f}")

            # Early stopping check
            # --------------------
            if val_loss < self.best_val_loss:
                # New best model
                self.best_val_loss = val_loss
                self.patience_counter = 0

                # Save checkpoint
                if save_path:
                    self.save_checkpoint(save_path)
                    print(f"  ✓ Best model saved (val_loss: {val_loss:.4f})")
            else:
                # No improvement
                self.patience_counter += 1

                if self.patience_counter >= self.config.EARLY_STOP_PATIENCE:
                    print(f"\nEarly stopping triggered at epoch {epoch+1}")
                    print(f"No improvement for {self.config.EARLY_STOP_PATIENCE} epochs")
                    break

        # Training complete
        print("\n" + "=" * 80)
        print(f"Training complete!")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
        print(f"Total epochs: {len(self.history['train_loss'])}")
        print("=" * 80 + "\n")

        return self.history

    def save_checkpoint(self, path):
        """
        Save model checkpoint.

        Saves:
        - Model state dict
        - Optimizer state dict
        - Training history
        - Best validation loss
        - Config parameters

        Args:
            path: Path to save checkpoint file
        """
        # Create directory if needed
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        # Prepare checkpoint
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'best_val_loss': self.best_val_loss,
            'config': {
                'model_type': self.model_type,
                'input_size': self.config.INPUT_SIZE,
                'hidden_size': self.config.HIDDEN_SIZE,
                'num_layers': self.config.NUM_LAYERS,
                'learning_rate': self.config.LEARNING_RATE
            }
        }

        # Save
        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        """
        Load model checkpoint.

        Restores:
        - Model weights
        - Optimizer state
        - Training history
        - Best validation loss

        Args:
            path: Path to checkpoint file
        """
        # Load checkpoint
        checkpoint = torch.load(path, map_location=self.device)

        # Restore state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        self.best_val_loss = checkpoint['best_val_loss']

        print(f"Checkpoint loaded from {path}")
        print(f"  Best val loss: {self.best_val_loss:.6f}")
        print(f"  Epochs trained: {len(self.history['train_loss'])}")


def main():
    """
    Test the Trainer with dummy data.

    Creates a small Standard LSTM and trains for a few epochs
    to verify the training loop works correctly.
    """
    print("=" * 80)
    print("Testing Trainer")
    print("=" * 80 + "\n")

    # Add parent to path for imports
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))

    from models.baseline_lstm import StandardLSTM
    from training.config import Config
    from data.dataset import StockDataset, get_dataloaders

    # Config
    config = Config()

    # Create dummy data
    print("Creating dummy data...")
    np.random.seed(42)

    X_train = np.random.randn(200, 20, 4)
    y_train = np.random.randn(200, 1)
    vix_train = np.random.randn(200, 1) * 5 + 20

    X_val = np.random.randn(50, 20, 4)
    y_val = np.random.randn(50, 1)
    vix_val = np.random.randn(50, 1) * 5 + 20

    # Create datasets
    train_data = {'X': X_train, 'y': y_train, 'vix': vix_train}
    val_data = {'X': X_val, 'y': y_val, 'vix': vix_val}

    # Create dataloaders
    train_loader, val_loader, _ = get_dataloaders(
        train_data, val_data, val_data,
        batch_size=32
    )

    # Create model
    print("Creating Standard LSTM model...")
    model = StandardLSTM(input_size=4, hidden_size=32, num_layers=1)

    # Create trainer
    print("Creating trainer...\n")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        model_type='standard_lstm'
    )

    # Train for 3 epochs (quick test)
    config.NUM_EPOCHS = 3
    config.EARLY_STOP_PATIENCE = 5

    history = trainer.train()

    # Print results
    print("\nTraining history:")
    print(f"  Train losses: {[f'{x:.4f}' for x in history['train_loss']]}")
    print(f"  Val losses:   {[f'{x:.4f}' for x in history['val_loss']]}")
    print(f"  Train RMSE:   {[f'{x:.4f}' for x in history['train_rmse']]}")
    print(f"  Val RMSE:     {[f'{x:.4f}' for x in history['val_rmse']]}")

    # Verify decreasing loss
    if history['train_loss'][-1] < history['train_loss'][0]:
        print("\n✓ Training loss decreased (model is learning)")
    else:
        print("\n⚠ Training loss did not decrease (may need more epochs or different data)")

    print("\n" + "=" * 80)
    print("Trainer test complete!")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
