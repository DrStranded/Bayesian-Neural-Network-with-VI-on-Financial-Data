"""
Standard LSTM Baseline Model

This module implements a standard (non-Bayesian) LSTM model for stock price forecasting.
Serves as a baseline to compare against the Bayesian LSTM in terms of prediction
accuracy and uncertainty quantification.
"""

import torch
import torch.nn as nn


class StandardLSTM(nn.Module):
    """
    Standard (deterministic) LSTM model for time series forecasting.

    This model uses a standard LSTM architecture with deterministic weights.
    It serves as a baseline to compare against the Bayesian LSTM, demonstrating
    the added value of uncertainty quantification.

    Architecture:
    -------------
    Input [batch, seq_len, features]
      ↓
    LSTM layers (with dropout between layers if num_layers > 1)
      ↓
    Take last hidden state [batch, hidden_size]
      ↓
    Dropout
      ↓
    Linear layer → [batch, 1]

    Key Differences from Bayesian LSTM:
    ------------------------------------
    - Deterministic weights (single point estimate)
    - No weight uncertainty
    - No uncertainty quantification in predictions
    - No KL divergence term in loss
    - Faster training and inference
    - No need for multiple forward passes

    Advantages:
    -----------
    - Simpler to train (standard MSE loss)
    - Faster inference (single forward pass)
    - Well-established architecture
    - Good point prediction accuracy

    Limitations:
    ------------
    - No uncertainty estimates
    - Can be overconfident
    - No principled way to handle model uncertainty
    - Cannot distinguish between epistemic and aleatoric uncertainty

    Args:
        input_size: Number of input features (default: 4)
                   Expected features: [normalized_price, log_return,
                                      normalized_volume, volatility]
        hidden_size: LSTM hidden dimension (default: 64)
        num_layers: Number of stacked LSTM layers (default: 1)
        dropout: Dropout rate between LSTM layers (default: 0.2)
                Applied only if num_layers > 1
                Also applied after final LSTM layer

    Example:
        >>> model = StandardLSTM(input_size=4, hidden_size=64)
        >>> x = torch.randn(32, 20, 4)
        >>> output = model(x)
        >>> print(output.shape)
        torch.Size([32, 1])
    """

    def __init__(self, input_size=4, hidden_size=64, num_layers=1, dropout=0.2):
        """
        Initialize the Standard LSTM model.

        Args:
            input_size: Number of input features
            hidden_size: LSTM hidden state dimension
            num_layers: Number of LSTM layers to stack
            dropout: Dropout probability
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout

        # LSTM layer(s)
        # -------------
        # batch_first=True: input/output tensors are [batch, seq, feature]
        # dropout: Applied between LSTM layers (only if num_layers > 1)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Dropout layer
        # -------------
        # Applied after taking the last hidden state
        # Helps prevent overfitting
        self.dropout = nn.Dropout(dropout)

        # Output layer
        # ------------
        # Maps hidden state to single output value (next log return)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        Forward pass through the network.

        Process:
        --------
        1. Pass through LSTM layers
        2. Extract last hidden state (contains sequence information)
        3. Apply dropout for regularization
        4. Map to output via linear layer

        Args:
            x: Input tensor of shape [batch, seq_len, features]
               batch: Number of sequences in batch
               seq_len: Length of each sequence (e.g., 20 days)
               features: Number of features per timestep (e.g., 4)

        Returns:
            output: Predictions of shape [batch, 1]
                   Single prediction per sequence

        Note:
            This is deterministic - same input always gives same output
            (unless dropout is active during training)
        """
        # LSTM forward pass
        # -----------------
        # lstm_out: [batch, seq_len, hidden_size] - all hidden states
        # h_n: [num_layers, batch, hidden_size] - final hidden state
        # c_n: [num_layers, batch, hidden_size] - final cell state
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Take last timestep's hidden state
        # ----------------------------------
        # We use the last hidden state because it contains information
        # about the entire sequence (LSTM's memory)
        # Shape: [batch, hidden_size]
        last_hidden = lstm_out[:, -1, :]

        # Apply dropout for regularization
        # --------------------------------
        # During training: randomly zeros some elements
        # During eval: no-op (dropout is disabled)
        last_hidden = self.dropout(last_hidden)

        # Output layer
        # ------------
        # Map from hidden_size → 1 (next log return)
        output = self.fc(last_hidden)  # Shape: [batch, 1]

        return output

    def predict(self, x):
        """
        Make predictions (for consistency with other baseline models).

        This method provides a consistent interface with other models
        (MovingAverage, BayesianLSTM) that return (predictions, uncertainty).

        Process:
        --------
        1. Set model to eval mode (disables dropout)
        2. Disable gradient computation (saves memory)
        3. Forward pass
        4. Return predictions with None uncertainty

        Args:
            x: Input tensor [batch, seq_len, features]

        Returns:
            predictions: Tensor [batch, 1] - point predictions
            std: None - no uncertainty estimate for deterministic model

        Example:
            >>> model = StandardLSTM()
            >>> x = torch.randn(10, 20, 4)
            >>> pred, std = model.predict(x)
            >>> print(pred.shape, std)
            torch.Size([10, 1]) None
        """
        # Set to evaluation mode
        # Disables dropout and sets batch norm to eval mode (if present)
        self.eval()

        # Disable gradient computation for inference
        # Saves memory and computation
        with torch.no_grad():
            output = self.forward(x)

        # Return predictions and None for uncertainty
        # (consistent interface with probabilistic models)
        return output, None

    def __repr__(self):
        """
        String representation of the model.

        Returns:
            String showing architecture parameters
        """
        return (f"StandardLSTM(input_size={self.input_size}, "
                f"hidden_size={self.hidden_size}, "
                f"num_layers={self.num_layers})")


def main():
    """
    Test the StandardLSTM model.

    Demonstrates:
    - Model initialization
    - Parameter counting
    - Forward pass
    - Predict method
    - Output shapes and statistics
    """
    print("=" * 80)
    print("Testing StandardLSTM")
    print("=" * 80)

    # Create model with default parameters
    model = StandardLSTM(
        input_size=4,
        hidden_size=64,
        num_layers=1,
        dropout=0.2
    )
    print(f"\n{model}\n")

    # Count parameters
    # ----------------
    # Helps understand model complexity
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Parameter counts:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Break down by component
    print(f"\nParameter breakdown:")
    for name, param in model.named_parameters():
        print(f"  {name:20s}: {param.numel():6,} parameters, shape {list(param.shape)}")

    # Test forward pass
    print("\n" + "=" * 80)
    print("Testing Forward Pass")
    print("=" * 80)

    # Create dummy input
    batch_size = 32
    seq_len = 20
    features = 4
    x = torch.randn(batch_size, seq_len, features)

    # Forward pass
    model.train()  # Set to training mode (dropout active)
    output_train = model(x)

    print(f"\nTraining mode (dropout active):")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output_train.shape}")
    print(f"  Output mean: {output_train.mean().item():.6f}")
    print(f"  Output std: {output_train.std().item():.6f}")
    print(f"  Output min: {output_train.min().item():.6f}")
    print(f"  Output max: {output_train.max().item():.6f}")

    # Test predict method
    print("\n" + "=" * 80)
    print("Testing Predict Method")
    print("=" * 80)

    pred, std = model.predict(x)

    print(f"\nEvaluation mode (dropout disabled):")
    print(f"  Predictions shape: {pred.shape}")
    print(f"  Predictions mean: {pred.mean().item():.6f}")
    print(f"  Predictions std: {pred.std().item():.6f}")
    print(f"  Uncertainty estimate: {std}")

    # Test determinism in eval mode
    print("\n" + "=" * 80)
    print("Testing Determinism (Eval Mode)")
    print("=" * 80)

    pred1, _ = model.predict(x)
    pred2, _ = model.predict(x)

    diff = (pred1 - pred2).abs().max().item()
    print(f"\nTwo forward passes with same input:")
    print(f"  Max difference: {diff:.10f}")
    print(f"  Deterministic: {diff < 1e-6}")

    # Test stochasticity in train mode (due to dropout)
    print("\n" + "=" * 80)
    print("Testing Stochasticity (Train Mode with Dropout)")
    print("=" * 80)

    model.train()
    output1 = model(x)
    output2 = model(x)

    diff_train = (output1 - output2).abs().mean().item()
    print(f"\nTwo forward passes in training mode:")
    print(f"  Mean absolute difference: {diff_train:.6f}")
    print(f"  Stochastic due to dropout: {diff_train > 1e-6}")

    # Test with different architectures
    print("\n" + "=" * 80)
    print("Testing Different Architectures")
    print("=" * 80)

    architectures = [
        (1, 32),   # Small model
        (1, 64),   # Default
        (2, 64),   # Deeper
        (1, 128),  # Wider
    ]

    print(f"\n{'Layers':<8} {'Hidden':<8} {'Parameters':<15}")
    print("-" * 40)

    for num_layers, hidden_size in architectures:
        m = StandardLSTM(
            input_size=4,
            hidden_size=hidden_size,
            num_layers=num_layers
        )
        n_params = sum(p.numel() for p in m.parameters())
        print(f"{num_layers:<8} {hidden_size:<8} {n_params:>14,}")

    print("\n" + "=" * 80)
    print("StandardLSTM test complete!")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
