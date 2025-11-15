"""
Moving Average Baseline Model

This module implements simple moving average baseline models for stock price prediction.
Includes simple moving average (SMA) and exponential moving average (EMA) forecasting
to serve as baseline comparisons for more complex models.
"""

import numpy as np
import torch


class MovingAverageModel:
    """
    Simple Moving Average (SMA) baseline model.

    This model predicts the next value as the arithmetic mean of the past
    window values. It serves as a simple baseline to compare against more
    sophisticated models like LSTM and Bayesian LSTM.

    Mathematical Formulation:
    -------------------------
    For a sequence of past values [x_{t-w}, x_{t-w+1}, ..., x_{t-1}]:
        ŷ_t = (1/w) * Σ x_{t-i} for i in [1, w]

    where:
    - w is the window size
    - ŷ_t is the predicted value at time t

    Properties:
    -----------
    - No learnable parameters
    - Deterministic predictions (no uncertainty quantification)
    - Fast computation
    - Good baseline for trend-following strategies
    - Assumes recent past is informative about the future

    Limitations:
    ------------
    - No uncertainty estimates
    - Cannot capture complex patterns
    - Equally weights all values in window
    - No adaptation to changing volatility

    Args:
        window: Number of past values to average (default: 20)
                Typical values: 5-10 (short-term), 20 (medium-term), 50-200 (long-term)

    Example:
        >>> model = MovingAverageModel(window=20)
        >>> X = np.random.randn(100, 20, 4)
        >>> predictions, std = model.predict(X)
        >>> print(predictions.shape)
        (100, 1)
    """

    def __init__(self, window=20):
        """
        Initialize the Moving Average model.

        Args:
            window: Number of past values to average (must be > 0)
        """
        if window <= 0:
            raise ValueError(f"Window must be positive, got {window}")

        self.window = window

    def predict(self, X):
        """
        Predict using moving average of the target feature (log_return).

        The prediction is simply the mean of the last 'window' log returns.
        This is based on the assumption that the mean log return over the
        recent window is a good estimate of the next log return.

        Args:
            X: Input sequences of shape [N, seq_len, n_features]
               Can be numpy array or torch tensor
               Expected feature order:
                 - Feature 0: normalized_price
                 - Feature 1: log_return (this is what we predict)
                 - Feature 2: normalized_volume
                 - Feature 3: volatility

        Returns:
            predictions: numpy array of shape [N, 1]
                        Mean log return over the window
            std: None (no uncertainty estimate for this baseline)

        Example:
            >>> model = MovingAverageModel(window=10)
            >>> X = np.random.randn(50, 20, 4)
            >>> predictions, std = model.predict(X)
            >>> print(f"Shape: {predictions.shape}, Std: {std}")
            Shape: (50, 1), Std: None
        """
        # Convert torch tensor to numpy if needed
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()

        # Validate input shape
        if X.ndim != 3:
            raise ValueError(f"Expected 3D input [N, seq_len, features], got shape {X.shape}")

        N, seq_len, n_features = X.shape

        # Ensure we have enough data for the window
        if seq_len < self.window:
            raise ValueError(f"Sequence length {seq_len} is shorter than window {self.window}")

        # Extract log_return feature (index 1)
        # We take the last 'window' values from each sequence
        log_returns = X[:, -self.window:, 1]  # Shape: [N, window]

        # Compute simple moving average
        # For each sample, average the last 'window' log returns
        predictions = log_returns.mean(axis=1, keepdims=True)  # Shape: [N, 1]

        # No uncertainty estimate for this baseline
        # (Could compute std of window as naive uncertainty, but we return None)
        std = None

        return predictions, std

    def __repr__(self):
        """
        String representation of the model.

        Returns:
            String showing the window parameter
        """
        return f"MovingAverageModel(window={self.window})"


def main():
    """
    Test the MovingAverageModel.

    Demonstrates:
    - Model initialization
    - Prediction on dummy data
    - Handling of numpy and torch inputs
    - Output shapes and statistics
    """
    print("=" * 80)
    print("Testing MovingAverageModel")
    print("=" * 80)

    # Create model with default window
    model = MovingAverageModel(window=20)
    print(f"\n{model}\n")

    # Create dummy data
    # Shape: [N, seq_len, n_features]
    # N = 100 samples, seq_len = 20, n_features = 4
    np.random.seed(42)
    X_numpy = np.random.randn(100, 20, 4)

    print("Test 1: Numpy input")
    print("-" * 40)

    # Predict
    predictions, std = model.predict(X_numpy)

    print(f"Input shape: {X_numpy.shape}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions mean: {predictions.mean():.6f}")
    print(f"Predictions std: {predictions.std():.6f}")
    print(f"Predictions min: {predictions.min():.6f}")
    print(f"Predictions max: {predictions.max():.6f}")
    print(f"Uncertainty: {std}")

    # Test with torch tensor
    print("\n" + "=" * 80)
    print("Test 2: Torch tensor input")
    print("-" * 40)

    X_torch = torch.randn(50, 20, 4)
    predictions_torch, std_torch = model.predict(X_torch)

    print(f"Input shape: {X_torch.shape}")
    print(f"Input type: {type(X_torch)}")
    print(f"Predictions shape: {predictions_torch.shape}")
    print(f"Predictions type: {type(predictions_torch)}")
    print(f"Uncertainty: {std_torch}")

    # Test with different window sizes
    print("\n" + "=" * 80)
    print("Test 3: Different window sizes")
    print("-" * 40)

    for window in [5, 10, 20]:
        model = MovingAverageModel(window=window)
        preds, _ = model.predict(X_numpy)
        print(f"Window={window:2d}: mean prediction = {preds.mean():.6f}, "
              f"std = {preds.std():.6f}")

    # Demonstrate the averaging behavior
    print("\n" + "=" * 80)
    print("Test 4: Verify averaging behavior")
    print("-" * 40)

    # Create simple test case with known values
    X_test = np.zeros((1, 20, 4))
    X_test[0, :, 1] = np.arange(20)  # log_returns: 0, 1, 2, ..., 19

    model = MovingAverageModel(window=5)
    pred, _ = model.predict(X_test)

    # Last 5 values are [15, 16, 17, 18, 19]
    # Mean = (15 + 16 + 17 + 18 + 19) / 5 = 85 / 5 = 17.0
    expected = 17.0
    print(f"Last 5 log returns: {X_test[0, -5:, 1]}")
    print(f"Expected mean: {expected:.1f}")
    print(f"Predicted value: {pred[0, 0]:.1f}")
    print(f"Match: {np.isclose(pred[0, 0], expected)}")

    # Test error handling
    print("\n" + "=" * 80)
    print("Test 5: Error handling")
    print("-" * 40)

    try:
        model = MovingAverageModel(window=0)
        print("ERROR: Should have raised ValueError for window=0")
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")

    try:
        model = MovingAverageModel(window=10)
        X_short = np.random.randn(10, 5, 4)  # seq_len=5 < window=10
        model.predict(X_short)
        print("ERROR: Should have raised ValueError for short sequence")
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")

    try:
        X_wrong = np.random.randn(10, 20)  # 2D instead of 3D
        model.predict(X_wrong)
        print("ERROR: Should have raised ValueError for wrong dimensions")
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")

    print("\n" + "=" * 80)
    print("MovingAverageModel test complete!")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
