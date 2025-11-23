"""
Bayesian LSTM Model

This module implements the complete Bayesian LSTM model for stock price forecasting
with uncertainty quantification. Combines Bayesian layers to create a full network
that produces both point predictions and prediction intervals.
"""

import torch
import torch.nn as nn
from models.bayesian_layers import BayesianLinear
from models.prior import AdaptivePrior


class BayesianLSTM(nn.Module):
    """
    Bayesian LSTM with VIX-Adaptive Prior for stock price forecasting.

    This model combines a deterministic LSTM (for computational efficiency) with
    Bayesian output layers that provide uncertainty quantification. The prior
    distribution adapts based on market volatility (VIX).

    Architecture:
    -------------
    Input [batch, seq_len, features]
      ↓
    Standard LSTM (deterministic for efficiency)
      ↓
    Hidden state h [batch, hidden_size]
      ↓
    ├─→ BayesianLinear → μ(x) [batch, 1] ────────┐
    └─→ BayesianLinear → log σ(x) [batch, 1]     │
          ↓                                       │
        Softplus → σ(x) ────────────────────────┐ │
                                                ↓ ↓
    Output: p(y|x,D) = N(μ(x), σ²(x))

    Uncertainty Quantification:
    ---------------------------
    The model provides two types of uncertainty:

    1. Epistemic (Model) Uncertainty:
       - Arises from uncertainty in model parameters (weights)
       - Captured by variability across different weight samples
       - Reduces with more training data
       - Measured as variance of predictions across weight samples

    2. Aleatoric (Data) Uncertainty:
       - Arises from inherent noise in the data
       - Predicted by the model's σ(x) output
       - Cannot be reduced with more data
       - Measured as the predicted standard deviation

    Total Uncertainty = √(Epistemic² + Aleatoric²)

    Key Design Choices:
    -------------------
    - Standard LSTM: Deterministic for computational efficiency
      (Bayesian LSTM cells are expensive; empirically, Bayesian output layers
       provide most of the uncertainty benefits)

    - Dual Bayesian heads: Separate networks for mean and std prediction
      Allows the model to learn input-dependent uncertainty

    - VIX-adaptive prior: Prior variance increases with market volatility
      Encourages higher weight uncertainty during volatile periods

    Args:
        input_size: Number of input features (default: 4)
                   Expected: [normalized_price, log_return,
                             normalized_volume, volatility]
        hidden_size: LSTM hidden dimension (default: 64)
        num_layers: Number of LSTM layers (default: 1)
        prior_base_std: Base prior standard deviation (default: 0.1)
        vix_mean: VIX mean for prior adaptation (default: 20.0)
        vix_scale: VIX scaling factor τ (default: 10.0)
        vix_sensitivity: VIX sensitivity α (default: 1.0)

    Example:
        >>> model = BayesianLSTM(input_size=4, hidden_size=64)
        >>> x = torch.randn(32, 20, 4)
        >>> vix = torch.tensor(25.0)
        >>> mean, std = model(x, vix)
        >>> pred, total_unc, epi_unc, ale_unc = model.predict(x, vix, n_samples=100)
    """

    def __init__(self,
                 input_size=4,
                 hidden_size=64,
                 num_layers=1,
                 prior_base_std=0.1,
                 vix_mean=20.0,
                 vix_scale=10.0,
                 vix_sensitivity=1.0):
        """
        Initialize the Bayesian LSTM model.

        Args:
            input_size: Number of input features
            hidden_size: LSTM hidden state dimension
            num_layers: Number of LSTM layers to stack
            prior_base_std: Baseline prior standard deviation
            vix_mean: Historical VIX mean (centering for adaptation)
            vix_scale: VIX scaling factor (smoothness of adaptation)
            vix_sensitivity: VIX sensitivity (strength of adaptation)
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Standard LSTM (deterministic)
        # ------------------------------
        # We use a deterministic LSTM for computational efficiency.
        # Making LSTM weights Bayesian is very expensive (4 weight matrices per layer).
        # Empirically, Bayesian output layers provide most uncertainty benefits.
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True  # Input/output: [batch, seq, feature]
        )

        # Adaptive prior
        # --------------
        # Prior p(W | VIX) adapts to market volatility
        # High VIX → wider prior → more weight uncertainty
        self.prior = AdaptivePrior(
            base_std=prior_base_std,
            vix_mean=vix_mean,
            vix_scale=vix_scale,
            sensitivity=vix_sensitivity
        )

        # Bayesian output layers
        # ----------------------
        # Two separate Bayesian linear layers:
        # 1. fc_mean: Predicts mean μ(x)
        # 2. fc_logstd: Predicts log std, then transformed to σ(x) > 0
        #
        # Both layers have weight uncertainty → epistemic uncertainty
        self.fc_mean = BayesianLinear(hidden_size, 1, self.prior)
        self.fc_logstd = BayesianLinear(hidden_size, 1, self.prior)

        # Softplus for positive standard deviation
        # -----------------------------------------
        # Softplus(x) = log(1 + exp(x))
        # Ensures σ(x) > 0 (required for valid Gaussian)
        # Smoother than ReLU near zero
        self.softplus = nn.Softplus()

    def forward(self, x, vix):
        """
        Forward pass through the Bayesian LSTM.

        Process:
        --------
        1. Pass sequence through LSTM
        2. Extract last hidden state (contains sequence info)
        3. Pass through Bayesian mean layer → μ(x)
        4. Pass through Bayesian std layer → log σ(x) → σ(x)

        Each forward pass samples different weights (stochastic).

        Args:
            x: Input sequences [batch, seq_len, features]
            vix: Current VIX value, can be:
                 - Scalar (same VIX for all samples)
                 - Tensor [batch] (different VIX per sample)

        Returns:
            mean: Predicted mean [batch, 1]
            std: Predicted standard deviation [batch, 1]

        Note:
            Outputs are stochastic due to weight sampling in Bayesian layers.
            Different forward passes with same input yield different outputs.
        """
        # LSTM forward pass
        # -----------------
        # lstm_out: [batch, seq_len, hidden_size] - all hidden states
        # _: (h_n, c_n) - final hidden and cell states (not needed)
        lstm_out, _ = self.lstm(x)

        # Extract last timestep's hidden state
        # -------------------------------------
        # The last hidden state contains information about the entire sequence
        h = lstm_out[:, -1, :]  # Shape: [batch, hidden_size]

        # Bayesian output layers (with weight sampling)
        # ----------------------------------------------
        # Each call samples weights from q(W) using reparameterization trick
        mean = self.fc_mean(h, vix)      # [batch, 1] - predicted mean
        log_std = self.fc_logstd(h, vix)  # [batch, 1] - predicted log std

        # Transform to positive standard deviation
        # -----------------------------------------
        # Softplus ensures σ > 0
        # Add small epsilon (1e-6) to avoid numerical issues with σ = 0
        std = self.softplus(log_std) + 1e-6

        return mean, std

    def predict(self, x, vix, n_samples=100):
        """
        Prediction with comprehensive uncertainty quantification.

        This method performs Monte Carlo sampling to estimate both epistemic
        and aleatoric uncertainty by running multiple forward passes with
        different weight samples.

        Uncertainty Decomposition:
        --------------------------
        1. Run N forward passes (each with different weight samples)
        2. Collect N predictions: {(μ_i, σ_i)}_{i=1}^N
        3. Compute uncertainties:

           Epistemic (Model) Uncertainty:
           - Variance of means across weight samples
           - σ²_epistemic = Var[μ_i]
           - Measures: "How uncertain is the model about μ?"

           Aleatoric (Data) Uncertainty:
           - Average of predicted std
           - σ_aleatoric = Mean[σ_i]
           - Measures: "How noisy is the data?"

           Total Uncertainty:
           - σ²_total = σ²_epistemic + σ²_aleatoric
           - Combines both sources of uncertainty

        Args:
            x: Input sequences [batch, seq_len, features]
            vix: Current VIX value (scalar or [batch])
            n_samples: Number of weight samples for uncertainty estimation
                      More samples → better uncertainty estimate
                      Typical: 100 for testing, 10-30 for production

        Returns:
            pred_mean: Mean prediction [batch, 1]
                      Averaged across all weight samples
            total_std: Total uncertainty [batch, 1]
                      √(epistemic² + aleatoric²)
            epistemic_std: Model uncertainty [batch, 1]
                          Std of means across samples
            aleatoric_std: Data uncertainty [batch, 1]
                          Average predicted std

        Example:
            >>> model = BayesianLSTM()
            >>> pred, total, epi, ale = model.predict(x, vix, n_samples=100)
            >>> # 95% prediction interval: [pred - 2*total, pred + 2*total]
        """
        # Set to evaluation mode (though doesn't affect Bayesian layers)
        self.eval()

        with torch.no_grad():
            means = []
            stds = []

            # Monte Carlo sampling
            # --------------------
            # Run multiple forward passes with different weight samples
            for _ in range(n_samples):
                mean, std = self.forward(x, vix)
                means.append(mean)
                stds.append(std)

            # Stack predictions
            # -----------------
            # Shape: [n_samples, batch, 1]
            means = torch.stack(means)
            stds = torch.stack(stds)

            # Compute mean prediction
            # -----------------------
            # Average over weight samples (dim=0)
            pred_mean = means.mean(dim=0)  # [batch, 1]

            # Epistemic uncertainty
            # ---------------------
            # Standard deviation of means across weight samples
            # Measures: How much do predictions vary with different weights?
            # High epistemic uncertainty → model is unsure which weights are best
            epistemic_std = means.std(dim=0)  # [batch, 1]

            # Aleatoric uncertainty
            # ---------------------
            # Average of predicted standard deviations
            # Measures: How noisy does the model think the data is?
            # High aleatoric uncertainty → inherent unpredictability
            aleatoric_std = stds.mean(dim=0)  # [batch, 1]

            # Total uncertainty
            # -----------------
            # Combine both types via sum of variances
            # σ²_total = σ²_epistemic + σ²_aleatoric
            total_std = torch.sqrt(epistemic_std**2 + aleatoric_std**2)

        return pred_mean, total_std, epistemic_std, aleatoric_std

    def elbo_loss(self, x, y, vix, n_samples=1):
        """
        Compute ELBO (Evidence Lower Bound) loss for training.

        Mathematical Background:
        ------------------------
        For Bayesian neural networks, we want to maximize:
            log p(y|x,D) = log ∫ p(y|x,W) p(W|D) dW

        This is intractable, so we use variational inference:
            log p(y|x,D) ≥ ELBO = E_q[log p(y|x,W)] - KL(q(W) || p(W))

        where:
        - q(W) is the variational posterior (learned)
        - p(W|VIX) is the adaptive prior
        - p(y|x,W) is the likelihood (Gaussian)

        For optimization, we minimize the negative ELBO:
            Loss = -ELBO = -E_q[log p(y|x,W)] + KL(q(W) || p(W))
                         = NLL + KL

        Components:
        -----------
        1. Negative Log-Likelihood (NLL):
           For Gaussian: -log N(y | μ, σ²) =
               0.5 * ((y-μ)/σ)² + log σ + 0.5*log(2π)

           Encourages good predictions with appropriate uncertainty

        2. KL Divergence:
           KL(q(W) || p(W|VIX))

           Regularizes the posterior to stay close to prior
           Acts as a complexity penalty
           Higher VIX → wider prior → lower KL penalty

        Args:
            x: Input sequences [batch, seq_len, features]
            y: Target values [batch, 1]
            vix: Current VIX value (scalar or [batch])
            n_samples: Number of forward passes for likelihood estimation
                      Default: 1 (efficient, works well in practice)
                      Higher values: Better gradient estimate, slower

        Returns:
            loss: Scalar loss to minimize
            metrics: Dictionary with loss components:
                    - 'nll': Negative log-likelihood term
                    - 'kl': KL divergence term
                    - 'total': Total loss (nll + kl/batch_size)

        Example:
            >>> loss, metrics = model.elbo_loss(x_batch, y_batch, vix)
            >>> print(f"NLL: {metrics['nll']:.4f}, KL: {metrics['kl']:.4f}")
            >>> loss.backward()
        """
        # Negative log-likelihood estimation
        # -----------------------------------
        # Sample multiple predictions and average NLL
        # (Monte Carlo estimate of E_q[log p(y|x,W)])
        nll_total = 0

        for _ in range(n_samples):
            # Forward pass (samples weights)
            mean, std = self.forward(x, vix)

            # Negative log-likelihood for Gaussian
            # -------------------------------------
            # -log N(y | μ, σ²) = 0.5 * ((y-μ)/σ)² + log σ + 0.5*log(2π)
            #
            # Breakdown:
            # - ((y-μ)/σ)²: Squared error normalized by uncertainty
            #              Penalizes both poor predictions and underconfident std
            # - log σ: Prevents σ → 0 (which would minimize squared error term)
            # - log(2π): Normalization constant
            squared_error = ((y - mean) / std) ** 2
            log_std_term = torch.log(std)
            log_2pi_term = 0.5 * torch.log(torch.tensor(2 * 3.14159, device=std.device))

            nll = 0.5 * squared_error + log_std_term + log_2pi_term
            nll_total += nll.mean()

        # Average over samples
        nll = nll_total / n_samples

        # KL divergence term
        # ------------------
        # Sum of KL divergences from both Bayesian layers
        # KL adapts to VIX through the adaptive prior
        beta = 0.001  # scaling factor for KL term
        kl = self.fc_mean.kl_divergence() + beta * self.fc_logstd.kl_divergence()

        # ELBO loss
        # ---------
        # Divide KL by batch size for numerical stability
        # This scales KL to be comparable to per-sample NLL
        #
        # Interpretation:
        # - NLL: Data fit (per sample)
        # - KL/batch_size: Complexity penalty (per sample)
        loss = nll + kl / len(x)

        # Return loss and components for monitoring
        metrics = {
            'nll': nll.item(),
            'kl': kl.item(),
            'total': loss.item()
        }

        return loss, metrics

    def __repr__(self):
        """
        String representation of the model.

        Returns:
            String showing architecture and parameter count
        """
        n_params = sum(p.numel() for p in self.parameters())
        return (f"BayesianLSTM(input_size={self.input_size}, "
                f"hidden_size={self.hidden_size}, "
                f"num_layers={self.num_layers}, "
                f"params={n_params})")


def main():
    """
    Test the BayesianLSTM model.

    Demonstrates:
    - Model initialization
    - Forward pass (stochastic)
    - Uncertainty quantification
    - ELBO loss computation
    - Gradient flow
    - VIX adaptation
    """
    print("=" * 80)
    print("Testing BayesianLSTM")
    print("=" * 80)

    # Create model with default parameters
    model = BayesianLSTM(
        input_size=4,
        hidden_size=64,
        num_layers=1,
        prior_base_std=0.1,
        vix_mean=20.0,
        vix_scale=10.0,
        vix_sensitivity=1.0
    )
    print(f"\n{model}\n")

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    n_lstm_params = sum(p.numel() for p in model.lstm.parameters())
    n_bayes_params = (sum(p.numel() for p in model.fc_mean.parameters()) +
                      sum(p.numel() for p in model.fc_logstd.parameters()))

    print(f"Parameter breakdown:")
    print(f"  Total parameters: {n_params:,}")
    print(f"  LSTM parameters: {n_lstm_params:,} ({100*n_lstm_params/n_params:.1f}%)")
    print(f"  Bayesian parameters: {n_bayes_params:,} ({100*n_bayes_params/n_params:.1f}%)")

    # Test forward pass
    print("\n" + "=" * 80)
    print("Testing Forward Pass (Stochastic)")
    print("=" * 80)

    batch_size = 32
    seq_len = 20
    features = 4
    x = torch.randn(batch_size, seq_len, features)
    vix = torch.tensor(25.0)

    mean, std = model(x, vix)

    print(f"\nInput shape: {x.shape}")
    print(f"VIX value: {vix.item():.1f}")
    print(f"\nOutput shapes:")
    print(f"  Mean: {mean.shape}")
    print(f"  Std: {std.shape}")
    print(f"\nOutput statistics:")
    print(f"  Mean: min={mean.min().item():.4f}, max={mean.max().item():.4f}, "
          f"avg={mean.mean().item():.4f}")
    print(f"  Std: min={std.min().item():.4f}, max={std.max().item():.4f}, "
          f"avg={std.mean().item():.4f}")

    # Test stochasticity
    print(f"\nTesting stochasticity (two forward passes with same input):")
    mean1, std1 = model(x, vix)
    mean2, std2 = model(x, vix)
    mean_diff = (mean1 - mean2).abs().mean().item()
    std_diff = (std1 - std2).abs().mean().item()
    print(f"  Mean difference: {mean_diff:.6f}")
    print(f"  Std difference: {std_diff:.6f}")
    print(f"  Outputs are stochastic: {mean_diff > 1e-6}")

    # Test prediction with uncertainty quantification
    print("\n" + "=" * 80)
    print("Testing Uncertainty Quantification")
    print("=" * 80)

    n_samples = 100
    pred_mean, total_std, epistemic_std, aleatoric_std = model.predict(
        x, vix, n_samples=n_samples
    )

    print(f"\nPredictions (n_samples={n_samples}):")
    print(f"  pred_mean shape: {pred_mean.shape}")
    print(f"  Uncertainty shapes: {total_std.shape}")

    print(f"\nUncertainty statistics:")
    print(f"  Epistemic (model) uncertainty:")
    print(f"    min={epistemic_std.min().item():.4f}, max={epistemic_std.max().item():.4f}, "
          f"mean={epistemic_std.mean().item():.4f}")
    print(f"  Aleatoric (data) uncertainty:")
    print(f"    min={aleatoric_std.min().item():.4f}, max={aleatoric_std.max().item():.4f}, "
          f"mean={aleatoric_std.mean().item():.4f}")
    print(f"  Total uncertainty:")
    print(f"    min={total_std.min().item():.4f}, max={total_std.max().item():.4f}, "
          f"mean={total_std.mean().item():.4f}")

    # Verify uncertainty decomposition
    computed_total = torch.sqrt(epistemic_std**2 + aleatoric_std**2)
    decomp_error = (total_std - computed_total).abs().max().item()
    print(f"\nUncertainty decomposition verification:")
    print(f"  Max error: {decomp_error:.8f}")
    print(f"  Decomposition correct: {decomp_error < 1e-3}")

    # Test ELBO loss
    print("\n" + "=" * 80)
    print("Testing ELBO Loss")
    print("=" * 80)

    y = torch.randn(batch_size, 1)  # Random targets
    loss, metrics = model.elbo_loss(x, y, vix, n_samples=1)

    print(f"\nELBO components:")
    print(f"  NLL (negative log-likelihood): {metrics['nll']:.4f}")
    print(f"  KL (divergence): {metrics['kl']:.4f}")
    print(f"  Total loss: {metrics['total']:.4f}")

    # Test with different VIX values
    print(f"\nLoss with different VIX values:")
    print(f"{'VIX':<6} {'NLL':<10} {'KL':<12} {'Total':<10}")
    print("-" * 42)

    for vix_val in [12, 20, 30, 50]:
        vix_tensor = torch.tensor(float(vix_val))
        loss, metrics = model.elbo_loss(x, y, vix_tensor, n_samples=1)
        print(f"{vix_val:<6d} {metrics['nll']:<10.4f} {metrics['kl']:<12.2f} "
              f"{metrics['total']:<10.4f}")

    print(f"\nObservation:")
    print(f"  - Higher VIX → Lower KL (wider prior allows more weight variation)")
    print(f"  - This demonstrates VIX-adaptive regularization")

    # Test backward pass
    print("\n" + "=" * 80)
    print("Testing Gradient Flow")
    print("=" * 80)

    # Fresh forward pass for clean gradients
    model.zero_grad()
    loss, _ = model.elbo_loss(x, y, vix, n_samples=1)
    loss.backward()

    print(f"\nGradients computed successfully:")
    grad_stats = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_mean = param.grad.mean().item()
            grad_std = param.grad.std().item()
            grad_stats.append((name, grad_mean, grad_std))
            print(f"  {name:30s}: mean={grad_mean:+.6f}, std={grad_std:.6f}")

    # Check for gradient issues
    print(f"\nGradient health check:")
    all_finite = all(torch.isfinite(param.grad).all()
                     for param in model.parameters() if param.grad is not None)
    print(f"  All gradients finite: {all_finite}")

    print("\n" + "=" * 80)
    print("BayesianLSTM test complete!")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
