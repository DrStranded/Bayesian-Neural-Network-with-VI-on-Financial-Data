"""
Bayesian Neural Network Layers

This module implements Bayesian versions of common neural network layers.
Includes Bayesian Linear layers and Bayesian LSTM cells with weight uncertainty
using variational inference (mean-field approximation or more complex posteriors).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BayesianLinear(nn.Module):
    """
    Bayesian Linear Layer with Variational Inference.

    This layer maintains a distribution over weights rather than point estimates,
    enabling uncertainty quantification in predictions.

    Mathematical Background:
    ------------------------
    Instead of deterministic weights W, we maintain:
    - Posterior (variational) distribution: q(W) = N(μ_q, diag(σ²_q))
    - Prior distribution: p(W | VIX) = N(0, σ²_prior(VIX)) from AdaptivePrior

    Training:
    ---------
    We optimize the Evidence Lower Bound (ELBO):
        ELBO = E_q[log p(y|x,W)] - KL(q(W) || p(W|VIX))
             = Likelihood term - KL divergence term

    Reparameterization Trick:
    --------------------------
    To sample from q(W), we use: W = μ + σ ⊙ ε, where ε ~ N(0, I)
    This allows gradients to flow through μ and σ during backpropagation.

    Args:
        in_features: Number of input features
        out_features: Number of output features
        prior: AdaptivePrior instance providing p(W | VIX)

    Example:
        >>> from models.prior import AdaptivePrior
        >>> prior = AdaptivePrior(base_std=0.1)
        >>> layer = BayesianLinear(10, 5, prior)
        >>> x = torch.randn(32, 10)
        >>> output = layer(x, vix=20.0)
        >>> kl = layer.kl_divergence()
    """

    def __init__(self, in_features, out_features, prior):
        """
        Initialize the Bayesian linear layer.

        Args:
            in_features: Input dimension
            out_features: Output dimension
            prior: AdaptivePrior instance for p(W | VIX)
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior = prior

        
        # Posterior parameters (trainable)
        # --------------------------------
        # For weights W ~ N(μ_w, σ²_w), we maintain μ and log(σ)
        # Using log(σ) ensures σ > 0 through exp() transformation

        # Weight mean: μ_w ∈ R^(out_features × in_features)
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))

        # Weight log std: log(σ_w) ∈ R^(out_features × in_features)
        # Actual std is exp(weight_logsigma)
        self.weight_logsigma = nn.Parameter(torch.Tensor(out_features, in_features))

        # Bias mean: μ_b ∈ R^(out_features)
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))

        # Bias log std: log(σ_b) ∈ R^(out_features)
        self.bias_logsigma = nn.Parameter(torch.Tensor(out_features))

        # Initialize parameters
        self.reset_parameters()

        # Store current VIX for KL computation
        # This is set during forward pass
        self.current_vix = None

    def reset_parameters(self):
        """
        Initialize posterior parameters.

        Strategy:
        ---------
        - μ (mean): Use Xavier/Glorot uniform initialization
          This helps with gradient flow and prevents vanishing/exploding gradients
        - log(σ) (log std): Initialize to -3.0, giving σ ≈ 0.05
          Small initial uncertainty allows the model to start near deterministic
          and increase uncertainty as needed during training
        """
        # Xavier uniform initialization for mean parameters
        # stdv = 1 / sqrt(fan_in)
        stdv = 1.0 / math.sqrt(self.in_features)

        # Weight mean: uniform in [-stdv, stdv]
        self.weight_mu.data.uniform_(-stdv, stdv)

        # Weight log std: -3.0 → σ = exp(-3) ≈ 0.05 (small initial uncertainty)
        self.weight_logsigma.data.fill_(-3.0)

        # Bias: similar initialization
        self.bias_mu.data.uniform_(-stdv, stdv)
        self.bias_logsigma.data.fill_(-3.0)

    def forward(self, x, vix=None):
        """
        Forward pass with reparameterization trick.

        Process:
        --------
        1. Convert log(σ) to σ via exp()
        2. Sample ε ~ N(0, I)
        3. Compute W = μ + σ ⊙ ε (reparameterization trick)
        4. Compute output = xW^T + b

        This makes the sampling operation differentiable w.r.t. μ and σ.

        Args:
            x: Input tensor [batch_size, in_features]
            vix: Current VIX value (scalar or tensor)
                 Used for adaptive prior in KL computation

        Returns:
            output: [batch_size, out_features]

        Note:
            Each forward pass samples different weights, making predictions stochastic.
            During training, use single sample (efficient).
            During testing, average multiple samples for better uncertainty estimates.
        """
        # Store VIX for KL computation
        if vix is not None:
            self.current_vix = vix

        # Convert log(σ) to σ
        # σ = exp(log(σ)) ensures σ > 0
        weight_sigma = torch.exp(self.weight_logsigma)
        bias_sigma = torch.exp(self.bias_logsigma)

        # Reparameterization trick for weights
        # -------------------------------------
        # Sample ε ~ N(0, I) with same shape as weights
        epsilon_weight = torch.randn_like(self.weight_mu)

        # W = μ_w + σ_w ⊙ ε
        # The ⊙ denotes element-wise multiplication
        weight = self.weight_mu + weight_sigma * epsilon_weight

        # Reparameterization trick for bias
        # ----------------------------------
        epsilon_bias = torch.randn_like(self.bias_mu)
        bias = self.bias_mu + bias_sigma * epsilon_bias

        # Linear transformation: y = xW^T + b
        # F.linear computes x @ weight.T + bias
        output = F.linear(x, weight, bias)

        return output

    def kl_divergence(self):
        """
        Compute KL divergence: KL(q(W) || p(W|VIX))

        Mathematical Derivation:
        ------------------------
        For two Gaussians:
        - q(w) = N(μ_q, σ²_q)  [posterior]
        - p(w) = N(μ_p, σ²_p)  [prior]

        The KL divergence is:
            KL(q || p) = ∫ q(w) log(q(w)/p(w)) dw

        For Gaussians, this has a closed form:
            KL = log(σ_p/σ_q) + (σ²_q + (μ_q - μ_p)²) / (2σ²_p) - 1/2

        In our case, μ_p = 0 (zero-mean prior), so:
            KL = log(σ_p/σ_q) + (σ²_q + μ²_q) / (2σ²_p) - 1/2

        This can be rewritten as:
            KL = -log(σ_q) + log(σ_p) + (σ²_q + μ²_q) / (2σ²_p) - 1/2

        Physical Interpretation:
        ------------------------
        - log(σ_p/σ_q): Measures difference in uncertainty (entropy difference)
        - (σ²_q + μ²_q) / (2σ²_p): Measures how much posterior mass is "too far" from prior
        - -1/2: Normalization constant

        Returns:
            kl: Scalar tensor (sum over all weight and bias parameters)

        Note:
            The KL term acts as a regularizer, preventing the posterior from
            deviating too much from the prior. Higher VIX → wider prior → lower KL.
        """
        # Use default VIX if not set
        if self.current_vix is None:
            self.current_vix = self.prior.vix_mean

        # Get adaptive prior standard deviation based on current VIX
        # For high VIX, prior_std is larger → prior is more uncertain
        prior_std = self.prior.get_prior_std(self.current_vix)
        prior_var = prior_std ** 2

        # Posterior parameters
        # Convert log(σ) to σ
        weight_sigma = torch.exp(self.weight_logsigma)
        bias_sigma = torch.exp(self.bias_logsigma)

        # KL divergence for weights
        # --------------------------
        # For each weight w_ij:
        # KL(q(w_ij) || p(w_ij)) = log(σ_p/σ_q) + (σ²_q + μ²_q) / (2σ²_p) - 1/2
        #
        # We sum over all weights to get total KL
        kl_weight = (
            torch.log(prior_std / weight_sigma) +  # Entropy difference
            (weight_sigma**2 + self.weight_mu**2) / (2 * prior_var) -  # Distance from prior
            0.5  # Normalization constant
        ).sum()

        # KL divergence for bias
        # ----------------------
        # Same formula, applied to bias parameters
        kl_bias = (
            torch.log(prior_std / bias_sigma) +
            (bias_sigma**2 + self.bias_mu**2) / (2 * prior_var) -
            0.5
        ).sum()

        # Total KL = KL(weights) + KL(bias)
        return kl_weight + kl_bias

    def extra_repr(self):
        """
        Extra representation for printing.

        Returns:
            String with layer dimensions
        """
        return f'in_features={self.in_features}, out_features={self.out_features}'


def main():
    """
    Test the BayesianLinear layer.

    Demonstrates:
    - Layer initialization
    - Forward pass with reparameterization trick
    - Stochastic outputs (different each forward pass)
    - KL divergence computation
    - Effect of VIX on KL divergence
    """
    print("=" * 80)
    print("Testing BayesianLinear Layer")
    print("=" * 80)

    # Import prior
    from prior import AdaptivePrior

    # Create adaptive prior
    prior = AdaptivePrior(
        base_std=0.1,
        vix_mean=20.0,
        vix_scale=10.0,
        sensitivity=1.0
    )

    # Create Bayesian linear layer
    layer = BayesianLinear(in_features=10, out_features=5, prior=prior)
    print(f"\nLayer: {layer}")
    print(f"\nPosterior parameters:")
    print(f"  weight_mu shape    : {layer.weight_mu.shape}")
    print(f"  weight_logsigma shape: {layer.weight_logsigma.shape}")
    print(f"  bias_mu shape      : {layer.bias_mu.shape}")
    print(f"  bias_logsigma shape  : {layer.bias_logsigma.shape}")

    # Total number of parameters
    n_params = (layer.weight_mu.numel() + layer.weight_logsigma.numel() +
                layer.bias_mu.numel() + layer.bias_logsigma.numel())
    print(f"\nTotal trainable parameters: {n_params}")

    # Test forward pass
    print("\n" + "=" * 80)
    print("Testing Forward Pass (Reparameterization Trick)")
    print("=" * 80)

    # Create input batch
    batch_size = 32
    x = torch.randn(batch_size, 10)
    vix = 25.0

    print(f"\nInput shape: {x.shape}")
    print(f"VIX value: {vix}")

    # Multiple forward passes should give different outputs (stochastic)
    print(f"\nRunning 3 forward passes (should differ due to sampling):")
    outputs = []
    for i in range(3):
        output = layer(x, vix)
        outputs.append(output)
        print(f"  Pass {i+1}: shape={output.shape}, mean={output.mean().item():.4f}, "
              f"std={output.std().item():.4f}")

    # Check that outputs are different (stochastic due to weight sampling)
    diff_0_1 = (outputs[0] - outputs[1]).abs().mean().item()
    diff_1_2 = (outputs[1] - outputs[2]).abs().mean().item()
    print(f"\nMean absolute difference between passes:")
    print(f"  Pass 1 vs Pass 2: {diff_0_1:.4f}")
    print(f"  Pass 2 vs Pass 3: {diff_1_2:.4f}")
    print(f"Outputs are stochastic: {diff_0_1 > 1e-6}")

    # Test KL divergence
    print("\n" + "=" * 80)
    print("Testing KL Divergence KL(q(W) || p(W|VIX))")
    print("=" * 80)

    kl = layer.kl_divergence()
    print(f"\nKL divergence at VIX={vix}: {kl.item():.4f}")

    # Test with different VIX values
    print("\nKL divergence for different VIX values:")
    print("-" * 50)
    print(f"{'VIX':<10} {'KL Divergence':<20} {'Interpretation'}")
    print("-" * 50)

    for vix_val in [12, 20, 30, 50]:
        layer.current_vix = vix_val
        kl = layer.kl_divergence()

        # Interpret
        if vix_val < 20:
            interp = "Tighter prior → higher KL"
        elif vix_val == 20:
            interp = "Baseline prior"
        else:
            interp = "Wider prior → lower KL"

        print(f"{vix_val:<10d} {kl.item():<20.4f} {interp}")

    print(f"\nObservation:")
    print(f"  - Higher VIX → wider prior → lower KL penalty")
    print(f"  - Lower VIX → tighter prior → higher KL penalty")
    print(f"  - Model learns to balance likelihood and KL based on market conditions")

    # Test gradient flow
    print("\n" + "=" * 80)
    print("Testing Gradient Flow")
    print("=" * 80)

    # Create dummy loss
    x = torch.randn(32, 10)
    output = layer(x, vix=20.0)
    kl = layer.kl_divergence()

    # Simulate ELBO: likelihood - KL
    # (using dummy likelihood for demonstration)
    likelihood = -output.pow(2).mean()
    loss = -likelihood + 0.01 * kl  # ELBO = likelihood - KL

    # Backward pass
    loss.backward()

    # Check gradients exist
    print(f"\nGradients computed successfully:")
    print(f"  weight_mu grad: {layer.weight_mu.grad is not None}")
    print(f"  weight_logsigma grad: {layer.weight_logsigma.grad is not None}")
    print(f"  bias_mu grad: {layer.bias_mu.grad is not None}")
    print(f"  bias_logsigma grad: {layer.bias_logsigma.grad is not None}")

    if layer.weight_mu.grad is not None:
        print(f"\nGradient statistics:")
        print(f"  weight_mu grad mean: {layer.weight_mu.grad.mean().item():.6f}")
        print(f"  weight_logsigma grad mean: {layer.weight_logsigma.grad.mean().item():.6f}")

    print("\n" + "=" * 80)
    print("BayesianLinear layer test complete!")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
