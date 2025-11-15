"""
Prior Distributions Module

This module defines prior distributions for Bayesian neural network weights.
Implements various prior specifications (Gaussian, scale mixtures, etc.)
and utilities for prior initialization in Bayesian layers.
"""

import torch
import math


class AdaptivePrior:
    """
    Volatility-adaptive prior distribution for Bayesian neural network weights.

    The prior distribution adjusts its variance based on market volatility (VIX),
    allowing the model to have higher weight uncertainty during volatile periods.

    Mathematical Formulation:
    ------------------------
    Prior: p(W | VIX) = N(0, σ²_prior(VIX))

    where the adaptive standard deviation is:
        σ_prior(VIX) = σ_base * [1 + α * tanh((VIX - μ_VIX) / τ)]

    Parameters:
    -----------
    - σ_base (base_std): Baseline standard deviation when VIX = μ_VIX
    - μ_VIX (vix_mean): Historical mean VIX (centering parameter)
    - τ (vix_scale): Scaling factor controlling sensitivity to VIX changes
    - α (sensitivity): Controls the magnitude of adaptation

    Properties:
    -----------
    - When VIX is low (calm markets): σ_prior decreases → tighter prior → less uncertainty
    - When VIX is high (volatile markets): σ_prior increases → wider prior → more uncertainty
    - The tanh function ensures smooth, bounded adaptation between -1 and +1

    Example:
    --------
    >>> prior = AdaptivePrior(base_std=0.1, vix_mean=20.0, sensitivity=1.0)
    >>> # Low volatility
    >>> std_low = prior.get_prior_std(12.0)  # VIX = 12
    >>> # High volatility
    >>> std_high = prior.get_prior_std(40.0)  # VIX = 40
    >>> print(f"Low VIX std: {std_low:.4f}, High VIX std: {std_high:.4f}")

    Args:
        base_std: Baseline standard deviation σ_base (default: 0.1)
        vix_mean: Historical VIX mean μ_VIX (default: 20.0)
        vix_scale: Scaling factor τ (default: 10.0)
        sensitivity: Sensitivity parameter α (default: 1.0)
    """

    def __init__(self, base_std=0.1, vix_mean=20.0, vix_scale=10.0, sensitivity=1.0):
        """
        Initialize the adaptive prior with volatility-dependent parameters.

        Args:
            base_std: Baseline standard deviation (used when VIX = vix_mean)
            vix_mean: Historical VIX mean (center point for adaptation)
            vix_scale: Scaling factor controlling smoothness of transition
            sensitivity: Controls strength of VIX-based adaptation
        """
        self.base_std = base_std
        self.vix_mean = vix_mean
        self.vix_scale = vix_scale
        self.sensitivity = sensitivity

    def get_prior_std(self, vix_value):
        """
        Compute prior standard deviation based on current VIX value.

        Mathematical Formula:
        --------------------
        σ_prior = σ_base * [1 + α * tanh((VIX - μ_VIX) / τ)]

        The tanh function provides:
        - Smooth transition between low and high volatility regimes
        - Bounded adaptation (tanh ∈ [-1, 1])
        - When VIX << μ_VIX: tanh → -1, so σ → σ_base * (1 - α)
        - When VIX = μ_VIX: tanh → 0, so σ → σ_base
        - When VIX >> μ_VIX: tanh → +1, so σ → σ_base * (1 + α)

        Args:
            vix_value: Current VIX value (scalar, float, or torch.Tensor)
                      Can be a single value or batch of values

        Returns:
            prior_std: torch.Tensor containing the adaptive standard deviation(s)

        Example:
            >>> prior = AdaptivePrior(base_std=0.1, vix_mean=20.0, sensitivity=1.0)
            >>> std = prior.get_prior_std(30.0)  # VIX = 30
            >>> print(std)
            tensor(0.1760)
        """
        # Convert to tensor if needed (handles scalars, lists, numpy arrays)
        if not isinstance(vix_value, torch.Tensor):
            vix_value = torch.tensor(vix_value, dtype=torch.float32)

        # Handle NaN values: replace with vix_mean as a safe default
        # This is important for missing VIX data in real-world scenarios
        if torch.isnan(vix_value).any():
            vix_value = torch.where(
                torch.isnan(vix_value),
                torch.tensor(self.vix_mean, dtype=torch.float32),
                vix_value
            )

        # Normalize VIX relative to historical mean
        # This centers the adaptation around the typical VIX value
        normalized_vix = (vix_value - self.vix_mean) / self.vix_scale

        # Compute adaptive scaling factor using tanh
        # tanh squashes the normalized VIX to [-1, 1] for bounded adaptation
        scale_factor = 1.0 + self.sensitivity * torch.tanh(normalized_vix)

        # Apply scaling to base standard deviation
        prior_std = self.base_std * scale_factor

        return prior_std

    def log_prior(self, weight, vix_value):
        """
        Compute log prior probability: log p(W | VIX).

        Mathematical Formulation:
        -------------------------
        For a Gaussian prior N(0, σ²):
            p(w) = (1 / √(2πσ²)) * exp(-w² / (2σ²))

        Taking the log:
            log p(w) = -0.5 * log(2πσ²) - w² / (2σ²)
                     = -0.5 * (w²/σ² + log(2πσ²))

        For multiple weights, we sum the log probabilities:
            log p(W) = Σ log p(w_i) = Σ [-0.5 * (w_i²/σ² + log(2πσ²))]

        This is used in the ELBO (Evidence Lower Bound) for variational inference:
            ELBO = E_q[log p(y|x,W)] - KL(q(W) || p(W))
        where p(W) is our adaptive prior.

        Args:
            weight: Weight tensor of any shape [d1, d2, ..., dn]
                   These are the neural network weights to evaluate
            vix_value: Current VIX value (scalar or tensor)
                      Determines the prior variance

        Returns:
            log_prob: Scalar tensor representing the sum of log probabilities
                     over all weight elements

        Example:
            >>> prior = AdaptivePrior()
            >>> weights = torch.randn(10, 5)  # 50 weights
            >>> vix = 25.0
            >>> log_p = prior.log_prior(weights, vix)
            >>> print(log_p.shape)
            torch.Size([])  # Scalar
        """
        # Get adaptive prior standard deviation based on current VIX
        prior_std = self.get_prior_std(vix_value)

        # Compute variance (σ²)
        prior_var = prior_std ** 2

        # Compute Gaussian log probability for each weight
        # log p(w) = -0.5 * (w²/σ² + log(2πσ²))
        log_prob = -0.5 * (
            (weight ** 2) / prior_var +           # Mahalanobis distance term
            torch.log(2 * math.pi * prior_var)    # Normalization constant
        )

        # Sum over all weights to get total log prior
        # This is the standard practice in Bayesian neural networks
        return log_prob.sum()

    def __repr__(self):
        """
        String representation of the AdaptivePrior.

        Returns:
            String showing all parameter values
        """
        return (f"AdaptivePrior(base_std={self.base_std}, "
                f"vix_mean={self.vix_mean}, "
                f"vix_scale={self.vix_scale}, "
                f"sensitivity={self.sensitivity})")


def main():
    """
    Test the AdaptivePrior class.

    Demonstrates:
    - How prior std adapts to different VIX values
    - Log prior computation for weight tensors
    - Effect of market volatility on weight uncertainty
    """
    print("=" * 80)
    print("Testing AdaptivePrior")
    print("=" * 80)

    # Create prior with default parameters from Config
    prior = AdaptivePrior(
        base_std=0.1,      # Baseline uncertainty
        vix_mean=20.0,     # Historical VIX average
        vix_scale=10.0,    # Scaling factor
        sensitivity=1.0    # Full sensitivity to VIX
    )
    print(f"\n{prior}\n")

    # Test with different VIX values representing different market conditions
    vix_values = [12, 20, 30, 50]
    print("Prior standard deviation for different VIX values:")
    print("-" * 40)
    print(f"{'VIX':<10} {'σ_prior':<15} {'Interpretation'}")
    print("-" * 40)

    for vix in vix_values:
        std = prior.get_prior_std(vix)

        # Interpret the VIX level
        if vix < 15:
            interp = "Very low volatility"
        elif vix < 20:
            interp = "Low volatility"
        elif vix < 30:
            interp = "Normal volatility"
        else:
            interp = "High volatility"

        print(f"{vix:<10d} {std.item():<15.4f} {interp}")

    print(f"\nObservation:")
    print(f"  - As VIX increases, prior std increases → model becomes more uncertain")
    print(f"  - As VIX decreases, prior std decreases → model becomes more certain")
    print(f"  - This adaptive behavior helps with uncertainty quantification")

    # Test log prior computation
    print("\n" + "=" * 80)
    print("Testing log_prior computation")
    print("=" * 80)

    # Create a random weight matrix (simulating part of a neural network)
    torch.manual_seed(42)
    weight = torch.randn(10, 10)  # 100 weights
    vix = 25.0

    print(f"\nWeight tensor shape: {weight.shape}")
    print(f"Number of weights: {weight.numel()}")
    print(f"VIX value: {vix}")

    # Compute log prior
    log_p = prior.log_prior(weight, vix)

    print(f"\nLog prior p(W | VIX={vix}): {log_p.item():.4f}")

    # Compare with different VIX values
    print(f"\nLog prior for different VIX values:")
    print("-" * 40)
    for vix in [12, 20, 30, 50]:
        log_p = prior.log_prior(weight, vix)
        print(f"  VIX = {vix:2d} → log p(W) = {log_p.item():8.2f}")

    print(f"\nObservation:")
    print(f"  - Higher VIX → higher (less negative) log prior")
    print(f"  - This means the same weights are more 'plausible' during volatile periods")
    print(f"  - The prior adapts to market conditions automatically")

    print("\n" + "=" * 80)
    print("AdaptivePrior test complete!")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
