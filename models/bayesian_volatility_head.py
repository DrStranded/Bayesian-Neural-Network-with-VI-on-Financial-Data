"""
Bayesian Volatility Head

A variational Bayesian network for volatility prediction.
Takes LSTM hidden state + VIX as input, outputs σ_t with epistemic uncertainty.

Prior: p(θ) = N(0, σ²_prior)
Likelihood: ε_t | θ ~ N(0, σ_t²) where σ_t = g(h_t, VIX_t; θ)
Posterior: q(θ) = N(μ_q, σ²_q) via variational inference
Loss: NLL + KL[q(θ) || p(θ)]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from models.prior import AdaptivePrior


class BayesianLinear(nn.Module):
    """
    Bayesian Linear Layer with variational inference.
    Weights are distributions, not point estimates.
    """

    def __init__(self, in_features, out_features, prior_std=1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = prior_std

        # Variational posterior parameters: q(w) = N(mu, sigma²)
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_logsigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_logsigma = nn.Parameter(torch.Tensor(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize mu with Xavier, logsigma small (low initial uncertainty)
        nn.init.xavier_normal_(self.weight_mu)
        self.weight_logsigma.data.fill_(-4.0)  # sigma ≈ 0.018
        self.bias_mu.data.zero_()
        self.bias_logsigma.data.fill_(-4.0)

    def forward(self, x):
        """
        Forward pass with reparameterization trick.
        w = mu + sigma * epsilon, epsilon ~ N(0,1)
        """
        weight_sigma = torch.exp(self.weight_logsigma)
        bias_sigma = torch.exp(self.bias_logsigma)

        # Reparameterization trick
        weight = self.weight_mu + weight_sigma * torch.randn_like(self.weight_mu)
        bias = self.bias_mu + bias_sigma * torch.randn_like(self.bias_mu)

        return F.linear(x, weight, bias)

    def kl_divergence(self):
        """
        KL[q(w) || p(w)] for Gaussian prior p(w) = N(0, prior_std²)

        KL = log(σ_p/σ_q) + (σ_q² + μ_q²)/(2σ_p²) - 0.5
        """
        prior_var = self.prior_std ** 2

        weight_sigma = torch.exp(self.weight_logsigma)
        bias_sigma = torch.exp(self.bias_logsigma)

        kl_weight = (
            math.log(self.prior_std) - self.weight_logsigma +
            (weight_sigma**2 + self.weight_mu**2) / (2 * prior_var) - 0.5
        ).sum()

        kl_bias = (
            math.log(self.prior_std) - self.bias_logsigma +
            (bias_sigma**2 + self.bias_mu**2) / (2 * prior_var) - 0.5
        ).sum()

        return kl_weight + kl_bias


class BayesianVolatilityHead(nn.Module):
    """
    Bayesian head for volatility prediction.

    Input: [h_t, VIX_t] where h_t is LSTM hidden state
    Output: σ_t with epistemic uncertainty from weight sampling

    Architecture:
        [h_t, VIX_normalized] -> BayesianLinear(hidden+1, 32) -> ReLU
                              -> BayesianLinear(32, 1) -> Softplus -> σ_t
    """

    def __init__(self, hidden_size, prior_std=1, vix_mean=20.0, vix_scale=10.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.vix_mean = vix_mean
        self.vix_scale = vix_scale

        # Bayesian layers
        self.fc1 = BayesianLinear(hidden_size + 1, 32, prior_std)  # +1 for VIX
        self.fc2 = BayesianLinear(32, 1, prior_std)

        # For output transformation
        self.softplus = nn.Softplus()

    def forward(self, h, vix):
        """
        Forward pass.

        Args:
            h: LSTM hidden state [batch, hidden_size]
            vix: VIX values [batch] or scalar

        Returns:
            sigma: Predicted volatility [batch, 1]
        """
        batch_size = h.size(0)
        device = h.device
        dtype = h.dtype

        # Process VIX
        if not isinstance(vix, torch.Tensor):
            vix = torch.tensor(vix, device=device, dtype=dtype)
        if vix.dim() == 0:
            vix = vix.unsqueeze(0).expand(batch_size)
        if vix.dim() == 2:
            vix = vix.squeeze(-1)
        vix = vix.view(batch_size)

        # Normalize VIX
        vix_norm = ((vix - self.vix_mean) / self.vix_scale).unsqueeze(-1)  # [batch, 1]

        # Concatenate hidden state with VIX
        x = torch.cat([h, vix_norm], dim=-1)  # [batch, hidden_size + 1]

        # Forward through Bayesian layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # Ensure positive output, constrain to reasonable range [0.005, 0.15]
        sigma = 0.005 + 0.145 * torch.sigmoid(x)

        return sigma

    def kl_divergence(self):
        """Total KL divergence from both layers."""
        return self.fc1.kl_divergence() + self.fc2.kl_divergence()

    def predict(self, h, vix, n_samples=100):
        """
        Predict with uncertainty quantification.

        Returns:
            sigma_mean: E[σ_t] - expected volatility
            sigma_epistemic: Std[σ_t] - epistemic uncertainty on volatility
        """
        sigmas = []

        for _ in range(n_samples):
            sigma = self.forward(h, vix)
            sigmas.append(sigma)

        sigmas = torch.stack(sigmas)  # [n_samples, batch, 1]

        sigma_mean = sigmas.mean(dim=0)      # Aleatoric: expected σ
        sigma_epistemic = sigmas.std(dim=0)  # Epistemic: uncertainty about σ

        return sigma_mean, sigma_epistemic
