"""
MC Dropout LSTM - Based on Gal & Ghahramani (2016)
Dropout as a Bayesian Approximation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MCDropoutLSTM(nn.Module):
    """
    LSTM with MC Dropout for uncertainty quantification

    Reference:
    - Gal & Ghahramani (2016): Dropout as a Bayesian Approximation
    - Zhu & Laptev (2017): Deep and Confident Prediction for Time Series at Uber
    """

    def __init__(self, input_size, hidden_size, dropout_rate=0.2, device='cpu'):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.device = device

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=1
        )

        # Dropout (key for Bayesian approximation)
        self.dropout = nn.Dropout(dropout_rate)

        # Output layer
        self.fc = nn.Linear(hidden_size, 1)

        # Aleatoric uncertainty (learnable parameter)
        # log_var to ensure positivity: var = exp(log_var)
        self.log_var = nn.Parameter(torch.tensor(-4.0))

        self.to(device)

    def forward(self, x):
        """
        Single forward pass

        Args:
            x: [batch, seq_len, input_size]
        Returns:
            mu: [batch, 1] predicted mean
        """
        # LSTM
        h, _ = self.lstm(x)  # [batch, seq_len, hidden]
        h = h[:, -1, :]  # Take last timestep [batch, hidden]

        # Dropout (stays on even during eval for MC sampling)
        h = self.dropout(h)

        # Output
        mu = self.fc(h)  # [batch, 1]

        return mu

    def loss(self, x, y):
        """
        Heteroscedastic loss (Kendall & Gal 2017)

        Loss = 0.5 * exp(-log_var) * (y - mu)^2 + 0.5 * log_var

        This balances prediction accuracy and learned noise level
        """
        mu = self.forward(x)

        # Precision (inverse variance)
        precision = torch.exp(-self.log_var)

        # Negative log-likelihood
        nll = 0.5 * precision * (y - mu) ** 2 + 0.005 * self.log_var

        return nll.mean()

    def predict_with_uncertainty(self, x, n_samples=100):
        """
        MC Dropout sampling for uncertainty quantification

        Args:
            x: [batch, seq_len, input_size]
            n_samples: number of MC samples

        Returns:
            mu_mean: [batch, 1] predictive mean
            epistemic: [batch, 1] epistemic uncertainty (model)
            aleatoric: [batch, 1] aleatoric uncertainty (data noise)
            total: [batch, 1] total uncertainty
        """
        # Enable dropout for MC sampling
        self.train()

        # Collect predictions from multiple dropout masks
        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                mu = self.forward(x)
                predictions.append(mu)

        predictions = torch.stack(predictions)  # [n_samples, batch, 1]

        # Epistemic uncertainty (model uncertainty)
        mu_mean = predictions.mean(dim=0)  # [batch, 1]
        epistemic = predictions.std(dim=0)  # [batch, 1]

        # Aleatoric uncertainty (data noise)
        aleatoric = torch.exp(0.5 * self.log_var).expand_as(mu_mean)

        # Total uncertainty
        total = torch.sqrt(epistemic ** 2 + aleatoric ** 2)

        return mu_mean, epistemic, aleatoric, total


def create_mc_dropout_lstm(config):
    """Factory function to create model"""
    return MCDropoutLSTM(
        input_size=config.INPUT_SIZE,
        hidden_size=config.HIDDEN_SIZE,
        dropout_rate=0.2,
        device=config.DEVICE
    )
