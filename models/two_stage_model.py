"""
Two-Stage Model: Deterministic LSTM (mean) + Bayesian Volatility Head (std)

Stage 1: μ_t = f_LSTM(x_{1:t}) - trained with MSE, then frozen
Stage 2: σ_t = g_Bayesian(h_t, VIX_t; θ) - trained with NLL + KL on residuals

Final output: p(y_t | x) = N(μ_t, σ_t²)
"""

import torch
import torch.nn as nn
import math

from models.bayesian_volatility_head import BayesianVolatilityHead


class TwoStageModel(nn.Module):
    """
    Two-stage model for stock return prediction with Bayesian uncertainty.

    Architecture:
        Input [batch, seq_len, features]
            |
            v
        LSTM (deterministic) --> h_t [batch, hidden_size]
            |                       |
            v                       v
        fc_mean --> μ_t      [h_t, VIX] --> BayesianVolHead --> σ_t
            |                                   |
            v                                   v
        Point prediction              Epistemic uncertainty on σ
    """

    def __init__(self,
                 input_size=4,
                 hidden_size=64,
                 num_layers=1,
                 dropout=0.2,
                 prior_std=0.1,
                 vix_mean=20.0,
                 vix_scale=10.0,
                 kl_weight=1e-4):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kl_weight = kl_weight

        # Stage 1: LSTM + deterministic mean head
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.post_lstm = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.fc_mean = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )


        # Stage 2: Bayesian volatility head
        self.vol_head = BayesianVolatilityHead(
            hidden_size=hidden_size,
            prior_std=prior_std,
            vix_mean=vix_mean,
            vix_scale=vix_scale
        )

        # Track training stage
        self.stage = 1  # 1 = train mean, 2 = train volatility

    def freeze_mean_model(self):
        """Freeze Stage 1 (LSTM + mean head) after training."""
        for param in self.lstm.parameters():
            param.requires_grad = False
        for param in self.fc_mean.parameters():
            param.requires_grad = False
        self.stage = 2
        print("Stage 1 (mean model) frozen. Now training Stage 2 (volatility head).")

    def unfreeze_mean_model(self):
        """Unfreeze Stage 1 (for fine-tuning if needed)."""
        for param in self.lstm.parameters():
            param.requires_grad = True
        for param in self.fc_mean.parameters():
            param.requires_grad = True
        self.stage = 1

    def get_hidden_state(self, x):
        """Extract enriched LSTM hidden state."""
        lstm_out, _ = self.lstm(x)
        h_last = lstm_out[:, -1, :]          # [batch, hidden_size]
        h = self.post_lstm(h_last)           # [batch, hidden_size]
        return h

    def forward(self, x, vix):
        """
        Full forward pass.

        Args:
            x: Input [batch, seq_len, features]
            vix: VIX values [batch] or scalar

        Returns:
            mu: Predicted mean [batch, 1]
            sigma: Predicted std [batch, 1]
        """
        h = self.get_hidden_state(x)
        mu = self.fc_mean(h)
        sigma = self.vol_head(h, vix)
        return mu, sigma

    def stage1_loss(self, x, y):
        """
        Stage 1 loss: MSE for mean prediction.
        Train LSTM + fc_mean to predict returns.
        """
        h = self.get_hidden_state(x)
        mu = self.fc_mean(h)
        loss = nn.functional.mse_loss(mu, y)
        return loss, {'mse': loss.item(), 'rmse': math.sqrt(loss.item())}

    def stage2_loss(self, x, y, vix):
        """
        Stage 2 loss: NLL + KL for Bayesian volatility head.
        Train on residuals: ε_t = y_t - μ_t

        NLL = 0.5 * (ε/σ)² + log(σ)
        Loss = NLL + kl_weight * KL
        """
        with torch.no_grad():
            h = self.get_hidden_state(x)
            mu = self.fc_mean(h)
            residual = y - mu  # ε_t = y_t - μ_t

        # Need fresh h for gradient flow through vol_head
        h = self.get_hidden_state(x)
        sigma = self.vol_head(h, vix)

        # Gaussian NLL: -log N(ε | 0, σ²)
        nll = 0.5 * (residual / sigma) ** 2 + torch.log(sigma) + 0.5 * math.log(2 * math.pi)
        nll = nll.mean()

        # KL divergence
        kl = self.vol_head.kl_divergence()
        kl_scaled = self.kl_weight * kl / x.size(0)

        # Total loss
        loss = nll + kl_scaled

        metrics = {
            'nll': nll.item(),
            'kl': kl.item(),
            'kl_scaled': kl_scaled.item(),
            'total': loss.item(),
            'mean_sigma': sigma.mean().item(),
            'mean_residual_sq': (residual ** 2).mean().item()
        }

        return loss, metrics

    def predict(self, x, vix, n_samples=100):
        """
        Predict with full uncertainty quantification.

        Returns:
            mu: Mean prediction [batch, 1]
            sigma_aleatoric: Expected σ (data uncertainty) [batch, 1]
            sigma_epistemic: Std of σ (model uncertainty about volatility) [batch, 1]
            total_std: Combined uncertainty [batch, 1]
        """
        self.eval()

        with torch.no_grad():
            h = self.get_hidden_state(x)
            mu = self.fc_mean(h)

            sigma_aleatoric, sigma_epistemic = self.vol_head.predict(h, vix, n_samples)

            # Total uncertainty combines both
            # For 95% CI width consideration
            total_std = sigma_aleatoric  # Primary uncertainty is aleatoric

        return mu, sigma_aleatoric, sigma_epistemic, total_std


def test_two_stage_model():
    """Test the two-stage model."""
    print("=" * 70)
    print("Testing TwoStageModel")
    print("=" * 70)

    model = TwoStageModel(
        input_size=4,
        hidden_size=64,
        prior_std=0.1,
        kl_weight=1e-4
    )

    batch_size = 32
    seq_len = 20
    x = torch.randn(batch_size, seq_len, 4)
    y = torch.randn(batch_size, 1) * 0.02
    vix = torch.full((batch_size,), 25.0)

    # Test Stage 1
    print("\n--- Stage 1: Mean Model ---")
    loss1, metrics1 = model.stage1_loss(x, y)
    print(f"MSE: {metrics1['mse']:.6f}, RMSE: {metrics1['rmse']:.6f}")

    # Freeze and test Stage 2
    model.freeze_mean_model()
    print("\n--- Stage 2: Bayesian Volatility Head ---")
    loss2, metrics2 = model.stage2_loss(x, y, vix)
    print(f"NLL: {metrics2['nll']:.4f}, KL: {metrics2['kl']:.2f}")
    print(f"Mean σ: {metrics2['mean_sigma']:.4f}")

    # Test VIX effect
    print("\n--- VIX Effect on σ ---")
    model.eval()
    with torch.no_grad():
        h = model.get_hidden_state(x[:1])
        for vix_val in [10, 20, 30, 40, 50]:
            sigma = model.vol_head(h, float(vix_val))
            print(f"VIX={vix_val}: σ={sigma.item():.4f}")

    # Test uncertainty quantification
    print("\n--- Uncertainty Quantification ---")
    mu, sigma_ale, sigma_epi, total = model.predict(x, vix, n_samples=50)
    print(f"μ shape: {mu.shape}")
    print(f"σ_aleatoric mean: {sigma_ale.mean().item():.4f}")
    print(f"σ_epistemic mean: {sigma_epi.mean().item():.4f}")

    print("\n" + "=" * 70)
    print("Test complete!")
    print("=" * 70)


if __name__ == '__main__':
    test_two_stage_model()
