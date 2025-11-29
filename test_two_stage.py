"""
Diagnostic tests for Two-Stage Model.
Run this to verify the model works correctly before training.
"""

import torch
import sys
sys.path.append('.')

from models.two_stage_model import TwoStageModel
from models.bayesian_volatility_head import BayesianVolatilityHead
from training.config import Config


def test_vix_gradient_flow():
    """Test 1: VIX must influence sigma (non-zero gradient wrt VIX)."""
    print("\n" + "=" * 60)
    print("TEST 1: VIX Gradient Flow to σ")
    print("=" * 60)

    config = Config()
    model = TwoStageModel(
        input_size=config.INPUT_SIZE,
        hidden_size=config.HIDDEN_SIZE,
    )
    model.train()  # ensure the graph is differentiable

    # Randomly generate one time-series input (batch=1, seq_len, input_size)
    x = torch.randn(1, config.SEQ_LEN, config.INPUT_SIZE)

    # Treat VIX as an input that requires gradient (batch size = 1)
    vix = torch.tensor([25.0], requires_grad=True)

    # Forward pass: if this runs, VIX is part of the forward graph
    _, sigma = model(x, vix)   # sigma: [1, 1]

    # Backpropagate from sigma to VIX
    loss = sigma.mean()        # make it a scalar
    loss.backward()

    print(f"VIX grad: {vix.grad}")

    passed = (vix.grad is not None) and (vix.grad.abs().sum().item() > 0)
    status = "✓ PASSED" if passed else "✗ FAILED"
    print(f"{status}: σ must depend on VIX (non-zero dσ/dVIX)")
    return passed



def test_epistemic_uncertainty():
    """Test 2: Multiple forward passes should give different σ."""
    print("\n" + "=" * 60)
    print("TEST 2: Epistemic Uncertainty (Weight Sampling)")
    print("=" * 60)

    config = Config()
    model = TwoStageModel(
        input_size=config.INPUT_SIZE,
        hidden_size=config.HIDDEN_SIZE
    )
    model.train()  # Enable sampling

    x = torch.randn(1, config.SEQ_LEN, config.INPUT_SIZE)
    vix = 25.0

    sigmas = []
    for i in range(10):
        _, sigma = model(x, vix)
        sigmas.append(sigma.item())
        print(f"  Sample {i+1}: σ = {sigma.item():.4f}")

    std_of_sigmas = torch.tensor(sigmas).std().item()
    print(f"\nStd of σ across samples: {std_of_sigmas:.6f}")

    passed = std_of_sigmas > 1e-6
    status = "✓ PASSED" if passed else "✗ FAILED"
    print(f"{status}: Different samples should give different σ")
    return passed


def test_stage_separation():
    """Test 3: Stage 1 and Stage 2 losses work correctly."""
    print("\n" + "=" * 60)
    print("TEST 3: Stage Separation")
    print("=" * 60)

    config = Config()
    model = TwoStageModel(
        input_size=config.INPUT_SIZE,
        hidden_size=config.HIDDEN_SIZE
    )

    x = torch.randn(8, config.SEQ_LEN, config.INPUT_SIZE)
    y = torch.randn(8, 1) * 0.02
    vix = torch.full((8,), 25.0)

    # Test Stage 1
    loss1, metrics1 = model.stage1_loss(x, y)
    print(f"Stage 1 Loss (MSE): {loss1.item():.6f}")

    # Freeze and test Stage 2
    model.freeze_mean_model()
    loss2, metrics2 = model.stage2_loss(x, y, vix)
    print(f"Stage 2 Loss (ELBO): {loss2.item():.4f}")
    print(f"  NLL: {metrics2['nll']:.4f}")
    print(f"  KL: {metrics2['kl']:.2f}")
    print(f"  Mean σ: {metrics2['mean_sigma']:.4f}")

    # Check gradients
    model.lstm.zero_grad()
    model.vol_head.zero_grad()
    loss2.backward()

    lstm_grad = model.lstm.weight_ih_l0.grad
    vol_grad = model.vol_head.fc1.weight_mu.grad

    lstm_frozen = lstm_grad is None or lstm_grad.abs().sum() == 0
    vol_has_grad = vol_grad is not None and vol_grad.abs().sum() > 0

    print(f"\nLSTM frozen (no grad): {lstm_frozen}")
    print(f"Vol head has grad: {vol_has_grad}")

    passed = lstm_frozen and vol_has_grad
    status = "✓ PASSED" if passed else "✗ FAILED"
    print(f"{status}: Stage 2 should only update vol_head")
    return passed


def test_predict_shapes():
    """Test 4: Predict method returns correct shapes."""
    print("\n" + "=" * 60)
    print("TEST 4: Predict Output Shapes")
    print("=" * 60)

    config = Config()
    model = TwoStageModel(
        input_size=config.INPUT_SIZE,
        hidden_size=config.HIDDEN_SIZE
    )

    batch_size = 16
    x = torch.randn(batch_size, config.SEQ_LEN, config.INPUT_SIZE)
    vix = torch.full((batch_size,), 25.0)

    mu, sigma_ale, sigma_epi, total = model.predict(x, vix, n_samples=50)

    print(f"Input shape: {x.shape}")
    print(f"μ shape: {mu.shape}")
    print(f"σ_aleatoric shape: {sigma_ale.shape}")
    print(f"σ_epistemic shape: {sigma_epi.shape}")

    expected = (batch_size, 1)
    passed = (mu.shape == expected and
              sigma_ale.shape == expected and
              sigma_epi.shape == expected)

    status = "✓ PASSED" if passed else "✗ FAILED"
    print(f"\n{status}: All outputs should be [{batch_size}, 1]")
    return passed


def test_sigma_range():
    """Test 5: σ should be in reasonable range."""
    print("\n" + "=" * 60)
    print("TEST 5: σ Range Check")
    print("=" * 60)

    config = Config()
    model = TwoStageModel(
        input_size=config.INPUT_SIZE,
        hidden_size=config.HIDDEN_SIZE
    )
    model.eval()

    x = torch.randn(100, config.SEQ_LEN, config.INPUT_SIZE)
    vix = torch.rand(100) * 40 + 10  # VIX between 10-50

    with torch.no_grad():
        _, sigma = model(x, vix)

    min_sigma = sigma.min().item()
    max_sigma = sigma.max().item()
    mean_sigma = sigma.mean().item()

    print(f"σ range: [{min_sigma:.4f}, {max_sigma:.4f}]")
    print(f"σ mean: {mean_sigma:.4f}")
    print(f"Expected range: [0.005, 0.15]")

    passed = min_sigma >= 0.004 and max_sigma <= 0.16
    status = "✓ PASSED" if passed else "✗ FAILED"
    print(f"\n{status}: σ should be in [0.005, 0.15]")
    return passed


def main():
    print("=" * 60)
    print("TWO-STAGE MODEL DIAGNOSTIC TESTS")
    print("=" * 60)

    results = []
    results.append(("VIX Effect", test_vix_gradient_flow()))
    results.append(("Epistemic Uncertainty", test_epistemic_uncertainty()))
    results.append(("Stage Separation", test_stage_separation()))
    results.append(("Output Shapes", test_predict_shapes()))
    results.append(("Sigma Range", test_sigma_range()))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✓" if passed else "✗"
        print(f"  {status} {name}")
        if not passed:
            all_passed = False

    print("\n" + ("ALL TESTS PASSED ✓" if all_passed else "SOME TESTS FAILED ✗"))
    return all_passed


if __name__ == '__main__':
    main()
