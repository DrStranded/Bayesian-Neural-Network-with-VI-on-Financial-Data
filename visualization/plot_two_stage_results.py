"""
Plot results for Two-Stage Model:
  - Deterministic LSTM mean
  - Bayesian volatility head (aleatoric + epistemic)

Reads: results/predictions/all_two_stage.json
Saves: results/figures_two_stage/*.png
"""

import sys
from pathlib import Path

# Make project root importable
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import numpy as np
from visualization.plots import (
    plot_predictions_with_uncertainty,
    plot_uncertainty_decomposition,
    plot_uncertainty_vs_error,
    plot_vix_vs_prior,   # We will reuse this to show VIX vs sigma_aleatoric
)


def main():
    pred_dir = Path("results/predictions")
    fig_dir = Path("results/figures_two_stage")
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Load two-stage results
    with open(pred_dir / "all_two_stage.json", "r") as f:
        results = json.load(f)

    for key, data in results.items():
        ticker = data["ticker"]
        print(f"\nPlotting {ticker} (Two-Stage Model)...")

        # ----- Extract arrays -----
        y_true = np.array(data["targets"])              # (N, 1)
        y_pred = np.array(data["predictions"])          # (N, 1)
        sigma_ale = np.array(data["sigma_aleatoric"])   # (N, 1)
        sigma_epi = np.array(data["sigma_epistemic"])   # (N, 1)
        vix = np.array(data["vix"])                     # (N, 1) or (N,)

        # Flatten if needed
        y_true = y_true.reshape(-1)
        y_pred = y_pred.reshape(-1)
        sigma_ale = sigma_ale.reshape(-1)
        sigma_epi = sigma_epi.reshape(-1)
        vix = vix.reshape(-1)

        dates = np.arange(len(y_true))
        errors = np.abs(y_true - y_pred)

        # Total std for plotting (combine aleatoric + epistemic)
        # In our two-stage predict() we mainly trust aleatoric,
        # but for visualization it's natural to combine them:
        total_std = np.sqrt(sigma_ale**2 + sigma_epi**2)

        # ================== Figure 1: Full predictions with uncertainty ==================
        plot_predictions_with_uncertainty(
            dates,
            y_true,
            y_pred,
            total_std,
            title=f"{ticker}: Two-Stage Predictions with Uncertainty",
            save_path=fig_dir / f"{ticker}_two_stage_pred_full.png",
        )

        # ================== Figure 2: Zoom on first 200 points ==================
        zoom_n = min(200, len(dates))
        plot_predictions_with_uncertainty(
            dates[:zoom_n],
            y_true[:zoom_n],
            y_pred[:zoom_n],
            total_std[:zoom_n],
            title=f"{ticker}: Two-Stage Predictions (First {zoom_n} Days)",
            save_path=fig_dir / f"{ticker}_two_stage_pred_zoom.png",
        )

        # ================== Figure 3: Uncertainty decomposition over time ==================
        # epistemic vs aleatoric
        plot_uncertainty_decomposition(
            dates,
            sigma_epi,
            sigma_ale,
            title=f"{ticker}: Aleatoric vs Epistemic Uncertainty (Two-Stage)",
            save_path=fig_dir / f"{ticker}_two_stage_uncertainty_decomp.png",
        )

        # ================== Figure 4: Uncertainty vs absolute error ==================
        # Use total_std as x-axis to see if higher predicted uncertainty
        # corresponds to larger realized errors.
        plot_uncertainty_vs_error(
            total_std,
            errors,
            title=f"{ticker}: Uncertainty vs Error (Two-Stage)",
            save_path=fig_dir / f"{ticker}_two_stage_unc_vs_error.png",
        )

        # ================== Figure 5: VIX vs learned sigma (reuse plot_vix_vs_prior) ==================
        # Here we reuse the helper that used to plot "VIX vs prior std",
        # but now we feed in the *learned* aleatoric sigma instead of a prior.
        # So the second curve is effectively sigma_ale(t).
        plot_vix_vs_prior(
            dates,
            vix,
            sigma_ale.reshape(-1, 1),  # treat sigma_ale as a "prior-like" curve
            save_path=fig_dir / f"{ticker}_two_stage_vix_vs_sigma.png",
        )

        print(f"  ✓ Saved 5 plots for {ticker} in {fig_dir}")

    print(f"\nAll two-stage plots saved to {fig_dir}/")


if __name__ == "__main__":
    main()
