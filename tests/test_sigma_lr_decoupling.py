"""Regression test for σ/μ LR decoupling.

Confirms that sweeping E_sigma_q_lr (now the σ step size, not just a trust-
region clamp) produces a real, measurable difference in σ_q updates after a
single E-step iteration. Before the 2026-05-13 fix this test would have
returned ~identical σ_q for any sigma_lr value below the trust threshold.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run_one_retraction(sigma_lr: float, trust: float = 5.0) -> torch.Tensor:
    """Replicate the diagonal σ retraction with the chosen LR; return σ_new."""
    from transformer.core.vfe_utils import retract_spd_diagonal_torch

    torch.manual_seed(20260513)
    B, N, K = 2, 8, 6
    # Initial σ at ~1.0 (positive)
    sigma_init = torch.full((B, N, K), 0.4)
    # Synthetic natural gradient (typical magnitude ~0.5σ — δσ/σ ≈ 0.5)
    nat_grad_sigma = 0.5 * sigma_init * torch.randn(B, N, K)

    # Mirror variational_ffn._retract_sigma's plumbing:
    #   step_size = sigma_lr (decoupled, no decay in this single-step test)
    #   trust_region = trust (separate clamp)
    sigma_new = retract_spd_diagonal_torch(
        sigma_diag=sigma_init,
        delta_sigma=-nat_grad_sigma,    # _retract_sigma negates internally
        step_size=sigma_lr,
        trust_region=trust,
        eps=1e-6,
        sigma_max=5.0,
    )
    return sigma_new


def main() -> None:
    print("=" * 70)
    print("σ-LR decoupling smoke test")
    print("=" * 70)

    # Reference: tiny LR
    sigma_small = run_one_retraction(sigma_lr=1e-4)
    # User's current default
    sigma_default = run_one_retraction(sigma_lr=0.015)
    # 10× swept
    sigma_10x = run_one_retraction(sigma_lr=0.15)

    init = 0.4
    d_small = (sigma_small - init).abs().mean().item()
    d_default = (sigma_default - init).abs().mean().item()
    d_10x = (sigma_10x - init).abs().mean().item()

    print(f"\nMean |Δσ| over 1 retraction step (σ_init=0.4):")
    print(f"  sigma_lr = 1e-4   →  |Δσ| = {d_small:.2e}")
    print(f"  sigma_lr = 0.015  →  |Δσ| = {d_default:.2e}")
    print(f"  sigma_lr = 0.15   →  |Δσ| = {d_10x:.2e}")
    print(f"\nRatio (0.15 / 0.015) = {d_10x / d_default:.2f}× (expected ~10× for small whitened)")
    print(f"Ratio (0.015 / 1e-4) = {d_default / d_small:.2f}× (expected ~150× for small whitened)")

    # Hard assertions
    assert d_10x > 3 * d_default, (
        f"10× sigma_lr should produce ≥3× larger σ update, got {d_10x / d_default:.2f}×"
    )
    assert d_small < 0.5 * d_default, (
        f"Tiny sigma_lr should produce <0.5× the default update, got {d_small / d_default:.2f}×"
    )
    print("\nPASS — σ retraction magnitude scales with E_sigma_q_lr.")


if __name__ == "__main__":
    main()
