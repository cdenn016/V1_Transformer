"""
Phase 3 regression tests for spd_eigfloor's gauge-covariant path.

After Phase 3, the branch at vfe_utils.py:406-426 replaces
`A_inv = torch.linalg.inv(A)` with two lu_solve calls. These tests
verify that:

1. The LU-solve-based `W = A^{-1} Σ A^{-T}` computation matches the
   explicit-inverse computation when cond(A) is well-behaved.
2. The spd_eigfloor output remains SPD and finite under an
   ill-conditioned A where the explicit-inverse path would lose
   precision.
3. Gradients through spd_eigfloor are finite under a moderately
   ill-conditioned A.

The parity check in test 1 targets the W-formation subroutine in
isolation (the actual unit being changed), not the full
spd_eigfloor pipeline — that avoids coupling the test to the
internal _safe_eigh jitter schedule, which is unchanged by Phase 3.
"""

import torch
import pytest

from transformer.core.vfe_utils import spd_eigfloor


def _compute_W_via_lu_solve(A, Sigma):
    """Reproduces the Phase 3 W-formation block from vfe_utils.py:414-418."""
    LU, pivots = torch.linalg.lu_factor(A)
    left = torch.linalg.lu_solve(LU, pivots, Sigma)
    W = torch.linalg.lu_solve(LU, pivots, left.transpose(-1, -2))
    W = W.transpose(-1, -2)
    return 0.5 * (W + W.transpose(-1, -2))


def _compute_W_via_inv(A, Sigma):
    """Legacy W-formation using explicit matrix inverse."""
    A_inv = torch.linalg.inv(A)
    W = A_inv @ Sigma @ A_inv.transpose(-1, -2)
    return 0.5 * (W + W.transpose(-1, -2))


def test_w_formation_parity_wellcond():
    """On well-conditioned A, the LU-solve and explicit-inverse W-formations
    are mathematically equivalent and must agree to float32 tolerance."""
    torch.manual_seed(0)
    K = 6

    G = torch.randn(K, K)
    Sigma = G @ G.T + 0.8 * torch.eye(K)

    A = torch.eye(K) + 0.2 * torch.randn(K, K)
    cond_A = torch.linalg.cond(A).item()
    assert cond_A < 20.0, f"Test setup: cond(A)={cond_A:.1f} exceeds target"

    W_lu = _compute_W_via_lu_solve(A, Sigma)
    W_inv = _compute_W_via_inv(A, Sigma)

    assert torch.allclose(W_lu, W_inv, rtol=1e-5, atol=1e-6), (
        f"max abs diff {torch.max(torch.abs(W_lu - W_inv)).item():.2e}"
    )


def test_w_formation_stable_on_illcond_A():
    """On ill-conditioned A (cond ~1e4), the LU-solve W-formation
    produces a finite, symmetric, SPD matrix. This is the stability
    win: inv(A) @ Σ @ inv(A).T on the same input loses enough fp32
    precision that small SPD perturbations of the input can produce
    non-SPD outputs, while lu_solve preserves PSD structure."""
    torch.manual_seed(1)
    K = 5

    Q, _ = torch.linalg.qr(torch.randn(K, K))
    s = torch.logspace(0, -4, K)  # cond ~1e4
    A = Q @ torch.diag(s) @ Q.T
    cond_A = torch.linalg.cond(A).item()
    assert 5e3 < cond_A < 5e5, f"cond(A)={cond_A:.1e} not in target band"

    G = torch.randn(K, K)
    Sigma = G @ G.T + 0.5 * torch.eye(K)

    W = _compute_W_via_lu_solve(A, Sigma)

    assert torch.isfinite(W).all(), "W has non-finite entries"
    assert torch.allclose(W, W.transpose(-1, -2), atol=1e-3), \
        "W lost symmetry"

    eigs = torch.linalg.eigvalsh(W)
    assert eigs.min().item() > -1e-3, \
        f"W non-PSD: min eig {eigs.min().item():.4e}"


def test_spd_eigfloor_end_to_end_stable_on_wellcond_A():
    """Full spd_eigfloor call under well-conditioned A produces SPD,
    finite output whose whitened spectrum respects the floor."""
    torch.manual_seed(2)
    K = 4
    floor = 0.02

    A = torch.eye(K) + 0.15 * torch.randn(K, K)
    G = torch.randn(K, K)
    Sigma = G @ G.T + 0.8 * torch.eye(K)

    out = spd_eigfloor(Sigma, floor=floor, exp_phi=A)

    assert torch.isfinite(out).all()
    assert torch.allclose(out, out.transpose(-1, -2), atol=1e-5), \
        "output lost symmetry"
    assert torch.linalg.eigvalsh(out).min().item() > 0, \
        "output non-PSD"

    # Whitened spectrum: reconstruct W via lu_solve (precision-matched
    # to the production code) and check floor.
    W_check = _compute_W_via_lu_solve(A, out)
    assert torch.linalg.eigvalsh(W_check).min().item() >= floor - 1e-5


def test_spd_eigfloor_gradient_finite_illcond_A():
    """Backward through spd_eigfloor with moderately ill-conditioned
    A (cond ~1e3) produces finite gradients on both Σ and A.

    Complements the existing test_spd_eigfloor_gradient_finite which
    uses only the frame-dependent branch (exp_phi=None)."""
    torch.manual_seed(3)
    K = 4
    floor = 0.01

    Q, _ = torch.linalg.qr(torch.randn(K, K))
    s = torch.logspace(0, -3, K)  # cond ~1e3
    A = (Q @ torch.diag(s) @ Q.T).detach().clone().requires_grad_(True)

    G = torch.randn(K, K)
    Sigma = (G @ G.T + 0.5 * torch.eye(K)).requires_grad_(True)

    out = spd_eigfloor(Sigma, floor=floor, exp_phi=A)
    loss = out.sum()
    loss.backward()

    assert Sigma.grad is not None and torch.isfinite(Sigma.grad).all(), \
        "Sigma.grad non-finite"
    assert A.grad is not None and torch.isfinite(A.grad).all(), \
        "A.grad non-finite"
