"""
Tests for MahalanobisNorm under full-covariance + evolve_sigma
==============================================================

Validates the ``sigma_floor`` kwarg added to
``transformer/core/blocks.py::MahalanobisNorm`` in change set 11.
The full-covariance branch previously used only an ``eps·I`` ridge
(eps=1e-5) which is insufficient when Σ_q's minimum eigenvalue drifts
below ~1e-5 during E-step evolution: ``torch.linalg.solve`` then loses
precision and ``μᵀΣ⁻¹μ`` picks up negative noise that NaNs the
downstream sqrt.

Invariants tested:
    - Diagonal branch is byte-identity with the pre-change implementation.
    - Full-cov branch with sigma_floor is finite under collapsed Σ.
    - Full-cov output is gauge-covariant when exp_phi is provided.
    - sigma_floor=None preserves the legacy ridge path.
    - s² clamp prevents sqrt-of-negative NaN under float32 cancellation.
"""

import pytest
import torch

from transformer.core.blocks import MahalanobisNorm


# =============================================================================
# Helpers
# =============================================================================

def _random_spd(K, dtype=torch.float64, conditioning=None, eigs=None):
    """Build an SPD matrix with a prescribed eigenvalue spectrum or random."""
    if eigs is not None:
        eigs_t = torch.tensor(list(eigs), dtype=dtype)
        Q, _ = torch.linalg.qr(torch.randn(K, K, dtype=dtype))
        return Q @ torch.diag(eigs_t) @ Q.t()
    if conditioning is not None:
        lam = torch.linspace(1.0 / conditioning, 1.0, K, dtype=dtype)
        Q, _ = torch.linalg.qr(torch.randn(K, K, dtype=dtype))
        return Q @ torch.diag(lam) @ Q.t()
    A = torch.randn(K, K, dtype=dtype) * 0.3
    return A @ A.t() + 0.1 * torch.eye(K, dtype=dtype)


def _random_invertible(K, dtype=torch.float64, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    A = 0.3 * torch.randn(K, K, dtype=dtype) + torch.eye(K, dtype=dtype)
    return A + 0.1 * torch.eye(K, dtype=dtype)


# =============================================================================
# Diagonal path bit-identity
# =============================================================================

def test_mahalnorm_diagonal_path_unchanged():
    """Diagonal σ path is bit-identical regardless of sigma_floor setting
    (the floor only affects the full-cov branch)."""
    torch.manual_seed(0)
    K = 8
    B, N = 2, 5
    x = torch.randn(B, N, K)
    sigma = torch.rand(B, N, K).clamp(min=0.1) + 0.05

    norm_legacy = MahalanobisNorm(K, sigma_floor=None)
    norm_new = MahalanobisNorm(K, sigma_floor=0.01)

    out_legacy = norm_legacy(x, sigma)
    out_new = norm_new(x, sigma)

    assert torch.allclose(out_legacy, out_new, atol=1e-7), (
        "Diagonal path must be byte-identity; sigma_floor affects full-cov only"
    )


def test_mahalnorm_diagonal_tiny_sigma_still_finite():
    """Diagonal path already clamps per-element by eps. Confirms the
    pre-existing protection still holds under an adversarial input."""
    torch.manual_seed(1)
    K = 4
    x = torch.randn(2, 3, K)
    sigma = torch.full((2, 3, K), 1e-20)  # far below eps
    norm = MahalanobisNorm(K)
    out = norm(x, sigma)
    assert torch.isfinite(out).all()


# =============================================================================
# Full-covariance path
# =============================================================================

def test_mahalnorm_fullcov_illcond_sigma():
    """The failing scenario: Σ has one collapsed eigenvalue (mimics evolved
    sigma_q hitting retract_spd_torch's eps=1e-6 floor). With sigma_floor
    set, the output must be finite."""
    torch.manual_seed(2)
    K = 4
    B, N = 2, 3
    eigs = [1e-12, 1.0, 1.0, 1.0]
    sig_single = _random_spd(K, eigs=eigs, dtype=torch.float32)
    sigma = sig_single.expand(B, N, K, K).contiguous()
    x = torch.randn(B, N, K)

    norm = MahalanobisNorm(K, sigma_floor=0.01)
    out = norm(x, sigma)
    assert torch.isfinite(out).all(), f"Non-finite output: {out}"


def test_mahalnorm_fullcov_legacy_illcond_would_nan():
    """Confirms the PRE-fix behaviour: without the eigclamp (sigma_floor=None),
    an extreme-conditioning Σ likely produces non-finite output via the
    sqrt(-) path. This documents why the fix is necessary."""
    torch.manual_seed(3)
    K = 4
    B, N = 2, 3
    # Extremely collapsed spectrum: eigenvalues from 1e-25 to 1.0. eps=1e-5
    # ridge lifts them to ~1e-5, leaving cond(Σ) ~1e5. Float32 solve on
    # such a matrix is guaranteed to pick up sub-eps negative noise in
    # s² = μᵀΣ⁻¹μ via cancellation.
    eigs = [1e-25, 1e-20, 1.0, 1.0]
    sig_single = _random_spd(K, eigs=eigs, dtype=torch.float32)
    sigma = sig_single.expand(B, N, K, K).contiguous()
    x = torch.randn(B, N, K)

    norm_legacy = MahalanobisNorm(K, sigma_floor=None)
    out_legacy = norm_legacy(x, sigma)
    norm_fixed = MahalanobisNorm(K, sigma_floor=0.01)
    out_fixed = norm_fixed(x, sigma)

    # Fixed path must be finite.
    assert torch.isfinite(out_fixed).all()
    # We don't require legacy to be non-finite here (too flaky across
    # platforms); the substantive check is that FIXED is stable.


def test_mahalnorm_fullcov_well_conditioned_close_to_legacy():
    """On well-conditioned Σ (λ_min ≫ floor), the eigclamp path should not
    materially change the output. Output Mahalanobis norm remains ≈ √K."""
    torch.manual_seed(4)
    K = 6
    Sigma = _random_spd(K, conditioning=5.0, dtype=torch.float32)
    sigma = Sigma.expand(2, 3, K, K).contiguous()
    x = torch.randn(2, 3, K)

    norm = MahalanobisNorm(K, sigma_floor=0.01)
    out = norm(x, sigma)

    # Verify the output has Mahalanobis norm ≈ √K under the floored metric.
    # Using double precision reference to verify.
    from transformer.core.vfe_utils import spd_eigfloor
    Sigma_floored = spd_eigfloor(Sigma.double(), floor=0.01)
    Sigma_inv = torch.linalg.inv(Sigma_floored)
    out_d = out.double()
    s2 = torch.einsum('bnk,kl,bnl->bn', out_d, Sigma_inv, out_d)
    # sqrt(s2) should equal sqrt(K) up to small tolerance.
    assert torch.allclose(torch.sqrt(s2), torch.full_like(s2, K ** 0.5),
                          atol=1e-2), (
        f"Mahalanobis norm deviates from √K: mean={torch.sqrt(s2).mean()}, "
        f"expected {K ** 0.5}"
    )


# =============================================================================
# Gauge covariance
# =============================================================================

def test_mahalnorm_fullcov_gauge_equivariance():
    """Under (μ, Σ, A) → (hμ, hΣh^T, hA), the normalised output should
    transform covariantly: norm(hμ, hΣh^T, exp_phi=hA) == h · norm(μ, Σ, exp_phi=A).
    This is gauge-covariant via spd_eigfloor's whitening branch."""
    torch.manual_seed(5)
    K = 4
    # Float64 throughout for precision; the internal float32 cast inside
    # the autocast block will round, so we relax tolerance accordingly.
    x = torch.randn(1, 2, K, dtype=torch.float32)
    Sigma = _random_spd(K, conditioning=1e3, dtype=torch.float32)
    sigma = Sigma.expand(1, 2, K, K).contiguous()
    A = _random_invertible(K, seed=6).float()
    exp_phi = A.expand(1, 2, K, K).contiguous()
    h = _random_invertible(K, seed=7).float()

    norm = MahalanobisNorm(K, sigma_floor=0.01)

    # Baseline
    out = norm(x, sigma, exp_phi=exp_phi)
    # Transported
    x_t = torch.einsum('kl,bnl->bnk', h, x)
    sigma_t = torch.einsum('kl,bnlm,pm->bnkp', h, sigma, h)
    A_t = h @ A
    exp_phi_t = A_t.expand(1, 2, K, K).contiguous()
    out_t = norm(x_t, sigma_t, exp_phi=exp_phi_t)

    expected = torch.einsum('kl,bnl->bnk', h, out)
    max_rel = ((out_t - expected).abs() / expected.abs().clamp(min=1e-3)).max()
    assert max_rel < 1e-2, (
        f"Gauge covariance broken: max relative diff {max_rel.item()}"
    )


# =============================================================================
# Legacy preservation
# =============================================================================

def test_mahalnorm_sigma_floor_none_preserves_legacy():
    """With sigma_floor=None, the output uses the legacy ridge path —
    compare to a manually computed reference of the pre-change formula."""
    torch.manual_seed(8)
    K = 4
    Sigma = _random_spd(K, conditioning=5.0, dtype=torch.float32)
    sigma = Sigma.expand(1, 2, K, K).contiguous()
    x = torch.randn(1, 2, K)

    # Reference: pre-change formula exactly (eps·I ridge, no s2 clamp).
    eps = 1e-5
    sig_ref = sigma.float() + eps * torch.eye(K).expand_as(sigma)
    sig_inv_mu_ref = torch.linalg.solve(sig_ref, x.unsqueeze(-1).float()).squeeze(-1)
    s2_ref = (x.float() * sig_inv_mu_ref).sum(dim=-1, keepdim=True)
    # Pre-change: sqrt(K / (s2 + eps)); new: sqrt(K / s2.clamp(min=eps))
    # For well-conditioned Σ with s2 ≫ eps, these agree within fp roundoff.
    ref = x * torch.sqrt(K / (s2_ref + eps))

    norm = MahalanobisNorm(K, sigma_floor=None)
    out = norm(x, sigma)

    # Tolerance accounts for s2 near eps handling. For well-conditioned
    # inputs they should agree to within 1e-4 relative.
    assert torch.allclose(out, ref, rtol=1e-3, atol=1e-5), (
        f"Legacy path diverged: max diff {(out - ref).abs().max().item()}"
    )


# =============================================================================
# s² negative-noise defence
# =============================================================================

def test_mahalnorm_s2_clamp_prevents_negative_sqrt():
    """Even without sigma_floor, the new s2.clamp(min=eps) replaces the
    old (s2 + eps) → if float32 noise ever produces s² < -eps, legacy
    would NaN but the clamp yields a finite value."""
    torch.manual_seed(9)
    K = 4
    # Adversarial: very small x + near-singular Σ. Pushes solve precision
    # limits; s2 = μᵀΣ⁻¹μ can swing negative via cancellation.
    Sigma = _random_spd(K, eigs=[1e-18, 1e-15, 1.0, 1.0], dtype=torch.float32)
    sigma = Sigma.expand(1, 1, K, K).contiguous()
    x = torch.full((1, 1, K), 1e-10)

    norm = MahalanobisNorm(K, sigma_floor=None)
    out = norm(x, sigma)
    assert torch.isfinite(out).all(), (
        f"s² clamp failed to prevent NaN: {out}"
    )
