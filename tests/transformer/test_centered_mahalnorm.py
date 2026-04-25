"""
Tests for CenteredMahalanobisNorm
=================================

The centered variant normalises the residual δ = μ - μ_p under the
Mahalanobis metric Σ, then re-adds μ_p. This is NOT the gauge-theoretic
analog of LayerNorm (that role belongs to MahalanobisNorm + sl(K) gauge
projection; see the sl(K) theorem in ``VFE_Transformer_Idea.md`` §8.1)
— it is a gauge-covariant residual norm for research use.

Invariants tested:
    (a) reduce to MahalanobisNorm when mu_prior is None,
    (b) reduce to MahalanobisNorm when mu_p = 0 AND return_residual=True,
    (c) finite under ill-conditioned Σ with sigma_floor set,
    (d) gauge-covariant under (μ, Σ, μ_p, A) → (hμ, hΣhᵀ, hμ_p, hA),
    (e) additive relation between return_residual modes,
    (f) output depends on mu_prior (sanity),
    (g) diagonal-σ path matches hand-rolled formula,
    (h) ‖μ_out - μ_p‖_{Σ⁻¹}² ≈ K (constant residual Mahalanobis norm).
"""

import pytest
import torch

from transformer.core.blocks import MahalanobisNorm, CenteredMahalanobisNorm


# =============================================================================
# Helpers
# =============================================================================

def _random_spd(K, dtype=torch.float64, conditioning=None, eigs=None):
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
# (a) mu_prior=None reduces to MahalanobisNorm
# =============================================================================

def test_reduces_to_mahalnorm_when_mu_prior_none():
    torch.manual_seed(0)
    K = 6
    B, N = 2, 4
    x = torch.randn(B, N, K)
    sigma_diag = torch.rand(B, N, K) + 0.1

    mn = MahalanobisNorm(K, sigma_floor=None)
    cmn = CenteredMahalanobisNorm(K, sigma_floor=None)
    assert torch.allclose(mn(x, sigma_diag), cmn(x, sigma_diag, mu_prior=None),
                          atol=1e-7)

    Sigma = _random_spd(K, conditioning=5.0, dtype=torch.float32)
    sigma_full = Sigma.expand(B, N, K, K).contiguous()
    assert torch.allclose(mn(x, sigma_full), cmn(x, sigma_full, mu_prior=None),
                          atol=1e-5, rtol=1e-4)


# =============================================================================
# (b) mu_prior=zeros + return_residual=True reduces to MahalanobisNorm
# =============================================================================

def test_reduces_to_mahalnorm_when_mu_prior_zero():
    torch.manual_seed(1)
    K = 5
    B, N = 1, 3
    x = torch.randn(B, N, K)
    sigma = torch.rand(B, N, K) + 0.1
    mu_p = torch.zeros_like(x)

    mn = MahalanobisNorm(K, sigma_floor=None)
    cmn = CenteredMahalanobisNorm(K, sigma_floor=None, return_residual=True)
    assert torch.allclose(mn(x, sigma), cmn(x, sigma, mu_prior=mu_p), atol=1e-7)


# =============================================================================
# (c) ill-conditioned Σ produces finite output
# =============================================================================

def test_centered_fullcov_illcond_sigma():
    torch.manual_seed(2)
    K = 4
    B, N = 2, 3
    eigs = [1e-12, 1.0, 1.0, 1.0]
    sig_single = _random_spd(K, eigs=eigs, dtype=torch.float32)
    sigma = sig_single.expand(B, N, K, K).contiguous()
    x = torch.randn(B, N, K)
    mu_p = 0.5 * torch.randn(B, N, K)

    cmn = CenteredMahalanobisNorm(K, sigma_floor=0.01)
    out = cmn(x, sigma, mu_prior=mu_p)
    assert torch.isfinite(out).all(), f"Non-finite output: {out}"


# =============================================================================
# (d) gauge covariance with μ_p transforming
# =============================================================================

def test_centered_gauge_equivariance():
    torch.manual_seed(5)
    K = 4
    x = torch.randn(1, 2, K, dtype=torch.float32)
    mu_p = 0.3 * torch.randn(1, 2, K, dtype=torch.float32)
    Sigma = _random_spd(K, conditioning=1e3, dtype=torch.float32)
    sigma = Sigma.expand(1, 2, K, K).contiguous()
    A = _random_invertible(K, seed=6).float()
    exp_phi = A.expand(1, 2, K, K).contiguous()
    h = _random_invertible(K, seed=7).float()

    cmn = CenteredMahalanobisNorm(K, sigma_floor=0.01)

    out = cmn(x, sigma, mu_prior=mu_p, exp_phi=exp_phi)
    # Transform: (μ, Σ, μ_p, A) → (hμ, hΣhᵀ, hμ_p, hA)
    x_t = torch.einsum('kl,bnl->bnk', h, x)
    mu_p_t = torch.einsum('kl,bnl->bnk', h, mu_p)
    sigma_t = torch.einsum('kl,bnlm,pm->bnkp', h, sigma, h)
    A_t = h @ A
    exp_phi_t = A_t.expand(1, 2, K, K).contiguous()

    out_t = cmn(x_t, sigma_t, mu_prior=mu_p_t, exp_phi=exp_phi_t)
    expected = torch.einsum('kl,bnl->bnk', h, out)
    max_rel = ((out_t - expected).abs() / expected.abs().clamp(min=1e-3)).max()
    assert max_rel < 1e-2, f"Gauge covariance broken: max rel diff {max_rel.item()}"


# =============================================================================
# (e) return_residual=True output + μ_p == return_residual=False output
# =============================================================================

def test_centered_residual_mode_vs_centered_mode():
    torch.manual_seed(3)
    K = 6
    B, N = 2, 3
    x = torch.randn(B, N, K)
    mu_p = 0.4 * torch.randn(B, N, K)
    sigma = torch.rand(B, N, K) + 0.1

    cmn_full = CenteredMahalanobisNorm(K, return_residual=False)
    cmn_res = CenteredMahalanobisNorm(K, return_residual=True)
    out_full = cmn_full(x, sigma, mu_prior=mu_p)
    out_res = cmn_res(x, sigma, mu_prior=mu_p)

    assert torch.allclose(out_full, out_res + mu_p, atol=1e-7), (
        "return_residual=True should give (out_full - mu_p) exactly"
    )


# =============================================================================
# (f) output depends on μ_p (sanity)
# =============================================================================

def test_centered_mahalnorm_depends_on_mu_prior():
    """Changing μ_p (with x held fixed) must change the output."""
    torch.manual_seed(4)
    K = 6
    B, N = 1, 3
    x = torch.randn(B, N, K)
    mu_p = torch.randn(B, N, K)
    sigma = torch.rand(B, N, K) + 0.1

    cmn = CenteredMahalanobisNorm(K)
    out_1 = cmn(x, sigma, mu_prior=mu_p)
    out_2 = cmn(x, sigma, mu_prior=2.0 * mu_p)

    assert not torch.allclose(out_1, out_2, atol=1e-3), (
        "CenteredMahalanobisNorm output must depend on μ_p"
    )


# =============================================================================
# (g) diagonal path matches hand-rolled formula
# =============================================================================

def test_centered_diagonal_matches_elementwise_formula():
    torch.manual_seed(6)
    K = 5
    B, N = 2, 3
    x = torch.randn(B, N, K)
    mu_p = 0.3 * torch.randn(B, N, K)
    sigma = torch.rand(B, N, K) + 0.2

    cmn = CenteredMahalanobisNorm(K, eps=1e-5)
    out = cmn(x, sigma, mu_prior=mu_p)

    delta = x - mu_p
    s2 = (delta.pow(2) / sigma.clamp(min=1e-5)).sum(-1, keepdim=True)
    scale = torch.sqrt(K / s2.clamp(min=1e-5))
    ref = mu_p + delta * scale
    assert torch.allclose(out, ref, atol=1e-6)


# =============================================================================
# (h) residual Mahalanobis norm ≈ √K
# =============================================================================

def test_centered_residual_mahalanobis_norm_sqrtK():
    """‖μ_out - μ_p‖_{Σ⁻¹}² == K under the floored metric."""
    torch.manual_seed(7)
    K = 6
    B, N = 1, 3
    x = torch.randn(B, N, K)
    mu_p = 0.4 * torch.randn(B, N, K)
    Sigma = _random_spd(K, conditioning=5.0, dtype=torch.float32)
    sigma = Sigma.expand(B, N, K, K).contiguous()

    cmn = CenteredMahalanobisNorm(K, sigma_floor=0.01)
    out = cmn(x.float(), sigma, mu_prior=mu_p.float())

    from transformer.core.vfe_utils import spd_eigfloor
    Sigma_floored = spd_eigfloor(Sigma.double(), floor=0.01)
    Sigma_inv = torch.linalg.inv(Sigma_floored)
    delta_out = (out - mu_p.float()).double()
    s2 = torch.einsum('bnk,kl,bnl->bn', delta_out, Sigma_inv, delta_out)
    assert torch.allclose(torch.sqrt(s2),
                          torch.full_like(s2, K ** 0.5),
                          atol=1e-2), (
        f"Residual Mahalanobis norm != sqrt(K): mean={torch.sqrt(s2).mean().item()}, "
        f"expected {K ** 0.5}"
    )


# =============================================================================
# Factory integration
# =============================================================================

def test_make_norm_creates_centered_mahalnorm():
    from transformer.core.blocks import _make_norm
    norm = _make_norm('centered_mahalnorm', 8, sigma_floor=0.01)
    assert isinstance(norm, CenteredMahalanobisNorm)
    assert norm.sigma_floor == 0.01
    assert norm.K == 8


def test_make_norm_rejects_unknown_types():
    from transformer.core.blocks import _make_norm
    with pytest.raises(ValueError, match="Unknown norm_type"):
        _make_norm('bogus', 8)
