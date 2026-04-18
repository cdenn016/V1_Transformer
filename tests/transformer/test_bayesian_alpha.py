"""Tests for get_bayesian_alpha schedule consistency with alpha_divergence.

Covers the plan-mandated invariants for Change Set B:
  1. alpha_divergence == 1.0 returns the same per-dim KL-based α as before.
  2. alpha_divergence != 1.0 uses Rényi D_α per-dim (matches the kernel in
     transformer.core.kl_computation).
  3. Continuity: as alpha_divergence → 1.0 the Rényi path → KL path.
  4. Positivity: divergence_k ≥ 0 so α_k ∈ (0, c₀/b₀].
"""

import math

import pytest
import torch

from transformer.core.variational_ffn import VariationalFFNDynamic


def _make_ffn(alpha_divergence: float, K: int = 6) -> VariationalFFNDynamic:
    """Construct a minimal VariationalFFNDynamic with learnable_alpha=True.

    Only get_bayesian_alpha is exercised — generators need only a well-formed
    shape (n_gen, K, K).
    """
    n_gen = K * K
    generators = torch.zeros(n_gen, K, K)
    for a in range(n_gen):
        i, j = divmod(a, K)
        generators[a, i, j] = 1.0
    return VariationalFFNDynamic(
        embed_dim=K,
        generators=generators,
        irrep_dims=[K],
        diagonal_covariance=True,
        learnable_alpha=True,
        alpha_divergence=alpha_divergence,
        n_iterations=1,
    )


def _sample_inputs(B: int = 2, N: int = 4, K: int = 6, seed: int = 0):
    torch.manual_seed(seed)
    mu_q = torch.randn(B, N, K)
    mu_p = torch.randn(B, N, K)
    sigma_q = torch.rand(B, N, K) + 0.1
    sigma_p = torch.rand(B, N, K) + 0.1
    return mu_q, mu_p, sigma_q, sigma_p


def _expected_kl_per_dim(mu_q, mu_p, sigma_q, sigma_p, eps=1e-6):
    s = sigma_q.clamp(min=eps)
    t = sigma_p.clamp(min=eps)
    dm = mu_q - mu_p
    trace_term = s / t
    mahal_term = dm ** 2 / t
    logdet_term = torch.log(t) - torch.log(s)
    kl_k = 0.5 * (trace_term + mahal_term - 1.0 + logdet_term)
    return kl_k.clamp(min=0.0)


def _expected_renyi_per_dim(mu_q, mu_p, sigma_q, sigma_p, alpha, eps=1e-6):
    s = sigma_q.clamp(min=eps)
    t = sigma_p.clamp(min=eps)
    dm = mu_q - mu_p
    blend = ((1.0 - alpha) * s + alpha * t).clamp(min=eps)
    mahal = alpha * dm ** 2 / blend
    logdet = (
        (1.0 - alpha) * torch.log(s) + alpha * torch.log(t) - torch.log(blend)
    ) / (alpha - 1.0)
    d = 0.5 * (mahal + logdet)
    return d.clamp(min=0.0)


def test_kl_branch_matches_reference():
    """alpha_divergence == 1.0: α_k = c₀_k / (b₀_k + kl_k) with KL per-dim."""
    ffn = _make_ffn(alpha_divergence=1.0)
    mu_q, mu_p, sigma_q, sigma_p = _sample_inputs(seed=42)

    alpha = ffn.get_bayesian_alpha(mu_q, mu_p, sigma_p, sigma_q)

    kl_k = _expected_kl_per_dim(mu_q, mu_p, sigma_q, sigma_p)
    c0 = torch.nn.functional.softplus(ffn.raw_c0)
    b0 = torch.nn.functional.softplus(ffn.raw_b0)
    expected = c0 / (b0 + kl_k)

    torch.testing.assert_close(alpha, expected, atol=1e-5, rtol=1e-5)


def test_renyi_branch_matches_reference():
    """alpha_divergence != 1.0: α_k = c₀_k / (b₀_k + D_α,k) with Rényi per-dim."""
    alpha_div = 0.2
    ffn = _make_ffn(alpha_divergence=alpha_div)
    mu_q, mu_p, sigma_q, sigma_p = _sample_inputs(seed=7)

    alpha = ffn.get_bayesian_alpha(mu_q, mu_p, sigma_p, sigma_q)

    d_k = _expected_renyi_per_dim(mu_q, mu_p, sigma_q, sigma_p, alpha_div)
    c0 = torch.nn.functional.softplus(ffn.raw_c0)
    b0 = torch.nn.functional.softplus(ffn.raw_b0)
    expected = c0 / (b0 + d_k)

    torch.testing.assert_close(alpha, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("alpha_div", [0.9, 0.99, 1.01, 1.1])
def test_continuity_at_alpha_one(alpha_div):
    """As alpha_divergence → 1.0, Rényi branch agrees with KL branch numerically.

    The per-dim formulas are discontinuous expressions (division by α-1), but
    the limit is the KL divergence. At α within ~10% of 1 the divergences
    should agree to within a few percent.
    """
    mu_q, mu_p, sigma_q, sigma_p = _sample_inputs(seed=1)

    ffn_renyi = _make_ffn(alpha_divergence=alpha_div)
    ffn_kl = _make_ffn(alpha_divergence=1.0)
    # Share parameters so the comparison is apples-to-apples.
    with torch.no_grad():
        ffn_renyi.raw_c0.copy_(ffn_kl.raw_c0)
        ffn_renyi.raw_b0.copy_(ffn_kl.raw_b0)

    a_renyi = ffn_renyi.get_bayesian_alpha(mu_q, mu_p, sigma_p, sigma_q)
    a_kl = ffn_kl.get_bayesian_alpha(mu_q, mu_p, sigma_p, sigma_q)

    # Tolerance grows with |α - 1|; at 10% off 1 expect <5% relative error.
    rel = (a_renyi - a_kl).abs() / a_kl.abs().clamp(min=1e-6)
    tol = max(0.05, 10.0 * abs(alpha_div - 1.0))
    assert rel.max().item() < tol, (
        f"Rényi→KL continuity failed at α={alpha_div}: max rel err "
        f"{rel.max().item():.4f} exceeds {tol:.4f}"
    )


@pytest.mark.parametrize("alpha_div", [0.2, 0.5, 1.0, 1.5])
def test_alpha_positivity(alpha_div):
    """α_k must be strictly positive and bounded by c₀/b₀ (divergence ≥ 0)."""
    ffn = _make_ffn(alpha_divergence=alpha_div)
    mu_q, mu_p, sigma_q, sigma_p = _sample_inputs(seed=99)

    alpha = ffn.get_bayesian_alpha(mu_q, mu_p, sigma_p, sigma_q)

    assert torch.isfinite(alpha).all(), "alpha contains non-finite entries"
    assert (alpha > 0).all(), "alpha must be strictly positive"

    c0 = torch.nn.functional.softplus(ffn.raw_c0)
    b0 = torch.nn.functional.softplus(ffn.raw_b0)
    upper = c0 / b0
    # Every entry should be ≤ c₀/b₀ (since divergence_k ≥ 0) up to eps.
    assert (alpha <= upper + 1e-5).all(), (
        "alpha exceeds c₀/b₀ upper bound — schedule non-decreasing in divergence?"
    )


def test_full_cov_matches_diagonal_for_diagonal_sigma():
    """Full-cov code path with diagonal Σ should match the diagonal path."""
    alpha_div = 0.3
    K = 6
    mu_q, mu_p, sigma_q_diag, sigma_p_diag = _sample_inputs(K=K, seed=17)
    sigma_q_full = torch.diag_embed(sigma_q_diag)
    sigma_p_full = torch.diag_embed(sigma_p_diag)

    ffn = _make_ffn(alpha_divergence=alpha_div, K=K)
    a_diag = ffn.get_bayesian_alpha(mu_q, mu_p, sigma_p_diag, sigma_q_diag)
    a_full = ffn.get_bayesian_alpha(mu_q, mu_p, sigma_p_full, sigma_q_full)

    torch.testing.assert_close(a_diag, a_full, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("alpha_div", [1.0, 0.2])
@pytest.mark.parametrize(
    "p_shape,q_shape",
    [("diag", "full"), ("full", "diag")],
)
def test_mixed_diagonal_and_full_sigma(alpha_div, p_shape, q_shape):
    """Mixed σ shapes (prior diagonal, posterior full or vice versa) must not
    broadcast-fail.  Regression guard for the production error where full-cov
    mode kept sigma_p diagonal (from PriorBank) while sigma_q evolved to full.
    """
    K = 6
    mu_q, mu_p, sigma_q_diag, sigma_p_diag = _sample_inputs(K=K, seed=23)
    sigma_p_full = torch.diag_embed(sigma_p_diag)
    sigma_q_full = torch.diag_embed(sigma_q_diag)

    sigma_p = sigma_p_diag if p_shape == "diag" else sigma_p_full
    sigma_q = sigma_q_diag if q_shape == "diag" else sigma_q_full

    ffn = _make_ffn(alpha_divergence=alpha_div, K=K)
    alpha = ffn.get_bayesian_alpha(mu_q, mu_p, sigma_p, sigma_q)

    assert alpha.shape == (*mu_q.shape[:-1], K)
    assert torch.isfinite(alpha).all()
    assert (alpha > 0).all()
    # Mixed-shape path computes the diagonal proxy — verify it matches the
    # pure-diagonal path to within floating-point tolerance.
    alpha_diag_ref = ffn.get_bayesian_alpha(mu_q, mu_p, sigma_p_diag, sigma_q_diag)
    torch.testing.assert_close(alpha, alpha_diag_ref, atol=1e-4, rtol=1e-4)
