"""Tests for `transformer.core.vfe_closed_form.run_closed_form_e_step`.

Two concerns covered:

1. **Algebraic equivalence of the Omega-free refactor (audit 4.1)** —
   the refactored diagonal-CF body computes ``mu_j_t = Omega @ mu_j``
   and ``sigma_j_t_diag = diag(Omega @ diag(sigma) @ Omega^T)`` via
   two-step contractions that never materialise the full
   ``(B, N, N, d_h, d_h)`` pairwise Omega tensor. These pure-tensor
   tests verify the new contractions are bit-equivalent (float64) to
   a reference computation that does build the full Omega.

2. **`exact_diagonal_transport=True` honours the contract (audit 2.3)** —
   when the flag is set, the closed-form path must lift diagonal sigma
   to a full matrix and route through the Cholesky-inverse branch
   (preserving off-diagonal information that the element-wise inverse
   path would discard). Integration test: build a tiny FFN with
   ``closed_form_e_step=True`` and ``exact_diagonal_transport=True``,
   run forward, verify finite output and correct shape.

No prior tests exist for this code path; these are the first.
"""

from __future__ import annotations

import torch
import pytest

from transformer.core.variational_ffn import VariationalFFNDynamic


# ---------------------------------------------------------------------------
# 1. Algebraic identity tests (Omega-free contractions)
# ---------------------------------------------------------------------------


def _reference_mu_sigma_via_full_omega(
    exp_phi_h: torch.Tensor,
    exp_neg_phi_h: torch.Tensor,
    mu_h: torch.Tensor,
    sigma_h: torch.Tensor,
):
    """Reference: build full Omega, then diag-of-sandwich and Omega@mu."""
    Omega = torch.einsum('bikl,bjlm->bijkm', exp_phi_h, exp_neg_phi_h)  # (B,N,N,d_h,d_h)
    sigma_t = torch.einsum('bijkl,bijkl,bjl->bijk', Omega, Omega, sigma_h)
    mu_t = torch.einsum('bijkl,bjl->bijk', Omega, mu_h)
    return mu_t, sigma_t


def _refactored_mu_sigma_omega_free(
    exp_phi_h: torch.Tensor,
    exp_neg_phi_h: torch.Tensor,
    mu_h: torch.Tensor,
    sigma_h: torch.Tensor,
):
    """Refactored: Omega-free two-step contractions for both mu and sigma."""
    rotated_mu = torch.einsum('bjkl,bjl->bjk', exp_neg_phi_h, mu_h)
    mu_t = torch.einsum('bikl,bjl->bijk', exp_phi_h, rotated_mu)
    S = torch.einsum('bjml,bjnl,bjl->bjmn', exp_neg_phi_h, exp_neg_phi_h, sigma_h)
    sigma_t = torch.einsum('bikm,bikn,bjmn->bijk', exp_phi_h, exp_phi_h, S)
    return mu_t, sigma_t


@pytest.mark.parametrize('B,N,d_h', [(1, 4, 4), (2, 8, 6), (3, 6, 8)])
def test_omega_refactor_matches_reference(B: int, N: int, d_h: int) -> None:
    """Refactored Omega-free contractions equal the explicit-Omega reference at 1e-12 (float64)."""
    torch.manual_seed(20260517 + B * 100 + N * 10 + d_h)
    exp_phi = torch.randn(B, N, d_h, d_h, dtype=torch.float64) * 0.3
    exp_phi = exp_phi + torch.eye(d_h, dtype=torch.float64)  # near identity for invertibility
    exp_neg_phi = torch.linalg.inv(exp_phi)
    mu_h = torch.randn(B, N, d_h, dtype=torch.float64)
    sigma_h = torch.rand(B, N, d_h, dtype=torch.float64) + 0.2

    mu_ref, sigma_ref = _reference_mu_sigma_via_full_omega(
        exp_phi, exp_neg_phi, mu_h, sigma_h,
    )
    mu_new, sigma_new = _refactored_mu_sigma_omega_free(
        exp_phi, exp_neg_phi, mu_h, sigma_h,
    )

    assert torch.allclose(mu_new, mu_ref, atol=1e-12, rtol=1e-12)
    assert torch.allclose(sigma_new, sigma_ref, atol=1e-12, rtol=1e-12)


def test_omega_refactor_handles_non_orthogonal() -> None:
    """The Omega-free identity holds for non-orthogonal Omega (the case where
    a careless rewrite would diverge from the sandwich-product semantics)."""
    torch.manual_seed(20260517)
    B, N, d_h = 1, 6, 5
    # Construct deliberately non-orthogonal transport: random scale + shear.
    exp_phi = torch.eye(d_h, dtype=torch.float64).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1).clone()
    perturb = torch.randn(B, N, d_h, d_h, dtype=torch.float64) * 0.5
    exp_phi = exp_phi + perturb
    exp_neg_phi = torch.linalg.inv(exp_phi)

    mu_h = torch.randn(B, N, d_h, dtype=torch.float64)
    sigma_h = torch.rand(B, N, d_h, dtype=torch.float64) + 0.2

    mu_ref, sigma_ref = _reference_mu_sigma_via_full_omega(
        exp_phi, exp_neg_phi, mu_h, sigma_h,
    )
    mu_new, sigma_new = _refactored_mu_sigma_omega_free(
        exp_phi, exp_neg_phi, mu_h, sigma_h,
    )

    assert torch.allclose(mu_new, mu_ref, atol=1e-12, rtol=1e-12)
    assert torch.allclose(sigma_new, sigma_ref, atol=1e-12, rtol=1e-12)


# ---------------------------------------------------------------------------
# 2. `exact_diagonal_transport=True` integration test
# ---------------------------------------------------------------------------


def _make_ffn(
    K: int = 8,
    irrep_dim: int = 8,
    *,
    diagonal_covariance: bool = True,
    exact_diagonal_transport: bool = False,
    closed_form_e_step: bool = True,
):
    """Construct a minimal VariationalFFNDynamic for closed-form tests."""
    n_gen = K * K
    generators = torch.zeros(n_gen, K, K)
    for a in range(n_gen):
        i, j = divmod(a, K)
        generators[a, i, j] = 1.0

    return VariationalFFNDynamic(
        embed_dim=K,
        generators=generators,
        irrep_dims=[irrep_dim] * (K // irrep_dim),
        diagonal_covariance=diagonal_covariance,
        exact_diagonal_transport=exact_diagonal_transport,
        closed_form_e_step=closed_form_e_step,
        n_iterations=1,
        update_phi=False,        # closed-form does its own phi update
        gauge_mode='learned',
        update_sigma=True,
        alpha=0.01,
        lambda_belief=1.0,
        lambda_softmax=0.0,      # cleaner reference; softmax-coupling adds noise
        kappa=1.0,
        mu_lr=0.1,
        sigma_lr=0.001,
        learnable_lr=False,
    )


def test_closed_form_diagonal_finite() -> None:
    """Smoke: run_closed_form_e_step under closed_form_e_step=True with the
    refactored Omega-free contractions returns finite mu and sigma."""
    torch.manual_seed(20260517)
    ffn = _make_ffn(K=8, irrep_dim=4, exact_diagonal_transport=False)
    B, N, K = 2, 6, 8
    mu = torch.randn(B, N, K) * 0.5
    sigma = torch.rand(B, N, K) + 0.2

    # Token IDs not needed when use_prior_bank=False; the FFN derives priors
    # from mu/sigma_p arguments inside forward.
    out = ffn(
        mu=mu, sigma=sigma,
        mu_prior=torch.zeros_like(mu), sigma_prior=torch.ones_like(sigma),
        phi=torch.zeros(B, N, ffn.generators.shape[0]),
    )
    mu_out, sigma_out = out[0], out[1]
    assert torch.isfinite(mu_out).all()
    if sigma_out is not None:
        assert torch.isfinite(sigma_out).all()


def test_closed_form_exact_diagonal_transport_lifts_to_full() -> None:
    """When exact_diagonal_transport=True the CF path lifts sigma to full
    and routes through the Cholesky-inverse branch. The output sigma must
    remain diagonal-shaped (the lift is internal) and be finite."""
    torch.manual_seed(20260517)
    ffn = _make_ffn(K=8, irrep_dim=4, exact_diagonal_transport=True)
    B, N, K = 2, 6, 8
    mu = torch.randn(B, N, K) * 0.5
    sigma = torch.rand(B, N, K) + 0.2

    out = ffn(
        mu=mu, sigma=sigma,
        mu_prior=torch.zeros_like(mu), sigma_prior=torch.ones_like(sigma),
        phi=torch.randn(B, N, ffn.generators.shape[0]) * 0.05,
    )
    mu_out, sigma_out = out[0], out[1]
    assert mu_out.shape == (B, N, K)
    assert torch.isfinite(mu_out).all()
    if sigma_out is not None:
        # Caller contract: diagonal sigma in, diagonal sigma out
        # (exit re-extracts the diagonal of the lifted full matrix).
        assert sigma_out.shape == (B, N, K)
        assert torch.isfinite(sigma_out).all()
