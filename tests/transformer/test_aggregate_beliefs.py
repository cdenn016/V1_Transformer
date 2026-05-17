"""Tests for `transformer.vfe.attention.aggregate_beliefs`.

Audit finding 4.9 rearranged the ``is_diagonal=True`` branch of
``aggregate_beliefs`` to avoid materialising the full
``(B, N, N, d_h, d_h)`` Omega pairwise tensor. The new contractions
compute the same quantity (``diag(Omega @ diag(sigma) @ Omega^T)``)
via a two-step ``(m1, m2)`` outer-product factorisation.

These tests verify (at float64) that the Omega-free path produces
bit-equivalent output to a reference computation that builds Omega
explicitly. No prior tests exercise ``aggregate_beliefs`` directly.
"""

from __future__ import annotations

import torch
import pytest

from transformer.vfe.attention import aggregate_beliefs


def _reference_aggregate_beliefs_diagonal(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    block_exp_pairs: list,
    beta: torch.Tensor,
    irrep_dims: list,
    mode: str = 'mixture',
):
    """Reference implementation that builds full Omega and uses the
    `Omega**2 @ sigma` pattern. Pre-refactor behaviour of `aggregate_beliefs`."""
    mu_agg_parts, sigma_agg_parts = [], []
    block_start = 0
    for h, d_h in enumerate(irrep_dims):
        block_end = block_start + d_h
        exp_h, exp_neg_h = block_exp_pairs[h]
        mu_h = mu[:, :, block_start:block_end]
        sigma_h = sigma[:, :, block_start:block_end]

        # mu transport: factored form is mathematically identical (used by the
        # refactored code too); here we follow the same path so the test is
        # truly checking the sigma path.
        rotated_mu_j = torch.einsum('bjkl,bjl->bjk', exp_neg_h, mu_h)
        transported_ij = torch.einsum('bikl,bjl->bijk', exp_h, rotated_mu_j)
        mu_agg_h = torch.einsum('bij,bijd->bid', beta, transported_ij)

        # Reference sigma: build full Omega, then element-wise-squared.
        omega_ij = torch.einsum('bikm,bjml->bijkl', exp_h, exp_neg_h)
        sigma_transported_ij = torch.einsum(
            'bijkl,bjl->bijk', omega_ij ** 2, sigma_h,
        )

        if mode == 'precision':
            precision_ij = 1.0 / sigma_transported_ij.clamp(min=1e-6)
            precision_agg = torch.einsum('bij,bijd->bid', beta, precision_ij)
            sigma_agg_h = 1.0 / precision_agg.clamp(min=1e-6)
        else:
            sigma_within = torch.einsum('bij,bijd->bid', beta, sigma_transported_ij)
            mu_dev = transported_ij - mu_agg_h.unsqueeze(2)
            sigma_between = torch.einsum('bij,bijd->bid', beta, mu_dev ** 2)
            sigma_agg_h = (sigma_within + sigma_between).clamp(min=1e-6)

        mu_agg_parts.append(mu_agg_h)
        sigma_agg_parts.append(sigma_agg_h)
        block_start = block_end

    return torch.cat(mu_agg_parts, dim=-1), torch.cat(sigma_agg_parts, dim=-1)


def _make_inputs(B: int, N: int, irrep_dims: list, seed: int = 0):
    torch.manual_seed(seed)
    K = sum(irrep_dims)
    mu = torch.randn(B, N, K, dtype=torch.float64)
    sigma = torch.rand(B, N, K, dtype=torch.float64) + 0.2
    logits = torch.randn(B, N, N, dtype=torch.float64)
    beta = torch.softmax(logits, dim=-1)

    block_exp_pairs = []
    for d_h in irrep_dims:
        e = torch.randn(B, N, d_h, d_h, dtype=torch.float64) * 0.3
        e = e + torch.eye(d_h, dtype=torch.float64)
        e_inv = torch.linalg.inv(e)
        block_exp_pairs.append((e, e_inv))

    return mu, sigma, beta, block_exp_pairs


@pytest.mark.parametrize('irrep_dims', [[4], [4, 4], [3, 5, 4]])
@pytest.mark.parametrize('mode', ['mixture', 'precision'])
def test_aggregate_beliefs_diagonal_matches_reference(
    irrep_dims: list, mode: str,
) -> None:
    """Omega-free diagonal aggregation matches reference within 1e-12 (float64)."""
    B, N = 2, 6
    mu, sigma, beta, block_exp_pairs = _make_inputs(
        B, N, irrep_dims, seed=20260517 + len(irrep_dims),
    )

    mu_ref, sigma_ref = _reference_aggregate_beliefs_diagonal(
        mu, sigma, block_exp_pairs, beta, irrep_dims, mode=mode,
    )
    mu_new, sigma_new = aggregate_beliefs(
        mu, sigma, beta, block_exp_pairs, irrep_dims, mode=mode,
    )

    assert torch.allclose(mu_new, mu_ref, atol=1e-12, rtol=1e-12)
    assert torch.allclose(sigma_new, sigma_ref, atol=1e-12, rtol=1e-12)


def test_aggregate_beliefs_diagonal_non_orthogonal() -> None:
    """Identity holds for deliberately non-orthogonal Omega (the case where
    a careless rewrite would diverge from the sandwich-product semantics)."""
    B, N, d_h = 2, 6, 5
    irrep_dims = [d_h, d_h]
    K = sum(irrep_dims)

    torch.manual_seed(20260517)
    mu = torch.randn(B, N, K, dtype=torch.float64)
    sigma = torch.rand(B, N, K, dtype=torch.float64) + 0.2
    logits = torch.randn(B, N, N, dtype=torch.float64)
    beta = torch.softmax(logits, dim=-1)

    # Strongly non-orthogonal: 0.7 magnitude perturbation off identity
    block_exp_pairs = []
    for _ in irrep_dims:
        e = torch.eye(d_h, dtype=torch.float64).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1).clone()
        e = e + torch.randn(B, N, d_h, d_h, dtype=torch.float64) * 0.7
        e_inv = torch.linalg.inv(e)
        block_exp_pairs.append((e, e_inv))

    mu_ref, sigma_ref = _reference_aggregate_beliefs_diagonal(
        mu, sigma, block_exp_pairs, beta, irrep_dims, mode='mixture',
    )
    mu_new, sigma_new = aggregate_beliefs(
        mu, sigma, beta, block_exp_pairs, irrep_dims, mode='mixture',
    )

    assert torch.allclose(mu_new, mu_ref, atol=1e-12, rtol=1e-12)
    assert torch.allclose(sigma_new, sigma_ref, atol=1e-12, rtol=1e-12)


def test_aggregate_beliefs_diagonal_finite_small() -> None:
    """Smoke: aggregate_beliefs returns finite output for a tiny config."""
    B, N = 1, 4
    irrep_dims = [4]
    mu, sigma, beta, block_exp_pairs = _make_inputs(B, N, irrep_dims, seed=42)
    mu_out, sigma_out = aggregate_beliefs(
        mu, sigma, beta, block_exp_pairs, irrep_dims, mode='mixture',
    )
    assert torch.isfinite(mu_out).all()
    assert torch.isfinite(sigma_out).all()
