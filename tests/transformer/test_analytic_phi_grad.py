"""
Test analytic phi gradient against autograd reference.

Verifies that analytic_phi_gradient_block_diag produces gradients
matching torch.autograd.grad through the full KL → softmax pipeline.
"""

import pytest
import torch
import math

from transformer.core.gauge_utils import (
    fused_block_matrix_exp_pairs,
    fused_block_diagonal_kl_diag,
    analytic_phi_gradient_block_diag,
)


def _make_so3_generators(K: int, irrep_dims: list):
    """Create block-diagonal SO(3)-like generators for testing."""
    n_gen = 3  # SO(3)
    generators = torch.zeros(n_gen, K, K)
    start = 0
    for d in irrep_dims:
        if d == 1:
            pass  # scalar block: generators are zero
        elif d == 3:
            # Standard so(3) generators
            generators[0, start+1, start+2] = -1
            generators[0, start+2, start+1] = 1
            generators[1, start+0, start+2] = 1
            generators[1, start+2, start+0] = -1
            generators[2, start+0, start+1] = -1
            generators[2, start+1, start+0] = 1
        else:
            # Generic antisymmetric for other dims
            for a in range(min(n_gen, d*(d-1)//2)):
                i_idx = a // (d - 1)
                j_idx = a % (d - 1) + 1
                if j_idx <= i_idx:
                    j_idx = i_idx + 1
                if i_idx < d and j_idx < d:
                    generators[a, start+i_idx, start+j_idx] = -1
                    generators[a, start+j_idx, start+i_idx] = 1
        start += d
    return generators


def _compute_alignment_loss_autograd(
    mu_q, sigma_q, phi, generators, irrep_dims,
    lambda_belief, kappa, eps=1e-6
):
    """Compute F_align and its gradient w.r.t. phi via autograd."""
    B, N, K = mu_q.shape
    phi_for_grad = phi.clone().requires_grad_(True)

    block_exp_pairs = fused_block_matrix_exp_pairs(
        phi_for_grad, generators, irrep_dims
    )

    kl_matrix = fused_block_diagonal_kl_diag(
        mu_q, sigma_q, block_exp_pairs, irrep_dims, eps
    )

    tau = max(kappa * math.sqrt(max(K, 1)), eps)
    beta = torch.softmax(-kl_matrix / tau, dim=-1)

    alignment_loss = lambda_belief * (beta * kl_matrix).sum()

    grad_phi = torch.autograd.grad(
        alignment_loss, phi_for_grad,
        create_graph=False, retain_graph=False,
    )[0]

    return grad_phi, beta.detach(), kl_matrix.detach()


@pytest.mark.parametrize("irrep_dims", [
    [3, 3],        # Two SO(3) blocks
    [1, 1, 3],     # Scalars + SO(3)
    [3],           # Single block
])
@pytest.mark.parametrize("dexp_order", [4, 8])
def test_analytic_vs_autograd(irrep_dims, dexp_order):
    """Compare analytic phi gradient against autograd reference."""
    torch.manual_seed(42)

    B, N = 2, 8
    K = sum(irrep_dims)
    n_gen = 3
    lambda_belief = 1.0
    kappa = 1.0
    eps = 1e-6

    generators = _make_so3_generators(K, irrep_dims)
    mu_q = torch.randn(B, N, K) * 0.5
    sigma_q = torch.rand(B, N, K).clamp(min=0.1) + 0.5
    phi = torch.randn(B, N, n_gen) * 0.3  # Small phi for series convergence

    # Autograd reference
    grad_autograd, beta, kl_matrix = _compute_alignment_loss_autograd(
        mu_q, sigma_q, phi, generators, irrep_dims,
        lambda_belief, kappa, eps
    )

    # Analytic computation
    grad_analytic = analytic_phi_gradient_block_diag(
        mu_q=mu_q.detach(),
        sigma_q=sigma_q.detach(),
        beta=beta,
        kl_matrix=kl_matrix,
        phi=phi.detach(),
        generators=generators,
        irrep_dims=irrep_dims,
        lambda_belief=lambda_belief,
        kappa=kappa,
        eps=eps,
        dexp_order=dexp_order,
    )

    # Check shapes match
    assert grad_analytic.shape == grad_autograd.shape, (
        f"Shape mismatch: {grad_analytic.shape} vs {grad_autograd.shape}"
    )

    # Relative error (use higher tolerance for lower dexp_order)
    norm_autograd = grad_autograd.norm()
    if norm_autograd > 1e-8:
        rel_error = (grad_analytic - grad_autograd).norm() / norm_autograd
        tol = 0.05 if dexp_order >= 8 else 0.15
        assert rel_error < tol, (
            f"Relative error {rel_error:.4f} exceeds tolerance {tol} "
            f"(dexp_order={dexp_order}, irrep_dims={irrep_dims})"
        )

    # Cosine similarity (direction should match well)
    cos_sim = torch.nn.functional.cosine_similarity(
        grad_analytic.flatten().unsqueeze(0),
        grad_autograd.flatten().unsqueeze(0),
    ).item()
    assert cos_sim > 0.95, (
        f"Cosine similarity {cos_sim:.4f} too low "
        f"(dexp_order={dexp_order}, irrep_dims={irrep_dims})"
    )


def test_analytic_zero_phi():
    """At φ=0, Ω=I and dexp reduces to identity (k=0 term only)."""
    torch.manual_seed(123)

    B, N = 1, 4
    irrep_dims = [3, 3]
    K = sum(irrep_dims)
    n_gen = 3
    lambda_belief = 1.0
    kappa = 1.0

    generators = _make_so3_generators(K, irrep_dims)
    mu_q = torch.randn(B, N, K)
    sigma_q = torch.rand(B, N, K).clamp(min=0.1) + 0.5
    phi = torch.zeros(B, N, n_gen)

    grad_autograd, beta, kl_matrix = _compute_alignment_loss_autograd(
        mu_q, sigma_q, phi, generators, irrep_dims,
        lambda_belief, kappa
    )

    grad_analytic = analytic_phi_gradient_block_diag(
        mu_q=mu_q, sigma_q=sigma_q, beta=beta, kl_matrix=kl_matrix,
        phi=phi, generators=generators, irrep_dims=irrep_dims,
        lambda_belief=lambda_belief, kappa=kappa,
        dexp_order=1,  # At phi=0, order 1 should suffice
    )

    norm_autograd = grad_autograd.norm()
    if norm_autograd > 1e-8:
        rel_error = (grad_analytic - grad_autograd).norm() / norm_autograd
        assert rel_error < 0.01, f"At phi=0, relative error {rel_error:.6f} should be < 0.01"


def test_analytic_gradient_no_autograd_graph():
    """Verify analytic path doesn't create autograd graph."""
    torch.manual_seed(0)

    B, N = 1, 4
    irrep_dims = [3]
    K = 3
    n_gen = 3

    generators = _make_so3_generators(K, irrep_dims)
    mu_q = torch.randn(B, N, K)
    sigma_q = torch.rand(B, N, K).clamp(min=0.1) + 0.5
    phi = torch.randn(B, N, n_gen) * 0.2
    beta = torch.softmax(torch.randn(B, N, N), dim=-1)
    kl_matrix = torch.rand(B, N, N)

    grad = analytic_phi_gradient_block_diag(
        mu_q=mu_q, sigma_q=sigma_q, beta=beta, kl_matrix=kl_matrix,
        phi=phi, generators=generators, irrep_dims=irrep_dims,
        lambda_belief=1.0, kappa=1.0,
    )

    assert grad.grad_fn is None, "Analytic gradient should not have autograd graph"


if __name__ == '__main__':
    print("Running analytic phi gradient tests...")
    test_analytic_zero_phi()
    print("  ✓ test_analytic_zero_phi")
    test_analytic_gradient_no_autograd_graph()
    print("  ✓ test_analytic_gradient_no_autograd_graph")
    for irrep_dims in [[3, 3], [1, 1, 3], [3]]:
        for dexp_order in [4, 8]:
            test_analytic_vs_autograd(irrep_dims, dexp_order)
            print(f"  ✓ test_analytic_vs_autograd(irrep_dims={irrep_dims}, dexp_order={dexp_order})")
    print("\nAll tests passed!")
