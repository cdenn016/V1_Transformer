"""
Tests for transformer.core.vfe_gradients
==========================================

Validates the VFE gradient computation engine: the core E-step gradient
functions (compute_vfe_gradients_gpu) and the Fisher-metric natural gradient
projection (compute_natural_gradient_gpu).

Mathematical invariants tested:
    - Finite-difference vs analytic gradient agreement (grad_mu, grad_sigma)
    - Self-coupling gradient formula: alpha * (mu_q - mu_p) / sigma_p
    - Lambda scaling linearity
    - Natural gradient Fisher projection: nat_grad_mu = sigma * grad_mu
    - Output finiteness and shape correctness
"""

import pytest
import torch

from transformer.core.vfe_gradients import (
    compute_vfe_gradients_gpu,
    compute_natural_gradient_gpu,
)


# =============================================================================
# Constants for finite-difference checks
# =============================================================================

FD_EPS = 1e-5
FD_REL_TOL = 5e-3  # Relaxed tolerance for FD vs analytic


# =============================================================================
# Helpers
# =============================================================================

def _make_so3_generators(device='cpu'):
    """SO(3) generators (3, 3, 3)."""
    from math_utils.generators import generate_so3_generators
    return torch.from_numpy(generate_so3_generators(3)).float().to(device)


def _make_diagonal_vfe_inputs(B, N, K, n_gen, device='cpu', dtype=torch.float64):
    """Create complete VFE gradient input set with diagonal covariance."""
    mu_q = torch.randn(B, N, K, device=device, dtype=dtype) * 0.5
    sigma_q = torch.rand(B, N, K, device=device, dtype=dtype).clamp(min=0.3) + 0.2
    mu_p = torch.randn(B, N, K, device=device, dtype=dtype) * 0.3
    sigma_p = torch.rand(B, N, K, device=device, dtype=dtype).clamp(min=0.3) + 0.3

    # Construct valid attention weights (softmax normalized)
    logits = torch.randn(B, N, N, device=device, dtype=dtype)
    beta = torch.softmax(logits, dim=-1)

    phi = torch.randn(B, N, n_gen, device=device, dtype=dtype) * 0.1

    # Simple GL generators for testing (identity-like basis)
    generators = torch.zeros(n_gen, K, K, device=device, dtype=dtype)
    for g in range(min(n_gen, K * K)):
        i, j = divmod(g, K)
        if i < K and j < K:
            generators[g, i, j] = 1.0

    return mu_q, sigma_q, mu_p, sigma_p, beta, phi, generators


def _compute_vfe_scalar(mu_q, sigma_q, mu_p, sigma_p, beta, phi, generators,
                         alpha, lambda_belief, kappa, eps):
    """Compute scalar VFE value for finite-difference checks.

    Computes: F = alpha * KL(q||p) + lambda * sum_j beta_ij * KL(q_i || Omega_ij q_j)
    """
    from transformer.core.gauge_utils import stable_matrix_exp_pair

    B, N, K = mu_q.shape
    sigma_q_safe = sigma_q.clamp(min=eps)
    sigma_p_safe = sigma_p.clamp(min=eps)

    # Self-coupling KL: sum over positions and dimensions
    kl_self = 0.5 * (
        sigma_q_safe / sigma_p_safe
        + (mu_p - mu_q) ** 2 / sigma_p_safe
        - 1.0
        + torch.log(sigma_p_safe) - torch.log(sigma_q_safe)
    ).sum()

    # Transport operators
    phi_matrix = torch.einsum('bna,aij->bnij', phi, generators)
    exp_phi, exp_neg_phi = stable_matrix_exp_pair(phi_matrix)
    Omega = torch.einsum('bikl,bjlm->bijkm', exp_phi, exp_neg_phi)

    # Transported means
    mu_j_transported = torch.einsum('bijkl,bjl->bijk', Omega, mu_q)

    # Transported diagonal covariance
    sigma_j_transported = torch.einsum(
        'bijkl,bijkl,bjl->bijk', Omega, Omega, sigma_q_safe
    ).clamp(min=eps)

    # Pairwise KL
    delta_mu = mu_q.unsqueeze(2) - mu_j_transported
    kl_pairs = 0.5 * (
        sigma_q_safe[:, :, None, :] / sigma_j_transported
        + delta_mu ** 2 / sigma_j_transported
        - 1.0
        + torch.log(sigma_j_transported) - torch.log(sigma_q_safe[:, :, None, :])
    ).sum(dim=-1).clamp(min=0.0)

    # Weighted alignment
    alignment = (beta * kl_pairs).sum()

    return alpha * kl_self + lambda_belief * alignment


# =============================================================================
# TestComputeVFEGradientsGPUDiagonal
# =============================================================================

class TestComputeVFEGradientsGPUDiagonal:
    """compute_vfe_gradients_gpu with diagonal covariance: correctness tests."""

    def test_output_shapes(self, cpu_device):
        """Output shapes match input shapes."""
        B, N, K, n_gen = 2, 4, 6, 3
        mu_q, sigma_q, mu_p, sigma_p, beta, phi, gen = _make_diagonal_vfe_inputs(
            B, N, K, n_gen, cpu_device, dtype=torch.float32
        )
        grad_mu, grad_sigma = compute_vfe_gradients_gpu(
            mu_q, sigma_q, mu_p, sigma_p, beta, phi, gen,
            alpha=0.01, lambda_belief=1.0, kappa=1.0, eps=1e-6,
        )
        assert grad_mu.shape == (B, N, K)
        assert grad_sigma.shape == (B, N, K)

    def test_output_finite(self, cpu_device):
        """No NaN/Inf for random inputs."""
        B, N, K, n_gen = 2, 4, 6, 3
        mu_q, sigma_q, mu_p, sigma_p, beta, phi, gen = _make_diagonal_vfe_inputs(
            B, N, K, n_gen, cpu_device, dtype=torch.float32
        )
        grad_mu, grad_sigma = compute_vfe_gradients_gpu(
            mu_q, sigma_q, mu_p, sigma_p, beta, phi, gen,
            alpha=0.01, lambda_belief=1.0, kappa=1.0, eps=1e-6,
        )
        assert torch.isfinite(grad_mu).all(), "NaN/Inf in grad_mu"
        assert torch.isfinite(grad_sigma).all(), "NaN/Inf in grad_sigma"

    def test_self_coupling_only(self, cpu_device):
        r"""With lambda_belief=0, grad_mu = alpha * (mu_q - mu_p) / sigma_p.

        This tests the self-coupling term in isolation.
        """
        B, N, K = 1, 2, 4
        n_gen = K * K
        mu_q, sigma_q, mu_p, sigma_p, beta, phi, gen = _make_diagonal_vfe_inputs(
            B, N, K, n_gen, cpu_device, dtype=torch.float32
        )
        alpha = 0.5
        grad_mu, grad_sigma = compute_vfe_gradients_gpu(
            mu_q, sigma_q, mu_p, sigma_p, beta, phi, gen,
            alpha=alpha, lambda_belief=0.0, lambda_softmax=0.0,
            kappa=1.0, eps=1e-6,
        )
        expected_grad_mu = alpha * (mu_q - mu_p) / sigma_p.clamp(min=1e-6)
        assert torch.allclose(grad_mu, expected_grad_mu, atol=1e-4), \
            f"Self-coupling grad_mu error: {(grad_mu - expected_grad_mu).abs().max():.2e}"

    def test_self_coupling_grad_sigma(self, cpu_device):
        r"""With lambda_belief=0, grad_sigma = alpha * 0.5 * (1/sigma_p - 1/sigma_q)."""
        B, N, K = 1, 2, 4
        n_gen = K * K
        mu_q, sigma_q, mu_p, sigma_p, beta, phi, gen = _make_diagonal_vfe_inputs(
            B, N, K, n_gen, cpu_device, dtype=torch.float32
        )
        alpha = 0.5
        _, grad_sigma = compute_vfe_gradients_gpu(
            mu_q, sigma_q, mu_p, sigma_p, beta, phi, gen,
            alpha=alpha, lambda_belief=0.0, lambda_softmax=0.0,
            kappa=1.0, eps=1e-6,
        )
        expected = alpha * 0.5 * (1.0 / sigma_p.clamp(min=1e-6) - 1.0 / sigma_q.clamp(min=1e-6))
        assert torch.allclose(grad_sigma, expected, atol=1e-4), \
            f"Self-coupling grad_sigma error: {(grad_sigma - expected).abs().max():.2e}"

    def test_zero_phi_simplification(self, cpu_device):
        """phi=0 means Omega=I, simplifying transport to identity."""
        B, N, K = 1, 3, 4
        n_gen = K * K
        mu_q, sigma_q, mu_p, sigma_p, beta, _, gen = _make_diagonal_vfe_inputs(
            B, N, K, n_gen, cpu_device, dtype=torch.float32
        )
        phi_zero = torch.zeros(B, N, n_gen, device=cpu_device)
        grad_mu, grad_sigma = compute_vfe_gradients_gpu(
            mu_q, sigma_q, mu_p, sigma_p, beta, phi_zero, gen,
            alpha=0.01, lambda_belief=1.0, kappa=1.0, eps=1e-6,
        )
        assert torch.isfinite(grad_mu).all()
        assert torch.isfinite(grad_sigma).all()

    def test_lambda_scaling_linearity(self, cpu_device):
        """Doubling lambda_belief approximately doubles the alignment gradient.

        Not exact because of softmax coupling (nonlinear), but the direct
        term is linear. Test with lambda_softmax=0 for exact linearity.
        """
        B, N, K = 1, 3, 4
        n_gen = K * K
        mu_q, sigma_q, mu_p, sigma_p, beta, phi, gen = _make_diagonal_vfe_inputs(
            B, N, K, n_gen, cpu_device, dtype=torch.float32
        )
        # With lambda_softmax=0, only the direct term (linear in lambda_belief)
        grad1, _ = compute_vfe_gradients_gpu(
            mu_q, sigma_q, mu_p, sigma_p, beta, phi, gen,
            alpha=0.0, lambda_belief=1.0, lambda_softmax=0.0,
            kappa=1.0, eps=1e-6,
        )
        grad2, _ = compute_vfe_gradients_gpu(
            mu_q, sigma_q, mu_p, sigma_p, beta, phi, gen,
            alpha=0.0, lambda_belief=2.0, lambda_softmax=0.0,
            kappa=1.0, eps=1e-6,
        )
        assert torch.allclose(grad2, 2.0 * grad1, atol=1e-4), \
            f"Lambda scaling not linear: max dev {(grad2 - 2*grad1).abs().max():.2e}"

    def test_fd_grad_mu(self, cpu_device):
        """Finite-difference check for grad_mu (self-coupling term only).

        Tests only the self-coupling gradient (alpha > 0, lambda_belief=0)
        to avoid query/key cross-coupling in the alignment term. The
        self-coupling KL(q_i || p_i) depends only on mu_q[i], so FD is exact.
        """
        B, N, K = 1, 2, 3
        n_gen = K * K
        mu_q, sigma_q, mu_p, sigma_p, beta, phi, gen = _make_diagonal_vfe_inputs(
            B, N, K, n_gen, cpu_device, dtype=torch.float32
        )
        alpha = 0.5

        # Analytic: self-coupling only
        grad_mu, _ = compute_vfe_gradients_gpu(
            mu_q, sigma_q, mu_p, sigma_p, beta, phi, gen,
            alpha=alpha, lambda_belief=0.0, lambda_softmax=0.0,
            kappa=1.0, eps=1e-6,
        )

        # FD of self-coupling scalar: F_self = alpha * sum_i KL(q_i || p_i)
        eps_fd = 1e-4
        for b in range(B):
            for n in range(N):
                for k in range(K):
                    mu_plus = mu_q.clone()
                    mu_minus = mu_q.clone()
                    mu_plus[b, n, k] += eps_fd
                    mu_minus[b, n, k] -= eps_fd

                    # Self-coupling KL only
                    def _self_kl(mu):
                        sq, sp = sigma_q.clamp(min=1e-6), sigma_p.clamp(min=1e-6)
                        return alpha * 0.5 * (sq/sp + (mu_p - mu)**2/sp - 1 + torch.log(sp) - torch.log(sq)).sum()

                    fd = (_self_kl(mu_plus) - _self_kl(mu_minus)) / (2 * eps_fd)
                    analytic = grad_mu[b, n, k]
                    err = abs(fd - analytic) / (abs(analytic) + 1e-6)
                    assert err < 0.01, \
                        f"FD grad_mu[{b},{n},{k}]: analytic={analytic:.6f}, fd={fd:.6f}, rel_err={err:.4f}"

    def test_fd_grad_sigma(self, cpu_device):
        """Finite-difference check for grad_sigma (self-coupling term only)."""
        B, N, K = 1, 2, 3
        n_gen = K * K
        mu_q, sigma_q, mu_p, sigma_p, beta, phi, gen = _make_diagonal_vfe_inputs(
            B, N, K, n_gen, cpu_device, dtype=torch.float32
        )
        alpha = 0.5

        _, grad_sigma = compute_vfe_gradients_gpu(
            mu_q, sigma_q, mu_p, sigma_p, beta, phi, gen,
            alpha=alpha, lambda_belief=0.0, lambda_softmax=0.0,
            kappa=1.0, eps=1e-6,
        )

        eps_fd = 1e-4
        for b in range(B):
            for n in range(N):
                for k in range(K):
                    s_plus = sigma_q.clone()
                    s_minus = sigma_q.clone()
                    s_plus[b, n, k] += eps_fd
                    s_minus[b, n, k] -= eps_fd

                    def _self_kl_sigma(sq):
                        sq_s, sp_s = sq.clamp(min=1e-6), sigma_p.clamp(min=1e-6)
                        return alpha * 0.5 * (sq_s/sp_s + (mu_p - mu_q)**2/sp_s - 1 + torch.log(sp_s) - torch.log(sq_s)).sum()

                    fd = (_self_kl_sigma(s_plus) - _self_kl_sigma(s_minus)) / (2 * eps_fd)
                    analytic = grad_sigma[b, n, k]
                    err = abs(fd - analytic) / (abs(analytic) + 1e-6)
                    # Relaxed to 10% for sigma FD: the log(sigma) terms
                    # amplify float32 rounding in the FD perturbation
                    assert err < 0.1, \
                        f"FD grad_sigma[{b},{n},{k}]: analytic={analytic:.6f}, fd={fd:.6f}, rel_err={err:.4f}"

    def test_cached_transport(self, cpu_device):
        """Passing cached_transport gives same result as fresh computation."""
        B, N, K = 1, 3, 4
        n_gen = K * K
        mu_q, sigma_q, mu_p, sigma_p, beta, phi, gen = _make_diagonal_vfe_inputs(
            B, N, K, n_gen, cpu_device, dtype=torch.float32
        )
        # Compute transport
        from transformer.core.transport_ops import compute_transport_operators
        transport = compute_transport_operators(phi, gen, gauge_mode='learned')

        grad1, gs1 = compute_vfe_gradients_gpu(
            mu_q, sigma_q, mu_p, sigma_p, beta, phi, gen,
            alpha=0.01, lambda_belief=1.0, kappa=1.0, eps=1e-6,
        )
        grad2, gs2 = compute_vfe_gradients_gpu(
            mu_q, sigma_q, mu_p, sigma_p, beta, phi, gen,
            alpha=0.01, lambda_belief=1.0, kappa=1.0, eps=1e-6,
            cached_transport=transport,
        )
        assert torch.allclose(grad1, grad2, atol=1e-4), \
            f"Cached vs fresh grad_mu: {(grad1 - grad2).abs().max():.2e}"
        assert torch.allclose(gs1, gs2, atol=1e-4), \
            f"Cached vs fresh grad_sigma: {(gs1 - gs2).abs().max():.2e}"

    def test_exact_diagonal_transport(self, cpu_device):
        """exact_diagonal_transport=True produces finite results."""
        B, N, K = 1, 3, 4
        n_gen = K * K
        mu_q, sigma_q, mu_p, sigma_p, beta, phi, gen = _make_diagonal_vfe_inputs(
            B, N, K, n_gen, cpu_device, dtype=torch.float32
        )
        grad_mu, grad_sigma = compute_vfe_gradients_gpu(
            mu_q, sigma_q, mu_p, sigma_p, beta, phi, gen,
            alpha=0.01, lambda_belief=1.0, kappa=1.0, eps=1e-6,
            exact_diagonal_transport=True,
        )
        assert torch.isfinite(grad_mu).all()
        assert torch.isfinite(grad_sigma).all()
        # exact_diagonal_transport should return diagonal grad_sigma (B, N, K)
        assert grad_sigma.shape == (B, N, K)

    def test_exact_vs_approximate_at_identity(self, cpu_device):
        """When phi=0 (Omega=I), exact and approximate paths must agree.

        With identity transport, diag(I @ diag(sigma) @ I^T) = sigma, so the
        diagonal approximation is exact.  Any discrepancy indicates a wiring bug.
        """
        B, N, K = 1, 3, 4
        n_gen = K * K
        mu_q, sigma_q, mu_p, sigma_p, beta, _, gen = _make_diagonal_vfe_inputs(
            B, N, K, n_gen, cpu_device, dtype=torch.float32
        )
        phi = torch.zeros(B, N, n_gen, device=cpu_device, dtype=torch.float32)

        kwargs = dict(
            mu_q=mu_q, sigma_q=sigma_q, mu_p=mu_p, sigma_p=sigma_p,
            beta=beta, phi=phi, generators=gen,
            alpha=0.5, lambda_belief=1.0, lambda_softmax=1.0,
            kappa=1.0, eps=1e-8,
            irrep_dims=[K],
        )
        grad_mu_approx, grad_sigma_approx = compute_vfe_gradients_gpu(
            **kwargs, exact_diagonal_transport=False,
        )
        grad_mu_exact, grad_sigma_exact = compute_vfe_gradients_gpu(
            **kwargs, exact_diagonal_transport=True,
        )
        # Relaxed tolerance: the two paths use different numerics (Cholesky vs
        # direct diagonal division), so float32 agreement is ~2e-3 depending
        # on the random seed and KL magnitude.
        assert torch.allclose(grad_mu_approx, grad_mu_exact, atol=2e-3), \
            f"grad_mu mismatch at phi=0: max dev {(grad_mu_approx - grad_mu_exact).abs().max():.2e}"
        assert torch.allclose(grad_sigma_approx, grad_sigma_exact, atol=2e-3), \
            f"grad_sigma mismatch at phi=0: max dev {(grad_sigma_approx - grad_sigma_exact).abs().max():.2e}"

    def test_exact_diagonal_transport_fd_mu(self, cpu_device):
        r"""Finite-difference check for grad_mu with exact diagonal transport.

        compute_vfe_gradients_gpu returns the QUERY-SIDE partial gradient:
        for position i, it differentiates w.r.t. mu_i holding all mu_j (j!=i)
        fixed in the key/transported terms.  The FD scalar must match this
        by only perturbing mu in the query role (mu_query) while keeping the
        key role (mu_key) constant.

        Uses float64 and lambda_softmax=0 for clean FD comparison.
        """
        from transformer.core.gauge_utils import stable_matrix_exp_pair

        torch.manual_seed(20260517)
        B, N, K = 1, 3, 4
        n_gen = K * K
        mu_q, sigma_q, mu_p, sigma_p, beta, phi, gen = _make_diagonal_vfe_inputs(
            B, N, K, n_gen, cpu_device, dtype=torch.float64
        )
        phi = phi * 0.3  # Non-trivial gauge frames
        mu_key = mu_q.clone()  # Fixed key copy

        alpha, lam = 0.5, 1.0

        # Precompute transport (independent of mu_q)
        phi_mat = torch.einsum('bna,aij->bnij', phi, gen)
        exp_p, exp_np = stable_matrix_exp_pair(phi_mat)
        Omega = torch.einsum('bikl,bjlm->bijkm', exp_p, exp_np)

        def _vfe_query_side(mu_query):
            """Query-side VFE: mu_query as query, mu_key as key (fixed)."""
            sq = sigma_q.clamp(min=1e-8)
            sp = sigma_p.clamp(min=1e-8)
            # Self-coupling (only involves query mu)
            kl_self = 0.5 * (sq / sp + (mu_p - mu_query) ** 2 / sp - 1
                             + torch.log(sp) - torch.log(sq)).sum()
            # Transport KEY beliefs (fixed mu_key)
            mu_t = torch.einsum('bijkl,bjl->bijk', Omega, mu_key)
            sq_full = torch.diag_embed(sq)
            Sigma_t = torch.einsum('bijkl,bjlm,bijmn->bijkn',
                                   Omega, sq_full, Omega.transpose(-1, -2))
            # KL: query=mu_query, key=transported mu_key
            delta = mu_query[:, :, None, :] - mu_t
            Sigma_t_reg = Sigma_t + 1e-8 * torch.eye(K, device=mu_query.device, dtype=mu_query.dtype)
            Sigma_t_inv = torch.linalg.inv(Sigma_t_reg)
            sq_i = sq_full[:, :, None, :, :]
            trace = torch.einsum('bijkl,bijlk->bij', Sigma_t_inv, sq_i)
            mahal = torch.einsum('bijk,bijkl,bijl->bij', delta, Sigma_t_inv, delta)
            logdet_t = torch.linalg.slogdet(Sigma_t_reg)[1]
            logdet_i = torch.log(sq).sum(dim=-1, keepdim=True)
            kl_pairs = 0.5 * (trace + mahal - K + logdet_t - logdet_i[:, :, None, 0])
            kl_pairs = kl_pairs.clamp(min=0.0)
            return alpha * kl_self + lam * (beta * kl_pairs).sum()

        # Analytic gradient (exact transport)
        grad_mu, _ = compute_vfe_gradients_gpu(
            mu_q, sigma_q, mu_p, sigma_p, beta, phi, gen,
            alpha=alpha, lambda_belief=lam, lambda_softmax=0.0,
            kappa=1.0, eps=1e-8, irrep_dims=[K],
            exact_diagonal_transport=True,
        )

        eps_fd = 1e-5
        for b in range(B):
            for n in range(N):
                for k in range(K):
                    mu_plus = mu_q.clone(); mu_plus[b, n, k] += eps_fd
                    mu_minus = mu_q.clone(); mu_minus[b, n, k] -= eps_fd
                    fd = (_vfe_query_side(mu_plus) - _vfe_query_side(mu_minus)) / (2 * eps_fd)
                    analytic = grad_mu[b, n, k].item()
                    err = abs(fd.item() - analytic) / (abs(analytic) + 1e-8)
                    assert err < 0.02, \
                        f"FD grad_mu[{b},{n},{k}]: analytic={analytic:.6f}, fd={fd.item():.6f}, rel_err={err:.4f}"

    def test_exact_diagonal_transport_fd_sigma(self, cpu_device):
        r"""Finite-difference check for grad_sigma with exact diagonal transport.

        Same query-side convention as grad_mu: perturbs sigma_q[b,n,k] only in
        the query role (sigma_i in KL(q_i || Omega_ij q_j)).  The key role
        (sigma_j in the transported covariance) uses a fixed copy.
        """
        from transformer.core.gauge_utils import stable_matrix_exp_pair

        B, N, K = 1, 3, 4
        n_gen = K * K
        mu_q, sigma_q, mu_p, sigma_p, beta, phi, gen = _make_diagonal_vfe_inputs(
            B, N, K, n_gen, cpu_device, dtype=torch.float64
        )
        phi = phi * 0.3
        sigma_key = sigma_q.clone()  # Fixed key copy

        alpha, lam = 0.5, 1.0

        phi_mat = torch.einsum('bna,aij->bnij', phi, gen)
        exp_p, exp_np = stable_matrix_exp_pair(phi_mat)
        Omega = torch.einsum('bikl,bjlm->bijkm', exp_p, exp_np)

        def _vfe_query_side_sigma(sq_query):
            """Query-side VFE parameterized by sigma_q (query role only)."""
            sq_q = sq_query.clamp(min=1e-8)
            sq_k = sigma_key.clamp(min=1e-8)
            sp = sigma_p.clamp(min=1e-8)
            # Self-coupling uses query sigma
            kl_self = 0.5 * (sq_q / sp + (mu_p - mu_q) ** 2 / sp - 1
                             + torch.log(sp) - torch.log(sq_q)).sum()
            # Transport KEY sigma (fixed)
            sq_k_full = torch.diag_embed(sq_k)
            Sigma_t = torch.einsum('bijkl,bjlm,bijmn->bijkn',
                                   Omega, sq_k_full, Omega.transpose(-1, -2))
            mu_t = torch.einsum('bijkl,bjl->bijk', Omega, mu_q)
            delta = mu_q[:, :, None, :] - mu_t
            Sigma_t_reg = Sigma_t + 1e-8 * torch.eye(K, device=sq_query.device, dtype=sq_query.dtype)
            Sigma_t_inv = torch.linalg.inv(Sigma_t_reg)
            # Query sigma_i (what we differentiate)
            sq_q_full = torch.diag_embed(sq_q)
            sq_q_i = sq_q_full[:, :, None, :, :]
            trace = torch.einsum('bijkl,bijlk->bij', Sigma_t_inv, sq_q_i)
            mahal = torch.einsum('bijk,bijkl,bijl->bij', delta, Sigma_t_inv, delta)
            logdet_t = torch.linalg.slogdet(Sigma_t_reg)[1]
            logdet_i = torch.log(sq_q).sum(dim=-1, keepdim=True)
            kl_pairs = 0.5 * (trace + mahal - K + logdet_t - logdet_i[:, :, None, 0])
            kl_pairs = kl_pairs.clamp(min=0.0)
            return alpha * kl_self + lam * (beta * kl_pairs).sum()

        _, grad_sigma = compute_vfe_gradients_gpu(
            mu_q, sigma_q, mu_p, sigma_p, beta, phi, gen,
            alpha=alpha, lambda_belief=lam, lambda_softmax=0.0,
            kappa=1.0, eps=1e-8, irrep_dims=[K],
            exact_diagonal_transport=True,
        )

        eps_fd = 1e-5
        for b in range(B):
            for n in range(N):
                for k in range(K):
                    s_plus = sigma_q.clone(); s_plus[b, n, k] += eps_fd
                    s_minus = sigma_q.clone(); s_minus[b, n, k] -= eps_fd
                    fd = (_vfe_query_side_sigma(s_plus) - _vfe_query_side_sigma(s_minus)) / (2 * eps_fd)
                    analytic = grad_sigma[b, n, k].item()
                    err = abs(fd.item() - analytic) / (abs(analytic) + 1e-8)
                    assert err < 0.05, \
                        f"FD grad_sigma[{b},{n},{k}]: analytic={analytic:.6f}, fd={fd.item():.6f}, rel_err={err:.4f}"


# =============================================================================
# TestComputeVFEGradientsGPUFull
# =============================================================================

class TestComputeVFEGradientsGPUFull:
    """compute_vfe_gradients_gpu with full (K, K) covariance."""

    def test_output_shapes(self, cpu_device):
        """Full cov returns (B, N, K, K) grad_sigma."""
        B, N, K = 1, 3, 3
        n_gen = K * K
        mu_q = torch.randn(B, N, K, device=cpu_device)
        A = torch.randn(B, N, K, K, device=cpu_device) * 0.3
        sigma_q = A @ A.transpose(-1, -2) + 0.2 * torch.eye(K)
        mu_p = torch.randn(B, N, K, device=cpu_device)
        sigma_p = _make_spd(B, N, K, cpu_device)
        beta = torch.softmax(torch.randn(B, N, N), dim=-1)
        phi = torch.randn(B, N, n_gen) * 0.1
        gen = torch.zeros(n_gen, K, K)
        for g in range(n_gen):
            i, j = divmod(g, K)
            gen[g, i, j] = 1.0

        grad_mu, grad_sigma = compute_vfe_gradients_gpu(
            mu_q, sigma_q, mu_p, sigma_p, beta, phi, gen,
            alpha=0.01, lambda_belief=1.0, kappa=1.0, eps=1e-6,
        )
        assert grad_mu.shape == (B, N, K)
        assert grad_sigma.shape == (B, N, K, K)

    def test_output_finite(self, cpu_device):
        """Full cov gradients are finite."""
        B, N, K = 1, 3, 3
        n_gen = K * K
        mu_q = torch.randn(B, N, K)
        sigma_q = _make_spd(B, N, K, cpu_device)
        mu_p = torch.randn(B, N, K)
        sigma_p = _make_spd(B, N, K, cpu_device)
        beta = torch.softmax(torch.randn(B, N, N), dim=-1)
        phi = torch.randn(B, N, n_gen) * 0.1
        gen = torch.zeros(n_gen, K, K)
        for g in range(n_gen):
            i, j = divmod(g, K)
            gen[g, i, j] = 1.0

        grad_mu, grad_sigma = compute_vfe_gradients_gpu(
            mu_q, sigma_q, mu_p, sigma_p, beta, phi, gen,
            alpha=0.01, lambda_belief=1.0, kappa=1.0, eps=1e-6,
        )
        assert torch.isfinite(grad_mu).all(), "NaN/Inf in full-cov grad_mu"
        assert torch.isfinite(grad_sigma).all(), "NaN/Inf in full-cov grad_sigma"

    def test_grad_sigma_symmetric(self, cpu_device):
        """Full-cov grad_sigma should be symmetric (tangent to SPD manifold)."""
        B, N, K = 1, 3, 3
        n_gen = K * K
        mu_q = torch.randn(B, N, K)
        sigma_q = _make_spd(B, N, K, cpu_device)
        mu_p = torch.randn(B, N, K)
        sigma_p = _make_spd(B, N, K, cpu_device)
        beta = torch.softmax(torch.randn(B, N, N), dim=-1)
        phi = torch.randn(B, N, n_gen) * 0.1
        gen = torch.zeros(n_gen, K, K)
        for g in range(n_gen):
            i, j = divmod(g, K)
            gen[g, i, j] = 1.0

        _, grad_sigma = compute_vfe_gradients_gpu(
            mu_q, sigma_q, mu_p, sigma_p, beta, phi, gen,
            alpha=0.01, lambda_belief=0.0, lambda_softmax=0.0,
            kappa=1.0, eps=1e-6,
        )
        asym = (grad_sigma - grad_sigma.transpose(-1, -2)).abs().max()
        assert asym < 1e-4, f"grad_sigma not symmetric: max asymmetry {asym:.2e}"


# =============================================================================
# TestComputeNaturalGradientGPU
# =============================================================================

class TestComputeNaturalGradientGPU:
    """compute_natural_gradient_gpu: Fisher-metric natural gradient projection."""

    def test_diagonal_mu_projection(self, cpu_device):
        """nat_grad_mu = sigma * grad_mu (element-wise for diagonal)."""
        B, N, K = 2, 4, 6
        grad_mu = torch.randn(B, N, K)
        grad_sigma = torch.randn(B, N, K)
        sigma_q = torch.rand(B, N, K).clamp(min=0.1) + 0.2

        nat_mu, nat_sigma = compute_natural_gradient_gpu(grad_mu, grad_sigma, sigma_q)
        expected_mu = sigma_q * grad_mu
        assert torch.allclose(nat_mu, expected_mu, atol=1e-4), \
            f"Diagonal nat_grad_mu: max dev {(nat_mu - expected_mu).abs().max():.2e}"

    def test_diagonal_sigma_projection(self, cpu_device):
        """nat_grad_sigma = 2 * sigma^2 * grad_sigma (element-wise)."""
        B, N, K = 2, 4, 6
        grad_mu = torch.randn(B, N, K)
        grad_sigma = torch.randn(B, N, K)
        sigma_q = torch.rand(B, N, K).clamp(min=0.1) + 0.2

        _, nat_sigma = compute_natural_gradient_gpu(grad_mu, grad_sigma, sigma_q)
        expected = 2.0 * sigma_q * sigma_q * grad_sigma
        assert torch.allclose(nat_sigma, expected, atol=1e-4), \
            f"Diagonal nat_grad_sigma: max dev {(nat_sigma - expected).abs().max():.2e}"

    def test_full_cov_mu_projection(self, cpu_device):
        """nat_grad_mu = Sigma @ grad_mu (matrix multiply for full cov)."""
        B, N, K = 1, 2, 3
        grad_mu = torch.randn(B, N, K)
        grad_sigma = torch.randn(B, N, K, K)
        sigma_q = _make_spd(B, N, K, cpu_device)

        nat_mu, _ = compute_natural_gradient_gpu(grad_mu, grad_sigma, sigma_q)
        expected = torch.einsum('bnij,bnj->bni', sigma_q, grad_mu)
        assert torch.allclose(nat_mu, expected, atol=1e-4), \
            f"Full-cov nat_grad_mu: max dev {(nat_mu - expected).abs().max():.2e}"

    def test_full_cov_sigma_projection(self, cpu_device):
        """nat_grad_sigma = 2 * Sigma @ grad_sigma @ Sigma."""
        B, N, K = 1, 2, 3
        grad_mu = torch.randn(B, N, K)
        grad_sigma = torch.randn(B, N, K, K)
        # Symmetrize grad_sigma
        grad_sigma = 0.5 * (grad_sigma + grad_sigma.transpose(-1, -2))
        sigma_q = _make_spd(B, N, K, cpu_device)

        _, nat_sigma = compute_natural_gradient_gpu(grad_mu, grad_sigma, sigma_q)
        expected = 2.0 * torch.einsum('bnij,bnjk,bnkl->bnil', sigma_q, grad_sigma, sigma_q)
        assert torch.allclose(nat_sigma, expected, atol=1e-3), \
            f"Full-cov nat_grad_sigma: max dev {(nat_sigma - expected).abs().max():.2e}"

    def test_output_dtype_preserved(self, cpu_device):
        """Output dtype matches input dtype."""
        B, N, K = 1, 2, 4
        grad_mu = torch.randn(B, N, K)
        grad_sigma = torch.randn(B, N, K)
        sigma_q = torch.rand(B, N, K).clamp(min=0.1)

        nat_mu, nat_sigma = compute_natural_gradient_gpu(grad_mu, grad_sigma, sigma_q)
        assert nat_mu.dtype == grad_mu.dtype
        assert nat_sigma.dtype == grad_sigma.dtype

    def test_squeeze_trailing_singletons(self, cpu_device):
        """(B, N, K, 1) sigma is handled via squeeze."""
        B, N, K = 1, 2, 4
        grad_mu = torch.randn(B, N, K)
        grad_sigma = torch.randn(B, N, K)
        sigma_q = torch.rand(B, N, K, 1).clamp(min=0.1)  # trailing singleton

        nat_mu, nat_sigma = compute_natural_gradient_gpu(grad_mu, grad_sigma, sigma_q)
        assert nat_mu.shape == (B, N, K)
        assert torch.isfinite(nat_mu).all()


# =============================================================================
# TestBlockDiagonalDispatch
# =============================================================================

class TestBlockDiagonalDispatch:
    """Block-diagonal path dispatch in compute_vfe_gradients_gpu."""

    def test_irrep_dims_routes_to_block_diagonal(self, cpu_device):
        """Setting irrep_dims with diagonal sigma routes to block-diagonal path.

        Uses multi-irrep generators from generate_multi_irrep_generators so that
        the block structure is consistent (n_gen generators, each K x K, with
        block-diagonal structure matching irrep_dims).
        """
        from math_utils.generators import generate_multi_irrep_generators
        # Two spin-1 (dim=3) blocks: K = 3 + 3 = 6, n_gen = 3 (SO(3))
        irrep_dims = [3, 3]
        K = sum(irrep_dims)
        gen_np = generate_multi_irrep_generators([('l1', 2, 3)])
        gen = torch.from_numpy(gen_np).float().to(cpu_device)
        n_gen = gen.shape[0]

        B, N = 1, 4
        mu_q = torch.randn(B, N, K)
        sigma_q = torch.rand(B, N, K).clamp(min=0.2) + 0.1
        mu_p = torch.randn(B, N, K)
        sigma_p = torch.rand(B, N, K).clamp(min=0.2) + 0.1
        beta = torch.softmax(torch.randn(B, N, N), dim=-1)
        phi = torch.randn(B, N, n_gen) * 0.1

        grad_mu, grad_sigma = compute_vfe_gradients_gpu(
            mu_q, sigma_q, mu_p, sigma_p, beta, phi, gen,
            alpha=0.01, lambda_belief=1.0, kappa=1.0, eps=1e-6,
            irrep_dims=irrep_dims,
        )
        assert grad_mu.shape == (B, N, K)
        assert grad_sigma.shape == (B, N, K)
        assert torch.isfinite(grad_mu).all(), "Block-diag grad_mu has NaN/Inf"
        assert torch.isfinite(grad_sigma).all(), "Block-diag grad_sigma has NaN/Inf"

    def test_block_diagonal_full_cov(self, cpu_device):
        """Block-diagonal with full covariance produces finite results."""
        from math_utils.generators import generate_multi_irrep_generators
        irrep_dims = [3, 3]
        K = sum(irrep_dims)
        gen_np = generate_multi_irrep_generators([('l1', 2, 3)])
        gen = torch.from_numpy(gen_np).float().to(cpu_device)
        n_gen = gen.shape[0]

        B, N = 1, 3
        mu_q = torch.randn(B, N, K)
        A = torch.randn(B, N, K, K) * 0.2
        sigma_q = A @ A.transpose(-1, -2) + 0.2 * torch.eye(K)
        mu_p = torch.randn(B, N, K)
        sigma_p = _make_spd(B, N, K, cpu_device)
        beta = torch.softmax(torch.randn(B, N, N), dim=-1)
        phi = torch.randn(B, N, n_gen) * 0.1

        grad_mu, grad_sigma = compute_vfe_gradients_gpu(
            mu_q, sigma_q, mu_p, sigma_p, beta, phi, gen,
            alpha=0.01, lambda_belief=1.0, kappa=1.0, eps=1e-6,
            irrep_dims=irrep_dims,
        )
        assert grad_mu.shape == (B, N, K)
        assert grad_sigma.shape == (B, N, K, K)
        assert torch.isfinite(grad_mu).all()
        assert torch.isfinite(grad_sigma).all()


# =============================================================================
# Helper for full-covariance tests
# =============================================================================

def _make_spd(B, N, K, device='cpu'):
    """Generate random SPD matrices (B, N, K, K)."""
    A = torch.randn(B, N, K, K, device=device) * 0.3
    return A @ A.transpose(-1, -2) + 0.2 * torch.eye(K, device=device)


# =============================================================================
# F4.10: causal_lower_triangle parity tests
# =============================================================================


def _fused_kernel_inputs(B, N, K, n_gen, device='cpu', dtype=torch.float32):
    """Canonical input set for _fused_attention_and_vfe_gradients_block_diag."""
    torch.manual_seed(0)
    mu_q = torch.randn(B, N, K, device=device, dtype=dtype) * 0.5
    sigma_q = torch.rand(B, N, K, device=device, dtype=dtype).clamp(min=0.3) + 0.2
    mu_p = torch.randn(B, N, K, device=device, dtype=dtype) * 0.3
    sigma_p = torch.rand(B, N, K, device=device, dtype=dtype).clamp(min=0.3) + 0.3
    phi = torch.randn(B, N, n_gen, device=device, dtype=dtype) * 0.05
    generators = torch.zeros(n_gen, K, K, device=device, dtype=dtype)
    for g in range(min(n_gen, K * K)):
        i, j = divmod(g, K)
        if i < K and j < K:
            generators[g, i, j] = 1.0
    return mu_q, sigma_q, mu_p, sigma_p, phi, generators


class TestCausalLowerTriangleParity:
    """F4.10: the lower-triangle fast path must be bit-identical to the dense
    path for beta, grad_mu, grad_sigma under a strict causal mask. kl_matrix
    differs in the upper triangle only (real KL vs zero) — lower-triangle and
    diagonal must still match.
    """

    @pytest.mark.parametrize("use_rope", [False, True])
    @pytest.mark.parametrize("mask_self_attention", [False, True])
    @pytest.mark.parametrize("alpha_div", [1.0, 0.5])
    def test_parity_with_dense(self, use_rope, mask_self_attention, alpha_div):
        from transformer.core.vfe_gradients import (
            _fused_attention_and_vfe_gradients_block_diag,
        )

        B, N, K, n_gen = 2, 8, 6, 6 * 6
        irrep_dims = [3, 3]
        mu_q, sigma_q, mu_p, sigma_p, phi, generators = _fused_kernel_inputs(
            B, N, K, n_gen
        )
        mask = torch.tril(torch.ones(N, N)).unsqueeze(0).expand(B, -1, -1).contiguous()

        shared = dict(
            mu_q=mu_q, sigma_q=sigma_q, mu_p=mu_p, sigma_p=sigma_p,
            phi=phi, generators=generators,
            alpha=1.0, lambda_belief=1.0, lambda_softmax=0.5,
            kappa=1.0, eps=1e-6, irrep_dims=irrep_dims,
            compute_sigma_align_grad=True,
            mask=mask, mask_self_attention=mask_self_attention,
            use_rope=use_rope, rope_base=10000.0,
            return_kl=True, alpha_div=alpha_div,
        )

        beta_dense, gmu_dense, gsig_dense, kl_dense = (
            _fused_attention_and_vfe_gradients_block_diag(
                **shared, causal_lower_triangle=False
            )
        )
        beta_tri, gmu_tri, gsig_tri, kl_tri = (
            _fused_attention_and_vfe_gradients_block_diag(
                **shared, causal_lower_triangle=True
            )
        )

        # beta, grad_mu, grad_sigma must be bit-identical (or extremely close
        # within fp32 noise) because beta=0 at j>i in both paths annihilates
        # any value at those positions.
        torch.testing.assert_close(beta_tri, beta_dense, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(gmu_tri, gmu_dense, rtol=1e-4, atol=1e-6)
        torch.testing.assert_close(gsig_tri, gsig_dense, rtol=1e-4, atol=1e-6)

        # kl_matrix lower triangle must match; upper triangle is zero in the
        # triangle path and real in the dense path (this is documented).
        assert kl_dense is not None and kl_tri is not None
        tril_mask = torch.tril(torch.ones(N, N, dtype=torch.bool))
        torch.testing.assert_close(
            kl_tri[:, tril_mask], kl_dense[:, tril_mask], rtol=1e-4, atol=1e-5
        )
        # Upper triangle of kl_tri must be exactly zero.
        upper_idx = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
        assert (kl_tri[:, upper_idx] == 0).all()

    def test_beta_is_row_stochastic_under_triangle(self):
        from transformer.core.vfe_gradients import (
            _fused_attention_and_vfe_gradients_block_diag,
        )
        B, N, K, n_gen = 2, 6, 4, 16
        irrep_dims = [4]
        mu_q, sigma_q, mu_p, sigma_p, phi, generators = _fused_kernel_inputs(
            B, N, K, n_gen
        )
        mask = torch.tril(torch.ones(N, N)).unsqueeze(0).expand(B, -1, -1).contiguous()
        beta, _, _, _ = _fused_attention_and_vfe_gradients_block_diag(
            mu_q=mu_q, sigma_q=sigma_q, mu_p=mu_p, sigma_p=sigma_p,
            phi=phi, generators=generators,
            alpha=1.0, lambda_belief=1.0, lambda_softmax=0.0,
            kappa=1.0, eps=1e-6, irrep_dims=irrep_dims,
            compute_sigma_align_grad=False,
            mask=mask, mask_self_attention=False,
            return_kl=False, causal_lower_triangle=True,
        )
        # Row sums = 1 within fp32 tolerance.
        row_sums = beta.sum(dim=-1)
        torch.testing.assert_close(
            row_sums, torch.ones_like(row_sums), rtol=1e-5, atol=1e-5
        )
        # Upper-triangle entries must be exactly zero (softmax of -inf logits).
        upper_idx = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
        assert (beta[:, upper_idx] == 0).all()

    def test_finite_outputs_under_triangle(self):
        from transformer.core.vfe_gradients import (
            _fused_attention_and_vfe_gradients_block_diag,
        )
        B, N, K, n_gen = 2, 5, 4, 16
        irrep_dims = [4]
        mu_q, sigma_q, mu_p, sigma_p, phi, generators = _fused_kernel_inputs(
            B, N, K, n_gen
        )
        mask = torch.tril(torch.ones(N, N)).unsqueeze(0).expand(B, -1, -1).contiguous()
        beta, gmu, gsig, kl = _fused_attention_and_vfe_gradients_block_diag(
            mu_q=mu_q, sigma_q=sigma_q, mu_p=mu_p, sigma_p=sigma_p,
            phi=phi, generators=generators,
            alpha=1.0, lambda_belief=1.0, lambda_softmax=0.5,
            kappa=1.0, eps=1e-6, irrep_dims=irrep_dims,
            compute_sigma_align_grad=True,
            mask=mask, mask_self_attention=True,
            return_kl=True, causal_lower_triangle=True,
        )
        assert torch.isfinite(beta).all()
        assert torch.isfinite(gmu).all()
        assert torch.isfinite(gsig).all()
        assert kl is not None and torch.isfinite(kl).all()


class TestCausalLowerTriangleLegacyWiring:
    """F4.10: integration tests for the legacy VariationalFFNDynamic path
    that calls the fused kernel at variational_ffn.py:1992. Verifies that
    the toggle propagates from BlockConfig/kwargs through to the kernel,
    that forward() finishes with finite outputs, and that the
    once-per-forward mask validation rejects non-causal masks.
    """

    def _make_legacy_ffn(self, K, n_gen, causal_lower_triangle):
        from transformer.core.variational_ffn import VariationalFFNDynamic
        torch.manual_seed(7)
        generators = torch.zeros(n_gen, K, K)
        for g in range(min(n_gen, K * K)):
            i, j = divmod(g, K)
            if i < K and j < K:
                generators[g, i, j] = 1.0
        return VariationalFFNDynamic(
            embed_dim=K,
            generators=generators,
            alpha=0.1,
            lambda_belief=1.0,
            kappa=1.0,
            n_iterations=2,
            diagonal_covariance=True,
            irrep_dims=[K],
            mask_self_attention=False,
            causal_lower_triangle=causal_lower_triangle,
        )

    def test_legacy_path_runs_with_toggle_on(self):
        torch.manual_seed(1)
        B, N, K, n_gen = 2, 6, 4, 16
        ffn = self._make_legacy_ffn(K, n_gen, causal_lower_triangle=True)
        assert ffn.causal_lower_triangle is True

        mu = torch.randn(B, N, K)
        sigma = torch.rand(B, N, K).clamp(min=0.3) + 0.2
        phi = torch.randn(B, N, n_gen) * 0.05
        mu_prior = torch.randn(B, N, K)
        sigma_prior = torch.rand(B, N, K).clamp(min=0.3) + 0.3
        mask = torch.tril(torch.ones(N, N)).unsqueeze(0).expand(B, -1, -1).contiguous()

        out = ffn(
            mu=mu, sigma=sigma, phi=phi,
            mu_prior=mu_prior, sigma_prior=sigma_prior,
            mask=mask,
        )
        mu_out = out[0] if isinstance(out, tuple) else out
        assert torch.isfinite(mu_out).all()

    def test_legacy_path_raises_on_none_mask_with_toggle(self):
        ffn = self._make_legacy_ffn(K=4, n_gen=16, causal_lower_triangle=True)
        B, N, K, n_gen = 1, 4, 4, 16
        mu = torch.randn(B, N, K)
        sigma = torch.rand(B, N, K).clamp(min=0.3) + 0.2
        phi = torch.randn(B, N, n_gen) * 0.05
        mu_prior = torch.randn(B, N, K)
        sigma_prior = torch.rand(B, N, K).clamp(min=0.3) + 0.3

        with pytest.raises(ValueError, match=r"causal_lower_triangle=True requires a non-None"):
            ffn(
                mu=mu, sigma=sigma, phi=phi,
                mu_prior=mu_prior, sigma_prior=sigma_prior,
                mask=None,
            )

    def test_legacy_path_raises_on_non_causal_mask_with_toggle(self):
        ffn = self._make_legacy_ffn(K=4, n_gen=16, causal_lower_triangle=True)
        B, N, K, n_gen = 1, 4, 4, 16
        mu = torch.randn(B, N, K)
        sigma = torch.rand(B, N, K).clamp(min=0.3) + 0.2
        phi = torch.randn(B, N, n_gen) * 0.05
        mu_prior = torch.randn(B, N, K)
        sigma_prior = torch.rand(B, N, K).clamp(min=0.3) + 0.3
        # All-ones (non-causal) mask should fail the lower-triangular check.
        mask = torch.ones(B, N, N)

        with pytest.raises(ValueError, match=r"strict lower-triangular mask"):
            ffn(
                mu=mu, sigma=sigma, phi=phi,
                mu_prior=mu_prior, sigma_prior=sigma_prior,
                mask=mask,
            )

    def test_legacy_path_default_off_unchanged(self):
        """Default causal_lower_triangle=False: dense path, no validation,
        non-causal mask is accepted (regression guard for users not opting
        into the fast path).
        """
        torch.manual_seed(2)
        B, N, K, n_gen = 1, 4, 4, 16
        ffn = self._make_legacy_ffn(K, n_gen, causal_lower_triangle=False)
        assert ffn.causal_lower_triangle is False

        mu = torch.randn(B, N, K)
        sigma = torch.rand(B, N, K).clamp(min=0.3) + 0.2
        phi = torch.randn(B, N, n_gen) * 0.05
        mu_prior = torch.randn(B, N, K)
        sigma_prior = torch.rand(B, N, K).clamp(min=0.3) + 0.3
        mask = torch.ones(B, N, N)  # non-causal: must NOT raise under toggle=False

        out = ffn(
            mu=mu, sigma=sigma, phi=phi,
            mu_prior=mu_prior, sigma_prior=sigma_prior,
            mask=mask,
        )
        mu_out = out[0] if isinstance(out, tuple) else out
        assert torch.isfinite(mu_out).all()
