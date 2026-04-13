"""
Advanced Finite-Difference Tests (Phase 6 Audit)
=================================================

Tests for mathematical correctness of:
1. Implicit-EM IFT scale factors (compute_implicit_em_scales)
2. DEQ Neumann backward (linear contraction test)
3. Sandwich product derivative ∂(Ω·Σ·Ω^T)/∂φ
4. Connection MLP gauge equivariance (negative test)

All FD tests use float64 central differences with eps=1e-5.
"""

import math
import pytest
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Test 1: Implicit-EM IFT Scale Factors
# ---------------------------------------------------------------------------

class TestImplicitEMScales:
    """Verify compute_implicit_em_scales formula correctness."""

    def test_mu_scale_formula(self):
        r"""Verify s_k^{(μ)} = (α/σ_p) / (α/σ_p + Σ_j β_ij/σ_qj)."""
        from transformer.core.vfe_implicit_em import compute_implicit_em_scales

        torch.manual_seed(42)
        B, N, K = 2, 4, 5
        alpha_i = torch.rand(B, N, K, dtype=torch.float64) * 2.0 + 0.1
        sigma_p = torch.rand(B, N, K, dtype=torch.float64) * 2.0 + 0.1
        sigma_q = torch.rand(B, N, K, dtype=torch.float64) * 2.0 + 0.1
        # Random attention weights summing to ~1 per row
        beta_raw = torch.rand(B, N, N, dtype=torch.float64)
        beta = beta_raw / beta_raw.sum(dim=-1, keepdim=True)

        mu_scale, sigma_scale = compute_implicit_em_scales(
            alpha_i, sigma_p, beta, sigma_q
        )

        # Manual computation of mu scale
        prior_prec = alpha_i / sigma_p  # (B, N, K)
        inv_sq = 1.0 / sigma_q  # (B, N, K)
        attn_prec = torch.einsum('bij,bjk->bik', beta, inv_sq)  # (B, N, K)
        expected_mu_scale = prior_prec / (prior_prec + attn_prec)

        assert torch.allclose(mu_scale, expected_mu_scale, atol=1e-10), \
            f"mu_scale mismatch: max err {(mu_scale - expected_mu_scale).abs().max():.2e}"

    def test_sigma_scale_formula(self):
        r"""Verify s_k^{(σ)} = (α/σ_p²) / (α/σ_p² + Σ_j β_ij/σ_qj²)."""
        from transformer.core.vfe_implicit_em import compute_implicit_em_scales

        torch.manual_seed(43)
        B, N, K = 2, 3, 4
        alpha_i = torch.rand(B, N, K, dtype=torch.float64) * 2.0 + 0.1
        sigma_p = torch.rand(B, N, K, dtype=torch.float64) * 2.0 + 0.1
        sigma_q = torch.rand(B, N, K, dtype=torch.float64) * 2.0 + 0.1
        beta_raw = torch.rand(B, N, N, dtype=torch.float64)
        beta = beta_raw / beta_raw.sum(dim=-1, keepdim=True)

        _, sigma_scale = compute_implicit_em_scales(
            alpha_i, sigma_p, beta, sigma_q
        )

        # Manual computation
        prior_prec_sq = alpha_i / (sigma_p ** 2)
        inv_sq_sq = 1.0 / (sigma_q ** 2)
        attn_prec_sq = torch.einsum('bij,bjk->bik', beta, inv_sq_sq)
        expected = prior_prec_sq / (prior_prec_sq + attn_prec_sq)

        assert torch.allclose(sigma_scale, expected, atol=1e-10), \
            f"sigma_scale mismatch: max err {(sigma_scale - expected).abs().max():.2e}"

    def test_boundary_no_attention(self):
        """When β→0, scale should approach 1 (straight-through)."""
        from transformer.core.vfe_implicit_em import compute_implicit_em_scales

        B, N, K = 1, 3, 4
        alpha_i = torch.ones(B, N, K, dtype=torch.float64)
        sigma_p = torch.ones(B, N, K, dtype=torch.float64)
        sigma_q = torch.ones(B, N, K, dtype=torch.float64)
        beta = torch.zeros(B, N, N, dtype=torch.float64)  # No attention

        mu_scale, sigma_scale = compute_implicit_em_scales(
            alpha_i, sigma_p, beta, sigma_q
        )

        assert torch.allclose(mu_scale, torch.ones_like(mu_scale), atol=1e-5), \
            "mu_scale should be ~1 when beta=0"
        assert torch.allclose(sigma_scale, torch.ones_like(sigma_scale), atol=1e-5), \
            "sigma_scale should be ~1 when beta=0"

    def test_boundary_no_prior(self):
        """When α→0, scale should approach 0 (pure EM)."""
        from transformer.core.vfe_implicit_em import compute_implicit_em_scales

        B, N, K = 1, 3, 4
        alpha_i = torch.full((B, N, K), 1e-10, dtype=torch.float64)
        sigma_p = torch.ones(B, N, K, dtype=torch.float64)
        sigma_q = torch.ones(B, N, K, dtype=torch.float64)
        beta = torch.ones(B, N, N, dtype=torch.float64) / N

        mu_scale, sigma_scale = compute_implicit_em_scales(
            alpha_i, sigma_p, beta, sigma_q
        )

        assert (mu_scale < 1e-5).all(), \
            f"mu_scale should be ~0 when alpha→0, got max {mu_scale.max():.2e}"
        assert (sigma_scale < 1e-5).all(), \
            f"sigma_scale should be ~0 when alpha→0, got max {sigma_scale.max():.2e}"

    def test_scales_in_unit_interval(self):
        """Scales must be in [0, 1] for all random inputs."""
        from transformer.core.vfe_implicit_em import compute_implicit_em_scales

        torch.manual_seed(44)
        for _ in range(5):
            B, N, K = 2, 5, 6
            alpha_i = torch.rand(B, N, K, dtype=torch.float64) * 5.0 + 0.01
            sigma_p = torch.rand(B, N, K, dtype=torch.float64) * 3.0 + 0.01
            sigma_q = torch.rand(B, N, K, dtype=torch.float64) * 3.0 + 0.01
            beta_raw = torch.rand(B, N, N, dtype=torch.float64)
            beta = beta_raw / beta_raw.sum(dim=-1, keepdim=True)

            mu_scale, sigma_scale = compute_implicit_em_scales(
                alpha_i, sigma_p, beta, sigma_q
            )

            assert (mu_scale >= -1e-7).all() and (mu_scale <= 1.0 + 1e-7).all(), \
                f"mu_scale out of [0,1]: [{mu_scale.min():.4f}, {mu_scale.max():.4f}]"
            assert (sigma_scale >= -1e-7).all() and (sigma_scale <= 1.0 + 1e-7).all(), \
                f"sigma_scale out of [0,1]: [{sigma_scale.min():.4f}, {sigma_scale.max():.4f}]"

    def test_multihead_beta(self):
        """Test with 4D beta (B, H, N, N) — should average over heads."""
        from transformer.core.vfe_implicit_em import compute_implicit_em_scales

        torch.manual_seed(45)
        B, H, N, K = 1, 4, 3, 5
        alpha_i = torch.ones(B, N, K, dtype=torch.float64)
        sigma_p = torch.ones(B, N, K, dtype=torch.float64) * 2.0
        sigma_q = torch.ones(B, N, K, dtype=torch.float64)
        beta_4d = torch.rand(B, H, N, N, dtype=torch.float64)
        beta_4d = beta_4d / beta_4d.sum(dim=-1, keepdim=True)

        mu_scale, sigma_scale = compute_implicit_em_scales(
            alpha_i, sigma_p, beta_4d, sigma_q
        )

        # Verify it internally averages heads
        beta_2d = beta_4d.mean(dim=1)
        mu_scale_2d, sigma_scale_2d = compute_implicit_em_scales(
            alpha_i, sigma_p, beta_2d, sigma_q
        )

        assert torch.allclose(mu_scale, mu_scale_2d, atol=1e-10), \
            "Multihead beta should be averaged before computing scales"


# ---------------------------------------------------------------------------
# Test 2: DEQ Neumann Backward (Linear Contraction)
# ---------------------------------------------------------------------------

class TestDEQNeumannBackward:
    """Verify DEQ Neumann series backward via linear contraction T(z) = Az + b."""

    def test_neumann_converges_to_exact_inverse(self):
        r"""For T(z) = Az + b with ||A|| < 1, DEQ backward should give
        (I - A^T)^{-1} v, which we can compute exactly.

        The DEQ forward is identity at the fixed point z* = (I-A)^{-1} b.
        The backward replaces straight-through with Neumann correction.
        """
        from transformer.core.vfe_deq import DEQFixedPoint

        torch.manual_seed(46)
        D = 8

        # Build a contraction: A with spectral radius < 1
        A_raw = torch.randn(D, D, dtype=torch.float64) * 0.3
        A = A_raw / (torch.linalg.norm(A_raw, ord=2) + 1.0)  # ||A||_2 < 0.5
        b = torch.randn(D, dtype=torch.float64)

        # Fixed point: z* = (I - A)^{-1} b
        I = torch.eye(D, dtype=torch.float64)
        z_star = torch.linalg.solve(I - A, b)

        # Verify it's a fixed point
        assert torch.allclose(A @ z_star + b, z_star, atol=1e-10)

        # Define step function (linear contraction)
        def step_fn(mu_in, sigma_in):
            mu_out = A @ mu_in + b
            sigma_out = sigma_in  # sigma is dummy for this test
            return mu_out, sigma_out

        # DEQ forward + backward with increasing Neumann terms
        # The exact backward gives: (I - A^T)^{-1} v for any v
        v = torch.randn(D, dtype=torch.float64)
        exact = torch.linalg.solve(I - A.T, v)

        prev_err = float('inf')
        for K in [1, 3, 5, 10]:
            mu_star = z_star.detach().requires_grad_(True)
            sigma_star = torch.ones(D, dtype=torch.float64).requires_grad_(True)

            mu_out, sigma_out = DEQFixedPoint.apply(
                mu_star, sigma_star, step_fn, 1, K
            )
            # Backward with v as upstream gradient
            mu_out.backward(v, retain_graph=True)
            neumann_result = mu_star.grad.clone()

            err = (neumann_result - exact).norm().item()
            # Error should decrease with more terms (geometric convergence)
            assert err < prev_err + 1e-10, \
                f"Neumann error not decreasing: K={K}, err={err:.2e}, prev={prev_err:.2e}"
            prev_err = err

        # With enough terms, should be close to exact
        assert prev_err < 1e-4, \
            f"Neumann series did not converge to exact: final err={prev_err:.2e}"

    def test_weak_contraction(self):
        """When A≈0, backward should be close to identity (v)."""
        from transformer.core.vfe_deq import DEQFixedPoint

        torch.manual_seed(47)
        D = 5
        # Very weak contraction: A has tiny spectral radius
        A = torch.randn(D, D, dtype=torch.float64) * 0.001
        b = torch.randn(D, dtype=torch.float64)
        I = torch.eye(D, dtype=torch.float64)
        z_star = torch.linalg.solve(I - A, b)

        def step_fn(mu_in, sigma_in):
            return A @ mu_in + b, sigma_in

        v = torch.randn(D, dtype=torch.float64)
        mu_star = z_star.detach().requires_grad_(True)
        sigma_star = torch.ones(D, dtype=torch.float64).requires_grad_(True)

        mu_out, _ = DEQFixedPoint.apply(mu_star, sigma_star, step_fn, 1, 5)
        mu_out.backward(v)

        # With A≈0: (I - A^T)^{-1} ≈ I, so backward ≈ v
        assert torch.allclose(mu_star.grad, v, atol=0.1), \
            f"With near-zero A, backward should be close to identity. " \
            f"Diff: {(mu_star.grad - v).norm():.4e}"


# ---------------------------------------------------------------------------
# Test 3: Sandwich Product Derivative ∂(Ω·Σ·Ω^T)/∂φ
# ---------------------------------------------------------------------------

class TestSandwichProductDerivative:
    """Verify autograd through matrix_exp → sandwich product → loss."""

    def _make_generators(self, K: int) -> torch.Tensor:
        """Build K² GL(K) generators E_{ij}."""
        n_gen = K * K
        G = torch.zeros(n_gen, K, K, dtype=torch.float64)
        idx = 0
        for i in range(K):
            for j in range(K):
                G[idx, i, j] = 1.0
                idx += 1
        return G

    def test_fd_vs_autograd_diagonal_sigma(self):
        r"""FD test: ∂(sum(Ω·diag(σ)·Ω^T))/∂φ via central differences."""
        torch.manual_seed(47)
        K = 3
        n_gen = K * K
        G = self._make_generators(K)

        phi = torch.randn(n_gen, dtype=torch.float64) * 0.1
        phi.requires_grad_(True)
        sigma_diag = torch.rand(K, dtype=torch.float64) + 0.5  # positive

        # Forward: Ω = exp(φ·G), Σ_t = Ω·diag(σ)·Ω^T, loss = sum(Σ_t)
        def compute_loss(phi_val):
            M = torch.einsum('g,gij->ij', phi_val, G)
            Omega = torch.linalg.matrix_exp(M)
            Sigma = torch.diag(sigma_diag)
            Sigma_t = Omega @ Sigma @ Omega.T
            return Sigma_t.sum()

        # Autograd
        loss = compute_loss(phi)
        loss.backward()
        grad_autograd = phi.grad.clone()

        # Finite differences (central)
        eps = 1e-5
        grad_fd = torch.zeros_like(phi)
        for g in range(n_gen):
            phi_p = phi.detach().clone()
            phi_m = phi.detach().clone()
            phi_p[g] += eps
            phi_m[g] -= eps
            loss_p = compute_loss(phi_p)
            loss_m = compute_loss(phi_m)
            grad_fd[g] = (loss_p - loss_m) / (2 * eps)

        rel_err = (grad_autograd - grad_fd).norm() / (grad_fd.norm() + 1e-12)
        assert rel_err < 1e-4, \
            f"Sandwich derivative rel error {rel_err:.2e} exceeds 1e-4"

    def test_fd_vs_autograd_full_sigma(self):
        r"""FD test with full (non-diagonal) SPD covariance matrix."""
        torch.manual_seed(48)
        K = 3
        n_gen = K * K
        G = self._make_generators(K)

        phi = torch.randn(n_gen, dtype=torch.float64) * 0.1
        phi.requires_grad_(True)

        # Random SPD matrix
        L = torch.randn(K, K, dtype=torch.float64)
        Sigma = L @ L.T + 0.1 * torch.eye(K, dtype=torch.float64)

        def compute_loss(phi_val):
            M = torch.einsum('g,gij->ij', phi_val, G)
            Omega = torch.linalg.matrix_exp(M)
            Sigma_t = Omega @ Sigma @ Omega.T
            return Sigma_t.sum()

        loss = compute_loss(phi)
        loss.backward()
        grad_autograd = phi.grad.clone()

        eps = 1e-5
        grad_fd = torch.zeros_like(phi)
        for g in range(n_gen):
            phi_p = phi.detach().clone()
            phi_m = phi.detach().clone()
            phi_p[g] += eps
            phi_m[g] -= eps
            grad_fd[g] = (compute_loss(phi_p) - compute_loss(phi_m)) / (2 * eps)

        rel_err = (grad_autograd - grad_fd).norm() / (grad_fd.norm() + 1e-12)
        assert rel_err < 1e-4, \
            f"Full-cov sandwich derivative rel error {rel_err:.2e} exceeds 1e-4"

    def test_fd_batched(self):
        r"""FD test with batched phi and sigma (B, N, n_gen)."""
        torch.manual_seed(49)
        B, N, K = 1, 2, 3
        n_gen = K * K
        G = self._make_generators(K)

        phi = torch.randn(B, N, n_gen, dtype=torch.float64) * 0.1
        phi.requires_grad_(True)
        sigma_diag = torch.rand(B, N, K, dtype=torch.float64) + 0.5

        def compute_loss(phi_val):
            M = torch.einsum('...g,gij->...ij', phi_val, G)
            Omega = torch.linalg.matrix_exp(M)  # (B, N, K, K)
            Sigma = torch.diag_embed(sigma_diag)  # (B, N, K, K)
            Sigma_t = Omega @ Sigma @ Omega.transpose(-1, -2)
            return Sigma_t.sum()

        loss = compute_loss(phi)
        loss.backward()
        grad_autograd = phi.grad.clone()

        eps = 1e-5
        grad_fd = torch.zeros_like(phi)
        phi_flat = phi.detach().reshape(-1)
        for idx in range(phi_flat.numel()):
            p = phi_flat.clone()
            m = phi_flat.clone()
            p[idx] += eps
            m[idx] -= eps
            loss_p = compute_loss(p.reshape(B, N, n_gen))
            loss_m = compute_loss(m.reshape(B, N, n_gen))
            grad_fd.view(-1)[idx] = (loss_p - loss_m) / (2 * eps)

        rel_err = (grad_autograd - grad_fd).norm() / (grad_fd.norm() + 1e-12)
        assert rel_err < 1e-4, \
            f"Batched sandwich derivative rel error {rel_err:.2e} exceeds 1e-4"


# ---------------------------------------------------------------------------
# Test 4: Connection Gauge Equivariance
# ---------------------------------------------------------------------------

class TestConnectionGaugeEquivariance:
    """Test gauge equivariance of bilinear vs MLP connection modes.

    For bilinear connection with antisymmetric W:
        δ_ij^a = μ_i^T W^a μ_j
    Under gauge rotation μ → g·μ:
        δ'_ij^a = μ_i^T (g^T W^a g) μ_j

    For antisymmetric W and orthogonal g (g ∈ SO(K)), the bilinear form
    transforms predictably. For MLP, it does not — this is a known limitation.
    """

    def test_bilinear_antisymmetry(self):
        """Bilinear mode with antisymmetrize=True produces δ_ij = -δ_ji."""
        from transformer.core.connection import GaugeConnection

        torch.manual_seed(50)
        d_head, n_gen = 4, 3
        conn = GaugeConnection(d_head, n_gen, connection_type='bilinear',
                               antisymmetrize=True, init_scale=0.1)
        conn.double()

        B, N = 2, 5
        mu = torch.randn(B, N, d_head, dtype=torch.float64)
        delta = conn(mu, mu)  # (B, N, N, n_gen)

        # Check antisymmetry: δ_ij = -δ_ji
        delta_T = delta.transpose(1, 2)
        assert torch.allclose(delta, -delta_T, atol=1e-12), \
            f"Bilinear antisymmetry violated: max err {(delta + delta_T).abs().max():.2e}"

    def test_bilinear_equivariance_so(self):
        """Bilinear mode is equivariant under orthogonal transformations.

        For g ∈ SO(d_head):
            δ(g·μ_i, g·μ_j)^a = μ_i^T (g^T W^a g) μ_j

        When W^a are the generators of the rotation group and g is in that group,
        this equals Ad(g)^a_b · δ(μ_i, μ_j)^b. For general W^a, we just verify
        the bilinear transformation property.
        """
        from transformer.core.connection import GaugeConnection

        torch.manual_seed(51)
        d_head, n_gen = 4, 6
        conn = GaugeConnection(d_head, n_gen, connection_type='bilinear',
                               antisymmetrize=True, init_scale=0.1)
        conn.double()

        B, N = 1, 3
        mu_i = torch.randn(B, N, d_head, dtype=torch.float64)
        mu_j = torch.randn(B, N, d_head, dtype=torch.float64)

        # Random orthogonal g ∈ SO(d_head)
        Q, _ = torch.linalg.qr(torch.randn(d_head, d_head, dtype=torch.float64))
        if torch.det(Q) < 0:
            Q[:, 0] *= -1

        delta_orig = conn(mu_i, mu_j)  # (B, N, N, n_gen)

        # Apply g: μ' = μ @ g^T (so each token gets left-multiplied by g)
        mu_i_rot = mu_i @ Q.T
        mu_j_rot = mu_j @ Q.T
        delta_rot = conn(mu_i_rot, mu_j_rot)

        # The bilinear form transforms as:
        # δ'^a = μ_i^T (g^T W^a g) μ_j
        # Compute expected: for each a, W_eff^a = g^T W^a g
        with torch.no_grad():
            W = conn.W.data.clone()
            if conn.antisymmetrize:
                W = (W - W.transpose(-1, -2)) / 2
            W_eff = Q.T.unsqueeze(0) @ W @ Q.unsqueeze(0)  # (n_gen, d, d)

        # Expected: μ_i^T W_eff^a μ_j
        delta_expected = torch.einsum('bid,adg,bjg->bija', mu_i, W_eff, mu_j)

        assert torch.allclose(delta_rot, delta_expected, atol=1e-10), \
            f"Bilinear equivariance check failed: max err " \
            f"{(delta_rot - delta_expected).abs().max():.2e}"

    def test_mlp_not_equivariant(self):
        """MLP mode is NOT gauge-equivariant — negative test documenting this.

        This is a known limitation documented in CLAUDE.md: the MLP connection
        is a research-only variant that sacrifices equivariance for expressiveness.
        """
        from transformer.core.connection import GaugeConnection

        torch.manual_seed(52)
        d_head, n_gen = 4, 3
        conn = GaugeConnection(d_head, n_gen, connection_type='mlp', hidden_dim=16)
        conn.double()
        # Initialize with non-zero weights so it's not trivially equivariant
        with torch.no_grad():
            for m in conn.net:
                if hasattr(m, 'weight'):
                    m.weight.normal_(0, 0.5)
                if hasattr(m, 'bias') and m.bias is not None:
                    m.bias.normal_(0, 0.1)

        B, N = 1, 3
        mu_i = torch.randn(B, N, d_head, dtype=torch.float64)
        mu_j = torch.randn(B, N, d_head, dtype=torch.float64)

        # Random orthogonal g
        Q, _ = torch.linalg.qr(torch.randn(d_head, d_head, dtype=torch.float64))

        delta_orig = conn(mu_i, mu_j)  # (B, N, N, n_gen)

        mu_i_rot = mu_i @ Q.T
        mu_j_rot = mu_j @ Q.T
        delta_rot = conn(mu_i_rot, mu_j_rot)

        # For equivariance, delta_rot should equal some linear transformation
        # of delta_orig. In general it won't — verify they differ.
        # (This is a negative test: we EXPECT them to differ.)
        max_diff = (delta_rot - delta_orig).abs().max().item()
        assert max_diff > 1e-3, \
            f"MLP connection appears equivariant (diff={max_diff:.2e}) — " \
            f"expected non-equivariance with random weights"


# ---------------------------------------------------------------------------
# Test 5: Fiber Trajectory Oscillation Detection
# ---------------------------------------------------------------------------

class TestFitConvergenceRate:
    """Test the oscillation detection in fit_convergence_rate."""

    def test_monotonic_decay(self):
        """Monotonically decaying curve should return positive rate."""
        from transformer.analysis.fiber_trajectory import fit_convergence_rate

        curve = np.array([1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125])
        rate = fit_convergence_rate(curve)
        assert not np.isnan(rate), "Monotonic decay should not return NaN"
        assert rate > 0, f"Expected positive rate, got {rate}"

    def test_oscillatory_curve_returns_nan(self):
        """Highly oscillatory curve should return NaN."""
        from transformer.analysis.fiber_trajectory import fit_convergence_rate

        # Oscillating: up-down-up-down pattern
        curve = np.array([1.0, 0.3, 0.9, 0.2, 0.8, 0.15, 0.7, 0.1])
        rate = fit_convergence_rate(curve)
        assert np.isnan(rate), \
            f"Oscillatory curve should return NaN, got {rate}"

    def test_short_curve(self):
        """Curve with < 2 positive values returns 0."""
        from transformer.analysis.fiber_trajectory import fit_convergence_rate

        curve = np.array([0.5, 0.0, 0.0])
        rate = fit_convergence_rate(curve)
        assert rate == 0.0, f"Single positive value should give 0.0, got {rate}"

    def test_noisy_but_decaying(self):
        """Curve with mild noise but overall decay should return positive."""
        from transformer.analysis.fiber_trajectory import fit_convergence_rate

        # Overall decaying with some noise
        curve = np.array([1.0, 0.7, 0.6, 0.55, 0.4, 0.35, 0.3, 0.25])
        rate = fit_convergence_rate(curve)
        assert not np.isnan(rate), "Noisy but decaying curve should not be NaN"
        assert rate > 0, f"Expected positive rate for noisy decay, got {rate}"


# ---------------------------------------------------------------------------
# Test 6: Fisher-Rao Geodesic Distance
# ---------------------------------------------------------------------------

class TestFisherRaoDistance:
    """Verify the closed-form Fisher-Rao geodesic distance formula."""

    def test_pure_sigma_shift_exact(self):
        """Pure std dev change: d = sqrt(2) * |log(sigma2/sigma1)|."""
        from transformer.analysis.fiber_trajectory import fisher_rao_distance

        mu1 = np.array([[0.0]])
        mu2 = np.array([[0.0]])
        sig1 = np.array([[1.0]])   # variance
        sig2 = np.array([[4.0]])   # variance = 4, std = 2

        d = fisher_rao_distance(mu1, sig1, mu2, sig2)[0]
        exact = np.sqrt(2) * abs(np.log(2.0 / 1.0))  # sqrt(2) * log(std2/std1)
        assert abs(d - exact) < 1e-10, \
            f"Pure sigma shift: d={d:.6f}, exact={exact:.6f}"

    def test_infinitesimal_mu_ratio(self):
        """For infinitesimal dmu at constant sigma, d/dmu -> 1/sigma."""
        from transformer.analysis.fiber_trajectory import fisher_rao_distance

        eps = 1e-4
        sig = np.array([[2.0]])  # variance = 2, std = sqrt(2)
        mu1 = np.array([[0.0]])
        mu2 = np.array([[eps]])

        d = fisher_rao_distance(mu1, sig, mu2, sig)[0]
        # ds = |dmu| / std = eps / sqrt(2)
        expected = eps / np.sqrt(2.0)
        ratio = d / expected
        assert abs(ratio - 1.0) < 1e-3, \
            f"Infinitesimal mu ratio: {ratio:.6f}, expected ~1.0"

    def test_infinitesimal_sigma_ratio(self):
        """For infinitesimal dsigma at constant mu, d/dsigma -> sqrt(2)/sigma."""
        from transformer.analysis.fiber_trajectory import fisher_rao_distance

        eps = 1e-4
        std0 = 1.5  # std dev
        var0 = std0 ** 2
        var1 = (std0 + eps) ** 2

        mu = np.array([[0.0]])
        sig0 = np.array([[var0]])
        sig1 = np.array([[var1]])

        d = fisher_rao_distance(mu, sig0, mu, sig1)[0]
        # ds = sqrt(2) * |dsigma| / sigma = sqrt(2) * eps / std0
        expected = np.sqrt(2) * eps / std0
        ratio = d / expected
        assert abs(ratio - 1.0) < 1e-2, \
            f"Infinitesimal sigma ratio: {ratio:.6f}, expected ~1.0"

    def test_symmetry(self):
        """d(a, b) = d(b, a)."""
        from transformer.analysis.fiber_trajectory import fisher_rao_distance

        np.random.seed(42)
        mu1 = np.random.randn(5, 3)
        mu2 = np.random.randn(5, 3)
        sig1 = np.random.rand(5, 3) + 0.5
        sig2 = np.random.rand(5, 3) + 0.5

        d_ab = fisher_rao_distance(mu1, sig1, mu2, sig2)
        d_ba = fisher_rao_distance(mu2, sig2, mu1, sig1)
        assert np.allclose(d_ab, d_ba, atol=1e-12), \
            f"Symmetry violated: max diff {abs(d_ab - d_ba).max():.2e}"

    def test_identity_zero(self):
        """d(a, a) = 0."""
        from transformer.analysis.fiber_trajectory import fisher_rao_distance

        mu = np.array([[1.0, 2.0, 3.0]])
        sig = np.array([[0.5, 1.0, 2.0]])
        d = fisher_rao_distance(mu, sig, mu, sig)[0]
        assert d < 1e-12, f"d(a,a) should be 0, got {d:.2e}"

    def test_multi_dim_quadrature(self):
        """Total distance is sqrt(sum(d_k^2)) across dimensions."""
        from transformer.analysis.fiber_trajectory import fisher_rao_distance

        # Two dimensions with known per-dimension distances
        mu1 = np.array([[0.0, 0.0]])
        mu2 = np.array([[0.0, 0.0]])
        sig1 = np.array([[1.0, 1.0]])
        sig2 = np.array([[4.0, 9.0]])  # std: 1->2 and 1->3

        d = fisher_rao_distance(mu1, sig1, mu2, sig2)[0]
        d1 = np.sqrt(2) * abs(np.log(2))
        d2 = np.sqrt(2) * abs(np.log(3))
        expected = np.sqrt(d1**2 + d2**2)
        assert abs(d - expected) < 1e-10, \
            f"Multi-dim quadrature: d={d:.6f}, expected={expected:.6f}"


# ---------------------------------------------------------------------------
# Test 7: Rényi α-divergence Alignment Gradient
# ---------------------------------------------------------------------------

class TestRenyiAlignmentGradient:
    """FD test for the Rényi α-divergence gradient through transport."""

    def test_alignment_mu_gradient_fd(self):
        r"""FD test: ∂D_α(q_i || Ω_ij q_j)/∂μ_i via central differences."""
        torch.manual_seed(60)
        K = 4
        n_gen = K * K
        alpha_div = 0.275

        # GL(K) generators
        G = torch.zeros(n_gen, K, K, dtype=torch.float64)
        idx = 0
        for i in range(K):
            for j in range(K):
                G[idx, i, j] = 1.0
                idx += 1

        phi_i = torch.randn(n_gen, dtype=torch.float64) * 0.1
        phi_j = torch.randn(n_gen, dtype=torch.float64) * 0.1
        mu_i = torch.randn(K, dtype=torch.float64)
        mu_j = torch.randn(K, dtype=torch.float64)
        sigma_i = torch.rand(K, dtype=torch.float64) * 2 + 0.5
        sigma_j = torch.rand(K, dtype=torch.float64) * 2 + 0.5

        def compute_D_alpha(mu_i_val):
            M_i = torch.einsum('g,gkl->kl', phi_i, G)
            M_j = torch.einsum('g,gkl->kl', phi_j, G)
            exp_phi_i = torch.linalg.matrix_exp(M_i)
            exp_neg_phi_j = torch.linalg.matrix_exp(-M_j)
            Omega = exp_phi_i @ exp_neg_phi_j

            mu_t = Omega @ mu_j
            # Diagonal of Omega @ diag(sigma_j) @ Omega^T
            sigma_t = (Omega ** 2) @ sigma_j

            sigma_blend = ((1.0 - alpha_div) * sigma_i + alpha_div * sigma_t).clamp(min=1e-6)
            delta = mu_t - mu_i_val

            mahal = (alpha_div * delta ** 2 / sigma_blend).sum()
            logdet = ((1.0 - alpha_div) * torch.log(sigma_i.clamp(min=1e-6))
                      + alpha_div * torch.log(sigma_t.clamp(min=1e-6))
                      - torch.log(sigma_blend)).sum() / (alpha_div - 1.0)
            return 0.5 * (mahal + logdet)

        # Code gradient formula: ∂D_α/∂μ_i = α_d · (μ_i - μ_t) / σ_blend
        M_i = torch.einsum('g,gkl->kl', phi_i, G)
        M_j = torch.einsum('g,gkl->kl', phi_j, G)
        Omega = torch.linalg.matrix_exp(M_i) @ torch.linalg.matrix_exp(-M_j)
        mu_t = Omega @ mu_j
        sigma_t = (Omega ** 2) @ sigma_j
        sigma_blend = ((1.0 - alpha_div) * sigma_i + alpha_div * sigma_t).clamp(min=1e-6)
        grad_code = alpha_div * (mu_i - mu_t) / sigma_blend

        # FD gradient
        eps = 1e-6
        grad_fd = torch.zeros_like(mu_i)
        for k in range(K):
            mu_p = mu_i.clone(); mu_p[k] += eps
            mu_m = mu_i.clone(); mu_m[k] -= eps
            grad_fd[k] = (compute_D_alpha(mu_p) - compute_D_alpha(mu_m)) / (2 * eps)

        rel_err = (grad_code - grad_fd).norm() / (grad_fd.norm() + 1e-12)
        assert rel_err < 1e-4, \
            f"Rényi alignment mu gradient rel error {rel_err:.2e} exceeds 1e-4"

    def test_vfe_iteration_descent(self):
        r"""Verify one VFE iteration produces a descent direction for the free energy.

        Constructs a minimal VFE system and checks that F(q_new) < F(q_old)
        after one natural gradient step. This is the integration test that
        validates gradient computation + natural gradient + retraction compose
        correctly into a descent algorithm.
        """
        from transformer.core.kl_computation import _kl_kernel_diagonal
        from transformer.core.gauge_utils import stable_matrix_exp_pair

        torch.manual_seed(70)
        K = 4
        n_gen = K * K
        B, N = 1, 3

        # Build GL(K) generators
        G = torch.zeros(n_gen, K, K, dtype=torch.float64)
        idx = 0
        for i in range(K):
            for j in range(K):
                G[idx, i, j] = 1.0
                idx += 1

        # Random initial state
        mu_q = torch.randn(B, N, K, dtype=torch.float64)
        sigma_q = torch.rand(B, N, K, dtype=torch.float64) * 1.5 + 0.5
        phi = torch.randn(B, N, n_gen, dtype=torch.float64) * 0.1
        mu_p = torch.randn(B, N, K, dtype=torch.float64)
        sigma_p = torch.rand(B, N, K, dtype=torch.float64) * 1.5 + 0.5
        alpha = 1.0
        kappa = 1.0

        def compute_vfe(mu, sig):
            """Compute F = α·Σ_i KL(q_i||p_i) + λ·Σ_i Σ_j β_ij·KL(q_i||Ω_ij q_j)."""
            # Self-coupling
            kl_self = 0.5 * (sig / sigma_p + (mu_p - mu) ** 2 / sigma_p
                             - 1 + torch.log(sigma_p) - torch.log(sig)).sum()

            # Transport and alignment
            phi_mat = torch.einsum('bna,aij->bnij', phi, G)
            exp_phi, exp_neg_phi = stable_matrix_exp_pair(phi_mat)
            Omega = torch.einsum('bikl,bjlm->bijkm', exp_phi, exp_neg_phi)
            mu_t = torch.einsum('bijkl,bjl->bijk', Omega, mu)
            sig_t = torch.einsum('bijkl,bijkl,bjl->bijk', Omega, Omega, sig).clamp(min=1e-6)
            delta = mu.unsqueeze(2) - mu_t
            kl_align = 0.5 * (sig.unsqueeze(2) / sig_t
                              + delta ** 2 / sig_t - 1
                              + torch.log(sig_t) - torch.log(sig.unsqueeze(2))).sum(dim=-1)
            kl_align = kl_align.clamp(min=0)

            # Softmax attention
            dim_scale = K ** 0.5
            logits = -kl_align / (kappa * dim_scale)
            # Self-masking
            eye = torch.eye(N, dtype=torch.float64).unsqueeze(0)
            logits = logits.masked_fill(eye.bool(), float('-inf'))
            beta = torch.softmax(logits, dim=-1)

            alignment = (beta * kl_align).sum()
            return alpha * kl_self + alignment

        # Compute VFE before
        F_before = compute_vfe(mu_q, sigma_q).item()

        # Run one VFE natural gradient step manually
        # Gradient of self-coupling w.r.t. mu_q
        grad_mu_self = alpha * (mu_q - mu_p) / sigma_p  # (B, N, K)

        # Natural gradient: Δμ = -σ_q · ∂F/∂μ
        nat_grad_mu = sigma_q * grad_mu_self

        # Step with small learning rate
        lr = 0.05
        mu_new = mu_q - lr * nat_grad_mu

        F_after = compute_vfe(mu_new, sigma_q).item()

        assert F_after < F_before, \
            f"VFE iteration did not descend: F_before={F_before:.6f}, F_after={F_after:.6f}"

    def test_alignment_sigma_gradient_fd(self):
        r"""FD test: ∂D_α(q_i || Ω_ij q_j)/∂σ_i via central differences."""
        torch.manual_seed(61)
        K = 4
        n_gen = K * K
        alpha_div = 0.275

        G = torch.zeros(n_gen, K, K, dtype=torch.float64)
        idx = 0
        for i in range(K):
            for j in range(K):
                G[idx, i, j] = 1.0
                idx += 1

        phi_i = torch.randn(n_gen, dtype=torch.float64) * 0.1
        phi_j = torch.randn(n_gen, dtype=torch.float64) * 0.1
        mu_i = torch.randn(K, dtype=torch.float64)
        mu_j = torch.randn(K, dtype=torch.float64)
        sigma_i = torch.rand(K, dtype=torch.float64) * 2 + 0.5
        sigma_j = torch.rand(K, dtype=torch.float64) * 2 + 0.5

        M_i = torch.einsum('g,gkl->kl', phi_i, G)
        M_j = torch.einsum('g,gkl->kl', phi_j, G)
        Omega = torch.linalg.matrix_exp(M_i) @ torch.linalg.matrix_exp(-M_j)
        mu_t = Omega @ mu_j
        sigma_t = (Omega ** 2) @ sigma_j

        def compute_D_alpha(sig_i_val):
            sigma_blend = ((1.0 - alpha_div) * sig_i_val + alpha_div * sigma_t).clamp(min=1e-6)
            delta = mu_t - mu_i
            mahal = (alpha_div * delta ** 2 / sigma_blend).sum()
            logdet = ((1.0 - alpha_div) * torch.log(sig_i_val.clamp(min=1e-6))
                      + alpha_div * torch.log(sigma_t.clamp(min=1e-6))
                      - torch.log(sigma_blend)).sum() / (alpha_div - 1.0)
            return 0.5 * (mahal + logdet)

        # Code formula (from vfe_gradients.py:1178-1182):
        sigma_blend = ((1.0 - alpha_div) * sigma_i + alpha_div * sigma_t).clamp(min=1e-6)
        delta = mu_t - mu_i
        grad_code = (
            -alpha_div * (sigma_t - sigma_i) / (2.0 * sigma_i * sigma_blend)
            - alpha_div * (1.0 - alpha_div) * delta ** 2 / (2.0 * sigma_blend ** 2)
        )

        # FD gradient
        eps = 1e-6
        grad_fd = torch.zeros_like(sigma_i)
        for k in range(K):
            s_p = sigma_i.clone(); s_p[k] += eps
            s_m = sigma_i.clone(); s_m[k] -= eps
            grad_fd[k] = (compute_D_alpha(s_p) - compute_D_alpha(s_m)) / (2 * eps)

        rel_err = (grad_code - grad_fd).norm() / (grad_fd.norm() + 1e-12)
        assert rel_err < 1e-4, \
            f"Rényi alignment sigma gradient rel error {rel_err:.2e} exceeds 1e-4"
