# -*- coding: utf-8 -*-
"""
VariationalFFNDynamic Tests
===========================

Tests for transformer.core.variational_ffn.VariationalFFNDynamic.

VariationalFFNDynamic implements the VFE (Variational Free Energy) E-step:
iterative updates to belief states (mu, sigma, phi) that minimize free
energy against a prior. Generators of shape (n_gen, K, K) define the
gauge group; sigma can be diagonal or full.
"""

import pytest
import torch


class TestVariationalFFNDynamic:
    """Test VFE E-step iterations in VariationalFFNDynamic.

    Each iteration refines (mu, sigma, phi) toward the VFE optimum.
    Tests cover creation, forward pass with diagonal covariance,
    output finiteness, and multiple VFE iteration counts.
    """

    @pytest.fixture
    def generators(self):
        """Create random skew-symmetric generators of shape (3, K, K) with K=11."""
        K = 11
        generators = torch.randn(3, K, K)
        generators = generators - generators.transpose(-1, -2)
        return generators

    @pytest.fixture
    def ffn(self, generators, cpu_device):
        """Create a VariationalFFNDynamic instance."""
        from transformer.core.variational_ffn import VariationalFFNDynamic

        K = generators.shape[1]
        ffn = VariationalFFNDynamic(
            embed_dim=K,
            generators=generators,
            alpha=0.001,
            kappa=1.0,
            n_iterations=1,
            diagonal_covariance=True,
        )
        return ffn.to(cpu_device)

    def test_creation(self, generators, cpu_device):
        """Test creating VariationalFFNDynamic."""
        from transformer.core.variational_ffn import VariationalFFNDynamic

        K = generators.shape[1]
        ffn = VariationalFFNDynamic(
            embed_dim=K,
            generators=generators,
        )
        assert ffn is not None

    def test_forward_diagonal(self, ffn, cpu_device):
        """Test forward pass with diagonal covariance."""
        B, N, K = 2, 8, 11

        mu = torch.randn(B, N, K)
        mu_prior = torch.randn(B, N, K)
        phi = torch.randn(B, N, 3) * 0.1
        sigma = torch.abs(torch.randn(B, N, K)) + 0.1
        mask = torch.tril(torch.ones(N, N)).unsqueeze(0).expand(B, -1, -1)
        # VariationalFFNDynamic needs beta
        beta = torch.softmax(torch.randn(B, N, N), dim=-1)

        mu_out, sigma_out, phi_out, beta_history = ffn(
            mu=mu, beta=beta, mu_prior=mu_prior, phi=phi, sigma=sigma, mask=mask,
        )

        assert mu_out.shape == (B, N, K)
        assert sigma_out.shape == (B, N, K)
        assert phi_out.shape == (B, N, 3)
        assert torch.isfinite(mu_out).all()
        assert torch.isfinite(sigma_out).all()

    def test_forward_with_beta_history(self, ffn, cpu_device):
        """Test forward pass with beta history returned."""
        B, N, K = 2, 8, 11

        mu = torch.randn(B, N, K)
        mu_prior = torch.randn(B, N, K)
        phi = torch.randn(B, N, 3) * 0.1
        sigma = torch.abs(torch.randn(B, N, K)) + 0.1
        mask = torch.tril(torch.ones(N, N)).unsqueeze(0).expand(B, -1, -1)
        beta = torch.softmax(torch.randn(B, N, N), dim=-1)

        mu_out, sigma_out, phi_out, beta_history = ffn(
            mu=mu, beta=beta, mu_prior=mu_prior, phi=phi, sigma=sigma, mask=mask,
            return_beta_history=True,
        )

        assert mu_out.shape == (B, N, K)
        assert beta_history is not None

    def test_output_finite(self, ffn, cpu_device):
        """Test that outputs are always finite (no NaN/Inf)."""
        B, N, K = 1, 4, 11

        mu = torch.randn(B, N, K) * 0.1
        mu_prior = torch.randn(B, N, K) * 0.1
        phi = torch.zeros(B, N, 3)
        sigma = torch.ones(B, N, K)
        mask = torch.tril(torch.ones(N, N)).unsqueeze(0).expand(B, -1, -1)
        beta = torch.softmax(torch.randn(B, N, N), dim=-1)

        mu_out, sigma_out, phi_out, _bh = ffn(
            mu=mu, beta=beta, mu_prior=mu_prior, phi=phi, sigma=sigma, mask=mask,
        )

        assert torch.isfinite(mu_out).all(), "mu_out contains NaN/Inf"
        assert torch.isfinite(sigma_out).all(), "sigma_out contains NaN/Inf"
        assert torch.isfinite(phi_out).all(), "phi_out contains NaN/Inf"
        assert (sigma_out > 0).all(), "sigma_out must be positive"

    def test_multiple_vfe_iterations(self, generators, cpu_device):
        """Test with multiple VFE iterations."""
        from transformer.core.variational_ffn import VariationalFFNDynamic

        K = generators.shape[1]
        ffn = VariationalFFNDynamic(
            embed_dim=K,
            generators=generators,
            n_iterations=3,
            diagonal_covariance=True,
        ).to(cpu_device)

        B, N = 1, 4
        mu = torch.randn(B, N, K) * 0.1
        mu_prior = torch.randn(B, N, K) * 0.1
        phi = torch.zeros(B, N, 3)
        sigma = torch.ones(B, N, K)
        mask = torch.tril(torch.ones(N, N)).unsqueeze(0).expand(B, -1, -1)
        beta = torch.softmax(torch.randn(B, N, N), dim=-1)

        mu_out, sigma_out, phi_out, _bh = ffn(
            mu=mu, beta=beta, mu_prior=mu_prior, phi=phi, sigma=sigma, mask=mask,
        )

        assert torch.isfinite(mu_out).all()
        assert (sigma_out > 0).all()

    def test_learnable_alpha_creation(self, generators, cpu_device):
        """Test creating VariationalFFNDynamic with learnable_alpha=True."""
        from transformer.core.variational_ffn import VariationalFFNDynamic

        K = generators.shape[1]
        ffn = VariationalFFNDynamic(
            embed_dim=K,
            generators=generators,
            alpha=1.0,
            learnable_alpha=True,
            diagonal_covariance=True,
        ).to(cpu_device)

        assert hasattr(ffn, 'raw_c0')
        assert hasattr(ffn, 'raw_b0')
        assert ffn.raw_c0.shape == (K,)
        assert ffn.raw_b0.shape == (K,)
        assert ffn.raw_c0.requires_grad
        assert ffn.raw_b0.requires_grad

    def test_learnable_alpha_forward(self, generators, cpu_device):
        """Test forward pass with learnable_alpha produces finite outputs and gradients."""
        from transformer.core.variational_ffn import VariationalFFNDynamic

        K = generators.shape[1]
        ffn = VariationalFFNDynamic(
            embed_dim=K,
            generators=generators,
            alpha=1.0,
            learnable_alpha=True,
            diagonal_covariance=True,
        ).to(cpu_device)

        B, N = 2, 4
        mu = torch.randn(B, N, K) * 0.1
        mu_prior = torch.randn(B, N, K) * 0.1
        phi = torch.zeros(B, N, 3)
        sigma = torch.ones(B, N, K)
        mask = torch.tril(torch.ones(N, N)).unsqueeze(0).expand(B, -1, -1)
        beta = torch.softmax(torch.randn(B, N, N), dim=-1)

        mu_out, sigma_out, phi_out, _bh = ffn(
            mu=mu, beta=beta, mu_prior=mu_prior, phi=phi, sigma=sigma, mask=mask,
        )

        assert torch.isfinite(mu_out).all(), "mu_out has NaN/Inf with learnable alpha"
        assert torch.isfinite(sigma_out).all(), "sigma_out has NaN/Inf with learnable alpha"
        assert (sigma_out > 0).all(), "sigma_out must be positive with learnable alpha"

        # Verify gradients flow to raw_c0 and raw_b0
        loss = mu_out.sum()
        loss.backward()
        assert ffn.raw_c0.grad is not None, "raw_c0 received no gradient"
        assert ffn.raw_b0.grad is not None, "raw_b0 received no gradient"
        assert torch.isfinite(ffn.raw_c0.grad).all(), "raw_c0 gradient has NaN/Inf"
        assert torch.isfinite(ffn.raw_b0.grad).all(), "raw_b0 gradient has NaN/Inf"

    def test_learnable_alpha_bayesian_alpha_shape(self, generators, cpu_device):
        """Test get_bayesian_alpha returns correct shape and values."""
        from transformer.core.variational_ffn import VariationalFFNDynamic

        K = generators.shape[1]
        ffn = VariationalFFNDynamic(
            embed_dim=K,
            generators=generators,
            alpha=1.0,
            learnable_alpha=True,
            diagonal_covariance=True,
        ).to(cpu_device)

        B, N = 2, 4
        mu_q = torch.randn(B, N, K) * 0.1
        mu_p = torch.randn(B, N, K) * 0.1
        sigma_q = torch.ones(B, N, K)
        sigma_p = torch.ones(B, N, K)

        alpha = ffn.get_bayesian_alpha(mu_q, mu_p, sigma_p, sigma_q)

        assert alpha.shape == (B, N, K), f"Expected (B,N,K) shape, got {alpha.shape}"
        assert (alpha > 0).all(), "Bayesian alpha must be positive"
        assert torch.isfinite(alpha).all(), "Bayesian alpha has NaN/Inf"

    def test_gradient_checkpoint_vfe(self, generators, cpu_device):
        """Gradient checkpointing produces same output as non-checkpointed."""
        from transformer.core.variational_ffn import VariationalFFNDynamic

        K = generators.shape[1]
        B, N = 2, 4

        # Create two identical FFNs — one with checkpointing, one without
        torch.manual_seed(0)
        ffn_base = VariationalFFNDynamic(
            embed_dim=K, generators=generators, alpha=0.001, kappa=1.0,
            n_iterations=3, diagonal_covariance=True,
            gradient_checkpoint_vfe=False,
        ).to(cpu_device)

        torch.manual_seed(0)
        ffn_ckpt = VariationalFFNDynamic(
            embed_dim=K, generators=generators, alpha=0.001, kappa=1.0,
            n_iterations=3, diagonal_covariance=True,
            gradient_checkpoint_vfe=True,
        ).to(cpu_device)

        # Copy weights
        ffn_ckpt.load_state_dict(ffn_base.state_dict())

        torch.manual_seed(42)
        mu = torch.randn(B, N, K)
        mu_prior = torch.randn(B, N, K)
        phi = torch.randn(B, N, 3) * 0.1
        sigma = torch.abs(torch.randn(B, N, K)) + 0.1
        mask = torch.tril(torch.ones(N, N)).unsqueeze(0).expand(B, -1, -1)
        beta = torch.softmax(torch.randn(B, N, N), dim=-1)

        ffn_base.train()
        ffn_ckpt.train()

        mu_out_base, sigma_out_base, _, _ = ffn_base(
            mu=mu.clone(), beta=beta.clone(), mu_prior=mu_prior.clone(),
            phi=phi.clone(), sigma=sigma.clone(), mask=mask.clone(),
        )
        mu_out_ckpt, sigma_out_ckpt, _, _ = ffn_ckpt(
            mu=mu.clone(), beta=beta.clone(), mu_prior=mu_prior.clone(),
            phi=phi.clone(), sigma=sigma.clone(), mask=mask.clone(),
        )

        assert torch.allclose(mu_out_base, mu_out_ckpt, atol=1e-5), \
            f"Checkpointed mu differs: max diff {(mu_out_base - mu_out_ckpt).abs().max():.2e}"
        assert torch.allclose(sigma_out_base, sigma_out_ckpt, atol=1e-5), \
            f"Checkpointed sigma differs: max diff {(sigma_out_base - sigma_out_ckpt).abs().max():.2e}"
