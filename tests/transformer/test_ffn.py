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
