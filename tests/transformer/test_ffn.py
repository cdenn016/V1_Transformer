# -*- coding: utf-8 -*-
"""
GaugeFFN Tests
==============

Tests for transformer.core.ffn module (GaugeFFN wrapper around VariationalFFNDynamic).
"""

import pytest
import torch


class TestGaugeFFN:
    """Test GaugeFFN class."""

    @pytest.fixture
    def generators(self):
        """Create SO(3)-like skew-symmetric generators for K=11."""
        K = 11
        generators = torch.randn(3, K, K)
        generators = generators - generators.transpose(-1, -2)
        return generators

    @pytest.fixture
    def ffn(self, generators, cpu_device):
        """Create a GaugeFFN instance."""
        from transformer.core.ffn import GaugeFFN

        K = generators.shape[1]
        ffn = GaugeFFN(
            embed_dim=K,
            hidden_dim=44,
            generators=generators,
            alpha=0.001,
            kappa=1.0,
            n_iterations=1,
            diagonal_covariance=True,
        )
        return ffn.to(cpu_device)

    def test_creation(self, generators, cpu_device):
        """Test creating GaugeFFN."""
        from transformer.core.ffn import GaugeFFN

        K = generators.shape[1]
        ffn = GaugeFFN(
            embed_dim=K,
            hidden_dim=44,
            generators=generators,
        )
        assert ffn is not None
        assert ffn.get_mode() == 'VFE_dynamic'

    def test_requires_generators(self, cpu_device):
        """Test that GaugeFFN raises without generators."""
        from transformer.core.ffn import GaugeFFN

        with pytest.raises(ValueError, match="generators required"):
            GaugeFFN(embed_dim=11, hidden_dim=44, generators=None)

    def test_forward_diagonal(self, ffn, cpu_device):
        """Test forward pass with diagonal covariance."""
        B, N, K = 2, 8, 11

        mu = torch.randn(B, N, K)
        mu_prior = torch.randn(B, N, K)
        phi = torch.randn(B, N, 3) * 0.1
        sigma = torch.abs(torch.randn(B, N, K)) + 0.1
        mask = torch.tril(torch.ones(N, N)).unsqueeze(0).expand(B, -1, -1)

        result = ffn(mu=mu, mu_prior=mu_prior, phi=phi, sigma=sigma, mask=mask)

        assert len(result) == 3
        mu_out, sigma_out, phi_out = result
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

        result = ffn(
            mu=mu, mu_prior=mu_prior, phi=phi, sigma=sigma, mask=mask,
            return_beta_history=True,
        )

        assert len(result) == 4
        mu_out, sigma_out, phi_out, beta_history = result
        assert mu_out.shape == (B, N, K)

    def test_forward_requires_mu_prior_and_phi(self, ffn, cpu_device):
        """Test that forward raises without mu_prior or phi."""
        B, N, K = 2, 8, 11
        mu = torch.randn(B, N, K)

        with pytest.raises(ValueError, match="VFE_dynamic requires"):
            ffn(mu=mu, mu_prior=None, phi=None)

    def test_output_finite(self, ffn, cpu_device):
        """Test that outputs are always finite (no NaN/Inf)."""
        B, N, K = 1, 4, 11

        # Use small values to avoid numerical issues
        mu = torch.randn(B, N, K) * 0.1
        mu_prior = torch.randn(B, N, K) * 0.1
        phi = torch.zeros(B, N, 3)
        sigma = torch.ones(B, N, K)
        mask = torch.tril(torch.ones(N, N)).unsqueeze(0).expand(B, -1, -1)

        mu_out, sigma_out, phi_out = ffn(
            mu=mu, mu_prior=mu_prior, phi=phi, sigma=sigma, mask=mask,
        )

        assert torch.isfinite(mu_out).all(), "mu_out contains NaN/Inf"
        assert torch.isfinite(sigma_out).all(), "sigma_out contains NaN/Inf"
        assert torch.isfinite(phi_out).all(), "phi_out contains NaN/Inf"
        assert (sigma_out > 0).all(), "sigma_out must be positive"

    def test_pure_fep_mode_property(self, generators, cpu_device):
        """Test pure_fep_mode property."""
        from transformer.core.ffn import GaugeFFN

        K = generators.shape[1]
        ffn = GaugeFFN(
            embed_dim=K,
            hidden_dim=44,
            generators=generators,
            pure_fep_mode=False,
        )
        assert ffn.pure_fep_mode is False

    def test_create_ffn_factory(self, generators, cpu_device):
        """Test create_ffn factory function."""
        from transformer.core.ffn import create_ffn

        K = generators.shape[1]
        ffn = create_ffn(
            embed_dim=K,
            hidden_dim=44,
            generators=generators,
            alpha=0.001,
        )
        assert ffn is not None
        assert ffn.get_mode() == 'VFE_dynamic'

    def test_multiple_vfe_iterations(self, generators, cpu_device):
        """Test GaugeFFN with multiple VFE iterations."""
        from transformer.core.ffn import GaugeFFN

        K = generators.shape[1]
        ffn = GaugeFFN(
            embed_dim=K,
            hidden_dim=44,
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

        mu_out, sigma_out, phi_out = ffn(
            mu=mu, mu_prior=mu_prior, phi=phi, sigma=sigma, mask=mask,
        )

        assert torch.isfinite(mu_out).all()
        assert (sigma_out > 0).all()
