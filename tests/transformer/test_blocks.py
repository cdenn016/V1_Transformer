# -*- coding: utf-8 -*-
"""
Blocks Tests
============

Tests for transformer.core.blocks module.

GaugeTransformerBlock is a single layer (attention + VFE FFN) operating
on belief states (mu, sigma, phi). GaugeTransformerStack chains multiple
blocks. Both accept generators of shape (n_gen, K, K) that define the
gauge group; tests here use random skew-symmetric generators.
"""

import pytest
import torch
from transformer.core.block_config import BlockConfig


def _make_generators(K=16):
    """Create random skew-symmetric generators of shape (3, K, K)."""
    G = torch.randn(3, K, K)
    return G - G.transpose(-1, -2)


def _make_block_config(K=16, hidden_dim=32, n_layers=1, **overrides):
    """Create a BlockConfig for testing."""
    generators = _make_generators(K)
    cfg = BlockConfig(
        embed_dim=K,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        kappa_beta=1.0,
        irrep_spec=[('l0', 8, 1), ('l0b', 8, 1)],
        diagonal_covariance=True,
        generators=generators,
        ffn_mode='VFE_dynamic',
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


class TestGaugeTransformerBlock:
    """Test single GaugeTransformerBlock (attention + VFE FFN layer)."""

    @pytest.fixture
    def block(self, cpu_device):
        """Create a transformer block."""
        from transformer.core.blocks import GaugeTransformerBlock
        cfg = _make_block_config()
        return GaugeTransformerBlock(cfg).to(cpu_device)

    def test_creation(self, cpu_device):
        """Test creating transformer block."""
        from transformer.core.blocks import GaugeTransformerBlock
        cfg = _make_block_config()
        block = GaugeTransformerBlock(cfg)
        assert block is not None

    def test_forward(self, block, cpu_device):
        """Test forward pass."""
        B, N, K = 2, 8, 16
        mu = torch.randn(B, N, K, device=cpu_device)
        sigma = torch.abs(torch.randn(B, N, K, device=cpu_device)) + 0.1
        phi = torch.randn(B, N, 3, device=cpu_device) * 0.1
        mu_prior = mu.clone()
        mask = torch.tril(torch.ones(B, N, N, device=cpu_device))

        mu_out, sigma_out, phi_out = block(
            mu, sigma, phi, block.generators, mask=mask, mu_prior=mu_prior
        )
        assert mu_out.shape == (B, N, K)

    def test_forward_output_finite(self, block, cpu_device):
        """Test forward outputs are finite."""
        B, N, K = 2, 8, 16
        mu = torch.randn(B, N, K, device=cpu_device)
        sigma = torch.abs(torch.randn(B, N, K, device=cpu_device)) + 0.1
        phi = torch.randn(B, N, 3, device=cpu_device) * 0.1
        mu_prior = mu.clone()
        mask = torch.tril(torch.ones(B, N, N, device=cpu_device))

        mu_out, sigma_out, phi_out = block(
            mu, sigma, phi, block.generators, mask=mask, mu_prior=mu_prior
        )
        assert torch.isfinite(mu_out).all()
        assert torch.isfinite(sigma_out).all()


class TestGaugeTransformerStack:
    """Test GaugeTransformerStack: sequential composition of multiple blocks."""

    @pytest.fixture
    def stack(self, cpu_device):
        """Create a transformer stack."""
        from transformer.core.blocks import GaugeTransformerStack
        cfg = _make_block_config(n_layers=2)
        return GaugeTransformerStack(cfg).to(cpu_device)

    def test_creation(self, cpu_device):
        """Test creating transformer stack."""
        from transformer.core.blocks import GaugeTransformerStack
        cfg = _make_block_config(n_layers=2)
        stack = GaugeTransformerStack(cfg)
        assert stack is not None
        assert len(stack.blocks) == 2

    def test_forward(self, stack, cpu_device):
        """Test forward pass through stack."""
        B, N, K = 2, 8, 16
        mu = torch.randn(B, N, K, device=cpu_device)
        sigma = torch.abs(torch.randn(B, N, K, device=cpu_device)) + 0.1
        phi = torch.randn(B, N, 3, device=cpu_device) * 0.1
        mu_prior = mu.clone()
        mask = torch.tril(torch.ones(B, N, N, device=cpu_device))

        generators = stack.blocks[0].generators
        mu_out, sigma_out, phi_out, intermediates = stack(
            mu, sigma, phi, generators, mask=mask, mu_prior=mu_prior
        )
        assert mu_out.shape == (B, N, K)

    def test_forward_output_finite(self, stack, cpu_device):
        """Test forward outputs are finite."""
        B, N, K = 2, 8, 16
        mu = torch.randn(B, N, K, device=cpu_device)
        sigma = torch.abs(torch.randn(B, N, K, device=cpu_device)) + 0.1
        phi = torch.randn(B, N, 3, device=cpu_device) * 0.1
        mu_prior = mu.clone()
        mask = torch.tril(torch.ones(B, N, N, device=cpu_device))

        generators = stack.blocks[0].generators
        mu_out, sigma_out, phi_out, intermediates = stack(
            mu, sigma, phi, generators, mask=mask, mu_prior=mu_prior
        )
        assert torch.isfinite(mu_out).all()
        assert torch.isfinite(sigma_out).all()

    def test_multiple_layers(self, cpu_device):
        """Test stack with various layer counts."""
        from transformer.core.blocks import GaugeTransformerStack

        K = 16
        for n_layers in [1, 2, 4]:
            cfg = _make_block_config(K=K, n_layers=n_layers)
            stack = GaugeTransformerStack(cfg).to(cpu_device)
            assert len(stack.blocks) == n_layers

            B, N = 2, 8
            mu = torch.randn(B, N, K, device=cpu_device)
            sigma = torch.abs(torch.randn(B, N, K, device=cpu_device)) + 0.1
            phi = torch.randn(B, N, 3, device=cpu_device) * 0.1
            mask = torch.tril(torch.ones(B, N, N, device=cpu_device))

            mu_out, _, _, intermediates = stack(
                mu, sigma, phi, cfg.generators, mask=mask, mu_prior=mu.clone()
            )
            assert torch.isfinite(mu_out).all()
