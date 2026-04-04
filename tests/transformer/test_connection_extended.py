"""
Extended tests for transformer/core/connection.py
===================================================

Existing tests (in test_non_flat_transport.py) cover shapes and zero-init.
These tests cover mathematical properties: antisymmetry, gradient flow,
and holonomy-connection correlation.
"""

import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_connection(d_head=6, n_gen=3, connection_type='bilinear', antisymmetrize=True):
    """Create a GaugeConnection for testing."""
    from transformer.core.connection import GaugeConnection
    return GaugeConnection(
        d_head=d_head,
        n_gen=n_gen,
        connection_type=connection_type,
        antisymmetrize=antisymmetrize,
        init_scale=0.0,
    )


# ===========================================================================
# TestAntisymmetry
# ===========================================================================

class TestAntisymmetry:
    """Tests for δ_ij + δ_ji = 0 property."""

    def test_delta_antisymmetric(self):
        """δ_ij + δ_ji = 0 exactly when antisymmetrize=True."""
        conn = _make_connection(d_head=6, n_gen=3, connection_type='bilinear',
                                antisymmetrize=True)
        with torch.no_grad():
            for p in conn.parameters():
                p.normal_(0, 0.1)

        mu = torch.randn(1, 4, 6)
        delta = conn(mu, mu)  # forward(mu_i, mu_j)
        delta_ij = delta
        delta_ji = delta.transpose(1, 2)
        residual = delta_ij + delta_ji
        assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-6), \
            f"Antisymmetry violated: max residual {residual.abs().max():.2e}"

    def test_no_antisymmetry_when_disabled(self):
        """Without antisymmetrize, δ_ij + δ_ji ≠ 0 in general."""
        conn = _make_connection(d_head=6, n_gen=3, connection_type='bilinear',
                                antisymmetrize=False)
        with torch.no_grad():
            for p in conn.parameters():
                p.normal_(0, 0.5)

        mu = torch.randn(1, 4, 6)
        delta = conn(mu, mu)
        delta_ij = delta
        delta_ji = delta.transpose(1, 2)
        residual = delta_ij + delta_ji
        assert residual.abs().max() > 1e-4, \
            "Without antisymmetrize, residual should be nonzero"


# ===========================================================================
# TestGradientFlow
# ===========================================================================

class TestGradientFlow:
    """Tests for gradient signal through connection."""

    def test_bilinear_gradient_exists(self):
        """Bilinear connection produces gradients."""
        conn = _make_connection(d_head=6, n_gen=3, connection_type='bilinear')
        with torch.no_grad():
            for p in conn.parameters():
                p.normal_(0, 0.1)

        mu = torch.randn(1, 4, 6, requires_grad=True)
        delta = conn(mu, mu)
        loss = delta.sum()
        loss.backward()
        assert mu.grad is not None
        for p in conn.parameters():
            if p.requires_grad:
                assert p.grad is not None

    def test_output_shape(self):
        """Connection output is (B, N, N, n_gen)."""
        conn = _make_connection(d_head=6, n_gen=3)
        mu = torch.randn(2, 8, 6)
        delta = conn(mu, mu)
        assert delta.shape == (2, 8, 8, 3)


# ===========================================================================
# TestConnectionProperties
# ===========================================================================

class TestConnectionProperties:
    """Tests for connection->holonomy relationship."""

    def test_zero_init_gives_zero_output(self):
        """init_scale=0 -> δ = 0 at initialization."""
        conn = _make_connection(d_head=6, n_gen=3, connection_type='bilinear')
        mu = torch.randn(1, 4, 6)
        with torch.no_grad():
            delta = conn(mu, mu)
        assert torch.allclose(delta, torch.zeros_like(delta), atol=1e-6), \
            f"Zero-init should give zero output, got max {delta.abs().max():.2e}"

    def test_larger_weights_larger_delta(self):
        """Larger connection weights -> larger ||δ||."""
        conn_small = _make_connection(d_head=6, n_gen=3, connection_type='bilinear')
        conn_large = _make_connection(d_head=6, n_gen=3, connection_type='bilinear')

        with torch.no_grad():
            for p in conn_small.parameters():
                p.normal_(0, 0.01)
            for p in conn_large.parameters():
                p.normal_(0, 1.0)

        mu = torch.randn(1, 4, 6)
        with torch.no_grad():
            delta_small = conn_small(mu, mu)
            delta_large = conn_large(mu, mu)

        norm_small = delta_small.norm()
        norm_large = delta_large.norm()
        assert norm_large > norm_small, \
            f"Larger weights should give larger delta: {norm_large:.4f} vs {norm_small:.4f}"

    def test_output_finite(self):
        """Connection output is always finite."""
        conn = _make_connection(d_head=6, n_gen=3, connection_type='bilinear')
        with torch.no_grad():
            for p in conn.parameters():
                p.normal_(0, 1.0)
        mu = torch.randn(2, 8, 6)
        delta = conn(mu, mu)
        assert torch.isfinite(delta).all()
