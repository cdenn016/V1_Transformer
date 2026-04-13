"""Shared fixtures and helpers for pure VFE tests."""

import torch
import pytest

# ---------------------------------------------------------------------------
# Constants for gradient tests
# ---------------------------------------------------------------------------
DEVICE = 'cpu'
K = 4       # Small K for fast finite-diff
EPS = 1e-5  # FD perturbation
REL_TOL = 1e-3  # Relative error tolerance for FD vs analytical
B, H, N = 1, 1, 4  # Minimal batch/head/sequence


# ---------------------------------------------------------------------------
# Random tensor factories
# ---------------------------------------------------------------------------

def random_spd(K, batch_shape=()):
    """Generate random SPD matrix."""
    A = torch.randn(*batch_shape, K, K)
    return A @ A.transpose(-2, -1) + 0.1 * torch.eye(K)


def random_gl(K, batch_shape=(), scale=0.3):
    """Generate random GL(K) matrix near identity."""
    return torch.eye(K).expand(*batch_shape, K, K).clone() + scale * torch.randn(*batch_shape, K, K)


# ---------------------------------------------------------------------------
# PureVFEConfig factory
# ---------------------------------------------------------------------------

def make_pure_vfe_config(**overrides):
    """Create a minimal PureVFEConfig for testing."""
    from ..config import PureVFEConfig
    defaults = dict(
        vocab_size=20,
        belief_dim=K,
        n_heads=H,
        head_dim=K,
        n_esteps=3,
        max_seq_len=8,
        device='cpu',
        use_cuda_kernels=False,
        mu_q_lr=0.05,
        sigma_q_lr=0.005,
        phi_lr=0.05,
        mu_p_lr=0.02,
        sigma_p_lr=0.005,
    )
    defaults.update(overrides)
    return PureVFEConfig(**defaults)


def make_model(config=None):
    """Create a PureVFETransformer on CPU for testing."""
    from ..model import PureVFETransformer
    if config is None:
        config = make_pure_vfe_config()
    return PureVFETransformer(config)


def make_precomp(B=1, H=1, N=4, K_h=4, seed=42):
    """Build a synthetic precomp dict for gradient tests."""
    from ..gaussians import precompute_tokens
    torch.manual_seed(seed)
    mu_h = torch.randn(B, N, H, K_h)
    Sigma_h = random_spd(K_h, batch_shape=(B, N, H))
    Omega = random_gl(K_h, batch_shape=(B, N, H))
    precomp = precompute_tokens(mu_h, Sigma_h, Omega)
    return precomp, mu_h, Sigma_h, Omega
