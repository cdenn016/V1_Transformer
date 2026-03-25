# -*- coding: utf-8 -*-
"""
Pytest Configuration and Shared Fixtures
=========================================

Provides common fixtures for gauge-theoretic transformer tests.

Fixtures supply model configs, belief-state tensors (mu, sigma, phi),
and pre-built GaugeTransformerLM instances. Gauge group and phi_dim
are determined by the irrep_spec / generators shape (n_gen, K, K);
fixtures here default to small configs suitable for CI.
"""

import pytest
import sys
from pathlib import Path

# Suppress warnings before any imports
import warnings
warnings.filterwarnings("ignore", message="Failed to find cuobjdump", module="triton")
warnings.filterwarnings("ignore", message="Failed to find nvdisasm", module="triton")
warnings.filterwarnings("ignore", message="CUDA path could not be detected", module="cupy")

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch


# =============================================================================
# Device Fixtures
# =============================================================================

@pytest.fixture
def device():
    """Get available device (CPU for CI, GPU if available)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def cpu_device():
    """Force CPU device."""
    return torch.device('cpu')


# =============================================================================
# Model Configuration Fixtures
# =============================================================================

@pytest.fixture
def minimal_config():
    """Minimal model config for fast tests."""
    return {
        'vocab_size': 100,
        'embed_dim': 15,
        'n_layers': 1,
        'irrep_spec': [('l0', 6, 1), ('l1', 3, 3)],
        'hidden_dim': 32,
        'max_seq_len': 32,
        'kappa_beta': 1.0,
        'dropout': 0.0,
        'pos_encoding_mode': 'learned',
        'evolve_sigma': True,
        'evolve_phi': False,
        'tie_embeddings': True,
        'diagonal_covariance': True,
        'ffn_mode': 'VFE_dynamic',
    }


@pytest.fixture
def small_config():
    """Small but realistic config for integration tests."""
    return {
        'vocab_size': 256,
        'embed_dim': 25,
        'n_layers': 2,
        'irrep_spec': [('l0', 5, 1), ('l1', 3, 3), ('l2', 1, 5)],
        'hidden_dim': 64,
        'max_seq_len': 64,
        'kappa_beta': 1.0,
        'dropout': 0.1,
        'pos_encoding_mode': 'learned',
        'evolve_sigma': True,
        'evolve_phi': True,
        'tie_embeddings': True,
        'diagonal_covariance': True,
        'ffn_mode': 'VFE_dynamic',
    }



# =============================================================================
# Tensor Fixtures
# =============================================================================

@pytest.fixture
def batch_tensors(minimal_config):
    """Create batch of input tensors."""
    B, N = 2, 16
    vocab_size = minimal_config['vocab_size']

    input_ids = torch.randint(0, vocab_size, (B, N))
    targets = torch.randint(0, vocab_size, (B, N))

    return {
        'input_ids': input_ids,
        'targets': targets,
        'batch_size': B,
        'seq_len': N,
    }



# =============================================================================
# Model Fixtures
# =============================================================================

@pytest.fixture
def gauge_model(minimal_config, cpu_device):
    """Create a minimal GaugeTransformerLM for testing."""
    from transformer.core.model import GaugeTransformerLM

    model = GaugeTransformerLM(minimal_config)
    model = model.to(cpu_device)
    model.eval()
    return model


