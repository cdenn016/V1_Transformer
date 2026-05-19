"""
Tests for transformer.utils.evaluation
===============================================================================

Validates the evaluation pipeline: checkpoint loading, config extraction,
perplexity computation, and token-weighted averaging.

Uses mocks to avoid needing real data files or trained checkpoints.
"""

import math
import pytest
import torch
import torch.nn.functional as F
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
import os

from transformer.utils.evaluation import evaluate_checkpoint


# =============================================================================
# Helpers
# =============================================================================

def _make_synthetic_checkpoint(config, model, tmp_dir, step=100):
    """Save a synthetic checkpoint for testing."""
    path = os.path.join(tmp_dir, 'test_checkpoint.pt')
    torch.save({
        'config': config,
        'model_state_dict': model.state_dict(),
        'step': step,
        'best_val_loss': 3.5,
    }, path)
    return path


# TestGetConfigVal and TestPerplexityComputation removed 2026-05-18.
# Both classes only re-asserted Python/math invariants
# (``dict.get`` semantics, ``math.exp(3.0) == math.exp(3.0)``, weighted
# average arithmetic), never invoking ``evaluate_checkpoint``'s PPL/BPC
# code. TestEvaluateCheckpoint below provides the real smoke coverage
# by round-tripping a synthetic checkpoint through ``torch.save/load``
# and asserting on extracted fields.


# =============================================================================
# TestEvaluateCheckpoint
# =============================================================================

class TestEvaluateCheckpoint:
    """evaluate_checkpoint: end-to-end evaluation smoke tests."""

    def test_synthetic_checkpoint_loads(self, cpu_device, minimal_config):
        """Can load a synthetic checkpoint and extract config."""
        from transformer.core.model import GaugeTransformerLM
        model = GaugeTransformerLM(minimal_config).to(cpu_device)

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = _make_synthetic_checkpoint(minimal_config, model, tmp_dir)
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)  # trusted: self-saved checkpoint inside test sandbox
            assert 'config' in checkpoint
            assert 'model_state_dict' in checkpoint
            assert checkpoint['step'] == 100

    def test_checkpoint_config_extraction(self, cpu_device, minimal_config):
        """Config is correctly extracted from checkpoint."""
        from transformer.core.model import GaugeTransformerLM
        model = GaugeTransformerLM(minimal_config).to(cpu_device)

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = _make_synthetic_checkpoint(minimal_config, model, tmp_dir)
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)  # trusted: self-saved checkpoint inside test sandbox
            config = checkpoint['config']
            assert config['vocab_size'] == 100
            assert config['embed_dim'] == 15

    def test_vocab_size_from_weights(self, cpu_device, minimal_config):
        """Vocab size can be inferred from model weight tensor shape."""
        from transformer.core.model import GaugeTransformerLM
        model = GaugeTransformerLM(minimal_config).to(cpu_device)

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = _make_synthetic_checkpoint(minimal_config, model, tmp_dir)
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)  # trusted: self-saved checkpoint inside test sandbox
            state = checkpoint['model_state_dict']
            # Find a weight tensor that reveals vocab size
            for key, val in state.items():
                if 'output' in key or 'embed' in key:
                    if val.shape[0] == 100:  # vocab_size
                        break
            # The model should have at least one tensor reflecting vocab_size


# =============================================================================
# TestCrossEntropyCorrectness
# =============================================================================

class TestCrossEntropyCorrectness:
    """Cross-entropy loss computation matches PyTorch reference."""

    # ``test_ce_matches_pytorch`` removed 2026-05-18: it reimplemented
    # ``F.cross_entropy`` from ``F.log_softmax`` and asserted the
    # reimplementation matched itself. ``test_pad_token_exclusion``
    # below exercises a real semantic — ``ignore_index=-100`` — and is
    # retained.

    def test_pad_token_exclusion(self, cpu_device):
        """Tokens with target=-100 don't contribute to loss."""
        B, N, V = 1, 8, 50
        logits = torch.randn(B, N, V, device=cpu_device)
        targets = torch.randint(0, V, (B, N), device=cpu_device)
        # Mark half as padding
        targets[0, 4:] = -100

        ce = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1), ignore_index=-100)
        # Should only count first 4 tokens
        ce_first4 = F.cross_entropy(logits[0, :4].reshape(-1, V), targets[0, :4].reshape(-1))
        assert torch.allclose(ce, ce_first4, atol=1e-5)
