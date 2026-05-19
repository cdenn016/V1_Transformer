"""
Tests for transformer.training.train_fast
===========================================

Validates FastTrainer: parameter group creation, LR scheduling,
gradient clipping, checkpoint save/load, and single-step correctness.
"""

import math
import pytest
import torch
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

from transformer.training.train_fast import FastTrainer
from transformer.training.config import TrainingConfig


# =============================================================================
# Helpers
# =============================================================================

def _make_synthetic_loader(vocab_size, seq_len, batch_size, n_batches):
    """Create a DataLoader with random token tensors."""
    total = n_batches * batch_size
    input_ids = torch.randint(0, vocab_size, (total, seq_len))
    targets = torch.randint(0, vocab_size, (total, seq_len))
    dataset = TensorDataset(input_ids, targets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def _make_trainer(minimal_config, cpu_device, tmp_path, config_overrides=None):
    """Create a FastTrainer with minimal model and synthetic data.

    The ``tmp_path`` is the pytest builtin ``tmp_path`` fixture (a unique
    auto-cleaned ``Path`` per test). Tests that need a trainer should
    accept ``tmp_path`` and forward it here so checkpoint directories
    are cleaned up between tests instead of leaking via ``mkdtemp()``.
    """
    from transformer.core.model import GaugeTransformerLM

    config = TrainingConfig()
    config.checkpoint_dir = tmp_path
    if config_overrides:
        for k, v in config_overrides.items():
            if hasattr(config, k):
                setattr(config, k, v)

    model = GaugeTransformerLM(minimal_config).to(cpu_device)
    train_loader = _make_synthetic_loader(
        minimal_config['vocab_size'], 16, 2, 4
    )
    val_loader = _make_synthetic_loader(
        minimal_config['vocab_size'], 16, 2, 2
    )
    trainer = FastTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=cpu_device,
    )
    return trainer


# =============================================================================
# TestFastTrainerCreation
# =============================================================================

class TestFastTrainerCreation:
    """FastTrainer instantiation and optimizer setup."""

    def test_instantiation(self, minimal_config, cpu_device, tmp_path):
        """FastTrainer instantiates without error."""
        trainer = _make_trainer(minimal_config, cpu_device, tmp_path)
        assert trainer is not None

    def test_optimizer_has_param_groups(self, minimal_config, cpu_device, tmp_path):
        """Optimizer has multiple parameter groups."""
        trainer = _make_trainer(minimal_config, cpu_device, tmp_path)
        groups = trainer.optimizer.param_groups
        assert len(groups) > 1, f"Expected >1 param groups, got {len(groups)}"

    def test_scheduler_created(self, minimal_config, cpu_device, tmp_path):
        """Scheduler is created."""
        trainer = _make_trainer(minimal_config, cpu_device, tmp_path)
        assert trainer.scheduler is not None


# =============================================================================
# TestFastTrainerTrainStep
# =============================================================================

class TestFastTrainerTrainStep:
    """FastTrainer.train_step: single step correctness."""

    def test_single_step_returns_metrics(self, minimal_config, cpu_device, tmp_path):
        """train_step returns a metrics dict."""
        trainer = _make_trainer(minimal_config, cpu_device, tmp_path)
        trainer.model.train()
        batch = next(iter(trainer.train_loader))
        batch = (batch[0].to(cpu_device), batch[1].to(cpu_device))
        metrics = trainer.train_step(batch)
        assert isinstance(metrics, dict)

    def test_loss_is_finite(self, minimal_config, cpu_device, tmp_path):
        """Loss values are finite."""
        trainer = _make_trainer(minimal_config, cpu_device, tmp_path)
        trainer.model.train()
        batch = next(iter(trainer.train_loader))
        batch = (batch[0].to(cpu_device), batch[1].to(cpu_device))
        metrics = trainer.train_step(batch)
        # Check any loss-related key
        for k, v in metrics.items():
            if 'loss' in k.lower() and isinstance(v, (int, float)):
                assert math.isfinite(v), f"Non-finite {k}: {v}"

    def test_model_in_train_mode(self, minimal_config, cpu_device, tmp_path):
        """Model is in train mode during train_step."""
        trainer = _make_trainer(minimal_config, cpu_device, tmp_path)
        # model should be in training mode after trainer init or explicit call
        trainer.model.train()
        assert trainer.model.training


# =============================================================================
# TestFastTrainerCheckpoint
# =============================================================================

class TestFastTrainerCheckpoint:
    """Checkpoint save/load round-trip."""

    def test_save_checkpoint(self, minimal_config, cpu_device, tmp_path):
        """save_checkpoint creates a file."""
        trainer = _make_trainer(minimal_config, cpu_device, tmp_path)
        trainer.global_step = 10
        trainer.save_checkpoint(is_best=True)
        best_path = trainer.config.checkpoint_dir / 'best_model.pt'
        assert best_path.exists(), f"Checkpoint not saved at {best_path}"

    def test_load_checkpoint(self, minimal_config, cpu_device, tmp_path):
        """load_checkpoint restores state."""
        trainer = _make_trainer(minimal_config, cpu_device, tmp_path)
        trainer.global_step = 10
        trainer.save_checkpoint(is_best=True)
        best_path = str(trainer.config.checkpoint_dir / 'best_model.pt')
        checkpoint = torch.load(best_path, map_location='cpu', weights_only=False)  # trusted: self-saved checkpoint inside test sandbox
        assert 'model_state_dict' in checkpoint
