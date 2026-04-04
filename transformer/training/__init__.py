"""
Training Infrastructure for Gauge-Theoretic Transformer
=======================================================

Consolidates training configuration, optimizer creation, and metrics
tracking for the VFE (Variational Free Energy) training loop.

Components:
    - TrainingConfig: Unified configuration dataclass (all training modes)
    - create_optimizer: Parameter group-aware AdamW with natural gradient LRs
    - create_param_groups: Per-parameter-type grouping (mu, sigma, phi, attn, ffn)
    - MetricsTracker: CSV logging for training metrics

The main training loop lives in train_publication.py (PublicationTrainer).
Loss computation (CE + VFE regularizers) is in transformer.train.compute_free_energy_loss.
"""

from transformer.training.config import TrainingConfig
from transformer.training.optimizer import (
    create_optimizer,
    create_param_groups,
)
from transformer.training.metrics import MetricsTracker


__all__ = [
    'TrainingConfig',
    'create_optimizer',
    'create_param_groups',
    'MetricsTracker',
]
