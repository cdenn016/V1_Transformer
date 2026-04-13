"""
Utility Module
==============

Utilities for the gauge-theoretic transformer (VFE-based training with
SO(N)/GL(K) gauge transport and KL-divergence attention).

- Checkpoint: Save/load GaugeTransformerLM checkpoints and configs
- Evaluation: Validate checkpoints on held-out data (validation and test splits)
- Testing: Diagnostic tools for attention uniformity and belief-space analysis
"""

from transformer.utils.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    load_model,
    get_tokenizer,
    load_checkpoint_info,
)

__all__ = [
    'save_checkpoint',
    'load_checkpoint',
    'load_model',
    'get_tokenizer',
    'load_checkpoint_info',
]
