"""
Experimental modules - not part of main training pipeline.

Contains:
    - fep_transformer: Free Energy Principle transformer variant
    - train_fep: Training script for FEPTransformer

These implementations are experimental and not recommended for production use.
For standard training, use the main transformer module:
    from transformer import GaugeTransformerLM, Trainer, TrainingConfig
"""

# Lazy imports to avoid import errors if dependencies change
# Use: from transformer.experimental.fep_transformer import FEPTransformer

__all__ = []
