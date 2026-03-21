# -*- coding: utf-8 -*-
"""
Gauge-Theoretic Transformer Package
====================================

Implements gauge-theoretic transformers with KL-divergence based attention
and variational free energy dynamics.

Package Structure:
    transformer/
    ├── core/           # Core model components
    ├── training/       # Training infrastructure
    ├── data/           # Data loading
    ├── analysis/       # Analysis and metrics
    ├── visualization/  # Plotting and visualization
    ├── utils/          # Utilities
    └── baselines/      # Baseline models
"""

# Suppress noisy Triton warnings
import warnings
warnings.filterwarnings("ignore", message="Failed to find cuobjdump", category=UserWarning, module="triton")
warnings.filterwarnings("ignore", message="Failed to find nvdisasm", category=UserWarning, module="triton")

# =============================================================================
# Core Model, Training, Data — wrapped in try/except so that lightweight
# subpackages (e.g. pure_vfe) can be imported without full deps like scipy.
# =============================================================================
try:
    from transformer.core.model import GaugeTransformerLM

    # NOTE: PublicationTrainer is imported lazily via __getattr__ to avoid a
    # circular import (train_publication.py imports from transformer.core which
    # triggers this __init__.py).
    from transformer.training.config import TrainingConfig
    from transformer.training import (
        create_optimizer,
        create_param_groups,
        MetricsTracker,
    )
    from transformer.training.config import (
        get_standard_config,
        get_vfe_dynamic_config,
    )

    from transformer.data import (
        create_dataloaders,
        create_char_dataloaders,
    )

    def __getattr__(name):
        if name == "Trainer":
            from transformer.train_publication import PublicationTrainer
            return PublicationTrainer
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    __all__ = [
        # Core model
        'GaugeTransformerLM',

        # Training
        'Trainer',
        'TrainingConfig',
        'create_optimizer',
        'create_param_groups',
        'MetricsTracker',
        'get_standard_config',
        'get_vfe_dynamic_config',

        # Data loading
        'create_dataloaders',
        'create_char_dataloaders',
    ]
except ImportError:
    # Allow pure_vfe subpackage to be imported without full deps (scipy, etc.)
    __all__ = []
