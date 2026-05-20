"""
Canonical Active Inference for the Gauge-Theoretic VFE Transformer.

Implements Expected Free Energy `G(π)` over horizon-D policies (= future
token sequences) per [ParrPezzuloFriston2022 Ch. 7] and
[Friston2021SophisticatedInference]. Generation-time policy selection only;
training-time stays standard variational F (no E-step augmentation).

The module wraps a trained `VFEModel` by composition; it never modifies
model parameters at generation time and preserves all CLAUDE.md hard
constraints (gauge equivariance, no neural networks beyond PriorBank
decode, no CLI args, Law-1 E-step blindness).

Phase 1 default (`horizon_D=1, beam_width=16`) reduces bitwise to the
existing `transformer/vfe/efe.py:VFEExpectedFreeEnergy` depth-1 path.
Phase 2 (beam tree search at D>1) and Phase 3 (Friston-2021 sophisticated
recursion) are exposed via `AIFConfig.horizon_D` and `branching_strategy`.
"""

from transformer.aif.config import AIFConfig
from transformer.aif.preferences import (
    Preference,
    EmpiricalMarginalPreference,
    LowEntropyPreference,
    TaskConditionedPreference,
    build_preference,
)
from transformer.aif.policy import PolicyNode, EFEComponents
from transformer.aif.belief_cache import BeliefStateCache
from transformer.aif.efe_score import compute_G_at_node
from transformer.aif.generator import AIFGenerator

__all__ = [
    'AIFConfig',
    'Preference',
    'EmpiricalMarginalPreference',
    'LowEntropyPreference',
    'TaskConditionedPreference',
    'build_preference',
    'PolicyNode',
    'EFEComponents',
    'BeliefStateCache',
    'compute_G_at_node',
    'AIFGenerator',
]
