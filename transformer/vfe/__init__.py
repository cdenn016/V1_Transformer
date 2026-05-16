"""
Clean gauge-theoretic VFE transformer.

Imports stateless math from transformer/core/, owns model logic.
No BlockConfig, no EM modes, no DEQ/closed-form/hebbian.
Single E-step path: iterative natural gradient with semi-gradient flow at the EM boundary.
"""

from transformer.vfe.config import VFEConfig
from transformer.core.types import BeliefState
from transformer.vfe.prior_bank import VFEPriorBank
from transformer.vfe.positional import VFEPositionalEncoding
from transformer.vfe.e_step import VFEEStep
from transformer.vfe.block import VFEBlock
from transformer.vfe.stack import VFEStack
from transformer.vfe.model import VFEModel
from transformer.vfe.active_inference import VFEActiveInference
from transformer.vfe.efe import VFEExpectedFreeEnergy
from transformer.vfe.trainer import VFETrainer

__all__ = [
    'VFEConfig',
    'BeliefState',
    'VFEPriorBank',
    'VFEPositionalEncoding',
    'VFEEStep',
    'VFEBlock',
    'VFEStack',
    'VFEModel',
    'VFEActiveInference',
    'VFEExpectedFreeEnergy',
    'VFETrainer',
]
