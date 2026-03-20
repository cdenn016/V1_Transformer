"""
Pure Variational Free Energy Transformer.

No nn.Module. No autograd. No backprop.
Inference and learning via natural gradient descent on the gauge-covariant VFE.
"""

from .config import PureVFEConfig
from .model import PureVFETransformer
