"""
VFEBlock: single VFE layer.

Each block runs the E-step and applies normalization. No separate attention
sublayer — the E-step internally computes and updates attention at each
iteration (dynamic beta).
"""

from typing import Optional, Callable

import torch
import torch.nn as nn

from transformer.core.types import BeliefState
from transformer.core.blocks import MahalanobisNorm, RMSNorm
from transformer.vfe.e_step import VFEEStep


class VFEBlock(nn.Module):
    """Single gauge-VFE transformer block.

    Architecture: E-step → normalization.

    The E-step internally recomputes attention weights at each iteration
    (dynamic beta), so there is no separate attention sublayer.

    Args:
        cfg: VFEConfig.
        generators: ``(n_gen, K, K)`` Lie algebra generators.
    """

    def __init__(self, cfg: 'VFEConfig', generators: torch.Tensor) -> None:
        super().__init__()
        self.e_step = VFEEStep(cfg, generators)

        # Normalization
        if cfg.norm_type == 'mahalnorm':
            self.norm = MahalanobisNorm(cfg.embed_dim)
        elif cfg.norm_type == 'rmsnorm':
            self.norm = RMSNorm(cfg.embed_dim)
        else:
            self.norm = None

    def forward(
        self,
        beliefs: BeliefState,
        priors: BeliefState,
        mask: Optional[torch.Tensor] = None,
        active_inference_fn: Optional[Callable] = None,
    ) -> BeliefState:
        """Forward pass: E-step + normalization.

        Args:
            beliefs: Current beliefs ``(mu, sigma, phi)``.
            priors: Layer priors ``(mu_p, sigma_p, phi_p)``.
            mask: ``(B, N, N)`` causal mask.
            active_inference_fn: Optional callback for active inference gradients.

        Returns:
            Updated BeliefState.
        """
        beliefs = self.e_step(beliefs, priors, mask, active_inference_fn)

        if self.norm is not None:
            mu_normed = self.norm(beliefs.mu, beliefs.sigma)
            beliefs = beliefs._replace(mu=mu_normed)

        return beliefs
