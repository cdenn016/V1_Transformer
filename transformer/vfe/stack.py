"""
VFEStack: L layers with cross-layer prior handoff.

Cross-layer handoff (VFE_Transformer_Idea.md Section 7):
    mu_prior^{l+1}    = (1-rho_mu) mu_prior^l + rho_mu mu*^l
    sigma_prior^{l+1} = (1-rho_sigma) sigma_embed + rho_sigma sigma*^l
    phi_prior^{l+1}   = phi*^l  (when prior_handoff_phi=True)

Defaults: rho_mu=1.0 (full mu flow), rho_sigma=0.0 (frozen at embedding),
prior_handoff_phi=False. Sigma damping prevents progressive variance
collapse (sigma cascade) while allowing gradual prior refinement.

Sigma freezing is architecturally enforced — priors.sigma is never
reassigned from the posterior.
"""

from __future__ import annotations

from typing import Optional, Callable, List, TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from transformer.vfe.config import VFEConfig

from transformer.core.types import BeliefState
from transformer.vfe.block import VFEBlock


class VFEStack(nn.Module):
    """Stack of L VFE blocks with cross-layer prior handoff.

    Args:
        cfg: VFEConfig.
        generators: ``(n_gen, K, K)`` Lie algebra generators.
    """

    def __init__(self, cfg: 'VFEConfig', generators: torch.Tensor) -> None:
        super().__init__()
        self.prior_handoff_rho = cfg.prior_handoff_rho
        self.prior_handoff_sigma = cfg.prior_handoff_sigma
        self.prior_handoff_phi = cfg.prior_handoff_phi
        self.sigma_max = cfg.sigma_max
        self.blocks = nn.ModuleList([
            VFEBlock(cfg, generators) for _ in range(cfg.n_layers)
        ])

    def forward(
        self,
        beliefs: BeliefState,
        initial_priors: BeliefState,
        mask: Optional[torch.Tensor] = None,
        active_inference_fn: Optional[Callable] = None,
    ) -> BeliefState:
        r"""Forward through all blocks with cross-layer prior handoff.

        The posterior from layer :math:`\ell` becomes the prior for layer
        :math:`\ell+1`:

        - :math:`\mu_p^{(\ell+1)} = \mu^{(\ell)\star}` (attached for gradient flow)
        - :math:`\Sigma_p^{(\ell+1)} = \Sigma_{\text{embedding}}` (frozen)
        - :math:`\phi_p^{(\ell+1)} = \phi_{\text{embedding}}` (frozen)

        Args:
            beliefs: Initial beliefs from PriorBank.encode + positional.
            initial_priors: Same as beliefs at layer 0. Sigma and phi from
                this are frozen across all layers.
            mask: ``(B, N, N)`` causal mask.
            active_inference_fn: Optional callback for active inference.

        Returns:
            Final beliefs after all L layers.
        """
        priors = initial_priors

        for block in self.blocks:
            beliefs = block(beliefs, priors, mask, active_inference_fn)

            # Cross-layer prior handoff (Section 7):
            # Full (μ, Σ, φ) handoff with per-component damping

            # μ handoff: damped interpolation
            rho_mu = self.prior_handoff_rho
            if rho_mu == 1.0:
                new_prior_mu = beliefs.mu
            else:
                new_prior_mu = (1 - rho_mu) * priors.mu + rho_mu * beliefs.mu

            # Σ handoff: blend posterior with embedding to prevent cascade
            rho_sigma = self.prior_handoff_sigma
            if rho_sigma == 0.0:
                new_prior_sigma = initial_priors.sigma  # frozen (legacy default)
            else:
                new_prior_sigma = (1 - rho_sigma) * initial_priors.sigma + rho_sigma * beliefs.sigma
                # Floor: prevent collapse below embedding minimum
                if new_prior_sigma.dim() == 3:
                    new_prior_sigma = new_prior_sigma.clamp(min=1e-4, max=self.sigma_max)
                else:
                    # Full-cov: clamp diagonal
                    diag_idx = torch.arange(new_prior_sigma.shape[-1], device=new_prior_sigma.device)
                    new_prior_sigma[..., diag_idx, diag_idx] = (
                        new_prior_sigma[..., diag_idx, diag_idx].clamp(min=1e-4, max=self.sigma_max)
                    )

            # φ handoff: propagate posterior gauge frames
            new_prior_phi = beliefs.phi if self.prior_handoff_phi else initial_priors.phi

            priors = BeliefState(
                mu=new_prior_mu,
                sigma=new_prior_sigma,
                phi=new_prior_phi,
            )

        return beliefs
