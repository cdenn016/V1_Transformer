"""
VFEStack: L layers with cross-layer prior handoff.

Cross-layer handoff (VFE_Transformer_Idea.md Section 7):
    mu_prior^{l+1}    = (1-rho_mu) mu_prior^l + rho_mu mu*^l
    sigma_prior^{l+1} = (1-rho_sigma) sigma_embed + rho_sigma sigma*^l
    phi_prior^{l+1}   = phi_embedding  (frozen; phi flows via beliefs)

Defaults: rho_mu=1.0 (full mu flow), rho_sigma=0.0 (frozen at embedding).
Sigma damping prevents progressive variance collapse (sigma cascade) while
allowing gradual prior refinement.

Sigma freezing is architecturally enforced — priors.sigma is never
reassigned from the posterior. priors.phi is likewise never consumed by
VFEEStep (phi is initialised from beliefs.phi each layer), so phi is
held at the embedding value in the prior state.

Implementation note — mean-only cascade vs canonical hierarchical VI.
====================================================================
Under the default ``prior_handoff_rho=1.0, prior_handoff_sigma=0.0``,
only the posterior MEAN of layer L flows into the prior of layer L+1;
the posterior VARIANCE and the posterior GAUGE FRAME are discarded and
the prior at L+1 reuses the embedding sigma and embedding phi. This is
a *point-estimate handoff*, not the full distributional handoff that
canonical hierarchical variational inference (Friston 2017; Parr,
Pezzulo, Friston 2022; Blei, Kucukelbir, Jordan 2017) prescribes.

Effect: cross-layer uncertainty about ``s_l`` is dropped at every
boundary — layer L+1's KL(q^{L+1} || prior) treats the embedding sigma
as ground truth even though layer L's posterior has refined it. The
codepath for full handoff exists (``prior_handoff_sigma>0`` triggers
the SPD eigenvalue floor branch), but the default is point-passing.
Set ``prior_handoff_sigma=1.0`` (and a matching mechanism for phi if
desired) to recover the canonical scheme.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

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
        self.sigma_max = cfg.sigma_max
        self.blocks = nn.ModuleList([
            VFEBlock(cfg, generators) for _ in range(cfg.n_layers)
        ])

    def forward(
        self,
        beliefs: BeliefState,
        initial_priors: BeliefState,
        mask: Optional[torch.Tensor] = None,
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

        Returns:
            Final beliefs after all L layers.
        """
        priors = initial_priors

        for block in self.blocks:
            beliefs = block(beliefs, priors, mask)

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
                # Apply the same [1e-4, sigma_max] clamp the rho>0 branch
                # applies, so the safety envelope doesn't switch silently
                # when the user moves rho_sigma off 0. Diagonal path only;
                # full-cov frozen embedding stays as-is (eigh+clamp on
                # every forward would be expensive and the embedding cannot
                # drift outside its initialization range under rho=0).
                if new_prior_sigma.dim() == 3:
                    new_prior_sigma = new_prior_sigma.clamp(min=1e-4, max=self.sigma_max)
            else:
                new_prior_sigma = (1 - rho_sigma) * initial_priors.sigma + rho_sigma * beliefs.sigma
                # Floor: prevent collapse below embedding minimum
                if new_prior_sigma.dim() == 3:
                    new_prior_sigma = new_prior_sigma.clamp(min=1e-4, max=self.sigma_max)
                else:
                    # Full-cov: symmetrize and lift to SPD via an out-of-place
                    # eigenvalue floor (preserves autograd; an in-place indexed
                    # write into an autograd-attached intermediate would trip
                    # PyTorch's version counter and either error at backward
                    # or silently corrupt gradients).
                    new_prior_sigma = 0.5 * (
                        new_prior_sigma + new_prior_sigma.transpose(-1, -2)
                    )
                    eigvals, eigvecs = torch.linalg.eigh(new_prior_sigma)
                    eigvals = eigvals.clamp(min=1e-4, max=self.sigma_max)
                    new_prior_sigma = (
                        eigvecs @ torch.diag_embed(eigvals) @ eigvecs.transpose(-1, -2)
                    )
                    new_prior_sigma = 0.5 * (
                        new_prior_sigma + new_prior_sigma.transpose(-1, -2)
                    )

            # priors.phi is never consumed by VFEEStep (phi is initialised
            # from beliefs.phi, which already flows posterior -> next-layer
            # input), so the prior phi stays at the embedding value.
            new_prior_phi = initial_priors.phi

            priors = BeliefState(
                mu=new_prior_mu,
                sigma=new_prior_sigma,
                phi=new_prior_phi,
            )

        return beliefs
