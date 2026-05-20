"""
VFEBlock: single VFE layer.

Each block runs the E-step and applies normalization. No separate attention
sublayer — the E-step internally computes and updates attention at each
iteration (dynamic beta).
"""

from __future__ import annotations

import warnings
from typing import Optional, TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from transformer.vfe.config import VFEConfig

from transformer.core.types import BeliefState
from transformer.core.blocks import MahalanobisNorm, CenteredMahalanobisNorm, RMSNorm
from transformer.vfe.e_step import VFEEStep
from transformer.vfe.head_mixer import VFEHeadMixer


class _LayerNormSigmaAdapter(nn.Module):
    r"""Adapter wrapping ``nn.LayerNorm`` with the VFE-internal
    ``forward(mu, sigma) → mu_normed`` signature.

    This is a **gauge-blind** normalization: it operates on ``mu`` only and
    discards the covariance ``sigma`` argument entirely. Use only as an
    explicit ablation against the gauge-equivariant ``mahalnorm`` /
    ``centered_mahalnorm`` paths. Mirrors the documented-exception pattern
    used for ``attention.py::use_output_projection`` (see CLAUDE.md).
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        warnings.warn(
            "VFE norm_type='layernorm' is gauge-blind: nn.LayerNorm operates "
            "on mu only and ignores sigma, breaking strict gauge equivariance. "
            "Use 'mahalnorm' (default) or 'centered_mahalnorm' for the "
            "gauge-aware path; 'layernorm' is intended for ablation only.",
            UserWarning,
            stacklevel=2,
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, mu: torch.Tensor, sigma: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.norm(mu)


def _resolve_vfe_norm(norm_type: str, dim: int) -> Optional[nn.Module]:
    r"""Resolve a VFE ``norm_type`` string to a normalization module.

    All returned modules expose the VFE-internal signature
    ``forward(mu, sigma) → mu_normed``. ``CenteredMahalanobisNorm`` also
    accepts an optional ``mu_prior=`` kwarg at the call site (see
    :class:`VFEBlock` and :class:`VFEModel` forward methods). ``None`` is
    returned for ``norm_type='none'`` so callers can short-circuit.

    Args:
        norm_type: One of ``'mahalnorm'``, ``'centered_mahalnorm'``,
            ``'rmsnorm'``, ``'layernorm'``, or ``'none'``.
        dim: Embedding dimension (K).

    Raises:
        ValueError: If ``norm_type`` is not one of the five accepted values.
    """
    if norm_type == 'mahalnorm':
        return MahalanobisNorm(dim)
    if norm_type == 'centered_mahalnorm':
        return CenteredMahalanobisNorm(dim)
    if norm_type == 'rmsnorm':
        warnings.warn(
            "VFE norm_type='rmsnorm' has a learnable elementwise scale gamma "
            "that breaks O(K) gauge equivariance even though sqrt(mean(mu**2)) "
            "is itself O(K)-invariant. Use 'mahalnorm' or 'centered_mahalnorm' "
            "for the gauge-equivariant path; 'rmsnorm' is intended for ablation only.",
            UserWarning,
            stacklevel=2,
        )
        return RMSNorm(dim)
    if norm_type == 'layernorm':
        return _LayerNormSigmaAdapter(dim)
    if norm_type == 'none':
        return None
    raise ValueError(
        f"VFEConfig.norm_type={norm_type!r} not recognized; expected "
        "'mahalnorm', 'centered_mahalnorm', 'rmsnorm', 'layernorm', or 'none'."
    )


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

        # Optional Schur-commutant head mixer. Applied AFTER the E-step and
        # BEFORE normalization so it sees the converged belief and feeds the
        # norm/handoff path. None when the flag is off — zero compute cost.
        if cfg.use_equivariant_head_mixer:
            # Equivariance of the head mixer requires a TIED per-token gauge
            # across heads (one shared gauge action per token). GL(K) with
            # `generate_glK_multihead_generators` instead gives each head its
            # OWN gl(d_head) sub-algebra — independent per-head gauges — so
            # `kron(A, I_d)` does not commute with the actual gauge action.
            # SO3 / SON share a single group across heads and satisfy the
            # tied condition. Warn the user that the GL(K) multihead path
            # only approximately preserves equivariance (the deviation is
            # zero at the identity init `mixer_delta=0` and grows as A drifts
            # from I during training).
            if (
                cfg.gauge_group == 'GLK'
                and sum(mult for _, mult, _ in cfg.irrep_spec) > 1
            ):
                warnings.warn(
                    "use_equivariant_head_mixer=True with gauge_group='GLK' "
                    "and >1 head: the multihead GL(K) generator construction "
                    "(generate_glK_multihead_generators) gives each head an "
                    "INDEPENDENT gl(d_head) sub-algebra, so the mixer's "
                    "kron(A, I_d) action does not commute with the per-head "
                    "gauge action. Strict gauge equivariance holds only "
                    "under SO3/SON (single shared group across heads). For "
                    "GLK the mixer is exact at init (mixer_delta=0) and "
                    "deviates as A drifts from I.",
                    UserWarning,
                    stacklevel=2,
                )
            # Pass the super-block head-grouping so the mixer can exclude
            # heads that belong to a merged super-block (GL(d_super) commutant
            # is scalar — mixing across constituent heads breaks equivariance).
            # No-op when cross_couplings is empty.
            self.head_mixer = VFEHeadMixer(
                cfg.irrep_spec, cfg.embed_dim,
                super_block_head_groups=cfg.super_block_head_groups,
            )
        else:
            self.head_mixer = None

        # Normalization (resolves all five accepted norm_type values)
        self.norm = _resolve_vfe_norm(cfg.norm_type, cfg.embed_dim)

    def forward(
        self,
        beliefs: BeliefState,
        priors: BeliefState,
        mask: Optional[torch.Tensor] = None,
    ) -> BeliefState:
        """Forward pass: E-step + normalization.

        Args:
            beliefs: Current beliefs ``(mu, sigma, phi)``.
            priors: Layer priors ``(mu_p, sigma_p, phi_p)``.
            mask: ``(B, N, N)`` causal mask.

        Returns:
            Updated BeliefState.
        """
        beliefs = self.e_step(beliefs, priors, mask)

        # Head mixer (optional, opt-in). Applies (μ, Σ) ↦ (Mμ, MΣM^T) per type.
        # Skipped when disabled; mixer.is_identity() short-circuit isn't used
        # here because we want autograd to flow through mixer_delta even when
        # it's at its init value (otherwise gradients would be silently zeroed
        # from a Python-side branch).
        if self.head_mixer is not None:
            mu_mixed, sigma_mixed = self.head_mixer(beliefs.mu, beliefs.sigma)
            beliefs = beliefs._replace(mu=mu_mixed, sigma=sigma_mixed)

        if self.norm is not None:
            # CenteredMahalanobisNorm needs mu_prior to actually center the
            # residual delta = mu - mu_prior. Other norms ignore mu_prior.
            if isinstance(self.norm, CenteredMahalanobisNorm):
                mu_normed = self.norm(beliefs.mu, beliefs.sigma, mu_prior=priors.mu)
            else:
                mu_normed = self.norm(beliefs.mu, beliefs.sigma)
            beliefs = beliefs._replace(mu=mu_normed)

        return beliefs
