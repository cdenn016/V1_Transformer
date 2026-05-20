"""
Training-time Expected Free Energy augmentation (Phase 4 / Regime A).

Per-batch, gradient-tracked computation of
:math:`L_{\\text{AIF}} = E_t[G_t(\\pi_{\\text{observed}})]` where the
observed training sequence is treated as the agent's chosen policy
(trajectory-as-policy interpretation; see
``docs/plans/2026-05-19-aif-transformer-buildout/06_plan.md`` §6 Phase 4
and ``01_canonical_efe_for_lm.md`` §4).

The standard CE loss covers the discriminative training signal
:math:`-\\log q(y_t \\mid \\text{context}_t)`. The AIF augmentation
:math:`L_{\\text{AIF}}` adds the canonical EFE decomposition at each
position computed from the model's OWN predictive and beliefs (no
target tokens — Law 1 preserved):

.. math::
    G_t = E_{q(o \\mid s_t)}[-\\log p^*(o \\mid C)]
        + E_{q(s_t)}[H[p(o \\mid s_t)]]
        - I_{q(s_t, o)}(s_t ; o).

The full training loss with augmentation is then
:math:`L_{\\text{total}} = L_{\\text{CE}} + \\lambda_{\\text{AIF}} \\,
L_{\\text{AIF}}` where :math:`\\lambda_{\\text{AIF}}` is
``cfg.aif_loss_weight``.

Each EFE term is individually gated by a boolean flag in ``AIFConfig``
(``train_include_pragmatic`` / ``train_include_ambiguity`` /
``train_include_epistemic``) so the user can drop the expensive BALD MC
sampling without losing the preference-aligned pragmatic regulariser.
At full configuration the MC adds approximately :math:`S \\times` the
model's existing decode memory cost — at the active config (B=64,
N=128, V=50257, S=4) that is 6.6 GB additional, comfortably within the
RTX 5090's 32 GB envelope.

Regime A (this module): no tree expansion at training time. The
observed trajectory IS the policy; G is evaluated per-position on the
model's own forward outputs without any extra forward passes. Regime B
(tree expansion at training time) is infeasible at ~200-500x CE cost
per ``04_compute_feasibility.md`` §9 and is not implemented.
"""

from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

import torch
import torch.nn.functional as F

from transformer.core.types import BeliefState

if TYPE_CHECKING:
    from transformer.aif.config import AIFConfig
    from transformer.aif.preferences import Preference


_EPS: float = 1e-12


def _pragmatic_per_position(
    logits: torch.Tensor,
    preference: 'Preference',
    cfg: 'AIFConfig',
) -> torch.Tensor:
    r"""Per-position pragmatic value :math:`E_{q(o \mid a)}[-\log p^*(o)]`.

    Reshapes the (B, N, V) logits into (B*N, V), applies the preference's
    pragmatic operator, reshapes back. Gradient flows through the
    softmax into the logits.

    Args:
        logits: (B, N, V) gradient-tracked.
        preference: Preference instance.
        cfg: AIFConfig (uses ``cfg.decode_tau`` for the temperature).

    Returns:
        (B, N) per-position pragmatic value.
    """
    B, N, V = logits.shape
    probs = F.softmax(logits / max(cfg.decode_tau, _EPS), dim=-1)  # (B, N, V)
    pragmatic_flat = preference.pragmatic(probs.view(B * N, V))  # (B*N,)
    return pragmatic_flat.view(B, N)


def _bald_ambiguity_and_epistemic_per_position(
    beliefs: BeliefState,
    prior_bank,
    cfg: 'AIFConfig',
) -> tuple:
    r"""Per-position BALD ambiguity and epistemic value.

    Ambiguity :math:`E_{q(s_t)}[H[p(o \mid s_t)]]` and epistemic
    :math:`I_{q(s_t, o)}(s_t; o) = H[\bar p] - E_s[H[p_s]]` share the
    same MC sampling pass. Each MC sample reparameterizes
    :math:`z_s = \mu + \sqrt{\Sigma}\, \varepsilon`, decodes through the
    PriorBank at the point-conditional ``sigma=0`` (to avoid the double-
    counting that happens when ``decode`` adds its own KL-trace term to
    a reparameterized sample), and accumulates per-sample entropies.

    Gradient flows through the reparameterized samples into both ``mu``
    AND ``sigma`` (the latter via the ``sqrt(sigma)`` reparameterization).

    Args:
        beliefs: BeliefState with gradient-tracked mu / sigma.
        prior_bank: VFEPriorBank for the decode readout.
        cfg: AIFConfig (uses ``cfg.epistemic_samples`` and
            ``cfg.decode_tau``).

    Returns:
        ``(ambiguity, epistemic)`` — both (B, N) tensors.
    """
    mu = beliefs.mu  # (B, N, K)
    if beliefs.sigma.dim() == 4:
        sigma = torch.diagonal(beliefs.sigma, dim1=-2, dim2=-1)
    else:
        sigma = beliefs.sigma  # (B, N, K)
    std = sigma.clamp(min=1e-6).sqrt()

    zero_sigma = torch.zeros_like(sigma)
    probs_avg: Optional[torch.Tensor] = None
    sample_entropies: List[torch.Tensor] = []

    for _ in range(cfg.epistemic_samples):
        noise = torch.randn_like(mu)
        z_s = mu + std * noise  # gradient through mu and (via std) sigma
        logits_s = prior_bank.decode(z_s, zero_sigma, tau=max(cfg.decode_tau, _EPS))
        probs_s = F.softmax(logits_s, dim=-1)  # (B, N, V)
        log_probs_s = probs_s.clamp(min=_EPS).log()
        sample_entropies.append(-(probs_s * log_probs_s).sum(dim=-1))  # (B, N)
        contrib = probs_s / cfg.epistemic_samples
        probs_avg = contrib if probs_avg is None else probs_avg + contrib

    log_probs_avg = probs_avg.clamp(min=_EPS).log()
    H_bar = -(probs_avg * log_probs_avg).sum(dim=-1)  # (B, N)
    ambiguity = torch.stack(sample_entropies).mean(dim=0)  # (B, N)
    epistemic = (H_bar - ambiguity).clamp(min=0.0)  # (B, N), MI is non-negative
    return ambiguity, epistemic


def compute_training_efe_loss(
    logits: torch.Tensor,
    beliefs: BeliefState,
    preference: 'Preference',
    prior_bank,
    cfg: 'AIFConfig',
) -> torch.Tensor:
    r"""Compute the per-batch AIF training augmentation :math:`L_{\text{AIF}}`.

    Treats the batch's observed trajectory as the chosen policy: for
    each position t the per-position EFE :math:`G_t` is computed from
    the model's converged beliefs at t and the predictive at t. NO
    target tokens are used — Law 1 (E-step blindness) is preserved by
    construction.

    .. math::
        L_{\text{AIF}} = \frac{1}{B N} \sum_{b, t}
            \bigl[ \text{prag}_{b,t} + \text{amb}_{b,t}
            - \lambda_{\text{epi}}\, \text{epi}_{b,t} \bigr]

    Each term is gated by a boolean flag in ``AIFConfig`` so users can
    drop the expensive BALD MC sampling. The full canonical form
    includes all three; setting ``train_include_ambiguity=False`` and
    ``train_include_epistemic=False`` recovers the pragmatic-only
    regularisation (Regime A "<1% overhead" mode per
    ``04_compute_feasibility.md`` §9).

    Args:
        logits: ``(B, N, V)`` gradient-tracked predictive logits from
            ``VFEModel.forward_with_beliefs``.
        beliefs: gradient-tracked BeliefState from the same forward.
        preference: configured ``Preference`` instance.
        prior_bank: ``VFEPriorBank`` reference for the BALD decode pass.
        cfg: ``AIFConfig`` for sampling, weights, and term-inclusion flags.

    Returns:
        Scalar tensor ``L_AIF`` ready to add into the M-step loss:
        ``L_total = L_CE + cfg.aif_loss_weight * L_AIF``. The returned
        tensor carries gradients into ``logits`` (through pragmatic) and
        into ``beliefs.mu`` / ``beliefs.sigma`` (through BALD
        reparameterization).
    """
    if cfg.training_objective != 'efe_augmented':
        raise RuntimeError(
            "compute_training_efe_loss called but "
            f"cfg.training_objective={cfg.training_objective!r}. "
            "Set training_objective='efe_augmented' to use this loss."
        )

    if not (
        cfg.train_include_pragmatic
        or cfg.train_include_ambiguity
        or cfg.train_include_epistemic
    ):
        raise ValueError(
            "compute_training_efe_loss called with all three "
            "train_include_* flags False — the augmentation is a no-op. "
            "Enable at least one term, or disable training_objective='efe_augmented'."
        )

    B, N, _V = logits.shape
    device = logits.device
    dtype = logits.dtype

    G_per_position = torch.zeros(B, N, device=device, dtype=dtype)

    if cfg.train_include_pragmatic:
        G_per_position = G_per_position + _pragmatic_per_position(
            logits=logits, preference=preference, cfg=cfg,
        )

    if cfg.train_include_ambiguity or cfg.train_include_epistemic:
        ambiguity, epistemic = _bald_ambiguity_and_epistemic_per_position(
            beliefs=beliefs, prior_bank=prior_bank, cfg=cfg,
        )
        if cfg.train_include_ambiguity:
            G_per_position = G_per_position + ambiguity
        if cfg.train_include_epistemic:
            G_per_position = G_per_position - cfg.epistemic_weight * epistemic

    return G_per_position.mean()
