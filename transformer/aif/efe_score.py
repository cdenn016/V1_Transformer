"""
EFE scoring helpers for the AIF policy tree.

The per-node Expected Free Energy is computed in canonical Form-3 BALD
decomposition (per `01_canonical_efe_for_lm.md` eq 11):

.. math::
    G(\\pi) = E_{q(o|\\pi)}[-\\log p^*(o|C)]
           + E_{q(s|\\pi)}[H[p(o|s)]]
           - I_{q(s,o|\\pi)}(s ; o)

The function reuses the BALD MI estimator from
`transformer/vfe/efe.py:VFEExpectedFreeEnergy._compute_epistemic_value`
adapted to take the BeliefState directly (no re-encode). All gradients
are detached — generation is `torch.no_grad()`.
"""

from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING, Tuple

import torch
import torch.nn.functional as F

from transformer.aif.policy import EFEComponents
from transformer.core.types import BeliefState

if TYPE_CHECKING:
    from transformer.aif.config import AIFConfig
    from transformer.aif.preferences import Preference
    from transformer.vfe.model import VFEModel


_EPS: float = 1e-12


@torch.no_grad()
def _bald_mi_and_ambiguity(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    prior_bank,
    n_samples: int,
    decode_tau: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Compute BALD mutual information and mean predictive entropy at one position.

    Both quantities come out of the same sampling pass. The MI is the
    canonical epistemic value; the mean predictive entropy is the
    canonical ambiguity per [ParrPezzuloFriston2022 Ch. 2] (the marginal
    H[p_bar] form would double-count uncertainty, see the disclosure at
    `transformer/core/expected_free_energy.py:117-126`).

    Args:
        mu: ``(1, 1, K)`` belief mean at the position of interest (the
            last position of the candidate context).
        sigma: ``(1, 1, K)`` diagonal variance at the same position.
        prior_bank: ``VFEPriorBank`` instance providing ``decode``.
        n_samples: Monte Carlo sample count S.
        decode_tau: PriorBank.decode temperature for the AIF readout.

    Returns:
        ``(mi, mean_H)`` — both 0-dim tensors. MI is the BALD MI used as
        epistemic value; mean_H is the canonical ambiguity term.
    """
    if sigma.dim() == 4:
        sigma = torch.diagonal(sigma, dim1=-2, dim2=-1)
    std = sigma.clamp(min=1e-6).sqrt()

    # Pass z_s = mu + std * eps through the decode. The reparameterized
    # sample already absorbs sigma so we pass sigma=0 to the decoder; the
    # KL-trace term inside decode would otherwise count sigma twice (see
    # the discussion at transformer/core/active_inference.py:225-237).
    zero_sigma = torch.zeros_like(sigma)
    probs_avg: Optional[torch.Tensor] = None
    sample_entropies: List[torch.Tensor] = []
    for _ in range(n_samples):
        noise = torch.randn_like(mu)
        z_s = mu + std * noise
        logits_s = prior_bank.decode(z_s, zero_sigma, tau=decode_tau)  # (1, 1, V)
        probs_s = F.softmax(logits_s[:, -1, :], dim=-1)  # (1, V)
        log_probs_s = probs_s.clamp(min=_EPS).log()
        sample_entropies.append(-(probs_s * log_probs_s).sum(dim=-1))  # (1,)
        contrib = probs_s / n_samples
        probs_avg = contrib if probs_avg is None else probs_avg + contrib

    log_probs_avg = probs_avg.clamp(min=_EPS).log()
    H_bar = -(probs_avg * log_probs_avg).sum(dim=-1)  # (1,)
    mean_H = torch.stack(sample_entropies).mean(dim=0)  # (1,)
    mi = (H_bar - mean_H).clamp(min=0.0)  # MI is non-negative
    return mi.squeeze(), mean_H.squeeze()


@torch.no_grad()
def score_components_from_beliefs(
    beliefs: BeliefState,
    model: 'VFEModel',
    preference: 'Preference',
    cfg: 'AIFConfig',
) -> EFEComponents:
    r"""Score EFE components from a cached BeliefState — no forward pass.

    Used by the tree-search inner loop when a candidate's belief is already
    cached (e.g. it survived from a prior commit via
    :meth:`BeliefStateCache.commit_action`). Bitwise equivalent to
    :func:`compute_G_at_node` minus the MC randomness in BALD: the BeliefState
    determines mu/sigma at the last position; the only stochastic component
    is the ``epistemic_samples`` Monte Carlo draws against the same
    ``torch.randn`` stream.

    Decoding from the cached belief uses ``prior_bank.decode``. The cached
    belief already carries ``mu = mu_final`` (post-final-norm) because
    `VFEModel._encode_step_decode` substitutes mu_final before returning.

    Args:
        beliefs: cached converged BeliefState.
        model: trained ``VFEModel`` (used only for ``prior_bank``).
        preference: configured ``Preference`` instance.
        cfg: ``AIFConfig`` for sampling and weights.

    Returns:
        ``EFEComponents`` decomposition.
    """
    # Last-position belief.
    mu_last = beliefs.mu[:, -1:, :]
    if beliefs.sigma.dim() == 4:
        sigma_last = beliefs.sigma[:, -1:, :, :]
    else:
        sigma_last = beliefs.sigma[:, -1:, :]

    # Pragmatic: from the decoded predictive at the last position.
    last_logits = model.prior_bank.decode(
        mu_last, sigma_last, tau=max(cfg.decode_tau, _EPS),
    )
    last_probs = F.softmax(last_logits[:, 0, :], dim=-1)  # (1, V)
    pragmatic = preference.pragmatic(last_probs).squeeze()

    # Ambiguity + epistemic from one BALD sampling pass.
    mi, mean_H = _bald_mi_and_ambiguity(
        mu_last, sigma_last,
        prior_bank=model.prior_bank,
        n_samples=cfg.epistemic_samples,
        decode_tau=cfg.decode_tau,
    )

    pragmatic_f = float(pragmatic.item())
    ambiguity_f = float(mean_H.item())
    epistemic_f = float(mi.item())
    g_local = pragmatic_f + ambiguity_f - cfg.epistemic_weight * epistemic_f
    return EFEComponents(
        pragmatic=pragmatic_f,
        ambiguity=ambiguity_f,
        epistemic=epistemic_f,
        G_local=g_local,
    )


@torch.no_grad()
def compute_G_at_node(
    context_ids: torch.Tensor,
    candidate_action: int,
    model: 'VFEModel',
    preference: 'Preference',
    cfg: 'AIFConfig',
) -> Tuple[EFEComponents, BeliefState]:
    r"""Compute the per-node EFE components for a candidate action.

    Appends ``candidate_action`` to ``context_ids``, runs
    ``model.forward_with_beliefs`` (no double-encode), decomposes the
    resulting predictive into pragmatic + ambiguity - epistemic.

    Args:
        context_ids: ``(1, N)`` token IDs of the context leading to this
            node. Includes any committed-action tokens already emitted.
        candidate_action: token ID being scored.
        model: trained ``VFEModel``.
        preference: configured ``Preference`` instance.
        cfg: ``AIFConfig`` for sampling and weights.

    Returns:
        ``(components, beliefs)`` — the EFE decomposition and the
        converged belief tuple at the extended context (used for caching
        and for downstream multi-step expansion).
    """
    device = context_ids.device
    max_len = model.cfg.max_seq_len

    # Append candidate, truncate to max_seq_len (matches vfe/efe.py:170-171).
    trial_ids = torch.cat(
        [context_ids, torch.tensor([[candidate_action]], device=device)],
        dim=1,
    )
    if trial_ids.shape[1] > max_len:
        trial_ids = trial_ids[:, -max_len:]

    # Single forward — returns both logits and converged BeliefState so
    # the AIF path does not re-encode (closes the double-encode in
    # vfe/efe.py:191-196 per verifier §8.8).
    _logits, beliefs = model.forward_with_beliefs(trial_ids)

    # Last-position predictive for the candidate's pragmatic term.
    # Re-decode at the AIF-controlled temperature `cfg.decode_tau` —
    # independent of `model.cfg.decode_tau` so AIF generation can be tuned
    # separately. The forward's `_logits` are scaled by the model's tau;
    # for AIF scoring we want the AIF tau, which is one extra (V, K)
    # matmul per node (small relative to the rest of the forward).
    mu_last = beliefs.mu[:, -1:, :]
    if beliefs.sigma.dim() == 4:
        sigma_last = beliefs.sigma[:, -1:, :, :]
    else:
        sigma_last = beliefs.sigma[:, -1:, :]
    last_logits = model.prior_bank.decode(
        mu_last, sigma_last, tau=max(cfg.decode_tau, _EPS),
    )
    last_probs = F.softmax(last_logits[:, 0, :], dim=-1)  # (1, V)

    # Pragmatic: E_{q(o|a)}[-log p*(o|C)].
    pragmatic = preference.pragmatic(last_probs).squeeze()

    # Ambiguity + epistemic share one BALD sampling pass at the last
    # position. mu_last / sigma_last already extracted above.
    mi, mean_H = _bald_mi_and_ambiguity(
        mu_last, sigma_last,
        prior_bank=model.prior_bank,
        n_samples=cfg.epistemic_samples,
        decode_tau=cfg.decode_tau,
    )

    pragmatic_f = float(pragmatic.item())
    ambiguity_f = float(mean_H.item())
    epistemic_f = float(mi.item())
    g_local = pragmatic_f + ambiguity_f - cfg.epistemic_weight * epistemic_f

    components = EFEComponents(
        pragmatic=pragmatic_f,
        ambiguity=ambiguity_f,
        epistemic=epistemic_f,
        G_local=g_local,
    )
    return components, beliefs
