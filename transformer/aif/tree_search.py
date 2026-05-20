"""
Tree-search policies for multi-step Expected Free Energy.

Phase 1 shipped the depth-1 reduction anchor; Phase 2 adds the per-branch
beam search at ``horizon_D > 1``. The expansion grows the tree
level-by-level: at each level every node in the frontier is expanded by
its top-``beam_width`` candidates from the model's predictive distribution
at that node's position. Each child is scored via
:func:`compute_G_at_node` (or :func:`score_components_from_beliefs` on a
cache hit), and its ``G_cum`` is set to the geometric-discounted path-sum
of ``G_local`` from the root.

Back-propagation aggregates leaves up to the root's immediate children.
This Phase-2 implementation uses MEAN aggregation: a root child's
back-propagated value is the average of ``G_cum`` over the
``beam_width^(D-1)`` leaves reachable through it. Mean-over-beam is the
uniform-policy expectation of the canonical Friston-2021 recursion (the
softmax-weighted form is Phase 3 / sophisticated inference).

Phase 3 (`branching_strategy='sophisticated'`) will replace the mean
aggregation with the explicit child-action policy posterior:
:math:`E_{q(a' | s_d)}[G_{d+1}]` per
[Friston2021SophisticatedInference]. The stub :func:`sophisticated_expand`
raises ``NotImplementedError``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import torch
import torch.nn.functional as F

from transformer.aif.efe_score import (
    _bald_mi_and_ambiguity,
    compute_G_at_node,
    score_components_from_beliefs,
)
from transformer.aif.policy import EFEComponents, PolicyNode
from transformer.core.types import BeliefState

if TYPE_CHECKING:
    from transformer.aif.belief_cache import BeliefStateCache
    from transformer.aif.config import AIFConfig
    from transformer.aif.preferences import Preference
    from transformer.vfe.model import VFEModel


_EPS: float = 1e-12


@torch.no_grad()
def pick_top_candidates(
    last_probs: torch.Tensor,
    beam_width: int,
) -> List[int]:
    r"""Return the token IDs of the top-`beam_width` candidates by probability.

    Args:
        last_probs: ``(V,)`` predictive distribution at the position from
            which the candidates branch.
        beam_width: number of candidates to keep.

    Returns:
        A list of ``beam_width`` token IDs (Python ints).
    """
    k = min(beam_width, last_probs.shape[-1])
    _, top_ids = torch.topk(last_probs, k)
    return [int(t) for t in top_ids.tolist()]


def _build_extended_context(
    prompt_ids: torch.Tensor,
    action_seq: Tuple[int, ...],
    max_seq_len: int,
) -> torch.Tensor:
    r"""Concatenate prompt + action_seq, truncating to the last ``max_seq_len`` tokens.

    Matches the truncation pattern in
    ``transformer/vfe/efe.py:170-171`` and
    ``transformer/aif/efe_score.py:compute_G_at_node``: the model always
    sees only the trailing ``max_seq_len`` tokens. The cache key uses the
    untruncated ``action_seq`` so the prefix-share guarantee holds for
    any tree depth up to ``max_seq_len``.
    """
    if not action_seq:
        return prompt_ids
    device = prompt_ids.device
    extension = torch.tensor([list(action_seq)], dtype=prompt_ids.dtype, device=device)
    extended = torch.cat([prompt_ids, extension], dim=1)
    if extended.shape[1] > max_seq_len:
        extended = extended[:, -max_seq_len:]
    return extended


@torch.no_grad()
def _last_position_predictive(
    beliefs: BeliefState,
    prior_bank,
    decode_tau: float,
) -> torch.Tensor:
    r"""Decode the cached belief into a ``(V,)`` predictive at the last position.

    Used during beam expansion when a node's predictive is needed but only
    its cached BeliefState is available. The cached belief carries
    ``mu = mu_final`` (post-final-norm) so the decode matches the result
    that ``VFEModel.forward_with_beliefs`` produced when the node was first
    scored.
    """
    mu_last = beliefs.mu[:, -1:, :]
    if beliefs.sigma.dim() == 4:
        sigma_last = beliefs.sigma[:, -1:, :, :]
    else:
        sigma_last = beliefs.sigma[:, -1:, :]
    logits = prior_bank.decode(mu_last, sigma_last, tau=max(decode_tau, _EPS))
    return F.softmax(logits[:, 0, :], dim=-1).squeeze(0)


def _score_child_with_cache(
    parent_context_ids: torch.Tensor,
    parent_action_seq: Tuple[int, ...],
    action: int,
    model: 'VFEModel',
    preference: 'Preference',
    cache: 'BeliefStateCache',
    cfg: 'AIFConfig',
) -> EFEComponents:
    r"""Score a child action, hitting the cache when its belief is already known.

    On miss: calls :func:`compute_G_at_node` (a full forward), stores the
    converged belief in ``cache`` under the child's action_seq.

    On hit: re-decodes the cached belief via
    :func:`score_components_from_beliefs` — no forward pass — but the
    Monte Carlo samples for BALD MI are drawn fresh (the estimator is
    unbiased; the variance shows up as score noise across calls).
    """
    child_key = parent_action_seq + (action,)
    cached = cache.get(child_key)
    if cached is not None:
        return score_components_from_beliefs(cached, model, preference, cfg)
    components, beliefs = compute_G_at_node(
        context_ids=parent_context_ids,
        candidate_action=action,
        model=model,
        preference=preference,
        cfg=cfg,
    )
    cache.put(child_key, beliefs)
    return components


@torch.no_grad()
def beam_expand(
    prompt_ids: torch.Tensor,
    model: 'VFEModel',
    preference: 'Preference',
    cache: 'BeliefStateCache',
    cfg: 'AIFConfig',
) -> List[PolicyNode]:
    r"""Per-branch beam tree expansion to depth ``cfg.horizon_D``.

    Builds a tree where each node has ``beam_width`` children drawn from
    the top of that node's predictive distribution. Leaves at depth D
    have ``G_cum`` equal to the geometric-discounted path-sum
    :math:`\sum_d \gamma_{\text{disc}}^d G_{\text{local},d}`. Internal
    nodes (including root children) receive a back-propagated value equal
    to the MEAN of ``G_cum`` over the leaves reachable through them.

    For ``horizon_D == 1`` the function returns the root's children with
    ``G_cum = G_local`` and no back-propagation, matching the existing
    depth-1 anchor in ``AIFGenerator._score_root_children``.

    Args:
        prompt_ids: ``(1, N_prompt)`` token IDs of the current context.
        model: trained ``VFEModel``.
        preference: configured ``Preference`` instance.
        cache: ``BeliefStateCache`` for cross-node and cross-commit
            prefix sharing.
        cfg: ``AIFConfig`` controlling horizon, beam, discount, sampling.

    Returns:
        List of ``beam_width`` (or fewer) root children with
        ``G_cum`` set to the back-propagated value. The caller softmins
        over these to pick the committed action.
    """
    # Root predictive — seed the level-1 expansion.
    root_cached = cache.get(())
    if root_cached is None:
        _, root_beliefs = model.forward_with_beliefs(prompt_ids)
        cache.put((), root_beliefs)
        root_last_probs = _last_position_predictive(
            root_beliefs, model.prior_bank, cfg.decode_tau,
        )
    else:
        root_last_probs = _last_position_predictive(
            root_cached, model.prior_bank, cfg.decode_tau,
        )

    root = PolicyNode(action_seq=(), depth=0)

    # Level 0 holds [root], levels 1..D hold the corresponding-depth nodes.
    level_nodes: List[List[PolicyNode]] = [[root]]
    parent_predictives: List[torch.Tensor] = [root_last_probs]

    max_seq_len = model.cfg.max_seq_len

    for depth_d in range(cfg.horizon_D):
        next_level: List[PolicyNode] = []
        next_predictives: List[torch.Tensor] = []

        parents_at_level = level_nodes[depth_d]
        for parent_idx, parent in enumerate(parents_at_level):
            parent_last_probs = parent_predictives[parent_idx]
            parent_context_ids = _build_extended_context(
                prompt_ids, parent.action_seq, max_seq_len,
            )

            child_candidates = pick_top_candidates(
                parent_last_probs, cfg.beam_width,
            )
            discount_factor = cfg.discount ** depth_d

            for action in child_candidates:
                components = _score_child_with_cache(
                    parent_context_ids=parent_context_ids,
                    parent_action_seq=parent.action_seq,
                    action=action,
                    model=model,
                    preference=preference,
                    cache=cache,
                    cfg=cfg,
                )
                child_g_cum = parent.G_cum + discount_factor * components.G_local
                child = parent.child(
                    action=action,
                    components=components,
                    G_cum=child_g_cum,
                )
                next_level.append(child)

                # Fetch the child's belief from the cache to compute the
                # NEXT-level predictive (needed only if depth_d + 1 < D).
                if depth_d + 1 < cfg.horizon_D:
                    child_beliefs = cache.get(child.belief_cache_key)
                    if child_beliefs is None:
                        # The cache was populated either by _score_child_with_cache
                        # (on miss path) or by a prior commit. If somehow absent,
                        # re-forward defensively.
                        extended_ids = _build_extended_context(
                            prompt_ids, child.action_seq, max_seq_len,
                        )
                        _, child_beliefs = model.forward_with_beliefs(extended_ids)
                        cache.put(child.belief_cache_key, child_beliefs)
                    child_predictive = _last_position_predictive(
                        child_beliefs, model.prior_bank, cfg.decode_tau,
                    )
                    next_predictives.append(child_predictive)

        level_nodes.append(next_level)
        parent_predictives = next_predictives

    # Back-propagation: mean over child V's, leaves contribute their G_cum.
    # Use id(node) as the dict key because PolicyNode is frozen and hashable
    # by value, but two distinct nodes with the same action_seq would
    # collide; id() is safer.
    node_V: Dict[int, float] = {}
    for leaf in level_nodes[cfg.horizon_D]:
        node_V[id(leaf)] = leaf.G_cum

    # Build child-list per parent to support the mean aggregation.
    children_of: Dict[int, List[PolicyNode]] = {}
    for level_idx in range(1, cfg.horizon_D + 1):
        for node in level_nodes[level_idx]:
            parent_id = id(node.parent)
            children_of.setdefault(parent_id, []).append(node)

    # Compute V from depth (D-1) up to root.
    for depth_back in range(cfg.horizon_D - 1, -1, -1):
        for node in level_nodes[depth_back]:
            children = children_of.get(id(node), [])
            if children:
                node_V[id(node)] = (
                    sum(node_V[id(c)] for c in children) / len(children)
                )
            else:
                node_V[id(node)] = node.G_cum

    # Rebuild root children with backpropagated G_cum.
    final_root_children: List[PolicyNode] = []
    for child in level_nodes[1]:
        v = node_V[id(child)]
        updated = PolicyNode(
            action_seq=child.action_seq,
            depth=child.depth,
            parent=child.parent,
            components=child.components,
            G_cum=v,
        )
        final_root_children.append(updated)
    return final_root_children


def sophisticated_expand(*_args, **_kwargs):
    r"""Phase 3 stub for the [Friston2021SophisticatedInference] recursion.

    Raises ``NotImplementedError`` so any caller that selects
    ``branching_strategy='sophisticated'`` fails fast.
    ``AIFConfig.__post_init__`` raises the same error earlier; this stub
    guards against a path that bypasses the config check.
    """
    raise NotImplementedError(
        "Sophisticated-inference recursion per [Friston2021SophisticatedInference] "
        "is Phase 3 of the build-out and not yet implemented. Use "
        "branching_strategy='beam' for the Phase 2 tree search."
    )
