"""
Tree-search policies for multi-step Expected Free Energy.

Phase 1 shipped the depth-1 reduction anchor; Phase 2 added per-branch
beam search at ``horizon_D > 1`` with mean back-propagation; Phase 3
adds the canonical [Friston2021SophisticatedInference] recursion with a
softmax-weighted child-action posterior. The expansion grows the tree
level-by-level: at each level every node in the frontier is expanded by
its top-``beam_width`` candidates from the model's predictive
distribution at that node's position. Each child is scored via
:func:`compute_G_at_node` (or :func:`score_components_from_beliefs` on
a cache hit).

Back-propagation walks the tree depth :math:`D \\to 0` computing a
single recursive value :math:`V` at each node:

.. math::
    V(\\text{leaf}) = G_{\\text{local}}(\\text{leaf})

.. math::
    V(\\text{internal}, d) = G_{\\text{local}}(d) + \\gamma_{\\text{disc}}
        \\sum_{a' \\in \\text{beam}(d+1)} q(a' \\mid s_d) \\, V(a').

The choice of child posterior :math:`q(a' \\mid s_d)` is dispatched on
``cfg.branching_strategy``:

- ``'beam'`` and ``'top_k'``: uniform — :math:`q(a') = 1/|\\text{beam}|`.
  Equivalent to the mean back-propagation; this is the
  :math:`\\gamma \\to 0` limit of the sophisticated form. Under uniform
  posterior the recursion unrolls to the geometric-discounted path-sum
  averaged over leaves, matching Phase 2's behaviour bitwise (modulo
  floating-point order).
- ``'sophisticated'``: softmax-weighted —
  :math:`q(a' \\mid s_d) \\propto \\exp(-\\gamma\\, V(a'))`. Implements
  the canonical [Friston2021SophisticatedInference] recursion. At
  :math:`\\gamma \\to 0` it converges to ``'beam'`` (uniform); at
  :math:`\\gamma \\to \\infty` it converges to the argmin over the beam.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

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

    # Back-propagation: canonical recursive V computation per
    # [Friston2021SophisticatedInference]. Uses `id(node)` as the dict key
    # because PolicyNode is frozen and hashable by value, but two distinct
    # nodes with the same action_seq would collide on the path-sum
    # interpretation; id() is the safe-by-construction choice.
    aggregator = _make_aggregator(cfg.branching_strategy, cfg.gamma)
    node_V = _backprop_V(level_nodes, cfg.horizon_D, cfg.discount, aggregator)

    # Rebuild root children with the back-propagated V stored as G_cum.
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


def _make_aggregator(
    strategy: str,
    gamma: float,
) -> Callable[[List[float]], float]:
    r"""Build the child-posterior expectation operator for back-propagation.

    Returns a callable that takes a list of child V values and returns the
    aggregated value :math:`\sum_{a'} q(a') V(a')`. The shape of
    :math:`q(a')` is determined by ``strategy``:

    - ``'beam'`` / ``'top_k'``: uniform — equivalent to ``mean(child_Vs)``.
    - ``'sophisticated'``: ``q(a') ∝ exp(-gamma · V(a'))`` (Friston 2021
      sophisticated inference). Uses ``torch.logsumexp`` for numerical
      stability.
    """
    if strategy in ('beam', 'top_k'):
        def aggregate_uniform(child_Vs: List[float]) -> float:
            if not child_Vs:
                return 0.0
            return sum(child_Vs) / len(child_Vs)
        return aggregate_uniform

    if strategy == 'sophisticated':
        def aggregate_sophisticated(child_Vs: List[float]) -> float:
            if not child_Vs:
                return 0.0
            # `q(c) ∝ exp(-gamma · V(c))`. Compute in log-space to avoid
            # overflow at large gamma; fp64 to preserve precision for the
            # gamma → ∞ argmin limit.
            vs = torch.tensor(child_Vs, dtype=torch.float64)
            log_q = -gamma * vs
            log_q = log_q - torch.logsumexp(log_q, dim=0)
            q = log_q.exp()
            return float((q * vs).sum().item())
        return aggregate_sophisticated

    raise ValueError(
        f"_make_aggregator: unknown branching_strategy={strategy!r}. "
        "Accepted: 'beam', 'top_k', 'sophisticated'."
    )


def _backprop_V(
    level_nodes: List[List[PolicyNode]],
    horizon_D: int,
    discount: float,
    aggregator: Callable[[List[float]], float],
) -> Dict[int, float]:
    r"""Compute :math:`V` bottom-up over the policy tree.

    Leaves at depth ``horizon_D`` contribute ``V = components.G_local``.
    Internal nodes at depth ``d < horizon_D`` compute
    ``V = components.G_local + discount · aggregator([V(child) for child in beam])``.
    The root (depth 0) has no ``components`` (no action taken yet); its
    own V is not consumed downstream — only ``V(root_children)`` matters
    for the policy posterior.

    Args:
        level_nodes: nested list with ``level_nodes[d]`` the nodes at depth d.
        horizon_D: planning horizon (depth of leaves).
        discount: geometric discount factor :math:`\gamma_{\text{disc}}`.
        aggregator: pure function ``List[float] -> float`` that computes
            the child-posterior-weighted expectation.

    Returns:
        Dict mapping ``id(node)`` to its back-propagated V value, for
        every node at depths 0 through ``horizon_D``.
    """
    V: Dict[int, float] = {}

    # Leaves.
    for leaf in level_nodes[horizon_D]:
        leaf_local = (
            leaf.components.G_local if leaf.components is not None else leaf.G_cum
        )
        V[id(leaf)] = leaf_local

    # Build child-list per parent for the recursion.
    children_of: Dict[int, List[PolicyNode]] = {}
    for level_idx in range(1, horizon_D + 1):
        for node in level_nodes[level_idx]:
            parent_id = id(node.parent)
            children_of.setdefault(parent_id, []).append(node)

    # Walk depth (D-1) up to 0.
    for depth_back in range(horizon_D - 1, -1, -1):
        for node in level_nodes[depth_back]:
            children = children_of.get(id(node), [])
            if children:
                child_Vs = [V[id(c)] for c in children]
                aggregated = aggregator(child_Vs)
                local = (
                    node.components.G_local if node.components is not None else 0.0
                )
                V[id(node)] = local + discount * aggregated
            else:
                # Defensive: every non-leaf should have children by construction
                # of the tree. Fall back to G_local (or 0 at the root).
                V[id(node)] = (
                    node.components.G_local if node.components is not None else 0.0
                )
    return V
