"""
AIFGenerator: canonical Expected Free Energy generation for a VFEModel.

Wraps a trained ``VFEModel`` by composition and emits token sequences by
softminning the policy posterior :math:`q(\\pi) \\propto \\exp(-\\gamma G(\\pi))`
over horizon-D futures. At ``horizon_D=1`` the generator reduces bitwise
to the existing single-step path in ``transformer/vfe/efe.py``; at
``horizon_D > 1`` it runs beam-search tree expansion.

Phase 1 supports ``horizon_D=1`` only (the reduction anchor). Phase 2
will extend `_expand_tree` to multi-step beam search. Phase 3 will add
the [Friston2021SophisticatedInference] recursive form.

Law 1 (E-step blindness) is structurally preserved: the generator only
calls ``model.forward_with_beliefs`` which has no ``targets`` parameter.
The wrapped ``VFEModel`` never sees a token it has not committed to or
been prompted with.
"""

from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

import torch
import torch.nn.functional as F

from transformer.aif.belief_cache import BeliefStateCache
from transformer.aif.efe_score import compute_G_at_node
from transformer.aif.policy import PolicyNode
from transformer.aif.preferences import Preference, build_preference
from transformer.aif.tree_search import pick_top_candidates, sophisticated_expand

if TYPE_CHECKING:
    from transformer.aif.config import AIFConfig
    from transformer.vfe.model import VFEModel


class AIFGenerator:
    r"""Canonical active-inference generator over a trained `VFEModel`.

    Args:
        model: trained ``VFEModel``. The generator never modifies its
            parameters.
        cfg: ``AIFConfig`` controlling planning horizon, branching,
            EFE weights, and preferences.
        preference: optional pre-built ``Preference`` instance. If
            ``None``, one is built from ``cfg.preference_type`` via
            ``preferences.build_preference``.
    """

    def __init__(
        self,
        model: 'VFEModel',
        cfg: 'AIFConfig',
        preference: Optional[Preference] = None,
    ) -> None:
        cfg.validate_against_model(model.cfg)
        self.model = model
        self.cfg = cfg
        if preference is None:
            preference = build_preference(
                preference_type=cfg.preference_type,
                preference_path=cfg.preference_path,
                low_entropy_beta=cfg.low_entropy_beta,
            )
        self.preference = preference
        self.cache = BeliefStateCache(max_entries=cfg.belief_cache_max_entries)

    @torch.no_grad()
    def _score_root_children(
        self,
        context_ids: torch.Tensor,
    ) -> List[PolicyNode]:
        r"""Build and score the root's children at the current context.

        Returns a list of ``beam_width`` scored ``PolicyNode`` children.
        Each child carries its own ``EFEComponents`` and its ``G_cum``
        equals its local G (depth-1 path; multi-step aggregation enters
        in Phase 2).
        """
        # Get the model's predictive at the current context to choose candidates.
        logits, _ = self.model.forward_with_beliefs(context_ids)
        last_probs = F.softmax(
            logits[:, -1, :] / max(self.cfg.decode_tau, 1e-12),
            dim=-1,
        ).squeeze(0)  # (V,)
        candidates = pick_top_candidates(last_probs, self.cfg.beam_width)

        root = PolicyNode(action_seq=(), depth=0, parent=None)
        children: List[PolicyNode] = []
        for action in candidates:
            components, beliefs = compute_G_at_node(
                context_ids=context_ids,
                candidate_action=action,
                model=self.model,
                preference=self.preference,
                cfg=self.cfg,
            )
            # Cache the converged belief under the action-prefix key. Phase 2
            # reads this when expanding deeper into the tree.
            child = root.child(
                action=action,
                components=components,
                G_cum=components.G_local,
            )
            self.cache.put(child.belief_cache_key, beliefs)
            children.append(child)

        return children

    @torch.no_grad()
    def _commit_action(
        self,
        children: List[PolicyNode],
    ) -> int:
        r"""Sample or argmin the committed action from the root posterior.

        ``q(a) = softmax(-gamma * G_cum)`` at the root level. Sampling
        strategy is controlled by ``cfg.sampling_strategy``.
        """
        g_values = torch.tensor([c.G_cum for c in children], dtype=torch.float32)
        if self.cfg.sampling_strategy == 'argmin':
            idx = int(torch.argmin(g_values).item())
        else:
            log_q = -self.cfg.gamma * g_values / max(self.cfg.sampling_temperature, 1e-12)
            q = F.softmax(log_q, dim=-1)
            idx = int(torch.multinomial(q, num_samples=1).item())
        return children[idx].action_seq[0]

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 50,
    ) -> torch.Tensor:
        r"""Generate ``max_new_tokens`` tokens by EFE-weighted policy sampling.

        At ``horizon_D=1`` this is the depth-1 reduction anchor: for each
        new token, score the top-`beam_width` candidates by `G`, softmin
        over them, sample (or argmin), commit, and recurse.

        Args:
            prompt_ids: ``(1, N_prompt)`` or ``(N_prompt,)`` prompt token IDs.
            max_new_tokens: number of tokens to emit.

        Returns:
            ``(1, N_prompt + max_new_tokens)`` token IDs (truncated to
            ``model.cfg.max_seq_len`` per token if the cumulative length
            would overflow).
        """
        if self.cfg.horizon_D > 1:
            raise NotImplementedError(
                "AIFGenerator at horizon_D > 1 is Phase 2 of the build-out "
                "(beam tree search). Phase 1 ships the depth-1 reduction "
                "anchor. See "
                "docs/plans/2026-05-19-aif-transformer-buildout/06_plan.md §6."
            )

        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        ids = prompt_ids
        max_len = self.model.cfg.max_seq_len

        for _ in range(max_new_tokens):
            ids_cond = ids if ids.shape[1] <= max_len else ids[:, -max_len:]
            children = self._score_root_children(ids_cond)
            action = self._commit_action(children)
            ids = torch.cat(
                [ids, torch.tensor([[action]], device=ids.device)], dim=1,
            )
            # Per-commit cache pruning: depth-1 children become stale once
            # an action is committed. With horizon_D=1 the cache is always
            # at depth 1, so we evict everything at depth < 1 (no-op since
            # nothing is shallower than the cached entries) and clear when
            # the cache grows past the LRU cap (already handled by `put`).

        return ids
