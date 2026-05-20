"""
AIFGenerator: canonical Expected Free Energy generation for a VFEModel.

Wraps a trained ``VFEModel`` by composition and emits token sequences by
softminning the policy posterior :math:`q(\\pi) \\propto \\exp(-\\gamma G(\\pi))`
over horizon-D futures. Calls :func:`tree_search.beam_expand` for every
``horizon_D >= 1``:

- ``horizon_D = 1``: the beam expansion runs one level and returns the
  root's children with ``G_cum = G_local`` (no back-propagation). Matches
  the depth-1 reduction anchor that ``transformer/vfe/efe.py`` provided
  before this build-out.
- ``horizon_D > 1``: full per-branch beam tree with mean back-propagation
  of leaf ``G_cum`` to root children. Per-commit cache re-keying enables
  prefix sharing across emit steps.

Phase 3 will switch to ``branching_strategy='sophisticated'`` (the Friston
2021 recursive form); ``AIFConfig.__post_init__`` already rejects that
strategy so callers fail fast.

Law 1 (E-step blindness) is structurally preserved: the generator only
calls ``model.forward_with_beliefs`` and ``model.prior_bank.decode``, both
of which have no ``targets`` parameter. The wrapped ``VFEModel`` never
sees a token it has not committed to or been prompted with.
"""

from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

import torch
import torch.nn.functional as F

from transformer.aif.belief_cache import BeliefStateCache
from transformer.aif.policy import PolicyNode
from transformer.aif.preferences import Preference, build_preference
from transformer.aif.tree_search import beam_expand

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

        At each step:

        1. Run :func:`beam_expand` to build a depth-``cfg.horizon_D`` tree
           and back-propagate leaf G_cum to root children.
        2. Softmin :math:`q(a) \propto \exp(-\gamma G_{\text{cum}}(a))` over
           the root children; pick the committed action.
        3. Append the action to ``ids``; re-key the cache via
           :meth:`BeliefStateCache.commit_action` so the subtree under the
           committed action survives for cross-commit prefix sharing.

        Args:
            prompt_ids: ``(1, N_prompt)`` or ``(N_prompt,)`` prompt token IDs.
            max_new_tokens: number of tokens to emit.

        Returns:
            ``(1, N_prompt + max_new_tokens)`` token IDs (truncated to
            ``model.cfg.max_seq_len`` per-step at the beam-expansion side
            if the cumulative length would overflow).
        """
        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        ids = prompt_ids
        max_len = self.model.cfg.max_seq_len

        for _ in range(max_new_tokens):
            ids_cond = ids if ids.shape[1] <= max_len else ids[:, -max_len:]
            children = beam_expand(
                prompt_ids=ids_cond,
                model=self.model,
                preference=self.preference,
                cache=self.cache,
                cfg=self.cfg,
            )
            action = self._commit_action(children)
            ids = torch.cat(
                [ids, torch.tensor([[action]], device=ids.device)], dim=1,
            )
            # Re-key the cache so the subtree under the committed action
            # survives for the next step. Entries under non-committed
            # siblings are evicted (no longer reachable).
            self.cache.commit_action(action)

        return ids
