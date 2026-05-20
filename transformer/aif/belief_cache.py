"""
BeliefStateCache: prefix-keyed cache of converged BeliefStates.

The AIF tree search re-runs the full E-step at each node (per
`02_codebase_audit.md` §5: incremental belief updates would introduce new
transport code paths and risk gauge-equivariance bugs). The cost of this
re-encoding is offset by sharing the encoded belief across sibling
policies whose common ancestor coincides with the cached prefix.

The cache is a simple dict with two eviction mechanisms:

- **Per-commit pruning**: after the generator commits a token, all cached
  entries at depths below the committed depth are pruned because they
  can never be revisited.
- **LRU**: when the cache grows past `max_entries`, the oldest entry is
  evicted. Default cap is 4096 entries; at K=20, N=128 each snapshot
  is roughly 125 KB so the default cap is ~500 MB — well above the
  72 MB working-set budget at the recommended demo preset (D=2, b=4).

Cache keys are tuples of token IDs (the action sequence from the root).
Two policies that share a prefix share the cached belief at that prefix.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Iterator, Optional, Tuple

from transformer.core.types import BeliefState


class BeliefStateCache:
    r"""Prefix-keyed cache of converged ``BeliefState`` tuples.

    Args:
        max_entries: LRU cap. Default 4096 entries.
    """

    def __init__(self, max_entries: int = 4096) -> None:
        if max_entries < 1:
            raise ValueError(
                f"BeliefStateCache.max_entries must be >= 1 (got {max_entries})."
            )
        self.max_entries = max_entries
        self._store: 'OrderedDict[Tuple[int, ...], BeliefState]' = OrderedDict()
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, key: Tuple[int, ...]) -> bool:
        return key in self._store

    def get(self, key: Tuple[int, ...]) -> Optional[BeliefState]:
        r"""Look up the belief at `key`. Updates the LRU ordering on hit.

        Returns ``None`` on miss. Increments `hits` / `misses` for the
        cache-rate diagnostic.
        """
        beliefs = self._store.get(key)
        if beliefs is None:
            self.misses += 1
            return None
        self._store.move_to_end(key)
        self.hits += 1
        return beliefs

    def put(self, key: Tuple[int, ...], beliefs: BeliefState) -> None:
        r"""Store `beliefs` under `key`. Evicts oldest entry past the cap.

        Detaches every tensor before storing so the cache holds graph-
        independent snapshots; the tree search never builds an autograd
        graph through cached beliefs.
        """
        beliefs_detached = BeliefState(
            mu=beliefs.mu.detach(),
            sigma=beliefs.sigma.detach(),
            phi=beliefs.phi.detach(),
            omega=(
                [(o.detach(), oi.detach()) for (o, oi) in beliefs.omega]
                if beliefs.omega is not None else None
            ),
        )
        self._store[key] = beliefs_detached
        self._store.move_to_end(key)
        while len(self._store) > self.max_entries:
            self._store.popitem(last=False)

    def evict_below_depth(self, min_depth: int) -> int:
        r"""Drop every cached entry whose key has length < `min_depth`.

        Called after the generator commits a token: shorter prefixes can
        never be revisited because the tree has moved forward. Returns
        the number of entries evicted (for the cache-rate diagnostic).
        """
        to_evict = [k for k in self._store if len(k) < min_depth]
        for k in to_evict:
            del self._store[k]
        return len(to_evict)

    def commit_action(self, action: int) -> int:
        r"""Re-key the cache after the agent commits to ``action``.

        Entries whose key starts with ``(action,)`` are re-keyed by stripping
        the leading element — their belief state is unchanged because
        ``[old_prompt, action, ...]`` is the same context as
        ``[new_prompt, ...]`` where ``new_prompt = old_prompt ++ (action,)``.
        Entries whose key starts with a different first action are evicted
        because the subtree is no longer reachable from the new root.

        Truncation safety: when ``len(old_prompt) + len(key)`` exceeds
        ``max_seq_len`` the model's forward pass truncates to the last
        ``max_seq_len`` tokens; the cached belief reflects that truncation.
        After commit, ``[new_prompt, key[1:]]`` truncates to the same
        ``max_seq_len`` tokens, so the cached belief remains valid.

        Returns:
            Number of entries kept after re-keying (for the
            cache-hit-rate diagnostic).
        """
        new_store: 'OrderedDict[Tuple[int, ...], BeliefState]' = OrderedDict()
        for key, beliefs in self._store.items():
            if len(key) >= 1 and key[0] == action:
                new_store[key[1:]] = beliefs
        self._store = new_store
        return len(new_store)

    def clear(self) -> None:
        self._store.clear()
        self.hits = 0
        self.misses = 0

    def keys(self) -> Iterator[Tuple[int, ...]]:
        return iter(self._store.keys())
