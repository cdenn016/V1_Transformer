"""
PolicyNode and EFEComponents: data structures for the AIF policy tree.

A policy is an ordered sequence of token-emission actions
:math:`\\pi = (a_{T+1}, \\dots, a_{T+D})`. The tree is built by expanding
each node by `beam_width` child actions; each child is a longer policy
:math:`\\pi' = \\pi \\circ a`. The root is the empty policy (action_seq is
empty, depth is 0).

PolicyNode is immutable; the search constructs new nodes rather than
mutating existing ones. Parent pointers support back-propagation of
cumulative G when the tree is scored leaf-to-root.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass(frozen=True)
class EFEComponents:
    r"""Decomposition of `G_d` at a single tree node.

    Fields are scalar floats (or 0-dim torch tensors), one per term in the
    canonical Form-3 BALD decomposition:

    .. math::
        G_d = \mathrm{pragmatic}_d + \mathrm{ambiguity}_d - \mathrm{epistemic}_d

    where pragmatic = :math:`E_{q(o|\pi)}[-\log p^*(o|C)]`, ambiguity =
    :math:`E_{q(s|\pi)}[H[p(o|s)]]` (mean predictive entropy under sampled
    states), and epistemic = :math:`I_{q(s,o|\pi)}(s; o)` (BALD MI).
    """

    pragmatic: float
    ambiguity: float
    epistemic: float
    G_local: float
    """``pragmatic + ambiguity - weighted_epistemic`` at this node."""


@dataclass(frozen=True)
class PolicyNode:
    r"""Immutable node in the policy tree.

    A node represents the policy obtained by following `action_seq` from
    the root. The root has an empty `action_seq` and `depth=0`. Each child
    extends the parent's `action_seq` by exactly one action token.

    The `G_cum` field holds the cumulative G over the path from the root,
    used for the policy posterior at the root level: at horizon D the
    root's children's G_cum values feed into
    ``q(a) = softmax(-gamma * G_cum)``.

    `belief_cache_key` is a hashable identifier for the belief state at
    this node — typically the action sequence as a tuple. Concrete cache
    implementations key off this value.
    """

    action_seq: Tuple[int, ...]
    """Tuple of token IDs from the root to this node. Empty at root."""

    depth: int
    """Distance from root. Root has depth 0. A node at depth d has
    action_seq of length d."""

    parent: Optional['PolicyNode'] = None
    """Parent node, or None at the root."""

    components: Optional[EFEComponents] = None
    """Per-node EFE decomposition. Populated by `compute_G_at_node`."""

    G_cum: float = 0.0
    """Cumulative G over the path from root to this node (with any
    configured `discount` applied to the root contribution last)."""

    @property
    def is_root(self) -> bool:
        return self.parent is None

    @property
    def belief_cache_key(self) -> Tuple[int, ...]:
        r"""Key for the belief-state cache. Equal to `action_seq` so the
        cache shares prefix-encoded beliefs across sibling policies whose
        common ancestor coincides with the cached key."""
        return self.action_seq

    def child(
        self,
        action: int,
        components: Optional[EFEComponents] = None,
        G_cum: float = 0.0,
    ) -> 'PolicyNode':
        r"""Construct a child node extending this policy by one action."""
        return PolicyNode(
            action_seq=self.action_seq + (action,),
            depth=self.depth + 1,
            parent=self,
            components=components,
            G_cum=G_cum,
        )
