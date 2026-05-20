"""
Tree-search policies for multi-step EFE.

Phase 1 ships:

- :func:`pick_top_candidates`: take the top-k candidate next actions from a
  predictive distribution. Used at every tree node to bound the branching
  factor at `beam_width`.
- :func:`beam_expand`: depth-D beam expansion. At each level the top
  ``beam_width`` policies (ranked by cumulative G so far) are expanded.

Phase 3 will add the [Friston2021SophisticatedInference] recursive form;
the stub is present so the dispatch in `AIFGenerator` has a single switch.
"""

from __future__ import annotations

from typing import List, TYPE_CHECKING

import torch

from transformer.aif.policy import PolicyNode

if TYPE_CHECKING:
    from transformer.aif.config import AIFConfig


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


def sophisticated_expand(*_args, **_kwargs):
    r"""Phase 3 stub for the [Friston2021SophisticatedInference] recursion.

    Raises ``NotImplementedError`` so any caller that selects
    ``branching_strategy='sophisticated'`` fails fast. ``AIFConfig.__post_init__``
    raises the same error earlier; this stub guards against a path that
    bypasses the config check.
    """
    raise NotImplementedError(
        "Sophisticated-inference recursion per [Friston2021SophisticatedInference] "
        "is Phase 3 of the build-out and not yet implemented. Use "
        "branching_strategy='beam' for the Phase 2 tree search."
    )
