from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch


@dataclass
class BeliefBundle:
    r"""Per-token belief geometry extracted from a VFE model.

    A flat, framework-agnostic container that the semantic-clustering pipeline
    consumes. All tensors are detached and CPU-resident. One row per token
    (either a token-in-context occurrence for ``source='contextual'`` or a token
    type for ``source='vocab'``).

    Fields
    ------
    mu : torch.Tensor
        ``(n, K)`` token means.
    sigma : torch.Tensor
        ``(n, K)`` diagonal variances when ``diagonal`` is True, else ``(n, K, K)``
        full covariances.
    phi : torch.Tensor
        ``(n, n_gen)`` Lie-algebra coefficients. The algebra element for a token
        is :math:`A = \sum_c \phi_c G_c` where :math:`G` is :attr:`generators`.
    token_ids : torch.Tensor
        ``(n,)`` integer token ids.
    token_strings : list[str] | None
        Decoded strings (length ``n``) when a tokenizer was available, else None.
    generators : torch.Tensor | None
        ``(n_gen, K, K)`` generator bank :math:`G`. The per-token group element is
        :math:`\Omega = \exp(\sum_c \phi_c G_c)`. Must be the EXACT bank the model
        uses for transport; None only when unavailable (Omega views are then skipped).
    irrep_dims : list[int]
        Per-head irrep dimensions giving the block structure of :math:`\Omega`
        (e.g. ``[10] * 20`` for 20 heads of dim 10).
    source : str
        ``'contextual'`` or ``'vocab'``.
    layer : str | int
        ``'final'`` or an integer layer index.
    diagonal : bool
        True iff ``sigma`` has shape ``(n, K)``.
    """

    mu: torch.Tensor
    sigma: torch.Tensor
    phi: torch.Tensor
    token_ids: torch.Tensor
    token_strings: Optional[list[str]]
    generators: Optional[torch.Tensor]
    irrep_dims: list[int]
    source: str
    layer: Union[str, int]
    diagonal: bool

    @property
    def n(self) -> int:
        """Number of tokens (rows)."""
        return int(self.mu.shape[0])

    @property
    def K(self) -> int:
        """Embedding dimension."""
        return int(self.mu.shape[1])
