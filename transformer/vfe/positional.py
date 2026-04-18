"""
VFEPositionalEncoding: position enters as gauge composition, not additive features.

Position is a Lie-algebra element p_i composed with the token gauge frame via BCH:
    phi_i^(0) = BCH(phi_token, p_i)

For bch_order=1 (abelian approximation): phi + p (simple addition).
For bch_order>=2: full BCH series via lie_compose_bch_general_torch.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from transformer.vfe.config import VFEConfig


class VFEPositionalEncoding(nn.Module):
    r"""Positional encoding via Lie-algebra gauge composition.

    Learnable parameters:
        - ``pos_phi`` :math:`(N_{\max}, n_{\text{gen}})` — positional Lie algebra elements.

    At forward time, composes token phi with positional phi:

    .. math::
        \phi_i^{(0)} = \mathrm{BCH}(\phi_i^{\text{token}},\, p_i)

    Args:
        cfg: VFEConfig with max_seq_len, bch_order.
        n_gen: Number of Lie algebra generators.
        generators: ``(n_gen, K, K)`` Lie algebra generators (needed for BCH order >= 2).
    """

    def __init__(
        self,
        cfg: 'VFEConfig',
        n_gen: int,
        generators: torch.Tensor,
    ) -> None:
        super().__init__()
        self.bch_order = cfg.bch_order
        self.max_seq_len = cfg.max_seq_len

        self.pos_phi = nn.Parameter(
            torch.randn(cfg.max_seq_len, n_gen) * 0.01
        )
        self.register_buffer('generators', generators)

    def forward(self, phi: torch.Tensor, seq_len: int) -> torch.Tensor:
        r"""Compose token phi with positional phi.

        Args:
            phi: ``(B, N, n_gen)`` token gauge frames from PriorBank.encode.
            seq_len: Sequence length N (must be <= max_seq_len).

        Returns:
            ``(B, N, n_gen)`` composed gauge frames.
        """
        pos = self.pos_phi[:seq_len]  # (N, n_gen)

        if self.bch_order <= 1:
            # First-order BCH: simple addition (exact for abelian groups)
            return phi + pos.unsqueeze(0)

        # Higher-order BCH
        try:
            from math_utils.generators import lie_compose_bch_general_torch
        except ImportError:
            # Fallback to additive if BCH not available
            if not getattr(self, '_bch_fallback_warned', False):
                import warnings
                warnings.warn(
                    f"bch_order={self.bch_order} requested but "
                    "math_utils.generators.lie_compose_bch_general_torch is "
                    "unavailable; falling back to first-order additive composition. "
                    "Non-abelian generators (e.g. GL(K), SO(N>2)) will lose BCH "
                    "correction terms.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._bch_fallback_warned = True
            return phi + pos.unsqueeze(0)

        return lie_compose_bch_general_torch(
            phi, pos.unsqueeze(0).expand_as(phi),
            self.generators, order=self.bch_order,
        )
