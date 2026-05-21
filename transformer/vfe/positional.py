"""
VFEPositionalEncoding: position enters as gauge composition, not additive features.

Position is a Lie-algebra element p_i composed with the token gauge frame via BCH:
    phi_i^(0) = BCH(phi_token, p_i)

For bch_order=1 (abelian approximation): phi + p (simple addition).
For bch_order>=2: full BCH series via lie_compose_bch_general_torch.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional, Sequence

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from transformer.vfe.config import VFEConfig


class VFEPositionalEncoding(nn.Module):
    r"""Positional encoding via Lie-algebra gauge composition.

    Learnable parameters:
        - ``pos_phi_free`` positional Lie algebra coordinates.
          When ``cfg.phi_project_slk`` is True, stored in sl(K) free coords of
          shape :math:`(N_{\max}, n_{\text{gen}} - H)` and expanded at forward
          time via an orthonormal basis of the traceless subalgebra, so the
          positional element is structurally traceless. Otherwise stored in
          full :math:`(N_{\max}, n_{\text{gen}})` coords.

    At forward time, composes token phi with positional phi:

    .. math::
        \phi_i^{(0)} = \mathrm{BCH}(\phi_i^{\text{token}},\, p_i)

    Since :math:`\operatorname{tr}([X,Y])=0`, the BCH series satisfies
    :math:`\operatorname{tr}\mathrm{BCH}(X,Y)=\operatorname{tr}(X)+\operatorname{tr}(Y)`,
    so making both inputs traceless preserves det(Ω)=1 to all BCH orders.

    Args:
        cfg: VFEConfig with max_seq_len, bch_order, phi_project_slk.
        n_gen: Number of Lie algebra generators.
        generators: ``(n_gen, K, K)`` Lie algebra generators (needed for BCH order >= 2).
        irrep_dims: Block dimensions for the sl(K) re-parameterization. Required
            when ``cfg.phi_project_slk`` is True.
    """

    def __init__(
        self,
        cfg: 'VFEConfig',
        n_gen: int,
        generators: torch.Tensor,
        irrep_dims: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        self.bch_order = cfg.bch_order
        self.max_seq_len = cfg.max_seq_len
        self.phi_project_slk = bool(getattr(cfg, 'phi_project_slk', False))
        self.register_buffer('generators', generators)

        # Resolve the BCH compositor at construction time. Previously this was
        # resolved on the first forward and silently fell back to first-order
        # additive composition when math_utils was missing — non-abelian
        # generators would then lose BCH correction terms for the entire run.
        # Failing fast at init surfaces the misconfiguration before any
        # training step burns.
        self._bch_compose = None
        if self.bch_order > 1:
            try:
                from math_utils.generators import lie_compose_bch_general_torch
            except ImportError as exc:
                raise ImportError(
                    f"VFEPositionalEncoding: bch_order={self.bch_order} requires "
                    "math_utils.generators.lie_compose_bch_general_torch but the "
                    "import failed. Either install math_utils or set bch_order=1 "
                    "(first-order additive composition, exact only for abelian "
                    "generators)."
                ) from exc
            self._bch_compose = lie_compose_bch_general_torch

        if self.phi_project_slk:
            if irrep_dims is None:
                raise ValueError(
                    "VFEPositionalEncoding: phi_project_slk=True requires irrep_dims "
                    "to build the sl(K) basis for the positional parameter."
                )
            from transformer.core.vfe_utils import build_slk_basis
            _, P = build_slk_basis(generators, list(irrep_dims))  # (n_gen, n_gen - H)
            self.register_buffer('pos_phi_basis', P)
            # Scale positional gauge init by cfg.phi_scale so the user-facing
            # phi_scale knob actually zeroes the initial gauge frame when set
            # to 0. A prior hardcoded `* 0.01` silently injected a nonzero
            # positional phi regardless of phi_scale, contaminating any
            # phi_scale=0 sweep with a residual 0.01-magnitude rotation.
            self.pos_phi_free = nn.Parameter(
                torch.randn(cfg.max_seq_len, P.shape[-1]) * cfg.phi_scale
            )
        else:
            self.pos_phi_basis = None
            self.pos_phi_free = nn.Parameter(
                torch.randn(cfg.max_seq_len, n_gen) * cfg.phi_scale
            )

    def forward(self, phi: torch.Tensor, seq_len: int) -> torch.Tensor:
        r"""Compose token phi with positional phi.

        Args:
            phi: ``(B, N, n_gen)`` token gauge frames from PriorBank.encode.
            seq_len: Sequence length N (must be <= max_seq_len).

        Returns:
            ``(B, N, n_gen)`` composed gauge frames.
        """
        if self.pos_phi_basis is not None:
            # sl(K) free coords → full n_gen via orthonormal basis of traceless subalgebra.
            pos = self.pos_phi_free[:seq_len] @ self.pos_phi_basis.T  # (N, n_gen)
        else:
            pos = self.pos_phi_free[:seq_len]  # (N, n_gen)

        if self.bch_order <= 1:
            # First-order BCH: simple addition (exact for abelian groups)
            return phi + pos.unsqueeze(0)

        return self._bch_compose(
            phi, pos.unsqueeze(0).expand_as(phi),
            self.generators, order=self.bch_order,
        )
