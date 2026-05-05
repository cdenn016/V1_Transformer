"""
Gauge Connection: Edge-local Lie algebra elements for non-flat transport.
=========================================================================

Parameterizes the "connection" δ_ij on the token graph so that the transport
operator becomes:

    Ω_ij = exp(φ_i · G) · exp(δ_ij · G) · exp(-φ_j · G)

When δ_ij = 0, this is identically the flat transport Ω_ij = exp(φ_i)·exp(-φ_j).

The connection controls holonomy:
    H_ijk = exp(φ_i) · C_ijk · exp(-φ_i)
where C_ijk = exp(δ_ij·G) · exp(δ_jk·G) · exp(δ_ki·G) is the connection holonomy.

Two parameterizations:
    - Bilinear:  δ_ij^a = μ_i^T W^a μ_j  (parameter-efficient, default)
    - MLP:       δ_ij = MLP([μ_i; μ_j])   (more expressive, higher memory)

Both are zero-initialized so the model starts in the flat regime.

This module exposes two classes:
    - ``GaugeConnection``: a single per-head connection on one sub-fiber.
    - ``PerHeadGaugeConnection``: a container of per-head connections that
      slices ``μ`` per irrep block, dispatches each slice to its own
      ``GaugeConnection``, and concatenates the per-head δ outputs into a
      single ``(B, N, N, n_gen_total)`` tensor.  This preserves the
      block-diagonal structure of the transport operator: head-h's δ
      coefficients are a function of head-h's μ features only, matching the
      block-diagonal generator support.

Generator partitioning: ``partition_generators_by_block`` splits a global
``(n_gen, K, K)`` generator tensor into per-head tensors of shape
``(n_gen_h, d_h, d_h)`` based on Frobenius-mass localization in the
corresponding irrep block.
"""

from typing import List

import math
import torch
import torch.nn as nn


class GaugeConnection(nn.Module):
    """Edge-local connection producing Lie algebra elements for non-flat transport.

    For each edge (i, j), produces δ_ij ∈ ℝ^{n_gen} such that:
        Ω_ij = exp(φ_i · G) · exp(δ_ij · G) · exp(-φ_j · G)

    The connection is zero-initialized: δ_ij = 0 at init → flat transport.
    The model learns to deviate from flatness only where the data warrants it.

    Args:
        d_head: Per-head dimension (belief mean dimension for this head).
        n_gen: Number of Lie algebra generators for this head's gauge group.
        connection_type: 'bilinear' or 'mlp'.
        hidden_dim: Hidden dimension for MLP connection (ignored for bilinear).
        antisymmetrize: If True (default), enforce δ_ij = -δ_ji via
                        W → (W - W^T)/2.  This is the natural condition for
                        a connection on an undirected token graph and the
                        prerequisite for the holonomy penalty to measure a
                        well-defined curvature-like quantity.  A
                        non-antisymmetric connection is a torsion connection
                        and violates the cocycle interpretation of the
                        transport; only override to False for a deliberate
                        torsion-bearing ablation.  Only applies to bilinear.
    """

    def __init__(
        self,
        d_head: int,
        n_gen: int,
        connection_type: str = 'bilinear',
        hidden_dim: int = 64,
        antisymmetrize: bool = True,
        init_scale: float = 0.0,
    ):
        super().__init__()
        self.d_head = d_head
        self.n_gen = n_gen
        self.connection_type = connection_type
        self.antisymmetrize = antisymmetrize

        if connection_type == 'bilinear':
            # δ_ij^a = μ_i^T W^a μ_j — one bilinear form per generator
            # Parameters: n_gen × d_head × d_head
            #
            # init_scale=0 → zero init (flat saddle point — no gradient signal!)
            # init_scale>0 → small random init breaks the flat saddle point so
            # the optimizer can discover useful curvature.  Recommended: 0.01.
            #
            # TODO(holonomy-W-regularization): W has no weight decay, no
            # spectral clamp, and no penalty on its norm.  transport_ops.py
            # clamps ‖scaled_delta‖ ≤ 5.0 at the forward pass, but that does
            # NOT bound W itself — W can drift upward indefinitely while the
            # forward clamp masks it.  The holonomy diagnostic
            # (publication_metrics.py:_extract_exp_delta) recomputes δ
            # without the training-time clamp and exposes the drift as
            # delta_max_spec values in the hundreds, which then overflow
            # exp(δ) in float32.  Fix options to consider:
            #   (a) add weight_decay to W via the optimizer param-group,
            #   (b) spectral clamp on W per step (project ‖W‖_2 ≤ W_max),
            #   (c) wire up the existing holonomy_penalty hook so curvature
            #       contributes to the loss and back-pressures W.
            # Picking the right option requires deciding whether W
            # divergence is a symptom (E-step pathology) or a cause
            # (insufficient regularization).  See 2026-05-05_edits.md.
            W = torch.zeros(n_gen, d_head, d_head)
            if init_scale > 0:
                W = W + init_scale * torch.randn_like(W) / math.sqrt(d_head)
            self.W = nn.Parameter(W)
        elif connection_type == 'mlp':
            self.net = nn.Sequential(
                nn.Linear(2 * d_head, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, n_gen),
            )
            # Zero-init output layer → flat at initialization
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)
        else:
            raise ValueError(f"Unknown connection_type: {connection_type}. Use 'bilinear' or 'mlp'.")

    def forward(self, mu_i: torch.Tensor, mu_j: torch.Tensor) -> torch.Tensor:
        """Compute edge-local connection δ_ij.

        Args:
            mu_i: (B, N, d_head) query belief means.
            mu_j: (B, N, d_head) key belief means.

        Returns:
            delta: (B, N, N, n_gen) Lie algebra coefficients per edge.
        """
        if self.connection_type == 'bilinear':
            W = self.W
            if self.antisymmetrize:
                # W → (W - W^T) / 2 ensures δ_ij = -δ_ji
                W = (W - W.transpose(-1, -2)) / 2

            # δ_ij^a = μ_i^T W^a μ_j
            # mu_i: (B, N, d) — broadcast over j
            # W:    (n_gen, d, d)
            # mu_j: (B, N, d) — broadcast over i
            # Result: (B, N, N, n_gen)
            delta = torch.einsum('bid,adg,bjg->bija', mu_i, W, mu_j)
        elif self.connection_type == 'mlp':
            B, N, D = mu_i.shape
            # Expand to all pairs: (B, N, N, 2D)
            mu_i_exp = mu_i.unsqueeze(2).expand(-1, -1, N, -1)
            mu_j_exp = mu_j.unsqueeze(1).expand(-1, N, -1, -1)
            pair = torch.cat([mu_i_exp, mu_j_exp], dim=-1)
            delta = self.net(pair)  # (B, N, N, n_gen)

        return delta

    def extra_repr(self) -> str:
        return (
            f"d_head={self.d_head}, n_gen={self.n_gen}, "
            f"type={self.connection_type}, antisym={self.antisymmetrize}"
        )


def partition_generators_by_block(
    generators: torch.Tensor,      # (n_gen, K, K)
    irrep_dims: List[int],         # [d_1, d_2, ..., d_H] summing to K
    localization_threshold: float = 0.999,
) -> List[torch.Tensor]:
    """Split a global block-diagonal generator tensor into per-head tensors.

    For block-diagonal gauge groups (GL(K) as H copies of GL(d_h), SO(K) as
    H copies of SO(d_h), mixed irrep_spec variants), each generator T_a has
    Frobenius support localized in exactly one irrep block.  This helper
    partitions the ``(n_gen_total, K, K)`` generator stack into a list of
    per-head ``(n_gen_h, d_h, d_h)`` stacks.

    Args:
        generators: ``(n_gen, K, K)`` Lie algebra generators.
        irrep_dims: Per-head dimensions ``[d_1, ..., d_H]``; ``sum(d_h) = K``.
        localization_threshold: Minimum fraction of Frobenius mass a
            generator must have inside a candidate block before it is
            assigned there.  Default ``0.999`` — block-diagonal generators
            should have ≥99.9% of their mass in one block; the slack
            tolerates floating-point noise in constructed generators.

    Returns:
        List of ``H`` tensors; entry ``h`` has shape ``(n_gen_h, d_h, d_h)``
        and contains the ``d_h × d_h`` diagonal block of each generator
        that localizes to head ``h``.

    Raises:
        ValueError: If ``sum(irrep_dims) != generators.shape[-1]``, or if
            any generator fails to localize in a single block above the
            threshold (indicates non-block-diagonal support, which this
            helper does not handle).
    """
    n_gen, K, K2 = generators.shape
    if K != K2:
        raise ValueError(f"generators must be square; got shape {generators.shape}")
    if sum(irrep_dims) != K:
        raise ValueError(
            f"sum(irrep_dims)={sum(irrep_dims)} does not match K={K}"
        )

    # Precompute block boundaries and per-block masks.
    bounds = []
    start = 0
    for d_h in irrep_dims:
        bounds.append((start, start + d_h))
        start += d_h

    # Full-matrix Frobenius mass per generator, plus per-block masses.
    gens_sq = generators.pow(2)                       # (n_gen, K, K)
    total_mass = gens_sq.sum(dim=(-2, -1))            # (n_gen,)
    # Guard against all-zero generators (degenerate; assign to head 0 by fiat).
    total_mass_safe = total_mass.clamp(min=1e-30)

    per_head_blocks: List[List[torch.Tensor]] = [[] for _ in irrep_dims]
    for a in range(n_gen):
        G_a = generators[a]                            # (K, K)
        assigned = False
        for h, (s, e) in enumerate(bounds):
            block_mass = gens_sq[a, s:e, s:e].sum()
            if total_mass[a].item() < 1e-30:
                # Zero generator — assign to head 0 deterministically.
                per_head_blocks[0].append(torch.zeros(
                    irrep_dims[0], irrep_dims[0],
                    dtype=generators.dtype, device=generators.device,
                ))
                assigned = True
                break
            if (block_mass / total_mass_safe[a]).item() >= localization_threshold:
                per_head_blocks[h].append(G_a[s:e, s:e].clone())
                assigned = True
                break
        if not assigned:
            # Find the closest block for a diagnostic message.
            masses = [
                (gens_sq[a, s:e, s:e].sum() / total_mass_safe[a]).item()
                for s, e in bounds
            ]
            raise ValueError(
                f"generator {a} does not localize to any single irrep block "
                f"above {localization_threshold}. Per-block mass fractions: "
                f"{masses}. partition_generators_by_block requires "
                f"block-diagonal generator support."
            )

    # Stack each head's generators (may be empty for padding-only heads).
    return [
        torch.stack(blocks) if blocks
        else torch.empty(0, d_h, d_h, dtype=generators.dtype, device=generators.device)
        for blocks, d_h in zip(per_head_blocks, irrep_dims)
    ]


class PerHeadGaugeConnection(nn.Module):
    r"""Container of per-head :class:`GaugeConnection` instances.

    Slices the input ``μ`` tensor along the last dim according to
    ``irrep_dims``, dispatches each ``(B, N, d_h)`` slice to its own
    per-head ``GaugeConnection``, and concatenates the per-head
    ``(B, N, N, n_gen_h)`` δ outputs along the last axis into a single
    ``(B, N, N, n_gen_total)`` tensor.

    This matches the block-diagonal structure of the gauge group: head-h's
    connection is a function of head-h's μ features only, and produces
    coefficients for head-h's generators only.  Contrast with a global
    (full-fiber) connection whose bilinear form ``W: (n_gen_total, K, K)``
    would mix μ features across heads.

    Args:
        irrep_dims: Per-head dimensions ``[d_1, ..., d_H]`` summing to
            the full fiber dimension ``K``.
        per_head_generators: List of ``(n_gen_h, d_h, d_h)`` tensors
            produced by :func:`partition_generators_by_block`.  Only the
            length and per-head dimensions are used for sizing the
            bilinear forms / MLPs; the generator values themselves enter
            elsewhere (``compute_transport_operators``).
        connection_type: ``'bilinear'`` or ``'mlp'``.
        hidden_dim: Hidden dim for MLP connections.
        antisymmetrize: Apply ``W → (W - Wᵀ)/2`` per head (bilinear only).
        init_scale: Small random init on bilinear W to break the flat
            saddle point.  Default ``0.0`` (pure zero init).
    """

    def __init__(
        self,
        irrep_dims: List[int],
        per_head_generators: List[torch.Tensor],
        connection_type: str = 'bilinear',
        hidden_dim: int = 64,
        antisymmetrize: bool = True,
        init_scale: float = 0.0,
    ):
        super().__init__()
        if len(irrep_dims) != len(per_head_generators):
            raise ValueError(
                f"irrep_dims length {len(irrep_dims)} does not match "
                f"per_head_generators length {len(per_head_generators)}"
            )
        self.irrep_dims = list(irrep_dims)
        self.n_gen_per_head = [int(g.shape[0]) for g in per_head_generators]
        self.n_gen_total = sum(self.n_gen_per_head)

        self.heads = nn.ModuleList([
            GaugeConnection(
                d_head=d_h,
                n_gen=n_gen_h,
                connection_type=connection_type,
                hidden_dim=hidden_dim,
                antisymmetrize=antisymmetrize,
                init_scale=init_scale,
            )
            for d_h, n_gen_h in zip(self.irrep_dims, self.n_gen_per_head)
        ])

    def forward(self, mu_i: torch.Tensor, mu_j: torch.Tensor) -> torch.Tensor:
        """Compute per-head δ_ij and concatenate along the generator axis.

        Args:
            mu_i: ``(B, N, K)`` full-fiber query means.
            mu_j: ``(B, N, K)`` full-fiber key means.

        Returns:
            ``(B, N, N, n_gen_total)`` — per-head δ blocks concatenated.
        """
        delta_blocks = []
        start = 0
        for head, d_h in zip(self.heads, self.irrep_dims):
            end = start + d_h
            mu_i_h = mu_i[..., start:end]
            mu_j_h = mu_j[..., start:end]
            delta_h = head(mu_i_h, mu_j_h)               # (B, N, N, n_gen_h)
            delta_blocks.append(delta_h)
            start = end
        return torch.cat(delta_blocks, dim=-1)           # (B, N, N, n_gen_total)

    def extra_repr(self) -> str:
        return (
            f"n_heads={len(self.heads)}, irrep_dims={self.irrep_dims}, "
            f"n_gen_per_head={self.n_gen_per_head}, "
            f"n_gen_total={self.n_gen_total}"
        )
