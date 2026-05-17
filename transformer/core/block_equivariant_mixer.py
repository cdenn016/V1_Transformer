r"""Block-level gauge-equivariant head mixer.

Companion to ``IrrepMultiHeadAttention``'s in-attention mixer (see
``attention.py::_apply_equivariant_mixer``). Same math (Schur commutant of
the irrep decomposition), different location: this module runs in
``GaugeTransformerBlock.forward`` after the FFN's E-step output and before
the residual+norm, so the operator is available when the attention sublayer
is bypassed (``skip_attention=True``).

Math. For ``V = ⊕_t (V_t)^{⊕ n_t}`` with irrep type ``t`` of dim ``d_t`` and
multiplicity ``n_t``, the commutant ``Hom_G(V, V) = ⊕_t M_{n_t}(R)``
(Schur's lemma over the reals, no Hermitian/quaternionic twists in this
codebase). We parameterize one ``A_t ∈ R^{n_t × n_t}`` per irrep type and
assemble ``M = ⊕_t kron(A_t, I_{d_t})``. The mixer is applied symmetrically
to ``(μ, Σ)``:

    μ' = M·μ            (B, N, K)  @  (K, K)
    Σ' = M·Σ·Mᵀ         (sandwich product; matches gauge transport convention)

Identity init. Internally ``A_t`` is stored as ``I + Δ_t`` with ``Δ_t`` an
``nn.Parameter`` zero-initialised, so a freshly-constructed mixer is a
bitwise no-op on (μ, Σ) before any training step.

Gauge equivariance. ``M`` lies in the Schur commutant of the gauge
representation. Under a tied gauge ``h`` acting block-diagonally on each
irrep copy, ``M h = h M``, so ``μ → hμ`` ⇒ ``Mμ → hMμ`` (μ-equivariance)
and ``Σ → hΣhᵀ`` ⇒ ``MΣMᵀ → h(MΣMᵀ)hᵀ`` (Σ-equivariance). Under
independent per-head gauges (the default GLK-multi-head configuration
without ``cross_couplings``) the commutant collapses to per-head scalar
diagonals and off-diagonal entries of ``A_t`` break equivariance the same
way ``use_output_projection`` does. The block layer warns at construction
in that case (mirrors the in-attention mixer's policy at
``attention.py:1678``).
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class BlockEquivariantMixer(nn.Module):
    r"""Schur-commutant mixer applied post-FFN in ``GaugeTransformerBlock``.

    Parameters
    ----------
    irrep_spec
        ``[(label, multiplicity, dim), ...]`` — same format as
        ``BlockConfig.irrep_spec`` / ``IrrepMultiHeadAttention``.
    embed_dim
        Total ``K = Σ_t n_t · d_t``. Cross-checked against ``irrep_spec``.
    diagonal_covariance
        If True, ``forward`` expects ``sigma`` as ``(B, N, K)`` and returns
        a ``(B, N, K)`` tensor via the closed-form diagonal-of-sandwich
        ``σ'[m·d + k] = Σ_n A[m,n]² σ[n·d + k]`` (mathematically equal to
        ``diag(MΣMᵀ)`` without materializing the K×K full matrix).
    gauge_group
        ``'GLK'`` or ``'SO3'`` / ``'SON'``. Used only for the
        independent-gauge warning predicate at construction.
    """

    def __init__(
        self,
        irrep_spec: List[Tuple[str, int, int]],
        embed_dim: int,
        diagonal_covariance: bool,
        gauge_group: str = 'GLK',
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.diagonal_covariance = diagonal_covariance

        irrep_dims, irrep_labels = self._unpack_irrep_spec(irrep_spec, gauge_group, embed_dim)
        total = sum(irrep_dims)
        if total != embed_dim:
            raise ValueError(
                f"irrep_spec sums to {total}, must equal embed_dim={embed_dim}"
            )
        self.irrep_dims: List[int] = irrep_dims
        self.irrep_labels: List[str] = irrep_labels

        # Group heads by (collapsed_label, dim). For GLK multi-head we
        # collapse 'glk_head_i' → 'glk_fund' so all GL(d) heads cluster.
        groups: Dict[Tuple[str, int], List[int]] = defaultdict(list)
        for h, (lbl, d_h) in enumerate(zip(self.irrep_labels, self.irrep_dims)):
            key = 'glk_fund' if lbl.startswith('glk_head_') else lbl
            groups[(key, d_h)].append(h)
        self._mixer_groups: List[Tuple[str, int, List[int]]] = [
            (key, dim, heads) for (key, dim), heads in groups.items()
        ]

        # Identity init: store Δ_t separately from I so construction is bitwise no-op.
        self.mixer_delta = nn.ParameterList([
            nn.Parameter(torch.zeros(len(heads), len(heads)))
            for _, _, heads in self._mixer_groups
        ])

        # Precompute per-head start offsets for matrix assembly.
        offsets = [0]
        for d_h in self.irrep_dims:
            offsets.append(offsets[-1] + d_h)
        self._offsets: List[int] = offsets

        if gauge_group == 'GLK' and any(
            lbl.startswith('glk_head_') for lbl in self.irrep_labels
        ) and any(len(h) > 1 for _, _, h in self._mixer_groups):
            warnings.warn(
                "BlockEquivariantMixer constructed on GL(K) multi-head with "
                "independent per-head gauges (irrep_labels='glk_head_*'). The "
                "commutant collapses to per-head scalar diagonals — off-diagonal "
                "entries of A_t break gauge equivariance the same way "
                "use_output_projection does. To recover the full Schur commutant, "
                "use cross_couplings to tie gauges within super-blocks.",
                RuntimeWarning,
                stacklevel=2,
            )

    @staticmethod
    def _unpack_irrep_spec(
        irrep_spec: List[Tuple[str, int, int]],
        gauge_group: str,
        embed_dim: int,
    ) -> Tuple[List[int], List[str]]:
        r"""Expand irrep_spec → (irrep_dims, irrep_labels).

        Matches ``IrrepMultiHeadAttention.__init__`` conventions
        (attention.py:1432-1499) for the cases this module supports:
        GLK single-head, GLK multi-head, and SO(N) multi-irrep.
        """
        dims: List[int] = []
        labels: List[str] = []
        if gauge_group == 'GLK':
            if len(irrep_spec) == 1 and irrep_spec[0][0] == 'full':
                _, _, d = irrep_spec[0]
                return [d], ['full']
            # Multi-head GLK: [(label, n_heads, d_head)]
            label, n_heads, d_head = irrep_spec[0]
            return [d_head] * n_heads, [f'glk_head_{h}' for h in range(n_heads)]
        # SO(N) / SO(3): expand each (label, mult, dim)
        for label, mult, dim in irrep_spec:
            for _ in range(mult):
                dims.append(dim)
                labels.append(label)
        # Pad to embed_dim with scalar slots if needed.
        total = sum(dims)
        if total < embed_dim:
            for _ in range(embed_dim - total):
                dims.append(1)
                labels.append('ℓ0_pad')
        return dims, labels

    def _build_mixer_matrix(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        r"""Assemble the K×K commutant mixer M from per-group (I + Δ) blocks."""
        K = self.embed_dim
        M = torch.zeros(K, K, device=device, dtype=dtype)
        for (_, dim, heads), delta_param in zip(self._mixer_groups, self.mixer_delta):
            n = len(heads)
            A = torch.eye(n, device=device, dtype=dtype) + delta_param.to(device=device, dtype=dtype)
            I_d = torch.eye(dim, device=device, dtype=dtype)
            for i_out, h_out in enumerate(heads):
                r0, r1 = self._offsets[h_out], self._offsets[h_out] + dim
                for j_in, h_in in enumerate(heads):
                    c0, c1 = self._offsets[h_in], self._offsets[h_in] + dim
                    M[r0:r1, c0:c1] = A[i_out, j_in] * I_d
        return M

    def forward(
        self, mu: torch.Tensor, sigma: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""Apply M to (μ, Σ) symmetrically.

        μ_out = μ @ Mᵀ        # row-vector form of M·μ in column-vector sense
        Σ_out = M·Σ·Mᵀ        # full cov: explicit sandwich
                              # diag cov: closed form σ'[m·d+k] = Σ_n A[m,n]² σ[n·d+k]
        """
        M = self._build_mixer_matrix(mu.device, mu.dtype)
        mu_out = mu @ M.transpose(-1, -2)

        if sigma is None:
            return mu_out, None

        if sigma.dim() == 4:
            sig_out = M @ sigma @ M.transpose(-1, -2)
            sig_out = 0.5 * (sig_out + sig_out.transpose(-1, -2))
            return mu_out, sig_out

        # Diagonal Σ closed form: equivalent to diag(M·diag(σ)·Mᵀ) but O(K) not O(K²).
        sig_out = torch.zeros_like(sigma)
        for (_, dim, heads), delta_param in zip(self._mixer_groups, self.mixer_delta):
            n = len(heads)
            A = torch.eye(n, device=mu.device, dtype=mu.dtype) + delta_param.to(
                device=mu.device, dtype=mu.dtype
            )
            A_sq = A.pow(2)  # (n, n)
            # Gather σ slices for this group → (..., n, dim).
            sigma_blocks = torch.stack(
                [sigma[..., self._offsets[h]: self._offsets[h] + dim] for h in heads],
                dim=-2,
            )  # (..., n, dim)
            # Each output head m gets Σ_n A[m,n]² · σ_block[n, :] → (..., n, dim).
            mixed = torch.einsum('mn,...nd->...md', A_sq, sigma_blocks)
            for i_out, h_out in enumerate(heads):
                sig_out[..., self._offsets[h_out]: self._offsets[h_out] + dim] = mixed[..., i_out, :]
        return mu_out, sig_out

    def extra_repr(self) -> str:
        n_params = sum(p.numel() for p in self.mixer_delta)
        return (
            f"embed_dim={self.embed_dim}, n_groups={len(self._mixer_groups)}, "
            f"n_params={n_params}, diagonal_covariance={self.diagonal_covariance}"
        )
