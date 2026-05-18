r"""
Equivariant head mixer for the /vfe path.

Mixes heads of the **same irrep type** with a small matrix `A_t ∈ R^{n_t × n_t}`,
embedded as ``kron(A_t, I_{d_t})``. The mixer commutes with the block-diagonal
gauge action of a tied per-token gauge (every head of one token shares the same
gauge frame :math:`\phi_i` in /vfe), so the operation preserves gauge
equivariance of :math:`\mu` and the sandwich-conjugated :math:`\Sigma`.

Initialization is exactly the identity (zero parameters in :math:`A_t - I`),
so a fresh model with the mixer enabled is bitwise indistinguishable from the
mixer-disabled path at step 0.

Math (per irrep type :math:`t` with multiplicity :math:`n_t` and dim :math:`d_t`)::

    M_t = kron(A_t, I_{d_t})                ∈ R^{n_t d_t × n_t d_t}
    μ'  = M_t · μ                           (head-mix on means)
    Σ'  = M_t · Σ · M_t^T                   (sandwich product — full-cov)
    σ'[m, c] = sum_n A_t[m, n]^2 · σ[n, c]  (diagonal-cov closed form)

The diagonal-cov closed form is **only correct under the diagonal-of-sandwich
approximation already used everywhere else in /vfe** when ``diagonal_covariance=True``:

.. math::
    \operatorname{diag}(M_t \operatorname{diag}(\sigma) M_t^T)_i
    = \sum_k M_t[i, k]^2 \sigma_k.

For ``kron(A, I_d)``, indices factor as :math:`i = (m, c)` and :math:`k = (n, c')`
with the Kronecker delta :math:`\delta_{c, c'}`, giving :math:`\sigma'[m, c] =
\sum_n A[m, n]^2 \sigma[n, c]`. No cross-head terms appear — that's why the
diagonal approximation survives.

Gauge equivariance argument: Under a block-diagonal gauge transform
:math:`h = \mathrm{diag}(h_1, \ldots, h_{n_t})` with each :math:`h_k \in
GL(d_t)`, we have :math:`(\mathrm{kron}(A, I_d)) \cdot h \neq h \cdot
(\mathrm{kron}(A, I_d))` in general — the mixer commutes only with **tied**
block-diagonal transforms :math:`h_k = h_0` for all :math:`k`. In /vfe the
per-token gauge :math:`\phi_i` is shared across heads within a token (one
:math:`\phi_i \in R^{n_{\text{gen}}}` projected onto block-diagonal generators),
so the tied-gauge condition holds by construction. See CLAUDE.md ``Documented
exceptions`` and the legacy mixer at ``transformer/core/attention.py``.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import List, Tuple

import torch
import torch.nn as nn


class VFEHeadMixer(nn.Module):
    r"""Schur-commutant head mixer.

    Per-irrep-type :math:`A_t \in R^{n_t \times n_t}` initialized at identity.
    Applied symmetrically to :math:`\mu` and :math:`\Sigma` so the gauge
    transformation laws survive (under tied per-token gauges; see module
    docstring).

    Args:
        irrep_spec: List of ``(type_name, mult, dim)`` triples matching
            :attr:`VFEConfig.irrep_spec`. Heads of the same ``type_name`` share
            one :math:`A_t` matrix; heads of different types do not mix.
        embed_dim: Total :math:`K = \sum_t n_t \cdot d_t`. Used only for
            sanity-checking against the irrep_spec.
    """

    def __init__(
        self,
        irrep_spec: List[Tuple[str, int, int]],
        embed_dim: int,
    ) -> None:
        super().__init__()

        # Validate dim
        computed = sum(mult * dim for _, mult, dim in irrep_spec)
        if computed != embed_dim:
            raise ValueError(
                f"VFEHeadMixer: irrep_spec gives K={computed} but "
                f"embed_dim={embed_dim}; these must agree."
            )

        # Group consecutive heads by type. We do NOT permute heads — the
        # block-diagonal layout in /vfe matches `irrep_spec` order
        # (transformer/vfe/config.py:irrep_dims). If two non-contiguous entries
        # share a type_name, they get their OWN slice list (still grouped under
        # one A_t) so the mixer treats them as one type. The layout assumed by
        # `_apply_block` is contiguous per-type only when irrep_spec is sorted
        # by type — we don't enforce that here; the per-type slice list
        # captures whatever the layout actually is.
        type_to_slices: "OrderedDict[str, List[Tuple[int, int]]]" = OrderedDict()
        cursor = 0
        for type_name, mult, dim in irrep_spec:
            slices = type_to_slices.setdefault(type_name, [])
            for _ in range(mult):
                slices.append((cursor, cursor + dim, dim))
                cursor += dim

        # One A_t parameter per type, initialized at identity. Storing the
        # "delta-from-identity" rather than A_t itself keeps the init exact
        # (no float rounding from torch.eye on init) and makes the bitwise
        # equivalence to the no-mixer path obvious at step 0.
        self._type_names: List[str] = list(type_to_slices.keys())
        self._type_slices: dict = dict(type_to_slices)
        self.mixer_delta = nn.ParameterDict()
        for type_name, slices in type_to_slices.items():
            n = len(slices)
            # All slices of a type should share the same dim. Validate.
            dims = {s[2] for s in slices}
            if len(dims) != 1:
                raise ValueError(
                    f"VFEHeadMixer: irrep type {type_name!r} has heads of "
                    f"different dimensions {dims}; mixer assumes one dim per "
                    "type. Split into distinct type_names if intentional."
                )
            self.mixer_delta[type_name] = nn.Parameter(torch.zeros(n, n))

    def _A(self, type_name: str) -> torch.Tensor:
        r"""Return :math:`A_t = I + \Delta_t` for the named type."""
        delta = self.mixer_delta[type_name]
        n = delta.shape[0]
        # eye matches delta's device + dtype automatically.
        return torch.eye(n, device=delta.device, dtype=delta.dtype) + delta

    def is_identity(self) -> bool:
        r"""True iff every :math:`A_t` is exactly the identity matrix.

        Useful for fast-paths in tests / diagnostics and to assert init
        invariance.
        """
        return all((d.detach() == 0).all().item() for d in self.mixer_delta.values())

    def forward(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Apply :math:`\mu \mapsto M \mu`, :math:`\Sigma \mapsto M \Sigma M^T`.

        Branches on ``sigma.dim()``:
        - 3: diagonal sigma, use the closed-form
          :math:`\sigma'[m, c] = \sum_n A[m, n]^2 \sigma[n, c]` per type.
        - 4: full sigma, apply the exact sandwich :math:`M \Sigma M^T` per type
          on the per-type sub-block.

        Args:
            mu: ``(B, N, K)`` belief means.
            sigma: ``(B, N, K)`` diagonal variances OR ``(B, N, K, K)`` full
                covariance.

        Returns:
            ``(mu_mixed, sigma_mixed)`` with the same shapes as inputs.
        """
        is_diagonal = sigma.dim() == 3

        # Defensive clone — we write per-type sub-blocks into the outputs.
        # The clone is cheap (one tensor) and avoids any chance of aliasing
        # surprises if callers reuse the input later.
        mu_out = mu.clone()
        sigma_out = sigma.clone()

        for type_name, slices in self._type_slices.items():
            A = self._A(type_name)
            n, d = len(slices), slices[0][2]

            # Stack the per-head sub-blocks into shape (B, N, n, d).
            # We use cat after a list comprehension because the head slices
            # may not be contiguous in K (interleaved layouts are unusual but
            # not forbidden by irrep_spec).
            mu_blocks = torch.stack(
                [mu[..., s:e] for (s, e, _) in slices], dim=-2
            )  # (B, N, n, d)

            # mu mixed: M · mu  ⇒  mixed[m, c] = sum_n A[m, n] · mu_blocks[n, c]
            mu_mixed = torch.einsum('mn,...nd->...md', A, mu_blocks)

            # Write back into the output.
            for k, (s, e, _) in enumerate(slices):
                mu_out[..., s:e] = mu_mixed[..., k, :]

            if is_diagonal:
                # σ'[m, c] = sum_n A[m, n]² · σ[n, c] — see module docstring.
                sigma_blocks = torch.stack(
                    [sigma[..., s:e] for (s, e, _) in slices], dim=-2
                )  # (B, N, n, d)
                A_sq = A * A  # (n, n)
                sigma_mixed = torch.einsum('mn,...nd->...md', A_sq, sigma_blocks)
                for k, (s, e, _) in enumerate(slices):
                    sigma_out[..., s:e] = sigma_mixed[..., k, :]
            else:
                # Full cov: extract the per-type sub-block (n·d × n·d).
                # For a contiguous-per-type layout this is one slice; for an
                # interleaved layout we extract a permuted view. Implement
                # the contiguous case directly and fall back to a per-pair
                # loop otherwise — interleaved layouts are uncommon and not
                # currently used by any preset config.
                #
                # Contiguity check: slices form one contiguous run iff each
                # next start equals previous end.
                is_contiguous_run = all(
                    slices[k + 1][0] == slices[k][1]
                    for k in range(len(slices) - 1)
                )
                if is_contiguous_run:
                    s0 = slices[0][0]
                    e0 = slices[-1][1]
                    block = sigma_out[..., s0:e0, s0:e0]   # (B, N, nd, nd)
                    # Reshape into (B, N, n, d, n, d) and apply A on each n axis.
                    B_, N_ = mu.shape[0], mu.shape[1]
                    block_r = block.reshape(B_, N_, n, d, n, d)
                    block_m = torch.einsum(
                        'mp,...pdqe->...mdqe', A, block_r
                    )  # apply A on first n axis
                    block_m = torch.einsum(
                        'nq,...mdqe->...mdne', A, block_m
                    )  # apply A on second n axis
                    sigma_out[..., s0:e0, s0:e0] = block_m.reshape(B_, N_, n * d, n * d)
                else:
                    # Fallback: per-pair sandwich. Materializes A_{m,n}·A_{m',n'}
                    # contributions across head index pairs. This path is
                    # exercised only by exotic irrep_spec layouts.
                    for m_out, (sm, em, _) in enumerate(slices):
                        for mp_out, (smp, emp, _) in enumerate(slices):
                            acc = torch.zeros_like(sigma[..., sm:em, smp:emp])
                            for nn1, (sn, en, _) in enumerate(slices):
                                for nn2, (snp, enp, _) in enumerate(slices):
                                    acc = acc + (
                                        A[m_out, nn1] * A[mp_out, nn2]
                                    ) * sigma[..., sn:en, snp:enp]
                            sigma_out[..., sm:em, smp:emp] = acc

        return mu_out, sigma_out
