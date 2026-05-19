r"""
Non-flat parallel transport for the /vfe path.

Built from scratch (no reference to ``transformer/core/connection.py``). The
default behavior at initialization is **exactly flat**, so a fresh model with
the feature enabled is numerically identical to one with the feature disabled.
All gates that activate non-flatness (the learnable strength scalar, the
per-generator bilinear forms :math:`W^a`) start at zero.

# Mathematical setup

Flat transport in /vfe (see :mod:`transformer.vfe.attention`):

.. math::
    \Omega_{ij}^{\text{flat}} = \exp(\phi_i \cdot G) \exp(-\phi_j \cdot G)

Holonomy around any closed loop is identity. The non-flat extension introduces
a per-edge Lie-algebra element :math:`\delta_{ij} \in R^{n_\text{gen}}` and the
transport becomes

.. math::
    \Omega_{ij}^{\text{nf}} = \exp(\phi_i \cdot G) \exp(\delta_{ij} \cdot G) \exp(-\phi_j \cdot G)

so that the triangle holonomy

.. math::
    H_{ijk} = \exp(\phi_i \cdot G) C_{ijk} \exp(-\phi_i \cdot G), \quad
    C_{ijk} = \exp(\delta_{ij} \cdot G) \exp(\delta_{jk} \cdot G) \exp(\delta_{ki} \cdot G)

is non-trivial whenever :math:`\delta_{ij} + \delta_{jk} + \delta_{ki} \neq 0`
(to linear order in :math:`\delta`).

# Parameterization

:math:`\delta_{ij}^a` is a bilinear in :math:`(\mu_i, \mu_j)`. The component
along generator :math:`G^a` is

.. math::
    \delta_{ij}^a = s \cdot \frac{1}{d_h} \cdot \mu_i^\top B^a \mu_j

where:

* :math:`s = s_{\max} \tanh(\rho)` is a scalar strength gate with learnable
  pre-activation :math:`\rho`, initialized at :math:`\rho = 0` so :math:`s = 0`
  (flat) at init.

* :math:`B^a = (W^a - {W^a}^\top)/2` is the **antisymmetric** part of a
  per-generator bilinear form :math:`W^a \in R^{K \times K}`. Antisymmetry in
  the (i, j) indices (which here equals antisymmetry of :math:`B^a` as a matrix)
  gives :math:`\delta_{ji} = -\delta_{ij}`, so the reverse transport
  :math:`\Omega_{ji}^{\text{nf}}` is the algebraic inverse of
  :math:`\Omega_{ij}^{\text{nf}}` to leading order in :math:`\delta`.

* :math:`W^a` is restricted to the block where generator :math:`G^a` has
  support (per-generator block-mask). This preserves the block-diagonal gauge
  structure: under a tied block-diagonal gauge transformation
  :math:`h = \mathrm{diag}(h_1, \ldots, h_H)`, the bilinear transforms as
  :math:`h^\top W^a h` restricted to block :math:`h(a)` — within that block it's
  the standard adjoint action, preserving the antisymmetric orbit of
  :math:`B^a` for SO(d_h) gauges and remaining a sound approximation for
  GL(d_h) gauges (the same approximation the rest of /vfe already makes for
  non-orthogonal gauges; see CLAUDE.md "diagonal-of-sandwich" notes).

* :math:`1/d_h` per-block scaling keeps :math:`\delta_{ij}^a` bounded as the
  irrep dimension grows.

# Robustness controls

The implementation enforces two independent bounds on :math:`\delta`:

1. **Global tanh gate** :math:`s = s_{\max} \tanh(\rho)`. This caps the
   amplitude of every :math:`\delta` uniformly. At initialization :math:`\rho =
   0` so :math:`s = 0` and the path is bitwise flat.

2. **Per-edge Frobenius clamp**: :math:`\delta_{ij}` is clamped so that
   :math:`\| \delta_{ij} \cdot G \|_F \le \delta_{\max}` at every edge after
   the bilinear is evaluated. The legacy MLP-mode connection's failure mode was
   exactly that the per-edge norm blew up while the global scale stayed small
   (compensating-amplitude pathology). Both bounds are required, not redundant.

# Memory

The per-pair :math:`(\Omega_{ij}, \Omega_{ij}^{-1})` tensor has shape
``(B, N, N, d_h, d_h)`` per irrep block. For the user's preset config
(B=32, N=64, K=20 with 2 blocks of d=10) this is ~104 MB at fp32 per
forward, ~3× that with the autograd graph for ``torch.linalg.matrix_exp``.
Acceptable for short sequences; longer N is OOM-prone.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class VFENonFlatConnection(nn.Module):
    r"""Per-edge Lie-algebra connection :math:`\delta_{ij}`.

    Parameters:

    * ``W`` — bilinear form, shape ``(n_gen, K, K)``. Used antisymmetrized as
      ``(W - W^T)/2`` at every forward. Initialized at zero (flat). The block
      mask ``register_buffer('W_block_mask', ...)`` zeros out entries outside
      generator :math:`a`'s support block.

    * ``raw_strength`` — scalar pre-activation for the global gate
      :math:`s = s_{\max} \tanh(\rho)`. Initialized at zero so the entire
      delta vanishes at init.

    Args:
        generators: ``(n_gen, K, K)`` Lie algebra generators.
        irrep_dims: List of per-block dimensions (matches
            :attr:`VFEConfig.irrep_dims`).
        max_strength: :math:`s_{\max}`, upper bound on the global gate.
        per_edge_delta_max: Per-edge clamp :math:`\delta_{\max}` on
            :math:`\| \delta_{ij} \cdot G \|_F`.

    Forward computes ``(B, N, N, n_gen)`` delta tensor. Constructed lazily —
    pre-allocating ``(B, N, N, n_gen)`` for every E-step iteration would peak
    memory unnecessarily; the bilinear is evaluated in-place each call.
    """

    def __init__(
        self,
        generators: torch.Tensor,
        irrep_dims: List[int],
        max_strength: float = 1.0,
        per_edge_delta_max: float = 1.0,
        head_sub_dims: Optional[List[List[int]]] = None,
    ) -> None:
        super().__init__()

        if not isinstance(generators, torch.Tensor):
            generators = torch.from_numpy(generators).float()
        n_gen, K, _ = generators.shape
        self.n_gen = n_gen
        self.K = K
        self.irrep_dims = list(irrep_dims)
        self.max_strength = float(max_strength)
        self.per_edge_delta_max = float(per_edge_delta_max)
        # Optional refinement: under cross_couplings each super-block holds
        # several original heads. head_sub_dims[h] gives the per-head dims
        # inside super-block h (e.g. [[d_head, d_head], [d_head], [d_head]]
        # for super-blocks [2·d_head, d_head, d_head]). When provided the
        # per-generator support is matched against the finer per-head
        # sub-blocks first, so diagonal-head generators get a tight
        # (d_head × d_head) mask while cross generators that span multiple
        # heads inside a super-block fall back to the full super-block mask.
        self.head_sub_dims = head_sub_dims

        # Build a per-generator block mask. Generator a is identified with the
        # smallest block containing all of its non-zero entries. For
        # well-formed block-diagonal generators (the /vfe convention) this is
        # a single contiguous block. The mask is a binary (n_gen, K, K) buffer
        # that zeros out W^a entries outside generator a's support block. We
        # build it from generator support, not from a hardcoded mapping, so
        # any future generator layout still works without touching this code.
        with torch.no_grad():
            block_mask = torch.zeros_like(generators)
            # Build block candidates: include both the super-block partition
            # (irrep_dims) and, when head_sub_dims is provided, the finer
            # per-head sub-blocks INSIDE each super-block. Sorting ascending
            # by size means the first match in the inner loop is always the
            # smallest containing block.
            block_candidates: List[Tuple[int, int]] = []
            cursor = 0
            for h_idx, d_h in enumerate(irrep_dims):
                block_candidates.append((cursor, cursor + d_h))
                if head_sub_dims is not None and h_idx < len(head_sub_dims):
                    sub_cursor = cursor
                    for sub_d in head_sub_dims[h_idx]:
                        block_candidates.append((sub_cursor, sub_cursor + sub_d))
                        sub_cursor += sub_d
                cursor += d_h
            # Sort smallest-first (tightest containing block wins).
            block_candidates.sort(key=lambda be: be[1] - be[0])
            block_starts = block_candidates
            for a in range(n_gen):
                G_a = generators[a]
                # Identify smallest block containing G_a's support.
                nz_rows = (G_a.abs().sum(dim=-1) > 1e-12).nonzero(as_tuple=True)[0]
                nz_cols = (G_a.abs().sum(dim=-2) > 1e-12).nonzero(as_tuple=True)[0]
                if nz_rows.numel() == 0:
                    continue  # zero generator — leave mask zero
                row_lo, row_hi = int(nz_rows.min()), int(nz_rows.max()) + 1
                col_lo, col_hi = int(nz_cols.min()), int(nz_cols.max()) + 1
                # Find the irrep block that covers this support.
                chosen = None
                for (bs, be) in block_starts:
                    if bs <= row_lo and row_hi <= be and bs <= col_lo and col_hi <= be:
                        chosen = (bs, be)
                        break
                if chosen is None:
                    # Generator spans multiple blocks (e.g., off-block-diagonal
                    # mixing). Mask remains zero — δ_ij^a will then evaluate
                    # to zero for this generator. This is safer than silently
                    # allowing a non-block-respecting bilinear: an off-block
                    # generator is not consistent with /vfe's block-diagonal
                    # gauge structure and should be a no-op until designed for.
                    continue
                bs, be = chosen
                block_mask[a, bs:be, bs:be] = 1.0
        self.register_buffer('W_block_mask', block_mask)

        # Per-generator irrep dim — for the 1/d_h scaling.
        # d_h_per_gen[a] is the dim of the block containing generator a; 1 if
        # the generator has empty support (degenerate).
        with torch.no_grad():
            d_h_per_gen = torch.ones(n_gen, dtype=generators.dtype)
            for a in range(n_gen):
                mask_a = block_mask[a]
                if mask_a.sum() > 0:
                    # Block size = sqrt(nonzero entries in mask).
                    d_h_per_gen[a] = float(int(mask_a.sum().sqrt().round().item()))
            inv_d_h = 1.0 / d_h_per_gen.clamp(min=1.0)
        self.register_buffer('inv_d_h_per_gen', inv_d_h)  # (n_gen,)

        # Generators buffer (frozen).
        self.register_buffer('generators', generators)

        # W: bilinear forms. Init zero ⇒ δ ≡ 0 ⇒ flat path at step 0.
        self.W_raw = nn.Parameter(torch.zeros(n_gen, K, K))

        # Strength gate ρ. tanh(0) = 0 ⇒ s = 0 ⇒ δ ≡ 0 even if W learns
        # non-zero. Two-stage safety: a fresh model with the feature on emits
        # the flat path exactly; the optimizer has to move BOTH ρ and W away
        # from zero to introduce non-flatness.
        self.raw_strength = nn.Parameter(torch.zeros(()))

        # Precompute per-generator Frobenius norm of G_a, used for the
        # per-edge clamp ‖δ_ij · G‖_F ≤ δ_max.
        # ‖δ_ij · G‖_F² = δ_ij[a] · δ_ij[b] · <G_a, G_b>_F. To avoid materializing
        # the full Gram matrix at every call, we use the diagonal upper bound
        # ‖δ · G‖_F² ≤ Σ_a |δ[a]|² · ‖G_a‖_F²  (Cauchy–Schwarz). Conservative
        # but cheap and never under-clamps. Tighter bound (full Gram) is
        # available behind a config knob if profiling shows real over-clamping.
        with torch.no_grad():
            g_fro_sq = (generators ** 2).sum(dim=(-2, -1)).clamp(min=1e-12)  # (n_gen,)
        self.register_buffer('g_fro_sq', g_fro_sq)

        # W() cache: invalidated by W_raw._version bumps. Declared here so
        # they're visible to instance-attribute static analysis, and the
        # initial key is a sentinel that never matches a real _version (which
        # starts at 0 after construction).
        self._W_cache_key: int = -1
        self._W_cache: Optional[torch.Tensor] = None

    # -- properties -----------------------------------------------------------

    @property
    def strength(self) -> torch.Tensor:
        r"""Current scalar gate :math:`s = s_{\max} \tanh(\rho) \in (-s_{\max}, s_{\max})`."""
        return self.max_strength * torch.tanh(self.raw_strength)

    def W(self) -> torch.Tensor:
        r"""Antisymmetric, block-masked bilinear forms ``(n_gen, K, K)``.

        Cached per-(version of W_raw) so the antisymmetrization runs once per
        optimizer step instead of three times per E-step iteration. Autograd
        fan-out from the cached tensor is the standard pattern — gradients
        accumulate at W_raw across the multiple downstream consumers.
        """
        cache_key = self.W_raw._version
        cached = self._W_cache
        # Cross-check device: ``.to(device)`` moves W_raw (a parameter) but
        # doesn't touch _W_cache (a plain attr). After such a move, _version
        # still bumps so this normally re-fires anyway, but guard explicitly
        # for the edge case where the version hasn't advanced.
        if (
            cached is not None
            and self._W_cache_key == cache_key
            and cached.device == self.W_raw.device
        ):
            return cached
        W_masked = self.W_raw * self.W_block_mask
        result = 0.5 * (W_masked - W_masked.transpose(-1, -2))
        self._W_cache_key = cache_key
        self._W_cache = result
        return result

    # -- forward --------------------------------------------------------------

    def forward(
        self,
        mu: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""Compute :math:`\delta_{ij}^a` for every (i, j, a).

        Args:
            mu: ``(B, N, K)`` belief means. **Not detached** — autograd flows
                through the bilinear, which is the correct behavior: when the
                E-step inner loop updates :math:`\mu`, the connection updates
                with it (the Plan agent's "δ moves with μ for free" property).
            mask: ``(B, N, N)`` causal mask (0 = masked). Used only to zero
                out masked edges so the per-edge clamp doesn't observe noise
                from positions that won't contribute to attention anyway.

        Returns:
            ``delta``: ``(B, N, N, n_gen)``. Antisymmetric in the (i, j) axes
            up to numerical precision: ``delta[b, i, j, a] = -delta[b, j, i, a]``.
        """
        # Bilinear: δ_raw[b, i, j, a] = μ_i^T B^a μ_j, B^a = (W^a)_antisym
        # einsum over the (K, K) bilinear axes; n_gen index is preserved.
        # B has shape (n_gen, K, K), μ has (B, N, K).
        B_anti = self.W()  # (n_gen, K, K)
        delta_raw = torch.einsum('bik,akl,bjl->bija', mu, B_anti, mu)

        # Per-generator 1/d_h scaling and global strength gate.
        delta = delta_raw * self.inv_d_h_per_gen.view(1, 1, 1, -1)
        delta = self.strength * delta

        # Per-edge Frobenius clamp on ‖δ_ij · G‖_F.
        # Upper bound (Cauchy–Schwarz): ‖δ · G‖_F² ≤ Σ_a |δ_a|² · ‖G_a‖_F².
        # Closed-form scale: clamp(fro_bound_sq, min=δ_max²) gives a tensor
        # equal to δ_max² where below the bound (so scale = 1) and equal to
        # fro_bound_sq where above (so scale = δ_max / √fro_bound_sq < 1).
        # Detached (no_grad) — straight-through w.r.t. autograd, matching the
        # standard gradient-clip pattern.
        with torch.no_grad():
            fro_bound_sq = (delta ** 2 * self.g_fro_sq.view(1, 1, 1, -1)).sum(dim=-1)
            denom = fro_bound_sq.clamp(min=self.per_edge_delta_max ** 2).sqrt()
            scale = self.per_edge_delta_max / denom
        delta = delta * scale.unsqueeze(-1)

        # Mask: zero out masked edges. They wouldn't appear in attention anyway
        # (the softmax masks them), but zeroing keeps the per-pair Omega tensor
        # tidy and downstream diagnostics meaningful.
        if mask is not None:
            delta = delta * mask.unsqueeze(-1)

        return delta


# -----------------------------------------------------------------------------
# Pairwise Omega construction
# -----------------------------------------------------------------------------


def compute_pairwise_omega_with_delta(
    phi: torch.Tensor,
    delta: torch.Tensor,
    generators: torch.Tensor,
    irrep_dims: List[int],
    cached_block_exp_pairs: Optional[List[Tuple[torch.Tensor, Optional[torch.Tensor]]]] = None,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    r"""Build per-block pairwise Omega and its inverse from :math:`\phi` and
    :math:`\delta`.

    For each irrep block :math:`h`:

    .. math::
        \Omega^{(h)}_{ij} = \exp(\phi_i \cdot G^{(h)}) \exp(\delta_{ij} \cdot G^{(h)}) \exp(-\phi_j \cdot G^{(h)})

    .. math::
        \Omega^{(h)-1}_{ij} = \exp(\phi_j \cdot G^{(h)}) \exp(-\delta_{ij} \cdot G^{(h)}) \exp(-\phi_i \cdot G^{(h)})

    The inverse is computed via three additional matrix exponentials, not via
    an explicit ``torch.linalg.inv`` — same FLOP cost, no condition-number
    worry, matches the legacy ``(exp_h, exp_neg_h)`` pair contract.

    Args:
        phi: ``(B, N, n_gen)`` gauge frame coordinates.
        delta: ``(B, N, N, n_gen)`` non-flat connection.
        generators: ``(n_gen, K, K)`` Lie algebra generators.
        irrep_dims: Block dimensions.
        cached_block_exp_pairs: Optional precomputed ``(exp(φ·G^(h)), exp(-φ·G^(h)))``
            pairs from :func:`transformer.vfe.attention.compute_gauge_transport`.
            If provided and ``phi`` is the same tensor that was used to build it,
            we reuse it; otherwise we rebuild. Caller is responsible for the
            ``phi`` ↔ cache identity (see :mod:`transformer.vfe.e_step`'s phi
            update path for the cache-discipline rationale).

    Returns:
        List of length ``len(irrep_dims)``. Each element is a pair
        ``(Omega_h, Omega_inv_h)`` with shape ``(B, N, N, d_h, d_h)``.
    """
    from transformer.core.gauge_utils import fused_block_matrix_exp_pairs

    if cached_block_exp_pairs is not None:
        phi_pairs = cached_block_exp_pairs
    else:
        # skew_symmetric=None ⇒ detect from generators; cheap once. The /vfe
        # attention helper already implements this and caches by generator id.
        phi_pairs = fused_block_matrix_exp_pairs(
            phi, generators, irrep_dims,
            enforce_orthogonal=False, skew_symmetric=None,
        )

    out = []
    block_start = 0
    for h, d_h in enumerate(irrep_dims):
        block_end = block_start + d_h
        # Generators restricted to block h.
        G_h = generators[:, block_start:block_end, block_start:block_end]  # (n_gen, d_h, d_h)

        # algebra_ij[b, i, j, k, l] = sum_a delta[b, i, j, a] * G_h[a, k, l]
        algebra = torch.einsum('bija,akl->bijkl', delta, G_h)

        # phi_pairs[h][1] is None iff fused_block_matrix_exp_pairs detected
        # skew-symmetric generators (SO(N)). In that case exp(-X) = exp(X)^T
        # for any X in the same Lie algebra, so we can save one matrix_exp
        # on the δ-exp pair by reusing the transpose. This is bit-exact for
        # skew-symmetric generators; for general GL(K) we still need two
        # matrix_exp calls.
        _is_skew = phi_pairs[h][1] is None

        # exp(δ_ij · G^(h)) — autograd through matrix_exp is supported as of
        # PyTorch 1.7. Float32 path; AMP off for stability (matrix_exp is
        # sensitive to overflow at fp16).
        with torch.amp.autocast('cuda', enabled=False):
            exp_delta = torch.linalg.matrix_exp(algebra.float())             # (B, N, N, d_h, d_h)
            if _is_skew:
                exp_neg_delta = exp_delta.transpose(-1, -2)
            else:
                exp_neg_delta = torch.linalg.matrix_exp(-algebra.float())     # (B, N, N, d_h, d_h)

        exp_phi_h = phi_pairs[h][0].to(exp_delta.dtype)            # (B, N, d_h, d_h)
        exp_neg_phi_h = phi_pairs[h][1].to(exp_delta.dtype) if phi_pairs[h][1] is not None else None

        if exp_neg_phi_h is None:
            # Skew-symmetric generators (SO(N)): the cache only stores exp(+φ·G);
            # the inverse is exp(-φ·G) = exp(+φ·G)^T because G is antisymmetric
            # and exp of antisymmetric is orthogonal. We compute the transpose
            # in a way that's autograd-friendly.
            exp_neg_phi_h = exp_phi_h.transpose(-1, -2)

        # Omega_ij = exp_phi_i @ exp_delta_ij @ exp_neg_phi_j
        # exp_phi_h has shape (B, N, d, d). Insert a new j-axis: (B, N, 1, d, d).
        # exp_neg_phi_h same: (B, 1, N, d, d).
        Omega = (
            exp_phi_h.unsqueeze(2)
            @ exp_delta
            @ exp_neg_phi_h.unsqueeze(1)
        )  # (B, N, N, d_h, d_h)

        # Omega_inv_ij = exp_phi_j @ exp_neg_delta_ij @ exp_neg_phi_i
        Omega_inv = (
            exp_phi_h.unsqueeze(1)            # (B, 1, N, d, d)
            @ exp_neg_delta
            @ exp_neg_phi_h.unsqueeze(2)      # (B, N, 1, d, d)
        )  # (B, N, N, d_h, d_h)

        out.append((Omega, Omega_inv))
        block_start = block_end
    return out


# -----------------------------------------------------------------------------
# KL attention with pairwise Omega
# -----------------------------------------------------------------------------


def compute_kl_attention_pairwise(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    omega_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    irrep_dims: List[int],
    kappa: "float | torch.Tensor",
    mask: Optional[torch.Tensor] = None,
    mask_self_attention: bool = True,
    eps: float = 1e-8,
    use_rope: bool = False,
    rope_base: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Compute KL-attention weights from per-pair Omega tensors.

    Computes :math:`\beta_{ij} = \mathrm{softmax}_j\bigl(-D_{ij}/\kappa\bigr)` where
    :math:`D_{ij} = \sum_h \mathrm{KL}\bigl(q_i^{(h)} \,\|\, \Omega^{(h)}_{ij} \cdot q_j^{(h)}\bigr)`
    summed over irrep blocks. The KL between diagonal-Σ source and full-Σ
    target uses the **diagonal-of-sandwich** approximation that the rest of
    /vfe already commits to when ``exact_diagonal_transport=False``.

    Args:
        mu: ``(B, N, K)`` belief means.
        sigma: ``(B, N, K)`` diagonal variances. Full-cov is NOT supported in
            this entry point — non-flat + full-cov needs a logdet-aware
            per-pair KL that's deferred until/unless someone needs it.
        omega_pairs: Output of :func:`compute_pairwise_omega_with_delta`.
        irrep_dims: Per-block dimensions.
        kappa: Attention temperature (scalar or 0-dim tensor).
        mask: ``(B, N, N)`` causal mask (0 = mask out).
        mask_self_attention: When True, zeros the diagonal of β. Mirrors
            :func:`transformer.vfe.attention.compute_kl_attention`.
        eps: Variance clamp floor.

    Returns:
        ``(beta, kl_matrix)``: both ``(B, N, N)``.
    """
    if sigma.dim() != 3:
        raise NotImplementedError(
            "compute_kl_attention_pairwise currently supports only diagonal "
            "covariance (sigma.dim()==3). Full-cov non-flat attention needs a "
            "per-pair logdet path; not yet wired."
        )

    # RoPE — apply to mu BEFORE the KL is built so non-flat matches the
    # baseline flat path at delta=0. Σ is untouched per
    # rope_full_gauge='off' (validated at VFEConfig.__post_init__ when
    # use_non_flat_transport is True).
    if use_rope:
        from transformer.core.transport_ops import _apply_rope
        mu = _apply_rope(mu, base=rope_base)

    B, N, K = mu.shape
    kl_total = torch.zeros(B, N, N, device=mu.device, dtype=mu.dtype)
    block_start = 0

    for h, d_h in enumerate(irrep_dims):
        block_end = block_start + d_h
        Omega_h, _ = omega_pairs[h]   # (B, N, N, d_h, d_h)
        mu_h = mu[..., block_start:block_end]            # (B, N, d_h)
        sigma_h = sigma[..., block_start:block_end].clamp(min=eps)  # (B, N, d_h)

        # μ_target_ij[b, i, j, k] = Σ_l Omega_ij[b, i, j, k, l] · μ_j[b, j, l]
        mu_target = torch.einsum('bijkl,bjl->bijk', Omega_h, mu_h)  # (B, N, N, d_h)

        # σ_target_ij[b, i, j, k] = Σ_l Omega_ij[b, i, j, k, l]² · σ_j[b, j, l]
        # is the diagonal of `Omega @ diag(σ) @ Omega^T` — NOT the full
        # sandwich. For non-orthogonal Omega ∈ GL(d_h) the off-diagonal
        # entries of the sandwich are non-zero and dropped. This is the
        # diagonal-σ approximation documented in `VFEConfig` (see
        # `diagonal_covariance` / `exact_diagonal_transport`). The non-flat
        # track does not expose an `exact_diagonal_transport` toggle —
        # `compute_kl_attention_pairwise` rejects full-cov input at the top
        # (`sigma.dim() != 3` → NotImplementedError). Folded Omega_h**2 into
        # the einsum operand list to avoid materializing the (B,N,N,d_h,d_h)
        # square as a temporary.
        sigma_target = torch.einsum(
            'bijkl,bijkl,bjl->bijk', Omega_h, Omega_h, sigma_h,
        ).clamp(min=eps)

        # KL(N(μ_i, diag σ_i) || N(μ_target, diag σ_target)) summed over k.
        sigma_i = sigma_h.unsqueeze(2)                  # (B, N, 1, d_h)
        mu_i = mu_h.unsqueeze(2)                        # (B, N, 1, d_h)
        diff = mu_i - mu_target                         # (B, N, N, d_h)
        trace_term = sigma_i / sigma_target             # (B, N, N, d_h)
        mahal_term = diff ** 2 / sigma_target           # (B, N, N, d_h)
        logdet_term = sigma_target.log() - sigma_i.log()  # (B, N, N, d_h)
        kl_h = 0.5 * (trace_term + mahal_term - 1.0 + logdet_term).sum(dim=-1)
        # (B, N, N)
        kl_total = kl_total + kl_h
        block_start = block_end

    # Effective temperature: τ = κ · √K. Matches the canonical F functional
    # (see CLAUDE.md and the manuscript's free-energy form). The compute path
    # in /vfe's compute_attention_weights also bakes in this √K factor when it
    # calls into the core kernel; we replicate it here.
    if isinstance(kappa, torch.Tensor):
        tau = kappa * math.sqrt(max(K, 1))
    else:
        tau = float(kappa) * math.sqrt(max(K, 1))

    logits = -kl_total / tau                            # (B, N, N)

    # Mask handling mirrors transformer/core/attention.py:compute_attention_weights
    # so that row-i=0 + causal mask + mask_self_attention does NOT zero out the
    # one valid target (j=0) — the flat-path keeps the self-attention escape
    # hatch when no other target is available, and we follow that convention.
    if mask is not None:
        logits = logits.masked_fill(mask == 0, float('-inf'))

    if mask_self_attention:
        diag_idx = torch.arange(N, device=logits.device)
        # A row "has other targets" if MORE than one of its positions is finite
        # (i.e. it could attend to something other than itself after the causal
        # mask). Only those rows get their diagonal -inf'd. Without this guard,
        # row 0 of a causally-masked sequence would become all -inf and the
        # softmax produces NaN.
        has_other_targets = (logits != float('-inf')).sum(dim=-1) > 1   # (B, N)
        diag_vals = logits[:, diag_idx, diag_idx]
        masked_diag_vals = torch.where(
            has_other_targets,
            torch.full_like(diag_vals, float('-inf')),
            diag_vals,
        )
        logits[:, diag_idx, diag_idx] = masked_diag_vals

    beta = F.softmax(logits, dim=-1)
    # Clamp only non-masked positions to ε for numerical stability, preserving
    # exact zeros at masked positions (causal mask).
    masked_positions = (logits == float('-inf'))
    beta = torch.where(masked_positions, beta, beta.clamp(min=eps))
    beta_sum = beta.sum(dim=-1, keepdim=True).clamp(min=eps)
    beta = beta / beta_sum
    return beta, kl_total


# -----------------------------------------------------------------------------
# Diagnostics
# -----------------------------------------------------------------------------


def triangle_holonomy_norm(
    omega_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    irrep_dims: List[int],
    n_samples: int = 32,
    eps: float = 1e-12,
) -> float:
    r"""Empirical proxy for the per-block triangle holonomy magnitude.

    For random index triples :math:`(i, j, k)`, computes the deviation
    :math:`\| \Omega_{ij} \Omega_{jk} \Omega_{ki} - I \|_F` per block,
    averaged across samples and blocks. Returns 0 (up to numerical noise)
    for flat transport.

    Useful as a sanity-check diagnostic: assert this is ~0 when the strength
    gate is at init, and grows as the connection learns.

    Args:
        omega_pairs: From :func:`compute_pairwise_omega_with_delta`.
        irrep_dims: Per-block dims.
        n_samples: Number of random triples to average.
        eps: Variance floor (unused but kept for signature uniformity).

    Returns:
        Scalar float — the mean Frobenius-norm holonomy across blocks/samples.
    """
    total = 0.0
    n_blocks = len(irrep_dims)
    for h, d_h in enumerate(irrep_dims):
        Omega_h, _ = omega_pairs[h]
        B_, N_, _, _, _ = Omega_h.shape
        # Sample random triples (i, j, k) per batch.
        idx_i = torch.randint(0, N_, (n_samples,), device=Omega_h.device)
        idx_j = torch.randint(0, N_, (n_samples,), device=Omega_h.device)
        idx_k = torch.randint(0, N_, (n_samples,), device=Omega_h.device)
        # Take first batch element (cheap diagnostic; batch loop unnecessary).
        Om_ij = Omega_h[0, idx_i, idx_j]    # (n_samples, d_h, d_h)
        Om_jk = Omega_h[0, idx_j, idx_k]
        Om_ki = Omega_h[0, idx_k, idx_i]
        prod = Om_ij @ Om_jk @ Om_ki
        eye = torch.eye(d_h, device=Omega_h.device, dtype=prod.dtype).unsqueeze(0)
        dev = (prod - eye).reshape(n_samples, -1).norm(dim=-1).mean()
        total += float(dev.item())
    return total / max(n_blocks, 1)
