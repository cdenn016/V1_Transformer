r"""
Diagnostics for cross-head coupling in the /vfe path.

All metrics are stateless, take tensors plus a :class:`VFEConfig`, and return
plain Python scalars / numpy arrays / dicts ready for logging and plotting.
They are no-ops (return empty / NaN-tagged) when ``cfg.cross_couplings`` is
empty, so unconditional calls from the training loop stay safe.

Reading guide:

* ``phi_energy_partition`` — splits the L2 mass of ``phi`` between the
  diagonal generator group and the cross-coupling group, per layer / batch.
* ``omega_block_strength`` — average Frobenius mass of each (super-block, head,
  head) sub-block of the per-token transport ``Omega``. The off-diagonal
  entries inside a merged super-block are the cross-coupling contribution.
* ``cross_block_kl_share`` — fraction of the total per-block KL attention
  matrix that lives in the cross-coupled super-blocks vs the singletons.
* ``per_super_block_effective_rank`` — exponential-entropy effective rank of
  the diagonal covariance per super-block (a coarse "is this block being
  used?" signal).
* ``per_super_block_attention_entropy`` — Shannon entropy of the per-block
  beta_ij distribution per super-block.
* ``holonomy_norm_cross_blocks`` — Frobenius norm of the loop-3 holonomy
  (Omega_ij · Omega_jk · Omega_ki) on the cross-coupled super-blocks. A
  proxy for "how much of the non-trivial transport is happening between
  cross-coupled heads".

Conventions:

* All inputs accept torch tensors on either CPU or CUDA. Returned scalars are
  Python floats; per-block tensors are returned as numpy ``float64`` for
  downstream plotting / CSV logging.
* When ``cfg.cross_couplings`` is empty every function returns the same
  type-shape it would under cross-coupling but with the cross-side fields
  zeroed (or ``None`` where appropriate). This keeps the metrics safe to log
  unconditionally without branching on coupling in the trainer.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import math

import numpy as np
import torch


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().to(torch.float64).cpu().numpy()


def _diag_generator_indices(cfg) -> List[int]:
    r"""Indices of the per-head diagonal generators in the reordered basis.

    Under ``cross_couplings``, ``_build_generators`` emits
    ``n_heads · d_head^2`` diagonal generators in original head order, followed
    by ``|cross_couplings| · d_head^2`` cross generators, then permutes the K
    axis so super-blocks are contiguous. The generator index ORDER is
    unchanged by the K-axis permutation, so the diagonal indices are exactly
    ``range(n_heads * d_head^2)``.

    Returns the empty list when cross-coupling is inactive — every generator
    is "diagonal" in that case.
    """
    if not cfg.cross_couplings:
        return []
    _, n_heads, d_head = cfg.irrep_spec[0]
    return list(range(n_heads * d_head * d_head))


def _cross_generator_indices(cfg) -> List[int]:
    r"""Indices of the cross-head generators in the reordered basis.

    Empty when cross-coupling is inactive.
    """
    if not cfg.cross_couplings:
        return []
    _, n_heads, d_head = cfg.irrep_spec[0]
    n_diag = n_heads * d_head * d_head
    n_cross = len(cfg.cross_couplings) * d_head * d_head
    return list(range(n_diag, n_diag + n_cross))


# -----------------------------------------------------------------------------
# Phi energy partition
# -----------------------------------------------------------------------------


def phi_energy_partition(
    phi: torch.Tensor,
    cfg,
) -> Dict[str, float]:
    r"""Split ``||phi||^2`` between diagonal and cross-coupling generator groups.

    .. math::
        E_{\text{diag}}  &= \sum_{a \in I_{\text{diag}}}  \overline{\phi_a^2}, \\
        E_{\text{cross}} &= \sum_{a \in I_{\text{cross}}} \overline{\phi_a^2},

    where the overline denotes mean over batch / sequence axes.

    Returns:
        Dict with ``'phi_energy_diag'``, ``'phi_energy_cross'``,
        ``'phi_energy_total'``, and ``'phi_energy_cross_share'``
        (= cross / total, clamped to 0 when total = 0).
    """
    cross_idx = _cross_generator_indices(cfg)
    if not cross_idx:
        e_tot = float((phi ** 2).mean(dim=(0, 1)).sum().item())
        return {
            'phi_energy_diag': e_tot,
            'phi_energy_cross': 0.0,
            'phi_energy_total': e_tot,
            'phi_energy_cross_share': 0.0,
        }
    diag_idx = _diag_generator_indices(cfg)
    e_per_gen = (phi ** 2).mean(dim=(0, 1))    # (n_gen,)
    e_diag = float(e_per_gen[diag_idx].sum().item())
    e_cross = float(e_per_gen[cross_idx].sum().item())
    e_tot = e_diag + e_cross
    share = e_cross / e_tot if e_tot > 0 else 0.0
    return {
        'phi_energy_diag': e_diag,
        'phi_energy_cross': e_cross,
        'phi_energy_total': e_tot,
        'phi_energy_cross_share': share,
    }


# -----------------------------------------------------------------------------
# Per-super-block Omega block strength
# -----------------------------------------------------------------------------


def omega_block_strength(
    omega: torch.Tensor,
    cfg,
) -> np.ndarray:
    r"""Mean Frobenius mass of each (head_a, head_b) sub-block of the per-token
    transport :math:`\Omega = \exp(\phi \cdot G)` aggregated over (B, N).

    Under cross_couplings each merged super-block has full GL(d_super) gauge,
    so :math:`\Omega` has non-zero off-diagonal head-blocks WITHIN the super-
    block — those off-diagonal entries are precisely the cross-coupling
    contribution. This function returns the (n_heads, n_heads) matrix whose
    entry (a, b) is the mean Frobenius mass of :math:`\Omega` restricted to
    the (head_a, head_b) sub-block, averaged over (B, N).

    Args:
        omega: ``(B, N, K, K)`` per-token transport (composed across blocks).
            Constructed downstream as ``block_diag(exp(phi_h · G^h))`` from
            ``fused_block_matrix_exp_pairs``, then ``block_diag``-assembled
            into the K dim.
        cfg: VFEConfig with active cross-coupling structure.

    Returns:
        ``(n_heads, n_heads)`` numpy array. Returns the all-zeros matrix
        when cross-coupling is inactive.
    """
    _, n_heads, d_head = cfg.irrep_spec[0]
    K = cfg.embed_dim

    if not cfg.cross_couplings:
        # The transport is exactly block-diagonal — non-diagonal entries are
        # algebraically zero. Return a zero matrix for shape stability.
        return np.zeros((n_heads, n_heads), dtype=np.float64)

    # The super-block permutation reordered the K axis. To address by
    # *original* head indices, invert the permutation.
    perm = getattr(cfg, '_cross_head_perm', None)
    if perm is None:
        # Should not happen if cross_couplings is non-empty — defensive.
        return np.zeros((n_heads, n_heads), dtype=np.float64)

    inv_perm = np.argsort(perm)
    # Index into omega's K axis with original head order.
    inv_perm_t = torch.from_numpy(inv_perm).long().to(omega.device)
    omega_orig = omega.index_select(-2, inv_perm_t).index_select(-1, inv_perm_t)
    # Frobenius mass per (d_head x d_head) block, averaged over (B, N).
    out = np.zeros((n_heads, n_heads), dtype=np.float64)
    for a in range(n_heads):
        a_s, a_e = a * d_head, (a + 1) * d_head
        for b in range(n_heads):
            b_s, b_e = b * d_head, (b + 1) * d_head
            block = omega_orig[..., a_s:a_e, b_s:b_e]    # (B, N, d, d)
            out[a, b] = float((block ** 2).sum(dim=(-2, -1)).mean().item())
    return out


# -----------------------------------------------------------------------------
# Cross-block KL share
# -----------------------------------------------------------------------------


def cross_block_kl_share(
    kl_per_block: torch.Tensor,
    cfg,
    mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    r"""Fraction of the total per-super-block KL that lives in cross-coupled
    (merged) super-blocks vs singleton super-blocks.

    Args:
        kl_per_block: ``(B, N, N, H_super)`` per-super-block KL values from
            the E-step block-diagonal kernel. ``H_super`` is the number of
            super-blocks (``len(cfg.effective_block_dims)``).
        cfg: VFEConfig.
        mask: ``(B, N, N)`` causal mask (1 = attend, 0 = mask). Optional.

    Returns:
        Dict with::

            'kl_total'      : sum over all super-blocks
            'kl_merged'     : sum over merged super-blocks only
            'kl_singleton'  : sum over singleton super-blocks only
            'kl_merged_share' : kl_merged / kl_total (0 if total is 0)
    """
    if mask is not None:
        kl_per_block = kl_per_block * mask.unsqueeze(-1)

    super_dims = cfg.effective_block_dims
    if not cfg.cross_couplings:
        total = float(kl_per_block.sum().item())
        return {
            'kl_total': total,
            'kl_merged': 0.0,
            'kl_singleton': total,
            'kl_merged_share': 0.0,
        }
    _, _n_heads, d_head = cfg.irrep_spec[0]
    merged_blocks = [i for i, d in enumerate(super_dims) if d > d_head]
    singleton_blocks = [i for i, d in enumerate(super_dims) if d == d_head]

    per_block_sums = kl_per_block.sum(dim=(0, 1, 2))  # (H_super,)
    total = float(per_block_sums.sum().item())
    merged_sum = (
        float(per_block_sums[merged_blocks].sum().item()) if merged_blocks else 0.0
    )
    singleton_sum = (
        float(per_block_sums[singleton_blocks].sum().item()) if singleton_blocks else 0.0
    )
    share = merged_sum / total if total > 0 else 0.0
    return {
        'kl_total': total,
        'kl_merged': merged_sum,
        'kl_singleton': singleton_sum,
        'kl_merged_share': share,
    }


# -----------------------------------------------------------------------------
# Per-super-block effective rank
# -----------------------------------------------------------------------------


def per_super_block_effective_rank(
    sigma: torch.Tensor,
    cfg,
    eps: float = 1e-12,
) -> np.ndarray:
    r"""Exponential-entropy effective rank per super-block.

    For diagonal sigma ``(B, N, K)``: normalize within each super-block, then
    :math:`R_h = \exp(-\sum_k p_k \log p_k)` with :math:`p_k = \sigma_k /
    \sum_{k'} \sigma_{k'}`. Returned matrix is averaged over (B, N).

    For full-cov ``(B, N, K, K)``: use the diagonal of the per-block sub-
    matrix (a tight underestimate that avoids an eigendecomposition).

    Returns:
        ``(H_super,)`` numpy array of mean effective ranks. Bounded in
        ``[1, d_super]``. Returns the per-irrep-block effective rank when
        cross-coupling is inactive (same shape, no semantic difference).
    """
    super_dims = cfg.effective_block_dims
    H = len(super_dims)
    out = np.zeros((H,), dtype=np.float64)

    if sigma.dim() == 4:
        sigma_diag = torch.diagonal(sigma, dim1=-2, dim2=-1)  # (B, N, K)
    else:
        sigma_diag = sigma

    block_start = 0
    for h, d_h in enumerate(super_dims):
        block_end = block_start + d_h
        s_h = sigma_diag[..., block_start:block_end].clamp(min=eps)
        p = s_h / s_h.sum(dim=-1, keepdim=True).clamp(min=eps)
        # Entropy: -sum p log p ∈ [0, log(d_h)]
        ent = -(p * p.clamp(min=eps).log()).sum(dim=-1)        # (B, N)
        eff_rank = ent.exp()                                   # (B, N)
        out[h] = float(eff_rank.mean().item())
        block_start = block_end
    return out


# -----------------------------------------------------------------------------
# Per-super-block attention entropy
# -----------------------------------------------------------------------------


def per_super_block_attention_entropy(
    beta_per_block: torch.Tensor,
    cfg,
    eps: float = 1e-12,
) -> np.ndarray:
    r"""Mean row-wise Shannon entropy of beta_ij per super-block.

    Args:
        beta_per_block: ``(B, N, N, H_super)`` per-super-block attention
            weights from the per-head-softmax path in :mod:`vfe.e_step`. Each
            ``beta_per_block[..., h]`` is row-stochastic over the N axis.
        cfg: VFEConfig.

    Returns:
        ``(H_super,)`` numpy array. Each entry is the mean Shannon entropy
        (nats) over queries (B, N).
    """
    H = beta_per_block.shape[-1]
    out = np.zeros((H,), dtype=np.float64)
    # Row-entropy in nats: -sum_j beta_ij log beta_ij
    log_beta = beta_per_block.clamp(min=eps).log()
    row_ent = -(beta_per_block * log_beta).sum(dim=-2)        # (B, N, H_super)
    for h in range(H):
        out[h] = float(row_ent[..., h].mean().item())
    return out


# -----------------------------------------------------------------------------
# Holonomy norm for cross-coupled super-blocks
# -----------------------------------------------------------------------------


def holonomy_norm_cross_blocks(
    omega_pairwise: List[Tuple[torch.Tensor, torch.Tensor]],
    cfg,
    triangles: Optional[List[Tuple[int, int, int]]] = None,
    n_random_triangles: int = 64,
) -> Dict[str, float]:
    r"""Frobenius norm of three-edge holonomy on each super-block.

    For each triangle (i, j, k) and each super-block h:

    .. math::
        H^{(h)}_{ijk} = \Omega^{(h)}_{ij} \cdot \Omega^{(h)}_{jk} \cdot
                       \Omega^{(h)}_{ki}.

    Flat transport gives :math:`H = I`. The metric reports

    .. math::
        \|H - I\|_F

    averaged over a random sample of triangles, batched over B. The break-out
    by merged vs singleton super-blocks is the cross-coupling-specific
    signal.

    Args:
        omega_pairwise: ``[(Omega_h, Omega_h_inv)] × H_super``, each of shape
            ``(B, N, N, d_h, d_h)``. Output of
            :func:`vfe.omega_direct.compute_pairwise_omega_from_endpoints`
            or the flat path in :mod:`vfe.attention.compute_gauge_transport`.
        cfg: VFEConfig.
        triangles: Optional explicit list of (i, j, k) tuples. When None, a
            random sample of size ``n_random_triangles`` is drawn from the
            full N^3 triangle set.
        n_random_triangles: Number of random triangles when ``triangles`` is
            None.

    Returns:
        Dict with::

            'holonomy_norm_total'     : mean over all super-blocks
            'holonomy_norm_merged'    : mean over merged super-blocks (NaN if none)
            'holonomy_norm_singleton' : mean over singleton super-blocks (NaN if none)
    """
    H_super = len(omega_pairwise)
    if H_super == 0:
        return {
            'holonomy_norm_total': 0.0,
            'holonomy_norm_merged': float('nan'),
            'holonomy_norm_singleton': float('nan'),
        }
    Om_h, _ = omega_pairwise[0]
    B, N, _, _, _ = Om_h.shape
    if triangles is None:
        # Random sample of distinct (i, j, k) triangles.
        # Allow self-loops degenerately — those contribute identity Omega
        # at the diagonal and inflate the singleton-block baseline slightly.
        # Acceptable for a diagnostic.
        triangles = [
            (
                int(torch.randint(N, (1,)).item()),
                int(torch.randint(N, (1,)).item()),
                int(torch.randint(N, (1,)).item()),
            )
            for _ in range(n_random_triangles)
        ]

    super_dims = cfg.effective_block_dims
    _, n_heads, d_head = cfg.irrep_spec[0]
    merged_h = [i for i, d in enumerate(super_dims) if d > d_head]
    singleton_h = [i for i, d in enumerate(super_dims) if d == d_head]

    per_h_norm: List[float] = []
    for h, (Om_h, _Om_inv) in enumerate(omega_pairwise):
        d_h = super_dims[h]
        I_h = torch.eye(d_h, device=Om_h.device, dtype=Om_h.dtype)
        accum: List[float] = []
        for (i, j, k) in triangles:
            A = Om_h[:, i, j]              # (B, d, d)
            Bm = Om_h[:, j, k]
            C = Om_h[:, k, i]
            H_ijk = A @ Bm @ C             # (B, d, d)
            res = H_ijk - I_h
            accum.append(float((res ** 2).sum(dim=(-2, -1)).sqrt().mean().item()))
        per_h_norm.append(float(np.mean(accum)) if accum else 0.0)

    total_mean = float(np.mean(per_h_norm))
    merged_mean = (
        float(np.mean([per_h_norm[h] for h in merged_h])) if merged_h else float('nan')
    )
    singleton_mean = (
        float(np.mean([per_h_norm[h] for h in singleton_h])) if singleton_h else float('nan')
    )
    return {
        'holonomy_norm_total': total_mean,
        'holonomy_norm_merged': merged_mean,
        'holonomy_norm_singleton': singleton_mean,
    }


# -----------------------------------------------------------------------------
# Convenience: collect all metrics
# -----------------------------------------------------------------------------


def collect_cross_coupling_metrics(
    *,
    phi: torch.Tensor,
    sigma: torch.Tensor,
    cfg,
    omega: Optional[torch.Tensor] = None,
    omega_pairwise: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    kl_per_block: Optional[torch.Tensor] = None,
    beta_per_block: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
) -> Dict[str, object]:
    r"""Compute every cross-coupling diagnostic that has enough inputs.

    Any input that is None is silently skipped — the function only returns
    metrics it could actually compute. Convenient for the training loop:
    pass whatever artifacts the current step has and log the resulting dict
    via ``log_metrics(**out)``.
    """
    out: Dict[str, object] = {}
    out.update(phi_energy_partition(phi, cfg))
    out['per_super_block_effective_rank'] = per_super_block_effective_rank(sigma, cfg)
    if omega is not None:
        out['omega_block_strength'] = omega_block_strength(omega, cfg)
    if kl_per_block is not None:
        out.update(cross_block_kl_share(kl_per_block, cfg, mask=mask))
    if beta_per_block is not None:
        out['per_super_block_attention_entropy'] = per_super_block_attention_entropy(
            beta_per_block, cfg,
        )
    if omega_pairwise is not None:
        out.update(holonomy_norm_cross_blocks(omega_pairwise, cfg))
    return out
