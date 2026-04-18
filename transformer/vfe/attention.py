"""
VFE attention: stateless functions for KL-based gauge-covariant attention.

No class needed — pure functional computation.

Core thesis: attention weights are a Gibbs kernel on the statistical manifold
with gauge transport, not produced by learned W_Q, W_K projections.

    beta_ij = softmax(-KL(q_i || Omega_ij[q_j]) / kappa)
"""

from typing import Optional, List, Tuple

import torch

from transformer.core.gauge_utils import fused_block_matrix_exp_pairs
from transformer.core.attention import compute_attention_weights


# Cache keyed by generator tensor identity (id) — avoids per-call GPU sync.
# `generators` is a registered buffer set once at module construction; its
# skew-symmetry property never changes after init.
_SKEW_CACHE: dict = {}


def compute_gauge_transport(
    phi: torch.Tensor,
    generators: torch.Tensor,
    irrep_dims: List[int],
    enforce_orthogonal: bool = False,
    skew_symmetric: Optional[bool] = None,
) -> List[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
    r"""Compute block-diagonal matrix exponential pairs from phi.

    Returns per-block ``(exp(phi_i . G_h), exp(-phi_i . G_h))`` pairs
    used for gauge transport :math:`\Omega_{ij} = g_i g_j^{-1}`.

    Args:
        phi: ``(B, N, n_gen)`` gauge frame coordinates.
        generators: ``(n_gen, K, K)`` Lie algebra generators.
        irrep_dims: Block dimensions ``[d_1, d_2, ...]``.
        enforce_orthogonal: Project to SO(K) via Newton-Schulz.

    Returns:
        List of ``(exp_h, exp_neg_h)`` pairs, one per irrep block.
    """
    # Detect skew-symmetric generators (SO(N)) across ALL generators
    # (previously checked only generators[0], which mis-classifies mixed sets).
    if skew_symmetric is None:
        _key = id(generators)
        _skew = _SKEW_CACHE.get(_key)
        if _skew is None:
            _skew = bool(torch.allclose(
                generators, -generators.transpose(-1, -2), atol=1e-6
            ))
            _SKEW_CACHE[_key] = _skew
    else:
        _skew = skew_symmetric
    return fused_block_matrix_exp_pairs(
        phi, generators, irrep_dims,
        enforce_orthogonal=enforce_orthogonal,
        skew_symmetric=_skew,
    )


def compute_kl_attention(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    phi: torch.Tensor,
    generators: torch.Tensor,
    irrep_dims: List[int],
    kappa: float,
    mask: Optional[torch.Tensor] = None,
    use_rope: bool = False,
    rope_base: float = 10000.0,
    cached_block_exp_pairs: Optional[list] = None,
    enforce_orthogonal: bool = False,
    mask_self_attention: bool = True,
    exact_diagonal_transport: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Compute KL-based gauge-covariant attention weights.

    .. math::
        \beta_{ij} = \frac{\exp(-D_{ij}/\kappa)\, m_{ij}}{\sum_k \exp(-D_{ik}/\kappa)\, m_{ik}}

    where :math:`D_{ij} = \mathrm{KL}(q_i \| \Omega_{ij}[q_j])`.

    Args:
        mu: ``(B, N, K)`` belief means.
        sigma: ``(B, N, K)`` diagonal variances or ``(B, N, K, K)`` full.
        phi: ``(B, N, n_gen)`` gauge frame coordinates.
        generators: ``(n_gen, K, K)`` Lie algebra generators.
        irrep_dims: Block dimensions for block-diagonal KL.
        kappa: Attention temperature.
        mask: ``(B, N, N)`` causal mask (0 = masked).
        use_rope: Apply RoPE rotations to mu before KL.
        rope_base: RoPE frequency base.
        cached_block_exp_pairs: Precomputed block exp pairs (avoids recompute).
        enforce_orthogonal: Project to SO(K).

    Returns:
        beta: ``(B, N, N)`` attention weights.
        kl_matrix: ``(B, N, N)`` pairwise KL divergences.
    """
    is_diagonal = sigma.dim() == 3

    result = compute_attention_weights(
        mu_q=mu,
        sigma_q=sigma,
        phi=phi,
        generators=generators,
        kappa=kappa,
        epsilon=1e-8,
        mask=mask,
        return_kl=True,
        diagonal_covariance=is_diagonal,
        irrep_dims=irrep_dims,
        mask_self_attention=mask_self_attention,
        gauge_mode='learned',
        enforce_orthogonal=enforce_orthogonal,
        use_rope=use_rope,
        rope_base=rope_base,
        cached_block_exp_pairs=cached_block_exp_pairs,
        exact_diagonal_transport=exact_diagonal_transport,
    )

    if isinstance(result, tuple):
        beta, kl_matrix = result
    else:
        # compute_attention_weights should always return (beta, kl_matrix).
        # A non-tuple result indicates an upstream signature mismatch — fail
        # loudly rather than silently aliasing kl_matrix to beta (which
        # would corrupt any diagnostic that consumes kl_matrix).
        raise RuntimeError(
            "compute_attention_weights returned non-tuple result of type "
            f"{type(result).__name__}. Expected (beta, kl_matrix) tuple. "
            "Check transformer/core/attention.py:compute_attention_weights."
        )

    return beta, kl_matrix


def aggregate_beliefs(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    beta: torch.Tensor,
    block_exp_pairs: List[Tuple[torch.Tensor, Optional[torch.Tensor]]],
    irrep_dims: List[int],
    mode: str = 'mixture',
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Aggregate transported beliefs using attention weights.

    Computes the transported message mean (Section 4 of VFE_Transformer_Idea.md):

    .. math::
        \bar{\mu}_i = \sum_j \beta_{ij}\, \Omega_{ij}\, \mu_j

    where :math:`\Omega_{ij} = \exp(\phi_i \cdot G)\,\exp(-\phi_j \cdot G)`.

    **Mixture (moment-matching, Section 4.1)**:

    .. math::
        \bar{\Sigma}_i = \sum_j \beta_{ij} \bigl(
            \Omega_{ij}\Sigma_j\Omega_{ij}^\top
            + (\Omega_{ij}\mu_j - \bar{\mu}_i)(\cdots)^\top
        \bigr)

    Total uncertainty = average within-component + between-component spread.

    **Precision (product-of-experts, Section 4.2)**:

    .. math::
        \bar{\Lambda}_i = \sum_j \beta_{ij}\, \Lambda_{ij}, \quad
        \bar{\Sigma}_i = \bar{\Lambda}_i^{-1}

    where :math:`\Lambda_{ij} = (\Omega_{ij} \Sigma_j \Omega_{ij}^\top)^{-1}`.

    Args:
        mu: ``(B, N, K)`` belief means.
        sigma: ``(B, N, K)`` diagonal variances.
        beta: ``(B, N, N)`` attention weights.
        block_exp_pairs: Per-block ``(exp_h, exp_neg_h)`` from
            :func:`compute_gauge_transport`.
        irrep_dims: Block dimensions from ``VFEConfig.irrep_dims``.
        mode: ``'mixture'`` or ``'precision'``.

    Returns:
        mu_agg: ``(B, N, K)`` aggregated means.
        sigma_agg: ``(B, N, K)`` aggregated diagonal variances.
    """
    B, N, K = mu.shape
    is_diagonal = sigma.dim() == 3

    mu_agg_parts = []
    sigma_agg_parts = []
    block_start = 0

    for h, d_h in enumerate(irrep_dims):
        block_end = block_start + d_h
        exp_h, exp_neg_h = block_exp_pairs[h]  # (B, N, d_h, d_h)
        mu_h = mu[:, :, block_start:block_end]  # (B, N, d_h)

        # Transport: Omega_ij mu_j = exp_phi_i @ exp_neg_phi_j @ mu_j
        # Step 1: rotate mu_j into neutral frame
        rotated_mu_j = torch.einsum('bjkl,bjl->bjk', exp_neg_h, mu_h)  # (B, N, d_h)
        # Step 2: rotate into frame i for all (i, j) pairs
        transported_ij = torch.einsum('bikl,bjl->bijk', exp_h, rotated_mu_j)  # (B, N, N, d_h)
        # Step 3: weighted sum
        mu_agg_h = torch.einsum('bij,bijd->bid', beta, transported_ij)  # (B, N, d_h)
        mu_agg_parts.append(mu_agg_h)

        if is_diagonal:
            sigma_h = sigma[:, :, block_start:block_end]  # (B, N, d_h)

            # Transported covariance diagonal: diag(Omega @ diag(sigma) @ Omega^T)
            # = sum_l Omega_kl^2 * sigma_l
            omega_ij = torch.einsum('bikm,bjml->bijkl', exp_h, exp_neg_h)  # (B, N, N, d_h, d_h)
            sigma_transported_ij = torch.einsum(
                'bijkl,bjl->bijk', omega_ij ** 2, sigma_h,
            )  # (B, N, N, d_h)

            if mode == 'precision':
                # Precision aggregation: Lambda_bar = sum_j beta_ij * Lambda_ij
                precision_ij = 1.0 / sigma_transported_ij.clamp(min=1e-6)
                precision_agg = torch.einsum('bij,bijd->bid', beta, precision_ij)  # (B, N, d_h)
                sigma_agg_h = 1.0 / precision_agg.clamp(min=1e-6)
            else:
                # Mixture (moment-matching): E[Sigma] + Var[mu]
                sigma_within = torch.einsum(
                    'bij,bijd->bid', beta, sigma_transported_ij,
                )  # (B, N, d_h)
                mu_dev = transported_ij - mu_agg_h.unsqueeze(2)  # (B, N, N, d_h)
                sigma_between = torch.einsum(
                    'bij,bijd->bid', beta, mu_dev ** 2,
                )  # (B, N, d_h)
                sigma_agg_h = (sigma_within + sigma_between).clamp(min=1e-6)

            sigma_agg_parts.append(sigma_agg_h)

        block_start = block_end

    mu_agg = torch.cat(mu_agg_parts, dim=-1)
    sigma_agg = torch.cat(sigma_agg_parts, dim=-1) if sigma_agg_parts else sigma

    return mu_agg, sigma_agg


