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
# skew-symmetry property never changes after init. Bounded to ``_SKEW_CACHE_MAX``
# entries with FIFO eviction so the cache cannot grow unboundedly across
# many-model sessions (e.g. ablation suites constructing hundreds of models
# in one process). With a small live set, ``id()`` reuse after GC is harmless.
_SKEW_CACHE: dict = {}
_SKEW_CACHE_MAX = 16


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
            if len(_SKEW_CACHE) >= _SKEW_CACHE_MAX:
                _SKEW_CACHE.pop(next(iter(_SKEW_CACHE)))
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
    cached_block_exp_pairs: Optional[List[Tuple[torch.Tensor, Optional[torch.Tensor]]]] = None,
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


