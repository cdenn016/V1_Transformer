"""
VFE attention: stateless functions for KL-based gauge-covariant attention.

No class needed — pure functional computation.

Core thesis: attention weights are a Gibbs kernel on the statistical manifold
with gauge transport, not produced by learned W_Q, W_K projections.

    beta_ij = softmax(-KL(q_i || Omega_ij[q_j]) / kappa)

Implementation note — structural self-attractor at i = j.
==========================================================
Because ``Omega_ii = exp(phi_i) . exp(-phi_i) = I`` for every token (the
transport from a token to itself is the identity, independent of phi), the
self-pair KL is structurally zero:

    KL(q_i || Omega_ii q_i) = KL(q_i || q_i) = 0.

Under softmax(-KL / (kappa * sqrt(K))) the diagonal logit is therefore the
largest (or tied for largest) entry in every row, so unmasked self-attention
saturates onto the diagonal. The active ``train_vfe.py`` config sets
``mask_self_attention: False`` so the diagonal is allowed; the alternative
``mask_self_attention: True`` zeros the diagonal logit, which avoids the
self-attractor at the cost of changing the row-Lagrangian normalisation.
The standard dot-product attention literature does not have this issue
because ``q_i . k_i`` is not structurally zero.

Treat the unmasked diagonal as a "stay-put" gate (mass that doesn't propagate
through transport); document that empirical results under
``mask_self_attention=False`` rely on it.
"""

from typing import Optional, List, Tuple

import torch

from transformer.core.gauge_utils import fused_block_matrix_exp_pairs
from transformer.core.attention import compute_attention_weights


# Cache keyed by a content fingerprint of the generator bank — avoids
# per-call GPU sync. `generators` is a registered buffer set once at module
# construction; its skew-symmetry property never changes after init.
#
# The fingerprint `(data_ptr, shape, dtype, device)` is hazard-free across
# GC, in contrast to keying on `id(generators)` which Python may reuse for
# an unrelated tensor after the original is collected — that reuse would
# silently inherit a stale skew verdict.
#
# Bounded to ``_SKEW_CACHE_MAX`` entries with FIFO eviction so the cache
# cannot grow unboundedly across many-model sessions (e.g. ablation suites
# constructing hundreds of models in one process).
_SKEW_CACHE: dict = {}
_SKEW_CACHE_MAX = 16


def _skew_cache_key(g: torch.Tensor) -> tuple:
    # Content-stable: a small fingerprint of tensor data + shape + dtype/device.
    # Pure data_ptr keying is unsafe across generator-buffer lifetimes — the
    # allocator can recycle a freed ptr for an unrelated (non-skew) tensor of
    # the same shape, silently inheriting the cached skew verdict. Including
    # a cheap content fingerprint (sum + a few sampled entries) makes the key
    # invalidate when the underlying data changes even if the ptr does not.
    with torch.no_grad():
        _flat = g.reshape(-1)
        _n = _flat.numel()
        if _n == 0:
            fp = (0.0,)
        else:
            # Sum + first/middle/last entries — cheap (~O(n) sum, O(1) reads)
            # and well-distributed for collision avoidance across distinct
            # generator banks of the same shape.
            fp = (
                float(_flat.sum().detach().cpu().item()),
                float(_flat[0].detach().cpu().item()),
                float(_flat[_n // 2].detach().cpu().item()),
                float(_flat[-1].detach().cpu().item()),
            )
    return (fp, tuple(g.shape), g.dtype, str(g.device))


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
        _key = _skew_cache_key(generators)
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
    kappa: "float | torch.Tensor",
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


