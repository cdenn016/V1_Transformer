"""
KL Divergence Computation -- Unified Module
===========================================

Consolidates 9 previously scattered KL matrix computation variants into a
single parametric entry point with three focused kernel functions.

Three covariance modes:
    DENSE          -- full (B, N, K, K) covariance; Cholesky-based KL
    DIAGONAL       -- diagonal (B, N, K) covariance; closed-form, O(N^2K)
    BLOCK_DIAGONAL -- block-diagonal structure; delegates to fused kernels
                     in gauge_utils.py, reducing CUDA launches to O(unique dims)

Chunking is handled entirely inside ``compute_kl_matrix``; the three kernel
functions contain only the core mathematics.

Gauge equivariance invariant: transported covariance always uses the sandwich
product ``Omega @ Sigma @ Omega.T``.  This module never touches that transport
itself -- the kernels receive already-transported tensors (mu_t, sigma_t).

Usage
-----
>>> from transformer.core.kl_computation import compute_kl_matrix, KLMode, safe_kl_clamp
>>> kl = compute_kl_matrix(mu_q, sigma_q, mu_t, sigma_t, mode=KLMode.DIAGONAL)
"""

from enum import Enum
from typing import List, Optional

import torch
from math_utils.numerical_monitor import record as _nr

from transformer.core.gauge_utils import (
    fused_block_diagonal_kl_diag,
    fused_block_diagonal_kl_full,
    fused_block_matrix_exp_pairs,
)


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    "KLMode",
    "compute_kl_matrix",
    "safe_kl_clamp",
]


class KLMode(Enum):
    """Covariance representation used when computing the KL matrix."""
    DENSE = "dense"
    DIAGONAL = "diagonal"
    BLOCK_DIAGONAL = "block_diagonal"


# =============================================================================
# Utility
# =============================================================================

def safe_kl_clamp(
    kl: torch.Tensor,
    kl_max: float = 100.0,
    propagate_nonfinite: bool = False,
) -> torch.Tensor:
    r"""Clamp a KL tensor to a finite, non-negative range.

    Default behaviour (``propagate_nonfinite=False``): applies
    ``clamp(0, kl_max)`` then replaces NaN/+inf with ``kl_max`` and -inf
    with 0.  Using ``nan=kl_max`` (repulsive) rather than ``nan=0.0``
    (attractive) ensures that numerically degenerate pairs are ignored
    by the downstream softmax rather than attended to.

    When ``propagate_nonfinite=True``: preserves NaN/±inf in the output
    (still applies the upper ``kl_max`` clamp on finite entries and
    zeros negatives).  Use this in diagnostic runs to make divergence
    loud instead of masking it — any non-finite KL will poison the
    softmax and trip the training loop's ``assert_finite_loss`` guard
    rather than saturating silently.

    Args:
        kl: Tensor of (possibly un-clamped) KL values.
        kl_max: Upper bound.  Default 100.0.
        propagate_nonfinite: If True, do NOT replace NaN/inf with
            ``kl_max``; let them propagate.  Default False.

    Returns:
        Clamped tensor, same shape and device as *kl*.
    """
    # Saturation counter: how many entries hit the NaN/inf → kl_max path,
    # or the upper clamp ceiling. Observational only.
    _nonfinite = ~torch.isfinite(kl)
    if _nonfinite.any():
        _nr("kl_nonfinite", count=int(_nonfinite.sum().item()))
    _at_ceiling = kl >= kl_max
    if _at_ceiling.any():
        _nr("kl_saturated", count=int(_at_ceiling.sum().item()))
    if propagate_nonfinite:
        # Preserve NaN/±inf; only clamp finite entries to [0, kl_max].
        finite_clamped = kl.clamp(min=0.0, max=kl_max)
        return torch.where(_nonfinite, kl, finite_clamped)
    kl = kl.clamp(min=0.0, max=kl_max)
    return kl.nan_to_num(nan=kl_max, posinf=kl_max, neginf=0.0)


# =============================================================================
# Kernel Functions -- pure math, no chunking logic
# =============================================================================

def _kl_kernel_dense(
    mu_q: torch.Tensor,      # (..., K) query means
    sigma_q: torch.Tensor,   # (..., K, K) query covariances
    mu_t: torch.Tensor,      # (..., K) transported key means
    sigma_t: torch.Tensor,   # (..., K, K) transported key covariances
    kl_max: float,
    eps: float,
    alpha_div: float = 1.0,
    exp_phi_q: torch.Tensor = None,   # (..., K, K) local frame at q (optional)
    exp_phi_t: torch.Tensor = None,   # (..., K, K) local frame at t (optional)
    sigma_floor: Optional[float] = None,
    spd_floor_mode: str = 'eigclamp',
    enable_spd_diagnostics: bool = False,
    propagate_nonfinite: bool = False,
) -> torch.Tensor:
    r"""Full-covariance KL or Rényi :math:`\alpha`-divergence.

    When ``alpha_div == 1.0`` (default), computes the standard KL divergence:

    .. math::
        \mathrm{KL}(\mathcal{N}(\mu_q, \Sigma_q) \,\|\, \mathcal{N}(\mu_t, \Sigma_t))
        = \tfrac{1}{2}\bigl(\operatorname{tr}(\Sigma_t^{-1}\Sigma_q)
          + (\mu_t-\mu_q)^\top \Sigma_t^{-1}(\mu_t-\mu_q)
          - K + \log|\Sigma_t| - \log|\Sigma_q|\bigr)

    Implemented via Cholesky for numerical stability.  Falls back to
    progressive diagonal regularisation on near-singular inputs.

    When ``alpha_div != 1.0``, computes the Rényi :math:`\alpha`-divergence
    using the blended covariance :math:`\tilde{\Sigma} = (1-\alpha)\Sigma_q +
    \alpha\Sigma_t`:

    .. math::
        D_\alpha(q \| p) = \tfrac{1}{2}\Bigl[
            \alpha \cdot \delta\mu^\top \tilde{\Sigma}^{-1} \delta\mu
            + \tfrac{1}{\alpha-1}\bigl(
                (1-\alpha)\log|\Sigma_q| + \alpha\log|\Sigma_t|
                - \log|\tilde{\Sigma}|
            \bigr)
        \Bigr]

    The limit :math:`\alpha \to 1` recovers the standard KL divergence.

    Args:
        mu_q: (..., K) query means.
        sigma_q: (..., K, K) query covariances.
        mu_t: (..., K) transported key means.
        sigma_t: (..., K, K) transported key covariances.
        kl_max: Clamp ceiling (typically ``max(100, 20*K)``).
        eps: Regularisation floor.
        alpha_div: Order of the Rényi divergence.  1.0 gives standard KL.

    Returns:
        kl: (...,) non-negative divergence values.
    """
    K = mu_q.shape[-1]
    device = mu_q.device
    orig_dtype = mu_q.dtype

    # Force float32: Cholesky, solve_triangular, and log-det all break in fp16.
    mu_q = mu_q.float()
    sigma_q = sigma_q.float()
    mu_t = mu_t.float()
    sigma_t = sigma_t.float()

    I = torch.eye(K, device=device, dtype=torch.float32)
    # Optional gauge-covariant ridge: use eps * (g g^T) so regularized Σ
    # transforms exactly as Σ under h Σ h^T. Falls back to eps * I.
    if exp_phi_q is not None:
        _gq = exp_phi_q.to(dtype=torch.float32)
        R_q = _gq @ _gq.transpose(-1, -2)
    else:
        R_q = I
    if exp_phi_t is not None:
        _gt = exp_phi_t.to(dtype=torch.float32)
        R_t = _gt @ _gt.transpose(-1, -2)
    else:
        R_t = I
    sigma_q_reg = sigma_q + eps * R_q
    sigma_t_reg = sigma_t + eps * R_t

    # Eigenvalue floor in the transported (post-sandwich) frame. This bounds
    # κ(Σ_t) before the Cholesky, which otherwise relies on 5-round jitter
    # escalation starting at eps and is unaware of the E-step σ_p floor.
    if sigma_floor is not None and spd_floor_mode == 'eigclamp':
        from transformer.core.vfe_utils import spd_eigfloor as _spd_eigfloor
        sigma_q_reg = _spd_eigfloor(sigma_q_reg, sigma_floor, exp_phi=exp_phi_q)
        sigma_t_reg = _spd_eigfloor(sigma_t_reg, sigma_floor, exp_phi=exp_phi_t)

    if enable_spd_diagnostics:
        # O(B·N·N·K³) per call — guarded behind flag.
        try:
            with torch.no_grad():
                eig_t = torch.linalg.eigvalsh(sigma_t_reg)
                eig_q = torch.linalg.eigvalsh(sigma_q_reg)
                _nr("spd_eig_min_t", value=float(eig_t.min().item()))
                _nr("spd_eig_min_q", value=float(eig_q.min().item()))
                _nr("spd_cond_t", value=float((eig_t.max() / eig_t.clamp(min=1e-30).min()).item()))
        except RuntimeError:
            _nr("spd_diagnostic_eigh_fail")

    # NaN guard: transported covariances can contain NaN when phi is very large.
    # We track which pairs have NaN and set their KL to kl_max after
    # computation, rather than replacing sigma_t with identity (which
    # would produce an arbitrary finite KL inconsistent with safe_kl_clamp's
    # NaN→kl_max policy).
    nan_mask = torch.isnan(sigma_t_reg).any(dim=-1).any(dim=-1)
    if nan_mask.any():
        _nr("nan_replace")
        # Replace with identity so Cholesky doesn't crash, but we'll
        # overwrite the KL values for these pairs with kl_max below.
        sigma_t_reg = torch.where(
            nan_mask.unsqueeze(-1).unsqueeze(-1),
            I.expand_as(sigma_t_reg),
            sigma_t_reg,
        )

    def _cholesky_with_fallback(mat: torch.Tensor) -> torch.Tensor:
        # TODO: when spd_floor_mode='eigclamp' with sigma_floor set, the
        #       pre-clamp above guarantees κ(mat) ≤ σ_max²/floor, and this
        #       5-round ridge escalation becomes dead code. Retained as a
        #       safety net for callers that do not thread sigma_floor.
        #       Audit call sites and remove the escalation path once every
        #       production caller is eigclamp-gated.
        L, info = torch.linalg.cholesky_ex(mat)
        if not info.any():
            return L
        # Some matrices failed -- apply progressive regularization
        reg = eps
        for _ in range(5):
            reg *= 10.0
            mat_reg = mat + (reg - eps) * I
            mat_reg = 0.5 * (mat_reg + mat_reg.transpose(-1, -2))
            L, info = torch.linalg.cholesky_ex(mat_reg)
            if not info.any():
                _nr("chol_recover")
                return L
        # Total failure after 5 rounds of 10x escalation (max reg ≈ 1e-1).
        # The previous implementation returned chol(I + eps*I) here, which
        # silently produced KL ≈ 0 for the whole batch (trace ≈ K,
        # Mahalanobis ≈ 0, logdet ≈ 0) and poisoned the M-step with a zero
        # coupling signal.  Raise instead: the outer try/except in
        # _kl_kernel_dense catches RuntimeError and converts it to a NaN
        # KL tensor, which propagates loudly to the training loss and
        # halts training before the silent corruption accumulates.
        _nr("chol_fail")
        raise RuntimeError(
            f"Cholesky failed after 5 regularization rounds (max reg={reg:.1e}). "
            f"Likely cause: extreme phi values producing NaN/Inf in transported "
            f"covariances, or a sigma_p that has collapsed below the floor. "
            f"Check transformer/core/variational_ffn.py E-step sigma handling."
        )

    try:
        if abs(alpha_div - 1.0) < 1e-6:
            # Standard KL divergence.
            L_p = _cholesky_with_fallback(sigma_t_reg)

            # Trace term: tr(Sigma_t^{-1} Sigma_q)
            Y = torch.linalg.solve_triangular(L_p, sigma_q_reg, upper=False)
            Z = torch.linalg.solve_triangular(L_p.transpose(-1, -2), Y, upper=True)
            trace_term = torch.diagonal(Z, dim1=-2, dim2=-1).sum(dim=-1)

            # Mahalanobis term
            delta_mu = mu_t - mu_q
            v = torch.linalg.solve_triangular(
                L_p, delta_mu.unsqueeze(-1), upper=False
            ).squeeze(-1)
            mahal_term = torch.sum(v ** 2, dim=-1)

            # Log-det terms
            logdet_p = 2.0 * torch.sum(
                torch.log(torch.diagonal(L_p, dim1=-2, dim2=-1).clamp(min=1e-12)), dim=-1
            )
            L_q = _cholesky_with_fallback(sigma_q_reg)
            logdet_q = 2.0 * torch.sum(
                torch.log(torch.diagonal(L_q, dim1=-2, dim2=-1).clamp(min=1e-12)), dim=-1
            )

            kl = 0.5 * (trace_term + mahal_term - K + logdet_p - logdet_q)
        else:
            # Rényi α-divergence for full covariance.
            # Blended covariance: Σ̃ = (1-α)Σ_q + α·Σ_t
            sigma_blend = (1.0 - alpha_div) * sigma_q_reg + alpha_div * sigma_t_reg
            sigma_blend = 0.5 * (sigma_blend + sigma_blend.transpose(-1, -2))  # symmetrize

            L_blend = _cholesky_with_fallback(sigma_blend)

            # Mahalanobis: α · δμᵀ Σ̃⁻¹ δμ, where δμ = μ_t - μ_q
            delta_mu = mu_t - mu_q  # (..., K)
            v = torch.linalg.solve_triangular(
                L_blend, delta_mu.unsqueeze(-1), upper=False
            ).squeeze(-1)  # (..., K)
            mahal_term = alpha_div * torch.sum(v ** 2, dim=-1)  # (...,)

            # Log-det term: ((1-α)log|Σ_q| + α·log|Σ_t| - log|Σ̃|) / (α-1)
            L_q = _cholesky_with_fallback(sigma_q_reg)
            L_t = _cholesky_with_fallback(sigma_t_reg)

            logdet_q = 2.0 * torch.sum(
                torch.log(torch.diagonal(L_q, dim1=-2, dim2=-1).clamp(min=1e-12)), dim=-1
            )
            logdet_t = 2.0 * torch.sum(
                torch.log(torch.diagonal(L_t, dim1=-2, dim2=-1).clamp(min=1e-12)), dim=-1
            )
            logdet_blend = 2.0 * torch.sum(
                torch.log(torch.diagonal(L_blend, dim1=-2, dim2=-1).clamp(min=1e-12)), dim=-1
            )

            logdet_term = (
                (1.0 - alpha_div) * logdet_q + alpha_div * logdet_t - logdet_blend
            ) / (alpha_div - 1.0)

            kl = 0.5 * (mahal_term + logdet_term)

        kl = safe_kl_clamp(kl, kl_max, propagate_nonfinite=propagate_nonfinite)
        # Enforce NaN-flagged pairs: either mask to kl_max (default) or let
        # them propagate for loud divergence under diagnostic mode.
        if nan_mask.any():
            if propagate_nonfinite:
                _nan_val = torch.tensor(float('nan'), device=kl.device, dtype=kl.dtype)
                kl = torch.where(nan_mask, _nan_val, kl)
            else:
                kl = torch.where(nan_mask, torch.tensor(kl_max, device=kl.device, dtype=kl.dtype), kl)
        return kl.to(orig_dtype)

    except RuntimeError as exc:
        # Cholesky failed totally even after escalating regularization.
        # Return NaN KL for the whole chunk: this propagates loudly to
        # the training loss (triggering NaN loss and early termination)
        # rather than silently poisoning the M-step with a zero coupling
        # signal.  Emit a warning with the underlying exception message
        # so the failure is visible in the training log.
        import warnings
        warnings.warn(
            f"_kl_kernel_dense: Cholesky fallback exhausted, returning NaN KL "
            f"chunk.  Underlying error: {exc}",
            RuntimeWarning,
            stacklevel=3,
        )
        # Shape matches a reduction over the last two covariance dims.
        kl_shape = mu_q.shape[:-1]
        return torch.full(
            kl_shape, float('nan'),
            device=mu_q.device, dtype=orig_dtype,
        )


def _kl_kernel_diagonal(
    mu_q: torch.Tensor,    # (..., K) query means
    sigma_q: torch.Tensor, # (..., K) query diagonal variances
    mu_t: torch.Tensor,    # (..., K) transported key means
    sigma_t: torch.Tensor, # (..., K) transported key diagonal variances
    kl_max: float,
    eps: float,
    alpha_div: float = 1.0,
) -> torch.Tensor:
    r"""Diagonal-covariance KL or Rényi :math:`\alpha`-divergence.

    **Standard KL** (``alpha_div == 1.0``, default):

    .. math::
        \mathrm{KL}(q \,\|\, p)
        = \tfrac{1}{2}\!\left(\sum_k \frac{s_k}{t_k}
          + \sum_k \frac{(\mu_t^k - \mu_q^k)^2}{t_k}
          - K
          + \sum_k \log\frac{t_k}{s_k}\right)

    where :math:`s_k = \sigma_q^k` and :math:`t_k = \sigma_t^k`.
    :math:`O(K)` per pair — no Cholesky, no matrix inversion.

    **Rényi** :math:`\alpha`-**divergence** (``alpha_div != 1.0``):

    For diagonal Gaussians the order-:math:`\alpha` Rényi divergence admits a
    closed form.  Defining the blended variance
    :math:`\tilde\sigma_k = (1-\alpha)s_k + \alpha t_k`:

    .. math::
        D_\alpha(q \,\|\, p)
        = \frac{1}{2}\!\left[
            \alpha \sum_k \frac{(\mu_t^k - \mu_q^k)^2}{\tilde\sigma_k}
            + \frac{1}{\alpha-1}\sum_k
              \Bigl((1-\alpha)\log s_k + \alpha\log t_k - \log\tilde\sigma_k\Bigr)
          \right]

    The limit :math:`\alpha\to 1` recovers the standard KL; a Taylor-expansion
    branch is used when ``abs(alpha_div - 1.0) < 1e-6`` to ensure smooth
    numerical behaviour near :math:`\alpha = 1`.

    For :math:`\alpha > 1` the blend :math:`\tilde\sigma_k` can approach zero
    when :math:`s_k \gg t_k` (large query variance), since
    :math:`(1-\alpha) s_k + \alpha t_k < 0` requires
    :math:`s_k > \alpha t_k / (\alpha - 1)`.  ``sigma_blend`` is clamped
    to ``eps`` to prevent division by zero.

    Reference: Rényi (1961), *On measures of entropy and information*,
    Proc. 4th Berkeley Sympos. Math. Statist. Probab., Vol. 1, pp. 547–561.

    Args:
        mu_q: (..., K) query means.
        sigma_q: (..., K) query diagonal variances (positive).
        mu_t: (..., K) transported key means.
        sigma_t: (..., K) transported key diagonal variances (positive).
        kl_max: Clamp ceiling.
        eps: Floor applied to variances and ``sigma_blend`` before
            division or logarithm.
        alpha_div: Rényi order :math:`\alpha`.  Default ``1.0`` recovers
            the standard KL divergence exactly (via the :math:`\alpha\to 1`
            branch).  Must be positive; values :math:`\alpha \leq 0` are
            not supported.

    Returns:
        kl: (...,) non-negative divergence values.
    """
    K = mu_q.shape[-1]
    orig_dtype = mu_q.dtype

    # Force float32 for sigma divisions and logs to survive AMP float16.
    with torch.amp.autocast('cuda', enabled=False):
        mu_q = mu_q.float()
        sigma_q = sigma_q.float().clamp(min=eps)
        mu_t = mu_t.float()
        sigma_t = sigma_t.float().clamp(min=eps)

        if abs(alpha_div - 1.0) < 1e-6:
            # Standard KL: closed-form, identical to pre-alpha behaviour.
            trace_term = (sigma_q / sigma_t).sum(dim=-1)
            delta = mu_t - mu_q
            mahal_term = ((delta ** 2) / sigma_t).sum(dim=-1)
            logdet_term = (torch.log(sigma_t) - torch.log(sigma_q)).sum(dim=-1)
            kl = 0.5 * (trace_term + mahal_term - K + logdet_term)
        else:
            # Rényi alpha-divergence for diagonal Gaussians.
            # Blended variance: sigma_blend_k = (1-alpha)*s_k + alpha*t_k.
            # Essential clamp: for alpha > 1, (1-alpha)*s_k is negative, so
            # sigma_blend can dip below zero when s_k >> t_k.
            sigma_blend = (
                (1.0 - alpha_div) * sigma_q + alpha_div * sigma_t
            ).clamp(min=eps)  # (..., K)

            delta = mu_t - mu_q  # (..., K)

            # Mahalanobis term: alpha * sum_k delta_k^2 / sigma_blend_k
            mahal_term = (alpha_div * (delta ** 2) / sigma_blend).sum(dim=-1)

            # Log-determinant term (scalar per sample after summing over k):
            #   [(1-alpha)*log(s_k) + alpha*log(t_k) - log(sigma_blend_k)] / (alpha-1)
            logdet_per_dim = (
                (1.0 - alpha_div) * torch.log(sigma_q)
                + alpha_div * torch.log(sigma_t)
                - torch.log(sigma_blend)
            )  # (..., K)
            logdet_term = logdet_per_dim.sum(dim=-1) / (alpha_div - 1.0)

            kl = 0.5 * (mahal_term + logdet_term)

        kl = safe_kl_clamp(kl, kl_max)

    return kl.to(orig_dtype)


def _kl_kernel_block_diagonal(
    mu_q: torch.Tensor,
    sigma_q: torch.Tensor,
    block_exp_pairs: list,
    irrep_dims: List[int],
    diagonal_sigma: bool,
    kl_max: float,
    eps: float,
    alpha_div: float = 1.0,
    sigma_floor: Optional[float] = None,
) -> torch.Tensor:
    r"""Block-diagonal KL or Rényi :math:`\alpha`-divergence -- delegates to fused kernels.

    Exploits block-diagonal structure of the gauge group representation:
        D_α(q || p) = sum_b D_α(q_b || p_b)

    Delegates to the fused batch kernels in ``gauge_utils.py`` which group
    same-sized blocks together and process them with a single matrix_exp call,
    reducing CUDA kernel launches from O(num_blocks) to O(num_unique_dims).

    Args:
        mu_q: (B, N, K) belief means.
        sigma_q: (B, N, K) diagonal variances or (B, N, K, K) full covariance.
        block_exp_pairs: list of ``(exp_phi, exp_neg_phi)`` per block as returned
            by ``fused_block_matrix_exp_pairs``.
        irrep_dims: Block dimension list [d_1, d_2, ...] summing to K.
        diagonal_sigma: If True, ``sigma_q`` is (B, N, K); if False, (B, N, K, K).
        kl_max: Not used directly; the fused kernels use their own internal ceiling
            derived from K.  Kept for API symmetry.
        eps: Numerical stability floor passed to the fused kernels.
        alpha_div: Rényi divergence order (default 1.0 = KL).  Supported for
            both ``diagonal_sigma=True`` and ``diagonal_sigma=False``.

    Returns:
        kl: (B, N, N) total divergence across all blocks.
    """
    if diagonal_sigma:
        return fused_block_diagonal_kl_diag(
            mu_q, sigma_q, block_exp_pairs, irrep_dims, eps=eps,
            alpha_div=alpha_div,
        )
    else:
        return fused_block_diagonal_kl_full(
            mu_q, sigma_q, block_exp_pairs, irrep_dims, eps=eps,
            alpha_div=alpha_div, sigma_floor=sigma_floor,
        )


# =============================================================================
# Public Entry Point
# =============================================================================

def compute_kl_matrix(
    mu_q: torch.Tensor,
    sigma_q: torch.Tensor,
    mu_transported: torch.Tensor,
    sigma_transported: torch.Tensor,
    mode: KLMode = KLMode.DENSE,
    chunk_size: Optional[int] = None,
    # Block-diagonal specific
    block_exp_pairs: Optional[list] = None,
    irrep_dims: Optional[List[int]] = None,
    # Shared options
    kl_max: float = 100.0,
    eps: float = 1e-6,
    alpha_divergence: float = 1.0,
    # Full-covariance SPD floor (only consumed by DENSE mode)
    sigma_floor: Optional[float] = None,
    spd_floor_mode: str = 'eigclamp',
    enable_spd_diagnostics: bool = False,
    propagate_nonfinite: bool = False,
) -> torch.Tensor:
    r"""Compute pairwise KL or Rényi :math:`\alpha`-divergence matrix.

    Returns the :math:`(B, N, N)` matrix
    :math:`D_\alpha(q_i \,\|\, \Omega_{ij}[q_j])` for all query–key pairs.

    When ``alpha_divergence == 1.0`` (default) this reduces exactly to the
    standard KL divergence :math:`\mathrm{KL}(q_i \,\|\, \Omega_{ij}[q_j])`,
    preserving full backward compatibility.

    For ``alpha_divergence != 1.0``, the Rényi :math:`\alpha`-divergence is
    used instead.  Support status by mode:

    - ``KLMode.DIAGONAL``: full support (closed-form expression in
      :func:`_kl_kernel_diagonal`).
    - ``KLMode.BLOCK_DIAGONAL`` with diagonal sigma: full support
      (``alpha_div`` passed to :func:`fused_block_diagonal_kl_diag`).
    - ``KLMode.BLOCK_DIAGONAL`` with full covariance: full support
      (``alpha_div`` passed to :func:`fused_block_diagonal_kl_full`).
    - ``KLMode.DENSE``: full support via blended covariance Cholesky.

    This is the single parametric entry point for all KL matrix computations.
    Callers are responsible for:

    1. Building transport operators (Omega, block_exp_pairs, or
       exp_phi/exp_neg_phi).
    2. Transporting key means and covariances: ``mu_transported``,
       ``sigma_transported``.
    3. Selecting the appropriate mode and passing compatible sigma shapes.

    For BLOCK_DIAGONAL mode, ``mu_transported`` and ``sigma_transported`` are
    *not* used; the kernel re-derives transport internally from
    ``block_exp_pairs`` via the fused gauge_utils kernels.  This is
    intentional: the fused path avoids materialising the full
    :math:`(B, N, N, d, d)` :math:`\Omega` tensor.

    Args:
        mu_q: (B, N, K) query belief means.
        sigma_q: Query covariances.
            DENSE:          (B, N, K, K) full covariance.
            DIAGONAL:       (B, N, K) diagonal variances.
            BLOCK_DIAGONAL: (B, N, K) diagonal or (B, N, K, K) full.
        mu_transported: (B, N, N, K) transported key means.
            Ignored for BLOCK_DIAGONAL mode.
        sigma_transported: Transported key covariances.
            DENSE:    (B, N, N, K, K).
            DIAGONAL: (B, N, N, K).
            Ignored for BLOCK_DIAGONAL mode.
        mode: KLMode enum selecting the computation kernel.
        chunk_size: If set, the (N, N) KL matrix is assembled in
            ``chunk_size x chunk_size`` tiles to bound peak memory.
            None = no chunking (all pairs in one vectorised pass).
            Ignored for BLOCK_DIAGONAL mode (fused kernels handle tiling
            internally with their own ``_tile_size`` parameter).
        block_exp_pairs: List of ``(exp_phi, exp_neg_phi)`` per irrep block.
            Required for BLOCK_DIAGONAL mode.
        irrep_dims: Block dimensions [d_1, ...] summing to K.
            Required for BLOCK_DIAGONAL mode.
        kl_max: Clamp ceiling for KL values.  Scaled automatically inside the
            fused block-diagonal kernels; passed through for DENSE/DIAGONAL.
        eps: Numerical stability floor.
        alpha_divergence: Order :math:`\alpha` of the Rényi divergence.
            Default ``1.0`` gives the standard KL divergence.  Supported
            for all modes and covariance shapes.

    Returns:
        kl_matrix: (B, N, N) pairwise divergence matrix.

    Raises:
        ValueError: If BLOCK_DIAGONAL mode is requested without
            ``block_exp_pairs`` or ``irrep_dims``.
        NotImplementedError: If an unsupported mode combination is requested.
    """
    if mode is KLMode.BLOCK_DIAGONAL:
        if block_exp_pairs is None or irrep_dims is None:
            raise ValueError(
                "KLMode.BLOCK_DIAGONAL requires both block_exp_pairs and irrep_dims."
            )
        diagonal_sigma = (sigma_q.dim() == 3)
        return _kl_kernel_block_diagonal(
            mu_q, sigma_q, block_exp_pairs, irrep_dims,
            diagonal_sigma=diagonal_sigma,
            kl_max=kl_max,
            eps=eps,
            alpha_div=alpha_divergence,
            sigma_floor=sigma_floor,
        )

    # DENSE and DIAGONAL modes: select kernel, optionally chunk.
    if chunk_size is None:
        return _compute_unchunked(
            mu_q, sigma_q, mu_transported, sigma_transported,
            mode, kl_max, eps, alpha_div=alpha_divergence,
            sigma_floor=sigma_floor,
            spd_floor_mode=spd_floor_mode,
            enable_spd_diagnostics=enable_spd_diagnostics,
            propagate_nonfinite=propagate_nonfinite,
        )
    else:
        return _compute_chunked(
            mu_q, sigma_q, mu_transported, sigma_transported,
            mode, chunk_size, kl_max, eps, alpha_div=alpha_divergence,
            sigma_floor=sigma_floor,
            spd_floor_mode=spd_floor_mode,
            enable_spd_diagnostics=enable_spd_diagnostics,
            propagate_nonfinite=propagate_nonfinite,
        )


# =============================================================================
# Internal: unchunked and chunked assembly
# =============================================================================

def _compute_unchunked(
    mu_q: torch.Tensor,
    sigma_q: torch.Tensor,
    mu_transported: torch.Tensor,   # (B, N, N, K)
    sigma_transported: torch.Tensor,
    mode: KLMode,
    kl_max: float,
    eps: float,
    alpha_div: float = 1.0,
    sigma_floor: Optional[float] = None,
    spd_floor_mode: str = 'eigclamp',
    enable_spd_diagnostics: bool = False,
    propagate_nonfinite: bool = False,
) -> torch.Tensor:
    """Compute full (B, N, N) divergence matrix in one vectorised pass."""
    B, N, K = mu_q.shape
    # Expand query beliefs over all key positions (views, no copy needed --
    # downstream kernels cast to float32 internally which creates contiguous copies)
    mu_i = mu_q[:, :, None, :].expand(-1, -1, N, -1)   # (B, N, N, K)

    if mode is KLMode.DIAGONAL:
        sigma_i = sigma_q[:, :, None, :].expand(-1, -1, N, -1)
        return _kl_kernel_diagonal(mu_i, sigma_i, mu_transported, sigma_transported,
                                   kl_max=kl_max, eps=eps, alpha_div=alpha_div)
    else:
        # DENSE
        sigma_i = sigma_q[:, :, None, :, :].expand(-1, -1, N, -1, -1)
        return _kl_kernel_dense(mu_i, sigma_i, mu_transported, sigma_transported,
                                kl_max=kl_max, eps=eps, alpha_div=alpha_div,
                                sigma_floor=sigma_floor,
                                spd_floor_mode=spd_floor_mode,
                                enable_spd_diagnostics=enable_spd_diagnostics,
                                propagate_nonfinite=propagate_nonfinite)


def _compute_chunked(
    mu_q: torch.Tensor,
    sigma_q: torch.Tensor,
    mu_transported: torch.Tensor,   # (B, N, N, K)
    sigma_transported: torch.Tensor,
    mode: KLMode,
    chunk_size: int,
    kl_max: float,
    eps: float,
    alpha_div: float = 1.0,
    sigma_floor: Optional[float] = None,
    spd_floor_mode: str = 'eigclamp',
    enable_spd_diagnostics: bool = False,
    propagate_nonfinite: bool = False,
) -> torch.Tensor:
    """Assemble (B, N, N) divergence matrix by processing chunk_size x chunk_size tiles."""
    B, N, K = mu_q.shape
    row_chunks: list = []

    for i_start in range(0, N, chunk_size):
        i_end = min(i_start + chunk_size, N)
        n_i = i_end - i_start

        mu_i_chunk = mu_q[:, i_start:i_end].contiguous()

        col_chunks: list = []
        for j_start in range(0, N, chunk_size):
            j_end = min(j_start + chunk_size, N)
            n_j = j_end - j_start

            # Slice pre-transported quantities for this chunk
            mu_t_chunk = mu_transported[:, i_start:i_end, j_start:j_end].contiguous()
            sigma_t_chunk = sigma_transported[:, i_start:i_end, j_start:j_end].contiguous()

            # Expand query beliefs to match chunk shape
            if mode is KLMode.DIAGONAL:
                sigma_i_chunk = sigma_q[:, i_start:i_end].contiguous()
                mu_i_exp = mu_i_chunk[:, :, None, :].expand(-1, -1, n_j, -1)
                sigma_i_exp = sigma_i_chunk[:, :, None, :].expand(-1, -1, n_j, -1)
                kl_chunk = _kl_kernel_diagonal(
                    mu_i_exp, sigma_i_exp, mu_t_chunk, sigma_t_chunk,
                    kl_max=kl_max, eps=eps, alpha_div=alpha_div,
                )
            else:
                # DENSE
                sigma_i_chunk = sigma_q[:, i_start:i_end].contiguous()
                mu_i_exp = mu_i_chunk[:, :, None, :].expand(-1, -1, n_j, -1)
                sigma_i_exp = sigma_i_chunk[:, :, None, :, :].expand(-1, -1, n_j, -1, -1)
                kl_chunk = _kl_kernel_dense(
                    mu_i_exp, sigma_i_exp, mu_t_chunk, sigma_t_chunk,
                    kl_max=kl_max, eps=eps, alpha_div=alpha_div,
                    sigma_floor=sigma_floor,
                    spd_floor_mode=spd_floor_mode,
                    enable_spd_diagnostics=enable_spd_diagnostics,
                    propagate_nonfinite=propagate_nonfinite,
                )

            col_chunks.append(kl_chunk)

        row_chunks.append(torch.cat(col_chunks, dim=2))  # (B, n_i, N)

    return torch.cat(row_chunks, dim=1)  # (B, N, N)
