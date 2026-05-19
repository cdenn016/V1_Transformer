"""
VFE Gradient Computation
========================

Extracted VFE gradient functions from variational_ffn.py.  These pure
functions compute variational free energy gradients for the E-step belief
update over Gaussian tuples (mu, Sigma, phi).

Exported public API
-------------------
- compute_vfe_gradients_gpu    — dispatcher: routes to correct path by covariance type
- compute_natural_gradient_gpu — Fisher-metric natural gradient projection

Private helpers (used internally and by compute_vfe_gradients_gpu)
-------------------------------------------------------------------
- _compute_vfe_gradients_block_diagonal       — full covariance, block-diagonal KL
- _compute_vfe_gradients_block_diagonal_diag  — diagonal covariance, block-diagonal KL
- _fused_attention_and_vfe_gradients_block_diag — fused beta+gradient, diagonal mode

The _VFE_GRAD_DEBUG module-level dict is re-exported from vfe_utils so that
callers can inspect gradient component norms without importing vfe_utils directly.
"""

import math
import torch
from typing import List, Optional, Tuple

from transformer.core.gauge_utils import (
    _cached_eye,
    stable_matrix_exp_pair,
    newton_schulz_orthogonalize,
    fused_block_matrix_exp_pairs,
)
from math_utils.numerical_monitor import record as _nr
from transformer.core.transport_ops import _apply_rope, _un_apply_rope_pair_outer, _apply_rope_to_covariance

import transformer.core.vfe_utils as _vfe_utils_mod
from transformer.core.vfe_utils import (
    kl_diagnostics_enabled as _kl_diagnostics_enabled,
    TRANSPORT_JITTER,
    KL_CEIL_BASE,
    KL_CEIL_SCALE,
    _grad_norm,
    _per_pos_stats,
    squeeze_trailing_singletons,
    _safe_spd_inv,
)

# =============================================================================
# Memory-Efficient VFE Gradient Helpers
# =============================================================================

def _compute_vfe_gradients_block_diagonal(
    mu_q: torch.Tensor,        # (B, N, K) belief means
    sigma_q: torch.Tensor,     # (B, N, K, K) full block-diagonal covariances
    mu_p: torch.Tensor,        # (B, N, K) prior means
    sigma_p: torch.Tensor,     # (B, N, K, K) prior covariances
    beta: torch.Tensor,        # (B, N, N) attention weights
    phi: torch.Tensor,         # (B, N, n_gen) gauge frames
    generators: torch.Tensor,  # (n_gen, K, K) generators
    alpha: 'float | torch.Tensor',
    lambda_belief: float,
    lambda_softmax: float,
    kappa: float,
    eps: float,
    irrep_dims: List[int],
    compute_sigma_align_grad: bool,
    enforce_orthogonal: bool = False,  # If True, enforce Ω ∈ SO(K) via Newton-Schulz
    alpha_c0: Optional[torch.Tensor] = None,  # (K,) for product-rule correction when alpha is learnable
    cached_block_exp_pairs: Optional[list] = None,
    use_rope: bool = False,
    rope_base: float = 10000.0,
    alpha_div: float = 1.0,
    gauge_covariant_ridge: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Block-diagonal VFE gradient computation for full covariance mode.

    Processes each irrep block separately to reduce memory from O(N^2 K^2) to
    O(N^2 * max(d_i^2)). Includes sigma softmax coupling
    (dBeta/dSigma) via per-block per-pair storage.

    Args:
        mu_q: Belief means (B, N, K).
        sigma_q: Full block-diagonal covariances (B, N, K, K).
        mu_p: Prior means (B, N, K).
        sigma_p: Prior covariances (B, N, K, K).
        beta: Attention weights (B, N, N).
        phi: Gauge frames (B, N, phi_dim), phi_dim = n_gen.
        generators: Lie algebra generators (n_gen, K, K).
        alpha: Self-coupling weight, scalar or (B, N, K) Bayesian precision.
        lambda_belief: Belief alignment weight.
        kappa: Temperature for softmax coupling.
        eps: Numerical stability floor.
        irrep_dims: Block dimensions [d_1, d_2, ...] for block-diagonal KL.
        compute_sigma_align_grad: Whether to compute dF/dSigma from alignment.
        enforce_orthogonal: If True, enforce Omega in SO(K) via Newton-Schulz.
        alpha_c0: (K,) softplus(raw_c0) for product-rule correction when alpha is learnable.
        cached_block_exp_pairs: Precomputed (exp_phi, exp_neg_phi) per block.
        use_rope: When True, the externally-supplied β was softmaxed from
            KL_RoPE (see compute_attention_weights with use_rope=True).  The
            μ softmax-coupling chain rule is then routed through
            ∂KL_RoPE/∂(R μ) and un-rotated by R(θ_i)^T per query.  Σ is left
            raw to preserve the diagonal helper's σ asymmetry convention.
        rope_base: RoPE base frequency, must match compute_attention_weights.

        alpha_div: Rényi divergence order (default 1.0 = KL).  Fully supported
            for full covariance; uses blended covariance
            :math:`\\tilde{\\Sigma} = (1-\\alpha_d)\\Sigma_q + \\alpha_d\\Sigma_p`.

    Returns:
        grad_mu: (B, N, K) gradient w.r.t. mu.
        grad_sigma: (B, N, K, K) gradient w.r.t. Sigma.
    """
    B, N, K = mu_q.shape
    device = mu_q.device
    dtype = mu_q.dtype

    # Initialize output gradients
    grad_mu = torch.zeros(B, N, K, device=device, dtype=dtype)
    grad_sigma = torch.zeros(B, N, K, K, device=device, dtype=dtype)

    # Build full-K block-diagonal local frame once when the covariant ridge
    # is enabled. exp_phi_full[:, n] = block_diag(exp_phi_h[:, n] for h).
    # Under a gauge change g → h g, this transforms as exp_phi_full → h exp_phi_full,
    # and ε · exp_phi_full exp_phi_full^T → h · (that) · h^T — gauge covariant.
    exp_phi_full = None
    if gauge_covariant_ridge and cached_block_exp_pairs is not None:
        exp_phi_full = torch.zeros(B, N, K, K, device=device, dtype=dtype)
        _bs = 0
        for _h, _d_h in enumerate(irrep_dims):
            _be = _bs + _d_h
            exp_phi_full[:, :, _bs:_be, _bs:_be] = cached_block_exp_pairs[_h][0].to(dtype=dtype)
            _bs = _be

    # =================================================================
    # 1. Self-Coupling Gradient (block-wise but simpler)
    # =================================================================
    delta_mu = mu_q - mu_p  # (B, N, K)
    sigma_q_inv = _safe_spd_inv(sigma_q, eps=eps, exp_phi=exp_phi_full)
    # For full covariance (4D), alpha (B,N,1) needs extra dim to broadcast with (B,N,K,K)
    alpha_4d = alpha.unsqueeze(-1) if isinstance(alpha, torch.Tensor) else alpha

    if abs(alpha_div - 1.0) < 1e-6:
        # Standard KL self-coupling gradient.
        # ∂KL(q||p)/∂μ_q = Σ_p⁻¹ (μ_q - μ_p)
        # ∂KL(q||p)/∂Σ_q = ½(Σ_p⁻¹ - Σ_q⁻¹)
        sigma_p_inv = _safe_spd_inv(sigma_p, eps=eps, exp_phi=exp_phi_full)
        grad_mu_self = alpha * torch.einsum('bnij,bnj->bni', sigma_p_inv, delta_mu)
        grad_sigma_self = alpha_4d * 0.5 * (sigma_p_inv - sigma_q_inv)

        # Product-rule correction for learnable alpha (full covariance):
        # ∂(α·KL)/∂θ = α·∂KL/∂θ + (∂α/∂θ)·KL
        # When α_k = c₀_k/(b₀_k + kl_k), ∂α_k/∂θ = -α_k²/c₀_k · ∂kl_k/∂θ
        if alpha_c0 is not None and isinstance(alpha, torch.Tensor):
            # Per-dimension KL proxy from diagonal elements
            prod_qp = torch.matmul(sigma_p_inv, sigma_q)  # (B, N, K, K)
            trace_k = prod_qp.diagonal(dim1=-2, dim2=-1)  # (B, N, K)
            sp_inv_delta = torch.einsum('bnij,bnj->bni', sigma_p_inv, delta_mu)
            mahal_k = delta_mu * sp_inv_delta  # (B, N, K)
            logdet_p = torch.linalg.slogdet(sigma_p.float())[1]  # (B, N)
            logdet_q = torch.linalg.slogdet(sigma_q.float())[1]  # (B, N)
            logdet_k = ((logdet_p - logdet_q) / K).unsqueeze(-1).expand_as(delta_mu)  # (B, N, K)
            kl_k = 0.5 * (trace_k + mahal_k - 1 + logdet_k).clamp(min=0.0)
            # Correction to mu gradient
            grad_mu_self = grad_mu_self - (alpha ** 2 / alpha_c0) * kl_k * torch.einsum('bnij,bnj->bni', sigma_p_inv, delta_mu)
            # Correction to sigma gradient (4D broadcast)
            correction_scale = ((alpha ** 2 / alpha_c0) * kl_k).unsqueeze(-1)  # (B, N, K, 1)
            grad_sigma_self = grad_sigma_self - correction_scale * 0.5 * (sigma_p_inv - sigma_q_inv)
    else:
        # Rényi α-divergence self-coupling gradient for full covariance.
        # Blended covariance: Σ̃ = (1-α_d)Σ_q + α_d·Σ_p
        #
        # ∂D_α/∂μ_q = α_d · Σ̃⁻¹ · (μ_q - μ_p)
        # ∂D_α/∂Σ_q = ½(Σ̃⁻¹ - Σ_q⁻¹) - ½·α_d·(1-α_d)·Σ̃⁻¹·ΔμΔμᵀ·Σ̃⁻¹
        sigma_blend = (1.0 - alpha_div) * sigma_q + alpha_div * sigma_p  # (B, N, K, K)
        I_K = _cached_eye(K, device, dtype)
        if exp_phi_full is not None:
            _R_K_blend = exp_phi_full @ exp_phi_full.transpose(-1, -2)
        else:
            _R_K_blend = I_K
        sigma_blend = sigma_blend + eps * _R_K_blend
        sigma_blend = 0.5 * (sigma_blend + sigma_blend.transpose(-1, -2))  # symmetrize
        sigma_blend_inv = _safe_spd_inv(sigma_blend, eps=eps, exp_phi=exp_phi_full)  # (B, N, K, K)

        # ∂D_α/∂μ_q = α_d · Σ̃⁻¹ · Δμ
        grad_mu_self = alpha * alpha_div * torch.einsum(
            'bnij,bnj->bni', sigma_blend_inv, delta_mu
        )  # (B, N, K)

        # ∂D_α/∂Σ_q = ½(Σ̃⁻¹ - Σ_q⁻¹) - ½·α_d·(1-α_d)·Σ̃⁻¹·ΔμΔμᵀ·Σ̃⁻¹
        grad_sigma_self = alpha_4d * 0.5 * (sigma_blend_inv - sigma_q_inv)

        # Outer product term: Σ̃⁻¹ Δμ Δμᵀ Σ̃⁻¹
        blend_inv_delta = torch.einsum(
            'bnij,bnj->bni', sigma_blend_inv, delta_mu
        )  # (B, N, K)
        outer_term = torch.einsum(
            'bni,bnj->bnij', blend_inv_delta, blend_inv_delta
        )  # (B, N, K, K)
        grad_sigma_self = (
            grad_sigma_self
            - alpha_4d * 0.5 * alpha_div * (1.0 - alpha_div) * outer_term
        )

        # Product-rule correction for learnable alpha (Rényi branch):
        # ∂(α·D_α)/∂θ = α·∂D_α/∂θ + (∂α/∂θ)·D_α.  With the now-consistent
        # schedule α_k = c₀_k / (b₀_k + D_α,k), ∂α_k/∂θ = -α_k²/c₀_k · ∂D_α,k/∂θ,
        # giving an additional self-coupling term -(α²/c₀)·D_α,k·∂D_α,k/∂θ.
        # Per-dim proxy for D_α,k uses the diagonal of blend / Σ_q / Σ_p
        # (matches the full-cov proxy in VariationalFFNDynamic.get_bayesian_alpha).
        if alpha_c0 is not None and isinstance(alpha, torch.Tensor):
            sigma_blend_diag = sigma_blend.diagonal(dim1=-2, dim2=-1).clamp(min=eps)  # (B, N, K)
            sigma_p_diag = sigma_p.diagonal(dim1=-2, dim2=-1).clamp(min=eps)          # (B, N, K)
            sigma_q_diag = sigma_q.diagonal(dim1=-2, dim2=-1).clamp(min=eps)          # (B, N, K)
            mahal_k = alpha_div * (delta_mu ** 2) / sigma_blend_diag                  # (B, N, K)
            logdet_k = (
                (1.0 - alpha_div) * torch.log(sigma_q_diag)
                + alpha_div * torch.log(sigma_p_diag)
                - torch.log(sigma_blend_diag)
            ) / (alpha_div - 1.0)
            d_alpha_k = 0.5 * (mahal_k + logdet_k).clamp(min=0.0)                    # (B, N, K)
            # Correction to mu gradient: -(α²/c₀) · D_α,k · α_d · Σ̃⁻¹·Δμ
            grad_mu_self = grad_mu_self - (alpha ** 2 / alpha_c0) * d_alpha_k * (
                alpha_div * torch.einsum('bnij,bnj->bni', sigma_blend_inv, delta_mu)
            )
            # Correction to sigma gradient (4D broadcast over the matrix axes)
            correction_scale = ((alpha ** 2 / alpha_c0) * d_alpha_k).unsqueeze(-1)   # (B, N, K, 1)
            grad_sigma_self = grad_sigma_self - correction_scale * 0.5 * (
                sigma_blend_inv - sigma_q_inv
                - alpha_div * (1.0 - alpha_div) * outer_term
            )

    grad_mu = grad_mu + grad_mu_self
    grad_sigma = grad_sigma + grad_sigma_self

    # Debug: self-coupling component norms
    if _vfe_utils_mod._VFE_GRAD_DEBUG is not None:
        _vfe_utils_mod._VFE_GRAD_DEBUG['grad_mu_self'] = _grad_norm(grad_mu_self)
        _vfe_utils_mod._VFE_GRAD_DEBUG['grad_sigma_self'] = _grad_norm(grad_sigma_self)
        _ps = _per_pos_stats(grad_sigma_self)
        _vfe_utils_mod._VFE_GRAD_DEBUG['grad_sigma_self_pos_mean'] = _ps[0]
        _vfe_utils_mod._VFE_GRAD_DEBUG['grad_sigma_self_pos_max'] = _ps[1]
        # sigma_p eigenvalue range (shows how tight priors are)
        sp_diag = torch.diagonal(sigma_p, dim1=-2, dim2=-1)
        _vfe_utils_mod._VFE_GRAD_DEBUG['sigma_p_min'] = sp_diag.min().item()
        _vfe_utils_mod._VFE_GRAD_DEBUG['sigma_p_max'] = sp_diag.max().item()
        # sigma_q eigenvalue range
        try:
            sq_eig = torch.linalg.eigvalsh(sigma_q)
            _vfe_utils_mod._VFE_GRAD_DEBUG['sigma_q_eig_min'] = sq_eig.min().item()
            _vfe_utils_mod._VFE_GRAD_DEBUG['sigma_q_eig_max'] = sq_eig.max().item()
        except (RuntimeError, torch.linalg.LinAlgError):
            _vfe_utils_mod._VFE_GRAD_DEBUG['sigma_q_eig_min'] = float('nan')
            _vfe_utils_mod._VFE_GRAD_DEBUG['sigma_q_eig_max'] = float('nan')

    # =================================================================
    # 2. Belief Alignment Gradient (block-diagonal + chunked processing)
    # =================================================================
    # When use_rope is active, the externally-supplied β was softmaxed from
    # KL_RoPE (computed via compute_attention_weights with use_rope=True).
    # The chain rule for ∂β/∂μ_raw must therefore go through ∂KL_RoPE/∂μ_raw
    # (= R(θ)^T · ∂KL_RoPE/∂(R μ)), NOT ∂KL_raw/∂μ.  We compute the rope-space
    # gradient in parallel and un-rotate it after the loop for the chain rule.
    #
    # IMPORTANT (RoPE σ asymmetry).  Mirroring the diagonal-cov helper and the
    # standard-transformer convention, only μ is rope-rotated; Σ is left raw
    # (Σ_j_t still uses only the LEARNED Ω, not R).  See `BlockConfig.rope_full_gauge`
    # for the experimental opt-in that implements the framework-consistent
    # full Σ rotation.
    if use_rope:
        # _apply_rope, _un_apply_rope_pair_outer imported at module level
        mu_q_rope = _apply_rope(mu_q, base=rope_base)
    else:
        mu_q_rope = mu_q

    # Precompute matrix exponentials — FUSED by dimension group
    if cached_block_exp_pairs is not None:
        _fused_pairs = cached_block_exp_pairs
    else:
        _fused_pairs = fused_block_matrix_exp_pairs(
            phi, generators, irrep_dims, enforce_orthogonal=enforce_orthogonal
        )
    block_exp_phi = [p[0] for p in _fused_pairs]
    block_exp_neg_phi = [p[1] for p in _fused_pairs]

    # Accumulators for alignment gradients
    grad_mu_align = torch.zeros_like(mu_q)
    grad_sigma_align = torch.zeros_like(sigma_q)

    # For KL values and gradients - accumulate per-pair data for softmax coupling.
    # NOTE: Full (B, N, N) and (B, N, N, K) tensors are needed because the softmax
    # coupling term requires all pairwise KL values and gradients simultaneously.
    # Chunking over query positions (i) still saves memory on intermediate Omega,
    # transported beliefs, and inverses - just not on the final accumulators.
    kl_values = torch.zeros(B, N, N, device=device, dtype=dtype)
    grad_kl_per_pair_full = torch.zeros(B, N, N, K, device=device, dtype=dtype)
    # ∂KL_RoPE/∂(R μ_i): un-rotated per i (after the loop) for chain rule.
    # Allocated only when RoPE is active so the no-rope path costs nothing.
    grad_kl_rope_per_pair = (
        torch.zeros(B, N, N, K, device=device, dtype=dtype) if use_rope else None
    )

    # Per-block per-pair sigma gradients for softmax coupling (Pass 2).
    # Memory: Σ_b B*N²*d_b² instead of B*N²*K² (~82× savings for typical irrep specs).
    grad_sigma_per_pair_blocks = None
    if compute_sigma_align_grad:
        grad_sigma_per_pair_blocks = [
            torch.zeros(B, N, N, d, d, device=device, dtype=dtype)
            for d in irrep_dims
        ]

    # Process all query positions (no chunking)
    C = N
    for i_start in range(0, N, C):
        i_end = min(i_start + C, N)
        C_actual = i_end - i_start

        # Process each irrep block for this chunk of query positions
        block_start = 0
        for block_idx, d in enumerate(irrep_dims):
            block_end = block_start + d

            # Extract block beliefs - use .contiguous() to create copies and avoid
            # inplace modification errors during backward pass
            mu_block = mu_q[:, :, block_start:block_end].contiguous()  # (B, N, d) raw
            mu_block_rope = (
                mu_q_rope[:, :, block_start:block_end].contiguous() if use_rope else mu_block
            )
            sigma_block = sigma_q[:, :, block_start:block_end, block_start:block_end].contiguous()  # (B, N, d, d)

            # Get chunked exponentials for query positions - use .contiguous() for same reason
            exp_phi_i = block_exp_phi[block_idx][:, i_start:i_end].contiguous()  # (B, C, d, d)
            exp_neg_phi_j = block_exp_neg_phi[block_idx].contiguous()  # (B, N, d, d)

            # Compute Omega for this chunk: (B, C, N, d, d)
            Omega_chunk = torch.einsum(
                'bikl,bjlm->bijkm',
                exp_phi_i, exp_neg_phi_j
            )  # (B, C, N, d, d)

            # NaN guard on Omega.  When φ drifts to extreme values during the
            # M-step (especially in gauge_param='phi' mode with killing
            # preconditioner), stable_matrix_exp_pair's norm-clamp can still
            # propagate NaN if the input φ contains NaN.  An unguarded NaN in
            # Omega corrupts mu_j_transported, sigma_j_transported, KL,
            # softmax, and gradients — poisoning the whole batch.  Replace
            # any NaN-containing pair (i, j) with identity so the offending
            # pair contributes zero KL and zero gradient instead.  Mirrors
            # the guard in _fused_attention_and_vfe_gradients_block_diag.
            # Unconditional torch.where avoids a host sync; clean inputs
            # are unchanged (isnan returns False everywhere).
            I_d = _cached_eye(d, device, dtype)
            _omega_nan = torch.isnan(Omega_chunk)
            if _kl_diagnostics_enabled() and bool(_omega_nan.any().item()):
                _nr("vfe_full_omega_nan")
            Omega_chunk = torch.where(_omega_nan, I_d.expand_as(Omega_chunk), Omega_chunk)

            # Transport means and covariances for this chunk.
            # When use_rope is active, we also transport the rope-rotated mu so
            # that the rope-space gradient ∂KL_RoPE/∂(R μ_i) can be computed
            # in parallel.  The covariance transport uses ONLY the learned Ω
            # (not R), matching the diagonal helper's σ asymmetry convention.
            mu_j_transported = torch.einsum('bijkl,bjl->bijk', Omega_chunk, mu_block)  # raw, (B, C, N, d)
            if use_rope:
                mu_j_transported_rope = torch.einsum(
                    'bijkl,bjl->bijk', Omega_chunk, mu_block_rope
                )  # rope, (B, C, N, d)
            sigma_j_transported = torch.einsum(
                'bijkl,bjlm,bijmn->bijkn',
                Omega_chunk, sigma_block, Omega_chunk.transpose(-1, -2)
            )  # (B, C, N, d, d)

            del Omega_chunk

            # Second-line NaN guards.  Even with a clean Omega, sigma_block
            # itself can be NaN (extreme M-step σ_p update).  clamp() does
            # not replace NaN, only out-of-range finite values, so we
            # explicitly mask here too.  Unconditional torch.where avoids
            # a host sync per chunk; counter incurs a sync only when the
            # diagnostics flag is enabled.
            _sigma_nan = torch.isnan(sigma_j_transported)
            if _kl_diagnostics_enabled() and bool(_sigma_nan.any().item()):
                _nr("vfe_full_sigma_t_nan")
            sigma_j_transported = torch.where(
                _sigma_nan, I_d.expand_as(sigma_j_transported), sigma_j_transported,
            )
            _mu_nan = torch.isnan(mu_j_transported)
            if _kl_diagnostics_enabled() and bool(_mu_nan.any().item()):
                _nr("vfe_full_mu_t_nan")
            mu_j_transported = torch.where(
                _mu_nan, torch.zeros_like(mu_j_transported), mu_j_transported,
            )
            if use_rope:
                _mu_r_nan = torch.isnan(mu_j_transported_rope)
                if _kl_diagnostics_enabled() and bool(_mu_r_nan.any().item()):
                    _nr("vfe_full_mu_t_rope_nan")
                mu_j_transported_rope = torch.where(
                    _mu_r_nan, torch.zeros_like(mu_j_transported_rope), mu_j_transported_rope,
                )

            # Regularize and invert (adaptive regularization for numerical stability)
            # Use TRANSPORT_JITTER (not eps=1e-6) — GL(K) transport can produce
            # near-singular covariances that need stronger regularization.
            sigma_j_transported = 0.5 * (sigma_j_transported + sigma_j_transported.transpose(-1, -2))
            # Σ_j_transported lives at position i (post Ω sandwich), so its
            # local frame is exp_phi_i broadcast over j.
            if gauge_covariant_ridge:
                _ep_i_bcast_d = exp_phi_i[:, :, None, :, :].expand(-1, -1, N, -1, -1)
                _R_d_pair = _ep_i_bcast_d @ _ep_i_bcast_d.transpose(-1, -2)
            else:
                _ep_i_bcast_d = None
                _R_d_pair = I_d
            sigma_j_reg = sigma_j_transported + TRANSPORT_JITTER * _R_d_pair
            sigma_j_inv = _safe_spd_inv(
                sigma_j_reg, eps=TRANSPORT_JITTER, exp_phi=_ep_i_bcast_d,
            )  # (B, C, N, d, d)

            # Delta mu for this block (query chunk) - contiguous to avoid view issues
            mu_block_i = mu_block[:, i_start:i_end].contiguous()  # (B, C, d) raw
            delta_mu_block = mu_block_i[:, :, None, :] - mu_j_transported  # raw, (B, C, N, d)

            # ∂KL_raw/∂μ_i for this block — used for the DIRECT alignment term.
            grad_kl_block = torch.einsum('bijkl,bijl->bijk', sigma_j_inv, delta_mu_block)  # (B, C, N, d)
            grad_kl_per_pair_full[:, i_start:i_end, :, block_start:block_end] = grad_kl_block

            # ∂KL_RoPE/∂(R μ_i) for this block — used (after R^T un-rotation
            # outside the loop) for the SOFTMAX-coupling chain rule.  The
            # operator is the same Σ_j_t^{-1} (Σ is unchanged by R), only the
            # delta_mu is in rope coordinates.
            if grad_kl_rope_per_pair is not None:
                mu_block_i_rope = mu_block_rope[:, i_start:i_end].contiguous()
                delta_mu_block_rope = mu_block_i_rope[:, :, None, :] - mu_j_transported_rope
                grad_kl_rope_block = torch.einsum(
                    'bijkl,bijl->bijk', sigma_j_inv, delta_mu_block_rope
                )
                grad_kl_rope_per_pair[:, i_start:i_end, :, block_start:block_end] = grad_kl_rope_block

            # KL terms for this block
            mahal_block = torch.einsum('bijk,bijk->bij', delta_mu_block, grad_kl_block)  # (B, C, N)

            # Use contiguous slice and clone for expand to avoid view issues
            sigma_i_block_slice = sigma_block[:, i_start:i_end].contiguous()  # (B, C, d, d)
            sigma_i_block = sigma_i_block_slice[:, :, None, :, :].expand(-1, -1, N, -1, -1)  # (B, C, N, d, d) — einsum accepts non-contiguous
            trace_block = torch.einsum('bijkk->bij', torch.einsum('bijkl,bijlm->bijkm', sigma_j_inv, sigma_i_block))

            try:
                L_j = torch.linalg.cholesky(sigma_j_reg)
                logdet_j = 2.0 * torch.sum(torch.log(torch.diagonal(L_j, dim1=-2, dim2=-1) + eps), dim=-1)
            except RuntimeError:
                # Fallback: use slogdet instead of zeroing (which biases KL)
                sign_j, logdet_j = torch.linalg.slogdet(sigma_j_reg)
                logdet_j = torch.where(sign_j > 0, logdet_j, torch.zeros_like(logdet_j))

            if gauge_covariant_ridge:
                _R_d_self = exp_phi_i @ exp_phi_i.transpose(-1, -2)
            else:
                _R_d_self = I_d
            sigma_i_block_diag = sigma_i_block_slice + eps * _R_d_self  # (B, C, d, d)
            try:
                L_i = torch.linalg.cholesky(sigma_i_block_diag)
                logdet_i = 2.0 * torch.sum(torch.log(torch.diagonal(L_i, dim1=-2, dim2=-1) + eps), dim=-1)
            except RuntimeError:
                # Fallback: use slogdet instead of zeroing (which biases KL)
                sign_i, logdet_i = torch.linalg.slogdet(sigma_i_block_diag)
                logdet_i = torch.where(sign_i > 0, logdet_i, torch.zeros_like(logdet_i))

            kl_block = 0.5 * (trace_block + mahal_block - d + logdet_j - logdet_i[:, :, None])
            # Clamp KL to [0, max] for numerical stability.
            # Use total K (not block dim d) to match gauge_utils fused KL ceilings.
            kl_values[:, i_start:i_end, :] = kl_values[:, i_start:i_end, :] + kl_block.clamp(min=0.0, max=max(KL_CEIL_BASE, KL_CEIL_SCALE * K))

            # Sigma alignment gradient for this block
            if compute_sigma_align_grad:
                sigma_i_inv_block = _safe_spd_inv(sigma_i_block_diag, eps=TRANSPORT_JITTER)  # (B, C, d, d)
                # Use .clone() after expand to ensure contiguous memory layout
                sigma_i_inv_exp = sigma_i_inv_block[:, :, None, :, :].expand(-1, -1, N, -1, -1)  # einsum accepts non-contiguous
                grad_sigma_block = 0.5 * (sigma_j_inv - sigma_i_inv_exp)  # (B, C, N, d, d)
                beta_chunk = beta[:, i_start:i_end, :].contiguous()  # (B, C, N)
                grad_sigma_block_weighted = lambda_belief * torch.einsum('bij,bijkl->bikl', beta_chunk, grad_sigma_block)
                grad_sigma_align[:, i_start:i_end, block_start:block_end, block_start:block_end] = (
                    grad_sigma_align[:, i_start:i_end, block_start:block_end, block_start:block_end] + grad_sigma_block_weighted
                )
                # Store per-pair sigma gradients for softmax coupling (Pass 2)
                grad_sigma_per_pair_blocks[block_idx][:, i_start:i_end, :, :, :] = grad_sigma_block

            del sigma_j_transported, sigma_j_inv, mu_j_transported
            if use_rope:
                del mu_j_transported_rope
            block_start = block_end

    # Direct alignment term: β · ∂KL_raw/∂μ (objective is raw KL).
    avg_grad = torch.einsum('bij,bijk->bik', beta, grad_kl_per_pair_full)
    grad_mu_direct = lambda_belief * avg_grad

    # Softmax coupling term: ∂β/∂μ_raw · KL_raw with chain rule through ∂KL_RoPE.
    # When use_rope=False, the rope and raw gradients are equal and the
    # un-rotation is a no-op so we fall through to the cheap path.
    # ∂β_ij/∂μ_raw_i = -(β_ij/κ)(δ_ij - β_ij) · R(θ_i)^T · ∂KL_RoPE_ij/∂(R μ_i).
    # Scale kappa by √K to match attention temperature τ = √K.
    kappa_scaled = max(kappa * math.sqrt(max(K, 1)), eps)
    if grad_kl_rope_per_pair is not None:
        # Un-rotate the rope-space gradient per i: applies R(θ_i)^T to the
        # K dimension while broadcasting over the j-key dimension.
        grad_kl_for_coupling = _un_apply_rope_pair_outer(
            grad_kl_rope_per_pair, base=rope_base
        )
    else:
        grad_kl_for_coupling = grad_kl_per_pair_full

    avg_grad_for_coupling = torch.einsum('bij,bijk->bik', beta, grad_kl_for_coupling)
    grad_deviation = avg_grad_for_coupling.unsqueeze(2) - grad_kl_for_coupling
    d_beta_d_mu = beta.unsqueeze(-1) * grad_deviation / kappa_scaled
    # Multiplier in ∂(β·KL_raw)/∂μ is KL_raw (mahal_block was computed from
    # raw delta_mu, so kl_values is the raw-mu KL — same convention as the
    # diagonal helper).
    grad_mu_softmax = lambda_softmax * torch.einsum('bij,bijk->bik', kl_values, d_beta_d_mu)

    grad_mu_align = grad_mu_direct + grad_mu_softmax
    grad_mu = grad_mu + grad_mu_align
    grad_sigma = grad_sigma + grad_sigma_align

    # Debug: alignment component norms (before softmax coupling)
    if _vfe_utils_mod._VFE_GRAD_DEBUG is not None:
        _vfe_utils_mod._VFE_GRAD_DEBUG['grad_mu_direct'] = _grad_norm(grad_mu_direct)
        _vfe_utils_mod._VFE_GRAD_DEBUG['grad_mu_softmax'] = _grad_norm(grad_mu_softmax)
        _vfe_utils_mod._VFE_GRAD_DEBUG['grad_sigma_align_direct'] = _grad_norm(grad_sigma_align)
        _ps = _per_pos_stats(grad_sigma_align)
        _vfe_utils_mod._VFE_GRAD_DEBUG['grad_sigma_align_pos_mean'] = _ps[0]
        _vfe_utils_mod._VFE_GRAD_DEBUG['grad_sigma_align_pos_max'] = _ps[1]
        # KL pairwise stats (drives softmax coupling magnitude)
        _vfe_utils_mod._VFE_GRAD_DEBUG['kl_pairwise_mean'] = kl_values.mean().item()
        _vfe_utils_mod._VFE_GRAD_DEBUG['kl_pairwise_max'] = kl_values.max().item()
        _vfe_utils_mod._VFE_GRAD_DEBUG['kappa_scaled'] = kappa_scaled
        # Fraction of pairs near the KL ceiling (diagnoses clamp saturation)
        _kl_ceil = max(KL_CEIL_BASE, KL_CEIL_SCALE * K)
        _vfe_utils_mod._VFE_GRAD_DEBUG['kl_frac_above_90pct'] = (kl_values > 0.9 * _kl_ceil).float().mean().item()
        _vfe_utils_mod._VFE_GRAD_DEBUG['kl_p95'] = kl_values.quantile(0.95).item()

    # Sigma softmax coupling (Pass 2): ∂β/∂Σ term computed per-block.
    # Uses stored per-block per-pair sigma gradients to avoid (B, N, N, K, K) memory.
    _grad_sigma_before_softmax = grad_sigma.clone() if (_vfe_utils_mod._VFE_GRAD_DEBUG is not None) else None
    if compute_sigma_align_grad and grad_sigma_per_pair_blocks is not None:
        kappa_scaled = max(kappa * math.sqrt(max(K, 1)), eps)
        block_start = 0
        for block_idx, d in enumerate(irrep_dims):
            block_end = block_start + d
            g_per_pair = grad_sigma_per_pair_blocks[block_idx]  # (B, N, N, d, d)
            avg_g = torch.einsum('bij,bijkl->bikl', beta, g_per_pair)  # (B, N, d, d)
            g_deviation = avg_g.unsqueeze(2) - g_per_pair  # (B, N, N, d, d)
            d_beta_d_sigma = beta.unsqueeze(-1).unsqueeze(-1) * g_deviation / kappa_scaled  # (B, N, N, d, d)
            grad_sigma_softmax_block = lambda_softmax * torch.einsum('bij,bijkl->bikl', kl_values, d_beta_d_sigma)  # (B, N, d, d)
            grad_sigma[:, :, block_start:block_end, block_start:block_end] = (
                grad_sigma[:, :, block_start:block_end, block_start:block_end] + grad_sigma_softmax_block
            )
            block_start = block_end
        del grad_sigma_per_pair_blocks

    # Debug: softmax coupling contribution and final totals
    if _vfe_utils_mod._VFE_GRAD_DEBUG is not None:
        if _grad_sigma_before_softmax is not None:
            _sigma_softmax_contrib = grad_sigma - _grad_sigma_before_softmax
            _vfe_utils_mod._VFE_GRAD_DEBUG['grad_sigma_softmax'] = _grad_norm(_sigma_softmax_contrib)
            _ps = _per_pos_stats(_sigma_softmax_contrib)
            _vfe_utils_mod._VFE_GRAD_DEBUG['grad_sigma_softmax_pos_mean'] = _ps[0]
            _vfe_utils_mod._VFE_GRAD_DEBUG['grad_sigma_softmax_pos_max'] = _ps[1]
        _vfe_utils_mod._VFE_GRAD_DEBUG['grad_mu_total'] = _grad_norm(grad_mu)
        _vfe_utils_mod._VFE_GRAD_DEBUG['grad_sigma_total'] = _grad_norm(grad_sigma)
        _ps = _per_pos_stats(grad_sigma)
        _vfe_utils_mod._VFE_GRAD_DEBUG['grad_sigma_total_pos_mean'] = _ps[0]
        _vfe_utils_mod._VFE_GRAD_DEBUG['grad_sigma_total_pos_max'] = _ps[1]

    return grad_mu, grad_sigma


def _compute_vfe_gradients_block_diagonal_diag(
    mu_q: torch.Tensor,        # (B, N, K) belief means
    sigma_q: torch.Tensor,     # (B, N, K) diagonal variances
    mu_p: torch.Tensor,        # (B, N, K) prior means
    sigma_p: torch.Tensor,     # (B, N, K) prior variances
    beta: torch.Tensor,        # (B, N, N) attention weights
    phi: torch.Tensor,         # (B, N, n_gen) gauge frames
    generators: torch.Tensor,  # (n_gen, K, K) generators
    alpha: 'float | torch.Tensor',
    lambda_belief: float,
    lambda_softmax: float,
    kappa: float,
    eps: float,
    irrep_dims: List[int],
    compute_sigma_align_grad: bool,
    enforce_orthogonal: bool = False,
    alpha_c0: Optional[torch.Tensor] = None,  # (K,) for product-rule correction
    cached_block_exp_pairs: Optional[list] = None,
    use_rope: bool = False,
    rope_base: float = 10000.0,
    alpha_div: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Block-diagonal VFE gradient computation for diagonal covariance mode.

    Processes each irrep block separately with small d x d Omega tensors
    and O(d) diagonal KL formulas. No matrix inverse or Cholesky needed.
    Includes sigma softmax coupling (dBeta/dSigma term).

    Memory: O(N^2 * max(d_i^2)) instead of O(N^2 * K^2).

    When ``alpha_div != 1.0``, the self-coupling and belief-alignment terms
    use the Rényi α-divergence instead of KL. For diagonal Gaussians with
    :math:`s_k = \sigma_{q,k}^2` and :math:`t_k = \sigma_{p,k}^2`, the
    blended variance is :math:`\sigma_{\text{blend},k} = (1-\alpha_d) s_k +
    \alpha_d t_k`, and the μ gradient becomes
    :math:`\alpha_d \Delta\mu_k / \sigma_{\text{blend},k}`.  At
    ``alpha_div = 1.0`` the code reduces identically to the KL path.

    Args:
        mu_q: Belief means (B, N, K).
        sigma_q: Diagonal variances (B, N, K).
        mu_p: Prior means (B, N, K).
        sigma_p: Prior diagonal variances (B, N, K).
        beta: Attention weights (B, N, N).
        phi: Gauge frames (B, N, phi_dim), phi_dim = n_gen.
        generators: Lie algebra generators (n_gen, K, K).
        alpha: Self-coupling weight, scalar or (B, N, K) Bayesian precision.
        lambda_belief: Belief alignment weight.
        kappa: Temperature for softmax coupling.
        eps: Numerical stability floor.
        irrep_dims: Block dimensions [d_1, d_2, ...] for block-diagonal KL.
        compute_sigma_align_grad: Whether to compute dF/dSigma from alignment.
        enforce_orthogonal: If True, enforce Omega in SO(K) via Newton-Schulz.
        alpha_c0: (K,) softplus(raw_c0) for product-rule correction when alpha is learnable.
        cached_block_exp_pairs: Precomputed (exp_phi, exp_neg_phi) per block.
        alpha_div: Rényi divergence order (default 1.0 = KL). When != 1.0,
            self-coupling and alignment use D_{alpha_div} instead of KL.

    Returns:
        grad_mu: (B, N, K) gradient w.r.t. mu.
        grad_sigma: (B, N, K) gradient w.r.t. diagonal sigma.
    """
    # Squeeze trailing singleton dimensions for robustness
    sigma_q = squeeze_trailing_singletons(sigma_q)
    sigma_p = squeeze_trailing_singletons(sigma_p)

    B, N, K = mu_q.shape
    device = mu_q.device
    dtype = mu_q.dtype

    # Ensure float32 for sigma divisions, logs, and KL computation.
    # Caller (variational_ffn.py) already upcasts when AMP is active;
    # skip redundant .float() copies when already float32.
    _f32 = torch.float32
    if mu_q.dtype != _f32:
        mu_q = mu_q.float()
        mu_p = mu_p.float()
        sigma_q = sigma_q.float()
        sigma_p = sigma_p.float()
        beta = beta.float()

    sigma_q_safe = sigma_q.clamp(min=eps)
    sigma_p_safe = sigma_p.clamp(min=eps)

    # =================================================================
    # 1. Self-Coupling Gradient (diagonal, no blocks needed)
    # =================================================================
    delta_mu = mu_q - mu_p
    if alpha_div == 1.0:
        # Standard KL self-coupling
        grad_mu_self = alpha * delta_mu / sigma_p_safe
        grad_sigma_self = alpha * 0.5 * (1.0 / sigma_p_safe - 1.0 / sigma_q_safe)

        # Product-rule correction for learnable alpha (KL path only)
        if alpha_c0 is not None and isinstance(alpha, torch.Tensor):
            kl_k = 0.5 * (sigma_q_safe / sigma_p_safe + delta_mu ** 2 / sigma_p_safe
                          - 1.0 + torch.log(sigma_p_safe) - torch.log(sigma_q_safe))
            kl_k = kl_k.clamp(min=0.0)
            grad_mu_self = grad_mu_self - (alpha ** 2 / alpha_c0) * kl_k * delta_mu / sigma_p_safe
            grad_sigma_self = grad_sigma_self - (alpha ** 2 / alpha_c0) * kl_k * 0.5 * (1.0 / sigma_p_safe - 1.0 / sigma_q_safe)
    else:
        # α-divergence self-coupling gradient.
        # Blended variance: σ_blend = (1-α_d)·σ_q + α_d·σ_p
        # ∂D_α/∂μ = α_d · Δμ / σ_blend
        # ∂D_α/∂s = -α_d(t-s)/(2s·σ_blend) - α_d(1-α_d)Δμ²/(2·σ_blend²)
        # where s = σ_q, t = σ_p.
        sigma_blend_self = ((1.0 - alpha_div) * sigma_q_safe + alpha_div * sigma_p_safe).clamp(min=eps)
        grad_mu_self = alpha * alpha_div * delta_mu / sigma_blend_self
        grad_sigma_self = alpha * (
            -alpha_div * (sigma_p_safe - sigma_q_safe) / (2.0 * sigma_q_safe * sigma_blend_self)
            - alpha_div * (1.0 - alpha_div) * delta_mu ** 2 / (2.0 * sigma_blend_self ** 2)
        )
        # Product-rule correction (Rényi branch): schedule α_k = c₀_k/(b₀_k + D_α,k)
        # makes ∂α/∂θ = -α²/c₀ · ∂D_α,k/∂θ, so the product rule adds -(α²/c₀)·D_α,k·∂D_α,k/∂θ.
        if alpha_c0 is not None and isinstance(alpha, torch.Tensor):
            mahal_k = alpha_div * delta_mu ** 2 / sigma_blend_self
            logdet_k = (
                (1.0 - alpha_div) * torch.log(sigma_q_safe)
                + alpha_div * torch.log(sigma_p_safe)
                - torch.log(sigma_blend_self)
            ) / (alpha_div - 1.0)
            d_alpha_k = 0.5 * (mahal_k + logdet_k).clamp(min=0.0)
            grad_mu_self = grad_mu_self - (alpha ** 2 / alpha_c0) * d_alpha_k * (
                alpha_div * delta_mu / sigma_blend_self
            )
            grad_sigma_self = grad_sigma_self - (alpha ** 2 / alpha_c0) * d_alpha_k * (
                -alpha_div * (sigma_p_safe - sigma_q_safe) / (2.0 * sigma_q_safe * sigma_blend_self)
                - alpha_div * (1.0 - alpha_div) * delta_mu ** 2 / (2.0 * sigma_blend_self ** 2)
            )

    # Debug: self-coupling component norms
    if _vfe_utils_mod._VFE_GRAD_DEBUG is not None:
        _vfe_utils_mod._VFE_GRAD_DEBUG['grad_mu_self'] = _grad_norm(grad_mu_self)
        _vfe_utils_mod._VFE_GRAD_DEBUG['grad_sigma_self'] = _grad_norm(grad_sigma_self)
        _ps = _per_pos_stats(grad_sigma_self)
        _vfe_utils_mod._VFE_GRAD_DEBUG['grad_sigma_self_pos_mean'] = _ps[0]
        _vfe_utils_mod._VFE_GRAD_DEBUG['grad_sigma_self_pos_max'] = _ps[1]
        _vfe_utils_mod._VFE_GRAD_DEBUG['sigma_p_min'] = sigma_p_safe.min().item()
        _vfe_utils_mod._VFE_GRAD_DEBUG['sigma_p_max'] = sigma_p_safe.max().item()
        _vfe_utils_mod._VFE_GRAD_DEBUG['sigma_q_eig_min'] = sigma_q_safe.min().item()
        _vfe_utils_mod._VFE_GRAD_DEBUG['sigma_q_eig_max'] = sigma_q_safe.max().item()

    # =================================================================
    # 2. Belief Alignment Gradient (block-diagonal + diagonal formulas)
    # =================================================================
    # When use_rope is active, the externally-supplied β was softmaxed from
    # KL_RoPE (computed via compute_attention_weights with use_rope=True).
    # The chain rule for ∂β/∂μ_raw must therefore go through ∂KL_RoPE/∂μ_raw
    # (= R(θ)^T · ∂KL_RoPE/∂(R μ)), NOT ∂KL_raw/∂μ.  We compute the rope-space
    # gradient in parallel and un-rotate it after the loop for the chain rule.
    #
    # IMPORTANT (RoPE σ asymmetry).  This path follows the standard-transformer
    # convention: only μ is rope-rotated; Σ is left raw.  The framework's
    # gauge-transport rule would prescribe Σ → RΣR^T as well, breaking
    # diagonal σ structure.  See `BlockConfig.rope_full_gauge` for the
    # experimental opt-in that implements the full rotation.
    if use_rope:
        # _apply_rope, _un_apply_rope_pair_outer imported at module level
        mu_q_rope = _apply_rope(mu_q, base=rope_base)
    else:
        mu_q_rope = mu_q

    # Precompute matrix exponentials — FUSED by dimension group
    if cached_block_exp_pairs is not None:
        _fused_pairs = cached_block_exp_pairs
    else:
        _fused_pairs = fused_block_matrix_exp_pairs(
            phi, generators, irrep_dims, enforce_orthogonal=enforce_orthogonal
        )
    block_exp_phi = [p[0] for p in _fused_pairs]
    block_exp_neg_phi = [p[1] for p in _fused_pairs]

    # Accumulators for per-pair KL values and gradients across all blocks
    kl_values = torch.zeros(B, N, N, device=device, dtype=dtype)
    kl_values_raw = torch.zeros(B, N, N, device=device, dtype=dtype) if use_rope else None
    grad_kl_per_pair_full = torch.zeros(B, N, N, K, device=device, dtype=dtype)
    # ∂KL_RoPE/∂(R μ_i): un-rotated per i (after the loop) for chain rule.
    grad_kl_rope_per_pair = torch.zeros(B, N, N, K, device=device, dtype=dtype) if use_rope else None
    grad_sigma_align = torch.zeros_like(sigma_q)
    # Accumulator for sigma softmax coupling (same memory cost as grad_kl_per_pair_full)
    grad_sigma_per_pair_full = torch.zeros(B, N, N, K, device=device, dtype=dtype) if (
        compute_sigma_align_grad) else None

    # Process each irrep block sequentially with direct Omega construction.
    # This avoids the factored transport approach (A @ (B diag(σ) B^T) @ A^T)
    # which accumulates more float32 rounding error through intermediate matmuls.
    block_start = 0
    for block_idx, d in enumerate(irrep_dims):
        block_end = block_start + d

        # Slices on the last dim are non-contiguous; the downstream einsums
        # accept non-contiguous inputs natively, so we skip the .contiguous()
        # copy (each was a B*N*d allocation per block per E-step iteration).
        mu_block = mu_q[:, :, block_start:block_end]                     # (B, N, d) raw
        mu_block_rope = mu_q_rope[:, :, block_start:block_end] if use_rope else mu_block
        sigma_block = sigma_q_safe[:, :, block_start:block_end]          # (B, N, d)

        # Block Omega: (B, N, N, d, d) — direct construction for numerical precision
        Omega_block = torch.einsum(
            'bikl,bjlm->bijkm',
            block_exp_phi[block_idx], block_exp_neg_phi[block_idx]
        )

        # Transport means: for the alignment objective (raw KL), use raw mu;
        # for β-side KL (rope KL), use rope-rotated mu.
        mu_j_transported = torch.einsum('bijkl,bjl->bijk', Omega_block, mu_block)  # raw, (B, N, N, d)
        if use_rope:
            mu_j_transported_rope = torch.einsum('bijkl,bjl->bijk', Omega_block, mu_block_rope)
        else:
            mu_j_transported_rope = mu_j_transported

        # Diagonal covariance transport: σ_t[k] = Σ_l Ω_kl² * σ[l]
        # This correctly extracts diag(Ω @ diag(σ) @ Ω^T).  The transport
        # itself is exact.  The APPROXIMATION is below: the KL computation
        # treats σ_t as if the full transported covariance were diagonal,
        # using 1/σ_t[k] as the precision and Σ log(σ_t[k]) as the logdet.
        # For non-identity Ω these differ from (Ω Σ Ω^T)^{-1}_{kk} and
        # log|Ω Σ Ω^T|, breaking strict gauge equivariance.  Use
        # exact_diagonal_transport=True to lift to full covariance and
        # compute exact KL with proper matrix inverse and logdet.
        sigma_j_transported = torch.einsum(
            'bijkl,bijkl,bjl->bijk', Omega_block, Omega_block, sigma_block
        ).clamp(min=eps)  # (B, N, N, d)

        del Omega_block

        # Delta mu (broadcast instead of expand+clone to avoid 59M-element copy)
        mu_block_i = mu_block[:, :, None, :]  # (B, N, 1, d) - broadcasts with (B, N, N, d)
        delta_mu = mu_block_i - mu_j_transported  # raw, (B, N, N, d)

        sigma_i_block = sigma_block[:, :, None, :]  # (B, N, 1, d)

        if alpha_div == 1.0:
            # Standard KL alignment gradient: ∂KL/∂μ_i = Δμ / σ_j_t
            grad_kl_block = delta_mu / sigma_j_transported  # (B, N, N, d)

            # ∂KL_RoPE/∂(R μ_i) = (R μ_i - Ω R μ_j) / σ_j_t — for chain rule (un-rotated below)
            if grad_kl_rope_per_pair is not None:
                mu_block_i_rope = mu_block_rope[:, :, None, :]
                delta_mu_rope = mu_block_i_rope - mu_j_transported_rope
                grad_kl_rope_per_pair[:, :, :, block_start:block_end] = delta_mu_rope / sigma_j_transported

            # Diagonal KL for this block — used for SOFTMAX-coupling MULTIPLIER (KL_raw)
            trace_block = (sigma_i_block / sigma_j_transported).sum(dim=-1)
            mahal_block = (delta_mu ** 2 / sigma_j_transported).sum(dim=-1)
            logdet_block = (torch.log(sigma_j_transported.clamp(min=eps))
                            - torch.log(sigma_i_block.clamp(min=eps))).sum(dim=-1)
            kl_block = 0.5 * (trace_block + mahal_block - d + logdet_block)
            kl_block = kl_block.clamp(min=0.0, max=max(KL_CEIL_BASE, KL_CEIL_SCALE * K))
        else:
            # α-divergence alignment gradient.
            # σ_blend = (1-α_d)·σ_i + α_d·σ_j_t, ∂D_α/∂μ_i = α_d·Δμ / σ_blend
            sigma_blend_align = (
                (1.0 - alpha_div) * sigma_i_block + alpha_div * sigma_j_transported
            ).clamp(min=eps)  # (B, N, N, d)
            grad_kl_block = alpha_div * delta_mu / sigma_blend_align  # (B, N, N, d)

            # RoPE chain-rule gradient in blend space
            if grad_kl_rope_per_pair is not None:
                mu_block_i_rope = mu_block_rope[:, :, None, :]
                delta_mu_rope = mu_block_i_rope - mu_j_transported_rope
                grad_kl_rope_per_pair[:, :, :, block_start:block_end] = alpha_div * delta_mu_rope / sigma_blend_align

            # D_α value: Mahalanobis + log-det blend term
            mahal_block = (alpha_div * delta_mu ** 2 / sigma_blend_align).sum(dim=-1)
            logdet_block = (
                (1.0 - alpha_div) * torch.log(sigma_i_block.clamp(min=eps))
                + alpha_div * torch.log(sigma_j_transported.clamp(min=eps))
                - torch.log(sigma_blend_align)
            ).sum(dim=-1) / (alpha_div - 1.0)
            kl_block = 0.5 * (mahal_block + logdet_block)
            kl_block = kl_block.clamp(min=0.0, max=max(KL_CEIL_BASE, KL_CEIL_SCALE * K))

        grad_kl_per_pair_full[:, :, :, block_start:block_end] = grad_kl_block
        kl_values = kl_values + kl_block

        # When RoPE is active, kl_values_raw collects the raw-mu divergence value;
        # when RoPE is inactive, kl_values already IS the raw value.
        if kl_values_raw is not None:
            kl_values_raw = kl_values_raw + kl_block  # delta_mu already uses raw mu

        # Sigma alignment gradient for this block
        if compute_sigma_align_grad:
            if alpha_div == 1.0:
                sigma_j_inv_diag = 1.0 / sigma_j_transported  # (B, N, N, d)
                sigma_i_inv = 1.0 / sigma_block  # (B, N, d)
                grad_sigma_pair = 0.5 * (sigma_j_inv_diag - sigma_i_inv[:, :, None, :])  # broadcast
            else:
                # α-divergence σ gradient:
                # ∂D_α/∂s = -α_d(t-s)/(2s·σ_blend) - α_d(1-α_d)Δμ²/(2·σ_blend²)
                # where s = σ_i, t = σ_j_t, computed per-element.
                sigma_blend_s = (
                    (1.0 - alpha_div) * sigma_i_block + alpha_div * sigma_j_transported
                ).clamp(min=eps)
                grad_sigma_pair = (
                    -alpha_div * (sigma_j_transported - sigma_i_block)
                    / (2.0 * sigma_i_block.clamp(min=eps) * sigma_blend_s)
                    - alpha_div * (1.0 - alpha_div) * delta_mu ** 2 / (2.0 * sigma_blend_s ** 2)
                )  # (B, N, N, d)

            grad_sigma_align[:, :, block_start:block_end] = (
                grad_sigma_align[:, :, block_start:block_end]
                + lambda_belief * torch.einsum('bij,bijk->bik', beta, grad_sigma_pair)
            )
            if grad_sigma_per_pair_full is not None:
                grad_sigma_per_pair_full[:, :, :, block_start:block_end] = grad_sigma_pair

        del sigma_j_transported, mu_j_transported, delta_mu
        block_start = block_end

    # Direct alignment term: β · ∂KL_raw/∂μ (objective is raw KL).
    avg_grad = torch.einsum('bij,bijk->bik', beta, grad_kl_per_pair_full)
    grad_mu_direct = lambda_belief * avg_grad

    # Softmax coupling: ∂β/∂μ_raw · KL_raw with chain rule through ∂KL_RoPE.
    # When use_rope=False, the rope and raw gradients are equal and the
    # un-rotation is a no-op so we fall through to the cheap path.
    kappa_scaled = max(kappa * math.sqrt(max(K, 1)), eps)
    if grad_kl_rope_per_pair is not None:
        grad_kl_for_coupling = _un_apply_rope_pair_outer(
            grad_kl_rope_per_pair, base=rope_base
        )
    else:
        grad_kl_for_coupling = grad_kl_per_pair_full

    avg_grad_for_coupling = torch.einsum('bij,bijk->bik', beta, grad_kl_for_coupling)
    grad_deviation = avg_grad_for_coupling.unsqueeze(2) - grad_kl_for_coupling
    d_beta_d_mu = beta.unsqueeze(-1) * grad_deviation / kappa_scaled
    # Multiplier in ∂(β·KL_raw)/∂μ is KL_raw, not the externally-supplied β-side KL.
    kl_for_coupling = kl_values_raw if kl_values_raw is not None else kl_values
    grad_mu_softmax = lambda_softmax * torch.einsum('bij,bijk->bik', kl_for_coupling, d_beta_d_mu)

    grad_mu_align = grad_mu_direct + grad_mu_softmax
    grad_mu = grad_mu_self + grad_mu_align

    # Debug: alignment component norms (before softmax coupling for sigma)
    if _vfe_utils_mod._VFE_GRAD_DEBUG is not None:
        _vfe_utils_mod._VFE_GRAD_DEBUG['grad_mu_direct'] = _grad_norm(grad_mu_direct)
        _vfe_utils_mod._VFE_GRAD_DEBUG['grad_mu_softmax'] = _grad_norm(grad_mu_softmax)
        _vfe_utils_mod._VFE_GRAD_DEBUG['grad_sigma_align_direct'] = _grad_norm(grad_sigma_align)
        _ps = _per_pos_stats(grad_sigma_align)
        _vfe_utils_mod._VFE_GRAD_DEBUG['grad_sigma_align_pos_mean'] = _ps[0]
        _vfe_utils_mod._VFE_GRAD_DEBUG['grad_sigma_align_pos_max'] = _ps[1]
        _vfe_utils_mod._VFE_GRAD_DEBUG['kl_pairwise_mean'] = kl_values.mean().item()
        _vfe_utils_mod._VFE_GRAD_DEBUG['kl_pairwise_max'] = kl_values.max().item()
        _vfe_utils_mod._VFE_GRAD_DEBUG['kappa_scaled'] = kappa_scaled
        # Fraction of pairs near the KL ceiling (diagnoses clamp saturation)
        _kl_ceil = max(KL_CEIL_BASE, KL_CEIL_SCALE * K)
        _vfe_utils_mod._VFE_GRAD_DEBUG['kl_frac_above_90pct'] = (kl_values > 0.9 * _kl_ceil).float().mean().item()
        _vfe_utils_mod._VFE_GRAD_DEBUG['kl_p95'] = kl_values.quantile(0.95).item()

    # Sigma softmax coupling: Σ_j KL_raw_ij · ∂β_ij/∂σ_i
    # σ derivatives don't depend on RoPE rotation (only μ does), so the
    # σ chain rule grad_sigma_per_pair_full is unchanged.  However the
    # multiplier should still be the alignment-objective KL (raw) for
    # consistency with the μ coupling above.
    grad_sigma_softmax_norm = 0.0
    if grad_sigma_per_pair_full is not None:
        avg_sigma_grad = torch.einsum('bij,bijk->bik', beta, grad_sigma_per_pair_full)
        sigma_grad_deviation = avg_sigma_grad.unsqueeze(2) - grad_sigma_per_pair_full
        d_beta_d_sigma = beta.unsqueeze(-1) * sigma_grad_deviation / kappa_scaled
        grad_sigma_softmax = lambda_softmax * torch.einsum('bij,bijk->bik', kl_for_coupling, d_beta_d_sigma)
        if _vfe_utils_mod._VFE_GRAD_DEBUG is not None:
            grad_sigma_softmax_norm = _grad_norm(grad_sigma_softmax)
        grad_sigma_align = grad_sigma_align + grad_sigma_softmax

    grad_sigma = grad_sigma_self + grad_sigma_align

    # Debug: final totals
    if _vfe_utils_mod._VFE_GRAD_DEBUG is not None:
        _vfe_utils_mod._VFE_GRAD_DEBUG['grad_sigma_softmax'] = grad_sigma_softmax_norm
        _vfe_utils_mod._VFE_GRAD_DEBUG['grad_mu_total'] = _grad_norm(grad_mu)
        _vfe_utils_mod._VFE_GRAD_DEBUG['grad_sigma_total'] = _grad_norm(grad_sigma)
        _ps = _per_pos_stats(grad_sigma)
        _vfe_utils_mod._VFE_GRAD_DEBUG['grad_sigma_total_pos_mean'] = _ps[0]
        _vfe_utils_mod._VFE_GRAD_DEBUG['grad_sigma_total_pos_max'] = _ps[1]

    return grad_mu.to(dtype), grad_sigma.to(dtype)


def _fused_attention_and_vfe_gradients_block_diag(
    mu_q: torch.Tensor,        # (B, N, K) belief means
    sigma_q: torch.Tensor,     # (B, N, K) diagonal variances
    mu_p: torch.Tensor,        # (B, N, K) prior means
    sigma_p: torch.Tensor,     # (B, N, K) prior variances
    phi: torch.Tensor,         # (B, N, n_gen) gauge frames
    generators: torch.Tensor,  # (n_gen, K, K) generators
    alpha: 'float | torch.Tensor',
    lambda_belief: float,
    lambda_softmax: float,
    kappa: float,
    eps: float,
    irrep_dims: List[int],
    compute_sigma_align_grad: bool,
    enforce_orthogonal: bool = False,
    alpha_c0: Optional[torch.Tensor] = None,
    cached_block_exp_pairs: Optional[list] = None,
    mask: Optional[torch.Tensor] = None,
    mask_self_attention: bool = False,
    use_rope: bool = False,
    rope_base: float = 10000.0,
    return_kl: bool = False,
    alpha_div: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Fused attention + VFE gradient computation for block-diagonal diagonal mode.

    Computes beta (attention weights) AND VFE gradients in a single pass over
    irrep blocks, sharing the Omega construction. Eliminates the redundant Omega
    computation that occurs when compute_attention_weights and
    compute_vfe_gradients_gpu are called separately. Includes sigma softmax
    coupling (dBeta/dSigma). Incompatible with exact_diagonal_transport (which
    lifts to full covariance and disables fused diagonal paths).

    For a config with 5 heads x d=15, this saves 5 x O(B*N^2*d^2) Omega
    constructions per VFE iteration.

    Args:
        mu_q: Belief means (B, N, K).
        sigma_q: Diagonal variances (B, N, K).
        mu_p: Prior means (B, N, K).
        sigma_p: Prior diagonal variances (B, N, K).
        phi: Gauge frames (B, N, phi_dim), phi_dim = n_gen.
        generators: Lie algebra generators (n_gen, K, K).
        alpha: Self-coupling weight, scalar or (B, N, K) Bayesian precision.
        lambda_belief: Belief alignment weight.
        kappa: Temperature for softmax coupling.
        eps: Numerical stability floor.
        irrep_dims: Block dimensions [d_1, d_2, ...] for block-diagonal KL.
        compute_sigma_align_grad: Whether to compute dF/dSigma from alignment.
        enforce_orthogonal: If True, enforce Omega in SO(K) via Newton-Schulz.
        alpha_c0: (K,) softplus(raw_c0) for product-rule correction.
        cached_block_exp_pairs: Precomputed (exp_phi, exp_neg_phi) per block.
        mask: Causal mask (B, N, N), 0 = cannot attend.
        mask_self_attention: If True, mask diagonal (no self-attention).
        use_rope: Apply rotary position embeddings to mu for KL computation.
        rope_base: RoPE base frequency.
        return_kl: If True, also return pairwise KL matrix.
        alpha_div: Rényi divergence order (default 1.0 = KL). When != 1.0,
            self-coupling and alignment use D_{alpha_div} instead of KL.

    Returns:
        beta: (B, N, N) attention weights.
        grad_mu: (B, N, K) gradient w.r.t. mu.
        grad_sigma: (B, N, K) gradient w.r.t. diagonal sigma.
        kl_matrix: (B, N, N) pairwise divergences, or None if return_kl=False.
    """
    sigma_q = squeeze_trailing_singletons(sigma_q)
    sigma_p = squeeze_trailing_singletons(sigma_p)

    B, N, K = mu_q.shape
    device = mu_q.device
    dtype = mu_q.dtype

    # Ensure float32 (caller already upcasts under AMP; skip copy when already f32)
    _f32 = torch.float32
    if mu_q.dtype != _f32:
        mu_q = mu_q.float()
        mu_p = mu_p.float()
        sigma_q = sigma_q.float()
        sigma_p = sigma_p.float()

    sigma_q_safe = sigma_q.clamp(min=eps)
    sigma_p_safe = sigma_p.clamp(min=eps)

    # Apply RoPE to a copy of mu for KL computation (not for gradients).
    # When use_rope=True, β is softmaxed from KL_RoPE while the alignment
    # objective uses raw-space KL.  The chain rule for ∂β/∂μ must therefore
    # go through ∂KL_RoPE/∂μ_raw, NOT ∂KL_raw/∂μ_raw — see the rope-space
    # gradient accumulator and un-rotation below.
    #
    # IMPORTANT (RoPE σ asymmetry).  This path rotates only μ via _apply_rope.
    # Σ is left raw — both Σ_i (the source covariance) and Σ_t (the
    # transported covariance, computed only with the LEARNED Ω, not with R).
    # Under the GL(K) framework's gauge-transport rule (μ → Rμ, Σ → RΣR^T),
    # this is incomplete: a fully framework-consistent rope KL would also
    # rotate the covariances by R.  The current implementation matches the
    # standard-transformer Q/K rotation pattern and preserves diagonal σ.
    # See `BlockConfig.rope_full_gauge` for the experimental flag enabling
    # the full Σ rotation.
    if use_rope:
        # _apply_rope, _un_apply_rope_pair_outer imported at module level
        mu_q_rope = _apply_rope(mu_q, base=rope_base)
    else:
        mu_q_rope = mu_q

    # Self-coupling gradient
    delta_mu_sp = mu_q - mu_p
    if alpha_div == 1.0:
        # Standard KL self-coupling
        grad_mu_self = alpha * delta_mu_sp / sigma_p_safe
        grad_sigma_self = alpha * 0.5 * (1.0 / sigma_p_safe - 1.0 / sigma_q_safe)

        # Product-rule correction for learnable alpha (KL path only)
        if alpha_c0 is not None and isinstance(alpha, torch.Tensor):
            kl_k = 0.5 * (sigma_q_safe / sigma_p_safe + delta_mu_sp ** 2 / sigma_p_safe
                          - 1.0 + torch.log(sigma_p_safe) - torch.log(sigma_q_safe))
            kl_k = kl_k.clamp(min=0.0)
            grad_mu_self = grad_mu_self - (alpha ** 2 / alpha_c0) * kl_k * delta_mu_sp / sigma_p_safe
            grad_sigma_self = grad_sigma_self - (alpha ** 2 / alpha_c0) * kl_k * 0.5 * (1.0 / sigma_p_safe - 1.0 / sigma_q_safe)
    else:
        # α-divergence self-coupling gradient.
        # σ_blend = (1-α_d)·σ_q + α_d·σ_p, ∂D_α/∂μ = α_d·Δμ / σ_blend
        sigma_blend_self = ((1.0 - alpha_div) * sigma_q_safe + alpha_div * sigma_p_safe).clamp(min=eps)
        grad_mu_self = alpha * alpha_div * delta_mu_sp / sigma_blend_self
        grad_sigma_self = alpha * (
            -alpha_div * (sigma_p_safe - sigma_q_safe) / (2.0 * sigma_q_safe * sigma_blend_self)
            - alpha_div * (1.0 - alpha_div) * delta_mu_sp ** 2 / (2.0 * sigma_blend_self ** 2)
        )
        # Product-rule correction (Rényi branch): α_k = c₀_k/(b₀_k + D_α,k) ⇒
        # ∂α_k/∂θ = -α_k²/c₀ · ∂D_α,k/∂θ.
        if alpha_c0 is not None and isinstance(alpha, torch.Tensor):
            mahal_k = alpha_div * delta_mu_sp ** 2 / sigma_blend_self
            logdet_k = (
                (1.0 - alpha_div) * torch.log(sigma_q_safe)
                + alpha_div * torch.log(sigma_p_safe)
                - torch.log(sigma_blend_self)
            ) / (alpha_div - 1.0)
            d_alpha_k = 0.5 * (mahal_k + logdet_k).clamp(min=0.0)
            grad_mu_self = grad_mu_self - (alpha ** 2 / alpha_c0) * d_alpha_k * (
                alpha_div * delta_mu_sp / sigma_blend_self
            )
            grad_sigma_self = grad_sigma_self - (alpha ** 2 / alpha_c0) * d_alpha_k * (
                -alpha_div * (sigma_p_safe - sigma_q_safe) / (2.0 * sigma_q_safe * sigma_blend_self)
                - alpha_div * (1.0 - alpha_div) * delta_mu_sp ** 2 / (2.0 * sigma_blend_self ** 2)
            )

    # Debug: self-coupling component norms (fused path)
    if _vfe_utils_mod._VFE_GRAD_DEBUG is not None:
        _vfe_utils_mod._VFE_GRAD_DEBUG['grad_mu_self'] = _grad_norm(grad_mu_self)
        _vfe_utils_mod._VFE_GRAD_DEBUG['grad_sigma_self'] = _grad_norm(grad_sigma_self)
        _ps = _per_pos_stats(grad_sigma_self)
        _vfe_utils_mod._VFE_GRAD_DEBUG['grad_sigma_self_pos_mean'] = _ps[0]
        _vfe_utils_mod._VFE_GRAD_DEBUG['grad_sigma_self_pos_max'] = _ps[1]
        _vfe_utils_mod._VFE_GRAD_DEBUG['sigma_p_min'] = sigma_p_safe.min().item()
        _vfe_utils_mod._VFE_GRAD_DEBUG['sigma_p_max'] = sigma_p_safe.max().item()
        _vfe_utils_mod._VFE_GRAD_DEBUG['sigma_q_eig_min'] = sigma_q_safe.min().item()
        _vfe_utils_mod._VFE_GRAD_DEBUG['sigma_q_eig_max'] = sigma_q_safe.max().item()

    # Precompute matrix exponentials
    if cached_block_exp_pairs is not None:
        _fused_pairs = cached_block_exp_pairs
    else:
        _fused_pairs = fused_block_matrix_exp_pairs(
            phi, generators, irrep_dims, enforce_orthogonal=enforce_orthogonal
        )
    block_exp_phi = [p[0] for p in _fused_pairs]
    block_exp_neg_phi = [p[1] for p in _fused_pairs]

    # Accumulators
    kl_values = torch.zeros(B, N, N, device=device, dtype=torch.float32)
    # When RoPE is active, kl_values uses RoPE-rotated mu (for attention β) but
    # the alignment objective uses raw mu. The softmax-coupling chain rule
    # requires KL VALUES from the alignment objective (raw KL multiplier) AND
    # KL GRADIENTS from the function β was softmaxed from (RoPE KL).  We
    # accumulate raw-mu KLs separately for the multiplier, and rope-space
    # gradients separately so we can un-rotate them per-i for the chain rule.
    kl_values_raw = torch.zeros(B, N, N, device=device, dtype=torch.float32) if use_rope else None
    grad_kl_per_pair_full = torch.zeros(B, N, N, K, device=device, dtype=torch.float32)
    # ∂KL_RoPE/∂(R μ_i): used (after R^T un-rotation per i) in the softmax
    # coupling chain rule.  Only allocated when RoPE is active.
    grad_kl_rope_per_pair = torch.zeros(B, N, N, K, device=device, dtype=torch.float32) if use_rope else None
    grad_sigma_align = torch.zeros_like(sigma_q)
    grad_sigma_per_pair_full = torch.zeros(B, N, N, K, device=device, dtype=torch.float32) if (
        compute_sigma_align_grad) else None
    # Single pass over blocks: compute Omega, KL, and gradients together
    block_start = 0
    for block_idx, d in enumerate(irrep_dims):
        block_end = block_start + d

        # Use RoPE-rotated mu for KL/attention, raw mu for gradients.
        # Skip .contiguous(): downstream einsums handle non-contiguous slices.
        mu_block_rope = mu_q_rope[:, :, block_start:block_end]
        mu_block = mu_q[:, :, block_start:block_end]
        sigma_block = sigma_q_safe[:, :, block_start:block_end]

        # Build Omega ONCE for this block
        Omega_block = torch.einsum(
            'bikl,bjlm->bijkm',
            block_exp_phi[block_idx], block_exp_neg_phi[block_idx]
        )

        # NaN guard: if stable_matrix_exp_pair produced NaN for an extreme
        # phi that slipped through the stability layer, the NaN propagates
        # into sigma_j_transported (clamp() does not replace NaN) and then
        # into the KL / softmax / gradient chain, corrupting the whole
        # batch. Replace Omega rows containing NaN with identity so the
        # offending pair contributes zero KL and zero gradient rather than
        # poisoning everything downstream. Mirrors the NaN guard in
        # kl_computation._kl_kernel_dense and gauge_utils fused KL kernels.
        # Unconditional torch.where avoids the host sync that an `.any()`
        # branch would force on every block of every E-step iteration.
        _eye_d = _cached_eye(d, Omega_block.device, Omega_block.dtype)
        _nan_mask = torch.isnan(Omega_block).any(dim=-1).any(dim=-1)  # (B, N, N)
        if _kl_diagnostics_enabled() and _nan_mask.any():
            _nr("fused_vfe_omega_nan", count=int(_nan_mask.sum().item()))
        Omega_block = torch.where(
            _nan_mask.unsqueeze(-1).unsqueeze(-1),
            _eye_d.expand_as(Omega_block),
            Omega_block,
        )

        # Transport (use RoPE mu for KL computation)
        mu_j_transported = torch.einsum('bijkl,bjl->bijk', Omega_block, mu_block_rope)
        # Floor on transported diagonal variance.  Use max(eps, 1e-7) to
        # guarantee the division 1/sigma_j_transported (line ~1025) stays
        # bounded.  Previous hardcoded 1e-4 was inconsistent with the eps
        # parameter (typically 1e-6) that the self-coupling path uses and
        # biased small transported variances upward by up to 2 orders of
        # magnitude in normal training.
        _sigma_floor = max(float(eps), 1e-7)
        sigma_j_transported = torch.einsum(
            'bijkl,bijkl,bjl->bijk', Omega_block, Omega_block, sigma_block
        ).clamp(min=_sigma_floor)

        # Second-line guard: clamp() cannot remove NaN, only out-of-range
        # finite values. If Omega was finite but the einsum produced NaN
        # from an extreme sigma_block, mask it out so kl / beta / gradients
        # stay finite. Unconditional torch.where over isnan only swaps NaN
        # values — it leaves +/-inf untouched, so this is NOT equivalent
        # to a bare nan_to_num() (which would also replace inf with ~3.4e38
        # and then log() of that produces spurious KL ≈ 88).
        _sigma_nan = torch.isnan(sigma_j_transported)
        if _kl_diagnostics_enabled() and _sigma_nan.any():
            _nr("fused_vfe_sigma_t_nan", count=int(_sigma_nan.sum().item()))
        sigma_j_transported = torch.where(
            _sigma_nan,
            torch.full_like(sigma_j_transported, 1.0),
            sigma_j_transported,
        )
        _mu_nan = torch.isnan(mu_j_transported)
        if _kl_diagnostics_enabled() and _mu_nan.any():
            _nr("fused_vfe_mu_t_nan", count=int(_mu_nan.sum().item()))
        mu_j_transported = torch.where(
            _mu_nan,
            torch.zeros_like(mu_j_transported),
            mu_j_transported,
        )

        # Divergence computation (for attention weights β)
        mu_block_i_rope = mu_block_rope[:, :, None, :]  # broadcast, no clone needed
        delta_mu_kl = mu_block_i_rope - mu_j_transported

        sigma_block_safe = sigma_block[:, :, None, :].clamp(min=_sigma_floor)

        if alpha_div == 1.0:
            trace_block = (sigma_block_safe / sigma_j_transported).sum(dim=-1)
            mahal_block = (delta_mu_kl ** 2 / sigma_j_transported).sum(dim=-1)
            logdet_block = (torch.log(sigma_j_transported) - torch.log(sigma_block_safe)).sum(dim=-1)
            kl_block = 0.5 * (trace_block + mahal_block - d + logdet_block)
            kl_block = kl_block.clamp(min=0.0, max=max(KL_CEIL_BASE, KL_CEIL_SCALE * K))
        else:
            # α-divergence for attention weights (rope-space mu)
            sigma_blend_kl = (
                (1.0 - alpha_div) * sigma_block_safe + alpha_div * sigma_j_transported
            ).clamp(min=eps)
            mahal_kl = (alpha_div * delta_mu_kl ** 2 / sigma_blend_kl).sum(dim=-1)
            logdet_kl = (
                (1.0 - alpha_div) * torch.log(sigma_block_safe.clamp(min=eps))
                + alpha_div * torch.log(sigma_j_transported.clamp(min=eps))
                - torch.log(sigma_blend_kl)
            ).sum(dim=-1) / (alpha_div - 1.0)
            kl_block = 0.5 * (mahal_kl + logdet_kl)
            kl_block = kl_block.clamp(min=0.0, max=max(KL_CEIL_BASE, KL_CEIL_SCALE * K))

        kl_values = kl_values + kl_block

        # Gradient computation for the DIRECT alignment term (use raw mu).
        # The alignment objective uses raw-space divergence: F_align = Σ_ij β_ij D_ij.
        if use_rope:
            mu_j_transported_raw = torch.einsum('bijkl,bjl->bijk', Omega_block, mu_block)
            delta_mu_grad = mu_block[:, :, None, :] - mu_j_transported_raw
        else:
            delta_mu_grad = mu_block[:, :, None, :] - mu_j_transported

        if alpha_div == 1.0:
            grad_kl_block = delta_mu_grad / sigma_j_transported
        else:
            # α-divergence gradient uses blend of σ_i and σ_j_t (raw-space)
            sigma_i_block_raw = sigma_block[:, :, None, :].clamp(min=_sigma_floor)
            sigma_blend_grad = (
                (1.0 - alpha_div) * sigma_i_block_raw + alpha_div * sigma_j_transported
            ).clamp(min=eps)
            grad_kl_block = alpha_div * delta_mu_grad / sigma_blend_grad

        grad_kl_per_pair_full[:, :, :, block_start:block_end] = grad_kl_block

        # Accumulate raw-mu divergence for softmax-coupling MULTIPLIER consistency.
        if kl_values_raw is not None:
            if alpha_div == 1.0:
                mahal_raw = (delta_mu_grad ** 2 / sigma_j_transported).sum(dim=-1)
                kl_block_raw = 0.5 * (trace_block + mahal_raw - d + logdet_block)
            else:
                sigma_i_block_raw2 = sigma_block[:, :, None, :].clamp(min=_sigma_floor)
                sigma_blend_raw = (
                    (1.0 - alpha_div) * sigma_i_block_raw2 + alpha_div * sigma_j_transported
                ).clamp(min=eps)
                mahal_raw = (alpha_div * delta_mu_grad ** 2 / sigma_blend_raw).sum(dim=-1)
                logdet_raw = (
                    (1.0 - alpha_div) * torch.log(sigma_i_block_raw2.clamp(min=eps))
                    + alpha_div * torch.log(sigma_j_transported.clamp(min=eps))
                    - torch.log(sigma_blend_raw)
                ).sum(dim=-1) / (alpha_div - 1.0)
                kl_block_raw = 0.5 * (mahal_raw + logdet_raw)
            kl_values_raw = kl_values_raw + kl_block_raw.clamp(min=0.0, max=max(KL_CEIL_BASE, KL_CEIL_SCALE * K))

        # Rope-space gradient for the softmax-coupling chain rule.
        # delta_mu_kl is already in rope space (mu_block_i_rope - mu_j_transported).
        if grad_kl_rope_per_pair is not None:
            if alpha_div == 1.0:
                grad_kl_rope_block = delta_mu_kl / sigma_j_transported
            else:
                grad_kl_rope_block = alpha_div * delta_mu_kl / sigma_blend_kl
            grad_kl_rope_per_pair[:, :, :, block_start:block_end] = grad_kl_rope_block

        if compute_sigma_align_grad:
            if alpha_div == 1.0:
                sigma_j_inv_diag = 1.0 / sigma_j_transported
                sigma_i_inv = 1.0 / sigma_block
                grad_sigma_pair = 0.5 * (sigma_j_inv_diag - sigma_i_inv[:, :, None, :])
            else:
                # α-divergence σ gradient using raw-space blend
                sigma_i_b = sigma_block[:, :, None, :].clamp(min=_sigma_floor)
                sigma_blend_s = (
                    (1.0 - alpha_div) * sigma_i_b + alpha_div * sigma_j_transported
                ).clamp(min=eps)
                grad_sigma_pair = (
                    -alpha_div * (sigma_j_transported - sigma_i_b)
                    / (2.0 * sigma_i_b * sigma_blend_s)
                    - alpha_div * (1.0 - alpha_div) * delta_mu_grad ** 2 / (2.0 * sigma_blend_s ** 2)
                )
            if grad_sigma_per_pair_full is not None:
                grad_sigma_per_pair_full[:, :, :, block_start:block_end] = grad_sigma_pair

        del Omega_block, sigma_j_transported, mu_j_transported
        block_start = block_end

    # Compute attention weights from KL values
    dim_scale = math.sqrt(max(K, 1))
    logits = -kl_values / (kappa * dim_scale)

    if mask is not None:
        logits = logits.masked_fill(mask == 0, float('-inf'))

    if mask_self_attention:
        diag_idx = torch.arange(N, device=device)
        has_other_targets = (logits != float('-inf')).sum(dim=-1) > 1
        logits = logits.clone()
        diag_vals = logits[:, diag_idx, diag_idx]
        masked_diag_vals = torch.where(
            has_other_targets,
            torch.full_like(diag_vals, float('-inf')),
            diag_vals
        )
        logits[:, diag_idx, diag_idx] = masked_diag_vals

    beta = torch.nn.functional.softmax(logits, dim=-1)
    masked_positions = (logits == float('-inf'))
    beta = torch.where(masked_positions, beta, beta.clamp(min=eps))
    beta_sum = beta.sum(dim=-1, keepdim=True).clamp(min=eps)
    beta = beta / beta_sum

    # Direct alignment term: β_RoPE · ∂KL_raw/∂μ_raw  (objective is raw KL).
    avg_grad = torch.einsum('bij,bijk->bik', beta, grad_kl_per_pair_full)
    grad_mu_direct = lambda_belief * avg_grad

    kappa_scaled = max(kappa * math.sqrt(max(K, 1)), eps)

    # Softmax coupling term: ∂β/∂μ_raw · KL_raw.
    # ∂β_ij/∂μ_raw_i = -(β_ij/κ)(δ_ij - β_ij) · ∂KL_RoPE_ij/∂μ_raw_i
    # where ∂KL_RoPE/∂μ_raw_i = R(θ_i)^T · ∂KL_RoPE/∂(R(θ_i)μ_i).
    # Without RoPE the rope gradient equals the raw gradient and the
    # un-rotation is a no-op, so we fall through to the existing path.
    if grad_kl_rope_per_pair is not None:
        # Un-rotate the rope-space gradient per i: applies R(θ_i)^T to the
        # K dimension while broadcasting over the j-key dimension.
        grad_kl_for_coupling = _un_apply_rope_pair_outer(
            grad_kl_rope_per_pair, base=rope_base
        )
    else:
        grad_kl_for_coupling = grad_kl_per_pair_full

    avg_grad_for_coupling = torch.einsum('bij,bijk->bik', beta, grad_kl_for_coupling)
    grad_deviation = avg_grad_for_coupling.unsqueeze(2) - grad_kl_for_coupling
    d_beta_d_mu = beta.unsqueeze(-1) * grad_deviation / kappa_scaled
    # Multiplier in ∂(β·KL_raw)/∂μ is KL_raw, not KL_RoPE.
    kl_for_coupling = kl_values_raw if kl_values_raw is not None else kl_values
    grad_mu_softmax = lambda_softmax * torch.einsum('bij,bijk->bik', kl_for_coupling, d_beta_d_mu)

    grad_mu = grad_mu_self + grad_mu_direct + grad_mu_softmax

    # Sigma gradients
    grad_sigma_softmax_norm = 0.0
    if grad_sigma_per_pair_full is not None:
        # Direct sigma gradient
        block_start = 0
        for block_idx, d in enumerate(irrep_dims):
            block_end = block_start + d
            grad_sigma_pair = grad_sigma_per_pair_full[:, :, :, block_start:block_end]
            grad_sigma_align[:, :, block_start:block_end] = (
                grad_sigma_align[:, :, block_start:block_end]
                + lambda_belief * torch.einsum('bij,bijk->bik', beta, grad_sigma_pair)
            )
            block_start = block_end

        # Debug: capture direct alignment norm before softmax coupling
        if _vfe_utils_mod._VFE_GRAD_DEBUG is not None:
            _vfe_utils_mod._VFE_GRAD_DEBUG['grad_sigma_align_direct'] = _grad_norm(grad_sigma_align)
            _ps = _per_pos_stats(grad_sigma_align)
            _vfe_utils_mod._VFE_GRAD_DEBUG['grad_sigma_align_pos_mean'] = _ps[0]
            _vfe_utils_mod._VFE_GRAD_DEBUG['grad_sigma_align_pos_max'] = _ps[1]

        # Sigma softmax coupling (use raw-mu KL for consistency, same as mu coupling)
        avg_sigma_grad = torch.einsum('bij,bijk->bik', beta, grad_sigma_per_pair_full)
        sigma_grad_deviation = avg_sigma_grad.unsqueeze(2) - grad_sigma_per_pair_full
        d_beta_d_sigma = beta.unsqueeze(-1) * sigma_grad_deviation / kappa_scaled
        grad_sigma_softmax = lambda_softmax * torch.einsum('bij,bijk->bik', kl_for_coupling, d_beta_d_sigma)
        if _vfe_utils_mod._VFE_GRAD_DEBUG is not None:
            grad_sigma_softmax_norm = _grad_norm(grad_sigma_softmax)
        grad_sigma_align = grad_sigma_align + grad_sigma_softmax

    grad_sigma = grad_sigma_self + grad_sigma_align

    # Debug: final totals (fused path)
    if _vfe_utils_mod._VFE_GRAD_DEBUG is not None:
        _vfe_utils_mod._VFE_GRAD_DEBUG['grad_mu_direct'] = _grad_norm(grad_mu_direct)
        _vfe_utils_mod._VFE_GRAD_DEBUG['grad_mu_softmax'] = _grad_norm(grad_mu_softmax)
        _vfe_utils_mod._VFE_GRAD_DEBUG['grad_sigma_softmax'] = grad_sigma_softmax_norm
        _vfe_utils_mod._VFE_GRAD_DEBUG['kl_pairwise_mean'] = kl_values.mean().item()
        _vfe_utils_mod._VFE_GRAD_DEBUG['kl_pairwise_max'] = kl_values.max().item()
        _vfe_utils_mod._VFE_GRAD_DEBUG['kappa_scaled'] = kappa_scaled
        # Fraction of pairs near the KL ceiling (diagnoses clamp saturation)
        _kl_ceil = max(KL_CEIL_BASE, KL_CEIL_SCALE * K)
        _vfe_utils_mod._VFE_GRAD_DEBUG['kl_frac_above_90pct'] = (kl_values > 0.9 * _kl_ceil).float().mean().item()
        _vfe_utils_mod._VFE_GRAD_DEBUG['kl_p95'] = kl_values.quantile(0.95).item()
        _vfe_utils_mod._VFE_GRAD_DEBUG['grad_mu_total'] = _grad_norm(grad_mu)
        _vfe_utils_mod._VFE_GRAD_DEBUG['grad_sigma_total'] = _grad_norm(grad_sigma)
        _ps = _per_pos_stats(grad_sigma)
        _vfe_utils_mod._VFE_GRAD_DEBUG['grad_sigma_total_pos_mean'] = _ps[0]
        _vfe_utils_mod._VFE_GRAD_DEBUG['grad_sigma_total_pos_max'] = _ps[1]

    kl_out = kl_values if return_kl else None
    return beta.to(dtype), grad_mu.to(dtype), grad_sigma.to(dtype), kl_out


# Chunked VFE gradient path removed — block-diagonal path handles memory
# efficiency via irrep decomposition (always enabled via use_block_diagonal_kl=True).
# See _compute_vfe_gradients_block_diagonal_diag for the active path.
# =============================================================================
# GPU-Based Gradient Computation (PyTorch - FAST!)
# =============================================================================

def compute_vfe_gradients_gpu(
    mu_q: torch.Tensor,        # (B, N, K) belief means
    sigma_q: torch.Tensor,     # (B, N, K) diagonal variances or (B, N, K, K) full
    mu_p: torch.Tensor,        # (B, N, K) prior means
    sigma_p: torch.Tensor,     # (B, N, K) diagonal or (B, N, K, K) full
    beta: torch.Tensor,        # (B, N, N) attention weights
    phi: torch.Tensor,         # (B, N, n_gen) gauge frames where n_gen is # of generators
    generators: torch.Tensor,  # (n_gen, K, K) Lie algebra generators
    alpha: 'float | torch.Tensor' = 0.01,  # Self-coupling weight: scalar or (B, N, K) per-dim Bayesian precision
    lambda_belief: float = 1.0,  # Boltzmann GLU weight (direct: β·∇KL — GELU/SiLU analog)
    lambda_softmax: float = 1.0,  # Attention-variance coupling weight (∂β/∂θ · KL)
    kappa: float = 1.0,        # Temperature (for normalization)
    eps: float = 1e-6,
    alpha_c0: Optional[torch.Tensor] = None,  # (K,) softplus(raw_c0) for product-rule correction when alpha is learnable
    cached_transport: Optional[dict] = None,  # Precomputed transport operators
    compute_sigma_align_grad: bool = True,  # Compute sigma gradient from alignment term
    irrep_dims: Optional[List[int]] = None,  # Block dimensions for block-diagonal processing
    enforce_orthogonal: bool = False,  # If True, enforce Ω ∈ SO(K) via Newton-Schulz
    cached_block_exp_pairs: Optional[list] = None,  # Precomputed block exponential pairs
    exact_diagonal_transport: bool = False,  # Lift diagonal σ for exact transport
    use_rope: bool = False,    # When True, β was softmaxed from KL_RoPE → fix chain rule
    rope_base: float = 10000.0,
    alpha_div: float = 1.0,    # Rényi divergence order (1.0 = KL)
    gauge_covariant_ridge: bool = False,  # If True, ε·I ridges on Σ become ε·(gg^T)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute VFE gradients entirely on GPU using PyTorch.

    Fully vectorized. Supports SO(3), SO(N), and GL(K) gauge groups.
    The number of generators (n_gen) determines the group: 3 for SO(3),
    N(N-1)/2 for SO(N), K^2 for GL(K).

    Gradients computed:
    1. Self-coupling: d/d_mu_q [alpha * KL(q||p)]
    2. Belief alignment: d/d_mu_q [lambda * Sum_j beta_ij * KL(q_i || Omega_ij q_j)]

    The dBeta/dSigma softmax coupling term is always included:
        dF/dSigma_i = Sum_j beta_ij * dKL_ij/dSigma_i + Sum_j KL_ij * dBeta_ij/dSigma_i

    Dispatches to memory-efficient block-diagonal paths when irrep_dims is set:
    - irrep_dims + full cov  -> _compute_vfe_gradients_block_diagonal
    - irrep_dims + diagonal  -> _compute_vfe_gradients_block_diagonal_diag

    Args:
        mu_q: Belief means (B, N, K).
        sigma_q: Belief variances - diagonal (B, N, K) or full (B, N, K, K).
        mu_p: Prior means (B, N, K).
        sigma_p: Prior variances - diagonal (B, N, K) or full (B, N, K, K).
        beta: Attention weights (B, N, N), already normalized.
        phi: Gauge frames (B, N, phi_dim) where phi_dim = n_gen.
        generators: Lie algebra generators (n_gen, K, K).
        alpha: Weight for KL(q||p) self-coupling. Scalar (uniform) or (B, N, K)
            tensor from per-dimension Bayesian precision (learnable_alpha).
        lambda_belief: Weight for belief alignment term.
        kappa: Temperature parameter.
        eps: Numerical stability floor.
        alpha_c0: (K,) softplus(raw_c0) for product-rule correction when alpha
            is a learnable (B, N, K) tensor.
        cached_transport: Optional dict with precomputed 'Omega' (B, N, N, K, K)
            from compute_transport_operators(). Avoids redundant matrix exponentials.
        compute_sigma_align_grad: If True (default), include the sigma alignment
            gradient dKL/dSigma_q = 0.5 * (Sigma_transported^{-1} - Sigma_q^{-1}).
        irrep_dims: Block dimensions [d_1, ...] for block-diagonal KL decomposition.
            Reduces memory from O(N^2 K^2) to O(N^2 * max(d_i^2)).
        enforce_orthogonal: If True, enforce Omega in SO(K) via Newton-Schulz.
        cached_block_exp_pairs: Precomputed (exp_phi, exp_neg_phi) per irrep block.
        exact_diagonal_transport: When True and sigma is diagonal, lifts sigma to
            full covariance via diag_embed for exact gauge transport, then extracts
            the diagonal from the result. Disables fused diagonal paths.
        alpha_div: Rényi divergence order α_d (default 1.0 = KL divergence).
            When != 1.0, both the self-coupling and belief-alignment terms use
            D_{α_d} instead of KL. Only the diagonal and block-diagonal diagonal
            paths implement this; the inline full-covariance path ignores it.

    Returns:
        grad_mu: Gradient w.r.t. mu_q, shape (B, N, K).
        grad_sigma: Gradient w.r.t. sigma_q, shape (B, N, K) diagonal or
            (B, N, K, K) full, matching input.
    """
    # Squeeze trailing singleton dimensions for robustness
    sigma_q = squeeze_trailing_singletons(sigma_q)
    sigma_p = squeeze_trailing_singletons(sigma_p)

    B, N, K = mu_q.shape
    device = mu_q.device
    dtype = mu_q.dtype

    # Detect diagonal vs full covariance
    is_diagonal = sigma_q.dim() == 3

    # Exact diagonal transport: lift to full, use full-cov path, extract diagonal
    if exact_diagonal_transport and is_diagonal:
        sigma_q_full = torch.diag_embed(sigma_q)
        sigma_p_full = torch.diag_embed(sigma_p)
        grad_mu, grad_sigma_full = compute_vfe_gradients_gpu(
            mu_q, sigma_q_full, mu_p, sigma_p_full, beta, phi, generators,
            alpha, lambda_belief, lambda_softmax, kappa, eps, alpha_c0,
            cached_transport, compute_sigma_align_grad, irrep_dims,
            enforce_orthogonal, cached_block_exp_pairs,
            exact_diagonal_transport=False,
            use_rope=use_rope,
            rope_base=rope_base,
            alpha_div=alpha_div,
            gauge_covariant_ridge=gauge_covariant_ridge,
        )
        return grad_mu, torch.diagonal(grad_sigma_full, dim1=-2, dim2=-1)

    # =================================================================
    # MEMORY-EFFICIENT PATH: Block-diagonal processing
    # =================================================================
    if irrep_dims is not None and not is_diagonal:
        return _compute_vfe_gradients_block_diagonal(
            mu_q, sigma_q, mu_p, sigma_p, beta, phi, generators,
            alpha, lambda_belief, lambda_softmax, kappa, eps, irrep_dims,
            compute_sigma_align_grad, enforce_orthogonal,
            alpha_c0=alpha_c0,
            cached_block_exp_pairs=cached_block_exp_pairs,
            use_rope=use_rope,
            rope_base=rope_base,
            alpha_div=alpha_div,
            gauge_covariant_ridge=gauge_covariant_ridge,
        )

    # =================================================================
    # MEMORY-EFFICIENT PATH: Block-diagonal for diagonal covariance
    # =================================================================
    if irrep_dims is not None and is_diagonal:
        return _compute_vfe_gradients_block_diagonal_diag(
            mu_q, sigma_q, mu_p, sigma_p, beta, phi, generators,
            alpha, lambda_belief, lambda_softmax, kappa, eps, irrep_dims,
            compute_sigma_align_grad, enforce_orthogonal,
            alpha_c0=alpha_c0,
            cached_block_exp_pairs=cached_block_exp_pairs,
            use_rope=use_rope,
            rope_base=rope_base,
            alpha_div=alpha_div,
        )

    # The remaining in-line paths (no irrep_dims) do not implement the rope
    # chain-rule fix. Previously this was a warning; now we hard-error because
    # the resulting gradient is biased and the warn-and-continue policy let
    # users silently train models with wrong descent directions. Callers must
    # provide irrep_dims (the per-head block decomposition) to use RoPE.
    if use_rope:
        raise ValueError(
            "use_rope=True without irrep_dims is not supported: this code path "
            "does not implement the rope chain-rule fix, and the softmax-coupling "
            "term would use raw-μ gradients instead of R^T·∂KL_RoPE/∂(R μ), "
            "biasing the descent direction. Pass irrep_dims (per-head block "
            "decomposition) so the per-head loop's chain-rule fix applies, or "
            "set use_rope=False."
        )

    # =================================================================
    # 1. Self-Coupling Gradient: ∂/∂μ_q [α · KL(q||p)]
    # =================================================================
    # For diagonal Gaussians:
    #   KL(q||p) = 0.5 * Σ_k [ σ_q[k]/σ_p[k] + (μ_p[k]-μ_q[k])²/σ_p[k] - 1 + log(σ_p[k]/σ_q[k]) ]
    #   ∂KL/∂μ_q = (μ_q - μ_p) / σ_p
    #   ∂KL/∂σ_q = 0.5 * (1/σ_p - 1/σ_q)

    if is_diagonal:
        # Force float32 for all sigma divisions, logs, and KL computation.
        # The Omega einsum and mu transport can stay in AMP dtype for speed,
        # but sigma ratios and log-det terms need float32 precision.
        _orig_dtype = sigma_q.dtype
        sigma_q = sigma_q.float()
        sigma_p = sigma_p.float()
        mu_q = mu_q.float()
        mu_p = mu_p.float()
        beta = beta.float()

        # Clamp for stability
        sigma_q_safe = sigma_q.clamp(min=eps)
        sigma_p_safe = sigma_p.clamp(min=eps)

        # Self-coupling gradient w.r.t. μ and σ
        delta_mu = mu_q - mu_p  # (B, N, K)
        if alpha_div == 1.0:
            grad_mu_self = alpha * delta_mu / sigma_p_safe  # (B, N, K)
            grad_sigma_self = alpha * 0.5 * (1.0 / sigma_p_safe - 1.0 / sigma_q_safe)  # (B, N, K)

            # Product-rule correction: ∂(α·KL)/∂θ = α·∂KL/∂θ + (∂α/∂θ)·KL
            # When α_k = c₀_k/(b₀_k + kl_k), ∂α_k/∂θ = -α_k²/c₀_k · ∂kl_k/∂θ
            if alpha_c0 is not None and isinstance(alpha, torch.Tensor):
                kl_k = 0.5 * (sigma_q_safe / sigma_p_safe + delta_mu ** 2 / sigma_p_safe
                              - 1.0 + torch.log(sigma_p_safe) - torch.log(sigma_q_safe))
                kl_k = kl_k.clamp(min=0.0)
                # ∂α/∂μ · KL = -α²/c₀ · (δμ/σ_p) · kl_k
                grad_mu_self = grad_mu_self - (alpha ** 2 / alpha_c0) * kl_k * delta_mu / sigma_p_safe
                # ∂α/∂σ_q · KL = -α²/c₀ · 0.5·(1/σ_p - 1/σ_q) · kl_k
                grad_sigma_self = grad_sigma_self - (alpha ** 2 / alpha_c0) * kl_k * 0.5 * (1.0 / sigma_p_safe - 1.0 / sigma_q_safe)
        else:
            # α-divergence self-coupling gradient (inline diagonal path)
            sigma_blend_self = ((1.0 - alpha_div) * sigma_q_safe + alpha_div * sigma_p_safe).clamp(min=eps)
            grad_mu_self = alpha * alpha_div * delta_mu / sigma_blend_self
            grad_sigma_self = alpha * (
                -alpha_div * (sigma_p_safe - sigma_q_safe) / (2.0 * sigma_q_safe * sigma_blend_self)
                - alpha_div * (1.0 - alpha_div) * delta_mu ** 2 / (2.0 * sigma_blend_self ** 2)
            )
            # Product-rule correction (Rényi branch): α_k = c₀_k/(b₀_k + D_α,k) ⇒
            # ∂α_k/∂θ = -α_k²/c₀ · ∂D_α,k/∂θ.
            if alpha_c0 is not None and isinstance(alpha, torch.Tensor):
                mahal_k = alpha_div * delta_mu ** 2 / sigma_blend_self
                logdet_k = (
                    (1.0 - alpha_div) * torch.log(sigma_q_safe)
                    + alpha_div * torch.log(sigma_p_safe)
                    - torch.log(sigma_blend_self)
                ) / (alpha_div - 1.0)
                d_alpha_k = 0.5 * (mahal_k + logdet_k).clamp(min=0.0)
                grad_mu_self = grad_mu_self - (alpha ** 2 / alpha_c0) * d_alpha_k * (
                    alpha_div * delta_mu / sigma_blend_self
                )
                grad_sigma_self = grad_sigma_self - (alpha ** 2 / alpha_c0) * d_alpha_k * (
                    -alpha_div * (sigma_p_safe - sigma_q_safe) / (2.0 * sigma_q_safe * sigma_blend_self)
                    - alpha_div * (1.0 - alpha_div) * delta_mu ** 2 / (2.0 * sigma_blend_self ** 2)
                )
    else:
        # Full covariance - use matrix operations
        # ∂KL/∂μ_q = Σ_p^{-1} (μ_q - μ_p)
        sigma_p_inv = _safe_spd_inv(sigma_p, eps=eps)

        delta_mu = mu_q - mu_p  # (B, N, K)
        grad_mu_self = alpha * torch.einsum('bnij,bnj->bni', sigma_p_inv, delta_mu)

        # ∂KL/∂Σ_q = 0.5 * (Σ_p^{-1} - Σ_q^{-1})
        sigma_q_inv = _safe_spd_inv(sigma_q, eps=eps)
        # For full covariance (4D), alpha (B,N,K) needs extra dim to broadcast with (B,N,K,K)
        alpha_4d = alpha.unsqueeze(-1) if isinstance(alpha, torch.Tensor) else alpha
        grad_sigma_self = alpha_4d * 0.5 * (sigma_p_inv - sigma_q_inv)

        # Product-rule correction for learnable alpha (full covariance):
        # ∂(α·KL)/∂θ = α·∂KL/∂θ + (∂α/∂θ)·KL
        # Per-dimension KL via eigendecomposition of Σ_p^{-1/2} Σ_q Σ_p^{-1/2},
        # which is symmetric with eigenvalues λ_k giving per-mode contributions
        # kl_mode_k = 0.5(λ_k - 1 - log λ_k). Projected back to original basis
        # via eigenvector magnitudes.
        if alpha_c0 is not None and isinstance(alpha, torch.Tensor):
            sp_inv_delta = torch.einsum('bnij,bnj->bni', sigma_p_inv, delta_mu)
            mahal_k = delta_mu * sp_inv_delta  # (B, N, K) — per-dim Mahalanobis

            # Per-mode KL from eigendecomposition of L^{-1} Σ_q L^{-T}
            # where L = cholesky(Σ_p). Eigenvalues λ_k of this symmetric matrix
            # equal eigenvalues of Σ_p^{-1} Σ_q.
            try:
                L_p = torch.linalg.cholesky(sigma_p.float())
                # M = L^{-1} Σ_q L^{-T} (symmetric, same eigenvalues as Σ_p^{-1} Σ_q)
                Lp_inv_Sq = torch.linalg.solve_triangular(L_p, sigma_q.float(), upper=False)
                M = torch.linalg.solve_triangular(
                    L_p, Lp_inv_Sq.transpose(-1, -2), upper=False
                ).transpose(-1, -2)
                eigvals, eigvecs = torch.linalg.eigh(M)  # (B,N,K), (B,N,K,K)
                eigvals = eigvals.clamp(min=eps).to(sigma_q.dtype)
                eigvecs = eigvecs.to(sigma_q.dtype)
                # Per-eigenmode KL: kl_mode_k = 0.5(λ_k - 1 - log λ_k) ≥ 0
                kl_mode = 0.5 * (eigvals - 1.0 - torch.log(eigvals))  # (B, N, K)
                # Project to original basis via L^{-T} V: kl_orig_k = Σ_m |V_km|² kl_mode_m
                # eigvecs are in the L^{-1} basis; |V_km|² distributes modes to original dims
                V_sq = eigvecs ** 2  # (B, N, K, K) — squared eigenvector components
                kl_k_trace_logdet = torch.einsum('bnkm,bnm->bnk', V_sq, kl_mode)
                kl_k = (kl_k_trace_logdet + 0.5 * mahal_k).clamp(min=0.0)
            except RuntimeError:
                # Fallback: uniform logdet distribution (original approximation)
                prod_qp = torch.matmul(sigma_p_inv, sigma_q)
                trace_k = prod_qp.diagonal(dim1=-2, dim2=-1)
                logdet_p = torch.linalg.slogdet(sigma_p.float())[1]
                logdet_q = torch.linalg.slogdet(sigma_q.float())[1]
                logdet_k = ((logdet_p - logdet_q) / K).unsqueeze(-1).expand_as(delta_mu)
                kl_k = 0.5 * (trace_k + mahal_k - 1 + logdet_k).clamp(min=0.0)

            grad_mu_self = grad_mu_self - (alpha ** 2 / alpha_c0) * kl_k * sp_inv_delta
            correction_scale = ((alpha ** 2 / alpha_c0) * kl_k).unsqueeze(-1)  # (B, N, K, 1)
            grad_sigma_self = grad_sigma_self - correction_scale * 0.5 * (sigma_p_inv - sigma_q_inv)

    # =================================================================
    # 2. Belief Alignment Gradient: ∂/∂μ_i [λ · Σ_j β_ij · KL(q_i || Ω_ij q_j)]
    # =================================================================
    # Full gradient via product rule:
    #   ∂/∂μ_i [Σ_j β_ij · KL_ij] = Σ_j β_ij · ∂KL_ij/∂μ_i + Σ_j KL_ij · ∂β_ij/∂μ_i
    #                                  ↑ Boltzmann GLU          ↑ attention-variance coupling
    #                                  (GELU/SiLU analog)       (second-order correction)
    #
    # The direct term is the Boltzmann GLU: β gates ∇KL (structural analog of GELU).
    # The softmax coupling is a higher-order correction — attention-variance coupling:
    #   ∂β_ij/∂μ_i = -β_ij · [∂KL_ij/∂μ_i - Σ_k β_ik · ∂KL_ik/∂μ_i] / κ

    if is_diagonal:
        # Get transport operators (use cached if available)
        if cached_transport is not None and 'Omega' in cached_transport:
            Omega = cached_transport['Omega']
        else:
            # Compute transport operators (vectorized)
            phi_matrix = torch.einsum('bna,aij->bnij', phi, generators)  # (B, N, K, K)
            exp_phi, exp_neg_phi = stable_matrix_exp_pair(phi_matrix)

            # Re-orthogonalization for SO(K) if requested
            if enforce_orthogonal and K >= 2:
                exp_phi = newton_schulz_orthogonalize(exp_phi)
                exp_neg_phi = newton_schulz_orthogonalize(exp_neg_phi)

            # Transport: Ω_ij = exp(φ_i) @ exp(-φ_j)
            # For all pairs: (B, N, N, K, K)
            Omega = torch.einsum('bikl,bjlm->bijkm', exp_phi, exp_neg_phi)

        # Transport all μ_j to frame i: μ_j_transported[i,j] = Ω_ij @ μ_j
        mu_j_transported = torch.einsum('bijkl,bjl->bijk', Omega, mu_q)  # (B, N, N, K)

        # Difference: μ_i - μ_j_transported (for each pair i,j)
        delta_mu_ij = mu_q.unsqueeze(2) - mu_j_transported  # (B, N, N, K)

        # =================================================================
        # DIAGONAL COVARIANCE TRANSPORT: diag(Ω @ diag(σ_j) @ Ω^T)
        # =================================================================
        # For diagonal input, the diagonal of the transported covariance is:
        #   σ_j_transported[k] = Σ_l Ω_kl² * σ_j[l]
        # This avoids materializing any (B, N, N, K, K) covariance tensors.

        sigma_q_safe = sigma_q.clamp(min=eps)  # (B, N, K)

        # Transported diagonal covariance via 3-operand einsum (no Omega² intermediate)
        sigma_j_transported_diag = torch.einsum(
            'bijkl,bijkl,bjl->bijk', Omega, Omega, sigma_q_safe
        ).clamp(min=eps)  # (B, N, N, K)

        sigma_i_expanded = sigma_q_safe[:, :, None, :]  # (B, N, 1, K) - broadcasts

        if alpha_div == 1.0:
            # Standard KL alignment gradient
            grad_kl_per_pair = delta_mu_ij / sigma_j_transported_diag  # (B, N, N, K)

            # Diagonal KL: trace + Mahalanobis + logdet terms
            trace_term = (sigma_i_expanded / sigma_j_transported_diag).sum(dim=-1)
            mahal_term = (delta_mu_ij ** 2 / sigma_j_transported_diag).sum(dim=-1)
            logdet_term = (torch.log(sigma_j_transported_diag.clamp(min=eps))
                           - torch.log(sigma_i_expanded.clamp(min=eps))).sum(dim=-1)
            kl_values = 0.5 * (trace_term + mahal_term - K + logdet_term)
        else:
            # α-divergence alignment gradient
            sigma_blend_align = (
                (1.0 - alpha_div) * sigma_i_expanded + alpha_div * sigma_j_transported_diag
            ).clamp(min=eps)  # (B, N, N, K)
            grad_kl_per_pair = alpha_div * delta_mu_ij / sigma_blend_align  # (B, N, N, K)

            mahal_term = (alpha_div * delta_mu_ij ** 2 / sigma_blend_align).sum(dim=-1)
            logdet_term = (
                (1.0 - alpha_div) * torch.log(sigma_i_expanded.clamp(min=eps))
                + alpha_div * torch.log(sigma_j_transported_diag.clamp(min=eps))
                - torch.log(sigma_blend_align)
            ).sum(dim=-1) / (alpha_div - 1.0)
            kl_values = 0.5 * (mahal_term + logdet_term)

        kl_ceil = max(KL_CEIL_BASE, KL_CEIL_SCALE * K)
        kl_values = kl_values.clamp(min=0.0, max=kl_ceil)  # (B, N, N)

        # =================================================================
        # 2a. Direct term: Σ_j β_ij · ∂D_ij/∂μ_i
        # =================================================================
        avg_grad = torch.einsum('bij,bijk->bik', beta, grad_kl_per_pair)  # (B, N, K)
        grad_mu_direct = lambda_belief * avg_grad

        # =================================================================
        # 2b. Softmax coupling term (THE NONLINEARITY!):
        #     ∂β_ij/∂μ_i = -β_ij · [∂D_ij/∂μ_i - Σ_k β_ik · ∂D_ik/∂μ_i] / κ
        #     grad_softmax = Σ_j D_ij · ∂β_ij/∂μ_i
        # =================================================================
        grad_deviation = avg_grad.unsqueeze(2) - grad_kl_per_pair  # (B, N, N, K)
        kappa_scaled = max(kappa * math.sqrt(max(K, 1)), eps)
        d_beta_d_mu = beta.unsqueeze(-1) * grad_deviation / kappa_scaled  # (B, N, N, K)
        grad_mu_softmax = lambda_softmax * torch.einsum('bij,bijk->bik', kl_values, d_beta_d_mu)

        # Total alignment gradient (direct + softmax coupling)
        grad_mu_align = grad_mu_direct + grad_mu_softmax

        # =================================================================
        # Sigma gradient from alignment term (diagonal case)
        # =================================================================
        if compute_sigma_align_grad:
            if alpha_div == 1.0:
                sigma_j_inv_diag = 1.0 / sigma_j_transported_diag  # (B, N, N, K)
                sigma_i_inv = 1.0 / sigma_q_safe  # (B, N, K)
                grad_sigma_per_pair = 0.5 * (sigma_j_inv_diag - sigma_i_inv[:, :, None, :])
            else:
                # α-divergence σ gradient
                sigma_blend_s = (
                    (1.0 - alpha_div) * sigma_i_expanded + alpha_div * sigma_j_transported_diag
                ).clamp(min=eps)
                grad_sigma_per_pair = (
                    -alpha_div * (sigma_j_transported_diag - sigma_i_expanded)
                    / (2.0 * sigma_i_expanded.clamp(min=eps) * sigma_blend_s)
                    - alpha_div * (1.0 - alpha_div) * delta_mu_ij ** 2 / (2.0 * sigma_blend_s ** 2)
                )

            # Direct term: Σ_j β_ij * ∂D_ij/∂σ_i
            grad_sigma_direct = lambda_belief * torch.einsum('bij,bijk->bik', beta, grad_sigma_per_pair)

            # Softmax coupling: Σ_j D_ij · ∂β_ij/∂σ_i
            avg_sigma_grad = torch.einsum('bij,bijk->bik', beta, grad_sigma_per_pair)
            sigma_grad_deviation = avg_sigma_grad.unsqueeze(2) - grad_sigma_per_pair
            d_beta_d_sigma = beta.unsqueeze(-1) * sigma_grad_deviation / kappa_scaled
            grad_sigma_softmax = lambda_softmax * torch.einsum('bij,bijk->bik', kl_values, d_beta_d_sigma)
            grad_sigma_align = grad_sigma_direct + grad_sigma_softmax
        else:
            # Simplified: no sigma gradient from alignment (legacy behavior)
            grad_sigma_align = torch.zeros_like(sigma_q)
    else:
        # Full covariance belief alignment
        # Get transport operators (use cached if available)
        if cached_transport is not None and 'Omega' in cached_transport:
            Omega = cached_transport['Omega']
        else:
            phi_matrix = torch.einsum('bna,aij->bnij', phi, generators)
            exp_phi, exp_neg_phi = stable_matrix_exp_pair(phi_matrix)

            # Re-orthogonalization for SO(K) if requested
            if enforce_orthogonal and K >= 2:
                exp_phi = newton_schulz_orthogonalize(exp_phi)
                exp_neg_phi = newton_schulz_orthogonalize(exp_neg_phi)

            Omega = torch.einsum('bikl,bjlm->bijkm', exp_phi, exp_neg_phi)

        # Transport means
        mu_j_transported = torch.einsum('bijkl,bjl->bijk', Omega, mu_q)
        delta_mu_ij = mu_q.unsqueeze(2) - mu_j_transported

        # Transport covariances: Σ_j_transported = Ω @ Σ_j @ Ω^T
        sigma_j_transported = torch.einsum(
            'bijkl,bjlm,bijmn->bijkn',
            Omega, sigma_q, Omega.transpose(-1, -2)
        )  # (B, N, N, K, K)
        # Symmetrize: the sandwich product is mathematically symmetric but
        # float32 matrix-multiply roundoff leaves O(eps_machine) off-diagonal
        # asymmetry that causes needless Cholesky fallbacks at line ~1585.
        # Matches the symmetrization pattern in the fused block-diag path.
        sigma_j_transported = 0.5 * (sigma_j_transported + sigma_j_transported.transpose(-1, -2))

        # Regularize and invert. Unconditional torch.where + cached eye
        # avoids a host sync per call on the full-covariance natural-gradient
        # path; the diagnostic counter is gated by the diagnostics flag.
        eye_K_cached = _cached_eye(K, device, dtype)
        sigma_j_reg = sigma_j_transported + max(eps, TRANSPORT_JITTER) * eye_K_cached
        _sigma_reg_nan = torch.isnan(sigma_j_reg)
        if _kl_diagnostics_enabled() and bool(_sigma_reg_nan.any().item()):
            _nr("vfe_fullcov_sigma_reg_nan")
        sigma_j_reg = torch.where(_sigma_reg_nan, eye_K_cached.expand_as(sigma_j_reg), sigma_j_reg)
        sigma_j_inv = _safe_spd_inv(sigma_j_reg, eps=eps)  # (B, N, N, K, K)

        # ∂KL_ij/∂μ_i
        grad_kl_per_pair = torch.einsum('bijkl,bijl->bijk', sigma_j_inv, delta_mu_ij)

        # Compute FULL KL values (not just Mahalanobis - include trace and logdet!)
        # KL(q_i || Ω_ij[q_j]) = 0.5 * (tr(Σ_j_t^{-1} Σ_i) + mahal - K + log|Σ_j_t| - log|Σ_i|)

        # Mahalanobis term: δμ^T @ Σ_j_transported^{-1} @ δμ
        mahal_term = torch.einsum('bijk,bijk->bij', delta_mu_ij, grad_kl_per_pair)  # (B, N, N)

        # Trace term: tr(Σ_j_transported^{-1} @ Σ_i)
        # Use .clone() after expand for contiguous memory layout
        sigma_i_expanded = sigma_q[:, :, None, :, :].expand(-1, -1, N, -1, -1)  # (B, N, N, K, K) — einsum accepts non-contiguous
        trace_term = torch.einsum('bijkk->bij', torch.einsum('bijkl,bijlm->bijkm', sigma_j_inv, sigma_i_expanded))

        # Log-determinant terms using Cholesky with fallback
        try:
            L_j_t = torch.linalg.cholesky(sigma_j_reg)  # (B, N, N, K, K)
            logdet_j_t = 2.0 * torch.sum(torch.log(torch.diagonal(L_j_t, dim1=-2, dim2=-1) + eps), dim=-1)  # (B, N, N)
        except RuntimeError:
            eigvals = torch.linalg.eigvalsh(sigma_j_reg)
            logdet_j_t = torch.sum(torch.log(eigvals.clamp(min=eps)), dim=-1)

        sigma_i_reg = sigma_q + eps * eye_K_cached
        try:
            L_i = torch.linalg.cholesky(sigma_i_reg)  # (B, N, K, K)
            logdet_i = 2.0 * torch.sum(torch.log(torch.diagonal(L_i, dim1=-2, dim2=-1) + eps), dim=-1)  # (B, N)
        except RuntimeError:
            eigvals = torch.linalg.eigvalsh(sigma_i_reg)
            logdet_i = torch.sum(torch.log(eigvals.clamp(min=eps)), dim=-1)
        # Use .clone() after expand for contiguous memory layout
        logdet_i_expanded = logdet_i[:, :, None].expand(-1, -1, N)  # (B, N, N) — broadcast view

        # Full KL divergence
        kl_values = 0.5 * (trace_term + mahal_term - K + logdet_j_t - logdet_i_expanded)
        # Clamp KL to [0, max] for numerical stability (scale ceiling with K)
        kl_ceil = max(KL_CEIL_BASE, KL_CEIL_SCALE * K)
        kl_values = kl_values.clamp(min=0.0, max=kl_ceil)  # (B, N, N)

        # avg_grad = Σ_k β_ik · ∂KL_ik/∂μ_i (used for both direct and softmax terms)
        avg_grad = torch.einsum('bij,bijk->bik', beta, grad_kl_per_pair)
        grad_mu_direct = lambda_belief * avg_grad

        # Softmax coupling term
        # Scale kappa by √K to match attention temperature scaling (τ = √K)
        kappa_scaled = max(kappa * math.sqrt(max(K, 1)), eps)
        grad_deviation = avg_grad.unsqueeze(2) - grad_kl_per_pair
        d_beta_d_mu = beta.unsqueeze(-1) * grad_deviation / kappa_scaled
        grad_mu_softmax = lambda_softmax * torch.einsum('bij,bijk->bik', kl_values, d_beta_d_mu)

        grad_mu_align = grad_mu_direct + grad_mu_softmax

        # =================================================================
        # Sigma gradient from alignment term (full covariance case)
        # ∂KL/∂Σ_i = 0.5 * (Σ_j_transported^{-1} - Σ_i^{-1})
        # Weighted by attention: Σ_j β_ij * ∂KL_ij/∂Σ_i
        # =================================================================
        if compute_sigma_align_grad:
            # Use Σ_i^{-1} computed earlier in self-coupling section (sigma_q_inv)
            # Use .clone() after expand for contiguous memory layout
            sigma_i_inv_expanded = sigma_q_inv[:, :, None, :, :].expand(-1, -1, N, -1, -1)  # (B, N, N, K, K) — einsum accepts non-contiguous

            # Gradient per pair: 0.5 * (Σ_j_transported^{-1} - Σ_i^{-1})
            grad_sigma_per_pair = 0.5 * (sigma_j_inv - sigma_i_inv_expanded)  # (B, N, N, K, K)

            # Direct term: Σ_j β_ij * ∂KL_ij/∂Σ_i
            grad_sigma_direct = lambda_belief * torch.einsum('bij,bijkl->bikl', beta, grad_sigma_per_pair)  # (B, N, K, K)

            # Softmax coupling for full covariance (always enabled)
            avg_sigma_grad = torch.einsum('bij,bijkl->bikl', beta, grad_sigma_per_pair)  # (B, N, K, K)
            sigma_grad_deviation = avg_sigma_grad.unsqueeze(2) - grad_sigma_per_pair  # (B, N, N, K, K)
            d_beta_d_sigma = beta.unsqueeze(-1).unsqueeze(-1) * sigma_grad_deviation / kappa_scaled  # (B, N, N, K, K)
            grad_sigma_softmax = lambda_softmax * torch.einsum('bij,bijkl->bikl', kl_values, d_beta_d_sigma)  # (B, N, K, K)
            grad_sigma_align = grad_sigma_direct + grad_sigma_softmax
        else:
            # Simplified: no sigma gradient from alignment (legacy behavior)
            grad_sigma_align = torch.zeros_like(sigma_q)

    # =================================================================
    # 3. Combine Gradients
    # =================================================================
    grad_mu = grad_mu_self + grad_mu_align
    grad_sigma = grad_sigma_self + grad_sigma_align

    # Cast back from float32 (diagonal path upcasts for numerical safety under AMP)
    return grad_mu.to(dtype), grad_sigma.to(dtype)


def compute_natural_gradient_gpu(
    grad_mu: torch.Tensor,     # (B, N, K) Euclidean gradient
    grad_sigma: torch.Tensor,  # (B, N, K) or (B, N, K, K)
    sigma_q: torch.Tensor,     # (B, N, K) or (B, N, K, K)
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Project Euclidean gradients to natural gradients using Fisher metric.

    For Gaussian distributions, the Fisher information metric is:
        F_μ = Σ^{-1}  →  natural_grad_μ = Σ @ euclidean_grad_μ
        F_σ = (1/2)Σ^{-2} →  natural_grad_σ = 2 * Σ² @ euclidean_grad_σ (diagonal approx)

    Derivation: The Fisher metric on the covariance Σ of a Gaussian is
    g(δΣ₁, δΣ₂) = (1/2) tr(Σ⁻¹ δΣ₁ Σ⁻¹ δΣ₂). For diagonal Σ = diag(σ),
    this gives g_{kk} = 1/(2σ_k²), so g^{kk} = 2σ_k².

    Args:
        grad_mu: Euclidean gradient w.r.t. μ
        grad_sigma: Euclidean gradient w.r.t. σ
        sigma_q: Current covariance
        eps: Numerical stability

    Returns:
        nat_grad_mu: Natural gradient for μ
        nat_grad_sigma: Natural gradient for σ
    """
    # Squeeze trailing singleton dimensions for robustness
    sigma_q = squeeze_trailing_singletons(sigma_q)

    is_diagonal = sigma_q.dim() == 3
    orig_dtype = sigma_q.dtype

    # Force float32: sigma^2 products and small sigma divisions break in float16
    with torch.amp.autocast('cuda', enabled=False):
        sigma_q = sigma_q.float()
        grad_mu = grad_mu.float()
        grad_sigma = grad_sigma.float()

        if is_diagonal:
            # Diagonal case: simple element-wise multiplication
            sigma_safe = sigma_q.clamp(min=eps)
            nat_grad_mu = sigma_safe * grad_mu  # (B, N, K)
            nat_grad_sigma = 2.0 * sigma_safe * sigma_safe * grad_sigma  # (B, N, K)
        else:
            # Full covariance: matrix multiplication
            nat_grad_mu = torch.einsum('bnij,bnj->bni', sigma_q, grad_mu)
            # Full Fisher natural gradient: δΣ = 2 * Σ @ ∇_Σ @ Σ
            nat_grad_sigma = 2.0 * torch.einsum('bnij,bnjk,bnkl->bnil', sigma_q, grad_sigma, sigma_q)
            # Explicit symmetrization: the sandwich Σ @ ∇_Σ @ Σ is
            # mathematically symmetric iff ∇_Σ is symmetric, which holds
            # analytically but can drift from roundoff or from malformed
            # upstream gradients.  Symmetrizing here guarantees the M-step
            # sigma update keeps Σ on the SPD manifold.
            nat_grad_sigma = 0.5 * (nat_grad_sigma + nat_grad_sigma.transpose(-1, -2))

    return nat_grad_mu.to(orig_dtype), nat_grad_sigma.to(orig_dtype)


# =============================================================================
# Experimental: rope_full_gauge gradient path
# =============================================================================

def _compute_rope_full_gauge_gradient_per_head(
    mu_h: torch.Tensor,         # (B, N, d_h) per-head means
    sigma_h: torch.Tensor,      # (B, N, d_h) diagonal or (B, N, d_h, d_h) full covariance
    mu_p_h: torch.Tensor,       # (B, N, d_h) per-head prior means
    sigma_p_h: torch.Tensor,    # (B, N, d_h) diagonal or (B, N, d_h, d_h) full prior covariance
    phi: torch.Tensor,          # (B, N, n_gen) gauge frames (per-head, after slicing)
    gen_h: torch.Tensor,        # (n_gen, d_h, d_h) per-head generators
    alpha: 'float | torch.Tensor',
    lambda_belief: float,
    lambda_softmax: float,
    kappa: 'float | torch.Tensor',
    eps: float,
    rope_base: float,
    d_h: int,
    cached_block_exp_pairs: Optional[list] = None,
    enforce_orthogonal: bool = False,
    mask: Optional[torch.Tensor] = None,
    mask_self_attention: bool = False,
    gauge_covariant_ridge: bool = False,
    alpha_c0: Optional[torch.Tensor] = None,  # (d_h,) per-head softplus(raw_c0) for product-rule correction when alpha is learnable
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""EXPERIMENTAL: per-head VFE gradient under the rope_full_gauge interpretation.

    Implements the framework-consistent interpretation of RoPE: the
    position-dependent gauge transport R(θ_i) acts on Gaussian beliefs by
    BOTH μ → R μ AND Σ → R Σ R^T (the standard sandwich product).  This
    matches the GL(K) manuscript's derivation of RoPE as Ω restricted to
    SO(2)^{d_k/2} ⊂ GL(K).

    The default rope path (rope_full_gauge=False) follows the standard-
    transformer pattern: rotate only μ, leave Σ raw.  This function provides
    the alternative for empirical comparison.

    Implementation: applies the rope rotation to both μ and Σ (lifting
    diagonal Σ to full covariance if needed), transports the source belief
    by Ω^learned, computes the full-covariance KL, and uses
    torch.autograd.grad to obtain ∂F/∂μ_raw and ∂F/∂σ_raw.  Slower than
    the analytical path but avoids hand-deriving the chain rule through
    R Σ R^T.

    Args:
        mu_h: per-head belief means (B, N, d_h).
        sigma_h: per-head belief covariance — (B, N, d_h) diagonal or
            (B, N, d_h, d_h) full covariance.
        mu_p_h, sigma_p_h: per-head prior mean / covariance (same shape
            convention as sigma_h).
        phi: gauge frames (B, N, n_gen) — pre-sliced to per-head if needed.
        gen_h: per-head generators (n_gen, d_h, d_h).
        alpha: self-coupling weight (scalar or per-dim Bayesian precision).
        lambda_belief, lambda_softmax: alignment direct/coupling weights.
        kappa: per-head temperature.
        eps: numerical stability floor.
        rope_base: RoPE frequency base (must match _apply_rope's base).
        d_h: head dimension.
        cached_block_exp_pairs: optional precomputed (exp_phi, exp_neg_phi) per
            head — single-element list with the per-head pair.
        enforce_orthogonal: passed to fused_block_matrix_exp_pairs if needed.
        mask: optional causal mask.
        mask_self_attention: if True, mask the diagonal of the attention matrix.

    Returns:
        beta_h: per-head attention weights (B, N, N) computed from rope-full KL.
        grad_mu_h: ∂F_total/∂μ_h, shape (B, N, d_h).
        grad_sigma_h: ∂F_total/∂σ_h — (B, N, d_h) if diagonal, (B, N, d_h, d_h) if full.
    """

    B, N, _ = mu_h.shape
    device = mu_h.device

    # IMPORTANT: this path uses torch.autograd.grad internally.  When the
    # caller is inside torch.no_grad() (e.g. during validation), autograd is
    # globally disabled and `requires_grad_(True)` is silently ignored — the
    # subsequent autograd.grad call would fail with "element 0 of tensors does
    # not require grad and does not have a grad_fn".  We force-enable grad
    # tracking for the duration of this function via torch.enable_grad().
    _is_diag = sigma_h.dim() == 3

    with torch.enable_grad():
        # Cast to float32 for stability and detach + require grad for autograd-based path.
        _f32 = torch.float32
        mu_var = mu_h.detach().to(_f32).requires_grad_(True)
        sigma_var = sigma_h.detach().to(_f32).requires_grad_(True)
        mu_p_h_d = mu_p_h.detach().to(_f32)
        sigma_p_h_d = sigma_p_h.detach().to(_f32)
        phi_d = phi.detach().to(_f32)

        # Lift diagonal sigma to full covariance for the rope rotation,
        # or pass through if already full.
        if _is_diag:
            sigma_full = torch.diag_embed(sigma_var)  # (B, N, d_h, d_h)
        else:
            sigma_full = sigma_var                     # already (B, N, d_h, d_h)

        # Apply rope rotation to both means and (lifted) covariances
        mu_rope = _apply_rope(mu_var, base=rope_base)             # (B, N, d_h)
        sigma_rope = _apply_rope_to_covariance(sigma_full, base=rope_base)  # (B, N, d_h, d_h)

        # Compute per-head Ω^learned via fused matrix exp pairs.
        #
        # CRITICAL: we MUST detach the cached exp_phi/exp_neg_phi tensors.
        # When update_phi_per_iteration=False and amortized_inference=True,
        # the outer caller passes cached pairs that are STILL connected to
        # the phi autograd graph (phi was NOT detached before caching).  If
        # we use them as-is in the autograd-traced einsum below, the
        # torch.autograd.grad(retain_graph=False) call at the bottom of
        # this function will traverse and FREE the phi → exp_phi → Omega
        # subgraph.  The subsequent M-step loss.backward() would then fail
        # with "Trying to backward through the graph a second time" or
        # "saved tensors freed" when it encounters the same exp_phi tensor
        # from any other downstream path (e.g., the attention sublayer).
        # Detaching here is semantically correct: phi is treated as a
        # constant within the E-step gradient computation (the grad is
        # w.r.t. mu and sigma only).
        if cached_block_exp_pairs is not None:
            exp_phi_h, exp_neg_phi_h = cached_block_exp_pairs[0]
            exp_phi_h = exp_phi_h.detach()
            exp_neg_phi_h = exp_neg_phi_h.detach()
        else:
            bep = fused_block_matrix_exp_pairs(
                phi_d, gen_h, [d_h],
                enforce_orthogonal=enforce_orthogonal,
            )
            exp_phi_h, exp_neg_phi_h = bep[0]
            exp_phi_h = exp_phi_h.detach()
            exp_neg_phi_h = exp_neg_phi_h.detach()

        # Build pairwise transport: Ω_ij = exp(φ_i) · exp(-φ_j)
        Omega = torch.einsum('bikl,bjlm->bijkm', exp_phi_h, exp_neg_phi_h)  # (B, N, N, d_h, d_h)

        # Transport rope-rotated j-belief to i's frame:
        #   μ_t = Ω · μ_j_rope
        #   Σ_t = Ω · Σ_j_rope · Ω^T
        mu_t = torch.einsum('bijkl,bjl->bijk', Omega, mu_rope)  # (B, N, N, d_h)
        sigma_t = torch.einsum(
            'bijkl,bjlm,bijnm->bijkn',
            Omega, sigma_rope, Omega
        )  # (B, N, N, d_h, d_h)
        # Numerical stability: symmetrize and add eps to diagonal
        sigma_t = 0.5 * (sigma_t + sigma_t.transpose(-1, -2))
        eye_dh = _cached_eye(d_h, device, _f32)
        # Gauge-covariant ridge: Σ_t and Σ_i_rope both live in position-i's
        # frame (Σ_t after Ω Σ_j Ω^T transport, Σ_i_rope is Σ_i rotated by RoPE
        # which does not change the gauge-transport frame at position i).
        if gauge_covariant_ridge:
            _ep_h_f32 = exp_phi_h.to(dtype=_f32)
            _ep_bcast_dh = _ep_h_f32[:, :, None, :, :].expand(-1, -1, N, -1, -1)
            _R_dh_per_pair = _ep_bcast_dh @ _ep_bcast_dh.transpose(-1, -2)
        else:
            _R_dh_per_pair = eye_dh
        sigma_t = sigma_t + eps * _R_dh_per_pair

        # Source-side rope-rotated covariance, broadcast across j
        sigma_i_rope = sigma_rope.unsqueeze(2).expand(-1, -1, N, -1, -1)  # (B, N, N, d_h, d_h)
        sigma_i_rope = 0.5 * (sigma_i_rope + sigma_i_rope.transpose(-1, -2)) + eps * _R_dh_per_pair

        # Full-covariance KL between two Gaussians:
        # KL(p || q) = 0.5 [tr(Σ_q^{-1} Σ_p) + (μ_q - μ_p)^T Σ_q^{-1} (μ_q - μ_p)
        #                   - d + log|Σ_q| - log|Σ_p|]
        # Here p = q_i_rope, q = transported_q_j_rope
        # Use _safe_spd_inv (Cholesky-backed after C6 refactor) for numerical
        # stability: sigma_t is symmetrized + eps-regularized above, so it is
        # SPD by construction, but torch.linalg.inv has no Cholesky fallback
        # and can produce NaN on near-singular GL(K) transports.
        sigma_t_inv = _safe_spd_inv(sigma_t, eps=eps)

        # Trace term: tr(Σ_t^{-1} Σ_i_rope) per (i, j)
        trace_term = torch.einsum('bijkl,bijlk->bij', sigma_t_inv, sigma_i_rope)

        # Mahalanobis: (μ_t - μ_i_rope)^T Σ_t^{-1} (μ_t - μ_i_rope)
        delta = mu_t - mu_rope[:, :, None, :]   # (B, N, N, d_h)
        mahal_term = torch.einsum('bijk,bijkl,bijl->bij', delta, sigma_t_inv, delta)

        # Log-determinants
        sign_t, logdet_t = torch.linalg.slogdet(sigma_t)
        sign_i, logdet_i = torch.linalg.slogdet(sigma_i_rope)

        kl = 0.5 * (trace_term + mahal_term - d_h + logdet_t - logdet_i)
        kl = kl.clamp(min=0.0, max=max(KL_CEIL_BASE, KL_CEIL_SCALE * d_h))

        # Apply mask before softmax
        if isinstance(kappa, torch.Tensor):
            kappa_val = kappa
        else:
            kappa_val = float(kappa)
        dim_scale = math.sqrt(max(d_h, 1))
        logits = -kl / (kappa_val * dim_scale)
        if mask is not None:
            logits = logits.masked_fill(mask == 0, float('-inf'))
        if mask_self_attention:
            diag_idx = torch.arange(N, device=device)
            has_other = (logits != float('-inf')).sum(dim=-1) > 1
            logits = logits.clone()
            diag_vals = logits[:, diag_idx, diag_idx]
            masked_diag = torch.where(
                has_other, torch.full_like(diag_vals, float('-inf')), diag_vals
            )
            logits[:, diag_idx, diag_idx] = masked_diag

        beta = torch.nn.functional.softmax(logits, dim=-1)

        # Build the alignment loss with separate weights for the direct (β·∂KL/∂μ)
        # and softmax-coupling (∂β/∂μ·KL) terms.  This is the standard trick for
        # decoupling lambda_belief and lambda_softmax in an autograd-based loss.
        F_align_direct = lambda_belief * (beta.detach() * kl).sum()
        F_align_softmax = lambda_softmax * (beta * kl.detach()).sum()
        F_align = F_align_direct + F_align_softmax

        # Self-coupling KL(q_i || p_i)
        delta_p = mu_var - mu_p_h_d
        if _is_diag:
            # Diagonal: efficient element-wise formula
            sigma_p_safe = sigma_p_h_d.clamp(min=eps)
            sigma_q_safe = sigma_var.clamp(min=eps)
            kl_self = 0.5 * (
                sigma_q_safe / sigma_p_safe
                + delta_p ** 2 / sigma_p_safe
                - 1.0
                + torch.log(sigma_p_safe)
                - torch.log(sigma_q_safe)
            )
            if isinstance(alpha, torch.Tensor):
                F_self = (alpha * kl_self).sum()
            else:
                F_self = alpha * kl_self.sum()
            # Product-rule correction for learnable alpha = c0/(b0+KL).
            # The autograd above treats `alpha` as a constant w.r.t. (μ,σ) since
            # alpha was computed from pre-iteration (μ,σ) tensors that are NOT
            # the autograd-traced mu_var/sigma_var here. Without correction,
            # autograd produces α·dKL/dθ but misses the −(α²/c0)·KL·(dKL/dθ)
            # term from dα/dθ. Adding a phantom term −0.5·(α²/c0)·KL² (with α
            # and c0 detached) makes autograd produce exactly the missing term:
            # d/dθ[−0.5·(α²/c0)·KL²] = −(α²/c0)·KL·(dKL/dθ).
            if alpha_c0 is not None and isinstance(alpha, torch.Tensor):
                _alpha_det = alpha.detach()
                _c0_det = alpha_c0.detach().clamp(min=eps)
                F_self = F_self - 0.5 * (_alpha_det.pow(2) / _c0_det * kl_self.pow(2)).sum()
        else:
            # Full covariance: KL(q||p) = 0.5[tr(Sp^{-1} Sq) + delta^T Sp^{-1} delta - d + log|Sp| - log|Sq|]
            eye_dh_self = _cached_eye(d_h, device, _f32)
            # Gauge-covariant ridge for the self-KL: Σ_p and Σ_q both live at
            # position i with local frame exp_phi_h (per-head, per-position).
            if gauge_covariant_ridge:
                _ep_h_self = exp_phi_h.to(dtype=_f32)
                _R_dh_self = _ep_h_self @ _ep_h_self.transpose(-1, -2)
            else:
                _R_dh_self = eye_dh_self
            sigma_p_reg = sigma_p_h_d + eps * _R_dh_self
            sigma_q_reg = sigma_var + eps * _R_dh_self
            sigma_p_inv = _safe_spd_inv(
                sigma_p_reg, eps=eps,
                exp_phi=(_ep_h_self if gauge_covariant_ridge else None),
            )
            trace_self = torch.einsum('bnij,bnji->bn', sigma_p_inv, sigma_q_reg)
            mahal_self = torch.einsum('bni,bnij,bnj->bn', delta_p, sigma_p_inv, delta_p)
            _, logdet_p = torch.linalg.slogdet(sigma_p_reg)
            _, logdet_q = torch.linalg.slogdet(sigma_q_reg)
            kl_self_per_token = 0.5 * (trace_self + mahal_self - d_h + logdet_p - logdet_q)
            kl_self_per_token = kl_self_per_token.clamp(min=0.0)
            if isinstance(alpha, torch.Tensor):
                # alpha is per-dim (B, N, d_h) — full-cov KL already sums over dims,
                # so use mean alpha as scalar weight per token.
                F_self = (alpha.mean(dim=-1) * kl_self_per_token).sum()
            else:
                F_self = alpha * kl_self_per_token.sum()
            # Product-rule correction for learnable alpha (full-cov path).
            # The full-cov KL is dimension-summed, so we use scalar reductions of
            # alpha and c0; this mirrors the alpha.mean(dim=-1) used above and
            # restores the descent direction the diagonal path enjoys.
            if alpha_c0 is not None and isinstance(alpha, torch.Tensor):
                _alpha_scalar = alpha.mean(dim=-1).detach()                 # (B, N)
                _c0_scalar = alpha_c0.detach().mean().clamp(min=eps)        # ()
                F_self = F_self - 0.5 * (_alpha_scalar.pow(2) / _c0_scalar * kl_self_per_token.pow(2)).sum()

        F_total = F_self + F_align

        # Autograd through F_total to get gradients w.r.t. raw mu and raw sigma.
        # This handles the chain rule through R Σ R^T automatically.
        grad_mu_h, grad_sigma_h = torch.autograd.grad(
            F_total, [mu_var, sigma_var], create_graph=False, retain_graph=False
        )

    return beta.detach(), grad_mu_h.detach(), grad_sigma_h.detach()

