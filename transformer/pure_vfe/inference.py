"""
E-Step: Variational free energy descent (the forward pass).

This IS the forward pass. No neural network. No autograd.
Beliefs (μ, Σ) and gauge frames (Ω) are updated via natural gradient descent
on the belief VFE until convergence. Returns logits via KL-based decoding.

E-step minimizes prior + alignment terms only.
Observation gradient enters exclusively in M-step (learning.py).

Mathematical reference:
  - Main paper Eq. 7 (full VFE), Eq. 21 (mean gradient), Eq. 24 (softmax correction)
  - Supplementary Eq. B.1 (covariance gradient), Appendix D (numerical methods)
"""

import warnings

import torch

from .gaussians import (
    precompute_tokens,
    pairwise_kl,
    kl_attention,
    kl_divergence,
    kl_decode_logits,
    build_alibi_log_prior,
    vfe_grad_mu_alignment,
    vfe_grad_Sigma_alignment,
    vfe_grad_mu_prior,
    state_dependent_alpha,
    natural_grad_sigma,
    retract_spd,
    clip_norm,
    clip_matrix_norm,
    safe_inverse,
    symmetrize,
)
from .gauge import (
    vfe_grad_Omega,
    lie_algebra_clip_grad,
    regularize_omega_conditioning,
)


# ---------------------------------------------------------------------------
# RoPE: Rotary Position Embeddings for KL-based attention
# ---------------------------------------------------------------------------
# Reuse the implementation from transformer.core.attention but keep a local
# copy to avoid circular imports and maintain pure_vfe independence.

def _build_rope_freqs(K: int, base: float = 10000.0,
                      device=None, dtype=None) -> torch.Tensor:
    r"""Compute RoPE frequency bands for K-dimensional beliefs.

    Returns:
        freqs: (K//2,) inverse frequency bands
    """
    half_K = K // 2
    freqs = 1.0 / (base ** (torch.arange(0, half_K, device=device, dtype=dtype) / half_K))
    return freqs


def _apply_rope(mu: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    r"""Apply Rotary Position Embeddings to belief means.

    Rotates consecutive pairs of dimensions by position-dependent angles,
    making KL divergences sensitive to relative position via SO(2)^{K/2}.

    Args:
        mu: (B, N, K) belief means
        base: RoPE frequency base

    Returns:
        mu_rotated: (B, N, K) position-rotated belief means
    """
    B, N, K = mu.shape
    half_K = K // 2

    freqs = _build_rope_freqs(K, base, device=mu.device, dtype=mu.dtype)
    positions = torch.arange(N, device=mu.device, dtype=mu.dtype)
    angles = torch.outer(positions, freqs)  # (N, K//2)

    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)

    mu_even = mu[:, :, :2*half_K:2]
    mu_odd = mu[:, :, 1:2*half_K:2]

    mu_rotated = mu.clone()
    mu_rotated[:, :, :2*half_K:2] = mu_even * cos_a - mu_odd * sin_a
    mu_rotated[:, :, 1:2*half_K:2] = mu_even * sin_a + mu_odd * cos_a

    return mu_rotated


def _apply_layernorm(mu: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    r"""Apply LayerNorm to belief means (optional, for testing).

    This is a neural-like component included as an optional toggle.

    Args:
        mu: (B, N, K) belief means

    Returns:
        mu_normed: (B, N, K) normalized belief means
    """
    mean = mu.mean(dim=-1, keepdim=True)
    var = mu.var(dim=-1, keepdim=True, unbiased=False)
    return (mu - mean) / (var + eps).sqrt()


def extract_block_diag(Sigma, H, K_h):
    """
    Extract H diagonal blocks of size K_h from full [B, N, K, K] covariance.

    Returns: [B, N, H, K_h, K_h]
    """
    B, N, K, _ = Sigma.shape
    blocks = []
    for h in range(H):
        start = h * K_h
        end = start + K_h
        blocks.append(Sigma[:, :, start:end, start:end])
    return torch.stack(blocks, dim=2)


def set_block_diag(Sigma, blocks, H, K_h):
    """
    Write H diagonal blocks back into full [B, N, K, K] covariance.

    Args:
        Sigma: [B, N, K, K] (modified in-place)
        blocks: [B, N, H, K_h, K_h]
    """
    for h in range(H):
        start = h * K_h
        end = start + K_h
        Sigma[:, :, start:end, start:end] = blocks[:, :, h]
    return Sigma


def compute_vfe(mu, Sigma, prior_mu, prior_Sigma, alpha, beta, kl_ij):
    """
    Compute the scalar VFE for monitoring.

    F = Σ_i α_i KL(q_i||p_i) + Σ_{ij} β_ij KL(q_i||Ω_ij q_j)

    Returns: scalar VFE value
    """
    # Prior term
    kl_prior = kl_divergence(mu, Sigma, prior_mu, prior_Sigma)  # [B, N]
    prior_term = (alpha * kl_prior).sum()

    # Alignment term (β_ij already softmax-normalized)
    align_term = (beta * kl_ij).sum()

    return (prior_term + align_term).item()


@torch.no_grad()
def e_step(token_ids, model, config, effective_lrs=None):
    r"""Run VFE descent to convergence. This IS the forward pass.

    Minimizes belief VFE (prior + alignment terms only) via natural
    gradient descent on q-variables (mu_q, Sigma_q, phi).
    Observation gradient enters ONLY in the prior update step.

    Features:
      - RoPE: SO(2)^{K/2} rotations on μ for position-sensitive attention
      - LayerNorm: optional normalization of μ between iterations
      - Diagonal covariance: optional diagonal-only Σ for speed
      - Per-variable learning rates (mu_q_lr, sigma_q_lr, phi_lr)
      - Element-wise gradient clamping (grad_clamp)
      - Whitened trust region for mu updates
      - NaN/Inf detection and recovery
      - VFE divergence early stopping

    Args:
        token_ids: [B, N] long tensor of token indices
        model: PureVFETransformer instance
        config: PureVFEConfig
        effective_lrs: optional dict of scheduled learning rates from
            ``model.get_effective_lrs()``. If None, uses base config LRs.

    Returns:
        mu: [B, N, K] converged beliefs (means)
        Sigma: [B, N, K, K] converged beliefs (covariances)
        Omega: [B, N, H, K_h, K_h] converged gauge frames
        logits: [B, N, V] decoding logits
        vfe_history: list of VFE values per step
        diagnostics: dict with gradient norms, attention weights, holonomy
    """
    B, N = token_ids.shape
    K = config.belief_dim
    H = config.n_heads
    K_h = config.head_dim
    dev = config.device
    grad_clamp = getattr(config, 'grad_clamp', 1e3)
    omega_grad_clamp = getattr(config, 'omega_grad_clamp', 10.0)
    nan_recovery = getattr(config, 'nan_recovery', True)

    # Per-variable learning rates (scheduled or base)
    if effective_lrs is None:
        _mu_q_lr = config.mu_q_lr
        _sigma_q_lr = config.sigma_q_lr
        _phi_lr = config.phi_lr
    else:
        _mu_q_lr = effective_lrs['mu_q_lr']
        _sigma_q_lr = effective_lrs['sigma_q_lr']
        _phi_lr = effective_lrs['phi_lr']
    omega_cond_max = getattr(config, 'omega_cond_max', 50.0)
    use_rope = getattr(config, 'use_rope', False)
    rope_base = getattr(config, 'rope_base', 10000.0)
    use_layernorm = getattr(config, 'use_layernorm', True)
    diagonal_cov = getattr(config, 'diagonal_covariance', False)
    use_holonomy = getattr(config, 'use_holonomy', False)

    # --- Initialize beliefs from priors ---
    mu = model.prior_mu[token_ids].clone()          # [B, N, K]
    Sigma = model.prior_Sigma[token_ids].clone()     # [B, N, K, K]

    # Initialize gauge frames from token priors (position via RoPE)
    Omega = model.prior_Omega[token_ids].clone()     # [B, N, H, K_h, K_h]

    # Prior references (fixed during E-step)
    prior_mu = model.prior_mu[token_ids]             # [B, N, K]
    prior_Sigma = model.prior_Sigma[token_ids]       # [B, N, K, K]

    # --- Build attention prior (manuscript §3.3.4) ---
    alibi_slope = getattr(config, 'alibi_slope', 0.0)
    mask_self = getattr(config, 'mask_self_attention', False)
    if alibi_slope > 0:
        log_prior = build_alibi_log_prior(N, alibi_slope, device=dev)
    else:
        log_prior = None

    vfe_history = []
    nan_events = 0
    diagnostics = {
        'grad_norm_mu': [],
        'grad_norm_sigma': [],
        'grad_norm_omega': [],
    }

    # Holonomy tracking: measure gauge curvature if enabled
    holonomy_data = [] if use_holonomy else None

    for step in range(config.n_esteps):

        # --- Optional LayerNorm on μ before KL computation ---
        if use_layernorm:
            mu = _apply_layernorm(mu)

        # --- Extract per-head block-diagonal ---
        mu_h = mu.view(B, N, H, K_h)                             # [B, N, H, K_h]
        Sigma_h = extract_block_diag(Sigma, H, K_h)              # [B, N, H, K_h, K_h]

        # Diagonal covariance: zero out off-diagonal entries
        if diagonal_cov:
            diag_vals = torch.diagonal(Sigma_h, dim1=-2, dim2=-1)  # [B, N, H, K_h]
            Sigma_h = torch.diag_embed(diag_vals)                  # [B, N, H, K_h, K_h]

        Sigma_h_inv = safe_inverse(Sigma_h)

        # --- RoPE: apply position-dependent SO(2)^{K/2} rotations ---
        # Use RoPE-rotated μ for KL/attention, raw μ for gradients.
        if use_rope:
            mu_rope = _apply_rope(mu, base=rope_base)              # [B, N, K]
            mu_h_rope = mu_rope.view(B, N, H, K_h)                # [B, N, H, K_h]
            precomp_rope = precompute_tokens(mu_h_rope, Sigma_h, Omega, Sigma_h_inv)
        else:
            precomp_rope = None

        # Raw precomp (always needed for gradients; also for KL when RoPE is off)
        precomp = precompute_tokens(mu_h, Sigma_h, Omega, Sigma_h_inv)

        # Early NaN detection: catch NaN/Inf from Omega inversion before it
        # propagates through all gradient computations.
        if nan_recovery and (
            torch.isnan(precomp['P']).any() or torch.isinf(precomp['P']).any()
        ):
            nan_events += 1
            warnings.warn(
                f"E-step NaN/Inf in precomputed quantities at step {step}, "
                f"resetting beliefs (event #{nan_events})",
                RuntimeWarning, stacklevel=2,
            )
            mu = prior_mu.clone()
            Sigma = model.prior_Sigma[token_ids].clone()
            Omega = model.prior_Omega[token_ids].clone()
            break

        # Select which precomp to use for KL/attention
        precomp_attn = precomp_rope if use_rope else precomp

        # --- Pairwise KL and attention weights ---
        kl_ij = pairwise_kl(precomp_attn, causal=config.causal)   # [B, H, N, N]
        beta = kl_attention(kl_ij, config.tau, causal=config.causal,
                            log_prior=log_prior, mask_self=mask_self)  # [B, H, N, N]

        # --- State-dependent prior precision ---
        _alpha_floor = getattr(config, 'alpha_floor', 0.01)
        # Warm start: stronger floor in early iterations to prevent initial drift.
        # Decays linearly from 3× floor to 1× floor over E-step iterations.
        _alpha_floor_scaled = _alpha_floor * (1.0 + 2.0 * (1.0 - step / max(config.n_esteps - 1, 1)))
        alpha = state_dependent_alpha(
            mu, Sigma, prior_mu, prior_Sigma, config.alpha_b0, config.alpha_c0,
            alpha_floor=_alpha_floor_scaled,
        )  # [B, N]

        # --- Monitor VFE ---
        # Use raw precomp KL for VFE monitoring (geometric truth, not RoPE-rotated)
        if use_rope:
            kl_ij_raw = pairwise_kl(precomp, causal=config.causal)
            vfe = compute_vfe(mu, Sigma, prior_mu, prior_Sigma, alpha, beta, kl_ij_raw)
        else:
            kl_ij_raw = kl_ij
            vfe = compute_vfe(mu, Sigma, prior_mu, prior_Sigma, alpha, beta, kl_ij)
        vfe_history.append(vfe)

        # --- VFE divergence early stopping ---
        if len(vfe_history) >= 2:
            _prev_vfe = abs(vfe_history[-2]) + 1e-10
            _vfe_ratio = vfe_history[-1] / _prev_vfe
            if _vfe_ratio > 1.1:  # 10% increase (was 1.5x)
                warnings.warn(
                    f"E-step VFE divergence at step {step}: "
                    f"{vfe_history[-2]:.2f} -> {vfe_history[-1]:.2f} "
                    f"(ratio {_vfe_ratio:.3f}), stopping early",
                    RuntimeWarning, stacklevel=2,
                )
                break

        # --- Holonomy: measure gauge curvature ---
        if use_holonomy and step == config.n_esteps - 1:
            holonomy_data = _compute_holonomy(Omega, H, K_h)

        # ================================================================
        # MEAN GRADIENT: ∂F/∂μ_i (uses raw precomp for gradients)
        # ================================================================
        # 1. Alignment gradient (with softmax correction, Eq. 21 + 24)
        # When RoPE is active: beta comes from RoPE-KL (position-aware attention),
        # but kl_ij in the softmax correction w_ij = β_ij[1 + (E[KL] - KL_ij)/τ]
        # must use raw-mu KL for geometric consistency of the gradient direction.
        # See VFE_dynamic variational_ffn.py:1183-1187 for the same pattern.
        grad_mu_align = vfe_grad_mu_alignment(precomp, beta, kl_ij_raw, config.tau)
        # Shape: [B, H, N, K_h] — transpose to [B, N, H, K_h] and reshape to [B, N, K]
        grad_mu_align = grad_mu_align.permute(0, 2, 1, 3).reshape(B, N, K)

        # 2. Prior gradient
        grad_mu_prior = vfe_grad_mu_prior(mu, Sigma, prior_mu, prior_Sigma, alpha)
        # Shape: [B, N, K]

        # 3. Total mean gradient + element-wise clamp
        grad_mu = grad_mu_align + grad_mu_prior
        grad_mu = torch.clamp(grad_mu, -grad_clamp, grad_clamp)
        diagnostics['grad_norm_mu'].append(grad_mu.norm().item())

        # 4. Natural gradient: Δμ = -η Σ ∂F/∂μ (Fisher-Rao for Gaussian mean)
        nat_mu = torch.einsum('bnij,bnj->bni', Sigma, grad_mu)

        # 5. Whitened trust-region clip (matching VFE dynamic)
        #    Clip ||δμ / √σ|| instead of raw ||δμ||
        sigma_diag = torch.diagonal(Sigma, dim1=-2, dim2=-1).clamp(min=1e-6)
        whitened = nat_mu / sigma_diag.sqrt()
        whitened_norm = whitened.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = torch.clamp(config.trust_region_mu / whitened_norm, max=1.0)
        nat_mu = nat_mu * scale

        # 6. Update
        mu = mu - _mu_q_lr * nat_mu

        # ================================================================
        # COVARIANCE GRADIENT: ∂F/∂Σ_i
        # ================================================================
        # 1. Alignment: weighted transported precision (with softmax correction,
        # matching the mu gradient which also uses kl_ij_raw and tau)
        weighted_prec = vfe_grad_Sigma_alignment(precomp, beta, kl_ij_raw, config.tau)
        # Shape: [B, H, N, K_h, K_h] — permute to [B, N, H, K_h, K_h]
        weighted_prec = weighted_prec.permute(0, 2, 1, 3, 4)

        # 2. Prior precision
        prior_Sigma_h = extract_block_diag(prior_Sigma, H, K_h)
        prior_prec_h = safe_inverse(prior_Sigma_h)  # [B, N, H, K_h, K_h]

        # 3. Full gradient per head: ½[α_i Σ_prior⁻¹ + Σ_j β_ij (transported prec) - (α+1)Σ_i⁻¹]
        # Entropy coefficient: α from prior KL + 1 from alignment KL
        # Sigma_h_inv is [B, N, H, K_h, K_h] from extract_block_diag
        alpha_expanded = alpha.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        grad_Sigma_h = 0.5 * (
            alpha_expanded * prior_prec_h + weighted_prec
            - (alpha_expanded + 1.0) * Sigma_h_inv
        )
        grad_Sigma_h = torch.clamp(grad_Sigma_h, -grad_clamp, grad_clamp)
        diagnostics['grad_norm_sigma'].append(grad_Sigma_h.norm().item())

        # 4. Natural gradient on SPD: ΔΣ = -2 Σ sym(∂F/∂Σ) Σ
        nat_Sigma_h = natural_grad_sigma(grad_Sigma_h, Sigma_h)

        # 5. Retract on SPD manifold
        nat_Sigma_h = clip_matrix_norm(nat_Sigma_h, config.trust_region_sigma)
        Sigma_h_new = retract_spd(
            Sigma_h, nat_Sigma_h, _sigma_q_lr,
            eps_min=config.spd_eps_min,
            kappa_max=config.spd_kappa_max,
            exp_clip=config.spd_exp_clip,
        )

        # Enforce diagonal if configured
        if diagonal_cov:
            diag_new = torch.diagonal(Sigma_h_new, dim1=-2, dim2=-1)
            Sigma_h_new = torch.diag_embed(diag_new)

        # 6. Write back to full Sigma
        Sigma = set_block_diag(Sigma, Sigma_h_new, H, K_h)

        # ================================================================
        # GAUGE GRADIENT: ∂F/∂Ω_i
        # ================================================================
        # Compute gauge gradient (attention-weighted)
        grad_Omega = vfe_grad_Omega(mu_h, Sigma_h, Omega, beta, kl_ij, precomp)
        # Shape: [B, N, H, K_h, K_h]
        # Element-wise safety clamp on Euclidean gradient (numerical guard
        # against extreme values from cascading inversions in vfe_grad_Omega)
        grad_Omega = torch.clamp(grad_Omega, -omega_grad_clamp, omega_grad_clamp)
        diagnostics['grad_norm_omega'].append(grad_Omega.norm().item())

        # Natural gradient via Lie algebra with Riemannian trust region:
        # ξ = Ωᵀ·g, clip ||ξ||_F ≤ trust_radius, ΔΩ = Ω·ξ
        nat_Omega = lie_algebra_clip_grad(
            grad_Omega, Omega, trust_radius=config.trust_region_omega,
        )
        Omega = Omega - _phi_lr * nat_Omega

        # Post-update Omega conditioning
        Omega = regularize_omega_conditioning(Omega, omega_cond_max)

        # ================================================================
        # NaN/Inf GUARD
        # ================================================================
        if nan_recovery and (torch.isnan(mu).any() or torch.isinf(mu).any()
                             or torch.isnan(Sigma).any()):
            nan_events += 1
            warnings.warn(
                f"E-step NaN/Inf at step {step}, resetting beliefs to priors "
                f"(event #{nan_events})",
                RuntimeWarning, stacklevel=2,
            )
            mu = prior_mu.clone()
            Sigma = model.prior_Sigma[token_ids].clone()
            Omega = model.prior_Omega[token_ids].clone()
            break

    # --- Decode: logit_v = -KL(q_i || π_v) / τ_decode ---
    _decode_tau = getattr(config, 'decode_tau', 1.0)
    logits = kl_decode_logits(mu, Sigma, model.prior_mu, model.prior_Sigma,
                              decode_tau=_decode_tau)

    diagnostics['nan_events'] = nan_events
    diagnostics['final_beta'] = beta  # For attention visualization
    if holonomy_data is not None:
        diagnostics['holonomy'] = holonomy_data

    return mu, Sigma, Omega, logits, vfe_history, diagnostics


def _compute_holonomy(Omega, H, K_h):
    r"""Compute holonomy metrics from gauge frames.

    Measures gauge curvature via Wilson loops on small triangles:
    C_{ijk} = Ω_{ij} · Ω_{jk} · Ω_{ki} should be identity for flat connections.

    Args:
        Omega: [B, N, H, K_h, K_h] gauge frames

    Returns:
        dict with holonomy metrics
    """
    B, N = Omega.shape[:2]
    if N < 3:
        return {'mean_norm': 0.0, 'max_norm': 0.0}

    # Sample a few triangles (i, i+1, i+2)
    Om = Omega.permute(0, 2, 1, 3, 4)  # [B, H, N, K_h, K_h]
    Om_inv = torch.linalg.inv(Om)

    # Wilson loop: C = Ω_i Ω_j^{-1} · Ω_j Ω_k^{-1} · Ω_k Ω_i^{-1}
    # = Ω_{ij} · Ω_{jk} · Ω_{ki}
    # For flat connection: C = I
    n_triangles = min(N - 2, 16)  # Sample up to 16 triangles
    norms = []
    eye = torch.eye(K_h, device=Omega.device, dtype=Omega.dtype)
    for t in range(n_triangles):
        i, j, k = t, t + 1, t + 2
        Omega_ij = Om[:, :, i] @ Om_inv[:, :, j]  # [B, H, K_h, K_h]
        Omega_jk = Om[:, :, j] @ Om_inv[:, :, k]
        Omega_ki = Om[:, :, k] @ Om_inv[:, :, i]
        C = Omega_ij @ Omega_jk @ Omega_ki  # Should be I
        deviation = (C - eye).norm(dim=(-2, -1))  # [B, H]
        norms.append(deviation.mean().item())

    return {
        'mean_norm': sum(norms) / len(norms) if norms else 0.0,
        'max_norm': max(norms) if norms else 0.0,
        'n_triangles': len(norms),
    }
