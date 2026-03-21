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
def e_step(token_ids, model, config):
    """
    Run VFE descent to convergence. This IS the forward pass.

    E-step minimizes belief VFE (prior + alignment terms only).
    Observation gradient enters ONLY in M-step.

    Numerical stability features (ported from VFE dynamic):
      - E-step LR decay (linear, 1.0 → 0.5)
      - Separate sigma LR (sigma_lr_ratio × mu LR)
      - Element-wise gradient clamping (grad_clamp)
      - Whitened trust region for mu updates
      - NaN/Inf detection and recovery
      - VFE divergence early stopping

    Args:
        token_ids: [B, N] long tensor of token indices
        model: PureVFETransformer instance
        config: PureVFEConfig

    Returns:
        mu: [B, N, K] converged beliefs (means)
        Sigma: [B, N, K, K] converged beliefs (covariances)
        Omega: [B, N, H, K_h, K_h] converged gauge frames
        logits: [B, N, V] decoding logits
        vfe_history: list of VFE values per step
    """
    B, N = token_ids.shape
    K = config.belief_dim
    H = config.n_heads
    K_h = config.head_dim
    dev = config.device
    grad_clamp = getattr(config, 'grad_clamp', 1e3)
    omega_grad_clamp = getattr(config, 'omega_grad_clamp', 10.0)
    sigma_lr_ratio = getattr(config, 'sigma_lr_ratio', 0.05)
    e_step_lr_decay = getattr(config, 'e_step_lr_decay', 0.5)
    nan_recovery = getattr(config, 'nan_recovery', True)
    omega_cond_max = getattr(config, 'omega_cond_max', 50.0)

    # --- Initialize beliefs from priors ---
    mu = model.prior_mu[token_ids].clone()          # [B, N, K]
    Sigma = model.prior_Sigma[token_ids].clone()     # [B, N, K, K]

    # Initialize gauge frames: token prior + positional composition
    Omega = model.prior_Omega[token_ids].clone()     # [B, N, H, K_h, K_h]
    pos_Om = model.pos_Omega[:N].unsqueeze(0).expand(B, -1, -1, -1, -1)
    Omega = Omega @ pos_Om                           # Compose: Ω_token · Ω_pos

    # Prior references (fixed during E-step)
    prior_mu = model.prior_mu[token_ids]             # [B, N, K]
    prior_Sigma = model.prior_Sigma[token_ids]       # [B, N, K, K]

    vfe_history = []
    nan_events = 0
    diagnostics = {
        'grad_norm_mu': [],
        'grad_norm_sigma': [],
        'grad_norm_omega': [],
    }

    for step in range(config.n_esteps):
        # --- E-step LR decay (matching VFE dynamic) ---
        if config.n_esteps > 1:
            decay = 1.0 - e_step_lr_decay * (step / (config.n_esteps - 1))
        else:
            decay = 1.0
        effective_lr = config.eta_E * decay
        sigma_lr = effective_lr * sigma_lr_ratio

        # --- Extract per-head block-diagonal ---
        mu_h = mu.view(B, N, H, K_h)                             # [B, N, H, K_h]
        Sigma_h = extract_block_diag(Sigma, H, K_h)              # [B, N, H, K_h, K_h]
        Sigma_h_inv = safe_inverse(Sigma_h)

        # --- Precompute per-token quantities ---
        precomp = precompute_tokens(mu_h, Sigma_h, Omega, Sigma_h_inv)

        # --- Pairwise KL and attention weights ---
        kl_ij = pairwise_kl(precomp, causal=config.causal)       # [B, H, N, N]
        beta = kl_attention(kl_ij, config.tau, causal=config.causal)  # [B, H, N, N]

        # --- State-dependent prior precision ---
        alpha = state_dependent_alpha(
            mu, Sigma, prior_mu, prior_Sigma, config.alpha_b0, config.alpha_c0
        )  # [B, N]

        # --- Monitor VFE ---
        vfe = compute_vfe(mu, Sigma, prior_mu, prior_Sigma, alpha, beta, kl_ij)
        vfe_history.append(vfe)

        # --- VFE divergence early stopping ---
        if len(vfe_history) >= 2 and vfe_history[-1] > vfe_history[-2] * 1.5:
            warnings.warn(
                f"E-step VFE divergence at step {step}: "
                f"{vfe_history[-2]:.2f} -> {vfe_history[-1]:.2f}, stopping early",
                RuntimeWarning, stacklevel=2,
            )
            break

        # ================================================================
        # MEAN GRADIENT: ∂F/∂μ_i
        # ================================================================
        # 1. Alignment gradient (with softmax correction, Eq. 21 + 24)
        grad_mu_align = vfe_grad_mu_alignment(precomp, beta, kl_ij, config.tau)
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
        mu = mu - effective_lr * nat_mu

        # ================================================================
        # COVARIANCE GRADIENT: ∂F/∂Σ_i
        # ================================================================
        # 1. Alignment: weighted transported precision
        weighted_prec = vfe_grad_Sigma_alignment(precomp, beta)
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

        # 5. Retract on SPD manifold (sigma uses reduced LR)
        nat_Sigma_h = clip_matrix_norm(nat_Sigma_h, config.trust_region_sigma)
        Sigma_h_new = retract_spd(
            Sigma_h, nat_Sigma_h, sigma_lr,
            eps_min=config.spd_eps_min,
            kappa_max=config.spd_kappa_max,
            exp_clip=config.spd_exp_clip,
        )

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
        Omega = Omega - effective_lr * nat_Omega

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
            Omega = model.prior_Omega[token_ids].clone() @ pos_Om
            break

    # --- Decode: logit_v = -KL(q_i || π_v) ---
    logits = kl_decode_logits(mu, Sigma, model.prior_mu, model.prior_Sigma)

    diagnostics['nan_events'] = nan_events

    return mu, Sigma, Omega, logits, vfe_history, diagnostics
