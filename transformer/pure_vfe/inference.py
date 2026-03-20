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
    natural_grad_omega,
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

    for step in range(config.n_esteps):
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

        # 3. Total mean gradient
        grad_mu = grad_mu_align + grad_mu_prior

        # 4. Natural gradient: Δμ = -η Σ ∂F/∂μ (Fisher-Rao for Gaussian mean)
        nat_mu = torch.einsum('bnij,bnj->bni', Sigma, grad_mu)

        # 5. Trust-region clip and update
        nat_mu = clip_norm(nat_mu, config.trust_region_mu)
        mu = mu - config.eta_E * nat_mu

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

        # 3. Full gradient per head: ½[α_i Σ_prior⁻¹ + Σ_j β_ij (transported prec) - 2Σ_i⁻¹]
        # Note: the -Σ_i⁻¹ term comes from ∂(-ln det Σ_i)/∂Σ_i in the KL
        alpha_expanded = alpha.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        grad_Sigma_h = 0.5 * (
            alpha_expanded * prior_prec_h + weighted_prec
            - 2.0 * Sigma_h_inv.permute(0, 2, 1, 3, 4)
        )  # Note: Sigma_h_inv from precomp is [B,H,N,K,K], need [B,N,H,K,K]

        # Actually fix: Sigma_h_inv from extract is [B,N,H,K,K]
        # precomp['Sigma_inv'] is [B,H,N,K,K] — let's use the direct one
        Sigma_h_inv_bn = Sigma_h_inv  # [B, N, H, K_h, K_h]
        grad_Sigma_h = 0.5 * (
            alpha_expanded * prior_prec_h + weighted_prec - 2.0 * Sigma_h_inv_bn
        )

        # 4. Natural gradient on SPD: ΔΣ = -2 Σ sym(∂F/∂Σ) Σ
        nat_Sigma_h = natural_grad_sigma(grad_Sigma_h, Sigma_h)

        # 5. Retract on SPD manifold
        nat_Sigma_h = clip_matrix_norm(nat_Sigma_h, config.trust_region_sigma)
        Sigma_h_new = retract_spd(
            Sigma_h, nat_Sigma_h, config.eta_E,
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

        # Natural gradient on GL(K): left-invariant
        nat_Omega = natural_grad_omega(grad_Omega, Omega)

        # Trust-region clip and update
        nat_Omega = clip_matrix_norm(nat_Omega, config.trust_region_omega)
        Omega = Omega - config.eta_E * nat_Omega

    # --- Decode: logit_v = -KL(q_i || π_v) ---
    logits = kl_decode_logits(mu, Sigma, model.prior_mu, model.prior_Sigma)

    return mu, Sigma, Omega, logits, vfe_history
