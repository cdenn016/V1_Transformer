"""
M-Step: Prior bank parameter updates via natural gradient on marginal VFE.

Observation gradient enters HERE (not in E-step).
No autograd — all gradients are analytic.

Mathematical reference:
  - Main paper §4.3 (learning)
  - Supplementary Appendix G (model-channel formalism)
"""

import torch

from .gaussians import (
    kl_decode_logits,
    softmax_ce_gradient,
    safe_inverse,
    safe_logdet,
    natural_grad_sigma,
    retract_spd,
    symmetrize,
    clip_norm,
    clip_matrix_norm,
)
from .gauge import natural_grad_omega


def _precompute_obs_gradient(ce_grad, mu_star, Sigma_star, update_tokens):
    """
    Precompute observation gradient quantities over ALL (b,n) positions
    for a set of vocabulary tokens.

    The CE observation gradient for prior μ_v is:
      ∂CE/∂μ_v = (1/BN) Σ_{b,n} ce_grad[b,n,v] · Σ_v⁻¹(μ*_{b,n} - μ_v)

    The CE observation gradient for prior Σ_v is:
      ∂CE/∂Σ_v = (1/BN) Σ_{b,n} ce_grad[b,n,v] · ∂logit_v/∂Σ_v
    where ∂logit_v/∂Σ_v = ½[Σ_v⁻¹(Σ*+δδᵀ)Σ_v⁻¹ - Σ_v⁻¹]

    We precompute the position-summed quantities needed for both.

    Args:
        ce_grad: [B, N, V] softmax CE gradient
        mu_star: [B, N, K] converged beliefs
        Sigma_star: [B, N, K, K] converged covariances
        update_tokens: [T] long tensor of token indices to precompute for

    Returns:
        dict with:
          obs_weighted_mu: [T, K] — Σ_{b,n} ce_grad[b,n,v] · μ*_{b,n}
          obs_ce_sum: [T] — Σ_{b,n} ce_grad[b,n,v]
          obs_weighted_Sigma: [T, K, K] — Σ_{b,n} ce_grad[b,n,v] · Σ*_{b,n}
          obs_weighted_outer: [T, K, K] — Σ_{b,n} ce_grad[b,n,v] · μ*μ*ᵀ
    """
    B, N, K = mu_star.shape
    BN = B * N

    mu_flat = mu_star.reshape(BN, K)               # [BN, K]
    Sigma_flat = Sigma_star.reshape(BN, K, K)       # [BN, K, K]
    ce_flat = ce_grad.reshape(BN, -1)[:, update_tokens]  # [BN, T]

    # Σ_{b,n} ce_grad[b,n,v] · μ*_{b,n} for each v in update_tokens
    obs_weighted_mu = ce_flat.T @ mu_flat           # [T, K]

    # Σ_{b,n} ce_grad[b,n,v]
    obs_ce_sum = ce_flat.sum(0)                     # [T]

    # Σ_{b,n} ce_grad[b,n,v] · Σ*_{b,n}
    obs_weighted_Sigma = torch.einsum('nt,nkl->tkl', ce_flat, Sigma_flat)  # [T, K, K]

    # Σ_{b,n} ce_grad[b,n,v] · μ*_{b,n} μ*_{b,n}ᵀ
    obs_weighted_outer = torch.einsum('nt,nk,nl->tkl', ce_flat, mu_flat, mu_flat)  # [T, K, K]

    return {
        'obs_weighted_mu': obs_weighted_mu,
        'obs_ce_sum': obs_ce_sum,
        'obs_weighted_Sigma': obs_weighted_Sigma,
        'obs_weighted_outer': obs_weighted_outer,
    }


@torch.no_grad()
def m_step(token_ids, targets, mu_star, Sigma_star, Omega_star, model, config,
           logits=None):
    """
    Update prior bank via natural gradient on marginal VFE.
    Observation gradient enters HERE (not in E-step).

    Args:
        token_ids: [B, N] input token indices
        targets: [B, N] target token indices
        mu_star: [B, N, K] converged E-step beliefs (means)
        Sigma_star: [B, N, K, K] converged E-step beliefs (covariances)
        Omega_star: [B, N, H, K_h, K_h] converged gauge frames
        model: PureVFETransformer
        config: PureVFEConfig
        logits: [B, N, V] optional precomputed logits from e_step

    Returns:
        ce_loss: scalar cross-entropy loss (for monitoring only)
    """
    B, N = token_ids.shape
    K = config.belief_dim
    H = config.n_heads
    K_h = config.head_dim
    BN = B * N

    # ================================================================
    # 1. Observation gradient (analytic softmax-CE)
    # ================================================================
    if logits is None:
        logits = kl_decode_logits(mu_star, Sigma_star, model.prior_mu, model.prior_Sigma)
    ce_grad = softmax_ce_gradient(logits, targets)  # [B, N, V]

    # CE loss for monitoring
    log_probs = torch.log_softmax(logits, dim=-1)
    ce_loss = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1).mean()

    # ================================================================
    # 2. Precompute exact observation gradients over ALL positions
    # ================================================================
    # Update tokens that appear as inputs OR targets (both receive gradient)
    update_tokens = torch.unique(torch.cat([
        token_ids.reshape(-1), targets.reshape(-1)
    ]))

    # Build index map: token_id -> position in update_tokens
    token_to_idx = torch.full(
        (config.vocab_size,), -1, dtype=torch.long, device=update_tokens.device
    )
    token_to_idx[update_tokens] = torch.arange(
        len(update_tokens), device=update_tokens.device
    )

    obs = _precompute_obs_gradient(ce_grad, mu_star, Sigma_star, update_tokens)
    obs_weighted_mu = obs['obs_weighted_mu']
    obs_ce_sum = obs['obs_ce_sum']
    obs_weighted_Sigma = obs['obs_weighted_Sigma']
    obs_weighted_outer = obs['obs_weighted_outer']

    # ================================================================
    # 3. Update prior bank (vectorized over all update tokens)
    # ================================================================
    T = len(update_tokens)
    dev = mu_star.device

    # Gather current priors for all update tokens: [T, K], [T, K, K]
    mu_all = model.prior_mu[update_tokens]          # [T, K]
    Sigma_all = model.prior_Sigma[update_tokens]    # [T, K, K]
    Sigma_all_inv = safe_inverse(Sigma_all)          # [T, K, K]

    # ---- Compute VFE prior gradients via scatter-based aggregation ----
    # Map each (b,n) position to its token index in update_tokens
    flat_ids = token_ids.reshape(-1)                  # [BN]
    flat_idx = token_to_idx[flat_ids]                 # [BN] -> index in [0..T-1]

    # Count occurrences of each update token as input
    n_counts = torch.zeros(T, device=dev, dtype=mu_star.dtype)
    n_counts.scatter_add_(0, flat_idx, torch.ones(B * N, device=dev, dtype=mu_star.dtype))

    # Sum of mu_star per token: [T, K]
    mu_star_flat = mu_star.reshape(BN, K)              # [BN, K]
    mu_star_sum = torch.zeros(T, K, device=dev, dtype=mu_star.dtype)
    mu_star_sum.scatter_add_(0, flat_idx.unsqueeze(-1).expand(-1, K), mu_star_flat)

    # Sum of Sigma_star per token: [T, K, K]
    Sigma_star_flat = Sigma_star.reshape(BN, K, K)
    Sigma_star_sum = torch.zeros(T, K, K, device=dev, dtype=mu_star.dtype)
    Sigma_star_sum.scatter_add_(0, flat_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, K, K),
                                 Sigma_star_flat)

    # Sum of (mu_star - mu_v)(mu_star - mu_v)^T per token
    # mu_diff_flat[bn] = mu_star[bn] - mu_v[token_of_bn]
    mu_v_expanded = mu_all[flat_idx]                   # [BN, K]
    mu_diff_flat = mu_star_flat - mu_v_expanded        # [BN, K]
    outer_flat = mu_diff_flat.unsqueeze(-1) * mu_diff_flat.unsqueeze(-2)  # [BN, K, K]
    outer_sum = torch.zeros(T, K, K, device=dev, dtype=mu_star.dtype)
    outer_sum.scatter_add_(0, flat_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, K, K),
                            outer_flat)

    # Omega_star aggregation per token
    Omega_star_flat = Omega_star.reshape(BN, H, K_h, K_h)
    Omega_star_sum = torch.zeros(T, H, K_h, K_h, device=dev, dtype=mu_star.dtype)
    Omega_star_sum.scatter_add_(
        0, flat_idx.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, H, K_h, K_h),
        Omega_star_flat
    )

    # Mask for tokens that actually appear as input (n_v > 0)
    has_input = n_counts > 0                           # [T]
    n_safe = n_counts.clamp(min=1)                     # avoid div by 0

    # ---- Prior mean gradient (vectorized) ----
    # VFE gradient: -Σ_v⁻¹ (mean(mu_star_v) - mu_v) for tokens with input
    mu_star_avg = mu_star_sum / n_safe.unsqueeze(-1)   # [T, K]
    mu_diff_avg = mu_star_avg - mu_all                 # [T, K]
    grad_mu_vfe = -torch.einsum('tij,tj->ti', Sigma_all_inv, mu_diff_avg)
    grad_mu_vfe[~has_input] = 0.0

    # Hyper-prior
    grad_mu = grad_mu_vfe + mu_all / config.hyper_var

    # Observation gradient
    obs_diff = obs_weighted_mu - obs_ce_sum.unsqueeze(-1) * mu_all  # [T, K]
    obs_grad_mu = torch.einsum('tij,tj->ti', Sigma_all_inv, obs_diff / BN)
    grad_mu = grad_mu + obs_grad_mu

    # Natural gradient and update
    nat_mu = torch.einsum('tij,tj->ti', Sigma_all, grad_mu)  # [T, K]
    nat_mu = clip_norm(nat_mu, 1.0)
    model.prior_mu[update_tokens] = mu_all - config.eta_M * nat_mu

    # ---- Prior covariance gradient (vectorized) ----
    Sigma_star_avg = Sigma_star_sum / n_safe.unsqueeze(-1).unsqueeze(-1)
    outer_avg = outer_sum / n_safe.unsqueeze(-1).unsqueeze(-1)

    grad_Sigma_vfe = 0.5 * (
        Sigma_all_inv
        - Sigma_all_inv @ (Sigma_star_avg + outer_avg) @ Sigma_all_inv
    )
    grad_Sigma_vfe[~has_input] = 0.0

    # Hyper-prior
    grad_Sigma = grad_Sigma_vfe + 0.5 * Sigma_all_inv / config.hyper_var

    # Observation gradient for Σ_v
    # W_v = obs_weighted_Sigma + obs_weighted_outer
    #       - obs_weighted_mu·μ_vᵀ - μ_v·obs_weighted_muᵀ + obs_ce_sum·μ_vμ_vᵀ
    W = (
        obs_weighted_Sigma + obs_weighted_outer
        - obs_weighted_mu.unsqueeze(-1) * mu_all.unsqueeze(-2)
        - mu_all.unsqueeze(-1) * obs_weighted_mu.unsqueeze(-2)
        + obs_ce_sum.unsqueeze(-1).unsqueeze(-1) * (mu_all.unsqueeze(-1) * mu_all.unsqueeze(-2))
    )  # [T, K, K]
    obs_grad_Sigma = 0.5 * (
        Sigma_all_inv @ W @ Sigma_all_inv
        - obs_ce_sum.unsqueeze(-1).unsqueeze(-1) * Sigma_all_inv
    ) / BN
    obs_grad_Sigma = symmetrize(obs_grad_Sigma)
    grad_Sigma = grad_Sigma + obs_grad_Sigma

    # Natural gradient on SPD and retract
    nat_Sigma = natural_grad_sigma(grad_Sigma, Sigma_all)
    nat_Sigma = clip_matrix_norm(nat_Sigma, 0.3)
    model.prior_Sigma[update_tokens] = retract_spd(
        Sigma_all, nat_Sigma, config.eta_M,
        eps_min=config.spd_eps_min, kappa_max=config.spd_kappa_max,
    )

    # ---- Prior gauge frame gradient (vectorized) ----
    Omega_all = model.prior_Omega[update_tokens]       # [T, H, K_h, K_h]
    Omega_star_avg = Omega_star_sum / n_safe.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    grad_Omega_all = -(Omega_star_avg - Omega_all)
    grad_Omega_all[~has_input] = 0.0

    nat_Omega = natural_grad_omega(grad_Omega_all, Omega_all)
    nat_Omega = clip_matrix_norm(nat_Omega, 0.3)
    model.prior_Omega[update_tokens] = Omega_all - config.eta_M * nat_Omega

    # ================================================================
    # 4. Update positional gauge offsets
    # ================================================================
    _update_pos_omega(Omega_star, token_ids, model, config)

    return ce_loss.item()


def _update_pos_omega(Omega_star, token_ids, model, config):
    """
    Update positional gauge offsets toward mean converged frames per position.
    Vectorized over all positions.
    """
    B, N = token_ids.shape

    # Average converged gauge at each position across batch: [N, H, K_h, K_h]
    Om_avg = Omega_star.mean(0)               # [N, H, K_h, K_h]
    pos_Om = model.pos_Omega[:N]              # [N, H, K_h, K_h]

    grad = -(Om_avg - pos_Om)
    nat = natural_grad_omega(grad, pos_Om)
    nat = clip_matrix_norm(nat, 0.3)
    model.pos_Omega[:N] = pos_Om - config.eta_M * 0.1 * nat
