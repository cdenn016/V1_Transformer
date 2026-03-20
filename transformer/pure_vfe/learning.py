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
def m_step(token_ids, targets, mu_star, Sigma_star, Omega_star, model, config):
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
    # 3. Update prior bank
    # ================================================================
    for idx, v in enumerate(update_tokens):
        v = v.item()

        # Current priors
        mu_v = model.prior_mu[v]              # [K]
        Sigma_v = model.prior_Sigma[v]        # [K, K]
        Sigma_v_inv = safe_inverse(Sigma_v.unsqueeze(0)).squeeze(0)

        # Check if token appears as input (has VFE prior gradient)
        input_mask = (token_ids == v)
        n_v = input_mask.sum().item()

        # ---- Prior mean gradient ----
        if n_v > 0:
            mu_star_v = mu_star[input_mask]   # [n_v, K]
            mu_diff_v = mu_star_v - mu_v      # [n_v, K]

            # ∂KL(q*||p_v)/∂μ_v = -Σ_v⁻¹(μ* - μ_v)
            grad_mu_v = -(Sigma_v_inv @ mu_diff_v.mean(0))
        else:
            grad_mu_v = torch.zeros(K, device=mu_v.device, dtype=mu_v.dtype)

        # Hyper-prior pull: regularize toward origin
        grad_mu_v = grad_mu_v + mu_v / config.hyper_var

        # Exact observation gradient for μ_v over ALL positions:
        # ∂CE/∂μ_v = (1/BN) Σ_{b,n} ce_grad[b,n,v] · Σ_v⁻¹(μ*_{b,n} - μ_v)
        #          = (1/BN) · Σ_v⁻¹ · (obs_weighted_mu[v] - obs_ce_sum[v]·μ_v)
        obs_diff = obs_weighted_mu[idx] - obs_ce_sum[idx] * mu_v
        obs_grad_mu = Sigma_v_inv @ (obs_diff / BN)
        grad_mu_v = grad_mu_v + obs_grad_mu

        # Natural gradient: Δμ_v = -η Σ_v ∂F/∂μ_v
        nat_mu_v = Sigma_v @ grad_mu_v
        nat_mu_v = clip_norm(nat_mu_v.unsqueeze(0), 1.0).squeeze(0)
        model.prior_mu[v] = mu_v - config.eta_M * nat_mu_v

        # ---- Prior covariance gradient ----
        if n_v > 0:
            mu_star_v = mu_star[input_mask]
            Sigma_star_v = Sigma_star[input_mask]
            mu_diff_v = mu_star_v - mu_v

            # ∂KL(q*||p_v)/∂Σ_v = ½[Σ_v⁻¹ - Σ_v⁻¹(E[Σ*] + E[ΔμΔμᵀ])Σ_v⁻¹]
            Sigma_star_avg = Sigma_star_v.mean(0)     # [K, K]
            outer_avg = (mu_diff_v.unsqueeze(-1) * mu_diff_v.unsqueeze(-2)).mean(0)

            grad_Sigma_v = 0.5 * (
                Sigma_v_inv
                - Sigma_v_inv @ (Sigma_star_avg + outer_avg) @ Sigma_v_inv
            )
        else:
            grad_Sigma_v = torch.zeros(K, K, device=mu_v.device, dtype=mu_v.dtype)

        # Hyper-prior regularization
        grad_Sigma_v = grad_Sigma_v + 0.5 * Sigma_v_inv / config.hyper_var

        # Exact observation gradient for Σ_v over ALL positions:
        # ∂logit_v/∂Σ_v = ½[Σ_v⁻¹(Σ* + δδᵀ)Σ_v⁻¹ - Σ_v⁻¹]
        # ∂CE/∂Σ_v = (1/BN) Σ_{b,n} ce_grad[b,n,v] · ∂logit_v/∂Σ_v
        #
        # W_v = Σ_{b,n} ce_grad[b,n,v]·(Σ* + (μ*-μ_v)(μ*-μ_v)ᵀ)
        #     = obs_weighted_Sigma + obs_weighted_outer
        #       - obs_weighted_mu·μ_vᵀ - μ_v·obs_weighted_muᵀ + obs_ce_sum·μ_vμ_vᵀ
        w_mu = obs_weighted_mu[idx]  # [K]
        s_v = obs_ce_sum[idx]        # scalar
        W_v = (
            obs_weighted_Sigma[idx]
            + obs_weighted_outer[idx]
            - w_mu.unsqueeze(-1) * mu_v.unsqueeze(0)
            - mu_v.unsqueeze(-1) * w_mu.unsqueeze(0)
            + s_v * mu_v.unsqueeze(-1) * mu_v.unsqueeze(0)
        )
        obs_grad_Sigma = 0.5 * (Sigma_v_inv @ W_v @ Sigma_v_inv - s_v * Sigma_v_inv) / BN
        obs_grad_Sigma = symmetrize(obs_grad_Sigma)
        grad_Sigma_v = grad_Sigma_v + obs_grad_Sigma

        # Natural gradient on SPD
        nat_Sigma_v = natural_grad_sigma(
            grad_Sigma_v.unsqueeze(0), Sigma_v.unsqueeze(0)
        ).squeeze(0)
        nat_Sigma_v = clip_matrix_norm(nat_Sigma_v.unsqueeze(0), 0.3).squeeze(0)

        model.prior_Sigma[v] = retract_spd(
            Sigma_v.unsqueeze(0), nat_Sigma_v.unsqueeze(0), config.eta_M,
            eps_min=config.spd_eps_min, kappa_max=config.spd_kappa_max,
        ).squeeze(0)

        # ---- Prior gauge frame gradient ----
        if n_v > 0:
            Omega_star_v = Omega_star[input_mask]   # [n_v, H, K_h, K_h]
            Omega_v = model.prior_Omega[v]          # [H, K_h, K_h]

            Omega_star_avg = Omega_star_v.mean(0)   # [H, K_h, K_h]
            grad_Omega_v = -(Omega_star_avg - Omega_v)

            nat_Omega_v = natural_grad_omega(grad_Omega_v, Omega_v)
            nat_Omega_v = clip_matrix_norm(nat_Omega_v, 0.3)
            model.prior_Omega[v] = Omega_v - config.eta_M * nat_Omega_v

    # ================================================================
    # 4. Update positional gauge offsets
    # ================================================================
    _update_pos_omega(Omega_star, token_ids, model, config)

    return ce_loss.item()


def _update_pos_omega(Omega_star, token_ids, model, config):
    """
    Update positional gauge offsets toward mean converged frames per position.
    """
    B, N = token_ids.shape
    H = config.n_heads
    K_h = config.head_dim

    for n in range(N):
        # Average converged gauge at position n across batch
        Om_avg_n = Omega_star[:, n].mean(0)  # [H, K_h, K_h]

        # Pull positional frame toward average
        pos_Om_n = model.pos_Omega[n]        # [H, K_h, K_h]
        grad = -(Om_avg_n - pos_Om_n)
        nat = natural_grad_omega(grad, pos_Om_n)
        nat = clip_matrix_norm(nat, 0.3)
        model.pos_Omega[n] = pos_Om_n - config.eta_M * 0.1 * nat  # Slower rate for positions
