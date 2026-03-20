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

    # ================================================================
    # 1. Observation gradient (analytic softmax-CE)
    # ================================================================
    logits = kl_decode_logits(mu_star, Sigma_star, model.prior_mu, model.prior_Sigma)
    ce_grad = softmax_ce_gradient(logits, targets)  # [B, N, V]

    # CE loss for monitoring
    log_probs = torch.log_softmax(logits, dim=-1)
    ce_loss = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1).mean()

    # ================================================================
    # 2. Update prior bank for tokens seen in batch
    # ================================================================
    unique_tokens = token_ids.unique()

    for v in unique_tokens:
        mask = (token_ids == v)     # [B, N] boolean
        n_v = mask.sum().item()
        if n_v == 0:
            continue

        # Current priors
        mu_v = model.prior_mu[v]              # [K]
        Sigma_v = model.prior_Sigma[v]        # [K, K]
        Sigma_v_inv = safe_inverse(Sigma_v.unsqueeze(0)).squeeze(0)

        # Gather converged beliefs for this token
        mu_star_v = mu_star[mask]             # [n_v, K]
        Sigma_star_v = Sigma_star[mask]       # [n_v, K, K]

        # ---- Prior mean gradient ----
        # ∂KL(q*||p_v)/∂μ_v = -Σ_v⁻¹(μ* - μ_v)
        mu_diff = mu_star_v - mu_v            # [n_v, K]
        grad_mu_v = -(Sigma_v_inv @ mu_diff.mean(0))

        # Hyper-prior pull: regularize toward origin
        grad_mu_v = grad_mu_v + mu_v / config.hyper_var

        # Observation gradient: ∂CE/∂μ_v through KL decoding
        # ∂(-KL(q||π_v))/∂μ_v = Σ_v⁻¹(μ_q - μ_v)
        # ∂CE/∂μ_v = Σ_i ce_grad[i,v] · Σ_v⁻¹(μ*_i - μ_v)
        ce_weights = ce_grad[:, :, v][mask]    # [n_v]
        weighted_diff = (ce_weights.unsqueeze(-1) * mu_diff).mean(0)
        obs_grad_mu = Sigma_v_inv @ weighted_diff
        grad_mu_v = grad_mu_v + obs_grad_mu

        # Natural gradient: Δμ_v = -η Σ_v ∂F/∂μ_v
        nat_mu_v = Sigma_v @ grad_mu_v
        nat_mu_v = clip_norm(nat_mu_v.unsqueeze(0), 1.0).squeeze(0)
        model.prior_mu[v] = mu_v - config.eta_M * nat_mu_v

        # ---- Prior covariance gradient ----
        # ∂KL(q*||p_v)/∂Σ_v = ½[Σ_v⁻¹ - Σ_v⁻¹ E[Σ*] Σ_v⁻¹ - Σ_v⁻¹ E[ΔμΔμᵀ] Σ_v⁻¹]
        Sigma_star_avg = Sigma_star_v.mean(0)     # [K, K]
        outer_avg = (mu_diff.unsqueeze(-1) * mu_diff.unsqueeze(-2)).mean(0)  # [K, K]

        grad_Sigma_v = 0.5 * (
            Sigma_v_inv
            - Sigma_v_inv @ (Sigma_star_avg + outer_avg) @ Sigma_v_inv
        )

        # Hyper-prior regularization
        grad_Sigma_v = grad_Sigma_v + 0.5 * Sigma_v_inv / config.hyper_var

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
        Omega_star_v = Omega_star[mask]       # [n_v, H, K_h, K_h]
        Omega_v = model.prior_Omega[v]        # [H, K_h, K_h]

        # Simple gradient: pull prior toward converged frames
        # ∂F/∂Ω_v ≈ -(Ω_v⁻¹)(Ω*_avg - Ω_v)  (direction toward mean)
        Omega_star_avg = Omega_star_v.mean(0)  # [H, K_h, K_h]
        grad_Omega_v = -(Omega_star_avg - Omega_v)

        nat_Omega_v = natural_grad_omega(grad_Omega_v, Omega_v)
        nat_Omega_v = clip_matrix_norm(nat_Omega_v, 0.3)
        model.prior_Omega[v] = Omega_v - config.eta_M * nat_Omega_v

    # ================================================================
    # 3. Update positional gauge offsets
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
