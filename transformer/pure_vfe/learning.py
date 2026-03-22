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
from .gauge import (
    natural_grad_omega,
    lie_algebra_clip_grad,
    relative_trust_clip,
    regularize_omega_conditioning,
    retract_phi,
)


def _precompute_obs_gradient(ce_grad, mu_star, Sigma_star, update_tokens):
    """
    Precompute observation gradient quantities over ALL (b,n) positions
    for a set of vocabulary tokens.

    The CE observation gradient for prior μ_v is:
      ∂CE/∂μ_v = (1/n_v) Σ_{b,n} ce_grad[b,n,v] · Σ_v⁻¹(μ*_{b,n} - μ_v)

    The CE observation gradient for prior Σ_v is:
      ∂CE/∂Σ_v = (1/n_v) Σ_{b,n} ce_grad[b,n,v] · ∂logit_v/∂Σ_v
    where n_v is the number of input occurrences of token v in the batch.
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
        _decode_tau = getattr(config, 'decode_tau', 1.0)
        logits = kl_decode_logits(mu_star, Sigma_star, model.prior_mu, model.prior_Sigma,
                                  decode_tau=_decode_tau)
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

    # Use eta_M directly. No confidence weighting — at initialization CE ≈ 11
    # (random PPL), which would shrink eta_M by ~12x via 1/(1+CE), creating a
    # Catch-22 where the model can't learn because predictions are bad, and
    # predictions are bad because the model can't learn.
    effective_eta_M = config.eta_M

    grad_clamp = getattr(config, 'grad_clamp', 1e3)

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

    # Hyper-prior (frequency-adaptive: stronger for rare tokens)
    _rare_reg = getattr(config, 'rare_token_reg', 0.0)
    if _rare_reg > 0:
        _freq_weight = 1.0 + _rare_reg / n_safe  # [T] — larger for rare tokens
        grad_mu = grad_mu_vfe + _freq_weight.unsqueeze(-1) * mu_all / config.hyper_var
    else:
        grad_mu = grad_mu_vfe + mu_all / config.hyper_var

    # Observation gradient — normalize per-token with a floor to prevent
    # rare tokens (n_v=1) from getting disproportionately large updates.
    # Without floor: a single-occurrence token gets gradient / 1 vs / BN,
    # causing 4096x amplification that overfits rare token priors.
    obs_diff = obs_weighted_mu - obs_ce_sum.unsqueeze(-1) * mu_all  # [T, K]
    _obs_floor = getattr(config, 'obs_norm_floor', 0)
    if _obs_floor <= 0:
        _obs_floor = max(8, int(B * N * 0.01))  # Auto: 1% of BN
    obs_norm = n_safe.clamp(min=_obs_floor).unsqueeze(-1)  # [T, 1]
    obs_grad_mu = torch.einsum('tij,tj->ti', Sigma_all_inv, obs_diff / obs_norm)
    # Chain rule: logits = -KL/τ, so ∂logit/∂μ_v includes a 1/τ factor
    _decode_tau = getattr(config, 'decode_tau', 1.0)
    if _decode_tau != 1.0:
        obs_grad_mu = obs_grad_mu / _decode_tau
    grad_mu = grad_mu + obs_grad_mu

    # Gradient clamping (ported from VFE dynamic)
    grad_mu = torch.clamp(grad_mu, -grad_clamp, grad_clamp)

    # Natural gradient and update
    nat_mu = torch.einsum('tij,tj->ti', Sigma_all, grad_mu)  # [T, K]
    nat_mu = clip_norm(nat_mu, config.m_step_trust_mu)
    mu_new = mu_all - effective_eta_M * nat_mu

    # Enforce prior mean norm constraint (prevents mean spread → logit explosion)
    if config.prior_mu_max_norm > 0:
        mu_norms = mu_new.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = torch.clamp(config.prior_mu_max_norm / mu_norms, max=1.0)
        mu_new = mu_new * scale

    model.prior_mu[update_tokens] = mu_new

    # ---- Prior covariance gradient (vectorized) ----
    Sigma_star_avg = Sigma_star_sum / n_safe.unsqueeze(-1).unsqueeze(-1)
    outer_avg = outer_sum / n_safe.unsqueeze(-1).unsqueeze(-1)

    grad_Sigma_vfe = 0.5 * (
        Sigma_all_inv
        - Sigma_all_inv @ (Sigma_star_avg + outer_avg) @ Sigma_all_inv
    )
    grad_Sigma_vfe[~has_input] = 0.0

    # Hyper-prior: ∂KL(p_v || h)/∂Σ_v where h = N(0, σ²_h I)
    # = ½[(1/σ²_h)I - Σ_v⁻¹]  (pulls Σ_v toward σ²_h·I, not always-shrink)
    _eye = torch.eye(K, device=dev, dtype=mu_star.dtype)
    grad_Sigma = grad_Sigma_vfe + 0.5 * (_eye / config.hyper_var - Sigma_all_inv)

    # Observation gradient for Σ_v (togglable via config.sigma_obs_grad)
    sigma_obs_mode = getattr(config, 'sigma_obs_grad', 'none')
    if sigma_obs_mode == 'full':
        # Full analytic observation gradient (original, error-prone)
        W = (
            obs_weighted_Sigma + obs_weighted_outer
            - obs_weighted_mu.unsqueeze(-1) * mu_all.unsqueeze(-2)
            - mu_all.unsqueeze(-1) * obs_weighted_mu.unsqueeze(-2)
            + obs_ce_sum.unsqueeze(-1).unsqueeze(-1) * (mu_all.unsqueeze(-1) * mu_all.unsqueeze(-2))
        )  # [T, K, K]
        obs_grad_Sigma = 0.5 * (
            Sigma_all_inv @ W @ Sigma_all_inv
            - obs_ce_sum.unsqueeze(-1).unsqueeze(-1) * Sigma_all_inv
        ) / obs_norm.unsqueeze(-1)
        obs_grad_Sigma = symmetrize(obs_grad_Sigma)
        # Chain rule: logits = -KL/τ, so ∂logit/∂Σ_v includes a 1/τ factor
        if _decode_tau != 1.0:
            obs_grad_Sigma = obs_grad_Sigma / _decode_tau
        grad_Sigma = grad_Sigma + obs_grad_Sigma
    elif sigma_obs_mode == 'diagonal':
        # Diagonal approximation: only use the diagonal of the full obs gradient
        W_diag = (
            torch.diagonal(obs_weighted_Sigma, dim1=-2, dim2=-1)
            + torch.diagonal(obs_weighted_outer, dim1=-2, dim2=-1)
            - 2 * obs_weighted_mu * mu_all
            + obs_ce_sum.unsqueeze(-1) * mu_all ** 2
        )  # [T, K]
        Sigma_all_inv_diag = torch.diagonal(Sigma_all_inv, dim1=-2, dim2=-1)  # [T, K]
        obs_diag = 0.5 * (Sigma_all_inv_diag ** 2 * W_diag - obs_ce_sum.unsqueeze(-1) * Sigma_all_inv_diag) / obs_norm
        # Chain rule: logits = -KL/τ
        if _decode_tau != 1.0:
            obs_diag = obs_diag / _decode_tau
        grad_Sigma = grad_Sigma + torch.diag_embed(obs_diag)
    # else: sigma_obs_mode == 'none' — match VFE dynamic, no obs gradient for Sigma

    # Gradient clamping
    grad_Sigma = torch.clamp(grad_Sigma, -grad_clamp, grad_clamp)

    # Natural gradient on SPD and retract
    nat_Sigma = natural_grad_sigma(grad_Sigma, Sigma_all)
    nat_Sigma = clip_matrix_norm(nat_Sigma, config.trust_region_sigma)
    # Use even smaller step for Sigma (5x slower, matching dynamic transformer's sigma_lr/mu_lr ratio)
    Sigma_new = retract_spd(
        Sigma_all, nat_Sigma, effective_eta_M * 0.2,
        eps_min=config.spd_eps_min, kappa_max=config.spd_kappa_max,
    )

    # Enforce prior covariance spectral floor (prevents collapse → divergence)
    floor = config.prior_sigma_floor
    if floor > 0:
        eigs, V = torch.linalg.eigh(Sigma_new)
        clamped = eigs.clamp(min=floor)
        if (eigs < floor).any():
            Sigma_new = V @ torch.diag_embed(clamped) @ V.transpose(-2, -1)
            Sigma_new = symmetrize(Sigma_new)

    model.prior_Sigma[update_tokens] = Sigma_new

    # ---- Prior gauge frame gradient (vectorized) ----
    omega_grad_clamp = getattr(config, 'omega_grad_clamp', 10.0)

    Omega_all = model.prior_Omega[update_tokens]       # [T, H, K_h, K_h]
    Omega_star_avg = Omega_star_sum / n_safe.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    grad_Omega_all = -(Omega_star_avg - Omega_all)
    grad_Omega_all[~has_input] = 0.0

    # Element-wise safety clamp on Euclidean gradient
    grad_Omega_all = torch.clamp(grad_Omega_all, -omega_grad_clamp, omega_grad_clamp)

    if model.prior_phi is not None:
        # Phi path: update phi coordinates, then recompute Omega
        # Gradient in Lie algebra: ∂F/∂φ^a = tr(∂F/∂Ω · Ω · T_a)
        # (first-order approximation; dexp correction negligible near identity)
        phi_all = model.prior_phi[update_tokens]  # [T, H, n_gen_h]
        grad_phi = torch.einsum(
            'thij,thik,akj->tha',
            grad_Omega_all, Omega_all, model.gl_generators
        )  # [T, H, n_gen_h]
        grad_phi = torch.clamp(grad_phi, -grad_clamp, grad_clamp)
        phi_new = retract_phi(phi_all, -effective_eta_M * grad_phi, config.phi_max_norm)
        model.prior_phi[update_tokens] = phi_new
    else:
        # Omega path: Lie algebra clip + conditioning regularization
        nat_Omega = lie_algebra_clip_grad(
            grad_Omega_all, Omega_all,
            trust_radius=config.trust_region_omega,
        )
        Omega_new = Omega_all - effective_eta_M * nat_Omega
        Omega_new = regularize_omega_conditioning(Omega_new, config.omega_cond_max)
        model.prior_Omega[update_tokens] = Omega_new

    # ================================================================
    # 4. Update positional gauge offsets
    # ================================================================
    _update_pos_omega(Omega_star, token_ids, model, config)

    # Sync Omega from phi if using phi path
    if model.prior_phi is not None:
        model.sync_omega_from_phi()

    return ce_loss.item()


def _update_pos_omega(Omega_star, token_ids, model, config):
    """
    Update positional gauge offsets toward mean converged frames per position.
    Vectorized over all positions.
    """
    B, N = token_ids.shape
    omega_grad_clamp = getattr(config, 'omega_grad_clamp', 10.0)
    grad_clamp = getattr(config, 'grad_clamp', 1e3)

    # Average converged gauge at each position across batch: [N, H, K_h, K_h]
    Om_avg = Omega_star.mean(0)               # [N, H, K_h, K_h]
    pos_Om = model.pos_Omega[:N]              # [N, H, K_h, K_h]

    grad = -(Om_avg - pos_Om)
    grad = torch.clamp(grad, -omega_grad_clamp, omega_grad_clamp)

    if model.pos_phi is not None:
        # Phi path: update pos_phi coordinates
        pos_phi = model.pos_phi[:N]  # [N, H, n_gen_h]
        grad_phi = torch.einsum(
            'nhij,nhik,akj->nha',
            grad, pos_Om, model.gl_generators
        )  # [N, H, n_gen_h]
        grad_phi = torch.clamp(grad_phi, -grad_clamp, grad_clamp)
        model.pos_phi[:N] = retract_phi(
            pos_phi, -config.eta_M * 0.1 * grad_phi, config.phi_max_norm
        )
    else:
        # Omega path: Lie algebra clip + conditioning regularization
        nat = lie_algebra_clip_grad(
            grad, pos_Om, trust_radius=config.trust_region_omega,
        )
        pos_Om_new = pos_Om - config.eta_M * 0.1 * nat
        pos_Om_new = regularize_omega_conditioning(pos_Om_new, config.omega_cond_max)
        model.pos_Omega[:N] = pos_Om_new
