r"""M-Step: Prior bank parameter updates via natural gradient on marginal VFE.

Observation gradient enters HERE (not in E-step).
No autograd for mu/Sigma priors — those gradients are analytic.
The Omega prior gradient uses a single autograd pass through the coupling
VFE to capture the full derivative including softmax correction terms.

The M-step updates global parameters (the prior bank) using sufficient
statistics from converged E-step beliefs. These statistics are additive
across data positions, so they factorize across micro-batches. The
``MStepAccumulator`` collects them into vocabulary-sized buffers; after
all micro-batches, ``apply_m_step_from_accumulated`` consumes the totals
in a single parameter update.

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
    precompute_tokens,
    pairwise_kl,
    kl_attention,
)
from .gauge import (
    natural_grad_omega,
    lie_algebra_clip_grad,
    relative_trust_clip,
    regularize_omega_conditioning,
    retract_phi,
    vfe_grad_Omega,
)
from .inference import extract_block_diag


# =========================================================================
# Analytical M-step Omega gradient
# =========================================================================

def _compute_m_step_omega_grad(
    mu_star: torch.Tensor,
    Sigma_star: torch.Tensor,
    token_ids: torch.Tensor,
    model,
    config,
) -> torch.Tensor:
    r"""Compute :math:`\partial F_{\text{coupling}} / \partial \Omega_i`
    at prior Omega values using converged beliefs.

    The VFE coupling term is
    :math:`F = \sum_{ij} \beta_{ij} \, \mathrm{KL}(q_i \| \Omega_{ij} q_j)`.
    The M-step gradient for ``prior_Omega_v`` is
    :math:`\sum_{i: \text{token}_i = v} \partial F / \partial \Omega_i`
    evaluated at the initialization point :math:`\Omega_i = \text{prior\_Omega}_v`
    (not the converged value, where the gradient vanishes by E-step optimality).

    Uses a single autograd pass through the differentiable coupling VFE to get
    the exact gradient, including the softmax correction
    :math:`\partial \beta / \partial \Omega` terms.

    Args:
        mu_star: [B, N, K] converged belief means.
        Sigma_star: [B, N, K, K] converged belief covariances.
        token_ids: [B, N] input token indices.
        model: PureVFETransformer (provides ``prior_Omega``).
        config: PureVFEConfig (provides H, K_h, tau, causal).

    Returns:
        [B, N, H, K_h, K_h] per-position gradient (detached).
    """
    B, N, K = mu_star.shape
    H = config.n_heads
    K_h = config.head_dim

    # Prior Omega (initialization point — where we evaluate the gradient)
    prior_Omega = model.prior_Omega[token_ids].clone()  # [B, N, H, K_h, K_h]

    # Build per-head quantities from converged full-K beliefs (detached)
    mu_h = mu_star.detach().view(B, N, H, K_h)
    Sigma_h = extract_block_diag(Sigma_star.detach(), H, K_h)

    # Temporarily enable grad (M-step runs under @torch.no_grad)
    with torch.enable_grad():
        prior_Omega.requires_grad_(True)

        # Forward: compute coupling VFE differentiably through prior_Omega
        precomp = precompute_tokens(mu_h, Sigma_h, prior_Omega)
        kl_ij = pairwise_kl(precomp, causal=config.causal)
        tau = config.tau if config.tau is not None else K_h ** 0.5
        beta = kl_attention(
            kl_ij, tau, causal=config.causal,
            mask_self=getattr(config, 'mask_self_attention', False),
        )
        F_coupling = (beta * kl_ij).sum()

        # Single backward pass — cheap (one VFE evaluation, no E-step)
        grad_Omega = torch.autograd.grad(
            F_coupling, prior_Omega, create_graph=False,
        )[0]

    return grad_Omega.detach()


# =========================================================================
# Gradient accumulation for M-step
# =========================================================================

class MStepAccumulator:
    r"""Accumulates M-step sufficient statistics across micro-batches.

    The EM M-step gradient is a sum over all data:

    .. math::
        \nabla_\theta F = \sum_{n=1}^N \nabla_\theta F_n

    A single-batch M-step uses a noisy estimate of this sum. This class
    collects the per-token sufficient statistics — ``n_counts``,
    ``mu_star_sum``, ``Sigma_star_sum``, ``outer_sum``, ``Omega_star_sum``,
    and observation gradient quantities — into vocabulary-sized buffers.
    After K micro-batches, ``apply_m_step_from_accumulated`` applies one
    M-step using the accumulated (lower-variance) gradient.

    All buffers are indexed by vocabulary token ID (V-sized), so different
    micro-batches with different token sets merge automatically via addition.
    """

    def __init__(self, config, device):
        V = config.vocab_size
        K = config.belief_dim
        H = config.n_heads
        K_h = config.head_dim
        N = config.max_seq_len

        # Per-token E-step statistics (VFE gradient)
        self.n_counts = torch.zeros(V, device=device)
        self.mu_star_sum = torch.zeros(V, K, device=device)
        self.Sigma_star_sum = torch.zeros(V, K, K, device=device)
        self.outer_sum = torch.zeros(V, K, K, device=device)
        self.Omega_star_sum = torch.zeros(V, H, K_h, K_h, device=device)

        # Analytical Omega gradient (accumulated across micro-batches)
        self.grad_Omega_sum = torch.zeros(V, H, K_h, K_h, device=device)

        # Observation gradient quantities
        self.obs_weighted_mu = torch.zeros(V, K, device=device)
        self.obs_ce_sum = torch.zeros(V, device=device)
        # Sigma obs quantities (only used when sigma_obs_grad != 'none')
        sigma_obs = getattr(config, 'sigma_obs_grad', 'none')
        if sigma_obs != 'none':
            self.obs_weighted_Sigma = torch.zeros(V, K, K, device=device)
            self.obs_weighted_outer = torch.zeros(V, K, K, device=device)
        else:
            self.obs_weighted_Sigma = None
            self.obs_weighted_outer = None

        # Monitoring
        self.ce_loss_sum = 0.0
        self.n_micro_batches = 0

        self._config = config
        self._device = device

    @torch.no_grad()
    def accumulate(self, token_ids, targets, mu_star, Sigma_star, Omega_star,
                   model, config, logits=None):
        r"""Add one micro-batch's sufficient statistics.

        Args:
            token_ids: [B, N] input token indices
            targets: [B, N] target token indices
            mu_star: [B, N, K] converged beliefs
            Sigma_star: [B, N, K, K] converged covariances
            Omega_star: [B, N, H, K_h, K_h] converged gauge frames
            model: PureVFETransformer (priors read but not modified)
            config: PureVFEConfig
            logits: [B, N, V] optional precomputed logits

        Returns:
            ce_loss: scalar CE loss for this micro-batch (monitoring)
        """
        B, N = token_ids.shape
        K = config.belief_dim
        H = config.n_heads
        K_h = config.head_dim
        BN = B * N
        dev = mu_star.device

        # --- Observation gradient (analytic softmax-CE) ---
        if logits is None:
            _decode_tau = getattr(config, 'decode_tau', 1.0)
            logits = kl_decode_logits(
                mu_star, Sigma_star, model.prior_mu, model.prior_Sigma,
                decode_tau=_decode_tau,
            )
        ce_grad = softmax_ce_gradient(logits, targets)  # [B, N, V]

        # CE loss for monitoring
        log_probs = torch.log_softmax(logits, dim=-1)
        ce_loss = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1).mean()

        # --- Tokens to update ---
        update_tokens = torch.unique(torch.cat([
            token_ids.reshape(-1), targets.reshape(-1),
        ]))

        # --- Observation quantities (scatter into V-sized buffers) ---
        obs = _precompute_obs_gradient(ce_grad, mu_star, Sigma_star, update_tokens)
        self.obs_weighted_mu[update_tokens] += obs['obs_weighted_mu']
        self.obs_ce_sum[update_tokens] += obs['obs_ce_sum']
        if self.obs_weighted_Sigma is not None:
            self.obs_weighted_Sigma[update_tokens] += obs['obs_weighted_Sigma']
            self.obs_weighted_outer[update_tokens] += obs['obs_weighted_outer']

        # --- Per-token E-step statistics (scatter into V-sized buffers) ---
        flat_ids = token_ids.reshape(-1)  # [BN]
        mu_star_flat = mu_star.reshape(BN, K)
        Sigma_star_flat = Sigma_star.reshape(BN, K, K)
        Omega_star_flat = Omega_star.reshape(BN, H, K_h, K_h)

        # n_counts
        self.n_counts.scatter_add_(
            0, flat_ids, torch.ones(BN, device=dev, dtype=self.n_counts.dtype),
        )

        # mu_star_sum
        self.mu_star_sum.scatter_add_(
            0, flat_ids.unsqueeze(-1).expand(-1, K), mu_star_flat,
        )

        # Sigma_star_sum
        self.Sigma_star_sum.scatter_add_(
            0, flat_ids.unsqueeze(-1).unsqueeze(-1).expand(-1, K, K),
            Sigma_star_flat,
        )

        # outer_sum: (mu_star - mu_v)(mu_star - mu_v)^T per token
        mu_v_expanded = model.prior_mu[flat_ids]  # [BN, K]
        mu_diff = mu_star_flat - mu_v_expanded
        outer = mu_diff.unsqueeze(-1) * mu_diff.unsqueeze(-2)  # [BN, K, K]
        self.outer_sum.scatter_add_(
            0, flat_ids.unsqueeze(-1).unsqueeze(-1).expand(-1, K, K), outer,
        )

        # Omega_star_sum (kept for fallback / diagnostics)
        self.Omega_star_sum.scatter_add_(
            0, flat_ids.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, H, K_h, K_h),
            Omega_star_flat,
        )

        # Analytical Omega gradient at prior Omega values
        if getattr(config, 'use_analytical_omega_grad', True):
            grad_Omega_pos = _compute_m_step_omega_grad(
                mu_star, Sigma_star, token_ids, model, config,
            )  # [B, N, H, K_h, K_h]
            grad_Omega_flat = grad_Omega_pos.reshape(BN, H, K_h, K_h)
            self.grad_Omega_sum.scatter_add_(
                0,
                flat_ids.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, H, K_h, K_h),
                grad_Omega_flat,
            )

        # --- Monitoring ---
        self.ce_loss_sum += ce_loss.item()
        self.n_micro_batches += 1

        return ce_loss.item()

    def reset(self):
        """Zero all buffers for the next accumulation window."""
        self.n_counts.zero_()
        self.mu_star_sum.zero_()
        self.Sigma_star_sum.zero_()
        self.outer_sum.zero_()
        self.Omega_star_sum.zero_()
        self.grad_Omega_sum.zero_()
        self.obs_weighted_mu.zero_()
        self.obs_ce_sum.zero_()
        if self.obs_weighted_Sigma is not None:
            self.obs_weighted_Sigma.zero_()
            self.obs_weighted_outer.zero_()
        self.ce_loss_sum = 0.0
        self.n_micro_batches = 0

    @property
    def avg_ce_loss(self):
        if self.n_micro_batches == 0:
            return 0.0
        return self.ce_loss_sum / self.n_micro_batches


@torch.no_grad()
def apply_m_step_from_accumulated(accum, model, config, effective_lrs=None):
    r"""Apply prior updates using accumulated sufficient statistics.

    Consumes the vocabulary-sized buffers in ``accum`` and updates the
    model's prior bank. Equivalent to running ``m_step`` on the union
    of all micro-batches that were accumulated.

    Args:
        accum: MStepAccumulator with accumulated statistics
        model: PureVFETransformer
        config: PureVFEConfig
        effective_lrs: optional dict of per-variable LRs (from scheduler)

    Returns:
        ce_loss: average CE loss across accumulated micro-batches
    """
    K = config.belief_dim
    H = config.n_heads
    K_h = config.head_dim
    dev = accum.n_counts.device

    if effective_lrs is None:
        effective_lrs = {
            'mu_p_lr': config.mu_p_lr,
            'sigma_p_lr': config.sigma_p_lr,
            'phi_lr': config.phi_lr,
        }
    use_adam = getattr(config, 'use_adam_m_step', False) and model.m1_mu is not None
    grad_clamp = getattr(config, 'grad_clamp', 1e3)

    # --- Find tokens that appeared in ANY micro-batch ---
    has_input = accum.n_counts > 0
    update_tokens = has_input.nonzero(as_tuple=True)[0]  # [T]
    T = len(update_tokens)
    if T == 0:
        return accum.avg_ce_loss

    # Extract accumulated stats for active tokens
    n_counts = accum.n_counts[update_tokens]              # [T]
    n_safe = n_counts.clamp(min=1)
    mu_star_sum = accum.mu_star_sum[update_tokens]        # [T, K]
    Sigma_star_sum = accum.Sigma_star_sum[update_tokens]  # [T, K, K]
    outer_sum = accum.outer_sum[update_tokens]            # [T, K, K]
    Omega_star_sum = accum.Omega_star_sum[update_tokens]  # [T, H, K_h, K_h]
    obs_weighted_mu = accum.obs_weighted_mu[update_tokens]  # [T, K]
    obs_ce_sum = accum.obs_ce_sum[update_tokens]            # [T]

    # Current priors
    mu_all = model.prior_mu[update_tokens]                # [T, K]
    Sigma_all = model.prior_Sigma[update_tokens]          # [T, K, K]
    Sigma_all_inv = safe_inverse(Sigma_all)

    # ================================================================
    # Prior mean gradient (same as m_step but from accumulated stats)
    # ================================================================
    mu_star_avg = mu_star_sum / n_safe.unsqueeze(-1)
    mu_diff_avg = mu_star_avg - mu_all
    grad_mu_vfe = -torch.einsum('tij,tj->ti', Sigma_all_inv, mu_diff_avg)
    grad_mu_vfe[n_counts == 0] = 0.0

    # Hyper-prior
    _rare_reg = getattr(config, 'rare_token_reg', 0.0)
    if _rare_reg > 0:
        _freq_weight = 1.0 + _rare_reg / n_safe
        grad_mu = grad_mu_vfe + _freq_weight.unsqueeze(-1) * mu_all / config.hyper_var
    else:
        grad_mu = grad_mu_vfe + mu_all / config.hyper_var

    # Observation gradient (accumulated across micro-batches)
    obs_diff = obs_weighted_mu - obs_ce_sum.unsqueeze(-1) * mu_all
    # Total BN across all micro-batches for obs_norm floor
    total_BN = int(accum.n_counts.sum().item())
    _obs_floor = getattr(config, 'obs_norm_floor', 0)
    if _obs_floor <= 0:
        _obs_floor = max(8, int(total_BN * 0.01))
    obs_norm = n_safe.clamp(min=_obs_floor).unsqueeze(-1)
    obs_grad_mu = torch.einsum('tij,tj->ti', Sigma_all_inv, obs_diff / obs_norm)
    _decode_tau = getattr(config, 'decode_tau', 1.0)
    if _decode_tau != 1.0:
        obs_grad_mu = obs_grad_mu / _decode_tau
    grad_mu = grad_mu + obs_grad_mu

    grad_mu = torch.clamp(grad_mu, -grad_clamp, grad_clamp)

    # Natural gradient and update
    nat_mu = torch.einsum('tij,tj->ti', Sigma_all, grad_mu)
    nat_mu = clip_norm(nat_mu, config.m_step_trust_mu)
    if use_adam:
        nat_mu = _adam_update_mu(nat_mu, update_tokens, model, config)
    mu_new = mu_all - effective_lrs['mu_p_lr'] * nat_mu

    if config.prior_mu_max_norm > 0:
        mu_norms = mu_new.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = torch.clamp(config.prior_mu_max_norm / mu_norms, max=1.0)
        mu_new = mu_new * scale
    model.prior_mu[update_tokens] = mu_new

    # ================================================================
    # Prior covariance gradient
    # ================================================================
    Sigma_star_avg = Sigma_star_sum / n_safe.unsqueeze(-1).unsqueeze(-1)
    outer_avg = outer_sum / n_safe.unsqueeze(-1).unsqueeze(-1)

    grad_Sigma_vfe = 0.5 * (
        Sigma_all_inv
        - Sigma_all_inv @ (Sigma_star_avg + outer_avg) @ Sigma_all_inv
    )
    grad_Sigma_vfe[n_counts == 0] = 0.0

    _eye = torch.eye(K, device=dev, dtype=mu_all.dtype)
    grad_Sigma = grad_Sigma_vfe + 0.5 * (_eye / config.hyper_var - Sigma_all_inv)

    # Observation gradient for Sigma (if enabled and accumulated)
    sigma_obs_mode = getattr(config, 'sigma_obs_grad', 'none')
    if sigma_obs_mode == 'full' and accum.obs_weighted_Sigma is not None:
        ows = accum.obs_weighted_Sigma[update_tokens]
        owo = accum.obs_weighted_outer[update_tokens]
        W = (
            ows + owo
            - obs_weighted_mu.unsqueeze(-1) * mu_all.unsqueeze(-2)
            - mu_all.unsqueeze(-1) * obs_weighted_mu.unsqueeze(-2)
            + obs_ce_sum.unsqueeze(-1).unsqueeze(-1) * (mu_all.unsqueeze(-1) * mu_all.unsqueeze(-2))
        )
        obs_grad_Sigma = 0.5 * (
            Sigma_all_inv @ W @ Sigma_all_inv
            - obs_ce_sum.unsqueeze(-1).unsqueeze(-1) * Sigma_all_inv
        ) / obs_norm.unsqueeze(-1)
        obs_grad_Sigma = symmetrize(obs_grad_Sigma)
        if _decode_tau != 1.0:
            obs_grad_Sigma = obs_grad_Sigma / _decode_tau
        grad_Sigma = grad_Sigma + obs_grad_Sigma
    elif sigma_obs_mode == 'diagonal' and accum.obs_weighted_Sigma is not None:
        ows_diag = torch.diagonal(accum.obs_weighted_Sigma[update_tokens], dim1=-2, dim2=-1)
        owo_diag = torch.diagonal(accum.obs_weighted_outer[update_tokens], dim1=-2, dim2=-1)
        W_diag = ows_diag + owo_diag - 2 * obs_weighted_mu * mu_all + obs_ce_sum.unsqueeze(-1) * mu_all ** 2
        Sigma_all_inv_diag = torch.diagonal(Sigma_all_inv, dim1=-2, dim2=-1)
        obs_diag = 0.5 * (Sigma_all_inv_diag ** 2 * W_diag - obs_ce_sum.unsqueeze(-1) * Sigma_all_inv_diag) / obs_norm
        if _decode_tau != 1.0:
            obs_diag = obs_diag / _decode_tau
        grad_Sigma = grad_Sigma + torch.diag_embed(obs_diag)

    grad_Sigma = torch.clamp(grad_Sigma, -grad_clamp, grad_clamp)

    nat_Sigma = natural_grad_sigma(grad_Sigma, Sigma_all)
    nat_Sigma = clip_matrix_norm(nat_Sigma, config.trust_region_sigma)
    if use_adam and model.m1_Sigma is not None:
        nat_Sigma = _momentum_update(nat_Sigma, model.m1_Sigma, update_tokens, config.adam_beta1)

    Sigma_new = retract_spd(
        Sigma_all, nat_Sigma, effective_lrs['sigma_p_lr'],
        eps_min=config.spd_eps_min, kappa_max=config.spd_kappa_max,
    )

    floor = config.prior_sigma_floor
    if floor > 0:
        eigs, V = torch.linalg.eigh(Sigma_new)
        clamped = eigs.clamp(min=floor)
        if (eigs < floor).any():
            Sigma_new = V @ torch.diag_embed(clamped) @ V.transpose(-2, -1)
            Sigma_new = symmetrize(Sigma_new)
    model.prior_Sigma[update_tokens] = Sigma_new

    # ================================================================
    # Prior gauge frame gradient
    # ================================================================
    omega_grad_clamp = getattr(config, 'omega_grad_clamp', 10.0)
    Omega_all = model.prior_Omega[update_tokens]

    if getattr(config, 'use_analytical_omega_grad', True):
        # Analytical VFE coupling gradient accumulated during micro-batches
        grad_Omega_sum = accum.grad_Omega_sum[update_tokens]   # [T, H, K_h, K_h]
        grad_Omega_all = grad_Omega_sum / n_safe.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    else:
        # Fallback: moment-matching heuristic (pull toward mean E-step value)
        Omega_star_avg = Omega_star_sum / n_safe.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        grad_Omega_all = -(Omega_star_avg - Omega_all)

    grad_Omega_all[n_counts == 0] = 0.0
    grad_Omega_all = torch.clamp(grad_Omega_all, -omega_grad_clamp, omega_grad_clamp)

    if model.prior_phi is not None:
        phi_all = model.prior_phi[update_tokens]
        grad_phi = torch.einsum(
            'thij,thik,akj->tha',
            grad_Omega_all, Omega_all, model.gl_generators,
        )
        grad_phi = torch.clamp(grad_phi, -grad_clamp, grad_clamp)
        phi_new = retract_phi(phi_all, -effective_lrs['phi_lr'] * grad_phi, config.phi_max_norm)
        model.prior_phi[update_tokens] = phi_new
    else:
        nat_Omega = lie_algebra_clip_grad(
            grad_Omega_all, Omega_all, trust_radius=config.trust_region_omega,
        )
        if use_adam and model.m1_Omega is not None:
            nat_Omega = _momentum_update(
                nat_Omega, model.m1_Omega, update_tokens, config.adam_beta1,
            )
        Omega_new = Omega_all - effective_lrs['phi_lr'] * nat_Omega
        Omega_new = regularize_omega_conditioning(Omega_new, config.omega_cond_max)
        model.prior_Omega[update_tokens] = Omega_new

    # Sync Omega from phi if using phi path
    if model.prior_phi is not None:
        model.sync_omega_from_phi()

    return accum.avg_ce_loss


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


def _adam_update_mu(nat_mu, update_tokens, model, config):
    r"""Apply Adam-like momentum to prior_mu natural gradient.

    Tracks EMA of first moment (momentum) and second moment (adaptive scaling)
    for variance reduction across batches. No neural components — purely an
    optimization algorithm on natural gradient outputs.

    Args:
        nat_mu: [T, K] natural gradient for active tokens
        update_tokens: [T] token indices
        model: PureVFETransformer with momentum buffers
        config: PureVFEConfig

    Returns:
        nat_mu_corrected: [T, K] bias-corrected Adam update direction
    """
    beta1 = config.adam_beta1
    beta2 = config.adam_beta2
    eps = config.adam_eps

    model.adam_step += 1
    t = model.adam_step

    # Update running moments for active tokens only
    model.m1_mu[update_tokens] = beta1 * model.m1_mu[update_tokens] + (1 - beta1) * nat_mu
    model.m2_mu[update_tokens] = beta2 * model.m2_mu[update_tokens] + (1 - beta2) * nat_mu ** 2

    # Bias-corrected estimates
    m1_hat = model.m1_mu[update_tokens] / (1 - beta1 ** t)
    m2_hat = model.m2_mu[update_tokens] / (1 - beta2 ** t)

    return m1_hat / (m2_hat.sqrt() + eps)


def _momentum_update(nat_grad, momentum_buffer, update_tokens, beta1):
    r"""Apply simple momentum (first moment EMA) to a natural gradient.

    Used for Sigma and Omega where Adam's per-element adaptive scaling
    interacts poorly with manifold geometry.

    Args:
        nat_grad: [T, ...] natural gradient for active tokens
        momentum_buffer: [V, ...] running first moment
        update_tokens: [T] token indices
        beta1: EMA decay factor

    Returns:
        corrected: [T, ...] momentum-corrected gradient
    """
    momentum_buffer[update_tokens] = beta1 * momentum_buffer[update_tokens] + (1 - beta1) * nat_grad
    return momentum_buffer[update_tokens]


@torch.no_grad()
def m_step(token_ids, targets, mu_star, Sigma_star, Omega_star, model, config,
           logits=None, effective_lrs=None):
    r"""Update prior bank via natural gradient on VFE.

    Observation gradient enters HERE (not in belief descent).

    Args:
        token_ids: [B, N] input token indices
        targets: [B, N] target token indices
        mu_star: [B, N, K] converged beliefs (means)
        Sigma_star: [B, N, K, K] converged beliefs (covariances)
        Omega_star: [B, N, H, K_h, K_h] converged gauge frames
        model: PureVFETransformer
        config: PureVFEConfig
        logits: [B, N, V] optional precomputed logits from e_step
        effective_lrs: optional dict of per-variable LRs (from scheduler)

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

    if effective_lrs is None:
        effective_lrs = {
            'mu_p_lr': config.mu_p_lr,
            'sigma_p_lr': config.sigma_p_lr,
            'phi_lr': config.phi_lr,
        }
    use_adam = getattr(config, 'use_adam_m_step', False) and model.m1_mu is not None

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

    # Adam momentum for μ (variance reduction + adaptive step sizes)
    if use_adam:
        nat_mu = _adam_update_mu(nat_mu, update_tokens, model, config)

    mu_new = mu_all - effective_lrs['mu_p_lr'] * nat_mu

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

    # Momentum for Σ (first moment only — adaptive scaling conflicts with SPD geometry)
    if use_adam and model.m1_Sigma is not None:
        nat_Sigma = _momentum_update(
            nat_Sigma, model.m1_Sigma, update_tokens, config.adam_beta1
        )

    Sigma_new = retract_spd(
        Sigma_all, nat_Sigma, effective_lrs['sigma_p_lr'],
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

    # ---- Prior gauge frame gradient ----
    omega_grad_clamp = getattr(config, 'omega_grad_clamp', 10.0)
    Omega_all = model.prior_Omega[update_tokens]       # [T, H, K_h, K_h]

    if getattr(config, 'use_analytical_omega_grad', True):
        # Analytical VFE coupling gradient at prior Omega values
        grad_Omega_pos = _compute_m_step_omega_grad(
            mu_star, Sigma_star, token_ids, model, config,
        )  # [B, N, H, K_h, K_h]
        # Scatter-add per-position gradients to per-token [T, H, K_h, K_h]
        grad_Omega_flat = grad_Omega_pos.reshape(BN, H, K_h, K_h)
        grad_Omega_token = torch.zeros(T, H, K_h, K_h, device=dev, dtype=mu_star.dtype)
        grad_Omega_token.scatter_add_(
            0,
            flat_idx.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, H, K_h, K_h),
            grad_Omega_flat,
        )
        grad_Omega_all = grad_Omega_token / n_safe.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    else:
        # Fallback: moment-matching heuristic
        Omega_star_avg = Omega_star_sum / n_safe.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        grad_Omega_all = -(Omega_star_avg - Omega_all)

    grad_Omega_all[~has_input] = 0.0
    grad_Omega_all = torch.clamp(grad_Omega_all, -omega_grad_clamp, omega_grad_clamp)

    if model.prior_phi is not None:
        phi_all = model.prior_phi[update_tokens]  # [T, H, n_gen_h]
        grad_phi = torch.einsum(
            'thij,thik,akj->tha',
            grad_Omega_all, Omega_all, model.gl_generators
        )  # [T, H, n_gen_h]
        grad_phi = torch.clamp(grad_phi, -grad_clamp, grad_clamp)
        phi_new = retract_phi(phi_all, -effective_lrs['phi_lr'] * grad_phi, config.phi_max_norm)
        model.prior_phi[update_tokens] = phi_new
    else:
        nat_Omega = lie_algebra_clip_grad(
            grad_Omega_all, Omega_all,
            trust_radius=config.trust_region_omega,
        )
        if use_adam and model.m1_Omega is not None:
            nat_Omega = _momentum_update(
                nat_Omega, model.m1_Omega, update_tokens, config.adam_beta1
            )
        Omega_new = Omega_all - effective_lrs['phi_lr'] * nat_Omega
        Omega_new = regularize_omega_conditioning(Omega_new, config.omega_cond_max)
        model.prior_Omega[update_tokens] = Omega_new

    # ================================================================
    # 4. Update positional gauge offsets
    # ================================================================
    # Sync Omega from phi if using phi path
    if model.prior_phi is not None:
        model.sync_omega_from_phi()

    return ce_loss.item()
