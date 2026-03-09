# -*- coding: utf-8 -*-
"""
Variational Free Energy Training Loss
=======================================

Full VFE loss with all six terms from the gauge-theoretic framework:

    L = CE
        + λ_β · Σ β_ij · KL(q_i || Ω_ij q_j)      [belief coupling]
        + α   · Σ KL(q_i || p_i)                     [self-coupling]
        + λ_γ · Σ γ_ij · KL(s_i || Ω_ij s_j)        [model coupling]
        + λ_h · Σ KL(s_i || h)                       [hyper-prior]
        + (α_φ/2) · Σ ||φ_i||²                       [gauge prior]

Terms 1-3 operate on beliefs q (fast E-step subsystem).
Terms 4-6 operate on models s (slow M-step subsystem).

Setting λ_γ = λ_h = α_φ = 0 recovers the standard transformer training loss.
"""

import math
import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from transformer_v2.attention import compute_attention_weights
from transformer_v2.kl_ops import compute_kl_matrix


# =============================================================================
# Per-Agent Gaussian KL Divergence
# =============================================================================

def gaussian_kl_divergence(
    mu_q: torch.Tensor,
    sigma_q: Optional[torch.Tensor],
    mu_p: torch.Tensor,
    sigma_p: Optional[torch.Tensor],
    eps: float = 1e-6,
) -> torch.Tensor:
    """KL(N(μ_q, Σ_q) || N(μ_p, Σ_p)) per agent.

    Handles full (B, N, K, K), diagonal (B, N, K), and None (identity) covariances.

    Returns:
        kl: (B, N) per-agent KL divergences.
    """
    K = mu_q.shape[-1]
    device = mu_q.device
    dtype = mu_q.dtype

    sigma_q_is_diag = sigma_q is not None and sigma_q.dim() == 3
    sigma_p_is_diag = sigma_p is not None and sigma_p.dim() == 3
    use_diagonal = sigma_q_is_diag or sigma_p_is_diag

    if use_diagonal:
        if sigma_q is None:
            sq = torch.ones(*mu_q.shape, device=device, dtype=dtype)
        elif sigma_q.dim() == 3:
            sq = sigma_q
        else:
            sq = torch.diagonal(sigma_q, dim1=-2, dim2=-1)

        if sigma_p is None:
            sp = torch.ones(*mu_p.shape, device=device, dtype=dtype)
        elif sigma_p.dim() == 3:
            sp = sigma_p
        else:
            sp = torch.diagonal(sigma_p, dim1=-2, dim2=-1)

        sq = sq.clamp(min=eps)
        sp = sp.clamp(min=eps)

        trace = (sq / sp).sum(dim=-1)
        delta = mu_p - mu_q
        mahal = ((delta ** 2) / sp).sum(dim=-1)
        logdet = (torch.log(sp) - torch.log(sq)).sum(dim=-1)
        kl = 0.5 * (trace + mahal - K + logdet)
    else:
        I = torch.eye(K, device=device, dtype=dtype)
        if sigma_q is None:
            sigma_q = I.expand(*mu_q.shape[:-1], K, K)
        if sigma_p is None:
            sigma_p = I.expand(*mu_p.shape[:-1], K, K)

        sq_reg = sigma_q + eps * I
        sp_reg = sigma_p + eps * I

        L_p = _safe_cholesky(sp_reg, eps, I)
        L_q = _safe_cholesky(sq_reg, eps, I)

        # tr(Σ_p⁻¹ Σ_q)
        Y = torch.linalg.solve_triangular(L_p, sq_reg, upper=False)
        Z = torch.linalg.solve_triangular(L_p.transpose(-1, -2), Y, upper=True)
        trace = torch.diagonal(Z, dim1=-2, dim2=-1).sum(dim=-1)

        # (μ_p - μ_q)ᵀ Σ_p⁻¹ (μ_p - μ_q)
        delta = (mu_p - mu_q).unsqueeze(-1)
        v = torch.linalg.solve_triangular(L_p, delta, upper=False).squeeze(-1)
        mahal = (v ** 2).sum(dim=-1)

        logdet_p = 2.0 * torch.log(torch.diagonal(L_p, dim1=-2, dim2=-1).clamp(min=1e-12)).sum(dim=-1)
        logdet_q = 2.0 * torch.log(torch.diagonal(L_q, dim1=-2, dim2=-1).clamp(min=1e-12)).sum(dim=-1)
        kl = 0.5 * (trace + mahal - K + logdet_p - logdet_q)

    kl_ceil = max(100.0, 5.0 * K)
    kl = torch.clamp(kl, min=0.0, max=kl_ceil)
    return kl.nan_to_num(nan=0.0, posinf=kl_ceil, neginf=0.0)


def _safe_cholesky(M: torch.Tensor, eps: float, I: torch.Tensor) -> torch.Tensor:
    """Cholesky with progressive regularization fallback."""
    try:
        return torch.linalg.cholesky(M)
    except RuntimeError:
        reg = eps
        for _ in range(5):
            reg *= 10.0
            M_reg = M + reg * I
            M_reg = 0.5 * (M_reg + M_reg.transpose(-1, -2))
            try:
                return torch.linalg.cholesky(M_reg)
            except RuntimeError:
                continue
        return torch.linalg.cholesky(I.expand_as(M) + eps * I)


# =============================================================================
# Full VFE Training Loss
# =============================================================================

def compute_vfe_loss(
    model,
    token_ids: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.0,
    lambda_beta: float = 0.0,
    lambda_gamma: float = 0.0,
    kappa_gamma: float = 1.0,
    lambda_hyper: float = 0.0,
    alpha_phi: float = 0.0,
    pad_token_id: int = -100,
    use_obs_in_vfe: bool = False,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute full VFE training loss.

    The six-term loss from the gauge-theoretic framework:
        L = CE + α·KL(q||p) + λ_β·Σβ·KL(q||Ωq)
          + λ_γ·Σγ·KL(s||Ωs) + λ_h·KL(s||h) + (α_φ/2)||φ||²

    Args:
        model: GaugeTransformerLM with forward_with_attention()
        token_ids: (B, N) input tokens
        targets: (B, N) target tokens
        alpha: Self-coupling weight KL(q_i || p_i)
        lambda_beta: Belief coupling weight
        lambda_gamma: Model coupling weight (0 = off, the standard transformer limit)
        kappa_gamma: Temperature for γ_ij model coupling weights
        lambda_hyper: Hyper-prior weight KL(s_i || h)
        alpha_phi: Gauge prior weight (α_φ/2) Σ ||φ_i||²
        pad_token_id: Token ID to ignore in CE loss
        use_obs_in_vfe: Pass targets into VFE E-step

    Returns:
        total_loss: Scalar for backprop
        metrics: Dict with per-term losses and diagnostics
    """
    # Forward pass with attention tracking
    vfe_targets = targets if use_obs_in_vfe else None
    logits, attn_info = model.forward_with_attention(token_ids, targets=vfe_targets)

    beta = attn_info['beta']        # (n_layers, B, n_heads, N, N)
    kl = attn_info['kl']            # (n_layers, B, n_heads, N, N)
    mu_q = attn_info['mu']          # (B, N, K) evolved beliefs (fast)
    sigma_q = attn_info['sigma']    # (B, N, K, K) or (B, N, K) or None

    # Models s_i = embedding outputs (slow subsystem)
    mu_s = attn_info['mu_prior']        # (B, N, K)
    sigma_s = attn_info['sigma_prior']  # (B, N, K, K) or (B, N, K)
    phi_s = attn_info['phi_prior']      # (B, N, n_gen)
    generators = model.generators       # (n_gen, K, K)

    # Priors p_i = f(s_i). Currently p = s (identity mapping).
    mu_p = mu_s
    sigma_p = sigma_s

    K = mu_q.shape[-1]
    dim_scale = math.sqrt(max(K, 1))

    # ─── 1. Observation likelihood: CE ────────────────────────────────
    ce_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction='mean',
        ignore_index=pad_token_id,
    )

    # ─── 2. Belief coupling: λ_β · Σ β_ij · KL(q_i || Ω_ij q_j) ────
    if lambda_beta > 0.0 and beta is not None and kl is not None:
        beta_final = beta[-1]  # Last layer
        kl_final = kl[-1]
        weighted_kl = beta_final * kl_final
        belief_align_loss = lambda_beta * weighted_kl.sum(dim=(-2, -1)).mean() / dim_scale
    else:
        belief_align_loss = torch.tensor(0.0, device=ce_loss.device)

    # ─── 3. Self-coupling: α · KL(q_i || p_i) ────────────────────────
    if alpha > 0.0:
        kl_self = gaussian_kl_divergence(
            mu_q, sigma_q.detach() if sigma_q is not None else None,
            mu_p, sigma_p.detach() if sigma_p is not None else None,
        )
        self_consistency_loss = alpha * kl_self.mean() / dim_scale
    else:
        self_consistency_loss = torch.tensor(0.0, device=ce_loss.device)

    # ─── 4. Model coupling: λ_γ · Σ γ_ij · KL(s_i || Ω_ij s_j) ─────
    if lambda_gamma > 0.0:
        B, N, _ = mu_s.shape
        mask = torch.tril(torch.ones(N, N, device=mu_s.device))
        mask = mask.unsqueeze(0).expand(B, -1, -1)

        diagonal_cov = sigma_s is not None and sigma_s.dim() == 3
        gamma, kl_model = compute_attention_weights(
            mu_s, sigma_s, phi_s, generators,
            kappa_gamma,
            mask=mask,
            return_kl=True,
            diagonal_covariance=diagonal_cov,
        )
        weighted_kl_model = gamma * kl_model
        model_align_loss = lambda_gamma * weighted_kl_model.sum(dim=(-2, -1)).mean() / dim_scale
    else:
        model_align_loss = torch.tensor(0.0, device=ce_loss.device)
        gamma = None
        kl_model = None

    # ─── 5. Hyper-prior: λ_h · Σ KL(s_i || h) ───────────────────────
    if lambda_hyper > 0.0:
        # Centroid h = mean of all models (detached: fixed target)
        mu_h = mu_s.mean(dim=1, keepdim=True).detach().expand_as(mu_s)

        if sigma_s is not None:
            # Broadened variance (2×) allows model diversity
            sigma_h = (sigma_s.mean(dim=1, keepdim=True).detach() * 2.0).expand_as(sigma_s)
        else:
            sigma_h = None

        kl_hyper = gaussian_kl_divergence(mu_s, sigma_s, mu_h, sigma_h)
        hyper_prior_loss = lambda_hyper * kl_hyper.mean() / dim_scale
    else:
        hyper_prior_loss = torch.tensor(0.0, device=ce_loss.device)

    # ─── 6. Gauge prior: (α_φ/2) Σ ||φ_i||² ─────────────────────────
    if alpha_phi > 0.0:
        phi_norm_sq = (phi_s ** 2).sum(dim=-1).mean()
        gauge_prior_loss = (alpha_phi / 2.0) * phi_norm_sq / dim_scale
    else:
        gauge_prior_loss = torch.tensor(0.0, device=ce_loss.device)

    # ─── Total loss ───────────────────────────────────────────────────
    total_loss = (
        ce_loss + belief_align_loss + self_consistency_loss
        + model_align_loss + hyper_prior_loss + gauge_prior_loss
    )

    # ─── Metrics ──────────────────────────────────────────────────────
    with torch.no_grad():
        if beta is not None:
            beta_avg = beta[-1].mean(dim=1)
            beta_safe = beta_avg.clamp(min=1e-10)
            attn_entropy = -(beta_safe * beta_safe.log()).sum(dim=-1).mean().item()
            attn_concentration = beta_avg.max(dim=-1)[0].mean().item()
        else:
            attn_entropy = 0.0
            attn_concentration = 0.0

    metrics = {
        'loss/total': total_loss.item(),
        'loss/ce': ce_loss.item(),
        'loss/belief_align': belief_align_loss.item(),
        'loss/self_consistency': self_consistency_loss.item() if alpha > 0 else 0.0,
        'loss/model_coupling': model_align_loss.item() if lambda_gamma > 0 else 0.0,
        'loss/hyper_prior': hyper_prior_loss.item() if lambda_hyper > 0 else 0.0,
        'loss/gauge_prior': gauge_prior_loss.item() if alpha_phi > 0 else 0.0,
        'attention/beta_mean': beta.mean().item() if beta is not None else 0.0,
        'attention/kl_mean': kl.mean().item() if kl is not None else 0.0,
        'attention/entropy': attn_entropy,
        'attention/concentration': attn_concentration,
    }

    if gamma is not None:
        metrics['attention/gamma_mean'] = gamma.mean().item()
        metrics['attention/kl_model_mean'] = kl_model.mean().item()

    return total_loss, metrics


def compute_vfe_loss_from_config(
    model,
    token_ids: torch.Tensor,
    targets: torch.Tensor,
    pad_token_id: int = -100,
    use_obs_in_vfe: bool = False,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Convenience: compute VFE loss using model's config for all hyperparameters.

    Reads all _loss suffixed params from model.config.
    """
    config = model.config
    return compute_vfe_loss(
        model=model,
        token_ids=token_ids,
        targets=targets,
        alpha=config.alpha_loss,
        lambda_beta=config.lambda_beta_loss,
        lambda_gamma=config.lambda_gamma_loss,
        kappa_gamma=config.kappa_gamma_loss,
        lambda_hyper=config.lambda_hyper_loss,
        alpha_phi=config.alpha_phi_loss,
        pad_token_id=pad_token_id,
        use_obs_in_vfe=use_obs_in_vfe,
    )
