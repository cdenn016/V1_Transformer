"""
VFE Closed-Form vs. Iterative Comparison
=========================================

Compares the **precision-weighted closed-form Picard fixed point** against
**iterative gradient descent** for VFE minimization.  Both trajectories start
from identical initial conditions.  The central question: do they converge to
the same q*, and how does convergence rate compare?

Closed-form Picard update (per head h, diagonal covariance):
    mu_i* = [alpha * mu_p / sigma_p  +  lambda * sum_j beta_ij * (Omega_ij mu_j) / sigma_j_t]
           / [alpha / sigma_p          +  lambda * sum_j beta_ij / sigma_j_t]
    sigma_i* = 1 / [alpha / sigma_p  +  lambda * sum_j beta_ij / sigma_j_t]

where sigma_j_t = diag(Omega_ij @ diag(sigma_j) @ Omega_ij^T).

Because beta itself depends on mu, this is a Picard iteration, not a one-shot
solution.  E_lambda_softmax is set to 0 so the closed-form captures all linear
terms exactly.

Usage:
    Edit CFG below, then press Run.

Output:
    scripts/vfe_convergence_output/vfe_closedform_vs_iterative.png
    scripts/vfe_convergence_output/vfe_closedform_vs_iterative.pdf
    scripts/vfe_convergence_output/metrics_iterative.csv
    scripts/vfe_convergence_output/metrics_closedform.csv
"""

# -- Path setup ---------------------------------------------------------------
import sys, os
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import math
import csv
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np

# -- Project imports ----------------------------------------------------------
from math_utils.generators import generate_glK_multihead_generators
from transformer.core.vfe_gradients import (
    compute_vfe_gradients_gpu,
    compute_natural_gradient_gpu,
)
from transformer.core.vfe_utils import (
    retract_spd_diagonal_torch,
    _retract_phi,
)
from transformer.core.attention import compute_attention_weights
from transformer.core.gauge_utils import fused_block_matrix_exp_pairs


# =============================================================================
# CONFIG -- edit here, then press Run.  No CLI arguments.
# =============================================================================

@dataclass
class CFConfig:
    """All parameters for the closed-form vs. iterative comparison.

    E_lambda_softmax is fixed to 0.0 because the precision-weighted closed-form
    captures only the linear (non-softmax-coupled) alignment terms.  Setting it
    to zero makes the gradient-descent path use the same objective so the
    comparison is fair.
    """
    # Experiment
    n_iterations: int = 200
    seed: int = 42

    # Geometry
    embed_dim: int = 20              # K  (total belief dimension)
    irrep_spec: list = field(        # [('fund', n_heads, d_head)]
        default_factory=lambda: [('fund', 2, 10)]
    )
    gauge_group: str = 'GLK'

    # Batch / sequence
    batch_size: int = 64
    seq_len: int = 64

    # VFE hyperparameters
    E_alpha: float = 1.0
    E_lambda_belief: float = 1.0
    # NOTE: softmax coupling disabled so closed-form is exact for all linear terms
    E_lambda_softmax: float = 0.0
    kappa: float = 3.16

    # Step sizes (iterative path)
    E_mu_q_lr: float = 0.05
    E_sigma_q_lr: float = 0.05
    E_phi_lr: float = 0.05

    # Closed-form Picard damping: mu_new = (1-d)*mu + d*mu_star
    cf_damping: float = 0.5

    # Covariance
    diagonal_covariance: bool = True
    sigma_max: float = 5.0
    e_step_sigma_floor: float = 0.1

    # Masking
    mask_self_attention: bool = True
    use_causal_mask: bool = True

    # Gauge
    enforce_orthogonal: bool = False

    # Initialisation
    mu_init_std: float = 1.0
    sigma_init: float = 1.0
    phi_init_std: float = 0.1

    # Output
    output_dir: str = 'scripts/vfe_convergence_output'


CFG = CFConfig()


# =============================================================================
# Helpers
# =============================================================================

def _diagonal_kl(
    mu_q: torch.Tensor,
    sigma_q: torch.Tensor,
    mu_p: torch.Tensor,
    sigma_p: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    r"""KL(q || p) for diagonal Gaussians, summed over K, returning (B, N).

    KL = 0.5 * sum_k [ sigma_q_k/sigma_p_k
                       + (mu_q_k - mu_p_k)^2 / sigma_p_k
                       - 1
                       + log(sigma_p_k / sigma_q_k) ]
    """
    sq = sigma_q.clamp(min=eps)
    sp = sigma_p.clamp(min=eps)
    return 0.5 * (sq / sp + (mu_q - mu_p) ** 2 / sp - 1.0 + torch.log(sp / sq)).sum(-1)


def _attention_entropy(beta: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Mean Shannon entropy of beta rows, in nats.  (B, N, N) -> scalar."""
    return -(beta * (beta + eps).log()).sum(-1).mean()


def _build_causal_mask(N: int, device: torch.device) -> torch.Tensor:
    """Lower-triangular causal mask, shape (1, N, N), dtype bool."""
    return torch.tril(torch.ones(N, N, device=device, dtype=torch.bool)).unsqueeze(0)


def _make_metric_store() -> Dict[str, list]:
    """Return an empty metric store with all tracked keys."""
    return {
        'iteration':    [],
        'F_total':      [],
        'F_self':       [],
        'F_align':      [],
        'grad_mu_norm': [],
        'delta_mu_norm':[],
        'sigma_mean':   [],
        'attn_entropy': [],
    }


# =============================================================================
# Closed-form Picard fixed-point update (one step)
# =============================================================================

def _closedform_picard_step(
    mu_q: torch.Tensor,          # (B, N, K) current beliefs
    sigma_q: torch.Tensor,       # (B, N, K) current diagonal variances
    mu_p: torch.Tensor,          # (B, N, K) prior means
    sigma_p: torch.Tensor,       # (B, N, K) prior diagonal variances (floored)
    phi: torch.Tensor,           # (B, N, n_gen) current gauge frames
    generators: torch.Tensor,    # (n_gen, K, K)
    irrep_dims: List[int],
    alpha: float,
    lambda_belief: float,
    kappa: float,
    damping: float,
    eps: float,
    mask: Optional[torch.Tensor],
    mask_self_attention: bool,
) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
    r"""One Picard iteration of the precision-weighted closed-form update.

    For each head h:
        1.  Compute beta_h from current (mu_q, sigma_q) via compute_attention_weights.
        2.  Build Omega_h = exp_phi_h @ exp_neg_phi_h.
        3.  Compute transported means mu_j_t and transported diagonal sigma sigma_j_t.
        4.  Precision-weighted aggregation:
                prior_prec  = alpha / sigma_p_h
                prior_info  = alpha * mu_p_h / sigma_p_h
                align_prec  = lambda * sum_j beta_ij / sigma_j_t_ij
                align_info  = lambda * sum_j beta_ij * mu_j_t_ij / sigma_j_t_ij
                total_prec  = prior_prec + align_prec
                mu_star_h   = (prior_info + align_info) / total_prec
                sigma_star_h = 1 / total_prec
        5.  Damped update:
                mu_new  = (1 - damping) * mu_q + damping * mu_star
                sigma_new = retract_spd_diagonal_torch to sigma_star

    Args:
        mu_q: Current belief means.
        sigma_q: Current diagonal variances.
        mu_p: Prior means.
        sigma_p: Prior diagonal variances (already floored by caller).
        phi: Current gauge frames.
        generators: Lie algebra generators.
        irrep_dims: Per-head block dimensions.
        alpha: Self-coupling weight.
        lambda_belief: Alignment weight.
        kappa: Attention temperature (bare; compute_attention_weights scales by sqrt(d_h)).
        damping: Picard damping coefficient in (0, 1].
        eps: Numerical floor.
        mask: Causal mask or None.
        mask_self_attention: Whether to mask the diagonal.

    Returns:
        mu_new, sigma_new: Updated (B, N, K) tensors.
        beta_heads: List of per-head beta tensors for downstream metric computation.
    """
    B, N, K = mu_q.shape
    device = mu_q.device

    mu_star_full = torch.zeros_like(mu_q)
    sigma_star_full = torch.zeros_like(sigma_q)
    beta_heads: List[torch.Tensor] = []

    # Precompute block exp pairs (shared across heads)
    block_exp_pairs = fused_block_matrix_exp_pairs(
        phi.detach(), generators, irrep_dims,
        enforce_orthogonal=False,
    )

    block_start = 0
    for h, d_h in enumerate(irrep_dims):
        block_end = block_start + d_h

        mu_h    = mu_q[:, :, block_start:block_end].detach().contiguous()    # (B, N, d_h)
        sigma_h = sigma_q[:, :, block_start:block_end].detach().contiguous() # (B, N, d_h)
        mu_p_h  = mu_p[:, :, block_start:block_end].contiguous()
        sigma_p_h = sigma_p[:, :, block_start:block_end].contiguous()
        gen_h = generators[:, block_start:block_end, block_start:block_end]

        # Step 1: compute beta_h for this head
        beta_h = compute_attention_weights(
            mu_q=mu_h, sigma_q=sigma_h,
            phi=phi.detach(), generators=gen_h,
            kappa=kappa, epsilon=eps, mask=mask,
            return_kl=False,
            diagonal_covariance=True,
            irrep_dims=[d_h],
            cached_block_exp_pairs=[block_exp_pairs[h]],
            mask_self_attention=mask_self_attention,
            gauge_mode='learned',
        )
        beta_heads.append(beta_h.detach())

        # Step 2: transport operators for head h
        exp_phi_h, exp_neg_phi_h = block_exp_pairs[h]  # (B, N, d_h, d_h) each

        # Omega_h[b, i, j] = exp_phi_h[b, i] @ exp_neg_phi_h[b, j]  => (B, N, N, d_h, d_h)
        Omega_h = torch.einsum('bikl,bjlm->bijkm', exp_phi_h, exp_neg_phi_h)

        # Step 3a: transported diagonal sigma: diag(Omega @ diag(sigma_j) @ Omega^T)
        # = sum_l Omega_kl^2 * sigma_j_l    =>  (B, N, N, d_h)
        sigma_j_t = torch.einsum(
            'bijkl,bijkl,bjl->bijk', Omega_h, Omega_h, sigma_h
        ).clamp(min=eps)
        inv_sigma_j_t = 1.0 / sigma_j_t  # (B, N, N, d_h)

        # Step 3b: transported means: Omega_ij @ mu_j  =>  (B, N, N, d_h)
        mu_j_t = torch.einsum('bijkl,bjl->bijk', Omega_h, mu_h)

        # Step 4: precision-weighted aggregation
        inv_sigma_p_h = 1.0 / sigma_p_h.clamp(min=eps)  # (B, N, d_h)
        prior_prec = alpha * inv_sigma_p_h               # (B, N, d_h)
        prior_info = alpha * mu_p_h * inv_sigma_p_h      # (B, N, d_h)

        # beta_h: (B, N, N) -- beta[b, i, j] = attention from i to j
        info_per_pair = mu_j_t * inv_sigma_j_t           # (B, N, N, d_h)

        align_info = lambda_belief * torch.einsum('bij,bijk->bik', beta_h, info_per_pair)  # (B, N, d_h)
        align_prec = lambda_belief * torch.einsum('bij,bijk->bik', beta_h, inv_sigma_j_t)  # (B, N, d_h)

        total_prec    = (prior_prec + align_prec).clamp(min=eps)  # (B, N, d_h)
        mu_star_h     = (prior_info + align_info) / total_prec    # (B, N, d_h)
        sigma_star_h  = 1.0 / total_prec                          # (B, N, d_h)

        mu_star_full[:, :, block_start:block_end]    = mu_star_h
        sigma_star_full[:, :, block_start:block_end] = sigma_star_h

        del Omega_h, sigma_j_t, inv_sigma_j_t, mu_j_t
        block_start = block_end

    # Step 5: damped update
    mu_new = (1.0 - damping) * mu_q.detach() + damping * mu_star_full

    # For sigma: move toward sigma_star via SPD retraction with a single step.
    # The "gradient" in log-space is log(sigma_star / sigma_q), and the retraction
    # exponentiates the whitened tangent: sigma_new = sigma_q * exp(tau * log(sigma_star/sigma_q)).
    # With tau=damping this gives sigma_q^{1-damping} * sigma_star^{damping} (geometric interpolation).
    log_ratio = torch.log(sigma_star_full.clamp(min=eps) / sigma_q.detach().clamp(min=eps))
    sigma_new = retract_spd_diagonal_torch(
        sigma_diag=sigma_q.detach(),
        delta_sigma=log_ratio * sigma_q.detach(),  # whitened tangent * sigma = log_ratio direction
        step_size=damping,
        trust_region=5.0,
        eps=eps,
        sigma_max=CFG.sigma_max,
    )

    return mu_new, sigma_new, beta_heads


# =============================================================================
# Compute VFE scalar (shared by both methods)
# =============================================================================

def _compute_vfe_scalar(
    mu_q: torch.Tensor,
    sigma_q: torch.Tensor,
    mu_p: torch.Tensor,
    sigma_p: torch.Tensor,
    phi: torch.Tensor,
    generators: torch.Tensor,
    irrep_dims: List[int],
    alpha: float,
    lambda_belief: float,
    kappa: float,
    eps: float,
    mask: Optional[torch.Tensor],
    mask_self_attention: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
    """Compute F_self, F_align, F_total, and per-head beta without gradients.

    Returns (F_self, F_align, F_total, beta_heads).
    """
    B, N, K = mu_q.shape

    block_exp_pairs = fused_block_matrix_exp_pairs(
        phi.detach(), generators, irrep_dims, enforce_orthogonal=False,
    )

    F_align_total = torch.tensor(0.0, device=mu_q.device)
    beta_heads: List[torch.Tensor] = []

    block_start = 0
    for h, d_h in enumerate(irrep_dims):
        block_end = block_start + d_h
        mu_h    = mu_q[:, :, block_start:block_end].detach().contiguous()
        sigma_h = sigma_q[:, :, block_start:block_end].detach().contiguous()
        gen_h = generators[:, block_start:block_end, block_start:block_end]

        beta_h, kl_h = compute_attention_weights(
            mu_q=mu_h, sigma_q=sigma_h,
            phi=phi.detach(), generators=gen_h,
            kappa=kappa, epsilon=eps, mask=mask,
            return_kl=True,
            diagonal_covariance=True,
            irrep_dims=[d_h],
            cached_block_exp_pairs=[block_exp_pairs[h]],
            mask_self_attention=mask_self_attention,
            gauge_mode='learned',
        )
        beta_heads.append(beta_h.detach())
        F_align_total = F_align_total + (beta_h.detach() * kl_h.detach()).sum()
        block_start = block_end

    kl_self = _diagonal_kl(mu_q, sigma_q, mu_p, sigma_p)    # (B, N)
    F_self  = alpha * kl_self.mean()
    F_align = lambda_belief * F_align_total / (B * N)
    F_total = F_self + F_align
    return F_self, F_align, F_total, beta_heads


# =============================================================================
# Iterative gradient-descent E-step
# =============================================================================

def run_iterative(
    mu_q_init: torch.Tensor,
    sigma_q_init: torch.Tensor,
    phi_init: torch.Tensor,
    mu_p: torch.Tensor,
    sigma_p: torch.Tensor,
    sigma_p_estep: torch.Tensor,
    generators: torch.Tensor,
    irrep_dims: List[int],
    cfg: CFConfig,
    device: torch.device,
) -> Dict[str, list]:
    """Run iterative gradient descent and return per-iteration metrics."""
    B, N, K = mu_q_init.shape
    n_heads = len(irrep_dims)
    eps = 1e-6
    MAX_NAT_GRAD = 500.0
    mask = _build_causal_mask(N, device) if cfg.use_causal_mask else None

    mu_q    = mu_q_init.clone()
    sigma_q = sigma_q_init.clone()
    phi     = phi_init.clone()

    metrics = _make_metric_store()

    for t in range(cfg.n_iterations):
        mu_prev = mu_q.detach().clone()

        # ---- Per-head beta + VFE gradients ----------------------------------
        grad_mu    = torch.zeros_like(mu_q)
        grad_sigma = torch.zeros_like(sigma_q)
        beta_heads = []

        block_start = 0
        for h, d_h in enumerate(irrep_dims):
            block_end = block_start + d_h
            mu_h    = mu_q[:, :, block_start:block_end].detach().contiguous()
            sigma_h = sigma_q[:, :, block_start:block_end].detach().contiguous()
            mu_p_h  = mu_p[:, :, block_start:block_end].contiguous()
            sp_h    = sigma_p_estep[:, :, block_start:block_end].contiguous()
            gen_h = generators[:, block_start:block_end, block_start:block_end]

            beta_h = compute_attention_weights(
                mu_q=mu_h, sigma_q=sigma_h,
                phi=phi.detach(), generators=gen_h,
                kappa=cfg.kappa, epsilon=eps, mask=mask,
                return_kl=False,
                diagonal_covariance=True,
                irrep_dims=[d_h],
                mask_self_attention=cfg.mask_self_attention,
                gauge_mode='learned',
            )
            grad_mu_h, grad_sigma_h = compute_vfe_gradients_gpu(
                mu_q=mu_h, sigma_q=sigma_h,
                mu_p=mu_p_h, sigma_p=sp_h,
                beta=beta_h, phi=phi.detach(), generators=gen_h,
                alpha=cfg.E_alpha,
                lambda_belief=cfg.E_lambda_belief,
                lambda_softmax=cfg.E_lambda_softmax,  # 0.0
                kappa=cfg.kappa, eps=eps,
                irrep_dims=[d_h],
                enforce_orthogonal=cfg.enforce_orthogonal,
            )
            grad_mu[:, :, block_start:block_end]    = grad_mu_h
            grad_sigma[:, :, block_start:block_end] = grad_sigma_h
            beta_heads.append(beta_h.detach())
            block_start = block_end

        # ---- Natural gradient + clipping ------------------------------------
        nat_grad_mu, nat_grad_sigma = compute_natural_gradient_gpu(
            grad_mu, grad_sigma, sigma_q.detach(), eps=eps,
        )
        ng_mu_norm = torch.linalg.norm(nat_grad_mu, dim=-1, keepdim=True)
        nat_grad_mu = nat_grad_mu * torch.clamp(MAX_NAT_GRAD / (ng_mu_norm + eps), max=1.0)
        ng_sig_norm = torch.linalg.norm(nat_grad_sigma, dim=-1, keepdim=True)
        nat_grad_sigma = nat_grad_sigma * torch.clamp(MAX_NAT_GRAD / (ng_sig_norm + eps), max=1.0)

        # ---- Phi gradient via autograd --------------------------------------
        phi_for_grad = phi.detach().clone().requires_grad_(True)
        phi_loss = torch.tensor(0.0, device=device)
        block_start = 0
        for h, d_h in enumerate(irrep_dims):
            block_end = block_start + d_h
            mu_h    = mu_q[:, :, block_start:block_end].detach()
            sigma_h = sigma_q[:, :, block_start:block_end].detach()
            gen_h = generators[:, block_start:block_end, block_start:block_end]
            beta_phi_h, kl_phi_h = compute_attention_weights(
                mu_q=mu_h, sigma_q=sigma_h,
                phi=phi_for_grad, generators=gen_h,
                kappa=cfg.kappa, epsilon=eps, mask=mask,
                return_kl=True,
                diagonal_covariance=True,
                irrep_dims=[d_h],
                mask_self_attention=cfg.mask_self_attention,
                gauge_mode='learned',
            )
            # lambda_softmax=0, so only direct alignment term
            phi_loss = phi_loss + cfg.E_lambda_belief * (beta_phi_h.detach() * kl_phi_h).sum()
            block_start = block_end

        grad_phi = torch.zeros_like(phi)
        if phi_loss.grad_fn is not None:
            grad_phi = torch.autograd.grad(
                phi_loss, phi_for_grad, create_graph=False, retain_graph=False,
            )[0]
            gp_norm = torch.norm(grad_phi, dim=-1, keepdim=True)
            grad_phi = torch.where(
                gp_norm > 10.0,
                grad_phi * 10.0 / (gp_norm + 1e-6),
                grad_phi,
            )

        # ---- Update mu (whitened trust region) ------------------------------
        delta_mu = -cfg.E_mu_q_lr * nat_grad_mu
        sigma_sqrt = torch.sqrt(sigma_q.detach().clamp(min=eps))
        whitened_norm = torch.linalg.norm(delta_mu / sigma_sqrt, dim=-1, keepdim=True)
        scale = torch.clamp(2.0 / (whitened_norm + eps), max=1.0)
        mu_q = mu_q.detach() + scale * delta_mu

        # ---- Update sigma (SPD retraction) ----------------------------------
        sigma_q = retract_spd_diagonal_torch(
            sigma_diag=sigma_q.detach(),
            delta_sigma=-nat_grad_sigma,
            step_size=1.0,
            trust_region=cfg.E_sigma_q_lr,
            eps=eps,
            sigma_max=cfg.sigma_max,
        )
        # Condition clamping (ratio <= 10)
        s_min = sigma_q.min(dim=-1, keepdim=True).values.clamp(min=eps)
        s_max_v = sigma_q.max(dim=-1, keepdim=True).values
        needs_clamp = (s_max_v / s_min) > 10.0
        geo_mean = sigma_q.log().mean(dim=-1, keepdim=True).exp()
        lower = geo_mean / (10.0 ** 0.5)
        upper = geo_mean * (10.0 ** 0.5)
        sigma_q = torch.where(
            needs_clamp.expand_as(sigma_q),
            sigma_q.clamp(min=lower, max=upper),
            sigma_q,
        )

        # ---- Update phi -----------------------------------------------------
        phi = _retract_phi(
            phi.detach(), -grad_phi, generators,
            step_size=cfg.E_phi_lr,
            gauge_group=cfg.gauge_group,
        )

        # ---- VFE scalar and metrics -----------------------------------------
        F_self, F_align, F_total, beta_heads_metric = _compute_vfe_scalar(
            mu_q, sigma_q, mu_p, sigma_p_estep, phi, generators, irrep_dims,
            cfg.E_alpha, cfg.E_lambda_belief, cfg.kappa, eps, mask, cfg.mask_self_attention,
        )
        beta_avg = sum(beta_heads_metric) / n_heads

        metrics['iteration'].append(t)
        metrics['F_total'].append(F_total.item())
        metrics['F_self'].append(F_self.item())
        metrics['F_align'].append(F_align.item())
        metrics['grad_mu_norm'].append(nat_grad_mu.norm().item())
        metrics['delta_mu_norm'].append((mu_q - mu_prev).norm().item())
        metrics['sigma_mean'].append(sigma_q.mean().item())
        metrics['attn_entropy'].append(_attention_entropy(beta_avg).item())

        if t % 20 == 0 or t == cfg.n_iterations - 1:
            print(f"  [iter] step {t:4d}  F={F_total.item():8.4f}  "
                  f"(self={F_self.item():.4f}  align={F_align.item():.4f})  "
                  f"|d_mu|={metrics['delta_mu_norm'][-1]:.4f}")

    return metrics, mu_q, sigma_q, phi


# =============================================================================
# Closed-form Picard loop
# =============================================================================

def run_closedform(
    mu_q_init: torch.Tensor,
    sigma_q_init: torch.Tensor,
    phi_init: torch.Tensor,
    mu_p: torch.Tensor,
    sigma_p: torch.Tensor,
    sigma_p_estep: torch.Tensor,
    generators: torch.Tensor,
    irrep_dims: List[int],
    cfg: CFConfig,
    device: torch.device,
) -> Tuple[Dict[str, list], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run closed-form Picard iteration and return per-iteration metrics."""
    B, N, K = mu_q_init.shape
    n_heads = len(irrep_dims)
    eps = 1e-6
    mask = _build_causal_mask(N, device) if cfg.use_causal_mask else None

    mu_q    = mu_q_init.clone()
    sigma_q = sigma_q_init.clone()
    phi     = phi_init.clone()

    metrics = _make_metric_store()

    for t in range(cfg.n_iterations):
        mu_prev = mu_q.detach().clone()

        # ---- Closed-form Picard step for mu and sigma -----------------------
        mu_q, sigma_q, beta_heads_cf = _closedform_picard_step(
            mu_q=mu_q,
            sigma_q=sigma_q,
            mu_p=mu_p,
            sigma_p=sigma_p_estep,
            phi=phi,
            generators=generators,
            irrep_dims=irrep_dims,
            alpha=cfg.E_alpha,
            lambda_belief=cfg.E_lambda_belief,
            kappa=cfg.kappa,
            damping=cfg.cf_damping,
            eps=eps,
            mask=mask,
            mask_self_attention=cfg.mask_self_attention,
        )

        # ---- Phi gradient via autograd (same as iterative) ------------------
        phi_for_grad = phi.detach().clone().requires_grad_(True)
        phi_loss = torch.tensor(0.0, device=device)
        block_start = 0
        for h, d_h in enumerate(irrep_dims):
            block_end = block_start + d_h
            mu_h    = mu_q[:, :, block_start:block_end].detach()
            sigma_h = sigma_q[:, :, block_start:block_end].detach()
            gen_h = generators[:, block_start:block_end, block_start:block_end]
            beta_phi_h, kl_phi_h = compute_attention_weights(
                mu_q=mu_h, sigma_q=sigma_h,
                phi=phi_for_grad, generators=gen_h,
                kappa=cfg.kappa, epsilon=eps, mask=mask,
                return_kl=True,
                diagonal_covariance=True,
                irrep_dims=[d_h],
                mask_self_attention=cfg.mask_self_attention,
                gauge_mode='learned',
            )
            phi_loss = phi_loss + cfg.E_lambda_belief * (beta_phi_h.detach() * kl_phi_h).sum()
            block_start = block_end

        if phi_loss.grad_fn is not None:
            grad_phi = torch.autograd.grad(
                phi_loss, phi_for_grad, create_graph=False, retain_graph=False,
            )[0]
            gp_norm = torch.norm(grad_phi, dim=-1, keepdim=True)
            grad_phi = torch.where(
                gp_norm > 10.0,
                grad_phi * 10.0 / (gp_norm + 1e-6),
                grad_phi,
            )
        else:
            grad_phi = torch.zeros_like(phi)

        phi = _retract_phi(
            phi.detach(), -grad_phi, generators,
            step_size=cfg.E_phi_lr,
            gauge_group=cfg.gauge_group,
        )

        # ---- VFE scalar and metrics -----------------------------------------
        F_self, F_align, F_total, beta_heads_metric = _compute_vfe_scalar(
            mu_q, sigma_q, mu_p, sigma_p_estep, phi, generators, irrep_dims,
            cfg.E_alpha, cfg.E_lambda_belief, cfg.kappa, eps, mask, cfg.mask_self_attention,
        )
        beta_avg = sum(beta_heads_metric) / n_heads

        metrics['iteration'].append(t)
        metrics['F_total'].append(F_total.item())
        metrics['F_self'].append(F_self.item())
        metrics['F_align'].append(F_align.item())
        metrics['grad_mu_norm'].append(float('nan'))  # no natural gradient in CF path
        metrics['delta_mu_norm'].append((mu_q - mu_prev).norm().item())
        metrics['sigma_mean'].append(sigma_q.mean().item())
        metrics['attn_entropy'].append(_attention_entropy(beta_avg).item())

        if t % 20 == 0 or t == cfg.n_iterations - 1:
            print(f"  [cf]   step {t:4d}  F={F_total.item():8.4f}  "
                  f"(self={F_self.item():.4f}  align={F_align.item():.4f})  "
                  f"|d_mu|={metrics['delta_mu_norm'][-1]:.4f}")

    return metrics, mu_q, sigma_q, phi


# =============================================================================
# CSV helpers
# =============================================================================

def _save_csv(metrics: Dict[str, list], path: Path) -> None:
    """Write metrics dict to CSV."""
    keys = list(metrics.keys())
    n = len(metrics[keys[0]])
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for i in range(n):
            writer.writerow({k: metrics[k][i] for k in keys})


# =============================================================================
# Plotting
# =============================================================================

def plot_comparison(
    m_iter: Dict[str, list],
    m_cf: Dict[str, list],
    mu_iter_final: torch.Tensor,
    mu_cf_final: torch.Tensor,
    divergence: List[float],
    cfg: CFConfig,
    out_dir: Path,
) -> None:
    """Publication-quality 6-panel figure overlaying iterative vs closed-form."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from transformer.visualization.pub_style import set_pub_style, PUB_COLORS

    set_pub_style()
    C = PUB_COLORS

    t = np.array(m_iter['iteration'])

    fig = plt.figure(figsize=(14, 9))
    gs = gridspec.GridSpec(2, 3, hspace=0.40, wspace=0.38)

    # Shared line styles
    ls_iter = dict(color=C['blue'],   linewidth=1.8, linestyle='-',  label='Iterative (gradient descent)')
    ls_cf   = dict(color=C['red'],    linewidth=1.8, linestyle='--', label='Closed-form (Picard)')

    # -- Panel 1: F_total(t) --------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, m_iter['F_total'], **ls_iter)
    ax1.plot(t, m_cf['F_total'],   **ls_cf)
    ax1.set_xlabel('E-step iteration')
    ax1.set_ylabel('Total free energy $F$')
    ax1.set_title('Total VFE $F(t)$')
    ax1.legend(fontsize=7, loc='best')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # -- Panel 2: F_self(t) ---------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t, m_iter['F_self'], **ls_iter)
    ax2.plot(t, m_cf['F_self'],   **ls_cf)
    ax2.set_xlabel('E-step iteration')
    ax2.set_ylabel(r'$\alpha \cdot \mathrm{KL}(q \| p)$')
    ax2.set_title(r'Self-coupling $F_{\mathrm{self}}(t)$')
    ax2.legend(fontsize=7, loc='best')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # -- Panel 3: F_align(t) --------------------------------------------------
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(t, m_iter['F_align'], **ls_iter)
    ax3.plot(t, m_cf['F_align'],   **ls_cf)
    ax3.set_xlabel('E-step iteration')
    ax3.set_ylabel(r'$\lambda \cdot \sum_{ij} \beta_{ij} \, \mathrm{KL}_{ij}$')
    ax3.set_title(r'Alignment $F_{\mathrm{align}}(t)$')
    ax3.legend(fontsize=7, loc='best')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    # -- Panel 4: |delta_mu|(t) log scale -------------------------------------
    ax4 = fig.add_subplot(gs[1, 0])
    dm_iter = np.array(m_iter['delta_mu_norm'])
    dm_cf   = np.array(m_cf['delta_mu_norm'])
    safe_iter = np.where(dm_iter > 0, dm_iter, np.nan)
    safe_cf   = np.where(dm_cf > 0, dm_cf,   np.nan)
    ax4.semilogy(t, safe_iter, **ls_iter)
    ax4.semilogy(t, safe_cf,   **ls_cf)
    ax4.set_xlabel('E-step iteration')
    ax4.set_ylabel(r'$\|\delta\mu\|$ (log scale)')
    ax4.set_title(r'Belief displacement $\|\delta\mu\|(t)$')
    ax4.legend(fontsize=7, loc='best')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)

    # -- Panel 5: sigma_mean(t) -----------------------------------------------
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(t, m_iter['sigma_mean'], **ls_iter)
    ax5.plot(t, m_cf['sigma_mean'],   **ls_cf)
    ax5.set_xlabel('E-step iteration')
    ax5.set_ylabel(r'$\langle \sigma \rangle$')
    ax5.set_title(r'Mean posterior variance $\langle\sigma\rangle(t)$')
    ax5.legend(fontsize=7, loc='best')
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)

    # -- Panel 6: ||mu_iter - mu_cf||(t) ----------------------------------
    ax6 = fig.add_subplot(gs[1, 2])
    div_arr = np.array(divergence)
    safe_div = np.where(div_arr > 0, div_arr, np.nan)
    ax6.semilogy(t, safe_div, color=C['green'], linewidth=1.8, linestyle='-')
    ax6.set_xlabel('E-step iteration')
    ax6.set_ylabel(r'$\|\mu_{\mathrm{iter}} - \mu_{\mathrm{cf}}\|$ (log scale)')
    ax6.set_title('Solution divergence')
    ax6.spines['top'].set_visible(False)
    ax6.spines['right'].set_visible(False)

    # Overall title
    _, n_heads, d_head = cfg.irrep_spec[0]
    fig.suptitle(
        f'Iterative vs. Closed-form Picard  '
        f'(K={cfg.embed_dim}, {n_heads} heads x {d_head}, '
        f'B={cfg.batch_size}, N={cfg.seq_len}, '
        r'$\alpha$=' + f'{cfg.E_alpha}, '
        r'$\lambda_b$=' + f'{cfg.E_lambda_belief}, '
        r'$\kappa$=' + f'{cfg.kappa})',
        fontsize=10, y=1.01,
    )

    fig.tight_layout()

    png_path = out_dir / 'vfe_closedform_vs_iterative.png'
    pdf_path = out_dir / 'vfe_closedform_vs_iterative.pdf'
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)
    print(f"\nFigure saved: {png_path}")
    print(f"Figure saved: {pdf_path}")


# =============================================================================
# Entry point
# =============================================================================

def main() -> None:
    cfg = CFG
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B, N, K = cfg.batch_size, cfg.seq_len, cfg.embed_dim
    _, n_heads, d_head = cfg.irrep_spec[0]
    irrep_dims = [d_head] * n_heads

    print("=" * 68)
    print("VFE Closed-Form vs. Iterative Comparison")
    print("=" * 68)
    print(f"Device        : {device}")
    print(f"Beliefs       : B={B}, N={N}, K={K}  ({n_heads} heads x {d_head})")
    print(f"VFE params    : alpha={cfg.E_alpha}, lambda_b={cfg.E_lambda_belief}, "
          f"lambda_s={cfg.E_lambda_softmax}  (0=disabled for closed-form validity)")
    print(f"               kappa={cfg.kappa}")
    print(f"Step sizes    : mu={cfg.E_mu_q_lr}, sigma={cfg.E_sigma_q_lr}, phi={cfg.E_phi_lr}")
    print(f"CF damping    : {cfg.cf_damping}")
    print(f"Iterations    : {cfg.n_iterations}")
    print()

    # -- Build generators (same as GaugeTransformerLM._build_generators) ------
    generators_np = generate_glK_multihead_generators(K, n_heads)
    generators = torch.from_numpy(generators_np).float().to(device)
    n_gen = generators.shape[0]
    print(f"Generators    : {n_gen} x ({K},{K})  [GL({d_head})^{n_heads}]")

    # -- Shared initial conditions (IDENTICAL for both methods) ---------------
    mu_p    = torch.randn(B, N, K, device=device) * cfg.mu_init_std
    sigma_p = torch.ones(B, N, K, device=device) * cfg.sigma_init

    mu_q_0    = mu_p.clone() + torch.randn(B, N, K, device=device) * 0.3
    sigma_q_0 = sigma_p.clone() * (
        1.0 + 0.2 * torch.randn(B, N, K, device=device)
    ).clamp(min=0.1)
    phi_0 = torch.randn(B, N, n_gen, device=device) * cfg.phi_init_std

    sigma_p_estep = sigma_p.clamp(min=cfg.e_step_sigma_floor)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Run iterative
    # =========================================================================
    print("\n--- Iterative gradient-descent E-step ---")
    m_iter, mu_iter_final, sigma_iter_final, phi_iter_final = run_iterative(
        mu_q_init=mu_q_0.clone(),
        sigma_q_init=sigma_q_0.clone(),
        phi_init=phi_0.clone(),
        mu_p=mu_p, sigma_p=sigma_p,
        sigma_p_estep=sigma_p_estep,
        generators=generators, irrep_dims=irrep_dims,
        cfg=cfg, device=device,
    )

    # =========================================================================
    # Run closed-form Picard
    # =========================================================================
    print("\n--- Closed-form Picard iteration ---")
    m_cf, mu_cf_final, sigma_cf_final, phi_cf_final = run_closedform(
        mu_q_init=mu_q_0.clone(),
        sigma_q_init=sigma_q_0.clone(),
        phi_init=phi_0.clone(),
        mu_p=mu_p, sigma_p=sigma_p,
        sigma_p_estep=sigma_p_estep,
        generators=generators, irrep_dims=irrep_dims,
        cfg=cfg, device=device,
    )

    # =========================================================================
    # Compute per-step divergence ||mu_iter(t) - mu_cf(t)||
    # We cannot go back in time, so we re-run both at each step to track sync.
    # Instead: we report the divergence of the FINAL solutions (scalar), and
    # compute step-wise divergence by re-running both trajectories together
    # with per-step snapshots.
    # =========================================================================
    print("\n--- Computing step-wise divergence (joint replay) ---")
    divergence: List[float] = _compute_stepwise_divergence(
        mu_q_0=mu_q_0.clone(),
        sigma_q_0=sigma_q_0.clone(),
        phi_0=phi_0.clone(),
        mu_p=mu_p, sigma_p=sigma_p,
        sigma_p_estep=sigma_p_estep,
        generators=generators, irrep_dims=irrep_dims,
        cfg=cfg, device=device,
    )

    # Final solution divergence
    final_div = (mu_iter_final - mu_cf_final).norm().item()
    print(f"\nFinal ||mu_iter - mu_cf|| = {final_div:.6f}")
    print(f"Final F_iter = {m_iter['F_total'][-1]:.6f}")
    print(f"Final F_cf   = {m_cf['F_total'][-1]:.6f}")

    # =========================================================================
    # Save CSVs
    # =========================================================================
    csv_iter = out_dir / 'metrics_iterative.csv'
    csv_cf   = out_dir / 'metrics_closedform.csv'
    _save_csv(m_iter, csv_iter)
    _save_csv(m_cf,   csv_cf)
    print(f"\nCSV saved: {csv_iter}")
    print(f"CSV saved: {csv_cf}")

    # =========================================================================
    # Plot
    # =========================================================================
    plot_comparison(
        m_iter=m_iter, m_cf=m_cf,
        mu_iter_final=mu_iter_final, mu_cf_final=mu_cf_final,
        divergence=divergence,
        cfg=cfg, out_dir=out_dir,
    )


def _compute_stepwise_divergence(
    mu_q_0: torch.Tensor,
    sigma_q_0: torch.Tensor,
    phi_0: torch.Tensor,
    mu_p: torch.Tensor,
    sigma_p: torch.Tensor,
    sigma_p_estep: torch.Tensor,
    generators: torch.Tensor,
    irrep_dims: List[int],
    cfg: CFConfig,
    device: torch.device,
) -> List[float]:
    r"""Re-run both methods in lockstep and record ||mu_iter(t) - mu_cf(t)||.

    Both trajectories advance one step per iteration.  The divergence vector
    tracks whether they stay in sync or drift apart.

    Args:
        All arguments identical to run_iterative / run_closedform.

    Returns:
        List of floats, length n_iterations.
    """
    B, N, K = mu_q_0.shape
    n_heads = len(irrep_dims)
    eps = 1e-6
    MAX_NAT_GRAD = 500.0
    mask = _build_causal_mask(N, device) if cfg.use_causal_mask else None

    # ---- Iterative state ----
    mu_i    = mu_q_0.clone()
    sigma_i = sigma_q_0.clone()
    phi_i   = phi_0.clone()

    # ---- Closed-form state ----
    mu_c    = mu_q_0.clone()
    sigma_c = sigma_q_0.clone()
    phi_c   = phi_0.clone()

    divergence: List[float] = []

    for t in range(cfg.n_iterations):
        # ----------- One iterative step --------------------------------------
        grad_mu_i = torch.zeros_like(mu_i)
        grad_sigma_i = torch.zeros_like(sigma_i)
        block_start = 0
        for h, d_h in enumerate(irrep_dims):
            block_end = block_start + d_h
            mu_h    = mu_i[:, :, block_start:block_end].detach().contiguous()
            sigma_h = sigma_i[:, :, block_start:block_end].detach().contiguous()
            mu_p_h  = mu_p[:, :, block_start:block_end].contiguous()
            sp_h    = sigma_p_estep[:, :, block_start:block_end].contiguous()
            gen_h   = generators[:, block_start:block_end, block_start:block_end]
            beta_h = compute_attention_weights(
                mu_q=mu_h, sigma_q=sigma_h,
                phi=phi_i.detach(), generators=gen_h,
                kappa=cfg.kappa, epsilon=eps, mask=mask,
                return_kl=False, diagonal_covariance=True,
                irrep_dims=[d_h], mask_self_attention=cfg.mask_self_attention,
                gauge_mode='learned',
            )
            gm_h, gs_h = compute_vfe_gradients_gpu(
                mu_q=mu_h, sigma_q=sigma_h,
                mu_p=mu_p_h, sigma_p=sp_h,
                beta=beta_h, phi=phi_i.detach(), generators=gen_h,
                alpha=cfg.E_alpha, lambda_belief=cfg.E_lambda_belief,
                lambda_softmax=0.0, kappa=cfg.kappa, eps=eps,
                irrep_dims=[d_h], enforce_orthogonal=cfg.enforce_orthogonal,
            )
            grad_mu_i[:, :, block_start:block_end]    = gm_h
            grad_sigma_i[:, :, block_start:block_end] = gs_h
            block_start = block_end

        nat_mu_i, nat_sig_i = compute_natural_gradient_gpu(
            grad_mu_i, grad_sigma_i, sigma_i.detach(), eps=eps,
        )
        nm = torch.linalg.norm(nat_mu_i, dim=-1, keepdim=True)
        nat_mu_i = nat_mu_i * torch.clamp(MAX_NAT_GRAD / (nm + eps), max=1.0)
        ns = torch.linalg.norm(nat_sig_i, dim=-1, keepdim=True)
        nat_sig_i = nat_sig_i * torch.clamp(MAX_NAT_GRAD / (ns + eps), max=1.0)

        # phi gradient for iterative
        phi_fg = phi_i.detach().clone().requires_grad_(True)
        pl = torch.tensor(0.0, device=device)
        block_start = 0
        for h, d_h in enumerate(irrep_dims):
            block_end = block_start + d_h
            gen_h = generators[:, block_start:block_end, block_start:block_end]
            bph, kph = compute_attention_weights(
                mu_q=mu_i[:, :, block_start:block_end].detach(),
                sigma_q=sigma_i[:, :, block_start:block_end].detach(),
                phi=phi_fg, generators=gen_h,
                kappa=cfg.kappa, epsilon=eps, mask=mask,
                return_kl=True, diagonal_covariance=True,
                irrep_dims=[d_h], mask_self_attention=cfg.mask_self_attention,
                gauge_mode='learned',
            )
            pl = pl + cfg.E_lambda_belief * (bph.detach() * kph).sum()
            block_start = block_end

        gp_i = torch.zeros_like(phi_i)
        if pl.grad_fn is not None:
            gp_i = torch.autograd.grad(pl, phi_fg, create_graph=False)[0]
            gpn = torch.norm(gp_i, dim=-1, keepdim=True)
            gp_i = torch.where(gpn > 10.0, gp_i * 10.0 / (gpn + 1e-6), gp_i)

        delta_mu_i = -cfg.E_mu_q_lr * nat_mu_i
        sw = torch.sqrt(sigma_i.detach().clamp(min=eps))
        wn = torch.linalg.norm(delta_mu_i / sw, dim=-1, keepdim=True)
        sc = torch.clamp(2.0 / (wn + eps), max=1.0)
        mu_i = mu_i.detach() + sc * delta_mu_i
        sigma_i = retract_spd_diagonal_torch(
            sigma_i.detach(), -nat_sig_i, step_size=1.0,
            trust_region=cfg.E_sigma_q_lr, eps=eps, sigma_max=cfg.sigma_max,
        )
        phi_i = _retract_phi(
            phi_i.detach(), -gp_i, generators,
            step_size=cfg.E_phi_lr, gauge_group=cfg.gauge_group,
        )

        # ----------- One closed-form step ------------------------------------
        mu_c, sigma_c, _ = _closedform_picard_step(
            mu_q=mu_c, sigma_q=sigma_c,
            mu_p=mu_p, sigma_p=sigma_p_estep, phi=phi_c,
            generators=generators, irrep_dims=irrep_dims,
            alpha=cfg.E_alpha, lambda_belief=cfg.E_lambda_belief,
            kappa=cfg.kappa, damping=cfg.cf_damping,
            eps=eps, mask=mask, mask_self_attention=cfg.mask_self_attention,
        )

        phi_fg_c = phi_c.detach().clone().requires_grad_(True)
        pl_c = torch.tensor(0.0, device=device)
        block_start = 0
        for h, d_h in enumerate(irrep_dims):
            block_end = block_start + d_h
            gen_h = generators[:, block_start:block_end, block_start:block_end]
            bph_c, kph_c = compute_attention_weights(
                mu_q=mu_c[:, :, block_start:block_end].detach(),
                sigma_q=sigma_c[:, :, block_start:block_end].detach(),
                phi=phi_fg_c, generators=gen_h,
                kappa=cfg.kappa, epsilon=eps, mask=mask,
                return_kl=True, diagonal_covariance=True,
                irrep_dims=[d_h], mask_self_attention=cfg.mask_self_attention,
                gauge_mode='learned',
            )
            pl_c = pl_c + cfg.E_lambda_belief * (bph_c.detach() * kph_c).sum()
            block_start = block_end

        gp_c = torch.zeros_like(phi_c)
        if pl_c.grad_fn is not None:
            gp_c = torch.autograd.grad(pl_c, phi_fg_c, create_graph=False)[0]
            gpn_c = torch.norm(gp_c, dim=-1, keepdim=True)
            gp_c = torch.where(gpn_c > 10.0, gp_c * 10.0 / (gpn_c + 1e-6), gp_c)

        phi_c = _retract_phi(
            phi_c.detach(), -gp_c, generators,
            step_size=cfg.E_phi_lr, gauge_group=cfg.gauge_group,
        )

        # ----------- Record divergence at this step --------------------------
        div = (mu_i - mu_c).norm().item()
        divergence.append(div)

        if t % 40 == 0 or t == cfg.n_iterations - 1:
            print(f"  [div]  step {t:4d}  ||mu_iter - mu_cf|| = {div:.6f}")

    return divergence


if __name__ == '__main__':
    main()
