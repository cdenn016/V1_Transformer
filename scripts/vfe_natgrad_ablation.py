"""
VFE Natural Gradient vs Euclidean Gradient Ablation
====================================================

Compares VFE E-step convergence between two gradient modes:

    1. Natural gradient: Fisher-metric projection via compute_natural_gradient_gpu,
       yielding nat_grad_mu = sigma_q * grad_mu and
       nat_grad_sigma = 2 * sigma_q^2 * grad_sigma (diagonal approximation).

    2. Euclidean gradient: raw grad_mu and grad_sigma, bypassing the Fisher
       projection entirely.

Both modes run from identical initial conditions (same seed) and share the
same trust regions, clipping, SPD retraction, condition clamping, and phi
autograd retraction. The only difference is whether the Fisher metric is
applied to (grad_mu, grad_sigma) before the update step.

Uses the same multihead VFE path as VariationalFFNDynamic: per-head beta via
compute_attention_weights, per-head gradients via compute_vfe_gradients_gpu,
then aggregation across heads.

Usage:
    Edit the NatGradConfig dataclass below, then press Run.

Output:
    scripts/vfe_convergence_output/vfe_natgrad_ablation.png
    scripts/vfe_convergence_output/vfe_natgrad_ablation.pdf
    scripts/vfe_convergence_output/vfe_natgrad_ablation_natural.csv
    scripts/vfe_convergence_output/vfe_natgrad_ablation_euclidean.csv
"""

# -- Path setup ---------------------------------------------------------------
import sys
import os

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import math
import csv
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple

import torch
import numpy as np

# -- Project imports (same functions the pipeline uses) -----------------------
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


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class NatGradConfig:
    """All parameters for the natural-gradient vs Euclidean ablation.

    Defaults mirror the EM_CONFIG used in train_publication.py so that results
    are directly comparable to the production pipeline.
    """
    # --- Experiment ---
    n_iterations: int = 500
    seed: int = 42

    # --- Geometry ---
    embed_dim: int = 20              # K  (total belief dimension)
    irrep_spec: list = field(        # [('fund', n_heads, d_head)]
        default_factory=lambda: [('fund', 2, 10)]
    )
    gauge_group: str = 'GLK'

    # --- Batch / sequence ---
    batch_size: int = 64
    seq_len: int = 64

    # --- VFE hyperparameters (E-step) ---
    E_alpha: float = 1.0
    E_lambda_belief: float = 1.0
    E_lambda_softmax: float = 5.0
    kappa: float = 3.16

    E_mu_q_lr: float = 0.05
    E_sigma_q_lr: float = 0.05
    E_phi_lr: float = 0.05

    # --- Covariance ---
    diagonal_covariance: bool = True
    sigma_max: float = 5.0
    e_step_sigma_floor: float = 0.1

    # --- Masking ---
    mask_self_attention: bool = True
    use_causal_mask: bool = True

    # --- Gauge ---
    enforce_orthogonal: bool = False

    # --- Initialisation ---
    mu_init_std: float = 1.0
    sigma_init: float = 1.0
    phi_init_std: float = 0.1

    # --- Output ---
    output_dir: str = 'scripts/vfe_convergence_output'


CONFIG = NatGradConfig()


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
    r"""Per-position KL(q || p) for diagonal Gaussians.

    .. math::

        \mathrm{KL}(q \| p) = \frac{1}{2} \sum_k \left(
            \frac{\sigma_q^k}{\sigma_p^k}
            + \frac{(\mu_q^k - \mu_p^k)^2}{\sigma_p^k}
            - 1
            + \log \frac{\sigma_p^k}{\sigma_q^k}
        \right)

    Args:
        mu_q: Belief means (B, N, K).
        sigma_q: Diagonal variances (B, N, K).
        mu_p: Prior means (B, N, K).
        sigma_p: Prior variances (B, N, K).
        eps: Numerical floor applied to variances.

    Returns:
        kl: Per-position KL values (B, N).
    """
    sq = sigma_q.clamp(min=eps)
    sp = sigma_p.clamp(min=eps)
    return 0.5 * (sq / sp + (mu_q - mu_p) ** 2 / sp - 1.0 + torch.log(sp / sq)).sum(-1)


def _attention_entropy(beta: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Mean entropy of attention distribution beta in nats.

    Args:
        beta: Attention weights (B, N, N).
        eps: Small constant for log stability.

    Returns:
        Scalar entropy value.
    """
    return -(beta * (beta + eps).log()).sum(-1).mean()


def _build_causal_mask(N: int, device: torch.device) -> torch.Tensor:
    """Lower-triangular causal mask, shape (1, N, N), broadcastable over batch."""
    return torch.tril(torch.ones(N, N, device=device, dtype=torch.bool)).unsqueeze(0)


def _clip_norm(
    t: torch.Tensor,
    max_norm: float,
    dim: int = -1,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Per-vector norm clipping along ``dim``.

    Args:
        t: Input tensor.
        max_norm: Maximum allowed per-vector L2 norm.
        dim: Dimension over which norms are computed.
        eps: Numerical stability constant.

    Returns:
        Clipped tensor with same shape as input.
    """
    norms = torch.linalg.norm(t, dim=dim, keepdim=True)
    scale = torch.clamp(max_norm / (norms + eps), max=1.0)
    return t * scale


def _init_beliefs(
    cfg: NatGradConfig,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Initialise mu_q, sigma_q, phi from the fixed seed, plus frozen priors.

    The same torch.manual_seed call is used by both gradient modes so that
    they start from exactly the same (mu_q, sigma_q, phi, mu_p, sigma_p).

    Args:
        cfg: Experiment configuration.
        device: Torch device.

    Returns:
        Tuple of (mu_q, sigma_q, phi, mu_p, sigma_p) all on ``device``.
    """
    torch.manual_seed(cfg.seed)
    B, N, K = cfg.batch_size, cfg.seq_len, cfg.embed_dim

    mu_p = torch.randn(B, N, K, device=device) * cfg.mu_init_std
    sigma_p = torch.ones(B, N, K, device=device) * cfg.sigma_init

    mu_q = mu_p.clone() + torch.randn(B, N, K, device=device) * 0.3
    sigma_q = sigma_p.clone() * (
        1.0 + 0.2 * torch.randn(B, N, K, device=device)
    ).clamp(min=0.1)

    _, n_heads, _ = cfg.irrep_spec[0]
    n_gen = n_heads * (cfg.embed_dim // n_heads) ** 2
    phi = torch.randn(B, N, n_gen, device=device) * cfg.phi_init_std

    return mu_q, sigma_q, phi, mu_p, sigma_p


# =============================================================================
# Core: single-mode convergence loop
# =============================================================================

def run_mode(
    mode: Literal['natural', 'euclidean'],
    cfg: NatGradConfig,
    generators: torch.Tensor,
    mu_q_init: torch.Tensor,
    sigma_q_init: torch.Tensor,
    phi_init: torch.Tensor,
    mu_p: torch.Tensor,
    sigma_p: torch.Tensor,
) -> Dict[str, list]:
    """Run VFE E-step loop for a given gradient mode.

    For ``mode='natural'`` the (grad_mu, grad_sigma) pair produced by
    compute_vfe_gradients_gpu is passed through compute_natural_gradient_gpu
    (Fisher metric multiplication) before the update.  For
    ``mode='euclidean'`` the raw gradients are used directly, skipping the
    Fisher projection.

    All other update machinery (norm clipping, whitened mu trust region, SPD
    sigma retraction, condition clamping, phi autograd retraction) is
    identical between the two modes.

    Args:
        mode: 'natural' or 'euclidean'.
        cfg: Experiment configuration.
        generators: Lie algebra generators (n_gen, K, K) on device.
        mu_q_init: Initial belief means (B, N, K), detached.
        sigma_q_init: Initial diagonal variances (B, N, K), detached.
        phi_init: Initial gauge frames (B, N, n_gen), detached.
        mu_p: Prior means (B, N, K), frozen.
        sigma_p: Prior variances (B, N, K), frozen.

    Returns:
        Dict mapping metric names to per-iteration lists.
    """
    device = generators.device
    B, N, K = mu_q_init.shape
    _, n_heads, d_head = cfg.irrep_spec[0]
    irrep_dims = [d_head] * n_heads

    mu_q = mu_q_init.clone()
    sigma_q = sigma_q_init.clone()
    phi = phi_init.clone()

    sigma_p_estep = sigma_p.clamp(min=cfg.e_step_sigma_floor)
    mask = _build_causal_mask(N, device) if cfg.use_causal_mask else None

    eps = 1e-6
    MAX_NAT_GRAD = 500.0

    metrics: Dict[str, list] = {
        'iteration':       [],
        'F_total':         [],
        'F_self':          [],
        'F_align':         [],
        'grad_mu_norm':    [],
        'delta_mu_norm':   [],
        'attn_entropy':    [],
        'sigma_mean':      [],
    }

    for t in range(cfg.n_iterations):
        mu_prev = mu_q.detach().clone()

        # ====================================================================
        # STEP 1: Per-head beta and VFE gradients
        # Mirrors the multihead VFE path in VariationalFFNDynamic.
        # ====================================================================
        grad_mu = torch.zeros_like(mu_q)
        grad_sigma = torch.zeros_like(sigma_q)
        beta_heads: List[torch.Tensor] = []
        F_align_total = torch.tensor(0.0, device=device)

        block_start = 0
        for h, d_h in enumerate(irrep_dims):
            block_end = block_start + d_h

            mu_h = mu_q[:, :, block_start:block_end].detach().contiguous()
            sigma_h = sigma_q[:, :, block_start:block_end].detach().contiguous()
            mu_p_h = mu_p[:, :, block_start:block_end].contiguous()
            sigma_p_h = sigma_p_estep[:, :, block_start:block_end].contiguous()
            # Slice both the generator index and the spatial block dimensions.
            # generators is block-diagonal: head h generators are non-zero only
            # in rows/cols [block_start:block_end] of the K x K space.
            gen_h = generators[:, block_start:block_end, block_start:block_end]

            # Per-head attention weights (no KL return for gradient step)
            beta_h = compute_attention_weights(
                mu_q=mu_h,
                sigma_q=sigma_h,
                phi=phi.detach(),
                generators=gen_h,
                kappa=cfg.kappa,
                epsilon=eps,
                mask=mask,
                return_kl=False,
                diagonal_covariance=cfg.diagonal_covariance,
                irrep_dims=[d_h],
                mask_self_attention=cfg.mask_self_attention,
                gauge_mode='learned',
            )

            # Per-head VFE gradients
            grad_mu_h, grad_sigma_h = compute_vfe_gradients_gpu(
                mu_q=mu_h,
                sigma_q=sigma_h,
                mu_p=mu_p_h,
                sigma_p=sigma_p_h,
                beta=beta_h,
                phi=phi.detach(),
                generators=gen_h,
                alpha=cfg.E_alpha,
                lambda_belief=cfg.E_lambda_belief,
                lambda_softmax=cfg.E_lambda_softmax,
                kappa=cfg.kappa,
                eps=eps,
                irrep_dims=[d_h],
                enforce_orthogonal=cfg.enforce_orthogonal,
            )

            grad_mu[:, :, block_start:block_end] = grad_mu_h
            grad_sigma[:, :, block_start:block_end] = grad_sigma_h
            beta_heads.append(beta_h.detach())

            # Alignment KL for the F_align scalar (separate call with return_kl=True)
            beta_kl_h, kl_h = compute_attention_weights(
                mu_q=mu_h,
                sigma_q=sigma_h,
                phi=phi.detach(),
                generators=gen_h,
                kappa=cfg.kappa,
                epsilon=eps,
                mask=mask,
                return_kl=True,
                diagonal_covariance=cfg.diagonal_covariance,
                irrep_dims=[d_h],
                mask_self_attention=cfg.mask_self_attention,
                gauge_mode='learned',
            )
            F_align_total = F_align_total + (beta_kl_h.detach() * kl_h.detach()).sum()

            block_start = block_end

        # Average beta across heads for entropy metric
        beta_avg = sum(beta_heads) / n_heads

        # ====================================================================
        # STEP 2: VFE scalar
        # F_self = alpha * mean_pos KL(q || p)
        # F_align = lambda_b * sum(beta * kl) / (B * N)
        # ====================================================================
        kl_self_per_pos = _diagonal_kl(mu_q, sigma_q, mu_p, sigma_p_estep)
        F_self = cfg.E_alpha * kl_self_per_pos.mean()
        F_align = cfg.E_lambda_belief * F_align_total / (B * N)
        F_total = F_self + F_align

        # ====================================================================
        # STEP 3: Apply gradient mode (the only difference between modes)
        # ====================================================================
        if mode == 'natural':
            # Fisher-metric natural gradient: sigma * grad_mu, 2*sigma^2 * grad_sigma
            eff_grad_mu, eff_grad_sigma = compute_natural_gradient_gpu(
                grad_mu, grad_sigma, sigma_q.detach(), eps=eps,
            )
        else:
            # Euclidean: pass raw gradients through unchanged
            eff_grad_mu = grad_mu
            eff_grad_sigma = grad_sigma

        # ====================================================================
        # STEP 4: Norm clipping (max 500, per-vector along last dim)
        # ====================================================================
        eff_grad_mu = _clip_norm(eff_grad_mu, MAX_NAT_GRAD, dim=-1, eps=eps)
        eff_grad_sigma = _clip_norm(eff_grad_sigma, MAX_NAT_GRAD, dim=-1, eps=eps)

        # ====================================================================
        # STEP 5: Phi gradient via autograd on alignment loss
        # ====================================================================
        phi_for_grad = phi.detach().clone().requires_grad_(True)
        alignment_loss_phi = torch.tensor(0.0, device=device)

        block_start = 0
        for h, d_h in enumerate(irrep_dims):
            block_end = block_start + d_h
            mu_h = mu_q[:, :, block_start:block_end].detach()
            sigma_h = sigma_q[:, :, block_start:block_end].detach()
            gen_h = generators[:, block_start:block_end, block_start:block_end]

            beta_phi_h, kl_phi_h = compute_attention_weights(
                mu_q=mu_h,
                sigma_q=sigma_h,
                phi=phi_for_grad,
                generators=gen_h,
                kappa=cfg.kappa,
                epsilon=eps,
                mask=mask,
                return_kl=True,
                diagonal_covariance=cfg.diagonal_covariance,
                irrep_dims=[d_h],
                mask_self_attention=cfg.mask_self_attention,
                gauge_mode='learned',
            )
            alignment_loss_phi = alignment_loss_phi + (
                cfg.E_lambda_belief * (beta_phi_h.detach() * kl_phi_h).sum()
                + cfg.E_lambda_softmax * (beta_phi_h * kl_phi_h.detach()).sum()
            )
            block_start = block_end

        grad_phi = torch.zeros_like(phi)
        if alignment_loss_phi.grad_fn is not None:
            grad_phi = torch.autograd.grad(
                alignment_loss_phi,
                phi_for_grad,
                create_graph=False,
                retain_graph=False,
            )[0]
            gp_norm = torch.norm(grad_phi, dim=-1, keepdim=True)
            grad_phi = torch.where(
                gp_norm > 10.0,
                grad_phi * 10.0 / (gp_norm + eps),
                grad_phi,
            )

        # ====================================================================
        # STEP 6: Belief updates (identical for both modes)
        # ====================================================================
        effective_lr = cfg.E_mu_q_lr

        # 6a. Mu: whitened trust region (radius 2.0)
        delta_mu = -effective_lr * eff_grad_mu
        sigma_sqrt = torch.sqrt(sigma_q.detach().clamp(min=eps))
        whitened_delta = delta_mu / sigma_sqrt
        whitened_norm = torch.linalg.norm(whitened_delta, dim=-1, keepdim=True)
        scale = torch.clamp(2.0 / (whitened_norm + eps), max=1.0)
        mu_q = mu_q.detach() + scale * delta_mu

        # 6b. Sigma: SPD retraction (step_size=1.0, trust_region=sigma_lr)
        sigma_q = retract_spd_diagonal_torch(
            sigma_diag=sigma_q.detach(),
            delta_sigma=-eff_grad_sigma,
            step_size=1.0,
            trust_region=cfg.E_sigma_q_lr,
            eps=eps,
            sigma_max=cfg.sigma_max,
        )

        # 6c. Sigma condition clamping (max ratio 10)
        max_cond = 10.0
        s_min = sigma_q.min(dim=-1, keepdim=True).values.clamp(min=eps)
        s_max_v = sigma_q.max(dim=-1, keepdim=True).values
        needs_clamp = (s_max_v / s_min) > max_cond
        geo_mean = sigma_q.log().mean(dim=-1, keepdim=True).exp()
        lower = geo_mean / (max_cond ** 0.5)
        upper = geo_mean * (max_cond ** 0.5)
        sigma_clamped = sigma_q.clamp(min=lower, max=upper)
        sigma_q = torch.where(needs_clamp.expand_as(sigma_q), sigma_clamped, sigma_q)

        # 6d. Phi retraction
        phi = _retract_phi(
            phi.detach(),
            -grad_phi,
            generators,
            step_size=cfg.E_phi_lr,
            gauge_group=cfg.gauge_group,
        )

        # ====================================================================
        # STEP 7: Record metrics
        # ====================================================================
        metrics['iteration'].append(t)
        metrics['F_total'].append(F_total.item())
        metrics['F_self'].append(F_self.item())
        metrics['F_align'].append(F_align.item())
        metrics['grad_mu_norm'].append(eff_grad_mu.norm().item())
        metrics['delta_mu_norm'].append((mu_q - mu_prev).norm().item())
        metrics['attn_entropy'].append(_attention_entropy(beta_avg).item())
        metrics['sigma_mean'].append(sigma_q.mean().item())

        if t % 50 == 0 or t == cfg.n_iterations - 1:
            print(
                f"  [{mode:10s}] step {t:4d}  F={F_total.item():8.4f}  "
                f"(self={F_self.item():.4f}  align={F_align.item():.4f})  "
                f"|eff_grad_mu|={eff_grad_mu.norm().item():.4f}  "
                f"|d_mu|={metrics['delta_mu_norm'][-1]:.4f}  "
                f"H(beta)={metrics['attn_entropy'][-1]:.3f}"
            )

    return metrics


# =============================================================================
# Plotting
# =============================================================================

def plot_ablation(
    metrics_nat: Dict[str, list],
    metrics_euc: Dict[str, list],
    cfg: NatGradConfig,
) -> "plt.Figure":  # type: ignore[name-defined]
    """Publication-quality 6-panel overlay figure.

    Each panel overlays natural gradient (solid blue) vs Euclidean gradient
    (dashed red) on the same axes.

    Panels:
        1. F_total(t)
        2. F_self(t)
        3. F_align(t)
        4. |eff_grad_mu|(t)  -- log scale
        5. |delta_mu|(t)     -- log scale
        6. H(beta)(t)        -- attention entropy

    Args:
        metrics_nat: Per-iteration metrics for the natural gradient mode.
        metrics_euc: Per-iteration metrics for the Euclidean gradient mode.
        cfg: Experiment configuration (for annotations).

    Returns:
        Matplotlib Figure object.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from transformer.visualization.pub_style import set_pub_style, PUB_COLORS

    set_pub_style()

    C = PUB_COLORS
    NAT_COLOR = C['blue']
    EUC_COLOR = C['red']
    LW_SOLID = 1.8
    LW_DASH = 1.6
    ALPHA_EUC = 0.85

    t_nat = np.array(metrics_nat['iteration'])
    t_euc = np.array(metrics_euc['iteration'])

    fig = plt.figure(figsize=(14, 9))
    gs = gridspec.GridSpec(2, 3, hspace=0.38, wspace=0.35)

    def _overlay(ax, key, ylabel, title, log_scale=False):
        """Plot both modes on ax for metric ``key``."""
        y_nat = np.array(metrics_nat[key])
        y_euc = np.array(metrics_euc[key])
        if log_scale:
            fn_nat = ax.semilogy
            fn_euc = ax.semilogy
        else:
            fn_nat = ax.plot
            fn_euc = ax.plot
        h_nat, = fn_nat(
            t_nat, y_nat,
            color=NAT_COLOR, linewidth=LW_SOLID, linestyle='-',
            label='Natural gradient',
        )
        h_euc, = fn_euc(
            t_euc, y_euc,
            color=EUC_COLOR, linewidth=LW_DASH, linestyle='--',
            alpha=ALPHA_EUC,
            label='Euclidean gradient',
        )
        ax.set_xlabel('E-step iteration')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        return h_nat, h_euc

    # -- Panel 1: F_total ------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    h_nat, h_euc = _overlay(
        ax1, 'F_total',
        ylabel='Free energy',
        title=r'Total VFE $F_{\mathrm{total}}(t)$',
        log_scale=False,
    )
    ax1.legend(
        handles=[h_nat, h_euc],
        labels=['Natural gradient', 'Euclidean gradient'],
        fontsize=7,
        loc='best',
    )

    # -- Panel 2: F_self -------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    _overlay(
        ax2, 'F_self',
        ylabel=r'$\alpha \cdot \mathrm{KL}(q \| p)$',
        title=r'Self-Coupling $F_{\mathrm{self}}(t)$',
        log_scale=False,
    )

    # -- Panel 3: F_align ------------------------------------------------------
    ax3 = fig.add_subplot(gs[0, 2])
    _overlay(
        ax3, 'F_align',
        ylabel=r'$\lambda_b \cdot F_{\mathrm{align}}$',
        title=r'Alignment Term $F_{\mathrm{align}}(t)$',
        log_scale=False,
    )

    # -- Panel 4: |eff_grad_mu| (log scale) -----------------------------------
    ax4 = fig.add_subplot(gs[1, 0])
    _overlay(
        ax4, 'grad_mu_norm',
        ylabel=r'$\|\tilde{\nabla}_\mu F\|$  (log)',
        title=r'Effective $\mu$ Gradient Norm',
        log_scale=True,
    )

    # -- Panel 5: |delta_mu| (log scale) --------------------------------------
    ax5 = fig.add_subplot(gs[1, 1])
    _overlay(
        ax5, 'delta_mu_norm',
        ylabel=r'$\|\delta\mu\|$  (log)',
        title=r'Belief Update Norm $\|\mu_{t+1} - \mu_t\|$',
        log_scale=True,
    )

    # -- Panel 6: Attention entropy --------------------------------------------
    ax6 = fig.add_subplot(gs[1, 2])
    _overlay(
        ax6, 'attn_entropy',
        ylabel='Entropy (nats)',
        title=r'Attention Entropy $H(\beta)$',
        log_scale=False,
    )
    H_uniform = math.log(cfg.seq_len)
    ax6.axhline(
        H_uniform,
        color=C['gray'],
        linestyle=':',
        linewidth=0.8,
        label=f'Uniform ({H_uniform:.2f} nats)',
    )
    ax6.legend(fontsize=7, loc='best')

    # -- Suptitle -------------------------------------------------------------
    _, n_heads, d_head = cfg.irrep_spec[0]
    fig.suptitle(
        r'Natural Gradient vs. Euclidean Gradient: VFE E-Step Convergence'
        '\n'
        rf'$K$={cfg.embed_dim}, {n_heads} heads ($d_h$={d_head}), '
        rf'$N$={cfg.seq_len}, $B$={cfg.batch_size}, '
        rf'$\alpha$={cfg.E_alpha}, $\lambda_b$={cfg.E_lambda_belief}, '
        rf'$\kappa$={cfg.kappa}',
        fontsize=11,
        y=0.995,
    )

    return fig


# =============================================================================
# CSV output
# =============================================================================

def save_metrics_csv(metrics: Dict[str, list], path: Path) -> None:
    """Write per-iteration metrics to CSV.

    Args:
        metrics: Dict mapping metric names to per-iteration lists.
        path: Output file path.
    """
    keys = list(metrics.keys())
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        n = len(metrics['iteration'])
        for i in range(n):
            writer.writerow([metrics[k][i] for k in keys])
    print(f"Saved metrics -> {path}")


# =============================================================================
# Summary statistics
# =============================================================================

def _print_summary(
    label: str,
    metrics: Dict[str, list],
    cfg: NatGradConfig,
) -> None:
    """Print convergence summary to stdout.

    Args:
        label: Mode label ('Natural gradient' or 'Euclidean gradient').
        metrics: Per-iteration metrics dict.
        cfg: Experiment config.
    """
    F_vals = metrics['F_total']
    delta_norms = metrics['delta_mu_norm']
    threshold = delta_norms[0] * 0.01

    converged_at = cfg.n_iterations
    for i, d in enumerate(delta_norms):
        if d < threshold:
            converged_at = i
            break

    print(f"  {label}")
    print(f"    F(0)   = {F_vals[0]:.6f}")
    print(f"    F(*)   = {F_vals[-1]:.6f}")
    dF = F_vals[0] - F_vals[-1]
    rel = dF / max(abs(F_vals[0]), 1e-12) * 100.0
    print(f"    dF     = {dF:.6f}  ({rel:.1f}%)")
    print(f"    Converged (1% d_mu threshold) at step: {converged_at}")
    print(f"    Final |d_mu|    = {delta_norms[-1]:.6f}")
    print(f"    Final H(beta)   = {metrics['attn_entropy'][-1]:.4f} nats")
    print(f"    Final sigma_mean = {metrics['sigma_mean'][-1]:.4f}")


# =============================================================================
# Entry point
# =============================================================================

def main() -> None:
    """Run the natural-gradient vs Euclidean-gradient ablation and save results."""
    cfg = CONFIG
    out_dir = Path(_project_root) / cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B, N, K = cfg.batch_size, cfg.seq_len, cfg.embed_dim
    _, n_heads, d_head = cfg.irrep_spec[0]

    print(f"Device: {device}")
    print(f"Beliefs: B={B}, N={N}, K={K}  ({n_heads} heads x {d_head})")
    print(f"VFE params: alpha={cfg.E_alpha}, lambda_b={cfg.E_lambda_belief}, "
          f"lambda_s={cfg.E_lambda_softmax}, kappa={cfg.kappa}")
    print(f"Step sizes: mu={cfg.E_mu_q_lr}, sigma={cfg.E_sigma_q_lr}, phi={cfg.E_phi_lr}")
    print(f"Iterations: {cfg.n_iterations}  seed: {cfg.seed}\n")

    # Build generators once; shared by both modes
    generators_np = generate_glK_multihead_generators(K, n_heads)
    generators = torch.from_numpy(generators_np).float().to(device)
    n_gen = generators.shape[0]
    print(f"Generators: {n_gen} x ({K}, {K})  [GL({d_head})^{n_heads}]\n")

    # Initialise beliefs once; both modes use identical starting point
    mu_q_init, sigma_q_init, phi_init, mu_p, sigma_p = _init_beliefs(cfg, device)

    # -- Natural gradient run --------------------------------------------------
    print("=" * 64)
    print("MODE: Natural gradient (Fisher metric)")
    print("=" * 64)
    metrics_nat = run_mode(
        mode='natural',
        cfg=cfg,
        generators=generators,
        mu_q_init=mu_q_init,
        sigma_q_init=sigma_q_init,
        phi_init=phi_init,
        mu_p=mu_p,
        sigma_p=sigma_p,
    )

    # -- Euclidean gradient run ------------------------------------------------
    print()
    print("=" * 64)
    print("MODE: Euclidean gradient (no Fisher projection)")
    print("=" * 64)
    metrics_euc = run_mode(
        mode='euclidean',
        cfg=cfg,
        generators=generators,
        mu_q_init=mu_q_init,
        sigma_q_init=sigma_q_init,
        phi_init=phi_init,
        mu_p=mu_p,
        sigma_p=sigma_p,
    )

    # -- Save CSVs -------------------------------------------------------------
    print()
    csv_nat = out_dir / 'vfe_natgrad_ablation_natural.csv'
    csv_euc = out_dir / 'vfe_natgrad_ablation_euclidean.csv'
    save_metrics_csv(metrics_nat, csv_nat)
    save_metrics_csv(metrics_euc, csv_euc)

    # -- Save figure -----------------------------------------------------------
    try:
        fig = plot_ablation(metrics_nat, metrics_euc, cfg)
        png_path = out_dir / 'vfe_natgrad_ablation.png'
        pdf_path = out_dir / 'vfe_natgrad_ablation.pdf'
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
        fig.savefig(pdf_path, bbox_inches='tight')
        print(f"Saved figure -> {png_path}")
        print(f"Saved figure -> {pdf_path}")
        import matplotlib.pyplot as plt
        plt.close(fig)
    except ImportError:
        print("matplotlib not available -- skipping figure (CSVs saved).")

    # -- Summary ---------------------------------------------------------------
    print()
    print("=" * 64)
    print("SUMMARY")
    print("=" * 64)
    _print_summary('Natural gradient', metrics_nat, cfg)
    print()
    _print_summary('Euclidean gradient', metrics_euc, cfg)

    # Comparative: which mode achieved lower final F?
    F_nat_final = metrics_nat['F_total'][-1]
    F_euc_final = metrics_euc['F_total'][-1]
    diff = F_euc_final - F_nat_final
    print()
    if diff > 0:
        print(f"  Natural gradient reached lower final F by {diff:.6f}")
    elif diff < 0:
        print(f"  Euclidean gradient reached lower final F by {-diff:.6f}")
    else:
        print("  Both modes reached identical final F.")
    print("=" * 64)


if __name__ == '__main__':
    main()
