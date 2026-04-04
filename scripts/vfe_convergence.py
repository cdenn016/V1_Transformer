"""
VFE Convergence Analysis
========================

Run pure VFE gradient descent on Gaussian beliefs and plot the free energy
curve F(t) over iterations. Reveals how many E-steps are needed to reach
the equilibrium q* and characterises the convergence basin.

Uses the **same config values, gradient functions, and natural-gradient
retractions** as ``VariationalFFNDynamic`` in the main training pipeline.
Per-head beta and gradient computation mirrors the multihead VFE path.

Usage:
    Edit the CONFIG dict below, then press Run.

Output:
    scripts/vfe_convergence_output/vfe_convergence.png   (6-panel figure)
    scripts/vfe_convergence_output/vfe_convergence.pdf
    scripts/vfe_convergence_output/metrics.csv            (raw data)
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
from typing import List, Optional, Dict

import torch
import torch.nn.functional as F
import numpy as np

# -- Project imports (same functions the pipeline uses) ------------------------
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
# CONFIG -- edit these to match your EM_CONFIG or experiment with alternatives
# =============================================================================
@dataclass
class VFEConvergenceConfig:
    """All parameters for the convergence experiment.

    Defaults mirror EM_CONFIG in train_publication.py.
    """
    # --- Experiment ---
    n_iterations: int = 500          # How many E-steps to run
    seed: int = 42

    # --- Geometry (must match train_publication EM_CONFIG) ---
    embed_dim: int = 20              # K  (total belief dimension)
    irrep_spec: list = field(        # [('fund', n_heads, d_head)]
        default_factory=lambda: [('fund', 2, 10)]
    )
    gauge_group: str = 'GLK'

    # --- Batch / sequence ---
    batch_size: int = 64
    seq_len: int = 64

    # --- VFE hyperparameters (E-step, from EM_CONFIG) ---
    E_alpha: float = 1.0             # Self-coupling weight alpha
    E_lambda_belief: float = 1.0     # Direct alignment weight lambda_b
    E_lambda_softmax: float = 5.0    # Softmax coupling weight lambda_s
    kappa: float = 3.16              # Attention temperature kappa

    E_mu_q_lr: float = 0.05          # mu natural-gradient step size
    E_sigma_q_lr: float = 0.05      # sigma SPD-retraction trust region
    E_phi_lr: float = 0.05          # phi Lie-algebra step size

    # --- Covariance mode ---
    diagonal_covariance: bool = True
    sigma_max: float = 5.0
    e_step_sigma_floor: float = 0.1

    # --- Masking ---
    mask_self_attention: bool = True  # Prevent KL(q_i||q_i)=0 collapse
    use_causal_mask: bool = True      # Autoregressive mask

    # --- Gauge ---
    enforce_orthogonal: bool = False

    # --- Initialisation ---
    mu_init_std: float = 1.0         # Std of random belief means
    sigma_init: float = 1.0          # Initial diagonal variance
    phi_init_std: float = 0.1        # Std of random gauge frames

    # --- Output ---
    output_dir: str = 'scripts/vfe_convergence_output'


CONFIG = VFEConvergenceConfig()


# =============================================================================
# Helpers
# =============================================================================

def _diagonal_kl(mu_q, sigma_q, mu_p, sigma_p, eps=1e-6):
    r"""KL(q || p) for diagonal Gaussians, per position.

    Returns (B, N) per-position KL values.
    """
    sq = sigma_q.clamp(min=eps)
    sp = sigma_p.clamp(min=eps)
    return 0.5 * (sq / sp + (mu_q - mu_p) ** 2 / sp - 1.0 + torch.log(sp / sq)).sum(-1)


def _attention_entropy(beta, eps=1e-12):
    """Mean entropy of attention distribution beta, in nats.  (B, N, N) -> scalar."""
    return -(beta * (beta + eps).log()).sum(-1).mean()


def _build_causal_mask(N, device):
    """Lower-triangular causal mask (B=1 broadcastable)."""
    return torch.tril(torch.ones(N, N, device=device, dtype=torch.bool)).unsqueeze(0)


# =============================================================================
# Main convergence loop
# =============================================================================

def run_vfe_convergence(cfg: VFEConvergenceConfig) -> Dict[str, list]:
    """Run VFE gradient descent and return per-iteration metrics.

    Mirrors the multihead VFE path in VariationalFFNDynamic._vfe_iteration:
    - Per-head beta_h computation via compute_attention_weights
    - Per-head gradient via compute_vfe_gradients_gpu (same as pipeline)
    - Natural gradient + norm clipping (max 500)
    - Whitened mu trust region (radius 2.0)
    - SPD sigma retraction with trust region
    - Condition clamping (max ratio 10)
    - Phi gradient via autograd on alignment loss
    """
    torch.manual_seed(cfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B, N, K = cfg.batch_size, cfg.seq_len, cfg.embed_dim
    _, n_heads, d_head = cfg.irrep_spec[0]
    irrep_dims = [d_head] * n_heads

    print(f"Device: {device}")
    print(f"Beliefs: B={B}, N={N}, K={K}  ({n_heads} heads x {d_head})")
    print(f"VFE params: alpha={cfg.E_alpha}, lambda_b={cfg.E_lambda_belief}, "
          f"lambda_s={cfg.E_lambda_softmax}, kappa={cfg.kappa}")
    print(f"Step sizes: mu={cfg.E_mu_q_lr}, sigma={cfg.E_sigma_q_lr}, phi={cfg.E_phi_lr}")
    print(f"Running {cfg.n_iterations} E-step iterations ...\n")

    # -- Build generators (same as GaugeTransformerLM._build_generators) ------
    generators_np = generate_glK_multihead_generators(K, n_heads)
    generators = torch.from_numpy(generators_np).float().to(device)
    n_gen = generators.shape[0]
    print(f"Generators: {n_gen} x ({K}, {K})  [GL({d_head})^{n_heads}]")

    # -- Initialise beliefs and priors ----------------------------------------
    mu_p = torch.randn(B, N, K, device=device) * cfg.mu_init_std
    sigma_p = torch.ones(B, N, K, device=device) * cfg.sigma_init

    # Beliefs: start from perturbed copy of priors (mimics embedding init)
    mu_q = mu_p.clone() + torch.randn(B, N, K, device=device) * 0.3
    sigma_q = sigma_p.clone() * (1.0 + 0.2 * torch.randn(B, N, K, device=device)).clamp(min=0.1)
    phi = torch.randn(B, N, n_gen, device=device) * cfg.phi_init_std

    # Floor sigma_p for E-step stability (same as VariationalFFNDynamic)
    sigma_p_estep = sigma_p.clamp(min=cfg.e_step_sigma_floor)

    # Causal mask
    mask = _build_causal_mask(N, device) if cfg.use_causal_mask else None

    # -- Metric storage -------------------------------------------------------
    metrics = {
        'iteration':       [], 'F_total':         [], 'F_self':          [],
        'F_align':         [], 'grad_mu_norm':    [], 'grad_sigma_norm': [],
        'grad_phi_norm':   [], 'delta_mu_norm':   [], 'delta_sigma_norm':[],
        'delta_phi_norm':  [], 'attn_entropy':    [], 'sigma_mean':      [],
        'sigma_min':       [], 'sigma_max':       [], 'kl_self_mean':    [],
    }

    eps = 1e-6
    MAX_NAT_GRAD = 500.0

    # -- E-step loop ----------------------------------------------------------
    for t in range(cfg.n_iterations):
        mu_prev = mu_q.detach().clone()
        sigma_prev = sigma_q.detach().clone()
        phi_prev = phi.detach().clone()

        # ================================================================
        # STEP 1: Per-head beta + gradients (multihead VFE path)
        # Mirrors variational_ffn.py:2455-2586
        # ================================================================
        grad_mu = torch.zeros_like(mu_q)
        grad_sigma = torch.zeros_like(sigma_q)
        beta_heads = []
        F_align_total = torch.tensor(0.0, device=device)

        block_start = 0
        for h, d_h in enumerate(irrep_dims):
            block_end = block_start + d_h

            mu_h = mu_q[:, :, block_start:block_end].detach().contiguous()
            sigma_h = sigma_q[:, :, block_start:block_end].detach().contiguous()
            mu_p_h = mu_p[:, :, block_start:block_end].contiguous()
            sigma_p_h = sigma_p_estep[:, :, block_start:block_end].contiguous()
            gen_h = generators[:, block_start:block_end, block_start:block_end]
            kappa_h = cfg.kappa  # bare kappa; compute_attention_weights scales by sqrt(d_h)

            # Compute per-head beta
            beta_h = compute_attention_weights(
                mu_q=mu_h, sigma_q=sigma_h,
                phi=phi.detach(), generators=gen_h,
                kappa=kappa_h, epsilon=eps, mask=mask,
                return_kl=False,
                diagonal_covariance=cfg.diagonal_covariance,
                irrep_dims=[d_h],
                mask_self_attention=cfg.mask_self_attention,
                gauge_mode='learned',
            )

            # Compute per-head VFE gradients
            grad_mu_h, grad_sigma_h = compute_vfe_gradients_gpu(
                mu_q=mu_h, sigma_q=sigma_h,
                mu_p=mu_p_h, sigma_p=sigma_p_h,
                beta=beta_h, phi=phi.detach(), generators=gen_h,
                alpha=cfg.E_alpha,
                lambda_belief=cfg.E_lambda_belief,
                lambda_softmax=cfg.E_lambda_softmax,
                kappa=kappa_h, eps=eps,
                irrep_dims=[d_h],
                enforce_orthogonal=cfg.enforce_orthogonal,
            )

            # Accumulate
            grad_mu[:, :, block_start:block_end] = grad_mu_h
            grad_sigma[:, :, block_start:block_end] = grad_sigma_h
            beta_heads.append(beta_h.detach())

            # Compute alignment KL for this head (for F_align scalar)
            beta_h_kl = compute_attention_weights(
                mu_q=mu_h, sigma_q=sigma_h,
                phi=phi.detach(), generators=gen_h,
                kappa=kappa_h, epsilon=eps, mask=mask,
                return_kl=True,
                diagonal_covariance=cfg.diagonal_covariance,
                irrep_dims=[d_h],
                mask_self_attention=cfg.mask_self_attention,
                gauge_mode='learned',
            )
            beta_kl, kl_h = beta_h_kl
            F_align_total = F_align_total + (beta_kl.detach() * kl_h.detach()).sum()

            block_start = block_end

        # Average beta across heads (for entropy metric)
        beta_avg = sum(beta_heads) / n_heads

        # ================================================================
        # STEP 2: VFE scalar
        # ================================================================
        kl_self_per_pos = _diagonal_kl(mu_q, sigma_q, mu_p, sigma_p_estep)
        F_self = cfg.E_alpha * kl_self_per_pos.mean()
        F_align = cfg.E_lambda_belief * F_align_total / (B * N)
        F_total = F_self + F_align

        # ================================================================
        # STEP 3: Natural gradient + norm clipping
        # (variational_ffn.py:2761-2790)
        # ================================================================
        nat_grad_mu, nat_grad_sigma = compute_natural_gradient_gpu(
            grad_mu, grad_sigma, sigma_q.detach(), eps=eps,
        )

        ng_mu_norm = torch.linalg.norm(nat_grad_mu, dim=-1, keepdim=True)
        nat_grad_mu = nat_grad_mu * torch.clamp(MAX_NAT_GRAD / (ng_mu_norm + eps), max=1.0)
        ng_sig_norm = torch.linalg.norm(nat_grad_sigma, dim=-1, keepdim=True)
        nat_grad_sigma = nat_grad_sigma * torch.clamp(MAX_NAT_GRAD / (ng_sig_norm + eps), max=1.0)

        # ================================================================
        # STEP 4: Phi gradient via autograd on alignment loss
        # (variational_ffn.py:1080-1178)
        # ================================================================
        phi_for_grad = phi.detach().clone().requires_grad_(True)
        alignment_loss_phi = torch.tensor(0.0, device=device)
        block_start = 0
        for h, d_h in enumerate(irrep_dims):
            block_end = block_start + d_h
            mu_h = mu_q[:, :, block_start:block_end].detach()
            sigma_h = sigma_q[:, :, block_start:block_end].detach()
            gen_h = generators[:, block_start:block_end, block_start:block_end]

            beta_phi_h, kl_phi_h = compute_attention_weights(
                mu_q=mu_h, sigma_q=sigma_h,
                phi=phi_for_grad, generators=gen_h,
                kappa=cfg.kappa, epsilon=eps, mask=mask,
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
                alignment_loss_phi, phi_for_grad,
                create_graph=False, retain_graph=False,
            )[0]
            gp_norm = torch.norm(grad_phi, dim=-1, keepdim=True)
            grad_phi = torch.where(
                gp_norm > 10.0,
                grad_phi * 10.0 / (gp_norm + 1e-6),
                grad_phi,
            )

        # ================================================================
        # STEP 5: Update beliefs with trust regions
        # (variational_ffn.py:2953-3057)
        # ================================================================
        effective_lr = cfg.E_mu_q_lr

        # 5a. Mu: whitened trust region (radius 2.0)
        delta_mu = -effective_lr * nat_grad_mu
        sigma_sqrt = torch.sqrt(sigma_q.detach().clamp(min=eps))
        whitened_delta = delta_mu / sigma_sqrt
        whitened_norm = torch.linalg.norm(whitened_delta, dim=-1, keepdim=True)
        scale = torch.clamp(2.0 / (whitened_norm + eps), max=1.0)
        mu_q = mu_q.detach() + scale * delta_mu

        # 5b. Sigma: SPD retraction (step_size=1, trust_region=sigma_lr)
        sigma_q = retract_spd_diagonal_torch(
            sigma_diag=sigma_q.detach(),
            delta_sigma=-nat_grad_sigma,
            step_size=1.0,
            trust_region=cfg.E_sigma_q_lr,
            eps=eps,
            sigma_max=cfg.sigma_max,
        )

        # 5c. Sigma condition clamping (max ratio 10)
        max_cond = 10.0
        s_min = sigma_q.min(dim=-1, keepdim=True).values.clamp(min=eps)
        s_max_v = sigma_q.max(dim=-1, keepdim=True).values
        needs_clamp = (s_max_v / s_min) > max_cond
        geo_mean = sigma_q.log().mean(dim=-1, keepdim=True).exp()
        lower = geo_mean / (max_cond ** 0.5)
        upper = geo_mean * (max_cond ** 0.5)
        sigma_clamped = sigma_q.clamp(min=lower, max=upper)
        sigma_q = torch.where(needs_clamp.expand_as(sigma_q), sigma_clamped, sigma_q)

        # 5d. Phi retraction
        phi = _retract_phi(
            phi.detach(), -grad_phi, generators,
            step_size=cfg.E_phi_lr,
            gauge_group=cfg.gauge_group,
        )

        # ================================================================
        # STEP 6: Record metrics
        # ================================================================
        metrics['iteration'].append(t)
        metrics['F_total'].append(F_total.item())
        metrics['F_self'].append(F_self.item())
        metrics['F_align'].append(F_align.item())
        metrics['grad_mu_norm'].append(nat_grad_mu.norm().item())
        metrics['grad_sigma_norm'].append(nat_grad_sigma.norm().item())
        metrics['grad_phi_norm'].append(grad_phi.norm().item())
        metrics['delta_mu_norm'].append((mu_q - mu_prev).norm().item())
        metrics['delta_sigma_norm'].append((sigma_q - sigma_prev).norm().item())
        metrics['delta_phi_norm'].append((phi - phi_prev).norm().item())
        metrics['attn_entropy'].append(_attention_entropy(beta_avg).item())
        metrics['sigma_mean'].append(sigma_q.mean().item())
        metrics['sigma_min'].append(sigma_q.min().item())
        metrics['sigma_max'].append(sigma_q.max().item())
        metrics['kl_self_mean'].append(kl_self_per_pos.mean().item())

        if t % 20 == 0 or t == cfg.n_iterations - 1:
            print(f"  step {t:4d}  F={F_total.item():8.4f}  "
                  f"(self={F_self.item():.4f}  align={F_align.item():.4f})  "
                  f"|grad_mu|={nat_grad_mu.norm().item():.4f}  "
                  f"|d_mu|={metrics['delta_mu_norm'][-1]:.4f}  "
                  f"H(beta)={metrics['attn_entropy'][-1]:.3f}")

    print(f"\nDone. Final F = {metrics['F_total'][-1]:.6f}")
    return metrics


# =============================================================================
# Plotting
# =============================================================================

def plot_convergence(metrics: Dict[str, list], cfg: VFEConvergenceConfig):
    """Publication-quality 6-panel convergence figure."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from transformer.visualization.pub_style import set_pub_style, PUB_COLORS

    set_pub_style()

    C = PUB_COLORS
    t = np.array(metrics['iteration'])

    fig = plt.figure(figsize=(14, 9))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.35)

    # -- Panel 1: VFE F(t) ---------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, metrics['F_total'], color=C['blue'], linewidth=1.8, label=r'$F_{\mathrm{total}}$')
    ax1.plot(t, metrics['F_self'], color=C['orange'], linewidth=1.2, linestyle='--', label=r'$\alpha \cdot \mathrm{KL}(q \| p)$')
    ax1.plot(t, metrics['F_align'], color=C['green'], linewidth=1.2, linestyle='--', label=r'$\lambda \cdot F_{\mathrm{align}}$')
    ax1.set_xlabel('E-step iteration')
    ax1.set_ylabel('Free energy')
    ax1.set_title('VFE Convergence')
    ax1.legend(fontsize=7, loc='best')

    # -- Panel 2: Gradient norms (log scale) ----------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.semilogy(t, metrics['grad_mu_norm'], color=C['blue'], linewidth=1.4, label=r'$\|\tilde{\nabla}_\mu F\|$')
    ax2.semilogy(t, metrics['grad_sigma_norm'], color=C['orange'], linewidth=1.4, label=r'$\|\tilde{\nabla}_\sigma F\|$')
    ax2.semilogy(t, metrics['grad_phi_norm'], color=C['green'], linewidth=1.4, label=r'$\|\nabla_\phi F\|$')
    ax2.set_xlabel('E-step iteration')
    ax2.set_ylabel('Gradient norm (log)')
    ax2.set_title('Natural Gradient Norms')
    ax2.legend(fontsize=7)

    # -- Panel 3: Belief deltas (convergence rate) ----------------------------
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.semilogy(t, metrics['delta_mu_norm'], color=C['blue'], linewidth=1.4, label=r'$\|\delta\mu\|$')
    ax3.semilogy(t, metrics['delta_sigma_norm'], color=C['orange'], linewidth=1.4, label=r'$\|\delta\sigma\|$')
    ax3.semilogy(t, metrics['delta_phi_norm'], color=C['green'], linewidth=1.4, label=r'$\|\delta\phi\|$')
    ax3.set_xlabel('E-step iteration')
    ax3.set_ylabel('Step size (log)')
    ax3.set_title(r'Belief Update $\|\theta_{t+1} - \theta_t\|$')
    ax3.legend(fontsize=7)

    # -- Panel 4: Attention entropy -------------------------------------------
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(t, metrics['attn_entropy'], color=C['purple'], linewidth=1.4)
    H_uniform = math.log(cfg.seq_len)
    ax4.axhline(H_uniform, color=C['gray'], linestyle=':', linewidth=0.8,
                label=f'Uniform ({H_uniform:.2f} nats)')
    ax4.set_xlabel('E-step iteration')
    ax4.set_ylabel('Entropy (nats)')
    ax4.set_title(r'Attention Entropy $H(\beta)$')
    ax4.legend(fontsize=7)

    # -- Panel 5: Covariance evolution ----------------------------------------
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(t, metrics['sigma_mean'], color=C['blue'], linewidth=1.4, label=r'$\bar{\sigma}$')
    ax5.fill_between(t, metrics['sigma_min'], metrics['sigma_max'],
                     color=C['blue'], alpha=0.12, label=r'$[\sigma_{\min}, \sigma_{\max}]$')
    ax5.set_xlabel('E-step iteration')
    ax5.set_ylabel(r'$\sigma$ (diagonal)')
    ax5.set_title('Covariance Evolution')
    ax5.legend(fontsize=7)

    # -- Panel 6: Self-coupling KL --------------------------------------------
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(t, metrics['kl_self_mean'], color=C['red'], linewidth=1.4)
    ax6.set_xlabel('E-step iteration')
    ax6.set_ylabel(r'$\mathrm{KL}(q \| p)$')
    ax6.set_title('Mean Self-Coupling KL')

    # -- Suptitle -------------------------------------------------------------
    fig.suptitle(
        f'VFE E-Step Convergence   '
        f'($K$={cfg.embed_dim}, {cfg.irrep_spec[0][1]} heads, '
        f'$N$={cfg.seq_len}, '
        rf'$\alpha$={cfg.E_alpha}, $\kappa$={cfg.kappa})',
        fontsize=13, y=0.98,
    )

    return fig


def save_metrics_csv(metrics: Dict[str, list], path: Path):
    """Write metrics to CSV for external analysis."""
    keys = list(metrics.keys())
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for i in range(len(metrics['iteration'])):
            writer.writerow([metrics[k][i] for k in keys])
    print(f"Saved metrics -> {path}")


# =============================================================================
# Entry point
# =============================================================================

def main():
    cfg = CONFIG
    out_dir = Path(_project_root) / cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = run_vfe_convergence(cfg)

    save_metrics_csv(metrics, out_dir / 'metrics.csv')

    try:
        fig = plot_convergence(metrics, cfg)
        fig.savefig(out_dir / 'vfe_convergence.png', dpi=300, bbox_inches='tight')
        fig.savefig(out_dir / 'vfe_convergence.pdf', bbox_inches='tight')
        print(f"Saved figure -> {out_dir / 'vfe_convergence.png'}")
        import matplotlib.pyplot as plt
        plt.close(fig)
    except ImportError:
        print("matplotlib not available -- skipping plot (CSV saved).")

    # Summary
    F_vals = metrics['F_total']
    F_init, F_final = F_vals[0], F_vals[-1]
    delta_norms = metrics['delta_mu_norm']
    threshold = delta_norms[0] * 0.01
    converged_at = cfg.n_iterations
    for i, d in enumerate(delta_norms):
        if d < threshold:
            converged_at = i
            break

    print(f"\n{'='*60}")
    print(f"  F(0)  = {F_init:.6f}")
    print(f"  F(*)  = {F_final:.6f}")
    print(f"  dF    = {F_init - F_final:.6f}  ({(F_init - F_final) / max(abs(F_init), 1e-12) * 100:.1f}%)")
    print(f"  Approx convergence (1% d_mu threshold): step {converged_at}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
