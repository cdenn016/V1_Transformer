"""
VFE Convergence with Synthetic Observations
============================================

Extension of vfe_convergence.py that adds a synthetic cross-entropy (CE)
observation term to the VFE gradient descent.  Runs TWO experiments from
identical initial conditions:

  1. No observations  -- pure VFE minimization (reproduces vfe_convergence.py)
  2. With observations -- adds CE gradient from random targets + W_out

The observation term grounds beliefs to data, producing a sharp q*
equilibrium.  Overlaying both curves shows how data coupling collapses
the limit-cycle behaviour seen without observations.

Observation gradient derivation (Stein's lemma / second-order Gaussian):
    d/dmu   E_q[CE] = (softmax(mu @ W_out.T) - one_hot) @ W_out
    d/dsigma E_q[CE] = (1/2) * Var_W[w_k]  where  Var_W = E[w^2] - E[w]^2
                      (diagonal approximation, exact for Gaussian beliefs)

Usage:
    Edit CONFIG below, then press Run.

Output:
    scripts/vfe_convergence_output/vfe_obs_comparison.png
    scripts/vfe_convergence_output/vfe_obs_comparison.pdf
    scripts/vfe_convergence_output/metrics_obs.csv
    scripts/vfe_convergence_output/metrics_noobs.csv
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
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
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
# CONFIG  --  edit these, then press Run
# =============================================================================

@dataclass
class Config:
    r"""All parameters for the observation-grounding convergence experiment.

    Defaults mirror EM_CONFIG in train_publication.py.  Observation params
    (vocab_size, obs_sigma_weight) extend the base VFEConvergenceConfig.
    """
    # --- Experiment ---
    n_iterations: int = 500
    seed: int = 42

    # --- Geometry ---
    embed_dim: int = 20                    # K  (total belief dimension)
    irrep_spec: list = field(              # [('fund', n_heads, d_head)]
        default_factory=lambda: [('fund', 2, 10)]
    )
    gauge_group: str = 'GLK'

    # --- Batch / sequence ---
    batch_size: int = 32
    seq_len: int = 64

    # --- VFE hyperparameters (E-step) ---
    E_alpha: float = 1.0
    E_lambda_belief: float = 1.0
    E_lambda_softmax: float = 1.0
    kappa: float = 3.16

    E_mu_q_lr: float = 0.05
    E_sigma_q_lr: float = 0.05
    E_phi_lr: float = 0.05

    # --- Covariance mode ---
    diagonal_covariance: bool = False
    sigma_max: float = 12.0
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

    # --- Observation parameters ---
    vocab_size: int = 500       # Small vocab for synthetic experiment
    obs_sigma_weight: float = 1.0

    # --- Output ---
    output_dir: str = 'scripts/vfe_convergence_output'


CONFIG = Config()


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
    r"""KL(q || p) for diagonal Gaussians, per position.

    .. math::
        \mathrm{KL}(q \| p) = \frac{1}{2}\sum_k
            \left[\frac{\sigma_q^k}{\sigma_p^k}
            + \frac{(\mu_q^k - \mu_p^k)^2}{\sigma_p^k} - 1
            + \ln\frac{\sigma_p^k}{\sigma_q^k}\right]

    Args:
        mu_q:    (B, N, K) belief means.
        sigma_q: (B, N, K) belief diagonal variances.
        mu_p:    (B, N, K) prior means.
        sigma_p: (B, N, K) prior diagonal variances.
        eps:     Numerical floor applied to variances.

    Returns:
        Tensor of shape (B, N) containing per-position KL values.
    """
    sq = sigma_q.clamp(min=eps)
    sp = sigma_p.clamp(min=eps)
    return 0.5 * (sq / sp + (mu_q - mu_p) ** 2 / sp - 1.0 + torch.log(sp / sq)).sum(-1)


def _attention_entropy(beta: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Mean Shannon entropy of the attention distribution (in nats).

    Args:
        beta: (B, N, N) attention weight tensor.
        eps:  Small constant for numerical stability.

    Returns:
        Scalar tensor with mean entropy across batch and query positions.
    """
    return -(beta * (beta + eps).log()).sum(-1).mean()


def _build_causal_mask(N: int, device: torch.device) -> torch.Tensor:
    """Lower-triangular causal mask broadcastable over batch.

    Args:
        N:      Sequence length.
        device: Target device.

    Returns:
        Boolean tensor of shape (1, N, N).
    """
    return torch.tril(torch.ones(N, N, device=device, dtype=torch.bool)).unsqueeze(0)


def _ce_loss(
    mu: torch.Tensor,
    W_out: torch.Tensor,
    targets: torch.Tensor,
    mask_obs: torch.Tensor,
) -> torch.Tensor:
    r"""Cross-entropy observation loss for diagnostic tracking.

    Computes :math:`-\log p(o \mid \mu)` under the softmax likelihood,
    averaged over valid (non-masked) positions.

    Args:
        mu:       (B, N, K) current belief means.
        W_out:    (vocab_size, K) output projection.
        targets:  (B, N) integer target token indices; -1 = masked.
        mask_obs: (B, N, 1) float mask; 1 where target is valid.

    Returns:
        Scalar mean CE loss over unmasked positions.
    """
    logits = mu.detach() @ W_out.T           # (B, N, V)
    log_probs = F.log_softmax(logits, dim=-1)
    targets_clamped = targets.clamp(min=0)
    # Gather log-prob of the correct token at each position
    lp = log_probs.gather(-1, targets_clamped.unsqueeze(-1)).squeeze(-1)  # (B, N)
    mask_2d = mask_obs.squeeze(-1)                                         # (B, N)
    n_valid = mask_2d.sum().clamp(min=1.0)
    return -(lp * mask_2d).sum() / n_valid


# =============================================================================
# Observation gradient
# =============================================================================

def _compute_obs_gradients(
    mu: torch.Tensor,
    W_out: torch.Tensor,
    one_hot: torch.Tensor,
    mask_obs: torch.Tensor,
    W_out_sq: torch.Tensor,
    obs_sigma_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""CE observation gradients w.r.t. mu and diagonal sigma.

    Implements the Stein-lemma Hessian approximation from
    ``variational_ffn.py:2673-2722``:

    .. math::
        \nabla_\mu  F_\mathrm{obs} &= (\hat{p} - e_y)\,W_{\mathrm{out}} \\
        \nabla_\sigma F_\mathrm{obs} &= \tfrac{1}{2}\,
            \mathbb{V}\!_{\hat{p}}[W_{\mathrm{out},k}]

    where :math:`\hat{p} = \mathrm{softmax}(\mu W^T)`.

    Args:
        mu:               (B, N, K) current belief means (detached).
        W_out:            (V, K) output projection matrix.
        one_hot:          (B, N, V) one-hot targets, zeroed at masked positions.
        mask_obs:         (B, N, 1) float observation mask.
        W_out_sq:         (V, K) pre-computed W_out ** 2.
        obs_sigma_weight: Scalar weight applied to the sigma gradient.

    Returns:
        Tuple (obs_mu_grad, obs_sigma_grad), each of shape (B, N, K).
    """
    logits = mu @ W_out.T                          # (B, N, V)
    probs = F.softmax(logits, dim=-1)              # (B, N, V)

    # mu gradient: (p_hat - one_hot) @ W_out
    grad_error = (probs - one_hot) * mask_obs      # (B, N, V)
    obs_mu_grad = grad_error @ W_out               # (B, N, K)

    # sigma gradient via Stein's lemma / Hessian diagonal:
    #   dE_q[CE]/d sigma_k = (1/2) Var_p[W_k]  >= 0
    EW2 = probs @ W_out_sq                         # (B, N, K)
    EW = probs @ W_out                             # (B, N, K)
    hessian_diag = (EW2 - EW ** 2).clamp(min=0.0) # (B, N, K)
    obs_sigma_grad = (
        0.5 * obs_sigma_weight * hessian_diag * mask_obs
    ).clamp(max=10.0)                              # (B, N, K)

    return obs_mu_grad, obs_sigma_grad


# =============================================================================
# Core convergence loop
# =============================================================================

def run_vfe_convergence(
    cfg: Config,
    with_observations: bool,
    shared_init: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, list]:
    r"""Run VFE gradient descent and return per-iteration metrics.

    Mirrors the multihead VFE path in ``VariationalFFNDynamic._vfe_iteration``:

    - Per-head beta_h via ``compute_attention_weights``
    - Per-head gradient via ``compute_vfe_gradients_gpu`` (same as pipeline)
    - Optional observation gradient (CE) appended after per-head accumulation
    - Natural gradient + norm clipping (max 500)
    - Whitened mu trust region (radius 2.0)
    - SPD sigma retraction with trust region
    - Condition clamping (max ratio 10)
    - Phi gradient via autograd on alignment loss

    Args:
        cfg:               Experiment configuration dataclass.
        with_observations: If True, adds CE observation gradient each step.
        shared_init:       Pre-built tensors from a first call (for identical
                           initial conditions).  If None, tensors are created
                           from ``cfg.seed``.

    Returns:
        Dictionary mapping metric names to lists of per-iteration values.
        Includes ``F_obs`` (CE loss); this is zero for the no-observations run.
    """
    label = "with obs" if with_observations else "no obs"
    torch.manual_seed(cfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B, N, K = cfg.batch_size, cfg.seq_len, cfg.embed_dim
    _, n_heads, d_head = cfg.irrep_spec[0]
    irrep_dims = [d_head] * n_heads

    print(f"[{label}] Device: {device}")
    print(f"[{label}] Beliefs: B={B}, N={N}, K={K}  ({n_heads} heads x {d_head})")
    print(f"[{label}] VFE: alpha={cfg.E_alpha}, lambda_b={cfg.E_lambda_belief}, "
          f"lambda_s={cfg.E_lambda_softmax}, kappa={cfg.kappa}")
    print(f"[{label}] Step sizes: mu={cfg.E_mu_q_lr}, sigma={cfg.E_sigma_q_lr}, "
          f"phi={cfg.E_phi_lr}")
    if with_observations:
        print(f"[{label}] Vocab: {cfg.vocab_size}, obs_sigma_weight={cfg.obs_sigma_weight}")
    print(f"[{label}] Running {cfg.n_iterations} E-step iterations ...\n")

    # -- Build generators (same as GaugeTransformerLM._build_generators) ------
    generators_np = generate_glK_multihead_generators(K, n_heads)
    generators = torch.from_numpy(generators_np).float().to(device)
    n_gen = generators.shape[0]
    print(f"[{label}] Generators: {n_gen} x ({K}, {K})  [GL({d_head})^{n_heads}]")

    # -- Initialise beliefs and priors ----------------------------------------
    if shared_init is not None:
        mu_p    = shared_init['mu_p'].clone().to(device)
        sigma_p = shared_init['sigma_p'].clone().to(device)
        mu_q    = shared_init['mu_q'].clone().to(device)
        sigma_q = shared_init['sigma_q'].clone().to(device)
        phi     = shared_init['phi'].clone().to(device)
    else:
        mu_p    = torch.randn(B, N, K, device=device) * cfg.mu_init_std
        sigma_p = torch.ones(B, N, K, device=device) * cfg.sigma_init
        mu_q    = mu_p.clone() + torch.randn(B, N, K, device=device) * 0.3
        sigma_q = sigma_p.clone() * (
            1.0 + 0.2 * torch.randn(B, N, K, device=device)
        ).clamp(min=0.1)
        phi     = torch.randn(B, N, n_gen, device=device) * cfg.phi_init_std

    # Floor sigma_p for E-step stability
    sigma_p_estep = sigma_p.clamp(min=cfg.e_step_sigma_floor)

    # Causal mask
    mask = _build_causal_mask(N, device) if cfg.use_causal_mask else None

    # -- Observation setup (with_observations mode only) ----------------------
    W_out: Optional[torch.Tensor] = None
    targets: Optional[torch.Tensor] = None
    one_hot: Optional[torch.Tensor] = None
    mask_obs: Optional[torch.Tensor] = None
    W_out_sq: Optional[torch.Tensor] = None

    if with_observations:
        if shared_init is not None and 'W_out' in shared_init:
            W_out   = shared_init['W_out'].clone().to(device)
            targets = shared_init['targets'].clone().to(device)
        else:
            # Random Gaussian output projection (not trained, just provides signal)
            W_out   = torch.randn(cfg.vocab_size, K, device=device) * (K ** -0.5)
            # Random integer targets in [0, vocab_size)
            targets = torch.randint(0, cfg.vocab_size, (B, N), device=device)

        # Pre-compute observation cache (same pattern as variational_ffn.py:3303-3314)
        targets_clipped = targets.clamp(min=0)
        mask_obs = (targets != -1).unsqueeze(-1).float()   # (B, N, 1)  -- all valid here
        one_hot  = F.one_hot(targets_clipped, cfg.vocab_size).float() * mask_obs  # (B, N, V)
        W_out_sq = W_out ** 2                              # (V, K)

    # -- Metric storage -------------------------------------------------------
    metrics: Dict[str, list] = {
        'iteration':        [],
        'F_total':          [],
        'F_self':           [],
        'F_align':          [],
        'F_obs':            [],
        'grad_mu_norm':     [],
        'grad_sigma_norm':  [],
        'grad_phi_norm':    [],
        'delta_mu_norm':    [],
        'delta_sigma_norm': [],
        'delta_phi_norm':   [],
        'attn_entropy':     [],
        'sigma_mean':       [],
        'sigma_min':        [],
        'sigma_max':        [],
        'kl_self_mean':     [],
    }

    eps = 1e-6
    MAX_NAT_GRAD = 500.0

    # -- E-step loop ----------------------------------------------------------
    for t in range(cfg.n_iterations):
        mu_prev    = mu_q.detach().clone()
        sigma_prev = sigma_q.detach().clone()
        phi_prev   = phi.detach().clone()

        # ====================================================================
        # STEP 1: Per-head beta + gradients (multihead VFE path)
        # Mirrors variational_ffn.py:2455-2586
        # ====================================================================
        grad_mu    = torch.zeros_like(mu_q)
        grad_sigma = torch.zeros_like(sigma_q)
        beta_heads: List[torch.Tensor] = []
        F_align_total = torch.tensor(0.0, device=device)

        block_start = 0
        for h, d_h in enumerate(irrep_dims):
            block_end = block_start + d_h

            mu_h      = mu_q[:, :, block_start:block_end].detach().contiguous()
            sigma_h   = sigma_q[:, :, block_start:block_end].detach().contiguous()
            mu_p_h    = mu_p[:, :, block_start:block_end].contiguous()
            sigma_p_h = sigma_p_estep[:, :, block_start:block_end].contiguous()
            gen_h     = generators[:, block_start:block_end, block_start:block_end]
            kappa_h   = cfg.kappa  # bare kappa; compute_attention_weights scales by sqrt(d_h)

            # Per-head attention weights
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

            # Per-head VFE gradients
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

            grad_mu[:, :, block_start:block_end]    = grad_mu_h
            grad_sigma[:, :, block_start:block_end] = grad_sigma_h
            beta_heads.append(beta_h.detach())

            # Alignment KL for F_align scalar (return_kl=True)
            beta_h_kl, kl_h = compute_attention_weights(
                mu_q=mu_h, sigma_q=sigma_h,
                phi=phi.detach(), generators=gen_h,
                kappa=kappa_h, epsilon=eps, mask=mask,
                return_kl=True,
                diagonal_covariance=cfg.diagonal_covariance,
                irrep_dims=[d_h],
                mask_self_attention=cfg.mask_self_attention,
                gauge_mode='learned',
            )
            F_align_total = F_align_total + (beta_h_kl.detach() * kl_h.detach()).sum()

            block_start = block_end

        # Average beta across heads (for entropy metric)
        beta_avg = sum(beta_heads) / n_heads  # type: ignore[arg-type]

        # ====================================================================
        # STEP 2: Observation gradient (with_observations mode only)
        # Implements variational_ffn.py:2673-2722
        # ====================================================================
        F_obs_val = torch.tensor(0.0, device=device)
        if with_observations:
            obs_mu_grad, obs_sigma_grad = _compute_obs_gradients(
                mu=mu_q.detach(),
                W_out=W_out,
                one_hot=one_hot,
                mask_obs=mask_obs,
                W_out_sq=W_out_sq,
                obs_sigma_weight=cfg.obs_sigma_weight,
            )
            grad_mu    = grad_mu    + obs_mu_grad
            grad_sigma = grad_sigma + obs_sigma_grad
            F_obs_val  = _ce_loss(mu_q, W_out, targets, mask_obs)

        # ====================================================================
        # STEP 3: VFE scalar
        # ====================================================================
        kl_self_per_pos = _diagonal_kl(mu_q, sigma_q, mu_p, sigma_p_estep)
        F_self  = cfg.E_alpha * kl_self_per_pos.mean()
        F_align = cfg.E_lambda_belief * F_align_total / (B * N)
        F_total = F_self + F_align + F_obs_val

        # ====================================================================
        # STEP 4: Natural gradient + norm clipping
        # variational_ffn.py:2761-2790
        # ====================================================================
        nat_grad_mu, nat_grad_sigma = compute_natural_gradient_gpu(
            grad_mu, grad_sigma, sigma_q.detach(), eps=eps,
        )

        ng_mu_norm  = torch.linalg.norm(nat_grad_mu,    dim=-1, keepdim=True)
        nat_grad_mu = nat_grad_mu * torch.clamp(MAX_NAT_GRAD / (ng_mu_norm + eps), max=1.0)

        ng_sig_norm    = torch.linalg.norm(nat_grad_sigma, dim=-1, keepdim=True)
        nat_grad_sigma = nat_grad_sigma * torch.clamp(MAX_NAT_GRAD / (ng_sig_norm + eps), max=1.0)

        # ====================================================================
        # STEP 5: Phi gradient via autograd on alignment loss
        # variational_ffn.py:1080-1178
        # ====================================================================
        phi_for_grad = phi.detach().clone().requires_grad_(True)
        alignment_loss_phi = torch.tensor(0.0, device=device)

        block_start = 0
        for h, d_h in enumerate(irrep_dims):
            block_end = block_start + d_h
            mu_h    = mu_q[:, :, block_start:block_end].detach()
            sigma_h = sigma_q[:, :, block_start:block_end].detach()
            gen_h   = generators[:, block_start:block_end, block_start:block_end]

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
                cfg.E_lambda_belief  * (beta_phi_h.detach() * kl_phi_h).sum()
                + cfg.E_lambda_softmax * (beta_phi_h * kl_phi_h.detach()).sum()
            )
            block_start = block_end

        grad_phi = torch.zeros_like(phi)
        if alignment_loss_phi.grad_fn is not None:
            grad_phi = torch.autograd.grad(
                alignment_loss_phi, phi_for_grad,
                create_graph=False, retain_graph=False,
            )[0]
            gp_norm  = torch.norm(grad_phi, dim=-1, keepdim=True)
            grad_phi = torch.where(
                gp_norm > 10.0,
                grad_phi * 10.0 / (gp_norm + eps),
                grad_phi,
            )

        # ====================================================================
        # STEP 6: Update beliefs with trust regions
        # variational_ffn.py:2953-3057
        # ====================================================================
        effective_lr = cfg.E_mu_q_lr

        # 6a. Mu: whitened trust region (radius 2.0)
        delta_mu      = -effective_lr * nat_grad_mu
        sigma_sqrt    = torch.sqrt(sigma_q.detach().clamp(min=eps))
        whitened_delta = delta_mu / sigma_sqrt
        whitened_norm  = torch.linalg.norm(whitened_delta, dim=-1, keepdim=True)
        scale          = torch.clamp(2.0 / (whitened_norm + eps), max=1.0)
        mu_q           = mu_q.detach() + scale * delta_mu

        # 6b. Sigma: SPD retraction
        sigma_q = retract_spd_diagonal_torch(
            sigma_diag=sigma_q.detach(),
            delta_sigma=-nat_grad_sigma,
            step_size=1.0,
            trust_region=cfg.E_sigma_q_lr,
            eps=eps,
            sigma_max=cfg.sigma_max,
        )

        # 6c. Sigma condition clamping (max ratio 10)
        max_cond = 10.0
        s_min    = sigma_q.min(dim=-1, keepdim=True).values.clamp(min=eps)
        s_max_v  = sigma_q.max(dim=-1, keepdim=True).values
        needs_clamp = (s_max_v / s_min) > max_cond
        geo_mean    = sigma_q.log().mean(dim=-1, keepdim=True).exp()
        lower       = geo_mean / (max_cond ** 0.5)
        upper       = geo_mean * (max_cond ** 0.5)
        sigma_clamped = sigma_q.clamp(min=lower, max=upper)
        sigma_q = torch.where(needs_clamp.expand_as(sigma_q), sigma_clamped, sigma_q)

        # 6d. Phi retraction
        phi = _retract_phi(
            phi.detach(), -grad_phi, generators,
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
        metrics['F_obs'].append(F_obs_val.item())
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
            obs_str = f"  F_obs={F_obs_val.item():.4f}" if with_observations else ""
            print(
                f"  [{label}] step {t:4d}  F={F_total.item():8.4f}"
                f"  (self={F_self.item():.4f}  align={F_align.item():.4f}"
                f"{obs_str})"
                f"  |grad_mu|={nat_grad_mu.norm().item():.4f}"
                f"  H(beta)={metrics['attn_entropy'][-1]:.3f}"
            )

    print(f"\n[{label}] Done. Final F = {metrics['F_total'][-1]:.6f}\n")
    return metrics


# =============================================================================
# Shared initialisation
# =============================================================================

def build_shared_init(cfg: Config) -> Dict[str, torch.Tensor]:
    """Create tensors shared across both experiment conditions.

    Both runs start from identical (mu_p, sigma_p, mu_q, sigma_q, phi)
    so that differences in convergence are attributable solely to the
    presence or absence of the observation gradient.

    Args:
        cfg: Experiment configuration.

    Returns:
        Dictionary of named tensors (CPU, moved to device inside run loop).
    """
    torch.manual_seed(cfg.seed)
    B, N, K = cfg.batch_size, cfg.seq_len, cfg.embed_dim
    _, n_heads, d_head = cfg.irrep_spec[0]

    # Generators needed only to get n_gen for phi shape
    generators_np = generate_glK_multihead_generators(K, n_heads)
    n_gen = generators_np.shape[0]

    mu_p    = torch.randn(B, N, K) * cfg.mu_init_std
    sigma_p = torch.ones(B, N, K) * cfg.sigma_init
    mu_q    = mu_p.clone() + torch.randn(B, N, K) * 0.3
    sigma_q = sigma_p.clone() * (1.0 + 0.2 * torch.randn(B, N, K)).clamp(min=0.1)
    phi     = torch.randn(B, N, n_gen) * cfg.phi_init_std

    # Observation tensors (shared so both runs see same targets)
    targets = torch.randint(0, cfg.vocab_size, (B, N))
    W_out   = torch.randn(cfg.vocab_size, K) * (K ** -0.5)

    return {
        'mu_p':    mu_p,
        'sigma_p': sigma_p,
        'mu_q':    mu_q,
        'sigma_q': sigma_q,
        'phi':     phi,
        'targets': targets,
        'W_out':   W_out,
    }


# =============================================================================
# Plotting
# =============================================================================

def plot_comparison(
    metrics_noobs: Dict[str, list],
    metrics_obs: Dict[str, list],
    cfg: Config,
) -> 'matplotlib.figure.Figure':  # noqa: F821
    """Publication-quality 6-panel overlay figure.

    Panels:
        1. F_total(t)  -- total VFE, both conditions overlaid
        2. F_self(t)   -- prior consistency KL
        3. F_align(t)  -- belief alignment term
        4. Gradient norms (log scale)
        5. Attention entropy H(beta)
        6. F_obs(t)    -- CE observation loss (non-zero only for obs condition)

    Args:
        metrics_noobs: Per-iteration metrics from the no-observations run.
        metrics_obs:   Per-iteration metrics from the with-observations run.
        cfg:           Experiment configuration (for axis annotations).

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
    t = np.array(metrics_noobs['iteration'])

    # -------------------------------------------------------------------------
    # Line style convention
    #   no obs  -- dashed  (alpha 0.75)
    #   with obs -- solid  (alpha 1.0)
    # -------------------------------------------------------------------------
    LW_MAIN  = 1.9
    LW_MINOR = 1.4
    ALPHA_NO  = 0.75
    ALPHA_OBS = 1.0

    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 3, hspace=0.38, wspace=0.36)

    # Helper: add both-condition traces to an axis
    def _dual(ax, key, color, lw=LW_MAIN, label_stem=''):
        y_no  = np.array(metrics_noobs[key])
        y_obs = np.array(metrics_obs[key])
        ax.plot(t, y_no,  color=color, linewidth=lw, linestyle='--',
                alpha=ALPHA_NO,  label=f'{label_stem} (no obs)')
        ax.plot(t, y_obs, color=color, linewidth=lw, linestyle='-',
                alpha=ALPHA_OBS, label=f'{label_stem} (obs)')

    # -- Panel 1: F_total -----------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    _dual(ax1, 'F_total', C['blue'], lw=LW_MAIN, label_stem=r'$F_{\mathrm{total}}$')
    ax1.set_xlabel('E-step iteration')
    ax1.set_ylabel('Free energy')
    ax1.set_title('Total VFE')
    ax1.legend(fontsize=7, loc='best')

    # -- Panel 2: F_self ------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    _dual(ax2, 'F_self', C['orange'], lw=LW_MINOR,
          label_stem=r'$\alpha\,\mathrm{KL}(q\|p)$')
    ax2.set_xlabel('E-step iteration')
    ax2.set_ylabel('Free energy')
    ax2.set_title('Self-Coupling KL')
    ax2.legend(fontsize=7)

    # -- Panel 3: F_align -----------------------------------------------------
    ax3 = fig.add_subplot(gs[0, 2])
    _dual(ax3, 'F_align', C['green'], lw=LW_MINOR,
          label_stem=r'$\lambda\,F_{\mathrm{align}}$')
    ax3.set_xlabel('E-step iteration')
    ax3.set_ylabel('Free energy')
    ax3.set_title('Belief Alignment')
    ax3.legend(fontsize=7)

    # -- Panel 4: Gradient norms (log scale) ----------------------------------
    ax4 = fig.add_subplot(gs[1, 0])
    # mu norms
    ax4.semilogy(t, metrics_noobs['grad_mu_norm'],   color=C['blue'],
                 lw=LW_MINOR, linestyle='--', alpha=ALPHA_NO,
                 label=r'$\|\tilde{\nabla}_\mu F\|$ (no obs)')
    ax4.semilogy(t, metrics_obs['grad_mu_norm'],     color=C['blue'],
                 lw=LW_MINOR, linestyle='-',  alpha=ALPHA_OBS,
                 label=r'$\|\tilde{\nabla}_\mu F\|$ (obs)')
    # sigma norms
    ax4.semilogy(t, metrics_noobs['grad_sigma_norm'], color=C['orange'],
                 lw=LW_MINOR, linestyle='--', alpha=ALPHA_NO,
                 label=r'$\|\tilde{\nabla}_\sigma F\|$ (no obs)')
    ax4.semilogy(t, metrics_obs['grad_sigma_norm'],   color=C['orange'],
                 lw=LW_MINOR, linestyle='-',  alpha=ALPHA_OBS,
                 label=r'$\|\tilde{\nabla}_\sigma F\|$ (obs)')
    ax4.set_xlabel('E-step iteration')
    ax4.set_ylabel('Gradient norm (log)')
    ax4.set_title('Natural Gradient Norms')
    ax4.legend(fontsize=6)

    # -- Panel 5: Attention entropy -------------------------------------------
    ax5 = fig.add_subplot(gs[1, 1])
    _dual(ax5, 'attn_entropy', C['purple'], lw=LW_MAIN,
          label_stem=r'$H(\beta)$')
    H_uniform = math.log(cfg.seq_len)
    ax5.axhline(H_uniform, color=C['gray'], linestyle=':', linewidth=0.8,
                label=f'Uniform ({H_uniform:.2f} nats)')
    ax5.set_xlabel('E-step iteration')
    ax5.set_ylabel('Entropy (nats)')
    ax5.set_title(r'Attention Entropy $H(\beta)$')
    ax5.legend(fontsize=7)

    # -- Panel 6: F_obs (CE observation loss) ---------------------------------
    ax6 = fig.add_subplot(gs[1, 2])
    f_obs_arr = np.array(metrics_obs['F_obs'])
    ax6.plot(t, f_obs_arr, color=C['red'], linewidth=LW_MAIN,
             label='with obs')
    # Draw a flat zero reference for the no-obs run
    ax6.axhline(0.0, color=C['gray'], linestyle='--', linewidth=0.9,
                alpha=0.7, label='no obs (0)')
    ax6.set_xlabel('E-step iteration')
    ax6.set_ylabel(r'$-\log p(o \mid \mu)$')
    ax6.set_title('Observation CE Loss')
    ax6.legend(fontsize=7)

    # -- Suptitle -------------------------------------------------------------
    fig.suptitle(
        r'VFE Convergence: observations vs.\ no observations'
        f'   ($K$={cfg.embed_dim}, {cfg.irrep_spec[0][1]} heads, '
        f'$N$={cfg.seq_len}, '
        rf'$\alpha$={cfg.E_alpha}, $\kappa$={cfg.kappa}, '
        f'vocab={cfg.vocab_size})',
        fontsize=13, y=0.995,
    )

    return fig


# =============================================================================
# CSV export
# =============================================================================

def save_metrics_csv(metrics: Dict[str, list], path: Path) -> None:
    """Write per-iteration metrics to CSV.

    Args:
        metrics: Dictionary of metric name to list of scalar values.
        path:    Destination file path (created or overwritten).
    """
    keys = list(metrics.keys())
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for i in range(len(metrics['iteration'])):
            writer.writerow([metrics[k][i] for k in keys])
    print(f"Saved metrics -> {path}")


# =============================================================================
# Summary statistics
# =============================================================================

def _print_summary(label: str, metrics: Dict[str, list], cfg: Config) -> None:
    """Print a brief convergence summary to stdout.

    Args:
        label:   Short description of the run (e.g. 'no obs').
        metrics: Per-iteration metrics dictionary.
        cfg:     Experiment configuration.
    """
    F_vals  = metrics['F_total']
    F_init  = F_vals[0]
    F_final = F_vals[-1]
    delta_norms = metrics['delta_mu_norm']
    threshold   = delta_norms[0] * 0.01
    converged_at = cfg.n_iterations
    for i, d in enumerate(delta_norms):
        if d < threshold:
            converged_at = i
            break
    pct = (F_init - F_final) / max(abs(F_init), 1e-12) * 100.0
    print(f"  [{label}]  F(0)={F_init:.6f}  F(*)={F_final:.6f}"
          f"  dF={F_init - F_final:.6f} ({pct:.1f}%)"
          f"  approx_convergence_step={converged_at}")


# =============================================================================
# Entry point
# =============================================================================

def main() -> None:
    """Run both convergence experiments and save results."""
    cfg     = CONFIG
    out_dir = Path(_project_root) / cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  VFE Convergence: observation grounding experiment")
    print(f"  n_iterations={cfg.n_iterations}  seed={cfg.seed}")
    print(f"  embed_dim={cfg.embed_dim}  seq_len={cfg.seq_len}  batch={cfg.batch_size}")
    print("=" * 60 + "\n")

    # -- Build shared initial conditions (identical for both runs) ------------
    print("Building shared initial conditions ...")
    shared_init = build_shared_init(cfg)

    # -- Run 1: No observations -----------------------------------------------
    print("\n--- Run 1: no observations ---")
    metrics_noobs = run_vfe_convergence(cfg, with_observations=False,
                                        shared_init=shared_init)

    # -- Run 2: With observations ---------------------------------------------
    print("\n--- Run 2: with observations ---")
    metrics_obs   = run_vfe_convergence(cfg, with_observations=True,
                                        shared_init=shared_init)

    # -- Save CSVs ------------------------------------------------------------
    save_metrics_csv(metrics_noobs, out_dir / 'metrics_noobs.csv')
    save_metrics_csv(metrics_obs,   out_dir / 'metrics_obs.csv')

    # -- Plot -----------------------------------------------------------------
    try:
        fig = plot_comparison(metrics_noobs, metrics_obs, cfg)
        png_path = out_dir / 'vfe_obs_comparison.png'
        pdf_path = out_dir / 'vfe_obs_comparison.pdf'
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
        fig.savefig(pdf_path,           bbox_inches='tight')
        print(f"Saved figure -> {png_path}")
        import matplotlib.pyplot as plt
        plt.close(fig)
    except ImportError:
        print("matplotlib not available -- skipping plot (CSVs saved).")

    # -- Summary --------------------------------------------------------------
    print(f"\n{'=' * 60}")
    _print_summary('no obs',  metrics_noobs, cfg)
    _print_summary('with obs', metrics_obs,  cfg)
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
