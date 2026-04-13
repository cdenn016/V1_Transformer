"""
VFE Parameter Sweep: (alpha, kappa) Landscape
==============================================

Sweeps over a 7x7 grid of (alpha, kappa) values, running 100 E-step iterations
of multihead VFE gradient descent for each combination. Collects equilibrium
free energy F(*), convergence rate, final gradient norm, and attention entropy
into 2D arrays, then plots publication-quality heatmaps.

This is a landscape analysis: each cell of the heatmap is an independent
short VFE rollout. The EM_CONFIG operating point (alpha=1.0, kappa=3.16) is
marked with a red star on every panel.

Usage:
    Edit SWEEP_CONFIG below, then press Run.

Output:
    scripts/vfe_convergence_output/vfe_parameter_sweep.png
    scripts/vfe_convergence_output/vfe_parameter_sweep.pdf
"""

# -- Path setup ---------------------------------------------------------------
import sys
import os

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import torch

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


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class SweepConfig:
    """All parameters for the (alpha, kappa) parameter sweep.

    Geometry and VFE hyperparameters mirror EM_CONFIG in train_publication.py
    except for alpha and kappa, which are swept.

    Attributes:
        n_iterations: E-step iterations per (alpha, kappa) point.
        seed: Global RNG seed for reproducibility.
        embed_dim: Total belief dimension K.
        irrep_spec: Block structure as [(kind, n_heads, d_head)].
        gauge_group: Gauge group identifier for phi retraction.
        batch_size: Mini-batch size B (small for sweep speed).
        seq_len: Sequence length N.
        E_lambda_belief: Direct alignment weight lambda_b.
        E_lambda_softmax: Softmax coupling weight lambda_s.
        E_mu_q_lr: Natural gradient step size for mu.
        E_sigma_q_lr: SPD retraction trust radius for sigma.
        E_phi_lr: Lie-algebra step size for phi.
        diagonal_covariance: Use diagonal (B,N,K) covariances.
        sigma_max: Upper bound on sigma values.
        e_step_sigma_floor: Prior sigma floor during E-step.
        mask_self_attention: Mask KL(q_i||q_i) = 0 diagonal.
        use_causal_mask: Apply lower-triangular causal mask.
        enforce_orthogonal: Enforce Omega in SO(K).
        mu_init_std: Std of random belief mean initialisation.
        sigma_init: Initial diagonal variance value.
        phi_init_std: Std of random gauge frame initialisation.
        alpha_values: Grid of self-coupling weights to sweep.
        kappa_values: Grid of attention temperatures to sweep.
        output_dir: Directory for output files (relative to project root).
    """

    n_iterations: int = 100
    seed: int = 42

    # Geometry
    embed_dim: int = 20
    irrep_spec: list = field(default_factory=lambda: [('fund', 2, 10)])
    gauge_group: str = 'GLK'

    # Batch / sequence (small for sweep speed)
    batch_size: int = 8
    seq_len: int = 16

    # VFE weights (alpha and kappa are swept; these are the other weights)
    E_lambda_belief: float = 1.0
    E_lambda_softmax: float = 5.0

    # Step sizes
    E_mu_q_lr: float = 0.05
    E_sigma_q_lr: float = 0.05
    E_phi_lr: float = 0.05

    # Covariance mode
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

    # Sweep grid
    alpha_values: list = field(
        default_factory=lambda: [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    )
    kappa_values: list = field(
        default_factory=lambda: [0.5, 1.0, 2.0, 3.16, 5.0, 10.0, 20.0]
    )

    # EM_CONFIG operating point (marked with red star)
    op_alpha: float = 1.0
    op_kappa: float = 3.16

    # Output
    output_dir: str = 'scripts/vfe_convergence_output'


SWEEP_CONFIG = SweepConfig()


# =============================================================================
# Helpers (match vfe_convergence.py exactly)
# =============================================================================

def _diagonal_kl(
    mu_q: torch.Tensor,
    sigma_q: torch.Tensor,
    mu_p: torch.Tensor,
    sigma_p: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    r"""KL(q || p) for diagonal Gaussians.

    .. math::

        \mathrm{KL}(q \| p) = \tfrac{1}{2} \left(
            \frac{\sigma_q}{\sigma_p} + \frac{(\mu_q - \mu_p)^2}{\sigma_p}
            - 1 + \ln \frac{\sigma_p}{\sigma_q}
        \right)

    Args:
        mu_q: Posterior means (B, N, K).
        sigma_q: Posterior diagonal variances (B, N, K).
        mu_p: Prior means (B, N, K).
        sigma_p: Prior diagonal variances (B, N, K).
        eps: Variance floor for numerical stability.

    Returns:
        kl: Per-position KL divergences (B, N).
    """
    sq = sigma_q.clamp(min=eps)
    sp = sigma_p.clamp(min=eps)
    return 0.5 * (sq / sp + (mu_q - mu_p) ** 2 / sp - 1.0 + torch.log(sp / sq)).sum(-1)


def _attention_entropy(beta: torch.Tensor, eps: float = 1e-12) -> float:
    r"""Mean entropy of attention distribution in nats.

    :math:`H(\beta) = -\sum_j \beta_{ij} \ln \beta_{ij}`, averaged over B and i.

    Args:
        beta: Attention weights (B, N, N).
        eps: Log stability floor.

    Returns:
        Scalar entropy in nats.
    """
    return -(beta * (beta + eps).log()).sum(-1).mean().item()


def _build_causal_mask(N: int, device: torch.device) -> torch.Tensor:
    """Lower-triangular causal mask broadcastable to (1, N, N).

    Args:
        N: Sequence length.
        device: Target device.

    Returns:
        Boolean mask (1, N, N) with True on lower triangle.
    """
    return torch.tril(torch.ones(N, N, device=device, dtype=torch.bool)).unsqueeze(0)


# =============================================================================
# Per-point inner loop
# =============================================================================

def run_single_point(
    alpha: float,
    kappa: float,
    cfg: SweepConfig,
    generators: torch.Tensor,
    irrep_dims: List[int],
    device: torch.device,
    rng_state: torch.ByteTensor,
) -> Dict[str, float]:
    """Run VFE gradient descent for a single (alpha, kappa) pair.

    Mirrors the multihead VFE path in ``VariationalFFNDynamic`` and in
    ``vfe_convergence.py``.  Uses per-head beta and gradients via
    ``compute_attention_weights`` and ``compute_vfe_gradients_gpu``.

    Args:
        alpha: Self-coupling weight for this grid point.
        kappa: Attention temperature for this grid point.
        cfg: Sweep configuration (geometry, step sizes, etc.).
        generators: Pre-built GL(K) multihead generators on device.
        irrep_dims: List of per-head dimensions, e.g. [10, 10].
        device: Torch device.
        rng_state: Saved RNG state so each grid point starts from the
            same random initialisation.

    Returns:
        Dictionary with keys:
            ``F_final``         — equilibrium VFE F(*).
            ``converge_step``   — first iteration where |delta_mu| < 1% initial,
                                  capped at n_iterations.
            ``grad_mu_norm``    — final natural gradient norm for mu.
            ``attn_entropy``    — final mean attention entropy in nats.
    """
    # Restore RNG so every (alpha, kappa) starts from the same sample
    torch.set_rng_state(rng_state)

    B = cfg.batch_size
    N = cfg.seq_len
    K = cfg.embed_dim
    n_heads = len(irrep_dims)
    eps = 1e-6
    MAX_NAT_GRAD = 500.0

    # -- Initialise beliefs and priors (same as vfe_convergence.py) -----------
    mu_p = torch.randn(B, N, K, device=device) * cfg.mu_init_std
    sigma_p = torch.ones(B, N, K, device=device) * cfg.sigma_init

    mu_q = mu_p.clone() + torch.randn(B, N, K, device=device) * 0.3
    sigma_q = sigma_p.clone() * (
        1.0 + 0.2 * torch.randn(B, N, K, device=device)
    ).clamp(min=0.1)
    n_gen = generators.shape[0]
    phi = torch.randn(B, N, n_gen, device=device) * cfg.phi_init_std

    sigma_p_estep = sigma_p.clamp(min=cfg.e_step_sigma_floor)
    mask = _build_causal_mask(N, device) if cfg.use_causal_mask else None

    # Track initial delta_mu norm for 1% threshold
    initial_delta_mu_norm: float = float('nan')
    converge_step: int = cfg.n_iterations

    for t in range(cfg.n_iterations):
        mu_prev = mu_q.detach().clone()

        # ================================================================
        # STEP 1: Per-head beta + gradients
        # ================================================================
        grad_mu = torch.zeros_like(mu_q)
        grad_sigma = torch.zeros_like(sigma_q)
        beta_heads = []

        block_start = 0
        for h, d_h in enumerate(irrep_dims):
            block_end = block_start + d_h

            mu_h = mu_q[:, :, block_start:block_end].detach().contiguous()
            sigma_h = sigma_q[:, :, block_start:block_end].detach().contiguous()
            mu_p_h = mu_p[:, :, block_start:block_end].contiguous()
            sigma_p_h = sigma_p_estep[:, :, block_start:block_end].contiguous()
            gen_h = generators[:, block_start:block_end, block_start:block_end]

            beta_h = compute_attention_weights(
                mu_q=mu_h,
                sigma_q=sigma_h,
                phi=phi.detach(),
                generators=gen_h,
                kappa=kappa,
                epsilon=eps,
                mask=mask,
                return_kl=False,
                diagonal_covariance=cfg.diagonal_covariance,
                irrep_dims=[d_h],
                mask_self_attention=cfg.mask_self_attention,
                gauge_mode='learned',
            )

            grad_mu_h, grad_sigma_h = compute_vfe_gradients_gpu(
                mu_q=mu_h,
                sigma_q=sigma_h,
                mu_p=mu_p_h,
                sigma_p=sigma_p_h,
                beta=beta_h,
                phi=phi.detach(),
                generators=gen_h,
                alpha=alpha,
                lambda_belief=cfg.E_lambda_belief,
                lambda_softmax=cfg.E_lambda_softmax,
                kappa=kappa,
                eps=eps,
                irrep_dims=[d_h],
                enforce_orthogonal=cfg.enforce_orthogonal,
            )

            grad_mu[:, :, block_start:block_end] = grad_mu_h
            grad_sigma[:, :, block_start:block_end] = grad_sigma_h
            beta_heads.append(beta_h.detach())
            block_start = block_end

        beta_avg = sum(beta_heads) / n_heads

        # ================================================================
        # STEP 2: VFE scalar
        # ================================================================
        kl_self_per_pos = _diagonal_kl(mu_q, sigma_q, mu_p, sigma_p_estep)
        # Alignment term: sum over per-head F_align contributions
        F_align_total = torch.tensor(0.0, device=device)
        block_start = 0
        for h, d_h in enumerate(irrep_dims):
            block_end = block_start + d_h
            mu_h = mu_q[:, :, block_start:block_end].detach()
            sigma_h = sigma_q[:, :, block_start:block_end].detach()
            gen_h = generators[:, block_start:block_end, block_start:block_end]

            beta_kl_pair = compute_attention_weights(
                mu_q=mu_h,
                sigma_q=sigma_h,
                phi=phi.detach(),
                generators=gen_h,
                kappa=kappa,
                epsilon=eps,
                mask=mask,
                return_kl=True,
                diagonal_covariance=cfg.diagonal_covariance,
                irrep_dims=[d_h],
                mask_self_attention=cfg.mask_self_attention,
                gauge_mode='learned',
            )
            beta_kl, kl_h = beta_kl_pair
            F_align_total = F_align_total + (beta_kl.detach() * kl_h.detach()).sum()
            block_start = block_end

        F_self = alpha * kl_self_per_pos.mean()
        F_align = cfg.E_lambda_belief * F_align_total / (B * N)
        F_total = F_self + F_align

        # ================================================================
        # STEP 3: Natural gradient + norm clipping
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
                mu_q=mu_h,
                sigma_q=sigma_h,
                phi=phi_for_grad,
                generators=gen_h,
                kappa=kappa,
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
                grad_phi * 10.0 / (gp_norm + 1e-6),
                grad_phi,
            )

        # ================================================================
        # STEP 5: Parameter updates with trust regions
        # ================================================================
        delta_mu = -cfg.E_mu_q_lr * nat_grad_mu
        sigma_sqrt = torch.sqrt(sigma_q.detach().clamp(min=eps))
        whitened_delta = delta_mu / sigma_sqrt
        whitened_norm = torch.linalg.norm(whitened_delta, dim=-1, keepdim=True)
        scale = torch.clamp(2.0 / (whitened_norm + eps), max=1.0)
        mu_q = mu_q.detach() + scale * delta_mu

        sigma_q = retract_spd_diagonal_torch(
            sigma_diag=sigma_q.detach(),
            delta_sigma=-nat_grad_sigma,
            step_size=1.0,
            trust_region=cfg.E_sigma_q_lr,
            eps=eps,
            sigma_max=cfg.sigma_max,
        )

        # Sigma condition clamping (max ratio 10)
        max_cond = 10.0
        s_min = sigma_q.min(dim=-1, keepdim=True).values.clamp(min=eps)
        s_max_v = sigma_q.max(dim=-1, keepdim=True).values
        needs_clamp = (s_max_v / s_min) > max_cond
        geo_mean = sigma_q.log().mean(dim=-1, keepdim=True).exp()
        lower = geo_mean / (max_cond ** 0.5)
        upper = geo_mean * (max_cond ** 0.5)
        sigma_clamped = sigma_q.clamp(min=lower, max=upper)
        sigma_q = torch.where(
            needs_clamp.expand_as(sigma_q), sigma_clamped, sigma_q
        )

        phi = _retract_phi(
            phi.detach(),
            -grad_phi,
            generators,
            step_size=cfg.E_phi_lr,
            gauge_group=cfg.gauge_group,
        )

        # ================================================================
        # STEP 6: Convergence tracking
        # ================================================================
        delta_mu_norm = (mu_q - mu_prev).norm().item()

        # Record initial delta_mu to set 1% threshold
        if t == 0:
            initial_delta_mu_norm = delta_mu_norm

        # First step below 1% of initial delta_mu
        if (
            converge_step == cfg.n_iterations
            and not math.isnan(initial_delta_mu_norm)
            and initial_delta_mu_norm > 0.0
            and delta_mu_norm < 0.01 * initial_delta_mu_norm
        ):
            converge_step = t

    # Final metrics
    final_grad_mu_norm = nat_grad_mu.norm().item()
    final_attn_entropy = _attention_entropy(beta_avg)
    F_final = F_total.item()

    return {
        'F_final': F_final,
        'converge_step': converge_step,
        'grad_mu_norm': final_grad_mu_norm,
        'attn_entropy': final_attn_entropy,
    }


# =============================================================================
# Full sweep
# =============================================================================

def run_sweep(cfg: SweepConfig) -> Dict[str, np.ndarray]:
    r"""Sweep (alpha, kappa) grid and collect 2D metric arrays.

    For each of the ``len(alpha_values) * len(kappa_values)`` grid points
    an independent 100-iteration VFE rollout is performed.  All rollouts share
    the same initial belief tensor (achieved by resetting the RNG state before
    each point).

    Returns a dictionary with 2D arrays shaped
    ``(len(alpha_values), len(kappa_values))`` for each metric:

    - ``F_final``       — equilibrium free energy F(*).
    - ``converge_step`` — iteration where :math:`|\delta\mu| < 1\%` initial
                          (capped at ``n_iterations``).
    - ``grad_mu_norm``  — final natural gradient norm :math:`\|\tilde{\nabla}_\mu F\|`.
    - ``attn_entropy``  — final mean attention entropy :math:`H(\beta)` in nats.

    Args:
        cfg: Sweep configuration.

    Returns:
        Dictionary mapping metric name to (n_alpha, n_kappa) numpy array.
    """
    torch.manual_seed(cfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    B = cfg.batch_size
    N = cfg.seq_len
    K = cfg.embed_dim
    _, n_heads, d_head = cfg.irrep_spec[0]
    irrep_dims = [d_head] * n_heads

    print(f"Device: {device}")
    print(f"Sweep grid: {len(cfg.alpha_values)} alpha x {len(cfg.kappa_values)} kappa "
          f"= {len(cfg.alpha_values) * len(cfg.kappa_values)} points")
    print(f"Beliefs: B={B}, N={N}, K={K}  ({n_heads} heads x {d_head})")
    print(f"Iterations per point: {cfg.n_iterations}")
    print(f"Operating point: alpha={cfg.op_alpha}, kappa={cfg.op_kappa}")
    print()

    # Build generators once (shared across all points)
    generators_np = generate_glK_multihead_generators(K, n_heads)
    generators = torch.from_numpy(generators_np).float().to(device)
    n_gen = generators.shape[0]
    print(f"Generators: {n_gen} x ({K}, {K})  [GL({d_head})^{n_heads}]")

    # Save a reference RNG state so every point starts from the same tensors
    rng_state = torch.get_rng_state()

    n_alpha = len(cfg.alpha_values)
    n_kappa = len(cfg.kappa_values)

    F_grid = np.zeros((n_alpha, n_kappa), dtype=np.float64)
    conv_grid = np.zeros((n_alpha, n_kappa), dtype=np.float64)
    grad_grid = np.zeros((n_alpha, n_kappa), dtype=np.float64)
    entropy_grid = np.zeros((n_alpha, n_kappa), dtype=np.float64)

    total = n_alpha * n_kappa
    done = 0

    for i, alpha in enumerate(cfg.alpha_values):
        for j, kappa in enumerate(cfg.kappa_values):
            result = run_single_point(
                alpha=alpha,
                kappa=kappa,
                cfg=cfg,
                generators=generators,
                irrep_dims=irrep_dims,
                device=device,
                rng_state=rng_state,
            )
            F_grid[i, j] = result['F_final']
            conv_grid[i, j] = result['converge_step']
            grad_grid[i, j] = result['grad_mu_norm']
            entropy_grid[i, j] = result['attn_entropy']

            done += 1
            print(
                f"  [{done:3d}/{total}]  alpha={alpha:6.3f}  kappa={kappa:5.2f}"
                f"  F(*)={result['F_final']:8.4f}"
                f"  conv_step={result['converge_step']:3d}"
                f"  |grad_mu|={result['grad_mu_norm']:.4f}"
                f"  H(beta)={result['attn_entropy']:.3f}"
            )

    print(f"\nSweep complete.")
    return {
        'F_final': F_grid,
        'converge_step': conv_grid,
        'grad_mu_norm': grad_grid,
        'attn_entropy': entropy_grid,
    }


# =============================================================================
# Plotting
# =============================================================================

def _find_op_indices(
    alpha_values: List[float],
    kappa_values: List[float],
    op_alpha: float,
    op_kappa: float,
) -> Tuple[float, float]:
    r"""Find the plot-space (x, y) coordinates of the operating point.

    For ``pcolormesh`` the cell centres coincide with the grid indices.
    Returns (j, i) in x-y order for ``ax.plot``.

    Args:
        alpha_values: Sorted list of alpha grid values (y-axis).
        kappa_values: Sorted list of kappa grid values (x-axis).
        op_alpha: Operating alpha to locate.
        op_kappa: Operating kappa to locate.

    Returns:
        Tuple (x_pos, y_pos) in axis units (column index, row index).
    """
    # Use the nearest index
    alphas = np.array(alpha_values)
    kappas = np.array(kappa_values)
    i_op = int(np.argmin(np.abs(alphas - op_alpha)))
    j_op = int(np.argmin(np.abs(kappas - op_kappa)))
    return float(j_op), float(i_op)


def plot_sweep(
    results: Dict[str, np.ndarray],
    cfg: SweepConfig,
) -> 'matplotlib.figure.Figure':
    r"""Publication-quality 2x2 heatmap figure for the (alpha, kappa) sweep.

    Four panels:

    1. Equilibrium free energy :math:`F^*(\alpha, \kappa)` — log-scale colour.
    2. Convergence step — iteration at which :math:`|\delta\mu| < 1\%` initial.
    3. Final natural gradient norm :math:`\|\tilde{\nabla}_\mu F\|`.
    4. Final attention entropy :math:`H(\beta)` in nats.

    The EM_CONFIG operating point is marked with a red star on every panel.

    Args:
        results: Dictionary returned by ``run_sweep``.
        cfg: Sweep configuration (grid values, operating point, etc.).

    Returns:
        Matplotlib Figure object.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.ticker as mticker
    from transformer.visualization.pub_style import set_pub_style

    set_pub_style()

    alpha_vals = np.array(cfg.alpha_values)
    kappa_vals = np.array(cfg.kappa_values)

    # Operating point marker position (column, row in 0-indexed grid)
    op_x, op_y = _find_op_indices(
        cfg.alpha_values, cfg.kappa_values, cfg.op_alpha, cfg.op_kappa
    )

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    fig.subplots_adjust(hspace=0.38, wspace=0.38)

    # ------------------------------------------------------------------
    # Helper: draw one heatmap panel
    # ------------------------------------------------------------------
    def _heatmap(
        ax: 'plt.Axes',
        data: np.ndarray,
        title: str,
        cbar_label: str,
        cmap: str,
        log_scale: bool = False,
        symmetric: bool = False,
    ) -> None:
        """Draw a single pcolormesh heatmap on ax.

        Args:
            ax: Target axes.
            data: (n_alpha, n_kappa) 2D array.  Rows = alpha (y), cols = kappa (x).
            title: Panel title (LaTeX allowed).
            cbar_label: Colorbar axis label.
            cmap: Matplotlib colormap name.
            log_scale: Use log-normalised colour scale.
            symmetric: Use symmetric diverging normalisation centred on median.
        """
        # pcolormesh expects (y_edges, x_edges) for proper cell sizing.
        # Use cell-centre coordinates; matplotlib will draw cell boundaries.
        n_r, n_c = data.shape  # (n_alpha, n_kappa)

        # Build edge arrays for proper pcolormesh grid
        def _edges(centres: np.ndarray) -> np.ndarray:
            """Compute bin edges from bin centres (uniform in log space)."""
            log_c = np.log10(centres)
            half = np.diff(log_c) / 2.0
            left = log_c[0] - half[0]
            right = log_c[-1] + half[-1]
            edges = np.concatenate([[left], log_c[:-1] + half, [right]])
            return 10.0 ** edges

        x_edges = _edges(kappa_vals)
        y_edges = _edges(alpha_vals)

        if log_scale:
            data_pos = np.clip(data, 1e-12, None)
            norm = mcolors.LogNorm(vmin=data_pos.min(), vmax=data_pos.max())
        elif symmetric:
            vabs = np.abs(data).max()
            norm = mcolors.TwoSlopeNorm(
                vmin=-vabs, vcenter=float(np.median(data)), vmax=vabs
            )
        else:
            norm = mcolors.Normalize(vmin=data.min(), vmax=data.max())

        mesh = ax.pcolormesh(
            x_edges, y_edges, data,
            cmap=cmap, norm=norm, shading='flat',
        )
        cbar = fig.colorbar(mesh, ax=ax, pad=0.02, fraction=0.046)
        cbar.set_label(cbar_label, fontsize=9)
        cbar.ax.tick_params(labelsize=7)

        # Axis formatting — log scale on both axes (grid is quasi-log spaced)
        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.set_xticks(kappa_vals)
        ax.set_xticklabels(
            [str(v) if v == int(v) else str(v) for v in kappa_vals],
            fontsize=7,
        )
        ax.set_yticks(alpha_vals)
        ax.set_yticklabels(
            [str(v) if v == int(v) else str(v) for v in alpha_vals],
            fontsize=7,
        )
        ax.set_xlabel(r'$\kappa$', fontsize=11)
        ax.set_ylabel(r'$\alpha$', fontsize=11)
        ax.set_title(title, fontsize=11, pad=6)

        # Operating point star (op_x = kappa index -> actual kappa value,
        # op_y = alpha index -> actual alpha value)
        star_x = kappa_vals[int(op_x)]
        star_y = alpha_vals[int(op_y)]
        ax.plot(
            star_x, star_y,
            marker='*', markersize=14,
            color='#D55E00',    # red (Okabe-Ito)
            markeredgecolor='white',
            markeredgewidth=0.8,
            zorder=10,
            label=fr'$\alpha={cfg.op_alpha},\;\kappa={cfg.op_kappa}$',
        )
        ax.legend(
            loc='lower right',
            fontsize=7,
            handlelength=0.8,
            borderpad=0.5,
            framealpha=0.85,
        )

    # ------------------------------------------------------------------
    # Panel 1: Equilibrium F* — log-scale colour
    # ------------------------------------------------------------------
    _heatmap(
        axes[0, 0],
        results['F_final'],
        title=r'Equilibrium Free Energy $F^*(\alpha, \kappa)$',
        cbar_label=r'$F^*$ (log scale)',
        cmap='viridis',
        log_scale=True,
    )

    # ------------------------------------------------------------------
    # Panel 2: Convergence step
    # ------------------------------------------------------------------
    _heatmap(
        axes[0, 1],
        results['converge_step'],
        title=r'Convergence Step ($|\delta\mu| < 1\%$ initial)',
        cbar_label='Iteration',
        cmap='plasma_r',
        log_scale=False,
    )

    # ------------------------------------------------------------------
    # Panel 3: Final gradient norm
    # ------------------------------------------------------------------
    _heatmap(
        axes[1, 0],
        results['grad_mu_norm'],
        title=r'Final $\|\tilde{\nabla}_\mu F\|$',
        cbar_label=r'Gradient norm (log scale)',
        cmap='inferno',
        log_scale=True,
    )

    # ------------------------------------------------------------------
    # Panel 4: Final attention entropy
    # ------------------------------------------------------------------
    _heatmap(
        axes[1, 1],
        results['attn_entropy'],
        title=r'Final Attention Entropy $H(\beta)$',
        cbar_label=r'$H(\beta)$ (nats)',
        cmap='cividis',
        log_scale=False,
    )

    fig.suptitle(
        rf'VFE Landscape: $(\alpha, \kappa)$ Parameter Sweep'
        rf'  ($K$={cfg.embed_dim}, {cfg.irrep_spec[0][1]} heads, '
        rf'$N$={cfg.seq_len}, $T$={cfg.n_iterations} steps)',
        fontsize=12,
        y=0.995,
    )

    return fig


# =============================================================================
# Entry point
# =============================================================================

def main() -> None:
    """Run sweep, save CSV of raw grids, and save publication figures."""
    from pathlib import Path
    import csv

    cfg = SWEEP_CONFIG
    out_dir = Path(_project_root) / cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    results = run_sweep(cfg)

    # -- Save raw grids as CSV ------------------------------------------------
    csv_path = out_dir / 'vfe_parameter_sweep.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['metric', 'alpha', 'kappa', 'value']
        writer.writerow(header)
        for metric, grid in results.items():
            for i, alpha in enumerate(cfg.alpha_values):
                for j, kappa in enumerate(cfg.kappa_values):
                    writer.writerow([metric, alpha, kappa, grid[i, j]])
    print(f"Saved raw grids -> {csv_path}")

    # -- Plot and save --------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig = plot_sweep(results, cfg)
        png_path = out_dir / 'vfe_parameter_sweep.png'
        pdf_path = out_dir / 'vfe_parameter_sweep.pdf'
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
        fig.savefig(pdf_path, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved figure -> {png_path}")
        print(f"Saved figure -> {pdf_path}")
    except ImportError:
        print("matplotlib not available -- skipping figure (CSV saved).")

    # -- Summary statistics ---------------------------------------------------
    F = results['F_final']
    C = results['converge_step']
    G = results['grad_mu_norm']
    E = results['attn_entropy']

    alphas = cfg.alpha_values
    kappas = cfg.kappa_values
    op_ai = int(np.argmin(np.abs(np.array(alphas) - cfg.op_alpha)))
    op_ki = int(np.argmin(np.abs(np.array(kappas) - cfg.op_kappa)))

    print()
    print('=' * 64)
    print('  Summary statistics across sweep grid')
    print('=' * 64)
    print(f'  F(*) range:       [{F.min():.4f}, {F.max():.4f}]')
    print(f'  Conv step range:  [{int(C.min())}, {int(C.max())}]')
    print(f'  |grad_mu| range:  [{G.min():.4f}, {G.max():.4f}]')
    print(f'  H(beta) range:    [{E.min():.4f}, {E.max():.4f}]')
    print()
    print(f'  Operating point (alpha={cfg.op_alpha}, kappa={cfg.op_kappa}):')
    print(f'    F(*) = {F[op_ai, op_ki]:.6f}')
    print(f'    conv_step = {int(C[op_ai, op_ki])}')
    print(f'    |grad_mu| = {G[op_ai, op_ki]:.6f}')
    print(f'    H(beta)   = {E[op_ai, op_ki]:.6f}')
    print('=' * 64)


if __name__ == '__main__':
    main()
