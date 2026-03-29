"""
Optimizer Creation with Parameter Grouping
==========================================

Parameter-group-aware optimizers for the gauge-theoretic transformer.
Three optimizer types:

1. AdamW (default): Standard adaptive optimizer. Diagonal Fisher approximation
   via exponential moving average of squared gradients.

2. RiemannianAdamW: AdamW with geometric preconditioning.
   - phi params: Killing-form metric g̃_ab = 2K tr(G_a^T G_b) - 2 tr(G_a) tr(G_b)
     on the Lie algebra. For SO(N) with Frobenius-orthonormal generators this is
     trivial (scalar); for GL(K) it couples the trace direction.
   - mu params: Fisher metric for Gaussian location parameters. Scales gradients
     by the current variance σ²_v per token per dimension, giving the natural
     gradient ∇_nat μ = Σ_v · ∇_E μ. High-uncertainty dimensions move faster.
   - sigma params: Fisher metric for log-variance is constant (= 1/2), so the
     natural gradient is 2× the Euclidean gradient. Handled via LR scaling.

3. NaturalGradientOptimizer: Per-token block-diagonal empirical Fisher.
   Maintains K×K Fisher blocks for each vocabulary token via EMA of gradient
   outer products. The off-diagonal structure captures dimension correlations
   that Adam misses. O(V·K³) per step; memory O(V·K²).

Embedding weight decay acts as a Gaussian hyper-prior N(0, 1/(2*wd))
at the top of the VFE Bayesian hierarchy.
"""

import math
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple
from transformer.training.config import TrainingConfig


# =============================================================================
# Riemannian AdamW
# =============================================================================

class RiemannianAdamW(torch.optim.AdamW):
    r"""AdamW with Riemannian preconditioning for gauge-theoretic parameters.

    Extends AdamW by applying the geometrically correct metric to gradients
    before the Adam moment update. Since the Lie algebra is a flat vector space,
    parallel transport is trivial and the exponential map is addition, so
    Riemannian Adam reduces to: (1) transform gradient by metric inverse,
    (2) run standard Adam on the transformed gradient.

    Metrics applied per parameter group:
        - 'phi_embed' / 'omega_embed': Killing-form inverse on gl(K).
          Precomputed as g̃^{-1} where g̃_ab = 2K tr(G_a^T G_b) - 2 tr(G_a) tr(G_b).
          For SO(N) with orthonormal generators this is ~(1/2K)·I (scalar rescaling);
          for GL(K) the trace direction is regularized.
        - 'mu_embed': Fisher metric for Gaussian location. Natural gradient is
          Σ_v · ∇_E μ_v (scale by variance). Only active when model has
          learnable sigma (otherwise all tokens share the same variance and
          the scaling is equivalent to an LR change).
        - 'sigma_embed': Fisher metric for log-variance is 1/2, so natural
          gradient = 2 · ∇_E η. Applied as a fixed factor-of-2 rescaling.
        - All others: standard AdamW (no preconditioning).

    Args:
        params: Parameter groups from create_param_groups()
        model: Reference to GaugeTransformerLM for accessing sigma values
        killing_inv: Precomputed inverse Killing metric (n_gen, n_gen) or None
        **kwargs: Forwarded to torch.optim.AdamW (lr, betas, eps, weight_decay)
    """

    def __init__(
        self,
        params,
        model: nn.Module = None,
        killing_inv: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        super().__init__(params, **kwargs)
        self._model = model
        self._killing_inv = killing_inv

    @torch.no_grad()
    def step(self, closure=None):
        # Precondition gradients before AdamW processes them
        for group in self.param_groups:
            name = group.get('name', '')

            if 'phi' in name or 'omega' in name:
                self._precondition_phi(group)
            elif 'mu' in name:
                self._precondition_mu(group)
            elif 'sigma' in name:
                self._precondition_sigma(group)

        return super().step(closure)

    def _precondition_phi(self, group: dict) -> None:
        """Apply inverse Killing metric to phi/omega gradients."""
        if self._killing_inv is None:
            return
        for p in group['params']:
            if p.grad is None:
                continue
            # grad: (V, n_gen) or (n_gen,) — metric: (n_gen, n_gen)
            dev = p.grad.device
            K_inv = self._killing_inv.to(device=dev, dtype=p.grad.dtype)
            p.grad = p.grad @ K_inv

    def _precondition_mu(self, group: dict) -> None:
        r"""Apply Fisher metric for Gaussian location: ∇_nat μ = Σ · ∇_E μ.

        The Fisher information for the mean of N(μ, Σ) is Σ^{-1}.
        The natural gradient is therefore F^{-1} g = Σ · g, which scales each
        dimension by the current variance. High-uncertainty directions get
        larger steps (the optimizer should explore more where it knows less).
        """
        if self._model is None:
            return
        sigma = self._get_sigma()
        if sigma is None:
            return
        for p in group['params']:
            if p.grad is None:
                continue
            # p.grad: (V, K), sigma: (V, K) — elementwise multiply
            if p.grad.shape == sigma.shape:
                p.grad = p.grad * sigma

    def _precondition_sigma(self, group: dict) -> None:
        r"""Apply Fisher metric for log-variance: ∇_nat η = 2 · ∇_E η.

        For the log-variance parameterization η = log(σ²), the Fisher
        information is F_ηη = 1/2 (constant). The natural gradient is
        F^{-1} g = 2g.
        """
        for p in group['params']:
            if p.grad is not None:
                p.grad = p.grad * 2.0

    def _get_sigma(self) -> Optional[torch.Tensor]:
        """Retrieve current variance values from model embeddings."""
        embed = getattr(self._model, 'token_embed', None)
        if embed is None:
            return None
        if hasattr(embed, 'log_sigma_diag') and isinstance(embed.log_sigma_diag, nn.Parameter):
            return torch.exp(embed.log_sigma_diag.data).clamp(min=1e-6, max=10.0)
        return None


# =============================================================================
# Natural Gradient Optimizer (block-diagonal empirical Fisher)
# =============================================================================

class NaturalGradientOptimizer(torch.optim.Optimizer):
    r"""Natural gradient descent with per-token block-diagonal Fisher information.

    For embedding parameters (V, K), maintains a K×K empirical Fisher block
    per vocabulary token, updated via EMA of gradient outer products:

        F̂_v^{(t)} = (1 - ρ) F̂_v^{(t-1)} + ρ · g_v g_v^T

    The natural gradient update is:

        θ_v ← θ_v - lr · (F̂_v + λI)^{-1} g_v

    This captures per-dimension correlations that diagonal approximations
    (Adam) miss. The Fisher F̂_v converges to the true Fisher E[g_v g_v^T]
    under ergodic sampling.

    Cost: O(K²) per token for outer product, O(K³) for solve. Total per step:
    O(|batch_tokens| · K³) for the solve (only tokens in the batch are updated).
    Memory: O(V · K²) for storing Fisher blocks — substantial for large V.
    For V=50K, K=64: ~819 MB. Use with small vocabularies or reduce K.

    For non-embedding parameters (1D, small 2D), falls back to standard
    gradient descent with weight decay (no Fisher tracking).

    WARNING on damping: For rarely-seen tokens, the Fisher is near-zero and
    (F + λI)^{-1} ≈ (1/λ)I. Small λ (e.g., 1e-4) amplifies rare tokens'
    gradients by up to 1/λ (clipped to max_ratio=10×), causing systematic
    over-updating of rare embeddings → embedding homogenization → attention
    collapse. Use λ ≥ 1e-2 for stability.

    Args:
        params: Parameter groups from create_param_groups()
        lr: Learning rate (default 1e-3)
        weight_decay: Decoupled weight decay coefficient
        ema_decay: EMA decay for Fisher estimation (default 0.95)
        damping: Tikhonov regularization λ for Fisher inversion (default 1e-2)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        ema_decay: float = 0.95,
        damping: float = 1e-2,
    ):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.ema_decay = ema_decay
        self.damping = damping
        # Per-group gradient norm diagnostics (populated by step())
        self._grad_norms: Dict[str, Dict[str, float]] = {}

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Reset per-step norm accumulators
        group_euclidean_sq: Dict[str, float] = {}
        group_natural_sq: Dict[str, float] = {}

        for group in self.param_groups:
            lr = group['lr']
            wd = group['weight_decay']
            gname = group.get('name', 'unnamed')

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # Decoupled weight decay (applies to all rows, every step)
                if wd > 0:
                    p.mul_(1.0 - lr * wd)

                # For 2D embedding-like parameters: per-row Fisher blocks
                if grad.dim() == 2 and grad.shape[-1] >= 4:
                    eucl_sq, nat_sq = self._natural_step_embedding(p, grad, state, lr)
                    group_euclidean_sq[gname] = group_euclidean_sq.get(gname, 0.0) + eucl_sq
                    group_natural_sq[gname] = group_natural_sq.get(gname, 0.0) + nat_sq
                else:
                    # Fallback: plain gradient descent for 1D / small params
                    p.add_(grad, alpha=-lr)
                    g_sq = grad.norm().item() ** 2
                    group_euclidean_sq[gname] = group_euclidean_sq.get(gname, 0.0) + g_sq
                    group_natural_sq[gname] = group_natural_sq.get(gname, 0.0) + g_sq

        # Store L2 norms per group
        self._grad_norms = {}
        for gname in set(list(group_euclidean_sq.keys()) + list(group_natural_sq.keys())):
            self._grad_norms[gname] = {
                'euclidean': math.sqrt(group_euclidean_sq.get(gname, 0.0)),
                'natural': math.sqrt(group_natural_sq.get(gname, 0.0)),
            }

        return loss

    def _natural_step_embedding(
        self,
        param: torch.Tensor,    # (V, K)
        grad: torch.Tensor,     # (V, K)
        state: dict,
        lr: float,
    ) -> Tuple[float, float]:
        """Per-token natural gradient step with block-diagonal Fisher.

        Returns:
            (euclidean_norm_sq, natural_norm_sq): Squared L2 norms of the
            raw gradient and the Fisher-preconditioned gradient for this param.
        """
        V, K = grad.shape
        device = grad.device
        dtype = grad.dtype

        # Initialize Fisher blocks on first call
        if 'fisher' not in state:
            state['fisher'] = torch.zeros(V, K, K, device=device, dtype=dtype)
            state['step'] = 0
        state['step'] += 1

        fisher = state['fisher']

        # Find tokens with nonzero gradient (only batch tokens)
        grad_norm = grad.abs().sum(dim=-1)  # (V,)
        nz_mask = grad_norm > 0
        if not nz_mask.any():
            eucl_sq = grad.norm().item() ** 2
            return eucl_sq, eucl_sq

        nz_idx = nz_mask.nonzero(as_tuple=True)[0]  # (n_active,)
        g = grad[nz_idx]  # (n_active, K)

        # Update Fisher EMA: F = (1-ρ)F + ρ g gᵀ
        outer = g.unsqueeze(-1) * g.unsqueeze(-2)  # (n_active, K, K)
        rho = self.ema_decay
        # Bias correction for early steps: use Adam-style correction factor
        # to compensate for cold-start (Fisher initialized at zero).
        # Without correction, the early Fisher is biased toward zero,
        # making the natural gradient ≈ Euclidean gradient.
        step = state['step']
        bias_correction = 1.0 - (1.0 - rho) ** step
        fisher[nz_idx] = (1.0 - rho) * fisher[nz_idx] + rho * outer

        # Natural gradient: ng = (F̂ + λI)⁻¹ g  where F̂ = F / bias_correction
        F_active = fisher[nz_idx] / bias_correction  # Bias-corrected Fisher
        I_K = torch.eye(K, device=device, dtype=dtype)
        F_damped = F_active + self.damping * I_K

        # Solve F_damped @ ng = g  (batched K×K linear solve)
        ng = torch.linalg.solve(F_damped, g.unsqueeze(-1)).squeeze(-1)  # (n_active, K)

        # Record norms before clipping
        eucl_sq = g.norm().item() ** 2
        nat_sq = ng.norm().item() ** 2

        # Clip natural gradient to prevent explosion from ill-conditioned Fisher
        ng_norm = ng.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        g_norm = g.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        max_ratio = 10.0  # Natural gradient shouldn't be >10x the Euclidean gradient
        scale = torch.clamp(max_ratio * g_norm / ng_norm, max=1.0)
        ng = ng * scale

        param[nz_idx] -= lr * ng

        return eucl_sq, nat_sq

    def get_grad_norms(self) -> Dict[str, Dict[str, float]]:
        r"""Return per-group gradient norms from the last step() call.

        Returns:
            Dict mapping group name → {'euclidean': ‖g‖, 'natural': ‖F⁻¹g‖}.
            For non-embedding params (plain SGD fallback), both values are equal.
        """
        return dict(self._grad_norms)


# =============================================================================
# Utility: Precompute Killing metric inverse from generators
# =============================================================================

def compute_killing_metric_inv(
    generators: torch.Tensor,
    center_reg: float = None,
) -> torch.Tensor:
    r"""Compute inverse modified Killing metric for M-step phi preconditioning.

    Uses the Cartan-involution-modified Killing form:
        g̃_ab = 2K · tr(G_a^T G_b) - 2 · tr(G_a) · tr(G_b)

    This is positive semidefinite, degenerate only on the center ℝ·I of gl(K).
    The center direction is regularized to prevent pathological amplification.

    For SO(N) with orthonormal generators (tr(G_a^T G_b) = δ_{ab}/2, tr(G_a) = 0):
        g̃ = K · I  →  g̃^{-1} = (1/K) · I  (trivial scalar rescaling)

    For GL(K) with standard E_{ij} basis:
        Non-trivial coupling through the trace direction.

    Args:
        generators: (n_gen, K, K) Lie algebra basis
        center_reg: Regularization for the degenerate center direction.
            Default None → 2K (matches non-center eigenvalues for isotropic
            conditioning). WARNING: Small values like 1e-4 create condition
            numbers of 2K/center_reg (200,000× for K=10), amplifying the
            trace direction and causing phi runaway → attention collapse.

    Returns:
        inv_metric: (n_gen, n_gen) inverse metric tensor
    """
    # Reuse the existing implementation in gauge_preconditioner
    from transformer.core.gauge_preconditioner import build_killing_form_preconditioner
    return build_killing_form_preconditioner(generators, center_reg=center_reg)


def create_param_groups(
    model: nn.Module,
    config: TrainingConfig,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Create parameter groups for multi-group optimization.

    Assigns different learning rates to belief parameters (mu, sigma),
    gauge frame parameters (phi -- Lie algebra elements), and standard
    neural network parameters (attention, ffn, output). This exploits
    natural gradient structure on the statistical manifold.

    Args:
        model: The gauge transformer (or standard transformer) model.
        config: TrainingConfig with per-group learning rates and weight decay.
        verbose: If True, print parameter group information.

    Returns:
        List of parameter group dicts suitable for torch.optim.AdamW.
    """
    # Collect parameters by type
    mu_params = []
    sigma_params = []
    phi_params = []
    omega_params = []  # Direct GL(K) gauge frame matrices
    attention_params = []
    ffn_params = []
    no_decay_params = []  # Norms, biases, and VFE hyperparameters (no weight decay)
    output_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Mean embeddings (matches both mu_prior and prior_mu naming conventions)
        if 'mu_embed' in name or 'mu_prior' in name or 'prior_mu' in name:
            mu_params.append(param)
        # Covariance embeddings (matches log_sigma, sigma_prior, log_prior_sigma, etc.)
        elif 'sigma_embed' in name or 'log_sigma' in name or 'sigma_prior' in name or 'prior_sigma' in name or 'log_prior' in name:
            sigma_params.append(param)
        # Direct Omega gauge frame embeddings (gauge_param='omega')
        elif 'omega_embed' in name:
            omega_params.append(param)
        # Lie algebra gauge frame embeddings (gauge_param='phi')
        elif 'phi_embed' in name or 'phi_prior' in name:
            phi_params.append(param)
        # Positional encoding (treat as gauge frames)
        elif 'pos_encoding' in name or 'position' in name:
            phi_params.append(param)
        # Attention mechanism
        elif 'attention' in name or 'attn' in name:
            attention_params.append(param)
        # Output projection
        elif 'out_proj' in name or 'lm_head' in name:
            output_params.append(param)
        # LayerNorm, biases, and VFE hyperparameters: never weight-decay
        elif 'norm' in name or 'bias' in name or 'raw_' in name or 'gate' in name or 'log_scale' in name:
            no_decay_params.append(param)
        # FFN (default for everything else)
        else:
            ffn_params.append(param)

    # Create parameter groups
    param_groups = []

    # Embedding weight decay = Level 3 hyper-prior: N(0, 1/(2·wd))
    # None → inherit from weight_decay; 0.0 → uninformative hyper-prior
    embed_wd = config.embed_weight_decay if config.embed_weight_decay is not None else config.weight_decay

    if mu_params:
        param_groups.append({
            'params': mu_params,
            'lr': config.mu_lr,
            'weight_decay': embed_wd,
            'name': 'mu_embed',
        })
        if verbose:
            print(f"  Parameter group 'mu_embed': {len(mu_params)} tensors @ lr={config.mu_lr}, wd={embed_wd}")

    if sigma_params:
        param_groups.append({
            'params': sigma_params,
            'lr': config.sigma_lr,
            'weight_decay': embed_wd,
            'name': 'sigma_embed',
        })
        if verbose:
            print(f"  Parameter group 'sigma_embed': {len(sigma_params)} tensors @ lr={config.sigma_lr}, wd={embed_wd}")

    if phi_params:
        param_groups.append({
            'params': phi_params,
            'lr': config.phi_lr,
            'weight_decay': embed_wd,
            'name': 'phi_embed',
        })
        if verbose:
            print(f"  Parameter group 'phi_embed': {len(phi_params)} tensors @ lr={config.phi_lr}, wd={embed_wd}")

    if omega_params:
        omega_lr = getattr(config, 'omega_lr', config.phi_lr)
        param_groups.append({
            'params': omega_params,
            'lr': omega_lr,
            'weight_decay': embed_wd,
            'name': 'omega_embed',
        })
        if verbose:
            print(f"  Parameter group 'omega_embed': {len(omega_params)} tensors @ lr={omega_lr}, wd={embed_wd}")

    if attention_params:
        param_groups.append({
            'params': attention_params,
            'lr': config.attention_lr,
            'weight_decay': config.weight_decay,
            'name': 'attention',
        })
        if verbose:
            print(f"  Parameter group 'attention': {len(attention_params)} tensors @ lr={config.attention_lr}")

    if ffn_params:
        param_groups.append({
            'params': ffn_params,
            'lr': config.ffn_lr,
            'weight_decay': config.weight_decay,
            'name': 'ffn',
        })
        if verbose:
            print(f"  Parameter group 'ffn': {len(ffn_params)} tensors @ lr={config.ffn_lr}")

    if no_decay_params:
        param_groups.append({
            'params': no_decay_params,
            'lr': config.ffn_lr,
            'weight_decay': 0.0,
            'name': 'no_decay',
        })
        if verbose:
            print(f"  Parameter group 'no_decay': {len(no_decay_params)} tensors @ lr={config.ffn_lr}, wd=0.0")

    if output_params:
        param_groups.append({
            'params': output_params,
            'lr': config.output_lr,
            'weight_decay': 0.0,  # Often tied to embeddings
            'name': 'output',
        })
        if verbose:
            print(f"  Parameter group 'output': {len(output_params)} tensors @ lr={config.output_lr}")

    return param_groups


def create_simple_param_groups(
    model: nn.Module,
    config: TrainingConfig,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Create simple 2-group parameter groups (decay vs no-decay).

    Args:
        model: The model to create parameter groups for
        config: Training configuration
        verbose: If True, print parameter group information

    Returns:
        List of parameter group dicts for torch.optim
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'bias' in name or 'norm' in name or 'embed' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

    param_groups = [
        {'params': decay_params, 'weight_decay': config.weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ]

    if verbose:
        print(f"  Parameter groups: {len(decay_params)} with decay, {len(no_decay_params)} without")

    return param_groups


def create_optimizer(
    model: nn.Module,
    config: TrainingConfig,
    verbose: bool = True,
) -> torch.optim.Optimizer:
    r"""Create optimizer with configurable type and parameter grouping.

    Optimizer types:
        'adamw': Standard AdamW. Diagonal Fisher via EMA of g².
        'riemannian_adam': AdamW + Killing metric on phi + Fisher on mu.
        'natural_gradient': Per-token K×K empirical Fisher blocks.

    Args:
        model: The model to optimize
        config: Training configuration (optimizer_type, use_param_groups, ...)
        verbose: If True, print optimizer information

    Returns:
        Configured optimizer
    """
    optimizer_type = getattr(config, 'optimizer_type', 'adamw')

    if config.use_param_groups:
        if verbose:
            print(f"Creating multi-group optimizer ({optimizer_type}):")
        param_groups = create_param_groups(model, config, verbose=verbose)
    else:
        if verbose:
            print(f"Creating simple optimizer ({optimizer_type}):")
        param_groups = create_simple_param_groups(model, config, verbose=verbose)

    base_kwargs = dict(
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
    )

    if optimizer_type == 'riemannian_adam':
        # Precompute Killing metric inverse from model generators
        killing_inv = None
        generators = getattr(model, 'generators', None)
        if generators is not None:
            killing_inv = compute_killing_metric_inv(generators)
            if verbose:
                n_gen = generators.shape[0]
                K = generators.shape[1]
                # Check if metric is approximately scalar (SO(N) case)
                diag_var = killing_inv.diag().var().item()
                off_diag = (killing_inv - torch.diag(killing_inv.diag())).abs().max().item()
                if off_diag < 1e-3 * killing_inv.diag().abs().mean().item():
                    scale = killing_inv.diag().mean().item()
                    print(f"  Killing metric: ~{scale:.4f}·I (SO-like, {n_gen} generators)")
                else:
                    cond = torch.linalg.cond(killing_inv).item()
                    print(f"  Killing metric: non-trivial ({n_gen}×{n_gen}), cond={cond:.1f}")
        else:
            if verbose:
                print("  Warning: No generators found; phi preconditioning disabled")

        optimizer = RiemannianAdamW(
            param_groups,
            model=model,
            killing_inv=killing_inv,
            **base_kwargs,
        )

    elif optimizer_type == 'natural_gradient':
        ema_decay = getattr(config, 'fisher_ema_decay', 0.95)
        damping = getattr(config, 'fisher_damping', 1e-2)

        # Estimate memory cost
        if verbose:
            total_fisher_params = 0
            for group in param_groups:
                for p in group['params']:
                    if p.dim() == 2 and p.shape[-1] >= 4:
                        V, K = p.shape
                        total_fisher_params += V * K * K
            mem_mb = total_fisher_params * 4 / (1024 ** 2)
            print(f"  Fisher memory: {mem_mb:.0f} MB ({total_fisher_params:,} floats)")
            print(f"  EMA decay: {ema_decay}, damping: {damping}")
            if mem_mb > 500:
                print(f"  Warning: Fisher storage is large ({mem_mb:.0f} MB). "
                      f"Consider reducing vocab_size or embed_dim.")

        optimizer = NaturalGradientOptimizer(
            param_groups,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            ema_decay=ema_decay,
            damping=damping,
        )

    else:
        # Default: standard AdamW
        optimizer = torch.optim.AdamW(
            param_groups,
            **base_kwargs,
        )

    return optimizer


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
    """
    Create learning rate scheduler.

    Args:
        optimizer: The optimizer to schedule
        config: Training configuration

    Returns:
        LR scheduler or None if constant
    """
    if config.lr_decay == 'constant':
        return None

    # Compute minimum decay ratio (fraction of peak LR at end of cosine).
    # In multi-group mode, each group decays to this fraction of its base LR.
    # Use the global learning_rate as denominator (the "reference" LR).
    # Clamp to avoid nonsensical ratios if min_lr >= learning_rate.
    min_ratio = min(config.min_lr / max(config.learning_rate, 1e-12), 1.0)

    def lr_lambda(step):
        # Warmup phase
        if step < config.warmup_steps:
            return step / max(1, config.warmup_steps)

        # Decay phase
        progress = (step - config.warmup_steps) / max(1, config.max_steps - config.warmup_steps)
        progress = min(1.0, progress)  # Clamp to [0, 1]

        if config.lr_decay == 'cosine':
            import math
            return min_ratio + \
                   0.5 * (1 - min_ratio) * \
                   (1 + math.cos(progress * math.pi))
        elif config.lr_decay == 'linear':
            return max(min_ratio, 1 - progress)
        else:
            return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
