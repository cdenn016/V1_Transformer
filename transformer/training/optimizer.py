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

    After preconditioning, per-group Riemannian trust region clipping is
    applied (when grad_clip > 0). Each group is clipped in its native metric:
        - phi: ||ξ||_K = sqrt(ξ^T K ξ) where K is the Killing form
        - mu: ||ξ||_F = sqrt(ξ^T Σ^{-1} ξ) (Fisher-Rao norm)
        - sigma: ||ξ||_F = ||∇f||_2 (constant Fisher metric)
        - others: Euclidean norm (standard clip_grad_norm_)
    This bounds the geodesic step size on each factor of the product manifold,
    making the trust radius geometrically meaningful regardless of coordinate chart.

    Args:
        params: Parameter groups from create_param_groups()
        model: Reference to GaugeTransformerLM for accessing sigma values
        killing_inv: Precomputed inverse Killing metric (n_gen, n_gen) or None
        killing_metric: Precomputed Killing metric (n_gen, n_gen) or None,
            needed for Riemannian norm computation on phi
        grad_clip: Trust radius δ for per-group Riemannian clipping (0 = disabled)
        **kwargs: Forwarded to torch.optim.AdamW (lr, betas, eps, weight_decay)
    """

    def __init__(
        self,
        params,
        model: nn.Module = None,
        killing_inv: Optional[torch.Tensor] = None,
        killing_metric: Optional[torch.Tensor] = None,
        grad_clip: float = 0.0,
        metric: str = 'killing',
        generators: Optional[torch.Tensor] = None,
        structure_constants: Optional[torch.Tensor] = None,
        gram: Optional[torch.Tensor] = None,
        pullback_series_order: int = 6,
        **kwargs,
    ):
        super().__init__(params, **kwargs)
        self._model = model
        self._killing_inv = killing_inv
        self._killing_metric = killing_metric
        self._grad_clip = grad_clip
        # NaN-grad guard telemetry
        self._nan_step_count = 0
        self._last_nan_param = None
        self._last_nan_step = None
        self._step_index = 0

        if metric not in ('killing', 'pullback'):
            raise ValueError(
                f"metric must be 'killing' or 'pullback', got {metric!r}"
            )
        self._metric = metric
        self._pullback_series_order = pullback_series_order

        # Pullback path: store generators + structure constants + gram for
        # per-token metric builds.  Required if metric='pullback'; optional
        # otherwise (constructor accepts but ignores).
        self._generators = generators
        self._structure_constants = structure_constants
        self._gram = gram
        if metric == 'pullback':
            if generators is None:
                raise ValueError(
                    "metric='pullback' requires `generators` to be passed to "
                    "RiemannianAdamW.__init__."
                )
            if structure_constants is None:
                from transformer.core.gauge_preconditioner import build_structure_constants
                self._structure_constants = build_structure_constants(generators)
            if gram is None:
                self._gram = torch.einsum('aij,bij->ab', generators, generators)

    @torch.no_grad()
    def step(self, closure=None):
        self._step_index += 1

        # 0. NaN/Inf grad guard: skip the entire AdamW step when any parameter
        #    has a non-finite gradient.  Preconditioning and AdamW's second-
        #    moment EMA (exp_avg_sq) blindly propagate NaN, so a single bad
        #    backward silently corrupts the optimizer state for every finite
        #    step that follows.  Skipping the whole step preserves AdamW
        #    step-count coherence (bias correction, scheduler).  Zeroing
        #    individual NaN grads is unsafe because exp_avg_sq then sees a
        #    "0 grad" sample and the bias-corrected LR explodes on the next
        #    finite step.
        _first_bad_name = None
        for group in self.param_groups:
            _gname = group.get('name', '')
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                if not torch.isfinite(p.grad).all():
                    _first_bad_name = f"{_gname}[{i}]"
                    break
            if _first_bad_name is not None:
                break
        if _first_bad_name is not None:
            self._nan_step_count += 1
            self._last_nan_param = _first_bad_name
            self._last_nan_step = self._step_index
            # Zero all grads so downstream state stays clean; do NOT call
            # super().step() -- AdamW would advance its per-param state.
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        p.grad = None
            return None

        # 1. Precondition: Euclidean → natural gradient
        for group in self.param_groups:
            name = group.get('name', '')

            if 'phi' in name or 'omega' in name:
                self._precondition_phi(group)
            elif 'mu' in name:
                self._precondition_mu(group)
            elif 'sigma' in name:
                self._precondition_sigma(group)

        # 2. Per-group Riemannian trust region clipping
        if self._grad_clip > 0:
            self._riemannian_clip()

        return super().step(closure)

    def _precondition_phi(self, group: dict) -> None:
        r"""Apply the inverse gauge metric to phi/omega gradients.

        Two metric options:

        - ``metric='killing'`` (default): ``grad ← grad @ killing_inv``.
          Position-independent; cheap fixed ``(n_gen, n_gen)`` matmul.
          Exact for the so(K) part of gl(K); approximate for the symmetric
          part.

        - ``metric='pullback'``: ``grad ← G(φ)^{-1} · grad`` per token.
          Gauge-natural: exact inverse of the metric pulled back from the
          Frobenius inner product on GL(K) through the exponential map.
          Expensive (one ``(n_gen, n_gen)`` solve per token per step) but
          theoretically correct — the M-step mirror of
          ``phi_natural_gradient='pullback'`` on the E-step.
        """
        if self._metric == 'killing':
            if self._killing_inv is None:
                return
            for p in group['params']:
                if p.grad is None:
                    continue
                dev = p.grad.device
                K_inv = self._killing_inv.to(device=dev, dtype=p.grad.dtype)
                p.grad = p.grad @ K_inv
            return

        # Pullback path: solve G(φ) · nat_grad = grad per token.
        from transformer.core.gauge_preconditioner import apply_pullback_natural_gradient
        for p in group['params']:
            if p.grad is None:
                continue
            dev = p.grad.device
            dtype = p.grad.dtype
            gens = self._generators.to(device=dev, dtype=dtype)
            struct = self._structure_constants.to(device=dev, dtype=dtype)
            gram = self._gram.to(device=dev, dtype=dtype)
            p.grad = apply_pullback_natural_gradient(
                grad_phi=p.grad,
                phi=p.data,
                generators=gens,
                structure_constants=struct,
                gram=gram,
                series_order=self._pullback_series_order,
            )

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
        r"""Retrieve current variance values for the mu Fisher preconditioner.

        Checks ``prior_bank`` first (active source when ``use_prior_bank=True``),
        then falls back to ``token_embed``. When ``use_prior_bank=True``,
        ``token_embed`` is frozen, so its sigma values are stale — the live
        sigma lives in ``prior_bank.log_prior_sigma`` or
        ``prior_bank.base_log_prior_sigma``.

        Supported layouts:

        **PriorBank** (checked first):
        - ``log_prior_sigma`` (V, K): per-token log-variance
          (``gauge_fixed_priors=False``).
        - ``base_log_prior_sigma`` (K,): shared base log-variance
          (``gauge_fixed_priors=True``).

        **TokenEmbedding** (fallback):
        - ``log_sigma_diag`` (V, K): per-token log-variance.
        - ``base_log_sigma_diag`` (K,): shared base log-variance.
        """
        # Check prior_bank first — it is the active sigma source when present
        pb = getattr(self._model, 'prior_bank', None)
        if pb is not None:
            if hasattr(pb, 'log_prior_sigma') and isinstance(pb.log_prior_sigma, nn.Parameter):
                return torch.exp(pb.log_prior_sigma.data).clamp(min=1e-6, max=10.0)
            if hasattr(pb, 'base_log_prior_sigma') and isinstance(pb.base_log_prior_sigma, nn.Parameter):
                return torch.exp(pb.base_log_prior_sigma.data).clamp(min=1e-6, max=10.0)

        # Fallback to token_embed (used when prior_bank is absent)
        embed = getattr(self._model, 'token_embed', None)
        if embed is None:
            return None
        if hasattr(embed, 'log_sigma_diag') and isinstance(embed.log_sigma_diag, nn.Parameter):
            return torch.exp(embed.log_sigma_diag.data).clamp(min=1e-6, max=10.0)
        if hasattr(embed, 'base_log_sigma_diag') and isinstance(embed.base_log_sigma_diag, nn.Parameter):
            return torch.exp(embed.base_log_sigma_diag.data).clamp(min=1e-6, max=10.0)
        return None

    def _riemannian_clip(self) -> None:
        r"""Per-group Riemannian trust region clipping.

        Clips the natural gradient ξ = G^{-1} ∇f in the Riemannian norm
        ||ξ||_G = sqrt(ξ^T G ξ), separately for each parameter group.
        This bounds the geodesic step size on each factor of the product
        manifold M_mu × M_sigma × M_phi × R^n.

        For phi (Killing metric K):
            ||ξ||_K^2 = Σ_v ξ_v^T K ξ_v  where ξ = K^{-1} g
            Equivalently = Σ_v g_v^T K^{-1} g_v (Mahalanobis norm of Euclidean grad)

        For mu (Fisher metric Σ^{-1}):
            ||ξ||_F^2 = Σ_v ξ_v^T Σ_v^{-1} ξ_v  where ξ = Σ g
            = Σ_v (Σ_v g_v)^T Σ_v^{-1} (Σ_v g_v) = Σ_v g_v^T Σ_v g_v
            For diagonal Σ: ||ξ||_F^2 = Σ_k ξ_k^2 / σ_k^2  (variance-weighted)

        For sigma (constant Fisher F = (1/2)I):
            ||ξ||_F^2 = Σ_v ξ_v^T (I/2) ξ_v = (1/2)||ξ||_2^2
            Since ξ = 2g: ||ξ||_F = sqrt(2) ||g||_2

        For Euclidean params: standard L2 norm.
        """
        sigma = self._get_sigma()

        for group in self.param_groups:
            name = group.get('name', '')
            graded = [p for p in group['params'] if p.grad is not None]
            if not graded:
                continue

            if ('phi' in name or 'omega' in name):
                # Riemannian norm selection.
                # metric='killing': ||ξ||_K = sqrt(Σ_v ξ_v^T K ξ_v) with the
                # fixed Killing metric K. One matmul over the group.
                # metric='pullback': ||ξ||_G(φ) = sqrt(Σ_v ξ_v^T G(φ_v) ξ_v)
                # with the per-token pullback metric G(φ_v). Theoretically
                # exact; pays one (n_gen, n_gen) build per token per step.
                if self._metric == 'killing' and self._killing_metric is not None:
                    K = self._killing_metric.to(graded[0].device, graded[0].dtype)
                    sq_norm = 0.0
                    for p in graded:
                        xi = p.grad  # already preconditioned: ξ = K^{-1} g
                        sq_norm += (xi @ K * xi).sum().item()
                    riem_norm = sq_norm ** 0.5
                    if riem_norm > self._grad_clip:
                        scale = self._grad_clip / (riem_norm + 1e-8)
                        for p in graded:
                            p.grad.mul_(scale)
                elif self._metric == 'pullback':
                    from transformer.core.gauge_preconditioner import build_pullback_metric_tensor
                    for p in graded:
                        if p.grad is None:
                            continue
                        dev = p.grad.device
                        dtype = p.grad.dtype
                        gens = self._generators.to(device=dev, dtype=dtype)
                        struct = self._structure_constants.to(device=dev, dtype=dtype)
                        gram = self._gram.to(device=dev, dtype=dtype)
                        # Build per-token metric G(φ_v): (..., n_gen, n_gen)
                        G_phi = build_pullback_metric_tensor(
                            phi=p.data,
                            generators=gens,
                            structure_constants=struct,
                            gram=gram,
                            series_order=self._pullback_series_order,
                        )
                        xi = p.grad
                        # Per-token Riemannian norm-squared ξ^T G(φ) ξ.
                        norm_sq_per = torch.einsum(
                            '...i,...ij,...j->...', xi, G_phi, xi,
                        )
                        norm_per = norm_sq_per.clamp(min=0.0).sqrt()
                        # Clip each row independently in its native metric.
                        scale = (self._grad_clip / (norm_per + 1e-8)).clamp(max=1.0)
                        # Broadcast scale over the n_gen axis.
                        p.grad.mul_(scale.unsqueeze(-1))

            elif 'mu' in name and sigma is not None:
                # Riemannian norm: ||ξ||_{Σ^{-1}} = sqrt(Σ_v ξ_v^T Σ_v^{-1} ξ_v)
                # ξ = Σ g is stored in p.grad. So ξ_v / σ_v = g_v (original grad).
                sq_norm = 0.0
                for p in graded:
                    if p.grad.shape == sigma.shape:
                        # ξ_v^T Σ^{-1} ξ_v = ξ_v^2 / σ_v^2 (diagonal Σ, σ_v^2 = variance)
                        sq_norm += (p.grad ** 2 / sigma.clamp(min=1e-6)).sum().item()
                    else:
                        sq_norm += (p.grad ** 2).sum().item()
                riem_norm = sq_norm ** 0.5
                if riem_norm > self._grad_clip:
                    scale = self._grad_clip / (riem_norm + 1e-8)
                    for p in graded:
                        p.grad.mul_(scale)

            elif 'sigma' in name:
                # Fisher = (1/2)I. ξ = 2g. ||ξ||_F^2 = (1/2)||ξ||_2^2 = 2||g||_2^2.
                # Since ξ is stored in p.grad: ||ξ||_F = ||p.grad||_2 / sqrt(2).
                sq_norm = sum((p.grad ** 2).sum().item() for p in graded)
                riem_norm = (sq_norm / 2.0) ** 0.5  # ||ξ||_F = ||ξ||_2 / sqrt(2)
                if riem_norm > self._grad_clip:
                    scale = self._grad_clip / (riem_norm + 1e-8)
                    for p in graded:
                        p.grad.mul_(scale)

            else:
                # Euclidean params (attention, output, ffn, no_decay)
                torch.nn.utils.clip_grad_norm_(graded, self._grad_clip)


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

    Fisher blocks are only allocated for named embedding parameter groups
    (mu_embed, sigma_embed, phi_embed, omega_embed, sign_embed). Non-embedding
    groups (attention, ffn, output, no_decay) and unnamed groups fall back to
    standard gradient descent with weight decay (no Fisher tracking).
    Fisher blocks are always stored in fp32 regardless of model precision,
    for numerical stability of the outer product and linear solve.

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

    # Parameter groups that receive per-token Fisher treatment.
    # All other groups (attention, ffn, output, no_decay, unnamed) get plain SGD.
    _EMBEDDING_GROUPS = frozenset({
        'mu_embed', 'sigma_embed', 'phi_embed', 'omega_embed', 'sign_embed',
    })

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
        group_natural_clipped_sq: Dict[str, float] = {}
        group_clip_frac_sum: Dict[str, float] = {}
        group_clip_count: Dict[str, int] = {}

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

                # For named embedding groups with 2D parameters: per-row Fisher blocks
                is_embedding_group = gname in self._EMBEDDING_GROUPS
                if is_embedding_group and grad.dim() == 2 and grad.shape[-1] >= 4:
                    eucl_sq, nat_sq, nat_clip_sq, clip_frac = self._natural_step_embedding(
                        p, grad, state, lr)
                    group_euclidean_sq[gname] = group_euclidean_sq.get(gname, 0.0) + eucl_sq
                    group_natural_sq[gname] = group_natural_sq.get(gname, 0.0) + nat_sq
                    group_natural_clipped_sq[gname] = group_natural_clipped_sq.get(gname, 0.0) + nat_clip_sq
                    group_clip_frac_sum[gname] = group_clip_frac_sum.get(gname, 0.0) + clip_frac
                    group_clip_count[gname] = group_clip_count.get(gname, 0) + 1
                else:
                    # Fallback: plain gradient descent for 1D / small params
                    p.add_(grad, alpha=-lr)
                    g_sq = grad.norm().item() ** 2
                    group_euclidean_sq[gname] = group_euclidean_sq.get(gname, 0.0) + g_sq
                    group_natural_sq[gname] = group_natural_sq.get(gname, 0.0) + g_sq
                    group_natural_clipped_sq[gname] = group_natural_clipped_sq.get(gname, 0.0) + g_sq

        # Store L2 norms per group
        all_gnames = set(list(group_euclidean_sq.keys()) + list(group_natural_sq.keys()))
        self._grad_norms = {}
        for gname in all_gnames:
            count = group_clip_count.get(gname, 0)
            self._grad_norms[gname] = {
                'euclidean': math.sqrt(group_euclidean_sq.get(gname, 0.0)),
                'natural': math.sqrt(group_natural_sq.get(gname, 0.0)),
                'natural_clipped': math.sqrt(group_natural_clipped_sq.get(gname, 0.0)),
                'clip_fraction': group_clip_frac_sum.get(gname, 0.0) / count if count > 0 else 0.0,
            }

        return loss

    def _natural_step_embedding(
        self,
        param: torch.Tensor,    # (V, K)
        grad: torch.Tensor,     # (V, K)
        state: dict,
        lr: float,
    ) -> Tuple[float, float, float, float]:
        """Per-token natural gradient step with block-diagonal Fisher.

        Returns:
            (euclidean_norm_sq, natural_norm_sq, natural_clipped_norm_sq, clip_fraction):
            Squared L2 norms of the raw gradient, the Fisher-preconditioned
            gradient (pre-clip), the clipped natural gradient (post-clip), and
            the fraction of active tokens where max_ratio clipping fired.
        """
        V, K = grad.shape
        device = grad.device
        dtype = grad.dtype

        # Initialize Fisher blocks on first call (always fp32 for numerical
        # stability of outer product and linear solve under AMP)
        if 'fisher' not in state:
            state['fisher'] = torch.zeros(V, K, K, device=device, dtype=torch.float32)
            state['step'] = 0
        elif state['fisher'].dtype != torch.float32:
            # Handle checkpoint loaded with non-fp32 Fisher
            state['fisher'] = state['fisher'].float()
        state['step'] += 1

        fisher = state['fisher']

        # Find tokens with nonzero gradient (only batch tokens)
        grad_norm = grad.abs().sum(dim=-1)  # (V,)
        nz_mask = grad_norm > 0
        if not nz_mask.any():
            eucl_sq = grad.norm().item() ** 2
            return eucl_sq, eucl_sq, eucl_sq, 0.0

        nz_idx = nz_mask.nonzero(as_tuple=True)[0]  # (n_active,)
        g = grad[nz_idx]  # (n_active, K)

        # Upcast to fp32 for numerical stability of outer product and solve
        g_f32 = g.float()  # (n_active, K) in fp32

        # Update Fisher EMA: F = (1-ρ)F + ρ g gᵀ
        outer = g_f32.unsqueeze(-1) * g_f32.unsqueeze(-2)  # (n_active, K, K)
        rho = self.ema_decay
        # Bias correction for early steps: use Adam-style correction factor
        # to compensate for cold-start (Fisher initialized at zero).
        # Without correction, the early Fisher is biased toward zero,
        # making the natural gradient ≈ Euclidean gradient.
        step = state['step']
        bias_correction = 1.0 - (1.0 - rho) ** step
        fisher[nz_idx] = (1.0 - rho) * fisher[nz_idx] + rho * outer

        # Natural gradient: ng = (F̂ + λI)⁻¹ g  where F̂ = F / bias_correction
        F_active = fisher[nz_idx] / bias_correction  # Bias-corrected Fisher (fp32)
        I_K = torch.eye(K, device=device, dtype=torch.float32)
        F_damped = F_active + self.damping * I_K

        # Solve F_damped @ ng = g  (batched K×K linear solve, fp32)
        ng = torch.linalg.solve(F_damped, g_f32.unsqueeze(-1)).squeeze(-1)  # (n_active, K)

        # Record norms before clipping
        eucl_sq = g_f32.norm().item() ** 2
        nat_sq = ng.norm().item() ** 2

        # Clip natural gradient to prevent explosion from ill-conditioned Fisher
        ng_norm = ng.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        g_norm = g_f32.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        max_ratio = 10.0  # Natural gradient shouldn't be >10x the Euclidean gradient
        scale = torch.clamp(max_ratio * g_norm / ng_norm, max=1.0)
        ng = ng * scale

        # Post-clip diagnostics
        nat_clipped_sq = ng.norm().item() ** 2
        n_clipped = (scale.squeeze(-1) < 1.0).sum().item()
        clip_frac = n_clipped / g.shape[0]

        # Cast back to param dtype before applying update
        param[nz_idx] -= lr * ng.to(dtype=dtype)

        return eucl_sq, nat_sq, nat_clipped_sq, clip_frac

    def get_grad_norms(self) -> Dict[str, Dict[str, float]]:
        r"""Return per-group gradient norms from the last step() call.

        Returns:
            Dict mapping group name → {
                'euclidean': ‖g‖ (raw Euclidean gradient norm),
                'natural': ‖F⁻¹g‖ (pre-clip natural gradient norm),
                'natural_clipped': ‖clip(F⁻¹g)‖ (post-clip, the actual applied step),
                'clip_fraction': fraction of active tokens where max_ratio clipping fired.
            }.
            For non-embedding params (plain SGD fallback), euclidean == natural == natural_clipped
            and clip_fraction == 0.
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

    sign_params = []  # O(K) reflection sign vectors (learnable_reflection=True)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # O(K) reflection sign logits (learnable_reflection=True)
        # Must be checked BEFORE mu_embed to avoid false match on 'embed'
        if 'sign_logit' in name:
            sign_params.append(param)
        # Mean embeddings (matches both mu_prior and prior_mu naming conventions).
        # `name.endswith('base_mu')` catches `token_embed.base_mu`, the
        # single-tensor prior mean used when `gauge_fixed_priors=True`
        # (embeddings.py:227).  Without this anchor the parameter falls
        # through every subsequent branch and lands in the `ffn` group at
        # `M_vfe_hyperparam_lr`, silently dropping `M_mu_p_lr`.
        elif (
            'mu_embed' in name
            or 'mu_prior' in name
            or 'prior_mu' in name
            or name.endswith('base_mu')
        ):
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
        # Learnable per-head kappa (temperature) — no weight decay.
        # Must be checked BEFORE 'attention' to avoid false match on
        # attention.log_kappa_per_head being routed to attention_params.
        elif 'log_kappa' in name:
            no_decay_params.append(param)
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
    non_embed_wd = getattr(config, 'non_embed_weight_decay', getattr(config, 'weight_decay', 0.01))
    embed_wd = config.embed_weight_decay if config.embed_weight_decay is not None else non_embed_wd

    if mu_params:
        param_groups.append({
            'params': mu_params,
            'lr': config.M_mu_p_lr,
            'weight_decay': embed_wd,
            'name': 'mu_embed',
        })
        if verbose:
            print(f"  Parameter group 'mu_embed': {len(mu_params)} tensors @ lr={config.M_mu_p_lr}, wd={embed_wd}")

    if sigma_params:
        param_groups.append({
            'params': sigma_params,
            'lr': config.M_sigma_p_lr,
            'weight_decay': embed_wd,
            'name': 'sigma_embed',
        })
        if verbose:
            print(f"  Parameter group 'sigma_embed': {len(sigma_params)} tensors @ lr={config.M_sigma_p_lr}, wd={embed_wd}")

    if phi_params:
        param_groups.append({
            'params': phi_params,
            'lr': config.M_phi_lr,
            'weight_decay': embed_wd,
            'name': 'phi_embed',
        })
        if verbose:
            print(f"  Parameter group 'phi_embed': {len(phi_params)} tensors @ lr={config.M_phi_lr}, wd={embed_wd}")

    if omega_params:
        omega_lr = getattr(config, 'omega_lr', config.M_phi_lr)
        # omega_embed is initialized near identity (I + eps·randn). The generic
        # embed_wd decays toward zero, which is the SINGULAR matrix — exactly
        # the direction we need to avoid. Decay-to-identity would require a
        # custom optimizer step; for now use wd=0 and rely on
        # TrainingConfig.omega_det_penalty (V9) for determinant control.
        param_groups.append({
            'params': omega_params,
            'lr': omega_lr,
            'weight_decay': 0.0,
            'name': 'omega_embed',
        })
        if verbose:
            print(f"  Parameter group 'omega_embed': {len(omega_params)} tensors @ lr={omega_lr}, wd=0.0 (avoid decay-to-singular)")

    if sign_params:
        param_groups.append({
            'params': sign_params,
            'lr': config.M_mu_p_lr,
            'weight_decay': embed_wd,
            'name': 'sign_embed',
        })
        if verbose:
            print(f"  Parameter group 'sign_embed': {len(sign_params)} tensors @ lr={config.M_mu_p_lr}, wd={embed_wd}")

    if attention_params:
        param_groups.append({
            'params': attention_params,
            'lr': config.M_attention_lr,
            'weight_decay': non_embed_wd,
            'name': 'attention',
        })
        if verbose:
            print(f"  Parameter group 'attention': {len(attention_params)} tensors @ lr={config.M_attention_lr}")

    if ffn_params:
        param_groups.append({
            'params': ffn_params,
            'lr': config.M_vfe_hyperparam_lr,
            'weight_decay': non_embed_wd,
            'name': 'ffn',
        })
        if verbose:
            print(f"  Parameter group 'ffn': {len(ffn_params)} tensors @ lr={config.M_vfe_hyperparam_lr}")

    if no_decay_params:
        param_groups.append({
            'params': no_decay_params,
            'lr': config.M_vfe_hyperparam_lr,
            'weight_decay': 0.0,
            'name': 'no_decay',
        })
        if verbose:
            print(f"  Parameter group 'no_decay': {len(no_decay_params)} tensors @ lr={config.M_vfe_hyperparam_lr}, wd=0.0")

    if output_params:
        param_groups.append({
            'params': output_params,
            'lr': config.M_output_lr,
            'weight_decay': 0.0,
            'name': 'output',
        })
        if verbose:
            print(f"  Parameter group 'output': {len(output_params)} tensors @ lr={config.M_output_lr}")

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
        {'params': decay_params, 'weight_decay': config.non_embed_weight_decay},
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
        # Precompute Killing metric and its inverse from model generators
        killing_inv = None
        killing_metric = None
        generators = getattr(model, 'generators', None)
        if generators is not None:
            from transformer.core.gauge_preconditioner import build_killing_form_preconditioner
            killing_inv, killing_metric = build_killing_form_preconditioner(
                generators, return_both=True)
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

        grad_clip = getattr(config, 'grad_clip', 0.0)
        metric = getattr(config, 'phi_optimizer_metric', 'killing')
        if metric not in ('killing', 'pullback'):
            raise ValueError(
                f"phi_optimizer_metric must be 'killing' or 'pullback', "
                f"got {metric!r}"
            )

        # Pullback path precomputes structure constants and gram from
        # generators; theoretically exact gauge-natural metric, position-
        # dependent, expensive per-token build.  Killing path uses the
        # precomputed fixed (n_gen, n_gen) metric — cheap and default.
        structure_constants = None
        gram = None
        if metric == 'pullback':
            if generators is None:
                raise ValueError(
                    "phi_optimizer_metric='pullback' requires model.generators "
                    "to be populated."
                )
            from transformer.core.gauge_preconditioner import build_structure_constants
            structure_constants = build_structure_constants(generators)
            gram = torch.einsum('aij,bij->ab', generators, generators)
            if verbose:
                print(
                    f"  RAdamW metric: 'pullback' (per-token (n_gen,n_gen) "
                    f"build + solve; gauge-natural, expensive)"
                )

        optimizer = RiemannianAdamW(
            param_groups,
            model=model,
            killing_inv=killing_inv,
            killing_metric=killing_metric,
            grad_clip=grad_clip,
            metric=metric,
            generators=generators,
            structure_constants=structure_constants,
            gram=gram,
            pullback_series_order=getattr(config, 'pullback_series_order', 6),
            **base_kwargs,
        )
        if verbose and grad_clip > 0:
            print(f"  Riemannian trust region clipping: δ={grad_clip} (metric={metric})")

    elif optimizer_type == 'natural_gradient':
        ema_decay = getattr(config, 'fisher_ema_decay', 0.95)
        damping = getattr(config, 'fisher_damping', 1e-2)

        # Estimate memory cost (only embedding groups get Fisher blocks)
        if verbose:
            embed_groups = NaturalGradientOptimizer._EMBEDDING_GROUPS
            total_fisher_params = 0
            for group in param_groups:
                if group.get('name', 'unnamed') not in embed_groups:
                    continue
                for p in group['params']:
                    if p.dim() == 2 and p.shape[-1] >= 4:
                        V, K = p.shape
                        total_fisher_params += V * K * K
            mem_mb = total_fisher_params * 4 / (1024 ** 2)  # Always fp32
            print(f"  Fisher memory: {mem_mb:.0f} MB ({total_fisher_params:,} floats)")
            print(f"  EMA decay: {ema_decay}, damping: {damping}")
            if mem_mb > 500:
                print(f"  Warning: Fisher storage is large ({mem_mb:.0f} MB). "
                      f"Consider reducing vocab_size or embed_dim.")

        optimizer = NaturalGradientOptimizer(
            param_groups,
            lr=config.learning_rate,
            weight_decay=config.non_embed_weight_decay,
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

