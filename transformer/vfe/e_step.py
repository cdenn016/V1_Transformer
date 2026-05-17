"""
VFEEStep: iterative VFE minimization inner loop.

Replaces the 2,893-line variational_ffn.py with a single clear E-step path.
No EM mode branching, no DEQ, no closed-form, no hebbian.

Law 1 enforced: forward() has no targets parameter. Target leakage is
structurally impossible.

Law 2 enforced: all transport goes through fused block-diagonal kernels
which internally compute Omega @ Sigma @ Omega^T.
"""

from __future__ import annotations

import math
import warnings
from typing import Optional, Callable, List, Tuple, TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from transformer.vfe.config import VFEConfig

from transformer.core.types import BeliefState
from transformer.core.vfe_gradients import (
    compute_vfe_gradients_gpu,
    compute_natural_gradient_gpu,
    _compute_rope_full_gauge_gradient_per_head,
)
from transformer.core.vfe_utils import (
    retract_sigma_e_step,
    _retract_phi,
)
from transformer.core.gauge_preconditioner import (
    build_cartan_projector,
    build_killing_form_preconditioner,
)
from transformer.core.phi_evolution import precondition_phi_gradient
from transformer.vfe.attention import (
    compute_kl_attention,
    compute_gauge_transport,
)


class VFEEStep(nn.Module):
    r"""Iterative VFE minimization on :math:`(\mu, \Sigma, \phi)`.

    Each iteration:
        1. Compute transport and KL attention (beta recomputed each step)
        2. Compute VFE gradients:
           :math:`F = \alpha\,\mathrm{KL}(q\|p) + \lambda_{\text{align}}\sum\beta\mathrm{KL} + \lambda_{\text{soft}}C(q,\phi)`
        3. Optional: add active inference gradients (injected callback)
        4. Natural gradient projection: :math:`\tilde{\nabla}_\mu = \Sigma \nabla_\mu`
        5. Mean update: :math:`\mu \gets \mu - \eta_\mu \tilde{\nabla}_\mu`
        6. SPD retraction for sigma
        7. Phi update with Killing form preconditioning

    Args:
        cfg: VFEConfig.
        generators: ``(n_gen, K, K)`` Lie algebra generators.
    """

    def __init__(self, cfg: 'VFEConfig', generators: torch.Tensor) -> None:
        super().__init__()
        if cfg.n_e_steps < 1:
            raise ValueError(
                f"VFEConfig.n_e_steps must be >= 1 (got {cfg.n_e_steps}); "
                "a zero-iteration E-step turns every layer into an identity."
            )
        self.n_e_steps = cfg.n_e_steps
        self.alpha = cfg.alpha
        self.E_learnable_alpha = cfg.E_learnable_alpha
        if cfg.E_learnable_alpha:
            K = cfg.embed_dim
            alpha_init = max(cfg.alpha, 0.01)
            b0_init = 1.0
            c0_init = alpha_init * b0_init
            self.raw_c0 = nn.Parameter(torch.full((K,), self._softplus_inverse(c0_init)))
            self.raw_b0 = nn.Parameter(torch.full((K,), self._softplus_inverse(b0_init)))
        self.lambda_align = cfg.lambda_align
        self.lambda_soft = cfg.lambda_soft
        self.include_attention_entropy = cfg.include_attention_entropy
        self.embed_dim = cfg.embed_dim
        self._learnable_kappa = cfg.learnable_kappa
        if cfg.learnable_kappa:
            self.log_kappa = nn.Parameter(torch.tensor(math.log(cfg.kappa)))
            self._kappa_init = cfg.kappa
        else:
            self.kappa = cfg.kappa
        self.e_mu_lr = cfg.e_mu_lr
        self.e_sigma_lr = cfg.e_sigma_lr
        self.e_sigma_q_trust = cfg.e_sigma_q_trust
        self.e_phi_lr = cfg.e_phi_lr
        self.sigma_max = cfg.sigma_max
        self.diagonal_covariance = cfg.diagonal_covariance
        self.isotropic_covariance = cfg.isotropic_covariance
        self.exact_diagonal_transport = cfg.exact_diagonal_transport
        self.alpha_divergence = cfg.alpha_divergence
        self.irrep_dims: List[int] = cfg.irrep_dims
        self.use_rope = cfg.use_rope
        self.rope_base = cfg.rope_base
        self.rope_full_gauge = cfg.rope_full_gauge
        self.enforce_orthogonal = cfg.enforce_orthogonal
        self.mask_self_attention = cfg.mask_self_attention
        self.gauge_covariant_ridge = getattr(cfg, 'gauge_covariant_ridge', False)
        self.track_layer_diagnostics = getattr(cfg, 'track_layer_diagnostics', False)
        self.gauge_group = cfg.gauge_group
        self.phi_preconditioner_mode = cfg.phi_preconditioner
        self.phi_project_slk = cfg.phi_project_slk
        self.phi_trace_clamp = cfg.phi_trace_clamp

        if not isinstance(generators, torch.Tensor):
            generators = torch.from_numpy(generators).float()
        self.register_buffer('generators', generators)

        # Build phi preconditioner (Killing metric or Cartan projector)
        _phi_prec = None
        if cfg.phi_preconditioner == 'cartan':
            _phi_prec = build_cartan_projector(generators)
        elif cfg.phi_preconditioner == 'killing':
            _phi_prec = build_killing_form_preconditioner(generators)
        if _phi_prec is not None:
            self.register_buffer('_phi_preconditioner', _phi_prec)
        else:
            self._phi_preconditioner = None

    @staticmethod
    def _softplus_inverse(x: float) -> float:
        """Inverse of softplus: returns y such that softplus(y) = x."""
        if x > 20:
            return x
        return math.log(math.exp(x) - 1.0)

    def get_bayesian_alpha(
        self,
        mu_q: torch.Tensor,
        mu_p: torch.Tensor,
        sigma_q: torch.Tensor,
        sigma_p: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        r"""Per-dimension Bayesian precision: :math:`\alpha_k = c_0 / (b_0 + \mathrm{KL}_k)`.

        Each belief dimension k gets its own precision via Gamma-Normal
        conjugacy, so different irrep blocks can learn different
        regularization curves.

        Returns:
            alpha: ``(B, N, K)`` per-dimension precision.
        """
        import torch.nn.functional as F_fn
        c0 = F_fn.softplus(self.raw_c0)  # (K,)
        b0 = F_fn.softplus(self.raw_b0)  # (K,)

        delta_mu = mu_q - mu_p  # (B, N, K)

        # Full-cov: extract diagonal for per-dimension KL decomposition
        if sigma_q.dim() == 4:
            sigma_q = torch.diagonal(sigma_q, dim1=-2, dim2=-1)
        if sigma_p.dim() == 4:
            sigma_p = torch.diagonal(sigma_p, dim1=-2, dim2=-1)

        sigma_p_safe = sigma_p.clamp(min=eps)
        sigma_q_safe = sigma_q.clamp(min=eps)

        # Per-dimension KL contributions (diagonal covariance)
        trace_k = sigma_q_safe / sigma_p_safe
        mahal_k = delta_mu ** 2 / sigma_p_safe
        logdet_k = torch.log(sigma_p_safe) - torch.log(sigma_q_safe)
        kl_k = 0.5 * (trace_k + mahal_k - 1 + logdet_k).clamp(min=0.0)  # (B, N, K)

        return c0 / (b0 + kl_k)  # (B, N, K)

    @property
    def effective_kappa(self):
        r"""Resolve kappa: learned (clamped) or fixed scalar."""
        if self._learnable_kappa:
            k = torch.exp(self.log_kappa)
            k0 = self._kappa_init
            return k.clamp(min=0.5 * k0, max=2.0 * k0)
        return self.kappa

    def forward(
        self,
        beliefs: BeliefState,
        priors: BeliefState,
        mask: Optional[torch.Tensor] = None,
        active_inference_fn: Optional[Callable] = None,
    ) -> BeliefState:
        r"""Run the iterative E-step.

        **Law 1 enforced**: No ``targets`` parameter exists. Target leakage
        is structurally impossible.

        Args:
            beliefs: Current Gaussian beliefs :math:`(\mu, \Sigma, \phi)`.
            priors: Layer priors :math:`(\mu_p, \Sigma_p, \phi_p)`.
            mask: ``(B, N, N)`` causal mask.
            active_inference_fn: Optional callback ``(mu, sigma) -> (grad_mu, grad_sigma)``.

        Returns:
            Updated BeliefState after E-step convergence.
        """
        mu = beliefs.mu
        sigma = beliefs.sigma
        phi = beliefs.phi
        mu_p = priors.mu
        sigma_p = priors.sigma

        is_diagonal = self.diagonal_covariance
        eps = 1e-6

        # Diagnostics buffer: populated on last iteration for the trainer to read
        self._last_diagnostics = {}

        # Cache effective_kappa once per forward — the property runs
        # torch.exp(log_kappa).clamp(...) and was previously called 4× per
        # E-step iteration (plus once inside _update_phi).
        _kappa = self.effective_kappa

        # Hoist iter-independent softplus(raw_c0) out of the loop.
        alpha_c0_full_hoisted: Optional[torch.Tensor] = None
        if self.E_learnable_alpha:
            import torch.nn.functional as F_fn
            alpha_c0_full_hoisted = F_fn.softplus(self.raw_c0)

        # Bogacz 2017 invariant: F(t+1) <= F(t) for correct E-step descent.
        # Track the per-iteration scalar free energy so silent divergence is
        # surfaced as a warning instead of corrupting training without signal.
        # Diagonal covariance only — full-cov KL(q||p) requires log|det| and
        # is skipped here (monotonicity still holds, just not monitored).
        # Tolerance is (rel, abs) combined: warn only on violations that
        # exceed BOTH relative and absolute floors.  Finite-step natural
        # gradient can overshoot in a single step without implying
        # divergence; we flag only persistent or large excursions.
        f_history: List[float] = []
        f_prev: Optional[float] = None
        f_monotone_rel_tol = 0.05   # 5% of current |F|
        f_monotone_abs_tol = 1e-2

        for t in range(self.n_e_steps):
            # 1. Compute transport and KL attention
            block_exp_pairs = compute_gauge_transport(
                phi, self.generators, self.irrep_dims,
                enforce_orthogonal=self.enforce_orthogonal,
            )

            # Bayesian adaptive alpha: α_k = c0/(b0 + KL_k) per dimension
            alpha_eff = self.alpha
            # alpha_c0_full is the per-dim concentration parameter c0 = softplus(raw_c0).
            # When E_learnable_alpha=True, alpha is a function of (μ,σ) via KL, so the
            # gradient of α·KL wrt (μ,σ) carries a product-rule term −(α²/c0)·KL·(dKL/dθ);
            # compute_vfe_gradients_gpu and the rope helper apply this correction iff
            # alpha_c0 is passed in. Without it, the descent direction is biased.
            alpha_c0_full = alpha_c0_full_hoisted
            if self.E_learnable_alpha:
                alpha_eff = self.get_bayesian_alpha(mu, mu_p, sigma, sigma_p)

            # rope_full_gauge: per-head autograd path that rotates BOTH μ and Σ.
            # NOTE: vfe currently treats 'vfe_only' and 'both' identically;
            # differentiating them requires splitting this branch.
            if (
                self.rope_full_gauge != 'off'
                and self.use_rope
                and not torch.is_inference_mode_enabled()
                and torch.is_grad_enabled()
            ):
                grad_mu_parts = []
                grad_sigma_parts = []
                beta = None
                block_start = 0
                for h, d_h in enumerate(self.irrep_dims):
                    block_end = block_start + d_h
                    mu_h = mu[:, :, block_start:block_end]
                    sigma_h = sigma[:, :, block_start:block_end] if is_diagonal else sigma[:, :, block_start:block_end, block_start:block_end]
                    mu_p_h = mu_p[:, :, block_start:block_end]
                    sigma_p_h = sigma_p[:, :, block_start:block_end] if is_diagonal else sigma_p[:, :, block_start:block_end, block_start:block_end]
                    _head_bep = [block_exp_pairs[h]]

                    beta_h, gm_h, gs_h = _compute_rope_full_gauge_gradient_per_head(
                        mu_h=mu_h, sigma_h=sigma_h,
                        mu_p_h=mu_p_h, sigma_p_h=sigma_p_h,
                        phi=phi, gen_h=self.generators[:, block_start:block_end, block_start:block_end],
                        alpha=alpha_eff if not isinstance(alpha_eff, torch.Tensor) else alpha_eff[:, :, block_start:block_end],
                        lambda_belief=self.lambda_align,
                        # Envelope identity: at the softmax fixed point of beta,
                        # the manuscript F gradient is just sum_j beta * dKL/dtheta
                        # — the softmax-coupling term sum_j KL * dbeta/dtheta
                        # cancels exactly against the entropy-gradient term
                        # tau * sum_j log(beta) * dbeta/dtheta. So when the
                        # entropy term is included in F, pass lambda_softmax=0.
                        lambda_softmax=0.0 if self.include_attention_entropy else self.lambda_soft,
                        kappa=_kappa,
                        eps=eps,
                        rope_base=self.rope_base,
                        d_h=d_h,
                        cached_block_exp_pairs=_head_bep,
                        enforce_orthogonal=self.enforce_orthogonal,
                        mask=mask,
                        mask_self_attention=self.mask_self_attention,
                        gauge_covariant_ridge=self.gauge_covariant_ridge,
                        alpha_c0=(alpha_c0_full[block_start:block_end]
                                  if alpha_c0_full is not None else None),
                    )
                    grad_mu_parts.append(gm_h)
                    grad_sigma_parts.append(gs_h)
                    if beta is None:
                        beta = beta_h
                    block_start = block_end

                grad_mu = torch.cat(grad_mu_parts, dim=-1)
                if is_diagonal:
                    grad_sigma = torch.cat(grad_sigma_parts, dim=-1)
                else:
                    K = mu.shape[-1]
                    grad_sigma = torch.zeros_like(sigma)
                    block_start = 0
                    for h, d_h in enumerate(self.irrep_dims):
                        block_end = block_start + d_h
                        grad_sigma[:, :, block_start:block_end, block_start:block_end] = grad_sigma_parts[h]
                        block_start = block_end
                kl_matrix = beta  # beta_h from last head (for diagnostics)
            else:
                # Standard path: fused attention + gradient computation
                beta, kl_matrix = compute_kl_attention(
                    mu, sigma, phi, self.generators,
                    self.irrep_dims, _kappa, mask,
                    use_rope=self.use_rope,
                    rope_base=self.rope_base,
                    cached_block_exp_pairs=block_exp_pairs,
                    enforce_orthogonal=self.enforce_orthogonal,
                    mask_self_attention=self.mask_self_attention,
                    exact_diagonal_transport=self.exact_diagonal_transport,
                )

                grad_mu, grad_sigma = compute_vfe_gradients_gpu(
                    mu_q=mu,
                    sigma_q=sigma,
                    mu_p=mu_p,
                    sigma_p=sigma_p,
                    beta=beta,
                    phi=phi,
                    generators=self.generators,
                    alpha=alpha_eff,
                    alpha_c0=alpha_c0_full,
                    alpha_div=self.alpha_divergence,
                    lambda_belief=self.lambda_align,
                    # See rope-branch envelope-identity comment above.
                    lambda_softmax=0.0 if self.include_attention_entropy else self.lambda_soft,
                    kappa=_kappa,
                    eps=eps,
                    compute_sigma_align_grad=True,
                    irrep_dims=self.irrep_dims,
                    enforce_orthogonal=self.enforce_orthogonal,
                    cached_block_exp_pairs=block_exp_pairs,
                    use_rope=self.use_rope,
                    rope_base=self.rope_base,
                    exact_diagonal_transport=self.exact_diagonal_transport,
                    gauge_covariant_ridge=self.gauge_covariant_ridge,
                )

            # 2b. Monotone free-energy check (diagonal covariance only).
            # F(t) is measured at the belief state AT THE START of this
            # iteration, before any update. Compared against F(t-1) measured
            # at the previous iteration's entry state.  Off-manifold detour:
            # alpha_eff may be a Tensor (E_learnable_alpha), so we reduce it
            # to a scalar via .mean() for the bookkeeping.
            if is_diagonal:
                with torch.no_grad():
                    _sp = sigma_p.clamp(min=eps)
                    _sq = sigma.clamp(min=eps)
                    kl_qp_sum = 0.5 * (
                        _sq / _sp
                        + (mu - mu_p) ** 2 / _sp
                        - 1.0
                        + _sp.log()
                        - _sq.log()
                    ).sum()
                    if isinstance(alpha_eff, torch.Tensor):
                        _alpha_scalar = float(alpha_eff.detach().mean().item())
                    else:
                        _alpha_scalar = float(alpha_eff)
                    _beta_det = beta.detach()
                    f_align_sum = (_beta_det * kl_matrix.detach()).sum()
                    # Manuscript Eq. eq:free_energy_functional_final:
                    #   F_align = lambda * [(beta * KL).sum() + tau * (beta * log(beta/pi)).sum()]
                    # The entropy term must enter the monotone monitor when
                    # include_attention_entropy=True; otherwise the monitor
                    # tracks a different objective than the one being descended
                    # and emits false-positive monotonicity violations.
                    # Uniform prior pi = 1/N; log N is an additive constant
                    # dropped here to match the alignment_loss in _update_phi.
                    if self.include_attention_entropy:
                        _kappa_scalar = float(_kappa.detach().item()) if isinstance(_kappa, torch.Tensor) else float(_kappa)
                        _dim_scale = math.sqrt(max(self.embed_dim, 1))
                        _beta_safe = _beta_det.clamp(min=1e-30)
                        f_ent_sum = (_beta_safe * _beta_safe.log()).sum()
                        f_align_total = float(f_align_sum.item()) + _kappa_scalar * _dim_scale * float(f_ent_sum.item())
                    else:
                        f_align_total = float(f_align_sum.item())
                    f_t = (
                        _alpha_scalar * float(kl_qp_sum.item())
                        + float(self.lambda_align) * f_align_total
                    )
                    f_history.append(f_t)
                    if f_prev is not None:
                        _tol = max(
                            f_monotone_abs_tol,
                            f_monotone_rel_tol * abs(f_prev),
                        )
                        if f_t > f_prev + _tol:
                            warnings.warn(
                                f"E-step iter {t}: F increased "
                                f"{f_prev:.6f} -> {f_t:.6f} "
                                f"(delta={f_t - f_prev:.2e}). Monotone "
                                f"descent violated; check e_*_lr or "
                                f"conditioning.",
                                RuntimeWarning,
                                stacklevel=2,
                            )
                    f_prev = f_t

            # 3. Optional: add active inference gradients
            if active_inference_fn is not None:
                ai_grad_mu, ai_grad_sigma = active_inference_fn(mu, sigma)
                if ai_grad_mu is not None:
                    grad_mu = grad_mu + ai_grad_mu
                if ai_grad_sigma is not None:
                    grad_sigma = grad_sigma + ai_grad_sigma

            # 4. Natural gradient projection
            nat_grad_mu, nat_grad_sigma = compute_natural_gradient_gpu(
                grad_mu, grad_sigma, sigma, eps=eps,
            )

            # On the final iteration we always store the raw `_last_attention`
            # and `_last_kl_matrix` tensor references (cheap; trainer plots
            # consume them). The expensive `.item()`-heavy diagnostics dict is
            # gated behind `track_layer_diagnostics` because each `.item()`
            # call is a CUDA sync that stalls the stream on every forward.
            if t == self.n_e_steps - 1:
                self._last_attention = beta.detach()
                self._last_kl_matrix = kl_matrix.detach()
            if t == self.n_e_steps - 1 and self.track_layer_diagnostics:
                with torch.no_grad():
                    self._last_diagnostics = {
                        # Gradient norms (E-step)
                        'nat_grad_mu_norm': nat_grad_mu.norm().item(),
                        'nat_grad_sigma_norm': nat_grad_sigma.norm().item(),
                        'grad_mu_norm': grad_mu.norm().item(),
                        'grad_sigma_norm': grad_sigma.norm().item(),
                        # Attention statistics
                        'beta_mean': beta.mean().item(),
                        'beta_std': beta.std().item(),
                        'kl_mean': kl_matrix.mean().item(),
                        'kl_max': kl_matrix.max().item(),
                        'attention_entropy': -(beta.clamp(min=1e-10) * beta.clamp(min=1e-10).log()).sum(-1).mean().item(),
                        'attention_entropy_loss': (
                            (_kappa * math.sqrt(max(self.embed_dim, 1))
                             * (beta.clamp(min=1e-30) * beta.clamp(min=1e-30).log()).sum()).item()
                            if self.include_attention_entropy else 0.0
                        ),
                        'attention_concentration': beta.max(dim=-1)[0].mean().item(),
                        # Covariance health
                        'sigma_q_mean': sigma.mean().item(),
                        'sigma_q_min': sigma.min().item(),
                        'sigma_q_max': sigma.max().item(),
                        'sigma_q_std': sigma.std().item(),
                        'sigma_p_mean': sigma_p.mean().item(),
                        # Phi norms
                        'phi_norm_mean': phi.norm(dim=-1).mean().item(),
                        'phi_norm_std': phi.norm(dim=-1).std().item(),
                        'phi_norm_max': phi.norm(dim=-1).max().item(),
                    }
                    # Per-dimension KL(q*||p) for prior-belief gap
                    if is_diagonal:
                        sp = sigma_p.clamp(min=eps)
                        sq = sigma.clamp(min=eps)
                        kl_qp = 0.5 * (sq / sp + (mu - mu_p)**2 / sp - 1 + sp.log() - sq.log())
                        self._last_diagnostics['prior_belief_kl_mean'] = kl_qp.sum(-1).mean().item()
                        self._last_diagnostics['prior_belief_kl_max'] = kl_qp.sum(-1).max().item()
                        self._last_diagnostics['prior_belief_kl_std'] = kl_qp.sum(-1).std().item()

                    # Free-energy trajectory (diagonal covariance only; empty
                    # list on full-cov paths).  Downstream can assert
                    # monotonicity as a hard correctness gate in unit tests.
                    if f_history:
                        self._last_diagnostics['f_history'] = list(f_history)
                        self._last_diagnostics['f_final'] = f_history[-1]
                        def _ok(i: int) -> bool:
                            prev = f_history[i - 1]
                            tol = max(
                                f_monotone_abs_tol,
                                f_monotone_rel_tol * abs(prev),
                            )
                            return f_history[i] <= prev + tol
                        self._last_diagnostics['f_monotone'] = all(
                            _ok(i) for i in range(1, len(f_history))
                        )

            # 5. Mean update: mu -= eta_mu * nat_grad_mu
            mu = mu - self.e_mu_lr * nat_grad_mu

            # 6. SPD retraction for sigma
            sigma = retract_sigma_e_step(
                sigma_current=sigma,
                nat_grad_sigma=nat_grad_sigma,
                effective_lr=self.e_sigma_lr,
                is_diagonal=is_diagonal,
                eps=eps,
                update_sigma=True,
                sigma_trust=self.e_sigma_q_trust,
                sigma_max=self.sigma_max,
                isotropic_covariance=self.isotropic_covariance,
            )

            # 7. Phi update with preconditioning
            phi = self._update_phi(phi, mu, sigma, is_diagonal, mask, eps, block_exp_pairs, _kappa)

        return BeliefState(mu=mu, sigma=sigma, phi=phi)

    def _update_phi(
        self,
        phi: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        is_diagonal: bool,
        mask: Optional[torch.Tensor],
        eps: float,
        cached_block_exp_pairs: Optional[list] = None,
        kappa: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""Compute phi gradient via autograd and retract.

        Uses autograd through ``compute_kl_attention`` to get
        :math:`\partial F_{\text{align}} / \partial\phi`, then applies
        Killing form preconditioning and retracts on the Lie algebra.
        """
        # IMPORTANT: this path uses torch.autograd.grad internally.  When the
        # caller is inside torch.no_grad() (e.g. during validation), autograd
        # is globally disabled and `requires_grad_(True)` is silently ignored
        # — the ``if alignment_loss.grad_fn is not None`` guard below would
        # then skip the update rather than raise, masking the failure.  We
        # force-enable grad tracking here to match the defensive pattern in
        # _compute_rope_full_gauge_gradient_per_head (vfe_gradients.py:2106).
        with torch.enable_grad():
            phi_for_grad = phi.detach().requires_grad_(True)

            # Single forward through compute_kl_attention; both beta and
            # kl_matrix share the same autograd subgraph through phi_for_grad.
            # The product rule decomposition below uses .detach() to route
            # gradients along exactly one of the two paths per term.
            _kappa = kappa if kappa is not None else self.effective_kappa
            # NOTE: cached_block_exp_pairs is intentionally NOT forwarded
            # here. The cache was built from `phi` (not `phi_for_grad`) so
            # passing it would detach Omega from the phi autograd path and
            # zero out d/dphi alignment_loss. compute_kl_attention must
            # recompute the exponentials from `phi_for_grad` to keep the
            # gradient subgraph live.
            beta_phi, kl_h = compute_kl_attention(
                mu.detach(), sigma.detach() if sigma is not None else None,
                phi_for_grad, self.generators,
                self.irrep_dims, _kappa, mask,
                use_rope=self.use_rope,
                rope_base=self.rope_base,
                enforce_orthogonal=self.enforce_orthogonal,
                mask_self_attention=self.mask_self_attention,
                exact_diagonal_transport=self.exact_diagonal_transport,
            )

            if self.include_attention_entropy:
                # Manuscript Eq. eq:free_energy_functional_final F functional, direct form:
                #   F = (β · KL).sum() + τ_eff · (β · log(β/π)).sum(),  τ_eff = κ · √K
                # All attached so autograd produces the envelope-correct gradient at
                # the softmax β stationary point: ∂F/∂θ = Σ β·∂KL/∂θ for θ ∈ (μ,Σ,φ),
                # and ∂F/∂κ = √K · (β·log β).sum(). lambda_align scales the entire F
                # uniformly; lambda_soft is IGNORED in this branch because the manuscript
                # F has no separate "softmax-coupling" knob — the product-rule split is
                # an autograd convenience for the entropy-suppressed surrogate path only.
                # Uniform prior π = 1/N; constant log N dropped (additive const in F).
                # Verified by tests/test_entropy_envelope.py: production config matches
                # the envelope prediction to ~1e-18 for any λ_soft.
                beta_safe = beta_phi.clamp(min=1e-30)
                _dim_scale = math.sqrt(max(self.embed_dim, 1))
                _F = (beta_phi * kl_h).sum() + _kappa * _dim_scale * (beta_safe * beta_safe.log()).sum()
                alignment_loss = self.lambda_align * _F
            else:
                # Legacy product-rule decomposition — entropy-suppressed surrogate path.
                # d/dphi [sum(beta * KL)] = beta * dKL/dphi + KL * dbeta/dphi
                alignment_loss = (
                    self.lambda_align * (beta_phi.detach() * kl_h).sum()
                    + self.lambda_soft * (beta_phi * kl_h.detach()).sum()
                )

            if alignment_loss.grad_fn is not None:
                grad_phi = torch.autograd.grad(
                    alignment_loss, phi_for_grad,
                    create_graph=False, retain_graph=False,
                )[0]

                # Precondition
                grad_phi = precondition_phi_gradient(
                    grad_phi, phi,
                    mode=self.phi_preconditioner_mode,
                    preconditioner=self._phi_preconditioner,
                    generators=self.generators,
                )

                # Retract
                phi = _retract_phi(
                    phi, grad_phi, self.generators,
                    step_size=self.e_phi_lr,
                    gauge_group=self.gauge_group,
                    project_slk=self.phi_project_slk,
                    trace_clamp=self.phi_trace_clamp,
                    irrep_dims=self.irrep_dims,
                )

        return phi
