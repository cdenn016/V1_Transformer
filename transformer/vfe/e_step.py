"""
VFEEStep: iterative VFE minimization inner loop.

Replaces the 2,893-line variational_ffn.py with a single clear E-step path.
No EM mode branching, no DEQ, no closed-form, no hebbian.

Law 1 enforced: forward() has no targets parameter. Target leakage is
structurally impossible.

Law 2 enforced: all transport goes through fused block-diagonal kernels
which internally compute Omega @ Sigma @ Omega^T.

Implementation note — F functional realized vs manuscript canonical.
==================================================================
The manuscript free-energy functional (``\\label{eq:free_energy_functional_final}``)
has FIVE terms:

    F = alpha * KL(q || p)                                      # self-coupling
      + lambda_h * KL(s || h)                                   # hyper-prior
      + sum_ij [ beta_ij * KL(q_i || Omega_ij q_j)
                 + tau * beta_ij * log(beta_ij / pi_ij) ]       # belief coupling + entropy
      + sum_ij [ gamma_ij * KL(s_i || Omega_ij s_j)
                 + tau * gamma_ij * log(gamma_ij / pi^s_ij) ]   # model coupling + meta entropy
      - E_q[log p(o | x)]                                       # observation likelihood

This E-step assembles only THREE of those five terms in the inner-loop
gradient:

    alpha * KL(q || p)
    sum_ij beta_ij * KL(q_i || Omega_ij q_j)
    tau * beta_ij * log(beta_ij / pi_ij)        (when include_attention_entropy=True)

The gamma-coupled model-coupling term (``sum_ij gamma_ij KL(s_i || Omega_ij s_j)``)
and the hyper-prior term (``lambda_h * KL(s || h)``) are NOT implemented in this
path — there is no gamma-attention parameter anywhere in the package, and
``s`` does not exist as a separate distribution from ``q``. Treat the
implemented F as a structural subset of the manuscript F; cross-layer
"model coupling" is realised instead by the prior-handoff cascade in
``transformer/vfe/stack.py`` (see "mean-only cascade" note there).

Outer M-step (see ``transformer/vfe/model.py``) minimises cross-entropy
plus a ``mass_phi * ||phi||^2`` regulariser plus
``sum block._aux_hyperparam_loss``, NOT the converged-q value of F.
The alpha*KL(q||p) and beta*KL terms enter the M-step only as gradients
through the unrolled E-step iterations — this is structurally amortised
inference (the embeddings are tuned so that CE is small after the E-step
relaxes), not classical variational EM where E and M alternate on the
same F functional.

Implementation note — Bayesian alpha is per-dimension, not scalar.
==================================================================
Manuscript ``\\label{eq:state_dependent_alpha}`` defines a scalar
adaptive precision per agent: ``alpha_i* = c0 / (b0 + D_KL(q_i || p_i))``
with a single scalar log-barrier ``R(alpha_i) = b0 * alpha_i - c0 * log alpha_i``.
This implementation generalises to per-K-dimension:
``raw_c0``, ``raw_b0`` are ``nn.Parameter`` of shape ``(K,)`` (see
``__init__``), and ``get_bayesian_alpha`` returns ``c0 / (b0 + kl_k)``
with per-dimension ``kl_k`` of shape ``(B, N, K)``. Each belief dimension
therefore carries its own adaptive precision. The product-rule correction
in ``core/vfe_gradients.py:_apply_alpha_correction`` is similarly per-K
(``alpha**2 / alpha_c0 * kl_k * delta_mu_sp / sigma_p_safe``).

This is a stronger generalisation than the published scalar form and is
not currently derived in the manuscript. Treat as research-track until
the per-dim Gamma-Normal conjugacy + per-dim log-barrier derivation
lands in ``Attention/GL(K)_attention.tex``.
"""

from __future__ import annotations

import math
import warnings
from typing import Optional, Callable, List, Tuple, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from transformer.vfe.config import VFEConfig

from transformer.core.types import BeliefState
from transformer.core.vfe_gradients import (
    compute_vfe_gradients_gpu,
    compute_natural_gradient_gpu,
    _compute_rope_full_gauge_gradient_per_head,
    _fused_attention_and_vfe_gradients_block_diag,
)
from transformer.core.vfe_utils import (
    retract_sigma_e_step,
    _retract_phi,
)
from transformer.core.gauge_preconditioner import (
    build_cartan_projector,
    build_killing_form_preconditioner,
    build_killing_form_preconditioner_per_block,
)
from transformer.core.phi_evolution import precondition_phi_gradient
from transformer.vfe.attention import (
    compute_kl_attention,
    compute_gauge_transport,
)
from transformer.vfe.non_flat import (
    VFENonFlatConnection,
    compute_pairwise_omega_with_delta,
    compute_kl_attention_pairwise,
)
from transformer.vfe.omega_direct import (
    compute_pairwise_omega_from_endpoints,
    omega_natural_grad_step,
    project_omega_to_slk,
    _build_killing_matrix_per_block,
)
from transformer.vfe._numerics import (
    VFENonFiniteError,
    apply_mu_trust_region,
    check_finite,
)


# Common floor for β·log β entropy computations. β is row-stochastic so the
# only way to underflow is when softmax saturates onto one column; 1e-30 sits
# safely above float64 underflow (~1e-323) and below any meaningful softmax
# probability, so β·log β > -69 ≈ log(1e-30) is bounded.
_BETA_LOG_FLOOR: float = 1e-30


def _diag_kl(
    mu_q: torch.Tensor,
    mu_p: torch.Tensor,
    sigma_q: torch.Tensor,
    sigma_p: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    r"""Per-dimension standard :math:`\mathrm{KL}(q\|p)` for diagonal Gaussians.

    .. math::
        \mathrm{KL}_k = \tfrac{1}{2}\,\bigl(\sigma_{q,k}/\sigma_{p,k}
        + (\mu_{q,k}-\mu_{p,k})^2/\sigma_{p,k}
        - 1 + \log\sigma_{p,k} - \log\sigma_{q,k}\bigr)

    Both :math:`\sigma_q` and :math:`\sigma_p` are floored at ``eps`` before
    division and ``log``. Returns the per-dim tensor of shape
    ``(..., K)``; callers sum (over K, or over (N, K), or over all axes) as needed.

    Standard KL only; for the Rényi alpha-divergence the F-monotone monitor
    delegates to ``_kl_kernel_diagonal`` (which supports ``alpha_div``).
    Other callers (Bayesian-alpha auxiliary, prior-belief diagnostic,
    autograd / non-flat paths) explicitly use standard KL.
    """
    _sp = sigma_p.clamp(min=eps)
    _sq = sigma_q.clamp(min=eps)
    return 0.5 * (
        _sq / _sp
        + (mu_q - mu_p) ** 2 / _sp
        - 1.0
        + _sp.log()
        - _sq.log()
    )


def _f_monotone_step(
    *,
    mu_q: torch.Tensor,
    mu_p: torch.Tensor,
    sigma_q: torch.Tensor,
    sigma_p: torch.Tensor,
    eps: float,
    beta_det: torch.Tensor,
    kl_det: torch.Tensor,
    alpha_eff: "torch.Tensor | float",
    kappa: "torch.Tensor | float",
    dim_scale: float,
    include_attention_entropy: bool,
    lambda_align: float,
    alpha_div: float,
    f_history: List[float],
    f_prev: Optional[float],
    f_abs_tol: float,
    f_rel_tol: float,
    iter_idx: int,
    label: str,
) -> float:
    r"""Single diagonal-covariance F-monotone scalar check.

    Computes :math:`F_t = \alpha \sum D(q\|p)
        + \lambda_{\text{align}} \sum \beta_{ij} \mathrm{KL}_{ij}
        + (\text{optional}) \,\tau \sum \beta \log \beta` (entropy term),
    where ``D`` is the configured divergence (standard KL when
    ``alpha_div == 1.0``, Rényi alpha-divergence otherwise; delegates to
    ``_kl_kernel_diagonal``). Appends to ``f_history`` and emits a
    ``RuntimeWarning`` when ``f_t > f_prev + tol``. All ``.item()`` syncs
    are concentrated here — callers should already be in
    ``torch.no_grad()``.

    Returns the new scalar ``f_t`` (caller assigns to ``f_prev``).

    **Caveat — the warning is expected under canonical E-step, not a bug.**
    The E-step retracts all q_i in parallel using the mean-field VMP
    gradient (query-side partial of F holding all other q_j fixed).
    Parallel mean-field updates are NOT guaranteed to produce a monotone
    F decrease per outer iteration — only fully sequential coordinate
    descent has that property (Jordan et al. 1999 §6; Beal 2003 §2.3).
    So a ``RuntimeWarning: F increased`` under the analytic kernel is a
    feature of parallel VMP, not a sign of mis-conditioned LRs or a
    gradient bug. The earlier audit-2026-05-17 fix that wired
    ``_compute_mu_sigma_grad_autograd`` to silence this warning by
    switching to a total-derivative gradient is superseded — that path
    introduces a non-canonical gradient (sigma RMS ratio ~20–145x the
    canonical kernel; see ``post_edit_2026-05-19.md`` Wave 6) and
    produces measurably worse training results. Treat this monitor as a
    sanity-bound diagnostic, not a strict convergence criterion;
    re-tune LRs only if the violations are large in magnitude or
    persistent across many iterations.

    Note on the scalar's relationship to the canonical F functional:
    the manuscript canonical F has the attention-entropy term `tau * sum
    beta * log(beta/pi)` as a structural part of the *unreduced* F, NOT
    scaled by `lambda_align`. The monitor scales the entropy term TOGETHER
    with the alignment term by `lambda_align`, matching the runtime loss
    construction in `_update_phi` (which folds `lambda_align` over the
    full `(beta*KL + tau*beta*log(beta/pi))` block). The monitored
    quantity is therefore the runtime-realised "scaled F", not the
    canonical F. For `lambda_align == 1.0` (the manuscript default) the
    two agree exactly.
    """
    from transformer.core.kl_computation import _kl_kernel_diagonal
    # Use the canonical kernel so the monitor respects alpha_divergence.
    # Returns shape (..., ) sum-over-K; .sum() collapses the remaining axes.
    kl_qp_sum = _kl_kernel_diagonal(
        mu_q, sigma_q, mu_p, sigma_p,
        kl_max=float('inf'), eps=eps, alpha_div=alpha_div,
    ).sum()
    # CAVEAT (audit-2026-05-18-v4 F6.3): when `E_learnable_alpha=True`,
    # `alpha_eff` is per-(B,N,K) and the true self-coupling term is
    # `Σ_{B,N,k} α_k · kl_k`. This monitor proxies it with
    # `mean(α) · Σ kl` which only equals the true value when α and kl are
    # uncorrelated. The Bayesian-alpha rule α_k = c₀/(b₀ + kl_k) makes them
    # ANTI-correlated by construction, so this monitor over-estimates F.
    # Dormant under the active config (monitor_monotonicity=False,
    # track_layer_diagnostics=False); only affects diagnostic plots when
    # explicitly enabled.
    _alpha_scalar = (
        float(alpha_eff.detach().mean().item())
        if isinstance(alpha_eff, torch.Tensor)
        else float(alpha_eff)
    )
    f_align_sum = (beta_det * kl_det).sum()
    if include_attention_entropy:
        # Per-head κ_h (shape (H,)) collapses to a single monitor scalar
        # via mean — the F-monitor is a diagnostic, not a tight identity;
        # using mean keeps it interpretable as a "typical" temperature
        # without crashing on .item() of a 1-D tensor.
        if isinstance(kappa, torch.Tensor):
            if kappa.dim() == 0:
                _kappa_scalar = float(kappa.detach().item())
            else:
                _kappa_scalar = float(kappa.detach().mean().item())
        else:
            _kappa_scalar = float(kappa)
        _beta_safe = beta_det.clamp(min=_BETA_LOG_FLOOR)
        f_ent_sum = (_beta_safe * _beta_safe.log()).sum()
        f_align_total = (
            float(f_align_sum.item())
            + _kappa_scalar * dim_scale * float(f_ent_sum.item())
        )
    else:
        f_align_total = float(f_align_sum.item())
    f_t = (
        _alpha_scalar * float(kl_qp_sum.item())
        + float(lambda_align) * f_align_total
    )
    f_history.append(f_t)
    if f_prev is not None:
        _tol = max(f_abs_tol, f_rel_tol * abs(f_prev))
        if f_t > f_prev + _tol:
            warnings.warn(
                f"{label} iter {iter_idx}: F increased "
                f"{f_prev:.6f} -> {f_t:.6f} (delta={f_t - f_prev:.2e}). "
                f"Parallel mean-field VMP is not guaranteed per-iteration "
                f"monotone (Jordan et al. 1999 §6); only large or persistent "
                f"violations indicate an LR or conditioning issue.",
                RuntimeWarning,
                stacklevel=3,
            )
    return f_t


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
        self._kappa_per_head = getattr(cfg, 'kappa_per_head', False)
        if cfg.learnable_kappa:
            if self._kappa_per_head:
                # Per-head learnable κ_h vector. Each head learns its own
                # log-space scalar; init all heads at log(cfg.kappa) so the
                # initial behavior matches the scalar-κ run. Config-level
                # __post_init__ guarantees len(irrep_dims) > 1 and
                # per_head_softmax=True when this branch is taken.
                #
                # Under cross_couplings the "head" structure visible to the
                # softmax is the super-block partition (effective_block_dims),
                # so the per-head κ vector has one entry per super-block, not
                # one per original head.
                n_heads = len(cfg.effective_block_dims)
                self.log_kappa_per_head = nn.Parameter(
                    torch.full((n_heads,), math.log(cfg.kappa))
                )
                self._kappa_init = cfg.kappa
            else:
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
        # Gauge-block partition: under cross_couplings this is the super-block
        # layout (sum of merged-head dims) rather than the original per-head
        # dims. All block-iteration sites in this module walk this list, so
        # per-block softmax / per-block KL / per-block Ω naturally operate on
        # the correct gauge-block-diagonal structure.
        self.irrep_dims: List[int] = cfg.effective_block_dims
        # Preserve the original per-head layout for the rare consumer that
        # genuinely needs per-head granularity (e.g. cross-head-coupling
        # diagnostics that compare super-block aggregates to per-head values).
        self._original_irrep_dims: List[int] = cfg.irrep_dims
        self.use_rope = cfg.use_rope
        self.rope_base = cfg.rope_base
        self.rope_full_gauge = cfg.rope_full_gauge
        self.enforce_orthogonal = cfg.enforce_orthogonal
        self.mask_self_attention = cfg.mask_self_attention
        # Multi-head softmax convention. True (default): per-head softmax at
        # τ_h = κ·√d_head — canonical for the GL(d_head)^H multi-head reduction
        # (Attention/GL(K)_attention.tex eq:per_head_temperature, line 1744).
        # False: single softmax over Σ_h KL_h at τ = κ·√K — single-head limit
        # only; retained for ablation and pre-2026-05-19 backward compatibility.
        self.per_head_softmax: bool = getattr(cfg, 'per_head_softmax', True)
        self.gauge_covariant_ridge = getattr(cfg, 'gauge_covariant_ridge', False)
        self.track_layer_diagnostics = getattr(cfg, 'track_layer_diagnostics', False)
        # F-monotone monitor: records F at every iter and warns on non-monotone
        # descent. Fires .item() CUDA syncs per iter when on; default off.
        self.monitor_monotonicity = getattr(cfg, 'monitor_monotonicity', False)
        # Per-forward trainer-controlled gates. The post-iteration attention
        # state rebuild (compute_gauge_transport on the converged φ) is only
        # consumed by the periodic attention plot, so the trainer should set
        # this False on non-eval steps to skip the rebuild. Default True so
        # callers that don't reset it preserve historical behavior.
        self._capture_attention_state: bool = True
        # Hoist embed_dim scale used by the attention-entropy term and a few
        # diagnostic expressions; static value, no need to recompute per iter.
        self._dim_scale = math.sqrt(max(cfg.embed_dim, 1))
        self.use_autograd_mu_sigma = getattr(cfg, 'use_autograd_mu_sigma', False)
        # F4.10: opt-in causal-lower-triangle fast path for the fused kernel.
        # When True, _fused_attention_and_vfe_gradients_block_diag computes
        # only the lower triangle (j<=i) of per-pair tensors and scatters
        # into dense accumulators. Bit-identical to dense for beta/grad_mu/
        # grad_sigma under a strict lower-triangular mask. Validated once
        # per forward() at the kernel call site below.
        self.causal_lower_triangle = getattr(cfg, 'causal_lower_triangle', False)
        self.gauge_group = cfg.gauge_group
        self.phi_preconditioner_mode = cfg.phi_preconditioner
        self.phi_project_slk = cfg.phi_project_slk
        self.phi_trace_clamp = cfg.phi_trace_clamp
        # Opt-in numerical-conditioning flags (2026-05-19). All default to
        # values that preserve the pure path bitwise; see VFEConfig docstrings.
        self.e_mu_q_trust: Optional[float] = getattr(cfg, 'e_mu_q_trust', None)
        self.e_nan_check: str = getattr(cfg, 'e_nan_check', 'off')
        self.phi_spec_max: Optional[float] = getattr(cfg, 'phi_spec_max', None)
        self.phi_strict_glplus: bool = getattr(cfg, 'phi_strict_glplus', False)

        if not isinstance(generators, torch.Tensor):
            generators = torch.from_numpy(generators).float()
        self.register_buffer('generators', generators)

        # Non-flat parallel transport (opt-in). When enabled, transport
        # becomes Ω_ij = exp(φ_i·G) · exp(δ_ij·G) · exp(-φ_j·G) per block.
        # Init at zero ⇒ flat path bitwise-equivalent at step 0.
        self.use_non_flat_transport = cfg.use_non_flat_transport
        if cfg.use_non_flat_transport:
            # Under cross_couplings, supply the per-head sub-partition inside
            # each super-block so diagonal generators get a tight d_head mask
            # while cross generators still receive the full super-block mask.
            head_sub_dims: Optional[List[List[int]]] = None
            if cfg.is_cross_coupled:
                _, _n_heads, d_head = cfg.irrep_spec[0]
                groups = cfg.super_block_head_groups or []
                head_sub_dims = [[d_head] * len(g) for g in groups]
            self.non_flat_connection = VFENonFlatConnection(
                generators=generators,
                irrep_dims=cfg.effective_block_dims,
                max_strength=cfg.non_flat_max_strength,
                per_edge_delta_max=cfg.non_flat_per_edge_delta_max,
                head_sub_dims=head_sub_dims,
            )
        else:
            self.non_flat_connection = None

        # Omega-direct gauge parameterization (opt-in). When enabled, Ω per
        # token is itself the state; the E-step updates Ω via right-invariant
        # natural gradient on GL+(K) per block. φ is held fixed at its
        # encode-time value.
        self.gauge_parameterization = cfg.gauge_parameterization
        self.omega_normalize_every = cfg.omega_normalize_every
        self.omega_project_slk = cfg.omega_project_slk
        if cfg.gauge_parameterization == 'omega_direct':
            # Cache the per-block Killing/Frobenius Gram-inverse used to
            # project the unconstrained Ω-direction back onto span(G^a). The
            # cache is data-independent; computing it once at construction
            # saves a small constant per E-step iteration.
            self._omega_killing_cache = _build_killing_matrix_per_block(
                generators, cfg.effective_block_dims,
            )
        else:
            self._omega_killing_cache = None

        # Build phi preconditioner (Killing metric or Cartan projector).
        # 'killing'           — ambient gl(K_full) Killing form on the
        #                       restricted block-diagonal subalgebra (cross-
        #                       block coupling via -2·tr⊗tr).
        # 'killing_per_block' — direct-sum Killing form
        #                       gl(d_1) ⊕ ... ⊕ gl(d_H); block-diagonal in
        #                       generator index, no cross-block coupling.
        _phi_prec = None
        if cfg.phi_preconditioner == 'cartan':
            _phi_prec = build_cartan_projector(generators)
        elif cfg.phi_preconditioner == 'killing':
            _phi_prec = build_killing_form_preconditioner(generators)
        elif cfg.phi_preconditioner == 'killing_per_block':
            _phi_prec = build_killing_form_preconditioner_per_block(
                generators, cfg.effective_block_dims,
            )
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

    def _get_alpha_c0(self) -> Optional[torch.Tensor]:
        """Iter-independent Bayesian-alpha concentration c0 = softplus(raw_c0).

        Returns None when E_learnable_alpha is disabled — the caller short-
        circuits the per-dim Bayesian path in that case.
        """
        if not self.E_learnable_alpha:
            return None
        return F.softplus(self.raw_c0)

    def get_bayesian_alpha(
        self,
        mu_q: torch.Tensor,
        mu_p: torch.Tensor,
        sigma_q: torch.Tensor,
        sigma_p: torch.Tensor,
        eps: float = 1e-6,
        c0: Optional[torch.Tensor] = None,
        b0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""Per-dimension Bayesian precision: :math:`\alpha_k = c_0 / (b_0 + \mathrm{KL}_k)`.

        Each belief dimension k gets its own precision via Gamma-Normal
        conjugacy, so different irrep blocks can learn different
        regularization curves. Passing ``c0`` / ``b0`` lets the E-step
        outer scope hoist ``softplus(raw_c0)`` / ``softplus(raw_b0)`` and
        reuse them across iterations rather than recomputing the activations
        every call.

        Returns:
            alpha: ``(B, N, K)`` per-dimension precision.
        """
        if c0 is None:
            c0 = F.softplus(self.raw_c0)  # (K,)
        if b0 is None:
            b0 = F.softplus(self.raw_b0)  # (K,)

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
    def effective_kappa(self) -> torch.Tensor | float:
        r"""Resolve kappa.

        Returns:
            - ``float`` ``self.kappa`` when ``learnable_kappa=False``.
            - 0-dim ``torch.Tensor`` ``exp(log_kappa).clamp(0.5·κ₀, 2·κ₀)``
              when ``learnable_kappa=True`` AND ``kappa_per_head=False``.
            - 1-D ``torch.Tensor`` of shape ``(H,)`` with element-wise
              clamp when ``learnable_kappa=True`` AND ``kappa_per_head=True``.
              Downstream per-head dispatch indexes this with ``_kappa[h]``;
              scalar consumers (F-monitor diagnostics) take ``.mean()``.
        """
        if self._learnable_kappa:
            if self._kappa_per_head:
                k = torch.exp(self.log_kappa_per_head)  # (H,)
            else:
                k = torch.exp(self.log_kappa)            # ()
            k0 = self._kappa_init
            return k.clamp(min=0.5 * k0, max=2.0 * k0)
        return self.kappa

    def _auxiliary_hyperparam_loss(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        phi: torch.Tensor,
        mu_p: torch.Tensor,
        sigma_p: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        r"""Scalar F evaluated with hyperparameters live and beliefs detached.

        Without this term, the M-step parameters ``raw_c0``, ``raw_b0``, and
        ``log_kappa`` are gradient-starved. The descent kernels
        (analytic + autograd) extract gradients only w.r.t. ``(mu, sigma)``
        via ``torch.autograd.grad``; ``_update_phi`` does the same w.r.t.
        ``phi``. Hyperparameters appearing in those inner subgraphs never
        accumulate ``.grad`` on the outer ``loss.backward()``. This routes a
        single auxiliary scalar to CE so the canonical gradients survive:

        - :math:`\partial F/\partial c_0 = \mathrm{KL}_k / (b_0 + \mathrm{KL}_k)`
          and analogous for :math:`b_0`, both through ``get_bayesian_alpha``.
        - :math:`\partial F/\partial \kappa = \sqrt{K}\,\bigl(\sum \beta
          \log(\beta N) + \text{coupling-envelope terms}\bigr)`,
          through ``effective_kappa`` in the entropy term.

        Beliefs (mu, sigma, phi) and M-step priors (mu_p, sigma_p) are
        DETACHED so this loss contributes nothing to base_mu /
        base_log_sigma / phi_embed / previous-layer phi gradients. The
        single backward path is hyperparameter -> F -> CE.

        Returns ``None`` when neither hyperparameter is learnable; callers
        skip the aux-term contribution to ``loss`` in that case.
        """
        if not (self.E_learnable_alpha or self._learnable_kappa):
            return None

        # Detach everything the aux loss reads except the hyperparameters
        # themselves. sigma_p is already detached by the forward() preamble,
        # but defensively re-detach in case of future refactor.
        mu_d = mu.detach()
        sigma_d = sigma.detach()
        phi_d = phi.detach()
        mu_p_d = mu_p.detach()
        sigma_p_d = sigma_p.detach()

        eps = 1e-6
        aux: Optional[torch.Tensor] = None

        # Normalize per-token to match F.cross_entropy's mean reduction, so
        # m_hyper_lr operates on the same scale as m_mu_lr / m_phi_lr / etc.
        # B * N_q (queries) is the token count; sigma/alpha live on K-dim
        # per token; beta lives on N_k per query per batch.
        B, N_q = mu_d.shape[0], mu_d.shape[1]
        token_count = float(max(B * N_q, 1))

        # Self-coupling alpha * KL(q || p) -- delivers gradient to raw_c0, raw_b0.
        if self.E_learnable_alpha:
            # Extract diagonals for both sigma_q and sigma_p; get_bayesian_alpha
            # also does this internally but doing it once here keeps the KL
            # computation aligned and avoids duplicate diagonal() calls.
            sigma_d_diag = (
                torch.diagonal(sigma_d, dim1=-2, dim2=-1) if sigma_d.dim() == 4
                else sigma_d
            )
            sigma_p_d_diag = (
                torch.diagonal(sigma_p_d, dim1=-2, dim2=-1) if sigma_p_d.dim() == 4
                else sigma_p_d
            )
            alpha_attached = self.get_bayesian_alpha(
                mu_d, mu_p_d, sigma_d_diag, sigma_p_d_diag, eps=eps,
            )
            kl_qp_per_dim = _diag_kl(
                mu_d, mu_p_d, sigma_d_diag, sigma_p_d_diag, eps=eps,
            )
            self_term = (alpha_attached * kl_qp_per_dim).sum() / token_count
            aux = self_term if aux is None else aux + self_term

        # Attention term lambda_align * [sum beta * KL + tau * sum beta * log(beta * N)]
        # -- delivers gradient to log_kappa via effective_kappa.
        # Skipped when learnable_kappa is False (raw_c0/raw_b0 path above is
        # the only signal needed in that mode).
        if self._learnable_kappa:
            kappa_attached = self.effective_kappa  # scalar or (H,) tensor
            # Pre-compute log(N) constant once (independent of head).
            if mask is not None and mask.dim() >= 2:
                n_valid = mask.sum(dim=-1).clamp(min=1)
                log_N_const = n_valid.to(mu_d.dtype).log().mean()
            else:
                if mask is not None and mask.dim() < 2:
                    warnings.warn(
                        f"_auxiliary_hyperparam_loss received a 1D mask "
                        f"(shape={tuple(mask.shape)}); falling back to "
                        f"full N for log_N. Per-row N_valid correction "
                        f"is bypassed.",
                        UserWarning,
                        stacklevel=2,
                    )
                log_N_const = math.log(max(mu_d.shape[1], 1))

            if self._kappa_per_head and self.irrep_dims and len(self.irrep_dims) > 1:
                # Per-head aux: build the F_attn term per head and sum.
                # Each head's term carries gradient to its own
                # log_kappa_per_head[h] via the kappa_attached[h] slice
                # passed into compute_kl_attention. Mirrors the forward
                # per-head dispatch so the gradient direction is consistent.
                attn_term = mu_d.new_zeros(())
                block_start = 0
                for h, d_h in enumerate(self.irrep_dims):
                    block_end = block_start + d_h
                    mu_h = mu_d[..., block_start:block_end]
                    sigma_h = sigma_d[..., block_start:block_end]
                    gen_h = self.generators[:, block_start:block_end, block_start:block_end]
                    kappa_h_attached = kappa_attached[h]  # 0-dim tensor
                    beta_h, kl_attn_h = compute_kl_attention(
                        mu_h, sigma_h, phi_d, gen_h,
                        [d_h], kappa_h_attached, mask,
                        use_rope=self.use_rope,
                        rope_base=self.rope_base,
                        enforce_orthogonal=self.enforce_orthogonal,
                        mask_self_attention=self.mask_self_attention,
                        exact_diagonal_transport=self.exact_diagonal_transport,
                    )
                    beta_h_safe = beta_h.clamp(min=_BETA_LOG_FLOOR)
                    sum_beta_kl_h = (beta_h * kl_attn_h).sum()
                    tau_h = kappa_h_attached * math.sqrt(d_h)
                    entropy_term_h = (
                        tau_h * (beta_h_safe * beta_h_safe.log()).sum()
                        + tau_h * log_N_const * beta_h.sum()
                    )
                    attn_term = attn_term + (
                        self.lambda_align * (sum_beta_kl_h + entropy_term_h) / token_count
                    )
                    block_start = block_end
                aux = attn_term if aux is None else aux + attn_term
            else:
                # Scalar-κ path (legacy single-softmax over full-K, or
                # per_head_softmax=True with single head H=1).
                beta, kl_attn = compute_kl_attention(
                    mu_d, sigma_d, phi_d, self.generators,
                    self.irrep_dims, kappa_attached, mask,
                    use_rope=self.use_rope,
                    rope_base=self.rope_base,
                    enforce_orthogonal=self.enforce_orthogonal,
                    mask_self_attention=self.mask_self_attention,
                    exact_diagonal_transport=self.exact_diagonal_transport,
                )
                beta_safe = beta.clamp(min=_BETA_LOG_FLOOR)
                sum_beta_kl = (beta * kl_attn).sum()
                tau = kappa_attached * self._dim_scale
                entropy_term = (
                    tau * (beta_safe * beta_safe.log()).sum()
                    + tau * log_N_const * beta.sum()
                )
                attn_term = self.lambda_align * (sum_beta_kl + entropy_term) / token_count
                aux = attn_term if aux is None else aux + attn_term

        return aux

    def forward(
        self,
        beliefs: BeliefState,
        priors: BeliefState,
        mask: Optional[torch.Tensor] = None,
    ) -> BeliefState:
        r"""Run the iterative E-step.

        **Law 1 enforced**: No ``targets`` parameter exists. Target leakage
        is structurally impossible.

        Args:
            beliefs: Current Gaussian beliefs :math:`(\mu, \Sigma, \phi)`.
            priors: Layer priors :math:`(\mu_p, \Sigma_p, \phi_p)`.
            mask: ``(B, N, N)`` causal mask.

        Returns:
            Updated BeliefState after E-step convergence.
        """
        # Reset hyperparameter-aux-loss cache at every forward() entry, before
        # any branch. Otherwise a stale tensor from the previous batch's graph
        # (already freed by backward()) survives on the instance and either
        # delivers wrong gradients or errors on .backward(). Every path that
        # exits forward() must overwrite this with either None (when
        # E_learnable_alpha and learnable_kappa are both False) or a fresh
        # scalar built from the current batch's beliefs.
        self._aux_hyperparam_loss = None

        # F4.10: when the causal-lower-triangle fast path is enabled, validate
        # ONCE per forward() that the mask really is lower-triangular. The
        # fused kernel performs no per-call validation (would sync per E-step
        # iteration); we sync once here at forward() entry.
        if self.causal_lower_triangle:
            if mask is None:
                raise ValueError(
                    "VFEConfig.causal_lower_triangle=True requires a non-None "
                    "causal mask. Either pass a strict lower-triangular mask "
                    "or set causal_lower_triangle=False (the dense path)."
                )
            _m2d = mask if mask.dim() == 2 else mask[0]
            if _m2d.triu(diagonal=1).any().item():
                raise ValueError(
                    "VFEConfig.causal_lower_triangle=True requires a strict "
                    "lower-triangular mask (mask[..., i, j] == 0 for j > i). "
                    "Mask contains non-zero entries in the upper triangle; "
                    "the fast path would silently skip those pairs."
                )

        # Dispatch to the omega-direct iteration when the gauge state is
        # group-level. Keeps the φ-mode loop below unchanged.
        if self.gauge_parameterization == 'omega_direct':
            return self._forward_omega_direct(beliefs, priors, mask)

        mu = beliefs.mu
        sigma = beliefs.sigma
        phi = beliefs.phi
        mu_p = priors.mu
        # sigma_p is an M-step parameter — the E-step reads it but must not
        # write gradients to it (CLAUDE.md). Detach at extraction so any
        # downstream KL / alpha computation cannot accidentally route gradient
        # back into priors.sigma during the E-step inner iterations.
        sigma_p = priors.sigma.detach()

        is_diagonal = self.diagonal_covariance
        eps = 1e-6

        # Diagnostics buffer: populated on last iteration for the trainer to read
        self._last_diagnostics = {}

        # Cache effective_kappa once per forward — the property runs
        # torch.exp(log_kappa).clamp(...) and was previously called 4× per
        # E-step iteration (plus once inside _update_phi).
        _kappa = self.effective_kappa

        # Hoist iter-independent softplus(raw_c0) / softplus(raw_b0) out of
        # the loop — both are M-step parameters constant across E-step
        # iterations; previously softplus(raw_b0) was recomputed each call
        # to get_bayesian_alpha.
        alpha_c0_full_hoisted = self._get_alpha_c0()
        if self.E_learnable_alpha:
            alpha_b0_full_hoisted: Optional[torch.Tensor] = F.softplus(self.raw_b0)
        else:
            alpha_b0_full_hoisted = None

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

        # Pre-iteration snapshot for Item 1 (NaN sentinel revert/abort modes).
        # Allocated only when the sentinel is armed; 'off' and 'warn' modes
        # pay zero memory overhead (the variable stays None).
        _nan_check_armed: bool = self.e_nan_check in ('revert', 'abort')
        _iter_snapshot: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None

        for t in range(self.n_e_steps):
            if _nan_check_armed:
                # Snapshot the pre-iteration tensors WITHOUT detach: on a
                # 'revert', we return tensors that retain the autograd graph
                # back to the model inputs so M-step gradients still flow.
                # A previous detach() here silently severed the path and
                # zeroed the M-step contribution for any minibatch that
                # triggered a revert.
                _iter_snapshot = (mu.clone(), sigma.clone(), phi.clone())
            # 1. Compute transport and KL attention.
            # The "flat" cache (block_exp_pairs) is built unconditionally
            # because both the flat path and the non-flat path need it as a
            # pre-stage: non-flat uses it as input to compute_pairwise_omega.
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
                alpha_eff = self.get_bayesian_alpha(
                    mu, mu_p, sigma, sigma_p,
                    c0=alpha_c0_full_hoisted, b0=alpha_b0_full_hoisted,
                )

            # Non-flat transport branch: precomputes pairwise (Ω_ij, Ω_ij^{-1})
            # per block from the bilinear connection + φ-exp cache, then drives
            # KL attention + autograd gradient computation through the pairwise
            # tensor. Config validation guarantees we never reach here with
            # rope_full_gauge != 'off' or diagonal_covariance == False.
            if self.use_non_flat_transport:
                grad_mu, grad_sigma, beta, kl_matrix, omega_pairs_nf = self._iter_nonflat(
                    mu=mu, sigma=sigma, phi=phi,
                    mu_p=mu_p, sigma_p=sigma_p,
                    mask=mask,
                    eps=eps,
                    kappa=_kappa,
                    alpha_eff=alpha_eff,
                    block_exp_pairs=block_exp_pairs,
                )
            # rope_full_gauge: per-head autograd path that rotates BOTH μ and Σ.
            # NOTE: vfe currently treats 'vfe_only' and 'both' identically;
            # differentiating them requires splitting this branch.
            elif (
                self.rope_full_gauge != 'off'
                and self.use_rope
                and not torch.is_inference_mode_enabled()
                and torch.is_grad_enabled()
            ):
                # rope_full_gauge does its own per-head loop but doesn't
                # populate _last_attention_per_head — clear stale value.
                self._last_attention_per_head = None
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
                # _compute_rope_full_gauge_gradient_per_head returns only β
                # (not the per-pair KL), so we have no real KL matrix to
                # surface as a diagnostic here. Marking None avoids the
                # earlier `kl_matrix = beta` aliasing that mis-labelled
                # softmax probabilities as KL values in dashboards.
                kl_matrix = None
            else:
                # Fused path: single-pass β + ∇F computation when the gating
                # conditions of `_fused_attention_and_vfe_gradients_block_diag`
                # are met. Eliminates the duplicate Omega construction that
                # the separate `compute_kl_attention` + `compute_vfe_gradients_gpu`
                # pair performs (each E-step iteration previously built Omega
                # twice; the fused path builds it once).
                _can_use_fused = (
                    is_diagonal
                    and bool(self.irrep_dims)
                    and not self.exact_diagonal_transport
                    and not self.use_autograd_mu_sigma
                )
                if _can_use_fused and self.per_head_softmax and len(self.irrep_dims) > 1:
                    # Per-head softmax dispatch — canonical for the multi-head
                    # GL(d_head)^H reduction. One fused-kernel call per head with
                    # irrep_dims=[d_h]; per-head βs accumulated for plotting,
                    # per-head gradients concatenated into the full-K grad_mu /
                    # grad_sigma. Matches the legacy
                    # transformer/core/variational_ffn.py:1997-2014 pattern and
                    # the manuscript eq:per_head_temperature (κ_h · √d_head
                    # softmax temperature per head; here κ_h = κ shared since
                    # the per-head κ vector is reserved for a follow-up change).
                    grad_mu = torch.zeros_like(mu)
                    grad_sigma = torch.zeros_like(sigma)
                    beta_heads: List[torch.Tensor] = []
                    kl_heads: List[Optional[torch.Tensor]] = []
                    block_start = 0
                    _lambda_softmax_eff = (
                        0.0 if self.include_attention_entropy else self.lambda_soft
                    )
                    for h, d_h in enumerate(self.irrep_dims):
                        block_end = block_start + d_h
                        mu_h = mu[..., block_start:block_end].contiguous()
                        sigma_h = sigma[..., block_start:block_end].contiguous()
                        mu_p_h = mu_p[..., block_start:block_end].contiguous()
                        sigma_p_h = sigma_p[..., block_start:block_end].contiguous()
                        gen_h = self.generators[:, block_start:block_end, block_start:block_end]
                        _head_bep = (
                            [block_exp_pairs[h]] if block_exp_pairs is not None else None
                        )
                        # alpha may be scalar, tensor, or per-dim (B, N, K) —
                        # slice the per-dim case to d_h matching the kernel
                        # signature.
                        if isinstance(alpha_eff, torch.Tensor) and alpha_eff.dim() == 3:
                            alpha_h = alpha_eff[..., block_start:block_end]
                        else:
                            alpha_h = alpha_eff
                        alpha_c0_h = (
                            alpha_c0_full[block_start:block_end]
                            if alpha_c0_full is not None
                            else None
                        )
                        # Per-head κ_h slice when kappa_per_head=True
                        # (_kappa is a (H,) tensor); otherwise the same
                        # scalar κ is shared across all heads.
                        if isinstance(_kappa, torch.Tensor) and _kappa.dim() == 1:
                            kappa_h_eff = _kappa[h]
                        else:
                            kappa_h_eff = _kappa
                        beta_h, grad_mu_h, grad_sigma_h, kl_h = (
                            _fused_attention_and_vfe_gradients_block_diag(
                                mu_q=mu_h, sigma_q=sigma_h,
                                mu_p=mu_p_h, sigma_p=sigma_p_h,
                                phi=phi, generators=gen_h,
                                alpha=alpha_h,
                                lambda_belief=self.lambda_align,
                                lambda_softmax=_lambda_softmax_eff,
                                kappa=kappa_h_eff,
                                eps=eps,
                                irrep_dims=[d_h],
                                compute_sigma_align_grad=True,
                                enforce_orthogonal=self.enforce_orthogonal,
                                alpha_c0=alpha_c0_h,
                                cached_block_exp_pairs=_head_bep,
                                mask=mask,
                                mask_self_attention=self.mask_self_attention,
                                use_rope=self.use_rope,
                                rope_base=self.rope_base,
                                return_kl=True,
                                alpha_div=self.alpha_divergence,
                                causal_lower_triangle=self.causal_lower_triangle,
                            )
                        )
                        beta_heads.append(beta_h)
                        kl_heads.append(kl_h)
                        grad_mu[..., block_start:block_end] = grad_mu_h
                        grad_sigma[..., block_start:block_end] = grad_sigma_h
                        block_start = block_end
                    # Per-head β stack (B, H, N, N) — exposed for the trainer
                    # plot path; aggregated mean preserves the (B, N, N) shape
                    # of `_last_attention` that downstream consumers expect.
                    self._last_attention_per_head = torch.stack(
                        beta_heads, dim=1
                    ).detach()
                    beta = torch.stack(beta_heads, dim=1).mean(dim=1)
                    # kl_matrix is exposed for the F-monotone monitor and
                    # diagnostics; mean over heads keeps the (B, N, N) shape.
                    if all(k is not None for k in kl_heads):
                        kl_matrix = torch.stack(kl_heads, dim=0).mean(dim=0)
                    else:
                        kl_matrix = None
                elif _can_use_fused:
                    # Single global softmax over Σ_h KL_h (per_head_softmax=False
                    # OR single-head H=1 where the two conventions coincide).
                    # Clear the per-head snapshot so the plotter doesn't read
                    # stale per-head βs from a previous toggle state.
                    self._last_attention_per_head = None
                    beta, grad_mu, grad_sigma, kl_matrix = _fused_attention_and_vfe_gradients_block_diag(
                        mu_q=mu, sigma_q=sigma,
                        mu_p=mu_p, sigma_p=sigma_p,
                        phi=phi, generators=self.generators,
                        alpha=alpha_eff,
                        lambda_belief=self.lambda_align,
                        # Envelope identity: the entropy term and softmax-
                        # coupling term cancel at the β fixed point, so the
                        # σβ·KL gradient is zero when the entropy term is in F.
                        lambda_softmax=0.0 if self.include_attention_entropy else self.lambda_soft,
                        kappa=_kappa,
                        eps=eps,
                        irrep_dims=self.irrep_dims,
                        compute_sigma_align_grad=True,
                        enforce_orthogonal=self.enforce_orthogonal,
                        alpha_c0=alpha_c0_full,
                        cached_block_exp_pairs=block_exp_pairs,
                        mask=mask,
                        mask_self_attention=self.mask_self_attention,
                        use_rope=self.use_rope,
                        rope_base=self.rope_base,
                        return_kl=True,
                        alpha_div=self.alpha_divergence,
                        causal_lower_triangle=self.causal_lower_triangle,
                    )
                else:
                    # Fallback: separate β + gradient passes (rebuilds Omega
                    # internally). Used for autograd μ/σ, exact-diagonal
                    # transport, full-cov, or unspecified irrep_dims.
                    # Clear stale per-head snapshot so the trainer's plot
                    # path doesn't read it as fresh.
                    self._last_attention_per_head = None
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

                    if self.use_autograd_mu_sigma:
                        grad_mu, grad_sigma = self._compute_mu_sigma_grad_autograd(
                            mu=mu, sigma=sigma,
                            mu_p=mu_p, sigma_p=sigma_p,
                            phi=phi,
                            alpha_eff=alpha_eff,
                            block_exp_pairs=block_exp_pairs,
                            mask=mask,
                            kappa=_kappa,
                            eps=eps,
                            is_diagonal=is_diagonal,
                        )
                    else:
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
            # Gated by monitor_monotonicity OR track_layer_diagnostics — the
            # downstream test ``test_f_history_populated_with_entropy`` needs
            # f_history when track_layer_diagnostics is on, but a default
            # production run (both flags off) pays no .item() syncs here.
            # Skipped when kl_matrix is None (rope_full_gauge per-head path
            # returns only β, no KL matrix to monitor).
            if (
                is_diagonal
                and kl_matrix is not None
                and (self.monitor_monotonicity or self.track_layer_diagnostics)
            ):
                with torch.no_grad():
                    f_prev = _f_monotone_step(
                        mu_q=mu, mu_p=mu_p, sigma_q=sigma, sigma_p=sigma_p,
                        eps=eps,
                        beta_det=beta.detach(), kl_det=kl_matrix.detach(),
                        alpha_eff=alpha_eff, kappa=_kappa,
                        dim_scale=self._dim_scale,
                        include_attention_entropy=self.include_attention_entropy,
                        lambda_align=self.lambda_align,
                        alpha_div=self.alpha_divergence,
                        f_history=f_history, f_prev=f_prev,
                        f_abs_tol=f_monotone_abs_tol,
                        f_rel_tol=f_monotone_rel_tol,
                        iter_idx=t,
                        label="E-step",
                    )

            # 4. Natural gradient projection
            nat_grad_mu, nat_grad_sigma = compute_natural_gradient_gpu(
                grad_mu, grad_sigma, sigma, eps=eps,
            )

            # On the final iteration we always store the raw `_last_attention`
            # and `_last_kl_matrix` tensor references (cheap; trainer plots
            # consume them). The expensive `.item()`-heavy diagnostics dict is
            # gated behind `track_layer_diagnostics` because each `.item()`
            # call is a CUDA sync that stalls the stream on every forward.
            #
            # TIMING NOTE: `_last_attention` here reflects the β that drove
            # THIS iteration's μ/σ update — the actual training signal. The
            # `_last_attention_state` snapshot at the bottom of the loop
            # (after `_update_phi`) reflects the post-update φ — the frame
            # the NEXT forward would start from. The two snapshots therefore
            # describe different φ checkpoints by design; the trainer's
            # `_attention_summary` reads `_last_attention` (this-iter realized
            # β) while `_compute_per_head_beta` reads `_last_attention_state`
            # (next-iter starting frame). Plots compared across the two views
            # are NOT comparing the same φ.
            if t == self.n_e_steps - 1:
                self._last_attention = beta.detach()
                self._last_kl_matrix = (
                    kl_matrix.detach() if kl_matrix is not None else None
                )
                # _last_attention_per_head is populated by the per-head
                # dispatch above and cleared (set to None) by the
                # single-softmax branch; we don't touch it here so the
                # plotter can read whichever the active branch produced.

                # Cheap diagnostics — populated regardless of
                # `track_layer_diagnostics`. One batched `.tolist()` per
                # forward (per layer) replaces ~12 individual `.item()`
                # syncs from the expensive block below. The columns the
                # vfe CSV remap reads via prefixed lookups (cov/*,
                # transport/*, beta_*, attention_*) all come from this dict,
                # so the CSV stays populated even when the user opts out of
                # the expensive diagnostics path.
                #
                # SKIP when track_layer_diagnostics=True: the expensive
                # block below at line ~901 unconditionally overwrites
                # self._last_diagnostics with a richer dict, so the cheap
                # work here would be wasted (was wasted prior to this gate).
                if not self.track_layer_diagnostics:
                    with torch.no_grad():
                        _bsafe = beta.clamp(min=_BETA_LOG_FLOOR)
                        _phi_norm = phi.norm(dim=-1)
                        if is_diagonal:
                            _sigma_for_stats = sigma
                        else:
                            _sigma_for_stats = sigma.diagonal(dim1=-2, dim2=-1)
                        if kl_matrix is not None:
                            _kl_mean = kl_matrix.mean()
                            _kl_max = kl_matrix.max()
                        else:
                            _kl_mean = torch.zeros((), device=beta.device, dtype=beta.dtype)
                            _kl_max = torch.zeros((), device=beta.device, dtype=beta.dtype)
                        _stats = torch.stack([
                            beta.mean(),
                            beta.std(unbiased=False) if beta.numel() > 1 else torch.zeros((), device=beta.device, dtype=beta.dtype),
                            beta.max(dim=-1)[0].mean(),
                            -(_bsafe * _bsafe.log()).sum(-1).mean(),  # attention_entropy
                            _sigma_for_stats.mean(),
                            _sigma_for_stats.min(),
                            _sigma_for_stats.max(),
                            _sigma_for_stats.std(unbiased=False) if _sigma_for_stats.numel() > 1 else torch.zeros((), device=sigma.device, dtype=sigma.dtype),
                            sigma_p.mean(),
                            _phi_norm.mean(),
                            _phi_norm.std(unbiased=False) if _phi_norm.numel() > 1 else torch.zeros((), device=phi.device, dtype=phi.dtype),
                            _phi_norm.max(),
                            _kl_mean,
                            _kl_max,
                        ])
                        _v = _stats.detach().cpu().tolist()
                        self._last_diagnostics = {
                            'beta_mean':                  _v[0],
                            'beta_std':                   _v[1],
                            'attention_concentration':    _v[2],
                            'attention_entropy':          _v[3],
                            'sigma_q_mean':               _v[4],
                            'sigma_q_min':                _v[5],
                            'sigma_q_max':                _v[6],
                            'sigma_q_std':                _v[7],
                            'sigma_p_mean':               _v[8],
                            'phi_norm_mean':              _v[9],
                            'phi_norm_std':               _v[10],
                            'phi_norm_max':               _v[11],
                            'kl_mean':                    _v[12],
                            'kl_max':                     _v[13],
                        }

            if t == self.n_e_steps - 1 and self.track_layer_diagnostics:
                with torch.no_grad():
                    # Match the cheap-diagnostics path: when sigma is full
                    # covariance (B, N, K, K) the bare .mean/.min/.max would
                    # include off-diagonal entries that can be arbitrarily
                    # negative; extract the diagonal first so the diagnostic
                    # describes per-dim variances on both paths.
                    if is_diagonal:
                        _sigma_for_stats = sigma
                    else:
                        _sigma_for_stats = sigma.diagonal(dim1=-2, dim2=-1)
                    # attention_entropy_loss matches the runtime aux-loss form
                    # at line ~555: tau * sum(beta * log beta) + tau * log(N) *
                    # sum(beta). Omitting the +tau*log(N)*sum(beta) constant
                    # makes the diagnostic disagree with the actual loss term
                    # the kappa-equilibrium uses by exactly tau*log(N)*sum(beta).
                    _N_eff = max(beta.shape[-1], 1)
                    _log_N = math.log(_N_eff)
                    self._last_diagnostics = {
                        # Gradient norms (E-step)
                        'nat_grad_mu_norm': nat_grad_mu.norm().item(),
                        'nat_grad_sigma_norm': nat_grad_sigma.norm().item(),
                        'grad_mu_norm': grad_mu.norm().item(),
                        'grad_sigma_norm': grad_sigma.norm().item(),
                        # Attention statistics
                        'beta_mean': beta.mean().item(),
                        # unbiased=False (correction=0, population std) — diagnostics
                        # describe the full batch×seq grid, not a sample from a larger
                        # population. Also avoids the std-of-numel=1 warning during
                        # _generate_sample (B=N=1 for the first token).
                        'beta_std': beta.std(unbiased=False).item(),
                        'kl_mean': kl_matrix.mean().item() if kl_matrix is not None else 0.0,
                        'kl_max': kl_matrix.max().item() if kl_matrix is not None else 0.0,
                        # Share the clamp+log between both entropy metrics — the
                        # log dominates the FLOP count here.
                        **(lambda _bsafe, _blog: {
                            'attention_entropy': float((-(_bsafe * _blog).sum(-1).mean()).item()),
                            'attention_entropy_loss': (
                                float((
                                    _kappa * self._dim_scale * (_bsafe * _blog).sum()
                                    + _kappa * self._dim_scale * _log_N * beta.sum()
                                ).item())
                                if self.include_attention_entropy else 0.0
                            ),
                        })(beta.clamp(min=_BETA_LOG_FLOOR), beta.clamp(min=_BETA_LOG_FLOOR).log()),
                        'attention_concentration': beta.max(dim=-1)[0].mean().item(),
                        # Covariance health (per-dim variances; diagonal of sigma when full-cov)
                        'sigma_q_mean': _sigma_for_stats.mean().item(),
                        'sigma_q_min': _sigma_for_stats.min().item(),
                        'sigma_q_max': _sigma_for_stats.max().item(),
                        # unbiased=False (population std) — matches adjacent stats
                        # and avoids NaN on numel=1 (B=N=1 single-token generate).
                        'sigma_q_std': _sigma_for_stats.std(unbiased=False).item(),
                        'sigma_p_mean': sigma_p.mean().item(),
                        # Phi norms — share one norm() reduction across all three stats.
                        # unbiased=False on .std() — population std, well-defined for
                        # numel=1 (B=N=1 single-token generate path).
                        **(lambda _pn: {
                            'phi_norm_mean': _pn.mean().item(),
                            'phi_norm_std': _pn.std(unbiased=False).item(),
                            'phi_norm_max': _pn.max().item(),
                        })(phi.norm(dim=-1)),
                    }
                    # Per-dimension KL(q*||p) for prior-belief gap
                    if is_diagonal:
                        kl_qp = _diag_kl(mu, mu_p, sigma, sigma_p, eps=eps)
                        self._last_diagnostics['prior_belief_kl_mean'] = kl_qp.sum(-1).mean().item()
                        self._last_diagnostics['prior_belief_kl_max'] = kl_qp.sum(-1).max().item()
                        # unbiased=False — see note above for beta_std.
                        self._last_diagnostics['prior_belief_kl_std'] = kl_qp.sum(-1).std(unbiased=False).item()

                    # Free-energy trajectory (diagonal covariance only; empty
                    # list on full-cov paths).  Downstream can assert
                    # monotonicity as a hard correctness gate in unit tests.
                    # Note: f_history[-1] is F measured at the START of the
                    # final iteration (before that iteration's μ/σ/φ updates),
                    # which is why we also expose `f_pre_final_update` as an
                    # alias to make the semantics explicit. The post-final-
                    # update F is not measured (computing it would require an
                    # extra β + KL recomputation after step 7).
                    if f_history:
                        self._last_diagnostics['f_history'] = list(f_history)
                        self._last_diagnostics['f_final'] = f_history[-1]
                        self._last_diagnostics['f_pre_final_update'] = f_history[-1]
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
            #    Item 2 (e_mu_q_trust): optional sigma-whitened per-component
            #    clamp on the proposed delta_mu. None ⇒ pure path.
            _delta_mu = self.e_mu_lr * nat_grad_mu
            if self.e_mu_q_trust is not None:
                _delta_mu = apply_mu_trust_region(
                    _delta_mu, sigma,
                    trust=self.e_mu_q_trust,
                    is_diagonal=is_diagonal,
                    eps=eps,
                )
            mu = mu - _delta_mu

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

            # 7. Phi update with preconditioning.
            # Non-flat path threads the pairwise Omega builder so autograd
            # captures dF/dφ through both the φ-exp pair AND the φ-dependent
            # δ pre/post multiplication. δ itself doesn't depend on φ (it's a
            # function of μ + connection params), but the φ-exp factors flank
            # exp(δ·G) and so the gradient w.r.t. φ runs through them.
            if self.use_non_flat_transport:
                phi = self._update_phi_nonflat(
                    phi, mu, sigma, is_diagonal, mask, eps, _kappa,
                )
            else:
                # block_exp_pairs is intentionally not forwarded — _update_phi
                # must rebuild Omega from a fresh phi leaf for the autograd
                # path. The dead arg is dropped from the call to make this
                # explicit. Short-circuit when e_phi_lr==0 to skip the full
                # autograd.grad + matrix_exp rebuild that would only produce
                # a zero update.
                if self.e_phi_lr > 0.0:
                    phi = self._update_phi(phi, mu, sigma, is_diagonal, mask, eps, kappa=_kappa)

            # Item 1 (e_nan_check): NaN/Inf sentinel at end-of-iteration.
            # Default 'off' is a no-op. 'warn' logs and continues. 'revert'
            # restores _iter_snapshot and breaks. 'abort' propagates the
            # VFENonFiniteError to the trainer.
            if self.e_nan_check != 'off':
                try:
                    check_finite(
                        {'mu': mu, 'sigma': sigma, 'phi': phi},
                        mode=self.e_nan_check,
                        step_label=f'iter_{t}_post_update',
                    )
                except VFENonFiniteError:
                    if self.e_nan_check == 'revert' and _iter_snapshot is not None:
                        mu, sigma, phi = _iter_snapshot
                        break
                    raise

            # Final-iteration per-head diagnostic snapshot. Rebuild
            # block_exp_pairs from the post-phi-update φ so the plot reflects
            # the converged frame, not the iteration-start frame. Performed
            # only once per forward (`t == n_e_steps - 1`) AND only when the
            # trainer has opted in (`_capture_attention_state=True`) or the
            # expensive diagnostics path is active. The trainer flips the
            # flag for the steps it intends to plot; the snapshot is a fresh
            # compute_gauge_transport call (matrix-exp + inv per token per
            # block) so skipping it on every non-eval step is the dominant
            # per-forward win.
            if t == self.n_e_steps - 1 and (
                self._capture_attention_state or self.track_layer_diagnostics
            ):
                if is_diagonal:
                    _last_sigma = sigma.detach()
                else:
                    _last_sigma = sigma.detach().diagonal(dim1=-2, dim2=-1)
                with torch.no_grad():
                    _post_phi_bep = compute_gauge_transport(
                        phi.detach(),
                        self.generators,
                        self.irrep_dims,
                        enforce_orthogonal=self.enforce_orthogonal,
                    )
                # Store the kappa tensor (not a Python float) — converting
                # on demand in _compute_per_head_beta avoids a host sync
                # on every forward.
                _kappa_cached = (
                    _kappa.detach() if isinstance(_kappa, torch.Tensor)
                    else torch.tensor(float(_kappa))
                )
                self._last_attention_state = {
                    'mu_q': mu.detach(),
                    'sigma_q': _last_sigma,
                    # Post-E-step phi for cross-coupling phi_energy_partition
                    # and any holonomy/transport-norm diagnostic that needs the
                    # gauge coordinates directly. Cheap: phi is already a leaf
                    # tensor at this point.
                    'phi': phi.detach(),
                    'block_exp_pairs': [
                        (
                            p[0].detach() if p[0] is not None else None,
                            p[1].detach() if p[1] is not None else None,
                        )
                        for p in _post_phi_bep
                    ],
                    'kappa': _kappa_cached,
                    'irrep_dims': list(self.irrep_dims) if self.irrep_dims else None,
                    # RoPE state for plot reconstruction. Without these the
                    # per-head plotter at trainer.py:_compute_per_head_beta
                    # silently dropped RoPE and rendered a non-position-aware
                    # β, even when training-time β was RoPE-weighted. The
                    # plotter applies _apply_rope(mu_q, base=rope_base) when
                    # use_rope is True to match the kernel exactly.
                    'use_rope': bool(self.use_rope),
                    'rope_base': float(self.rope_base),
                }

        # Cache an auxiliary scalar F to deliver gradients to raw_c0, raw_b0,
        # log_kappa. None when neither hyperparameter is learnable. The model
        # walks stack.blocks post-stack and adds these to CE.
        self._aux_hyperparam_loss = self._auxiliary_hyperparam_loss(
            mu=mu, sigma=sigma, phi=phi,
            mu_p=mu_p, sigma_p=sigma_p, mask=mask,
        )

        return BeliefState(mu=mu, sigma=sigma, phi=phi)

    def _update_phi(
        self,
        phi: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        is_diagonal: bool,
        mask: Optional[torch.Tensor],
        eps: float,
        kappa: "torch.Tensor | float",
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
            _kappa = kappa
            # Omega exponentials are rebuilt inside compute_kl_attention from
            # phi_for_grad — the caller's block_exp_pairs cache cannot be
            # reused here because it was built from a phi leaf that is not in
            # this subgraph.
            _per_head_kappa_path = (
                isinstance(_kappa, torch.Tensor)
                and _kappa.dim() == 1
                and self.irrep_dims
                and len(self.irrep_dims) > 1
            )
            if _per_head_kappa_path:
                # Per-head φ gradient under kappa_per_head=True. Loop over
                # heads to keep each κ_h scoped to its own block — passing
                # a (H,) tensor to compute_kl_attention's single-softmax
                # logits divisor would broadcast incorrectly. The full
                # phi_for_grad is reused across heads (phi is shared); each
                # head's alignment_loss accumulates into the same leaf so
                # autograd.grad produces the summed phi-gradient at the
                # bottom of this block.
                alignment_loss = phi_for_grad.new_zeros(())
                block_start = 0
                mu_d = mu.detach()
                sigma_d = sigma.detach() if sigma is not None else None
                for h, d_h in enumerate(self.irrep_dims):
                    block_end = block_start + d_h
                    mu_h = mu_d[..., block_start:block_end]
                    sigma_h = (
                        sigma_d[..., block_start:block_end]
                        if sigma_d is not None
                        else None
                    )
                    gen_h = self.generators[:, block_start:block_end, block_start:block_end]
                    kappa_h = _kappa[h]
                    beta_phi_h, kl_h = compute_kl_attention(
                        mu_h, sigma_h, phi_for_grad, gen_h,
                        [d_h], kappa_h, mask,
                        use_rope=self.use_rope,
                        rope_base=self.rope_base,
                        enforce_orthogonal=self.enforce_orthogonal,
                        mask_self_attention=self.mask_self_attention,
                        exact_diagonal_transport=self.exact_diagonal_transport,
                    )
                    if self.include_attention_entropy:
                        beta_safe_h = beta_phi_h.clamp(min=_BETA_LOG_FLOOR)
                        _F_h = (
                            (beta_phi_h * kl_h).sum()
                            + kappa_h * math.sqrt(d_h)
                            * (beta_safe_h * beta_safe_h.log()).sum()
                        )
                        alignment_loss = alignment_loss + self.lambda_align * _F_h
                    else:
                        alignment_loss = alignment_loss + (
                            self.lambda_align * (beta_phi_h.detach() * kl_h).sum()
                            + self.lambda_soft * (beta_phi_h * kl_h.detach()).sum()
                        )
                    block_start = block_end
            else:
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
                    # The α·KL(q||p) self-coupling term is intentionally OMITTED from _F:
                    # it depends only on (q, p), neither of which depends on φ, so its
                    # partial derivative w.r.t. φ is identically zero (and including it
                    # would only add an attached-but-zero-grad scalar). The φ-loss therefore
                    # collapses to the (β·KL + τ·β·log(β/π)) terms that actually couple to φ
                    # through Ω_ij = exp(φ_i)·exp(-φ_j).
                    # Verified by tests/test_entropy_envelope.py: production config matches
                    # the envelope prediction to ~1e-18 for any λ_soft.
                    beta_safe = beta_phi.clamp(min=_BETA_LOG_FLOOR)
                    _F = (beta_phi * kl_h).sum() + _kappa * self._dim_scale * (beta_safe * beta_safe.log()).sum()
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

    def _compute_mu_sigma_grad_autograd(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        mu_p: torch.Tensor,
        sigma_p: torch.Tensor,
        phi: torch.Tensor,
        alpha_eff: 'torch.Tensor | float',
        block_exp_pairs: Optional[List[Tuple[torch.Tensor, Optional[torch.Tensor]]]],
        mask: Optional[torch.Tensor],
        kappa: 'torch.Tensor | float',
        eps: float,
        is_diagonal: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Total-derivative dF/d(mu,sigma) — NON-CANONICAL research variant.

        **Default off, and it should stay off.** The analytic kernel
        ``compute_vfe_gradients_gpu`` (and its fused sibling
        ``_fused_attention_and_vfe_gradients_block_diag``) computes the
        canonical mean-field variational message-passing gradient — the
        query-side partial holding all other q_j fixed (Beal 2003 Ch. 2;
        Winn & Bishop 2005; Friston 2010). This method instead computes
        the total derivative of the manuscript F functional via
        ``torch.autograd.grad(F_total, [mu_g, sigma_g])``, which adds the
        key-side contribution: mu_k and Sigma_k also appear in
        KL(q_i || Omega_ik q_k) for every i != k through the transported
        mean Omega_ik mu_k and the transported covariance Omega_ik Sigma_k Omega_ik^T.

        Under parallel q-updates (the case here: all positions are
        retracted simultaneously each iteration) this double-counts the
        asymmetric coupling KL(q_i || Omega_ij q_j) != KL(q_j || Omega_ji q_i):
        each q_i receives both its incoming-message gradient (query-side)
        and the gradient it should be sending to q_j (key-side) in the
        same step. Empirically (probe at
        ``scripts/diagnostic_autograd_gradient_scale_probe.py``) the
        sigma-gradient RMS ratio (autograd/analytic) grows from ~20x at
        iteration 0 to ~145x by iteration 5 at the production config; the
        per-token sigma-cosine reaches -0.3 in the tail, so the direction
        is partly orthogonal to the canonical gradient, not just rescaled.
        Step sizes ``e_mu_lr``, ``e_sigma_lr`` calibrated against the
        analytic kernel therefore overshoot dramatically when this flag
        is on, and retuning alone cannot recover the sigma direction.

        Historical context: this path was added in audit-2026-05-17 to
        suppress the F-monotone monitor warning that fires under the
        analytic path. That justification does not hold — parallel
        mean-field VMP is *not* expected to monotonically decrease F per
        iteration (Jordan et al. 1999 §6; only fully sequential
        coordinate descent gives that guarantee). The warning is
        expected, not a defect. See ``_f_monotone_step`` and
        ``post_edit_2026-05-19.md`` Wave 6 for the full diagnostic.

        Mechanics: ``beta`` is detached inside the autograd block so the
        gradient produced is the envelope-correct
        ``sum beta . dKL/d(mu,sigma)`` — the entropy term's d/d(mu,sigma)
        contribution vanishes under ``beta.detach()`` and is omitted.

        Returns:
            grad_mu: (B, N, K) total derivative dF/dmu (NOT the canonical
                mean-field gradient).
            grad_sigma: (B, N, K) (diagonal) total derivative dF/dsigma
                (NOT the canonical mean-field gradient).

        Raises:
            RuntimeError: if ``alpha_divergence != 1.0`` — this path
                reconstructs the standard KL and silently ignores the
                Rényi-alpha exponent. Either set ``alpha_divergence=1.0``
                or disable ``use_autograd_mu_sigma`` /
                ``use_non_flat_transport`` /
                ``gauge_parameterization='omega_direct'``.
            NotImplementedError: if ``is_diagonal`` is False — full-cov
                self-KL via autograd needs logdet through eigendecomposition,
                not yet wired.
        """
        if abs(self.alpha_divergence - 1.0) > 1e-9:
            raise RuntimeError(
                f"alpha_divergence={self.alpha_divergence} is incompatible "
                "with the autograd (μ,σ) gradient path: this kernel "
                "reconstructs the standard KL functional and silently "
                "ignores the Rényi α exponent. Either set alpha_divergence=1.0 "
                "or disable use_autograd_mu_sigma / use_non_flat_transport / "
                "gauge_parameterization='omega_direct'."
            )
        if not is_diagonal:
            # Self-KL requires full-cov logdet (eigendecomposition through
            # autograd). Out of scope for this fix; restrict to diagonal.
            raise NotImplementedError(
                "use_autograd_mu_sigma=True is currently only supported for "
                "diagonal_covariance=True (full-cov self-KL via autograd "
                "needs logdet through eigendecomposition; not yet wired). "
                "Set diagonal_covariance=True or disable the toggle."
            )

        with torch.enable_grad():
            mu_g = mu.detach().requires_grad_(True)
            sigma_g = sigma.detach().requires_grad_(True)
            phi_d = phi.detach()
            mu_p_d = mu_p.detach()
            sigma_p_d = sigma_p.detach()
            _kappa_d = kappa.detach() if isinstance(kappa, torch.Tensor) else kappa

            # Self-coupling: alpha * KL(q_i || p_i), diagonal Gaussians.
            kl_qp_per_dim = _diag_kl(mu_g, mu_p_d, sigma_g, sigma_p_d, eps=eps)  # (B, N, K)

            if self.E_learnable_alpha:
                # Reconstruct alpha inside the autograd graph so the product-rule
                # term d(alpha_k)/d(mu,sigma) = -(alpha_k^2 / c0) * d(KL_k)/dtheta
                # is captured automatically (matches the analytic path's
                # alpha_c0 correction in compute_vfe_gradients_gpu).
                alpha_in_graph = self.get_bayesian_alpha(
                    mu_g, mu_p_d, sigma_g, sigma_p_d
                )
                self_kl_term = (alpha_in_graph * kl_qp_per_dim).sum()
            elif isinstance(alpha_eff, torch.Tensor):
                self_kl_term = (alpha_eff.detach() * kl_qp_per_dim).sum()
            else:
                self_kl_term = float(alpha_eff) * kl_qp_per_dim.sum()

            # Alignment: lambda_align * Sum beta_ij . KL(q_i || Omega_ij q_j).
            # Reuse the same block_exp_pairs cache built from current phi (the
            # cache is detached from autograd; mu_g and sigma_g flow through
            # the KL computation itself, which is what we differentiate).
            beta_g, kl_g = compute_kl_attention(
                mu_g, sigma_g, phi_d, self.generators,
                self.irrep_dims, _kappa_d, mask,
                use_rope=self.use_rope,
                rope_base=self.rope_base,
                cached_block_exp_pairs=block_exp_pairs,
                enforce_orthogonal=self.enforce_orthogonal,
                mask_self_attention=self.mask_self_attention,
                exact_diagonal_transport=self.exact_diagonal_transport,
            )

            # Envelope: detach beta so the entropy gradient vanishes and only
            # Sum beta . dKL/dtheta survives. The entropy term itself has no
            # (mu, sigma) dependence under beta.detach() and is omitted.
            beta_d = beta_g.detach()

            if self.include_attention_entropy:
                alignment_term = self.lambda_align * (beta_d * kl_g).sum()
            else:
                # Legacy product-rule decomposition for the entropy-suppressed
                # surrogate path: split the d/dtheta [Sum beta . KL] derivative
                # into a beta.detach() branch (lambda_align weight) and a
                # kl.detach() branch (lambda_soft weight). Mirrors _update_phi.
                alignment_term = (
                    self.lambda_align * (beta_g.detach() * kl_g).sum()
                    + self.lambda_soft * (beta_g * kl_g.detach()).sum()
                )

            F_total = self_kl_term + alignment_term

            grad_mu, grad_sigma = torch.autograd.grad(
                F_total, [mu_g, sigma_g],
                create_graph=False, retain_graph=False,
            )

        return grad_mu, grad_sigma

    # -- non-flat transport methods ------------------------------------------

    def _iter_nonflat(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        phi: torch.Tensor,
        mu_p: torch.Tensor,
        sigma_p: torch.Tensor,
        mask: Optional[torch.Tensor],
        eps: float,
        kappa: 'torch.Tensor | float',
        alpha_eff: 'torch.Tensor | float',
        block_exp_pairs: List[Tuple[torch.Tensor, Optional[torch.Tensor]]],
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        List[Tuple[torch.Tensor, torch.Tensor]],
    ]:
        r"""One E-step iteration with non-flat transport.

        Drives the autograd path through:
            connection(μ) → δ → pairwise Ω = exp(φ·G) · exp(δ·G) · exp(-φ·G)
                → pairwise KL → β → F.

        Returns ``(grad_mu, grad_sigma, beta, kl_matrix, omega_pairs)``.
        ``omega_pairs`` is returned for diagnostic reuse (e.g., triangle
        holonomy probe) — it is detached from the autograd graph in the
        caller's downstream consumers.
        """
        is_diagonal = sigma.dim() == 3
        if not is_diagonal:
            raise NotImplementedError(
                "use_non_flat_transport=True requires diagonal_covariance=True; "
                "this should have been rejected at config __post_init__."
            )

        # Compute β + KL for the F-monotone monitor (no autograd needed here,
        # detached). The same KL is then recomputed inside the autograd
        # subgraph for the (μ, σ) gradient — duplicated compute but isolates
        # the gradient path from the monitor's bookkeeping.
        with torch.no_grad():
            delta_det = self.non_flat_connection(mu.detach(), mask=mask).detach()
            omega_pairs_det = compute_pairwise_omega_with_delta(
                phi.detach(), delta_det,
                self.generators, self.irrep_dims,
                cached_block_exp_pairs=block_exp_pairs,
                phi_spec_max=self.phi_spec_max,
            )
            beta_det, kl_det = compute_kl_attention_pairwise(
                mu.detach(), sigma.detach(), omega_pairs_det,
                self.irrep_dims, kappa,
                mask=mask, mask_self_attention=self.mask_self_attention,
                eps=1e-8,
                use_rope=self.use_rope, rope_base=self.rope_base,
            )

        # Autograd path for dF/d(μ, σ) over the full F functional. The
        # connection inputs μ here as a leaf so autograd captures the
        # "δ moves with μ" contribution.
        with torch.enable_grad():
            mu_g = mu.detach().requires_grad_(True)
            sigma_g = sigma.detach().requires_grad_(True)
            phi_d = phi.detach()
            mu_p_d = mu_p.detach()
            sigma_p_d = sigma_p.detach()
            _kappa_d = kappa.detach() if isinstance(kappa, torch.Tensor) else kappa

            # Self-coupling α · KL(q || p), per-dim diagonal.
            kl_qp_per_dim = _diag_kl(mu_g, mu_p_d, sigma_g, sigma_p_d, eps=eps)
            if self.E_learnable_alpha:
                alpha_in_graph = self.get_bayesian_alpha(
                    mu_g, mu_p_d, sigma_g, sigma_p_d,
                )
                self_kl_term = (alpha_in_graph * kl_qp_per_dim).sum()
            elif isinstance(alpha_eff, torch.Tensor):
                self_kl_term = (alpha_eff.detach() * kl_qp_per_dim).sum()
            else:
                self_kl_term = float(alpha_eff) * kl_qp_per_dim.sum()

            # Re-evaluate δ and pairwise Ω inside the graph so dF/dμ captures
            # the connection's μ-dependence.
            delta_g = self.non_flat_connection(mu_g, mask=mask)
            omega_pairs_g = compute_pairwise_omega_with_delta(
                phi_d, delta_g,
                self.generators, self.irrep_dims,
                cached_block_exp_pairs=block_exp_pairs,
                phi_spec_max=self.phi_spec_max,
            )
            beta_g, kl_g = compute_kl_attention_pairwise(
                mu_g, sigma_g, omega_pairs_g, self.irrep_dims, _kappa_d,
                mask=mask, mask_self_attention=self.mask_self_attention, eps=1e-8,
                use_rope=self.use_rope, rope_base=self.rope_base,
            )
            beta_d_in_graph = beta_g.detach()  # envelope at softmax fixed pt

            if self.include_attention_entropy:
                alignment_term = self.lambda_align * (beta_d_in_graph * kl_g).sum()
            else:
                alignment_term = (
                    self.lambda_align * (beta_g.detach() * kl_g).sum()
                    + self.lambda_soft * (beta_g * kl_g.detach()).sum()
                )

            F_total = self_kl_term + alignment_term
            grad_mu, grad_sigma = torch.autograd.grad(
                F_total, [mu_g, sigma_g],
                create_graph=False, retain_graph=False,
            )

        return grad_mu, grad_sigma, beta_det, kl_det, omega_pairs_det

    def _update_phi_nonflat(
        self,
        phi: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        is_diagonal: bool,
        mask: Optional[torch.Tensor],
        eps: float,
        kappa: 'torch.Tensor | float',
    ) -> torch.Tensor:
        r"""φ retraction step with non-flat transport.

        Mirrors :meth:`_update_phi` but routes the KL functional through the
        pairwise Omega kernel. δ is computed from detached μ — the φ-update
        treats the connection as a function of the current belief snapshot,
        not a moving target inside the iteration.
        """
        with torch.enable_grad():
            phi_for_grad = phi.detach().requires_grad_(True)
            _kappa = kappa if kappa is not None else self.effective_kappa

            # δ from detached μ. The connection is a function of μ + params,
            # so for the φ-update we freeze μ to isolate the dF/dφ direction.
            with torch.no_grad():
                delta = self.non_flat_connection(mu.detach(), mask=mask).detach()

            # Pairwise Omega built from phi_for_grad (no cache forwarded —
            # the cache was built from `phi` which is detached, and forwarding
            # it would sever the autograd subgraph through phi_for_grad).
            omega_pairs = compute_pairwise_omega_with_delta(
                phi_for_grad, delta,
                self.generators, self.irrep_dims,
                cached_block_exp_pairs=None,
                phi_spec_max=self.phi_spec_max,
            )
            beta_phi, kl_h = compute_kl_attention_pairwise(
                mu.detach(),
                sigma.detach() if sigma is not None else None,
                omega_pairs, self.irrep_dims, _kappa,
                mask=mask, mask_self_attention=self.mask_self_attention, eps=1e-8,
                use_rope=self.use_rope, rope_base=self.rope_base,
            )

            if self.include_attention_entropy:
                beta_safe = beta_phi.clamp(min=_BETA_LOG_FLOOR)
                _F = (
                    (beta_phi * kl_h).sum()
                    + _kappa * self._dim_scale * (beta_safe * beta_safe.log()).sum()
                )
                alignment_loss = self.lambda_align * _F
            else:
                alignment_loss = (
                    self.lambda_align * (beta_phi.detach() * kl_h).sum()
                    + self.lambda_soft * (beta_phi * kl_h.detach()).sum()
                )

            if alignment_loss.grad_fn is not None:
                grad_phi = torch.autograd.grad(
                    alignment_loss, phi_for_grad,
                    create_graph=False, retain_graph=False,
                )[0]
                grad_phi = precondition_phi_gradient(
                    grad_phi, phi,
                    mode=self.phi_preconditioner_mode,
                    preconditioner=self._phi_preconditioner,
                    generators=self.generators,
                )
                phi = _retract_phi(
                    phi, grad_phi, self.generators,
                    step_size=self.e_phi_lr,
                    gauge_group=self.gauge_group,
                    project_slk=self.phi_project_slk,
                    trace_clamp=self.phi_trace_clamp,
                    irrep_dims=self.irrep_dims,
                )

        return phi

    # -- omega-direct forward path -------------------------------------------

    def _forward_omega_direct(
        self,
        beliefs: BeliefState,
        priors: BeliefState,
        mask: Optional[torch.Tensor],
    ) -> BeliefState:
        r"""E-step inner loop with :math:`\Omega \in G` as the gauge state.

        Iterates :math:`(\mu, \sigma, \Omega)` instead of :math:`(\mu, \sigma,
        \phi)`. φ is held at its encode-time value (used only by RoPE / mass_φ
        / diagnostics). The transport is built from the stored per-token
        :math:`(\Omega_i, \Omega_i^{-1})` pair via
        :func:`compute_pairwise_omega_from_endpoints`.

        Update rule (per block, right-invariant Riemannian step):
            :math:`\Omega \mapsto \Omega \cdot \exp(-\eta\, X_{\text{proj}})`,
            :math:`X = \mathrm{proj}_{\mathrm{span}(G^a)}(\Omega^{-1} dF/d\Omega)`.

        :func:`project_omega_to_slk` rescales each block to det=1 every
        :attr:`omega_normalize_every` iterations when the flag is on (controls
        Killing-degenerate trace drift on :math:`\mathbb{R}\cdot I \subset
        \mathfrak{gl}(K)`).
        """
        # Same guard as _compute_mu_sigma_grad_autograd: the omega-direct
        # path reconstructs standard KL and would silently drop the Rényi α.
        if abs(self.alpha_divergence - 1.0) > 1e-9:
            raise RuntimeError(
                f"alpha_divergence={self.alpha_divergence} is incompatible "
                "with gauge_parameterization='omega_direct': this path "
                "reconstructs the standard KL functional and silently "
                "ignores the Rényi α exponent. Either set alpha_divergence=1.0 "
                "or use gauge_parameterization='phi'."
            )
        if beliefs.omega is None:
            raise RuntimeError(
                "gauge_parameterization='omega_direct' requires "
                "beliefs.omega to be populated. This should happen in "
                "VFEModel.forward right after the positional BCH step. If "
                "you're calling VFEEStep directly, build the omega pair via "
                "transformer.vfe.omega_direct.init_omega_from_phi."
            )

        mu = beliefs.mu
        sigma = beliefs.sigma
        phi = beliefs.phi
        omega = list(beliefs.omega)        # list of (Ω_h, Ω_h_inv)
        mu_p = priors.mu
        # Detach to match the phi-mode forward and the CLAUDE.md hard
        # constraint: the E-step reads sigma_p but must not write gradients
        # to it. Without this detach, get_bayesian_alpha() at the call below
        # builds a live autograd path through priors.sigma; downstream
        # consumers are all detached today so no actual leak materializes,
        # but the asymmetry with the phi-mode path is a future-edit hazard.
        sigma_p = priors.sigma.detach()

        is_diagonal = self.diagonal_covariance
        eps = 1e-6
        self._last_diagnostics = {}

        if not is_diagonal:
            # Caught at config; defensive guard.
            raise RuntimeError(
                "omega-direct + diagonal_covariance=False reached the runtime "
                "but should have been blocked at __post_init__."
            )

        _kappa = self.effective_kappa
        alpha_c0_full_hoisted = self._get_alpha_c0()

        f_history: List[float] = []
        f_prev: Optional[float] = None
        f_monotone_rel_tol = 0.05
        f_monotone_abs_tol = 1e-2

        # Pre-iteration snapshot for Item 1 (NaN sentinel revert/abort modes).
        _nan_check_armed: bool = self.e_nan_check in ('revert', 'abort')
        _iter_snapshot_od: Optional[
            Tuple[torch.Tensor, torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]
        ] = None

        for t in range(self.n_e_steps):
            if _nan_check_armed:
                _iter_snapshot_od = (
                    mu.detach().clone(),
                    sigma.detach().clone(),
                    [(o.detach().clone(), oi.detach().clone()) for (o, oi) in omega],
                )
            alpha_eff = self.alpha
            if self.E_learnable_alpha:
                alpha_eff = self.get_bayesian_alpha(mu, mu_p, sigma, sigma_p)

            # 1. Build pairwise Omega from current per-token Ω, optionally with
            # non-flat δ. The delta dependence on μ is captured by autograd
            # when we re-enter the graph below for the (μ, σ) gradient.
            if self.use_non_flat_transport:
                with torch.no_grad():
                    delta_det = self.non_flat_connection(mu.detach(), mask=mask).detach()
                omega_pairs_det = compute_pairwise_omega_from_endpoints(
                    [(o.detach(), oi.detach()) for (o, oi) in omega],
                    self.irrep_dims,
                    delta=delta_det,
                    generators=self.generators,
                    phi_spec_max=self.phi_spec_max,
                )
            else:
                omega_pairs_det = compute_pairwise_omega_from_endpoints(
                    [(o.detach(), oi.detach()) for (o, oi) in omega],
                    self.irrep_dims,
                    phi_spec_max=self.phi_spec_max,
                )

            with torch.no_grad():
                beta_det, kl_det = compute_kl_attention_pairwise(
                    mu.detach(), sigma.detach(), omega_pairs_det,
                    self.irrep_dims, _kappa,
                    mask=mask, mask_self_attention=self.mask_self_attention,
                    eps=1e-8,
                )

            # 2. Compute dF/dΩ + dF/dμ + dF/dσ via autograd over the pairwise-Ω F functional.
            #    Each Ω_h is treated as a leaf; we collect gradients per block.
            with torch.enable_grad():
                mu_g = mu.detach().requires_grad_(True)
                sigma_g = sigma.detach().requires_grad_(True)
                omega_g: List[Tuple[torch.Tensor, torch.Tensor]] = []
                grad_target_Om: List[torch.Tensor] = []
                for o, oi in omega:
                    o_g = o.detach().requires_grad_(True)
                    # Pair inverse with the SAME leaf so it tracks Ω_h_new
                    # via the matrix-exp retraction; we do NOT make the inverse
                    # an independent leaf, since that would let it drift out of
                    # consistency with Ω_h.
                    omega_g.append((o_g, oi.detach()))
                    grad_target_Om.append(o_g)
                mu_p_d = mu_p.detach()
                sigma_p_d = sigma_p.detach()
                _kappa_d = _kappa.detach() if isinstance(_kappa, torch.Tensor) else _kappa

                # Self-coupling α · KL(q || p) — independent of Ω.
                kl_qp_per_dim = _diag_kl(mu_g, mu_p_d, sigma_g, sigma_p_d, eps=eps)
                if self.E_learnable_alpha:
                    alpha_in_graph = self.get_bayesian_alpha(
                        mu_g, mu_p_d, sigma_g, sigma_p_d,
                    )
                    self_kl_term = (alpha_in_graph * kl_qp_per_dim).sum()
                elif isinstance(alpha_eff, torch.Tensor):
                    self_kl_term = (alpha_eff.detach() * kl_qp_per_dim).sum()
                else:
                    self_kl_term = float(alpha_eff) * kl_qp_per_dim.sum()

                # Non-flat δ inside the graph (so dF/dμ captures it).
                if self.use_non_flat_transport:
                    delta_g = self.non_flat_connection(mu_g, mask=mask)
                else:
                    delta_g = None

                pairwise_g = compute_pairwise_omega_from_endpoints(
                    omega_g, self.irrep_dims,
                    delta=delta_g,
                    generators=self.generators if delta_g is not None else None,
                    phi_spec_max=self.phi_spec_max,
                )
                beta_g, kl_g = compute_kl_attention_pairwise(
                    mu_g, sigma_g, pairwise_g, self.irrep_dims, _kappa_d,
                    mask=mask, mask_self_attention=self.mask_self_attention, eps=1e-8,
                )
                beta_d_in_graph = beta_g.detach()

                if self.include_attention_entropy:
                    alignment_term = self.lambda_align * (beta_d_in_graph * kl_g).sum()
                else:
                    alignment_term = (
                        self.lambda_align * (beta_g.detach() * kl_g).sum()
                        + self.lambda_soft * (beta_g * kl_g.detach()).sum()
                    )

                F_total = self_kl_term + alignment_term
                grads = torch.autograd.grad(
                    F_total, [mu_g, sigma_g, *grad_target_Om],
                    create_graph=False, retain_graph=False,
                    allow_unused=True,
                )
                grad_mu = grads[0]
                grad_sigma = grads[1]
                grad_omega: List[torch.Tensor] = []
                for h, _d_h in enumerate(self.irrep_dims):
                    g_h = grads[2 + h]
                    if g_h is None:
                        g_h = torch.zeros_like(omega[h][0])
                    grad_omega.append(g_h)

            # 3. F-monotone monitor (detached, scalar). Gated by
            # monitor_monotonicity OR track_layer_diagnostics so default
            # production runs don't pay the .item() syncs.
            if self.monitor_monotonicity or self.track_layer_diagnostics:
                with torch.no_grad():
                    f_prev = _f_monotone_step(
                        mu_q=mu, mu_p=mu_p, sigma_q=sigma, sigma_p=sigma_p,
                        eps=eps,
                        beta_det=beta_det, kl_det=kl_det,
                        alpha_eff=alpha_eff, kappa=_kappa,
                        dim_scale=self._dim_scale,
                        include_attention_entropy=self.include_attention_entropy,
                        lambda_align=self.lambda_align,
                        alpha_div=self.alpha_divergence,
                        f_history=f_history, f_prev=f_prev,
                        f_abs_tol=f_monotone_abs_tol,
                        f_rel_tol=f_monotone_rel_tol,
                        iter_idx=t,
                        label="E-step (omega-direct)",
                    )

            # 5. Natural gradient projection for (μ, σ).
            nat_grad_mu, nat_grad_sigma = compute_natural_gradient_gpu(
                grad_mu, grad_sigma, sigma, eps=eps,
            )

            # Diagnostics (last iter only).
            if t == self.n_e_steps - 1:
                self._last_attention = beta_det
                self._last_kl_matrix = kl_det
                if self.track_layer_diagnostics:
                    with torch.no_grad():
                        # Sample (1st batch element, 1st block) of Ω for log.
                        Om0 = omega[0][0]
                        _Om0_fro = Om0.norm(dim=(-2, -1))
                        _stats = torch.stack([
                            nat_grad_mu.norm(),
                            nat_grad_sigma.norm(),
                            grad_mu.norm(),
                            grad_sigma.norm(),
                            beta_det.mean(),
                            # unbiased=False — population std, defined for numel=1
                            # (B=N=1 during single-token generate forward).
                            beta_det.std(unbiased=False)
                            if beta_det.numel() > 1
                            else torch.zeros((), device=beta_det.device, dtype=beta_det.dtype),
                            kl_det.mean(),
                            kl_det.max(),
                            sigma.mean(),
                            sigma.min(),
                            sigma.max(),
                            _Om0_fro.mean(),
                            _Om0_fro.max(),
                            torch.linalg.det(Om0.float()).mean(),
                        ])
                        _v = _stats.detach().cpu().tolist()
                        self._last_diagnostics = {
                            'nat_grad_mu_norm':    _v[0],
                            'nat_grad_sigma_norm': _v[1],
                            'grad_mu_norm':        _v[2],
                            'grad_sigma_norm':     _v[3],
                            'beta_mean':           _v[4],
                            'beta_std':            _v[5],
                            'kl_mean':             _v[6],
                            'kl_max':              _v[7],
                            'sigma_q_mean':        _v[8],
                            'sigma_q_min':         _v[9],
                            'sigma_q_max':         _v[10],
                            'omega_fro_mean':      _v[11],
                            'omega_fro_max':       _v[12],
                            'omega_det_mean':      _v[13],
                        }
                        if f_history:
                            self._last_diagnostics['f_history'] = list(f_history)
                            self._last_diagnostics['f_final'] = f_history[-1]

            # 6. Mean and covariance updates.
            #    Item 2 (e_mu_q_trust): optional sigma-whitened delta_mu clamp.
            _delta_mu = self.e_mu_lr * nat_grad_mu
            if self.e_mu_q_trust is not None:
                _delta_mu = apply_mu_trust_region(
                    _delta_mu, sigma,
                    trust=self.e_mu_q_trust,
                    is_diagonal=is_diagonal,
                    eps=eps,
                )
            mu = mu - _delta_mu
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

            # 7. Omega update via right-invariant natural gradient.
            omega = omega_natural_grad_step(
                omega, grad_omega,
                generators=self.generators,
                irrep_dims=self.irrep_dims,
                killing_cache=self._omega_killing_cache,
                lr=self.e_phi_lr,    # reuse φ-step LR as the algebra-step LR
            )

            # 8. Periodic det renormalization (controls Killing-degenerate drift).
            #    Item 5 (phi_strict_glplus): threading the strict mode through
            #    so an ablation that demands GL+(K) only sees the abort path.
            if (
                self.omega_project_slk
                and self.omega_normalize_every > 0
                and (t + 1) % self.omega_normalize_every == 0
            ):
                omega = project_omega_to_slk(
                    omega, self.irrep_dims,
                    strict=self.phi_strict_glplus,
                )

            # Item 1 (e_nan_check): end-of-iteration NaN sentinel.
            if self.e_nan_check != 'off':
                _omega_flat = [o for pair in omega for o in pair]
                try:
                    check_finite(
                        {
                            'mu': mu, 'sigma': sigma,
                            **{f'omega_h{h}': _omega_flat[2 * h] for h in range(len(omega))},
                        },
                        mode=self.e_nan_check,
                        step_label=f'iter_{t}_post_update_omega_direct',
                    )
                except VFENonFiniteError:
                    if self.e_nan_check == 'revert' and _iter_snapshot_od is not None:
                        mu, sigma, omega = _iter_snapshot_od
                        break
                    raise

        # Hyperparameter aux loss (raw_c0, raw_b0, log_kappa gradient delivery).
        # phi is held at encode-time value in this path but is still the input
        # the attention term reads from; aux loss only needs (mu, sigma, phi,
        # mu_p, sigma_p) and uses them all detached so the omega-direct
        # iteration semantics are not perturbed.
        self._aux_hyperparam_loss = self._auxiliary_hyperparam_loss(
            mu=mu, sigma=sigma, phi=phi,
            mu_p=mu_p, sigma_p=sigma_p, mask=mask,
        )

        return BeliefState(mu=mu, sigma=sigma, phi=phi, omega=omega)
