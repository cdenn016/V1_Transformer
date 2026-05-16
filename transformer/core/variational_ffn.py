"""
Variational Feed-Forward Networks for Gauge Transformer
========================================================

VFE E-step belief evolution: iteratively updates beliefs (mu, Sigma) and
optionally gauge frames (phi) by minimizing variational free energy with
dynamic attention weights beta recomputed at each iteration.

Supports SO(3), SO(N), and GL(K) gauge groups. Generator count determines
the group: phi_dim=3 for SO(3), N(N-1)/2 for SO(N), K**2 for GL(K).

Key features:
- Dynamic beta: attention weights co-evolve with beliefs each VFE iteration
- Sigma softmax coupling: includes dBeta/dSigma term in VFE gradients
- Block-diagonal KL decomposition via irrep_dims for memory efficiency
- Fused attention+gradient paths for diagonal covariance mode
- Multi-head VFE: per-head beta_h through VFE iterations (multihead_vfe)
- Isotropic covariance: force Sigma = sigma^2 I (isotropic_covariance)
- DEQ mode: implicit differentiation for E-step fixed point (use_deq)
- Amortized inference: gradient flow through priors for learned E-step init
- Learnable alpha: Bayesian precision via Gamma-Normal conjugacy
- PriorBank: token-dependent priors via token_ids (prior_bank / use_prior_bank)
- exact_diagonal_transport: lifts diagonal sigma to full for exact transport

Mathematical Foundation:
-----------------------
Free Energy (E-STEP):
    F = alpha * Sum_i KL(q_i||p_i)                         # Prior consistency (self-coupling)
      + lambda_belief * Sum_{i,j} beta_ij * KL(q_i||Omega_{ij}q_j)  # Belief alignment (direct: beta * dKL/dtheta)
      + lambda_softmax * Sum_{i,j} KL_ij * d(beta_ij)/dtheta        # Attention-variance coupling (dbeta/dtheta * KL)
      + kappa * Sum_{i,j} beta_ij * log(beta_ij / pi_ij)            # Attention entropy (manuscript eq:free_energy_functional_final);
                                                                    # opt-in via include_attention_entropy (default True), uniform prior π=1/N.

E-step: Minimize F w.r.t. mu, Sigma (does not see targets — outer CE loss
    in compute_free_energy_loss provides the likelihood term).
M-step: Minimize outer (F + CE) w.r.t. embeddings, W_out.

Gradient computation:
    dF/dtheta for theta = {mu_q, Sigma_q, mu_p, Sigma_p, phi}

With natural gradient projection:
    Delta_theta = -eta * F_inv(theta) * grad_F(theta)

Where F(theta) is the Fisher-Rao metric.
"""

import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

from transformer.core.gauge_utils import (
    newton_schulz_orthogonalize,
    fused_block_matrix_exp_pairs,
)

import transformer.core.vfe_utils as _vfe_utils_mod

from transformer.core.vfe_diagnostics import (
    print_vfe_grad_debug as _print_vfe_grad_debug_impl,
    record_iteration_diagnostics as _record_iteration_diagnostics_impl,
)
from transformer.core.vfe_utils import (
    apply_natural_gradient_step as _apply_natgrad_step,
    retract_sigma_e_step as _retract_sigma_impl,
)
from transformer.core.vfe_utils import (
    _grad_norm,
    _per_pos_stats,
    _aggregate_multihead_vfe_debug,
    squeeze_trailing_singletons,
    _safe_spd_inv,
    _safe_eigh,
    spd_eigfloor,
    retract_spd_torch,
    retract_spd_diagonal_torch,
    _retract_phi,
)

from transformer.core.active_inference import compute_ai_gradients


# =============================================================================
# Implicit EM Gradient — extracted to transformer/core/vfe_implicit_em.py
# =============================================================================
# The two autograd Functions and the scale-factor helper below were moved
# out of this file as part of a side-quest extraction.  They are re-exported
# here so external callers that do
#     from transformer.core.variational_ffn import ImplicitEMGradient
# continue to work unchanged.  See vfe_implicit_em.py for the definitions
# and the full derivation.
from transformer.core.vfe_implicit_em import (
    ImplicitEMGradient,
    ImplicitEMGradientSigma,
    compute_implicit_em_scales,
)


# Import attention computation for dynamic β
from transformer.core.attention import compute_attention_weights
from transformer.core.transport_ops import compute_transport_operators, compute_transport_operators_direct

# Numerical event monitor (shared with attention.py)
from math_utils.numerical_monitor import record as _nr



# =============================================================================
# VFE Gradient Computation (moved to vfe_gradients.py)
# =============================================================================
# These functions are defined in transformer.core.vfe_gradients and imported
# here for backward compatibility.  Do not add gradient logic here.
from transformer.core.vfe_gradients import (
    _compute_vfe_gradients_block_diagonal,
    _compute_vfe_gradients_block_diagonal_diag,
    _fused_attention_and_vfe_gradients_block_diag,
    compute_vfe_gradients_gpu,
    compute_natural_gradient_gpu,
    _compute_rope_full_gauge_gradient_per_head,
)
from transformer.core.gauge_preconditioner import (
    apply_cartan_preconditioning,
    apply_killing_form_natural_gradient,
    apply_pullback_natural_gradient,
)


# retract_spd_torch and retract_spd_diagonal_torch are imported from vfe_utils above.


# =============================================================================
# DEQ Fixed-Point Implicit Differentiation
# =============================================================================
# Extracted to transformer/core/vfe_deq.py.  The two autograd Functions
# below are re-exported here so external callers that do
#     from transformer.core.variational_ffn import DEQFixedPoint
# continue to work unchanged.  See vfe_deq.py for the implementations,
# the Neumann-series derivation, and the divergence safeguards.
from transformer.core.vfe_deq import (
    DEQFixedPoint,
    DEQFixedPointFull,
    make_deq_step_fn as _make_deq_step_fn_free,
    make_deq_step_fn_with_phi as _make_deq_step_fn_with_phi_free,
)

# =============================================================================
# Closed-form E-step — extracted to transformer/core/vfe_closed_form.py
# =============================================================================
# The 500-line _closed_form_e_step method body was moved to its own module
# as part of the side-quest refactor.  The method remains on
# VariationalFFNDynamic as a thin delegator so external callers continue
# to work.  See vfe_closed_form.py for the diagonal / full-cov branches,
# Picard resolve loops, and phi evolution.
from transformer.core.vfe_closed_form import (
    run_closed_form_e_step as _run_closed_form_e_step,
)


# =============================================================================
# Dynamic-β VFE: Full Active Inference with Evolving Attention (RECOMMENDED!)
# =============================================================================

class VariationalFFNDynamic(nn.Module):
    """
    Dynamic-beta Variational FFN: VFE E-step with beta recomputed each iteration.

    At each integration step, beliefs (mu, Sigma) and attention weights (beta)
    co-evolve:

        1. Compute beta from current beliefs: beta_ij = softmax(-KL(q_i||Omega_ij[q_j])/kappa)
        2. Compute full VFE gradient: dF/dtheta (includes dBeta/dtheta nonlinearity)
        3. Update beliefs via natural gradient descent
        4. (Optional) Update gauge frames phi via dF/dphi
        5. Repeat for n_iterations steps

    Two nonlinearities replace GELU/ReLU:
        1. Boltzmann GLU (direct term, lambda_belief): beta_ij gates ∇KL — analog of SiLU/GELU
        2. Attention-variance coupling (lambda_softmax): ∂β/∂μ · KL — second-order correction
           dBeta_ij/dmu_i = -beta_ij * [dKL_ij/dmu_i - Sum_k beta_ik * dKL_ik/dmu_i] / kappa

    Supports SO(3), SO(N), and GL(K) gauge groups via generators (n_gen, K, K)
    and gauge frames phi (B, N, phi_dim). Additional capabilities:
        - multihead_vfe: per-head beta_h through VFE iterations (requires irrep_dims)
        - prior_bank / use_prior_bank: token-dependent priors via PriorBank and token_ids
        - learnable_alpha: Bayesian precision via Gamma-Normal conjugacy
        - isotropic_covariance: force Sigma = sigma^2 I
        - DEQ mode (use_deq): implicit differentiation for E-step fixed point
        - amortized_inference: gradient flow through priors for learned E-step init
        - exact_diagonal_transport: lift diagonal sigma to full for exact transport,
          then extract diagonal from result; disables fused diagonal paths

    Convergence note: The softmax coupling beta(mu, Sigma) makes F non-convex
    in (mu, Sigma) jointly, so monotone VFE decrease is not theoretically
    guaranteed.  However, E-step convergence is empirically verified via
    scripts/analyze_e_step_convergence.py, and the trust regions (whitened mu
    trust, SPD retraction trust, natural gradient caps) prevent practical
    divergence.

    Complexity: O(n_steps * N^2 * K)
    """

    def __init__(
        self,
        embed_dim: int,
        generators: torch.Tensor,  # (n_gen, K, K) Lie algebra generators
        alpha: float = 0.01,       # Self-coupling weight (KL(q||p)) for E-step VFE descent.
                                   # NOTE: Decoupled from the M-step loss alpha (config['alpha']).
                                   # With amortized inference, dCE/dθ flows through q*(θ) via the
                                   # VFE computation graph, so dq*/dθ already encodes the
                                   # self-coupling effect. Adding explicit KL(q||p) in the M-step
                                   # loss double-counts the coupling. Correct separation: E-step
                                   # handles belief regularization (α_E > 0), M-step handles
                                   # prediction quality (CE only, α_M = 0).
        lambda_belief: float =  1.0,  # Boltzmann GLU weight (direct: β·∇KL — GELU/SiLU analog)
        lambda_softmax: float = 1.0,  # Attention-variance coupling weight (∂β/∂θ · KL)
        include_attention_entropy: bool = True,  # Add κ·Σβ·log(β/π) to alignment_loss (manuscript eq:free_energy_functional_final).
        kappa: float =          1.0,        # Attention temperature
        n_iterations: int =     1,    # VFE descent steps (more steps = deeper equilibration)
        
        mu_lr: float =          0.1,           # E-step μ step size (used when learnable_lr=False)
        sigma_lr: float =       0.001,      # E-step σ step size (DECOUPLED from μ LR — drives the retraction directly)
        sigma_trust: float =    5.0,        # E-step σ trust-region clamp on the whitened tangent δσ/σ (clamp magnitude, not step)
        phi_lr: float =         0.05,      # Learning rate for phi updates
        
        learnable_lr: bool =              True, # Learn step size?
        update_sigma: bool =              True, # Update covariances?
        diagonal_covariance: bool =       False,  # Use diagonal Σ for efficiency
        exact_diagonal_transport: bool =  False,  # Lift diagonal σ for exact Ω@Σ@Ω^T transport
                                                  # (~K× memory: pairwise cov goes from (B,N²,K) to (B,N²,K,K))
        compute_sigma_align_grad: bool =  True,  # Compute sigma gradient from alignment term
        
        # Phi (gauge frame) evolution via VFE gradients
        update_phi: bool = True,  # If True, update phi via ∂F/∂φ (after E-step loop)
        update_phi_per_iteration: bool =  True,  # If True, update phi during EACH E-step iteration
        
        phi_max_norm: Optional[float] =   None,  # Max phi norm; None = auto (π for SO(N), 5.0 for GL(K))
        omega_trust_region: float =       0.3,   # Trust-region clamp for direct Omega retraction (gauge_param='omega')
        phi_project_slk: bool =           False, # GL(K) only: project φ → sl(K) after retraction (det Ω = 1)
        phi_trace_clamp: Optional[float] = None, # GL(K) only: soft cap |tr(φ·G)| ≤ T per token
        prior_bank: Optional[nn.Module] = None,  # Token-dependent PriorBank (if provided)
        use_prior_bank: bool =            False,  # If True, use PriorBank (token-dependent) instead of position-dependent priors
        
        # Memory-efficient options (NEW!)
        irrep_dims: Optional[List[int]] = None,  # Block dimensions for principled KL decomposition
        
        # Self-attention masking (prevents attention collapse)
        mask_self_attention: bool =       False,  # If True, mask out diagonal (no self-attention)
        
        # Bayesian precision (learned prior self-coupling)
        learnable_alpha: bool =           False,  # If True, use Gamma-Normal conjugate precision
        # Phi gradient preconditioning mode
        phi_natural_gradient: str =          'killing',  # 'clip'|'cartan'|'killing'|'pullback'
        killing_center_reg: Optional[float] = None,  # Killing form center regularization (None=2K)
        
        # DEQ implicit differentiation
        use_deq: bool =          False,                # Use DEQ backward for E-step fixed point
        deq_neumann_terms: int = 5,           # Neumann series terms for DEQ backward
        
        # Gauge mode
        gauge_mode: str = 'learned',          # 'learned', 'trivial' (Ω = I), or 'constant'
        # Constant gauge: per-head learnable Ω from the attention module.
        # When gauge_mode='constant', these are used for transport in VFE iterations
        # instead of identity (which would be inconsistent with the attention module).
        constant_omega: Optional[nn.ParameterList] = None,
        
        # Isotropic covariance limit
        isotropic_covariance: bool = False,   # If True, force Σ = σ²I after each E-step update
        # EM gradient-flow mode — replaces individual amortized_inference / amortize_sigma /
        # exact_phi_grad / implicit_em / em_phi_mode flags (all derived from em_mode).
        em_mode: str = 'ift_phi',
        
        # Rotary Position Embeddings (RoPE) — must match attention sublayer setting
        use_rope: bool =             True,
        rope_base: float =           10000.0,
        
        gauge_param: str =           'phi',# 'phi' (Lie algebra) or 'omega' (direct GL(K))
        sigma_max: float =           5.0,  # Upper bound on σ (prevents nat_grad blowup from 2σ²·∇σ)
        
        e_step_sigma_floor: float =  0.1,  # Floor on σ_p inside E-step (caps 1/σ_p at 1/floor)
        spd_floor_mode: str =        'eigclamp',  # 'eigclamp' | 'ridge' | 'none' — see BlockConfig.
        propagate_kl_nonfinite: bool = False,  # If True, safe_kl_clamp lets NaN/±inf propagate
                                               # (diagnostic runs; default preserves masking).
        detach_phi: bool =           False,# Detach phi from backprop in non-amortized mode
                                           # (enables fully backprop-free training with phi P-flow)
        deq_include_phi: bool =      False,# Include phi in DEQ fixed-point variables.
                                           # When True, the Neumann-series IFT correction applies
                                           # to the joint (mu, sigma, phi) fixed point, giving the
                                           # exact M-step phi gradient instead of straight-through.
                                           # Requires use_deq=True and evolve_phi=True.
        closed_form_e_step: bool =   False,# Use closed-form precision-weighted fixed point
                                           # instead of gradient descent. Diagonal path uses
                                           # the enhanced form that absorbs softmax coupling
                                           # (S, c terms); full-cov uses linear-only CF.
        learnable_head_kappa: bool =    False,# If True, learn per-head κ_h
        n_picard_steps: int =           0,  # Re-solve iterations (diagonal) or Picard steps (full-cov)
        picard_trust_region: float =    5.0,# Whitened trust region for Picard steps
        e_step_early_exit_tol: float = None, # Relative change threshold for early E-step exit
        compile_vfe: bool = False,         # torch.compile the VFE iteration (Finding 25)
        gradient_checkpoint_vfe: bool = False,  # Activation checkpointing for VFE loop (Finding 26)
        alpha_divergence: float = 1.0,   # Renyi alpha-divergence parameter (1.0 = KL)
        enforce_orthogonal: bool = False,  # If True, project Omega to SO(K) via Newton-Schulz
    ):
        """
        Initialize dynamic-beta VFE FFN.

        Args:
            embed_dim: K, dimension of belief vectors.
            generators: Lie algebra generators (n_gen, K, K). n_gen = 3 for SO(3),
                N(N-1)/2 for SO(N), K^2 for GL(K).
            alpha: Weight for KL(q||p) self-coupling (prior anchoring).
            lambda_belief: Weight for belief alignment term.
            kappa: Temperature for attention softmax (higher = softer).
            n_iterations: Number of VFE descent iterations per forward pass.
            learnable_lr: If True, step size eta is a learnable parameter.
                If False, uses fixed mu_lr and sigma_lr instead.
            mu_lr: E-step μ natural gradient step size (used when learnable_lr=False).
            sigma_lr: E-step σ trust region scale (used when learnable_lr=False).
            update_sigma: If True, also update covariance matrices Sigma.
            diagonal_covariance: Use diagonal Sigma for O(K) instead of O(K^2).
            compute_sigma_align_grad: If True, compute sigma gradient from alignment.
            update_phi: If True, update phi via dF/dphi after E-step loop.
            update_phi_per_iteration: If True, evolve phi during each E-step iteration.
            phi_lr: Learning rate for phi updates.
            phi_max_norm: Max norm for phi; None = auto-select in retraction
                (π for SO(N), 5.0 for GL(K)).
            prior_bank: Optional PriorBank module providing token-dependent priors via
                token_ids. When set with use_prior_bank=True, replaces position-dependent
                priors with token-dependent priors from the PriorBank.
            use_prior_bank: If True, use PriorBank for token-dependent priors.
                Requires prior_bank and token_ids in forward().
            irrep_dims: Block dimensions [d_1, d_2, ...] for memory-efficient
                block-diagonal KL decomposition.
            mask_self_attention: If True, mask diagonal to prevent attention collapse.
            learnable_alpha: If True, Bayesian precision via Gamma-Normal conjugacy:
                alpha_k = c0_k / (b0_k + kl_k). Per-dimension, gauge-invariant.
            multihead_vfe: If True, per-head beta_h through VFE iterations.
                Requires irrep_dims.
            phi_natural_gradient: Phi gradient preconditioning mode
                ('clip'|'cartan'|'killing'|'pullback').
            use_deq: If True, use DEQ implicit differentiation for E-step fixed point.
            deq_neumann_terms: Neumann series terms for DEQ backward pass.
            gauge_mode: 'learned', 'trivial' (Omega=I), or 'constant'.
            constant_omega: Per-head learnable Omega from the attention module for
                gauge_mode='constant'.
            isotropic_covariance: If True, force Sigma = sigma^2 I after each update.
            amortized_inference: If True, gradients flow through priors for learned
                E-step initialization.
            exact_diagonal_transport: When True with diagonal_covariance, lifts sigma
                to full covariance via diag_embed for exact gauge transport, then
                extracts the diagonal from the result. Disables fused diagonal paths.
            use_rope: Apply rotary position embeddings (must match attention sublayer).
            rope_base: RoPE base frequency.
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.register_buffer('generators', generators)
        # Cache skew-symmetry flag: for SO(K), exp(-A) = exp(A)^T (saves one matrix_exp)
        self._generators_are_skew = bool(torch.allclose(
            generators + generators.transpose(-1, -2),
            torch.zeros_like(generators), atol=1e-5
        ))
        self.n_iterations = n_iterations
        self.gauge_param = gauge_param
        self.mask_self_attention = mask_self_attention
        self.update_sigma = update_sigma
        self.diagonal_covariance = diagonal_covariance
        self.exact_diagonal_transport = exact_diagonal_transport
        self.compute_sigma_align_grad = compute_sigma_align_grad
        self.gauge_mode = gauge_mode
        self.isotropic_covariance = isotropic_covariance
        from transformer.core.block_config import _EM_MODE_TABLE
        self.em_mode = em_mode
        _flags = _EM_MODE_TABLE[em_mode]
        self.amortized_inference = _flags['amortized_inference']
        self.amortize_sigma = _flags['amortize_sigma']
        self.exact_phi_grad = _flags['exact_phi_grad']
        self.implicit_em = _flags['implicit_em']
        self.em_phi_mode = _flags['em_phi_mode']
        self.sigma_max = sigma_max
        self.e_step_sigma_floor = e_step_sigma_floor
        if spd_floor_mode not in ('eigclamp', 'ridge', 'none'):
            raise ValueError(
                f"spd_floor_mode must be 'eigclamp', 'ridge', or 'none'; got {spd_floor_mode!r}"
            )
        self.spd_floor_mode = spd_floor_mode
        self.propagate_kl_nonfinite = propagate_kl_nonfinite
        self.enforce_orthogonal = enforce_orthogonal
        self.detach_phi = detach_phi
        self.closed_form_e_step = closed_form_e_step
        self.e_step_early_exit_tol = e_step_early_exit_tol
        self.gradient_checkpoint_vfe = gradient_checkpoint_vfe
        self.n_picard_steps = n_picard_steps
        self.picard_trust_region = picard_trust_region
        self._last_implicit_mu_scale = None   # (B, N, K) stored after E-step for model.py
        self._last_implicit_sigma_scale = None
        if self.implicit_em:
            logger.info(f"[VariationalFFNDynamic] Implicit EM enabled: IFT-based M-step gradient")
            logger.info(f"  → Detaches mu/sigma at E-step start, applies s_k = (α/σ²_p)/A_k scaling")
        # RoPE in the VFE E-step uses a hybrid "attention gauge ≠ value gauge"
        # objective that mirrors the attention sublayer's factorisation:
        #
        #     F_align = α·KL(q‖p) + λ·Σ_ij β_ij^{RoPE} · KL_raw(q_i ‖ Ω_ij q_j)
        #
        # where β is softmaxed from the RoPE-rotated KL (position-aware) but the
        # KL being re-weighted by β is the raw-μ KL (content-only).  This is the
        # same factorisation the attention sublayer uses: β depends on RoPE but
        # the aggregated message Σ_j β_ij · Ω_ij · μ_j uses raw μ_j (see
        # attention.py:1789-1792).  The elaborate chain-rule infrastructure in
        # vfe_gradients.py (kl_values_raw accumulator, grad_kl_rope_per_pair,
        # _un_apply_rope_pair_outer at line 968) exists specifically to make
        # this hybrid mathematically consistent: the direct term
        # β·∂KL_raw/∂μ uses raw-μ gradient (line 894-897), the softmax
        # coupling term ∂β/∂μ_raw·KL_raw applies R(θ_i)^T to un-rotate the
        # rope-space gradient (line 965-970), and the multiplier in
        # ∂(β·KL_raw)/∂μ is KL_raw (line 940).
        #
        # The `use_rope` kwarg is now honoured: setting use_rope=False disables
        # RoPE in BOTH the attention sublayer and the VFE E-step consistently,
        # so the user has a single switch governing position awareness.  When
        # use_rope=True (default), both sublayers use the hybrid objective and
        # position information enters β uniformly across the block.
        #
        # Historical note: an earlier comment claimed RoPE should be disabled
        # in the VFE path due to "double-counting position", but that concern
        # does not apply to the current implementation — the raw-vs-RoPE KL
        # separation in vfe_gradients.py explicitly prevents double counting by
        # keeping the alignment objective (KL_raw) distinct from the attention
        # routing function (KL_RoPE via β).
        self._use_rope_vfe = use_rope
        self._rope_base_vfe = rope_base
        self.alpha_divergence = alpha_divergence
        # EXPERIMENTAL: when != 'off', the rope KL also rotates Σ (R Σ R^T
        # sandwich), implementing the framework-consistent gauge-transport
        # interpretation of RoPE.  Default 'off' uses the standard-transformer
        # pattern (rotate only μ).  Plumbed through from
        # BlockConfig.rope_full_gauge.  Tri-state mode in
        # {'off', 'vfe_only', 'both'}.  The FFN VFE-gradient helper fires for
        # {'vfe_only', 'both'}; the attention-side σ rotation is gated by the
        # parallel attention.rope_full_gauge_mode attribute and is independent
        # of this flag (set externally by VariationalFFNDynamic.__init__).
        self._rope_full_gauge_vfe = 'off'
        # Constant gauge: store reference to attention module's per-head Ω parameters.
        # When gauge_mode='constant', these are used to build transport operators
        # in VFE iterations, ensuring consistency with the attention module.
        # Without this, the FFN would use Ω=I (identity), computing different
        # attention patterns than the attention module.
        # Use __dict__ to bypass nn.Module.__setattr__ and prevent double-
        # registration of the attention module's constant_omega ParameterList.
        # Without this, state_dict() produces duplicate keys under both
        # attention.constant_omega.* and ffn.constant_omega.*.
        self.__dict__['constant_omega'] = constant_omega

        # Phi evolution via VFE gradients (principled approach)
        self.update_phi = update_phi
        self.update_phi_per_iteration = update_phi_per_iteration  # Dynamical gauge frames
        if update_phi_per_iteration and gauge_mode == 'constant':
            import warnings
            warnings.warn(
                "evolve_phi_e_step=True with gauge_mode='constant' is ineffective: "
                "phi is not used for transport in constant gauge mode (Ω is a direct parameter). "
                "Set evolve_phi_e_step=False to avoid wasted computation.",
                UserWarning,
            )
        if update_phi_per_iteration:
            logger.info(f"[VariationalFFNDynamic] φ will evolve DURING E-step iterations (dynamical gauge frames)")
        self.phi_lr = phi_lr
        self.phi_max_norm = phi_max_norm
        self.omega_trust_region = omega_trust_region
        self.phi_project_slk = phi_project_slk
        self.phi_trace_clamp = phi_trace_clamp

        # Phi gradient preconditioning mode
        self.phi_natural_gradient = phi_natural_gradient
        # Register as buffers so they move to the correct device with the model
        self.register_buffer('_phi_preconditioner', None)
        self.register_buffer('_structure_constants', None)
        self.register_buffer('_gram', None)
        if phi_natural_gradient not in ('clip', 'cartan', 'killing', 'pullback'):
            raise ValueError(f"phi_natural_gradient must be 'clip'|'cartan'|'killing'|'pullback', got '{phi_natural_gradient}'")
        if phi_natural_gradient in ('cartan', 'killing', 'pullback'):
            from transformer.core.gauge_preconditioner import (
                build_cartan_projector, build_killing_form_preconditioner,
                build_structure_constants,
            )
            if phi_natural_gradient == 'cartan':
                self._phi_preconditioner = build_cartan_projector(generators)
                logger.info(f"[VariationalFFNDynamic] φ preconditioning: Cartan (sym_dampening=0.1)")
            elif phi_natural_gradient == 'killing':
                self._phi_preconditioner = build_killing_form_preconditioner(
                    generators, center_reg=killing_center_reg,
                )
                _cr = killing_center_reg if killing_center_reg is not None else 2.0 * generators.shape[-1]
                logger.info(f"[VariationalFFNDynamic] φ preconditioning: Killing form natural gradient (center_reg={_cr:.1f})")
            elif phi_natural_gradient == 'pullback':
                self._structure_constants = build_structure_constants(generators)
                # Frobenius inner product: <T_a, T_b> = tr(T_a^T T_b)
                self._gram = torch.einsum('aij,bij->ab', generators, generators)
                logger.info(f"[VariationalFFNDynamic] φ preconditioning: pullback natural gradient (exact)")

        # Memory-efficient options
        self.irrep_dims = irrep_dims

        # VFE hyperparameters
        self.alpha = alpha
        self.lambda_belief = lambda_belief
        self.lambda_softmax = lambda_softmax
        self.include_attention_entropy = include_attention_entropy
        self.kappa = kappa
        self.learnable_head_kappa = learnable_head_kappa

        # =================================================================
        # Multi-head VFE: per-block β through iterations
        # =================================================================
        self.multihead_vfe = irrep_dims is not None
        if self.multihead_vfe:
            n_heads = len(irrep_dims)
            if learnable_head_kappa:
                logger.info(f"[VariationalFFNDynamic] Multi-head VFE: {n_heads} heads with learnable per-head κ (init from κ={kappa})")
            else:
                logger.info(f"[VariationalFFNDynamic] Multi-head VFE: {n_heads} heads with shared κ={kappa}")

        # =================================================================
        # Per-head learnable temperature κ_h
        # =================================================================
        # κ_h is supposed to be a single physical quantity — the temperature
        # in β_ij = softmax(−KL/(κ·√d_h)).  The manuscript interprets β as
        # the posterior p(j|i), which is a single distribution, not two.
        # Historically this module owned its own nn.Parameter and the
        # attention sublayer owned a separate nn.Parameter, so gradient
        # descent was free to drive them apart — the attention β and the
        # E-step internal β would then describe different posteriors.
        #
        # Fix: the VFE FFN now borrows the attention sublayer's parameter
        # via blocks.py:GaugeTransformerBlock.__init__ using __dict__
        # assignment (same pattern as _prior_bank_ref in
        # active_inference.py).  This bypasses nn.Module parameter
        # registration in the FFN so the parameter is not double-counted
        # in the optimizer or state_dict, and both sublayers read from a
        # single source of truth.
        #
        # At construction time this FFN does not know about the attention
        # sublayer yet, so we initialise a safety-net local parameter here
        # and the block overrides it after both sub-modules are built.
        # If the block never wires the shared parameter in (e.g., a
        # standalone unit test that instantiates the FFN directly), the
        # safety-net parameter is what _get_kappa_h reads, so the FFN
        # remains functional on its own.  When the block wiring runs,
        # it deletes the safety-net from self._parameters / self._buffers
        # and installs a direct reference to the attention sublayer's
        # tensor via __dict__ assignment.
        if learnable_head_kappa and irrep_dims is not None:
            init_kappas = torch.tensor([
                kappa for _d_h in irrep_dims
            ])
            self.log_kappa_per_head = nn.Parameter(torch.log(init_kappas))
            self.register_buffer('_kappa_init', init_kappas)
        else:
            self.log_kappa_per_head = None
            self._kappa_init = None

        # =================================================================
        # Bayesian Precision: Log-barrier form (Eq. 882-884)
        # =================================================================
        # α* = c₀ / (b₀ + KL(q‖p))
        #
        # Log-barrier regulariser: α shrinks as the full KL divergence
        # between the variational posterior q and prior p grows.
        # Gauge-invariant (KL is a gauge scalar).
        # Initialized so α ≈ alpha (the scalar value) when KL ≈ 0.
        self.learnable_alpha = learnable_alpha
        if learnable_alpha:
            # Initialize: c₀ = alpha * b₀, b₀ = 1
            # so that α = c₀ / (b₀ + 0) = alpha when KL = 0
            # Per-dimension: each belief dimension k gets its own (c₀_k, b₀_k)
            # so different irrep blocks can learn different precision curves.
            alpha_init = max(alpha, 0.01)  # avoid division by zero
            b0_init = 1.0
            c0_init = alpha_init * b0_init
            # Parameterize via softplus to ensure positivity — shape (K,)
            self.raw_c0 = nn.Parameter(torch.full((embed_dim,), self._softplus_inverse(c0_init)))
            self.raw_b0 = nn.Parameter(torch.full((embed_dim,), self._softplus_inverse(b0_init)))
            logger.info(f"[VariationalFFNDynamic] Bayesian precision enabled (per-dim): "
                        f"c₀={c0_init:.4f}, b₀={b0_init:.1f}, "
                        f"initial α≈{alpha_init} (K={embed_dim})")

        # PriorBank integration
        self.use_prior_bank = use_prior_bank
        self.prior_bank = prior_bank

        if use_prior_bank and prior_bank is not None:
            self.prior_bank = prior_bank
            logger.info(f"[VariationalFFNDynamic] Using PriorBank with token-dependent priors (vocab_size={prior_bank.vocab_size})")
        elif use_prior_bank and prior_bank is None:
            raise ValueError(
                "use_prior_bank=True requires prior_bank to be provided! "
                "Create a PriorBank and pass it to VariationalFFNDynamic."
            )

        # Per-iteration diagnostics (set externally by trainer)
        self._collect_iteration_diagnostics = False
        self._iteration_diagnostics: list = []

        # Fiber trajectory recording (set externally via enable_fiber_recording)
        self._record_fiber_trajectory: bool = False
        self._fiber_snapshots: list = []
        self._fiber_token_indices: np.ndarray = np.array([], dtype=np.int64)

        # Lightweight E-step gradient norms (always stored, last iteration only)
        # 'nat_grad_mu'/'nat_grad_sigma'/'grad_phi' are RAW (pre-clip) norms.
        # 'nat_grad_mu_clipped'/'nat_grad_sigma_clipped' are post-clip norms.
        self._e_step_grad_norms: Dict[str, float] = {
            'nat_grad_mu': 0.0, 'nat_grad_sigma': 0.0, 'grad_phi': 0.0,
            'nat_grad_mu_clipped': 0.0, 'nat_grad_sigma_clipped': 0.0,
        }

        # Debug: set to True to print per-component gradient breakdown each E-step iteration.
        # Shows self-coupling, alignment, softmax, obs, Euclidean total, nat_grad amplification.
        self._debug_vfe_gradients: bool = False

        # Lightweight VFE gradient decomposition (always stored when _collect_vfe_metrics=True).
        # Populated from _VFE_GRAD_DEBUG at end of E-step; readable after forward().
        self._collect_vfe_metrics: bool = False
        self.last_vfe_debug: Optional[Dict[str, float]] = None

        # DEQ implicit differentiation
        self.use_deq = use_deq
        self.deq_neumann_terms = deq_neumann_terms
        self.deq_include_phi = deq_include_phi
        # Learnable step sizes (stored in unconstrained space; softplus → positive LR).
        # μ and σ are INDEPENDENT: separate raw parameters, separate softplus+clamp
        # properties, separate gradients when learnable_lr=True. This is the
        # decoupling that makes E_sigma_q_lr a real σ knob rather than coupled to
        # the μ step size.
        if learnable_lr:
            self.raw_lr = nn.Parameter(torch.tensor(self._softplus_inverse(mu_lr)))
            self.raw_sigma_lr = nn.Parameter(torch.tensor(self._softplus_inverse(sigma_lr)))
        else:
            self.register_buffer('raw_lr', torch.tensor(self._softplus_inverse(mu_lr)))
            self.register_buffer('raw_sigma_lr', torch.tensor(self._softplus_inverse(sigma_lr)))
        # σ trust-region clamp on the whitened tangent (semantically separate
        # from the step size). Caps |δσ/σ| so the multiplicative update
        # exp(step·whitened) stays bounded. Default 5.0 matches the historical
        # vfe_utils.retract_spd_diagonal_torch default; full-cov path applies
        # an internal ×0.5 factor.
        self.e_step_sigma_trust = sigma_trust

        # torch.compile the VFE iteration inner loop (Finding 25).
        # Fuses small element-wise ops and reduces kernel launch overhead.
        # Disabled by default because torch.compile adds compilation latency
        # on the first forward pass and may interact with dynamic shapes.
        # Uses mode='default' (not 'reduce-overhead') because _vfe_iteration has
        # extensive Python control flow (multihead vs single-β, fused vs fallback,
        # diagonal vs full cov) that is incompatible with CUDA graph capture.
        self._compile_vfe = compile_vfe
        if compile_vfe:
            self._vfe_iteration = torch.compile(
                self._vfe_iteration,
                mode='default',
                fullgraph=False,
            )
            logger.info("[VariationalFFNDynamic] torch.compile applied to _vfe_iteration")

    @property
    def lr(self) -> torch.Tensor:
        """E-step μ learning rate, constrained to (0, 0.5] via softplus + clamp."""
        return F.softplus(self.raw_lr).clamp(max=0.5)

    @property
    def sigma_lr(self) -> torch.Tensor:
        """E-step σ learning rate, constrained to (0, 0.5] via softplus + clamp.

        Independent of ``self.lr`` (μ LR): backed by ``raw_sigma_lr``, which is
        an ``nn.Parameter`` when ``learnable_lr=True`` and a buffer otherwise.
        Drives the σ retraction step size directly — see ``_retract_sigma``.
        """
        return F.softplus(self.raw_sigma_lr).clamp(max=0.5)

    def _get_kappa_h(self, head_idx: int, d_h: int):
        r"""Get per-head temperature κ_h.

        When learnable_head_kappa=True: κ_h = exp(log_kappa_per_head[h])
        When False: κ_h = self.kappa (bare; callers apply √d_h scaling)

        When the enclosing block has wired a shared reference to the
        attention sublayer (the ``_kappa_attn_ref`` attribute, set via
        __dict__ in ``GaugeTransformerBlock.__init__``), the lookup goes
        through ``ref.log_kappa_per_head`` and ``ref._kappa_init`` rather
        than reading self's local attributes.  This is what makes the
        kappa parameter sharing survive ``model.to(device)``: PyTorch
        replaces Parameters/buffers with fresh tensors on the target
        device but never replaces Module objects, so the reference
        always resolves to the current device-resident tensor.
        """
        if not self.learnable_head_kappa:
            return self.kappa

        # Prefer the shared reference from the attention sublayer if the
        # block wired one up.  Access the tensors through the Module so
        # the lookup resolves at call time (not wiring time) and sees the
        # current post-.to() state.
        ref = self.__dict__.get('_kappa_attn_ref', None)
        if ref is not None:
            log_kappa = ref.log_kappa_per_head
            kappa_init = ref._kappa_init
        else:
            log_kappa = self.log_kappa_per_head
            kappa_init = self._kappa_init

        if log_kappa is None:
            return self.kappa

        kappa_h = torch.exp(log_kappa[head_idx])
        # Clamp to [0.5, 1.5] × init, matching attention module
        k0 = kappa_init[head_idx]
        return kappa_h.clamp(min=0.5 * k0, max=1.5 * k0)

    def enable_fiber_recording(
        self, token_indices: Optional[np.ndarray] = None,
        n_tokens: int = 8, seq_len: int = 128,
    ) -> None:
        r"""Enable per-iteration fiber trajectory recording.

        Records $(\mu, \Sigma)$ snapshots at each VFE E-step iteration
        for a subset of token positions.  Memory cost is bounded:
        ``n_iterations * n_tokens * K * sizeof(float32)`` per forward pass.

        Args:
            token_indices: Explicit token indices to record. If None,
                selects ``n_tokens`` uniformly spaced positions.
            n_tokens: Number of tokens to record (ignored if token_indices given).
            seq_len: Sequence length for uniform spacing (ignored if token_indices given).
        """
        self._record_fiber_trajectory = True
        if token_indices is not None:
            self._fiber_token_indices = np.asarray(token_indices, dtype=np.int64)
        else:
            self._fiber_token_indices = np.linspace(
                0, seq_len - 1, n_tokens, dtype=np.int64,
            )
        self._fiber_snapshots = []

    def disable_fiber_recording(self) -> None:
        """Disable fiber trajectory recording and clear snapshots."""
        self._record_fiber_trajectory = False
        self._fiber_snapshots = []

    def get_fiber_snapshots(self) -> list:
        """Return collected fiber trajectory snapshots and clear buffer."""
        snapshots = list(self._fiber_snapshots)
        self._fiber_snapshots = []
        return snapshots

    def _get_sigma_trust(self, effective_lr: torch.Tensor) -> float:
        r"""E-step σ trust region clamp (caps the whitened tangent δσ/σ).

        Separate from the σ step size (sigma_lr). The trust region only
        clamps the per-iteration tangent magnitude; the step size is the
        actual multiplier applied to the clamped tangent before exp.

        Decoupled from effective_lr (the μ step size). The fallback path
        (`effective_lr * 0.01`) is retained only for very old checkpoints
        whose VFE module was instantiated before e_step_sigma_trust existed.
        """
        if hasattr(self, 'e_step_sigma_trust'):
            return self.e_step_sigma_trust
        # Legacy fallback (pre-2026-05-13 checkpoints)
        return effective_lr * 0.01

    @staticmethod
    def _softplus_inverse(x: float) -> float:
        """Compute inverse of softplus: log(exp(x) - 1)."""
        if x > 20.0:
            return x  # softplus ≈ identity for large x
        return float(np.log(np.expm1(x)))

    def get_bayesian_alpha(
        self,
        mu_q: torch.Tensor,      # (B, N, K)
        mu_p: torch.Tensor,      # (B, N, K)
        sigma_p: torch.Tensor,   # (B, N, K) diagonal or (B, N, K, K) full
        sigma_q: torch.Tensor,   # (B, N, K) diagonal or (B, N, K, K) full
        eps: float = 1e-6,
    ) -> torch.Tensor:
        r"""
        Compute per-dimension Bayesian precision via log-barrier form.

        :math:`\alpha_k = c_{0,k} / (b_{0,k} + D_k(q \| p))`

        where :math:`D_k` is the per-dimension divergence contribution.  The
        divergence family is selected by ``self.alpha_divergence``:

        - ``alpha_divergence == 1.0`` (default): KL divergence, per-dim
          :math:`\mathrm{KL}_k = \tfrac{1}{2}(s_k/t_k + \delta\mu_k^2/t_k
          - 1 + \log(t_k/s_k))` where :math:`s_k = \sigma_q^k`,
          :math:`t_k = \sigma_p^k`.
        - ``alpha_divergence != 1.0``: Rényi :math:`\alpha`-divergence, per-dim
          :math:`D_{\alpha,k} = \tfrac{1}{2}\bigl(\alpha\,\delta\mu_k^2/
          \tilde\sigma_k + ((1-\alpha)\log s_k + \alpha\log t_k
          - \log\tilde\sigma_k)/(\alpha-1)\bigr)` with
          :math:`\tilde\sigma_k = (1-\alpha)s_k + \alpha t_k`.

        This matches the Rényi kernel in ``kl_computation._kl_kernel_diagonal``
        at the per-dimension level — the summed form recovers the same
        divergence used in the self-coupling gradient in
        ``vfe_gradients.py``.  Using a single divergence family for both the
        schedule and the gradient keeps the product-rule correction valid:
        :math:`\partial\alpha_k/\partial\theta
        = -\alpha_k^2/c_{0,k}\cdot\partial D_k/\partial\theta`.

        Diagonal covariance: :math:`D_k` decomposes exactly per dimension.
        Full covariance: uses diagonal elements as per-dim proxy (matches the
        existing KL-branch full-cov heuristic).

        Returns:
            alpha: ``(B, N, K)`` per-dimension-per-agent precision.
        """
        c0 = F.softplus(self.raw_c0)  # (K,)
        b0 = F.softplus(self.raw_b0)  # (K,)

        delta_mu = mu_q - mu_p  # (B, N, K)
        K = mu_q.shape[-1]
        alpha_div = self.alpha_divergence

        # Extract per-dim variances, robust to mixed shapes.  In full-cov
        # mode the prior sigma_p may still be diagonal (B, N, K) from
        # PriorBank while the posterior sigma_q evolves to full (B, N, K, K)
        # during the E-step.  We compute per-dim proxies independently for
        # each tensor: diagonal → clamp, full → extract matrix diagonal.
        # The full-cov KL formula (with Σ_p⁻¹ Σ_q matmul) is only used when
        # BOTH tensors are full, so its off-diagonal correlation signal is
        # preserved where available.
        is_p_diag = (sigma_p.dim() == 3)
        is_q_diag = (sigma_q.dim() == 3)

        sigma_p_diag = (
            sigma_p if is_p_diag else sigma_p.diagonal(dim1=-2, dim2=-1)
        ).clamp(min=eps)
        sigma_q_diag = (
            sigma_q if is_q_diag else sigma_q.diagonal(dim1=-2, dim2=-1)
        ).clamp(min=eps)

        if abs(alpha_div - 1.0) < 1e-6:
            # === KL branch (alpha_div == 1) ===
            if is_p_diag and is_q_diag:
                # Pure diagonal path: closed-form per-dim KL.
                trace_term = sigma_q_diag / sigma_p_diag              # (B, N, K)
                mahal_term = delta_mu ** 2 / sigma_p_diag             # (B, N, K)
                logdet_term = torch.log(sigma_p_diag) - torch.log(sigma_q_diag)
            elif (not is_p_diag) and (not is_q_diag):
                # Pure full-cov path: diagonal proxy of (Σ_p⁻¹ Σ_q) plus
                # exact per-dim Mahalanobis.
                sigma_p_inv = _safe_spd_inv(sigma_p, eps=eps)
                prod = torch.matmul(sigma_p_inv, sigma_q)
                trace_term = prod.diagonal(dim1=-2, dim2=-1)          # (B, N, K)
                sp_inv_delta = torch.einsum('bnij,bnj->bni', sigma_p_inv, delta_mu)
                mahal_term = delta_mu * sp_inv_delta                  # (B, N, K)
                logdet_p = torch.linalg.slogdet(sigma_p.float())[1]
                logdet_q = torch.linalg.slogdet(sigma_q.float())[1]
                logdet_term = ((logdet_p - logdet_q) / K).unsqueeze(-1).expand_as(delta_mu)
            else:
                # Mixed path: diagonal proxy on both (cannot do matrix inverse
                # or slogdet without lifting one side; the diagonal proxy is
                # consistent with the Rényi branch below).
                trace_term = sigma_q_diag / sigma_p_diag
                mahal_term = delta_mu ** 2 / sigma_p_diag
                logdet_term = torch.log(sigma_p_diag) - torch.log(sigma_q_diag)

            divergence_k = 0.5 * (trace_term + mahal_term - 1 + logdet_term)
        else:
            # === Rényi α-divergence branch ===
            # Blended per-dim variance: σ̃_k = (1-α)·σ_q,k + α·σ_p,k.
            # Clamped to eps: for α ∈ (0,1) the blend is strictly positive
            # (convex combination of positives); for α outside (0,1) it can
            # still be positive but the clamp protects against edge cases
            # when σ_q >> σ_p or vice versa.
            sigma_blend = (
                (1.0 - alpha_div) * sigma_q_diag + alpha_div * sigma_p_diag
            ).clamp(min=eps)
            # Mahalanobis: α · (δμ_k)² / σ̃_k
            mahal_term = alpha_div * (delta_mu ** 2) / sigma_blend
            # Log-det per-dim: [(1-α)log σ_q,k + α log σ_p,k - log σ̃_k] / (α-1)
            logdet_term = (
                (1.0 - alpha_div) * torch.log(sigma_q_diag)
                + alpha_div * torch.log(sigma_p_diag)
                - torch.log(sigma_blend)
            ) / (alpha_div - 1.0)
            divergence_k = 0.5 * (mahal_term + logdet_term)

        divergence_k = divergence_k.clamp(min=0.0)

        alpha = c0 / (b0 + divergence_k)  # (B, N, K)

        return alpha

    def _precondition_phi_grad(
        self,
        grad_phi: torch.Tensor,   # (..., n_gen)
        phi: torch.Tensor,        # (..., n_gen)
    ) -> torch.Tensor:
        """
        Apply phi gradient preconditioning based on self.phi_natural_gradient mode.

        Modes:
            'clip': Simple norm clipping to 10.0 (no geometric awareness)
            'cartan': Cartan decomposition with fixed sym_dampening=0.1
            'killing': Killing form natural gradient (position-independent, no free params)
            'pullback': Full pullback metric through exp (position-dependent, exact)

        Args:
            grad_phi: Raw Euclidean gradient ∂F/∂φ^a
            phi: Current gauge frame coordinates (needed for 'pullback' mode)

        Returns:
            Preconditioned gradient, same shape as grad_phi
        """
        if self.phi_natural_gradient == 'cartan':
            return apply_cartan_preconditioning(grad_phi, self._phi_preconditioner)

        elif self.phi_natural_gradient == 'killing':
            return apply_killing_form_natural_gradient(grad_phi, self._phi_preconditioner)

        elif self.phi_natural_gradient == 'pullback':
            return apply_pullback_natural_gradient(
                grad_phi, phi, self.generators,
                self._structure_constants, self._gram,
            )

        else:  # 'clip' (default)
            grad_phi_norm = torch.norm(grad_phi, dim=-1, keepdim=True)
            return torch.where(
                grad_phi_norm > 10.0,
                grad_phi * 10.0 / (grad_phi_norm + 1e-6),
                grad_phi
            )

    # =================================================================
    # Direct Omega Gradient (No Lie Algebra / No matrix_exp)
    # =================================================================

    def _compute_omega_grad_direct(
        self,
        omega_current: torch.Tensor,    # (B, N, K, K) direct group elements
        mu_current: torch.Tensor,       # (B, N, K) belief means
        sigma_current: Optional[torch.Tensor],  # (B, N, K) diagonal variances
        is_diagonal: bool,
        mask: Optional[torch.Tensor],
        eps: float,
    ) -> Optional[torch.Tensor]:
        """Compute ∂F_align/∂Ω_i via autograd (vectorized) or analytic fallback.

        Mirrors the phi path: builds alignment_loss as a differentiable computation
        through compute_attention_weights with omega.requires_grad_(), then calls
        torch.autograd.grad for fully vectorized C++ backward pass.

        Falls back to analytic tiled loop when autograd is unavailable.

        Returns:
            grad_omega: (B, N, K, K) gradient ∂F_align/∂Ω_i, or None if no gradient computed.
        """
        if sigma_current is None:
            return None  # No covariance → no KL alignment term → no ∂F_align/∂Ω

        B, N, K = mu_current.shape
        device = mu_current.device
        irrep_dims = self.irrep_dims if self.irrep_dims is not None else [K]

        # ── Autograd path (vectorized, matches phi autograd) ──────────
        omega_for_grad = omega_current.detach().clone().requires_grad_(True)

        # Build per-block (omega_h, omega_h_inv) pairs with grad tracking.
        # Safe inverse: omega can drift toward low rank during training, so
        # ridge the block before inverting and fall back to pinv on failure.
        # Mirrors the fix applied to compute_transport_operators_direct and
        # omega_to_block_exp_pairs in transport_ops.py.
        _ridge = 1e-6
        _block_exp_pairs = []
        block_start = 0
        for d in irrep_dims:
            block_end = block_start + d
            om_blk = omega_for_grad[:, :, block_start:block_end, block_start:block_end]
            _eye_d = torch.eye(d, device=om_blk.device, dtype=om_blk.dtype)
            om_blk_reg = om_blk + _ridge * _eye_d
            try:
                om_inv_blk = torch.linalg.inv(om_blk_reg)
            except (torch.linalg.LinAlgError, RuntimeError):
                om_inv_blk = torch.linalg.pinv(om_blk_reg)
            _block_exp_pairs.append((om_blk, om_inv_blk))
            block_start = block_end

        # Compute alignment loss per head (differentiable through omega_for_grad).
        #
        # Split the product rule into direct + softmax-coupling terms with
        # separate weights, matching the phi gradient path in _compute_phi_grad
        # at lines 1171-1174.  The previous unified weighting
        #   alignment_loss += lambda_belief * (beta_h * kl_h).sum()
        # silently ignored self.lambda_softmax, giving the omega path a
        # different gradient from the phi path whenever lambda_belief !=
        # lambda_softmax.  By stop-gradient on one factor at a time:
        #   d/dω [(sg[β])·KL] = β·dKL/dω  (direct term, weighted by lambda_belief)
        #   d/dω [β·(sg[KL])] = KL·dβ/dω  (softmax coupling, weighted by lambda_softmax)
        # The sum recovers d/dω[β·KL] = β·dKL/dω + KL·dβ/dω when
        # lambda_belief == lambda_softmax and re-weights the two contributions
        # independently otherwise.
        alignment_loss = torch.tensor(0.0, device=device, dtype=mu_current.dtype)
        block_start = 0
        for h, d_h in enumerate(irrep_dims):
            block_end = block_start + d_h
            mu_h = mu_current[:, :, block_start:block_end].detach()
            # Per-block covariance slice. Diagonal path: (B, N, d_h) vector of
            # variances. Full-cov path: (B, N, d_h, d_h) per-block submatrix —
            # mirrors the phi-grad slicing at _compute_phi_grad lines 1166-1169.
            if is_diagonal:
                sigma_h = sigma_current[:, :, block_start:block_end].detach()
            else:
                sigma_h = sigma_current[:, :,
                                       block_start:block_end,
                                       block_start:block_end].detach()
            gen_h = self.generators[:, block_start:block_end, block_start:block_end]

            kappa_h = self._get_kappa_h(h, d_h)  # Match main multihead path scaling
            beta_kl_h = compute_attention_weights(
                mu_q=mu_h, sigma_q=sigma_h,
                phi=torch.zeros(B, N, 1, device=device),  # dummy
                generators=gen_h,
                kappa=kappa_h, epsilon=eps, mask=mask,
                return_kl=True,
                diagonal_covariance=is_diagonal,
                gauge_param='omega', omega=omega_for_grad,
                gauge_mode=self.gauge_mode,
                irrep_dims=[d_h],
                mask_self_attention=self.mask_self_attention,
                cached_block_exp_pairs=[_block_exp_pairs[h]],
                sigma_floor=(self.e_step_sigma_floor
                             if self.spd_floor_mode == 'eigclamp' and not is_diagonal
                             else None),
                spd_floor_mode=self.spd_floor_mode,
                propagate_nonfinite=self.propagate_kl_nonfinite,
            )
            if isinstance(beta_kl_h, tuple):
                beta_h, kl_h = beta_kl_h
            else:
                beta_h = beta_kl_h
                kl_h = beta_h
            if self.include_attention_entropy:
                # Manuscript F = β·KL + τ_eff·β·log(β/π), τ_eff = κ_h·√d_h. All attached →
                # envelope-correct gradient at softmax β. lambda_belief scales the whole F.
                # lambda_softmax is IGNORED in this branch (see vfe/e_step.py for rationale).
                _beta_safe = beta_h.clamp(min=1e-30)
                _sqrt_dh = math.sqrt(max(d_h, 1))
                _F_h = (beta_h * kl_h).sum() + kappa_h * _sqrt_dh * (_beta_safe * _beta_safe.log()).sum()
                alignment_loss = alignment_loss + self.lambda_belief * _F_h
            else:
                alignment_loss = alignment_loss + (
                    self.lambda_belief * (beta_h.detach() * kl_h).sum()
                    + self.lambda_softmax * (beta_h * kl_h.detach()).sum()
                )
            block_start = block_end

        if alignment_loss.grad_fn is not None:
            grad_omega = torch.autograd.grad(
                alignment_loss,
                omega_for_grad,
                create_graph=False,
                retain_graph=False,
            )[0]
            return grad_omega

        return None

    def _retract_omega(
        self,
        omega: torch.Tensor,      # (B, N, K, K)
        grad_omega: torch.Tensor,  # (B, N, K, K) Euclidean gradient
        step_size: float,
        trust_region: float = 0.3,
    ) -> torch.Tensor:
        """Retract Omega update on GL(K) via left-invariant Lie algebra step.

        Computes the natural gradient in the Lie algebra gl(K) using the
        left-invariant pullback:

            ξ = Ω⁻¹ · ∂F/∂Ω           (left-invariant pullback)
            clip ||ξ||_F ≤ trust_region  (Riemannian trust region)
            Ω_new = Ω · exp(-η·ξ)     (exact retraction)
                  ≈ Ω - η · (Ω · ξ)   (first-order Euler)
                  = Ω - η · ∂F/∂Ω     (since Ω · Ω⁻¹ = I)

        After first-order Euler this reduces to Frobenius gradient descent,
        which is what the code below implements.  The key invariant is that
        the Riemannian norm ||ξ||_F = ||Ω⁻¹ · grad_Ω||_F is left-invariant
        under Ω → A·Ω for any A ∈ GL(K), so the trust region bounds the
        intrinsic step size rather than the coordinate-dependent one.

        Historical note: an earlier version used ξ = Ωᵀ · grad_Ω, which is
        left-invariant only under orthogonal A and therefore broke the trust-
        region claim for general GL(K).  For SO(K), Ωᵀ = Ω⁻¹ and the two
        formulas agree — the old code was correct only for the orthogonal
        special case.

        When irrep_dims is set, works block-diagonally to avoid O(K³) matmuls
        on the full K×K matrix (e.g., 5×10×10 instead of 50×50).

        Args:
            omega: Current group elements (B, N, K, K)
            grad_omega: Euclidean gradient ∂F/∂Ω (B, N, K, K)
            step_size: Learning rate
            trust_region: Max Riemannian step size ||ξ||_F = ||Ω⁻¹ grad||_F

        Returns:
            omega_new: Updated group elements (B, N, K, K)
        """
        irrep_dims = self.irrep_dims if self.irrep_dims is not None else None
        _ridge = 1e-6  # ridge for the inverse; ω can drift toward low rank

        if irrep_dims is not None:
            # Block-diagonal retraction: process each head block independently
            omega_new = omega.clone()
            block_start = 0
            for d in irrep_dims:
                block_end = block_start + d
                om_blk = omega[:, :, block_start:block_end, block_start:block_end]
                gr_blk = grad_omega[:, :, block_start:block_end, block_start:block_end]

                # Left-invariant pullback via solve instead of explicit inv for
                # numerical stability.  ridge·I prevents catastrophic failure if
                # om_blk drifts toward singular during training.
                _eye = torch.eye(d, device=om_blk.device, dtype=om_blk.dtype)
                om_reg = om_blk + _ridge * _eye
                try:
                    xi_blk = torch.linalg.solve(om_reg, gr_blk)
                except (torch.linalg.LinAlgError, RuntimeError):
                    xi_blk = torch.linalg.pinv(om_reg) @ gr_blk

                # Clip in Lie algebra norm (left-invariant under GL(K))
                xi_norm = xi_blk.flatten(-2).norm(dim=-1, keepdim=True).unsqueeze(-1)
                scale = torch.clamp(trust_region / (xi_norm + 1e-8), max=1.0)
                xi_blk = xi_blk * scale

                # Euler retraction: Ω·exp(-η·ξ) ≈ Ω - η·(Ω·ξ)
                omega_new[:, :, block_start:block_end, block_start:block_end] = (
                    om_blk - step_size * (om_blk @ xi_blk)
                )
                block_start = block_end
            return omega_new

        # Fallback: full K×K retraction (no irrep structure)
        K = omega.shape[-1]
        _eye = torch.eye(K, device=omega.device, dtype=omega.dtype)
        om_reg = omega + _ridge * _eye
        try:
            xi = torch.linalg.solve(om_reg, grad_omega)
        except (torch.linalg.LinAlgError, RuntimeError):
            xi = torch.linalg.pinv(om_reg) @ grad_omega

        # Clip in Lie algebra norm
        xi_norm = xi.flatten(-2).norm(dim=-1, keepdim=True).unsqueeze(-1)
        scale = torch.clamp(trust_region / (xi_norm + 1e-8), max=1.0)
        xi = xi * scale

        # Euler retraction
        omega_new = omega - step_size * (omega @ xi)

        return omega_new

    def _compute_phi_grad(
        self,
        phi_current: torch.Tensor,
        mu_current: torch.Tensor,
        sigma_current: Optional[torch.Tensor],
        is_diagonal: bool,
        mask: Optional[torch.Tensor],
        eps: float,
        cached_block_exp_pairs: Optional[list] = None,
        beta_current: Optional[torch.Tensor] = None,
        beta_heads: Optional[list] = None,
    ) -> Optional[torch.Tensor]:
        r"""Compute ∂F/∂φ via autograd (alignment term).

        By default, beliefs (mu_current, sigma_current) are detached, giving the
        partial derivative ∂F/∂φ|_{q fixed} — a semi-gradient that treats beliefs
        as constants. When ``exact_phi_grad=True`` and ``amortized_inference=True``,
        beliefs stay attached, giving the full total derivative dF/dφ through
        the E-step iteration graph (IFT-correct).

        Returns the preconditioned gradient, or None if no gradient could be computed.
        """
        phi_for_grad = phi_current.clone().requires_grad_(True)

        if self.multihead_vfe:
            alignment_loss = torch.tensor(0.0, device=mu_current.device,
                                          dtype=mu_current.dtype)
            _phi_bep = None
            if self.irrep_dims is not None:
                _phi_bep = fused_block_matrix_exp_pairs(
                    phi_for_grad, self.generators, self.irrep_dims,
                    enforce_orthogonal=getattr(self, 'enforce_orthogonal', False),
                    skew_symmetric=self._generators_are_skew,
                )
            _detach_beliefs = not (self.amortized_inference and self.exact_phi_grad)
            block_start = 0
            for h, d_h in enumerate(self.irrep_dims):
                block_end = block_start + d_h
                mu_h = mu_current[:, :, block_start:block_end]
                if _detach_beliefs:
                    mu_h = mu_h.detach()
                if sigma_current is None:
                    sigma_h = None
                elif is_diagonal:
                    sigma_h = sigma_current[:, :, block_start:block_end]
                    if _detach_beliefs:
                        sigma_h = sigma_h.detach()
                else:
                    sigma_h = sigma_current[:, :, block_start:block_end, block_start:block_end]
                    if _detach_beliefs:
                        sigma_h = sigma_h.detach()
                gen_h = self.generators[:, block_start:block_end, block_start:block_end]
                kappa_h = self._get_kappa_h(h, d_h)  # Normalize for block dimension
                _phi_head_bep = [_phi_bep[h]] if _phi_bep is not None else None

                beta_phi_h_result = compute_attention_weights(
                    mu_q=mu_h, sigma_q=sigma_h,
                    phi=phi_for_grad, generators=gen_h,
                    kappa=kappa_h, epsilon=eps, mask=mask,
                    return_kl=True,
                    diagonal_covariance=is_diagonal,
                    irrep_dims=[d_h],
                    mask_self_attention=self.mask_self_attention,
                    gauge_mode=self.gauge_mode,
                    cached_block_exp_pairs=_phi_head_bep,
                    exact_diagonal_transport=self.exact_diagonal_transport,
                    sigma_floor=(self.e_step_sigma_floor
                                 if self.spd_floor_mode == 'eigclamp' and not is_diagonal
                                 else None),
                    spd_floor_mode=self.spd_floor_mode,
                    propagate_nonfinite=self.propagate_kl_nonfinite,
                )
                beta_phi_h, kl_h = beta_phi_h_result
                # Separate direct and softmax coupling weights for phi gradient.
                # d/dphi [sum(beta * KL)] = sum(dBeta/dphi * KL) + sum(beta * dKL/dphi)
                # Direct term (beta * dKL/dphi) gets lambda_belief.
                # Softmax coupling (dBeta/dphi * KL) gets lambda_softmax.
                # Achieved via stop-gradient: detach the factor NOT being differentiated.
                if self.include_attention_entropy:
                    # Manuscript F direct form; lambda_softmax ignored. See e_step.py.
                    _beta_safe = beta_phi_h.clamp(min=1e-30)
                    _sqrt_dh = math.sqrt(max(d_h, 1))
                    _F_h = (beta_phi_h * kl_h).sum() + kappa_h * _sqrt_dh * (_beta_safe * _beta_safe.log()).sum()
                    alignment_loss = alignment_loss + self.lambda_belief * _F_h
                else:
                    alignment_loss = alignment_loss + (
                        self.lambda_belief * (beta_phi_h.detach() * kl_h).sum()
                        + self.lambda_softmax * (beta_phi_h * kl_h.detach()).sum()
                    )
                block_start = block_end
        else:
            _detach_beliefs = not (self.amortized_inference and self.exact_phi_grad)
            beta_for_phi_result = compute_attention_weights(
                mu_q=mu_current.detach() if _detach_beliefs else mu_current,
                sigma_q=(sigma_current.detach() if _detach_beliefs else sigma_current) if sigma_current is not None else None,
                phi=phi_for_grad,
                generators=self.generators,
                kappa=self.kappa,
                epsilon=eps,
                mask=mask,
                return_kl=True,
                diagonal_covariance=is_diagonal,
                irrep_dims=self.irrep_dims,
                mask_self_attention=self.mask_self_attention,
                gauge_mode=self.gauge_mode,
                exact_diagonal_transport=self.exact_diagonal_transport,
                sigma_floor=(self.e_step_sigma_floor
                             if self.spd_floor_mode == 'eigclamp' and not is_diagonal
                             else None),
                spd_floor_mode=self.spd_floor_mode,
                propagate_nonfinite=self.propagate_kl_nonfinite,
            )
            if isinstance(beta_for_phi_result, tuple):
                beta_phi, kl_matrix = beta_for_phi_result
            else:
                beta_phi = beta_for_phi_result
                kl_matrix = beta_phi
            if self.include_attention_entropy:
                # Manuscript F = β·KL + τ_eff·β·log(β/π) direct form; lambda_softmax ignored.
                # Per-head live kappa_h carries the κ-gradient when learnable_head_kappa=True.
                # Uniform-irrep assumption: uses irrep_dims[0] for √d_h.
                _beta_safe = beta_phi.clamp(min=1e-30)
                _d_h_canon = self.irrep_dims[0] if self.irrep_dims else 1
                _sqrt_dh = math.sqrt(max(_d_h_canon, 1))
                _bk_term = (beta_phi * kl_matrix).sum()
                if self.learnable_head_kappa and _beta_safe.dim() >= 4:
                    _n_heads = _beta_safe.shape[1]
                    _entropy_per_head = (_beta_safe * _beta_safe.log()).sum(dim=(0, -2, -1))
                    _kappas = torch.stack([
                        self._get_kappa_h(h_idx, _d_h_canon)
                        for h_idx in range(_n_heads)
                    ])
                    _entropy_term = _sqrt_dh * (_kappas * _entropy_per_head).sum()
                else:
                    _entropy_term = self.kappa * _sqrt_dh * (_beta_safe * _beta_safe.log()).sum()
                alignment_loss = self.lambda_belief * (_bk_term + _entropy_term)
            else:
                alignment_loss = (
                    self.lambda_belief * (beta_phi.detach() * kl_matrix).sum()
                    + self.lambda_softmax * (beta_phi * kl_matrix.detach()).sum()
                )

        if alignment_loss.grad_fn is not None:
            grad_phi = torch.autograd.grad(
                alignment_loss,
                phi_for_grad,
                create_graph=False,
                retain_graph=False,
            )[0]
            return self._precondition_phi_grad(grad_phi, phi_current)

        return None

    def _make_deq_step_fn(self, phi_current, mu_p_current, sigma_p,
                           mask, is_diagonal, eps, dtype):
        """Create a differentiable (μ, Σ) E-step closure for DEQ backward.

        Thin wrapper around vfe_deq.make_deq_step_fn that passes ``self``
        as the first argument.  The closure body was extracted in round 8
        as part of the side-quest refactor; see vfe_deq.py for the full
        implementation.
        """
        return _make_deq_step_fn_free(
            self, phi_current, mu_p_current, sigma_p,
            mask, is_diagonal, eps, dtype,
        )

    def _make_deq_step_fn_with_phi(
        self,
        mu_p_current: torch.Tensor,
        sigma_p: torch.Tensor,
        mask: Optional[torch.Tensor],
        is_diagonal: bool,
        eps: float,
        dtype: torch.dtype,
    ):
        r"""Create a differentiable joint (μ, Σ, φ) E-step closure for DEQ.

        Thin wrapper around vfe_deq.make_deq_step_fn_with_phi that passes
        ``self`` as the first argument.  The closure body was extracted
        in round 8 as part of the side-quest refactor; see vfe_deq.py for
        the full implementation.
        """
        return _make_deq_step_fn_with_phi_free(
            self, mu_p_current, sigma_p,
            mask, is_diagonal, eps, dtype,
        )

    # =================================================================
    # Helper: Build block exp-pairs for transport operators
    # =================================================================

    def _build_block_exp_pairs(
        self,
        phi_current: torch.Tensor,        # (B, N, phi_dim)
        omega_current: Optional[torch.Tensor],  # (B, N, K, K) or None
        B: int,
        N: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[list]:
        r"""Build per-head (exp_phi_h, exp_neg_phi_h) pairs for transport operators.

        Returns a list of (Omega_h, Omega_h_inv) tuples, one per irrep block,
        or None when irrep_dims is not set.

        Branches:
            gauge_param='omega': extract per-head blocks from omega_current, invert.
            gauge_mode='trivial': return identity pairs for each block.
            gauge_mode='constant' with constant_omega: use constant per-head Omega.
            default (learned): call fused_block_matrix_exp_pairs on phi_current.
        """
        if self.irrep_dims is None:
            return None

        if omega_current is not None and self.gauge_param == 'omega':
            # Safe per-block inverse with ridge + pinv fallback (same
            # rationale as _compute_omega_grad_direct and
            # transport_ops.compute_transport_operators_direct: omega can
            # drift toward low rank during training, and a raw inverse on
            # a near-singular matrix poisons the attention graph with NaN).
            _ridge = 1e-6
            bep = []
            block_start = 0
            for d_h in self.irrep_dims:
                omega_h = omega_current[:, :, block_start:block_start + d_h,
                                        block_start:block_start + d_h]
                _eye_h = torch.eye(d_h, device=omega_h.device, dtype=omega_h.dtype)
                omega_h_reg = omega_h + _ridge * _eye_h
                try:
                    omega_h_inv = torch.linalg.inv(omega_h_reg)
                except (torch.linalg.LinAlgError, RuntimeError):
                    omega_h_inv = torch.linalg.pinv(omega_h_reg)
                bep.append((omega_h, omega_h_inv))
                block_start += d_h
            return bep

        if self.gauge_mode == 'trivial':
            bep = []
            for d_h in self.irrep_dims:
                eye_h = (torch.eye(d_h, device=device, dtype=dtype)
                         .unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1).contiguous())
                bep.append((eye_h, eye_h))
            return bep

        if self.gauge_mode == 'constant' and self.constant_omega is not None:
            bep = []
            for h, d_h in enumerate(self.irrep_dims):
                omega_h = self.constant_omega[h].to(device=device, dtype=dtype)
                if getattr(self, 'enforce_orthogonal', False) and d_h >= 2:
                    omega_h = newton_schulz_orthogonalize(omega_h.unsqueeze(0)).squeeze(0)
                eye_h = torch.eye(d_h, device=device, dtype=dtype)
                exp_phi_h = omega_h.unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1).contiguous()
                exp_neg_phi_h = eye_h.unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1).contiguous()
                bep.append((exp_phi_h, exp_neg_phi_h))
            return bep

        if self.gauge_mode == 'constant':
            # constant without constant_omega: fall back to identity (legacy)
            bep = []
            for d_h in self.irrep_dims:
                eye_h = (torch.eye(d_h, device=device, dtype=dtype)
                         .unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1).contiguous())
                bep.append((eye_h, eye_h))
            return bep

        # Default: learned phi
        return fused_block_matrix_exp_pairs(
            phi_current, self.generators, self.irrep_dims,
            enforce_orthogonal=getattr(self, 'enforce_orthogonal', False),
            skew_symmetric=self._generators_are_skew,
        )

    # =================================================================
    # Helper: Finalize E-step (store state for M-step)
    # =================================================================

    def _finalize_e_step(
        self,
        alpha_effective,
        sigma_p: torch.Tensor,
        sigma_current: Optional[torch.Tensor],
        beta_current: Optional[torch.Tensor],
        beta_heads: list,
        omega_current: Optional[torch.Tensor],
        eps: float,
    ) -> None:
        r"""Store post-E-step state on ``self`` for the M-step.

        Writes: ``_last_alpha_i``, ``_last_beta_for_implicit``,
        ``_last_implicit_mu_scale``, ``_last_implicit_sigma_scale``,
        ``_last_omega``.
        """
        # Alpha
        if self.learnable_alpha:
            self._last_alpha_i = alpha_effective.detach()
        else:
            self._last_alpha_i = None

        # Beta for implicit EM
        if self.implicit_em:
            if beta_heads:
                self._last_beta_for_implicit = torch.stack(beta_heads, dim=1).detach()
            elif beta_current is not None:
                self._last_beta_for_implicit = beta_current.detach()
            else:
                self._last_beta_for_implicit = None

        # Implicit EM gradient scales
        if self.implicit_em:
            _beta_for_scale = getattr(self, '_last_beta_for_implicit', None)
            if _beta_for_scale is not None:
                _alpha_for_scale = alpha_effective if alpha_effective is not None else self.alpha
                mu_scale, sigma_scale = compute_implicit_em_scales(
                    alpha_i=_alpha_for_scale,
                    sigma_p=sigma_p,
                    beta=_beta_for_scale,
                    sigma_q=sigma_current if sigma_current is not None else sigma_p,
                    eps=eps,
                )
                self._last_implicit_mu_scale = mu_scale
                self._last_implicit_sigma_scale = sigma_scale
            else:
                self._last_implicit_mu_scale = None
                self._last_implicit_sigma_scale = None
        else:
            self._last_implicit_mu_scale = None
            self._last_implicit_sigma_scale = None

        # Omega for multi-layer propagation
        self._last_omega = omega_current

        # Latest-iteration β for downstream attention-pattern visualization,
        # always stored regardless of return_beta_history. When the enclosing
        # block has skip_attention=True there is no attention sublayer β to
        # plot, but the FFN's own β is the meaningful per-token attention
        # pattern and downstream plotting code reads this attribute via the
        # block. Detached so it doesn't keep the E-step graph alive past here.
        if beta_heads:
            self._last_beta = torch.stack(beta_heads, dim=1).detach()
        elif beta_current is not None:
            self._last_beta = beta_current.detach()
        else:
            self._last_beta = None

    # =================================================================
    # Helper: Prepare E-step inputs (sigma init, prior setup, alpha)
    # =================================================================

    def _prepare_e_step_inputs(
        self,
        mu: torch.Tensor,           # (B, N, K)
        sigma: Optional[torch.Tensor],
        mu_prior: torch.Tensor,     # (B, N, K)
        phi: torch.Tensor,          # (B, N, phi_dim)
        omega: Optional[torch.Tensor],
        sigma_prior: Optional[torch.Tensor],
        B: int,
        N: int,
        K: int,
        device: torch.device,
        dtype: torch.dtype,
        eps: float,
    ) -> dict:
        r"""Set up all inputs needed before E-step iterations begin.

        Performs sigma initialisation, prior (mu_p, sigma_p) extraction,
        sigma_p floor clamping, initial sigma clamping, phi/omega cloning,
        and alpha computation.  Returns a dict with keys:

            mu_p_current, sigma_p, mu_current, sigma_current,
            phi_current, omega_current, alpha_effective, _alpha_c0,
            is_diagonal, beta_history (empty list or None)
        """
        # ── Sigma initialisation ────────────────────────────────────────
        if sigma is None:
            if self.diagonal_covariance:
                sigma = torch.ones(B, N, K, device=device, dtype=dtype) * 0.1
            else:
                sigma = (0.1 * torch.eye(K, device=device, dtype=dtype)
                         .unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1).contiguous())

        sigma = squeeze_trailing_singletons(sigma)
        is_diagonal = sigma.dim() == 3

        # ── Prior setup: mu_p, sigma_p ──────────────────────────────────
        # No .clone() needed: downstream ops (clamp, +, =) create new tensors,
        # never mutating the original. .detach() suffices for graph separation.
        #
        # EM modes: mu_p is always an M-step variable from E-step's perspective.
        _em_active = self.em_phi_mode in ('E_phi_q', 'M_phi_p')
        if (self.amortized_inference and not self.implicit_em) and not _em_active:
            mu_p_current = mu_prior
        else:
            mu_p_current = mu_prior.detach()

        # sigma_p: detached by default (M-step parameter). When amortize_sigma=True
        # and amortized_inference=True, kept attached so E-step gradients flow
        # back through prior covariances — enables learned prior uncertainty.
        _attach_sigma = self.amortized_inference and self.amortize_sigma and not self.implicit_em
        if sigma_prior is not None:
            sigma_p = sigma_prior if _attach_sigma else sigma_prior.detach()
        else:
            sigma_p = sigma if _attach_sigma else sigma.detach()

        # E-step sigma_p floor
        _floor = self.e_step_sigma_floor
        if sigma_p.dim() == 3:
            # Diagonal covariance: elementwise clamp bounds every 1/σ_p_k
            # individually by 1/_floor. Already condition-number safe.
            sigma_p = sigma_p.clamp(min=_floor)
        elif self.spd_floor_mode == 'eigclamp':
            # Full covariance: enforce λ_min(Σ_p) ≥ _floor via spectral clamp.
            # This bounds the condition number κ(Σ_p) ≤ σ_max² / _floor, so
            # Σ_p^{-1} in the E-step natural gradient is bounded independently
            # of any off-diagonal correlation pattern. The previous 'ridge'
            # path (Σ_p + _floor·I) shifts every eigenvalue but does NOT bound
            # κ(Σ_p): a matrix [[1, 0.999], [0.999, 1]] has λ_min = 0.001,
            # ridge with _floor=0.01 gives λ_min = 0.011 — cond ≈ 180 — and
            # after GL(K) transport cond compounds as cond(Ω)²·cond(Σ_p).
            #
            # Gauge-covariance caveat: the clamp is applied in the stored
            # frame of Σ_p (the coordinate frame of prior_bank's sandwich
            # Σ_v = A_v·diag(exp(base_log_prior_sigma))·A_v^T). The
            # spd_eigfloor primitive has a strict GL(K)-covariant path
            # activated by passing exp_phi=A_v — see vfe_utils.spd_eigfloor.
            # At this site A_v is not exposed by prior_bank's current API;
            # the frame-dependent path is used, which still bounds
            # κ(Σ_p) ≤ σ_max²/_floor (a frame-independent invariant even
            # though the exact values are frame-dependent).
            # TODO: expose A_v from prior_bank and pass exp_phi=A_v_source
            #       here for strict GL(K) covariance at this site. The
            #       primitive is ready; only the plumbing is missing.
            sigma_p = spd_eigfloor(sigma_p, _floor, sigma_max=self.sigma_max)
        elif self.spd_floor_mode == 'ridge':
            # Legacy path for bit-identity with pre-2026-04-19 runs.
            K_sigma = sigma_p.shape[-1]
            I_K = torch.eye(K_sigma, device=sigma_p.device, dtype=sigma_p.dtype)
            sigma_p = sigma_p + _floor * I_K
        # spd_floor_mode == 'none' → no floor applied.

        # Convert diagonal sigma_p to full covariance if needed
        if not is_diagonal and sigma_p.dim() == 3:
            sigma_p = torch.diag_embed(sigma_p)

        # ── Initial belief state ─────────────────────────────────────────
        # No .clone() needed: the E-step loop reassigns mu_current/sigma_current
        # (mu_current = mu_current + delta), never mutating in-place.
        if self.implicit_em:
            mu_current = mu.detach()
            sigma_current = sigma.detach()
        else:
            mu_current = mu
            sigma_current = sigma

        # Clamp initial sigma to [eps, sigma_max]
        if self.update_sigma:
            _eps = 1e-6
            if sigma_current.dim() == 3:
                sigma_current = sigma_current.clamp(min=_eps, max=self.sigma_max)
            else:
                eigvals, eigvecs = _safe_eigh(sigma_current, jitter=_eps)
                eigvals = eigvals.clamp(min=_eps, max=self.sigma_max)
                sigma_current = eigvecs * eigvals.unsqueeze(-2) @ eigvecs.transpose(-1, -2)

        # ── Phi / omega setup ────────────────────────────────────────────
        # No .clone() needed: phi_current is reassigned (not mutated in-place)
        # in _retract_phi calls within the iteration loop.
        #
        # M_phi_p: phi is θ (model parameter), frozen during E-step.
        # E_phi_q: phi is q (belief), stays attached for E-step optimization.
        if self.em_phi_mode == 'M_phi_p':
            phi_current = phi.detach()
        elif not self.amortized_inference and self.detach_phi:
            phi_current = phi.detach()
        else:
            phi_current = phi
        # Omega analog of phi detach: under M_phi_p the gauge element is an
        # M-step parameter and must be frozen (graph severed) during the E-step.
        # Without this, KL expressions inside the E-step accumulate semi-gradient
        # into omega_embed through β/transport, turning em_phi_p into a hybrid.
        if self.em_phi_mode == 'M_phi_p' and omega is not None:
            omega_current = omega.detach()
        else:
            omega_current = omega

        # ── Alpha computation ────────────────────────────────────────────
        if self.learnable_alpha:
            alpha_effective = self.get_bayesian_alpha(
                mu_current, mu_p_current, sigma_p, sigma_current, eps=eps
            )
            _alpha_c0 = F.softplus(self.raw_c0)
        else:
            alpha_effective = self.alpha
            _alpha_c0 = None

        return {
            'mu_p_current': mu_p_current,
            'sigma_p': sigma_p,
            'mu_current': mu_current,
            'sigma_current': sigma_current,
            'phi_current': phi_current,
            'omega_current': omega_current,
            'alpha_effective': alpha_effective,
            '_alpha_c0': _alpha_c0,
            'is_diagonal': is_diagonal,
        }

    # =================================================================
    # Helper: Closed-form E-step (precision-weighted fixed point)
    # =================================================================

    def _closed_form_e_step(
        self,
        mu_current: torch.Tensor,
        sigma_current: Optional[torch.Tensor],
        phi_current: torch.Tensor,
        omega_current: Optional[torch.Tensor],
        mu_p_current: torch.Tensor,
        sigma_p: torch.Tensor,
        alpha_effective,
        _alpha_c0,
        is_diagonal: bool,
        B: int,
        N: int,
        device: torch.device,
        dtype: torch.dtype,
        eps: float,
        mask: Optional[torch.Tensor],
        return_beta_history: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor], list, Optional[list]]:
        r"""Compute the precision-weighted closed-form VFE fixed point.

        Thin wrapper around vfe_closed_form.run_closed_form_e_step that
        passes ``self`` as the first argument.  The method body was
        extracted in round 8 as part of the side-quest refactor; see
        vfe_closed_form.py for the full 500-line implementation (diagonal
        + full-covariance branches, Picard resolve loops, and phi
        evolution).
        """
        return _run_closed_form_e_step(
            self,
            mu_current=mu_current,
            sigma_current=sigma_current,
            phi_current=phi_current,
            omega_current=omega_current,
            mu_p_current=mu_p_current,
            sigma_p=sigma_p,
            alpha_effective=alpha_effective,
            _alpha_c0=_alpha_c0,
            is_diagonal=is_diagonal,
            B=B, N=N,
            device=device, dtype=dtype, eps=eps,
            mask=mask,
            return_beta_history=return_beta_history,
        )


    # =================================================================
    # Helper: VFE gradient debug printing (extracted from _vfe_iteration)
    # =================================================================

    def _print_vfe_grad_debug(
        self,
        iteration: int,
        is_diagonal: bool,
        mu_current: torch.Tensor,
        grad_mu: torch.Tensor,
        grad_sigma: torch.Tensor,
        nat_grad_mu: torch.Tensor,
        nat_grad_sigma: torch.Tensor,
        nat_grad_mu_norm: torch.Tensor,
        nat_grad_sigma_norm: torch.Tensor,
        max_nat_grad_norm: float,
        max_nat_grad_sigma_norm: float,
    ) -> None:
        r"""Print per-component gradient breakdown for VFE debug mode.

        Delegates to :func:`vfe_diagnostics.print_vfe_grad_debug`.
        """
        _print_vfe_grad_debug_impl(
            self, iteration, is_diagonal, mu_current,
            grad_mu, grad_sigma, nat_grad_mu, nat_grad_sigma,
            nat_grad_mu_norm, nat_grad_sigma_norm,
            max_nat_grad_norm, max_nat_grad_sigma_norm,
        )

    # =================================================================
    # Helper: Sigma SPD retraction + condition clamping + isotropy
    # =================================================================

    def _retract_sigma(
        self,
        sigma_current: torch.Tensor,
        nat_grad_sigma: torch.Tensor,
        effective_lr: float,
        is_diagonal: bool,
        eps: float,
    ) -> torch.Tensor:
        r"""Apply the SPD-preserving retraction to ``sigma_current``.

        Step size is ``self.sigma_lr`` (independent of μ LR, backed by
        ``raw_sigma_lr`` — Parameter when learnable_lr=True, buffer
        otherwise) modulated by the same per-iteration cosine ``decay_factor``
        that the caller's ``effective_lr`` carries (= ``self.lr * decay_factor``).
        We recover the decay factor by dividing out ``self.lr`` so μ and σ
        share the in-loop annealing schedule but remain independent in
        magnitude — when learnable, they receive separate autograd gradients.

        Trust region is ``self.e_step_sigma_trust`` (clamp on the whitened
        tangent δσ/σ before exp). Delegates to
        :func:`vfe_utils.retract_sigma_e_step`.
        """
        # F4: thread e_step_sigma_floor as the retraction eps for full-cov
        # so the Σ_q eigenvalue floor matches the Σ_p floor in magnitude.
        # Diagonal path keeps the caller-supplied eps (typically 1e-6) since
        # the elementwise clamp downstream already handles the per-dim floor.
        _retract_eps = self.e_step_sigma_floor if (
            not is_diagonal and self.spd_floor_mode == 'eigclamp'
        ) else eps
        # Recover the cosine decay factor from effective_lr / self.lr so σ
        # follows the same annealing schedule across E-step iterations
        # without being coupled to the μ step magnitude.
        _decay_factor = effective_lr / self.lr.detach().clamp(min=eps)
        _sigma_step = self.sigma_lr * _decay_factor
        return _retract_sigma_impl(
            sigma_current, nat_grad_sigma, _sigma_step,
            is_diagonal, _retract_eps,
            update_sigma=self.update_sigma,
            sigma_trust=self._get_sigma_trust(effective_lr),
            sigma_max=self.sigma_max,
            isotropic_covariance=self.isotropic_covariance,
        )

    # =================================================================
    # Helper: Per-iteration diagnostics + fiber trajectory recording
    # =================================================================

    def _record_iteration_diagnostics(
        self,
        iteration: int,
        grad_mu: torch.Tensor,
        grad_sigma: torch.Tensor,
        nat_grad_mu: torch.Tensor,
        nat_grad_sigma: torch.Tensor,
        delta_mu: torch.Tensor,
        mu_current: torch.Tensor,
        sigma_current: torch.Tensor,
        mu_p_current: Optional[torch.Tensor],
        is_diagonal: bool,
        eps: float,
        effective_lr,
        scale: torch.Tensor,
        beta_current: Optional[torch.Tensor],
        beta_heads: Optional[list],
    ) -> None:
        r"""Collect per-iteration convergence diagnostics and fiber snapshots.

        Delegates to :func:`vfe_diagnostics.record_iteration_diagnostics`.
        """
        _record_iteration_diagnostics_impl(
            self, iteration, grad_mu, grad_sigma,
            nat_grad_mu, nat_grad_sigma, delta_mu,
            mu_current, sigma_current, mu_p_current,
            is_diagonal, eps, effective_lr, scale,
            beta_current, beta_heads,
        )

    # =================================================================
    # Helper: Multi-head VFE gradient computation (per-block β_h loop)
    # =================================================================

    def _compute_multihead_vfe_gradients(
        self,
        mu_current: torch.Tensor,
        sigma_current: torch.Tensor,
        phi_current: torch.Tensor,
        omega_current: Optional[torch.Tensor],
        mu_p_current: torch.Tensor,
        sigma_p: torch.Tensor,
        alpha_effective,
        _alpha_c0,
        is_diagonal: bool,
        B: int,
        N: int,
        eps: float,
        mask: Optional[torch.Tensor],
        _detach_e_step: bool,
        _precomputed_block_exp_pairs,
        _nonflat_omega: Optional[torch.Tensor],
        _nonflat_exp_phi: Optional[torch.Tensor] = None,  # (B, N, K, K) per-token exp(φ); paired with _nonflat_omega so gauge_covariant_ridge does not silently degrade to eps*I
        return_beta_history: bool = False,
    ):
        r"""Compute per-head β_h and VFE gradients for the multihead path.

        Each irrep block gets its own attention pattern β_h, maintaining head
        diversity through all VFE iterations.

        Returns a 6-tuple:
            ``(grad_mu, grad_sigma, beta_heads, beta_current,
               beta_history_entry, _mh_cached_bep)``

        where ``_mh_cached_bep`` is the cached block-exponential pairs (needed
        by the phi gradient and active-inference helpers).
        """
        # =============================================================
        # MULTI-HEAD VFE: Per-block β_h through iterations
        # =============================================================
        # Each irrep block gets its own attention pattern.
        # This maintains head diversity through all VFE iterations,
        # enabling different heads to cluster at different scales.
        grad_mu = torch.zeros_like(mu_current)
        grad_sigma = torch.zeros_like(sigma_current)
        beta_heads = []  # For history tracking

        # Precompute block_exp_pairs ONCE for all heads.
        # When update_phi_per_iteration=False, the caller passes in
        # precomputed pairs to avoid redundant matrix exp across iterations.
        if _precomputed_block_exp_pairs is not None:
            _mh_cached_bep = _precomputed_block_exp_pairs
        else:
            _mh_cached_bep = self._build_block_exp_pairs(
                phi_current, omega_current, B, N,
                mu_current.device, mu_current.dtype,
            )

        # =============================================================
        # FUSED MULTI-HEAD VFE: Compute β_h and gradients in single
        # Omega pass per head (eliminates redundant Omega construction).
        # Previously: compute_attention_weights + compute_vfe_gradients_gpu
        # built Omega separately → 2× Omega per head. Now: 1× per head.
        # =============================================================
        _use_fused_mh = (is_diagonal and self.irrep_dims is not None
                         and not self.exact_diagonal_transport)
        block_start = 0
        for h, d_h in enumerate(self.irrep_dims):
            block_end = block_start + d_h

            mu_h = mu_current[:, :, block_start:block_end]
            if _detach_e_step:
                mu_h = mu_h.detach()
            mu_h = mu_h.contiguous()
            mu_p_h = mu_p_current[:, :, block_start:block_end].contiguous()
            if is_diagonal:
                sigma_h = sigma_current[:, :, block_start:block_end]
                if _detach_e_step:
                    sigma_h = sigma_h.detach()
                sigma_h = sigma_h.contiguous()
                sigma_p_h = sigma_p[:, :, block_start:block_end].contiguous()
            else:
                sigma_h = sigma_current[:, :, block_start:block_end, block_start:block_end]
                if _detach_e_step:
                    sigma_h = sigma_h.detach()
                sigma_h = sigma_h.contiguous()
                sigma_p_h = sigma_p[:, :, block_start:block_end, block_start:block_end].contiguous()
            gen_h = self.generators[:, block_start:block_end, block_start:block_end]

            # Scale kappa by sqrt(d_h) to normalize KL across different-dim
            # super-blocks. Without this, larger blocks (e.g., 12-dim from
            # cross-coupled heads) produce proportionally larger KL values,
            # causing attention sharpness imbalance vs smaller blocks.
            kappa_h = self._get_kappa_h(h, d_h)
            _head_bep = [_mh_cached_bep[h]] if _mh_cached_bep is not None else None

            alpha_h = alpha_effective[:, :, block_start:block_end] if isinstance(alpha_effective, torch.Tensor) and alpha_effective.dim() == 3 else alpha_effective
            c0_h = _alpha_c0[block_start:block_end] if _alpha_c0 is not None else None

            # Non-flat transport: extract per-head Omega slice for this block.
            # IMPORTANT: include 'exp_phi' alongside 'Omega' so the downstream
            # gauge_covariant_ridge branches in attention.aggregate_messages can
            # build the covariant ridge eps*(g·g^T). Omitting 'exp_phi' makes
            # those branches silently fall back to eps*I, turning the opt-in
            # flag into a no-op under non-flat transport.
            _head_ct = None
            if _nonflat_omega is not None:
                if _nonflat_exp_phi is None:
                    raise ValueError(
                        "_nonflat_omega is set but _nonflat_exp_phi is None. "
                        "Both must be provided together so the cached transport "
                        "carries enough information to honor gauge_covariant_ridge "
                        "(which builds eps*(g·g^T) from exp_phi). The caller that "
                        "produced _nonflat_omega must also pass the matching "
                        "_nonflat_exp_phi tensor."
                    )
                _head_ct = {
                    'Omega': _nonflat_omega[:, :, :, block_start:block_end, block_start:block_end],
                    'exp_phi': _nonflat_exp_phi[:, :, block_start:block_end, block_start:block_end],
                }

            # EXPERIMENTAL: rope_full_gauge path uses an autograd-based
            # gradient computation that lifts σ to full covariance and
            # rotates BOTH μ and Σ by R(θ).  Slower than the fused path.
            # Only diagonal σ is supported.  See vfe_gradients.py for the
            # implementation.
            #
            # CRITICAL: torch.inference_mode() (used by model.generate)
            # marks tensors as inference tensors that cannot be promoted
            # to require_grad even inside enable_grad().  Detect this and
            # fall back to the standard fused path for generation.  The
            # fallback path still has the rope chain-rule fix (Option C),
            # so it's correct, just not the experimental Option-B.
            _use_rope_full = (
                getattr(self, '_rope_full_gauge_vfe', 'off') in ('vfe_only', 'both')
                and self._use_rope_vfe
                and _nonflat_omega is None
                and not torch.is_inference_mode_enabled()
            )
            if _use_rope_full:
                beta_h, grad_mu_h, grad_sigma_h = _compute_rope_full_gauge_gradient_per_head(
                    mu_h=mu_h, sigma_h=sigma_h,
                    mu_p_h=mu_p_h, sigma_p_h=sigma_p_h,
                    phi=phi_current, gen_h=gen_h,
                    alpha=alpha_h,
                    lambda_belief=self.lambda_belief,
                    lambda_softmax=self.lambda_softmax,
                    kappa=kappa_h, eps=eps,
                    rope_base=self._rope_base_vfe,
                    d_h=d_h,
                    cached_block_exp_pairs=_head_bep,
                    enforce_orthogonal=getattr(self, 'enforce_orthogonal', False),
                    mask=mask,
                    mask_self_attention=self.mask_self_attention,
                )
            elif _use_fused_mh and _nonflat_omega is None:
                # FUSED: single pass computes β_h AND gradients (1× Omega)
                # Not compatible with non-flat transport (fused path builds
                # Omega from block exp pairs internally).
                beta_h, grad_mu_h, grad_sigma_h, _ = _fused_attention_and_vfe_gradients_block_diag(
                    mu_q=mu_h, sigma_q=sigma_h,
                    mu_p=mu_p_h, sigma_p=sigma_p_h,
                    phi=phi_current, generators=gen_h,
                    alpha=alpha_h, lambda_belief=self.lambda_belief, lambda_softmax=self.lambda_softmax,
                    kappa=kappa_h, eps=eps,
                    irrep_dims=[d_h],
                    compute_sigma_align_grad=self.compute_sigma_align_grad,
                    enforce_orthogonal=getattr(self, 'enforce_orthogonal', False),
                    alpha_c0=c0_h,
                    cached_block_exp_pairs=_head_bep,
                    mask=mask,
                    mask_self_attention=self.mask_self_attention,
                    use_rope=self._use_rope_vfe,
                    rope_base=self._rope_base_vfe,
                    alpha_div=self.alpha_divergence,
                )
            else:
                # Fallback: separate attention + gradient (full covariance)
                # When _head_ct has non-flat Omega, it takes priority over
                # cached_block_exp_pairs in both functions.
                beta_h = compute_attention_weights(
                    mu_q=mu_h, sigma_q=sigma_h,
                    phi=phi_current, generators=gen_h,
                    kappa=kappa_h, epsilon=eps, mask=mask,
                    return_kl=False,
                    diagonal_covariance=is_diagonal,
                    irrep_dims=[d_h],
                    mask_self_attention=self.mask_self_attention,
                    gauge_mode=self.gauge_mode,
                    cached_block_exp_pairs=_head_bep,
                    cached_transport=_head_ct,
                    use_rope=self._use_rope_vfe,
                    rope_base=self._rope_base_vfe,
                    exact_diagonal_transport=self.exact_diagonal_transport,
                    alpha_divergence=self.alpha_divergence,
                    sigma_floor=(self.e_step_sigma_floor
                                 if self.spd_floor_mode == 'eigclamp' and not is_diagonal
                                 else None),
                    spd_floor_mode=self.spd_floor_mode,
                    propagate_nonfinite=self.propagate_kl_nonfinite,
                )
                grad_mu_h, grad_sigma_h = compute_vfe_gradients_gpu(
                    mu_q=mu_h, sigma_q=sigma_h,
                    mu_p=mu_p_h, sigma_p=sigma_p_h,
                    beta=beta_h, phi=phi_current, generators=gen_h,
                    alpha=alpha_h, lambda_belief=self.lambda_belief, lambda_softmax=self.lambda_softmax,
                    kappa=kappa_h, eps=eps, alpha_c0=c0_h,
                    compute_sigma_align_grad=self.compute_sigma_align_grad,
                    irrep_dims=[d_h],
                    cached_block_exp_pairs=_head_bep,
                    cached_transport=_head_ct,
                    exact_diagonal_transport=self.exact_diagonal_transport,
                    use_rope=self._use_rope_vfe,
                    rope_base=self._rope_base_vfe,
                    alpha_div=self.alpha_divergence,
                )

            beta_heads.append(beta_h)
            grad_mu[:, :, block_start:block_end] = grad_mu_h
            if is_diagonal:
                grad_sigma[:, :, block_start:block_end] = grad_sigma_h
            else:
                # compute_vfe_gradients_gpu squeezes trailing singletons,
                # so d_h=1 heads return (B,N,1) instead of (B,N,1,1).
                # Restore full covariance shape for block assignment.
                if grad_sigma_h.dim() == 3 and d_h == 1:
                    grad_sigma_h = grad_sigma_h.unsqueeze(-1)
                grad_sigma[:, :, block_start:block_end, block_start:block_end] = grad_sigma_h

            # Accumulate per-head debug norms (sum of squares for proper norm aggregation)
            if _vfe_utils_mod._VFE_GRAD_DEBUG is not None and _vfe_utils_mod._VFE_GRAD_DEBUG:
                _pfx = f'head{h}(d={d_h})'
                for _k, _v in list(_vfe_utils_mod._VFE_GRAD_DEBUG.items()):
                    # Store per-head values with prefix, clear base keys
                    _vfe_utils_mod._VFE_GRAD_DEBUG[f'{_pfx}/{_k}'] = _v
                # Reset base keys for next head
                for _k in [k for k in _vfe_utils_mod._VFE_GRAD_DEBUG if '/' not in k]:
                    del _vfe_utils_mod._VFE_GRAD_DEBUG[_k]

            block_start = block_end

        beta_history_entry = None
        if return_beta_history:
            beta_stacked = torch.stack(beta_heads, dim=1)
            beta_history_entry = beta_stacked.detach()
        beta_current = beta_heads[-1]

        return grad_mu, grad_sigma, beta_heads, beta_current, beta_history_entry, _mh_cached_bep

    # =================================================================
    # Helper: Single VFE iteration (extracted from forward loop)
    # =================================================================

    def _vfe_iteration(
        self,
        iteration: int,
        mu_current: torch.Tensor,
        sigma_current: Optional[torch.Tensor],
        phi_current: torch.Tensor,
        omega_current: Optional[torch.Tensor],
        mu_p_current: torch.Tensor,
        sigma_p: torch.Tensor,
        alpha_effective,
        _alpha_c0,
        is_diagonal: bool,
        B: int,
        N: int,
        eps: float,
        mask: Optional[torch.Tensor],
        return_beta_history: bool,
        _detach_e_step: bool = True,
        _precomputed_block_exp_pairs=None,
        connection_delta: Optional[torch.Tensor] = None,
        cocycle_relaxation: float = 0.0,
    ):
        r"""Execute one VFE natural-gradient iteration.

        Args:
            _precomputed_block_exp_pairs: If provided, reuse these block exp
                pairs instead of recomputing. Used when update_phi_per_iteration
                is False to avoid redundant matrix exponentials across iterations.
            connection_delta: Frozen non-flat connection δ_ij (B, N, N, n_gen).
                When provided, transport becomes Ω = exp(φ_i)·exp(α·δ·G)·exp(−φ_j).
            cocycle_relaxation: Scale factor for connection_delta.

        Returns:
            (mu_current, sigma_current, phi_current, omega_current,
             beta_current, beta_heads, alpha_effective, beta_history_entry)
            where beta_history_entry is a stacked beta (if return_beta_history)
            or None.
        """
        beta_heads = []
        beta_current = None
        beta_history_entry = None
        _is_final_iter = (iteration == self.n_iterations - 1)

        # Cosine decay: lr drops from 1.0 to 0.1 across iterations
        # Steeper than linear 0.5 decay — stabilizes later iterations where
        # natural gradients can amplify and cause oscillatory divergence
        if self.n_iterations > 1:
            progress = iteration / (self.n_iterations - 1)  # 0→1
            decay_factor = 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            decay_factor = 1.0
        effective_lr = self.lr * decay_factor

        # =================================================================
        # STEP 0: Precompute transport operators ONCE per iteration
        # =================================================================
        # Both compute_attention_weights and compute_vfe_gradients_gpu need
        # the same Ω_ij = exp(φ_i)·exp(-φ_j). Computing once and passing
        # cached_transport avoids redundant matrix exponentials.
        # Skip caching when using block-diagonal or chunked paths (they
        # compute transport internally in chunks to save memory).
        cached_transport = None
        _nonflat_omega = None  # Per-head sliceable non-flat Omega (multihead path)

        # =================================================================
        # Determine alpha: Bayesian precision or fixed scalar
        # Recompute per-iteration for learnable alpha (state-dependent);
        # for fixed alpha, alpha_effective was set before the loop.
        # =================================================================
        if self.learnable_alpha:
            alpha_effective = self.get_bayesian_alpha(
                mu_current, mu_p_current, sigma_p, sigma_current, eps=eps
            )  # (B, N, K) - per-dim gauge-invariant, state-dependent

        # Initialize cached block exp pairs for phi gradient reuse
        _mh_cached_bep = None
        beta_heads = None

        # Enable/disable VFE gradient debug dict for this iteration
        # _vfe_utils_mod._VFE_GRAD_DEBUG accessed via _vfe_utils_mod
        if self._debug_vfe_gradients or self._collect_vfe_metrics:
            _vfe_utils_mod._VFE_GRAD_DEBUG = {}
        else:
            _vfe_utils_mod._VFE_GRAD_DEBUG = None

        if self.multihead_vfe:
            (grad_mu, grad_sigma, beta_heads, beta_current,
             beta_history_entry, _mh_cached_bep) = self._compute_multihead_vfe_gradients(
                mu_current=mu_current,
                sigma_current=sigma_current,
                phi_current=phi_current,
                omega_current=omega_current,
                mu_p_current=mu_p_current,
                sigma_p=sigma_p,
                alpha_effective=alpha_effective,
                _alpha_c0=_alpha_c0,
                is_diagonal=is_diagonal,
                B=B, N=N, eps=eps,
                mask=mask,
                _detach_e_step=_detach_e_step,
                _precomputed_block_exp_pairs=_precomputed_block_exp_pairs,
                _nonflat_omega=_nonflat_omega,
                return_beta_history=return_beta_history,
            )
        else:
            raise RuntimeError(
                "VariationalFFNDynamic requires irrep_dims to be set. "
                "The single-beta (irrep_dims=None) path has been removed. "
                "Set ffn_irrep_dims in BlockConfig or pass irrep_dims to the constructor."
            )
        # =====================================================================
        # Active inference / EFE gradients (delegated)
        # =====================================================================
        # compute_ai_gradients handles the master toggle, weight gates, ref
        # resolution, and debug dict writes — see active_inference.py.
        # Returns (None, None) when all AI terms are off (default).
        # W_out readout is resolved inside compute_ai_gradients from the
        # _ai_w_out_ref wired by wire_readout_references.
        _bep_for_ai = _mh_cached_bep
        (_ai_pending_grad_mu, _ai_pending_grad_sigma) = compute_ai_gradients(
            ffn=self,
            mu_current=mu_current,
            sigma_current=sigma_current,
            W_out=None,
            beta_current=beta_current,
            beta_heads=beta_heads,
            cached_block_exp_pairs=_bep_for_ai,
            irrep_dims=self.irrep_dims,
        )

        # =====================================================================
        # Unified natural gradient: fold EFE into grad_mu/sigma
        # =====================================================================
        # Fold EFE Euclidean grads into grad_mu / grad_sigma BEFORE the
        # Fisher natural-gradient projection.  The Fisher metric F(θ) is a
        # property of the belief parameterization q_θ = N(μ, Σ), not the
        # loss function (Amari 1998).  All scalar functionals of the belief
        # parameters — KL, entropy, MI — get the same natural gradient:
        # ∇̃f = F⁻¹·∇f.  The observation CE gradient (discrete_obs_grad,
        # added above) already gets this treatment; EFE is structurally
        # identical and must not be treated differently.
        if _is_final_iter:
            if _ai_pending_grad_mu is not None:
                self._e_step_grad_norms['ai_efe_raw_grad_norm'] = (
                    _ai_pending_grad_mu.detach().norm().item())
            if _ai_pending_grad_sigma is not None:
                self._e_step_grad_norms['ai_efe_sigma_raw_grad_norm'] = (
                    _ai_pending_grad_sigma.detach().norm().item())
        if _ai_pending_grad_mu is not None:
            grad_mu = grad_mu + _ai_pending_grad_mu
        if _ai_pending_grad_sigma is not None:
            grad_sigma = grad_sigma + _ai_pending_grad_sigma

        # Debug: Euclidean totals (after obs + EFE folding, before clip)
        if _vfe_utils_mod._VFE_GRAD_DEBUG is not None:
            _vfe_utils_mod._VFE_GRAD_DEBUG['euclidean_mu_total'] = _grad_norm(grad_mu)
            _vfe_utils_mod._VFE_GRAD_DEBUG['euclidean_sigma_total'] = _grad_norm(grad_sigma)
            _ps = _per_pos_stats(grad_mu)
            _vfe_utils_mod._VFE_GRAD_DEBUG['euclidean_mu_pos_mean'] = _ps[0]
            _vfe_utils_mod._VFE_GRAD_DEBUG['euclidean_mu_pos_max'] = _ps[1]
            _ps = _per_pos_stats(grad_sigma)
            _vfe_utils_mod._VFE_GRAD_DEBUG['euclidean_sigma_pos_mean'] = _ps[0]
            _vfe_utils_mod._VFE_GRAD_DEBUG['euclidean_sigma_pos_max'] = _ps[1]

        # =================================================================
        # STEP 3+4: Natural gradient projection, clamping, trust region
        # =================================================================
        _ng = _apply_natgrad_step(
            grad_mu=grad_mu,
            grad_sigma=grad_sigma,
            sigma_current=sigma_current,
            is_diagonal=is_diagonal,
            eps=eps,
            effective_lr=effective_lr,
            isotropic_covariance=self.isotropic_covariance,
            compute_natural_gradient_fn=compute_natural_gradient_gpu,
        )
        nat_grad_mu = _ng['nat_grad_mu']
        nat_grad_sigma = _ng['nat_grad_sigma']
        nat_grad_mu_norm = _ng['nat_grad_mu_norm']
        nat_grad_sigma_norm = _ng['nat_grad_sigma_norm']
        max_nat_grad_norm = _ng['max_nat_grad_norm']
        max_nat_grad_sigma_norm = _ng['max_nat_grad_sigma_norm']
        delta_mu = _ng['delta_mu']
        scale = _ng['scale']
        grad_mu = _ng['grad_mu_clipped']
        grad_sigma = _ng['grad_sigma_clipped']
        mu_current = mu_current + _ng['mu_delta']

        # Store E-step gradient norms (final iteration only to avoid CUDA sync)
        if _is_final_iter:
            _mu_norm = nat_grad_mu.detach().norm()
            _sig_norm = nat_grad_sigma.detach().norm()
            if is_diagonal:
                _cap_frac = (
                    nat_grad_sigma_norm.squeeze(-1) >= max_nat_grad_sigma_norm * 0.99
                ).float().mean()
            else:
                _cap_frac = (
                    nat_grad_sigma_norm.squeeze(-1).squeeze(-1) >= max_nat_grad_sigma_norm * 0.99
                ).float().mean()
            _mu_cap = (
                nat_grad_mu_norm.squeeze(-1) >= max_nat_grad_norm * 0.99
            ).float().mean()
            _scalars = torch.stack([_mu_norm, _sig_norm, _mu_cap, _cap_frac]).cpu().tolist()
            self._e_step_grad_norms['nat_grad_mu'] = _scalars[0]
            self._e_step_grad_norms['nat_grad_sigma'] = _scalars[1]
            self._e_step_grad_norms['nat_grad_mu_clipped'] = _scalars[0]
            self._e_step_grad_norms['nat_grad_sigma_clipped'] = _scalars[1]
            self._e_step_grad_norms['mu_cap_frac'] = _scalars[2]
            self._e_step_grad_norms['sigma_cap_frac'] = _scalars[3]
            self._e_step_grad_norms['mu_trust_frac'] = (
                scale.squeeze(-1) < 0.99
            ).float().mean().item()
            self._e_step_grad_norms['whitened_mu_mean'] = _ng['whitened_norm'].mean().item()
            self._e_step_grad_norms['whitened_mu_max'] = _ng['whitened_norm'].max().item()

        # Debug: per-component gradient breakdown
        self._print_vfe_grad_debug(
            iteration=iteration,
            is_diagonal=is_diagonal,
            mu_current=mu_current,
            grad_mu=grad_mu,
            grad_sigma=grad_sigma,
            nat_grad_mu=nat_grad_mu,
            nat_grad_sigma=nat_grad_sigma,
            nat_grad_mu_norm=nat_grad_mu_norm,
            nat_grad_sigma_norm=nat_grad_sigma_norm,
            max_nat_grad_norm=max_nat_grad_norm,
            max_nat_grad_sigma_norm=max_nat_grad_sigma_norm,
        )

        sigma_current = self._retract_sigma(
            sigma_current=sigma_current,
            nat_grad_sigma=nat_grad_sigma,
            effective_lr=effective_lr,
            is_diagonal=is_diagonal,
            eps=eps,
        )

        self._record_iteration_diagnostics(
            iteration=iteration,
            grad_mu=grad_mu,
            grad_sigma=grad_sigma,
            nat_grad_mu=nat_grad_mu,
            nat_grad_sigma=nat_grad_sigma,
            delta_mu=delta_mu,
            mu_current=mu_current,
            sigma_current=sigma_current,
            mu_p_current=mu_p_current,
            is_diagonal=is_diagonal,
            eps=eps,
            effective_lr=effective_lr,
            scale=scale,
            beta_current=beta_current,
            beta_heads=beta_heads,
        )

        # =============================================================
        # STEP 4b: Optional Gauge Frame Evolution DURING E-step
        # =============================================================
        _skip_phi_update = self.gauge_mode in ('trivial', 'constant')
        _use_omega = omega_current is not None and getattr(self, 'gauge_param', 'phi') == 'omega'

        # M_phi_p contract: φ is a model (M-step) parameter and must remain
        # frozen in value during E-step iterations. Earlier code allowed
        # _retract_phi to evolve phi_current here even when em_phi_mode='M_phi_p'
        # (silently violating the contract); excluded explicitly now.
        if (self.update_phi_per_iteration and torch.is_grad_enabled()
                and not _skip_phi_update
                and self.em_phi_mode != 'M_phi_p'):

            if _use_omega:
                # Direct Omega path: no matrix_exp, no dexp series
                grad_omega = self._compute_omega_grad_direct(
                    omega_current, mu_current, sigma_current,
                    is_diagonal, mask, eps,
                )
                if grad_omega is not None:
                    self._e_step_grad_norms['grad_phi'] = grad_omega.detach().norm().item()
                    omega_current = self._retract_omega(
                        omega_current, grad_omega, self.phi_lr,
                        trust_region=getattr(self, 'omega_trust_region', 0.3),
                    )
            else:
                # Phi path (existing): matrix_exp + dexp series
                _phi_bep = _mh_cached_bep
                grad_phi = self._compute_phi_grad(
                    phi_current, mu_current, sigma_current,
                    is_diagonal, mask, eps,
                    cached_block_exp_pairs=_phi_bep,
                    beta_current=beta_current,
                    beta_heads=beta_heads,
                )
                if grad_phi is not None:
                    self._e_step_grad_norms['grad_phi'] = grad_phi.detach().norm().item()
                    phi_current = _retract_phi(
                        phi=phi_current,
                        delta_phi=-grad_phi,
                        generators=self.generators,
                        step_size=self.phi_lr,
                        max_norm=self.phi_max_norm,
                        project_slk=self.phi_project_slk,
                        trace_clamp=self.phi_trace_clamp,
                        irrep_dims=self.irrep_dims,
                    )

        return (mu_current, sigma_current, phi_current, omega_current,
                beta_current, beta_heads, alpha_effective, beta_history_entry)

    def forward(
        self,
        mu: torch.Tensor,          # (B, N, K) - current beliefs
        beta: torch.Tensor = None, # (B, n_heads, N, N) - UNUSED, kept for API compat
        mu_prior: torch.Tensor = None,    # (B, N, K) - embedding priors
        phi: torch.Tensor = None,         # (B, N, phi_dim) - gauge frames
        sigma: Optional[torch.Tensor] = None,  # (B, N, K, K) or (B, N, K) if diagonal
        mask: Optional[torch.Tensor] = None,   # (B, N, N) - causal mask
        token_ids: Optional[torch.Tensor] = None,  # (B, N) - token IDs for PriorBank lookup
        return_beta_history: bool = False,  # Return β evolution for analysis
        omega: Optional[torch.Tensor] = None,  # (B, N, K, K) direct group elements (gauge_param='omega')
        sigma_prior: Optional[torch.Tensor] = None,  # (B, N, K, K) or (B, N, K) - embedding prior covariance
        connection_delta: Optional[torch.Tensor] = None,  # (B, N, N, n_gen) frozen non-flat connection
        cocycle_relaxation: float = 0.0,  # Scale for connection_delta: 0=flat, 1=fully non-flat
        precomputed_block_exp_pairs: Optional[list] = None,  # Shared block exp pairs from blocks.py
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[list]]:
        """
        Dynamic VFE E-step descent with beta recomputation at each iteration.

        Flow at each iteration:
            1. beta = softmax(-KL(q||Omega[q])/kappa)  [recompute from current beliefs]
            2. dF/dmu = alpha*(mu-mu_p)/sigma_p + lambda*Sum_j beta*(dKL/dmu)
                        + Sum_j KL*(dBeta/dmu) + dCE/dmu
            3. mu <- mu - eta * F_inv * dF/dmu  [natural gradient descent]
            4. (Optional) sigma <- retract_spd(sigma, -eta * dF/dsigma)
            5. (Optional) phi <- retract(phi, -eta_phi * dF/dphi)

        When multihead_vfe=True, each irrep block gets its own beta_h.
        When use_prior_bank=True, priors come from PriorBank via token_ids.

        Args:
            mu: Current belief means (B, N, K).
            beta: UNUSED, kept for API compatibility.
            mu_prior: Prior means from embeddings (B, N, K).
            phi: Gauge frames (B, N, phi_dim) where phi_dim = n_gen.
            sigma: Belief covariances - (B, N, K, K) full or (B, N, K) diagonal.
                When diagonal_covariance=True, shape is (B, N, K).
            mask: Causal mask (B, N, N) where 0 = cannot attend.
            token_ids: Token IDs (B, N) for PriorBank lookup. Required when
                use_prior_bank=True.
            return_beta_history: If True, return list of beta at each step.
            sigma_prior: Embedding prior covariance (B, N, K, K) or (B, N, K).
                When provided, used as sigma_p in the E-step (proper prior
                reference). When None, falls back to sigma.detach() (legacy).
            connection_delta: Frozen non-flat connection δ_ij (B, N, N, n_gen).
                Computed once from initial μ by GaugeConnection in blocks.py.
                Treated as an E-step constant (like σ_p).
            cocycle_relaxation: Scale factor for connection_delta (0=flat, 1=non-flat).

        Returns:
            mu_new: Updated beliefs (B, N, K).
            sigma_new: Updated covariances (same shape as input) or None.
            phi_new: Updated gauge frames (B, N, phi_dim).
            beta_history: List of beta tensors if return_beta_history, else None.
        """
        B, N, K = mu.shape
        device = mu.device
        dtype = mu.dtype
        eps = 1e-6

        # =================================================================
        # SAFETY: Disable autocast if active. The VFE inner loop uses
        # analytical gradients with eigh, sqrt, log, exp, matrix inv —
        # all of which need float32. If caller has autocast enabled,
        # disable it and upcast inputs.
        # =================================================================
        _amp_active = torch.is_autocast_enabled('cuda')
        _amp_ctx = torch.amp.autocast('cuda', enabled=False) if _amp_active else None
        if _amp_ctx is not None:
            _amp_ctx.__enter__()
            mu = mu.float()
            if sigma is not None:
                sigma = sigma.float()
            mu_prior = mu_prior.float()
            phi = phi.float()
            if sigma_prior is not None:
                sigma_prior = sigma_prior.float()
            if omega is not None:
                omega = omega.float()

        # Wrap VFE body in try/finally so autocast is always restored, even
        # if the loop raises (Cholesky failure, NaN, etc.).
        try:
            return self._forward_vfe_body(
                mu, sigma, mu_prior, phi, omega, sigma_prior,
                mask, token_ids, return_beta_history,
                connection_delta, cocycle_relaxation, precomputed_block_exp_pairs,
                B, N, K, device, dtype, eps,
            )
        finally:
            if _amp_ctx is not None:
                _amp_ctx.__exit__(None, None, None)

    def _forward_vfe_body(
        self,
        mu, sigma, mu_prior, phi, omega, sigma_prior,
        mask, token_ids, return_beta_history,
        connection_delta, cocycle_relaxation, precomputed_block_exp_pairs,
        B, N, K, device, dtype, eps,
    ):
        """VFE E-step body, factored out so forward() can wrap it in try/finally."""
        # ── Prepare all E-step inputs ────────────────────────────────────
        _state = self._prepare_e_step_inputs(
            mu, sigma, mu_prior, phi, omega, sigma_prior,
            B, N, K, device, dtype, eps,
        )
        mu_current = _state['mu_current']
        sigma_current = _state['sigma_current']
        phi_current = _state['phi_current']
        omega_current = _state['omega_current']
        mu_p_current = _state['mu_p_current']
        sigma_p = _state['sigma_p']
        is_diagonal = _state['is_diagonal']
        alpha_effective = _state['alpha_effective']
        _alpha_c0 = _state['_alpha_c0']

        # Track β evolution if requested
        beta_history = [] if return_beta_history else None
        # Detach belief inputs to the analytical VFE gradient kernels ONLY
        # when amortized_inference=False.  In amortized mode, keeping
        # mu_current / sigma_current attached preserves the full
        # iteration-to-iteration coupling so gradients propagate through
        # the entire E-step loop back to the priors (mu_p and phi).
        # See CLAUDE.md: "amortized_inference: gradient flow through
        # priors for learned E-step init".  The prior sigma_p remains
        # detached unconditionally elsewhere (M-step only per the
        # hard-constraint doc).
        _detach_e_step = not self.amortized_inference
        beta_current = None
        beta_heads = []

        # =====================================================================
        # CLOSED-FORM E-STEP: Precision-weighted fixed point (optional)
        # =====================================================================
        if self.closed_form_e_step:
            (mu_current, sigma_current, phi_current, omega_current,
             beta_heads, _cf_beta_history) = self._closed_form_e_step(
                mu_current=mu_current,
                sigma_current=sigma_current,
                phi_current=phi_current,
                omega_current=omega_current,
                mu_p_current=mu_p_current,
                sigma_p=sigma_p,
                alpha_effective=alpha_effective,
                _alpha_c0=_alpha_c0,
                is_diagonal=is_diagonal,
                B=B, N=N,
                device=device, dtype=dtype, eps=eps,
                mask=mask,
                return_beta_history=return_beta_history,
            )
            if return_beta_history and _cf_beta_history:
                beta_history = _cf_beta_history
            beta_current = beta_heads[-1] if beta_heads else None
        # =====================================================================
        # VFE Descent Loop with Dynamic β (runs outside AMP autocast)
        # =====================================================================
        # Skip when closed_form_e_step handled the E-step above.
        _n_iters = 0 if self.closed_form_e_step else self.n_iterations

        # When phi is frozen across iterations, precompute block exp pairs
        # once to avoid redundant matrix exponentials (Finding 23: ~2-3x speedup
        # on matrix exp cost with 3 iterations).
        # precomputed_block_exp_pairs from blocks.py: shared with attention sublayer,
        # avoids redundant fused_block_matrix_exp_pairs (same phi, same generators).
        # When phi evolves per iteration, externally precomputed BEP becomes
        # stale after iteration 1 — force recomputation from current phi.
        _hoisted_bep = precomputed_block_exp_pairs if not self.update_phi_per_iteration else None
        if _hoisted_bep is None and _n_iters > 1 and not self.update_phi_per_iteration:
            # detach_phi is only honored in non-amortized mode (matches the
            # pattern at lines 1381-1384).  In amortized mode, phi must
            # remain attached so the cached exp pairs carry the phi autograd
            # graph through the E-step iteration loop for M-step gradient
            # updates on phi.
            _amortized = getattr(self, 'amortized_inference', True)
            _honor_detach_phi = (not _amortized) and self.detach_phi
            _phi_for_cache = phi_current.detach() if _honor_detach_phi else phi_current
            _hoisted_bep = self._build_block_exp_pairs(
                _phi_for_cache,
                omega_current, B, N,
                mu_current.device, mu_current.dtype,
            )

        # Gradient checkpointing (Finding 26): trade ~2x compute for ~3x
        # memory savings with 3 iterations. Only checkpoint non-final
        # iterations — the final iteration's activations are needed for the
        # backward pass regardless.
        _use_ckpt = (self.gradient_checkpoint_vfe and self.training
                     and _n_iters > 1 and torch.is_grad_enabled())

        _early_exit_tol = self.e_step_early_exit_tol
        _mu_prev = None  # For early exit convergence check

        for iteration in range(_n_iters):
            _iter_kwargs = dict(
                iteration=iteration,
                mu_current=mu_current,
                sigma_current=sigma_current,
                phi_current=phi_current,
                omega_current=omega_current,
                mu_p_current=mu_p_current,
                sigma_p=sigma_p,
                alpha_effective=alpha_effective,
                _alpha_c0=_alpha_c0,
                is_diagonal=is_diagonal,
                B=B, N=N, eps=eps,
                mask=mask,
                return_beta_history=return_beta_history,
                _detach_e_step=_detach_e_step,
                _precomputed_block_exp_pairs=_hoisted_bep,
                connection_delta=connection_delta,
                cocycle_relaxation=cocycle_relaxation,
            )
            _is_final = (iteration == _n_iters - 1)
            if _use_ckpt and not _is_final:
                (mu_current, sigma_current, phi_current, omega_current,
                 beta_current, beta_heads, alpha_effective,
                 _iter_beta) = torch.utils.checkpoint.checkpoint(
                    self._vfe_iteration,
                    use_reentrant=False,
                    **_iter_kwargs,
                )
            else:
                (mu_current, sigma_current, phi_current, omega_current,
                 beta_current, beta_heads, alpha_effective,
                 _iter_beta) = self._vfe_iteration(**_iter_kwargs)
            if return_beta_history and _iter_beta is not None:
                beta_history.append(_iter_beta)

            # Early exit: if beliefs have converged, skip remaining non-final iterations.
            # The final iteration must always run for correct gradient graph construction.
            if _early_exit_tol is not None and iteration < _n_iters - 2:
                if _mu_prev is not None:
                    _delta = (mu_current.detach() - _mu_prev).norm()
                    _scale = _mu_prev.norm().clamp(min=eps)
                    if (_delta / _scale).item() < _early_exit_tol:
                        break
                _mu_prev = mu_current.detach()

        # =================================================================
        # DEQ implicit differentiation: replace straight-through backward
        # with Neumann-series approximation of (I - J)^{-1}
        # =================================================================
        if self.use_deq and self.training and torch.is_grad_enabled():
            if self.deq_include_phi and self.update_phi:
                # Joint (μ, Σ, φ) fixed-point: IFT corrects ALL three variables,
                # eliminating the straight-through bias in the M-step φ gradient.
                step_fn = self._make_deq_step_fn_with_phi(
                    mu_p_current, sigma_p,
                    mask, is_diagonal, eps, dtype,
                )
                mu_current, sigma_current, phi_current = DEQFixedPointFull.apply(
                    mu_current, sigma_current, phi_current, step_fn,
                    self.n_iterations, self.deq_neumann_terms,
                )
            else:
                # Original (μ, Σ)-only fixed point; φ gets straight-through gradient.
                step_fn = self._make_deq_step_fn(
                    phi_current, mu_p_current, sigma_p,
                    mask, is_diagonal, eps, dtype,
                )
                mu_current, sigma_current = DEQFixedPoint.apply(
                    mu_current, sigma_current, step_fn,
                    self.n_iterations, self.deq_neumann_terms,
                )

        # =================================================================
        # STEP 5: Optional Gauge Frame Evolution via VFE Gradient (after loop)
        # =================================================================
        # Skip when closed_form_e_step already handled phi evolution above.
        _use_omega = omega_current is not None and getattr(self, 'gauge_param', 'phi') == 'omega'
        _skip_phi_post = self.closed_form_e_step and is_diagonal  # Already done in closed-form path
        # em_phi_mode == 'M_phi_p' forbids any in-E-step evolution of phi/omega.
        # The per-iter block at ~line 2214 already honors this; the post-loop
        # block used to run unconditionally and silently violated the contract
        # when update_phi_per_iteration=False.
        if (self.update_phi and not self.update_phi_per_iteration
                and torch.is_grad_enabled()
                and self.gauge_mode not in ('trivial', 'constant')
                and self.em_phi_mode != 'M_phi_p'
                and not _skip_phi_post):

            if _use_omega:
                # Direct Omega path
                grad_omega = self._compute_omega_grad_direct(
                    omega_current, mu_current, sigma_current,
                    is_diagonal, mask, eps,
                )
                if grad_omega is not None:
                    omega_current = self._retract_omega(
                        omega_current, grad_omega, self.phi_lr,
                        trust_region=getattr(self, 'omega_trust_region', 0.3),
                    )
            else:
                # Phi path (existing)
                _phi_bep_post = None
                if self.irrep_dims is not None and self.gauge_mode == 'learned':
                    _phi_bep_post = fused_block_matrix_exp_pairs(
                        phi_current, self.generators, self.irrep_dims,
                        enforce_orthogonal=getattr(self, 'enforce_orthogonal', False),
                        skew_symmetric=self._generators_are_skew,
                    )
                grad_phi = self._compute_phi_grad(
                    phi_current, mu_current, sigma_current,
                    is_diagonal, mask, eps,
                    cached_block_exp_pairs=_phi_bep_post,
                    beta_current=beta_current,
                    beta_heads=beta_heads,
                )
                if grad_phi is not None:
                    phi_current = _retract_phi(
                        phi=phi_current,
                        delta_phi=-grad_phi,
                        generators=self.generators,
                        step_size=self.phi_lr,
                        max_norm=self.phi_max_norm,
                        project_slk=self.phi_project_slk,
                        trace_clamp=self.phi_trace_clamp,
                        irrep_dims=self.irrep_dims,
                    )

        # Cast results back to original dtype when AMP upcasted inputs
        if dtype != torch.float32:
            mu_current = mu_current.to(dtype)
            if sigma_current is not None:
                sigma_current = sigma_current.to(dtype)
            phi_current = phi_current.to(dtype)

        # Store post-E-step state for M-step
        self._finalize_e_step(
            alpha_effective, sigma_p, sigma_current,
            beta_current, beta_heads, omega_current, eps,
        )

        # ── EM boundary ──────────────────────────────────────────────────
        # In principled EM modes, q* is held fixed during the M-step.
        # Detach E-step outputs so no gradient flows back through the
        # E-step update map.  The M-step receives F(q*; θ) where q* is
        # a frozen constant.
        #
        # 'amortized' (default): outputs stay attached — straight-through
        # estimator where gradients flow through the E-step iteration graph
        # back to embeddings.  NOT proper EM, but enables amortized learning.
        #
        # 'E_phi_q': φ∈q — detach (μ, Σ, φ) at the boundary.
        # 'M_phi_p': φ∈θ — detach (μ, Σ); φ was already detached at E-step
        #            start and never evolved, so no additional detach needed.
        _em_active = self.em_phi_mode in ('E_phi_q', 'M_phi_p')
        if _em_active:
            mu_current = mu_current.detach()
            if sigma_current is not None:
                sigma_current = sigma_current.detach()
            if self.em_phi_mode == 'E_phi_q':
                phi_current = phi_current.detach()
            # Omega symmetry: when gauge_param='omega', the evolved omega carries
            # the E-step autograd graph (transport/β consumers) to the next layer
            # via _last_evolved_omega unless detached here. Under both EM modes
            # the gauge frame exits the E-step as a constant.
            if (getattr(self, 'gauge_param', 'phi') == 'omega'
                    and omega_current is not None):
                omega_current = omega_current.detach()

        if self.update_sigma:
            return mu_current, sigma_current, phi_current, beta_history
        else:
            return mu_current, None, phi_current, beta_history

    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, n_iterations={self.n_iterations}, "
            f"alpha={self.alpha}, lambda_belief={self.lambda_belief}, kappa={self.kappa}"
        )