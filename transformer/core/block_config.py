"""
BlockConfig: Single dataclass replacing the 40+ parameter waterfall.

Instead of:
    config dict → model.py (40 config.get calls) → Stack(40 kwargs) → Block(40 kwargs) → FFN(30 kwargs)

Now:
    config dict → BlockConfig.from_config(config) → Stack(cfg) → Block(cfg) → FFN(cfg)

All block-level parameters live here. Embedding-level parameters (learnable_reflection,
mu_normalize, mu_max_norm) are handled by model.py → GaugeTokenEmbedding directly.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List, Tuple, Literal


# Allowed values for rope_full_gauge:
#   'off'      — rotate neither μ nor Σ in attention/FFN; default RoPE on μ only is the
#                standard-transformer pattern preserved here.
#   'vfe_only' — apply the framework-consistent μ→Rμ AND Σ→RΣR^T transform inside the
#                FFN's per-head VFE gradient helper only. Attention still rotates μ only.
#   'both'     — additionally apply Σ→RΣR^T in the attention sublayer, lifting diagonal
#                σ to full covariance before KL dispatch. Strict GL(K) covariance through
#                both sublayers; substantially more expensive (O(K) memory blow-up).
RopeFullGaugeMode = Literal['off', 'vfe_only', 'both']
_ROPE_FULL_GAUGE_VALUES = ('off', 'vfe_only', 'both')


import torch
import torch.nn as nn

from transformer.core.em_modes import EM_MODE_TABLE

# Backward-compatible alias (external code may import the underscore-prefixed name)
_EM_MODE_TABLE = EM_MODE_TABLE


@dataclass
class BlockConfig:
    """All parameters needed to construct a GaugeTransformerBlock (and its sub-modules).

    Consumed by:
        - GaugeTransformerStack: uses n_layers
        - GaugeTransformerBlock: uses attention + FFN + non-flat transport fields
        - IrrepMultiHeadAttention: uses attention, gauge geometry, positional encoding fields
        - VariationalFFNDynamic: uses VFE dynamics, belief evolution, gauge geometry fields
    """

    # ╔═══════════════════════════════════════════════════════════════╗
    # ║  FIELD GROUPS (5 semantic domains, ~87 fields total)        ║
    # ║                                                             ║
    # ║  1. STRUCTURE      — dims, layers, irreps (~5 fields)       ║
    # ║  2. ATTENTION       — kappa, mask, heads, RoPE (~12 fields) ║
    # ║  3. E-STEP DYNAMICS — lr, alpha, lambda, iters (~20 fields) ║
    # ║  4. GAUGE GEOMETRY  — group, phi, omega modes (~10 fields)  ║
    # ║  5. COVARIANCE      — diag, sigma bounds (~8 fields)        ║
    # ║  + EM mode, non-flat transport, normalization, perf,        ║
    # ║    active inference, runtime objects                         ║
    # ╚═══════════════════════════════════════════════════════════════╝

    # === 1. STRUCTURE ===

    irrep_spec: List[Tuple[str, int, int]] = field(  # [(label, multiplicity, dim), ...]
        default_factory=lambda: [('ℓ0', 8, 1)]       #   e.g. [('ℓ0',75,1),('ℓ1',30,3),('ℓ2',18,5)]
    )
    
    embed_dim: int =  64                 # Total belief dimension K = Σ (mult_ℓ × dim_ℓ)
    hidden_dim: int = 256               # FFN hidden dimension (kept for config compat, not used by blocks)
    n_layers: int =   1                   # Number of stacked blocks (only used by Stack)

    # === 2. ATTENTION ===
    kappa_beta: float =          1.0             # Temperature τ for KL-based attention softmax
    learnable_head_kappa: bool = False  # If True, learn per-head κ_h via log_kappa_per_head
                                        # Initialized to kappa_beta (bare); compute_attention_weights
                                        # and gradient functions apply the √d_h scaling.
    attention_pattern: str =      'full'     # 'full' (only supported pattern)
    attention_window: int =       64          # Window size (unused, kept for API compat)
    
    mask_self_attention: bool =   False   # Prevent KL(q_i||q_i)=0 collapse
    # F4.10: opt-in causal-lower-triangle fast path in the fused E-step kernel
    # _fused_attention_and_vfe_gradients_block_diag. When True AND the caller
    # passes a strict lower-triangular causal mask, the kernel computes only
    # the lower-triangle pairs (j ≤ i, M = N(N+1)/2) inside the per-block loop
    # and scatters them into the dense (B,N,N,K) accumulators. Bit-identical
    # to the dense path for β, grad_μ, grad_σ; only kl_matrix's upper triangle
    # differs (real KL vs zero). Default False. Validated once per
    # VariationalFFNDynamic.forward() to avoid per-iteration host syncs.
    causal_lower_triangle: bool = False
    use_output_projection: bool = False  # W_O ∈ R^{K×K} after multi-head concat.
                                         # Default False: the PriorBank decode
                                         # expects a gauge-covariant belief, and
                                         # a generic learned K×K linear applied
                                         # to μ only (Σ unchanged) breaks that
                                         # invariance. See VFE_Transformer_Idea.md
                                         # §18.2. Set True only for ablation.
                                         
    use_equivariant_head_mixer: bool = False  # Opt-in principled replacement for
                                              # W_O: n² scalars forming the
                                              # commutant of the irrep decomp,
                                              # applied symmetrically to μ and Σ
                                              # (Σ → M·Σ·Mᵀ). Preserves gauge
                                              # equivariance by Schur's lemma.

    use_block_equivariant_mixer: bool = False  # Same Schur-commutant mixer math as
                                               # use_equivariant_head_mixer, but
                                               # applied post-FFN (inside
                                               # GaugeTransformerBlock.forward,
                                               # before the residual) rather than
                                               # post-attention-concat. Available
                                               # when skip_attention=True (the
                                               # attention-side mixer is bypassed
                                               # in that case). Mutually exclusive
                                               # with use_equivariant_head_mixer
                                               # and use_output_projection — both
                                               # are different insertion points
                                               # for the same K→K mixing
                                               # operation.
                                              
    # === 3. E-STEP DYNAMICS (belief evolution) ===
    E_learnable_lr: bool =     True    # Learn step size η for variational descent
    evolve_sigma: bool =       True    # Update covariances Σ via natural gradient
    evolve_phi: bool =         True    # Update gauge frames φ (M-step, after E-step loop)
    evolve_phi_e_step: bool =  True    # Update φ during EACH E-step iteration
    ffn_update_sigma: bool =   True    # Update covariances during E-step
    E_learnable_alpha: bool =  True    # Bayesian precision via Gamma-Normal conjugacy
    sigma_aggregation: str =   'mixture'  # 'mixture' (moment matching) or 'precision' (VFE equilibrium)
    
    
    phi_lr: float =            0.05                # Learning rate for ∂F/∂φ descent
    E_mu_q_lr: float =         0.1             # E-step μ natural gradient step size
    E_sigma_q_lr: float =      0.001        # E-step σ step size (decoupled from μ LR; drives the retraction directly)
    E_sigma_q_trust: float =   5.0          # E-step σ trust-region clamp on |δσ/σ| (default 5.0 = historical retract default)

              
    phi_max_norm: Optional[float] = None # Max phi norm; None = auto (π for SO(N), 5.0 for GL(K))
    omega_trust_region: float =  0.3     # Trust-region step clamp for direct Omega retraction (gauge_param='omega')
    # === Determinant control for GL(K) (off by default; both ignored for SO(N)) ===
    # GL(K) has an unbounded "trace" direction (the determinant) that L2-norm
    # clamping does not constrain — det(Ω_ij) = exp(tr(φ_i − φ_j)) can blow up
    # on outlier tokens even when ‖φ‖ is bounded. Enable ONE of:
    #   phi_project_slk=True : project φ onto sl(K) after each retraction so
    #                          tr(φ·G) ≡ 0 ⇒ det(Ω) ≡ 1 (geometric, principled).
    #   phi_trace_clamp=T    : clamp |tr(φ·G)| ≤ T per token after retraction
    #                          (soft cap; det(Ω_ij) ∈ [exp(-2T), exp(2T)]).
    # If both are set, phi_project_slk takes precedence.
    phi_project_slk: bool = False
    phi_trace_clamp: Optional[float] = None
    
    
    
    
    em_mode: Literal['ift_phi', 'em_phi_q', 'em_phi_p'] = 'ift_phi'
                                        # EM gradient-flow mode. Options:
                                        #   'ift_phi'  — μ_p,σ_p attached, full IFT φ gradient (default)
                                        #   'em_phi_q' — φ∈q: E-step optimizes μ,Σ,φ; all detached at EM boundary
                                        #   'em_phi_p' — φ∈θ: φ frozen in E-step; M-step only
    detach_phi: bool =          False   # Detach phi from backprop (Hebbian/P-flow only)

    # === 3b. E-STEP DYNAMICS (VFE weights and iterations) ===
    ffn_mode: str = 'VFE_dynamic'       # FFN mode (only 'VFE_dynamic' supported)
    
    ffn_kappa: float =         1.0              # Softmax temperature (unified with kappa_beta)
    ffn_n_iterations: int =    1           # VFE inference iterations per forward pass
    
    
    E_alpha: float =           1.0               # E-step prior self-coupling weight α
    E_lambda_belief: float =   1.0       # E-step belief alignment weight λ (direct: β·∇KL)
    E_lambda_softmax: float =  1.0      # E-step attention-variance coupling weight (∂β/∂θ · KL)
    include_attention_entropy: bool = True  # Add τ·Σβ·log(β/π) to alignment loss
                                        # (manuscript eq:free_energy_functional_final).
                                        # Required for softmax β to be a stationary
                                        # point of F; False reverts to the
                                        # entropy-suppressed surrogate Σβ·KL.
    alpha_divergence: float =  1.0      # Renyi alpha-divergence parameter for attention
                                        # 1.0 = KL divergence (default, backward-compatible)
                                        # 0.5 = Bhattacharyya distance (symmetric, bounded)
                                        # (0, 1) = mode-covering divergences
                                        # (1, 2] = tail-sensitive divergences
    
    
    isotropic_covariance: bool =     False  # Force Σ = σ²I (manuscript Limit 1)
    diagonal_covariance: bool =      False   # σ as (B,N,K) diagonal instead of (B,N,K,K) full   
    exact_diagonal_transport: bool = False  # When True + diagonal_covariance, lift σ to full
                                            # for exact Ω@diag(σ)@Ω^T transport (slower but exact)
   
    phi_dim: int =     3               # 3 for SO(3), N(N-1)/2 for SO(N), K² for GL(K)
    sigma_max: float = 5.0              # Upper bound on σ (diagonal) or eigenvalues (full cov).
                                        # Posterior σ should not exceed prior σ by much.
                                        # Default 5.0: with init_sigma_scale=1.0, allows 5× expansion
                                        # before clamping. Prevents nat_grad_sigma = 2σ²·∇σ blowup.
                                        # Matches VariationalFFNDynamic.__init__ default and the
                                        # from_config() dict default.
                                        
    # Gauge-covariant numerical ridge. When True, ε·I regularizers on
    # sandwich-transforming covariances Σ are replaced by ε·(gg^T) built from
    # the local frame g = exp(φ), preserving Σ → hΣh^T covariance exactly.
    # Default False preserves bitwise numerics. See transformer/core/gauge_ridge.py.
    gauge_covariant_ridge: bool = False
    e_step_sigma_floor: float = 0.1     # Floor on σ_p inside E-step (caps 1/σ_p gradient).
                                        # PriorBank allows σ_p ∈ [0.01, 5.0] for sharp decode,
                                        # but E-step needs a higher floor to prevent nat_grad blowup.
    propagate_kl_nonfinite: bool = False  # If True, safe_kl_clamp lets NaN/±inf propagate
                                           # through the softmax so divergence is visible.
                                           # Diagnostic use; default preserves masking.
    
    spd_floor_mode: str = 'eigclamp'   # How the full-cov σ_p floor is enforced:
                                        #   'eigclamp' — λ_min(Σ_p) clamped via eigendecomposition
                                        #                (bounds the condition number; the
                                        #                mathematically pure path under CLAUDE.md).
                                        #   'ridge'    — Σ_p + floor·I (legacy; shifts eigvals by
                                        #                +floor but does NOT bound cond(Σ), so
                                        #                Σ_p^{-1} can still amplify gradients under
                                        #                high off-diagonal correlation).
                                        #   'none'     — no floor (diagonal-only path or research).
                                        # Diagonal covariance always uses elementwise clamp(min=floor)
                                        # regardless of this flag — it is already per-dim bounded.
    
    e_step_early_exit_tol: Optional[float] = None  # Relative change threshold for E-step early exit.
                                        # When set, breaks the E-step loop if
                                        # ||Δμ||/||μ|| < tol between consecutive iterations.
                                        # None = disabled (fixed n_iterations). Typical: 1e-3.
                                        # More permissive than iterative E-step (2.0) since the
                                        # closed-form base point is already at the linear optimum.

    # === Performance ===
    gradient_checkpoint_vfe: bool = False  # Activation checkpointing for VFE iterations.
                                        # Trades ~2x compute for ~3x memory savings with 3 iters.

    # === 4. GAUGE GEOMETRY ===
    gauge_mode: str =          'learned'         # 'learned' | 'trivial' (Ω=I) | 'constant' (per-head Ω)
    gauge_param: str =         'phi'            # 'phi' (Lie algebra) | 'omega' (direct GL(K) matrices)
    enforce_orthogonal: bool = False    # Enforce Ω ∈ SO(K) via Newton-Schulz iteration
   
    phi_natural_gradient: str =           'killing'  # 'clip'|'cartan'|'killing'|'killing_per_block'|'pullback'
    killing_center_reg: Optional[float] = None  # Killing form center regularization (None=2K)
    

    # NOTE: learnable_reflection is an embedding-level feature, handled by
    # model.py → GaugeTokenEmbedding, not by blocks. Not stored here.

    # Derived in __post_init__ from _EM_MODE_TABLE[em_mode]; not user-settable.
    _em_phi_mode: str = field(default='', init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Enforce mode invariants that must hold regardless of how BlockConfig is constructed."""
        # EM mode validation and flag resolution
        if self.em_mode not in _EM_MODE_TABLE:
            raise ValueError(
                f"em_mode must be one of {list(_EM_MODE_TABLE.keys())}, got '{self.em_mode}'"
            )

        # FFN mode validation: the gauge VFE block only implements the
        # VFE_dynamic E-step. The 'standard'/'hybrid' values select a different
        # model class upstream and never reach a BlockConfig, so a non-VFE_dynamic
        # value here is a misconfiguration — fail loudly instead of silently
        # running VFE_dynamic regardless.
        if self.ffn_mode != 'VFE_dynamic':
            raise ValueError(
                f"ffn_mode must be 'VFE_dynamic' for the gauge VFE block, got '{self.ffn_mode}'"
            )

        _flags = _EM_MODE_TABLE[self.em_mode]
        # ``_amortized_inference`` / ``_amortize_sigma`` / ``_exact_phi_grad``
        # used to be cached on the dataclass here; removed 2026-05-17 because
        # ``VariationalFFNDynamic.__init__`` re-derives them directly from
        # ``_EM_MODE_TABLE[em_mode]`` and grep across the repo shows no other
        # consumer. ``_em_phi_mode`` is retained — it is read on the next line.
        self._em_phi_mode = _flags['em_phi_mode']

        # M_phi_p: phi is an M-step parameter — must not evolve during E-step
        if self._em_phi_mode == 'M_phi_p':
            self.evolve_phi = False
            self.evolve_phi_e_step = False

        # The separate attention sublayer was removed on 2026-06-01: the block is
        # always the pure-VFE form. skip_attention is retained as an (informational)
        # field but is effectively always True. A config that explicitly requests
        # skip_attention=False is asking for a behavior that no longer exists.
        if not self.skip_attention:
            import warnings as _warnings
            _warnings.warn(
                "skip_attention=False is no longer supported: the IrrepMultiHeadAttention "
                "sublayer was removed (2026-06-01) and the block is always the pure-VFE "
                "form (the FFN E-step computes its own β). Forcing skip_attention=True.",
                UserWarning,
                stacklevel=2,
            )
            self.skip_attention = True

        # Detaching EM modes (em_phi_p, em_phi_q) detach σ_p and/or φ inside the
        # FFN's E-step and historically relied on the attention sublayer as the
        # sole autograd path back to σ_embed and φ_embed. With the attention
        # sublayer gone, those embeddings receive no gradient under a detaching
        # mode and stay frozen at initialization — a silent failure (zero-variance
        # Σ across tokens, overlapping φ/Ω clusters). The clean mode is 'ift_phi',
        # where the FFN E-step itself provides the M-step gradients for σ and φ.
        _detaching_modes = {'em_phi_p', 'em_phi_q'}
        if self.em_mode in _detaching_modes:
            import warnings as _warnings
            _warnings.warn(
                f"em_mode={self.em_mode!r} will silently freeze σ_embed and φ_embed at "
                f"initialization: the FFN E-step detaches σ_p and/or φ, and the attention "
                f"sublayer that used to provide the only autograd path to those embeddings "
                f"has been removed. Use em_mode='ift_phi' for a working configuration.",
                UserWarning,
                stacklevel=2,
            )
        # use_equivariant_head_mixer lived inside IrrepMultiHeadAttention.forward,
        # which no longer exists. The flag is now always a no-op; the
        # post-FFN use_block_equivariant_mixer is the live Schur-commutant mixer.
        if self.use_equivariant_head_mixer:
            import warnings as _warnings
            _warnings.warn(
                "use_equivariant_head_mixer=True has no effect: the mixer lived inside the "
                "now-removed attention sublayer. Use use_block_equivariant_mixer=True for "
                "the post-FFN Schur-commutant mixer instead.",
                UserWarning,
                stacklevel=2,
            )
        # Constant/trivial gauge: phi is not used for transport
        if self.gauge_mode in ('constant', 'trivial'):
            self.evolve_phi = False
            self.evolve_phi_e_step = False
        # Isotropic covariance requires diagonal representation
        if self.isotropic_covariance and not self.diagonal_covariance:
            self.diagonal_covariance = True
        # rope_full_gauge under diagonal/isotropic is wasteful. The legacy
        # _compute_rope_full_gauge_gradient_per_head lifts diagonal σ to
        # (d_h, d_h) full covariance inside the rope rotation path — O(K²)
        # memory per token for no benefit in the isotropic case (R·(σ²I)·R^T
        # = σ²I is invariant) and no correctness gain over rope_full_gauge='off'
        # in the diagonal-non-isotropic case (the default path already handles
        # RoPE-on-μ with σ left raw, which is a deliberate design choice).
        # Warn loudly but permit (vfe/ is stricter — ValueError).
        if self.rope_full_gauge not in _ROPE_FULL_GAUGE_VALUES:
            raise ValueError(
                f"rope_full_gauge must be one of {_ROPE_FULL_GAUGE_VALUES}; "
                f"got {self.rope_full_gauge!r}."
            )
        if self.rope_full_gauge != 'off' and (self.diagonal_covariance or self.isotropic_covariance):
            import warnings as _warnings
            _which = "isotropic" if self.isotropic_covariance else "diagonal"
            _warnings.warn(
                f"rope_full_gauge={self.rope_full_gauge!r} under {_which} covariance: the rope "
                f"path will lift σ to full (B, N, K, K) and apply R Σ Rᵀ — "
                f"{'mathematically a no-op (R·(σ²I)·R^T = σ²I) with O(K²) cost' if self.isotropic_covariance else 'expensive vs. rope_full_gauge=off which leaves σ raw'}. "
                f"Consider rope_full_gauge='off'.",
                RuntimeWarning,
                stacklevel=3,
            )
        # Backward compat: use_layernorm=False with default norm_type → 'none'
        if not self.use_layernorm and self.norm_type == 'layernorm':
            self.norm_type = 'none'
        # use_output_projection and use_equivariant_head_mixer are mutually
        # exclusive: IrrepMultiHeadAttention.forward dispatches W_O over the
        # mixer when both are set (attention.py:2016-2022), leaving the
        # commutant mixer's mixer_params as dead state_dict weight. Fail fast
        # on the "allocated but silently ignored" principle.
        if self.use_output_projection and self.use_equivariant_head_mixer:
            raise ValueError(
                "use_output_projection=True and use_equivariant_head_mixer=True "
                "are mutually exclusive: IrrepMultiHeadAttention.forward dispatches "
                "to W_O when both are set, leaving the commutant mixer's "
                "mixer_params as dead state_dict weight. Choose one — "
                "use_output_projection for the legacy W_O linear mixer, or "
                "use_equivariant_head_mixer for the Schur-commutant principled "
                "replacement."
            )
        # use_block_equivariant_mixer is a different insertion point for the
        # same Schur-commutant operation as use_equivariant_head_mixer.
        # Setting both stacks the same mixing operation twice — almost
        # certainly user error. Fail fast.
        if self.use_block_equivariant_mixer and self.use_equivariant_head_mixer:
            raise ValueError(
                "use_block_equivariant_mixer=True and use_equivariant_head_mixer=True "
                "are mutually exclusive: both apply the same Schur-commutant mixer, "
                "differing only in location (post-FFN vs post-attention-concat). "
                "Choose one — use_block_equivariant_mixer when skip_attention=True "
                "or when you want the mixing to occur after the VFE E-step; "
                "use_equivariant_head_mixer when the attention sublayer is active "
                "and head-concat-side mixing is intended."
            )
        # use_output_projection is the gauge-breaking K→K linear that
        # use_block_equivariant_mixer is the principled replacement for —
        # setting both reproduces the silent-dead-weight pattern we explicitly
        # rejected above for the attention-side pair.
        if self.use_block_equivariant_mixer and self.use_output_projection:
            raise ValueError(
                "use_block_equivariant_mixer=True and use_output_projection=True "
                "are mutually exclusive: W_O breaks gauge equivariance "
                "(K→K linear on μ only, ignores Σ and the gauge action) and the "
                "block mixer is the principled commutant replacement. Choose one."
            )

        # Standalone use_output_projection=True (without the mixers above):
        # W_O ∈ R^{K×K} is applied to μ_agg only inside the attention sublayer,
        # leaving Σ untransformed. The (μ, Σ) pair then sits in two different
        # frames and the downstream KL becomes ill-defined under any gauge
        # action. The mixer flags above are the equivariant replacements;
        # warn when the user opted for the bare linear without setting either.
        if self.use_output_projection and not (
            self.use_equivariant_head_mixer or self.use_block_equivariant_mixer
        ):
            import warnings as _warnings
            _warnings.warn(
                "use_output_projection=True applies W_O ∈ R^{K×K} to μ only; "
                "Σ is not transformed, which breaks the gauge sandwich "
                "invariant Σ_t = Ω Σ Ω^T. Prefer use_equivariant_head_mixer "
                "(post-concat) or use_block_equivariant_mixer (post-FFN) — "
                "both are Schur-commutant and preserve equivariance.",
                category=RuntimeWarning,
                stacklevel=2,
            )

    # The old 5-flag system (amortized_inference / amortize_sigma /
    # exact_phi_grad / em_phi_mode / implicit_em) is fully internalized.
    # External callers should configure via ``em_mode`` only; the per-mode
    # gradient flags are derived inside ``__post_init__`` via
    # ``em_modes.EM_MODE_TABLE`` and stored as private ``_amortized_inference``
    # etc. attributes consumed by ``VariationalFFNDynamic.__init__``.

    # === Non-flat gauge transport (flat bundle experiments) ===
    # When non_flat_transport=True, the transport becomes:
    #   phi path:  Ω_ij = exp(φ_i·G) · exp(α·δ_ij·G) · exp(-φ_j·G)
    #   omega path: Ω_ij = Ω_i · exp(α·δ_ij·G) · Ω_j⁻¹
    # where δ_ij is an edge-local "connection" parameterized by connection_type.
    # When non_flat_transport=False (default), all non-flat fields are ignored
    # and the standard flat transport is used (cocycle condition holds, holonomy trivial).
    non_flat_transport: bool = False       # Enable edge-dependent connection δ_ij
    cocycle_relaxation: float = 0.5        # Scale factor for δ_ij: 0=flat, 1=fully non-flat
    connection_type: str = 'bilinear'      # 'bilinear' | 'mlp'
    connection_hidden_dim: int = 64        # Hidden dim for MLP connection
    connection_init_scale: float = 0.01    # W init scale (0=flat saddle, >0 breaks symmetry)
    holonomy_penalty: float = 0.0          # λ_H · E[‖H_ijk - I‖²_F] added to loss

    # === Positional encoding ===
    alibi_slope: Optional[float] = None    # ALiBi positional bias (negative = recency)
    use_rope: bool = True                 # Rotary position embeddings on μ before KL
    rope_base: float = 10000.0            # RoPE frequency base (GPT-style default).
                                          # Low values (e.g. 50) keep all rotation pairs active
                                          # for short sequences (B*≈13 for K=20, N=64).

    # === Memory efficiency ===
    ffn_irrep_dims: Optional[List[int]] = None  # Block dims for block-diagonal KL decomposition

    # === Pure VFE mode flags ===
    # Note on LayerNorm: the 2026-04-08 deep audit (PR-1 C1) flagged the
    # learned-affine LayerNorm as a possible constraint violation relative
    # to the "NO NEURAL NETWORKS" statement in CLAUDE.md. User judgment
    # overrode the audit finding: the stability contribution from LayerNorm
    # on the belief-mean channel is meaningful in practice and the learned
    # affine is considered part of the acknowledged design. Default stays
    # True; see edits_2026-04-08.md revert note.
    use_layernorm: bool = True         # LayerNorm on means (False for pure VFE ablation). Legacy; prefer norm_type.
    norm_type: str = 'layernorm'       # 'layernorm' | 'rmsnorm' | 'mahalnorm' | 'none'
    use_residual: bool = True          # Residual connections (False for pure VFE ablation)
    residual_type: str = 'additive'    # 'additive': mu_q = mu_q + mu_sub (matches the 71-PPL
                                        # baseline in TransformerOld/).  Default.
                                        # 'delta':    mu_q = mu_q + (mu_sub - mu_normalized),
                                        # the 2026-04-07 audit Fix #1 / Fix #20 form.  Correct
                                        # for deep unnormalised stacks but empirically worse
                                        # for single-layer LayerNorm'd configs (see
                                        # edits_2026-04-08.md Round 3).
    skip_attention: bool = True        # Vestigial (attention sublayer removed 2026-06-01);
                                        # the block is always the pure-VFE form. Kept for
                                        # config/diagnostic compatibility. False is forced to
                                        # True with a warning in __post_init__.

    # === Memory efficiency ===
    gradient_checkpointing: bool = False  # Checkpoint non-final layers (~60% memory savings, ~30% extra compute)

    # === Multi-layer depth signal ===
                                        # When False (default), all layers share the embedding prior.

    rope_full_gauge: RopeFullGaugeMode = 'off'  # EXPERIMENTAL tri-state {'off', 'vfe_only', 'both'}.
                                        # When != 'off' (and use_rope=True), implements the
                                        # framework-consistent interpretation of RoPE as a
                                        # position-dependent gauge transport that acts on
                                        # Gaussian beliefs by both μ → Rμ AND Σ → RΣR^T (the
                                        # standard sandwich product), as derived in the GL(K)
                                        # manuscript Section "RoPE as Position-Dependent Gauge
                                        # Frames".  The default ('off') follows the standard-
                                        # transformer Q/K rotation pattern: only μ is rope-
                                        # rotated, Σ stays raw.  Both implementations are
                                        # valid choices in the framework (see manuscript
                                        # ~line 1760 on attention-vs-value gauge factorization),
                                        # but they produce different β values and gradients.
                                        # 'vfe_only' and 'both' are SLOWER (lifts diagonal Σ
                                        # to full covariance and uses autograd for gradients)
                                        # and are intended for empirical comparison only.

    # === Non-serializable objects (set after construction) ===
    # These are torch tensors / nn.Modules that can't be part of a plain dataclass default.
    # Passed via from_config() or set directly after construction.
    generators: Optional[torch.Tensor] = field(default=None, repr=False)   # (n_gen, K, K) Lie algebra generators
    ffn_prior_bank: Optional[nn.Module] = field(default=None, repr=False)  # Token-dependent PriorBank module
    ffn_use_prior_bank: bool = False   # If True, FFN uses PriorBank for token-dependent priors

    @classmethod
    def from_config(cls, config: Dict[str, Any], generators: Optional[torch.Tensor] = None,
                    prior_bank: Optional[nn.Module] = None,
                    ffn_irrep_dims: Optional[List[int]] = None) -> 'BlockConfig':
        """Build BlockConfig from the flat config dict used by train_publication.py.

        Args:
            config: Flat dict with all hyperparameters. Required keys: embed_dim,
                    irrep_spec, hidden_dim, n_layers, kappa_beta. All others optional.
            generators: (n_gen, K, K) Lie algebra generators for gauge transport.
            prior_bank: Optional PriorBank nn.Module for token-dependent priors.
            ffn_irrep_dims: Optional flat list of block dimensions [d₁, d₂, ...].

        Note:
            Cross-head coupling permutations live on the LM module
            (``GaugeTransformerLM._cross_head_perm``) and are applied inside
            ``_apply_cross_head_perm`` before/after the stack call. They are
            no longer threaded through the BlockConfig dataclass.
        """
        kappa_beta = config['kappa_beta']
        return cls(
            # Structural
            embed_dim=config['embed_dim'],
            irrep_spec=config['irrep_spec'],
            hidden_dim=config.get('hidden_dim', 256),
            n_layers=config['n_layers'],
            # Attention
            kappa_beta=kappa_beta,
            learnable_head_kappa=config.get('learnable_head_kappa', False),
            attention_pattern=config.get('attention_pattern', 'full'),
            attention_window=config.get('attention_window', 64),
            # Default matches the dataclass declaration above (True).
            # Rationale: KL(q_i||q_i)=0 makes self-attention the maximum of
            # the softmax by construction, and without masking the diagonal
            # the attention sublayer at init aggregates ≈ mu_normalized[i]
            # into the residual stream each layer.  Training configs that
            # want the unmasked behavior should set this explicitly.
            mask_self_attention=config.get('mask_self_attention', True),
            # F4.10: opt-in causal-lower-triangle fast path in the fused
            # E-step kernel; default False preserves the dense behavior.
            causal_lower_triangle=config.get('causal_lower_triangle', False),
            use_output_projection=config.get('use_output_projection', False),
            use_equivariant_head_mixer=config.get('use_equivariant_head_mixer', False),
            use_block_equivariant_mixer=config.get('use_block_equivariant_mixer', False),
            # Belief evolution
            evolve_sigma=config.get('evolve_sigma', True),
            evolve_phi=config.get('evolve_phi', True),
            # Default matches the dataclass declaration (True).  Production
            # training configs set this explicitly to True, so the previous
            # from_config default of False was silently opposite the dataclass
            # default and contradicted the production behavior.
            evolve_phi_e_step=config.get('evolve_phi_e_step', True),
            phi_lr=config.get('E_phi_lr', config.get('e_step_phi_lr', config.get('phi_lr', 0.05))),
            phi_max_norm=config.get('phi_max_norm', None),
            omega_trust_region=config.get('omega_trust_region', 0.3),
            phi_project_slk=config.get('phi_project_slk', False),
            phi_trace_clamp=config.get('phi_trace_clamp', None),
            phi_dim=config.get('phi_dim', 3),
            phi_natural_gradient=config.get('phi_natural_gradient', 'killing'),
            killing_center_reg=config.get('killing_center_reg', None),
            diagonal_covariance=config.get('diagonal_covariance', False),
            exact_diagonal_transport=config.get('exact_diagonal_transport', False),
            sigma_aggregation=config.get('sigma_aggregation', 'mixture'),
            em_mode=config.get('em_mode', 'ift_phi'),
            detach_phi=config.get('detach_phi', False),
            # VFE dynamics (E-step) — new names with old-name fallbacks
            ffn_mode=config.get('ffn_mode', 'VFE_dynamic'),
            E_alpha=config.get('E_alpha', config.get('ffn_alpha', 1.0)),
            ffn_kappa=kappa_beta,  # Unified temperature
            ffn_n_iterations=config.get('ffn_n_iterations', 1),
            E_learnable_lr=config.get('E_learnable_lr', config.get('ffn_learnable_lr', True)),
            E_mu_q_lr=config.get('E_mu_q_lr', config.get('e_step_mu_lr', 0.1)),
            E_sigma_q_lr=config.get('E_sigma_q_lr', config.get('e_step_sigma_lr', 0.001)),
            E_sigma_q_trust=config.get('E_sigma_q_trust', config.get('e_step_sigma_trust', 5.0)),
            E_lambda_belief=config.get('E_lambda_belief', config.get('ffn_lambda_belief', 1.0)),
            E_lambda_softmax=config.get('E_lambda_softmax', config.get('ffn_lambda_softmax', 1.0)),
            include_attention_entropy=config.get('include_attention_entropy', True),
            ffn_update_sigma=config.get('ffn_update_sigma', True),
            # Default matches the dataclass declaration (True) and the
            # production training configs.  Previous default of False was
            # a silent boolean reversal relative to the dataclass: direct
            # BlockConfig() construction enabled Bayesian precision, but
            # BlockConfig.from_config(config_dict_without_key) disabled it.
            E_learnable_alpha=config.get('E_learnable_alpha', config.get('ffn_learnable_alpha', config.get('learnable_alpha', True))),
            alpha_divergence=config.get('alpha_divergence', 1.0),
            sigma_max=config.get('sigma_max', 5.0),
            gauge_covariant_ridge=config.get('gauge_covariant_ridge', False),
            e_step_sigma_floor=config.get('e_step_sigma_floor', 0.1),
            spd_floor_mode=config.get('spd_floor_mode', 'eigclamp'),
            propagate_kl_nonfinite=config.get('propagate_kl_nonfinite', False),
            e_step_early_exit_tol=config.get('e_step_early_exit_tol', None),
            # Performance
            gradient_checkpoint_vfe=config.get('gradient_checkpoint_vfe', False),
            # Gauge geometry
            gauge_mode=config.get('gauge_mode', 'learned'),
            gauge_param=config.get('gauge_param', 'phi'),
            enforce_orthogonal=config.get('enforce_orthogonal', False),
            isotropic_covariance=config.get('isotropic_covariance', False),
            # Non-flat gauge transport
            non_flat_transport=config.get('non_flat_transport', False),
            cocycle_relaxation=config.get('cocycle_relaxation', 0.5),
            connection_type=config.get('connection_type', 'bilinear'),
            connection_hidden_dim=config.get('connection_hidden_dim', 64),
            connection_init_scale=config.get('connection_init_scale', 0.01),
            holonomy_penalty=config.get('holonomy_penalty', 0.0),
            # Positional encoding
            alibi_slope=config.get('alibi_slope', None),
            use_rope=config.get('use_rope', True),
            rope_base=config.get('rope_base', 10000.0),
            # Memory efficiency
            ffn_irrep_dims=ffn_irrep_dims,
            # Pure VFE mode. Defaults match the dataclass declaration (LayerNorm
            # enabled). See 2026-04-08 PR-1 C1 revert note for the rationale.
            use_layernorm=config.get('use_layernorm', True),
            norm_type=config.get('norm_type', 'layernorm' if config.get('use_layernorm', True) else 'none'),
            use_residual=config.get('use_residual', True),
            residual_type=config.get('residual_type', 'additive'),
            skip_attention=config.get('skip_attention', True),
            # Memory efficiency
            gradient_checkpointing=config.get('gradient_checkpointing', False),
            rope_full_gauge=config.get('rope_full_gauge', 'off'),
            # Non-serializable
            generators=generators,
            ffn_prior_bank=prior_bank,
            ffn_use_prior_bank=config.get('use_prior_bank', False),
        )
