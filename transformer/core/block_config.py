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
from typing import Optional, List, Tuple
import math
import logging
import warnings

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _warn_sigma_residual_deprecated(value: bool) -> bool:
    """Emit a deprecation warning if the user explicitly set sigma_residual.

    sigma_residual was the flag controlling additive σ accumulation across
    sublayers.  The additive path double-counted absolute σ values and
    pegged σ at sigma_max within the first forward pass for multi-layer
    configs.  Both sublayers now use delta extraction (replacement)
    regardless of this flag.  Users who explicitly set sigma_residual=True
    are likely expecting the old behavior — warn loudly so the change is
    visible.
    """
    if value:
        msg = (
            "sigma_residual=True is DEPRECATED and is now a no-op.  The old "
            "additive σ update (sigma_q + sigma_sub) has been replaced with "
            "delta extraction (sigma_q = sigma_sub).  If you relied on the "
            "additive behavior for an empirical result, those results cannot "
            "be reproduced with the current code.  Remove sigma_residual from "
            "your config to silence this warning."
        )
        warnings.warn(msg, DeprecationWarning, stacklevel=3)
        logger.warning("[BlockConfig] " + msg)
    return value


@dataclass
class BlockConfig:
    """All parameters needed to construct a GaugeTransformerBlock (and its sub-modules).

    Consumed by:
        - GaugeTransformerStack: uses n_layers
        - GaugeTransformerBlock: uses attention + FFN + non-flat transport fields
        - IrrepMultiHeadAttention: uses attention, gauge geometry, positional encoding fields
        - VariationalFFNDynamic: uses VFE dynamics, belief evolution, gauge geometry fields
    """

    # === Structural ===
    
    irrep_spec: List[Tuple[str, int, int]] = field(  # [(label, multiplicity, dim), ...]
        default_factory=lambda: [('ℓ0', 8, 1)]       #   e.g. [('ℓ0',75,1),('ℓ1',30,3),('ℓ2',18,5)]
    )
    
    embed_dim: int =  64                 # Total belief dimension K = Σ (mult_ℓ × dim_ℓ)
    hidden_dim: int = 256               # FFN hidden dimension (kept for config compat, not used by blocks)
    n_layers: int =   1                   # Number of stacked blocks (only used by Stack)

    # === Attention ===
    kappa_beta: float =          1.0             # Temperature τ for KL-based attention softmax
    learnable_head_kappa: bool = False  # If True, learn per-head κ_h via log_kappa_per_head
                                        # Initialized to kappa_beta (bare); compute_attention_weights
                                        # and gradient functions apply the √d_h scaling.
    attention_pattern: str =      'full'     # 'full' (only supported pattern)
    attention_window: int =       64          # Window size (unused, kept for API compat)
    
    mask_self_attention: bool =   True   # Prevent KL(q_i||q_i)=0 collapse
    use_output_projection: bool = True # W_O ∈ R^{K×K} after multi-head concat
    multihead_vfe: bool =         True         # Per-head β_h through VFE iterations

    # === Belief evolution ===
    E_learnable_lr: bool =     True        # Learn step size η for variational descent
    evolve_sigma: bool =       True           # Update covariances Σ via natural gradient
    evolve_phi: bool =         True             # Update gauge frames φ (M-step, after E-step loop)
    evolve_phi_e_step: bool =  True    # Update φ during EACH E-step iteration
    ffn_update_sigma: bool =   True       # Update covariances during E-step
    E_learnable_alpha: bool =  True    # Bayesian precision via Gamma-Normal conjugacy
    obs_sigma_gradient: bool = True    # ∂E_q[CE]/∂σ Hessian-diagonal obs gradient for sigma
    sigma_aggregation: str =   'mixture'  # 'mixture' (moment matching) or 'precision' (VFE equilibrium)
    
    
    phi_lr: float =            0.05                # Learning rate for ∂F/∂φ descent
    E_mu_q_lr: float =         0.1             # E-step μ natural gradient step size
    E_sigma_q_lr: float =      0.001        # E-step σ trust region scale

              
    phi_max_norm: Optional[float] = None # Max phi norm; None = auto (π for SO(N), 5.0 for GL(K))
    
    
    
    
    amortized_inference: bool = True    # Gradient flow through priors for learned E-step init
    detach_phi: bool =          False   # Detach phi from backprop in non-amortized mode
    
                                        # (enables fully backprop-free training with phi P-flow)
    implicit_em: bool =         False   # IFT-based M-step: detach beliefs at E-step start,
                                        # apply info-geometric scale s_k = (α/σ²_p)/A_k

    # === VFE dynamics (E-step) ===
    ffn_mode: str = 'VFE_dynamic'       # FFN mode (only 'VFE_dynamic' supported)
    
    ffn_kappa: float =         1.0              # Softmax temperature (unified with kappa_beta)
    ffn_n_iterations: int =    1           # VFE inference iterations per forward pass
    
    
    E_alpha: float =           1.0               # E-step prior self-coupling weight α
    E_lambda_belief: float =   1.0       # E-step belief alignment weight λ (direct: β·∇KL)
    E_lambda_softmax: float =  1.0      # E-step softmax coupling weight (GELU-like ∂β/∂θ · KL)
    obs_sigma_weight: float =  1.0      # Weight for sigma observation gradient
    
    
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
    e_step_sigma_floor: float = 0.1    # Floor on σ_p inside E-step (caps 1/σ_p gradient).
                                        # PriorBank allows σ_p ∈ [0.01, 5.0] for sharp decode,
                                        # but E-step needs a higher floor to prevent nat_grad blowup.
    n_picard_steps: int = 0            # Picard corrections after closed-form E-step.
                                        # Preconditioned Picard iteration on the nonlinear VFE
                                        # residual: mu^(n+1) = mu_0 - sigma_0 * ∇F_softmax(mu^(n))
                                        # where mu_0, sigma_0 are the closed-form linear fixed point.
                                        # 0 = pure closed-form (no softmax coupling).
                                        # 1-3 = recommended range. Requires closed_form_e_step=True.
    picard_trust_region: float = 5.0   # Whitened trust region for Picard corrections.
                                        # More permissive than iterative E-step (2.0) since the
                                        # closed-form base point is already at the linear optimum.

    # === Performance ===
    compile_vfe: bool = False           # torch.compile the VFE iteration inner loop.
                                        # Fuses element-wise ops, reduces kernel launch overhead.
                                        # Adds compilation latency on first forward pass.
    gradient_checkpoint_vfe: bool = False  # Activation checkpointing for VFE iterations.
                                        # Trades ~2x compute for ~3x memory savings with 3 iters.

    # === Gauge geometry ===
    gauge_mode: str =          'learned'         # 'learned' | 'trivial' (Ω=I) | 'constant' (per-head Ω)
    gauge_param: str =         'phi'            # 'phi' (Lie algebra) | 'omega' (direct GL(K) matrices)
    enforce_orthogonal: bool = False    # Enforce Ω ∈ SO(K) via Newton-Schulz iteration
   
    phi_natural_gradient: str =           'killing'  # 'clip'|'cartan'|'killing'|'pullback' 
    killing_center_reg: Optional[float] = None  # Killing form center regularization (None=2K)
    
   
    # NOTE: learnable_reflection is an embedding-level feature, handled by
    # model.py → GaugeTokenEmbedding, not by blocks. Not stored here.

    def __post_init__(self):
        """Enforce mode invariants that must hold regardless of how BlockConfig is constructed."""
        # Constant/trivial gauge: phi is not used for transport
        if self.gauge_mode in ('constant', 'trivial'):
            self.evolve_phi = False
            self.evolve_phi_e_step = False
        # Isotropic covariance requires diagonal representation
        if self.isotropic_covariance and not self.diagonal_covariance:
            self.diagonal_covariance = True
        # Backward compat: use_layernorm=False with default norm_type → 'none'
        if not self.use_layernorm and self.norm_type == 'layernorm':
            self.norm_type = 'none'
        # Picard corrections are applied AFTER the closed-form fixed point
        # (see variational_ffn._closed_form_e_step line 2252).  Setting
        # n_picard_steps > 0 without closed_form_e_step=True is a silent
        # no-op — the Picard branch is never entered.  Fail fast so the
        # misconfiguration is visible at model construction rather than
        # producing the same loss whether n_picard_steps=0 or 3.
        if self.n_picard_steps > 0 and not self.closed_form_e_step:
            raise ValueError(
                f"n_picard_steps={self.n_picard_steps} requires "
                f"closed_form_e_step=True (Picard corrections are applied "
                f"after the closed-form fixed point; without closed-form, "
                f"this field is silently ignored). Either set "
                f"closed_form_e_step=True or set n_picard_steps=0."
            )

    # === Non-flat gauge transport (flat bundle experiments) ===
    # When non_flat_transport=True, the transport becomes:
    #   phi path:  Ω_ij = exp(φ_i·G) · exp(α·δ_ij·G) · exp(-φ_j·G)
    #   omega path: Ω_ij = Ω_i · exp(α·δ_ij·G) · Ω_j⁻¹
    # where δ_ij is an edge-local "connection" parameterized by connection_type.
    # When non_flat_transport=False (default), all non-flat fields are ignored
    # and the standard flat transport is used (cocycle condition holds, holonomy trivial).
    non_flat_transport: bool = False       # Enable edge-dependent connection δ_ij
    cocycle_relaxation: float = 0.5        # Scale factor for δ_ij: 0=flat, 1=fully non-flat
    per_head_flatness_gate: bool = False   # Learnable per-head g_h ∈ [0,1]
    connection_type: str = 'bilinear'      # 'bilinear' | 'mlp'
    connection_hidden_dim: int = 64        # Hidden dim for MLP connection
    connection_init_scale: float = 0.01    # W init scale (0=flat saddle, >0 breaks symmetry)
    holonomy_penalty: float = 0.0          # λ_H · E[‖H_ijk - I‖²_F] added to loss

    # === Positional encoding ===
    alibi_slope: Optional[float] = None    # ALiBi positional bias (negative = recency)
    use_rope: bool = True                 # Rotary position embeddings on μ before KL
    rope_base: float = 10000.0            # RoPE frequency base (matches GPT-style standard
                                          # and the from_config() dict default of 10000.0)

    # === Memory efficiency ===
    ffn_irrep_dims: Optional[List[int]] = None  # Block dims for block-diagonal KL decomposition

    # === DEQ (Deep Equilibrium) ===
    use_deq: bool = False              # Use implicit differentiation for E-step fixed point
    deq_neumann_terms: int = 5         # Neumann series terms for DEQ backward pass
    deq_include_phi: bool = False      # Include phi in DEQ fixed-point (joint mu, sigma, phi IFT)

    # === Pure VFE mode flags ===
    use_layernorm: bool = True         # LayerNorm on means (False for pure VFE ablation). Legacy; prefer norm_type.
    norm_type: str = 'layernorm'       # 'layernorm' | 'rmsnorm' | 'none'
    use_residual: bool = True          # Residual connections (False for pure VFE ablation)
    skip_attention: bool = False       # Skip attention sublayer; VFE E-step computes its own β
    closed_form_e_step: bool = False   # Use closed-form precision-weighted fixed point instead of gradient descent

    # === Memory efficiency ===
    gradient_checkpointing: bool = False  # Checkpoint non-final layers (~60% memory savings, ~30% extra compute)

    # === Multi-layer depth signal ===
    aux_layer_loss: bool = False       # Per-layer auxiliary CE loss (M-step task signal for non-final layers)
    aux_loss_weight: float = 0.3       # Weight for auxiliary per-layer CE losses
    sigma_residual: bool = False       # DEPRECATED no-op.  Was: additive σ residual.
                                        # The old additive path (sigma_q + sigma_attn / sigma_q +
                                        # sigma_ffn) double-counted absolute σ values across
                                        # sublayers and pegged σ at sigma_max within the first
                                        # forward pass for multi-layer configs.  Both sublayers
                                        # now use delta extraction (= replacement) regardless of
                                        # this flag, since σ has no normalization step (unlike μ)
                                        # so delta = σ_out - σ_in and σ_in + delta = σ_out.
                                        # Retained only for config-file backward compat.
    hierarchical_priors: bool = True   # Each layer's posterior μ becomes the next layer's prior μ
                                        # (sigma_prior stays at embedding value to prevent cascade).
                                        # When False (default), all layers share the embedding prior.
    # === Active inference / Expected Free Energy (E-step extension) ===
    # When pragmatic_weight > 0, the E-step adds a term −H[p_pred(v|μ_i)] to F:
    # this is the pragmatic component of EFE under self-observation, computed
    # via PriorBank.decode (KL-based readout).  It pushes beliefs toward
    # confident predictions without target leak.
    #
    # When epistemic_weight > 0, the E-step adds the BALD-style mutual
    # information term I(v; μ | q_i) ≈ H[E_q p(v|μ)] − E_q[H[p(v|μ)]], computed
    # by Monte Carlo sampling S values from N(μ_i, Σ_i).  This is the
    # canonical epistemic value in active inference: it rewards beliefs whose
    # predictive distribution depends meaningfully on the parameter, and
    # counter-balances the pragmatic term's self-reinforcement tendency.
    #
    # Both terms require the FFN to have access to the PriorBank instance.
    # The reference is plumbed in model.__init__ via __dict__ assignment to
    # avoid nn.Module sub-module registration of an already-owned module.
    #
    # Master toggle: when False (default), the entire EFE path is bypassed
    # regardless of weight values.  When True, the pragmatic and epistemic
    # weights below take effect.  Matches the project convention used by
    # non_flat_transport, rope_full_gauge, and hierarchical_priors.
                      # Master EFE on/off toggle
    active_inference_pragmatic_weight: float = 1.0   # λ_prag · H[p(v|μ)]
    active_inference_epistemic_weight: float = 0.5   # −λ_epi · MI(v; μ | q)
    active_inference_epistemic_samples: int =  4     # MC samples for BALD MI
    active_inference_decode_tau: float =       1.0   # Temperature for PriorBank.decode
    active_inference_trust_region: float =     0.5   # Whitened trust region for the
                                                     # separate EFE μ-update (applied
                                                     # AFTER the VFE trust region so EFE
                                                     # contribution is not diluted when
                                                     # VFE saturates its own budget).
    active_inference_lr: float =               1.0   # Step size for the Euclidean EFE
                                                     # μ-update.  Separate from E_mu_q_lr
                                                     # (typically 0.1) because entropy
                                                     # and MI gradients are much smaller
                                                     # than KL coupling gradients — a
                                                     # larger lr is needed to make EFE
                                                     # move μ meaningfully.

    # === Bootstrap self-distillation (BYOL-style E-step term) ===
    # Third active-inference term that does not collapse at initialization
    # and does not need an epistemic counterweight.  Adds a cross-entropy
    # loss against a stop-gradient target constructed from attention-weighted
    # transported neighbour readouts:
    #     L_distill,i = CE( sg[p_pred(μ̃_i)],  p_pred(μ_i) )
    # where μ̃_i = Σ_j sg[β_ij] · sg[Ω_ij μ_j] is the attention-aggregated
    # transported belief.  BOTH stop-gradients are essential: sg on the
    # target prevents BYOL-style collapse, sg on β prevents the separate
    # "attend-to-twins" collapse (SymPy-verified in docs/).
    # See docs/bootstrap_self_distillation.md for the full derivation.
    # Gated by weight > 0 (master toggle active_inference does NOT gate
    # this — distillation is a standalone term that can be enabled alone).
    active_inference_distill_weight: float =   0.0   # λ_distill · normalized CE
    active_inference_distill_lr: float =       1.0   # Euclidean step size
    active_inference_distill_normalize: bool = True  # Divide CE by log(V)
    active_inference_distill_mode: str = 'aggregated'  # 'aggregated' | 'per_pair'
    
    active_inference: bool = False 
    
    rope_full_gauge: bool =  False      # EXPERIMENTAL.  When True (and use_rope=True),
                                        # implements the framework-consistent interpretation
                                        # of RoPE as a position-dependent gauge transport that
                                        # acts on Gaussian beliefs by both μ → Rμ AND Σ → RΣR^T
                                        # (the standard sandwich product), as derived in the
                                        # GL(K) manuscript Section "RoPE as Position-Dependent
                                        # Gauge Frames".  The default (False) follows the
                                        # standard-transformer Q/K rotation pattern: only μ
                                        # is rope-rotated, Σ stays raw.  Both implementations
                                        # are valid choices in the framework (see manuscript
                                        # ~line 1760 on attention-vs-value gauge factorization),
                                        # but they produce different β values and gradients.
                                        # rope_full_gauge=True is SLOWER (lifts diagonal Σ to
                                        # full covariance and uses autograd for gradients) and
                                        # is intended for empirical comparison only.

    # === Non-serializable objects (set after construction) ===
    # These are torch tensors / nn.Modules that can't be part of a plain dataclass default.
    # Passed via from_config() or set directly after construction.
    generators: Optional[torch.Tensor] = field(default=None, repr=False)   # (n_gen, K, K) Lie algebra generators
    ffn_prior_bank: Optional[nn.Module] = field(default=None, repr=False)  # Token-dependent PriorBank module
    ffn_use_prior_bank: bool = False   # If True, FFN uses PriorBank for token-dependent priors
    cross_head_perm: Optional[object] = field(default=None, repr=False)    # Cross-head coupling permutation

    @classmethod
    def from_config(cls, config: dict, generators: Optional[torch.Tensor] = None,
                    prior_bank: Optional[nn.Module] = None,
                    cross_head_perm=None,
                    ffn_irrep_dims: Optional[List[int]] = None) -> 'BlockConfig':
        """Build BlockConfig from the flat config dict used by train_publication.py.

        Args:
            config: Flat dict with all hyperparameters. Required keys: embed_dim,
                    irrep_spec, hidden_dim, n_layers, kappa_beta. All others optional.
            generators: (n_gen, K, K) Lie algebra generators for gauge transport.
            prior_bank: Optional PriorBank nn.Module for token-dependent priors.
            cross_head_perm: Optional cross-head coupling permutation.
            ffn_irrep_dims: Optional flat list of block dimensions [d₁, d₂, ...].
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
            use_output_projection=config.get('use_output_projection', True),
            multihead_vfe=config.get('multihead_vfe', True),
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
            phi_dim=config.get('phi_dim', 3),
            phi_natural_gradient=config.get('phi_natural_gradient', 'killing'),
            killing_center_reg=config.get('killing_center_reg', None),
            diagonal_covariance=config.get('diagonal_covariance', False),
            exact_diagonal_transport=config.get('exact_diagonal_transport', False),
            sigma_aggregation=config.get('sigma_aggregation', 'mixture'),
            amortized_inference=config.get('amortized_inference', True),
            detach_phi=config.get('detach_phi', False),
            implicit_em=config.get('implicit_em', False),
            # VFE dynamics (E-step) — new names with old-name fallbacks
            ffn_mode=config.get('ffn_mode', 'VFE_dynamic'),
            E_alpha=config.get('E_alpha', config.get('ffn_alpha', 1.0)),
            ffn_kappa=kappa_beta,  # Unified temperature
            ffn_n_iterations=config.get('ffn_n_iterations', 1),
            E_learnable_lr=config.get('E_learnable_lr', config.get('ffn_learnable_lr', True)),
            E_mu_q_lr=config.get('E_mu_q_lr', config.get('e_step_mu_lr', 0.1)),
            E_sigma_q_lr=config.get('E_sigma_q_lr', config.get('e_step_sigma_lr', 0.001)),
            E_lambda_belief=config.get('E_lambda_belief', config.get('ffn_lambda_belief', 1.0)),
            E_lambda_softmax=config.get('E_lambda_softmax', config.get('ffn_lambda_softmax', 1.0)),
            ffn_update_sigma=config.get('ffn_update_sigma', True),
            # Default matches the dataclass declaration (True) and the
            # production training configs.  Previous default of False was
            # a silent boolean reversal relative to the dataclass: direct
            # BlockConfig() construction enabled Bayesian precision, but
            # BlockConfig.from_config(config_dict_without_key) disabled it.
            E_learnable_alpha=config.get('E_learnable_alpha', config.get('ffn_learnable_alpha', config.get('learnable_alpha', True))),
            obs_sigma_gradient=config.get('obs_sigma_gradient', True),
            obs_sigma_weight=config.get('obs_sigma_weight', 1.0),
            sigma_max=config.get('sigma_max', 5.0),
            e_step_sigma_floor=config.get('e_step_sigma_floor', 0.1),
            n_picard_steps=config.get('n_picard_steps', 0),
            picard_trust_region=config.get('picard_trust_region', 5.0),
            # Performance
            compile_vfe=config.get('compile_vfe', False),
            gradient_checkpoint_vfe=config.get('gradient_checkpoint_vfe', False),
            # Gauge geometry
            gauge_mode=config.get('gauge_mode', 'learned'),
            gauge_param=config.get('gauge_param', 'phi'),
            enforce_orthogonal=config.get('enforce_orthogonal', False),
            isotropic_covariance=config.get('isotropic_covariance', False),
            # Non-flat gauge transport
            non_flat_transport=config.get('non_flat_transport', False),
            cocycle_relaxation=config.get('cocycle_relaxation', 0.5),
            per_head_flatness_gate=config.get('per_head_flatness_gate', False),
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
            # DEQ
            use_deq=config.get('use_deq', False),
            deq_neumann_terms=config.get('deq_neumann_terms', 5),
            deq_include_phi=config.get('deq_include_phi', False),
            # Pure VFE mode
            use_layernorm=config.get('use_layernorm', True),
            norm_type=config.get('norm_type', 'layernorm' if config.get('use_layernorm', True) else 'none'),
            use_residual=config.get('use_residual', True),
            skip_attention=config.get('skip_attention', False),
            closed_form_e_step=config.get('closed_form_e_step', False),
            # Memory efficiency
            gradient_checkpointing=config.get('gradient_checkpointing', False),
            # Multi-layer depth signal
            aux_layer_loss=config.get('aux_layer_loss', False),
            aux_loss_weight=config.get('aux_loss_weight', 0.3),
            sigma_residual=(
                # Loud deprecation warning: sigma_residual is now a no-op.
                # If a user explicitly sets it True, they are likely expecting
                # the old additive behavior that compounded sigma across
                # sublayers — which was the source of the ceiling-saturation
                # bug fixed this session.  Emit a warning so the behavioral
                # change is visible.
                _warn_sigma_residual_deprecated(config.get('sigma_residual', False))
                if 'sigma_residual' in config
                else False
            ),
            hierarchical_priors=config.get('hierarchical_priors', True),
            rope_full_gauge=config.get('rope_full_gauge', False),
            # Active inference / EFE
            active_inference=config.get('active_inference', False),
            active_inference_pragmatic_weight=config.get('active_inference_pragmatic_weight', 1.0),
            active_inference_epistemic_weight=config.get('active_inference_epistemic_weight', 0.5),
            active_inference_epistemic_samples=config.get('active_inference_epistemic_samples', 4),
            active_inference_decode_tau=config.get('active_inference_decode_tau', 1.0),
            active_inference_trust_region=config.get('active_inference_trust_region', 0.5),
            active_inference_lr=config.get('active_inference_lr', 1.0),
            # Bootstrap self-distillation
            active_inference_distill_weight=config.get('active_inference_distill_weight', 0.0),
            active_inference_distill_lr=config.get('active_inference_distill_lr', 1.0),
            active_inference_distill_normalize=config.get('active_inference_distill_normalize', True),
            active_inference_distill_mode=config.get('active_inference_distill_mode', 'aggregated'),
            # Non-serializable
            generators=generators,
            ffn_prior_bank=prior_bank,
            ffn_use_prior_bank=config.get('use_prior_bank', False),
            cross_head_perm=cross_head_perm,
        )
