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

import torch
import torch.nn as nn


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
    embed_dim: int = 64                 # Total belief dimension K = Σ (mult_ℓ × dim_ℓ)
    irrep_spec: List[Tuple[str, int, int]] = field(  # [(label, multiplicity, dim), ...]
        default_factory=lambda: [('ℓ0', 8, 1)]       #   e.g. [('ℓ0',75,1),('ℓ1',30,3),('ℓ2',18,5)]
    )
    hidden_dim: int = 256               # FFN hidden dimension (kept for config compat, not used by blocks)
    n_layers: int = 1                   # Number of stacked blocks (only used by Stack)

    # === Attention ===
    kappa_beta: float = 1.0             # Temperature τ for KL-based attention softmax
    attention_pattern: str = 'full'     # 'full' (only supported pattern)
    attention_window: int = 64          # Window size (unused, kept for API compat)
    mask_self_attention: bool = False   # Prevent KL(q_i||q_i)=0 collapse
    use_output_projection: bool = False # W_O ∈ R^{K×K} after multi-head concat
    multihead_vfe: bool = False         # Per-head β_h through VFE iterations

    # === Belief evolution ===
    evolve_sigma: bool = True           # Update covariances Σ via natural gradient
    evolve_phi: bool = True             # Update gauge frames φ (M-step, after E-step loop)
    evolve_phi_e_step: bool = False     # Update φ during EACH E-step iteration
    phi_update_interval: int = 1        # Update phi every N E-step iterations (1=every)
    phi_lr: float = 0.05               # Learning rate for ∂F/∂φ descent
    phi_max_norm: float = math.pi       # Max phi norm (π radians = 180°)
    phi_dim: int = 3                    # 3 for SO(3), N(N-1)/2 for SO(N), K² for GL(K)
    phi_natural_gradient: str = 'clip'  # 'clip'|'cartan'|'killing'|'pullback'
    diagonal_covariance: bool = False   # σ as (B,N,K) diagonal instead of (B,N,K,K) full
    exact_diagonal_transport: bool = False  # When True + diagonal_covariance, lift σ to full
                                            # for exact Ω@diag(σ)@Ω^T transport (slower but exact)
    amortized_inference: bool = True    # Gradient flow through priors for learned E-step init

    # === Analytic phi gradient ===
    analytic_phi_grad: bool = False     # If True, bypass autograd for ∂F/∂φ (saves ~250MB)
    analytic_phi_grad_dexp_order: int = 4  # dexp series truncation order (4-8 typical)

    # === VFE dynamics (FFN E-step) ===
    ffn_mode: str = 'VFE_dynamic'       # FFN mode (only 'VFE_dynamic' supported)
    ffn_alpha: float = 0.001            # Prior self-coupling weight α in VFE loop
    ffn_kappa: float = 1.0              # Softmax temperature (unified with kappa_beta)
    ffn_n_iterations: int = 1           # VFE inference iterations per forward pass
    ffn_learnable_lr: bool = True       # Learn step size η for variational descent
    ffn_lambda_belief: float = 1.0      # Belief alignment weight λ
    ffn_update_sigma: bool = True       # Update covariances during FFN E-step
    ffn_learnable_alpha: bool = False   # Bayesian precision via Gamma-Normal conjugacy

    # === Gauge geometry ===
    gauge_mode: str = 'learned'         # 'learned' | 'trivial' (Ω=I) | 'constant' (per-head Ω)
    gauge_param: str = 'phi'            # 'phi' (Lie algebra) | 'omega' (direct GL(K) matrices)
    enforce_orthogonal: bool = False    # Enforce Ω ∈ SO(K) via Newton-Schulz iteration
    isotropic_covariance: bool = False  # Force Σ = σ²I (manuscript Limit 1)
    # NOTE: learnable_reflection is an embedding-level feature, handled by
    # model.py → GaugeTokenEmbedding, not by blocks. Not stored here.

    # === Non-flat gauge transport (flat bundle experiments) ===
    # When non_flat_transport=True, the transport becomes:
    #   phi path:  Ω_ij = exp(φ_i·G) · exp(α·δ_ij·G) · exp(-φ_j·G)
    #   omega path: Ω_ij = Ω_i · exp(α·δ_ij·G) · Ω_j⁻¹
    # where δ_ij is an edge-local "connection" parameterized by connection_type.
    # When non_flat_transport=False (default), all non-flat fields are ignored
    # and the standard flat transport is used (cocycle condition holds, holonomy trivial).
    non_flat_transport: bool = False       # Enable edge-dependent connection δ_ij
    cocycle_relaxation: float = 0.0        # Scale factor for δ_ij: 0=flat, 1=fully non-flat
    per_head_flatness_gate: bool = False   # Learnable per-head g_h ∈ [0,1]
    connection_type: str = 'bilinear'      # 'bilinear' | 'mlp'
    connection_hidden_dim: int = 64        # Hidden dim for MLP connection
    connection_init_scale: float = 0.01    # W init scale (0=flat saddle, >0 breaks symmetry)
    holonomy_penalty: float = 0.0          # λ_H · E[‖H_ijk - I‖²_F] added to loss

    # === Positional encoding ===
    alibi_slope: Optional[float] = None    # ALiBi positional bias (negative = recency)
    use_rope: bool = False                 # Rotary position embeddings on μ before KL
    rope_base: float = 10000.0             # RoPE frequency base

    # === Memory efficiency ===
    ffn_irrep_dims: Optional[List[int]] = None  # Block dims for block-diagonal KL decomposition
    ffn_chunk_size: Optional[int] = None         # Chunk size C for O(C²K²) memory processing

    # === DEQ (Deep Equilibrium) ===
    use_deq: bool = False              # Use implicit differentiation for E-step fixed point
    deq_neumann_terms: int = 5         # Neumann series terms for DEQ backward pass

    # === Pure VFE mode flags ===
    use_layernorm: bool = True         # LayerNorm on means (False for pure VFE ablation)
    use_residual: bool = True          # Residual connections (False for pure VFE ablation)

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
            hidden_dim=config['hidden_dim'],
            n_layers=config['n_layers'],
            # Attention
            kappa_beta=kappa_beta,
            attention_pattern=config.get('attention_pattern', 'full'),
            attention_window=config.get('attention_window', 64),
            mask_self_attention=config.get('mask_self_attention', False),
            use_output_projection=config.get('use_output_projection', False),
            multihead_vfe=config.get('multihead_vfe', False),
            # Belief evolution
            evolve_sigma=config.get('evolve_sigma', True),
            evolve_phi=config.get('evolve_phi', True),
            evolve_phi_e_step=config.get('evolve_phi_e_step', False),
            phi_update_interval=config.get('phi_update_interval', 1),
            phi_lr=config.get('phi_lr', 0.05),
            phi_max_norm=config.get('phi_max_norm', math.pi),
            phi_dim=config.get('phi_dim', 3),
            phi_natural_gradient=config.get('phi_natural_gradient', 'clip'),
            diagonal_covariance=config.get('diagonal_covariance', False),
            exact_diagonal_transport=config.get('exact_diagonal_transport', False),
            amortized_inference=config.get('amortized_inference', True),
            # Analytic phi gradient
            analytic_phi_grad=config.get('analytic_phi_grad', False),
            analytic_phi_grad_dexp_order=config.get('analytic_phi_grad_dexp_order', 4),
            # VFE dynamics
            ffn_mode=config.get('ffn_mode', 'VFE_dynamic'),
            ffn_alpha=config.get('ffn_alpha', 0.001),
            ffn_kappa=kappa_beta,  # Unified temperature
            ffn_n_iterations=config.get('ffn_n_iterations', 1),
            ffn_learnable_lr=config.get('ffn_learnable_lr', True),
            ffn_lambda_belief=config.get('ffn_lambda_belief', 1.0),
            ffn_update_sigma=config.get('ffn_update_sigma', True),
            ffn_learnable_alpha=config.get('ffn_learnable_alpha', config.get('learnable_alpha', False)),
            # Gauge geometry
            gauge_mode=config.get('gauge_mode', 'learned'),
            gauge_param=config.get('gauge_param', 'phi'),
            enforce_orthogonal=config.get('enforce_orthogonal', False),
            isotropic_covariance=config.get('isotropic_covariance', False),
            # Non-flat gauge transport
            non_flat_transport=config.get('non_flat_transport', False),
            cocycle_relaxation=config.get('cocycle_relaxation', 0.0),
            per_head_flatness_gate=config.get('per_head_flatness_gate', False),
            connection_type=config.get('connection_type', 'bilinear'),
            connection_hidden_dim=config.get('connection_hidden_dim', 64),
            connection_init_scale=config.get('connection_init_scale', 0.01),
            holonomy_penalty=config.get('holonomy_penalty', 0.0),
            # Positional encoding
            alibi_slope=config.get('alibi_slope', None),
            use_rope=config.get('use_rope', False),
            rope_base=config.get('rope_base', 10000.0),
            # Memory efficiency
            ffn_irrep_dims=ffn_irrep_dims,
            ffn_chunk_size=config.get('ffn_chunk_size', None),
            # DEQ
            use_deq=config.get('use_deq', False),
            deq_neumann_terms=config.get('deq_neumann_terms', 5),
            # Pure VFE mode
            use_layernorm=config.get('use_layernorm', True),
            use_residual=config.get('use_residual', True),
            # Non-serializable
            generators=generators,
            ffn_prior_bank=prior_bank,
            ffn_use_prior_bank=config.get('use_prior_bank', False),
            cross_head_perm=cross_head_perm,
        )
