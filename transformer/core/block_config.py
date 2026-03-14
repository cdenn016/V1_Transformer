"""
BlockConfig: Single dataclass replacing the 40+ parameter waterfall.

Instead of:
    config dict → model.py (40 config.get calls) → Stack(40 kwargs) → Block(40 kwargs) → FFN(30 kwargs)

Now:
    config dict → BlockConfig.from_config(config) → Stack(cfg) → Block(cfg) → FFN(cfg)
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import math

import torch
import torch.nn as nn


@dataclass
class BlockConfig:
    """All parameters needed to construct a GaugeTransformerBlock (and its sub-modules)."""

    # === Structural ===
    embed_dim: int = 64
    irrep_spec: List[Tuple[str, int, int]] = field(default_factory=lambda: [('ℓ0', 8, 1)])
    hidden_dim: int = 256
    n_layers: int = 1  # Only used by Stack

    # === Attention ===
    kappa_beta: float = 1.0       # Temperature for KL-based attention
    attention_pattern: str = 'full'
    attention_window: int = 64
    mask_self_attention: bool = False  # Prevent KL(q_i||q_i)=0 collapse
    use_output_projection: bool = False  # W_O after multi-head attention
    multihead_vfe: bool = False   # Per-head β through VFE iterations

    # === Belief evolution ===
    evolve_sigma: bool = True     # Update covariances Σ
    evolve_phi: bool = True       # Update gauge frames φ (M-step)
    evolve_phi_e_step: bool = False  # Update φ during E-step iterations
    phi_lr: float = 0.05          # Learning rate for ∂F/∂φ descent
    phi_max_norm: float = math.pi # Max phi norm (π radians)
    phi_dim: int = 3              # 3 for SO(3), N(N-1)/2 for SO(N)
    phi_natural_gradient: str = 'clip'  # 'clip'|'cartan'|'killing'|'pullback'
    diagonal_covariance: bool = False
    isotropic_covariance: bool = False  # Force Σ = σ²I (Limit 1: KL → squared Euclidean
                                       # when combined with O(K) gauge via enforce_orthogonal
                                       # or gauge_mode='trivial')
    learnable_reflection: bool = False  # If True, learn per-token sign vectors s_i ∈ {±1}^K
                                        # extending SO(K) transport to full O(K) = SO(K) ⋊ Z_2.
                                        # Ensures S(Ω) = 0 and isotropic covariance is preserved
                                        # under transport (ΩΩ^T = I).
    amortized_inference: bool = True  # Gradient flow through priors for learned E-step init

    # === VFE dynamics (FFN E-step) ===
    ffn_mode: str = 'VFE_dynamic'
    ffn_alpha: float = 0.001      # Prior weight inside VFE loop
    ffn_kappa: float = 1.0        # Softmax temperature (unified with kappa_beta)
    ffn_n_iterations: int = 1     # VFE inference iterations
    ffn_learnable_lr: bool = True # Learn step size for variational descent
    ffn_lambda_belief: float = 1.0  # Belief alignment weight
    ffn_update_sigma: bool = True # Update covariances during FFN
    ffn_learnable_alpha: bool = False  # Bayesian precision via Gamma-Normal

    # === Gauge geometry ===
    gauge_mode: str = 'learned'   # 'learned' or 'trivial' (Ω = I)
    enforce_orthogonal: bool = False  # Enforce Ω ∈ SO(K) via Newton-Schulz.
                                      # Combined with learnable_reflection=True, gives full O(K).
                                      # Required for isotropic_covariance to produce exact
                                      # squared Euclidean KL (S(Ω)=0, transported Σ stays isotropic).

    # === Positional encoding ===
    alibi_slope: Optional[float] = None  # ALiBi positional bias
    use_rope: bool = False
    rope_base: float = 10000.0

    # === Memory efficiency ===
    ffn_irrep_dims: Optional[List[int]] = None  # Block dims for KL decomposition
    ffn_chunk_size: Optional[int] = None

    # === DEQ (Deep Equilibrium) ===
    use_deq: bool = False
    deq_neumann_terms: int = 5

    # === Pure VFE mode flags ===
    use_layernorm: bool = True
    use_residual: bool = True

    # === Non-serializable objects (set after construction) ===
    # These are torch tensors / nn.Modules that can't be part of a plain dataclass default
    generators: Optional[torch.Tensor] = field(default=None, repr=False)
    ffn_prior_bank: Optional[nn.Module] = field(default=None, repr=False)
    ffn_use_prior_bank: bool = False
    cross_head_perm: Optional[object] = field(default=None, repr=False)

    @classmethod
    def from_config(cls, config: dict, generators: Optional[torch.Tensor] = None,
                    prior_bank: Optional[nn.Module] = None,
                    cross_head_perm=None,
                    ffn_irrep_dims: Optional[List[int]] = None) -> 'BlockConfig':
        """Build BlockConfig from the flat config dict used by train_publication.py."""
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
            phi_lr=config.get('phi_lr', 0.05),
            phi_max_norm=config.get('phi_max_norm', math.pi),
            phi_dim=config.get('phi_dim', 3),
            phi_natural_gradient=config.get('phi_natural_gradient', 'clip'),
            diagonal_covariance=config.get('diagonal_covariance', False),
            isotropic_covariance=config.get('isotropic_covariance', False),
            learnable_reflection=config.get('learnable_reflection', False),
            amortized_inference=config.get('amortized_inference', True),
            # VFE dynamics
            ffn_mode=config.get('ffn_mode', 'VFE_dynamic'),
            ffn_alpha=config.get('ffn_alpha', 0.001),
            ffn_kappa=kappa_beta,  # Unified temperature
            ffn_n_iterations=config.get('ffn_n_iterations', 1),
            ffn_learnable_lr=config.get('ffn_learnable_lr', True),
            ffn_lambda_belief=config.get('ffn_lambda_belief', 1.0),
            ffn_update_sigma=config.get('ffn_update_sigma', True),
            ffn_learnable_alpha=config.get('learnable_alpha', False),
            # Gauge geometry
            gauge_mode=config.get('gauge_mode', 'learned'),
            enforce_orthogonal=config.get('enforce_orthogonal', False),
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
