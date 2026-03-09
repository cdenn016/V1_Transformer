# -*- coding: utf-8 -*-
"""
Gauge-Transformer Configuration
================================

Single dataclass replacing the 50+ parameter forwarding chain:
    model.py → blocks.py → attention.py / variational_ffn.py

Every module receives this config object instead of dozens of kwargs.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class GaugeTransformerConfig:
    """Complete configuration for a gauge-theoretic transformer."""

    # ── Architecture ──────────────────────────────────────────────────
    vocab_size: int = 256
    embed_dim: int = 32
    n_layers: int = 6
    hidden_dim: int = 128          # FFN hidden dim (unused in pure VFE, kept for API)
    max_seq_len: int = 512
    irrep_spec: Optional[List[Tuple[str, int, int]]] = None  # [(label, mult, dim), ...]

    # ── Gauge group ───────────────────────────────────────────────────
    gauge_group: str = 'SO3'       # 'SO3' | 'SON' | 'GLK'
    gauge_dim: int = 3             # N for SO(N), K for GL(K)
    gauge_mode: str = 'learned'    # 'learned' | 'trivial'
    use_multi_irrep: bool = False
    phi_dim: int = 3               # Computed from gauge_group if not set
    enforce_orthogonal: bool = False

    # ── Covariance ────────────────────────────────────────────────────
    diagonal_covariance: bool = False
    evolve_sigma: bool = True
    evolve_phi: bool = True
    evolve_phi_e_step: bool = False

    # ── VFE parameters ────────────────────────────────────────────────
    alpha: float = 0.0             # Prior self-coupling weight in VFE loop
    kappa: float = 1.0             # Softmax temperature for attention
    n_vfe_iterations: int = 1      # E-step iterations per forward pass
    learnable_lr: bool = True      # Learn step size for variational descent
    lambda_belief: float = 0.0     # Belief alignment weight
    update_sigma: bool = True      # Update covariances during VFE
    sigma_softmax_coupling: bool = False  # Include ∂β/∂Σ in sigma gradient
    compute_sigma_align_grad: bool = True

    # ── Phi evolution ─────────────────────────────────────────────────
    phi_lr: float = 0.05
    phi_max_norm: float = math.pi  # π radians = 180° rotation
    phi_natural_gradient: str = 'clip'  # 'clip' | 'cartan' | 'killing' | 'pullback'

    # ── Memory efficiency ─────────────────────────────────────────────
    chunk_size: Optional[int] = None
    use_block_diagonal_kl: bool = True

    # ── Attention features ────────────────────────────────────────────
    mask_self_attention: bool = False
    use_identity_transport: bool = False
    alibi_slope: Optional[float] = None
    use_rope: bool = False
    rope_base: float = 10000.0
    attention_pattern: str = 'full'
    attention_window: int = 64

    # ── Multi-head ────────────────────────────────────────────────────
    multihead_vfe: bool = False
    per_head_kappa: bool = False
    use_output_projection: bool = False

    # ── Cross-head coupling ───────────────────────────────────────────
    cross_couplings: List = field(default_factory=list)

    # ── Bayesian precision ────────────────────────────────────────────
    learnable_alpha: bool = False

    # ── Pure FEP mode ─────────────────────────────────────────────────
    pure_fep_mode: bool = False
    use_prior_bank: bool = False
    prior_lr: float = 0.01
    gauge_fixed_priors: bool = False

    # ── Embeddings ────────────────────────────────────────────────────
    mu_init_std: Optional[float] = None   # Default: 2.0 in embedding
    phi_scale: float = 0.3
    mu_normalize: bool = False
    mu_max_norm: Optional[float] = None
    use_positional_embedding: bool = False
    pos_encoding_mode: str = 'none'
    pos_encoding_scale: float = 0.1
    tie_embeddings: bool = True

    # ── Model channel (slow subsystem) ───────────────────────────────
    lambda_gamma: float = 0.0      # Model coupling: Σ γ_ij · KL(s_i || Ω_ij s_j)
    kappa_gamma: float = 1.0       # Temperature for γ_ij model coupling weights
    lambda_hyper: float = 0.0      # Hyper-prior: KL(s_i || h) models toward centroid
    alpha_phi: float = 0.0         # Gauge prior: (α_φ/2) Σ ||φ_i||²

    # ── Regularization ────────────────────────────────────────────────
    dropout: float = 0.1
    use_layernorm: bool = True
    use_dropout: bool = True
    use_residual: bool = True

    # ── Derived (computed in __post_init__) ────────────────────────────
    # These are set automatically and should not be passed by the user.
    irrep_dims: Optional[List[int]] = field(default=None, repr=False)

    def __post_init__(self):
        # Validate gauge_mode
        if self.gauge_mode not in ('learned', 'trivial'):
            raise ValueError(f"gauge_mode must be 'learned' or 'trivial', got '{self.gauge_mode}'")

        # Trivial gauge → override transport and phi evolution
        if self.gauge_mode == 'trivial':
            self.use_identity_transport = True
            self.evolve_phi = False
            self.evolve_phi_e_step = False

        # Pure FEP requires untied embeddings
        if self.pure_fep_mode and self.tie_embeddings:
            self.tie_embeddings = False

        # Compute phi_dim from gauge group if using defaults
        if self.gauge_group == 'SO3':
            self.phi_dim = 3
        elif self.gauge_group == 'GLK':
            self.phi_dim = self.embed_dim * self.embed_dim
        elif self.gauge_group == 'SON':
            self.phi_dim = self.gauge_dim * (self.gauge_dim - 1) // 2

        # Compute irrep_dims from irrep_spec if block-diagonal KL is enabled
        if self.irrep_spec is not None and self.use_block_diagonal_kl:
            self.irrep_dims = []
            for _label, mult, dim in self.irrep_spec:
                self.irrep_dims.extend([dim] * mult)

    @classmethod
    def from_legacy_dict(cls, config: dict) -> 'GaugeTransformerConfig':
        """Create config from legacy dictionary format for backward compatibility."""
        mapping = {
            'vocab_size': 'vocab_size',
            'embed_dim': 'embed_dim',
            'n_layers': 'n_layers',
            'hidden_dim': 'hidden_dim',
            'max_seq_len': 'max_seq_len',
            'irrep_spec': 'irrep_spec',
            'kappa_beta': 'kappa',
            'dropout': 'dropout',
            'evolve_sigma': 'evolve_sigma',
            'evolve_phi': 'evolve_phi',
            'evolve_phi_e_step': 'evolve_phi_e_step',
            'tie_embeddings': 'tie_embeddings',
            'diagonal_covariance': 'diagonal_covariance',
            'gauge_group': 'gauge_group',
            'gauge_dim': 'gauge_dim',
            'gauge_mode': 'gauge_mode',
            'use_multi_irrep': 'use_multi_irrep',
            'ffn_alpha': 'alpha',
            'ffn_n_iterations': 'n_vfe_iterations',
            'ffn_learnable_lr': 'learnable_lr',
            'ffn_lambda_belief': 'lambda_belief',
            'ffn_update_sigma': 'update_sigma',
            'ffn_pure_fep_mode': 'pure_fep_mode',
            'ffn_prior_lr': 'prior_lr',
            'ffn_max_seq_len': 'max_seq_len',  # Note: maps to same field
            'ffn_chunk_size': 'chunk_size',
            'use_prior_bank': 'use_prior_bank',
            'gauge_fixed_priors': 'gauge_fixed_priors',
            'use_positional_embedding': 'use_positional_embedding',
            'pos_encoding_mode': 'pos_encoding_mode',
            'pos_encoding_scale': 'pos_encoding_scale',
            'alibi_slope': 'alibi_slope',
            'use_identity_transport': 'use_identity_transport',
            'attention_pattern': 'attention_pattern',
            'attention_window': 'attention_window',
            'mask_self_attention': 'mask_self_attention',
            'sigma_softmax_coupling': 'sigma_softmax_coupling',
            'enforce_orthogonal': 'enforce_orthogonal',
            'learnable_alpha': 'learnable_alpha',
            'per_head_kappa': 'per_head_kappa',
            'use_output_projection': 'use_output_projection',
            'multihead_vfe': 'multihead_vfe',
            'use_rope': 'use_rope',
            'rope_base': 'rope_base',
            'phi_natural_gradient': 'phi_natural_gradient',
            'phi_lr': 'phi_lr',
            'phi_max_norm': 'phi_max_norm',
            'use_layernorm': 'use_layernorm',
            'use_dropout': 'use_dropout',
            'use_residual': 'use_residual',
            'mu_init_std': 'mu_init_std',
            'phi_scale': 'phi_scale',
            'mu_normalize': 'mu_normalize',
            'mu_max_norm': 'mu_max_norm',
            'cross_couplings': 'cross_couplings',
            'use_block_diagonal_kl': 'use_block_diagonal_kl',
            'lambda_gamma': 'lambda_gamma',
            'kappa_gamma': 'kappa_gamma',
            'lambda_hyper': 'lambda_hyper',
            'alpha_phi': 'alpha_phi',
        }

        kwargs = {}
        for legacy_key, new_key in mapping.items():
            if legacy_key in config:
                kwargs[new_key] = config[legacy_key]

        return cls(**kwargs)
