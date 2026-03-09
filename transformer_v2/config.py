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

    # ── VFE E-step (FFN) parameters ────────────────────────────────────
    alpha_ffn: float = 1.0         # Precision α for VFE E-step in FFN
    kappa_ffn: float = 1.0         # Softmax temperature for β in FFN E-step
    lambda_beta_ffn: float = 1.0   # Belief coupling weight λ_β in FFN E-step
    lambda_gamma_ffn: float = 0.0  # Model coupling weight λ_γ in FFN E-step
    kappa_gamma_ffn: float = 1.0   # Temperature for γ_ij in FFN E-step
    lambda_hyper_ffn: float = 0.0  # Hyper-prior weight in FFN E-step
    alpha_phi_ffn: float = 0.0     # Gauge prior (α_φ/2)||φ||² in FFN E-step
    n_vfe_iterations: int = 1      # E-step iterations per forward pass
    learnable_lr: bool = True      # Learn step size for variational descent

    # ── Training loss weights (M-step) ───────────────────────────────
    alpha_loss: float = 0.0        # Self-coupling weight KL(q||p) in training loss
    kappa_loss: float = 1.0        # Softmax temperature for β in loss
    lambda_beta_loss: float = 0.0  # Belief coupling weight λ_β in training loss
    lambda_gamma_loss: float = 0.0 # Model coupling weight λ_γ in training loss
    kappa_gamma_loss: float = 1.0  # Temperature for γ_ij in training loss
    lambda_hyper_loss: float = 0.0 # Hyper-prior weight in training loss
    alpha_phi_loss: float = 0.0    # Gauge prior (α_φ/2)||φ||² in training loss
    update_sigma: bool = True      # Update covariances during VFE
    sigma_softmax_coupling: bool = False  # Include ∂β/∂Σ in sigma gradient
    compute_sigma_align_grad: bool = True

    # ── Phi evolution ─────────────────────────────────────────────────
    phi_lr: float = 0.05
    phi_max_norm: Optional[float] = None  # Set in __post_init__: π for SO, None (no clamp) for GL
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

        # phi_max_norm: π for SO (rotation angles), None for GL (no geometric bound)
        if self.phi_max_norm is None:
            if self.gauge_group in ('SO3', 'SON'):
                self.phi_max_norm = math.pi
            # GL(K): leave as None → no clamp

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
            'kappa_beta': 'kappa_ffn',
            'kappa': 'kappa_loss',
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
            'ffn_alpha': 'alpha_ffn',
            'alpha': 'alpha_loss',
            'ffn_n_iterations': 'n_vfe_iterations',
            'ffn_learnable_lr': 'learnable_lr',
            'ffn_lambda_belief': 'lambda_beta_ffn',
            'lambda_belief': 'lambda_beta_loss',
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
            'lambda_gamma': 'lambda_gamma_loss',
            'kappa_gamma': 'kappa_gamma_loss',
            'lambda_hyper': 'lambda_hyper_loss',
            'alpha_phi': 'alpha_phi_loss',
        }

        kwargs = {}
        for legacy_key, new_key in mapping.items():
            if legacy_key in config:
                kwargs[new_key] = config[legacy_key]

        return cls(**kwargs)
