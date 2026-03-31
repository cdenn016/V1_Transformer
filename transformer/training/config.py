"""
Unified Training Configuration
==============================

Single source of truth for training hyperparameters across all modes.
Supports standard transformers and VFE-based gauge-theoretic transformers
with configurable gauge groups (SO(N), GL(K)) via BlockConfig.

Modes are selected by training_mode ('standard' vs 'vfe_dynamic') and
parameter grouping strategy (use_param_groups). Gauge transport can be
trivialized (gauge_mode='trivial') to recover standard KL-attention.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List


@dataclass
class TrainingConfig:
    """
    Unified training configuration supporting all training modes.

    Optimizer Modes:
        - Simple (use_param_groups=False): Single LR for all parameters.
        - Multi-group (use_param_groups=True): Per-type LRs exploiting
          natural gradient structure (mu, sigma, phi, attention, ffn, output).

    Training Types:
        - 'standard': Standard transformer baseline (no gauge theory).
        - 'vfe_dynamic': VFE-based transformer with gauge transport.
          Gauge group (SO(N), GL(K)) is determined by BlockConfig.generators
          shape (n_gen, K, K); this config controls training, not architecture.

    The VFE loss hierarchy is:
        observations -> q_i (beliefs, E-step) -> p_i (priors, M-step) -> hyper-prior N(0, 1/(2*wd))
    """

    # ==========================================================================
    # Training Mode
    # ==========================================================================
    training_mode: str = 'vfe_dynamic'  # 'standard', 'vfe_dynamic', 'pure_fep'

    # ==========================================================================
    # Parameter Grouping Strategy
    # ==========================================================================
    use_param_groups: bool = True  # If True, use multi-group learning rates

    # Simple mode: Single learning rate (used when use_param_groups=False)
    learning_rate: float = 3e-4

    # Multi-group mode: M-step learning rates for AdamW parameter groups.
    # These control how fast nn.Parameter objects update via backprop (M-step).
    # The E-step (inner VFE loop) has its own rates: e_step_mu_lr, e_step_sigma_lr,
    # e_step_phi_lr — set in BlockConfig / EM_CONFIG, not here.
    # Note: mu_embed and log_sigma_diag serve dual roles (q₀ init AND prior μ_p/σ_p),
    # so these rates indirectly affect E-step initialization.
    mu_lr: float = 0.1           # Prior mean embeddings (μ_p)
    sigma_lr: float = 0.005      # Prior covariance embeddings (log σ_p)
    phi_lr: float = 0.01         # Gauge frame embeddings (φ)
    attention_lr: float = 0.01   # Attention params (W_O, constant_omega)
    ffn_lr: float = 0.001        # FFN params (raw_c0, raw_b0, raw_lr via no_decay group)
    output_lr: float = 0.001     # Output projection (vocab logits)

    # ==========================================================================
    # Optimizer Hyperparameters
    # ==========================================================================
    # Optimizer type:
    #   'adamw': Standard AdamW (default). Diagonal Fisher approximation via EMA.
    #   'riemannian_adam': AdamW + Killing-form metric for phi + Fisher metric for mu.
    #   'natural_gradient': Per-token block-diagonal empirical Fisher.
    optimizer_type: str = 'adamw'

    # Natural gradient settings (only used when optimizer_type='natural_gradient')
    fisher_ema_decay: float = 0.95   # EMA decay for Fisher matrix estimation
    fisher_damping: float = 1e-4     # Tikhonov regularization λI for invertibility

    # weight_decay implements a Gaussian hyper-prior on parameters:
    #   p(θ) = N(0, 1/(2·wd))  →  -log p(θ) = wd·||θ||²
    weight_decay: float = 0.1
    embed_weight_decay: Optional[float] = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    grad_clip: float = 1.0
    grad_accumulation_steps: int = 1

    # ==========================================================================
    # Learning Rate Schedule
    # ==========================================================================
    epochs: Optional[int] = None  # If set, overrides max_steps
    warmup_steps: int = 1000
    max_steps: int = 50000
    lr_decay: str = 'cosine'  # 'cosine', 'linear', 'constant'
    min_lr: float = 3e-5           # Absolute minimum LR (used by presets)
    min_lr_ratio: float = 0.1      # Min LR as fraction of peak (used by FastTrainer)

    # ==========================================================================
    # Free Energy Weights
    # ==========================================================================
    alpha: float = 0.0           # Self-consistency: KL(q||p) to embedding priors
    lambda_beta: float = 0.0     # Belief alignment: Σβ_ij·KL  [M-step loss]
    beta: float = 0.0            # Alias for lambda_beta (used by FastTrainer)
    ffn_lambda_belief: float = 1.0  # Belief alignment weight inside VFE E-step
    lambda_gamma: float = 0.0    # Model alignment (disabled by default)
    kappa_gamma: float = 1.0     # Temperature for γ_ij coupling weights
    lambda_hyper: float = 0.0    # Hyper-prior: KL(s_i||h) models to centroid
    use_obs_in_vfe: bool = False # Pass targets as observations into VFE E-step (last layer only)
    obs_sigma_gradient: bool = True  # ∂E_q[CE]/∂σ Hessian-diagonal obs gradient for sigma
    obs_sigma_weight: float = 1.0     # Weight for sigma observation gradient

    # Multi-layer depth signal
    aux_layer_loss: bool = False     # Per-layer auxiliary CE loss
    aux_loss_weight: float = 0.3     # Weight for auxiliary per-layer CE losses
    sigma_residual: bool = False     # Additive σ residual across layers

    # ==========================================================================
    # Training Loop
    # ==========================================================================
    batch_size: int = 64
    max_seq_len: int = 64

    # ==========================================================================
    # Logging & Evaluation
    # ==========================================================================
    log_interval: int = 100
    eval_interval: int = 1000
    checkpoint_interval: int = 10000

    # ==========================================================================
    # Checkpointing
    # ==========================================================================
    checkpoint_dir: Optional[Path] = None
    save_total_limit: int = 3
    resume_from: Optional[str] = None  # Path to checkpoint to resume from

    # ==========================================================================
    # Hardware
    # ==========================================================================
    device: str = 'cuda'

    # ==========================================================================
    # Gauge Group
    # ==========================================================================
    gauge_mode: str = 'learned'    # 'learned', 'trivial', 'constant'
    gauge_param: str = 'phi'       # 'phi' or 'omega'
    omega_lr: float = 0.01         # LR for direct Omega embeddings (gauge_param='omega')
    omega_trust_region: float = 0.3
    isotropic_covariance: bool = False  # Force Σ = σ²I (Limit 1 from manuscript)

    # ==========================================================================
    # Positional Encoding
    # ==========================================================================
    use_rope: bool = True       # RoPE: SO(2)^{K/2} position rotations on μ
    rope_base: float = 10000.0

    # ==========================================================================
    # Model Architecture (for creation, not training)
    # ==========================================================================
    embed_dim: int = 128
    n_layers: int = 4
    vocab_size: int = 50257  # GPT-2 tokenizer size

    # ==========================================================================
    # Gauge Geometry: Phi gradient preconditioning (M-step)
    # ==========================================================================
    alpha_phi: float = 0.0                     # Gauge prior: (α_φ/2)||φ||² loss term
    use_slk_projection: bool = False           # Project phi to traceless sl(K) after each step
    use_killing_form: bool = False             # Cartan decomposition preconditioning for phi grads
    killing_form_sym_dampening: float = 0.1    # Dampening for non-compact (symmetric) directions

    # ==========================================================================
    # P-Flow & Delta Rule (backprop-free learning)
    # ==========================================================================
    use_p_flow: bool = False          # EMA update of token embeddings toward successful beliefs
    p_flow_ema_decay: float = 0.99
    sigma_ce_scale: float = 0.01      # Scale CE gradient to sigma_p (0=detach, 1=full)
    detach_phi: bool = False           # Detach phi from backprop (enables backprop-free phi)
    use_delta_rule_w_out: bool = False # Delta rule for W_out (backprop-free)
    delta_rule_lr: float = 0.001

    # ==========================================================================
    # Diagnostics
    # ==========================================================================
    track_layer_diagnostics: bool = False
    track_iteration_diagnostics: bool = False
    diagnostics_interval: int = 50
    verbose_diagnostics: bool = True

    def __post_init__(self):
        """Convert checkpoint_dir to Path if string."""
        if isinstance(self.checkpoint_dir, str):
            self.checkpoint_dir = Path(self.checkpoint_dir)


# =============================================================================
# Preset Configurations
# =============================================================================

def get_standard_config(**overrides) -> TrainingConfig:
    """Get configuration for standard transformer baseline (no VFE, no gauge transport)."""
    config = TrainingConfig(
        training_mode='standard',
        use_param_groups=False,
        learning_rate=3e-4,
        alpha=0.0,
        lambda_beta=0.0,
        lambda_gamma=0.0,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def get_vfe_dynamic_config(**overrides) -> TrainingConfig:
    """
    Get configuration for VFE-dynamic gauge transformer.

    Uses multi-group natural gradient LRs and VFE loss terms (alpha, lambda_beta).
    Pass gauge_mode='trivial' to disable gauge transport (Omega_ij = I),
    or gauge_mode='learned' for full per-token phi with transport.
    """
    config = TrainingConfig(
        training_mode='vfe_dynamic',
        use_param_groups=True,
        mu_lr=0.1,
        sigma_lr=0.005,
        phi_lr=0.01,
        attention_lr=0.01,
        ffn_lr=0.001,
        output_lr=0.001,
        alpha=0.1,
        lambda_beta=1.0,
        lambda_gamma=0.0,
        use_obs_in_vfe=True,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def get_pure_fep_config(**overrides) -> TrainingConfig:
    """
    Get configuration for Pure VFE transformer (no autograd, no optimizer).

    PureFEP handles its own VFE loss and natural gradient updates internally,
    so alpha/lambda_beta/lambda_gamma are set to 0 (unused by Lightning wrapper).
    """
    config = TrainingConfig(
        training_mode='pure_fep',
        use_param_groups=False,
        alpha=0.0,
        lambda_beta=0.0,
        lambda_gamma=0.0,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config
