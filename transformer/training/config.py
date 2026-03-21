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

    # Multi-group mode: Per-parameter group learning rates
    mu_lr: float = 0.1           # Mean embeddings (natural gradient scale)
    sigma_lr: float = 0.005      # Covariance embeddings (smaller for stability)
    phi_lr: float = 0.01         # Gauge frames
    attention_lr: float = 0.01   # Attention parameters
    ffn_lr: float = 0.001        # FFN parameters (standard)
    output_lr: float = 0.001     # Output projection

    # ==========================================================================
    # Optimizer Hyperparameters
    # ==========================================================================
    # weight_decay implements a Gaussian hyper-prior on parameters:
    #   p(θ) = N(0, 1/(2·wd))  →  -log p(θ) = wd·||θ||²
    #
    # In the hierarchical VFE, this is the top level of the Bayesian hierarchy:
    #   observations → q_i (beliefs, E-step) → p_i (priors, M-step) → N(0, 1/(2·wd))
    #
    # For embedding parameters (μ_p, σ_p, φ_p), weight decay is the hyper-prior
    # precision that prevents prior drift. Larger wd = tighter hyper-prior.
    weight_decay: float = 0.1
    # Embedding-specific weight decay (Level 3 hyper-prior on priors).
    # None = use weight_decay (same as other params).
    # 0.0 = uninformative hyper-prior (no pull toward zero).
    # The VFE hierarchy is: N(0,1/(2·wd)) → p_i(embeddings) → q_i(beliefs) → obs
    # For mu_embed: wd pulls prior means toward zero
    # For sigma_embed: wd on log_sigma pulls covariances toward σ=1
    # For phi_embed: wd pulls gauge frames toward identity (exp(0)=I)
    # Default 0.0: VFE loss terms handle embedding regularization:
    #   - alpha · KL(q||p) couples μ_p/Σ_p to posterior (self-consistency)
    #   - alpha_phi · ||φ||²/2 is L2 on gauge frames
    # Optimizer WD on embeddings conflicts with these principled terms.
    embed_weight_decay: Optional[float] = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0

    # ==========================================================================
    # Learning Rate Schedule
    # ==========================================================================
    warmup_steps: int = 1000
    max_steps: int = 50000
    lr_decay: str = 'cosine'  # 'cosine', 'linear', 'constant'
    min_lr: float = 3e-5

    # ==========================================================================
    # Free Energy Weights
    # ==========================================================================
    # NOTE: alpha > 0 is CRITICAL for gradient flow to embeddings!
    alpha: float = 0.1           # Self-consistency: KL(q||p) to embedding priors
    lambda_beta: float = 1.0     # Belief alignment: Σβ_ij·KL (CRUCIAL!)
    lambda_gamma: float = 0.0    # Model alignment (disabled by default)
    kappa_gamma: float = 1.0     # Temperature for γ_ij coupling weights
    use_obs_in_vfe: bool = False # Pass targets as observations into VFE E-step (last layer only)
    obs_sigma_gradient: bool = False  # ∂E_q[CE]/∂σ Hessian-diagonal obs gradient for sigma
    obs_sigma_weight: float = 1.0     # Weight for sigma observation gradient

    # ==========================================================================
    # Training Loop
    # ==========================================================================
    batch_size: int = 16
    max_seq_len: int = 256
    num_epochs: Optional[int] = None  # If set, overrides max_steps
    accumulation_steps: int = 1

    # ==========================================================================
    # Logging & Evaluation
    # ==========================================================================
    log_interval: int = 10
    eval_interval: int = 100
    checkpoint_interval: int = 500

    # ==========================================================================
    # Checkpointing
    # ==========================================================================
    checkpoint_dir: Optional[Path] = None
    save_optimizer: bool = True
    save_total_limit: int = 3
    resume_from: Optional[str] = None  # Path to checkpoint to resume from

    # ==========================================================================
    # Weights & Biases
    # ==========================================================================
    use_wandb: bool = False
    wandb_project: str = 'gauge-transformer'
    wandb_run_name: Optional[str] = None

    # ==========================================================================
    # Hardware
    # ==========================================================================
    device: str = 'cuda'

    # ==========================================================================
    # Gauge Group
    # ==========================================================================
    # Gauge mode: 'learned' (per-token frames, full transport Ω_ij)
    # or 'trivial' (global frame, Ω = I, standard KL-attention).
    gauge_mode: str = 'learned'

    # Gauge parameterization: how gauge frames are stored and optimized.
    #   'phi':   Lie algebra coefficients φ_i ∈ ℝ^{n_gen}, with Ω_i = exp(φ_i·G).
    #            Requires matrix_exp forward + dexp series backward.
    #            Only reaches GL⁺(K) (det > 0); needs separate reflection for O(K).
    #   'omega': Direct group elements Ω_i ∈ GL(K), stored as K_h×K_h matrices per head.
    #            No matrix_exp needed. Gradient via chain rule through Ω_ij = Ω_i·Ω_j⁻¹.
    #            Covers full GL(K) including reflections (det < 0). Major compute savings.
    gauge_param: str = 'phi'  # 'phi' or 'omega'

    # Learning rate for direct Omega embeddings (used when gauge_param='omega')
    omega_lr: float = 0.01

    # Trust region for Omega retraction on GL(K) manifold
    omega_trust_region: float = 0.3

    # Isotropic covariance: force Σ = σ²I (scalar variance × identity)
    # This is Limit 1 from the manuscript: KL reduces to scaled squared Euclidean.
    # Combined with gauge_mode='trivial' (Limit 2), recovers standard attention.
    isotropic_covariance: bool = False

    # ==========================================================================
    # Positional Encoding
    # ==========================================================================
    use_rope: bool = False       # RoPE: SO(2)^{K/2} position rotations on μ in attention
    rope_base: float = 10000.0   # RoPE frequency base

    # ==========================================================================
    # Model Architecture (for creation, not training)
    # ==========================================================================
    embed_dim: int = 128
    n_layers: int = 4
    vocab_size: int = 50257  # GPT-2 tokenizer size

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
