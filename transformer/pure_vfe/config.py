"""Configuration for the Pure VFE Transformer."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PureVFEConfig:
    # Vocabulary
    vocab_size: int = 50257

    # Belief geometry
    belief_dim: int = 32       # K: full belief dimension
    n_heads: int = 4           # H: number of heads (block-diagonal Σ)
    head_dim: int = 8          # K_h = K / H per head

    # VFE descent (inference = forward pass)
    n_esteps: int = 12         # Iterations of VFE descent (replaces "depth")
    tau: float = None           # Attention temperature (defaults to √K_h)

    # Per-variable natural gradient learning rates
    # Single VFE objective F(mu_q, Sigma_q, phi, mu_p, Sigma_p) --
    # no E/M distinction, just per-variable step sizes.
    mu_q_lr: float = 0.1       # Belief mean step size
    sigma_q_lr: float = 0.005  # Belief covariance step size (SPD-sensitive)
    phi_lr: float = 0.1        # Gauge connection step size (all frames: prior, positional, E-step)
    mu_p_lr: float = 0.05      # Prior mean step size
    sigma_p_lr: float = 0.01   # Prior covariance step size (SPD-sensitive)

    # Prior precision (state-dependent α)
    alpha_b0: float = 1.0      # Denominator offset
    alpha_c0: float = 1.0      # Numerator scale

    # Hyper-prior regularization -- must be large enough that the hyper-prior
    # doesn't overpower the observation gradient.  At hyper_var=1 the pull-to-zero
    # force (|μ_v|~=2.8) dwarfs the CE signal; 100 weakens it to ~0.03.
    hyper_var: float = 100.0   # Variance of hyper-prior on prior means

    # Sequence
    max_seq_len: int = 64      # N: maximum sequence length
    batch_size: int = 8

    # Initialization
    sigma_init: float = 1.0             # Initial covariance scale
    omega_init_scale: float = 0.01      # GL(K) frame perturbation from I

    # Numerical stability
    grad_clamp: float = 1e3             # Element-wise gradient clamp before natural gradient

    # Trust regions (Frobenius norm caps on natural gradient steps)
    trust_region_mu: float = 2.0        # Whitened trust region for mu (VFE dynamic: 2.0)
    trust_region_sigma: float = 0.15    # Trust region for sigma retraction (tightened from 0.3)
    trust_region_omega: float = 0.3     # Relative trust region for omega updates

    # SPD retraction safeguards
    spd_eps_min: float = 1e-3           # Spectral floor (tightened from 1e-4)
    spd_kappa_max: float = 1e4          # Condition number cap
    spd_exp_clip: float = 5.0            # Eigenvalue exponent clip -- exp(+/-5) ~= [0.007, 148], max ratio ~2e4

    # Prior safeguards
    prior_sigma_floor: float = 0.5      # Min eigenvalue of prior Σ_v (prevents collapse)
    prior_mu_max_norm: float = 10.0     # Max L2 norm of prior μ_v (was 3.0, too tight -- init already ~=2.83)
    m_step_trust_mu: float = 0.5        # Trust region for M-step μ updates

    # Gauge frame parameterization
    gauge_param: str = 'omega'          # 'omega' (direct GL(K), both det signs) or 'phi' (Lie algebra -> GL⁺(K))
    omega_cond_max: float = 50.0        # Max condition number for Omega (regularize toward polar factor)
    omega_grad_clamp: float = 10.0      # Element-wise clamp for omega gradients (tighter than general grad_clamp)
    omega_negative_det_fraction: float = 0.0  # Fraction of omega frames initialized in GL⁻(K) (det < 0)
    phi_max_norm: Optional[float] = None  # Max phi norm; None = auto (π for SO(N), 5.0 for GL(K))


    # Observation gradient options
    sigma_obs_grad: str = 'none'        # 'none' (match VFE dynamic), 'diagonal', 'full'

    # Covariance mode
    diagonal_covariance: bool = False   # Use diagonal Σ (K,) instead of full (K, K) -- faster, less expressive

    # LayerNorm (optional, for testing -- this IS a neural-like component)
    use_layernorm: bool = False         # Apply LayerNorm to μ between E-step iterations

    # Holonomy / non-flat transport
    use_holonomy: bool = False          # Enable holonomy monitoring (measure gauge curvature)

    # Overfitting prevention
    obs_norm_floor: int = 0             # Floor for per-token obs gradient normalization (0 = auto: 1% of BN)
    rare_token_reg: float = 0.0         # Frequency-adaptive hyper-prior strength (0 = disabled)
    alpha_floor: float = 0.01           # Minimum prior precision α (prevents drift runaway in multi-step E-step)
    decode_tau: float = 1.0             # Temperature for KL-decode logits (>1 softens predictions)

    # Recovery
    nan_recovery: bool = True           # Reset beliefs to priors on NaN detection

    # Attention masking and priors (manuscript §3.3.4)
    causal: bool = True
    mask_self_attention: bool = False    # Mask KL diagonal to prevent self-attention collapse
    alibi_slope: float = 0.0            # ALiBi slope m (0 = uniform prior, >0 = recency bias)

    # Rotary Position Embeddings (RoPE)
    # SO(2)^{K/2} rotations applied to μ before KL scoring for attention β.
    # Raw (un-rotated) μ used for gradient computation.
    use_rope: bool = False              # Enable RoPE position encoding
    rope_base: float = 10000.0          # RoPE frequency base

    # M-step momentum (Adam-like natural gradient)
    # Tracks EMA of natural gradient first/second moments for variance reduction.
    # No neural components -- purely an optimization algorithm improvement.
    use_adam_m_step: bool = False        # Enable Adam-like momentum in M-step
    adam_beta1: float = 0.9             # First moment decay (momentum)
    adam_beta2: float = 0.999           # Second moment decay (adaptive scaling)
    adam_eps: float = 1e-8              # Denominator epsilon for numerical stability

    # LR scheduling (applies to all 5 per-variable learning rates)
    warmup_steps: int = 0               # Linear warmup (0 = no warmup)
    lr_schedule: str = 'constant'       # 'constant', 'cosine'
    min_lr_ratio: float = 0.1           # Floor for cosine decay as fraction of base LR
    max_steps: int = 30000              # Total training steps (for cosine schedule)

    # Gradient accumulation
    # Accumulates M-step sufficient statistics over K micro-batches before
    # applying one M-step update. E-step runs per micro-batch (beliefs are
    # sequence-local), but the M-step gradient is a sum over all data --
    # accumulation gives a K-times-lower-variance estimate of the true M-step.
    grad_accum_steps: int = 1           # Number of micro-batches per M-step (1 = no accumulation)

    # Device
    device: str = "cuda"
    use_cuda_kernels: bool = True       # Use custom CUDA kernels when available

    def __post_init__(self):
        assert self.belief_dim == self.n_heads * self.head_dim, \
            f"belief_dim ({self.belief_dim}) must equal n_heads * head_dim ({self.n_heads}*{self.head_dim})"
        if self.tau is None:
            self.tau = self.head_dim ** 0.5
