"""Configuration for the Pure VFE Transformer."""

from dataclasses import dataclass


@dataclass
class PureVFEConfig:
    # Vocabulary
    vocab_size: int = 50257

    # Belief geometry
    belief_dim: int = 32       # K: full belief dimension
    n_heads: int = 4           # H: number of heads (block-diagonal Σ)
    head_dim: int = 8          # K_h = K / H per head

    # E-step (inference = forward pass)
    n_esteps: int = 12         # Iterations of VFE descent (replaces "depth")
    tau: float = None           # Attention temperature (defaults to √K_h)
    eta_E: float = 0.1         # E-step natural gradient step size

    # M-step (learning = parameter update)
    # Must be comparable to VFE_dynamic's mu_lr=0.05 for priors to move.
    # Was 0.001, which with /BN and confidence weighting gave ~2e-8 per scalar.
    eta_M: float = 0.05        # M-step natural gradient step size

    # Prior precision (state-dependent α)
    alpha_b0: float = 1.0      # Denominator offset
    alpha_c0: float = 1.0      # Numerator scale

    # Hyper-prior regularization — must be large enough that the hyper-prior
    # doesn't overpower the observation gradient.  At hyper_var=1 the pull-to-zero
    # force (|μ_v|≈2.8) dwarfs the CE signal; 100 weakens it to ~0.03.
    hyper_var: float = 100.0   # Variance of hyper-prior on prior means

    # Sequence
    max_seq_len: int = 64      # N: maximum sequence length
    batch_size: int = 8

    # Initialization
    sigma_init: float = 1.0             # Initial covariance scale
    omega_init_scale: float = 0.01      # GL(K) frame perturbation from I

    # E-step numerical stability (ported from VFE dynamic)
    sigma_lr_ratio: float = 0.05        # Sigma evolves at this × mu rate (VFE dynamic: 0.05)
    e_step_lr_decay: float = 0.5        # LR decays to (1 - this) × eta_E over E-step iterations
    grad_clamp: float = 1e3             # Element-wise gradient clamp before natural gradient

    # Trust regions (Frobenius norm caps on natural gradient steps)
    trust_region_mu: float = 2.0        # Whitened trust region for mu (VFE dynamic: 2.0)
    trust_region_sigma: float = 0.15    # Trust region for sigma retraction (tightened from 0.3)
    trust_region_omega: float = 0.3     # Relative trust region for omega updates

    # SPD retraction safeguards
    spd_eps_min: float = 1e-3           # Spectral floor (tightened from 1e-4)
    spd_kappa_max: float = 1e4          # Condition number cap
    spd_exp_clip: float = 20.0          # Eigenvalue exponent clip (tightened from 50.0)

    # Prior safeguards
    prior_sigma_floor: float = 0.5      # Min eigenvalue of prior Σ_v (prevents collapse)
    prior_mu_max_norm: float = 10.0     # Max L2 norm of prior μ_v (was 3.0, too tight — init already ≈2.83)
    m_step_trust_mu: float = 0.5        # Trust region for M-step μ updates

    # Gauge frame parameterization
    gauge_param: str = 'omega'          # 'omega' (direct GL(K)) or 'phi' (Lie algebra)
    omega_cond_max: float = 50.0        # Max condition number for Omega (regularize if exceeded)
    omega_grad_clamp: float = 10.0      # Element-wise clamp for omega gradients (tighter than general grad_clamp)
    omega_nat_max_norm: float = 1.0     # Absolute Frobenius norm cap on omega natural gradient
    phi_max_norm: float = 3.14159       # Max norm for phi (π = 180° rotation)

    # M-step options
    sigma_obs_grad: str = 'none'        # 'none' (match VFE dynamic), 'diagonal', 'full'
    m_step_eta_floor: float = 0.01      # Min multiplier for confidence-weighted eta_M

    # Recovery
    nan_recovery: bool = True           # Reset beliefs to priors on NaN detection

    # Causal masking
    causal: bool = True

    # Device
    device: str = "cuda"
    use_cuda_kernels: bool = True       # Use custom CUDA kernels when available

    def __post_init__(self):
        assert self.belief_dim == self.n_heads * self.head_dim, \
            f"belief_dim ({self.belief_dim}) must equal n_heads * head_dim ({self.n_heads}*{self.head_dim})"
        if self.tau is None:
            self.tau = self.head_dim ** 0.5
