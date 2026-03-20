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
    eta_M: float = 0.001       # M-step natural gradient step size

    # Prior precision (state-dependent α)
    alpha_b0: float = 1.0      # Denominator offset
    alpha_c0: float = 1.0      # Numerator scale

    # Hyper-prior regularization
    hyper_var: float = 1.0     # Variance of hyper-prior on prior means

    # Sequence
    max_seq_len: int = 64      # N: maximum sequence length
    batch_size: int = 8

    # Initialization
    sigma_init: float = 1.0             # Initial covariance scale
    omega_init_scale: float = 0.01      # GL(K) frame perturbation from I

    # Trust regions (Frobenius norm caps on natural gradient steps)
    trust_region_mu: float = 1.0
    trust_region_sigma: float = 0.3
    trust_region_omega: float = 0.3

    # SPD retraction safeguards
    spd_eps_min: float = 1e-4           # Spectral floor
    spd_kappa_max: float = 1e4          # Condition number cap
    spd_exp_clip: float = 50.0          # Eigenvalue exponent clip

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
