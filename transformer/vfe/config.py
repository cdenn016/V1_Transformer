"""
VFEConfig: clean configuration for the gauge-theoretic VFE transformer.

~25 fields in 5 semantic groups. No EM mode branching, no DEQ/hebbian/closed-form.
Only the iterative natural-gradient E-step with straight_through gradient flow.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import torch

from transformer.core.block_config import (
    RopeFullGaugeMode,
    _ROPE_FULL_GAUGE_VALUES,
)


@dataclass
class VFEConfig:
    r"""Configuration for the VFE transformer.

    All parameters needed to construct a :class:`VFEModel`. Compared to
    ``BlockConfig`` (~87 fields), this has ~25 fields with no mode branching.

    The ``use_obs_in_vfe`` flag does not exist because the E-step has no
    observation term — Law 1 is architecturally enforced.
    """

    # === Structure ===
    vocab_size: int = 50257
    embed_dim: int = 64                  # K: total belief dimension
    irrep_spec: List[Tuple[str, int, int]] = field(
        default_factory=lambda: [('l0', 8, 8)]  # 8 heads x dim 8 = K=64
    )
    n_layers: int = 4                    # L: number of VFE blocks
    max_seq_len: int = 128

    # === E-step dynamics ===
    n_e_steps: int = 3                   # T_E: inner loop iterations per layer
    e_mu_lr: float = 0.1                 # eta_mu: mean natural gradient step size
    e_sigma_lr: float = 0.001            # eta_sigma: covariance retraction step size
    e_phi_lr: float = 0.05              # eta_phi: gauge frame step size
    alpha: float = 1.0                   # KL(q||p) prior self-coupling weight
    alpha_divergence: float = 1.0        # Rényi α (1.0=KL, 0.5=Bhattacharyya)
    E_learnable_alpha: bool = False      # Bayesian adaptive α_i = c0/(b0+KL) per dim
    lambda_align: float = 1.0            # Boltzmann GLU weight (beta * grad_KL)
    lambda_soft: float = 1.0             # Attention-variance coupling (KL * grad_beta)
    kappa: float = 1.0                   # Attention temperature
    prior_handoff_rho: float = 1.0       # μ cross-layer damping (1.0 = no damping, <1 = smoother)
    prior_handoff_sigma: float = 0.0    # σ cross-layer handoff (0.0 = frozen at embedding, >0 = blends posterior)
    prior_handoff_phi: bool = False     # Deprecated no-op: priors.phi is never consumed by VFEEStep (phi flows via beliefs)
    learnable_kappa: bool = False        # Learn per-layer kappa via log-space parameter
    include_attention_entropy: bool = True  # Add κ·Σβ·log(β/π) to alignment_loss (manuscript eq:free_energy_functional_final).
                                            # Defaults ON for theoretical correctness; disable to recover
                                            # entropy-suppressed surrogate behavior (β.detach used so envelope
                                            # theorem gives zero (μ,Σ,φ)-gradient at softmax fixed point; live
                                            # κ multiplier gives correct -H(β) gradient for learnable_kappa).

    # === Covariance ===
    # NOTE: The diagonal GL(K) regime is an approximation, not exact gauge-Gaussian
    # transport. diag(Ω diag(σ) Ωᵀ) ≠ Ω diag(σ) Ωᵀ for non-orthogonal Ω.
    # Set exact_diagonal_transport=True to lift to full for exact transport
    # (more compute, mathematically exact). For SO(K), diagonal is lossless.
    diagonal_covariance: bool = True     # True = (B,N,K), False = (B,N,K,K)
    isotropic_covariance: bool = False   # If True, force Σ = σ²I (scalar variance × identity)
    exact_diagonal_transport: bool = True   # Lift diagonal σ to full for exact Ω@Σ@Ω^T transport
    sigma_max: float = 5.0              # Upper bound on sigma
    sigma_init: float = 1.0             # Initial covariance scale
    sigma_aggregation: str = 'mixture'   # 'mixture' or 'precision' (for aggregate_beliefs utility; not used in gradient-based E-step)

    # === Gauge geometry ===
    gauge_group: str = 'GLK'             # 'SO3', 'SON', 'GLK'
    phi_preconditioner: str = 'killing'  # 'clip', 'cartan', 'killing', 'pullback'
    # GL(K) determinant control (no-op for SO(N), tr(G)=0 already). Pick at most one:
    phi_project_slk: bool = False        # Hard project φ → sl(K) ⇒ det(Ω) ≡ 1
    phi_trace_clamp: Optional[float] = None  # Soft cap |tr(φ·G)| ≤ T ⇒ det(Ω_ij) ∈ [exp(-2T), exp(2T)]
    enforce_orthogonal: bool = False     # Project Omega to SO(K)
    mask_self_attention: bool = True     # Mask diagonal in attention (prevents KL=0 collapse)
    mass_phi: float = 0.0               # Gauge prior: (mass_φ/2)||φ||²

    # Gauge-covariant numerical ridge. When True, ε·I regularizers on
    # sandwich-transforming covariances Σ are replaced by ε·(gg^T) built from
    # the local frame g = exp(φ), preserving Σ → hΣh^T covariance exactly.
    # Default False preserves bitwise numerics.
    gauge_covariant_ridge: bool = False

    # === Positional encoding ===
    # NOTE: RoPE currently rotates only μ, not Σ. The framework-consistent gauge
    # action also rotates Σ via RΣRᵀ (rope_full_gauge != 'off'). This is a known
    # theory gap documented in VFE_Transformer_Idea.md. Non-'off' values require
    # full covariance and are rejected at __post_init__ under diagonal σ.
    use_rope: bool = True
    rope_base: float = 10000.0
    rope_full_gauge: RopeFullGaugeMode = 'off'  # 'off' | 'vfe_only' | 'both'. vfe/ currently
                                                # treats any non-'off' value identically; tri-state
                                                # differentiation is reserved for a future change.
    bch_order: int = 1                   # BCH truncation order (1=additive)

    # === Embedding init ===
    mu_init_std: float = 1.0             # Std for base_mu initialization
    phi_scale: float = 0.1              # Scale for phi_embed initialization

    # === Active inference (off by default) ===
    active_inference: bool = False
    pragmatic_weight: float = 1.0
    epistemic_weight: float = 0.5
    epistemic_samples: int = 4
    decode_tau: float = 1.0              # Temperature for PriorBank.decode in AI
    # NOTE: learnable_decode_tau was removed because the AI gradient architecture
    # computes gradients inside a fresh autograd graph with detached mu/sigma leaves.
    # tau doesn't enter this graph, so gradients cannot flow to a learnable tau.
    # Making it truly learnable requires restructuring the AI gradient computation.

    # === Normalization ===
    norm_type: str = 'mahalnorm'         # 'mahalnorm', 'centered_mahalnorm', 'rmsnorm', 'layernorm' (gauge-blind, ablation-only), 'none'

    # === Training ===
    learning_rate: float = 3e-4
    weight_decay: float = 0.05
    warmup_steps: int = 1000
    max_steps: int = 50000
    batch_size: int = 8
    grad_clip: float = 1.0
    normalize_ce_by_dim: bool = False    # Divide CE by sqrt(K) for dim-independent scaling

    # === Logging / evaluation ===
    log_interval: int = 100
    eval_interval: int = 3000
    checkpoint_interval: int = 25000
    semantic_analysis_interval: int = 10000
    gauge_geometry_interval: int = 5000
    fiber_trajectory_interval: int = 5000

    # === Diagnostics ===
    track_layer_diagnostics: bool = False
    track_iteration_diagnostics: bool = False
    diagnostics_interval: int = 25

    # === Runtime (set after construction, not serialized) ===
    generators: Optional[torch.Tensor] = field(default=None, repr=False)
    device: str = 'cpu'

    # --- Derived (computed in __post_init__) ---

    def __post_init__(self) -> None:
        """Compute derived fields from irrep_spec."""
        # Validate embed_dim matches irrep_spec
        computed_dim = sum(mult * dim for _, mult, dim in self.irrep_spec)
        if computed_dim != self.embed_dim:
            raise ValueError(
                f"irrep_spec gives K={computed_dim} but embed_dim={self.embed_dim}. "
                f"These must match."
            )
        if self.prior_handoff_phi:
            import warnings
            warnings.warn(
                "VFEConfig.prior_handoff_phi is a deprecated no-op: priors.phi "
                "is not consumed by VFEEStep; phi already flows across layers "
                "via the belief state. Setting this flag has no effect.",
                DeprecationWarning,
                stacklevel=2,
            )
        if self.rope_full_gauge not in _ROPE_FULL_GAUGE_VALUES:
            raise ValueError(
                f"rope_full_gauge must be one of {_ROPE_FULL_GAUGE_VALUES}; "
                f"got {self.rope_full_gauge!r}."
            )
        if self.rope_full_gauge != 'off' and self.diagonal_covariance:
            raise ValueError(
                f"rope_full_gauge={self.rope_full_gauge!r} requires "
                f"diagonal_covariance=False. RoPE gauge transport acts on Σ "
                f"via RΣR^T, which needs full covariance."
            )
        # Full-cov + RoPE without rope_full_gauge is a gauge-inconsistent
        # configuration (μ rotated, Σ not). We reject it explicitly rather
        # than silently promoting the flag, which would be a checkpoint
        # round-trip hazard.
        if not self.diagonal_covariance and self.use_rope and self.rope_full_gauge == 'off':
            raise ValueError(
                "diagonal_covariance=False + use_rope=True requires "
                "rope_full_gauge='vfe_only' or 'both'. "
                "Silent promotion was removed; set the flag explicitly."
            )

    @property
    def irrep_dims(self) -> List[int]:
        """Flat list of block dimensions from irrep_spec.

        For irrep_spec = [('l0', 8, 8)]: returns [8, 8, 8, 8, 8, 8, 8, 8]
        For irrep_spec = [('l0', 75, 1), ('l1', 30, 3)]: returns [1]*75 + [3]*30
        """
        dims = []
        for _, mult, dim in self.irrep_spec:
            dims.extend([dim] * mult)
        return dims

    @property
    def n_gen(self) -> int:
        """Number of Lie algebra generators."""
        if self.generators is not None:
            return self.generators.shape[0]
        # Estimate from gauge group and irrep structure
        if self.gauge_group == 'SO3':
            return 3
        elif self.gauge_group == 'GLK':
            # Block-diagonal: sum of d_h^2 per head
            return sum(d ** 2 for d in self.irrep_dims)
        else:  # SON
            K = self.embed_dim
            return K * (K - 1) // 2
