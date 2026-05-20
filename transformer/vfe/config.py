"""
VFEConfig: clean configuration for the gauge-theoretic VFE transformer.

~25 fields in 5 semantic groups. No EM mode branching, no DEQ/hebbian/closed-form.
Only the iterative natural-gradient E-step with semi-gradient flow at the EM boundary.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Literal
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

    .. warning::
       ``__post_init__`` mutates ``use_autograd_mu_sigma`` when
       ``gauge_parameterization == 'omega_direct'`` or
       ``use_non_flat_transport == True`` (promotion path that the analytic
       gradient kernel cannot handle). It also dedups ``cross_couplings`` and
       computes the ``super_block_dims`` / ``super_block_head_groups`` caches
       on first access. ``dataclasses.replace(cfg, ...)`` does NOT re-run
       ``__post_init__``, so a replace that toggles any of these flags
       (``use_autograd_mu_sigma``, ``use_non_flat_transport``,
       ``gauge_parameterization``, ``cross_couplings``,
       ``auto_close_cross_head_basis``) will leave derived state stale or
       skip validation. Construct a fresh ``VFEConfig`` via the normal
       constructor instead of using ``replace`` when toggling those.

    .. note::
       **RoPE × cross_couplings.** RoPE rotations operate within each head's
       ``d_head`` slot (which the super-block reordering preserves head-
       contiguous), so the μ rotation itself is well-defined under cross
       coupling. However, the cross-head generators do *not* commute with
       per-head RoPE rotations, so the combined transport ``Ω · R`` and
       ``R · Ω`` differ — RoPE × cross_couplings has the same "rotated μ
       but un-rotated Σ" approximation as the existing single-head case
       documented in CLAUDE.md "KNOWN GAP — RoPE × MahalanobisNorm", and
       no worse. Strict joint covariance for cross_couplings would require
       ``rope_full_gauge != 'off'`` plus a derived per-pair handling of the
       cross generators that is not currently implemented; ``rope_full_gauge
       != 'off'`` already requires full covariance via existing
       ``__post_init__`` validation.
    """

    # === Structure ===
    vocab_size: int = 50257
    embed_dim: int = 64                  # K: total belief dimension
    irrep_spec: List[Tuple[str, int, int]] = field(
        default_factory=lambda: [('l0', 8, 8)]  # 8 heads x dim 8 = K=64
    )
    n_layers: int = 1                    # L: number of VFE blocks
    max_seq_len: int = 128

    # === E-step dynamics ===
    n_e_steps: int = 1                   # T_E: inner loop iterations per layer
    e_mu_lr: float = 0.1                 # eta_mu: mean natural gradient step size
    e_sigma_lr: float = 0.001            # eta_sigma: covariance retraction step size
    e_sigma_q_trust: float = 5.0         # Trust-region clamp on whitened |delta_sigma/sigma|
                                         # for the diagonal-sigma retraction. Independent of
                                         # the step LR (CLAUDE.md, 2026-05-13 onward).
    e_mu_q_trust: Optional[float] = None
    # r"""Per-component sigma-whitened trust-region clamp on the mean update.
    # None (default) — no clamping; pure path bitwise unchanged.
    # float > 0     — clamp |delta_mu_k / sqrt(sigma_q_kk)| <= e_mu_q_trust
    #                 element-wise, matching the sigma retraction's whitened
    #                 inf-norm form (vfe_utils.retract_spd_diagonal_torch).
    # Applied in transformer/vfe/e_step.py just before the mu retraction in
    # both the phi-mode and omega-direct E-step loops. Whitening uses
    # diag(Sigma_q) under full covariance, matching the diagonal-only
    # treatment of the sigma trust region for full-cov inputs."""
    e_nan_check: Literal['off', 'warn', 'revert', 'abort'] = 'off'
    # r"""NaN / Inf sentinel policy for the E-step inner loop.
    # 'off'    (default) — no checking; pure path with zero overhead.
    # 'warn'   — log a RuntimeWarning on detection; iteration continues
    #            from the corrupt state (diagnostic only).
    # 'revert' — restore the pre-iteration snapshot of (mu, sigma, phi)
    #            and break the loop on detection.
    # 'abort'  — raise transformer.vfe._numerics.VFENonFiniteError on
    #            detection, carrying the checkpoint label and field name.
    # Snapshot memory is allocated only when mode in {'revert', 'abort'};
    # 'warn' and 'off' pay zero memory overhead. Each check forces a host
    # sync via .all().item() — enable only when diagnosing instability."""
    e_phi_lr: float = 0.05              # eta_phi: gauge frame step size
    
    alpha: float = 1.0                   # KL(q||p) prior self-coupling weight
    alpha_divergence: float = 1.0        # Rényi α (1.0=KL, 0.5=Bhattacharyya)
    E_learnable_alpha: bool = False      # Bayesian adaptive α_i = c0/(b0+KL) per dim
    lambda_align: float = 1.0            # Boltzmann GLU weight (beta * grad_KL)
    lambda_soft: float = 1.0             # Attention-variance coupling (KL * grad_beta)
    kappa: float = 1.0                   # Attention temperature
    # per_head_softmax: routing convention for multi-head (H > 1) attention.
    # True (default, canonical per manuscript eq:per_head_temperature at
    #   Attention/GL(K)_attention.tex:1744): one softmax per gauge head at
    #   τ_h = κ · √d_head; per-head βs accumulated for both gradients and
    #   plotting. Matches the reduced gauge group GL(d_head)^H ⊂ GL(K) the
    #   multi-head decomposition prescribes and the legacy
    #   transformer/core/variational_ffn.py pattern at lines 1997-2014.
    # False: single global softmax over Σ_h KL_h at τ = κ · √K. Canonical
    #   only in the single-head limit (H = 1). Retained as opt-in for
    #   ablation against the multi-head reduction and for backward
    #   compatibility with /vfe runs from before 2026-05-19.
    # Takes effect only in the fused path (block-diagonal Σ, irrep_dims
    # set, exact_diagonal_transport=False, use_autograd_mu_sigma=False);
    # other paths run their existing single-call dispatch.
    per_head_softmax: bool = True
    prior_handoff_rho: float = 1.0       # μ cross-layer damping (1.0 = no damping, <1 = smoother)
    prior_handoff_sigma: float = 0.0    # σ cross-layer handoff (0.0 = frozen at embedding, >0 = blends posterior)
    learnable_kappa: bool = False        # Learn per-layer kappa via log-space parameter
    # Per-head learnable κ_h. When True, replaces the scalar log_kappa
    # parameter with a per-head vector log_kappa_per_head of shape (H,),
    # giving each gauge head its own softmax temperature
    # τ_h = exp(log_kappa_per_head[h]).clamp(0.5·κ₀, 2·κ₀) · √d_head.
    # Matches the manuscript's eq:per_head_temperature
    # (Attention/GL(K)_attention.tex:1744) verbatim. Requires
    # learnable_kappa=True AND per_head_softmax=True AND
    # len(irrep_dims) > 1 (all enforced by __post_init__).
    kappa_per_head: bool = False
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
    exact_diagonal_transport: bool = False   # Lift diagonal σ to full for exact Ω@Σ@Ω^T transport
    sigma_max: float = 5.0              # Upper bound on sigma
    sigma_init: float = 1.0             # Initial covariance scale

    # === Gauge geometry ===
    gauge_group: Literal['SO3', 'SON', 'GLK'] = 'GLK'
    phi_preconditioner: Literal[
        'clip', 'cartan', 'killing', 'killing_per_block'
    ] = 'killing'
    # 'killing'           — ambient gl(K_full) Killing form on the restricted
    #                       subalgebra (cross-block coupling via -2·tr⊗tr).
    # 'killing_per_block' — direct-sum Killing form on
    #                       gl(d_1) ⊕ ... ⊕ gl(d_H); block-diagonal in the
    #                       generator index, no cross-block coupling. Added
    #                       2026-05-19 to address audit-2026-05-18-v4 F6.1.
    # 'pullback' is intentionally absent from the inner E-step preconditioner
    # path: the upstream call chain in `_update_phi` does not thread the
    # `structure_constants` tensor that `apply_pullback_natural_gradient`
    # requires, so selecting it raised TypeError at runtime. The metric math
    # in `build_pullback_metric_tensor` was corrected on 2026-05-20 (the
    # position-dependent factor `exp(phi) exp(phi)^T` is now included for
    # non-compact gl(K) directions; see supplementary §C.5 debate verdict).
    # The corrected metric is already in use via the outer optimizer
    # `RiemannianAdamW(metric='pullback')`, which threads structure_constants
    # itself. Re-enable here only after wiring structure_constants through
    # `VFEEStep.__init__` and `precondition_phi_gradient`.
    # GL(K) determinant control (no-op for SO(N), tr(G)=0 already). Pick at most one:
    phi_project_slk: bool = False        # Hard project φ → sl(K) ⇒ det(Ω) ≡ 1
    phi_trace_clamp: Optional[float] = None  # Soft cap |tr(φ·G)| ≤ T ⇒ det(Ω_ij) ∈ [exp(-2T), exp(2T)]
    phi_spec_max: Optional[float] = None
    # r"""Pre-exp Frobenius rescale on the algebra tensor before matrix_exp.
    # None (default) — no clamp; pure path bitwise unchanged.
    # float > 0     — rescale ``algebra <- algebra * min(1, phi_spec_max /
    #                 ||algebra||_F)`` per (B, N, N, head) slice before
    #                 :func:`torch.linalg.matrix_exp` at the pairwise-Omega
    #                 construction sites in :mod:`transformer.vfe.omega_direct`
    #                 and :mod:`transformer.vfe.non_flat`. Bounds the
    #                 condition number of the sandwich transport
    #                 ``Sigma -> Omega Sigma Omega^T`` since
    #                 ``||exp(X)||_2 <= exp(||X||_F)`` (Hall, *Lie Groups, Lie
    #                 Algebras, and Representations*).
    # Caveat: clamping the algebra without writing back to stored phi
    # temporarily breaks the phi <-> Omega consistency on iterations where
    # the clamp binds. Acceptable as opt-in defensive layer; not the
    # default. Field name uses ``spec`` (spectral) as colloquial shorthand
    # for the Frobenius cap; matches the existing ``phi_*`` namespace."""
    phi_strict_glplus: bool = False
    # r"""When True, :func:`transformer.vfe.omega_direct.project_omega_to_slk`
    # raises :class:`transformer.vfe.omega_direct.VFEGaugeOrientationError`
    # on detecting a per-block ``det(Omega) < 0`` at the periodic
    # renormalization step.
    # False (default) — preserve the sign factor in the rescale (current
    # behavior; consistent with the manuscript's GL(K) gauge group, which
    # contains both orientation components ``Attention/GL(K)_attention.tex``
    # line 432 and the reflection extension at lines 1096-1099).
    # Set True to enforce the GL+(K) identity-component contract — useful
    # for ablations that should never reach the reflection branch."""
    enforce_orthogonal: bool = False     # Project Omega to SO(K)
    mask_self_attention: bool = False    # Default: self-attention allowed. Per-user 2026-05-17:
                                         # unmasking matches the active train_vfe.py configs and
                                         # keeps row-0 well-defined under causal masking without
                                         # the "fully-masked row" edge case. Set True to revert
                                         # to the previous KL=0-collapse-suppression behavior.
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
    rope_base: float = 100.0
    rope_full_gauge: RopeFullGaugeMode = 'off'  # 'off' | 'vfe_only' | 'both'. vfe/ currently
                                                # treats any non-'off' value identically; tri-state
                                                # differentiation is reserved for a future change.
    bch_order: int = 4                   # BCH truncation order (1=additive)

    # === Embedding init ===
    mu_init_std: float = 1.0             # Std for base_mu / mu_embed initialization
    phi_scale: float = 0.1              # Scale for phi_embed initialization

    # === Prior parameterization ===
    # True: token priors are gauge transforms of a single universal Gaussian
    #   (μ_v = A_v μ_0, Σ_v = A_v diag(σ_0) A_v^T). Per-token capacity = phi_embed
    #   only (V × n_gen). Maximally pure: every token is one point on a single
    #   gauge orbit.
    # False: each token has its own Gaussian prior (μ_v, σ_v) looked up directly;
    #   phi is retained as the per-token gauge frame for cross-token transport
    #   only, not for prior construction. Per-token capacity = V × (2K + n_gen).
    gauge_fixed_priors: bool = True

    # === Decode and generation-time EFE (canonical path) ===
    # decode_tau is the *fixed* component of the decode softmax temperature
    # passed to PriorBank.decode (training-time forward at model.py:209; EFE
    # candidate scoring in vfe/efe.py).  The full decode temperature is
    # learnable:
    #     tau = decode_tau * exp(-PriorBank.decode_log_scale)
    # where decode_log_scale is an nn.Parameter clamped to [-3, 3], so the
    # effective tau spans [decode_tau * exp(-3), decode_tau * exp(3)]
    # ~ [0.05, 20] at decode_tau=1.0.  This mirrors the attention temperature
    # tau_attn = kappa * sqrt(K) where kappa is learnable; here the
    # learnable scalar is exp(-decode_log_scale) and the fixed scaling is
    # decode_tau.  epistemic_weight / epistemic_samples are consumed by
    # VFEExpectedFreeEnergy at generation; the canonical /aif module carries
    # its own copies in AIFConfig for the multi-step build-out.
    epistemic_weight: float = 0.5
    epistemic_samples: int = 4
    decode_tau: float = 1.0

    # === Normalization ===
    # 'layernorm' is gauge-blind (ablation-only); 'none' disables normalization.
    norm_type: Literal['mahalnorm', 'centered_mahalnorm', 'rmsnorm', 'layernorm', 'none'] = 'mahalnorm'

    # === Gauge parameterization ===
    # 'phi' (default): per-token φ ∈ g is the state; Ω is recomputed via
    #     exp(φ·G) at every use. Updates flow as Killing-preconditioned
    #     natural gradient on the Lie algebra.
    # 'omega_direct': per-token Ω ∈ G is the state, stored as a per-block
    #     (Ω_h, Ω_h^{-1}) pair on BeliefState.omega. Updates use the
    #     right-invariant Riemannian gradient on the group manifold:
    #         Ω_new = Ω · exp(-η · X_proj),  X = proj_span(G^a)(Ω^{-1} · dF/dΩ)
    #     The proj_span step preserves block-diagonal structure; the Killing
    #     form on R·I ⊂ gl(K) is degenerate, so omega_normalize_every > 0
    #     should be used to periodically rescale det(Ω_h) = 1 in GL(K).
    # Validation: omega_direct currently requires diagonal_covariance=True,
    # rope_full_gauge='off', and use_autograd_mu_sigma=True (the analytic
    # gradient kernel is φ-only). __post_init__ enforces all three.
    gauge_parameterization: Literal['phi', 'omega_direct'] = 'phi'
    omega_normalize_every: int = 0     # 0 = never; >0 = renormalize det every N E-steps
    omega_project_slk: bool = False    # True = renormalize per-block det → 1 (controls trace drift on R·I)

    # === Cross-head coupling (GL(K) multi-head only) ===
    # Sparse off-diagonal gauge mixing between heads. Each (a, b) pair adds
    # d_head² generators that span the head-a → head-b subspace; the resulting
    # Ω = exp(φ·G) acquires non-zero blocks at (a, b) and (b, a), so gauge
    # transport itself moves information between heads (different mechanism
    # from VFEHeadMixer, which is a fixed post-E-step value mixer).
    #
    # The math kernel lives in math_utils.generators (shared with legacy
    # transformer/core/model.py):
    #   - generate_glK_cross_head_generators: emits diag + cross generators
    #   - merge_coupled_heads: union-find on coupling pairs → super-block dims
    #   - reorder_cross_head_generators: permutes basis so super-blocks are
    #     contiguous in the K dimension
    #
    # Constraints:
    #   1. gauge_group must be 'GLK' (cross-head coupling is GL(K)-specific).
    #   2. irrep_spec must be single-type multihead (one (type, n_heads,
    #      d_head) entry with n_heads >= 2). Each head must have the same
    #      d_head — heterogeneous-dim heads are not supported.
    #   3. Pairs (a, b) with a == b are rejected. Distinct orientations
    #      (a, b) vs (b, a) are preserved (directional generators).
    #   4. Exact duplicate pairs are silently de-duplicated.
    cross_couplings: List[Tuple[int, int]] = field(default_factory=list)
    # When True, after building the cross-head basis, close it under the
    # matrix commutator [.,.] so it forms a true Lie subalgebra. Required for
    # mathematically exact BCH composition (otherwise the bracket silently
    # projects onto the open basis, producing an approximation). Costs:
    # changes phi_dim, breaks checkpoint compatibility, can introduce
    # generators that cross the user-supplied super-block partition.
    auto_close_cross_head_basis: bool = False
    # When True (default), validate the cross-head basis is closed under
    # [.,.] and emit a non-fatal warning if it is not. Disable to suppress
    # the warning if you have verified the basis or are using an approximate
    # BCH composition deliberately.
    validate_cross_head_closure: bool = True

    # === Non-flat parallel transport ===
    # When True, transport becomes Ω_ij = exp(φ_i·G) · exp(δ_ij·G) · exp(-φ_j·G)
    # with a per-edge connection δ_ij computed by an antisymmetric bilinear
    # form on (μ_i, μ_j); see transformer/vfe/non_flat.py for the full math.
    # Holonomy around closed loops is non-trivial when the connection learns
    # away from zero. All gates start at zero — fresh model with the flag on
    # is bitwise-equivalent to the flag-off path at step 0.
    #
    # Constraints when use_non_flat_transport=True:
    #   1. use_autograd_mu_sigma is forced to True (the analytic gradient
    #      kernel compute_vfe_gradients_gpu does not support non-flat
    #      transport). __post_init__ promotes it and warns if the user set it
    #      to False.
    #   2. rope_full_gauge must be 'off'. RoPE × non-flat × per-pair Σ is a
    #      coupling we have not yet derived; __post_init__ rejects it.
    #   3. diagonal_covariance must be True. Full-cov pairwise KL needs a
    #      per-pair logdet path that is not yet implemented; __post_init__
    #      rejects it.
    use_non_flat_transport: bool = False
    non_flat_max_strength: float = 1.0       # s_max in s = s_max·tanh(ρ)
    non_flat_per_edge_delta_max: float = 1.0  # δ_max bound on ‖δ_ij·G‖_F
    non_flat_tile_size: int = 0               # Reserved for future j-axis chunked aggregation. The tiled path is not yet implemented; non-zero values are silently treated as 0 today.

    # === Head mixer ===
    # Schur-commutant per-irrep-type mixer applied after the E-step (and
    # before normalization) inside every VFEBlock. Per type t with multiplicity
    # n_t and dim d_t the parameter is A_t ∈ R^{n_t × n_t}; the action is
    # μ ↦ kron(A_t, I_{d_t}) · μ and Σ ↦ M · Σ · M^T applied symmetrically.
    # Initialized at identity (mixer_delta=0), so a fresh model with the
    # mixer enabled is bitwise-equivalent to the mixer-disabled path at step 0.
    # The mixer is strictly gauge-equivariant only under tied per-token
    # gauges; /vfe satisfies this by construction (one φ_i per token, shared
    # across heads via block-diagonal generators) — see CLAUDE.md.
    use_equivariant_head_mixer: bool = False

    # === Decode head ===
    # True (default): PriorBank.decode computes logits = -KL(q || π_v) / τ,
    #   reusing the same gauge-orbit prior used for encode (Law 3 — same
    #   manifold for encode/infer/decode, exact when diagonal_covariance=True;
    #   the full-covariance branch projects to diagonal at decode for O(V·K)
    #   efficiency, see prior_bank.py:478-479).  τ here is the learnable
    #   decode softmax temperature τ = decode_tau · exp(-decode_log_scale),
    #   one nn.Parameter scalar (decode_log_scale, clamped to [-3, 3]).
    #   No nn.Linear / MLP / activation on this path.
    # False: replace the decode head with a plain nn.Linear(K, V) projection
    #   on mu_final only (sigma is ignored). This is the one documented
    #   "neural" exception in CLAUDE.md ("The only retained neural component
    #   is a linear output projection from K dimensions to vocabulary size").
    #   Encode still uses the gauge-orbit PriorBank — the toggle controls
    #   only the decode side.
    use_prior_bank: bool = False

    # === Exact full-covariance decode (Law 3 pure path) ===
    # When True, replace the diagonal projection at decode (prior_bank.py:494)
    # with the exact per-block KL between block-diagonal Gaussians. The decode
    # then matches encode/E-step on the full block-diagonal Gaussian manifold,
    # closing the Law-3 partial documented for the
    # (gauge_fixed_priors=True, diagonal_covariance=False) combination.
    # Cost is O(B*N*V*Σ_h d_h²) per forward — roughly 10–20× the diagonal
    # decode at K=100, 5 blocks of d_h=20 — but a single Cholesky per (v,h)
    # is cached for the forward.
    # Validation (__post_init__): requires gauge_fixed_priors=True (otherwise
    # Σ_p is structurally diagonal-embedded and the flag does no work),
    # diagonal_covariance=False (otherwise Σ_p is a (K,) tensor and there is
    # no off-diagonal mass to recover), and use_prior_bank=True (otherwise
    # the decode is nn.Linear). Defaults to False.
    exact_full_cov_decode: bool = False

    # === Training ===
    # Per-group M-step (outer AdamW) learning rates. Mirrors the e_*_lr style
    # of the E-step. Each LR applies to the param group whose names match the
    # pattern in trainer.py::_build_optimizer; all groups share the same
    # cosine+warmup schedule (LambdaLR, trainer.py::_build_scheduler), so the
    # ratios stay fixed across training and the absolute values cosine-decay
    # together.
    m_mu_lr: float = 1e-3        # PriorBank.base_mu
    m_sigma_lr: float = 5e-5     # PriorBank.base_log_sigma, decode_log_scale
    m_phi_lr: float = 1e-4       # PriorBank.phi_embed, Positional.pos_phi
    m_hyper_lr: float = 1e-5     # E-step learnable hyperparams: raw_c0, raw_b0,
                                 # log_kappa, _phi_preconditioner
    m_other_lr: float = 3e-4     # Everything else (head_mixer, non_flat W_raw,
                                 # output projection, ...). Also the AdamW
                                 # default lr (harmless: every group sets its own).
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

    # === Diagnostics ===
    track_layer_diagnostics: bool = False
    # F-monotone monitor: when True, the E-step records F at every iter and
    # warns on non-monotone descent. Fires .item() CUDA syncs per iter — leave
    # off for production training. See e_step.py around the f_history block.
    monitor_monotonicity: bool = False

    # === E-step gradient kernel selection ===
    # When True, route (mu, sigma) E-step gradient through torch.autograd.grad
    # over the full F functional (manuscript eq:free_energy_functional_final),
    # capturing BOTH query-side and key-side contributions to dF/dmu_k.
    # The analytic kernel compute_vfe_gradients_gpu (default, False) is the
    # mean-field convention: it returns the query-side partial only, missing
    # the key-side term arising when mu_k appears as the transport key in
    # KL(q_i || Omega_ik q_k) for i != k. The autograd path matches the
    # phi-update mechanism at e_step.py:_update_phi and is mathematically
    # the total derivative of F. Costs ~+40-60% per E-step iteration.
    # Required for strict monotone descent of the manuscript F monitor.
    # See docs/audits/audit-2026-05-17.md §"Layer 2 Deep-Audit".
    use_autograd_mu_sigma: bool = False

    # When True, _fused_attention_and_vfe_gradients_block_diag computes only
    # the lower triangle (j <= i) of per-pair tensors (Omega, mu_j_transported,
    # sigma_j_transported, KL, gradients) inside the block loop and scatters
    # them into the dense (B,N,N,K) accumulators via tril_indices. Upper-
    # triangle (j > i) entries of those accumulators stay at their zero
    # initialization; the existing causal mask at vfe_gradients.py line 1319
    # still produces beta=0 there via softmax(-inf), so all downstream
    # consumer einsums (beta * grad_kl_per_pair) are algebraically
    # annihilated. Bit-identical to the dense path for beta, grad_mu,
    # grad_sigma under a strict causal mask. The kl_matrix return value
    # differs only in its upper triangle (real KL vs zero).
    #
    # REQUIRES: caller passes a strict lower-triangular causal mask
    # (mask[..., i, j] = 0 for j > i). The flag is validated once per
    # forward at the VFEEStep call site to avoid per-iteration host syncs.
    # Default False: dense path, no behavioral change.
    causal_lower_triangle: bool = False

    # === Runtime (set after construction, not serialized) ===
    generators: Optional[torch.Tensor] = field(default=None, repr=False)
    device: str = 'cuda'

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
        for _lr_field in ('m_mu_lr', 'm_sigma_lr', 'm_phi_lr', 'm_hyper_lr', 'm_other_lr'):
            _v = getattr(self, _lr_field)
            if _v < 0.0:
                raise ValueError(
                    f"{_lr_field}={_v} must be >= 0. Use 0 only to deliberately "
                    f"freeze the corresponding param group; negative values are "
                    f"never valid."
                )
        # --- Exact full-cov decode validation ------------------------------
        # exact_full_cov_decode=True is meaningful only when the prior bank
        # carries non-zero off-diagonal mass within each gauge block AND the
        # decode path is the prior-bank KL readout. The three prerequisites
        # are gauge_fixed_priors=True (so Σ_p_v = A_v Σ_0 A_v^T has dense
        # within-block structure), diagonal_covariance=False (so Σ_q is
        # propagated as a (K, K) tensor through encode and E-step), and
        # use_prior_bank=True (so decode is the KL readout rather than
        # nn.Linear). A misconfigured flag raises here rather than silently
        # falling back to the diagonal path.
        if self.exact_full_cov_decode:
            _ex_problems: List[str] = []
            if not self.gauge_fixed_priors:
                _ex_problems.append(
                    "gauge_fixed_priors=False (per-token Σ_v is diagonal-"
                    "embedded; no off-diagonal mass to recover)"
                )
            if self.diagonal_covariance:
                _ex_problems.append(
                    "diagonal_covariance=True (Σ is (K,) diagonal; no full-cov "
                    "decode applies)"
                )
            if not self.use_prior_bank:
                _ex_problems.append(
                    "use_prior_bank=False (decode is nn.Linear; the flag is "
                    "dead on this path)"
                )
            if _ex_problems:
                raise ValueError(
                    "exact_full_cov_decode=True is incompatible with: "
                    + "; ".join(_ex_problems)
                    + ". Set exact_full_cov_decode=False or fix the upstream "
                    "flags."
                )
        # --- Per-head kappa validation -------------------------------------
        # kappa_per_head requires both upstream flags: learnable_kappa
        # (otherwise there's no parameter to make per-head) and
        # per_head_softmax (otherwise per-head κ_h has nothing to apply to,
        # since the single-softmax path uses one global κ). Also requires
        # at least two heads — H=1 collapses to the scalar case.
        if self.kappa_per_head:
            if not self.learnable_kappa:
                raise ValueError(
                    "kappa_per_head=True requires learnable_kappa=True. "
                    "Without learnable_kappa there is no κ parameter to "
                    "split into per-head; the scalar self.kappa is the "
                    "only state."
                )
            if not self.per_head_softmax:
                raise ValueError(
                    "kappa_per_head=True requires per_head_softmax=True. "
                    "Per-head κ_h has no effect under the single global "
                    "softmax (which uses one κ over the full-K KL)."
                )
            # irrep_dims is computed below from irrep_spec; check the
            # spec directly to ensure H > 1.
            n_heads_spec = sum(mult for _, mult, _ in self.irrep_spec)
            if n_heads_spec <= 1:
                raise ValueError(
                    f"kappa_per_head=True requires more than one gauge "
                    f"head; irrep_spec={self.irrep_spec!r} gives "
                    f"{n_heads_spec} head(s)."
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

        # --- Omega-direct parameterization constraints ---------------------
        if self.gauge_parameterization not in ('phi', 'omega_direct'):
            raise ValueError(
                f"gauge_parameterization must be 'phi' or 'omega_direct'; "
                f"got {self.gauge_parameterization!r}."
            )
        if self.gauge_parameterization == 'omega_direct':
            if not self.diagonal_covariance:
                raise ValueError(
                    "gauge_parameterization='omega_direct' currently requires "
                    "diagonal_covariance=True (the omega-direct path uses the "
                    "same pairwise-Omega KL kernel as use_non_flat_transport, "
                    "which is diagonal-σ only)."
                )
            if self.rope_full_gauge != 'off':
                raise ValueError(
                    "gauge_parameterization='omega_direct' is incompatible "
                    f"with rope_full_gauge={self.rope_full_gauge!r}. RoPE × "
                    "omega-direct on per-pair Σ is not derived."
                )
            if not self.use_autograd_mu_sigma:
                import warnings
                warnings.warn(
                    "gauge_parameterization='omega_direct' forced "
                    "use_autograd_mu_sigma=True (the analytic gradient kernel "
                    "is φ-only; omega-direct needs autograd through the "
                    "pairwise Ω construction). E-step iterations will be "
                    "~40-60% slower than the φ + analytic kernel baseline.",
                    UserWarning,
                    stacklevel=2,
                )
                self.use_autograd_mu_sigma = True

        # --- phi_preconditioner extra guard --------------------------------
        # The Literal enum already excludes 'pullback', so a static type
        # checker rejects it. Re-check at runtime in case the user constructs
        # the dataclass with a string that bypasses the Literal narrowing.
        if self.phi_preconditioner not in (
            'clip', 'cartan', 'killing', 'killing_per_block'
        ):
            raise ValueError(
                f"phi_preconditioner={self.phi_preconditioner!r} is not supported. "
                f"Valid options: 'clip', 'cartan', 'killing', 'killing_per_block'. "
                f"'pullback' is gated here because the inner E-step call chain "
                f"in `VFEEStep._update_phi` does not thread the structure_constants "
                f"tensor that apply_pullback_natural_gradient requires. The metric "
                f"math itself was corrected on 2026-05-20; the corrected formula is "
                f"already available via the outer optimizer "
                f"RiemannianAdamW(metric='pullback')."
            )

        # --- Cross-head coupling validation -------------------------------
        if self.cross_couplings:
            # GLK-only.
            if self.gauge_group != 'GLK':
                raise ValueError(
                    f"cross_couplings is non-empty but gauge_group="
                    f"{self.gauge_group!r}. Cross-head coupling adds GL(K)-"
                    f"specific off-diagonal generators; it has no SO(N) or "
                    f"SO(3) analog. Either drop cross_couplings or set "
                    f"gauge_group='GLK'."
                )
            # Must be single-type multihead.
            if len(self.irrep_spec) != 1:
                raise ValueError(
                    f"cross_couplings requires a single-type irrep_spec "
                    f"(one entry of the form (type, n_heads, d_head)); "
                    f"got {len(self.irrep_spec)} types: {self.irrep_spec!r}."
                )
            _type_name, n_heads, d_head = self.irrep_spec[0]
            if n_heads < 2:
                raise ValueError(
                    f"cross_couplings requires n_heads >= 2; "
                    f"got n_heads={n_heads} from irrep_spec={self.irrep_spec!r}."
                )
            # Dedup and validate pairs.
            seen: set = set()
            deduped: list = []
            for pair in self.cross_couplings:
                if not (isinstance(pair, (tuple, list)) and len(pair) == 2):
                    raise ValueError(
                        f"cross_couplings entries must be (a, b) pairs; "
                        f"got {pair!r}."
                    )
                a, b = int(pair[0]), int(pair[1])
                if a == b:
                    raise ValueError(
                        f"Self-coupling ({a},{a}) is not allowed in "
                        f"cross_couplings — already covered by diagonal "
                        f"generators."
                    )
                if not (0 <= a < n_heads and 0 <= b < n_heads):
                    raise ValueError(
                        f"cross_couplings pair ({a},{b}) has head index "
                        f"out of range [0,{n_heads})."
                    )
                if (a, b) in seen:
                    continue
                seen.add((a, b))
                deduped.append((a, b))
            self.cross_couplings = deduped
            # Non-flat path: per-pair Σ block-mask buffer needs to cover the
            # super-block off-diagonal entries — we adapt the mask in
            # non_flat.py to read super_block_dims, so this combination is
            # supported as of 2026-05-19.
            # Omega-direct: per-block Ω walks super_block_dims; supported.

        # --- non_flat_tile_size reserved future field ----------------------
        if self.non_flat_tile_size != 0:
            raise NotImplementedError(
                f"non_flat_tile_size={self.non_flat_tile_size} is reserved for "
                "future j-axis chunked aggregation; the tiled path is not yet "
                "implemented. Set non_flat_tile_size=0 (the default) to use the "
                "unchunked aggregation, or remove the field from your config."
            )

        # --- Non-flat transport constraints --------------------------------
        if self.use_non_flat_transport:
            if not self.diagonal_covariance:
                raise ValueError(
                    "use_non_flat_transport=True currently requires "
                    "diagonal_covariance=True. Per-pair full-cov KL with "
                    "logdet is not implemented for the non-flat path."
                )
            if self.rope_full_gauge != 'off':
                raise ValueError(
                    "use_non_flat_transport=True is incompatible with "
                    f"rope_full_gauge={self.rope_full_gauge!r}. The RoPE × "
                    "non-flat coupling on per-pair Σ has not been derived; "
                    "set rope_full_gauge='off' or disable non-flat transport."
                )
            if not self.use_autograd_mu_sigma:
                # Promote rather than reject — the analytic kernel simply
                # doesn't have a non-flat path, but the autograd path does.
                # Promotion is the unique correct behavior; warn so the user
                # is aware of the cost (~+40-60% per E-step iteration).
                import warnings
                warnings.warn(
                    "use_non_flat_transport=True forced "
                    "use_autograd_mu_sigma=True (the analytic gradient kernel "
                    "does not support non-flat transport). E-step iterations "
                    "will be ~40-60% slower.",
                    UserWarning,
                    stacklevel=2,
                )
                self.use_autograd_mu_sigma = True

        # --- alpha_divergence is only honored by the analytic gradient kernel.
        # The autograd, non-flat, and omega_direct paths reconstruct the
        # standard KL functional from scratch and silently ignore the Renyi
        # alpha exponent. Warn the user if they combined alpha != 1.0 with any
        # of those paths so the divergence is not silently dropped.
        if (
            self.alpha_divergence != 1.0
            and (
                self.use_autograd_mu_sigma
                or self.use_non_flat_transport
                or self.gauge_parameterization == 'omega_direct'
            )
        ):
            import warnings
            warnings.warn(
                f"alpha_divergence={self.alpha_divergence} is only honored by "
                f"the analytic gradient kernel. The current configuration "
                f"(use_autograd_mu_sigma={self.use_autograd_mu_sigma}, "
                f"use_non_flat_transport={self.use_non_flat_transport}, "
                f"gauge_parameterization={self.gauge_parameterization!r}) "
                f"routes through a path that reconstructs the standard KL and "
                f"will ignore the Renyi exponent.",
                UserWarning,
                stacklevel=2,
            )

        # The E-step inner loop in e_step.py orders updates as
        # (μ-update → σ-retract → φ-retract). With n_e_steps==1 the
        # retracted σ and φ are the last tensors written before the loop
        # exits, and no later step in the same iteration consumes them.
        # Whether σ_new and φ_new reach the loss after stack(...) returns
        # then depends on what reads them downstream:
        #   σ lift: n_e_steps>=2  (next iter's nat_grad_mu reads sigma)
        #        OR n_layers>=2   (next layer's compute_kl_attention reads sigma)
        #        OR use_prior_bank=True   (prior_bank.decode reads sigma_q)
        #        OR norm_type in {'mahalnorm','centered_mahalnorm'}   (norm reads sigma)
        #   φ lift: n_e_steps>=2  OR n_layers>=2
        #        (prior_bank.decode(mu_q, sigma_q, tau) does not take phi;
        #         no norm reads phi.)
        # When the relevant lift fails, sweeping e_sigma_lr / e_phi_lr scales
        # tensors that are autograd-disconnected from CE and the loss is
        # bitwise identical across the sweep.
        # σ-orphan lift conditions depend on whether direct-mode prior is
        # active: under `gauge_fixed_priors=False` the σ parameter is
        # `sigma_log_embed`, which receives gradient through the starting-σ
        # path in the E-step regardless of `e_sigma_lr` — so sweeping
        # `e_sigma_lr` is still orphaned (it scales a tensor discarded after
        # the inner update), but `sigma_log_embed` itself learns. The
        # warning text below now disambiguates the two cases.
        _sigma_lift_via_norm = self.norm_type in ('mahalnorm', 'centered_mahalnorm')
        _sigma_orphan = (
            self.n_e_steps == 1
            and self.n_layers == 1
            and not self.use_prior_bank
            and not _sigma_lift_via_norm
        )
        _phi_orphan = (self.n_e_steps == 1 and self.n_layers == 1)
        if _sigma_orphan or _phi_orphan:
            _dead = []
            if _sigma_orphan:
                _dead.append("e_sigma_lr")
            if _phi_orphan:
                _dead.append("e_phi_lr")
            _direct_mode_note = (
                " NOTE: gauge_fixed_priors=False — the σ parameter "
                "(`sigma_log_embed`) still learns via the starting-σ path; "
                "only `e_sigma_lr` (the inner-iteration σ retraction rate) "
                "is dead under this configuration."
                if not self.gauge_fixed_priors and _sigma_orphan
                else ""
            )
            import warnings
            warnings.warn(
                f"Orphaned E-step retraction: with n_e_steps={self.n_e_steps}, "
                f"n_layers={self.n_layers}, use_prior_bank={self.use_prior_bank}, "
                f"norm_type={self.norm_type!r}, the following LR(s) scale "
                f"tensors that never reach the loss and will show no effect "
                f"when swept: {', '.join(_dead)}. Lift conditions — σ: "
                f"n_e_steps>=2 OR n_layers>=2 OR use_prior_bank=True OR "
                f"norm_type in {{'mahalnorm','centered_mahalnorm'}}; φ: "
                f"n_e_steps>=2 OR n_layers>=2."
                f"{_direct_mode_note}",
                UserWarning,
                stacklevel=2,
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
        """Number of Lie algebra generators.

        With ``cross_couplings`` active, the count includes the off-diagonal
        E_ij blocks: ``n_heads · d_head² + |cross_couplings| · d_head²``.
        """
        if self.generators is not None:
            return self.generators.shape[0]
        # Estimate from gauge group and irrep structure
        if self.gauge_group == 'SO3':
            return 3
        elif self.gauge_group == 'GLK':
            base = sum(d ** 2 for d in self.irrep_dims)
            if self.cross_couplings:
                _, _n_heads, d_head = self.irrep_spec[0]
                base += len(self.cross_couplings) * d_head * d_head
            return base
        else:  # SON
            K = self.embed_dim
            return K * (K - 1) // 2

    @property
    def is_cross_coupled(self) -> bool:
        """True iff ``cross_couplings`` is non-empty (GL(K) multi-head)."""
        return bool(self.cross_couplings)

    @property
    def super_block_dims(self) -> Optional[List[int]]:
        """Super-block dimensions when ``cross_couplings`` is active.

        Coupled heads are merged via union-find on the coupling pairs;
        uncoupled heads remain singleton blocks. Sum equals ``embed_dim``.
        Returns ``None`` when no cross-coupling is active.

        Lazily computed from ``math_utils.generators.merge_coupled_heads``;
        the result is cached on the instance.
        """
        if not self.cross_couplings:
            return None
        cached = getattr(self, '_super_block_dims_cache', None)
        if cached is not None:
            return cached
        from math_utils.generators import merge_coupled_heads
        _, n_heads, d_head = self.irrep_spec[0]
        dims, groups = merge_coupled_heads(n_heads, d_head, list(self.cross_couplings))
        # Cache both via object.__setattr__ to bypass dataclass __setattr__ in
        # frozen configurations (not currently frozen, but defensive).
        object.__setattr__(self, '_super_block_dims_cache', dims)
        object.__setattr__(self, '_super_block_head_groups_cache', groups)
        return dims

    @property
    def super_block_head_groups(self) -> Optional[List[List[int]]]:
        """List of head-index groups per super-block.

        Each entry is the list of original head indices unioned into that
        super-block, in original order (sorted by smallest head). Returns
        ``None`` when no cross-coupling is active.
        """
        if not self.cross_couplings:
            return None
        # Triggers computation + cache of both fields.
        _ = self.super_block_dims
        return getattr(self, '_super_block_head_groups_cache', None)

    @property
    def effective_block_dims(self) -> List[int]:
        """Block dimensions for the gauge-block iteration.

        Returns ``super_block_dims`` when ``cross_couplings`` is active;
        otherwise returns ``irrep_dims``. This is the layout that downstream
        block-iteration code in e_step / attention / prior_bank / etc.
        should consume — under cross-coupling the gauge-block-diagonal
        structure is the super-block partition, not the original per-head
        partition.
        """
        if self.cross_couplings:
            sbd = self.super_block_dims
            assert sbd is not None
            return sbd
        return self.irrep_dims
