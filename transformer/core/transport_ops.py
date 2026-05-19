"""
Transport Operator Functions for the Gauge Transformer
=======================================================

Parallel transport operators for the gauge-theoretic attention mechanism.
Extracted from attention.py to allow independent import by variational_ffn.py
and blocks.py without pulling in the full attention module.

Key mathematical objects:

    Transport operator:  Omega_ij = exp(phi_i * G) * exp(-phi_j * G)
    Mean transport:      mu_transported = Omega_ij @ mu_j
    Covariance transport: Sigma_transported = Omega_ij @ Sigma_j @ Omega_ij.T

The sandwich product for covariance transport is the single most critical
correctness invariant — never transport covariance without the conjugation.

Rotary position encoding (RoPE) applies SO(2)^{K/2} rotations to belief
means before KL divergence computation, making attention position-sensitive
without affecting gauge transport.
"""

import torch
from typing import List, Optional, Tuple, TypedDict


class _TransportDictRequired(TypedDict):
    """Required keys for `TransportDict` — always populated."""
    exp_phi: torch.Tensor       # (B, N, K, K)
    exp_neg_phi: torch.Tensor   # (B, N, K, K)
    Omega: torch.Tensor         # (B, N, N, K, K)


class TransportDict(_TransportDictRequired, total=False):
    """Return shape of ``compute_transport_operators`` / ``...direct``.

    Inherits three always-present keys (`exp_phi`, `exp_neg_phi`, `Omega`)
    from `_TransportDictRequired`. The optional `exp_delta` is present only
    when `connection_delta` is supplied.
    """
    exp_delta: torch.Tensor     # (B, N, N, K, K), optional

from transformer.core.gauge_utils import (
    stable_matrix_exp_pair,
    newton_schulz_orthogonalize,
)


__all__ = [
    'compute_transport_operators',
    'compute_transport_operators_direct',
    'omega_to_block_exp_pairs',
    '_apply_rope',
    '_build_rope_freqs',
]


# =============================================================================
# Rotary Position Embeddings (RoPE) for KL-Divergence Attention
# =============================================================================
# RoPE applies position-dependent SO(2)^{K/2} rotations to belief means μ
# before computing KL divergences. This makes attention position-sensitive
# without affecting gauge transport Ω_ij.
#
# In the gauge-theoretic framework (see GL(K)_attention.tex §3):
#   Ω_ij^{RoPE} = R(θ_{j-i}) · Ω_ij^{content}
# where R(θ) ∈ SO(2)^{K/2} ⊂ GL(K) is the position-dependent rotation.
# =============================================================================

def _build_rope_freqs(K: int, base: float = 10000.0,
                      device: Optional[torch.device] = None,
                      dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Compute RoPE frequency bands for K-dimensional beliefs.

    Returns:
        freqs: (K//2,) inverse frequency bands
    """
    half_K = K // 2
    freqs = 1.0 / (base ** (torch.arange(0, half_K, device=device, dtype=dtype) / half_K))
    return freqs


# RoPE cos/sin cache: keyed by (K, base, N, device) to avoid recomputation.
# Bounded to _ROPE_CACHE_MAX entries to prevent unbounded growth with
# varying sequence lengths.
_rope_cache: dict = {}
_ROPE_CACHE_MAX: int = 16


def _get_rope_cos_sin(
    K: int, N: int, base: float,
    device: torch.device, dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get cached RoPE cos/sin tensors, recomputing only when args change."""
    key = (K, N, base, device, dtype)
    cached = _rope_cache.get(key)
    if cached is not None:
        return cached
    freqs = _build_rope_freqs(K, base, device=device, dtype=dtype)  # (K//2,)
    positions = torch.arange(N, device=device, dtype=dtype)  # (N,)
    angles = torch.outer(positions, freqs)  # (N, K//2)
    cos_sin = (torch.cos(angles), torch.sin(angles))
    if len(_rope_cache) >= _ROPE_CACHE_MAX:
        _rope_cache.pop(next(iter(_rope_cache)))
    _rope_cache[key] = cos_sin
    return cos_sin


def _apply_rope(mu: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    r"""Apply Rotary Position Embeddings to belief means.

    Rotates consecutive pairs of dimensions by position-dependent angles,
    making KL divergences sensitive to relative position via SO(2)^{K//2}.

    When K is odd, the last dimension is left unrotated (standard RoPE
    convention — only K//2 pairs are formed from K dimensions).

    INTERPRETATION (important).  This implementation rotates only μ, not
    the covariance Σ.  Under the GL(K) manuscript (Section "Rotary Position
    Embeddings as Position-Dependent Gauge Frames"), RoPE is derived as a
    gauge transport operator restricted to SO(2)^{K/2} ⊂ GL(K), with
    R(θ_{j-i}) = exp(-φ_i^pos)·exp(φ_j^pos) playing the role of the
    inter-position transport.  By the framework's gauge-transport rule,
    such an operator should act on Σ via the sandwich product
    Σ → R Σ R^T.  This function does *not* implement that — it follows the
    standard-transformer Q/K rotation pattern (rotate the means used for
    scoring, leave the covariance untouched).  The result is that the
    forward-pass KL is computed with rotated μ but raw Σ, which is neither
    the manuscript's full gauge interpretation nor a textbook KL between
    rotated Gaussians.

    The framework supports this factorization (manuscript line ~1760: "the
    asymmetry in RoPE — with position-dependent attention but
    position-independent values — corresponds to factoring the gauge
    transport into an attention gauge and a value gauge that need not
    coincide"), but the in-attention-gauge σ omission is a separate
    pragmatic choice tied to the cost of breaking diagonal Σ.  See
    `BlockConfig.rope_full_gauge` for an experimental flag that enables
    the full Σ rotation.

    Args:
        mu: (B, N, K) belief means
        base: RoPE frequency base (default 10000.0)

    Returns:
        mu_rotated: (B, N, K) position-rotated belief means
    """
    B, N, K = mu.shape
    half_K = K // 2

    # Cached cos/sin (recomputed only when K, N, base, or device changes)
    cos_angles, sin_angles = _get_rope_cos_sin(K, N, base, mu.device, mu.dtype)

    # Split μ into even/odd pairs and apply 2D rotation
    mu_even = mu[:, :, :2*half_K:2]   # (B, N, K//2) - dims 0,2,4,...
    mu_odd = mu[:, :, 1:2*half_K:2]   # (B, N, K//2) - dims 1,3,5,...

    # R(θ) @ [x, y]^T = [x·cos(θ) - y·sin(θ), x·sin(θ) + y·cos(θ)]
    rot_even = mu_even * cos_angles - mu_odd * sin_angles
    rot_odd = mu_even * sin_angles + mu_odd * cos_angles

    if K % 2 == 0:
        # All dims are paired — interleave without clone
        mu_rotated = torch.stack([rot_even, rot_odd], dim=-1).reshape(B, N, K)
    else:
        # Odd K: last dim is unpaired, copy it through
        mu_rotated = torch.empty_like(mu)
        mu_rotated[:, :, :2*half_K:2] = rot_even
        mu_rotated[:, :, 1:2*half_K:2] = rot_odd
        mu_rotated[:, :, 2*half_K:] = mu[:, :, 2*half_K:]

    return mu_rotated


def _apply_rope_to_covariance(
    sigma_full: torch.Tensor, base: float = 10000.0
) -> torch.Tensor:
    r"""Apply R(θ_i) Σ_i R(θ_i)^T per position to a full covariance tensor.

    Used for the experimental `rope_full_gauge` flag, which implements the
    framework-consistent interpretation of RoPE as a position-dependent
    gauge transport that acts on Gaussian beliefs by both μ → Rμ AND
    Σ → RΣR^T (the standard sandwich product for covariance transport).

    Args:
        sigma_full: (B, N, K, K) full covariance tensor (must be lifted
                    from diagonal beforehand if needed).
        base: RoPE frequency base (must match the value used by _apply_rope).

    Returns:
        (B, N, K, K) covariance tensor with R(θ_i) sandwich applied per i.
        For diagonal input with σ_a, σ_b on a (2k, 2k+1) pair, the result
        has 2x2 block structure
            [c² σ_a + s² σ_b,    sc(σ_a − σ_b) ]
            [sc(σ_a − σ_b),     s² σ_a + c² σ_b]
        where c=cos(i·ω_k), s=sin(i·ω_k), ω_k = base^{-2k/K}.  The
        determinant is preserved (R is orthogonal).
    """
    B, N, K, K2 = sigma_full.shape
    assert K == K2, f"sigma_full must be square in last two dims, got {sigma_full.shape}"
    half_K = K // 2

    # cos/sin of size (N, K//2) — angles per position per frequency band
    cos_angles, sin_angles = _get_rope_cos_sin(K, N, base, sigma_full.device, sigma_full.dtype)

    # Build R(θ_i) ∈ R^{N, K, K} as a sparse block-diagonal rotation:
    # for each i, R has 2x2 blocks on the (2k, 2k+1) pairs and identity
    # elsewhere (last dim untouched if K is odd). Vectorised assembly via
    # advanced indexing (one launch each for the four block-element groups)
    # avoids the per-frequency Python loop that previously scatter-wrote
    # K/2 times per call.
    R = torch.zeros(N, K, K, device=sigma_full.device, dtype=sigma_full.dtype)
    if K % 2 == 1:
        R[:, K - 1, K - 1] = 1.0
    _even = torch.arange(half_K, device=sigma_full.device) * 2
    _odd = _even + 1
    R[:, _even, _even] = cos_angles                              # cos on (2k, 2k)
    R[:, _even, _odd] = -sin_angles                              # -sin on (2k, 2k+1)
    R[:, _odd, _even] = sin_angles                               #  sin on (2k+1, 2k)
    R[:, _odd, _odd] = cos_angles                                #  cos on (2k+1, 2k+1)

    # R Σ R^T per position: (N, K, K) acts on (B, N, K, K) along the
    # trailing two dims, broadcasting over batch.
    R_b = R.unsqueeze(0)  # (1, N, K, K)
    sigma_rot = torch.einsum('bnkl,bnlm,bnpm->bnkp', R_b, sigma_full, R_b)
    # Symmetrize for numerical stability
    return 0.5 * (sigma_rot + sigma_rot.transpose(-1, -2))


def _un_apply_rope_pair_outer(
    grad: torch.Tensor, base: float = 10000.0
) -> torch.Tensor:
    r"""Apply R(θ_i)^T to a (B, N_i, N_j, K) gradient tensor along the last dim.

    Used for chain-rule consistency when the KL is computed from RoPE-rotated μ
    but the gradient must be expressed in raw-μ space:

        ∂KL_RoPE/∂μ_raw_i = R(θ_i)^T · ∂KL_RoPE/∂(R(θ_i)·μ_i)

    The rotation R(θ_i) acts only on the i-th query position (the second
    dimension of the input tensor); the j-th key position (third dimension)
    is broadcast over.  Since R is orthogonal, R^T = R(-θ), implemented by
    negating sin while keeping cos.

    Args:
        grad: (B, N_i, N_j, K) gradient tensor in RoPE coordinates.
        base: RoPE frequency base (must match the value used by _apply_rope).

    Returns:
        (B, N_i, N_j, K) gradient tensor expressed in raw-μ coordinates.
    """
    B, N_i, N_j, K = grad.shape
    half_K = K // 2

    cos_angles, sin_angles = _get_rope_cos_sin(K, N_i, base, grad.device, grad.dtype)
    # Broadcast (N_i, K//2) over (B, N_i, N_j, K//2) so the rotation depends only on i.
    cos_b = cos_angles[None, :, None, :]
    sin_b = sin_angles[None, :, None, :]

    g_even = grad[:, :, :, :2*half_K:2]   # (B, N_i, N_j, K//2)
    g_odd  = grad[:, :, :, 1:2*half_K:2]

    # R(θ) [x, y] = [x cos - y sin, x sin + y cos]
    # R^T(θ) = R(-θ) [x, y] = [x cos + y sin, -x sin + y cos]
    out = grad.clone()
    out[:, :, :, :2*half_K:2] = g_even * cos_b + g_odd * sin_b
    out[:, :, :, 1:2*half_K:2] = -g_even * sin_b + g_odd * cos_b
    return out


# =============================================================================
# Transport Operator Caching (for evolve_phi=False optimization)
# =============================================================================

def compute_transport_operators(
    phi: torch.Tensor,         # (B, N, n_gen) gauge frames where n_gen is # of generators
    generators: torch.Tensor,  # (n_gen, K, K) Lie algebra generators
    enforce_orthogonal: bool = False,  # If True, project to SO(K) via Newton-Schulz
    gauge_mode: str = 'learned',  # 'learned', 'trivial', or 'constant'
    connection_delta: Optional[torch.Tensor] = None,  # (B, N, N, n_gen) edge-local connection
    cocycle_relaxation: float = 0.0,  # Scale factor for connection_delta: 0=flat, 1=fully non-flat
    generators_are_skew: Optional[bool] = None,  # Pre-computed skew-symmetry flag
) -> TransportDict:
    """
    Precompute transport operators for caching when phi is fixed.

    Works for SO(3), SO(N), and GL⁺(K) gauge groups:
    - SO(3): n_gen = 3, phi ∈ ℝ³, enforce_orthogonal=True
    - SO(N): n_gen = N(N-1)/2, phi ∈ ℝ^{N(N-1)/2}, enforce_orthogonal=True
    - GL⁺(K): n_gen = K², phi ∈ ℝ^{K²}, enforce_orthogonal=False
      (exp parameterization reaches identity component GL⁺(K) only;
       det(exp(X)) = exp(tr(X)) > 0 always. The product Ω_ij =
       exp(X_i)·exp(-X_j) covers all of GL⁺(K).)

    Gauge Modes:
    - 'learned': Standard mode where φ is learned per-token. Transport Ω_ij
                 encodes relative frame transformations between tokens.
    - 'trivial': Global frame mode where Ω = I for all pairs. This is the
                 mathematically principled "trivial gauge" or "gauge fixing"
                 that recovers standard attention as a special case.
                 Equivalent to setting φ = 0 everywhere.
    - 'constant': Per-head learnable Ω ∈ GL(d_head), same for all pairs.
                  The constant Ω is stored as nn.Parameter in the attention
                  module (IrrepMultiHeadAttention.constant_omega), not here.
                  This function returns identity transport for 'constant'
                  mode; the attention module injects the learned Ω directly.

    When evolve_phi=False, these operators are constant across layers.
    Computing once saves 2 matrix exponentials per head per layer.

    Args:
        phi: Gauge frames (B, N, n_gen) in Lie algebra
             - For SO(3): shape (B, N, 3)
             - For SO(N): shape (B, N, N*(N-1)/2)
             - For GL(K): shape (B, N, K²)
        generators: Lie algebra generators (n_gen, K, K)
        enforce_orthogonal: If True, apply Newton-Schulz to ensure Ω ∈ SO(K).
                           If False, allow Ω ∈ GL⁺(K) (faster, still gauge-invariant).
        gauge_mode: 'learned' for per-token frames, 'trivial' for Ω=I,
                   'constant' for per-head learned Ω (returns I here; actual Ω
                   injected by attention module)
        connection_delta: Optional edge-local Lie algebra elements (B, N, N, n_gen).
                         When provided, transport becomes:
                         Ω_ij = exp(φ_i·G) · exp(α·δ_ij·G) · exp(-φ_j·G)
                         where α = cocycle_relaxation ∈ [0,1].
        cocycle_relaxation: Scale factor for connection_delta (0=flat, 1=fully non-flat).

    Returns:
        dict with:
            'exp_phi': (B, N, K, K) - exp(φ·G) for each token
            'exp_neg_phi': (B, N, K, K) - exp(-φ·G) for each token
            'Omega': (B, N, N, K, K) - full pairwise transport Ω_ij = exp(φ_i)exp(-φ_j)
    """
    B, N, _ = phi.shape
    K = generators.shape[1]
    dtype = phi.dtype
    device = phi.device

    # =================================================================
    # TRIVIAL / CONSTANT GAUGE: Ω = I (identity transport)
    # =================================================================
    # 'trivial': φ = 0, Ω = I everywhere (global frame, standard attention).
    # 'constant': Per-head Ω ∈ GL(d_head) is handled by the attention module
    #   (IrrepMultiHeadAttention) which injects its constant_omega directly.
    #   Here we return identity because this function has no access to the
    #   per-head learned Ω parameters. The attention module overrides these
    #   identity operators with the actual constant_omega in its forward().
    if gauge_mode in ('trivial', 'constant'):
        eye_K = torch.eye(K, device=device, dtype=dtype)
        exp_phi = eye_K.expand(B, N, K, K).contiguous()      # (B, N, K, K)
        exp_neg_phi = eye_K.expand(B, N, K, K).contiguous()  # (B, N, K, K)
        Omega = eye_K.expand(B, N, N, K, K).contiguous()     # (B, N, N, K, K)
        return {
            'exp_phi': exp_phi,
            'exp_neg_phi': exp_neg_phi,
            'Omega': Omega,
        }

    # =================================================================
    # LEARNED GAUGE: φ per-token, Ω_ij = exp(φ_i)·exp(-φ_j)
    # =================================================================
    # φ·G: combine gauge frames with generators
    phi_matrix = torch.einsum('bna,aij->bnij', phi, generators)  # (B, N, K, K)

    # Check if generators are skew-symmetric (SO(K) gauge group).
    # For skew-symmetric A: exp(-A) = exp(A)^T, saving one matrix_exp call.
    _is_skew = generators_are_skew
    if _is_skew is None:
        _is_skew = torch.allclose(
            generators + generators.transpose(-1, -2),
            torch.zeros_like(generators), atol=1e-5
        )

    # Float64 matrix_exp for GL(K) numerical stability (prevents NaN
    # from Padé scaling-squaring overflow when phi values grow large).
    # When generators are skew-symmetric, skip computing exp(-M) (use transpose).
    exp_phi, exp_neg_phi = stable_matrix_exp_pair(
        phi_matrix, skew_symmetric=_is_skew
    )

    # Re-orthogonalization for SO(K) gauge groups
    # NOTE: For GL⁺(K), this is NOT required - VFE is invariant under GL(K)!
    # Only enable if you explicitly want SO(K) (e.g., for Haar measure averaging)
    if enforce_orthogonal and K >= 2:
        exp_phi = newton_schulz_orthogonalize(exp_phi)
        exp_neg_phi = newton_schulz_orthogonalize(exp_neg_phi)

    # Full pairwise transport
    exp_delta = None
    if connection_delta is not None and cocycle_relaxation > 0:
        # Non-flat transport: Ω_ij = exp(φ_i·G) · exp(α·δ_ij·G) · exp(-φ_j·G)
        scaled_delta = cocycle_relaxation * connection_delta  # (B, N, N, n_gen)
        # Numerical stability: if the bilinear/MLP connection produces a
        # Lie algebra coefficient with very large norm, matrix_exp can
        # overflow or produce NaN.  Clip the per-edge scaled delta norm to
        # a conservative bound so the transport stays well-defined even if
        # the connection weights drift during training.  5.0 matches the
        # GL(K) max phi norm convention (see retract_glK_torch trust
        # region); for SO(N) the penalty is looser but still finite.
        _delta_max_norm = 5.0
        _delta_norms = scaled_delta.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        _delta_scale = torch.clamp(_delta_max_norm / _delta_norms, max=1.0)
        scaled_delta = scaled_delta * _delta_scale
        delta_matrix = torch.einsum('bija,akl->bijkl', scaled_delta, generators)  # (B, N, N, K, K)
        exp_delta = torch.linalg.matrix_exp(delta_matrix.float()).to(dtype)  # (B, N, N, K, K)
        # Ω_ij = exp(φ_i) @ exp(δ_ij) @ exp(-φ_j)
        Omega = torch.einsum(
            'bikl,bijlm,bjmn->bijkn', exp_phi, exp_delta, exp_neg_phi
        )  # (B, N, N, K, K)
    else:
        # Standard flat transport: Ω_ij = exp(φ_i) @ exp(-φ_j)
        Omega = torch.einsum('bikl,bjlm->bijkm', exp_phi, exp_neg_phi)  # (B, N, N, K, K)

    return {
        'exp_phi': exp_phi,
        'exp_neg_phi': exp_neg_phi,
        'Omega': Omega,
        'exp_delta': exp_delta,
    }


# =============================================================================
# Direct Omega Transport (No Lie Algebra / No matrix_exp)
# =============================================================================

def compute_transport_operators_direct(
    omega: torch.Tensor,  # (B, N, K, K) direct group elements
    gauge_mode: str = 'learned',
    connection_delta: Optional[torch.Tensor] = None,  # (B, N, N, n_gen) edge-local
    generators: Optional[torch.Tensor] = None,  # needed only for non-flat connection
    cocycle_relaxation: float = 0.0,
) -> TransportDict:
    """
    Compute transport operators from direct GL(K) group elements (no matrix_exp).

    Flat:      Ω_ij = Ω_i · Ω_j⁻¹  (cocycle condition exact to float precision)
    Non-flat:  Ω_ij = Ω_i · exp(α·δ_ij·G) · Ω_j⁻¹

    The cocycle identity Ω_ij @ Ω_jk = Ω_ik is exact when the LU-based
    solve succeeds (primary path). If omega is near-singular, fallbacks
    (ridge regularization, then pseudoinverse) break the cocycle by O(ε)
    or worse. These fallbacks fire only when the raw matrix is ill-conditioned.

    Unlike compute_transport_operators (phi-based), this:
    - Stores Ω_i ∈ R^{K×K} directly (no Lie algebra coefficients)
    - No matrix_exp for per-token frames (only for edge-local connection if non-flat)
    - Initialized near identity but NOT constrained to GL(K) during training —
      gradient updates can push Ω toward singular matrices, at which point
      fallbacks degrade transport quality (see fallback notes below)
    - Simpler gradient: ∂KL/∂Ω_i via chain rule, no dexp series

    Note: the phi-based path (gauge_param='phi') guarantees Ω = exp(φ·G) ∈ GL(K)
    by construction (matrix exponentials are always invertible). This direct path
    trades that guarantee for simpler gradients and the ability to represent
    reflections (det < 0), but requires external regularization to prevent
    Ω from approaching singularity. See TrainingConfig.omega_det_penalty
    for the (log|det Ω|)² regularizer that supplies this gradient pressure.



    Args:
        omega: (B, N, K, K) per-token matrices Ω_i, initialized near identity.
               Not constrained to GL(K) — may become singular during training.
        gauge_mode: 'learned' for per-token frames, 'trivial' for Ω = I.
        connection_delta: Optional (B, N, N, n_gen) edge-local Lie algebra elements.
                         Only used when cocycle_relaxation > 0.
        generators: (n_gen, K, K) Lie algebra generators. Only needed for non-flat.
        cocycle_relaxation: Scale factor for connection_delta (0=flat, 1=fully non-flat).

    Returns:
        dict with:
            'omega_i': (B, N, K, K) — per-token Ω_i
            'omega_j_inv': (B, N, K, K) — per-token Ω_j⁻¹
            'Omega': (B, N, N, K, K) — pairwise transport Ω_ij
    """
    B, N, K, _ = omega.shape
    dtype = omega.dtype
    device = omega.device

    if gauge_mode in ('trivial', 'constant'):
        eye_K = torch.eye(K, device=device, dtype=dtype)
        omega_i = eye_K.expand(B, N, K, K).contiguous()
        omega_j_inv = eye_K.expand(B, N, K, K).contiguous()
        Omega = eye_K.expand(B, N, N, K, K).contiguous()
        return {
            'omega_i': omega_i,
            'omega_j_inv': omega_j_inv,
            'Omega': Omega,
        }

    # Per-token inverse Ω_j⁻¹ (needed for transport Ω_ij = Ω_i Ω_j⁻¹).
    #
    # The cocycle identity Ω_ij @ Ω_jk = Ω_ik holds exactly when the
    # inverse is exact.  We use torch.linalg.solve (LU with pivoting)
    # as the primary path — more stable than inv() and ridge-free, so
    # the cocycle is exact to floating-point precision.
    #
    # Fallback 1: ridge 1e-6 + solve (if omega is near-singular).
    #   Breaks cocycle by O(ε), but keeps the pipeline finite.
    # Fallback 2: pinv (if ridge + solve also fails).
    #   Destroys GL(K) structure; transport is rank-deficient.
    eye_K = torch.eye(K, device=device, dtype=dtype)
    try:
        omega_j_inv = torch.linalg.solve(omega, eye_K.expand_as(omega))  # (B, N, K, K)
    except (torch.linalg.LinAlgError, RuntimeError):
        try:
            from math_utils.numerical_monitor import record as _nr
            _nr("omega_inv_ridge_fallback")
            _ridge = 1e-6
            omega_j_inv = torch.linalg.solve(
                omega + _ridge * eye_K, eye_K.expand_as(omega))
        except (torch.linalg.LinAlgError, RuntimeError):
            from math_utils.numerical_monitor import record as _nr
            _nr("omega_inv_pinv_fallback")
            omega_j_inv = torch.linalg.pinv(omega)

    if connection_delta is not None and cocycle_relaxation > 0 and generators is not None:
        # Non-flat: Ω_ij = Ω_i · exp(α·δ_ij·G) · Ω_j⁻¹
        scaled_delta = cocycle_relaxation * connection_delta  # (B, N, N, n_gen)
        delta_matrix = torch.einsum('bija,akl->bijkl', scaled_delta, generators)
        exp_delta = torch.linalg.matrix_exp(delta_matrix.float()).to(dtype)  # (B, N, N, K, K)
        # Ω_ij = Ω_i @ exp(δ_ij) @ Ω_j⁻¹
        Omega = torch.einsum(
            'bikl,bijlm,bjmn->bijkn', omega, exp_delta, omega_j_inv
        )
    else:
        # Flat: Ω_ij = Ω_i · Ω_j⁻¹ (cocycle exact when solve succeeds)
        Omega = torch.einsum('bikl,bjlm->bijkm', omega, omega_j_inv)

    return {
        'omega_i': omega,
        'omega_j_inv': omega_j_inv,
        'Omega': Omega,
    }


def omega_to_block_exp_pairs(
    omega: torch.Tensor,  # (B, N, K, K) direct group elements
    irrep_dims: List[int],
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Convert direct Omega matrices to block_exp_pairs format for KL computation.

    The block-diagonal KL functions expect a list of (forward, inverse) pairs
    per irrep block, originally from fused_block_matrix_exp_pairs (exp(phi), exp(-phi)).
    For the direct Omega path, we provide (omega_block, omega_block_inv) instead.

    Args:
        omega: (B, N, K, K) per-token group elements.
        irrep_dims: List of block dimensions [d₁, d₂, ...].

    Returns:
        List of (omega_block, omega_block_inv) tuples, each shape (B, N, d, d).
    """
    # Dimension guard (matches fused_block_matrix_exp_pairs, Fix #36)
    _id_sum = sum(irrep_dims)
    K = omega.shape[-1]
    if _id_sum != K:
        raise ValueError(
            f"omega_to_block_exp_pairs: sum(irrep_dims)={_id_sum} does not "
            f"equal K={K}. irrep_dims={irrep_dims}, omega.shape={tuple(omega.shape)}."
        )

    # Per-block inverse via solve (LU with pivoting) — ridge-free primary
    # path preserves exact cocycle.  Fallback: ridge + solve, then pinv.
    results = []
    start = 0
    for d in irrep_dims:
        end = start + d
        omega_blk = omega[:, :, start:end, start:end].contiguous()
        eye_d = torch.eye(d, device=omega_blk.device, dtype=omega_blk.dtype)
        try:
            omega_blk_inv = torch.linalg.solve(omega_blk, eye_d.expand_as(omega_blk))
        except (torch.linalg.LinAlgError, RuntimeError):
            try:
                from math_utils.numerical_monitor import record as _nr
                _nr("omega_blk_inv_ridge_fallback")
                omega_blk_inv = torch.linalg.solve(
                    omega_blk + 1e-6 * eye_d, eye_d.expand_as(omega_blk))
            except (torch.linalg.LinAlgError, RuntimeError):
                from math_utils.numerical_monitor import record as _nr
                _nr("omega_blk_inv_pinv_fallback")
                omega_blk_inv = torch.linalg.pinv(omega_blk)
        results.append((omega_blk, omega_blk_inv))
        start = end
    return results