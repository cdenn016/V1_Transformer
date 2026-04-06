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

import math
import torch
from typing import List, Optional, Tuple

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
                      device: torch.device = None,
                      dtype: torch.dtype = None) -> torch.Tensor:
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
    mu_rotated = mu.clone()
    mu_rotated[:, :, :2*half_K:2] = mu_even * cos_angles - mu_odd * sin_angles
    mu_rotated[:, :, 1:2*half_K:2] = mu_even * sin_angles + mu_odd * cos_angles

    return mu_rotated


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
    **kwargs,  # generators_are_skew: Optional[bool] — pre-computed skew-symmetry flag
) -> dict:
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
    # Accept pre-computed flag to avoid torch.allclose on every forward pass.
    _is_skew = kwargs.get('generators_are_skew', None)
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
    if enforce_orthogonal and K >= 16:
        exp_phi = newton_schulz_orthogonalize(exp_phi)
        exp_neg_phi = newton_schulz_orthogonalize(exp_neg_phi)

    # Full pairwise transport
    exp_delta = None
    if connection_delta is not None and cocycle_relaxation > 0:
        # Non-flat transport: Ω_ij = exp(φ_i·G) · exp(α·δ_ij·G) · exp(-φ_j·G)
        scaled_delta = cocycle_relaxation * connection_delta  # (B, N, N, n_gen)
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
) -> dict:
    """
    Compute transport operators from direct GL(K) group elements (no matrix_exp).

    Flat:      Ω_ij = Ω_i · Ω_j⁻¹  (cocycle condition automatic)
    Non-flat:  Ω_ij = Ω_i · exp(α·δ_ij·G) · Ω_j⁻¹

    Unlike compute_transport_operators (phi-based), this:
    - Stores Ω_i ∈ GL(K) directly (no Lie algebra coefficients)
    - No matrix_exp for per-token frames (only for edge-local connection if non-flat)
    - Covers full GL(K) including reflections (det < 0)
    - Simpler gradient: ∂KL/∂Ω_i via chain rule, no dexp series

    Args:
        omega: (B, N, K, K) per-token group elements Ω_i ∈ GL(K).
               Can have any determinant sign (reflections allowed).
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

    # Per-token inverse (needed for transport and gradients)
    omega_j_inv = torch.linalg.inv(omega)  # (B, N, K, K)

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
        # Flat: Ω_ij = Ω_i · Ω_j⁻¹ (cocycle condition automatic)
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
    results = []
    start = 0
    for d in irrep_dims:
        end = start + d
        omega_blk = omega[:, :, start:end, start:end].contiguous()
        omega_blk_inv = torch.linalg.inv(omega_blk)
        results.append((omega_blk, omega_blk_inv))
        start = end
    return results
