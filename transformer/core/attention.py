"""
KL-Divergence Based Attention Mechanism (0D Gauge Transformer)
===============================================================

Implements attention via information geometry instead of learned Q, K projections:

    β_ij = softmax(-KL(q_i || Ω_ij[q_j]) / κ)

where:
    - q_i = N(μ_i, Σ_i): Agent i's belief distribution
    - Ω_ij: Parallel transport operator (gauge connection)
    - KL: Kullback-Leibler divergence (information distance)
    - κ: Temperature parameter

Key Insight: NO W_Q, W_K matrices! Attention emerges from geometry.

0D Architecture:
    - All agents at single point c*
    - β_ij are scalars (not spatial fields)
    - No spatial integrals, just sums over agents

Author: Implementation from plan.py
Date: November 2025
"""

# Suppress noisy warnings BEFORE other imports
import warnings
warnings.filterwarnings("ignore", message="CUDA path could not be detected", module="cupy")
warnings.filterwarnings("ignore", message="Failed to find cuobjdump", module="triton")
warnings.filterwarnings("ignore", message="Failed to find nvdisasm", module="triton")


import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List, Union

from transformer.core.gauge_utils import (
    stable_matrix_exp_pair,
    newton_schulz_orthogonalize,
    fused_block_matrix_exp_pairs,
)
from transformer.core.kl_computation import (
    _kl_kernel_dense,
    _kl_kernel_diagonal,
)
from transformer.core.vfe_utils import _safe_spd_inv

# Import transport operators (extracted to transport_ops.py)
from transformer.core.transport_ops import (
    compute_transport_operators,
    compute_transport_operators_direct,
    omega_to_block_exp_pairs,
    _apply_rope,
    _apply_rope_to_covariance,
)

logger = logging.getLogger(__name__)

# KL divergence ceiling: kl_max = max(KL_CEIL_BASE, KL_CEIL_SCALE * dim).
# Single source of truth is vfe_utils.py.
from transformer.core.vfe_utils import KL_CEIL_BASE, KL_CEIL_SCALE
KL_CEIL_MULT = KL_CEIL_SCALE  # alias for backward compat within this module

try:
    from math_utils.generators import generate_so3_generators
    TRANSPORT_AVAILABLE = True
except ImportError:
    TRANSPORT_AVAILABLE = False
    logger.warning("Transport module not available")


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Core attention functions
    'compute_attention_weights',
    'aggregate_messages',

    # KL divergence computation
    'compute_kl_matrix_from_phi',

    # Multi-head attention class
    'IrrepMultiHeadAttention',

    # Utilities
    'create_attention_mask',
    'compute_transport_operators',

    # Constants for checking availability
    'TRANSPORT_AVAILABLE',
]


# =============================================================================
# Sparse Attention Patterns
# =============================================================================

def create_attention_mask(
    num_agents: int,
    pattern: str = 'full',
    window: int = 64,
    device: torch.device = torch.device('cpu'),
    causal: bool = True,
) -> torch.Tensor:
    """
    Create attention mask.

    Args:
        num_agents: Number of agents (sequence length)
        pattern: 'full' (only supported pattern)
        window: Unused, kept for API compatibility
        device: Device to create tensor on
        causal: If True, apply causal masking (i can't attend to j>i)

    Returns:
        mask: (N, N) binary mask where 1 = can attend, 0 = cannot attend
    """
    N = num_agents
    mask = torch.ones(N, N, device=device)

    if causal:
        causal_mask = torch.tril(torch.ones(N, N, device=device))
        mask = mask * causal_mask

    return mask


# =============================================================================
# Transport Operator Caching (for evolve_phi=False optimization)
# =============================================================================
# compute_transport_operators, compute_transport_operators_direct,
# omega_to_block_exp_pairs, _apply_rope, and _build_rope_freqs have been
# extracted to transformer/core/transport_ops.py and imported above.
# =============================================================================


# =============================================================================
# Core Attention: KL-Based Weights
# =============================================================================

def compute_attention_weights(
    mu_q: torch.Tensor,        # (B, N, K) belief means
    sigma_q: torch.Tensor,     # (B, N, K, K) or (B, N, K) if diagonal_covariance=True
    phi: torch.Tensor,         # (B, N, phi_dim) gauge frames (phi_dim=3 for SO(3), N(N-1)/2 for SO(N), K² for GL(K))
    generators: torch.Tensor,  # (n_gen, K, K) Lie algebra generators
    kappa: float,              # Temperature
    epsilon: float = 1e-8,     # Numerical stability
    mask: Optional[torch.Tensor] = None,  # (B, N, N) causal mask
    return_kl: bool = False,   # Return KL matrix for loss computation
    diagonal_covariance: bool = False,  # Use diagonal sigma (B,N,K) instead of full (B,N,K,K)
    cached_transport: Optional[dict] = None,  # Precomputed transport operators (from compute_transport_operators)
    # Memory-efficient options
    irrep_dims: Optional[List[int]] = None,  # Block-diagonal structure [d₁, d₂, ...] for principled KL decomposition
    chunk_size: Optional[int] = None,  # Chunk size for memory-efficient computation (None = auto)
    # ALiBi-style positional bias (NEW!)
    alibi_slope: Optional[float] = None,  # If set, adds slope * (i-j) to logits for relative position
    # Gauge mode: 'learned' (per-token φ), 'trivial' (Ω = I), or 'constant' (per-head Ω)
    gauge_mode: str = 'learned',
    # Self-attention masking (prevents attention collapse)
    mask_self_attention: bool = False,  # If True, mask out diagonal (no self-attention)
    # Gauge group control
    enforce_orthogonal: bool = False,  # If True, enforce Ω ∈ SO(K) via Newton-Schulz
    # Rotary Position Embeddings (RoPE)
    use_rope: bool = False,            # If True, apply RoPE rotations to μ before KL computation
    rope_base: float = 10000.0,        # RoPE frequency base
    # Cached block exponentials (avoids redundant fused_block_matrix_exp_pairs calls)
    cached_block_exp_pairs: Optional[list] = None,
    # Exact diagonal transport: lift diagonal σ to full for exact Ω@diag(σ)@Ω^T
    exact_diagonal_transport: bool = False,
    # Direct Omega parameterization (gauge_param='omega')
    gauge_param: str = 'phi',  # 'phi' (Lie algebra) or 'omega' (direct GL(K))
    # Gauge-covariant numerical ridge: use eps * (g g^T) instead of eps * I.
    gauge_covariant_ridge: bool = False,
    omega: Optional[torch.Tensor] = None,  # (B, N, K, K) direct group elements (when gauge_param='omega')
    alpha_divergence: float = 1.0,  # Renyi alpha-divergence parameter (1.0 = KL)
    # Full-covariance SPD floor (forwarded to _kl_kernel_dense). None = no clamp.
    sigma_floor: Optional[float] = None,
    spd_floor_mode: str = 'eigclamp',
    enable_spd_diagnostics: bool = False,
    propagate_nonfinite: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Compute attention weights from KL divergences (0D version).

    Formula:
        β_ij = softmax_j(-KL(q_i || Ω_ij[q_j]) / κ)

    where Ω_ij = exp(φ_i) · exp(-φ_j) transports q_j to i's frame.

    0D Structure:
        - All agents at single point c*, so β_ij are SCALARS
        - No spatial fields β_ij(c), just one number per pair
        - No spatial integration, just O(N²) agent-pair loop

    Args:
        mu_q: Query belief means, shape (B, N, K)
              N = num_agents at single point c*
        sigma_q: Query covariances, shape (B, N, K, K) if full, (B, N, K) if diagonal
        phi: Gauge frames, shape (B, N, phi_dim) in Lie algebra
             phi_dim = 3 for SO(3), N(N-1)/2 for SO(N), K² for GL(K)
        generators: Lie algebra generators, shape (n_gen, K, K)
        kappa: Temperature parameter (higher = softer attention)
        epsilon: Softmax stability constant
        mask: Optional causal mask (B, N, N) - 0 masks out position
        diagonal_covariance: If True, sigma_q is (B,N,K) diagonal variances.
                            Uses O(N²×K) memory instead of O(N²×K²)!
        cached_transport: Optional precomputed transport operators from compute_transport_operators().
                         When evolve_phi=False, caching avoids redundant matrix exponentials.
        irrep_dims: Optional list of irrep block dimensions [d₁, d₂, ...].
                   When provided, uses block-diagonal KL computation which is:
                   1. Theoretically principled (respects gauge structure)
                   2. Memory efficient: O(N² × Σᵢdᵢ²) vs O(N² × K²)
                   For K=255 with 75×ℓ₀ + 30×ℓ₁ + 18×ℓ₂: ~82× memory savings!
        chunk_size: Optional chunk size for memory-efficient processing.
                   When provided, processes N×N attention in C×C chunks.
                   None = no chunking (fast but memory-hungry)
        alibi_slope: Optional ALiBi-style positional bias slope.
                    When set, adds slope * (i - j) to attention logits.
                    Negative values favor recent tokens (recency bias).
                    Unlike positional embeddings in μ or φ, this doesn't
                    affect transport Ω_ij, keeping attention content-based
                    while adding controlled positional information.
                    Typical values: -0.1 to -0.01 (for recency bias)
        gauge_mode: 'learned' for per-token gauge frames with full transport,
                   'trivial' for global frame (Ω = I, no transport),
                   'constant' for per-head learnable Ω (manuscript Limit 2).
                   Trivial mode computes raw KL(q_i || q_j) without rotation.
        mask_self_attention: If True, mask out diagonal of attention matrix.
                            This prevents the model from attending to itself,
                            forcing it to attend to other tokens. Critical for
                            preventing attention collapse since KL(q_i||q_i)=0
                            always makes self-attention the most attractive.
        enforce_orthogonal: If True, enforce Ω ∈ SO(K) via Newton-Schulz iteration.
        use_rope: If True, apply RoPE rotations to μ before KL computation.
        rope_base: RoPE frequency base (default 10000.0).
        exact_diagonal_transport: When True and diagonal_covariance=True, lifts σ
                                 to full via diag_embed for exact Ω@Σ@Ω^T transport,
                                 then extracts diagonal from result.

    Returns:
        beta: Attention weights, shape (B, N, N)
              beta[b, i, j] = attention from agent i to agent j
        kl_matrix: (Optional) KL divergence matrix (B, N, N) if return_kl=True
                   kl_matrix[b, i, j] = KL(q_i || Ω_ij[q_j])

    Example:
        >>> B, N, K = 2, 10, 32
        >>> mu = torch.randn(B, N, K)
        >>> sigma = torch.eye(K).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)
        >>> n_gen = 3  # SO(3); use N*(N-1)//2 for SO(N), K**2 for GL(K)
        >>> phi = torch.randn(B, N, n_gen) * 0.1
        >>> G = torch.from_numpy(generate_so3_generators(K)).float()  # (n_gen, K, K)
        >>> beta = compute_attention_weights(mu, sigma, phi, G, kappa=1.0)
        >>> beta.shape
        torch.Size([2, 10, 10])
        >>> beta.sum(dim=-1)  # Should sum to 1 (plus epsilon)
        tensor([[1.0000, 1.0000, ...], ...])
    """
    # =========================================================================
    # Direct Omega path: precompute transport and inject as cached_transport
    # =========================================================================
    if gauge_param == 'omega' and omega is not None:
        if cached_transport is None:
            cached_transport = compute_transport_operators_direct(
                omega=omega,
                gauge_mode=gauge_mode,
                generators=generators,
            )
        # For block-diagonal paths: convert per-head omega to block_exp_pairs format.
        # The KL functions expect (omega_h, omega_h_inv) in place of (exp_phi_h, exp_neg_phi_h).
        if irrep_dims is not None and cached_block_exp_pairs is None:
            cached_block_exp_pairs = omega_to_block_exp_pairs(omega, irrep_dims)

    # =========================================================================
    # Exact diagonal transport: lift diagonal σ to full and use full-cov paths.
    # KL output is (B,N,N) regardless, so no output conversion needed.
    # =========================================================================
    if exact_diagonal_transport and diagonal_covariance and sigma_q.dim() == 3:
        sigma_q = torch.diag_embed(sigma_q)  # (B, N, K) → (B, N, K, K)
        diagonal_covariance = False

    batch_size, num_agents, K = mu_q.shape
    device = mu_q.device
    dtype = mu_q.dtype

    # =========================================================================
    # RoPE: Apply position-dependent SO(2)^{K/2} rotations to belief means.
    #
    # This makes KL(q_i || Ω_ij[q_j]) sensitive to relative position (j-i)
    # by injecting a position-dependent SO(2)^{K/2} rotation into the
    # transport operator (mirroring the standard-transformer Q/K rotation).
    #
    # Two scopes of "applied only to scoring" matter here:
    #
    # 1. Attention vs. value path (handled by the caller).  RoPE is applied
    #    to μ used for attention scoring, NOT to μ used for message
    #    aggregation.  IrrepMultiHeadAttention.forward separates
    #    `mu_q_for_attn` (rope-rotated, for KL/scoring) from `mu_blocks_raw`
    #    (raw, for aggregate_messages).  See attention.py:1594-1605.  This
    #    factorization corresponds to the manuscript's "attention gauge ≠
    #    value gauge" decomposition (Section on RoPE, ~line 1760).
    #
    # 2. μ vs. Σ within the KL itself (handled here).  The current code
    #    rotates only μ; Σ is left raw.  Under the GL(K) framework, RoPE
    #    is a gauge transport restricted to SO(2)^{K/2}, and gauge transport
    #    acts on Gaussian beliefs by μ → Rμ, Σ → RΣR^T (sandwich product).
    #    The implementation here departs from that prescription: the
    #    Mahalanobis term is computed in rope-rotated mean coordinates with
    #    a raw covariance, which is neither a textbook KL between rotated
    #    Gaussians nor the manuscript's full gauge interpretation.  It
    #    matches the standard-transformer pattern (rotate Q/K, leave
    #    covariance untouched) and preserves the diagonal-σ optimization.
    #
    # The local `mu_q = _apply_rope(...)` reassignment is a Python local-
    # variable rebinding only; `_apply_rope` clones internally so the
    # caller's input tensor is not mutated.
    #
    # See `BlockConfig.rope_full_gauge` for the experimental flag that
    # enables the framework-consistent Σ rotation (at the cost of breaking
    # diagonal σ structure).
    # =========================================================================
    if use_rope:
        mu_q = _apply_rope(mu_q, base=rope_base)

    # =========================================================================
    # Compute all pairwise KL divergences: KL(q_i || Ω_ij[q_j])
    # Helper functions return KL tensors (not in-place) to preserve autograd graph
    # =========================================================================

    # Dispatch to the unified KL computation helper.
    # Priority: block-diagonal > chunked-diagonal > diagonal > chunked-full > full
    kl_matrix = _dispatch_kl_matrix(
        mu_q=mu_q,
        sigma_q=sigma_q,
        phi=phi,
        generators=generators,
        diagonal_covariance=diagonal_covariance,
        irrep_dims=irrep_dims,
        chunk_size=chunk_size,
        cached_transport=cached_transport,
        cached_block_exp_pairs=cached_block_exp_pairs,
        gauge_mode=gauge_mode,
        enforce_orthogonal=enforce_orthogonal,
        alpha_divergence=alpha_divergence,
        gauge_covariant_ridge=gauge_covariant_ridge,
        sigma_floor=sigma_floor,
        spd_floor_mode=spd_floor_mode,
        enable_spd_diagnostics=enable_spd_diagnostics,
        propagate_nonfinite=propagate_nonfinite,
    )

    # =========================================================================
    # Convert KL distances to attention weights
    # =========================================================================

    # DIMENSION-AWARE KL NORMALIZATION:
    # KL between K-dimensional Gaussians: KL = ½(trace + mahal - K + logdet)
    # KL magnitudes grow as O(K), so logit differences have std ∝ √K.
    # We divide by √K to normalize to O(1), analogous to 1/√d_k in standard attention.
    #
    # The parameter κ is related to the belief covariance: in the isotropic limit
    # Σ = σ²I, κ ∝ σ² (the covariance IS the temperature). In the full theory,
    # Σ provides per-head, per-token, per-direction temperature control; κ serves
    # as a convenient global scalar handle on attention sharpness.
    #
    # Effective temperature: τ_eff = κ · √K
    # In the dot-product form (σ absorbed into W_Q W_K^T): τ = √d_k
    # In the squared-distance form (½ from KL explicit):   τ = 2√d_k
    dim_scale = math.sqrt(max(K, 1))

    # Attention logits: -KL / (κ · √K)
    logits = -kl_matrix / (kappa * dim_scale)  # (B, N, N)

    # ==========================================================================
    # ALiBi-STYLE POSITIONAL BIAS: Add relative position information
    # Unlike positional embeddings in μ or φ, this doesn't affect transport Ω_ij.
    # The bias is purely additive: logits[i,j] += slope * (i - j)
    # Negative slope encourages attending to recent tokens (recency bias).
    # This provides explicit, controlled positional information while keeping
    # the gauge transport purely content-based.
    # ==========================================================================
    if alibi_slope is not None and alibi_slope != 0.0:
        B, N, _ = logits.shape
        positions = torch.arange(N, device=logits.device, dtype=logits.dtype)
        # rel_pos[i, j] = i - j (positive for future, negative for past)
        rel_pos = positions[:, None] - positions[None, :]  # (N, N)
        # Apply slope (typically negative to favor recent tokens)
        alibi_bias = alibi_slope * rel_pos  # (N, N)
        logits = logits + alibi_bias.unsqueeze(0)  # (B, N, N)

    # Apply causal mask if provided (BEFORE self-attention masking)
    if mask is not None:
        # mask[b, i, j] = 0 means agent i CANNOT attend to agent j
        # masked_fill is not in-place (returns new tensor), safe for autograd
        logits = logits.masked_fill(mask == 0, float('-inf'))

    # ==========================================================================
    # SELF-ATTENTION MASKING: Force model to attend to other tokens
    # KL(q_i||q_i)=0 always, making diagonal dominant. Masking diagonal forces
    # the model to learn meaningful attention patterns over other tokens.
    #
    # IMPORTANT: Only mask diagonal where there are OTHER valid targets!
    # With causal masking, position 0 can only attend to itself - masking it
    # would leave no valid targets, causing NaN in softmax.
    # ==========================================================================
    if mask_self_attention:
        B, N, _ = logits.shape
        diag_idx = torch.arange(N, device=logits.device)
        # Check which positions have at least one other valid target
        # A position has other targets if any off-diagonal element is not -inf
        has_other_targets = (logits != float('-inf')).sum(dim=-1) > 1  # (B, N)
        # No clone needed — logits is already a fresh tensor from the division/
        # masked_fill above, not a view into the autograd graph.
        # Apply masking only where safe (where there are other targets)
        diag_vals = logits[:, diag_idx, diag_idx]  # (B, N)
        masked_diag_vals = torch.where(
            has_other_targets,
            torch.full_like(diag_vals, float('-inf')),
            diag_vals
        )
        logits[:, diag_idx, diag_idx] = masked_diag_vals

    # Softmax over keys (dimension 2)
    beta = F.softmax(logits, dim=-1)  # (B, N, N)

    # Clamp only non-masked positions to epsilon for numerical stability,
    # preserving exact zeros from -inf masked positions (e.g. causal mask)
    masked_positions = (logits == float('-inf'))
    # Apply clamp only where positions are NOT masked, keeping masked positions at 0
    beta = torch.where(masked_positions, beta, beta.clamp(min=epsilon))
    # Re-normalize (guard against all-masked rows producing zero sums)
    beta_sum = beta.sum(dim=-1, keepdim=True).clamp(min=epsilon)
    beta = beta / beta_sum

    if return_kl:
        return beta, kl_matrix
    else:
        return beta


def compute_kl_matrix_from_phi(
    mu_q: torch.Tensor,        # (B, N, K) belief means
    sigma_q: torch.Tensor,     # (B, N, K, K) or (B, N, K) if diagonal_covariance=True
    phi: torch.Tensor,         # (B, N, n_gen) gauge frames
    generators: torch.Tensor,  # (n_gen, K, K) Lie algebra generators
    diagonal_covariance: bool = False,
    irrep_dims: Optional[List[int]] = None,
    chunk_size: Optional[int] = None,
    gauge_mode: str = 'learned',
    enforce_orthogonal: bool = False,  # If True, enforce Ω ∈ SO(K) via Newton-Schulz
    exact_diagonal_transport: bool = False,
    alpha_divergence: float = 1.0,
) -> torch.Tensor:
    """
    Compute pairwise KL divergence matrix: KL(q_i || Ω_ij[q_j]).

    This is a convenience function that directly returns the KL matrix
    without computing attention weights. Useful for debugging, loss
    computation, and analysis.

    Args:
        mu_q: (B, N, K) belief means
        sigma_q: (B, N, K, K) full covariances or (B, N, K) diagonal variances
        phi: (B, N, n_gen) gauge frames
        generators: (n_gen, K, K) Lie algebra generators
        diagonal_covariance: If True, sigma_q is (B,N,K) diagonal variances
        irrep_dims: Optional list of irrep block dimensions for block-diagonal mode
        chunk_size: Optional chunk size for memory-efficient computation
        gauge_mode: 'learned' for full transport, 'trivial' for Ω = I,
                   'constant' for per-head learned Ω (manuscript Limit 2)
        enforce_orthogonal: If True, enforce Ω ∈ SO(K) via Newton-Schulz iteration
        exact_diagonal_transport: When True and diagonal_covariance=True, lifts σ
                                 to full via diag_embed for exact Ω@Σ@Ω^T transport,
                                 then extracts diagonal from result.

    Returns:
        kl_matrix: (B, N, N) pairwise KL divergences
                   kl_matrix[b, i, j] = KL(q_i || Ω_ij[q_j])

    Example:
        >>> B, N, K = 2, 10, 32
        >>> mu = torch.randn(B, N, K)
        >>> sigma = torch.ones(B, N, K)  # Diagonal variances
        >>> n_gen = 3  # SO(3); use N*(N-1)//2 for SO(N), K**2 for GL(K)
        >>> phi = torch.randn(B, N, n_gen) * 0.1
        >>> G = generate_so3_generators(K)  # (n_gen, K, K)
        >>> kl = compute_kl_matrix_from_phi(mu, sigma, phi, G, diagonal_covariance=True)
        >>> kl.shape
        torch.Size([2, 10, 10])
    """
    # Exact diagonal transport: lift to full
    if exact_diagonal_transport and diagonal_covariance and sigma_q.dim() == 3:
        sigma_q = torch.diag_embed(sigma_q)
        diagonal_covariance = False

    batch_size, num_agents, K = mu_q.shape
    device = mu_q.device
    dtype = mu_q.dtype

    # Delegate to the unified dispatch helper.
    return _dispatch_kl_matrix(
        mu_q=mu_q,
        sigma_q=sigma_q,
        phi=phi,
        generators=generators,
        diagonal_covariance=diagonal_covariance,
        irrep_dims=irrep_dims,
        chunk_size=chunk_size,
        cached_transport=None,
        cached_block_exp_pairs=None,
        gauge_mode=gauge_mode,
        enforce_orthogonal=enforce_orthogonal,
        alpha_divergence=alpha_divergence,
    )


# _compute_kl_matrix_torch, _transport_gaussian_torch, _kl_gaussian_torch,
# _compute_kl_matrix_diagonal, _compute_kl_matrix_chunked, and
# _compute_kl_matrix_diagonal_chunked have been consolidated into
# transformer/core/kl_computation.py.
# The unified dispatch for all modes lives in _dispatch_kl_matrix below.


def _dispatch_kl_matrix(
    mu_q: torch.Tensor,
    sigma_q: torch.Tensor,
    phi: torch.Tensor,
    generators: torch.Tensor,
    diagonal_covariance: bool,
    irrep_dims: Optional[List[int]],
    chunk_size: Optional[int],
    cached_transport: Optional[dict],
    cached_block_exp_pairs: Optional[list],
    gauge_mode: str,
    enforce_orthogonal: bool,
    alpha_divergence: float = 1.0,
    gauge_covariant_ridge: bool = False,
    sigma_floor: Optional[float] = None,
    spd_floor_mode: str = 'eigclamp',
    enable_spd_diagnostics: bool = False,
    propagate_nonfinite: bool = False,
) -> torch.Tensor:
    r"""Unified KL matrix dispatcher for all covariance modes and gauge configurations.

    Handles transport-operator construction then delegates to
    ``transformer.core.kl_computation.compute_kl_matrix``.

    Priority order:
        1. BLOCK_DIAGONAL (diagonal sigma)
        2. BLOCK_DIAGONAL (full sigma) — optionally chunked
        3. DENSE or DIAGONAL — build Omega, call unified kernel

    Covariance transport invariant: Sigma_transported = Omega @ Sigma @ Omega.T
    (the sandwich product is never bypassed).

    Args:
        mu_q: (B, N, K) belief means.
        sigma_q: (B, N, K) diagonal or (B, N, K, K) full covariance.
        phi: (B, N, n_gen) gauge field coefficients.
        generators: (n_gen, K, K) Lie algebra generators.
        diagonal_covariance: If True, sigma_q is (B, N, K).
        irrep_dims: Block dimensions for block-diagonal mode.
        chunk_size: Chunk size for memory-bounded computation.
        cached_transport: Optional precomputed transport dict.
        cached_block_exp_pairs: Optional precomputed block exponentials.
        gauge_mode: 'learned', 'trivial', or 'constant'.
        enforce_orthogonal: If True, apply Newton-Schulz to exp pairs.

    Returns:
        kl_matrix: (B, N, N) pairwise KL divergences.
    """
    from transformer.core.kl_computation import (
        KLMode,
        compute_kl_matrix as _unified_kl,
    )

    B, N, K = mu_q.shape
    device = mu_q.device
    dtype = mu_q.dtype
    eps = 1e-6
    kl_max = max(KL_CEIL_BASE, KL_CEIL_MULT * K)

    # =========================================================================
    # 1. BLOCK-DIAGONAL MODE
    # Fused kernels in gauge_utils.py handle transport internally.
    # =========================================================================
    if irrep_dims is not None:
        # Squeeze trailing singleton dimensions for diagonal sigma
        if diagonal_covariance:
            while sigma_q.dim() > 3 and sigma_q.shape[-1] == 1:
                sigma_q = sigma_q.squeeze(-1)
            sigma_q = sigma_q.clamp(min=eps)

        # Build or reuse block exponential pairs
        if cached_block_exp_pairs is not None:
            bep = cached_block_exp_pairs
        else:
            bep = fused_block_matrix_exp_pairs(
                phi, generators, irrep_dims,
                enforce_orthogonal=enforce_orthogonal,
            )

        if diagonal_covariance:
            # Block-diagonal + diagonal sigma: fastest path
            return _unified_kl(
                mu_q=mu_q,
                sigma_q=sigma_q,
                mu_transported=None,  # not used in BLOCK_DIAGONAL mode
                sigma_transported=None,
                mode=KLMode.BLOCK_DIAGONAL,
                block_exp_pairs=bep,
                irrep_dims=irrep_dims,
                kl_max=kl_max,
                eps=eps,
                alpha_divergence=alpha_divergence,
                sigma_floor=sigma_floor,
            )
        else:
            # Block-diagonal + full sigma
            # chunk_size is handled inside fused_block_diagonal_kl_full's _tile_size.
            # The chunked variant below is retained for cases where the caller
            # explicitly requests chunking over position pairs (different from
            # the internal Omega tile in fused_block_diagonal_kl_full).
            if chunk_size is not None:
                # Chunked block-diagonal: process position pairs in tiles.
                # Build exp lists and loop manually (replicates old chunked path).
                block_exp_phi = [p[0] for p in bep]
                block_exp_neg_phi = [p[1] for p in bep]

                row_chunks: list = []
                for i_start in range(0, N, chunk_size):
                    i_end = min(i_start + chunk_size, N)
                    n_i = i_end - i_start
                    col_chunks: list = []
                    for j_start in range(0, N, chunk_size):
                        j_end = min(j_start + chunk_size, N)
                        n_j = j_end - j_start

                        kl_chunk = torch.zeros(B, n_i, n_j, device=device, dtype=dtype)
                        block_start = 0
                        for block_idx, d in enumerate(irrep_dims):
                            block_end = block_start + d
                            I_d = torch.eye(d, device=device, dtype=dtype)

                            ep_i = block_exp_phi[block_idx][:, i_start:i_end].contiguous()
                            en_j = block_exp_neg_phi[block_idx][:, j_start:j_end].contiguous()
                            Omega_b = torch.einsum('bikl,bjlm->bijkm', ep_i, en_j)

                            mu_i = mu_q[:, i_start:i_end, block_start:block_end].contiguous()
                            mu_j = mu_q[:, j_start:j_end, block_start:block_end].contiguous()
                            sig_i = sigma_q[:, i_start:i_end, block_start:block_end,
                                            block_start:block_end].contiguous()
                            sig_j = sigma_q[:, j_start:j_end, block_start:block_end,
                                            block_start:block_end].contiguous()

                            mu_t = torch.einsum('bijkl,bjl->bijk', Omega_b, mu_j)
                            sig_t = torch.einsum(
                                'bijkl,bjlm,bijmn->bijkn',
                                Omega_b, sig_j, Omega_b.transpose(-1, -2)
                            )
                            del Omega_b

                            mu_i_exp = mu_i[:, :, None, :].expand(-1, -1, n_j, -1)
                            sig_i_exp = sig_i[:, :, None, :, :].expand(-1, -1, n_j, -1, -1)

                            # Gauge-covariant ridge at position i (covers both
                            # Σ_i and the transport-sandwich Σ_t, since Σ_t
                            # lives in position-i's frame ep_i after Ω_ij Σ_j Ω_ij^T).
                            if gauge_covariant_ridge:
                                _ep_i_bcast = ep_i[:, :, None, :, :].expand(-1, -1, n_j, -1, -1)
                                _R_d = _ep_i_bcast @ _ep_i_bcast.transpose(-1, -2)
                            else:
                                _R_d = I_d
                            _exp_phi_kw = _ep_i_bcast if gauge_covariant_ridge else None

                            kl_b = _kl_kernel_dense(
                                mu_i_exp, sig_i_exp + eps * _R_d,
                                mu_t, sig_t + eps * _R_d,
                                kl_max=max(KL_CEIL_BASE, KL_CEIL_MULT * d),
                                eps=eps,
                                exp_phi_q=_exp_phi_kw,
                                exp_phi_t=_exp_phi_kw,
                                sigma_floor=sigma_floor,
                                spd_floor_mode=spd_floor_mode,
                                enable_spd_diagnostics=enable_spd_diagnostics,
                                propagate_nonfinite=propagate_nonfinite,
                                alpha_div=alpha_divergence,
                            )
                            kl_chunk = kl_chunk + kl_b
                            del sig_t, mu_t
                            block_start = block_end

                        col_chunks.append(kl_chunk)
                    row_chunks.append(torch.cat(col_chunks, dim=2))
                return torch.cat(row_chunks, dim=1)
            else:
                return _unified_kl(
                    mu_q=mu_q,
                    sigma_q=sigma_q,
                    mu_transported=None,
                    sigma_transported=None,
                    mode=KLMode.BLOCK_DIAGONAL,
                    block_exp_pairs=bep,
                    irrep_dims=irrep_dims,
                    kl_max=kl_max,
                    eps=eps,
                    alpha_divergence=alpha_divergence,
                    sigma_floor=sigma_floor,
                )

    # =========================================================================
    # 2. DENSE / DIAGONAL MODE: build Omega, transport, then call unified kernel
    # =========================================================================

    # --- Build / cache exp pairs and optionally Omega ---
    # Hoist matrix_exp computation above dispatch branches so it's never redundant.
    exp_phi = exp_neg_phi = Omega = None

    if gauge_mode == 'trivial' or (gauge_mode == 'constant' and cached_transport is None):
        pass  # Identity transport: Omega stays None
    elif cached_transport is not None and 'Omega' in cached_transport:
        Omega = cached_transport['Omega']   # (B, N, N, K, K)
    elif (cached_transport is not None
          and 'exp_phi' in cached_transport
          and 'exp_neg_phi' in cached_transport):
        exp_phi = cached_transport['exp_phi']
        exp_neg_phi = cached_transport['exp_neg_phi']
        # Only materialize full Omega if we won't chunk (avoids O(BN²K²) allocation)
        if chunk_size is None:
            Omega = torch.einsum('bikl,bjlm->bijkm', exp_phi, exp_neg_phi)
    else:
        # Compute exp pairs once — reused by both unchunked and chunked paths
        phi_matrix = torch.einsum('bna,aij->bnij', phi, generators)
        exp_phi, exp_neg_phi = stable_matrix_exp_pair(phi_matrix)
        if enforce_orthogonal and K >= 2:
            exp_phi = newton_schulz_orthogonalize(exp_phi)
            exp_neg_phi = newton_schulz_orthogonalize(exp_neg_phi)
        # Only materialize full Omega if we won't chunk
        if chunk_size is None:
            Omega = torch.einsum('bikl,bjlm->bijkm', exp_phi, exp_neg_phi)

    # --- Transport ---
    if diagonal_covariance:
        while sigma_q.dim() > 3 and sigma_q.shape[-1] == 1:
            sigma_q = sigma_q.squeeze(-1)
        sigma_q = sigma_q.clamp(min=eps)

        if Omega is None:
            # Identity: mu_transported = mu_q broadcast, sigma transported = sigma_q broadcast
            # Views suffice — downstream KL kernels cast to float32 internally (no in-place mutation)
            mu_t = mu_q[:, None, :, :].expand(-1, N, -1, -1)
            sig_t = sigma_q[:, None, :, :].expand(-1, N, -1, -1)
        else:
            mu_t = torch.einsum('bijkl,bjl->bijk', Omega, mu_q)
            # AMP guard: covariance sandwich product must stay float32
            with torch.amp.autocast('cuda', enabled=False):
                _sq = sigma_q if sigma_q.dtype == torch.float32 else sigma_q.float()
                _Om = Omega if Omega.dtype == torch.float32 else Omega.float()
                sigma_j = _sq[:, None, :, :].expand(-1, N, -1, -1)
                sig_t = torch.einsum(
                    'bijkl,bijkl,bijl->bijk', _Om, _Om, sigma_j
                ).clamp(min=eps)

        # Call unified kernel (with optional chunking)
        if chunk_size is not None and exp_phi is not None:
            # Chunked diagonal: use pre-computed exp pairs (hoisted above)
            row_chunks = []
            for i_start in range(0, N, chunk_size):
                i_end = min(i_start + chunk_size, N)
                n_i = i_end - i_start
                ep_i = exp_phi[:, i_start:i_end].contiguous()
                mu_i = mu_q[:, i_start:i_end].contiguous()
                sig_i = sigma_q[:, i_start:i_end].contiguous()

                col_chunks = []
                for j_start in range(0, N, chunk_size):
                    j_end = min(j_start + chunk_size, N)
                    n_j = j_end - j_start
                    en_j = exp_neg_phi[:, j_start:j_end].contiguous()
                    mu_j = mu_q[:, j_start:j_end].contiguous()
                    sig_j = sigma_q[:, j_start:j_end].contiguous()

                    Omega_c = torch.einsum('bikl,bjlm->bijkm', ep_i, en_j)
                    mu_tc = torch.einsum('bijkl,bjl->bijk', Omega_c, mu_j)
                    # AMP guard: covariance sandwich product must stay float32
                    with torch.amp.autocast('cuda', enabled=False):
                        _sj = sig_j if sig_j.dtype == torch.float32 else sig_j.float()
                        _Oc = Omega_c if Omega_c.dtype == torch.float32 else Omega_c.float()
                        sig_j_exp = _sj[:, None, :, :].expand(-1, n_i, -1, -1)
                        sig_tc = torch.einsum(
                            'bijkl,bijkl,bijl->bijk', _Oc, _Oc, sig_j_exp
                        ).clamp(min=eps)
                    del Omega_c

                    mu_i_exp = mu_i[:, :, None, :].expand(-1, -1, n_j, -1)
                    sig_i_exp = sig_i[:, :, None, :].expand(-1, -1, n_j, -1)
                    kl_c = _kl_kernel_diagonal(
                        mu_i_exp, sig_i_exp, mu_tc, sig_tc,
                        kl_max=kl_max, eps=eps,
                        alpha_div=alpha_divergence,
                    )
                    del sig_tc, mu_tc
                    col_chunks.append(kl_c)
                row_chunks.append(torch.cat(col_chunks, dim=2))
            return torch.cat(row_chunks, dim=1)
        else:
            # Unchunked diagonal (includes chunked-with-cached-transport case)
            return _unified_kl(
                mu_q=mu_q,
                sigma_q=sigma_q,
                mu_transported=mu_t,
                sigma_transported=sig_t,
                mode=KLMode.DIAGONAL,
                chunk_size=None,  # already pre-expanded; no extra chunking needed
                kl_max=kl_max,
                eps=eps,
                alpha_divergence=alpha_divergence,
            )
    else:
        # Full covariance
        if Omega is None:
            mu_t = mu_q[:, None, :, :].expand(-1, N, -1, -1)
            sig_t = sigma_q[:, None, :, :, :].expand(-1, N, -1, -1, -1)
        else:
            mu_t = torch.einsum('bijkl,bjl->bijk', Omega, mu_q)
            # AMP guard: covariance sandwich product must stay float32
            with torch.amp.autocast('cuda', enabled=False):
                _Om = Omega if Omega.dtype == torch.float32 else Omega.float()
                _sq = sigma_q if sigma_q.dtype == torch.float32 else sigma_q.float()
                sig_t = torch.einsum(
                    'bijkl,bjlm,bijmn->bijkn',
                    _Om, _sq, _Om.transpose(-1, -2)
                )
                sig_t = 0.5 * (sig_t + sig_t.transpose(-1, -2))

        if chunk_size is not None and exp_phi is not None:
            # Full-covariance chunked: use pre-computed exp pairs (hoisted above)
            row_chunks = []
            for i_start in range(0, N, chunk_size):
                i_end = min(i_start + chunk_size, N)
                n_i = i_end - i_start
                ep_i = exp_phi[:, i_start:i_end].contiguous()
                mu_i = mu_q[:, i_start:i_end].contiguous()
                sig_i = sigma_q[:, i_start:i_end].contiguous()

                col_chunks = []
                for j_start in range(0, N, chunk_size):
                    j_end = min(j_start + chunk_size, N)
                    n_j = j_end - j_start
                    en_j = exp_neg_phi[:, j_start:j_end].contiguous()
                    mu_j = mu_q[:, j_start:j_end].contiguous()
                    sig_j = sigma_q[:, j_start:j_end].contiguous()

                    Omega_c = torch.einsum('bikl,bjlm->bijkm', ep_i, en_j)
                    mu_tc = torch.einsum('bijkl,bjl->bijk', Omega_c, mu_j)
                    # AMP guard: covariance sandwich product must stay float32
                    with torch.amp.autocast('cuda', enabled=False):
                        _Oc = Omega_c if Omega_c.dtype == torch.float32 else Omega_c.float()
                        _sj = sig_j if sig_j.dtype == torch.float32 else sig_j.float()
                        sig_tc = torch.einsum(
                            'bijkl,bjlm,bijmn->bijkn',
                            _Oc, _sj, _Oc.transpose(-1, -2)
                        )
                        sig_tc = 0.5 * (sig_tc + sig_tc.transpose(-1, -2))
                    del Omega_c

                    mu_i_exp = mu_i[:, :, None, :].expand(-1, -1, n_j, -1)
                    sig_i_exp = sig_i[:, :, None, :, :].expand(-1, -1, n_j, -1, -1)
                    I = torch.eye(K, device=device, dtype=dtype)
                    # Gauge-covariant ridge: ep_i here is the full-K per-position
                    # frame (exp_phi[:, i_start:i_end]). Σ_tc lives at position i
                    # after the Ω sandwich, so it also transforms with ep_i.
                    if gauge_covariant_ridge:
                        _ep_i_bcast_K = ep_i[:, :, None, :, :].expand(-1, -1, n_j, -1, -1)
                        _R_K = _ep_i_bcast_K @ _ep_i_bcast_K.transpose(-1, -2)
                    else:
                        _R_K = I
                    _exp_phi_kw_K = _ep_i_bcast_K if gauge_covariant_ridge else None
                    kl_c = _kl_kernel_dense(
                        mu_i_exp, sig_i_exp + eps * _R_K,
                        mu_tc, sig_tc + eps * _R_K,
                        kl_max=kl_max, eps=eps,
                        alpha_div=alpha_divergence,
                        exp_phi_q=_exp_phi_kw_K,
                        exp_phi_t=_exp_phi_kw_K,
                        sigma_floor=sigma_floor,
                        spd_floor_mode=spd_floor_mode,
                        enable_spd_diagnostics=enable_spd_diagnostics,
                        propagate_nonfinite=propagate_nonfinite,
                    )
                    del sig_tc, mu_tc
                    col_chunks.append(kl_c)
                row_chunks.append(torch.cat(col_chunks, dim=2))
            return torch.cat(row_chunks, dim=1)

        # Unchunked full covariance (or fallback when chunked not applicable)
        return _unified_kl(
            mu_q=mu_q,
            sigma_q=sigma_q,
            mu_transported=mu_t,
            sigma_transported=sig_t,
            mode=KLMode.DENSE,
            chunk_size=None,
            kl_max=kl_max,
            eps=eps,
            alpha_divergence=alpha_divergence,
            sigma_floor=sigma_floor,
            spd_floor_mode=spd_floor_mode,
            enable_spd_diagnostics=enable_spd_diagnostics,
            propagate_nonfinite=propagate_nonfinite,
        )


# =============================================================================
# Message Aggregation with Parallel Transport
# =============================================================================

def aggregate_messages(
    mu_q: torch.Tensor,         # (B, N, K)
    sigma_q: torch.Tensor,      # (B, N, K, K) or (B, N, K) if diagonal
    phi: torch.Tensor,          # (B, N, phi_dim) gauge frames
    beta: torch.Tensor,         # (B, N, N) attention weights
    generators: torch.Tensor,   # (n_gen, K, K) Lie algebra generators
    aggregate_mode: str = 'mean_only',  # 'mean_only' or 'full_distribution'
    diagonal_covariance: bool = False,
    cached_transport: Optional[dict] = None,  # Precomputed transport operators
    exact_diagonal_transport: bool = False,
    sigma_aggregation: str = 'mixture',  # 'mixture' or 'precision'
    gauge_covariant_ridge: bool = False,  # If True, ε·I ridges become ε·(gg^T)
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    r"""
    Aggregate messages with gauge transport.

    m_i = Σ_j β_ij Ω_ij μ_j  (primal transport for all gauge groups)

    0D Version: Simple weighted sum over agents, no spatial integration!

    Two modes:
        1. 'mean_only': Only aggregate means (faster)
           Returns: (messages, None)

        2. 'full_distribution': Aggregate means + covariances
           Returns: (mu_aggregated, sigma_aggregated)

    Sigma aggregation strategies (``sigma_aggregation``):

        'mixture' (default): Mixture-of-Gaussians moment matching.
            Cov(X) = E_Z[Cov(X|Z)] + Cov_Z(E[X|Z])
            Wider covariance that captures neighbor disagreement.
            Appropriate for single-step message passing.

        'precision': VFE equilibrium precision averaging
            (GL(K)_supplementary.tex, eq. sigma_fixed_point_beta).
            Σ_i^{-1} = Σ_j β_ij (Ω_ij Σ_j Ω_ij^T)^{-1}
            Tighter covariance (harmonic mean of precisions).
            Guaranteed PD, no catastrophic cancellation.

    Args:
        mu_q: Belief means (B, N, K)
        sigma_q: Belief covariances (B, N, K, K) full or (B, N, K) if diagonal
        phi: Gauge frames (B, N, phi_dim) in Lie algebra
        beta: Attention weights (B, N, N) - SCALARS, not fields!
        generators: Lie algebra generators (n_gen, K, K)
        aggregate_mode: 'mean_only' or 'full_distribution'
        diagonal_covariance: If True, sigma_q is (B, N, K) diagonal variances
        cached_transport: Optional dict with precomputed 'Omega' from compute_transport_operators()
        exact_diagonal_transport: When True and diagonal_covariance=True, lifts σ
                                 to full via diag_embed for exact Ω@Σ@Ω^T transport,
                                 then extracts diagonal from result.
        sigma_aggregation: 'mixture' (moment matching) or 'precision' (VFE equilibrium)

    Returns:
        mu_agg: Aggregated means (B, N, K)
        sigma_agg: Aggregated covariances (B, N, K, K), (B, N, K) if diagonal, or None

    Example:
        >>> mu_agg, _ = aggregate_messages(mu, sigma, phi, beta, G, mode='mean_only')
        >>> # mu_agg[b, i] = Σ_j β[b,i,j] * Ω_ij[μ[b,j]]
    """
    # Exact diagonal transport: lift to full, aggregate, extract diagonal
    _exact_diag_lift = exact_diagonal_transport and diagonal_covariance and sigma_q.dim() == 3
    if _exact_diag_lift:
        sigma_q = torch.diag_embed(sigma_q)  # (B, N, K) → (B, N, K, K)
        diagonal_covariance = False

    batch_size, num_agents, K = mu_q.shape
    device = mu_q.device
    dtype = mu_q.dtype

    # =========================================================================
    # FACTORED transport path: use per-position exp_phi/exp_neg_phi
    # instead of full pairwise Omega (B,N,N,K,K).
    # Memory: O(BNK²) vs O(BN²K²) — massive savings for large N.
    # =========================================================================
    _has_exp_pairs = (cached_transport is not None
                      and 'exp_phi' in cached_transport
                      and 'exp_neg_phi' in cached_transport)

    if _has_exp_pairs:
        _exp_phi = cached_transport['exp_phi']      # (B, N, K, K)
        _exp_neg_phi = cached_transport['exp_neg_phi']  # (B, N, K, K)

        # FACTORED mu aggregation (no Omega needed):
        # mu_agg_i = exp_phi_i @ Σ_j β_ij * (exp_neg_phi_j @ μ_j)
        w = torch.einsum('bjkl,bjl->bjk', _exp_neg_phi, mu_q)  # (B, N, K)
        w_weighted = torch.einsum('bij,bjk->bik', beta, w)  # (B, N, K)
        mu_aggregated = torch.einsum('bikl,bil->bik', _exp_phi, w_weighted)  # (B, N, K)

        # Covariance aggregation (if requested)
        if aggregate_mode == 'full_distribution':
            _tile_size = 16
            _eps = 1e-6
            if diagonal_covariance:
                sigma_q_diag = sigma_q
                while sigma_q_diag.dim() > 3 and sigma_q_diag.shape[-1] == 1:
                    sigma_q_diag = sigma_q_diag.squeeze(-1)

                if sigma_aggregation == 'precision':
                    # Precision aggregation: 1/σ_agg = Σ_j β_ij / σ_t_j
                    # AMP guard: sigma transport + division must stay float32
                    with torch.amp.autocast('cuda', enabled=False):
                        _f32 = torch.float32
                        precision_agg = torch.zeros(batch_size, num_agents, K,
                                                    device=device, dtype=_f32)
                        _sq_diag_f32 = sigma_q_diag if sigma_q_diag.dtype == _f32 else sigma_q_diag.float()
                        _ep_f32 = _exp_phi if _exp_phi.dtype == _f32 else _exp_phi.float()
                        _en_f32 = _exp_neg_phi if _exp_neg_phi.dtype == _f32 else _exp_neg_phi.float()
                        for i_start in range(0, num_agents, _tile_size):
                            i_end = min(i_start + _tile_size, num_agents)
                            ep_tile = _ep_f32[:, i_start:i_end]
                            Omega_tile = torch.einsum(
                                'bikl,bjlm->bijkm', ep_tile, _en_f32
                            )
                            sigma_t_tile = torch.einsum(
                                'bijkl,bijkl,bjl->bijk',
                                Omega_tile, Omega_tile, _sq_diag_f32
                            ).clamp(min=_eps)
                            precision_agg[:, i_start:i_end] = torch.einsum(
                                'bij,bijk->bik', beta[:, i_start:i_end].float(), 1.0 / sigma_t_tile
                            )
                            del Omega_tile
                        sigma_aggregated = (1.0 / precision_agg.clamp(min=_eps)).clamp(min=_eps)
                else:
                    # Mixture moment matching: Cov = E[Var] + Var[E]
                    # AMP guard: sigma transport must stay float32
                    with torch.amp.autocast('cuda', enabled=False):
                        _f32 = torch.float32
                        sigma_agg_accum = torch.zeros(batch_size, num_agents, K,
                                                      device=device, dtype=_f32)
                        _sq_diag_f32 = sigma_q_diag if sigma_q_diag.dtype == _f32 else sigma_q_diag.float()
                        _ep_f32 = _exp_phi if _exp_phi.dtype == _f32 else _exp_phi.float()
                        _en_f32 = _exp_neg_phi if _exp_neg_phi.dtype == _f32 else _exp_neg_phi.float()
                        _mu_f32 = mu_q if mu_q.dtype == _f32 else mu_q.float()
                        for i_start in range(0, num_agents, _tile_size):
                            i_end = min(i_start + _tile_size, num_agents)
                            ep_tile = _ep_f32[:, i_start:i_end]
                            Omega_tile = torch.einsum(
                                'bikl,bjlm->bijkm', ep_tile, _en_f32
                            )
                            sigma_t_tile = torch.einsum(
                                'bijkl,bijkl,bjl->bijk',
                                Omega_tile, Omega_tile, _sq_diag_f32
                            ).clamp(min=_eps)
                            mu_t_tile = torch.einsum(
                                'bijkl,bjl->bijk', Omega_tile, _mu_f32
                            )
                            second_moment_tile = sigma_t_tile + mu_t_tile ** 2
                            sigma_agg_accum[:, i_start:i_end] = torch.einsum(
                                'bij,bijk->bik', beta[:, i_start:i_end].float(), second_moment_tile
                            )
                            del Omega_tile
                        sigma_aggregated = (sigma_agg_accum - mu_aggregated.float() ** 2).clamp(min=_eps)
            else:
                # AMP guard: full covariance transport, inv, eigh must stay float32
                with torch.amp.autocast('cuda', enabled=False):
                    _f32 = torch.float32
                    _ep_f32 = _exp_phi if _exp_phi.dtype == _f32 else _exp_phi.float()
                    _en_f32 = _exp_neg_phi if _exp_neg_phi.dtype == _f32 else _exp_neg_phi.float()
                    _sq_f32 = sigma_q if sigma_q.dtype == _f32 else sigma_q.float()
                    _mu_f32 = mu_q if mu_q.dtype == _f32 else mu_q.float()
                    if sigma_aggregation == 'precision':
                        # Full covariance precision aggregation
                        precision_agg = torch.zeros(batch_size, num_agents, K, K,
                                                    device=device, dtype=torch.float32)
                        I_K = torch.eye(K, device=device, dtype=torch.float32)
                        for i_start in range(0, num_agents, _tile_size):
                            i_end = min(i_start + _tile_size, num_agents)
                            ep_tile = _ep_f32[:, i_start:i_end]
                            Omega_tile = torch.einsum(
                                'bikl,bjlm->bijkm', ep_tile, _en_f32
                            )
                            Sigma_t = torch.einsum(
                                'bijkl,bjlm,bijmn->bijkn',
                                Omega_tile, _sq_f32, Omega_tile.transpose(-1, -2)
                            )
                            if gauge_covariant_ridge:
                                _ep_bcast_t = ep_tile[:, :, None, :, :].expand(-1, -1, _sq_f32.shape[1], -1, -1)
                                _R_K_tile = _ep_bcast_t @ _ep_bcast_t.transpose(-1, -2)
                            else:
                                _R_K_tile = I_K
                            Sigma_t = 0.5 * (Sigma_t + Sigma_t.transpose(-1, -2)) + _eps * _R_K_tile
                            # Cholesky-backed SPD inverse via _safe_spd_inv:
                            # Sigma_t is a symmetrized, eps-regularized sandwich
                            # product, so it is SPD by construction.  The
                            # Cholesky path is 2-3x more numerically stable
                            # than torch.linalg.inv for ill-conditioned GL(K)
                            # transports, and gauge-covariant as a precision:
                            # under g, Sigma_t → g Sigma_t g.T implies
                            # Sigma_t_inv → g^{-T} Sigma_t_inv g^{-1}.
                            Sigma_t_inv = _safe_spd_inv(
                                Sigma_t, eps=_eps,
                                exp_phi=(_ep_bcast_t if gauge_covariant_ridge else None),
                            )
                            precision_agg[:, i_start:i_end] = torch.einsum(
                                'bij,bijkl->bikl', beta[:, i_start:i_end].float(), Sigma_t_inv
                            )
                            del Omega_tile
                        if gauge_covariant_ridge:
                            _R_K_agg = _ep_f32 @ _ep_f32.transpose(-1, -2)
                        else:
                            _R_K_agg = I_K
                        precision_agg = 0.5 * (precision_agg + precision_agg.transpose(-1, -2)) + _eps * _R_K_agg
                        # Same Cholesky-backed inverse for the aggregated
                        # precision → covariance conversion.
                        sigma_aggregated = _safe_spd_inv(
                            precision_agg, eps=_eps,
                            exp_phi=(_ep_f32 if gauge_covariant_ridge else None),
                        )
                        sigma_aggregated = 0.5 * (sigma_aggregated + sigma_aggregated.transpose(-1, -2))
                    else:
                        # Full covariance mixture moment matching with SPD protection
                        sigma_agg_accum = torch.zeros(batch_size, num_agents, K, K,
                                                      device=device, dtype=torch.float32)
                        for i_start in range(0, num_agents, _tile_size):
                            i_end = min(i_start + _tile_size, num_agents)
                            ep_tile = _ep_f32[:, i_start:i_end]
                            Omega_tile = torch.einsum(
                                'bikl,bjlm->bijkm', ep_tile, _en_f32
                            )
                            Sigma_t = torch.einsum(
                                'bijkl,bjlm,bijmn->bijkn',
                                Omega_tile, _sq_f32, Omega_tile.transpose(-1, -2)
                            )
                            mu_t_tile = torch.einsum(
                                'bijkl,bjl->bijk', Omega_tile, _mu_f32
                            )
                            second_moment = Sigma_t + torch.einsum(
                                'bijk,bijl->bijkl', mu_t_tile, mu_t_tile
                            )
                            sigma_agg_accum[:, i_start:i_end] = torch.einsum(
                                'bij,bijkl->bikl', beta[:, i_start:i_end].float(), second_moment
                            )
                            del Omega_tile
                        sigma_aggregated = sigma_agg_accum - torch.einsum(
                            'bik,bil->bikl', mu_aggregated.float(), mu_aggregated.float()
                        )
                        # SPD protection: symmetrize + eigenvalue floor
                        sigma_aggregated = 0.5 * (sigma_aggregated + sigma_aggregated.transpose(-1, -2))
                        try:
                            eigvals, eigvecs = torch.linalg.eigh(sigma_aggregated)
                            eigvals = eigvals.clamp(min=1e-4)
                            sigma_aggregated = eigvecs * eigvals.unsqueeze(-2) @ eigvecs.transpose(-1, -2)
                        except (RuntimeError, torch.linalg.LinAlgError):
                            if gauge_covariant_ridge:
                                _R_fallback = _ep_f32 @ _ep_f32.transpose(-1, -2)
                            else:
                                _R_fallback = torch.eye(
                                    K, device=device, dtype=torch.float32)
                            sigma_aggregated = sigma_aggregated + 1e-3 * _R_fallback
        else:
            sigma_aggregated = None

        if _exact_diag_lift and sigma_aggregated is not None:
            sigma_aggregated = torch.diagonal(sigma_aggregated, dim1=-2, dim2=-1)
        return mu_aggregated, sigma_aggregated

    # =========================================================================
    # LEGACY path: full Omega (B,N,N,K,K) — used when no exp pairs cached
    # =========================================================================

    # Step 1: Get transport operators (use cached if available)
    exp_phi_legacy = None  # Per-position local frame (B, N, K, K); used only by covariant ridge path.
    if cached_transport is not None and 'Omega' in cached_transport:
        Omega = cached_transport['Omega']
        if gauge_covariant_ridge and 'exp_phi' in cached_transport:
            exp_phi_legacy = cached_transport['exp_phi']
    else:
        phi_matrix = torch.einsum('bna,aij->bnij', phi, generators)
        exp_phi, exp_neg_phi = stable_matrix_exp_pair(phi_matrix)
        Omega = torch.einsum('bikl,bjlm->bijkm', exp_phi, exp_neg_phi)
        if gauge_covariant_ridge:
            exp_phi_legacy = exp_phi

    # Step 2: Transport all means (primal: m_i = Σ β_ij Ω_ij μ_j)
    mu_transported = torch.einsum('bijkl,bjl->bijk', Omega, mu_q)  # (B, N, N, K)

    # Step 3: Weighted aggregation: m_i = Σ_j β_ij * μ_j^{→i}
    mu_aggregated = torch.einsum('bij,bijk->bik', beta, mu_transported)  # (B, N, K)

    # Step 4: Covariance aggregation
    if aggregate_mode == 'full_distribution':
        B, N, K = mu_q.shape
        _eps = 1e-6
        if diagonal_covariance:
            sigma_q_diag = sigma_q
            while sigma_q_diag.dim() > 3 and sigma_q_diag.shape[-1] == 1:
                sigma_q_diag = sigma_q_diag.squeeze(-1)

            sigma_transported_diag = torch.einsum(
                'bijkl,bijkl,bjl->bijk', Omega, Omega, sigma_q_diag
            ).clamp(min=_eps)

            if sigma_aggregation == 'precision':
                precision_agg = torch.einsum(
                    'bij,bijk->bik', beta, 1.0 / sigma_transported_diag
                )
                sigma_aggregated = (1.0 / precision_agg.clamp(min=_eps)).clamp(min=_eps)
            else:
                second_moment = sigma_transported_diag + mu_transported ** 2
                sigma_aggregated = torch.einsum('bij,bijk->bik', beta, second_moment)
                sigma_aggregated = (sigma_aggregated - mu_aggregated ** 2).clamp(min=_eps)
        else:
            Sigma_transported = torch.einsum(
                'bijkl,bjlm,bijmn->bijkn',
                Omega, sigma_q, Omega.transpose(-1, -2)
            )

            if sigma_aggregation == 'precision':
                Sigma_transported = 0.5 * (Sigma_transported + Sigma_transported.transpose(-1, -2))
                I_K = torch.eye(K, device=Sigma_transported.device, dtype=Sigma_transported.dtype)
                # Sigma_transported is (B, N, N, K, K); local frame at position i
                # broadcast over j is exp_phi_legacy[:, :, None, :, :].
                if gauge_covariant_ridge and exp_phi_legacy is not None:
                    _ep_legacy_bcast = exp_phi_legacy[:, :, None, :, :].expand(
                        -1, -1, Sigma_transported.shape[2], -1, -1
                    )
                    _R_legacy_t = _ep_legacy_bcast @ _ep_legacy_bcast.transpose(-1, -2)
                    _R_legacy_agg = exp_phi_legacy @ exp_phi_legacy.transpose(-1, -2)
                else:
                    _ep_legacy_bcast = None
                    _R_legacy_t = I_K
                    _R_legacy_agg = I_K
                Sigma_transported = Sigma_transported + _eps * _R_legacy_t
                # Cholesky-backed SPD inverse via _safe_spd_inv.  See the
                # tiled path above for the gauge-covariance argument.
                Sigma_t_inv = _safe_spd_inv(
                    Sigma_transported, eps=_eps, exp_phi=_ep_legacy_bcast,
                )
                precision_agg = torch.einsum('bij,bijkl->bikl', beta, Sigma_t_inv)
                precision_agg = 0.5 * (precision_agg + precision_agg.transpose(-1, -2)) + _eps * _R_legacy_agg
                sigma_aggregated = _safe_spd_inv(
                    precision_agg, eps=_eps,
                    exp_phi=(exp_phi_legacy if gauge_covariant_ridge else None),
                )
                sigma_aggregated = 0.5 * (sigma_aggregated + sigma_aggregated.transpose(-1, -2))
            else:
                second_moment = Sigma_transported + torch.einsum(
                    'bijk,bijl->bijkl', mu_transported, mu_transported
                )
                sigma_aggregated = torch.einsum('bij,bijkl->bikl', beta, second_moment)
                sigma_aggregated = sigma_aggregated - torch.einsum(
                    'bik,bil->bikl', mu_aggregated, mu_aggregated
                )
                # SPD protection
                sigma_aggregated = 0.5 * (sigma_aggregated + sigma_aggregated.transpose(-1, -2))
                try:
                    eigvals, eigvecs = torch.linalg.eigh(sigma_aggregated)
                    eigvals = eigvals.clamp(min=1e-4)
                    sigma_aggregated = eigvecs * eigvals.unsqueeze(-2) @ eigvecs.transpose(-1, -2)
                except (RuntimeError, torch.linalg.LinAlgError):
                    if gauge_covariant_ridge and exp_phi_legacy is not None:
                        _gf_leg = exp_phi_legacy.to(dtype=sigma_aggregated.dtype)
                        _R_eigh_fallback = _gf_leg @ _gf_leg.transpose(-1, -2)
                    else:
                        _R_eigh_fallback = torch.eye(
                            K, device=sigma_aggregated.device, dtype=sigma_aggregated.dtype)
                    sigma_aggregated = sigma_aggregated + 1e-3 * _R_eigh_fallback
    else:
        sigma_aggregated = None

    if _exact_diag_lift and sigma_aggregated is not None:
        sigma_aggregated = torch.diagonal(sigma_aggregated, dim1=-2, dim2=-1)
    return mu_aggregated, sigma_aggregated


# =============================================================================
# Multi-Head Attention with Irrep Structure
# =============================================================================

class IrrepMultiHeadAttention(nn.Module):
    """
    Multi-head attention using KL divergence on gauge-transported belief distributions.

    Supports SO(3), SO(N), and GL(K) gauge groups. Heads correspond to irreducible
    representations (SO(3)/SO(N)) or block-diagonal GL(d_head) factors.

    Standard Transformer:
        - n_heads separate (W_Q, W_K, W_V) projections
        - Head dim = embed_dim / n_heads

    Gauge Transformer:
        - NO W_Q, W_K! (attention from KL divergence on transported beliefs)
        - Heads = irrep blocks (SO(3)/SO(N)) or GL(d_head) blocks
        - Each head has its own gauge transport Ω_ij

    Features:
        - RoPE: Position-dependent SO(2)^{K/2} rotations (use_rope, rope_base)
        - ALiBi: Additive positional bias on logits (alibi_slope)
        - Self-attention masking (mask_self_attention)
        - Orthogonal projection to SO(K) (enforce_orthogonal)
        - Gauge modes: 'learned' (per-token φ), 'trivial' (Ω=I), 'constant' (per-head Ω)
        - Exact diagonal transport (exact_diagonal_transport)
        - Optional W_O output projection (use_output_projection)

    Irrep Decomposition (SO(3) example, 96-dim):
        K = 12×1 + 7×3 + 5×5 + 2×7 = 96
        ℓ0: 12 scalar channels (gauge-invariant)
        ℓ1: 7 vector channels (transform as vectors)
        ℓ2: 5 rank-2 tensor channels
        ℓ3: 2 rank-3 tensor channels
    """

    def __init__(
        self,
        embed_dim: int,
        irrep_spec: List[Tuple[str, int, int]],
        kappa_beta: float,
        epsilon: float = 1e-8,
        aggregate_mode: str = 'mean_only',
        diagonal_covariance: bool = False,
        attention_pattern: str = 'full',  # Kept for API compat; only 'full' supported
        attention_window: int = 64,  # Unused, kept for API compat
        gauge_group: str = 'SO3',  # 'SO3' or 'SON'
        gauge_dim: int = 3,        # N for SO(N) - only used when gauge_group='SON'
        global_generators: Optional[torch.Tensor] = None,  # (n_gen, K, K) for SO(N) mode
        alibi_slope: Optional[float] = None,  # ALiBi-style positional bias (negative = recency bias)
        gauge_mode: str = 'learned',  # 'learned', 'trivial' (Ω = I), or 'constant' (per-head Ω)
        mask_self_attention: bool = False,  # If True, mask out diagonal (no self-attention)
        enforce_orthogonal: bool = False,  # If True, enforce Ω ∈ SO(K) via Newton-Schulz
        use_output_projection: bool = False,  # If True, add W_O linear projection after heads
        use_equivariant_head_mixer: bool = False,  # Commutant mixer (n² scalars) applied to μ and M·Σ·Mᵀ.
                                                   # Gauge-equivariant only under TIED gauges across heads
                                                   # (shared Ω factor on each irrep copy). Warns when the
                                                   # effective gauge is per-head independent (e.g. default GLK
                                                   # multi-head). Zero cost when disabled.
        irrep_dims_override: Optional[List[int]] = None,  # Override block dims (for cross-head coupling)
        use_rope: bool = False,  # If True, apply RoPE rotations to μ before KL computation
        rope_base: float = 10000.0,  # RoPE frequency base
        exact_diagonal_transport: bool = False,  # Lift diagonal σ for exact transport
        sigma_aggregation: str = 'mixture',  # 'mixture' or 'precision'
        learnable_head_kappa: bool = False,  # If True, learn per-head κ_h
        alpha_divergence: float = 1.0,  # Renyi alpha-divergence parameter (1.0 = KL)
        gauge_covariant_ridge: bool = False,  # If True, ε·I ridges become ε·(gg^T)
    ):
        """
        Initialize irrep-structured multi-head attention.

        Args:
            embed_dim: Total embedding dimension K
            irrep_spec: List of (label, multiplicity, dim) tuples
                Example: [('ℓ0', 12, 1), ('ℓ1', 7, 3), ...]
            kappa_beta: Temperature for attention softmax
            learnable_head_kappa: If True, learn per-head κ_h (initialized to kappa_beta * sqrt(d_h))
            epsilon: Numerical stability constant
            aggregate_mode: 'mean_only' or 'full_distribution'
            diagonal_covariance: If True, sigma is (B,N,K) diagonal variances
            attention_pattern: Only 'full' is supported (kept for API compatibility)
            attention_window: Unused, kept for API compatibility
            gauge_group: 'SO3', 'SON', or 'GLK'
            gauge_dim: N for SO(N) mode - determines generator structure
            global_generators: Pre-computed generators (n_gen, K, K).
                              Required when gauge_group='SON' or 'GLK'.
            alibi_slope: ALiBi-style positional bias slope. Negative = recency bias.
            gauge_mode: 'learned' (per-token φ), 'trivial' (Ω=I), or
                       'constant' (per-head learnable Ω)
            mask_self_attention: If True, mask out diagonal (no self-attention).
                                Prevents attention collapse since KL(q_i||q_i)=0.
            enforce_orthogonal: If True, enforce Ω ∈ SO(K) via Newton-Schulz.
            use_output_projection: If True, add learned W_O ∈ R^{K×K} after heads.
            irrep_dims_override: Override block dims (for cross-head coupling).
            use_rope: If True, apply RoPE rotations to μ before KL computation.
            rope_base: RoPE frequency base (default 10000.0).
            exact_diagonal_transport: When True and diagonal_covariance=True, lifts σ
                                     to full via diag_embed for exact Ω@Σ@Ω^T transport,
                                     then extracts diagonal from result.
        """
        super().__init__()
        self.diagonal_covariance = diagonal_covariance
        self.exact_diagonal_transport = exact_diagonal_transport
        self.sigma_aggregation = sigma_aggregation
        self.embed_dim = embed_dim
        self.kappa_beta = kappa_beta
        self.epsilon = epsilon
        self.aggregate_mode = aggregate_mode
        self.alibi_slope = alibi_slope
        self.gauge_mode = gauge_mode
        self.mask_self_attention = mask_self_attention
        self.enforce_orthogonal = enforce_orthogonal
        self.use_rope = use_rope
        self.rope_base = rope_base
        # Tri-state RoPE σ-rotation mode in {'off', 'vfe_only', 'both'}.
        # The attention sublayer applies σ → R Σ Rᵀ (alongside μ → R μ) only
        # when set to 'both'. Set externally by GaugeTransformerBlock.__init__
        # so attention's rotation policy mirrors the FFN's. Requires
        # diagonal_covariance=False (the lifted-σ path is the only one with
        # well-defined R Σ Rᵀ semantics; diagonal σ is lifted upstream).
        self.rope_full_gauge_mode = 'off'
        self.alpha_divergence = alpha_divergence
        self.gauge_covariant_ridge = gauge_covariant_ridge

        # Build irrep block structure
        self.irrep_dims = []
        self.irrep_labels = []
        total_dim = 0

        # =================================================================
        # GL(K) MODE: Single-head or multi-head GL(K) attention
        # =================================================================
        # GL(K) has no natural irrep decomposition like SO(K). However, we can
        # still use multi-head attention via block-diagonal structure:
        #   GL(d_head)^H ⊂ GL(K)
        # where d_head = K/H and each head has its own GL(d_head) gauge.
        #
        # Determine number of GL(K) heads from irrep_spec:
        #   - [('fund', H, d_head)] → H heads of dimension d_head
        #   - [('full', 1, K)] → single head (original behavior)
        if gauge_group == 'GLK':
            # Check if multi-head is requested via irrep_spec
            if len(irrep_spec) == 1 and irrep_spec[0][0] == 'full':
                # Single-head GL(K): original behavior
                self.irrep_dims = [embed_dim]
                self.irrep_labels = ['full']
                total_dim = embed_dim
                self.glk_multihead = False
                logger.info(f"[GL(K) mode] Single-head attention: dim={embed_dim}, generators={embed_dim}²={embed_dim**2}")
            else:
                # Multi-head GL(K): block-diagonal structure
                # Parse irrep_spec as [(label, n_heads, d_head)]
                label, n_heads, d_head = irrep_spec[0]
                if n_heads * d_head != embed_dim:
                    raise ValueError(
                        f"GL(K) multi-head: n_heads({n_heads}) × d_head({d_head}) = {n_heads * d_head} "
                        f"must equal embed_dim={embed_dim}"
                    )

                total_dim = embed_dim
                if irrep_dims_override is not None:
                    # Cross-head coupling: use super-block dims from merge_coupled_heads.
                    # Generators have been reordered so super-blocks are contiguous.
                    # NOTE: each super-block represents one or more *original* heads
                    # that were transitively connected by cross_couplings; per-block
                    # quantities downstream (e.g. learnable kappa) are therefore
                    # per-super-block, not per-original-head.
                    self.irrep_dims = list(irrep_dims_override)
                    self.irrep_labels = [f'glk_superblock_{i}' for i in range(len(irrep_dims_override))]
                    self.glk_multihead = True
                    logger.info(
                        "[GL(K) cross-head] super-blocks=%s d_head=%d "
                        "(merged from %d original heads; per-block params are "
                        "per-super-block, not per-original-head).",
                        irrep_dims_override, d_head, n_heads,
                    )
                else:
                    self.irrep_dims = [d_head] * n_heads
                    self.irrep_labels = [f'glk_head_{h}' for h in range(n_heads)]
                    self.glk_multihead = True
                    logger.info(f"[GL(K) multi-head] {n_heads} heads × GL({d_head}), generators per head={d_head}²={d_head**2}")
        else:
            # SO(3) / SO(N) mode: Use irrep decomposition
            for label, multiplicity, dim in irrep_spec:
                for _ in range(multiplicity):
                    self.irrep_dims.append(dim)
                    self.irrep_labels.append(label)
                    total_dim += dim

            # Pad to embed_dim if needed - add SCALAR heads (dim=1), not one big head
            if total_dim < embed_dim:
                padding = embed_dim - total_dim
                for _ in range(padding):
                    self.irrep_dims.append(1)  # Each padding is a scalar head
                    self.irrep_labels.append('ℓ0_pad')
                total_dim = embed_dim
            elif total_dim > embed_dim:
                raise ValueError(
                    f"Irrep spec sums to {total_dim}, exceeds embed_dim={embed_dim}"
                )

        self.n_heads = len(self.irrep_dims)
        self.learnable_head_kappa = learnable_head_kappa

        # =================================================================
        # Per-head learnable temperature κ_h
        # =================================================================
        # Manuscript: β_ij^(a) = softmax(-KL / (κ_a √d_h))
        # κ_h is the bare per-head kappa; compute_attention_weights applies √d_h.
        # Learnable: κ_h = clamp(exp(log_kappa[h]), 0.5×κ₀, 1.5×κ₀)
        if learnable_head_kappa:
            init_kappas = torch.tensor([
                kappa_beta for _d_h in self.irrep_dims
            ])
            # Length is len(self.irrep_dims), the number of *effective blocks*.
            # In the cross-head-coupled path this equals the number of
            # super-blocks (after merge_coupled_heads), NOT the original head
            # count — so a learnable kappa is per-super-block in that path,
            # shared across the original heads merged into the super-block.
            # The parameter name "log_kappa_per_head" is preserved for state-dict
            # compatibility; see the kappa_per_super_block read-only property
            # below for the post-coupling-aware accessor.
            self.log_kappa_per_head = nn.Parameter(torch.log(init_kappas))
            self.register_buffer('_kappa_init', init_kappas)
        else:
            self.log_kappa_per_head = None
            self._kappa_init = None


        # =================================================================
        # Create generators for each head dimension
        # =================================================================
        # SO(3) mode:
        #   - For ℓ=0 (dim=1): Zero generator (scalars don't transform)
        #   - For ℓ≥1 (dim=3,5,7,...): Proper Wigner D-matrix generators
        #
        # SO(N) mode:
        #   - Use global generators (block-diagonal structure)
        #   - Extract appropriate blocks for each head
        #
        # Store as a list of buffers (can't use ParameterList since non-trainable)
        self.head_generators = nn.ModuleList()  # Will hold generator-holding modules

        # Track cumulative dimension for extracting blocks in SO(N) mode
        cum_dim = 0

        for head_idx, dim in enumerate(self.irrep_dims):
            if gauge_group == 'SO3':
                # SO(3) mode: Create Wigner D-matrix generators per head
                if dim == 1:
                    # Scalar irrep: zero generator (no transformation)
                    gen = torch.zeros(3, 1, 1)
                elif dim % 2 == 1 and dim >= 3:
                    # Proper SO(3) irrep: use Wigner D-matrix generators
                    gen_np = generate_so3_generators(dim)
                    gen = torch.from_numpy(gen_np).float()
                else:
                    # Even dimension - not a valid SO(3) irrep!
                    raise ValueError(
                        f"Head {head_idx} has dim={dim}, which is not a valid SO(3) irrep dimension. "
                        f"SO(3) irreps must have odd dimensions (1, 3, 5, 7, ...). "
                        f"For even dimensions, use gauge_group='SON' with appropriate gauge_dim."
                    )
            elif gauge_group == 'GLK':
                # GL(K) mode: Single-head or multi-head
                if global_generators is None:
                    raise ValueError(
                        f"GL(K) mode requires global_generators to be provided."
                    )

                if hasattr(self, 'glk_multihead') and self.glk_multihead:
                    # Multi-head GL(K) (standard or cross-coupled):
                    # Extract the dim×dim spatial block from ALL generators.
                    # phi has coefficients for all n_gen generators, so we keep
                    # the full first axis to match phi's last dimension.
                    # Generators that are zero in this spatial block contribute
                    # nothing to the Lie algebra element Σ_a φ^a G_a[block].
                    gen = global_generators[:, cum_dim:cum_dim+dim, cum_dim:cum_dim+dim].clone()
                else:
                    # Single-head GL(K): Use full K² generators on entire space
                    gen = global_generators.clone()  # (K², K, K)
            else:
                # SO(N) mode: Extract block from global generators
                if global_generators is None:
                    raise ValueError(
                        f"SO(N) mode requires global_generators to be provided. "
                        f"Pass generators from model.py to IrrepMultiHeadAttention."
                    )
                # Extract the block for this head from the global block-diagonal generators
                # global_generators: (n_gen, K, K) where K = embed_dim
                n_gen = global_generators.shape[0]
                gen = global_generators[:, cum_dim:cum_dim+dim, cum_dim:cum_dim+dim].clone()
                # gen shape: (n_gen, dim, dim)

            # Wrap in a module to register as buffer
            gen_holder = nn.Module()
            gen_holder.register_buffer('gen', gen)
            self.head_generators.append(gen_holder)

            cum_dim += dim

        # =================================================================
        # Constant gauge: per-head learnable Ω ∈ GL(d_head)
        # =================================================================
        # When gauge_mode='constant', Ω_ij = Ω for all pairs (i,j).
        # This is the manuscript's Limit 2 (constant gauge specialization):
        # S(Ω) cancels under softmax, Ω⁻¹ absorbed into learned projections.
        # Unlike the cocycle parameterization (which forces constant → I),
        # this allows a free GL(K) matrix per head with direct gradient descent.
        if gauge_mode == 'constant':
            self.constant_omega = nn.ParameterList()
            total_omega_params = 0
            for dim in self.irrep_dims:
                omega = nn.Parameter(torch.eye(dim))  # Initialize to identity
                self.constant_omega.append(omega)
                total_omega_params += dim * dim
            logger.info(f"[Constant gauge] {self.n_heads} heads, per-head Ω ∈ GL(d_head)")
            logger.info(f"  -> {total_omega_params} learnable transport parameters (initialized to I)")
        else:
            self.constant_omega = None

        # =================================================================
        # W_O output projection (optional cross-head mixing)
        # =================================================================
        if use_output_projection:
            self.output_proj = nn.Linear(embed_dim, embed_dim, bias=False)
            logger.info(f"  W_O output projection: {embed_dim}×{embed_dim} = {embed_dim**2} params")
        else:
            self.output_proj = None

        # =================================================================
        # Gauge-equivariant head mixer (principled W_O replacement)
        # =================================================================
        # Under tied gauges (same Ω acting on every copy of an irrep), the
        # Schur commutant of the rep decomposition is a block sum ⊕_type M_n_t(ℝ)
        # where n_t is the multiplicity of irrep type t. We parameterize that
        # commutant as n_t × n_t scalar matrices per type; the mixer matrix
        # M ∈ R^{K×K} is built as kron(a_t, I_{d_t}) blocks placed at the head
        # slots of type t. Applied symmetrically: μ ↦ Mμ, Σ ↦ MΣM^T. At init
        # a_t = I_n_t so the mixer is the identity.
        #
        # IMPORTANT — gauge compatibility: this is strictly equivariant only
        # when every head assigned to the same irrep type shares the same Ω.
        # Under independent per-head gauges (default GLK multi-head with
        # cross_couplings=[]), Ω differs across heads and the mixer breaks
        # equivariance exactly as W_O does. We emit a warning in that case so
        # users opt in knowingly.
        if use_equivariant_head_mixer:
            from collections import defaultdict
            groups: Dict[Tuple[str, int], List[int]] = defaultdict(list)
            for h, (lbl, d_h) in enumerate(zip(self.irrep_labels, self.irrep_dims)):
                # Collapse 'glk_head_i' → 'glk_fund' so all GL(d) heads cluster.
                key = 'glk_fund' if lbl.startswith('glk_head_') else lbl
                groups[(key, d_h)].append(h)
            self._mixer_groups: List[Tuple[str, int, List[int]]] = [
                (key, dim, heads) for (key, dim), heads in groups.items()
            ]
            # Per-group n × n scalar parameter, identity-initialized.
            self.mixer_params = nn.ParameterList([
                nn.Parameter(torch.eye(len(heads)))
                for _, _, heads in self._mixer_groups
            ])
            total_params = sum(len(h) ** 2 for _, _, h in self._mixer_groups)
            logger.info(
                f"  equivariant head mixer: {len(self._mixer_groups)} irrep type(s), "
                f"{total_params} scalar parameter(s) (commutant of irrep decomp)"
            )
            # Check for independent per-head gauge and warn.
            _independent_glk = (
                gauge_group == 'GLK'
                and getattr(self, 'glk_multihead', False)
                and irrep_dims_override is None
                and any(len(h) > 1 for _, _, h in self._mixer_groups)
            )
            if _independent_glk:
                import warnings
                warnings.warn(
                    "use_equivariant_head_mixer=True with GL(K) multi-head and "
                    "cross_couplings=[]: the effective gauge is (GL(d_head))^n_heads "
                    "(independent per-head). The commutant mixer is strictly "
                    "equivariant only under a TIED gauge (same Ω on every head). "
                    "The mixer will train, but it breaks gauge equivariance in "
                    "this configuration — no better than use_output_projection. "
                    "For principled cross-head mixing under GL(K) heads, enable "
                    "cross_couplings in your config instead.",
                    RuntimeWarning, stacklevel=2,
                )
            self._mixer_active = True
        else:
            self._mixer_groups = []
            self.mixer_params = None
            self._mixer_active = False

        # Print attention configuration
        if gauge_group == 'GLK':
            # GL(K) mode: single head with full generators
            n_gen = global_generators.shape[0] if global_generators is not None else embed_dim**2
            logger.info(f"[GL(K) Attention] Single head, dim={embed_dim}, n_generators={n_gen}")
            logger.info(f"  -> Full GL({embed_dim}) transport on entire embedding space")
        else:
            # SO(3) / SO(N) mode: count scalar vs non-scalar heads
            n_scalar_heads = sum(1 for d in self.irrep_dims if d == 1)
            n_gauge_active_heads = self.n_heads - n_scalar_heads
            scalar_channels = sum(d for d in self.irrep_dims if d == 1)

            logger.info(f"IrrepMultiHeadAttention: {self.n_heads} heads, dims={self.irrep_dims}")

            # Warn if a large fraction of channels are gauge-invariant
            if n_scalar_heads > 0:
                import warnings
                scalar_fraction = scalar_channels / embed_dim
                if scalar_fraction > 0.5:
                    warnings.warn(
                        f"IrrepMultiHeadAttention: {n_scalar_heads}/{self.n_heads} heads are ℓ=0 (scalar), "
                        f"comprising {scalar_channels}/{embed_dim} = {100*scalar_fraction:.1f}% of channels. "
                        f"Scalar channels are GAUGE-INVARIANT: transport Ω_ij acts as identity, "
                        f"so gauge frame evolution (update_phi=True) won't affect them. "
                        f"Consider increasing non-scalar irreps (ℓ≥1) for gauge-sensitive representations.",
                        UserWarning
                    )
                logger.info(f"  -> {n_scalar_heads} scalar (l=0) heads: GAUGE-INVARIANT (Omega=I)")
                logger.info(f"  -> {n_gauge_active_heads} non-scalar heads: gauge-active (transport via Wigner D)")

        # Cache skew-symmetry for fused_block_matrix_exp_pairs (SO(K): exp(-M)=exp(M)^T)
        if global_generators is not None:
            self._generators_are_skew = torch.allclose(
                global_generators + global_generators.transpose(-1, -2),
                torch.zeros_like(global_generators), atol=1e-5,
            )
        else:
            self._generators_are_skew = False

    def forward(
        self,
        mu_q: torch.Tensor,
        sigma_q: torch.Tensor,
        phi: torch.Tensor,
        generators: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        cached_head_transports: Optional[List[dict]] = None,  # Cross-layer cache
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass through multi-head KL-divergence attention.

        Computes per-head KL(q_i || Ω_ij[q_j]) attention weights and
        aggregates gauge-transported messages.

        Args:
            mu_q: (B, N, K) belief means
            sigma_q: (B, N, K, K) full covariances or (B, N, K) if diagonal_covariance
            phi: (B, N, phi_dim) gauge frames in Lie algebra
            generators: (n_gen, K, K) Lie algebra generators
            mask: (B, N, N) optional causal mask
            return_attention: If True, return attention weights and KL matrices
            cached_head_transports: Optional list of precomputed transport dicts, one per head.
                                   When evolve_phi=False, this can be computed once at model
                                   entry and reused across all layers.

        Returns:
            mu_out: (B, N, K) updated means
            sigma_out: (B, N, K, K) or (B, N, K) updated covariances (or None)
            attention_weights: (B, n_heads, N, N) for visualization (or None)
            kl_matrices: (B, n_heads, N, N) KL divergences (or None)
        """
        batch_size, num_agents, K = mu_q.shape

        # =====================================================================
        # RoPE: Apply to FULL mu_q BEFORE splitting into heads
        # =====================================================================
        # RoPE frequency bands are computed as freq_i = 1/base^(2i/K).
        # Applying RoPE per-head (after split) gives each head identical
        # frequencies scaled by 1/d_h instead of non-overlapping bands from
        # the global 1/K spectrum. By applying to the full K-dimensional mu
        # first, each head's slice inherits the correct frequency band:
        #   head 0 → low frequencies (slow rotations)
        #   head H-1 → high frequencies (fast rotations)
        # This matches standard transformer RoPE which applies to full Q/K
        # before the head split.
        if self.use_rope:
            mu_q_for_attn = _apply_rope(mu_q, base=self.rope_base)
        else:
            mu_q_for_attn = mu_q

        # rope_full_gauge='both': also rotate Σ via the sandwich product
        # Σ → R Σ Rᵀ, matching the FFN VFE-helper's full-gauge convention.
        # This makes the attention β values consistent with the GL(K)-restricted-
        # to-SO(2)^{K/2} interpretation of RoPE through the WHOLE stack rather
        # than only inside the FFN VFE loop. Requires non-diagonal σ because
        # R Σ Rᵀ has off-diagonal terms in general; users opting into 'both'
        # accept the O(K) memory blow-up of full covariance.
        sigma_q_for_attn = sigma_q
        if self.use_rope and self.rope_full_gauge_mode == 'both':
            if self.diagonal_covariance:
                raise ValueError(
                    "rope_full_gauge='both' requires diagonal_covariance=False. "
                    "The R Σ Rᵀ rotation produces off-diagonal terms that "
                    "cannot be represented in diagonal form, so the attention "
                    "sublayer's σ rotation is undefined under diagonal covariance. "
                    "Either set rope_full_gauge='vfe_only' (FFN-only full-gauge), "
                    "or switch to diagonal_covariance=False."
                )
            # Lift if shape is (B, N, K); _apply_rope_to_covariance expects (..., K, K).
            if sigma_q.dim() == 3:
                sigma_q_for_attn = torch.diag_embed(sigma_q)
            sigma_q_for_attn = _apply_rope_to_covariance(sigma_q_for_attn, base=self.rope_base)

        # =====================================================================
        # Split into irrep blocks
        # =====================================================================
        # Split RoPE-rotated mu for attention scoring
        mu_blocks = self._split_irreps(mu_q_for_attn)  # List of (B, N, dim_ℓ)
        # Split raw mu for message aggregation (values — no RoPE)
        mu_blocks_raw = self._split_irreps(mu_q)        # List of (B, N, dim_ℓ)
        sigma_blocks = self._split_irreps_sigma(sigma_q_for_attn)  # List of (B, N, dim_ℓ, dim_ℓ)

        # =====================================================================
        # Process each head (irrep block)
        # =====================================================================
        head_outputs_mu = []
        head_outputs_sigma = []
        all_attention_weights = []
        all_kl_matrices = []

        # Precompute per-position exp_phi/exp_neg_phi for ALL heads in one
        # batched matrix_exp call (via fused_block_matrix_exp_pairs).
        # This replaces per-head compute_transport_operators which builds
        # full Omega (B,N,N,d,d) — an O(N²d²) memory hog.
        if self.gauge_mode == 'constant':
            # Constant gauge: skip matrix exponentials entirely.
            # Per-head Ω is a direct nn.Parameter; transport constructed in per-head loop.
            _attn_block_exp_pairs = None
        elif cached_head_transports is not None:
            # Cross-layer cache: reuse transport computed at model entry
            _attn_block_exp_pairs = None  # will use cached_head_transports per head
        else:
            # Collect per-head generators into the format expected by fused_block_matrix_exp_pairs
            # Build a combined generator tensor that is block-diagonal
            _head_gens = [
                self.head_generators[h].gen.to(
                    device=generators.device, dtype=generators.dtype
                ) for h in range(self.n_heads)
            ]
            n_gen = _head_gens[0].shape[0]
            # Build block-diagonal generators: (n_gen, K, K)
            _combined_gens = torch.zeros(n_gen, K, K,
                                         device=generators.device, dtype=generators.dtype)
            _start = 0
            for h, d_h in enumerate(self.irrep_dims):
                _combined_gens[:, _start:_start+d_h, _start:_start+d_h] = _head_gens[h]
                _start += d_h

            _attn_block_exp_pairs = fused_block_matrix_exp_pairs(
                phi, _combined_gens, self.irrep_dims,
                enforce_orthogonal=self.enforce_orthogonal,
                skew_symmetric=getattr(self, '_generators_are_skew', False),
            )

        for head_idx, (mu_head, mu_head_raw, sigma_head, dim_head, label) in enumerate(
            zip(mu_blocks, mu_blocks_raw, sigma_blocks, self.irrep_dims, self.irrep_labels)
        ):
            gen_head = self.head_generators[head_idx].gen.to(
                device=generators.device, dtype=generators.dtype
            )

            # Per-head temperature κ_h for attention softmax.
            # Learnable: κ_h = clamp(exp(log_kappa[h]), 0.5×κ₀, 1.5×κ₀).
            # Static: κ_h = kappa_beta * √d_h, normalizes KL across head dims.
            if self.learnable_head_kappa:
                kappa_h = torch.exp(self.log_kappa_per_head[head_idx])
                k0 = self._kappa_init[head_idx]
                kappa_h = kappa_h.clamp(min=0.5 * k0, max=1.5 * k0)
            else:
                kappa_h = self.kappa_beta

            if self.gauge_mode == 'constant':
                # Constant gauge: Ω_ij = Ω for all pairs.
                # Set exp_phi = Ω (broadcast to all positions), exp_neg_phi = I.
                # Factored aggregation: m_i = Ω @ Σ_j β_ij μ_j
                omega_h = self.constant_omega[head_idx]  # (d_head, d_head)
                if self.enforce_orthogonal and dim_head >= 2:
                    omega_h = newton_schulz_orthogonalize(
                        omega_h.unsqueeze(0)
                    ).squeeze(0)  # Project to O(K)
                eye_h = torch.eye(dim_head, device=omega_h.device, dtype=omega_h.dtype)
                # Expand to (B, N, d, d) matching expected shape
                exp_phi_h = omega_h.unsqueeze(0).unsqueeze(0).expand(
                    batch_size, num_agents, -1, -1
                ).contiguous()
                exp_neg_phi_h = eye_h.unsqueeze(0).unsqueeze(0).expand(
                    batch_size, num_agents, -1, -1
                ).contiguous()
                head_cached_transport = {
                    'exp_phi': exp_phi_h,
                    'exp_neg_phi': exp_neg_phi_h,
                }
                _head_bep = None
            elif cached_head_transports is not None:
                # Cross-layer cache: use precomputed per-position pairs
                head_cached_transport = cached_head_transports[head_idx]
                # If cache has per-position exp_phi/exp_neg_phi (not full Omega),
                # convert to block_exp_pairs for the fast block-diagonal KL path.
                if 'exp_phi' in head_cached_transport and 'Omega' not in head_cached_transport:
                    _head_bep = [(head_cached_transport['exp_phi'],
                                  head_cached_transport['exp_neg_phi'])]
                else:
                    _head_bep = None
            else:
                # Use per-position exp pairs (no full Omega!)
                head_cached_transport = {
                    'exp_phi': _attn_block_exp_pairs[head_idx][0],
                    'exp_neg_phi': _attn_block_exp_pairs[head_idx][1],
                }
                _head_bep = [_attn_block_exp_pairs[head_idx]]

            # Compute attention for this head
            # NOTE: use_rope=False here because RoPE was already applied to the
            # full mu_q BEFORE splitting into heads (see above). This ensures
            # each head inherits the correct frequency band from the global
            # spectrum rather than all heads getting identical d_h-based freqs.
            if return_attention:
                beta_head, kl_head = compute_attention_weights(
                    mu_head,
                    sigma_head,
                    phi,
                    gen_head,
                    kappa_h,
                    self.epsilon,
                    mask,
                    return_kl=True,
                    diagonal_covariance=self.diagonal_covariance,
                    cached_transport=head_cached_transport if _head_bep is None else None,
                    irrep_dims=[dim_head] if _head_bep is not None else None,
                    cached_block_exp_pairs=_head_bep,
                    alibi_slope=self.alibi_slope,
                    gauge_mode=self.gauge_mode,
                    mask_self_attention=self.mask_self_attention,
                    enforce_orthogonal=self.enforce_orthogonal,
                    use_rope=False,
                    exact_diagonal_transport=self.exact_diagonal_transport,
                    alpha_divergence=self.alpha_divergence,
                    gauge_covariant_ridge=self.gauge_covariant_ridge,
                )  # (B, N, N), (B, N, N)
                all_attention_weights.append(beta_head)
                all_kl_matrices.append(kl_head)
            else:
                beta_head = compute_attention_weights(
                    mu_head,
                    sigma_head,
                    phi,
                    gen_head,
                    kappa_h,
                    self.epsilon,
                    mask,
                    return_kl=False,
                    diagonal_covariance=self.diagonal_covariance,
                    cached_transport=head_cached_transport if _head_bep is None else None,
                    irrep_dims=[dim_head] if _head_bep is not None else None,
                    cached_block_exp_pairs=_head_bep,
                    alibi_slope=self.alibi_slope,
                    gauge_mode=self.gauge_mode,
                    mask_self_attention=self.mask_self_attention,
                    enforce_orthogonal=self.enforce_orthogonal,
                    use_rope=False,
                    exact_diagonal_transport=self.exact_diagonal_transport,
                    alpha_divergence=self.alpha_divergence,
                    gauge_covariant_ridge=self.gauge_covariant_ridge,
                )  # (B, N, N)
                kl_head = None

            # Aggregate messages using raw (un-rotated) mu — RoPE affects
            # attention scores only (the "Q·K" analog), not values.
            mu_agg, sigma_agg = aggregate_messages(
                mu_head_raw,
                sigma_head,
                phi,
                beta_head,
                gen_head,
                aggregate_mode=self.aggregate_mode,
                diagonal_covariance=self.diagonal_covariance,
                cached_transport=head_cached_transport,
                exact_diagonal_transport=self.exact_diagonal_transport,
                sigma_aggregation=self.sigma_aggregation,
                gauge_covariant_ridge=self.gauge_covariant_ridge,
            )

            head_outputs_mu.append(mu_agg)
            if sigma_agg is not None:
                head_outputs_sigma.append(sigma_agg)

        # =====================================================================
        # Concatenate head outputs
        # =====================================================================
        mu_concat = torch.cat(head_outputs_mu, dim=-1)  # (B, N, K)

        if head_outputs_sigma:
            # Block-diagonal covariance
            sigma_concat = self._block_diag_sigma(head_outputs_sigma)  # (B, N, K, K)
        else:
            sigma_concat = None

        # =====================================================================
        # Optional W_O output projection (cross-head mixing, gauge-breaking)
        # =====================================================================
        if self.output_proj is not None:
            mu_out = self.output_proj(mu_concat)  # (B, N, K) - learned cross-head mixing
        elif self._mixer_active:
            # Gauge-equivariant (under tied gauge) mixer: μ ↦ M·μ, Σ ↦ M·Σ·M^T.
            mu_out, sigma_concat = self._apply_equivariant_mixer(mu_concat, sigma_concat)
        else:
            mu_out = mu_concat  # (B, N, K) - pristine VFE, no mixing

        # Stack attention weights and KL matrices for loss computation
        if return_attention:
            attention_weights = torch.stack(all_attention_weights, dim=1)  # (B, n_heads, N, N)
            kl_matrices = torch.stack(all_kl_matrices, dim=1)  # (B, n_heads, N, N)
        else:
            attention_weights = None
            kl_matrices = None

        return mu_out, sigma_concat, attention_weights, kl_matrices

    def _build_mixer_matrix(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        r"""Assemble the K×K commutant mixer M from per-group scalar matrices.

        For each irrep group g with multiplicity n_g and dim d_g, the learned
        scalar matrix a_g ∈ R^{n_g×n_g} is expanded via kron(a_g, I_{d_g}) and
        placed at the head slots of that group. The resulting M lies in the
        Schur commutant under tied gauge action.
        """
        K = self.embed_dim
        M = torch.zeros(K, K, device=device, dtype=dtype)
        # Precompute head start offsets once.
        offsets = [0]
        for d_h in self.irrep_dims:
            offsets.append(offsets[-1] + d_h)
        for (_, dim, heads), a_param in zip(self._mixer_groups, self.mixer_params):
            a = a_param.to(device=device, dtype=dtype)
            I_d = torch.eye(dim, device=device, dtype=dtype)
            for i_out, h_out in enumerate(heads):
                r0, r1 = offsets[h_out], offsets[h_out] + dim
                for j_in, h_in in enumerate(heads):
                    c0, c1 = offsets[h_in], offsets[h_in] + dim
                    M[r0:r1, c0:c1] = a[i_out, j_in] * I_d
        return M

    def _apply_equivariant_mixer(
        self, mu: torch.Tensor, sigma: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""Apply the commutant mixer symmetrically to μ and Σ.

        μ' = M·μ, Σ' = M·Σ·M^T (symmetrized). For diagonal Σ, the transformed
        full covariance is projected back to diagonal (lossy approximation —
        full-covariance mode preserves the exact commutant action).
        """
        M = self._build_mixer_matrix(mu.device, mu.dtype)
        # Row-vector form: (B, N, K) @ M^T computes M @ μ in column-vector sense.
        mu_out = mu @ M.transpose(-1, -2)
        if sigma is None:
            return mu_out, None
        if sigma.dim() == 4:  # (B, N, K, K) full covariance
            sig_out = M @ sigma @ M.transpose(-1, -2)
            sig_out = 0.5 * (sig_out + sig_out.transpose(-1, -2))
            return mu_out, sig_out
        # Diagonal Σ: full transformation is non-diagonal in general; collapse
        # to the diagonal of M·diag(σ)·M^T so downstream (B, N, K) shape holds.
        sig_full = torch.diag_embed(sigma)
        sig_out_full = M @ sig_full @ M.transpose(-1, -2)
        sig_out = torch.diagonal(sig_out_full, dim1=-2, dim2=-1)
        return mu_out, sig_out

    def _split_irreps(self, mu: torch.Tensor) -> List[torch.Tensor]:
        """Split embedding into irrep blocks.

        Returns contiguous copies to avoid inplace modification issues during backward.
        """
        blocks = []
        start_idx = 0
        for dim in self.irrep_dims:
            # Use .contiguous() to create a copy, avoiding inplace modification issues
            blocks.append(mu[..., start_idx:start_idx+dim].contiguous())
            start_idx += dim
        return blocks

    def _split_irreps_sigma(self, sigma: torch.Tensor) -> List[torch.Tensor]:
        """Split covariance into irrep blocks.

        For full covariance (B, N, K, K): extracts diagonal blocks
        For diagonal (B, N, K): extracts slices

        Handles mismatches between expected and actual sigma format by converting
        between diagonal and full covariance representations.
        """
        # Squeeze trailing singleton dimensions first
        while sigma.dim() > 3 and sigma.shape[-1] == 1:
            sigma = sigma.squeeze(-1)

        # Detect actual sigma format based on shape
        sigma_is_diagonal = sigma.dim() == 3

        # Handle format mismatches
        if self.diagonal_covariance and not sigma_is_diagonal:
            # Attention expects diagonal (B, N, K) but got full covariance (B, N, K, K)
            # Extract diagonal from full covariance matrix
            # Use .clone() to avoid view-related gradient issues (diagonal returns a view)
            sigma = torch.diagonal(sigma, dim1=-2, dim2=-1).clone()  # (B, N, K, K) -> (B, N, K)
        elif not self.diagonal_covariance and sigma_is_diagonal:
            # Attention expects full (B, N, K, K) but got diagonal (B, N, K)
            # Convert diagonal to full covariance
            sigma = torch.diag_embed(sigma)  # (B, N, K) -> (B, N, K, K)

        blocks = []
        start_idx = 0
        for dim in self.irrep_dims:
            if self.diagonal_covariance:
                # Diagonal mode: sigma is (B, N, K), just slice
                # Use .contiguous() to create a copy, avoiding inplace modification issues
                blocks.append(sigma[..., start_idx:start_idx+dim].contiguous())
            else:
                # Full mode: sigma is (B, N, K, K), extract diagonal block
                # Use .contiguous() to create a copy, avoiding inplace modification issues
                blocks.append(
                    sigma[..., start_idx:start_idx+dim, start_idx:start_idx+dim].contiguous()
                )
            start_idx += dim
        return blocks

    def _block_diag_sigma(self, sigma_blocks: List[torch.Tensor]) -> torch.Tensor:
        """Construct covariance from irrep blocks.

        For diagonal mode: concatenates (B, N, dim) slices → (B, N, K)
        For full mode: builds block-diagonal (B, N, K, K)
        """
        batch_size, num_agents = sigma_blocks[0].shape[:2]
        K = sum(self.irrep_dims)

        if self.diagonal_covariance:
            # Diagonal mode: just concatenate along last dim
            return torch.cat(sigma_blocks, dim=-1)  # (B, N, K)
        else:
            # Full mode: build block-diagonal matrix
            sigma_full = torch.zeros(
                batch_size, num_agents, K, K,
                device=sigma_blocks[0].device,
                dtype=sigma_blocks[0].dtype
            )

            start_idx = 0
            for sigma_block, dim in zip(sigma_blocks, self.irrep_dims):
                sigma_full[..., start_idx:start_idx+dim, start_idx:start_idx+dim] = sigma_block
                start_idx += dim

            return sigma_full

    def precompute_head_transports(
        self,
        phi: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> List[dict]:
        """
        Precompute transport operators for all heads.

        Call this once at model entry when evolve_phi=False, then pass the result
        to forward() as cached_head_transports to skip redundant matrix exponentials.

        Args:
            phi: (B, N, phi_dim) gauge frames
            device: Device for generators
            dtype: Dtype for generators

        Returns:
            List of transport dicts, one per head. Each dict contains:
                'exp_phi': (B, N, dim, dim)
                'exp_neg_phi': (B, N, dim, dim)
                'Omega': (B, N, N, dim, dim)
        """
        cached_transports = []
        B, N = phi.shape[0], phi.shape[1]

        for head_idx in range(self.n_heads):
            if self.gauge_mode == 'constant':
                # Constant gauge: construct transport from per-head Ω parameter
                dim_h = self.irrep_dims[head_idx]
                omega_h = self.constant_omega[head_idx].to(device=device, dtype=dtype)
                if self.enforce_orthogonal and dim_h >= 2:
                    omega_h = newton_schulz_orthogonalize(omega_h.unsqueeze(0)).squeeze(0)
                eye_h = torch.eye(dim_h, device=device, dtype=dtype)
                exp_phi_h = omega_h.unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1).contiguous()
                exp_neg_phi_h = eye_h.unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1).contiguous()
                # Build full Omega (B, N, N, d, d) for cross-layer cache
                Omega_h = omega_h.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(
                    B, N, N, -1, -1
                ).contiguous()
                cached_transports.append({
                    'exp_phi': exp_phi_h,
                    'exp_neg_phi': exp_neg_phi_h,
                    'Omega': Omega_h,
                })
            else:
                gen_head = self.head_generators[head_idx].gen.to(device=device, dtype=dtype)
                cached_transports.append(compute_transport_operators(
                    phi, gen_head, enforce_orthogonal=self.enforce_orthogonal,
                    gauge_mode=self.gauge_mode
                ))
        return cached_transports

    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, "
            f"n_heads={self.n_heads}, "
            f"irrep_dims={self.irrep_dims[:3]}..., "
            f"kappa={self.kappa_beta}"
        )
