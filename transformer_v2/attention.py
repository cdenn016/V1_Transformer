# -*- coding: utf-8 -*-
"""
KL-Divergence Based Attention (Refactored)
============================================

Attention via information geometry:
    β_ij = softmax(-KL(q_i || Ω_ij[q_j]) / (κ · √K))

NO W_Q, W_K matrices — attention emerges from geometry.

Refactored from legacy attention.py (2,795 lines → ~500 lines):
- KL computation delegated to kl_ops.py
- Transport computation delegated to kl_ops.compute_transport_operators()
- Config-driven IrrepMultiHeadAttention (no 50+ params)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union

from transformer_v2.config import GaugeTransformerConfig
from transformer_v2.kl_ops import (
    compute_kl_matrix,
    compute_transport_operators,
)
from transformer_v2.gauge_utils import stable_matrix_exp_pair

try:
    from .generators import generate_so3_generators
    GENERATORS_AVAILABLE = True
except ImportError:
    GENERATORS_AVAILABLE = False


# =============================================================================
# RoPE: Rotary Position Embeddings for KL-Divergence Attention
# =============================================================================

def _build_rope_freqs(
    K: int, base: float = 10000.0,
    device: torch.device = None, dtype: torch.dtype = None,
) -> torch.Tensor:
    """Compute RoPE frequency bands for K-dimensional beliefs."""
    half_K = K // 2
    freqs = 1.0 / (base ** (torch.arange(0, half_K, device=device, dtype=dtype) / half_K))
    return freqs


def _apply_rope(mu: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """Apply Rotary Position Embeddings to belief means.

    Rotates consecutive pairs of dimensions by position-dependent angles,
    making KL divergences sensitive to relative position.
    """
    B, N, K = mu.shape
    half_K = K // 2

    freqs = _build_rope_freqs(K, base, device=mu.device, dtype=mu.dtype)
    positions = torch.arange(N, device=mu.device, dtype=mu.dtype)
    angles = torch.outer(positions, freqs)

    cos_angles = torch.cos(angles)
    sin_angles = torch.sin(angles)

    mu_even = mu[:, :, :2*half_K:2]
    mu_odd = mu[:, :, 1:2*half_K:2]

    mu_rotated = mu.clone()
    mu_rotated[:, :, :2*half_K:2] = mu_even * cos_angles - mu_odd * sin_angles
    mu_rotated[:, :, 1:2*half_K:2] = mu_even * sin_angles + mu_odd * cos_angles

    return mu_rotated


# =============================================================================
# Attention Mask
# =============================================================================

def create_attention_mask(
    num_agents: int,
    pattern: str = 'full',
    window: int = 64,
    device: torch.device = torch.device('cpu'),
    causal: bool = True,
) -> torch.Tensor:
    """Create attention mask.

    Args:
        num_agents: Sequence length
        pattern: Only 'full' supported
        window: Unused (API compatibility)
        device: Device
        causal: If True, apply causal masking

    Returns:
        mask: (N, N) binary mask
    """
    N = num_agents
    mask = torch.ones(N, N, device=device)
    if causal:
        mask = mask * torch.tril(torch.ones(N, N, device=device))
    return mask


# =============================================================================
# Core: compute_attention_weights
# =============================================================================

def compute_attention_weights(
    mu_q: torch.Tensor,
    sigma_q: torch.Tensor,
    phi: torch.Tensor,
    generators: torch.Tensor,
    kappa: float,
    epsilon: float = 1e-8,
    mask: Optional[torch.Tensor] = None,
    return_kl: bool = False,
    diagonal_covariance: bool = False,
    cached_transport: Optional[dict] = None,
    irrep_dims: Optional[List[int]] = None,
    chunk_size: Optional[int] = None,
    alibi_slope: Optional[float] = None,
    use_identity_transport: bool = False,
    mask_self_attention: bool = False,
    enforce_orthogonal: bool = False,
    use_rope: bool = False,
    rope_base: float = 10000.0,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Compute attention weights from KL divergences.

    β_ij = softmax_j(-KL(q_i || Ω_ij[q_j]) / (κ · √K))

    Args:
        mu_q: (B, N, K) belief means
        sigma_q: (B, N, K, K) or (B, N, K) covariances
        phi: (B, N, n_gen) gauge frames
        generators: (n_gen, K, K) Lie algebra generators
        kappa: Temperature parameter
        epsilon: Numerical stability constant
        mask: (B, N, N) optional causal mask (0 = masked)
        return_kl: Return KL matrix alongside attention weights
        diagonal_covariance: True if sigma_q is (B, N, K) diagonal
        cached_transport: Precomputed transport operators
        irrep_dims: Block dims for block-diagonal KL
        chunk_size: Chunk size for memory-efficient computation
        alibi_slope: ALiBi positional bias slope
        use_identity_transport: If True, Ω = I
        mask_self_attention: Mask diagonal to prevent attention collapse
        enforce_orthogonal: Project Ω to SO(K)
        use_rope: Apply RoPE rotations to μ
        rope_base: RoPE frequency base

    Returns:
        beta: (B, N, N) attention weights
        kl_matrix: (B, N, N) if return_kl=True
    """
    B, N, K = mu_q.shape

    # RoPE: position-dependent rotations on belief means
    if use_rope:
        mu_q = _apply_rope(mu_q, base=rope_base)

    # Compute pairwise KL divergences via unified kl_ops
    kl_matrix = compute_kl_matrix(
        mu_q, sigma_q, phi, generators,
        diagonal_covariance=diagonal_covariance,
        irrep_dims=irrep_dims,
        chunk_size=chunk_size,
        use_identity_transport=use_identity_transport,
        enforce_orthogonal=enforce_orthogonal,
        cached_transport=cached_transport,
    )

    # Dimension-aware normalization: τ = √K
    dim_scale = math.sqrt(max(K, 1))
    logits = -kl_matrix / (kappa * dim_scale)

    # ALiBi positional bias
    if alibi_slope is not None and alibi_slope != 0.0:
        positions = torch.arange(N, device=logits.device, dtype=logits.dtype)
        rel_pos = positions[:, None] - positions[None, :]
        logits = logits + (alibi_slope * rel_pos).unsqueeze(0)

    # Causal mask
    if mask is not None:
        logits = logits.masked_fill(mask == 0, float('-inf'))

    # Self-attention masking (only where safe)
    if mask_self_attention:
        diag_idx = torch.arange(N, device=logits.device)
        has_other_targets = (logits != float('-inf')).sum(dim=-1) > 1
        logits = logits.clone()
        diag_vals = logits[:, diag_idx, diag_idx]
        masked_diag_vals = torch.where(
            has_other_targets,
            torch.full_like(diag_vals, float('-inf')),
            diag_vals,
        )
        logits[:, diag_idx, diag_idx] = masked_diag_vals

    # Softmax → attention weights
    beta = F.softmax(logits, dim=-1)

    # Clamp non-masked positions for stability, preserve exact zeros from -inf
    masked_positions = (logits == float('-inf'))
    beta = torch.where(masked_positions, beta, beta.clamp(min=epsilon))
    beta_sum = beta.sum(dim=-1, keepdim=True).clamp(min=epsilon)
    beta = beta / beta_sum

    if return_kl:
        return beta, kl_matrix
    return beta


# =============================================================================
# Message Aggregation with Parallel Transport
# =============================================================================

def aggregate_messages(
    mu_q: torch.Tensor,
    sigma_q: torch.Tensor,
    phi: torch.Tensor,
    beta: torch.Tensor,
    generators: torch.Tensor,
    aggregate_mode: str = 'mean_only',
    diagonal_covariance: bool = False,
    cached_transport: Optional[dict] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Aggregate messages with GL(K) metric correction.

    For SO(K): m_i = Σ_j β_ij Ω_ij μ_j            (Ω orthogonal)
    For GL(K): m_i = Σ_j β_ij Ω_ij^{-T} μ_j       (metric correction)

    Args:
        mu_q: (B, N, K) belief means
        sigma_q: (B, N, K, K) or (B, N, K) covariances
        phi: (B, N, n_gen) gauge frames
        beta: (B, N, N) attention weights
        generators: (n_gen, K, K) generators
        aggregate_mode: 'mean_only' or 'full_distribution'
        diagonal_covariance: True if sigma_q is (B, N, K)
        cached_transport: Precomputed transport operators

    Returns:
        mu_agg: (B, N, K) aggregated means
        sigma_agg: (B, N, K, K) or (B, N, K) or None
    """
    B, N, K = mu_q.shape

    # Get transport operators
    if cached_transport is not None and 'Omega' in cached_transport:
        Omega = cached_transport['Omega']
    else:
        phi_matrix = torch.einsum('bna,aij->bnij', phi, generators)
        exp_phi, exp_neg_phi = stable_matrix_exp_pair(phi_matrix)
        Omega = torch.einsum('bikl,bjlm->bijkm', exp_phi, exp_neg_phi)

    # GL(K) metric correction: check if generators are skew-symmetric (SO(K))
    _is_skew = torch.allclose(
        generators + generators.transpose(-1, -2),
        torch.zeros_like(generators), atol=1e-5,
    )
    if _is_skew:
        Omega_msg = Omega
    else:
        # GL(K): Ω^{-T} = Ω_ji^T
        Omega_msg = Omega.permute(0, 2, 1, 3, 4).transpose(-1, -2)

    # Transport and aggregate means
    mu_transported = torch.einsum('bijkl,bjl->bijk', Omega_msg, mu_q)
    mu_aggregated = torch.einsum('bij,bijk->bik', beta, mu_transported)

    # Covariance aggregation
    if aggregate_mode == 'full_distribution':
        if diagonal_covariance:
            sigma_q_diag = sigma_q
            while sigma_q_diag.dim() > 3 and sigma_q_diag.shape[-1] == 1:
                sigma_q_diag = sigma_q_diag.squeeze(-1)

            sigma_full = torch.diag_embed(sigma_q_diag)
            Sigma_transported_full = torch.einsum(
                'bijkl,bjlm,bijmn->bijkn',
                Omega_msg, sigma_full, Omega_msg.transpose(-1, -2),
            )
            sigma_transported_diag = torch.diagonal(
                Sigma_transported_full, dim1=-2, dim2=-1,
            ).clone()

            second_moment = sigma_transported_diag + mu_transported ** 2
            sigma_aggregated = torch.einsum('bij,bijk->bik', beta, second_moment)
            sigma_aggregated = (sigma_aggregated - mu_aggregated ** 2).clamp(min=1e-6)
        else:
            Sigma_transported = torch.einsum(
                'bijkl,bjlm,bijmn->bijkn',
                Omega_msg, sigma_q, Omega_msg.transpose(-1, -2),
            )
            second_moment = Sigma_transported + torch.einsum(
                'bijk,bijl->bijkl', mu_transported, mu_transported,
            )
            sigma_aggregated = torch.einsum('bij,bijkl->bikl', beta, second_moment)
            sigma_aggregated = sigma_aggregated - torch.einsum(
                'bik,bil->bikl', mu_aggregated, mu_aggregated,
            )
            eps_eye = 1e-6 * torch.eye(K, device=sigma_aggregated.device, dtype=sigma_aggregated.dtype)
            sigma_aggregated = sigma_aggregated + eps_eye
    else:
        sigma_aggregated = None

    return mu_aggregated, sigma_aggregated


# =============================================================================
# Multi-Head Attention with Irrep Structure
# =============================================================================

class IrrepMultiHeadAttention(nn.Module):
    """Multi-head attention where heads correspond to irrep blocks.

    SO(3): heads = Wigner D-matrix irreps (ℓ=0,1,2,...)
    SO(N): heads = irrep blocks from global generators
    GL(K): heads = block-diagonal GL(d_head) subgroups
    """

    def __init__(self, config: GaugeTransformerConfig, generators: torch.Tensor):
        """Initialize from config and global generators.

        Args:
            config: GaugeTransformerConfig
            generators: (n_gen, K, K) global generators
        """
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        self.kappa_beta = config.kappa_ffn
        self.epsilon = 1e-8
        self.aggregate_mode = 'mean_only'
        self.diagonal_covariance = config.diagonal_covariance
        self.alibi_slope = config.alibi_slope
        self.use_identity_transport = config.use_identity_transport
        self.mask_self_attention = config.mask_self_attention
        self.enforce_orthogonal = config.enforce_orthogonal
        self.use_rope = config.use_rope
        self.rope_base = config.rope_base

        # Build irrep block structure from config
        self.irrep_dims, self.irrep_labels = self._build_irrep_structure(config)
        self.n_heads = len(self.irrep_dims)
        self.total_dim = sum(self.irrep_dims)

        # Store gauge group info
        self.gauge_group = config.gauge_group

        # Create per-head generators
        self.head_generators = nn.ModuleList()
        cum_dim = 0

        for head_idx, dim in enumerate(self.irrep_dims):
            if config.gauge_group == 'SO3':
                if dim == 1:
                    gen = torch.zeros(3, 1, 1)
                elif dim % 2 == 1 and dim >= 3:
                    if not GENERATORS_AVAILABLE:
                        raise RuntimeError("math_utils.generators required for SO3 irreps")
                    gen_np = generate_so3_generators(dim)
                    gen = torch.from_numpy(gen_np).float()
                else:
                    raise ValueError(
                        f"Head {head_idx} dim={dim} is not a valid SO(3) irrep. "
                        f"SO(3) irreps must have odd dimensions (1, 3, 5, 7, ...)."
                    )
            else:
                # SO(N) / GL(K): extract block from global generators
                gen = generators[:, cum_dim:cum_dim+dim, cum_dim:cum_dim+dim].clone()

            gen_holder = nn.Module()
            gen_holder.register_buffer('gen', gen)
            self.head_generators.append(gen_holder)
            cum_dim += dim

        # Per-head κ (learned temperature per head)
        self.per_head_kappa = config.per_head_kappa
        if config.per_head_kappa:
            self.log_kappa = nn.Parameter(
                torch.full((self.n_heads,), math.log(max(config.kappa_ffn, 1e-6)))
            )
        else:
            self.log_kappa = None

        # Optional W_O output projection
        self.use_output_projection = config.use_output_projection
        if config.use_output_projection:
            self.output_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        else:
            self.output_proj = None

    @staticmethod
    def _build_irrep_structure(
        config: GaugeTransformerConfig,
    ) -> Tuple[List[int], List[str]]:
        """Build irrep block dimensions and labels from config."""
        irrep_dims = []
        irrep_labels = []
        total_dim = 0
        K = config.embed_dim

        if config.gauge_group == 'GLK':
            if config.irrep_spec is None or (
                len(config.irrep_spec) == 1 and config.irrep_spec[0][0] == 'full'
            ):
                return [K], ['full']
            else:
                label, n_heads, d_head = config.irrep_spec[0]
                if n_heads * d_head != K:
                    raise ValueError(
                        f"GL(K) multi-head: {n_heads}×{d_head}={n_heads*d_head} != embed_dim={K}"
                    )
                # Check for cross-head coupling override
                if config.irrep_dims is not None:
                    return list(config.irrep_dims), [
                        f'glk_superblock_{i}' for i in range(len(config.irrep_dims))
                    ]
                return [d_head] * n_heads, [f'glk_head_{h}' for h in range(n_heads)]
        else:
            # SO(3) / SO(N)
            if config.irrep_spec is not None:
                for label, multiplicity, dim in config.irrep_spec:
                    for _ in range(multiplicity):
                        irrep_dims.append(dim)
                        irrep_labels.append(label)
                        total_dim += dim

            # Pad to embed_dim with scalar heads
            if total_dim < K:
                padding = K - total_dim
                for _ in range(padding):
                    irrep_dims.append(1)
                    irrep_labels.append('ℓ0_pad')
            elif total_dim > K:
                raise ValueError(f"Irrep spec sums to {total_dim}, exceeds embed_dim={K}")

            return irrep_dims, irrep_labels

    def forward(
        self,
        mu_q: torch.Tensor,
        sigma_q: torch.Tensor,
        phi: torch.Tensor,
        generators: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        cached_head_transports: Optional[List[dict]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Forward pass through multi-head attention.

        Args:
            mu_q: (B, N, K) belief means
            sigma_q: (B, N, K, K) or (B, N, K) covariances
            phi: (B, N, n_gen) gauge frames
            generators: (n_gen, K, K) global generators
            mask: (B, N, N) optional causal mask
            return_attention: Return attention weights and KL matrices
            cached_head_transports: Precomputed transports per head

        Returns:
            mu_out: (B, N, K)
            sigma_out: (B, N, K, K) or None
            attention_weights: (B, n_heads, N, N) or None
            kl_matrices: (B, n_heads, N, N) or None
        """
        B, N, K = mu_q.shape

        mu_blocks = self._split_irreps(mu_q)
        sigma_blocks = self._split_irreps_sigma(sigma_q)

        head_outputs_mu = []
        head_outputs_sigma = []
        all_attention_weights = []
        all_kl_matrices = []

        for head_idx, (mu_head, sigma_head, dim_head) in enumerate(
            zip(mu_blocks, sigma_blocks, self.irrep_dims)
        ):
            gen_head = self.head_generators[head_idx].gen.to(
                device=generators.device, dtype=generators.dtype,
            )

            # Transport operators: cached or computed
            if cached_head_transports is not None:
                head_cached = cached_head_transports[head_idx]
            else:
                head_cached = compute_transport_operators(
                    phi, gen_head, enforce_orthogonal=self.enforce_orthogonal,
                )

            # Per-head temperature
            if self.per_head_kappa:
                kappa_h = torch.exp(self.log_kappa[head_idx]).item()
            else:
                kappa_h = self.kappa_beta

            # Compute attention for this head
            attn_result = compute_attention_weights(
                mu_head, sigma_head, phi, gen_head, kappa_h,
                self.epsilon, mask,
                return_kl=return_attention,
                diagonal_covariance=self.diagonal_covariance,
                cached_transport=head_cached,
                alibi_slope=self.alibi_slope,
                use_identity_transport=self.use_identity_transport,
                mask_self_attention=self.mask_self_attention,
                enforce_orthogonal=self.enforce_orthogonal,
                use_rope=self.use_rope,
                rope_base=self.rope_base,
            )

            if return_attention:
                beta_head, kl_head = attn_result
                all_attention_weights.append(beta_head)
                all_kl_matrices.append(kl_head)
            else:
                beta_head = attn_result

            # Aggregate messages (reuse cached transport)
            mu_agg, sigma_agg = aggregate_messages(
                mu_head, sigma_head, phi, beta_head, gen_head,
                aggregate_mode=self.aggregate_mode,
                diagonal_covariance=self.diagonal_covariance,
                cached_transport=head_cached,
            )

            head_outputs_mu.append(mu_agg)
            if sigma_agg is not None:
                head_outputs_sigma.append(sigma_agg)

        # Concatenate head outputs
        mu_concat = torch.cat(head_outputs_mu, dim=-1)

        if head_outputs_sigma:
            sigma_concat = self._block_diag_sigma(head_outputs_sigma)
        else:
            sigma_concat = None

        # Optional W_O output projection
        if self.output_proj is not None:
            mu_out = self.output_proj(mu_concat)
        else:
            mu_out = mu_concat

        # Stack per-head attention/KL
        if return_attention:
            attention_weights = torch.stack(all_attention_weights, dim=1)
            kl_matrices = torch.stack(all_kl_matrices, dim=1)
        else:
            attention_weights = None
            kl_matrices = None

        return mu_out, sigma_concat, attention_weights, kl_matrices

    # ── Irrep splitting/joining helpers ───────────────────────────────

    def _split_irreps(self, mu: torch.Tensor) -> List[torch.Tensor]:
        """Split embedding into irrep blocks (contiguous copies)."""
        blocks = []
        start = 0
        for dim in self.irrep_dims:
            blocks.append(mu[..., start:start+dim].contiguous())
            start += dim
        return blocks

    def _split_irreps_sigma(self, sigma: torch.Tensor) -> List[torch.Tensor]:
        """Split covariance into irrep blocks, handling format mismatches."""
        # Squeeze trailing singletons
        while sigma.dim() > 3 and sigma.shape[-1] == 1:
            sigma = sigma.squeeze(-1)

        sigma_is_diag = sigma.dim() == 3

        # Handle format mismatches
        if self.diagonal_covariance and not sigma_is_diag:
            sigma = torch.diagonal(sigma, dim1=-2, dim2=-1).clone()
        elif not self.diagonal_covariance and sigma_is_diag:
            sigma = torch.diag_embed(sigma)

        blocks = []
        start = 0
        for dim in self.irrep_dims:
            if self.diagonal_covariance:
                blocks.append(sigma[..., start:start+dim].contiguous())
            else:
                blocks.append(sigma[..., start:start+dim, start:start+dim].contiguous())
            start += dim
        return blocks

    def _block_diag_sigma(self, sigma_blocks: List[torch.Tensor]) -> torch.Tensor:
        """Construct covariance from irrep blocks."""
        B, N = sigma_blocks[0].shape[:2]
        K = sum(self.irrep_dims)

        if self.diagonal_covariance:
            return torch.cat(sigma_blocks, dim=-1)

        sigma_full = torch.zeros(
            B, N, K, K,
            device=sigma_blocks[0].device, dtype=sigma_blocks[0].dtype,
        )
        start = 0
        for sigma_block, dim in zip(sigma_blocks, self.irrep_dims):
            sigma_full[..., start:start+dim, start:start+dim] = sigma_block
            start += dim
        return sigma_full

    def precompute_head_transports(
        self, phi: torch.Tensor, device: torch.device, dtype: torch.dtype,
    ) -> List[dict]:
        """Precompute transport operators for all heads (cross-layer cache)."""
        cached = []
        for head_idx in range(self.n_heads):
            gen_head = self.head_generators[head_idx].gen.to(device=device, dtype=dtype)
            cached.append(compute_transport_operators(
                phi, gen_head, enforce_orthogonal=self.enforce_orthogonal,
            ))
        return cached

    def extra_repr(self) -> str:
        dims_str = str(self.irrep_dims[:3]) + '...' if len(self.irrep_dims) > 3 else str(self.irrep_dims)
        return f"embed_dim={self.embed_dim}, n_heads={self.n_heads}, irrep_dims={dims_str}, kappa={self.kappa_beta}"
