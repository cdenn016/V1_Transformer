"""
Gauge-Theoretic Transformer Block (0D Architecture)
====================================================

Complete transformer block with:
1. Gauge-theoretic multi-head attention (KL-based, no W_Q/W_K!)
2. Feedforward network (prior evolution)
3. Layer normalization
4. Residual connections

Standard Architecture, Gauge Mechanism:
    x → LayerNorm → Attention → Residual → LayerNorm → FFN → Residual

But with gauge-theoretic attention:
    (μ, Σ, φ) → Attention(via KL + transport) → (μ', Σ', φ')
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple, List

from transformer.core.block_config import BlockConfig

# Import our gauge attention
from transformer.core.attention import IrrepMultiHeadAttention

# Import VFE FFN directly (no wrapper)
from transformer.core.variational_ffn import VariationalFFNDynamic

# Trajectory tracking (optional)
try:
    from transformer.analysis.trajectory import get_global_recorder
    TRAJECTORY_TRACKING_AVAILABLE = True
except ImportError:
    TRAJECTORY_TRACKING_AVAILABLE = False
    def get_global_recorder():
        return None


def _infer_gauge_group(generators):
    """Infer gauge group and dimension from generators shape."""
    if generators is None:
        return 'SO3', 3

    n_gen = generators.shape[0]
    K = generators.shape[1]

    if n_gen == 3:
        return 'SO3', 3
    elif n_gen == K * K:
        return 'GLK', K
    else:
        # Check if n_gen matches SO(N): n_gen = N*(N-1)/2
        disc = 1 + 8 * n_gen
        sqrt_disc = int(math.sqrt(disc))
        if sqrt_disc * sqrt_disc == disc:
            N_candidate = (1 + sqrt_disc) // 2
            if N_candidate * (N_candidate - 1) // 2 == n_gen:
                return 'SON', N_candidate
        return 'GLK', K


class GaugeTransformerBlock(nn.Module):
    """
    Single transformer block with gauge-theoretic attention.

    Architecture:
        1. Self-attention sublayer:
           - LayerNorm on means
           - IrrepMultiHeadAttention (KL-based)
           - Residual connection

        2. Feedforward sublayer:
           - LayerNorm on means
           - VFE-based belief evolution (variational free energy minimization)
           - Residual connection

    Note: We primarily evolve means (μ), while covariances (Σ) and
          gauge frames (φ) can be evolved or kept fixed depending on mode.
          Phi evolution uses ∂F/∂φ gradient descent, NOT neural networks.

    0D Structure:
        - All agents at single point c*
        - Attention computed via KL divergence
        - No spatial convolutions or position-dependent operations
    """

    def __init__(self, cfg: BlockConfig):
        super().__init__()
        self.embed_dim = cfg.embed_dim
        self.hidden_dim = cfg.hidden_dim
        self.evolve_sigma = cfg.evolve_sigma
        self.evolve_phi = cfg.evolve_phi
        self.ffn_mode = cfg.ffn_mode
        self.generators = cfg.generators
        self.diagonal_covariance = cfg.diagonal_covariance

        # Pure VFE mode flags
        self.use_layernorm = cfg.use_layernorm
        self.use_residual = cfg.use_residual

        # =====================================================================
        # Attention Sublayer
        # =====================================================================
        gauge_group, gauge_dim_inferred = _infer_gauge_group(cfg.generators)

        self.attention = IrrepMultiHeadAttention(
            embed_dim=cfg.embed_dim,
            irrep_spec=cfg.irrep_spec,
            kappa_beta=cfg.kappa_beta,
            epsilon=1e-8,
            aggregate_mode='full_distribution' if cfg.evolve_sigma else 'mean_only',
            diagonal_covariance=cfg.diagonal_covariance,
            attention_pattern=cfg.attention_pattern,
            attention_window=cfg.attention_window,
            gauge_group=gauge_group,
            gauge_dim=gauge_dim_inferred,
            global_generators=cfg.generators,
            alibi_slope=cfg.alibi_slope,
            gauge_mode=cfg.gauge_mode,
            mask_self_attention=cfg.mask_self_attention,
            enforce_orthogonal=cfg.enforce_orthogonal,
            use_output_projection=cfg.use_output_projection,
            irrep_dims_override=cfg.ffn_irrep_dims if (gauge_group == 'GLK' and cfg.cross_head_perm is not None) else None,
            use_rope=cfg.use_rope,
            rope_base=cfg.rope_base,
        )

        # Conditionally create LayerNorm (disabled for pure VFE)
        self.norm1 = nn.LayerNorm(cfg.embed_dim) if cfg.use_layernorm else nn.Identity()

        # =====================================================================
        # VFE_dynamic FFN Sublayer (VariationalFFNDynamic directly, no wrapper)
        # =====================================================================
        if cfg.generators is None:
            raise ValueError("generators required for VFE_dynamic mode")

        self.ffn = VariationalFFNDynamic(
            embed_dim=cfg.embed_dim,
            generators=cfg.generators,
            alpha=cfg.ffn_alpha,
            lambda_belief=cfg.ffn_lambda_belief,
            kappa=cfg.ffn_kappa,
            n_iterations=cfg.ffn_n_iterations,
            learnable_lr=cfg.ffn_learnable_lr,
            update_sigma=cfg.ffn_update_sigma,
            diagonal_covariance=cfg.diagonal_covariance,
            update_phi=cfg.evolve_phi,
            update_phi_per_iteration=cfg.evolve_phi_e_step,
            phi_lr=cfg.phi_lr,
            phi_max_norm=cfg.phi_max_norm,
            irrep_dims=cfg.ffn_irrep_dims,
            chunk_size=cfg.ffn_chunk_size,
            mask_self_attention=cfg.mask_self_attention,
            learnable_alpha=cfg.ffn_learnable_alpha,
            multihead_vfe=cfg.multihead_vfe,
            phi_natural_gradient=cfg.phi_natural_gradient,
            use_deq=cfg.use_deq,
            deq_neumann_terms=cfg.deq_neumann_terms,
            gauge_mode=cfg.gauge_mode,
        )

        self.norm2 = nn.LayerNorm(cfg.embed_dim) if cfg.use_layernorm else nn.Identity()

        # =====================================================================
        # Gauge Frame Evolution Configuration
        # =====================================================================
        self.phi_dim = cfg.phi_dim
        self.phi_max_norm = cfg.phi_max_norm

    def forward(
        self,
        mu_q: torch.Tensor,
        sigma_q: torch.Tensor,
        phi: torch.Tensor,
        generators: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        mu_prior: Optional[torch.Tensor] = None,
        token_ids: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        W_out: Optional[torch.Tensor] = None,
        cached_head_transports: Optional[list] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through transformer block.

        Args:
            mu_q: Belief means (B, N, K)
            sigma_q: Belief covariances (B, N, K, K)
            phi: Gauge frames (B, N, 3)
            generators: SO(3) generators (3, K, K)
            mask: Optional causal mask (B, N, N)
            mu_prior: Embedding priors (B, N, K) - required for variational FFN
            targets: Target token IDs (B, N) - for E-step discrete observations
            W_out: Output projection (V, K) - for computing CE gradient in E-step
            cached_head_transports: Precomputed transport dicts per head.

        Returns:
            mu_q_out: Updated means (B, N, K)
            sigma_q_out: Updated covariances (B, N, K, K)
            phi_out: Updated gauge frames (B, N, 3)
        """
        # =====================================================================
        # 1. Attention Sublayer with Pre-Norm + Residual
        # =====================================================================

        # Pre-layer normalization on means
        mu_normalized = self.norm1(mu_q)

        # Multi-head attention (gauge-theoretic!)
        recorder = get_global_recorder() if TRAJECTORY_TRACKING_AVAILABLE else None
        recording_attention = recorder is not None and recorder.enabled and recorder.record_attention
        need_beta = self.ffn_mode == 'VFE_dynamic'
        need_attention_output = need_beta or recording_attention

        mu_attn, sigma_attn, beta, kl_matrix = self.attention(
            mu_normalized,
            sigma_q,
            phi,
            generators,
            mask=mask,
            return_attention=need_attention_output,
            cached_head_transports=cached_head_transports,
        )

        # Record attention for trajectory tracking
        if recording_attention and beta is not None:
            recorder.record_attention(beta, kl_matrix)

        # Residual connection (optional for pure VFE)
        if self.use_residual:
            mu_q = mu_q + mu_attn
        else:
            mu_q = mu_attn

        # Update covariances if evolving
        if self.evolve_sigma and sigma_attn is not None:
            sigma_q = sigma_attn

        # =====================================================================
        # 2. Feedforward Sublayer (with optional Pre-Norm + Residual)
        # =====================================================================

        mu_normalized = self.norm2(mu_q)

        if mu_prior is None:
            raise ValueError("VFE_dynamic mode requires mu_prior argument")

        mu_ffn, sigma_ffn, phi_out, _beta_history = self.ffn(
            mu=mu_normalized,
            beta=beta,
            mu_prior=mu_prior,
            phi=phi,
            sigma=sigma_q,
            mask=mask,
            token_ids=token_ids,
            targets=targets,
            W_out=W_out,
        )

        # Update covariances from FFN if evolving
        if self.evolve_sigma and sigma_ffn is not None:
            sigma_q = sigma_ffn

        # Residual connection (optional for pure VFE)
        if self.use_residual:
            mu_q = mu_q + mu_ffn
        else:
            mu_q = mu_ffn

        return mu_q, sigma_q, phi_out

    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, "
            f"hidden_dim={self.hidden_dim}, "
            f"evolve_sigma={self.evolve_sigma}, "
            f"evolve_phi={self.evolve_phi}"
        )


# =============================================================================
# Stack of Transformer Blocks
# =============================================================================

class GaugeTransformerStack(nn.Module):
    """
    Stack of N gauge transformer blocks.

    This is the main "encoder" of the model, transforming initial
    embeddings through multiple layers of gauge-theoretic attention.
    """

    def __init__(self, cfg: BlockConfig):
        super().__init__()
        self.n_layers = cfg.n_layers

        self.blocks = nn.ModuleList([
            GaugeTransformerBlock(cfg)
            for _ in range(cfg.n_layers)
        ])

        # Final layer norm (optional for pure VFE)
        self.final_norm = nn.LayerNorm(cfg.embed_dim) if cfg.use_layernorm else nn.Identity()

    def forward(
        self,
        mu_q: torch.Tensor,
        sigma_q: torch.Tensor,
        phi: torch.Tensor,
        generators: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        mu_prior: Optional[torch.Tensor] = None,
        token_ids: Optional[torch.Tensor] = None,
        return_intermediates: bool = False,
        cached_head_transports: Optional[list] = None,
        targets: Optional[torch.Tensor] = None,
        W_out: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[List]]:
        """
        Forward through all transformer blocks.

        Args:
            mu_q: Initial means (B, N, K)
            sigma_q: Initial covariances (B, N, K, K)
            phi: Initial gauge frames (B, N, 3)
            generators: Lie algebra generators (n_gen, K, K)
            mask: Optional causal mask
            mu_prior: Embedding priors (B, N, K) - for variational FFN
            return_intermediates: If True, return states after each layer
            cached_head_transports: Precomputed transport dicts per head.
                                   When evolve_phi=False, reuse across all layers (6× speedup).
            targets: Target token IDs (B, N) - for E-step observations (final layer only)
            W_out: Output projection weights (V, K) - for CE gradient in E-step

        Returns:
            mu_q: Final means (B, N, K)
            sigma_q: Final covariances (B, N, K, K)
            phi: Final gauge frames (B, N, 3)
            intermediates: Optional list of intermediate states
        """
        intermediates = [] if return_intermediates else None

        # Get trajectory recorder
        recorder = get_global_recorder() if TRAJECTORY_TRACKING_AVAILABLE else None
        recording_enabled = recorder is not None and recorder.enabled

        n_blocks = len(self.blocks)
        for layer_idx, block in enumerate(self.blocks):
            # Trajectory recording: start layer
            if recording_enabled:
                recorder.start_layer(layer_idx)
                recorder.record_layer_input(mu_q, sigma_q, phi)

            # Only pass targets/W_out to the final layer (observation grounding)
            is_final = (layer_idx == n_blocks - 1)
            mu_q, sigma_q, phi = block(
                mu_q, sigma_q, phi, generators, mask, mu_prior,
                token_ids=token_ids,
                cached_head_transports=cached_head_transports,
                targets=targets if is_final else None,
                W_out=W_out if is_final else None,
            )

            # Trajectory recording: record output
            if recording_enabled:
                recorder.record_layer_output(mu_q, sigma_q, phi)
                recorder.end_layer()

            if return_intermediates:
                intermediates.append({
                    'layer': layer_idx,
                    'mu': mu_q.detach(),
                    'sigma': sigma_q.detach() if sigma_q is not None else None,
                    'phi': phi.detach(),
                })

        # Final normalization
        mu_q = self.final_norm(mu_q)

        return mu_q, sigma_q, phi, intermediates
