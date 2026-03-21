"""
Gauge-Theoretic Transformer Block (0D Architecture)
====================================================

Complete transformer block with:
1. Gauge-theoretic multi-head attention (IrrepMultiHeadAttention, KL-based)
2. Variational free energy FFN (VariationalFFNDynamic, E-step belief evolution)
3. Optional LayerNorm and residual connections (toggled for pure VFE ablation)
4. Optional non-flat gauge transport via edge-local GaugeConnection

Data flow:
    (μ, Σ, φ) → LayerNorm(μ) → Attention(KL + transport) → Residual
              → LayerNorm(μ) → VFE FFN(E-step iterations)  → Residual → (μ', Σ', φ')

All configuration flows through BlockConfig — no raw kwargs.
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

# Import gauge connection for non-flat transport
from transformer.core.connection import GaugeConnection

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
           - LayerNorm on means (optional, cfg.use_layernorm)
           - IrrepMultiHeadAttention: KL-based attention with gauge transport
           - Residual connection (optional, cfg.use_residual)

        2. Feedforward sublayer:
           - LayerNorm on means
           - VariationalFFNDynamic: VFE E-step belief evolution with dynamic β
           - Residual connection

    Belief updates:
        - μ: always updated (natural gradient descent on VFE)
        - Σ: updated if cfg.evolve_sigma=True (SPD retraction)
        - φ: updated if cfg.evolve_phi=True (∂F/∂φ descent, not neural)

    Optional features wired through BlockConfig:
        - Non-flat gauge transport: GaugeConnection produces edge-local δ_ij,
          modifying Ω_ij = exp(φ_i)·exp(α·δ_ij)·exp(-φ_j)
        - PriorBank: token-dependent priors for VFE dynamics
        - RoPE: rotary position embeddings on μ before KL scoring
        - exact_diagonal_transport: lift diagonal σ for exact Ω@Σ@Ω^T
    """

    def __init__(self, cfg: BlockConfig):
        super().__init__()
        self.embed_dim = cfg.embed_dim
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
            exact_diagonal_transport=cfg.exact_diagonal_transport,
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
            exact_diagonal_transport=cfg.exact_diagonal_transport,
            update_phi=cfg.evolve_phi,
            update_phi_per_iteration=cfg.evolve_phi_e_step,
            phi_update_interval=cfg.phi_update_interval,
            phi_lr=cfg.phi_lr,
            phi_max_norm=cfg.phi_max_norm,
            prior_bank=cfg.ffn_prior_bank,
            use_prior_bank=cfg.ffn_use_prior_bank,
            irrep_dims=cfg.ffn_irrep_dims,
            chunk_size=cfg.ffn_chunk_size,
            mask_self_attention=cfg.mask_self_attention,
            learnable_alpha=cfg.ffn_learnable_alpha,
            multihead_vfe=cfg.multihead_vfe,
            phi_natural_gradient=cfg.phi_natural_gradient,
            use_deq=cfg.use_deq,
            deq_neumann_terms=cfg.deq_neumann_terms,
            gauge_mode=cfg.gauge_mode,
            # Pass constant_omega from the attention module so the FFN's VFE
            # iterations use the same per-head Ω transport (manuscript Limit 2).
            # Without this, the FFN would use Ω=I, computing inconsistent
            # attention patterns relative to the attention sublayer.
            constant_omega=self.attention.constant_omega,
            amortized_inference=cfg.amortized_inference,
            isotropic_covariance=cfg.isotropic_covariance,
            analytic_phi_grad=cfg.analytic_phi_grad,
            analytic_phi_grad_dexp_order=cfg.analytic_phi_grad_dexp_order,
            obs_sigma_gradient=cfg.obs_sigma_gradient,
            obs_sigma_weight=cfg.obs_sigma_weight,
            use_rope=cfg.use_rope,
            rope_base=cfg.rope_base,
            gauge_param=cfg.gauge_param,
            detach_phi=cfg.detach_phi,
        )

        self.norm2 = nn.LayerNorm(cfg.embed_dim) if cfg.use_layernorm else nn.Identity()

        # =====================================================================
        # Non-Flat Gauge Transport (optional)
        # =====================================================================
        self.non_flat_transport = cfg.non_flat_transport
        self.cocycle_relaxation = cfg.cocycle_relaxation
        self.holonomy_penalty = cfg.holonomy_penalty
        if cfg.non_flat_transport:
            n_gen = cfg.generators.shape[0] if cfg.generators is not None else 3
            self.gauge_connection = GaugeConnection(
                d_head=cfg.embed_dim,
                n_gen=n_gen,
                connection_type=cfg.connection_type,
                hidden_dim=cfg.connection_hidden_dim,
                init_scale=cfg.connection_init_scale,
            )
            if cfg.per_head_flatness_gate:
                n_heads = len(cfg.irrep_spec)
                self.flatness_gate_logit = nn.Parameter(torch.zeros(n_heads))
        else:
            self.gauge_connection = None

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
        omega: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through transformer block.

        Flow: (μ, Σ, φ) → Attention sublayer → FFN sublayer → (μ', Σ', φ')

        Args:
            mu_q: Belief means (B, N, K).
            sigma_q: Belief covariances — (B, N, K, K) full or (B, N, K) diagonal
                     when cfg.diagonal_covariance=True.
            phi: Gauge frames (B, N, phi_dim) — phi_dim=3 for SO(3),
                 N(N-1)/2 for SO(N), K² for GL(K).
            generators: Lie algebra generators (n_gen, K, K) — n_gen matches phi_dim.
            mask: Optional causal mask (B, N, N) or (B, 1, N, N).
            mu_prior: Embedding priors (B, N, K) — required, used as VFE prior means.
            token_ids: Token IDs (B, N) — passed to PriorBank for token-dependent priors.
            targets: Target token IDs (B, N) — for E-step discrete observation grounding.
            W_out: Output projection weights (V, K) — for CE gradient in E-step.
            cached_head_transports: Precomputed transport dicts per head — list of
                {'Omega': (B, N, N, d_h, d_h)} per head. When evolve_phi=False,
                these can be reused across layers for ~6× speedup.

        Returns:
            mu_q_out: Updated means (B, N, K).
            sigma_q_out: Updated covariances — same shape as input sigma_q.
                         Unchanged when evolve_sigma=False.
            phi_out: Updated gauge frames (B, N, phi_dim).
                     Unchanged when evolve_phi=False.
        """
        # =====================================================================
        # 1. Attention Sublayer with Pre-Norm + Residual
        # =====================================================================

        # Pre-layer normalization on means
        mu_normalized = self.norm1(mu_q)

        # Non-flat transport: compute edge-local connection δ_ij and inject
        # into cached transport so attention sees the modified Ω_ij.
        if self.non_flat_transport and self.gauge_connection is not None and cached_head_transports is None:
            from transformer.core.attention import compute_transport_operators
            delta_ij = self.gauge_connection(mu_normalized, mu_normalized)  # (B, N, N, n_gen)
            transport = compute_transport_operators(
                phi, generators,
                gauge_mode='learned',
                connection_delta=delta_ij,
                cocycle_relaxation=self.cocycle_relaxation,
            )
            # Split full Omega into per-head cached transports
            Omega_full = transport['Omega']  # (B, N, N, K, K)
            irrep_dims = self.attention.irrep_dims
            cached_head_transports = []
            dim_start = 0
            for d in irrep_dims:
                cached_head_transports.append({
                    'Omega': Omega_full[:, :, :, dim_start:dim_start+d, dim_start:dim_start+d],
                })
                dim_start += d

        # Multi-head attention (gauge-theoretic!)
        # For direct omega mode: build per-head cached transports from omega blocks
        # so the attention sublayer uses Omega_h / Omega_h_inv instead of matrix_exp.
        if omega is not None and getattr(self.ffn, 'gauge_param', 'phi') == 'omega' and cached_head_transports is None:
            # Build per-head (omega_h, omega_h_inv) pairs using per-block inv
            # (avoids full K×K inv when omega is block-diagonal)
            irrep_dims = self.attention.irrep_dims
            cached_head_transports = []
            block_start = 0
            for d_h in irrep_dims:
                omega_h = omega[:, :, block_start:block_start+d_h, block_start:block_start+d_h]
                omega_h_inv = torch.linalg.inv(omega_h)  # (B, N, d_h, d_h)
                cached_head_transports.append({
                    'exp_phi': omega_h,
                    'exp_neg_phi': omega_h_inv,
                })
                block_start += d_h

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
            omega=omega,
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
        parts = [
            f"embed_dim={self.embed_dim}",
            f"evolve_sigma={self.evolve_sigma}",
            f"evolve_phi={self.evolve_phi}",
            f"diagonal_covariance={self.diagonal_covariance}",
            f"ffn_mode={self.ffn_mode!r}",
            f"use_layernorm={self.use_layernorm}",
            f"use_residual={self.use_residual}",
            f"non_flat_transport={self.non_flat_transport}",
        ]
        return ", ".join(parts)


# =============================================================================
# Stack of Transformer Blocks
# =============================================================================

class GaugeTransformerStack(nn.Module):
    """
    Stack of N identical GaugeTransformerBlock layers.

    Each layer applies: Attention(KL + gauge transport) → VFE FFN(E-step iterations).
    Beliefs (μ, Σ, φ) flow through all layers; targets/W_out are passed to the
    final layer only (observation grounding). A final LayerNorm is applied to μ.

    Supports gradient checkpointing (cfg.gradient_checkpointing) for ~60% memory
    savings at ~30% extra compute. The final layer is never checkpointed to
    preserve targets/W_out gradient flow.
    """

    def __init__(self, cfg: BlockConfig):
        super().__init__()
        self.n_layers = cfg.n_layers
        self.gradient_checkpointing = getattr(cfg, 'gradient_checkpointing', False)

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
        omega: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[List]]:
        """
        Forward through all transformer blocks sequentially.

        Args:
            mu_q: Initial means (B, N, K).
            sigma_q: Initial covariances — (B, N, K, K) or (B, N, K) diagonal.
            phi: Initial gauge frames (B, N, phi_dim).
            generators: Lie algebra generators (n_gen, K, K).
            mask: Optional causal mask (B, N, N) or (B, 1, N, N).
            mu_prior: Embedding priors (B, N, K) — for VFE prior means.
            token_ids: Token IDs (B, N) — for PriorBank token-dependent priors.
            return_intermediates: If True, return list of per-layer state dicts.
            cached_head_transports: Precomputed per-head transport dicts.
                When evolve_phi=False, reuse across all layers (~6× speedup).
            targets: Target token IDs (B, N) — passed to final layer only.
            W_out: Output projection (V, K) — passed to final layer only.

        Returns:
            mu_q: Final means (B, N, K) after final LayerNorm.
            sigma_q: Final covariances — same shape as input.
            phi: Final gauge frames (B, N, phi_dim).
            intermediates: List of {'layer', 'mu', 'sigma', 'phi'} dicts if
                return_intermediates=True, else None.
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

            if self.gradient_checkpointing and self.training and not is_final:
                # Gradient checkpointing: trade ~30% compute for ~60% memory savings
                # Skip final layer to preserve targets/W_out gradient flow
                def create_block_fn(blk):
                    def block_fn(mu, sigma, phi_arg):
                        return blk(
                            mu, sigma, phi_arg, generators, mask, mu_prior,
                            token_ids=token_ids,
                            cached_head_transports=cached_head_transports,
                            omega=omega,
                        )
                    return block_fn

                mu_q, sigma_q, phi = torch.utils.checkpoint.checkpoint(
                    create_block_fn(block),
                    mu_q, sigma_q, phi,
                    use_reentrant=False,
                )
            else:
                mu_q, sigma_q, phi = block(
                    mu_q, sigma_q, phi, generators, mask, mu_prior,
                    token_ids=token_ids,
                    cached_head_transports=cached_head_transports,
                    targets=targets if is_final else None,
                    W_out=W_out if is_final else None,
                    omega=omega,
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
