# -*- coding: utf-8 -*-
"""
Gauge-Transformer Blocks (Refactored)
======================================

GaugeTransformerBlock: Single transformer layer (attention + VFE FFN).
GaugeTransformerStack: Stack of N blocks.

All configuration via GaugeTransformerConfig — no parameter forwarding.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple

from .config import GaugeTransformerConfig
from .attention import IrrepMultiHeadAttention
from .variational_ffn import VariationalFFNDynamic


class GaugeTransformerBlock(nn.Module):
    """Single gauge-theoretic transformer block.

    Architecture:
        1. Pre-norm → Multi-head KL attention → Residual (optional)
        2. Pre-norm → VFE belief evolution → Residual (optional)

    All parameters come from config.
    """

    def __init__(
        self,
        config: GaugeTransformerConfig,
        generators: torch.Tensor,
        prior_bank: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        self.evolve_sigma = config.evolve_sigma
        self.evolve_phi = config.evolve_phi
        self.use_residual = config.use_residual

        # ── Attention sublayer ──────────────────────────────────────────
        self.attention = IrrepMultiHeadAttention(config, generators)

        self.norm1 = nn.LayerNorm(config.embed_dim) if config.use_layernorm else nn.Identity()
        self.dropout1 = nn.Dropout(config.dropout) if config.use_dropout else nn.Identity()

        # ── VFE FFN sublayer (direct — no GaugeFFN wrapper) ─────────────
        self.ffn = VariationalFFNDynamic(config, generators, prior_bank=prior_bank)

        self.norm2 = nn.LayerNorm(config.embed_dim) if config.use_layernorm else nn.Identity()

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
        """Forward pass through transformer block.

        Args:
            mu_q: Belief means (B, N, K)
            sigma_q: Belief covariances (B, N, K, K) or (B, N, K)
            phi: Gauge frames (B, N, n_gen)
            generators: Lie algebra generators (n_gen, K, K)
            mask: Optional causal mask (B, N, N)
            mu_prior: Embedding priors (B, N, K) — required for VFE
            token_ids: Token IDs for PriorBank lookup
            targets: Target token IDs (B, N) for E-step observations
            W_out: Output projection (V, K) for CE gradient in E-step
            cached_head_transports: Precomputed transport operators per head

        Returns:
            (mu_q, sigma_q, phi) — updated beliefs
        """
        # ── 1. Attention sublayer ───────────────────────────────────────
        mu_normalized = self.norm1(mu_q)

        mu_attn, sigma_attn, beta, _kl_matrix = self.attention(
            mu_normalized, sigma_q, phi, generators,
            mask=mask,
            return_attention=True,  # Need β for VFE FFN
            cached_head_transports=cached_head_transports,
        )

        mu_attn = self.dropout1(mu_attn)

        if self.use_residual:
            mu_q = mu_q + mu_attn
        else:
            mu_q = mu_attn

        if self.evolve_sigma and sigma_attn is not None:
            sigma_q = sigma_attn

        # ── 2. VFE FFN sublayer ─────────────────────────────────────────
        mu_normalized = self.norm2(mu_q)

        if mu_prior is None:
            raise ValueError("VFE mode requires mu_prior argument")

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

        if self.evolve_sigma and sigma_ffn is not None:
            sigma_q = sigma_ffn

        if self.use_residual:
            mu_q = mu_q + mu_ffn
        else:
            mu_q = mu_ffn

        return mu_q, sigma_q, phi_out

    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, "
            f"evolve_sigma={self.evolve_sigma}, "
            f"evolve_phi={self.evolve_phi}"
        )


class GaugeTransformerStack(nn.Module):
    """Stack of N gauge transformer blocks.

    Config-driven: no parameter forwarding. Each block gets the same config.
    """

    def __init__(
        self,
        config: GaugeTransformerConfig,
        generators: torch.Tensor,
        prior_bank: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.n_layers = config.n_layers

        self.blocks = nn.ModuleList([
            GaugeTransformerBlock(config, generators, prior_bank=prior_bank)
            for _ in range(config.n_layers)
        ])

        self.final_norm = (
            nn.LayerNorm(config.embed_dim) if config.use_layernorm
            else nn.Identity()
        )

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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[List]]:
        """Forward through all transformer blocks.

        Args:
            mu_q: Initial means (B, N, K)
            sigma_q: Initial covariances (B, N, K, K) or (B, N, K)
            phi: Initial gauge frames (B, N, n_gen)
            generators: Lie algebra generators (n_gen, K, K)
            mask: Optional causal mask
            mu_prior: Embedding priors (B, N, K)
            token_ids: Token IDs for PriorBank lookup
            return_intermediates: If True, return per-layer states
            cached_head_transports: Precomputed transport operators per head

        Returns:
            (mu_q, sigma_q, phi, intermediates)
        """
        intermediates = [] if return_intermediates else None

        for layer_idx, block in enumerate(self.blocks):
            mu_q, sigma_q, phi = block(
                mu_q, sigma_q, phi, generators, mask, mu_prior,
                token_ids=token_ids,
                cached_head_transports=cached_head_transports,
            )

            if return_intermediates:
                intermediates.append({
                    'layer': layer_idx,
                    'mu': mu_q.detach(),
                    'sigma': sigma_q.detach() if sigma_q is not None else None,
                    'phi': phi.detach(),
                })

        mu_q = self.final_norm(mu_q)

        return mu_q, sigma_q, phi, intermediates
