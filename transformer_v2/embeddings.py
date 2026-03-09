# -*- coding: utf-8 -*-
"""
Gauge-Theoretic Token Embeddings
=================================

Maps discrete tokens → agent beliefs (μ_i, Σ_i, φ_i) at single base manifold point c*.

Each token i → (μ_i, Σ_i, φ_i) where:
- μ_i ∈ ℝ^K: mean belief vector
- Σ_i ∈ SPD(K): covariance
- φ_i ∈ g: gauge frame (Lie algebra element)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from transformer_v2.config import GaugeTransformerConfig
from transformer_v2.gauge_utils import so3_log, so3_compose_bch, SON_BCH_AVAILABLE

try:
    from math_utils.generators import soN_compose_bch_torch
except ImportError:
    pass


class GaugeTokenEmbedding(nn.Module):
    """Map discrete tokens to gauge-equivariant agent beliefs.

    token_id → (μ, Σ, φ)

    Supports gauge-fixed priors where all token priors are rotations
    of a single base prior: p_i = R(φ_i) ▷ p_0.
    """

    def __init__(self, config: GaugeTransformerConfig, generators: torch.Tensor):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.embed_dim = config.embed_dim
        self.learnable_sigma = config.evolve_sigma
        self.learnable_phi = True  # Always learn phi for gauge structure
        self.gauge_fixed_priors = config.gauge_fixed_priors
        self.diagonal_covariance = config.diagonal_covariance
        self.phi_dim = config.phi_dim
        self.phi_scale = config.phi_scale
        self.mu_normalize = config.mu_normalize
        self.mu_max_norm = config.mu_max_norm
        self.use_positional_embedding = config.use_positional_embedding
        self.max_seq_len = config.max_seq_len

        init_std = config.mu_init_std if config.mu_init_std is not None else 2.0
        self.init_std = init_std
        init_sigma_scale = 1.0

        # Incompatibility check
        if self.gauge_fixed_priors and self.diagonal_covariance:
            import warnings
            warnings.warn(
                "gauge_fixed_priors=True is incompatible with diagonal_covariance=True. "
                "Forcing diagonal_covariance=False.",
                UserWarning
            )
            self.diagonal_covariance = False

        self.register_buffer('generators', generators)

        # ── Mean embeddings ──────────────────────────────────────────
        if self.gauge_fixed_priors:
            self.base_mu = nn.Parameter(torch.randn(config.embed_dim) * init_std)
        else:
            self.mu_embed = nn.Embedding(config.vocab_size, config.embed_dim)
            nn.init.normal_(self.mu_embed.weight, mean=0.0, std=init_std)

        # ── Covariance embeddings ────────────────────────────────────
        if self.gauge_fixed_priors:
            self.base_log_sigma_diag = nn.Parameter(
                torch.full((config.embed_dim,), math.log(init_sigma_scale))
            )
        elif self.learnable_sigma:
            self.log_sigma_diag = nn.Parameter(
                torch.full((config.vocab_size, config.embed_dim), math.log(init_sigma_scale))
            )
        else:
            self.register_buffer(
                'log_sigma_diag',
                torch.full((config.embed_dim,), math.log(init_sigma_scale))
            )

        # ── Gauge frame embeddings ───────────────────────────────────
        self.phi_embed = nn.Embedding(config.vocab_size, config.phi_dim)
        phi_init_std = config.phi_scale / (config.phi_dim ** 0.5)
        nn.init.normal_(self.phi_embed.weight, mean=0.0, std=phi_init_std)

        # ── Positional embeddings (optional) ─────────────────────────
        if self.use_positional_embedding:
            self.pos_embed = nn.Embedding(config.max_seq_len, config.embed_dim)
            nn.init.normal_(self.pos_embed.weight, mean=0.0, std=init_std)

    def forward(
        self, token_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed tokens as agent beliefs.

        Args:
            token_ids: (B, N) integer token indices

        Returns:
            mu: (B, N, K) mean beliefs
            sigma: (B, N, K, K) or (B, N, K) covariances
            phi: (B, N, phi_dim) gauge frames
        """
        batch_size, num_agents = token_ids.shape

        # Gauge frames
        phi = self.phi_embed(token_ids)

        # Mean and covariance
        if self.gauge_fixed_priors:
            phi_matrix = torch.einsum('bnc,ckl->bnkl', phi, self.generators)
            R = torch.linalg.matrix_exp(phi_matrix)
            mu = torch.einsum('bnkl,l->bnk', R, self.base_mu)

            sigma_diag_base = torch.exp(self.base_log_sigma_diag).clamp(min=0.01, max=5.0)
            Sigma_0 = torch.diag(sigma_diag_base)
            sigma = torch.einsum('bnij,jk,bnlk->bnil', R, Sigma_0, R)
        else:
            mu = self.mu_embed(token_ids)

            if self.learnable_sigma:
                log_sigma = self.log_sigma_diag[token_ids]
                sigma_diag = torch.exp(log_sigma).clamp(min=0.01, max=5.0)
            else:
                sigma_diag = torch.exp(self.log_sigma_diag).clamp(min=0.01, max=5.0)
                sigma_diag = sigma_diag.unsqueeze(0).unsqueeze(0).expand(batch_size, num_agents, -1)

            if self.diagonal_covariance:
                sigma = sigma_diag
            else:
                sigma = torch.diag_embed(sigma_diag)

        # Positional embeddings added to μ
        if self.use_positional_embedding:
            positions = torch.arange(num_agents, device=token_ids.device)
            mu = mu + self.pos_embed(positions).unsqueeze(0)

        # μ normalization
        if self.mu_normalize:
            mu = F.normalize(mu, dim=-1)
        elif self.mu_max_norm is not None:
            mu_norm = mu.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            scale = torch.clamp(self.mu_max_norm / mu_norm, max=1.0)
            mu = mu * scale

        return mu, sigma, phi

    def update_embeddings_from_beliefs(
        self,
        token_ids: torch.Tensor,
        mu_beliefs: torch.Tensor,
        prediction_errors: torch.Tensor,
        ema_decay: float = 0.99,
        min_weight: float = 0.01,
    ):
        """P-flow: Update token embeddings toward successful beliefs via EMA."""
        if self.gauge_fixed_priors:
            return

        B, N, K = mu_beliefs.shape
        lr = 1.0 - ema_decay

        with torch.no_grad():
            errors_clamped = prediction_errors.clamp(min=1e-6, max=20.0)
            weights = torch.softmax(-errors_clamped, dim=-1).clamp(min=min_weight)

            for token_id in token_ids.unique():
                mask = (token_ids == token_id)
                if mask.sum() == 0:
                    continue

                token_beliefs = mu_beliefs[mask]
                token_weights = weights[mask]
                total_weight = token_weights.sum()
                weighted_belief = (token_beliefs * token_weights.unsqueeze(-1)).sum(0) / total_weight

                current = self.mu_embed.weight.data[token_id]
                self.mu_embed.weight.data[token_id] = (1.0 - lr) * current + lr * weighted_belief


class GaugePositionalEncoding(nn.Module):
    """Agent-index-dependent gauge frame modulation.

    Encodes AGENT INDEX (not spatial position) via gauge frame modulation.
    All agents at same point c*, distinguished by gauge frames.

    Modes: 'none', 'learned', 'sinusoidal'
    Composition: 'add', 'bch1', 'bch2', 'exact' (SO(3) only)
    """

    def __init__(self, config: GaugeTransformerConfig, generators: Optional[torch.Tensor] = None):
        super().__init__()
        self.max_seq_len = config.max_seq_len
        self.mode = config.pos_encoding_mode
        self.scale = config.pos_encoding_scale
        self.phi_dim = config.phi_dim

        if generators is not None:
            self.register_buffer('generators', generators)
        else:
            self.generators = None

        # Determine composition mode
        composition = 'exact' if config.phi_dim == 3 else 'bch2'
        if config.phi_dim != 3 and composition == 'exact':
            composition = 'bch2'
        if composition in ['bch1', 'bch2'] and generators is None and not SON_BCH_AVAILABLE:
            composition = 'add'
        self.composition = composition

        if self.mode == 'none':
            self.register_buffer('pos_phi', torch.zeros(config.max_seq_len, config.phi_dim))
        elif self.mode == 'learned':
            self.pos_phi = nn.Parameter(
                torch.randn(config.max_seq_len, config.phi_dim) * config.pos_encoding_scale
            )
        elif self.mode == 'sinusoidal':
            self.register_buffer('pos_phi', self._make_sinusoidal(
                config.max_seq_len, config.pos_encoding_scale, config.phi_dim
            ))
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _make_sinusoidal(self, max_len: int, scale: float, phi_dim: int) -> torch.Tensor:
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, phi_dim, 1, dtype=torch.float32) * -(math.log(10000.0) / phi_dim)
        )
        phi = torch.zeros(max_len, phi_dim)
        for d in range(phi_dim):
            if d % 2 == 0:
                phi[:, d] = torch.sin(position.squeeze(-1) * div_term[d])
            else:
                phi[:, d] = torch.cos(position.squeeze(-1) * div_term[d])
        return phi * scale

    def forward(self, num_agents: int, device: Optional[torch.device] = None) -> torch.Tensor:
        if num_agents > self.max_seq_len:
            raise ValueError(f"Sequence length {num_agents} exceeds max {self.max_seq_len}")
        pos_phi = self.pos_phi[:num_agents]
        if device is not None:
            pos_phi = pos_phi.to(device)
        return pos_phi

    def compose(
        self, phi: torch.Tensor, num_agents: int,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Compose token gauge frames with positional gauge frames."""
        if self.mode == 'none':
            return phi

        pos_phi = self.forward(num_agents, device)
        pos_phi = pos_phi.unsqueeze(0).expand(phi.shape[0], -1, -1)

        if self.composition == 'add':
            return phi + pos_phi
        elif self.composition == 'bch1':
            if self.phi_dim == 3:
                return so3_compose_bch(phi, pos_phi, order=1)
            elif SON_BCH_AVAILABLE and self.generators is not None:
                return soN_compose_bch_torch(phi, pos_phi, self.generators, order=1)
            return phi + pos_phi
        elif self.composition == 'bch2':
            if self.phi_dim == 3:
                return so3_compose_bch(phi, pos_phi, order=2)
            elif SON_BCH_AVAILABLE and self.generators is not None:
                return soN_compose_bch_torch(phi, pos_phi, self.generators, order=2)
            return phi + pos_phi
        elif self.composition == 'exact':
            def skew_symmetric_batch(v):
                zeros = torch.zeros_like(v[..., 0])
                return torch.stack([
                    torch.stack([zeros, -v[..., 2], v[..., 1]], dim=-1),
                    torch.stack([v[..., 2], zeros, -v[..., 0]], dim=-1),
                    torch.stack([-v[..., 1], v[..., 0], zeros], dim=-1),
                ], dim=-2)

            R_phi = torch.linalg.matrix_exp(skew_symmetric_batch(phi))
            R_pos = torch.linalg.matrix_exp(skew_symmetric_batch(pos_phi))
            return so3_log(R_phi @ R_pos)
        else:
            raise ValueError(f"Unknown composition: {self.composition}")
