# -*- coding: utf-8 -*-
"""
Token-Dependent Prior Bank for VFE Transformers
================================================

Each vocabulary token v has a prior belief distribution: π_v = N(μ_v, Σ_v)

Enables:
1. ENCODING: Initialize beliefs from token priors (replaces nn.Embedding)
2. DECODING: Compute logits via KL to priors (replaces nn.Linear)
3. LEARNING: Update priors via prediction-error weighted EMA (pure FEP mode)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class PriorBank(nn.Module):
    """Token-dependent prior bank for pure FEP learning.

    Each token v has π_v = N(μ_v, Σ_v).

    GAUGE-FIXED PRIORS (optional):
        All token priors are rotations of a SINGLE base prior:
            π_v = R_v ▷ π_0   where R_v = exp(φ_v · G)

    NON-GAUGE-FIXED (default):
        Each token has independent μ_v, Σ_v.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        init_std: Optional[float] = None,
        init_sigma_scale: float = 1.0,
        learnable_sigma: bool = True,
        eps: float = 1e-6,
        gauge_fixed_priors: bool = False,
        generators: Optional[torch.Tensor] = None,
        phi_dim: int = 3,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.eps = eps
        self.learnable_sigma = learnable_sigma
        self.gauge_fixed_priors = gauge_fixed_priors
        self.phi_dim = phi_dim

        if init_std is None:
            init_std = 1.0 / math.sqrt(embed_dim)

        if gauge_fixed_priors:
            if generators is None:
                raise ValueError("gauge_fixed_priors=True requires generators")
            self.register_buffer('generators', generators)

            self.base_prior_mu = nn.Parameter(torch.randn(embed_dim) * init_std)
            if learnable_sigma:
                self.base_log_prior_sigma = nn.Parameter(
                    torch.full((embed_dim,), math.log(init_sigma_scale))
                )
            else:
                self.register_buffer(
                    'base_log_prior_sigma',
                    torch.full((embed_dim,), math.log(init_sigma_scale))
                )

            self.phi_embed = nn.Embedding(vocab_size, phi_dim)
            nn.init.zeros_(self.phi_embed.weight)
        else:
            self.prior_mu = nn.Parameter(torch.randn(vocab_size, embed_dim) * init_std)
            if learnable_sigma:
                self.log_prior_sigma = nn.Parameter(
                    torch.full((vocab_size, embed_dim), math.log(init_sigma_scale))
                )
            else:
                self.register_buffer(
                    'log_prior_sigma',
                    torch.full((vocab_size, embed_dim), math.log(init_sigma_scale))
                )

    @property
    def base_prior_sigma(self) -> torch.Tensor:
        return torch.exp(self.base_log_prior_sigma).clamp(min=self.eps)

    @property
    def prior_sigma(self) -> torch.Tensor:
        return torch.exp(self.log_prior_sigma).clamp(min=self.eps)

    def _compute_rotation(self, phi: torch.Tensor) -> torch.Tensor:
        phi_dot_G = torch.einsum('...a,aij->...ij', phi, self.generators)
        return torch.linalg.matrix_exp(phi_dot_G)

    def _get_prior_for_tokens(
        self, token_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get prior (μ, σ, φ) for given tokens. Indexed by TOKEN ID."""
        if self.gauge_fixed_priors:
            phi = self.phi_embed(token_ids)
            R = self._compute_rotation(phi)
            mu_p = torch.einsum('...ij,j->...i', R, self.base_prior_mu)
            base_sigma = self.base_prior_sigma
            R_sq = R ** 2
            sigma_p = torch.einsum('...kl,l->...k', R_sq, base_sigma)
            return mu_p, sigma_p, phi
        else:
            mu_p = self.prior_mu[token_ids]
            sigma_p = self.prior_sigma[token_ids]
            phi = torch.zeros(*token_ids.shape, self.phi_dim, device=token_ids.device)
            return mu_p, sigma_p, phi

    def encode(self, token_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode tokens by looking up their prior beliefs. Replaces nn.Embedding."""
        return self._get_prior_for_tokens(token_ids)

    def decode(
        self,
        mu_q: torch.Tensor,
        sigma_q: torch.Tensor,
        tau: float = 1.0,
    ) -> torch.Tensor:
        """Compute logits via KL to all token priors.

        p(y = v | q) ∝ exp(-KL(q || π_v) / τ)
        """
        B, N, K = mu_q.shape
        V = self.vocab_size
        device = mu_q.device

        all_token_ids = torch.arange(V, device=device)
        mu_p, sigma_p, _ = self._get_prior_for_tokens(all_token_ids)

        mu_q_exp = mu_q.unsqueeze(2)
        sigma_q_exp = sigma_q.unsqueeze(2)
        mu_p_exp = mu_p.unsqueeze(0).unsqueeze(0)
        sigma_p_exp = sigma_p.unsqueeze(0).unsqueeze(0)

        variance_floor = max(self.eps, 1e-4)
        sigma_q_safe = sigma_q_exp.clamp(min=variance_floor)
        sigma_p_safe = sigma_p_exp.clamp(min=variance_floor)

        kl_per_dim = 0.5 * (
            sigma_q_safe / sigma_p_safe
            + (mu_q_exp - mu_p_exp)**2 / sigma_p_safe
            - 1.0
            + torch.log(sigma_p_safe / sigma_q_safe)
        )

        kl_total = kl_per_dim.sum(dim=-1)
        return -kl_total / tau

    def update_from_beliefs(
        self,
        token_ids: torch.Tensor,
        mu_beliefs: torch.Tensor,
        sigma_beliefs: torch.Tensor,
        prediction_errors: torch.Tensor,
        lr: float = 0.01,
    ):
        """Update priors via prediction-error weighted EMA (pure FEP learning)."""
        if self.gauge_fixed_priors:
            return

        with torch.no_grad():
            B, N, K = mu_beliefs.shape
            unique_tokens = torch.unique(token_ids)

            for token_id in unique_tokens:
                mask = (token_ids == token_id)
                if mask.sum() == 0:
                    continue

                token_mu = mu_beliefs[mask]
                token_sigma = sigma_beliefs[mask]
                token_errors = prediction_errors[mask]

                weights = F.softmax(-token_errors.clamp(min=-10, max=10), dim=0)
                weighted_mu = (token_mu * weights.unsqueeze(-1)).sum(0)
                weighted_sigma = (token_sigma * weights.unsqueeze(-1)).sum(0)

                mean_error = token_errors.mean()
                confidence = 1.0 / (1.0 + mean_error)
                effective_lr = lr * confidence

                token_id_int = int(token_id.item())
                self.prior_mu.data[token_id_int] = (
                    (1.0 - effective_lr) * self.prior_mu.data[token_id_int] +
                    effective_lr * weighted_mu.detach()
                )

                if self.learnable_sigma:
                    sigma_lr = effective_lr * 0.1
                    current_sigma = torch.exp(self.log_prior_sigma.data[token_id_int])
                    new_sigma = (1.0 - sigma_lr) * current_sigma + sigma_lr * weighted_sigma.detach()
                    self.log_prior_sigma.data[token_id_int] = torch.log(new_sigma.clamp(min=self.eps))

    def forward(
        self,
        token_ids: Optional[torch.Tensor] = None,
        mu_q: Optional[torch.Tensor] = None,
        sigma_q: Optional[torch.Tensor] = None,
        mode: str = 'encode',
        tau: float = 1.0,
    ):
        if mode == 'encode':
            assert token_ids is not None
            return self.encode(token_ids)
        elif mode == 'decode':
            assert mu_q is not None and sigma_q is not None
            return self.decode(mu_q, sigma_q, tau)
        else:
            raise ValueError(f"Unknown mode: {mode}")
