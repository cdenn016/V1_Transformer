# -*- coding: utf-8 -*-
"""
Created on Fri Dec 26 17:06:07 2025

@author: chris and christine
"""

"""
Token-Dependent Prior Bank for VFE Transformers
================================================

Unified prior bank that serves as BOTH embedding and output projection.
Each vocabulary token v has a prior belief distribution: π_v = N(μ_v, Σ_v)

This module enables:
1. ENCODING: Initialize beliefs from token priors (replaces nn.Embedding)
2. DECODING: Compute logits via KL to priors (replaces nn.Linear)
3. LEARNING: Priors evolve via backprop (VFE_EM mode)

Date: December 2025
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Union


class PriorBank(nn.Module):
    """
    Token-dependent prior bank for VFE transformer.

    Each vocabulary token v has a prior belief distribution:
        π_v = N(μ_v, Σ_v)

    GAUGE-FIXED PRIORS (theoretically principled):
        All token priors are rotations of a SINGLE base prior:
            π_v = R_v ▷ π_0   where R_v = exp(φ_v · G)

        This guarantees gauge covariance: π_i = Ω_ij[π_j] for all i,j
        The model learns:
        - base_prior_mu (K,): shared base prior mean
        - phi_embed (V, phi_dim): per-token gauge frames defining rotations

    NON-GAUGE-FIXED (default):
        Each token has independent μ_v, Σ_v plus learnable gauge frames φ_v.

    ENCODING (replaces nn.Embedding):
        Given input token y_t, initialize belief from prior:
        q(z_t) ← π_{y_t}

    DECODING (replaces nn.Linear output projection):
        Given belief q = N(μ_q, Σ_q), compute observation likelihood:
        p(y = v | q) ∝ exp(-KL(q || π_v) / τ)

    Learning: Priors updated via backprop (VFE_EM mode).
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        init_std: float = None,  # Default: 1/sqrt(embed_dim) for O(1) KL
        init_sigma_scale: float = 1.0,  # Scaled to match init_std for O(1) KL
        learnable_sigma: bool = True,
        eps: float = 1e-6,
        # Gauge-fixed priors (principled approach)
        gauge_fixed_priors: bool = False,  # Default False for pure FEP flexibility
        generators: Optional[torch.Tensor] = None,  # (n_gen, K, K) Lie algebra generators
        phi_dim: int = 3,  # 3 for SO(3), N(N-1)/2 for SO(N)
        phi_scale: float = 1.0,  # Init scale for gauge frames (non-gauge-fixed mode)
        # Direct Omega parameterization
        gauge_param: str = 'phi',  # 'phi' or 'omega'
        omega_head_dims: Optional[list] = None,  # Per-head dims for omega path
        # Gradient scaling for sigma in decode (CE loss path)
        sigma_ce_scale: float = 0.01,  # Fraction of CE gradient passed to sigma_p
    ):
        """
        Initialize the prior bank.

        Args:
            vocab_size: Number of tokens in vocabulary
            embed_dim: Embedding dimension K
            init_std: Std for initializing prior means (default: 1/sqrt(embed_dim))
            init_sigma_scale: Initial scale for prior variances
            learnable_sigma: If True, Σ_v evolves during training
            eps: Numerical stability
            gauge_fixed_priors: If True, use single base prior with per-token rotations
            generators: Lie algebra generators for computing rotations (required if gauge_fixed_priors=True)
            phi_dim: Dimension of gauge frame (3 for SO(3))
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.eps = eps
        self.learnable_sigma = learnable_sigma
        self.gauge_fixed_priors = gauge_fixed_priors
        self.phi_dim = phi_dim
        self.gauge_param = gauge_param
        self.omega_head_dims = omega_head_dims
        self.sigma_ce_scale = sigma_ce_scale

        # Dimension-aware initialization: √(ln V / K) makes pairwise KL ≈ ln(V).
        # Old 1/√K made KL = O(1), but attention divides by √K_h →
        # logit differences O(1/(H√K_h)) vanish for large K (e.g. K=90 GL(15)).
        # With √(ln V / K): ||μ||² = ln(V) ≈ 10.8, KL ≈ ln(V), and attention
        # logits ≈ -ln(V)/(H√K_h) which stays discriminative up to K ≈ ln²(V).
        if init_std is None:
            init_std = math.sqrt(math.log(vocab_size) / embed_dim)

        if gauge_fixed_priors:
            # Validate generators
            if generators is None:
                raise ValueError("gauge_fixed_priors=True requires generators to be provided")
            self.register_buffer('generators', generators)

            # Single base prior mean μ_0 - all token priors are rotations of this
            self.base_prior_mu = nn.Parameter(torch.randn(embed_dim) * init_std)

            # Single base prior variance (diagonal)
            if learnable_sigma:
                self.base_log_prior_sigma = nn.Parameter(
                    torch.full((embed_dim,), math.log(init_sigma_scale))
                )
            else:
                self.register_buffer(
                    'base_log_prior_sigma',
                    torch.full((embed_dim,), math.log(init_sigma_scale))
                )

            # Per-token gauge frames φ_v ∈ so(n) - defines rotation R_v = exp(φ_v · G)
            self.phi_embed = nn.Embedding(vocab_size, phi_dim)
            nn.init.zeros_(self.phi_embed.weight)  # Start at identity rotation
        else:
            # Standard per-token priors (TOKEN-DEPENDENT, not position-dependent!)
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

            # Fixed sigma target for hyper-prior KL(s||h). Frozen at init values
            # so _get_sigma_target() finds a proper anchor instead of falling back
            # to the moving target (2×mean(sigma_s)) which collapses with sigma_s.
            self.register_buffer(
                'sigma_target',
                torch.full((embed_dim,), init_sigma_scale)
            )

            if gauge_param == 'omega' and omega_head_dims is not None:
                # Direct Omega parameterization: per-head K_h×K_h matrices
                total_omega_params = sum(d * d for d in omega_head_dims)
                self.omega_embed = nn.Embedding(vocab_size, total_omega_params)
                with torch.no_grad():
                    omega_scale = phi_scale * 0.1
                    weight = torch.zeros(vocab_size, total_omega_params)
                    offset = 0
                    for d in omega_head_dims:
                        eye_flat = torch.eye(d).reshape(-1)
                        weight[:, offset:offset + d * d] = eye_flat.unsqueeze(0) + omega_scale * torch.randn(vocab_size, d * d)
                        offset += d * d
                    self.omega_embed.weight.copy_(weight)
                # Dummy phi_embed for compatibility
                self.phi_embed = nn.Embedding(vocab_size, phi_dim)
                nn.init.zeros_(self.phi_embed.weight)
            else:
                # Per-token gauge frames φ_v — needed for gauge transport even
                # without gauge-fixed priors. Without this, phi has no gradient path.
                # IMPORTANT: Scale std inversely with sqrt(phi_dim) to maintain consistent
                # norm across different GL(K) dimensions, matching GaugeTokenEmbedding.
                self.phi_embed = nn.Embedding(vocab_size, phi_dim)
                phi_init_std = phi_scale / (phi_dim ** 0.5)
                nn.init.normal_(self.phi_embed.weight, std=phi_init_std)

    @property
    def base_prior_sigma(self) -> torch.Tensor:
        r"""Get base prior variance (always positive). Only for gauge_fixed_priors=True.

        Hard clamp bounds output to [0.01, 5.0] for numerical safety:
        - min=0.01: prevents covariance collapse (singular precision matrix)
        - max=5.0: prevents diffuse priors from producing vanishing KL gradients

        Hard clamp zeros gradient at boundaries, preventing log_sigma drift.
        """
        _SIGMA_MIN, _SIGMA_MAX = 0.01, 5.0
        sigma = torch.exp(self.base_log_prior_sigma)
        return sigma.clamp(_SIGMA_MIN, _SIGMA_MAX)

    @property
    def prior_sigma(self) -> torch.Tensor:
        r"""Get prior variances (always positive). Only for gauge_fixed_priors=False.

        Hard clamp bounds output to [0.01, 5.0] for numerical safety.
        Zeros gradient at boundaries, preventing log_sigma drift.
        """
        _SIGMA_MIN, _SIGMA_MAX = 0.01, 5.0
        sigma = torch.exp(self.log_prior_sigma)
        return sigma.clamp(_SIGMA_MIN, _SIGMA_MAX)

    def _compute_rotation(self, phi: torch.Tensor) -> torch.Tensor:
        """
        Compute rotation matrix R = exp(φ · G) from gauge frame.

        Args:
            phi: (..., phi_dim) gauge frames

        Returns:
            R: (..., K, K) rotation matrices
        """
        # Compute φ · G = Σ_a φ_a G_a
        # generators: (n_gen, K, K), phi: (..., phi_dim)
        # Result: (..., K, K)
        phi_dot_G = torch.einsum('...a,aij->...ij', phi, self.generators)

        # Matrix exponential
        R = torch.linalg.matrix_exp(phi_dot_G)
        return R

    def _get_prior_for_tokens(
        self,
        token_ids: torch.Tensor,  # (B, N) or (V,) for all vocab
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get prior (μ, σ, φ) for given tokens.

        CRITICAL: Indexing by TOKEN ID, not position!

        Returns:
            mu_p: prior means
            sigma_p: prior variances (diagonal)
            phi_p: gauge frames
        """
        if self.gauge_fixed_priors:
            # Get per-token gauge frames
            phi = self.phi_embed(token_ids)  # (..., phi_dim)

            # Compute rotation matrices R_v = exp(φ_v · G)
            R = self._compute_rotation(phi)  # (..., K, K)

            # Rotate base prior: μ_v = R_v @ μ_0
            mu_p = torch.einsum('...ij,j->...i', R, self.base_prior_mu)

            # Rotate base covariance: Σ_v = R_v @ diag(σ_0) @ R_v^T
            # WARNING: When gauge_fixed_priors=True, we compute the FULL rotated
            # covariance to preserve gauge covariance (Σ_i = Ω_ij Σ_j Ω_ij^T).
            # The diagonal-only approximation (R² @ σ_0) breaks this invariant.
            base_sigma = self.base_prior_sigma  # (K,)
            if getattr(self, 'diagonal_covariance', True):
                # Diagonal approximation: extract only diagonal of R @ diag(σ) @ R^T
                # This is an approximation that breaks gauge covariance in off-diagonals.
                # Use full covariance mode for exact gauge-equivariant transport.
                R_sq = R ** 2  # (..., K, K)
                sigma_p = torch.einsum('...kl,l->...k', R_sq, base_sigma)  # (..., K)
            else:
                # Full covariance: Σ_v = R @ diag(σ_0) @ R^T (gauge-covariant)
                Sigma_0 = torch.diag(base_sigma)  # (K, K)
                sigma_p = torch.einsum('...ij,jk,...lk->...il', R, Sigma_0, R)  # (..., K, K)

            return mu_p, sigma_p, phi
        else:
            # Standard per-token lookup (TOKEN-INDEXED!)
            mu_p = self.prior_mu[token_ids]  # Index by token ID
            sigma_p = self.prior_sigma[token_ids]
            # Learnable per-token gauge frames
            phi = self.phi_embed(token_ids)

            if self.gauge_param == 'omega' and hasattr(self, 'omega_embed'):
                # Return omega matrices alongside phi (phi is dummy zeros)
                omega_flat = self.omega_embed(token_ids)
                K = self.embed_dim
                omega = torch.zeros(*token_ids.shape, K, K,
                                    device=token_ids.device, dtype=omega_flat.dtype)
                offset = 0
                block_start = 0
                for d in self.omega_head_dims:
                    omega_blk = omega_flat[..., offset:offset + d * d].reshape(
                        *token_ids.shape, d, d)
                    omega[..., block_start:block_start + d, block_start:block_start + d] = omega_blk
                    offset += d * d
                    block_start += d
                return mu_p, sigma_p, phi, omega

            return mu_p, sigma_p, phi

    def encode(
        self,
        token_ids: torch.Tensor,  # (B, N)
    ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
               Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Encode tokens by looking up their prior beliefs.

        This replaces the standard nn.Embedding lookup.

        Args:
            token_ids: (B, N) input token IDs

        Returns:
            mu_q: (B, N, K) belief means initialized from priors
            sigma_q: (B, N, K) belief variances initialized from priors
            phi: (B, N, phi_dim) gauge frames for tokens
            omega: (B, N, K, K) direct gauge frames (only when gauge_param='omega')
        """
        return self._get_prior_for_tokens(token_ids)

    def decode(
        self,
        mu_q: torch.Tensor,      # (B, N, K) belief means
        sigma_q: torch.Tensor,   # (B, N, K) belief variances (diagonal)
        tau: float = 1.0,        # Temperature
    ) -> torch.Tensor:
        """
        Compute observation likelihood via KL to all token priors.

        p(y = v | q) ∝ exp(-KL(q || π_v) / τ)

        Fused single-matmul implementation. The full diagonal KL is:

            KL(q || π_v) = 0.5 * [tr(Σ_q/Σ_p) + (μ_q-μ_p)ᵀΣ_p⁻¹(μ_q-μ_p)
                                   - K + log|Σ_p|/|Σ_q|]

        Key optimizations:
        1. Fuse trace + quad_q - cross into ONE matmul via concatenation:
           [σ_q + μ_q², -2μ_q] @ [1/σ_p, μ_p/σ_p]ᵀ  → (B,N,2K) x (2K,V)
        2. Drop terms constant across V (cancel in softmax): -K, log|Σ_q|
        3. Combine prior-side constants into single (V,) bias

        Args:
            mu_q: (B, N, K) belief means
            sigma_q: (B, N, K) belief variances
            tau: Temperature for softmax

        Returns:
            logits: (B, N, vocab_size) log-probabilities (unnormalized)
        """
        B, N, K = mu_q.shape
        V = self.vocab_size
        device = mu_q.device

        # Get all token priors: mu_p (V, K), sigma_p (V, K)
        all_token_ids = torch.arange(V, device=device)
        _prior_out = self._get_prior_for_tokens(all_token_ids)
        mu_p, sigma_p = _prior_out[0], _prior_out[1]

        variance_floor = max(self.eps, 1e-4)
        sigma_q_safe = sigma_q.clamp(min=variance_floor)    # (B, N, K)
        sigma_p_clamped = sigma_p.clamp(min=variance_floor)  # (V, K)

        # Scale sigma_p gradient from CE's precision (1/σ_p) terms.
        # Gradient to log_sigma_p scales as (mu_q-mu_p)²/sigma_p, creating a
        # positive feedback loop: CE shrinks sigma_p for discrimination →
        # gradient grows as 1/sigma_p → sigma_p shrinks faster → explosion.
        # Detach-scale trick: forward value unchanged, backward gets scale×gradient.
        # sigma_q stays unscaled (its CE gradient flows back to E-step parameters).
        _s = self.sigma_ce_scale
        sigma_p_safe = sigma_p_clamped.detach() + _s * (sigma_p_clamped - sigma_p_clamped.detach())

        inv_sigma_p = 1.0 / sigma_p_safe                    # (V, K)
        mu_p_inv_sigma_p = mu_p * inv_sigma_p               # (V, K)

        # Fused matmul: trace + quad_q - cross in ONE operation
        # LHS = [σ_q + μ_q², -2·μ_q]         → (B, N, 2K)
        # RHS = [1/σ_p,  μ_p/σ_p]             → (V, 2K)
        # LHS @ RHS.T = σ_q·(1/σ_p) + μ_q²·(1/σ_p) - 2·μ_q·(μ_p/σ_p)
        #             = trace_term + quad_q - cross
        lhs = torch.cat([sigma_q_safe + mu_q ** 2, -2.0 * mu_q], dim=-1)  # (B, N, 2K)
        rhs = torch.cat([inv_sigma_p, mu_p_inv_sigma_p], dim=-1)           # (V, 2K)

        combined = torch.matmul(lhs, rhs.T)                 # (B, N, V) — single matmul

        # Prior-side bias (constant across batch): quad_p + log_det_p
        # quad_p = Σ_k μ_p[v,k]² / σ_p[v,k]
        # log_det_p = Σ_k log(σ_p[v,k])
        # Note: -K and -log|Σ_q| are constant across V, cancel in softmax
        prior_bias = ((mu_p ** 2 * inv_sigma_p).sum(dim=-1)
                      + torch.log(sigma_p_safe).sum(dim=-1))  # (V,)

        # logits = -KL/τ ≈ -0.5/τ * (combined + prior_bias)
        # (dropping softmax-invariant terms -K and log|Σ_q|)
        #
        # √K concentration-of-measure correction: with init_std = √(ln V / K),
        # each KL term is O(ln V / K), summing K terms gives E[KL] = ln(V) but
        # std(KL) = O(ln V / √K). Raw logit spread thus shrinks as 1/√K —
        # at K=90 the softmax is near-uniform, killing gradient signal.
        # Multiplying by √K restores K-independent logit spread.
        # (Attention uses /√d_h because per-head beliefs have O(1) variance;
        # here init_std already bakes in 1/√K, so we compensate oppositely.)
        dim_scale = math.sqrt(K)
        logits = -0.5 * dim_scale / tau * (combined + prior_bias.unsqueeze(0).unsqueeze(0))

        return logits

    def update_from_beliefs(
        self,
        token_ids: torch.Tensor,       # (B, N) token IDs
        mu_beliefs: torch.Tensor,      # (B, N, K) evolved belief means
        sigma_beliefs: torch.Tensor,   # (B, N, K) belief variances
        prediction_errors: torch.Tensor,  # (B, N) per-position CE loss
        lr: float = 0.01,
    ):
        """
        Update token priors via prediction-error weighted EMA.

        This is the pure FEP learning mechanism:
        - Beliefs with low prediction error are "good" - priors should move toward them
        - Beliefs with high prediction error are "bad" - priors should ignore them
        - For each token, aggregate across all its occurrences in the batch

        CRITICAL: Updates priors by TOKEN ID, not position!

        Args:
            token_ids: (B, N) token IDs in the batch
            mu_beliefs: (B, N, K) evolved belief means after VFE
            sigma_beliefs: (B, N, K) evolved belief variances
            prediction_errors: (B, N) per-position cross-entropy loss
            lr: Learning rate for prior updates
        """
        if self.gauge_fixed_priors:
            # For gauge-fixed priors, updates would need to be in phi space
            # TODO: Implement gauge-fixed prior updates
            return

        with torch.no_grad():
            B, N, K = mu_beliefs.shape

            # Flatten batch dimensions: (B*N,)
            flat_ids = token_ids.reshape(-1)           # (B*N,)
            flat_mu = mu_beliefs.reshape(-1, K)        # (B*N, K)
            flat_sigma = sigma_beliefs.reshape(-1, K)  # (B*N, K)
            flat_errors = prediction_errors.reshape(-1)  # (B*N,)

            # Compute per-token-type weights via segment-wise softmax:
            # 1. Find max error per token type (for numerical stability)
            neg_errors = -flat_errors.clamp(min=-10, max=10)
            unique_tokens, inverse_idx = torch.unique(flat_ids, return_inverse=True)
            n_unique = unique_tokens.shape[0]

            # Segment-wise max for stable softmax
            seg_max = torch.full((n_unique,), float('-inf'), device=flat_ids.device, dtype=flat_mu.dtype)
            seg_max.scatter_reduce_(0, inverse_idx, neg_errors, reduce='amax', include_self=False)
            shifted = neg_errors - seg_max[inverse_idx]
            exp_shifted = torch.exp(shifted)

            # Segment-wise sum of exp for normalization
            seg_sum = torch.zeros(n_unique, device=flat_ids.device, dtype=flat_mu.dtype)
            seg_sum.scatter_add_(0, inverse_idx, exp_shifted)
            weights = exp_shifted / seg_sum[inverse_idx].clamp(min=1e-12)  # (B*N,)

            # Weighted means per token type via scatter_add
            weighted_mu = torch.zeros(n_unique, K, device=flat_ids.device, dtype=flat_mu.dtype)
            weighted_mu.scatter_add_(0, inverse_idx.unsqueeze(-1).expand(-1, K), flat_mu * weights.unsqueeze(-1))

            # Mean error per token type for confidence-weighted LR
            seg_error_sum = torch.zeros(n_unique, device=flat_ids.device, dtype=flat_mu.dtype)
            seg_error_sum.scatter_add_(0, inverse_idx, flat_errors)
            seg_count = torch.zeros(n_unique, device=flat_ids.device, dtype=flat_mu.dtype)
            seg_count.scatter_add_(0, inverse_idx, torch.ones_like(flat_errors))
            mean_errors = seg_error_sum / seg_count.clamp(min=1)
            confidence = 1.0 / (1.0 + mean_errors)  # (n_unique,)
            effective_lr = lr * confidence  # (n_unique,)

            # Vectorized EMA update for mu: prior ← (1-lr)*prior + lr*belief
            self.prior_mu.data[unique_tokens] = (
                (1.0 - effective_lr.unsqueeze(-1)) * self.prior_mu.data[unique_tokens] +
                effective_lr.unsqueeze(-1) * weighted_mu
            )

            # Vectorized EMA update for sigma
            if self.learnable_sigma:
                weighted_sigma = torch.zeros(n_unique, K, device=flat_ids.device, dtype=flat_mu.dtype)
                weighted_sigma.scatter_add_(0, inverse_idx.unsqueeze(-1).expand(-1, K), flat_sigma * weights.unsqueeze(-1))
                sigma_lr = effective_lr * 0.1
                current_sigma = torch.exp(self.log_prior_sigma.data[unique_tokens])
                new_sigma = (1.0 - sigma_lr.unsqueeze(-1)) * current_sigma + sigma_lr.unsqueeze(-1) * weighted_sigma
                self.log_prior_sigma.data[unique_tokens] = torch.log(new_sigma.clamp(min=self.eps))

    def forward(
        self,
        token_ids: Optional[torch.Tensor] = None,
        mu_q: Optional[torch.Tensor] = None,
        sigma_q: Optional[torch.Tensor] = None,
        mode: str = 'encode',
        tau: float = 1.0,
    ):
        """
        Forward pass - encode or decode.

        Args:
            token_ids: (B, N) for encoding
            mu_q, sigma_q: (B, N, K) for decoding
            mode: 'encode' or 'decode'
            tau: Temperature for decoding
        """
        if mode == 'encode':
            assert token_ids is not None
            return self.encode(token_ids)
        elif mode == 'decode':
            assert mu_q is not None and sigma_q is not None
            return self.decode(mu_q, sigma_q, tau)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def extra_repr(self) -> str:
        """Pretty print for model summary."""
        return (
            f"vocab_size={self.vocab_size}, "
            f"embed_dim={self.embed_dim}, "
            f"learnable_sigma={self.learnable_sigma}, "
            f"gauge_fixed_priors={self.gauge_fixed_priors}"
        )
