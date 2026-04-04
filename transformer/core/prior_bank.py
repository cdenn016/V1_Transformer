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
from transformer.core.gauge_utils import stable_matrix_exp_pair


class PriorBank(nn.Module):
    """
    Token-dependent prior bank for VFE transformer.

    Each vocabulary token v has a prior belief distribution:
        π_v = N(μ_v, Σ_v)

    GAUGE-FIXED PRIORS (theoretically principled):
        All token priors are gauge transforms of a SINGLE base prior:
            π_v = A_v ▷ π_0   where A_v = exp(φ_v · G) ∈ GL⁺(K)

        Under GL(K), A_v is a general invertible matrix (not just a rotation).
        The orbit of a full-rank (μ_0, Σ_0) under GL(K) covers all nonzero
        means and all SPD covariances, so this parameterization is NOT
        restrictive -- it reparameterizes the full space through (base + frame).

        This guarantees gauge covariance: π_i = Ω_ij[π_j] for all i,j
        The model learns:
        - base_prior_mu (K,): shared base prior mean
        - base_log_prior_sigma (K,): shared base prior log-variance
        - phi_embed (V, phi_dim): per-token gauge frames (phi_dim = K^2 for GL(K))

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
        phi_dim: int = None,  # Auto-inferred from generators; K^2 for GL(K), 3 for SO(3)
        phi_scale: float = 1.0,  # Init scale for gauge frames (non-gauge-fixed mode)
        # Direct Omega parameterization
        gauge_param: str = 'phi',  # 'phi' or 'omega'
        omega_head_dims: Optional[list] = None,  # Per-head dims for omega path
        # Gradient scaling for sigma in decode (CE loss path)
        sigma_ce_scale: float = 0.1,  # Fraction of CE gradient passed to sigma_p
        # Learnable inverse-temperature for decode logits
        learnable_temperature: bool = False,  # If True, learn decode scale factor
        # Covariance mode (must match model's diagonal_covariance setting)
        diagonal_covariance: bool = True,  # If False, gauge_fixed_priors returns full Σ_v = R@diag(σ_0)@R^T
        sigma_max: float = 5.0,  # Upper bound for prior covariance clamp
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
            gauge_fixed_priors: If True, use single base prior with per-token gauge transforms
            generators: Lie algebra generators (required if gauge_fixed_priors=True).
                For GL(K): K^2 generators spanning gl(K). For SO(N): N(N-1)/2 generators.
            phi_dim: Dimension of gauge frame. Auto-inferred from generators.shape[0]
                if not provided. K^2 for GL(K), 3 for SO(3), N(N-1)/2 for SO(N).
            sigma_ce_scale: Fraction of CE gradient passed to sigma_p (0.01 recommended)
            learnable_temperature: If True, learn a decode scale factor via backprop.
                Starts at scale=1 (CE ~= ln V at init). Gradient is self-regulating:
                dCE/d(log_scale) = scale·[E_p[KL] - KL_target], which decreases scale
                when overconfident on wrong tokens and increases it as predictions improve.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.eps = eps
        self.learnable_sigma = learnable_sigma
        self.gauge_fixed_priors = gauge_fixed_priors
        # Auto-infer phi_dim from generators if not provided
        if phi_dim is None:
            phi_dim = generators.shape[0] if generators is not None else 3
        self.phi_dim = phi_dim
        self.gauge_param = gauge_param
        self.omega_head_dims = omega_head_dims
        self.sigma_ce_scale = sigma_ce_scale
        self.sigma_max = sigma_max
        self.learnable_temperature = learnable_temperature
        self.diagonal_covariance = diagonal_covariance

        # Learnable inverse-temperature for decode logits.
        # At init, scale = exp(0) = 1.0 -> no amplification -> CE ~= ln(V).
        # The model discovers the right K-dependent scale during training.
        if learnable_temperature:
            self.decode_log_scale = nn.Parameter(torch.tensor(0.0))

        # Dimension-aware initialization: √(ln V / K) makes pairwise KL ~= ln(V).
        # Old 1/√K made KL = O(1), but attention divides by √K_h ->
        # logit differences O(1/(H√K_h)) vanish for large K (e.g. K=90 GL(15)).
        # With √(ln V / K): ||μ||^2 = ln(V) ~= 10.8, KL ~= ln(V), and attention
        # logits ~= -ln(V)/(H√K_h) which stays discriminative up to K ~= ln^2(V).
        if init_std is None:
            init_std = math.sqrt(math.log(vocab_size) / embed_dim)

        if gauge_fixed_priors:
            # Validate generators
            if generators is None:
                raise ValueError("gauge_fixed_priors=True requires generators to be provided")
            self.register_buffer('generators', generators)

            # Single base prior mean μ_0 -- all token priors are gauge transforms:
            #   μ_v = A_v @ μ_0  where A_v = exp(φ_v · G) ∈ GL⁺(K)
            self.base_prior_mu = nn.Parameter(torch.randn(embed_dim) * init_std)

            # Single base prior variance (diagonal of Σ_0)
            # Covariance transport: Σ_v = A_v @ diag(σ_0) @ A_v^T
            if learnable_sigma:
                self.base_log_prior_sigma = nn.Parameter(
                    torch.full((embed_dim,), math.log(init_sigma_scale))
                )
            else:
                self.register_buffer(
                    'base_log_prior_sigma',
                    torch.full((embed_dim,), math.log(init_sigma_scale))
                )

            # Per-token gauge frames φ_v ∈ g (Lie algebra)
            # For GL(K): φ_v ∈ gl(K), A_v = exp(φ_v · G) ∈ GL⁺(K)
            # For SO(N): φ_v ∈ so(N), A_v = exp(φ_v · G) ∈ SO(N)
            #
            # MUST use random init -- zero init makes ALL tokens identical
            # (exp(0) = I -> μ_v = μ_0 for all v). Scale inversely with
            # sqrt(phi_dim) to maintain consistent ||φ|| across gauge groups.
            self.phi_embed = nn.Embedding(vocab_size, phi_dim)
            phi_init_std = phi_scale / (phi_dim ** 0.5)
            nn.init.normal_(self.phi_embed.weight, mean=0.0, std=phi_init_std)
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
            # to the moving target (2xmean(sigma_s)) which collapses with sigma_s.
            self.register_buffer(
                'sigma_target',
                torch.full((embed_dim,), init_sigma_scale)
            )

            if gauge_param == 'omega' and omega_head_dims is not None:
                # Direct Omega parameterization: per-head K_hxK_h matrices
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
                # Per-token gauge frames φ_v -- needed for gauge transport even
                # without gauge-fixed priors. Without this, phi has no gradient path.
                # IMPORTANT: Scale std inversely with sqrt(phi_dim) to maintain consistent
                # norm across different GL(K) dimensions, matching GaugeTokenEmbedding.
                self.phi_embed = nn.Embedding(vocab_size, phi_dim)
                phi_init_std = phi_scale / (phi_dim ** 0.5)
                nn.init.normal_(self.phi_embed.weight, std=phi_init_std)

    @property
    def base_prior_sigma(self) -> torch.Tensor:
        r"""Get base prior variance (always positive). Only for gauge_fixed_priors=True.

        Hard clamp bounds output to [0.01, sigma_max].
        The E-step applies its own higher floor (0.1) to prevent gradient blowup
        from 1/σ_p terms; the embedding keeps [0.01, sigma_max] for sharp decode.
        """
        _SIGMA_MIN = 0.01
        # AMP guard: exp() on log-params needs float32
        with torch.amp.autocast('cuda', enabled=False):
            _p = self.base_log_prior_sigma
            sigma = torch.exp(_p if _p.dtype == torch.float32 else _p.float())
            return sigma.clamp(_SIGMA_MIN, self.sigma_max)

    @property
    def prior_sigma(self) -> torch.Tensor:
        r"""Get prior variances (always positive). Only for gauge_fixed_priors=False.

        Hard clamp bounds output to [0.01, sigma_max] for numerical safety:
        - min=0.01: allows sharp decode priors (1/σ_p up to 100).
          The E-step applies its own higher floor (0.1) for gradient safety.
        - max=sigma_max: configurable upper bound matching the belief covariance ceiling.

        Hard clamp zeros gradient at boundaries, preventing log_sigma drift.
        """
        _SIGMA_MIN = 0.01
        # AMP guard: exp() on log-params needs float32
        with torch.amp.autocast('cuda', enabled=False):
            _p = self.log_prior_sigma
            sigma = torch.exp(_p if _p.dtype == torch.float32 else _p.float())
            return sigma.clamp(_SIGMA_MIN, self.sigma_max)

    def _compute_gauge_transform(self, phi: torch.Tensor) -> torch.Tensor:
        r"""Compute gauge transform A = exp(φ · G) from gauge frame parameters.

        For GL(K): A ∈ GL⁺(K), a general invertible matrix (det > 0).
        For SO(N): A ∈ SO(N), an orthogonal matrix (A^T A = I).

        Uses stable_matrix_exp_pair for norm clamping and float64 upcasting
        on larger matrices, preventing overflow in the Padé scaling-squaring
        algorithm.

        Args:
            phi: (..., phi_dim) gauge frame parameters in Lie algebra

        Returns:
            A: (..., K, K) gauge transform matrices
        """
        # Compute φ · G = Σ_a φ_a G_a  ∈  g (Lie algebra element)
        # generators: (n_gen, K, K), phi: (..., phi_dim)
        phi_dot_G = torch.einsum('...a,aij->...ij', phi, self.generators)

        # Numerically stable matrix exponential with norm clamping
        A, _ = stable_matrix_exp_pair(phi_dot_G)
        return A

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

            # Compute gauge transform A_v = exp(φ_v · G) ∈ GL⁺(K)
            A = self._compute_gauge_transform(phi)  # (..., K, K)

            # Transport base prior mean: μ_v = A_v @ μ_0
            mu_p = torch.einsum('...ij,j->...i', A, self.base_prior_mu)

            # Transport base covariance: Σ_v = A_v @ diag(σ_0) @ A_v^T
            # This is the gauge-covariant sandwich product.
            # For GL(K), A_v is general invertible, so Σ_v is a full SPD matrix.
            # For SO(N), A_v is orthogonal, so Σ_v has same eigenvalues as Σ_0.
            # AMP guard: sandwich product must stay float32.
            # Avoid .float() copy when already float32 (saves ~725MB for VxKxK at K=60).
            with torch.amp.autocast('cuda', enabled=False):
                base_sigma = self.base_prior_sigma  # (K,) -- already float32 from property
                A_f = A if A.dtype == torch.float32 else A.float()
                if getattr(self, 'diagonal_covariance', True):
                    # Extract diagonal of A @ diag(σ) @ A^T:
                    #   diag(A Σ_0 A^T)_k = Σ_j A_kj^2 σ_j
                    A_sq = A_f ** 2  # (..., K, K)
                    sigma_p = torch.einsum('...kl,l->...k', A_sq, base_sigma)  # (..., K)
                else:
                    # Full covariance: Σ_v = A @ diag(σ_0) @ A^T (gauge-covariant)
                    Sigma_0 = torch.diag(base_sigma)  # (K, K)
                    sigma_p = torch.einsum('...ij,jk,...lk->...il', A_f, Sigma_0, A_f)  # (..., K, K)

            return mu_p, sigma_p, phi
        else:
            # Standard per-token lookup (TOKEN-INDEXED!)
            mu_p = self.prior_mu[token_ids]  # Index by token ID
            sigma_p = self.prior_sigma[token_ids]  # Always (B, N, K) diagonal
            # PriorBank priors are inherently diagonal (per-dimension variances).
            # Downstream modules handle diagonal->full conversion when needed:
            #   - _split_irreps_sigma: converts (B,N,K) -> (B,N,K,K) if diagonal_covariance=False
            #   - FFN sigma_prior: converts via diag_embed at variational_ffn.py:3627
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
        sigma_q: torch.Tensor,   # (B, N, K) or (B, N, K, K) belief covariances
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
           [σ_q + μ_q^2, -2μ_q] @ [1/σ_p, μ_p/σ_p]ᵀ  -> (B,N,2K) x (2K,V)
        2. Drop terms constant across V (cancel in softmax): -K, log|Σ_q|
        3. Combine prior-side constants into single (V,) bias

        When sigma_q is full covariance (B, N, K, K), extracts the diagonal
        variances for the fused matmul. When gauge_fixed_priors=True with
        diagonal_covariance=False, sigma_p from _get_prior_for_tokens is
        (V, K, K); the diagonal is extracted for the fused KL computation.

        Args:
            mu_q: (B, N, K) belief means
            sigma_q: (B, N, K) diagonal variances or (B, N, K, K) full covariance
            tau: Temperature for softmax

        Returns:
            logits: (B, N, vocab_size) log-probabilities (unnormalized)
        """
        B, N, K = mu_q.shape

        # Full covariance (B, N, K, K) -> extract diagonal variances (B, N, K).
        # The fused KL only needs tr(Σ_q Σ_p⁻¹) = Σ_k σ_q[k]/σ_p[k] when
        # priors are diagonal, so off-diagonal Σ_q entries don't contribute.
        if sigma_q.dim() == 4:
            sigma_q = torch.diagonal(sigma_q, dim1=-2, dim2=-1)  # (B, N, K)
        V = self.vocab_size
        device = mu_q.device

        # Get all token priors: mu_p (V, K), sigma_p (V, K) or (V, K, K)
        all_token_ids = torch.arange(V, device=device)
        _prior_out = self._get_prior_for_tokens(all_token_ids)
        mu_p, sigma_p = _prior_out[0], _prior_out[1]

        # AMP guard: entire decode KL uses division, log, and 1/sigma -- float32 required.
        # Avoid .float() copy when already float32 to prevent OOM at large K.
        with torch.amp.autocast('cuda', enabled=False):
            if mu_q.dtype != torch.float32:
                mu_q = mu_q.float()
                mu_p = mu_p.float()
                sigma_q = sigma_q.float()
                sigma_p = sigma_p.float()

            # Decode always uses diagonal prior covariances for the fused KL matmul.
            if sigma_p.dim() == 3:
                sigma_p = torch.diagonal(sigma_p, dim1=-2, dim2=-1)  # (V, K)

            variance_floor = max(self.eps, 1e-4)
            sigma_q_safe = sigma_q.clamp(min=variance_floor)    # (B, N, K)
            sigma_p_clamped = sigma_p.clamp(min=variance_floor)  # (V, K)

            # Scale sigma_p gradient from CE's precision (1/σ_p) terms.
            _s = self.sigma_ce_scale
            sigma_p_safe = sigma_p_clamped.detach() + _s * (sigma_p_clamped - sigma_p_clamped.detach())

            inv_sigma_p = 1.0 / sigma_p_safe                    # (V, K)
            mu_p_inv_sigma_p = mu_p * inv_sigma_p               # (V, K)

            # Fused matmul: trace + quad_q - cross in ONE operation
            lhs = torch.cat([sigma_q_safe + mu_q ** 2, -2.0 * mu_q], dim=-1)  # (B, N, 2K)
            rhs = torch.cat([inv_sigma_p, mu_p_inv_sigma_p], dim=-1)           # (V, 2K)

            combined = torch.matmul(lhs, rhs.T)                 # (B, N, V) -- single matmul

            # Prior-side bias (constant across batch): quad_p + log_det_p
            prior_bias = ((mu_p ** 2 * inv_sigma_p).sum(dim=-1)
                          + torch.log(sigma_p_safe).sum(dim=-1))  # (V,)

        # logits = -KL/τ ~= -0.5/τ * (combined + prior_bias)
        # (dropping softmax-invariant terms -K and log|Σ_q|)
        #
        # No dimension scaling here: init_std = √(ln V / K) already calibrates
        # the average KL to O(ln V) regardless of K. The 1/√K decay of logit
        # DIFFERENCES at high K is a convergence speed issue, not an init issue.
        # Scaling by √K would make the model catastrophically overconfident at
        # init (logit differences ~√K x ln V ~= 97 for K=80), causing CE >> ln V
        # when the E-step hasn't learned yet. Use learnable_temperature or tune
        # prior_bank_tau for large K instead.
        scale = torch.exp(self.decode_log_scale.clamp(-3.0, 3.0)) if self.learnable_temperature else 1.0
        logits = -0.5 * scale / tau * (combined + prior_bias.unsqueeze(0).unsqueeze(0))

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
            # Gauge-fixed M-step: update base prior (μ_0, σ_0) via de-rotation.
            #
            # Each belief μ_q at token v was generated from prior μ_v = A_v @ μ_0.
            # The "de-rotated" belief target is A_v⁻¹ @ μ_q, which should ~= μ_0
            # if the prior is well-calibrated. We EMA toward the weighted average
            # of de-rotated beliefs across the batch.
            #
            # φ_v updates are handled by backprop through the VFE loss -- the
            # gradient ∂F/∂φ_v flows through A_v = exp(φ_v · G) in the forward
            # pass. The EMA update here handles the base prior only.
            self._update_gauge_fixed_base_prior(
                token_ids, mu_beliefs, sigma_beliefs, prediction_errors, lr
            )
            return

        with torch.no_grad():
            # Full covariance (B, N, K, K) -> diagonal variances (B, N, K)
            if sigma_beliefs.dim() == 4:
                sigma_beliefs = torch.diagonal(sigma_beliefs, dim1=-2, dim2=-1)

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

    def _update_gauge_fixed_base_prior(
        self,
        token_ids: torch.Tensor,
        mu_beliefs: torch.Tensor,
        sigma_beliefs: torch.Tensor,
        prediction_errors: torch.Tensor,
        lr: float,
    ):
        r"""Update base prior (μ_0, σ_0) via de-rotation of evolved beliefs.

        For gauge-fixed priors, each token's prior is μ_v = A_v @ μ_0 where
        A_v = exp(φ_v · G). Given evolved beliefs μ_q at token v, the
        de-rotated target for the base prior is:

            μ_0^{target} = A_v⁻¹ @ μ_q

        We compute prediction-error-weighted averages of these de-rotated
        beliefs and EMA-update the base prior toward them.

        Args:
            token_ids: (B, N) token IDs
            mu_beliefs: (B, N, K) evolved belief means
            sigma_beliefs: (B, N, K) or (B, N, K, K) belief variances
            prediction_errors: (B, N) per-position CE loss
            lr: Learning rate for EMA update
        """
        with torch.no_grad():
            if sigma_beliefs.dim() == 4:
                sigma_beliefs = torch.diagonal(sigma_beliefs, dim1=-2, dim2=-1)

            B, N, K = mu_beliefs.shape

            # Get gauge transforms for tokens in the batch
            phi = self.phi_embed(token_ids)  # (B, N, phi_dim)
            A = self._compute_gauge_transform(phi)  # (B, N, K, K)

            # Compute A_v⁻¹ via solving the linear system (more stable than .inverse())
            # A_v⁻¹ @ μ_q = solve(A_v, μ_q)
            mu_flat = mu_beliefs.reshape(-1, K, 1)  # (B*N, K, 1)
            A_flat = A.reshape(-1, K, K)  # (B*N, K, K)
            derotated_mu = torch.linalg.solve(A_flat, mu_flat).squeeze(-1)  # (B*N, K)

            # Similarly de-rotate sigma: diag(A⁻¹ diag(σ_q) A⁻ᵀ)
            # For diagonal sigma_q, the de-rotated diagonal is:
            #   diag(A⁻¹ diag(σ_q) A⁻ᵀ)_k = Σ_j (A⁻¹)_kj^2 σ_q_j
            A_inv = torch.linalg.solve(
                A_flat, torch.eye(K, device=A_flat.device).expand_as(A_flat)
            )  # (B*N, K, K)
            sigma_flat = sigma_beliefs.reshape(-1, K)  # (B*N, K)
            derotated_sigma = torch.einsum(
                'bkj,bj->bk', A_inv ** 2, sigma_flat
            )  # (B*N, K)

            # Prediction-error weighted aggregation (same logic as non-gauge-fixed path)
            flat_errors = prediction_errors.reshape(-1)
            neg_errors = -flat_errors.clamp(min=-10, max=10)

            # Global softmax over all positions (all contribute to base prior)
            weights = torch.softmax(neg_errors, dim=0)  # (B*N,)

            # Weighted mean of de-rotated beliefs
            weighted_mu = (derotated_mu * weights.unsqueeze(-1)).sum(dim=0)  # (K,)
            weighted_sigma = (derotated_sigma * weights.unsqueeze(-1)).sum(dim=0)  # (K,)

            # Confidence-weighted learning rate
            mean_error = flat_errors.mean()
            confidence = 1.0 / (1.0 + mean_error)
            effective_lr = lr * confidence

            # EMA update for base prior mean
            self.base_prior_mu.data.lerp_(weighted_mu, effective_lr)

            # EMA update for base prior sigma
            if self.learnable_sigma:
                current_sigma = torch.exp(self.base_log_prior_sigma.data)
                sigma_lr = effective_lr * 0.1  # Slower sigma updates
                new_sigma = current_sigma.lerp(weighted_sigma, sigma_lr)
                self.base_log_prior_sigma.data.copy_(
                    torch.log(new_sigma.clamp(min=self.eps))
                )

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
            f"gauge_fixed_priors={self.gauge_fixed_priors}, "
            f"learnable_temperature={self.learnable_temperature}"
        )
