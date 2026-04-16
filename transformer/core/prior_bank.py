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
from typing import List, Tuple, Optional, Union
from transformer.core.gauge_utils import stable_matrix_exp_pair, fused_block_matrix_exp_pairs


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
        restrictive — it reparameterizes the full space through (base + frame).

        This guarantees gauge covariance: π_i = Ω_ij[π_j] for all i,j
        The model learns:
        - base_prior_mu (K,): shared base prior mean
        - base_log_prior_sigma (K,): shared base prior log-variance
        - phi_embed (V, phi_dim): per-token gauge frames (phi_dim = K² for GL(K))

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
        phi_dim: int = None,  # Auto-inferred from generators; K² for GL(K), 3 for SO(3)
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
        irrep_dims: Optional[List[int]] = None,  # Per-head block dims for block-diagonal matrix_exp
        # Decode-prior caching: when True, the all-vocab matrix_exp output is
        # computed once at the first decode call of each forward pass and
        # reused across subsequent calls (active inference, multi-layer aux,
        # distillation).  Replaces the gradient checkpoint used previously.
        # See decode() for the trade-off discussion.
        cache_decode_priors: bool = False,
        # Exact diagonal transport: when True + diagonal_covariance, lift
        # diagonal σ to full (B,N,K,K) for decode, matching the exact
        # transport used in attention and E-step.
        exact_diagonal_transport: bool = False,
        # Full-covariance decode: explicit override. When True, always use
        # exact full-cov KL in decode regardless of other flags.
        # O(B·N·V·K²) vs O(B·N·V·K) diagonal.
        full_cov_decode: bool = False,
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
                For GL(K): K² generators spanning gl(K). For SO(N): N(N-1)/2 generators.
            phi_dim: Dimension of gauge frame. Auto-inferred from generators.shape[0]
                if not provided. K² for GL(K), 3 for SO(3), N(N-1)/2 for SO(N).
            sigma_ce_scale: Fraction of CE gradient passed to sigma_p (0.01 recommended)
            learnable_temperature: If True, learn a decode scale factor via backprop.
                Starts at scale=1 (CE ≈ ln V at init). Gradient is self-regulating:
                dCE/d(log_scale) = scale·[E_p[KL] - KL_target], which decreases scale
                when overconfident on wrong tokens and increases it as predictions improve.
            full_cov_decode: If True, decode uses exact full-covariance KL when
                sigma_q is (B,N,K,K). O(B·N·V·K²) vs O(B·N·V·K) diagonal.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.eps = eps
        self.learnable_sigma = learnable_sigma
        self.gauge_fixed_priors = gauge_fixed_priors
        # Auto-infer phi_dim from generators if not provided
        if phi_dim is None:
            if generators is not None:
                phi_dim = generators.shape[0]
            else:
                phi_dim = 3
                import logging
                logging.getLogger(__name__).warning(
                    "PriorBank: phi_dim not provided and no generators available. "
                    "Defaulting to phi_dim=3 (SO(3)). Pass phi_dim explicitly to suppress."
                )
        self.gauge_param = gauge_param
        self.omega_head_dims = omega_head_dims
        self.sigma_ce_scale = sigma_ce_scale
        self.sigma_max = sigma_max
        self.learnable_temperature = learnable_temperature
        self.diagonal_covariance = diagonal_covariance
        self.irrep_dims = irrep_dims
        self.cache_decode_priors = cache_decode_priors
        self.exact_diagonal_transport = exact_diagonal_transport
        self.full_cov_decode = full_cov_decode

        # Per-forward-pass cache for the all-vocab decode prior result.
        # Cleared by the model at the start of each forward via
        # ``clear_decode_cache()``.  Holds the autograd graph of the
        # matrix_exp output, so backward through any consumer routes
        # gradients back to phi_embed.  Stored as a regular Python attr
        # (not a buffer) so it does not appear in state_dict.
        self._decode_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        # Learnable inverse-temperature for decode logits.
        # At init, scale = exp(0) = 1.0 → no amplification → CE ≈ ln(V).
        # The model discovers the right K-dependent scale during training.
        if learnable_temperature:
            self.decode_log_scale = nn.Parameter(torch.tensor(0.0))

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
            # Cache skew-symmetry flag for block_exp_pairs (SO(K): exp(-M)=exp(M)^T)
            self._generators_are_skew = torch.allclose(
                generators + generators.transpose(-1, -2),
                torch.zeros_like(generators), atol=1e-5,
            )

            # Note: decode() uses gradient checkpointing for gauge_fixed_priors=True
            # to avoid retaining V×K×K autograd intermediates in memory.

            # Single base prior mean μ_0 — all token priors are gauge transforms:
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
            # MUST use random init — zero init makes ALL tokens identical
            # (exp(0) = I → μ_v = μ_0 for all v). Scale inversely with
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

    def clear_decode_cache(self) -> None:
        """Drop the cached all-vocab decode priors.

        Called by the model at the start of each forward pass when
        ``cache_decode_priors=True``.  After backward + optimizer.step()
        the matrix_exp graph is stale and the autograd intermediates have
        been freed; the next forward must recompute.  No-op when caching
        is disabled.
        """
        self._decode_cache = None

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

    def _compute_gauge_transform(
        self,
        phi: torch.Tensor,
        only_forward: bool = False,
    ) -> torch.Tensor:
        r"""Compute gauge transform A = exp(φ · G) from gauge frame parameters.

        For GL(K): A ∈ GL⁺(K), a general invertible matrix (det > 0).
        For SO(N): A ∈ SO(N), an orthogonal matrix (A^T A = I).

        Uses stable_matrix_exp_pair for norm clamping and float64 upcasting
        on larger matrices, preventing overflow in the Padé scaling-squaring
        algorithm.

        Args:
            phi: (..., phi_dim) gauge frame parameters in Lie algebra
            only_forward: If True, skip computing exp(-φ·G). Saves one
                matrix_exp call for GL(K). Use in decode where only A
                (not A⁻¹) is needed.

        Returns:
            A: (..., K, K) gauge transform matrices
        """
        # Compute φ · G = Σ_a φ_a G_a  ∈  g (Lie algebra element)
        # generators: (n_gen, K, K), phi: (..., phi_dim)
        phi_dot_G = torch.einsum('...a,aij->...ij', phi, self.generators)

        # Numerically stable matrix exponential with norm clamping
        A, _ = stable_matrix_exp_pair(phi_dot_G, only_forward=only_forward)
        return A

    def _get_prior_for_tokens(
        self,
        token_ids: torch.Tensor,  # (B, N) or (V,) for all vocab
        only_forward: bool = False,  # True for decode (no A⁻¹ needed)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get prior (μ, σ, φ) for given tokens.

        CRITICAL: Indexing by TOKEN ID, not position!

        Args:
            token_ids: Token IDs to look up.
            only_forward: If True, skip computing exp(-φ·G) in matrix_exp.
                Use True for decode (only needs A, not A⁻¹). Saves one
                matrix_exp call for GL(K).

        Returns:
            mu_p: prior means
            sigma_p: prior variances (diagonal)
            phi_p: gauge frames
        """
        if self.gauge_fixed_priors:
            # Get per-token gauge frames
            phi = self.phi_embed(token_ids)  # (..., phi_dim)
            K = self.embed_dim

            # AMP guard: sandwich product must stay float32.
            with torch.amp.autocast('cuda', enabled=False):
                base_sigma = self.base_prior_sigma  # (K,) — already float32 from property

            if self.irrep_dims is not None:
                # BLOCK-DIAGONAL PATH: Exploit generator structure for O(d_h³)
                # instead of O(K³) per matrix_exp. For K=60 with 6×10 heads:
                # 6 blocks of 10×10 in float32 vs one 60×60 in float64.
                # Memory: ~120 MB vs ~1.5 GB for V=50k decode.
                _skew = getattr(self, '_generators_are_skew', False)
                # fused_block_matrix_exp_pairs expects (B, N, n_gen) but decode
                # passes (V, phi_dim). Unsqueeze to (1, V, phi_dim) then squeeze.
                _phi_3d = phi if phi.dim() == 3 else phi.unsqueeze(0)
                block_exp_pairs = fused_block_matrix_exp_pairs(
                    _phi_3d, self.generators, self.irrep_dims,
                    skew_symmetric=_skew,
                    only_forward=only_forward,
                )
                # Squeeze back if we added a batch dim
                if phi.dim() == 2:
                    block_exp_pairs = [
                        (bep[0].squeeze(0), bep[1].squeeze(0) if bep[1] is not None else None)
                        for bep in block_exp_pairs
                    ]
                # Compute mu_p and sigma_p per-block
                mu_parts = []
                sigma_parts = []
                block_start = 0
                for h, d_h in enumerate(self.irrep_dims):
                    block_end = block_start + d_h
                    exp_h = block_exp_pairs[h][0]  # (..., d_h, d_h)
                    base_mu_h = self.base_prior_mu[block_start:block_end]
                    mu_parts.append(torch.einsum('...ij,j->...i', exp_h, base_mu_h))

                    with torch.amp.autocast('cuda', enabled=False):
                        base_sigma_h = base_sigma[block_start:block_end]
                        exp_h_f = exp_h if exp_h.dtype == torch.float32 else exp_h.float()
                        if getattr(self, 'diagonal_covariance', True):
                            sigma_parts.append(
                                torch.einsum('...kl,l->...k', exp_h_f ** 2, base_sigma_h)
                            )
                        else:
                            Sigma_0h = torch.diag(base_sigma_h)
                            sigma_parts.append(
                                torch.einsum('...ij,jk,...lk->...il', exp_h_f, Sigma_0h, exp_h_f)
                            )
                    block_start = block_end

                mu_p = torch.cat(mu_parts, dim=-1)
                if getattr(self, 'diagonal_covariance', True):
                    sigma_p = torch.cat(sigma_parts, dim=-1)
                else:
                    sigma_p = torch.zeros(*phi.shape[:-1], K, K,
                                          device=phi.device, dtype=sigma_parts[0].dtype)
                    block_start = 0
                    for h, d_h in enumerate(self.irrep_dims):
                        block_end = block_start + d_h
                        sigma_p[..., block_start:block_end, block_start:block_end] = sigma_parts[h]
                        block_start = block_end
            else:
                # FULL K×K FALLBACK: no block structure available
                A = self._compute_gauge_transform(phi, only_forward=only_forward)

                # Transport base prior mean: μ_v = A_v @ μ_0
                mu_p = torch.einsum('...ij,j->...i', A, self.base_prior_mu)

                # Transport base covariance: Σ_v = A_v @ diag(σ_0) @ A_v^T
                with torch.amp.autocast('cuda', enabled=False):
                    A_f = A if A.dtype == torch.float32 else A.float()
                    if getattr(self, 'diagonal_covariance', True):
                        A_sq = A_f ** 2
                        sigma_p = torch.einsum('...kl,l->...k', A_sq, base_sigma)
                    else:
                        Sigma_0 = torch.diag(base_sigma)
                        sigma_p = torch.einsum('...ij,jk,...lk->...il', A_f, Sigma_0, A_f)

            return mu_p, sigma_p, phi
        else:
            # Standard per-token lookup (TOKEN-INDEXED!)
            mu_p = self.prior_mu[token_ids]  # Index by token ID
            sigma_p = self.prior_sigma[token_ids]  # Always (B, N, K) diagonal
            # PriorBank priors are inherently diagonal (per-dimension variances).
            # Downstream modules handle diagonal→full conversion when needed:
            #   - _split_irreps_sigma: converts (B,N,K) → (B,N,K,K) if diagonal_covariance=False
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
        r"""Compute observation likelihood via KL to all token priors.

        p(y = v | q) \propto \exp(-\mathrm{KL}(q \| \pi_v) / \tau)

        Auto-dispatches between diagonal-KL and full-cov KL based on the
        model's covariance configuration:

        - ``diagonal_covariance=False``: exact full-cov KL, O(B·N·V·K²)
        - ``exact_diagonal_transport=True``: lifts diagonal to full, O(B·N·V·K²)
        - Otherwise: diagonal-KL via fused matmul, O(B·N·V·K)

        Fused single-matmul implementation. The diagonal KL is:

            KL(q || π_v) = 0.5 * [tr(Σ_q/Σ_p) + (μ_q-μ_p)ᵀΣ_p⁻¹(μ_q-μ_p)
                                   - K + log|Σ_p|/|Σ_q|]

        Key optimizations:
        1. Fuse trace + quad_q - cross into ONE matmul via concatenation:
           [σ_q + μ_q², -2μ_q] @ [1/σ_p, μ_p/σ_p]ᵀ  → (B,N,2K) x (2K,V)
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

        # Auto-dispatch: use full-cov decode when the pipeline uses full
        # covariance geometry.  This ensures the observation term matches
        # the KL geometry used in attention and the E-step.
        _use_full_cov = (
            self.full_cov_decode                          # explicit override
            or (not self.diagonal_covariance              # full-cov model
                and sigma_q.dim() == 4)
            or (self.exact_diagonal_transport             # exact diag transport
                and self.diagonal_covariance
                and sigma_q.dim() == 3)
        )
        if _use_full_cov:
            if sigma_q.dim() == 3:
                sigma_q = torch.diag_embed(sigma_q)       # (B,N,K) → (B,N,K,K)
            return self._decode_full_cov(mu_q, sigma_q, tau)

        # Diagonal path: extract diagonal variances if full covariance provided
        if sigma_q.dim() == 4:
            sigma_q = torch.diagonal(sigma_q, dim1=-2, dim2=-1)  # (B, N, K)
        V = self.vocab_size
        device = mu_q.device

        # Get all token priors: mu_p (V, K), sigma_p (V, K).
        #
        # Three paths:
        #
        # 1. cache_decode_priors=True: compute matrix_exp(phi_v · G) for the
        #    full vocab once on the first decode call of this forward pass,
        #    cache the (mu_p, sigma_p) tensors *with their autograd graph*,
        #    and reuse on subsequent calls.  All consumers route their
        #    gradients back through the same matrix_exp graph at backward.
        #    Replaces the gradient checkpoint.  Memory cost: ~80–200 MB extra
        #    for the matrix_exp saved tensors at V=50k, K=20.  Hugely
        #    beneficial when active_inference / aux_layer_loss / distillation
        #    drive multiple decode calls per forward (one matrix_exp instead
        #    of N).  For single-decode workflows it still wins by ~15% by
        #    skipping the gradient-checkpoint recompute on backward.
        #
        # 2. gauge_fixed_priors=True + training + cache_decode_priors=False:
        #    legacy gradient-checkpoint path.  Memory-cheapest but pays the
        #    full matrix_exp cost on every decode call AND a recompute on
        #    backward.  Default for backwards compatibility.
        #
        # 3. eval mode or gauge_fixed_priors=False: inline computation, no
        #    checkpoint, no cache (no autograd cost in eval; non-gauge-fixed
        #    is just a parameter lookup).
        #
        # only_forward=True: decode never needs exp(-φ·G), saving one
        # matrix_exp per token for GL(K).
        all_token_ids = torch.arange(V, device=device)
        if (self.cache_decode_priors
                and self._decode_cache is not None
                and self.gauge_fixed_priors):
            # Cache hit — reuse the (mu_p, sigma_p) computed earlier in this
            # forward pass.  The autograd graph is shared.
            mu_p, sigma_p = self._decode_cache
        elif (self.gauge_fixed_priors
              and self.training
              and not self.cache_decode_priors):
            # Legacy gradient-checkpoint path.  Lambda wrapper avoids
            # passing only_forward through checkpoint's **kwargs.
            _prior_out = torch.utils.checkpoint.checkpoint(
                lambda ids: self._get_prior_for_tokens(ids, only_forward=True),
                all_token_ids,
                use_reentrant=False,
            )
            mu_p, sigma_p = _prior_out[0], _prior_out[1]
        else:
            _prior_out = self._get_prior_for_tokens(all_token_ids, only_forward=True)
            mu_p, sigma_p = _prior_out[0], _prior_out[1]
            # Populate the cache for the rest of this forward pass.
            if self.cache_decode_priors and self.gauge_fixed_priors:
                self._decode_cache = (mu_p, sigma_p)

        # AMP guard: entire decode KL uses division, log, and 1/sigma — float32 required.
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

            combined = torch.matmul(lhs, rhs.T)                 # (B, N, V) — single matmul

            # Prior-side bias (constant across batch): quad_p + log_det_p
            prior_bias = ((mu_p ** 2 * inv_sigma_p).sum(dim=-1)
                          + torch.log(sigma_p_safe).sum(dim=-1))  # (V,)

        # logits = -KL/τ ≈ -0.5/τ * (combined + prior_bias)
        # (dropping softmax-invariant terms -K and log|Σ_q|)
        #
        # No dimension scaling here: init_std = √(ln V / K) already calibrates
        # the average KL to O(ln V) regardless of K. The 1/√K decay of logit
        # DIFFERENCES at high K is a convergence speed issue, not an init issue.
        # Scaling by √K would make the model catastrophically overconfident at
        # init (logit differences ~√K × ln V ≈ 97 for K=80), causing CE >> ln V
        # when the E-step hasn't learned yet. Use learnable_temperature or tune
        # prior_bank_tau for large K instead.
        scale = torch.exp(self.decode_log_scale.clamp(-3.0, 3.0)) if self.learnable_temperature else 1.0
        logits = -0.5 * scale / tau * (combined + prior_bias.unsqueeze(0).unsqueeze(0))

        return logits

    def _decode_full_cov(
        self,
        mu_q: torch.Tensor,      # (B, N, K) belief means
        sigma_q: torch.Tensor,   # (B, N, K, K) full belief covariance
        tau: float = 1.0,
    ) -> torch.Tensor:
        r"""Exact full-covariance KL decode.

        Computes logits ∝ -KL(q || π_v) / τ using the full Gaussian KL:

            KL(q || π_v) = 0.5 [tr(Σ_p^{-1} Σ_q)
                               + (μ_q - μ_p)^T Σ_p^{-1} (μ_q - μ_p)
                               - K + log|Σ_p| - log|Σ_q|]

        Key identity: tr(Σ_p^{-1} Σ_q) + μ_q^T Σ_p^{-1} μ_q = tr(Σ_p^{-1} M)
        where M = Σ_q + μ_q μ_q^T is the second moment. This fuses the trace
        and quadratic-in-μ_q terms into a single (B·N, K²) × (K², V) matmul.

        Cost: O(B·N·V·K²) for the second-moment matmul, vs O(B·N·V·K) for
        the diagonal path. The K²/K slowdown is the price of off-diagonal
        fidelity.

        Args:
            mu_q: (B, N, K) belief means
            sigma_q: (B, N, K, K) full belief covariance
            tau: Temperature for softmax

        Returns:
            logits: (B, N, vocab_size) unnormalized log-probabilities
        """
        B, N, K = mu_q.shape
        V = self.vocab_size
        device = mu_q.device

        # Retrieve priors (same caching logic as diagonal path)
        all_token_ids = torch.arange(V, device=device)
        if (self.cache_decode_priors
                and self._decode_cache is not None
                and self.gauge_fixed_priors):
            mu_p, sigma_p = self._decode_cache
        elif (self.gauge_fixed_priors
              and self.training
              and not self.cache_decode_priors):
            _prior_out = torch.utils.checkpoint.checkpoint(
                lambda ids: self._get_prior_for_tokens(ids, only_forward=True),
                all_token_ids,
                use_reentrant=False,
            )
            mu_p, sigma_p = _prior_out[0], _prior_out[1]
        else:
            _prior_out = self._get_prior_for_tokens(all_token_ids, only_forward=True)
            mu_p, sigma_p = _prior_out[0], _prior_out[1]
            if self.cache_decode_priors and self.gauge_fixed_priors:
                self._decode_cache = (mu_p, sigma_p)

        # AMP guard: all KL arithmetic in float32
        with torch.amp.autocast('cuda', enabled=False):
            if mu_q.dtype != torch.float32:
                mu_q = mu_q.float()
                mu_p = mu_p.float()
                sigma_q = sigma_q.float()
                sigma_p = sigma_p.float()

            variance_floor = max(self.eps, 1e-4)

            # Build Σ_p^{-1} and log|Σ_p| for all V tokens
            if sigma_p.dim() == 2:
                # Diagonal prior (V, K) → promote to full for uniform codepath
                sigma_p_safe = sigma_p.clamp(min=variance_floor)
                _s = self.sigma_ce_scale
                sigma_p_safe = sigma_p_safe.detach() + _s * (sigma_p_safe - sigma_p_safe.detach())
                Sigma_p_inv = torch.diag_embed(1.0 / sigma_p_safe)    # (V, K, K)
                log_det_p = torch.log(sigma_p_safe).sum(dim=-1)        # (V,)
            else:
                # Full prior (V, K, K) — Cholesky inversion
                sigma_p_clamped = sigma_p + variance_floor * torch.eye(K, device=device, dtype=sigma_p.dtype)
                _s = self.sigma_ce_scale
                sigma_p_safe = sigma_p_clamped.detach() + _s * (sigma_p_clamped - sigma_p_clamped.detach())
                L_p = torch.linalg.cholesky(sigma_p_safe)              # (V, K, K)
                Sigma_p_inv = torch.cholesky_inverse(L_p)              # (V, K, K)
                log_det_p = 2.0 * L_p.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)  # (V,)

            # Second moment M = Σ_q + μ_q μ_q^T  →  (B, N, K, K)
            M = sigma_q + mu_q.unsqueeze(-1) * mu_q.unsqueeze(-2)

            # Fused matmul: tr(Σ_p^{-1} M) for all (b,n,v) pairs
            # Reshape to 2D and use a single BLAS call: (B*N, K²) @ (K², V) → (B*N, V)
            M_flat = M.reshape(B * N, K * K)                           # (B*N, K²)
            Sigma_p_inv_flat = Sigma_p_inv.reshape(V, K * K)           # (V, K²)
            combined = torch.matmul(M_flat, Sigma_p_inv_flat.T).reshape(B, N, V)

            # Cross term: -2 μ_q^T (Σ_p^{-1} μ_p)
            Sigma_p_inv_mu_p = torch.einsum('vkl,vl->vk', Sigma_p_inv, mu_p)  # (V, K)
            cross = torch.matmul(
                mu_q.reshape(B * N, K), Sigma_p_inv_mu_p.T
            ).reshape(B, N, V)                                         # (B, N, V)

            # Prior quadratic: μ_p^T Σ_p^{-1} μ_p  →  (V,)
            quad_p = (mu_p * Sigma_p_inv_mu_p).sum(dim=-1)

            # Prior-side bias: quad_p + log_det_p  →  (V,)
            prior_bias = quad_p + log_det_p

        # logits = -KL/τ = -0.5/τ * (tr(Σ_p^{-1} M) - 2 cross + quad_p + log_det_p)
        # Drops softmax-invariant terms: -K and -log|Σ_q|
        scale = torch.exp(self.decode_log_scale.clamp(-3.0, 3.0)) if self.learnable_temperature else 1.0
        logits = -0.5 * scale / tau * (combined - 2.0 * cross + prior_bias.unsqueeze(0).unsqueeze(0))

        return logits

    # =========================================================================
    # M-STEP via prediction-error weighted EMA (Hebbian P-flow)
    # =========================================================================
    # Implementations live in ``transformer/core/hebbian.py``.  These
    # methods are thin delegators kept for backward compatibility with
    # external callers (e.g. ``GaugeTransformerLM.p_flow_update``).
    def update_from_beliefs(
        self,
        token_ids: torch.Tensor,
        mu_beliefs: torch.Tensor,
        sigma_beliefs: torch.Tensor,
        prediction_errors: torch.Tensor,
        lr: float = 0.01,
    ):
        """Thin delegator — see ``hebbian.update_prior_bank_from_beliefs``."""
        from transformer.core.hebbian import update_prior_bank_from_beliefs
        return update_prior_bank_from_beliefs(
            self, token_ids, mu_beliefs, sigma_beliefs, prediction_errors, lr=lr,
        )

    def _update_gauge_fixed_base_prior(
        self,
        token_ids: torch.Tensor,
        mu_beliefs: torch.Tensor,
        sigma_beliefs: torch.Tensor,
        prediction_errors: torch.Tensor,
        lr: float,
    ):
        """Thin delegator — see ``hebbian.update_gauge_fixed_base_prior``."""
        from transformer.core.hebbian import update_gauge_fixed_base_prior
        return update_gauge_fixed_base_prior(
            self, token_ids, mu_beliefs, sigma_beliefs, prediction_errors, lr,
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
