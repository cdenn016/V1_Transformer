# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 12:34:45 2025

@author: chris and christine
"""

"""
Gauge-Theoretic Token Embeddings (0D Transformer)
==================================================

Maps discrete tokens → agent beliefs (μ_i, Σ_i, φ_i) at single base manifold point c*.

Key Insight from plan.py:
    "0D Transformer: All N tokens → N agents at the SAME base point c*
     Each token i → (μ_i, Σ_i, φ_i) where:
     - μ_i ∈ ℝ^K: mean belief vector (NO spatial dependence)
     - Σ_i ∈ SPD(K): covariance (scalar matrix per agent)
     - φ_i ∈ gl(K): gauge frame (Lie algebra element)"

GL(K) Gauge Structure (NEW):
    The VFE is invariant under GL(K) gauge transformations, not just SO(K)!
    This is because f-divergences are invariant under pushforward:
        D_KL(Ω·P || Ω·Q) = D_KL(P || Q) for any Ω ∈ GL(K)

    Parameterization options:
    - phi_dim=3: so(3) subalgebra (3 generators, rotation-only)
    - phi_dim=K(K-1)/2: so(K) subalgebra (skew-symmetric, orthogonal)
    - phi_dim=K²: gl(K) full algebra (all K×K matrices, maximum flexibility)

Author: Implementation from plan.py
Date: November 2025
Updated: GL(K) generalization - February 2026
"""

import math
import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Optional
from transformer.core.gauge_utils import stable_matrix_exp_pair

# Import Lie algebra composition functions
try:
    from math_utils.generators import soN_compose_bch_torch
    SON_BCH_AVAILABLE = True
except ImportError:
    SON_BCH_AVAILABLE = False

try:
    from math_utils.generators import lie_compose_bch_general_torch
    GENERAL_BCH_AVAILABLE = True
except ImportError:
    GENERAL_BCH_AVAILABLE = False


class GaugeTokenEmbedding(nn.Module):
    """
    Map discrete tokens to gauge-equivariant agent beliefs at single point.

    0D Transformer: All N tokens → N agents at the SAME base point c*
    Each token i → (μ_i, Σ_i, φ_i) where:
    - μ_i ∈ ℝ^K: mean belief vector (NO spatial dependence)
    - Σ_i ∈ SPD(K): covariance (scalar matrix per agent)
    - φ_i ∈ gl(K): gauge frame (Lie algebra element)

    GL(K) Gauge Structure:
        The gauge group can be SO(K) (orthogonal) or GL(K) (general linear).
        For VFE-based attention, GL(K) is sufficient because f-divergences
        are invariant under all invertible linear transformations.

        Parameterization:
        - phi_dim=3: so(3) subalgebra (rotation-only, legacy)
        - phi_dim=K(K-1)/2: so(K) subalgebra (orthogonal transformations)
        - phi_dim=K²: gl(K) full algebra (maximum flexibility)

    Architecture:
        token_id → [Embedding Layer] → (μ, Σ, φ)

        where:
        - μ: Learnable embedding (standard)
        - Σ: Initialized to small isotropic (σ²I), optionally learnable
        - φ: Initialized near zero (near-identity gauge frame)
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        irrep_spec: list = None,
        init_std: float = None,  # Default: 2.0 for sharper attention
        init_sigma_scale: float = 1.0,  # Scaled to match init_std for O(1) KL
        learnable_sigma: bool = False,
        learnable_phi: bool = False,
        gauge_fixed_priors: bool = False,
        generators: Optional[torch.Tensor] = None,
        diagonal_covariance: bool = False,
        isotropic_covariance: bool = False,  # If True, force Σ = σ²I (scalar variance × identity)
                                             # Note: the "KL → squared Euclidean" simplification
                                             # requires orthogonal transport (Ω ∈ O(K)) so that
                                             # transported Σ stays isotropic. Use enforce_orthogonal=True
                                             # or learnable_reflection=True with SO(K) generators.
                                             # With GL(K), transported cov is NOT isotropic (S(Ω) ≠ 0).
        max_seq_len: int = 2048,
        use_positional_embedding: bool = False,
        phi_dim: int = 3,  # 3 for SO(3), N(N-1)/2 for SO(N)
        phi_scale: float = 0.3,  # Target ||φ|| norm for gauge frame initialization
        # Direct Omega parameterization (alternative to phi)
        gauge_param: str = 'phi',  # 'phi' (Lie algebra) or 'omega' (direct GL(K) matrices)
        omega_head_dims: Optional[List[int]] = None,  # Per-head dims [K_h1, K_h2, ...] for omega
        # Mean embedding normalization options
        mu_normalize: bool = False,  # If True, project μ to unit sphere
        mu_max_norm: Optional[float] = None,  # If set, clamp ||μ|| ≤ max_norm
        # O(K) reflection parameters
        learnable_reflection: bool = False,  # If True, learn per-token sign vectors s_i ∈ {±1}^K
                                             # extending SO(K) gauge transport to full O(K).
                                             # Ω_ij = diag(s_i)·exp(φ_i)·exp(-φ_j)·diag(s_j)
                                             # The reflection is applied as μ_i → s_i ⊙ μ_i
                                             # before the SO(K) rotation, so no changes needed
                                             # in attention or VFE code.
        sigma_max: float = 5.0,  # Upper bound for prior covariance clamp
    ):
        """
        Initialize gauge token embedding.

        Args:
            vocab_size: Number of tokens in vocabulary
            embed_dim: Embedding dimension K (fiber dimension)
            irrep_spec: List of (label, multiplicity, dim) for SO(3)/SO(N) irreps
            init_std: Std dev for initializing mean embeddings
            init_sigma_scale: Initial scale for covariance (σ in σ²I)
            learnable_sigma: If True, Σ evolves during training
            learnable_phi: If True, φ evolves during training
            gauge_fixed_priors: If True, priors are defined as GL(K) transformations of a
                               single base prior: p_i = G_i ▷ p_0. This guarantees
                               gauge covariance: p_i = Ω_ij[p_j] where Ω_ij = G_i G_j^{-1}.
                               Requires generators for computing transformations.
            generators: Lie algebra generators (n_gen, K, K), required if gauge_fixed_priors=True
            diagonal_covariance: If True, output sigma as (B,N,K) diagonal variances
                                instead of (B,N,K,K) full matrices. Saves O(K) memory.
            max_seq_len: Maximum sequence length for positional embeddings
            use_positional_embedding: If True, add learnable positional embeddings to μ
                                     (like standard transformers). This provides position
                                     info in the content while keeping gauge transport Ω_ij.
            phi_dim: Dimension of gauge frame φ. Options:
                    - 3: so(3) subalgebra (rotation-only, legacy)
                    - K(K-1)/2: so(K) subalgebra (orthogonal transformations)
                    - K²: gl(K) full algebra (maximum flexibility for GL(K) gauge)
            phi_scale: Target ||φ|| norm for gauge frame initialization. Higher values
                      (e.g., 1.0-2.0) encourage semantic clustering in gauge frames.
            isotropic_covariance: If True, force Σ = σ²I (scalar variance × identity).
                Simplifies KL to squared Euclidean but requires orthogonal transport
                (Ω ∈ O(K)) to preserve isotropy. With GL(K), transported cov is NOT isotropic.
            mu_normalize: If True, project μ to unit sphere after embedding lookup.
            mu_max_norm: If set, clamp ||μ|| ≤ max_norm after embedding lookup.
            learnable_reflection: If True, learn per-token sign vectors s_i ∈ {±1}^K
                extending SO(K) gauge transport to full O(K). Applied as μ_i → s_i ⊙ μ_i
                before rotation, so no changes needed in attention or VFE code.
        """
        super().__init__()
        self.phi_scale = phi_scale
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.irrep_spec = irrep_spec
        self.learnable_sigma = learnable_sigma
        self.learnable_phi = learnable_phi
        self.gauge_fixed_priors = gauge_fixed_priors
        self.learnable_reflection = learnable_reflection
        self.sigma_max = sigma_max

        # Embedding initialization scale
        # OLD: 1/sqrt(K) keeps ||μ||² = O(1) but makes all embeddings equidistant!
        # NEW: Larger init_std creates more variance in pairwise distances,
        #      enabling sharper KL-based attention from the start.
        if init_std is None:
            init_std = 2.0  # Was: 1.0 / np.sqrt(embed_dim) ≈ 0.15 for K=40
        self.init_std = init_std

        # Mean embedding normalization options
        self.mu_normalize = mu_normalize
        self.mu_max_norm = mu_max_norm

        # CRITICAL: diagonal_covariance is incompatible with gauge_fixed_priors!
        # When gauge_fixed_priors=True, Σ_i = R_i Σ_0 R_i^T produces a FULL matrix
        # even if Σ_0 is diagonal. Extracting only diagonal loses correlations
        # induced by rotation, breaking gauge covariance: Σ_i ≠ Ω_ij Σ_j Ω_ij^T
        if gauge_fixed_priors and diagonal_covariance:
            import warnings
            warnings.warn(
                "gauge_fixed_priors=True is incompatible with diagonal_covariance=True. "
                "Rotation R_i Σ_0 R_i^T produces full matrices. "
                "Forcing diagonal_covariance=False to preserve gauge covariance.",
                UserWarning
            )
            diagonal_covariance = False

        self.diagonal_covariance = diagonal_covariance
        self.isotropic_covariance = isotropic_covariance
        self.phi_dim = phi_dim
        self.gauge_param = gauge_param
        self.omega_head_dims = omega_head_dims

        if gauge_fixed_priors and generators is None:
            raise ValueError("gauge_fixed_priors=True requires generators to be provided")

        if generators is not None:
            self.register_buffer('generators', generators)

        # =================================================================
        # Mean Embeddings μ_i (or base prior μ_0 if gauge_fixed_priors)
        # =================================================================
        if gauge_fixed_priors:
            # Single base prior mean μ_0 - all token priors are rotations of this
            self.base_mu = nn.Parameter(torch.randn(embed_dim) * init_std)
        else:
            # Standard learnable embedding: vocab_size × embed_dim
            self.mu_embed = nn.Embedding(vocab_size, embed_dim)
            nn.init.normal_(self.mu_embed.weight, mean=0.0, std=init_std)

        # =================================================================
        # Covariance Embeddings Σ_i (or base prior Σ_0 if gauge_fixed_priors)
        # =================================================================
        # Parameterize via log-diagonal (ensures positivity):
        #   Σ = diag(exp(log_σ_diag))
        #
        # Diagonal priors are sufficient: gauge transport Ω@Σ@Ω.T rotates
        # diagonal Σ_p into non-diagonal covariance in the transported frame,
        # providing off-diagonal structure where it matters (between tokens).

        if gauge_fixed_priors:
            # Single base prior covariance Σ_0 - all token priors are rotations of this
            self.base_log_sigma_diag = nn.Parameter(
                torch.full((embed_dim,), math.log(init_sigma_scale))
            )
        elif learnable_sigma:
            # Per-token covariance as nn.Parameter (NOT nn.Embedding).
            # Both produce sparse gradients when indexed by token_ids, but
            # nn.Parameter yields a dense zero-padded gradient tensor.  This
            # matters for AdamW: Adam's v_t (second moment) accumulates the
            # zeros for absent tokens, gradually lowering v_t and raising the
            # effective LR when those tokens finally appear — an implicit
            # exploration bias for rare tokens.  Weight decay still applies to
            # all rows every step, acting as the Level 3 hyper-prior N(0, 1/(2·wd))
            # that pulls log_sigma toward 0 (i.e., σ² toward 1).
            self.log_sigma_diag = nn.Parameter(
                torch.full((vocab_size, embed_dim), math.log(init_sigma_scale))
            )
        else:
            # Shared isotropic covariance across all tokens
            self.register_buffer(
                'log_sigma_diag',
                torch.full((embed_dim,), math.log(init_sigma_scale))
            )

        # =================================================================
        # Sigma hyperprior target (frozen initial sigma for Level 3)
        # =================================================================
        # Fixed reference Σ_h for the hyperprior KL(s||h). Provides bidirectional
        # gradient on sigma_embed: pulls sigma toward init if it inflates OR deflates.
        # This is the "h" in the hierarchy h → s → p → q for covariance.
        sigma_target_val = init_sigma_scale  # scalar — initial σ value
        self.register_buffer(
            'sigma_target',
            torch.full((embed_dim,), sigma_target_val)
        )

        # =================================================================
        # Gauge Frame Embeddings
        # =================================================================
        if gauge_param == 'omega' and omega_head_dims is not None:
            # Direct Omega parameterization: store K_h×K_h matrices per head.
            # Covers full GL(K) including reflections. No matrix_exp needed.
            total_omega_params = sum(d * d for d in omega_head_dims)
            self.omega_embed = nn.Embedding(vocab_size, total_omega_params)
            # Initialize near identity: I + scale * randn per head block
            with torch.no_grad():
                omega_scale = phi_scale * 0.1  # Small perturbation from identity
                weight = torch.zeros(vocab_size, total_omega_params)
                offset = 0
                for d in omega_head_dims:
                    eye_flat = torch.eye(d).reshape(-1)  # (d*d,)
                    weight[:, offset:offset + d * d] = eye_flat.unsqueeze(0) + omega_scale * torch.randn(vocab_size, d * d)
                    offset += d * d
                self.omega_embed.weight.copy_(weight)
            # Dummy phi_base for compatibility (phi won't be used)
            self.register_buffer('phi_base', torch.zeros(phi_dim))
        elif learnable_phi or gauge_fixed_priors:
            # Lie algebra parameterization: φ_i ∈ ℝ^{n_gen}
            # CRITICAL: With gauge_fixed_priors=True, φ defines the token embedding!
            # μ_i = R(φ_i) @ μ_base, so different φ = different embeddings.
            # Zero init would make ALL tokens identical - must use random init!
            self.phi_embed = nn.Embedding(vocab_size, phi_dim)
            # IMPORTANT: Scale std inversely with sqrt(phi_dim) to maintain consistent
            # norm across different SO(N) dimensions. Target ||φ|| ≈ phi_scale regardless of N.
            phi_init_std = phi_scale / (phi_dim ** 0.5)
            nn.init.normal_(self.phi_embed.weight, mean=0.0, std=phi_init_std)
        else:
            # All tokens start at identity frame
            self.register_buffer('phi_base', torch.zeros(phi_dim))

        # =================================================================
        # Positional Embeddings (optional, like standard transformers)
        # =================================================================
        self.use_positional_embedding = use_positional_embedding
        self.max_seq_len = max_seq_len

        if use_positional_embedding:
            # Learnable positional embeddings added to μ
            self.pos_embed = nn.Embedding(max_seq_len, embed_dim)
            nn.init.normal_(self.pos_embed.weight, mean=0.0, std=init_std)

        # =================================================================
        # Reflection Embedding: per-token sign vector s_i ∈ {±1}^K
        # =================================================================
        # Adds discrete per-dimension sign flips to extend the continuous
        # gauge group to include reflections (det < 0 component):
        #
        #   phi path:  extends SO(K) → O(K) = SO(K) ⋊ (Z_2)^{K-1}
        #   omega path: extends GL⁺(K) → GL(K) via diag(s_i) · Ω_i
        #
        # The Lie algebra retraction preserves det sign, so continuous
        # gradient descent cannot cross between GL⁺ and GL⁻. The discrete
        # sign vectors provide the missing degree of freedom.
        #
        # Transport becomes: Ω_ij = diag(s_i) · Ω_i · Ω_j⁻¹ · diag(s_j)
        # which is implemented by applying s_i ⊙ μ_i before gauge transport.
        #
        # Properties (verified symbolically):
        #   - det(Ω_eff) covers both signs  ✓
        #   - S(Ω) = 0 (geometric bias vanishes)  ✓
        #   - Isotropic covariance preserved under transport  ✓
        #   - Clean Q-K factorization: Q_i = s_i ⊙ μ_i  ✓
        #
        # Gradient flow uses the straight-through estimator (STE):
        #   Forward:  sign(z)  (hard ±1)
        #   Backward: identity (grad flows through z)
        if learnable_reflection:
            # Continuous latent z_k; sign(z_k) gives the discrete ±1
            # Initialize with +1 (all positive) so model starts at SO(K)
            self.sign_logit = nn.Embedding(vocab_size, embed_dim)
            nn.init.ones_(self.sign_logit.weight)  # start at all +1 → no reflection

    def forward(
        self,
        token_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Embed tokens as agent beliefs at single base manifold point c*.

        Args:
            token_ids: (batch, seq_len) integer token indices

        Returns:
            mu: (batch, num_agents, K) mean beliefs (one per agent, NOT per spatial point)
            sigma: (batch, num_agents, K, K) covariances if diagonal_covariance=False
                   (batch, num_agents, K) diagonal variances if diagonal_covariance=True
            phi: (batch, num_agents, phi_dim) gauge frames (one per agent)
                 phi_dim = 3 for SO(3), N(N-1)/2 for SO(N), K² for GL(K)
            omega: (batch, num_agents, K, K) direct group elements (only when gauge_param='omega')
                   Block-diagonal GL(K) matrices. Returned as 4th element of tuple.

        NOTE: seq_len = number of agents at the single point c*
              This is NOT a spatial dimension!

        When gauge_fixed_priors=True:
            Priors are computed as p_i = R_i ▷ p_0 where R_i = exp(φ_i · generators).
            This guarantees p_i = Ω_ij[p_j] for all i,j, restoring gauge covariance.
        """
        batch_size, num_agents = token_ids.shape

        # =================================================================
        # Gauge Frame Embeddings (computed first for gauge_fixed_priors)
        # =================================================================
        omega = None  # Will be set if gauge_param='omega'
        if self.gauge_param == 'omega' and hasattr(self, 'omega_embed'):
            # Direct Omega path: reshape flat embedding to per-head K_h×K_h matrices
            omega_flat = self.omega_embed(token_ids)  # (B, N, total_omega_params)
            # Reshape into block-diagonal K×K matrix
            K = self.embed_dim
            omega = torch.zeros(batch_size, num_agents, K, K,
                                device=token_ids.device, dtype=omega_flat.dtype)
            offset = 0
            block_start = 0
            for d in self.omega_head_dims:
                omega_blk = omega_flat[:, :, offset:offset + d * d].reshape(
                    batch_size, num_agents, d, d)
                omega[:, :, block_start:block_start + d, block_start:block_start + d] = omega_blk
                offset += d * d
                block_start += d
            # phi is a dummy for compatibility
            phi = self.phi_base.unsqueeze(0).unsqueeze(0).expand(batch_size, num_agents, -1)
        elif self.learnable_phi or self.gauge_fixed_priors:
            # Per-token gauge frame (Lie algebra path)
            phi = self.phi_embed(token_ids)  # (B, N, phi_dim)
        else:
            # All agents at identity frame
            phi = self.phi_base.unsqueeze(0).unsqueeze(0)  # (1, 1, phi_dim)
            phi = phi.expand(batch_size, num_agents, -1)  # (B, N, phi_dim)

        # =================================================================
        # Mean and Covariance Embeddings
        # =================================================================
        if self.gauge_fixed_priors:
            # Compute rotation matrices R_i = exp(φ_i · generators)
            # phi: (B, N, 3), generators: (3, K, K)
            phi_matrix = torch.einsum('bnc,ckl->bnkl', phi, self.generators)  # (B, N, K, K)
            # Use stable_matrix_exp_pair with norm clamping to prevent overflow
            # in non-compact (symmetric) directions of GL(K)
            R, _ = stable_matrix_exp_pair(phi_matrix)  # (B, N, K, K)

            # Rotate base prior mean: μ_i = R_i @ μ_0
            # base_mu: (K,), R: (B, N, K, K)
            mu = torch.einsum('bnkl,l->bnk', R, self.base_mu)  # (B, N, K)

            # Build base covariance Σ_0 = diag(exp(log_σ_0))
            sigma_diag_base = torch.exp(self.base_log_sigma_diag).clamp(min=0.01, max=self.sigma_max)  # (K,)
            Sigma_0 = torch.diag(sigma_diag_base)  # (K, K)

            # Rotate base prior covariance: Σ_i = R_i @ Σ_0 @ R_i^T
            # R: (B, N, K, K), Sigma_0: (K, K)
            # NOTE: diagonal_covariance is forced False when gauge_fixed_priors=True
            # (see __init__), so we always output full matrices here.
            sigma = torch.einsum('bnij,jk,bnlk->bnil', R, Sigma_0, R)  # (B, N, K, K)
        else:
            # Standard per-token embeddings
            # μ(token_i) for each agent i at c*
            mu = self.mu_embed(token_ids)  # (B, N, K) where N = num_agents

            # Build diagonal covariances: Σ = diag(exp(log_σ))
            #
            # Use exp() for the ℝ → ℝ⁺ map (standard log-parameterization of
            # SPD diagonals), then hard clamp for numerical safety.  Hard clamp
            # zeros gradient at the boundary, which is correct: it prevents
            # log_sigma from drifting out of range.  (The detach-clamp variant
            # causes gradient explosion — see commit message.)
            _SIGMA_MIN = 0.01
            _SIGMA_MAX = self.sigma_max
            if self.learnable_sigma:
                log_sigma = self.log_sigma_diag[token_ids]  # (B, N, K)
                sigma_diag = torch.exp(log_sigma).clamp(_SIGMA_MIN, _SIGMA_MAX)  # (B, N, K)
            else:
                # Shared covariance
                sigma_diag = torch.exp(self.log_sigma_diag).clamp(_SIGMA_MIN, _SIGMA_MAX)  # (K,)
                sigma_diag = sigma_diag.unsqueeze(0).unsqueeze(0)  # (1, 1, K)
                sigma_diag = sigma_diag.expand(batch_size, num_agents, -1)  # (B, N, K)

            if self.diagonal_covariance:
                # Keep as diagonal variances (B, N, K)
                sigma = sigma_diag
            else:
                # Convert to full covariance matrices (diagonal)
                sigma = torch.diag_embed(sigma_diag)  # (B, N, K, K)

        # =================================================================
        # O(K) Reflection: apply per-token sign flip to μ
        # =================================================================
        # This implements R_i · μ_i = s_i ⊙ μ_i where s_i ∈ {±1}^K.
        # Combined with exp(φ) ∈ SO(K), this gives full O(K) transport.
        # Uses straight-through estimator: forward = sign(z), backward = identity.
        if self.learnable_reflection:
            z = self.sign_logit(token_ids)           # (B, N, K) continuous latent
            signs = z.sign()                          # hard {-1, +1}
            signs = z + (signs - z).detach()           # STE: grad flows through z
            mu = mu * signs                            # R_i · μ_i = s_i ⊙ μ_i

        # =================================================================
        # Isotropic covariance enforcement: Σ → σ²I
        # =================================================================
        # When isotropic_covariance=True, collapse per-dimension variances to
        # a single scalar (mean across K), then expand back. This enforces
        # the Limit 1 from the manuscript: all dimensions share one variance,
        # so KL(q_i || q_j) reduces to scaled squared Euclidean distance.
        if self.isotropic_covariance:
            if self.diagonal_covariance:
                # sigma shape: (B, N, K) — average across K to get scalar per agent
                scalar_var = sigma.mean(dim=-1, keepdim=True)  # (B, N, 1)
                sigma = scalar_var.expand_as(sigma)             # (B, N, K) all equal
            else:
                # sigma shape: (B, N, K, K) — extract diagonal, average, rebuild
                diag_vals = torch.diagonal(sigma, dim1=-2, dim2=-1)  # (B, N, K)
                scalar_var = diag_vals.mean(dim=-1, keepdim=True)     # (B, N, 1)
                K = sigma.shape[-1]
                sigma = scalar_var.unsqueeze(-1) * torch.eye(K, device=sigma.device, dtype=sigma.dtype)

        # =================================================================
        # Add positional embeddings to μ (like standard transformers)
        # =================================================================
        if self.use_positional_embedding:
            # Create position indices [0, 1, 2, ..., N-1]
            positions = torch.arange(num_agents, device=token_ids.device)  # (N,)
            pos_emb = self.pos_embed(positions)  # (N, K)
            mu = mu + pos_emb.unsqueeze(0)  # (B, N, K) + (1, N, K) -> (B, N, K)

        # =================================================================
        # Apply μ normalization/clamping (for sharper KL-based attention)
        # =================================================================
        if self.mu_normalize:
            # Project to unit sphere: ||μ|| = 1
            mu = torch.nn.functional.normalize(mu, dim=-1)
        elif self.mu_max_norm is not None:
            # Clamp norm: ||μ|| ≤ max_norm (like gradient clipping for embeddings)
            mu_norm = mu.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            scale = torch.clamp(self.mu_max_norm / mu_norm, max=1.0)
            mu = mu * scale

        if omega is not None:
            return mu, sigma, phi, omega
        return mu, sigma, phi

    def extra_repr(self) -> str:
        """Pretty print for model summary."""
        return (
            f"vocab_size={self.vocab_size}, "
            f"embed_dim={self.embed_dim}, "
            f"learnable_sigma={self.learnable_sigma}, "
            f"learnable_phi={self.learnable_phi}, "
            f"gauge_fixed_priors={self.gauge_fixed_priors}, "
            f"use_positional_embedding={self.use_positional_embedding}"
        )

    # =========================================================================
    # P-FLOW: EMA update of token embeddings toward successful beliefs
    # =========================================================================
    def _compute_pflow_weights(
        self,
        token_ids: torch.Tensor,           # (B, N)
        prediction_errors: torch.Tensor,   # (B, N)
        pad_token_id: int = -1,
    ) -> tuple:
        r"""Compute per-token-type prediction-error-weighted averages.

        Returns (unique_tokens, inverse_idx, weights) where weights are
        segment-wise softmax(-error) per token type, matching PriorBank's
        implementation. Uses scatter ops for O(B*N) vectorized computation.
        """
        flat_ids = token_ids.reshape(-1)            # (B*N,)
        flat_errors = prediction_errors.reshape(-1)  # (B*N,)

        # Mask padding
        valid = (flat_ids != pad_token_id)
        neg_errors = (-flat_errors.clamp(min=-10, max=10))
        # Set padding to very negative so it gets near-zero softmax weight
        neg_errors = neg_errors.masked_fill(~valid, -20.0)

        unique_tokens, inverse_idx = torch.unique(flat_ids, return_inverse=True)
        n_unique = unique_tokens.shape[0]

        # Segment-wise softmax: max per token type for numerical stability
        seg_max = torch.full((n_unique,), float('-inf'), device=flat_ids.device, dtype=neg_errors.dtype)
        seg_max.scatter_reduce_(0, inverse_idx, neg_errors, reduce='amax', include_self=False)
        shifted = neg_errors - seg_max[inverse_idx]
        exp_shifted = torch.exp(shifted)
        exp_shifted = exp_shifted * valid.float()  # Zero padding contributions

        # Normalize per token type
        seg_sum = torch.zeros(n_unique, device=flat_ids.device, dtype=neg_errors.dtype)
        seg_sum.scatter_add_(0, inverse_idx, exp_shifted)
        weights = exp_shifted / seg_sum[inverse_idx].clamp(min=1e-12)

        return unique_tokens, inverse_idx, weights, n_unique

    def update_embeddings_from_beliefs(
        self,
        token_ids: torch.Tensor,           # (B, N) token IDs in this batch
        mu_beliefs: torch.Tensor,          # (B, N, K) final beliefs after VFE
        prediction_errors: torch.Tensor,   # (B, N) per-position CE loss
        ema_decay: float = 0.99,           # EMA decay rate (higher = slower update)
        sigma_beliefs: Optional[torch.Tensor] = None,  # (B, N, K) belief variances
        pad_token_id: int = -1,            # Padding token ID to ignore
    ):
        r"""P-flow: Update token embeddings toward successful beliefs via EMA.

        Uses vectorized scatter ops (matching PriorBank implementation) with
        segment-wise softmax per token type. Updates both mu and sigma embeddings.

        Formula:
            \mu_{\text{token}} \leftarrow (1 - \eta) \mu_{\text{token}} + \eta \bar{\mu}_{\text{belief}}

        where \bar{\mu} is the prediction-error-weighted mean across all
        occurrences of this token in the batch.

        Args:
            token_ids: (B, N) token indices that were in this batch
            mu_beliefs: (B, N, K) final belief means after VFE dynamics
            prediction_errors: (B, N) per-position cross-entropy loss
            ema_decay: EMA decay rate (0.99 = slow, 0.9 = faster)
            sigma_beliefs: (B, N, K) belief variances (optional, for sigma P-flow)
            pad_token_id: Token ID for padding positions (excluded from update)
        """
        if self.gauge_fixed_priors:
            return

        B, N, K = mu_beliefs.shape
        lr = 1.0 - ema_decay

        with torch.no_grad():
            unique_tokens, inverse_idx, weights, n_unique = self._compute_pflow_weights(
                token_ids, prediction_errors, pad_token_id
            )

            flat_mu = mu_beliefs.reshape(-1, K)  # (B*N, K)

            # Weighted mean mu per token type via scatter_add
            weighted_mu = torch.zeros(n_unique, K, device=flat_mu.device, dtype=flat_mu.dtype)
            weighted_mu.scatter_add_(0, inverse_idx.unsqueeze(-1).expand(-1, K),
                                     flat_mu * weights.unsqueeze(-1))

            # Confidence-weighted LR: tokens with lower mean error get higher LR
            flat_errors = prediction_errors.reshape(-1)
            seg_error_sum = torch.zeros(n_unique, device=flat_mu.device, dtype=flat_mu.dtype)
            seg_error_sum.scatter_add_(0, inverse_idx, flat_errors.clamp(min=0))
            seg_count = torch.zeros(n_unique, device=flat_mu.device, dtype=flat_mu.dtype)
            seg_count.scatter_add_(0, inverse_idx, torch.ones_like(flat_errors))
            mean_errors = seg_error_sum / seg_count.clamp(min=1)
            confidence = 1.0 / (1.0 + mean_errors)
            effective_lr = lr * confidence  # (n_unique,)

            # Skip padding token if present
            pad_mask = (unique_tokens != pad_token_id)
            update_tokens = unique_tokens[pad_mask]
            update_mu = weighted_mu[pad_mask]
            update_lr = effective_lr[pad_mask]

            # Vectorized EMA update for mu
            self.mu_embed.weight.data[update_tokens] = (
                (1.0 - update_lr.unsqueeze(-1)) * self.mu_embed.weight.data[update_tokens] +
                update_lr.unsqueeze(-1) * update_mu
            )

            # Sigma P-flow: update log_sigma_diag if learnable
            if sigma_beliefs is not None and self.learnable_sigma and hasattr(self, 'log_sigma_diag'):
                # Handle full covariance (B, N, K, K) → extract diagonal
                if sigma_beliefs.dim() == 4:
                    sigma_beliefs_diag = sigma_beliefs.diagonal(dim1=-2, dim2=-1)  # (B, N, K)
                else:
                    sigma_beliefs_diag = sigma_beliefs  # Already (B, N, K)
                flat_sigma = sigma_beliefs_diag.reshape(-1, K)  # (B*N, K)
                weighted_sigma = torch.zeros(n_unique, K, device=flat_mu.device, dtype=flat_mu.dtype)
                weighted_sigma.scatter_add_(0, inverse_idx.unsqueeze(-1).expand(-1, K),
                                            flat_sigma * weights.unsqueeze(-1))
                update_sigma = weighted_sigma[pad_mask]
                sigma_lr = update_lr * 0.1  # Slower sigma updates for stability
                # log_sigma_diag is nn.Parameter (V, K), not nn.Embedding
                current_sigma = torch.exp(self.log_sigma_diag.data[update_tokens])
                new_sigma = (
                    (1.0 - sigma_lr.unsqueeze(-1)) * current_sigma +
                    sigma_lr.unsqueeze(-1) * update_sigma
                )
                self.log_sigma_diag.data[update_tokens] = torch.log(new_sigma.clamp(min=1e-6))

    def update_phi_from_beliefs(
        self,
        token_ids: torch.Tensor,           # (B, N) token IDs
        phi_evolved: torch.Tensor,         # (B, N, phi_dim) VFE-evolved phi values
        prediction_errors: torch.Tensor,   # (B, N) per-position CE loss
        ema_decay: float = 0.99,           # EMA decay rate
        pad_token_id: int = -1,            # Padding token ID to ignore
    ):
        r"""Phi P-flow: Update gauge frame embeddings toward VFE-evolved values.

        The E-step evolves phi within each forward pass to minimize transported KL.
        This method persists those evolved values back to phi_embed via EMA,
        providing a local learning rule for gauge frames without backprop.

        Args:
            token_ids: (B, N) token indices
            phi_evolved: (B, N, phi_dim) evolved phi after VFE iterations
            prediction_errors: (B, N) per-position CE loss
            ema_decay: EMA decay rate
            pad_token_id: Padding token ID
        """
        if not hasattr(self, 'phi_embed'):
            return

        phi_dim = phi_evolved.shape[-1]
        lr = 1.0 - ema_decay

        with torch.no_grad():
            unique_tokens, inverse_idx, weights, n_unique = self._compute_pflow_weights(
                token_ids, prediction_errors, pad_token_id
            )

            flat_phi = phi_evolved.reshape(-1, phi_dim)  # (B*N, phi_dim)

            # Weighted mean phi per token type
            weighted_phi = torch.zeros(n_unique, phi_dim, device=flat_phi.device, dtype=flat_phi.dtype)
            weighted_phi.scatter_add_(0, inverse_idx.unsqueeze(-1).expand(-1, phi_dim),
                                      flat_phi * weights.unsqueeze(-1))

            # Skip padding
            pad_mask = (unique_tokens != pad_token_id)
            update_tokens = unique_tokens[pad_mask]
            update_phi = weighted_phi[pad_mask]

            # Vectorized EMA update for phi
            self.phi_embed.weight.data[update_tokens] = (
                (1.0 - lr) * self.phi_embed.weight.data[update_tokens] +
                lr * update_phi
            )

    def get_embedding_stats(self) -> dict:
        """Get statistics about embeddings for logging."""
        with torch.no_grad():
            if hasattr(self, 'mu_embed'):
                mu_weight = self.mu_embed.weight
            elif hasattr(self, 'base_mu'):
                mu_weight = self.base_mu.unsqueeze(0)  # (1, K) for consistent stats
            else:
                return {}
            return {
                'embed_mu_mean': mu_weight.mean().item(),
                'embed_mu_std': mu_weight.std().item(),
                'embed_mu_norm_mean': mu_weight.norm(dim=-1).mean().item(),
            }


def so3_log_torch(R: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Logarithm map from SO(3) → so(3) (PyTorch version).

    Given R ∈ SO(3), find φ ∈ ℝ³ such that exp([φ]_×) = R.

    Formula:
        θ = arccos((tr(R) - 1) / 2)
        φ = (θ / (2 sin θ)) * vex(R - Rᵀ)

    Args:
        R: Rotation matrices, shape (..., 3, 3)
        eps: Threshold for small angle approximation

    Returns:
        phi: Lie algebra elements, shape (..., 3)
    """
    # Compute rotation angle from trace
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]  # (...)
    cos_theta = (trace - 1.0) / 2.0
    # Upcast to float64 before acos to avoid precision loss near ±1 in float32
    cos_theta = torch.clamp(cos_theta.double(), -1.0 + eps, 1.0 - eps)
    theta = torch.acos(cos_theta).float()  # (...)

    # Extract skew-symmetric part: vex(R - R^T) / 2
    # vex extracts [v_x, v_y, v_z] from skew-symmetric matrix
    skew = R - R.transpose(-1, -2)  # (..., 3, 3)
    v_x = (skew[..., 2, 1] - skew[..., 1, 2]) / 2.0
    v_y = (skew[..., 0, 2] - skew[..., 2, 0]) / 2.0
    v_z = (skew[..., 1, 0] - skew[..., 0, 1]) / 2.0
    vex_skew = torch.stack([v_x, v_y, v_z], dim=-1)  # (..., 3)

    # Coefficient: θ / (2 sin θ), handle small angles and near-pi
    sin_theta = torch.sin(theta)
    # For small θ: θ/(2sinθ) ≈ 1/2 + θ²/12
    small_angle = theta < eps
    # For θ near π: sin(θ) → 0, vex(R - R^T) → 0, need special handling
    near_pi = theta > (3.141592653589793 - 0.01)

    coeff = torch.where(
        small_angle,
        0.5 + theta**2 / 12.0,
        theta / (2.0 * sin_theta + eps)
    )

    # Standard formula works for most angles
    phi = coeff.unsqueeze(-1) * vex_skew

    # Near θ=π: extract axis from R + I (the column with largest norm)
    if near_pi.any():
        RpI = R + torch.eye(3, device=R.device, dtype=R.dtype)  # (..., 3, 3)
        # Find column with largest norm for each matrix
        col_norms = RpI.norm(dim=-2)  # (..., 3)
        best_col = col_norms.argmax(dim=-1)  # (...)
        # Gather the best column
        idx = best_col.unsqueeze(-1).unsqueeze(-1).expand(*best_col.shape, 3, 1)
        axis = torch.gather(RpI, dim=-1, index=idx).squeeze(-1)  # (..., 3)
        axis = axis / (axis.norm(dim=-1, keepdim=True) + 1e-12)
        # Sign convention: ensure consistency with vex direction
        # dot product with vex_skew determines sign
        dot = (axis * vex_skew).sum(dim=-1, keepdim=True)
        axis = torch.where(dot >= 0, axis, -axis)
        phi_pi = theta.unsqueeze(-1) * axis
        phi = torch.where(near_pi.unsqueeze(-1), phi_pi, phi)

    return phi


def so3_compose_bch(
    phi1: torch.Tensor,
    phi2: torch.Tensor,
    order: int = 1,
) -> torch.Tensor:
    """
    Compose two so(3) elements using Baker-Campbell-Hausdorff formula.

    log(exp(φ₁)·exp(φ₂)) = φ₁ + φ₂ + ½[φ₁,φ₂] + (1/12)[φ₁,[φ₁,φ₂]] - ...

    For so(3), the Lie bracket is: [X, Y] = X × Y (cross product)

    Args:
        phi1: First so(3) element, shape (..., 3)
        phi2: Second so(3) element, shape (..., 3)
        order: BCH expansion order (0=addition, 1=first correction, 2=second)

    Returns:
        phi_composed: Composed element in so(3), shape (..., 3)
    """
    if order == 0:
        # Simple addition (valid for small angles only)
        return phi1 + phi2

    # First-order BCH: φ₁ + φ₂ + ½[φ₁,φ₂]
    # In so(3): [φ₁,φ₂] = φ₁ × φ₂ (cross product)
    bracket_12 = torch.cross(phi1, phi2, dim=-1)
    result = phi1 + phi2 + 0.5 * bracket_12

    if order >= 2:
        # Second-order: + (1/12)[φ₁,[φ₁,φ₂]] - (1/12)[φ₂,[φ₁,φ₂]]
        bracket_1_12 = torch.cross(phi1, bracket_12, dim=-1)
        bracket_2_12 = torch.cross(phi2, bracket_12, dim=-1)
        result = result + (1.0/12.0) * bracket_1_12 - (1.0/12.0) * bracket_2_12

    return result


class GaugePositionalEncoding(nn.Module):
    """
    Agent-index-dependent gauge frame modulation (0D positional encoding).

    In 0D: Position encodes AGENT INDEX, not spatial location.
    All agents are at the same point c*, but need to distinguish
    their roles in the sequence via gauge frame modulation.

    Encoding modes:
        - 'none': No positional encoding in gauge space. Transport Ω_ij is purely
                  content-based. Use with use_positional_embedding=True to put
                  position info in μ instead, or for position-invariant attention.
        - 'learned': Learnable per-position gauge frames φ_pos[i] ∈ so(n)
        - 'sinusoidal': Fixed sinusoidal encoding (like original Transformer)

    Composition modes (for combining token φ with positional φ):
        - 'add': φ_combined = φ_base + φ_pos (valid for small angles)
        - 'bch1': φ_combined = φ_base + φ_pos + ½[φ_base, φ_pos] (BCH order 1)
        - 'bch2': Higher-order BCH correction
        - 'exact': Full SO(3) composition via exp → multiply → log [SO(3) only]

    WARNING: Positional encoding in gauge space creates ABSOLUTE position-dependent
    transport operators. This can cause attention to be dominated by position rather
    than content. For translation-invariant attention, use mode='none'.

    Supported gauge groups:
        - 'SO3': SO(3) with 3 generators (cross-product bracket, exact composition)
        - 'SON': SO(N) with N(N-1)/2 generators (matrix commutator bracket)
        - 'GLK': GL(K) with K² generators or multi-head GL(d_head)^H
                 Uses general Lie bracket via transport generators.
    """

    def __init__(
        self,
        max_seq_len: int,
        mode: str = 'none',  # Default: no positional encoding in gauge space
        scale: float = 0.1,
        composition: str = 'exact',  # Default: full SO(3) composition (most accurate)
        phi_dim: int = 3,  # 3 for SO(3), N(N-1)/2 for SO(N), K² for GL(K)
        generators: Optional[torch.Tensor] = None,  # Transport generators for BCH composition
        gauge_group: str = 'SO3',  # 'SO3', 'SON', or 'GLK'
    ):
        """
        Initialize positional encoding in gauge space.

        Args:
            max_seq_len: Maximum sequence length (max number of agents at c*)
            mode: 'none', 'learned', or 'sinusoidal'
                  - 'none': No positional gauge encoding (transport is content-only)
                  - 'learned': Learnable positional gauge frames
                  - 'sinusoidal': Fixed sinusoidal encoding
            scale: Scaling factor for positional encodings (ignored if mode='none')
            composition: How to combine token φ with positional φ:
                - 'add': Simple addition (φ_base + φ_pos) - fast but only valid for small angles
                - 'bch1': BCH order 1 correction
                - 'bch2': BCH order 2 correction
                - 'exact': Full SO(3) composition (SO(3) only)
            phi_dim: Dimension of gauge frame φ. 3 for SO(3), N(N-1)/2 for SO(N),
                     K² for GL(K), H*d² for multi-head GL(K).
            generators: Transport generators (n_gen, K, K). Required for BCH composition
                        with SO(N) or GL(K) gauge groups.
            gauge_group: Gauge group type ('SO3', 'SON', or 'GLK').
                         Determines which Lie bracket computation to use.
        """
        super().__init__()
        self.max_seq_len = max_seq_len
        self.mode = mode
        self.scale = scale
        self.phi_dim = phi_dim
        self.gauge_group = gauge_group

        # Store generators for BCH composition
        if generators is not None:
            self.register_buffer('generators', generators)
        else:
            self.generators = None

        # Validate composition mode for each gauge group
        if gauge_group == 'GLK':
            # GL(K): 'exact' not supported (matrix log is problematic for GL),
            # but BCH works via general Lie bracket with transport generators
            if composition == 'exact':
                print(f"[WARNING] Composition 'exact' not supported for GL(K). "
                      f"Falling back to 'bch2' for phi_dim={phi_dim}.")
                composition = 'bch2'
            if composition in ['bch1', 'bch2'] and generators is None and not GENERAL_BCH_AVAILABLE:
                print(f"[WARNING] GL(K) BCH requires generators. "
                      f"Falling back to 'add' for phi_dim={phi_dim}.")
                composition = 'add'
        elif phi_dim != 3:
            # SO(N) with N > 3: 'exact' is SO(3)-only
            if composition == 'exact':
                print(f"[WARNING] Composition 'exact' only supported for SO(3) (phi_dim=3). "
                      f"Falling back to 'bch2' for phi_dim={phi_dim}.")
                composition = 'bch2'
            if composition in ['bch1', 'bch2'] and generators is None and not SON_BCH_AVAILABLE:
                print(f"[WARNING] SO(N) BCH requires generators. "
                      f"Falling back to 'add' for phi_dim={phi_dim}.")
                composition = 'add'
        self.composition = composition

        if mode == 'none':
            # No positional encoding in gauge space - φ stays content-only
            # Use this when position info comes from μ (use_positional_embedding)
            # or when you want purely content-based transport operators
            self.register_buffer('pos_phi', torch.zeros(max_seq_len, phi_dim))

        elif mode == 'learned':
            # Learnable agent-index-specific gauge biases
            # Each agent index i gets a unique φ_pos(i) ∈ so(n)
            self.pos_phi = nn.Parameter(torch.randn(max_seq_len, phi_dim) * scale)

        elif mode == 'sinusoidal':
            # Sinusoidal encoding projected to so(n)
            # Fixed (not learnable)
            self.register_buffer('pos_phi', self._make_sinusoidal(max_seq_len, scale, phi_dim))

        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'none', 'learned', or 'sinusoidal'.")

    def _make_sinusoidal(self, max_len: int, scale: float, phi_dim: int = 3) -> torch.Tensor:
        """
        Create sinusoidal positional encoding in so(n).

        This encodes agent index i, not spatial position!

        Formula (adapted from Transformer):
            For each dimension d in [0, phi_dim):
            φ_pos[i, d] = scale * sin/cos(i / 10000^(d/phi_dim))

        Args:
            max_len: Maximum sequence length
            scale: Scaling factor
            phi_dim: Dimension of gauge frame (3 for SO(3), N(N-1)/2 for SO(N))

        Returns:
            pos_phi: (max_len, phi_dim) positional gauge frames
        """
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)  # (L, 1)
        # Per-dimension unique frequencies: each gauge coordinate d gets its own
        # frequency f_d = 10000^(-d/phi_dim), with sin at even d, cos at odd d.
        # This is intentionally different from the standard transformer encoding
        # (which shares frequencies across sin/cos pairs). For gauge frames, each
        # Lie algebra coordinate controls a different generator, so independent
        # positional modulation per coordinate gives richer structure.
        div_term = torch.exp(torch.arange(0, phi_dim, 1, dtype=torch.float32) * -(math.log(10000.0) / phi_dim))

        phi = torch.zeros(max_len, phi_dim)
        for d in range(phi_dim):
            if d % 2 == 0:
                phi[:, d] = torch.sin(position.squeeze(-1) * div_term[d])
            else:
                phi[:, d] = torch.cos(position.squeeze(-1) * div_term[d])

        return phi * scale

    def forward(self, num_agents: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Get positional gauge frames for given number of agents.

        Args:
            num_agents: Number of agents (sequence length)
            device: Device to place output on

        Returns:
            pos_phi: (num_agents, phi_dim) agent-index-dependent gauge frames

        NOTE: This is NOT a spatial field! Just one φ per agent index.
        """
        if num_agents > self.max_seq_len:
            raise ValueError(
                f"Sequence length {num_agents} exceeds max {self.max_seq_len}. "
                f"Increase max_seq_len in config."
            )

        pos_phi = self.pos_phi[:num_agents]  # (N, phi_dim)

        if device is not None:
            pos_phi = pos_phi.to(device)

        return pos_phi

    def compose(
        self,
        phi: torch.Tensor,
        num_agents: int,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Compose token gauge frames with positional gauge frames using proper Lie group composition.

        Dispatches to the correct bracket computation based on gauge_group:
        - SO(3): Cross product bracket (fast, specialized)
        - SO(N): soN_compose_bch_torch (matrix commutator in N×N space)
        - GL(K): lie_compose_bch_general_torch (general bracket via transport generators)

        Args:
            phi: Token gauge frames, shape (B, N, phi_dim)
            num_agents: Number of agents (sequence length)
            device: Device to place output on

        Returns:
            phi_combined: Composed gauge frames, shape (B, N, phi_dim)

        Mathematical background:
            The correct composition is R_combined = exp(φ·G) · exp(φ_pos·G)
            In the Lie algebra, the BCH formula gives:
            log(exp(X)·exp(Y)) = X + Y + ½[X,Y] + (1/12)[X,[X,Y]] - (1/12)[Y,[X,Y]] + ...
            For so(3): [X,Y] = X × Y (cross product)
            For so(N)/gl(K): [X,Y] = XY - YX (matrix commutator)
        """
        # Short-circuit: if mode='none', φ_pos is all zeros, so return unchanged φ
        # This avoids unnecessary tensor operations and BCH computations
        if self.mode == 'none':
            return phi

        pos_phi = self.forward(num_agents, device)  # (N, phi_dim)
        pos_phi = pos_phi.unsqueeze(0).expand(phi.shape[0], -1, -1)  # (B, N, phi_dim)

        if self.composition == 'add':
            # Simple addition (original behavior, valid for small angles only)
            return phi + pos_phi

        elif self.composition in ('bch1', 'bch2'):
            order = 1 if self.composition == 'bch1' else 2

            if self.gauge_group == 'GLK':
                # GL(K): Use general Lie bracket via transport generators
                if GENERAL_BCH_AVAILABLE and self.generators is not None:
                    return lie_compose_bch_general_torch(phi, pos_phi, self.generators, order=order)
                else:
                    # Fallback to addition
                    return phi + pos_phi
            elif self.phi_dim == 3 and self.gauge_group == 'SO3':
                # SO(3)-specific BCH (cross product — fastest)
                return so3_compose_bch(phi, pos_phi, order=order)
            elif SON_BCH_AVAILABLE and self.generators is not None:
                # SO(N) BCH (matrix commutator in N×N gauge space)
                return soN_compose_bch_torch(phi, pos_phi, self.generators, order=order)
            elif GENERAL_BCH_AVAILABLE and self.generators is not None:
                # General fallback: works for any algebra
                return lie_compose_bch_general_torch(phi, pos_phi, self.generators, order=order)
            else:
                # Last resort: addition
                return phi + pos_phi

        elif self.composition == 'exact':
            # Full SO(3) composition: log(exp(φ) · exp(φ_pos))
            # Build skew-symmetric matrices and exponentiate
            # [φ]_× for so(3) → SO(3)
            def skew_symmetric_batch(v):
                """v: (..., 3) -> (..., 3, 3) skew-symmetric"""
                zeros = torch.zeros_like(v[..., 0])
                return torch.stack([
                    torch.stack([zeros, -v[..., 2], v[..., 1]], dim=-1),
                    torch.stack([v[..., 2], zeros, -v[..., 0]], dim=-1),
                    torch.stack([-v[..., 1], v[..., 0], zeros], dim=-1),
                ], dim=-2)

            phi_skew = skew_symmetric_batch(phi)  # (B, N, 3, 3)
            pos_phi_skew = skew_symmetric_batch(pos_phi)  # (B, N, 3, 3)

            R_phi = torch.linalg.matrix_exp(phi_skew)  # (B, N, 3, 3)
            R_pos = torch.linalg.matrix_exp(pos_phi_skew)  # (B, N, 3, 3)

            R_combined = R_phi @ R_pos  # (B, N, 3, 3)

            # Map back to so(3) via logarithm
            phi_combined = so3_log_torch(R_combined)  # (B, N, 3)
            return phi_combined

        else:
            raise ValueError(f"Unknown composition mode: {self.composition}")

    def extra_repr(self) -> str:
        return f"max_seq_len={self.max_seq_len}, mode={self.mode}, scale={self.scale}, composition={self.composition}"


# =============================================================================
# Testing & Visualization
# =============================================================================

if __name__ == '__main__':
    print("="*70)
    print("GAUGE TOKEN EMBEDDING TEST (0D Transformer)")
    print("="*70)

    # Test configuration
    vocab_size = 100
    embed_dim = 32
    batch_size = 4
    seq_len = 10

    # Create embedding layer
    print("\n[1] Creating GaugeTokenEmbedding...")
    embedder = GaugeTokenEmbedding(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        init_std=0.02,
        init_sigma_scale=0.1,
        learnable_sigma=False,  # Start simple
        learnable_phi=False,
    )
    print(embedder)

    # Create random tokens
    print(f"\n[2] Embedding random tokens...")
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"    Token IDs shape: {token_ids.shape}")

    # Forward pass
    mu, sigma, phi = embedder(token_ids)

    print(f"\n[3] Output shapes:")
    print(f"    μ (means):      {mu.shape}     # (B, N, K) where N=num_agents at c*")
    print(f"    Σ (covariances): {sigma.shape}   # (B, N, K, K)")
    print(f"    φ (gauge frames): {phi.shape}      # (B, N, 3) in so(3)")

    # Validate covariance is SPD
    print(f"\n[4] Validating covariances...")
    eigenvalues = torch.linalg.eigvalsh(sigma[0, 0])  # Check first agent
    print(f"    Eigenvalues of Σ[0,0]: {eigenvalues.numpy()}")
    assert torch.all(eigenvalues > 0), "Covariance not positive definite!"
    print("    ✓ All eigenvalues positive (SPD verified)")

    # Test positional encoding
    print(f"\n{'='*70}")
    print("GAUGE POSITIONAL ENCODING TEST")
    print('='*70)

    max_seq_len = 64

    # Test learned encoding
    print("\n[5] Testing learned positional encoding...")
    pos_enc_learned = GaugePositionalEncoding(max_seq_len, mode='learned', scale=0.1)
    pos_phi_learned = pos_enc_learned(seq_len)
    print(f"    Learned φ_pos shape: {pos_phi_learned.shape}  # (N, 3)")
    print(f"    φ_pos[0]: {pos_phi_learned[0].detach().numpy()}")
    print(f"    φ_pos[9]: {pos_phi_learned[9].detach().numpy()}")

    # Test sinusoidal encoding
    print("\n[6] Testing sinusoidal positional encoding...")
    pos_enc_sin = GaugePositionalEncoding(max_seq_len, mode='sinusoidal', scale=0.1)
    pos_phi_sin = pos_enc_sin(seq_len)
    print(f"    Sinusoidal φ_pos shape: {pos_phi_sin.shape}")
    print(f"    φ_pos[0]: {pos_phi_sin[0].numpy()}")
    print(f"    φ_pos[9]: {pos_phi_sin[9].numpy()}")

    # Combined: Embedding + Position
    print(f"\n[7] Combined embedding with positional encoding...")
    phi_combined = phi + pos_phi_learned.unsqueeze(0)  # (B, N, 3)
    print(f"    φ_total = φ_base + φ_pos: {phi_combined.shape}")

    # Parameter count
    total_params = sum(p.numel() for p in embedder.parameters())
    print(f"\n[8] Parameter count:")
    print(f"    Token embedder: {total_params:,} parameters")
    pos_params = sum(p.numel() for p in pos_enc_learned.parameters())
    print(f"    Position encoder (learned): {pos_params:,} parameters")

    print("\n" + "="*70)
    print("✓ All tests passed!")
    print("="*70)
