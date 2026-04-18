"""
Gauge-Theoretic Transformer Block (0D Architecture)
====================================================

Complete transformer block with:
1. Gauge-theoretic multi-head attention (IrrepMultiHeadAttention, KL-based)
2. Variational free energy FFN (VariationalFFNDynamic, E-step belief evolution)
3. Optional normalization (LayerNorm/RMSNorm) and residual connections (toggled for pure VFE ablation)
4. Optional non-flat gauge transport via edge-local GaugeConnection

Data flow:
    (μ, Σ, φ) → LayerNorm(μ) → Attention(KL + transport) → Residual
              → LayerNorm(μ) → VFE FFN(E-step iterations)  → Residual → (μ', Σ', φ')

All configuration flows through BlockConfig — no raw kwargs.
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Union

from transformer.core.block_config import BlockConfig

# Import our gauge attention
from transformer.core.attention import IrrepMultiHeadAttention

# Import VFE FFN directly (no wrapper)
from transformer.core.variational_ffn import VariationalFFNDynamic

from transformer.core.active_inference import configure_ffn_active_inference

# Import gauge connection for non-flat transport
from transformer.core.connection import (
    GaugeConnection,
    PerHeadGaugeConnection,
    partition_generators_by_block,
)

# Import block-diagonal matrix exp for shared transport caching
from transformer.core.gauge_utils import fused_block_matrix_exp_pairs

# Trajectory tracking (core-side protocol — analysis layer registers via set_global_recorder)
from transformer.core.vfe_utils import get_global_recorder
from transformer.core.gauge_ridge import make_ridge


class RMSNorm(nn.Module):
    r"""Root Mean Square Layer Normalization.

    .. math::
        \mu_{\text{rms}} = \frac{\mu}{\text{RMS}(\mu)} \cdot \gamma

    where :math:`\text{RMS}(\mu) = \sqrt{\frac{1}{K}\sum_k \mu_k^2 + \epsilon}`.

    Unlike LayerNorm, RMSNorm does not subtract the mean, preserving the mean
    of belief vectors while normalizing scale. This partially preserves the
    GL(K) scaling component of gauge transport.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.eps = eps

    def forward(self, x: torch.Tensor, sigma: torch.Tensor = None) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight

    def extra_repr(self) -> str:
        return f"{self.weight.shape[0]}, eps={self.eps}"


class MahalanobisNorm(nn.Module):
    r"""Gauge-equivariant Mahalanobis normalization.

    Projects beliefs onto the constant-Mahalanobis-norm submanifold:

    .. math::
        \mu_{\text{norm}} = \mu \cdot \sqrt{\frac{K}{\mu^\top \Sigma^{-1} \mu + \epsilon}}

    **Gauge equivariance proof**: The Mahalanobis norm
    :math:`s^2 = \mu^\top \Sigma^{-1} \mu` is a gauge scalar (invariant
    under :math:`\mu \to g\mu,\; \Sigma \to g\Sigma g^\top`), so the
    normalization factor :math:`\sqrt{K/s^2}` is gauge-invariant. Since
    :math:`\mu` transforms as a vector and the factor is a scalar, their
    product transforms as a vector: the normalization commutes with gauge
    transport.

    **Key-norm bias cancellation** (isotropic/shared-metric regime):
    After normalization, :math:`\|\mu_{\text{norm}}\|_M^2 = K` for all
    tokens. When all keys share the same metric (isotropic
    :math:`\Sigma = \sigma^2 I` or shared :math:`\Sigma_j`), the
    key-dependent bias becomes constant and cancels under softmax.
    With token-dependent :math:`\Sigma_j`, non-orthogonal :math:`\Omega`,
    or transported key covariances, the cancellation is approximate.

    **Isotropic limit**: When :math:`\Sigma = \sigma^2 I`, reduces to
    :math:`\mu / \text{RMS}(\mu) \cdot \sigma`, recovering standard
    RMSNorm up to a global scale — consistent with the RG flow to the
    standard transformer fixed point.

    For diagonal covariance: :math:`s^2 = \sum_k \mu_k^2 / \sigma_k`,
    cost is O(K) — same as standard RMSNorm.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.K = normalized_shape
        self.eps = eps

    def forward(self, x: torch.Tensor, sigma: torch.Tensor = None,
                exp_phi: torch.Tensor = None) -> torch.Tensor:
        r"""Normalize mu using the Mahalanobis norm with covariance sigma.

        Args:
            x: (..., K) belief means.
            sigma: (..., K) diagonal variances or (..., K, K) full covariance.
                If None, falls back to standard RMSNorm (Euclidean).
            exp_phi: Optional (..., K, K) local gauge frame g = exp(phi).
                When provided (and sigma is full-cov), the numerical ridge
                uses ``eps * (g g^T)`` instead of ``eps * I``, preserving
                ``Sigma -> h Sigma h^T`` covariance exactly. When ``None``,
                falls back to ``eps * I`` (bitwise identical to prior behavior).

        Returns:
            Normalized means with :math:`\|\mu\|_M^2 = K`.
        """
        if sigma is None:
            # Fallback: standard RMSNorm (no covariance available)
            rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
            return x / rms

        if sigma.dim() == x.dim():
            # Diagonal covariance: sigma is (..., K)
            # s^2 = sum_k mu_k^2 / sigma_k
            s2 = (x.pow(2) / sigma.clamp(min=self.eps)).sum(dim=-1, keepdim=True)
        else:
            # Full covariance: sigma is (..., K, K)
            # s^2 = mu^T Sigma^{-1} mu via direct solve (more numerically
            # stable than Cholesky for gauge equivariance)
            with torch.amp.autocast('cuda', enabled=False):
                x_f32 = x.float()
                ridge = make_ridge(
                    self.K, self.eps,
                    exp_phi=exp_phi.float() if exp_phi is not None else None,
                    device=sigma.device, dtype=torch.float32,
                )
                sigma_f32 = sigma.float() + ridge
                # Sigma^{-1} mu via solve: avoids explicit inverse
                sig_inv_mu = torch.linalg.solve(
                    sigma_f32, x_f32.unsqueeze(-1)
                ).squeeze(-1)
                s2 = (x_f32 * sig_inv_mu).sum(dim=-1, keepdim=True).to(x.dtype)

        scale = torch.sqrt(self.K / (s2 + self.eps))
        return x * scale

    def extra_repr(self) -> str:
        return f"K={self.K}, eps={self.eps}"


def _make_norm(norm_type: str, dim: int) -> nn.Module:
    """Factory for normalization layers.

    Args:
        norm_type: 'layernorm', 'rmsnorm', 'mahalnorm', or 'none'
        dim: Normalized dimension (embed_dim).
    """
    if norm_type == 'layernorm':
        return nn.LayerNorm(dim)
    elif norm_type == 'rmsnorm':
        return RMSNorm(dim)
    elif norm_type == 'mahalnorm':
        return MahalanobisNorm(dim)
    elif norm_type == 'none':
        return nn.Identity()
    else:
        raise ValueError(f"Unknown norm_type: {norm_type!r}. Expected 'layernorm', 'rmsnorm', 'mahalnorm', or 'none'.")


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
        self.norm_type = cfg.norm_type
        self.use_residual = cfg.use_residual
        # residual_type: 'additive' (default, matches the 71-PPL 2026-04-07
        # TransformerOld baseline) vs 'delta' (2026-04-07 audit Fix #1 / #20
        # form).  See blocks.py comment blocks at the residual sites and
        # edits_2026-04-08.md Round 3 for the history.
        self.residual_type = getattr(cfg, 'residual_type', 'additive')
        self.sigma_max = cfg.sigma_max
        self.skip_attention = getattr(cfg, 'skip_attention', False)

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
            irrep_dims_override=cfg.ffn_irrep_dims if (gauge_group == 'GLK' and cfg.ffn_irrep_dims is not None) else None,
            use_rope=cfg.use_rope,
            rope_base=cfg.rope_base,
            sigma_aggregation=cfg.sigma_aggregation,
            learnable_head_kappa=cfg.learnable_head_kappa,
            alpha_divergence=getattr(cfg, 'alpha_divergence', 1.0),
            gauge_covariant_ridge=getattr(cfg, 'gauge_covariant_ridge', False),
        )

        # Normalization (LayerNorm, RMSNorm, or Identity)
        self.norm1 = _make_norm(cfg.norm_type, cfg.embed_dim)

        # =====================================================================
        # VFE_dynamic FFN Sublayer (VariationalFFNDynamic directly, no wrapper)
        # =====================================================================
        if cfg.generators is None:
            raise ValueError("generators required for VFE_dynamic mode")

        self.ffn = VariationalFFNDynamic(
            embed_dim=cfg.embed_dim,
            generators=cfg.generators,
            alpha=cfg.E_alpha,
            lambda_belief=cfg.E_lambda_belief,
            lambda_softmax=cfg.E_lambda_softmax,
            kappa=cfg.ffn_kappa,
            n_iterations=cfg.ffn_n_iterations,
            learnable_lr=cfg.E_learnable_lr,
            mu_lr=cfg.E_mu_q_lr,
            sigma_lr=cfg.E_sigma_q_lr,
            update_sigma=cfg.ffn_update_sigma,
            diagonal_covariance=cfg.diagonal_covariance,
            exact_diagonal_transport=cfg.exact_diagonal_transport,
            update_phi=cfg.evolve_phi,
            update_phi_per_iteration=cfg.evolve_phi_e_step,
            phi_lr=cfg.phi_lr,
            phi_max_norm=cfg.phi_max_norm,
            phi_project_slk=cfg.phi_project_slk,
            phi_trace_clamp=cfg.phi_trace_clamp,
            prior_bank=cfg.ffn_prior_bank,
            use_prior_bank=cfg.ffn_use_prior_bank,
            irrep_dims=cfg.ffn_irrep_dims,
            mask_self_attention=cfg.mask_self_attention,
            learnable_alpha=cfg.E_learnable_alpha,
            phi_natural_gradient=cfg.phi_natural_gradient,
            killing_center_reg=cfg.killing_center_reg,
            use_deq=cfg.use_deq,
            deq_neumann_terms=cfg.deq_neumann_terms,
            deq_include_phi=cfg.deq_include_phi,
            gauge_mode=cfg.gauge_mode,
            # Pass constant_omega from the attention module so the FFN's VFE
            # iterations use the same per-head Ω transport (manuscript Limit 2).
            # Without this, the FFN would use Ω=I, computing inconsistent
            # attention patterns relative to the attention sublayer.
            constant_omega=self.attention.constant_omega,
            em_mode=cfg.em_mode,
            isotropic_covariance=cfg.isotropic_covariance,
            sigma_max=cfg.sigma_max,
            e_step_sigma_floor=cfg.e_step_sigma_floor,
            use_rope=cfg.use_rope,
            rope_base=cfg.rope_base,
            gauge_param=cfg.gauge_param,
            detach_phi=cfg.detach_phi,
            closed_form_e_step=getattr(cfg, 'closed_form_e_step', False),
            learnable_head_kappa=cfg.learnable_head_kappa,
            n_picard_steps=cfg.n_picard_steps,
            picard_trust_region=cfg.picard_trust_region,
            e_step_early_exit_tol=getattr(cfg, 'e_step_early_exit_tol', None),
            compile_vfe=cfg.compile_vfe,
            gradient_checkpoint_vfe=cfg.gradient_checkpoint_vfe,
            alpha_divergence=getattr(cfg, 'alpha_divergence', 1.0),
            enforce_orthogonal=cfg.enforce_orthogonal,
        )
        # EXPERIMENTAL: rope_full_gauge rotates Σ as well as μ in the KL.
        # Dispatch lives in the per-head VFE loop (_compute_multihead_vfe_gradients).
        # Tri-state mode {'off', 'vfe_only', 'both'} — see block_config.RopeFullGaugeMode.
        # FFN VFE-gradient helper fires for {'vfe_only', 'both'}.
        # Attention-side σ rotation fires only for 'both' (gated inside attention).
        _rope_mode = getattr(cfg, 'rope_full_gauge', 'off')
        self.ffn._rope_full_gauge_vfe = _rope_mode
        self.attention.rope_full_gauge_mode = _rope_mode

        # Active inference / EFE plumbing — delegated to active_inference.py.
        # Sets the 13 _ai_* instance attributes and initialises _prior_bank_ref.
        # The PriorBank reference itself is wired in later by the model via
        # wire_readout_references() using __dict__ assignment.
        configure_ffn_active_inference(self.ffn, cfg)

        # =====================================================================
        # Share per-head learnable κ between attention sublayer and VFE FFN
        # =====================================================================
        # κ is the single temperature in β_ij = softmax(−KL/(κ·√d_h)).  Under
        # the manuscript's framework β is the posterior p(j|i), a single
        # physical quantity.  Historically the attention sublayer and the
        # VFE FFN each owned their own nn.Parameter, so gradient descent
        # could drive them apart and the two sublayers would end up
        # describing different posteriors.
        #
        # Solution: store a reference to the attention *Module* (not to its
        # tensors) on the FFN via __dict__.  This serves two purposes:
        #
        # 1. **Bypasses nn.Module.__setattr__** — direct attribute assignment
        #    would auto-register the attention sublayer as a child module of
        #    the FFN, causing its parameters to be double-counted in the
        #    optimizer.  The __dict__ bypass is the same pattern that
        #    active_inference.py uses for _prior_bank_ref.
        #
        # 2. **Survives .to(device)** — PyTorch's Module._apply walks the
        #    module tree and REPLACES every Parameter/buffer with a fresh
        #    tensor on the target device.  The Module object itself is
        #    never replaced.  By storing a reference to the Module and
        #    looking up its `.log_kappa_per_head` attribute at access time
        #    (rather than caching the tensor at wiring time), every call
        #    resolves to the current device-resident tensor.  An earlier
        #    version of this fix cached the tensor directly and broke on
        #    CUDA because the cache still pointed to the CPU-resident
        #    tensor after model.to('cuda').
        #
        # _get_kappa_h in variational_ffn.py reads through the reference
        # via `ref.log_kappa_per_head` / `ref._kappa_init` when the block
        # has wired one up; otherwise it falls back to the FFN's local
        # safety-net parameter (created for standalone unit-test use).
        if cfg.learnable_head_kappa and getattr(self.attention, 'log_kappa_per_head', None) is not None:
            # Drop the FFN's own parameter and buffer from its registered
            # dicts so the optimizer does not see a duplicate.  The local
            # attribute lookup would otherwise still find them on the FFN.
            if 'log_kappa_per_head' in self.ffn._parameters:
                del self.ffn._parameters['log_kappa_per_head']
            if '_kappa_init' in self.ffn._buffers:
                del self.ffn._buffers['_kappa_init']
            # Store a plain Python reference to the attention *Module*.
            # The reference survives .to(device) because the Module object
            # itself is never replaced; only its internal tensors are.
            self.ffn.__dict__['_kappa_attn_ref'] = self.attention
            # Set the direct attributes to None so a naive
            # `ffn.log_kappa_per_head` access (if any test does one)
            # gets None rather than an AttributeError.
            self.ffn.__dict__['log_kappa_per_head'] = None
            self.ffn.__dict__['_kappa_init'] = None

        self.norm2 = _make_norm(cfg.norm_type, cfg.embed_dim)

        # =====================================================================
        # Non-Flat Gauge Transport (optional)
        # =====================================================================
        self.non_flat_transport = cfg.non_flat_transport
        self.cocycle_relaxation = cfg.cocycle_relaxation
        self.holonomy_penalty = cfg.holonomy_penalty
        if cfg.non_flat_transport:
            if cfg.generators is None:
                raise ValueError(
                    "non_flat_transport=True requires cfg.generators to be "
                    "populated; got None."
                )
            # Per-head construction: partition the global (n_gen_total, K, K)
            # generator tensor by irrep block, then build one GaugeConnection
            # per head sized for that head's sub-fiber.  This matches the
            # block-diagonal structure of the transport operator — head-h's
            # connection sees head-h's μ features only and produces
            # coefficients for head-h's generators only.
            try:
                per_head_gens = partition_generators_by_block(
                    cfg.generators, self.attention.irrep_dims,
                )
                self.gauge_connection = PerHeadGaugeConnection(
                    irrep_dims=self.attention.irrep_dims,
                    per_head_generators=per_head_gens,
                    connection_type=cfg.connection_type,
                    hidden_dim=cfg.connection_hidden_dim,
                    init_scale=cfg.connection_init_scale,
                )
                self._connection_mode = 'per_head'
            except ValueError as exc:
                # Shared multi-irrep generators (e.g. SO(N) with mult >= 2 on a
                # non-scalar irrep, where one G_fund is replicated across every
                # multiplicity copy) cannot localize to a single irrep block —
                # per-block Frobenius mass is at most 1/mult, below
                # localization_threshold=0.999. Fall back to a single global
                # GaugeConnection over the full embed_dim, matching the
                # pre-per-head-split behaviour. The block-diagonal output shape
                # (B, N, N, n_gen_total) is identical between the per-head and
                # global paths, so downstream compute_transport_operators is
                # unchanged.
                import warnings
                warnings.warn(
                    f"PerHeadGaugeConnection: generators do not partition into "
                    f"block-localized supports ({exc}). Falling back to a single "
                    f"global GaugeConnection over embed_dim={cfg.embed_dim}. "
                    f"This typically happens with SO(N) multi-irrep specs where "
                    f"the same N(N-1)/2 generators are replicated across every "
                    f"irrep multiplicity.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                n_gen_total = cfg.generators.shape[0]
                self.gauge_connection = GaugeConnection(
                    d_head=cfg.embed_dim,
                    n_gen=n_gen_total,
                    connection_type=cfg.connection_type,
                    hidden_dim=cfg.connection_hidden_dim,
                    init_scale=cfg.connection_init_scale,
                )
                self._connection_mode = 'global'
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
        cached_head_transports: Optional[list] = None,
        omega: Optional[torch.Tensor] = None,
        sigma_prior: Optional[torch.Tensor] = None,
        *,
        return_attention: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]],
    ]:
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
        # When skip_attention=True, the VFE E-step IS the entire block:
        # it computes its own β internally, so the separate attention sublayer
        # is redundant. Skip it and go straight to VFE gradients.

        beta = None
        kl_matrix = None  # Set by attention sublayer; stays None when skip_attention=True
        delta_ij = None  # Non-flat connection (frozen E-step constant when passed to FFN)
        _shared_bep = None  # Shared block exp pairs for attention + FFN
        if not self.skip_attention:
            # Pre-layer normalization on means
            # MahalanobisNorm requires sigma; LayerNorm/RMSNorm ignore it.
            mu_normalized = self.norm1(mu_q, sigma_q) if isinstance(self.norm1, MahalanobisNorm) else self.norm1(mu_q)

            # Non-flat transport: compute edge-local connection δ_ij from the
            # pre-iteration μ and hold it fixed for the entire E-step.
            # This is intentional: re-computing δ(μ) per VFE iteration would
            # create a coupled fixed-point (δ depends on μ, μ depends on δ)
            # that complicates convergence. The frozen snapshot is a standard
            # first-order approximation analogous to expectation-propagation.
            if self.non_flat_transport and self.gauge_connection is not None and cached_head_transports is None:
                from transformer.core.transport_ops import compute_transport_operators
                delta_ij = self.gauge_connection(mu_normalized, mu_normalized)  # (B, N, N, n_gen)
                transport = compute_transport_operators(
                    phi, generators,
                    gauge_mode='learned',
                    connection_delta=delta_ij,
                    cocycle_relaxation=self.cocycle_relaxation,
                )
                # Split full Omega into per-head cached transports
                Omega_full = transport['Omega']  # (B, N, N, K, K)
                exp_phi_full = transport['exp_phi']  # (B, N, K, K)
                # Store exp_delta for holonomy penalty (if configured)
                self._last_exp_delta = transport.get('exp_delta')
                irrep_dims = self.attention.irrep_dims
                cached_head_transports = []
                dim_start = 0
                for d in irrep_dims:
                    # Include 'exp_phi' so downstream consumers (notably the
                    # gauge_covariant_ridge branches in attention.aggregate_messages)
                    # can build the covariant ridge eps*(g·g^T). Without it, those
                    # branches silently fall back to eps*I and the opt-in becomes
                    # a no-op under non_flat_transport=True.
                    cached_head_transports.append({
                        'Omega': Omega_full[:, :, :, dim_start:dim_start+d, dim_start:dim_start+d],
                        'exp_phi': exp_phi_full[:, :, dim_start:dim_start+d, dim_start:dim_start+d],
                    })
                    dim_start += d

            # Multi-head attention (gauge-theoretic!)
            # For direct omega mode: build per-head cached transports from omega blocks
            # so the attention sublayer uses Omega_h / Omega_h_inv instead of matrix_exp.
            if omega is not None and getattr(self.ffn, 'gauge_param', 'phi') == 'omega' and cached_head_transports is None:
                # Build per-head (omega_h, omega_h_inv) pairs using per-block inv
                # (avoids full K×K inv when omega is block-diagonal).
                # Ridge regularization + pinv fallback match the per-token
                # pattern in transport_ops.compute_transport_operators_direct
                # (line ~462).  Raw torch.linalg.inv on a near-singular GL(K)
                # block silently produces NaN that poisons the entire forward
                # pass.  Since omega_h is GL+ (not SPD), Cholesky does not
                # apply — we stick with LU-based inv + pinv fallback.
                irrep_dims = self.attention.irrep_dims
                cached_head_transports = []
                block_start = 0
                _omega_ridge = 1e-6
                for d_h in irrep_dims:
                    omega_h = omega[:, :, block_start:block_start+d_h, block_start:block_start+d_h]
                    _eye_dh = torch.eye(d_h, device=omega_h.device, dtype=omega_h.dtype)
                    omega_h_reg = omega_h + _omega_ridge * _eye_dh
                    try:
                        omega_h_inv = torch.linalg.inv(omega_h_reg)  # (B, N, d_h, d_h)
                    except (torch.linalg.LinAlgError, RuntimeError):
                        omega_h_inv = torch.linalg.pinv(omega_h_reg)
                    cached_head_transports.append({
                        'exp_phi': omega_h,
                        'exp_neg_phi': omega_h_inv,
                    })
                    block_start += d_h

            # SHARED TRANSPORT: Compute block exp pairs ONCE for both attention
            # and FFN. Both sublayers use the same phi, so computing independently
            # wastes 2× matrix_exp per block. Compute here, convert to
            # cached_head_transports for attention and pass directly to FFN.
            if (cached_head_transports is None
                    and self.attention.gauge_mode not in ('trivial', 'constant')
                    and self.attention.irrep_dims is not None):
                _skew = getattr(self.attention, '_generators_are_skew', False)
                _shared_bep = fused_block_matrix_exp_pairs(
                    phi, generators, self.attention.irrep_dims,
                    enforce_orthogonal=self.attention.enforce_orthogonal,
                    skew_symmetric=_skew,
                )
                # Build cached_head_transports for attention from shared pairs
                cached_head_transports = [
                    {'exp_phi': bep[0], 'exp_neg_phi': bep[1]}
                    for bep in _shared_bep
                ]
            elif (cached_head_transports is not None
                  and _shared_bep is None
                  and self.attention.gauge_mode not in ('trivial', 'constant')):
                # Extract BEP from incoming cached_head_transports (e.g., from
                # embedding cache in model._embed_and_prepare). The FFN needs
                # the (exp_phi, exp_neg_phi) tuple format, not the dict format.
                _try_bep = []
                _bep_ok = True
                for cht in cached_head_transports:
                    if 'exp_phi' in cht and 'exp_neg_phi' in cht and 'Omega' not in cht:
                        _try_bep.append((cht['exp_phi'], cht['exp_neg_phi']))
                    else:
                        _bep_ok = False
                        break
                if _bep_ok and _try_bep:
                    _shared_bep = _try_bep

            recorder = get_global_recorder()
            recording_attention = recorder is not None and recorder.enabled and recorder.record_attention
            # Request attention weights for trajectory recording or when the caller
            # requests them via return_attention=True (training path).
            need_attention_output = recording_attention or return_attention

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

            # Store mu_attn for optional post-call diagnostics
            # (model.forward_with_attention reads block._last_mu_attn when
            # _collect_layer_diagnostics=True to avoid re-computing it externally).
            self._last_mu_attn = mu_attn

            # Residual connection (optional for pure VFE).
            #
            # As of edits_2026-04-08.md Round 3, the residual form is
            # selected by self.residual_type.  The default is 'additive'
            # (plain mu_q + mu_attn, matching the 71-PPL TransformerOld
            # baseline).  The 'delta' form below was introduced by
            # 2026-04-07 audit Fix #1 / Fix #20 — see the reasoning below
            # — and is kept as an opt-in for configs where the audit
            # pathology actually applies (deep unnormalised stacks where
            # the residual stream would otherwise compound copies of the
            # pre-normalization input).
            #
            # Delta-extraction rationale (Fix #1 / #20, retained for the
            # 'delta' branch): mu_attn = Σ_j β_ij · Ω_ij · mu_normalized[j]
            # is an aggregated version of the normalized input, NOT a
            # zero-centered correction.  When self-attention dominates
            # (KL(q_i||q_i)=0 makes β_ii maximal whenever
            # mask_self_attention=False), mu_attn[i] ≈ mu_normalized[i],
            # so the plain residual mu_q + mu_attn would dump norm(mu)
            # into the residual stream each layer.  Extracting
            # (mu_attn - mu_normalized) so the residual stream accumulates
            # corrections rather than copies is mathematically
            # defensible — it's just empirically worse on the
            # single-layer K=90 GL(15) LayerNorm'd config because the
            # LayerNorm Jacobian subtraction cancels part of the identity
            # gradient path from the loss back to the embeddings.
            if self.use_residual:
                if self.residual_type == 'delta':
                    mu_q = mu_q + (mu_attn - mu_normalized)
                else:  # 'additive' 
                    mu_q = mu_q + mu_attn
            else:
                mu_q = mu_attn

            # Update covariances if evolving.
            # Delta extraction: sigma_attn is computed by aggregate_messages
            # from sigma_q (transport + weighted aggregation), so the delta is
            # sigma_attn - sigma_q.  Adding delta to sigma_q yields sigma_attn —
            # the same replacement semantics applied symmetrically to both
            # attention and FFN sublayers.  Old additive behavior (sigma_q +
            # sigma_attn) compounded multiplicatively across layers and pegged
            # sigma at sigma_max within the first forward pass.
            if self.evolve_sigma and sigma_attn is not None:
                sigma_q = sigma_attn.clamp(min=1e-4, max=self.sigma_max)

        # =====================================================================
        # 2. VFE E-step (with optional Pre-Norm + Residual)
        # =====================================================================
        # When skip_attention=True, this is the ONLY sublayer: the VFE gradient
        # computes β internally and handles all cross-position communication.

        if isinstance(self.norm2 if not self.skip_attention else self.norm1, MahalanobisNorm):
            mu_normalized = self.norm2(mu_q, sigma_q) if not self.skip_attention else self.norm1(mu_q, sigma_q)
        else:
            mu_normalized = self.norm2(mu_q) if not self.skip_attention else self.norm1(mu_q)

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
            omega=omega,
            sigma_prior=sigma_prior,
            connection_delta=delta_ij,
            cocycle_relaxation=self.cocycle_relaxation,
            precomputed_block_exp_pairs=_shared_bep,
        )

        # Update covariances from FFN if evolving.
        # Delta extraction (analogous to mu residual at line 516):
        # The FFN receives sigma_q directly (no normalization), so delta =
        # sigma_ffn - sigma_q.  Adding delta to the residual stream yields
        # sigma_ffn — preventing the double-counting inflation where the old
        # additive path (sigma_q + sigma_ffn ≈ 2σ) pegged sigma at sigma_max
        # within the first forward pass for multi-layer configs.
        if self.evolve_sigma and sigma_ffn is not None:
            sigma_q = sigma_ffn.clamp(min=1e-4, max=self.sigma_max)

        # Per-layer sigma diagnostics (opt-in via model._debug_sigma = True)
        if getattr(self, '_debug_sigma', False) and sigma_q is not None:
            _sd = sigma_q.detach()
            _eps = 1e-6
            if _sd.dim() == 3:
                _cond = (_sd.max(dim=-1).values / _sd.min(dim=-1).values.clamp(min=_eps)).mean().item()
            else:
                _diag_vals = torch.diagonal(_sd, dim1=-2, dim2=-1)
                _cond = (_diag_vals.max(dim=-1).values / _diag_vals.min(dim=-1).values.clamp(min=_eps)).mean().item()
            if not hasattr(self, '_sigma_log'):
                self._sigma_log = []
            self._sigma_log.append({
                'sigma_mean': _sd.mean().item(),
                'sigma_max': _sd.max().item(),
                'sigma_min': _sd.min().item(),
                'sigma_condition': _cond,
            })

        # Propagate evolved omega from FFN E-step (gauge_param='omega').
        # Without this, omega evolution is lost between layers.
        evolved_omega = getattr(self.ffn, '_last_omega', None)
        if evolved_omega is not None:
            self._last_evolved_omega = evolved_omega

        # Store mu_ffn for optional post-call diagnostics
        # (model.forward_with_attention reads block._last_mu_ffn when
        # _collect_layer_diagnostics=True to avoid re-computing it externally).
        self._last_mu_ffn = mu_ffn

        # Residual connection (optional for pure VFE).
        # Delta-extraction rationale (retained for the 'delta' branch):
        # the VFE FFN returns the full evolved state
        # (mu_normalized + correction), not just the correction.
        # Extracting (mu_ffn - mu_normalized) so the residual stream
        # accumulates corrections rather than copies of normalized
        # inputs is mathematically defensible for deep unnormalised
        # stacks, but empirically worse for single-layer LayerNorm'd
        # configs — the LayerNorm Jacobian subtraction weakens the
        # loss→embedding gradient signal.
        if self.use_residual:
            if self.residual_type == 'delta':
                mu_q = mu_q + (mu_ffn - mu_normalized)
            else:  # 'additive' (default, matches 71-PPL baseline)
                mu_q = mu_q + mu_ffn
        else:
            mu_q = mu_ffn

        if return_attention:
            return mu_q, sigma_q, phi_out, beta, kl_matrix
        return mu_q, sigma_q, phi_out

    def extra_repr(self) -> str:
        parts = [
            f"embed_dim={self.embed_dim}",
            f"evolve_sigma={self.evolve_sigma}",
            f"evolve_phi={self.evolve_phi}",
            f"diagonal_covariance={self.diagonal_covariance}",
            f"ffn_mode={self.ffn_mode!r}",
            f"norm_type={self.norm_type!r}",
            f"use_residual={self.use_residual}",
            f"skip_attention={self.skip_attention}",
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
    Beliefs (μ, Σ, φ) flow through all layers. A final LayerNorm is applied to μ.
    The E-step does not see target tokens — the observation likelihood is
    provided by the outer CE loss in compute_free_energy_loss.

    Supports gradient checkpointing (cfg.gradient_checkpointing) for ~60% memory
    savings at ~30% extra compute.
    """

    def __init__(self, cfg: BlockConfig):
        super().__init__()
        self.n_layers = cfg.n_layers
        self.gradient_checkpointing = getattr(cfg, 'gradient_checkpointing', False)
        self.hierarchical_priors = getattr(cfg, 'hierarchical_priors', True)
        # Stored for the hierarchical cascade: when True, the posterior μ of
        # layer l becomes the prior μ of layer l+1 WITHOUT detach, so
        # cross-layer gradients reach early-layer embeddings.  See the
        # comment on the detach site below in forward().
        self.em_mode = cfg.em_mode
        self.amortized_inference = cfg.amortized_inference  # property, from em_mode
        self.em_phi_mode = cfg.em_phi_mode  # property, from em_mode

        self.blocks =nn.ModuleList([
            GaugeTransformerBlock(cfg)
            for _ in range(cfg.n_layers)
        ])

        # Final normalization (optional for pure VFE)
        self.final_norm = _make_norm(cfg.norm_type, cfg.embed_dim)

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
        omega: Optional[torch.Tensor] = None,
        sigma_prior: Optional[torch.Tensor] = None,
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

        Returns:
            mu_q: Final means (B, N, K) after final LayerNorm.
            sigma_q: Final covariances — same shape as input.
            phi: Final gauge frames (B, N, phi_dim).
            intermediates: List of {'layer', 'mu', 'sigma', 'phi'} dicts if
                return_intermediates=True, else None.
        """
        intermediates = [] if return_intermediates else None

        # Get trajectory recorder
        recorder = get_global_recorder()
        recording_enabled = recorder is not None and recorder.enabled

        n_blocks = len(self.blocks)
        for layer_idx, block in enumerate(self.blocks):
            is_final = (layer_idx == n_blocks - 1)

            # Trajectory recording: start layer
            if recording_enabled:
                recorder.start_layer(layer_idx)
                recorder.record_layer_input(mu_q, sigma_q, phi)

            if self.gradient_checkpointing and self.training:
                # Gradient checkpointing: trade ~30% compute for ~60% memory savings.
                # Capture mu_prior/omega by value (default arg) so that
                # backward recomputation uses the correct per-layer values
                # when hierarchical_priors or omega evolution mutate them.
                def create_block_fn(blk, _mu_prior=mu_prior, _omega=omega):
                    def block_fn(mu, sigma, phi_arg):
                        return blk(
                            mu, sigma, phi_arg, generators, mask, _mu_prior,
                            token_ids=token_ids,
                            cached_head_transports=cached_head_transports,
                            omega=_omega,
                            sigma_prior=sigma_prior,
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
                    omega=omega,
                    sigma_prior=sigma_prior,
                )

            # Propagate evolved omega from E-step to next layer (gauge_param='omega').
            # Without this, each layer receives the original embedding omega,
            # discarding E-step omega evolution from previous layers.
            evolved_omega = getattr(block, '_last_evolved_omega', None)
            if evolved_omega is not None:
                omega = evolved_omega

            # Hierarchical priors: each layer's posterior μ becomes the next
            # layer's prior μ.  sigma_prior stays at the embedding value to
            # prevent progressive tightening (sigma cascade).
            #
            # EM modes: mu_q is already detached by the FFN EM boundary,
            # but the residual mu_q = mu_q_pre_ffn + mu_ffn still carries
            # gradients from the attention sublayer.  For principled EM,
            # fully detach so each layer's q* is frozen for the next layer.
            #
            # Amortized (default): keep mu_q ATTACHED so cross-layer
            # gradients reach early-layer embeddings (CLAUDE.md).
            if self.hierarchical_priors and not is_final:
                _em_active = self.em_phi_mode in ('E_phi_q', 'M_phi_p')
                if _em_active:
                    mu_prior = mu_q.detach()
                elif self.amortized_inference:
                    mu_prior = mu_q
                else:
                    mu_prior = mu_q.detach()

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
        mu_q = self.final_norm(mu_q, sigma_q) if isinstance(self.final_norm, MahalanobisNorm) else self.final_norm(mu_q)

        return mu_q, sigma_q, phi, intermediates