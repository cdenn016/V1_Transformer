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
from typing import Optional, Tuple, List

from transformer.core.block_config import BlockConfig

# Import our gauge attention
from transformer.core.attention import IrrepMultiHeadAttention

# Import VFE FFN directly (no wrapper)
from transformer.core.variational_ffn import VariationalFFNDynamic

# Import gauge connection for non-flat transport
from transformer.core.connection import GaugeConnection

# Import block-diagonal matrix exp for shared transport caching
from transformer.core.gauge_utils import fused_block_matrix_exp_pairs

# Trajectory tracking (optional)
try:
    from transformer.analysis.trajectory import get_global_recorder
    TRAJECTORY_TRACKING_AVAILABLE = True
except ImportError:
    TRAJECTORY_TRACKING_AVAILABLE = False
    def get_global_recorder():
        return None


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight

    def extra_repr(self) -> str:
        return f"{self.weight.shape[0]}, eps={self.eps}"


def _make_norm(norm_type: str, dim: int) -> nn.Module:
    """Factory for normalization layers.

    Args:
        norm_type: 'layernorm', 'rmsnorm', or 'none'
        dim: Normalized dimension (embed_dim).
    """
    if norm_type == 'layernorm':
        return nn.LayerNorm(dim)
    elif norm_type == 'rmsnorm':
        return RMSNorm(dim)
    elif norm_type == 'none':
        return nn.Identity()
    else:
        raise ValueError(f"Unknown norm_type: {norm_type!r}. Expected 'layernorm', 'rmsnorm', or 'none'.")


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
        self.sigma_residual = getattr(cfg, 'sigma_residual', False)
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
            irrep_dims_override=cfg.ffn_irrep_dims if (gauge_group == 'GLK' and cfg.cross_head_perm is not None) else None,
            use_rope=cfg.use_rope,
            rope_base=cfg.rope_base,
            sigma_aggregation=cfg.sigma_aggregation,
            learnable_head_kappa=cfg.learnable_head_kappa,
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
            prior_bank=cfg.ffn_prior_bank,
            use_prior_bank=cfg.ffn_use_prior_bank,
            irrep_dims=cfg.ffn_irrep_dims,
            mask_self_attention=cfg.mask_self_attention,
            learnable_alpha=cfg.E_learnable_alpha,
            multihead_vfe=cfg.multihead_vfe,
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
            amortized_inference=cfg.amortized_inference,
            isotropic_covariance=cfg.isotropic_covariance,
            obs_sigma_gradient=cfg.obs_sigma_gradient,
            obs_sigma_weight=cfg.obs_sigma_weight,
            sigma_max=cfg.sigma_max,
            e_step_sigma_floor=cfg.e_step_sigma_floor,
            use_rope=cfg.use_rope,
            rope_base=cfg.rope_base,
            gauge_param=cfg.gauge_param,
            detach_phi=cfg.detach_phi,
            implicit_em=cfg.implicit_em,
            closed_form_e_step=getattr(cfg, 'closed_form_e_step', False),
            learnable_head_kappa=cfg.learnable_head_kappa,
            n_picard_steps=cfg.n_picard_steps,
            picard_trust_region=cfg.picard_trust_region,
            compile_vfe=cfg.compile_vfe,
            gradient_checkpoint_vfe=cfg.gradient_checkpoint_vfe,
        )
        # EXPERIMENTAL: rope_full_gauge enables the framework-consistent
        # interpretation of RoPE (rotates Σ as well as μ in the KL).
        # NOTE: the rope_full_gauge dispatch lives in the MULTI-HEAD loop
        # only.  When multihead_vfe=False, the flag is silently ignored by
        # the single-β branch which dispatches straight to the analytic
        # fused path.  Warn the user so the silent fallback is visible.
        self.ffn._rope_full_gauge_vfe = getattr(cfg, 'rope_full_gauge', False)
        if self.ffn._rope_full_gauge_vfe and not getattr(cfg, 'multihead_vfe', True):
            import warnings as _w
            _w.warn(
                "rope_full_gauge=True has no effect when multihead_vfe=False — "
                "the rope_full_gauge dispatch is only implemented in the "
                "per-head VFE loop.  The single-β path will fall back to the "
                "standard analytical fused gradient (which has the RoPE chain "
                "rule fix but rotates only μ, not Σ).  Enable multihead_vfe=True "
                "to use the experimental rope_full_gauge path.",
                UserWarning,
                stacklevel=2,
            )

        # Active inference / EFE plumbing.  Master toggle (active_inference)
        # gates the entire path; weights only take effect when toggle is True.
        # When enabled, the E-step adds the pragmatic and epistemic EFE terms
        # via PriorBank.decode().  The PriorBank reference itself is set later
        # by the model in __init__ via __dict__ assignment (avoids nn.Module
        # sub-module auto-registration of an already-owned module).
        self.ffn._ai_enabled = getattr(cfg, 'active_inference', False)
        self.ffn._ai_pragmatic_weight = getattr(cfg, 'active_inference_pragmatic_weight', 1.0)
        self.ffn._ai_epistemic_weight = getattr(cfg, 'active_inference_epistemic_weight', 0.5)
        self.ffn._ai_epistemic_samples = getattr(cfg, 'active_inference_epistemic_samples', 4)
        self.ffn._ai_decode_tau = getattr(cfg, 'active_inference_decode_tau', 1.0)
        self.ffn._ai_trust_region = getattr(cfg, 'active_inference_trust_region', 0.5)
        self.ffn._ai_lr = getattr(cfg, 'active_inference_lr', 1.0)
        # _prior_bank_ref defaults to None; the model wires it in after the
        # full module hierarchy is constructed.
        self.ffn.__dict__.setdefault('_prior_bank_ref', None)

        self.norm2 = _make_norm(cfg.norm_type, cfg.embed_dim)

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
        sigma_prior: Optional[torch.Tensor] = None,
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
        # When skip_attention=True, the VFE E-step IS the entire block:
        # it computes its own β internally, so the separate attention sublayer
        # is redundant. Skip it and go straight to VFE gradients.

        beta = None
        delta_ij = None  # Non-flat connection (frozen E-step constant when passed to FFN)
        _shared_bep = None  # Shared block exp pairs for attention + FFN
        if not self.skip_attention:
            # Pre-layer normalization on means
            mu_normalized = self.norm1(mu_q)

            # Non-flat transport: compute edge-local connection δ_ij and inject
            # into cached transport so attention sees the modified Ω_ij.
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
                # Store exp_delta for holonomy penalty (if configured)
                self._last_exp_delta = transport.get('exp_delta')
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

            recorder = get_global_recorder() if TRAJECTORY_TRACKING_AVAILABLE else None
            recording_attention = recorder is not None and recorder.enabled and recorder.record_attention
            # Request attention weights for trajectory recording. The FFN recomputes
            # its own beta internally (the beta arg is unused), so this is only needed
            # when attention diagnostics are being collected.
            need_attention_output = recording_attention

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
            targets=targets,
            W_out=W_out,
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

        # Residual connection (optional for pure VFE)
        # The VFE FFN returns the full evolved state (mu_normalized + delta),
        # not just the correction delta. Extract the delta so the residual
        # stream accumulates corrections, not copies of normalized inputs.
        # Without this, each layer dumps norm(mu) into the residual, burying
        # the VFE correction and preventing effective depth scaling.
        if self.use_residual:
            mu_q = mu_q + (mu_ffn - mu_normalized)
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
        self.hierarchical_priors = getattr(cfg, 'hierarchical_priors', True)

        self.blocks = nn.ModuleList([
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
        targets: Optional[torch.Tensor] = None,
        W_out: Optional[torch.Tensor] = None,
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
                # Skip final layer to preserve targets/W_out gradient flow.
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
                    targets=targets if is_final else None,
                    W_out=W_out if is_final else None,
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
            # prevent progressive tightening (sigma cascade).  The .detach()
            # preserves proper EM: the prior is an E-step constant.
            if self.hierarchical_priors and not is_final:
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
        mu_q = self.final_norm(mu_q)

        return mu_q, sigma_q, phi, intermediates
