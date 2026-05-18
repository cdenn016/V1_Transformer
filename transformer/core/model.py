"""
Complete Gauge-Theoretic Language Model (0D Architecture)
==========================================================

Full transformer language model using gauge theory and variational free energy.

Architecture:
    token_ids → GaugeTokenEmbedding → (μ, Σ, φ) → Positional Encoding
    → N × GaugeTransformerBlock (KL-attention + VFE E-step FFN)
    → LayerNorm(μ) → Output Projection → logits

Key Innovation: Attention via KL divergence on statistical manifold with gauge
    transport — no learned W_Q, W_K, W_V matrices. Beliefs (μ, Σ, φ) evolve
    through variational inference rather than neural network layers.

Gauge groups: SO(3), SO(N), GL(K) — determined by generator shape.
Configuration: flat dict → BlockConfig (see block_config.py for all 60+ params).
"""

# Suppress noisy warnings BEFORE torch import (torch may trigger imports)
import warnings
warnings.filterwarnings("ignore", message="Failed to find cuobjdump", module="triton")
warnings.filterwarnings("ignore", message="Failed to find nvdisasm", module="triton")
warnings.filterwarnings("ignore", message="CUDA path could not be detected", module="cupy")

import copy
import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List, Union
import numpy as np

logger = logging.getLogger(__name__)

# Import our components
from transformer.core.embeddings import GaugeTokenEmbedding, GaugePositionalEncoding
from transformer.core.blocks import GaugeTransformerStack, RMSNorm, MahalanobisNorm
from transformer.core.block_config import BlockConfig
from transformer.core.active_inference import wire_readout_references

# Trajectory tracking (core-side protocol — analysis layer registers via set_global_recorder)
from transformer.core.vfe_utils import get_global_recorder

# Try to import generators (fallback to random if unavailable)
try:
    from math_utils.generators import (
        generate_so3_generators,
        generate_soN_generators,
        generate_multi_irrep_generators,
        generate_multi_irrep_soN_generators,
        generate_glK_generators,
        generate_glK_multihead_generators,
        generate_glK_cross_head_generators,
        merge_coupled_heads,
        reorder_cross_head_generators,
        _dedup_cross_couplings,
        validate_generator_closure,
        close_under_brackets,
    )
    GENERATORS_AVAILABLE = True
except ImportError:
    GENERATORS_AVAILABLE = False


class GaugeTransformerLM(nn.Module):
    """
    Complete gauge-theoretic language model.

    Architecture Flow:
        token_ids → (μ, Σ, φ) → Positional Encoding → N × Transformer Blocks
        → LayerNorm(μ) → W_out @ μ → logits

    Components:
        1. GaugeTokenEmbedding: tokens → beliefs (μ, Σ, φ) with optional sign-flip reflections
        2. GaugePositionalEncoding: agent-index encoding in Lie algebra (so(3)/so(N)/gl(K))
        3. GaugeTransformerStack: N layers of KL-attention + VFE E-step FFN
        4. Output projection: μ → logits over vocabulary (optionally tied to embeddings)

    Gauge groups: SO(3), SO(N), GL(K) — determined by generators shape.
    Belief state: μ ∈ ℝ^K (means), Σ ∈ SPD(K) (covariances), φ ∈ g (gauge frames).
    All configuration flows through a flat config dict → BlockConfig.
    """

    def __init__(self, config: Dict):
        """
        Initialize gauge transformer language model.

        Args:
            config: Flat dictionary with all hyperparameters. Required keys:
                - vocab_size: Vocabulary size V
                - embed_dim: Belief dimension K = Σ(mult_ℓ × dim_ℓ)
                - n_layers: Number of transformer blocks
                - irrep_spec: [(label, multiplicity, dim), ...] defining head structure
                - hidden_dim: FFN hidden dimension (kept for config compat)
                - max_seq_len: Maximum sequence length
                - kappa_beta: Attention/VFE temperature τ

                Optional keys (with defaults):
                - pos_encoding_mode: 'none' (default) | 'learned' | 'sinusoidal'
                - evolve_sigma: Update covariances (default True)
                - evolve_phi: Update gauge frames (default True)
                - tie_embeddings: Tie input/output projection weights (default True)
                - diagonal_covariance: Σ as (B,N,K) diagonal (default False)
                - exact_diagonal_transport: Lift diagonal for exact transport (default False)
                - gauge_mode: 'learned' | 'trivial' | 'constant' (default 'learned')
                - phi_dim: Gauge frame dimension (default 3)
                - use_rope: Rotary position embeddings (default False)
                - non_flat_transport: Edge-local connection (default False)
                - use_prior_bank: Token-dependent priors (default False)
                See BlockConfig for the full list of 60+ parameters.
        """
        super().__init__()
        # Snapshot the global torch RNG state at entry and restore on exit
        # so that model construction is RNG-neutral for the surrounding
        # stream. Without this, optional submodules (e.g. attention.W_O
        # under use_output_projection=True) consume extra global RNG draws
        # during construction and during self.apply(self._init_weights),
        # leaving downstream consumers (e.g. DataLoader shuffling) at a
        # different RNG position depending on which optional flags are
        # set. Parameters inside the model remain deterministic from the
        # user seed; only the surrounding global stream is held invariant.
        _rng_entry_cpu = torch.random.get_rng_state()
        _rng_entry_cuda = (
            [torch.cuda.get_rng_state(i) for i in range(torch.cuda.device_count())]
            if torch.cuda.is_available() else []
        )
        # Snapshot the user's config dict so that (a) later edits to the
        # original dict don't silently change model behaviour, and (b) the
        # mutation at `config['gauge_param'] = gauge_param` below does not
        # leak into the caller's namespace.  The previous behaviour
        # (`self.config = config` storing a live reference) meant that
        # reusing the same config dict for multiple models would cross-
        # contaminate, and users could not safely introspect their
        # "original" config after construction.
        config = copy.deepcopy(config)
        self.config = config

        # Initialize cross-head coupling attributes (may be set later in gauge setup)
        self._cross_head_perm = None
        self._super_block_dims = None

        # Extract config
        vocab_size = config['vocab_size']
        embed_dim = config['embed_dim']
        n_layers = config['n_layers']
        irrep_spec = config['irrep_spec']
        max_seq_len = config['max_seq_len']
        kappa_beta = config['kappa_beta']
        pos_mode = config.get('pos_encoding_mode', 'none')  # Default: no position in gauge space
        evolve_sigma = config.get('evolve_sigma', True)
        evolve_phi = config.get('evolve_phi', True)
        evolve_phi_e_step = config.get('evolve_phi_e_step', False)  # Update φ during E-step iterations
        tie_embeddings = config.get('tie_embeddings', True)

        # VFE FFN config (ffn_alpha / ffn_kappa / ffn_n_iterations / ffn_learnable_lr /
        # ffn_lambda_belief / ffn_update_sigma were previously unpacked here but are
        # now read directly by BlockConfig.from_config from the same config dict.)
        ffn_mode = config.get('ffn_mode', 'VFE_dynamic')

        # PriorBank: token-dependent priors for principled encode/decode
        use_prior_bank = config.get('use_prior_bank', False)

        # Gauge-fixed priors (for gauge covariance)
        gauge_fixed_priors = config.get('gauge_fixed_priors', False)

        # Diagonal covariance mode (memory optimization)
        diagonal_covariance = config.get('diagonal_covariance', False)

        # Isotropic covariance: Σ = σ²I (Limit 1 → KL reduces to squared Euclidean)
        isotropic_covariance = config.get('isotropic_covariance', False)
        if isotropic_covariance:
            logger.info("[INFO] Isotropic covariance mode: Sigma = sigma^2*I (Limit 1 -- KL -> squared Euclidean)")
            if not diagonal_covariance:
                logger.info("       (Forcing diagonal_covariance=True for isotropic mode)")
                diagonal_covariance = True

        # Positional embedding added to μ (like standard transformers)
        # DEFAULT: True - position in μ, not gauge frame φ. 
        use_positional_embedding = config.get('use_positional_embedding', False)

        # Position encoding scale (for φ gauge frame encoding)
        pos_encoding_scale = config.get('pos_encoding_scale', 0.1)

        # ALiBi-style positional bias (adds slope*(i-j) to attention logits)
        # Negative values create recency bias (attend more to nearby tokens)
        alibi_slope = config.get('alibi_slope', None)

        # Store evolve_phi for cross-layer transport caching optimization
        self.evolve_phi = evolve_phi

        # Sparse attention config
        self.attention_pattern = config.get('attention_pattern', 'full')
        self.attention_window = config.get('attention_window', 64)

        # =================================================================
        # Gauge Group, Mode, and Parameterization
        # =================================================================
        gauge_group, gauge_dim, gauge_mode, gauge_param, evolve_phi, evolve_phi_e_step = (
            self._resolve_gauge_mode(config, evolve_phi, evolve_phi_e_step, use_prior_bank, isotropic_covariance)
        )
        self.gauge_group = gauge_group
        self.gauge_dim = gauge_dim
        self.gauge_param = gauge_param

        # Compute omega_head_dims from irrep_spec for direct omega path
        if gauge_param == 'omega':
            self.omega_head_dims = [dim for _, mult, dim in irrep_spec for _ in range(mult)]
        else:
            self.omega_head_dims = None

        # =================================================================
        # Cross-Head Coupling (sparse off-diagonal gauge mixing)
        # =================================================================
        cross_couplings = config.get('cross_couplings', [])

        # Canonicalize: drop exact (a,b) duplicates so a duplicated entry does
        # not silently inflate phi_dim and produce a rank-deficient basis.
        # Distinct orientations (a,b) vs (b,a) are preserved (directional).
        if cross_couplings:
            cross_couplings, _n_dropped = _dedup_cross_couplings(list(cross_couplings))

        if cross_couplings and gauge_mode == 'trivial':
            logger.warning("cross_couplings have no effect with gauge_mode='trivial' (Omega=I, no mixing)")

        # Hard-error on the silent coordinate-frame mismatch combination.
        # When use_block_diagonal_kl=False, _build_generators still reorders
        # generators into super-block coordinates (via reorder_cross_head_generators),
        # but BlockConfig.ffn_irrep_dims is set to None and attention.py then
        # falls back to [d_head] * n_heads. The two coordinate frames disagree
        # silently; the only safe combinations are (a) cross_couplings empty,
        # or (b) use_block_diagonal_kl=True.
        if cross_couplings and not config.get('use_block_diagonal_kl', True):
            raise ValueError(
                "cross_couplings is non-empty but use_block_diagonal_kl=False. "
                "These options are incompatible: the cross-head builder reorders "
                "generators into super-block coordinates that the per-head KL "
                "fallback path cannot honor, producing a silent coordinate-frame "
                "mismatch between the gauge basis and attention's block layout. "
                "Either set use_block_diagonal_kl=True or remove cross_couplings."
            )

        # Compute phi dimension and build Lie algebra generators
        self.phi_dim = self._compute_phi_dim(gauge_group, gauge_dim, embed_dim, irrep_spec, cross_couplings)
        generators = self._build_generators(gauge_group, gauge_dim, embed_dim, irrep_spec, cross_couplings)

        # Optional Lie-closure handling for cross-head coupling.
        # Default: validate post-reorder basis and warn (non-fatal) if the basis
        #          is not closed under the matrix commutator. The bracket /
        #          BCH machinery in math_utils.generators silently projects
        #          onto the basis; if the user is unaware their basis is open,
        #          downstream BCH composition becomes a projected approximation.
        # Opt-in:  auto_close_cross_head_basis=True replaces the basis with
        #          its bracket closure, yielding a true Lie subalgebra. This
        #          changes phi_dim and breaks checkpoint compatibility, hence
        #          off by default.
        if (
            cross_couplings
            and gauge_group == 'GLK'
            and gauge_mode != 'trivial'
        ):
            auto_close = bool(config.get('auto_close_cross_head_basis', False))
            do_validate = bool(config.get('validate_cross_head_closure', True))
            if auto_close:
                import numpy as _np
                G_closed, close_info = close_under_brackets(
                    _np.asarray(generators)
                )
                logger.info(
                    "auto_close_cross_head_basis=True: closed basis %d -> %d "
                    "generators in %d iter(s) (converged=%s, hit_max_dim=%s).",
                    close_info['initial_dim'], close_info['final_dim'],
                    close_info['n_iters'], close_info['converged'],
                    close_info['hit_max_dim'],
                )
                if close_info['n_added'] > 0:
                    logger.warning(
                        "auto_close_cross_head_basis added %d new generators. "
                        "These can span across the user-supplied super-block "
                        "partition (e.g. closing [(0,1),(1,2)] introduces an "
                        "E^{02} block that mixes the original super-blocks). "
                        "self._super_block_dims and _cross_head_perm still "
                        "reflect the pre-closure partition; downstream code that "
                        "assumes block-diagonal structure inside super-blocks may "
                        "now see non-zero off-block components. Use this flag "
                        "only when you have verified that the closed basis is "
                        "compatible with your block-diagonal KL configuration.",
                        close_info['n_added'],
                    )
                generators = G_closed
                self.phi_dim = int(generators.shape[0])
            elif do_validate:
                report = validate_generator_closure(generators)
                if not report['closed']:
                    logger.warning(
                        "cross-head generator basis is NOT closed under [.,.] "
                        "(max relative residual %.3e across %d/%d unordered pairs). "
                        "BCH/bracket composition on this basis silently projects "
                        "onto the span, producing an approximation rather than the "
                        "exact Lie composition. Worst offenders (a, b, residual): %s. "
                        "Set auto_close_cross_head_basis=True to obtain a true "
                        "Lie subalgebra (changes phi_dim and breaks checkpoint "
                        "compatibility), or set validate_cross_head_closure=False "
                        "to suppress this warning.",
                        report['max_residual'], report['n_offending_pairs'],
                        report['n_pairs'], report['offending_pairs'][:5],
                    )

        self.register_buffer(
            'generators',
            torch.from_numpy(generators).float()
        )

        # =================================================================
        # Embedding Layers, Position Encoding, and PriorBank
        # =================================================================
        # Compute irrep_dims early for (a) embedding block-diagonal matrix_exp
        # optimization under gauge_fixed_priors, and (b) per-block sl(K)
        # projection / trace clamp — which needs the block structure even
        # when gauge_fixed_priors=False so the traceless projection targets
        # each head's trace independently instead of collapsing to the full-K
        # trace.
        _need_block_dims_for_det_ctrl = (
            config.get('phi_project_slk', False)
            or config.get('phi_trace_clamp', None) is not None
        )
        _embed_irrep_dims = (
            self._get_effective_irrep_dims(irrep_spec)
            if (config.get('use_block_diagonal_kl', True) and gauge_fixed_priors)
               or _need_block_dims_for_det_ctrl
            else None
        )
        self._build_embeddings(
            config, vocab_size, embed_dim, irrep_spec, max_seq_len,
            pos_mode, pos_encoding_scale, gauge_mode, gauge_param,
            gauge_fixed_priors, diagonal_covariance, use_prior_bank,
            use_positional_embedding,
            irrep_dims=_embed_irrep_dims,
        )

        # =================================================================
        # Transformer Stack
        # =================================================================
        # Ensure gauge_param is in config for BlockConfig
        if 'gauge_param' not in config:
            config['gauge_param'] = gauge_param
        block_cfg = BlockConfig.from_config(
            config,
            generators=self.generators,
            prior_bank=self.prior_bank,
            ffn_irrep_dims=self._get_effective_irrep_dims(irrep_spec) if config.get('use_block_diagonal_kl', True) else None,
        )
        # Override derived values that model.py computes
        block_cfg.phi_dim = self.phi_dim
        block_cfg.attention_pattern = self.attention_pattern
        block_cfg.attention_window = self.attention_window
        block_cfg.alibi_slope = alibi_slope
        block_cfg.ffn_use_prior_bank = use_prior_bank

        self.transformer = GaugeTransformerStack(block_cfg)

        # =================================================================
        # Output Projection
        # =================================================================
        self.out_proj = nn.Linear(embed_dim, vocab_size, bias=False)

        # =================================================================
        # Active inference readout wiring — delegated to active_inference.py.
        # Must run AFTER both self.transformer and self.out_proj are created.
        # Plumbs PriorBank (or W_out fallback) into each block's FFN and
        # emits diagnostic warnings for incompatible configurations.
        # =================================================================
        wire_readout_references(self.transformer, self.prior_bank, self.out_proj, logger=logger)

        # =================================================================
        # Initialize Weights (BEFORE tying embeddings, so that the
        # embedding's calibrated init_std is not overwritten by out_proj's
        # std=0.02 initialization)
        # =================================================================
        self.apply(self._init_weights)

        # Re-seed out_proj.weight from a sub-generator derived from the
        # user seed but independent of self.apply's depth-first traversal
        # order. Any optional nn.Linear inside self.transformer (e.g.
        # attention.output_proj when use_output_projection=True) consumes
        # RNG draws before out_proj is reached, otherwise giving the
        # vocab head different initial values from the same user seed.
        _out_proj_seed = (torch.initial_seed() ^ 0xC0FFEE5A11D0E4) & 0xFFFFFFFFFFFFFFFF
        _out_proj_gen = torch.Generator(device=self.out_proj.weight.device)
        _out_proj_gen.manual_seed(_out_proj_seed)
        with torch.no_grad():
            self.out_proj.weight.normal_(mean=0.0, std=0.02, generator=_out_proj_gen)

        # Tie input/output embeddings (standard practice)
        # Note: Can't tie when PriorBank is active (it IS the shared encode/decode)
        # or when gauge_fixed_priors=True (no per-token embedding)
        if use_prior_bank:
            # PriorBank is the unified encode/decode layer — no tying needed
            if tie_embeddings:
                logger.info("tie_embeddings ignored: PriorBank serves as both encoder and decoder")
        elif tie_embeddings and not gauge_fixed_priors:
            self.out_proj.weight = self.token_embed.mu_embed.weight
        elif tie_embeddings and gauge_fixed_priors:
            logger.warning("tie_embeddings disabled because gauge_fixed_priors=True")

        # Per-layer diagnostics (set externally by trainer)
        self._collect_layer_diagnostics = False
        self._layer_diagnostics: list = []

        # VFE dynamics metrics collection (set externally by trainer)
        self._collect_dynamics_metrics = False
        self._dynamics_metrics_interval = 50  # Only compute every N forward calls
        self._dynamics_metrics_counter = 0

        # Cached causal mask (avoids torch.ones + torch.tril every forward pass)
        _causal = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer('_causal_mask', _causal, persistent=False)

        # Count parameters
        n_params = sum(p.numel() for p in self.parameters())
        logger.info(f"GaugeTransformerLM initialized: {n_params/1e6:.2f}M parameters")

        # Restore the entry-time global RNG state. See snapshot comment at
        # the top of __init__.
        torch.random.set_rng_state(_rng_entry_cpu)
        if _rng_entry_cuda:
            for _i, _s in enumerate(_rng_entry_cuda):
                torch.cuda.set_rng_state(_s, _i)

    # =========================================================================
    # Step 1: Extracted helper — cross-head permutation
    # =========================================================================

    def _apply_cross_head_perm(
        self,
        mu: torch.Tensor,
        sigma: Optional[torch.Tensor],
        device: torch.device,
        inverse: bool = False,
        omega: Optional[torch.Tensor] = None,
    ):
        r"""Apply (or invert) the cross-head dimension permutation.

        When GL(K) cross-head coupling is active, coupled heads are reordered
        so that their dimensions are contiguous in memory. This permutation must
        be applied to mu and sigma after embedding lookup and reversed before the
        vocabulary projection so that mu aligns with the generator block structure.

        The permutation is a pure index reordering — no gauge-covariance concern.

        Args:
            mu:      (B, N, K) belief means.
            sigma:   (B, N, K) diagonal or (B, N, K, K) full covariance, or None.
            device:  Target device for the permutation tensor.
            inverse: If True apply the inverse permutation (restore original order).
            omega:   Optional (B, N, K, K) direct gauge frames (gauge_param='omega').
                     When provided, the function returns a 3-tuple and applies the
                     sandwich permutation omega[..., perm, :][..., :, perm] so the
                     K-axes stay consistent with the permuted mu/sigma.

        Returns:
            (mu_permuted, sigma_permuted) if ``omega`` is None, else
            (mu_permuted, sigma_permuted, omega_permuted).
        """
        if getattr(self, '_cross_head_perm', None) is None:
            if omega is None:
                return mu, sigma
            return mu, sigma, omega

        if inverse:
            perm = self._inv_perm_tensor.to(device=device)
        else:
            perm = self._perm_tensor.to(device=device)

        mu = mu[:, :, perm]
        if sigma is not None:
            if sigma.dim() == 3:
                # Diagonal covariance: (B, N, K)
                sigma = sigma[:, :, perm]
            else:
                # Full covariance: (B, N, K, K) — sandwich permutation
                sigma = sigma[:, :, perm][:, :, :, perm]

        if omega is None:
            return mu, sigma
        # Sandwich permutation on omega so downstream per-head block slicing
        # (omega[..., block_start:block_end, block_start:block_end]) aligns with
        # the permuted mu/sigma block structure. Without this, the attention
        # sublayer consumes un-permuted omega blocks while the VFE sublayer
        # operates on permuted mu/sigma — a silent coordinate mismatch.
        omega_permuted = omega[:, :, perm][:, :, :, perm]
        return mu, sigma, omega_permuted

    # =========================================================================
    # Step 2/3: Extracted helpers — embed_and_prepare, compute_logits
    # =========================================================================

    def _embed_and_prepare(
        self,
        token_ids: torch.Tensor,
        device: torch.device,
    ) -> Dict:
        r"""Shared embedding and preparation prolog for both forward methods.

        Performs:
          1. Embedding lookup via PriorBank.encode or GaugeTokenEmbedding.
          2. Forward cross-head permutation (when GL(K) cross-head coupling active).
          3. Saving priors mu_prior / sigma_prior before positional encoding.
          4. Position encoding composition onto phi.
          5. Causal attention mask construction.
          6. Optional cached transport operator precomputation.

        Returns a dict with keys:
            mu_q, sigma_q, phi, omega,
            mu_prior, sigma_prior,
            mask, cached_head_transports
        """
        batch_size, num_agents = token_ids.shape

        # 1. Embedding lookup
        omega = None
        if self.use_prior_bank and self.prior_bank is not None:
            embed_out = self.prior_bank.encode(token_ids)
        else:
            embed_out = self.token_embed(token_ids)

        if len(embed_out) == 4:
            mu_q, sigma_q, phi, omega = embed_out
        else:
            mu_q, sigma_q, phi = embed_out

        # 2. Forward cross-head permutation
        # Under gauge_param='omega' we also permute the omega tensor via the
        # sandwich product so downstream per-head block slicing aligns with the
        # permuted mu/sigma block structure.
        if omega is not None and getattr(self, 'gauge_param', 'phi') == 'omega':
            mu_q, sigma_q, omega = self._apply_cross_head_perm(
                mu_q, sigma_q, device, inverse=False, omega=omega,
            )
        else:
            mu_q, sigma_q = self._apply_cross_head_perm(mu_q, sigma_q, device, inverse=False)

        # 3. Save priors (position-independent semantics) and pre-encoding phi
        mu_prior = mu_q.clone()
        sigma_prior = sigma_q.clone() if sigma_q is not None else None
        phi_prior = phi.clone()  # phi before positional encoding (for attention_info)

        # 4. Position encoding — compose token phi with positional phi
        # Phi path: phi ← BCH(phi_token, phi_pos). Omega path: compose_omega
        # right-multiplies omega by exp(phi_pos · G), giving the shift-invariant
        # positional dependence on the gauge transport without relying on a
        # dummy phi tensor. compose(phi, ...) still runs so the dummy phi
        # carries positional info for any downstream diagnostic that inspects
        # it, but the actual gauge transport picks up position via omega.
        phi = self.pos_encoding.compose(phi, num_agents, device=device)
        if omega is not None and getattr(self, 'gauge_param', 'phi') == 'omega':
            omega = self.pos_encoding.compose_omega(omega, num_agents, device=device)

        # 5. Causal attention mask (cached at __init__, sliced by seq length)
        mask = self._causal_mask[:num_agents, :num_agents].unsqueeze(0).expand(
            batch_size, -1, -1
        )  # (B, N, N)

        # 6. Precompute transport operators when phi does not evolve
        if omega is not None and self.gauge_param == 'omega':
            irrep_dims = self.transformer.blocks[0].attention.irrep_dims
            cached_head_transports = []
            block_start = 0
            _omega_ridge = 1e-6
            for d_h in irrep_dims:
                omega_h = omega[:, :, block_start:block_start + d_h, block_start:block_start + d_h]
                # Ridge + fallback for GL(K) inversion.  Raw torch.linalg.inv
                # has no safety on near-singular omega blocks; this matches
                # transport_ops.compute_transport_operators_direct line ~462.
                _eye_dh = torch.eye(d_h, device=omega_h.device, dtype=omega_h.dtype)
                omega_h_reg = omega_h + _omega_ridge * _eye_dh
                try:
                    omega_h_inv = torch.linalg.inv(omega_h_reg)
                except (torch.linalg.LinAlgError, RuntimeError):
                    omega_h_inv = torch.linalg.pinv(omega_h_reg)
                cached_head_transports.append({
                    'exp_phi': omega_h,
                    'exp_neg_phi': omega_h_inv,
                })
                block_start += d_h
        elif not self.evolve_phi:
            first_attention = self.transformer.blocks[0].attention
            cached_head_transports = first_attention.precompute_head_transports(
                phi, device, mu_q.dtype
            )
        else:
            # evolve_phi=True: phi changes between blocks, so we CANNOT cache
            # transport across layers. Leave cached_head_transports=None.
            # Each block computes shared BEP internally (blocks.py shared
            # transport logic) — this eliminates attention↔FFN redundancy
            # within each block without risking stale cache across layers.
            cached_head_transports = None

        return {
            'mu_q': mu_q,
            'sigma_q': sigma_q,
            'phi': phi,
            'omega': omega,
            'mu_prior': mu_prior,
            'sigma_prior': sigma_prior,
            'phi_prior': phi_prior,
            'mask': mask,
            'cached_head_transports': cached_head_transports,
        }

    def _compute_logits(
        self,
        mu_q: torch.Tensor,
        sigma_q: Optional[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        r"""Project final belief means to vocabulary logits.

        Dispatches between:
          - PriorBank KL-decode: logits = -KL(q || π_v) / τ
          - Standard linear projection with optional O(K) sign-vector correction
            when learnable_reflection is active.

        Args:
            mu_q:    (B, N, K) final belief means (already in original dim order).
            sigma_q: (B, N, K) or (B, N, K, K) covariance, or None.
            device:  Device for index tensor construction.

        Returns:
            logits: (B, N, V)
        """
        if self.use_prior_bank and self.prior_bank is not None:
            return self.prior_bank.decode(mu_q, sigma_q, tau=self.prior_bank_tau)

        if getattr(self.token_embed, 'learnable_reflection', False):
            # Apply per-token sign vectors: logits[b,n,v] = <μ_q[b,n], s_v ⊙ W[v]>
            all_ids = torch.arange(self.out_proj.weight.shape[0], device=device)
            z = self.token_embed.sign_logit(all_ids)  # (V, K)
            signs = z.sign()
            signs = z + (signs - z).detach()  # STE
            W_signed = self.out_proj.weight * signs  # (V, K)
            return torch.matmul(mu_q, W_signed.T)  # (B, N, V)

        return self.out_proj(mu_q)  # (B, N, V)

    # =========================================================================
    # Original utility methods
    # =========================================================================

    def _compute_irrep_dims(self, irrep_spec: List[Tuple[str, int, int]]) -> List[int]:
        """
        Compute flat list of block dimensions from irrep_spec.

        For irrep_spec = [('ℓ0', 75, 1), ('ℓ1', 30, 3), ('ℓ2', 18, 5)]:
        Returns: [1, 1, ...(75 times)..., 3, 3, ...(30 times)..., 5, 5, ...(18 times)...]

        This is used for block-diagonal KL computation which exploits
        the gauge structure for massive memory savings.
        """
        irrep_dims = []
        for label, mult, dim in irrep_spec:
            irrep_dims.extend([dim] * mult)
        return irrep_dims

    def _get_effective_irrep_dims(self, irrep_spec: List[Tuple[str, int, int]]) -> List[int]:
        """
        Get effective block dimensions, accounting for cross-head super-blocks.

        When cross_couplings are active, coupled heads are merged into larger
        super-blocks. The super-block dims replace the per-head dims for the
        coupled groups while uncoupled heads keep their original dimensions.

        Falls back to _compute_irrep_dims when no cross-coupling is active.
        """
        if getattr(self, '_super_block_dims', None) is not None:
            return self._super_block_dims
        return self._compute_irrep_dims(irrep_spec)

    # =========================================================================
    # Step 4: __init__ factory methods
    # =========================================================================

    @staticmethod
    def _resolve_gauge_mode(
        config: Dict,
        evolve_phi: bool,
        evolve_phi_e_step: bool,
        use_prior_bank: bool,
        isotropic_covariance: bool,
    ) -> Tuple:
        r"""Resolve gauge group, mode, and parameterization from config.

        Validates gauge_mode, prints informational messages for non-default
        configurations, and enforces the flag invariants:

            - ``gauge_mode='trivial'``: Ω = I; disables phi evolution.
            - ``gauge_mode='constant'``: Ω is a learned global parameter; disables phi evolution.
            - ``learnable_reflection`` is incompatible with ``use_prior_bank``.

        Returns:
            (gauge_group, gauge_dim, gauge_mode, gauge_param,
             evolve_phi, evolve_phi_e_step)
        """
        gauge_group = config.get('gauge_group', 'SO3')
        gauge_dim = config.get('gauge_dim', 3)
        gauge_mode = config.get('gauge_mode', 'learned')

        if gauge_mode not in ('learned', 'trivial', 'constant'):
            raise ValueError(
                f"gauge_mode must be 'learned', 'trivial', or 'constant', got '{gauge_mode}'"
            )

        gauge_param = config.get('gauge_param', 'phi')

        if gauge_param == 'omega':
            irrep_spec = config['irrep_spec']
            omega_head_dims = [dim for _, mult, dim in irrep_spec for _ in range(mult)]
            logger.info(f"Direct Omega parameterization: per-head dims {omega_head_dims}")
            logger.info("       No matrix_exp needed. Full GL(K) including reflections.")

        learnable_reflection = config.get('learnable_reflection', False)
        if learnable_reflection and use_prior_bank:
            raise ValueError(
                "learnable_reflection=True is incompatible with use_prior_bank=True. "
                "PriorBank bypasses GaugeTokenEmbedding (where sign_logit lives), so "
                "the sign vectors would never be applied. Disable one of these options."
            )
        if learnable_reflection:
            logger.info("Reflection enabled: per-token s_i in {+-1}^K sign vectors")
            logger.info("       Content-level sign flip only: mu_i -> s_i odot mu_i")
            logger.info("       Transport Omega_ij and Sigma are NOT modified by signs")
            if isotropic_covariance:
                logger.info("       With isotropic Sigma: KL uses sign-flipped means Q_i = s_i odot mu_i")

        if gauge_mode == 'constant':
            evolve_phi = False
            evolve_phi_e_step = False
            logger.info("Constant gauge mode: Omega_ij = Omega in GL(d_head) for all pairs (i,j)")
            logger.info("       Manuscript Limit 2: S(Omega) cancels under softmax, Omega^{-T} -> W_Q W_K^T")
            logger.info("       Per-head Omega initialized to I, learned via direct gradient descent")
            if isotropic_covariance:
                logger.info("       With Sigma = sigma^2*I: attention proportional to exp(-||Omega^{-1}mu_i - mu_j||^2 / (2sigma^2))")

        if gauge_mode == 'trivial':
            evolve_phi = False
            evolve_phi_e_step = False
            logger.info("Trivial gauge mode: phi = 0, Omega = I (global frame / standard attention limit)")
            logger.info("       This recovers standard KL-attention: KL(q_i || q_j) with no transport.")
            if isotropic_covariance:
                logger.info("Limits 1+2 active: Sigma = sigma^2*I + Omega = I -> attention proportional to exp(-||mu_i - mu_j||^2 / (2sigma^2))")
                logger.info("       Equivalent to standard dot-product attention (up to absorbing sigma^{-2} into W_Q*W_K^T)")

        return gauge_group, gauge_dim, gauge_mode, gauge_param, evolve_phi, evolve_phi_e_step

    def _compute_phi_dim(
        self,
        gauge_group: str,
        gauge_dim: int,
        embed_dim: int,
        irrep_spec: List,
        cross_couplings: List,
    ) -> int:
        r"""Compute the number of Lie algebra generators (phi dimension).

        The dimension of the gauge frame parameter φ_i ∈ ℝ^{n_gen} equals:

            SO(3):  3
            SO(N):  N(N-1)/2
            GL(K), single-head:  K²
            GL(K), multi-head H×d:  H × d² + |cross_couplings| × d²

        Args:
            gauge_group:    'SO3', 'SON', or 'GLK'.
            gauge_dim:      N for SO(N), K for GL(K).
            embed_dim:      K = total embedding dimension.
            irrep_spec:     List of (label, multiplicity, dim) triples.
            cross_couplings: List of (head_a, head_b) pairs.

        Returns:
            phi_dim: int, number of Lie algebra generators.
        """
        if gauge_group == 'SO3':
            return 3
        if gauge_group == 'GLK':
            is_multihead = (
                irrep_spec is not None
                and len(irrep_spec) == 1
                and irrep_spec[0][0] != 'full'
                and irrep_spec[0][1] > 1
            )
            if is_multihead:
                _, n_heads, d_head = irrep_spec[0]
                n_cross_gen = len(cross_couplings) * d_head * d_head
                return n_heads * d_head * d_head + n_cross_gen
            return embed_dim * embed_dim
        # SO(N)
        return gauge_dim * (gauge_dim - 1) // 2

    def _build_generators(
        self,
        gauge_group: str,
        gauge_dim: int,
        embed_dim: int,
        irrep_spec: List,
        cross_couplings: List,
    ) -> np.ndarray:
        r"""Construct the Lie algebra generator matrices.

        Returns a numpy array of shape (n_gen, K, K) where each slice
        G_a is a K×K skew-symmetric (SO) or general (GL) matrix forming
        a basis for the Lie algebra.  The generators are registered as a
        buffer (``self.generators``) after this call returns.

        For GL(K) multi-head with cross-couplings, also registers permutation
        buffers ``_perm_tensor`` / ``_inv_perm_tensor`` and sets the
        ``_cross_head_perm``, ``_super_block_dims``,
        ``_super_block_head_groups`` attributes.

        Args:
            gauge_group:    'SO3', 'SON', or 'GLK'.
            gauge_dim:      N for SO(N).
            embed_dim:      K = total embedding dimension.
            irrep_spec:     List of (label, multiplicity, dim) triples.
            cross_couplings: List of (head_a, head_b) pairs.

        Returns:
            generators: np.ndarray of shape (n_gen, K, K).
        """
        if not GENERATORS_AVAILABLE:
            # Fail loudly.  The previous behaviour silently substituted
            # random skew-symmetric matrices for the Lie algebra basis,
            # which is strictly worse than aborting: training would proceed
            # without the correct gauge structure (Casimir relations, block-
            # diagonal structure, commutation relations all gone) and the
            # model would never implement the gauge transformer thesis.
            # A hard error forces the user to fix their installation instead
            # of running a meaningless experiment.
            raise RuntimeError(
                "math_utils.generators failed to import. The Lie algebra "
                "generators are a required architectural component — without "
                "them the model cannot implement gauge transport and training "
                "would silently produce a broken model. Fix the math_utils "
                "installation (check sys.path and dependencies) before "
                "constructing GaugeTransformerLM."
            )

        if gauge_group == 'SO3':
            return generate_multi_irrep_generators(irrep_spec)

        if gauge_group == 'GLK':
            is_multihead = (
                irrep_spec is not None
                and len(irrep_spec) == 1
                and irrep_spec[0][0] != 'full'
                and irrep_spec[0][1] > 1
            )
            if not is_multihead:
                generators = generate_glK_generators(embed_dim)
                logger.info(f"GL(K) single-head: {embed_dim}^2 = {embed_dim**2} generators")
                return generators

            _, n_heads, d_head = irrep_spec[0]
            if cross_couplings:
                generators = generate_glK_cross_head_generators(embed_dim, n_heads, cross_couplings)
                super_block_dims, super_block_head_groups = merge_coupled_heads(
                    n_heads, d_head, cross_couplings
                )
                generators, perm = reorder_cross_head_generators(
                    generators, n_heads, d_head, cross_couplings, super_block_head_groups,
                )
                self._cross_head_perm = perm
                self.register_buffer('_perm_tensor', torch.from_numpy(perm).long(), persistent=False)
                self.register_buffer('_inv_perm_tensor', torch.from_numpy(np.argsort(perm)).long(), persistent=False)
                self._super_block_dims = super_block_dims

                n_cross = len(cross_couplings) * d_head**2
                logger.info(f"GL(K) cross-head: {n_heads} heads x GL({d_head}), "
                            f"{n_heads * d_head**2} diag + {n_cross} cross generators = "
                            f"{generators.shape[0]} total")
                logger.info(f"       Super-blocks: {super_block_dims} "
                            f"(groups: {super_block_head_groups})")
                return generators

            generators = generate_glK_multihead_generators(embed_dim, n_heads)
            self._cross_head_perm = None
            self._super_block_dims = None
            logger.info(f"GL(K) multi-head: {n_heads} heads x GL({d_head}), "
                        f"{n_heads * d_head**2} generators (vs {embed_dim**2} single-head)")
            return generators

        # SO(N)
        return generate_multi_irrep_soN_generators(irrep_spec, gauge_dim)

    def _build_embeddings(
        self,
        config: Dict,
        vocab_size: int,
        embed_dim: int,
        irrep_spec: List,
        max_seq_len: int,
        pos_mode: str,
        pos_encoding_scale: float,
        gauge_mode: str,
        gauge_param: str,
        gauge_fixed_priors: bool,
        diagonal_covariance: bool,
        use_prior_bank: bool,
        use_positional_embedding: bool,
        irrep_dims: Optional[List[int]] = None,
    ) -> None:
        r"""Build GaugeTokenEmbedding, GaugePositionalEncoding, and PriorBank.

        Sets ``self.token_embed``, ``self.pos_encoding``,
        ``self.prior_bank``, ``self.use_prior_bank``, and ``self.prior_bank_tau``.

        If ``use_prior_bank=True``, token_embed parameters are frozen
        (they are never used in the computation graph when PriorBank is active).

        Args:
            config:                 Full config dict (for optional key fallback).
            vocab_size:             Vocabulary size V.
            embed_dim:              Belief dimension K.
            irrep_spec:             Irrep spec list.
            max_seq_len:            Maximum sequence length.
            pos_mode:               Position encoding mode ('none', 'learned', 'sinusoidal').
            pos_encoding_scale:     Scale for gauge-frame positional encoding.
            gauge_mode:             'learned', 'trivial', or 'constant'.
            gauge_param:            'phi' or 'omega'.
            gauge_fixed_priors:     If True, priors are gauge-fixed (not per-token free).
            diagonal_covariance:    If True, Σ stored as (B, N, K) diagonal vector.
            use_prior_bank:         If True, create and use PriorBank.
            use_positional_embedding: If True, add position to μ (standard transformer style).
        """
        self.token_embed = GaugeTokenEmbedding(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            irrep_spec=irrep_spec,
            init_std=config.get('mu_init_std', None),
            init_sigma_scale=1.0,
            learnable_sigma=config.get('evolve_sigma', True),
            learnable_phi=config.get('learnable_phi', gauge_mode == 'learned'),
            gauge_fixed_priors=gauge_fixed_priors,
            generators=self.generators,
            diagonal_covariance=diagonal_covariance,
            isotropic_covariance=config.get('isotropic_covariance', False),
            max_seq_len=max_seq_len,
            use_positional_embedding=use_positional_embedding,
            phi_dim=self.phi_dim,
            phi_scale=config.get('phi_scale', 0.3),
            mu_normalize=config.get('mu_normalize', False),
            mu_max_norm=config.get('mu_max_norm', None),
            learnable_reflection=config.get('learnable_reflection', False),
            sigma_max=config.get('sigma_max', 5.0),
            gauge_param=gauge_param,
            omega_head_dims=self.omega_head_dims,
            irrep_dims=irrep_dims,
            phi_project_slk=config.get('phi_project_slk', False),
            phi_trace_clamp=config.get('phi_trace_clamp', None),
        )

        # Position encoding for φ (gauge frame) — encodes RELATIVE position via transport.
        # φ_i = φ_token_i + φ_pos(i), transport Ω_ij = exp(φ_i·G)·exp(-φ_j·G)
        # depends on relative position, giving shift-invariant attention.
        self.pos_encoding = GaugePositionalEncoding(
            max_seq_len=max_seq_len,
            mode=pos_mode,
            scale=pos_encoding_scale,
            phi_dim=self.phi_dim,
            generators=self.generators,
            gauge_group=self.gauge_group,
            irrep_dims=irrep_dims,
            phi_project_slk=config.get('phi_project_slk', False),
        )

        self.use_prior_bank = use_prior_bank
        self.prior_bank_tau = config.get('prior_bank_tau', 1.0)
        self.prior_bank = None

        if use_prior_bank:
            from transformer.core.prior_bank import PriorBank

            self.prior_bank = PriorBank(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                init_std=config.get('mu_init_std', None),
                init_sigma_scale=1.0,
                learnable_sigma=config.get('evolve_sigma', True),
                gauge_fixed_priors=gauge_fixed_priors,
                # Always pass generators so PriorBank can build _phi_trace_vec
                # for sl(K) projection / trace clamp even when the standard
                # (non-gauge-fixed) per-token phi_embed path is active.
                generators=self.generators,
                phi_dim=self.phi_dim,
                phi_scale=config.get('phi_scale', 0.3),
                gauge_param=gauge_param,
                omega_head_dims=self.omega_head_dims,
                sigma_ce_scale=config.get('sigma_ce_scale', 0.01),
                learnable_temperature=config.get(
                    'learnable_pb_temperature',
                    config.get('learnable_temperature', False),
                ),
                diagonal_covariance=diagonal_covariance,
                sigma_max=config.get('sigma_max', 5.0),
                irrep_dims=irrep_dims,
                cache_decode_priors=config.get('cache_decode_priors', False),
                exact_diagonal_transport=config.get('exact_diagonal_transport', False),
                phi_project_slk=config.get('phi_project_slk', False),
                phi_trace_clamp=config.get('phi_trace_clamp', None),
            )
            logger.info(f"GaugeTransformerLM: Created PriorBank with token-dependent priors "
                        f"(vocab_size={vocab_size})")
            logger.info(f"                     gauge_fixed_priors={gauge_fixed_priors}, "
                        f"tau={self.prior_bank_tau}")

            # Freeze token_embed: PriorBank replaces it for encode/decode.
            # Without this, token_embed parameters receive optimizer weight_decay
            # updates every step despite never appearing in the computation graph.
            self.token_embed.requires_grad_(False)

    def _init_weights(self, module):
        """Initialize weights following best practices.

        Note: Skip nn.Embedding modules — their initialization is handled
        by GaugeTokenEmbedding.__init__() with calibrated std values
        (e.g., init_std=2.0 for mu_embed, scaled phi_embed).
        Overwriting with std=0.02 would destroy the gauge-theoretic init.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
        elif isinstance(module, RMSNorm):
            torch.nn.init.ones_(module.weight)

    def forward(
        self,
        token_ids: torch.Tensor,
        return_agents: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Forward pass through 0D gauge transformer.

        Args:
            token_ids: (batch, seq_len) token indices
                       seq_len = number of agents at the single point c*
            return_agents: If True, return intermediate agent states

        Returns:
            logits: (batch, num_agents, vocab_size) next-token predictions
            agents: Optional dict with mu, sigma, phi for each agent

        0D STRUCTURE:
            - All agents exist at single base manifold point c*
            - No spatial variation: mu[i], sigma[i], phi[i] are per-agent, not per-location
            - Attention β_ij are scalars, not spatial fields
        """
        batch_size, num_agents = token_ids.shape
        device = token_ids.device

        # Drop the per-forward-pass decode-prior cache (no-op when caching is
        # disabled).  Stale graph from the previous forward pass would have
        # been freed by backward; the next decode call must recompute.
        if self.use_prior_bank and self.prior_bank is not None:
            self.prior_bank.clear_decode_cache()

        # =================================================================
        # Trajectory Recording: Start forward pass
        # =================================================================
        recorder = get_global_recorder()
        if recorder is not None and recorder.enabled:
            ffn_mode = self.config.get('ffn_mode', 'VFE_dynamic')
            recorder.start_forward(batch_size, num_agents, ffn_mode=ffn_mode)

        # =================================================================
        # 1-5. Embed, permute, save priors, position-encode, build mask,
        #      precompute transport operators.
        # =================================================================
        prep = self._embed_and_prepare(token_ids, device)
        mu_q = prep['mu_q']
        sigma_q = prep['sigma_q']
        phi = prep['phi']
        omega = prep['omega']
        mu_prior = prep['mu_prior']
        sigma_prior = prep['sigma_prior']
        mask = prep['mask']
        cached_head_transports = prep['cached_head_transports']

        # Record embeddings for trajectory tracking (after positional encoding)
        if recorder is not None and recorder.enabled:
            recorder.record_embeddings(mu_q, sigma_q, phi)

        # =================================================================
        # 6. Forward Through Transformer Stack
        # =================================================================
        # The E-step does not see target tokens — the observation likelihood
        # is provided by the outer CE loss in compute_free_energy_loss.
        mu_q, sigma_q, phi, intermediates = self.transformer(
            mu_q,
            sigma_q,
            phi,
            self.generators,
            mask=mask,
            mu_prior=mu_prior,
            token_ids=token_ids,
            return_intermediates=return_agents,
            cached_head_transports=cached_head_transports,
            omega=omega,
            sigma_prior=sigma_prior,
        )

        # =================================================================
        # 6b. Inverse Cross-Head Permutation (restore original dim order)
        # =================================================================
        mu_q, sigma_q = self._apply_cross_head_perm(mu_q, sigma_q, device, inverse=True)

        # =================================================================
        # 7. Project to Vocabulary (one prediction per agent)
        # =================================================================
        logits = self._compute_logits(mu_q, sigma_q, device)

        # =================================================================
        # Trajectory Recording: End forward pass
        # =================================================================
        if recorder is not None and recorder.enabled:
            recorder.end_forward(mu_q, logits)

        if return_agents:
            agent_states = {
                'mu': mu_q.detach(),
                'sigma': sigma_q.detach() if sigma_q is not None else None,
                'phi': phi.detach(),
                'intermediates': intermediates,
            }
            return logits, agent_states

        return logits

    def forward_with_attention(
        self,
        token_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass that returns attention weights and KL matrices for loss computation.

        This is used during training to compute the attention-weighted free energy:
            F = Σ_ij β_ij · KL(q_i || Ω_ij[q_j]) - E[log p(o|x)]

        The E-step does not see target tokens. `targets` here is forwarded only
        to the final-layer CE in compute_free_energy_loss; it is never passed
        into the belief inference loop.

        Args:
            token_ids: (batch, seq_len) token indices
            targets: (batch, seq_len) target tokens — used only for the outer
                CE loss. Not available to the E-step.

        Returns:
            logits: (batch, num_agents, vocab_size) predictions
            attention_info: Dict with:
                - 'beta': (n_layers, B, n_heads, N, N) attention weights per head per layer
                - 'kl': (n_layers, B, n_heads, N, N) KL divergences per head per layer
                - 'mu': (B, N, K) final belief means
                - 'sigma': (B, N, K, K) or (B, N, K) final covariances
                - 'phi': (B, N, phi_dim) final gauge frames
                - 'mu_prior': (B, N, K) prior means (before position encoding)
                - 'sigma_prior': (B, N, K, K) or (B, N, K) prior covariances
                - 'phi_prior': (B, N, phi_dim) prior gauge frames
                - 'n_layers': int number of layers
        """
        batch_size, num_agents = token_ids.shape
        device = token_ids.device

        # Drop the per-forward-pass decode-prior cache (no-op when caching is
        # disabled).  Mirror of the same call in forward().
        if self.use_prior_bank and self.prior_bank is not None:
            self.prior_bank.clear_decode_cache()

        # =================================================================
        # 1-5. Shared embedding and preparation prolog
        # =================================================================
        prep = self._embed_and_prepare(token_ids, device)
        mu_q = prep['mu_q']
        sigma_q = prep['sigma_q']
        phi = prep['phi']
        omega = prep['omega']
        mu_prior = prep['mu_prior']
        sigma_prior = prep['sigma_prior']
        phi_prior = prep['phi_prior']  # phi before positional encoding
        mask = prep['mask']
        cached_head_transports = prep['cached_head_transports']

        # Forward through ALL transformer blocks WITH attention tracking.
        # Each layer's beta/kl is captured for visualization.
        # The E-step does not see target tokens — the observation likelihood
        # is provided by the outer CE loss in compute_free_energy_loss.
        all_betas = []
        all_kls = []
        n_blocks = len(self.transformer.blocks)

        for layer_idx, block in enumerate(self.transformer.blocks):
            is_final = (layer_idx == n_blocks - 1)

            # Save pre-layer state for diagnostics
            if self._collect_layer_diagnostics:
                _mu_before_layer = mu_q.detach().clone()

            # Delegate the full per-layer block computation to GaugeTransformerBlock.forward.
            # return_attention=True yields the 5-tuple (mu_q, sigma_q, phi, beta, kl)
            # so we can collect per-layer attention weights for the loss function
            # without duplicating the block body here.
            mu_q, sigma_q, phi, beta, kl = block(
                mu_q, sigma_q, phi, self.generators,
                mask=mask,
                mu_prior=mu_prior,
                token_ids=token_ids,
                cached_head_transports=cached_head_transports,
                omega=omega,
                sigma_prior=sigma_prior,
                return_attention=True,
            )

            # Store per-layer attention (keep gradients for loss computation)
            all_betas.append(beta)
            all_kls.append(kl)

            # Propagate evolved omega from E-step to next layer (gauge_param='omega').
            # Without this, each layer receives the original embedding omega,
            # discarding E-step omega evolution from previous layers.
            evolved_omega = getattr(block, '_last_evolved_omega', None)
            if evolved_omega is not None:
                omega = evolved_omega
                # The precomputed cached_head_transports reflect the INITIAL
                # embedding omega. Once a block has evolved omega via retract
                # (em_modes other than em_phi_p), the cache is stale and must
                # be invalidated so the next block rebuilds from fresh omega.
                # blocks.py:586-611 handles the rebuild when this is None.
                cached_head_transports = None

            # mu_prior stays at the embedding-derived prior for every layer
            # (no cross-layer μ cascade). sigma_prior also stays at the
            # embedding value to prevent progressive tightening.

            # =============================================================
            # Per-layer diagnostics collection
            # =============================================================
            if self._collect_layer_diagnostics:
                # mu_attn and mu_ffn are block-internal; block.forward stores them
                # on the block instance so diagnostics can read them post-call.
                _mu_attn = getattr(block, '_last_mu_attn', None)
                _mu_ffn = getattr(block, '_last_mu_ffn', None)
                # Batch scalar extractions to minimize CUDA syncs
                _mu_in_norm = _mu_before_layer.norm()
                _mu_out_norm = mu_q.detach().norm()
                _delta_mu = (mu_q.detach() - _mu_before_layer)
                _delta_norm = _delta_mu.norm()
                _phi_norm = phi.detach().norm()
                # torch.stack requires all tensors on the same device; pin
                # the None-fallback scalars to mu_q's device so this probe
                # works under CUDA when _mu_attn / _mu_ffn / sigma_q are absent.
                _zero = torch.zeros((), device=mu_q.device, dtype=mu_q.dtype)
                _mu_attn_norm = _mu_attn.detach().norm() if _mu_attn is not None else _zero
                _mu_ffn_norm = _mu_ffn.detach().norm() if _mu_ffn is not None else _zero
                _sigma_mean = sigma_q.detach().mean() if sigma_q is not None else _zero
                _mu_pos_std = mu_q.detach().std(dim=1).mean()
                _norms = torch.stack([
                    _mu_in_norm, _mu_out_norm, _delta_norm, _phi_norm,
                    _mu_attn_norm, _mu_ffn_norm, _sigma_mean, _mu_pos_std,
                ]).cpu().tolist()
                _ld = {
                    'layer': layer_idx,
                    'mu_input_norm': _norms[0],
                    'mu_output_norm': _norms[1],
                    'delta_mu_norm': _norms[2],
                    'delta_mu_relative': _norms[2] / (_norms[0] + 1e-8),
                    'sigma_mean_diag': _norms[6],
                    'phi_norm': _norms[3],
                    'mu_attn_norm': _norms[4],
                    'mu_ffn_norm': _norms[5],
                    'residual_ratio': _norms[5] / (_norms[1] + 1e-8),
                    'mu_position_std': _norms[7],
                }
                if beta is not None:
                    _beta_d = beta.detach().clamp(min=1e-10)
                    _ld['attention_entropy'] = -(_beta_d * _beta_d.log()).sum(dim=-1).mean().item()
                if kl is not None:
                    _ld['kl_mean'] = kl.detach().mean().item()
                    _ld['kl_std'] = kl.detach().std().item()
                # Per-layer CE probe
                if targets is not None:
                    with torch.no_grad():
                        _probe_sigma = sigma_q.detach() if sigma_q is not None else None
                        _fn = self.transformer.final_norm
                        _probe_mu = _fn(mu_q.detach(), _probe_sigma) if isinstance(_fn, (RMSNorm, MahalanobisNorm)) else _fn(mu_q.detach())
                        # Undo cross-head permutation before projecting to vocab
                        _probe_mu, _probe_sigma = self._apply_cross_head_perm(
                            _probe_mu, _probe_sigma, _probe_mu.device, inverse=True
                        )
                        # Use PriorBank decode when active (out_proj is untrained in that case)
                        if self.use_prior_bank and self.prior_bank is not None:
                            _probe_logits = self.prior_bank.decode(
                                _probe_mu, _probe_sigma, tau=self.prior_bank_tau
                            )
                        else:
                            _probe_logits = self.out_proj(_probe_mu)
                        _ld['ce_loss'] = F.cross_entropy(
                            _probe_logits.reshape(-1, _probe_logits.size(-1)),
                            targets.reshape(-1),
                            reduction='mean',
                            ignore_index=-100,
                        ).item()
                        _ld['perplexity'] = math.exp(min(_ld['ce_loss'], 20.0))
                self._layer_diagnostics.append(_ld)

        # Final norm (pass sigma for MahalanobisNorm/RMSNorm; LayerNorm doesn't accept it)
        _fn = self.transformer.final_norm
        mu_q = _fn(mu_q, sigma_q) if isinstance(_fn, (RMSNorm, MahalanobisNorm)) else _fn(mu_q)

        # Inverse cross-head permutation (restore original dim order)
        mu_q, sigma_q = self._apply_cross_head_perm(mu_q, sigma_q, device, inverse=True)

        # Project to vocabulary
        logits = self._compute_logits(mu_q, sigma_q, device)

        # Stack per-layer attention into (n_layers, B, n_heads, N, N) tensors
        # Filter out None entries (shouldn't happen, but defensive)
        valid_betas = [b for b in all_betas if b is not None]
        valid_kls = [k for k in all_kls if k is not None]
        stacked_beta = torch.stack(valid_betas, dim=0) if valid_betas else None
        stacked_kl = torch.stack(valid_kls, dim=0) if valid_kls else None

        # Retrieve adaptive alpha_i from last block's FFN (if learnable_alpha enabled)
        last_block = self.transformer.blocks[-1]
        alpha_i = getattr(last_block.ffn, '_last_alpha_i', None)

        # =====================================================================
        # Collect VFE gradient decomposition from last block's FFN
        # =====================================================================
        vfe_debug = None
        for block in self.transformer.blocks:
            _dbg = getattr(block.ffn, 'last_vfe_debug', None)
            if _dbg is not None:
                vfe_debug = _dbg  # Use last layer's debug (most informative)

        # =====================================================================
        # Compute transport operator & covariance health diagnostics
        # =====================================================================
        transport_metrics = {}
        covariance_metrics = {}
        if self._collect_dynamics_metrics:
            self._dynamics_metrics_counter += 1
        if self._collect_dynamics_metrics and (self._dynamics_metrics_counter % self._dynamics_metrics_interval == 0):
            with torch.no_grad():
                # --- Covariance health ---
                if sigma_q is not None:
                    if sigma_q.dim() == 3:
                        # Diagonal: (B, N, K)
                        _sq = sigma_q.detach().float()
                        covariance_metrics['sigma_q_mean'] = _sq.mean().item()
                        covariance_metrics['sigma_q_min'] = _sq.min().item()
                        covariance_metrics['sigma_q_max'] = _sq.max().item()
                        covariance_metrics['sigma_q_std'] = _sq.std().item()
                        # Condition number: max/min per position
                        _cond = _sq.max(dim=-1).values / _sq.min(dim=-1).values.clamp(min=1e-8)
                        covariance_metrics['sigma_q_cond_mean'] = _cond.mean().item()
                        covariance_metrics['sigma_q_cond_max'] = _cond.max().item()
                    else:
                        # Full: (B, N, K, K)
                        _sq_diag = torch.diagonal(sigma_q.detach().float(), dim1=-2, dim2=-1)
                        covariance_metrics['sigma_q_mean'] = _sq_diag.mean().item()
                        covariance_metrics['sigma_q_min'] = _sq_diag.min().item()
                        covariance_metrics['sigma_q_max'] = _sq_diag.max().item()
                        covariance_metrics['sigma_q_std'] = _sq_diag.std().item()
                        _cond = _sq_diag.max(dim=-1).values / _sq_diag.min(dim=-1).values.clamp(min=1e-8)
                        covariance_metrics['sigma_q_cond_mean'] = _cond.mean().item()
                        covariance_metrics['sigma_q_cond_max'] = _cond.max().item()
                if sigma_prior is not None:
                    if sigma_prior.dim() == 3:
                        _sp = sigma_prior.detach().float()
                        covariance_metrics['sigma_p_mean'] = _sp.mean().item()
                        covariance_metrics['sigma_p_min'] = _sp.min().item()
                        covariance_metrics['sigma_p_max'] = _sp.max().item()
                    else:
                        _sp_diag = torch.diagonal(sigma_prior.detach().float(), dim1=-2, dim2=-1)
                        covariance_metrics['sigma_p_mean'] = _sp_diag.mean().item()
                        covariance_metrics['sigma_p_min'] = _sp_diag.min().item()
                        covariance_metrics['sigma_p_max'] = _sp_diag.max().item()

                # --- Prior-belief gap: per-position KL(q*||p) ---
                if mu_q is not None and mu_prior is not None and sigma_q is not None and sigma_prior is not None:
                    _mq = mu_q.detach().float()
                    _mp = mu_prior.detach().float()
                    if sigma_q.dim() == 3 and sigma_prior.dim() == 3:
                        _sq = sigma_q.detach().float().clamp(min=1e-8)
                        _sp = sigma_prior.detach().float().clamp(min=1e-8)
                        K = _sq.shape[-1]
                        # Diagonal Gaussian KL: 0.5 * [tr(Sp^{-1}Sq) + (mp-mq)^T Sp^{-1}(mp-mq) - K + ln(|Sp|/|Sq|)]
                        ratio = _sq / _sp  # (B, N, K)
                        diff = _mp - _mq
                        mahal = (diff ** 2 / _sp).sum(dim=-1)  # (B, N)
                        kl_per_pos = 0.5 * (ratio.sum(dim=-1) + mahal - K + (_sp.log() - _sq.log()).sum(dim=-1))
                        covariance_metrics['prior_belief_kl_mean'] = kl_per_pos.mean().item()
                        covariance_metrics['prior_belief_kl_max'] = kl_per_pos.max().item()
                        covariance_metrics['prior_belief_kl_std'] = kl_per_pos.std().item()

                # --- Transport proxy: phi norm statistics ---
                # ||φ_i - φ_j|| governs how far Ω_ij deviates from identity.
                # Large phi norms → large transport deviations.
                _phi_final = phi.detach().float()
                _phi_norms = _phi_final.norm(dim=-1)  # (B, N)
                transport_metrics['phi_norm_mean'] = _phi_norms.mean().item()
                transport_metrics['phi_norm_std'] = _phi_norms.std().item()
                transport_metrics['phi_norm_max'] = _phi_norms.max().item()
                # Pairwise phi distance (sample to avoid O(N²) cost)
                _N = _phi_final.shape[1]
                _n_sample = min(32, _N)
                _idx = torch.randperm(_N, device=_phi_final.device)[:_n_sample]
                _phi_sample = _phi_final[:, _idx]  # (B, n_sample, phi_dim)
                _phi_diff = _phi_sample.unsqueeze(2) - _phi_sample.unsqueeze(1)  # (B, n_s, n_s, phi_dim)
                _pair_dist = _phi_diff.norm(dim=-1)  # (B, n_s, n_s)
                transport_metrics['phi_pairwise_dist_mean'] = _pair_dist.mean().item()
                transport_metrics['phi_pairwise_dist_max'] = _pair_dist.max().item()

                # --- Attention information-theoretic metrics ---
                if stacked_beta is not None:
                    _beta_d = stacked_beta.detach().float().clamp(min=1e-10)
                    # Per-head entropy: H(β_i) = -Σ_j β_ij log β_ij
                    _entropy = -(_beta_d * _beta_d.log()).sum(dim=-1)  # (..., N)
                    # Effective rank via nuclear norm / max singular value
                    _beta_last = _beta_d[-1, 0]  # (n_heads, N, N) - last layer, first batch
                    # Per-head entropy statistics
                    _h_per_head = _entropy[-1, 0].mean(dim=-1)  # (n_heads,)
                    transport_metrics['attn_entropy_per_head_mean'] = _h_per_head.mean().item()
                    transport_metrics['attn_entropy_per_head_std'] = _h_per_head.std().item() if _h_per_head.numel() > 1 else 0.0
                    transport_metrics['attn_entropy_per_head_min'] = _h_per_head.min().item()
                    transport_metrics['attn_entropy_per_head_max'] = _h_per_head.max().item()
                    # Head diversity: std of per-head mean attention patterns
                    _head_means = _beta_last.mean(dim=1)  # (n_heads, N)
                    _head_corr = torch.corrcoef(_head_means) if _head_means.shape[0] > 1 else None
                    if _head_corr is not None:
                        # Mean off-diagonal correlation (lower = more diverse)
                        _mask = ~torch.eye(_head_corr.shape[0], dtype=torch.bool, device=_head_corr.device)
                        transport_metrics['head_correlation_mean'] = _head_corr[_mask].mean().item()

        attention_info = {
            'beta': stacked_beta,      # (n_layers, B, n_heads, N, N)
            'kl': stacked_kl,          # (n_layers, B, n_heads, N, N)
            'n_layers': n_blocks,      # Number of layers
            'mu': mu_q,        # (B, N, K) - evolved beliefs q_i (fast/E-step)
            'sigma': sigma_q,  # (B, N, K, K) or None
            'phi': phi,  # (B, N, gauge_dim) - post-FFN phi (updated by each block's forward)
            # Models s_i (saved before position encoding).
            # In the FEP hierarchy h→s→p→q, these are the slow variables
            # (embedding params = what backprop updates). Currently p_i = s_i.
            'mu_prior': mu_prior,        # (B, N, K) - model means s_i
            'sigma_prior': sigma_prior,  # (B, N, K, K) - model covariances
            'phi_prior': phi_prior,      # (B, N, gauge_dim) - model gauge frames
            # Adaptive alpha_i from E-step (if learnable_alpha enabled)
            'alpha_i': alpha_i,          # (B, N, K) or None
            # VFE gradient decomposition (from last layer's E-step)
            'vfe_debug': vfe_debug,      # Dict or None
            # Transport & covariance health
            'transport_metrics': transport_metrics,
            'covariance_metrics': covariance_metrics,
            # Last-layer evolved gauge frame when gauge_param='omega'; None for
            # the phi path. Diagnostics (det_omega, spectrum) use this to
            # bypass the dummy phi tensor that the omega path carries.
            'omega': omega if getattr(self, 'gauge_param', 'phi') == 'omega' else None,
            # Initial per-token Omega (embedding lookup, attached to omega_embed)
            # for the omega path. Exposed separately because the evolved omega
            # above is detached at the EM boundary under em_phi_p and so
            # cannot deliver an M-step gradient to omega_embed; the penalty
            # term (log|det Ω|)² in compute_free_energy_loss consumes this
            # attached tensor instead.
            'omega_initial': (prep['omega']
                              if getattr(self, 'gauge_param', 'phi') == 'omega'
                              else None),
        }

        return logits, attention_info

    @torch.inference_mode()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation.

        Note: Unlike standard transformers, VFE transformers cannot use KV-caching
        because beliefs evolve iteratively through the VFE E-step. Each token's
        representation depends on all other tokens through the gauge transport
        operators Ω_ij, which change as the sequence grows.

        Args:
            prompt_ids: (1, prompt_len) initial tokens
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling (optional)
            top_p: Nucleus sampling (optional)

        Returns:
            generated: (1, prompt_len + max_new_tokens) full sequence
        """
        was_training = self.training
        self.eval()
        try:
            generated = prompt_ids.clone()
            max_seq_len = self.config['max_seq_len']

            for _ in range(max_new_tokens):
                # Use sliding window when sequence exceeds max_seq_len
                context = generated[:, -max_seq_len:] if generated.shape[1] > max_seq_len else generated

                # Forward pass - handle both tuple and single return value
                result = self.forward(context)
                logits = result[0] if isinstance(result, tuple) else result  # (1, T, V)

                # Greedy short-circuit: temperature <= 0 means "always take the
                # argmax", which the temperature-scaled softmax path cannot
                # express safely (dividing by zero produces inf, and
                # softmax(inf)=NaN, which then crashes torch.multinomial).
                # Do the greedy argmax directly and skip sampling.
                if temperature <= 0.0:
                    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    generated = torch.cat([generated, next_token], dim=1)
                    continue

                # Get logits for last token
                logits_next = logits[:, -1, :] / temperature  # (1, V)

                # Apply top-k filtering
                if top_k is not None:
                    v, _ = torch.topk(logits_next, min(top_k, logits_next.size(-1)))
                    logits_next[logits_next < v[:, [-1]]] = -float('inf')

                # Apply top-p (nucleus) filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits_next, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift right to keep first token above threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    # Scatter back to original indexing.  Because sorted_indices
                    # is a full permutation of [0, V-1], every output position
                    # gets overwritten by the scatter, so the choice of initial
                    # tensor (sorted_indices_to_remove vs zeros) does not
                    # affect the result — but zeros_like is clearer.
                    indices_to_remove = torch.zeros_like(logits_next, dtype=torch.bool).scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits_next[indices_to_remove] = -float('inf')

                # Sample
                probs = F.softmax(logits_next, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)

                # Append
                generated = torch.cat([generated, next_token], dim=1)

            return generated
        finally:
            if was_training:
                self.train()

    def generate_active_inference(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
        gamma: float = 1.0,
        top_k: int = 50,
        preference_mode: str = 'current_belief',
        include_epistemic: bool = False,
        epistemic_samples: int = 4,
        temperature: float = 1.0,
        verbose: bool = False,
    ) -> torch.Tensor:
        """Autoregressive generation using canonical active inference.

        Selects tokens by minimizing expected free energy (EFE) over a
        top-K candidate set, rather than sampling from temperature-scaled
        logits.  See ``expected_free_energy.generate_active_inference`` for
        full documentation.

        NOT decorated with @torch.inference_mode() — uses torch.no_grad()
        internally.
        """
        from transformer.core.expected_free_energy import (
            generate_active_inference as _gen_ai,
        )
        return _gen_ai(
            model=self,
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            gamma=gamma,
            top_k=top_k,
            preference_mode=preference_mode,
            include_epistemic=include_epistemic,
            epistemic_samples=epistemic_samples,
            temperature=temperature,
            verbose=verbose,
        )

    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Return number of parameters.

        Args:
            non_embedding: If True, exclude embedding parameters

        Returns:
            n_params: Total parameter count
        """
        n_params = sum(p.numel() for p in self.parameters())

        if non_embedding:
            if self.use_prior_bank and self.prior_bank is not None:
                # PriorBank parameters serve as both embedding and output projection
                n_params -= sum(p.numel() for p in self.prior_bank.parameters())
            elif hasattr(self.token_embed, 'mu_embed'):
                # Standard per-token embeddings
                n_params -= self.token_embed.mu_embed.weight.numel()
            elif hasattr(self.token_embed, 'base_mu'):
                # Gauge-fixed priors: base_mu + base_log_sigma_diag + gauge embed
                n_params -= self.token_embed.base_mu.numel()
                n_params -= self.token_embed.base_log_sigma_diag.numel()
                if hasattr(self.token_embed, 'phi_embed'):
                    n_params -= self.token_embed.phi_embed.weight.numel()
                elif hasattr(self.token_embed, 'omega_embed'):
                    n_params -= self.token_embed.omega_embed.weight.numel()

        return n_params

    # =========================================================================
    # P-FLOW / DELTA RULE: backprop-free Hebbian learning
    # =========================================================================
    # The implementations live in ``transformer/core/hebbian.py``.  These
    # methods are thin delegators kept on the model for backward
    # compatibility with external callers (e.g. PublicationTrainer).
    def p_flow_update(
        self,
        token_ids: torch.Tensor,
        mu_beliefs: torch.Tensor,
        prediction_errors: torch.Tensor,
        ema_decay: float = 0.99,
        sigma_beliefs: Optional[torch.Tensor] = None,
        pad_token_id: int = -100,
    ):
        """Thin delegator — see ``hebbian.p_flow_update_model``."""
        from transformer.core.hebbian import p_flow_update_model
        return p_flow_update_model(
            self, token_ids, mu_beliefs, prediction_errors,
            ema_decay=ema_decay, sigma_beliefs=sigma_beliefs,
            pad_token_id=pad_token_id,
        )

    def phi_flow_update(
        self,
        token_ids: torch.Tensor,
        phi_evolved: torch.Tensor,
        prediction_errors: torch.Tensor,
        ema_decay: float = 0.99,
        pad_token_id: int = -100,
    ):
        """Thin delegator — see ``hebbian.phi_flow_update_model``."""
        from transformer.core.hebbian import phi_flow_update_model
        return phi_flow_update_model(
            self, token_ids, phi_evolved, prediction_errors,
            ema_decay=ema_decay, pad_token_id=pad_token_id,
        )

    def delta_rule_update_w_out(
        self,
        mu_beliefs: torch.Tensor,
        targets: torch.Tensor,
        lr: float = 0.1,
        pad_token_id: int = -100,
    ):
        """Thin delegator — see ``hebbian.delta_rule_update_w_out_model``."""
        from transformer.core.hebbian import delta_rule_update_w_out_model
        return delta_rule_update_w_out_model(
            self, mu_beliefs, targets, lr=lr, pad_token_id=pad_token_id,
        )

