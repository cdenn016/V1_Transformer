"""
VFEModel: full gauge-theoretic VFE transformer.

    token_ids → PriorBank.encode → positional BCH → VFEStack → norm → PriorBank.decode → logits

No nn.Linear in default config — PriorBank IS the decoder (Law 3).
No targets in E-step — beliefs inferred from context only (Law 1).
All transport uses sandwich product (Law 2).

Implementation note — outer M-step minimises CE + aux, not full F.
================================================================
The training objective assembled in ``forward()`` is

    loss = ce_loss + 0.5 * mass_phi * ||phi||^2 + sum(block._aux_hyperparam_loss)

NOT the manuscript free-energy functional F. The manuscript F's
``alpha * KL(q || p)`` and ``sum_ij beta_ij * KL(q_i || Omega_ij q_j)``
terms enter this loss only as *gradients* through the unrolled
E-step iterations — they never appear as backward-graph-visible scalars
the outer optimizer sees. The auxiliary scalar
``_aux_hyperparam_loss`` exists solely to route gradients into the
M-step hyperparameters (``raw_c0``, ``raw_b0``, ``log_kappa``) that the
E-step inner loop reads but does not differentiate through.

This is structurally amortised inference: the embedding parameters
(``base_mu``, ``base_log_sigma``, ``phi_embed``) are tuned so that CE is
small *after* the E-step has relaxed the beliefs, not by alternating
E and M on the same F. The variational interpretation still holds —
F appears in the inner loop's descent direction — but the outer loop
is not minimising F directly. See ``transformer/vfe/e_step.py``
module docstring for the term-by-term breakdown of what the E-step
inner loop actually evaluates.
"""

import math
from typing import Optional, Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer.core.types import BeliefState
from transformer.core.blocks import MahalanobisNorm, RMSNorm
from transformer.vfe.block import _resolve_vfe_norm
from transformer.vfe.config import VFEConfig
from transformer.vfe.prior_bank import VFEPriorBank
from transformer.vfe.positional import VFEPositionalEncoding
from transformer.vfe.stack import VFEStack


class VFEModel(nn.Module):
    r"""Gauge-theoretic VFE transformer language model.

    Architecture (VFE_Transformer_Idea.md Section 12):

    .. math::
        \{x_i\} \xrightarrow{\text{encode}} \{(\mu, \Sigma, \phi)\}
        \xrightarrow{\text{BCH}} \xrightarrow{L\text{ blocks}}
        \xrightarrow{\text{norm}} \xrightarrow{\text{decode}} p(y_i | x_{<i})

    Each block performs: compute :math:`\beta_{ij}` from gauge-transported KL;
    compute E-step gradients of :math:`F`; update :math:`(\mu, \Sigma, \phi)`
    via natural gradient; pass posterior onward as next layer's prior.

    Three design laws:
        1. E-step does not see targets (no ``use_obs_in_vfe`` — architecturally impossible)
        2. Covariance transport uses :math:`\Omega \Sigma \Omega^\top` (sandwich product)
        3. Encode, infer, and decode on the same Gaussian manifold (PriorBank for both)

    Args:
        cfg: VFEConfig with all hyperparameters.
    """

    def __init__(self, cfg: VFEConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Build generators
        generators = self._build_generators(cfg)
        if not isinstance(generators, torch.Tensor):
            generators = torch.from_numpy(generators).float()
        self.register_buffer('generators', generators)
        cfg.generators = generators

        # Modules
        # PriorBank is always used for ENCODE (gauge-orbit token initialization).
        # cfg.use_prior_bank controls only whether DECODE is the KL-to-prior
        # readout (True, Law 3) or a plain nn.Linear(K, V) projection on mu_final
        # (False, the one documented neural exception in CLAUDE.md).
        self.prior_bank = VFEPriorBank(cfg, generators)
        self.use_prior_bank = cfg.use_prior_bank
        if not cfg.use_prior_bank:
            # Linear output projection. Initialize Xavier-uniform — matches
            # nn.Linear default but documented for the reader. No bias: bias
            # is a constant additive shift in V and doesn't change softmax /
            # cross-entropy gradients meaningfully under cross-entropy training
            # with a balanced vocabulary; keeping it parameter-free matches
            # the "minimal neural exception" framing in CLAUDE.md.
            self.output_proj = nn.Linear(cfg.embed_dim, cfg.vocab_size, bias=False)
        else:
            self.output_proj = None
        self.pos_enc = VFEPositionalEncoding(
            cfg, generators.shape[0], generators, irrep_dims=cfg.irrep_dims,
        )
        self.stack = VFEStack(cfg, generators)

        # Final normalization (applied after last layer, before decode).
        # For 'centered_mahalnorm' there is no natural prior at the model
        # output, so we omit mu_prior at the call site (line ~141) and the
        # centered norm degenerates to MahalanobisNorm. The centered
        # variant's effect is delivered at the per-block sites where the
        # layer prior is the natural reference. See VFEBlock.forward.
        self.final_norm = _resolve_vfe_norm(cfg.norm_type, cfg.embed_dim)

        # Active inference callback (Section 10)
        self._active_inference_fn: "Optional[VFEActiveInference]" = None
        if cfg.active_inference:
            from transformer.vfe.active_inference import VFEActiveInference
            self.active_inference_module = VFEActiveInference(cfg, self.prior_bank)
            self._active_inference_fn = self.active_inference_module

    def forward(
        self,
        token_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        r"""Full forward pass: encode → positional → stack → norm → decode.

        The E-step infers beliefs from context only. Targets are used
        exclusively for the M-step cross-entropy loss (never enter the E-step).

        Args:
            token_ids: ``(B, N)`` input token IDs.
            targets: ``(B, N)`` target token IDs for loss computation.
                If provided, returns ``(logits, loss, ce_for_log)``. If None,
                returns ``logits``.

        Returns:
            logits: ``(B, N, V)`` token logit predictions.
            loss: Optimizer scalar (CE plus any active regularizers); only
                returned when ``targets`` is provided.
            ce_for_log: Unscaled, regularizer-free cross-entropy detached
                for reporting (``PPL = exp(ce_for_log)``,
                ``BPC = ce_for_log / ln 2``); only returned with ``targets``.
        """
        B, N = token_ids.shape

        # Invalidate decode cache from prior forward pass. The cache is only
        # populated by PriorBank.decode; harmless to call when use_prior_bank
        # is False, but skip to avoid the no-op write.
        if self.use_prior_bank:
            self.prior_bank.invalidate_cache()

        # 1. Encode: tokens → Gaussian beliefs (Section 2)
        beliefs = self.prior_bank.encode(token_ids)

        # 2. Positional BCH composition (Section 3)
        beliefs = beliefs._replace(
            phi=self.pos_enc(beliefs.phi, N)
        )

        # 2b. Omega-direct initialization. After positional BCH, compute the
        # per-block (Ω_i, Ω_i^{-1}) pair from φ and stash it in beliefs.omega.
        # The E-step then iterates Ω directly, leaving φ at its encode-time
        # value (used only by RoPE if active, by the mass_φ penalty, and by
        # downstream diagnostics). One-time cost per forward pass; reuses the
        # fused matrix-exp kernel that the φ-mode path already calls.
        if self.cfg.gauge_parameterization == 'omega_direct':
            from transformer.vfe.omega_direct import init_omega_from_phi
            omega_pairs = init_omega_from_phi(
                beliefs.phi, self.generators, self.cfg.irrep_dims,
            )
            beliefs = beliefs._replace(omega=omega_pairs)

        # 3. Store initial beliefs as priors for cross-layer handoff.
        # Clone sigma/phi to defend against any downstream in-place mutation
        # (e.g., stack.py's diag clamp) writing through the posterior path.
        initial_priors = BeliefState(
            mu=beliefs.mu,
            sigma=beliefs.sigma.clone(),
            phi=beliefs.phi.clone(),
        )

        # 4. Causal mask
        mask = torch.tril(torch.ones(N, N, device=token_ids.device))
        mask = mask.unsqueeze(0).expand(B, -1, -1)  # (B, N, N)

        # 5. VFE stack: L layers of E-step + normalization (Sections 4-7)
        beliefs = self.stack(
            beliefs, initial_priors, mask,
            active_inference_fn=self._active_inference_fn,
        )

        # 6. Final normalization (Section 8)
        mu_final = beliefs.mu
        if self.final_norm is not None:
            mu_final = self.final_norm(mu_final, beliefs.sigma)

        # 7. Decode: beliefs → logits. Two paths gated by cfg.use_prior_bank.
        if self.use_prior_bank:
            # Law-3 decode: KL-to-prior on the gauge manifold. Pass the
            # configured decode_tau so the user-set softmax temperature
            # reaches the logit construction (defaulted to 1.0 previously).
            logits = self.prior_bank.decode(
                mu_final, beliefs.sigma, tau=self.cfg.decode_tau,
            )
        else:
            # Linear projection ablation: mu_final → logits, sigma discarded.
            # No KL geometry at the decode boundary; only the encode + E-step
            # paths remain gauge-aware.
            logits = self.output_proj(mu_final)

        if targets is not None:
            ce_loss = F.cross_entropy(
                logits.view(-1, self.cfg.vocab_size),
                targets.view(-1),
                ignore_index=-100,
            )
            # Capture the unscaled, regularizer-free CE for reporting BEFORE
            # any optional 1/sqrt(K) scaling or mass_phi penalty is applied.
            # PPL = exp(ce_for_log) and BPC = ce_for_log / ln(2) are the
            # only correct derivations regardless of how the optimizer's
            # combined `loss` is composed.
            ce_for_log = ce_loss.detach()

            # Normalize CE by sqrt(K) to match VFE dim_scale normalization.
            # Aux-hyperparameter and mass_phi terms are normalized by the
            # SAME factor below so the gradient ratio between CE and the
            # auxiliary terms is invariant to this rescaling. Without this
            # the aux gradient to raw_c0/raw_b0/log_kappa is `sqrt(K)` times
            # larger than the CE gradient under normalize_ce_by_dim=True.
            if self.cfg.normalize_ce_by_dim:
                loss_scale = 1.0 / (self.cfg.embed_dim ** 0.5)
                ce_loss = ce_loss * loss_scale
            else:
                loss_scale = 1.0

            loss = ce_loss

            # Gauge prior: (mass_φ/2) mean(||φ_i||²) over all positions
            if self.cfg.mass_phi > 0:
                # Normalize by (B * N) so penalty is per-position, independent of
                # batch size, sequence length, and generator count
                phi_norm_sq = (beliefs.phi ** 2).sum() / (beliefs.phi.shape[0] * beliefs.phi.shape[1])
                loss = loss + loss_scale * (0.5 * self.cfg.mass_phi * phi_norm_sq)

            # Auxiliary hyperparameter loss: each E-step caches a scalar F
            # (mu/sigma/phi detached, kappa/alpha attached) so raw_c0,
            # raw_b0, log_kappa receive gradients on the outer backward.
            # Scaled by `loss_scale` so the gradient magnitude matches CE
            # under normalize_ce_by_dim=True.
            for block in self.stack.blocks:
                aux = getattr(block.e_step, '_aux_hyperparam_loss', None)
                if aux is not None:
                    loss = loss + loss_scale * aux

            return logits, loss, ce_for_log

        return logits

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
        use_efe: bool = False,
        efe_gamma: float = 1.0,
    ) -> torch.Tensor:
        r"""Autoregressive generation with optional EFE policy (Section 11).

        When ``use_efe=True``, next-token selection uses expected free energy:
        :math:`q(a) \propto \exp(-\gamma\, G(a))` where G = risk + ambiguity.

        Args:
            prompt_ids: ``(1, N_prompt)`` or ``(N_prompt,)`` prompt token IDs.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-k filtering.
            use_efe: If True, use EFE-weighted action selection.
            efe_gamma: Inverse temperature for EFE policy.

        Returns:
            ``(1, N_prompt + max_new_tokens)`` generated token IDs.
        """
        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)

        ids = prompt_ids
        max_len = self.cfg.max_seq_len

        # Build EFE policy if requested (Section 11)
        efe_policy = None
        if use_efe:
            from transformer.vfe.efe import VFEExpectedFreeEnergy
            efe_policy = VFEExpectedFreeEnergy(
                self, gamma=efe_gamma,
                epistemic_weight=self.cfg.epistemic_weight,
                epistemic_samples=self.cfg.epistemic_samples,
            )

        for _ in range(max_new_tokens):
            # Truncate to max_seq_len
            ids_cond = ids if ids.shape[1] <= max_len else ids[:, -max_len:]

            if efe_policy is not None:
                next_id = efe_policy.select_action(
                    ids_cond, top_k=top_k, temperature=temperature,
                )
                ids = torch.cat(
                    [ids, torch.tensor([[next_id]], device=ids.device, dtype=ids.dtype)],
                    dim=1,
                )
            else:
                logits = self.forward(ids_cond)
                logits_next = logits[:, -1, :] / temperature  # (1, V)

                # Top-k filtering
                if top_k > 0:
                    v, _ = torch.topk(logits_next, min(top_k, logits_next.size(-1)))
                    logits_next[logits_next < v[:, [-1]]] = -float('inf')

                probs = F.softmax(logits_next, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
                ids = torch.cat([ids, next_id], dim=1)

        return ids

    @staticmethod
    def _build_generators(cfg: VFEConfig) -> torch.Tensor:
        """Build Lie algebra generators from config."""
        try:
            from math_utils.generators import (
                generate_so3_generators,
                generate_soN_generators,
                generate_glK_generators,
                generate_glK_multihead_generators,
                generate_multi_irrep_generators,
                generate_multi_irrep_soN_generators,
            )
        except ImportError as e:
            raise ImportError(
                f"math_utils.generators not available: {e}. "
                f"Cannot build gauge group generators."
            )

        K = cfg.embed_dim
        irrep_spec = cfg.irrep_spec

        if cfg.gauge_group == 'SO3':
            return generate_multi_irrep_generators(irrep_spec)

        if cfg.gauge_group == 'GLK':
            # Check if multihead
            if len(irrep_spec) == 1 and irrep_spec[0][1] > 1:
                _, n_heads, d_head = irrep_spec[0]
                return generate_glK_multihead_generators(K, n_heads)
            return generate_glK_generators(K)

        # SON
        if cfg.gauge_group == 'SON':
            return generate_multi_irrep_soN_generators(irrep_spec, K)

        raise ValueError(f"Unknown gauge_group: {cfg.gauge_group}")
