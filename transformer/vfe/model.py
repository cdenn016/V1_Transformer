"""
VFEModel: full gauge-theoretic VFE transformer.

    token_ids → PriorBank.encode → positional BCH → VFEStack → norm → PriorBank.decode → logits

No nn.Linear in default config — PriorBank IS the decoder (Law 3).
No targets in E-step — beliefs inferred from context only (Law 1).
All transport uses sandwich product (Law 2).
"""

import math
from typing import Optional, Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer.core.types import BeliefState
from transformer.core.blocks import MahalanobisNorm, RMSNorm
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
        self.prior_bank = VFEPriorBank(cfg, generators)
        self.pos_enc = VFEPositionalEncoding(
            cfg, generators.shape[0], generators, irrep_dims=cfg.irrep_dims,
        )
        self.stack = VFEStack(cfg, generators)

        # Final normalization (applied after last layer, before decode)
        if cfg.norm_type == 'mahalnorm':
            self.final_norm = MahalanobisNorm(cfg.embed_dim)
        elif cfg.norm_type == 'rmsnorm':
            self.final_norm = RMSNorm(cfg.embed_dim)
        elif cfg.norm_type == 'none':
            self.final_norm = None
        else:
            raise ValueError(
                f"VFEConfig.norm_type={cfg.norm_type!r} not recognized; "
                "expected 'mahalnorm', 'rmsnorm', or 'none'."
            )

        # Active inference callback (Section 10)
        self._active_inference_fn = None
        if cfg.active_inference:
            from transformer.vfe.active_inference import VFEActiveInference
            self.active_inference_module = VFEActiveInference(cfg, self.prior_bank)
            self._active_inference_fn = self.active_inference_module

    def forward(
        self,
        token_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        r"""Full forward pass: encode → positional → stack → norm → decode.

        The E-step infers beliefs from context only. Targets are used
        exclusively for the M-step cross-entropy loss (never enter the E-step).

        Args:
            token_ids: ``(B, N)`` input token IDs.
            targets: ``(B, N)`` target token IDs for loss computation.
                If provided, returns ``(logits, loss)``. If None, returns ``logits``.

        Returns:
            logits: ``(B, N, V)`` token logit predictions.
            loss: Cross-entropy loss (only if targets provided).
        """
        B, N = token_ids.shape

        # Invalidate decode cache from prior forward pass
        self.prior_bank.invalidate_cache()

        # 1. Encode: tokens → Gaussian beliefs (Section 2)
        beliefs = self.prior_bank.encode(token_ids)

        # 2. Positional BCH composition (Section 3)
        beliefs = beliefs._replace(
            phi=self.pos_enc(beliefs.phi, N)
        )

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

        # 7. Decode: beliefs → logits via KL-to-prior (Section 9)
        logits = self.prior_bank.decode(mu_final, beliefs.sigma)

        if targets is not None:
            ce_loss = F.cross_entropy(
                logits.view(-1, self.cfg.vocab_size),
                targets.view(-1),
                ignore_index=-100,
            )
            # Normalize CE by sqrt(K) to match VFE dim_scale normalization
            if self.cfg.normalize_ce_by_dim:
                ce_loss = ce_loss / (self.cfg.embed_dim ** 0.5)

            loss = ce_loss

            # Gauge prior: (mass_φ/2) mean(||φ_i||²) over all positions
            if self.cfg.mass_phi > 0:
                # Normalize by (B * N) so penalty is per-position, independent of
                # batch size, sequence length, and generator count
                phi_norm_sq = (beliefs.phi ** 2).sum() / (beliefs.phi.shape[0] * beliefs.phi.shape[1])
                loss = loss + 0.5 * self.cfg.mass_phi * phi_norm_sq

            return logits, loss

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
