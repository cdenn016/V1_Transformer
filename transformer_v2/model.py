# -*- coding: utf-8 -*-
"""
Gauge-Transformer Language Model (Refactored)
===============================================

token_ids → (μ, Σ, φ) → Transformer Stack → μ_final → logits

All configuration via GaugeTransformerConfig.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List, Union
import numpy as np

from .config import GaugeTransformerConfig
from .embeddings import GaugeTokenEmbedding, GaugePositionalEncoding
from .blocks import GaugeTransformerStack
from .attention import create_attention_mask

# Import generators (required — not optional in refactored code)
try:
    from .generators import (
        generate_so3_generators,
        generate_soN_generators,
        generate_multi_irrep_generators,
        generate_multi_irrep_soN_generators,
        generate_glK_generators,
        generate_glK_multihead_generators,
        generate_glK_cross_head_generators,
        merge_coupled_heads,
        reorder_cross_head_generators,
    )
    GENERATORS_AVAILABLE = True
except ImportError:
    GENERATORS_AVAILABLE = False


class GaugeTransformerLM(nn.Module):
    """Complete gauge-theoretic language model.

    Architecture: token_ids → (μ, Σ, φ) → N × Blocks → μ → logits
    """

    def __init__(self, config: GaugeTransformerConfig):
        super().__init__()
        self.config = config

        # Cross-head coupling state
        self._cross_head_perm = None
        self._super_block_dims = None

        # ── Generator construction ──────────────────────────────────────
        generators = self._build_generators(config)
        self.register_buffer('generators', torch.from_numpy(generators).float())

        # ── Embedding layers ────────────────────────────────────────────
        self.token_embed = GaugeTokenEmbedding(config, self.generators)
        self.pos_encoding = GaugePositionalEncoding(config, self.generators)

        # ── PriorBank (pure FEP mode) ──────────────────────────────────
        self.prior_bank = None
        if config.pure_fep_mode and config.use_prior_bank:
            from .prior_bank import PriorBank
            self.prior_bank = PriorBank(
                vocab_size=config.vocab_size,
                embed_dim=config.embed_dim,
                init_std=config.mu_init_std,
                init_sigma_scale=1.0,
                learnable_sigma=config.evolve_sigma,
                gauge_fixed_priors=config.gauge_fixed_priors,
                generators=self.generators if config.gauge_fixed_priors else None,
                phi_dim=config.phi_dim,
            )

        # ── Transformer stack ───────────────────────────────────────────
        # Override irrep_dims in config if cross-head coupling changed block structure
        if self._super_block_dims is not None:
            config.irrep_dims = self._super_block_dims

        self.transformer = GaugeTransformerStack(
            config, self.generators, prior_bank=self.prior_bank
        )

        # ── Output projection ───────────────────────────────────────────
        self.out_proj = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

        # Initialize weights (before tying)
        self.apply(self._init_weights)

        # Tie input/output embeddings
        if config.tie_embeddings and not config.gauge_fixed_priors:
            self.out_proj.weight = self.token_embed.mu_embed.weight

    def _build_generators(self, config: GaugeTransformerConfig) -> np.ndarray:
        """Build Lie algebra generators based on gauge group config."""
        if not GENERATORS_AVAILABLE:
            import warnings
            warnings.warn(
                "math_utils/generators.py import failed. Using random fallback.",
                RuntimeWarning,
            )
            n_gen = config.phi_dim
            rng = np.random.RandomState(seed=42)
            G = rng.randn(n_gen, config.embed_dim, config.embed_dim)
            return 0.5 * (G - G.transpose(0, 2, 1))

        if config.gauge_group == 'SO3':
            if config.use_multi_irrep and config.irrep_spec is not None:
                return generate_multi_irrep_generators(config.irrep_spec)
            return generate_so3_generators(config.embed_dim)

        elif config.gauge_group == 'GLK':
            return self._build_glk_generators(config)

        else:  # SO(N)
            if config.use_multi_irrep and config.irrep_spec is not None:
                return generate_multi_irrep_soN_generators(
                    config.irrep_spec, config.gauge_dim
                )
            return generate_soN_generators(config.gauge_dim)

    def _build_glk_generators(self, config: GaugeTransformerConfig) -> np.ndarray:
        """Build GL(K) generators, handling multi-head and cross-head coupling."""
        is_multihead = (
            config.use_multi_irrep
            and config.irrep_spec is not None
            and len(config.irrep_spec) == 1
            and config.irrep_spec[0][0] != 'full'
            and config.irrep_spec[0][1] > 1
        )

        if not is_multihead:
            return generate_glK_generators(config.embed_dim)

        _, n_heads, d_head = config.irrep_spec[0]

        if not config.cross_couplings:
            return generate_glK_multihead_generators(config.embed_dim, n_heads)

        # Cross-head coupling
        generators = generate_glK_cross_head_generators(
            config.embed_dim, n_heads, config.cross_couplings
        )
        super_block_dims, super_block_head_groups = merge_coupled_heads(
            n_heads, d_head, config.cross_couplings
        )
        generators, perm = reorder_cross_head_generators(
            generators, n_heads, d_head,
            config.cross_couplings, super_block_head_groups,
        )
        self._cross_head_perm = perm
        self._super_block_dims = super_block_dims
        return generators

    def _init_weights(self, module):
        """Initialize weights. Skip nn.Embedding (handled by GaugeTokenEmbedding)."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def _apply_cross_head_perm(self, mu_q, sigma_q, device):
        """Apply cross-head permutation to align dims with generator blocks."""
        if self._cross_head_perm is None:
            return mu_q, sigma_q
        perm = torch.from_numpy(self._cross_head_perm).to(device=device, dtype=torch.long)
        mu_q = mu_q[:, :, perm]
        if sigma_q is not None:
            if sigma_q.dim() == 3:
                sigma_q = sigma_q[:, :, perm]
            else:
                sigma_q = sigma_q[:, :, perm][:, :, :, perm]
        return mu_q, sigma_q

    def _apply_inverse_perm(self, mu_q, device):
        """Inverse cross-head permutation to restore original dim order."""
        if self._cross_head_perm is None:
            return mu_q
        inv_perm = torch.from_numpy(
            np.argsort(self._cross_head_perm)
        ).to(device=device, dtype=torch.long)
        return mu_q[:, :, inv_perm]

    def forward(
        self,
        token_ids: torch.Tensor,
        return_agents: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """Forward pass: token_ids → logits.

        Args:
            token_ids: (B, N) token indices
            return_agents: If True, return intermediate agent states

        Returns:
            logits: (B, N, V) or (logits, agent_states) if return_agents
        """
        batch_size, num_agents = token_ids.shape
        device = token_ids.device

        # Embed + position encode
        mu_q, sigma_q, phi = self.token_embed(token_ids)
        mu_q, sigma_q = self._apply_cross_head_perm(mu_q, sigma_q, device)
        mu_prior = mu_q.clone()
        phi = self.pos_encoding.compose(phi, num_agents, device=device)

        # Attention mask
        mask = create_attention_mask(
            num_agents=num_agents,
            pattern=self.config.attention_pattern,
            device=device,
            causal=True,
        )
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)

        # Precompute transport operators when phi is static
        cached_head_transports = None
        if not self.config.evolve_phi:
            first_attention = self.transformer.blocks[0].attention
            cached_head_transports = first_attention.precompute_head_transports(
                phi, device, mu_q.dtype
            )

        # Transformer stack
        mu_q, sigma_q, phi, intermediates = self.transformer(
            mu_q, sigma_q, phi, self.generators,
            mask=mask, mu_prior=mu_prior,
            token_ids=token_ids,
            return_intermediates=return_agents,
            cached_head_transports=cached_head_transports,
        )

        mu_q = self._apply_inverse_perm(mu_q, device)
        logits = self.out_proj(mu_q)

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
        """Forward pass returning per-layer attention weights and KL matrices.

        Used during training for attention-weighted free energy loss.

        Returns:
            logits: (B, N, V)
            attention_info: Dict with beta, kl, mu, sigma, phi, priors
        """
        batch_size, num_agents = token_ids.shape
        device = token_ids.device

        # Embed
        mu_q, sigma_q, phi = self.token_embed(token_ids)
        mu_q, sigma_q = self._apply_cross_head_perm(mu_q, sigma_q, device)
        mu_prior = mu_q.clone()
        sigma_prior = sigma_q.clone() if sigma_q is not None else None
        phi_prior = phi.clone()
        phi = self.pos_encoding.compose(phi, num_agents, device=device)

        mask = create_attention_mask(
            num_agents=num_agents,
            pattern=self.config.attention_pattern,
            device=device, causal=True,
        )
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)

        cached_head_transports = None
        if not self.config.evolve_phi:
            first_attention = self.transformer.blocks[0].attention
            cached_head_transports = first_attention.precompute_head_transports(
                phi, device, mu_q.dtype
            )

        # Manual layer-by-layer for attention tracking
        all_betas = []
        all_kls = []
        n_blocks = len(self.transformer.blocks)

        for layer_idx, block in enumerate(self.transformer.blocks):
            is_final = (layer_idx == n_blocks - 1)

            # Attention sublayer
            mu_normalized = block.norm1(mu_q)
            mu_attn, sigma_attn, beta, kl = block.attention(
                mu_normalized, sigma_q, phi, self.generators,
                mask=mask, return_attention=True,
                cached_head_transports=cached_head_transports,
            )
            all_betas.append(beta)
            all_kls.append(kl)

            if block.use_residual:
                mu_q = mu_q + mu_attn
            else:
                mu_q = mu_attn
            if block.evolve_sigma and sigma_attn is not None:
                sigma_q = sigma_attn

            # FFN sublayer
            mu_normalized = block.norm2(mu_q)
            mu_ffn, sigma_ffn, phi_out, _beta_hist = block.ffn(
                mu=mu_normalized, beta=beta, mu_prior=mu_prior,
                phi=phi, sigma=sigma_q, mask=mask,
                targets=targets if is_final else None,
                W_out=self.out_proj.weight if hasattr(self.out_proj, 'weight') else None,
                token_ids=token_ids,
            )

            if block.evolve_sigma and sigma_ffn is not None:
                sigma_q = sigma_ffn
            if block.use_residual:
                mu_q = mu_q + mu_ffn
            else:
                mu_q = mu_ffn
            if phi_out is not None:
                phi = phi_out

        mu_q = self.transformer.final_norm(mu_q)
        mu_q = self._apply_inverse_perm(mu_q, device)
        logits = self.out_proj(mu_q)

        valid_betas = [b for b in all_betas if b is not None]
        valid_kls = [k for k in all_kls if k is not None]

        attention_info = {
            'beta': torch.stack(valid_betas, dim=0) if valid_betas else None,
            'kl': torch.stack(valid_kls, dim=0) if valid_kls else None,
            'n_layers': n_blocks,
            'mu': mu_q,
            'sigma': sigma_q,
            'phi': phi,
            'mu_prior': mu_prior,
            'sigma_prior': sigma_prior,
            'phi_prior': phi_prior,
        }
        return logits, attention_info

    def forward_with_rg_tracking(
        self,
        token_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """Forward pass tracking RG flow across VFE iterations.

        Captures beta_history showing how attention evolves during VFE descent.

        Returns:
            logits: (B, N, V)
            rg_info: Dict with beta_history, all_layer_betas, mu, sigma, phi
        """
        batch_size, num_agents = token_ids.shape
        device = token_ids.device

        mu_q, sigma_q, phi = self.token_embed(token_ids)
        mu_q, sigma_q = self._apply_cross_head_perm(mu_q, sigma_q, device)
        mu_prior = mu_q.clone()
        phi = self.pos_encoding.compose(phi, num_agents, device=device)

        mask = create_attention_mask(
            num_agents=num_agents,
            pattern=self.config.attention_pattern,
            device=device, causal=True,
        )
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)

        cached_head_transports = None
        if not self.config.evolve_phi:
            first_attention = self.transformer.blocks[0].attention
            cached_head_transports = first_attention.precompute_head_transports(
                phi, device, mu_q.dtype
            )

        all_layer_betas = []
        n_blocks = len(self.transformer.blocks)
        beta_history = None

        for layer_idx, block in enumerate(self.transformer.blocks):
            is_final = (layer_idx == n_blocks - 1)

            mu_normalized = block.norm1(mu_q)
            mu_attn, sigma_attn, beta, kl = block.attention(
                mu_normalized, sigma_q, phi, self.generators,
                mask=mask, return_attention=True,
                cached_head_transports=cached_head_transports,
            )
            all_layer_betas.append(beta.detach() if beta is not None else None)

            if block.use_residual:
                mu_q = mu_q + mu_attn
            else:
                mu_q = mu_attn
            if block.evolve_sigma and sigma_attn is not None:
                sigma_q = sigma_attn

            mu_normalized = block.norm2(mu_q)
            mu_ffn, sigma_ffn, phi_out, layer_beta_history = block.ffn(
                mu=mu_normalized, beta=beta, mu_prior=mu_prior,
                phi=phi, sigma=sigma_q, mask=mask,
                targets=targets if is_final else None,
                W_out=self.out_proj.weight if hasattr(self.out_proj, 'weight') else None,
                token_ids=token_ids,
                return_beta_history=is_final,
            )

            if is_final:
                beta_history = layer_beta_history

            if block.evolve_sigma and sigma_ffn is not None:
                sigma_q = sigma_ffn
            if block.use_residual:
                mu_q = mu_q + mu_ffn
            else:
                mu_q = mu_ffn
            if phi_out is not None:
                phi = phi_out

        mu_q = self.transformer.final_norm(mu_q)
        mu_q = self._apply_inverse_perm(mu_q, device)
        logits = self.out_proj(mu_q)

        rg_info = {
            'beta_history': beta_history,
            'all_layer_betas': all_layer_betas,
            'mu': mu_q,
            'sigma': sigma_q,
            'phi': phi,
            'n_iterations': self.transformer.blocks[-1].ffn.n_iterations,
            'n_layers': n_blocks,
            'beta_final': beta_history[-1] if beta_history else all_layer_betas[-1],
        }
        return logits, rg_info

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """Autoregressive generation.

        Args:
            prompt_ids: (1, prompt_len) initial tokens
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling

        Returns:
            (1, prompt_len + max_new_tokens) generated sequence
        """
        was_training = self.training
        self.eval()
        try:
            generated = prompt_ids.clone()
            max_seq_len = self.config.max_seq_len

            for _ in range(max_new_tokens):
                if generated.shape[1] > max_seq_len:
                    generated = generated[:, -max_seq_len:]

                result = self.forward(generated)
                logits = result[0] if isinstance(result, tuple) else result
                logits_next = logits[:, -1, :] / temperature

                if top_k is not None:
                    v, _ = torch.topk(logits_next, min(top_k, logits_next.size(-1)))
                    logits_next[logits_next < v[:, [-1]]] = -float('inf')

                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits_next, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits_next[indices_to_remove] = -float('inf')

                probs = F.softmax(logits_next, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)

            return generated
        finally:
            if was_training:
                self.train()

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Return parameter count, optionally excluding embeddings."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            if hasattr(self.token_embed, 'mu_embed'):
                n_params -= self.token_embed.mu_embed.weight.numel()
            elif hasattr(self.token_embed, 'base_mu'):
                n_params -= self.token_embed.base_mu.numel()
                n_params -= self.token_embed.base_log_sigma_diag.numel()
                n_params -= self.token_embed.phi_embed.weight.numel()
        return n_params

    # ── P-flow: EMA update of embeddings toward successful beliefs ───
    def p_flow_update(
        self,
        token_ids: torch.Tensor,
        mu_beliefs: torch.Tensor,
        prediction_errors: torch.Tensor,
        ema_decay: float = 0.99,
    ):
        """Update token embeddings toward beliefs that predicted well."""
        if hasattr(self.token_embed, 'update_embeddings_from_beliefs'):
            self.token_embed.update_embeddings_from_beliefs(
                token_ids=token_ids,
                mu_beliefs=mu_beliefs,
                prediction_errors=prediction_errors,
                ema_decay=ema_decay,
            )

    def delta_rule_update_w_out(
        self,
        mu_beliefs: torch.Tensor,
        targets: torch.Tensor,
        lr: float = 0.001,
    ):
        """Backprop-free delta rule update for W_out."""
        with torch.no_grad():
            B, N, K = mu_beliefs.shape
            V = self.config.vocab_size

            logits = self.out_proj(mu_beliefs)
            predictions = F.softmax(logits, dim=-1)
            targets_onehot = F.one_hot(targets, num_classes=V).float()
            error = targets_onehot - predictions

            error_flat = error.reshape(-1, V)
            mu_flat = mu_beliefs.reshape(-1, K)
            delta_W = error_flat.t() @ mu_flat
            delta_W /= (B * N)

            self.out_proj.weight.add_(lr * delta_W)
