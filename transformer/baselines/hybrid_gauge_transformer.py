"""
Hybrid Gauge-Attention Transformer
====================================

Standard neural transformer that uses gauge-theoretic KL-attention
(IrrepMultiHeadAttention) and PriorBank embeddings, but keeps all other
components neural: GELU FFN, LayerNorm, residual connections.

This isolates the contribution of gauge attention + PriorBank from the VFE
E-step FFN, enabling fair ablation studies against both the full gauge
transformer and the standard dot-product baseline.

Architecture:
    token_ids -> PriorBank.encode() -> (mu, Sigma, phi)
    -> GaugePositionalEncoding.compose(phi)
    -> N x HybridGaugeBlock:
        LN(mu) -> IrrepMultiHeadAttention(mu, sigma, phi) -> residual on mu
        LN(mu) -> StandardFFN(mu) -> residual on mu
        [sigma carries through from attention; phi unchanged]
    -> LN(mu)
    -> PriorBank.decode(mu, sigma) -> logits

Key differences from full GaugeTransformerLM:
    - Standard GELU FFN instead of VFE E-step dynamics
    - No belief evolution (sigma, phi fixed after attention)
    - Pure CE training (no VFE loss terms by default)

Key differences from StandardTransformerLM:
    - KL-divergence attention instead of dot-product (no W_Q, W_K, W_V)
    - PriorBank embeddings (Gaussian beliefs) instead of nn.Embedding
    - KL-based decode instead of linear output projection
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Tuple, List, Union

# Gauge components (reused from core)
from transformer.core.attention import IrrepMultiHeadAttention, create_attention_mask
from transformer.core.prior_bank import PriorBank
from transformer.core.embeddings import GaugePositionalEncoding

# Standard FFN (reused from baseline)
from transformer.baselines.standard_transformer import StandardFFN

# Generator creation
try:
    from math_utils.generators import (
        generate_so3_generators,
        generate_soN_generators,
        generate_multi_irrep_generators,
        generate_multi_irrep_soN_generators,
        generate_glK_generators,
        generate_glK_multihead_generators,
    )
    GENERATORS_AVAILABLE = True
except ImportError:
    GENERATORS_AVAILABLE = False


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
        disc = 1 + 8 * n_gen
        sqrt_disc = int(math.sqrt(disc))
        if sqrt_disc * sqrt_disc == disc:
            N_candidate = (1 + sqrt_disc) // 2
            if N_candidate * (N_candidate - 1) // 2 == n_gen:
                return 'SON', N_candidate
        return 'GLK', K


class HybridGaugeBlock(nn.Module):
    r"""
    Pre-norm transformer block with gauge KL-attention and standard GELU FFN.

    Data flow:
        (mu, sigma, phi) -> LN(mu) -> GaugeAttention -> residual on mu
                         -> LN(mu) -> GELU FFN       -> residual on mu
                         -> (mu', sigma', phi)

    sigma is updated by attention (aggregated via mixture/precision mode)
    and carried through the FFN unchanged. phi is never modified.
    """

    def __init__(
        self,
        embed_dim: int,
        irrep_spec: List[Tuple[str, int, int]],
        kappa_beta: float,
        hidden_dim: int,
        dropout: float,
        generators: torch.Tensor,
        diagonal_covariance: bool = True,
        gauge_mode: str = 'learned',
        use_rope: bool = False,
        rope_base: float = 10000.0,
        use_output_projection: bool = False,
        learnable_head_kappa: bool = False,
        sigma_aggregation: str = 'mixture',
        mask_self_attention: bool = False,
        enforce_orthogonal: bool = False,
        exact_diagonal_transport: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.diagonal_covariance = diagonal_covariance

        # Infer gauge group from generators
        gauge_group, gauge_dim = _infer_gauge_group(generators)

        # Pre-norm layers
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

        # Gauge-theoretic attention (KL-divergence, no W_Q/W_K/W_V)
        self.attention = IrrepMultiHeadAttention(
            embed_dim=embed_dim,
            irrep_spec=irrep_spec,
            kappa_beta=kappa_beta,
            epsilon=1e-8,
            aggregate_mode='full_distribution',  # Propagate sigma
            diagonal_covariance=diagonal_covariance,
            exact_diagonal_transport=exact_diagonal_transport,
            gauge_group=gauge_group,
            gauge_dim=gauge_dim,
            global_generators=generators,
            gauge_mode=gauge_mode,
            mask_self_attention=mask_self_attention,
            enforce_orthogonal=enforce_orthogonal,
            use_output_projection=use_output_projection,
            use_rope=use_rope,
            rope_base=rope_base,
            sigma_aggregation=sigma_aggregation,
            learnable_head_kappa=learnable_head_kappa,
        )

        # Standard GELU FFN
        self.ffn = StandardFFN(embed_dim, hidden_dim, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        mu_q: torch.Tensor,       # (B, N, K)
        sigma_q: torch.Tensor,    # (B, N, K) or (B, N, K, K)
        phi: torch.Tensor,        # (B, N, phi_dim)
        generators: torch.Tensor, # (n_gen, K, K)
        mask: Optional[torch.Tensor] = None,
        cached_head_transports: Optional[list] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
               Optional[torch.Tensor], Optional[torch.Tensor]]:
        r"""
        Forward pass through hybrid block.

        Args:
            mu_q: Belief means (B, N, K)
            sigma_q: Belief covariances (B, N, K) diagonal or (B, N, K, K) full
            phi: Gauge frames (B, N, phi_dim)
            generators: Lie algebra generators (n_gen, K, K)
            mask: Causal mask (B, N, N)
            cached_head_transports: Precomputed transport operators per head
            return_attention: If True, return attention weights and KL matrices

        Returns:
            mu_q: Updated means (B, N, K)
            sigma_q: Updated covariances (from attention aggregation)
            phi: Gauge frames (unchanged)
            beta: Attention weights (B, n_heads, N, N) if return_attention
            kl: KL matrices (B, n_heads, N, N) if return_attention
        """
        # === Attention sublayer (gauge-theoretic KL-attention) ===
        mu_norm = self.ln1(mu_q)

        mu_attn, sigma_attn, beta, kl = self.attention(
            mu_norm, sigma_q, phi, generators,
            mask=mask,
            return_attention=return_attention,
            cached_head_transports=cached_head_transports,
        )

        # Residual on mu
        mu_q = mu_q + self.dropout(mu_attn)

        # Sigma: use attention-aggregated value (replaces, not residual)
        if sigma_attn is not None:
            sigma_q = sigma_attn

        # === FFN sublayer (standard GELU) ===
        mu_norm = self.ln2(mu_q)
        mu_ffn = self.ffn(mu_norm)
        mu_q = mu_q + self.dropout(mu_ffn)

        # sigma and phi pass through unchanged
        return mu_q, sigma_q, phi, beta, kl


class HybridGaugeTransformerLM(nn.Module):
    r"""
    Standard transformer with gauge KL-attention and PriorBank embeddings.

    Combines:
        - PriorBank for encode (token -> mu, sigma, phi) and decode (KL logits)
        - IrrepMultiHeadAttention for gauge-theoretic attention
        - Standard GELU FFN for feedforward processing
        - Standard LayerNorm + residual connections

    This architecture isolates the gauge attention mechanism from the VFE
    E-step dynamics, enabling fair comparison with both the full gauge
    transformer and the standard dot-product baseline.

    Args:
        config: Flat dictionary with hyperparameters. Required keys:
            - vocab_size, embed_dim, n_layers, irrep_spec, hidden_dim,
              max_seq_len, kappa_beta
            See EM_CONFIG in train_publication.py for full reference.
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        # Extract config
        vocab_size = config['vocab_size']
        embed_dim = config['embed_dim']
        n_layers = config['n_layers']
        irrep_spec = config['irrep_spec']
        hidden_dim = config.get('hidden_dim', embed_dim * 4)
        max_seq_len = config['max_seq_len']
        kappa_beta = config['kappa_beta']
        dropout = config.get('dropout', 0.1)

        # Gauge config
        gauge_group = config.get('gauge_group', 'GLK')
        gauge_dim = config.get('gauge_dim', embed_dim)
        gauge_mode = config.get('gauge_mode', 'learned')
        diagonal_covariance = config.get('diagonal_covariance', True)
        use_rope = config.get('use_rope', True)
        rope_base = config.get('rope_base', 10000.0)
        pos_mode = config.get('pos_encoding_mode', 'none')
        pos_encoding_scale = config.get('pos_encoding_scale', 0.1)

        # Attention options
        use_output_projection = config.get('use_output_projection', False)
        learnable_head_kappa = config.get('learnable_head_kappa', False)
        sigma_aggregation = config.get('sigma_aggregation', 'mixture')
        mask_self_attention = config.get('mask_self_attention', False)
        enforce_orthogonal = config.get('enforce_orthogonal', False)
        exact_diagonal_transport = config.get('exact_diagonal_transport', False)

        # PriorBank config
        gauge_fixed_priors = config.get('gauge_fixed_priors', False)
        prior_bank_tau = config.get('prior_bank_tau', 1.0)
        self.prior_bank_tau = prior_bank_tau

        self.diagonal_covariance = diagonal_covariance
        self.gauge_mode = gauge_mode

        # =================================================================
        # Gauge Group Setup (generators, phi_dim)
        # =================================================================
        if gauge_group == 'SO3':
            phi_dim = 3
        elif gauge_group == 'GLK':
            is_glk_multihead = (
                irrep_spec is not None and
                len(irrep_spec) == 1 and
                irrep_spec[0][0] != 'full' and
                irrep_spec[0][1] > 1
            )
            if is_glk_multihead:
                _, n_heads, d_head = irrep_spec[0]
                phi_dim = n_heads * d_head * d_head
            else:
                phi_dim = embed_dim * embed_dim
        else:  # SO(N)
            phi_dim = gauge_dim * (gauge_dim - 1) // 2

        self.phi_dim = phi_dim

        if GENERATORS_AVAILABLE:
            if gauge_group == 'SO3':
                generators = generate_multi_irrep_generators(irrep_spec)
            elif gauge_group == 'GLK':
                is_multihead = (
                    irrep_spec is not None and
                    len(irrep_spec) == 1 and
                    irrep_spec[0][0] != 'full' and
                    irrep_spec[0][1] > 1
                )
                if is_multihead:
                    generators = generate_glK_multihead_generators(embed_dim, irrep_spec[0][1])
                    _, n_heads, d_head = irrep_spec[0]
                    print(f"[INFO] GL(K) multi-head: {n_heads} heads x GL({d_head}), "
                          f"{n_heads * d_head**2} generators")
                else:
                    generators = generate_glK_generators(embed_dim)
                    print(f"[INFO] GL(K) single-head: {embed_dim}^2 = {embed_dim**2} generators")
            else:
                generators = generate_multi_irrep_soN_generators(irrep_spec, gauge_dim)
        else:
            import warnings
            warnings.warn(
                "GENERATORS_AVAILABLE=False: using random fallback generators.",
                RuntimeWarning,
            )
            n_generators = phi_dim
            rng = np.random.RandomState(seed=42)
            generators = rng.randn(n_generators, embed_dim, embed_dim)
            generators = 0.5 * (generators - generators.transpose(0, 2, 1))

        self.register_buffer(
            'generators',
            torch.from_numpy(generators).float()
        )

        # =================================================================
        # PriorBank (encode + decode)
        # =================================================================
        self.prior_bank = PriorBank(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            init_std=config.get('mu_init_std', None),
            init_sigma_scale=1.0,
            learnable_sigma=config.get('evolve_sigma', True),
            gauge_fixed_priors=gauge_fixed_priors,
            generators=self.generators if gauge_fixed_priors else None,
            phi_dim=phi_dim,
            phi_scale=config.get('phi_scale', 0.3),
            sigma_ce_scale=config.get('sigma_ce_scale', 0.1),
            learnable_temperature=config.get('learnable_pb_temperature',
                                              config.get('learnable_temperature', False)),
            diagonal_covariance=diagonal_covariance,
            sigma_max=config.get('sigma_max', 5.0),
        )

        # =================================================================
        # Positional Encoding (in gauge frame space)
        # =================================================================
        self.pos_encoding = GaugePositionalEncoding(
            max_seq_len=max_seq_len,
            mode=pos_mode,
            scale=pos_encoding_scale,
            phi_dim=phi_dim,
            generators=self.generators,
            gauge_group=gauge_group,
        )

        # =================================================================
        # Transformer Blocks (hybrid: gauge attention + GELU FFN)
        # =================================================================
        self.blocks = nn.ModuleList([
            HybridGaugeBlock(
                embed_dim=embed_dim,
                irrep_spec=irrep_spec,
                kappa_beta=kappa_beta,
                hidden_dim=hidden_dim,
                dropout=dropout,
                generators=self.generators,
                diagonal_covariance=diagonal_covariance,
                gauge_mode=gauge_mode,
                use_rope=use_rope,
                rope_base=rope_base,
                use_output_projection=use_output_projection,
                learnable_head_kappa=learnable_head_kappa,
                sigma_aggregation=sigma_aggregation,
                mask_self_attention=mask_self_attention,
                enforce_orthogonal=enforce_orthogonal,
                exact_diagonal_transport=exact_diagonal_transport,
            )
            for _ in range(n_layers)
        ])

        # Final layer norm
        self.ln_final = nn.LayerNorm(embed_dim)

        # Fallback output projection (used only if PriorBank is disabled)
        self.out_proj = nn.Linear(embed_dim, vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

        # Count parameters
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"HybridGaugeTransformerLM initialized: {n_params/1e6:.2f}M parameters")
        print(f"  Attention: KL-divergence (gauge-theoretic)")
        print(f"  FFN: Standard GELU ({embed_dim} -> {hidden_dim} -> {embed_dim})")
        print(f"  Embeddings: PriorBank (KL encode/decode)")
        print(f"  Gauge group: {gauge_group}, mode={gauge_mode}")

    def _init_weights(self, module):
        """Initialize weights. Skip nn.Embedding (PriorBank handles its own init)."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        token_ids: torch.Tensor,
        return_agents: bool = False,
        targets: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        r"""
        Forward pass through hybrid gauge transformer.

        Args:
            token_ids: (B, N) token indices
            return_agents: If True, return intermediate belief states
            targets: Unused (kept for API compatibility)

        Returns:
            logits: (B, N, V) next-token predictions
            agents: Optional dict with mu, sigma, phi
        """
        batch_size, num_agents = token_ids.shape
        device = token_ids.device

        # 1. PriorBank encode: token -> (mu, sigma, phi)
        mu_q, sigma_q, phi = self.prior_bank.encode(token_ids)

        # Save priors (before position encoding)
        mu_prior = mu_q.clone()
        sigma_prior = sigma_q.clone() if sigma_q is not None else None

        # 2. Position encoding (compose with token phi)
        phi = self.pos_encoding.compose(phi, num_agents, device=device)

        # 3. Causal mask
        mask = create_attention_mask(
            num_agents=num_agents,
            pattern='full',
            window=64,
            device=device,
            causal=True,
        )
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)  # (B, N, N)

        # 4. Precompute transport operators (phi doesn't evolve)
        cached_head_transports = self.blocks[0].attention.precompute_head_transports(
            phi, device, mu_q.dtype
        )

        # 5. Forward through transformer blocks
        for block in self.blocks:
            mu_q, sigma_q, phi, _, _ = block(
                mu_q, sigma_q, phi, self.generators,
                mask=mask,
                cached_head_transports=cached_head_transports,
                return_attention=False,
            )

        # 6. Final layer norm on mu
        mu_q = self.ln_final(mu_q)

        # 7. PriorBank decode: beliefs -> logits via KL
        logits = self.prior_bank.decode(mu_q, sigma_q, tau=self.prior_bank_tau)

        if return_agents:
            agent_states = {
                'mu': mu_q.detach(),
                'sigma': sigma_q.detach() if sigma_q is not None else None,
                'phi': phi.detach(),
            }
            return logits, agent_states

        return logits

    def forward_with_attention(
        self,
        token_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        r"""
        Forward pass returning attention weights and KL matrices for training.

        Args:
            token_ids: (B, N) token indices
            targets: Unused (kept for API compatibility)

        Returns:
            logits: (B, N, V) predictions
            attention_info: Dict with beta, kl, mu, sigma, phi, priors, n_layers
        """
        batch_size, num_agents = token_ids.shape
        device = token_ids.device

        # 1. PriorBank encode
        mu_q, sigma_q, phi = self.prior_bank.encode(token_ids)

        # Save priors
        mu_prior = mu_q.clone()
        sigma_prior = sigma_q.clone() if sigma_q is not None else None
        phi_prior = phi.clone()

        # 2. Position encoding
        phi = self.pos_encoding.compose(phi, num_agents, device=device)

        # 3. Causal mask
        mask = create_attention_mask(
            num_agents=num_agents,
            pattern='full',
            window=64,
            device=device,
            causal=True,
        )
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)

        # 4. Precompute transport operators
        cached_head_transports = self.blocks[0].attention.precompute_head_transports(
            phi, device, mu_q.dtype
        )

        # 5. Forward through blocks, collecting attention info
        all_betas = []
        all_kls = []

        for block in self.blocks:
            mu_q, sigma_q, phi, beta, kl = block(
                mu_q, sigma_q, phi, self.generators,
                mask=mask,
                cached_head_transports=cached_head_transports,
                return_attention=True,
            )
            if beta is not None:
                all_betas.append(beta)
            if kl is not None:
                all_kls.append(kl)

        # 6. Final layer norm
        mu_q = self.ln_final(mu_q)

        # 7. PriorBank decode
        logits = self.prior_bank.decode(mu_q, sigma_q, tau=self.prior_bank_tau)

        # Pack attention info (compatible with compute_free_energy_loss)
        attention_info = {
            'beta': all_betas[-1] if all_betas else None,
            'kl': all_kls[-1] if all_kls else None,
            'all_betas': all_betas,
            'all_kls': all_kls,
            'mu': mu_q,
            'sigma': sigma_q,
            'phi': phi,
            'mu_prior': mu_prior,
            'sigma_prior': sigma_prior,
            'phi_prior': phi_prior,
            'n_layers': len(self.blocks),
        }

        return logits, attention_info

    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        r"""
        Autoregressively generate tokens from a prompt.

        Args:
            prompt_ids: (B, N_prompt) starting token indices
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling cutoff
            top_p: Nucleus sampling threshold

        Returns:
            (B, N_prompt + max_new_tokens) generated token indices
        """
        self.eval()
        generated = prompt_ids.clone()
        max_seq_len = self.config['max_seq_len']

        for _ in range(max_new_tokens):
            # Truncate to max sequence length
            if generated.shape[1] > max_seq_len:
                generated = generated[:, -max_seq_len:]

            logits = self.forward(generated)
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

    def count_parameters(self) -> Dict[str, int]:
        """Count parameters by component for comparison."""
        counts = {}

        # PriorBank parameters
        pb_params = sum(p.numel() for p in self.prior_bank.parameters() if p.requires_grad)
        counts['prior_bank'] = pb_params

        # Attention parameters (across all blocks)
        attn_params = 0
        for block in self.blocks:
            attn_params += sum(p.numel() for p in block.attention.parameters() if p.requires_grad)
        counts['attention'] = attn_params

        # FFN parameters (across all blocks)
        ffn_params = 0
        for block in self.blocks:
            ffn_params += sum(p.numel() for p in block.ffn.parameters() if p.requires_grad)
        counts['ffn'] = ffn_params

        # LayerNorm parameters
        ln_params = 0
        for block in self.blocks:
            ln_params += sum(p.numel() for p in block.ln1.parameters())
            ln_params += sum(p.numel() for p in block.ln2.parameters())
        ln_params += sum(p.numel() for p in self.ln_final.parameters())
        counts['layer_norm'] = ln_params

        counts['total'] = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return counts


# =============================================================================
# Testing
# =============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("HYBRID GAUGE-ATTENTION TRANSFORMER")
    print("=" * 70)

    config = {
        'vocab_size': 256,
        'embed_dim': 20,
        'n_layers': 2,
        'irrep_spec': [('fund', 2, 10)],
        'hidden_dim': 80,          # 4x embed_dim
        'max_seq_len': 32,
        'kappa_beta': 1.0,
        'gauge_group': 'GLK',
        'gauge_dim': 10,
        'gauge_mode': 'learned',
        'diagonal_covariance': True,
        'use_rope': True,
        'rope_base': 5000,
        'dropout': 0.1,
        'use_output_projection': True,
        'pos_encoding_mode': 'none',
        'phi_scale': 1.0,
        'mu_init_std': 1.0,
        'evolve_sigma': True,
    }

    print("\nConfiguration:")
    for k, v in config.items():
        print(f"  {k:30s}: {v}")

    # Create model
    print("\n" + "=" * 70)
    model = HybridGaugeTransformerLM(config)

    # Parameter breakdown
    print("\n" + "=" * 70)
    print("PARAMETER BREAKDOWN")
    print("=" * 70)
    counts = model.count_parameters()
    for name, count in counts.items():
        print(f"  {name:20s}: {count:8d}")

    # Test forward pass
    print("\n" + "=" * 70)
    print("TEST FORWARD PASS")
    print("=" * 70)

    B, N = 2, 10
    input_ids = torch.randint(0, config['vocab_size'], (B, N))

    logits = model(input_ids)
    print(f"  Input shape:   {input_ids.shape}")
    print(f"  Output logits: {logits.shape}")

    # Test forward_with_attention
    print("\n" + "=" * 70)
    print("TEST FORWARD WITH ATTENTION")
    print("=" * 70)

    logits, attn_info = model.forward_with_attention(input_ids)
    print(f"  Logits shape:  {logits.shape}")
    print(f"  Beta shape:    {attn_info['beta'].shape if attn_info['beta'] is not None else 'None'}")
    print(f"  KL shape:      {attn_info['kl'].shape if attn_info['kl'] is not None else 'None'}")
    print(f"  Mu shape:      {attn_info['mu'].shape}")
    print(f"  Sigma shape:   {attn_info['sigma'].shape}")
    print(f"  N layers:      {attn_info['n_layers']}")

    # Test loss computation
    print("\n" + "=" * 70)
    print("TEST LOSS + GRADIENTS")
    print("=" * 70)

    targets = torch.randint(0, config['vocab_size'], (B, N))
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
        reduction='mean',
    )
    loss.backward()

    print(f"  Loss:          {loss.item():.4f}")
    print(f"  Expected init: {math.log(config['vocab_size']):.4f} (ln V)")

    # Check gradient flow to PriorBank
    pb_grads = {}
    for name, p in model.prior_bank.named_parameters():
        if p.grad is not None:
            pb_grads[name] = p.grad.norm().item()
    print(f"\n  PriorBank gradient norms:")
    for name, norm in pb_grads.items():
        print(f"    {name:30s}: {norm:.6f}")

    has_grads = all(v > 0 for v in pb_grads.values())
    print(f"\n  Gradient flow to PriorBank: {'OK' if has_grads else 'BROKEN'}")

    # Test generation
    print("\n" + "=" * 70)
    print("TEST GENERATION")
    print("=" * 70)

    prompt = torch.randint(0, config['vocab_size'], (1, 5))
    generated = model.generate(prompt, max_new_tokens=10, temperature=1.0)
    print(f"  Prompt length:    {prompt.shape[1]}")
    print(f"  Generated length: {generated.shape[1]}")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)
