"""
Standard Transformer Baseline
==============================

Vanilla transformer with dot-product attention for fair comparison against the
gauge-theoretic transformer. The gauge model replaces Q/K projections with
KL-divergence attention over gauge-transported Gaussian beliefs and uses a VFE
E-step FFN; this baseline uses conventional softmax(QK^T/sqrt(d)) attention and
a standard two-layer GELU FFN, providing an apples-to-apples compute comparison.

Supports multiple configurations for ablation studies:
    - Standard transformer (attention + FFN)
    - Attention-only (FFN disabled) to isolate attention mechanism contribution
    - RoPE positional encoding (to match gauge model's position scheme)
    - Parameter-equalized via wider layers
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple


# =============================================================================
# RoPE Implementation (matches gauge model's attention.py implementation)
# =============================================================================

def _build_rope_freqs(dim: int, base: float = 10000.0,
                      device: torch.device = None,
                      dtype: torch.dtype = None) -> torch.Tensor:
    """Compute RoPE frequency bands for dim-dimensional vectors.

    Returns:
        freqs: (dim//2,) inverse frequency bands
    """
    half_dim = dim // 2
    freqs = 1.0 / (base ** (torch.arange(0, half_dim, device=device, dtype=dtype) / half_dim))
    return freqs


def apply_rope_to_qk(Q: torch.Tensor, K: torch.Tensor,
                      rope_base: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply Rotary Position Embeddings to Q and K tensors.

    Standard RoPE as in Su et al. (2021) / Llama / etc.
    Rotates consecutive pairs of dimensions by position-dependent angles.

    Args:
        Q: (B, H, N, D) query tensor
        K: (B, H, N, D) key tensor
        rope_base: RoPE frequency base

    Returns:
        Q_rotated, K_rotated: position-encoded Q and K
    """
    B, H, N, D = Q.shape
    half_D = D // 2

    freqs = _build_rope_freqs(D, rope_base, device=Q.device, dtype=Q.dtype)  # (D//2,)
    positions = torch.arange(N, device=Q.device, dtype=Q.dtype)  # (N,)
    angles = torch.outer(positions, freqs)  # (N, D//2)

    cos_angles = torch.cos(angles)  # (N, D//2)
    sin_angles = torch.sin(angles)  # (N, D//2)

    # Reshape for broadcasting: (1, 1, N, D//2)
    cos_angles = cos_angles.unsqueeze(0).unsqueeze(0)
    sin_angles = sin_angles.unsqueeze(0).unsqueeze(0)

    def rotate(x):
        x_even = x[..., :2*half_D:2]   # dims 0,2,4,...
        x_odd = x[..., 1:2*half_D:2]   # dims 1,3,5,...
        x_rotated = x.clone()
        x_rotated[..., :2*half_D:2] = x_even * cos_angles - x_odd * sin_angles
        x_rotated[..., 1:2*half_D:2] = x_even * sin_angles + x_odd * cos_angles
        return x_rotated

    return rotate(Q), rotate(K)


class StandardMultiHeadAttention(nn.Module):
    """
    Standard dot-product multi-head attention with optional RoPE.

    Unlike the gauge transformer's KL-divergence attention (which needs no W_Q or
    W_K), this module uses explicit Q/K/V projections with scaled dot-product
    scoring: beta_ij = softmax(Q_i K_j^T / sqrt(d_k)).

    Args:
        embed_dim: Total model dimension (split across heads).
        n_heads: Number of attention heads.
        dropout: Dropout probability on attention weights.
        use_rope: Apply Rotary Position Embeddings to Q and K.
        rope_base: RoPE frequency base.
    """

    def __init__(self, embed_dim: int, n_heads: int = 1, dropout: float = 0.1,
                 use_rope: bool = False, rope_base: float = 10000.0):
        super().__init__()
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.use_rope = use_rope
        self.rope_base = rope_base

        # Warn about pathologically small head dimensions
        if self.head_dim < 16:
            import warnings
            warnings.warn(
                f"head_dim={self.head_dim} is very small (embed_dim={embed_dim}, n_heads={n_heads}). "
                f"This severely limits model capacity. Recommended head_dim >= 32. "
                f"Consider using fewer heads (e.g., n_heads={embed_dim // 32} for head_dim=32).",
                UserWarning
            )

        # Standard attention projections (THIS IS WHAT GAUGE MODEL DOESN'T HAVE!)
        self.W_Q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_K = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_V = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_O = nn.Linear(embed_dim, embed_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> torch.Tensor:
        """Compute multi-head self-attention.

        Args:
            x: Input tensor of shape (B, N, embed_dim).
            mask: Optional causal mask of shape broadcastable to (B, H, N, N).
            return_attention: If True, return (output, attn_weights).

        Returns:
            Output tensor (B, N, embed_dim), or a tuple of (output, attn_weights)
            when *return_attention* is True.
        """
        B, N, embed_dim = x.shape

        # Project to Q, K, V
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        # Reshape for multi-head
        Q = Q.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, N, D)
        K = K.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K if enabled
        if self.use_rope:
            Q, K = apply_rope_to_qk(Q, K, rope_base=self.rope_base)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, H, N, N)

        # Apply causal mask
        if mask is not None:
            while mask.dim() < 4:
                mask = mask.unsqueeze(0)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)  # (B, H, N, N)
        attn_weights_dropped = self.dropout(attn_weights)

        # Apply attention to values
        out = torch.matmul(attn_weights_dropped, V)  # (B, H, N, D)

        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(B, N, embed_dim)

        # Output projection
        out = self.W_O(out)

        if return_attention:
            return out, attn_weights
        return out


class StandardFFN(nn.Module):
    """
    Standard feed-forward network.

    FFN(x) = W2 @ GELU(W1 @ x + b1) + b2
    """

    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class StandardTransformerBlock(nn.Module):
    """
    Pre-norm transformer block: LN -> MHA -> residual -> LN -> FFN -> residual.

    In the gauge transformer the FFN is replaced by a VFE E-step update; here a
    standard two-layer GELU FFN is used. Set *disable_ffn=True* to skip the FFN
    sublayer entirely (attention-only ablation).

    Args:
        embed_dim: Model dimension.
        n_heads: Number of attention heads.
        hidden_dim: FFN intermediate dimension.
        dropout: Dropout probability.
        disable_ffn: If True, omit the FFN sublayer.
        use_rope: Apply RoPE inside the attention sublayer.
        rope_base: RoPE frequency base.
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        hidden_dim: int,
        dropout: float = 0.1,
        disable_ffn: bool = False,
        use_rope: bool = False,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        self.disable_ffn = disable_ffn

        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = StandardMultiHeadAttention(
            embed_dim, n_heads, dropout,
            use_rope=use_rope, rope_base=rope_base,
        )

        if not disable_ffn:
            self.ln2 = nn.LayerNorm(embed_dim)
            self.ffn = StandardFFN(embed_dim, hidden_dim, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> torch.Tensor:
        # Attention block (pre-norm)
        residual = x
        x = self.ln1(x)
        if return_attention:
            x, attn_weights = self.attn(x, mask, return_attention=True)
        else:
            x = self.attn(x, mask)
            attn_weights = None
        x = self.dropout(x)
        x = residual + x

        # FFN block (pre-norm) - skip if attention-only mode
        if not self.disable_ffn:
            residual = x
            x = self.ln2(x)
            x = self.ffn(x)
            x = self.dropout(x)
            x = residual + x

        if return_attention:
            return x, attn_weights
        return x


class StandardTransformerLM(nn.Module):
    """
    Standard transformer language model for baseline comparison.

    Architecture:
        Token Embedding -> Position Encoding -> N x TransformerBlock -> LM Head

    This is the conventional counterpart to the gauge transformer LM, which
    replaces Q/K projections with KL-divergence attention over gauge-transported
    beliefs (SO(3)/SO(N)/GL(K) groups) and swaps the FFN for VFE E-step updates.
    None of those mechanisms are present here.

    Supports:
        - Learned positional embeddings (default)
        - RoPE positional encoding (use_rope=True)
        - Attention-only mode (disable_ffn=True) for ablation
        - No positional encoding (no_pos_encoding=True)

    Args:
        config: Dict with keys ``vocab_size``, ``embed_dim``, ``n_layers``,
            ``n_heads``, ``hidden_dim``, ``max_seq_len``, and optional
            ``dropout``, ``tie_embeddings``, ``disable_ffn``, ``use_rope``,
            ``rope_base``, ``no_pos_encoding``.
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        vocab_size = config['vocab_size']
        embed_dim = config['embed_dim']
        n_layers = config['n_layers']
        n_heads = config.get('n_heads', 1)
        hidden_dim = config['hidden_dim']
        max_seq_len = config['max_seq_len']
        dropout = config.get('dropout', 0.1)
        tie_embeddings = config.get('tie_embeddings', True)
        disable_ffn = config.get('disable_ffn', False)
        use_rope = config.get('use_rope', False)
        rope_base = config.get('rope_base', 10000.0)
        no_pos_encoding = config.get('no_pos_encoding', False)

        self.use_rope = use_rope
        self.no_pos_encoding = no_pos_encoding

        # Token embeddings
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=0.02)

        # Positional embeddings (learned) - only if not using RoPE and not disabled
        if not use_rope and not no_pos_encoding:
            self.pos_embed = nn.Embedding(max_seq_len, embed_dim)
            nn.init.normal_(self.pos_embed.weight, mean=0.0, std=0.02)
        else:
            self.pos_embed = None

        # Transformer blocks
        self.blocks = nn.ModuleList([
            StandardTransformerBlock(
                embed_dim, n_heads, hidden_dim, dropout,
                disable_ffn=disable_ffn,
                use_rope=use_rope,
                rope_base=rope_base,
            )
            for _ in range(n_layers)
        ])

        # Output layer
        self.ln_final = nn.LayerNorm(embed_dim)

        if tie_embeddings:
            self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
            self.lm_head.weight = self.token_embed.weight
        else:
            self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        self.dropout = nn.Dropout(dropout)

        # Create causal mask
        self.register_buffer(
            'causal_mask',
            torch.tril(torch.ones(max_seq_len, max_seq_len)).unsqueeze(0)
        )

        # Count parameters
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        mode_str = "attention-only" if disable_ffn else "standard"
        pos_str = "RoPE" if use_rope else ("none" if no_pos_encoding else "learned")
        print(f"StandardTransformerLM initialized ({mode_str}, pos={pos_str}): {n_params/1e6:.2f}M parameters")

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        pad_token_id: int = -100,
    ) -> Dict[str, torch.Tensor]:
        """Run a forward pass and optionally compute cross-entropy loss.

        Args:
            input_ids: Token indices of shape (B, N).
            labels: Target token indices (B, N). If provided, cross-entropy
                loss is included in the output dict under ``'loss'``.
            pad_token_id: Index to ignore in the loss computation.

        Returns:
            Dict with ``'logits'`` (B, N, V) and optionally ``'loss'``.
        """
        B, N = input_ids.shape
        device = input_ids.device

        # Embed tokens
        x = self.token_embed(input_ids)  # (B, N, K)

        # Add positional embeddings (if using learned pos encoding)
        if self.pos_embed is not None:
            pos_ids = torch.arange(N, device=device).unsqueeze(0)
            x = x + self.pos_embed(pos_ids)

        x = self.dropout(x)

        # Apply transformer blocks
        mask = self.causal_mask[:, :N, :N]
        for block in self.blocks:
            x = block(x, mask)

        # Final layer norm
        x = self.ln_final(x)

        # LM head
        logits = self.lm_head(x)

        output = {'logits': logits}

        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                reduction='mean',
                ignore_index=pad_token_id,
            )
            output['loss'] = loss

        return output

    def forward_with_attention(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass that also returns per-layer attention weights.

        Args:
            input_ids: Token indices of shape (B, N).
            targets: Unused; kept for API compatibility with the gauge model.

        Returns:
            Tuple of (logits, attn_info) where *attn_info* contains
            ``'beta'`` (last-layer weights) and ``'all_attention'`` (list of
            per-layer weight tensors, each (B, H, N, N)).
        """
        B, N = input_ids.shape
        device = input_ids.device

        x = self.token_embed(input_ids)

        if self.pos_embed is not None:
            pos_ids = torch.arange(N, device=device).unsqueeze(0)
            x = x + self.pos_embed(pos_ids)

        x = self.dropout(x)

        mask = self.causal_mask[:, :N, :N]
        all_attn_weights = []
        for block in self.blocks:
            x, attn_weights = block(x, mask, return_attention=True)
            all_attn_weights.append(attn_weights)

        x = self.ln_final(x)
        logits = self.lm_head(x)

        attn_info = {
            'beta': all_attn_weights[-1],
            'all_attention': all_attn_weights,
        }

        return logits, attn_info

    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """Autoregressively generate tokens from a prompt.

        Args:
            prompt_ids: Starting token indices of shape (B, N_prompt).
            max_new_tokens: Number of tokens to generate.
            temperature: Sampling temperature (1.0 = unmodified logits).
            top_k: If set, restrict sampling to the top-k highest-probability tokens.
            top_p: If set, apply nucleus (top-p) sampling.

        Returns:
            Tensor of shape (B, N_prompt + max_new_tokens) with generated ids.
        """
        self.eval()
        generated = prompt_ids.clone()

        for _ in range(max_new_tokens):
            if generated.shape[1] > self.config['max_seq_len']:
                generated = generated[:, -self.config['max_seq_len']:]

            output = self.forward(generated)
            logits = output['logits']
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
        """Count parameters by component."""
        counts = {}
        counts['token_embed'] = self.token_embed.weight.numel()
        if self.pos_embed is not None:
            counts['pos_embed'] = self.pos_embed.weight.numel()

        block_params = 0
        for block in self.blocks:
            block_params += sum(p.numel() for p in block.parameters())
        counts['transformer_blocks'] = block_params

        counts['ln_final'] = sum(p.numel() for p in self.ln_final.parameters())

        if not self.config.get('tie_embeddings', True):
            counts['lm_head'] = self.lm_head.weight.numel()

        counts['total'] = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return counts


# =============================================================================
# Testing
# =============================================================================

if __name__ == '__main__':
    print("="*70)
    print("STANDARD TRANSFORMER BASELINE")
    print("="*70)

    # Match gauge model config
    config = {
        'vocab_size': 256,
        'embed_dim': 11,
        'n_layers': 2,
        'n_heads': 1,  # Single head for K=11
        'hidden_dim': 44,
        'max_seq_len': 32,
        'dropout': 0.1,
        'tie_embeddings': True,
    }

    print("\nConfiguration:")
    for k, v in config.items():
        print(f"  {k:20s}: {v}")

    # Create model
    print("\n" + "="*70)
    model = StandardTransformerLM(config)

    # Count parameters
    print("\n" + "="*70)
    print("PARAMETER BREAKDOWN")
    print("="*70)

    counts = model.count_parameters()
    for name, count in counts.items():
        print(f"  {name:20s}: {count:6d}")

    # Test forward pass
    print("\n" + "="*70)
    print("TEST FORWARD PASS")
    print("="*70)

    B, N = 2, 10
    input_ids = torch.randint(0, config['vocab_size'], (B, N))

    output = model(input_ids, labels=input_ids)

    print(f"  Input shape:  {input_ids.shape}")
    print(f"  Output logits: {output['logits'].shape}")
    print(f"  Loss:         {output['loss'].item():.4f}")

    # Test RoPE variant
    print("\n" + "="*70)
    print("TEST RoPE VARIANT")
    print("="*70)

    rope_config = config.copy()
    rope_config['use_rope'] = True
    rope_config['embed_dim'] = 12  # Must be even for RoPE
    rope_model = StandardTransformerLM(rope_config)
    input_ids_rope = torch.randint(0, config['vocab_size'], (B, N))
    output_rope = rope_model(input_ids_rope, labels=input_ids_rope)
    print(f"  Loss (RoPE):  {output_rope['loss'].item():.4f}")

    # Test attention-only variant
    print("\n" + "="*70)
    print("TEST ATTENTION-ONLY VARIANT")
    print("="*70)

    attn_config = config.copy()
    attn_config['disable_ffn'] = True
    attn_model = StandardTransformerLM(attn_config)
    output_attn = attn_model(input_ids, labels=input_ids)
    print(f"  Loss (attn-only): {output_attn['loss'].item():.4f}")

    print("\n✓ Standard transformer baseline ready!")
    print("="*70)
