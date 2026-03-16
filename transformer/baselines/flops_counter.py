"""
FLOPs Counter for Gauge and Standard Transformers
===================================================

Reports FLOPs per training step and total training compute.
Addresses peer review concern (M2e): the gauge model requires matrix exponentials
and per-pair KL computations that are substantially more expensive than standard
dot-product attention.

Two estimation modes:
    1. Analytical: Formula-based estimation from model config (fast, deterministic)
    2. Profile-based: torch.profiler measurement (accurate but requires GPU)

FLOPs are reported as multiply-accumulate operations (MACs) * 2 = FLOPs.

Author: Peer review response (M2e)
Date: March 2026
"""

import math
from typing import Dict, Optional


def count_standard_transformer_flops(
    vocab_size: int,
    embed_dim: int,
    n_layers: int,
    n_heads: int,
    hidden_dim: int,
    seq_len: int,
    batch_size: int = 1,
    disable_ffn: bool = False,
    tie_embeddings: bool = False,
) -> Dict[str, int]:
    """
    Analytically count FLOPs for a standard transformer forward pass.

    Follows the Kaplan et al. (2020) / Hoffmann et al. (2022) convention:
        - Each matrix multiply of (M, K) @ (K, N) costs 2*M*K*N FLOPs
        - Forward pass only; backward ~= 2x forward

    Args:
        vocab_size: V
        embed_dim: d_model
        n_layers: L
        n_heads: H
        hidden_dim: d_ff
        seq_len: N
        batch_size: B
        disable_ffn: If True, FFN is disabled (attention-only)
        tie_embeddings: If True, lm_head shares weights with token_embed

    Returns:
        Dict with per-component and total FLOPs
    """
    B = batch_size
    N = seq_len
    d = embed_dim
    d_k = embed_dim // n_heads
    H = n_heads
    L = n_layers
    d_ff = hidden_dim
    V = vocab_size

    flops = {}

    # --- Token embedding: lookup (negligible FLOPs) ---
    flops['token_embed'] = 0

    # --- Positional embedding: lookup or RoPE ---
    # Learned: lookup (negligible). RoPE: 6*N*d per layer (sin/cos/mul), small.
    flops['pos_encoding'] = 0  # Negligible vs matmuls

    # --- Per-layer attention ---
    # Q, K, V projections: 3 * (B*N*d * d) = 3 * 2*B*N*d^2
    qkv_flops = 3 * 2 * B * N * d * d

    # Q @ K^T: B*H * (N*d_k) @ (d_k*N) = 2*B*H*N*N*d_k = 2*B*N*N*d
    attn_score_flops = 2 * B * N * N * d

    # softmax: ~5*B*H*N*N (exp, sum, div) - approximate
    softmax_flops = 5 * B * H * N * N

    # attn_weights @ V: 2*B*N*N*d
    attn_value_flops = 2 * B * N * N * d

    # Output projection W_O: 2*B*N*d*d
    output_proj_flops = 2 * B * N * d * d

    # LayerNorm (attention): ~5*B*N*d (mean, var, normalize)
    ln_attn_flops = 5 * B * N * d

    per_layer_attn = qkv_flops + attn_score_flops + softmax_flops + attn_value_flops + output_proj_flops + ln_attn_flops
    flops['attention_per_layer'] = per_layer_attn

    # --- Per-layer FFN ---
    if not disable_ffn:
        # fc1: 2*B*N*d*d_ff + GELU: ~8*B*N*d_ff
        # fc2: 2*B*N*d_ff*d
        # LayerNorm: 5*B*N*d
        ffn_flops = 2 * B * N * d * d_ff + 8 * B * N * d_ff + 2 * B * N * d_ff * d + 5 * B * N * d
        flops['ffn_per_layer'] = ffn_flops
    else:
        ffn_flops = 0
        flops['ffn_per_layer'] = 0

    # --- Total transformer layers ---
    flops['all_layers'] = L * (per_layer_attn + ffn_flops)

    # --- Final LayerNorm ---
    flops['ln_final'] = 5 * B * N * d

    # --- LM head: 2*B*N*d*V ---
    flops['lm_head'] = 2 * B * N * d * V

    # --- Total forward ---
    flops['forward_total'] = flops['all_layers'] + flops['ln_final'] + flops['lm_head']

    # --- Backward ~= 2x forward (gradient computation + weight updates) ---
    flops['backward_total'] = 2 * flops['forward_total']

    # --- Total per training step (forward + backward) ---
    flops['step_total'] = flops['forward_total'] + flops['backward_total']

    return flops


def count_gauge_transformer_flops(
    vocab_size: int,
    embed_dim: int,
    n_layers: int,
    n_heads: int,
    head_dim: int,
    seq_len: int,
    batch_size: int = 1,
    phi_dim: int = 100,
    ffn_n_iterations: int = 1,
    use_rope: bool = True,
    diagonal_covariance: bool = True,
) -> Dict[str, int]:
    """
    Analytically count FLOPs for a gauge transformer forward pass.

    Key additional costs vs. standard transformer:
        1. Matrix exponentials: exp(phi_i . G) for each token pair
        2. Per-pair KL divergence: KL(q_i || Omega_ij[q_j]) for all (i,j)
        3. Gaussian transport: push-forward of beliefs through Omega
        4. VFE E-step iterations (may run multiple times)

    Args:
        vocab_size: V
        embed_dim: K (total embedding dimension)
        n_layers: L
        n_heads: H
        head_dim: d_head per attention head
        seq_len: N
        batch_size: B
        phi_dim: number of Lie algebra parameters per token
        ffn_n_iterations: T_inner E-step iterations
        use_rope: whether RoPE is applied to mu
        diagonal_covariance: if True, Sigma is diagonal (much cheaper KL)

    Returns:
        Dict with per-component and total FLOPs
    """
    B = batch_size
    N = seq_len
    K = embed_dim
    H = n_heads
    d = head_dim
    L = n_layers
    V = vocab_size
    T = ffn_n_iterations

    flops = {}

    # --- Token embedding: lookup of (mu, sigma, phi) ---
    flops['token_embed'] = 0  # Lookup

    # --- RoPE on mu ---
    if use_rope:
        # sin/cos computation + 4 multiplies per dimension pair: ~6*B*N*K
        flops['rope'] = 6 * B * N * K
    else:
        flops['rope'] = 0

    # --- Per-layer: Matrix exponentials for transport operators ---
    # For each token: exp(sum_a phi_a * G_a) where G_a is (d, d)
    # Step 1: Linear combination sum_a phi_a * G_a: phi_dim * d^2 multiplies per token
    # Step 2: Matrix exponential (Pade approximation or Taylor, ~30 d^3 FLOPs typical)
    # We need exp(phi_i) and exp(-phi_j) for each (i,j) pair.
    # With caching: compute 2*N matrix exponentials per batch element per head.
    #   lin_comb: 2*N * phi_dim_per_head * d^2
    #   mat_exp:  2*N * 30 * d^3  (Pade [6/6] ≈ 30 matrix multiplications of size d)
    phi_dim_per_head = phi_dim // H if H > 0 else phi_dim
    lin_comb_flops = 2 * B * N * phi_dim_per_head * d * d
    # Matrix exponential via scaling-and-squaring with Pade[6/6]:
    # ~13 matrix multiplications of (d,d), each 2*d^3 FLOPs, plus scaling
    mat_exp_flops = 2 * B * N * H * 13 * 2 * d * d * d
    flops['transport_lin_comb'] = lin_comb_flops
    flops['transport_mat_exp'] = mat_exp_flops

    # --- Per-layer: Transport operators Omega_ij = exp(phi_i) @ exp(-phi_j) ---
    # For all N^2 pairs: N^2 matrix multiplies of (d,d)
    # But with caching, it's computed as products of cached exp(phi_i) and exp(-phi_j)
    transport_product_flops = B * H * N * N * 2 * d * d * d
    flops['transport_product'] = transport_product_flops

    # --- Per-layer: KL divergence computation ---
    if diagonal_covariance:
        # KL(N(mu_i, diag(sigma_i)) || Omega_ij * N(mu_j, diag(sigma_j)))
        # After transport: mu_j' = Omega_ij @ mu_j (matrix-vector: 2*d^2)
        #                  sigma_j' = Omega_ij @ diag(sigma_j) @ Omega_ij^T (2*d^3 + d^2)
        # KL computation: trace + log det + quadratic form ≈ 10*d ops (diagonal)
        # Total per pair: ~2*d^3 + 4*d^2 + 10*d
        # For N^2 pairs:
        kl_transport_flops = B * H * N * N * (2 * d * d * d + 4 * d * d + 10 * d)
    else:
        # Full covariance: much more expensive
        # Sigma_j' = Omega @ Sigma_j @ Omega^T: 2 * 2*d^3
        # KL: trace(Sigma_i^-1 Sigma_j'): d^3 for inverse + d^3 for product
        # quadratic form: d^2, log det: d (with Cholesky: d^3/3)
        kl_transport_flops = B * H * N * N * (6 * d * d * d + 2 * d * d)
    flops['kl_divergence'] = kl_transport_flops

    # --- Per-layer: Softmax over KL scores ---
    softmax_flops = 5 * B * H * N * N
    flops['softmax'] = softmax_flops

    # --- Per-layer: Message aggregation ---
    # Weighted sum of transported beliefs: for each i, sum_j beta_ij * Omega_ij[q_j]
    # mu aggregation: N queries, each summing N transported means (d-dim): 2*B*H*N*N*d
    # sigma aggregation (diagonal): same cost
    msg_flops = 2 * B * H * N * N * d * 2  # mu + sigma
    flops['message_aggregation'] = msg_flops

    # --- Per-layer: VFE E-step iterations ---
    # Each iteration recomputes KL, transport, softmax, messages
    # Cost per iteration ≈ kl_transport + softmax + msg
    vfe_per_iter = kl_transport_flops + softmax_flops + msg_flops
    flops['vfe_iterations'] = T * vfe_per_iter

    # --- Per-layer: LayerNorm ---
    ln_flops = 5 * B * N * K
    flops['layernorm_per_layer'] = ln_flops

    # --- Per-layer total ---
    per_layer = (
        lin_comb_flops + mat_exp_flops + transport_product_flops +
        kl_transport_flops + softmax_flops + msg_flops +
        flops['vfe_iterations'] + ln_flops
    )
    flops['per_layer_total'] = per_layer

    # --- All layers ---
    flops['all_layers'] = L * per_layer

    # --- Output projection: 2*B*N*K*V ---
    flops['lm_head'] = 2 * B * N * K * V

    # --- Total forward ---
    flops['forward_total'] = flops['all_layers'] + flops['lm_head'] + flops['rope']

    # --- Backward ~= 2x forward ---
    flops['backward_total'] = 2 * flops['forward_total']

    # --- Total per training step ---
    flops['step_total'] = flops['forward_total'] + flops['backward_total']

    return flops


def format_flops(flops: int) -> str:
    """Format FLOPs count in human-readable form."""
    if flops >= 1e18:
        return f"{flops/1e18:.2f} EFLOPs"
    elif flops >= 1e15:
        return f"{flops/1e15:.2f} PFLOPs"
    elif flops >= 1e12:
        return f"{flops/1e12:.2f} TFLOPs"
    elif flops >= 1e9:
        return f"{flops/1e9:.2f} GFLOPs"
    elif flops >= 1e6:
        return f"{flops/1e6:.2f} MFLOPs"
    else:
        return f"{flops:,} FLOPs"


def compare_flops(
    gauge_config: Dict,
    standard_config: Dict,
    seq_len: int = 128,
    batch_size: int = 64,
    max_steps: int = 15000,
) -> Dict[str, any]:
    """
    Compare FLOPs between gauge and standard transformer configurations.

    Args:
        gauge_config: Config dict for gauge model
        standard_config: Config dict for standard model
        seq_len: Sequence length
        batch_size: Batch size
        max_steps: Total training steps

    Returns:
        Comparison dict with per-step and total FLOPs for both models
    """
    # Extract gauge model params
    gauge_embed = gauge_config.get('embed_dim', 90)
    gauge_irrep = gauge_config.get('irrep_spec', [('fund', 1, gauge_embed)])
    n_heads_gauge = gauge_irrep[0][1] if gauge_irrep else 1
    head_dim_gauge = gauge_irrep[0][2] if gauge_irrep else gauge_embed
    phi_dim = gauge_embed * gauge_embed  # GL(K) default

    gauge_flops = count_gauge_transformer_flops(
        vocab_size=gauge_config.get('vocab_size', 50257),
        embed_dim=gauge_embed,
        n_layers=gauge_config.get('n_layers', 1),
        n_heads=n_heads_gauge,
        head_dim=head_dim_gauge,
        seq_len=seq_len,
        batch_size=batch_size,
        phi_dim=phi_dim,
        ffn_n_iterations=gauge_config.get('ffn_n_iterations', 1),
        use_rope=gauge_config.get('use_rope', True),
        diagonal_covariance=gauge_config.get('diagonal_covariance', True),
    )

    # Extract standard model params
    std_embed = standard_config.get('embed_dim', 90)
    std_heads = standard_config.get('n_heads', 1)

    standard_flops = count_standard_transformer_flops(
        vocab_size=standard_config.get('vocab_size', 50257),
        embed_dim=std_embed,
        n_layers=standard_config.get('n_layers', 1),
        n_heads=std_heads,
        hidden_dim=standard_config.get('hidden_dim', std_embed * 4),
        seq_len=seq_len,
        batch_size=batch_size,
        disable_ffn=standard_config.get('disable_ffn', False),
        tie_embeddings=standard_config.get('tie_embeddings', False),
    )

    gauge_total = gauge_flops['step_total'] * max_steps
    std_total = standard_flops['step_total'] * max_steps
    ratio = gauge_flops['step_total'] / max(standard_flops['step_total'], 1)

    result = {
        'gauge': {
            'flops_per_step': gauge_flops['step_total'],
            'flops_per_step_str': format_flops(gauge_flops['step_total']),
            'total_training_flops': gauge_total,
            'total_training_flops_str': format_flops(gauge_total),
            'breakdown': {k: format_flops(v) for k, v in gauge_flops.items()},
        },
        'standard': {
            'flops_per_step': standard_flops['step_total'],
            'flops_per_step_str': format_flops(standard_flops['step_total']),
            'total_training_flops': std_total,
            'total_training_flops_str': format_flops(std_total),
            'breakdown': {k: format_flops(v) for k, v in standard_flops.items()},
        },
        'gauge_to_standard_ratio': ratio,
        'max_steps': max_steps,
        'seq_len': seq_len,
        'batch_size': batch_size,
    }

    return result


def print_flops_comparison(result: Dict) -> None:
    """Pretty-print FLOPs comparison results."""
    print("\n" + "="*70)
    print("FLOPs COMPARISON: GAUGE vs STANDARD TRANSFORMER")
    print("="*70)
    print(f"  Sequence length: {result['seq_len']}")
    print(f"  Batch size:      {result['batch_size']}")
    print(f"  Training steps:  {result['max_steps']:,}")

    print(f"\n  {'Metric':<30s} {'Standard':>15s} {'Gauge':>15s} {'Ratio':>10s}")
    print(f"  {'-'*70}")

    std = result['standard']
    gauge = result['gauge']
    ratio = result['gauge_to_standard_ratio']

    print(f"  {'FLOPs/step':<30s} {std['flops_per_step_str']:>15s} {gauge['flops_per_step_str']:>15s} {ratio:>9.1f}x")
    print(f"  {'Total training FLOPs':<30s} {std['total_training_flops_str']:>15s} {gauge['total_training_flops_str']:>15s} {ratio:>9.1f}x")

    print(f"\n  Gauge model is {ratio:.1f}x more expensive per training step")

    # Breakdown
    for model_name, model_data in [('Standard', std), ('Gauge', gauge)]:
        print(f"\n  {model_name} Transformer Breakdown:")
        for component, flop_str in model_data['breakdown'].items():
            if component in ('step_total', 'forward_total', 'backward_total'):
                continue
            print(f"    {component:<30s}: {flop_str}")

    print("="*70)


# =============================================================================
# Convenience: Profile-based measurement (more accurate, requires model + GPU)
# =============================================================================

def profile_flops(
    model,
    input_ids,
    labels=None,
    device='cpu',
) -> Dict[str, float]:
    """
    Profile actual FLOPs using PyTorch's FlopCounterMode.

    Requires PyTorch >= 2.1 with torch.utils.flop_counter.

    Args:
        model: The model to profile
        input_ids: (B, N) input tensor
        labels: Optional (B, N) label tensor
        device: Device string

    Returns:
        Dict with measured FLOPs
    """
    import torch

    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:
        print("Warning: torch.utils.flop_counter not available (requires PyTorch >= 2.1)")
        print("Falling back to analytical estimation.")
        return None

    model = model.to(device)
    input_ids = input_ids.to(device)
    if labels is not None:
        labels = labels.to(device)

    model.eval()
    with torch.no_grad():
        flop_counter = FlopCounterMode(display=False)
        with flop_counter:
            if hasattr(model, 'forward_with_attention'):
                # Gauge model
                model.forward_with_attention(input_ids)
            else:
                model(input_ids, labels=labels)

        total_flops = flop_counter.get_total_flops()

    return {
        'forward_flops_measured': total_flops,
        'forward_flops_str': format_flops(total_flops),
        # Backward ~2x forward
        'step_flops_estimated': 3 * total_flops,
        'step_flops_str': format_flops(3 * total_flops),
    }


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    # Example: compare the configs from train_publication.py
    gauge_config = {
        'vocab_size': 50257,
        'embed_dim': 10,
        'n_layers': 1,
        'irrep_spec': [('fund', 1, 10)],
        'ffn_n_iterations': 1,
        'use_rope': True,
        'diagonal_covariance': True,
    }

    standard_config = {
        'vocab_size': 50257,
        'embed_dim': 10,
        'n_layers': 1,
        'n_heads': 1,
        'hidden_dim': 24527,
        'disable_ffn': False,
        'tie_embeddings': False,
    }

    result = compare_flops(
        gauge_config=gauge_config,
        standard_config=standard_config,
        seq_len=128,
        batch_size=64,
        max_steps=15000,
    )
    print_flops_comparison(result)
