#!/usr/bin/env python3
"""
Token Covariance Ranking
========================

Load a trained GaugeTransformerLM checkpoint and rank all vocabulary tokens
by their prior covariance magnitude (trace of Σ_v or mean diagonal variance).

Small Σ -> high precision -> the model is confident about this token's meaning.
Large Σ -> high uncertainty -> the token's representation is diffuse / context-dependent.

Instructions:
    1. Set CHECKPOINT_PATH below
    2. Run: python scripts/covariance_ranking.py
"""

# =============================================================================
# CONFIGURATION -- EDIT THESE
# =============================================================================

CHECKPOINT_PATH = r"checkpoints_publication/ffn_VFE_dynamic/best_model.pt"

# Override dataset for tokenizer selection (None = auto-detect from checkpoint)
DATASET = None

# How many tokens to show at each extreme
TOP_K = 50

# Covariance summary statistic: 'trace' (sum of variances), 'mean' (avg variance),
# 'max' (largest variance dimension), 'det' (log-determinant)
METRIC = 'trace'

# =============================================================================
# CODE -- No need to edit below
# =============================================================================

import sys
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformer.utils.checkpoint import load_model, get_tokenizer


def get_prior_covariances(model) -> torch.Tensor:
    """Extract per-token prior variances from the model.

    Returns:
        sigma: (vocab_size, K) diagonal variances for each token.
    """
    # PriorBank path
    if hasattr(model, 'use_prior_bank') and model.use_prior_bank and model.prior_bank is not None:
        pb = model.prior_bank
        if pb.gauge_fixed_priors:
            # All tokens share a single base variance (rotated, but diagonal magnitude preserved)
            base_sigma = torch.exp(pb.base_log_prior_sigma).clamp(min=0.01, max=5.0)  # (K,)
            return base_sigma.unsqueeze(0).expand(pb.vocab_size, -1)
        else:
            return torch.exp(pb.log_prior_sigma).clamp(min=0.01, max=5.0)  # (V, K)

    # GaugeTokenEmbedding path
    te = model.token_embed
    if hasattr(te, 'gauge_fixed_priors') and te.gauge_fixed_priors:
        base_sigma = torch.exp(te.base_log_sigma_diag).clamp(min=0.01, max=5.0)  # (K,)
        return base_sigma.unsqueeze(0).expand(te.vocab_size, -1)
    elif hasattr(te, 'log_sigma_diag'):
        log_sigma = te.log_sigma_diag
        if log_sigma.dim() == 1:
            # Shared across vocab -- (K,)
            return torch.exp(log_sigma).clamp(min=0.01, max=5.0).unsqueeze(0).expand(te.vocab_size, -1)
        else:
            return torch.exp(log_sigma).clamp(min=0.01, max=5.0)  # (V, K)
    else:
        raise RuntimeError("Cannot find covariance parameters in model. "
                           "Check that the checkpoint has learnable_sigma=True or uses PriorBank.")


def compute_metric(sigma: torch.Tensor, metric: str) -> torch.Tensor:
    """Reduce (V, K) variances to (V,) scalar per token.

    Args:
        sigma: (V, K) diagonal variances.
        metric: 'trace', 'mean', 'max', or 'det'.

    Returns:
        scores: (V,) scalar covariance summary per token.
    """
    if metric == 'trace':
        return sigma.sum(dim=-1)
    elif metric == 'mean':
        return sigma.mean(dim=-1)
    elif metric == 'max':
        return sigma.max(dim=-1).values
    elif metric == 'det':
        # Log-determinant = sum of log-variances (diagonal case)
        return sigma.clamp(min=1e-8).log().sum(dim=-1)
    else:
        raise ValueError(f"Unknown metric '{metric}'. Use 'trace', 'mean', 'max', or 'det'.")


def decode_token_safe(tokenizer, token_id: int) -> str:
    """Decode a single token ID to a readable string."""
    try:
        text = tokenizer.decode([token_id])
    except Exception:
        text = f"<id={token_id}>"
    # Make whitespace visible
    text = text.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
    return text


def main():
    print("=" * 72)
    print("Token Covariance Ranking")
    print("=" * 72)

    # Load model
    checkpoint_path = Path(CHECKPOINT_PATH)
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Set CHECKPOINT_PATH at the top of this script.")
        sys.exit(1)

    model, config = load_model(str(checkpoint_path))
    model.eval()

    dataset_name = DATASET or config.get('dataset', 'wikitext-103')
    tokenizer = get_tokenizer(config, dataset_name=dataset_name)
    if tokenizer is None:
        print("Could not load tokenizer. Install tiktoken: pip install tiktoken")
        sys.exit(1)

    K = config['embed_dim']
    V = config['vocab_size']
    print(f"\nK={K}, vocab_size={V}, metric={METRIC}")

    # Extract covariances
    with torch.no_grad():
        sigma = get_prior_covariances(model)  # (V, K)

    # Check if all tokens share the same variance (gauge-fixed with shared base)
    if sigma.dim() == 2 and sigma.shape[0] == V:
        unique_rows = sigma.unique(dim=0)
        if unique_rows.shape[0] == 1:
            print("\nAll tokens share the same base covariance (gauge-fixed priors).")
            print(f"  Base variance per dim: mean={sigma[0].mean():.4f}, "
                  f"min={sigma[0].min():.4f}, max={sigma[0].max():.4f}")
            print("Individual token ranking is not meaningful in this mode.")
            return

    scores = compute_metric(sigma, METRIC)  # (V,)

    # Sort
    sorted_indices = scores.argsort()
    smallest_ids = sorted_indices[:TOP_K]
    largest_ids = sorted_indices[-TOP_K:].flip(0)

    # Global statistics
    print(f"\nGlobal covariance statistics ({METRIC}):")
    print(f"  mean  = {scores.mean():.4f}")
    print(f"  std   = {scores.std():.4f}")
    print(f"  min   = {scores.min():.4f}")
    print(f"  max   = {scores.max():.4f}")
    print(f"  median= {scores.median():.4f}")

    # Print smallest (most precise)
    print(f"\n{'=' * 72}")
    print(f"TOP {TOP_K} SMALLEST COVARIANCE (most precise / most certain)")
    print(f"{'=' * 72}")
    print(f"{'Rank':>5}  {'Token ID':>8}  {METRIC:>10}  Token")
    print(f"{'-' * 5}  {'-' * 8}  {'-' * 10}  {'-' * 40}")
    for rank, tid in enumerate(smallest_ids, 1):
        tid_int = tid.item()
        token_str = decode_token_safe(tokenizer, tid_int)
        print(f"{rank:>5}  {tid_int:>8}  {scores[tid_int]:>10.4f}  {repr(token_str)}")

    # Print largest (most uncertain)
    print(f"\n{'=' * 72}")
    print(f"TOP {TOP_K} LARGEST COVARIANCE (most uncertain / most diffuse)")
    print(f"{'=' * 72}")
    print(f"{'Rank':>5}  {'Token ID':>8}  {METRIC:>10}  Token")
    print(f"{'-' * 5}  {'-' * 8}  {'-' * 10}  {'-' * 40}")
    for rank, tid in enumerate(largest_ids, 1):
        tid_int = tid.item()
        token_str = decode_token_safe(tokenizer, tid_int)
        print(f"{rank:>5}  {tid_int:>8}  {scores[tid_int]:>10.4f}  {repr(token_str)}")

    # Per-dimension analysis: which dimensions carry most variance?
    dim_variance = sigma.mean(dim=0)  # (K,) average variance per dimension across vocab
    dim_sorted = dim_variance.argsort(descending=True)
    print(f"\n{'=' * 72}")
    print(f"DIMENSION ANALYSIS (variance averaged across vocab)")
    print(f"{'=' * 72}")
    print(f"{'Dim':>5}  {'Mean Var':>10}  {'Std Var':>10}")
    print(f"{'-' * 5}  {'-' * 10}  {'-' * 10}")
    for d in dim_sorted[:min(20, K)]:
        d_int = d.item()
        dim_std = sigma[:, d_int].std()
        print(f"{d_int:>5}  {dim_variance[d_int]:>10.4f}  {dim_std:>10.4f}")

    # Correlation with token frequency (if we can estimate it)
    print(f"\n{'=' * 72}")
    print(f"VARIANCE vs TOKEN ID (proxy for frequency)")
    print(f"{'=' * 72}")
    # In GPT-2 BPE, lower token IDs tend to be more frequent.
    # Compute correlation between token ID and covariance score.
    token_ids_tensor = torch.arange(V, dtype=torch.float32)
    # Pearson correlation
    x = token_ids_tensor - token_ids_tensor.mean()
    y = scores - scores.mean()
    corr = (x * y).sum() / (x.norm() * y.norm() + 1e-8)
    print(f"Pearson r(token_id, {METRIC}) = {corr:.4f}")
    print("(In GPT-2 BPE, lower ID ~ higher frequency. Positive r means")
    print(" rare tokens have larger variance, negative r means common tokens do.)")

    # Quartile analysis
    q1, q2, q3 = torch.quantile(scores, torch.tensor([0.25, 0.5, 0.75]))
    print(f"\nQuartile boundaries: Q1={q1:.4f}, Q2={q2:.4f}, Q3={q3:.4f}")

    # Sample tokens from each quartile
    for label, lo, hi in [("Q1 (most precise)", scores.min(), q1),
                           ("Q2", q1, q2),
                           ("Q3", q2, q3),
                           ("Q4 (most diffuse)", q3, scores.max() + 1e-6)]:
        mask = (scores >= lo) & (scores < hi)
        ids_in_quartile = mask.nonzero(as_tuple=True)[0]
        n = ids_in_quartile.shape[0]
        sample_size = min(10, n)
        if n > 0:
            sample = ids_in_quartile[torch.randperm(n)[:sample_size]]
            tokens = [decode_token_safe(tokenizer, t.item()) for t in sample]
            print(f"\n  {label} ({n} tokens): {', '.join(repr(t) for t in tokens)}")


if __name__ == '__main__':
    main()
