#!/usr/bin/env python3
"""
Click-to-Run: Interactive Belief Space Visualization
=====================================================

Just run it:
    python scripts/run_interactive_viz.py

No arguments needed. Automatically:
1. Finds the latest checkpoint in the repo, OR
2. Instantiates a fresh GaugeTransformerLM with random weights

Then produces UMAP + Plotly + SHAP visualizations in outputs/interactive_viz/.

Authors: chris and christine
Date: March 2026
"""

import sys
import warnings
from pathlib import Path

# Suppress noisy GPU warnings
warnings.filterwarnings("ignore", message="Failed to find cuobjdump")
warnings.filterwarnings("ignore", message="Failed to find nvdisasm")
warnings.filterwarnings("ignore", message="CUDA path could not be detected")

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np


# ============================================================================
# Auto-discovery
# ============================================================================

def find_latest_checkpoint() -> Path | None:
    """Search the repo for the most recent .pt checkpoint file."""
    candidates = sorted(
        PROJECT_ROOT.rglob("best_model.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]

    # Fall back to any .pt file that looks like a checkpoint
    candidates = sorted(
        (p for p in PROJECT_ROOT.rglob("*.pt")
         if "checkpoint" in p.name.lower() or "model" in p.name.lower()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def build_fresh_model():
    """Instantiate a fresh GaugeTransformerLM with random weights.

    Uses a realistic config (GPT-2 vocab, GL(K) gauge group) so the
    visualizations show meaningful structure even without training.
    """
    from transformer.core.model import GaugeTransformerLM

    config = {
        'vocab_size': 50257,
        'embed_dim': 25,
        'n_layers': 2,
        'irrep_spec': [('l0', 5, 1), ('l1', 3, 3), ('l2', 1, 5)],
        'hidden_dim': 64,
        'max_seq_len': 128,
        'kappa_beta': 1.0,
        'dropout': 0.0,
        'pos_encoding_mode': 'learned',
        'evolve_sigma': True,
        'evolve_phi': True,
        'tie_embeddings': True,
        'diagonal_covariance': True,
        'ffn_mode': 'VFE_dynamic',
    }

    model = GaugeTransformerLM(config)
    model.eval()

    print(f"  Config: K={config['embed_dim']}, vocab={config['vocab_size']}, "
          f"layers={config['n_layers']}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    return model, config


def get_model_and_config():
    """Get a model: from checkpoint if available, otherwise fresh random."""
    ckpt = find_latest_checkpoint()

    if ckpt is not None:
        print(f"Found checkpoint: {ckpt}")
        from transformer.utils.checkpoint import load_model
        model, config = load_model(str(ckpt))
        return model, config, ckpt
    else:
        print("No checkpoint found -- using fresh model with random weights.")
        print("(Visualizations show embedding structure before training.)\n")
        model, config = build_fresh_model()
        return model, config, None


# ============================================================================
# Token extraction (works with or without checkpoint)
# ============================================================================

def extract_embeddings(model, config):
    """Extract mu, sigma, phi embeddings for semantic token sets.

    Tries (in order):
    1. Semantic token sets via tokenizer + get_token_embeddings
    2. Raw embedding table with BPE-type category labels
    Always succeeds -- never crashes on missing tokenizer or network.
    """
    # Try the semantic token path first
    try:
        from transformer.visualization.belief_space_viz import (
            ALL_TOKENS, TOKEN_CATEGORIES, get_token_embeddings,
        )
        from transformer.utils.checkpoint import get_tokenizer

        tokenizer = get_tokenizer(config)
        if tokenizer is not None:
            mu, sigma, phi, token_ids, valid_tokens = get_token_embeddings(
                model, ALL_TOKENS, tokenizer,
            )
            valid_categories = [
                TOKEN_CATEGORIES[ALL_TOKENS.index(t)] for t in valid_tokens
            ]
            return mu, sigma, phi, token_ids, valid_tokens, valid_categories
    except Exception as e:
        print(f"  Tokenizer unavailable ({type(e).__name__}), using raw embeddings")

    # Fallback: pull directly from the embedding table
    return _extract_raw_embeddings(model)


def _extract_raw_embeddings(model, n_tokens=200):
    """Fallback: extract raw embedding weights for first N tokens.

    Assigns simple bin-based categories by token ID range. Completely
    offline -- no tokenizer or network access needed.
    """
    token_ids = list(range(n_tokens))
    mu_list, sigma_list, phi_list = [], [], []

    with torch.no_grad():
        for tid in token_ids:
            t = torch.tensor([[tid]])
            mu, sigma, phi_val = model.token_embed(t)
            mu_list.append(mu[0, 0].cpu().numpy())
            sigma_list.append(sigma[0, 0].cpu().numpy())
            phi_list.append(phi_val[0, 0].cpu().numpy())

    # Simple offline categories by token ID range (no tokenizer needed)
    # GPT-2 BPE: 0-255 are byte tokens, 256+ are merges
    def _offline_category(tid):
        if tid < 33:
            return 'control'       # ASCII control chars
        elif tid < 48:
            return 'punctuation'   # !"#$%&'()*+,-./
        elif tid < 58:
            return 'digit'         # 0-9
        elif tid < 65:
            return 'punctuation'   # :;<=>?@
        elif tid < 91:
            return 'upper_letter'  # A-Z
        elif tid < 97:
            return 'punctuation'   # [\]^_`
        elif tid < 123:
            return 'lower_letter'  # a-z
        elif tid < 256:
            return 'extended'      # extended ASCII / byte tokens
        else:
            return 'merge'         # BPE merge tokens

    valid_tokens = [f"tok_{tid}" for tid in token_ids]
    categories = [_offline_category(tid) for tid in token_ids]

    return (
        np.stack(mu_list),
        np.stack(sigma_list),
        np.stack(phi_list),
        token_ids,
        valid_tokens,
        categories,
    )


# ============================================================================
# Main pipeline
# ============================================================================

def run():
    """Run the full interactive visualization pipeline."""
    from transformer.visualization.interactive_belief_viz import (
        compute_umap_embedding,
        compute_multi_scale_umap,
        plot_belief_space_3d,
        plot_multi_space_comparison,
        plot_umap_multiscale,
        plot_silhouette_analysis,
        discover_clusters,
        plot_discovered_clusters_3d,
        shap_embedding_attribution,
    )
    from sklearn.metrics import silhouette_score

    print("=" * 70)
    print("INTERACTIVE BELIEF SPACE VISUALIZATION")
    print("  UMAP + Plotly + SHAP  |  Click-to-Run")
    print("=" * 70)
    print()

    # ---- Load model ----
    print("[Setup] Loading model...")
    model, config, ckpt_path = get_model_and_config()

    # ---- Output directory ----
    if ckpt_path:
        output_dir = ckpt_path.parent / "interactive_viz"
    else:
        output_dir = PROJECT_ROOT / "outputs" / "interactive_viz"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[Setup] Output directory: {output_dir}/\n")

    # ---- Extract embeddings ----
    print("[1/6] Extracting embeddings...")
    mu, sigma, phi, token_ids, tokens, categories = extract_embeddings(model, config)
    # Flatten sigma if full covariance (N, K, K) -> (N, K) diagonal
    if sigma.ndim == 3:
        sigma = np.diagonal(sigma, axis1=1, axis2=2)  # (N, K)

    print(f"  Tokens: {len(tokens)}")
    print(f"  mu: {mu.shape}, sigma: {sigma.shape}, phi: {phi.shape}")
    print(f"  Categories: {sorted(set(categories))}")

    # Build a color map that covers whatever categories we got
    _palette = [
        '#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6',
        '#1ABC9C', '#E67E22', '#95A5A6', '#34495E', '#D35400',
    ]
    from transformer.visualization.belief_space_viz import CATEGORY_COLORS as _semantic_colors
    color_map = dict(_semantic_colors)  # start with semantic defaults
    for i, cat in enumerate(sorted(set(categories))):
        if cat not in color_map:
            color_map[cat] = _palette[i % len(_palette)]

    # ---- UMAP 3D ----
    print("\n[2/6] Computing UMAP 3D embedding of belief space...")
    umap_3d = compute_umap_embedding(mu, n_components=3, n_neighbors=15, min_dist=0.1)
    fig_3d = plot_belief_space_3d(
        umap_3d, tokens, categories,
        title="Belief Space (μ) -- UMAP 3D Interactive",
        color_map=color_map,
        save_html=output_dir / "belief_space_umap_3d.html",
        save_png=output_dir / "belief_space_umap_3d.png",
    )
    print(f"  -> {output_dir / 'belief_space_umap_3d.html'}")

    # ---- Multi-space comparison ----
    print("\n[3/6] Computing multi-space UMAP (μ vs Σ vs φ)...")
    mu_2d = compute_umap_embedding(mu, n_components=2)
    sigma_2d = compute_umap_embedding(sigma, n_components=2)
    phi_2d = compute_umap_embedding(phi, n_components=2)
    fig_multi = plot_multi_space_comparison(
        mu_2d, sigma_2d, phi_2d,
        tokens, categories,
        color_map=color_map,
        save_html=output_dir / "multi_space_comparison.html",
        save_png=output_dir / "multi_space_comparison.png",
    )
    print(f"  -> {output_dir / 'multi_space_comparison.html'}")

    # ---- Multi-scale UMAP ----
    print("\n[4/6] Computing multi-scale UMAP...")
    multi_scale = compute_multi_scale_umap(mu, n_neighbors_list=[5, 15, 50])
    fig_ms = plot_umap_multiscale(
        multi_scale, tokens, categories,
        color_map=color_map,
        save_html=output_dir / "umap_multiscale_rg_flow.html",
        save_png=output_dir / "umap_multiscale_rg_flow.png",
    )
    print(f"  -> {output_dir / 'umap_multiscale_rg_flow.html'}")

    # ---- Silhouette ----
    print("\n[5/6] Computing silhouette analysis...")
    unique_cats = sorted(set(categories))
    cat_to_int = {c: i for i, c in enumerate(unique_cats)}
    labels_int = np.array([cat_to_int[c] for c in categories])

    if len(unique_cats) >= 2:
        sil = silhouette_score(mu, labels_int)
        print(f"  Silhouette score: {sil:.3f}")
        fig_sil = plot_silhouette_analysis(
            mu, categories,
            title=f"Semantic Clustering -- Silhouette Analysis (score={sil:.3f})",
            save_path=output_dir / "silhouette_analysis.png",
        )
        if fig_sil is not None:
            import matplotlib.pyplot as plt
            plt.close(fig_sil)
    else:
        sil = None
        print("  Skipped (need >= 2 categories)")

    # ---- HDBSCAN ----
    print("\n[6/6] Running HDBSCAN cluster discovery...")
    cluster_labels, cluster_info = discover_clusters(
        mu, min_cluster_size=max(3, len(mu) // 20),
    )
    print(f"  Discovered {cluster_info['n_clusters']} clusters "
          f"({cluster_info['noise_fraction']:.0%} noise)")

    fig_hdb = plot_discovered_clusters_3d(
        umap_3d, cluster_labels, tokens,
        true_categories=categories,
        title=f"HDBSCAN: {cluster_info['n_clusters']} Discovered Clusters",
        save_html=output_dir / "hdbscan_clusters_3d.html",
        save_png=output_dir / "hdbscan_clusters_3d.png",
    )
    print(f"  -> {output_dir / 'hdbscan_clusters_3d.html'}")

    # ---- SHAP ----
    print("\n[Bonus] Running SHAP feature attribution...")
    shap_result = shap_embedding_attribution(
        model, token_ids, tokens,
        n_background=min(30, len(token_ids)),
        save_path=output_dir / "shap_attribution.png",
    )
    if shap_result is not None:
        top_5 = [
            shap_result['feature_names'][i]
            for i in np.argsort(shap_result['mean_abs_shap'])[-5:]
        ]
        print(f"  Top features: {top_5}")
        import matplotlib.pyplot as plt
        plt.close(shap_result['figure'])

    # ---- Summary ----
    print()
    print("=" * 70)
    print("DONE -- All visualizations saved!")
    print("=" * 70)
    print(f"\n  Output: {output_dir}/")
    print()

    html_files = sorted(output_dir.glob("*.html"))
    png_files = sorted(output_dir.glob("*.png"))
    if html_files:
        print("  Interactive (open in browser):")
        for f in html_files:
            print(f"    {f.name}")
    if png_files:
        print("  Static figures:")
        for f in png_files:
            print(f"    {f.name}")

    if sil is not None:
        print()
        if sil > 0.3:
            print(f"  Semantic clustering: STRONG (silhouette={sil:.3f})")
            print("  Evidence FOR meta-agent hypothesis!")
        elif sil > 0.1:
            print(f"  Semantic clustering: MODERATE (silhouette={sil:.3f})")
        else:
            print(f"  Semantic clustering: WEAK (silhouette={sil:.3f})")

    # ---- Try to open in browser ----
    if html_files:
        try:
            import webbrowser
            main_html = output_dir / "belief_space_umap_3d.html"
            if main_html.exists():
                webbrowser.open(str(main_html))
                print(f"\n  Opened {main_html.name} in browser.")
        except Exception:
            pass

    print()


# ============================================================================
if __name__ == "__main__":
    run()
