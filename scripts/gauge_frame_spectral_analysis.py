#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T8: Gauge Frame Spectral Analysis Across Trained Transformers
=============================================================

Tests the gauge-theoretic prediction that standard transformer attention
implements W_Q W_K^T = σ⁻^2 Ω⁻ᵀ (Limit 3 of the gauge-theoretic hierarchy).

If true, the spectral structure of M^h = W_Q^h (W_K^h)^T should reveal:

  (a) det(M^h) > 0 for all heads -- GL⁺(K) prediction
      (gauge transport lives in the identity component of GL(K))

  (b) Eigenvalue spectra cluster by head function --
      block-diagonal gauge algebra prediction

  (c) Spectral entropy of M^h decreases with layer depth --
      Spectral entropy trend (coarse-graining concentrates spectral weight)

Models analyzed: BERT-base, GPT-2, DistilBERT
"""

import os
import sys
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist

# Suppress tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODELS = {
    "bert-base-uncased": {"type": "encoder", "abbrev": "BERT"},
    "gpt2": {"type": "decoder", "abbrev": "GPT-2"},
    "distilbert-base-uncased": {"type": "encoder", "abbrev": "DistilBERT"},
}

SAVE_DIR = Path("./fig_spectral_analysis")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Publication styling
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ---------------------------------------------------------------------------
# Weight extraction
# ---------------------------------------------------------------------------
def extract_qk_weights(model, model_name: str) -> List[Tuple[torch.Tensor, torch.Tensor, int, int]]:
    """Extract per-head W_Q, W_K weight matrices from a pretrained model.

    For each layer, returns (W_Q_heads, W_K_heads, num_heads, head_dim) where
    W_Q_heads[h] has shape [head_dim, hidden_dim] and similarly for W_K.

    The attention logit is:
        a_{ij}^h = h_i^T (W_Q^h)^T W_K^h h_j / sqrt(d_h)

    So M^h = W_Q^h @ (W_K^h)^T ∈ R^{d_h x d_h} is the per-head attention kernel.

    Returns:
        List of (W_Q_full, W_K_full, num_heads, head_dim) per layer.
        W_Q_full/W_K_full have shape [hidden_dim, hidden_dim].
    """
    layers_data = []

    if "gpt2" in model_name:
        for layer in model.transformer.h:
            # GPT-2 uses a combined c_attn projection: [hidden, 3*hidden]
            # Output order: Q, K, V concatenated along dim=0 of weight
            c_attn_weight = layer.attn.c_attn.weight  # [hidden, 3*hidden] (Conv1D)
            hidden_dim = c_attn_weight.shape[0]
            # Conv1D stores weight as [in, out], so slice on dim=1
            W_Q = c_attn_weight[:, :hidden_dim].T  # [hidden, hidden] -> transpose to match nn.Linear convention
            W_K = c_attn_weight[:, hidden_dim:2*hidden_dim].T
            num_heads = layer.attn.num_heads
            head_dim = hidden_dim // num_heads
            layers_data.append((W_Q.detach().cpu(), W_K.detach().cpu(), num_heads, head_dim))

    elif "distilbert" in model_name:
        for layer in model.transformer.layer:
            attn = layer.attention
            W_Q = attn.q_lin.weight.detach().cpu()  # [hidden, hidden]
            W_K = attn.k_lin.weight.detach().cpu()
            num_heads = attn.n_heads
            head_dim = W_Q.shape[0] // num_heads
            layers_data.append((W_Q, W_K, num_heads, head_dim))

    else:
        # BERT / RoBERTa
        encoder_layers = model.encoder.layer
        for layer in encoder_layers:
            attn_self = layer.attention.self
            W_Q = attn_self.query.weight.detach().cpu()  # [hidden, hidden]
            W_K = attn_self.key.weight.detach().cpu()
            num_heads = attn_self.num_attention_heads
            head_dim = attn_self.attention_head_size
            layers_data.append((W_Q, W_K, num_heads, head_dim))

    return layers_data


def compute_per_head_M(W_Q: torch.Tensor, W_K: torch.Tensor,
                       num_heads: int, head_dim: int) -> List[np.ndarray]:
    """Compute M^h = W_Q^h @ (W_K^h)^T for each head.

    W_Q has shape [hidden_dim, hidden_dim] (nn.Linear weight).
    Per head h: W_Q^h = W_Q[h*d_h : (h+1)*d_h, :], shape [d_h, hidden_dim].
    M^h = W_Q^h @ (W_K^h)^T, shape [d_h, d_h].

    Returns list of d_h x d_h numpy arrays.
    """
    heads = []
    for h in range(num_heads):
        start = h * head_dim
        end = (h + 1) * head_dim
        Wq_h = W_Q[start:end, :].float().numpy()  # [d_h, hidden]
        Wk_h = W_K[start:end, :].float().numpy()  # [d_h, hidden]
        M_h = Wq_h @ Wk_h.T  # [d_h, d_h]
        heads.append(M_h)
    return heads


# ---------------------------------------------------------------------------
# T8a: Determinant analysis (GL+ prediction)
# ---------------------------------------------------------------------------
def analyze_determinants(all_M: Dict[str, List[List[np.ndarray]]]) -> Dict:
    """Test whether det(M^h) > 0 for all heads (GL⁺(K) prediction).

    Under the gauge-theoretic identification W_Q W_K^T = σ⁻^2 Ω⁻ᵀ,
    since Ω ∈ GL⁺(K) (connected component containing identity),
    det(Ω) > 0, hence det(M) = σ⁻^2ᴷ det(Ω⁻ᵀ) > 0.

    Returns summary statistics and per-model results.
    """
    results = {}
    for model_name, layers_M in all_M.items():
        dets = []
        log_abs_dets = []
        signs = []
        for l_idx, heads in enumerate(layers_M):
            for h_idx, M in enumerate(heads):
                sign, logabsdet = np.linalg.slogdet(M)
                dets.append({
                    "layer": l_idx,
                    "head": h_idx,
                    "sign": float(sign),
                    "log_abs_det": float(logabsdet),
                })
                signs.append(sign)
                log_abs_dets.append(logabsdet)

        signs = np.array(signs)
        n_positive = int(np.sum(signs > 0))
        n_negative = int(np.sum(signs < 0))
        n_zero = int(np.sum(signs == 0))
        n_total = len(signs)

        results[model_name] = {
            "n_positive": n_positive,
            "n_negative": n_negative,
            "n_zero": n_zero,
            "n_total": n_total,
            "fraction_positive": n_positive / n_total,
            "gl_plus_holds": n_positive == n_total,
            "log_abs_det_mean": float(np.mean(log_abs_dets)),
            "log_abs_det_std": float(np.std(log_abs_dets)),
            "per_head": dets,
        }

    return results


# ---------------------------------------------------------------------------
# T8b: Eigenvalue spectral clustering
# ---------------------------------------------------------------------------
def compute_spectral_features(M: np.ndarray) -> Dict:
    """Compute spectral features of the attention kernel M^h.

    Returns eigenvalues, spectral entropy, effective rank,
    and other gauge-theoretic diagnostics.
    """
    eigvals = np.linalg.eigvals(M)

    # Sort by magnitude
    mag = np.abs(eigvals)
    sort_idx = np.argsort(-mag)
    eigvals_sorted = eigvals[sort_idx]
    mag_sorted = mag[sort_idx]

    # Spectral entropy: H = -sum p_i log p_i where p_i = |λ_i| / sum|λ_j|
    total_mag = mag_sorted.sum()
    if total_mag > 0:
        p = mag_sorted / total_mag
        p = p[p > 0]
        spectral_entropy = -np.sum(p * np.log(p))
    else:
        spectral_entropy = 0.0

    # Normalized spectral entropy (0 to 1)
    max_entropy = np.log(len(eigvals))
    norm_spectral_entropy = spectral_entropy / max_entropy if max_entropy > 0 else 0.0

    # Effective rank (exponential of spectral entropy of singular values)
    svd_vals = np.linalg.svd(M, compute_uv=False)
    svd_total = svd_vals.sum()
    if svd_total > 0:
        p_svd = svd_vals / svd_total
        p_svd = p_svd[p_svd > 0]
        svd_entropy = -np.sum(p_svd * np.log(p_svd))
        effective_rank = np.exp(svd_entropy)
    else:
        svd_entropy = 0.0
        effective_rank = 0.0

    # Fraction of eigenvalues with positive real part
    frac_positive_real = np.mean(eigvals.real > 0)

    # Eigenvalue phase distribution (argument)
    phases = np.angle(eigvals)

    # Condition number
    cond = mag_sorted[0] / mag_sorted[-1] if mag_sorted[-1] > 0 else np.inf

    # Asymmetry: ||M - M^T|| / ||M|| measures departure from symmetric
    asym = np.linalg.norm(M - M.T) / np.linalg.norm(M) if np.linalg.norm(M) > 0 else 0.0

    # Trace and Frobenius norm
    trace_val = np.trace(M)

    return {
        "eigenvalues": eigvals_sorted,
        "magnitudes": mag_sorted,
        "phases": phases,
        "spectral_entropy": spectral_entropy,
        "norm_spectral_entropy": norm_spectral_entropy,
        "svd_entropy": svd_entropy,
        "effective_rank": effective_rank,
        "frac_positive_real": frac_positive_real,
        "condition_number": cond,
        "asymmetry": asym,
        "trace": trace_val,
        "frobenius_norm": np.linalg.norm(M),
    }


def cluster_heads_by_spectrum(
    spectral_features: List[List[Dict]],
    n_clusters: int = 4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cluster attention heads by their eigenvalue spectral features.

    Uses hierarchical clustering on a feature vector per head:
    [norm_spectral_entropy, effective_rank, frac_positive_real,
     asymmetry, log(condition_number), top-5 normalized magnitudes].

    Returns (labels, feature_matrix, linkage_matrix).
    """
    features = []
    for layer_feats in spectral_features:
        for head_feat in layer_feats:
            d_h = len(head_feat["magnitudes"])
            top_k = min(5, d_h)
            top_mags = head_feat["magnitudes"][:top_k]
            if top_mags[0] > 0:
                top_mags_norm = top_mags / top_mags[0]
            else:
                top_mags_norm = top_mags

            feat_vec = [
                head_feat["norm_spectral_entropy"],
                head_feat["effective_rank"],
                head_feat["frac_positive_real"],
                head_feat["asymmetry"],
                np.log1p(head_feat["condition_number"]) if np.isfinite(head_feat["condition_number"]) else 20.0,
            ] + list(top_mags_norm)
            features.append(feat_vec)

    feature_matrix = np.array(features)

    # Standardize features
    means = feature_matrix.mean(axis=0)
    stds = feature_matrix.std(axis=0)
    stds[stds == 0] = 1.0
    feature_matrix_std = (feature_matrix - means) / stds

    # Hierarchical clustering (Ward's method)
    Z = linkage(feature_matrix_std, method="ward")
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")

    return labels, feature_matrix, Z


# ---------------------------------------------------------------------------
# T8c: Spectral entropy vs layer depth
# ---------------------------------------------------------------------------
def analyze_spectral_entropy_trend(spectral_features: List[List[Dict]]) -> Dict:
    """Test whether spectral entropy decreases with layer depth.

    Deeper layers may concentrate spectral weight into fewer dominant
    modes (lower entropy).

    Returns per-layer statistics and regression results.
    """
    n_layers = len(spectral_features)
    layer_entropies = []
    layer_eff_ranks = []
    layer_asymmetries = []

    for l_idx, layer_feats in enumerate(spectral_features):
        entropies = [f["norm_spectral_entropy"] for f in layer_feats]
        eff_ranks = [f["effective_rank"] for f in layer_feats]
        asyms = [f["asymmetry"] for f in layer_feats]

        layer_entropies.append({
            "layer": l_idx,
            "mean": float(np.mean(entropies)),
            "std": float(np.std(entropies)),
            "median": float(np.median(entropies)),
            "values": [float(e) for e in entropies],
        })
        layer_eff_ranks.append({
            "layer": l_idx,
            "mean": float(np.mean(eff_ranks)),
            "std": float(np.std(eff_ranks)),
            "values": [float(r) for r in eff_ranks],
        })
        layer_asymmetries.append({
            "layer": l_idx,
            "mean": float(np.mean(asyms)),
            "std": float(np.std(asyms)),
            "values": [float(a) for a in asyms],
        })

    # Linear regression: entropy vs layer
    layers = np.arange(n_layers)
    mean_entropies = np.array([le["mean"] for le in layer_entropies])
    mean_eff_ranks = np.array([le["mean"] for le in layer_eff_ranks])
    mean_asyms = np.array([le["mean"] for le in layer_asymmetries])

    entropy_slope, entropy_intercept, entropy_r, entropy_p, entropy_se = stats.linregress(
        layers, mean_entropies
    )
    rank_slope, rank_intercept, rank_r, rank_p, rank_se = stats.linregress(
        layers, mean_eff_ranks
    )
    asym_slope, asym_intercept, asym_r, asym_p, asym_se = stats.linregress(
        layers, mean_asyms
    )

    # Spearman correlation (more robust)
    spearman_entropy = stats.spearmanr(layers, mean_entropies)
    spearman_rank = stats.spearmanr(layers, mean_eff_ranks)

    return {
        "layer_entropies": layer_entropies,
        "layer_effective_ranks": layer_eff_ranks,
        "layer_asymmetries": layer_asymmetries,
        "entropy_regression": {
            "slope": float(entropy_slope),
            "intercept": float(entropy_intercept),
            "r_value": float(entropy_r),
            "p_value": float(entropy_p),
            "std_err": float(entropy_se),
            "decreasing": entropy_slope < 0,
            "significant": entropy_p < 0.05,
        },
        "effective_rank_regression": {
            "slope": float(rank_slope),
            "intercept": float(rank_intercept),
            "r_value": float(rank_r),
            "p_value": float(rank_p),
        },
        "asymmetry_regression": {
            "slope": float(asym_slope),
            "intercept": float(asym_intercept),
            "r_value": float(asym_r),
            "p_value": float(asym_p),
        },
        "spearman_entropy": {
            "rho": float(spearman_entropy.statistic),
            "p_value": float(spearman_entropy.pvalue),
        },
        "spearman_effective_rank": {
            "rho": float(spearman_rank.statistic),
            "p_value": float(spearman_rank.pvalue),
        },
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def plot_determinant_analysis(det_results: Dict, save_dir: Path):
    """Figure 1: Determinant sign distribution and log|det| across models."""
    n_models = len(det_results)
    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 3.5), squeeze=False)

    for idx, (model_name, res) in enumerate(det_results.items()):
        ax = axes[0, idx]
        abbrev = MODELS.get(model_name, {}).get("abbrev", model_name)

        per_head = res["per_head"]
        layers = [d["layer"] for d in per_head]
        heads = [d["head"] for d in per_head]
        signs = [d["sign"] for d in per_head]
        log_dets = [d["log_abs_det"] for d in per_head]

        colors = ["#2ca02c" if s > 0 else "#d62728" if s < 0 else "#7f7f7f" for s in signs]
        ax.scatter(layers, log_dets, c=colors, s=20, alpha=0.7, edgecolors="k", linewidths=0.3)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
        ax.set_xlabel("Layer")
        ax.set_ylabel("log |det(M)|")
        ax.set_title(f"{abbrev}\nGL⁺: {res['fraction_positive']:.0%} positive det")

        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ca02c",
                   markersize=6, label=f"det > 0 ({res['n_positive']})"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62728",
                   markersize=6, label=f"det < 0 ({res['n_negative']})"),
        ]
        ax.legend(handles=legend_elements, loc="best", frameon=False)

    plt.tight_layout()
    fig.savefig(save_dir / "T8a_determinant_signs.png")
    fig.savefig(save_dir / "T8a_determinant_signs.pdf")
    plt.close(fig)


def plot_eigenvalue_spectra(
    all_spectral: Dict[str, List[List[Dict]]],
    save_dir: Path,
):
    """Figure 2: Eigenvalue magnitude spectra across layers for each model."""
    n_models = len(all_spectral)
    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 3.5), squeeze=False)

    for idx, (model_name, spectral_features) in enumerate(all_spectral.items()):
        ax = axes[0, idx]
        abbrev = MODELS.get(model_name, {}).get("abbrev", model_name)
        n_layers = len(spectral_features)

        cmap = plt.cm.viridis
        for l_idx, layer_feats in enumerate(spectral_features):
            color = cmap(l_idx / max(n_layers - 1, 1))
            # Average magnitude spectrum across heads
            all_mags = np.array([f["magnitudes"] for f in layer_feats])
            mean_mags = all_mags.mean(axis=0)
            # Normalize by largest
            if mean_mags[0] > 0:
                mean_mags_norm = mean_mags / mean_mags[0]
            else:
                mean_mags_norm = mean_mags
            ax.semilogy(
                np.arange(len(mean_mags_norm)), mean_mags_norm,
                color=color, alpha=0.8, linewidth=1.0,
                label=f"L{l_idx}" if l_idx % max(n_layers // 4, 1) == 0 else None,
            )

        ax.set_xlabel("Eigenvalue index (sorted by magnitude)")
        ax.set_ylabel("Normalized |λ|")
        ax.set_title(f"{abbrev}: Eigenvalue Spectra")
        ax.legend(loc="best", frameon=False, ncol=2)

    plt.tight_layout()
    fig.savefig(save_dir / "T8b_eigenvalue_spectra.png")
    fig.savefig(save_dir / "T8b_eigenvalue_spectra.pdf")
    plt.close(fig)


def plot_eigenvalue_complex_plane(
    all_spectral: Dict[str, List[List[Dict]]],
    save_dir: Path,
):
    """Figure 3: Eigenvalues in the complex plane, colored by layer."""
    n_models = len(all_spectral)
    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 4), squeeze=False)

    for idx, (model_name, spectral_features) in enumerate(all_spectral.items()):
        ax = axes[0, idx]
        abbrev = MODELS.get(model_name, {}).get("abbrev", model_name)
        n_layers = len(spectral_features)

        cmap = plt.cm.viridis
        for l_idx, layer_feats in enumerate(spectral_features):
            color = cmap(l_idx / max(n_layers - 1, 1))
            for head_feat in layer_feats:
                eigs = head_feat["eigenvalues"]
                ax.scatter(
                    eigs.real, eigs.imag,
                    s=3, c=[color], alpha=0.3, rasterized=True,
                )

        ax.axhline(0, color="gray", linewidth=0.3)
        ax.axvline(0, color="gray", linewidth=0.3)
        ax.set_xlabel("Re(λ)")
        ax.set_ylabel("Im(λ)")
        ax.set_title(f"{abbrev}: Eigenvalues in ℂ")
        ax.set_aspect("equal")

        # Add colorbar for layer
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, n_layers - 1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
        cbar.set_label("Layer")

    plt.tight_layout()
    fig.savefig(save_dir / "T8b_eigenvalues_complex_plane.png")
    fig.savefig(save_dir / "T8b_eigenvalues_complex_plane.pdf")
    plt.close(fig)


def plot_spectral_clustering(
    all_spectral: Dict[str, List[List[Dict]]],
    all_clusters: Dict[str, Tuple],
    save_dir: Path,
):
    """Figure 4: Hierarchical clustering dendrogram and cluster-labeled heads."""
    n_models = len(all_clusters)
    fig, axes = plt.subplots(2, n_models, figsize=(5 * n_models, 7), squeeze=False)

    cluster_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for idx, (model_name, (labels, feat_mat, Z)) in enumerate(all_clusters.items()):
        abbrev = MODELS.get(model_name, {}).get("abbrev", model_name)
        n_layers = len(all_spectral[model_name])
        n_heads = len(all_spectral[model_name][0])

        # Dendrogram
        ax_dendro = axes[0, idx]
        dendrogram(Z, ax=ax_dendro, leaf_rotation=90, leaf_font_size=6,
                   color_threshold=Z[-3, 2] if len(Z) >= 3 else None)
        ax_dendro.set_title(f"{abbrev}: Head Clustering Dendrogram")
        ax_dendro.set_xlabel("Head index (layer x n_heads + head)")
        ax_dendro.set_ylabel("Ward distance")

        # Heatmap: cluster labels by (layer, head)
        ax_heat = axes[1, idx]
        label_grid = labels.reshape(n_layers, n_heads)
        im = ax_heat.imshow(label_grid, aspect="auto", cmap="Set1",
                            vmin=1, vmax=max(labels))
        ax_heat.set_xlabel("Head")
        ax_heat.set_ylabel("Layer")
        ax_heat.set_title(f"{abbrev}: Cluster Assignment")
        plt.colorbar(im, ax=ax_heat, shrink=0.8, label="Cluster")

    plt.tight_layout()
    fig.savefig(save_dir / "T8b_spectral_clustering.png")
    fig.savefig(save_dir / "T8b_spectral_clustering.pdf")
    plt.close(fig)


def plot_spectral_entropy_trend(all_spectral_trends: Dict[str, Dict], save_dir: Path):
    """Figure 5: Spectral entropy and effective rank vs layer depth."""
    n_models = len(all_spectral_trends)
    fig, axes = plt.subplots(2, n_models, figsize=(4.5 * n_models, 6), squeeze=False)

    for idx, (model_name, rg) in enumerate(all_spectral_trends.items()):
        abbrev = MODELS.get(model_name, {}).get("abbrev", model_name)

        # Spectral entropy
        ax_ent = axes[0, idx]
        layer_ents = rg["layer_entropies"]
        layers = [le["layer"] for le in layer_ents]
        means = [le["mean"] for le in layer_ents]
        stds = [le["std"] for le in layer_ents]

        ax_ent.errorbar(layers, means, yerr=stds, fmt="o-", markersize=4,
                        capsize=3, color="#1f77b4", linewidth=1.0)

        # Regression line
        reg = rg["entropy_regression"]
        x_fit = np.array(layers)
        y_fit = reg["slope"] * x_fit + reg["intercept"]
        ax_ent.plot(x_fit, y_fit, "--", color="#d62728", linewidth=0.8,
                    label=f"slope={reg['slope']:.4f}, p={reg['p_value']:.3f}")

        # Plot individual head values
        for le in layer_ents:
            ax_ent.scatter(
                [le["layer"]] * len(le["values"]),
                le["values"],
                s=8, alpha=0.2, color="#1f77b4",
            )

        ax_ent.set_xlabel("Layer")
        ax_ent.set_ylabel("Normalized Spectral Entropy")
        ax_ent.set_title(f"{abbrev}: Spectral Entropy vs Depth")
        ax_ent.legend(loc="best", frameon=False)

        # Effective rank
        ax_rank = axes[1, idx]
        layer_ranks = rg["layer_effective_ranks"]
        r_means = [lr["mean"] for lr in layer_ranks]
        r_stds = [lr["std"] for lr in layer_ranks]

        ax_rank.errorbar(layers, r_means, yerr=r_stds, fmt="s-", markersize=4,
                         capsize=3, color="#ff7f0e", linewidth=1.0)

        rank_reg = rg["effective_rank_regression"]
        y_rank_fit = rank_reg["slope"] * x_fit + rank_reg["intercept"]
        ax_rank.plot(x_fit, y_rank_fit, "--", color="#d62728", linewidth=0.8,
                     label=f"slope={rank_reg['slope']:.2f}, p={rank_reg['p_value']:.3f}")

        for lr in layer_ranks:
            ax_rank.scatter(
                [lr["layer"]] * len(lr["values"]),
                lr["values"],
                s=8, alpha=0.2, color="#ff7f0e",
            )

        ax_rank.set_xlabel("Layer")
        ax_rank.set_ylabel("Effective Rank")
        ax_rank.set_title(f"{abbrev}: Effective Rank vs Depth")
        ax_rank.legend(loc="best", frameon=False)

    plt.tight_layout()
    fig.savefig(save_dir / "T8c_spectral_entropy.png")
    fig.savefig(save_dir / "T8c_spectral_entropy.pdf")
    plt.close(fig)


def plot_asymmetry_analysis(
    all_spectral: Dict[str, List[List[Dict]]],
    save_dir: Path,
):
    """Figure 6: Asymmetry ||M - M^T||/||M|| vs layer.

    A symmetric M corresponds to Ω being orthogonal (SO(K) gauge).
    Asymmetry indicates GL(K) \\ O(K) gauge structure.
    """
    n_models = len(all_spectral)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    for model_name, spectral_features in all_spectral.items():
        abbrev = MODELS.get(model_name, {}).get("abbrev", model_name)
        n_layers = len(spectral_features)
        layer_asym = []
        for layer_feats in spectral_features:
            asym = np.mean([f["asymmetry"] for f in layer_feats])
            layer_asym.append(asym)

        ax.plot(range(n_layers), layer_asym, "o-", markersize=4, label=abbrev, linewidth=1.0)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Asymmetry ||M - M^T|| / ||M||")
    ax.set_title("Gauge Structure: Asymmetry of W_Q W_K^T")
    ax.legend(frameon=False)

    plt.tight_layout()
    fig.savefig(save_dir / "T8_asymmetry_vs_layer.png")
    fig.savefig(save_dir / "T8_asymmetry_vs_layer.pdf")
    plt.close(fig)


def plot_summary_dashboard(
    det_results: Dict, all_spectral_trends: Dict, all_spectral: Dict, save_dir: Path,
):
    """Summary dashboard: key results across all models."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Panel A: GL+ fraction per model
    ax = axes[0, 0]
    model_names = list(det_results.keys())
    abbrevs = [MODELS.get(m, {}).get("abbrev", m) for m in model_names]
    fracs = [det_results[m]["fraction_positive"] for m in model_names]
    bars = ax.bar(abbrevs, fracs, color=["#1f77b4", "#ff7f0e", "#2ca02c"][:len(abbrevs)],
                  edgecolor="k", linewidth=0.5)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_ylabel("Fraction det(M) > 0")
    ax.set_title("(a) GL⁺(K) Prediction")
    ax.set_ylim(0, 1.1)
    for bar, frac in zip(bars, fracs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{frac:.1%}", ha="center", va="bottom", fontsize=8)

    # Panel B: Entropy slope per model
    ax = axes[0, 1]
    slopes = [all_spectral_trends[m]["entropy_regression"]["slope"] for m in model_names]
    pvals = [all_spectral_trends[m]["entropy_regression"]["p_value"] for m in model_names]
    colors_slope = ["#2ca02c" if s < 0 else "#d62728" for s in slopes]
    bars = ax.bar(abbrevs, slopes, color=colors_slope, edgecolor="k", linewidth=0.5)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_ylabel("Entropy slope (per layer)")
    ax.set_title("(b) Spectral Entropy Trend")
    for bar, p in zip(bars, pvals):
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        y_offset = -0.001 if bar.get_height() < 0 else 0.001
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + y_offset,
                sig, ha="center", va="bottom" if bar.get_height() >= 0 else "top", fontsize=8)

    # Panel C: Mean effective rank per layer (overlay all models)
    ax = axes[1, 0]
    for model_name in model_names:
        abbrev = MODELS.get(model_name, {}).get("abbrev", model_name)
        rg = all_spectral_trends[model_name]
        layers = [le["layer"] for le in rg["layer_effective_ranks"]]
        ranks = [le["mean"] for le in rg["layer_effective_ranks"]]
        ax.plot(layers, ranks, "o-", markersize=4, label=abbrev, linewidth=1.0)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Effective Rank")
    ax.set_title("(c) Effective Rank vs Depth")
    ax.legend(frameon=False)

    # Panel D: Mean asymmetry per layer
    ax = axes[1, 1]
    for model_name in model_names:
        abbrev = MODELS.get(model_name, {}).get("abbrev", model_name)
        spectral_features = all_spectral[model_name]
        asym_per_layer = []
        for layer_feats in spectral_features:
            asym_per_layer.append(np.mean([f["asymmetry"] for f in layer_feats]))
        ax.plot(range(len(asym_per_layer)), asym_per_layer, "o-", markersize=4,
                label=abbrev, linewidth=1.0)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Asymmetry ||M - M^T|| / ||M||")
    ax.set_title("(d) Gauge Structure Asymmetry")
    ax.legend(frameon=False)

    plt.suptitle("T8: Gauge Frame Spectral Analysis -- Summary", fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(save_dir / "T8_summary_dashboard.png")
    fig.savefig(save_dir / "T8_summary_dashboard.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Synthetic gauge-theoretic weight generation
# ---------------------------------------------------------------------------
def generate_synthetic_gauge_weights(
    n_layers: int = 12,
    n_heads: int = 12,
    head_dim: int = 64,
    hidden_dim: int = 768,
    gauge_group: str = "GL+",
    entropy_decay: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[List[np.ndarray]]]:
    """Generate synthetic W_Q W_K^T matrices with known gauge-theoretic properties.

    Creates three synthetic "models" to validate the analysis pipeline:

    1. "synthetic_GL+" -- M^h = σ⁻^2 Ω⁻ᵀ where Ω ∈ GL⁺(K), with spectral entropy
       decreasing across layers. All det(M) > 0 by construction.

    2. "synthetic_mixed" -- Half heads GL⁺, half GL⁻ (det < 0). Tests T8a
       discrimination.

    3. "synthetic_random" -- Random Gaussian matrices (null model).
       No gauge structure, no spectral trend expected.

    Args:
        entropy_decay: Rate at which spectral entropy decreases per layer.
    """
    rng = np.random.RandomState(seed)
    results = {}

    # --- Model 1: GL⁺ with spectral entropy decay ---
    from scipy.linalg import expm
    layers_M_glp = []
    for l in range(n_layers):
        heads = []
        for h in range(n_heads):
            # Generate Ω ∈ GL⁺(K) via exponential map: Ω = exp(A)
            # spectral weight concentrates into fewer dominant modes
            # at deeper layers (coarse-graining -> simpler gauge structure).
            # Achieved by making the singular value decay steeper with depth.
            A = rng.randn(head_dim, head_dim) * 0.3
            np.fill_diagonal(A, A.diagonal() + 1.0 + 0.1 * h / n_heads)
            Omega = expm(A)
            M_base = np.linalg.inv(Omega).T / 0.5  # σ⁻^2 Ω⁻ᵀ, det > 0

            # Apply spectral shaping: steepen singular value decay with depth
            U, s, Vt = np.linalg.svd(M_base)
            # Power-law decay exponent increases with layer
            decay_exp = 1.0 + entropy_decay * l
            s_shaped = s[0] * (s / s[0]) ** decay_exp
            M = U @ np.diag(s_shaped) @ Vt
            # Preserve GL⁺: det(M) = det(M_base) * prod(s_shaped/s) which
            # is positive since we only changed magnitudes, not signs
            heads.append(M)
        layers_M_glp.append(heads)
    results["synthetic_GL+"] = layers_M_glp

    # --- Model 2: Mixed GL⁺/GL⁻ ---
    layers_M_mixed = []
    for l in range(n_layers):
        heads = []
        for h in range(n_heads):
            A = rng.randn(head_dim, head_dim) * 0.3
            np.fill_diagonal(A, A.diagonal() + 1.0)
            Omega = expm(A)
            M = np.linalg.inv(Omega).T / 0.5
            # For GL⁻: negate a single row to flip det sign (works for any d)
            if h % 2 == 1:
                M[0, :] = -M[0, :]
            heads.append(M)
        layers_M_mixed.append(heads)
    results["synthetic_mixed"] = layers_M_mixed

    # --- Model 3: Random null ---
    layers_M_rand = []
    for l in range(n_layers):
        heads = []
        for h in range(n_heads):
            # Random Gaussian: no gauge structure
            M = rng.randn(head_dim, head_dim) * 0.1
            heads.append(M)
        layers_M_rand.append(heads)
    results["synthetic_random"] = layers_M_rand

    return results


SYNTHETIC_MODELS = {
    "synthetic_GL+": {"type": "synthetic", "abbrev": "GL⁺(K)"},
    "synthetic_mixed": {"type": "synthetic", "abbrev": "Mixed+/-"},
    "synthetic_random": {"type": "synthetic", "abbrev": "Random"},
}


def run_synthetic_validation(save_dir: Path = SAVE_DIR) -> Dict:
    """Run the T8 pipeline on synthetic data with known ground truth.

    Validates that the analysis correctly identifies:
    - All det > 0 for GL⁺ model
    - Mixed det signs for mixed model
    - Decreasing entropy for GL⁺ model
    - No trend for random model (null)
    """
    print("=" * 72)
    print("T8: Synthetic Validation (Known Ground Truth)")
    print("=" * 72)

    synthetic_M = generate_synthetic_gauge_weights()
    all_spectral = {}
    all_clusters = {}
    all_spectral_trends = {}
    det_results = {}

    for model_name, layers_M in synthetic_M.items():
        abbrev = SYNTHETIC_MODELS[model_name]["abbrev"]
        print(f"\n{'─' * 60}")
        print(f"  Synthetic Model: {abbrev} ({model_name})")
        print(f"{'─' * 60}")

        n_layers = len(layers_M)
        n_heads = len(layers_M[0])
        head_dim = layers_M[0][0].shape[0]
        print(f"  Architecture: {n_layers} layers x {n_heads} heads, d_h = {head_dim}")

        # Spectral features
        layers_spectral = []
        for heads_M in layers_M:
            head_feats = [compute_spectral_features(M) for M in heads_M]
            layers_spectral.append(head_feats)
        all_spectral[model_name] = layers_spectral

        # T8a
        det_res = analyze_determinants({model_name: layers_M})
        det_results.update(det_res)
        r = det_res[model_name]
        print(f"  [T8a] det > 0: {r['n_positive']}/{r['n_total']} ({r['fraction_positive']:.1%})")

        # T8b
        labels, feat_mat, Z = cluster_heads_by_spectrum(layers_spectral, n_clusters=4)
        all_clusters[model_name] = (labels, feat_mat, Z)
        unique, counts = np.unique(labels, return_counts=True)
        print(f"  [T8b] Clusters: {dict(zip(unique.tolist(), counts.tolist()))}")

        # T8c
        rg = analyze_spectral_entropy_trend(layers_spectral)
        all_spectral_trends[model_name] = rg
        ent = rg["entropy_regression"]
        print(f"  [T8c] Entropy slope: {ent['slope']:.6f}, p={ent['p_value']:.4f}")

    # Use SYNTHETIC_MODELS for figure labels
    global MODELS
    orig_models = MODELS
    MODELS = {**orig_models, **SYNTHETIC_MODELS}

    plot_determinant_analysis(det_results, save_dir)
    plot_eigenvalue_spectra(all_spectral, save_dir)
    plot_eigenvalue_complex_plane(all_spectral, save_dir)
    plot_spectral_clustering(all_spectral, all_clusters, save_dir)
    plot_spectral_entropy_trend(all_spectral_trends, save_dir)
    plot_asymmetry_analysis(all_spectral, save_dir)
    plot_summary_dashboard(det_results, all_spectral_trends, all_spectral, save_dir)
    print(f"\n  Figures saved to: {save_dir.resolve()}")

    MODELS = orig_models

    # Validate ground truth expectations
    print(f"\n{'═' * 72}")
    print("  GROUND TRUTH VALIDATION")
    print(f"{'═' * 72}")

    checks = []

    # GL+ should have all positive determinants
    glp = det_results["synthetic_GL+"]
    check1 = glp["gl_plus_holds"]
    print(f"  GL⁺ all det > 0: {check1} (expected: True)")
    checks.append(check1)

    # Mixed should have ~50% positive
    mix = det_results["synthetic_mixed"]
    check2 = 0.4 < mix["fraction_positive"] < 0.6
    print(f"  Mixed ~50% det > 0: {mix['fraction_positive']:.1%} (expected: ~50%): {check2}")
    checks.append(check2)

    # GL+ should have decreasing entropy (negative slope)
    glp_ent = all_spectral_trends["synthetic_GL+"]["entropy_regression"]
    check3 = glp_ent["decreasing"]
    print(f"  GL⁺ entropy decreasing: {check3} (slope={glp_ent['slope']:.6f})")
    checks.append(check3)

    # Random should NOT have significant entropy trend
    rand_ent = all_spectral_trends["synthetic_random"]["entropy_regression"]
    check4 = not rand_ent["significant"]
    print(f"  Random no significant trend: {check4} (p={rand_ent['p_value']:.3f})")
    checks.append(check4)

    all_pass = all(checks)
    print(f"\n  All validations passed: {all_pass}")

    # Save results
    results = {
        "experiment": "T8 Synthetic Validation",
        "checks": {
            "gl_plus_all_positive_det": bool(check1),
            "mixed_approx_half_positive": bool(check2),
            "gl_plus_entropy_decreasing": bool(check3),
            "random_no_significant_trend": bool(check4),
            "all_passed": bool(all_pass),
        },
        "details": {
            name: {
                "determinant": {k: v for k, v in det_results[name].items() if k != "per_head"},
                "entropy_regression": all_spectral_trends[name]["entropy_regression"],
            }
            for name in synthetic_M.keys()
        },
    }
    with open(save_dir / "T8_synthetic_validation.json", "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    return results


# ---------------------------------------------------------------------------
# Main analysis pipeline
# ---------------------------------------------------------------------------
def run_analysis(model_names: Optional[List[str]] = None, save_dir: Path = SAVE_DIR):
    """Run the complete T8 spectral analysis pipeline on pretrained models."""
    from transformers import AutoModel, AutoTokenizer

    if model_names is None:
        model_names = list(MODELS.keys())

    print("=" * 72)
    print("T8: Gauge Frame Spectral Analysis Across Trained Transformers")
    print("=" * 72)

    # Storage for all results
    all_M = {}           # model -> list of list of M matrices
    all_spectral = {}    # model -> list of list of spectral feature dicts
    all_clusters = {}    # model -> (labels, features, linkage)
    all_spectral_trends = {}   # model -> spectral entropy trend analysis
    det_results = {}     # model -> determinant analysis dict

    for model_name in model_names:
        abbrev = MODELS.get(model_name, {}).get("abbrev", model_name)
        print(f"\n{'─' * 60}")
        print(f"  Model: {abbrev} ({model_name})")
        print(f"{'─' * 60}")

        # Load model
        print("  Loading model...")
        model = AutoModel.from_pretrained(model_name)
        model.eval()
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params / 1e6:.1f}M")

        # Extract weights
        print("  Extracting W_Q, W_K weight matrices...")
        layers_data = extract_qk_weights(model, model_name)
        n_layers = len(layers_data)
        n_heads = layers_data[0][2]
        head_dim = layers_data[0][3]
        print(f"  Architecture: {n_layers} layers x {n_heads} heads, d_h = {head_dim}")

        # Compute M^h per layer per head
        print("  Computing M^h = W_Q^h (W_K^h)^T ...")
        layers_M = []
        for l_idx, (W_Q, W_K, nh, dh) in enumerate(layers_data):
            heads_M = compute_per_head_M(W_Q, W_K, nh, dh)
            layers_M.append(heads_M)
        all_M[model_name] = layers_M

        # Compute spectral features
        print("  Computing eigendecompositions...")
        layers_spectral = []
        for l_idx, heads_M in enumerate(layers_M):
            head_feats = []
            for h_idx, M in enumerate(heads_M):
                feat = compute_spectral_features(M)
                head_feats.append(feat)
            layers_spectral.append(head_feats)
        all_spectral[model_name] = layers_spectral

        # T8a: Determinant analysis
        print("\n  [T8a] Determinant analysis (GL⁺ prediction)...")
        det_res = analyze_determinants({model_name: layers_M})
        det_results.update(det_res)
        r = det_res[model_name]
        print(f"    det > 0: {r['n_positive']}/{r['n_total']} ({r['fraction_positive']:.1%})")
        print(f"    det < 0: {r['n_negative']}/{r['n_total']}")
        print(f"    GL⁺(K) holds: {r['gl_plus_holds']}")
        print(f"    log|det| = {r['log_abs_det_mean']:.2f} +/- {r['log_abs_det_std']:.2f}")

        # T8b: Spectral clustering
        print("\n  [T8b] Spectral clustering...")
        labels, feat_mat, Z = cluster_heads_by_spectrum(layers_spectral, n_clusters=4)
        all_clusters[model_name] = (labels, feat_mat, Z)
        unique, counts = np.unique(labels, return_counts=True)
        for u, c in zip(unique, counts):
            print(f"    Cluster {u}: {c} heads")

        # T8c: spectral entropy trend
        print("\n  [T8c] Spectral entropy trend analysis (entropy vs depth)...")
        rg = analyze_spectral_entropy_trend(layers_spectral)
        all_spectral_trends[model_name] = rg
        ent_reg = rg["entropy_regression"]
        print(f"    Entropy slope: {ent_reg['slope']:.6f}")
        print(f"    Entropy decreasing: {ent_reg['decreasing']}")
        print(f"    p-value: {ent_reg['p_value']:.4f}")
        print(f"    Significant (p < 0.05): {ent_reg['significant']}")
        sp = rg["spearman_entropy"]
        print(f"    Spearman ρ: {sp['rho']:.4f} (p = {sp['p_value']:.4f})")

        rank_reg = rg["effective_rank_regression"]
        print(f"    Effective rank slope: {rank_reg['slope']:.4f} (p = {rank_reg['p_value']:.4f})")

        # Free model memory
        del model

    # Generate all figures
    print(f"\n{'═' * 72}")
    print("  Generating figures...")
    print(f"{'═' * 72}")

    plot_determinant_analysis(det_results, save_dir)
    print("  [saved] T8a_determinant_signs.png/pdf")

    plot_eigenvalue_spectra(all_spectral, save_dir)
    print("  [saved] T8b_eigenvalue_spectra.png/pdf")

    plot_eigenvalue_complex_plane(all_spectral, save_dir)
    print("  [saved] T8b_eigenvalues_complex_plane.png/pdf")

    plot_spectral_clustering(all_spectral, all_clusters, save_dir)
    print("  [saved] T8b_spectral_clustering.png/pdf")

    plot_spectral_entropy_trend(all_spectral_trends, save_dir)
    print("  [saved] T8c_spectral_entropy.png/pdf")

    plot_asymmetry_analysis(all_spectral, save_dir)
    print("  [saved] T8_asymmetry_vs_layer.png/pdf")

    plot_summary_dashboard(det_results, all_spectral_trends, all_spectral, save_dir)
    print("  [saved] T8_summary_dashboard.png/pdf")

    # Save numerical results
    results_summary = {
        "experiment": "T8: Gauge Frame Spectral Analysis",
        "models": {},
    }
    for model_name in model_names:
        abbrev = MODELS.get(model_name, {}).get("abbrev", model_name)
        results_summary["models"][model_name] = {
            "abbreviation": abbrev,
            "determinant_analysis": {
                k: v for k, v in det_results[model_name].items() if k != "per_head"
            },
            "spectral_entropy_trend": {
                "entropy_regression": all_spectral_trends[model_name]["entropy_regression"],
                "effective_rank_regression": all_spectral_trends[model_name]["effective_rank_regression"],
                "asymmetry_regression": all_spectral_trends[model_name]["asymmetry_regression"],
                "spearman_entropy": all_spectral_trends[model_name]["spearman_entropy"],
                "spearman_effective_rank": all_spectral_trends[model_name]["spearman_effective_rank"],
            },
            "cluster_sizes": {
                int(u): int(c) for u, c in zip(
                    *np.unique(all_clusters[model_name][0], return_counts=True)
                )
            },
        }

    with open(save_dir / "T8_results.json", "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\n  [saved] T8_results.json")

    # Print final summary
    print(f"\n{'═' * 72}")
    print("  SUMMARY OF FINDINGS")
    print(f"{'═' * 72}")

    print("\n  T8a -- GL⁺(K) Prediction (det > 0):")
    for model_name in model_names:
        abbrev = MODELS.get(model_name, {}).get("abbrev", model_name)
        r = det_results[model_name]
        status = "CONFIRMED" if r["gl_plus_holds"] else "PARTIAL"
        print(f"    {abbrev:12s}: {r['fraction_positive']:6.1%} positive  [{status}]")

    print("\n  T8c -- Spectral Entropy Trend (decreasing with depth):")
    for model_name in model_names:
        abbrev = MODELS.get(model_name, {}).get("abbrev", model_name)
        rg = all_spectral_trends[model_name]
        ent = rg["entropy_regression"]
        direction = "↓ decreasing" if ent["decreasing"] else "↑ increasing"
        sig = f"p={ent['p_value']:.3f}"
        status = "CONFIRMED" if ent["decreasing"] and ent["significant"] else "NOT CONFIRMED"
        print(f"    {abbrev:12s}: slope={ent['slope']:.6f} {direction} ({sig})  [{status}]")

    print(f"\n  Figures saved to: {save_dir.resolve()}")
    print(f"{'═' * 72}\n")

    return results_summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import click

    @click.command()
    @click.option("--models", "-m", multiple=True, default=None,
                  help="Model names to analyze (default: all)")
    @click.option("--save-dir", "-o", default=str(SAVE_DIR),
                  help="Output directory for figures and results")
    @click.option("--n-clusters", "-k", default=4, help="Number of spectral clusters")
    @click.option("--synthetic", is_flag=True, default=False,
                  help="Run synthetic validation with known ground truth")
    def main(models, save_dir, n_clusters, synthetic):
        """T8: Gauge Frame Spectral Analysis Across Trained Transformers."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        if synthetic:
            run_synthetic_validation(save_dir=save_path)
        else:
            model_list = list(models) if models else None
            run_analysis(model_names=model_list, save_dir=save_path)

    main()
