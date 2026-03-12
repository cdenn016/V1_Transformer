# -*- coding: utf-8 -*-
"""
RoBERTa Diagnostic Analysis
============================
Investigates WHY RoBERTa shows lower α–β correlation than other models.

Four hypotheses:
  1. Key-norm CV: RoBERTa's key vectors have higher norm variance,
     amplifying the residual -λ||K_j||² term that breaks the isotropic
     Gaussian approximation.
  2. Per-head optimal temperature: RoBERTa's optimal τ differs from
     the universal prediction τ = 2√d_k = 19.0 (for d_k = 90.25).
  3. Angular vs norm decomposition: The dot-product logit Q·K decomposes
     as ||Q||||K||cos(θ). If norm variation dominates over angular
     variation, the KL (pure distance) approximation loses information
     about the multiplicative norm structure.
  4. Per-head effective temperature dispersion: Standard attention heads
     have implicit per-head temperatures set by Q/K norm scales. If
     RoBERTa's heads have more dispersed effective temperatures, a
     single universal τ fits worse. (Motivated by the gauge transformer's
     learned per-head κ.)

Usage:
    python roberta_diagnostics.py
"""

import os
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModel, AutoTokenizer
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict, List, Tuple
import json

# Reuse infrastructure from transformer_test.py
from transformer_test import (
    CORPUS, DEVICE, _get_qkv_generic, _get_num_layers, _fast_pearsonr,
    MULTI_MODEL_NAMES,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_PASSAGES = 30          # passages to use (balance speed vs statistics)
TAU_PREDICTED = 19.0     # Empirical optimum; theory predicts 2√(d_head) = 2√64 = 16 (19% deviation)
TAU_FINE_SWEEP = np.concatenate([
    np.arange(1.0, 10.0, 1.0),
    np.arange(10.0, 30.0, 1.0),
    np.arange(30.0, 51.0, 5.0),
])
SAVE_DIR = Path("./fig_roberta_diagnostics")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Matplotlib styling (matches transformer_test.py)
import matplotlib as mpl
mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ---------------------------------------------------------------------------
# Experiment 1: Key-norm coefficient of variation
# ---------------------------------------------------------------------------
def compute_key_norm_stats(
    model, tokenizer, model_name: str, corpus: List[str],
    n_passages: int, device: str,
) -> Dict:
    """
    For each (layer, head), compute the coefficient of variation (CV)
    of ||K_j|| across tokens, averaged over passages.

    Returns dict with per-head CVs and summary statistics.
    """
    sub_corpus = corpus[:n_passages]
    n_layers = _get_num_layers(model, model_name)

    # Probe for n_heads, head_dim
    test_inp = tokenizer(sub_corpus[0], return_tensors="pt",
                         truncation=True, max_length=512).to(device)
    with torch.no_grad():
        test_out = model(**test_inp)
    _, _, _, n_heads, head_dim = _get_qkv_generic(
        model, test_out.hidden_states, 0, model_name)

    # Accumulate per-head CVs across passages
    cv_accum = np.zeros((n_layers, n_heads))
    count = 0

    for text in sub_corpus:
        inputs = tokenizer(text, return_tensors="pt",
                           truncation=True, max_length=512).to(device)
        if inputs["input_ids"].shape[1] < 5:
            continue
        with torch.no_grad():
            outputs = model(**inputs)
        hs = outputs.hidden_states

        for li in range(n_layers):
            try:
                Q, K, V, nh, hd = _get_qkv_generic(model, hs, li, model_name)
            except Exception:
                continue
            for hi in range(nh):
                Kh = K[:, hi, :]  # [seq, d]
                norms = torch.norm(Kh, dim=-1).detach().cpu().numpy()  # [seq]
                mu = norms.mean()
                if mu > 1e-8:
                    cv_accum[li, hi] += norms.std() / mu
        count += 1

    cv_accum /= max(count, 1)
    return {
        "per_head_cv": cv_accum,  # [n_layers, n_heads]
        "mean_cv": float(cv_accum.mean()),
        "std_cv": float(cv_accum.std()),
        "n_layers": n_layers,
        "n_heads": n_heads,
        "head_dim": head_dim,
    }


# ---------------------------------------------------------------------------
# Experiment 2: Per-model temperature sweep (fine grid)
# ---------------------------------------------------------------------------
def temperature_sweep(
    model, tokenizer, model_name: str, corpus: List[str],
    n_passages: int, device: str, tau_values: np.ndarray,
) -> Dict:
    """
    For each τ in tau_values, compute the grand-mean Pearson r between
    standard attention α and KL-based attention β across all (layer, head)
    pairs and passages.

    Returns dict with tau_values, mean_r per tau, and optimal tau.
    """
    sub_corpus = corpus[:n_passages]
    n_layers = _get_num_layers(model, model_name)

    # Probe architecture
    test_inp = tokenizer(sub_corpus[0], return_tensors="pt",
                         truncation=True, max_length=512).to(device)
    with torch.no_grad():
        test_out = model(**test_inp)
    _, _, _, n_heads, head_dim = _get_qkv_generic(
        model, test_out.hidden_states, 0, model_name)

    # Pre-extract all QKV to avoid redundant forward passes
    all_qkv = []  # list of (Q, K) per passage per (layer, head)
    for text in sub_corpus:
        inputs = tokenizer(text, return_tensors="pt",
                           truncation=True, max_length=512).to(device)
        if inputs["input_ids"].shape[1] < 5:
            continue
        with torch.no_grad():
            outputs = model(**inputs)
        hs = outputs.hidden_states

        passage_data = []
        for li in range(n_layers):
            try:
                Q, K, V, nh, hd = _get_qkv_generic(model, hs, li, model_name)
            except Exception:
                continue
            for hi in range(nh):
                Qh = Q[:, hi, :]
                Kh = K[:, hi, :]
                # Pre-compute standard attention
                d = Qh.shape[1]
                scores_dot = (Qh @ Kh.T) / np.sqrt(d)
                alpha = F.softmax(scores_dot, dim=1)
                # Pre-compute squared distances
                diff = Qh.unsqueeze(1) - Kh.unsqueeze(0)
                sqdist = torch.sum(diff * diff, dim=-1)
                passage_data.append((
                    alpha.detach().cpu().numpy().ravel(),
                    sqdist.detach().cpu(),
                    li, hi,
                ))
        all_qkv.append(passage_data)

    # Sweep tau
    mean_rs = []
    per_head_optimal = {}  # (layer, head) -> optimal tau

    for tau in tau_values:
        rs = []
        for passage_data in all_qkv:
            for alpha_flat, sqdist, li, hi in passage_data:
                beta = F.softmax(-sqdist / tau, dim=1)
                beta_flat = beta.detach().cpu().numpy().ravel()
                r, _ = _fast_pearsonr(alpha_flat, beta_flat)
                rs.append(r)
        mean_rs.append(float(np.mean(rs)) if rs else 0.0)

    # Find per-head optimal tau
    # Re-organize: for each (layer, head), find best tau
    head_tau_rs = {}  # (li, hi) -> list of r per tau
    for ti, tau in enumerate(tau_values):
        for passage_data in all_qkv:
            for alpha_flat, sqdist, li, hi in passage_data:
                key = (li, hi)
                if key not in head_tau_rs:
                    head_tau_rs[key] = [[] for _ in range(len(tau_values))]
                beta = F.softmax(-sqdist / tau, dim=1)
                beta_flat = beta.detach().cpu().numpy().ravel()
                r, _ = _fast_pearsonr(alpha_flat, beta_flat)
                head_tau_rs[key][ti].append(r)

    per_head_optimal_tau = {}
    for key, rs_by_tau in head_tau_rs.items():
        mean_by_tau = [np.mean(rs) if rs else 0.0 for rs in rs_by_tau]
        best_idx = int(np.argmax(mean_by_tau))
        per_head_optimal_tau[f"L{key[0]}H{key[1]}"] = {
            "optimal_tau": float(tau_values[best_idx]),
            "optimal_r": float(mean_by_tau[best_idx]),
        }

    optimal_idx = int(np.argmax(mean_rs))
    return {
        "tau_values": tau_values.tolist(),
        "mean_rs": mean_rs,
        "optimal_tau": float(tau_values[optimal_idx]),
        "optimal_r": float(mean_rs[optimal_idx]),
        "r_at_predicted": float(mean_rs[
            int(np.argmin(np.abs(tau_values - TAU_PREDICTED)))
        ]),
        "per_head_optimal_tau": per_head_optimal_tau,
        "n_heads": n_heads,
        "head_dim": head_dim,
    }


# ---------------------------------------------------------------------------
# Experiment 3: Angular vs norm decomposition of logit variance
# ---------------------------------------------------------------------------
def logit_variance_decomposition(
    model, tokenizer, model_name: str, corpus: List[str],
    n_passages: int, device: str,
) -> Dict:
    """
    Decompose the variance of dot-product logits Q_i · K_j into
    contributions from norms and angles.

    Q_i · K_j = ||Q_i|| ||K_j|| cos(θ_{ij})

    We measure:
      - Var(Q·K) total logit variance
      - Var(cos θ) angular variance (holding norms constant)
      - Var(||Q||||K||) norm-product variance (holding angle constant)
      - Fraction of logit variance explained by norms vs angles
    """
    sub_corpus = corpus[:n_passages]
    n_layers = _get_num_layers(model, model_name)

    test_inp = tokenizer(sub_corpus[0], return_tensors="pt",
                         truncation=True, max_length=512).to(device)
    with torch.no_grad():
        test_out = model(**test_inp)
    _, _, _, n_heads, head_dim = _get_qkv_generic(
        model, test_out.hidden_states, 0, model_name)

    # Per-head accumulators
    angular_var_accum = np.zeros((n_layers, n_heads))
    norm_var_accum = np.zeros((n_layers, n_heads))
    logit_var_accum = np.zeros((n_layers, n_heads))
    norm_frac_accum = np.zeros((n_layers, n_heads))
    count = 0

    for text in sub_corpus:
        inputs = tokenizer(text, return_tensors="pt",
                           truncation=True, max_length=512).to(device)
        if inputs["input_ids"].shape[1] < 5:
            continue
        with torch.no_grad():
            outputs = model(**inputs)
        hs = outputs.hidden_states

        for li in range(n_layers):
            try:
                Q, K, V, nh, hd = _get_qkv_generic(model, hs, li, model_name)
            except Exception:
                continue
            for hi in range(nh):
                Qh = Q[:, hi, :].detach().cpu().numpy()  # [seq, d]
                Kh = K[:, hi, :].detach().cpu().numpy()

                # Norms
                q_norms = np.linalg.norm(Qh, axis=1, keepdims=True)  # [seq, 1]
                k_norms = np.linalg.norm(Kh, axis=1, keepdims=True)  # [seq, 1]

                # Avoid division by zero
                q_norms_safe = np.maximum(q_norms, 1e-8)
                k_norms_safe = np.maximum(k_norms, 1e-8)

                # Unit vectors
                Q_hat = Qh / q_norms_safe
                K_hat = Kh / k_norms_safe

                # Full logits: Q · K^T (unscaled)
                logits = Qh @ Kh.T  # [seq, seq]

                # Cosine similarity matrix
                cos_theta = Q_hat @ K_hat.T  # [seq, seq]

                # Norm product matrix
                norm_prod = q_norms @ k_norms.T  # [seq, seq]

                # Variances (over all i,j pairs)
                logit_var = np.var(logits)
                angular_var = np.var(cos_theta)
                norm_var = np.var(norm_prod)

                logit_var_accum[li, hi] += logit_var
                angular_var_accum[li, hi] += angular_var
                norm_var_accum[li, hi] += norm_var

                # Correlation of logits with norm_prod vs cos_theta
                logits_flat = logits.ravel()
                cos_flat = cos_theta.ravel()
                norm_flat = norm_prod.ravel()

                if logit_var > 1e-12:
                    r_norm, _ = _fast_pearsonr(logits_flat, norm_flat)
                    r_ang, _ = _fast_pearsonr(logits_flat, cos_flat)
                    # Fraction of logit variance explained by norm product
                    norm_frac_accum[li, hi] += r_norm ** 2
                else:
                    norm_frac_accum[li, hi] += 0.5

        count += 1

    n = max(count, 1)
    logit_var_accum /= n
    angular_var_accum /= n
    norm_var_accum /= n
    norm_frac_accum /= n

    return {
        "logit_var": logit_var_accum,      # [n_layers, n_heads]
        "angular_var": angular_var_accum,
        "norm_var": norm_var_accum,
        "norm_frac_r2": norm_frac_accum,   # R² of logit ~ norm_prod
        "mean_norm_frac": float(norm_frac_accum.mean()),
        "mean_angular_var": float(angular_var_accum.mean()),
        "mean_norm_var": float(norm_var_accum.mean()),
        "n_layers": n_layers,
        "n_heads": n_heads,
    }


# ---------------------------------------------------------------------------
# Experiment 4: Per-head effective temperature dispersion
# ---------------------------------------------------------------------------
def compute_effective_temperatures(
    model, tokenizer, model_name: str, corpus: List[str],
    n_passages: int, device: str,
) -> Dict:
    """
    Measure the implicit per-head temperature in standard attention.

    In dot-product attention: score_ij = Q_i · K_j / √d.
    In KL attention: score_ij = -||Q_i - K_j||² / τ.

    Expanding: Q·K = (||Q||² + ||K||² - ||Q-K||²) / 2, so
    Q·K/√d ≈ -||Q-K||²/(2√d) + (||Q||² + ||K||²)/(2√d).

    The effective temperature that makes the KL scores best match the
    dot-product scores is τ_eff ≈ 2√d · (mean_norm_product / d), but
    more directly, we fit τ per head by finding the τ that maximizes
    correlation between α and β for that specific head.

    Here we instead measure a simpler proxy: the ratio of attention-logit
    standard deviation across heads. Heads with higher logit-std have
    effectively lower temperature (sharper attention). If this ratio
    varies more across heads for RoBERTa than BERT, per-head κ matters
    more.
    """
    sub_corpus = corpus[:n_passages]
    n_layers = _get_num_layers(model, model_name)

    test_inp = tokenizer(sub_corpus[0], return_tensors="pt",
                         truncation=True, max_length=512).to(device)
    with torch.no_grad():
        test_out = model(**test_inp)
    _, _, _, n_heads, head_dim = _get_qkv_generic(
        model, test_out.hidden_states, 0, model_name)

    # Per-head logit std accumulator
    logit_std_accum = np.zeros((n_layers, n_heads))
    # Per-head mean QK norm product
    norm_product_accum = np.zeros((n_layers, n_heads))
    count = 0

    for text in sub_corpus:
        inputs = tokenizer(text, return_tensors="pt",
                           truncation=True, max_length=512).to(device)
        if inputs["input_ids"].shape[1] < 5:
            continue
        with torch.no_grad():
            outputs = model(**inputs)
        hs = outputs.hidden_states

        for li in range(n_layers):
            try:
                Q, K, V, nh, hd = _get_qkv_generic(model, hs, li, model_name)
            except Exception:
                continue
            for hi in range(nh):
                Qh = Q[:, hi, :].detach().cpu()  # [seq, d]
                Kh = K[:, hi, :].detach().cpu()
                d = Qh.shape[1]

                # Dot-product logits (unscaled)
                logits = (Qh @ Kh.T).numpy()
                logit_std_accum[li, hi] += logits.std()

                # Mean norm product
                q_norms = torch.norm(Qh, dim=1)
                k_norms = torch.norm(Kh, dim=1)
                norm_product_accum[li, hi] += (q_norms.mean() * k_norms.mean()).item()

        count += 1

    n = max(count, 1)
    logit_std_accum /= n
    norm_product_accum /= n

    # Cross-head dispersion of logit std (CV across heads per layer)
    # Higher CV = more per-head temperature heterogeneity
    per_layer_cv = []
    for li in range(n_layers):
        row = logit_std_accum[li, :]
        mu = row.mean()
        if mu > 1e-8:
            per_layer_cv.append(row.std() / mu)
        else:
            per_layer_cv.append(0.0)

    return {
        "logit_std": logit_std_accum,           # [n_layers, n_heads]
        "norm_product": norm_product_accum,     # [n_layers, n_heads]
        "per_layer_logit_std_cv": per_layer_cv, # CV across heads per layer
        "mean_logit_std_cv": float(np.mean(per_layer_cv)),
        "logit_std_global_cv": float(logit_std_accum.std() / max(logit_std_accum.mean(), 1e-8)),
        "n_layers": n_layers,
        "n_heads": n_heads,
        "head_dim": head_dim,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_key_norm_cv_comparison(all_cv_results: Dict[str, Dict]):
    """Bar chart: mean key-norm CV per model, sorted."""
    models = sorted(all_cv_results.keys(),
                    key=lambda m: all_cv_results[m]["mean_cv"])
    means = [all_cv_results[m]["mean_cv"] for m in models]
    stds = [all_cv_results[m]["std_cv"] for m in models]

    fig, ax = plt.subplots(figsize=(5, 3))
    colors = ["#e74c3c" if "roberta" in m else "#3498db" for m in models]
    bars = ax.bar(range(len(models)), means, yerr=stds, capsize=3,
                  color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([m.split("/")[-1] for m in models],
                       rotation=30, ha="right")
    ax.set_ylabel("Mean key-norm CV")
    ax.set_title("Key-norm coefficient of variation by model")
    fig.tight_layout()
    fig.savefig(SAVE_DIR / "key_norm_cv_comparison.png")
    plt.close(fig)
    print(f"  Saved: {SAVE_DIR / 'key_norm_cv_comparison.png'}")


def plot_temperature_sweeps(all_tau_results: Dict[str, Dict]):
    """Overlay temperature sweep curves for all models."""
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for mname, res in all_tau_results.items():
        label = mname.split("/")[-1]
        lw = 2.0 if "roberta" in mname else 1.0
        ls = "-" if "roberta" in mname else "--"
        ax.plot(res["tau_values"], res["mean_rs"],
                label=f"{label} (opt={res['optimal_tau']:.0f})",
                linewidth=lw, linestyle=ls)
    ax.axvline(TAU_PREDICTED, color="gray", linestyle=":", linewidth=0.8,
               label=f"predicted τ={TAU_PREDICTED:.0f}")
    ax.set_xlabel("Temperature τ")
    ax.set_ylabel("Mean Pearson r (α vs β)")
    ax.set_title("Temperature sweep by model")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(SAVE_DIR / "temperature_sweep_comparison.png")
    plt.close(fig)
    print(f"  Saved: {SAVE_DIR / 'temperature_sweep_comparison.png'}")


def plot_per_head_optimal_tau_distribution(all_tau_results: Dict[str, Dict]):
    """Histogram of per-head optimal τ for each model."""
    n_models = len(all_tau_results)
    fig, axes = plt.subplots(1, n_models, figsize=(3 * n_models, 3),
                             sharey=True)
    if n_models == 1:
        axes = [axes]

    for ax, (mname, res) in zip(axes, all_tau_results.items()):
        opt_taus = [v["optimal_tau"]
                    for v in res["per_head_optimal_tau"].values()]
        ax.hist(opt_taus, bins=20, color="#e74c3c" if "roberta" in mname
                else "#3498db", edgecolor="black", linewidth=0.5)
        ax.axvline(TAU_PREDICTED, color="gray", linestyle=":", linewidth=0.8)
        ax.set_xlabel("Optimal τ")
        ax.set_title(mname.split("/")[-1])
        median_tau = np.median(opt_taus)
        ax.axvline(median_tau, color="red", linestyle="--", linewidth=0.8,
                   label=f"median={median_tau:.1f}")
        ax.legend(fontsize=6)

    axes[0].set_ylabel("Count (heads)")
    fig.suptitle("Per-head optimal temperature distribution", fontsize=10)
    fig.tight_layout()
    fig.savefig(SAVE_DIR / "per_head_optimal_tau.png")
    plt.close(fig)
    print(f"  Saved: {SAVE_DIR / 'per_head_optimal_tau.png'}")


def plot_norm_fraction_comparison(all_decomp_results: Dict[str, Dict]):
    """Bar chart: mean norm-fraction R² per model."""
    models = sorted(all_decomp_results.keys(),
                    key=lambda m: all_decomp_results[m]["mean_norm_frac"])
    fracs = [all_decomp_results[m]["mean_norm_frac"] for m in models]

    fig, ax = plt.subplots(figsize=(5, 3))
    colors = ["#e74c3c" if "roberta" in m else "#3498db" for m in models]
    ax.bar(range(len(models)), fracs, color=colors,
           edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([m.split("/")[-1] for m in models],
                       rotation=30, ha="right")
    ax.set_ylabel("Mean R²(logit ~ norm product)")
    ax.set_title("Fraction of logit variance explained by norms")
    fig.tight_layout()
    fig.savefig(SAVE_DIR / "norm_fraction_comparison.png")
    plt.close(fig)
    print(f"  Saved: {SAVE_DIR / 'norm_fraction_comparison.png'}")


def plot_cv_vs_correlation(all_cv_results: Dict[str, Dict],
                           all_tau_results: Dict[str, Dict]):
    """Scatter: key-norm CV vs α–β correlation (the key diagnostic plot)."""
    fig, ax = plt.subplots(figsize=(4, 3.5))
    for mname in all_cv_results:
        if mname not in all_tau_results:
            continue
        cv = all_cv_results[mname]["mean_cv"]
        r = all_tau_results[mname]["r_at_predicted"]
        color = "#e74c3c" if "roberta" in mname else "#3498db"
        ax.scatter(cv, r, color=color, s=60, edgecolor="black",
                   linewidth=0.5, zorder=5)
        ax.annotate(mname.split("/")[-1], (cv, r),
                    textcoords="offset points", xytext=(5, 5), fontsize=7)

    ax.set_xlabel("Mean key-norm CV")
    ax.set_ylabel(f"Mean Pearson r at τ={TAU_PREDICTED:.0f}")
    ax.set_title("Key-norm heterogeneity vs α–β correlation")
    fig.tight_layout()
    fig.savefig(SAVE_DIR / "cv_vs_correlation.png")
    plt.close(fig)
    print(f"  Saved: {SAVE_DIR / 'cv_vs_correlation.png'}")


def plot_effective_temp_dispersion(all_eff_temp: Dict[str, Dict]):
    """Bar chart: cross-head logit-std CV per model (temperature dispersion)."""
    models = sorted(all_eff_temp.keys(),
                    key=lambda m: all_eff_temp[m]["logit_std_global_cv"])
    cvs = [all_eff_temp[m]["logit_std_global_cv"] for m in models]

    fig, ax = plt.subplots(figsize=(5, 3))
    colors = ["#e74c3c" if "roberta" in m else "#3498db" for m in models]
    ax.bar(range(len(models)), cvs, color=colors,
           edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([m.split("/")[-1] for m in models],
                       rotation=30, ha="right")
    ax.set_ylabel("CV of per-head logit std")
    ax.set_title("Effective temperature dispersion across heads")
    fig.tight_layout()
    fig.savefig(SAVE_DIR / "effective_temp_dispersion.png")
    plt.close(fig)
    print(f"  Saved: {SAVE_DIR / 'effective_temp_dispersion.png'}")


def plot_temp_dispersion_vs_correlation(all_eff_temp: Dict[str, Dict],
                                        all_tau_results: Dict[str, Dict]):
    """Scatter: effective temperature dispersion vs α–β correlation."""
    fig, ax = plt.subplots(figsize=(4, 3.5))
    for mname in all_eff_temp:
        if mname not in all_tau_results:
            continue
        disp = all_eff_temp[mname]["logit_std_global_cv"]
        r = all_tau_results[mname]["r_at_predicted"]
        color = "#e74c3c" if "roberta" in mname else "#3498db"
        ax.scatter(disp, r, color=color, s=60, edgecolor="black",
                   linewidth=0.5, zorder=5)
        ax.annotate(mname.split("/")[-1], (disp, r),
                    textcoords="offset points", xytext=(5, 5), fontsize=7)

    ax.set_xlabel("CV of per-head logit std (temp dispersion)")
    ax.set_ylabel(f"Mean Pearson r at τ={TAU_PREDICTED:.0f}")
    ax.set_title("Temperature dispersion vs α–β correlation")
    fig.tight_layout()
    fig.savefig(SAVE_DIR / "temp_dispersion_vs_correlation.png")
    plt.close(fig)
    print(f"  Saved: {SAVE_DIR / 'temp_dispersion_vs_correlation.png'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("RoBERTa Diagnostic Analysis")
    print("=" * 60)

    all_cv = {}
    all_tau = {}
    all_decomp = {}
    all_eff_temp = {}

    for mname in MULTI_MODEL_NAMES:
        print(f"\n{'─' * 50}")
        print(f"Model: {mname}")
        print(f"{'─' * 50}")

        try:
            tokenizer = AutoTokenizer.from_pretrained(mname)
            model = AutoModel.from_pretrained(mname, output_hidden_states=True)
            model.eval().to(DEVICE)
        except Exception as e:
            print(f"  SKIP ({e})")
            continue

        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params / 1e6:.1f}M")

        # --- Experiment 1: Key-norm CV ---
        print("\n  [1/4] Key-norm CV...")
        cv_res = compute_key_norm_stats(
            model, tokenizer, mname, CORPUS, N_PASSAGES, DEVICE)
        all_cv[mname] = cv_res
        print(f"    Mean CV = {cv_res['mean_cv']:.4f} "
              f"(std = {cv_res['std_cv']:.4f})")

        # --- Experiment 2: Temperature sweep ---
        print("\n  [2/4] Temperature sweep...")
        tau_res = temperature_sweep(
            model, tokenizer, mname, CORPUS, N_PASSAGES, DEVICE,
            TAU_FINE_SWEEP)
        all_tau[mname] = tau_res
        print(f"    Optimal τ = {tau_res['optimal_tau']:.1f} "
              f"(r = {tau_res['optimal_r']:.4f})")
        print(f"    r at predicted τ={TAU_PREDICTED} = "
              f"{tau_res['r_at_predicted']:.4f}")
        # Per-head optimal tau stats
        opt_taus = [v["optimal_tau"]
                    for v in tau_res["per_head_optimal_tau"].values()]
        print(f"    Per-head optimal τ: "
              f"median={np.median(opt_taus):.1f}, "
              f"mean={np.mean(opt_taus):.1f}, "
              f"std={np.std(opt_taus):.1f}")

        # --- Experiment 3: Logit variance decomposition ---
        print("\n  [3/4] Logit variance decomposition...")
        decomp_res = logit_variance_decomposition(
            model, tokenizer, mname, CORPUS, N_PASSAGES, DEVICE)
        all_decomp[mname] = decomp_res
        print(f"    Mean R²(logit ~ norm) = {decomp_res['mean_norm_frac']:.4f}")
        print(f"    Mean angular var = {decomp_res['mean_angular_var']:.6f}")
        print(f"    Mean norm var = {decomp_res['mean_norm_var']:.2f}")

        # --- Experiment 4: Per-head effective temperature dispersion ---
        print("\n  [4/4] Effective temperature dispersion...")
        eff_temp_res = compute_effective_temperatures(
            model, tokenizer, mname, CORPUS, N_PASSAGES, DEVICE)
        all_eff_temp[mname] = eff_temp_res
        print(f"    Global logit-std CV = {eff_temp_res['logit_std_global_cv']:.4f}")
        print(f"    Mean per-layer logit-std CV = {eff_temp_res['mean_logit_std_cv']:.4f}")

        # Free memory
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- Generate plots ---
    print(f"\n{'=' * 60}")
    print("Generating plots...")
    print(f"{'=' * 60}")

    if all_cv:
        plot_key_norm_cv_comparison(all_cv)
    if all_tau:
        plot_temperature_sweeps(all_tau)
        plot_per_head_optimal_tau_distribution(all_tau)
    if all_decomp:
        plot_norm_fraction_comparison(all_decomp)
    if all_cv and all_tau:
        plot_cv_vs_correlation(all_cv, all_tau)
    if all_eff_temp:
        plot_effective_temp_dispersion(all_eff_temp)
    if all_eff_temp and all_tau:
        plot_temp_dispersion_vs_correlation(all_eff_temp, all_tau)

    # --- Summary table ---
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Model':<25} {'CV':>6} {'τ_opt':>6} {'r@τ*':>7} "
          f"{'r@19':>7} {'R²_norm':>8} {'τ_disp':>7}")
    print("─" * 75)
    for mname in MULTI_MODEL_NAMES:
        if mname not in all_cv:
            continue
        cv = all_cv[mname]["mean_cv"]
        tau_opt = all_tau[mname]["optimal_tau"] if mname in all_tau else 0
        r_opt = all_tau[mname]["optimal_r"] if mname in all_tau else 0
        r_pred = all_tau[mname]["r_at_predicted"] if mname in all_tau else 0
        nf = all_decomp[mname]["mean_norm_frac"] if mname in all_decomp else 0
        td = all_eff_temp[mname]["logit_std_global_cv"] if mname in all_eff_temp else 0
        short = mname.split("/")[-1]
        print(f"{short:<25} {cv:>6.4f} {tau_opt:>6.1f} {r_opt:>7.4f} "
              f"{r_pred:>7.4f} {nf:>8.4f} {td:>7.4f}")

    # Save raw results as JSON
    summary = {}
    for mname in MULTI_MODEL_NAMES:
        if mname not in all_cv:
            continue
        summary[mname] = {
            "key_norm_cv": all_cv[mname]["mean_cv"],
            "key_norm_cv_std": all_cv[mname]["std_cv"],
            "optimal_tau": all_tau.get(mname, {}).get("optimal_tau", None),
            "r_at_optimal_tau": all_tau.get(mname, {}).get("optimal_r", None),
            "r_at_predicted_tau": all_tau.get(mname, {}).get(
                "r_at_predicted", None),
            "norm_frac_r2": all_decomp.get(mname, {}).get(
                "mean_norm_frac", None),
            "effective_temp_dispersion": all_eff_temp.get(mname, {}).get(
                "logit_std_global_cv", None),
        }

    with open(SAVE_DIR / "diagnostic_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {SAVE_DIR / 'diagnostic_summary.json'}")


if __name__ == "__main__":
    main()
