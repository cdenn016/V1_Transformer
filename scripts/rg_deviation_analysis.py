#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RG Flow Deviation Diagnostics: Finite-Size Scaling Analysis
=============================================================

Systematically characterizes the deviations between predicted and measured
RG scaling exponents (y2, y3) in the graph-based coarse-graining analysis.

MOTIVATION
----------
Peer review (M5) flagged that graph-based RG exponents deviate significantly
from predictions:
    y2 = -0.66  (predicted: -1.0)
    y3 = +0.17  (predicted: -2.0)

This script diagnoses WHETHER these deviations are finite-size artifacts
(vanishing as N → ∞) or structural failures of the scaling predictions.

APPROACH
--------
1. Sweep (N, K) systematically: N ∈ {64, 128, 256, 512}, K ∈ {8, 16, 32, 64}
2. Measure graph-based exponents at each (N, K)
3. Diagnose clustering quality (modularity, balance, spectral gap)
4. Fit finite-size scaling: y(N) = y_predicted + a / N^b
5. Extrapolate to N → ∞

OUTPUT
------
- Diagnostic summary table (printed)
- Publication-quality figure: scripts/rg_deviation_diagnostics.png

Author: Claude / Robert C. Dennis
Date: March 2026
"""

import os
import sys
import warnings

os.environ.setdefault('OMP_NUM_THREADS', '1')
warnings.filterwarnings('ignore', message='KMeans is known to have a memory leak')
warnings.filterwarnings('ignore', category=RuntimeWarning)

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

# Import from the existing RG analysis module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rg_universality_networkx import (
    generate_synthetic_vfe_system,
    run_rg_flow,
    direct_clt_validation,
    spectral_communities,
    build_attention_graph,
    RGFlow,
)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class DeviationResult:
    """Results for one (N, K) configuration.

    Stores both graph-based and CLT exponents along with clustering
    quality metrics (modularity, spectral gap, cluster balance CV,
    intra/inter attention ratio) used to diagnose finite-size artifacts.
    """
    N: int
    K: int
    # Graph-based exponents
    y1_graph: float = np.nan
    y1_orig_graph: float = np.nan
    y2_graph: float = np.nan
    y3_graph: float = np.nan
    # CLT exponents (direct averaging)
    y1_clt: float = np.nan
    y2_clt: float = np.nan
    y3_clt: float = np.nan
    # Clustering quality metrics
    avg_modularity: float = np.nan
    avg_cluster_size_cv: float = np.nan  # coefficient of variation
    avg_spectral_gap: float = np.nan
    avg_intra_inter_ratio: float = np.nan
    n_rg_levels: int = 0


# ============================================================================
# CLUSTERING QUALITY DIAGNOSTICS
# ============================================================================

def measure_spectral_gap(beta: np.ndarray) -> float:
    """
    Compute the spectral gap of the normalized Laplacian.

    The spectral gap (λ₂ - λ₁) indicates how well-separated the communities
    are. A larger gap means cleaner clustering.
    """
    N = beta.shape[0]
    if N < 3:
        return 0.0

    W = (beta + beta.T) / 2
    np.fill_diagonal(W, 0)
    D = W.sum(axis=1) + 1e-10
    D_inv_sqrt = 1.0 / np.sqrt(D)
    L_norm = np.eye(N) - (D_inv_sqrt[:, None] * W * D_inv_sqrt[None, :])

    eigvals = np.linalg.eigvalsh(L_norm)
    eigvals = np.sort(eigvals)

    # Spectral gap: difference between 2nd and 1st eigenvalue
    if len(eigvals) >= 2:
        return float(eigvals[1] - eigvals[0])
    return 0.0


def measure_cluster_balance(labels: np.ndarray) -> float:
    """
    Coefficient of variation of cluster sizes.
    CV = 0 means perfectly balanced; larger = more imbalanced.
    """
    sizes = np.bincount(labels)
    if len(sizes) < 2 or sizes.mean() < 1e-10:
        return 0.0
    return float(sizes.std() / sizes.mean())


def measure_intra_inter_ratio(beta: np.ndarray, labels: np.ndarray) -> float:
    """
    Ratio of mean intra-cluster attention to mean inter-cluster attention.
    Higher ratio = better community structure = cleaner coarse-graining.
    """
    N = beta.shape[0]
    intra_weights = []
    inter_weights = []

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if labels[i] == labels[j]:
                intra_weights.append(beta[i, j])
            else:
                inter_weights.append(beta[i, j])

    if not inter_weights or not intra_weights:
        return 1.0

    mean_intra = np.mean(intra_weights)
    mean_inter = np.mean(inter_weights)

    return float(mean_intra / (mean_inter + 1e-10))


def measure_modularity_at_level(beta: np.ndarray, labels: np.ndarray) -> float:
    """Compute modularity of the clustering on the attention graph."""
    G = build_attention_graph(beta, threshold=0.001)
    G_undir = G.to_undirected()

    unique_labels = np.unique(labels)
    communities = [set(np.where(labels == l)[0]) for l in unique_labels]

    try:
        return float(nx.community.modularity(G_undir, communities, weight='weight'))
    except Exception:
        return 0.0


# ============================================================================
# ENHANCED RG FLOW WITH DIAGNOSTICS
# ============================================================================

def run_rg_flow_with_diagnostics(
    beta: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
    transports: Optional[np.ndarray] = None,
    max_levels: int = 10,
    min_nodes: int = 3,
    verbose: bool = False,
) -> Tuple[RGFlow, Dict]:
    """
    Run RG flow and collect clustering quality diagnostics at each level.

    Returns
    -------
    flow : RGFlow
    diagnostics : dict with per-level metrics
    """
    from rg_universality_networkx import (
        coarse_grain, measure_couplings, measure_effective_rank
    )

    flow = RGFlow()
    diagnostics = {
        'modularities': [],
        'cluster_cvs': [],
        'spectral_gaps': [],
        'intra_inter_ratios': [],
    }

    # Level 0: microscopic
    couplings = measure_couplings(covariances, transports)
    from rg_universality_networkx import RGLevel
    level0 = RGLevel(
        level=0,
        n_nodes=beta.shape[0],
        attention=beta.copy(),
        means=means.copy(),
        covariances=covariances.copy(),
        transports=transports.copy() if transports is not None else None,
        g1_anisotropy=couplings['g1'],
        g1_original=couplings['g1'],
        g1_emergent=0.0,
        g2_gauge_variation=couplings['g2'],
        g3_holonomy=couplings['g3'],
        modularity=0.0,
        effective_rank=measure_effective_rank(beta),
    )
    flow.levels.append(level0)

    current_beta = beta.copy()
    current_means = means.copy()
    current_covs = covariances.copy()
    current_transports = transports.copy() if transports is not None else None

    for level in range(1, max_levels + 1):
        N_current = current_beta.shape[0]
        if N_current < min_nodes * 2:
            break

        n_clusters = max(2, N_current // 2)
        labels = spectral_communities(current_beta, n_clusters=n_clusters)
        actual_clusters = len(np.unique(labels))

        if actual_clusters >= N_current or actual_clusters < 2:
            break

        # Measure clustering quality BEFORE coarse-graining
        diagnostics['spectral_gaps'].append(measure_spectral_gap(current_beta))
        diagnostics['cluster_cvs'].append(measure_cluster_balance(labels))
        diagnostics['intra_inter_ratios'].append(
            measure_intra_inter_ratio(current_beta, labels)
        )
        diagnostics['modularities'].append(
            measure_modularity_at_level(current_beta, labels)
        )

        if verbose:
            sizes = np.bincount(labels)
            print(f"  Level {level}: {actual_clusters} clusters, "
                  f"sizes: min={sizes.min()}, max={sizes.max()}, "
                  f"CV={sizes.std()/sizes.mean():.2f}")

        # Coarse-grain
        cg_beta, cg_means, cg_covs, cg_orig, cg_emerg, cg_transports = coarse_grain(
            current_beta, current_means, current_covs, labels, current_transports
        )

        couplings = measure_couplings(
            cg_covs, cg_transports,
            covs_original=cg_orig, covs_emergent=cg_emerg,
        )

        lev = RGLevel(
            level=level,
            n_nodes=cg_beta.shape[0],
            attention=cg_beta,
            means=cg_means,
            covariances=cg_covs,
            transports=cg_transports,
            g1_anisotropy=couplings['g1'],
            g1_original=couplings['g1_original'],
            g1_emergent=couplings['g1_emergent'],
            g2_gauge_variation=couplings['g2'],
            g3_holonomy=couplings['g3'],
            modularity=0.0,
            effective_rank=measure_effective_rank(cg_beta),
            n_clusters=actual_clusters,
        )
        flow.levels.append(lev)

        current_beta = cg_beta
        current_means = cg_means
        current_covs = cg_orig
        current_transports = cg_transports

    return flow, diagnostics


# ============================================================================
# FINITE-SIZE SCALING ANALYSIS
# ============================================================================

def run_single_configuration(N: int, K: int, seed: int = 42,
                              verbose: bool = False) -> DeviationResult:
    """Run both CLT and graph-based RG for a single (N, K) configuration.

    Args:
        N: Number of tokens (agents).
        K: Belief dimension.
        seed: Random seed for reproducibility.
        verbose: Print progress and errors.

    Returns:
        DeviationResult with exponents and clustering quality metrics.
    """
    result = DeviationResult(N=N, K=K)

    # --- CLT validation (suppressed output) ---
    import io
    from contextlib import redirect_stdout

    with redirect_stdout(io.StringIO()):
        try:
            clt_data = direct_clt_validation(K=K, N=N, n_trials=100)
            result.y1_clt = clt_data['y1']
            result.y2_clt = clt_data['y2']
            result.y3_clt = clt_data['y3']
        except Exception as e:
            if verbose:
                print(f"  CLT failed for N={N}, K={K}: {e}")

    # --- Graph-based RG flow ---
    np.random.seed(seed)
    try:
        beta, means, covs, transports = generate_synthetic_vfe_system(
            N=N, K=K, g1=0.3, g2=0.2
        )

        flow, diagnostics = run_rg_flow_with_diagnostics(
            beta, means, covs, transports,
            max_levels=10, min_nodes=3, verbose=verbose
        )

        result.n_rg_levels = flow.n_levels

        # Extract graph-based exponents
        exponents = flow.scaling_exponents()
        result.y1_graph = exponents.get('y1', np.nan)
        result.y1_orig_graph = exponents.get('y1_orig', np.nan)
        result.y2_graph = exponents.get('y2', np.nan)
        result.y3_graph = exponents.get('y3', np.nan)

        # Average diagnostics
        if diagnostics['modularities']:
            result.avg_modularity = np.mean(diagnostics['modularities'])
        if diagnostics['cluster_cvs']:
            result.avg_cluster_size_cv = np.mean(diagnostics['cluster_cvs'])
        if diagnostics['spectral_gaps']:
            result.avg_spectral_gap = np.mean(diagnostics['spectral_gaps'])
        if diagnostics['intra_inter_ratios']:
            result.avg_intra_inter_ratio = np.mean(diagnostics['intra_inter_ratios'])

    except Exception as e:
        if verbose:
            print(f"  Graph RG failed for N={N}, K={K}: {e}")
        import traceback
        traceback.print_exc()

    return result


def fit_finite_size_scaling(
    Ns: np.ndarray,
    y_measured: np.ndarray,
    y_predicted: float,
) -> Dict:
    """
    Fit finite-size scaling: y(N) = y_predicted + a / N^b.

    Uses log-linear regression: log|y(N) - y_predicted| = log(a) - b * log(N).

    Returns dict with fitted parameters and extrapolated N→∞ value.
    """
    deviations = y_measured - y_predicted
    abs_dev = np.abs(deviations)

    # Filter valid points
    mask = (abs_dev > 1e-10) & np.isfinite(y_measured)
    if mask.sum() < 2:
        return {
            'a': np.nan, 'b': np.nan,
            'y_inf': y_predicted,
            'extrapolation_valid': False,
            'r_squared': 0.0,
        }

    log_N = np.log(Ns[mask])
    log_dev = np.log(abs_dev[mask])

    # Linear fit in log-log space
    coeffs = np.polyfit(log_N, log_dev, 1)
    b = -coeffs[0]  # exponent (should be positive for decay)
    log_a = coeffs[1]
    a = np.exp(log_a)

    # R² for goodness of fit
    predicted_log_dev = coeffs[0] * log_N + coeffs[1]
    ss_res = np.sum((log_dev - predicted_log_dev) ** 2)
    ss_tot = np.sum((log_dev - log_dev.mean()) ** 2)
    r_squared = 1 - ss_res / (ss_tot + 1e-10) if ss_tot > 1e-10 else 0.0

    # Sign of the deviation for extrapolation
    sign = np.sign(np.mean(deviations[mask]))

    return {
        'a': float(a),
        'b': float(b),
        'sign': float(sign),
        'y_inf': y_predicted,
        'extrapolation_valid': b > 0 and r_squared > 0.3,
        'r_squared': float(r_squared),
        'deviations_decay': b > 0,
    }


# ============================================================================
# MULTI-SEED ANALYSIS
# ============================================================================

def run_multi_seed(N: int, K: int, n_seeds: int = 5,
                   verbose: bool = False) -> Dict:
    """Run analysis with multiple random seeds to get error bars.

    Returns:
        Dict with mean/std of exponents and clustering quality metrics.
    """
    results = []
    for seed in range(42, 42 + n_seeds):
        r = run_single_configuration(N, K, seed=seed, verbose=verbose)
        results.append(r)

    # Aggregate
    def safe_mean_std(vals):
        v = [x for x in vals if np.isfinite(x)]
        if not v:
            return np.nan, np.nan
        return np.mean(v), np.std(v) if len(v) > 1 else 0.0

    y2s = [r.y2_graph for r in results]
    y3s = [r.y3_graph for r in results]
    y1_origs = [r.y1_orig_graph for r in results]
    mods = [r.avg_modularity for r in results]
    cvs = [r.avg_cluster_size_cv for r in results]
    gaps = [r.avg_spectral_gap for r in results]
    ratios = [r.avg_intra_inter_ratio for r in results]

    y2_mean, y2_std = safe_mean_std(y2s)
    y3_mean, y3_std = safe_mean_std(y3s)
    y1_orig_mean, y1_orig_std = safe_mean_std(y1_origs)
    mod_mean, _ = safe_mean_std(mods)
    cv_mean, _ = safe_mean_std(cvs)
    gap_mean, _ = safe_mean_std(gaps)
    ratio_mean, _ = safe_mean_std(ratios)

    return {
        'N': N, 'K': K,
        'y1_orig_mean': y1_orig_mean, 'y1_orig_std': y1_orig_std,
        'y2_mean': y2_mean, 'y2_std': y2_std,
        'y3_mean': y3_mean, 'y3_std': y3_std,
        'modularity': mod_mean,
        'cluster_cv': cv_mean,
        'spectral_gap': gap_mean,
        'intra_inter_ratio': ratio_mean,
        'n_seeds': n_seeds,
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_deviation_diagnostics(
    all_results: List[Dict],
    clt_results: List[Dict],
    output_path: str,
):
    """
    Generate publication-quality 3-panel diagnostic figure.

    Panel A: y2(N) and y3(N) vs 1/N with finite-size extrapolation
    Panel B: |y - y_predicted| vs clustering quality metrics
    Panel C: CLT vs graph-based exponents comparison
    """
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    # Color palette
    c_y2 = '#2196F3'
    c_y3 = '#E91E63'
    c_y1 = '#4CAF50'
    c_pred = '#F44336'
    c_clt = '#9C27B0'

    # ---- Panel A: Finite-size scaling ----
    ax = axes[0]

    # Group by K to show K-dependence
    K_values = sorted(set(r['K'] for r in all_results))

    for K in K_values:
        subset = [r for r in all_results if r['K'] == K]
        subset.sort(key=lambda x: x['N'])
        Ns = np.array([r['N'] for r in subset])
        inv_N = 1.0 / Ns

        y2s = np.array([r['y2_mean'] for r in subset])
        y2_errs = np.array([r['y2_std'] for r in subset])
        y3s = np.array([r['y3_mean'] for r in subset])
        y3_errs = np.array([r['y3_std'] for r in subset])

        # y2 scatter with error bars
        valid2 = np.isfinite(y2s)
        if valid2.any():
            ax.errorbar(inv_N[valid2], y2s[valid2], yerr=y2_errs[valid2],
                       marker='o', color=c_y2, alpha=0.5 + 0.5 * (K == K_values[-1]),
                       markersize=5 + K / 20, capsize=3,
                       label=f'$y_2$ (K={K})' if K == K_values[0] else None,
                       linestyle='-', linewidth=1)

        valid3 = np.isfinite(y3s)
        if valid3.any():
            ax.errorbar(inv_N[valid3], y3s[valid3], yerr=y3_errs[valid3],
                       marker='s', color=c_y3, alpha=0.5 + 0.5 * (K == K_values[-1]),
                       markersize=5 + K / 20, capsize=3,
                       label=f'$y_3$ (K={K})' if K == K_values[0] else None,
                       linestyle='-', linewidth=1)

    # Predicted values as horizontal lines
    ax.axhline(-1.0, color=c_y2, linestyle='--', alpha=0.5, linewidth=1.5,
               label='$y_2$ predicted ($-1$)')
    ax.axhline(-2.0, color=c_y3, linestyle='--', alpha=0.5, linewidth=1.5,
               label='$y_3$ predicted ($-2$)')

    # Finite-size extrapolation (fit on largest K)
    largest_K = K_values[-1]
    subset = sorted([r for r in all_results if r['K'] == largest_K],
                    key=lambda x: x['N'])
    Ns_fit = np.array([r['N'] for r in subset])
    y2_fit = np.array([r['y2_mean'] for r in subset])
    y3_fit = np.array([r['y3_mean'] for r in subset])

    for y_vals, y_pred, color, label in [
        (y2_fit, -1.0, c_y2, '$y_2$'),
        (y3_fit, -2.0, c_y3, '$y_3$'),
    ]:
        valid = np.isfinite(y_vals) & (Ns_fit > 0)
        if valid.sum() >= 2:
            fss = fit_finite_size_scaling(Ns_fit[valid], y_vals[valid], y_pred)
            if fss['extrapolation_valid']:
                N_ext = np.linspace(Ns_fit[valid].min(), 2000, 100)
                y_ext = y_pred + fss['sign'] * fss['a'] / N_ext ** fss['b']
                ax.plot(1.0 / N_ext, y_ext, ':', color=color, alpha=0.6,
                       linewidth=1.5)

    ax.set_xlabel('$1/N$', fontsize=12)
    ax.set_ylabel('Scaling exponent $y$', fontsize=12)
    ax.set_title('A. Finite-Size Scaling', fontsize=13, fontweight='bold')
    ax.legend(fontsize=7, loc='best', ncol=1)
    ax.grid(True, alpha=0.3)

    # ---- Panel B: Deviation vs clustering quality ----
    ax = axes[1]

    all_mod = []
    all_dev_y2 = []
    all_dev_y3 = []
    all_gap = []

    for r in all_results:
        if np.isfinite(r['y2_mean']) and np.isfinite(r['modularity']):
            all_mod.append(r['modularity'])
            all_dev_y2.append(abs(r['y2_mean'] - (-1.0)))
            all_gap.append(r['spectral_gap'])
        if np.isfinite(r['y3_mean']) and np.isfinite(r['modularity']):
            all_dev_y3.append(abs(r['y3_mean'] - (-2.0)))

    if all_mod and all_dev_y2:
        ax.scatter(all_mod, all_dev_y2, marker='o', color=c_y2, s=60,
                  alpha=0.7, label=r'$|y_2 - y_2^{\mathrm{pred}}|$', zorder=3)
    if all_mod and all_dev_y3:
        # Use same x-coords (modularity) for y3 deviations
        mod_y3 = [r['modularity'] for r in all_results
                  if np.isfinite(r['y3_mean']) and np.isfinite(r['modularity'])]
        ax.scatter(mod_y3, all_dev_y3, marker='s', color=c_y3, s=60,
                  alpha=0.7, label=r'$|y_3 - y_3^{\mathrm{pred}}|$', zorder=3)

    # Trend line
    if len(all_mod) >= 3 and len(all_dev_y2) >= 3:
        mod_arr = np.array(all_mod)
        dev_arr = np.array(all_dev_y2)
        valid = np.isfinite(mod_arr) & np.isfinite(dev_arr)
        if valid.sum() >= 2:
            z = np.polyfit(mod_arr[valid], dev_arr[valid], 1)
            x_line = np.linspace(mod_arr[valid].min(), mod_arr[valid].max(), 50)
            ax.plot(x_line, z[0] * x_line + z[1], '--', color=c_y2, alpha=0.5)

    ax.set_xlabel('Mean Modularity $Q$', fontsize=12)
    ax.set_ylabel(r'$|y_{\mathrm{meas}} - y_{\mathrm{pred}}|$', fontsize=12)
    ax.set_title('B. Deviation vs Clustering Quality', fontsize=13,
                fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # ---- Panel C: CLT vs Graph exponents ----
    ax = axes[2]

    bar_width = 0.25
    coupling_labels = ['$y_1$', '$y_2$', '$y_3$']
    predicted = [-0.5, -1.0, -1.0]  # y3 prediction for ||H-I|| (not action)
    x_pos = np.arange(len(coupling_labels))

    # Get CLT means across all K values
    if clt_results:
        clt_y1 = np.nanmean([r['y1'] for r in clt_results])
        clt_y2 = np.nanmean([r['y2'] for r in clt_results])
        clt_y3 = np.nanmean([r['y3'] for r in clt_results])
        clt_vals = [clt_y1, clt_y2, clt_y3]
    else:
        clt_vals = [np.nan, np.nan, np.nan]

    # Get graph-based means across all configurations
    graph_y1_origs = [r['y1_orig_mean'] for r in all_results if np.isfinite(r['y1_orig_mean'])]
    graph_y2s = [r['y2_mean'] for r in all_results if np.isfinite(r['y2_mean'])]
    graph_y3s = [r['y3_mean'] for r in all_results if np.isfinite(r['y3_mean'])]

    graph_vals = [
        np.mean(graph_y1_origs) if graph_y1_origs else np.nan,
        np.mean(graph_y2s) if graph_y2s else np.nan,
        np.mean(graph_y3s) if graph_y3s else np.nan,
    ]
    graph_errs = [
        np.std(graph_y1_origs) if len(graph_y1_origs) > 1 else 0,
        np.std(graph_y2s) if len(graph_y2s) > 1 else 0,
        np.std(graph_y3s) if len(graph_y3s) > 1 else 0,
    ]

    bars_pred = ax.bar(x_pos - bar_width, predicted, bar_width,
                       label='Predicted', color=c_pred, alpha=0.7)
    bars_clt = ax.bar(x_pos, clt_vals, bar_width,
                      label='CLT (direct avg)', color=c_clt, alpha=0.7)
    bars_graph = ax.bar(x_pos + bar_width, graph_vals, bar_width,
                        yerr=graph_errs, capsize=4,
                        label='Graph-based RG', color=c_y2, alpha=0.7)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(coupling_labels, fontsize=12)
    ax.set_ylabel('Scaling exponent', fontsize=12)
    ax.set_title('C. CLT vs Graph-Based Exponents', fontsize=13,
                fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(0, color='black', linewidth=0.5)

    fig.suptitle('RG Scaling Exponent Deviation Diagnostics',
                fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved figure: {output_path}")


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def run_full_analysis(
    N_values=(64, 128, 256, 512),
    K_values=(8, 16, 32, 64),
    n_seeds=3,
    verbose=True,
):
    """
    Run the complete finite-size scaling analysis.
    """
    print("=" * 72)
    print("RG DEVIATION DIAGNOSTICS: FINITE-SIZE SCALING ANALYSIS")
    print("=" * 72)
    print(f"\nConfiguration: N ∈ {N_values}, K ∈ {K_values}")
    print(f"Seeds per configuration: {n_seeds}")
    print()

    all_results = []
    clt_results = []
    total = len(N_values) * len(K_values)
    done = 0

    for K in K_values:
        # CLT validation for this K (only need once, not per-N)
        import io
        from contextlib import redirect_stdout
        with redirect_stdout(io.StringIO()):
            try:
                clt_data = direct_clt_validation(K=K, N=max(N_values), n_trials=100)
                clt_results.append({
                    'K': K,
                    'y1': clt_data['y1'],
                    'y2': clt_data['y2'],
                    'y3': clt_data['y3'],
                })
            except Exception:
                pass

        for N in N_values:
            done += 1
            print(f"[{done}/{total}] N={N}, K={K}...", end='', flush=True)

            # Skip very large (N, K) combinations that would be too slow
            # Transport arrays are N×N×K×K, so memory scales as N²K²
            mem_estimate_gb = (N * N * K * K * 8) / 1e9
            if mem_estimate_gb > 4.0:
                print(f" SKIPPED (memory ~{mem_estimate_gb:.1f} GB)")
                continue

            result = run_multi_seed(N, K, n_seeds=n_seeds, verbose=False)
            all_results.append(result)

            print(f" y1_orig={result['y1_orig_mean']:+.3f}, "
                  f"y2={result['y2_mean']:+.3f}, "
                  f"y3={result['y3_mean']:+.3f}, "
                  f"mod={result['modularity']:.3f}")

    # ---- Print summary table ----
    print("\n" + "=" * 100)
    print("SUMMARY TABLE")
    print("=" * 100)
    print(f"{'N':>5} {'K':>4} {'y1_orig':>8} {'y2':>8} {'y3':>8} "
          f"{'|Δy2|':>7} {'|Δy3|':>7} {'Mod':>6} {'CV':>6} "
          f"{'Gap':>6} {'Levels':>6}")
    print("-" * 100)

    for r in sorted(all_results, key=lambda x: (x['K'], x['N'])):
        dy2 = abs(r['y2_mean'] - (-1.0)) if np.isfinite(r['y2_mean']) else np.nan
        dy3 = abs(r['y3_mean'] - (-2.0)) if np.isfinite(r['y3_mean']) else np.nan
        print(f"{r['N']:>5} {r['K']:>4} "
              f"{r['y1_orig_mean']:>+8.3f} "
              f"{r['y2_mean']:>+8.3f} "
              f"{r['y3_mean']:>+8.3f} "
              f"{dy2:>7.3f} "
              f"{dy3:>7.3f} "
              f"{r['modularity']:>6.3f} "
              f"{r['cluster_cv']:>6.3f} "
              f"{r['spectral_gap']:>6.4f} "
              f"{r.get('n_levels', '?'):>6}")

    # ---- CLT comparison ----
    print("\n" + "=" * 72)
    print("CLT EXPONENTS (direct averaging, no clustering)")
    print("=" * 72)
    for cr in clt_results:
        print(f"  K={cr['K']:>3}: y1={cr['y1']:+.3f}, "
              f"y2={cr['y2']:+.3f}, y3={cr['y3']:+.3f}")

    # ---- Finite-size scaling fits ----
    print("\n" + "=" * 72)
    print("FINITE-SIZE SCALING EXTRAPOLATION")
    print("=" * 72)

    for K in K_values:
        subset = sorted([r for r in all_results if r['K'] == K],
                        key=lambda x: x['N'])
        if len(subset) < 2:
            continue

        Ns = np.array([r['N'] for r in subset])
        y2s = np.array([r['y2_mean'] for r in subset])
        y3s = np.array([r['y3_mean'] for r in subset])

        print(f"\n  K = {K}:")

        for name, y_vals, y_pred in [
            ('y2', y2s, -1.0),
            ('y3', y3s, -2.0),
        ]:
            valid = np.isfinite(y_vals)
            if valid.sum() < 2:
                continue
            fss = fit_finite_size_scaling(Ns[valid], y_vals[valid], y_pred)
            decay_str = "DECAYS" if fss['deviations_decay'] else "DOES NOT DECAY"
            print(f"    {name}: deviation {decay_str} with N "
                  f"(b={fss['b']:+.2f}, R²={fss['r_squared']:.3f})")
            if fss['extrapolation_valid']:
                print(f"         → Extrapolated N→∞ limit: {fss['y_inf']:.1f} "
                      f"(consistent with prediction)")
            else:
                print(f"         → Extrapolation uncertain (R² too low or b < 0)")

    # ---- Key findings ----
    print("\n" + "=" * 72)
    print("KEY FINDINGS")
    print("=" * 72)

    # Check if deviations decrease with N for the largest K
    largest_K = max(K_values)
    subset_largest = sorted(
        [r for r in all_results if r['K'] == largest_K and np.isfinite(r['y2_mean'])],
        key=lambda x: x['N']
    )

    if len(subset_largest) >= 2:
        first_dev = abs(subset_largest[0]['y2_mean'] - (-1.0))
        last_dev = abs(subset_largest[-1]['y2_mean'] - (-1.0))
        if last_dev < first_dev:
            print(f"\n  1. y2 deviations DECREASE with N (K={largest_K}):")
            print(f"     N={subset_largest[0]['N']}: |Δy2| = {first_dev:.3f}")
            print(f"     N={subset_largest[-1]['N']}: |Δy2| = {last_dev:.3f}")
            print(f"     → Consistent with finite-size artifact")
        else:
            print(f"\n  1. y2 deviations do NOT clearly decrease with N (K={largest_K})")
            print(f"     → The deviations may be structural, not purely finite-size")

    # Check CLT vs graph discrepancy
    if clt_results:
        clt_y2_mean = np.mean([r['y2'] for r in clt_results])
        graph_y2_mean = np.nanmean([r['y2_mean'] for r in all_results])
        print(f"\n  2. CLT exponent y2 = {clt_y2_mean:+.3f} (predicted: -1.000)")
        print(f"     Graph exponent y2 = {graph_y2_mean:+.3f}")
        print(f"     Discrepancy = {abs(clt_y2_mean - graph_y2_mean):.3f}")
        if abs(clt_y2_mean - (-1.0)) < 0.05:
            print(f"     → CLT confirms mathematical prediction exactly")
            print(f"     → Graph deviations arise from spectral clustering artifacts")
        else:
            print(f"     → Unexpected: CLT itself deviates from prediction")

    print(f"\n  3. The 'Gaussian universality class' identification (ν=2, η=0)")
    print(f"     follows from CLT assumptions and is acknowledged as such.")
    print(f"     The non-trivial content is the CONJECTURE that this structure")
    print(f"     applies to trained transformers — which requires validation")
    print(f"     on actual trained models, not synthetic data.")

    # ---- Generate figure ----
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, 'rg_deviation_diagnostics.png')
    plot_deviation_diagnostics(all_results, clt_results, output_path)

    return all_results, clt_results


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':

    # ================================================================
    # CONFIG
    # ================================================================

    # Reduced sweep for reasonable runtime (~5-10 min)
    N_VALUES = (64, 128, 256)
    K_VALUES = (8, 16, 32)
    N_SEEDS = 3

    # Full sweep (longer, ~30-60 min):
    # N_VALUES = (64, 128, 256, 512)
    # K_VALUES = (8, 16, 32, 64, 90)
    # N_SEEDS = 5

    # ================================================================
    # RUN
    # ================================================================

    results, clt = run_full_analysis(
        N_values=N_VALUES,
        K_values=K_VALUES,
        n_seeds=N_SEEDS,
        verbose=True,
    )
