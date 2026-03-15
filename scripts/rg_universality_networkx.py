#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Graph-Theoretic RG Coarse-Graining Analysis (NetworkX)
======================================================

Implements the attention-graph coarse-graining that underlies the
RG universality argument for the transformer limit.

The key claim: under graph coarse-graining of the attention matrix,
both gauge VFE and standard transformer attention converge to the
same modular structure. This is the graph-theoretic signature of
belonging to the same universality class.

STRUCTURE
---------
1. Build attention graphs from β_ij (weighted directed graphs)
2. Detect communities (meta-agents) via spectral clustering
3. Coarse-grain: contract communities to single nodes
4. Measure coupling constants at each coarse-graining level
5. Track RG flow across levels and compare VFE vs transformer

USAGE
-----
    Click-to-run: edit the CONFIG section at the bottom of this file,
    then run:
        python scripts/rg_universality_networkx.py

Author: Claude / Robert C. Dennis
Date: March 2026
"""

import os
import warnings
os.environ.setdefault('OMP_NUM_THREADS', '1')
warnings.filterwarnings('ignore', message='KMeans is known to have a memory leak')

import numpy as np
import networkx as nx
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass, field



# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class RGLevel:
    """State of the theory at one coarse-graining level."""
    level: int
    n_nodes: int
    attention: np.ndarray          # (n_nodes, n_nodes)
    means: np.ndarray              # (n_nodes, K)
    covariances: np.ndarray        # (n_nodes, K, K)
    transports: Optional[np.ndarray] = None  # (n_nodes, n_nodes, K, K)

    # Measured couplings
    g1_anisotropy: float = 0.0
    g1_original: float = 0.0      # anisotropy from original Σ_i only
    g1_emergent: float = 0.0      # anisotropy from within-cluster Var_A(μ)
    g2_gauge_variation: float = 0.0
    g3_holonomy: float = 0.0
    modularity: float = 0.0
    effective_rank: float = 0.0
    n_clusters: int = 0


@dataclass
class RGFlow:
    """Complete RG flow trajectory."""
    levels: List[RGLevel] = field(default_factory=list)

    @property
    def n_levels(self):
        return len(self.levels)

    def scaling_exponents(self) -> Dict[str, float]:
        """
        Fit scaling exponents from the flow using log(n_nodes) as the RG scale.

        Uses log(N₀/N_ℓ) as the coarse-graining scale ζ_ℓ, fitting:
            log g_α = y_α · ζ + const
        """
        if self.n_levels < 3:
            return {}

        import numpy as np

        exponents = {}
        n_nodes = np.array([lev.n_nodes for lev in self.levels])
        # Use cumulative coarse-graining ratio as RG scale
        zetas = np.log(n_nodes[0] / n_nodes)

        for name, attr in [('y1', 'g1_anisotropy'),
                           ('y1_orig', 'g1_original'),
                           ('y2', 'g2_gauge_variation'),
                           ('y3', 'g3_holonomy')]:
            vals = np.array([getattr(lev, attr, 0.0) for lev in self.levels])
            mask = vals > 1e-10
            if mask.sum() >= 2:
                log_vals = np.log(vals[mask])
                zs = zetas[mask]
                if len(zs) >= 2:
                    coeffs = np.polyfit(zs, log_vals, 1)
                    exponents[name] = coeffs[0]
                else:
                    exponents[name] = np.nan
            else:
                exponents[name] = np.nan

        return exponents


# ============================================================================
# ATTENTION GRAPH CONSTRUCTION
# ============================================================================

def build_attention_graph(beta: np.ndarray, threshold: float = 0.01) -> nx.DiGraph:
    """
    Build a weighted directed graph from attention matrix β_ij.

    Parameters
    ----------
    beta : ndarray of shape (N, N)
        Attention weights. beta[i, j] = weight from query i to key j.
    threshold : float
        Minimum weight to include an edge.

    Returns
    -------
    G : nx.DiGraph
        Weighted directed attention graph.
    """
    N = beta.shape[0]
    G = nx.DiGraph()
    G.add_nodes_from(range(N))

    for i in range(N):
        for j in range(N):
            if beta[i, j] > threshold and i != j:
                G.add_edge(i, j, weight=beta[i, j])

    return G


def compute_modularity_directed(G: nx.DiGraph) -> float:
    """
    Compute modularity of a directed graph using the undirected projection.
    """
    G_undir = G.to_undirected()
    try:
        from networkx.algorithms.community import greedy_modularity_communities
        communities = greedy_modularity_communities(G_undir, weight='weight')
        return nx.algorithms.community.modularity(G_undir, communities, weight='weight')
    except Exception:
        return 0.0


def spectral_communities(beta: np.ndarray, n_clusters: Optional[int] = None) -> np.ndarray:
    """
    Detect communities via spectral clustering of the attention matrix.

    Uses the symmetrized attention (β + βᵀ)/2 as an affinity matrix.

    Parameters
    ----------
    beta : ndarray of shape (N, N)
    n_clusters : int or None
        If None, determined by eigengap heuristic.

    Returns
    -------
    labels : ndarray of shape (N,)
        Cluster assignment for each token.
    """
    N = beta.shape[0]
    if N <= 1:
        return np.zeros(N, dtype=int)

    # Symmetrize
    W = (beta + beta.T) / 2
    np.fill_diagonal(W, 0)

    # Degree matrix
    D = np.diag(W.sum(axis=1) + 1e-10)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D)))

    # Normalized Laplacian
    L_norm = np.eye(N) - D_inv_sqrt @ W @ D_inv_sqrt

    # Eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(L_norm)

    if n_clusters is None:
        # Eigengap heuristic: find largest gap in first half of eigenvalues
        gaps = np.diff(eigvals[:N // 2 + 1])
        n_clusters = max(2, np.argmax(gaps) + 1)
        n_clusters = min(n_clusters, N // 2)

    # K-means on first n_clusters eigenvectors
    from sklearn.cluster import KMeans
    features = eigvecs[:, :n_clusters]
    # Normalize rows
    norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-10
    features = features / norms

    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = km.fit_predict(features)

    return labels


# ============================================================================
# COARSE-GRAINING MAP
# ============================================================================

def coarse_grain(
    beta: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
    labels: np.ndarray,
    transports: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Coarse-grain the VFE system by contracting clusters into meta-agents.

    This is the core RG transformation R_n.

    Parameters
    ----------
    beta : (N, N) attention weights
    means : (N, K) belief means
    covariances : (N, K, K) belief covariances
    labels : (N,) cluster assignments
    transports : (N, N, K, K) or None, gauge transport matrices

    Returns
    -------
    beta_cg : (M, M) coarse-grained attention
    means_cg : (M, K) meta-agent means
    covs_cg : (M, K, K) meta-agent covariances (total = original + emergent)
    covs_original : (M, K, K) averaged original covariances only
    covs_emergent : (M, K, K) within-cluster Var_A(μ) only
    transports_cg : (M, M, K, K) or None
    """
    unique_labels = np.unique(labels)
    M = len(unique_labels)
    K = means.shape[1]

    means_cg = np.zeros((M, K))
    covs_cg = np.zeros((M, K, K))
    covs_original = np.zeros((M, K, K))
    covs_emergent = np.zeros((M, K, K))
    beta_cg = np.zeros((M, M))

    label_to_idx = {l: i for i, l in enumerate(unique_labels)}

    for a_label in unique_labels:
        a = label_to_idx[a_label]
        members_a = np.where(labels == a_label)[0]
        n_a = len(members_a)

        # Meta-agent mean
        means_cg[a] = means[members_a].mean(axis=0)

        # Decomposed covariance: original (averaged Σ_i) + emergent (Var_A(μ))
        avg_cov = covariances[members_a].mean(axis=0)
        covs_original[a] = avg_cov
        if n_a > 1:
            deviations = means[members_a] - means_cg[a]
            within_var = (deviations.T @ deviations) / n_a
        else:
            within_var = np.zeros((K, K))
        covs_emergent[a] = within_var
        covs_cg[a] = avg_cov + within_var

        # Meta-agent attention
        for b_label in unique_labels:
            b = label_to_idx[b_label]
            members_b = np.where(labels == b_label)[0]
            beta_cg[a, b] = beta[np.ix_(members_a, members_b)].mean()

    # Renormalize attention rows
    row_sums = beta_cg.sum(axis=1, keepdims=True)
    beta_cg = beta_cg / (row_sums + 1e-10)

    # Coarse-grain transports (if provided)
    transports_cg = None
    if transports is not None:
        transports_cg = np.zeros((M, M, K, K))
        for a_label in unique_labels:
            a = label_to_idx[a_label]
            members_a = np.where(labels == a_label)[0]
            for b_label in unique_labels:
                b = label_to_idx[b_label]
                members_b = np.where(labels == b_label)[0]
                transports_cg[a, b] = transports[
                    np.ix_(members_a, members_b)
                ].mean(axis=(0, 1))

    return beta_cg, means_cg, covs_cg, covs_original, covs_emergent, transports_cg


# ============================================================================
# COUPLING CONSTANT MEASUREMENT
# ============================================================================

def measure_couplings(
    covariances: np.ndarray,
    transports: Optional[np.ndarray] = None,
    covs_original: Optional[np.ndarray] = None,
    covs_emergent: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Measure the effective coupling constants at a given RG level.

    Parameters
    ----------
    covariances : (M, K, K) meta-agent covariances (total)
    transports : (M, M, K, K) or None
    covs_original : (M, K, K) or None — averaged original covariances
    covs_emergent : (M, K, K) or None — within-cluster Var_A(μ)

    Returns
    -------
    dict with g1, g1_original, g1_emergent, g2, g3
    """
    M, K, _ = covariances.shape

    # g₁ (total): anisotropy = average ||Δ_i||_F / ||σ²I||_F
    def _anisotropy(covs):
        vals = []
        for i in range(covs.shape[0]):
            sigma2_eff = np.trace(covs[i]) / K
            Delta_i = covs[i] - sigma2_eff * np.eye(K)
            vals.append(np.linalg.norm(Delta_i) / (sigma2_eff + 1e-10))
        return np.mean(vals) if vals else 0.0

    g1 = _anisotropy(covariances)
    g1_orig = _anisotropy(covs_original) if covs_original is not None else g1
    g1_emerg = _anisotropy(covs_emergent) if covs_emergent is not None else 0.0

    # g₂: gauge variation = std of transports
    g2 = 0.0
    if transports is not None:
        Omega_mean = transports.mean(axis=(0, 1))
        deviations = transports - Omega_mean[None, None, :, :]
        g2 = np.mean(np.linalg.norm(
            deviations.reshape(-1, K, K), axis=(1, 2)
        )) / (np.linalg.norm(Omega_mean) + 1e-10)

    # g₃: holonomy = average ||H_ijk - I|| over triples
    g3 = 0.0
    if transports is not None and M >= 3:
        holonomy_norms = []
        n_samples = min(100, M * (M - 1) * (M - 2) // 6)
        for _ in range(n_samples):
            i, j, k = np.random.choice(M, 3, replace=False)
            H = transports[i, j] @ transports[j, k] @ transports[k, i]
            holonomy_norms.append(np.linalg.norm(H - np.eye(K)))
        g3 = np.mean(holonomy_norms) if holonomy_norms else 0.0

    return {'g1': g1, 'g1_original': g1_orig, 'g1_emergent': g1_emerg,
            'g2': g2, 'g3': g3}


def measure_effective_rank(beta: np.ndarray) -> float:
    """Effective rank via spectral entropy of the attention matrix."""
    _, s, _ = np.linalg.svd(beta)
    s = s[s > 1e-10]
    p = s / s.sum()
    entropy = -np.sum(p * np.log(p + 1e-20))
    return np.exp(entropy)


# ============================================================================
# FULL RG FLOW
# ============================================================================

def run_rg_flow(
    beta: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
    transports: Optional[np.ndarray] = None,
    max_levels: int = 10,
    min_nodes: int = 2,
) -> RGFlow:
    """
    Run the full RG flow by iterating binary coarse-graining (halving).

    Uses forced n_clusters = N//2 at each step instead of eigengap
    heuristic, producing many more RG levels for reliable exponent fits.

    Parameters
    ----------
    beta : (N, N) attention matrix
    means : (N, K) belief means
    covariances : (N, K, K) belief covariances
    transports : (N, N, K, K) or None
    max_levels : int
    min_nodes : int

    Returns
    -------
    flow : RGFlow
    """
    flow = RGFlow()

    # Level 0: microscopic
    couplings = measure_couplings(covariances, transports)
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

    # Iterate binary coarse-graining (halve at each step)
    current_beta = beta.copy()
    current_means = means.copy()
    current_covs = covariances.copy()
    current_transports = transports.copy() if transports is not None else None

    for level in range(1, max_levels + 1):
        N_current = current_beta.shape[0]
        if N_current < min_nodes * 2:
            break

        # Force binary coarse-graining: always halve
        n_clusters = max(2, N_current // 2)
        labels = spectral_communities(current_beta, n_clusters=n_clusters)
        actual_clusters = len(np.unique(labels))

        if actual_clusters >= N_current or actual_clusters < 2:
            break

        # Coarse-grain with decomposition
        cg_beta, cg_means, cg_covs, cg_orig, cg_emerg, cg_transports = coarse_grain(
            current_beta, current_means, current_covs, labels, current_transports
        )

        # Measure couplings with decomposition
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
        current_covs = cg_covs
        current_transports = cg_transports

    return flow


# ============================================================================
# SYNTHETIC DEMO
# ============================================================================

def generate_synthetic_vfe_system(
    N: int = 64,
    K: int = 8,
    g1: float = 0.3,
    g2: float = 0.2,
    sigma2: float = 1.0,
    n_true_clusters: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a synthetic VFE system with known couplings.

    Creates N tokens organized into clusters with controlled anisotropy (g₁)
    and gauge variation (g₂). The coupling strengths are scaled by 1/√K to
    keep the system in the linearized regime where the RG predictions apply.

    Parameters
    ----------
    N : int
        Number of tokens.
    K : int
        Belief dimension.
    g1 : float
        Target anisotropy coupling (before 1/√K scaling).
    g2 : float
        Target gauge variation coupling (before 1/√K scaling).
    sigma2 : float
        Isotropic variance scale.
    n_true_clusters : int or None
        Number of ground-truth clusters. If None, uses max(4, N//8).
    """
    np.random.seed(42)

    if n_true_clusters is None:
        n_true_clusters = max(4, N // 8)

    # Scale couplings by 1/√K to stay in the linearized (small-g) regime.
    # Without this, large K produces huge initial couplings that violate
    # the CLT-based RG predictions (those are linearized around g=0).
    g1_eff = g1 / np.sqrt(K)
    g2_eff = g2 / np.sqrt(K)

    # Cluster centers — spread proportional to √K for proper scaling
    cluster_centers = np.random.randn(n_true_clusters, K) * np.sqrt(K) * 0.5
    cluster_assignments = np.random.randint(0, n_true_clusters, N)

    # Token means: cluster center + within-cluster noise
    within_noise = 0.3  # keep small so clusters are well-separated
    means = np.array([
        cluster_centers[cluster_assignments[i]] + np.random.randn(K) * within_noise
        for i in range(N)
    ])

    # Covariances with controlled anisotropy
    covariances = np.zeros((N, K, K))
    for i in range(N):
        Delta = np.random.randn(K, K)
        Delta = (Delta + Delta.T) / 2
        Delta -= np.trace(Delta) / K * np.eye(K)
        # Normalize perturbation by its Frobenius norm so g1_eff controls magnitude
        Delta_norm = np.linalg.norm(Delta)
        if Delta_norm > 1e-10:
            Delta = Delta / Delta_norm * sigma2 * g1_eff
        Sigma = sigma2 * np.eye(K) + Delta
        # Ensure positive definite
        eigvals = np.linalg.eigvalsh(Sigma)
        if eigvals.min() < 0.01:
            Sigma += (0.02 - eigvals.min()) * np.eye(K)
        covariances[i] = Sigma

    # Gauge transports with controlled variation
    Omega0 = np.eye(K) + 0.02 / np.sqrt(K) * np.random.randn(K, K)
    transports = np.zeros((N, N, K, K))
    for i in range(N):
        for j in range(N):
            dOmega = np.random.randn(K, K) * g2_eff / np.sqrt(K)
            transports[i, j] = Omega0 + dOmega

    # Attention from KL (simplified: use squared distance with temperature)
    tau = 2.0 * np.sqrt(K)  # predicted optimal temperature
    logits = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            diff = means[i] - means[j]
            logits[i, j] = -0.5 * np.dot(diff, diff) / (sigma2 * tau)

    # Softmax
    logits -= logits.max(axis=1, keepdims=True)
    beta = np.exp(logits)
    beta /= beta.sum(axis=1, keepdims=True)

    return beta, means, covariances, transports


def run_synthetic_demo(K: int = 8, N: int = 64):
    """Run the full RG analysis on a synthetic system."""

    print("=" * 72)
    print(f"SYNTHETIC RG FLOW DEMO (K={K}, N={N})")
    print("=" * 72)

    # Generate system
    beta, means, covs, transports = generate_synthetic_vfe_system(N=N, K=K)

    print(f"\nGenerated {N} tokens with K={K} dimensions")
    print(f"Initial attention matrix: {beta.shape}")

    # Run RG flow (binary coarse-graining)
    flow = run_rg_flow(beta, means, covs, transports, max_levels=10)

    print(f"\nRG Flow ({flow.n_levels} levels, binary coarse-graining):")
    print(f"{'Level':>5} {'Nodes':>5} {'g₁(tot)':>8} {'g₁(orig)':>9} "
          f"{'g₁(emer)':>9} {'g₂':>8} {'g₃':>8}")
    print("-" * 70)
    for lev in flow.levels:
        print(f"{lev.level:>5} {lev.n_nodes:>5} {lev.g1_anisotropy:>8.4f} "
              f"{lev.g1_original:>9.4f} {lev.g1_emergent:>9.4f} "
              f"{lev.g2_gauge_variation:>8.4f} {lev.g3_holonomy:>8.4f}")

    # Scaling exponents
    exponents = flow.scaling_exponents()
    print(f"\nMeasured scaling exponents:")
    predicted = {'y1': -0.5, 'y1_orig': -0.5, 'y2': -1.0, 'y3': -2.0}
    for name, val in exponents.items():
        pred = predicted.get(name, np.nan)
        label = {
            'y1': 'g₁ total    ',
            'y1_orig': 'g₁ original',
            'y2': 'g₂         ',
            'y3': 'g₃         ',
        }.get(name, name)
        print(f"  {label}: measured = {val:+.3f}, predicted = {pred:+.3f}")

    print(f"\nPredictions from universality theorem:")
    print(f"  y₁ = -1/2 (original anisotropy decays as 1/√n)")
    print(f"  y₂ = -1   (gauge variation decays as 1/n)")
    print(f"  y₃ = -2   (holonomy decays as 1/n²)")
    print()
    print(f"NOTE: g₁(total) rises because emergent anisotropy Var_A(μ)")
    print(f"      grows under coarse-graining. This is the mechanism that")
    print(f"      transformers absorb into W_Q, W_K — track g₁(original)")
    print(f"      for the CLT decay prediction.")

    return flow


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':

    # ================================================================
    # CONFIG — edit these settings, then click Run (no CLI needed)
    # ================================================================

    CHECKPOINT_PATH = None  # <-- paste your checkpoint path here as a raw string, e.g.:
                            #     r'C:\Users\name\checkpoints\best_model.pt'
                            #     r'/home/user/checkpoints/best_model.pt'
                            # If None, runs synthetic demo.

    K = 8                   # Belief dimension (for synthetic demo)
    N = 64                  # Number of tokens (for synthetic demo)

    # ================================================================
    # RUN — no need to edit below this line
    # ================================================================

    if CHECKPOINT_PATH:
        print("Loading from checkpoint... (not yet implemented)")
        print("Running synthetic demo instead.")

    flow = run_synthetic_demo(K=K, N=N)
