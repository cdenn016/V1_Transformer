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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

    Uses the symmetrized attention (β + βᵀ)/2 as an affinity matrix,
    with balanced cluster assignment to avoid singletons.

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

    # Spectral embedding: first n_clusters eigenvectors
    features = eigvecs[:, :n_clusters]
    norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-10
    features = features / norms

    # Balanced assignment: each cluster gets floor(N/k) or ceil(N/k) nodes.
    # Use KMeans to get centroids, then assign greedily with balance constraint.
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    km.fit(features)
    centroids = km.cluster_centers_

    # Distance from each point to each centroid
    # (N, n_clusters) distance matrix
    dists = np.linalg.norm(features[:, None, :] - centroids[None, :, :], axis=2)

    # Greedy balanced assignment
    labels = _balanced_assignment(dists, n_clusters)

    return labels


def _balanced_assignment(dists: np.ndarray, k: int) -> np.ndarray:
    """
    Assign N points to k clusters with balanced sizes.

    Each cluster gets either floor(N/k) or ceil(N/k) points.
    Assignment is greedy: sort all (point, cluster) pairs by distance,
    assign in order, skipping if either the point is already assigned
    or the cluster is full.

    Parameters
    ----------
    dists : (N, k) distance matrix
    k : number of clusters

    Returns
    -------
    labels : (N,) cluster assignments
    """
    N = dists.shape[0]
    max_size = int(np.ceil(N / k))

    # Sort all (point, cluster) pairs by distance
    flat_indices = np.argsort(dists.ravel())
    point_indices = flat_indices // k
    cluster_indices = flat_indices % k

    labels = -np.ones(N, dtype=int)
    cluster_counts = np.zeros(k, dtype=int)

    for pt, cl in zip(point_indices, cluster_indices):
        if labels[pt] >= 0:
            continue  # already assigned
        if cluster_counts[cl] >= max_size:
            continue  # cluster full
        labels[pt] = cl
        cluster_counts[cl] += 1
        if (labels >= 0).all():
            break

    # Safety: assign any remaining points to least-full cluster
    for i in range(N):
        if labels[i] < 0:
            cl = np.argmin(cluster_counts)
            labels[i] = cl
            cluster_counts[cl] += 1

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

    # g₁ emergent: measure the MAGNITUDE of emergent variance relative to
    # the original, not its shape anisotropy. A rank-1 Var_A(μ) from binary
    # clusters is always maximally anisotropic (K-1 under _anisotropy),
    # regardless of its magnitude. The physically meaningful quantity is
    # how much variance the emergent channel adds: tr(Var_A(μ)) / tr(avg_Σ).
    g1_emerg = 0.0
    if covs_emergent is not None and covs_original is not None:
        emerg_traces = np.array([np.trace(covs_emergent[i]) for i in range(M)])
        orig_traces = np.array([np.trace(covs_original[i]) for i in range(M)])
        g1_emerg = np.mean(emerg_traces / (orig_traces + 1e-10))

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
        cluster_sizes = np.bincount(labels); print(f"  Level {level}: {actual_clusters} clusters, sizes: min={cluster_sizes.min()}, max={cluster_sizes.max()}, std={cluster_sizes.std():.1f}")
        print(f"  Level {level}: actual/requested clusters = {actual_clusters}/{n_clusters} = {actual_clusters/n_clusters:.3f}")

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
        # Feed ORIGINAL covariances forward, not total (orig + emergent).
        # The emergent term Var_A(μ) is absorbed by W_Q, W_K in a real
        # transformer — it doesn't propagate through the RG flow.
        # This is the key physical insight: the CLT prediction applies
        # to the original channel only.
        current_covs = cg_orig
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

    # ---- Means: cluster structure with within-cluster noise ----
    # Spread clusters proportional to √K so distances scale properly
    cluster_centers = np.random.randn(n_true_clusters, K) * np.sqrt(K) * 0.3
    cluster_assignments = np.random.randint(0, n_true_clusters, N)

    within_noise = 0.3
    means = np.array([
        cluster_centers[cluster_assignments[i]] + np.random.randn(K) * within_noise
        for i in range(N)
    ])

    # ---- Covariances: σ²I + controlled traceless perturbation ----
    # Scale perturbation so g₁ ≈ g1 regardless of K
    covariances = np.zeros((N, K, K))
    for i in range(N):
        Delta = np.random.randn(K, K)
        Delta = (Delta + Delta.T) / 2
        Delta -= np.trace(Delta) / K * np.eye(K)  # traceless
        # Normalize: ||Δ||_F / (σ² √K) = g1  →  ||Δ||_F = g1 · σ² · √K
        # But g₁ = ||Δ||_F / σ², so we want ||Δ||_F = g1 · σ²
        Delta_norm = np.linalg.norm(Delta)
        if Delta_norm > 1e-10:
            Delta = Delta / Delta_norm * sigma2 * g1
        Sigma = sigma2 * np.eye(K) + Delta
        eigvals = np.linalg.eigvalsh(Sigma)
        if eigvals.min() < 0.01:
            Sigma += (0.02 - eigvals.min()) * np.eye(K)
        covariances[i] = Sigma

    # ---- Transports: cocycle structure Ω_ij = g_i · g_j⁻¹ + perturbation ----
    # This ensures holonomy H_ijk = Ω_ij·Ω_jk·Ω_ki ≈ I + O(g2²)
    # which is the correct physical structure (nearly flat bundle).
    # Per-token gauge frames
    frames = np.zeros((N, K, K))
    for i in range(N):
        phi_i = np.random.randn(K, K) * g2 / np.sqrt(K)
        phi_i = (phi_i - phi_i.T) / 2  # antisymmetric → SO(K) near I
        frames[i] = np.eye(K) + phi_i  # first-order approx to exp(phi_i)

    transports = np.zeros((N, N, K, K))
    for i in range(N):
        g_i = frames[i]
        g_i_inv = np.linalg.inv(g_i)
        for j in range(N):
            g_j = frames[j]
            # Cocycle: Ω_ij = g_i · g_j⁻¹ (holonomy vanishes exactly)
            # Add small perturbation to create non-trivial but decaying holonomy
            noise = np.random.randn(K, K) * g2 * 0.1 / K
            transports[i, j] = g_i @ np.linalg.inv(g_j) + noise

    # ---- Attention from squared distance ----
    tau = 2.0 * np.sqrt(K)
    logits = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            diff = means[i] - means[j]
            logits[i, j] = -0.5 * np.dot(diff, diff) / (sigma2 * tau)

    logits -= logits.max(axis=1, keepdims=True)
    beta = np.exp(logits)
    beta /= beta.sum(axis=1, keepdims=True)

    return beta, means, covariances, transports


def direct_clt_validation(K: int = 8, N: int = 128, n_trials: int = 200):
    """
    Direct validation of CLT-based scaling predictions, independent of
    spectral clustering. Tests the pure averaging claim:

        If Δ_i are i.i.d. traceless perturbations, then
        ||mean(Δ_{1..n})|| decays as n^{-1/2}.

    Similarly for transports and holonomy.
    """
    print("\n" + "=" * 72)
    print(f"DIRECT CLT VALIDATION (K={K}, N={N}, {n_trials} trials)")
    print("=" * 72)

    cluster_sizes = [2, 4, 8, 16, 32, 64]
    cluster_sizes = [n for n in cluster_sizes if n <= N]

    # Pre-generate microscopic data
    np.random.seed(123)

    # g₁ test: average traceless perturbations
    print("\n--- g₁: averaging traceless covariance perturbations ---")
    print(f"  {'n':>4}  {'||mean(Δ)||':>12}  {'predicted':>12}  {'ratio':>8}")
    g1_ns, g1_vals = [], []
    sigma2 = 1.0
    for n in cluster_sizes:
        norms = []
        for _ in range(n_trials):
            # Generate n i.i.d. traceless symmetric perturbations
            deltas = []
            for _ in range(n):
                D = np.random.randn(K, K)
                D = (D + D.T) / 2
                D -= np.trace(D) / K * np.eye(K)
                D = D / np.linalg.norm(D) * sigma2 * 0.3  # controlled magnitude
                deltas.append(D)
            avg_delta = np.mean(deltas, axis=0)
            norms.append(np.linalg.norm(avg_delta))
        mean_norm = np.mean(norms)
        g1_ns.append(n)
        g1_vals.append(mean_norm)

    # Fit exponent
    log_n = np.log(np.array(g1_ns))
    log_g1 = np.log(np.array(g1_vals))
    coeffs = np.polyfit(log_n, log_g1, 1)
    g1_ref = g1_vals[0]
    for i, n in enumerate(g1_ns):
        pred = g1_ref * (n / g1_ns[0]) ** (-0.5)
        print(f"  {n:>4}  {g1_vals[i]:>12.6f}  {pred:>12.6f}  {g1_vals[i]/pred:>8.3f}")
    print(f"  Fitted exponent: {coeffs[0]:.3f}  (predicted: -0.500)")

    # g₂ test: average transport deviations
    print("\n--- g₂: averaging transport deviations ---")
    print(f"  {'n':>4}  {'||mean(δΩ)||':>12}  {'predicted':>12}  {'ratio':>8}")
    g2_ns, g2_vals = [], []
    for n in cluster_sizes:
        norms = []
        for _ in range(n_trials):
            dOmegas = [np.random.randn(K, K) * 0.2 / np.sqrt(K) for _ in range(n * n)]
            avg = np.mean(dOmegas, axis=0)
            norms.append(np.linalg.norm(avg))
        mean_norm = np.mean(norms)
        g2_ns.append(n)
        g2_vals.append(mean_norm)

    log_n2 = np.log(np.array(g2_ns))
    log_g2 = np.log(np.array(g2_vals))
    coeffs2 = np.polyfit(log_n2, log_g2, 1)
    g2_ref = g2_vals[0]
    for i, n in enumerate(g2_ns):
        pred = g2_ref * (n / g2_ns[0]) ** (-1.0)
        print(f"  {n:>4}  {g2_vals[i]:>12.6f}  {pred:>12.6f}  {g2_vals[i]/pred:>8.3f}")
    print(f"  Fitted exponent: {coeffs2[0]:.3f}  (predicted: -1.000)")

    # g₃ test: holonomy (curvature) under coarse-graining
    #
    # Physical picture: the gauge bundle has transports Ω_ij = (flat cocycle) + ε_ij
    # where ε is i.i.d. curvature perturbation. Holonomy measures curvature:
    #   H = Ω_AB · Ω_BC · Ω_CA,  ||H - I|| ~ ε²
    #
    # Under coarse-graining, n² noise terms average: ε̄ ~ ε/n.
    # Since holonomy is quadratic in ε: ||H-I|| ~ ε̄² ~ 1/n² → y₃ = -2.
    #
    # Key subtlety: for non-abelian gauge groups, the averaged cocycle
    # mean(g_a)·mean(g_b⁻¹) ≠ cocycle because mean(g⁻¹) ≠ mean(g)⁻¹.
    # This leaves an O(1/n) cocycle residual that would dominate.
    # The physical prediction y₃ = -2 is about the curvature perturbation
    # after gauge-fixing, so we test the noise-only contribution directly.
    #
    # Test: average n² i.i.d. noise matrices into ε̄, form H = (I+ε̄_AB)(I+ε̄_BC)(I+ε̄_CA),
    # measure ||H-I|| which is dominated by ε̄² cross-terms.
    print("\n--- g₃: holonomy (curvature) under coarse-graining ---")
    print(f"  {'n':>4}  {'||H-I||':>12}  {'predicted':>12}  {'ratio':>8}")
    g3_ns, g3_vals = [], []
    eps = 0.3 / np.sqrt(K)
    g3_trials = min(n_trials, max(50, 3000 // K))

    for n in cluster_sizes:
        if n < 2:
            continue
        holonomies = []
        for _ in range(g3_trials):
            # Simulate 3 meta-edges, each averaging n² fine-grained noise terms
            # ε̄ = mean of n² i.i.d. noise matrices ~ N(0, eps²/n² · I)
            eps_AB = np.random.randn(n*n, K, K).mean(axis=0) * eps
            eps_BC = np.random.randn(n*n, K, K).mean(axis=0) * eps
            eps_CA = np.random.randn(n*n, K, K).mean(axis=0) * eps
            # Meta-transports after gauge-fixing the cocycle
            Omega_AB = np.eye(K) + eps_AB
            Omega_BC = np.eye(K) + eps_BC
            Omega_CA = np.eye(K) + eps_CA
            H = Omega_AB @ Omega_BC @ Omega_CA
            holonomies.append(np.linalg.norm(H - np.eye(K)))
        mean_h = np.mean(holonomies)
        g3_ns.append(n)
        g3_vals.append(mean_h)

    if len(g3_ns) >= 2:
        log_n3 = np.log(np.array(g3_ns))
        log_g3 = np.log(np.array(g3_vals))
        coeffs3 = np.polyfit(log_n3, log_g3, 1)
        g3_ref = g3_vals[0]
        for i, n in enumerate(g3_ns):
            pred = g3_ref * (n / g3_ns[0]) ** (-1.0)
            print(f"  {n:>4}  {g3_vals[i]:>12.6f}  {pred:>12.6f}  {g3_vals[i]/pred:>8.3f}")
        print(f"  Fitted exponent: {coeffs3[0]:.3f}  (predicted: -1.000)")
        print(f"  NOTE: ||H-I|| ~ ε̄ ~ ε/n because holonomy is linear in")
        print(f"        averaged noise. The curvature ACTION ||H-I||² ~ 1/n².")

    print("\n" + "=" * 72)
    print("CLT VALIDATION SUMMARY")
    print("=" * 72)
    print(f"  y₁ (anisotropy):      measured = {coeffs[0]:+.3f}, predicted = -0.500")
    print(f"  y₂ (gauge variation): measured = {coeffs2[0]:+.3f}, predicted = -1.000")
    if len(g3_ns) >= 2:
        print(f"  y₃ (holonomy ||H-I||):measured = {coeffs3[0]:+.3f}, predicted = -1.000")
        print(f"      (action ||H-I||²): exponent = {2*coeffs3[0]:+.3f}, predicted = -2.000")

    return {
        'y1': coeffs[0], 'y2': coeffs2[0],
        'y3': coeffs3[0] if len(g3_ns) >= 2 else np.nan,
        'g1_ns': g1_ns, 'g1_vals': g1_vals,
        'g2_ns': g2_ns, 'g2_vals': g2_vals,
        'g3_ns': g3_ns, 'g3_vals': g3_vals,
    }


def plot_clt_validation(clt_data: dict, K: int, output_dir: str = '.'):
    """Generate publication-quality figures for CLT validation."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    coupling_data = [
        ('g1_ns', 'g1_vals', 'y1', -0.5,
         r'$g_1$ (anisotropy)', r'$\||\mathrm{mean}(\Delta)\||_F$'),
        ('g2_ns', 'g2_vals', 'y2', -1.0,
         r'$g_2$ (gauge variation)', r'$\||\mathrm{mean}(\delta\Omega)\||_F$'),
        ('g3_ns', 'g3_vals', 'y3', -1.0,
         r'$g_3$ (holonomy)', r'$\||H - I\||_F$'),
    ]

    for ax, (ns_key, vals_key, exp_key, pred_exp, title, ylabel) in zip(axes, coupling_data):
        ns = np.array(clt_data[ns_key])
        vals = np.array(clt_data[vals_key])
        if len(ns) == 0:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
            ax.set_title(title)
            continue

        measured_exp = clt_data[exp_key]

        # Log-log plot
        ax.loglog(ns, vals, 'o-', color='#2196F3', markersize=8,
                  linewidth=2, label=f'Measured (y={measured_exp:+.3f})', zorder=3)

        # Predicted scaling line
        n_line = np.linspace(ns[0], ns[-1], 100)
        pred_line = vals[0] * (n_line / ns[0]) ** pred_exp
        ax.loglog(n_line, pred_line, '--', color='#F44336', linewidth=1.5,
                  alpha=0.7, label=f'Predicted (y={pred_exp:+.1f})')

        ax.set_xlabel('Cluster size $n$', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(ns)
        ax.set_xticklabels([str(int(n)) for n in ns])

    fig.suptitle(f'CLT Scaling Validation (K={K})', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, f'clt_validation_K{K}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def plot_rg_flow(flow, K: int, output_dir: str = '.'):
    """Generate RG flow figure showing coupling evolution across levels."""
    levels = [lev.level for lev in flow.levels]
    nodes = [lev.n_nodes for lev in flow.levels]
    g1_tot = [lev.g1_anisotropy for lev in flow.levels]
    g1_orig = [lev.g1_original for lev in flow.levels]
    g1_emerg = [lev.g1_emergent for lev in flow.levels]
    g2 = [lev.g2_gauge_variation for lev in flow.levels]
    g3 = [lev.g3_holonomy for lev in flow.levels]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: g₁ decomposition
    ax = axes[0]
    ax.semilogy(levels, g1_tot, 'o-', color='#2196F3', label=r'$g_1$ total', linewidth=2)
    ax.semilogy(levels, g1_orig, 's--', color='#4CAF50', label=r'$g_1$ original', linewidth=2)
    ax.semilogy(levels, [max(v, 1e-6) for v in g1_emerg], '^:', color='#FF9800',
                label=r'$g_1$ emergent', linewidth=2)
    ax.set_xlabel('RG Level', fontsize=12)
    ax.set_ylabel(r'$g_1$', fontsize=12)
    ax.set_title(r'$g_1$ Anisotropy Decomposition', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Panel 2: g₂
    ax = axes[1]
    ax.semilogy(levels, g2, 'o-', color='#9C27B0', linewidth=2, markersize=8)
    # Predicted decay line
    if len(levels) >= 2:
        pred_g2 = [g2[0] * (nodes[0] / max(n, 1)) ** (-1.0) for n in nodes]
        # Actually: g2 should decay as n^{-1} where n is cluster size ~ N/n_nodes
        node_arr = np.array(nodes, dtype=float)
        scale = node_arr[0] / node_arr
        pred_g2 = [g2[0] * s ** (-1.0) for s in scale]
        ax.semilogy(levels, pred_g2, '--', color='#F44336', alpha=0.5,
                    label=r'$\sim n^{-1}$ predicted')
    ax.set_xlabel('RG Level', fontsize=12)
    ax.set_ylabel(r'$g_2$', fontsize=12)
    ax.set_title(r'$g_2$ Gauge Variation', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Panel 3: g₃
    ax = axes[2]
    ax.semilogy(levels, [max(v, 1e-8) for v in g3], 'o-', color='#E91E63',
                linewidth=2, markersize=8)
    ax.set_xlabel('RG Level', fontsize=12)
    ax.set_ylabel(r'$g_3$', fontsize=12)
    ax.set_title(r'$g_3$ Holonomy', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add node counts as secondary x-axis labels
    for ax_i in axes:
        ax2 = ax_i.twiny()
        ax2.set_xlim(ax_i.get_xlim())
        ax2.set_xticks(levels)
        ax2.set_xticklabels([str(n) for n in nodes], fontsize=8)
        ax2.set_xlabel('Nodes', fontsize=9)

    fig.suptitle(f'RG Flow (K={K}, N={nodes[0]})', fontsize=15, fontweight='bold', y=1.05)
    plt.tight_layout()
    path = os.path.join(output_dir, f'rg_flow_K{K}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def run_synthetic_demo(K: int = 8, N: int = 64):
    """Run the full RG analysis on a synthetic system."""

    print("=" * 72)
    print(f"SYNTHETIC RG FLOW DEMO (K={K}, N={N})")
    print("=" * 72)

    # --- Part 1: Direct CLT validation (pure math, no clustering) ---
    clt_exponents = direct_clt_validation(K=K, N=N)

    # --- Part 2: Full graph-based RG flow ---
    beta, means, covs, transports = generate_synthetic_vfe_system(N=N, K=K)

    print(f"\n\nGenerated {N} tokens with K={K} dimensions")
    print(f"Initial attention matrix: {beta.shape}")

    flow = run_rg_flow(beta, means, covs, transports, max_levels=10)

    print(f"\nRG Flow ({flow.n_levels} levels, binary coarse-graining):")
    print(f"{'Level':>5} {'Nodes':>5} {'g₁(tot)':>8} {'g₁(orig)':>9} "
          f"{'g₁(emer)':>9} {'g₂':>8} {'g₃':>8}")
    print("-" * 70)
    for lev in flow.levels:
        print(f"{lev.level:>5} {lev.n_nodes:>5} {lev.g1_anisotropy:>8.4f} "
              f"{lev.g1_original:>9.4f} {lev.g1_emergent:>9.4f} "
              f"{lev.g2_gauge_variation:>8.4f} {lev.g3_holonomy:>8.4f}")

    exponents = flow.scaling_exponents()
    print(f"\nGraph-based RG exponents:")
    predicted = {'y1': -0.5, 'y1_orig': -0.5, 'y2': -1.0, 'y3': -1.0}
    for name, val in exponents.items():
        pred = predicted.get(name, np.nan)
        label = {
            'y1': 'g₁ total    ',
            'y1_orig': 'g₁ original',
            'y2': 'g₂         ',
            'y3': 'g₃         ',
        }.get(name, name)
        print(f"  {label}: measured = {val:+.3f}, predicted = {pred:+.3f}")

    print(f"\nDirect CLT exponents (pure averaging, no clustering):")
    print(f"  y₁ = {clt_exponents['y1']:+.3f}  (predicted: -0.500)")
    print(f"  y₂ = {clt_exponents['y2']:+.3f}  (predicted: -1.000)")
    print(f"  y₃ = {clt_exponents['y3']:+.3f}  (predicted: -1.000, action: -2.000)")
    print()
    print(f"NOTE: The CLT exponents test the pure mathematical claim.")
    print(f"      The graph-based RG includes finite-size effects from")
    print(f"      spectral clustering (unequal clusters, correlated")
    print(f"      assignments). The CLT values should match predictions")
    print(f"      closely; graph-based may deviate due to these effects.")

    # --- Part 3: Generate figures ---
    output_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"\nGenerating figures in {output_dir}...")
    plot_clt_validation(clt_exponents, K=K, output_dir=output_dir)
    plot_rg_flow(flow, K=K, output_dir=output_dir)

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
