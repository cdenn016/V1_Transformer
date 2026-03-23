#!/usr/bin/env python3
"""
Analyze whether gauge frames φ encode semantic relationships.

This module provides functions for:
1. Analyzing gauge frame semantic structure during training
2. Generating visualization plots of φ embeddings
3. Computing distance metrics between token classes

Can be used as a standalone script or imported for use during training.
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for training
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    matplotlib = None
    plt = None
    MATPLOTLIB_AVAILABLE = False

try:
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    PCA = None
    SKLEARN_AVAILABLE = False

# =============================================================================
# Tokenizer Setup
# =============================================================================

_tokenizer = None

def get_tokenizer():
    """Lazily load the GPT-2 tokenizer."""
    global _tokenizer
    if _tokenizer is None:
        try:
            import tiktoken
            _tokenizer = tiktoken.get_encoding("gpt2")
        except ImportError:
            print("WARNING: tiktoken not installed. Install with: pip install tiktoken")
            return None
    return _tokenizer


# =============================================================================
# Helper Functions
# =============================================================================

def dist(t1: int, t2: int, embed: torch.Tensor) -> float:
    """Euclidean distance between embeddings."""
    if embed is None or t1 >= len(embed) or t2 >= len(embed):
        return float('nan')
    return torch.norm(embed[t1] - embed[t2]).item()


def get_token_id(word: str) -> Optional[int]:
    """Get token ID if word is a single BPE token."""
    tokenizer = get_tokenizer()
    if tokenizer is None:
        return None
    tokens = tokenizer.encode(word)
    if len(tokens) == 1:
        return tokens[0]
    return None


def categorize_token(tid: int) -> str:
    """Categorize a token by type."""
    tokenizer = get_tokenizer()
    if tokenizer is None:
        return 'other'
    try:
        s = tokenizer.decode([tid])
        if len(s) == 1:
            if s.isalpha():
                return 'letter'
            elif s.isdigit():
                return 'digit'
            elif not s.isalnum() and not s.isspace():
                return 'punct'
        if s.strip() in {'the', 'a', 'an', 'is', 'are', 'was', 'of', 'to', 'in', 'for', 'and', 'or'}:
            return 'function'
        return 'content'
    except Exception:
        return 'other'


def format_gauge_group_label(gauge_group: str, gauge_dim: int) -> str:
    """Format gauge group config values into a display label.

    Args:
        gauge_group: Config string ('SO3', 'SON', 'GLK')
        gauge_dim: Gauge dimension (N for SO(N), K for GL(K))

    Returns:
        Human-readable label like 'SO(3)', 'SO(10)', 'GL(30)'
    """
    if gauge_group == 'SO3':
        return "SO(3)"
    elif gauge_group == 'SON':
        return f"SO({gauge_dim})"
    elif gauge_group == 'GLK':
        return f"GL({gauge_dim})"
    else:
        return f"{gauge_group}({gauge_dim})"


def identify_gauge_group(phi_dim: int) -> str:
    """Identify gauge group from phi dimension (number of generators).

    This is a fallback heuristic used when the model config is not available.
    Prefer format_gauge_group_label() with model.gauge_group when possible.
    """
    # SO(2): 1 generator, SO(3): 3 generators, SO(N): N(N-1)/2 generators
    # GL(K): K² generators
    if phi_dim == 1:
        return "SO(2)"
    elif phi_dim == 3:
        return "SO(3)"
    else:
        # Check if it's a perfect square (GL(K) has K² generators)
        k_sqrt = int(round(np.sqrt(phi_dim)))
        if k_sqrt * k_sqrt == phi_dim:
            return f"GL({k_sqrt})"
        else:
            # Solve N(N-1)/2 = phi_dim for N (SO(N))
            n_approx = int(round((1 + np.sqrt(1 + 8 * phi_dim)) / 2))
            # Verify the solution: n*(n-1)/2 should equal phi_dim
            if n_approx * (n_approx - 1) // 2 == phi_dim:
                return f"SO({n_approx})"
            else:
                return f"Gauge({phi_dim}D)"


# =============================================================================
# Analysis Functions
# =============================================================================

def analyze_token_classes(
    mu_embed: torch.Tensor,
    phi_embed: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """
    Analyze distances between token classes (letters, digits, punctuation).

    Returns metrics indicating whether embeddings show class structure.
    """
    tokenizer = get_tokenizer()
    if tokenizer is None:
        return {'error': 'tokenizer not available'}

    # Identify token classes
    letter_ids = []
    digit_ids = []
    punct_ids = []

    for tid in range(256):
        try:
            s = tokenizer.decode([tid])
            if len(s) == 1:
                if s.isalpha():
                    letter_ids.append(tid)
                elif s.isdigit():
                    digit_ids.append(tid)
                elif not s.isalnum() and not s.isspace():
                    punct_ids.append(tid)
        except Exception:
            pass

    # Compute intra-class distances (letter-letter)
    intra_mu, intra_phi = [], []
    for i, t1 in enumerate(letter_ids[:10]):
        for t2 in letter_ids[i+1:10]:
            intra_mu.append(dist(t1, t2, mu_embed))
            if phi_embed is not None:
                intra_phi.append(dist(t1, t2, phi_embed))

    # Compute inter-class distances (letter-digit, letter-punct)
    inter_mu, inter_phi = [], []
    for t1 in letter_ids[:10]:
        for t2 in digit_ids[:5] + punct_ids[:5]:
            inter_mu.append(dist(t1, t2, mu_embed))
            if phi_embed is not None:
                inter_phi.append(dist(t1, t2, phi_embed))

    # Clean NaNs
    intra_mu = [x for x in intra_mu if not np.isnan(x)]
    intra_phi = [x for x in intra_phi if not np.isnan(x)]
    inter_mu = [x for x in inter_mu if not np.isnan(x)]
    inter_phi = [x for x in inter_phi if not np.isnan(x)]

    results = {
        'n_letters': len(letter_ids),
        'n_digits': len(digit_ids),
        'n_punct': len(punct_ids),
        'mu_intra_class_dist': np.mean(intra_mu) if intra_mu else 0,
        'mu_inter_class_dist': np.mean(inter_mu) if inter_mu else 0,
        'mu_class_ratio': (np.mean(inter_mu) / max(np.mean(intra_mu), 1e-10)) if intra_mu and inter_mu else 0,
    }

    if intra_phi and inter_phi:
        results['phi_intra_class_dist'] = np.mean(intra_phi)
        results['phi_inter_class_dist'] = np.mean(inter_phi)
        results['phi_class_ratio'] = np.mean(inter_phi) / max(np.mean(intra_phi), 1e-10)
        results['phi_shows_structure'] = results['phi_class_ratio'] > 1.2

    return results


def compute_clustering_metrics(
    embed: torch.Tensor,
    n_tokens: int = 500,
    embed_name: str = 'embed',
) -> Dict[str, Any]:
    """
    Compute quantitative clustering metrics in the full-dimensional space.

    These metrics address the concern that PCA projections can create
    illusory cluster structure. All metrics are computed in the original
    high-dimensional space, not in a PCA projection.

    Metrics computed:
        - silhouette_score: Mean silhouette coefficient [-1, 1]. Measures
          how similar each token is to its own category vs. nearest other
          category. Higher = better separation. Computed in full space.
        - calinski_harabasz: Ratio of between-cluster to within-cluster
          variance. Higher = better defined clusters. Unbounded.
        - inter_intra_ratio: Ratio of mean inter-class to mean intra-class
          distance across all categorized tokens in full space.
        - anova_f_stat: Mean F-statistic from one-way ANOVA across dimensions.
          Tests whether category means differ significantly.
        - anova_p_value: Geometric mean of per-dimension ANOVA p-values.
        - pca_variance_cumulative: Number of components for 50%, 90%, 95%
          of variance, contextualizing low per-component explained variance.
        - n_tokens_per_category: Counts per category used in analysis.

    Args:
        embed: Embedding tensor [vocab_size, embed_dim]
        n_tokens: Number of tokens to analyze (first n_tokens)
        embed_name: Name prefix for result keys (e.g., 'mu' or 'phi')

    Returns:
        Dictionary with all computed metrics.
    """
    embed_np = embed[:n_tokens].numpy() if isinstance(embed, torch.Tensor) else embed[:n_tokens]
    embed_dim = embed_np.shape[1]
    results = {}

    # Categorize tokens
    categories = [categorize_token(tid) for tid in range(len(embed_np))]
    unique_cats = sorted(set(categories))

    # Need at least 2 categories with 2+ members for clustering metrics
    cat_counts = {c: categories.count(c) for c in unique_cats}
    valid_cats = [c for c, cnt in cat_counts.items() if cnt >= 2]
    results[f'{embed_name}_n_tokens_per_category'] = cat_counts

    if len(valid_cats) < 2:
        results[f'{embed_name}_clustering_error'] = 'fewer than 2 categories with 2+ members'
        return results

    # Build label array for valid categories only
    cat_to_int = {c: i for i, c in enumerate(valid_cats)}
    mask = [categories[i] in cat_to_int for i in range(len(embed_np))]
    X = embed_np[mask]
    labels = np.array([cat_to_int[categories[i]] for i in range(len(embed_np)) if mask[i]])

    # --- Silhouette Score (full-dimensional) ---
    try:
        from sklearn.metrics import silhouette_score
        sil = silhouette_score(X, labels, metric='euclidean', sample_size=min(len(X), 2000))
        results[f'{embed_name}_silhouette_score'] = float(sil)
    except Exception as e:
        results[f'{embed_name}_silhouette_score'] = f'error: {e}'

    # --- Calinski-Harabasz Index ---
    try:
        from sklearn.metrics import calinski_harabasz_score
        ch = calinski_harabasz_score(X, labels)
        results[f'{embed_name}_calinski_harabasz'] = float(ch)
    except Exception as e:
        results[f'{embed_name}_calinski_harabasz'] = f'error: {e}'

    # --- Inter/Intra class distance ratio (full space, all tokens) ---
    try:
        intra_dists = []
        inter_dists = []
        # Sample within and between categories for efficiency
        rng = np.random.RandomState(42)
        for ci, cat in enumerate(valid_cats):
            idx_cat = np.where(labels == ci)[0]
            if len(idx_cat) < 2:
                continue
            # Sample pairs within category
            n_pairs = min(200, len(idx_cat) * (len(idx_cat) - 1) // 2)
            for _ in range(n_pairs):
                i, j = rng.choice(len(idx_cat), 2, replace=False)
                intra_dists.append(np.linalg.norm(X[idx_cat[i]] - X[idx_cat[j]]))
            # Sample pairs between this category and others
            other_idx = np.where(labels != ci)[0]
            n_inter = min(200, len(idx_cat) * len(other_idx))
            for _ in range(n_inter):
                i = rng.choice(len(idx_cat))
                j = rng.choice(len(other_idx))
                inter_dists.append(np.linalg.norm(X[idx_cat[i]] - X[other_idx[j]]))

        if intra_dists and inter_dists:
            mean_intra = np.mean(intra_dists)
            mean_inter = np.mean(inter_dists)
            results[f'{embed_name}_mean_intra_dist'] = float(mean_intra)
            results[f'{embed_name}_mean_inter_dist'] = float(mean_inter)
            results[f'{embed_name}_inter_intra_ratio'] = float(mean_inter / max(mean_intra, 1e-10))
    except Exception as e:
        results[f'{embed_name}_inter_intra_ratio'] = f'error: {e}'

    # --- Per-dimension ANOVA F-test ---
    try:
        from scipy.stats import f_oneway
        f_stats = []
        p_vals = []
        # Test each dimension: do category means differ?
        groups_per_dim = {ci: X[labels == ci] for ci in range(len(valid_cats))}
        n_dims_to_test = min(embed_dim, 100)  # cap for efficiency
        dim_indices = rng.choice(embed_dim, n_dims_to_test, replace=False) if embed_dim > 100 else range(embed_dim)
        for d in dim_indices:
            groups = [groups_per_dim[ci][:, d] for ci in range(len(valid_cats)) if len(groups_per_dim[ci]) >= 2]
            if len(groups) >= 2:
                f, p = f_oneway(*groups)
                if np.isfinite(f):
                    f_stats.append(f)
                    p_vals.append(max(p, 1e-300))  # floor for log

        if f_stats:
            results[f'{embed_name}_anova_mean_f'] = float(np.mean(f_stats))
            results[f'{embed_name}_anova_median_f'] = float(np.median(f_stats))
            # Geometric mean of p-values (more robust than arithmetic for small p)
            results[f'{embed_name}_anova_geomean_p'] = float(np.exp(np.mean(np.log(p_vals))))
            results[f'{embed_name}_anova_frac_significant'] = float(np.mean([p < 0.05 for p in p_vals]))
            results[f'{embed_name}_anova_n_dims_tested'] = len(f_stats)
    except ImportError:
        results[f'{embed_name}_anova_mean_f'] = 'scipy not available'
    except Exception as e:
        results[f'{embed_name}_anova_mean_f'] = f'error: {e}'

    # --- PCA variance profile ---
    try:
        from sklearn.decomposition import PCA as PCA_full
        n_components = min(embed_dim, len(X), 50)
        pca = PCA_full(n_components=n_components)
        pca.fit(X)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        results[f'{embed_name}_pca_var_3comp'] = float(cumvar[min(2, len(cumvar)-1)])
        for threshold in [0.5, 0.9, 0.95]:
            idx = np.searchsorted(cumvar, threshold)
            if idx < len(cumvar):
                results[f'{embed_name}_pca_n_components_{int(threshold*100)}pct'] = int(idx + 1)
            else:
                results[f'{embed_name}_pca_n_components_{int(threshold*100)}pct'] = f'>{n_components}'
        results[f'{embed_name}_pca_total_components'] = embed_dim
    except Exception as e:
        results[f'{embed_name}_pca_variance_profile'] = f'error: {e}'

    return results


SEMANTIC_FIELDS = {
    'animals': ['cat', 'dog', 'bird', 'fish', 'horse'],
    'colors': ['red', 'blue', 'green', 'black', 'white'],
    'temporal': ['now', 'then', 'soon', 'once', 'never'],
    'spatial': ['up', 'down', 'left', 'right', 'near'],
    'function_words': ['the', 'a', 'an', 'is', 'of'],
    'emotions': ['happy', 'sad', 'angry', 'fear', 'love'],
    'body': ['hand', 'head', 'eye', 'heart', 'face'],
    'verbs_motion': ['run', 'walk', 'go', 'come', 'move'],
    'verbs_cognition': ['think', 'know', 'see', 'feel', 'want'],
    'morphological': [('run', 'running'), ('has', 'had'), ('big', 'bigger')],
}


def compute_semantic_field_coherence(
    embed: torch.Tensor,
    fields: Optional[Dict[str, List]] = None,
    embed_name: str = 'embed',
) -> Dict[str, Any]:
    r"""Compute within-field and between-field distances for semantic fields.

    For each semantic field, computes mean pairwise distance among field members
    (intra-field) and mean distance to tokens from other fields (inter-field).
    The ratio inter/intra > 1 indicates the field forms a coherent cluster.

    Args:
        embed: Embedding tensor [vocab_size, embed_dim].
        fields: Dict mapping field names to lists of words. Defaults to SEMANTIC_FIELDS.
        embed_name: Prefix for result keys.

    Returns:
        Dict with per-field coherence ratios and aggregate statistics.
    """
    if fields is None:
        fields = {k: v for k, v in SEMANTIC_FIELDS.items() if k != 'morphological'}

    results = {}
    field_means = {}

    for field_name, words in fields.items():
        # Resolve token IDs (skip multi-token words)
        token_ids = []
        for w in words:
            tid = get_token_id(w)
            if tid is not None and tid < len(embed):
                token_ids.append(tid)

        if len(token_ids) < 2:
            results[f'{embed_name}_{field_name}_n_tokens'] = len(token_ids)
            continue

        field_embeds = embed[token_ids]
        if isinstance(field_embeds, torch.Tensor):
            field_embeds = field_embeds.float()

        # Intra-field distances
        intra = []
        for i in range(len(token_ids)):
            for j in range(i + 1, len(token_ids)):
                d = torch.norm(field_embeds[i] - field_embeds[j]).item()
                if np.isfinite(d):
                    intra.append(d)

        results[f'{embed_name}_{field_name}_n_tokens'] = len(token_ids)
        results[f'{embed_name}_{field_name}_intra_mean'] = float(np.mean(intra)) if intra else 0.0
        field_means[field_name] = field_embeds.mean(dim=0) if isinstance(field_embeds, torch.Tensor) else torch.tensor(field_embeds).mean(dim=0)

    # Inter-field distances (between field centroids)
    field_names = list(field_means.keys())
    inter_dists = []
    for i in range(len(field_names)):
        for j in range(i + 1, len(field_names)):
            d = torch.norm(field_means[field_names[i]] - field_means[field_names[j]]).item()
            if np.isfinite(d):
                inter_dists.append(d)

    if inter_dists:
        results[f'{embed_name}_inter_field_mean'] = float(np.mean(inter_dists))

    # Aggregate coherence: mean intra / inter ratio across fields
    intra_vals = [results.get(f'{embed_name}_{fn}_intra_mean', 0) for fn in field_names]
    mean_intra = np.mean([v for v in intra_vals if v > 0]) if any(v > 0 for v in intra_vals) else 0
    if mean_intra > 0 and inter_dists:
        results[f'{embed_name}_field_coherence_ratio'] = float(np.mean(inter_dists) / mean_intra)
    else:
        results[f'{embed_name}_field_coherence_ratio'] = 0.0

    results[f'{embed_name}_n_fields_resolved'] = len(field_names)

    return results


def analyze_word_pairs(
    mu_embed: torch.Tensor,
    phi_embed: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """
    Analyze distances between related vs unrelated word pairs.
    """
    pairs = [
        ("cat", "dog", "related"),
        ("cat", "the", "unrelated"),
        ("man", "day", "unrelated"),
        ("big", "new", "unrelated"),
        ("run", "see", "related"),
        ("has", "had", "related"),
        # Expanded: within-field pairs
        ("red", "blue", "related"),
        ("up", "down", "related"),
        ("hand", "head", "related"),
        ("think", "know", "related"),
        # Expanded: cross-field unrelated
        ("red", "run", "unrelated"),
        ("hand", "blue", "unrelated"),
        ("think", "up", "unrelated"),
        ("dog", "down", "unrelated"),
    ]

    related_mu, unrelated_mu = [], []
    related_phi, unrelated_phi = [], []
    pair_results = []

    for w1, w2, rel in pairs:
        t1 = get_token_id(w1)
        t2 = get_token_id(w2)

        if t1 is None or t2 is None:
            continue

        mu_d = dist(t1, t2, mu_embed)
        phi_d = dist(t1, t2, phi_embed) if phi_embed is not None else float('nan')

        pair_results.append({
            'word1': w1,
            'word2': w2,
            'relation': rel,
            'mu_dist': mu_d,
            'phi_dist': phi_d,
        })

        if "related" in rel:
            related_mu.append(mu_d)
            if not np.isnan(phi_d):
                related_phi.append(phi_d)
        else:
            unrelated_mu.append(mu_d)
            if not np.isnan(phi_d):
                unrelated_phi.append(phi_d)

    results = {
        'pairs': pair_results,
        'mu_related_mean': np.mean(related_mu) if related_mu else 0,
        'mu_unrelated_mean': np.mean(unrelated_mu) if unrelated_mu else 0,
    }

    if related_phi and unrelated_phi:
        results['phi_related_mean'] = np.mean(related_phi)
        results['phi_unrelated_mean'] = np.mean(unrelated_phi)
        results['phi_semantic_ratio'] = np.mean(unrelated_phi) / max(np.mean(related_phi), 1e-10)

    return results


def _reconstruct_gauge_fixed_embeddings(
    embed_module: Any,
    n_tokens: Optional[int] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    r"""Reconstruct per-token μ embeddings when ``gauge_fixed_priors=True``.

    When gauge_fixed_priors is active, there is no per-token ``mu_embed``.
    Instead:

    .. math::
        \mu_i = R(\phi_i) \, \mu_0, \quad R_i = \exp(\phi_i \cdot T_a)

    where :math:`T_a` are the Lie algebra generators.

    Works with both ``GaugeTokenEmbedding`` (``base_mu``) and ``PriorBank``
    (``base_prior_mu``).

    Args:
        embed_module: ``GaugeTokenEmbedding`` or ``PriorBank`` instance.
        n_tokens: If given, only reconstruct the first *n_tokens* rows.

    Returns:
        ``(mu_embed, phi_embed)`` tensors on CPU, or ``(None, None)``
        if *embed_module* is not in gauge_fixed_priors mode.
    """
    if not getattr(embed_module, 'gauge_fixed_priors', False):
        return None, None

    # PriorBank uses base_prior_mu; GaugeTokenEmbedding uses base_mu
    base_mu = getattr(embed_module, 'base_prior_mu', None)
    if base_mu is None:
        base_mu = getattr(embed_module, 'base_mu', None)
    if base_mu is None:
        return None, None

    phi_weight = embed_module.phi_embed.weight     # (vocab, phi_dim)
    generators = embed_module.generators           # (phi_dim, K, K)

    if n_tokens is not None:
        phi_weight = phi_weight[:n_tokens]

    with torch.no_grad():
        # φ_i · T_a  →  (n, K, K)
        phi_matrix = torch.einsum('nc,ckl->nkl', phi_weight, generators)
        from transformer.core.gauge_utils import stable_matrix_exp_pair
        R, _ = stable_matrix_exp_pair(phi_matrix)   # (n, K, K)
        # μ_i = R_i @ μ_0
        mu_embed = torch.einsum('nkl,l->nk', R, base_mu)  # (n, K)

    return mu_embed.cpu(), phi_weight.detach().cpu()


def analyze_gauge_semantics(
    model: Any = None,
    mu_embed: Optional[torch.Tensor] = None,
    phi_embed: Optional[torch.Tensor] = None,
    step: Optional[int] = None,
    save_dir: Optional[Path] = None,
    save_plots: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Comprehensive gauge frame semantic analysis.

    Can be called with either a model object or directly with embeddings.

    Args:
        model: Model with mu_embed and phi_embed attributes
        mu_embed: Alternatively, provide mu embeddings directly
        phi_embed: Alternatively, provide phi embeddings directly
        step: Training step (for labeling plots)
        save_dir: Directory to save plots
        save_plots: Whether to generate and save plots
        verbose: Print analysis results

    Returns:
        Dictionary with analysis results
    """
    # Extract embeddings from model if provided.
    # Priority: prior_bank (actually trained when use_prior_bank=True)
    #         > token_embed
    #         > gauge_fixed_priors reconstruction
    if model is not None:
        pb = getattr(model, 'prior_bank', None)
        use_pb = getattr(model, 'use_prior_bank', False) and pb is not None

        # --- mu embeddings ---
        if use_pb and hasattr(pb, 'prior_mu'):
            # prior_bank.prior_mu is nn.Parameter (vocab, K), not nn.Embedding
            mu_embed = pb.prior_mu.detach().cpu()
        elif use_pb and hasattr(pb, 'base_prior_mu'):
            # gauge_fixed_priors inside prior_bank: reconstruct from base + phi
            mu_recon, _ = _reconstruct_gauge_fixed_embeddings(pb)
            if mu_recon is not None:
                mu_embed = mu_recon
        elif hasattr(model, 'mu_embed'):
            mu_embed = model.mu_embed.weight.detach().cpu()
        elif hasattr(model, 'token_embed') and hasattr(model.token_embed, 'mu_embed'):
            mu_embed = model.token_embed.mu_embed.weight.detach().cpu()

        # --- phi embeddings ---
        if use_pb and hasattr(pb, 'phi_embed'):
            phi_embed = pb.phi_embed.weight.detach().cpu()
        elif hasattr(model, 'phi_embed'):
            phi_embed = model.phi_embed.weight.detach().cpu()
        elif hasattr(model, 'token_embed') and hasattr(model.token_embed, 'phi_embed'):
            phi_embed = model.token_embed.phi_embed.weight.detach().cpu()

    # gauge_fixed_priors mode on token_embed: reconstruct μ from base_mu + phi
    if mu_embed is None and model is not None:
        te = getattr(model, 'token_embed', None)
        if te is not None:
            mu_recon, phi_recon = _reconstruct_gauge_fixed_embeddings(te)
            if mu_recon is not None:
                mu_embed = mu_recon
            if phi_embed is None and phi_recon is not None:
                phi_embed = phi_recon

    if mu_embed is None:
        return {'error': 'No mu embeddings provided'}

    # Ensure CPU tensors
    if isinstance(mu_embed, torch.Tensor) and mu_embed.device.type != 'cpu':
        mu_embed = mu_embed.detach().cpu()
    if phi_embed is not None and isinstance(phi_embed, torch.Tensor) and phi_embed.device.type != 'cpu':
        phi_embed = phi_embed.detach().cpu()

    results = {
        'step': step,
        'mu_shape': list(mu_embed.shape),
        'phi_shape': list(phi_embed.shape) if phi_embed is not None else None,
    }

    # Determine gauge group label from model config (authoritative) or phi dim (fallback)
    gauge_group_label = None
    if model is not None and hasattr(model, 'gauge_group') and hasattr(model, 'gauge_dim'):
        gauge_group_label = format_gauge_group_label(model.gauge_group, model.gauge_dim)
    if phi_embed is not None:
        if gauge_group_label is None:
            gauge_group_label = identify_gauge_group(phi_embed.shape[1])
        results['gauge_group'] = gauge_group_label

    # Token class analysis
    class_results = analyze_token_classes(mu_embed, phi_embed)
    results['token_classes'] = class_results

    # Word pair analysis
    pair_results = analyze_word_pairs(mu_embed, phi_embed)
    results['word_pairs'] = pair_results

    # Quantitative clustering metrics (full-dimensional, not PCA)
    mu_cluster = compute_clustering_metrics(mu_embed, n_tokens=500, embed_name='mu')
    results['mu_clustering'] = mu_cluster
    if phi_embed is not None:
        phi_cluster = compute_clustering_metrics(phi_embed, n_tokens=500, embed_name='phi')
        results['phi_clustering'] = phi_cluster

    # Semantic field coherence
    mu_fields = compute_semantic_field_coherence(mu_embed, embed_name='mu')
    results['mu_field_coherence'] = mu_fields
    if phi_embed is not None:
        phi_fields = compute_semantic_field_coherence(phi_embed, embed_name='phi')
        results['phi_field_coherence'] = phi_fields

    # Sigma covariance analysis (if model provides sigma parameters)
    sigma_embed = None
    if model is not None:
        # Try sigma_embed (nn.Embedding) first
        for attr_path in [('sigma_embed',), ('token_embed', 'sigma_embed')]:
            obj = model
            for a in attr_path:
                obj = getattr(obj, a, None)
                if obj is None:
                    break
            if obj is not None and hasattr(obj, 'weight'):
                sigma_embed = obj.weight.detach().cpu()
                break
        # Fall back to log_sigma_diag (nn.Parameter, common in GaugeTokenEmbedding)
        if sigma_embed is None:
            for attr_path in [('log_sigma_diag',), ('token_embed', 'log_sigma_diag')]:
                obj = model
                for a in attr_path:
                    obj = getattr(obj, a, None)
                    if obj is None:
                        break
                if obj is not None and isinstance(obj, torch.Tensor):
                    # Convert log_sigma to sigma (diagonal covariance)
                    sigma_embed = torch.exp(obj.detach().cpu()).clamp(min=1e-6)
                    break
    if sigma_embed is not None:
        sigma_results = analyze_sigma_semantics(sigma_embed, n_tokens=500, verbose=verbose)
        results['sigma_analysis'] = sigma_results

    if verbose:
        step_str = f" (step {step})" if step is not None else ""
        print(f"\n{'='*60}")
        print(f"GAUGE FRAME SEMANTIC ANALYSIS{step_str}")
        print(f"{'='*60}")
        print(f"mu shape: {results['mu_shape']}")
        if results['phi_shape']:
            print(f"phi shape: {results['phi_shape']} ({results.get('gauge_group', 'N/A')})")

        print(f"\nToken Class Analysis:")
        print(f"  mu inter/intra ratio: {class_results.get('mu_class_ratio', 0):.2f}x")
        if 'phi_class_ratio' in class_results:
            print(f"  phi inter/intra ratio: {class_results['phi_class_ratio']:.2f}x")
            if class_results.get('phi_shows_structure'):
                print(f"  --> phi DOES show class structure!")
            else:
                print(f"  --> phi does NOT show clear class structure")

        # Semantic field coherence
        mu_fc = mu_fields.get('mu_field_coherence_ratio', 0)
        print(f"\nSemantic Field Coherence:")
        print(f"  mu inter/intra field ratio: {mu_fc:.3f} ({mu_fields.get('mu_n_fields_resolved', 0)} fields)")
        if phi_embed is not None and 'phi_field_coherence' in results:
            phi_fc = results['phi_field_coherence'].get('phi_field_coherence_ratio', 0)
            print(f"  phi inter/intra field ratio: {phi_fc:.3f}")

        if 'phi_semantic_ratio' in pair_results:
            print(f"\nWord Pair Analysis:")
            print(f"  phi unrelated/related ratio: {pair_results['phi_semantic_ratio']:.2f}x")

        # Report quantitative clustering metrics
        for name, metrics in [('mu', mu_cluster)] + ([('phi', results.get('phi_clustering', {}))] if phi_embed is not None else []):
            print(f"\n  {name} Clustering Metrics (full-dimensional):")
            sil = metrics.get(f'{name}_silhouette_score')
            if isinstance(sil, float):
                print(f"    Silhouette score: {sil:.3f} (range [-1,1], higher=better)")
            ch = metrics.get(f'{name}_calinski_harabasz')
            if isinstance(ch, float):
                print(f"    Calinski-Harabasz: {ch:.1f}")
            ratio = metrics.get(f'{name}_inter_intra_ratio')
            if isinstance(ratio, float):
                print(f"    Inter/intra distance ratio: {ratio:.3f}")
            f_stat = metrics.get(f'{name}_anova_mean_f')
            if isinstance(f_stat, float):
                p_val = metrics.get(f'{name}_anova_geomean_p', 'N/A')
                frac_sig = metrics.get(f'{name}_anova_frac_significant', 0)
                print(f"    ANOVA mean F: {f_stat:.1f}, geomean p: {p_val:.2e}, {frac_sig*100:.0f}% dims significant")
            pca3 = metrics.get(f'{name}_pca_var_3comp')
            if isinstance(pca3, float):
                n50 = metrics.get(f'{name}_pca_n_components_50pct', '?')
                n90 = metrics.get(f'{name}_pca_n_components_90pct', '?')
                n95 = metrics.get(f'{name}_pca_n_components_95pct', '?')
                total = metrics.get(f'{name}_pca_total_components', '?')
                print(f"    PCA: 3 comp = {pca3*100:.1f}% var; 50%@{n50}, 90%@{n90}, 95%@{n95} of {total} dims")

    # --- Omega_i group element analysis (if model provides generators) ---
    if model is not None:
        try:
            omega_results = analyze_omega_semantics(
                model=model,
                step=step,
                save_dir=save_dir if save_plots else None,
                save_plots=save_plots,
                verbose=verbose,
                n_tokens=500,
            )
            if 'error' not in omega_results:
                results['omega'] = omega_results
        except Exception as e:
            if verbose:
                print(f"  [WARN] Omega analysis failed: {e}")

    # Generate plots
    if save_plots:
        save_dir = Path(save_dir) if save_dir else Path("./outputs/figures")
        save_dir.mkdir(parents=True, exist_ok=True)

        # Plot phi embeddings (gauge frames)
        if phi_embed is not None:
            try:
                fig = plot_embedding_clustering(
                    phi_embed,
                    embed_type='phi',
                    step=step,
                    save_path=save_dir / f"gauge_frame_clustering{'_step'+str(step) if step is not None else ''}.png",
                    gauge_group_label=gauge_group_label,
                )
                plt.close(fig)
                results['phi_plot_saved'] = True
            except Exception as e:
                if verbose:
                    print(f"  [WARN] Could not generate phi plot: {e}")
                results['phi_plot_saved'] = False

        # Plot mu embeddings (beliefs)
        if mu_embed is not None:
            try:
                fig = plot_embedding_clustering(
                    mu_embed,
                    embed_type='mu',
                    step=step,
                    save_path=save_dir / f"belief_clustering{'_step'+str(step) if step is not None else ''}.png",
                )
                plt.close(fig)
                results['mu_plot_saved'] = True
            except Exception as e:
                if verbose:
                    print(f"  [WARN] Could not generate mu plot: {e}")
                results['mu_plot_saved'] = False

    return results


# =============================================================================
# Visualization Functions
# =============================================================================

CATEGORY_COLORS = {
    'letter': '#E74C3C',    # red
    'digit': '#3498DB',     # blue
    'punct': '#2ECC71',     # green
    'function': '#9B59B6',  # purple
    'content': '#F39C12',   # orange
    'other': '#95A5A6',     # gray
}


def plot_embedding_clustering(
    embed: torch.Tensor,
    embed_type: str = 'phi',
    step: Optional[int] = None,
    save_path: Optional[Path] = None,
    n_tokens: int = 500,
    gauge_group_label: Optional[str] = None,
) -> "Any":
    """
    Visualize embeddings (mu or phi) colored by token category.

    Args:
        embed: Embedding tensor [vocab_size, embed_dim]
        embed_type: 'phi' for gauge frames, 'mu' for beliefs
        step: Training step for title
        save_path: Path to save figure
        n_tokens: Number of tokens to plot
        gauge_group_label: Override gauge group label (e.g. 'GL(30)').
                          If None and embed_type='phi', inferred from dimension.

    Returns:
        matplotlib Figure
    """
    embed_np = embed[:n_tokens].numpy() if isinstance(embed, torch.Tensor) else embed[:n_tokens]
    embed_dim = embed_np.shape[1]

    if embed_type == 'phi':
        type_str = gauge_group_label if gauge_group_label else identify_gauge_group(embed_dim)
        title_prefix = f"{type_str} Gauge Frames"
    else:
        type_str = f"{embed_dim}D"
        title_prefix = f"Belief Embeddings (μ)"

    # Categorize tokens
    categories = [categorize_token(tid) for tid in range(len(embed_np))]
    colors = [CATEGORY_COLORS.get(c, '#95A5A6') for c in categories]

    step_str = f" (Step {step})" if step is not None else ""

    if embed_dim == 1 and embed_type == 'phi':
        # SO(2): 1D gauge frames - histogram and jittered scatter
        fig = plt.figure(figsize=(14, 6))

        # Histogram
        ax1 = fig.add_subplot(121)
        for cat in CATEGORY_COLORS:
            mask = [c == cat for c in categories]
            if any(mask):
                idx = [i for i, m in enumerate(mask) if m]
                vals = embed_np[idx, 0]
                ax1.hist(vals, bins=30, alpha=0.5, label=cat, color=CATEGORY_COLORS[cat])

        ax1.set_xlabel('φ (SO(2) angle)')
        ax1.set_ylabel('Count')
        ax1.set_title(f'SO(2) Gauge Frame Distribution{step_str}')
        ax1.legend(loc='upper right', fontsize=8)

        # Jittered scatter
        ax2 = fig.add_subplot(122)
        np.random.seed(42)
        for cat in CATEGORY_COLORS:
            mask = [c == cat for c in categories]
            if any(mask):
                idx = [i for i, m in enumerate(mask) if m]
                x_vals = embed_np[idx, 0]
                y_jitter = np.random.uniform(-0.4, 0.4, len(idx))
                ax2.scatter(x_vals, y_jitter, c=CATEGORY_COLORS[cat], label=cat, alpha=0.6, s=20)

        ax2.set_xlabel('φ (SO(2) angle)')
        ax2.set_ylabel('(jittered for visibility)')
        ax2.set_title(f'SO(2) Gauge Frames by Token Type{step_str}')
        ax2.legend(loc='upper right', fontsize=8)
        ax2.grid(True, alpha=0.3, axis='x')
        ax2.set_ylim(-0.6, 0.6)

    elif embed_dim == 3 and embed_type == 'phi':
        # SO(3): Direct 3D visualization on sphere
        fig = plt.figure(figsize=(14, 6))

        # Normalize to unit sphere
        embed_norms = np.linalg.norm(embed_np, axis=1, keepdims=True)
        embed_norms = np.clip(embed_norms, 1e-8, None)
        embed_unit = embed_np / embed_norms

        # 3D sphere plot
        ax1 = fig.add_subplot(121, projection='3d')

        # Draw unit sphere wireframe
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 20)
        x_sphere = np.outer(np.cos(u), np.sin(v))
        y_sphere = np.outer(np.sin(u), np.sin(v))
        z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
        ax1.plot_wireframe(x_sphere, y_sphere, z_sphere, alpha=0.1, color='gray')

        # Plot points
        for cat in CATEGORY_COLORS:
            mask = [c == cat for c in categories]
            if any(mask):
                idx = [i for i, m in enumerate(mask) if m]
                ax1.scatter(embed_unit[idx, 0], embed_unit[idx, 1], embed_unit[idx, 2],
                           c=CATEGORY_COLORS[cat], label=cat, alpha=0.6, s=20)

        ax1.set_xlabel('φ₁')
        ax1.set_ylabel('φ₂')
        ax1.set_zlabel('φ₃')
        ax1.set_title(f'SO(3) Gauge Frames on Unit Sphere{step_str}')
        ax1.legend(loc='upper left', fontsize=8)

        # 2D projection
        ax2 = fig.add_subplot(122)
        for cat in CATEGORY_COLORS:
            mask = [c == cat for c in categories]
            if any(mask):
                idx = [i for i, m in enumerate(mask) if m]
                ax2.scatter(embed_np[idx, 0], embed_np[idx, 1],
                           c=CATEGORY_COLORS[cat], label=cat, alpha=0.6, s=20)

        ax2.set_xlabel('φ₁')
        ax2.set_ylabel('φ₂')
        ax2.set_title(f'SO(3) Gauge Frames (φ₁ vs φ₂){step_str}')
        ax2.legend(loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')

    else:
        # High-dimensional: Use PCA
        if not SKLEARN_AVAILABLE:
            print("Warning: sklearn not available for PCA visualization")
            return None
        n_components = min(3, embed_dim)
        pca = PCA(n_components=n_components)
        embed_pca = pca.fit_transform(embed_np)

        var_explained = pca.explained_variance_ratio_
        var_str = " + ".join([f"{v:.1%}" for v in var_explained])

        fig = plt.figure(figsize=(14, 6))

        if n_components >= 3:
            # 3D PCA plot
            ax1 = fig.add_subplot(121, projection='3d')
            for cat in CATEGORY_COLORS:
                mask = [c == cat for c in categories]
                if any(mask):
                    idx = [i for i, m in enumerate(mask) if m]
                    ax1.scatter(embed_pca[idx, 0], embed_pca[idx, 1], embed_pca[idx, 2],
                               c=CATEGORY_COLORS[cat], label=cat, alpha=0.6, s=20)

            ax1.set_xlabel(f'PC1 ({var_explained[0]:.1%})')
            ax1.set_ylabel(f'PC2 ({var_explained[1]:.1%})')
            ax1.set_zlabel(f'PC3 ({var_explained[2]:.1%})')
            ax1.set_title(f'{title_prefix} (PCA){step_str}')
            ax1.legend(loc='upper left', fontsize=8)

            # 2D PCA plot
            ax2 = fig.add_subplot(122)
        else:
            ax2 = fig.add_subplot(111)

        if n_components >= 2:
            for cat in CATEGORY_COLORS:
                mask = [c == cat for c in categories]
                if any(mask):
                    idx = [i for i, m in enumerate(mask) if m]
                    ax2.scatter(embed_pca[idx, 0], embed_pca[idx, 1],
                               c=CATEGORY_COLORS[cat], label=cat, alpha=0.6, s=20)

            ax2.set_xlabel(f'PC1 ({var_explained[0]:.1%})')
            ax2.set_ylabel(f'PC2 ({var_explained[1]:.1%})')
            ax2.set_title(f'{title_prefix} (PCA from {embed_dim}D){step_str}')
            ax2.legend(loc='upper left', fontsize=8)
            ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_gauge_frame_clustering(phi_embed, step=None, save_path=None, n_tokens=500, gauge_group_label=None):
    """Alias for plot_embedding_clustering with embed_type='phi'."""
    return plot_embedding_clustering(phi_embed, embed_type='phi', step=step, save_path=save_path, n_tokens=n_tokens, gauge_group_label=gauge_group_label)


# =============================================================================
# Group Element Ω_i Analysis
# =============================================================================
#
# While φ_i ∈ ℝ^{n_gen} lives in a flat Lie algebra, the group element
#   Ω_i = exp(φ_i · G)  ∈  GL⁺(K)  (or SO(K))
# lives on a curved manifold.  Semantic structure in Ω-space may differ
# from φ-space because the exponential map is nonlinear.
#
# Key metrics for Ω_i:
#   - Frobenius distance:  ‖Ω_i − Ω_j‖_F   (ambient)
#   - Geodesic distance:   ‖log(Ω_i⁻¹ Ω_j)‖_F  (intrinsic, = ‖Ω_ij − I‖ at identity)
#   - Determinant profile: det(Ω_i) ≈ 1 for SO(K), > 0 for GL⁺(K)
#   - Spectral spread:     eigenvalue distribution of Ω_i
# =============================================================================


def extract_omega(
    model: "Any",
    n_tokens: int = 500,
    device: str = 'cpu',
) -> Optional[Tuple[torch.Tensor, str]]:
    """Extract per-token group elements Ω_i = exp(φ_i · G) from a model.

    Args:
        model: GaugeTransformerLM (or any model with phi_embed + generators).
        n_tokens: Number of tokens to extract (first n_tokens from vocab).
        device: Computation device.

    Returns:
        (omega, gauge_label) where omega is (n_tokens, K, K) on CPU,
        or None if extraction fails.
    """
    # --- Try direct omega_embed first (gauge_param='omega') ---
    omega_embed = None
    omega_head_dims = None
    for attr_path in [
        ('omega_embed',),
        ('token_embed', 'omega_embed'),
        ('token_embedding', 'omega_embed'),
    ]:
        obj = model
        for a in attr_path:
            obj = getattr(obj, a, None)
            if obj is None:
                break
        if obj is not None:
            omega_embed = obj
            break

    if omega_embed is not None:
        # Direct omega path: reshape flat embedding to block-diagonal K×K
        # Locate omega_head_dims from model
        for attr in ['omega_head_dims']:
            omega_head_dims = getattr(model, attr, None)
            if omega_head_dims is not None:
                break
        if omega_head_dims is None:
            # Fallback: try to infer from embed_dim / gauge_dim
            embed_dim = getattr(model, 'embed_dim', None)
            gauge_dim = getattr(model, 'gauge_dim', None)
            if embed_dim and gauge_dim:
                n_heads = embed_dim // gauge_dim
                omega_head_dims = [gauge_dim] * n_heads

        n = min(n_tokens, omega_embed.weight.shape[0])
        omega_flat = omega_embed.weight[:n].detach().to(device)  # (n, total_params)
        K = sum(d for d in omega_head_dims) if omega_head_dims else int(omega_flat.shape[1] ** 0.5)
        omega = torch.zeros(n, K, K, device=device, dtype=omega_flat.dtype)
        if omega_head_dims is not None:
            offset = 0
            block_start = 0
            for d in omega_head_dims:
                omega_blk = omega_flat[:, offset:offset + d * d].reshape(n, d, d)
                omega[:, block_start:block_start + d, block_start:block_start + d] = omega_blk
                offset += d * d
                block_start += d
        else:
            # Single block: assume square
            d = int(omega_flat.shape[1] ** 0.5)
            omega = omega_flat.reshape(n, d, d)

        omega = omega.float().cpu()

        gauge_label = None
        if hasattr(model, 'gauge_group') and hasattr(model, 'gauge_dim'):
            gauge_label = format_gauge_group_label(model.gauge_group, model.gauge_dim)
        if gauge_label is None:
            gauge_label = f'GL({K})'

        return omega, gauge_label

    # --- Fallback: locate phi_embed and compute Ω = exp(φ·G) ---
    phi_embed = None
    for attr_path in [
        ('phi_embed',),
        ('token_embed', 'phi_embed'),
        ('token_embedding', 'phi_embed'),
    ]:
        obj = model
        for a in attr_path:
            obj = getattr(obj, a, None)
            if obj is None:
                break
        if obj is not None:
            phi_embed = obj
            break

    if phi_embed is None:
        return None

    # --- locate generators ---
    generators = getattr(model, 'generators', None)
    if generators is None:
        return None

    # --- compute Ω_i = exp(φ_i · G) ---
    n = min(n_tokens, phi_embed.weight.shape[0])
    phi = phi_embed.weight[:n].detach().to(device)   # (n, n_gen)
    G = generators.detach().to(device)                # (n_gen, K, K)

    phi_matrix = torch.einsum('na,aij->nij', phi, G)  # (n, K, K)

    # Use float64 for large K to avoid matrix_exp overflow
    K = G.shape[1]
    compute_dtype = torch.float64 if K >= 16 else torch.float32
    omega = torch.linalg.matrix_exp(phi_matrix.to(compute_dtype)).float().cpu()

    # Determine gauge group label
    gauge_label = None
    if hasattr(model, 'gauge_group') and hasattr(model, 'gauge_dim'):
        gauge_label = format_gauge_group_label(model.gauge_group, model.gauge_dim)
    if gauge_label is None:
        gauge_label = identify_gauge_group(phi.shape[1])

    return omega, gauge_label


def omega_frobenius_dist(omega: torch.Tensor, i: int, j: int) -> float:
    """Frobenius distance ‖Ω_i − Ω_j‖_F (ambient metric)."""
    return torch.norm(omega[i] - omega[j], p='fro').item()


def omega_geodesic_dist(omega: torch.Tensor, i: int, j: int) -> float:
    """Geodesic distance ‖log(Ω_i⁻¹ Ω_j)‖_F (intrinsic metric on GL(K))."""
    try:
        Oij = torch.linalg.solve(omega[i].unsqueeze(0).double(),
                                  omega[j].unsqueeze(0).double())
        log_Oij = _safe_logm(Oij.squeeze(0))
        return torch.norm(log_Oij, p='fro').float().item()
    except Exception:
        # Fallback to Frobenius if logm fails (e.g. negative eigenvalues)
        return omega_frobenius_dist(omega, i, j)


def _safe_logm(M: torch.Tensor) -> torch.Tensor:
    """Matrix logarithm via eigendecomposition, handling complex eigenvalues."""
    evals, evecs = torch.linalg.eig(M)
    log_evals = torch.log(evals)  # complex log
    return (evecs @ torch.diag(log_evals) @ torch.linalg.inv(evecs)).real.float()


# =============================================================================
# Omega Clustering Metrics
# =============================================================================

def compute_omega_clustering_metrics(
    omega: torch.Tensor,
    n_tokens: int = 500,
) -> Dict[str, Any]:
    """Compute clustering quality of Ω_i in matrix space.

    Flattens each K×K matrix to a vector for sklearn metrics, but also
    computes group-specific metrics (det profile, spectral spread).

    Args:
        omega: Group elements (n_tokens, K, K).
        n_tokens: Limit analysis to first n tokens.

    Returns:
        Dict of clustering and group-geometry metrics.
    """
    n = min(n_tokens, omega.shape[0])
    K = omega.shape[1]
    omega_flat = omega[:n].reshape(n, -1).numpy()  # (n, K²)
    results = {}

    # --- Frobenius-space clustering (same machinery as phi) ---
    frob_metrics = compute_clustering_metrics(
        torch.from_numpy(omega_flat), n_tokens=n, embed_name='omega_frob',
    )
    results.update(frob_metrics)

    # --- Geodesic pairwise distances (sampled) for inter/intra class ---
    categories = [categorize_token(tid) for tid in range(n)]
    cat_counts = {}
    for c in categories:
        cat_counts[c] = cat_counts.get(c, 0) + 1
    valid_cats = sorted(c for c, cnt in cat_counts.items() if cnt >= 2)

    if len(valid_cats) >= 2:
        rng = np.random.RandomState(42)
        cat_to_idx = {}
        for i, c in enumerate(categories):
            cat_to_idx.setdefault(c, []).append(i)

        intra_geo, inter_geo = [], []
        intra_frob, inter_frob = [], []
        for cat in valid_cats:
            idx = cat_to_idx[cat]
            n_pairs = min(100, len(idx) * (len(idx) - 1) // 2)
            for _ in range(n_pairs):
                i, j = rng.choice(len(idx), 2, replace=False)
                intra_geo.append(omega_geodesic_dist(omega[:n], idx[i], idx[j]))
                intra_frob.append(omega_frobenius_dist(omega[:n], idx[i], idx[j]))
            other_idx = [i for i, c in enumerate(categories) if c != cat and c in valid_cats]
            for _ in range(min(100, len(idx) * len(other_idx))):
                i = rng.choice(idx)
                j = rng.choice(other_idx)
                inter_geo.append(omega_geodesic_dist(omega[:n], i, j))
                inter_frob.append(omega_frobenius_dist(omega[:n], i, j))

        if intra_geo and inter_geo:
            results['omega_geodesic_intra_mean'] = float(np.mean(intra_geo))
            results['omega_geodesic_inter_mean'] = float(np.mean(inter_geo))
            results['omega_geodesic_ratio'] = float(
                np.mean(inter_geo) / max(np.mean(intra_geo), 1e-10)
            )
            results['omega_frobenius_intra_mean'] = float(np.mean(intra_frob))
            results['omega_frobenius_inter_mean'] = float(np.mean(inter_frob))
            results['omega_frobenius_ratio'] = float(
                np.mean(inter_frob) / max(np.mean(intra_frob), 1e-10)
            )

    # --- Determinant profile ---
    dets = torch.det(omega[:n].double()).float().numpy()
    results['omega_det_mean'] = float(np.mean(dets))
    results['omega_det_std'] = float(np.std(dets))
    results['omega_det_min'] = float(np.min(dets))
    results['omega_det_max'] = float(np.max(dets))
    # For SO(K), det ≈ 1; deviation indicates GL drift
    results['omega_det_deviation'] = float(np.mean(np.abs(dets - 1.0)))

    # --- Spectral spread: eigenvalue distribution ---
    try:
        evals = torch.linalg.eigvals(omega[:n].double()).abs().float().numpy()  # (n, K)
        # Spread = max/min eigenvalue magnitude ratio (condition number proxy)
        evals_sorted = np.sort(evals, axis=1)  # ascending
        evals_max = evals_sorted[:, -1]
        evals_min = np.clip(evals_sorted[:, 0], 1e-10, None)
        cond = evals_max / evals_min
        results['omega_spectral_spread_mean'] = float(np.mean(cond))
        results['omega_spectral_spread_std'] = float(np.std(cond))
        # For SO(K), all eigenvalues have |λ| = 1
        results['omega_spectral_unit_deviation'] = float(np.mean(np.abs(evals - 1.0)))
    except Exception as e:
        results['omega_spectral_error'] = str(e)

    # --- Distance from identity: ‖Ω_i − I‖_F ---
    eye = torch.eye(K).unsqueeze(0).expand(n, K, K)
    dist_from_id = torch.norm((omega[:n] - eye).reshape(n, -1), dim=1).numpy()
    results['omega_identity_dist_mean'] = float(np.mean(dist_from_id))
    results['omega_identity_dist_std'] = float(np.std(dist_from_id))

    return results


# =============================================================================
# Main Omega Semantic Analysis Entry Point
# =============================================================================

def analyze_omega_semantics(
    model: "Any" = None,
    omega: Optional[torch.Tensor] = None,
    step: Optional[int] = None,
    save_dir: Optional[Path] = None,
    save_plots: bool = True,
    verbose: bool = True,
    n_tokens: int = 500,
) -> Dict[str, Any]:
    """Comprehensive semantic analysis of per-token group elements Ω_i.

    Parallels analyze_gauge_semantics() but operates on Ω_i = exp(φ_i · G)
    rather than the raw Lie algebra coefficients φ_i.  Because Ω lives on a
    curved group manifold, we report both ambient (Frobenius) and intrinsic
    (geodesic) distances, plus group-specific diagnostics (determinant, spectrum).

    Args:
        model: GaugeTransformerLM (extracts phi_embed + generators → Ω).
        omega: Alternatively, provide pre-computed (n_tokens, K, K) tensor.
        step: Training step for plot titles.
        save_dir: Directory for saved figures.
        save_plots: Whether to generate figures.
        verbose: Print analysis to console.
        n_tokens: Number of tokens to analyze.

    Returns:
        Dict with all Omega semantic metrics.
    """
    gauge_label = None

    # Extract Omega from model if not provided
    if omega is None and model is not None:
        result = extract_omega(model, n_tokens=n_tokens)
        if result is None:
            return {'error': 'Could not extract Omega from model (no phi_embed or generators)'}
        omega, gauge_label = result
    elif omega is not None:
        omega = omega.detach().cpu().float() if isinstance(omega, torch.Tensor) else omega
    else:
        return {'error': 'Must provide either model or omega tensor'}

    n = min(n_tokens, omega.shape[0])
    K = omega.shape[1]

    results = {
        'step': step,
        'omega_shape': list(omega[:n].shape),
        'K': K,
        'gauge_group': gauge_label,
    }

    # Quantitative clustering metrics (Frobenius + geodesic + spectral)
    cluster_metrics = compute_omega_clustering_metrics(omega, n_tokens=n)
    results['omega_clustering'] = cluster_metrics

    # Token class analysis (using Frobenius distance on Ω)
    omega_flat_embed = omega[:n].reshape(n, -1)  # (n, K²)
    class_results = analyze_token_classes(
        mu_embed=omega_flat_embed,  # reuse machinery with flattened Ω
        phi_embed=None,
    )
    # Rename keys: mu_ → omega_ since we passed Ω as "mu"
    omega_class = {}
    for k, v in class_results.items():
        new_k = k.replace('mu_', 'omega_') if k.startswith('mu_') else k
        omega_class[new_k] = v
    results['token_classes'] = omega_class

    # Word pair analysis (Frobenius distances between Ω_i)
    pair_results = analyze_word_pairs(mu_embed=omega_flat_embed, phi_embed=None)
    omega_pairs = {}
    for k, v in pair_results.items():
        new_k = k.replace('mu_', 'omega_') if k.startswith('mu_') else k
        omega_pairs[new_k] = v
    results['word_pairs'] = omega_pairs

    if verbose:
        step_str = f" (step {step})" if step is not None else ""
        print(f"\n{'='*60}")
        print(f"GROUP ELEMENT Omega_i SEMANTIC ANALYSIS{step_str}")
        print(f"{'='*60}")
        print(f"Omega shape: {results['omega_shape']}  (K={K}, gauge={gauge_label})")

        # Class structure
        cr = omega_class.get('omega_class_ratio', 0)
        print(f"\nToken Class Analysis (Frobenius distance on Omega):")
        print(f"  inter/intra ratio: {cr:.2f}x {'<-- structure!' if cr > 1.2 else ''}")

        # Geodesic vs Frobenius
        cm = cluster_metrics
        geo_r = cm.get('omega_geodesic_ratio')
        frob_r = cm.get('omega_frobenius_ratio')
        if geo_r is not None:
            print(f"\nDistance Ratios (inter-class / intra-class):")
            print(f"  Frobenius: {frob_r:.3f}")
            print(f"  Geodesic:  {geo_r:.3f}")

        # Determinant profile
        print(f"\nDeterminant Profile:")
        print(f"  mean: {cm.get('omega_det_mean', 0):.4f}  "
              f"std: {cm.get('omega_det_std', 0):.4f}  "
              f"range: [{cm.get('omega_det_min', 0):.4f}, {cm.get('omega_det_max', 0):.4f}]")
        print(f"  |det - 1| mean: {cm.get('omega_det_deviation', 0):.4f} "
              f"{'(SO(K): should be ~0)' if gauge_label and 'SO' in str(gauge_label) else ''}")

        # Spectral
        spread = cm.get('omega_spectral_spread_mean')
        if spread is not None:
            print(f"\nSpectral Analysis:")
            print(f"  Condition number (|λ_max/λ_min|): {spread:.3f} +/- {cm.get('omega_spectral_spread_std', 0):.3f}")
            print(f"  Unit deviation (|λ| - 1): {cm.get('omega_spectral_unit_deviation', 0):.4f}")

        # Identity distance
        print(f"\nDistance from Identity:")
        print(f"  mean ‖Omega_i - I‖_F: {cm.get('omega_identity_dist_mean', 0):.4f} "
              f"+/- {cm.get('omega_identity_dist_std', 0):.4f}")

        # Clustering quality
        sil = cm.get('omega_frob_silhouette_score')
        if isinstance(sil, float):
            print(f"\nClustering Quality (Frobenius-flattened Omega):")
            print(f"  Silhouette: {sil:.3f}")
            ch = cm.get('omega_frob_calinski_harabasz')
            if isinstance(ch, float):
                print(f"  Calinski-Harabasz: {ch:.1f}")

    # Generate plots
    if save_plots:
        save_dir = Path(save_dir) if save_dir else Path("./outputs/figures")
        save_dir.mkdir(parents=True, exist_ok=True)

        try:
            fig = plot_omega_clustering(
                omega[:n], step=step,
                save_path=save_dir / f"omega_clustering{'_step'+str(step) if step is not None else ''}.png",
                gauge_group_label=gauge_label,
                n_tokens=n,
            )
            if fig is not None:
                plt.close(fig)
                results['omega_plot_saved'] = True
        except Exception as e:
            if verbose:
                print(f"  [WARN] Could not generate omega plot: {e}")
            results['omega_plot_saved'] = False

    return results


# =============================================================================
# Omega Visualization
# =============================================================================

def plot_omega_clustering(
    omega: torch.Tensor,
    step: Optional[int] = None,
    save_path: Optional[Path] = None,
    n_tokens: int = 500,
    gauge_group_label: Optional[str] = None,
) -> Optional["Any"]:
    """Visualize per-token group elements Ω_i colored by token category.

    Layout (4 panels):
      (a) PCA of flattened Ω_i (K² → 2D)  — cluster structure
      (b) Determinant distribution by category  — group geometry
      (c) Distance from identity ‖Ω_i − I‖_F by category  — magnitude
      (d) Eigenvalue magnitude distribution   — spectral character

    Args:
        omega: Group elements (n_tokens, K, K).
        step: Training step for title.
        save_path: Output file path.
        n_tokens: Tokens to visualize.
        gauge_group_label: e.g. 'SO(3)' or 'GL(30)'.

    Returns:
        matplotlib Figure (or None if matplotlib unavailable).
    """
    if not MATPLOTLIB_AVAILABLE or not SKLEARN_AVAILABLE:
        return None

    n = min(n_tokens, omega.shape[0])
    K = omega.shape[1]
    omega_np = omega[:n].numpy() if isinstance(omega, torch.Tensor) else omega[:n]

    categories = [categorize_token(tid) for tid in range(n)]
    colors = [CATEGORY_COLORS.get(c, '#95A5A6') for c in categories]

    step_str = f" (Step {step})" if step is not None else ""
    group_str = gauge_group_label or f"K={K}"

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Group Element Ω_i Analysis — {group_str}{step_str}', fontsize=14, y=1.01)

    # ---- (a) PCA of flattened Omega ----
    ax = axes[0, 0]
    omega_flat = omega_np.reshape(n, -1)  # (n, K²)
    pca = PCA(n_components=min(3, omega_flat.shape[1]))
    omega_pca = pca.fit_transform(omega_flat)
    var = pca.explained_variance_ratio_

    for cat in CATEGORY_COLORS:
        mask = [c == cat for c in categories]
        if any(mask):
            idx = [i for i, m in enumerate(mask) if m]
            ax.scatter(omega_pca[idx, 0], omega_pca[idx, 1],
                       c=CATEGORY_COLORS[cat], label=cat, alpha=0.6, s=20)
    ax.set_xlabel(f'PC1 ({var[0]:.1%})')
    ax.set_ylabel(f'PC2 ({var[1]:.1%})')
    ax.set_title(f'(a) Ω_i PCA (from {K}×{K} matrices)')
    ax.legend(loc='upper right', fontsize=7)
    ax.grid(True, alpha=0.3)

    # ---- (b) Determinant distribution by category ----
    ax = axes[0, 1]
    dets = np.linalg.det(omega_np.astype(np.float64)).astype(np.float32)
    for cat in CATEGORY_COLORS:
        mask = [c == cat for c in categories]
        if any(mask):
            idx = [i for i, m in enumerate(mask) if m]
            vals = dets[idx]
            ax.hist(vals, bins=30, alpha=0.5, label=cat, color=CATEGORY_COLORS[cat])
    ax.axvline(x=1.0, color='k', linestyle='--', linewidth=1, label='det=1')
    ax.set_xlabel('det(Ω_i)')
    ax.set_ylabel('Count')
    ax.set_title('(b) Determinant Distribution')
    ax.legend(loc='upper right', fontsize=7)

    # ---- (c) Distance from identity by category ----
    ax = axes[1, 0]
    eye = np.eye(K, dtype=np.float32)
    dist_id = np.linalg.norm((omega_np - eye).reshape(n, -1), axis=1)
    cat_list = sorted(set(categories), key=lambda c: list(CATEGORY_COLORS.keys()).index(c) if c in CATEGORY_COLORS else 99)
    cat_data = []
    cat_labels = []
    cat_colors_bp = []
    for cat in cat_list:
        idx = [i for i, c in enumerate(categories) if c == cat]
        if idx:
            cat_data.append(dist_id[idx])
            cat_labels.append(cat)
            cat_colors_bp.append(CATEGORY_COLORS.get(cat, '#95A5A6'))

    bp = ax.boxplot(cat_data, labels=cat_labels, patch_artist=True, showfliers=False)
    for patch, color in zip(bp['boxes'], cat_colors_bp):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax.set_ylabel('‖Ω_i − I‖_F')
    ax.set_title('(c) Distance from Identity by Category')
    ax.grid(True, alpha=0.3, axis='y')

    # ---- (d) Eigenvalue magnitude spectrum ----
    ax = axes[1, 1]
    try:
        evals = np.abs(np.linalg.eigvals(omega_np.astype(np.float64))).astype(np.float32)
        # Histogram of all eigenvalue magnitudes
        for cat in CATEGORY_COLORS:
            mask = [c == cat for c in categories]
            if any(mask):
                idx = [i for i, m in enumerate(mask) if m]
                ax.hist(evals[idx].flatten(), bins=40, alpha=0.4,
                        label=cat, color=CATEGORY_COLORS[cat], density=True)
        ax.axvline(x=1.0, color='k', linestyle='--', linewidth=1, label='|λ|=1')
        ax.set_xlabel('|λ|  (eigenvalue magnitude)')
        ax.set_ylabel('Density')
        ax.set_title('(d) Eigenvalue Magnitude Spectrum')
        ax.legend(loc='upper right', fontsize=7)
    except Exception:
        ax.text(0.5, 0.5, 'Eigenvalue computation failed',
                transform=ax.transAxes, ha='center', va='center')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_omega_group_clustering(omega, step=None, save_path=None, n_tokens=500, gauge_group_label=None):
    """Alias for plot_omega_clustering (matches plot_gauge_frame_clustering pattern)."""
    return plot_omega_clustering(omega, step=step, save_path=save_path, n_tokens=n_tokens, gauge_group_label=gauge_group_label)


# =============================================================================
# Sigma Covariance Semantic Analysis
# =============================================================================

def analyze_sigma_semantics(
    sigma: torch.Tensor,
    n_tokens: int = 500,
    verbose: bool = True,
) -> Dict[str, Any]:
    r"""Analyze semantic structure in covariance embeddings Sigma.

    The covariance Sigma encodes uncertainty in beliefs. Tokens with similar
    uncertainty profiles may share representational roles (e.g., function words
    having high uncertainty vs content words with low uncertainty).

    For diagonal Sigma: analyzes the K-dimensional uncertainty vector directly.
    For full Sigma (K x K): analyzes both the diagonal and the Frobenius-flattened
    matrix, plus computes effective rank = exp(entropy of normalized eigenvalues).

    Args:
        sigma: Covariance tensor, either [vocab, K] (diagonal) or [vocab, K, K] (full).
        n_tokens: Number of tokens to analyze.
        verbose: Print results to console.

    Returns:
        Dict with clustering metrics, effective rank, and uncertainty profile.
    """
    n = min(n_tokens, sigma.shape[0])
    sigma_sub = sigma[:n].detach().cpu().float()
    results = {}
    is_full = sigma_sub.dim() == 3  # (n, K, K) vs (n, K)

    if is_full:
        K = sigma_sub.shape[1]
        results['sigma_mode'] = 'full'
        results['sigma_K'] = K

        # Diagonal uncertainty profile
        sigma_diag = torch.diagonal(sigma_sub, dim1=-2, dim2=-1)  # (n, K)

        # Effective rank per token: exp(H(normalized eigenvalues))
        try:
            evals = torch.linalg.eigvalsh(sigma_sub)  # (n, K), real since SPD
            evals = evals.clamp(min=1e-10)
            evals_norm = evals / evals.sum(dim=-1, keepdim=True)
            entropy = -(evals_norm * torch.log(evals_norm)).sum(dim=-1)  # (n,)
            eff_rank = torch.exp(entropy)
            results['sigma_effective_rank_mean'] = float(eff_rank.mean())
            results['sigma_effective_rank_std'] = float(eff_rank.std())

            # Per-category effective rank
            categories = [categorize_token(tid) for tid in range(n)]
            for cat in sorted(set(categories)):
                idx = [i for i, c in enumerate(categories) if c == cat]
                if len(idx) >= 2:
                    results[f'sigma_eff_rank_{cat}'] = float(eff_rank[idx].mean())
        except Exception as e:
            results['sigma_effective_rank_error'] = str(e)

        # Frobenius-space clustering on full matrix
        sigma_flat = sigma_sub.reshape(n, -1)
        full_metrics = compute_clustering_metrics(sigma_flat, n_tokens=n, embed_name='sigma_full')
        results.update(full_metrics)
    else:
        results['sigma_mode'] = 'diagonal'
        sigma_diag = sigma_sub  # (n, K)

    # Diagonal clustering metrics
    diag_metrics = compute_clustering_metrics(sigma_diag, n_tokens=n, embed_name='sigma_diag')
    results.update(diag_metrics)

    # Uncertainty magnitude by token category
    categories = [categorize_token(tid) for tid in range(min(n, len(sigma_diag)))]
    diag_np = sigma_diag.numpy()
    trace_per_token = diag_np.sum(axis=-1)  # total uncertainty

    for cat in sorted(set(categories)):
        idx = [i for i, c in enumerate(categories) if c == cat]
        if idx:
            results[f'sigma_trace_{cat}_mean'] = float(np.mean(trace_per_token[idx]))
            results[f'sigma_trace_{cat}_std'] = float(np.std(trace_per_token[idx]))

    # ANOVA on trace across categories
    try:
        from scipy.stats import f_oneway
        groups = []
        for cat in sorted(set(categories)):
            idx = [i for i, c in enumerate(categories) if c == cat]
            if len(idx) >= 2:
                groups.append(trace_per_token[idx])
        if len(groups) >= 2:
            f_stat, p_val = f_oneway(*groups)
            results['sigma_trace_anova_f'] = float(f_stat)
            results['sigma_trace_anova_p'] = float(p_val)
    except ImportError:
        pass

    if verbose:
        print(f"\n  Sigma Covariance Analysis ({results.get('sigma_mode', '?')}):")
        sil = results.get('sigma_diag_silhouette_score')
        if isinstance(sil, float):
            print(f"    Diagonal silhouette: {sil:.3f}")
        eff = results.get('sigma_effective_rank_mean')
        if eff is not None:
            print(f"    Effective rank: {eff:.2f} +/- {results.get('sigma_effective_rank_std', 0):.2f}")
        f_stat = results.get('sigma_trace_anova_f')
        if f_stat is not None:
            print(f"    Trace ANOVA F={f_stat:.1f}, p={results.get('sigma_trace_anova_p', 1):.2e}")

    return results


# =============================================================================
# Per-Layer Belief Evolution Analysis
# =============================================================================

def analyze_per_layer_semantics(
    model: "Any",
    token_ids: torch.Tensor,
    targets: Optional[torch.Tensor] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    r"""Analyze how semantic structure evolves through transformer layers.

    Runs a forward pass with attention tracking and extracts per-layer beliefs
    (mu, sigma, phi). For each layer, computes:
      - Token class inter/intra distance ratio on mu
      - Semantic field coherence on mu
      - Sigma effective rank (if full covariance)
      - Delta norms (how much each layer changes beliefs)

    This reveals whether semantic structure emerges gradually across layers
    or appears suddenly, and whether deeper layers refine or diffuse clusters.

    Args:
        model: GaugeTransformerLM with forward_with_attention method.
        token_ids: (batch, seq_len) input tokens.
        targets: Optional (batch, seq_len) target tokens.
        verbose: Print per-layer summary.

    Returns:
        Dict with per-layer semantic metrics and evolution summary.
    """
    if not hasattr(model, 'forward_with_attention'):
        return {'error': 'Model lacks forward_with_attention method'}

    device = next(model.parameters()).device
    token_ids = token_ids.to(device)
    if targets is not None:
        targets = targets.to(device)

    # Enable layer diagnostics to capture per-layer states
    model._collect_layer_diagnostics = True
    model._layer_diagnostics = []

    with torch.no_grad():
        logits, attn_info = model.forward_with_attention(token_ids, targets)

    layer_diags = model._layer_diagnostics
    model._collect_layer_diagnostics = False

    results = {
        'n_layers': attn_info.get('n_layers', len(layer_diags)),
        'layers': [],
    }

    # Analyze final-layer beliefs using attention_info
    mu_final = attn_info.get('mu')
    sigma_final = attn_info.get('sigma')
    phi_final = attn_info.get('phi')
    mu_prior = attn_info.get('mu_prior')

    if mu_final is not None and mu_prior is not None:
        # Compute per-position belief drift from embedding to final
        delta = (mu_final - mu_prior).detach().cpu()
        results['total_belief_drift_norm'] = float(delta.norm().item())
        results['per_position_drift_mean'] = float(delta.norm(dim=-1).mean().item())

    # Per-layer diagnostics from the model's collection
    for ld in layer_diags:
        layer_info = {
            'layer': ld['layer'],
            'delta_mu_norm': ld.get('delta_mu_norm', 0),
            'delta_mu_relative': ld.get('delta_mu_relative', 0),
            'sigma_mean_diag': ld.get('sigma_mean_diag', 0),
            'attention_entropy': ld.get('attention_entropy', None),
            'kl_mean': ld.get('kl_mean', None),
        }
        if 'ce_loss' in ld:
            layer_info['ce_loss'] = ld['ce_loss']
            layer_info['perplexity'] = ld.get('perplexity')
        results['layers'].append(layer_info)

    # Summarize evolution pattern
    if len(layer_diags) >= 2:
        deltas = [ld.get('delta_mu_relative', 0) for ld in layer_diags]
        results['delta_mu_trend'] = 'decreasing' if deltas[-1] < deltas[0] else 'increasing'
        results['max_delta_layer'] = int(np.argmax(deltas))

        if all('ce_loss' in ld for ld in layer_diags):
            ce_losses = [ld['ce_loss'] for ld in layer_diags]
            results['ce_improvement_per_layer'] = [
                ce_losses[i] - ce_losses[i + 1] for i in range(len(ce_losses) - 1)
            ]

    if verbose:
        print(f"\n  Per-Layer Belief Evolution ({len(layer_diags)} layers):")
        for ld in layer_diags:
            ce_str = f", CE={ld.get('ce_loss', 0):.3f}" if 'ce_loss' in ld else ""
            ent_str = f", H_attn={ld.get('attention_entropy', 0):.2f}" if ld.get('attention_entropy') else ""
            print(f"    L{ld['layer']}: Δμ_rel={ld.get('delta_mu_relative', 0):.4f}, "
                  f"σ_mean={ld.get('sigma_mean_diag', 0):.4f}{ent_str}{ce_str}")

    return results


# =============================================================================
# Holonomy-Semantic Correlation
# =============================================================================

def analyze_holonomy_semantic_correlation(
    model: "Any",
    n_tokens: int = 200,
    n_triangles: int = 100,
    verbose: bool = True,
) -> Dict[str, Any]:
    r"""Analyze whether holonomy (transport curvature) correlates with semantics.

    Gauge-theoretic prediction: tokens in the same semantic field should have
    smaller holonomy deficits when transported around triangles within the field
    vs triangles crossing field boundaries. Flat transport within a field means
    the gauge connection respects semantic coherence.

    Computes holonomy for within-field and cross-field token triangles, then
    tests whether the difference is significant.

    Args:
        model: GaugeTransformerLM with phi_embed and generators.
        n_tokens: Number of tokens to consider.
        n_triangles: Number of random triangles to sample per condition.
        verbose: Print results.

    Returns:
        Dict with within-field vs cross-field holonomy statistics.
    """
    from .holonomy import compute_holonomy

    # Extract omega
    result = extract_omega(model, n_tokens=n_tokens)
    if result is None:
        return {'error': 'Could not extract Omega from model'}

    omega, gauge_label = result
    n = omega.shape[0]

    # Build field membership
    fields = {k: v for k, v in SEMANTIC_FIELDS.items() if k != 'morphological'}
    token_to_field: Dict[int, str] = {}
    field_to_tokens: Dict[str, List[int]] = {}

    for field_name, words in fields.items():
        tids = []
        for w in words:
            tid = get_token_id(w)
            if tid is not None and tid < n:
                tids.append(tid)
                token_to_field[tid] = field_name
        if len(tids) >= 2:
            field_to_tokens[field_name] = tids

    if len(field_to_tokens) < 2:
        return {'error': 'Not enough resolved semantic fields for holonomy analysis'}

    rng = np.random.RandomState(42)

    def sample_holonomy_deficit(i: int, j: int, k: int) -> float:
        """Compute holonomy deficit for triangle (i, j, k)."""
        # Ω_ij Ω_jk Ω_ki should = I for flat connection
        try:
            Oij = torch.linalg.solve(omega[i].unsqueeze(0).double(),
                                      omega[j].unsqueeze(0).double()).squeeze(0)
            Ojk = torch.linalg.solve(omega[j].unsqueeze(0).double(),
                                      omega[k].unsqueeze(0).double()).squeeze(0)
            Oki = torch.linalg.solve(omega[k].unsqueeze(0).double(),
                                      omega[i].unsqueeze(0).double()).squeeze(0)
            holonomy = Oij @ Ojk @ Oki
            deficit = torch.norm(holonomy - torch.eye(holonomy.shape[0], dtype=torch.float64), p='fro').item()
            return deficit
        except Exception:
            return float('nan')

    # Within-field holonomy
    within_deficits = []
    all_field_tokens = []
    for fn, tids in field_to_tokens.items():
        all_field_tokens.extend(tids)
        if len(tids) >= 3:
            for _ in range(min(n_triangles // len(field_to_tokens), 50)):
                tri = rng.choice(tids, 3, replace=False)
                d = sample_holonomy_deficit(tri[0], tri[1], tri[2])
                if np.isfinite(d):
                    within_deficits.append(d)

    # Cross-field holonomy
    cross_deficits = []
    field_names = list(field_to_tokens.keys())
    for _ in range(n_triangles):
        # Pick tokens from 2-3 different fields
        chosen_fields = rng.choice(field_names, min(3, len(field_names)), replace=False)
        tri = []
        for fn in chosen_fields:
            tri.append(rng.choice(field_to_tokens[fn]))
        if len(tri) == 2:
            # Need a third token from any field
            remaining = [t for t in all_field_tokens if t not in tri]
            if remaining:
                tri.append(rng.choice(remaining))
        if len(tri) >= 3:
            d = sample_holonomy_deficit(tri[0], tri[1], tri[2])
            if np.isfinite(d):
                cross_deficits.append(d)

    results = {
        'gauge_group': gauge_label,
        'n_fields': len(field_to_tokens),
        'n_within_triangles': len(within_deficits),
        'n_cross_triangles': len(cross_deficits),
    }

    if within_deficits:
        results['within_field_deficit_mean'] = float(np.mean(within_deficits))
        results['within_field_deficit_std'] = float(np.std(within_deficits))
    if cross_deficits:
        results['cross_field_deficit_mean'] = float(np.mean(cross_deficits))
        results['cross_field_deficit_std'] = float(np.std(cross_deficits))

    if within_deficits and cross_deficits:
        results['deficit_ratio'] = float(
            np.mean(cross_deficits) / max(np.mean(within_deficits), 1e-10)
        )
        # Statistical test
        try:
            from scipy.stats import mannwhitneyu
            stat, p = mannwhitneyu(within_deficits, cross_deficits, alternative='less')
            results['mannwhitney_stat'] = float(stat)
            results['mannwhitney_p'] = float(p)
        except ImportError:
            pass

    if verbose:
        print(f"\n  Holonomy-Semantic Correlation ({gauge_label}):")
        print(f"    Fields resolved: {len(field_to_tokens)}")
        if within_deficits:
            print(f"    Within-field deficit: {results.get('within_field_deficit_mean', 0):.4f} "
                  f"+/- {results.get('within_field_deficit_std', 0):.4f}")
        if cross_deficits:
            print(f"    Cross-field deficit:  {results.get('cross_field_deficit_mean', 0):.4f} "
                  f"+/- {results.get('cross_field_deficit_std', 0):.4f}")
        ratio = results.get('deficit_ratio')
        if ratio is not None:
            print(f"    Ratio (cross/within): {ratio:.3f} {'(>1 = fields are flatter)' if ratio > 1 else ''}")
            p = results.get('mannwhitney_p')
            if p is not None:
                print(f"    Mann-Whitney p={p:.4e}")

    return results


# =============================================================================
# Semantic Trajectory Tracker (training-time)
# =============================================================================

class SemanticTrajectoryTracker:
    """Track evolution of semantic metrics across training steps.

    Records periodic snapshots of embedding-level semantic structure
    (clustering quality, field coherence, uncertainty profile) to detect
    when and how semantic organization emerges during training.

    Usage::

        tracker = SemanticTrajectoryTracker()
        # During training loop:
        if tracker.should_record(step):
            tracker.record(model, step)
        # At end:
        tracker.save(path)
        summary = tracker.summarize()
    """

    def __init__(self, interval: int = 5000):
        self.interval = interval
        self.snapshots: List[Dict[str, Any]] = []

    def should_record(self, step: int) -> bool:
        """Check if a snapshot should be recorded at this step."""
        return self.interval > 0 and step % self.interval == 0

    def record(
        self,
        model: "Any",
        step: int,
        n_tokens: int = 500,
    ) -> Dict[str, Any]:
        """Record a semantic snapshot at the current training step.

        Args:
            model: GaugeTransformerLM.
            step: Current training step.
            n_tokens: Number of tokens to analyze.

        Returns:
            Snapshot dict with all computed metrics.
        """
        snapshot = {'step': step}

        # Extract embeddings — prefer prior_bank (trained) over token_embed (dead)
        mu_embed = None
        phi_embed = None
        sigma_embed = None

        pb = getattr(model, 'prior_bank', None)
        use_pb = getattr(model, 'use_prior_bank', False) and pb is not None

        if use_pb:
            if hasattr(pb, 'prior_mu'):
                mu_embed = pb.prior_mu.detach().cpu()[:n_tokens]
            elif hasattr(pb, 'base_prior_mu'):
                mu_recon, phi_recon = _reconstruct_gauge_fixed_embeddings(
                    pb, n_tokens=n_tokens,
                )
                if mu_recon is not None:
                    mu_embed = mu_recon
                if phi_recon is not None:
                    phi_embed = phi_recon
            if hasattr(pb, 'phi_embed'):
                phi_embed = pb.phi_embed.weight.detach().cpu()[:n_tokens]
            if hasattr(pb, 'log_prior_sigma') and isinstance(pb.log_prior_sigma, torch.nn.Parameter):
                sigma_embed = torch.exp(pb.log_prior_sigma).detach().cpu()[:n_tokens]

        if mu_embed is None:
            for attr_path in [('token_embed',), ()]:
                obj = model
                for a in attr_path:
                    obj = getattr(obj, a, None)
                    if obj is None:
                        break
                if obj is None:
                    continue
                if hasattr(obj, 'mu_embed'):
                    mu_embed = obj.mu_embed.weight.detach().cpu()[:n_tokens]
                if phi_embed is None and hasattr(obj, 'phi_embed'):
                    phi_embed = obj.phi_embed.weight.detach().cpu()[:n_tokens]
                if hasattr(obj, 'sigma_embed'):
                    sigma_embed = obj.sigma_embed.weight.detach().cpu()[:n_tokens]
                elif hasattr(obj, 'log_sigma_diag') and isinstance(obj.log_sigma_diag, torch.Tensor):
                    sigma_embed = torch.exp(obj.log_sigma_diag.detach().cpu()).clamp(min=1e-6)[:n_tokens]
                if mu_embed is not None:
                    break

        # gauge_fixed_priors fallback on token_embed
        if mu_embed is None:
            te = getattr(model, 'token_embed', None)
            if te is not None:
                mu_recon, phi_recon = _reconstruct_gauge_fixed_embeddings(
                    te, n_tokens=n_tokens,
                )
                if mu_recon is not None:
                    mu_embed = mu_recon
                if phi_embed is None and phi_recon is not None:
                    phi_embed = phi_recon

        if mu_embed is None:
            snapshot['error'] = 'no embeddings found'
            self.snapshots.append(snapshot)
            return snapshot

        # Clustering quality on mu
        mu_cluster = compute_clustering_metrics(mu_embed, n_tokens=n_tokens, embed_name='mu')
        sil = mu_cluster.get('mu_silhouette_score')
        if isinstance(sil, float):
            snapshot['mu_silhouette'] = sil
        ch = mu_cluster.get('mu_calinski_harabasz')
        if isinstance(ch, float):
            snapshot['mu_calinski_harabasz'] = ch
        ratio = mu_cluster.get('mu_inter_intra_ratio')
        if isinstance(ratio, float):
            snapshot['mu_inter_intra_ratio'] = ratio

        # Semantic field coherence on mu
        field_results = compute_semantic_field_coherence(mu_embed, embed_name='mu')
        snapshot['mu_field_coherence'] = field_results.get('mu_field_coherence_ratio', 0)
        snapshot['mu_n_fields'] = field_results.get('mu_n_fields_resolved', 0)

        # Phi clustering if available
        if phi_embed is not None:
            phi_cluster = compute_clustering_metrics(phi_embed, n_tokens=n_tokens, embed_name='phi')
            sil_phi = phi_cluster.get('phi_silhouette_score')
            if isinstance(sil_phi, float):
                snapshot['phi_silhouette'] = sil_phi

        # Sigma analysis if available
        if sigma_embed is not None:
            sigma_results = analyze_sigma_semantics(sigma_embed, n_tokens=n_tokens, verbose=False)
            eff_rank = sigma_results.get('sigma_effective_rank_mean')
            if eff_rank is not None:
                snapshot['sigma_effective_rank'] = eff_rank

        # Embedding norms (detect collapse or explosion)
        snapshot['mu_norm_mean'] = float(mu_embed.norm(dim=-1).mean())
        if phi_embed is not None:
            snapshot['phi_norm_mean'] = float(phi_embed.norm(dim=-1).mean())

        self.snapshots.append(snapshot)
        return snapshot

    def summarize(self) -> Dict[str, Any]:
        """Summarize semantic evolution across all recorded snapshots.

        Returns:
            Dict with trend analysis: when structure emerged, stability, etc.
        """
        if not self.snapshots:
            return {'error': 'no snapshots recorded'}

        steps = [s['step'] for s in self.snapshots]
        summary = {
            'n_snapshots': len(self.snapshots),
            'step_range': [min(steps), max(steps)],
        }

        # Track silhouette over time
        sil_values = [(s['step'], s['mu_silhouette']) for s in self.snapshots if 'mu_silhouette' in s]
        if len(sil_values) >= 2:
            sil_steps, sil_vals = zip(*sil_values)
            summary['mu_silhouette_initial'] = sil_vals[0]
            summary['mu_silhouette_final'] = sil_vals[-1]
            summary['mu_silhouette_improvement'] = sil_vals[-1] - sil_vals[0]
            # Find when silhouette first exceeds 0 (meaningful clustering)
            for step, val in sil_values:
                if val > 0:
                    summary['mu_silhouette_positive_at_step'] = step
                    break

        # Track field coherence over time
        fc_values = [(s['step'], s['mu_field_coherence']) for s in self.snapshots if 'mu_field_coherence' in s]
        if len(fc_values) >= 2:
            fc_steps, fc_vals = zip(*fc_values)
            summary['field_coherence_initial'] = fc_vals[0]
            summary['field_coherence_final'] = fc_vals[-1]

        return summary

    def save(self, path: Path) -> None:
        """Save snapshot history to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        def _serialize(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            if isinstance(obj, np.bool_):
                return bool(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        serializable = []
        for snap in self.snapshots:
            serializable.append({k: _serialize(v) for k, v in snap.items()})

        with open(path, 'w') as f:
            json.dump(serializable, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "SemanticTrajectoryTracker":
        """Load snapshot history from JSON."""
        tracker = cls()
        with open(path, 'r') as f:
            tracker.snapshots = json.load(f)
        return tracker
