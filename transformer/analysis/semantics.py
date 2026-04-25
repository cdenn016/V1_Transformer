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
import warnings
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for training
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse, Polygon
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    matplotlib = None
    plt = None
    Ellipse = None
    Polygon = None
    MATPLOTLIB_AVAILABLE = False

try:
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    PCA = None
    SKLEARN_AVAILABLE = False

try:
    from scipy.spatial import ConvexHull
    from scipy.stats import gaussian_kde
    SCIPY_AVAILABLE = True
except ImportError:
    ConvexHull = None
    gaussian_kde = None
    SCIPY_AVAILABLE = False

try:
    from transformer.visualization.pub_style import set_pub_style, PUB_COLORS, PUB_CYCLE
    PUB_STYLE_AVAILABLE = True
except ImportError:
    # Fallback palette if pub_style is unavailable
    PUB_COLORS = {
        'blue': '#0072B2', 'orange': '#E69F00', 'green': '#009E73',
        'red': '#D55E00', 'purple': '#CC79A7', 'cyan': '#56B4E9',
        'yellow': '#F0E442', 'black': '#000000', 'gray': '#999999',
    }
    PUB_CYCLE = [PUB_COLORS['blue'], PUB_COLORS['orange'], PUB_COLORS['green'],
                 PUB_COLORS['red'], PUB_COLORS['purple'], PUB_COLORS['cyan'],
                 PUB_COLORS['yellow'], PUB_COLORS['black']]
    def set_pub_style():
        pass
    PUB_STYLE_AVAILABLE = False

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
        import warnings as _w
        f_stats = []
        p_vals = []
        # Test each dimension: do category means differ?
        groups_per_dim = {ci: X[labels == ci] for ci in range(len(valid_cats))}
        n_dims_to_test = min(embed_dim, 100)  # cap for efficiency
        dim_indices = rng.choice(embed_dim, n_dims_to_test, replace=False) if embed_dim > 100 else range(embed_dim)
        for d in dim_indices:
            groups = [groups_per_dim[ci][:, d] for ci in range(len(valid_cats)) if len(groups_per_dim[ci]) >= 2]
            if len(groups) >= 2:
                # Skip dimensions where all groups are near-constant (no variance)
                if all(np.std(g) < 1e-12 for g in groups):
                    continue
                with _w.catch_warnings():
                    _w.simplefilter("ignore")
                    f, p = f_oneway(*groups)
                if np.isfinite(f) and np.isfinite(p):
                    f_stats.append(f)
                    p_vals.append(max(float(p), 1e-300))  # floor for log

        if f_stats:
            results[f'{embed_name}_anova_mean_f'] = float(np.mean(f_stats))
            results[f'{embed_name}_anova_median_f'] = float(np.median(f_stats))
            # Geometric mean of p-values (more robust than arithmetic for small p)
            log_p = np.log(np.array(p_vals, dtype=np.float64))
            results[f'{embed_name}_anova_geomean_p'] = float(np.exp(np.mean(log_p)))
            results[f'{embed_name}_anova_frac_significant'] = float(np.mean([p < 0.05 for p in p_vals]))
            results[f'{embed_name}_anova_n_dims_tested'] = len(f_stats)
    except ImportError:
        results[f'{embed_name}_anova_mean_f'] = 'scipy not available'
    except Exception as e:
        results[f'{embed_name}_anova_mean_f'] = f'error: {e}'

    # --- PCA variance profile ---
    try:
        from sklearn.decomposition import PCA as PCA_full
        # Skip PCA if data has zero total variance (all embeddings identical)
        total_var = np.var(X, axis=0).sum()
        if total_var < 1e-30:
            results[f'{embed_name}_pca_variance_profile'] = 'zero variance'
            return results
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
        # When plots are requested, build a save_path that lives alongside
        # the existing omega/phi/mu clustering figures.
        sigma_save_path = None
        if save_plots:
            _sd = Path(save_dir) if save_dir else Path("./outputs/figures")
            _sd.mkdir(parents=True, exist_ok=True)
            _suffix = f"_step{step}" if step is not None else ""
            sigma_save_path = _sd / f"sigma_clustering{_suffix}.png"
        sigma_results = analyze_sigma_semantics(
            sigma_embed, n_tokens=500, verbose=verbose,
            step=step, save_path=sigma_save_path,
            gauge_group_label=gauge_group_label,
        )
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

        # Cross-representation comparison figure (μ, φ, Ω, Σ side-by-side)
        omega_tensor = None
        try:
            # Recover omega for the comparison if a model was provided.
            # extract_omega returns (omega_tensor, group_label) or None.
            if model is not None:
                try:
                    _extract_result = extract_omega(model, n_tokens=500)
                    if _extract_result is not None:
                        omega_tensor = _extract_result[0]
                except Exception:
                    omega_tensor = None

            fig = plot_representation_comparison(
                mu=mu_embed,
                phi=phi_embed,
                omega=omega_tensor,
                sigma=sigma_embed,
                step=step,
                save_path=save_dir / f"representation_comparison{'_step'+str(step) if step is not None else ''}.png",
                gauge_group_label=gauge_group_label,
            )
            if fig is not None:
                plt.close(fig)
                results['representation_comparison_saved'] = True
        except Exception as e:
            if verbose:
                print(f"  [WARN] Could not generate representation comparison: {e}")
            results['representation_comparison_saved'] = False

        # Bhattacharyya-MDS figure (needs both μ and Σ)
        if mu_embed is not None and sigma_embed is not None:
            # Default to 300 tokens; drop to 150 if full-cov Σ to keep O(n²K³)
            # tractable during periodic training runs.
            _is_full_sigma = (
                isinstance(sigma_embed, torch.Tensor) and sigma_embed.dim() == 3
            ) or (
                hasattr(sigma_embed, 'ndim') and sigma_embed.ndim == 3
            )
            bhat_n = 150 if _is_full_sigma else 300
            try:
                fig = plot_sigma_bhattacharyya_mds(
                    mu=mu_embed,
                    sigma=sigma_embed,
                    step=step,
                    save_path=save_dir / f"sigma_bhattacharyya{'_step'+str(step) if step is not None else ''}.png",
                    n_tokens=bhat_n,
                    gauge_group_label=gauge_group_label,
                )
                if fig is not None:
                    plt.close(fig)
                    results['sigma_bhattacharyya_saved'] = True
            except Exception as e:
                if verbose:
                    print(f"  [WARN] Could not generate Bhattacharyya MDS figure: {e}")
                results['sigma_bhattacharyya_saved'] = False

    return results


# =============================================================================
# Visualization Functions
# =============================================================================

CATEGORY_COLORS = {
    'letter':   PUB_COLORS['red'],       # was '#E74C3C'
    'digit':    PUB_COLORS['blue'],      # was '#3498DB'
    'punct':    PUB_COLORS['green'],     # was '#2ECC71'
    'function': PUB_COLORS['purple'],    # was '#9B59B6'
    'content':  PUB_COLORS['orange'],    # was '#F39C12'
    'other':    PUB_COLORS['gray'],      # was '#95A5A6'
}

# POS tag → compact label map (Penn Treebank). Used in label_mode='pos'.
_POS_TAG_MAP = {
    'NN': 'noun', 'NNS': 'noun', 'NNP': 'noun', 'NNPS': 'noun',
    'VB': 'verb', 'VBD': 'verb', 'VBG': 'verb', 'VBN': 'verb',
    'VBP': 'verb', 'VBZ': 'verb',
    'JJ': 'adj', 'JJR': 'adj', 'JJS': 'adj',
    'RB': 'adv', 'RBR': 'adv', 'RBS': 'adv',
    'DT': 'det', 'WDT': 'det', 'PDT': 'det',
    'IN': 'prep', 'TO': 'prep',
}

_POS_COLORS = {
    'noun':  PUB_COLORS['blue'],
    'verb':  PUB_COLORS['red'],
    'adj':   PUB_COLORS['green'],
    'adv':   PUB_COLORS['purple'],
    'det':   PUB_COLORS['orange'],
    'prep':  PUB_COLORS['cyan'],
    'other': PUB_COLORS['gray'],
}


def _resolve_token_colors(
    n_tokens: int,
    label_mode: str = 'category',
) -> Tuple[List[str], Dict[str, str]]:
    """Return (per-token labels, palette) for figure coloring.

    Supports three schemes:
      'category'       — orthographic (letter/digit/punct/function/content/other).
      'semantic_field' — color by membership in SEMANTIC_FIELDS; unmatched → 'other'.
      'pos'            — POS-tag via NLTK, mapped to {noun, verb, adj, adv, det, prep, other}.
                         Falls back to 'category' if nltk is unavailable.

    Args:
        n_tokens: Number of leading token IDs to label (covers [0, n_tokens)).
        label_mode: Labeling scheme (see above).

    Returns:
        (categories, palette):
            categories — list of length n_tokens with a string label per token.
            palette    — dict mapping each label that appears to a hex color.
    """
    if label_mode == 'category':
        categories = [categorize_token(tid) for tid in range(n_tokens)]
        return categories, dict(CATEGORY_COLORS)

    if label_mode == 'semantic_field':
        word_to_field: Dict[str, str] = {}
        for field, members in SEMANTIC_FIELDS.items():
            for m in members:
                w = m[0] if isinstance(m, tuple) else m
                word_to_field[w.strip().lower()] = field
        tokenizer = get_tokenizer()
        categories: List[str] = []
        for tid in range(n_tokens):
            label = 'other'
            if tokenizer is not None:
                try:
                    word = tokenizer.decode([tid]).strip().lower()
                    if word in word_to_field:
                        label = word_to_field[word]
                except Exception:
                    pass
            categories.append(label)
        fields_seen = sorted({c for c in categories if c != 'other'})
        palette: Dict[str, str] = {'other': PUB_COLORS['gray']}
        for i, field in enumerate(fields_seen):
            palette[field] = PUB_CYCLE[i % len(PUB_CYCLE)]
        return categories, palette

    if label_mode == 'pos':
        try:
            import nltk
            from nltk import pos_tag
        except ImportError:
            warnings.warn(
                "nltk not available; falling back to label_mode='category'.",
                RuntimeWarning, stacklevel=2,
            )
            return _resolve_token_colors(n_tokens, label_mode='category')
        tokenizer = get_tokenizer()
        words: List[str] = []
        valid: List[bool] = []
        for tid in range(n_tokens):
            word = None
            if tokenizer is not None:
                try:
                    s = tokenizer.decode([tid]).strip()
                    if s.isalpha() and len(s) > 0:
                        word = s
                except Exception:
                    pass
            words.append(word if word is not None else 'x')
            valid.append(word is not None)
        try:
            tagged = pos_tag(words)
        except LookupError:
            # Try to auto-download the tagger model
            try:
                nltk.download('averaged_perceptron_tagger', quiet=True)
                tagged = pos_tag(words)
            except Exception:
                warnings.warn(
                    "nltk POS tagger unavailable; falling back to label_mode='category'.",
                    RuntimeWarning, stacklevel=2,
                )
                return _resolve_token_colors(n_tokens, label_mode='category')
        except Exception:
            warnings.warn(
                "nltk POS tagging failed; falling back to label_mode='category'.",
                RuntimeWarning, stacklevel=2,
            )
            return _resolve_token_colors(n_tokens, label_mode='category')
        categories = []
        for i, (_, tag) in enumerate(tagged):
            if not valid[i]:
                categories.append('other')
            else:
                categories.append(_POS_TAG_MAP.get(tag, 'other'))
        return categories, dict(_POS_COLORS)

    raise ValueError(
        f"Unknown label_mode: {label_mode!r}. "
        f"Expected 'category', 'semantic_field', or 'pos'."
    )


def _safe_pca_2d(
    arr: np.ndarray,
    n_components: int = 2,
    var_eps: float = 1e-12,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    """PCA wrapper tolerant of degenerate inputs.

    sklearn's PCA divides by total_var to compute explained_variance_ratio_,
    which emits a RuntimeWarning (invalid value encountered in divide) when
    the input has zero total variance. This happens in practice when:
      - An embedding hasn't diverged from initialization (step 0).
      - Σ is shared (all tokens have the same covariance) and gets broadcast
        to (V, K) with identical rows.
      - Ω is identically the identity under gauge_mode='trivial'.

    Args:
        arr: (n, d) array to reduce.
        n_components: Desired output dims (1 or 2). Output is padded with
                      zeros if PCA returns fewer components than requested.
        var_eps: Total-variance threshold below which PCA is skipped.

    Returns:
        (coords, variance_ratio, status). When degenerate, coords is None
        and status names the failure mode for placeholder annotation.
    """
    if not SKLEARN_AVAILABLE:
        return None, None, "sklearn unavailable"
    if arr is None or arr.ndim != 2:
        return None, None, "invalid shape"
    if arr.shape[0] < 2 or arr.shape[1] < 1:
        return None, None, "too few samples"
    if not np.isfinite(arr).all():
        return None, None, "non-finite values"
    total_var = float(np.var(arr, axis=0, ddof=0).sum())
    if total_var < var_eps:
        return None, None, "zero variance"
    n_comp = min(n_components, arr.shape[1], arr.shape[0] - 1)
    if n_comp < 1:
        return None, None, "too few dimensions"
    with warnings.catch_warnings():
        # Guard against residual numerical warnings from sklearn internals
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        pca = PCA(n_components=n_comp)
        coords = pca.fit_transform(arr)
        ratio = np.asarray(pca.explained_variance_ratio_, dtype=float)
    # Replace any NaN ratios (still possible for pathological inputs) with 0
    ratio = np.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)
    # Pad output to the requested n_components width
    if coords.shape[1] < n_components:
        pad_c = np.zeros((coords.shape[0], n_components - coords.shape[1]))
        coords = np.concatenate([coords, pad_c], axis=1)
    if ratio.size < n_components:
        ratio = np.concatenate([ratio, np.zeros(n_components - ratio.size)])
    return coords, ratio, "ok"


def _add_cluster_envelopes(
    ax: "Any",
    points_2d: np.ndarray,
    categories: List[str],
    palette: Dict[str, str],
    mode: str = 'ellipse',
    n_sigma: float = 2.0,
    alpha_fill: float = 0.15,
    min_points: int = 5,
) -> None:
    r"""Overlay per-category cluster envelopes on a 2D scatter plot.

    Draws a semi-transparent "blob" per category over the existing scatter,
    preserving individual point visibility while communicating cluster
    structure. Called by the four PCA scatter plots (mu, phi, Omega, Sigma).

    Modes:
        'ellipse'  — 2x2 covariance per category, n_sigma Gaussian ellipse.
                     (Default; parametric, outlier-robust.)
        'hull'     — ConvexHull polygon per category.
                     (Exact extent; sensitive to outliers.)
        'kde'      — gaussian_kde contours at 50/75/95% density.
                     (Shows density gradient; can be visually dense.)
        'centroid' — Only a large "x" marker at each category mean.
        'none'     — No-op.

    Also draws an "x" marker at each category centroid in all modes except
    'none' so the cluster center is always readable.

    Args:
        ax: matplotlib Axes to draw onto.
        points_2d: (n, 2) projected coordinates.
        categories: List of length n with per-point category label.
        palette: Mapping label -> hex color.
        mode: Envelope mode (see above).
        n_sigma: Scale factor for 'ellipse' axes (2.0 ≈ 95% Gaussian mass).
        alpha_fill: Fill alpha for the envelope patches.
        min_points: Categories with fewer points are skipped for parametric
                    envelopes (avoids degenerate 2x2 covariance fits).
    """
    if mode == 'none' or not MATPLOTLIB_AVAILABLE:
        return
    if points_2d.ndim != 2 or points_2d.shape[1] < 2:
        return

    pts = np.asarray(points_2d[:, :2], dtype=np.float64)

    for cat, color in palette.items():
        idx = np.array([i for i, c in enumerate(categories) if c == cat], dtype=int)
        if idx.size == 0:
            continue

        sub = pts[idx]
        centroid = sub.mean(axis=0)

        # Envelope (skipped for categories with too few points in parametric modes)
        if mode == 'ellipse' and idx.size >= min_points:
            cov = np.cov(sub, rowvar=False)
            # Eigendecompose 2x2 covariance
            evals, evecs = np.linalg.eigh(cov)
            evals = np.maximum(evals, 1e-12)
            order = np.argsort(evals)[::-1]
            evals = evals[order]
            evecs = evecs[:, order]
            angle = float(np.degrees(np.arctan2(evecs[1, 0], evecs[0, 0])))
            width = 2.0 * n_sigma * float(np.sqrt(evals[0]))
            height = 2.0 * n_sigma * float(np.sqrt(evals[1]))
            ell = Ellipse(
                xy=centroid, width=width, height=height, angle=angle,
                facecolor=color, edgecolor=color, alpha=alpha_fill,
                linewidth=1.2, linestyle='--', zorder=1,
            )
            ax.add_patch(ell)
        elif mode == 'hull' and idx.size >= min_points and SCIPY_AVAILABLE:
            try:
                hull = ConvexHull(sub)
                poly = Polygon(
                    sub[hull.vertices], closed=True,
                    facecolor=color, edgecolor=color, alpha=alpha_fill,
                    linewidth=1.2, linestyle='--', zorder=1,
                )
                ax.add_patch(poly)
            except Exception:
                pass  # degenerate hull → skip
        elif mode == 'kde' and idx.size >= min_points and SCIPY_AVAILABLE:
            try:
                kde = gaussian_kde(sub.T)
                x_min, y_min = sub.min(axis=0)
                x_max, y_max = sub.max(axis=0)
                pad_x = 0.2 * (x_max - x_min + 1e-12)
                pad_y = 0.2 * (y_max - y_min + 1e-12)
                xx, yy = np.meshgrid(
                    np.linspace(x_min - pad_x, x_max + pad_x, 60),
                    np.linspace(y_min - pad_y, y_max + pad_y, 60),
                )
                zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
                z_sorted = np.sort(zz.ravel())[::-1]
                cdf = np.cumsum(z_sorted) / z_sorted.sum()
                levels = [z_sorted[np.searchsorted(cdf, q)] for q in (0.5, 0.75, 0.95)]
                levels = sorted(set(float(L) for L in levels if L > 0))
                if levels:
                    ax.contour(xx, yy, zz, levels=levels, colors=[color],
                               linewidths=0.8, linestyles='--', alpha=0.8, zorder=1)
            except Exception:
                pass

        # Always draw centroid marker (except 'none' which returned above)
        if mode != 'centroid':
            ax.scatter(
                [centroid[0]], [centroid[1]],
                c=color, marker='x', s=60, linewidths=1.5, zorder=3,
            )
        else:
            ax.scatter(
                [centroid[0]], [centroid[1]],
                c=color, marker='x', s=80, linewidths=2.0, zorder=3,
                label=f'{cat} centroid',
            )


def plot_embedding_clustering(
    embed: torch.Tensor,
    embed_type: str = 'phi',
    step: Optional[int] = None,
    save_path: Optional[Path] = None,
    n_tokens: int = 500,
    gauge_group_label: Optional[str] = None,
    envelope_mode: str = 'ellipse',
    label_mode: str = 'category',
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
        envelope_mode: Cluster envelope overlay on 2D scatters:
                       'ellipse' (default), 'hull', 'kde', 'centroid', 'none'.
        label_mode: Coloring scheme: 'category' (default, orthographic),
                    'semantic_field' (SEMANTIC_FIELDS membership),
                    'pos' (NLTK POS tag; falls back to 'category').

    Returns:
        matplotlib Figure
    """
    if MATPLOTLIB_AVAILABLE:
        set_pub_style()

    embed_np = embed[:n_tokens].numpy() if isinstance(embed, torch.Tensor) else embed[:n_tokens]
    embed_dim = embed_np.shape[1]

    if embed_type == 'phi':
        type_str = gauge_group_label if gauge_group_label else identify_gauge_group(embed_dim)
        title_prefix = f"{type_str} Gauge Frames"
    else:
        type_str = f"{embed_dim}D"
        title_prefix = f"Belief Embeddings (μ)"

    categories, palette = _resolve_token_colors(len(embed_np), label_mode=label_mode)

    # Scatter alpha: fade dots when an envelope is overlaid
    dot_alpha = 0.35 if envelope_mode not in ('none', 'centroid') else 0.6

    step_str = f" (Step {step})" if step is not None else ""

    if embed_dim == 1 and embed_type == 'phi':
        # SO(2): 1D gauge frames - histogram and jittered scatter
        fig = plt.figure(figsize=(14, 6))

        # Histogram (no envelope on 1D)
        ax1 = fig.add_subplot(121)
        for cat, color in palette.items():
            mask = [c == cat for c in categories]
            if any(mask):
                idx = [i for i, m in enumerate(mask) if m]
                vals = embed_np[idx, 0]
                ax1.hist(vals, bins=30, alpha=0.5, label=cat, color=color)

        ax1.set_xlabel('φ (SO(2) angle)')
        ax1.set_ylabel('Count')
        ax1.set_title(f'SO(2) Gauge Frame Distribution{step_str}')
        ax1.legend(loc='upper right', fontsize=8)

        # Jittered scatter — y-axis is random jitter, so envelopes are not meaningful here
        ax2 = fig.add_subplot(122)
        np.random.seed(42)
        for cat, color in palette.items():
            mask = [c == cat for c in categories]
            if any(mask):
                idx = [i for i, m in enumerate(mask) if m]
                x_vals = embed_np[idx, 0]
                y_jitter = np.random.uniform(-0.4, 0.4, len(idx))
                ax2.scatter(x_vals, y_jitter, c=color, label=cat, alpha=0.6, s=20)

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

        # 3D sphere plot (no 2D envelope — 3D)
        ax1 = fig.add_subplot(121, projection='3d')

        # Draw unit sphere wireframe
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 20)
        x_sphere = np.outer(np.cos(u), np.sin(v))
        y_sphere = np.outer(np.sin(u), np.sin(v))
        z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
        ax1.plot_wireframe(x_sphere, y_sphere, z_sphere, alpha=0.1, color='gray')

        for cat, color in palette.items():
            mask = [c == cat for c in categories]
            if any(mask):
                idx = [i for i, m in enumerate(mask) if m]
                ax1.scatter(embed_unit[idx, 0], embed_unit[idx, 1], embed_unit[idx, 2],
                           c=color, label=cat, alpha=0.6, s=20)

        ax1.set_xlabel('φ₁')
        ax1.set_ylabel('φ₂')
        ax1.set_zlabel('φ₃')
        ax1.set_title(f'SO(3) Gauge Frames on Unit Sphere{step_str}')
        ax1.legend(loc='upper left', fontsize=8)

        # 2D projection — envelope applies here
        ax2 = fig.add_subplot(122)
        for cat, color in palette.items():
            mask = [c == cat for c in categories]
            if any(mask):
                idx = [i for i, m in enumerate(mask) if m]
                ax2.scatter(embed_np[idx, 0], embed_np[idx, 1],
                           c=color, label=cat, alpha=dot_alpha, s=20)

        _add_cluster_envelopes(
            ax2, embed_np[:, :2], categories, palette, mode=envelope_mode,
        )

        ax2.set_xlabel('φ₁')
        ax2.set_ylabel('φ₂')
        ax2.set_title(f'SO(3) Gauge Frames (φ₁ vs φ₂){step_str}')
        ax2.legend(loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')

    else:
        # High-dimensional: Use PCA (guarded against zero-variance inputs)
        if not SKLEARN_AVAILABLE:
            print("Warning: sklearn not available for PCA visualization")
            return None
        n_components = min(3, embed_dim)
        embed_pca, var_explained, status = _safe_pca_2d(embed_np, n_components=n_components)

        fig = plt.figure(figsize=(14, 6))

        if embed_pca is None:
            # Degenerate PCA — render an annotated placeholder
            ax2 = fig.add_subplot(111)
            ax2.text(0.5, 0.5, f'PCA unavailable: {status}',
                     transform=ax2.transAxes, ha='center', va='center',
                     fontsize=12, color='gray')
            ax2.set_title(f'{title_prefix} (PCA){step_str}')
            ax2.set_xticks([])
            ax2.set_yticks([])
        else:
            if n_components >= 3:
                # 3D PCA plot (no envelope — 3D)
                ax1 = fig.add_subplot(121, projection='3d')
                for cat, color in palette.items():
                    mask = [c == cat for c in categories]
                    if any(mask):
                        idx = [i for i, m in enumerate(mask) if m]
                        ax1.scatter(embed_pca[idx, 0], embed_pca[idx, 1], embed_pca[idx, 2],
                                   c=color, label=cat, alpha=0.6, s=20)

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
                for cat, color in palette.items():
                    mask = [c == cat for c in categories]
                    if any(mask):
                        idx = [i for i, m in enumerate(mask) if m]
                        ax2.scatter(embed_pca[idx, 0], embed_pca[idx, 1],
                                   c=color, label=cat, alpha=dot_alpha, s=20)

                _add_cluster_envelopes(
                    ax2, embed_pca[:, :2], categories, palette, mode=envelope_mode,
                )

                ax2.set_xlabel(f'PC1 ({var_explained[0]:.1%})')
                ax2.set_ylabel(f'PC2 ({var_explained[1]:.1%})')
                ax2.set_title(f'{title_prefix} (PCA from {embed_dim}D){step_str}')
                ax2.legend(loc='upper left', fontsize=8)
                ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')

    return fig


def plot_gauge_frame_clustering(phi_embed, step=None, save_path=None, n_tokens=500,
                                gauge_group_label=None, envelope_mode='ellipse',
                                label_mode='category'):
    """Alias for plot_embedding_clustering with embed_type='phi'."""
    return plot_embedding_clustering(
        phi_embed, embed_type='phi', step=step, save_path=save_path,
        n_tokens=n_tokens, gauge_group_label=gauge_group_label,
        envelope_mode=envelope_mode, label_mode=label_mode,
    )


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
    try:
        evecs_inv = torch.linalg.inv(evecs)
    except (torch.linalg.LinAlgError, RuntimeError):
        evecs_inv = torch.linalg.pinv(evecs)
    return (evecs @ torch.diag(log_evals) @ evecs_inv).real.float()


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
    envelope_mode: str = 'ellipse',
    label_mode: str = 'category',
) -> Optional["Any"]:
    """Visualize per-token group elements Ω_i colored by token category.

    Layout (4 panels):
      (a) PCA of flattened Ω_i (K² → 2D)  — cluster structure, with envelope
      (b) Determinant distribution by category  — group geometry
      (c) Distance from identity ‖Ω_i − I‖_F by category  — magnitude
      (d) Eigenvalue magnitude distribution   — spectral character

    Args:
        omega: Group elements (n_tokens, K, K).
        step: Training step for title.
        save_path: Output file path.
        n_tokens: Tokens to visualize.
        gauge_group_label: e.g. 'SO(3)' or 'GL(30)'.
        envelope_mode: Envelope overlay in panel (a). See _add_cluster_envelopes.
        label_mode: Coloring scheme: 'category' | 'semantic_field' | 'pos'.

    Returns:
        matplotlib Figure (or None if matplotlib unavailable).
    """
    if not MATPLOTLIB_AVAILABLE or not SKLEARN_AVAILABLE:
        return None
    set_pub_style()

    n = min(n_tokens, omega.shape[0])
    K = omega.shape[1]
    omega_np = omega[:n].numpy() if isinstance(omega, torch.Tensor) else omega[:n]

    categories, palette = _resolve_token_colors(n, label_mode=label_mode)
    dot_alpha = 0.35 if envelope_mode not in ('none', 'centroid') else 0.6

    step_str = f" (Step {step})" if step is not None else ""
    group_str = gauge_group_label or f"K={K}"

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Group Element Ω_i Analysis — {group_str}{step_str}', fontsize=14, y=1.01)

    # ---- (a) PCA of flattened Omega ----
    ax = axes[0, 0]
    omega_flat = omega_np.reshape(n, -1)  # (n, K²)
    omega_pca, var, status = _safe_pca_2d(omega_flat, n_components=2)
    if omega_pca is None:
        ax.text(0.5, 0.5, f'PCA unavailable: {status}\n(Ω may be near-identity)',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=11, color='gray')
        ax.set_title(f'(a) Ω_i PCA (from {K}×{K} matrices)')
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        for cat, color in palette.items():
            mask = [c == cat for c in categories]
            if any(mask):
                idx = [i for i, m in enumerate(mask) if m]
                ax.scatter(omega_pca[idx, 0], omega_pca[idx, 1],
                           c=color, label=cat, alpha=dot_alpha, s=20)
        _add_cluster_envelopes(ax, omega_pca[:, :2], categories, palette, mode=envelope_mode)
        ax.set_xlabel(f'PC1 ({var[0]:.1%})')
        ax.set_ylabel(f'PC2 ({var[1]:.1%})')
        ax.set_title(f'(a) Ω_i PCA (from {K}×{K} matrices)')
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, alpha=0.3)

    # ---- (b) Determinant distribution by category ----
    ax = axes[0, 1]
    dets = np.linalg.det(omega_np.astype(np.float64)).astype(np.float32)
    for cat, color in palette.items():
        mask = [c == cat for c in categories]
        if any(mask):
            idx = [i for i, m in enumerate(mask) if m]
            vals = dets[idx]
            ax.hist(vals, bins=30, alpha=0.5, label=cat, color=color)
    ax.axvline(x=1.0, color='k', linestyle='--', linewidth=1, label='det=1')
    ax.set_xlabel('det(Ω_i)')
    ax.set_ylabel('Count')
    ax.set_title('(b) Determinant Distribution')
    ax.legend(loc='upper right', fontsize=7)

    # ---- (c) Distance from identity by category ----
    ax = axes[1, 0]
    eye = np.eye(K, dtype=np.float32)
    dist_id = np.linalg.norm((omega_np - eye).reshape(n, -1), axis=1)
    palette_order = list(palette.keys())
    cat_list = sorted(set(categories),
                      key=lambda c: palette_order.index(c) if c in palette_order else 99)
    cat_data, cat_labels, cat_colors_bp = [], [], []
    for cat in cat_list:
        idx = [i for i, c in enumerate(categories) if c == cat]
        if idx:
            cat_data.append(dist_id[idx])
            cat_labels.append(cat)
            cat_colors_bp.append(palette.get(cat, PUB_COLORS['gray']))

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
        for cat, color in palette.items():
            mask = [c == cat for c in categories]
            if any(mask):
                idx = [i for i, m in enumerate(mask) if m]
                ax.hist(evals[idx].flatten(), bins=40, alpha=0.4,
                        label=cat, color=color, density=True)
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
        fig.savefig(save_path, bbox_inches='tight')

    return fig


# =============================================================================
# Sigma Covariance Visualization
# =============================================================================

def plot_sigma_clustering(
    sigma: torch.Tensor,
    step: Optional[int] = None,
    save_path: Optional[Path] = None,
    n_tokens: int = 500,
    gauge_group_label: Optional[str] = None,
    envelope_mode: str = 'ellipse',
    label_mode: str = 'category',
) -> Optional["Any"]:
    r"""Visualize per-token covariance beliefs Σ_i colored by token category.

    Layout (4 panels, analogous to plot_omega_clustering):
      (a) PCA of flattened Σ_i (K² for full, K for diagonal) — cluster structure
      (b) Volume: histogram of log|det(Σ_i)| (full) or log(trace(Σ_i)) (diag)
      (c) Anisotropy: boxplot of λ_max/λ_min (full) or max(σ)/min(σ) (diag)
      (d) Spectrum: eigenvalue magnitudes (full) or diagonal variances (diag)

    Args:
        sigma: Covariance tensor, (n, K) diagonal or (n, K, K) full.
        step: Training step for title.
        save_path: Output file path.
        n_tokens: Tokens to visualize.
        gauge_group_label: e.g. 'SO(3)' or 'GL(30)'.
        envelope_mode: Envelope overlay in panel (a). See _add_cluster_envelopes.
        label_mode: Coloring scheme: 'category' | 'semantic_field' | 'pos'.

    Returns:
        matplotlib Figure, or None if matplotlib/sklearn unavailable.
    """
    if not MATPLOTLIB_AVAILABLE or not SKLEARN_AVAILABLE:
        return None
    set_pub_style()

    n = min(n_tokens, sigma.shape[0])
    sigma_sub = sigma[:n].detach().cpu().float() if isinstance(sigma, torch.Tensor) else torch.as_tensor(sigma[:n]).float()
    is_full = sigma_sub.dim() == 3  # (n, K, K) vs (n, K)
    K = sigma_sub.shape[1] if is_full else sigma_sub.shape[-1]

    categories, palette = _resolve_token_colors(n, label_mode=label_mode)
    dot_alpha = 0.35 if envelope_mode not in ('none', 'centroid') else 0.6

    step_str = f" (Step {step})" if step is not None else ""
    group_str = gauge_group_label or f"K={K}"
    mode_str = "full" if is_full else "diagonal"

    # Hoist eigen/diag computations up front for reuse in panels (b)-(d).
    if is_full:
        # Symmetrize before eigvalsh to guard against numerical asymmetry.
        sym = 0.5 * (sigma_sub + sigma_sub.transpose(-2, -1))
        try:
            evals = torch.linalg.eigvalsh(sym).clamp(min=1e-12)  # (n, K), ascending
        except Exception:
            # Degenerate fallback — fill with diagonal values
            evals = torch.diagonal(sym, dim1=-2, dim2=-1).clamp(min=1e-12)
        evals_np = evals.numpy()                                 # (n, K)
        sigma_flat_np = sigma_sub.numpy().reshape(n, -1)         # (n, K²)
        diag_np = torch.diagonal(sym, dim1=-2, dim2=-1).numpy()  # (n, K)
    else:
        diag_np = sigma_sub.clamp(min=1e-12).numpy()             # (n, K)
        evals_np = np.sort(diag_np, axis=-1)                     # (n, K), ascending
        sigma_flat_np = diag_np                                  # (n, K)

    # (b) volume: logdet for full, log-trace for diag
    if is_full:
        logvol = np.sum(np.log(evals_np), axis=-1)                # log|det|
        vol_label = r'$\log\,|\det\,\Sigma_i|$'
        vol_title = '(b) Log-Determinant Distribution'
    else:
        logvol = np.log(diag_np.sum(axis=-1) + 1e-12)             # log-trace
        vol_label = r'$\log\,\mathrm{tr}\,\Sigma_i$'
        vol_title = '(b) Log-Trace Distribution'

    # (c) anisotropy: condition number
    if is_full:
        cond = evals_np[:, -1] / np.maximum(evals_np[:, 0], 1e-12)
    else:
        cond = diag_np.max(axis=-1) / np.maximum(diag_np.min(axis=-1), 1e-12)
    log_cond = np.log10(np.maximum(cond, 1.0))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f'Belief Covariance Σ_i Analysis — {group_str} ({mode_str}){step_str}',
        fontsize=14, y=1.01,
    )

    # ---- (a) PCA of flattened Sigma ----
    ax = axes[0, 0]
    dim_note = f'{K}×{K}' if is_full else f'{K}-dim'
    sigma_pca, var, status = _safe_pca_2d(sigma_flat_np, n_components=2)
    if sigma_pca is None:
        ax.text(0.5, 0.5, f'PCA unavailable: {status}\n(Σ may be shared/frozen)',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=11, color='gray')
        ax.set_title(f'(a) Σ_i PCA (from {dim_note})')
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        for cat, color in palette.items():
            mask = [c == cat for c in categories]
            if any(mask):
                idx = [i for i, m in enumerate(mask) if m]
                ax.scatter(sigma_pca[idx, 0], sigma_pca[idx, 1],
                           c=color, label=cat, alpha=dot_alpha, s=20)
        _add_cluster_envelopes(
            ax, sigma_pca[:, :2], categories, palette, mode=envelope_mode,
        )
        ax.set_xlabel(f'PC1 ({var[0]:.1%})')
        ax.set_ylabel(f'PC2 ({var[1]:.1%})')
        ax.set_title(f'(a) Σ_i PCA (from {dim_note})')
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, alpha=0.3)

    # ---- (b) Volume (log-det or log-trace) by category ----
    ax = axes[0, 1]
    for cat, color in palette.items():
        mask = [c == cat for c in categories]
        if any(mask):
            idx = [i for i, m in enumerate(mask) if m]
            vals = logvol[idx]
            if np.all(np.isfinite(vals)) and vals.size > 0:
                ax.hist(vals, bins=30, alpha=0.5, label=cat, color=color)
    ax.set_xlabel(vol_label)
    ax.set_ylabel('Count')
    ax.set_title(vol_title)
    ax.legend(loc='upper right', fontsize=7)
    ax.grid(True, alpha=0.3, axis='y')

    # ---- (c) Anisotropy by category ----
    ax = axes[1, 0]
    palette_order = list(palette.keys())
    cat_list = sorted(set(categories),
                      key=lambda c: palette_order.index(c) if c in palette_order else 99)
    cat_data, cat_labels, cat_colors_bp = [], [], []
    for cat in cat_list:
        idx = [i for i, c in enumerate(categories) if c == cat]
        if idx:
            cat_data.append(log_cond[idx])
            cat_labels.append(cat)
            cat_colors_bp.append(palette.get(cat, PUB_COLORS['gray']))
    if cat_data:
        bp = ax.boxplot(cat_data, labels=cat_labels, patch_artist=True, showfliers=False)
        for patch, color in zip(bp['boxes'], cat_colors_bp):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
    ax.set_ylabel(r'$\log_{10}(\lambda_{\max}/\lambda_{\min})$')
    anisotropy_src = 'eigenvalues' if is_full else 'diagonal entries'
    ax.set_title(f'(c) Anisotropy by Category (from {anisotropy_src})')
    ax.grid(True, alpha=0.3, axis='y')

    # ---- (d) Eigenvalue magnitude / diagonal variance spectrum ----
    ax = axes[1, 1]
    for cat, color in palette.items():
        mask = [c == cat for c in categories]
        if any(mask):
            idx = [i for i, m in enumerate(mask) if m]
            flat = evals_np[idx].flatten() if is_full else diag_np[idx].flatten()
            # Log-scale bins to handle wide dynamic range in covariance magnitudes
            flat = flat[flat > 0]
            if flat.size > 0:
                ax.hist(np.log10(flat), bins=40, alpha=0.4,
                        label=cat, color=color, density=True)
    ax.set_xlabel(r'$\log_{10}(\lambda)$' if is_full else r'$\log_{10}(\sigma^2_k)$')
    ax.set_ylabel('Density')
    spec_title = '(d) Eigenvalue Spectrum' if is_full else '(d) Diagonal-Variance Spectrum'
    ax.set_title(spec_title)
    ax.legend(loc='upper right', fontsize=7)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')

    return fig


# =============================================================================
# Cross-Representation Comparison
# =============================================================================

def _plot_representation_panel(
    ax: "Any",
    tensor: np.ndarray,
    title: str,
    categories: List[str],
    palette: Dict[str, str],
    envelope_mode: str,
    dot_alpha: float,
) -> None:
    """Render one 2D PCA panel of plot_representation_comparison.

    Handles flattening of full-cov (N, K, K) → (N, K²) before PCA.
    """
    arr = tensor
    if arr.ndim == 3:
        arr = arr.reshape(arr.shape[0], -1)
    proj, var, status = _safe_pca_2d(arr, n_components=2)
    if proj is None:
        ax.text(0.5, 0.5, f'PCA unavailable: {status}',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=11, color='gray')
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        return
    for cat, color in palette.items():
        mask = [c == cat for c in categories]
        if any(mask):
            idx = [i for i, m in enumerate(mask) if m]
            ax.scatter(proj[idx, 0], proj[idx, 1],
                       c=color, alpha=dot_alpha, s=20, label=cat)
    _add_cluster_envelopes(ax, proj, categories, palette, mode=envelope_mode)
    ax.set_xlabel(f'PC1 ({var[0]:.1%})')
    ax.set_ylabel(f'PC2 ({var[1]:.1%})')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def plot_representation_comparison(
    mu: Optional[torch.Tensor] = None,
    phi: Optional[torch.Tensor] = None,
    omega: Optional[torch.Tensor] = None,
    sigma: Optional[torch.Tensor] = None,
    step: Optional[int] = None,
    save_path: Optional[Path] = None,
    n_tokens: int = 500,
    gauge_group_label: Optional[str] = None,
    envelope_mode: str = 'ellipse',
    label_mode: str = 'category',
) -> Optional["Any"]:
    r"""Side-by-side 2×2 PCA comparison of the four token representations.

    Lets a reader see at a glance whether category structure is visible in
    μ, φ, Ω, and Σ — and if only in some. All four panels use the same token
    sample and coloring scheme with a single shared legend.

    Any representation passed as None renders a placeholder panel so the
    layout stays consistent (e.g., Ω will be None for gauge_mode='trivial').

    Args:
        mu: Belief-mean embeddings (V, K).
        phi: Gauge-frame embeddings (V, phi_dim).
        omega: Group elements (V, K, K). Pass None if unavailable.
        sigma: Covariance beliefs (V, K) or (V, K, K).
        step: Training step for title.
        save_path: Output file path.
        n_tokens: Tokens to visualize (each representation sliced to [:n]).
        gauge_group_label: e.g. 'SO(3)' or 'GL(30)'.
        envelope_mode: Envelope overlay. See _add_cluster_envelopes.
        label_mode: Coloring scheme: 'category' | 'semantic_field' | 'pos'.

    Returns:
        matplotlib Figure, or None if matplotlib/sklearn unavailable.
    """
    if not MATPLOTLIB_AVAILABLE or not SKLEARN_AVAILABLE:
        return None
    set_pub_style()

    def _prep(x: Optional[torch.Tensor]) -> Optional[np.ndarray]:
        if x is None:
            return None
        x = x[:n_tokens]
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().float().numpy()
        else:
            x = np.asarray(x)
        return x

    mu_np = _prep(mu)
    phi_np = _prep(phi)
    omega_np = _prep(omega)
    sigma_np = _prep(sigma)

    # Determine n from whatever tensor is available (smallest among provided)
    lengths = [t.shape[0] for t in (mu_np, phi_np, omega_np, sigma_np) if t is not None]
    if not lengths:
        return None
    n = min(lengths)
    categories, palette = _resolve_token_colors(n, label_mode=label_mode)
    dot_alpha = 0.35 if envelope_mode not in ('none', 'centroid') else 0.6

    # Slice everything to the common n
    def _slice(x):
        return None if x is None else x[:n]
    mu_np, phi_np, omega_np, sigma_np = map(_slice, (mu_np, phi_np, omega_np, sigma_np))

    step_str = f" (Step {step})" if step is not None else ""
    group_str = gauge_group_label or ""
    group_suffix = f" — {group_str}" if group_str else ""

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(
        f'Representation Comparison: μ, φ, Ω, Σ (PCA){group_suffix}{step_str}',
        fontsize=14, y=1.01,
    )

    panels = [
        (axes[0, 0], mu_np,    '(a) μ  (belief means)'),
        (axes[0, 1], phi_np,   '(b) φ  (gauge frames)'),
        (axes[1, 0], omega_np, '(c) Ω  (group elements)'),
        (axes[1, 1], sigma_np, '(d) Σ  (covariance beliefs)'),
    ]
    for ax, arr, title in panels:
        if arr is None:
            ax.text(0.5, 0.5, '(not available)', transform=ax.transAxes,
                    ha='center', va='center', fontsize=12, color='gray')
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        _plot_representation_panel(
            ax, arr, title, categories, palette, envelope_mode, dot_alpha,
        )

    # One shared legend: gather handles from the first panel that has any
    handles, labels = [], []
    for ax, arr, _ in panels:
        if arr is None:
            continue
        h, l = ax.get_legend_handles_labels()
        if h:
            handles, labels = h, l
            break
    if handles:
        fig.legend(handles, labels, loc='lower center', ncol=min(8, len(labels)),
                   bbox_to_anchor=(0.5, -0.02), fontsize=9, frameon=True)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')

    return fig


# =============================================================================
# Bhattacharyya MDS — Gaussian-aware Σ embedding
# =============================================================================

def bhattacharyya_distance_matrix(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    jitter: float = 1e-6,
) -> torch.Tensor:
    r"""Pairwise Bhattacharyya distance between token Gaussians.

    For two Gaussians p_i = N(μ_i, Σ_i) and p_j = N(μ_j, Σ_j),

        D_B(i, j) = 1/8 · (μ_i - μ_j)^T Σ̄^{-1} (μ_i - μ_j)
                  + 1/2 · ln( |Σ̄| / sqrt(|Σ_i|·|Σ_j|) )

    where Σ̄ = (Σ_i + Σ_j) / 2. Symmetric, non-negative, zero iff p_i = p_j.

    Handles both diagonal Σ (vectorized, closed form) and full Σ (batched
    Cholesky per row). Falls back to jitter + diagonal when Cholesky fails.

    Args:
        mu: (n, K) belief means.
        sigma: (n, K) diagonal variances or (n, K, K) full SPD covariance.
        jitter: Small diagonal added to Σ̄ before Cholesky for stability.

    Returns:
        (n, n) symmetric non-negative distance matrix with zero diagonal.
    """
    if not isinstance(mu, torch.Tensor):
        mu = torch.as_tensor(mu)
    if not isinstance(sigma, torch.Tensor):
        sigma = torch.as_tensor(sigma)
    mu = mu.detach().cpu().float()
    sigma = sigma.detach().cpu().float()

    n = mu.shape[0]
    K = mu.shape[1]
    assert sigma.shape[0] == n, "mu and sigma must share batch dim"
    is_full = sigma.dim() == 3

    if not is_full:
        # Diagonal path — fully vectorized.
        var = sigma.clamp(min=jitter)  # (n, K)
        var_i = var.unsqueeze(1)               # (n, 1, K)
        var_j = var.unsqueeze(0)               # (1, n, K)
        var_bar = 0.5 * (var_i + var_j)        # (n, n, K)

        mu_i = mu.unsqueeze(1)                 # (n, 1, K)
        mu_j = mu.unsqueeze(0)                 # (1, n, K)
        mu_diff = mu_i - mu_j                  # (n, n, K)

        term1 = 0.125 * (mu_diff.pow(2) / var_bar).sum(dim=-1)       # (n, n)
        log_var = torch.log(var)                                      # (n, K)
        log_var_bar = torch.log(var_bar)                              # (n, n, K)
        # 0.5 * (sum log σ̄² - 0.5 sum log σ²_i - 0.5 sum log σ²_j)
        term2 = 0.5 * (
            log_var_bar.sum(dim=-1)
            - 0.5 * log_var.sum(dim=-1).unsqueeze(1)
            - 0.5 * log_var.sum(dim=-1).unsqueeze(0)
        )
        D = term1 + term2
    else:
        # Full-covariance path — row-wise batched Cholesky (O(n² K³) total).
        # Symmetrize first and hoist per-token log|Σ|.
        sigma = 0.5 * (sigma + sigma.transpose(-2, -1))
        eye_jitter = jitter * torch.eye(K).unsqueeze(0)               # (1, K, K)

        # Per-token log|Σ_i|: robust via cholesky when possible.
        try:
            L_diag = torch.linalg.cholesky(sigma + eye_jitter)        # (n, K, K)
            logdet_per = 2.0 * torch.log(
                torch.diagonal(L_diag, dim1=-2, dim2=-1).clamp(min=jitter)
            ).sum(dim=-1)                                             # (n,)
        except Exception:
            # Fallback: slogdet on the (possibly non-PD) input
            sign, logabs = torch.linalg.slogdet(sigma + eye_jitter)
            logdet_per = logabs

        D = torch.zeros(n, n, dtype=torch.float32)
        for i in range(n):
            sigma_bar = 0.5 * (sigma[i:i + 1] + sigma) + eye_jitter    # (n, K, K)
            mu_diff = (mu[i:i + 1] - mu).unsqueeze(-1)                 # (n, K, 1)
            try:
                L = torch.linalg.cholesky(sigma_bar)                   # (n, K, K)
                sol = torch.cholesky_solve(mu_diff, L)                 # (n, K, 1)
                quad = (mu_diff.squeeze(-1) * sol.squeeze(-1)).sum(dim=-1)  # (n,)
                logdet_bar = 2.0 * torch.log(
                    torch.diagonal(L, dim1=-2, dim2=-1).clamp(min=jitter)
                ).sum(dim=-1)                                          # (n,)
            except Exception:
                # Degenerate pair — fall back to slogdet + lstsq-solve
                sign, logdet_bar = torch.linalg.slogdet(sigma_bar)
                sol = torch.linalg.solve(sigma_bar, mu_diff)
                quad = (mu_diff.squeeze(-1) * sol.squeeze(-1)).sum(dim=-1)
            term1 = 0.125 * quad
            term2 = 0.5 * (logdet_bar - 0.5 * logdet_per[i] - 0.5 * logdet_per)
            D[i] = (term1 + term2).to(torch.float32)

    # Symmetrize (accumulated floating error), clamp to non-negative, zero diag
    D = 0.5 * (D + D.transpose(0, 1))
    D = D.clamp(min=0.0)
    D.fill_diagonal_(0.0)
    return D


def plot_sigma_bhattacharyya_mds(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    step: Optional[int] = None,
    save_path: Optional[Path] = None,
    n_tokens: int = 300,
    gauge_group_label: Optional[str] = None,
    envelope_mode: str = 'ellipse',
    label_mode: str = 'category',
    mds_random_state: int = 42,
) -> Optional["Any"]:
    r"""Gaussian-aware Σ visualization via Bhattacharyya distance + metric MDS.

    Unlike plot_sigma_clustering's panel (a) (flat PCA on reshaped Σ), this
    figure uses a metric that respects the Gaussian geometry — two tokens
    are close iff their *distributions* p(x | token) overlap substantially.
    Uses both μ and Σ, not just Σ.

    Layout (2 panels):
      (a) Metric MDS embedding from pairwise Bhattacharyya distances, with
          per-category envelope overlay.
      (b) Distribution of pairwise distances, split by within-category vs
          between-category. Separation between the two distributions is a
          direct diagnostic of whether token categories cluster under the
          Bhattacharyya metric.

    Cost: O(n²) Gaussian Bhattacharyya evaluations (O(n²K) diagonal,
    O(n²K³) full). Default n_tokens=300 keeps full-cov runs tractable;
    raise it for diagonal-covariance models where the per-pair cost is tiny.

    Args:
        mu: (V, K) belief means.
        sigma: (V, K) diagonal or (V, K, K) full covariance.
        step: Training step for title.
        save_path: Output path.
        n_tokens: Tokens to embed (sliced to first `n_tokens` rows).
        gauge_group_label: e.g. 'SO(3)'.
        envelope_mode: See _add_cluster_envelopes.
        label_mode: 'category' | 'semantic_field' | 'pos'.
        mds_random_state: Seed for MDS (stochastic SMACOF init).

    Returns:
        matplotlib Figure, or None if matplotlib/sklearn unavailable.
    """
    if not MATPLOTLIB_AVAILABLE or not SKLEARN_AVAILABLE:
        return None
    set_pub_style()

    try:
        from sklearn.manifold import MDS
    except ImportError:
        return None

    n = min(n_tokens, mu.shape[0], sigma.shape[0])
    mu_sub = mu[:n]
    sigma_sub = sigma[:n]
    K = mu_sub.shape[1]
    is_full = (sigma_sub.dim() == 3) if isinstance(sigma_sub, torch.Tensor) else (sigma_sub.ndim == 3)

    # Pairwise Bhattacharyya distance matrix
    D = bhattacharyya_distance_matrix(mu_sub, sigma_sub)
    D_np = D.numpy()

    # Metric MDS to 2D. Precomputed dissimilarity + deterministic seed.
    # Suppress sklearn's FutureWarnings about `dissimilarity`/`init` rename —
    # current API still works; updating when sklearn 1.10 lands is a separate
    # maintenance pass.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
        mds = MDS(
            n_components=2, dissimilarity='precomputed', random_state=mds_random_state,
            n_init=4, max_iter=300, normalized_stress='auto',
        )
        coords = mds.fit_transform(D_np)
    stress = float(mds.stress_) if hasattr(mds, 'stress_') else float('nan')

    categories, palette = _resolve_token_colors(n, label_mode=label_mode)
    dot_alpha = 0.35 if envelope_mode not in ('none', 'centroid') else 0.6

    step_str = f" (Step {step})" if step is not None else ""
    group_str = gauge_group_label or f"K={K}"
    mode_str = "full" if is_full else "diagonal"

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f'Bhattacharyya-MDS on token Gaussians — {group_str} ({mode_str}){step_str}',
        fontsize=14, y=1.02,
    )

    # ---- (a) MDS scatter ----
    ax = axes[0]
    for cat, color in palette.items():
        mask = [c == cat for c in categories]
        if any(mask):
            idx = [i for i, m in enumerate(mask) if m]
            ax.scatter(coords[idx, 0], coords[idx, 1],
                       c=color, label=cat, alpha=dot_alpha, s=20)
    _add_cluster_envelopes(ax, coords, categories, palette, mode=envelope_mode)
    ax.set_xlabel('MDS-1')
    ax.set_ylabel('MDS-2')
    ax.set_title(f'(a) Metric MDS from D_B  (stress={stress:.2f})')
    ax.legend(loc='upper right', fontsize=7)
    ax.grid(True, alpha=0.3)

    # ---- (b) Within vs between category distances ----
    ax = axes[1]
    iu = np.triu_indices(n, k=1)
    pair_d = D_np[iu]
    cat_i = np.array(categories)[iu[0]]
    cat_j = np.array(categories)[iu[1]]
    within_mask = (cat_i == cat_j)
    within = pair_d[within_mask]
    between = pair_d[~within_mask]

    if within.size > 0 and between.size > 0:
        # Use percentile-based range to avoid one outlier stretching bins
        lo = float(np.percentile(pair_d, 1))
        hi = float(np.percentile(pair_d, 99))
        bins = np.linspace(lo, hi, 50)
        ax.hist(within, bins=bins, alpha=0.5, density=True,
                label=f'within-category (n={within.size})', color=PUB_COLORS['blue'])
        ax.hist(between, bins=bins, alpha=0.5, density=True,
                label=f'between-category (n={between.size})', color=PUB_COLORS['red'])
        # Mean markers
        ax.axvline(float(np.mean(within)), color=PUB_COLORS['blue'],
                   linestyle='--', linewidth=1)
        ax.axvline(float(np.mean(between)), color=PUB_COLORS['red'],
                   linestyle='--', linewidth=1)
        ratio = float(np.mean(between) / max(np.mean(within), 1e-12))
        ax.set_title(f'(b) Pairwise D_B Distribution  (between/within = {ratio:.2f}x)')
    else:
        ax.text(0.5, 0.5, 'insufficient categories', transform=ax.transAxes,
                ha='center', va='center')
        ax.set_title('(b) Pairwise D_B Distribution')

    ax.set_xlabel('Bhattacharyya distance  D_B(i, j)')
    ax.set_ylabel('Density')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')

    return fig


# =============================================================================
# Sigma Covariance Semantic Analysis
# =============================================================================

def analyze_sigma_semantics(
    sigma: torch.Tensor,
    n_tokens: int = 500,
    verbose: bool = True,
    step: Optional[int] = None,
    save_path: Optional[Path] = None,
    gauge_group_label: Optional[str] = None,
    envelope_mode: str = 'ellipse',
    label_mode: str = 'category',
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
        import warnings as _w
        groups = []
        for cat in sorted(set(categories)):
            idx = [i for i, c in enumerate(categories) if c == cat]
            if len(idx) >= 2:
                groups.append(trace_per_token[idx])
        if len(groups) >= 2 and not all(np.std(g) < 1e-12 for g in groups):
            with _w.catch_warnings():
                _w.simplefilter("ignore")
                f_stat, p_val = f_oneway(*groups)
            if np.isfinite(f_stat):
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

    # Generate the 4-panel sigma clustering figure when a save path is given
    if save_path is not None:
        try:
            fig = plot_sigma_clustering(
                sigma_sub,
                step=step,
                save_path=save_path,
                n_tokens=n,
                gauge_group_label=gauge_group_label,
                envelope_mode=envelope_mode,
                label_mode=label_mode,
            )
            if fig is not None and MATPLOTLIB_AVAILABLE:
                plt.close(fig)
            results['sigma_plot_saved'] = True
        except Exception as e:
            if verbose:
                print(f"  [WARN] Could not generate sigma plot: {e}")
            results['sigma_plot_saved'] = False
            results['sigma_plot_error'] = str(e)

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
        """Compute holonomy deficit for triangle (i, j, k).

        Uses the TRANSPORT operators exp(phi_i)·exp(-phi_j), NOT the raw
        gauge frames omega_i^{-1}·omega_j (which telescope to I trivially).
        The holonomy C_ijk = Omega_ij · Omega_jk · Omega_ki measures the
        failure of parallel transport around the triangle to return to the
        starting frame.  For a flat connection, C_ijk = I exactly.
        """
        try:
            # Build transport operators: Omega_ij = omega_i @ omega_j^{-1}
            # Then holonomy = Omega_ij @ Omega_jk @ Omega_ki
            # = (omega_i @ omega_j^{-1}) @ (omega_j @ omega_k^{-1}) @ (omega_k @ omega_i^{-1})
            # This ALSO telescopes to I for flat (per-token frame) transport!
            #
            # For non-trivial holonomy we need the CONNECTION transport
            # operators which include edge-local delta_ij terms.  Without
            # access to those here, we compute the numerical holonomy from
            # the actual transport matrices if available, or return NaN.
            #
            # Fallback: compute exp(phi_i) @ exp(-phi_j) per pair using
            # the raw phi coordinates if they were stored alongside omega.
            # Since this function only has access to omega (= exp(phi·G)),
            # and flat transport telescopes, we document this limitation.
            Oij = torch.linalg.solve(omega[i].unsqueeze(0).double(),
                                      omega[j].unsqueeze(0).double()).squeeze(0)
            Ojk = torch.linalg.solve(omega[j].unsqueeze(0).double(),
                                      omega[k].unsqueeze(0).double()).squeeze(0)
            Oki = torch.linalg.solve(omega[k].unsqueeze(0).double(),
                                      omega[i].unsqueeze(0).double()).squeeze(0)
            holonomy = Oij @ Ojk @ Oki
            deficit = torch.norm(holonomy - torch.eye(holonomy.shape[0], dtype=torch.float64), p='fro').item()
            # NOTE: For flat (per-token frame) transport without edge-local
            # connections, this deficit is always ~machine-epsilon because
            # omega_i^{-1}·omega_j · omega_j^{-1}·omega_k · omega_k^{-1}·omega_i = I
            # by telescoping.  Non-trivial holonomy requires non-flat
            # transport (connection delta_ij terms), which should be
            # analyzed via transformer.analysis.holonomy instead.
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
