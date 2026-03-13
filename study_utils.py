"""
Shared utilities for holonomy study scripts.
=============================================

Extracts the ~600 lines of identical code shared between
run_metaphor_study.py, run_idiom_study.py, and run_holonomy_study.py.

Each study script defines a StudyConfig and calls run_study().
"""

import sys
import os
import time
import subprocess
import json
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional
from pathlib import Path

# ── Dependencies ──────────────────────────────────────────────────────────

REQUIRED = ['torch', 'transformers', 'scipy', 'matplotlib', 'numpy']


def check_deps():
    missing = []
    for pkg in REQUIRED:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"Installing missing packages: {', '.join(missing)}")
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'install', '--quiet'] + missing
        )


def find_project_root():
    """Walk up from this file to find project root (contains analysis/holonomy_study/)."""
    _here = Path(__file__).resolve().parent
    for _ancestor in [_here] + list(_here.parents):
        if (_ancestor / 'analysis' / 'holonomy_study' / '__init__.py').exists():
            return _ancestor
    return _here


check_deps()

import torch
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = find_project_root()
sys.path.insert(0, str(ROOT))

# Lazy imports: analysis.holonomy_study may not be installed in all environments.
# These are imported inside run_study() at runtime.
load_model = None
attention_flow_asymmetry = None
layerwise_jacobian_holonomy = None
discrete_curvature = None
HolonomyResult = None


def _import_analysis():
    """Import analysis modules (deferred to runtime)."""
    global load_model, attention_flow_asymmetry, layerwise_jacobian_holonomy
    global discrete_curvature, HolonomyResult
    from analysis.holonomy_study.transport import (
        load_model as _load_model,
        attention_flow_asymmetry as _attention_flow_asymmetry,
        layerwise_jacobian_holonomy as _layerwise_jacobian_holonomy,
        discrete_curvature as _discrete_curvature,
    )
    from analysis.holonomy_study.holonomy import HolonomyResult as _HolonomyResult
    load_model = _load_model
    attention_flow_asymmetry = _attention_flow_asymmetry
    layerwise_jacobian_holonomy = _layerwise_jacobian_holonomy
    discrete_curvature = _discrete_curvature
    HolonomyResult = _HolonomyResult


# ── Study Configuration ──────────────────────────────────────────────────

@dataclass
class StudyConfig:
    """Configuration for a holonomy study."""
    # Labels: phenomenon_label is 'metaphorical', 'idiomatic', or 'ironic'
    phenomenon_label: str  # The non-literal, non-control label
    # Short name for the phenomenon (used in print statements)
    phenomenon_short: str  # e.g. 'met', 'idiom', 'ironic'
    # Dataset loader: returns list of SentencePair objects
    load_pairs: Callable
    # by_label and get_paired_only from the dataset module
    by_label: Callable
    get_paired_only: Callable
    # How to get the phrase from a SentencePair
    get_phrase: Callable  # e.g. lambda sp: sp.metaphor
    # Colors for plotting
    colors: Dict[str, str]
    # Output
    output_dir: Path
    result_filename: str
    dataset_name: str
    # Synthesis text (study-specific interpretation printed at end)
    synthesis_lines: List[str] = field(default_factory=list)
    # Model
    model_name: str = 'gpt2'
    device: str = field(default_factory=lambda: 'cuda' if torch.cuda.is_available() else 'cpu')
    max_tri: int = 300
    max_pairs: int = 150


# ── Helpers ──────────────────────────────────────────────────────────────

def hbar(text='', width=60):
    if text:
        pad = width - len(text) - 2
        print(f"\n{'='*(pad//2)} {text} {'='*(pad - pad//2)}")
    else:
        print('=' * width)


def tokenize(tokenizer, text, device):
    return torch.tensor([tokenizer.encode(text)], device=device)


def fmt_p(p):
    if p < 0.001:
        return f'{p:.2e}'
    return f'{p:.4f}'


def benjamini_hochberg(p_values):
    """Benjamini-Hochberg FDR correction."""
    n = len(p_values)
    if n == 0:
        return []

    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    q_values = [0.0] * n

    min_q = 1.0
    for rank_from_end, (orig_idx, p) in enumerate(reversed(indexed)):
        rank = n - rank_from_end
        q = p * n / rank
        min_q = min(min_q, q)
        q_values[orig_idx] = min(min_q, 1.0)

    return q_values


def find_phrase_positions(tokenizer, full_text, phrase):
    """
    Find token positions of a phrase within full text.

    Uses tokenizer alignment to find exact token positions.
    Returns list of position indices or None if not found.
    """
    if not phrase or not full_text:
        return None

    full_tokens = tokenizer.encode(full_text)

    # Try exact token subsequence matching
    phrase_tokens = tokenizer.encode(phrase)
    if not phrase_tokens:
        return None

    # Remove BOS token if present
    if hasattr(tokenizer, 'bos_token_id') and phrase_tokens and phrase_tokens[0] == tokenizer.bos_token_id:
        phrase_tokens = phrase_tokens[1:]

    if not phrase_tokens:
        return None

    # Sliding window search
    plen = len(phrase_tokens)
    for i in range(len(full_tokens) - plen + 1):
        if full_tokens[i:i+plen] == phrase_tokens:
            return list(range(i, i + plen))

    # Fuzzy: try encoding phrase with a space prefix (GPT-2 tokenization quirk)
    phrase_tokens_sp = tokenizer.encode(' ' + phrase)
    if hasattr(tokenizer, 'bos_token_id') and phrase_tokens_sp and phrase_tokens_sp[0] == tokenizer.bos_token_id:
        phrase_tokens_sp = phrase_tokens_sp[1:]

    if phrase_tokens_sp:
        plen_sp = len(phrase_tokens_sp)
        for i in range(len(full_tokens) - plen_sp + 1):
            if full_tokens[i:i+plen_sp] == phrase_tokens_sp:
                return list(range(i, i + plen_sp))

    # Last resort: character-level alignment
    char_start = full_text.lower().find(phrase.lower())
    if char_start < 0:
        return None

    char_end = char_start + len(phrase)
    positions = []
    current_pos = 0
    for tok_idx, tok_id in enumerate(full_tokens):
        tok_text = tokenizer.decode([tok_id])
        tok_start = current_pos
        tok_end = current_pos + len(tok_text)

        if tok_end > char_start and tok_start < char_end:
            positions.append(tok_idx)

        current_pos = tok_end

    return positions if positions else None


def cross_boundary_positions(phrase_pos, n_tokens, context_window=3):
    """Get positions spanning phrase boundary (phrase + nearby context)."""
    phrase_set = set(phrase_pos)
    extended = set(phrase_pos)
    for p in phrase_pos:
        for offset in range(-context_window, context_window + 1):
            new_p = p + offset
            if 0 <= new_p < n_tokens:
                extended.add(new_p)
    return sorted(extended), sorted(extended - phrase_set)


# ── Plotting ─────────────────────────────────────────────────────────────

def plot_distributions(kappas_by_label, title, ylabel, output_path, colors=None):
    """Plot violin + box + strip chart for distributions by label."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    labels = [l for l in kappas_by_label if len(kappas_by_label[l]) > 0]
    data = [kappas_by_label[l] for l in labels]
    label_colors = [colors.get(l, '#999999') if colors else '#999999' for l in labels]

    ax = axes[0]
    parts = ax.violinplot(data, showmeans=True, showmedians=True)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(label_colors[i])
        pc.set_alpha(0.4)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    ax = axes[1]
    for i, (label, vals) in enumerate(zip(labels, data)):
        ax.hist(vals, bins=20, alpha=0.5, color=label_colors[i], label=label, density=True)
    ax.set_xlabel(ylabel)
    ax.set_ylabel('Density')
    ax.set_title(f'{title} (histogram)')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'  Saved: {output_path}')
    plt.close()


def plot_layer_profiles(profiles_by_label, title, output_path, n_layers=12, colors=None):
    """Plot mean +/- SEM layer profiles."""
    fig, ax = plt.subplots(figsize=(10, 5))
    layers = list(range(n_layers))

    for label, profiles in profiles_by_label.items():
        if not profiles:
            continue
        arr = np.array(profiles)
        mean = np.mean(arr, axis=0)[:n_layers]
        sem = np.std(arr, axis=0)[:n_layers] / np.sqrt(len(arr))
        color = colors.get(label, '#999999') if colors else '#999999'
        ax.plot(layers, mean, 'o-', label=f'{label} (n={len(profiles)})', color=color)
        ax.fill_between(layers, mean - sem, mean + sem, alpha=0.2, color=color)

    ax.set_xlabel('Layer (VFE Step)')
    ax.set_ylabel(r'Mean holonomy $\kappa$')
    ax.set_title(title)
    ax.legend()
    ax.set_xticks(layers)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'  Saved: {output_path}')
    plt.close()


def plot_curvature_layer_profiles(profiles_by_label, output_path, n_layers=12, colors=None):
    """Plot curvature layer profiles."""
    fig, ax = plt.subplots(figsize=(10, 5))
    layers = list(range(n_layers))

    for label, profiles in profiles_by_label.items():
        if not profiles:
            continue
        arr = np.array(profiles)
        mean = np.mean(arr, axis=0)[:n_layers]
        sem = np.std(arr, axis=0)[:n_layers] / np.sqrt(len(arr))
        color = colors.get(label, '#999999') if colors else '#999999'
        ax.plot(layers, mean, 'o-', label=f'{label} (n={len(profiles)})', color=color)
        ax.fill_between(layers, mean - sem, mean + sem, alpha=0.2, color=color)

    ax.set_xlabel('Layer')
    ax.set_ylabel('Mean curvature (commutator norm)')
    ax.set_title('Curvature by Layer')
    ax.legend()
    ax.set_xticks(layers)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'  Saved: {output_path}')
    plt.close()


def plot_paired_comparison(phenom_by_pair, lit_by_pair, shared, phrases_by_pair,
                           ylabel, title, output_path):
    """Plot paired comparison (same phrase, different usage)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    pvals = [phenom_by_pair[p] for p in shared]
    lvals = [lit_by_pair[p] for p in shared]

    for i, pid in enumerate(shared):
        ax.plot([0, 1], [pvals[i], lvals[i]], 'o-', color='grey', alpha=0.3)
    ax.plot([0], [np.mean(pvals)], 'D', markersize=12, color='red', zorder=5)
    ax.plot([1], [np.mean(lvals)], 'D', markersize=12, color='green', zorder=5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Phenomenon', 'Literal'])
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'  Saved: {output_path}')
    plt.close()


# ── Statistics ───────────────────────────────────────────────────────────

def run_stats(kappas, label_a, label_b, metric_name='kappa'):
    """Run Mann-Whitney U test between two groups."""
    if label_a not in kappas or label_b not in kappas:
        return None
    a, b = kappas[label_a], kappas[label_b]
    if len(a) < 2 or len(b) < 2:
        return None
    U, p = stats.mannwhitneyu(a, b, alternative='two-sided')
    pooled_std = np.sqrt((np.var(a) + np.var(b)) / 2)
    d = (np.mean(a) - np.mean(b)) / pooled_std if pooled_std > 0 else 0
    print(f'  {label_a} vs {label_b} ({metric_name}): '
          f'U={U:.0f}  p={fmt_p(p)}  d={d:+.3f}')
    return {'U': float(U), 'p_value': float(p), 'cohens_d': float(d)}


def run_permutation_test(a, b, label_a, label_b, metric_name='kappa', n_perm=10000, seed=42):
    """Two-sample permutation test."""
    obs_diff = np.mean(a) - np.mean(b)
    combined = np.concatenate([a, b])
    n_a = len(a)
    rng = np.random.RandomState(seed)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(len(combined))
        perm_diff = np.mean(combined[perm[:n_a]]) - np.mean(combined[perm[n_a:]])
        if abs(perm_diff) >= abs(obs_diff):
            count += 1
    p_perm = (count + 1) / (n_perm + 1)
    print(f'  {label_a} vs {label_b} ({metric_name}): '
          f'obs_diff={obs_diff:+.4f}  p_perm={fmt_p(p_perm)}')
    return p_perm


# ── Main Study Runner ────────────────────────────────────────────────────

def run_study(cfg: StudyConfig):
    """
    Run a complete holonomy study.

    This implements the shared 14-step workflow used by all three study scripts.
    """
    _import_analysis()

    t_start = time.time()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    phenom = cfg.phenomenon_label  # e.g. 'metaphorical'
    short = cfg.phenomenon_short   # e.g. 'met'
    labels_ordered = [phenom, 'literal', 'control']
    comparison_pairs = [(phenom, 'literal'), (phenom, 'control'), ('literal', 'control')]

    # ── 1. Load model ────────────────────────────────────────────────────
    hbar('Loading GPT-2')
    model, tokenizer = load_model(cfg.model_name, device=cfg.device)
    n_params = sum(p.numel() for p in model.parameters())
    n_layers = len(model.h)
    d_model = model.config.n_embd
    print(f'  Model:      {cfg.model_name}')
    print(f'  Parameters: {n_params:,}')
    print(f'  Layers:     {n_layers}')
    print(f'  Hidden dim: {d_model}')
    print(f'  Device:     {cfg.device}')

    # ── 2. Load dataset ──────────────────────────────────────────────────
    hbar(f'{phenom.title()} Dataset')
    all_pairs = cfg.load_pairs()
    groups = cfg.by_label(all_pairs)
    for label, items in groups.items():
        print(f'  {label:13s}: {len(items)} sentences')
    paired = cfg.get_paired_only(all_pairs)
    print(f'  paired:        {len(paired)} (strongest test: same phrase, different usage)')

    # ── 3. Method 0: Attention path defect ───────────────────────────────
    hbar('Method 0: Attention Path Defect')
    asymmetry_by_label = {}
    for label, items in groups.items():
        vals = []
        for sp in items:
            ids = tokenize(tokenizer, sp.text, cfg.device)
            r = attention_flow_asymmetry(model, ids)
            vals.append(r['defect_per_layer'].mean().item())
        asymmetry_by_label[label] = np.array(vals)
        print(f'  {label:13s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}')

    for la, lb in comparison_pairs:
        U, p = stats.mannwhitneyu(asymmetry_by_label[la], asymmetry_by_label[lb], alternative='two-sided')
        print(f'  {la} vs {lb}: p = {fmt_p(p)}')

    # ── 4. Layer-wise Jacobian holonomy ──────────────────────────────────
    hbar('Layer-wise Jacobian Holonomy (VFE Steps)')

    results = {l: [] for l in labels_ordered}
    layer_profiles = {l: [] for l in labels_ordered}
    total = sum(len(v) for v in groups.values())
    done = 0

    for label, items in groups.items():
        for sp in items:
            t0 = time.time()
            ids = tokenize(tokenizer, sp.text, cfg.device)
            N = ids.shape[1]
            if N < 3:
                done += 1
                continue

            plh = layerwise_jacobian_holonomy(model, ids, max_triples=cfg.max_tri)

            layer_profiles[label].append(plh['kappa_per_layer'])

            kappa_arr = plh['kappa_all']
            if kappa_arr.ndim == 2:
                kappa_arr = np.nanmean(kappa_arr, axis=0)

            phrase_val = cfg.get_phrase(sp) or ''
            hr = HolonomyResult(
                kappa=kappa_arr,
                triangles=plh['triples'],
                kappa_mean=plh['kappa_mean'],
                kappa_median=plh['kappa_median'],
                kappa_max=plh['kappa_max'],
                kappa_std=plh['kappa_std'],
                metadata={
                    'text': sp.text,
                    'label': label,
                    'pair_id': sp.pair_id,
                    'phrase': phrase_val,
                    'method': 'layerwise_jacobian',
                    'n_tokens': N,
                    'kappa_per_layer': plh['kappa_per_layer'],
                    'cos_sim_per_layer': plh['cos_sim_per_layer'],
                    'n_forward_passes': plh['n_forward_passes'],
                },
            )
            results[label].append(hr)

            done += 1
            dt = time.time() - t0
            print(f'  [{done:3d}/{total}] {label:13s} kappa={plh["kappa_mean"]:.4f}  '
                  f'({N} tok, {plh["n_forward_passes"]} fwd, {dt:.1f}s)  '
                  f'{sp.text[:50]}...')

    # ── 5. Discrete curvature ────────────────────────────────────────────
    hbar('Discrete Riemann Curvature (Transport Commutator)')

    curvature_results = {l: [] for l in labels_ordered}
    curvature_layer_profiles = {l: [] for l in labels_ordered}
    done = 0

    for label, items in groups.items():
        for sp in items:
            t0 = time.time()
            ids = tokenize(tokenizer, sp.text, cfg.device)
            N = ids.shape[1]
            if N < 3:
                done += 1
                continue

            cr = discrete_curvature(model, ids, max_triples=cfg.max_pairs)

            curvature_results[label].append(cr['curvature_mean'])
            curvature_layer_profiles[label].append(cr['curvature_per_layer'])

            done += 1
            dt = time.time() - t0
            print(f'  [{done:3d}/{total}] {label:13s} curv={cr["curvature_mean"]:.6f}  '
                  f'({N} tok, {dt:.1f}s)  {sp.text[:50]}...')

    for label in curvature_results:
        curvature_results[label] = np.array(curvature_results[label])

    # ── 6. Statistical analysis ──────────────────────────────────────────
    hbar('Statistical Analysis: Layer-wise Holonomy')

    kappas = {
        label: np.array([hr.kappa_mean for hr in hrs])
        for label, hrs in results.items() if hrs
    }

    for label in labels_ordered:
        if label in kappas:
            k = kappas[label]
            print(f'  {label:13s}: mean={np.mean(k):.4f}  median={np.median(k):.4f}  std={np.std(k):.4f}')

    stat_results = {}
    for la, lb in comparison_pairs:
        r = run_stats(kappas, la, lb, 'holonomy')
        if r:
            stat_results[f'{la}_vs_{lb}_holonomy'] = r

    hbar('Statistical Analysis: Curvature')
    for label in labels_ordered:
        if label in curvature_results and len(curvature_results[label]) > 0:
            c = curvature_results[label]
            print(f'  {label:13s}: mean={np.mean(c):.6f}  median={np.median(c):.6f}  std={np.std(c):.6f}')

    for la, lb in comparison_pairs:
        r = run_stats(curvature_results, la, lb, 'curvature')
        if r:
            stat_results[f'{la}_vs_{lb}_curvature'] = r

    # ── 7. Paired comparison ─────────────────────────────────────────────
    hbar('Paired Comparison (Same Phrase, Different Usage)')

    phenom_by_pair = {hr.metadata['pair_id']: hr.kappa_mean for hr in results[phenom]}
    lit_by_pair = {hr.metadata['pair_id']: hr.kappa_mean for hr in results['literal']}
    phrases_by_pair = {}
    for hr in results[phenom]:
        phrases_by_pair[hr.metadata['pair_id']] = hr.metadata.get('phrase', '')
    shared = sorted(set(phenom_by_pair) & set(lit_by_pair))

    n_higher = 0
    for pid in shared:
        mk, lk = phenom_by_pair[pid], lit_by_pair[pid]
        arrow = '>' if mk > lk else '<'
        tag = '*' if mk > lk else ' '
        phrase = phrases_by_pair.get(pid, '')
        print(f'  {tag} pair {pid:2d}: {short}={mk:.4f} {arrow} lit={lk:.4f}  "{phrase}"')
        if mk > lk:
            n_higher += 1

    pct = 100 * n_higher / len(shared) if shared else 0
    print(f'\n  {phenom.title()} > Literal in {n_higher}/{len(shared)} pairs ({pct:.0f}%)')

    if len(shared) >= 5:
        paired_phenom = [phenom_by_pair[p] for p in shared]
        paired_lit = [lit_by_pair[p] for p in shared]
        stat_w, p_w = stats.wilcoxon(paired_phenom, paired_lit, alternative='two-sided')
        print(f'  Wilcoxon signed-rank: W={stat_w:.0f}, p={fmt_p(p_w)}')
        stat_results['paired_wilcoxon'] = {'W': float(stat_w), 'p_value': float(p_w)}

    # Paired curvature
    curv_phenom_by_pair = {}
    curv_lit_by_pair = {}
    for idx, sp in enumerate(groups[phenom]):
        if idx < len(curvature_results[phenom]):
            curv_phenom_by_pair[sp.pair_id] = curvature_results[phenom][idx]
    for idx, sp in enumerate(groups['literal']):
        if idx < len(curvature_results['literal']):
            curv_lit_by_pair[sp.pair_id] = curvature_results['literal'][idx]

    shared_curv = sorted(set(curv_phenom_by_pair) & set(curv_lit_by_pair))
    if len(shared_curv) >= 5:
        pc_phenom = [curv_phenom_by_pair[p] for p in shared_curv]
        pc_lit = [curv_lit_by_pair[p] for p in shared_curv]
        n_higher_c = sum(1 for m, l in zip(pc_phenom, pc_lit) if m > l)
        pct_c = 100 * n_higher_c / len(shared_curv)
        print(f'\n  Curvature: {phenom.title()} > Literal in {n_higher_c}/{len(shared_curv)} pairs ({pct_c:.0f}%)')
        stat_w_c, p_w_c = stats.wilcoxon(pc_phenom, pc_lit, alternative='two-sided')
        print(f'  Curvature Wilcoxon: W={stat_w_c:.0f}, p={fmt_p(p_w_c)}')

    # ── 8. Permutation tests ─────────────────────────────────────────────
    hbar('Permutation Tests (10,000 shuffles)')
    for la, lb in comparison_pairs:
        if la in kappas and lb in kappas:
            run_permutation_test(kappas[la], kappas[lb], la, lb, 'holonomy')
        if la in curvature_results and lb in curvature_results:
            if len(curvature_results[la]) > 0 and len(curvature_results[lb]) > 0:
                run_permutation_test(curvature_results[la], curvature_results[lb], la, lb, 'curvature')

    if len(shared) >= 5:
        paired_diffs = np.array(paired_phenom) - np.array(paired_lit)
        obs_mean_diff = np.mean(paired_diffs)
        rng = np.random.RandomState(42)
        count = 0
        for _ in range(10000):
            signs = rng.choice([-1, 1], size=len(paired_diffs))
            if abs(np.mean(paired_diffs * signs)) >= abs(obs_mean_diff):
                count += 1
        p_pp = (count + 1) / 10001
        print(f'  paired sign-flip (holonomy): obs_diff={obs_mean_diff:+.4f}  p_perm={fmt_p(p_pp)}')

    # ── 9. Length-controlled analysis ─────────────────────────────────────
    hbar('Length-Controlled Analysis')

    all_k, all_len, all_lab = [], [], []
    for label, hrs in results.items():
        for hr in hrs:
            all_k.append(hr.kappa_mean)
            all_len.append(hr.metadata['n_tokens'])
            all_lab.append(label)

    all_k = np.array(all_k)
    all_len = np.array(all_len, dtype=float)
    all_lab = np.array(all_lab)

    for label in labels_ordered:
        mask = all_lab == label
        if mask.any():
            print(f'  {label:13s}: mean_len={np.mean(all_len[mask]):.1f}  mean_kappa={np.mean(all_k[mask]):.4f}')

    r_len, p_len = stats.pearsonr(all_len, all_k)
    print(f'  kappa ~ length: r={r_len:+.3f}  p={fmt_p(p_len)}')

    slope, intercept = np.polyfit(all_len, all_k, 1)
    residuals = all_k - (slope * all_len + intercept)
    print(f'  Regression: kappa = {slope:+.5f} * n_tokens + {intercept:.4f}')

    resid_by_label = {}
    for label in labels_ordered:
        mask = all_lab == label
        if mask.any():
            resid_by_label[label] = residuals[mask]
            print(f'  {label:13s} residual: mean={np.mean(residuals[mask]):+.4f}  std={np.std(residuals[mask]):.4f}')

    print('\n  Length-controlled Mann-Whitney (on residuals):')
    for la, lb in comparison_pairs:
        if la in resid_by_label and lb in resid_by_label:
            a, b = resid_by_label[la], resid_by_label[lb]
            U, p = stats.mannwhitneyu(a, b, alternative='two-sided')
            d_val = (np.mean(a) - np.mean(b)) / np.sqrt((np.var(a) + np.var(b)) / 2) if (np.var(a) + np.var(b)) > 0 else 0
            print(f'    {la} vs {lb}: U={U:.0f}  p={fmt_p(p)}  d={d_val:+.3f}')

    # ── 10. Bootstrap CIs ────────────────────────────────────────────────
    hbar(f'Bootstrap CIs (Paired {phenom.title()} - Literal)')
    N_BOOT = 10_000
    rng = np.random.RandomState(42)

    if len(shared) >= 5:
        paired_diffs = np.array(paired_phenom) - np.array(paired_lit)
        boot_means = np.zeros(N_BOOT)
        for i in range(N_BOOT):
            sample = rng.choice(paired_diffs, size=len(paired_diffs), replace=True)
            boot_means[i] = np.mean(sample)
        ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
        boot_p = 2 * min(np.mean(boot_means <= 0), np.mean(boot_means >= 0))
        print(f'  Mean paired diff: {np.mean(paired_diffs):+.4f}')
        print(f'  95% CI:           [{ci_lo:+.4f}, {ci_hi:+.4f}]')
        print(f'  Bootstrap p:      {fmt_p(boot_p)}')
        print(f'  CI excludes 0:    {"YES" if ci_lo > 0 or ci_hi < 0 else "NO"}')

    # ── 11. Per-layer analysis ───────────────────────────────────────────
    hbar('Per-Layer Analysis (Where Does the Effect Emerge?)')

    layer_pvals = []
    layer_rows = []
    for l in range(n_layers):
        kl_phenom = [prof[l] for prof in layer_profiles[phenom] if len(prof) > l]
        kl_lit = [prof[l] for prof in layer_profiles['literal'] if len(prof) > l]
        if len(kl_phenom) >= 2 and len(kl_lit) >= 2:
            U, p = stats.mannwhitneyu(kl_phenom, kl_lit, alternative='two-sided')
            d_mean = np.mean(kl_phenom) - np.mean(kl_lit)
            layer_pvals.append(p)
            layer_rows.append((l, np.mean(kl_phenom), np.mean(kl_lit), d_mean, p))
        else:
            layer_pvals.append(1.0)
            layer_rows.append((l, 0, 0, 0, 1.0))

    layer_qvals = benjamini_hochberg(layer_pvals)
    for (l, mp, ml, d_mean, p), q in zip(layer_rows, layer_qvals):
        sig = '*' if q < 0.05 else ' '
        print(f'  Layer {l:2d}: {short}={mp:.4f}  '
              f'lit={ml:.4f}  diff={d_mean:+.4f}  p={p:.4f}  q={q:.4f} {sig}')

    # ── 12. Phrase-localized holonomy ────────────────────────────────────
    hbar('Phrase-Localized Holonomy (Length Control)')
    print(f'  Measuring holonomy ONLY on {phenom}-phrase tokens.')
    print('  Same phrase in both contexts → perfect length control.')
    print('  Full sentence context preserved (model sees everything).')
    print()

    loc_phenom_by_pair = {}
    loc_lit_by_pair = {}
    loc_curv_phenom_by_pair = {}
    loc_curv_lit_by_pair = {}
    loc_layer_profiles = {phenom: [], 'literal': []}
    loc_curv_layer_profiles = {phenom: [], 'literal': []}

    pair_map = {}
    for sp in all_pairs:
        phrase_val = cfg.get_phrase(sp)
        if sp.label in (phenom, 'literal') and phrase_val:
            pair_map.setdefault(sp.pair_id, {})[sp.label] = sp

    n_found = 0
    n_skip = 0
    for pid in sorted(pair_map.keys()):
        if phenom not in pair_map[pid] or 'literal' not in pair_map[pid]:
            continue
        sp_p = pair_map[pid][phenom]
        sp_l = pair_map[pid]['literal']
        phrase = cfg.get_phrase(sp_p)

        ids_p = tokenize(tokenizer, sp_p.text, cfg.device)
        ids_l = tokenize(tokenizer, sp_l.text, cfg.device)

        pos_p = find_phrase_positions(tokenizer, sp_p.text, phrase)
        pos_l = find_phrase_positions(tokenizer, sp_l.text, phrase)

        if pos_p is None or pos_l is None or len(pos_p) < 3 or len(pos_l) < 3:
            n_skip += 1
            print(f'  SKIP pair {pid}: phrase "{phrase}" — '
                  f'pos_p={pos_p}, pos_l={pos_l}')
            continue

        n_found += 1
        t0 = time.time()

        plh_p = layerwise_jacobian_holonomy(model, ids_p, max_triples=cfg.max_tri, positions=pos_p)
        plh_l = layerwise_jacobian_holonomy(model, ids_l, max_triples=cfg.max_tri, positions=pos_l)

        loc_phenom_by_pair[pid] = plh_p['kappa_mean']
        loc_lit_by_pair[pid] = plh_l['kappa_mean']
        loc_layer_profiles[phenom].append(plh_p['kappa_per_layer'])
        loc_layer_profiles['literal'].append(plh_l['kappa_per_layer'])

        cr_p = discrete_curvature(model, ids_p, max_triples=cfg.max_pairs, positions=pos_p)
        cr_l = discrete_curvature(model, ids_l, max_triples=cfg.max_pairs, positions=pos_l)

        loc_curv_phenom_by_pair[pid] = cr_p['curvature_mean']
        loc_curv_lit_by_pair[pid] = cr_l['curvature_mean']
        loc_curv_layer_profiles[phenom].append(cr_p['curvature_per_layer'])
        loc_curv_layer_profiles['literal'].append(cr_l['curvature_per_layer'])

        dt = time.time() - t0
        arrow_h = '>' if plh_p['kappa_mean'] > plh_l['kappa_mean'] else '<'
        arrow_c = '>' if cr_p['curvature_mean'] > cr_l['curvature_mean'] else '<'
        print(f'  pair {pid:2d} "{phrase}": '
              f'hol {plh_p["kappa_mean"]:.4f} {arrow_h} {plh_l["kappa_mean"]:.4f}  '
              f'curv {cr_p["curvature_mean"]:.5f} {arrow_c} {cr_l["curvature_mean"]:.5f}  '
              f'({len(pos_p)}/{len(pos_l)} tok, {dt:.1f}s)')

    print(f'\n  Found phrase positions: {n_found} pairs  (skipped: {n_skip})')

    # ── 12a. Phrase-localized statistics ─────────────────────────────────
    loc_shared = sorted(set(loc_phenom_by_pair) & set(loc_lit_by_pair))
    if len(loc_shared) >= 5:
        hbar('Phrase-Localized Statistics')

        loc_pp = np.array([loc_phenom_by_pair[p] for p in loc_shared])
        loc_pl = np.array([loc_lit_by_pair[p] for p in loc_shared])
        loc_diffs = loc_pp - loc_pl

        n_higher_loc = int(np.sum(loc_pp > loc_pl))
        pct_loc = 100 * n_higher_loc / len(loc_shared)
        print(f'  Holonomy: {phenom.title()} > Literal in {n_higher_loc}/{len(loc_shared)} pairs ({pct_loc:.0f}%)')
        print(f'  Mean {short}: {np.mean(loc_pp):.4f}  literal: {np.mean(loc_pl):.4f}  '
              f'diff: {np.mean(loc_diffs):+.4f}')

        stat_w, p_w = stats.wilcoxon(loc_pp, loc_pl, alternative='two-sided')
        print(f'  Wilcoxon signed-rank: W={stat_w:.0f}, p={fmt_p(p_w)}')

        d_paired = np.mean(loc_diffs) / np.std(loc_diffs) if np.std(loc_diffs) > 0 else 0
        print(f'  Paired d: {d_paired:+.3f}')

        U_loc, p_loc = stats.mannwhitneyu(loc_pp, loc_pl, alternative='two-sided')
        pooled_std = np.sqrt((np.var(loc_pp) + np.var(loc_pl)) / 2)
        d_loc = (np.mean(loc_pp) - np.mean(loc_pl)) / pooled_std if pooled_std > 0 else 0
        print(f'  Mann-Whitney U={U_loc:.0f}, p={fmt_p(p_loc)}, d={d_loc:+.3f}')

        rng_loc = np.random.RandomState(42)
        boot_loc = np.zeros(10000)
        for bi in range(10000):
            sample = rng_loc.choice(loc_diffs, size=len(loc_diffs), replace=True)
            boot_loc[bi] = np.mean(sample)
        ci_lo_loc, ci_hi_loc = np.percentile(boot_loc, [2.5, 97.5])
        boot_p_loc = 2 * min(np.mean(boot_loc <= 0), np.mean(boot_loc >= 0))
        print(f'  Bootstrap 95% CI: [{ci_lo_loc:+.4f}, {ci_hi_loc:+.4f}]')
        print(f'  Bootstrap p: {fmt_p(boot_p_loc)}')
        print(f'  CI excludes 0: {"YES" if ci_lo_loc > 0 or ci_hi_loc < 0 else "NO"}')

        obs_mean = np.mean(loc_diffs)
        rng_perm = np.random.RandomState(42)
        count_perm = 0
        for _ in range(10000):
            signs = rng_perm.choice([-1, 1], size=len(loc_diffs))
            if abs(np.mean(loc_diffs * signs)) >= abs(obs_mean):
                count_perm += 1
        p_perm_loc = (count_perm + 1) / 10001
        print(f'  Paired sign-flip permutation: obs_diff={obs_mean:+.4f}  p={fmt_p(p_perm_loc)}')

        stat_results['phrase_localized_holonomy'] = {
            'n_pairs': len(loc_shared),
            'pct_higher': float(pct_loc),
            'wilcoxon_W': float(stat_w),
            'wilcoxon_p': float(p_w),
            'paired_d': float(d_paired),
            'mann_whitney_U': float(U_loc),
            'mann_whitney_p': float(p_loc),
            'cohens_d': float(d_loc),
            'bootstrap_ci': [float(ci_lo_loc), float(ci_hi_loc)],
            'bootstrap_p': float(boot_p_loc),
            'permutation_p': float(p_perm_loc),
        }

    # Phrase-localized curvature
    loc_shared_c = sorted(set(loc_curv_phenom_by_pair) & set(loc_curv_lit_by_pair))
    if len(loc_shared_c) >= 5:
        loc_cp = np.array([loc_curv_phenom_by_pair[p] for p in loc_shared_c])
        loc_cl = np.array([loc_curv_lit_by_pair[p] for p in loc_shared_c])
        fm = np.isfinite(loc_cp) & np.isfinite(loc_cl)
        loc_cp_f, loc_cl_f = loc_cp[fm], loc_cl[fm]
        if len(loc_cp_f) >= 5:
            n_hc = int(np.sum(loc_cp_f > loc_cl_f))
            pct_hc = 100 * n_hc / len(loc_cp_f)
            print(f'\n  Phrase-localized curvature: {phenom.title()} > Lit in {n_hc}/{len(loc_cp_f)} ({pct_hc:.0f}%)')
            sw_c, pw_c = stats.wilcoxon(loc_cp_f, loc_cl_f, alternative='two-sided')
            d_c = np.mean(loc_cp_f - loc_cl_f) / np.std(loc_cp_f - loc_cl_f) if np.std(loc_cp_f - loc_cl_f) > 0 else 0
            print(f'  Wilcoxon: W={sw_c:.0f}, p={fmt_p(pw_c)}, paired d={d_c:+.3f}')
            stat_results['phrase_localized_curvature'] = {
                'n_pairs': int(len(loc_cp_f)),
                'pct_higher': float(pct_hc),
                'wilcoxon_W': float(sw_c),
                'wilcoxon_p': float(pw_c),
                'paired_d': float(d_c),
            }

    # ── 12b. Phrase-localized per-layer ─────────────────────────────────
    if loc_layer_profiles[phenom] and loc_layer_profiles['literal']:
        hbar('Phrase-Localized Per-Layer Analysis')
        pl_pvals = []
        pl_rows = []
        for l in range(n_layers):
            kl_p = [prof[l] for prof in loc_layer_profiles[phenom] if len(prof) > l]
            kl_l = [prof[l] for prof in loc_layer_profiles['literal'] if len(prof) > l]
            if len(kl_p) >= 2 and len(kl_l) >= 2:
                U_l, p_l = stats.mannwhitneyu(kl_p, kl_l, alternative='two-sided')
                d_mean_l = np.mean(kl_p) - np.mean(kl_l)
                pl_pvals.append(p_l)
                pl_rows.append((l, np.mean(kl_p), np.mean(kl_l), d_mean_l, p_l))
            else:
                pl_pvals.append(1.0)
                pl_rows.append((l, 0, 0, 0, 1.0))

        pl_qvals = benjamini_hochberg(pl_pvals)
        for (l, mp, ml, d_mean_l, p_l), q_l in zip(pl_rows, pl_qvals):
            sig = '*' if q_l < 0.05 else ' '
            print(f'  Layer {l:2d}: {short}={mp:.4f}  '
                  f'lit={ml:.4f}  diff={d_mean_l:+.4f}  p={p_l:.4f}  q={q_l:.4f} {sig}')

    # ── 12c. Cross-boundary holonomy ─────────────────────────────────────
    hbar('Cross-Boundary Holonomy (Phrase-Context Interaction)')
    print(f'  Measuring holonomy on {phenom} phrase + nearby context tokens.')
    print('  Triples span the phrase-context boundary.')
    print()

    CONTEXT_WINDOW = 3

    xb_phenom_by_pair = {}
    xb_lit_by_pair = {}
    xb_curv_phenom_by_pair = {}
    xb_curv_lit_by_pair = {}
    xb_layer_profiles = {phenom: [], 'literal': []}
    xb_pos_counts = {phenom: [], 'literal': []}

    n_xb_found = 0
    for pid in sorted(pair_map.keys()):
        if phenom not in pair_map[pid] or 'literal' not in pair_map[pid]:
            continue
        sp_p = pair_map[pid][phenom]
        sp_l = pair_map[pid]['literal']
        phrase = cfg.get_phrase(sp_p)

        ids_p = tokenize(tokenizer, sp_p.text, cfg.device)
        ids_l = tokenize(tokenizer, sp_l.text, cfg.device)
        N_p = ids_p.shape[1]
        N_l = ids_l.shape[1]

        pos_p = find_phrase_positions(tokenizer, sp_p.text, phrase)
        pos_l = find_phrase_positions(tokenizer, sp_l.text, phrase)

        if pos_p is None or pos_l is None or len(pos_p) < 2 or len(pos_l) < 2:
            continue

        xb_pos_p, _ = cross_boundary_positions(pos_p, N_p, CONTEXT_WINDOW)
        xb_pos_l, _ = cross_boundary_positions(pos_l, N_l, CONTEXT_WINDOW)

        if len(xb_pos_p) != len(xb_pos_l):
            target_len = min(len(xb_pos_p), len(xb_pos_l))
            if len(xb_pos_p) > target_len:
                center_p = (min(pos_p) + max(pos_p)) / 2
                xb_pos_p = sorted(sorted(xb_pos_p, key=lambda p: abs(p - center_p))[:target_len])
            if len(xb_pos_l) > target_len:
                center_l = (min(pos_l) + max(pos_l)) / 2
                xb_pos_l = sorted(sorted(xb_pos_l, key=lambda p: abs(p - center_l))[:target_len])

        if len(xb_pos_p) < 3 or len(xb_pos_l) < 3:
            continue

        n_xb_found += 1
        xb_pos_counts[phenom].append(len(xb_pos_p))
        xb_pos_counts['literal'].append(len(xb_pos_l))
        t0 = time.time()

        plh_p = layerwise_jacobian_holonomy(model, ids_p, max_triples=cfg.max_tri, positions=xb_pos_p)
        plh_l = layerwise_jacobian_holonomy(model, ids_l, max_triples=cfg.max_tri, positions=xb_pos_l)

        xb_phenom_by_pair[pid] = plh_p['kappa_mean']
        xb_lit_by_pair[pid] = plh_l['kappa_mean']
        xb_layer_profiles[phenom].append(plh_p['kappa_per_layer'])
        xb_layer_profiles['literal'].append(plh_l['kappa_per_layer'])

        cr_p = discrete_curvature(model, ids_p, max_triples=cfg.max_pairs, positions=xb_pos_p)
        cr_l = discrete_curvature(model, ids_l, max_triples=cfg.max_pairs, positions=xb_pos_l)

        xb_curv_phenom_by_pair[pid] = cr_p['curvature_mean']
        xb_curv_lit_by_pair[pid] = cr_l['curvature_mean']

        dt = time.time() - t0
        arrow_h = '>' if plh_p['kappa_mean'] > plh_l['kappa_mean'] else '<'
        arrow_c = '>' if cr_p['curvature_mean'] > cr_l['curvature_mean'] else '<'
        print(f'  pair {pid:2d}: hol {plh_p["kappa_mean"]:.4f} {arrow_h} {plh_l["kappa_mean"]:.4f}  '
              f'curv {cr_p["curvature_mean"]:.5f} {arrow_c} {cr_l["curvature_mean"]:.5f}  '
              f'({len(xb_pos_p)}/{len(xb_pos_l)} pos, {dt:.1f}s)  "{phrase}"')

    print(f'\n  Cross-boundary pairs: {n_xb_found}')

    # Cross-boundary statistics
    xb_shared = sorted(set(xb_phenom_by_pair) & set(xb_lit_by_pair))
    xb_layer_qvals = None
    if len(xb_shared) >= 5:
        hbar('Cross-Boundary Statistics')

        xb_pp = np.array([xb_phenom_by_pair[p] for p in xb_shared])
        xb_pl = np.array([xb_lit_by_pair[p] for p in xb_shared])
        xb_diffs = xb_pp - xb_pl

        n_higher_xb = int(np.sum(xb_pp > xb_pl))
        pct_xb = 100 * n_higher_xb / len(xb_shared)
        print(f'  Holonomy: {phenom.title()} > Lit in {n_higher_xb}/{len(xb_shared)} pairs ({pct_xb:.0f}%)')
        print(f'  Mean {short}: {np.mean(xb_pp):.4f}  literal: {np.mean(xb_pl):.4f}  '
              f'diff: {np.mean(xb_diffs):+.4f}')

        stat_w_xb, p_w_xb = stats.wilcoxon(xb_pp, xb_pl, alternative='two-sided')
        d_xb = np.mean(xb_diffs) / np.std(xb_diffs) if np.std(xb_diffs) > 0 else 0
        print(f'  Wilcoxon: W={stat_w_xb:.0f}, p={fmt_p(p_w_xb)}, paired d={d_xb:+.3f}')

        rng_xb = np.random.RandomState(42)
        boot_xb = np.zeros(10000)
        for bi in range(10000):
            sample = rng_xb.choice(xb_diffs, size=len(xb_diffs), replace=True)
            boot_xb[bi] = np.mean(sample)
        ci_lo_xb, ci_hi_xb = np.percentile(boot_xb, [2.5, 97.5])
        boot_p_xb = 2 * min(np.mean(boot_xb <= 0), np.mean(boot_xb >= 0))
        print(f'  Bootstrap 95% CI: [{ci_lo_xb:+.4f}, {ci_hi_xb:+.4f}]')
        print(f'  Bootstrap p: {fmt_p(boot_p_xb)}')
        print(f'  CI excludes 0: {"YES" if ci_lo_xb > 0 or ci_hi_xb < 0 else "NO"}')

        n_exact_eq = sum(1 for a, b in zip(xb_pos_counts[phenom], xb_pos_counts['literal']) if a == b)
        print(f'  Equalized positions: {n_exact_eq}/{len(xb_pos_counts[phenom])} pairs identical  '
              f'mean {short}={np.mean(xb_pos_counts[phenom]):.1f}  '
              f'literal={np.mean(xb_pos_counts["literal"]):.1f}')

        stat_results['cross_boundary_holonomy'] = {
            'n_pairs': len(xb_shared),
            'pct_higher': float(pct_xb),
            'wilcoxon_W': float(stat_w_xb),
            'wilcoxon_p': float(p_w_xb),
            'paired_d': float(d_xb),
            'bootstrap_ci': [float(ci_lo_xb), float(ci_hi_xb)],
            'bootstrap_p': float(boot_p_xb),
            'context_window': CONTEXT_WINDOW,
        }

    # Cross-boundary curvature
    xb_shared_c = sorted(set(xb_curv_phenom_by_pair) & set(xb_curv_lit_by_pair))
    if len(xb_shared_c) >= 5:
        xb_cp = np.array([xb_curv_phenom_by_pair[p] for p in xb_shared_c])
        xb_cl = np.array([xb_curv_lit_by_pair[p] for p in xb_shared_c])
        fm = np.isfinite(xb_cp) & np.isfinite(xb_cl)
        if np.sum(fm) >= 5:
            xb_cp_f, xb_cl_f = xb_cp[fm], xb_cl[fm]
            n_hc_xb = int(np.sum(xb_cp_f > xb_cl_f))
            pct_hc_xb = 100 * n_hc_xb / len(xb_cp_f)
            sw_xb, pw_xb = stats.wilcoxon(xb_cp_f, xb_cl_f, alternative='two-sided')
            d_c_xb = np.mean(xb_cp_f - xb_cl_f) / np.std(xb_cp_f - xb_cl_f) if np.std(xb_cp_f - xb_cl_f) > 0 else 0
            print(f'\n  Cross-boundary curvature: {phenom.title()} > Lit in {n_hc_xb}/{len(xb_cp_f)} ({pct_hc_xb:.0f}%)')
            print(f'  Wilcoxon: W={sw_xb:.0f}, p={fmt_p(pw_xb)}, paired d={d_c_xb:+.3f}')
            stat_results['cross_boundary_curvature'] = {
                'n_pairs': int(len(xb_cp_f)),
                'pct_higher': float(pct_hc_xb),
                'wilcoxon_W': float(sw_xb),
                'wilcoxon_p': float(pw_xb),
                'paired_d': float(d_c_xb),
            }

    # Cross-boundary per-layer
    if xb_layer_profiles[phenom] and xb_layer_profiles['literal']:
        xb_layer_pvals = []
        xb_layer_rows = []
        hbar('Cross-Boundary Per-Layer Analysis')
        for l in range(n_layers):
            kl_p = [prof[l] for prof in xb_layer_profiles[phenom] if len(prof) > l]
            kl_l = [prof[l] for prof in xb_layer_profiles['literal'] if len(prof) > l]
            if len(kl_p) >= 2 and len(kl_l) >= 2:
                U_l, p_l = stats.mannwhitneyu(kl_p, kl_l, alternative='two-sided')
                d_mean_l = np.mean(kl_p) - np.mean(kl_l)
                xb_layer_pvals.append(p_l)
                xb_layer_rows.append((l, np.mean(kl_p), np.mean(kl_l), d_mean_l, p_l))
            else:
                xb_layer_pvals.append(1.0)
                xb_layer_rows.append((l, 0, 0, 0, 1.0))

        xb_layer_qvals = benjamini_hochberg(xb_layer_pvals)
        for (l, mp, ml, d_mean_l, p_l), q_l in zip(xb_layer_rows, xb_layer_qvals):
            sig = '*' if q_l < 0.05 else ' '
            print(f'  Layer {l:2d}: {short}={mp:.4f}  '
                  f'lit={ml:.4f}  diff={d_mean_l:+.4f}  p={p_l:.4f}  q={q_l:.4f} {sig}')

    # ── 13. Plots ────────────────────────────────────────────────────────
    hbar('Generating Plots')
    colors = cfg.colors

    plot_distributions(kappas, 'Holonomy Distribution by Condition',
                       r'Mean holonomy $\kappa$',
                       str(cfg.output_dir / 'holonomy_distributions.png'), colors=colors)

    plot_distributions(curvature_results, 'Curvature Distribution by Condition',
                       'Curvature (commutator norm)',
                       str(cfg.output_dir / 'curvature_distributions.png'), colors=colors)

    plot_layer_profiles(layer_profiles, 'Layer-wise Holonomy (VFE Steps)',
                        str(cfg.output_dir / 'layer_holonomy_profiles.png'),
                        n_layers=n_layers, colors=colors)

    plot_curvature_layer_profiles(curvature_layer_profiles,
                                  str(cfg.output_dir / 'layer_curvature_profiles.png'),
                                  n_layers=n_layers, colors=colors)

    if shared:
        plot_paired_comparison(phenom_by_pair, lit_by_pair, shared, phrases_by_pair,
                               r'Mean holonomy $\kappa$',
                               'Paired Comparison: Same Phrase, Different Usage',
                               str(cfg.output_dir / 'paired_holonomy.png'))

    if shared_curv:
        plot_paired_comparison(curv_phenom_by_pair, curv_lit_by_pair, shared_curv, phrases_by_pair,
                               'Curvature (commutator norm)',
                               'Paired Curvature: Same Phrase, Different Usage',
                               str(cfg.output_dir / 'paired_curvature.png'))

    if loc_shared:
        plot_paired_comparison(loc_phenom_by_pair, loc_lit_by_pair, loc_shared, phrases_by_pair,
                               r'Phrase-localized $\kappa$',
                               'Phrase-Localized Holonomy (Length Controlled)',
                               str(cfg.output_dir / 'phrase_localized_holonomy.png'))

    if loc_shared_c:
        plot_paired_comparison(loc_curv_phenom_by_pair, loc_curv_lit_by_pair, loc_shared_c, phrases_by_pair,
                               'Phrase-localized curvature',
                               'Phrase-Localized Curvature (Length Controlled)',
                               str(cfg.output_dir / 'phrase_localized_curvature.png'))

    if loc_layer_profiles[phenom] and loc_layer_profiles['literal']:
        plot_layer_profiles(loc_layer_profiles,
                            'Phrase-Localized Layer Holonomy (Length Controlled)',
                            str(cfg.output_dir / 'phrase_localized_layer_profiles.png'),
                            n_layers=n_layers, colors=colors)

    if loc_curv_layer_profiles[phenom] and loc_curv_layer_profiles['literal']:
        plot_curvature_layer_profiles(loc_curv_layer_profiles,
                                      str(cfg.output_dir / 'phrase_localized_layer_curvature.png'),
                                      n_layers=n_layers, colors=colors)

    if xb_shared:
        plot_paired_comparison(xb_phenom_by_pair, xb_lit_by_pair, xb_shared, phrases_by_pair,
                               r'Cross-boundary $\kappa$',
                               'Cross-Boundary Holonomy (Phrase+Context)',
                               str(cfg.output_dir / 'cross_boundary_holonomy.png'))

    if xb_shared_c:
        plot_paired_comparison(xb_curv_phenom_by_pair, xb_curv_lit_by_pair, xb_shared_c, phrases_by_pair,
                               'Cross-boundary curvature',
                               'Cross-Boundary Curvature (Phrase+Context)',
                               str(cfg.output_dir / 'cross_boundary_curvature.png'))

    if xb_layer_profiles[phenom] and xb_layer_profiles['literal']:
        plot_layer_profiles(xb_layer_profiles,
                            'Cross-Boundary Layer Holonomy (Phrase+Context)',
                            str(cfg.output_dir / 'cross_boundary_layer_profiles.png'),
                            n_layers=n_layers, colors=colors)

    # ── 13b. Double dissociation plot ────────────────────────────────────
    if xb_shared and xb_shared_c:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        ax = axes[0]
        scales = []
        hol_ds = []
        curv_ds = []

        if 'paired_wilcoxon' in stat_results:
            scales.append('Whole\nsentence')
            ws_pp = np.array([phenom_by_pair[p] for p in shared])
            ws_pl = np.array([lit_by_pair[p] for p in shared])
            ws_diff = ws_pp - ws_pl
            hol_ds.append(np.mean(ws_diff) / np.std(ws_diff) if np.std(ws_diff) > 0 else 0)
            if shared_curv:
                c_pp = np.array([curv_phenom_by_pair.get(p, float('nan')) for p in shared_curv])
                c_pl = np.array([curv_lit_by_pair.get(p, float('nan')) for p in shared_curv])
                c_diff = c_pp - c_pl
                fm = np.isfinite(c_diff)
                curv_ds.append(np.mean(c_diff[fm]) / np.std(c_diff[fm]) if np.std(c_diff[fm]) > 0 else 0)
            else:
                curv_ds.append(0)

        if loc_shared:
            scales.append('Phrase\nonly')
            hol_ds.append(float(stat_results.get('phrase_localized_holonomy', {}).get('paired_d', 0)))
            curv_ds.append(float(stat_results.get('phrase_localized_curvature', {}).get('paired_d', 0)))

        scales.append('Cross-\nboundary')
        hol_ds.append(float(stat_results.get('cross_boundary_holonomy', {}).get('paired_d', 0)))
        curv_ds.append(float(stat_results.get('cross_boundary_curvature', {}).get('paired_d', 0)))

        x = np.arange(len(scales))
        w = 0.35
        ax.bar(x - w/2, hol_ds, w, label='Holonomy (path defect)', color='#1f77b4', alpha=0.8)
        ax.bar(x + w/2, curv_ds, w, label='Curvature (superposition)', color='#ff7f0e', alpha=0.8)
        ax.axhline(y=0, color='black', linewidth=0.5, linestyle='-')
        ax.set_xticks(x)
        ax.set_xticklabels(scales)
        ax.set_ylabel(f"Paired Cohen's d ({short} - literal)")
        ax.set_title('Effect Size by Scale')
        ax.legend(loc='upper left', fontsize=9)

        ax = axes[1]
        xb_hol_diffs = np.array([xb_phenom_by_pair[p] - xb_lit_by_pair[p] for p in xb_shared])
        xb_curv_diffs_paired = []
        xb_shared_both = sorted(set(xb_shared) & set(xb_shared_c))
        for p in xb_shared_both:
            xb_curv_diffs_paired.append(xb_curv_phenom_by_pair[p] - xb_curv_lit_by_pair[p])
        xb_curv_diffs_arr = np.array(xb_curv_diffs_paired)

        ax.hist(xb_hol_diffs, bins=20, alpha=0.5, color='#1f77b4', label='Holonomy diff', density=True)
        ax.hist(xb_curv_diffs_arr, bins=20, alpha=0.5, color='#ff7f0e', label='Curvature diff', density=True)
        ax.axvline(x=0, color='black', linewidth=0.8, linestyle='--')
        ax.axvline(x=np.mean(xb_hol_diffs), color='#1f77b4', linewidth=2, linestyle='-',
                   label=f'Hol mean={np.mean(xb_hol_diffs):+.4f}')
        if len(xb_curv_diffs_arr) > 0:
            ax.axvline(x=np.mean(xb_curv_diffs_arr), color='#ff7f0e', linewidth=2, linestyle='-',
                       label=f'Curv mean={np.mean(xb_curv_diffs_arr):+.4f}')
        ax.set_xlabel(f'{phenom.title()} - Literal')
        ax.set_ylabel('Density')
        ax.set_title('Cross-Boundary: Holonomy vs Curvature')
        ax.legend(fontsize=8)

        ax = axes[2]
        if xb_layer_profiles[phenom] and xb_layer_profiles['literal']:
            layers = list(range(n_layers))
            stack_p = np.array(xb_layer_profiles[phenom])
            stack_l = np.array(xb_layer_profiles['literal'])
            diff_per_layer = np.mean(stack_p, axis=0)[:n_layers] - np.mean(stack_l, axis=0)[:n_layers]
            sem_diff = np.sqrt(
                (np.var(stack_p, axis=0)[:n_layers] / len(stack_p)) +
                (np.var(stack_l, axis=0)[:n_layers] / len(stack_l))
            )
            colors_layer = ['#ff7f0e' if d > 0 else '#1f77b4' for d in diff_per_layer]
            ax.bar(layers, diff_per_layer, color=colors_layer, alpha=0.7)
            ax.errorbar(layers, diff_per_layer, yerr=sem_diff, fmt='none', ecolor='black',
                       capsize=3, linewidth=1)
            ax.axhline(y=0, color='black', linewidth=0.5, linestyle='-')
            ax.set_xlabel('Layer')
            ax.set_ylabel(fr'$\Delta\kappa$ ({short} - literal)')
            ax.set_title('Cross-Boundary Holonomy Difference by Layer')
            ax.set_xticks(layers)

            if xb_layer_qvals is not None and len(xb_layer_qvals) == n_layers:
                for l in layers:
                    if xb_layer_qvals[l] < 0.05:
                        ax.text(l, diff_per_layer[l] - 0.002 * np.sign(diff_per_layer[l]),
                               '*', ha='center', va='center', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(str(cfg.output_dir / 'double_dissociation.png'), dpi=150, bbox_inches='tight')
        print(f'  Saved: {cfg.output_dir / "double_dissociation.png"}')
        plt.close()

    # ── 13c. Synthesis ───────────────────────────────────────────────────
    hbar('SYNTHESIS')
    for line in cfg.synthesis_lines:
        print(line)

    if 'cross_boundary_holonomy' in stat_results:
        xb_h = stat_results['cross_boundary_holonomy']
        direction_h = 'HIGHER' if xb_h['paired_d'] > 0 else 'LOWER'
        print(f'  Holonomy (path defect)| {direction_h:6s} d={xb_h["paired_d"]:+.3f} p={xb_h["wilcoxon_p"]:.4f}')
    if 'cross_boundary_curvature' in stat_results:
        xb_c = stat_results['cross_boundary_curvature']
        direction_c = 'HIGHER' if xb_c['paired_d'] > 0 else 'LOWER'
        print(f'  Curvature (F=dA+A^A) | {direction_c:6s} d={xb_c["paired_d"]:+.3f} p={xb_c["wilcoxon_p"]:.4f}')

    # ── 14. Save results ─────────────────────────────────────────────────
    hbar('Saving Results')
    summary = {
        'model': cfg.model_name,
        'device': cfg.device,
        'n_layers': n_layers,
        'd_model': d_model,
        'dataset': cfg.dataset_name,
        f'n_{phenom}': len(results[phenom]),
        'n_literal': len(results['literal']),
        'n_control': len(results['control']),
        'n_paired': len(shared),
        'stats': stat_results,
        'holonomy': {},
        'curvature': {},
    }

    for label in labels_ordered:
        if label in kappas:
            summary['holonomy'][label] = {
                'mean': float(np.mean(kappas[label])),
                'median': float(np.median(kappas[label])),
                'std': float(np.std(kappas[label])),
                'values': kappas[label].tolist(),
            }
        if label in curvature_results and len(curvature_results[label]) > 0:
            summary['curvature'][label] = {
                'mean': float(np.mean(curvature_results[label])),
                'median': float(np.median(curvature_results[label])),
                'std': float(np.std(curvature_results[label])),
                'values': curvature_results[label].tolist(),
            }

    summary['layer_holonomy_profiles'] = {
        label: [p for p in profiles]
        for label, profiles in layer_profiles.items() if profiles
    }
    summary['layer_curvature_profiles'] = {
        label: [p for p in profiles]
        for label, profiles in curvature_layer_profiles.items() if profiles
    }

    if shared:
        summary['paired_holonomy'] = [
            {'pair_id': p, 'phrase': phrases_by_pair.get(p, ''),
             phenom: phenom_by_pair[p], 'literal': lit_by_pair[p]}
            for p in shared
        ]

    for label, a in asymmetry_by_label.items():
        summary[f'{label}_asymmetry'] = a.tolist()

    if loc_shared:
        summary['phrase_localized_holonomy'] = [
            {'pair_id': p, 'phrase': phrases_by_pair.get(p, ''),
             phenom: float(loc_phenom_by_pair[p]),
             'literal': float(loc_lit_by_pair[p])}
            for p in loc_shared
        ]
    if loc_shared_c:
        summary['phrase_localized_curvature'] = [
            {'pair_id': p, 'phrase': phrases_by_pair.get(p, ''),
             phenom: float(loc_curv_phenom_by_pair[p]),
             'literal': float(loc_curv_lit_by_pair[p])}
            for p in loc_shared_c
        ]
    if xb_shared:
        summary['cross_boundary_holonomy_pairs'] = [
            {'pair_id': p, 'phrase': phrases_by_pair.get(p, ''),
             phenom: float(xb_phenom_by_pair[p]),
             'literal': float(xb_lit_by_pair[p])}
            for p in xb_shared
        ]
    if xb_shared_c:
        summary['cross_boundary_curvature_pairs'] = [
            {'pair_id': p, 'phrase': phrases_by_pair.get(p, ''),
             phenom: float(xb_curv_phenom_by_pair[p]),
             'literal': float(xb_curv_lit_by_pair[p])}
            for p in xb_shared_c
        ]

    with open(cfg.output_dir / cfg.result_filename, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f'  Saved: {cfg.output_dir / cfg.result_filename}')

    # ── Done ─────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    hbar(f'Done ({elapsed:.0f}s)')
    print(f'  Results: {cfg.output_dir}/')
