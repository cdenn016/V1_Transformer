#!/usr/bin/env python3
"""
Idiom Holonomy Study: Non-Compositionality as Gauge Curvature
================================================================

Tests whether non-compositional language (idioms) exhibits higher gauge
curvature than compositional language (literal usage of the same phrases).

Key insight: idioms are non-compositional by definition. "Kick the bucket"
cannot be understood by composing kick + the + bucket. In gauge-theoretic
terms, this IS curvature: parallel transport of meaning through component
tokens is path-dependent.

Three measurements:
    1. Layer-wise Jacobian holonomy (each layer = one VFE step)
    2. Discrete curvature (composition non-additivity per layer)
    3. Full statistical analysis + layer-resolved profiles

Usage:
    python run_idiom_study.py
"""

import sys
import os
import time
import subprocess

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

check_deps()

import torch
import numpy as np
from pathlib import Path

_here = Path(__file__).resolve().parent
ROOT = _here
for _ancestor in [_here] + list(_here.parents):
    if (_ancestor / 'analysis' / 'holonomy_study' / '__init__.py').exists():
        ROOT = _ancestor
        break
sys.path.insert(0, str(ROOT))

from analysis.holonomy_study.transport import (
    load_model,
    attention_flow_asymmetry,
    layerwise_jacobian_holonomy,
    jacobian_holonomy,
    discrete_curvature,
)
from analysis.holonomy_study.holonomy import HolonomyResult
from analysis.holonomy_study.idiom_datasets import load_idiom_pairs, by_label, get_paired_only

from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json


# ── Config ────────────────────────────────────────────────────────────────

MODEL_NAME  = 'gpt2'
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_TRI     = 300          # triangles per sentence
MAX_PAIRS   = 150          # pairs for curvature
OUTPUT_DIR  = ROOT / 'results' / 'idiom_study'

COLORS = {
    'idiomatic': '#d62728',   # red
    'literal':   '#2ca02c',   # green
    'control':   '#1f77b4',   # blue
}


# ── Helpers ───────────────────────────────────────────────────────────────

def hbar(text='', width=60):
    if text:
        pad = width - len(text) - 2
        print(f"\n{'='*(pad//2)} {text} {'='*(pad - pad//2)}")
    else:
        print('=' * width)

def tokenize(tokenizer, text):
    return torch.tensor([tokenizer.encode(text)], device=DEVICE)

def fmt_p(p):
    if p < 0.001:  return f'{p:.2e} ***'
    if p < 0.01:   return f'{p:.4f} **'
    if p < 0.05:   return f'{p:.4f} *'
    return f'{p:.4f} ns'


# ── Plotting ──────────────────────────────────────────────────────────────

def plot_distributions(kappas_by_label, title, ylabel, output_path):
    """Violin + strip + histogram for any metric."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    labels_order = ['literal', 'idiomatic', 'control']

    # Left: violin + strip
    ax = axes[0]
    positions, data, colors, tick_labels = [], [], [], []
    for idx, label in enumerate(labels_order):
        if label in kappas_by_label and len(kappas_by_label[label]) > 0:
            positions.append(idx)
            data.append(kappas_by_label[label])
            colors.append(COLORS[label])
            tick_labels.append(f"{label}\n(n={len(kappas_by_label[label])})")

    if data:
        parts = ax.violinplot(data, positions=positions, showmeans=True, showmedians=True)
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.3)
        parts['cmeans'].set_color('black')
        parts['cmedians'].set_color('gray')
        for i, (pos, d) in enumerate(zip(positions, data)):
            jitter = np.random.RandomState(42).uniform(-0.1, 0.1, size=len(d))
            ax.scatter(pos + jitter, d, c=colors[i], alpha=0.5, s=15, zorder=3)

    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Right: histogram
    ax = axes[1]
    all_vals = np.concatenate([v for v in kappas_by_label.values() if len(v) > 0])
    bins = np.linspace(np.min(all_vals), np.max(all_vals), 30)
    for label in labels_order:
        if label in kappas_by_label and len(kappas_by_label[label]) > 0:
            ax.hist(kappas_by_label[label], bins=bins, alpha=0.4,
                    color=COLORS[label], label=label, density=True)
    ax.set_xlabel(ylabel)
    ax.set_ylabel('Density')
    ax.set_title(f'{title} (Histogram)')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_layer_profiles(profiles_by_label, title, output_path, n_layers=12):
    """Plot per-layer kappa or curvature as a function of depth."""
    fig, ax = plt.subplots(figsize=(10, 6))
    layers = list(range(n_layers))

    for label in ['literal', 'idiomatic', 'control']:
        if label not in profiles_by_label:
            continue
        all_profiles = profiles_by_label[label]  # list of (n_layers,) arrays
        if not all_profiles:
            continue

        # Individual traces
        for prof in all_profiles:
            ax.plot(layers, prof[:n_layers], color=COLORS[label], alpha=0.08, linewidth=0.8)

        # Mean + SEM
        stacked = np.array(all_profiles)
        mean = np.nanmean(stacked, axis=0)[:n_layers]
        sem = np.nanstd(stacked, axis=0)[:n_layers] / np.sqrt(len(all_profiles))
        ax.plot(layers, mean, color=COLORS[label], linewidth=2.5, label=f'{label} (n={len(all_profiles)})')
        ax.fill_between(layers, mean - sem, mean + sem, color=COLORS[label], alpha=0.15)

    ax.set_xlabel('Layer (VFE step)')
    ax.set_ylabel(r'Mean $\kappa$')
    ax.set_title(title)
    ax.legend()
    ax.set_xticks(layers)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_paired_comparison(ironic_by_pair, literal_by_pair, shared, idioms_by_pair, ylabel, title, output_path):
    """Paired lines connecting idiomatic vs literal for same phrase."""
    fig, ax = plt.subplots(figsize=(8, 6))

    ival = [ironic_by_pair[p] for p in shared]
    lval = [literal_by_pair[p] for p in shared]

    for iv, lv in zip(ival, lval):
        color = COLORS['idiomatic'] if iv > lv else COLORS['literal']
        ax.plot([0, 1], [lv, iv], color=color, alpha=0.4, linewidth=1)

    ax.scatter([0]*len(lval), lval, c=COLORS['literal'], s=40, zorder=3, label='literal')
    ax.scatter([1]*len(ival), ival, c=COLORS['idiomatic'], s=40, zorder=3, label='idiomatic')

    n_higher = sum(1 for i, l in zip(ival, lval) if i > l)
    pct = 100 * n_higher / len(shared) if shared else 0

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Literal usage', 'Idiomatic usage'])
    ax.set_ylabel(ylabel)
    ax.set_title(f'{title}\nIdiomatic > Literal in {n_higher}/{len(shared)} pairs ({pct:.0f}%)')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_curvature_layer_profiles(profiles_by_label, output_path, n_layers=12):
    """Plot per-layer curvature (commutator) as a function of depth."""
    fig, ax = plt.subplots(figsize=(10, 6))
    layers = list(range(n_layers))

    for label in ['literal', 'idiomatic', 'control']:
        if label not in profiles_by_label:
            continue
        all_profiles = profiles_by_label[label]
        if not all_profiles:
            continue

        for prof in all_profiles:
            ax.plot(layers, prof[:n_layers], color=COLORS[label], alpha=0.08, linewidth=0.8)

        stacked = np.array(all_profiles)
        mean = np.nanmean(stacked, axis=0)[:n_layers]
        sem = np.nanstd(stacked, axis=0)[:n_layers] / np.sqrt(len(all_profiles))
        ax.plot(layers, mean, color=COLORS[label], linewidth=2.5, label=f'{label} (n={len(all_profiles)})')
        ax.fill_between(layers, mean - sem, mean + sem, color=COLORS[label], alpha=0.15)

    ax.set_xlabel('Layer (VFE step)')
    ax.set_ylabel('Curvature (commutator norm)')
    ax.set_title('Discrete Riemann Curvature by Layer')
    ax.legend()
    ax.set_xticks(layers)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


# ── Statistical tests ─────────────────────────────────────────────────────

def run_stats(kappas, label_a, label_b, metric_name='kappa'):
    """Run Mann-Whitney U + effect sizes."""
    if label_a not in kappas or label_b not in kappas:
        return None
    a, b = kappas[label_a], kappas[label_b]
    if len(a) < 2 or len(b) < 2:
        return None
    U, p = stats.mannwhitneyu(a, b, alternative='two-sided')
    pooled = np.sqrt((np.var(a) + np.var(b)) / 2)
    d = (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else 0.0
    r = 1 - (2*U) / (len(a)*len(b))
    print(f'  {label_a} vs {label_b} ({metric_name}): U={U:.0f}  p={fmt_p(p)}  d={d:+.3f}  r={r:+.3f}')
    return {'U': float(U), 'p_value': float(p), 'cohens_d': float(d), 'rank_biserial_r': float(r)}


def run_permutation_test(a, b, label_a, label_b, metric_name='kappa', n_perm=10000, seed=42):
    """Permutation test for difference in means."""
    rng = np.random.RandomState(seed)
    obs_diff = np.mean(a) - np.mean(b)
    pooled = np.concatenate([a, b])
    na = len(a)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        if abs(np.mean(pooled[:na]) - np.mean(pooled[na:])) >= abs(obs_diff):
            count += 1
    p_perm = (count + 1) / (n_perm + 1)
    print(f'  {label_a} vs {label_b} ({metric_name}): obs_diff={obs_diff:+.4f}  p_perm={fmt_p(p_perm)}')
    return float(p_perm)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load model ────────────────────────────────────────────────────
    hbar('Loading GPT-2')
    model, tokenizer = load_model(MODEL_NAME, device=DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    n_layers = len(model.h)
    d_model = model.config.n_embd
    print(f'  Model:      {MODEL_NAME}')
    print(f'  Parameters: {n_params:,}')
    print(f'  Layers:     {n_layers}')
    print(f'  Hidden dim: {d_model}')
    print(f'  Device:     {DEVICE}')

    # ── 2. Load idiom dataset ─────────────────────────────────────────────
    hbar('Idiom Dataset')
    all_pairs = load_idiom_pairs()
    groups = by_label(all_pairs)
    for label, items in groups.items():
        print(f'  {label:11s}: {len(items)} sentences')
    paired = get_paired_only(all_pairs)
    print(f'  paired:      {len(paired)} (strongest test: same phrase, different usage)')

    # ── 3. Method 0: Attention path defect (fast sanity check) ────────────
    hbar('Method 0: Attention Path Defect')
    asymmetry_by_label = {}
    for label, items in groups.items():
        vals = []
        for sp in items:
            ids = tokenize(tokenizer, sp.text)
            r = attention_flow_asymmetry(model, ids)
            vals.append(r['defect_per_layer'].mean().item())
        asymmetry_by_label[label] = np.array(vals)
        print(f'  {label:11s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}')

    for la, lb in [('idiomatic','literal'), ('idiomatic','control'), ('literal','control')]:
        U, p = stats.mannwhitneyu(asymmetry_by_label[la], asymmetry_by_label[lb], alternative='two-sided')
        print(f'  {la} vs {lb}: p = {fmt_p(p)}')

    # ── 4. Layer-wise Jacobian holonomy (each layer = VFE step) ──────────
    hbar('Layer-wise Jacobian Holonomy (VFE Steps)')

    results = {'idiomatic': [], 'literal': [], 'control': []}
    layer_profiles = {'idiomatic': [], 'literal': [], 'control': []}
    total = sum(len(v) for v in groups.values())
    done = 0

    for label, items in groups.items():
        for sp in items:
            t0 = time.time()
            ids = tokenize(tokenizer, sp.text)
            N = ids.shape[1]
            if N < 3:
                done += 1
                continue

            plh = layerwise_jacobian_holonomy(
                model, ids, max_triples=MAX_TRI,
            )

            # Store per-layer profile
            layer_profiles[label].append(plh['kappa_per_layer'])

            # Wrap for compatibility
            kappa_arr = plh['kappa_all']
            if kappa_arr.ndim == 2:
                kappa_arr = np.nanmean(kappa_arr, axis=0)

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
                    'idiom': getattr(sp, 'idiom', ''),
                    'method': 'layerwise_jacobian',
                    'n_tokens': N,
                    'd_model': plh['d_model'],
                    'n_layers': plh['n_layers'],
                    'kappa_per_layer': plh['kappa_per_layer'],
                    'cos_sim_per_layer': plh['cos_sim_per_layer'],
                    'n_forward_passes': plh['n_forward_passes'],
                },
            )
            results[label].append(hr)

            done += 1
            dt = time.time() - t0
            print(f'  [{done:3d}/{total}] {label:11s} kappa={plh["kappa_mean"]:.4f}  '
                  f'({N} tok, {plh["n_forward_passes"]} fwd, {dt:.1f}s)  '
                  f'{sp.text[:55]}...')

    # ── 5. Discrete curvature (commutator) ────────────────────────────────
    hbar('Discrete Riemann Curvature (Transport Commutator)')

    curvature_results = {'idiomatic': [], 'literal': [], 'control': []}
    curvature_layer_profiles = {'idiomatic': [], 'literal': [], 'control': []}
    done = 0

    for label, items in groups.items():
        for sp in items:
            t0 = time.time()
            ids = tokenize(tokenizer, sp.text)
            N = ids.shape[1]
            if N < 3:
                done += 1
                continue

            cr = discrete_curvature(
                model, ids, max_triples=MAX_PAIRS,
            )

            curvature_results[label].append(cr['curvature_mean'])
            curvature_layer_profiles[label].append(cr['curvature_per_layer'])

            done += 1
            dt = time.time() - t0
            print(f'  [{done:3d}/{total}] {label:11s} curv={cr["curvature_mean"]:.6f}  '
                  f'({N} tok, {dt:.1f}s)  {sp.text[:55]}...')

    # Convert to arrays
    for label in curvature_results:
        curvature_results[label] = np.array(curvature_results[label])

    # ── 6. Statistical analysis ──────────────────────────────────────────
    hbar('Statistical Analysis: Layer-wise Holonomy')

    kappas = {
        label: np.array([hr.kappa_mean for hr in hrs])
        for label, hrs in results.items() if hrs
    }

    for label in ['idiomatic', 'literal', 'control']:
        if label in kappas:
            k = kappas[label]
            print(f'  {label:11s}: mean={np.mean(k):.4f}  median={np.median(k):.4f}  std={np.std(k):.4f}')

    stat_results = {}
    for la, lb in [('idiomatic','literal'), ('idiomatic','control'), ('literal','control')]:
        r = run_stats(kappas, la, lb, 'holonomy')
        if r:
            stat_results[f'{la}_vs_{lb}_holonomy'] = r

    hbar('Statistical Analysis: Curvature')
    for label in ['idiomatic', 'literal', 'control']:
        if label in curvature_results and len(curvature_results[label]) > 0:
            c = curvature_results[label]
            print(f'  {label:11s}: mean={np.mean(c):.6f}  median={np.median(c):.6f}  std={np.std(c):.6f}')

    for la, lb in [('idiomatic','literal'), ('idiomatic','control'), ('literal','control')]:
        r = run_stats(curvature_results, la, lb, 'curvature')
        if r:
            stat_results[f'{la}_vs_{lb}_curvature'] = r

    # ── 7. Paired comparison ──────────────────────────────────────────────
    hbar('Paired Comparison (Same Phrase, Different Usage)')

    idiom_by_pair = {hr.metadata['pair_id']: hr.kappa_mean for hr in results['idiomatic']}
    literal_by_pair = {hr.metadata['pair_id']: hr.kappa_mean for hr in results['literal']}
    idioms_by_pair = {}
    for hr in results['idiomatic']:
        idioms_by_pair[hr.metadata['pair_id']] = hr.metadata.get('idiom', '')
    shared = sorted(set(idiom_by_pair) & set(literal_by_pair))

    n_higher = 0
    for pid in shared:
        ik, lk = idiom_by_pair[pid], literal_by_pair[pid]
        arrow = '>' if ik > lk else '<'
        tag = '*' if ik > lk else ' '
        idiom = idioms_by_pair.get(pid, '')
        print(f'  {tag} pair {pid:2d}: idiomatic={ik:.4f} {arrow} literal={lk:.4f}  "{idiom}"')
        if ik > lk:
            n_higher += 1

    pct = 100 * n_higher / len(shared) if shared else 0
    print(f'\n  Idiomatic > Literal in {n_higher}/{len(shared)} pairs ({pct:.0f}%)')

    if len(shared) >= 5:
        paired_idiom = [idiom_by_pair[p] for p in shared]
        paired_literal = [literal_by_pair[p] for p in shared]
        stat_w, p_w = stats.wilcoxon(paired_idiom, paired_literal, alternative='two-sided')
        print(f'  Wilcoxon signed-rank: W={stat_w:.0f}, p={fmt_p(p_w)}')
        stat_results['paired_wilcoxon'] = {'W': float(stat_w), 'p_value': float(p_w)}

    # Also paired comparison for curvature
    curv_idiom_by_pair = {}
    curv_literal_by_pair = {}
    ci = 0
    for label, items in groups.items():
        for sp in items:
            if label == 'idiomatic':
                curv_idiom_by_pair[sp.pair_id] = curvature_results['idiomatic'][ci] if ci < len(curvature_results['idiomatic']) else float('nan')
            elif label == 'literal':
                curv_literal_by_pair[sp.pair_id] = curvature_results['literal'][ci - len(groups['idiomatic'])] if (ci - len(groups['idiomatic'])) < len(curvature_results['literal']) else float('nan')
            ci += 1

    # Simpler approach for curvature pairing
    curv_idiom_by_pair = {}
    curv_literal_by_pair = {}
    for idx, sp in enumerate(groups['idiomatic']):
        if idx < len(curvature_results['idiomatic']):
            curv_idiom_by_pair[sp.pair_id] = curvature_results['idiomatic'][idx]
    for idx, sp in enumerate(groups['literal']):
        if idx < len(curvature_results['literal']):
            curv_literal_by_pair[sp.pair_id] = curvature_results['literal'][idx]

    shared_curv = sorted(set(curv_idiom_by_pair) & set(curv_literal_by_pair))
    if len(shared_curv) >= 5:
        pc_idiom = [curv_idiom_by_pair[p] for p in shared_curv]
        pc_literal = [curv_literal_by_pair[p] for p in shared_curv]
        n_higher_c = sum(1 for i, l in zip(pc_idiom, pc_literal) if i > l)
        pct_c = 100 * n_higher_c / len(shared_curv)
        print(f'\n  Curvature: Idiomatic > Literal in {n_higher_c}/{len(shared_curv)} pairs ({pct_c:.0f}%)')
        stat_w, p_w = stats.wilcoxon(pc_idiom, pc_literal, alternative='two-sided')
        print(f'  Curvature Wilcoxon: W={stat_w:.0f}, p={fmt_p(p_w)}')

    # ── 8. Permutation tests ─────────────────────────────────────────────
    hbar('Permutation Tests (10,000 shuffles)')
    for la, lb in [('idiomatic','literal'), ('idiomatic','control'), ('literal','control')]:
        if la in kappas and lb in kappas:
            run_permutation_test(kappas[la], kappas[lb], la, lb, 'holonomy')
        if la in curvature_results and lb in curvature_results:
            if len(curvature_results[la]) > 0 and len(curvature_results[lb]) > 0:
                run_permutation_test(curvature_results[la], curvature_results[lb], la, lb, 'curvature')

    # Paired sign-flip
    if len(shared) >= 5:
        paired_diffs = np.array(paired_idiom) - np.array(paired_literal)
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

    for label in ['idiomatic', 'literal', 'control']:
        mask = all_lab == label
        if mask.any():
            print(f'  {label:11s}: mean_len={np.mean(all_len[mask]):.1f}  mean_kappa={np.mean(all_k[mask]):.4f}')

    r_len, p_len = stats.pearsonr(all_len, all_k)
    print(f'  kappa ~ length: r={r_len:+.3f}  p={fmt_p(p_len)}')

    slope, intercept = np.polyfit(all_len, all_k, 1)
    residuals = all_k - (slope * all_len + intercept)
    print(f'  Regression: kappa = {slope:+.5f} * n_tokens + {intercept:.4f}')

    resid_by_label = {}
    for label in ['idiomatic', 'literal', 'control']:
        mask = all_lab == label
        if mask.any():
            resid_by_label[label] = residuals[mask]
            print(f'  {label:11s} residual: mean={np.mean(residuals[mask]):+.4f}  std={np.std(residuals[mask]):.4f}')

    print('\n  Length-controlled Mann-Whitney (on residuals):')
    for la, lb in [('idiomatic','literal'), ('idiomatic','control'), ('literal','control')]:
        if la in resid_by_label and lb in resid_by_label:
            a, b = resid_by_label[la], resid_by_label[lb]
            U, p = stats.mannwhitneyu(a, b, alternative='two-sided')
            d_val = (np.mean(a) - np.mean(b)) / np.sqrt((np.var(a) + np.var(b)) / 2) if (np.var(a) + np.var(b)) > 0 else 0
            print(f'    {la} vs {lb}: U={U:.0f}  p={fmt_p(p)}  d={d_val:+.3f}')

    # ── 10. Bootstrap CIs ────────────────────────────────────────────────
    hbar('Bootstrap CIs (Paired Idiomatic - Literal)')
    N_BOOT = 10_000
    rng = np.random.RandomState(42)

    if len(shared) >= 5:
        paired_diffs = np.array(paired_idiom) - np.array(paired_literal)
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

    # ── 11. Per-layer statistical tests ──────────────────────────────────
    hbar('Per-Layer Analysis (Where Does Curvature Emerge?)')

    for l in range(n_layers):
        kl_idiom = [prof[l] for prof in layer_profiles['idiomatic'] if len(prof) > l]
        kl_literal = [prof[l] for prof in layer_profiles['literal'] if len(prof) > l]
        if len(kl_idiom) >= 2 and len(kl_literal) >= 2:
            U, p = stats.mannwhitneyu(kl_idiom, kl_literal, alternative='two-sided')
            d_mean = np.mean(kl_idiom) - np.mean(kl_literal)
            sig = '*' if p < 0.05 else ' '
            print(f'  Layer {l:2d}: idiomatic={np.mean(kl_idiom):.4f}  '
                  f'literal={np.mean(kl_literal):.4f}  diff={d_mean:+.4f}  p={p:.4f} {sig}')

    # ── 12. Plots ─────────────────────────────────────────────────────────
    hbar('Generating Plots')

    plot_distributions(kappas, 'Holonomy Distribution by Condition',
                       r'Mean holonomy $\kappa$',
                       str(OUTPUT_DIR / 'holonomy_distributions.png'))

    plot_distributions(curvature_results, 'Curvature Distribution by Condition',
                       'Curvature (commutator norm)',
                       str(OUTPUT_DIR / 'curvature_distributions.png'))

    plot_layer_profiles(layer_profiles,
                        'Layer-wise Holonomy (VFE Steps)',
                        str(OUTPUT_DIR / 'layer_holonomy_profiles.png'),
                        n_layers=n_layers)

    plot_curvature_layer_profiles(curvature_layer_profiles,
                                  str(OUTPUT_DIR / 'layer_curvature_profiles.png'),
                                  n_layers=n_layers)

    if shared:
        plot_paired_comparison(idiom_by_pair, literal_by_pair, shared, idioms_by_pair,
                               r'Mean holonomy $\kappa$',
                               'Paired Comparison: Same Phrase, Different Usage',
                               str(OUTPUT_DIR / 'paired_holonomy.png'))

    if shared_curv:
        plot_paired_comparison(curv_idiom_by_pair, curv_literal_by_pair, shared_curv, idioms_by_pair,
                               'Curvature (commutator norm)',
                               'Paired Curvature: Same Phrase, Different Usage',
                               str(OUTPUT_DIR / 'paired_curvature.png'))

    # ── 13. Save results ──────────────────────────────────────────────────
    hbar('Saving Results')
    summary = {
        'model': MODEL_NAME,
        'device': DEVICE,
        'n_layers': n_layers,
        'd_model': d_model,
        'dataset': 'idiom_pairs',
        'n_idiomatic': len(results['idiomatic']),
        'n_literal': len(results['literal']),
        'n_control': len(results['control']),
        'n_paired': len(shared),
        'stats': stat_results,
        'holonomy': {},
        'curvature': {},
    }

    for label in ['idiomatic', 'literal', 'control']:
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

    # Per-layer profiles
    summary['layer_holonomy_profiles'] = {
        label: [p for p in profiles]
        for label, profiles in layer_profiles.items() if profiles
    }
    summary['layer_curvature_profiles'] = {
        label: [p for p in profiles]
        for label, profiles in curvature_layer_profiles.items() if profiles
    }

    # Paired results
    if shared:
        summary['paired_holonomy'] = [
            {'pair_id': p, 'idiom': idioms_by_pair.get(p, ''),
             'idiomatic': idiom_by_pair[p], 'literal': literal_by_pair[p]}
            for p in shared
        ]

    # Asymmetry
    for label, a in asymmetry_by_label.items():
        summary[f'{label}_asymmetry'] = a.tolist()

    with open(OUTPUT_DIR / 'idiom_results.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f'  Saved: {OUTPUT_DIR / "idiom_results.json"}')

    # ── Done ──────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    hbar(f'Done ({elapsed:.0f}s)')
    print(f'  Results: {OUTPUT_DIR}/')
    print(f'  Plots:')
    print(f'    holonomy_distributions.png    — violin + histogram of kappa')
    print(f'    curvature_distributions.png   — violin + histogram of curvature')
    print(f'    layer_holonomy_profiles.png   — per-layer kappa (VFE steps)')
    print(f'    layer_curvature_profiles.png  — per-layer curvature')
    print(f'    paired_holonomy.png           — paired comparison')
    print(f'    paired_curvature.png          — paired curvature comparison')


if __name__ == '__main__':
    main()
