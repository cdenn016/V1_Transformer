#!/usr/bin/env python3
"""
Holonomy Study: One-Click Runner
=================================

Measures gauge curvature in pretrained GPT-2 to test the flat bundle
conjecture: ironic language should exhibit higher holonomy than literal.

Usage:
    python run_holonomy_study.py
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

# Find project root: walk up from this script until we find analysis/holonomy_study/__init__.py
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
    attention_decomposed_transport,
    per_layer_holonomy,
    jacobian_holonomy,
)
from analysis.holonomy_study.holonomy import loop_holonomy, sentence_holonomy, multilayer_holonomy, HolonomyResult
from analysis.holonomy_study.datasets import load_irony_pairs, by_label, get_paired_only
from analysis.holonomy_study.visualization import (
    plot_holonomy_distributions,
    plot_paired_comparison,
    plot_asymmetry_comparison,
)
from analysis.holonomy_study.experiment import ExperimentResult

from scipy import stats


# ── Config ────────────────────────────────────────────────────────────────

MODEL_NAME  = 'gpt2'
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
METHOD      = 2            # 0=asymmetry, 1=attn-decomposed, 2=jacobian
MAX_TRI     = 500          # triangles per sentence
OUTPUT_DIR  = ROOT / 'results' / 'holonomy_study'


# ── Helpers ───────────────────────────────────────────────────────────────

def hbar(text='', width=60):
    if text:
        pad = width - len(text) - 2
        print(f"\n{'─'*(pad//2)} {text} {'─'*(pad - pad//2)}")
    else:
        print('─' * width)

def tokenize(tokenizer, text):
    return torch.tensor([tokenizer.encode(text)], device=DEVICE)

def fmt_p(p):
    if p < 0.001:  return f'{p:.2e} ***'
    if p < 0.01:   return f'{p:.4f} **'
    if p < 0.05:   return f'{p:.4f} *'
    return f'{p:.4f} ns'


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load model ────────────────────────────────────────────────────
    hbar('Loading GPT-2')
    model, tokenizer = load_model(MODEL_NAME, device=DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Model:      {MODEL_NAME}')
    print(f'  Parameters: {n_params:,}')
    print(f'  Layers:     {len(model.h)}')
    print(f'  Hidden dim: {model.config.n_embd}')
    print(f'  Device:     {DEVICE}')

    # ── 2. Load dataset ──────────────────────────────────────────────────
    hbar('Dataset')
    all_pairs = load_irony_pairs()
    groups = by_label(all_pairs)
    for label, items in groups.items():
        print(f'  {label:8s}: {len(items)} sentences')

    # ── 3. Method 0: Attention asymmetry (fast sanity check) ─────────────
    hbar('Method 0: Attention Path Defect')
    asymmetry_by_label = {}
    for label, items in groups.items():
        vals = []
        for sp in items:
            ids = tokenize(tokenizer, sp.text)
            r = attention_flow_asymmetry(model, ids)
            vals.append(r['defect_per_layer'].mean().item())
        asymmetry_by_label[label] = np.array(vals)
        print(f'  {label:8s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}')

    for la, lb in [('ironic','literal'), ('ironic','control'), ('literal','control')]:
        U, p = stats.mannwhitneyu(asymmetry_by_label[la], asymmetry_by_label[lb], alternative='two-sided')
        print(f'  {la} vs {lb}: p = {fmt_p(p)}')

    plot_asymmetry_comparison(asymmetry_by_label, str(OUTPUT_DIR / 'asymmetry.png'))

    if METHOD == 0:
        hbar('Done')
        return

    # ── 4. Transport holonomy ────────────────────────────────────────────
    method_names = {1: 'Per-Layer Attention', 2: 'Jacobian (Exact)'}
    hbar(f'Method {METHOD}: {method_names.get(METHOD, "?")} Transport Holonomy')

    results_by_label = {'ironic': [], 'literal': [], 'control': []}
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

            if METHOD == 2:
                plh = jacobian_holonomy(model, ids, max_triples=MAX_TRI)
                cs = plh.get('cos_sim_mean', float('nan'))
            else:
                plh = per_layer_holonomy(model, ids, max_triples=MAX_TRI)
                cs = float('nan')

            # Wrap into HolonomyResult for downstream compatibility
            kappa_arr = plh.get('kappa_all', np.array([]))
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
                    'method': 'jacobian' if METHOD == 2 else 'per_layer',
                    'n_tokens': N,
                    'd_model': plh['d_model'],
                    'n_triangles': len(plh['triples']),
                    'n_degenerate': 0,
                    'cos_sim_mean': cs,
                },
            )
            results_by_label[label].append(hr)

            done += 1
            dt = time.time() - t0
            cs_str = f'  cos={cs:.4f}' if not np.isnan(cs) else ''
            print(f'  [{done:2d}/{total}] {label:8s} kappa={hr.kappa_mean:.4f}{cs_str}  '
                  f'({N} tok, {dt:.1f}s)  {sp.text[:50]}...')

    # ── 5. Statistical analysis ──────────────────────────────────────────
    hbar('Statistical Analysis')

    kappas = {
        label: np.array([hr.kappa_mean for hr in hrs])
        for label, hrs in results_by_label.items() if hrs
    }

    # Also gather cosine similarities if available (Method 2)
    cos_sims_by_label = {}
    for label, hrs in results_by_label.items():
        cs_vals = [hr.metadata.get('cos_sim_mean', float('nan')) for hr in hrs]
        if any(not np.isnan(v) for v in cs_vals):
            cos_sims_by_label[label] = np.array(cs_vals)

    for label in ['ironic', 'literal', 'control']:
        if label in kappas:
            k = kappas[label]
            line = f'  {label:8s}: mean={np.mean(k):.4f}  median={np.median(k):.4f}  std={np.std(k):.4f}'
            if label in cos_sims_by_label:
                c = cos_sims_by_label[label]
                line += f'  cos_sim={np.mean(c):.4f}±{np.std(c):.4f}'
            print(line)

    stat_results = {}
    for la, lb in [('ironic','literal'), ('ironic','control'), ('literal','control')]:
        if la not in kappas or lb not in kappas:
            continue
        a, b = kappas[la], kappas[lb]
        U, p = stats.mannwhitneyu(a, b, alternative='two-sided')
        pooled = np.sqrt((np.var(a) + np.var(b)) / 2)
        d = (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else 0.0
        r = 1 - (2*U) / (len(a)*len(b))
        stat_results[f'{la}_vs_{lb}'] = {'U': U, 'p_value': p, 'cohens_d': d, 'rank_biserial_r': r}
        print(f'  {la} vs {lb}: U={U:.0f}  p={fmt_p(p)}  d={d:+.3f}  r={r:+.3f}')

    # Cosine similarity tests (Method 2)
    if cos_sims_by_label:
        print()
        print('  Cosine similarity (higher = flatter transport):')
        for la, lb in [('ironic','literal'), ('ironic','control'), ('literal','control')]:
            if la in cos_sims_by_label and lb in cos_sims_by_label:
                a, b = cos_sims_by_label[la], cos_sims_by_label[lb]
                U, p = stats.mannwhitneyu(a, b, alternative='two-sided')
                print(f'  {la} vs {lb}: U={U:.0f}  p={fmt_p(p)}  '
                      f'means={np.mean(a):.4f} vs {np.mean(b):.4f}')

    # ── 6. Paired comparison ─────────────────────────────────────────────
    hbar('Paired Comparison (Same Sentence, Different Context)')

    ironic_by_pair = {hr.metadata['pair_id']: hr.kappa_mean for hr in results_by_label['ironic']}
    literal_by_pair = {hr.metadata['pair_id']: hr.kappa_mean for hr in results_by_label['literal']}
    shared = sorted(set(ironic_by_pair) & set(literal_by_pair))

    n_higher = 0
    for pid in shared:
        ik, lk = ironic_by_pair[pid], literal_by_pair[pid]
        arrow = '>' if ik > lk else '<'
        tag = '*' if ik > lk else ' '
        # Find the target text
        target = ''
        for sp in all_pairs:
            if sp.pair_id == pid and sp.target:
                target = sp.target
                break
        print(f'  {tag} pair {pid:2d}: ironic={ik:.4f} {arrow} literal={lk:.4f}  "{target}"')
        if ik > lk:
            n_higher += 1

    pct = 100 * n_higher / len(shared) if shared else 0
    print(f'\n  Ironic > Literal in {n_higher}/{len(shared)} pairs ({pct:.0f}%)')

    if len(shared) >= 5:
        paired_ironic = [ironic_by_pair[p] for p in shared]
        paired_literal = [literal_by_pair[p] for p in shared]
        stat_w, p_w = stats.wilcoxon(paired_ironic, paired_literal, alternative='two-sided')
        print(f'  Wilcoxon signed-rank: W={stat_w:.0f}, p={fmt_p(p_w)}')

    # ── 6b. Permutation test (10k shuffles) ───────────────────────────────
    hbar('Permutation Tests (10,000 shuffles)')
    N_PERM = 10_000
    rng = np.random.RandomState(42)

    for la, lb in [('ironic','literal'), ('ironic','control'), ('literal','control')]:
        if la not in kappas or lb not in kappas:
            continue
        a, b = kappas[la], kappas[lb]
        obs_diff = np.mean(a) - np.mean(b)
        pooled = np.concatenate([a, b])
        na = len(a)
        count = 0
        for _ in range(N_PERM):
            rng.shuffle(pooled)
            perm_diff = np.mean(pooled[:na]) - np.mean(pooled[na:])
            if abs(perm_diff) >= abs(obs_diff):
                count += 1
        p_perm = (count + 1) / (N_PERM + 1)
        print(f'  {la} vs {lb}: obs_diff={obs_diff:+.4f}  p_perm={fmt_p(p_perm)}')

    # Paired permutation test (sign-flip)
    if len(shared) >= 5:
        paired_diffs = np.array(paired_ironic) - np.array(paired_literal)
        obs_mean_diff = np.mean(paired_diffs)
        count = 0
        for _ in range(N_PERM):
            signs = rng.choice([-1, 1], size=len(paired_diffs))
            if abs(np.mean(paired_diffs * signs)) >= abs(obs_mean_diff):
                count += 1
        p_paired_perm = (count + 1) / (N_PERM + 1)
        print(f'  paired ironic-literal: obs_diff={obs_mean_diff:+.4f}  p_perm={fmt_p(p_paired_perm)}')

    # ── 6c. Length-controlled analysis ─────────────────────────────────────
    hbar('Length-Controlled Analysis')

    # Gather (kappa, n_tokens, label) for all sentences
    all_kappas = []
    all_lengths = []
    all_labels = []
    for label, hrs in results_by_label.items():
        for hr in hrs:
            all_kappas.append(hr.kappa_mean)
            all_lengths.append(hr.metadata['n_tokens'])
            all_labels.append(label)

    all_kappas = np.array(all_kappas)
    all_lengths = np.array(all_lengths, dtype=float)
    all_labels = np.array(all_labels)

    # Print mean lengths per group
    for label in ['ironic', 'literal', 'control']:
        mask = all_labels == label
        print(f'  {label:8s}: mean_len={np.mean(all_lengths[mask]):.1f}  '
              f'mean_kappa={np.mean(all_kappas[mask]):.4f}')

    # Pearson correlation: kappa vs length
    r_len, p_len = stats.pearsonr(all_lengths, all_kappas)
    print(f'  kappa ~ length: r={r_len:+.3f}  p={fmt_p(p_len)}')

    # Residualize kappa against length (simple linear regression)
    slope, intercept = np.polyfit(all_lengths, all_kappas, 1)
    residuals = all_kappas - (slope * all_lengths + intercept)
    print(f'  Regression: kappa = {slope:+.5f} * n_tokens + {intercept:.4f}')

    # Re-test on residuals
    resid_by_label = {}
    for label in ['ironic', 'literal', 'control']:
        mask = all_labels == label
        resid_by_label[label] = residuals[mask]
        print(f'  {label:8s} residual: mean={np.mean(residuals[mask]):+.4f}  '
              f'std={np.std(residuals[mask]):.4f}')

    print()
    print('  Length-controlled Mann-Whitney (on residuals):')
    for la, lb in [('ironic','literal'), ('ironic','control'), ('literal','control')]:
        if la in resid_by_label and lb in resid_by_label:
            a, b = resid_by_label[la], resid_by_label[lb]
            U, p = stats.mannwhitneyu(a, b, alternative='two-sided')
            d_val = (np.mean(a) - np.mean(b)) / np.sqrt((np.var(a) + np.var(b)) / 2) if (np.var(a) + np.var(b)) > 0 else 0
            print(f'  {la} vs {lb}: U={U:.0f}  p={fmt_p(p)}  d={d_val:+.3f}')

    # ── 6d. Bootstrap CIs on paired differences ───────────────────────────
    hbar('Bootstrap CIs (Paired Ironic - Literal)')
    N_BOOT = 10_000

    if len(shared) >= 5:
        paired_diffs = np.array(paired_ironic) - np.array(paired_literal)
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
    else:
        print('  (Not enough paired data)')

    # ── 7. Plots ─────────────────────────────────────────────────────────
    hbar('Generating Plots')

    exp_result = ExperimentResult(
        results_by_label=results_by_label,
        ironic_vs_literal=stat_results.get('ironic_vs_literal'),
        ironic_vs_control=stat_results.get('ironic_vs_control'),
        literal_vs_control=stat_results.get('literal_vs_control'),
        asymmetry_by_label=asymmetry_by_label,
        metadata={'model': MODEL_NAME, 'method': METHOD},
    )

    plot_holonomy_distributions(exp_result, str(OUTPUT_DIR / 'holonomy_distributions.png'))
    plot_paired_comparison(exp_result, str(OUTPUT_DIR / 'paired_comparison.png'))

    # ── 8. Save raw kappas ───────────────────────────────────────────────
    import json
    summary = {
        'model': MODEL_NAME,
        'method': METHOD,
        'device': DEVICE,
        'stats': stat_results,
    }
    for label, k in kappas.items():
        summary[f'{label}_kappas'] = k.tolist()
    if asymmetry_by_label:
        for label, a in asymmetry_by_label.items():
            summary[f'{label}_asymmetry'] = a.tolist()

    with open(OUTPUT_DIR / 'holonomy_results.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # ── Done ─────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    hbar('Done')
    print(f'  Time:    {elapsed:.0f}s')
    print(f'  Results: {OUTPUT_DIR}/')
    print(f'  Plots:   asymmetry.png, holonomy_distributions.png, paired_comparison.png')


if __name__ == '__main__':
    main()
