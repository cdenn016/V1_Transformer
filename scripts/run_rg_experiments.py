#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RG Universality Experiments -- Phase 0 & Phase 1
=================================================

Concrete, runnable experiments testing the RG universality claim:
"The standard transformer is a stable IR fixed point of the gauge VFE."

Extracts attention matrices (beta) and belief states (mu, sigma) from
trained GaugeTransformerLM checkpoints, runs RG coarse-graining, and
measures scaling exponents. Sigma can be diagonal (B, N, K) or full
(B, N, K, K) depending on the model's diagonal_covariance setting.

PHASE 0 (post-hoc, no training needed):
    0A: RG coarse-graining exponents from existing checkpoints
    0B: Emergent anisotropy measurement (Var_A(mu) decomposition)

PHASE 1 (training runs):
    1A: Sample-efficiency comparison at varying K
    1B: Scaling exponent comparison (same/different beta?)

Usage:
    python scripts/run_rg_experiments.py
"""

import sys
import os

# Ensure project root is importable
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
import torch.nn.functional as F
import numpy as np
import json
import csv
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional


# ============================================================================
# PHASE 0A: RG COARSE-GRAINING EXPONENTS (HF6.3)
# ============================================================================

def phase_0a_coarse_graining_exponents(checkpoint_path: str, n_samples: int = 50):
    """Extract attention and beliefs from a trained model, run RG flow.

    Tests the predicted scaling exponents: y1 = -1/2, y2 = -1, y3 = -2.
    Attention beta and belief states (mu, sigma) are extracted from the
    model's forward_with_attention pass. Sigma is handled as either
    diagonal (N, K) or full (N, K, K).

    Args:
        checkpoint_path: Path to a trained GaugeTransformerLM .pt file.
        n_samples: Number of validation-set samples to analyze.
    """
    from transformer.utils.checkpoint import load_checkpoint, get_tokenizer
    from transformer.core.model import GaugeTransformerLM
    from transformer.data import create_dataloaders

    print("=" * 70)
    print("PHASE 0A: RG Coarse-Graining Exponents [HF6.3]")
    print("=" * 70)
    print(f"Checkpoint: {checkpoint_path}")

    # --- Load model ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = load_checkpoint(checkpoint_path, device=str(device))

    config = checkpoint.get('config', {})
    # Handle both dict and json-string configs
    if isinstance(config, str):
        config = json.loads(config)

    model = GaugeTransformerLM(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device).eval()

    K = config.get('embed_dim', 10)
    seq_len = config.get('max_seq_len', 128)
    print(f"Model: K={K}, seq_len={seq_len}")

    # --- Get validation data ---
    _, val_loader, vocab_size = create_dataloaders(
        max_seq_len=seq_len,
        batch_size=1,
        dataset=config.get('dataset', 'wikitext-103'),
    )

    # --- Collect attention + belief data ---
    all_results = []
    sample_count = 0

    for batch_idx, (input_ids, targets) in enumerate(val_loader):
        if sample_count >= n_samples:
            break

        input_ids = input_ids.to(device)
        with torch.no_grad():
            logits, attn_info = model.forward_with_attention(input_ids)

        # Extract numpy arrays
        beta = attn_info['beta']  # (n_layers, B, n_heads, N, N)
        mu = attn_info['mu']      # (B, N, K)
        sigma = attn_info.get('sigma', None)  # (B, N, K) or (B, N, K, K)

        # Average over heads for RG analysis, take first layer
        beta_avg = beta[0, 0].mean(dim=0).cpu().numpy()  # (N, N)
        mu_np = mu[0].cpu().numpy()                        # (N, K)

        # Build covariance matrices
        if sigma is not None:
            sigma_np = sigma[0].cpu().numpy()  # (N, K) or (N, K, K)
            if sigma_np.ndim == 2:
                # Diagonal: expand to full matrices
                N_tok, K_dim = sigma_np.shape
                covs = np.zeros((N_tok, K_dim, K_dim))
                for i in range(N_tok):
                    covs[i] = np.diag(sigma_np[i])
            else:
                covs = sigma_np
        else:
            N_tok = mu_np.shape[0]
            covs = np.eye(K)[None].repeat(N_tok, axis=0)

        all_results.append({
            'beta': beta_avg,
            'mu': mu_np,
            'covs': covs,
        })
        sample_count += 1

        if sample_count % 10 == 0:
            print(f"  Collected {sample_count}/{n_samples} samples...")

    # --- Run RG coarse-graining ---
    from scripts.rg_universality_networkx import (
        run_rg_flow, measure_couplings, RGFlow,
    )

    print(f"\nRunning RG flow on {len(all_results)} samples...")

    all_exponents = {'y1': [], 'y2': [], 'y3': []}

    for i, result in enumerate(all_results):
        flow = run_rg_flow(
            result['beta'], result['mu'], result['covs'],
            transports=None,  # No gauge transport for standard models
            max_levels=4, min_nodes=4,
        )

        exponents = flow.scaling_exponents()
        for key in ['y1', 'y2', 'y3']:
            if key in exponents and not np.isnan(exponents[key]):
                all_exponents[key].append(exponents[key])

    # --- Report ---
    print("\n" + "-" * 50)
    print("RESULTS: Measured Scaling Exponents")
    print("-" * 50)
    print(f"{'Coupling':>15} {'Measured':>10} {'Predicted':>10} {'N_samples':>10}")
    for key, predicted in [('y1', -0.5), ('y2', -1.0), ('y3', -2.0)]:
        vals = all_exponents[key]
        if vals:
            mean = np.mean(vals)
            std = np.std(vals) / np.sqrt(len(vals))
            print(f"{key:>15} {mean:>10.3f} ± {std:.3f} {predicted:>10.1f} {len(vals):>10}")
        else:
            print(f"{key:>15} {'N/A':>10} {predicted:>10.1f} {0:>10}")

    print("\nInterpretation:")
    print("  y₁ < 0 → anisotropy is IRRELEVANT (decays under coarse-graining)")
    print("  y₂ < 0 → gauge variation is IRRELEVANT")
    print("  y₃ < 0 → holonomy is IRRELEVANT")
    print("  All negative → transformer is a STABLE IR fixed point ✓")

    return all_exponents


# ============================================================================
# PHASE 0B: EMERGENT ANISOTROPY MEASUREMENT (HF6.4)
# ============================================================================

def phase_0b_emergent_anisotropy(checkpoint_path: str, n_samples: int = 50):
    """
    Measure emergent anisotropy: even starting isotropic, meta-agent
    covariances become anisotropic from within-cluster mean variance.

    Args:
        checkpoint_path: Path to trained model
        n_samples: Number of samples to analyze
    """
    from transformer.utils.checkpoint import load_checkpoint
    from transformer.core.model import GaugeTransformerLM
    from transformer.data import create_dataloaders
    from scripts.rg_universality_networkx import spectral_communities

    print("\n" + "=" * 70)
    print("PHASE 0B: Emergent Anisotropy Measurement [HF6.4]")
    print("=" * 70)
    print(f"Checkpoint: {checkpoint_path}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = load_checkpoint(checkpoint_path, device=str(device))
    config = checkpoint.get('config', {})
    if isinstance(config, str):
        config = json.loads(config)

    model = GaugeTransformerLM(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device).eval()

    K = config.get('embed_dim', 10)
    seq_len = config.get('max_seq_len', 128)

    _, val_loader, _ = create_dataloaders(
        max_seq_len=seq_len, batch_size=1,
        dataset=config.get('dataset', 'wikitext-103'),
    )

    aniso_original = []
    aniso_emergent = []
    aniso_total = []

    for batch_idx, (input_ids, _) in enumerate(val_loader):
        if batch_idx >= n_samples:
            break

        input_ids = input_ids.to(device)
        with torch.no_grad():
            logits, attn_info = model.forward_with_attention(input_ids)

        beta = attn_info['beta'][0, 0].mean(dim=0).cpu().numpy()
        mu = attn_info['mu'][0].cpu().numpy()
        sigma = attn_info.get('sigma', None)

        if sigma is not None:
            sigma_np = sigma[0].cpu().numpy()
            if sigma_np.ndim == 2:
                N_tok, K_dim = sigma_np.shape
                covs = np.zeros((N_tok, K_dim, K_dim))
                for i in range(N_tok):
                    covs[i] = np.diag(sigma_np[i])
            else:
                covs = sigma_np
        else:
            N_tok = mu.shape[0]
            covs = np.eye(K)[None].repeat(N_tok, axis=0)

        # Cluster tokens
        labels = spectral_communities(beta)
        unique = np.unique(labels)

        for lbl in unique:
            members = np.where(labels == lbl)[0]
            if len(members) < 3:
                continue

            # Original anisotropy: average of individual Σ_i
            avg_cov = covs[members].mean(axis=0)
            sigma2_orig = np.trace(avg_cov) / K
            delta_orig = avg_cov - sigma2_orig * np.eye(K)
            g1_orig = np.linalg.norm(delta_orig) / (sigma2_orig + 1e-10)

            # Within-cluster variance (emergent anisotropy)
            mu_A = mu[members].mean(axis=0)
            devs = mu[members] - mu_A
            within_var = (devs.T @ devs) / len(members)
            sigma2_wv = np.trace(within_var) / K
            delta_wv = within_var - sigma2_wv * np.eye(K)
            g1_emergent = np.linalg.norm(delta_wv) / (sigma2_wv + 1e-10) if sigma2_wv > 1e-10 else 0

            # Total meta-agent covariance
            total_cov = avg_cov + within_var
            sigma2_total = np.trace(total_cov) / K
            delta_total = total_cov - sigma2_total * np.eye(K)
            g1_total = np.linalg.norm(delta_total) / (sigma2_total + 1e-10)

            aniso_original.append(g1_orig)
            aniso_emergent.append(g1_emergent)
            aniso_total.append(g1_total)

    print(f"\nAnalyzed {len(aniso_original)} meta-agent clusters across {n_samples} samples")
    print("\n" + "-" * 50)
    print("RESULTS: Anisotropy Decomposition")
    print("-" * 50)
    print(f"{'Component':>20} {'Mean g₁':>10} {'Std':>10}")
    print(f"{'Original (Σ_i)':>20} {np.mean(aniso_original):>10.4f} {np.std(aniso_original):>10.4f}")
    print(f"{'Emergent (Var_A(μ))':>20} {np.mean(aniso_emergent):>10.4f} {np.std(aniso_emergent):>10.4f}")
    print(f"{'Total (Σ_A)':>20} {np.mean(aniso_total):>10.4f} {np.std(aniso_total):>10.4f}")

    print("\nInterpretation:")
    if np.mean(aniso_emergent) > 0.1:
        print("  ✓ SIGNIFICANT emergent anisotropy detected!")
        print("    This is the structure that transformers must absorb into W_Q, W_K")
        print("    while the gauge VFE tracks it explicitly in Σ_i.")
    else:
        print("  ✗ Emergent anisotropy is small — within-cluster means are isotropic")
        print("    This could mean the model hasn't trained long enough, or K is small")

    ratio = np.mean(aniso_emergent) / (np.mean(aniso_original) + 1e-10)
    print(f"\n  Emergent / Original ratio: {ratio:.2f}")
    print(f"  If > 1: emergent anisotropy DOMINATES → strong case for gauge VFE")
    print(f"  If < 1: original anisotropy dominates → model already captures structure")

    return {
        'original': aniso_original,
        'emergent': aniso_emergent,
        'total': aniso_total,
    }


# ============================================================================
# PHASE 1A: SAMPLE-EFFICIENCY COMPARISON (HF6.1)
# ============================================================================

def phase_1a_sample_efficiency(
    K_values: List[int] = [8, 16, 32, 64],
    dataset: str = 'wikitext-103',
    max_steps: int = 15000,
    target_ppl: float = 150.0,
    output_dir: str = 'checkpoints_rg_experiments',
):
    """
    Train gauge VFE and standard transformer at each K,
    record tokens-to-target-PPL.

    Key prediction: R(K) = tokens_TF / tokens_VFE grows with K.

    Args:
        K_values: Belief dimensions to test
        dataset: WikiText variant
        max_steps: Max training steps per run
        target_ppl: Perplexity target for sample-efficiency comparison
        output_dir: Where to save results
    """
    from transformer.core.model import GaugeTransformerLM
    from transformer.baselines.standard_transformer import StandardTransformerLM
    from transformer.data import create_dataloaders
    from transformer.train import compute_free_energy_loss
    from transformer.training.train_fast import FastTrainer, FastTrainingConfig

    print("=" * 70)
    print("PHASE 1A: Sample-Efficiency Comparison [HF6.1]")
    print("=" * 70)
    print(f"K values: {K_values}")
    print(f"Dataset: {dataset}")
    print(f"Target PPL: {target_ppl}")
    print(f"Max steps: {max_steps}")

    output_path = Path(output_dir) / 'phase_1a'
    output_path.mkdir(parents=True, exist_ok=True)

    results = []

    for K in K_values:
        for arch in ['gauge_vfe', 'transformer']:
            run_name = f"K={K}_{arch}"
            print(f"\n{'='*50}")
            print(f"TRAINING: {run_name}")
            print(f"{'='*50}")

            # Build config
            if arch == 'gauge_vfe':
                config = _build_vfe_config(K, max_steps, dataset)
            else:
                config = _build_transformer_config(K, max_steps, dataset)

            config['checkpoint_dir'] = str(output_path / run_name)
            Path(config['checkpoint_dir']).mkdir(parents=True, exist_ok=True)

            # Save config
            with open(output_path / run_name / 'config.json', 'w') as f:
                json.dump(config, f, indent=2, default=str)

            # Use FastTrainer for actual training
            fast_config = FastTrainingConfig(
                embed_dim=K,
                n_layers=1,
                batch_size=config['batch_size'],
                max_steps=max_steps,
                warmup_steps=config.get('warmup_steps', 100),
                checkpoint_dir=config['checkpoint_dir'],
                dataset=dataset,
                max_seq_len=config['max_seq_len'],
                log_interval=100,
                eval_interval=500,
            )

            print(f"  Config saved to {config['checkpoint_dir']}/config.json")
            print(f"  To train, run:")
            print(f"    python transformer/train_publication.py")
            print(f"    (after editing VFE_EM_CONFIG with embed_dim={K})")
            print()

            # Record planned experiment
            results.append({
                'K': K,
                'architecture': arch,
                'config_path': str(output_path / run_name / 'config.json'),
                'status': 'planned',
                'target_ppl': target_ppl,
            })

    # Save experiment plan
    plan_path = output_path / 'experiment_plan.json'
    with open(plan_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"EXPERIMENT PLAN SAVED: {plan_path}")
    print(f"{'='*70}")
    print(f"\n{len(results)} training runs planned.")
    print(f"\nTo execute, modify train_publication.py configs for each K and run.")
    print(f"Or use the generated configs in {output_path}/*/config.json")

    _print_phase1_instructions(K_values, output_path)

    return results


def _build_vfe_config(K: int, max_steps: int, dataset: str) -> dict:
    """Build a gauge VFE config for a given belief dimension K.

    Gauge group and head count are determined by K: smaller K uses
    single-head GL(K), larger K splits across multiple heads with
    reduced irrep dimension.
    """
    # Determine gauge group dimensions
    # For small K, use single head; for larger K, multi-head
    if K <= 16:
        irrep_dim = K
        n_heads = 1
    elif K <= 32:
        irrep_dim = K // 2
        n_heads = 2
    else:
        irrep_dim = K // 4
        n_heads = 4

    return {
        'vocab_size': 50257,
        'embed_dim': K,
        'n_layers': 1,
        'max_seq_len': 128,
        'batch_size': 64 if K <= 32 else 32,  # Reduce batch for larger K

        # VFE settings
        'ffn_mode': 'VFE_dynamic',
        'evolve_sigma': True,
        'evolve_phi': True,
        'diagonal_covariance': True,
        'gauge_group': 'GLK',
        'gauge_mode': 'learned',
        'gauge_dim': irrep_dim,
        'irrep_spec': [('fund', n_heads, irrep_dim)],

        # Training
        'max_steps': max_steps,
        'warmup_steps': 100,
        'mu_lr': 0.05,
        'sigma_lr': 0.005,
        'phi_lr': 0.005,
        'ffn_lr': 0.05,
        'output_lr': 0.05,
        'alpha': 0.075,
        'weight_decay': 0.01,
        'grad_clip': 1.0,
        'use_rope': True,
        'use_layernorm': True,
        'use_residual': True,

        'ffn_n_iterations': 1,
        'ffn_alpha': 1.0,
        'ffn_lambda_belief': 1.0,

        'log_interval': 100,
        'eval_interval': 500,
        'checkpoint_interval': 5000,
    }


def _build_transformer_config(K: int, max_steps: int, dataset: str) -> dict:
    """Build a standard transformer config, parameter-matched to gauge VFE.

    Extra parameters that VFE spends on sigma and phi are redirected
    to the MLP hidden dimension for a fair comparison.
    """
    # Standard transformer at matched embedding dimension
    # MLP hidden dim absorbs the parameters VFE spends on σ, φ
    n_heads = max(1, K // 8)
    # VFE extra params: K (sigma) + K*gauge_dim (phi) per token type
    # Approximately: 50257 * (K + K²/n_heads) extra params
    # Give these to MLP instead
    extra_params_approx = 50257 * K * 2
    hidden_dim = max(4 * K, extra_params_approx // (2 * K))
    hidden_dim = min(hidden_dim, 32768)  # Cap

    return {
        'vocab_size': 50257,
        'embed_dim': K,
        'n_layers': 1,
        'n_heads': n_heads,
        'hidden_dim': hidden_dim,
        'max_seq_len': 128,
        'batch_size': 64 if K <= 32 else 32,

        # Standard transformer
        'ffn_mode': 'standard',
        'attention_type': 'standard',
        'evolve_sigma': False,
        'evolve_phi': False,
        'pos_encoding_mode': 'learned',

        # Training
        'max_steps': max_steps,
        'warmup_steps': 100,
        'mu_lr': 3e-4,
        'weight_decay': 0.01,
        'grad_clip': 1.0,

        'log_interval': 100,
        'eval_interval': 500,
        'checkpoint_interval': 5000,
    }


def _print_phase1_instructions(K_values: List[int], output_path: Path):
    """Print concrete instructions for running Phase 1."""
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║  HOW TO RUN PHASE 1 EXPERIMENTS                                 ║
╚══════════════════════════════════════════════════════════════════╝

For each K in {K_values}, train TWO models:

  1. GAUGE VFE:
     Edit transformer/train_publication.py → VFE_EM_CONFIG:
       'embed_dim': K,
       'irrep_spec': [('fund', n_heads, K // n_heads)],
     Then: python transformer/train_publication.py

  2. STANDARD TRANSFORMER:
     Edit transformer/train_publication.py → change DEFAULT_MODE = 'standard'
     Edit STANDARD_CONFIG:
       'embed_dim': K,
       'n_heads': max(1, K // 8),
     Then: python transformer/train_publication.py

  3. COLLECT RESULTS:
     After each run, the metrics.csv in the checkpoint dir has
     per-step perplexity. Find the step where PPL first drops
     below the target.

ESTIMATED TIME (RTX 5090, WikiText-103):
  K=8:   ~30 min per model  (2 models = 1 hour)
  K=16:  ~1 hour per model  (2 models = 2 hours)
  K=32:  ~3 hours per model (2 models = 6 hours)
  K=64:  ~8 hours per model (2 models = 16 hours)
  Total: ~25 GPU-hours (~3 days if serial, ~1.5 days if parallel)

ANALYSIS:
  After all runs complete:
    python scripts/run_rg_experiments.py --phase 1-analyze \\
        --results-dir {output_path}

  This will:
    - Plot PPL vs tokens for VFE and transformer at each K
    - Compute tokens-to-target for each
    - Fit R(K) = tokens_TF / tokens_VFE ~ K^δ
    - Test HF6.1: is δ > 0? (predicted: δ ≈ 0.5)
    - Test HF6.2: is β_VFE ≈ β_TF? (same universality class)
""")


# ============================================================================
# PHASE 1 ANALYSIS: Post-training analysis of sample efficiency
# ============================================================================

def phase_1_analyze(results_dir: str):
    """
    Analyze Phase 1 training curves after all runs complete.

    Reads metrics.csv files from each run, computes:
    1. Tokens-to-target-PPL ratio R(K)
    2. Scaling exponent β for PPL(D) = A·D^{-β}
    3. Fit R(K) ~ K^δ
    """
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Results directory not found: {results_dir}")
        return

    print("=" * 70)
    print("PHASE 1 ANALYSIS: Sample-Efficiency & Scaling Exponents")
    print("=" * 70)

    # Collect all metrics.csv files
    K_results = {}

    for run_dir in sorted(results_path.iterdir()):
        if not run_dir.is_dir():
            continue

        metrics_file = run_dir / 'metrics.csv'
        if not metrics_file.exists():
            # Try finding it in subdirectories
            for f in run_dir.rglob('metrics.csv'):
                metrics_file = f
                break

        if not metrics_file.exists():
            print(f"  No metrics.csv in {run_dir.name}, skipping")
            continue

        # Parse run name: K=XX_architecture
        name = run_dir.name
        try:
            parts = name.split('_')
            K = int(parts[0].split('=')[1])
            arch = '_'.join(parts[1:])
        except (IndexError, ValueError):
            print(f"  Can't parse run name: {name}, skipping")
            continue

        # Read metrics
        steps, ppls, tokens = [], [], []
        with open(metrics_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                step = int(row.get('step', 0))
                ppl = float(row.get('val_ppl', row.get('ppl', 0)))
                tok = float(row.get('tokens_seen', step * 128 * 64))
                if ppl > 0:
                    steps.append(step)
                    ppls.append(ppl)
                    tokens.append(tok)

        if not ppls:
            continue

        if K not in K_results:
            K_results[K] = {}
        K_results[K][arch] = {
            'steps': np.array(steps),
            'ppl': np.array(ppls),
            'tokens': np.array(tokens),
        }

        print(f"  {name}: {len(ppls)} datapoints, "
              f"final PPL={ppls[-1]:.1f}")

    if not K_results:
        print("\nNo results found! Run Phase 1 training first.")
        return

    # --- Compute tokens-to-target ---
    target_ppl = 150.0
    print(f"\n--- Tokens to reach PPL = {target_ppl} ---")
    print(f"{'K':>5} {'VFE tokens':>15} {'TF tokens':>15} {'Ratio R(K)':>12}")

    ratios = {}
    for K in sorted(K_results.keys()):
        vfe_data = K_results[K].get('gauge_vfe', {})
        tf_data = K_results[K].get('transformer', {})

        vfe_tokens = _tokens_to_target(vfe_data, target_ppl)
        tf_tokens = _tokens_to_target(tf_data, target_ppl)

        ratio = tf_tokens / vfe_tokens if vfe_tokens > 0 and tf_tokens > 0 else np.nan
        ratios[K] = ratio

        vfe_str = f"{vfe_tokens:.0f}" if vfe_tokens > 0 else "N/A"
        tf_str = f"{tf_tokens:.0f}" if tf_tokens > 0 else "N/A"
        ratio_str = f"{ratio:.2f}" if not np.isnan(ratio) else "N/A"
        print(f"{K:>5} {vfe_str:>15} {tf_str:>15} {ratio_str:>12}")

    # --- Fit R(K) ~ K^δ ---
    valid_K = [k for k in sorted(ratios.keys()) if not np.isnan(ratios[k])]
    if len(valid_K) >= 2:
        log_K = np.log(np.array(valid_K, dtype=float))
        log_R = np.log(np.array([ratios[k] for k in valid_K]))
        coeffs = np.polyfit(log_K, log_R, 1)
        delta = coeffs[0]

        print(f"\n--- Scaling Fit: R(K) ~ K^δ ---")
        print(f"  δ = {delta:.3f}")
        print(f"  Predicted: δ ≈ 0.5")
        if delta > 0:
            print(f"  ✓ Gauge VFE advantage GROWS with K (δ > 0)")
        else:
            print(f"  ✗ No K-dependent advantage detected (δ ≤ 0)")

    # --- Fit scaling exponents β ---
    print(f"\n--- Scaling Exponents: PPL(D) ~ D^{{-β}} ---")
    for K in sorted(K_results.keys()):
        for arch in ['gauge_vfe', 'transformer']:
            data = K_results[K].get(arch, {})
            if 'tokens' not in data or len(data['tokens']) < 5:
                continue
            log_D = np.log(data['tokens'])
            log_ppl = np.log(data['ppl'])
            coeffs = np.polyfit(log_D, log_ppl, 1)
            beta = -coeffs[0]
            print(f"  K={K}, {arch}: β = {beta:.4f}")


def _tokens_to_target(data: dict, target: float) -> float:
    """Find tokens where PPL first drops below target."""
    if 'ppl' not in data or 'tokens' not in data:
        return -1
    ppls = data['ppl']
    tokens = data['tokens']
    below = np.where(ppls < target)[0]
    if len(below) > 0:
        return tokens[below[0]]
    return -1


# ============================================================================
# MAIN
# ============================================================================

def find_latest_checkpoint() -> Optional[str]:
    """Find the most recent checkpoint in common locations."""
    search_dirs = [
        Path('checkpoints_publication'),
        Path('checkpoints'),
        Path('Attention'),
    ]
    latest = None
    latest_time = 0

    for d in search_dirs:
        if d.exists():
            for pt in d.rglob('*.pt'):
                mtime = pt.stat().st_mtime
                if mtime > latest_time:
                    latest_time = mtime
                    latest = str(pt)

    # Also check experiment configs
    for d in search_dirs:
        if d.exists():
            for cfg in d.rglob('experiment_config.json'):
                mtime = cfg.stat().st_mtime
                if mtime > latest_time:
                    latest_time = mtime
                    # Look for .pt in same dir
                    for pt in cfg.parent.glob('*.pt'):
                        latest = str(pt)
                        latest_time = pt.stat().st_mtime

    return latest


if __name__ == '__main__':

    # ================================================================
    # CONFIG — edit these settings, then click Run (no CLI needed)
    # ================================================================

    # Which phases to run (set to True/False):
    RUN_PHASE_0  = True     # Post-hoc analysis from checkpoint
    RUN_PHASE_1  = True     # Training sweep across K values
    RUN_ANALYZE  = True     # Analyze Phase 1 results after training

    # Phase 0 settings:
    CHECKPOINT_PATH = None  # <-- paste your checkpoint path here as a raw string, e.g.:
                            #     r'C:\Users\name\checkpoints\best_model.pt'
                            #     r'/home/user/checkpoints/best_model.pt'
                            # If None, auto-finds the latest checkpoint.

    N_SAMPLES = 50          # Number of text samples for Phase 0 analysis

    # Phase 1 settings:
    K_VALUES   = [8, 16, 32, 64]
    DATASET    = 'wikitext-103'
    MAX_STEPS  = 15000
    OUTPUT_DIR = 'checkpoints_rg_experiments'

    # Phase 1-analyze settings:
    RESULTS_DIR = None      # If None, defaults to OUTPUT_DIR/phase_1a

    # ================================================================
    # RUN — no need to edit below this line
    # ================================================================

    if RUN_PHASE_0:
        ckpt = CHECKPOINT_PATH
        if ckpt is None:
            ckpt = find_latest_checkpoint()
            if ckpt:
                print(f"Auto-found latest checkpoint: {ckpt}")

        if ckpt is None:
            print("⚠  Phase 0 skipped: no checkpoint found.")
            print("   Set CHECKPOINT_PATH above, or train a model first.")
        else:
            phase_0a_coarse_graining_exponents(ckpt, n_samples=N_SAMPLES)
            phase_0b_emergent_anisotropy(ckpt, n_samples=N_SAMPLES)

    if RUN_PHASE_1:
        phase_1a_sample_efficiency(
            K_values=K_VALUES,
            dataset=DATASET,
            max_steps=MAX_STEPS,
            output_dir=OUTPUT_DIR,
        )

    if RUN_ANALYZE:
        results_dir = RESULTS_DIR or OUTPUT_DIR + '/phase_1a'
        if Path(results_dir).exists():
            phase_1_analyze(results_dir)
        else:
            print(f"⚠  Phase 1-analyze skipped: {results_dir} not found.")
            print("   Run Phase 1 first, or set RESULTS_DIR above.")
