"""
Optuna Hyperparameter Optimization for VFE Gauge Transformer
=============================================================

Two-phase budget-aware HPO:
  Phase 1: Coarse scan (short runs, many trials, LHS + TPE)
  Phase 2: Fine-tune (longer runs, top configs, multi-seed)

Usage: Edit HPO_CONFIG below, then run this file.
       No CLI arguments (click-to-run per CLAUDE.md).

Dependencies: pip install optuna
Optional:     pip install optuna-dashboard  (web UI for results)
"""

import copy
import gc
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure project root is on sys.path so 'transformer' package is importable
# regardless of whether this script is run from scripts/ or project root.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(_PROJECT_ROOT)  # match click-to-run working directory convention

import numpy as np
import torch
import optuna
from optuna.samplers import TPESampler, QMCSampler


# ============================================================================
# CONFIGURATION — Edit this dict, then press Run
# ============================================================================

HPO_CONFIG = {
    # Study identity
    'study_name': 'vfe_hpo_v1',
    'output_dir': 'hpo_results',

    # Phase 1: coarse exploration
    'n_trials_phase1': 40,
    'max_steps_phase1': 1000,        # ~2 min/trial at K=20
    'n_startup_trials': 25,         # LHS before switching to TPE

    # Phase 2: multi-seed refinement
    'n_trials_phase2': 15,
    'max_steps_phase2': 3000,       # ~12 min/trial
    'seeds_phase2': [6],

    # Infrastructure
    'dataset': 'wikitext-103',
    'device': 'cuda',
    'batch_size': 256,
    'num_workers': 8,
}

# Base config — matches run_ablation_suite.py BASELINE_CONFIG.
# Only the searched params get overridden per trial.
BASE_CONFIG = {
   # === Architecture ===
   'vocab_size':                 50257,
   'embed_dim':                  20,
   'max_seq_len':                64,
   
   'batch_size':                 256, 
   'max_steps':                  3000,
   
   'n_layers':                   1,
   'ffn_n_iterations':           1,
   
   'alpha_divergence':           0.8180,
   #'grad_accumulation_steps': 1,
   #'gradient_checkpoint_vfe': False,
   
   'gauge_dim':                   10,
   'irrep_spec':       [('fund', 2, 10)],

   'use_prior_bank':             True,
   'gauge_fixed_priors':         True,

   'learnable_pb_temperature':   True,    #prior bank temperature
   'mask_self_attention':        False,  # Prevent attention collapse?
 

   'kappa_beta':                 1,
   'kappa_warmup_steps':         7500,  # freeze kappa for first n steps
   'learnable_head_kappa':       True, # If True, learn per-head κ_h via log_kappa_per_head
   'e_step_sigma_floor':         0.01,   # Floor on σ_p inside E-step (caps 1/σ_p at 1/floor)
   
   # === EM gradient-flow mode ===
   'em_mode':                    'ift_phi',

   'cache_decode_priors':        True,
   'skip_attention':             True,   # attention sublayer removed 2026-06-01; always pure-VFE
   
   # === M-step: Optimizer ===  
   'optimizer_type':             'riemannian_adam',# or 'natural_gradient' or 'adamw' or 'riemannian_adam'
   'fisher_ema_decay':           0.95,            # for natural_gradient
   'fisher_damping':             1e-2,              # for natural_gradient


   'use_layernorm':              True,   #breaks gauge equivariance
   'use_residual':               True,
   'use_output_projection':      True,
   'evolve_sigma':               True,
   'evolve_phi':                 True,  #M-step phi evolution
   'evolve_phi_e_step':          True,
   
   'E_learnable_alpha':          True,   # Adaptive α_i = c0/(b0 + KL) per dimension   
   'E_learnable_lr':             True,   # Learnable E-step LR
   
   'min_lr_ratio':               0.1,
   'lr_decay':                   'linear',   #'linear', 'cosine', 'constant'
   
   'norm_type':                  'layernorm',  # 'layernorm' | 'rmsnorm' | 'mahalnorm' | 'none'
   'residual_type':              'additive',    # 'additive': mu_q = mu_q + mu_sub 
                                        # 'delta':    mu_q = mu_q + (mu_sub - mu_normalized),

   # === E-step Weights ===

   'E_alpha':                    1,      # E-step prior coupling weight
   'E_lambda_belief':            1,    # E-step belief alignment weight
   'E_lambda_softmax':           1,
      
   # === E-step Learning Rates ===
   
   'E_mu_q_lr':                  0.3,    # E-step μ step size (whitened, within trust=2.0)
   'E_sigma_q_lr':               0.05,   # E-step σ step size (conservative)    
   'E_phi_lr':                   0.05,   # E-step φ step size

   # === M-step Weights ===        
   
   'M_alpha':                    0.00,   # M-step KL(q||p) self-consistency
   'M_beta':                     0.0,    # M-step belief alignment
   'mass_phi':                   0.0,    # Gauge prior: (mass_φ/2)||φ||²
   'lambda_hyper':               0.00,    # KL(s||h) explicit loss (pulls tokens toward centroid)
   'lambda_gamma':               0,
   # === M-step Learning Rates (AdamW parameter groups) ===
   
   'M_mu_p_lr':                  0.0721,   # M-step prior mean embeddings (μ_p) 0.05
   'M_sigma_p_lr':               0.0417,     # M-step prior covariance embeddings (log σ_p) 0.015
   'M_phi_lr':                   0.0037,    # M-step gauge frame embeddings (φ) 0.0075
   
   # === M-step Other LR's (AdamW parameter groups) ===
   'M_vfe_hyperparam_lr':        0.0952,  # M-step VFE hyperparams (raw_c0, raw_b0, raw_lr) 0.05
   'M_attention_lr':             0.0125,  # M-step attention params (W_O, constant_omega) was0.06
   'M_output_lr':                0.0021,  # M-step output projection (vocab logits) 0.05
   'embed_weight_decay':         0.0002,   # L2 hyper-prior on embeddings (μ_p, σ_p, φ) via AdamW
   'non_embed_weight_decay':     0.0004,  # L2 on non-embedding params (attention, output)

   # === Gauge group: GL(K) with multi-head block-diagonal structure ===
   'gauge_group':                'GLK',
   'gauge_mode':                 'learned',
   'gauge_param':                'phi',

   'diagonal_covariance':        True,
    
   'exact_diagonal_transport':   False,  # exact diagonal transport - more expensive                                        
   'isotropic_covariance':       False, # If True, force Σ = σ²I (scalar variance × identity)
   'enforce_orthogonal':         False,    
   'learnable_reflection':       False,# Per-token s_i ∈ {±1}^K → O(K)  - enforce orthogonal=true with glk
                                       # Set gauge-mode=constant and the above 3 = true for transf limit

   # === Phi gradient geometry ===
   'phi_natural_gradient':       'killing',
   'use_killing_form':           False,
   'killing_form_sym_dampening': 0.4,

   # === Position encoding ===
   'use_rope':                   True,
   'rope_base':                  75,   #75 for N=64, 128
   'rope_full_gauge':            'off',
   'pos_encoding_mode':          'none',

   # === Embedding init ===
   'mu_init_std':                1.5,
   'phi_scale':                  0.5,
   
   'mu_normalize':               False,
   'mu_max_norm':                None,


   # === Logging ===
   'log_interval':               100,
   'eval_interval':              1000,
   'checkpoint_interval':        25000,
   'semantic_analysis_interval': 10000,
   'gauge_geometry_interval':    2000,   # Gauge field Dirichlet energy + invariants
   'fiber_trajectory_interval':  5000,   # Fisher-Rao E-step trajectory (requires ffn_n_iterations > 1)

   # =================================================================
   # NON-FLAT GAUGE TRANSPORT (holonomy)
   # =================================================================
   # When enabled, transport acquires an edge-local connection δ_ij:
   #   phi path:  Ω_ij = exp(φ_i·G) · exp(α·δ_ij·G) · exp(-φ_j·G)
   #   omega path: Ω_ij = Ω_i · exp(α·δ_ij·G) · Ω_j⁻¹
   # δ_ij is zero-initialized so the model starts flat and learns
   # curvature only where the data warrants it.
   # Holonomy H_ijk = Ω_ij·Ω_jk·Ω_ki ≠ I when δ ≠ 0.
   
   'non_flat_transport':         False,        # Enable edge-dependent connection δ_ij
   'cocycle_relaxation':         0.5,          # Scale for δ_ij: 0=flat, 1=fully non-flat    
   'connection_type':            'bilinear',  # 'bilinear' (δ_ij^a = μ_i^T W^a μ_j) | 'mlp'   
   'connection_hidden_dim':      64,   # Hidden dim for MLP connection (ignored for bilinear)   
   'connection_init_scale':      0.01,   # W init scale (0=flat saddle point, 0.01 recommended)    
   'holonomy_penalty':           0.0,  # λ_H · E[‖C_ijk - I‖²_F] regularizer (0 = off)

   # === Cross Head Gauge Couplings ====
   #Option A: couple just 0↔1, head 2 stays independent
   # 'cross_couplings': [(0, 1), (1, 0)],
   # → super-blocks: [20, 10]  (heads 0,1 merged into GL(20), head 2 alone)
   
   # === Layer/iteration diagnostics ===
   'track_layer_diagnostics':     False,
   'track_iteration_diagnostics': False,
   'diagnostics_interval':        25,
   
   
   'tie_embeddings':              False,
   'ffn_mode':                    'VFE_dynamic',
   
   'debug_vfe_grads':             False,
   'verbose_diagnostics':         False,
   

   # === Regularization ===
   'sigma_ce_scale':              0.7,
   'sigma_max':                   12.0,
   'grad_clip':                   50,
   'hidden_dim':                  508,
   
   
   'warmup_steps':                100,
   'num_workers':                 0,   # 0 is faster on Windows (spawn multiprocessing overhead)


    'dataset': 'wikitext-103', #'wiki-2' for quick sweeps
}


# ============================================================================
# SEARCH SPACE
# ============================================================================

def define_search_space(trial: optuna.Trial) -> dict:
    r"""Map an Optuna trial to a VFE config dict.

    Searches over M-step learning rates, VFE term weights, regularization,
    and architectural choices.  E-step LRs and kappa are learned internally
    (E_learnable_lr=True, learnable_head_kappa=True) so they are not searched.
    """
    cfg = copy.deepcopy(BASE_CONFIG)

    # --- M-step learning rates (log-uniform) ---
    cfg['M_mu_p_lr']       = trial.suggest_float('M_mu_p_lr',       0.0001,  0.3,   log=True)
    cfg['M_sigma_p_lr']    = trial.suggest_float('M_sigma_p_lr',    0.01,  0.5,  log=True)
    cfg['M_phi_lr']        = trial.suggest_float('M_phi_lr',        0.001,  0.01,  log=True)
    cfg['M_attention_lr']  = trial.suggest_float('M_attention_lr',  0.01,  0.2,   log=True)
    cfg['M_output_lr']     = trial.suggest_float('M_output_lr',     0.001,  0.2,   log=True)
    cfg['M_vfe_hyperparam_lr'] = trial.suggest_float('M_vfe_hyperparam_lr', 0.0001, 0.3, log=True)

    # --- VFE term weights (the physics of the free energy) ---
    cfg['E_alpha']          = trial.suggest_float('E_alpha',          0.001,   10.0,  log=True)
    cfg['E_lambda_belief']  = trial.suggest_float('E_lambda_belief',  0.001,   10.0,   log=True)
    cfg['E_lambda_softmax'] = trial.suggest_float('E_lambda_softmax', 0.001,   50.0,   log=True)

    # --- M-step VFE loss weights ---
   # cfg['M_alpha']          = trial.suggest_float('M_alpha',          0.0,   1.0)
    #cfg['M_beta']           = trial.suggest_float('M_beta',           0.0,   1.0)
   # cfg['lambda_hyper']     = trial.suggest_float('lambda_hyper',     0.0,   0.2)
   # cfg['mass_phi']         = trial.suggest_float('mass_phi',         0.0,   0.5)

    # --- Regularization ---
    cfg['alpha_divergence']       = trial.suggest_float('alpha_divergence',       0.01,   1.0)
    cfg['embed_weight_decay']     = trial.suggest_float('embed_weight_decay',     1e-4,  0.2,  log=True)
    cfg['non_embed_weight_decay'] = trial.suggest_float('non_embed_weight_decay', 1e-4,  0.2,  log=True)
    cfg['sigma_ce_scale']         = trial.suggest_float('sigma_ce_scale',         0.01,   1.0)
   # cfg['grad_clip']              = trial.suggest_float('grad_clip',             1,   50.0, log=True)

    # --- Embedding initialization ---
   # cfg['mu_init_std']  = trial.suggest_float('mu_init_std',  0.2, 3.0)
   # cfg['phi_scale']    = trial.suggest_float('phi_scale',    0.2, 3.0)

    # --- Architecture (categorical) ---
   # cfg['norm_type']       = trial.suggest_categorical('norm_type', ['layernorm', 'rmsnorm', 'mahalnorm'])
   # cfg['residual_type']   = trial.suggest_categorical('residual_type', ['additive', 'delta'])
   # cfg['ffn_n_iterations'] = trial.suggest_int('ffn_n_iterations', 1, 3)
   # cfg['rope_base']       = trial.suggest_categorical('rope_base', [10, 30, 75, 150, 1000])

    return cfg


# ============================================================================
# OBJECTIVE FUNCTION
# ============================================================================

def _set_seed(seed: int) -> None:
    """Set all RNG seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _cleanup() -> None:
    """Force memory reclamation between trials."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def objective(
    trial: optuna.Trial,
    shared_data: tuple,
    max_steps: int,
    device: torch.device,
    output_dir: Path,
    seed: int = 6,
) -> float:
    r"""Train one config and return val_ppl.

    Args:
        trial: Optuna trial with suggested hyperparameters.
        shared_data: Pre-loaded ``(train_loader, val_loader, test_loader,
            vocab_size, tokenizer)`` tuple.
        max_steps: Training steps for this trial.
        device: CUDA or CPU device.
        output_dir: Root directory for trial outputs.
        seed: Random seed.

    Returns:
        Validation perplexity (lower is better).
    """
    from transformer.training.experiment_runner import run_single_experiment

    cfg = define_search_space(trial)
    cfg['max_steps'] = max_steps
    # Only evaluate at the end — intermediate evals waste time in HPO
    cfg['eval_interval'] = max_steps
    cfg['log_interval'] = max(max_steps // 5, 100)
    cfg['checkpoint_interval'] = max_steps + 1  # no checkpoints during HPO
    cfg['semantic_analysis_interval'] = max_steps + 1  # no semantic analysis

    run_dir = output_dir / f'trial_{trial.number:04d}'
    run_dir.mkdir(parents=True, exist_ok=True)

    _set_seed(seed)

    try:
        t0 = time.time()
        result = run_single_experiment(
            config=cfg,
            ffn_mode='VFE_dynamic',
            device=device,
            checkpoint_dir=run_dir,
            enable_publication_metrics=False,
            quiet=True,
            skip_test_eval=True,
            skip_post_training_viz=True,
            preloaded_data=shared_data,
        )
        elapsed = time.time() - t0

        if result is None:
            return float('inf')

        val_ppl = result.get('final_ppl', float('inf'))

        # Save trial metadata
        trial_meta = {
            'trial_number': trial.number,
            'params': trial.params,
            'val_ppl': val_ppl,
            'elapsed_sec': elapsed,
            'seed': seed,
            'max_steps': max_steps,
        }
        with open(run_dir / 'trial_result.json', 'w') as f:
            json.dump(trial_meta, f, indent=2)

        print(f"  Trial {trial.number}: val_ppl={val_ppl:.2f} ({elapsed:.0f}s)")
        return val_ppl

    except Exception as e:
        print(f"  Trial {trial.number}: FAILED — {e}")
        return float('inf')

    finally:
        _cleanup()


def run_multi_seed(
    params: dict,
    shared_data: tuple,
    max_steps: int,
    device: torch.device,
    output_dir: Path,
    seeds: List[int],
) -> Tuple[float, float]:
    r"""Run a fixed config with multiple seeds.

    Args:
        params: Optuna params dict (from ``trial.params``).
        shared_data: Pre-loaded dataloaders.
        max_steps: Training steps per seed.
        device: CUDA or CPU device.
        output_dir: Directory for this config's runs.
        seeds: List of random seeds.

    Returns:
        ``(mean_ppl, std_ppl)`` across seeds.
    """
    from transformer.training.experiment_runner import run_single_experiment

    cfg = copy.deepcopy(BASE_CONFIG)
    cfg.update(params)
    cfg['max_steps'] = max_steps
    cfg['eval_interval'] = max(max_steps // 3, 500)  # eval 3x during longer Phase 2
    cfg['log_interval'] = max(max_steps // 10, 100)
    cfg['checkpoint_interval'] = max_steps + 1
    cfg['semantic_analysis_interval'] = max_steps + 1

    ppls = []
    for seed in seeds:
        seed_dir = output_dir / f'seed_{seed}'
        seed_dir.mkdir(parents=True, exist_ok=True)

        _set_seed(seed)

        try:
            result = run_single_experiment(
                config=cfg,
                ffn_mode='VFE_dynamic',
                device=device,
                checkpoint_dir=seed_dir,
                enable_publication_metrics=False,
                quiet=True,
                skip_test_eval=True,
                skip_post_training_viz=True,
                preloaded_data=shared_data,
            )
            ppl = result.get('final_ppl', float('inf')) if result else float('inf')
            ppls.append(ppl)
            print(f"    seed={seed}: val_ppl={ppl:.2f}")
        except Exception as e:
            print(f"    seed={seed}: FAILED — {e}")
            ppls.append(float('inf'))
        finally:
            _cleanup()

    mean_ppl = float(np.mean(ppls))
    std_ppl = float(np.std(ppls))
    return mean_ppl, std_ppl


# ============================================================================
# MAIN: TWO-PHASE HPO
# ============================================================================

def main() -> None:
    """Run two-phase Optuna HPO study."""
    hpo = HPO_CONFIG
    output_dir = Path(hpo['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(hpo['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # SQLite storage for persistence and dashboard
    storage = f"sqlite:///{output_dir / 'study.db'}"

    # ------------------------------------------------------------------
    # Load data ONCE (reused across all trials)
    # ------------------------------------------------------------------
    print("Loading dataset (one-time cost)...")
    from transformer.training.experiment_runner import _create_dataloaders
    from transformer.data import datasets as _ds_mod
    _prev_quiet = _ds_mod.QUIET
    _ds_mod.QUIET = True
    try:
        shared_data = _create_dataloaders(BASE_CONFIG)
    finally:
        _ds_mod.QUIET = _prev_quiet
    print(f"  Dataset loaded: vocab_size={shared_data[3]}")

    # ------------------------------------------------------------------
    # PHASE 1: Coarse scan
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"PHASE 1: Coarse Scan ({hpo['n_trials_phase1']} trials, "
          f"{hpo['max_steps_phase1']} steps each)")
    print(f"{'='*70}")

    # QMC (quasi-Monte Carlo / LHS) for initial space-filling, then TPE
    sampler = TPESampler(
        n_startup_trials=hpo['n_startup_trials'],
        seed=42,
    )

    study = optuna.create_study(
        study_name=hpo['study_name'],
        storage=storage,
        direction='minimize',
        sampler=sampler,
        load_if_exists=True,
    )

    # Only run remaining trials (supports resume)
    n_existing = len(study.trials)
    n_remaining = max(0, hpo['n_trials_phase1'] - n_existing)

    if n_remaining > 0:
        print(f"  {n_existing} existing trials found, running {n_remaining} more")

        phase1_dir = output_dir / 'phase1'
        phase1_dir.mkdir(parents=True, exist_ok=True)

        study.optimize(
            lambda trial: objective(
                trial, shared_data, hpo['max_steps_phase1'], device, phase1_dir,
            ),
            n_trials=n_remaining,
            gc_after_trial=True,
        )
    else:
        print(f"  Phase 1 complete ({n_existing} trials already done)")

    # Phase 1 summary
    completed_trials = [t for t in study.trials
                        if t.state == optuna.trial.TrialState.COMPLETE]
    failed_trials = [t for t in study.trials
                     if t.state != optuna.trial.TrialState.COMPLETE]

    print(f"\n--- Phase 1 Results ---")
    print(f"  Completed: {len(completed_trials)}, Failed/Pruned: {len(failed_trials)}")

    if not completed_trials:
        print("  WARNING: No trials completed successfully!")
        print("  Check trial logs in phase1/ for errors.")
        # Show failed trial info for debugging
        for t in failed_trials:
            state_name = t.state.name
            print(f"    Trial {t.number}: {state_name}")
            if t.state == optuna.trial.TrialState.FAIL:
                # Try to load the trial result JSON for error details
                trial_result = output_dir / 'phase1' / f'trial_{t.number:04d}' / 'trial_result.json'
                if trial_result.exists():
                    with open(trial_result) as f:
                        print(f"      {json.load(f)}")
        return

    print(f"  Best trial: #{study.best_trial.number}")
    print(f"  Best val_ppl: {study.best_value:.2f}")
    print(f"  Best params:")
    for k, v in sorted(study.best_params.items()):
        print(f"    {k}: {v}")

    # Save Phase 1 results
    phase1_results = []
    for t in completed_trials:
        phase1_results.append({
            'trial': t.number,
            'val_ppl': t.value,
            **t.params,
        })

    import pandas as pd
    df1 = pd.DataFrame(phase1_results).sort_values('val_ppl')
    df1.to_csv(output_dir / 'phase1_results.csv', index=False)
    print(f"\n  Phase 1 results saved to {output_dir / 'phase1_results.csv'}")

    # ------------------------------------------------------------------
    # PHASE 2: Multi-seed refinement of top configs
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"PHASE 2: Multi-Seed Refinement (top {hpo['n_trials_phase2']} configs, "
          f"{len(hpo['seeds_phase2'])} seeds x {hpo['max_steps_phase2']} steps)")
    print(f"{'='*70}")

    # Get top N configs from Phase 1
    top_trials = sorted(
        [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE],
        key=lambda t: t.value,
    )[:hpo['n_trials_phase2']]

    phase2_dir = output_dir / 'phase2'
    phase2_dir.mkdir(parents=True, exist_ok=True)

    phase2_results = []
    for rank, trial in enumerate(top_trials):
        config_dir = phase2_dir / f'rank_{rank:02d}_trial_{trial.number}'

        # Check for resume
        result_file = config_dir / 'multi_seed_result.json'
        if result_file.exists():
            with open(result_file) as f:
                cached = json.load(f)
            print(f"\n  Rank {rank}: trial #{trial.number} [CACHED] "
                  f"mean_ppl={cached['mean_ppl']:.2f} +/- {cached['std_ppl']:.2f}")
            phase2_results.append(cached)
            continue

        print(f"\n  Rank {rank}: trial #{trial.number} "
              f"(Phase 1 ppl={trial.value:.2f})")

        mean_ppl, std_ppl = run_multi_seed(
            params=trial.params,
            shared_data=shared_data,
            max_steps=hpo['max_steps_phase2'],
            device=device,
            output_dir=config_dir,
            seeds=hpo['seeds_phase2'],
        )

        result = {
            'rank': rank,
            'phase1_trial': trial.number,
            'phase1_ppl': trial.value,
            'mean_ppl': mean_ppl,
            'std_ppl': std_ppl,
            'seeds': hpo['seeds_phase2'],
            'max_steps': hpo['max_steps_phase2'],
            'params': trial.params,
        }
        phase2_results.append(result)

        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)

        print(f"    -> mean_ppl={mean_ppl:.2f} +/- {std_ppl:.2f}")

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"FINAL RESULTS")
    print(f"{'='*70}")

    phase2_results.sort(key=lambda r: r['mean_ppl'])
    df2 = pd.DataFrame(phase2_results)
    df2.to_csv(output_dir / 'phase2_results.csv', index=False)

    for i, r in enumerate(phase2_results[:5]):
        print(f"\n  #{i+1}: mean_ppl={r['mean_ppl']:.2f} +/- {r['std_ppl']:.2f}")
        if isinstance(r['params'], dict):
            for k, v in sorted(r['params'].items()):
                val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
                print(f"       {k}: {val_str}")

    best = phase2_results[0]
    print(f"\n  BEST CONFIG: mean_ppl={best['mean_ppl']:.2f} +/- {best['std_ppl']:.2f}")
    print(f"  Results saved to: {output_dir}")
    print(f"  Dashboard: optuna-dashboard sqlite:///{output_dir / 'study.db'}")

    # Save best config as a standalone dict
    best_config = copy.deepcopy(BASE_CONFIG)
    if isinstance(best['params'], dict):
        best_config.update(best['params'])
    with open(output_dir / 'best_config.json', 'w') as f:
        json.dump(best_config, f, indent=2, default=str)
    print(f"  Best config saved to: {output_dir / 'best_config.json'}")


if __name__ == '__main__':
    main()
