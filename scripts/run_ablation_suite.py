# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 22:17:49 2026

@author: chris and christine
"""

#!/usr/bin/env python3
"""
Systematic Ablation Suite for Gauge VFE Transformer
=====================================================

Sweeps hyperparameters one-at-a-time against a baseline (Exp 74 config),
then runs interaction sweeps for significant factors.

Usage:
    # Run all ablations sequentially
    python scripts/run_ablation_suite.py --device cuda --dataset wikitext-103

    # Run a specific sweep
    python scripts/run_ablation_suite.py --sweep alpha --device cuda

    # Run with reduced steps (quick validation)
    python scripts/run_ablation_suite.py --sweep alpha --max_steps 5000

    # Analyze results from a completed sweep
    python scripts/run_ablation_suite.py --analyze ablation_results/

    # List available sweeps
    python scripts/run_ablation_suite.py --list
"""

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformer.training.experiment_runner import run_single_experiment


# =============================================================================
# BASELINE CONFIG (Exp 74: best known, 74.86 test PPL)
# =============================================================================

BASELINE_CONFIG = {
    # === Architecture ===
    'vocab_size':            50257,
    'embed_dim':             20,
    'max_seq_len':           32,
    
    'batch_size':            1024, 
    'max_steps':             1875,
    
    'n_layers':              1,
    'ffn_n_iterations':      1,
    
    'gauge_dim':                          10,
    'irrep_spec':            [('fund', 2, 10)],

    'use_prior_bank':           False,
    'learnable_pb_temperature': False,
    'mask_self_attention':      False,  # Prevent attention collapse?
  
    # === M-step: implicit differentiation ===
    'implicit_em':           False,
    'amortized_inference':   True,
    'use_obs_in_vfe':        False,  #cheats when true
       
    'skip_attention':        True,   #skips ad hoc attention sublayer
    'closed_form_e_step':    False,  #closed form...ignores non-linear softmax gradient
    
    
    # === M-step: Optimizer ===  
    'optimizer_type':        'riemannian_adam',# or 'natural_gradient' or 'adamw' or 'riemannian_adam'
    'fisher_ema_decay':      0.95,            # for natural_gradient
    'fisher_damping':        1e-2,             # for natural_gradient

    'aux_layer_loss':        True,
    'aux_loss_weight':       0.3,
    'sigma_residual':        True,
    
    'use_layernorm':         True,
    'use_residual':          True,
    'use_output_projection': True,
    'multihead_vfe':         True,
    
    'evolve_sigma':          True,
    'evolve_phi':            True,
    'evolve_phi_e_step':     True,

    # === E-step dynamics ===
    'E_learnable_lr':        True,  # E-step

    'E_alpha':               1,       # Prior coupling inside VFE E-step
    'E_lambda_belief':       1,       # Belief alignment inside VFE E-step
    'E_lambda_softmax':      0,


    'E_learnable_alpha':     True,   # when true Adaptive α_i = c0/(b0 + KL) per dimension

    'E_mu_q_lr':             0.1,    # whitened steps ~0.1, well within trust=2.0
    'E_sigma_q_lr':          0.05,   # conservative sigma movement
    'E_phi_lr':              0.05,   # keep as-is, already reasonable

    # === Gauge group: GL(K) with multi-head block-diagonal structure ===
    'gauge_group':      'GLK',
    'gauge_mode':       'learned',
    'gauge_param':      'phi',


    'diagonal_covariance':      True,
    'exact_diagonal_transport': False,  # exact diagonal transport - more expensive
                                        # If True, force Σ = σ²I (scalar variance × identity)
    'isotropic_covariance':     False,
    'enforce_orthogonal':       False,    
    'learnable_reflection':     False,# Per-token s_i ∈ {±1}^K → O(K)  - enforce orthogonal=true with glk
                                        # Set gauge-mode=constant and the above 3 = true for transf limit
 
    # === VFE loss weights (M-step objective) ===
    # E-step: prior + alignment (no observations with n_iterations=1).
    # CE enters through M-step via IFT (s_k ≈ 0.5 from fixed E_alpha=1).
    # M_alpha=0: KL(q*||p) homogenizes (q* is smoothed, not data-grounded).
    # M_beta=0: alignment term is vacuum-seeking. E-step handles it internally.
    
    'M_alpha':             0.00,
    'M_beta':              0.0,
    'mass_phi':            0.05,            # Gauge prior: (mass_φ/2)||φ||²
    'lambda_hyper':        0.0,            # KL(s||h) with fixed Σ_h set if if using embed-weight-decay
    'lambda_gamma':        0.0,
    'kappa_gamma':         1.0,

    'embed_weight_decay':  0.05,   #acts like lambda_hyper N(o, 1/2sig) set zero when using lambda_hyper/alpha_phi
    'non_embed_weight_decay': 0.01,   #acts on non-vfe params
    
    # === Phi gradient geometry ===
    'phi_natural_gradient':       'killing',
    'use_killing_form':           True,
    'killing_form_sym_dampening': 0.1,

    # === Position encoding ===
    'use_rope':           True,
    'rope_base':          1000, 
    'pos_encoding_mode': 'none',

    # === Embedding init ===
    'mu_init_std':     1.0,
    'phi_scale':       1.0,
    
    'kappa_beta':      1,
    
    'mu_normalize':    False,
    'mu_max_norm':     None,

    'sigma_aggregation': 'mixture',
    # === M-step learning rates (AdamW parameter groups) ===
    # These update nn.Parameter objects via backprop. The E-step (inner VFE
    # loop) uses e_step_mu_lr / e_step_sigma_lr / e_step_phi_lr above.
    # mu_embed and log_sigma_diag have dual roles: they initialize E-step
    # beliefs (q₀) AND serve as prior parameters (μ_p, σ_p), so these rates
    # indirectly affect E-step initialization speed.
    'mu_lr':        0.05,   # Prior mean embeddings (μ_p)
    'sigma_lr':     0.005, # Prior covariance embeddings (log σ_p)
    'phi_lr':       0.0075, # Gauge frame embeddings (φ)
    'ffn_lr':       0.05,   # FFN params (raw_c0, raw_b0, raw_lr)
    'attention_lr': 0.005,  # Attention params (W_O, constant_omega)
    'output_lr':    0.05,   # Output projection (vocab logits)
    
    # === Logging ===
    'log_interval':               100,
    'eval_interval':              1500,
    'checkpoint_interval':        25000,
    'semantic_analysis_interval': 10000,

    'use_deq':           False,
    'deq_include_phi':   True,    # Corrects M-step phi gradient
    'deq_neumann_terms': 0,

    # =================================================================
    # NON-FLAT GAUGE TRANSPORT (holonomy)
    # =================================================================
    # When enabled, transport acquires an edge-local connection δ_ij:
    #   phi path:  Ω_ij = exp(φ_i·G) · exp(α·δ_ij·G) · exp(-φ_j·G)
    #   omega path: Ω_ij = Ω_i · exp(α·δ_ij·G) · Ω_j⁻¹
    # δ_ij is zero-initialized so the model starts flat and learns
    # curvature only where the data warrants it.
    # Holonomy H_ijk = Ω_ij·Ω_jk·Ω_ki ≠ I when δ ≠ 0.
    
    'non_flat_transport':    False,        # Enable edge-dependent connection δ_ij
    'cocycle_relaxation':    0.5,          # Scale for δ_ij: 0=flat, 1=fully non-flat    
    'connection_type':       'bilinear',  # 'bilinear' (δ_ij^a = μ_i^T W^a μ_j) | 'mlp'   
    'connection_hidden_dim': 64,   # Hidden dim for MLP connection (ignored for bilinear)   
    'connection_init_scale': 0.01,   # W init scale (0=flat saddle point, 0.01 recommended)    
    'holonomy_penalty':      0.0,  # λ_H · E[‖C_ijk - I‖²_F] regularizer (0 = off)

    # Option A: couple just 0↔1, head 2 stays independent
    # 'cross_couplings': [(0, 1), (1, 0)],
    # → super-blocks: [20, 10]  (heads 0,1 merged into GL(20), head 2 alone)
    # === Layer/iteration diagnostics ===
    'track_layer_diagnostics':     True,
    'track_iteration_diagnostics': True,
    'diagnostics_interval':        25,
    
    'debug_vfe_grads':             False,
    'verbose_diagnostics':         False,
    
    'gauge_fixed_priors':          False,
    'tie_embeddings':              False,
    'ffn_mode':                    'VFE_dynamic',
    
    # === Regularization ===
    'sigma_max':     10.0,
    'grad_clip':     1.0,
    'hidden_dim':    508,
    'warmup_steps':  100,
    'num_workers':   10,


    'dataset': 'wikitext-103', #'wiki-2' for quick sweeps
}


# =============================================================================
# SWEEP DEFINITIONS
# =============================================================================
# Each sweep is: { param_name: [values] } or { (p1, p2, ...): [(v1, v2, ...), ...] }
# For categorical ablations, we override multiple params at once.

SWEEPS = {
    # --- Tier 1: Free energy weights ---
    'alpha': {
        'description': 'Self-consistency KL(q||p) weight in training loss',
        'param': 'alpha',
        'values': [0, 0.001, 0.005, 0.01, 0.05, 0.1],
        'baseline_value': 0.00,
    },

    'ffn_alpha': {
        'description': 'Prior coupling weight inside VFE E-step iterations',
        'param': 'ffn_alpha',
        'values': [0, 0.01, 0.1, 0.5, 1, 2.0],
        'baseline_value': 1.0,
    },

    'ffn_lambda_belief': {
        'description': 'Belief alignment weight inside VFE E-step iterations',
        'param': 'ffn_lambda_belief',
        'values': [0, 0.5, 1.0, 2.0, 5],
        'baseline_value': 1.0,
    },

    'ffn_lambda_softmax': {
        'description': 'Softmax coupling weight (GELU-like ∂β/∂θ·KL term) inside VFE E-step',
        'param': 'ffn_lambda_softmax',
        'values': [0, 0.5, 1.0, 2.0, 5.0],
        'baseline_value': 0,
    },

    'beta': {
        'description': 'Belief alignment weight in outer training loss (lambda_beta)',
        'param': 'beta',
        'values': [0.0, 0.001, 0.005, 0.01, 0.1],
        'baseline_value': 0.0,
    },

    'lambda_hyper': {
        'description': 'Hyper-prior coupling KL(s||h) weight in training loss',
        'param': 'lambda_hyper',
        'values': [0.0, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0],
        'baseline_value': 0.0,
    },

    # --- Tier 1b: Temperature / kappa parameters ---
    'kappa_beta': {
        'description': 'Attention/VFE temperature τ: β_ij = softmax(-KL/κ√K). Higher = softer attention',
        'param': 'kappa_beta',
        'values': [0.5, 1, 1.5, 2, 2.25, 2.5],
        'baseline_value': 1.0,
    },

   

    'prior_bank_tau': {
        'description': 'PriorBank decode temperature: logits = -KL(q||π_v)/τ. Lower = sharper decode',
        'param': 'prior_bank_tau',
        'values': [0.1, 0.25, 0.5, 1.0, 2.0],
        'baseline_value': 1.0,
    },

    'rope_base': {
        'description': 'RoPE frequency base: θ_n = 1/base^(2n/K). Lower = faster position decay',
        'param': 'rope_base',
        'values': [100, 500, 1000, 5000, 10000, 50000],
        'baseline_value': 1000,
    },

    # --- Tier 2: Architecture ---
    'K': {
        'description': 'Embedding dimension (belief dimensionality)',
        'param': None,  # Multi-param override
        'configs': [
            {'embed_dim': 10, 'irrep_spec': [('fund', 1, 10)], 'label': 'K=10'},
            {'embed_dim': 20, 'irrep_spec': [('fund', 2, 10)], 'label': 'K=20'},
            {'embed_dim': 40, 'irrep_spec': [('fund', 4, 10)], 'label': 'K=40'},
            {'embed_dim': 60, 'irrep_spec': [('fund', 6, 10)], 'label': 'K=60'},
            {'embed_dim': 80, 'irrep_spec': [('fund', 8, 10)], 'label': 'K=80'},
            {'embed_dim': 100,'irrep_spec': [('fund', 10, 10)], 'label': 'K=100'},
        ],
        'baseline_value': 'K=20',
    },

    'n_vfe_iterations': {
        'description': 'Number of E-step fixed-point iterations per forward pass',
        'param': 'ffn_n_iterations',
        'values': [1, 3, 5, 7, 10],
        'baseline_value': 1,
    },

    'n_layers': {
        'description': 'Number of transformer layers',
        'param': 'n_layers',
        'values': [1, 2, 3],
        'baseline_value': 1,
    },

    # --- Tier 3: Gauge structure ---
    'gauge_mode': {
        'description': 'Gauge transport mode: learned φ vs trivial (Ω=I) vs constant',
        'param': None,
        'configs': [
            {'gauge_mode': 'learned', 'label': 'learned'},
            {'gauge_mode': 'trivial', 'label': 'trivial'},
            {'gauge_mode': 'constant', 'label': 'constant'},
        ],
        'baseline_value': 'learned',
    },

    'covariance': {
        'description': 'Covariance structure: diagonal vs isotropic',
        'param': None,
        'configs': [
            {'isotropic_covariance': False, 'diagonal_covariance': True, 'label': 'diagonal'},
            {'isotropic_covariance': True, 'diagonal_covariance': True, 'label': 'isotropic'},
        ],
        'baseline_value': 'diagonal',
    },

    'rope': {
        'description': 'Rotary position encoding on/off',
        'param': 'use_rope',
        'values': [True, False],
        'baseline_value': True,
    },

    # --- Tier 4: Phi preconditioning ---
    'phi_preconditioner': {
        'description': 'Gauge frame gradient preconditioning mode',
        'param': None,
        'configs': [
            {'phi_natural_gradient': 'clip', 'use_killing_form': False, 'label': 'clip'},
            {'phi_natural_gradient': 'killing', 'use_killing_form': True, 'label': 'killing'},
            {'phi_natural_gradient': 'cartan', 'use_killing_form': False, 'label': 'cartan'},
        ],
        'baseline_value': 'killing',
    },

    # --- Tier 5: Gauge dim (GL(K) K parameter) ---
    'gauge_dim': {
        'description': 'GL(K) gauge group dimension (d_head)',
        'param': None,
        'configs': [
            {'gauge_dim': 2, 'irrep_spec': [('fund', 24, 2)], 'embed_dim': 48, 'label': 'GL(2)_h24'},
            {'gauge_dim': 4, 'irrep_spec': [('fund', 12, 4)], 'embed_dim': 48, 'label': 'GL(4)_h12'},
            {'gauge_dim': 6, 'irrep_spec': [('fund', 8, 6)], 'embed_dim': 48, 'label': 'GL(6)_h8'},
            {'gauge_dim': 8, 'irrep_spec': [('fund', 6, 8)], 'embed_dim': 48, 'label': 'GL(8)_h6'},
            {'gauge_dim': 12, 'irrep_spec': [('fund', 4, 12)], 'embed_dim': 48, 'label': 'GL(12)_h4'},
            {'gauge_dim': 16, 'irrep_spec': [('fund', 3, 16)], 'embed_dim': 48, 'label': 'GL(16)_h3'},
        ],
        'baseline_value': 'GL(10)_h2',
    },

    # --- Tier 6: Alpha_phi (gauge prior) ---
    'alpha_phi': {
        'description': 'Gauge frame L2 prior weight (α_φ/2)||φ||²',
        'param': 'alpha_phi',
        'values': [0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
        'baseline_value': 0.1,
    },

    # --- Tier 7: Learning rates ---
    'mu_lr': {
        'description': 'Belief mean (μ) learning rate',
        'param': 'mu_lr',
        'values': [0.0025, 0.01, 0.05, 0.075, 0.1],
        'baseline_value': 0.05,
    },

    'sigma_lr': {
        'description': 'Belief precision (σ) learning rate',
        'param': 'sigma_lr',
        'values': [0.001, 0.005, 0.0075, 0.0125, 0.05, 0.1],
        'baseline_value': 0.005,
    },

    'phi_lr': {
        'description': 'Gauge frame (φ) learning rate',
        'param': 'phi_lr',
        'values': [0.001, 0.005, 0.0125, 0.02, 0.05, 0.1],
        'baseline_value': 0.005,
    },

    'ffn_lr': {
        'description': 'FFN / VFE module learning rate',
        'param': 'ffn_lr',
        'values': [0.0025, 0.01, 0.05, 0.077, 0.15],
        'baseline_value': 0.05,
    },

    'attention_lr': {
        'description': 'Attention module learning rate',
        'param': 'attention_lr',
        'values': [0.005, 0.006, 0.007, 0.008, 0.009],
        'baseline_value': 0.005,
    },

    'output_lr': {
        'description': 'Output projection learning rate',
        'param': 'output_lr',
        'values': [0.005, 0.01, 0.02, 0.05, 0.1, 0.2],
        'baseline_value': 0.05,
    },

    # --- Tier 8: E-step learning rates ---
    'e_step_mu_lr': {
        'description': 'E-step μ natural gradient step size',
        'param': 'e_step_mu_lr',
        'values': [0.01, 0.05, 0.1, 0.5, 2.0],
        'baseline_value': 0.1,
    },

    'e_step_sigma_lr': {
        'description': 'E-step σ trust region scale',
        'param': 'e_step_sigma_lr',
        'values': [0.005, 0.01, 0.05, 0.1, 0.2, 0.5],
        'baseline_value': 0.05,
    },

    'e_step_phi_lr': {
        'description': 'E-step φ gauge frame step size',
        'param': 'e_step_phi_lr',
        'values': [0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
        'baseline_value': 0.05,
    },

    # --- Tier 9: Regularization ---
    'embed_weight_decay': {
        'description': 'Weight decay on embedding parameters (hyper-prior precision)',
        'param': 'embed_weight_decay',
        'values': [0.04, 0.05, 0.075, 0.1, 0.25],
        'baseline_value': 0.01,
    },

    'weight_decay': {
        'description': 'Weight decay on non-embedding parameters',
        'param': 'weight_decay',
        'values': [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2],
        'baseline_value': 0.01,
    },
}

# Sweep execution order (cheapest → most expensive)
SWEEP_ORDER = [
    
    #'gauge_dim', 
    #'K', 
    
      
    #'M_alpha',Done
    #'M_beta', Done
    #'mass_phi', Done

    'lambda_hyper',

    #'embed_weight_decay',
   # 'non_embed_weight_decay',

    'E_lambda_belief',
    'E_alpha',

    #'M_mu_p_lr',
    #'M_sigma_p_lr',
    #'M_phi_lr',

    #'M_vfe_hyperparam_lr',


    #'E_mu_q_lr',
    #'E_sigma_q_lr',
    #'E_phi_lr',

    # 'kappa_beta',Done

    #'prior_bank_tau',

    #'rope_base', Done

    #'M_attention_lr',
    #'M_output_lr',
    
    #'covariance', 
    
    #'gauge_mode', 
    #'phi_preconditioner',
    
    
    #'rope',

    
   

    #'n_layers',
    #'n_vfe_iterations', 

   
    
]



# =============================================================================
# RUNNER
# =============================================================================

def make_run_configs(sweep_name: str, base_config: dict) -> list:
    """Generate (label, config) pairs for a sweep."""
    sweep = SWEEPS[sweep_name]
    runs = []

    if 'configs' in sweep:
        # Multi-param categorical sweep
        for entry in sweep['configs']:
            cfg = copy.deepcopy(base_config)
            label = entry.pop('label')
            cfg.update(entry)
            # Restore label in sweep def (we popped it)
            entry['label'] = label
            runs.append((label, cfg))
    else:
        # Single-param sweep
        param = sweep['param']
        for val in sweep['values']:
            cfg = copy.deepcopy(base_config)
            cfg[param] = val
            label = f"{param}={val}"
            runs.append((label, cfg))

    return runs


def run_sweep(
    sweep_name: str,
    base_config: dict,
    device: torch.device,
    output_dir: Path,
    seed: int = 42,
    max_steps_override: int = None,
    resume: bool = False,
) -> pd.DataFrame:
    """Run all configs in a sweep, return results DataFrame."""
    sweep = SWEEPS[sweep_name]
    sweep_dir = output_dir / sweep_name
    sweep_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"SWEEP: {sweep_name}")
    print(f"  {sweep['description']}")
    print(f"  Baseline: {sweep['baseline_value']}")
    print(f"  Output: {sweep_dir}")
    if resume:
        print(f"  Resume: ON (skipping completed runs)")
    print(f"{'='*70}\n")

    runs = make_run_configs(sweep_name, base_config)
    results = []

    # In resume mode, load any previously completed results
    if resume:
        for label, _cfg in runs:
            run_dir = sweep_dir / label.replace('=', '_').replace(' ', '_')
            result_file = run_dir / 'ablation_result.json'
            if result_file.exists():
                with open(result_file) as f:
                    results.append(json.load(f))

    for i, (label, cfg) in enumerate(runs):
        run_dir = sweep_dir / label.replace('=', '_').replace(' ', '_')

        # Skip already-completed runs in resume mode
        if resume and (run_dir / 'ablation_result.json').exists():
            print(f"\n--- Run {i+1}/{len(runs)}: {label} [CACHED] ---")
            continue

        print(f"\n--- Run {i+1}/{len(runs)}: {label} ---")

        # Override max_steps if requested
        if max_steps_override is not None:
            cfg['max_steps'] = max_steps_override

        # Set seed for reproducibility
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        try:
            t0 = time.time()
            result = run_single_experiment(
                config=cfg,
                ffn_mode=cfg.get('ffn_mode', 'VFE_dynamic'),
                device=device,
                checkpoint_dir=run_dir,
                enable_publication_metrics=False,
            )
            elapsed = time.time() - t0

            if result is not None:
                result['sweep'] = sweep_name
                result['label'] = label
                result['elapsed_sec'] = elapsed
                result['seed'] = seed
                results.append(result)

                # Save individual result
                with open(run_dir / 'ablation_result.json', 'w') as f:
                    json.dump(result, f, indent=2)

                # Save per-run CSV (individual + append to sweep CSV)
                run_df = pd.DataFrame([result])
                run_df.to_csv(run_dir / 'run_result.csv', index=False)

                sweep_csv = sweep_dir / 'sweep_results.csv'
                run_df.to_csv(
                    sweep_csv,
                    mode='a',
                    header=not sweep_csv.exists(),
                    index=False,
                )

                print(f"  -> Val PPL: {result['final_ppl']:.2f}, "
                      f"Test PPL: {result.get('test_ppl', 'N/A')}, "
                      f"Time: {elapsed:.0f}s")
            else:
                print(f"  -> FAILED (returned None)")

        except Exception as e:
            print(f"  -> ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'sweep': sweep_name,
                'label': label,
                'error': str(e),
                'final_ppl': float('inf'),
            })

    # Save sweep summary
    df = pd.DataFrame(results)
    df.to_csv(sweep_dir / 'sweep_results.csv', index=False)

    # Save sweep metadata
    meta = {
        'sweep_name': sweep_name,
        'description': sweep['description'],
        'baseline_value': str(sweep['baseline_value']),
        'n_runs': len(runs),
        'seed': seed,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(sweep_dir / 'sweep_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*70}")
    print(f"SWEEP COMPLETE: {sweep_name}")
    print(f"{'='*70}")
    if not df.empty and 'final_ppl' in df.columns:
        valid = df[df['final_ppl'] < float('inf')]
        if not valid.empty:
            best = valid.loc[valid['final_ppl'].idxmin()]
            print(f"  Best: {best.get('label', '?')} -> Val PPL {best['final_ppl']:.2f}")
            if 'test_ppl' in best and pd.notna(best.get('test_ppl')):
                print(f"                          Test PPL {best['test_ppl']:.2f}")
    print()

    return df


# =============================================================================
# ANALYSIS
# =============================================================================

def analyze_sweep(sweep_dir: Path) -> dict:
    """Analyze results from a completed sweep directory."""
    csv_path = sweep_dir / 'sweep_results.csv'
    if not csv_path.exists():
        print(f"No results found in {sweep_dir}")
        return {}

    df = pd.read_csv(csv_path)
    meta_path = sweep_dir / 'sweep_meta.json'
    meta = json.load(open(meta_path)) if meta_path.exists() else {}

    print(f"\n{'='*70}")
    print(f"ANALYSIS: {meta.get('sweep_name', sweep_dir.name)}")
    print(f"  {meta.get('description', '')}")
    print(f"{'='*70}\n")

    # Sort by val PPL
    valid = df[df['final_ppl'] < float('inf')].copy()
    valid = valid.sort_values('final_ppl')

    print(f"{'Label':<30} {'Val PPL':>10} {'Test PPL':>10} {'Params':>12} {'Time':>8}")
    print('-' * 75)
    for _, row in valid.iterrows():
        test_ppl = f"{row['test_ppl']:.2f}" if pd.notna(row.get('test_ppl')) else 'N/A'
        params = f"{row.get('total_params', 0):,}" if pd.notna(row.get('total_params')) else 'N/A'
        elapsed = f"{row.get('elapsed_sec', 0):.0f}s" if pd.notna(row.get('elapsed_sec')) else 'N/A'
        print(f"{row['label']:<30} {row['final_ppl']:>10.2f} {test_ppl:>10} {params:>12} {elapsed:>8}")

    # Compute relative improvements
    if len(valid) > 1:
        baseline_ppl = valid.iloc[0]['final_ppl']  # Best as reference
        print(f"\nRelative to best ({valid.iloc[0]['label']}):")
        for _, row in valid.iterrows():
            delta = ((row['final_ppl'] - baseline_ppl) / baseline_ppl) * 100
            print(f"  {row['label']:<30} {delta:+.1f}%")

    return {'df': df, 'meta': meta}


def analyze_all(output_dir: Path):
    """Analyze all completed sweeps and produce a summary."""
    print(f"\n{'='*70}")
    print(f"ABLATION SUITE SUMMARY")
    print(f"  Directory: {output_dir}")
    print(f"{'='*70}\n")

    all_results = []
    for sweep_dir in sorted(output_dir.iterdir()):
        if sweep_dir.is_dir() and (sweep_dir / 'sweep_results.csv').exists():
            result = analyze_sweep(sweep_dir)
            if result:
                all_results.append(result)

    if not all_results:
        print("No completed sweeps found.")
        return

    # Cross-sweep summary: best from each
    print(f"\n{'='*70}")
    print(f"CROSS-SWEEP SUMMARY (best per sweep)")
    print(f"{'='*70}\n")
    print(f"{'Sweep':<25} {'Best Config':<30} {'Val PPL':>10} {'Test PPL':>10}")
    print('-' * 80)

    for r in all_results:
        df = r['df']
        valid = df[df['final_ppl'] < float('inf')]
        if valid.empty:
            continue
        best = valid.loc[valid['final_ppl'].idxmin()]
        sweep = r['meta'].get('sweep_name', '?')
        test_ppl = f"{best['test_ppl']:.2f}" if pd.notna(best.get('test_ppl')) else 'N/A'
        print(f"{sweep:<25} {best['label']:<30} {best['final_ppl']:>10.2f} {test_ppl:>10}")


def generate_plots(output_dir: Path):
    """Generate publication-quality ablation plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib/seaborn not available, skipping plots")
        return

    sns.set_theme(style='whitegrid', font_scale=1.2)

    sweep_dirs = [d for d in sorted(output_dir.iterdir())
                  if d.is_dir() and (d / 'sweep_results.csv').exists()]

    if not sweep_dirs:
        print("No sweeps to plot.")
        return

    fig_dir = output_dir / 'figures'
    fig_dir.mkdir(exist_ok=True)

    for sweep_dir in sweep_dirs:
        df = pd.read_csv(sweep_dir / 'sweep_results.csv')
        meta_path = sweep_dir / 'sweep_meta.json'
        meta = json.load(open(meta_path)) if meta_path.exists() else {}
        sweep_name = meta.get('sweep_name', sweep_dir.name)

        valid = df[df['final_ppl'] < float('inf')].copy()
        if valid.empty:
            continue

        # Try to extract numeric parameter value from label
        numeric_values = []
        for label in valid['label']:
            try:
                # Handle "param=value" format
                val = label.split('=')[-1]
                numeric_values.append(float(val))
            except (ValueError, IndexError):
                numeric_values.append(None)

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))

        if all(v is not None for v in numeric_values):
            # Numeric sweep: line plot
            valid = valid.copy()
            valid['param_value'] = numeric_values
            valid = valid.sort_values('param_value')
            ax.plot(valid['param_value'], valid['final_ppl'], 'o-', linewidth=2, markersize=8)
            ax.set_xlabel(sweep_name)

            # Mark baseline
            baseline_val = meta.get('baseline_value')
            if baseline_val:
                try:
                    bv = float(baseline_val)
                    baseline_row = valid.loc[(valid['param_value'] - bv).abs().idxmin()]
                    ax.axvline(bv, color='red', linestyle='--', alpha=0.5, label=f'baseline={bv}')
                    ax.legend()
                except (ValueError, TypeError):
                    pass
        else:
            # Categorical sweep: bar plot
            valid = valid.sort_values('final_ppl')
            colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(valid))]
            ax.barh(range(len(valid)), valid['final_ppl'], color=colors)
            ax.set_yticks(range(len(valid)))
            ax.set_yticklabels(valid['label'])
            ax.invert_yaxis()

        ax.set_ylabel('Validation Perplexity')
        ax.set_title(f"Ablation: {meta.get('description', sweep_name)}")
        fig.tight_layout()
        fig.savefig(fig_dir / f'{sweep_name}.png', dpi=150)
        fig.savefig(fig_dir / f'{sweep_name}.pdf')
        plt.close(fig)
        print(f"  Saved: {fig_dir / sweep_name}.{{png,pdf}}")

    # Combined summary figure
    all_bests = []
    for sweep_dir in sweep_dirs:
        df = pd.read_csv(sweep_dir / 'sweep_results.csv')
        meta = json.load(open(sweep_dir / 'sweep_meta.json')) if (sweep_dir / 'sweep_meta.json').exists() else {}
        valid = df[df['final_ppl'] < float('inf')]
        if valid.empty:
            continue
        best = valid.loc[valid['final_ppl'].idxmin()]
        worst = valid.loc[valid['final_ppl'].idxmax()]
        all_bests.append({
            'sweep': meta.get('sweep_name', sweep_dir.name),
            'best_ppl': best['final_ppl'],
            'worst_ppl': worst['final_ppl'],
            'range': worst['final_ppl'] - best['final_ppl'],
            'best_label': best['label'],
        })

    if all_bests:
        bdf = pd.DataFrame(all_bests).sort_values('range', ascending=False)
        fig, ax = plt.subplots(1, 1, figsize=(10, max(4, len(bdf) * 0.5)))
        y_pos = range(len(bdf))
        ax.barh(y_pos, bdf['range'], color='#e74c3c', alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{r['sweep']}\n(best: {r['best_label']})" for _, r in bdf.iterrows()])
        ax.set_xlabel('PPL Range (worst - best)')
        ax.set_title('Hyperparameter Sensitivity (PPL range per sweep)')
        ax.invert_yaxis()
        fig.tight_layout()
        fig.savefig(fig_dir / 'sensitivity_summary.png', dpi=150)
        fig.savefig(fig_dir / 'sensitivity_summary.pdf')
        plt.close(fig)
        print(f"  Saved: {fig_dir / 'sensitivity_summary'}.{{png,pdf}}")

    print(f"\nAll figures saved to: {fig_dir}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Systematic Ablation Suite for Gauge VFE Transformer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--sweep', type=str, default=None,
                        help='Run a specific sweep (e.g., "alpha", "K"). '
                             'If not set, runs all sweeps in order.')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--dataset', type=str, default='wikitext-103',
                        choices=['wikitext-2', 'wikitext-103'])
    parser.add_argument('--output_dir', type=str, default='ablation_results')
    parser.add_argument('--max_steps', type=int, default=None,
                        help='Override max_steps for all runs (e.g., 5000 for quick test)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--analyze', type=str, default=None, nargs='?', const='ablation_results',
                        help='Analyze results in directory (default: ablation_results/)')
    parser.add_argument('--plot', type=str, default=None, nargs='?', const='ablation_results',
                        help='Generate plots from results in directory')
    parser.add_argument('--resume', action='store_true',
                        help='Resume interrupted run: skip sweeps/runs that already '
                             'have ablation_result.json on disk')
    parser.add_argument('--list', action='store_true',
                        help='List all available sweeps')

    args = parser.parse_args()

    # List mode
    if args.list:
        print(f"\nAvailable sweeps ({len(SWEEPS)}):\n")
        print(f"{'Name':<25} {'# Configs':>10} {'Description'}")
        print('-' * 80)
        for name in SWEEP_ORDER:
            s = SWEEPS[name]
            n = len(s.get('configs', s.get('values', [])))
            print(f"{name:<25} {n:>10} {s['description']}")
        total = sum(len(SWEEPS[n].get('configs', SWEEPS[n].get('values', []))) for n in SWEEP_ORDER)
        print(f"\nTotal runs: {total}")
        return

    # Analyze mode
    if args.analyze is not None:
        analyze_all(Path(args.analyze))
        return

    # Plot mode
    if args.plot is not None:
        generate_plots(Path(args.plot))
        return

    # Training mode
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_config = copy.deepcopy(BASELINE_CONFIG)
    base_config['dataset'] = args.dataset

    # Determine which sweeps to run
    if args.sweep:
        if args.sweep not in SWEEPS:
            print(f"Unknown sweep: {args.sweep}")
            print(f"Available: {', '.join(SWEEP_ORDER)}")
            return
        sweep_names = [args.sweep]
    else:
        sweep_names = SWEEP_ORDER

    print(f"\nAblation Suite Configuration:")
    print(f"  Device:    {device}")
    print(f"  Dataset:   {args.dataset}")
    print(f"  Output:    {output_dir}")
    print(f"  Seed:      {args.seed}")
    print(f"  Sweeps:    {', '.join(sweep_names)}")
    if args.max_steps:
        print(f"  Max steps: {args.max_steps} (override)")
    if args.resume:
        print(f"  Resume:    ON (skipping completed runs)")
    print()

    # Run sweeps
    all_dfs = {}
    for sweep_name in sweep_names:
        df = run_sweep(
            sweep_name=sweep_name,
            base_config=base_config,
            device=device,
            output_dir=output_dir,
            seed=args.seed,
            max_steps_override=args.max_steps,
            resume=args.resume,
        )
        all_dfs[sweep_name] = df

        # Generate plots after each sweep so figures are never lost
        generate_plots(output_dir)

    # Final summary
    analyze_all(output_dir)


if __name__ == '__main__':
    main()
