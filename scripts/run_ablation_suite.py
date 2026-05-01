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
import gc
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

from transformer.training.experiment_runner import (
    run_single_experiment,
    _create_dataloaders,
)
from transformer.train_publication import EM_CONFIG


# =============================================================================
# BASELINE CONFIG (Exp 74: best known, 74.86 test PPL)
# Inherits all keys from EM_CONFIG; only differing keys are listed here.
# =============================================================================

BASELINE_CONFIG = {
   # === Architecture ===
   'vocab_size':                 50257,
   'embed_dim':                  20,
   'max_seq_len':                32,
   
   'batch_size':                 128, 
   'max_steps':                  10000,
    
   'stride':                     32,                                                                                                
   'random_offset_per_epoch':    True,
   'stride_base_seed':           6,
   
   'n_layers':                   1,
   'ffn_n_iterations':           3,
   
   'alpha_divergence':           0.5,
   #'grad_accumulation_steps': 1,
   #'gradient_checkpoint_vfe': False,
   
   'gauge_dim':                   10,
   'irrep_spec':       [('fund', 2, 10)],

   'use_prior_bank':             False,
   'gauge_fixed_priors':         False,    
   'hierarchical_priors':        True,
   
   'learnable_pb_temperature':   False,    #prior bank temperature
   'mask_self_attention':        False,  # Prevent attention collapse?
 

   'kappa_beta':                 1,
   'kappa_warmup_steps':         7500,  # freeze kappa for first n steps
   'learnable_head_kappa':       False, # If True, learn per-head κ_h via log_kappa_per_head

   'e_step_sigma_floor':         0.1,   # Floor on σ_p inside E-step (caps 1/σ_p at 1/floor)
   
   # === EM gradient-flow mode ===
   'em_mode':                    'straight_through',  # φ∈q: E-step optimizes μ,Σ,φ; all detached at EM boundary
                                              # - 'straight_through' (default) — mu_p, sigma_p attached, semi-gradient phi
                                              # - 'ift_phi' — same + full IFT phi gradient
                                              # - 'em_phi_q' — clean EM, phi in q, all detached at boundary
                                              # - 'em_phi_p' — clean EM, phi frozen in E-step
                                              # - 'implicit_ift' — experimental IFT-scaled M-step

   # === GL(K) determinant control (off by default; pick at most one) ===
   # GL(K) has an unbounded trace direction that L2-norm clamping does not
   # constrain — det(Ω_ij) = exp(tr(φ_i − φ_j)) blows up on outlier tokens.
   # Recommended for gauge_group='GLK' with phi_max_norm > ~3.
   'phi_project_slk':            False,   # Hard project φ → sl(K): det(Ω) ≡ 1 always
   'phi_trace_clamp':            0.75,    # Soft cap |tr(φ·G)| ≤ T (e.g., 0.35 → det ∈ [0.5, 2])


   'active_inference':           False,   #requires priorbank true
   
   'cache_decode_priors':        False,
   'skip_attention':             False,   #skips ad hoc attention sublayer
   
   # === M-step: Optimizer ===  
   'optimizer_type':             'riemannian_adam',# or 'natural_gradient' or 'adamw' or 'riemannian_adam'
   'phi_optimizer_metric':       'killing',
   'fisher_ema_decay':           0.90,            # for natural_gradient
   'fisher_damping':             1e-2,              # for natural_gradient


   'use_layernorm':              True,   #breaks gauge equivariance
   'use_residual':               True,  #set False if skip-attention=True
   'use_output_projection':      True,
   'evolve_sigma':               True,
   'evolve_phi':                 True,  #M-step phi evolution
   'evolve_phi_e_step':          True,
   'normalize_ce_by_dim':        True,
   'ce_label_smoothing':         0.0,    # Label smoothing on CE loss only; ce_loss_raw preserved for PPL

   'E_learnable_alpha':          True,   # Adaptive α_i = c0/(b0 + KL) per dimension
   'E_learnable_lr':             True,   # Learnable E-step LR
   
   'min_lr_ratio':               0.1,
   'lr_decay':                   'cosine',   #'linear', 'cosine', 'constant'
   
   'norm_type':                  'layernorm',  # 'layernorm' | 'rmsnorm' | 'mahalnorm' | 'none'
   'residual_type':              'additive',    # 'additive': mu_q = mu_q + mu_sub 
                                        # 'delta':    mu_q = mu_q + (mu_sub - mu_normalized),
   #'centered_mahalnorm'
   # === E-step Weights ===

   'E_alpha':                    1,      # E-step prior coupling weight
   'E_lambda_belief':            15,    # E-step belief alignment weight
   'E_lambda_softmax':           0,
      
   # === E-step Learning Rates ===
   
   'E_mu_q_lr':                  0.3,    # E-step μ step size (whitened, within trust=2.0)
   'E_sigma_q_lr':               0.015,   # E-step σ step size (conservative)    
   'E_phi_lr':                   0.05,   # E-step φ step size

   # === M-step Weights ===        
   
   'M_alpha':                    0.0,   # M-step KL(q||p) self-consistency
   'M_beta':                     0.0,    # M-step belief alignment
   'mass_phi':                   0.00,    # Gauge prior: (mass_φ/2)||φ||²
   'lambda_hyper':               0.0,    # KL(s||h) explicit loss (pulls tokens toward centroid)
   'lambda_gamma':               0,
   # === M-step Learning Rates (AdamW parameter groups) ===
   
   'M_mu_p_lr':                  0.07,   # M-step prior mean embeddings (μ_p) 0.05
   'M_sigma_p_lr':               0.015,     # M-step prior covariance embeddings (log σ_p) 0.015
   'M_phi_lr':                   0.005,    # M-step gauge frame embeddings (φ) 0.0075
   
   # === M-step Other LR's (AdamW parameter groups) ===
   'M_vfe_hyperparam_lr':        0.05,  # M-step VFE hyperparams (raw_c0, raw_b0, raw_lr) 0.05
   'M_attention_lr':             0.013,  # M-step attention params (W_O, constant_omega) was0.06
   'M_output_lr':                0.1,  # M-step output projection (vocab logits) 0.05
   'embed_weight_decay':         0.01,   # L2 hyper-prior on embeddings (μ_p, σ_p, φ) via AdamW
   'non_embed_weight_decay':     0.0005,  # L2 on non-embedding params (attention, output)

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

   # === Position encoding ===
   'use_rope':                   True,
   'rope_base':                  75,   
   'rope_full_gauge':            'off', # 'off', 'both', 'vfe_only'
   'pos_encoding_mode':          'none',

   # === Embedding init ===
   'mu_init_std':                0.1,
   'phi_scale':                  0.01,
   
   'mu_normalize':               False,
   'mu_max_norm':                None,


   # === Logging ===
   'log_interval':               100,
   'eval_interval':              20000,
   'checkpoint_interval':        25000,
   'semantic_analysis_interval': 40000,
   'gauge_geometry_interval':    40000,   # Gauge field Dirichlet energy + invariants
   'fiber_trajectory_interval':  40000,   # Fisher-Rao E-step trajectory (requires ffn_n_iterations > 1)

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
   
   # === Multi-layer depth signal ===
   'aux_layer_loss':              False,   # Enable for multi-layer: per-layer M-step CE loss
   'aux_loss_weight':             0.3,     # Weight for auxiliary per-layer CE losses

   # === Regularization ===
   'sigma_ce_scale':              0.7,
   'sigma_max':                   12.0,
   'grad_clip':                   50.0,
   'hidden_dim':                  508,
   
   'spd_floor_mode':             'eigclamp',      # new default, can omit
   'enable_spd_diagnostics':     True,    # shows spd_eig_min_t/q and cond
   'assert_finite_loss':         True,        # raise instead of silent skip
   
   'warmup_steps':                100,
   'num_workers':                 0,   # 0 is faster on Windows (spawn multiprocessing overhead)
   
   'use_amp':                     False, 
   'use_compile':                 False,
   'compile_mode':                'default',  # 'default', 'reduce-overhead', 'max-autotune'


   # ===== Active Inference =======
   'active_inference_pragmatic_weight':  2,   # start small
   'active_inference_epistemic_weight':  5,   # keep both ON to avoid feedback loop
   'active_inference_epistemic_samples': 6,     # MC samples for BALD
   
   
   
   
    'dataset':                    'wikitext-103',
}


# =============================================================================
# SWEEP DEFINITIONS
# =============================================================================
# Single-parameter sweeps support two value-source formats:
#   'values': [v1, v2, ...]            — explicit list
#   'range':  [start, stop, step]      — inclusive arithmetic range
#                                         e.g. [0.0, 0.1, 0.025] ->
#                                         [0.0, 0.025, 0.05, 0.075, 0.1]
# Multi-parameter categorical sweeps use 'configs': [ {param: value, ..., 'label': ...}, ... ]

SWEEPS = {
    # --- Tier 1: Free energy weights ---
    
    'E_mu_q_lr': {
        'description': 'E-step μ natural gradient step size',
        'param': 'E_mu_q_lr',
        'range': [0.0, 0.25, 0.025],
        'baseline_value': 0.1,
    },

    'E_sigma_q_lr': {
        'description': 'E-step σ trust region scale',
        'param': 'E_sigma_q_lr',
        'range': [0.0, 0.1, 0.005],
        'baseline_value': 0.05,
    },

    'E_phi_lr': {
        'description': 'E-step φ gauge frame step size',
        'param': 'E_phi_lr',
        'range': [0.0, 0.25, 0.025],
        'baseline_value': 0.05,
    },
    
    
    'M_beta': {
        'description': 'Belief alignment weight in outer training loss (lambda_beta)',
        'param': 'M_beta',
        'range': [0.0, 1, 0.05],
        'baseline_value': 0.0,
    },

    'lambda_hyper': {
        'description': 'Hyper-prior coupling KL(s||h) weight in training loss',
        'param': 'lambda_hyper',
        'range': [0.0, 0.2, 0.01],
        'baseline_value': 0.0,
    },

    # --- Tier 1b: Temperature / kappa parameters ---
    'kappa_beta': {
        'description': 'Attention/VFE temperature τ: β_ij = softmax(-KL/κ√K). Higher = softer attention',
        'param': 'kappa_beta',
        'range': [0.0, 10, 1],
        'baseline_value': 1,
    },

   

    'prior_bank_tau': {
        'description': 'PriorBank decode temperature: logits = -KL(q||π_v)/τ. Lower = sharper decode',
        'param': 'prior_bank_tau',
        'range': [0.0, 5, 0.1],
        'baseline_value': 1.0,
    },

    'rope_base': {
        'description': 'RoPE frequency base: θ_n = 1/base^(2n/K). Lower = faster position decay',
        'param': 'rope_base',
        'values': [25, 50, 75, 100, 150, 200],
        'baseline_value': 10,
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
        'values': [1, 3, 5, 7],
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
            #{'gauge_dim': 2, 'irrep_spec': [('fund', 24, 2)], 'embed_dim': 48, 'label': 'GL(2)_h24'},
           # {'gauge_dim': 4, 'irrep_spec': [('fund', 12, 4)], 'embed_dim': 48, 'label': 'GL(4)_h12'},
            {'gauge_dim': 6, 'irrep_spec': [('fund', 8, 6)], 'embed_dim': 48, 'label': 'GL(6)_h8'},
            {'gauge_dim': 8, 'irrep_spec': [('fund', 6, 8)], 'embed_dim': 48, 'label': 'GL(8)_h6'},
            {'gauge_dim': 12, 'irrep_spec': [('fund', 4, 12)], 'embed_dim': 48, 'label': 'GL(12)_h4'},
            {'gauge_dim': 16, 'irrep_spec': [('fund', 3, 16)], 'embed_dim': 48, 'label': 'GL(16)_h3'},
            {'gauge_dim': 24, 'irrep_spec': [('fund', 2, 24)], 'embed_dim': 48, 'label': 'GL(24)_h2'},
            {'gauge_dim': 48, 'irrep_spec': [('fund', 1, 48)], 'embed_dim': 48, 'label': 'GL(48)_h1'},
        ],
        'baseline_value': 'GL(10)_h2',
    },

    # --- Tier 11: LR schedule ---
    'min_lr_ratio': {
        'description': 'Minimum LR as fraction of peak (scheduler floor): effective_min = base_lr × ratio',
        'param': 'min_lr_ratio',
        'values': [0.0, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5],
        'baseline_value': 0.1,
    },



    'E_alpha': {
        'description': 'Prior coupling weight inside VFE E-step iterations',
        'param': 'E_alpha',
        'range': [0.0, 1, 0.1],
        'baseline_value': 1.0,
    },

    
    'E_lambda_belief': {
        'description': 'E-step belief alignment weight (VFE coupling term)',
        'param': 'E_lambda_belief',
        'range': [0, 25, 5],
        'baseline_value': 1,
    },

    'E_lambda_softmax': {
        'description': 'E-step softmax coupling weight (∂β/∂θ·KL Boltzmann gate)',
        'param': 'E_lambda_softmax',
        'range': [0, 19, 2],
        'baseline_value': 1,
    },
    
    
    
    
    'M_alpha': {
            'description': 'Self-consistency KL(q||p) weight in training loss',
            'param': 'M_alpha',
            'values': [0.0, 0.025, 0.05, 0.1],
            'baseline_value': 0.00,
        },


    'mass_phi': {
        'description': 'Gauge frame L2 prior weight (α_φ/2)||φ||²',
        'param': 'mass_phi',
        'values': [0, 1e-4, 2e-4, 3e-4, 4e-4, 5e-4],
        'baseline_value': 0.0,
    },





    # --- Tier 7: Learning rates ---
    'M_mu_p_lr': {
        'description': 'Belief mean (μ) learning rate',
        'param': 'M_mu_p_lr',
        'values': [0.02, 0.04, 0.06, 0.07, 0.08, 0.1],
        'baseline_value': 0.078,
    },

    'M_sigma_p_lr': {
        'description': 'Belief precision (σ) learning rate',
        'param': 'M_sigma_p_lr',
        'values': [0.0025, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.0175, 0.02],
        'baseline_value': 0.015,
    },

    'M_phi_lr': {
        'description': 'Gauge frame (φ) learning rate',
        'param': 'M_phi_lr',
        'values': [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007],
        'baseline_value': 0.00325,
    },

    'M_vfe_hyperparam_lr': {
        'description': 'FFN / VFE module learning rate',
        'param': 'M_vfe_hyperparam_lr',
        'values': [0,0.0001, 0.0005, 0.001, 0.005,  0.01, 0.05, 0.1, 0.5,  1],
        'baseline_value': 0.1,
    },

    'M_attention_lr': {
        'description': 'Attention module learning rate',
        'param': 'M_attention_lr',
        'values': [0,0.0001, 0.0005, 0.001, 0.005,  0.01, 0.05, 0.1, 0.5,  1],
        'baseline_value': 0.06,
    },

    'M_output_lr': {
        'description': 'Output projection learning rate',
        'param': 'M_output_lr',
        'values': [0,0.0001,0.0005,  0.001,0.005,  0.01, 0.05, 0.1,0.5,  1],
        'baseline_value': 0.065,
    },

    
    

    # --- Tier 9: M-step gradient scaling ---
    'sigma_ce_scale': {
        'description': 'CE→σ_p gradient scale in PriorBank decode (0=detach, 1=full)',
        'param': 'sigma_ce_scale',
        'range': [0.0, 1, 0.1],
        'baseline_value': 0.01,
    },

    # --- Tier 10: Regularization ---
    'embed_weight_decay': {
        'description': 'Weight decay on embedding parameters (hyper-prior precision)',
        'param': 'embed_weight_decay',
        'values': [0,0.0001,0.0005,  0.001,0.005,  0.01, 0.05, 0.1,0.5,  1],
        'baseline_value': 0.002,
    },

    'non_embed_weight_decay': {
        'description': 'Weight decay on non-embedding parameters',

        'param': 'non_embed_weight_decay',
        'values': [0,0.0001,0.0005,  0.001,0.005,  0.01, 0.05, 0.1,0.5,  1],
        'baseline_value': 0.005,
    },
    
    # --- Tier 9: M-step gradient scaling ---
    'phi_scale': {
        'description': 'phi init scale',
        'param': 'phi_scale',
        'values': [0, 1e-5, 1e-4, 1e-3, 1e-2],
        'baseline_value': 0.5,
    },

    # --- Tier 9: M-step gradient scaling ---
    'mu_init_std': {
        'description': 'mu-init-scale',
        'param': 'mu_init_std',
        'values': [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        'baseline_value': 0.01,
    },

    # --- Tier 12: Divergence family ---
    'alpha_divergence': {
        'description': 'Renyi alpha-divergence order (1.0=KL, 0.5=Bhattacharyya)',
        'param': 'alpha_divergence',
        'range': [0.4, 0.6, 0.02],
        'baseline_value': 0.75,
    },

    # --- E-step σ_p floor (caps E-step precision at 1/floor) ---
    'e_step_sigma_floor': {
        'description': 'Floor on σ_p inside E-step (caps 1/σ_p at 1/floor); 0.01 → precision ≤ 100',
        'param': 'e_step_sigma_floor',
        'values': [0.01, 0.2, 0.3, 0.4, 0.5],
        'baseline_value': 0.01,
    },

    # --- CE label smoothing (loss path only; raw CE preserved for PPL) ---
    'ce_label_smoothing': {
        'description': 'Label smoothing ε on CE loss term only (raw CE/PPL un-smoothed)',
        'param': 'ce_label_smoothing',
        'values': [0.0, 0.05, 0.1, 0.2],
        'baseline_value': 0.0,
    },

    # --- Tier 13: EM inference mode ---
    'em_mode': {
        'description': 'EM inference mode: amortization scope and phi treatment',
        'param': None,
        'configs': [
            # 1. Straight-through: μ_p,σ_p attached, semi-gradient φ
            {'em_mode': 'straight_through', 'label': 'straight_through'},
            # 2. IFT φ: μ_p,σ_p attached, full IFT φ gradient
            {'em_mode': 'ift_phi',          'label': 'ift_phi'},
            # 3. Clean EM: φ∈q — E-step optimizes μ,Σ,φ; all detached at boundary
            {'em_mode': 'em_phi_q', 'evolve_phi': True, 'evolve_phi_e_step': True, 'label': 'em_phi_q'},
            # 4. Clean EM: φ∈θ — φ frozen in E-step; M-step only
            {'em_mode': 'em_phi_p',         'label': 'em_phi_p'},
        ],
        'baseline_value': 'straight_through',
    },

}

# Sweep execution order (cheapest → most expensive)
SWEEP_ORDER = [
    
  #  'mass_phi',
    'M_alpha',
    #'em_mode',
   
   # 'M_sigma_p_lr',  #0.015
     
  #  
  #  'M_mu_p_lr',     #0.0825
   # 'sigma_ce_scale',   #0.7
  #  'alpha_divergence',     #0.275
  #  'M_phi_lr', #0.00325 
  #  'phi_scale',
  #  'mu_init_std',
    
     
  #  'M_vfe_hyperparam_lr', #0.1
  
    
  # 'rope_base',
  #  'E_sigma_q_lr',
  #  'embed_weight_decay',   #0.002
   # 'E_alpha',
   # 'E_lambda_belief',
  #  'E_lambda_softmax',
  #  'M_attention_lr',    #0.06
   # 'M_output_lr',       #0.065
   # 'non_embed_weight_decay', #0.005
    
 #  'E_phi_lr',
  # 'E_mu_q_lr',
  
    
   #  'e_step_sigma_floor',
   #  'ce_label_smoothing',
    
    
   #
    #'kappa_beta',
   # 'lambda_hyper',
    
    
  
  #  'M_beta',
    
   # 'min_lr_ratio',  
   
     #'gauge_dim', 
     #'K', 

     # 'prior_bank_tau',

  #  'covariance', 
    
  #  'gauge_mode', 
  #  'phi_preconditioner',
    
   # 'rope',

    

   # 'n_layers',
    #'n_vfe_iterations',

]




# =============================================================================
# RUNNER
# =============================================================================

def _expand_range(range_spec) -> list:
    """Expand a ``'range': [start, stop, step]`` spec into an explicit value list.

    Inclusive of ``stop`` when it lands on the grid (up to float tolerance).
    Values are rounded to 10 decimals so labels stay clean (``0.05`` not
    ``0.05000000000000001``).  Integer ``start/stop/step`` produce ints.
    """
    if isinstance(range_spec, dict):
        start = range_spec['start']
        stop = range_spec['stop']
        step = range_spec['step']
    else:
        if len(range_spec) != 3:
            raise ValueError(
                f"'range' spec must be [start, stop, step], got {range_spec!r}"
            )
        start, stop, step = range_spec
    if step == 0:
        raise ValueError("'range' step must be non-zero")

    all_int = all(isinstance(v, int) and not isinstance(v, bool)
                  for v in (start, stop, step))
    values = []
    n_steps = int(round((stop - start) / step))
    tol = abs(step) * 1e-9
    for i in range(n_steps + 2):  # +2 for float safety, break below
        v = start + i * step
        if step > 0 and v > stop + tol:
            break
        if step < 0 and v < stop - tol:
            break
        values.append(v if all_int else round(v, 10))
    return values


def _sweep_values(sweep: dict) -> list:
    """Return the explicit value list for a single-param sweep.

    Accepts either ``sweep['values']`` or ``sweep['range']``; returns ``[]``
    for categorical (``'configs'``) sweeps where values are not meaningful.
    """
    if 'configs' in sweep:
        return []
    if 'values' in sweep:
        return list(sweep['values'])
    if 'range' in sweep:
        return _expand_range(sweep['range'])
    raise KeyError(
        "Single-param sweep must define 'values' or 'range' "
        f"(sweep: {sweep!r})"
    )


def _sweep_num_runs(sweep: dict) -> int:
    """Number of (label, config) pairs this sweep will generate."""
    if 'configs' in sweep:
        return len(sweep['configs'])
    return len(_sweep_values(sweep))


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
        # Single-param sweep — supports 'values' and 'range'
        param = sweep['param']
        for val in _sweep_values(sweep):
            cfg = copy.deepcopy(base_config)
            cfg[param] = val
            label = f"{param}={val}"
            runs.append((label, cfg))

    return runs


def _cleanup_after_experiment():
    """Force memory reclamation between ablation runs."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_sweep(
    sweep_name: str,
    base_config: dict,
    device: torch.device,
    output_dir: Path,
    seed: int = 6,
    max_steps_override: int = None,
    resume: bool = False,
) -> pd.DataFrame:
    """Run all configs in a sweep, return results DataFrame."""
    sweep = SWEEPS[sweep_name]
    sweep_dir = output_dir / sweep_name
    sweep_dir.mkdir(parents=True, exist_ok=True)

    runs = make_run_configs(sweep_name, base_config)

    print(f"\n{'='*70}")
    print(f"SWEEP: {sweep_name} ({len(runs)} configs)")
    print(f"  {sweep['description']}")
    print(f"  Baseline: {sweep['baseline_value']}")
    print(f"  Output: {sweep_dir}")
    if resume:
        print(f"  Resume: ON (skipping completed runs)")
    print(f"{'='*70}\n")

    results = []

    # In resume mode, load any previously completed results
    if resume:
        for label, _cfg in runs:
            run_dir = sweep_dir / label.replace('=', '_').replace(' ', '_')
            result_file = run_dir / 'ablation_result.json'
            if result_file.exists():
                with open(result_file) as f:
                    results.append(json.load(f))

    # Pre-build dataloaders once per sweep and reuse them across every
    # parameter value.  All current SWEEPS only vary training/architecture
    # hyperparameters — none touch 'dataset', 'vocab_size', 'max_seq_len',
    # 'batch_size', 'num_workers', or 'tokenizer' — so the loaders are
    # identical across runs.  This avoids re-loading the (multi-hundred-MB)
    # token cache and respawning DataLoader workers on every single run,
    # which on Windows was the dominant between-run delay.
    print("  Building shared dataloaders for sweep (one-time cost)...")
    from transformer.data import datasets as _datasets_mod
    _prev_quiet = _datasets_mod.QUIET
    _datasets_mod.QUIET = True
    try:
        shared_loaders = _create_dataloaders(base_config)
    finally:
        _datasets_mod.QUIET = _prev_quiet

    for i, (label, cfg) in enumerate(runs):
        run_dir = sweep_dir / label.replace('=', '_').replace(' ', '_')

        # Skip already-completed runs in resume mode
        if resume and (run_dir / 'ablation_result.json').exists():
            print(f"\n--- Run {i+1}/{len(runs)}: {label} [CACHED] ---")
            continue

        print(f"\n--- Run {i+1}/{len(runs)}: {label} ---\n\n\n\n")

        # Override max_steps if requested
        if max_steps_override is not None:
            cfg['max_steps'] = max_steps_override

        # Set seed for reproducibility
        from transformer.training.utils import set_all_seeds
        set_all_seeds(seed)

        try:
            t0 = time.time()
            result = run_single_experiment(
                config=cfg,
                ffn_mode=cfg.get('ffn_mode', 'VFE_dynamic'),
                device=device,
                checkpoint_dir=run_dir,
                enable_publication_metrics=False,
                quiet=True,
                skip_test_eval=True,
                skip_post_training_viz=True,
                preloaded_data=shared_loaders,
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

                # Count remaining runs (exclude cached)
                _completed = i + 1
                _remaining = len(runs) - _completed
                _cached = sum(1 for lbl, _ in runs[:i] if resume and (sweep_dir / lbl.replace('=', '_').replace(' ', '_') / 'ablation_result.json').exists())
                _remaining -= _cached if resume else 0

                if _completed == 1 and _remaining > 0:
                    # After first run: project total sweep time
                    _est_total = elapsed * (_remaining + 1)
                    _est_remain = elapsed * _remaining
                    print(f"  -> Val PPL: {result['final_ppl']:.2f}, "
                          f"Time: {elapsed:.0f}s")
                    print(f"  ** Estimated sweep total: {_est_total/60:.0f} min "
                          f"({_remaining} remaining x {elapsed:.0f}s = ~{_est_remain/60:.0f} min left)")
                else:
                    print(f"  -> Val PPL: {result['final_ppl']:.2f}, "
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
        finally:
            _cleanup_after_experiment()

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

    print(f"{'Label':<30} {'Val PPL':>10} {'Params':>12} {'Time':>8}")
    print('-' * 65)
    for _, row in valid.iterrows():
        params = f"{row.get('total_params', 0):,}" if pd.notna(row.get('total_params')) else 'N/A'
        elapsed = f"{row.get('elapsed_sec', 0):.0f}s" if pd.notna(row.get('elapsed_sec')) else 'N/A'
        print(f"{row['label']:<30} {row['final_ppl']:>10.2f} {params:>12} {elapsed:>8}")

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
    print(f"{'Sweep':<25} {'Best Config':<30} {'Val PPL':>10}")
    print('-' * 70)

    for r in all_results:
        df = r['df']
        valid = df[df['final_ppl'] < float('inf')]
        if valid.empty:
            continue
        best = valid.loc[valid['final_ppl'].idxmin()]
        sweep = r['meta'].get('sweep_name', '?')
        print(f"{sweep:<25} {best['label']:<30} {best['final_ppl']:>10.2f}")


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

        # Always use val PPL for sweep comparisons (test reserved for final reporting)
        ppl_col = 'final_ppl'
        ppl_label = 'Validation Perplexity'

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
            ax.plot(valid['param_value'], valid[ppl_col], 'o-', linewidth=2, markersize=8)
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
            valid = valid.sort_values(ppl_col)
            colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(valid))]
            ax.barh(range(len(valid)), valid[ppl_col], color=colors)
            ax.set_yticks(range(len(valid)))
            ax.set_yticklabels(valid['label'])
            ax.invert_yaxis()

        ax.set_ylabel(ppl_label)
        ax.set_title(f"Ablation: {meta.get('description', sweep_name)}")
        fig.tight_layout()
        fig.savefig(fig_dir / f'{sweep_name}.png', dpi=150)
        
        plt.close(fig)
       

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
        ax.set_xlabel('Val PPL Range (worst - best)')
        ax.set_title('Hyperparameter Sensitivity (Val PPL range per sweep)')
        ax.invert_yaxis()
        fig.tight_layout()
        fig.savefig(fig_dir / 'sensitivity_summary.png', dpi=150)
       
        plt.close(fig)
        print(f"  Saved: {fig_dir / 'sensitivity_summary'}.{{png}}")

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
    parser.add_argument('--seed', type=int, default=6)
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
            n = _sweep_num_runs(s)
            print(f"{name:<25} {n:>10} {s['description']}")
        total = sum(_sweep_num_runs(SWEEPS[n]) for n in SWEEP_ORDER)
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