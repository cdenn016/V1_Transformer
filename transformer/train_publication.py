# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 08:09:57 2026

@author: chris and christine
"""


# =============================================================================
# PATH SETUP - Ensure the project root is in the Python path
# This allows the script to be run from any directory (including the transformer/ folder)
# =============================================================================
import sys
import os

# Get the directory containing this script
_script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the project root (parent of transformer/)
_project_root = os.path.dirname(_script_dir)
# Add project root to path if not already there
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

"""
Publication Proof-of-Principle Training Script
===============================================

Language modeling on WikiText-2/103 with byte-level encoding for minimal publishable claim.

Demonstrates:
1. Variational FFN works - inference comparable to learned MLP
2. Architecture is trainable - converges to reasonable performance
3. Theoretical framework is sound - gauge-invariant inference holds

  

Comprehensive Metrics Tracking:
    - Free energy components (α, β, γ terms)
    - Gradient norms (total, μ, FFN)
    - All learning rates (μ, σ, φ, FFN)
    - Bits-per-character (BPC)
    - Attention statistics (β_mean, KL_mean)
    - Performance (step time, tokens/sec)
    - Hamiltonian diagnostics (H_init, H_final, ΔH) for hamiltonian mode

Output Files:
    - checkpoints_publication/ffn_{mode}/metrics.csv - comprehensive training metrics
    - checkpoints_publication/ffn_{mode}/best_model.pt - best model checkpoint
    - checkpoints_publication/result_{mode}.json - final summary (if single mode)
    - checkpoints_publication/ablation_results.json - comparison (if --run_ablation)

Usage:
    # Just click Run (edit defaults below)
    python transformer/train_publication.py


Author: Designed for minimal publishable claim
Date: December 2025
"""

import torch
import torch.nn.functional as F
import argparse
import json
import csv
import time
import math
import subprocess
import platform
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any


from transformer.core.model import GaugeTransformerLM
from transformer.baselines.standard_transformer import StandardTransformerLM
from transformer.pure_vfe.model import PureVFETransformer
from transformer.pure_vfe.config import PureVFEConfig
from transformer.pure_vfe.gauge import monitor_omega_health
from transformer.baselines.flops_counter import (
    count_standard_transformer_flops,
    count_gauge_transformer_flops,
    format_flops,
    compare_flops,
    print_flops_comparison,
)
from transformer.data import create_dataloaders, create_char_dataloaders
from transformer.train import (
    compute_free_energy_loss,
    compute_rg_metrics_from_attention,
    compute_dynamic_rg_metrics,
)
from transformer.training.train_fast import FastTrainer, FastTrainingConfig
from transformer.analysis.publication_metrics import PublicationMetrics, ExperimentResult
from math_utils.numerical_monitor import flush as _flush_numerical_events


# ============================================================================
# EDIT THESE DEFAULTS TO RUN WITHOUT COMMAND-LINE ARGS (just click Run!)
# ============================================================================
# Modes available:
#   'standard'    - Standard transformer baseline (dot-product attention + MLP)
#   'VFE_dynamic' - VFE with EM-step dynamics (backprop training)
#   'pure_vfe'    - Pure VFE transformer (no backprop, natural gradient only)
#   'standard_attn_only',      # (b) attention-only at d_model=90
#   'standard_param_equalized', # (b') param-equalized wider FFN
#   'standard_rope',            # (c) standard + RoPE at d=10
#   'standard_rope_d90',        # (c') standard + RoPE at d=90

DEFAULT_MODE = 'VFE_dynamic'      # Which mode to run

# Dataset
DEFAULT_DATASET = 'wikitext-103'  
# 'wikitext-2' (~2M tokens) or 'wikitext-103' (~103M tokens), 'wiki-ja' japanese (~1B tokens, 100k vocab)
# ============================================================================



# =============================================================================
# CONFIG 1: STANDARD TRANSFORMER (Baseline)
# =============================================================================
# Standard dot-product attention + learned MLP for fair comparison.
# This is the BASELINE to beat!
#
# Architecture:
#   - Attention: Q·K^T / √d (standard dot-product softmax)
#   - FFN: Linear → GELU → Linear (learned MLP)
#   - Output: Linear projection to vocab
#   - Learning: Backpropagation (standard)
#   - Position: Learned positional embeddings
# =============================================================================
STANDARD_CONFIG = {
    # Model architecture — param-matched to VFE_EM_CONFIG (1.52M)
    'vocab_size': 50257,
    'embed_dim': 10,              # Same as VFE for apples-to-apples comparison
    'n_layers': 1,                # Same depth
    'n_heads': 1,                 # Single head (head_dim=10)
    'hidden_dim': 24527,          # Absorbs params that VFE spends on σ table + VFE machinery
    'max_seq_len': 128,

    # Training — match VFE config
    'batch_size': 64,
    'use_amp': False,
    'num_workers': 10,
    'epochs': None,
    'max_steps': 15000,
    'warmup_steps': 100,

    # Standard transformer settings
    'ffn_mode': 'standard',
    'attention_type': 'standard',
    'pos_encoding_mode': 'learned',
    'tie_embeddings': False,        # Match VFE (tie=False)

    # Disable gauge features
    'evolve_sigma': False,
    'evolve_phi': False,
    'diagonal_covariance': True,
    'isotropic_covariance': False,
    'use_positional_embedding': True,

    # Learning rates
    'mu_lr': 3e-4,
    'sigma_lr': 0.0001,
    'phi_lr': 0.0001,
    'ffn_lr': 3e-4,

    # Free energy weights (not used in standard mode)
    'alpha': 0,
    'beta': 0,
    'lambda_gamma': 0,
    'kappa_gamma': 1.0,

    # Regularization
    'weight_decay': 0.01,
    'dropout': 0.1,
    'grad_clip': 1.0,

    # Logging — match VFE config
    'log_interval': 100,
    'eval_interval': 1000,
    'checkpoint_interval': 25000,
    'patience': 5,

    # Unused in standard mode
    'kappa_beta': 1.0,
    'attention_pattern': 'full',
    'attention_window': 24,
    'gauge_group': 'SO3',
    'gauge_dim': 3,
    'gauge_mode': 'learned',
    'gauge_fixed_priors': True,
    'irrep_spec': [('ℓ0', 5, 1)],
    'compute_rg_metrics': False,
}




# =============================================================================
# CONFIG 2: VFE_EM (VFE with EM-step dynamics, uses backprop)
# =============================================================================
# Gauge-equivariant transformer with Variational Free Energy dynamics.
# Uses EM-step belief updates with backprop for training.
#
# Architecture:
#   - Attention: KL-divergence based (gauge-equivariant)
#   - FFN: VFE EM-step dynamics (belief inference)
#   - Output: Linear projection to vocab
#   - Learning: Backpropagation
#   - Position: None (emergent from data)
# =============================================================================
SEED = 6

VFE_EM_CONFIG = {
    # Model architecture
    'vocab_size': 50257,          # Will be overridden by tokenizer
    'embed_dim': 10,              # Embedding dimension K
    'n_layers': 1,                # Transformer depth
    'hidden_dim': 508,            # Only used if ffn_mode='learned'
    'max_seq_len': 128,           # Context length N

    'learnable_alpha': False,
    
    'use_obs_in_vfe': False,        #cheats when true!  low trainPPL huge val PPL
    'obs_sigma_gradient': False,    # Add ∂E_q[CE]/∂σ Hessian-diagonal gradient in VFE E-step
                                    # Requires use_obs_in_vfe=True to have any effect
    'obs_sigma_weight': 1.0,        # Weight for sigma observation gradient
    'amortized_inference': True,   # Gradient flow through priors → embeddings learn good E-step init

    'use_deq': False,
    'deq_neumann_terms': 0,

    # Training
    'batch_size': 32,
    'num_workers': 10,            #CPU workers 8--12
    'epochs': None,               # Set to 1-3 for WikiText-2, None for WikiText-103 (use max_steps)
    'max_steps': 30000,           # ~105% coverage on WikiText-103
    'warmup_steps': 100,

    # VFE transformer settings
    'ffn_mode': 'VFE_dynamic',    # VFE EM-step dynamics
    'mask_self_attention': True,  # Prevent attention collapse? needed if learnable-reflection true??
    'tie_embeddings': False,

    # Gauge geometry
    'evolve_sigma': True,         # Learn covariances Σ
    'evolve_phi': True,           # Learn gauge frames φ (M-step, via backprop)
    'evolve_phi_e_step': True,    # Update φ during E-step iterations (dynamical gauge frames)
                                  # When True: φ evolves via ∂F/∂φ at each VFE iteration
                                  # When False: φ only updated via backprop (M-step)

    'phi_update_interval': 1,
    'analytic_phi_grad': False,   # If True, bypass autograd for ∂F/∂φ (saves ~250MB per update)
                                  # Uses hand-coded backward through matrix_exp → KL → softmax.
                                  # Requires diagonal_covariance=True and irrep_dims (block-diag).
    'analytic_phi_grad_dexp_order': 4,  # dexp series truncation (4=good, 8=very accurate)
    
    'diagonal_covariance': True,    # approximate diag(Ω @ diag(σ) @ Ω^T) path runs with zero overhead
    
    'exact_diagonal_transport': False,  #exact diagonal transport - more expensive
    'isotropic_covariance': False,    # If True, force Σ = σ²I (scalar variance × identity)
                                       # This is Limit 1 from the manuscript: KL reduces to
                                       # scaled squared Euclidean distance. Combined with
                                       # gauge_mode='trivial' (Limit 2), recovers standard attention.
    
    'enforce_orthogonal': False,   # If True, enforce Ω ∈ SO(K) via Newton-Schulz
                                   # Set False for GL(K) (faster, still gauge-invarian
    'learnable_reflection': False ,   # Per-token s_i ∈ {±1}^K → O(K)  - enforce orthogonal=true with glk 
                                      # Set gauge-mode=constant and the above 3 = true for transf limit
    
    
    'use_positional_embedding': False,
    'pos_encoding_mode': 'none',           #'none' 'learned' or 'sinusoidal'
    'use_rope': True,
    
    
    'alibi_slope': None,

    # Temperature: κ is a scalar sharpness dial; dimension scaling (√K) is hardcoded in attention
    'kappa_beta': 1,

    # Embedding initialization
    'mu_init_std': 1.0,
    'mu_normalize': False,
    'mu_max_norm': None,
    'phi_scale': 1.0,             # Gauge frame initialization scale (try 1.0-2.0 for clustering)

    # VFE dynamics
    'ffn_n_iterations': 1,
    'ffn_learnable_lr': True,
    'ffn_chunk_size': None,          #smaller if running out of memory. make large as possible

    # Learning rates
    'mu_lr':     0.05,
    'sigma_lr':  0.005,
    'phi_lr':    0.005,
    'ffn_lr':    0.05 ,

    'attention_lr': 0.005,
    'output_lr': 0.05,
    
    # Free energy loss weights (see compute_free_energy_loss in train.py)
    # NOTE: config['beta'] maps to the lambda_beta parameter in compute_free_energy_loss().
    # This is the belief coupling weight Σ β_ij·KL(q_i||Ω_ij q_j) in the TRAINING LOSS,
    # NOT the attention weights β_ij used inside VFE dynamics. Confusing but entrenched.
    'alpha':        0.075,        # KL(q||p) self-consistency weight
    'alpha_phi':    0.1,          # Gauge prior: (α_φ/2)||φ||² mass term (0 = disabled)
    'beta':         0,            # beta=lambda_beta in loss: belief alignment weight (0 = off)
    'beta_warmup_steps': 2000,    # Linear warmup for beta: 0 → target over this many steps
                                  # Lets CE differentiate embeddings before coupling kicks in
    'lambda_gamma': 0,            # Model coupling: Σγ_ij·KL(s_i||Ω s_j) (0 = off)
    'lambda_hyper': 0,            # Hyper-prior: KL(s_i||h) models to centroid (0 = off)
    'kappa_gamma':  1,            # Temperature for γ_ij coupling weights
    
    # VFE E-step internal weights (inside VariationalFFNDynamic, NOT the training loss)
    'ffn_lambda_belief': 1,       # Belief alignment inside VFE iterations
    'ffn_alpha': 1,               # Prior coupling inside VFE iterations

    # Regularization
    'weight_decay': 0.01,
    'grad_clip':    1.0,

    'use_layernorm': True,      # Critical!
    'use_residual':  True,

    'log_interval': 100,
    'eval_interval': 1000,
    'checkpoint_interval': 25000,
    'semantic_analysis_interval': 10000,

    # =================================================================
    # GAUGE GROUP SELECTION (Generators from so(N), Transport in GL(K))
    # =================================================================
    # NOTE: The VFE is invariant under GL(K), not just SO(K)!
    # We use so(N) generators to parameterize φ, but transport operators
    # Ω = exp(φ·G) live in GL(K). No orthogonality constraint is needed.
    #
    # SO3: so(3) generators with 3 parameters (rotation-only subalgebra)
    #      Requires embed_dim = sum(mult * dim) for irrep_spec or odd embed_dim
    # SON: so(N) generators with N(N-1)/2 parameters
    #      Supports multiple irrep types for representational diversity:
    #        - 'scalar': dim = 1              (gauge-invariant)
    #        - 'fund':   dim = N              (fundamental/vector)
    #        - 'wedge2': dim = N*(N-1)/2      (antisymmetric 2-tensor ∧²V)
    #        - 'sym2':   dim = N*(N+1)/2 - 1  (symmetric traceless Sym²₀V)
    #
    #      Different irreps have different Casimir eigenvalues:
    #        fund ~1.0x, wedge2 ~1.5x, sym2 ~2.5x
    #      This provides genuine transformation diversity (like SO(3) spin-ℓ)
    # =================================================================
    
    'gauge_group': 'GLK',       # 'SO3', 'SON', or 'GLK'
    'gauge_dim': 10,            # N for SO(N) - only used when gauge_group='SON'
    'gauge_mode': 'learned',    # 'learned': per-token φ, Ω_ij = exp(φ_i)·exp(-φ_j) (cocycle)
                                # 'constant': per-head Ω ∈ GL(d_head), Ω_ij = Ω (manuscript Limit 2)
                                # 'trivial': φ = 0, Ω = I (standard attention)
    
    # Gauge geometry: principled phi gradient control (replaces ad-hoc clipping)
    'phi_natural_gradient': 'killing',  # E-step: 'pullback', 'killing', 'cartan', 'clip'
    
    
    'use_slk_projection': False,         # Project phi to traceless sl(K) after each step
    'use_killing_form': True,            # M-step Cartan decomposition preconditioning for phi grads
    'killing_form_sym_dampening': 0.1,   # M-step Dampening for non-compact directions (0.1 = 10× reduction)
    
                        

    # P-FLOW: EMA update of token embeddings toward successful beliefs
    # This is the key learning mechanism from fep_transformer.py
    'use_p_flow': False,           # Enable P-flow updates on token embeddings
    'p_flow_ema_decay': 0.95,      # EMA decay (higher = slower update, 0.99 = 1% per step)

    # PHI DETACH: Detach phi from backprop in non-amortized mode.
    # When True + use_p_flow, phi learns via phi P-flow (EMA toward VFE-evolved values)
    # instead of backprop. Combined with P-flow + delta rule, makes training fully backprop-free.
    'detach_phi': False,               # Detach phi in non-amortized mode (default False for compat)

    # DELTA RULE: Backprop-free learning for W_out
    # If True, W_out is updated via delta rule instead of backpropagation
    # Combined with P-flow + detach_phi, this makes learning fully backprop-free.
    'use_delta_rule_w_out': False,  # Enable delta rule for W_out (instead of backprop)
    'delta_rule_lr': 0.1,           # Learning rate for delta rule updates

    
    # Irrep structure for SO(N)
    # Example for SO(5) with K=132:
    #   [('scalar', 10, 1), ('fund', 8, 5), ('wedge2', 4, 10), ('sym2', 3, 14)]
    #   = 10 + 40 + 40 + 42 = 132
    'irrep_spec': [
      # ('ℓ0', 50, 1),   # 75 dimensions (scalars)
      # ('ℓ1', 1, 3),   # 90 dimensions (vectors)
      # ('ℓ2', 2, 5),   # 90 dimensions (rank-2 tensors)
     #  ('ℓ3', 1, 7),
      # ('ℓ4', 1, 9),
      #('ℓ5', 9, 11),
     # ('ℓ6', 1, 13),
     # ('ℓ7', 1, 15),
      # ('ℓ50', 1, 101),
      ('fund', 1, 10)  #For SO(8)
     # ('fund', 10, 5),   # SO(5)
       
     # SO(5) multi-irrep example:
     # ('scalar', 10, 1),   # 10 dims (invariant)
     # ('fund', 8, 5),      # 40 dims (vector)
     # ('wedge2', 4, 10),   # 40 dims (∧² - angular momentum)
     # ('sym2', 3, 14),     # 42 dims (Sym²₀ - quadrupolar)
    ],
     
         
    # Option A: couple just 0↔1, head 2 stays independent
    #'cross_couplings': [(0, 1), (1, 0)],
    # → super-blocks: [20, 10]  (heads 0,1 merged into GL(20), head 2 alone)


    # Per-head specialization & multi-head VFE
    'use_output_projection': True,  # W_O cross-head mixing after attention (toggle)
    'multihead_vfe': True,          # Maintain per-head β_h through VFE iterations

    # RG metrics
    #'compute_rg_metrics': True,   # Enable RG flow analysis
    #'rg_metrics_interval': 500,   # Compute every 100 steps (not too frequent)
    #'rg_auto_cluster': True,
    #'rg_n_clusters': None,
    #'track_dynamic_rg': True,  # Track RG flow across VFE iterations (requires n_iterations > 1)


    'pos_encoding_scale': 0.3,
    'use_prior_bank': False,

    # =================================================================
    # NON-FLAT GAUGE TRANSPORT (holonomy)
    # =================================================================
    # When enabled, transport acquires an edge-local connection δ_ij:
    #   phi path:  Ω_ij = exp(φ_i·G) · exp(α·δ_ij·G) · exp(-φ_j·G)
    #   omega path: Ω_ij = Ω_i · exp(α·δ_ij·G) · Ω_j⁻¹
    # δ_ij is zero-initialized so the model starts flat and learns
    # curvature only where the data warrants it.
    # Holonomy H_ijk = Ω_ij·Ω_jk·Ω_ki ≠ I when δ ≠ 0.
    'non_flat_transport': False,        # Enable edge-dependent connection δ_ij
    'cocycle_relaxation': 0.0,          # Scale for δ_ij: 0=flat, 1=fully non-flat
    'connection_type': 'bilinear',      # 'bilinear' (δ_ij^a = μ_i^T W^a μ_j) | 'mlp'
    'connection_hidden_dim': 64,        # Hidden dim for MLP connection (ignored for bilinear)
    'connection_init_scale': 0.01,      # W init scale (0=flat saddle point, 0.01 recommended)
    'holonomy_penalty': 0.0,            # λ_H · E[‖C_ijk - I‖²_F] regularizer (0 = off)
    'holonomy_interval': 500,           # Holonomy diagnostics every N steps (0 = disabled)
    'holonomy_sample_size': 500,        # Random triples per holonomy computation

}


# =============================================================================
# CONFIG 3: PURE VFE TRANSFORMER (No backprop, natural gradient only)
# =============================================================================
# The purest realization of the free energy principle for sequence modeling.
# NO nn.Module, NO autograd, NO optimizer. The entire system — inference AND
# learning — operates through natural gradient descent on the gauge-covariant
# VFE with analytic closed-form gradients.
#
# Architecture:
#   - Model: Prior bank {N(μ_v, Σ_v), Ω_v} per vocabulary token
#   - Inference: E-step VFE descent (replaces forward pass)
#   - Learning: M-step natural gradient on prior bank
#   - Attention: KL-divergence based with gauge transport
#   - No linear projections, no output head — logits = −KL(q||π_v)
# =============================================================================
PURE_VFE_CONFIG = {
    # Belief geometry
    'vocab_size': 50257,
    'belief_dim': 32,             # K: full belief dimension
    'n_heads': 4,                 # H: number of heads (block-diagonal)
    'head_dim': 8,                # K_h = K / H

    # E-step (inference = forward pass)
    'n_esteps': 12,               # Iterations of VFE descent (replaces "depth")
    'tau': None,                  # Attention temperature (defaults to √K_h)
    'eta_E': 0.1,                 # E-step natural gradient step size

    # M-step (learning = parameter update)
    'eta_M': 0.05,                # M-step natural gradient step size (match VFE_dynamic mu_lr)

    # Prior precision (state-dependent α)
    'alpha_b0': 1.0,
    'alpha_c0': 1.0,

    # Hyper-prior regularization
    'hyper_var': 100.0,

    # Sequence
    'max_seq_len': 128,           # Match other modes
    'batch_size': 32,

    # Initialization
    'sigma_init': 1.0,
    'omega_init_scale': 0.01,

    # E-step numerical stability
    'sigma_lr_ratio': 0.05,
    'e_step_lr_decay': 0.5,
    'grad_clamp': 1e3,

    # Trust regions
    'trust_region_mu': 2.0,
    'trust_region_sigma': 0.15,
    'trust_region_omega': 0.3,

    # SPD retraction
    'spd_eps_min': 1e-3,
    'spd_kappa_max': 1e4,
    'spd_exp_clip': 20.0,

    # Prior safeguards
    'prior_sigma_floor': 0.5,
    'prior_mu_max_norm': 10.0,
    'm_step_trust_mu': 0.5,

    # Gauge frame parameterization
    'gauge_param': 'omega',       # 'omega' (direct GL(K)) or 'phi' (Lie algebra)
    'omega_cond_max': 100.0,
    'phi_max_norm': 3.14159,

    # M-step options
    'sigma_obs_grad': 'none',
    'm_step_eta_floor': 0.01,

    # Recovery
    'nan_recovery': True,

    # Causal masking
    'causal': True,

    # Device & kernels
    'device': 'cuda',
    'use_cuda_kernels': True,

    # Training loop params (used by run_pure_vfe_experiment, not PureVFEConfig)
    'max_steps': 30000,
    'log_interval': 100,
    'eval_interval': 1000,
    'checkpoint_interval': 25000,
    'num_workers': 10,
}



# =============================================================================
# INTERMEDIATE BASELINES (Peer Review M2b, M2c, M2e)
# =============================================================================
# These baselines address reviewer concerns about confounded comparisons:
#
# (b) ATTENTION-ONLY BASELINE: Standard transformer at d_model=90 with MLP
#     disabled, to isolate the attention mechanism contribution.
#
# (b') PARAM-EQUALIZED WIDER BASELINE: Standard transformer with parameter
#      count equalized to gauge model via wider FFN layers.
#
# (c) MATCHED POSITIONAL ENCODING: Standard transformer using RoPE (same as
#     gauge model) to control for positional encoding confound.
#
# (e) FLOPs are reported via flops_counter.py for all configurations.
# =============================================================================


# Config (b): Standard transformer at d_model=90, MLP disabled (attention-only)
# Isolates the contribution of dot-product attention without FFN
STANDARD_ATTN_ONLY_CONFIG = {
    # Model architecture — match gauge model's embed_dim
    'vocab_size': 50257,
    'embed_dim': 90,              # Same as gauge model embed_dim
    'n_layers': 1,                # Same depth
    'n_heads': 9,                 # 9 heads * 10 = 90 (head_dim=10, matching gauge d_head)
    'hidden_dim': 360,            # Not used (FFN disabled), but needed for config
    'max_seq_len': 128,
    'disable_ffn': True,          # KEY: no FFN, attention only

    # Training — match VFE config
    'batch_size': 64,
    'use_amp': False,
    'num_workers': 10,
    'epochs': None,
    'max_steps': 15000,
    'warmup_steps': 100,

    # Standard transformer settings
    'ffn_mode': 'standard',
    'attention_type': 'standard',
    'pos_encoding_mode': 'learned',
    'tie_embeddings': False,

    # Disable gauge features
    'evolve_sigma': False,
    'evolve_phi': False,
    'diagonal_covariance': True,
    'isotropic_covariance': False,
    'use_positional_embedding': True,

    # Learning rates
    'mu_lr': 3e-4,
    'sigma_lr': 0.0001,
    'phi_lr': 0.0001,
    'ffn_lr': 3e-4,

    # Free energy weights (not used in standard mode)
    'alpha': 0,
    'beta': 0,
    'lambda_gamma': 0,
    'kappa_gamma': 1.0,

    # Regularization
    'weight_decay': 0.01,
    'dropout': 0.1,
    'grad_clip': 1.0,

    # Logging
    'log_interval': 100,
    'eval_interval': 1000,
    'checkpoint_interval': 25000,
    'patience': 5,

    # Unused in standard mode
    'kappa_beta': 1.0,
    'attention_pattern': 'full',
    'attention_window': 24,
    'gauge_group': 'SO3',
    'gauge_dim': 3,
    'gauge_mode': 'learned',
    'gauge_fixed_priors': True,
    'irrep_spec': [('ℓ0', 5, 1)],
    'compute_rg_metrics': False,
}


# Config (b'): Standard transformer parameter-equalized via wider FFN
# Match the gauge model's total parameter count (~58.8M) by widening FFN
STANDARD_PARAM_EQUALIZED_CONFIG = {
    # Model architecture — wider FFN to absorb parameter budget
    'vocab_size': 50257,
    'embed_dim': 90,              # Same as gauge model
    'n_layers': 1,
    'n_heads': 9,                 # 9 heads * 10 = 90
    'hidden_dim': 360,            # Will be auto-calculated to match param count
    'max_seq_len': 128,

    # Training
    'batch_size': 64,
    'use_amp': False,
    'num_workers': 10,
    'epochs': None,
    'max_steps': 15000,
    'warmup_steps': 100,

    # Standard transformer settings
    'ffn_mode': 'standard',
    'attention_type': 'standard',
    'pos_encoding_mode': 'learned',
    'tie_embeddings': False,

    # Disable gauge features
    'evolve_sigma': False,
    'evolve_phi': False,
    'diagonal_covariance': True,
    'isotropic_covariance': False,
    'use_positional_embedding': True,

    # Learning rates
    'mu_lr': 3e-4,
    'sigma_lr': 0.0001,
    'phi_lr': 0.0001,
    'ffn_lr': 3e-4,

    # Free energy weights
    'alpha': 0,
    'beta': 0,
    'lambda_gamma': 0,
    'kappa_gamma': 1.0,

    # Regularization
    'weight_decay': 0.01,
    'dropout': 0.1,
    'grad_clip': 1.0,

    # Logging
    'log_interval': 100,
    'eval_interval': 1000,
    'checkpoint_interval': 25000,
    'patience': 5,

    # Unused in standard mode
    'kappa_beta': 1.0,
    'attention_pattern': 'full',
    'attention_window': 24,
    'gauge_group': 'SO3',
    'gauge_dim': 3,
    'gauge_mode': 'learned',
    'gauge_fixed_priors': True,
    'irrep_spec': [('ℓ0', 5, 1)],
    'compute_rg_metrics': False,
}


# Config (c): Standard transformer with RoPE (matching gauge model's PE)
# Isolates attention mechanism contribution by controlling for positional encoding
STANDARD_ROPE_CONFIG = {
    # Model architecture — same as STANDARD_CONFIG but with RoPE
    'vocab_size': 50257,
    'embed_dim': 10,              # Same as VFE for apples-to-apples
    'n_layers': 1,
    'n_heads': 1,
    'hidden_dim': 24527,          # Same as STANDARD_CONFIG
    'max_seq_len': 128,
    'use_rope': True,             # KEY: RoPE to match gauge model
    'rope_base': 10000.0,

    # Training
    'batch_size': 64,
    'use_amp': False,
    'num_workers': 10,
    'epochs': None,
    'max_steps': 15000,
    'warmup_steps': 100,

    # Standard transformer settings
    'ffn_mode': 'standard',
    'attention_type': 'standard',
    'pos_encoding_mode': 'rope',  # Use RoPE instead of learned
    'tie_embeddings': False,

    # Disable gauge features
    'evolve_sigma': False,
    'evolve_phi': False,
    'diagonal_covariance': True,
    'isotropic_covariance': False,
    'use_positional_embedding': False,  # No learned pos embed when using RoPE

    # Learning rates
    'mu_lr': 3e-4,
    'sigma_lr': 0.0001,
    'phi_lr': 0.0001,
    'ffn_lr': 3e-4,

    # Free energy weights
    'alpha': 0,
    'beta': 0,
    'lambda_gamma': 0,
    'kappa_gamma': 1.0,

    # Regularization
    'weight_decay': 0.01,
    'dropout': 0.1,
    'grad_clip': 1.0,

    # Logging
    'log_interval': 100,
    'eval_interval': 1000,
    'checkpoint_interval': 25000,
    'patience': 5,

    # Unused in standard mode
    'kappa_beta': 1.0,
    'attention_pattern': 'full',
    'attention_window': 24,
    'gauge_group': 'SO3',
    'gauge_dim': 3,
    'gauge_mode': 'learned',
    'gauge_fixed_priors': True,
    'irrep_spec': [('ℓ0', 5, 1)],
    'compute_rg_metrics': False,
}


# Config (c'): Standard transformer with RoPE, at d_model=90 and param-matched
# Both matched PE and matched params for the fairest possible comparison
STANDARD_ROPE_D90_CONFIG = {
    'vocab_size': 50257,
    'embed_dim': 90,
    'n_layers': 1,
    'n_heads': 9,                 # 9 heads * 10 = 90
    'hidden_dim': 360,            # Moderate FFN for fair comparison
    'max_seq_len': 128,
    'use_rope': True,             # Match gauge model
    'rope_base': 10000.0,

    'batch_size': 64,
    'use_amp': False,
    'num_workers': 10,
    'epochs': None,
    'max_steps': 15000,
    'warmup_steps': 100,

    'ffn_mode': 'standard',
    'attention_type': 'standard',
    'pos_encoding_mode': 'rope',
    'tie_embeddings': False,

    'evolve_sigma': False,
    'evolve_phi': False,
    'diagonal_covariance': True,
    'isotropic_covariance': False,
    'use_positional_embedding': False,

    'mu_lr': 3e-4,
    'sigma_lr': 0.0001,
    'phi_lr': 0.0001,
    'ffn_lr': 3e-4,

    'alpha': 0,
    'beta': 0,
    'lambda_gamma': 0,
    'kappa_gamma': 1.0,

    'weight_decay': 0.01,
    'dropout': 0.1,
    'grad_clip': 1.0,

    'log_interval': 100,
    'eval_interval': 1000,
    'checkpoint_interval': 25000,
    'patience': 5,

    'kappa_beta': 1.0,
    'attention_pattern': 'full',
    'attention_window': 24,
    'gauge_group': 'SO3',
    'gauge_dim': 3,
    'gauge_mode': 'learned',
    'gauge_fixed_priors': True,
    'irrep_spec': [('ℓ0', 5, 1)],
    'compute_rg_metrics': False,
}


# =============================================================================



def get_git_info() -> Dict[str, str]:
    """Get current git commit info."""
    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()

        branch = subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()

        # Check for uncommitted changes
        status = subprocess.check_output(
            ['git', 'status', '--porcelain'],
            stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()
        dirty = len(status) > 0

        return {
            'commit': commit,
            'branch': branch,
            'dirty': dirty,
        }
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {'commit': 'unknown', 'branch': 'unknown', 'dirty': False}


def get_system_info() -> Dict[str, Any]:
    """Get system/hardware information."""
    info = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        info['cuda_version'] = torch.version.cuda
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['gpu_count'] = torch.cuda.device_count()
        info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9

    return info


def run_test_evaluation(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    vocab_size: int,
    max_samples: int = 128000,
    config: dict = None,
) -> Dict[str, float]:
    """
    Run final evaluation on test set.

    Uses the same code path as validation (compute_free_energy_loss with
    forward_with_attention) to ensure consistent evaluation.

    Args:
        model: Trained model
        test_loader: Test set dataloader
        device: Device to run evaluation on
        vocab_size: Vocabulary size for random baseline comparison
        max_samples: Maximum number of samples to evaluate (default: 128000,
                     equivalent to 2000 batches at batch_size=64). This ensures
                     consistent evaluation across configs with different batch sizes.
        config: Training config dict (for alpha/beta/lambda values).
                If None, uses pure CE evaluation.

    Returns:
        Dictionary with test metrics:
            - test_loss: Cross-entropy loss on test set
            - test_ppl: Perplexity on test set
            - test_bpc: Bits per character
            - random_ppl: Random baseline perplexity
            - improvement: Factor improvement over random
    """
    print("\n" + "="*70)
    print("FINAL TEST SET EVALUATION")
    print("="*70)

    print(f"  Evaluating up to {max_samples} samples...")

    # Pure CE evaluation — disable all VFE regularization terms for test.
    is_standard = isinstance(model, StandardTransformerLM)

    # Target padding uses -100 (PyTorch cross_entropy ignore_index default).
    pad_token_id = -100

    model.eval()
    total_ce_tokens = 0.0  # Sum of CE * non_pad_tokens (for token-weighted avg)
    total_tokens = 0
    num_batches = 0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, (input_ids, target_ids) in enumerate(test_loader):
            if total_samples >= max_samples:
                break

            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            # Count non-padding tokens for proper weighting
            non_pad = (target_ids != pad_token_id).sum().item()

            if is_standard:
                output = model(input_ids, labels=target_ids)
                ce_loss = output['loss'].item()
            else:
                _, metrics = compute_free_energy_loss(
                    model,
                    input_ids,
                    target_ids,
                    alpha=0.0,
                    lambda_beta=0.0,
                    lambda_gamma=0.0,
                    kappa_gamma=1.0,
                    pad_token_id=pad_token_id,
                )
                ce_loss = metrics['loss/ce']

            # Token-weighted accumulation (handles variable-size last batch)
            total_ce_tokens += ce_loss * non_pad
            total_tokens += non_pad
            num_batches += 1
            total_samples += input_ids.size(0)

            # Progress indicator
            if (batch_idx + 1) % 100 == 0:
                print(f"  Evaluated {total_samples}/{max_samples} samples ({num_batches} batches)...")

    # Token-weighted CE average (proper averaging for variable batch sizes)
    test_ce = total_ce_tokens / max(1, total_tokens)
    test_ppl = math.exp(min(test_ce, 20))  # Clamp to prevent overflow
    test_bpc = test_ce / math.log(2)
    random_ppl = vocab_size
    improvement = random_ppl / test_ppl if test_ppl > 0 else 0

    print(f"\nTest Set Results ({total_samples} samples across {num_batches} batches):")
    print(f"  Cross-entropy loss: {test_ce:.4f}")
    print(f"  Perplexity:         {test_ppl:.2f}")
    print(f"  Bits per character: {test_bpc:.3f}")
    print(f"  Random baseline:    {random_ppl:.0f}")
    print(f"  Improvement:        {improvement:.1f}x better than random")
    print("="*70 + "\n")

    model.train()

    return {
        'test_loss': test_ce,
        'test_ppl': test_ppl,
        'test_bpc': test_bpc,
        'random_ppl': random_ppl,
        'improvement': improvement,
    }


def save_experiment_config(
    config: Dict[str, Any],
    ffn_mode: str,
    checkpoint_dir: Path,
    args: argparse.Namespace = None,
) -> Path:
    """
    Save complete experiment configuration to JSON.

    Args:
        config: Model/training configuration dictionary
        ffn_mode: FFN mode being used
        checkpoint_dir: Directory to save config
        args: Command-line arguments (if available)

    Returns:
        Path to saved config file
    """
    experiment_config = {
        # Metadata
        'experiment_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'timestamp': datetime.now().isoformat(),
        'ffn_mode': ffn_mode,

        # Full model/training config
        'config': config,

        # Command-line args (if available)
        'args': vars(args) if args else None,

        # Git info for reproducibility
        'git': get_git_info(),

        # System info
        'system': get_system_info(),
    }

    # Save to checkpoint directory
    config_path = checkpoint_dir / 'experiment_config.json'
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, 'w') as f:
        json.dump(experiment_config, f, indent=2, default=str)

    print(f"📋 Saved experiment config: {config_path}")

    return config_path




class PublicationMetricsTracker:
    """Track ALL metrics needed for publication."""

    def __init__(self, save_path: Path):
        self.save_path = save_path
        self.history = []

        # Create CSV with comprehensive headers
        self.save_path.parent.mkdir(parents=True, exist_ok=True)

        self.headers = [
            # Core
            'step', 'timestamp',

            # Losses
            'train_loss_total', 'train_loss_ce', 'train_loss_belief_align',
            'train_loss_self_consistency', 'train_loss_model_coupling',
            'val_loss', 'val_ce',

            # Metrics
            'train_ppl', 'train_bpc', 'val_ppl', 'val_bpc',

            # Attention stats (crucial for interpretability!)
            'beta_mean', 'beta_std', 'kl_mean', 'kl_std',
            'attention_entropy', 'attention_concentration',

            # RG Metrics (meta-agent emergence!)
            'rg_modularity', 'rg_effective_rank', 'rg_n_clusters',
            'rg_kl_within_mean', 'rg_kl_within_std',
            'rg_kl_between_mean', 'rg_kl_between_std',
            'rg_beta_entropy',

            # Dynamic RG (across VFE iterations)
            'rg_dynamic_n_iterations',
            'rg_dynamic_modularity_init', 'rg_dynamic_modularity_final', 'rg_dynamic_modularity_change',
            'rg_dynamic_rank_init', 'rg_dynamic_rank_final', 'rg_dynamic_rank_change',

            # Learning rates
            'mu_lr', 'sigma_lr', 'phi_lr', 'ffn_lr',

            # Gradient norms
            'grad_norm_total', 'grad_norm_mu', 'grad_norm_ffn',

            # Bayesian alpha diagnostics
            'alpha_mean', 'alpha_std', 'alpha_min', 'alpha_max',
            'alpha_c0', 'alpha_b0', 'alpha_c0_std', 'alpha_b0_std',
            'alpha_mahal_sq_mean', 'alpha_mahal_sq_std',

            # Performance
            'step_time', 'tokens_per_sec',

            # Holonomy (non-flat transport curvature)
            'holonomy_mean_norm', 'holonomy_max_norm',
            'holonomy_frac_gt_01', 'holonomy_spectral_gap', 'holonomy_wilson_trace',

            # Numerical fallback counters
            'num_chol_recover', 'num_chol_fail', 'num_nan_replace', 'num_inv_pinv',
        ]

        with open(self.save_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)

    def log_step(self, step: int, metrics: Dict, lrs: Dict, grad_norms: Dict,
                 step_time: float, batch_size: int, seq_len: int):
        """Log training step with full metrics."""

        # Compute tokens/sec
        tokens_per_sec = (batch_size * seq_len) / step_time if step_time > 0 else 0

        # Bits per character (convert from nats)
        train_bpc = metrics.get('train_loss_ce', 0) / math.log(2)

        entry = {
            'step': step,
            'timestamp': time.time(),

            # Losses
            'train_loss_total': metrics.get('train_loss_total'),
            'train_loss_ce': metrics.get('train_loss_ce'),
            'train_loss_belief_align': metrics.get('train_loss_belief_align', 0),
            'train_loss_self_consistency': metrics.get('train_loss_self_consistency', 0),
            'train_loss_model_coupling': metrics.get('train_loss_model_coupling', 0),
            'val_loss': None,
            'val_ce': None,

            # Metrics
            'train_ppl': metrics.get('train_ppl'),
            'train_bpc': train_bpc,
            'val_ppl': None,
            'val_bpc': None,

            # Attention (crucial for interpretability!)
            'beta_mean': metrics.get('beta_mean'),
            'beta_std': metrics.get('beta_std'),
            'kl_mean': metrics.get('kl_mean'),
            'kl_std': metrics.get('kl_std'),
            'attention_entropy': metrics.get('attention_entropy'),
            'attention_concentration': metrics.get('attention_concentration'),

            # RG Metrics (meta-agent emergence!)
            'rg_modularity': metrics.get('rg/modularity'),
            'rg_effective_rank': metrics.get('rg/effective_rank'),
            'rg_n_clusters': metrics.get('rg/n_clusters'),
            'rg_kl_within_mean': metrics.get('rg/kl_within_mean'),
            'rg_kl_within_std': metrics.get('rg/kl_within_std'),
            'rg_kl_between_mean': metrics.get('rg/kl_between_mean'),
            'rg_kl_between_std': metrics.get('rg/kl_between_std'),
            'rg_beta_entropy': metrics.get('rg/beta_entropy'),

            # Dynamic RG (across VFE iterations)
            'rg_dynamic_n_iterations': metrics.get('rg/dynamic/n_iterations'),
            'rg_dynamic_modularity_init': metrics.get('rg/dynamic/modularity_init'),
            'rg_dynamic_modularity_final': metrics.get('rg/dynamic/modularity_final'),
            'rg_dynamic_modularity_change': metrics.get('rg/dynamic/modularity_change'),
            'rg_dynamic_rank_init': metrics.get('rg/dynamic/rank_init'),
            'rg_dynamic_rank_final': metrics.get('rg/dynamic/rank_final'),
            'rg_dynamic_rank_change': metrics.get('rg/dynamic/rank_change'),

            # Learning rates
            'mu_lr': lrs.get('mu_embed', 0),
            'sigma_lr': lrs.get('sigma_embed', 0),
            'phi_lr': lrs.get('phi_embed', 0),
            'ffn_lr': lrs.get('ffn', 0),

            # Gradients
            'grad_norm_total': grad_norms.get('total', 0) if grad_norms else 0,
            'grad_norm_mu': grad_norms.get('mu', 0) if grad_norms else 0,
            'grad_norm_ffn': grad_norms.get('ffn', 0) if grad_norms else 0,

            # Bayesian alpha diagnostics
            'alpha_mean': metrics.get('bayesian/alpha_mean'),
            'alpha_std': metrics.get('bayesian/alpha_std'),
            'alpha_min': metrics.get('bayesian/alpha_min'),
            'alpha_max': metrics.get('bayesian/alpha_max'),
            'alpha_c0': metrics.get('bayesian/c0'),
            'alpha_b0': metrics.get('bayesian/b0'),
            'alpha_c0_std': metrics.get('bayesian/c0_std'),
            'alpha_b0_std': metrics.get('bayesian/b0_std'),
            'alpha_mahal_sq_mean': metrics.get('bayesian/mahal_sq_mean'),
            'alpha_mahal_sq_std': metrics.get('bayesian/mahal_sq_std'),

            # Performance
            'step_time': step_time,
            'tokens_per_sec': tokens_per_sec,

            # Holonomy
            'holonomy_mean_norm': metrics.get('holonomy/mean_norm'),
            'holonomy_max_norm': metrics.get('holonomy/max_norm'),
            'holonomy_frac_gt_01': metrics.get('holonomy/frac_gt_0.1'),
            'holonomy_spectral_gap': metrics.get('holonomy/spectral_gap'),
            'holonomy_wilson_trace': metrics.get('holonomy/wilson_trace'),

            # Numerical fallback counters
            'num_chol_recover': metrics.get('num/chol_recover', 0),
            'num_chol_fail': metrics.get('num/chol_fail', 0),
            'num_nan_replace': metrics.get('num/nan_replace', 0),
            'num_inv_pinv': metrics.get('num/inv_pinv', 0),
        }

        self.history.append(entry)

    def log_val(self, step: int, val_metrics: Dict):
        """Update entry with validation metrics."""
        for entry in reversed(self.history):
            if entry['step'] == step:
                entry['val_loss'] = val_metrics.get('loss')
                entry['val_ce'] = val_metrics.get('ce_loss', val_metrics.get('loss'))
                entry['val_ppl'] = val_metrics.get('perplexity')
                entry['val_bpc'] = entry['val_ce'] / math.log(2) if entry['val_ce'] else None
                break

    def save(self):
        """Save to CSV."""
        if not self.history:
            return

        with open(self.save_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.headers)
            writer.writeheader()
            writer.writerows(self.history)


class PublicationTrainer(FastTrainer):
    """Enhanced trainer with publication-quality metrics."""

    def __init__(self, *args, publication_metrics: PublicationMetrics = None, tokenizer=None, **kwargs):
        super().__init__(*args, **kwargs)

        # Basic CSV metrics tracker
        metrics_path = self.config.checkpoint_dir / 'metrics.csv'
        self.metrics_tracker = PublicationMetricsTracker(metrics_path)
        print(f"[INFO] Logging publication metrics to: {metrics_path}")

        # Comprehensive publication metrics (optional)
        self.pub_metrics = publication_metrics
        if self.pub_metrics:
            print(f"[INFO] Comprehensive metrics enabled: {self.pub_metrics.experiment_dir}")

        # Tokenizer for decoding sequences in interpretability outputs
        self.tokenizer = tokenizer

        # Track attention visualization count
        self._attention_viz_count = 0

        # =================================================================
        # Gauge geometry: Cartan preconditioning & SL(K) projection
        # =================================================================
        self._cartan_preconditioner = None
        self._slk_trace_vec = None

        use_killing_form = getattr(self.config, 'use_killing_form', False)
        use_slk = getattr(self.config, 'use_slk_projection', False)

        if (use_killing_form or use_slk) and hasattr(self.model, 'generators'):
            from transformer.core.gauge_preconditioner import (
                build_cartan_projector,
                build_slk_projector,
            )
            generators = self.model.generators  # (n_gen, K, K)

            if use_killing_form:
                sym_dampening = getattr(self.config, 'killing_form_sym_dampening', 0.1)
                self._cartan_preconditioner = build_cartan_projector(
                    generators, sym_dampening=sym_dampening
                ).to(self.device)
                print(f"[INFO] Killing form preconditioning enabled (M-step): "
                      f"sym_dampening={sym_dampening} "
                      f"(non-compact directions dampened {1/sym_dampening:.0f}×)")
                # Note: E-step phi preconditioning is now controlled by
                # phi_natural_gradient config ('clip'|'cartan'|'killing'|'pullback')
                # which flows through model → blocks → ffn → VariationalFFNDynamic.

            if use_slk:
                self._slk_trace_vec = build_slk_projector(generators).to(self.device)
                trace_norm = self._slk_trace_vec.norm().item()
                print(f"[INFO] SL(K) projection enabled: "
                      f"removing trace component (||v||={trace_norm:.2f}, "
                      f"n_gen={generators.shape[0]} → {generators.shape[0]-1} effective d.o.f.)")

    def _get_head_irrep_labels(self) -> list:
        """
        Map head indices to irrep types for diagnostic labeling.

        Returns:
            List of strings like "ℓ0", "ℓ1", "ℓ2" for each head.
        """
        irrep_spec = self.config.irrep_spec
        labels = []
        for irrep_name, num_heads, dim in irrep_spec:
            for _ in range(num_heads):
                labels.append(irrep_name)
        return labels

    def _setup_cjk_fonts(self, plt):
        """Configure matplotlib to render CJK (Japanese/Chinese/Korean) characters."""
        if getattr(self, '_cjk_fonts_configured', False):
            return
        import matplotlib.font_manager as fm
        # Try common CJK fonts available on Windows, Linux, and macOS
        cjk_fonts = [
            'MS Gothic', 'Yu Gothic', 'Meiryo',          # Windows Japanese
            'Microsoft YaHei', 'SimHei',                   # Windows Chinese
            'Noto Sans CJK JP', 'Noto Sans JP',           # Linux/cross-platform
            'IPAGothic', 'IPAexGothic', 'TakaoGothic',    # Linux Japanese
            'Hiragino Sans', 'Hiragino Kaku Gothic Pro',   # macOS Japanese
        ]
        available = {f.name for f in fm.fontManager.ttflist}
        for font_name in cjk_fonts:
            if font_name in available:
                plt.rcParams['font.family'] = 'sans-serif'
                plt.rcParams['font.sans-serif'] = [font_name] + plt.rcParams.get('font.sans-serif', [])
                plt.rcParams['axes.unicode_minus'] = False
                self._cjk_fonts_configured = True
                return
        # No CJK font found - titles with Japanese text will show boxes
        # but at least suppress the per-glyph warnings
        import warnings
        warnings.filterwarnings('ignore', message='Glyph .* missing from font')
        self._cjk_fonts_configured = True

    def save_attention_visualization(self, step: int, batch: Tuple[torch.Tensor, torch.Tensor]):
        """
        Save attention pattern visualization for interpretability analysis.

        CRITICAL FIXES (based on visualization analysis):
        1. Save PER-HEAD attention (not averaged - averaging destroys patterns!)
        2. Show WHAT sequence is being visualized (token IDs + decoded text)
        3. Label each head with its irrep type (ℓ0, ℓ1, ℓ2, etc.)
        """
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            from matplotlib.gridspec import GridSpec
            import numpy as np
        except ImportError:
            return  # Skip if matplotlib unavailable

        # Configure CJK font support for Japanese text in plot titles
        dataset_name = getattr(self.train_loader.dataset, 'dataset_name', 'wikitext-2')
        if dataset_name == 'wiki-ja':
            self._setup_cjk_fonts(plt)

        self.model.eval()
        input_ids, target_ids = batch
        input_ids = input_ids.to(self.device)

        # Get attention from forward pass
        with torch.no_grad():
            if hasattr(self.model, 'forward_with_attention'):
                _, attn_info = self.model.forward_with_attention(input_ids, targets=None)
                beta = attn_info.get('beta')
                kl = attn_info.get('kl')
                n_layers = attn_info.get('n_layers', 1)

                if beta is not None:
                    # beta shape: (n_layers, B, n_heads, N, N)
                    # Handle legacy single-layer format for backward compatibility
                    if beta.dim() == 4:
                        # Old format: (B, n_heads, N, N) — wrap in layer dim
                        beta = beta.unsqueeze(0)
                        if kl is not None:
                            kl = kl.unsqueeze(0)
                        n_layers = 1
                    elif beta.dim() == 3:
                        # Very old format: (B, N, N) — no heads, no layers
                        beta = beta.unsqueeze(0).unsqueeze(2)  # (1, B, 1, N, N)
                        if kl is not None:
                            kl = kl.unsqueeze(0).unsqueeze(2)
                        n_layers = 1

                    n_layers_actual, B, n_heads, N, _ = beta.shape

                    # Get irrep labels for each head
                    try:
                        head_labels = self._get_head_irrep_labels()
                    except (AttributeError, KeyError):
                        head_labels = [f"H{i}" for i in range(n_heads)]

                    # Show what sequence we're visualizing
                    if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                        try:
                            decoded = self.tokenizer.decode(input_ids[0].tolist())
                            preview = decoded[:80] + ('...' if len(decoded) > 80 else '')
                            seq_info = f"Step {step}, Text: {preview}"
                        except Exception:
                            seq_info = f"Step {step}, Tokens: {input_ids[0, :20].tolist()}..."
                    else:
                        seq_info = f"Step {step}, Tokens: {input_ids[0, :20].tolist()}..."

                    # Save directory
                    save_dir = self.config.checkpoint_dir / 'attention_patterns'
                    save_dir.mkdir(parents=True, exist_ok=True)

                    # ============================================================
                    # SAVE PER-LAYER, PER-HEAD VISUALIZATIONS (NOT AVERAGED!)
                    # ============================================================
                    for layer_idx in range(n_layers_actual):
                        beta_layer_np = beta[layer_idx, 0].cpu().numpy()  # (n_heads, N, N)

                        for head_idx in range(n_heads):
                            fig, ax = plt.subplots(figsize=(8, 6))

                            attn_head = beta_layer_np[head_idx]  # (N, N)
                            attn_plot = attn_head.copy()
                            #np.fill_diagonal(attn_plot, np.nan)  # Mask diagonal
                            attn_plot = np.log10(np.maximum(attn_plot, 1e-6))  # Log scale

                            im = ax.imshow(attn_plot, cmap='viridis', aspect='auto', vmin=-6, vmax=0)
                            ax.set_xlabel('Key Position (j)')
                            ax.set_ylabel('Query Position (i)')

                            irrep_label = head_labels[head_idx] if head_idx < len(head_labels) else f"H{head_idx}"
                            layer_label = f"L{layer_idx}" if n_layers_actual > 1 else ""
                            title_prefix = f"{layer_label} " if layer_label else ""
                            ax.set_title(
                                f'{title_prefix}Head {head_idx} ({irrep_label}) - {seq_info}',
                                fontsize=10,
                            )
                            plt.colorbar(im, ax=ax, label='log\u2081\u2080(\u03b2)')

                            fig.savefig(
                                save_dir / f'attention_step_{step:06d}_layer{layer_idx}_head{head_idx}.png',
                                dpi=100, bbox_inches='tight',
                            )
                            plt.close(fig)

                    # ============================================================
                    # LOG INFO
                    # ============================================================
                    self._attention_viz_count += 1
                    if self._attention_viz_count == 1:
                        print(f"\n[INFO] Attention patterns saved to: {save_dir}/")
                        print(f"  Saving per-layer, per-head visualizations ({n_layers_actual} layers, {n_heads} heads)")

        self.model.train()

    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Train step with comprehensive metrics and AMP support."""
        self.model.train()

        input_ids, target_ids = batch
        input_ids = input_ids.to(self.device)
        target_ids = target_ids.to(self.device)

        # Check if we should compute RG metrics this step
        # NOTE: Use (step + 1) to align with eval_interval which also uses (step + 1)
        compute_rg = (
            getattr(self.config, 'compute_rg_metrics', False) and
            (self.global_step + 1) % getattr(self.config, 'rg_metrics_interval', 100) == 0
        )

        # Check if using standard transformer (no VFE loss)
        is_standard = isinstance(self.model, StandardTransformerLM)

        # Check if using delta rule for W_out (backprop-free)
        use_delta_rule = getattr(self.config, 'use_delta_rule_w_out', False) and not is_standard

        # If delta rule is enabled, exclude W_out from backprop.
        # When tie_embeddings=True, out_proj.weight IS mu_embed.weight (same tensor),
        # so we must NOT disable requires_grad (it would kill embedding gradients too).
        # Instead, we zero out the out_proj gradient after backward.
        _tied_weights = (use_delta_rule and hasattr(self.model, 'out_proj')
                         and hasattr(self.model, 'token_embed')
                         and hasattr(self.model.token_embed, 'mu_embed')
                         and self.model.out_proj.weight is self.model.token_embed.mu_embed.weight)
        if use_delta_rule and hasattr(self.model, 'out_proj') and not _tied_weights:
            self.model.out_proj.weight.requires_grad = False

        # Beta warmup: ramp lambda_beta from 0 → target over beta_warmup_steps.
        # Prevents uniform attention collapse by letting CE differentiate embeddings
        # before belief coupling gradient (which is uniform when β ≈ 1/N) kicks in.
        target_beta = self.config.beta
        beta_warmup = getattr(self.config, 'beta_warmup_steps', 0)
        if beta_warmup > 0 and self.global_step < beta_warmup:
            effective_beta = target_beta * (self.global_step / beta_warmup)
        else:
            effective_beta = target_beta

        # Forward pass with full metrics (with optional AMP)
        if self.scaler is not None:
            # Mixed precision forward pass
            with torch.amp.autocast('cuda'):
                if is_standard:
                    # Standard transformer: simple cross-entropy loss
                    output = self.model(input_ids, labels=target_ids)
                    loss = output['loss']
                    full_metrics = {
                        'loss/total': loss.item(),
                        'loss/ce': loss.item(),
                    }
                else:
                    loss, full_metrics = compute_free_energy_loss(
                        self.model,
                        input_ids,
                        target_ids,
                        alpha=self.config.alpha,
                        lambda_beta=effective_beta,
                        lambda_gamma=self.config.lambda_gamma,
                        kappa_gamma=self.config.kappa_gamma,
                        lambda_hyper=self.config.lambda_hyper,
                        pad_token_id=self.pad_token_id,
                        use_obs_in_vfe=self.config.use_obs_in_vfe,
                        alpha_phi=self.config.alpha_phi,

                    )
            # Scaled backward
            self.scaler.scale(loss).backward()
        else:
            # Standard forward pass
            if is_standard:
                # Standard transformer: simple cross-entropy loss
                output = self.model(input_ids, labels=target_ids)
                loss = output['loss']
                full_metrics = {
                    'loss/total': loss.item(),
                    'loss/ce': loss.item(),
                }
            else:
                loss, full_metrics = compute_free_energy_loss(
                    self.model,
                    input_ids,
                    target_ids,
                    alpha=self.config.alpha,
                    lambda_beta=effective_beta,
                    lambda_gamma=self.config.lambda_gamma,
                    kappa_gamma=self.config.kappa_gamma,
                    lambda_hyper=self.config.lambda_hyper,
                    pad_token_id=self.pad_token_id,
                    use_obs_in_vfe=self.config.use_obs_in_vfe,
                    alpha_phi=self.config.alpha_phi,
                )
            loss.backward()

        # Tied weights + delta rule: zero out W_out gradient component.
        # With tie_embeddings, out_proj.weight IS mu_embed.weight. In non-amortized
        # mode mu_embed is detached from VFE, so any gradient on this shared tensor
        # comes purely from the output projection (logits = W_out @ mu_q). Zero it
        # so delta rule is the sole W_out update and P-flow is the sole mu update.
        if _tied_weights and self.model.out_proj.weight.grad is not None:
            self.model.out_proj.weight.grad.zero_()

        # =================================================================
        # KILLING FORM PRECONDITIONING (Cartan decomposition)
        # =================================================================
        # Apply BEFORE gradient norm logging and clipping.
        # Dampens the non-compact (symmetric) directions of gl(K) that cause
        # gradient explosions through matrix_exp backward pass.
        # This IS the natural gradient on GL(K) — the Killing form metric
        # assigns higher cost to non-compact directions.
        if self._cartan_preconditioner is not None:
            from transformer.core.gauge_preconditioner import apply_cartan_preconditioning
            for name, param in self.model.named_parameters():
                if param.grad is not None and ('phi_embed' in name or 'phi' in name.lower()):
                    if param.grad.shape[-1] == self._cartan_preconditioner.shape[0]:
                        param.grad.data = apply_cartan_preconditioning(
                            param.grad.data, self._cartan_preconditioner
                        )

        # Compute gradient norms BEFORE clipping
        # Check if this is a log step (need to check global_step here)
        is_log_step = (self.global_step + 1) % self.config.log_interval == 0
        grad_norms = self._compute_gradient_norms() if is_log_step else None

        # Clip and step (with scaler if AMP enabled)
        # Per-group clipping for large gauge groups (SO(N>3)):
        # phi_embed gradients dominate global norm, starving mu/sigma.
        # FastTrainingConfig always uses param groups; TrainingConfig has use_param_groups flag.
        _use_param_groups = getattr(self.config, 'use_param_groups', True)
        if self.scaler is not None:
            if self.config.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                if _use_param_groups:
                    for group in self.optimizer.param_groups:
                        if group['params']:
                            torch.nn.utils.clip_grad_norm_(
                                group['params'],
                                self.config.grad_clip,
                            )
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.grad_clip,
                    )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            if self.config.grad_clip > 0:
                if _use_param_groups:
                    for group in self.optimizer.param_groups:
                        if group['params']:
                            torch.nn.utils.clip_grad_norm_(
                                group['params'],
                                self.config.grad_clip,
                            )
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.grad_clip,
                    )
            self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()
        self.optimizer.zero_grad()

        # =================================================================
        # SL(K) PROJECTION: Remove trace component from phi_embed
        # =================================================================
        # Projects φ to the traceless subalgebra sl(K), ensuring
        # det(Ω_ij) = exp(tr(M_i - M_j)) = exp(0) = 1.
        # This removes the single most dangerous non-compact degree of
        # freedom (uniform scaling) without restricting rotations or shears.
        if self._slk_trace_vec is not None:
            from transformer.core.gauge_preconditioner import apply_slk_projection
            if hasattr(self.model, 'token_embed') and hasattr(self.model.token_embed, 'phi_embed'):
                with torch.no_grad():
                    phi_weight = self.model.token_embed.phi_embed.weight
                    phi_weight.data = apply_slk_projection(
                        phi_weight.data, self._slk_trace_vec
                    )
            # Also project PriorBank phi_embed if present
            if hasattr(self.model, 'prior_bank') and self.model.prior_bank is not None:
                if hasattr(self.model.prior_bank, 'phi_embed'):
                    with torch.no_grad():
                        pb_phi = self.model.prior_bank.phi_embed.weight
                        pb_phi.data = apply_slk_projection(
                            pb_phi.data, self._slk_trace_vec
                        )

        # Re-enable requires_grad for W_out if it was disabled (non-tied case)
        if use_delta_rule and hasattr(self.model, 'out_proj') and not _tied_weights:
            self.model.out_proj.weight.requires_grad = True

        # =================================================================
        # P-FLOW: EMA update of token embeddings toward successful beliefs
        # =================================================================
        # This is the key learning mechanism from fep_transformer.py
        # After backprop updates W_out, P-flow updates token embeddings
        # toward beliefs that predicted successfully (low CE)
        use_p_flow = getattr(self.config, 'use_p_flow', False)
        if use_p_flow and not is_standard and 'p_flow/mu_q' in full_metrics:
            mu_beliefs = full_metrics['p_flow/mu_q']
            ce_per_position = full_metrics['p_flow/ce_per_position']
            ema_decay = getattr(self.config, 'p_flow_ema_decay', 0.99)

            # Call P-flow update on the model (mu + sigma)
            sigma_beliefs = full_metrics.get('p_flow/sigma_q')
            if hasattr(self.model, 'p_flow_update'):
                self.model.p_flow_update(
                    token_ids=input_ids,
                    mu_beliefs=mu_beliefs,
                    prediction_errors=ce_per_position,
                    ema_decay=ema_decay,
                    sigma_beliefs=sigma_beliefs,
                    pad_token_id=self.pad_token_id,
                )

            # Phi P-flow: update gauge frames toward VFE-evolved values
            # Only when detach_phi=True (phi is detached from backprop)
            if (getattr(self.config, 'detach_phi', False) and
                    'p_flow/phi_evolved' in full_metrics and
                    hasattr(self.model, 'phi_flow_update')):
                self.model.phi_flow_update(
                    token_ids=input_ids,
                    phi_evolved=full_metrics['p_flow/phi_evolved'],
                    prediction_errors=ce_per_position,
                    ema_decay=ema_decay,
                    pad_token_id=self.pad_token_id,
                )

        # =================================================================
        # DELTA RULE: Backprop-free update of W_out
        # =================================================================
        # Uses local learning rule: ΔW = η · (target - prediction) ⊗ μ^T
        # Combined with P-flow + detach_phi, this makes learning fully backprop-free.
        if use_delta_rule and 'p_flow/mu_q' in full_metrics:
            mu_beliefs = full_metrics['p_flow/mu_q']
            delta_lr = getattr(self.config, 'delta_rule_lr', 0.1)

            # Call delta rule update on the model
            if hasattr(self.model, 'delta_rule_update_w_out'):
                self.model.delta_rule_update_w_out(
                    mu_beliefs=mu_beliefs,
                    targets=target_ids,
                    lr=delta_lr,
                    pad_token_id=self.pad_token_id,
                )

        # Format comprehensive metrics
        metrics = {
            'train_loss_total': full_metrics['loss/total'],
            'train_loss_ce': full_metrics['loss/ce'],
            'train_loss_belief_align': full_metrics.get('loss/belief_align', 0),
            'train_loss_self_consistency': full_metrics.get('loss/self_consistency', 0),
            'train_loss_model_coupling': full_metrics.get('loss/model_coupling', 0),
            'train_ppl': math.exp(min(full_metrics['loss/ce'], 20)),  # Clamp to prevent overflow
            'beta_mean': full_metrics.get('attention/beta_mean', 0),
            'beta_std': 0,  # Could compute if needed
            'kl_mean': full_metrics.get('attention/kl_mean', 0),
            'kl_std': 0,
            # Crucial attention interpretability metrics
            'attention_entropy': full_metrics.get('attention/entropy', 0),
            'attention_concentration': full_metrics.get('attention/concentration', 0),
            'schedule/effective_beta': effective_beta,
        }

        # Carry over Bayesian alpha diagnostics
        for key in ['bayesian/alpha_mean', 'bayesian/alpha_std', 'bayesian/alpha_min',
                     'bayesian/alpha_max', 'bayesian/c0', 'bayesian/b0',
                     'bayesian/c0_std', 'bayesian/b0_std',
                     'bayesian/mahal_sq_mean', 'bayesian/mahal_sq_std']:
            if key in full_metrics:
                metrics[key] = full_metrics[key]

        # Compute RG metrics if enabled and attention info was returned
        if compute_rg and 'attention_info' in full_metrics:
            rg_metrics = compute_rg_metrics_from_attention(
                attn_info=full_metrics['attention_info'],
                step=self.global_step,
                auto_cluster=getattr(self.config, 'rg_auto_cluster', True),
                n_clusters=getattr(self.config, 'rg_n_clusters', None),
            )
            # Add RG metrics with proper key mapping for CSV
            for key, value in rg_metrics.items():
                metrics[key] = value

            # Dynamic RG tracking (across VFE iterations within forward pass)
            track_dynamic = getattr(self.config, 'track_dynamic_rg', False)
            if track_dynamic and hasattr(self.model, 'forward_with_rg_tracking'):
                try:
                    # Run a separate forward pass with RG tracking
                    # This captures beta_history across VFE iterations
                    with torch.no_grad():
                        _, rg_info = self.model.forward_with_rg_tracking(
                            token_ids=input_ids,
                            targets=target_ids,
                        )
                    dynamic_metrics = compute_dynamic_rg_metrics(rg_info, self.global_step)

                    # Add dynamic RG metrics
                    for key, value in dynamic_metrics.items():
                        metrics[key] = value
                except Exception as e:
                    # Don't crash training on RG tracking errors
                    print(f"[WARNING] Dynamic RG tracking failed: {e}")

        return metrics, grad_norms

    def _compute_gradient_norms(self) -> Dict[str, float]:
        """Compute gradient norms for different parameter groups."""
        norms = {'total': 0, 'mu': 0, 'sigma': 0, 'phi': 0, 'ffn': 0}

        total_norm = 0
        mu_norm = 0
        sigma_norm = 0
        phi_norm = 0
        ffn_norm = 0

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm ** 2

                if 'mu_embed' in name or 'mu' in name.lower():
                    mu_norm += param_norm ** 2
                elif 'sigma_embed' in name or 'sigma' in name.lower() or 'L_embed' in name:
                    sigma_norm += param_norm ** 2
                elif 'phi_embed' in name or 'phi' in name.lower():
                    phi_norm += param_norm ** 2
                elif 'ffn' in name:
                    ffn_norm += param_norm ** 2

        norms['total'] = math.sqrt(total_norm)
        norms['mu'] = math.sqrt(mu_norm)
        norms['sigma'] = math.sqrt(sigma_norm)
        norms['phi'] = math.sqrt(phi_norm)
        norms['ffn'] = math.sqrt(ffn_norm)

        return norms


    def sample_text(
        self,
        prompt: str = "The",
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 40,
    ) -> str:
        """
        Generate text to verify the model is learning.

        Args:
            prompt: Starting text
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (lower = more deterministic)
            top_k: Top-k sampling

        Returns:
            Generated text string
        """
        self.model.eval()

        # Get dataset which has encode/decode methods
        dataset = self.train_loader.dataset

        # Encode prompt using dataset's method
        prompt_ids = dataset.encode(prompt)
        prompt_tensor = torch.tensor([prompt_ids], device=self.device)

        # Generate
        with torch.no_grad():
            generated = self.model.generate(
                prompt_ids=prompt_tensor,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )

        # Decode using dataset's method
        generated_text = dataset.decode(generated[0])

        self.model.train()
        return generated_text

    def train(self):
        """Training loop with publication metrics."""
        print(f"{'='*70}")
        print("PUBLICATION-QUALITY TRAINING")
        print(f"{'='*70}\n")

        # Support resuming from a checkpoint
        start_step = self.global_step
        if start_step > 0:
            print(f"  Resuming from step {start_step}")

        

        start_time = time.time()
        train_iterator = iter(self.train_loader)

        # Calculate total steps: epochs takes precedence over max_steps
        epochs = getattr(self.config, 'epochs', None)
        if epochs is not None and epochs > 0:
            steps_per_epoch = len(self.train_loader)
            total_steps = epochs * steps_per_epoch
            print(f"  Training for {epochs} epoch(s) ({steps_per_epoch} steps/epoch = {total_steps:,} total steps)")
        else:
            total_steps = self.config.max_steps
            steps_per_epoch = len(self.train_loader)
            equiv_epochs = total_steps / steps_per_epoch if steps_per_epoch > 0 else 0
            print(f"  Training for {total_steps:,} steps (~{equiv_epochs:.1f} epochs)")

        try:
            from tqdm import tqdm
            pbar = tqdm(
                range(start_step, total_steps),
                desc="Training",
                initial=start_step,
                total=total_steps
            )
            use_tqdm = True
        except ImportError:
            pbar = range(start_step, total_steps)
            use_tqdm = False

        # Run initial gauge frame semantic analysis (only if starting fresh)
        if start_step == 0 and self.pub_metrics:
            try:
                print("[Semantic] Running initial analysis (step 0)...")
                self.pub_metrics.run_semantic_analysis(
                    model=self.model,
                    step=0,
                    verbose=True,
                )
            except Exception as e:
                print(f"[WARN] Initial semantic analysis failed: {e}")

        for step in pbar:
            self.global_step = step
            step_start = time.time()

            # Get batch
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(self.train_loader)
                batch = next(train_iterator)

            # Train step with full metrics (grad_norms computed inside before zero_grad)
            metrics, grad_norms = self.train_step(batch)

            step_time = time.time() - step_start

            is_log_step = (step + 1) % self.config.log_interval == 0
            has_rg = metrics.get('rg/modularity') is not None

            # Get learning rates
            lrs = {group['name']: group['lr'] for group in self.optimizer.param_groups}

            # Log to basic tracker and console at log intervals OR when RG metrics were computed
            if is_log_step or has_rg:
                # Flush numerical fallback counters and inject into metrics
                _num_events = _flush_numerical_events()
                for _nk, _nv in _num_events.items():
                    metrics[f'num/{_nk}'] = _nv

                batch_size = batch[0].shape[0]
                seq_len = batch[0].shape[1]
                self.metrics_tracker.log_step(
                    step + 1, metrics, lrs, grad_norms, step_time, batch_size, seq_len
                )

                # Log to comprehensive publication metrics (if enabled)
                if self.pub_metrics:
                    self.pub_metrics.record_training_step(
                        step=step + 1,
                        epoch=(step + 1) / len(self.train_loader),
                        train_metrics={
                            'loss': metrics['train_loss_total'],
                            'ce_loss': metrics['train_loss_ce'],
                            'attention_entropy': metrics.get('attention_entropy', 0),
                            'attention_concentration': metrics.get('attention_concentration', 0),
                        },
                        diagnostics=None,
                        grad_norms=grad_norms,
                        lrs=lrs,
                        step_time=step_time,
                        batch_size=batch_size,
                        seq_len=seq_len,
                    )

                # Console logging
                log_msg = (
                    f"Step {step+1}/{total_steps} | "
                    f"Loss: {metrics['train_loss_total']:.4f} | "
                    f"CE: {metrics['train_loss_ce']:.4f} | "
                    f"β: {metrics['train_loss_belief_align']:.4f} | "
                    f"PPL: {metrics['train_ppl']:.1f}"
                )

                # RG metrics console output
                _rg_msg = None
                if has_rg:
                    _rg_msg = (
                        f"  [RG] Q={metrics['rg/modularity']:.4f} | "
                        f"rank={metrics['rg/effective_rank']:.1f} | "
                        f"clusters={metrics['rg/n_clusters']} | "
                        f"H={metrics['rg/beta_entropy']:.3f}"
                    )
                    if metrics.get('rg/dynamic/n_iterations') is not None and metrics['rg/dynamic/n_iterations'] > 1:
                        _rg_msg += (
                            f" | dyn({metrics['rg/dynamic/n_iterations']}it): "
                            f"Q {metrics.get('rg/dynamic/modularity_init', 0):.3f}->{metrics.get('rg/dynamic/modularity_final', 0):.3f}"
                        )

                if use_tqdm:
                    pbar.set_description(log_msg)
                    # Print gradient norms using tqdm.write for proper display
                    if grad_norms:
                        tqdm.write(f"  [GRAD] total: {grad_norms['total']:.3e} | "
                                   f"mu: {grad_norms['mu']:.3e} | sigma: {grad_norms['sigma']:.3e} | "
                                   f"phi: {grad_norms['phi']:.3e}\n\n")
                    # Print Bayesian alpha diagnostics
                    if metrics.get('bayesian/alpha_mean') is not None:
                        tqdm.write(f"  [ALPHA] mean: {metrics['bayesian/alpha_mean']:.4f} | "
                                   f"std: {metrics['bayesian/alpha_std']:.4f} | "
                                   f"range: [{metrics['bayesian/alpha_min']:.4f}, {metrics['bayesian/alpha_max']:.4f}] | "
                                   f"c0: {metrics['bayesian/c0']:.4f}±{metrics.get('bayesian/c0_std', 0):.4f} | "
                                   f"b0: {metrics['bayesian/b0']:.4f}±{metrics.get('bayesian/b0_std', 0):.4f} | "
                                   f"mahal: {metrics['bayesian/mahal_sq_mean']:.4f}")
                    if _rg_msg:
                        tqdm.write(_rg_msg)
                else:
                    print(log_msg)
                    if grad_norms:
                        print(f"  [GRAD] total: {grad_norms['total']:.3e} | "
                              f"mu: {grad_norms['mu']:.3e} | sigma: {grad_norms['sigma']:.3e} | "
                              f"phi: {grad_norms['phi']:.3e}\n\n")
                    if metrics.get('bayesian/alpha_mean') is not None:
                        print(f"  [ALPHA] mean: {metrics['bayesian/alpha_mean']:.4f} | "
                              f"std: {metrics['bayesian/alpha_std']:.4f} | "
                              f"range: [{metrics['bayesian/alpha_min']:.4f}, {metrics['bayesian/alpha_max']:.4f}] | "
                              f"c0: {metrics['bayesian/c0']:.4f}±{metrics.get('bayesian/c0_std', 0):.4f} | "
                              f"b0: {metrics['bayesian/b0']:.4f}±{metrics.get('bayesian/b0_std', 0):.4f} | "
                              f"mahal: {metrics['bayesian/mahal_sq_mean']:.4f}\n\n")
                    if _rg_msg:
                        print(_rg_msg)

                # Report numerical fallback counters if any fired
                if _num_events:
                    _num_msg = "  [NUM] " + " | ".join(
                        f"{k}: {v}" for k, v in sorted(_num_events.items())
                    )
                    if use_tqdm:
                        tqdm.write(_num_msg)
                    else:
                        print(_num_msg)

            # Validation
            if (step + 1) % self.config.eval_interval == 0:
                val_metrics = self.validate()
                self.metrics_tracker.log_val(step + 1, val_metrics)

                # Log to comprehensive metrics
                if self.pub_metrics:
                    self.pub_metrics.record_validation(step + 1, val_metrics)

                # Log attention entropy/concentration for interpretability
                attn_entropy = metrics.get('attention_entropy', 0)
                attn_concentration = metrics.get('attention_concentration', 0)

                print(f"\n  Validation @ step {step+1}:")
                print(f"    Loss: {val_metrics['loss']:.4f}")
                print(f"    CE: {val_metrics['ce_loss']:.4f}")
                print(f"    PPL: {val_metrics['perplexity']:.2f}")
                print(f"    BPC: {val_metrics['ce_loss']/math.log(2):.3f}")
                print(f"    Attn entropy: {attn_entropy:.3f} | concentration: {attn_concentration:.3f}\n\n")

                # Log RG metrics if available (meta-agent emergence!)
                if metrics.get('rg/modularity') is not None:
                    print(f"    RG Metrics (meta-agent emergence):")
                    print(f"      Modularity Q: {metrics['rg/modularity']:.4f} (higher = more structure)")
                    print(f"      Effective rank: {metrics['rg/effective_rank']:.2f} (lower = concentrated)")
                    print(f"      Clusters (meta-agents): {metrics['rg/n_clusters']}")
                    print(f"      KL within: {metrics['rg/kl_within_mean']:.4f} (lower = tighter)")
                    print(f"      KL between: {metrics['rg/kl_between_mean']:.4f}\n\n")

                    # Dynamic RG flow (within forward pass)
                    if metrics.get('rg/dynamic/n_iterations') is not None:
                        n_iters = metrics['rg/dynamic/n_iterations']
                        if n_iters > 1:
                            mod_change = metrics.get('rg/dynamic/modularity_change', 0)
                            rank_change = metrics.get('rg/dynamic/rank_change', 0)
                            print(f"    Dynamic RG ({n_iters} VFE iterations):")
                            print(f"      Modularity: {metrics.get('rg/dynamic/modularity_init', 0):.4f} → {metrics.get('rg/dynamic/modularity_final', 0):.4f} (Δ={mod_change:+.4f})")
                            print(f"      Eff. Rank:  {metrics.get('rg/dynamic/rank_init', 0):.1f} → {metrics.get('rg/dynamic/rank_final', 0):.1f} (Δ={rank_change:+.1f})")

                # Generate sample text to verify learning (varied prompts for diversity)
                try:
                    import random
                    # Use language-appropriate prompts
                    dataset_name = getattr(self.train_loader.dataset, 'dataset_name', 'wikitext-2')
                    if dataset_name == 'wiki-ja':
                        prompts = ["日本", "東京", "世界", "歴史", "文化", "科学", "政治", "経済", "教育", "自然",
                                   "社会", "技術", "音楽", "映画", "大学"]
                    else:
                        prompts = ["The", "In", "A", "It", "This", "As", "One", "When", "For",
                                   "After", "Before", "During", "While", "Although", "However"]
                    prompt = random.choice(prompts)
                    # Use temperature 0.9 and lower top_k for more diversity
                    sample = self.sample_text(prompt=prompt, max_new_tokens=30, temperature=0.9, top_k=30)
                    print(f"    Sample: {sample[:100]}...")
                except Exception as e:
                    import traceback
                    print(f"    Sample generation failed: {e}")
                    traceback.print_exc()
                print()

                # Save attention visualization periodically
                try:
                    sample_batch = next(iter(self.val_loader))
                    self.save_attention_visualization(step + 1, sample_batch)
                except StopIteration:
                    pass

                # Save best model based on CE loss (not total loss)
                # CE loss is the proper metric since PPL = exp(CE)
                if val_metrics['ce_loss'] < self.best_val_ce:
                    self.best_val_ce = val_metrics['ce_loss']
                    self.save_checkpoint(is_best=True)

            # Checkpointing
            if (step + 1) % self.config.checkpoint_interval == 0:
                self.save_checkpoint(is_best=False)
                self.metrics_tracker.save()

            # Periodic holonomy diagnostics (non-flat transport curvature)
            if self.pub_metrics and self.pub_metrics.should_compute_holonomy(step + 1):
                try:
                    holonomy_dict = self.pub_metrics.compute_holonomy_diagnostics(
                        model=self.model,
                        step=step + 1,
                        verbose=True,
                    )
                    if holonomy_dict:
                        # Merge holonomy metrics into the current step's CSV entry
                        merged = False
                        for entry in reversed(self.metrics_tracker.history):
                            if entry['step'] == step + 1:
                                entry['holonomy_mean_norm'] = holonomy_dict.get('holonomy/mean_norm')
                                entry['holonomy_max_norm'] = holonomy_dict.get('holonomy/max_norm')
                                entry['holonomy_frac_gt_01'] = holonomy_dict.get('holonomy/frac_gt_0.1')
                                entry['holonomy_spectral_gap'] = holonomy_dict.get('holonomy/spectral_gap')
                                entry['holonomy_wilson_trace'] = holonomy_dict.get('holonomy/wilson_trace')
                                merged = True
                                break
                        if not merged:
                            print(f"[WARN] Holonomy at step {step+1}: no matching CSV entry "
                                  f"(holonomy_interval may not be divisible by log_interval)")
                except Exception as e:
                    print(f"[WARN] Holonomy computation failed at step {step+1}: {e}")

            # Periodic gauge frame semantic analysis
            if self.pub_metrics and self.pub_metrics.should_run_semantic_analysis(step + 1):
                try:
                    self.pub_metrics.run_semantic_analysis(
                        model=self.model,
                        step=step + 1,
                        verbose=False,  # Minimal output during training
                    )
                except Exception as e:
                    print(f"[WARN] Semantic analysis failed at step {step+1}: {e}")

        # Flush any remaining numerical events accumulated after the last log step
        _final_num_events = _flush_numerical_events()
        if _final_num_events:
            print("  [NUM] Final: " + " | ".join(
                f"{k}: {v}" for k, v in sorted(_final_num_events.items())
            ))

        # Save final metrics
        self.metrics_tracker.save()
        print(f"\n[INFO] Final metrics saved to: {self.metrics_tracker.save_path}")

        # Save comprehensive publication metrics
        if self.pub_metrics:
            self.pub_metrics.save_all()
            self.pub_metrics.generate_all_figures()

            # Run final gauge frame semantic analysis
            try:
                self.pub_metrics.run_final_semantic_analysis(
                    model=self.model,
                    verbose=True,
                )
            except Exception as e:
                print(f"[WARN] Final semantic analysis failed: {e}")

            # Generate final holonomy figures (non-flat transport)
            if self.pub_metrics.holonomy_history:
                try:
                    print("\n[PublicationMetrics] Generating holonomy figures...")
                    self.pub_metrics.generate_holonomy_figures(
                        model=self.model,
                        save_prefix='holonomy',
                    )
                except Exception as e:
                    print(f"[WARN] Final holonomy figure generation failed: {e}")

            # Generate interpretability outputs using a sample batch from validation
            try:
                sample_batch = next(iter(self.val_loader))
                self.pub_metrics.generate_interpretability_outputs(
                    model=self.model,
                    sample_batch=sample_batch,
                    tokenizer=self.tokenizer,  # Dataset with .decode() method
                    device=self.device,
                )
            except Exception as e:
                import traceback
                print(f"[WARNING] Could not generate interpretability outputs: {e}")
                print(f"  Traceback: {traceback.format_exc()}")

            self.pub_metrics.print_summary()

        # Summary
        elapsed = time.time() - start_time
        print(f"\n{'='*70}")
        print("TRAINING COMPLETE!")
        print(f"{'='*70}")
        print(f"Time: {elapsed/3600:.2f} hours")
        print(f"Best val CE: {self.best_val_ce:.4f} (PPL: {math.exp(min(self.best_val_ce, 20.0)):.2f})")
        print(f"{'='*70}\n")


def run_single_experiment(
    config: dict,
    ffn_mode: str,
    device: torch.device,
    checkpoint_dir: Path,
    use_wandb: bool = False,
    args: argparse.Namespace = None,
    enable_publication_metrics: bool = True,
) -> Dict:
    """
    Run a single training experiment.

    Args:
        config: Configuration dictionary
        ffn_mode: FFN mode ('VFE_dynamic')
        device: Device to train on
        checkpoint_dir: Directory to save checkpoints
        use_wandb: Whether to use Weights & Biases logging
        args: Command-line arguments for logging
        enable_publication_metrics: Whether to enable comprehensive publication metrics

    Returns:
        Dictionary with final metrics
    """
    print("\n" + "="*70)
    print(f"EXPERIMENT: FFN_MODE = {ffn_mode}")
    print("="*70)

    # Override FFN mode in config
    config = config.copy()
    config['ffn_mode'] = ffn_mode

    # Create experiment-specific checkpoint directory
    exp_checkpoint_dir = checkpoint_dir / f"ffn_{ffn_mode}"
    exp_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save experiment configuration at the START
    save_experiment_config(config, ffn_mode, exp_checkpoint_dir, args)

    # =================================================================
    # Data Loading (BPE tokenization using GPT-2 tokenizer)
    # =================================================================

    dataset_name = config.get('dataset', 'wikitext-2')
    print("\n" + "="*70)
    print(f"LOADING {dataset_name.upper()} DATA")
    print("="*70)

    # Tokenizer selection: 'char', 'bpe', or 'auto' (default)
    # 'auto' uses char for vocab_size <= 256, bpe otherwise
    tokenizer_mode = config.get('tokenizer', 'auto')
    if tokenizer_mode == 'auto':
        use_char = config['vocab_size'] <= 256
    else:
        use_char = (tokenizer_mode == 'char')

    test_loader = None  # Will be set if available
    if use_char:
        print(f"Using CHARACTER-LEVEL tokenizer (vocab_size={config['vocab_size']})")
        # Note: create_char_dataloaders doesn't support test set yet
        train_loader, val_loader, actual_vocab_size = create_char_dataloaders(
            max_seq_len=config['max_seq_len'],
            batch_size=config['batch_size'],
            num_workers=config.get('num_workers', 0),
        )
        tokenizer = None  # Character-level doesn't need tokenizer for decode
    else:
        print(f"Using BPE tokenizer (vocab_size={config['vocab_size']})")
        train_loader, val_loader, test_loader, actual_vocab_size, tokenizer = create_dataloaders(
            max_seq_len=config['max_seq_len'],
            batch_size=config['batch_size'],
            vocab_size=config['vocab_size'],  # Top K BPE tokens
            num_workers=config.get('num_workers', 0),
            dataset=dataset_name,
            include_test=True,  # Include test set for final evaluation
            return_tokenizer=True,  # Get tokenizer for interpretability outputs
        )

    config['vocab_size'] = actual_vocab_size

    # =================================================================
    # Model Creation - Three distinct modes
    # =================================================================

    print("\n" + "="*70)
    print("CREATING MODEL")
    print("="*70)
    print(f"  N (seq len): {config['max_seq_len']}")
    print(f"  K (embed): {config['embed_dim']}")
    print(f"  Layers: {config['n_layers']}")
    print(f"  Vocab: {actual_vocab_size} ({'char' if use_char else 'BPE'})")

    # =====================================================================
    # MODE 1: STANDARD TRANSFORMER (baseline)
    # =====================================================================
    if ffn_mode == 'standard':
        print("  Model type: STANDARD TRANSFORMER (dot-product attention)")
        print("  - Attention: Q·K softmax")
        print("  - FFN: Learned MLP (GELU)")
        print("  - Output: Linear projection")
        print("  - Learning: Backprop")
        model_config = {
            'vocab_size': actual_vocab_size,
            'embed_dim': config['embed_dim'],
            'n_layers': config['n_layers'],
            'n_heads': config.get('n_heads', 1),
            'hidden_dim': config.get('hidden_dim', config['embed_dim'] * 4),
            'max_seq_len': config['max_seq_len'],
            'dropout': config.get('dropout', 0.1),
            # Baseline ablation options (peer review M2b, M2c)
            'disable_ffn': config.get('disable_ffn', False),
            'use_rope': config.get('use_rope', False),
            'rope_base': config.get('rope_base', 10000.0),
            'tie_embeddings': config.get('tie_embeddings', True),
            'no_pos_encoding': config.get('use_positional_embedding', True) is False and not config.get('use_rope', False),
        }
        model = StandardTransformerLM(model_config)

    # =====================================================================
    # MODE 2: PURE FEP TRANSFORMER (most principled)
    # =====================================================================
    # MODE 2: VFE_DYNAMIC TRANSFORMER (EM-step, uses backprop)
    # =====================================================================
    else:
        print("  Model type: GAUGE VFE TRANSFORMER (KL-divergence attention)")
        print("  - Attention: KL-divergence based")
        print("  - FFN: VFE EM-step dynamics")
        print("  - Output: Linear projection")
        print("  - Learning: Backprop")
        print("  - Position: None (emergent)")

        # kappa_beta: scalar sharpness dial (dimension scaling τ=2√K is hardcoded in attention)
        if 'kappa_beta' not in config:
            config['kappa_beta'] = 1.0
        print(f"  kappa_beta: {config['kappa_beta']}")

        model = GaugeTransformerLM(config)

    model = model.to(device)

    # Get parameter counts
    if hasattr(model, 'get_num_params'):
        total_params = model.get_num_params(non_embedding=False)
        non_embed_params = model.get_num_params(non_embedding=True)
    else:
        total_params = sum(p.numel() for p in model.parameters())
        non_embed_params = sum(p.numel() for name, p in model.named_parameters() if 'embed' not in name)

    print(f"\nModel Parameters:")
    print(f"  Total:         {total_params:,}")
    print(f"  Non-embedding: {non_embed_params:,}")
    print(f"  Embedding:     {total_params - non_embed_params:,}")

    # =================================================================
    # FLOPs Estimation (Peer Review M2e)
    # =================================================================
    seq_len = config['max_seq_len']
    batch_size = config['batch_size']
    max_steps = config['max_steps']

    is_standard = isinstance(model, StandardTransformerLM)
    if is_standard:
        flops_result = count_standard_transformer_flops(
            vocab_size=config['vocab_size'],
            embed_dim=config['embed_dim'],
            n_layers=config['n_layers'],
            n_heads=config.get('n_heads', 1),
            hidden_dim=config.get('hidden_dim', config['embed_dim'] * 4),
            seq_len=seq_len,
            batch_size=batch_size,
            disable_ffn=config.get('disable_ffn', False),
            tie_embeddings=config.get('tie_embeddings', False),
        )
    else:
        gauge_irrep = config.get('irrep_spec', [('fund', 1, config['embed_dim'])])
        n_heads_g = gauge_irrep[0][1] if gauge_irrep else 1
        head_dim_g = gauge_irrep[0][2] if gauge_irrep else config['embed_dim']
        phi_dim = config['embed_dim'] * config['embed_dim']  # GL(K) default
        flops_result = count_gauge_transformer_flops(
            vocab_size=config['vocab_size'],
            embed_dim=config['embed_dim'],
            n_layers=config['n_layers'],
            n_heads=n_heads_g,
            head_dim=head_dim_g,
            seq_len=seq_len,
            batch_size=batch_size,
            phi_dim=phi_dim,
            ffn_n_iterations=config.get('ffn_n_iterations', 1),
            use_rope=config.get('use_rope', False),
            diagonal_covariance=config.get('diagonal_covariance', True),
        )

    step_flops = flops_result['step_total']
    total_flops = step_flops * max_steps
    print(f"\nFLOPs Estimation (Peer Review M2e):")
    print(f"  FLOPs/step:     {format_flops(step_flops)}")
    print(f"  Total training: {format_flops(total_flops)}")
    if not is_standard:
        print(f"  Key costs: mat_exp={format_flops(flops_result.get('transport_mat_exp', 0))}/layer, "
              f"KL={format_flops(flops_result.get('kl_divergence', 0))}/layer")

    # =================================================================
    # Training Configuration
    # =================================================================

    train_config = FastTrainingConfig(
        epochs=config.get('epochs', None),
        max_steps=config['max_steps'],
        warmup_steps=config['warmup_steps'],

        # Learning rates
        # For standard transformer: attention_lr should match ffn_lr (all standard Adam)
        # For gauge transformer: attention_lr matches phi_lr (natural gradient scale)
        mu_lr=config['mu_lr'],
        sigma_lr=config['sigma_lr'],
        phi_lr=config['phi_lr'],
        attention_lr=config.get('attention_lr', config['ffn_lr'] if ffn_mode == 'standard' else config['phi_lr']),
        ffn_lr=config['ffn_lr'],
        output_lr=config['ffn_lr'],

        weight_decay=config['weight_decay'],
        grad_clip=config['grad_clip'],

        # Free energy loss weights
        alpha=config['alpha'],
        beta=config['beta'],             # → lambda_beta in compute_free_energy_loss
        lambda_gamma=config['lambda_gamma'],
        lambda_hyper=config.get('lambda_hyper', 0.0),
        use_obs_in_vfe=config.get('use_obs_in_vfe', False),

        # Gauge geometry: phi gradient control
        alpha_phi=config.get('alpha_phi', 0.0),
        use_slk_projection=config.get('use_slk_projection', False),
        use_killing_form=config.get('use_killing_form', False),
        killing_form_sym_dampening=config.get('killing_form_sym_dampening', 0.1),

        log_interval=config['log_interval'],
        eval_interval=config['eval_interval'],
        checkpoint_interval=config['checkpoint_interval'],

        use_wandb=use_wandb,
        checkpoint_dir=exp_checkpoint_dir,

        # P-FLOW: EMA update of token embeddings toward successful beliefs
        use_p_flow=config.get('use_p_flow', False),
        p_flow_ema_decay=config.get('p_flow_ema_decay', 0.99),

        # DELTA RULE: Backprop-free learning for W_out
        use_delta_rule_w_out=config.get('use_delta_rule_w_out', False),
        delta_rule_lr=config.get('delta_rule_lr', 0.001),

        # RG METRICS: Track renormalization group flow
        compute_rg_metrics=config.get('compute_rg_metrics', False),
        rg_metrics_interval=config.get('rg_metrics_interval', 100),
        rg_auto_cluster=config.get('rg_auto_cluster', True),
        rg_n_clusters=config.get('rg_n_clusters', None),
        track_dynamic_rg=config.get('track_dynamic_rg', False),
    )

    print("\n" + "="*70)
    print("TRAINING CONFIGURATION")
    print("="*70)
    # Calculate training duration metrics
    steps_per_epoch = len(train_loader)
    batch_size = config['batch_size']
    seq_len = config['max_seq_len']
    tokens_per_step = batch_size * seq_len

    # Get dataset size for coverage calculation
    try:
        dataset_tokens = len(train_loader.dataset.tokens)
    except AttributeError:
        dataset_tokens = None

    if train_config.epochs is not None and train_config.epochs > 0:
        effective_steps = train_config.epochs * steps_per_epoch
        total_tokens = effective_steps * tokens_per_step
        print(f"  Epochs:         {train_config.epochs}")
        print(f"  Steps/epoch:    {steps_per_epoch:,}")
        print(f"  Total steps:    {effective_steps:,}")
        print(f"  Tokens seen:    {total_tokens:,} ({total_tokens/1e6:.1f}M)")
        if dataset_tokens:
            coverage = total_tokens / dataset_tokens * 100
            print(f"  Dataset:        {dataset_tokens:,} ({dataset_tokens/1e6:.1f}M) - {coverage:.1f}% coverage")
    else:
        equiv_epochs = train_config.max_steps / steps_per_epoch
        total_tokens = train_config.max_steps * tokens_per_step
        print(f"  Max steps:      {train_config.max_steps:,}")
        print(f"  Steps/epoch:    {steps_per_epoch:,}")
        print(f"  *** EPOCHS:     {equiv_epochs:.4f} ***")
        print(f"  Tokens seen:    {total_tokens:,} ({total_tokens/1e6:.1f}M)")
        if dataset_tokens:
            coverage = total_tokens / dataset_tokens * 100
            print(f"  Dataset:        {dataset_tokens:,} ({dataset_tokens/1e6:.1f}M) - {coverage:.1f}% coverage")
    print(f"  Warmup:         {train_config.warmup_steps}")
    print(f"  Batch size:     {batch_size}")
    print(f"  Seq length:     {seq_len}")
    print(f"  Num workers:    {config.get('num_workers', 0)}")
    print(f"\nFree Energy Weights:")
    print(f"  α (self-consistency): {train_config.alpha}")
    print(f"  β (belief align):     {train_config.beta}")
    print(f"  γ (model align):      {train_config.lambda_gamma}")

    # P-FLOW configuration
    if train_config.use_p_flow:
        print(f"\nP-FLOW (EMA prior updates): ENABLED")
        print(f"  EMA decay: {train_config.p_flow_ema_decay} ({(1-train_config.p_flow_ema_decay)*100:.1f}% update per step)")
    else:
        print(f"\nP-FLOW: disabled")

    # DELTA RULE configuration
    if train_config.use_delta_rule_w_out:
        print(f"\nDELTA RULE (backprop-free W_out): ENABLED")
        print(f"  Learning rate: {train_config.delta_rule_lr}")
        if train_config.use_p_flow:
            print(f"  ** FULLY BACKPROP-FREE MODE **")
    else:
        print(f"\nDELTA RULE: disabled (using backprop for W_out)")

    # RG METRICS configuration
    if train_config.compute_rg_metrics:
        print(f"\nRG METRICS (meta-agent emergence): ENABLED")
        print(f"  Compute interval: every {train_config.rg_metrics_interval} steps")
        print(f"  Dynamic RG tracking: {train_config.track_dynamic_rg}")
    else:
        print(f"\nRG METRICS: disabled")

    # =================================================================
    # Create Trainer (Pure FEP or Standard)
    # =================================================================

    print("\n" + "="*70)
    print("INITIALIZING TRAINER")
    print("="*70)

    # Create comprehensive publication metrics tracker
    pub_metrics = None
    if enable_publication_metrics:
        experiment_name = f"{ffn_mode}_{time.strftime('%Y%m%d_%H%M%S')}"
        pub_metrics = PublicationMetrics(
            experiment_name=experiment_name,
            base_dir=exp_checkpoint_dir / "publication_outputs"
        )

        # Configure gauge frame semantic analysis interval
        # Priority: config dict > CLI args > default (10000)
        semantic_interval = config.get('semantic_analysis_interval',
                                       getattr(args, 'semantic_analysis_interval', 10000) if args else 10000)
        pub_metrics.set_semantic_analysis_interval(semantic_interval)
        print(f"[Config] Gauge frame semantic analysis every {semantic_interval} steps")

        # Configure holonomy diagnostics interval
        holonomy_interval = config.get('holonomy_interval', 500)
        holonomy_sample_size = config.get('holonomy_sample_size', 500)
        pub_metrics.set_holonomy_interval(holonomy_interval, holonomy_sample_size)
        print(f"[Config] Holonomy diagnostics every {holonomy_interval} steps (sample_size={holonomy_sample_size})")

    trainer = PublicationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_config,
        device=device,
        publication_metrics=pub_metrics,
        tokenizer=tokenizer,  # For decoding in interpretability outputs
    )

    # =================================================================
    # Training (Standard Backprop)
    # =================================================================

    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)
    print(f"Device: {device}")
    print(f"FFN mode: {ffn_mode}")
    # Show epochs-based info if set
    if train_config.epochs is not None and train_config.epochs > 0:
        eff_steps = train_config.epochs * steps_per_epoch
        print(f"Epochs: {train_config.epochs} ({steps_per_epoch:,} steps/epoch = {eff_steps:,} total)")
    else:
        print(f"Total steps: {train_config.max_steps:,}")
    print("\nNOTE: First few batches may be slow (JIT compilation)")
    print("="*70 + "\n")

    try:
        trainer.train()

        print("\n" + "="*70)
        print("✓ TRAINING COMPLETE!")
        print("="*70)

        # Final evaluation
        final_metrics = trainer.validate()

        print(f"\nFinal Validation Metrics:")
        print(f"  Loss:       {final_metrics['loss']:.4f}")
        print(f"  Perplexity: {final_metrics['perplexity']:.2f}")

        # vs random baseline
        random_ppl = actual_vocab_size
        improvement = random_ppl / final_metrics['perplexity']
        print(f"\nValidation improvement over random:")
        print(f"  Random:     {random_ppl:.0f}")
        print(f"  Model:      {final_metrics['perplexity']:.2f}")
        print(f"  Factor:     {improvement:.1f}x better!")

        # Save final checkpoint
        final_ckpt = trainer.save_checkpoint(is_best=True)
        print(f"\n✓ Saved: {final_ckpt}")

        # Run test set evaluation if test loader is available
        test_metrics = None
        if test_loader is not None:
            test_metrics = run_test_evaluation(
                model=model,
                test_loader=test_loader,
                device=device,
                vocab_size=actual_vocab_size,
                config=config,
            )

        # Return metrics
        result = {
            'ffn_mode': ffn_mode,
            'final_loss': final_metrics['loss'],
            'final_ppl': final_metrics['perplexity'],
            'random_ppl': random_ppl,
            'improvement': improvement,
            'total_params': total_params,
            'vocab_size': actual_vocab_size,
            'checkpoint': str(final_ckpt),
            # Training duration stats
            'total_steps': train_config.max_steps if train_config.epochs is None else train_config.epochs * steps_per_epoch,
            'tokens_seen': total_tokens,
            'dataset_tokens': dataset_tokens,
            'dataset_coverage': total_tokens / dataset_tokens if dataset_tokens else None,
            'batch_size': batch_size,
            'seq_len': seq_len,
            # FLOPs (Peer Review M2e)
            'flops_per_step': step_flops,
            'flops_per_step_str': format_flops(step_flops),
            'total_training_flops': total_flops,
            'total_training_flops_str': format_flops(total_flops),
        }

        # Add test metrics if available
        if test_metrics is not None:
            result['test_loss'] = test_metrics['test_loss']
            result['test_ppl'] = test_metrics['test_ppl']
            result['test_bpc'] = test_metrics['test_bpc']
            result['test_improvement'] = test_metrics['improvement']

        return result

    except KeyboardInterrupt:
        print("\n\n" + "="*70)
        print("TRAINING INTERRUPTED")
        print("="*70)
        ckpt = trainer.save_checkpoint(is_best=False)
        print(f"✓ Saved: {ckpt}")
        return None

    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        raise


# =============================================================================
# PURE VFE EXPERIMENT (dedicated loop — no nn.Module, no optimizer)
# =============================================================================

def _validate_pure_vfe(model, loader, device, max_samples=12800):
    """Validation for PureVFETransformer: forward-only, compute CE manually."""
    total_ce_tokens = 0.0
    total_tokens = 0
    total_samples = 0

    with torch.no_grad():
        for input_ids, target_ids in loader:
            if total_samples >= max_samples:
                break
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            logits = model.forward(input_ids)
            ce = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1),
                ignore_index=-100,
            ).item()

            non_pad = (target_ids != -100).sum().item()
            total_ce_tokens += ce * non_pad
            total_tokens += non_pad
            total_samples += input_ids.size(0)

    avg_ce = total_ce_tokens / max(1, total_tokens)
    return {
        'loss': avg_ce,
        'ce_loss': avg_ce,
        'perplexity': math.exp(min(avg_ce, 20)),
    }


def run_pure_vfe_experiment(
    config: dict,
    device: torch.device,
    checkpoint_dir: Path,
    args: argparse.Namespace = None,
) -> Dict:
    """
    Run a training experiment with the Pure VFE Transformer.

    This is a dedicated training loop since PureVFETransformer is NOT an
    nn.Module — it has no parameters(), no backward(), no optimizer.
    Training happens via model.update() which internally runs E-step
    (VFE descent) + M-step (natural gradient on priors).
    """
    print("\n" + "="*70)
    print("EXPERIMENT: PURE VFE TRANSFORMER")
    print("="*70)

    ffn_mode = 'pure_vfe'
    exp_checkpoint_dir = checkpoint_dir / f"ffn_{ffn_mode}"
    exp_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save experiment configuration
    save_experiment_config(config, ffn_mode, exp_checkpoint_dir, args)

    # =================================================================
    # Data Loading (same pipeline as other modes)
    # =================================================================
    dataset_name = config.get('dataset', 'wikitext-103')
    print("\n" + "="*70)
    print(f"LOADING {dataset_name.upper()} DATA")
    print("="*70)

    tokenizer_mode = config.get('tokenizer', 'auto')
    if tokenizer_mode == 'auto':
        use_char = config['vocab_size'] <= 256
    else:
        use_char = (tokenizer_mode == 'char')

    test_loader = None
    if use_char:
        print(f"Using CHARACTER-LEVEL tokenizer (vocab_size={config['vocab_size']})")
        train_loader, val_loader, actual_vocab_size = create_char_dataloaders(
            max_seq_len=config['max_seq_len'],
            batch_size=config['batch_size'],
            num_workers=config.get('num_workers', 0),
        )
    else:
        print(f"Using BPE tokenizer (vocab_size={config['vocab_size']})")
        train_loader, val_loader, test_loader, actual_vocab_size, tokenizer = create_dataloaders(
            max_seq_len=config['max_seq_len'],
            batch_size=config['batch_size'],
            vocab_size=config['vocab_size'],
            num_workers=config.get('num_workers', 0),
            dataset=dataset_name,
            include_test=True,
            return_tokenizer=True,
        )

    config['vocab_size'] = actual_vocab_size

    # =================================================================
    # Model Creation
    # =================================================================
    print("\n" + "="*70)
    print("CREATING PURE VFE MODEL")
    print("="*70)

    # Build PureVFEConfig from the config dict (only pass fields that PureVFEConfig accepts)
    import dataclasses
    vfe_field_names = {f.name for f in dataclasses.fields(PureVFEConfig)}
    vfe_kwargs = {k: v for k, v in config.items() if k in vfe_field_names}
    vfe_kwargs['device'] = str(device)
    pure_config = PureVFEConfig(**vfe_kwargs)

    model = PureVFETransformer(pure_config)
    params = model.param_count()
    total_params = params['total']

    print(f"  K (belief_dim): {pure_config.belief_dim}")
    print(f"  H (n_heads):    {pure_config.n_heads}")
    print(f"  K_h (head_dim): {pure_config.head_dim}")
    print(f"  N (seq len):    {pure_config.max_seq_len}")
    print(f"  E-steps:        {pure_config.n_esteps}")
    print(f"  eta_E:          {pure_config.eta_E}")
    print(f"  eta_M:          {pure_config.eta_M}")
    print(f"  gauge_param:    {pure_config.gauge_param}")
    print(f"  Vocab:          {actual_vocab_size}")
    print(f"\nModel Parameters: {total_params:,}")
    for k, v in params.items():
        if k != 'total':
            print(f"  {k}: {v:,}")

    # Attempt to load CUDA kernels
    if pure_config.use_cuda_kernels and str(device) == 'cuda':
        try:
            from transformer.pure_vfe.cuda_ext import get_cuda_ext
            cuda_ext = get_cuda_ext()
            if cuda_ext:
                print("\n[CUDA kernels active]")
            else:
                print("\n[Falling back to PyTorch ops]")
        except Exception:
            print("\n[CUDA kernels not available, using PyTorch ops]")

    # =================================================================
    # Training Configuration
    # =================================================================
    max_steps = config.get('max_steps', 30000)
    log_interval = config.get('log_interval', 100)
    eval_interval = config.get('eval_interval', 1000)
    checkpoint_interval = config.get('checkpoint_interval', 25000)

    steps_per_epoch = len(train_loader)
    batch_size = config['batch_size']
    seq_len = config['max_seq_len']
    tokens_per_step = batch_size * seq_len
    total_tokens = max_steps * tokens_per_step

    try:
        dataset_tokens = len(train_loader.dataset.tokens)
    except AttributeError:
        dataset_tokens = None

    print("\n" + "="*70)
    print("TRAINING CONFIGURATION")
    print("="*70)
    print(f"  Max steps:      {max_steps:,}")
    print(f"  Steps/epoch:    {steps_per_epoch:,}")
    equiv_epochs = max_steps / steps_per_epoch if steps_per_epoch > 0 else 0
    print(f"  *** EPOCHS:     {equiv_epochs:.4f} ***")
    print(f"  Tokens seen:    {total_tokens:,} ({total_tokens/1e6:.1f}M)")
    if dataset_tokens:
        coverage = total_tokens / dataset_tokens * 100
        print(f"  Dataset:        {dataset_tokens:,} ({dataset_tokens/1e6:.1f}M) - {coverage:.1f}% coverage")
    print(f"  Batch size:     {batch_size}")
    print(f"  Seq length:     {seq_len}")
    print(f"  No optimizer (natural gradient only)")

    # =================================================================
    # Metrics Tracker
    # =================================================================
    metrics_path = exp_checkpoint_dir / 'metrics.csv'
    metrics_tracker = PublicationMetricsTracker(metrics_path)
    print(f"\n[INFO] Logging metrics to: {metrics_path}")

    # =================================================================
    # Training Loop
    # =================================================================
    print("\n" + "="*70)
    print("STARTING PURE VFE TRAINING")
    print("="*70)
    print(f"  Device: {device}")
    print(f"  No backprop — E-step VFE descent + M-step natural gradient")
    print("="*70 + "\n")

    best_val_ce = float('inf')
    train_iterator = iter(train_loader)
    start_time = time.time()

    try:
        from tqdm import tqdm
        pbar = tqdm(range(max_steps), desc="Training")
        use_tqdm = True
    except ImportError:
        pbar = range(max_steps)
        use_tqdm = False

    try:
        for step in pbar:
            step_start = time.time()

            # Get batch
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                batch = next(train_iterator)

            input_ids, target_ids = batch
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            # Training step: E-step + M-step (no backward!)
            logits, ce_loss, vfe_history, _diag = model.update(input_ids, target_ids)

            step_time = time.time() - step_start

            # Logging
            if (step + 1) % log_interval == 0:
                ppl = math.exp(min(ce_loss, 20))
                bpc = ce_loss / math.log(2)
                vfe_0 = vfe_history[0] if vfe_history else 0.0
                vfe_f = vfe_history[-1] if vfe_history else 0.0

                metrics = {
                    'train_loss_total': ce_loss,
                    'train_loss_ce': ce_loss,
                    'train_loss_belief_align': 0,
                    'train_loss_self_consistency': 0,
                    'train_loss_model_coupling': 0,
                    'train_ppl': ppl,
                    'beta_mean': 0,
                    'beta_std': 0,
                    'kl_mean': 0,
                    'kl_std': 0,
                    'attention_entropy': 0,
                    'attention_concentration': 0,
                }

                # No optimizer LRs — use eta_E/eta_M as stand-ins
                lrs = {
                    'eta_E': pure_config.eta_E,
                    'eta_M': pure_config.eta_M,
                }

                metrics_tracker.log_step(
                    step + 1, metrics, lrs, None, step_time, batch_size, seq_len
                )

                log_msg = (
                    f"Step {step+1}/{max_steps} | "
                    f"CE: {ce_loss:.4f} | "
                    f"PPL: {ppl:.1f} | "
                    f"BPC: {bpc:.3f} | "
                    f"VFE: {vfe_0:.1f}->{vfe_f:.1f} | "
                    f"{step_time:.2f}s"
                )

                if use_tqdm:
                    pbar.set_description(log_msg)
                else:
                    print(log_msg)

                # Prior health diagnostics
                with torch.no_grad():
                    sig_eigs = torch.linalg.eigvalsh(model.prior_Sigma[:100])
                    sig_min = sig_eigs[..., 0].min().item()
                    sig_max = sig_eigs[..., -1].max().item()
                    mu_norms = model.prior_mu.norm(dim=-1)
                    mu_mean = mu_norms.mean().item()
                    mu_max = mu_norms.max().item()

                    if sig_min < 0.05 or mu_max > 10.0:
                        health_msg = (f"  [WARN] Sigma_min={sig_min:.4f} Sigma_max={sig_max:.2f} "
                                      f"mu_mean={mu_mean:.2f} mu_max={mu_max:.2f}")
                        if use_tqdm:
                            tqdm.write(health_msg)
                        else:
                            print(health_msg)

                    health = monitor_omega_health(model.prior_Omega[:100], "prior_Omega")
                    if health['prior_Omega/cond_max'] > 100:
                        omega_msg = f"  [WARN] Omega cond number high: {health['prior_Omega/cond_max']:.1f}"
                        if use_tqdm:
                            tqdm.write(omega_msg)
                        else:
                            print(omega_msg)

                # Flush numerical events
                _num_events = _flush_numerical_events()
                if _num_events:
                    _num_msg = "  [NUM] " + " | ".join(
                        f"{k}: {v}" for k, v in sorted(_num_events.items())
                    )
                    if use_tqdm:
                        tqdm.write(_num_msg)
                    else:
                        print(_num_msg)

            # Validation
            if (step + 1) % eval_interval == 0:
                val_metrics = _validate_pure_vfe(model, val_loader, device)
                metrics_tracker.log_val(step + 1, val_metrics)

                print(f"\n  Validation @ step {step+1}:")
                print(f"    Loss: {val_metrics['loss']:.4f}")
                print(f"    PPL: {val_metrics['perplexity']:.2f}")
                print(f"    BPC: {val_metrics['ce_loss']/math.log(2):.3f}\n")

                # Save best model
                if val_metrics['ce_loss'] < best_val_ce:
                    best_val_ce = val_metrics['ce_loss']
                    best_path = exp_checkpoint_dir / 'best_model.pt'
                    model.save(best_path)
                    print(f"    Saved best model (CE={best_val_ce:.4f})")

            # Checkpointing
            if (step + 1) % checkpoint_interval == 0:
                ckpt_path = exp_checkpoint_dir / f'checkpoint_step_{step+1}.pt'
                model.save(ckpt_path)
                metrics_tracker.save()

        # Save final metrics
        metrics_tracker.save()
        print(f"\n[INFO] Final metrics saved to: {metrics_path}")

        # Final evaluation
        print("\n" + "="*70)
        print("TRAINING COMPLETE!")
        print("="*70)

        elapsed = time.time() - start_time
        print(f"Total time: {elapsed/60:.1f} minutes ({elapsed/3600:.2f} hours)")

        final_metrics = _validate_pure_vfe(model, val_loader, device)

        print(f"\nFinal Validation Metrics:")
        print(f"  Loss:       {final_metrics['loss']:.4f}")
        print(f"  Perplexity: {final_metrics['perplexity']:.2f}")

        random_ppl = actual_vocab_size
        improvement = random_ppl / final_metrics['perplexity']
        print(f"\nValidation improvement over random:")
        print(f"  Random:     {random_ppl:.0f}")
        print(f"  Model:      {final_metrics['perplexity']:.2f}")
        print(f"  Factor:     {improvement:.1f}x better!")

        # Save final checkpoint
        final_path = exp_checkpoint_dir / 'best_model.pt'
        model.save(final_path)
        print(f"\nSaved: {final_path}")

        # Test set evaluation
        test_metrics = None
        if test_loader is not None:
            print("\n" + "="*70)
            print("FINAL TEST SET EVALUATION")
            print("="*70)
            test_val = _validate_pure_vfe(model, test_loader, device, max_samples=128000)
            test_bpc = test_val['ce_loss'] / math.log(2)
            test_improvement = random_ppl / test_val['perplexity']
            test_metrics = {
                'test_loss': test_val['loss'],
                'test_ppl': test_val['perplexity'],
                'test_bpc': test_bpc,
                'improvement': test_improvement,
            }
            print(f"  Test Loss: {test_val['loss']:.4f}")
            print(f"  Test PPL:  {test_val['perplexity']:.2f}")
            print(f"  Test BPC:  {test_bpc:.3f}")
            print(f"  vs random: {test_improvement:.1f}x better")

        # Return result dict (same format as run_single_experiment)
        result = {
            'ffn_mode': ffn_mode,
            'final_loss': final_metrics['loss'],
            'final_ppl': final_metrics['perplexity'],
            'random_ppl': random_ppl,
            'improvement': improvement,
            'total_params': total_params,
            'vocab_size': actual_vocab_size,
            'checkpoint': str(final_path),
            'total_steps': max_steps,
            'tokens_seen': total_tokens,
            'dataset_tokens': dataset_tokens,
            'dataset_coverage': total_tokens / dataset_tokens if dataset_tokens else None,
            'batch_size': batch_size,
            'seq_len': seq_len,
            # Pure VFE-specific
            'belief_dim': pure_config.belief_dim,
            'n_heads': pure_config.n_heads,
            'n_esteps': pure_config.n_esteps,
            'eta_E': pure_config.eta_E,
            'eta_M': pure_config.eta_M,
            'gauge_param': pure_config.gauge_param,
        }

        if test_metrics is not None:
            result['test_loss'] = test_metrics['test_loss']
            result['test_ppl'] = test_metrics['test_ppl']
            result['test_bpc'] = test_metrics['test_bpc']
            result['test_improvement'] = test_metrics['improvement']

        return result

    except KeyboardInterrupt:
        print("\n\n" + "="*70)
        print("TRAINING INTERRUPTED")
        print("="*70)
        ckpt_path = exp_checkpoint_dir / 'interrupted_model.pt'
        model.save(ckpt_path)
        print(f"Saved: {ckpt_path}")
        metrics_tracker.save()
        return None

    except Exception as e:
        print(f"\n\n Error: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description='Publication Training Script')

    # Mode selection
    parser.add_argument('--mode', type=str, default=DEFAULT_MODE,
                        choices=[
                            'standard', 'VFE_dynamic',
                            # Pure VFE (no backprop)
                            'pure_vfe',                 # Pure VFE transformer (natural gradient only)
                            # Intermediate baselines (peer review M2b, M2c)
                            'standard_attn_only',      # (b) attention-only at d_model=90
                            'standard_param_equalized', # (b') param-equalized wider FFN
                            'standard_rope',            # (c) standard + RoPE at d=10
                            'standard_rope_d90',        # (c') standard + RoPE at d=90
                        ],
                        help='Training mode: standard, VFE_dynamic, pure_vfe, or intermediate baselines')

    # Legacy alias for backwards compatibility
    parser.add_argument('--ffn_mode', type=str, default=None,
                        choices=['VFE_dynamic', 'standard'],
                        help='DEPRECATED: Use --mode instead')
    # System
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_publication')
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--dataset', type=str, default=DEFAULT_DATASET,
                        choices=['wikitext-2', 'wikitext-103', 'wiki-ja'],
                        help='Dataset to use: wikitext-2 (~2M tokens), wikitext-103 (~103M tokens), or wiki-ja (Japanese Wikipedia)')
    parser.add_argument('--semantic_analysis_interval', type=int, default=10000,
                        help='Run gauge frame semantic analysis every N steps (0 to disable)')

    args = parser.parse_args()

    # Handle legacy --ffn_mode argument
    if args.ffn_mode is not None:
        print("⚠ WARNING: --ffn_mode is deprecated. Use --mode instead.")
        args.mode = args.ffn_mode

    # Set random seed for reproducibility
    # Default to seed=42 if not specified, for consistent results
    seed = args.seed if args.seed is not None else SEED
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Enable deterministic CUDA operations (may slow down training slightly)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")

    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print("="*70)
    print("PUBLICATION PROOF-OF-PRINCIPLE TRAINING")
    print("="*70)
    print(f"\nDevice: {device}")

    checkpoint_dir = Path(args.checkpoint_dir)

    # =================================================================
    # SELECT CONFIG BASED ON MODE
    # =================================================================
    # Configs:
    #   STANDARD_CONFIG               - Baseline transformer (dot-product + MLP)
    #   VFE_EM_CONFIG                 - VFE with EM-step dynamics (backprop)
    #   STANDARD_ATTN_ONLY_CONFIG     - (M2b) Attention-only at d_model=90
    #   STANDARD_PARAM_EQUALIZED_CONFIG - (M2b') Param-equalized wider FFN
    #   STANDARD_ROPE_CONFIG          - (M2c) Standard + RoPE at d=10
    #   STANDARD_ROPE_D90_CONFIG      - (M2c') Standard + RoPE at d=90
    # =================================================================

    mode = args.mode

    if mode == 'standard':
        print("\n" + "="*70)
        print("MODE: STANDARD TRANSFORMER (Baseline)")
        print("="*70)
        print("   Attention: Q·K^T / √d (dot-product softmax)")
        print("   FFN: Linear → GELU → Linear (learned MLP)")
        print("   Output: Linear projection")
        print("   Learning: Backpropagation")
        print("="*70 + "\n")
        config = STANDARD_CONFIG.copy()
        ffn_mode = 'standard'

    elif mode == 'VFE_dynamic':
        print("\n" + "="*70)
        print("MODE: VFE_EM (VFE with EM-step dynamics)")
        print("="*70)
        print("   Attention: KL-divergence based (gauge-equivariant)")
        print("   FFN: VFE EM-step dynamics")
        print("   Output: Linear projection")
        print("   Learning: Backpropagation")
        print("   Position: None (emergent)")
        print("="*70 + "\n")
        config = VFE_EM_CONFIG.copy()
        ffn_mode = 'VFE_dynamic'

    elif mode == 'standard_attn_only':
        print("\n" + "="*70)
        print("MODE: STANDARD ATTENTION-ONLY (Peer Review M2b)")
        print("="*70)
        print("   d_model=90 (matching gauge model)")
        print("   Attention: Q·K^T / √d (dot-product softmax)")
        print("   FFN: DISABLED (attention mechanism only)")
        print("   Purpose: Isolate attention contribution")
        print("="*70 + "\n")
        config = STANDARD_ATTN_ONLY_CONFIG.copy()
        ffn_mode = 'standard'

    elif mode == 'standard_param_equalized':
        print("\n" + "="*70)
        print("MODE: STANDARD PARAM-EQUALIZED (Peer Review M2b')")
        print("="*70)
        print("   d_model=90 with wider FFN layers")
        print("   Attention: Q·K^T / √d (dot-product softmax)")
        print("   FFN: Wider layers for parameter matching")
        print("   Purpose: Fair param-count comparison")
        print("="*70 + "\n")
        config = STANDARD_PARAM_EQUALIZED_CONFIG.copy()
        ffn_mode = 'standard'

    elif mode == 'standard_rope':
        print("\n" + "="*70)
        print("MODE: STANDARD + RoPE (Peer Review M2c)")
        print("="*70)
        print("   d_model=10, matching STANDARD_CONFIG dimensions")
        print("   Attention: Q·K^T / √d with RoPE")
        print("   FFN: Linear → GELU → Linear")
        print("   Purpose: Control for positional encoding confound")
        print("="*70 + "\n")
        config = STANDARD_ROPE_CONFIG.copy()
        ffn_mode = 'standard'

    elif mode == 'standard_rope_d90':
        print("\n" + "="*70)
        print("MODE: STANDARD + RoPE at d=90 (Peer Review M2c')")
        print("="*70)
        print("   d_model=90, matching gauge model embedding dim")
        print("   Attention: Q·K^T / √d with RoPE")
        print("   FFN: Linear → GELU → Linear")
        print("   Purpose: Matched PE + matched embed_dim comparison")
        print("="*70 + "\n")
        config = STANDARD_ROPE_D90_CONFIG.copy()
        ffn_mode = 'standard'

    elif mode == 'pure_vfe':
        print("\n" + "="*70)
        print("MODE: PURE VFE TRANSFORMER (No backprop)")
        print("="*70)
        print("   Model: Prior bank {N(mu_v, Sigma_v), Omega_v} per token")
        print("   Inference: E-step VFE descent (replaces forward pass)")
        print("   Learning: M-step natural gradient on prior bank")
        print("   Attention: KL-divergence based with gauge transport")
        print("   No nn.Module, no autograd, no optimizer")
        print("="*70 + "\n")
        config = PURE_VFE_CONFIG.copy()
        config['dataset'] = args.dataset

        if args.dataset == 'wiki-ja' and config['vocab_size'] == 50257:
            config['vocab_size'] = 100277
            print(f"\n[wiki-ja] Auto-adjusted vocab_size: 50257 -> 100277 (cl100k_base full vocab)")

        # Pure VFE uses a dedicated experiment loop (not run_single_experiment)
        result = run_pure_vfe_experiment(
            config=config,
            device=device,
            checkpoint_dir=checkpoint_dir,
            args=args,
        )

        if result is not None:
            result_file = checkpoint_dir / f"result_{mode}.json"
            result_file.parent.mkdir(parents=True, exist_ok=True)
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nSaved result: {result_file}")

        print("\n" + "="*70)
        print("SESSION COMPLETE")
        print("="*70)
        return

    else:
        print(f"\nError: Unknown mode '{mode}'")
        print("Valid modes: standard, VFE_dynamic, pure_vfe, standard_attn_only, "
              "standard_param_equalized, standard_rope, standard_rope_d90")
        return

    config['dataset'] = args.dataset

    # For wiki-ja, use cl100k_base's full vocab (100277) instead of GPT-2's (50257)
    # The cl100k_base tokenizer has much better CJK coverage; restricting to 50257
    # would discard important Japanese tokens and map them to UNK
    if args.dataset == 'wiki-ja' and config['vocab_size'] == 50257:
        config['vocab_size'] = 100277
        print(f"\n[wiki-ja] Auto-adjusted vocab_size: 50257 -> 100277 (cl100k_base full vocab)")

    result = run_single_experiment(
        config=config,
        ffn_mode=ffn_mode,
        device=device,
        checkpoint_dir=checkpoint_dir,
        use_wandb=args.use_wandb,
        args=args,
    )

    if result is not None:
        # Save result
        result_file = checkpoint_dir / f"result_{mode}.json"
        result_file.parent.mkdir(parents=True, exist_ok=True)
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved result: {result_file}")

    print("\n" + "="*70)
    print("SESSION COMPLETE")
    print("="*70)


if __name__ == '__main__':

    main()

