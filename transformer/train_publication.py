
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
import argparse
import json
from pathlib import Path


# ============================================================================
# EDIT THESE DEFAULTS TO RUN WITHOUT COMMAND-LINE ARGS (just click Run!)
# ============================================================================
# Primary modes:
#   'standard'   - Dot-product attention + MLP baseline (backprop)
#   'em'         - Gauge VFE + IFT implicit differentiation M-step (backprop)
#   'hebbian'    - Gauge VFE + P-flow/delta-rule (no backprop)
#   'pure_vfe'   - Pure natural gradient E/M (no autograd)
#   'hybrid'

# =================================================================
# GAUGE GROUP SELECTION (Generators from so(N), Transport in GL(K))
# =================================================================
# NOTE: The VFE is invariant under GL(K), not just SO(K)!
# We use so(N) generators to parameterize φ, but transport operators
# Ω = exp(φ·G) live in GL(K). No orthogonality constraint is needed.
# SON: so(N) generators with N(N-1)/2 parameters
#
# SO(3), e.g.  ('ℓ0', 50, 1),   
# SO(5) multi-irrep example:
# ('scalar', 10, 1),   # 10 dims (invariant)
# ('fund', 8, 5),      # 40 dims (vector)
# ('wedge2', 4, 10),   # 40 dims (∧² - angular momentum)
# ('sym2', 3, 14),     # 42 dims (Sym²₀ - quadrupolar)
#
#      Supports multiple irrep types for representational diversity:
#        - 'scalar': dim = 1              (gauge-invariant)
#        - 'fund':   dim = N              (fundamental/vector)
#        - 'wedge2': dim = N*(N-1)/2      (antisymmetric 2-tensor ∧²V)
#        - 'sym2':   dim = N*(N+1)/2 - 1  (symmetric traceless Sym²₀V)

DEFAULT_MODE = 'em'               # Which mode to run
SEED = 6         #6,23,111
# Dataset
DEFAULT_DATASET = 'wikitext-103'  
# 'wikitext-2' (~2M tokens), 'wikitext-103' (~103M tokens), 'wiki-ja' japanese (~190M tokens at default cap, 100k vocab),
# or 'wiki-en' english (~5B tokens at full dump, 100k vocab — cross-lingual counterpart to wiki-ja)
_DEBUG_VFE_GRADS = False
# ============================================================================

# =============================================================================
# CONFIG: EM — Principled E/M with implicit differentiation (mode='em')
# =============================================================================
# Gauge-covariant VFE transformer with proper E/M separation:
#
#   E-step: Natural gradient descent on F w.r.t. (μ_q, Σ_q, φ) inside forward pass.
#           Fisher-preconditioned μ, SPD retraction for Σ, Killing-form for φ.
#           Adaptive α_i = c0/(b0 + KL) gates prior coupling per agent.
#
#   M-step: Backprop through IFT-scaled gradient. Scale s_k = (α/σ²_p)/A_k
#           where A_k is the effective precision at the E-step fixed point.
#           Replaces ad-hoc straight-through (s=1) with the info-geometrically
#           correct value. CE → W_out directly; CE → embeddings via IFT scale.
#           KL(q*||p) → embeddings directly. KL(s||h) → sigma with fixed Σ_h.
#
#   Hierarchy: h(fixed) → s(embed params) → p=s → q(E-step beliefs) → obs
# =============================================================================

#EM_CONFIG is optimized by ablation sweeps
EM_CONFIG = {
    # === Architecture ===
    'vocab_size':                 50257,
    'embed_dim':                  20,
    'max_seq_len':                128,
    
    'batch_size':                 32, 
    'max_steps':                  30000,
     
    'stride':                     128,  
    #'eval_stride':                128,                                                                                              
    'random_offset_per_epoch':    True,
    'stride_base_seed':           6,
    
    'n_layers':                   1,
    'ffn_n_iterations':           1,
    
    'alpha_divergence':           0.3,
    #'grad_accumulation_steps': 1,
    #'gradient_checkpoint_vfe': False,
    
    'gauge_dim':                   10,
    'irrep_spec':        [('fund', 2, 10)],

    'use_prior_bank':             False,
    'gauge_fixed_priors':         False,    
    
    'learnable_pb_temperature':   False,    #prior bank temperature
    'mask_self_attention':        False,  # Prevent attention collapse?
  

    'kappa_beta':                 1,
    'kappa_warmup_steps':         2000,  # freeze kappa for first n steps
    'learnable_head_kappa':       False, # If True, learn per-head κ_h via log_kappa_per_head
    'include_attention_entropy':  True,
    
    'e_step_sigma_floor':         0.01,   # Floor on σ_p inside E-step (caps 1/σ_p at 1/floor)
    
    # === EM gradient-flow mode ===
    'em_mode':                    'ift_phi',  # - 'ift_phi' (default) — mu_p, sigma_p attached, full IFT phi gradient
                                               # - 'em_phi_q' — clean EM, phi in q, all detached at boundary
                                               # - 'em_phi_p' — clean EM, phi frozen in E-step

    # === GL(K) determinant control (off by default; pick at most one) ===
    # GL(K) has an unbounded trace direction that L2-norm clamping does not
    # constrain — det(Ω_ij) = exp(tr(φ_i − φ_j)) blows up on outlier tokens.
    # Recommended for gauge_group='GLK' with phi_max_norm > ~3.
    
    'phi_project_slk':            False,   # Hard project φ → sl(K): det(Ω) ≡ 1 always
    'phi_trace_clamp':            0.75,    # Soft cap |tr(φ·G)| ≤ T (e.g., 0.35 → det ∈ [0.5, 2])


    'active_inference':           False,   #requires priorbank true
    
    'cache_decode_priors':        False,
    'skip_attention':             True,   #skips ad hoc attention sublayer
    
    # === M-step: Optimizer ===  
    'optimizer_type':             'riemannian_adam',# or 'natural_gradient' or 'adamw' or 'riemannian_adam'
    'phi_optimizer_metric':       'killing',
    'fisher_ema_decay':           0.90,            # for natural_gradient
    'fisher_damping':             1e-2,              # for natural_gradient


    'use_layernorm':              True,  #breaks gauge equivariance unless mahal
    'use_residual':               False,  #set False if skip-attention=True
    
    'use_output_projection':      True,
    'use_equivariant_head_mixer': False,  # Opt-in principled replacement for W_o
    
    
    'evolve_sigma':               True,
    'evolve_phi':                 True,  #M-step phi evolution
    'evolve_phi_e_step':          True,
    'normalize_ce_by_dim':        True,
    'ce_label_smoothing':         0.0,    # Label smoothing on CE loss only; PPL stays un-smoothed

    'E_learnable_alpha':          True,   # Adaptive α_i = c0/(b0 + KL) per dimension
    'E_learnable_lr':             True,   # Learnable E-step LR
    
    'min_lr_ratio':               0.1,
    'lr_decay':                   'cosine',   #'linear', 'cosine', 'constant'
    
    'norm_type':                  'layernorm',  # 'layernorm' | 'rmsnorm' | 'mahalnorm' | 'none'
                                                #'centered_mahalnorm'
                                                
    'residual_type':              'additive',    # 'additive': mu_q = mu_q + mu_sub 
                                         # 'delta':    mu_q = mu_q + (mu_sub - mu_normalized),
    
    'closed_form_e_step':         False,
    'n_picard_steps':             0,
    
    # === E-step Weights ===
 
    'E_alpha':                    1,      # E-step prior coupling weight
    'E_lambda_belief':            9,    # E-step belief alignment weight
    'E_lambda_softmax':           0,
       
    # === E-step Learning Rates ===
    
    'E_mu_q_lr':                  0.3,    # E-step μ step size (whitened, within trust=2.0)
    'E_sigma_q_lr':               0.015,  # E-step σ step size — DECOUPLED from E_mu_q_lr.
                                          # Drives the σ retraction directly:
                                          # σ_new = σ · exp(E_sigma_q_lr · decay_t · clamp(δσ/σ, ±E_sigma_q_trust))
    'E_sigma_q_trust':            5.0,    # E-step σ trust-region clamp on |δσ/σ| (separate from step size)
    'E_phi_lr':                   0.05,   # E-step φ step size

    # === M-step Weights ===        
    
    'M_alpha':                    0,   # M-step KL(q||p) self-consistency
    'M_beta':                     0,    # M-step belief alignment
    'mass_phi':                   0,    # Gauge prior: (mass_φ/2)||φ||²
    'lambda_hyper':               0,    # KL(s||h) explicit loss (pulls tokens toward centroid)
    'lambda_gamma':               0,
    # === M-step Learning Rates (AdamW parameter groups) ===
    
    'M_mu_p_lr':                  0.07,   # M-step prior mean embeddings (μ_p) 0.05
    'M_sigma_p_lr':               0.015,     # M-step prior covariance embeddings (log σ_p) 0.015
    'M_phi_lr':                   0.003,    # M-step gauge frame embeddings (φ) 0.0075
    
    # === M-step Other LR's (AdamW parameter groups) ===
    'M_vfe_hyperparam_lr':        0.095,  # M-step VFE hyperparams (raw_c0, raw_b0, raw_lr) 0.05
    'M_attention_lr':             0.013,  # M-step attention params (W_O, constant_omega) was0.06
    'M_output_lr':                0.05,  # M-step output projection (vocab logits) 0.05
    'embed_weight_decay':         0.0016,   # L2 hyper-prior on embeddings (μ_p, σ_p, φ) via AdamW
    'non_embed_weight_decay':     0.0043,  # L2 on non-embedding params (attention, output)

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
    'rope_base':                  100,   
    'rope_full_gauge':            'off', # 'off', 'both', 'vfe_only'
    'pos_encoding_mode':          'none',

    # === Embedding init ===
    'mu_init_std':                0.4,
    'phi_scale':                  0.05,
    
    'mu_normalize':               False,
    'mu_max_norm':                None,


    # === Logging ===
    'log_interval':               200,
    'eval_interval':              2000,
    'checkpoint_interval':        25000,
    'semantic_analysis_interval': 4000,
    'gauge_geometry_interval':    4000,   # Gauge field Dirichlet energy + invariants
    'fiber_trajectory_interval':  4000,   # Fisher-Rao E-step trajectory (requires ffn_n_iterations > 1)

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
    #'cross_couplings': [(0, 1), (1, 0)],
    # → super-blocks: [20, 10]  (heads 0,1 merged into GL(20), head 2 alone)
    
    # === Layer/iteration diagnostics ===
    'track_layer_diagnostics':     True,
    'track_iteration_diagnostics': True,
    'diagnostics_interval':        25,
    
    
    'tie_embeddings':              False,
    'ffn_mode':                    'VFE_dynamic',
    
    'debug_vfe_grads':             False,
    'verbose_diagnostics':         False,
    

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
    'active_inference_epistemic_samples': 10,     # MC samples for BALD
}


# =============================================================================
# CONFIG: HEBBIAN — Backprop-free learning via P-flow + delta rule (mode='hebbian')
# =============================================================================
# Same gauge-VFE model as EM, but ALL parameter learning is local/Hebbian:
#
#   E-step: Same VFE natural gradient descent on (μ_q, Σ_q, φ).
#
#   M-step (no backprop):
#     μ_embed, σ_embed: P-flow EMA toward successful beliefs (prediction-error weighted)
#     φ_embed:          P-flow EMA toward E-step evolved φ (detached from backprop)
#     W_out:            Delta rule ΔW = η·(target - pred) ⊗ μ^T (Widrow-Hoff)
#
#   Loss: CE only (alpha=0, beta=0). VFE regularizers are implicit in the
#         E-step dynamics, not in the training loss.
# =============================================================================
HEBBIAN_CONFIG = {
    # === Architecture (same model as EM) ===
    'vocab_size':   50257,
    'embed_dim':    10,
    'gauge_dim':    10,
    'n_layers':     1,
    'hidden_dim':   508,
    'max_seq_len':  128,

    # === Training ===
    'batch_size':   64,
    'num_workers':  0,
    'max_steps':    15000,
    'warmup_steps': 100,

    'ffn_mode':              'VFE_dynamic',
    'tie_embeddings':        False,

    'use_layernorm':         True,
    # 'norm_type':           'layernorm',  # 'layernorm' | 'rmsnorm' | 'none'
    'use_residual':          True,
    'use_output_projection': True,
    'E_learnable_lr':      True,

    # === Gauge group ===
    'gauge_group':   'GLK',
    'gauge_mode':    'learned',
    'gauge_param':   'phi',
    'irrep_spec':    [('fund', 1, 10)],

    # === E-step dynamics (same as EM) ===
    'ffn_n_iterations': 1,

    'E_alpha':         1.0,
    'E_lambda_belief': 1.0,

    'evolve_sigma':        True,
    'evolve_phi':          True,
    'evolve_phi_e_step':   True,
    'diagonal_covariance': True,

    # === Hebbian M-step: no backprop ===
    'em_mode':              'em_phi_q',  # Detach beliefs at EM boundary; Hebbian uses P-flow + delta-rule for M-step

    # EMA update of embeddings toward successful beliefs
    'use_p_flow':           True,
    'use_delta_rule_w_out': True,   # Widrow-Hoff local learning for W_out
    # phi learns via P-flow only (no backprop)
    'detach_phi':           True,

    'p_flow_ema_decay':     0.95,
    'delta_rule_lr':        0.1,


    # === Loss weights: CE only (no VFE regularizers in backprop loss) ===
    'M_alpha':      0.0,
    'M_beta':       0.0,
    'mass_phi':     0.0,
    'lambda_hyper': 0.0,
    'lambda_gamma': 0.0,
    'kappa_gamma':  1.0,

    # === Phi gradient geometry ===
    'phi_natural_gradient':      'killing',
    'use_killing_form':           True,
    'killing_form_sym_dampening': 0.1,

    # === Position encoding ===
    'use_rope':          True,
    'pos_encoding_mode': 'none',

    # === Embedding init ===
    'mu_init_std': 1.0,
    'phi_scale':   1.0,
    'kappa_beta':  1.0,

    # === Learning rates ===
    # mu/sigma/phi: 0.0 because P-flow EMA is the sole embedding update mechanism.
    # Nonzero backprop LR here would create a hybrid (optimizer + P-flow) that
    # conflicts with the "no backprop" design. Attention/FFN params still need
    # backprop since they have no P-flow equivalent.
    'M_mu_p_lr':        0.0,
    'M_sigma_p_lr':     0.0,
    'M_phi_lr':         0.0,     # phi learns via phi_flow_update only (detach_phi=True)
    'M_vfe_hyperparam_lr': 0.05,
    'M_attention_lr':   0.005,
    'M_output_lr':      0.0,     # W_out learns via delta rule only

    # === Regularization ===
    'non_embed_weight_decay': 0.01,
    'grad_clip':    1.0,

    # === Logging ===
    'log_interval':        100,
    'eval_interval':       1000,
    'checkpoint_interval': 25000,
    
    # === Layer/iteration diagnostics ===
    'track_layer_diagnostics':     False,
    'track_iteration_diagnostics': False,
    'diagnostics_interval':        25,
  
    'debug_vfe_grads':             False,
    'verbose_diagnostics':         False,
}


# =============================================================================
# CONFIG: PURE VFE — No backprop, natural gradient only (mode='pure_vfe')
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
    'n_heads':    4,                 # H: number of heads (block-diagonal)
    'head_dim':   8,                # K_h = K / H

    # Sequence
    'max_seq_len': 64,           # Match other modes
    'batch_size':  32,
    'max_steps':   30000,

    # VFE descent
    'n_esteps': 24,                  # Iterations of VFE descent (replaces "depth")
    'tau':      None,                # Attention temperature (defaults to √K_h)

    # Per-variable natural gradient learning rates
    'mu_q_lr':    0.1,               # Belief mean step size
    'sigma_q_lr': 0.005,             # Belief covariance step size
    'phi_lr':     0.1,               # Gauge connection step size (all frames)
    'mu_p_lr':    0.05,              # Prior mean step size
    'sigma_p_lr': 0.01,              # Prior covariance step size

    # Prior precision (state-dependent α)
    'alpha_b0': 1.0,
    'alpha_c0': 1.0,

    # Hyper-prior regularization
    'hyper_var': 100.0,


    # Initialization
    'sigma_init':       1.0,
    'omega_init_scale': 0.01,

    # Numerical stability
    'grad_clamp':      1e3,

    # Trust regions
    'trust_region_mu':    2.0,
    'trust_region_sigma': 0.15,
    'trust_region_omega': 0.3,

    # SPD retraction
    'spd_eps_min':   1e-3,
    'spd_kappa_max': 1e4,
    'spd_exp_clip':  20.0,

    # Prior safeguards
    'prior_sigma_floor': 0.5,
    'prior_mu_max_norm': 10.0,
    'm_step_trust_mu':   0.5,

    # Gauge frame parameterization
    # 'omega' (direct GL(K)) or 'phi' (Lie algebra)
    'gauge_param':    'omega',
    'omega_cond_max': 100.0,

    # Observation gradient options
    'sigma_obs_grad':   'none',

    # Recovery
    'nan_recovery':     True,

    # Causal masking
    'causal':           True,

    # --- New features ---

    # RoPE: SO(2)^{K/2} position rotations on μ before KL scoring
    'use_rope':         True,
    'rope_base':        75.0,

    # Adam momentum in M-step (variance reduction across batches)
    'use_adam_m_step':  True,
    'adam_beta1':       0.9,
    'adam_beta2':       0.999,
    'adam_eps':         1e-8,

    # LR scheduling (warmup + cosine decay)
    'warmup_steps':     500,
    'lr_schedule':      'cosine',
    'min_lr_ratio':     0.1,

    # Diagonal covariance (faster, optional)
    'diagonal_covariance': True,

    # LayerNorm (optional, for testing)
    'use_layernorm':    False,

    # Holonomy monitoring (measure gauge curvature)
    'use_holonomy':     False,

    # Device & kernels
    'device': 'cuda',
    'use_cuda_kernels': True,

    # Gradient accumulation (K micro-batches per M-step)
    'grad_accum_steps':    1,

    # Training loop params (used by run_pure_vfe_experiment, not PureVFEConfig)
    'log_interval':        100,
    'eval_interval':       1000,
    'checkpoint_interval': 25000,
    'num_workers':         0,
    'figure_interval':     2000,   # Save attention/diagnostic figures every N steps
}


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
    # Model architecture — param-matched to EM_CONFIG (1.52M)
    'vocab_size': 50257,
    'embed_dim': 10,              # Same as VFE for apples-to-apples comparison
    
    'n_layers': 1,                # Same depth
    'n_heads': 1,                 # Single head (head_dim=10)
    'hidden_dim': 24527,          # Absorbs params that VFE spends on σ table + VFE machinery
    'max_seq_len': 128,

    # Training — match VFE config
    'batch_size': 64,
    'num_workers': 0,
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
    'M_mu_p_lr': 3e-4,
    'M_sigma_p_lr': 0.0001,
    'M_phi_lr': 0.0001,
    'M_vfe_hyperparam_lr': 3e-4,

    # Free energy weights (not used in standard mode)
    'M_alpha': 0,
    'M_beta': 0,
    'lambda_gamma': 0,
    'kappa_gamma': 1.0,

    # Regularization
    'non_embed_weight_decay': 0.01,
    'dropout': 0.1,
    'grad_clip': 1.0,

    # Logging — match VFE config
    'log_interval': 100,
    'eval_interval': 1000,
    'checkpoint_interval': 25000,

    # Unused in standard mode
    'kappa_beta': 1.0,
    'attention_pattern': 'full',
    'attention_window': 24,
    'gauge_group': 'SO3',
    'gauge_dim': 3,
    'gauge_mode': 'learned',
    'gauge_fixed_priors': False,
    'irrep_spec': [('ℓ0', 5, 1)],
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
# Isolates the contribution of dot-product attention without FFN.
# Derived from STANDARD_CONFIG — overrides embed_dim, n_heads, hidden_dim,
# and adds disable_ffn=True.
STANDARD_ATTN_ONLY_CONFIG = {
    **STANDARD_CONFIG,
    'embed_dim': 90,              # Same as gauge model embed_dim
    'n_heads': 9,                 # 9 heads * 10 = 90 (head_dim=10, matching gauge d_head)
    'hidden_dim': 360,            # Not used (FFN disabled), but needed for config
    'disable_ffn': True,          # KEY: no FFN, attention only
}

# =============================================================================
# HYBRID GAUGE-ATTENTION CONFIG
# =============================================================================
# Standard neural transformer with gauge KL-attention and PriorBank embeddings.
# Isolates the gauge attention mechanism from the VFE E-step FFN.
#
# Components:
#   - Embeddings: PriorBank (KL encode/decode)
#   - Attention: IrrepMultiHeadAttention (gauge-theoretic KL-divergence)
#   - FFN: Standard GELU (nn.Linear -> GELU -> nn.Linear)
#   - Training: Pure CE loss (no VFE terms)

HYBRID_CONFIG = {
    # === Architecture ===
    'vocab_size':            50257,
    'embed_dim':             40,
    'max_seq_len':           64,
    'n_layers':              1,
    'hidden_dim':            160,   # 4x embed_dim (standard convention)
    'dropout':               0.1,

    # === Gauge group: GL(K) with multi-head block-diagonal structure ===
    'gauge_group':           'GLK',
    'gauge_dim':             10,
    'irrep_spec':            [('fund', 4, 10)],
    'gauge_mode':            'learned',
    'diagonal_covariance':   True,
    'exact_diagonal_transport': False,
    'enforce_orthogonal':    False,
    'mask_self_attention':   False,

    # === Attention ===
    'kappa_beta':            1.0,
    'learnable_head_kappa':  False,
    'use_output_projection': True,
    'sigma_aggregation':     'mixture',

    # === Position encoding ===
    'use_rope':              True,
    'rope_base':             5000,
    'pos_encoding_mode':     'none',

    # === Embedding init ===
    'mu_init_std':           1.0,
    'phi_scale':             1.0,
    'evolve_sigma':          True,
    'gauge_fixed_priors':    False,
    'sigma_ce_scale':        0.1,
    'prior_bank_tau':        1.0,

    # === Training ===
    'batch_size':            128,
    'max_steps':             15000,
    'num_workers':           0,
    'warmup_steps':          100,

    # === Optimizer (pure CE, no VFE terms) ===
    'optimizer_type':        'adamw',
    'M_mu_p_lr':             0.05,
    'M_sigma_p_lr':          0.005,
    'M_phi_lr':              0.0075,
    'M_vfe_hyperparam_lr':   0.005,
    'M_attention_lr':        0.005,
    'M_output_lr':           0.05,

    # === Loss weights (pure CE by default) ===
    'M_alpha':               0.0,
    'M_beta':                0.0,
    'mass_phi':              0.01,
    'lambda_gamma':          0.0,
    'lambda_hyper':          0.0,
    'kappa_gamma':           1.0,

    # === Regularization ===
    'embed_weight_decay':    0.05,
    'non_embed_weight_decay': 0.01,
    'grad_clip':             1.0,

    # === Logging ===
    'log_interval':          100,
    'eval_interval':         1000,
    'checkpoint_interval':   25000,
    'debug_vfe_grads':             False,
    'verbose_diagnostics':         False,
}

# =============================================================================
# IMPORT TRAINING INFRASTRUCTURE
# =============================================================================
# All training machinery lives in transformer/training/experiment_runner.py.
# This file is the click-to-run entry point: edit configs above, then run.

from transformer.training.experiment_runner import (
    run_single_experiment,
    run_pure_vfe_experiment,
    PublicationTrainer,
    PublicationMetricsTracker,
    LayerDiagnosticsTracker,
    IterationDiagnosticsTracker,
    run_test_evaluation,
    save_experiment_config,
    get_git_info,
    get_system_info,
)


# =============================================================================
# MAIN — Select config based on mode, then run
# =============================================================================

def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description='Publication Training Script')

    # Mode selection
    parser.add_argument('--mode', type=str, default=DEFAULT_MODE,
                        choices=[
                            # Primary modes
                            'standard',                 # Dot-product attention + MLP baseline
                            'em',                       # Gauge VFE + IFT implicit differentiation M-step
                            # Gauge VFE + P-flow/delta-rule (no backprop)
                            'hebbian',
                            # Pure natural gradient (no autograd)
                            'pure_vfe',
                            # Peer review ablations
                            # (M2b) attention-only at d_model=90
                            'standard_attn_only',
                            # Hybrid: gauge KL-attention + PriorBank + standard GELU FFN
                            'hybrid',
                        ],
                        help='Training mode: standard, em, amortized, hebbian, pure_vfe')

    # System
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='checkpoints_publication')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--dataset', type=str, default=DEFAULT_DATASET,
                        choices=['wikitext-2', 'wikitext-103', 'wiki-ja', 'wiki-en'],
                        help='Dataset: wikitext-2 (~2M tokens), wikitext-103 (~103M tokens), '
                             'wiki-ja (Japanese Wikipedia, cl100k_base), or wiki-en '
                             '(English Wikipedia, cl100k_base, full ~5B-token dump)')
    parser.add_argument('--semantic_analysis_interval', type=int, default=10000,
                        help='Run gauge frame semantic analysis every N steps (0 to disable)')

    args = parser.parse_args()

    # Set random seed for reproducibility
    seed = args.seed if args.seed is not None else SEED
    from transformer.training.utils import set_all_seeds
    set_all_seeds(seed)
    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    checkpoint_dir = Path(args.checkpoint_dir)

    # =================================================================
    # SELECT CONFIG BASED ON MODE
    # =================================================================
    mode = args.mode

    if mode == 'standard':
        config = STANDARD_CONFIG.copy()
        ffn_mode = 'standard'

    elif mode == 'em':
        config = EM_CONFIG.copy()
        ffn_mode = 'VFE_dynamic'

    elif mode == 'hebbian':
        config = HEBBIAN_CONFIG.copy()
        ffn_mode = 'VFE_dynamic'

    elif mode == 'standard_attn_only':
        config = STANDARD_ATTN_ONLY_CONFIG.copy()
        ffn_mode = 'standard'

    elif mode == 'hybrid':
        config = HYBRID_CONFIG.copy()
        ffn_mode = 'hybrid'

    elif mode == 'pure_vfe':
        config = PURE_VFE_CONFIG.copy()
        config['dataset'] = args.dataset

        if args.dataset in ('wiki-ja', 'wiki-en') and config['vocab_size'] == 50257:
            config['vocab_size'] = 100277
            print(
                f"\n[{args.dataset}] Auto-adjusted vocab_size: 50257 -> 100277 (cl100k_base full vocab)")

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
        print(f"\nError: Unknown mode \'{mode}\'")
        print("Valid modes: standard, em, hebbian, hybrid, pure_vfe, standard_attn_only")
        return

    config['dataset'] = args.dataset
    config['debug_vfe_grads'] = _DEBUG_VFE_GRADS

    if args.dataset in ('wiki-ja', 'wiki-en') and config['vocab_size'] == 50257:
        config['vocab_size'] = 100277
        print(
            f"\n[{args.dataset}] Auto-adjusted vocab_size: 50257 -> 100277 (cl100k_base full vocab)")

    if args.dataset in ('wiki-ja', 'wiki-en') and config.get('eval_stride') is None:
        config['eval_stride'] = config.get('stride', 128)
        print(
            f"\n[{args.dataset}] Auto-set eval_stride={config['eval_stride']} to match training stride. "
            f"Default eval_stride=None (stride-1 sliding) gives only ~13k unique tokens of "
            f"validation coverage at max_samples=12800, which pins val_ppl on a tiny "
            f"non-representative prefix of the tail-sliced val split.")

    result = run_single_experiment(
        config=config,
        ffn_mode=ffn_mode,
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


if __name__ == '__main__':
    main()
