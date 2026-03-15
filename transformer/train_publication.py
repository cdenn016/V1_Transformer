# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 08:09:57 2026

@author: chris and christine
"""


# =============================================================================
# PATH SETUP - Ensure the project root is in the Python path# -*- coding: utf-8 -*-
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
from transformer.data import create_dataloaders, create_char_dataloaders
from transformer.train import (
    compute_free_energy_loss,
    compute_rg_metrics_from_attention,
    compute_dynamic_rg_metrics,
)
from transformer.training.train_fast import FastTrainer, FastTrainingConfig
from transformer.analysis.publication_metrics import PublicationMetrics, ExperimentResult



# ============================================================================
# EDIT THESE DEFAULTS TO RUN WITHOUT COMMAND-LINE ARGS (just click Run!)
# ============================================================================
# Two modes available:
#   'standard'    - Standard transformer baseline (dot-product attention + MLP)
#   'VFE_dynamic' - VFE with EM-step dynamics (backprop training)
DEFAULT_MODE = 'VFE_dynamic'      # Which mode to run

# Dataset
DEFAULT_DATASET = 'wikitext-103'  # 'wikitext-2' (~2M tokens) or 'wikitext-103' (~103M tokens)
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
    'amortized_inference': True,   # Gradient flow through priors → embeddings learn good E-step init

    'use_deq': False,
    'deq_neumann_terms': 0,

    # Training
    'batch_size': 64,
    'num_workers': 10,            #CPU workers 8--12
    'epochs': None,               # Set to 1-3 for WikiText-2, None for WikiText-103 (use max_steps)
    'max_steps': 15000,           # ~105% coverage on WikiText-103
    'warmup_steps': 100,

    # VFE transformer settings
    'ffn_mode': 'VFE_dynamic',    # VFE EM-step dynamics
    'mask_self_attention': False,  # Prevent attention collapse? needed if learnable-reflection true??
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
    'diagonal_covariance': True,
    'isotropic_covariance': True,    # If True, force Σ = σ²I (scalar variance × identity)
                                       # This is Limit 1 from the manuscript: KL reduces to
                                       # scaled squared Euclidean distance. Combined with
                                       # gauge_mode='trivial' (Limit 2), recovers standard attention.
    
    'enforce_orthogonal': True,   # If True, enforce Ω ∈ SO(K) via Newton-Schulz
                                   # Set False for GL(K) (faster, still gauge-invarian
    'learnable_reflection': False,   # Per-token s_i ∈ {±1}^K → O(K)  - enforce orthogonal=true with glk 
                                    #set gauge-mode=learned and the above 3 = true for transf limit
    
    
    'use_positional_embedding': True,
    'pos_encoding_mode': 'learned',           #'none' 'learned' or 'sinusoidal'
    'use_rope': False,
    
    
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
    'ffn_chunk_size': 512,          #smaller if running out of memory. make large as possible

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
    'gauge_mode': 'constant',    # 'learned': per-token φ, Ω_ij = exp(φ_i)·exp(-φ_j) (cocycle)
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

    # DELTA RULE: Backprop-free learning for W_out
    # If True, W_out is updated via delta rule instead of backpropagation
    # Combined with P-flow, this makes learning fully backprop-free!
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

    # Use same code path as validation: compute_free_energy_loss
    # which calls model.forward_with_attention() internally.
    is_standard = isinstance(model, StandardTransformerLM)

    # Extract config values (default to 0 for pure CE if no config)
    alpha = config.get('alpha', 0) if config else 0
    beta = config.get('beta', 0) if config else 0
    lambda_gamma = config.get('lambda_gamma', 0) if config else 0
    kappa_gamma = config.get('kappa_gamma', 1.0) if config else 1.0

    model.eval()
    total_ce = 0.0
    num_batches = 0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, (input_ids, target_ids) in enumerate(test_loader):
            if total_samples >= max_samples:
                break

            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            if is_standard:
                output = model(input_ids, labels=target_ids)
                ce_loss = output['loss'].item()
            else:
                pad_token_id = getattr(test_loader.dataset, 'pad_token_id', -100)
                loss, metrics = compute_free_energy_loss(
                    model,
                    input_ids,
                    target_ids,
                    alpha=alpha,
                    lambda_beta=beta,
                    lambda_gamma=lambda_gamma,
                    kappa_gamma=kappa_gamma,
                    pad_token_id=pad_token_id,
                )
                ce_loss = metrics['loss/ce']

            total_ce += ce_loss
            num_batches += 1
            total_samples += input_ids.size(0)

            # Progress indicator
            if (batch_idx + 1) % 100 == 0:
                print(f"  Evaluated {total_samples}/{max_samples} samples ({num_batches} batches)...")

    # Compute metrics (same averaging as validation: mean of batch means)
    test_ce = total_ce / max(1, num_batches)
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

        # If delta rule is enabled, exclude W_out from backprop
        if use_delta_rule and hasattr(self.model, 'out_proj'):
            self.model.out_proj.weight.requires_grad = False

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
                        lambda_beta=self.config.beta,  # config['beta'] → training loss belief coupling
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
                    lambda_beta=self.config.beta,  # config['beta'] → training loss belief coupling
                    lambda_gamma=self.config.lambda_gamma,
                    kappa_gamma=self.config.kappa_gamma,
                    lambda_hyper=self.config.lambda_hyper,
                    pad_token_id=self.pad_token_id,
                    use_obs_in_vfe=self.config.use_obs_in_vfe,
                    alpha_phi=self.config.alpha_phi,
                )
            loss.backward()

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

        # Re-enable requires_grad for W_out if it was disabled
        if use_delta_rule and hasattr(self.model, 'out_proj'):
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

            # Call P-flow update on the model
            if hasattr(self.model, 'p_flow_update'):
                self.model.p_flow_update(
                    token_ids=input_ids,
                    mu_beliefs=mu_beliefs,
                    prediction_errors=ce_per_position,
                    ema_decay=ema_decay,
                )

        # =================================================================
        # DELTA RULE: Backprop-free update of W_out
        # =================================================================
        # Uses local learning rule: ΔW = η · (target - prediction) ⊗ μ^T
        # Combined with P-flow, this makes learning fully backprop-free!
        if use_delta_rule and 'p_flow/mu_q' in full_metrics:
            mu_beliefs = full_metrics['p_flow/mu_q']
            delta_lr = getattr(self.config, 'delta_rule_lr', 0.001)

            # Call delta rule update on the model
            if hasattr(self.model, 'delta_rule_update_w_out'):
                self.model.delta_rule_update_w_out(
                    mu_beliefs=mu_beliefs,
                    targets=target_ids,
                    lr=delta_lr,
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


def main():
    parser = argparse.ArgumentParser(description='Publication Training Script')

    # Mode selection (three distinct modes)
    parser.add_argument('--mode', type=str, default=DEFAULT_MODE,
                        choices=['standard', 'VFE_dynamic'],
                        help='Training mode: standard (baseline), VFE_dynamic (EM-step)')

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
    seed = SEED
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
    # Three distinct configs for clarity:
    #   STANDARD_CONFIG  - Baseline transformer (dot-product + MLP)
    #   VFE_EM_CONFIG    - VFE with EM-step dynamics (backprop)
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
        print("\n\n\n>>> USING HARDCODED VFE_EM_CONFIG (defined at line 207 in train_publication.py) <<<\n\n\n")
        ffn_mode = 'VFE_dynamic'

    else:
        print(f"\nError: Unknown mode '{mode}'")
        print("Valid modes: standard, VFE_dynamic")
        return

    config['dataset'] = args.dataset

    # For wiki-ja, use cl100k_base's full vocab (100277) instead of GPT-2's (50257)
    # The cl100k_base tokenizer has much better CJK coverage; restricting to 50257
    # would discard important Japanese tokens and map them to UNK
    if args.dataset == 'wiki-ja' and config['vocab_size'] == 50257:
        config['vocab_size'] = 100277
        print(f"\n[wiki-ja] Auto-adjusted vocab_size: 50257 → 100277 (cl100k_base full vocab)")

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


