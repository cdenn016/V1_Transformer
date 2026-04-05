"""
Experiment Runner for Gauge VFE Transformer
=============================================

Training infrastructure: experiment execution, metrics tracking, evaluation,
and the PublicationTrainer class. This module contains all the machinery
that runs after config selection.

Extracted from train_publication.py to separate config (entry point) from
execution (this module).. The entry point sets configs and calls
run_single_experiment() or run_pure_vfe_experiment() from here.

Public API:
    - run_single_experiment()    — Run EM/standard/hebbian experiment
    - run_pure_vfe_experiment()  — Run pure VFE (no autograd) experiment
    - PublicationTrainer         — FastTrainer subclass with metrics/diagnostics
    - run_test_evaluation()      — Evaluate model on test set
    - PublicationMetricsTracker  — Step-level metrics collection
    - LayerDiagnosticsTracker    — Per-layer diagnostics
    - IterationDiagnosticsTracker — Per-VFE-iteration diagnostics
"""

import gc
import logging
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

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

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
from transformer.train import compute_free_energy_loss
from transformer.training.train_fast import FastTrainer
from transformer.training.config import TrainingConfig
from transformer.analysis.publication_metrics import PublicationMetrics, ExperimentResult
from math_utils.numerical_monitor import flush as _flush_numerical_events


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
        info['gpu_memory_gb'] = torch.cuda.get_device_properties(
            0).total_memory / 1e9

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
    logger.info("="*70)
    logger.info("FINAL TEST SET EVALUATION")
    logger.info("="*70)

    logger.info(f"  Evaluating up to {max_samples} samples...")

    # Pure CE evaluation — disable all VFE regularization terms for test.
    is_standard = isinstance(model, StandardTransformerLM)

    # Target padding uses -100 (PyTorch cross_entropy ignore_index default).
    pad_token_id = -100

    model.eval()
    # Sum of CE * non_pad_tokens (for token-weighted avg)
    total_ce_tokens = 0.0
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
                    M_alpha=0.0,
                    M_beta=0.0,
                    lambda_gamma=0.0,
                    kappa_gamma=1.0,
                    lambda_hyper=0.0,
                    pad_token_id=pad_token_id,
                    use_obs_in_vfe=False,
                    mass_phi=0.0,
                    normalize_ce_by_dim=getattr(config, 'normalize_ce_by_dim', False),
                )
                ce_loss = metrics.get('loss/ce_raw', metrics['loss/ce'])

            # Token-weighted accumulation (handles variable-size last batch)
            total_ce_tokens += ce_loss * non_pad
            total_tokens += non_pad
            num_batches += 1
            total_samples += input_ids.size(0)

            # Progress indicator
            if (batch_idx + 1) % 100 == 0:
                logger.info(f"  Evaluated {total_samples}/{max_samples} samples ({num_batches} batches)...")

    # Token-weighted CE average (proper averaging for variable batch sizes)
    test_ce = total_ce_tokens / max(1, total_tokens)
    test_ppl = math.exp(min(test_ce, 20))  # Clamp to prevent overflow
    test_bpc = test_ce / math.log(2)
    random_ppl = vocab_size
    improvement = random_ppl / test_ppl if test_ppl > 0 else 0

    logger.info(f"Test Set Results ({total_samples} samples across {num_batches} batches):")
    logger.info(f"  Cross-entropy loss: {test_ce:.4f}")
    logger.info(f"  Perplexity:         {test_ppl:.2f}")
    logger.info(f"  Bits per character: {test_bpc:.3f}")
    logger.info(f"  Random baseline:    {random_ppl:.0f}")
    logger.info(f"  Improvement:        {improvement:.1f}x better than random")
    logger.info("="*70)

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

    logger.info(f"Saved experiment config: {config_path}")

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
            'train_loss_aux_layer_ce',
            'val_loss', 'val_ce',

            # Metrics
            'train_ppl', 'train_bpc', 'val_ppl', 'val_bpc',

            # Attention stats (crucial for interpretability!)
            'beta_mean', 'beta_std', 'kl_mean', 'kl_std',
            'attention_entropy', 'attention_concentration',

            # Learning rates
            'mu_lr', 'sigma_lr', 'phi_lr', 'ffn_lr',

            # Gradient norms
            'grad_norm_total', 'grad_norm_mu', 'grad_norm_ffn',

            # Bayesian alpha diagnostics
            'alpha_mean', 'alpha_std', 'alpha_min', 'alpha_max',
            'alpha_c0', 'alpha_b0', 'alpha_c0_std', 'alpha_b0_std',
            'alpha_mahal_sq_mean', 'alpha_mahal_sq_std',

            # Learnable per-head kappa (temperature)
            'kappa_mean', 'kappa_std', 'kappa_min', 'kappa_max',

            # Performance
            'step_time', 'tokens_per_sec',

            # Holonomy (non-flat transport curvature)
            'holonomy_mean_norm', 'holonomy_max_norm',
            'holonomy_frac_gt_01', 'holonomy_spectral_gap', 'holonomy_wilson_trace',

            # Numerical fallback counters
            'num_chol_recover', 'num_chol_fail', 'num_nan_replace', 'num_inv_pinv',

            # Phi embedding spectral diagnostics
            'phi_effective_rank', 'phi_rank_ratio',
            'phi_top1_variance_fraction', 'phi_top5_variance_fraction',
            'phi_spectral_gap', 'phi_frobenius_norm',
            'phi_mean_token_norm', 'phi_std_token_norm',

            # VFE gradient decomposition (E-step component analysis)
            'vfe_grad_mu_self', 'vfe_grad_mu_direct', 'vfe_grad_mu_softmax', 'vfe_grad_mu_total',
            'vfe_grad_sigma_self', 'vfe_grad_sigma_align_direct',
            'vfe_grad_sigma_softmax', 'vfe_grad_sigma_total',
            'vfe_kl_pairwise_mean', 'vfe_kl_pairwise_max', 'vfe_kappa_scaled',
            'vfe_kl_frac_above_90pct', 'vfe_kl_p95',

            # Covariance health
            'sigma_q_mean', 'sigma_q_min', 'sigma_q_max', 'sigma_q_std',
            'sigma_q_cond_mean', 'sigma_q_cond_max',
            'sigma_p_mean', 'sigma_p_min', 'sigma_p_max',
            'prior_belief_kl_mean', 'prior_belief_kl_max', 'prior_belief_kl_std',

            # Transport & attention structure
            'phi_norm_mean', 'phi_norm_std', 'phi_norm_max',
            'phi_pairwise_dist_mean', 'phi_pairwise_dist_max',
            'attn_entropy_per_head_mean', 'attn_entropy_per_head_std',
            'attn_entropy_per_head_min', 'attn_entropy_per_head_max',
            'head_correlation_mean',
        ]

        with open(self.save_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)

    def log_step(self, step: int, metrics: Dict, lrs: Dict, grad_norms: Dict,
                 step_time: float, batch_size: int, seq_len: int):
        """Log training step with full metrics."""

        # Compute tokens/sec
        tokens_per_sec = (batch_size * seq_len) / \
            step_time if step_time > 0 else 0

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
            'train_loss_aux_layer_ce': metrics.get('train_loss_aux_layer_ce', 0),
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

            # Learnable per-head kappa
            'kappa_mean': metrics.get('kappa/per_head_mean'),
            'kappa_std': metrics.get('kappa/per_head_std'),
            'kappa_min': metrics.get('kappa/per_head_min'),
            'kappa_max': metrics.get('kappa/per_head_max'),

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

            # Phi embedding spectral diagnostics
            'phi_effective_rank': metrics.get('phi/effective_rank'),
            'phi_rank_ratio': metrics.get('phi/rank_ratio'),
            'phi_top1_variance_fraction': metrics.get('phi/top1_variance_fraction'),
            'phi_top5_variance_fraction': metrics.get('phi/top5_variance_fraction'),
            'phi_spectral_gap': metrics.get('phi/spectral_gap'),
            'phi_frobenius_norm': metrics.get('phi/frobenius_norm'),
            'phi_mean_token_norm': metrics.get('phi/mean_token_norm'),
            'phi_std_token_norm': metrics.get('phi/std_token_norm'),

            # VFE gradient decomposition
            'vfe_grad_mu_self': metrics.get('vfe/grad_mu_self'),
            'vfe_grad_mu_direct': metrics.get('vfe/grad_mu_direct'),
            'vfe_grad_mu_softmax': metrics.get('vfe/grad_mu_softmax'),
            'vfe_grad_mu_total': metrics.get('vfe/grad_mu_total'),
            'vfe_grad_sigma_self': metrics.get('vfe/grad_sigma_self'),
            'vfe_grad_sigma_align_direct': metrics.get('vfe/grad_sigma_align_direct'),
            'vfe_grad_sigma_softmax': metrics.get('vfe/grad_sigma_softmax'),
            'vfe_grad_sigma_total': metrics.get('vfe/grad_sigma_total'),
            'vfe_kl_pairwise_mean': metrics.get('vfe/kl_pairwise_mean'),
            'vfe_kl_pairwise_max': metrics.get('vfe/kl_pairwise_max'),
            'vfe_kappa_scaled': metrics.get('vfe/kappa_scaled'),
            'vfe_kl_frac_above_90pct': metrics.get('vfe/kl_frac_above_90pct'),
            'vfe_kl_p95': metrics.get('vfe/kl_p95'),

            # Covariance health
            'sigma_q_mean': metrics.get('cov/sigma_q_mean'),
            'sigma_q_min': metrics.get('cov/sigma_q_min'),
            'sigma_q_max': metrics.get('cov/sigma_q_max'),
            'sigma_q_std': metrics.get('cov/sigma_q_std'),
            'sigma_q_cond_mean': metrics.get('cov/sigma_q_cond_mean'),
            'sigma_q_cond_max': metrics.get('cov/sigma_q_cond_max'),
            'sigma_p_mean': metrics.get('cov/sigma_p_mean'),
            'sigma_p_min': metrics.get('cov/sigma_p_min'),
            'sigma_p_max': metrics.get('cov/sigma_p_max'),
            'prior_belief_kl_mean': metrics.get('cov/prior_belief_kl_mean'),
            'prior_belief_kl_max': metrics.get('cov/prior_belief_kl_max'),
            'prior_belief_kl_std': metrics.get('cov/prior_belief_kl_std'),

            # Transport & attention structure
            'phi_norm_mean': metrics.get('transport/phi_norm_mean'),
            'phi_norm_std': metrics.get('transport/phi_norm_std'),
            'phi_norm_max': metrics.get('transport/phi_norm_max'),
            'phi_pairwise_dist_mean': metrics.get('transport/phi_pairwise_dist_mean'),
            'phi_pairwise_dist_max': metrics.get('transport/phi_pairwise_dist_max'),
            'attn_entropy_per_head_mean': metrics.get('transport/attn_entropy_per_head_mean'),
            'attn_entropy_per_head_std': metrics.get('transport/attn_entropy_per_head_std'),
            'attn_entropy_per_head_min': metrics.get('transport/attn_entropy_per_head_min'),
            'attn_entropy_per_head_max': metrics.get('transport/attn_entropy_per_head_max'),
            'head_correlation_mean': metrics.get('transport/head_correlation_mean'),
        }

        self.history.append(entry)

    def log_val(self, step: int, val_metrics: Dict):
        """Update entry with validation metrics."""
        for entry in reversed(self.history):
            if entry['step'] == step:
                entry['val_loss'] = val_metrics.get('loss')
                entry['val_ce'] = val_metrics.get(
                    'ce_loss', val_metrics.get('loss'))
                entry['val_ppl'] = val_metrics.get('perplexity')
                entry['val_bpc'] = entry['val_ce'] / \
                    math.log(2) if entry['val_ce'] else None
                break

    def save(self):
        """Save to CSV."""
        if not self.history:
            return

        with open(self.save_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.headers)
            writer.writeheader()
            writer.writerows(self.history)


class LayerDiagnosticsTracker:
    """Write per-layer diagnostics to CSV (append mode, no in-memory accumulation)."""

    HEADERS = [
        'step', 'layer',
        'mu_input_norm', 'mu_output_norm', 'delta_mu_norm', 'delta_mu_relative',
        'sigma_mean_diag', 'phi_norm',
        'attention_entropy', 'kl_mean', 'kl_std',
        'mu_attn_norm', 'mu_ffn_norm', 'residual_ratio',
        'ce_loss', 'perplexity', 'mu_position_std',
    ]

    def __init__(self, save_path: Path):
        self.save_path = save_path
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.save_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.HEADERS)

    def log(self, diag: Dict):
        """Append a single row."""
        with open(self.save_path, 'a', newline='') as f:
            writer = csv.DictWriter(
                f, fieldnames=self.HEADERS, extrasaction='ignore')
            writer.writerow(diag)


class IterationDiagnosticsTracker:
    """Write per-VFE-iteration diagnostics to CSV (append mode)."""

    HEADERS = [
        'step', 'layer', 'iteration',
        'grad_mu_norm', 'grad_sigma_norm', 'nat_grad_mu_norm', 'nat_grad_mu_raw_norm',
        'delta_mu_norm', 'mu_norm', 'sigma_mean',
        'sigma_max', 'sigma_min', 'sigma_std',
        'effective_lr', 'scale_mean',
        'mu_diff_to_prior_norm', 'beta_entropy', 'mu_change_rel',
    ]

    def __init__(self, save_path: Path):
        self.save_path = save_path
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.save_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.HEADERS)

    def log(self, diag: Dict):
        """Append a single row."""
        with open(self.save_path, 'a', newline='') as f:
            writer = csv.DictWriter(
                f, fieldnames=self.HEADERS, extrasaction='ignore')
            writer.writerow(diag)


class PublicationTrainer(FastTrainer):
    """Enhanced trainer with publication-quality metrics."""

    @property
    def _model_blocks(self):
        """Resolve transformer blocks from either GaugeTransformerLM or HybridGaugeTransformerLM."""
        t = getattr(self.model, 'transformer', None)
        if t is not None:
            return t.blocks
        return getattr(self.model, 'blocks', [])

    def __init__(self, *args, publication_metrics: PublicationMetrics = None, tokenizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        _log = logger.debug if getattr(self.config, 'quiet', False) else logger.info

        # Basic CSV metrics tracker
        metrics_path = self.config.checkpoint_dir / 'metrics.csv'
        self.metrics_tracker = PublicationMetricsTracker(metrics_path)
        _log(f"Logging publication metrics to: {metrics_path}")

        # Comprehensive publication metrics (optional)
        self.pub_metrics = publication_metrics
        if self.pub_metrics:
            _log(f"Comprehensive metrics enabled: {self.pub_metrics.experiment_dir}")

        # Tokenizer for decoding sequences in interpretability outputs
        self.tokenizer = tokenizer

        # Track attention visualization count
        self._attention_viz_count = 0

        # Layer/iteration diagnostics trackers (gated by config flags)
        self.layer_tracker = None
        self.iter_tracker = None
        if getattr(self.config, 'track_layer_diagnostics', False):
            layer_path = self.config.checkpoint_dir / 'layer_diagnostics.csv'
            self.layer_tracker = LayerDiagnosticsTracker(layer_path)
            logger.info(f"Layer diagnostics enabled: {layer_path}")
        if getattr(self.config, 'track_iteration_diagnostics', False):
            iter_path = self.config.checkpoint_dir / 'iteration_diagnostics.csv'
            self.iter_tracker = IterationDiagnosticsTracker(iter_path)
            logger.info(f"Iteration diagnostics enabled: {iter_path}")

        # Enable VFE dynamics metrics collection on model and FFN modules.
        # This populates vfe_debug, transport_metrics, covariance_metrics
        # in forward_with_attention at negligible cost (no extra forward pass).
        if hasattr(self.model, '_collect_dynamics_metrics'):
            self.model._collect_dynamics_metrics = True
        for module in self.model.modules():
            if hasattr(module, '_collect_vfe_metrics'):
                module._collect_vfe_metrics = True
        _log("VFE dynamics metrics collection enabled")

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
                sym_dampening = getattr(
                    self.config, 'killing_form_sym_dampening', 0.1)
                self._cartan_preconditioner = build_cartan_projector(
                    generators, sym_dampening=sym_dampening
                ).to(self.device)
                _log(
                    f"Killing form preconditioning enabled (M-step): "
                    f"sym_dampening={sym_dampening} "
                    f"(non-compact directions dampened {1/sym_dampening:.0f}x)"
                )
                # Note: E-step phi preconditioning is now controlled by
                # phi_natural_gradient config ('clip'|'cartan'|'killing'|'pullback')
                # which flows through model → blocks → ffn → VariationalFFNDynamic.

            if use_slk:
                self._slk_trace_vec = build_slk_projector(
                    generators).to(self.device)
                trace_norm = self._slk_trace_vec.norm().item()
                _log(
                    f"SL(K) projection enabled: "
                    f"removing trace component (||v||={trace_norm:.2f}, "
                    f"n_gen={generators.shape[0]} -> {generators.shape[0]-1} effective d.o.f.)"
                )

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

    @torch.no_grad()
    def _compute_phi_diagnostics(self) -> Dict[str, float]:
        r"""Compute effective rank and spectral diagnostics of the phi embedding matrix.

        The phi embedding matrix \Phi \in \mathbb{R}^{V \times d_\phi} maps each
        vocabulary token to a Lie algebra coefficient vector. Its singular value
        spectrum reveals whether the gauge frames exploit the full algebra or
        collapse to a low-dimensional subspace (overfitting signature).

        Metrics returned:
            phi/effective_rank: exp(H(\hat\sigma)) where \hat\sigma_i = \sigma_i / \sum \sigma_j.
                Ranges from 1 (rank-1) to min(V, d_phi) (full rank). Values much
                smaller than d_phi indicate the frames live in a low-dimensional
                subspace and may be memorizing rather than learning structure.
            phi/rank_ratio: effective_rank / min(V, d_phi). Fraction of available
                dimensions actually used. < 0.3 is suspicious.
            phi/top1_variance_fraction: \sigma_1^2 / \sum \sigma_i^2. If close to 1,
                almost all phi variation is along a single direction.
            phi/top5_variance_fraction: \sum_{i=1}^{5} \sigma_i^2 / \sum \sigma_i^2.
            phi/spectral_gap: \sigma_1 / \sigma_2. Large gap means a dominant mode.
            phi/frobenius_norm: ||\Phi||_F. Tracks overall magnitude growth.
            phi/mean_token_norm: mean ||\phi_v||_2 across vocabulary.
            phi/std_token_norm: std of ||\phi_v||_2. Low std = uniform norms (healthy).
        """
        # Find the phi embedding weight
        # PriorBank takes priority: when active, token_embed is frozen but its
        # phi_embed attribute still exists — reading it gives stale values.
        phi_weight = None
        if hasattr(self.model, 'prior_bank') and self.model.prior_bank is not None:
            if hasattr(self.model.prior_bank, 'phi_embed'):
                phi_weight = self.model.prior_bank.phi_embed.weight
        if phi_weight is None and hasattr(self.model, 'token_embed') and hasattr(self.model.token_embed, 'phi_embed'):
            phi_weight = self.model.token_embed.phi_embed.weight  # (V, phi_dim)

        if phi_weight is None:
            return {}

        # (V, phi_dim) — work in float32 for numerical stability
        W = phi_weight.detach().float()
        V, d = W.shape

        # SVD (only need singular values for most metrics; compute thin U/S/V)
        # For large V, use the smaller dimension: SVD of W^T W or W W^T
        S = torch.linalg.svdvals(W)  # descending order, length = min(V, d)

        # Effective rank: exp(entropy of normalized singular values)
        # Roy & Bhattacharyya (2007): "Effective Rank"
        S_pos = S[S > 1e-12]  # filter numerical zeros
        p = S_pos / S_pos.sum()  # normalize to probability distribution
        entropy = -(p * p.log()).sum().item()
        effective_rank = math.exp(entropy)
        max_rank = min(V, d)
        rank_ratio = effective_rank / max_rank

        # Variance fractions (sigma^2 proportions)
        S_sq = S_pos ** 2
        total_var = S_sq.sum().item()
        top1_var_frac = (S_sq[0] / total_var).item() if total_var > 0 else 0.0
        top5_var_frac = (S_sq[:5].sum() / total_var).item() if total_var > 0 else 0.0

        # Spectral gap
        spectral_gap = (S[0] / S[1]).item() if len(S_pos) >= 2 else float('inf')

        # Norm statistics
        token_norms = W.norm(dim=1)  # (V,)
        frob_norm = W.norm().item()

        return {
            'phi/effective_rank': effective_rank,
            'phi/rank_ratio': rank_ratio,
            'phi/top1_variance_fraction': top1_var_frac,
            'phi/top5_variance_fraction': top5_var_frac,
            'phi/spectral_gap': spectral_gap,
            'phi/frobenius_norm': frob_norm,
            'phi/mean_token_norm': token_norms.mean().item(),
            'phi/std_token_norm': token_norms.std().item(),
        }

    def _format_vfe_dynamics(self, metrics: Dict) -> list:
        """Format VFE dynamics metrics for console output. Returns list of lines."""
        lines = []
        # VFE gradient decomposition
        _mu_self = metrics.get('vfe/grad_mu_self')
        _mu_dir = metrics.get('vfe/grad_mu_direct')
        _mu_sm = metrics.get('vfe/grad_mu_softmax')
        _mu_tot = metrics.get('vfe/grad_mu_total')
        if _mu_tot is not None:
            lines.append(
                f"  [VFE] grad_mu: self={_mu_self:.3e} align={_mu_dir:.3e} "
                f"softmax={_mu_sm:.3e} total={_mu_tot:.3e}"
            )
        _sig_self = metrics.get('vfe/grad_sigma_self')
        _sig_dir = metrics.get('vfe/grad_sigma_align_direct')
        _sig_tot = metrics.get('vfe/grad_sigma_total')
        if _sig_tot is not None:
            lines.append(
                f"  [VFE] grad_sig: self={_sig_self:.3e} align={_sig_dir:.3e} "
                f"total={_sig_tot:.3e}"
            )
        # Covariance health
        _sq_mean = metrics.get('cov/sigma_q_mean')
        _sq_cond = metrics.get('cov/sigma_q_cond_mean')
        _kl_pb = metrics.get('cov/prior_belief_kl_mean')
        if _sq_mean is not None:
            _cond_str = f" cond={_sq_cond:.1f}" if _sq_cond is not None else ""
            _kl_str = f" KL(q*||p)={_kl_pb:.2f}" if _kl_pb is not None else ""
            lines.append(f"  [COV] sigma_q: mean={_sq_mean:.4f}{_cond_str}{_kl_str}")
        # Attention structure
        _h_ent = metrics.get('transport/attn_entropy_per_head_mean')
        _h_corr = metrics.get('transport/head_correlation_mean')
        if _h_ent is not None:
            _corr_str = f" head_corr={_h_corr:.3f}" if _h_corr is not None else ""
            lines.append(f"  [ATTN] per-head entropy: mean={_h_ent:.3f}{_corr_str}")
        return lines

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
                plt.rcParams['font.sans-serif'] = [font_name] + \
                    plt.rcParams.get('font.sans-serif', [])
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
        dataset_name = getattr(self.train_loader.dataset,
                               'dataset_name', 'wikitext-2')
        if dataset_name == 'wiki-ja':
            self._setup_cjk_fonts(plt)

        self.model.eval()
        input_ids, target_ids = batch
        input_ids = input_ids.to(self.device)

        # Get attention from forward pass
        with torch.no_grad():
            if hasattr(self.model, 'forward_with_attention'):
                _, attn_info = self.model.forward_with_attention(
                    input_ids, targets=None)
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
                        beta = beta.unsqueeze(0).unsqueeze(
                            2)  # (1, B, 1, N, N)
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
                            decoded = self.tokenizer.decode(
                                input_ids[0].tolist())
                            preview = decoded[:80] + \
                                ('...' if len(decoded) > 80 else '')
                            seq_info = f"Step {step}, Text: {preview}"
                        except Exception:
                            seq_info = f"Step {step}, Tokens: { input_ids[0, :20].tolist()}..."
                    else:
                        seq_info = f"Step {step}, Tokens: { input_ids[0, :20].tolist()}..."

                    # Save directory
                    save_dir = self.config.checkpoint_dir / 'attention_patterns'
                    save_dir.mkdir(parents=True, exist_ok=True)

                    # ============================================================
                    # SAVE PER-LAYER, PER-HEAD VISUALIZATIONS (NOT AVERAGED!)
                    # ============================================================
                    for layer_idx in range(n_layers_actual):
                        # (n_heads, N, N)
                        beta_layer_np = beta[layer_idx, 0].cpu().numpy()

                        for head_idx in range(n_heads):
                            fig, ax = plt.subplots(figsize=(8, 6))

                            attn_head = beta_layer_np[head_idx]  # (N, N)
                            attn_plot = attn_head.copy()
                            # np.fill_diagonal(attn_plot, np.nan)  # Mask diagonal
                            attn_plot = np.log10(np.maximum(
                                attn_plot, 1e-5))  # Log scale

                            im = ax.imshow(attn_plot, cmap='viridis',
                                           aspect='auto', vmin=-5, vmax=0)
                            ax.set_xlabel('Key Position (j)')
                            ax.set_ylabel('Query Position (i)')

                            irrep_label = head_labels[head_idx] if head_idx < len(head_labels) else f"H{ head_idx}"
                            layer_label = f"L{ layer_idx}" if n_layers_actual > 1 else ""
                            title_prefix = f"{ layer_label} " if layer_label else ""
                            ax.set_title(
                                f'{title_prefix}Head { head_idx} ({irrep_label}) - {seq_info}',
                                fontsize=10,
                            )
                            plt.colorbar(
                                im, ax=ax, label='log\u2081\u2080(\u03b2)')

                            fig.savefig(
                                save_dir /
                                f'attention_step_{step:06d}_layer{ layer_idx}_head{head_idx}.png',
                                dpi=100, bbox_inches='tight',
                            )
                            plt.close(fig)

                    # ============================================================
                    # LOG INFO
                    # ============================================================
                    self._attention_viz_count += 1
                    if self._attention_viz_count == 1:
                        logger.info(f"Attention patterns saved to: {save_dir}/")
                        logger.info(
                            f"  Saving per-layer, per-head visualizations ({n_layers_actual} layers, {n_heads} heads)"
                        )

        self.model.train()

    def _run_forward_and_backward(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        is_standard: bool,
        use_delta_rule: bool,
        _tied_weights: bool,
        effective_beta: float,
    ) -> Tuple[Any, Dict]:
        r"""Run the forward pass, compute the VFE/CE loss, and call backward.

        For the standard transformer, returns a scalar CE loss wrapped in a
        minimal metrics dict. For gauge models, delegates to
        ``compute_free_energy_loss`` which computes

            F = CE + alpha*KL(q||p) + beta*KL(q||Omega*q) + ...

        When delta rule is active and embeddings are tied, the shared
        ``out_proj.weight`` gradient is zeroed after backward so that the
        delta rule is the sole W_out update.

        Returns:
            (loss, full_metrics): the scalar loss tensor and the raw metrics
            dict produced by ``compute_free_energy_loss`` (or the minimal
            CE-only dict for the standard model).
        """
        use_obs = getattr(self.config, 'use_obs_in_vfe', False)

        if is_standard:
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
                M_alpha=self.config.M_alpha,
                M_beta=effective_beta,
                lambda_gamma=self.config.lambda_gamma,
                kappa_gamma=self.config.kappa_gamma,
                lambda_hyper=self.config.lambda_hyper,
                pad_token_id=self.pad_token_id,
                use_obs_in_vfe=use_obs,
                mass_phi=self.config.mass_phi,
                aux_loss_weight=getattr(self.config, 'aux_loss_weight', 0.0) if getattr(self.config, 'aux_layer_loss', False) else 0.0,
                detach_beta_m_step=getattr(self.config, 'detach_beta_m_step', True),
                normalize_ce_by_dim=getattr(self.config, 'normalize_ce_by_dim', False),
            )

        # Scale loss for gradient accumulation (gradients accumulate across micro-batches)
        accum_steps = getattr(self.config, 'grad_accumulation_steps', 1)
        if accum_steps > 1:
            loss = loss / accum_steps
        loss.backward()

        # Tied weights + delta rule: zero out W_out gradient component.
        # With tie_embeddings, out_proj.weight IS mu_embed.weight. In non-amortized
        # mode mu_embed is detached from VFE, so any gradient on this shared tensor
        # comes purely from the output projection (logits = W_out @ mu_q). Zero it
        # so delta rule is the sole W_out update and P-flow is the sole mu update.
        if _tied_weights and self.model.out_proj.weight.grad is not None:
            self.model.out_proj.weight.grad.zero_()

        return loss, full_metrics

    def _apply_post_backward_projections(self) -> None:
        """Apply Killing form preconditioning, kappa clamping, and SL(K) projection.

        All three operations are post-backward, pre/post-optimizer manipulations
        that enforce gauge geometry constraints on model parameters.

        Killing form preconditioning (Cartan decomposition):
            Dampens the non-compact (symmetric) directions of gl(K) before
            gradient norm logging and clipping. This is the natural gradient
            on GL(K) — the Killing form metric assigns higher cost to
            non-compact directions.

        Kappa projection:
            Clamps ``log_kappa_per_head`` to [0.5, 1.5] × init. Without this
            the optimizer accumulates momentum in the dead zone above the
            forward-pass clamp.

        SL(K) projection:
            Removes the trace component from ``phi_embed``, projecting phi to
            the traceless subalgebra sl(K) so det(Omega_ij) = 1.
        """
        # NOTE: Cartan preconditioning is applied ONCE in train_step() (pre-optimizer,
        # lines ~1237-1245). Do NOT duplicate it here — with grad_accumulation_steps > 1,
        # non-accumulation steps skip zero_grad(), so a second application would square
        # the preconditioning and corrupt phi updates.

        # --- Kappa clamping (post-optimizer step) ---
        for block in self._model_blocks:
            for module in [getattr(block, 'attention', None), getattr(block, 'ffn', None)]:
                if module is None:
                    continue
                p = getattr(module, 'log_kappa_per_head', None)
                k0 = getattr(module, '_kappa_init', None)
                if p is not None and k0 is not None:
                    with torch.no_grad():
                        lo = torch.log(0.5 * k0)
                        hi = torch.log(1.5 * k0)
                        p.data.clamp_(min=lo, max=hi)

        # --- SL(K) projection ---
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

    def _apply_p_flow_and_delta_rule(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        full_metrics: Dict,
        is_standard: bool,
        use_delta_rule: bool,
    ) -> None:
        """Apply P-flow EMA update and delta rule W_out update.

        P-flow:
            EMA update of token embeddings (mu, sigma, optionally phi) toward
            beliefs that predicted successfully (low CE). Disabled for the
            standard transformer.

        Delta rule:
            Backprop-free Hebbian update of W_out:
                DeltaW = lr * (target - prediction) outer mu^T
            Combined with P-flow + detach_phi this makes learning fully
            backprop-free.
        """
        # --- P-FLOW ---
        use_p_flow = getattr(self.config, 'use_p_flow', False)
        if use_p_flow and not is_standard and 'p_flow/mu_q' in full_metrics:
            mu_beliefs = full_metrics['p_flow/mu_q']
            ce_per_position = full_metrics['p_flow/ce_per_position']
            ema_decay = getattr(self.config, 'p_flow_ema_decay', 0.99)

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

        # --- DELTA RULE ---
        if use_delta_rule and 'p_flow/mu_q' in full_metrics:
            mu_beliefs = full_metrics['p_flow/mu_q']
            delta_lr = getattr(self.config, 'delta_rule_lr', 0.1)
            if hasattr(self.model, 'delta_rule_update_w_out'):
                self.model.delta_rule_update_w_out(
                    mu_beliefs=mu_beliefs,
                    targets=target_ids,
                    lr=delta_lr,
                    pad_token_id=self.pad_token_id,
                )

    def _format_train_metrics(self, full_metrics: Dict, effective_beta: float) -> Dict:
        """Format the raw VFE metrics dict into the standard training metrics dict.

        Assembles the top-level training metrics (losses, perplexity, attention
        stats, scheduling) and carries over Bayesian alpha diagnostics, per-head
        kappa values, VFE gradient decomposition, covariance health, and
        transport/attention structure metrics.
        """
        metrics = {
            'train_loss_total': full_metrics['loss/total'],
            'train_loss_ce': full_metrics['loss/ce'],
            'train_loss_belief_align': full_metrics.get('loss/belief_align', 0),
            'train_loss_self_consistency': full_metrics.get('loss/self_consistency', 0),
            'train_loss_model_coupling': full_metrics.get('loss/model_coupling', 0),
            'train_loss_aux_layer_ce': full_metrics.get('loss/aux_layer_ce', 0),
            # PPL from raw (un-normalized) CE to avoid bogus values when normalize_ce_by_dim=True
            'train_ppl': math.exp(min(full_metrics.get('loss/ce_raw', full_metrics['loss/ce']), 20)),
            'beta_mean': full_metrics.get('attention/beta_mean', 0),
            'beta_std': 0,
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

        # Carry over per-head kappa diagnostics
        for key in ['kappa/per_head_mean', 'kappa/per_head_std',
                    'kappa/per_head_min', 'kappa/per_head_max']:
            if key in full_metrics:
                metrics[key] = full_metrics[key]
        # Per-head individual kappa values
        for key, val in full_metrics.items():
            if key.startswith('kappa/head_'):
                metrics[key] = val

        # Carry over VFE gradient decomposition, covariance health,
        # and transport/attention structure metrics for dashboard plots
        for key, val in full_metrics.items():
            if key.startswith(('vfe/', 'cov/', 'transport/')):
                metrics[key] = val

        return metrics

    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Train step with comprehensive metrics.

        Orchestrates four sub-operations in sequence:
        1. Forward pass + loss + backward (``_run_forward_and_backward``).
        2. Post-backward geometry projections (``_apply_post_backward_projections``).
        3. P-flow EMA and delta rule W_out updates (``_apply_p_flow_and_delta_rule``).
        4. Metrics formatting (``_format_train_metrics``).

        Also manages optimizer step, gradient clipping, scheduler step, and the
        optional layer/iteration diagnostic forward pass.
        """
        self.model.train()

        input_ids, target_ids = batch
        input_ids = input_ids.to(self.device)
        target_ids = target_ids.to(self.device)

        is_standard = isinstance(self.model, StandardTransformerLM)
        use_delta_rule = getattr(
            self.config, 'use_delta_rule_w_out', False) and not is_standard

        # When tie_embeddings=True, out_proj.weight IS mu_embed.weight. Do not
        # disable requires_grad on a tied tensor; zero the gradient post-backward.
        _tied_weights = (use_delta_rule and hasattr(self.model, 'out_proj')
                         and hasattr(self.model, 'token_embed')
                         and hasattr(self.model.token_embed, 'mu_embed')
                         and self.model.out_proj.weight is self.model.token_embed.mu_embed.weight)
        if use_delta_rule and hasattr(self.model, 'out_proj') and not _tied_weights:
            self.model.out_proj.weight.requires_grad = False

        effective_beta = self.config.M_beta

        # --- 1. Forward + backward ---
        _loss, full_metrics = self._run_forward_and_backward(
            input_ids, target_ids, is_standard, use_delta_rule, _tied_weights, effective_beta
        )

        # --- 2a. Pre-optimizer geometry: Killing form preconditioning ---
        if self._cartan_preconditioner is not None:
            from transformer.core.gauge_preconditioner import apply_cartan_preconditioning
            for name, param in self.model.named_parameters():
                if param.grad is not None and ('phi_embed' in name or 'phi' in name.lower()):
                    if param.grad.shape[-1] == self._cartan_preconditioner.shape[0]:
                        param.grad.data = apply_cartan_preconditioning(
                            param.grad.data, self._cartan_preconditioner
                        )

        # Compute gradient norms BEFORE clipping
        is_log_step = (self.global_step + 1) % self.config.log_interval == 0
        grad_norms = self._compute_gradient_norms() if is_log_step else None
        e_step_norms = self._collect_e_step_grad_norms() if is_log_step else None

        # Per-group clipping for large gauge groups (SO(N>3)):
        # phi_embed gradients dominate global norm, starving mu/sigma.
        # Only clip/step/zero on accumulation boundaries (every N micro-batches).
        _accum_steps = getattr(self.config, 'grad_accumulation_steps', 1)
        _is_accum_step = (self.global_step + 1) % _accum_steps == 0

        mstep_natural_norms = None
        if _is_accum_step:
            _use_param_groups = getattr(self.config, 'use_param_groups', True)
            if self.config.grad_clip > 0:
                if _use_param_groups:
                    # Scale clip threshold by sqrt(n_group / n_total) so that
                    # per-parameter gradient magnitude is equalized across groups.
                    # Without this, small groups (e.g., kappa) get a disproportionately
                    # large per-parameter gradient budget vs. large groups (phi_embed).
                    _total = sum(
                        p.numel() for g in self.optimizer.param_groups
                        for p in g['params'] if p.grad is not None
                    )
                    for group in self.optimizer.param_groups:
                        graded = [p for p in group['params'] if p.grad is not None]
                        if graded:
                            _n_group = sum(p.numel() for p in graded)
                            _scale = (_n_group / max(_total, 1)) ** 0.5
                            torch.nn.utils.clip_grad_norm_(
                                graded,
                                self.config.grad_clip * _scale,
                            )
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.grad_clip,
                    )
            self.optimizer.step()

            # Collect M-step natural gradient norms (after optimizer.step populates them)
            if is_log_step and hasattr(self.optimizer, 'get_grad_norms'):
                mstep_natural_norms = self.optimizer.get_grad_norms()

            if self.scheduler is not None:
                self.scheduler.step()
            self.optimizer.zero_grad()

        # --- 2b. Post-optimizer geometry: kappa clamp + SL(K) projection ---
        self._apply_post_backward_projections()

        # Re-enable requires_grad for W_out if it was disabled (non-tied case)
        if use_delta_rule and hasattr(self.model, 'out_proj') and not _tied_weights:
            self.model.out_proj.weight.requires_grad = True

        # --- 3. P-flow + delta rule ---
        self._apply_p_flow_and_delta_rule(
            input_ids, target_ids, full_metrics, is_standard, use_delta_rule
        )

        # --- 4. Format metrics ---
        metrics = self._format_train_metrics(full_metrics, effective_beta)

        # =================================================================
        # LAYER/ITERATION DIAGNOSTICS: Debug multi-layer/multi-iteration
        # =================================================================
        _diag_interval = getattr(self.config, 'diagnostics_interval', 50)
        _track_layers = self.layer_tracker is not None
        _track_iters = self.iter_tracker is not None

        if ((_track_layers or _track_iters)
                and not is_standard
                and (self.global_step + 1) % _diag_interval == 0):
            try:
                # Enable diagnostic flags on model/FFN
                if _track_iters:
                    for block in self._model_blocks:
                        if hasattr(block, 'ffn'):
                            block.ffn._collect_iteration_diagnostics = True
                            block.ffn._iteration_diagnostics = []
                if _track_layers:
                    self.model._collect_layer_diagnostics = True
                    self.model._layer_diagnostics = []

                # Diagnostic forward pass WITH grad enabled so phi evolves
                # across layers (phi update is gated by torch.is_grad_enabled()).
                # Zero grads afterward to avoid polluting the next train step.
                # Respect use_obs_in_vfe: only pass targets if training uses them,
                # so diagnostics measure the same E-step path as training.
                _diag_targets = target_ids if getattr(self.config, 'use_obs_in_vfe', False) else None
                self.model.forward_with_attention(
                    input_ids, targets=_diag_targets)

                # Zero any gradients accumulated during diagnostic pass
                self.model.zero_grad(set_to_none=True)

                # Write per-layer diagnostics
                if _track_layers and self.model._layer_diagnostics:
                    for ld in self.model._layer_diagnostics:
                        ld['step'] = self.global_step
                        self.layer_tracker.log(ld)

                # Write per-iteration diagnostics
                if _track_iters:
                    for layer_idx, block in enumerate(self._model_blocks):
                        if hasattr(block, 'ffn'):
                            for id_ in block.ffn._iteration_diagnostics:
                                id_['step'] = self.global_step
                                id_['layer'] = layer_idx
                                self.iter_tracker.log(id_)

                # Disable flags
                if _track_iters:
                    for block in self._model_blocks:
                        if hasattr(block, 'ffn'):
                            block.ffn._collect_iteration_diagnostics = False
                if _track_layers:
                    self.model._collect_layer_diagnostics = False
            except Exception as e:
                logger.warning(f"Layer/iteration diagnostics failed: {e}")

        return metrics, grad_norms, e_step_norms, mstep_natural_norms

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

    def _collect_e_step_grad_norms(self) -> Dict[str, float]:
        """Collect E-step natural gradient norms from FFN layers (last VFE iteration)."""
        norms = {
            'nat_grad_mu': 0.0, 'nat_grad_sigma': 0.0, 'grad_phi': 0.0,
            'nat_grad_mu_clipped': 0.0, 'nat_grad_sigma_clipped': 0.0,
        }
        n_layers = 0
        for module in self.model.modules():
            if hasattr(module, '_e_step_grad_norms'):
                for key in norms:
                    norms[key] += module._e_step_grad_norms.get(key, 0.0) ** 2
                n_layers += 1
        if n_layers > 0:
            for key in norms:
                norms[key] = math.sqrt(norms[key])
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

    def _set_kappa_frozen(self, frozen: bool):
        """Freeze or unfreeze all log_kappa_per_head parameters across layers."""
        for block in self._model_blocks:
            for module in [block.attention, block.ffn]:
                param = getattr(module, 'log_kappa_per_head', None)
                if param is not None:
                    param.requires_grad = not frozen

    def train(self):
        """Training loop with publication metrics."""
        _log = logger.debug if getattr(self.config, 'quiet', False) else logger.info
        _log("="*70)
        _log("PUBLICATION-QUALITY TRAINING")
        _log("="*70)

        # Support resuming from a checkpoint
        start_step = self.global_step
        if start_step > 0:
            logger.info(f"  Resuming from step {start_step}")

        start_time = time.time()
        train_iterator = iter(self.train_loader)

        # Calculate total steps: epochs takes precedence over max_steps
        epochs = getattr(self.config, 'epochs', None)
        if epochs is not None and epochs > 0:
            steps_per_epoch = len(self.train_loader)
            total_steps = epochs * steps_per_epoch
            logger.info(f"  Training for {epochs} epoch(s) ({steps_per_epoch} steps/epoch = {total_steps:,} total steps)")
        else:
            total_steps = self.config.max_steps
            steps_per_epoch = len(self.train_loader)
            equiv_epochs = total_steps / steps_per_epoch if steps_per_epoch > 0 else 0
            logger.info(f"  Training for {total_steps:,} steps (~{equiv_epochs:.1f} epochs)")

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

        _write = tqdm.write if use_tqdm else print

        # Run initial gauge frame semantic analysis (only if starting fresh)
        if start_step == 0 and self.pub_metrics:
            try:
                
                self.pub_metrics.run_semantic_analysis(
                    model=self.model,
                    step=0,
                    verbose=False,
                )
            except Exception as e:
                logger.warning(f"Initial semantic analysis failed: {e}")

        # Kappa warmup: freeze log_kappa_per_head until embeddings differentiate.
        # Without this, early uniform-attention gradients push kappa toward
        # values that permanently flatten attention.
        kappa_warmup = getattr(self.config, 'kappa_warmup_steps', 0)
        _kappa_frozen = False
        if kappa_warmup > 0:
            self._set_kappa_frozen(True)
            _kappa_frozen = True
            logger.info(f"  [kappa warmup] Frozen log_kappa_per_head for first {kappa_warmup} steps")

        for step in pbar:
            self.global_step = step
            step_start = time.time()

            # Unfreeze kappa after warmup period
            if _kappa_frozen and step >= kappa_warmup:
                self._set_kappa_frozen(False)
                _kappa_frozen = False
                logger.info(f"  [kappa warmup] Unfreezing log_kappa_per_head at step {step}")

            # Get batch
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(self.train_loader)
                batch = next(train_iterator)

            # Train step with full metrics (grad_norms computed inside before zero_grad)
            metrics, grad_norms, e_step_norms, mstep_natural_norms = self.train_step(batch)

            step_time = time.time() - step_start

            is_log_step = (step + 1) % self.config.log_interval == 0

            # Get learning rates
            lrs = {group['name']: group['lr']
                   for group in self.optimizer.param_groups}

            if is_log_step:
                # Flush numerical fallback counters and inject into metrics
                _num_events = _flush_numerical_events()
                for _nk, _nv in _num_events.items():
                    metrics[f'num/{_nk}'] = _nv

                # Phi embedding spectral diagnostics (effective rank, variance fractions)
                phi_diag = self._compute_phi_diagnostics()
                metrics.update(phi_diag)

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
                        e_step_norms=e_step_norms,
                        mstep_natural_norms=mstep_natural_norms,
                    )

                # Console logging
                log_msg = (
                    f"Step {step+1}/{total_steps} | "
                    f"Loss: {metrics['train_loss_total']:.4f} | "
                    f"CE: {metrics['train_loss_ce']:.4f} | "
                    f"β: {metrics['train_loss_belief_align']:.4f} | "
                    f"PPL: {metrics['train_ppl']:.1f}"
                )
                # Append per-head kappa range if learnable
                if 'kappa/per_head_mean' in metrics:
                    log_msg += (
                        f" | κ: [{metrics['kappa/per_head_min']:.2f}"
                        f"-{metrics['kappa/per_head_max']:.2f}]"
                    )

                _verbose = self.config.verbose_diagnostics
                if use_tqdm:
                    pbar.set_description(log_msg)
                    tqdm.write(log_msg)
                    if _verbose and grad_norms:
                        tqdm.write(f"\n\n  [M-STEP] total: {grad_norms['total']:.3e} | "
                                   f"mu: {grad_norms['mu']:.3e} | sigma: { grad_norms['sigma']:.3e} | "
                                   f"phi: {grad_norms['phi']:.3e}")
                    if _verbose and e_step_norms:
                        _mu_cap = e_step_norms.get('mu_cap_frac', 0.0) * 100
                        _sig_cap = e_step_norms.get('sigma_cap_frac', 0.0) * 100
                        _mu_tr = e_step_norms.get('mu_trust_frac', 0.0) * 100
                        _wh_mean = e_step_norms.get('whitened_mu_mean', 0.0)
                        _wh_max = e_step_norms.get('whitened_mu_max', 0.0)
                        tqdm.write(f"\n  [E-STEP] nat_mu: {e_step_norms['nat_grad_mu']:.3e} (cap: {_mu_cap:.0f}%) | "
                                   f"nat_sig: {e_step_norms['nat_grad_sigma']:.3e} (cap: {_sig_cap:.0f}%) | "
                                   f"phi: {e_step_norms['grad_phi']:.3e} | "
                                   f"trust: {_mu_tr:.0f}% (wh: {_wh_mean:.3f}/{_wh_max:.3f})\n")
                    if _verbose and phi_diag:
                        _erank = phi_diag['phi/effective_rank']
                        _rratio = phi_diag['phi/rank_ratio']
                        _top1 = phi_diag['phi/top1_variance_fraction']
                        _top5 = phi_diag['phi/top5_variance_fraction']
                        _sgap = phi_diag['phi/spectral_gap']
                        _mnorm = phi_diag['phi/mean_token_norm']
                        tqdm.write(f"  [PHI] eff_rank: {_erank:.1f} ({_rratio:.1%} of max) | "
                                   f"top1σ²: {_top1:.1%} top5σ²: {_top5:.1%} | "
                                   f"gap: {_sgap:.2f} | ||φ||: {_mnorm:.3f}")
                    if _verbose:
                        _vfe_lines = self._format_vfe_dynamics(metrics)
                        for _vl in _vfe_lines:
                            tqdm.write(_vl)
                else:
                    logger.info(log_msg)
                    if _verbose and grad_norms:
                        logger.info(
                            f"\n\n  [M-STEP] total: {grad_norms['total']:.3e} | "
                            f"mu: {grad_norms['mu']:.3e} | sigma: {grad_norms['sigma']:.3e} | "
                            f"phi: {grad_norms['phi']:.3e}"
                        )
                    if _verbose and e_step_norms:
                        _mu_cap = e_step_norms.get('mu_cap_frac', 0.0) * 100
                        _sig_cap = e_step_norms.get('sigma_cap_frac', 0.0) * 100
                        _mu_tr = e_step_norms.get('mu_trust_frac', 0.0) * 100
                        _wh_mean = e_step_norms.get('whitened_mu_mean', 0.0)
                        _wh_max = e_step_norms.get('whitened_mu_max', 0.0)
                        logger.info(
                            f"  [E-STEP] nat_mu: {e_step_norms['nat_grad_mu']:.3e} (cap: {_mu_cap:.0f}%) | "
                            f"nat_sig: {e_step_norms['nat_grad_sigma']:.3e} (cap: {_sig_cap:.0f}%) | "
                            f"phi: {e_step_norms['grad_phi']:.3e} | "
                            f"trust: {_mu_tr:.0f}% (wh: {_wh_mean:.3f}/{_wh_max:.3f})"
                        )
                    if _verbose and phi_diag:
                        _erank = phi_diag['phi/effective_rank']
                        _rratio = phi_diag['phi/rank_ratio']
                        _top1 = phi_diag['phi/top1_variance_fraction']
                        _top5 = phi_diag['phi/top5_variance_fraction']
                        _sgap = phi_diag['phi/spectral_gap']
                        _mnorm = phi_diag['phi/mean_token_norm']
                        logger.info(
                            f"  [PHI] eff_rank: {_erank:.1f} ({_rratio:.1%} of max) | "
                            f"top1s^2: {_top1:.1%} top5s^2: {_top5:.1%} | "
                            f"gap: {_sgap:.2f} | ||phi||: {_mnorm:.3f}"
                        )
                    if _verbose:
                        _vfe_lines = self._format_vfe_dynamics(metrics)
                        for _vl in _vfe_lines:
                            logger.info(_vl)

                # Report numerical fallback counters if any fired
                if _num_events:
                    _num_msg = "  [NUM] " + " | ".join(
                        f"{k}: {v}" for k, v in sorted(_num_events.items())
                    )
                    if use_tqdm:
                        tqdm.write(_num_msg)
                    else:
                        logger.info(_num_msg)

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

                _write(f"\n\n  Validation @ step {step+1}:")
                _write(f"    Loss: {val_metrics['loss']:.4f}")
                _write(f"    CE: {val_metrics['ce_loss']:.4f}")
                _write(f"    PPL: {val_metrics['perplexity']:.2f}")
                _write(f"    BPC: {val_metrics['ce_loss']/math.log(2):.3f}")
                _write(f"    Attn entropy: {attn_entropy:.3f} | concentration: {attn_concentration:.3f}\n\n")

                # Generate sample text to verify learning (varied prompts for diversity)
                try:
                    import random
                    # Use language-appropriate prompts
                    dataset_name = getattr(
                        self.train_loader.dataset, 'dataset_name', 'wikitext-2')
                    if dataset_name == 'wiki-ja':
                        prompts = ["日本", "東京", "世界", "歴史", "文化", "科学", "政治", "経済", "教育", "自然",
                                   "社会", "技術", "音楽", "映画", "大学"]
                    else:
                        prompts = ["The", "In", "A", "It", "This", "As", "Fuck", "When", "For",
                                   "After", "Before", "During", "While", "Although", "However"]
                    prompt = random.choice(prompts)
                    # Use temperature 0.9 and lower top_k for more diversity
                    sample = self.sample_text(
                        prompt=prompt, max_new_tokens=30, temperature=0.9, top_k=30)
                    _write(f"\n\n    Sample: {sample[:100]}...\n")
                except Exception as e:
                    import traceback
                    _write(f"    Sample generation failed: {e}")
                    traceback.print_exc()

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
                                entry['holonomy_mean_norm'] = holonomy_dict.get(
                                    'holonomy/mean_norm')
                                entry['holonomy_max_norm'] = holonomy_dict.get(
                                    'holonomy/max_norm')
                                entry['holonomy_frac_gt_01'] = holonomy_dict.get(
                                    'holonomy/frac_gt_0.1')
                                entry['holonomy_spectral_gap'] = holonomy_dict.get(
                                    'holonomy/spectral_gap')
                                entry['holonomy_wilson_trace'] = holonomy_dict.get(
                                    'holonomy/wilson_trace')
                                merged = True
                                break
                        if not merged:
                            logger.warning(
                                f"Holonomy at step {step+1}: no matching CSV entry "
                                f"(holonomy_interval may not be divisible by log_interval)"
                            )
                except Exception as e:
                    logger.warning(f"Holonomy computation failed at step {step+1}: {e}")

            # Lightweight semantic trajectory snapshot (higher frequency than full analysis)
            if self.pub_metrics:
                self.pub_metrics.maybe_record_semantic_trajectory(self.model, step + 1)

            # Periodic gauge frame semantic analysis (full: clustering, field coherence, omega, sigma)
            if self.pub_metrics and self.pub_metrics.should_run_semantic_analysis(step + 1):
                try:
                    self.pub_metrics.run_semantic_analysis(
                        model=self.model,
                        step=step + 1,
                        verbose=False,  # Minimal output during training
                    )
                except Exception as e:
                    logger.warning(f"Semantic analysis failed at step {step+1}: {e}")

        # Flush any remaining numerical events accumulated after the last log step
        _final_num_events = _flush_numerical_events()
        if _final_num_events:
            logger.info("  [NUM] Final: " + " | ".join(
                f"{k}: {v}" for k, v in sorted(_final_num_events.items())
            ))

        # Save final metrics
        self.metrics_tracker.save()
        logger.info(f"Final metrics saved to: {self.metrics_tracker.save_path}")

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
                logger.warning(f"Final semantic analysis failed: {e}")

            # Generate final holonomy figures (non-flat transport)
            if self.pub_metrics.holonomy_history:
                try:
                    logger.info("Generating holonomy figures...")
                    self.pub_metrics.generate_holonomy_figures(
                        model=self.model,
                        save_prefix='holonomy',
                    )
                except Exception as e:
                    logger.warning(f"Final holonomy figure generation failed: {e}")

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
                logger.warning(f"Could not generate interpretability outputs: {e}")
                logger.warning(f"  Traceback: {traceback.format_exc()}")

            self.pub_metrics.print_summary()

        # Final validation (ensures best_val_ce is updated even when
        # eval_interval > max_steps, e.g. in ablation sweeps)
        final_val = self.validate()
        if final_val['ce_loss'] < self.best_val_ce:
            self.best_val_ce = final_val['ce_loss']
            self.save_checkpoint(is_best=True)

        # Summary
        elapsed = time.time() - start_time
        logger.info("="*70)
        logger.info("TRAINING COMPLETE!")
        logger.info("="*70)
        logger.info(f"Time: {elapsed/3600:.2f} hours")
        logger.info(f"Best val CE: {self.best_val_ce:.4f} (PPL: {math.exp(min(self.best_val_ce, 20.0)):.2f})")
        logger.info("="*70)


def _create_dataloaders(config: dict):
    """Create train/val/test dataloaders and resolve the actual vocabulary size.

    Selects the tokenizer mode from ``config['tokenizer']`` ('char', 'bpe', or
    'auto'). In 'auto' mode, character-level tokenization is used when
    ``config['vocab_size'] <= 256``, BPE otherwise. Prints a one-line summary
    of the chosen tokenizer.

    Args:
        config: Training configuration dict. Requires keys: 'vocab_size',
            'max_seq_len', 'batch_size'. Optional keys: 'tokenizer',
            'num_workers', 'dataset'.

    Returns:
        Tuple of (train_loader, val_loader, test_loader, actual_vocab_size,
        tokenizer). For character-level tokenization, ``test_loader`` is None
        and ``tokenizer`` is None (create_char_dataloaders does not yet expose
        a test split or a tiktoken tokenizer object).
    """
    dataset_name = config.get('dataset', 'wikitext-2')

    tokenizer_mode = config.get('tokenizer', 'auto')
    if tokenizer_mode == 'auto':
        use_char = config['vocab_size'] <= 256
    else:
        use_char = (tokenizer_mode == 'char')

    if use_char:
        logger.info(f"Using CHARACTER-LEVEL tokenizer (vocab_size={config['vocab_size']})")
        # Note: create_char_dataloaders doesn't support test set yet
        train_loader, val_loader, actual_vocab_size = create_char_dataloaders(
            max_seq_len=config['max_seq_len'],
            batch_size=config['batch_size'],
            num_workers=config.get('num_workers', 0),
        )
        test_loader = None
        tokenizer = None
    else:
        logger.info(f"Using BPE tokenizer (vocab_size={config['vocab_size']})")
        train_loader, val_loader, test_loader, actual_vocab_size, tokenizer = create_dataloaders(
            max_seq_len=config['max_seq_len'],
            batch_size=config['batch_size'],
            vocab_size=config['vocab_size'],
            num_workers=config.get('num_workers', 0),
            dataset=dataset_name,
            include_test=True,
            return_tokenizer=True,
        )

    return train_loader, val_loader, test_loader, actual_vocab_size, tokenizer


def run_single_experiment(
    config: dict,
    ffn_mode: str,
    device: torch.device,
    checkpoint_dir: Path,
    args: argparse.Namespace = None,
    enable_publication_metrics: bool = True,
    quiet: bool = False,
) -> Dict:
    """
    Run a single training experiment.

    Args:
        config: Configuration dictionary
        ffn_mode: FFN mode ('VFE_dynamic')
        device: Device to train on
        checkpoint_dir: Directory to save checkpoints
        args: Command-line arguments for logging
        enable_publication_metrics: Whether to enable comprehensive publication metrics
        quiet: Suppress verbose banners/config dumps (for ablation sweeps)

    Returns:
        Dictionary with final metrics
    """
    # Use logger.debug for verbose blocks when quiet
    _info = logger.debug if quiet else logger.info

    _info("="*70)
    _info(f"EXPERIMENT: FFN_MODE = {ffn_mode}")
    _info("="*70)

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
    _info("="*70)
    _info(f"LOADING {dataset_name.upper()} DATA")
    _info("="*70)

    # Suppress dataset loading prints when quiet
    from transformer.data import datasets as _datasets_mod
    _prev_quiet = _datasets_mod.QUIET
    if quiet:
        _datasets_mod.QUIET = True
    try:
        train_loader, val_loader, test_loader, actual_vocab_size, tokenizer = _create_dataloaders(config)
    finally:
        _datasets_mod.QUIET = _prev_quiet
    use_char = tokenizer is None  # Character-level tokenizer returns None

    config['vocab_size'] = actual_vocab_size

    # =================================================================
    # Model Creation - Three distinct modes
    # =================================================================

    _info("="*70)
    _info("CREATING MODEL")
    _info("="*70)
    _info(f"  N (seq len): {config['max_seq_len']}")
    _info(f"  K (embed): {config['embed_dim']}")
    _info(f"  Layers: {config['n_layers']}")
    _info(f"  Vocab: {actual_vocab_size} ({'char' if use_char else 'BPE'})")

    # =====================================================================
    # MODE 1: STANDARD TRANSFORMER (baseline)
    # =====================================================================
    if ffn_mode == 'standard':
        _info("  Model type: STANDARD TRANSFORMER (dot-product attention)")
        _info("  - Attention: Q*K softmax")
        _info("  - FFN: Learned MLP (GELU)")
        _info("  - Output: Linear projection")
        _info("  - Learning: Backprop")
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
    # MODE 2: HYBRID GAUGE-ATTENTION TRANSFORMER
    # =====================================================================
    # KL-divergence attention + PriorBank encode/decode + standard GELU FFN.
    # Isolates the gauge attention contribution from VFE dynamics.
    elif ffn_mode == 'hybrid':
        from transformer.baselines.hybrid_gauge_transformer import HybridGaugeTransformerLM
        _pb = config.get('use_prior_bank', True)
        _info("  Model type: HYBRID GAUGE-ATTENTION TRANSFORMER")
        _info("  - Attention: KL-divergence based (gauge-theoretic)")
        _info("  - FFN: Standard GELU MLP")
        _info(f"  - Embeddings: {'PriorBank (KL encode/decode)' if _pb else 'nn.Embedding + nn.Linear'}")
        _info("  - Learning: Backprop")

        if 'kappa_beta' not in config:
            config['kappa_beta'] = 1.0
        _info(f"  kappa_beta: {config['kappa_beta']}")

        model = HybridGaugeTransformerLM(config)

    # =====================================================================
    # MODE 3: VFE_DYNAMIC TRANSFORMER (EM-step, uses backprop)
    # =====================================================================
    else:
        _info("  Model type: GAUGE VFE TRANSFORMER (KL-divergence attention)")
        _info("  - Attention: KL-divergence based")
        _info("  - FFN: VFE EM-step dynamics")
        _info("  - Output: Linear projection")
        _info("  - Learning: Backprop")
        _info("  - Position: None (emergent)")

        # kappa_beta: scalar sharpness dial (dimension scaling tau=2*sqrt(K) is hardcoded in attention)
        if 'kappa_beta' not in config:
            config['kappa_beta'] = 1.0
        _info(f"  kappa_beta: {config['kappa_beta']}")

        model = GaugeTransformerLM(config)

    model = model.to(device)

    # Enable E-step gradient component debug (prints per-component breakdown)
    # Set to True to diagnose gradient explosion sources; disable for production.
    # Reads from config dict (set by _DEBUG_VFE_GRADS in train_publication.py).
    if config.get('debug_vfe_grads', False):
        for module in model.modules():
            if hasattr(module, '_debug_vfe_gradients'):
                module._debug_vfe_gradients = True
        logger.debug("VFE gradient component debug ENABLED for all FFN layers")

    # Get parameter counts
    if hasattr(model, 'get_num_params'):
        total_params = model.get_num_params(non_embedding=False)
        non_embed_params = model.get_num_params(non_embedding=True)
    else:
        total_params = sum(p.numel() for p in model.parameters())
        non_embed_params = sum(
            p.numel() for name, p in model.named_parameters() if 'embed' not in name)

    _info("Model Parameters:")
    _info(f"  Total:         {total_params:,}")
    _info(f"  Non-embedding: {non_embed_params:,}")
    _info(f"  Embedding:     {total_params - non_embed_params:,}")

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
        gauge_irrep = config.get(
            'irrep_spec', [('fund', 1, config['embed_dim'])])
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
    _info("FLOPs Estimation (Peer Review M2e):")
    _info(f"  FLOPs/step:     {format_flops(step_flops)}")
    _info(f"  Total training: {format_flops(total_flops)}")
    if not is_standard:
        _info(
            f"  Key costs: mat_exp={format_flops(flops_result.get('transport_mat_exp', 0))}/layer, "
            f"KL={format_flops(flops_result.get('kl_divergence', 0))}/layer"
        )

    # =================================================================
    # Training Configuration
    # =================================================================

    train_config = TrainingConfig(
        epochs=config.get('epochs', None),
        max_steps=config['max_steps'],
        warmup_steps=config['warmup_steps'],

        # Learning rates
        # For standard transformer: attention_lr should match ffn_lr (all standard Adam)
        # For gauge transformer: attention_lr matches phi_lr (natural gradient scale)
        M_mu_p_lr=config.get('M_mu_p_lr', config.get('mu_lr', 0.1)),
        M_sigma_p_lr=config.get('M_sigma_p_lr', config.get('sigma_lr', 0.005)),
        M_phi_lr=config.get('M_phi_lr', config.get('phi_lr', 0.01)),
        M_attention_lr=config.get(
            'M_attention_lr', config.get('attention_lr',
                config.get('M_vfe_hyperparam_lr', config.get('ffn_lr', 0.001)) if ffn_mode == 'standard'
                else config.get('M_phi_lr', config.get('phi_lr', 0.01)))),
        M_vfe_hyperparam_lr=config.get('M_vfe_hyperparam_lr', config.get('ffn_lr', 0.001)),
        M_output_lr=config.get('M_output_lr', config.get('output_lr',
            config.get('M_vfe_hyperparam_lr', config.get('ffn_lr', 0.001)))),

        non_embed_weight_decay=config.get('non_embed_weight_decay', config.get('weight_decay', 0.01)),
        embed_weight_decay=config.get('embed_weight_decay', None),
        grad_clip=config['grad_clip'],

        # M-step optimizer type
        optimizer_type=config.get('optimizer_type', 'adamw'),
        fisher_ema_decay=config.get('fisher_ema_decay', 0.95),
        fisher_damping=config.get('fisher_damping', 1e-4),

        # Free energy loss weights
        M_alpha=config.get('M_alpha', config.get('alpha', 0.0)),
        # → M_beta in compute_free_energy_loss
        M_beta=config.get('M_beta', config.get('beta', 0.0)),

        lambda_gamma=config['lambda_gamma'],
        lambda_hyper=config.get('lambda_hyper', 0.0),
        use_obs_in_vfe=config.get('use_obs_in_vfe', False),

        # Multi-layer depth signal
        aux_layer_loss=config.get('aux_layer_loss', False),
        aux_loss_weight=config.get('aux_loss_weight', 0.3),
        sigma_residual=config.get('sigma_residual', False),

        # Gauge geometry: phi gradient control
        mass_phi=config.get('mass_phi', config.get('alpha_phi', 0.0)),
        use_slk_projection=config.get('use_slk_projection', False),
        use_killing_form=config.get('use_killing_form', False),
        killing_form_sym_dampening=config.get(
            'killing_form_sym_dampening', 0.1),

        log_interval=config['log_interval'],
        eval_interval=config['eval_interval'],
        checkpoint_interval=config['checkpoint_interval'],

        checkpoint_dir=exp_checkpoint_dir,

        # P-FLOW: EMA update of token embeddings toward successful beliefs
        use_p_flow=config.get('use_p_flow', False),
        p_flow_ema_decay=config.get('p_flow_ema_decay', 0.99),

        # DELTA RULE: Backprop-free learning for W_out
        use_delta_rule_w_out=config.get('use_delta_rule_w_out', False),
        delta_rule_lr=config.get('delta_rule_lr', 0.001),

        # Layer/iteration diagnostics
        track_layer_diagnostics=config.get('track_layer_diagnostics', False),
        track_iteration_diagnostics=config.get(
            'track_iteration_diagnostics', False),
        diagnostics_interval=config.get('diagnostics_interval', 50),
        verbose_diagnostics=config.get('verbose_diagnostics', True),

        # Learnable per-head kappa warmup
        kappa_warmup_steps=config.get('kappa_warmup_steps', 0),

        # Output verbosity
        quiet=quiet,
    )

    _info("="*70)
    _info("TRAINING CONFIGURATION")
    _info("="*70)
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
        _info(f"  Epochs:         {train_config.epochs}")
        _info(f"  Steps/epoch:    {steps_per_epoch:,}")
        _info(f"  Total steps:    {effective_steps:,}")
        _info(f"  Tokens seen:    {total_tokens:,} ({total_tokens/1e6:.1f}M)")
        if dataset_tokens:
            coverage = total_tokens / dataset_tokens * 100
            _info(f"  Dataset:        {dataset_tokens:,} ({dataset_tokens/1e6:.1f}M) - {coverage:.1f}% coverage")
    else:
        equiv_epochs = train_config.max_steps / steps_per_epoch
        total_tokens = train_config.max_steps * tokens_per_step
        _info(f"  Max steps:      {train_config.max_steps:,}")
        _info(f"  Steps/epoch:    {steps_per_epoch:,}")
        _info(f"  *** EPOCHS:     {equiv_epochs:.4f} ***")
        _info(f"  Tokens seen:    {total_tokens:,} ({total_tokens/1e6:.1f}M)")
        if dataset_tokens:
            coverage = total_tokens / dataset_tokens * 100
            _info(f"  Dataset:        {dataset_tokens:,} ({dataset_tokens/1e6:.1f}M) - {coverage:.1f}% coverage")
    _info(f"  Warmup:         {train_config.warmup_steps}")
    _info(f"  Batch size:     {batch_size}")
    _info(f"  Seq length:     {seq_len}")
    _info(f"  Num workers:    {config.get('num_workers', 0)}")
    _info("Free Energy Weights:")
    _info(f"  alpha (self-consistency): {train_config.M_alpha}")
    _info(f"  beta (belief align):      {train_config.M_beta}")
    _info(f"  gamma (model align):      {train_config.lambda_gamma}")

    # P-FLOW configuration
    if train_config.use_p_flow:
        _info("P-FLOW (EMA prior updates): ENABLED")
        _info(f"  EMA decay: {train_config.p_flow_ema_decay} ({(1-train_config.p_flow_ema_decay)*100:.1f}% update per step)")
    else:
        _info("P-FLOW: disabled")

    # DELTA RULE configuration
    if train_config.use_delta_rule_w_out:
        _info("DELTA RULE (backprop-free W_out): ENABLED")
        _info(f"  Learning rate: {train_config.delta_rule_lr}")
        if train_config.use_p_flow:
            _info("  ** FULLY BACKPROP-FREE MODE **")
    else:
        _info("DELTA RULE: disabled (using backprop for W_out)")

    # =================================================================
    # Create Trainer (Pure FEP or Standard)
    # =================================================================

    _info("="*70)
    _info("INITIALIZING TRAINER")
    _info("="*70)

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
        _info(f"Gauge frame semantic analysis every {semantic_interval} steps")

        # Configure holonomy diagnostics interval
        holonomy_interval = config.get('holonomy_interval', 500)
        holonomy_sample_size = config.get('holonomy_sample_size', 500)
        pub_metrics.set_holonomy_interval(
            holonomy_interval, holonomy_sample_size)
        _info(f"Holonomy diagnostics every {holonomy_interval} steps (sample_size={holonomy_sample_size})")

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

    # Print compact summary when quiet, full banners otherwise
    if quiet:
        _params_str = f"{total_params/1e6:.2f}M" if total_params >= 1e6 else f"{total_params/1e3:.1f}K"
        _steps = train_config.max_steps if (train_config.epochs is None or train_config.epochs <= 0) else train_config.epochs * steps_per_epoch
        logger.info(f"  [Experiment] {ffn_mode} | {_params_str} params | "
                     f"K={config['embed_dim']}, N={config['max_seq_len']}, L={config['n_layers']} | "
                     f"{_steps:,} steps")
    else:
        _info("="*70)
        _info("STARTING TRAINING")
        _info("="*70)
        _info(f"Device: {device}")
        _info(f"FFN mode: {ffn_mode}")
        # Show epochs-based info if set
        if train_config.epochs is not None and train_config.epochs > 0:
            eff_steps = train_config.epochs * steps_per_epoch
            _info(f"Epochs: {train_config.epochs} ({steps_per_epoch:,} steps/epoch = {eff_steps:,} total)")
        else:
            _info(f"Total steps: {train_config.max_steps:,}")
        _info("NOTE: First few batches may be slow (JIT compilation)")
        _info("="*70)

    try:
        trainer.train()

        _info("="*70)
        _info("TRAINING COMPLETE!")
        _info("="*70)

        # Final evaluation
        final_metrics = trainer.validate()

        # Update best_val_ce so the checkpoint (and summary) reflect the final result.
        # Periodic validation may not have run (eval_interval > max_steps in ablations),
        # leaving best_val_ce at inf despite a successful final eval.
        if final_metrics['ce_loss'] < trainer.best_val_ce:
            trainer.best_val_ce = final_metrics['ce_loss']

        _info("Final Validation Metrics:")
        _info(f"  Loss:       {final_metrics['loss']:.4f}")
        _info(f"  Perplexity: {final_metrics['perplexity']:.2f}")

        # vs random baseline
        random_ppl = actual_vocab_size
        improvement = random_ppl / final_metrics['perplexity']
        _info("Validation improvement over random:")
        _info(f"  Random:     {random_ppl:.0f}")
        _info(f"  Model:      {final_metrics['perplexity']:.2f}")
        _info(f"  Factor:     {improvement:.1f}x better!")

        # Save final checkpoint
        final_ckpt = trainer.save_checkpoint(is_best=True)
        _info(f"Saved: {final_ckpt}")

        # Generate VFE dynamics figures from training metrics CSV
        try:
            from transformer.visualization.vfe_dynamics_plots import generate_all_vfe_figures
            metrics_csv = exp_checkpoint_dir / 'metrics.csv'
            if metrics_csv.exists():
                vfe_fig_dir = exp_checkpoint_dir / 'vfe_dynamics_figures'
                saved_figs = generate_all_vfe_figures(metrics_csv, vfe_fig_dir)
                if saved_figs:
                    _info(f"Generated {len(saved_figs)} VFE dynamics figures in {vfe_fig_dir}")
        except Exception as e:
            logger.warning(f"VFE dynamics figure generation failed: {e}")

        # Generate per-head kappa plot if learnable_head_kappa was enabled
        try:
            from transformer.visualization.training_plots import plot_head_kappas, load_metrics_csv
            metrics_csv = exp_checkpoint_dir / 'metrics.csv'
            if metrics_csv.exists():
                csv_metrics = load_metrics_csv(metrics_csv)
                if csv_metrics.get('kappa_mean'):
                    kappa_fig_path = exp_checkpoint_dir / 'head_kappas.png'
                    plot_head_kappas(csv_metrics, kappa_fig_path)
        except Exception as e:
            logger.warning(f"Head kappa plot generation failed: {e}")

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
        logger.info("="*70)
        logger.info("TRAINING INTERRUPTED")
        logger.info("="*70)
        ckpt = trainer.save_checkpoint(is_best=False)
        logger.info(f"Saved: {ckpt}")
        return None

    except Exception as e:
        logger.error(f"Error: {e}")
        raise

    finally:
        # Drop references to large objects so GC can reclaim them.
        # Critical for ablation sweeps that run many experiments in one process.
        trainer = model = train_loader = val_loader = test_loader = None
        pub_metrics = tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


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


def _plot_pure_vfe_attention(model, val_loader, device, step, attn_dir, prefix):
    """Plot per-head attention heatmaps for one validation example.

    Saves one PNG per attention head in log10 scale to ``attn_dir``.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    try:
        batch = next(iter(val_loader))
        input_ids = batch[0][:1].to(device)  # Single example for viz
        logits, beta, _diag = model.forward_with_attention(input_ids)

        if beta is not None:
            B, H, N, N2 = beta.shape
            for h in range(H):
                fig, ax = plt.subplots(1, 1, figsize=(8, 6))
                attn = beta[0, h, :N, :N].cpu().float().numpy()
                # Log scale for better visibility
                attn_log = np.log10(np.maximum(attn, 1e-5))
                im = ax.imshow(attn_log, aspect='auto', cmap='viridis')
                ax.set_xlabel('Key position (j)')
                ax.set_ylabel('Query position (i)')
                ax.set_title(f'Attention head {h} (log10) — step {step}')
                plt.colorbar(im, ax=ax)
                fig.tight_layout()
                fig.savefig(attn_dir / f'{prefix}_head{h}.png', dpi=150)
                plt.close(fig)
    except Exception as e:
        logger.warning(f"Attention figure failed: {e}")


def _plot_pure_vfe_prior_stats(model, step, figures_dir, prefix):
    """Plot histograms of prior mean norms and covariance eigenvalues.

    Saves a three-panel figure showing ||mu_v||, lambda_min(Sigma_v), and
    lambda_max(Sigma_v) distributions across the first 500 vocabulary entries.
    """
    import matplotlib.pyplot as plt

    try:
        with torch.no_grad():
            mu_norms = model.prior_mu.norm(dim=-1).cpu().numpy()
            sig_eigs = torch.linalg.eigvalsh(model.prior_Sigma[:500])
            sig_min = sig_eigs[..., 0].cpu().numpy()
            sig_max = sig_eigs[..., -1].cpu().numpy()

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Prior mean norms
        axes[0].hist(mu_norms, bins=50, alpha=0.7, edgecolor='black')
        axes[0].set_xlabel(r'$\|\mu_v\|$')
        axes[0].set_ylabel('Count')
        axes[0].set_title(f'Prior Mean Norms (step {step})')

        # Min eigenvalues of prior Sigma
        axes[1].hist(sig_min, bins=50, alpha=0.7, color='orange', edgecolor='black')
        axes[1].set_xlabel(r'$\lambda_{\min}(\Sigma_v)$')
        axes[1].set_ylabel('Count')
        axes[1].set_title(f'Prior Covariance Min Eigenvalues')

        # Max eigenvalues
        axes[2].hist(sig_max, bins=50, alpha=0.7, color='green', edgecolor='black')
        axes[2].set_xlabel(r'$\lambda_{\max}(\Sigma_v)$')
        axes[2].set_ylabel('Count')
        axes[2].set_title(f'Prior Covariance Max Eigenvalues')

        fig.tight_layout()
        fig.savefig(figures_dir / f'{prefix}_prior_stats.png', dpi=150)
        plt.close(fig)
    except Exception as e:
        logger.warning(f"Prior stats figure failed: {e}")


def _plot_pure_vfe_gauge_health(model, step, figures_dir, prefix):
    """Plot condition numbers and log-determinants of gauge frame matrices.

    Saves a two-panel figure for the first 500 vocabulary frames, showing
    conditioning and log|det Omega_v| distributions.
    """
    import matplotlib.pyplot as plt

    try:
        with torch.no_grad():
            Om = model.prior_Omega[:500]
            conds = torch.linalg.cond(Om).cpu().numpy().flatten()
            _, dets_log = torch.linalg.slogdet(Om)
            dets = dets_log.cpu().numpy().flatten()

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        axes[0].hist(conds, bins=50, alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('Condition number')
        axes[0].set_title(f'Gauge Frame Conditioning (step {step})')

        axes[1].hist(dets, bins=50, alpha=0.7, color='purple', edgecolor='black')
        axes[1].set_xlabel(r'$\ln|\det \Omega_v|$')
        axes[1].set_title(f'Gauge Frame Log-Determinants')

        fig.tight_layout()
        fig.savefig(figures_dir / f'{prefix}_gauge_health.png', dpi=150)
        plt.close(fig)
    except Exception as e:
        logger.warning(f"Gauge health figure failed: {e}")


def _plot_pure_vfe_semantic_pca(model, figures_dir, prefix, tokenizer=None):
    """Plot a 2-D PCA projection of the prior mean embedding space.

    Uses the first 2000 vocabulary entries. When ``tokenizer`` is provided,
    annotates a small set of representative token indices.
    """
    import matplotlib.pyplot as plt

    try:
        from sklearn.decomposition import PCA

        with torch.no_grad():
            mu_embed = model.prior_mu.cpu().numpy()

        # PCA of prior means
        pca = PCA(n_components=2)
        mu_2d = pca.fit_transform(mu_embed[:2000])  # First 2000 tokens

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.scatter(mu_2d[:, 0], mu_2d[:, 1], s=1, alpha=0.3)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        ax.set_title('Prior Mean Embedding Space (PCA)')

        # Annotate a few tokens if tokenizer available
        if tokenizer is not None:
            for idx in [0, 1, 2, 3, 4, 5, 10, 100, 256, 500]:
                if idx < len(mu_2d):
                    try:
                        token_str = tokenizer.decode([idx])
                        ax.annotate(repr(token_str), (mu_2d[idx, 0], mu_2d[idx, 1]),
                                   fontsize=6, alpha=0.7)
                    except Exception:
                        pass

        fig.tight_layout()
        fig.savefig(figures_dir / f'{prefix}_semantic_pca.png', dpi=150)
        plt.close(fig)
    except Exception as e:
        logger.warning(f"Semantic PCA figure failed: {e}")


def _plot_pure_vfe_holonomy(model, val_loader, device, config):
    """Compute and print holonomy statistics for the final pure VFE model.

    Only executed when ``config.use_holonomy`` is True. Prints mean/max
    ||C - I||_F across triangles in the attention graph; does not save a
    figure (holonomy plotting is handled by the analysis module).
    """
    try:
        from transformer.pure_vfe.inference import _compute_holonomy

        with torch.no_grad():
            batch = next(iter(val_loader))
            input_ids = batch[0][:4].to(device)
            Omega_h = model.prior_Omega[input_ids].clone()

            holonomy = _compute_holonomy(
                Omega_h, config.n_heads, config.head_dim
            )

        logger.info("  Holonomy Analysis (final):")
        logger.info(f"    Mean ||C - I||_F: {holonomy['mean_norm']:.6f}")
        logger.info(f"    Max  ||C - I||_F: {holonomy['max_norm']:.6f}")
        logger.info(f"    Triangles:        {holonomy['n_triangles']}")
    except Exception as e:
        logger.warning(f"Holonomy analysis failed: {e}")


def _save_pure_vfe_figures(model, val_loader, device, step, attn_dir,
                           figures_dir, config, tokenizer=None, final=False):
    """Save diagnostic figures for pure VFE training.

    Orchestrates five figure types in sequence: attention heatmaps, prior
    mu/sigma statistics, gauge frame health, semantic PCA (final only), and
    holonomy analysis (final only, when config.use_holonomy is True). Each
    type is wrapped in its own try/except so a failure in one does not
    suppress the others.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt  # noqa: F401 — ensure backend is set
    except ImportError:
        return  # matplotlib not available

    prefix = "final" if final else f"step_{step:06d}"

    # 1. Attention heatmaps (every save)
    _plot_pure_vfe_attention(model, val_loader, device, step, attn_dir, prefix)

    # 2. Prior mu/Sigma statistics (every save)
    _plot_pure_vfe_prior_stats(model, step, figures_dir, prefix)

    # 3. Gauge frame condition numbers and log-determinants (every save)
    _plot_pure_vfe_gauge_health(model, step, figures_dir, prefix)

    # 4. Semantic PCA of prior mean embeddings (final save only)
    if final:
        _plot_pure_vfe_semantic_pca(model, figures_dir, prefix, tokenizer=tokenizer)

    # 5. Holonomy analysis (final save only, when enabled)
    if final and getattr(config, 'use_holonomy', False):
        _plot_pure_vfe_holonomy(model, val_loader, device, config)

    if final:
        logger.info(f"Final figures saved to: {figures_dir}")
    else:
        logger.info(f"  Figures saved at step {step}")


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
    logger.info("="*70)
    logger.info("EXPERIMENT: PURE VFE TRANSFORMER")
    logger.info("="*70)

    ffn_mode = 'pure_vfe'
    exp_checkpoint_dir = checkpoint_dir / f"ffn_{ffn_mode}"
    exp_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save experiment configuration
    save_experiment_config(config, ffn_mode, exp_checkpoint_dir, args)

    # =================================================================
    # Data Loading (same pipeline as other modes)
    # =================================================================
    dataset_name = config.get('dataset', 'wikitext-103')
    logger.info("="*70)
    logger.info(f"LOADING {dataset_name.upper()} DATA")
    logger.info("="*70)

    train_loader, val_loader, test_loader, actual_vocab_size, tokenizer = _create_dataloaders(config)
    use_char = tokenizer is None  # Character-level tokenizer returns None

    config['vocab_size'] = actual_vocab_size

    # =================================================================
    # Model Creation
    # =================================================================
    logger.info("="*70)
    logger.info("CREATING PURE VFE MODEL")
    logger.info("="*70)

    # Build PureVFEConfig from the config dict (only pass fields that PureVFEConfig accepts)
    import dataclasses
    vfe_field_names = {f.name for f in dataclasses.fields(PureVFEConfig)}
    vfe_kwargs = {k: v for k, v in config.items() if k in vfe_field_names}
    vfe_kwargs['device'] = str(device)
    pure_config = PureVFEConfig(**vfe_kwargs)

    model = PureVFETransformer(pure_config)
    params = model.param_count()
    total_params = params['total']

    logger.info(f"  K (belief_dim): {pure_config.belief_dim}")
    logger.info(f"  H (n_heads):    {pure_config.n_heads}")
    logger.info(f"  K_h (head_dim): {pure_config.head_dim}")
    logger.info(f"  N (seq len):    {pure_config.max_seq_len}")
    logger.info(f"  E-steps:        {pure_config.n_esteps}")
    logger.info(f"  mu_q_lr:        {pure_config.mu_q_lr}")
    logger.info(f"  sigma_q_lr:     {pure_config.sigma_q_lr}")
    logger.info(f"  phi_lr:         {pure_config.phi_lr}")
    logger.info(f"  mu_p_lr:        {pure_config.mu_p_lr}")
    logger.info(f"  sigma_p_lr:     {pure_config.sigma_p_lr}")
    logger.info(f"  gauge_param:    {pure_config.gauge_param}")
    logger.info(f"  Vocab:          {actual_vocab_size}")
    logger.info(f"Model Parameters: {total_params:,}")
    for k, v in params.items():
        if k != 'total':
            logger.info(f"  {k}: {v:,}")

    # Attempt to load CUDA kernels
    if pure_config.use_cuda_kernels and str(device) == 'cuda':
        try:
            from transformer.pure_vfe.cuda_ext import get_cuda_ext
            cuda_ext = get_cuda_ext()
            if cuda_ext:
                logger.info("[CUDA kernels active]")
            else:
                logger.info("[Falling back to PyTorch ops]")
        except Exception:
            logger.info("[CUDA kernels not available, using PyTorch ops]")

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

    logger.info("="*70)
    logger.info("TRAINING CONFIGURATION")
    logger.info("="*70)
    logger.info(f"  Max steps:      {max_steps:,}")
    logger.info(f"  Steps/epoch:    {steps_per_epoch:,}")
    equiv_epochs = max_steps / steps_per_epoch if steps_per_epoch > 0 else 0
    logger.info(f"  *** EPOCHS:     {equiv_epochs:.4f} ***")
    logger.info(f"  Tokens seen:    {total_tokens:,} ({total_tokens/1e6:.1f}M)")
    if dataset_tokens:
        coverage = total_tokens / dataset_tokens * 100
        logger.info(f"  Dataset:        {dataset_tokens:,} ({dataset_tokens/1e6:.1f}M) - {coverage:.1f}% coverage")
    logger.info(f"  Batch size:     {batch_size}")
    logger.info(f"  Seq length:     {seq_len}")
    logger.info("  No optimizer (natural gradient only)")

    # =================================================================
    # Metrics Tracker
    # =================================================================
    metrics_path = exp_checkpoint_dir / 'metrics.csv'
    metrics_tracker = PublicationMetricsTracker(metrics_path)
    logger.info(f"Logging metrics to: {metrics_path}")

    # =================================================================
    # Figure Directories
    # =================================================================
    figures_dir = exp_checkpoint_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    attn_dir = figures_dir / 'attention_patterns'
    attn_dir.mkdir(parents=True, exist_ok=True)
    figure_interval = config.get('figure_interval', 2000)

    # Propagate max_steps to model config for LR scheduling
    pure_config.max_steps = max_steps

    # Log new feature status
    logger.info("  Features enabled:")
    logger.info(f"    RoPE:          {pure_config.use_rope}")
    logger.info(f"    Adam M-step:   {pure_config.use_adam_m_step}")
    logger.info(f"    LR schedule:   {pure_config.lr_schedule} (warmup={pure_config.warmup_steps})")
    logger.info(f"    Grad accum:    {pure_config.grad_accum_steps}")
    logger.info(f"    Diagonal Sigma: {pure_config.diagonal_covariance}")
    logger.info(f"    LayerNorm:     {pure_config.use_layernorm}")
    logger.info(f"    Holonomy:      {pure_config.use_holonomy}")
    logger.info(f"    Figure save:   every {figure_interval} steps")

    # =================================================================
    # Training Loop
    # =================================================================
    logger.info("="*70)
    logger.info("STARTING PURE VFE TRAINING")
    logger.info("="*70)
    logger.info(f"  Device: {device}")
    logger.info("  No backprop -- E-step VFE descent + M-step natural gradient")
    logger.info("="*70)

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

    # Gradient accumulation setup
    grad_accum_steps = getattr(pure_config, 'grad_accum_steps', 1)
    if grad_accum_steps > 1:
        from transformer.pure_vfe.learning import (
            MStepAccumulator, apply_m_step_from_accumulated,
        )
        from transformer.pure_vfe.inference import e_step as pure_e_step
        accum = model.create_accumulator()
        logger.info(f"  Grad accum:  {grad_accum_steps} micro-batches per M-step")
        logger.info(f"  Effective batch: {batch_size * grad_accum_steps} x {seq_len}")

    try:
        for step in pbar:
            step_start = time.time()

            if grad_accum_steps > 1:
                # --- Accumulated M-step: K E-steps, one M-step ---
                accum.reset()
                vfe_history = []
                for _micro in range(grad_accum_steps):
                    try:
                        batch = next(train_iterator)
                    except StopIteration:
                        train_iterator = iter(train_loader)
                        batch = next(train_iterator)

                    input_ids, target_ids = batch
                    input_ids = input_ids.to(device)
                    target_ids = target_ids.to(device)

                    mu, Sigma, Omega, logits, vfe, _diag = pure_e_step(
                        input_ids, model, pure_config,
                    )
                    accum.accumulate(
                        input_ids, target_ids, mu, Sigma, Omega,
                        model, pure_config, logits=logits,
                    )
                    if _micro == 0:
                        vfe_history = vfe  # Track VFE from first micro-batch

                effective_lrs = model.get_effective_lrs()
                ce_loss = apply_m_step_from_accumulated(
                    accum, model, pure_config, effective_lrs=effective_lrs,
                )
                model.global_step += 1
            else:
                # --- Standard: single E-step + M-step ---
                try:
                    batch = next(train_iterator)
                except StopIteration:
                    train_iterator = iter(train_loader)
                    batch = next(train_iterator)

                input_ids, target_ids = batch
                input_ids = input_ids.to(device)
                target_ids = target_ids.to(device)

                logits, ce_loss, vfe_history, _diag = model.update(
                    input_ids, target_ids)

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

                # Use scheduled LRs
                lrs = model.get_effective_lrs()

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
                    logger.info(log_msg)

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
                            logger.warning(health_msg)

                    health = monitor_omega_health(
                        model.prior_Omega[:100], "prior_Omega")
                    if health['prior_Omega/cond_max'] > 100:
                        omega_msg = f"  [WARN] Omega cond number high: {health['prior_Omega/cond_max']:.1f}"
                        if use_tqdm:
                            tqdm.write(omega_msg)
                        else:
                            logger.warning(omega_msg)

                # Flush numerical events
                _num_events = _flush_numerical_events()
                if _num_events:
                    _num_msg = "  [NUM] " + " | ".join(
                        f"{k}: {v}" for k, v in sorted(_num_events.items())
                    )
                    if use_tqdm:
                        tqdm.write(_num_msg)
                    else:
                        logger.info(_num_msg)

            # Validation
            if (step + 1) % eval_interval == 0:
                val_metrics = _validate_pure_vfe(model, val_loader, device)
                metrics_tracker.log_val(step + 1, val_metrics)

                logger.info(f"\n  Validation @ step {step+1}:")
                logger.info(f"    Loss: {val_metrics['loss']:.4f}")
                logger.info(f"    PPL: {val_metrics['perplexity']:.2f}")
                logger.info(f"    BPC: {val_metrics['ce_loss']/math.log(2):.3f}")

                # Save best model
                if val_metrics['ce_loss'] < best_val_ce:
                    best_val_ce = val_metrics['ce_loss']
                    best_path = exp_checkpoint_dir / 'best_model.pt'
                    model.save(best_path)
                    logger.info(f"    Saved best model (CE={best_val_ce:.4f})")

            # Checkpointing
            if (step + 1) % checkpoint_interval == 0:
                ckpt_path = exp_checkpoint_dir / f'checkpoint_step_{step+1}.pt'
                model.save(ckpt_path)
                metrics_tracker.save()

            # =========================================================
            # Figure Saving (attention patterns, semantics, holonomy)
            # =========================================================
            if figure_interval > 0 and (step + 1) % figure_interval == 0:
                _save_pure_vfe_figures(
                    model, val_loader, device, step + 1, attn_dir,
                    figures_dir, pure_config,
                    tokenizer=tokenizer if not use_char else None,
                )

        # Save final metrics
        metrics_tracker.save()
        logger.info(f"Final metrics saved to: {metrics_path}")

        # Save final figures
        _save_pure_vfe_figures(
            model, val_loader, device, max_steps, attn_dir,
            figures_dir, pure_config,
            tokenizer=tokenizer if not use_char else None,
            final=True,
        )

        # Final evaluation
        logger.info("="*70)
        logger.info("TRAINING COMPLETE!")
        logger.info("="*70)

        elapsed = time.time() - start_time
        logger.info(f"Total time: {elapsed/60:.1f} minutes ({elapsed/3600:.2f} hours)")

        final_metrics = _validate_pure_vfe(model, val_loader, device)

        logger.info("Final Validation Metrics:")
        logger.info(f"  Loss:       {final_metrics['loss']:.4f}")
        logger.info(f"  Perplexity: {final_metrics['perplexity']:.2f}")

        random_ppl = actual_vocab_size
        improvement = random_ppl / final_metrics['perplexity']
        logger.info("Validation improvement over random:")
        logger.info(f"  Random:     {random_ppl:.0f}")
        logger.info(f"  Model:      {final_metrics['perplexity']:.2f}")
        logger.info(f"  Factor:     {improvement:.1f}x better!")

        # Save final checkpoint
        final_path = exp_checkpoint_dir / 'best_model.pt'
        model.save(final_path)
        logger.info(f"Saved: {final_path}")

        # Test set evaluation
        test_metrics = None
        if test_loader is not None:
            logger.info("="*70)
            logger.info("FINAL TEST SET EVALUATION")
            logger.info("="*70)
            test_val = _validate_pure_vfe(
                model, test_loader, device, max_samples=128000)
            test_bpc = test_val['ce_loss'] / math.log(2)
            test_improvement = random_ppl / test_val['perplexity']
            test_metrics = {
                'test_loss': test_val['loss'],
                'test_ppl': test_val['perplexity'],
                'test_bpc': test_bpc,
                'improvement': test_improvement,
            }
            logger.info(f"  Test Loss: {test_val['loss']:.4f}")
            logger.info(f"  Test PPL:  {test_val['perplexity']:.2f}")
            logger.info(f"  Test BPC:  {test_bpc:.3f}")
            logger.info(f"  vs random: {test_improvement:.1f}x better")

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
            'mu_q_lr': pure_config.mu_q_lr,
            'sigma_q_lr': pure_config.sigma_q_lr,
            'phi_lr': pure_config.phi_lr,
            'mu_p_lr': pure_config.mu_p_lr,
            'sigma_p_lr': pure_config.sigma_p_lr,
            'gauge_param': pure_config.gauge_param,
        }

        if test_metrics is not None:
            result['test_loss'] = test_metrics['test_loss']
            result['test_ppl'] = test_metrics['test_ppl']
            result['test_bpc'] = test_metrics['test_bpc']
            result['test_improvement'] = test_metrics['improvement']

        return result

    except KeyboardInterrupt:
        logger.info("="*70)
        logger.info("TRAINING INTERRUPTED")
        logger.info("="*70)
        ckpt_path = exp_checkpoint_dir / 'interrupted_model.pt'
        model.save(ckpt_path)
        logger.info(f"Saved: {ckpt_path}")
        metrics_tracker.save()
        return None

    except Exception as e:
        logger.error(f"Error: {e}")
        raise
