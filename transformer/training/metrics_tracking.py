"""
Metrics Tracking for Gauge VFE Transformer Training
====================================================

Step-level CSV trackers extracted from experiment_runner.py.

Classes:
    PublicationMetricsTracker  — full per-step metric history (CSV)
    LayerDiagnosticsTracker    — per-layer diagnostic rows (append-mode CSV)
    IterationDiagnosticsTracker — per-VFE-iteration diagnostic rows (append-mode CSV)
"""

import csv
import math
import time
from pathlib import Path
from typing import Dict


class PublicationMetricsTracker:
    """Track ALL metrics needed for publication."""

    def __init__(self, save_path: Path, tokens_per_char: float = None):
        """
        Args:
            save_path: Path to write the metrics CSV.
            tokens_per_char: Average BPE tokens per source character. Used to
                convert per-token CE to true bits-per-character via
                ``BPC = (CE_nats / ln 2) * tokens_per_char``. When ``None``,
                BPC is reported as bits-per-token (off by ~4x for GPT-2 BPE
                on English; ~10% for cl100k_base on Japanese) and a one-time
                warning is logged. Read off the train_loader's dataset:
                ``train_loader.dataset.tokens_per_char``.
        """
        self.save_path = save_path
        self.tokens_per_char = tokens_per_char
        self.history = []

        # Create CSV with comprehensive headers
        self.save_path.parent.mkdir(parents=True, exist_ok=True)

        self.headers = [
            # Core
            'step', 'timestamp',

            # Losses
            'train_loss_total', 'train_loss_ce', 'train_loss_ce_raw',
            'train_loss_belief_align',
            'train_loss_self_consistency', 'train_loss_model_coupling',
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

            # Holonomy columns moved to a dedicated CSV
            # (PublicationMetrics.holonomy_csv_path).

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

        # Use raw (un-normalized) CE for BPC — normalized CE / log(2) is meaningless
        train_ce_raw = metrics.get('train_loss_ce_raw', metrics.get('train_loss_ce', 0))
        # BPC = (CE_nats / ln 2) * tokens_per_char. tokens_per_char comes from
        # the dataset; when unavailable we fall back to bits-per-token with a
        # one-time warning (see transformer/training/bpc.py).
        from transformer.training.bpc import compute_bpc
        train_bpc = compute_bpc(
            train_ce_raw, self.tokens_per_char, fallback_key='metrics_tracker_train',
        )

        entry = {
            'step': step,
            'timestamp': time.time(),

            # Losses
            'train_loss_total': metrics.get('train_loss_total'),
            'train_loss_ce': metrics.get('train_loss_ce'),
            'train_loss_ce_raw': train_ce_raw,
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

            # Holonomy is written to a dedicated CSV (see
            # PublicationMetrics.holonomy_csv_path) — not merged here.

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
                if 'decode_margin' in val_metrics:
                    entry['val_decode_margin'] = val_metrics['decode_margin']
                from transformer.training.bpc import compute_bpc
                entry['val_bpc'] = (
                    compute_bpc(entry['val_ce'], self.tokens_per_char,
                                fallback_key='metrics_tracker_val')
                    if entry['val_ce'] else None
                )
                break

    def save(self):
        """Save to CSV.

        Dynamically discovers extra columns (e.g. gauge_*, fiber_*, holonomy_*)
        that periodic diagnostics merged into history entries after the header
        list was frozen at init.
        """
        if not self.history:
            return

        # Collect any extra keys that diagnostics added at runtime
        extra_keys: list = []
        _header_set = set(self.headers)
        for entry in self.history:
            for k in entry:
                if k not in _header_set:
                    _header_set.add(k)
                    extra_keys.append(k)
        all_headers = self.headers + sorted(extra_keys)

        with open(self.save_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_headers)
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
