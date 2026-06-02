"""
Experiment Runner for Gauge VFE Transformer
=============================================

Training infrastructure: experiment execution, metrics tracking, evaluation,
and the PublicationTrainer class. This module contains all the machinery
that runs after config selection.

Extracted from train_publication.py to separate config (entry point) from
execution (this module).. The entry point sets configs and calls
run_single_experiment() from here.

Public API:
    - run_single_experiment()    — Run EM/standard experiment
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
import json
import csv
import time
import math
import subprocess
import platform
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def _maybe_set_epoch(dataset, epoch: int) -> None:
    """Call dataset.set_epoch(epoch) when present. No-op otherwise.

    Used at the start of the training loop and at each StopIteration epoch
    boundary to update per-epoch window offsets when stride-based windowing
    is active. Silently skipped for datasets that don't implement the method
    or when random_offset_per_epoch is False (internal no-op inside set_epoch).
    """
    fn = getattr(dataset, "set_epoch", None)
    if callable(fn):
        fn(int(epoch))

from transformer.core.model import GaugeTransformerLM
from transformer.baselines.standard_transformer import StandardTransformerLM
from transformer.baselines.flops_counter import (
    count_standard_transformer_flops,
    count_gauge_transformer_flops,
    format_flops,
)
from transformer.data import create_dataloaders, create_char_dataloaders
from transformer.train import compute_free_energy_loss
from transformer.training.train_fast import FastTrainer
from transformer.training.config import TrainingConfig
from transformer.analysis.publication_metrics import PublicationMetrics
from math_utils.numerical_monitor import record as _nr, flush as _flush_numerical_events


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
                    mass_phi=0.0,
                    normalize_ce_by_dim=config.get('normalize_ce_by_dim', False) if isinstance(config, dict) else getattr(config, 'normalize_ce_by_dim', False),
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
    # BPC = (CE_nats / ln 2) * tokens_per_char. The tokens-per-char ratio
    # comes from the test loader's dataset; without it the reported value is
    # bits-per-token, off by ~4x for GPT-2 BPE on English and ~10% for
    # cl100k_base on Japanese.
    from transformer.training.bpc import bpc_from_dataset
    test_bpc = bpc_from_dataset(test_ce, test_loader, fallback_key='run_test_evaluation')
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
    args: Optional[Any] = None,
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


# Tracker classes live in metrics_tracking.py; re-exported here for backward compatibility.
from transformer.training.metrics_tracking import (
    PublicationMetricsTracker,
    LayerDiagnosticsTracker,
    IterationDiagnosticsTracker,
)


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
        from transformer.training.bpc import tokens_per_char_from_dataset
        self.metrics_tracker = PublicationMetricsTracker(
            metrics_path,
            tokens_per_char=tokens_per_char_from_dataset(self.train_loader),
        )
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
        self._killing_metric_for_clip = None
        self._slk_trace_vec = None

        use_killing_form = getattr(self.config, 'use_killing_form', False)
        use_slk = getattr(self.config, 'use_slk_projection', False)

        # RiemannianAdamW applies the Killing-form inverse to phi/omega grads
        # internally (optimizer.py:_precondition_phi) and keeps its own Killing
        # metric for the Riemannian trust-region clip.  The external Cartan
        # projector + killing_metric_for_clip at the apply site below
        # (_run_optimizer_step) is dedup-skipped whenever the optimizer is
        # RAdamW, so building either here is wasted setup.  Skip the build.
        from transformer.training.optimizer import RiemannianAdamW as _RAdamW
        _optimizer_is_radamw = isinstance(self.optimizer, _RAdamW)

        if use_killing_form and _optimizer_is_radamw:
            _log(
                "use_killing_form=True is redundant under optimizer_type="
                "'riemannian_adam' (RAdamW applies Killing inverse internally). "
                "Skipping external Cartan projector build."
            )

        if (use_killing_form or use_slk) and hasattr(self.model, 'generators'):
            from transformer.core.gauge_preconditioner import (
                build_cartan_projector,
                build_slk_projector,
            )
            generators = self.model.generators  # (n_gen, K, K)

            if use_killing_form and not _optimizer_is_radamw:
                sym_dampening = getattr(
                    self.config, 'killing_form_sym_dampening', 0.1)
                self._cartan_preconditioner = build_cartan_projector(
                    generators, sym_dampening=sym_dampening
                ).to(self.device)
                # Store the Killing metric for Riemannian norm clipping.
                # When the optimizer is plain AdamW (not RiemannianAdamW),
                # we need this to clip phi gradients in the correct geometry.
                from transformer.core.gauge_preconditioner import build_killing_form_preconditioner
                _, self._killing_metric_for_clip = build_killing_form_preconditioner(
                    generators, return_both=True,
                )
                self._killing_metric_for_clip = self._killing_metric_for_clip.to(self.device)
                _log(
                    f"Killing form preconditioning enabled (M-step, non-RAdamW path): "
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
        irrep_spec = self.model.config.get('irrep_spec', getattr(self.config, 'irrep_spec', None))
        if irrep_spec is None:
            raise AttributeError("irrep_spec not found on model or training config")
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
        """Configure matplotlib for per-glyph CJK fallback in figure text.

        Uses matplotlib 3.6+ per-glyph fallback by setting font.family to a list
        ['serif', cjk_font] so ASCII renders via DejaVu Serif (publication
        quality) and CJK glyphs fall back to the first available CJK font.

        Re-applies on every call (no cache) because pub_style.set_pub_style()
        and analysis/semantics.py periodically run rcParams.update({'font.family':
        'serif'}) — a single-string family with no fallback path. A one-shot
        cached setup silently strands CJK rendering after the first
        semantic_analysis_interval tick.
        """
        import matplotlib.font_manager as fm
        cjk_fonts = [
            'MS Gothic', 'Yu Gothic', 'Meiryo',          # Windows Japanese
            'Microsoft YaHei', 'SimHei',                   # Windows Chinese
            'Noto Sans CJK JP', 'Noto Sans JP',           # Linux/cross-platform
            'IPAGothic', 'IPAexGothic', 'TakaoGothic',    # Linux Japanese
            'Hiragino Sans', 'Hiragino Kaku Gothic Pro',   # macOS Japanese
        ]
        available = {f.name for f in fm.fontManager.ttflist}
        cjk = next((f for f in cjk_fonts if f in available), None)
        if cjk is None:
            if not getattr(self, '_cjk_warnings_silenced', False):
                import warnings
                warnings.filterwarnings(
                    'ignore', message='Glyph .* missing from font')
                self._cjk_warnings_silenced = True
            return
        plt.rcParams['font.family'] = ['serif', cjk]
        plt.rcParams['axes.unicode_minus'] = False

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
                        except (ValueError, TypeError, AttributeError, UnicodeDecodeError):
                            # Decode is purely diagnostic for attention-pattern
                            # display; fall back to raw token IDs rather than
                            # interrupting the training loop.
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
                                im, ax=ax, label=r'$\log_{10}(\beta)$')

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
        effective_beta: float,
    ) -> Tuple[Any, Dict]:
        r"""Run the forward pass, compute the VFE/CE loss, and call backward.

        For the standard transformer, returns a scalar CE loss wrapped in a
        minimal metrics dict. For gauge models, delegates to
        ``compute_free_energy_loss`` which computes

            F = CE + alpha*KL(q||p) + beta*KL(q||Omega*q) + ...

        Returns:
            (loss, full_metrics): the scalar loss tensor and the raw metrics
            dict produced by ``compute_free_energy_loss`` (or the minimal
            CE-only dict for the standard model).
        """
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
                mass_phi=self.config.mass_phi,
                omega_det_penalty=getattr(self.config, 'omega_det_penalty', 0.0),
                detach_beta_m_step=getattr(self.config, 'detach_beta_m_step', True),
                normalize_ce_by_dim=getattr(self.config, 'normalize_ce_by_dim', False),
                ce_label_smoothing=getattr(self.config, 'ce_label_smoothing', 0.0),
            )

        # NaN/Inf guard: skip backward to prevent poisoning optimizer momentum
        if torch.isnan(loss) or torch.isinf(loss):
            _nr("loss_nan_skip")
            logger.warning(
                f"Step {self.global_step}: loss is {loss.item()}, skipping backward")
            self.optimizer.zero_grad(set_to_none=True)
            return loss, {'loss/total': float('nan'), 'loss/ce': float('nan')}

        # Scale loss for gradient accumulation (gradients accumulate across micro-batches)
        accum_steps = getattr(self.config, 'grad_accumulation_steps', 1)
        if accum_steps > 1:
            loss = loss / accum_steps
        loss.backward()

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
            'train_loss_ce_raw': full_metrics.get('loss/ce_raw', full_metrics['loss/ce']),
            'train_loss_belief_align': full_metrics.get('loss/belief_align', 0),
            'train_loss_self_consistency': full_metrics.get('loss/self_consistency', 0),
            'train_loss_model_coupling': full_metrics.get('loss/model_coupling', 0),
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

        Orchestrates the sub-operations in sequence:
        1. Forward pass + loss + backward (``_run_forward_and_backward``).
        2. Post-backward geometry projections (``_apply_post_backward_projections``).
        3. Metrics formatting (``_format_train_metrics``).

        Also manages optimizer step, gradient clipping, scheduler step, and the
        optional layer/iteration diagnostic forward pass.
        """
        self.model.train()

        input_ids, target_ids = batch
        input_ids = input_ids.to(self.device)
        target_ids = target_ids.to(self.device)

        is_standard = isinstance(self.model, StandardTransformerLM)

        effective_beta = self.config.M_beta

        # --- 1. Forward + backward ---
        _loss, full_metrics = self._run_forward_and_backward(
            input_ids, target_ids, is_standard, effective_beta
        )

        # --- 2a. Pre-optimizer geometry: Killing form preconditioning ---
        # Skip when RiemannianAdamW already applies the Killing-form inverse
        # internally — applying both Cartan projection AND Killing inverse
        # produces a geometrically meaningless double-preconditioning.
        if self._cartan_preconditioner is not None:
            from transformer.training.optimizer import RiemannianAdamW as _RAdamW
            _optimizer_handles_precond = (
                isinstance(self.optimizer, _RAdamW)
                and self.optimizer._killing_inv is not None
            )
            if not _optimizer_handles_precond:
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
            # RiemannianAdamW handles per-group Riemannian trust region clipping
            # internally (after preconditioning, in the correct metric). Skip
            # external Euclidean clipping to avoid double-clipping in the wrong geometry.
            from transformer.training.optimizer import RiemannianAdamW as _RAdamW
            _optimizer_handles_clip = isinstance(self.optimizer, _RAdamW) and self.optimizer._grad_clip > 0
            if self.config.grad_clip > 0 and not _optimizer_handles_clip:
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
                            group_name = group.get('name', '')
                            _n_group = sum(p.numel() for p in graded)
                            _scale = (_n_group / max(_total, 1)) ** 0.5
                            _clip_val = self.config.grad_clip * _scale

                            if ('phi' in group_name) and self._killing_metric_for_clip is not None:
                                # Riemannian norm clipping for preconditioned phi gradients.
                                # After Cartan preconditioning, phi grads live in the Killing metric space.
                                # Clipping in Euclidean norm is geometrically wrong — use the Killing metric:
                                #   ||ξ||_K = sqrt(Σ_v ξ_v^T K ξ_v)
                                # This mirrors RiemannianAdamW._clip_riemannian (optimizer.py:212-224).
                                K_metric = self._killing_metric_for_clip.to(
                                    graded[0].device, graded[0].dtype
                                )
                                sq_norm = 0.0
                                for p in graded:
                                    xi = p.grad
                                    sq_norm += (xi @ K_metric * xi).sum().item()
                                riem_norm = sq_norm ** 0.5
                                if riem_norm > _clip_val:
                                    scale = _clip_val / (riem_norm + 1e-8)
                                    for p in graded:
                                        p.grad.mul_(scale)
                            else:
                                torch.nn.utils.clip_grad_norm_(
                                    graded,
                                    _clip_val,
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
            self.optimizer.zero_grad(set_to_none=True)

        # --- 2b. Post-optimizer geometry: kappa clamp + SL(K) projection ---
        self._apply_post_backward_projections()

        # --- 3. Format metrics ---
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
                # `targets` is forwarded only for the outer auxiliary CE loss;
                # the E-step never sees targets.
                self.model.forward_with_attention(
                    input_ids, targets=target_ids)

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
        """Freeze or unfreeze all log_kappa_per_head parameters across layers.

        The FFN is the sole owner of log_kappa_per_head since the attention
        sublayer was removed (2026-06-01).
        """
        for block in self._model_blocks:
            param = getattr(block.ffn, 'log_kappa_per_head', None)
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
        _epoch_idx = 0
        _maybe_set_epoch(self.train_loader.dataset, _epoch_idx)
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
            # Only freeze if log_kappa_per_head actually exists (learnable_head_kappa=True)
            _has_kappa_param = any(
                getattr(m, 'log_kappa_per_head', None) is not None
                for block in self._model_blocks
                for m in [getattr(block, 'attention', None), getattr(block, 'ffn', None)]
                if m is not None
            )
            if _has_kappa_param:
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
                _epoch_idx += 1
                _maybe_set_epoch(self.train_loader.dataset, _epoch_idx)
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
                            'ce_loss_raw': metrics.get('train_loss_ce_raw', metrics['train_loss_ce']),
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
                    # Print above the progress bar (not in the description,
                    # which gets overwritten by the next bar refresh).
                    pbar.set_description("Training")
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
                from transformer.training.bpc import bpc_from_dataset
                _write(
                    f"    BPC: "
                    f"{bpc_from_dataset(val_metrics['ce_loss'], self.val_loader, fallback_key='val_log'):.3f}"
                )
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
                    elif dataset_name == 'wiki-en':
                        prompts = ["The history of", "In the early", "Founded in", "Born in",
                                   "The city of", "The Battle of", "The first", "During the",
                                   "Located in", "Known for", "The University of",
                                   "The species", "The novel", "The film", "The album"]
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

            # Periodic holonomy diagnostics (non-flat transport curvature).
            # Writes its own CSV inside compute_holonomy_diagnostics ->
            # _append_holonomy_csv_row; no merge into the training CSV.
            if self.pub_metrics and self.pub_metrics.should_compute_holonomy(step + 1):
                try:
                    self.pub_metrics.compute_holonomy_diagnostics(
                        model=self.model,
                        step=step + 1,
                        verbose=True,
                    )
                except Exception as e:
                    logger.warning(f"Holonomy computation failed at step {step+1}: {e}")

            # Periodic gauge geometry diagnostics (Dirichlet energy, gauge invariants)
            if self.pub_metrics and self.pub_metrics.should_compute_gauge_geometry(step + 1):
                try:
                    _batch_input = batch[0].to(self.device) if isinstance(batch, (list, tuple)) else batch.to(self.device)
                    gauge_dict = self.pub_metrics.compute_gauge_geometry_diagnostics(
                        model=self.model,
                        step=step + 1,
                        batch=_batch_input,
                        verbose=True,
                    )
                    if gauge_dict:
                        merged = False
                        for entry in reversed(self.metrics_tracker.history):
                            if entry['step'] == step + 1:
                                for k, v in gauge_dict.items():
                                    entry[k.replace('/', '_')] = v
                                merged = True
                                break
                        if not merged:
                            logger.warning(
                                f"Gauge geometry at step {step+1}: no matching CSV entry"
                            )
                except Exception as e:
                    logger.warning(f"Gauge geometry computation failed at step {step+1}: {e}")

            # Periodic fiber trajectory diagnostics (Fisher-Rao E-step geometry)
            if self.pub_metrics and self.pub_metrics.should_compute_fiber_trajectory(step + 1):
                try:
                    _batch_input = batch[0].to(self.device) if isinstance(batch, (list, tuple)) else batch.to(self.device)
                    fiber_dict = self.pub_metrics.compute_fiber_trajectory_diagnostics(
                        model=self.model,
                        step=step + 1,
                        batch=_batch_input,
                        verbose=True,
                    )
                    if fiber_dict:
                        merged = False
                        for entry in reversed(self.metrics_tracker.history):
                            if entry['step'] == step + 1:
                                for k, v in fiber_dict.items():
                                    entry[k.replace('/', '_')] = v
                                merged = True
                                break
                        if not merged:
                            logger.warning(
                                f"Fiber trajectory at step {step+1}: no matching CSV entry"
                            )
                except Exception as e:
                    logger.warning(f"Fiber trajectory computation failed at step {step+1}: {e}")

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

            # Generate final gauge geometry figures
            if self.pub_metrics.gauge_geometry_history:
                try:
                    logger.info("Generating gauge geometry figures...")
                    self.pub_metrics.generate_gauge_geometry_figures(
                        save_prefix='gauge_geometry',
                    )
                except Exception as e:
                    logger.warning(f"Final gauge geometry figure generation failed: {e}")

            # Generate final fiber trajectory figures
            if self.pub_metrics.fiber_trajectory_history:
                try:
                    logger.info("Generating fiber trajectory figures...")
                    self.pub_metrics.generate_fiber_trajectory_figures(
                        save_prefix='fiber_trajectory',
                    )
                except Exception as e:
                    logger.warning(f"Final fiber trajectory figure generation failed: {e}")

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
        # eval_interval > max_steps, e.g. in ablation sweeps).  Stash the
        # result on the trainer so callers (e.g. run_single_experiment) can
        # reuse it instead of running a second full val pass.
        final_val = self.validate()
        self.final_val_metrics = final_val
        if final_val['ce_loss'] < self.best_val_ce:
            self.best_val_ce = final_val['ce_loss']
            if not getattr(self.config, 'skip_best_checkpoint', False):
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
            stride=config.get('stride', None),
            random_offset_per_epoch=config.get('random_offset_per_epoch', False),
            eval_stride=config.get('eval_stride', None),
            base_epoch_seed=config.get('stride_base_seed', 0),
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
            stride=config.get('stride', None),
            random_offset_per_epoch=config.get('random_offset_per_epoch', False),
            eval_stride=config.get('eval_stride', None),
            base_epoch_seed=config.get('stride_base_seed', 0),
        )

    return train_loader, val_loader, test_loader, actual_vocab_size, tokenizer


def run_single_experiment(
    config: dict,
    ffn_mode: str,
    device: torch.device,
    checkpoint_dir: Path,
    args: Optional[Any] = None,
    enable_publication_metrics: bool = True,
    quiet: bool = False,
    skip_test_eval: bool = False,
    skip_post_training_viz: bool = False,
    preloaded_data: Optional[Tuple] = None,
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
        skip_test_eval: Skip test set evaluation (use for sweeps — compare on val,
            reserve test for final reporting only)
        skip_post_training_viz: Skip end-of-run visualization + checkpoint work
            (VFE dynamics figures, head kappa plot, ``best_model.pt`` resave,
            and the duplicate final validation pass). Use for ablation sweeps
            where only the returned metric matters.
        preloaded_data: Optional pre-built dataloader bundle
            ``(train_loader, val_loader, test_loader, actual_vocab_size, tokenizer)``.
            When provided, bypasses ``_create_dataloaders(config)`` entirely —
            use from ablation sweeps to avoid re-loading the token cache and
            respawning DataLoader workers on every parameter value.

    Returns:
        Dictionary with final metrics
    """
    # Use logger.debug for verbose blocks when quiet
    _info = logger.debug if quiet else logger.info

    # Override FFN mode in config
    config = config.copy()
    config['ffn_mode'] = ffn_mode

    # Create experiment-specific checkpoint directory
    exp_checkpoint_dir = checkpoint_dir / f"ffn_{ffn_mode}"
    exp_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save experiment configuration at the START
    save_experiment_config(config, ffn_mode, exp_checkpoint_dir, args)

    # =================================================================
    # Data Loading — suppress verbose per-split prints
    # =================================================================

    if preloaded_data is not None:
        # Reuse loaders built once at the sweep level — avoids re-loading the
        # token cache and (on Windows) respawning DataLoader workers on every
        # parameter value.
        train_loader, val_loader, test_loader, actual_vocab_size, tokenizer = preloaded_data
    else:
        from transformer.data import datasets as _datasets_mod
        _prev_quiet = _datasets_mod.QUIET
        _datasets_mod.QUIET = True
        try:
            train_loader, val_loader, test_loader, actual_vocab_size, tokenizer = _create_dataloaders(config)
        finally:
            _datasets_mod.QUIET = _prev_quiet
    use_char = tokenizer is None  # Character-level tokenizer returns None

    config['vocab_size'] = actual_vocab_size

    # =================================================================
    # Model Creation — suppress verbose module-level logger.info
    # =================================================================

    # Temporarily raise log level on core modules to suppress init chatter,
    # and suppress known UserWarnings from embeddings (diagonal_covariance).
    import warnings as _warnings
    _core_loggers = []
    for _mod_name in [
        'transformer.core.model', 'transformer.core.variational_ffn',
        'transformer.core.attention', 'transformer.core.embeddings',
        'transformer.core.blocks', 'transformer.core.prior_bank',
        'transformer.core.connection', 'transformer.core.gauge_utils',
    ]:
        _lg = logging.getLogger(_mod_name)
        _core_loggers.append((_lg, _lg.level))
        if not quiet:
            _lg.setLevel(logging.WARNING)

    with _warnings.catch_warnings():
        _warnings.filterwarnings('ignore', category=UserWarning,
                                 module=r'transformer\.core\.embeddings')
        if ffn_mode == 'standard':
            model_config = {
                'vocab_size': actual_vocab_size,
                'embed_dim': config['embed_dim'],
                'n_layers': config['n_layers'],
                'n_heads': config.get('n_heads', 1),
                'hidden_dim': config.get('hidden_dim', config['embed_dim'] * 4),
                'max_seq_len': config['max_seq_len'],
                'dropout': config.get('dropout', 0.1),
                'disable_ffn': config.get('disable_ffn', False),
                'use_rope': config.get('use_rope', False),
                'rope_base': config.get('rope_base', 10000.0),
                'tie_embeddings': config.get('tie_embeddings', True),
                'no_pos_encoding': config.get('use_positional_embedding', True) is False and not config.get('use_rope', False),
            }
            model = StandardTransformerLM(model_config)
            _model_type = 'standard'

        elif ffn_mode == 'hybrid':
            from transformer.baselines.hybrid_gauge_transformer import HybridGaugeTransformerLM
            if 'kappa_beta' not in config:
                config['kappa_beta'] = 1.0
            model = HybridGaugeTransformerLM(config)
            _model_type = 'hybrid'

        else:
            if 'kappa_beta' not in config:
                config['kappa_beta'] = 1.0
            model = GaugeTransformerLM(config)
            _model_type = 'gauge_vfe'

    # Restore core logger levels
    for _lg, _lvl in _core_loggers:
        _lg.setLevel(_lvl)

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

    # =================================================================
    # Training Configuration
    # =================================================================

    train_config = TrainingConfig(
        epochs=config.get('epochs', None),
        max_steps=config['max_steps'],
        warmup_steps=config['warmup_steps'],

        # Learning rates
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
        phi_optimizer_metric=config.get('phi_optimizer_metric', 'killing'),
        pullback_series_order=config.get('pullback_series_order', 6),
        fisher_ema_decay=config.get('fisher_ema_decay', 0.95),
        fisher_damping=config.get('fisher_damping', 1e-4),

        # Free energy loss weights
        M_alpha=config.get('M_alpha', config.get('alpha', 0.0)),
        M_beta=config.get('M_beta', config.get('beta', 0.0)),

        lambda_gamma=config['lambda_gamma'],
        kappa_gamma=config.get('kappa_gamma', 1.0),
        lambda_hyper=config.get('lambda_hyper', 0.0),

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

        # Layer/iteration diagnostics
        track_layer_diagnostics=config.get('track_layer_diagnostics', False),
        track_iteration_diagnostics=config.get(
            'track_iteration_diagnostics', False),
        diagnostics_interval=config.get('diagnostics_interval', 50),
        verbose_diagnostics=config.get('verbose_diagnostics', False),

        # Learnable per-head kappa warmup
        kappa_warmup_steps=config.get('kappa_warmup_steps', 0),

        # LR schedule (previously missing — lr_decay/min_lr_ratio were silently
        # dropped, causing TrainingConfig defaults to override config values)
        lr_decay=config.get('lr_decay', 'linear'),
        min_lr_ratio=config.get('min_lr_ratio', 0.1),

        # CE loss handling (previously missing — values from config dict were silently
        # dropped at the dict→TrainingConfig boundary, falling back to dataclass defaults)
        normalize_ce_by_dim=config.get('normalize_ce_by_dim', True),
        ce_label_smoothing=config.get('ce_label_smoothing', 0.0),
        detach_beta_m_step=config.get('detach_beta_m_step', True),

        # Gradient accumulation
        grad_accumulation_steps=config.get('grad_accumulation_steps', 1),

        # Suppress FastTrainer init banner — compact summary below covers it
        quiet=True,

        # Stride windowing (threaded through for introspection / forward-compat;
        # actual logic lives in transformer/data/datasets.py).
        stride=config.get('stride', None),
        random_offset_per_epoch=config.get('random_offset_per_epoch', False),
        eval_stride=config.get('eval_stride', None),
        stride_base_seed=config.get('stride_base_seed', 0),
    )

    # Calculate training duration metrics for summary
    steps_per_epoch = len(train_loader)
    tokens_per_step = batch_size * seq_len

    try:
        dataset_tokens = len(train_loader.dataset.tokens)
    except AttributeError:
        dataset_tokens = None

    if train_config.epochs is not None and train_config.epochs > 0:
        total_steps_eff = train_config.epochs * steps_per_epoch
        total_tokens = total_steps_eff * tokens_per_step
    else:
        total_steps_eff = train_config.max_steps
        total_tokens = total_steps_eff * tokens_per_step

    # =================================================================
    # Create Trainer
    # =================================================================

    pub_metrics = None
    if enable_publication_metrics:
        experiment_name = f"{ffn_mode}_{time.strftime('%Y%m%d_%H%M%S')}"
        from transformer.training.bpc import tokens_per_char_from_dataset
        pub_metrics = PublicationMetrics(
            experiment_name=experiment_name,
            base_dir=exp_checkpoint_dir / "publication_outputs",
            tokens_per_char=tokens_per_char_from_dataset(train_loader),
        )

        semantic_interval = config.get('semantic_analysis_interval',
                                       getattr(args, 'semantic_analysis_interval', 10000) if args else 10000)
        pub_metrics.set_semantic_analysis_interval(semantic_interval)

        holonomy_interval = config.get('holonomy_interval', 500)
        holonomy_sample_size = config.get('holonomy_sample_size', 500)
        pub_metrics.set_holonomy_interval(
            holonomy_interval, holonomy_sample_size)

        gauge_geometry_interval = config.get('gauge_geometry_interval', 2000)
        pub_metrics.set_gauge_geometry_interval(gauge_geometry_interval)

        fiber_trajectory_interval = config.get('fiber_trajectory_interval', 5000)
        pub_metrics.set_fiber_trajectory_interval(fiber_trajectory_interval)

    # Ablation-mode hint: when we're skipping post-training viz, also suppress
    # the final best_model.pt save inside trainer.train().  Stashed as a plain
    # attribute so TrainingConfig stays untouched for everyone else.
    train_config.skip_best_checkpoint = skip_post_training_viz

    trainer = PublicationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_config,
        device=device,
        publication_metrics=pub_metrics,
        tokenizer=tokenizer,
    )

    # =================================================================
    # Compact init summary (replaces scattered banners)
    # =================================================================
    _params_str = f"{total_params/1e6:.2f}M" if total_params >= 1e6 else f"{total_params/1e3:.1f}K"
    _dataset_name = config.get('dataset', 'wikitext-2').upper()

    _mode_labels = {
        'standard': 'Standard Transformer',
        'hybrid': 'Hybrid (KL-attn + MLP)',
        'gauge_vfe': 'Gauge VFE Transformer',
    }
    _mode_label = _mode_labels.get(_model_type, ffn_mode)

    _n_heads = config.get('irrep_spec', [('fund', 1, config['embed_dim'])])[0][1] if _model_type == 'gauge_vfe' else config.get('n_heads', 1)

    logger.info("=" * 70)
    logger.info(f"  {_mode_label} | {_params_str} params | {device}")
    logger.info(f"  K={config['embed_dim']}, N={config['max_seq_len']}, L={config['n_layers']}, "
                f"heads={_n_heads} | {_dataset_name} ({total_tokens/1e6:.0f}M tokens)")
    _coverage_str = ""
    if dataset_tokens and dataset_tokens > 0:
        _coverage_pct = total_tokens / dataset_tokens * 100
        _coverage_str = f" ~ {_coverage_pct:.0f}% {_dataset_name.lower()}"
    logger.info(f"  {total_steps_eff:,} steps | B={batch_size}{_coverage_str} | "
                f"FLOPs/step: {format_flops(step_flops)} | Total: {format_flops(total_flops)}")
    logger.info(f"  LR: mu={train_config.M_mu_p_lr}, sigma={train_config.M_sigma_p_lr}, "
                f"phi={train_config.M_phi_lr}, out={train_config.M_output_lr}")
    if _model_type == 'gauge_vfe':
        logger.info(f"  VFE weights: alpha={train_config.M_alpha}, beta={train_config.M_beta}, "
                     f"gamma={train_config.lambda_gamma} | kappa={config.get('kappa_beta', 1.0)}")
    # Non-default features worth noting
    _extras = []
    if config.get('use_killing_form', False):
        _extras.append("killing-form")
    if config.get('kappa_warmup_steps', 0) > 0:
        _extras.append(f"kappa-warmup={config['kappa_warmup_steps']}")
    if _extras:
        logger.info(f"  Features: {', '.join(_extras)}")
    logger.info(f"  Output: {exp_checkpoint_dir}")
    logger.info("=" * 70)

    try:
        trainer.train()

        _info("="*70)
        _info("TRAINING COMPLETE!")
        _info("="*70)

        # Final evaluation — reuse the validation that trainer.train() already
        # ran at its end (stashed on trainer.final_val_metrics).  Falling back
        # to a fresh validate() keeps behavior correct for any caller that
        # bypasses trainer.train().
        final_metrics = getattr(trainer, 'final_val_metrics', None)
        if final_metrics is None:
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

        # Save final checkpoint (skipped in ablation sweeps — the suite only
        # consumes final_ppl and never re-opens these files).
        if skip_post_training_viz:
            final_ckpt = exp_checkpoint_dir / 'best_model.pt'
        else:
            final_ckpt = trainer.save_checkpoint(is_best=True)
            _info(f"Saved: {final_ckpt}")

        # Post-training visualizations (VFE dynamics figures + head kappa
        # plot).  Generating 10+ matplotlib renders after every ablation run
        # is the single biggest between-run delay, so this whole block is
        # gated on skip_post_training_viz.
        if not skip_post_training_viz:
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

        # Run test set evaluation (skip during sweeps — compare on val only)
        test_metrics = None
        if test_loader is not None and not skip_test_eval:
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
        raise