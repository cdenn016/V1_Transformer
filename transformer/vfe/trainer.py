"""
VFETrainer: training loop with full metrics, diagnostics, and publication output.

    E-step: model.forward(token_ids) — infer q* from context only (no target leak)
    M-step: loss.backward() — gradients flow through E-step via semi-gradient backprop

Reuses PublicationMetricsTracker (CSV) and TrainingTracker/PublicationFigures
from the legacy training infrastructure for full diagnostic parity.
"""

import math
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformer.vfe.config import VFEConfig
from transformer.vfe.model import VFEModel

logger = logging.getLogger(__name__)


def _format_duration(secs: float) -> str:
    """Compact human-readable duration: '42s', '5m23s', '1h05m', '2d03h'."""
    secs = max(0.0, float(secs))
    if secs < 60.0:
        return f"{secs:.0f}s"
    if secs < 3600.0:
        m = int(secs // 60)
        s = int(secs - 60 * m)
        return f"{m}m{s:02d}s"
    if secs < 86400.0:
        h = int(secs // 3600)
        m = int((secs - 3600 * h) // 60)
        return f"{h}h{m:02d}m"
    d = int(secs // 86400)
    h = int((secs - 86400 * d) // 3600)
    return f"{d}d{h:02d}h"


class VFETrainer:
    r"""Training loop for VFEModel with full metrics and diagnostics.

    Implements the clean E-step/M-step split with comprehensive logging:
        E-step: forward pass infers beliefs from context only
        M-step: backprop through E-step to update priors/embeddings

    Args:
        model: VFEModel instance.
        cfg: VFEConfig with training hyperparameters.
        train_loader: Training data loader yielding (input_ids, target_ids).
        val_loader: Optional validation data loader.
        device: Device to train on.
        output_dir: Directory for metrics CSV, checkpoints, and figures.
    """

    def __init__(
        self,
        model: VFEModel,
        cfg: VFEConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: str = 'cpu',
        output_dir: Optional[str] = None,
    ) -> None:
        self.model = model.to(device)
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Optimizer with per-type learning rates
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        self.global_step = 0

        # Output directory for metrics, checkpoints, figures
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # Metrics infrastructure (lazy init in train())
        self._metrics_tracker = None
        self._pub_tracker = None

        # Attention-plot scratch space — captured in evaluate(), consumed
        # in _plot_attention_patterns(). None means no eval has run yet.
        self._last_val_input_ids = None

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Build AdamW optimizer with parameter groups."""
        cfg = self.cfg
        model = self.model

        # Group parameters by type
        prior_mu_params = []
        prior_sigma_params = []
        prior_phi_params = []
        e_step_params = []
        other_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if 'base_mu' in name:
                prior_mu_params.append(param)
            elif 'base_log_sigma' in name or 'decode_log_scale' in name:
                prior_sigma_params.append(param)
            elif 'phi_embed' in name or 'pos_phi' in name:
                prior_phi_params.append(param)
            elif 'e_step' in name or '_phi_preconditioner' in name:
                e_step_params.append(param)
            else:
                other_params.append(param)

        param_groups = [
            {'params': prior_mu_params, 'lr': cfg.learning_rate * 3.0, 'name': 'prior_mu'},
            {'params': prior_sigma_params, 'lr': cfg.learning_rate * 0.15, 'name': 'prior_sigma'},
            {'params': prior_phi_params, 'lr': cfg.learning_rate * 0.3, 'name': 'prior_phi'},
        ]
        if e_step_params:
            param_groups.append(
                {'params': e_step_params, 'lr': cfg.learning_rate * 0.03, 'name': 'e_step'}
            )
        if other_params:
            param_groups.append(
                {'params': other_params, 'lr': cfg.learning_rate, 'name': 'other'}
            )

        # Filter empty groups
        param_groups = [g for g in param_groups if g['params']]

        return torch.optim.AdamW(
            param_groups,
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            betas=(0.9, 0.999),
        )

    def _build_scheduler(self) -> torch.optim.lr_scheduler.LambdaLR:
        """Build cosine schedule with linear warmup."""
        cfg = self.cfg

        def lr_lambda(step: int) -> float:
            if step < cfg.warmup_steps:
                return step / max(cfg.warmup_steps, 1)
            progress = (step - cfg.warmup_steps) / max(cfg.max_steps - cfg.warmup_steps, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _compute_gradient_norms(self) -> Dict[str, float]:
        """Compute gradient norms per parameter group (after backward, before step)."""
        norms = {'total': 0.0, 'mu': 0.0, 'sigma': 0.0, 'phi': 0.0, 'ffn': 0.0, 'other': 0.0}
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            g = param.grad.data.norm().item()
            norms['total'] += g ** 2
            if 'base_mu' in name:
                norms['mu'] += g ** 2
            elif 'base_log_sigma' in name or 'decode_log_scale' in name:
                norms['sigma'] += g ** 2
            elif 'phi_embed' in name or 'pos_phi' in name:
                norms['phi'] += g ** 2
            elif 'e_step' in name:
                norms['ffn'] += g ** 2
            else:
                norms['other'] += g ** 2
        return {k: math.sqrt(v) for k, v in norms.items()}

    def _collect_e_step_diagnostics(self) -> Dict[str, float]:
        """Collect E-step diagnostics aggregated across all layers.

        Numeric scalar metrics are averaged across layers; the final layer's
        values are also exposed under the suffix ``_final`` for comparison.
        """
        blocks = list(self.model.stack.blocks)
        per_layer = [
            getattr(b.e_step, '_last_diagnostics', {}) or {} for b in blocks
        ]
        diag: Dict[str, float] = {}
        if not per_layer:
            return diag
        # Mean over layers for every numeric key present in any layer.
        all_keys = set().union(*per_layer)
        for k in all_keys:
            vals = [
                float(ld[k]) for ld in per_layer
                if k in ld and isinstance(ld[k], (int, float))
            ]
            if vals:
                diag[k] = sum(vals) / len(vals)
        # Also surface the final layer's values under `_final` suffix.
        for k, v in per_layer[-1].items():
            if isinstance(v, (int, float)):
                diag[f'{k}_final'] = float(v)
        return diag

    def _get_learning_rates(self) -> Dict[str, float]:
        """Get current learning rates per parameter group.

        LambdaLR modifies group['lr'] in place, so it already includes
        both the per-group scale and the schedule factor.
        """
        lrs = {}
        for group in self.optimizer.param_groups:
            name = group.get('name', 'default')
            lrs[name] = group['lr']
        return lrs

    def _collect_bayesian_alpha_stats(self) -> Dict[str, float]:
        """Collect Bayesian alpha diagnostics if E_learnable_alpha is enabled."""
        if not self.cfg.E_learnable_alpha:
            return {}
        stats = {}
        with torch.no_grad():
            for block in self.model.stack.blocks:
                es = block.e_step
                c0 = F.softplus(es.raw_c0)
                b0 = F.softplus(es.raw_b0)
                alpha_at_zero = c0 / b0  # alpha when KL=0
                stats['alpha_c0'] = c0.mean().item()
                stats['alpha_b0'] = b0.mean().item()
                stats['alpha_c0_std'] = c0.std().item()
                stats['alpha_b0_std'] = b0.std().item()
                stats['alpha_mean'] = alpha_at_zero.mean().item()
                stats['alpha_std'] = alpha_at_zero.std().item()
                stats['alpha_min'] = alpha_at_zero.min().item()
                stats['alpha_max'] = alpha_at_zero.max().item()
                break  # Just report first layer
        return stats

    def _collect_kappa_stats(self) -> Dict[str, float]:
        """Collect learnable kappa diagnostics."""
        if not self.cfg.learnable_kappa:
            return {}
        kappas = []
        with torch.no_grad():
            for block in self.model.stack.blocks:
                k = block.e_step.effective_kappa
                kappas.append(k.item() if isinstance(k, torch.Tensor) else k)
        return {
            'kappa_mean': sum(kappas) / len(kappas),
            'kappa_min': min(kappas),
            'kappa_max': max(kappas),
        }

    def _decode_first_sample_tokens(self) -> Optional[List[str]]:
        r"""Decode the most recent val-batch sample 0 to per-position
        token strings, for attention-heatmap axis labels.

        Returns ``None`` if no eval has populated `self._last_val_input_ids`
        yet, or if the dataset lacks a `.decode` method (e.g. synthetic
        TensorDataset). Each label is truncated to 6 characters to fit on
        tick marks.
        """
        ids = getattr(self, '_last_val_input_ids', None)
        if ids is None:
            return None
        ds = getattr(self.val_loader, 'dataset', None) if self.val_loader else None
        decode = getattr(ds, 'decode', None)
        if decode is None:
            return None
        try:
            # TODO(future): cl100k_base byte-pair tokens that span multiple
            # chars may render replacement boxes when the [:6] slice cuts
            # mid-codepoint. Acceptable visual artifact; not a correctness
            # bug. Consider a sanitizer that truncates on grapheme clusters.
            return [str(decode([int(t)]))[:6] for t in ids.tolist()]
        except Exception:  # noqa: BLE001
            return None

    def _plot_attention_patterns(self, step: int) -> None:
        r"""Save a publication-quality β heatmap PNG into
        ``output_dir/attention/attention_step_{step:08d}.png``.

        One panel per layer, sample 0 of the most recent val batch.
        Causal upper triangle is NaN-masked and rendered light grey.
        Color scale is shared across panels. Failure to plot is logged
        and swallowed so a matplotlib glitch never aborts training.

        Reads from ``block.e_step._last_attention`` (populated in
        ``vfe/e_step.py`` at the end of the final E-step iteration).
        """
        if self.output_dir is None:
            return
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from transformer.visualization.pub_style import set_pub_style
        except ImportError as exc:
            logger.warning(f"  matplotlib unavailable, skipping attention plot: {exc}")
            return

        try:
            blocks = list(self.model.stack.blocks)
            panels = []
            for layer_idx, block in enumerate(blocks):
                beta_t = getattr(block.e_step, '_last_attention', None)
                if beta_t is None:
                    continue
                panels.append((layer_idx, beta_t[0].detach().cpu().numpy()))
            if not panels:
                return

            labels = self._decode_first_sample_tokens()

            set_pub_style()
            n = len(panels)
            fig, axes = plt.subplots(
                1, n,
                figsize=(max(5.5, 4.0 * n), 4.8),
                squeeze=False,
            )
            axes = axes[0]
            vmax = float(max(p[1].max() for p in panels)) or 1.0
            cmap = plt.get_cmap('viridis').copy()
            cmap.set_bad('#dddddd')

            im = None
            for ax, (layer_idx, b) in zip(axes, panels):
                masked = b.copy()
                iu = np.triu_indices_from(masked, k=1)
                masked[iu] = np.nan
                im = ax.imshow(masked, cmap=cmap, vmin=0.0, vmax=vmax, aspect='equal')
                ax.set_title(f'layer {layer_idx}')
                ax.set_xlabel('key pos')
                if ax is axes[0]:
                    ax.set_ylabel('query pos')
                if labels is not None:
                    N = b.shape[0]
                    stride = max(1, N // 16)
                    ticks = list(range(0, N, stride))
                    tlabels = [labels[t] for t in ticks if t < len(labels)]
                    if len(tlabels) == len(ticks):
                        ax.set_xticks(ticks)
                        ax.set_yticks(ticks)
                        ax.set_xticklabels(tlabels, rotation=45, ha='right', fontsize=7)
                        ax.set_yticklabels(tlabels, fontsize=7)
                ax.grid(False)

            cbar = fig.colorbar(im, ax=list(axes), shrink=0.85, pad=0.02)
            cbar.set_label(r'$\beta_{ij}$')
            fig.suptitle(rf'Attention $\beta$  |  step {step}', y=1.02)

            out_dir = self.output_dir / 'attention'
            out_dir.mkdir(parents=True, exist_ok=True)
            # TODO(future): for very long runs (>1M steps) consider a
            # rotating "keep last K" policy or a stride-by-K scheme.
            # At eval_interval=1000 / max_steps=30k this produces ~30
            # files / ~5MB total — no rotation needed yet.
            out_path = out_dir / f'attention_step_{step:08d}.png'
            fig.savefig(out_path)
            plt.close(fig)
            logger.info(f"  attention plot saved: {out_path}")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"  attention plot failed: {exc}")

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        r"""Single training step with full metrics collection.

        Args:
            batch: Dict with 'input_ids' and 'target_ids', each (B, N).

        Returns:
            Comprehensive metrics dict for logging and CSV output.
        """
        self.model.train()
        t0 = time.time()
        if isinstance(batch, (list, tuple)):
            input_ids = batch[0].to(self.device)
            target_ids = batch[1].to(self.device)
        else:
            input_ids = batch['input_ids'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)

        # E-step: infer q* from context only (no target in E-step)
        # M-step: CE loss from q* → target
        # ce_for_log is the unscaled, regularizer-free CE (used for PPL/BPC reporting);
        # `loss` is the optimizer's combined target (may include 1/sqrt(K) scaling
        # if normalize_ce_by_dim=True and/or 0.5·mass_phi·||phi||² regularizer).
        logits, loss, ce_for_log = self.model(input_ids, targets=target_ids)

        # Backward
        self.optimizer.zero_grad()
        loss.backward()

        # Collect gradient norms BEFORE clipping
        grad_norms = self._compute_gradient_norms()

        # Gradient clipping
        if self.cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.grad_clip
            )

        self.optimizer.step()
        self.scheduler.step()
        self.global_step += 1

        step_time = time.time() - t0
        loss_val = loss.item()        # combined optimizer loss (CE_scaled + mass_phi reg)
        ce_val = ce_for_log.item()    # unscaled, regularizer-free CE
        ppl = math.exp(min(ce_val, 20.0))
        bpc = ce_val / math.log(2)
        lr = self.scheduler.get_last_lr()[0]

        # Collect E-step diagnostics from model
        e_diag = self._collect_e_step_diagnostics()

        # Build comprehensive metrics dict
        B, N = input_ids.shape
        metrics = {
            # Core loss
            'loss': loss_val,    # combined optimizer loss
            'ce': ce_val,        # unscaled cross-entropy (PPL/BPC source of truth)
            'ppl': ppl,
            'bpc': bpc,
            'lr': lr,
            # Timing
            'step_time': step_time,
            'tokens_per_sec': (B * N) / step_time if step_time > 0 else 0,
            # Gradient norms (M-step)
            'grad_norm_total': grad_norms['total'],
            'grad_norm_mu': grad_norms['mu'],
            'grad_norm_sigma': grad_norms['sigma'],
            'grad_norm_phi': grad_norms['phi'],
            'grad_norm_ffn': grad_norms['ffn'],
            'grad_norm_other': grad_norms['other'],
        }

        # E-step diagnostics
        metrics.update({f'{k}': v for k, v in e_diag.items()})

        # Bayesian alpha stats
        metrics.update(self._collect_bayesian_alpha_stats())

        # Learnable kappa stats
        metrics.update(self._collect_kappa_stats())

        return metrics

    @torch.no_grad()
    def evaluate(self, loader: Optional[DataLoader] = None, max_samples: int = 12800) -> Dict[str, float]:
        """Evaluate on validation set.

        Args:
            loader: Data loader to evaluate on. Defaults to self.val_loader.
            max_samples: Maximum samples to evaluate (prevents hang on large datasets).

        Returns:
            Dict with 'val_loss', 'val_ppl', 'val_bpc'.
        """
        loader = loader or self.val_loader
        if loader is None:
            return {}

        self.model.eval()
        total_ce = 0.0   # accumulate unscaled CE for correct PPL/BPC at eval
        total_tokens = 0
        total_samples = 0

        first_batch_seen = False
        for batch in loader:
            if total_samples >= max_samples:
                break

            if isinstance(batch, (list, tuple)):
                input_ids = batch[0].to(self.device)
                target_ids = batch[1].to(self.device)
            else:
                input_ids = batch['input_ids'].to(self.device)
                target_ids = batch['target_ids'].to(self.device)

            # Capture sample 0 of the first val batch for attention-heatmap
            # axis labels. Refreshed every evaluate() call so the heatmap
            # always reflects what the most recent val batch saw.
            # TODO(future): expose a `fixed_reference_batch` config so the
            # plotted batch is held constant across eval intervals — that
            # makes attention-pattern temporal comparison strictly visual
            # (same input, evolving β).
            if not first_batch_seen:
                self._last_val_input_ids = input_ids[0].detach().cpu()
                first_batch_seen = True

            _, _, ce_for_log = self.model(input_ids, targets=target_ids)
            n_tokens = (target_ids != -100).sum().item()
            total_ce += ce_for_log.item() * n_tokens
            total_tokens += n_tokens
            total_samples += input_ids.shape[0]

        avg_ce = total_ce / max(total_tokens, 1)
        # `val_loss` semantically means avg unscaled CE (matches val_ppl/val_bpc).
        # Default configs (mass_phi=0, normalize_ce_by_dim=False) are unaffected.
        return {
            'val_loss': avg_ce,
            'val_ppl': math.exp(min(avg_ce, 20.0)),
            'val_bpc': avg_ce / math.log(2),
        }

    def _init_metrics(self) -> None:
        """Initialize metrics infrastructure (CSV + publication tracker)."""
        if self.output_dir is None:
            return
        try:
            from transformer.training.metrics_tracking import PublicationMetricsTracker
            from transformer.training.bpc import tokens_per_char_from_dataset
            csv_path = self.output_dir / 'metrics.csv'
            self._metrics_tracker = PublicationMetricsTracker(
                csv_path,
                tokens_per_char=tokens_per_char_from_dataset(
                    getattr(self, 'train_loader', None),
                ),
            )
            logger.info(f"Metrics CSV: {csv_path}")
        except ImportError:
            logger.warning("PublicationMetricsTracker not available — CSV logging disabled")

        try:
            from transformer.analysis.publication_metrics import TrainingTracker
            from transformer.training.bpc import tokens_per_char_from_dataset
            self._pub_tracker = TrainingTracker(
                tokens_per_char=tokens_per_char_from_dataset(
                    getattr(self, 'train_loader', None),
                ),
            )
            logger.info("Publication metrics tracker initialized")
        except ImportError:
            logger.warning("TrainingTracker not available — publication metrics disabled")

    def _log_to_csv(self, step: int, metrics: Dict[str, float], batch_size: int, seq_len: int) -> None:
        """Log metrics to CSV via PublicationMetricsTracker."""
        if self._metrics_tracker is None:
            return

        # VFE decomposed loss components (belief_align, self_consistency,
        # model_coupling) are not recovered in this training
        # path — the VFE objective is minimized implicitly via the E-step
        # inner loop, so those columns are omitted rather than logged as
        # misleading zeros.
        csv_metrics = {
            'train_loss_total': metrics['loss'],
            'train_loss_ce': metrics['loss'],
            'train_loss_ce_raw': metrics['loss'],
            'train_ppl': metrics['ppl'],
            'train_bpc': metrics['bpc'],
            'beta_mean': metrics.get('beta_mean', 0),
            'beta_std': metrics.get('beta_std', 0),
            'kl_mean': metrics.get('kl_mean', 0),
            'attention_entropy': metrics.get('attention_entropy', 0),
            'attention_concentration': metrics.get('attention_concentration', 0),
        }

        grad_norms = {
            'total': metrics.get('grad_norm_total', 0),
            'mu': metrics.get('grad_norm_mu', 0),
            'sigma': metrics.get('grad_norm_sigma', 0),
            'phi': metrics.get('grad_norm_phi', 0),
            'ffn': metrics.get('grad_norm_ffn', 0),
        }

        lrs = {}
        for group in self.optimizer.param_groups:
            name = group.get('name', 'default')
            # Map to legacy LR names
            lr_map = {
                'prior_mu': 'mu_embed',
                'prior_sigma': 'sigma_embed',
                'prior_phi': 'phi_embed',
                'e_step': 'ffn',
            }
            lrs[lr_map.get(name, name)] = group['lr']

        self._metrics_tracker.log_step(
            step, csv_metrics, lrs, grad_norms,
            metrics.get('step_time', 0), batch_size, seq_len,
        )

    def _log_to_pub_tracker(self, step: int, metrics: Dict[str, float]) -> None:
        """Log metrics to publication tracker for figure generation."""
        if self._pub_tracker is None:
            return

        train_metrics = {
            'loss': metrics['loss'],
            'ce_loss': metrics['loss'],
            'ce_loss_raw': metrics['loss'],
            'attention_entropy': metrics.get('attention_entropy', 0),
            'attention_concentration': metrics.get('attention_concentration', 0),
        }

        grad_norms = {
            'total': metrics.get('grad_norm_total', 0),
            'mu': metrics.get('grad_norm_mu', 0),
            'sigma': metrics.get('grad_norm_sigma', 0),
            'phi': metrics.get('grad_norm_phi', 0),
            'ffn': metrics.get('grad_norm_ffn', 0),
            'other': metrics.get('grad_norm_other', 0),
        }

        e_step_norms = {
            'nat_grad_mu': metrics.get('nat_grad_mu_norm', 0),
            'nat_grad_sigma': metrics.get('nat_grad_sigma_norm', 0),
        }

        B = self.cfg.batch_size
        N = self.cfg.max_seq_len
        self._pub_tracker.record(
            step=step,
            epoch=step / max(len(self.train_loader), 1),
            train_metrics=train_metrics,
            grad_norms=grad_norms,
            lr=metrics.get('lr', 0),
            step_time=metrics.get('step_time', 0),
            batch_size=B,
            seq_len=N,
            e_step_norms=e_step_norms,
        )

    def _save_checkpoint(self, step: int) -> None:
        """Save model checkpoint."""
        if self.output_dir is None:
            return
        ckpt_dir = self.output_dir / 'checkpoints'
        ckpt_dir.mkdir(exist_ok=True)
        path = ckpt_dir / f'step_{step}.pt'
        torch.save({
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, path)
        logger.info(f"  Checkpoint saved: {path}")

    def _generate_figures(self) -> None:
        """Generate publication figures at end of training."""
        if self._pub_tracker is None or self.output_dir is None:
            return
        try:
            from transformer.analysis.publication_metrics import PublicationFigures
            fig_dir = self.output_dir / 'figures'
            fig_dir.mkdir(exist_ok=True)
            figures = PublicationFigures(fig_dir)
            figures.plot_training_curves(self._pub_tracker)
            figures.plot_gradient_norms_split(self._pub_tracker)
            logger.info(f"Publication figures saved to {fig_dir}")
        except Exception as e:
            logger.warning(f"Figure generation failed: {e}")

        # VFE dynamics dashboard from CSV
        if self._metrics_tracker is not None:
            try:
                from transformer.visualization.vfe_dynamics_plots import (
                    generate_all_vfe_figures,
                )
                vfe_fig_dir = self.output_dir / 'vfe_dynamics_figures'
                vfe_fig_dir.mkdir(exist_ok=True)
                csv_path = self.output_dir / 'metrics.csv'
                generate_all_vfe_figures(csv_path, vfe_fig_dir)
                logger.info(f"VFE dynamics figures saved to {vfe_fig_dir}")
            except Exception as e:
                logger.warning(f"VFE dynamics figure generation failed: {e}")

    def _log_vfe_dynamics_to_csv(self, step: int, metrics: Dict[str, float]) -> None:
        """Append VFE-specific columns to the metrics CSV."""
        if self._metrics_tracker is None:
            return

        # Build VFE dynamics row (columns expected by vfe_dynamics_plots.py)
        vfe_row = {
            # Covariance health
            'sigma_q_mean': metrics.get('sigma_q_mean', ''),
            'sigma_q_min': metrics.get('sigma_q_min', ''),
            'sigma_q_max': metrics.get('sigma_q_max', ''),
            'sigma_q_std': metrics.get('sigma_q_std', ''),
            'sigma_p_mean': metrics.get('sigma_p_mean', ''),
            # Prior-belief divergence
            'prior_belief_kl_mean': metrics.get('prior_belief_kl_mean', ''),
            'prior_belief_kl_max': metrics.get('prior_belief_kl_max', ''),
            'prior_belief_kl_std': metrics.get('prior_belief_kl_std', ''),
            # Transport geometry
            'phi_norm_mean': metrics.get('phi_norm_mean', ''),
            'phi_norm_std': metrics.get('phi_norm_std', ''),
            'phi_norm_max': metrics.get('phi_norm_max', ''),
            # E-step gradient norms
            'e_step_nat_grad_mu': metrics.get('nat_grad_mu_norm', ''),
            'e_step_nat_grad_sigma': metrics.get('nat_grad_sigma_norm', ''),
        }

        # Bayesian alpha
        for k in ('alpha_mean', 'alpha_std', 'alpha_min', 'alpha_max',
                   'alpha_c0', 'alpha_b0', 'alpha_c0_std', 'alpha_b0_std'):
            vfe_row[k] = metrics.get(k, '')

        # Learnable kappa
        for k in ('kappa_mean', 'kappa_min', 'kappa_max'):
            vfe_row[k] = metrics.get(k, '')

        # Append extra columns to CSV. Supports either a dedicated method
        # (`_append_extra`) or the standard `log_step(step, **extra)` path.
        if hasattr(self._metrics_tracker, '_append_extra'):
            self._metrics_tracker._append_extra(step, vfe_row)
        elif hasattr(self._metrics_tracker, 'log_step'):
            try:
                self._metrics_tracker.log_step(step, **vfe_row)
            except Exception as exc:
                if not getattr(self, '_vfe_csv_warned', False):
                    logger.warning(
                        "VFE dynamics columns could not be written to CSV "
                        "(log_step rejected extra kwargs: %s). Subsequent "
                        "warnings suppressed.", exc,
                    )
                    self._vfe_csv_warned = True

    def train(self, num_steps: Optional[int] = None, log_interval: Optional[int] = None) -> None:
        """Main training loop with full metrics, checkpoints, and figure generation.

        Args:
            num_steps: Number of steps to train. Defaults to cfg.max_steps.
            log_interval: Log every N steps. Defaults to cfg.log_interval.
        """
        num_steps = num_steps or self.cfg.max_steps
        log_interval = log_interval or self.cfg.log_interval
        eval_interval = self.cfg.eval_interval
        checkpoint_interval = self.cfg.checkpoint_interval
        _epoch_idx = 0
        _fn = getattr(self.train_loader.dataset, "set_epoch", None)
        if callable(_fn):
            _fn(_epoch_idx)
        data_iter = iter(self.train_loader)
        t0 = time.time()

        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Starting training: {num_steps} steps, {n_params:,} params, device={self.device}")

        # Initialize metrics infrastructure
        self._init_metrics()

        best_val_loss = float('inf')

        for step in range(num_steps):
            # Get next batch (with tuple or dict support)
            try:
                batch = next(data_iter)
            except StopIteration:
                _epoch_idx += 1
                _fn = getattr(self.train_loader.dataset, "set_epoch", None)
                if callable(_fn):
                    _fn(_epoch_idx)
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

            # Handle both tuple (input_ids, target_ids) and dict formats
            if isinstance(batch, (list, tuple)):
                batch = {'input_ids': batch[0], 'target_ids': batch[1]}

            metrics = self.train_step(batch)

            # Log to CSV and publication tracker
            _ids = batch[0] if isinstance(batch, (list, tuple)) else batch['input_ids']
            B, N = _ids.shape[0], _ids.shape[1]
            self._log_to_csv(step + 1, metrics, B, N)
            self._log_to_pub_tracker(step + 1, metrics)

            # Console logging
            if (step + 1) % log_interval == 0:
                elapsed = time.time() - t0
                steps_per_sec = (step + 1) / elapsed
                remaining_steps = max(0, num_steps - (step + 1))
                eta = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0.0
                msg = (
                    f"step {step+1}/{num_steps} | "
                    f"loss {metrics['loss']:.4f} | "
                    f"ppl {metrics['ppl']:.1f} | "
                    f"lr {metrics['lr']:.2e} | "
                    f"{steps_per_sec:.1f} steps/s | "
                    f"elap {_format_duration(elapsed)} | "
                    f"eta {_format_duration(eta)}"
                )
                if 'attention_entropy' in metrics:
                    msg += f" | H(β) {metrics['attention_entropy']:.2f}"
                if 'attention_entropy_loss' in metrics:
                    msg += f" | F_H {metrics['attention_entropy_loss']:.3f}"
                if 'sigma_q_mean' in metrics:
                    msg += f" | σ_q {metrics['sigma_q_mean']:.3f}"
                logger.info(msg)

            # Periodic evaluation
            if self.val_loader is not None and (step + 1) % eval_interval == 0:
                val_metrics = self.evaluate()
                # Visual separation: blank line before and after the val block.
                # print() bypasses the logging formatter so the blank line is
                # truly empty (no timestamp prefix).
                print()
                logger.info(
                    f"  val_loss {val_metrics['val_loss']:.4f} | "
                    f"val_ppl {val_metrics['val_ppl']:.1f} | "
                    f"val_bpc {val_metrics['val_bpc']:.3f}"
                )
                # Save publication-quality attention β heatmap. β is fresh
                # because evaluate() just ran the model forward, populating
                # block.e_step._last_attention. Plot failures are caught
                # internally; never aborts training.
                self._plot_attention_patterns(step + 1)
                print()

                # Record validation in publication tracker
                if self._pub_tracker is not None:
                    self._pub_tracker.record_validation(step + 1, {
                        'loss': val_metrics['val_loss'],
                        'ce_loss': val_metrics['val_loss'],
                        'perplexity': val_metrics['val_ppl'],
                    })

                # Log validation to CSV (map to expected keys)
                if self._metrics_tracker is not None:
                    self._metrics_tracker.log_val(step + 1, {
                        'loss': val_metrics['val_loss'],
                        'ce_loss': val_metrics['val_loss'],
                        'perplexity': val_metrics['val_ppl'],
                    })

                # Save best model
                if val_metrics['val_loss'] < best_val_loss:
                    best_val_loss = val_metrics['val_loss']
                    if self.output_dir:
                        best_path = self.output_dir / 'best_model.pt'
                        torch.save(self.model.state_dict(), best_path)

            # Periodic checkpoints
            if self.output_dir and (step + 1) % checkpoint_interval == 0:
                self._save_checkpoint(step + 1)

        total_time = time.time() - t0
        logger.info(f"Training complete: {num_steps} steps in {total_time:.1f}s")

        # Flush metrics CSV
        if self._metrics_tracker is not None:
            self._metrics_tracker.save()
            logger.info(f"Metrics CSV saved: {self.output_dir / 'metrics.csv'}")

        # Generate publication figures
        self._generate_figures()

        # Save final checkpoint
        if self.output_dir:
            self._save_checkpoint(num_steps)

        # Save publication tracker history
        if self._pub_tracker and self.output_dir:
            summary = self._pub_tracker.get_summary()
            if summary:
                logger.info(f"Final train PPL: {summary.get('final_train_ppl', 'N/A'):.1f}")
                logger.info(f"Best val PPL: {summary.get('best_val_ppl', 'N/A')}")
