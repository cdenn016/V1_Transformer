"""
PyTorch Lightning Module for Gauge-Theoretic Transformer
========================================================

Wraps GaugeTransformerLM (and StandardTransformerLM baseline) as a
LightningModule with:
    - Per-parameter-type learning rates (natural gradients on belief manifold)
    - VFE loss computation via compute_free_energy_loss
    - Cosine / linear / constant LR scheduling with warmup
    - Automatic logging of CE loss, perplexity, and VFE components

Usage:
    from transformer.training.lightning_module import GaugeTransformerLitModule
    from transformer.training.config import TrainingConfig

    config = TrainingConfig(...)
    model = GaugeTransformerLM(model_config)
    lit_model = GaugeTransformerLitModule(model, config)
"""

import math
import torch
try:
    import pytorch_lightning as pl
except ImportError:
    import lightning.pytorch as pl
from typing import Any, Dict, Optional, Tuple

from transformer.train import compute_free_energy_loss
from transformer.training.config import TrainingConfig
from transformer.training.optimizer import create_param_groups, create_simple_param_groups


class GaugeTransformerLitModule(pl.LightningModule):
    """
    Lightning wrapper for GaugeTransformerLM with natural gradient parameter groups.

    Supports both GaugeTransformerLM (VFE loss with alpha, lambda_beta, etc.)
    and StandardTransformerLM (plain cross-entropy).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config: TrainingConfig,
    ):
        super().__init__()
        self.model = model
        self.config = config
        self.save_hyperparameters(ignore=["model"])

        self._is_standard = hasattr(model, 'transformer_blocks')  # crude check
        # More reliable: check class name
        self._is_standard = type(model).__name__ == 'StandardTransformerLM'

        self.pad_token_id = -100

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, token_ids: torch.Tensor, **kwargs):
        return self.model(token_ids, **kwargs)

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        input_ids, target_ids = batch

        if self._is_standard:
            output = self.model(input_ids, labels=target_ids)
            loss = output['loss']
            ce_loss = loss.item()
        else:
            loss, metrics = compute_free_energy_loss(
                self.model,
                input_ids,
                target_ids,
                alpha=self.config.alpha,
                lambda_beta=self.config.lambda_beta,
                lambda_gamma=self.config.lambda_gamma,
                kappa_gamma=self.config.kappa_gamma,
                pad_token_id=self.pad_token_id,
            )
            ce_loss = metrics['loss/ce']

        ppl = math.exp(min(ce_loss, 20.0))  # cap to avoid overflow

        self.log('train/loss', loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log('train/ce_loss', ce_loss, on_step=True, on_epoch=False)
        self.log('train/perplexity', ppl, prog_bar=True, on_step=True, on_epoch=False)

        if not self._is_standard:
            self.log('train/kl_alpha', metrics.get('loss/kl_alpha', 0.0), on_step=True, on_epoch=False)
            self.log('train/beta_align', metrics.get('loss/beta_align', 0.0), on_step=True, on_epoch=False)

        return loss

    # ------------------------------------------------------------------
    # Validation step
    # ------------------------------------------------------------------
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        input_ids, target_ids = batch

        if self._is_standard:
            output = self.model(input_ids, labels=target_ids)
            ce_loss = output['loss'].item()
        else:
            _, metrics = compute_free_energy_loss(
                self.model,
                input_ids,
                target_ids,
                alpha=0.0,
                lambda_beta=0.0,
                lambda_gamma=0.0,
                kappa_gamma=1.0,
                pad_token_id=self.pad_token_id,
            )
            ce_loss = metrics['loss/ce']

        ppl = math.exp(min(ce_loss, 20.0))

        self.log('val/ce_loss', ce_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/perplexity', ppl, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

    # ------------------------------------------------------------------
    # Optimizers & schedulers
    # ------------------------------------------------------------------
    def configure_optimizers(self) -> Dict[str, Any]:
        if self.config.use_param_groups:
            param_groups = create_param_groups(self.model, self.config, verbose=False)
        else:
            param_groups = create_simple_param_groups(self.model, self.config, verbose=False)

        optimizer = torch.optim.AdamW(
            param_groups,
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
            eps=self.config.eps,
        )

        if self.config.lr_decay == 'constant':
            return {"optimizer": optimizer}

        min_ratio = min(self.config.min_lr / max(self.config.learning_rate, 1e-12), 1.0)
        warmup_steps = self.config.warmup_steps
        max_steps = self.config.max_steps

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
            progress = min(1.0, progress)
            if self.config.lr_decay == 'cosine':
                return min_ratio + 0.5 * (1 - min_ratio) * (1 + math.cos(progress * math.pi))
            elif self.config.lr_decay == 'linear':
                return max(min_ratio, 1 - progress)
            return 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
