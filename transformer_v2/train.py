# -*- coding: utf-8 -*-
"""
Gauge-Transformer v2 Training
==============================

Training loop for the refactored gauge-theoretic transformer.

Uses:
- GaugeTransformerConfig for model architecture
- TrainingConfig (below) for training hyperparameters
- compute_vfe_loss / compute_vfe_loss_from_config for the six-term VFE loss
- Existing data pipeline from transformer.data.datasets

Usage:
    from transformer_v2.train import TrainingConfig, Trainer
    from transformer_v2 import GaugeTransformerConfig, GaugeTransformerLM
    from transformer.data.datasets import create_dataloaders

    model_config = GaugeTransformerConfig(vocab_size=50257, embed_dim=64, n_layers=4)
    model = GaugeTransformerLM(model_config)

    train_loader, val_loader, vocab_size = create_dataloaders(
        max_seq_len=model_config.max_seq_len, batch_size=32,
    )

    train_config = TrainingConfig(max_steps=10000, learning_rate=3e-4)
    trainer = Trainer(model, train_loader, val_loader, train_config)
    trainer.train()
"""

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformer_v2.loss import compute_vfe_loss

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# =============================================================================
# Training Configuration
# =============================================================================

@dataclass
class TrainingConfig:
    """Training hyperparameters (separate from model architecture).

    Modes:
    - Simple (use_param_groups=False): Single learning rate for all parameters
    - Multi-group (use_param_groups=True): Separate LRs for mu, sigma, phi, attention, ffn, output
    """

    # ── Parameter grouping ─────────────────────────────────────────────
    use_param_groups: bool = False

    # ── Simple mode: single LR ─────────────────────────────────────────
    learning_rate: float = 3e-4

    # ── Multi-group mode: per-type LRs ─────────────────────────────────
    mu_lr: float = 0.1
    sigma_lr: float = 0.005
    phi_lr: float = 0.01
    attention_lr: float = 0.01
    ffn_lr: float = 0.001
    output_lr: float = 0.001

    # ── Optimizer ──────────────────────────────────────────────────────
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0

    # ── LR schedule ───────────────────────────────────────────────────
    warmup_steps: int = 1000
    max_steps: int = 50000
    lr_decay: str = 'cosine'       # 'cosine' | 'linear' | 'constant'
    min_lr: float = 3e-5

    # ── VFE loss weights (override model config defaults) ─────────────
    # If None, uses model.config.*_loss values via compute_vfe_loss_from_config.
    # Set explicitly to override.
    alpha: Optional[float] = None
    lambda_beta: Optional[float] = None
    lambda_gamma: Optional[float] = None
    kappa_gamma: Optional[float] = None
    lambda_hyper: Optional[float] = None
    alpha_phi: Optional[float] = None
    use_obs_in_vfe: bool = False

    # ── P-flow (prior evolution via EMA) ──────────────────────────────
    use_p_flow: bool = False
    p_flow_ema_decay: float = 0.99

    # ── Delta rule (backprop-free W_out update) ───────────────────────
    use_delta_rule_w_out: bool = False
    delta_rule_lr: float = 0.001

    # ── Batching ──────────────────────────────────────────────────────
    batch_size: int = 16
    accumulation_steps: int = 1

    # ── Logging ───────────────────────────────────────────────────────
    log_every: int = 100
    eval_every: int = 1000
    save_every: int = 5000

    # ── Checkpointing ─────────────────────────────────────────────────
    checkpoint_dir: Optional[str] = None
    save_optimizer: bool = True
    resume_from: Optional[str] = None

    # ── W&B ───────────────────────────────────────────────────────────
    use_wandb: bool = False
    wandb_project: str = 'gauge-transformer-v2'
    wandb_run_name: Optional[str] = None

    # ── Device ────────────────────────────────────────────────────────
    device: str = 'cpu'
    use_amp: bool = False


# =============================================================================
# Trainer
# =============================================================================

class Trainer:
    """Training orchestration for gauge transformer v2."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[TrainingConfig] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or TrainingConfig()
        self.pad_token_id = -100

        self.device = torch.device(self.config.device)
        self.model = self.model.to(self.device)

        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        if self.config.use_amp and self.config.device == 'cuda':
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None

        self.step = 0
        self.epoch = 0
        self.best_val_ce = float('inf')

        if self.config.checkpoint_dir is not None:
            self._checkpoint_dir = Path(self.config.checkpoint_dir)
            self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._checkpoint_dir = None

        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name,
                config={
                    'training': vars(self.config),
                    'model': vars(self.model.config) if hasattr(self.model, 'config') else {},
                },
            )
            wandb.watch(self.model, log='all', log_freq=1000)

        n_params = sum(p.numel() for p in model.parameters())
        print(f"Trainer initialized")
        print(f"  Device: {self.device}")
        print(f"  Parameters: {n_params:,}")
        print(f"  Max steps: {self.config.max_steps:,}")
        if self.config.use_p_flow:
            print(f"  P-flow: ON (ema_decay={self.config.p_flow_ema_decay})")
        if self.config.use_delta_rule_w_out:
            print(f"  Delta rule W_out: ON (lr={self.config.delta_rule_lr})")

        if self.config.resume_from is not None:
            print(f"  Resuming from: {self.config.resume_from}")
            self.load_checkpoint(self.config.resume_from)

    # ─────────────────────────────────────────────────────────────────
    # Optimizer
    # ─────────────────────────────────────────────────────────────────

    def _create_optimizer(self) -> torch.optim.Optimizer:
        if self.config.use_param_groups:
            return self._create_multigroup_optimizer()
        return self._create_simple_optimizer()

    def _create_simple_optimizer(self) -> torch.optim.Optimizer:
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'bias' in name or 'norm' in name or 'embed' in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
        return torch.optim.AdamW(
            [
                {'params': decay_params, 'weight_decay': self.config.weight_decay},
                {'params': no_decay_params, 'weight_decay': 0.0},
            ],
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
            eps=self.config.eps,
        )

    def _create_multigroup_optimizer(self) -> torch.optim.Optimizer:
        groups = {
            'mu_embed': ([], self.config.mu_lr, 0.0),
            'sigma_embed': ([], self.config.sigma_lr, 0.0),
            'phi_embed': ([], self.config.phi_lr, 0.0),
            'attention': ([], self.config.attention_lr, self.config.weight_decay),
            'ffn': ([], self.config.ffn_lr, self.config.weight_decay),
            'output': ([], self.config.output_lr, 0.0),
        }

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'mu_embed' in name or 'mu_prior' in name or 'prior_mu' in name:
                groups['mu_embed'][0].append(param)
            elif 'sigma_embed' in name or 'log_sigma' in name or 'sigma_prior' in name or 'prior_sigma' in name or 'log_prior' in name:
                groups['sigma_embed'][0].append(param)
            elif 'phi_embed' in name or 'phi_prior' in name:
                groups['phi_embed'][0].append(param)
            elif 'pos_encoding' in name or 'position' in name:
                groups['phi_embed'][0].append(param)
            elif 'attention' in name or 'attn' in name:
                groups['attention'][0].append(param)
            elif 'out_proj' in name or 'lm_head' in name:
                groups['output'][0].append(param)
            else:
                groups['ffn'][0].append(param)

        param_groups = []
        for gname, (params, lr, wd) in groups.items():
            if params:
                param_groups.append({
                    'params': params, 'lr': lr, 'weight_decay': wd, 'name': gname,
                })
                print(f"  Group '{gname}': {len(params)} tensors @ lr={lr}")

        return torch.optim.AdamW(
            param_groups,
            betas=(self.config.beta1, self.config.beta2),
            eps=self.config.eps,
        )

    # ─────────────────────────────────────────────────────────────────
    # Scheduler
    # ─────────────────────────────────────────────────────────────────

    def _create_scheduler(self):
        if self.config.lr_decay == 'constant':
            return None

        base_lr = self.config.learning_rate

        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return step / max(1, self.config.warmup_steps)
            if self.config.lr_decay == 'cosine':
                progress = min(1.0, (step - self.config.warmup_steps) /
                               max(1, self.config.max_steps - self.config.warmup_steps))
                min_ratio = self.config.min_lr / base_lr
                return min_ratio + 0.5 * (1 - min_ratio) * (1 + math.cos(progress * math.pi))
            elif self.config.lr_decay == 'linear':
                min_ratio = self.config.min_lr / base_lr
                return max(min_ratio,
                           (self.config.max_steps - step) /
                           (self.config.max_steps - self.config.warmup_steps))
            return 1.0

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    # ─────────────────────────────────────────────────────────────────
    # Loss computation
    # ─────────────────────────────────────────────────────────────────

    def _compute_loss(
        self, token_ids: torch.Tensor, targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute VFE loss, using explicit overrides if set, else model config defaults."""
        cfg = self.config
        # If any loss weight is explicitly set, use compute_vfe_loss with all explicit values.
        # Otherwise, fall back to compute_vfe_loss_from_config (reads model.config.*_loss).
        has_override = any(
            getattr(cfg, k) is not None
            for k in ('alpha', 'lambda_beta', 'lambda_gamma',
                      'kappa_gamma', 'lambda_hyper', 'alpha_phi')
        )

        if has_override:
            mc = self.model.config
            return compute_vfe_loss(
                model=self.model,
                token_ids=token_ids,
                targets=targets,
                alpha=cfg.alpha if cfg.alpha is not None else mc.alpha_loss,
                lambda_beta=cfg.lambda_beta if cfg.lambda_beta is not None else mc.lambda_beta_loss,
                lambda_gamma=cfg.lambda_gamma if cfg.lambda_gamma is not None else mc.lambda_gamma_loss,
                kappa_gamma=cfg.kappa_gamma if cfg.kappa_gamma is not None else mc.kappa_gamma_loss,
                lambda_hyper=cfg.lambda_hyper if cfg.lambda_hyper is not None else mc.lambda_hyper_loss,
                alpha_phi=cfg.alpha_phi if cfg.alpha_phi is not None else mc.alpha_phi_loss,
                pad_token_id=self.pad_token_id,
                use_obs_in_vfe=cfg.use_obs_in_vfe,
            )
        else:
            from transformer_v2.loss import compute_vfe_loss_from_config
            return compute_vfe_loss_from_config(
                model=self.model,
                token_ids=token_ids,
                targets=targets,
                pad_token_id=self.pad_token_id,
                use_obs_in_vfe=cfg.use_obs_in_vfe,
            )

    # ─────────────────────────────────────────────────────────────────
    # Training step
    # ─────────────────────────────────────────────────────────────────

    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """Single training step with gradient accumulation."""
        token_ids, targets = batch
        token_ids = token_ids.to(self.device)
        targets = targets.to(self.device)

        # Forward + loss (no autocast — VFE needs float32 for Cholesky/eigendecomp)
        loss, metrics = self._compute_loss(token_ids, targets)
        loss = loss / self.config.accumulation_steps

        # Backward
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Gradient monitoring (on log steps only)
        if self.step % self.config.log_every == 0:
            grad_norms = {}
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    gn = param.grad.norm().item()
                    if 'mu_embed' in name:
                        grad_norms['grad/mu_embed'] = gn
                    elif 'sigma_embed' in name or 'log_sigma' in name:
                        grad_norms['grad/sigma_embed'] = gn
                    elif 'phi_embed' in name:
                        grad_norms['grad/phi_embed'] = gn
                    elif 'out_proj' in name:
                        grad_norms['grad/out_proj'] = gn
            metrics.update(grad_norms)

        # Optimizer step (when accumulation complete)
        if (self.step + 1) % self.config.accumulation_steps == 0:
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)

            if self.config.use_param_groups:
                for group in self.optimizer.param_groups:
                    if group['params']:
                        torch.nn.utils.clip_grad_norm_(group['params'], self.config.grad_clip)
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            self.optimizer.zero_grad()

        # P-flow: update token embeddings toward successful beliefs
        if self.config.use_p_flow and (self.step + 1) % self.config.accumulation_steps == 0:
            with torch.no_grad():
                logits, attn_info = self.model.forward_with_attention(token_ids)
                mu_beliefs = attn_info['mu']
                ce_per_token = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    reduction='none',
                    ignore_index=self.pad_token_id,
                ).reshape(targets.shape)
                self.model.p_flow_update(
                    token_ids, mu_beliefs, ce_per_token,
                    ema_decay=self.config.p_flow_ema_decay,
                )

        # Delta rule: backprop-free W_out update
        if self.config.use_delta_rule_w_out and (self.step + 1) % self.config.accumulation_steps == 0:
            with torch.no_grad():
                logits, attn_info = self.model.forward_with_attention(token_ids)
                mu_beliefs = attn_info['mu']
                self.model.delta_rule_update_w_out(
                    mu_beliefs, targets, lr=self.config.delta_rule_lr,
                )

        metrics['lr'] = self.optimizer.param_groups[0]['lr']
        return metrics

    # ─────────────────────────────────────────────────────────────────
    # Validation
    # ─────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def validate(self, max_batches: int = 200) -> Dict[str, float]:
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        total_ce = 0.0
        n_batches = 0

        for batch in self.val_loader:
            token_ids, targets = batch
            token_ids = token_ids.to(self.device)
            targets = targets.to(self.device)

            loss, metrics = self._compute_loss(token_ids, targets)
            total_loss += loss.item()
            total_ce += metrics['loss/ce']
            n_batches += 1
            if n_batches >= max_batches:
                break

        avg_loss = total_loss / max(n_batches, 1)
        avg_ce = total_ce / max(n_batches, 1)
        ppl = math.exp(min(avg_ce, 20.0))

        return {
            'val/loss': avg_loss,
            'val/ce_loss': avg_ce,
            'val/perplexity': ppl,
        }

    # ─────────────────────────────────────────────────────────────────
    # Main training loop
    # ─────────────────────────────────────────────────────────────────

    def train(self):
        print("\n" + "=" * 70)
        print("TRAINING (Gauge Transformer v2)")
        print("=" * 70)

        pbar = tqdm(total=self.config.max_steps, desc="Training") if TQDM_AVAILABLE else None
        start_time = time.time()
        self.model.train()

        try:
            while self.step < self.config.max_steps:
                for batch in self.train_loader:
                    metrics = self.train_step(batch)

                    # Logging
                    if self.step % self.config.log_every == 0:
                        elapsed = time.time() - start_time
                        print(
                            f"\nStep {self.step:6d} | "
                            f"Loss: {metrics['loss/total']:.4f} | "
                            f"CE: {metrics['loss/ce']:.4f} | "
                            f"Align: {metrics['loss/belief_align']:.4f} | "
                            f"LR: {metrics['lr']:.2e}"
                        )
                        grad_mu = metrics.get('grad/mu_embed', 0.0)
                        grad_out = metrics.get('grad/out_proj', 0.0)
                        grad_phi = metrics.get('grad/phi_embed', 0.0)
                        print(
                            f"         Grads | mu: {grad_mu:.4f} | "
                            f"out: {grad_out:.4f} | phi: {grad_phi:.4f}"
                        )
                        if self.config.use_wandb and WANDB_AVAILABLE:
                            wandb.log(metrics, step=self.step)

                    # Validation
                    if self.step % self.config.eval_every == 0 and self.step > 0:
                        val_metrics = self.validate()
                        self.model.train()
                        if val_metrics:
                            print(
                                f"\nValidation | Loss: {val_metrics['val/loss']:.4f} | "
                                f"PPL: {val_metrics['val/perplexity']:.2f}"
                            )
                            if self.config.use_wandb and WANDB_AVAILABLE:
                                wandb.log(val_metrics, step=self.step)
                            if val_metrics['val/ce_loss'] < self.best_val_ce:
                                self.best_val_ce = val_metrics['val/ce_loss']
                                self.save_checkpoint('best_model.pt')

                    # Periodic checkpoint
                    if self.step % self.config.save_every == 0 and self.step > 0:
                        self.save_checkpoint(f'checkpoint_step_{self.step}.pt')

                    if pbar is not None:
                        pbar.update(1)
                        pbar.set_postfix({'loss': f"{metrics['loss/total']:.4f}"})

                    self.step += 1
                    if self.step >= self.config.max_steps:
                        break

                self.epoch += 1

        except KeyboardInterrupt:
            print("\nTraining interrupted by user")

        finally:
            if pbar is not None:
                pbar.close()

        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)

        final_metrics = self.validate()
        if final_metrics:
            print(f"Final Validation Loss: {final_metrics['val/loss']:.4f}")
            print(f"Final Perplexity: {final_metrics['val/perplexity']:.2f}")

        self.save_checkpoint('final_model.pt')
        print("=" * 70)

    # ─────────────────────────────────────────────────────────────────
    # Checkpointing
    # ─────────────────────────────────────────────────────────────────

    def save_checkpoint(self, filename: str):
        if self._checkpoint_dir is None:
            return
        path = self._checkpoint_dir / filename
        checkpoint = {
            'step': self.step,
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'model_config': vars(self.model.config) if hasattr(self.model, 'config') else {},
            'training_config': vars(self.config),
            'best_val_ce': self.best_val_ce,
        }
        if self.config.save_optimizer:
            checkpoint['optimizer_state'] = self.optimizer.state_dict()
            if self.scheduler is not None:
                checkpoint['scheduler_state'] = self.scheduler.state_dict()
        torch.save(checkpoint, path)
        print(f"  Saved checkpoint: {path.name}")

    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state_key = 'model_state_dict' if 'model_state_dict' in checkpoint else 'model_state'
        self.model.load_state_dict(checkpoint[state_key])
        if 'optimizer_state' in checkpoint and self.config.save_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        if 'scheduler_state' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        self.step = checkpoint.get('step', 0)
        self.epoch = checkpoint.get('epoch', 0)
        self.best_val_ce = checkpoint.get('best_val_ce',
                                          checkpoint.get('best_val_loss', float('inf')))
        print(f"  Loaded checkpoint from step {self.step}")
