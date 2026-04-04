"""
Fast Training Loop with Natural Gradient Learning Rates
========================================================

Lightweight trainer with per-parameter-type learning rates that
exploit natural gradient structure on the belief manifold.

Parameter groups and default LRs:
    - mu (means):           0.1    -- location on Gaussian manifold
    - sigma (covariances):  0.005  -- curvature-sensitive, needs small LR
    - phi (gauge frames):   0.01   -- Lie algebra elements for SO(N)/GL(K)
    - attention:            0.01   -- KL-divergence attention weights
    - ffn:                  0.001  -- standard feed-forward parameters
    - output:               0.001  -- LM head projection

Loss is computed by transformer.train.compute_free_energy_loss (CE + VFE
regularizers: alpha*KL(q||p), lambda_beta*belief-alignment, etc.).

Also supports StandardTransformerLM for baseline comparisons.
"""

# Suppress noisy warnings BEFORE other imports
import warnings
warnings.filterwarnings("ignore", message="CUDA path could not be detected", module="cupy")
warnings.filterwarnings("ignore", message="Failed to find cuobjdump", module="triton")
warnings.filterwarnings("ignore", message="Failed to find nvdisasm", module="triton")

import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple, List
from pathlib import Path
import time

from transformer.training.config import TrainingConfig

from math_utils.numerical_monitor import flush as _flush_numerical_events

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# Import standard loss computation
from transformer.train import compute_free_energy_loss
from transformer.baselines.standard_transformer import StandardTransformerLM

# Backward-compatible alias -- old checkpoints and call sites may reference this name.
FastTrainingConfig = TrainingConfig



# =============================================================================
# Fast Trainer with Parameter Group Learning Rates
# =============================================================================

class FastTrainer:
    """
    Trainer with per-parameter-type learning rates for the gauge transformer.

    Exploits natural gradient structure on the statistical manifold of
    belief distributions. Each VFE E-step iteration co-evolves beliefs
    and attention; the M-step (optimizer) uses group-specific LRs:

    Parameter Groups:
        1. mu_embed:  Mean embeddings (default lr=0.1)
        2. sigma_embed: Covariance embeddings (default lr=0.005)
        3. phi_embed: Gauge frame embeddings (default lr=0.01)
        4. attention: KL-divergence attention (default lr=0.01)
        5. ffn: Feed-forward networks (default lr=0.001)
        6. output: Output/LM-head projection (default lr=0.001)

    Also supports StandardTransformerLM (routes all params to ffn/no_decay groups).

    Args:
        model: GaugeTransformerLM or StandardTransformerLM instance.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        config: TrainingConfig with LRs, loss weights, and schedule.
        device: Target device (defaults to CPU).
    """

    def __init__(
        self,
        model,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: 'TrainingConfig',
        device: torch.device = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device or torch.device('cpu')

        # Target padding uses -100 (PyTorch cross_entropy ignore_index default).
        # Dataset.pad_token_id is for INPUT padding only -- targets always use -100.
        self.pad_token_id = -100

        self.model.to(self.device)

        # ── Mixed precision setup ────────────────────────────────────────
        self.use_amp = getattr(config, 'use_amp', False) and self.device.type == 'cuda'
        _amp_dtype_str = getattr(config, 'amp_dtype', 'bfloat16')
        self.amp_dtype = torch.bfloat16 if _amp_dtype_str == 'bfloat16' else torch.float16
        # GradScaler only needed for float16 (bfloat16 has float32 exponent range)
        self.scaler = torch.amp.GradScaler(
            'cuda',
            enabled=self.use_amp and self.amp_dtype == torch.float16,
        )

        # ── torch.compile ────────────────────────────────────────────────
        if getattr(config, 'use_compile', False) and self.device.type == 'cuda':
            _compile_mode = getattr(config, 'compile_mode', 'reduce-overhead')
            print(f"  Compiling model with torch.compile(mode='{_compile_mode}')...")
            self.model = torch.compile(self.model, mode=_compile_mode)

        # Create optimizer with parameter groups
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        # Training state
        self.global_step = 0
        self.best_val_ce = float('inf')  # Track CE loss (not total loss) for best model

        # Create checkpoint directory
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*70}")
        print("FAST TRAINER INITIALIZED")
        print(f"{'='*70}")
        print(f"  Device: {self.device}")
        if self.use_amp:
            print(f"  AMP: enabled (dtype={_amp_dtype_str})")
        print(f"  Max steps: {self.config.max_steps:,}")
        print(f"\n  Learning Rates (Natural Gradients!):")
        print(f"    μ (means):        {config.M_mu_p_lr}")
        print(f"    Σ (covariances):  {config.M_sigma_p_lr}")
        print(f"    φ (gauge frames): {config.M_phi_lr}")
        print(f"    Attention:        {config.M_attention_lr}")
        print(f"    FFN:              {config.M_vfe_hyperparam_lr}")
        print(f"    Output:           {config.M_output_lr}")
        print(f"{'='*70}\n")

        # Resume from checkpoint if specified
        if config.resume_from is not None:
            print(f"  Resuming from checkpoint: {config.resume_from}")
            self.load_checkpoint(config.resume_from)

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer -- delegates to optimizer.py (single source of truth)."""
        from transformer.training.optimizer import create_optimizer
        return create_optimizer(self.model, self.config)

    def _create_scheduler(self):
        """Create learning rate scheduler for all parameter groups."""
        if self.config.lr_decay == 'constant':
            return None

        min_ratio = getattr(self.config, 'min_lr_ratio', 0.1)

        def lr_lambda(step):
            # Warmup
            if step < self.config.warmup_steps:
                return step / max(1, self.config.warmup_steps)

            # Decay with min_lr floor
            progress = (step - self.config.warmup_steps) / max(1, self.config.max_steps - self.config.warmup_steps)
            progress = min(progress, 1.0)  # Clamp for steps beyond max_steps

            if self.config.lr_decay == 'cosine':
                decay = 0.5 * (1.0 + math.cos(progress * math.pi))
                return max(min_ratio, decay)
            elif self.config.lr_decay == 'linear':
                return max(min_ratio, 1.0 - progress)
            else:
                return 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=[lr_lambda] * len(self.optimizer.param_groups),
        )

        return scheduler

    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """Single training step: forward, VFE loss, backward, optimizer update.

        Args:
            batch: (input_ids, target_ids) tensors.

        Returns:
            Dict with 'total_loss', 'ce_loss', and 'perplexity'.
        """
        self.model.train()

        input_ids, target_ids = batch
        input_ids = input_ids.to(self.device)
        target_ids = target_ids.to(self.device)

        # Forward pass (AMP autocast wraps forward + loss; sigma ops self-guard)
        with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=self.amp_dtype):
            loss, metrics = compute_free_energy_loss(
                self.model,
                input_ids,
                target_ids,
                M_alpha=self.config.M_alpha,
                M_beta=self.config.M_beta,
                lambda_gamma=self.config.lambda_gamma,
                kappa_gamma=self.config.kappa_gamma,
                lambda_hyper=self.config.lambda_hyper,
                pad_token_id=self.pad_token_id,
                use_obs_in_vfe=self.config.use_obs_in_vfe,
                mass_phi=getattr(self.config, 'mass_phi', 0.05),
                aux_loss_weight=getattr(self.config, 'aux_loss_weight', 0.0) if getattr(self.config, 'aux_layer_loss', False) else 0.0,
                detach_beta_m_step=getattr(self.config, 'detach_beta_m_step', True),
            )

        # Backward pass (scaler handles float16; no-op for bfloat16)
        loss = loss / self.config.grad_accumulation_steps
        self.scaler.scale(loss).backward()

        # Gradient accumulation
        if (self.global_step + 1) % self.config.grad_accumulation_steps == 0:
            # Unscale before clipping (required for correct grad norms)
            self.scaler.unscale_(self.optimizer)

            # Gradient clipping
            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip,
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Scheduler step
            if self.scheduler is not None:
                self.scheduler.step()

            self.optimizer.zero_grad(set_to_none=True)

        # Reformat metrics for logging.
        # Note: with grad_accumulation_steps > 1, this logs each micro-batch
        # independently. Losses are per-sample (mean-reduced) so values are
        # correct, but logged at every micro-step rather than only on updates.
        formatted_metrics = {
            'total_loss': metrics['loss/total'],
            'ce_loss': metrics['loss/ce'],
            'perplexity': torch.exp(torch.tensor(metrics['loss/ce'])).item(),
        }

        return formatted_metrics

    def validate(self, max_samples: int = 12800) -> Dict[str, float]:
        """Validation loop with token-weighted CE averaging (no VFE regularizers).

        Args:
            max_samples: Maximum number of samples to evaluate (default: 12800).
                Ensures consistent evaluation across configs with different batch sizes.

        Returns:
            Dict with 'loss' (pure CE), 'ce_loss', and 'perplexity'.
        """
        self.model.eval()

        total_ce_tokens = 0.0  # Sum of CE * non_pad_tokens (for token-weighted avg)
        total_tokens = 0
        num_batches = 0
        total_samples = 0

        is_standard = isinstance(self.model, StandardTransformerLM)

        with torch.no_grad(), torch.amp.autocast('cuda', enabled=self.use_amp, dtype=self.amp_dtype):
            for batch in self.val_loader:
                if total_samples >= max_samples:
                    break

                input_ids, target_ids = batch
                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)

                # Count non-padding tokens for proper weighting
                non_pad = (target_ids != self.pad_token_id).sum().item()

                if is_standard:
                    # Standard transformer: simple cross-entropy loss
                    output = self.model(input_ids, labels=target_ids)
                    ce_loss = output['loss'].item()
                else:
                    # Pure CE evaluation -- disable all VFE regularization terms
                    _, metrics = compute_free_energy_loss(
                        self.model,
                        input_ids,
                        target_ids,
                        M_alpha=0.0,
                        M_beta=0.0,
                        lambda_gamma=0.0,
                        kappa_gamma=1.0,
                        lambda_hyper=0.0,
                        pad_token_id=self.pad_token_id,
                        use_obs_in_vfe=False,
                        mass_phi=0.0,
                    )
                    ce_loss = metrics['loss/ce']

                # Token-weighted accumulation (handles variable-size last batch)
                total_ce_tokens += ce_loss * non_pad
                total_tokens += non_pad
                num_batches += 1
                total_samples += input_ids.size(0)

        avg_ce = total_ce_tokens / max(1, total_tokens)
        perplexity = torch.exp(torch.tensor(avg_ce)).item()

        return {
            'loss': avg_ce,       # Pure CE (no VFE regularization terms)
            'ce_loss': avg_ce,
            'perplexity': perplexity,
        }

    def train(self):
        """Main training loop with periodic validation, checkpointing, and early stopping."""
        print(f"{'='*70}")
        print("STARTING FAST TRAINING")
        print(f"{'='*70}\n")

        start_time = time.time()
        step_times = []

        # Training loop - start from current global_step (for resume support)
        start_step = self.global_step
        if start_step > 0:
            print(f"Resuming from step {start_step}")

        train_iterator = iter(self.train_loader)

        if TQDM_AVAILABLE:
            pbar = tqdm(
                range(start_step, self.config.max_steps),
                desc="Training",
                initial=start_step,
                total=self.config.max_steps
            )
        else:
            pbar = range(start_step, self.config.max_steps)

        for step in pbar:
            self.global_step = step
            step_start = time.time()

            # Get batch
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(self.train_loader)
                batch = next(train_iterator)

            # Train step
            metrics = self.train_step(batch)

            step_time = time.time() - step_start
            step_times.append(step_time)

            # Logging
            if (step + 1) % self.config.log_interval == 0:
                # Get current scheduled learning rates (not just base rates)
                if self.scheduler is not None:
                    scheduled_lrs = self.scheduler.get_last_lr()
                    lrs = {group['name']: slr for group, slr in zip(self.optimizer.param_groups, scheduled_lrs)}
                else:
                    lrs = {group['name']: group['lr'] for group in self.optimizer.param_groups}

                log_msg = (
                    f"Step {step+1}/{self.config.max_steps} | "
                    f"Loss: {metrics['total_loss']:.4f} | "
                    f"PPL: {metrics['perplexity']:.1f} | "
                    f"μ_lr: {lrs.get('mu_embed', 0):.2e} | "
                    f"Time: {step_time:.2f}s"
                )

                if TQDM_AVAILABLE:
                    pbar.set_description(log_msg)
                else:
                    print(log_msg)

                # Flush numerical fallback counters and report if any fired
                _num_events = _flush_numerical_events()
                if _num_events:
                    _num_msg = "  [NUM] " + " | ".join(
                        f"{k}: {v}" for k, v in sorted(_num_events.items())
                    )
                    if TQDM_AVAILABLE:
                        tqdm.write(_num_msg)
                    else:
                        print(_num_msg)

            # Validation
            if (step + 1) % self.config.eval_interval == 0:
                val_metrics = self.validate()

                print(f"\n  Validation @ step {step+1}:")
                print(f"    Loss: {val_metrics['loss']:.4f}")
                print(f"    Perplexity: {val_metrics['perplexity']:.2f}\n")

                # Save best model based on CE loss (not total loss)
                # CE loss is the proper metric since PPL = exp(CE)
                if val_metrics['ce_loss'] < self.best_val_ce:
                    self.best_val_ce = val_metrics['ce_loss']
                    self.save_checkpoint(is_best=True)

            # Checkpointing
            if (step + 1) % self.config.checkpoint_interval == 0:
                self.save_checkpoint(is_best=False)

        # Training complete
        elapsed = time.time() - start_time
        avg_step_time = sum(step_times) / len(step_times)

        print(f"\n{'='*70}")
        print("TRAINING COMPLETE!")
        print(f"{'='*70}")
        print(f"Total time: {elapsed/60:.1f} minutes ({elapsed/3600:.2f} hours)")
        print(f"Average step time: {avg_step_time:.2f} seconds")
        print(f"Steps per second: {1.0/avg_step_time:.2f}")
        print(f"Best validation CE: {self.best_val_ce:.4f} (PPL: {torch.exp(torch.tensor(self.best_val_ce)).item():.2f})")
        print(f"{'='*70}\n")

    def save_checkpoint(self, is_best: bool = False):
        """Save model, optimizer, and scheduler state to disk.

        Args:
            is_best: If True, saves as 'best_model.pt'; otherwise as
                'checkpoint_step_N.pt' with old checkpoint cleanup.

        Returns:
            Path to the saved checkpoint file.
        """
        checkpoint = {
            'step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_ce': self.best_val_ce,
            'config': vars(self.config),
        }
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        if self.scaler.is_enabled():
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        if is_best:
            path = self.config.checkpoint_dir / 'best_model.pt'
            print(f"  💾 Saving best model: {path}")
        else:
            path = self.config.checkpoint_dir / f'checkpoint_step_{self.global_step}.pt'

        torch.save(checkpoint, path, pickle_protocol=4)

        # Cleanup old checkpoints
        if not is_best:
            checkpoints = sorted(
                self.config.checkpoint_dir.glob('checkpoint_step_*.pt'),
                key=lambda p: p.stat().st_mtime,
            )
            while len(checkpoints) > self.config.save_total_limit:
                oldest = checkpoints.pop(0)
                oldest.unlink()

        return path

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load training checkpoint to resume training.

        Args:
            checkpoint_path: Path to checkpoint file (e.g., checkpoint_step_179999.pt)
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Load model state
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        elif 'model_state' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state'])

        # Load optimizer state
        if 'optimizer_state_dict' in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception as e:
                print(f"  Warning: Could not restore optimizer state: {e}")
        elif 'optimizer_state' in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            except Exception as e:
                print(f"  Warning: Could not restore optimizer state: {e}")

        # Restore training state
        self.global_step = checkpoint.get('step', checkpoint.get('global_step', 0))
        self.best_val_ce = checkpoint.get('best_val_ce', checkpoint.get('best_val_loss', float('inf')))

        # Restore scheduler state
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            try:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except Exception as e:
                print(f"  Warning: Could not restore scheduler state: {e}")

        # Restore GradScaler state
        if self.scaler.is_enabled() and 'scaler_state_dict' in checkpoint:
            try:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            except Exception as e:
                print(f"  Warning: Could not restore scaler state: {e}")

        print(f"  ✓ Loaded checkpoint from step {self.global_step}")


# =============================================================================
# Testing
# =============================================================================

if __name__ == '__main__':
    print("="*70)
    print("FAST TRAINER TEST")
    print("="*70)

    # This file just defines the trainer
    # See train_example.py for full integration

    print("\n✓ FastTrainer class defined")
    print("✓ Supports per-parameter group learning rates")
    print("\nExample usage:")
    print("""
from transformer.model import GaugeTransformerLM, VFEConfig
from transformer.data import create_dataloaders
from transformer.training.train_fast import FastTrainer
from transformer.training.config import TrainingConfig

# Create model & data
config = VFEConfig()  # Use default config
model = GaugeTransformerLM(config)
train_loader, val_loader, vocab_size = create_dataloaders(...)

# Training config
config = TrainingConfig(
    max_steps=1000,
    M_mu_p_lr=0.1,
    M_sigma_p_lr=0.005,
    M_phi_lr=0.01,
    M_vfe_hyperparam_lr=0.001,
)

# Train!
trainer = FastTrainer(model, train_loader, val_loader, config)
trainer.train()
""")

    print("="*70)
