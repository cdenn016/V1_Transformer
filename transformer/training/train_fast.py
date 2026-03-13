"""
Fast Training Loop with Natural Gradient Learning Rates
========================================================

Uses separate learning rates for different parameter types,
based on empirical convergence from test suite:
    - mu_q_lr = 0.1 (means)
    - Sigma_q_lr = 0.005 (covariances)
    - phi_lr = 0.01 (gauge frames)
    - ffn_lr = 0.001 (standard FFN parameters)

This exploits the natural gradient speedup on statistical manifolds.

Author: Optimized from test suite convergence
Date: November 2025
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
from dataclasses import dataclass, field
from pathlib import Path
import time
import json

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

# Import standard loss computation
from transformer.train import compute_free_energy_loss
from transformer.baselines.standard_transformer import StandardTransformerLM


# =============================================================================
# Fast Training Configuration
# =============================================================================

@dataclass
class FastTrainingConfig:
    """Training configuration with per-parameter group learning rates."""

    # Training steps (use epochs OR max_steps, epochs takes precedence)
    epochs: Optional[int] = None  # If set, overrides max_steps
    max_steps: int = 1000
    warmup_steps: int = 50

    # Per-parameter group learning rates (NATURAL GRADIENTS!)
    mu_lr: float = 0.1           # Mean embeddings (from test suite)
    sigma_lr: float = 0.005      # Covariance embeddings (from test suite)
    phi_lr: float = 0.01         # Gauge frames
    attention_lr: float = 0.01   # Attention parameters
    ffn_lr: float = 0.001        # FFN parameters (standard)
    output_lr: float = 0.001     # Output projection

    # Optimizer hyperparameters
    beta1: float = 0.9
    beta2: float = 0.95  # Match TrainingConfig (0.95 for natural gradient stability)
    eps: float = 1e-8
    # weight_decay implements the Level 3 hyper-prior N(0, 1/(2·wd)) on parameters.
    # For embedding parameters (μ_p, σ_p, φ), this is the top of the Bayesian hierarchy:
    #   x → q(E-step) → p(M-step) → N(0, 1/(2·wd))
    weight_decay: float = 0.1  # Match TrainingConfig
    # Embedding-specific weight decay (Level 3 hyper-prior on priors).
    # None = use weight_decay (same as non-embedding params).
    # 0.0 = uninformative hyper-prior (no regularization toward zero).
    embed_weight_decay: Optional[float] = None

    # Gradient control
    grad_clip: float = 1.0
    grad_accumulation_steps: int = 1

    # Free energy coefficients
    alpha: float = 0.1            # Self-consistency regularization (match TrainingConfig)
    beta: float = 1.0             # Belief alignment (maps to lambda_beta in loss)
    lambda_gamma: float = 0.0     # Model alignment (disabled by default)
    kappa_gamma: float = 1.0      # Temperature for γ_ij coupling weights
    lambda_hyper: float = 0.0     # Hyper-prior: KL(s_i||h) models to centroid

    # VFE observation coupling
    use_obs_in_vfe: bool = False  # Pass targets into VFE E-step (last layer only)

    # Learning rate schedule
    lr_decay: str = 'cosine'  # 'cosine', 'linear', 'constant'
    min_lr_ratio: float = 0.1  # Minimum LR as fraction of peak (floor for cosine decay)

    # Logging
    log_interval: int = 10
    eval_interval: int = 100
    checkpoint_interval: int = 200

    # Early stopping
    patience: int = 0  # If > 0, stop if no improvement for this many evals

    # Checkpointing
    checkpoint_dir: Path = Path('checkpoints')
    save_total_limit: int = 3

    # Weights & Biases
    use_wandb: bool = False
    wandb_project: str = 'gauge-transformer-fast'
    wandb_run_name: Optional[str] = None

    # Mixed precision
    use_amp: bool = False

    # P-FLOW: EMA update of token embeddings toward successful beliefs
    use_p_flow: bool = False          # Enable P-flow updates
    p_flow_ema_decay: float = 0.99    # EMA decay (0.99 = 1% update per step)

    # DELTA RULE: Backprop-free learning for W_out
    use_delta_rule_w_out: bool = False  # Enable delta rule for W_out
    delta_rule_lr: float = 0.001        # Learning rate for delta rule

    # RG METRICS: Track renormalization group flow
    compute_rg_metrics: bool = False     # Enable RG flow analysis
    rg_metrics_interval: int = 100       # Compute every N steps
    rg_auto_cluster: bool = True         # Auto-detect clusters
    rg_n_clusters: Optional[int] = None  # Fixed number of clusters (None = auto)
    track_dynamic_rg: bool = False       # Track beta evolution across VFE iterations

    # Resume from checkpoint
    resume_from: Optional[str] = None  # Path to checkpoint to resume from

    # GAUGE GEOMETRY: Principled phi gradient control
    # These replace ad-hoc gradient clipping with theoretically motivated approaches.
    alpha_phi: float = 0.0                     # Gauge prior: (α_φ/2)||φ||² loss term (0 = disabled)
    use_slk_projection: bool = False           # Project phi to traceless sl(K) after each step
    use_killing_form: bool = False             # Cartan decomposition preconditioning for phi grads
    killing_form_sym_dampening: float = 0.1    # Dampening for non-compact (symmetric) directions

    # Ablation toggles (for PPL regression experiments)
    use_exp_map_retraction: bool = True   # True=exp map, False=linear+Cholesky (original)
    use_full_nat_grad: bool = True        # True=Σ@∇@Σ, False=diag approx (original)
   


# =============================================================================
# Fast Trainer with Parameter Group Learning Rates
# =============================================================================

class FastTrainer:
    """
    Trainer with separate learning rates for each parameter type.

    Parameter Groups:
        1. mu_embed: Mean embeddings (lr=0.1)
        2. sigma_embed: Covariance embeddings (lr=0.005)
        3. phi_embed: Gauge frame embeddings (lr=0.01)
        4. attention: Attention mechanism (lr=0.01)
        5. ffn: Feed-forward networks (lr=0.001)
        6. output: Output projection (lr=0.001)

    This exploits natural gradient structure on statistical manifolds!
    """

    def __init__(
        self,
        model,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: FastTrainingConfig,
        device: torch.device = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device or torch.device('cpu')

        # Get pad_token_id from dataset for proper loss masking
        self.pad_token_id = getattr(train_loader.dataset, 'pad_token_id', -100)

        self.model.to(self.device)

        # Create optimizer with parameter groups
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        # Training state
        self.global_step = 0
        self.best_val_ce = float('inf')  # Track CE loss (not total loss) for best model
        self.patience_counter = 0  # Early stopping counter

        # Mixed precision (using modern AMP API for PyTorch 2.x / CUDA 12+)
        self.scaler = torch.amp.GradScaler('cuda') if config.use_amp else None

        # W&B logging
        if config.use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name,
                config=vars(config),
            )

        # Create checkpoint directory
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*70}")
        print("FAST TRAINER INITIALIZED")
        print(f"{'='*70}")
        print(f"  Device: {self.device}")
        print(f"  Max steps: {self.config.max_steps:,}")
        print(f"\n  Learning Rates (Natural Gradients!):")
        print(f"    μ (means):        {config.mu_lr}")
        print(f"    Σ (covariances):  {config.sigma_lr}")
        print(f"    φ (gauge frames): {config.phi_lr}")
        print(f"    Attention:        {config.attention_lr}")
        print(f"    FFN:              {config.ffn_lr}")
        print(f"    Output:           {config.output_lr}")
        print(f"{'='*70}\n")

        # Resume from checkpoint if specified
        if config.resume_from is not None:
            print(f"  Resuming from checkpoint: {config.resume_from}")
            self.load_checkpoint(config.resume_from)

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """
        Create optimizer with per-parameter group learning rates.

        Returns:
            AdamW optimizer with 6 parameter groups
        """
        # Collect parameters by type
        mu_params = []
        sigma_params = []
        phi_params = []
        attention_params = []
        ffn_params = []
        output_params = []
        no_decay_params = []  # Embeddings, LayerNorm, biases (no weight decay)

        is_standard = isinstance(self.model, StandardTransformerLM)

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            # Standard transformer: separate no-decay params (embeddings, LN, biases)
            # following GPT-2/GPT-3 convention
            if is_standard:
                if 'token_embed' in name or 'pos_embed' in name:
                    no_decay_params.append(param)
                elif 'ln' in name or 'norm' in name or name.endswith('.bias'):
                    no_decay_params.append(param)
                elif 'attention' in name or 'attn' in name:
                    attention_params.append(param)
                else:
                    ffn_params.append(param)
                continue

            # Gauge model parameter grouping
            # Mean embeddings
            if 'mu_embed' in name:
                mu_params.append(param)

            # Covariance embeddings (sigma_embed, log_sigma_diag, base_log_sigma_diag)
            elif 'sigma_embed' in name or 'log_sigma' in name:
                sigma_params.append(param)

            # Gauge frame embeddings
            elif 'phi_embed' in name:
                phi_params.append(param)

            # Positional encoding (treat as gauge frames)
            elif 'pos_encoding' in name:
                phi_params.append(param)

            # Attention mechanism
            elif 'attention' in name or 'attn' in name:
                attention_params.append(param)

            # Output projection
            elif 'out_proj' in name:
                output_params.append(param)

            # FFN (default for everything else)
            else:
                ffn_params.append(param)

        # Create parameter groups
        # Embedding weight decay = Level 3 hyper-prior: N(0, 1/(2·wd))
        # None → inherit from weight_decay; 0.0 → uninformative hyper-prior
        embed_wd = self.config.embed_weight_decay if self.config.embed_weight_decay is not None else self.config.weight_decay

        param_groups = []
        if mu_params:
            param_groups.append({
                'params': mu_params,
                'lr': self.config.mu_lr,
                'weight_decay': embed_wd,
                'name': 'mu_embed',
            })
            print(f"  Parameter group 'mu_embed': {len(mu_params)} tensors @ lr={self.config.mu_lr}, wd={embed_wd}")

        if sigma_params:
            param_groups.append({
                'params': sigma_params,
                'lr': self.config.sigma_lr,
                'weight_decay': embed_wd,
                'name': 'sigma_embed',
            })
            print(f"  Parameter group 'sigma_embed': {len(sigma_params)} tensors @ lr={self.config.sigma_lr}, wd={embed_wd}")

        if phi_params:
            param_groups.append({
                'params': phi_params,
                'lr': self.config.phi_lr,
                'weight_decay': embed_wd,
                'name': 'phi_embed',
            })
            print(f"  Parameter group 'phi_embed': {len(phi_params)} tensors @ lr={self.config.phi_lr}, wd={embed_wd}")

        if attention_params:
            param_groups.append({
                'params': attention_params,
                'lr': self.config.attention_lr,
                'weight_decay': self.config.weight_decay,
                'name': 'attention',
            })
            print(f"  Parameter group 'attention': {len(attention_params)} tensors @ lr={self.config.attention_lr}")

        if ffn_params:
            param_groups.append({
                'params': ffn_params,
                'lr': self.config.ffn_lr,
                'weight_decay': self.config.weight_decay,
                'name': 'ffn',
            })
            print(f"  Parameter group 'ffn': {len(ffn_params)} tensors @ lr={self.config.ffn_lr}")

        if output_params:
            param_groups.append({
                'params': output_params,
                'lr': self.config.output_lr,
                'weight_decay': 0.0,  # Often tied to embeddings
                'name': 'output',
            })
            print(f"  Parameter group 'output': {len(output_params)} tensors @ lr={self.config.output_lr}")

        if no_decay_params:
            # Embeddings, LayerNorm, biases — no weight decay (GPT-2/3 convention)
            param_groups.append({
                'params': no_decay_params,
                'lr': self.config.ffn_lr,
                'weight_decay': 0.0,
                'name': 'no_decay',
            })
            print(f"  Parameter group 'no_decay': {len(no_decay_params)} tensors @ lr={self.config.ffn_lr} (embeddings, LN, biases)")

        optimizer = torch.optim.AdamW(
            param_groups,
            betas=(self.config.beta1, self.config.beta2),
            eps=self.config.eps,
        )

        return optimizer

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
                decay = 0.5 * (1.0 + torch.cos(torch.tensor(progress * math.pi)).item())
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
        """Single training step."""
        self.model.train()

        input_ids, target_ids = batch
        input_ids = input_ids.to(self.device)
        target_ids = target_ids.to(self.device)

        # Forward pass (with optional AMP using modern API for PyTorch 2.x / CUDA 12+)
        if self.config.use_amp:
            with torch.amp.autocast('cuda'):
                loss, metrics = compute_free_energy_loss(
                    self.model,
                    input_ids,
                    target_ids,
                    alpha=self.config.alpha,
                    lambda_beta=self.config.beta,
                    lambda_gamma=self.config.lambda_gamma,
                    kappa_gamma=self.config.kappa_gamma,
                    pad_token_id=self.pad_token_id,
                    
                )
        else:
            loss, metrics = compute_free_energy_loss(
                self.model,
                input_ids,
                target_ids,
                alpha=self.config.alpha,
                lambda_beta=self.config.beta,
                lambda_gamma=self.config.lambda_gamma,
                kappa_gamma=self.config.kappa_gamma,
                pad_token_id=self.pad_token_id,
                
            )

        # Backward pass
        loss = loss / self.config.grad_accumulation_steps

        if self.config.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Gradient accumulation
        if (self.global_step + 1) % self.config.grad_accumulation_steps == 0:
            # Gradient clipping
            if self.config.grad_clip > 0:
                if self.config.use_amp:
                    self.scaler.unscale_(self.optimizer)

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip,
                )

            # Optimizer step
            if self.config.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            # Scheduler step
            if self.scheduler is not None:
                self.scheduler.step()

            self.optimizer.zero_grad()

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

    def validate(self, max_batches: int = 200) -> Dict[str, float]:
        """Validation loop.

        Args:
            max_batches: Maximum number of validation batches (default: 200).
        """
        self.model.eval()

        total_loss = 0.0
        total_ce = 0.0
        num_batches = 0

        is_standard = isinstance(self.model, StandardTransformerLM)

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids, target_ids = batch
                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)

                if is_standard:
                    # Standard transformer: simple cross-entropy loss
                    output = self.model(input_ids, labels=target_ids)
                    loss = output['loss']
                    ce_loss = loss.item()
                else:
                    loss, metrics = compute_free_energy_loss(
                        self.model,
                        input_ids,
                        target_ids,
                        alpha=self.config.alpha,
                        lambda_beta=self.config.beta,
                        lambda_gamma=self.config.lambda_gamma,
                        kappa_gamma=self.config.kappa_gamma,
                        pad_token_id=self.pad_token_id,
                       
                    )
                    ce_loss = metrics['loss/ce']

                total_loss += loss.item()
                total_ce += ce_loss
                num_batches += 1

                # Limit validation batches if specified
                if max_batches is not None and num_batches >= max_batches:
                    break

        avg_loss = total_loss / max(1, num_batches)
        avg_ce = total_ce / max(1, num_batches)
        perplexity = torch.exp(torch.tensor(avg_ce)).item()

        return {
            'loss': avg_loss,
            'ce_loss': avg_ce,
            'perplexity': perplexity,
        }

    def train(self):
        """Main training loop."""
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

                # W&B logging
                if self.config.use_wandb and WANDB_AVAILABLE:
                    wandb.log({
                        'train/loss': metrics['total_loss'],
                        'train/ce_loss': metrics['ce_loss'],
                        'train/perplexity': metrics['perplexity'],
                        'train/step_time': step_time,
                        **{f'lr/{k}': v for k, v in lrs.items()},
                    }, step=step)

            # Validation
            if (step + 1) % self.config.eval_interval == 0:
                val_metrics = self.validate()

                print(f"\n  Validation @ step {step+1}:")
                print(f"    Loss: {val_metrics['loss']:.4f}")
                print(f"    Perplexity: {val_metrics['perplexity']:.2f}\n")

                # W&B logging
                if self.config.use_wandb and WANDB_AVAILABLE:
                    wandb.log({
                        'val/loss': val_metrics['loss'],
                        'val/perplexity': val_metrics['perplexity'],
                    }, step=step)

                # Save best model based on CE loss (not total loss)
                # CE loss is the proper metric since PPL = exp(CE)
                if val_metrics['ce_loss'] < self.best_val_ce:
                    self.best_val_ce = val_metrics['ce_loss']
                    self.patience_counter = 0  # Reset patience
                    self.save_checkpoint(is_best=True)
                else:
                    self.patience_counter += 1  # Increment patience
                    if self.config.patience > 0 and self.patience_counter >= self.config.patience:
                        print(f"\n⚠ Early stopping triggered! No improvement for {self.config.patience} evaluations.")
                        print(f"  Best validation CE: {self.best_val_ce:.4f}")
                        break  # Stop training

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
        """Save checkpoint."""
        checkpoint = {
            'step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_ce': self.best_val_ce,
            'config': vars(self.config),
            'patience_counter': self.patience_counter,
        }
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        if is_best:
            path = self.config.checkpoint_dir / 'best_model.pt'
            print(f"  💾 Saving best model: {path}")
        else:
            path = self.config.checkpoint_dir / f'checkpoint_step_{self.global_step}.pt'

        # Use pickle protocol 4 to avoid the PyTorch zip serialization
        # 2GB limit on Windows (inline_container.cc position overflow).
        torch.save(checkpoint, path, pickle_protocol=4,
                   _use_new_zipfile_serialization=False)

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
        self.patience_counter = checkpoint.get('patience_counter', 0)

        # Restore scheduler state
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            try:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except Exception as e:
                print(f"  Warning: Could not restore scheduler state: {e}")

        # Restore AMP scaler state
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
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
from transformer.training.train_fast import FastTrainer, FastTrainingConfig

# Create model & data
config = VFEConfig()  # Use default config
model = GaugeTransformerLM(config)
train_loader, val_loader, vocab_size = create_dataloaders(...)

# Fast training config
config = FastTrainingConfig(
    max_steps=1000,
    mu_lr=0.1,
    sigma_lr=0.005,
    phi_lr=0.01,
    ffn_lr=0.001,
)

# Train!
trainer = FastTrainer(model, train_loader, val_loader, config)
trainer.train()
""")

    print("="*70)
