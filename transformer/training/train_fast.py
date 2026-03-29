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
from dataclasses import dataclass, field
from pathlib import Path
import time
import json

from math_utils.numerical_monitor import flush as _flush_numerical_events

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
    """Training configuration with per-parameter-type learning rates and VFE loss weights.

    This is the config for FastTrainer. For the unified config used by
    PublicationTrainer, see training.config.TrainingConfig.
    """

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
    beta2: float = 0.999  # Higher β₂ for stable second-moment estimates in FastTrainer
    eps: float = 1e-8
    # weight_decay implements the Level 3 hyper-prior N(0, 1/(2·wd)) on parameters.
    # For embedding parameters (μ_p, σ_p, φ), this is the top of the Bayesian hierarchy:
    #   x → q(E-step) → p(M-step) → N(0, 1/(2·wd))
    weight_decay: float = 0.01  # L2 for non-VFE params (attention, FFN) only
    # Embedding weight decay: 0.0 because VFE loss terms already regularize:
    #   - alpha · KL(q||p) couples μ_p, Σ_p to posterior (Bayesian self-consistency)
    #   - alpha_phi · ||φ||²/2 is literally L2 on gauge frames
    # Adding optimizer WD on top double-regularizes and conflicts with VFE gradients.
    embed_weight_decay: Optional[float] = 0.01

    # Gradient control
    grad_clip: float = 1.0
    grad_accumulation_steps: int = 1

    # Free energy coefficients
    alpha: float = 1.0            # Self-consistency KL(q||p) weight — stronger than TrainingConfig (0.1) for FastTrainer stability
    beta: float = 1.0             # Belief alignment (maps to lambda_beta in loss)

    lambda_gamma: float = 0.0     # Model alignment (disabled by default)
    kappa_gamma: float = 1.0      # Temperature for γ_ij coupling weights
    lambda_hyper: float = 0.0     # Hyper-prior: KL(s_i||h) models to centroid
    # VFE observation coupling
    use_obs_in_vfe: bool = False  # Pass targets into VFE E-step (last layer only)

    # Multi-layer depth signal
    aux_layer_loss: bool = False     # Per-layer auxiliary CE loss (M-step task signal for non-final layers)
    aux_loss_weight: float = 0.3     # Weight for auxiliary per-layer CE losses
    sigma_residual: bool = False     # Additive σ residual across layers (instead of replacement)

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

    # SIGMA DETACH: Scale CE gradient to sigma_p in PriorBank.decode
    # 0.0 = full detach (no CE gradient to sigma_p, only hyper-prior drives it)
    # 1.0 = full gradient (unscaled CE gradient, risks 1/sigma_p divergence)
    # 0.01 = default (1% of CE gradient passes through)
    sigma_ce_scale: float = 0.01

    # PHI DETACH: Detach phi from backprop in non-amortized mode
    detach_phi: bool = False             # Enables fully backprop-free with phi P-flow

    # DELTA RULE: Backprop-free learning for W_out
    use_delta_rule_w_out: bool = False  # Enable delta rule for W_out
    delta_rule_lr: float = 0.001        # Learning rate for delta rule

    # LAYER/ITERATION DIAGNOSTICS: Debug multi-layer/multi-iteration performance
    track_layer_diagnostics: bool = False      # Per-layer belief statistics
    track_iteration_diagnostics: bool = False  # Per-VFE-iteration convergence data
    diagnostics_interval: int = 50             # Collect every N steps (expensive)
    verbose_diagnostics: bool = True           # Print [M-STEP], [E-STEP], [PHI] to console

    # Resume from checkpoint
    resume_from: Optional[str] = None  # Path to checkpoint to resume from

    # GAUGE GEOMETRY: Principled phi gradient control
    # These replace ad-hoc gradient clipping with theoretically motivated approaches.
    alpha_phi: float = 0.0                     # Gauge prior: (α_φ/2)||φ||² loss term (0 = disabled)
    use_slk_projection: bool = False           # Project phi to traceless sl(K) after each step
    use_killing_form: bool = False             # Cartan decomposition preconditioning for phi grads
    killing_form_sym_dampening: float = 0.1    # Dampening for non-compact (symmetric) directions

    # M-step optimizer type
    # 'adamw': Standard AdamW (diagonal Fisher via EMA of g²)
    # 'riemannian_adam': AdamW + Killing metric on phi + Fisher on mu
    # 'natural_gradient': Per-token K×K empirical Fisher blocks
    optimizer_type: str = 'adamw'
    fisher_ema_decay: float = 0.95   # EMA decay for Fisher estimation (natural_gradient only)
    fisher_damping: float = 1e-4     # Tikhonov regularization λI (natural_gradient only)

    # Ablation toggles (for PPL regression experiments)
    use_exp_map_retraction: bool = True   # True=exp map, False=linear+Cholesky (original)
    use_full_nat_grad: bool = True        # True=Σ@∇@Σ, False=diag approx (original)
   


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
        config: FastTrainingConfig with LRs, loss weights, and schedule.
        device: Target device (defaults to CPU).
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

        # Target padding uses -100 (PyTorch cross_entropy ignore_index default).
        # Dataset.pad_token_id is for INPUT padding only — targets always use -100.
        self.pad_token_id = -100

        self.model.to(self.device)

        # Create optimizer with parameter groups
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        # Training state
        self.global_step = 0
        self.best_val_ce = float('inf')  # Track CE loss (not total loss) for best model
        self.patience_counter = 0  # Early stopping counter

        # Mixed precision (using modern AMP API for PyTorch 2.x / CUDA 12+)
        self.scaler = torch.amp.GradScaler() if config.use_amp and self.device.type == 'cuda' else None

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
        Create optimizer with per-parameter-type learning rates.

        Dispatches on config.optimizer_type:
            'adamw': Standard AdamW (default)
            'riemannian_adam': AdamW + Killing/Fisher preconditioning
            'natural_gradient': Per-token block-diagonal empirical Fisher

        For gauge models, creates up to 6 groups (mu, sigma, phi, attention,
        ffn, output). For StandardTransformerLM, splits into decay vs no-decay
        groups following GPT-2/3 convention.

        Returns:
            Configured optimizer with parameter groups.
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
            # Mean embeddings (token_embed.mu_embed AND prior_bank.prior_mu / base_prior_mu)
            if 'mu_embed' in name or 'prior_mu' in name or 'base_prior_mu' in name:
                mu_params.append(param)

            # Covariance embeddings (sigma_embed, log_sigma_diag, base_log_sigma_diag,
            # AND PriorBank's log_prior_sigma which doesn't match 'log_sigma')
            elif 'sigma_embed' in name or 'log_sigma' in name or 'log_prior_sigma' in name:
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

        optimizer_type = getattr(self.config, 'optimizer_type', 'adamw')

        if optimizer_type == 'riemannian_adam':
            from transformer.training.optimizer import RiemannianAdamW, compute_killing_metric_inv
            killing_inv = None
            generators = getattr(self.model, 'generators', None)
            if generators is not None:
                killing_inv = compute_killing_metric_inv(generators)
                print(f"  Riemannian Adam: Killing metric from {generators.shape[0]} generators")
            else:
                print("  Riemannian Adam: No generators found; phi preconditioning disabled")
            optimizer = RiemannianAdamW(
                param_groups,
                model=self.model,
                killing_inv=killing_inv,
                betas=(self.config.beta1, self.config.beta2),
                eps=self.config.eps,
            )

        elif optimizer_type == 'natural_gradient':
            from transformer.training.optimizer import NaturalGradientOptimizer
            ema_decay = getattr(self.config, 'fisher_ema_decay', 0.95)
            damping = getattr(self.config, 'fisher_damping', 1e-4)
            # Estimate memory cost
            total_fisher_params = 0
            for group in param_groups:
                for p in group['params']:
                    if p.dim() == 2 and p.shape[-1] >= 4:
                        V, K = p.shape
                        total_fisher_params += V * K * K
            mem_mb = total_fisher_params * 4 / (1024 ** 2)
            print(f"  Natural Gradient Optimizer:")
            print(f"    Fisher memory: {mem_mb:.0f} MB ({total_fisher_params:,} floats)")
            print(f"    EMA decay: {ema_decay}, damping: {damping}")
            if mem_mb > 500:
                print(f"    Warning: Fisher storage is large ({mem_mb:.0f} MB)")
            optimizer = NaturalGradientOptimizer(
                param_groups,
                lr=self.config.mu_lr,  # base LR (per-group LRs override)
                weight_decay=self.config.weight_decay,
                ema_decay=ema_decay,
                damping=damping,
            )

        else:
            # Default: standard AdamW
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
                    lambda_hyper=self.config.lambda_hyper,
                    pad_token_id=self.pad_token_id,
                    use_obs_in_vfe=self.config.use_obs_in_vfe,
                    alpha_phi=getattr(self.config, 'alpha_phi', 0.0),
                    aux_loss_weight=getattr(self.config, 'aux_loss_weight', 0.0) if getattr(self.config, 'aux_layer_loss', False) else 0.0,
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
                lambda_hyper=self.config.lambda_hyper,
                pad_token_id=self.pad_token_id,
                use_obs_in_vfe=self.config.use_obs_in_vfe,
                alpha_phi=getattr(self.config, 'alpha_phi', 0.0),
                aux_loss_weight=getattr(self.config, 'aux_loss_weight', 0.0) if getattr(self.config, 'aux_layer_loss', False) else 0.0,
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

        with torch.no_grad():
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
                    # Pure CE evaluation — disable all VFE regularization terms
                    _, metrics = compute_free_energy_loss(
                        self.model,
                        input_ids,
                        target_ids,
                        alpha=0.0,
                        lambda_beta=0.0,
                        lambda_gamma=0.0,
                        kappa_gamma=1.0,
                        lambda_hyper=0.0,
                        pad_token_id=self.pad_token_id,
                        use_obs_in_vfe=False,
                        alpha_phi=0.0,
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

                # W&B logging
                if self.config.use_wandb and WANDB_AVAILABLE:
                    wandb.log({
                        'train/loss': metrics['total_loss'],
                        'train/ce_loss': metrics['ce_loss'],
                        'train/perplexity': metrics['perplexity'],
                        'train/step_time': step_time,
                        **{f'lr/{k}': v for k, v in lrs.items()},
                    }, step=step)

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
