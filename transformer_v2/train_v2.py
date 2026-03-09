# -*- coding: utf-8 -*-
"""
v2 Click-to-Run Training Script
=================================

Edit the config below and run:
    python transformer_v2/train_v2.py

Or with CLI overrides:
    python transformer_v2/train_v2.py --device cuda --max_steps 50000 --dataset wikitext-103

Author: chris and christine
"""

# =============================================================================
# PATH SETUP
# =============================================================================
import sys
import os

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# =============================================================================
# Imports
# =============================================================================
import argparse
import json
import random
import time
import math
import csv
import subprocess
import platform
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch

from transformer_v2.config import GaugeTransformerConfig
from transformer_v2.model import GaugeTransformerLM
from transformer_v2.train import TrainingConfig, Trainer
from transformer.data.datasets import create_dataloaders


# =============================================================================
# SEED
# =============================================================================
SEED = 6

# =============================================================================
# EDIT THESE DEFAULTS — just click Run!
# =============================================================================
DEFAULT_DATASET = 'wikitext-103'   # 'wikitext-2' or 'wikitext-103'

# =============================================================================
# MODEL CONFIG  (architecture)
# =============================================================================
MODEL_CONFIG = dict(
    # Architecture
    vocab_size=50257,              # Overridden by tokenizer
    embed_dim=10,
    n_layers=1,
    hidden_dim=508,                # Only used if ffn_mode='learned'
    max_seq_len=128,

    # Gauge group
    gauge_group='GLK',             # 'SO3' | 'SON' | 'GLK'
    gauge_dim=10,
    gauge_mode='learned',
    use_multi_irrep=True,
    enforce_orthogonal=False,

    # Irrep structure
    irrep_spec=[
        ('fund', 1, 10),
    ],

    # Covariance
    diagonal_covariance=True,
    evolve_sigma=True,
    evolve_phi=True,
    evolve_phi_e_step=True,

    # VFE E-step (inside FFN)
    alpha_ffn=1.0,
    kappa_ffn=1.0,
    lambda_beta_ffn=1.0,
    n_vfe_iterations=1,
    learnable_lr=True,

    # Training loss weights (M-step)
    alpha_loss=0.1,
    lambda_beta_loss=0.0,
    lambda_gamma_loss=0.0,
    lambda_hyper_loss=0.0,
    alpha_phi_loss=0.01,

    # Sigma
    sigma_softmax_coupling=True,

    # Phi evolution
    phi_natural_gradient='pullback',

    # Attention
    mask_self_attention=True,
    use_rope=True,
    attention_pattern='full',

    # Multi-head
    per_head_kappa=True,
    use_output_projection=True,
    multihead_vfe=True,

    # Embeddings
    mu_init_std=1.0,
    mu_normalize=False,
    phi_scale=1.0,
    tie_embeddings=False,
    use_positional_embedding=False,
    pos_encoding_mode='none',

    # Regularization
    dropout=0.0,
    use_layernorm=True,
    use_residual=True,
    use_dropout=False,

    # Prior bank
    use_prior_bank=True,
)

# =============================================================================
# TRAINING CONFIG  (optimizer, schedule, logging)
# =============================================================================
TRAIN_CONFIG = dict(
    # Parameter groups
    use_param_groups=True,

    # Per-group learning rates
    mu_lr=0.05,
    sigma_lr=0.005,
    phi_lr=0.005,
    attention_lr=0.005,
    ffn_lr=0.05,
    output_lr=0.05,

    # Optimizer
    weight_decay=0.01,
    grad_clip=1.0,

    # Schedule
    warmup_steps=100,
    max_steps=12500,
    lr_decay='cosine',
    min_lr=3e-5,

    # VFE loss overrides (None = use model config defaults)
    alpha=0.1,
    lambda_beta=0,
    alpha_phi=0.01,

    # P-flow & delta rule
    use_p_flow=False,
    p_flow_ema_decay=0.95,
    use_delta_rule_w_out=False,
    delta_rule_lr=0.1,

    # Batching
    batch_size=64,
    accumulation_steps=1,

    # Logging
    log_every=100,
    eval_every=1000,
    save_every=25000,

    # Checkpointing
    checkpoint_dir='checkpoints_v2',
    save_optimizer=True,

    # W&B
    use_wandb=False,
    wandb_project='gauge-transformer-v2',

    # Device
    device='cpu',
    use_amp=False,
)


# =============================================================================
# CSV Metrics Tracker
# =============================================================================

class MetricsTracker:
    """Track training metrics to CSV for analysis."""

    def __init__(self, save_path: Path):
        self.save_path = save_path
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.headers = [
            'step', 'timestamp',
            'train_loss_total', 'train_loss_ce', 'train_loss_belief_align',
            'train_loss_self_consistency',
            'val_loss', 'val_ce', 'val_ppl',
            'train_bpc',
            'beta_mean', 'kl_mean',
            'attention_entropy', 'attention_concentration',
            'mu_lr', 'sigma_lr', 'phi_lr', 'ffn_lr',
            'grad_mu', 'grad_sigma', 'grad_phi', 'grad_out',
            'step_time', 'tokens_per_sec',
        ]
        with open(self.save_path, 'w', newline='') as f:
            csv.writer(f).writerow(self.headers)

    def log(self, entry: Dict):
        row = [entry.get(h, '') for h in self.headers]
        with open(self.save_path, 'a', newline='') as f:
            csv.writer(f).writerow(row)


# =============================================================================
# Utilities
# =============================================================================

def get_git_info() -> Dict[str, str]:
    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL
        ).decode().strip()
        branch = subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'], stderr=subprocess.DEVNULL
        ).decode().strip()
        return {'commit': commit, 'branch': branch}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {'commit': 'unknown', 'branch': 'unknown'}


def get_system_info() -> Dict[str, Any]:
    info = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9
    return info


# =============================================================================
# Main training function
# =============================================================================

def run_experiment(
    model_cfg: dict,
    train_cfg: dict,
    dataset: str = 'wikitext-103',
    num_workers: int = 4,
    seed: int = SEED,
) -> Dict:
    """Run a single training experiment end-to-end."""

    # ── Seed ────────────────────────────────────────────────────────
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seed: {seed}")

    # ── Device ──────────────────────────────────────────────────────
    device_str = train_cfg.get('device', 'cpu')
    if device_str == 'auto':
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        train_cfg = {**train_cfg, 'device': device_str}
    device = torch.device(device_str)

    print("\n" + "=" * 70)
    print("GAUGE TRANSFORMER v2 TRAINING")
    print("=" * 70)
    print(f"Device: {device}")

    # ── Data ────────────────────────────────────────────────────────
    print(f"\nLoading {dataset.upper()} data...")
    max_seq_len = model_cfg.get('max_seq_len', 128)
    batch_size = train_cfg.get('batch_size', 64)

    train_loader, val_loader, actual_vocab_size = create_dataloaders(
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        vocab_size=model_cfg.get('vocab_size'),
        num_workers=num_workers,
        dataset=dataset,
    )
    model_cfg = {**model_cfg, 'vocab_size': actual_vocab_size}
    print(f"  Vocab: {actual_vocab_size} (BPE)")
    print(f"  Batch: {batch_size} x {max_seq_len} = {batch_size * max_seq_len} tokens/step")

    # ── Model ───────────────────────────────────────────────────────
    print("\nCreating model...")
    config = GaugeTransformerConfig(**model_cfg)
    model = GaugeTransformerLM(config)

    total_params = sum(p.numel() for p in model.parameters())
    non_embed = sum(
        p.numel() for n, p in model.named_parameters() if 'embed' not in n
    )
    print(f"  K (embed_dim): {config.embed_dim}")
    print(f"  Layers: {config.n_layers}")
    print(f"  Gauge: {config.gauge_group} (dim={config.gauge_dim})")
    print(f"  Parameters: {total_params:,} (non-embed: {non_embed:,})")

    # ── Checkpoint dir ──────────────────────────────────────────────
    ckpt_dir = Path(train_cfg.get('checkpoint_dir', 'checkpoints_v2'))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save experiment config
    exp_config = {
        'experiment_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'model_config': model_cfg,
        'training_config': train_cfg,
        'dataset': dataset,
        'seed': seed,
        'git': get_git_info(),
        'system': get_system_info(),
    }
    config_path = ckpt_dir / 'experiment_config.json'
    with open(config_path, 'w') as f:
        json.dump(exp_config, f, indent=2, default=str)
    print(f"\n  Saved config: {config_path}")

    # ── Metrics tracker ─────────────────────────────────────────────
    tracker = MetricsTracker(ckpt_dir / 'metrics.csv')

    # ── Trainer ─────────────────────────────────────────────────────
    training_config = TrainingConfig(**train_cfg)
    trainer = Trainer(model, train_loader, val_loader, training_config)

    # ── Custom training loop with CSV logging ───────────────────────
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    try:
        from tqdm import tqdm
        pbar = tqdm(total=training_config.max_steps, desc="Training")
    except ImportError:
        pbar = None

    start_time = time.time()
    model.train()

    try:
        while trainer.step < training_config.max_steps:
            for batch in train_loader:
                step_start = time.time()
                metrics = trainer.train_step(batch)
                step_time = time.time() - step_start

                # CSV logging
                if trainer.step % training_config.log_every == 0:
                    elapsed = time.time() - start_time
                    tokens_per_sec = (
                        trainer.step * batch_size * max_seq_len
                    ) / elapsed if elapsed > 0 else 0

                    entry = {
                        'step': trainer.step,
                        'timestamp': time.time(),
                        'train_loss_total': metrics.get('loss/total', 0),
                        'train_loss_ce': metrics.get('loss/ce', 0),
                        'train_loss_belief_align': metrics.get('loss/belief_align', 0),
                        'train_loss_self_consistency': metrics.get('loss/self_consistency', 0),
                        'train_bpc': metrics.get('loss/ce', 0) / math.log(2),
                        'beta_mean': metrics.get('attention/beta_mean', 0),
                        'kl_mean': metrics.get('attention/kl_mean', 0),
                        'attention_entropy': metrics.get('attention/entropy', 0),
                        'attention_concentration': metrics.get('attention/concentration', 0),
                        'grad_mu': metrics.get('grad/mu_embed', 0),
                        'grad_sigma': metrics.get('grad/sigma_embed', 0),
                        'grad_phi': metrics.get('grad/phi_embed', 0),
                        'grad_out': metrics.get('grad/out_proj', 0),
                        'step_time': step_time,
                        'tokens_per_sec': tokens_per_sec,
                    }

                    # Get per-group LRs
                    for group in trainer.optimizer.param_groups:
                        gname = group.get('name', '')
                        if 'mu' in gname:
                            entry['mu_lr'] = group['lr']
                        elif 'sigma' in gname:
                            entry['sigma_lr'] = group['lr']
                        elif 'phi' in gname:
                            entry['phi_lr'] = group['lr']
                        elif 'ffn' in gname:
                            entry['ffn_lr'] = group['lr']

                    tracker.log(entry)

                    # Console output
                    ce = metrics.get('loss/ce', 0)
                    bpc = ce / math.log(2)
                    ppl = math.exp(min(ce, 20.0))
                    print(
                        f"\nStep {trainer.step:6d} | "
                        f"CE: {ce:.4f} | BPC: {bpc:.3f} | PPL: {ppl:.1f} | "
                        f"LR: {metrics.get('lr', 0):.2e} | "
                        f"{tokens_per_sec:.0f} tok/s"
                    )

                # Validation
                if trainer.step % training_config.eval_every == 0 and trainer.step > 0:
                    val_metrics = trainer.validate()
                    model.train()
                    if val_metrics:
                        val_bpc = val_metrics['val/ce_loss'] / math.log(2)
                        print(
                            f"\n  Val | CE: {val_metrics['val/ce_loss']:.4f} | "
                            f"BPC: {val_bpc:.3f} | PPL: {val_metrics['val/perplexity']:.1f}"
                        )

                        # Log validation to CSV
                        entry = {
                            'step': trainer.step,
                            'timestamp': time.time(),
                            'val_loss': val_metrics['val/loss'],
                            'val_ce': val_metrics['val/ce_loss'],
                            'val_ppl': val_metrics['val/perplexity'],
                        }
                        tracker.log(entry)

                        if val_metrics['val/ce_loss'] < trainer.best_val_ce:
                            trainer.best_val_ce = val_metrics['val/ce_loss']
                            trainer.save_checkpoint('best_model.pt')

                # Periodic checkpoint
                if trainer.step % training_config.save_every == 0 and trainer.step > 0:
                    trainer.save_checkpoint(f'checkpoint_step_{trainer.step}.pt')

                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix({
                        'ce': f"{metrics.get('loss/ce', 0):.3f}",
                        'bpc': f"{metrics.get('loss/ce', 0) / math.log(2):.3f}",
                    })

                trainer.step += 1
                if trainer.step >= training_config.max_steps:
                    break

            trainer.epoch += 1

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")

    finally:
        if pbar is not None:
            pbar.close()

    # ── Final evaluation ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    elapsed = time.time() - start_time
    print(f"  Total time: {elapsed / 60:.1f} min")
    print(f"  Steps: {trainer.step:,}")

    final_val = trainer.validate()
    if final_val:
        bpc = final_val['val/ce_loss'] / math.log(2)
        print(f"  Final CE: {final_val['val/ce_loss']:.4f}")
        print(f"  Final BPC: {bpc:.3f}")
        print(f"  Final PPL: {final_val['val/perplexity']:.1f}")
        random_bpc = math.log(actual_vocab_size) / math.log(2)
        print(f"  Random BPC: {random_bpc:.3f}")
        improvement = random_bpc / bpc if bpc > 0 else 0
        print(f"  Improvement: {improvement:.1f}x over random")

    trainer.save_checkpoint('final_model.pt')

    # Save summary
    summary = {
        'experiment_id': exp_config['experiment_id'],
        'steps': trainer.step,
        'elapsed_min': elapsed / 60,
        'best_val_ce': trainer.best_val_ce,
        'final_val': final_val,
        'model_params': total_params,
    }
    summary_path = ckpt_dir / 'result_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Results: {summary_path}")
    print("=" * 70)

    return summary


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Gauge Transformer v2 Training (click-to-run)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Just run with defaults (edit MODEL_CONFIG / TRAIN_CONFIG above)
    python transformer_v2/train_v2.py

    # Override common settings via CLI
    python transformer_v2/train_v2.py --device cuda --max_steps 50000
    python transformer_v2/train_v2.py --dataset wikitext-2 --batch_size 32
    python transformer_v2/train_v2.py --embed_dim 64 --n_layers 4 --max_steps 100000
""",
    )

    # Dataset
    parser.add_argument('--dataset', type=str, default=DEFAULT_DATASET,
                        choices=['wikitext-2', 'wikitext-103'])
    parser.add_argument('--num_workers', type=int, default=4)

    # Model overrides
    parser.add_argument('--embed_dim', type=int, default=None)
    parser.add_argument('--n_layers', type=int, default=None)
    parser.add_argument('--max_seq_len', type=int, default=None)
    parser.add_argument('--gauge_group', type=str, default=None,
                        choices=['SO3', 'SON', 'GLK'])
    parser.add_argument('--gauge_dim', type=int, default=None)
    parser.add_argument('--n_vfe_iterations', type=int, default=None)

    # Training overrides
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--max_steps', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='Override all LRs (sets use_param_groups=False)')
    parser.add_argument('--checkpoint_dir', type=str, default=None)
    parser.add_argument('--use_wandb', action='store_true', default=None)
    parser.add_argument('--seed', type=int, default=SEED)

    # P-flow / delta rule
    parser.add_argument('--use_p_flow', action='store_true', default=None)
    parser.add_argument('--use_delta_rule', action='store_true', default=None)

    args = parser.parse_args()

    # Build configs with CLI overrides
    model_cfg = dict(MODEL_CONFIG)
    train_cfg = dict(TRAIN_CONFIG)

    # Apply model overrides
    for key in ('embed_dim', 'n_layers', 'max_seq_len', 'gauge_group',
                'gauge_dim', 'n_vfe_iterations'):
        val = getattr(args, key, None)
        if val is not None:
            model_cfg[key] = val

    # Apply training overrides
    if args.device is not None:
        train_cfg['device'] = args.device
    if args.max_steps is not None:
        train_cfg['max_steps'] = args.max_steps
    if args.batch_size is not None:
        train_cfg['batch_size'] = args.batch_size
    if args.checkpoint_dir is not None:
        train_cfg['checkpoint_dir'] = args.checkpoint_dir
    if args.use_wandb is not None:
        train_cfg['use_wandb'] = args.use_wandb
    if args.use_p_flow is not None:
        train_cfg['use_p_flow'] = True
    if args.use_delta_rule is not None:
        train_cfg['use_delta_rule_w_out'] = True

    # Single LR mode
    if args.learning_rate is not None:
        train_cfg['use_param_groups'] = False
        train_cfg['learning_rate'] = args.learning_rate

    run_experiment(
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        dataset=args.dataset,
        num_workers=args.num_workers,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
