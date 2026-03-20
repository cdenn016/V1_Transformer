#!/usr/bin/env python3
"""
Train Gauge Transformer with PyTorch Lightning
===============================================

Unified entry point for training all three transformer architectures:
    - vfe_dynamic: Gauge-covariant VFE transformer (GaugeTransformerLM)
    - standard: Vanilla dot-product attention baseline (StandardTransformerLM)
    - pure_fep: Pure VFE with analytic natural gradients (PureVFETransformer)

Examples:
    # VFE-dynamic on WikiText-2 (quick test)
    python scripts/train_lightning.py --training_mode vfe_dynamic --dataset wikitext-2 --max_steps 500

    # Standard baseline
    python scripts/train_lightning.py --training_mode standard --dataset wikitext-2 --max_steps 500

    # Pure FEP (single-GPU, smaller batch)
    python scripts/train_lightning.py --training_mode pure_fep --dataset wikitext-2 --max_steps 500 --max_seq_len 64 --batch_size 8

    # Multi-GPU with mixed precision (vfe_dynamic or standard only)
    python scripts/train_lightning.py --accelerator gpu --devices 2 --precision 16-mixed

    # With Weights & Biases
    python scripts/train_lightning.py --wandb --wandb_project my-project
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)

from transformer.training.lightning_data import GaugeDataModule


def parse_args():
    p = argparse.ArgumentParser(description="Train Gauge Transformer with Lightning")

    # Training mode
    p.add_argument('--training_mode', default='vfe_dynamic',
                   choices=['standard', 'vfe_dynamic', 'pure_fep'],
                   help='Model architecture to train')

    # Data
    p.add_argument('--dataset', default='wikitext-103',
                   choices=['wikitext-2', 'wikitext-103', 'openwebtext', 'wiki-ja'])
    p.add_argument('--max_seq_len', type=int, default=256)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--num_workers', type=int, default=0)

    # Model architecture (shared)
    p.add_argument('--embed_dim', type=int, default=80)
    p.add_argument('--n_layers', type=int, default=1)
    p.add_argument('--hidden_dim', type=int, default=320)
    p.add_argument('--n_heads', type=int, default=10)
    p.add_argument('--head_dim', type=int, default=1,
                   help='Head dim for VFE_dynamic irrep spec')
    p.add_argument('--kappa_beta', type=float, default=1.0)

    # Standard-specific
    p.add_argument('--dropout', type=float, default=0.1,
                   help='Dropout rate (standard mode)')
    p.add_argument('--use_rope', action='store_true',
                   help='Use RoPE positional encoding (standard mode)')
    p.add_argument('--disable_ffn', action='store_true',
                   help='Attention-only ablation (standard mode)')

    # PureFEP-specific
    p.add_argument('--belief_dim', type=int, default=32,
                   help='Full belief dimension K (pure_fep mode)')
    p.add_argument('--pure_head_dim', type=int, default=8,
                   help='Per-head dimension K_h (pure_fep mode)')
    p.add_argument('--n_esteps', type=int, default=12,
                   help='VFE descent iterations (pure_fep mode)')
    p.add_argument('--eta_E', type=float, default=0.1,
                   help='E-step learning rate (pure_fep mode)')
    p.add_argument('--eta_M', type=float, default=0.001,
                   help='M-step learning rate (pure_fep mode)')

    # Training
    p.add_argument('--max_steps', type=int, default=15000)
    p.add_argument('--warmup_steps', type=int, default=1000)
    p.add_argument('--learning_rate', type=float, default=3e-4,
                   help='Learning rate (standard mode)')
    p.add_argument('--lr_decay', default='cosine', choices=['cosine', 'linear', 'constant'])

    # Natural gradient learning rates (vfe_dynamic mode)
    p.add_argument('--mu_lr', type=float, default=0.1)
    p.add_argument('--sigma_lr', type=float, default=0.005)
    p.add_argument('--phi_lr', type=float, default=0.01)
    p.add_argument('--attention_lr', type=float, default=0.01)
    p.add_argument('--ffn_lr', type=float, default=0.001)
    p.add_argument('--output_lr', type=float, default=0.001)

    # VFE loss weights (vfe_dynamic mode)
    p.add_argument('--alpha', type=float, default=0.1)
    p.add_argument('--lambda_beta', type=float, default=1.0)

    # Lightning trainer
    p.add_argument('--accelerator', default='auto')
    p.add_argument('--devices', default='auto')
    p.add_argument('--precision', default='32-true')
    p.add_argument('--gradient_clip_val', type=float, default=1.0)
    p.add_argument('--accumulate_grad_batches', type=int, default=1)
    p.add_argument('--val_check_interval', type=int, default=100)
    p.add_argument('--log_every_n_steps', type=int, default=10)

    # Checkpointing
    p.add_argument('--checkpoint_dir', default='checkpoints')
    p.add_argument('--patience', type=int, default=0, help='Early stopping patience (0=disabled)')

    # W&B
    p.add_argument('--wandb', action='store_true')
    p.add_argument('--wandb_project', default='gauge-transformer')
    p.add_argument('--wandb_run_name', default=None)

    # Misc
    p.add_argument('--seed', type=int, default=42)

    return p.parse_args()


def _create_vfe_dynamic_model(args, vocab_size):
    """Create GaugeTransformerLM + GaugeTransformerLitModule."""
    from transformer.core.model import GaugeTransformerLM
    from transformer.training.config import TrainingConfig
    from transformer.training.lightning_module import GaugeTransformerLitModule

    irrep_spec = [('\u21130', args.n_heads, args.head_dim)]
    model_config = {
        'vocab_size': vocab_size,
        'embed_dim': args.embed_dim,
        'n_layers': args.n_layers,
        'irrep_spec': irrep_spec,
        'hidden_dim': args.hidden_dim,
        'max_seq_len': args.max_seq_len,
        'kappa_beta': args.kappa_beta,
    }
    model = GaugeTransformerLM(model_config)

    training_config = TrainingConfig(
        training_mode='vfe_dynamic',
        use_param_groups=True,
        mu_lr=args.mu_lr,
        sigma_lr=args.sigma_lr,
        phi_lr=args.phi_lr,
        attention_lr=args.attention_lr,
        ffn_lr=args.ffn_lr,
        output_lr=args.output_lr,
        alpha=args.alpha,
        lambda_beta=args.lambda_beta,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        lr_decay=args.lr_decay,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
    )

    return GaugeTransformerLitModule(model, training_config)


def _create_standard_model(args, vocab_size):
    """Create StandardTransformerLM + GaugeTransformerLitModule."""
    from transformer.baselines.standard_transformer import StandardTransformerLM
    from transformer.training.config import get_standard_config
    from transformer.training.lightning_module import GaugeTransformerLitModule

    model_config = {
        'vocab_size': vocab_size,
        'embed_dim': args.embed_dim,
        'n_layers': args.n_layers,
        'n_heads': args.n_heads,
        'hidden_dim': args.hidden_dim,
        'max_seq_len': args.max_seq_len,
        'dropout': args.dropout,
        'use_rope': args.use_rope,
        'disable_ffn': args.disable_ffn,
    }
    model = StandardTransformerLM(model_config)

    training_config = get_standard_config(
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        lr_decay=args.lr_decay,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
    )

    return GaugeTransformerLitModule(model, training_config)


def _create_pure_fep_model(args, vocab_size):
    """Create PureVFETransformer + PureVFELitModule."""
    from transformer.pure_vfe.config import PureVFEConfig
    from transformer.training.lightning_pure_vfe import PureVFELitModule

    # For pure_fep, n_heads is derived from belief_dim / pure_head_dim
    n_heads_pure = args.belief_dim // args.pure_head_dim

    pure_config = PureVFEConfig(
        vocab_size=vocab_size,
        belief_dim=args.belief_dim,
        n_heads=n_heads_pure,
        head_dim=args.pure_head_dim,
        n_esteps=args.n_esteps,
        eta_E=args.eta_E,
        eta_M=args.eta_M,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
    )

    return PureVFELitModule(pure_config)


def main():
    args = parse_args()
    pl.seed_everything(args.seed)

    # -----------------------------------------------------------
    # Data
    # -----------------------------------------------------------
    dm = GaugeDataModule(
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        dataset=args.dataset,
    )
    dm.setup()

    # -----------------------------------------------------------
    # Model (dispatch by training_mode)
    # -----------------------------------------------------------
    if args.training_mode == 'vfe_dynamic':
        lit_model = _create_vfe_dynamic_model(args, dm.vocab_size)
    elif args.training_mode == 'standard':
        lit_model = _create_standard_model(args, dm.vocab_size)
    elif args.training_mode == 'pure_fep':
        lit_model = _create_pure_fep_model(args, dm.vocab_size)
    else:
        raise ValueError(f"Unknown training_mode: {args.training_mode}")

    print(f"Training mode: {args.training_mode}")

    # -----------------------------------------------------------
    # Callbacks
    # -----------------------------------------------------------
    callbacks = [
        ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            filename=f'{args.training_mode}' + '-{step}-{val/ce_loss:.4f}',
            monitor='val/ce_loss',
            mode='min',
            save_top_k=3,
            every_n_train_steps=args.val_check_interval,
        ),
    ]

    # LR monitor only for modes with an optimizer
    if args.training_mode != 'pure_fep':
        callbacks.append(LearningRateMonitor(logging_interval='step'))

    if args.patience > 0:
        callbacks.append(
            EarlyStopping(
                monitor='val/ce_loss',
                patience=args.patience,
                mode='min',
            )
        )

    # -----------------------------------------------------------
    # Logger
    # -----------------------------------------------------------
    if args.wandb:
        try:
            from pytorch_lightning.loggers import WandbLogger
            logger = WandbLogger(
                project=args.wandb_project,
                name=args.wandb_run_name or f'{args.training_mode}',
            )
        except ImportError:
            print("wandb not installed, falling back to CSV logger")
            logger = pl.loggers.CSVLogger(args.checkpoint_dir)
    else:
        logger = pl.loggers.CSVLogger(args.checkpoint_dir)

    # -----------------------------------------------------------
    # Trainer
    # -----------------------------------------------------------
    # PureFEP: single-GPU only, no gradient clipping (no autograd)
    if args.training_mode == 'pure_fep':
        devices = 1
        gradient_clip_val = None
    else:
        devices = args.devices
        gradient_clip_val = args.gradient_clip_val

    trainer = pl.Trainer(
        max_steps=args.max_steps,
        accelerator=args.accelerator,
        devices=devices,
        precision=args.precision,
        gradient_clip_val=gradient_clip_val,
        accumulate_grad_batches=args.accumulate_grad_batches,
        val_check_interval=args.val_check_interval,
        log_every_n_steps=args.log_every_n_steps,
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=True,
    )

    # -----------------------------------------------------------
    # Train
    # -----------------------------------------------------------
    trainer.fit(lit_model, datamodule=dm)


if __name__ == '__main__':
    main()
