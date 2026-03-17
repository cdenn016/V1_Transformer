#!/usr/bin/env python3
"""
Train Gauge Transformer with PyTorch Lightning
===============================================

Entry point for training with pl.Trainer. Supports all features of
the existing FastTrainer (natural gradient LRs, VFE loss, etc.) plus
Lightning benefits: multi-GPU, mixed precision, checkpointing, logging.

Examples:
    # Quick test on WikiText-2
    python scripts/train_lightning.py --dataset wikitext-2 --max_steps 500

    # Full training on WikiText-103
    python scripts/train_lightning.py --dataset wikitext-103 --max_steps 50000

    # Multi-GPU with mixed precision
    python scripts/train_lightning.py --accelerator gpu --devices 2 --precision 16-mixed

    # With Weights & Biases
    python scripts/train_lightning.py --wandb --wandb_project my-project
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)

from transformer.core.model import GaugeTransformerLM
from transformer.training.config import TrainingConfig
from transformer.training.lightning_module import GaugeTransformerLitModule
from transformer.training.lightning_data import GaugeDataModule


def parse_args():
    p = argparse.ArgumentParser(description="Train Gauge Transformer with Lightning")

    # Data
    p.add_argument('--dataset', default='wikitext-103',
                   choices=['wikitext-2', 'wikitext-103', 'openwebtext', 'wiki-ja'])
    p.add_argument('--max_seq_len', type=int, default=256)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--num_workers', type=int, default=0)

    # Model architecture
    p.add_argument('--embed_dim', type=int, default=80)
    p.add_argument('--n_layers', type=int, default=1)
    p.add_argument('--hidden_dim', type=int, default=320)
    p.add_argument('--n_heads', type=int, default=10)
    p.add_argument('--head_dim', type=int, default=1)
    p.add_argument('--kappa_beta', type=float, default=1.0)

    # Training
    p.add_argument('--max_steps', type=int, default=15000)
    p.add_argument('--warmup_steps', type=int, default=1000)
    p.add_argument('--lr_decay', default='cosine', choices=['cosine', 'linear', 'constant'])

    # Natural gradient learning rates
    p.add_argument('--mu_lr', type=float, default=0.1)
    p.add_argument('--sigma_lr', type=float, default=0.005)
    p.add_argument('--phi_lr', type=float, default=0.01)
    p.add_argument('--attention_lr', type=float, default=0.01)
    p.add_argument('--ffn_lr', type=float, default=0.001)
    p.add_argument('--output_lr', type=float, default=0.001)

    # VFE loss weights
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
    # Model
    # -----------------------------------------------------------
    irrep_spec = [('\u21130', args.n_heads, args.head_dim)]
    model_config = {
        'vocab_size': dm.vocab_size,
        'embed_dim': args.embed_dim,
        'n_layers': args.n_layers,
        'irrep_spec': irrep_spec,
        'hidden_dim': args.hidden_dim,
        'max_seq_len': args.max_seq_len,
        'kappa_beta': args.kappa_beta,
    }
    model = GaugeTransformerLM(model_config)

    # -----------------------------------------------------------
    # Training config
    # -----------------------------------------------------------
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

    lit_model = GaugeTransformerLitModule(model, training_config)

    # -----------------------------------------------------------
    # Callbacks
    # -----------------------------------------------------------
    callbacks = [
        ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            filename='gauge-{step}-{val/ce_loss:.4f}',
            monitor='val/ce_loss',
            mode='min',
            save_top_k=3,
            every_n_train_steps=args.val_check_interval,
        ),
        LearningRateMonitor(logging_interval='step'),
    ]

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
                name=args.wandb_run_name,
            )
        except ImportError:
            print("wandb not installed, falling back to CSV logger")
            logger = pl.loggers.CSVLogger(args.checkpoint_dir)
    else:
        logger = pl.loggers.CSVLogger(args.checkpoint_dir)

    # -----------------------------------------------------------
    # Trainer
    # -----------------------------------------------------------
    trainer = pl.Trainer(
        max_steps=args.max_steps,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        gradient_clip_val=args.gradient_clip_val,
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
