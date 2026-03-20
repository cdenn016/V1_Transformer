#!/usr/bin/env python3
"""
Train Gauge Transformer with PyTorch Lightning
===============================================

Edit the config dictionaries below and run this script directly.
Supports three model architectures:
    - VFE_dynamic: Gauge-covariant VFE transformer (GaugeTransformerLM)
    - standard: Vanilla dot-product attention baseline (StandardTransformerLM)
    - pure_fep: Pure VFE with analytic natural gradients (PureVFETransformer)

Usage:
    python scripts/train_lightning.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# SELECT MODE — change this to switch between model architectures
# ============================================================================
ACTIVE_MODE = 'VFE_dynamic'     # 'standard', 'VFE_dynamic', or 'pure_fep'
DATASET = 'wikitext-103'        # 'wikitext-2', 'wikitext-103', 'openwebtext', 'wiki-ja'
SEED = 42


# ============================================================================
# CONFIG 1: STANDARD TRANSFORMER (Baseline)
# ============================================================================
STANDARD_CONFIG = {
    # Model architecture
    'vocab_size': 50257,
    'embed_dim': 10,
    'n_layers': 1,
    'n_heads': 1,
    'hidden_dim': 24527,
    'max_seq_len': 128,
    'dropout': 0.1,
    'use_rope': False,
    'disable_ffn': False,

    # Training
    'batch_size': 64,
    'num_workers': 10,
    'max_steps': 15000,
    'warmup_steps': 100,
    'learning_rate': 3e-4,
    'lr_decay': 'cosine',
    'weight_decay': 0.01,
    'grad_clip': 1.0,

    # Logging
    'log_interval': 100,
    'eval_interval': 1000,
    'checkpoint_interval': 25000,

    # Lightning
    'accelerator': 'auto',
    'devices': 'auto',
    'precision': '32-true',
    'accumulate_grad_batches': 1,

    # W&B (set use_wandb=True to enable)
    'use_wandb': False,
    'wandb_project': 'gauge-transformer',
    'wandb_run_name': None,

    # Checkpointing
    'checkpoint_dir': 'checkpoints',
    'patience': 0,                    # Early stopping (0=disabled)
}


# ============================================================================
# CONFIG 2: VFE_DYNAMIC (Gauge-covariant VFE transformer)
# ============================================================================
VFE_DYNAMIC_CONFIG = {
    # Model architecture
    'vocab_size': 50257,
    'embed_dim': 10,
    'n_layers': 1,
    'hidden_dim': 508,
    'max_seq_len': 128,
    'kappa_beta': 1.0,

    # Irrep specification: (name, n_heads, head_dim)
    'irrep_spec': [
        ('fund', 1, 10),
    ],

    # Training
    'batch_size': 32,
    'num_workers': 10,
    'max_steps': 30000,
    'warmup_steps': 100,
    'lr_decay': 'cosine',
    'grad_clip': 1.0,

    # Natural gradient learning rates
    'mu_lr': 0.05,
    'sigma_lr': 0.005,
    'phi_lr': 0.005,
    'attention_lr': 0.005,
    'ffn_lr': 0.05,
    'output_lr': 0.05,

    # VFE loss weights
    'alpha': 0.075,
    'lambda_beta': 0.0,
    'lambda_gamma': 0.0,
    'kappa_gamma': 1.0,

    # Logging
    'log_interval': 100,
    'eval_interval': 1000,
    'checkpoint_interval': 25000,

    # Lightning
    'accelerator': 'auto',
    'devices': 'auto',
    'precision': '32-true',
    'accumulate_grad_batches': 1,

    # W&B
    'use_wandb': False,
    'wandb_project': 'gauge-transformer',
    'wandb_run_name': None,

    # Checkpointing
    'checkpoint_dir': 'checkpoints',
    'patience': 0,
}


# ============================================================================
# CONFIG 3: PURE FEP (Pure VFE — no autograd, no optimizer)
# ============================================================================
PURE_FEP_CONFIG = {
    # Belief geometry
    'vocab_size': 50257,
    'belief_dim': 32,
    'n_heads': 4,
    'head_dim': 8,
    'max_seq_len': 64,

    # E-step (inference = forward pass)
    'n_esteps': 12,
    'eta_E': 0.1,

    # M-step (learning = parameter update)
    'eta_M': 0.001,

    # Prior precision
    'alpha_b0': 1.0,
    'alpha_c0': 1.0,
    'hyper_var': 1.0,

    # Initialization
    'sigma_init': 1.0,
    'omega_init_scale': 0.01,

    # Trust regions
    'trust_region_mu': 1.0,
    'trust_region_sigma': 0.3,
    'trust_region_omega': 0.3,

    # SPD safeguards
    'spd_eps_min': 1e-4,
    'spd_kappa_max': 1e4,

    # Prior safeguards
    'prior_sigma_floor': 0.5,
    'prior_mu_max_norm': 3.0,

    # Causal masking
    'causal': True,

    # Training
    'batch_size': 8,
    'num_workers': 0,
    'max_steps': 15000,

    # Logging
    'log_interval': 100,
    'eval_interval': 1000,

    # Lightning (single-GPU only — DDP incompatible)
    'accelerator': 'auto',
    'precision': '32-true',

    # W&B
    'use_wandb': False,
    'wandb_project': 'gauge-transformer',
    'wandb_run_name': None,

    # Checkpointing
    'checkpoint_dir': 'checkpoints',
    'patience': 0,
}


# ============================================================================
# Implementation — no need to edit below this line
# ============================================================================

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)

from transformer.training.lightning_data import GaugeDataModule


def _build_vfe_dynamic(config, vocab_size):
    from transformer.core.model import GaugeTransformerLM
    from transformer.training.config import TrainingConfig
    from transformer.training.lightning_module import GaugeTransformerLitModule

    model_config = {
        'vocab_size': vocab_size,
        'embed_dim': config['embed_dim'],
        'n_layers': config['n_layers'],
        'irrep_spec': config['irrep_spec'],
        'hidden_dim': config['hidden_dim'],
        'max_seq_len': config['max_seq_len'],
        'kappa_beta': config.get('kappa_beta', 1.0),
    }
    model = GaugeTransformerLM(model_config)

    training_config = TrainingConfig(
        training_mode='vfe_dynamic',
        use_param_groups=True,
        mu_lr=config['mu_lr'],
        sigma_lr=config['sigma_lr'],
        phi_lr=config['phi_lr'],
        attention_lr=config['attention_lr'],
        ffn_lr=config['ffn_lr'],
        output_lr=config['output_lr'],
        alpha=config['alpha'],
        lambda_beta=config.get('lambda_beta', 0.0),
        lambda_gamma=config.get('lambda_gamma', 0.0),
        kappa_gamma=config.get('kappa_gamma', 1.0),
        warmup_steps=config['warmup_steps'],
        max_steps=config['max_steps'],
        lr_decay=config.get('lr_decay', 'cosine'),
        batch_size=config['batch_size'],
        max_seq_len=config['max_seq_len'],
        grad_clip=config.get('grad_clip', 1.0),
    )

    return GaugeTransformerLitModule(model, training_config)


def _build_standard(config, vocab_size):
    from transformer.baselines.standard_transformer import StandardTransformerLM
    from transformer.training.config import get_standard_config
    from transformer.training.lightning_module import GaugeTransformerLitModule

    model_config = {
        'vocab_size': vocab_size,
        'embed_dim': config['embed_dim'],
        'n_layers': config['n_layers'],
        'n_heads': config['n_heads'],
        'hidden_dim': config['hidden_dim'],
        'max_seq_len': config['max_seq_len'],
        'dropout': config.get('dropout', 0.1),
        'use_rope': config.get('use_rope', False),
        'disable_ffn': config.get('disable_ffn', False),
    }
    model = StandardTransformerLM(model_config)

    training_config = get_standard_config(
        learning_rate=config.get('learning_rate', 3e-4),
        warmup_steps=config.get('warmup_steps', 1000),
        max_steps=config['max_steps'],
        lr_decay=config.get('lr_decay', 'cosine'),
        batch_size=config['batch_size'],
        max_seq_len=config['max_seq_len'],
        weight_decay=config.get('weight_decay', 0.01),
        grad_clip=config.get('grad_clip', 1.0),
    )

    return GaugeTransformerLitModule(model, training_config)


def _build_pure_fep(config, vocab_size):
    from transformer.pure_vfe.config import PureVFEConfig
    from transformer.training.lightning_pure_vfe import PureVFELitModule

    pure_config = PureVFEConfig(
        vocab_size=vocab_size,
        belief_dim=config['belief_dim'],
        n_heads=config['n_heads'],
        head_dim=config['head_dim'],
        n_esteps=config.get('n_esteps', 12),
        eta_E=config.get('eta_E', 0.1),
        eta_M=config.get('eta_M', 0.001),
        max_seq_len=config['max_seq_len'],
        batch_size=config['batch_size'],
        alpha_b0=config.get('alpha_b0', 1.0),
        alpha_c0=config.get('alpha_c0', 1.0),
        hyper_var=config.get('hyper_var', 1.0),
        sigma_init=config.get('sigma_init', 1.0),
        omega_init_scale=config.get('omega_init_scale', 0.01),
        trust_region_mu=config.get('trust_region_mu', 1.0),
        trust_region_sigma=config.get('trust_region_sigma', 0.3),
        trust_region_omega=config.get('trust_region_omega', 0.3),
        spd_eps_min=config.get('spd_eps_min', 1e-4),
        spd_kappa_max=config.get('spd_kappa_max', 1e4),
        prior_sigma_floor=config.get('prior_sigma_floor', 0.5),
        prior_mu_max_norm=config.get('prior_mu_max_norm', 3.0),
        causal=config.get('causal', True),
    )

    return PureVFELitModule(pure_config)


def main():
    pl.seed_everything(SEED)

    # Select config by mode
    MODE_CONFIGS = {
        'standard': STANDARD_CONFIG,
        'VFE_dynamic': VFE_DYNAMIC_CONFIG,
        'pure_fep': PURE_FEP_CONFIG,
    }

    if ACTIVE_MODE not in MODE_CONFIGS:
        raise ValueError(f"Unknown ACTIVE_MODE: {ACTIVE_MODE!r}. "
                         f"Choose from: {list(MODE_CONFIGS.keys())}")

    config = MODE_CONFIGS[ACTIVE_MODE].copy()
    print(f"Mode: {ACTIVE_MODE}")
    print(f"Dataset: {DATASET}")

    # -----------------------------------------------------------
    # Data
    # -----------------------------------------------------------
    dm = GaugeDataModule(
        max_seq_len=config['max_seq_len'],
        batch_size=config['batch_size'],
        num_workers=config.get('num_workers', 0),
        dataset=DATASET,
    )
    dm.setup()

    # Override vocab_size from tokenizer
    config['vocab_size'] = dm.vocab_size

    # -----------------------------------------------------------
    # Model
    # -----------------------------------------------------------
    BUILDERS = {
        'standard': _build_standard,
        'VFE_dynamic': _build_vfe_dynamic,
        'pure_fep': _build_pure_fep,
    }
    lit_model = BUILDERS[ACTIVE_MODE](config, dm.vocab_size)

    # -----------------------------------------------------------
    # Callbacks
    # -----------------------------------------------------------
    eval_interval = config.get('eval_interval', 1000)

    callbacks = [
        ModelCheckpoint(
            dirpath=config.get('checkpoint_dir', 'checkpoints'),
            filename=f'{ACTIVE_MODE}' + '-{step}-{val/ce_loss:.4f}',
            monitor='val/ce_loss',
            mode='min',
            save_top_k=3,
            every_n_train_steps=eval_interval,
        ),
    ]

    # LR monitor only for modes with an optimizer
    if ACTIVE_MODE != 'pure_fep':
        callbacks.append(LearningRateMonitor(logging_interval='step'))

    patience = config.get('patience', 0)
    if patience > 0:
        callbacks.append(
            EarlyStopping(
                monitor='val/ce_loss',
                patience=patience,
                mode='min',
            )
        )

    # -----------------------------------------------------------
    # Logger
    # -----------------------------------------------------------
    if config.get('use_wandb', False):
        try:
            from pytorch_lightning.loggers import WandbLogger
            logger = WandbLogger(
                project=config.get('wandb_project', 'gauge-transformer'),
                name=config.get('wandb_run_name') or ACTIVE_MODE,
            )
        except ImportError:
            print("wandb not installed, falling back to CSV logger")
            logger = pl.loggers.CSVLogger(config.get('checkpoint_dir', 'checkpoints'))
    else:
        logger = pl.loggers.CSVLogger(config.get('checkpoint_dir', 'checkpoints'))

    # -----------------------------------------------------------
    # Trainer
    # -----------------------------------------------------------
    # PureFEP: single-GPU, no gradient clipping (no autograd)
    if ACTIVE_MODE == 'pure_fep':
        devices = 1
        gradient_clip_val = None
    else:
        devices = config.get('devices', 'auto')
        gradient_clip_val = config.get('grad_clip', 1.0)

    trainer = pl.Trainer(
        max_steps=config['max_steps'],
        accelerator=config.get('accelerator', 'auto'),
        devices=devices,
        precision=config.get('precision', '32-true'),
        gradient_clip_val=gradient_clip_val,
        accumulate_grad_batches=config.get('accumulate_grad_batches', 1),
        val_check_interval=eval_interval,
        log_every_n_steps=config.get('log_interval', 10),
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
