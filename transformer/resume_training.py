#!/usr/bin/env python3
"""
Resume Training from Checkpoint
===============================

Simple script to resume training from a checkpoint after power failure or interruption.
No CLI arguments needed - just edit the configuration below.

Usage:
    1. Set CHECKPOINT_PATH to your checkpoint file
    2. Set EXPERIMENT_DIR to the experiment directory (contains experiment_config.json)
    3. Optionally adjust TARGET_STEPS if you want to train longer
    4. Run: python transformer/resume_training.py
"""

import sys
import os

# Ensure project root is in path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import gc

import torch
import json
from pathlib import Path

from transformer.core.model import GaugeTransformerLM
from transformer.baselines.standard_transformer import StandardTransformerLM
from transformer.data import create_dataloaders, create_char_dataloaders
from transformer.training.config import TrainingConfig
from transformer.training.experiment_runner import run_test_evaluation, PublicationTrainer
from transformer.analysis.publication_metrics import PublicationMetrics


# =============================================================================
# CONFIGURATION - EDIT THESE VALUES
# =============================================================================

# Path to checkpoint file (e.g., checkpoint_step_179999.pt)
CHECKPOINT_PATH = "checkpoints_publication/ffn_VFE_dynamic/checkpoint_step_149999.pt"

# Experiment directory (contains experiment_config.json)
# If None, will use checkpoint's parent directory
EXPERIMENT_DIR = None

# Target total steps (set higher than checkpoint step to continue training)
# If None, will use original max_steps from config
TARGET_STEPS = 250000

# Override batch size (set to reduce memory usage if needed)
# If None, will use original batch_size from config
BATCH_SIZE = None

# Gradient accumulation steps (effective_batch = batch_size * grad_accum)
# Set higher to compensate for smaller batch size
GRAD_ACCUMULATION = 1

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# RESUME TRAINING LOGIC
# =============================================================================


def load_experiment_config(experiment_dir: Path) -> dict:
    """Load experiment configuration from JSON file."""
    config_path = experiment_dir / "experiment_config.json"
    if not config_path.exists():
        return {}  # Return empty dict if not found

    with open(config_path, 'r') as f:
        data = json.load(f)

    # Handle nested config structure
    if 'config' in data and isinstance(data['config'], dict):
        return data['config']
    return data


def extract_config_from_checkpoint(checkpoint: dict) -> dict:
    """Extract model config from checkpoint."""
    config = {}

    # Try 'config' key (most common)
    if 'config' in checkpoint:
        ckpt_config = checkpoint['config']
        if isinstance(ckpt_config, dict):
            config.update(ckpt_config)
        elif hasattr(ckpt_config, '__dict__'):
            # It's a dataclass or similar
            config.update(vars(ckpt_config))

    # Also check for individual keys that might be stored at top level
    for key in ['embed_dim', 'n_layers', 'vocab_size', 'max_seq_len', 'irrep_spec',
                'hidden_dim', 'n_heads', 'dropout', 'ffn_mode', 'gauge_group', 'gauge_dim']:
        if key in checkpoint and key not in config:
            config[key] = checkpoint[key]

    return config


def infer_config_from_state_dict(state_dict: dict) -> dict:
    """Infer model architecture from state_dict tensor shapes.

    This is the most reliable way to get the correct architecture
    when checkpoint config is missing or incorrect.
    """
    config = {}

    # Infer embed_dim from generator shape or embedding weight
    if 'generators' in state_dict:
        # generators shape: (n_gen, embed_dim, embed_dim)
        config['embed_dim'] = state_dict['generators'].shape[1]
    elif 'token_embed.mu_embed.weight' in state_dict:
        # mu_embed.weight shape: (vocab_size, embed_dim)
        config['embed_dim'] = state_dict['token_embed.mu_embed.weight'].shape[1]

    # Infer vocab_size from embedding
    if 'token_embed.mu_embed.weight' in state_dict:
        config['vocab_size'] = state_dict['token_embed.mu_embed.weight'].shape[0]
    elif 'out_proj.weight' in state_dict:
        config['vocab_size'] = state_dict['out_proj.weight'].shape[0]

    # Infer max_seq_len from positional encoding
    if 'pos_encoding.pos_phi' in state_dict:
        config['max_seq_len'] = state_dict['pos_encoding.pos_phi'].shape[0]

    # Infer n_layers by counting transformer blocks
    n_layers = 0
    for key in state_dict.keys():
        if 'transformer.blocks.' in key:
            # Extract block number from key like "transformer.blocks.0.attention..."
            parts = key.split('.')
            try:
                block_idx = int(parts[2])
                n_layers = max(n_layers, block_idx + 1)
            except (IndexError, ValueError):
                pass
    if n_layers > 0:
        config['n_layers'] = n_layers

    # Infer irrep_spec from attention head generators (from block 0 only —
    # all blocks share the same head structure).
    # Keys like: transformer.blocks.0.attention.head_generators.0.gen
    head_dims = []
    for key in state_dict.keys():
        if 'blocks.0.attention.head_generators.' in key and key.endswith('.gen'):
            # Shape: (n_gen, head_dim, head_dim)
            head_dim = state_dict[key].shape[1]
            # Extract head index
            parts = key.split('.')
            try:
                head_idx = int(parts[parts.index('head_generators') + 1])
                while len(head_dims) <= head_idx:
                    head_dims.append(None)
                head_dims[head_idx] = head_dim
            except (IndexError, ValueError):
                pass

    if head_dims and all(d is not None for d in head_dims):
        n_heads = len(head_dims)
        all_equal = len(set(head_dims)) == 1
        all_so3_valid = all(d == 1 or (d >= 3 and d % 2 == 1) for d in head_dims)

        # Cross-check against the global generator count to distinguish
        # GL(K) block-diagonal multi-head from an SO(3) irrep decomposition.
        # SO(3): n_gen = 3 * n_heads (3 generators per irrep head).
        # GL(K) block-diag multi-head: n_gen = n_heads * d_head^2.
        n_gen_global = state_dict['generators'].shape[0] if 'generators' in state_dict else None
        so3_expected = 3 * n_heads
        glk_expected = n_heads * head_dims[0] ** 2 if all_equal else None

        is_glk = False
        if not all_so3_valid:
            # Even dims are not valid SO(3) irreps → must be GL(K).
            is_glk = True
        elif n_gen_global is not None and all_equal:
            if n_gen_global == glk_expected and n_gen_global != so3_expected:
                is_glk = True
            elif n_gen_global == so3_expected:
                is_glk = False
            else:
                # Generator count matches neither pattern exactly; fall back
                # to head-dim heuristic (all_so3_valid already true here).
                is_glk = False

        if is_glk:
            if not all_equal:
                # Cross-head coupled GL(K) with non-uniform super-blocks.
                # Resuming this requires cross_couplings from the original
                # config — cannot safely reconstruct from state_dict alone.
                raise ValueError(
                    f"Detected GL(K) with non-uniform head dims {head_dims}. "
                    f"This indicates cross-head coupling; cannot infer "
                    f"cross_couplings from state_dict. Provide "
                    f"experiment_config.json with the original irrep_spec "
                    f"and cross_couplings."
                )
            d_head = head_dims[0]
            config['irrep_spec'] = [['fund', n_heads, d_head]]
            config['gauge_group'] = 'GLK'
            config['gauge_dim'] = d_head
        else:
            # SO(3) irrep decomposition: head_dim = 2*ell + 1
            irrep_spec = []
            for dim in head_dims:
                ell = (dim - 1) // 2
                irrep_spec.append([f'ℓ{ell}', 1, dim])
            config['irrep_spec'] = irrep_spec
            config['gauge_group'] = 'SO3'

    # Infer diagonal_covariance from sigma storage format
    # log_sigma_embed/log_sigma_diag = diagonal mode, log_sigma or sigma_embed = full mode
    if 'token_embed.log_sigma_embed.weight' in state_dict or 'token_embed.log_sigma_diag' in state_dict:
        config['diagonal_covariance'] = True
    elif 'token_embed.log_sigma' in state_dict or 'token_embed.sigma_embed' in state_dict:
        config['diagonal_covariance'] = False

    return config


def resume_training():
    """Resume training from checkpoint."""

    checkpoint_path = Path(CHECKPOINT_PATH)
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        print("\nAvailable checkpoints:")
        for pt_file in Path(".").rglob("checkpoint_step_*.pt"):
            print(f"  {pt_file}")
        return

    # Determine experiment directory
    if EXPERIMENT_DIR is not None:
        experiment_dir = Path(EXPERIMENT_DIR)
    else:
        experiment_dir = checkpoint_path.parent

    print("=" * 70)
    print("RESUMING TRAINING FROM CHECKPOINT")
    print("=" * 70)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Experiment dir: {experiment_dir}")
    print(f"Device: {DEVICE}")

    # Load checkpoint
    print("\nLoading checkpoint...")
    # Stage on CPU: avoids duplicating the checkpoint onto GPU while the
    # fresh model and optimizer are also being allocated. load_state_dict
    # copies per-parameter into the already-on-device target, so the full
    # state dict never needs to be GPU-resident.
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    start_step = checkpoint.get('step', 0)
    best_val_ce = checkpoint.get('best_val_ce', float('inf'))
    print(f"  Checkpoint step: {start_step}")
    print(f"  Best val CE: {best_val_ce:.4f}")

    # Load config - infer from state_dict first (most reliable), then merge others
    print("\nLoading config...")

    # Get model state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'model_state' in checkpoint:
        state_dict = checkpoint['model_state']
    else:
        state_dict = {}

    # FIRST: Infer architecture from actual tensor shapes (most reliable!)
    inferred_config = infer_config_from_state_dict(state_dict)
    if inferred_config:
        print(f"  Inferred from state_dict: {inferred_config}")

    # SECOND: Get config stored in checkpoint
    ckpt_config = extract_config_from_checkpoint(checkpoint)
    if ckpt_config:
        print(f"  Checkpoint config: {len(ckpt_config)} keys")

    # THIRD: Load experiment_config.json
    json_config = load_experiment_config(experiment_dir)
    if json_config:
        print(f"  experiment_config.json: {len(json_config)} keys")

    # Merge configs: inferred > checkpoint > json > defaults
    # Start with json, then checkpoint, then inferred (last wins for conflicts)
    config = {}
    config.update(json_config)
    config.update(ckpt_config)
    config.update(inferred_config)  # Inferred values override everything

    if not config:
        print("  WARNING: No config found!")
        print("  Will use defaults - this may not match your original model.")

    # Print key architecture params for verification
    print(f"\n  Model architecture (from state_dict):")
    print(f"    embed_dim: {config.get('embed_dim', 'NOT SET')}")
    print(f"    n_layers: {config.get('n_layers', 'NOT SET')}")
    print(f"    max_seq_len: {config.get('max_seq_len', 'NOT SET')}")
    print(f"    irrep_spec: {config.get('irrep_spec', 'NOT SET')}")

    # Override batch_size if specified
    if BATCH_SIZE is not None:
        original_batch = config.get('batch_size', 32)
        config['batch_size'] = BATCH_SIZE
        print(f"  Overriding batch_size: {original_batch} -> {BATCH_SIZE}")

    # Print memory-critical settings
    print(f"\n  Memory-critical settings:")
    print(f"    batch_size: {config.get('batch_size', 32)}")
    print(f"    hidden_dim: {config.get('hidden_dim', 'NOT SET')}")
    print(f"    diagonal_covariance: {config.get('diagonal_covariance', 'NOT SET')}")

    # Override max_steps if specified
    original_max_steps = config.get('max_steps', 200000)
    if TARGET_STEPS is not None:
        config['max_steps'] = TARGET_STEPS
        print(f"  Overriding max_steps: {original_max_steps} -> {TARGET_STEPS}")
    else:
        config['max_steps'] = original_max_steps

    if start_step >= config['max_steps']:
        print(f"\nWARNING: Checkpoint step ({start_step}) >= max_steps ({config['max_steps']})")
        print("Set TARGET_STEPS higher to continue training.")
        return

    remaining_steps = config['max_steps'] - start_step
    print(f"  Remaining steps: {remaining_steps}")

    # Create data loaders
    print("\nCreating data loaders...")
    # Try multiple keys for dataset name
    dataset_name = (config.get('dataset') or
                    config.get('dataset_name') or
                    'wikitext-103')  # Default to 103, not 2
    print(f"  Dataset: {dataset_name}")

    tokenizer_mode = config.get('tokenizer', 'auto')
    if tokenizer_mode == 'auto':
        use_char = config.get('vocab_size', 50257) <= 256
    else:
        use_char = (tokenizer_mode == 'char')

    if use_char:
        print(f"  Using character-level tokenizer")
        train_loader, val_loader, actual_vocab_size = create_char_dataloaders(
            max_seq_len=config.get('max_seq_len', 256),
            batch_size=config.get('batch_size', 32),
            num_workers=config.get('num_workers', 0),
            stride=config.get('stride', None),
            random_offset_per_epoch=config.get('random_offset_per_epoch', False),
            eval_stride=config.get('eval_stride', None),
            base_epoch_seed=config.get('stride_base_seed', 0),
        )
        test_loader = None
    else:
        print(f"  Using BPE tokenizer")
        train_loader, val_loader, test_loader, actual_vocab_size = create_dataloaders(
            max_seq_len=config.get('max_seq_len', 256),
            batch_size=config.get('batch_size', 32),
            vocab_size=config.get('vocab_size', 50257),
            num_workers=config.get('num_workers', 0),
            dataset=dataset_name,
            include_test=True,
            stride=config.get('stride', None),
            random_offset_per_epoch=config.get('random_offset_per_epoch', False),
            eval_stride=config.get('eval_stride', None),
            base_epoch_seed=config.get('stride_base_seed', 0),
        )

    config['vocab_size'] = actual_vocab_size
    print(f"  Vocab size: {actual_vocab_size}")

    # Ensure required model config keys have defaults
    model_defaults = {
        'kappa_beta': 1.0,
        'lambda_beta': 1.0,
        'M_alpha': 0.1,
        'M_beta': 1.0,
        'lambda_gamma': 0.0,
        'dropout': 0.1,
        'n_layers': 4,
        'n_heads': 1,
        'hidden_dim': config.get('embed_dim', 128) * 4,
        'ffn_mode': 'VFE_dynamic',
        'pos_encoding_mode': 'learned',
        'diagonal_covariance': True,
        'evolve_sigma': True,
        'evolve_phi': True,
        'tie_embeddings': True,
        'gauge_group': 'SO3',
        'gauge_dim': 3,
    }
    for key, default in model_defaults.items():
        if key not in config:
            config[key] = default

    # Create model
    print("\nCreating model...")
    ffn_mode = config.get('ffn_mode', 'VFE_dynamic')
    print(f"  FFN mode: {ffn_mode}")

    if ffn_mode == 'standard':
        model_config = {
            'vocab_size': actual_vocab_size,
            'embed_dim': config.get('embed_dim', 128),
            'n_layers': config.get('n_layers', 4),
            'n_heads': config.get('n_heads', 1),
            'hidden_dim': config.get('hidden_dim', config.get('embed_dim', 128) * 4),
            'max_seq_len': config.get('max_seq_len', 256),
            'dropout': config.get('dropout', 0.1),
        }
        model = StandardTransformerLM(model_config)
    else:
        model = GaugeTransformerLM(config)

    device = torch.device(DEVICE)
    model = model.to(device)

    # Load model weights
    print("Loading model weights from checkpoint...")
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    else:
        # Try loading directly (checkpoint might be just state dict)
        model.load_state_dict(checkpoint)

    # Free the source model state dict now that params have been copied
    # into the live model. Otherwise the CPU-resident copy lingers for the
    # entire remaining setup.
    checkpoint.pop('model_state_dict', None)
    checkpoint.pop('model_state', None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    # Create training config
    print("\nCreating training config...")

    # Print key performance settings
    print(f"  Performance settings:")
    print(f"    batch_size: {config.get('batch_size', 32)}")
    print(f"    grad_accumulation: {GRAD_ACCUMULATION}")
    print(f"    diagonal_covariance: {config.get('diagonal_covariance', True)}")

    train_config = TrainingConfig(
        # Training mode
        training_mode=config.get('training_mode', 'vfe_dynamic'),
        use_param_groups=config.get('use_param_groups', True),
        learning_rate=config.get('learning_rate', 3e-4),

        max_steps=config['max_steps'],
        warmup_steps=config.get('warmup_steps', 1000),

        # M-step learning rates (with old-name fallbacks for old checkpoints)
        M_mu_p_lr=config.get('M_mu_p_lr', config.get('mu_lr', 0.1)),
        M_sigma_p_lr=config.get('M_sigma_p_lr', config.get('sigma_lr', 0.005)),
        M_phi_lr=config.get('M_phi_lr', config.get('phi_lr', 0.01)),
        M_attention_lr=config.get('M_attention_lr', config.get('attention_lr', 0.01)),
        M_vfe_hyperparam_lr=config.get('M_vfe_hyperparam_lr', config.get('ffn_lr', 0.001)),
        M_output_lr=config.get('M_output_lr', config.get('output_lr', 0.001)),

        # Optimizer
        optimizer_type=config.get('optimizer_type', 'adamw'),
        non_embed_weight_decay=config.get('non_embed_weight_decay', config.get('weight_decay', 0.01)),
        embed_weight_decay=config.get('embed_weight_decay', 0.05),
        beta1=config.get('beta1', 0.9),
        beta2=config.get('beta2', 0.999),
        eps=config.get('eps', 1e-8),
        grad_clip=config.get('grad_clip', 1.0),
        grad_accumulation_steps=GRAD_ACCUMULATION,

        # LR schedule
        lr_decay=config.get('lr_decay', 'linear'),
        min_lr=config.get('min_lr', 3e-5),
        min_lr_ratio=config.get('min_lr_ratio', 0.1),
        kappa_warmup_steps=config.get('kappa_warmup_steps', 0),

        # Free energy weights
        M_alpha=config.get('M_alpha', config.get('alpha', 0.0)),
        M_beta=config.get('M_beta', config.get('beta', 0.0)),
        lambda_gamma=config.get('lambda_gamma', 0.0),
        kappa_gamma=config.get('kappa_gamma', 1.0),
        lambda_hyper=config.get('lambda_hyper', 0.0),
        detach_beta_m_step=config.get('detach_beta_m_step', True),
        normalize_ce_by_dim=config.get('normalize_ce_by_dim', True),
        ce_label_smoothing=config.get('ce_label_smoothing', 0.0),

        # Intervals
        log_interval=config.get('log_interval', 100),
        eval_interval=config.get('eval_interval', 500),
        checkpoint_interval=config.get('checkpoint_interval', 5000),

        # Checkpointing
        checkpoint_dir=experiment_dir,

        # Hardware / AMP / compile
        use_amp=config.get('use_amp', False),
        amp_dtype=config.get('amp_dtype', 'bfloat16'),
        use_compile=config.get('use_compile', False),
        compile_mode=config.get('compile_mode', 'reduce-overhead'),

        # Gauge group
        gauge_mode=config.get('gauge_mode', 'learned'),
        gauge_param=config.get('gauge_param', 'phi'),
        use_rope=config.get('use_rope', True),

        # P-FLOW and delta rule
        use_p_flow=config.get('use_p_flow', False),
        p_flow_ema_decay=config.get('p_flow_ema_decay', 0.99),
        sigma_ce_scale=config.get('sigma_ce_scale', 0.01),
        detach_phi=config.get('detach_phi', False),
        use_delta_rule_w_out=config.get('use_delta_rule_w_out', False),
        delta_rule_lr=config.get('delta_rule_lr', 0.001),

        # Phi preconditioning
        mass_phi=config.get('mass_phi', 0.05),
        omega_det_penalty=config.get('omega_det_penalty', 0.0),
        use_slk_projection=config.get('use_slk_projection', False),
        use_killing_form=config.get('use_killing_form', False),
        killing_form_sym_dampening=config.get('killing_form_sym_dampening', 0.1),

        # Stride windowing (threaded through for introspection / forward-compat).
        stride=config.get('stride', None),
        random_offset_per_epoch=config.get('random_offset_per_epoch', False),
        eval_stride=config.get('eval_stride', None),
        stride_base_seed=config.get('stride_base_seed', 0),
    )

    # Create publication metrics tracker for figures
    import time
    experiment_name = f"resumed_{time.strftime('%Y%m%d_%H%M%S')}"
    pub_metrics = PublicationMetrics(
        experiment_name=experiment_name,
        base_dir=experiment_dir / "publication_outputs"
    )

    # Create trainer (PublicationTrainer for metrics logging)
    print("\nInitializing trainer...")
    trainer = PublicationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_config,
        device=device,
        publication_metrics=pub_metrics,
    )

    # Restore training state
    print("\nRestoring training state...")
    trainer.global_step = start_step
    trainer.best_val_ce = best_val_ce

    # Restore optimizer state if available
    optimizer_restored = False
    if 'optimizer_state_dict' in checkpoint:
        try:
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            optimizer_restored = True
            print("  Restored optimizer state")
        except Exception as e:
            print(f"  Warning: Could not restore optimizer state: {e}")
            print("  Training will continue with fresh optimizer")
    elif 'optimizer_state' in checkpoint:
        try:
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state'])
            optimizer_restored = True
            print("  Restored optimizer state")
        except Exception as e:
            print(f"  Warning: Could not restore optimizer state: {e}")

    # Because the checkpoint was staged on CPU, optimizer moment tensors
    # land on CPU. Move them onto the training device once, so the first
    # optimizer.step() doesn't silently fall back or device-mismatch.
    if optimizer_restored:
        for state in trainer.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device, non_blocking=True)

    # Free the source optimizer state dict now that moments have been
    # copied into the live optimizer. Adam moments are ~2x model size.
    checkpoint.pop('optimizer_state_dict', None)
    checkpoint.pop('optimizer_state', None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Restore LR scheduler state if available
    if 'scheduler_state_dict' in checkpoint and trainer.scheduler is not None:
        try:
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("  Restored LR scheduler state")
        except Exception as e:
            print(f"  Warning: Could not restore scheduler state: {e}")
            print("  LR schedule will restart from current step")
    else:
        # Manually advance scheduler to match global_step so LR is correct
        if trainer.scheduler is not None and start_step > 0:
            for _ in range(start_step):
                trainer.scheduler.step()
            print(f"  Advanced scheduler to step {start_step}")

    # Restore GradScaler state if available (AMP training)
    if 'scaler_state_dict' in checkpoint and hasattr(trainer, 'scaler') and trainer.scaler is not None:
        try:
            trainer.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            print("  Restored GradScaler state")
        except Exception as e:
            print(f"  Warning: Could not restore scaler state: {e}")

    print(f"\nResuming from step {start_step} -> {config['max_steps']}")
    print("=" * 70)

    # Train! (PublicationTrainer handles metrics logging and figure generation)
    trainer.train()

    # Post-training figures that live outside PublicationTrainer.train()
    # (mirrors experiment_runner.run_single_experiment).
    try:
        from transformer.visualization.vfe_dynamics_plots import generate_all_vfe_figures
        metrics_csv = experiment_dir / 'metrics.csv'
        if metrics_csv.exists():
            vfe_fig_dir = experiment_dir / 'vfe_dynamics_figures'
            saved_figs = generate_all_vfe_figures(metrics_csv, vfe_fig_dir)
            if saved_figs:
                print(f"Generated {len(saved_figs)} VFE dynamics figures in {vfe_fig_dir}")
    except Exception as e:
        print(f"Warning: VFE dynamics figure generation failed: {e}")

    try:
        from transformer.visualization.training_plots import plot_head_kappas, load_metrics_csv
        metrics_csv = experiment_dir / 'metrics.csv'
        if metrics_csv.exists():
            csv_metrics = load_metrics_csv(metrics_csv)
            if csv_metrics.get('kappa_mean'):
                kappa_fig_path = experiment_dir / 'head_kappas.png'
                plot_head_kappas(csv_metrics, kappa_fig_path)
                print(f"Generated head kappa plot: {kappa_fig_path}")
    except Exception as e:
        print(f"Warning: Head kappa plot generation failed: {e}")

    # Run test set evaluation (not done by PublicationTrainer)
    if test_loader is not None:
        test_metrics = run_test_evaluation(
            model=model,
            test_loader=test_loader,
            device=device,
            vocab_size=config['vocab_size'],
            config=config,
        )
        # Save test metrics
        test_metrics_path = experiment_dir / 'test_metrics.json'
        with open(test_metrics_path, 'w') as f:
            json.dump(test_metrics, f, indent=2)
        print(f"Saved test metrics to: {test_metrics_path}")

    print("\n" + "=" * 70)
    print("ALL DONE!")
    print("=" * 70)


if __name__ == '__main__':
    resume_training()
