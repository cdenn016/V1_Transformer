"""
Validation-Set Evaluation for GaugeTransformerLM Checkpoints
============================================================

Quick validation-set evaluation of a trained checkpoint.  Computes
cross-entropy loss and perplexity with all VFE regularisation terms
disabled (alpha, lambda_beta, lambda_gamma = 0).

Click-to-run: edit ``CONFIG`` near the bottom of this file, then press
Run. No CLI arguments (per CLAUDE.md).
"""

import torch
from pathlib import Path
import numpy as np

from transformer.core.model import GaugeTransformerLM
from transformer.data import create_dataloaders
from transformer.train import compute_free_energy_loss


def evaluate_checkpoint(checkpoint_path: str, max_batches: int = 50, trusted: bool = False):
    """
    Load a GaugeTransformerLM checkpoint and evaluate on the validation split.

    Runs pure cross-entropy evaluation (all VFE regularisation coefficients
    zeroed) and prints loss, perplexity, and a qualitative assessment.

    Args:
        checkpoint_path: Path to checkpoint file (e.g. best_model.pt).
        max_batches: Number of validation batches to evaluate.
    """
    print("="*70)
    print("CHECKPOINT EVALUATION")
    print("="*70)

    # Load checkpoint
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=not trusted)

    config = checkpoint['config']
    step = checkpoint.get('step', 0)
    best_val_loss = checkpoint.get('best_val_loss', None)

    print(f"\nCheckpoint info:")
    print(f"  Step: {step}")
    if best_val_loss is not None:
        print(f"  Best val loss: {best_val_loss:.4f}")
        print(f"  Best val PPL: {np.exp(min(best_val_loss, 20.0)):.2f}")

    # Handle both dict and dataclass configs
    def get_config_val(cfg, key, default=None):
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        else:
            return getattr(cfg, key, default)

    # Extract vocab_size from checkpoint weights (more reliable than config)
    model_state = checkpoint['model_state_dict']
    if 'token_embed.mu_embed.weight' in model_state:
        vocab_size = model_state['token_embed.mu_embed.weight'].shape[0]
        embed_dim = model_state['token_embed.mu_embed.weight'].shape[1]
    else:
        vocab_size = get_config_val(config, 'vocab_size', 2000)
        embed_dim = get_config_val(config, 'embed_dim', 11)

    max_seq_len = get_config_val(config, 'max_seq_len', 40)
    n_layers = get_config_val(config, 'n_layers', 2)
    batch_size = get_config_val(config, 'batch_size', 4)

    print(f"\nModel config:")
    print(f"  Agents (N): {max_seq_len}")
    print(f"  Fiber (K): {embed_dim}")
    print(f"  Layers: {n_layers}")
    print(f"  Vocab: {vocab_size:,}")

    # Create model
    print(f"\nCreating model...")

    # Always build complete config dict (regardless of config type)
    config_dict = {
        'vocab_size': vocab_size,
        'embed_dim': embed_dim,
        'n_layers': n_layers,
        'max_seq_len': max_seq_len,
        'hidden_dim': get_config_val(config, 'hidden_dim', embed_dim * 4),
        'kappa_beta': get_config_val(config, 'kappa_beta', 1.0),
        'epsilon': get_config_val(config, 'epsilon', 1e-8),
        'pos_encoding_mode': get_config_val(config, 'pos_encoding_mode', 'learned'),
        'evolve_sigma': get_config_val(config, 'evolve_sigma', False),
        'evolve_phi': get_config_val(config, 'evolve_phi', False),
        'tie_embeddings': get_config_val(config, 'tie_embeddings', True),
        'dropout': get_config_val(config, 'dropout', 0.1),
        'irrep_spec': get_config_val(config, 'irrep_spec', [('ℓ0', 5, 1), ('ℓ1', 2, 3)]),
        'diagonal_covariance': get_config_val(config, 'diagonal_covariance', True),
    }

    model = GaugeTransformerLM(config_dict)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    total_params = model.get_num_params(non_embedding=False)
    print(f"  Total params: {total_params:,}")

    # Create data loaders
    print(f"\nLoading data...")
    train_loader, val_loader, actual_vocab_size = create_dataloaders(
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        vocab_size=vocab_size,
        num_workers=0,
    )

    # Check vocab size mismatch
    if actual_vocab_size != vocab_size:
        print(f"\n⚠ WARNING: Vocab size mismatch!")
        print(f"  Model expects: {vocab_size}")
        print(f"  Data has:      {actual_vocab_size}")
        print(f"  Adjusting model vocab size to match data...")

        # Recreate model with correct vocab size
        config_dict['vocab_size'] = actual_vocab_size
        model = GaugeTransformerLM(config_dict)

        # Load weights with strict=False to handle size mismatch
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model.eval()

        vocab_size = actual_vocab_size

    # Target padding uses -100 (PyTorch cross_entropy ignore_index default).
    # Dataset.pad_token_id is for INPUT padding only — targets always use -100.
    pad_token_id = -100

    # Evaluate on validation set
    print(f"\n{'='*70}")
    print("VALIDATION EVALUATION")
    print(f"{'='*70}")
    print(f"Evaluating on {max_batches} batches...")

    total_ce_tokens = 0.0
    total_tokens = 0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= max_batches:
                break

            input_ids, target_ids = batch

            # Count non-padding tokens for proper weighting
            non_pad = (target_ids != pad_token_id).sum().item()

            # Compute loss (all gauge terms disabled for pure CE evaluation)
            loss, metrics = compute_free_energy_loss(
                model,
                input_ids,
                target_ids,
                M_alpha=0.0,
                M_beta=0.0,
                lambda_gamma=0.0,
                kappa_gamma=1.0,
                lambda_hyper=0.0,
                pad_token_id=pad_token_id,
                mass_phi=0.0,
            )

            ce_loss = metrics.get('loss/ce_raw', metrics['loss/ce'])
            # Token-weighted accumulation: ce_loss is per-token average,
            # multiply by non_pad to recover total, then divide by total_tokens
            total_ce_tokens += ce_loss * non_pad
            total_tokens += non_pad
            num_batches += 1

            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx + 1}/{max_batches}...")

    # Results (token-weighted average)
    avg_ce = total_ce_tokens / max(1, total_tokens)
    perplexity = np.exp(avg_ce)

    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"Validation CE:   {avg_ce:.4f}")
    print(f"Validation PPL:  {perplexity:.2f}")
    print(f"\nComparison:")
    print(f"  Random baseline: ~{vocab_size:,} PPL")
    print(f"  Your model:      {perplexity:.2f} PPL")
    print(f"  Improvement:     {vocab_size/perplexity:.1f}x better!")

    # Performance assessment
    print(f"\nAssessment:")
    if perplexity < 150:
        print("  ✨ EXCELLENT! Comparable to much larger models!")
    elif perplexity < 250:
        print("  ✓ GOOD! Model is learning well.")
    elif perplexity < 400:
        print("  ~ ACCEPTABLE. Room for improvement.")
    else:
        print("  ⚠ POOR. Model barely learning.")

    print(f"\n{'='*70}\n")


CONFIG = {
    'checkpoint':  'checkpoints_realistic/best_model.pt',
    'max_batches': 50,
}


def main() -> None:
    checkpoint_path = Path(CONFIG['checkpoint'])
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        print(f"\nAvailable checkpoints:")
        checkpoint_dir = checkpoint_path.parent
        if checkpoint_dir.exists():
            for f in sorted(checkpoint_dir.glob("*.pt")):
                print(f"  - {f}")
        return

    # Self-saved checkpoints embed the config dataclass — opt into the
    # pickle path (refused by the new weights_only=True default).
    evaluate_checkpoint(str(checkpoint_path), CONFIG['max_batches'], trusted=True)


if __name__ == '__main__':
    main()