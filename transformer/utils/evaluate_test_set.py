"""
Test-Set Evaluation for GaugeTransformerLM Checkpoints
======================================================

Evaluate trained checkpoints on the held-out TEST split for final
publication metrics.  Computes token-weighted cross-entropy and
perplexity with all VFE regularisation terms disabled (pure CE).

The test set should only be used once, after all hyperparameter tuning
is complete.

Usage:
    python -m transformer.utils.evaluate_test_set --checkpoint path/to/best_model.pt

    # Evaluate multiple checkpoints
    python -m transformer.utils.evaluate_test_set \\
        --checkpoint checkpoints/vfe_best.pt checkpoints/std_best.pt

    # Full evaluation (all batches)
    python -m transformer.utils.evaluate_test_set --checkpoint path/to/model.pt --full
"""

import torch
import argparse
import json
from pathlib import Path
import numpy as np
from typing import Dict, Optional, Tuple
from torch.utils.data import DataLoader

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", message="Failed to find cuobjdump")
warnings.filterwarnings("ignore", message="Failed to find nvdisasm")


def create_test_dataloader(
    max_seq_len: int = 128,
    batch_size: int = 8,
    vocab_size: Optional[int] = None,
    vocab_mapping: Optional[Dict] = None,
    num_workers: int = 0,
    dataset: str = 'wikitext-103',
) -> Tuple[DataLoader, int]:
    """
    Create a TEST-split DataLoader.

    Args:
        max_seq_len: Maximum sequence length.
        batch_size: Batch size for evaluation.
        vocab_size: Expected vocabulary size (must match training).
        vocab_mapping: Token-to-id mapping from training (ensures ID consistency).
        num_workers: Number of data loading workers.
        dataset: Dataset identifier ('wikitext-103', 'wikitext-2', or 'wiki-ja').

    Returns:
        test_loader: DataLoader over the test split (unshuffled, no drop_last).
        actual_vocab_size: Vocabulary size reported by the dataset.
    """
    from transformer.data.datasets import (
        WikiText2TiktokenDataset,
        WikiText2Dataset,
        TIKTOKEN_AVAILABLE,
        BPE_AVAILABLE,
        _worker_init_fn,
    )

    if not BPE_AVAILABLE:
        raise ImportError("BPE tokenization requires tiktoken or transformers")

    print(f"Creating TEST dataloader for {dataset.upper()}...")

    if TIKTOKEN_AVAILABLE:
        test_dataset = WikiText2TiktokenDataset(
            split='test',  # KEY: Use test split
            max_seq_len=max_seq_len,
            vocab_size=vocab_size,
            vocab_mapping=vocab_mapping,
            dataset=dataset,
        )
    else:
        test_dataset = WikiText2Dataset(
            split='test',  # KEY: Use test split
            max_seq_len=max_seq_len,
            vocab_size=vocab_size,
            vocab_mapping=vocab_mapping,
        )

    actual_vocab_size = test_dataset.get_vocab_size()

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # No shuffling for evaluation
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=False,  # Keep all data for evaluation
        worker_init_fn=_worker_init_fn,
    )

    print(f"  Test batches: {len(test_loader):,}")
    print(f"  Vocabulary:   {actual_vocab_size:,} tokens")

    return test_loader, actual_vocab_size


def load_checkpoint(checkpoint_path: str, device: str = 'cpu') -> Dict:
    """Load a raw checkpoint dict and print training-state summary."""
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = checkpoint['config']
    step = checkpoint.get('step', 0)
    best_val_loss = checkpoint.get('best_val_loss', None)

    print(f"  Step: {step:,}")
    if best_val_loss is not None:
        print(f"  Best val loss: {best_val_loss:.4f}")
        print(f"  Best val PPL:  {np.exp(min(best_val_loss, 20.0)):.2f}")

    return checkpoint


def get_config_val(cfg, key, default=None):
    """Get value from config (handles both dict and dataclass)."""
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def evaluate_on_test(
    checkpoint_path: str,
    max_samples: Optional[int] = None,
    device: str = 'auto',
) -> Dict[str, float]:
    """
    Evaluate a GaugeTransformerLM checkpoint on the held-out test split.

    Runs a pure cross-entropy forward pass (all VFE regularisation coefficients
    set to zero) and reports token-weighted loss and perplexity.

    Args:
        checkpoint_path: Path to checkpoint file.
        max_samples: Maximum number of *samples* to evaluate (None = all).
            Sample-based limiting gives consistent evaluation across configs
            with different batch sizes.
        device: Device to use ('auto', 'cuda', 'cpu').

    Returns:
        Dict with keys: test/loss, test/ce_loss, test/perplexity,
        test/tokens_evaluated, test/batches_evaluated, checkpoint.
    """
    from transformer.core.model import GaugeTransformerLM
    from transformer.train import compute_free_energy_loss

    # Device selection
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")

    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_path, device)
    config = checkpoint['config']

    # Extract model configuration
    model_state = checkpoint['model_state_dict']
    if 'token_embed.mu_embed.weight' in model_state:
        vocab_size = model_state['token_embed.mu_embed.weight'].shape[0]
        embed_dim = model_state['token_embed.mu_embed.weight'].shape[1]
    else:
        vocab_size = get_config_val(config, 'vocab_size', 50257)
        embed_dim = get_config_val(config, 'embed_dim', 100)

    max_seq_len = get_config_val(config, 'max_seq_len', 128)
    n_layers = get_config_val(config, 'n_layers', 1)
    batch_size = get_config_val(config, 'batch_size', 8)
    dataset = get_config_val(config, 'dataset', 'wikitext-103')

    print(f"\nModel configuration:")
    print(f"  Embed dim (K): {embed_dim}")
    print(f"  Layers:        {n_layers}")
    print(f"  Max seq len:   {max_seq_len}")
    print(f"  Vocab size:    {vocab_size:,}")

    # Build model config dict
    config_dict = {
        'vocab_size': vocab_size,
        'embed_dim': embed_dim,
        'n_layers': n_layers,
        'max_seq_len': max_seq_len,
        'hidden_dim': get_config_val(config, 'hidden_dim', embed_dim * 4),
        'kappa_beta': get_config_val(config, 'kappa_beta', 1.0),
        'epsilon': get_config_val(config, 'epsilon', 1e-8),
        'pos_encoding_mode': get_config_val(config, 'pos_encoding_mode', 'none'),
        'evolve_sigma': get_config_val(config, 'evolve_sigma', False),
        'evolve_phi': get_config_val(config, 'evolve_phi', False),
        'tie_embeddings': get_config_val(config, 'tie_embeddings', False),
        'dropout': get_config_val(config, 'dropout', 0.1),
        'irrep_spec': get_config_val(config, 'irrep_spec', [('fund', 5, 20)]),
        'diagonal_covariance': get_config_val(config, 'diagonal_covariance', True),
    }

    # Create and load model
    print(f"\nCreating model...")
    model = GaugeTransformerLM(config_dict)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    total_params = model.get_num_params(non_embedding=False)
    print(f"  Total params: {total_params:,}")

    # Create dataloaders with include_test=True to ensure train vocab mapping
    # is shared across all splits (train, val, AND test).
    # This is critical: without sharing the mapping, each split builds its own
    # token->id map from its own frequency counts, causing ID mismatches.
    from transformer.data import create_dataloaders
    _, _, test_loader, actual_vocab_size = create_dataloaders(
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        vocab_size=vocab_size,
        num_workers=0,
        dataset=dataset,
        include_test=True,
    )

    # Target padding uses -100 (PyTorch cross_entropy ignore_index default).
    # Dataset.pad_token_id is for INPUT padding only — targets always use -100.
    pad_token_id = -100

    # Evaluate
    print(f"\n{'='*70}")
    print("TEST SET EVALUATION")
    print(f"{'='*70}")

    total_batches = len(test_loader)
    if max_samples is not None:
        print(f"Evaluating up to {max_samples} samples from {total_batches} total batches...")
    else:
        print(f"Evaluating on ALL {total_batches} batches...")

    total_loss = 0.0
    total_ce = 0.0
    total_tokens = 0
    num_batches = 0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if max_samples is not None and total_samples >= max_samples:
                break

            input_ids, target_ids = batch
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            # Compute loss (pure CE evaluation, no regularization terms)
            loss, metrics = compute_free_energy_loss(
                model,
                input_ids,
                target_ids,
                M_alpha=0.0,
                M_beta=0.0,
                lambda_gamma=0.0,
                kappa_gamma=1.0,
                pad_token_id=pad_token_id,
            )

            # Count non-padding tokens for proper weighting
            # metrics['loss/ce'] is mean over non-padding tokens (from ignore_index)
            non_pad = (target_ids != pad_token_id).sum().item()
            total_loss += loss.item() * non_pad
            total_ce += metrics.get('loss/ce_raw', metrics['loss/ce']) * non_pad
            total_tokens += non_pad
            num_batches += 1
            total_samples += input_ids.size(0)

            if (batch_idx + 1) % 50 == 0:
                current_ppl = np.exp(total_ce / max(total_tokens, 1))
                print(f"  {total_samples} samples ({num_batches} batches) - Running PPL: {current_ppl:.2f}")

    # Final results - token-weighted average (weights by non-padding tokens)
    avg_loss = total_loss / max(total_tokens, 1)
    avg_ce = total_ce / max(total_tokens, 1)
    perplexity = np.exp(avg_ce)

    results = {
        'test/loss': avg_loss,
        'test/ce_loss': avg_ce,
        'test/perplexity': perplexity,
        'test/tokens_evaluated': total_tokens,
        'test/batches_evaluated': num_batches,
        'checkpoint': str(checkpoint_path),
    }

    print(f"\n{'='*70}")
    print("TEST SET RESULTS")
    print(f"{'='*70}")
    print(f"Test Loss:       {avg_loss:.4f}")
    print(f"Test CE Loss:    {avg_ce:.4f}")
    print(f"Test Perplexity: {perplexity:.2f}")
    print(f"Tokens evaluated: {total_tokens:,}")
    print(f"\nComparison to baselines:")
    print(f"  Random ({vocab_size:,} vocab): PPL {vocab_size:,}")
    print(f"  Your model:            PPL {perplexity:.2f}")
    print(f"  Improvement:           {vocab_size/perplexity:.1f}x better")
    print(f"{'='*70}\n")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate checkpoint(s) on WikiText-103 TEST set'
    )
    parser.add_argument(
        '--checkpoint', '-c',
        type=str,
        nargs='+',
        required=True,
        help='Path(s) to checkpoint file(s)'
    )
    parser.add_argument(
        '--max_samples', '-n',
        type=int,
        default=None,
        help='Max samples to evaluate (default: all). Uses sample count '
             'instead of batch count for consistency across batch sizes.'
    )
    parser.add_argument(
        '--full',
        action='store_true',
        help='Evaluate on full test set (same as --max_samples None)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device to use'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output JSON file for results'
    )

    args = parser.parse_args()

    max_samples = None if args.full else args.max_samples

    all_results = {}

    for checkpoint_path in args.checkpoint:
        path = Path(checkpoint_path)
        if not path.exists():
            print(f"\n❌ Checkpoint not found: {checkpoint_path}")
            continue

        print(f"\n{'#'*70}")
        print(f"# EVALUATING: {path.name}")
        print(f"{'#'*70}")

        try:
            results = evaluate_on_test(
                str(path),
                max_samples=max_samples,
                device=args.device,
            )
            all_results[str(path)] = results
        except Exception as e:
            print(f"❌ Error evaluating {checkpoint_path}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        for path, results in all_results.items():
            print(f"  {Path(path).name}: PPL {results['test/perplexity']:.2f}")

    # Save results
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    return all_results


if __name__ == '__main__':
    main()
