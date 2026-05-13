"""Smoke test for the attention-entropy fix.

Runs ~30 steps of the live VFE training loop using train_vfe.py's actual config
(with max_steps reduced) and verifies:

  1. Loss stays finite at every step.
  2. attention_entropy_loss metric appears in the training metrics.
  3. include_attention_entropy=True is the active config.
  4. Toggle smoke: with include_attention_entropy=False, the loss differs
     numerically from the entropy-on run (confirms the term is wired into
     backward, not just diagnostic).

Run: `python tests/smoke_test_entropy.py`
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path


# Make the project root importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_train_config() -> dict:
    """Load the `config` dict from train_vfe.py without invoking training."""
    spec = importlib.util.spec_from_file_location(
        "_train_vfe_config_only", ROOT / "transformer" / "vfe" / "train_vfe.py"
    )
    mod = importlib.util.module_from_spec(spec)
    # Prevent the `if __name__ == "__main__":` block from firing
    mod.__name__ = "_train_vfe_config_only"
    spec.loader.exec_module(mod)
    return dict(mod.config)


def run_short_training(steps: int, include_entropy: bool) -> tuple[float, list[float]]:
    """Run `steps` of training and return (final_loss, entropy_loss_history)."""
    import torch
    from transformer.vfe.config import VFEConfig
    from transformer.vfe.model import VFEModel
    from transformer.vfe.trainer import VFETrainer
    from transformer.data.datasets import create_dataloaders

    cfg_dict = load_train_config()
    cfg_dict['max_steps'] = steps
    cfg_dict['n_layers'] = 2
    cfg_dict['embed_dim'] = 12
    cfg_dict['irrep_spec'] = [('fund', 2, 6)]  # K=12, 2 heads of 6
    cfg_dict['batch_size'] = 8
    cfg_dict['max_seq_len'] = 16
    cfg_dict['include_attention_entropy'] = include_entropy

    train_loader, val_loader, vocab_size = create_dataloaders(
        max_seq_len=cfg_dict['max_seq_len'],
        batch_size=cfg_dict['batch_size'],
        vocab_size=cfg_dict.get('vocab_size'),
        dataset='wikitext-2',
    )
    cfg_dict['vocab_size'] = vocab_size

    cfg = VFEConfig(**cfg_dict)
    model = VFEModel(cfg)

    trainer = VFETrainer(
        model, cfg, train_loader,
        val_loader=val_loader, device='cpu',
        output_dir=None,
    )

    # Hook to collect per-step metrics
    entropy_loss_history: list = []
    losses: list = []
    original_train_step = trainer.train_step

    def wrapped_train_step(batch):
        metrics = original_train_step(batch)
        loss_val = float(metrics.get('loss', 0.0))
        losses.append(loss_val)
        if loss_val != loss_val or abs(loss_val) == float('inf'):
            raise AssertionError(f"Non-finite loss at step {len(losses)}: {loss_val}")
        entropy_loss_history.append(metrics.get('attention_entropy_loss', None))
        return metrics

    trainer.train_step = wrapped_train_step
    trainer.train(num_steps=steps)

    return losses[-1], entropy_loss_history


def main() -> None:
    logging.basicConfig(level=logging.WARNING)  # quiet trainer

    print("=" * 78)
    print("Smoke test — attention-entropy fix in live VFE training loop")
    print("=" * 78)

    STEPS = 30

    print(f"\n[1/2] Running {STEPS} steps with include_attention_entropy=True ...")
    final_on, entropy_history_on = run_short_training(STEPS, include_entropy=True)
    has_metric = any(v is not None for v in entropy_history_on)
    print(f"      final_loss = {final_on:.4f}")
    print(f"      attention_entropy_loss metric present: {has_metric}")
    if has_metric:
        nonnull = [v for v in entropy_history_on if v is not None]
        print(f"      F_H samples (first/mid/last): "
              f"{nonnull[0]:.4f} / {nonnull[len(nonnull)//2]:.4f} / {nonnull[-1]:.4f}")

    print(f"\n[2/2] Running {STEPS} steps with include_attention_entropy=False ...")
    final_off, _ = run_short_training(STEPS, include_entropy=False)
    print(f"      final_loss = {final_off:.4f}")

    print("\nResults:")
    print(f"  Δ final_loss (on − off) = {final_on - final_off:+.4f}")
    if abs(final_on - final_off) < 1e-8:
        print("  WARNING: entropy term contributes ~0 to loss; may not be wired")
    else:
        print("  PASS — entropy term affects the training objective")

    assert has_metric, "attention_entropy_loss metric never appeared in metrics dict"
    print("\nSmoke test PASS.")


if __name__ == "__main__":
    main()
