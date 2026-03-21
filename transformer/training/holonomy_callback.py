"""
Lightning Callback for Holonomy Monitoring During Training.
============================================================

Periodically computes holonomy diagnostics and logs them to the
Lightning logger (wandb, TensorBoard, CSV). Also saves publication-
quality figures to disk.

Usage:
    from transformer.training.holonomy_callback import HolonomyCallback

    callback = HolonomyCallback(
        log_interval=500,
        sample_size=500,
        output_dir='holonomy_figures/',
    )
    trainer = pl.Trainer(callbacks=[callback])
"""

import math
from pathlib import Path
from typing import Optional, List, Dict

import torch
import torch.nn as nn
import numpy as np
try:
    import pytorch_lightning as pl
except ImportError:
    import lightning.pytorch as pl

from transformer.analysis.holonomy import compute_holonomy, holonomy_statistics
from transformer.analysis.holonomy_metrics import (
    compute_holonomy_snapshot,
    compute_curvature_by_distance,
    compute_flatness_trajectory,
    HolonomySnapshot,
    HolonomyProfile,
)


class HolonomyCallback(pl.Callback):
    """Periodically compute and log holonomy diagnostics during training.

    Hooks into the model's forward pass via register_forward_hook to capture
    the transport operators (exp_delta) without modifying the model code.

    Args:
        log_interval: Compute holonomy every N training steps.
        sample_size: Number of random triples per computation.
        output_dir: Directory for saving figures. None = don't save figures.
        save_figures: Whether to save publication figures at each interval.
        figure_interval: Save figures every N holonomy computations (1 = every time).
        seed: Random seed for reproducible triple sampling.
    """

    def __init__(
        self,
        log_interval: int = 500,
        sample_size: int = 500,
        output_dir: Optional[str] = None,
        save_figures: bool = True,
        figure_interval: int = 5,
        seed: int = 42,
    ):
        super().__init__()
        self.log_interval = log_interval
        self.sample_size = sample_size
        self.output_dir = Path(output_dir) if output_dir else None
        self.save_figures = save_figures
        self.figure_interval = figure_interval
        self.seed = seed

        # State
        self.history: List[HolonomyProfile] = []
        self._computation_count = 0
        self._captured_exp_deltas: Dict[int, torch.Tensor] = {}
        self._hooks = []

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str = None):
        """Create output directory."""
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ):
        """Check if it's time to compute holonomy."""
        step = trainer.global_step
        if step == 0 or step % self.log_interval != 0:
            return

        profile = self._compute_holonomy(pl_module, step)
        if profile is None:
            return

        self.history.append(profile)
        self._computation_count += 1

        # Log to Lightning logger
        log_dict = profile.to_log_dict()
        for k, v in log_dict.items():
            pl_module.log(k, v, on_step=True, on_epoch=False)

        # Save figures periodically
        if (self.save_figures and self.output_dir
                and self._computation_count % self.figure_interval == 0):
            self._save_figures(step)

    def _compute_holonomy(
        self,
        pl_module: pl.LightningModule,
        step: int,
    ) -> Optional[HolonomyProfile]:
        """Extract exp_delta from the model and compute holonomy metrics.

        Approach: recompute exp_delta from the block's GaugeConnection and
        current embeddings rather than hooking into forward pass, since
        the callback runs between batches.
        """
        model = pl_module.model

        # Find blocks with non-flat transport
        blocks = self._find_gauge_blocks(model)
        if not blocks:
            return None

        snapshots = []
        all_norms = []

        with torch.no_grad():
            for layer_idx, block in blocks:
                if block.gauge_connection is None:
                    continue

                # Get current embeddings to compute delta
                exp_delta = self._extract_exp_delta(model, block, layer_idx)
                if exp_delta is None:
                    continue

                snap = compute_holonomy_snapshot(
                    exp_delta,
                    step=step,
                    layer=layer_idx,
                    head=0,  # aggregated across heads
                    sample_size=self.sample_size,
                    seed=self.seed,
                )
                snapshots.append(snap)
                all_norms.append(snap.mean_norm)

        if not snapshots:
            return None

        profile = HolonomyProfile(
            step=step,
            snapshots=snapshots,
            global_mean_norm=float(np.mean(all_norms)),
            global_max_norm=float(np.max(all_norms)),
        )
        return profile

    def _find_gauge_blocks(self, model: nn.Module) -> list:
        """Find all GaugeTransformerBlocks with non-flat transport."""
        blocks = []
        # Try both attribute names: 'transformer' (GaugeTransformerLM) and 'stack' (legacy)
        stack = getattr(model, 'transformer', None) or getattr(model, 'stack', None)
        if stack is not None:
            block_list = getattr(stack, 'blocks', None)
            if block_list is not None:
                for i, block in enumerate(block_list):
                    if getattr(block, 'non_flat_transport', False):
                        blocks.append((i, block))
        return blocks

    def _extract_exp_delta(
        self,
        model: nn.Module,
        block: nn.Module,
        layer_idx: int,
    ) -> Optional[torch.Tensor]:
        """Recompute exp(δ_ij · G) from current model state.

        Uses the block's GaugeConnection + model's embedding means
        to reconstruct what exp_delta would be for the current parameters.
        Uses raw delta (without cocycle_relaxation scaling) so diagnostics
        measure the connection's intrinsic curvature.
        """
        try:
            # Get embedding parameters
            embed = getattr(model, 'token_embedding', None)
            if embed is None:
                return None

            mu_embed = getattr(embed, 'mu_embed', None)
            if mu_embed is None:
                return None

            # Use a sample of embeddings (first N tokens)
            N = min(32, mu_embed.weight.shape[0])
            mu = mu_embed.weight[:N].unsqueeze(0)  # (1, N, K)

            # Get generators
            generators = getattr(model, 'generators', None)
            if generators is None:
                generators = getattr(block, 'cfg', {})
                if hasattr(generators, 'generators'):
                    generators = generators.generators
                else:
                    return None

            # Compute delta
            delta = block.gauge_connection(mu, mu)  # (1, N, N, n_gen)

            # Build exp(δ · G) using raw delta for diagnostics —
            # cocycle_relaxation=0 would zero out the connection and
            # produce trivially flat holonomy, hiding learned curvature.
            delta_matrix = torch.einsum('bija,akl->bijkl', delta, generators)
            exp_delta = torch.linalg.matrix_exp(delta_matrix.float())

            return exp_delta

        except Exception:
            return None

    def _save_figures(self, step: int):
        """Save holonomy figures to disk."""
        try:
            from transformer.visualization.holonomy_plots import (
                plot_holonomy_evolution,
                plot_holonomy_distribution,
            )
        except ImportError:
            return

        if not self.history:
            return

        trajectory = compute_flatness_trajectory(self.history)

        # Evolution figure
        if len(trajectory['steps']) > 1:
            plot_holonomy_evolution(
                steps=trajectory['steps'],
                global_mean=trajectory['global_mean'],
                global_max=trajectory['global_max'],
                per_layer_mean=trajectory['per_layer_mean'],
                output_path=self.output_dir / f'holonomy_evolution_step{step}.png',
            )

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Save final summary figures."""
        if self.output_dir and self.history:
            self._save_figures(trainer.global_step)

    def state_dict(self):
        """Save callback state for checkpointing."""
        return {
            'history_steps': [h.step for h in self.history],
            'history_global_mean': [h.global_mean_norm for h in self.history],
            'history_global_max': [h.global_max_norm for h in self.history],
            'computation_count': self._computation_count,
        }

    def load_state_dict(self, state_dict):
        """Restore callback state from checkpoint."""
        self._computation_count = state_dict.get('computation_count', 0)
