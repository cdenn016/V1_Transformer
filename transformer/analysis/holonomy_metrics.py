"""
Holonomy Metrics: Training-integrated curvature diagnostics.
=============================================================

Provides a HolonomyTracker that plugs into the training loop (or Lightning
callback) to periodically compute and log holonomy statistics. Designed
for the ablation: "does language have curvature?" — comparing models
trained with vs without δ_ij.

Key metrics:
    - mean/max/median ‖C_ijk - I‖_F  (holonomy norm)
    - fraction of triples above curvature thresholds
    - per-layer holonomy profile
    - per-head holonomy decomposition
    - Wilson loop spectrum (eigenvalues of C_ijk)

All metrics are returned as plain dicts suitable for wandb.log() or CSV.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from transformer.analysis.holonomy import (
    compute_holonomy,
    holonomy_statistics,
    holonomy_by_token_pairs,
)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class HolonomySnapshot:
    """Single snapshot of holonomy diagnostics at one training step."""
    step: int
    layer: int
    head: int
    # Summary statistics
    mean_norm: float = 0.0
    max_norm: float = 0.0
    median_norm: float = 0.0
    std_norm: float = 0.0
    # Threshold fractions
    frac_gt_001: float = 0.0   # fraction with ‖C-I‖ > 0.01
    frac_gt_01: float = 0.0    # fraction with ‖C-I‖ > 0.1
    frac_gt_10: float = 0.0    # fraction with ‖C-I‖ > 1.0
    # Spectral info (eigenvalue spread of C_ijk)
    mean_spectral_gap: float = 0.0  # mean |λ_max - λ_min| across triples
    # Wilson loop trace
    mean_wilson_trace: float = 0.0  # mean |tr(C_ijk)/K - 1|

    def to_log_dict(self, prefix: str = 'holonomy') -> Dict[str, float]:
        """Convert to flat dict for logging (wandb / CSV)."""
        tag = f'{prefix}/L{self.layer}_H{self.head}'
        return {
            f'{tag}/mean_norm': self.mean_norm,
            f'{tag}/max_norm': self.max_norm,
            f'{tag}/median_norm': self.median_norm,
            f'{tag}/std_norm': self.std_norm,
            f'{tag}/frac_gt_0.01': self.frac_gt_001,
            f'{tag}/frac_gt_0.1': self.frac_gt_01,
            f'{tag}/frac_gt_1.0': self.frac_gt_10,
            f'{tag}/spectral_gap': self.mean_spectral_gap,
            f'{tag}/wilson_trace_dev': self.mean_wilson_trace,
        }


@dataclass
class HolonomyProfile:
    """Full holonomy profile across all layers and heads at one step."""
    step: int
    snapshots: List[HolonomySnapshot] = field(default_factory=list)
    # Aggregated across all layers/heads
    global_mean_norm: float = 0.0
    global_max_norm: float = 0.0

    def to_log_dict(self, prefix: str = 'holonomy') -> Dict[str, float]:
        """Flat dict for logging."""
        d = {
            f'{prefix}/global_mean_norm': self.global_mean_norm,
            f'{prefix}/global_max_norm': self.global_max_norm,
        }
        for snap in self.snapshots:
            d.update(snap.to_log_dict(prefix))
        return d


# =============================================================================
# Core Metric Functions
# =============================================================================

def compute_holonomy_snapshot(
    exp_delta: torch.Tensor,
    step: int,
    layer: int,
    head: int,
    sample_size: int = 500,
    seed: int = 42,
) -> HolonomySnapshot:
    """Compute a full holonomy snapshot from exp(δ_ij · G) matrices.

    Args:
        exp_delta: (B, N, N, K, K) per-edge transport perturbation.
        step: Current training step.
        layer: Layer index.
        head: Head index.
        sample_size: Number of random triples to sample.
        seed: Random seed for reproducibility.

    Returns:
        HolonomySnapshot with all metrics populated.
    """
    C, norms, _ = compute_holonomy(exp_delta, sample_size=sample_size, seed=seed)
    norms_cpu = norms.detach().cpu().float()
    C_cpu = C.detach().cpu().float()

    B, n_triples, K, _ = C_cpu.shape

    # Basic statistics
    snap = HolonomySnapshot(
        step=step, layer=layer, head=head,
        mean_norm=norms_cpu.mean().item(),
        max_norm=norms_cpu.max().item(),
        median_norm=norms_cpu.median().item(),
        std_norm=norms_cpu.std().item(),
        frac_gt_001=(norms_cpu > 0.01).float().mean().item(),
        frac_gt_01=(norms_cpu > 0.1).float().mean().item(),
        frac_gt_10=(norms_cpu > 1.0).float().mean().item(),
    )

    # Spectral analysis: eigenvalues of C_ijk
    # C is close to identity, so eigenvalues near 1 = flat
    try:
        C_flat = C_cpu.reshape(-1, K, K)
        # Use a subset for speed
        max_eig_samples = min(C_flat.shape[0], 200)
        eigvals = torch.linalg.eigvals(C_flat[:max_eig_samples])  # complex
        eigvals_abs = eigvals.abs()  # (n, K)
        spectral_gaps = eigvals_abs.max(dim=-1).values - eigvals_abs.min(dim=-1).values
        snap.mean_spectral_gap = spectral_gaps.mean().item()
    except Exception:
        snap.mean_spectral_gap = 0.0

    # Wilson loop trace: |tr(C)/K - 1| measures deviation from identity
    traces = torch.einsum('bnii->bn', C_cpu)  # (B, n_triples)
    wilson_dev = (traces / K - 1.0).abs()
    snap.mean_wilson_trace = wilson_dev.mean().item()

    return snap


def compute_curvature_by_distance(
    exp_delta: torch.Tensor,
    positions: Optional[torch.Tensor] = None,
    n_bins: int = 10,
    sample_size: int = 2000,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Compute holonomy norm as a function of triangle "size" (mean pairwise distance).

    If position indices are used (default), distance = mean |i-j| for triple (i,j,k).
    This tests whether curvature is local (short-range) or global (long-range).

    Args:
        exp_delta: (B, N, N, K, K) per-edge transport.
        positions: Optional (N,) position coordinates. If None, uses index positions.
        n_bins: Number of distance bins.
        sample_size: Number of triples to sample.
        seed: Random seed.

    Returns:
        Dict with 'bin_centers', 'mean_norms', 'std_norms', 'counts'.
    """
    B, N, _, K, _ = exp_delta.shape

    C, norms, triples = compute_holonomy(exp_delta, sample_size=sample_size, seed=seed)
    norms_cpu = norms.detach().cpu().float().mean(dim=0).numpy()  # average over batch
    triples_cpu = triples.cpu().numpy()

    if positions is None:
        positions = np.arange(N, dtype=np.float32)
    else:
        positions = positions.cpu().numpy() if torch.is_tensor(positions) else positions

    # Mean pairwise distance for each triple
    i, j, k = triples_cpu[:, 0], triples_cpu[:, 1], triples_cpu[:, 2]
    dists = (np.abs(positions[i] - positions[j])
             + np.abs(positions[j] - positions[k])
             + np.abs(positions[k] - positions[i])) / 3.0

    # Bin
    bin_edges = np.linspace(dists.min(), dists.max() + 1e-6, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    mean_norms = np.zeros(n_bins)
    std_norms = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=int)

    for b in range(n_bins):
        mask = (dists >= bin_edges[b]) & (dists < bin_edges[b + 1])
        if mask.sum() > 0:
            mean_norms[b] = norms_cpu[mask].mean()
            std_norms[b] = norms_cpu[mask].std()
            counts[b] = mask.sum()

    return {
        'bin_centers': bin_centers,
        'mean_norms': mean_norms,
        'std_norms': std_norms,
        'counts': counts,
    }


def compute_flatness_trajectory(
    holonomy_history: List[HolonomyProfile],
) -> Dict[str, np.ndarray]:
    """Extract holonomy evolution over training for plotting.

    Args:
        holonomy_history: List of HolonomyProfile objects from training.

    Returns:
        Dict with 'steps', 'global_mean', 'global_max',
        'per_layer_mean' (n_steps, n_layers), etc.
    """
    steps = np.array([h.step for h in holonomy_history])
    global_mean = np.array([h.global_mean_norm for h in holonomy_history])
    global_max = np.array([h.global_max_norm for h in holonomy_history])

    # Per-layer breakdown
    if holonomy_history and holonomy_history[0].snapshots:
        layers = sorted(set(s.layer for s in holonomy_history[0].snapshots))
        per_layer_mean = np.zeros((len(holonomy_history), len(layers)))
        for t, profile in enumerate(holonomy_history):
            layer_norms = {}
            for s in profile.snapshots:
                layer_norms.setdefault(s.layer, []).append(s.mean_norm)
            for li, layer in enumerate(layers):
                if layer in layer_norms:
                    per_layer_mean[t, li] = np.mean(layer_norms[layer])
    else:
        per_layer_mean = np.zeros((len(holonomy_history), 0))

    return {
        'steps': steps,
        'global_mean': global_mean,
        'global_max': global_max,
        'per_layer_mean': per_layer_mean,
    }
