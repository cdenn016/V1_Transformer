"""
Holonomy Metrics: Dataclasses and utilities for holonomy diagnostics.
====================================================================

Provides structured containers for holonomy measurements and helper
functions used by PublicationMetrics to track connection flatness
over training.

Depends on the low-level ``compute_holonomy`` from ``.holonomy``.
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict

from .holonomy import compute_holonomy


@dataclass
class HolonomySnapshot:
    r"""Holonomy statistics for a single layer at a single training step.

    Fields mirror the summary statistics from ``holonomy_statistics`` but
    add spectral-gap and Wilson-trace diagnostics for richer monitoring.

    Attributes:
        step: Training step at which this snapshot was taken.
        layer: Layer index.
        head: Head index (0 for head-averaged).
        variant: Which transport operator was measured — 'raw' applies the
            bare δ from gauge_connection (hypothetical curvature at
            cocycle_relaxation=1); 'scaled' applies the block's actual
            cocycle_relaxation factor and measures what training sees.
        mean_norm: Mean :math:`\|C_{ijk} - I\|_F` over sampled triples
            (computed over non-NaN entries only).
        std_norm: Standard deviation of holonomy norms (non-NaN only).
        median_norm: Median holonomy norm (non-NaN only).
        max_norm: Maximum holonomy norm observed (non-NaN only).
        frac_gt_001: Fraction of triples with norm > 0.01 (NaN counted as False).
        frac_gt_01: Fraction of triples with norm > 0.1 (NaN counted as False).
        mean_spectral_gap: Mean gap between largest and second-largest
            singular values of :math:`C_{ijk}`. Returns 0.0 if SVD fails.
        mean_wilson_trace: Mean :math:`|\mathrm{tr}(C_{ijk})| / K` (non-NaN only).
        nan_fraction: Fraction of sampled triples whose C contained NaN —
            typically from matrix_exp instability on outlier (i,j,k) whose
            delta spectral radius exceeds the float precision threshold.
        delta_max_spec: Max spectral norm of delta_matrix across ALL N²
            (i,j) edges — the quantity that governs matrix_exp stability.
        delta_p95_spec: 95th percentile of delta_matrix spectral norm.
        delta_mean_spec: Mean delta_matrix spectral norm.
        sample_size: Number of triples used.
    """
    step: int
    layer: int
    head: int
    mean_norm: float
    std_norm: float
    median_norm: float
    max_norm: float
    frac_gt_001: float
    frac_gt_01: float
    mean_spectral_gap: float
    mean_wilson_trace: float
    nan_fraction: float
    sample_size: int
    variant: str = 'raw'
    delta_max_spec: float = 0.0
    delta_p95_spec: float = 0.0
    delta_mean_spec: float = 0.0

    def to_log_dict(self, prefix: str = 'holonomy') -> Dict[str, float]:
        """Return a flat dictionary of metrics for logging.

        Key format: ``{prefix}/L{layer}_H{head}_{variant}/{metric}``.  The
        variant is embedded in the key so the raw and cocycle-scaled
        diagnostics share the same CSV row without collision.
        """
        key = f"{prefix}/L{self.layer}_H{self.head}_{self.variant}"
        return {
            f"{key}/mean_norm": self.mean_norm,
            f"{key}/std_norm": self.std_norm,
            f"{key}/median_norm": self.median_norm,
            f"{key}/max_norm": self.max_norm,
            f"{key}/frac_gt_001": self.frac_gt_001,
            f"{key}/frac_gt_01": self.frac_gt_01,
            f"{key}/spectral_gap": self.mean_spectral_gap,
            f"{key}/wilson_trace": self.mean_wilson_trace,
            f"{key}/nan_fraction": self.nan_fraction,
            f"{key}/delta_max_spec": self.delta_max_spec,
            f"{key}/delta_p95_spec": self.delta_p95_spec,
            f"{key}/delta_mean_spec": self.delta_mean_spec,
            f"{key}/sample_size": float(self.sample_size),
        }


@dataclass
class HolonomyProfile:
    r"""Aggregated holonomy profile across all layers at a single step.

    Attributes:
        step: Training step.
        snapshots: Per-layer snapshots.
        global_mean_norm: Mean holonomy norm across all layers.
        global_max_norm: Maximum holonomy norm across all layers.
    """
    step: int
    snapshots: List[HolonomySnapshot] = field(default_factory=list)
    global_mean_norm: float = 0.0
    global_max_norm: float = 0.0

    def to_log_dict(self, prefix: str = 'holonomy') -> Dict[str, float]:
        """Return a flat dictionary of global + per-snapshot metrics."""
        d: Dict[str, float] = {
            f"{prefix}/global_mean_norm": self.global_mean_norm,
            f"{prefix}/global_max_norm": self.global_max_norm,
        }
        for snap in self.snapshots:
            d.update(snap.to_log_dict(prefix=prefix))
        return d


def compute_holonomy_snapshot(
    exp_delta: torch.Tensor,
    step: int = 0,
    layer: int = 0,
    head: int = 0,
    sample_size: int = 500,
    seed: int = 42,
    variant: str = 'raw',
    delta_stats: Dict[str, float] = None,
) -> HolonomySnapshot:
    r"""Compute a single holonomy snapshot for one layer.

    Calls ``compute_holonomy`` and derives spectral-gap / Wilson-trace
    diagnostics from the resulting :math:`C_{ijk}` matrices.

    Args:
        exp_delta: ``(B, N, N, K, K)`` transport operator tensor.
        step: Current training step (for bookkeeping).
        layer: Layer index (for bookkeeping).
        head: Head index (for bookkeeping).
        sample_size: Number of random triples to sample.
        seed: Random seed for reproducibility.

    Returns:
        A populated ``HolonomySnapshot``.
    """
    C, norms, _ = compute_holonomy(exp_delta, sample_size=sample_size, seed=seed)
    # C: (B, n_triples, K, K), norms: (B, n_triples)
    norms_flat = norms.detach().cpu().float().reshape(-1)
    K = C.shape[-1]

    # A NaN in C (typically from float32 matrix_exp on outlier triples)
    # propagates to norms.  Compute reductions over the non-NaN subset only,
    # and expose nan_fraction as a first-class diagnostic so the failure
    # rate is visible rather than hidden behind NaN outputs.
    nan_mask = torch.isnan(norms_flat)
    nan_fraction = float(nan_mask.float().mean().item())
    valid = norms_flat[~nan_mask]
    if valid.numel() == 0:
        mean_norm = float('nan')
        std_norm = float('nan')
        median_norm = float('nan')
        max_norm = float('nan')
    else:
        mean_norm = float(valid.mean().item())
        std_norm = float(valid.std(unbiased=False).item()) if valid.numel() > 1 else 0.0
        median_norm = float(valid.median().item())
        max_norm = float(valid.max().item())
    # frac_gt_* treat NaN as False (survives the reduction); denominator
    # is the total triple count, matching the "fraction of all sampled
    # triples that exceeded the threshold" semantics.
    frac_gt_001 = float((norms_flat > 0.01).float().mean().item())
    frac_gt_01 = float((norms_flat > 0.1).float().mean().item())

    # Spectral gap: gap between top-2 singular values of C.  SVD on a NaN
    # matrix raises; on a well-conditioned matrix it returns singular
    # values.  We filter NaN-bearing C-slices out before calling svdvals
    # so a single bad triple does not zero the mean_spectral_gap via the
    # except-fallback.
    C_flat = C.detach().cpu().float().reshape(-1, K, K)
    C_nan_mask = torch.isnan(C_flat).any(dim=(-2, -1))
    C_clean = C_flat[~C_nan_mask]
    try:
        if C_clean.shape[0] == 0:
            mean_spectral_gap = float('nan')
        else:
            svs = torch.linalg.svdvals(C_clean)  # (n_clean, K)
            if svs.shape[-1] >= 2:
                gaps = svs[:, 0] - svs[:, 1]
                mean_spectral_gap = float(gaps.mean().item())
            else:
                mean_spectral_gap = 0.0
    except Exception:
        mean_spectral_gap = 0.0

    # Wilson trace: |tr(C)| / K, over non-NaN slices only.
    if C_clean.shape[0] == 0:
        mean_wilson_trace = float('nan')
    else:
        traces = torch.diagonal(C_clean, dim1=-2, dim2=-1).sum(dim=-1).abs() / K
        mean_wilson_trace = float(traces.mean().item())

    _ds = delta_stats or {}
    return HolonomySnapshot(
        step=step,
        layer=layer,
        head=head,
        mean_norm=mean_norm,
        std_norm=std_norm,
        median_norm=median_norm,
        max_norm=max_norm,
        frac_gt_001=frac_gt_001,
        frac_gt_01=frac_gt_01,
        mean_spectral_gap=mean_spectral_gap,
        mean_wilson_trace=mean_wilson_trace,
        nan_fraction=nan_fraction,
        sample_size=sample_size,
        variant=variant,
        delta_max_spec=float(_ds.get('delta_max_spec', 0.0)),
        delta_p95_spec=float(_ds.get('delta_p95_spec', 0.0)),
        delta_mean_spec=float(_ds.get('delta_mean_spec', 0.0)),
    )


def compute_curvature_by_distance(
    exp_delta: torch.Tensor,
    max_distance: int = 20,
    samples_per_distance: int = 200,
    seed: int = 42,
) -> Dict[str, object]:
    r"""Compute mean holonomy norm as a function of token distance.

    For each distance ``d`` in ``[1, max_distance]``, samples triples
    ``(i, i+d, i+2d)`` (wrapping) and computes mean :math:`\|C - I\|_F`.

    Args:
        exp_delta: ``(B, N, N, K, K)`` transport operator tensor.
        max_distance: Maximum token separation to probe.
        samples_per_distance: Triples per distance bin.
        seed: Random seed.

    Returns:
        Dict with keys ``distances``, ``mean_norms``, ``std_norms``
        suitable for unpacking into a plotting function.
    """
    B, N, _, K, _ = exp_delta.shape
    rng = np.random.RandomState(seed)

    distances = []
    mean_norms = []
    std_norms = []

    for d in range(1, min(max_distance + 1, N // 2)):
        triples = []
        for _ in range(samples_per_distance):
            i = rng.randint(0, N)
            j = (i + d) % N
            k = (i + 2 * d) % N
            if len({i, j, k}) == 3:
                triples.append([i, j, k])
        if len(triples) < 3:
            continue
        triples_t = torch.tensor(triples, device=exp_delta.device)
        _, norms, _ = compute_holonomy(exp_delta, triples=triples_t)
        norms_np = norms.detach().cpu().numpy().flatten()
        distances.append(d)
        mean_norms.append(float(np.mean(norms_np)))
        std_norms.append(float(np.std(norms_np)))

    return {
        'distances': np.array(distances),
        'mean_norms': np.array(mean_norms),
        'std_norms': np.array(std_norms),
    }


def compute_flatness_trajectory(
    holonomy_history: List[HolonomyProfile],
) -> Dict[str, object]:
    r"""Extract arrays from a sequence of ``HolonomyProfile`` for plotting.

    Args:
        holonomy_history: List of profiles recorded during training.

    Returns:
        Dict with keys:
        - ``steps``: array of training steps.
        - ``global_mean``: array of global mean norms.
        - ``global_max``: array of global max norms.
        - ``per_layer_mean``: dict mapping layer index to array of means.
        - ``layer_indices``: sorted list of layer indices seen.
    """
    steps = []
    global_mean = []
    global_max = []
    per_layer: Dict[int, List[float]] = {}

    for profile in holonomy_history:
        steps.append(profile.step)
        global_mean.append(profile.global_mean_norm)
        global_max.append(profile.global_max_norm)
        for snap in profile.snapshots:
            per_layer.setdefault(snap.layer, []).append(snap.mean_norm)

    layer_indices = sorted(per_layer.keys())

    return {
        'steps': np.array(steps),
        'global_mean': np.array(global_mean),
        'global_max': np.array(global_max),
        'per_layer_mean': {k: np.array(v) for k, v in per_layer.items()},
        'layer_indices': layer_indices,
    }
