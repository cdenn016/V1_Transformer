"""
Holonomy Computation from Transport Operators
==============================================

Given transport matrices T_{ij} (d x d) between token positions,
compute the holonomy around closed loops:

    H_{ijk} = T_{ij} @ T_{jk} @ T_{ki}

For a flat bundle, H_{ijk} = lambda * I. The curvature metric
kappa_{ijk} measures deviation from this:

    kappa = ||H_normalized - I||_F / sqrt(d)

where H_normalized = H / |det(H)|^{1/d} removes overall scaling.
"""

import torch
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass, field

from .transport import TransportResult, _sample_triangles


@dataclass
class HolonomyResult:
    """Container for holonomy measurements on a single sentence."""
    # Per-triangle curvature values
    kappa: np.ndarray              # (num_triangles,)
    triangles: List[Tuple[int, int, int]]

    # Aggregates
    kappa_mean: float = 0.0
    kappa_median: float = 0.0
    kappa_max: float = 0.0
    kappa_std: float = 0.0

    # Per-triangle holonomy matrices (optional, memory-heavy)
    holonomies: Optional[List[np.ndarray]] = None

    # Singular value spread per triangle (more detailed curvature info)
    sv_spread: Optional[np.ndarray] = None  # (num_triangles,)

    # Layer-resolved curvature (if computed with varying depth)
    kappa_by_depth: Optional[dict] = None

    metadata: dict = field(default_factory=dict)


def loop_holonomy(
    T: torch.Tensor,
    i: int,
    j: int,
    k: int,
) -> Tuple[torch.Tensor, float, float]:
    """
    Compute holonomy around triangle (i, j, k).

    H_{ijk} = T_{ij} @ T_{jk} @ T_{ki}

    Args:
        T: transport tensor, shape (N, N, d, d)
        i, j, k: token indices

    Returns:
        H: holonomy matrix (d, d)
        kappa: normalized Frobenius deviation from identity
        sv_spread: std of singular values of H_normalized (0 = flat)
    """
    H = T[i, j] @ T[j, k] @ T[k, i]  # (d, d)

    d = H.shape[0]

    # Normalize out overall scaling: H_norm = H / |det(H)|^{1/d}
    det = torch.det(H)
    if det.abs() < 1e-30:
        # Degenerate — transport collapsed a direction
        return H, float('inf'), float('inf')

    scale = det.abs().pow(1.0 / d)
    H_norm = H / scale

    # Curvature metric: ||H_norm - I||_F / sqrt(d)
    kappa = torch.norm(H_norm - torch.eye(d, device=H.device), p='fro') / np.sqrt(d)

    # Singular value spread: for flat transport, all SVs are equal
    svs = torch.linalg.svdvals(H_norm)
    sv_spread = svs.std().item()

    return H, kappa.item(), sv_spread


def sentence_holonomy(
    transport_result: TransportResult,
    max_triangles: int = 500,
    store_holonomies: bool = False,
    seed: int = 42,
) -> HolonomyResult:
    """
    Compute holonomy statistics for a sentence.

    Samples token triples and computes holonomy around each.

    Args:
        transport_result: TransportResult from transport extraction
        max_triangles: maximum number of triangles to sample
        store_holonomies: if True, store the full H matrices (memory-heavy)
        seed: random seed for triangle sampling

    Returns:
        HolonomyResult with per-triangle and aggregate statistics
    """
    T = transport_result.transport
    N = transport_result.n_tokens

    triangles = _sample_triangles(N, max_triangles=max_triangles, seed=seed)

    kappas = []
    sv_spreads = []
    holonomies = [] if store_holonomies else None

    for i, j, k in triangles:
        H, kappa, sv_spread = loop_holonomy(T, i, j, k)
        kappas.append(kappa)
        sv_spreads.append(sv_spread)
        if store_holonomies:
            holonomies.append(H.cpu().numpy())

    kappas = np.array(kappas)
    sv_spreads = np.array(sv_spreads)

    # Filter out degenerate triangles
    finite_mask = np.isfinite(kappas)
    kappas_clean = kappas[finite_mask]

    return HolonomyResult(
        kappa=kappas,
        triangles=triangles,
        kappa_mean=float(np.mean(kappas_clean)) if len(kappas_clean) > 0 else float('nan'),
        kappa_median=float(np.median(kappas_clean)) if len(kappas_clean) > 0 else float('nan'),
        kappa_max=float(np.max(kappas_clean)) if len(kappas_clean) > 0 else float('nan'),
        kappa_std=float(np.std(kappas_clean)) if len(kappas_clean) > 0 else float('nan'),
        holonomies=holonomies,
        sv_spread=sv_spreads,
        metadata={
            'method': transport_result.method,
            'n_tokens': N,
            'd_model': transport_result.d_model,
            'n_triangles': len(triangles),
            'n_degenerate': int((~finite_mask).sum()),
        },
    )


def layer_resolved_holonomy(
    model,
    input_ids: torch.Tensor,
    transport_fn,
    max_triangles: int = 200,
    attention_mask=None,
) -> dict:
    """
    Compute holonomy as a function of depth (number of layers included).

    For each depth L = 1, 2, ..., n_layers:
        - Compute transport using only layers 0..L-1
        - Compute holonomy statistics

    This reveals at which depth curvature emerges.

    Args:
        model: pretrained transformer model
        input_ids: (1, N) token IDs
        transport_fn: callable(model, input_ids, layers=...) -> TransportResult
        max_triangles: triangles to sample per depth

    Returns:
        dict mapping depth -> HolonomyResult
    """
    # Determine number of layers
    if hasattr(model, 'h'):
        n_layers = len(model.h)
    elif hasattr(model, 'transformer'):
        n_layers = len(model.transformer.h)
    else:
        raise ValueError("Cannot determine number of layers")

    results = {}
    for depth in range(1, n_layers + 1):
        layers = list(range(depth))
        tr = transport_fn(
            model, input_ids, attention_mask=attention_mask, layers=layers
        )
        hr = sentence_holonomy(tr, max_triangles=max_triangles)
        results[depth] = hr

    return results
