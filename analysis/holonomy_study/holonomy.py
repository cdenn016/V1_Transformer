"""
Holonomy (Path Composition Defect) for Causal Transformers
==========================================================

For causal models like GPT-2, T[i,j] is nonzero only when j <= i.
Closed loops are impossible — every triangle has at least one
anti-causal (zero) edge.

Instead, we measure *path composition defect* on ordered triples
a < b < c:

    D_{abc} = T[c,b] @ T[b,a] @ pinv(T[c,a])

For a flat bundle (path-independent transport):
    T[c,a] = T[c,b] @ T[b,a]  =>  D = I

Curvature kappa measures deviation from flatness:
    kappa = ||D_normalized - I||_F / sqrt(d)

where D_normalized = D / |det(D)|^{1/d} removes overall scaling.
"""

import torch
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass, field

from .transport import TransportResult, _sample_ordered_triples


@dataclass
class HolonomyResult:
    """Container for holonomy measurements on a single sentence."""
    # Per-triple curvature values
    kappa: np.ndarray              # (num_triples,)
    triangles: List[Tuple[int, int, int]]

    # Aggregates
    kappa_mean: float = 0.0
    kappa_median: float = 0.0
    kappa_max: float = 0.0
    kappa_std: float = 0.0

    # Per-triple defect matrices (optional, memory-heavy)
    holonomies: Optional[List[np.ndarray]] = None

    # Singular value spread per triple (more detailed curvature info)
    sv_spread: Optional[np.ndarray] = None  # (num_triples,)

    # Layer-resolved curvature (if computed with varying depth)
    kappa_by_depth: Optional[dict] = None

    metadata: dict = field(default_factory=dict)


def loop_holonomy(
    T: torch.Tensor,
    a: int,
    b: int,
    c: int,
) -> Tuple[torch.Tensor, float, float]:
    """
    Compute path composition defect for ordered triple (a, b, c) with a < b < c.

    Compares direct transport T[c,a] to indirect T[c,b] @ T[b,a].
    For flat transport these are identical; curvature means they differ.

    Primary metric (kappa): relative Frobenius defect
        kappa = ||T_indirect - T_direct||_F / ||T_direct||_F

    Secondary metric (sv_spread): std of log singular values of
        D = T_indirect @ solve(T_direct), measuring directional distortion.

    Args:
        T: transport tensor, shape (N, N, d, d)
        a, b, c: token indices with a < b < c

    Returns:
        T_indirect: the indirect transport matrix (d, d)
        kappa: relative Frobenius defect (0 = flat)
        sv_spread: std of log singular values of defect ratio (0 = flat)
    """
    # Direct transport: a -> c
    T_ca = T[c, a]  # (d, d)

    # Indirect transport: a -> b -> c
    T_ba = T[b, a]  # (d, d)
    T_cb = T[c, b]  # (d, d)
    T_indirect = T_cb @ T_ba  # (d, d)

    # Primary metric: relative Frobenius defect
    # No inversion, no determinant, no overflow — just norms
    diff_norm = torch.norm(T_indirect - T_ca, p='fro')
    ref_norm = torch.norm(T_ca, p='fro')
    kappa = (diff_norm / (ref_norm + 1e-30)).item()

    # Secondary metric: log-SV spread of defect ratio D = T_indirect @ inv(T_ca)
    # Use solve for numerical stability: D^T = solve(T_ca^T, T_indirect^T)
    try:
        D_T = torch.linalg.solve(T_ca.T, T_indirect.T)  # (d, d)
        svs = torch.linalg.svdvals(D_T)
        log_svs = torch.log(svs.clamp(min=1e-30))
        sv_spread = log_svs.std().item()
    except torch.linalg.LinAlgError:
        sv_spread = float('nan')

    return T_indirect, kappa, sv_spread


def sentence_holonomy(
    transport_result: TransportResult,
    max_triangles: int = 500,
    store_holonomies: bool = False,
    seed: int = 42,
) -> HolonomyResult:
    """
    Compute holonomy statistics for a sentence.

    Samples ordered token triples (a < b < c) and computes path
    composition defect for each.

    Args:
        transport_result: TransportResult from transport extraction
        max_triangles: maximum number of triples to sample
        store_holonomies: if True, store the full D matrices (memory-heavy)
        seed: random seed for triple sampling

    Returns:
        HolonomyResult with per-triple and aggregate statistics
    """
    T = transport_result.transport
    N = transport_result.n_tokens

    triples = _sample_ordered_triples(N, max_triples=max_triangles, seed=seed)

    kappas = []
    sv_spreads = []
    holonomies = [] if store_holonomies else None

    for a, b, c in triples:
        D, kappa, sv_spread = loop_holonomy(T, a, b, c)
        kappas.append(kappa)
        sv_spreads.append(sv_spread)
        if store_holonomies:
            holonomies.append(D.cpu().numpy())

    kappas = np.array(kappas)
    sv_spreads = np.array(sv_spreads)

    # Filter out degenerate triples
    finite_mask = np.isfinite(kappas)
    kappas_clean = kappas[finite_mask]

    return HolonomyResult(
        kappa=kappas,
        triangles=triples,
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
            'n_triangles': len(triples),
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
        max_triangles: triples to sample per depth

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
