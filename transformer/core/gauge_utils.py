"""
Shared Gauge-Geometry Utilities
================================

Shared utilities for gauge transport computations used across
attention.py, variational_ffn.py, and embeddings.py.

Consolidates duplicated matrix exponential and KL divergence patterns.
"""

import torch
from typing import Tuple


def stable_matrix_exp_pair(
    matrix: torch.Tensor,
    dim_threshold: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute exp(M) and exp(-M) with float64 upcasting for numerical stability.

    For GL(K) gauge groups with K >= dim_threshold, upcasts to float64 before
    computing the matrix exponential to prevent NaN from Padé scaling-squaring
    overflow when phi values grow large.

    Note on surjectivity:
        exp(M) always has det > 0 (since det(exp(M)) = exp(tr(M))), so the
        outputs live in GL⁺(K), the identity component. A single exp(M) does
        not cover all of GL⁺(K) — matrices with negative real eigenvalues of
        odd Jordan multiplicity have no real logarithm (Culver 1966). However,
        the pairwise product exp(M_i)·exp(-M_j) used for transport Ω_ij does
        cover all of GL⁺(K). For SO(K), exp: so(K) → SO(K) is surjective.

    Args:
        matrix: (..., d, d) matrix to exponentiate.
        dim_threshold: Upcast to float64 when d >= this value. Default 8.

    Returns:
        (exp_pos, exp_neg): Tuple of exp(M) and exp(-M), both same dtype as input.
    """
    d = matrix.shape[-1]
    if d >= dim_threshold:
        matrix_f64 = matrix.double()
        exp_pos = torch.linalg.matrix_exp(matrix_f64).to(matrix.dtype)
        exp_neg = torch.linalg.matrix_exp(-matrix_f64).to(matrix.dtype)
    else:
        exp_pos = torch.linalg.matrix_exp(matrix)
        exp_neg = torch.linalg.matrix_exp(-matrix)
    return exp_pos, exp_neg
