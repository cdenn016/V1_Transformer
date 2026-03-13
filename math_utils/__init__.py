# -*- coding: utf-8 -*-
"""
Math Utilities Module
=====================

Mathematical utilities for gauge-equivariant transformer (NumPy CPU).

Modules:
    - transport: Gauge transport operators
    - push_pull: Gaussian push-pull operations
    - generators: SO(N) Lie algebra generators
    - numerical_utils: Safe matrix inversion, KL divergence
    - numerical_monitor: Debug logging
    - numba_kernels: Optional Numba acceleration
    - cuda_kernels: Optional CUDA acceleration
"""

# NumPy utilities
from .transport import compute_transport as np_transport_operator
from .push_pull import push_gaussian as np_transport_gaussian
from .generators import (
    generate_so3_generators,
    generate_soN_generators,
    generate_wedge2_generators,
    generate_sym2_traceless_generators,
    generate_multi_irrep_soN_generators,
)
from .numerical_utils import safe_inv

# Simple NumPy utilities (not in separate module)
import numpy as np

def symmetrize(M: np.ndarray) -> np.ndarray:
    """Symmetrize a matrix: (M + M^T) / 2"""
    return 0.5 * (M + np.swapaxes(M, -1, -2))

def np_ensure_spd(Sigma: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Ensure matrix is symmetric positive definite (NumPy version)."""
    Sigma = symmetrize(Sigma)
    K = Sigma.shape[-1]
    Sigma = Sigma + eps * np.eye(K, dtype=Sigma.dtype)
    return Sigma

__all__ = [
    # NumPy
    'np_transport_operator',
    'np_transport_gaussian',
    'generate_so3_generators',
    'generate_soN_generators',
    'generate_wedge2_generators',
    'generate_sym2_traceless_generators',
    'generate_multi_irrep_soN_generators',
    'safe_inv',
    'symmetrize',
    'np_ensure_spd',
]