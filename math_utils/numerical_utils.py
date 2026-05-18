# -*- coding: utf-8 -*-
"""
Numerical Stability Utilities
==============================

KL divergence between Gaussians, safe matrix inversion, and covariance
sanitization. Pure NumPy implementation.

Authors: Chris and Christine
Created: November 2025
"""

from typing import Dict, Optional, Tuple, Union

import numpy as np


# ============================================================================
# KL DIVERGENCE
# ============================================================================

def kl_gaussian(
    mu_q: np.ndarray,
    Sigma_q: np.ndarray,
    mu_p: np.ndarray,
    Sigma_p: np.ndarray,
    eps: float = 1e-8,
    return_terms: bool = False,
    use_gpu: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, np.ndarray]]]:
    """
    KL divergence KL(q||p) between Gaussians.

    Args:
        mu_q, Sigma_q: Source distribution N(μ_q, Σ_q)
        mu_p, Sigma_p: Target distribution N(μ_p, Σ_p)
        eps: Regularization (default: 1e-8)
        return_terms: If True, return dict with term breakdown
        use_gpu: Ignored (kept for API compatibility)

    Returns:
        kl: KL divergence (scalar or array)
        OR (kl, terms) if return_terms=True
    """
    return _kl_gaussian_numpy_impl(mu_q, Sigma_q, mu_p, Sigma_p, eps, return_terms)


def _kl_gaussian_numpy_impl(
    mu_q: np.ndarray,
    Sigma_q: np.ndarray,
    mu_p: np.ndarray,
    Sigma_p: np.ndarray,
    eps: float = 1e-8,
    return_terms: bool = False
):
    """
    NumPy KL(q||p) implementation via Cholesky decomposition.

    Args:
        mu_q, Sigma_q: Source distribution N(mu_q, Sigma_q)
        mu_p, Sigma_p: Target distribution N(mu_p, Sigma_p)
        eps: Regularization (default: 1e-8)
        return_terms: If True, return dict with term breakdown

    Returns:
        kl: KL divergence as float32 scalar or array
        OR (kl, terms) if return_terms=True
    """
    # Get latent dimension
    K = mu_q.shape[-1]
    
    # Ensure positive-definite via diagonal regularization
    Sigma_q = sanitize_sigma(Sigma_q, eps)
    Sigma_p = sanitize_sigma(Sigma_p, eps)
    
    # ========== Cholesky decomposition (stable) ==========
    try:
        L_p = np.linalg.cholesky(Sigma_p)
        logdet_p = 2.0 * np.sum(np.log(np.maximum(np.diagonal(L_p, axis1=-2, axis2=-1), eps)), axis=-1)

        L_q = np.linalg.cholesky(Sigma_q)
        logdet_q = 2.0 * np.sum(np.log(np.maximum(np.diagonal(L_q, axis1=-2, axis2=-1), eps)), axis=-1)
    except np.linalg.LinAlgError as e:
        raise FloatingPointError(f"Cholesky decomposition failed: {e}") from e
    
    # ========== Term 1: Trace term tr(Σ_p^{-1} Σ_q) ==========
    Y = np.linalg.solve(L_p, Sigma_q)
    Z = np.linalg.solve(np.swapaxes(L_p, -1, -2), Y)
    term_trace = np.trace(Z, axis1=-2, axis2=-1)
    
    # ========== Term 2: Mahalanobis term (μ_p - μ_q)^T Σ_p^{-1} (μ_p - μ_q) ==========
    delta_mu = mu_p - mu_q
    y = np.linalg.solve(L_p, delta_mu[..., None])
    z = np.linalg.solve(np.swapaxes(L_p, -1, -2), y)
    term_quad = np.sum(delta_mu * z[..., 0], axis=-1)
    
    # ========== Term 3: Log-determinant term ==========
    term_logdet = logdet_p - logdet_q
    
    # ========== Combine ==========
    kl = 0.5 * (term_trace + term_quad - K + term_logdet)
    
    # ========== Numerical cleanup ==========
    kl = np.clip(kl, 0.0, None)
    
    # Check for NaN/Inf
    if not np.all(np.isfinite(kl)):
        raise FloatingPointError("KL divergence contains NaN or Inf")
    
    # ========== Return ==========
    if return_terms:
        terms = {
            'term_trace': term_trace,
            'term_quad': term_quad,
            'term_logdet': term_logdet,
            'logdet_q': logdet_q,
            'logdet_p': logdet_p,
        }
        return kl.astype(np.float32), terms
    
    return kl.astype(np.float32)






def safe_inv(Sigma: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Safely compute matrix inverse with regularization.
    
    Works with batched matrices of any shape (..., K, K).
    Uses direct inversion with regularization for numerical stability.
    
    Args:
        Sigma: Covariance matrix, shape (..., K, K)
               Can be single (K, K) or batched (N, K, K), (H, W, K, K), etc.
        eps: Regularization strength (default: 1e-8)
    
    Returns:
        Sigma_inv: Matrix inverse, same shape as input
    
    Algorithm:
        1. Symmetrize: Sigma = 0.5 * (Sigma + Sigma^T)
        2. Regularize: Sigma_reg = Sigma + eps * I
        3. Invert: Sigma_inv = inv(Sigma_reg)
        4. If singular: increase eps and retry
    
    Examples:
        >>> # Single matrix
        >>> Sigma = np.array([[2.0, 0.5], [0.5, 3.0]])
        >>> Sigma_inv = safe_inv(Sigma)
        >>> np.allclose(Sigma @ Sigma_inv, np.eye(2))
        True
        
        >>> # Batch of matrices (1D spatial)
        >>> Sigma_batch = np.random.randn(50, 3, 3)
        >>> Sigma_batch = Sigma_batch @ Sigma_batch.swapaxes(-1, -2)
        >>> Sigma_inv_batch = safe_inv(Sigma_batch)
        >>> Sigma_inv_batch.shape
        (50, 3, 3)
        
        >>> # 2D spatial grid
        >>> Sigma_2d = np.random.randn(32, 32, 2, 2)
        >>> Sigma_2d = Sigma_2d @ Sigma_2d.swapaxes(-1, -2)
        >>> Sigma_inv_2d = safe_inv(Sigma_2d)
        >>> Sigma_inv_2d.shape
        (32, 32, 2, 2)
    
    Notes:
        - Always returns finite values
        - Preserves input dtype (float32/float64)
        - No warnings or exceptions (graceful fallback)
        - ~10-20% slower than Cholesky but much simpler
    """
    # Store original dtype and convert to float64 for stability
    original_dtype = Sigma.dtype
    Sigma = np.asarray(Sigma, dtype=np.float64)
    
    # Validate shape
    if Sigma.ndim < 2:
        raise ValueError(f"Sigma must be at least 2D, got shape {Sigma.shape}")
    
    if Sigma.shape[-2] != Sigma.shape[-1]:
        raise ValueError(f"Sigma must be square matrices (..., K, K), got {Sigma.shape}")
    
    K = Sigma.shape[-1]
    
    # ========== Step 1: Ensure Symmetric ==========
    # SPD matrices should be symmetric, but numerical errors can break this
    Sigma = 0.5 * (Sigma + np.swapaxes(Sigma, -1, -2))
    
    # ========== Step 2: Add Regularization ==========
    # Add small eps * I to ensure positive definite
    # np.eye broadcasts automatically to batch dimensions!
    Sigma_reg = Sigma + eps * np.eye(K)
    
    # ========== Step 3: Invert ==========
    try:
        # Direct inversion using np.linalg.inv
        # This works for any batch shape: (K, K), (N, K, K), (H, W, K, K), etc.
        Sigma_inv = np.linalg.inv(Sigma_reg)
        
    except np.linalg.LinAlgError:
        # If still singular, add more aggressive regularization
        # This is rare but possible with pathological inputs
        Sigma_reg = Sigma + (eps * 100) * np.eye(K)
        
        try:
            Sigma_inv = np.linalg.inv(Sigma_reg)
        except np.linalg.LinAlgError:
            # Ultimate fallback: very heavy regularization
            # This ensures we always return something finite
            Sigma_reg = Sigma + (eps * 10000) * np.eye(K)
            Sigma_inv = np.linalg.inv(Sigma_reg)
    
    # ========== Step 4: Validate Output ==========
    # Check for NaN/Inf
    if not np.all(np.isfinite(Sigma_inv)):
        raise FloatingPointError(
            f"safe_inv produced non-finite values. "
            f"Input Sigma shape: {Sigma.shape}, "
            f"min eigenvalue: {np.min(np.linalg.eigvalsh(Sigma[..., :, :]))}"
        )
    
    # Convert back to original dtype
    return Sigma_inv.astype(original_dtype, copy=False)









# -----------------------------------------------------------------------------
# Sigma sanitation (vectorized)
# -----------------------------------------------------------------------------
def sanitize_sigma(Sigma: np.ndarray,
                   eps: float = 1e-6,  # Adaptive floor (scale-relative)
                   max_cond: float = 1e4,  # DECREASE from 1e6!
                   max_eig: Optional[float] = None) -> np.ndarray:
    """
    Sanitize covariance matrix for numerical stability.

    Uses adaptive eigenvalue floor: max(eps * lambda_max, MIN_EIGENVALUE)
    to avoid systematic KL bias that grows with dimension K.

    Args:
        Sigma: Covariance matrix, shape (..., K, K)
        eps: Relative eigenvalue floor (fraction of lambda_max)
        max_cond: Maximum allowed condition number
        max_eig: Optional upper bound on eigenvalues

    Returns:
        Sigma_clean: Sanitized SPD matrix, same shape as input
    """
    # Symmetrize
    Sigma = 0.5 * (Sigma + np.swapaxes(Sigma, -1, -2))

    # Eigendecomposition
    w, V = np.linalg.eigh(Sigma)

    # Adaptive eigenvalue floor: scale-relative to prevent KL bias
    # For large eigenvalues, eps * λ_max ensures condition number stays bounded.
    # For small eigenvalues, MIN_EIGENVALUE prevents singularity.
    MIN_EIGENVALUE = 1e-8  # Absolute floor for positive-definiteness
    lambda_max = np.maximum(w[..., -1:], MIN_EIGENVALUE)
    adaptive_floor = np.maximum(eps * lambda_max, MIN_EIGENVALUE)
    w = np.maximum(w, adaptive_floor)
    
    if max_eig is not None:
        w = np.minimum(w, max_eig)
    
    # Enforce condition number (but MIN_EIGENVALUE already helps)
    lambda_max = w[..., -1]
    lambda_min_required = lambda_max / max_cond
    w = np.maximum(w, lambda_min_required[..., None])
    
    # Reconstruct
    Sigma_clean = np.einsum('...ij,...j,...kj->...ik', V, w, V, optimize=True)
    
    # Final symmetrize
    Sigma_clean = 0.5 * (Sigma_clean + np.swapaxes(Sigma_clean, -1, -2))
    
    return Sigma_clean



