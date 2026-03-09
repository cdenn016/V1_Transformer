# -*- coding: utf-8 -*-
"""
Variational Free Energy FFN (Refactored)
==========================================

Dynamic-β VFE with attention-belief co-evolution.

E-step at each iteration:
    1. β = softmax(-KL(q||Ω[q])/κ)       [recompute from current beliefs]
    2. ∂F/∂μ = α(μ-μ_p)/σ_p + λΣβ(∂KL/∂μ) + Σ KL(∂β/∂μ) + ∂CE/∂μ
    3. μ ← μ - η·F⁻¹·∂F/∂μ              [natural gradient descent]
    4. σ ← retract_spd(σ, -η·∂F/∂σ)      [SPD-preserving update]
    5. φ ← retract(φ, -η_φ·∂F/∂φ)        [Lie group retraction]

Refactored from legacy variational_ffn.py (2,685 lines → ~600 lines):
- Config-driven VariationalFFNDynamic (no 30+ params)
- Phi update helper eliminates ~100 lines of duplication
- Dead code removed (MockMultiAgentSystem, convert_torch_to_numpy_system)
- Retraction/geometry functions moved to gauge_utils.py
- No AMP wrappers (incompatible with the theory)
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from transformer_v2.config import GaugeTransformerConfig
from transformer_v2.gauge_utils import (
    stable_matrix_exp_pair,
    safe_spd_inv,
    retract_spd,
    retract_spd_diagonal,
    retract_phi,
)
from transformer_v2.attention import compute_attention_weights
from transformer_v2.kl_ops import compute_transport_operators


# =============================================================================
# VFE Gradient Computation
# =============================================================================

def compute_vfe_gradients(
    mu_q: torch.Tensor,
    sigma_q: torch.Tensor,
    mu_p: torch.Tensor,
    sigma_p: torch.Tensor,
    beta: torch.Tensor,
    phi: torch.Tensor,
    generators: torch.Tensor,
    alpha: 'float | torch.Tensor' = 0.01,
    lambda_belief: float = 1.0,
    kappa: float = 1.0,
    eps: float = 1e-6,
    cached_transport: Optional[dict] = None,
    compute_sigma_align_grad: bool = True,
    sigma_softmax_coupling: bool = False,
    irrep_dims: Optional[List[int]] = None,
    chunk_size: Optional[int] = None,
    enforce_orthogonal: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute VFE gradients for μ and σ.

    Dispatches to block-diagonal, chunked, or standard paths based on args.
    """
    while sigma_q.dim() > 3 and sigma_q.shape[-1] == 1:
        sigma_q = sigma_q.squeeze(-1)
    while sigma_p.dim() > 3 and sigma_p.shape[-1] == 1:
        sigma_p = sigma_p.squeeze(-1)

    B, N, K = mu_q.shape
    is_diagonal = sigma_q.dim() == 3

    # Block-diagonal paths
    if irrep_dims is not None:
        if is_diagonal:
            return _vfe_gradients_block_diagonal_diag(
                mu_q, sigma_q, mu_p, sigma_p, beta, phi, generators,
                alpha, lambda_belief, kappa, eps, irrep_dims,
                compute_sigma_align_grad, enforce_orthogonal, sigma_softmax_coupling,
            )
        else:
            return _vfe_gradients_block_diagonal(
                mu_q, sigma_q, mu_p, sigma_p, beta, phi, generators,
                alpha, lambda_belief, kappa, eps, irrep_dims, chunk_size,
                compute_sigma_align_grad, enforce_orthogonal, sigma_softmax_coupling,
            )

    # Chunked diagonal path
    if chunk_size is not None and is_diagonal:
        return _vfe_gradients_chunked(
            mu_q, sigma_q, mu_p, sigma_p, beta, phi, generators,
            alpha, lambda_belief, kappa, eps, chunk_size,
            compute_sigma_align_grad, enforce_orthogonal, sigma_softmax_coupling,
        )

    # Standard path (inline)
    return _vfe_gradients_standard(
        mu_q, sigma_q, mu_p, sigma_p, beta, phi, generators,
        alpha, lambda_belief, kappa, eps, cached_transport,
        compute_sigma_align_grad, sigma_softmax_coupling, enforce_orthogonal,
        is_diagonal, K, N,
    )


def _self_coupling_gradient(
    mu_q, sigma_q, mu_p, sigma_p, alpha, eps, is_diagonal,
):
    """Compute self-coupling gradients: ∂/∂θ [α · KL(q||p)]."""
    if is_diagonal:
        sigma_q_safe = sigma_q.clamp(min=eps)
        sigma_p_safe = sigma_p.clamp(min=eps)
        grad_mu = alpha * (mu_q - mu_p) / sigma_p_safe
        grad_sigma = alpha * 0.5 * (1.0 / sigma_p_safe - 1.0 / sigma_q_safe)
        return grad_mu, grad_sigma
    else:
        sigma_p_inv = safe_spd_inv(sigma_p, eps=eps)
        delta_mu = mu_q - mu_p
        grad_mu = alpha * torch.einsum('bnij,bnj->bni', sigma_p_inv, delta_mu)
        sigma_q_inv = safe_spd_inv(sigma_q, eps=eps)
        alpha_4d = alpha.unsqueeze(-1) if isinstance(alpha, torch.Tensor) else alpha
        grad_sigma = alpha_4d * 0.5 * (sigma_p_inv - sigma_q_inv)
        return grad_mu, grad_sigma


def _softmax_coupling(
    beta, grad_kl_per_pair, kl_values, lambda_belief, kappa, K, eps,
):
    """Compute direct + softmax coupling gradients for μ."""
    kappa_scaled = max(kappa * math.sqrt(max(K, 1)), eps)
    avg_grad = torch.einsum('bij,bijk->bik', beta, grad_kl_per_pair)
    grad_direct = lambda_belief * avg_grad

    grad_deviation = avg_grad.unsqueeze(2) - grad_kl_per_pair
    d_beta_d_mu = beta.unsqueeze(-1) * grad_deviation / kappa_scaled
    grad_softmax = lambda_belief * torch.einsum('bij,bijk->bik', kl_values, d_beta_d_mu)

    return grad_direct + grad_softmax, kappa_scaled


def _vfe_gradients_standard(
    mu_q, sigma_q, mu_p, sigma_p, beta, phi, generators,
    alpha, lambda_belief, kappa, eps, cached_transport,
    compute_sigma_align_grad, sigma_softmax_coupling, enforce_orthogonal,
    is_diagonal, K, N,
):
    """Standard (non-block-diagonal) VFE gradient computation."""
    dtype = mu_q.dtype

    if is_diagonal:
        mu_q, mu_p = mu_q.float(), mu_p.float()
        sigma_q, sigma_p = sigma_q.float(), sigma_p.float()
        beta = beta.float()

    grad_mu_self, grad_sigma_self = _self_coupling_gradient(
        mu_q, sigma_q, mu_p, sigma_p, alpha, eps, is_diagonal,
    )

    # Get transport operators
    if cached_transport is not None and 'Omega' in cached_transport:
        Omega = cached_transport['Omega']
    else:
        transport = compute_transport_operators(phi, generators, enforce_orthogonal)
        Omega = transport['Omega']

    # Transport means
    mu_j_transported = torch.einsum('bijkl,bjl->bijk', Omega, mu_q)
    delta_mu_ij = mu_q.unsqueeze(2) - mu_j_transported

    if is_diagonal:
        sigma_q_safe = sigma_q.clamp(min=eps)
        sigma_j_transported = torch.einsum(
            'bijkl,bijkl,bjl->bijk', Omega, Omega, sigma_q_safe
        ).clamp(min=eps)

        grad_kl_per_pair = delta_mu_ij / sigma_j_transported
        sigma_i_exp = sigma_q_safe[:, :, None, :].expand(-1, -1, N, -1)
        trace_term = (sigma_i_exp / sigma_j_transported).sum(dim=-1)
        mahal_term = (delta_mu_ij ** 2 / sigma_j_transported).sum(dim=-1)
        logdet_term = (torch.log(sigma_j_transported.clamp(min=eps)) - torch.log(sigma_i_exp.clamp(min=eps))).sum(dim=-1)

        kl_values = 0.5 * (trace_term + mahal_term - K + logdet_term)
        kl_values = kl_values.clamp(min=0.0, max=max(100.0, 5.0 * K))

        grad_mu_align, kappa_scaled = _softmax_coupling(
            beta, grad_kl_per_pair, kl_values, lambda_belief, kappa, K, eps,
        )

        if compute_sigma_align_grad:
            sigma_j_inv = 1.0 / sigma_j_transported
            sigma_i_inv = 1.0 / sigma_q_safe
            sigma_i_inv_exp = sigma_i_inv[:, :, None, :].expand(-1, -1, N, -1)
            grad_sigma_pair = 0.5 * (sigma_j_inv - sigma_i_inv_exp)
            grad_sigma_direct = lambda_belief * torch.einsum('bij,bijk->bik', beta, grad_sigma_pair)

            if sigma_softmax_coupling:
                avg_sg = torch.einsum('bij,bijk->bik', beta, grad_sigma_pair)
                sg_dev = avg_sg.unsqueeze(2) - grad_sigma_pair
                d_beta_ds = beta.unsqueeze(-1) * sg_dev / kappa_scaled
                grad_sigma_align = grad_sigma_direct + lambda_belief * torch.einsum('bij,bijk->bik', kl_values, d_beta_ds)
            else:
                grad_sigma_align = grad_sigma_direct
        else:
            grad_sigma_align = torch.zeros_like(sigma_q)
    else:
        # Full covariance alignment
        sigma_j_transported = torch.einsum(
            'bijkl,bjlm,bijmn->bijkn', Omega, sigma_q, Omega.transpose(-1, -2)
        )
        device = mu_q.device
        sigma_j_reg = sigma_j_transported + max(eps, 1e-4) * torch.eye(K, device=device, dtype=mu_q.dtype)
        if torch.isnan(sigma_j_reg).any():
            nan_mask = torch.isnan(sigma_j_reg).any(dim=-1).any(dim=-1)
            eye_K = torch.eye(K, device=device, dtype=mu_q.dtype)
            sigma_j_reg = torch.where(nan_mask.unsqueeze(-1).unsqueeze(-1), eye_K.expand_as(sigma_j_reg), sigma_j_reg)

        sigma_j_inv = safe_spd_inv(sigma_j_reg, eps=eps)
        grad_kl_per_pair = torch.einsum('bijkl,bijl->bijk', sigma_j_inv, delta_mu_ij)
        mahal_term = torch.einsum('bijk,bijk->bij', delta_mu_ij, grad_kl_per_pair)

        sigma_i_exp = sigma_q[:, :, None, :, :].expand(-1, -1, N, -1, -1).clone()
        trace_term = torch.einsum('bijkk->bij', torch.einsum('bijkl,bijlm->bijkm', sigma_j_inv, sigma_i_exp))

        try:
            L_j = torch.linalg.cholesky(sigma_j_reg)
            logdet_j = 2.0 * torch.sum(torch.log(torch.diagonal(L_j, dim1=-2, dim2=-1) + eps), dim=-1)
        except RuntimeError:
            eigvals = torch.linalg.eigvalsh(sigma_j_reg)
            logdet_j = torch.sum(torch.log(eigvals.clamp(min=eps)), dim=-1)

        sigma_i_reg = sigma_q + eps * torch.eye(K, device=device, dtype=mu_q.dtype)
        try:
            L_i = torch.linalg.cholesky(sigma_i_reg)
            logdet_i = 2.0 * torch.sum(torch.log(torch.diagonal(L_i, dim1=-2, dim2=-1) + eps), dim=-1)
        except RuntimeError:
            eigvals = torch.linalg.eigvalsh(sigma_i_reg)
            logdet_i = torch.sum(torch.log(eigvals.clamp(min=eps)), dim=-1)
        logdet_i_exp = logdet_i[:, :, None].expand(-1, -1, N).clone()

        kl_values = 0.5 * (trace_term + mahal_term - K + logdet_j - logdet_i_exp)
        kl_values = kl_values.clamp(min=0.0, max=max(100.0, 5.0 * K))

        grad_mu_align, kappa_scaled = _softmax_coupling(
            beta, grad_kl_per_pair, kl_values, lambda_belief, kappa, K, eps,
        )

        if compute_sigma_align_grad:
            sigma_q_inv = safe_spd_inv(sigma_q, eps=eps)
            sigma_i_inv_exp = sigma_q_inv[:, :, None, :, :].expand(-1, -1, N, -1, -1).clone()
            grad_sigma_pair = 0.5 * (sigma_j_inv - sigma_i_inv_exp)
            grad_sigma_direct = lambda_belief * torch.einsum('bij,bijkl->bikl', beta, grad_sigma_pair)

            if sigma_softmax_coupling:
                avg_sg = torch.einsum('bij,bijkl->bikl', beta, grad_sigma_pair)
                sg_dev = avg_sg.unsqueeze(2) - grad_sigma_pair
                d_beta_ds = beta.unsqueeze(-1).unsqueeze(-1) * sg_dev / kappa_scaled
                grad_sigma_align = grad_sigma_direct + lambda_belief * torch.einsum('bij,bijkl->bikl', kl_values, d_beta_ds)
            else:
                grad_sigma_align = grad_sigma_direct
        else:
            grad_sigma_align = torch.zeros_like(sigma_q)

    grad_mu = grad_mu_self + grad_mu_align
    grad_sigma = grad_sigma_self + grad_sigma_align
    return grad_mu.to(dtype), grad_sigma.to(dtype)


def _vfe_gradients_block_diagonal_diag(
    mu_q, sigma_q, mu_p, sigma_p, beta, phi, generators,
    alpha, lambda_belief, kappa, eps, irrep_dims,
    compute_sigma_align_grad, enforce_orthogonal, sigma_softmax_coupling,
):
    """Block-diagonal VFE gradients for diagonal covariance mode."""
    while sigma_q.dim() > 3 and sigma_q.shape[-1] == 1:
        sigma_q = sigma_q.squeeze(-1)
    while sigma_p.dim() > 3 and sigma_p.shape[-1] == 1:
        sigma_p = sigma_p.squeeze(-1)

    B, N, K = mu_q.shape
    device, dtype = mu_q.device, mu_q.dtype

    mu_q, mu_p = mu_q.float(), mu_p.float()
    sigma_q, sigma_p = sigma_q.float(), sigma_p.float()
    beta = beta.float()

    sigma_q_safe = sigma_q.clamp(min=eps)
    sigma_p_safe = sigma_p.clamp(min=eps)

    # Self-coupling
    grad_mu_self = alpha * (mu_q - mu_p) / sigma_p_safe
    grad_sigma_self = alpha * 0.5 * (1.0 / sigma_p_safe - 1.0 / sigma_q_safe)

    # Precompute block exponentials
    block_exps = _precompute_block_exponentials(phi, generators, irrep_dims, enforce_orthogonal, device, dtype)

    kl_values = torch.zeros(B, N, N, device=device, dtype=torch.float32)
    grad_kl_full = torch.zeros(B, N, N, K, device=device, dtype=torch.float32)
    grad_sigma_align = torch.zeros_like(sigma_q)
    grad_sigma_per_pair = torch.zeros(B, N, N, K, device=device, dtype=torch.float32) if (
        sigma_softmax_coupling and compute_sigma_align_grad) else None

    block_start = 0
    for block_idx, d in enumerate(irrep_dims):
        block_end = block_start + d
        mu_block = mu_q[:, :, block_start:block_end].contiguous()
        sigma_block = sigma_q_safe[:, :, block_start:block_end].contiguous()

        exp_phi_b, exp_neg_phi_b = block_exps[block_idx]
        Omega = torch.einsum('bikl,bjlm->bijkm', exp_phi_b, exp_neg_phi_b)

        mu_j_t = torch.einsum('bijkl,bjl->bijk', Omega, mu_block)
        sigma_j_t = torch.einsum('bijkl,bijkl,bjl->bijk', Omega, Omega, sigma_block).clamp(min=eps)
        del Omega

        mu_i = mu_block[:, :, None, :].expand(-1, -1, N, -1)
        delta_mu = mu_i - mu_j_t
        grad_kl_block = delta_mu / sigma_j_t
        grad_kl_full[:, :, :, block_start:block_end] = grad_kl_block

        sigma_i = sigma_block[:, :, None, :].expand(-1, -1, N, -1)
        trace = (sigma_i / sigma_j_t).sum(dim=-1)
        mahal = (delta_mu ** 2 / sigma_j_t).sum(dim=-1)
        logdet = (torch.log(sigma_j_t.clamp(min=eps)) - torch.log(sigma_i.clamp(min=eps))).sum(dim=-1)
        kl_block = 0.5 * (trace + mahal - d + logdet)
        kl_values = kl_values + kl_block.clamp(min=0.0, max=max(100.0, 5.0 * K))

        if compute_sigma_align_grad:
            sg_pair = 0.5 * (1.0 / sigma_j_t - 1.0 / sigma_i)
            grad_sigma_align[:, :, block_start:block_end] = (
                grad_sigma_align[:, :, block_start:block_end]
                + lambda_belief * torch.einsum('bij,bijk->bik', beta, sg_pair)
            )
            if grad_sigma_per_pair is not None:
                grad_sigma_per_pair[:, :, :, block_start:block_end] = sg_pair

        del sigma_j_t, mu_j_t
        block_start = block_end

    grad_mu_align, kappa_scaled = _softmax_coupling(
        beta, grad_kl_full, kl_values, lambda_belief, kappa, K, eps,
    )

    if grad_sigma_per_pair is not None:
        avg_sg = torch.einsum('bij,bijk->bik', beta, grad_sigma_per_pair)
        sg_dev = avg_sg.unsqueeze(2) - grad_sigma_per_pair
        d_beta_ds = beta.unsqueeze(-1) * sg_dev / kappa_scaled
        grad_sigma_align = grad_sigma_align + lambda_belief * torch.einsum('bij,bijk->bik', kl_values, d_beta_ds)

    grad_mu = grad_mu_self + grad_mu_align
    grad_sigma = grad_sigma_self + grad_sigma_align
    return grad_mu.to(dtype), grad_sigma.to(dtype)


def _vfe_gradients_block_diagonal(
    mu_q, sigma_q, mu_p, sigma_p, beta, phi, generators,
    alpha, lambda_belief, kappa, eps, irrep_dims, chunk_size,
    compute_sigma_align_grad, enforce_orthogonal, sigma_softmax_coupling,
):
    """Block-diagonal VFE gradients for full covariance mode."""
    B, N, K = mu_q.shape
    device, dtype = mu_q.device, mu_q.dtype
    C = chunk_size if chunk_size is not None else N

    grad_mu = torch.zeros(B, N, K, device=device, dtype=dtype)
    grad_sigma = torch.zeros(B, N, K, K, device=device, dtype=dtype)

    # Self-coupling
    sigma_p_inv = safe_spd_inv(sigma_p, eps=eps)
    grad_mu = grad_mu + alpha * torch.einsum('bnij,bnj->bni', sigma_p_inv, mu_q - mu_p)
    sigma_q_inv = safe_spd_inv(sigma_q, eps=eps)
    alpha_4d = alpha.unsqueeze(-1) if isinstance(alpha, torch.Tensor) else alpha
    grad_sigma = grad_sigma + alpha_4d * 0.5 * (sigma_p_inv - sigma_q_inv)

    block_exps = _precompute_block_exponentials(phi, generators, irrep_dims, enforce_orthogonal, device, dtype)

    grad_sigma_align = torch.zeros_like(sigma_q)
    kl_values = torch.zeros(B, N, N, device=device, dtype=dtype)
    grad_kl_full = torch.zeros(B, N, N, K, device=device, dtype=dtype)

    for i_start in range(0, N, C):
        i_end = min(i_start + C, N)
        block_start = 0
        for block_idx, d in enumerate(irrep_dims):
            block_end = block_start + d
            mu_block = mu_q[:, :, block_start:block_end].contiguous()
            sigma_block = sigma_q[:, :, block_start:block_end, block_start:block_end].contiguous()

            exp_phi_i = block_exps[block_idx][0][:, i_start:i_end].contiguous()
            exp_neg_phi_j = block_exps[block_idx][1].contiguous()
            Omega = torch.einsum('bikl,bjlm->bijkm', exp_phi_i, exp_neg_phi_j)

            mu_j_t = torch.einsum('bijkl,bjl->bijk', Omega, mu_block)
            sigma_j_t = torch.einsum('bijkl,bjlm,bijmn->bijkn', Omega, sigma_block, Omega.transpose(-1, -2))
            del Omega

            I_d = torch.eye(d, device=device, dtype=dtype)
            sigma_j_reg = sigma_j_t + eps * I_d
            sigma_j_inv = safe_spd_inv(sigma_j_reg, eps=eps)

            mu_i = mu_block[:, i_start:i_end].contiguous()
            delta_mu = mu_i[:, :, None, :] - mu_j_t
            grad_kl_block = torch.einsum('bijkl,bijl->bijk', sigma_j_inv, delta_mu)
            grad_kl_full[:, i_start:i_end, :, block_start:block_end] = grad_kl_block

            mahal = torch.einsum('bijk,bijk->bij', delta_mu, grad_kl_block)
            sigma_i_slice = sigma_block[:, i_start:i_end].contiguous()
            sigma_i_exp = sigma_i_slice[:, :, None, :, :].expand(-1, -1, N, -1, -1).clone()
            trace = torch.einsum('bijkk->bij', torch.einsum('bijkl,bijlm->bijkm', sigma_j_inv, sigma_i_exp))

            try:
                L_j = torch.linalg.cholesky(sigma_j_reg)
                logdet_j = 2.0 * torch.sum(torch.log(torch.diagonal(L_j, dim1=-2, dim2=-1) + eps), dim=-1)
            except RuntimeError:
                sign_j, logdet_j = torch.linalg.slogdet(sigma_j_reg)
                logdet_j = torch.where(sign_j > 0, logdet_j, torch.zeros_like(logdet_j))

            sigma_i_reg = sigma_i_slice + eps * I_d
            try:
                L_i = torch.linalg.cholesky(sigma_i_reg)
                logdet_i = 2.0 * torch.sum(torch.log(torch.diagonal(L_i, dim1=-2, dim2=-1) + eps), dim=-1)
            except RuntimeError:
                sign_i, logdet_i = torch.linalg.slogdet(sigma_i_reg)
                logdet_i = torch.where(sign_i > 0, logdet_i, torch.zeros_like(logdet_i))

            kl_block = 0.5 * (trace + mahal - d + logdet_j - logdet_i[:, :, None])
            kl_values[:, i_start:i_end, :] = kl_values[:, i_start:i_end, :] + kl_block.clamp(min=0.0, max=max(100.0, 5.0 * K))

            if compute_sigma_align_grad:
                sigma_i_inv = safe_spd_inv(sigma_i_reg, eps=1e-4)
                sigma_i_inv_exp = sigma_i_inv[:, :, None, :, :].expand(-1, -1, N, -1, -1).clone()
                sg_block = 0.5 * (sigma_j_inv - sigma_i_inv_exp)
                beta_chunk = beta[:, i_start:i_end, :].contiguous()
                grad_sigma_align[:, i_start:i_end, block_start:block_end, block_start:block_end] = (
                    grad_sigma_align[:, i_start:i_end, block_start:block_end, block_start:block_end]
                    + lambda_belief * torch.einsum('bij,bijkl->bikl', beta_chunk, sg_block)
                )

            del sigma_j_t, sigma_j_inv, mu_j_t
            block_start = block_end

    grad_mu_align, _ = _softmax_coupling(beta, grad_kl_full, kl_values, lambda_belief, kappa, K, eps)
    grad_mu = grad_mu + grad_mu_align
    grad_sigma = grad_sigma + grad_sigma_align
    return grad_mu, grad_sigma


def _vfe_gradients_chunked(
    mu_q, sigma_q, mu_p, sigma_p, beta, phi, generators,
    alpha, lambda_belief, kappa, eps, chunk_size,
    compute_sigma_align_grad, enforce_orthogonal, sigma_softmax_coupling,
):
    """Chunked VFE gradients for diagonal covariance mode."""
    B, N, K = mu_q.shape
    device, dtype = mu_q.device, mu_q.dtype

    mu_q, mu_p = mu_q.float(), mu_p.float()
    sigma_q, sigma_p = sigma_q.float(), sigma_p.float()
    beta = beta.float()

    sigma_q_safe = sigma_q.clamp(min=eps)
    sigma_p_safe = sigma_p.clamp(min=eps)

    grad_mu_self = alpha * (mu_q - mu_p) / sigma_p_safe
    grad_sigma_self = alpha * 0.5 * (1.0 / sigma_p_safe - 1.0 / sigma_q_safe)

    # Precompute per-token matrix exponentials
    phi_matrix = torch.einsum('bna,aij->bnij', phi, generators)
    exp_phi, exp_neg_phi = stable_matrix_exp_pair(phi_matrix)
    del phi_matrix

    if enforce_orthogonal and K >= 16:
        eye_K = torch.eye(K, device=device, dtype=torch.float32)
        exp_phi = exp_phi @ ((3.0 * eye_K - exp_phi.transpose(-1, -2) @ exp_phi) / 2.0)
        exp_neg_phi = exp_neg_phi @ ((3.0 * eye_K - exp_neg_phi.transpose(-1, -2) @ exp_neg_phi) / 2.0)

    kl_values = torch.zeros(B, N, N, device=device, dtype=torch.float32)
    grad_kl_full = torch.zeros(B, N, N, K, device=device, dtype=torch.float32)
    grad_sigma_align = torch.zeros_like(sigma_q)
    grad_sigma_per_pair = torch.zeros(B, N, N, K, device=device, dtype=torch.float32) if (
        sigma_softmax_coupling and compute_sigma_align_grad) else None

    for i_start in range(0, N, chunk_size):
        i_end = min(i_start + chunk_size, N)
        n_i = i_end - i_start
        exp_phi_i = exp_phi[:, i_start:i_end].contiguous()
        mu_i = mu_q[:, i_start:i_end].contiguous()
        sigma_i = sigma_q_safe[:, i_start:i_end].contiguous()

        for j_start in range(0, N, chunk_size):
            j_end = min(j_start + chunk_size, N)
            n_j = j_end - j_start
            exp_neg_phi_j = exp_neg_phi[:, j_start:j_end].contiguous()
            mu_j = mu_q[:, j_start:j_end].contiguous()
            sigma_j = sigma_q_safe[:, j_start:j_end].contiguous()

            Omega = torch.einsum('bikl,bjlm->bijkm', exp_phi_i, exp_neg_phi_j)
            mu_j_t = torch.einsum('bijkl,bjl->bijk', Omega, mu_j)
            sigma_j_t = torch.einsum('bijkl,bijkl,bjl->bijk', Omega, Omega, sigma_j[:, None, :, :].expand(-1, n_i, -1, -1)).clamp(min=eps)
            del Omega

            mu_i_exp = mu_i[:, :, None, :].expand(-1, -1, n_j, -1)
            delta_mu = mu_i_exp - mu_j_t
            grad_kl_chunk = delta_mu / sigma_j_t
            grad_kl_full[:, i_start:i_end, j_start:j_end, :] = grad_kl_chunk

            sigma_i_exp = sigma_i[:, :, None, :].expand(-1, -1, n_j, -1)
            trace = (sigma_i_exp / sigma_j_t).sum(dim=-1)
            mahal = (delta_mu ** 2 / sigma_j_t).sum(dim=-1)
            logdet = (torch.log(sigma_j_t.clamp(min=eps)) - torch.log(sigma_i_exp.clamp(min=eps))).sum(dim=-1)

            kl_chunk = 0.5 * (trace + mahal - K + logdet)
            kl_values[:, i_start:i_end, j_start:j_end] = kl_chunk.clamp(min=0.0, max=max(100.0, 5.0 * K))

            if compute_sigma_align_grad:
                sg_pair = 0.5 * (1.0 / sigma_j_t - 1.0 / sigma_i_exp)
                beta_chunk = beta[:, i_start:i_end, j_start:j_end].contiguous()
                grad_sigma_align[:, i_start:i_end, :] = (
                    grad_sigma_align[:, i_start:i_end, :]
                    + lambda_belief * torch.einsum('bij,bijk->bik', beta_chunk, sg_pair)
                )
                if grad_sigma_per_pair is not None:
                    grad_sigma_per_pair[:, i_start:i_end, j_start:j_end, :] = sg_pair

            del sigma_j_t, mu_j_t

    grad_mu_align, kappa_scaled = _softmax_coupling(
        beta, grad_kl_full, kl_values, lambda_belief, kappa, K, eps,
    )

    if grad_sigma_per_pair is not None:
        avg_sg = torch.einsum('bij,bijk->bik', beta, grad_sigma_per_pair)
        sg_dev = avg_sg.unsqueeze(2) - grad_sigma_per_pair
        d_beta_ds = beta.unsqueeze(-1) * sg_dev / kappa_scaled
        grad_sigma_align = grad_sigma_align + lambda_belief * torch.einsum('bij,bijk->bik', kl_values, d_beta_ds)

    grad_mu = grad_mu_self + grad_mu_align
    grad_sigma = grad_sigma_self + grad_sigma_align
    return grad_mu.to(dtype), grad_sigma.to(dtype)


def _precompute_block_exponentials(phi, generators, irrep_dims, enforce_orthogonal, device, dtype):
    """Precompute per-block matrix exponentials."""
    block_exps = []
    block_start = 0
    for d in irrep_dims:
        block_end = block_start + d
        gen_block = generators[:, block_start:block_end, block_start:block_end]
        phi_mat = torch.einsum('bna,aij->bnij', phi, gen_block)
        exp_phi, exp_neg_phi = stable_matrix_exp_pair(phi_mat)
        if enforce_orthogonal and d >= 16:
            eye_d = torch.eye(d, device=device, dtype=dtype)
            exp_phi = exp_phi @ ((3.0 * eye_d - exp_phi.transpose(-1, -2) @ exp_phi) / 2.0)
            exp_neg_phi = exp_neg_phi @ ((3.0 * eye_d - exp_neg_phi.transpose(-1, -2) @ exp_neg_phi) / 2.0)
        block_exps.append((exp_phi, exp_neg_phi))
        block_start = block_end
    return block_exps


# =============================================================================
# Natural Gradient Projection
# =============================================================================

def compute_natural_gradient(
    grad_mu: torch.Tensor,
    grad_sigma: torch.Tensor,
    sigma_q: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project Euclidean gradients to natural gradients via Fisher metric.

    F_μ = Σ^{-1}  →  nat_grad_μ = Σ @ grad_μ
    F_σ = 2Σ^{-2} →  nat_grad_σ = 0.5 * Σ² @ grad_σ  (diagonal approx)
    """
    while sigma_q.dim() > 3 and sigma_q.shape[-1] == 1:
        sigma_q = sigma_q.squeeze(-1)

    is_diagonal = sigma_q.dim() == 3
    orig_dtype = sigma_q.dtype

    sigma_q = sigma_q.float()
    grad_mu = grad_mu.float()
    grad_sigma = grad_sigma.float()

    if is_diagonal:
        sigma_safe = sigma_q.clamp(min=eps)
        nat_grad_mu = sigma_safe * grad_mu
        nat_grad_sigma = 0.5 * sigma_safe * sigma_safe * grad_sigma
    else:
        nat_grad_mu = torch.einsum('bnij,bnj->bni', sigma_q, grad_mu)
        nat_grad_sigma = 0.5 * torch.einsum('bnij,bnjk,bnkl->bnil', sigma_q, grad_sigma, sigma_q)

    return nat_grad_mu.to(orig_dtype), nat_grad_sigma.to(orig_dtype)


# =============================================================================
# VariationalFFNDynamic
# =============================================================================

class VariationalFFNDynamic(nn.Module):
    """Dynamic-β VFE belief evolution. Config-driven."""

    def __init__(self, config: GaugeTransformerConfig, generators: torch.Tensor,
                 prior_bank: Optional[nn.Module] = None):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        self.register_buffer('generators', generators)
        self.n_iterations = config.n_vfe_iterations
        self.mask_self_attention = config.mask_self_attention
        self.update_sigma = config.update_sigma
        self.diagonal_covariance = config.diagonal_covariance
        self.compute_sigma_align_grad = config.compute_sigma_align_grad
        self.sigma_softmax_coupling = config.sigma_softmax_coupling

        # Phi evolution
        self.update_phi = config.evolve_phi
        self.update_phi_per_iteration = config.evolve_phi_e_step
        self.phi_lr = config.phi_lr
        self.phi_max_norm = config.phi_max_norm

        # Phi preconditioning
        self.phi_natural_gradient = config.phi_natural_gradient
        self.register_buffer('_phi_preconditioner', None)
        self.register_buffer('_structure_constants', None)
        self.register_buffer('_gram', None)
        if config.phi_natural_gradient in ('cartan', 'killing', 'pullback'):
            from transformer_v2.gauge_preconditioner import (
                build_cartan_projector, build_killing_form_preconditioner,
                build_structure_constants,
            )
            if config.phi_natural_gradient == 'cartan':
                self._phi_preconditioner = build_cartan_projector(generators)
            elif config.phi_natural_gradient == 'killing':
                self._phi_preconditioner = build_killing_form_preconditioner(generators)
            elif config.phi_natural_gradient == 'pullback':
                self._structure_constants = build_structure_constants(generators)
                self._gram = torch.einsum('aij,bij->ab', generators.transpose(-2, -1), generators)

        # Memory efficiency
        self.irrep_dims = config.irrep_dims
        self.chunk_size = config.chunk_size

        # VFE hyperparameters
        self.alpha = config.alpha_ffn
        self.lambda_belief = config.lambda_ffn
        self.kappa = config.kappa_ffn

        # Multi-head VFE
        self.multihead_vfe = config.multihead_vfe and config.irrep_dims is not None
        self.per_head_kappa_enabled = config.per_head_kappa
        if self.multihead_vfe and config.per_head_kappa and config.irrep_dims is not None:
            n_heads = len(config.irrep_dims)
            self.log_kappa_heads = nn.Parameter(
                torch.full((n_heads,), math.log(max(config.kappa_ffn, 1e-6)))
            )
        else:
            self.log_kappa_heads = None

        # Bayesian precision
        self.learnable_alpha = config.learnable_alpha
        if config.learnable_alpha:
            a0_init = 1.0
            alpha_init = max(config.alpha_ffn, 0.01)
            b0_init = (a0_init + config.embed_dim / 2.0) / alpha_init
            self.raw_a0 = nn.Parameter(torch.tensor(float(np.log(np.expm1(a0_init)))))
            self.raw_b0 = nn.Parameter(torch.tensor(float(np.log(np.expm1(b0_init)))))

        # Pure FEP mode
        self.pure_fep_mode = config.pure_fep_mode
        self.max_seq_len = config.max_seq_len
        self.prior_lr = config.prior_lr
        self.use_prior_bank = config.use_prior_bank
        self.prior_bank = prior_bank

        if config.pure_fep_mode:
            if config.use_prior_bank:
                if prior_bank is None:
                    raise ValueError("use_prior_bank=True requires prior_bank")
            else:
                self.register_buffer('prior_mu', torch.zeros(config.max_seq_len, config.embed_dim))
                self.register_buffer('prior_sigma', torch.ones(config.max_seq_len, config.embed_dim))
                self.register_buffer('prior_update_count', torch.zeros(config.max_seq_len))
                self.register_buffer('prior_initialized', torch.tensor(False))

        # Learnable step size
        if config.learnable_lr:
            self.lr = nn.Parameter(torch.tensor(0.1))
        else:
            self.register_buffer('lr', torch.tensor(0.1))

    def get_bayesian_alpha(self, mu_q, mu_p, sigma_p, eps=1e-6):
        """Compute Bayesian precision via Gamma-Normal conjugacy."""
        a0 = F.softplus(self.raw_a0)
        b0 = F.softplus(self.raw_b0)
        delta_mu = mu_q - mu_p
        is_diag = sigma_p.dim() == 3
        if is_diag:
            mahal_sq = (delta_mu ** 2 / sigma_p.clamp(min=eps)).sum(dim=-1, keepdim=True)
        else:
            sigma_p_inv = safe_spd_inv(sigma_p, eps=eps)
            mahal_sq = torch.einsum('bni,bnij,bnj->bn', delta_mu, sigma_p_inv, delta_mu).unsqueeze(-1)
        K = mu_q.shape[-1]
        return (a0 + K / 2.0) / (b0 + 0.5 * mahal_sq)

    def _precondition_phi_grad(self, grad_phi, phi):
        """Apply phi gradient preconditioning."""
        if self.phi_natural_gradient == 'cartan':
            from transformer_v2.gauge_preconditioner import apply_cartan_preconditioning
            return apply_cartan_preconditioning(grad_phi, self._phi_preconditioner)
        elif self.phi_natural_gradient == 'killing':
            from transformer_v2.gauge_preconditioner import apply_killing_form_natural_gradient
            return apply_killing_form_natural_gradient(grad_phi, self._phi_preconditioner)
        elif self.phi_natural_gradient == 'pullback':
            from transformer_v2.gauge_preconditioner import apply_pullback_natural_gradient
            return apply_pullback_natural_gradient(
                grad_phi, phi, self.generators, self._structure_constants, self._gram,
            )
        else:  # 'clip'
            norm = torch.norm(grad_phi, dim=-1, keepdim=True)
            return torch.where(norm > 10.0, grad_phi * 10.0 / (norm + 1e-6), grad_phi)

    def _update_phi_via_alignment(self, mu_current, sigma_current, phi_current,
                                   mask, is_diagonal, phi_lr_scale=1.0):
        """Unified phi update via alignment loss gradient. Eliminates duplication."""
        if not torch.is_grad_enabled():
            return phi_current

        phi_for_grad = phi_current.clone().requires_grad_(True)

        if self.multihead_vfe:
            alignment_loss = torch.tensor(0.0, device=mu_current.device, dtype=mu_current.dtype)
            block_start = 0
            for h, d_h in enumerate(self.irrep_dims):
                block_end = block_start + d_h
                mu_h = mu_current[:, :, block_start:block_end].detach()
                if is_diagonal:
                    sigma_h = sigma_current[:, :, block_start:block_end].detach()
                else:
                    sigma_h = sigma_current[:, :, block_start:block_end, block_start:block_end].detach()
                gen_h = self.generators[:, block_start:block_end, block_start:block_end]
                kappa_h = torch.exp(self.log_kappa_heads[h]).item() if self.log_kappa_heads is not None else self.kappa

                beta_h, kl_h = compute_attention_weights(
                    mu_h, sigma_h, phi_for_grad, gen_h, kappa_h, 1e-6, mask,
                    return_kl=True, diagonal_covariance=is_diagonal,
                    chunk_size=self.chunk_size, mask_self_attention=self.mask_self_attention,
                )
                alignment_loss = alignment_loss + self.lambda_belief * (beta_h * kl_h).sum()
                block_start = block_end
        else:
            beta_phi, kl_matrix = compute_attention_weights(
                mu_current.detach(),
                sigma_current.detach() if sigma_current is not None else None,
                phi_for_grad, self.generators, self.kappa, 1e-6, mask,
                return_kl=True, diagonal_covariance=is_diagonal,
                irrep_dims=self.irrep_dims, chunk_size=self.chunk_size,
                mask_self_attention=self.mask_self_attention,
            )
            alignment_loss = self.lambda_belief * (beta_phi * kl_matrix).sum()

        if alignment_loss.grad_fn is None:
            return phi_current

        grad_phi = torch.autograd.grad(alignment_loss, phi_for_grad, create_graph=False, retain_graph=False)[0]
        grad_phi = self._precondition_phi_grad(grad_phi, phi_current)

        return retract_phi(
            phi=phi_current, delta_phi=-grad_phi, generators=self.generators,
            step_size=self.phi_lr * phi_lr_scale, max_norm=self.phi_max_norm,
        )

    def forward(
        self,
        mu: torch.Tensor,
        beta: torch.Tensor,
        mu_prior: torch.Tensor,
        phi: torch.Tensor,
        sigma: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        W_out: Optional[torch.Tensor] = None,
        token_ids: Optional[torch.Tensor] = None,
        return_beta_history: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[list]]:
        """Dynamic VFE descent with β recomputation at each step."""
        B, N, K = mu.shape
        device, dtype = mu.device, mu.dtype
        eps = 1e-6

        # Initialize sigma if needed
        if sigma is None:
            if self.diagonal_covariance:
                sigma = torch.ones(B, N, K, device=device, dtype=dtype) * 0.1
            else:
                sigma = 0.1 * torch.eye(K, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1).clone()

        while sigma.dim() > 3 and sigma.shape[-1] == 1:
            sigma = sigma.squeeze(-1)
        is_diagonal = sigma.dim() == 3

        # Select priors
        mu_p_current, sigma_p = self._select_priors(mu_prior, sigma, is_diagonal, token_ids, N, B, device)

        mu_current = mu.clone()
        sigma_current = sigma.clone()
        phi_current = phi.clone()
        beta_history = [] if return_beta_history else None
        has_observations = targets is not None and W_out is not None
        base_lr = self.lr / self.n_iterations

        for iteration in range(self.n_iterations):
            decay_factor = 1.0 - 0.5 * (iteration / max(self.n_iterations - 1, 1)) if self.n_iterations > 1 else 1.0
            effective_lr = base_lr * decay_factor

            # Cache transport (skip for block-diagonal/chunked/multihead paths)
            if self.irrep_dims is None and self.chunk_size is None and not self.multihead_vfe:
                cached_transport = compute_transport_operators(
                    phi_current, self.generators, self.config.enforce_orthogonal,
                )
            else:
                cached_transport = None

            # Bayesian precision
            alpha_eff = self.get_bayesian_alpha(mu_current, mu_p_current, sigma_p, eps) if self.learnable_alpha else self.alpha

            if self.multihead_vfe:
                grad_mu, grad_sigma, beta_current = self._multihead_step(
                    mu_current, sigma_current, mu_p_current, sigma_p,
                    phi_current, mask, alpha_eff, is_diagonal, eps,
                )
                if return_beta_history:
                    beta_history.append(beta_current.detach().clone())
            else:
                beta_current = compute_attention_weights(
                    mu_current, sigma_current, phi_current, self.generators,
                    self.kappa, eps, mask, diagonal_covariance=is_diagonal,
                    cached_transport=cached_transport, irrep_dims=self.irrep_dims,
                    chunk_size=self.chunk_size, mask_self_attention=self.mask_self_attention,
                )
                if return_beta_history:
                    beta_history.append(beta_current.detach().clone())

                grad_mu, grad_sigma = compute_vfe_gradients(
                    mu_current, sigma_current, mu_p_current, sigma_p,
                    beta_current, phi_current, self.generators,
                    alpha_eff, self.lambda_belief, self.kappa, eps,
                    cached_transport, self.compute_sigma_align_grad,
                    self.sigma_softmax_coupling, self.irrep_dims, self.chunk_size,
                    self.config.enforce_orthogonal,
                )

            # Observation gradient
            if has_observations:
                logits = torch.matmul(mu_current.detach(), W_out.T)
                probs = F.softmax(logits, dim=-1)
                targets_valid = targets.clone()
                targets_valid[targets == -1] = 0
                one_hot = F.one_hot(targets_valid, num_classes=W_out.shape[0]).float()
                mask_obs = (targets != -1).unsqueeze(-1).float()
                grad_mu = grad_mu + torch.matmul((probs - one_hot) * mask_obs, W_out)

            grad_mu = torch.clamp(grad_mu, min=-1e3, max=1e3)
            grad_sigma = torch.clamp(grad_sigma, min=-1e3, max=1e3)

            # Natural gradient projection
            nat_grad_mu, nat_grad_sigma = compute_natural_gradient(grad_mu, grad_sigma, sigma_current, eps)

            # Update μ with whitened trust region
            delta_mu = -effective_lr * nat_grad_mu
            if is_diagonal:
                sigma_sqrt = torch.sqrt(sigma_current.float().clamp(min=eps)).to(dtype)
                whitened = delta_mu / sigma_sqrt
            else:
                sigma_diag = torch.diagonal(sigma_current, dim1=-2, dim2=-1).clone().float().clamp(min=eps)
                whitened = delta_mu / torch.sqrt(sigma_diag).to(dtype)

            w_norm = torch.linalg.norm(whitened, dim=-1, keepdim=True)
            scale = torch.clamp(2.0 / (w_norm + eps), max=1.0)
            mu_current = mu_current + scale * delta_mu

            # Update σ
            if self.update_sigma:
                sigma_lr = effective_lr * 0.05
                if is_diagonal:
                    sigma_current = retract_spd_diagonal(sigma_current, -nat_grad_sigma, sigma_lr, 0.2, eps)
                else:
                    sigma_current = retract_spd(sigma_current, -nat_grad_sigma, sigma_lr, 0.1, eps)

            # Phi update per iteration
            if self.update_phi_per_iteration:
                phi_current = self._update_phi_via_alignment(
                    mu_current, sigma_current, phi_current, mask, is_diagonal,
                    phi_lr_scale=1.0 / self.n_iterations,
                )

        # Post-loop phi update
        if self.update_phi and not self.update_phi_per_iteration:
            phi_current = self._update_phi_via_alignment(
                mu_current, sigma_current, phi_current, mask, is_diagonal,
            )

        if self.update_sigma:
            return mu_current, sigma_current, phi_current, beta_history
        return mu_current, None, phi_current, beta_history

    def _select_priors(self, mu_prior, sigma, is_diagonal, token_ids, N, B, device):
        """Select priors: PriorBank, persistent, or embedding."""
        if self.pure_fep_mode and self.use_prior_bank:
            if token_ids is None:
                raise ValueError("token_ids required when use_prior_bank=True")
            mu_p, sigma_p_bank, _ = self.prior_bank.encode(token_ids)
            sigma_p = sigma_p_bank if is_diagonal else torch.diag_embed(sigma_p_bank)
            return mu_p, sigma_p

        if self.pure_fep_mode and not self.use_prior_bank:
            self.initialize_priors_from_embeddings(mu_prior)
            mu_p, sigma_p_persist = self.get_persistent_priors(N, B, device)
            if sigma_p_persist is not None:
                sigma_p = sigma_p_persist.clone() if is_diagonal else torch.diag_embed(sigma_p_persist)
            else:
                sigma_p = sigma.clone()
            return mu_p.clone(), sigma_p

        return mu_prior.clone(), sigma.clone()

    def _multihead_step(self, mu_current, sigma_current, mu_p, sigma_p,
                         phi_current, mask, alpha_eff, is_diagonal, eps):
        """Execute one multihead VFE step."""
        B, N, K = mu_current.shape
        grad_mu = torch.zeros_like(mu_current)
        grad_sigma = torch.zeros_like(sigma_current)
        beta_heads = []

        block_start = 0
        for h, d_h in enumerate(self.irrep_dims):
            block_end = block_start + d_h
            mu_h = mu_current[:, :, block_start:block_end].detach().contiguous()
            mu_p_h = mu_p[:, :, block_start:block_end].contiguous()
            if is_diagonal:
                sigma_h = sigma_current[:, :, block_start:block_end].detach().contiguous()
                sigma_p_h = sigma_p[:, :, block_start:block_end].contiguous()
            else:
                sigma_h = sigma_current[:, :, block_start:block_end, block_start:block_end].detach().contiguous()
                sigma_p_h = sigma_p[:, :, block_start:block_end, block_start:block_end].contiguous()
            gen_h = self.generators[:, block_start:block_end, block_start:block_end]
            kappa_h = torch.exp(self.log_kappa_heads[h]).item() if self.log_kappa_heads is not None else self.kappa

            beta_h = compute_attention_weights(
                mu_h, sigma_h, phi_current, gen_h, kappa_h, eps, mask,
                diagonal_covariance=is_diagonal, chunk_size=self.chunk_size,
                mask_self_attention=self.mask_self_attention,
            )
            beta_heads.append(beta_h)

            grad_mu_h, grad_sigma_h = compute_vfe_gradients(
                mu_h, sigma_h, mu_p_h, sigma_p_h, beta_h, phi_current, gen_h,
                alpha_eff, self.lambda_belief, kappa_h, eps,
                compute_sigma_align_grad=self.compute_sigma_align_grad,
                sigma_softmax_coupling=self.sigma_softmax_coupling,
            )

            grad_mu[:, :, block_start:block_end] = grad_mu_h
            if is_diagonal:
                grad_sigma[:, :, block_start:block_end] = grad_sigma_h
            else:
                grad_sigma[:, :, block_start:block_end, block_start:block_end] = grad_sigma_h
            block_start = block_end

        beta_stacked = torch.stack(beta_heads, dim=1)
        return grad_mu, grad_sigma, beta_stacked

    # ── Pure FEP methods ─────────────────────────────────────────────

    def initialize_priors_from_embeddings(self, mu_embed):
        if not self.pure_fep_mode or self.use_prior_bank:
            return
        if self.prior_initialized:
            return
        B, N, K = mu_embed.shape
        N_update = min(N, self.max_seq_len)
        with torch.no_grad():
            self.prior_mu[:N_update] = mu_embed[:, :N_update].mean(dim=0)
            self.prior_initialized.fill_(True)

    def get_persistent_priors(self, seq_len, batch_size, device):
        if not self.pure_fep_mode or self.use_prior_bank:
            return None, None
        N = min(seq_len, self.max_seq_len)
        mu_p = self.prior_mu[:N].unsqueeze(0).expand(batch_size, -1, -1).clone()
        sigma_p = self.prior_sigma[:N].unsqueeze(0).expand(batch_size, -1, -1).clone()
        if seq_len > self.max_seq_len:
            pad_len = seq_len - self.max_seq_len
            mu_p = torch.cat([mu_p, torch.zeros(batch_size, pad_len, self.embed_dim, device=device)], dim=1)
            sigma_p = torch.cat([sigma_p, torch.ones(batch_size, pad_len, self.embed_dim, device=device)], dim=1)
        return mu_p, sigma_p

    def update_priors_from_beliefs(self, mu_beliefs, sigma_beliefs, prediction_errors, lr=None):
        if not self.pure_fep_mode or self.use_prior_bank:
            return
        lr = lr if lr is not None else self.prior_lr
        B, N, K = mu_beliefs.shape
        N_update = min(N, self.max_seq_len)
        eps = 1e-6
        with torch.no_grad():
            errors = prediction_errors[:, :N_update].clamp(min=eps, max=20.0)
            weights = F.softmax(-errors, dim=0)
            weighted_mu = (mu_beliefs[:, :N_update] * weights.unsqueeze(-1)).sum(dim=0)
            weighted_sigma = (sigma_beliefs[:, :N_update] * weights.unsqueeze(-1)).sum(dim=0)
            confidence = 1.0 / (1.0 + errors.mean(dim=0))
            effective_lr = lr * confidence.unsqueeze(-1)
            self.prior_mu[:N_update] = (1.0 - effective_lr) * self.prior_mu[:N_update] + effective_lr * weighted_mu
            sigma_lr = effective_lr * 0.1
            self.prior_sigma[:N_update] = ((1.0 - sigma_lr) * self.prior_sigma[:N_update] + sigma_lr * weighted_sigma).clamp(min=eps)
            self.prior_update_count[:N_update] += 1

    def get_prior_stats(self) -> Dict[str, float]:
        if not self.pure_fep_mode or self.use_prior_bank:
            return {}
        with torch.no_grad():
            active = self.prior_update_count > 0
            n = active.sum().item()
            if n == 0:
                return {'prior_active_positions': 0}
            return {
                'prior_active_positions': n,
                'prior_mu_mean': self.prior_mu[active].mean().item(),
                'prior_mu_std': self.prior_mu[active].std().item(),
                'prior_sigma_mean': self.prior_sigma[active].mean().item(),
                'prior_update_count_mean': self.prior_update_count[active].mean().item(),
            }

    def extra_repr(self):
        base = (f"embed_dim={self.embed_dim}, n_iterations={self.n_iterations}, "
                f"alpha={self.alpha}, lambda_belief={self.lambda_belief}, kappa={self.kappa}")
        if self.pure_fep_mode:
            base += f", pure_fep_mode=True, prior_lr={self.prior_lr}"
        return base
