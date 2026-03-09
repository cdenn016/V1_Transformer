"""
Training Loop for Gauge-Theoretic Transformer
==============================================

Implements COMPLETE free energy minimization with all gauge-theoretic terms.

Standard FEP hierarchy:
    h  (hyper-prior)  →  constrains s
    s  (model)        →  generates p
    p  (prior)        →  constrains q
    q  (beliefs)      →  inferred from observations

Full Free Energy:
    F = (1) α · Σ_i KL(q_i || p_i)                      [Self-coupling: beliefs to priors]
      + (2) λ_h · Σ_i KL(s_i || h)                      [Hyper-prior: models to centroid]
      + (3) λ_β · Σ_{i,j} β_{ij} · KL(q_i || Ω_{ij} q_j) [Belief coupling / attention]
      + (4) λ_γ · Σ_{i,j} γ_{ij} · KL(s_i || Ω_{ij} s_j) [Model coupling / meta-cognition]
      - (5) E[log p(o|x)]                               [Observation likelihood]

where:
    - q_i = N(μ_q, Σ_q): Beliefs (fast/E-step, evolved through attention)
    - p_i = f(s_i): Priors (derived from models, constrain beliefs)
    - s_i = N(μ_s, Σ_s): Models (slow/M-step, embedding params = what backprop updates)
    - h = centroid of {s_i}: Hyper-prior (prevents model memorization)
    - β_{ij} = softmax_j(-KL(q_i||Ω_{ij}q_j)/κ_β): Belief coupling weights
    - γ_{ij} = softmax_j(-KL(s_i||Ω_{ij}s_j)/κ_γ): Model coupling weights
    - Ω_{ij}: Parallel transport operator

Note: Currently p_i = s_i (identity mapping). The model IS the prior in the
simplest case. A future generalization could make p = f(s) non-trivial
(e.g., incorporating position-dependent context).

Author: Implementation from validated suite + complete gamma term
Date: November 2025
"""

# Suppress Triton and CuPy warnings BEFORE torch import (torch may trigger imports)
import warnings
warnings.filterwarnings("ignore", message="Failed to find cuobjdump", module="triton")
warnings.filterwarnings("ignore", message="Failed to find nvdisasm", module="triton")
warnings.filterwarnings("ignore", message="CUDA path could not be detected", module="cupy")

from math_utils.numerical_monitor import record as _nr, flush as _nflush

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import time
import json
import numpy as np
from transformer.analysis.rg_metrics import (
    compute_rg_diagnostics,
    RGDiagnostics,
    RGFlowSummary,
)
from transformer.analysis.rg_flow_enhanced import compute_full_rg_diagnostics

# Import attention computation for gamma term
from transformer.core.attention import compute_attention_weights



def compute_rg_metrics_from_attention(
    attn_info: Dict,
    step: int,
    auto_cluster: bool = True,
    n_clusters: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute RG metrics from attention info returned by forward_with_attention.

    This analyzes the emergent renormalization group structure in the
    attention-belief dynamics, detecting meta-agent emergence.

    Args:
        attn_info: Dict with 'beta', 'mu', 'sigma' from forward_with_attention
        step: Current training step
        auto_cluster: Auto-detect clusters via spectral clustering
        n_clusters: Fixed number of clusters (None = auto)

    Returns:
        Dict with RG metrics for logging:
            - rg/modularity: Block structure in attention (higher = more meta-agents)
            - rg/effective_rank: Effective dimensionality (lower = concentrated)
            - rg/n_clusters: Number of detected meta-agents
            - rg/kl_within_mean: KL divergence within clusters (lower = tighter)
            - rg/kl_between_mean: KL divergence between clusters (stable = distinct)
            - rg/beta_entropy: Attention distribution entropy
    """
    beta = attn_info.get('beta')  # (n_layers, B, n_heads, N, N)
    mu = attn_info.get('mu')      # (B, N, K)
    sigma = attn_info.get('sigma')  # (B, N, K) or (B, N, K, K)

    if beta is None or mu is None:
        return {}

    # Use final layer's attention for RG metrics, average over heads
    if beta.dim() == 5:
        beta_avg = beta[-1].mean(dim=1)  # (B, N, N) — last layer, head-averaged
    elif beta.dim() == 4:
        beta_avg = beta.mean(dim=1)  # (B, N, N) — legacy 4D format
    else:
        beta_avg = beta

    # Handle sigma - default to ones if None
    if sigma is None:
        sigma = torch.ones_like(mu)

    # Compute RG diagnostics
    try:
        diagnostics = compute_rg_diagnostics(
            mu=mu,
            sigma=sigma,
            beta=beta_avg,
            step=step,
            auto_cluster=auto_cluster,
            n_clusters=n_clusters,
        )

        # Convert to metrics dict
        rg_metrics = {
            'rg/modularity': diagnostics.modularity,
            'rg/effective_rank': diagnostics.effective_rank,
            'rg/n_clusters': diagnostics.n_clusters,
            'rg/kl_within_mean': diagnostics.kl_within_mean,
            'rg/kl_within_std': diagnostics.kl_within_std,
            'rg/kl_between_mean': diagnostics.kl_between_mean,
            'rg/kl_between_std': diagnostics.kl_between_std,
            'rg/beta_entropy': diagnostics.beta_entropy,
        }

        # Add meta-agent sizes if available
        if diagnostics.meta_agent_sizes:
            rg_metrics['rg/meta_agent_sizes'] = diagnostics.meta_agent_sizes

        # Enhanced gauge-frame metrics if phi is available
        phi = attn_info.get('phi')  # (B, N, gauge_dim)
        kl_matrix = attn_info.get('kl_matrix')  # (B, N, N) or (B, H, N, N)
        if phi is not None:
            try:
                enhanced = compute_full_rg_diagnostics(
                    mu=mu, sigma=sigma, phi=phi, beta=beta_avg,
                    kl_matrix=kl_matrix, step=step,
                    auto_cluster=auto_cluster,
                )
                rg_metrics['rg/gauge_coherence'] = enhanced.gauge_coherence
                rg_metrics['rg/phi_within_mean'] = enhanced.phi_within_mean
                rg_metrics['rg/phi_between_mean'] = enhanced.phi_between_mean
                rg_metrics['rg/kl_matrix_rank'] = enhanced.kl_matrix_rank
                if kl_matrix is not None:
                    rg_metrics['rg/fe_belief_align'] = enhanced.fe_belief_align
            except Exception as e:
                print(f"[WARNING] Enhanced RG metrics failed: {e}")

        return rg_metrics

    except Exception as e:
        # Return empty metrics on error (don't crash training)
        print(f"[WARNING] RG metrics computation failed: {e}")
        return {}


def compute_dynamic_rg_metrics(
    rg_info: Dict,
    step: int,
) -> Dict[str, Any]:
    """
    Compute RG flow metrics from beta_history (dynamic RG within forward pass).

    This tracks how attention structure evolves across VFE iterations,
    revealing dynamic cluster formation.

    Args:
        rg_info: Dict from forward_with_rg_tracking() containing 'beta_history'
        step: Current training step

    Returns:
        Dict with dynamic RG metrics:
            - rg/dynamic/n_iterations: Number of VFE steps
            - rg/dynamic/modularity_init: Modularity at first VFE step
            - rg/dynamic/modularity_final: Modularity at last VFE step
            - rg/dynamic/modularity_change: Final - Init (positive = emergence)
            - rg/dynamic/rank_init: Effective rank at first step
            - rg/dynamic/rank_final: Effective rank at last step
            - rg/dynamic/rank_change: Final - Init (negative = compression)
    """
    beta_history = rg_info.get('beta_history')

    if beta_history is None or len(beta_history) == 0:
        return {'rg/dynamic/n_iterations': 0}

    n_iterations = len(beta_history)

    # Import RG metrics
    from transformer.analysis.rg_metrics import compute_modularity, compute_effective_rank

    # Compute metrics at first and last step
    beta_init = beta_history[0]
    beta_final = beta_history[-1]

    if beta_init.dim() == 4:
        beta_init = beta_init.mean(dim=1)
        beta_final = beta_final.mean(dim=1)

    mod_init = compute_modularity(beta_init)
    mod_final = compute_modularity(beta_final)
    rank_init = compute_effective_rank(beta_init)
    rank_final = compute_effective_rank(beta_final)

    metrics = {
        'rg/dynamic/n_iterations': n_iterations,
        'rg/dynamic/modularity_init': mod_init,
        'rg/dynamic/modularity_final': mod_final,
        'rg/dynamic/modularity_change': mod_final - mod_init,
        'rg/dynamic/rank_init': rank_init,
        'rg/dynamic/rank_final': rank_final,
        'rg/dynamic/rank_change': rank_final - rank_init,
    }

    # If enough iterations, compute mid-point too
    if n_iterations >= 3:
        mid_idx = n_iterations // 2
        beta_mid = beta_history[mid_idx]
        if beta_mid.dim() == 4:
            beta_mid = beta_mid.mean(dim=1)
        metrics['rg/dynamic/modularity_mid'] = compute_modularity(beta_mid)
        metrics['rg/dynamic/rank_mid'] = compute_effective_rank(beta_mid)

    return metrics


# =============================================================================
# Gaussian KL Divergence (Proper Implementation)
# =============================================================================

def gaussian_kl_divergence(
    mu_q: torch.Tensor,      # (B, N, K)
    sigma_q: torch.Tensor,   # (B, N, K, K) full, (B, N, K) diagonal, or None (uses identity)
    mu_p: torch.Tensor,      # (B, N, K)
    sigma_p: torch.Tensor,   # (B, N, K, K) full, (B, N, K) diagonal, or None (uses identity)
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Compute KL(N(μ_q, Σ_q) || N(μ_p, Σ_p)) for Gaussian distributions.

    Handles both full covariance matrices (B, N, K, K) and diagonal covariances (B, N, K).
    Diagonal covariances are detected automatically based on tensor dimensions.

    For full covariances:
        KL = 0.5 * [tr(Σ_p⁻¹ Σ_q) + (μ_p - μ_q)ᵀ Σ_p⁻¹ (μ_p - μ_q) - K + log(|Σ_p|/|Σ_q|)]

    For diagonal covariances (O(K) instead of O(K³)):
        KL = 0.5 * [Σ_k(σ_q[k]/σ_p[k]) + Σ_k((μ_p[k]-μ_q[k])²/σ_p[k]) - K + Σ_k(log(σ_p[k])-log(σ_q[k]))]

    Args:
        mu_q: Posterior means (B, N, K)
        sigma_q: Posterior covariances - (B, N, K, K) full or (B, N, K) diagonal or None
        mu_p: Prior means (B, N, K)
        sigma_p: Prior covariances - (B, N, K, K) full or (B, N, K) diagonal or None
        eps: Numerical stability constant

    Returns:
        kl: KL divergence per agent, shape (B, N)
    """
    K = mu_q.shape[-1]
    device = mu_q.device
    dtype = mu_q.dtype

    # Detect if covariances are diagonal (3D) or full (4D)
    sigma_q_is_diagonal = sigma_q is not None and sigma_q.dim() == 3
    sigma_p_is_diagonal = sigma_p is not None and sigma_p.dim() == 3

    # If either is diagonal, use diagonal path for both (more efficient)
    use_diagonal = sigma_q_is_diagonal or sigma_p_is_diagonal

    if use_diagonal:
        # =================================================================
        # DIAGONAL PATH: O(K) per agent instead of O(K³)
        # =================================================================
        # Convert to diagonal variances if needed
        if sigma_q is None:
            sigma_q_diag = torch.ones(*mu_q.shape, device=device, dtype=dtype)
        elif sigma_q.dim() == 3:
            sigma_q_diag = sigma_q  # Already diagonal (B, N, K)
        else:
            # Extract diagonal from full matrix
            sigma_q_diag = torch.diagonal(sigma_q, dim1=-2, dim2=-1)  # (B, N, K)

        if sigma_p is None:
            sigma_p_diag = torch.ones(*mu_p.shape, device=device, dtype=dtype)
        elif sigma_p.dim() == 3:
            sigma_p_diag = sigma_p  # Already diagonal (B, N, K)
        else:
            # Extract diagonal from full matrix
            sigma_p_diag = torch.diagonal(sigma_p, dim1=-2, dim2=-1)  # (B, N, K)

        # Ensure positivity
        sigma_q_diag = sigma_q_diag.clamp(min=eps)
        sigma_p_diag = sigma_p_diag.clamp(min=eps)

        # Trace term: Σ_k (σ_q[k] / σ_p[k])
        trace_term = (sigma_q_diag / sigma_p_diag).sum(dim=-1)  # (B, N)

        # Mahalanobis term: Σ_k ((μ_p[k] - μ_q[k])² / σ_p[k])
        delta_mu = mu_p - mu_q  # (B, N, K)
        mahal_term = ((delta_mu ** 2) / sigma_p_diag).sum(dim=-1)  # (B, N)

        # Log determinant term: Σ_k (log(σ_p[k]) - log(σ_q[k]))
        logdet_term = (torch.log(sigma_p_diag) - torch.log(sigma_q_diag)).sum(dim=-1)  # (B, N)

        # KL divergence
        kl = 0.5 * (trace_term + mahal_term - K + logdet_term)

    else:
        # =================================================================
        # FULL COVARIANCE PATH: O(K³) via Cholesky
        # =================================================================
        # Default to identity covariances if not provided
        if sigma_q is None:
            sigma_q = torch.eye(K, device=device, dtype=dtype).expand(*mu_q.shape[:-1], K, K)
        if sigma_p is None:
            sigma_p = torch.eye(K, device=device, dtype=dtype).expand(*mu_p.shape[:-1], K, K)

        # Regularize for numerical stability
        sigma_q_reg = sigma_q + eps * torch.eye(K, device=device, dtype=dtype)
        sigma_p_reg = sigma_p + eps * torch.eye(K, device=device, dtype=dtype)

        # Compute Σ_p⁻¹ via Cholesky with progressive regularization fallback
        try:
            L_p = torch.linalg.cholesky(sigma_p_reg)
        except RuntimeError:
            reg = eps
            for attempt in range(5):
                reg *= 10.0
                sigma_p_reg = sigma_p + reg * torch.eye(K, device=device, dtype=dtype)
                sigma_p_reg = 0.5 * (sigma_p_reg + sigma_p_reg.transpose(-1, -2))
                try:
                    L_p = torch.linalg.cholesky(sigma_p_reg)
                    _nr("chol_recover")
                    break
                except RuntimeError:
                    continue
            else:
                _nr("chol_fail")
                L_p = torch.linalg.cholesky(torch.eye(K, device=device, dtype=dtype).expand_as(sigma_p_reg) + eps * torch.eye(K, device=device, dtype=dtype))

        # Trace term: tr(Σ_p⁻¹ Σ_q)
        # Solve L_p @ Y = Σ_q for Y, then tr(Σ_p⁻¹ Σ_q) = tr(L_p⁻ᵀ Y)
        Y = torch.linalg.solve_triangular(L_p, sigma_q_reg, upper=False)
        Z = torch.linalg.solve_triangular(L_p.transpose(-1, -2), Y, upper=True)
        trace_term = torch.diagonal(Z, dim1=-2, dim2=-1).sum(dim=-1)  # (B, N)

        # Mahalanobis term: (μ_p - μ_q)ᵀ Σ_p⁻¹ (μ_p - μ_q)
        delta_mu = mu_p - mu_q  # (B, N, K)
        # Solve L_p @ v = delta_mu
        v = torch.linalg.solve_triangular(L_p, delta_mu.unsqueeze(-1), upper=False).squeeze(-1)
        mahal_term = torch.sum(v ** 2, dim=-1)  # (B, N)

        # Log determinant terms
        logdet_p = 2.0 * torch.sum(torch.log(torch.diagonal(L_p, dim1=-2, dim2=-1).clamp(min=1e-12)), dim=-1)
        try:
            L_q = torch.linalg.cholesky(sigma_q_reg)
        except RuntimeError:
            reg = eps
            for attempt in range(5):
                reg *= 10.0
                sigma_q_reg = sigma_q + reg * torch.eye(K, device=device, dtype=dtype)
                sigma_q_reg = 0.5 * (sigma_q_reg + sigma_q_reg.transpose(-1, -2))
                try:
                    L_q = torch.linalg.cholesky(sigma_q_reg)
                    _nr("chol_recover")
                    break
                except RuntimeError:
                    continue
            else:
                _nr("chol_fail")
                L_q = torch.linalg.cholesky(torch.eye(K, device=device, dtype=dtype).expand_as(sigma_q_reg) + eps * torch.eye(K, device=device, dtype=dtype))
        logdet_q = 2.0 * torch.sum(torch.log(torch.diagonal(L_q, dim1=-2, dim2=-1).clamp(min=1e-12)), dim=-1)
        logdet_term = logdet_p - logdet_q  # (B, N)

        # KL divergence
        kl = 0.5 * (trace_term + mahal_term - K + logdet_term)

    # Clamp to [0, max] for numerical stability.
    # Scale ceiling with K: each dimension contributes O(1) to KL.
    kl_ceil = max(100.0, 5.0 * K)
    kl = torch.clamp(kl, min=0.0, max=kl_ceil)
    # NaN/Inf safety: replace any residual numerical failures
    bad_mask = torch.isnan(kl) | torch.isinf(kl)
    if bad_mask.any():
        _nr("nan_replace")
    kl = kl.nan_to_num(nan=0.0, posinf=kl_ceil, neginf=0.0)
    return kl

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# =============================================================================
# Free Energy Loss Computation (ATTENTION-WEIGHTED)
# =============================================================================

def compute_free_energy_loss(
    model,
    token_ids: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.0,           # Self-coupling: KL(q_i || p_i) — beliefs to priors
    lambda_beta: float = 1.0,     # Belief coupling weight
    lambda_gamma: float = 0.0,    # Model coupling weight
    kappa_gamma: float = 1.0,     # Temperature for γ_ij coupling weights
    lambda_hyper: float = 0.0,    # Hyper-prior: KL(s_i || h) — models to centroid
    pad_token_id: int = -100,     # Token ID to ignore in loss (padding)
    use_obs_in_vfe: bool = False, # Pass targets into VFE E-step (last layer only)
    alpha_phi: float = 0.0,       # Gauge prior weight: (α_φ/2) Σ_i ||φ_i||²
    detach_sigma_kl: bool = True, # Detach sigma in KL loss (prevents M-step sigma gradients)
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute training loss with optional auxiliary VFE regularization terms.

    The core VFE inference (belief updates via ∂F/∂q) happens INSIDE the
    VariationalFFN during the forward pass, controlled by ffn_alpha and
    ffn_lambda_belief. This function computes the TRAINING LOSS, which is:

        L = CE + α · Σ_i KL(q_i || p_i)

    The α term (default 0.1) provides sigma regularization by pulling
    beliefs back toward priors. Remaining auxiliary terms are off by default:
        + λ_β · Σ_{i,j} β_ij · KL(q_i || Ω_{ij}q_j)  [Belief coupling]
        + λ_γ · Σ_{i,j} γ_ij · KL(s_i || Ω_{ij}s_j)  [Model coupling]
        + λ_h · Σ_i KL(s_i || h)                      [Hyper-prior]
        + (α_φ/2) Σ_i ||φ_i||²                        [Gauge prior]

    Args:
        model: GaugeTransformerLM with forward_with_attention() method
        token_ids: (B, N) input token IDs
        targets: (B, N) target token IDs
        alpha: Weight for self-coupling KL(q||p) (default: 0.0)
        lambda_beta: Weight for belief coupling (default: 1.0)
        lambda_gamma: Weight for model coupling (default: 0.0)
        kappa_gamma: Temperature for γ_ij model coupling weights (default: 1.0)
        lambda_hyper: Weight for hyper-prior KL(s_i||h) (default: 0.0)
        pad_token_id: Token ID for padding (ignored in loss). Default -100.

    Returns:
        total_loss: Scalar loss for backprop
        metrics: Dict with loss components
    """
    # =================================================================
    # Forward pass with attention weights and KL matrices
    # =================================================================
    vfe_targets = targets if use_obs_in_vfe else None
    logits, attn_info = model.forward_with_attention(token_ids, targets=vfe_targets)

    beta = attn_info['beta']    # (n_layers, B, n_heads, N, N)
    kl = attn_info['kl']        # (n_layers, B, n_heads, N, N)
    mu_q = attn_info['mu']      # (B, N, K) - evolved beliefs (fast)
    sigma_q = attn_info['sigma']  # (B, N, K, K) or None

    # =================================================================
    # Extract models s_i and priors p_i from attention_info
    # =================================================================
    # Currently p_i = s_i = embedding output (before position encoding).
    # The embedding parameters ARE the models — they're the slow variables
    # that backprop updates. The priors are derived from models; in the
    # simplest case p = s (identity mapping).
    #
    # mu_prior/sigma_prior from model.py are the raw embedding outputs
    # cloned before position encoding is applied to phi.
    mu_s = attn_info['mu_prior']      # (B, N, K) - models (embedding params)
    sigma_s = attn_info['sigma_prior']  # (B, N, K, K) or (B, N, K)
    phi_s = attn_info['phi_prior']      # (B, N, gauge_dim)
    generators = model.generators      # (n_gen, K, K)

    # Priors: currently p_i = s_i. When p = f(s) becomes non-trivial,
    # derive mu_p, sigma_p from mu_s, sigma_s here.
    mu_p = mu_s        # p = s (identity mapping for now)
    sigma_p = sigma_s  # p = s

    # =================================================================
    # 1. Observation Likelihood: -E[log p(o|x)] = Cross-Entropy
    # =================================================================
    ce_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),  # (B*N, V)
        targets.reshape(-1),                   # (B*N,)
        reduction='mean',
        ignore_index=pad_token_id,
    )

    # =================================================================
    # 2. Belief Coupling: λ_β · Σ_ij β_ij · KL(q_i || Ω_ij q_j)
    # =================================================================
    if lambda_beta > 0.0:
        # Use final layer's beta/kl for loss (matches pre-refactor behavior)
        beta_final = beta[-1]  # (B, n_heads, N, N)
        kl_final = kl[-1]      # (B, n_heads, N, N)
        weighted_kl = beta_final * kl_final
        belief_align_loss = weighted_kl.sum(dim=(-2, -1)).mean()
        K = mu_q.shape[-1]
        dim_scale = math.sqrt(max(K, 1))
        belief_align_loss = lambda_beta * belief_align_loss / dim_scale
    else:
        belief_align_loss = torch.tensor(0.0, device=ce_loss.device)

    # =================================================================
    # 3. Self-Coupling: α · KL(q_i || p_i) — beliefs toward PRIORS
    # =================================================================
    # This pulls evolved beliefs (fast) back toward priors (derived from
    # models). This is NOT KL(q||s) — it's KL(q||p) where p = f(s).
    # Currently p = s, so they're numerically identical, but the
    # conceptual distinction matters for the hierarchy.
    # =================================================================
    if alpha > 0.0:
        K = mu_q.shape[-1]
        kl_per_agent = gaussian_kl_divergence(
            mu_q=mu_q,
            sigma_q=(sigma_q.detach() if detach_sigma_kl else sigma_q) if sigma_q is not None else None,
            mu_p=mu_p,        # PRIORS (not models directly)
            sigma_p=(sigma_p.detach() if detach_sigma_kl else sigma_p) if sigma_p is not None else None,
        )  # (B, N)
        dim_scale = math.sqrt(max(K, 1))
        self_consistency_loss = alpha * kl_per_agent.mean() / dim_scale
    else:
        self_consistency_loss = torch.tensor(0.0, device=ce_loss.device)

    # =================================================================
    # 4. Model Coupling: λ_γ · Σ_{i,j} γ_ij · KL(s_i || Ω_{ij} s_j)
    # =================================================================
    # Aligns generative MODELS across agents via gauge transport.
    # This operates on the slow timescale — it's the M-step analog of
    # belief coupling (β term, which operates on q in the E-step).
    #
    # γ_{ij} = softmax_j(-KL(s_i || Ω_{ij} s_j) / κ_γ)
    # Ω_{ij} = exp(φ_i) · exp(-φ_j)
    #
    # Uses s_i (models), NOT p_i (priors). The gamma term couples the
    # learned model parameters, not the derived priors.
    # =================================================================
    if lambda_gamma > 0.0:
        batch_size, num_agents, K = mu_s.shape
        device = mu_s.device

        mask = torch.tril(torch.ones(num_agents, num_agents, device=device))
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)

        diagonal_cov = sigma_s is not None and sigma_s.dim() == 3
        gamma, kl_model = compute_attention_weights(
            mu_s,       # MODELS (not priors)
            sigma_s,    # MODELS
            phi_s,      # Model gauge frames
            generators,
            kappa_gamma,
            epsilon=1e-8,
            mask=mask,
            use_numba=False,
            return_kl=True,
            diagonal_covariance=diagonal_cov,
        )

        weighted_kl_model = gamma * kl_model  # (B, N, N)
        K = mu_s.shape[-1]
        dim_scale = math.sqrt(max(K, 1))
        model_align_loss = lambda_gamma * weighted_kl_model.sum(dim=(-2, -1)).mean() / dim_scale
    else:
        model_align_loss = torch.tensor(0.0, device=ce_loss.device)

    # =================================================================
    # 5. Hyper-Prior: λ_h · Σ_i KL(s_i || h) — models toward centroid
    # =================================================================
    # h = centroid of all models {s_i}. This is the key anti-memorization
    # mechanism: constrains per-position models to stay near a shared
    # centroid, forcing generalization through gauge transport Ω_{ij}
    # rather than per-position lookup tables.
    #
    # h has broadened variance (2×) to allow model diversity while
    # preventing unconstrained drift.
    # =================================================================
    if lambda_hyper > 0.0:
        batch_size, num_agents, K = mu_s.shape
        dim_scale = math.sqrt(max(K, 1))

        # Centroid h = mean of all models (detach: treat as fixed target)
        mu_h = mu_s.mean(dim=1, keepdim=True).detach()  # (B, 1, K)

        if sigma_s is not None:
            # Broadened variance (2×) allows model diversity
            sigma_h = sigma_s.mean(dim=1, keepdim=True).detach() * 2.0
        else:
            sigma_h = None

        # Expand centroid to match all agents
        mu_h_expanded = mu_h.expand_as(mu_s)
        sigma_h_expanded = sigma_h.expand_as(sigma_s) if sigma_h is not None else None

        # KL(s_i || h) for each agent position
        kl_hyper = gaussian_kl_divergence(
            mu_q=mu_s,
            sigma_q=sigma_s.detach() if sigma_s is not None else None,
            mu_p=mu_h_expanded,
            sigma_p=sigma_h_expanded,
        )  # (B, N)

        hyper_prior_loss = lambda_hyper * kl_hyper.mean() / dim_scale
    else:
        hyper_prior_loss = torch.tensor(0.0, device=ce_loss.device)

    # =================================================================
    # 6. Gauge Prior: (α_φ/2) Σ_i ||φ_i||² — mass term for gauge field
    # =================================================================
    if alpha_phi > 0.0:
        K = mu_q.shape[-1]
        dim_scale = math.sqrt(max(K, 1))
        # phi_s is the model gauge frame from embeddings (B, N, n_gen)
        phi_norm_sq = (phi_s ** 2).sum(dim=-1).mean()  # Mean over batch and tokens
        gauge_prior_loss = (alpha_phi / 2.0) * phi_norm_sq / dim_scale
    else:
        gauge_prior_loss = torch.tensor(0.0, device=ce_loss.device)

    # =================================================================
    # Total Training Loss (CE + optional auxiliary VFE regularizers)
    # =================================================================
    total_loss = (ce_loss + belief_align_loss + self_consistency_loss
                  + model_align_loss + hyper_prior_loss + gauge_prior_loss)

    # Compute attention metrics outside the computation graph
    with torch.no_grad():
        beta_avg = beta[-1].mean(dim=1)  # (B, N, N) - last layer, average over heads
        beta_safe = beta_avg.clamp(min=1e-10)
        attn_entropy = -(beta_safe * beta_safe.log()).sum(dim=-1).mean()
        attn_concentration = beta_avg.max(dim=-1)[0].mean()

    # Metrics
    metrics = {
        'loss/total': total_loss.item(),
        'loss/ce': ce_loss.item(),
        'loss/belief_align': belief_align_loss.item(),
        'loss/self_consistency': self_consistency_loss.item() if alpha > 0 else 0.0,
        'loss/model_coupling': model_align_loss.item() if lambda_gamma > 0 else 0.0,
        'loss/hyper_prior': hyper_prior_loss.item() if lambda_hyper > 0 else 0.0,
        'loss/gauge_prior': gauge_prior_loss.item() if alpha_phi > 0 else 0.0,
        'attention/beta_mean': beta.mean().item(),
        'attention/kl_mean': kl.mean().item(),
        'attention/entropy': attn_entropy.item(),
        'attention/concentration': attn_concentration.item(),
    }

    # Bayesian alpha diagnostics
    with torch.no_grad():
        for block in model.transformer.blocks:
            vffn = getattr(block.ffn, 'variational_ffn', None)
            if vffn is not None and vffn.learnable_alpha and mu_q is not None and mu_p is not None:
                import torch.nn.functional as _F
                alpha_vals = vffn.get_bayesian_alpha(mu_q, mu_p, sigma_p)
                a0 = _F.softplus(vffn.raw_a0)
                b0 = _F.softplus(vffn.raw_b0)
                delta = mu_q - mu_p
                if sigma_p.dim() == 3:
                    mahal_sq = (delta ** 2 / sigma_p.clamp(min=1e-6)).sum(dim=-1)
                else:
                    K = mu_q.shape[-1]
                    sigma_p_metric = sigma_p + 1e-6 * torch.eye(K, device=mu_q.device)
                    try:
                        sp_inv = torch.linalg.inv(sigma_p_metric)
                    except (torch.linalg.LinAlgError, RuntimeError):
                        _nr("inv_pinv")
                        sp_inv = torch.linalg.pinv(sigma_p_metric)
                    mahal_sq = torch.einsum('bni,bnij,bnj->bn', delta, sp_inv, delta)
                metrics['bayesian/alpha_mean'] = alpha_vals.mean().item()
                metrics['bayesian/alpha_std'] = alpha_vals.std().item()
                metrics['bayesian/alpha_min'] = alpha_vals.min().item()
                metrics['bayesian/alpha_max'] = alpha_vals.max().item()
                metrics['bayesian/a0'] = a0.item()
                metrics['bayesian/b0'] = b0.item()
                metrics['bayesian/mahal_sq_mean'] = mahal_sq.mean().item()
                metrics['bayesian/mahal_sq_std'] = mahal_sq.std().item()
                break  # Only first layer for now

    if lambda_gamma > 0.0:
        metrics['attention/gamma_mean'] = gamma.mean().item()
        metrics['attention/kl_model_mean'] = kl_model.mean().item()

    # =================================================================
    # P-FLOW DATA: Include beliefs and per-position CE for optional P-flow updates
    # =================================================================
    # Compute per-position CE for weighting P-flow updates (low error = successful belief)
    with torch.no_grad():
        ce_per_position = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction='none',
            ignore_index=pad_token_id,
        ).reshape(targets.shape)  # (B, N)

    # Store in metrics for optional P-flow in training loop
    metrics['p_flow/mu_q'] = mu_q.detach()           # (B, N, K) final beliefs
    metrics['p_flow/ce_per_position'] = ce_per_position  # (B, N) per-position CE

    # Store attention info for RG metrics computation (detached)
    metrics['attention_info'] = {
        'beta': beta.detach(),
        'kl': kl.detach(),
        'mu': mu_q.detach(),
        'sigma': sigma_q.detach() if sigma_q is not None else None,
    }

    return total_loss, metrics


# =============================================================================
# Training Configuration
# =============================================================================

@dataclass
class TrainingConfig:
    """
    Unified training configuration supporting both simple and multi-group parameter optimization.

    Modes:
    - Simple (use_param_groups=False): Single learning rate for all parameters
    - Multi-group (use_param_groups=True): Separate learning rates for mu, sigma, phi, attention, ffn, output
    """

    # Parameter grouping strategy
    use_param_groups: bool = False  # If True, use multi-group learning rates (natural gradients!)

    # Simple mode: Single learning rate (used when use_param_groups=False)
    learning_rate: float = 3e-4

    # Multi-group mode: Per-parameter group learning rates (used when use_param_groups=True)
    mu_lr: float = 0.1           # Mean embeddings (natural gradient scale)
    sigma_lr: float = 0.005      # Covariance embeddings (smaller for stability)
    phi_lr: float = 0.01         # Gauge frames
    attention_lr: float = 0.01   # Attention parameters
    ffn_lr: float = 0.001        # FFN parameters (standard)
    output_lr: float = 0.001     # Output projection

    # Optimizer hyperparameters
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0

    # Learning rate schedule
    warmup_steps: int = 1000
    max_steps: int = 50000
    lr_decay: str = 'cosine'  # 'cosine', 'linear', 'constant'
    min_lr: float = 3e-5

    # Free energy weights (FEP hierarchy: h → s → p → q → observations)
    # NOTE: alpha > 0 is CRITICAL for gradient flow to embeddings!
    # KL(q||p) pulls evolved beliefs back to priors and provides
    # gradients to mu_embed even when FFN outputs are detached.
    alpha: float = 0.1           # Self-coupling: KL(q||p) beliefs to priors
    lambda_beta: float = 1.0     # Belief coupling: Σβ_ij·KL(q_i||Ω_ij q_j) (CRUCIAL!)
    lambda_gamma: float = 0.0    # Model coupling: Σγ_ij·KL(s_i||Ω_ij s_j) (disabled by default)
    kappa_gamma: float = 1.0     # Temperature for γ_ij model coupling weights
    lambda_hyper: float = 0.0    # Hyper-prior: KL(s_i||h) models to centroid (disabled by default)

    # Training
    batch_size: int = 16
    num_epochs: Optional[int] = None
    accumulation_steps: int = 1

    # Logging
    log_every: int = 100
    eval_every: int = 1000
    save_every: int = 5000
    log_interval: int = 10       # Alias for log_every (for compatibility)
    eval_interval: int = 100     # Alias for eval_every (for compatibility)
    checkpoint_interval: int = 200

    # Early stopping
    patience: int = 0  # If > 0, stop if no improvement for this many evals

    # Checkpointing
    checkpoint_dir: Optional[Path] = None
    save_optimizer: bool = True
    save_total_limit: int = 3
    resume_from: Optional[str] = None  # Path to checkpoint to resume from

    # Weights & Biases
    use_wandb: bool = False
    wandb_project: str = 'gauge-transformer'
    wandb_run_name: Optional[str] = None

    # Device
    device: str = 'cpu'
    use_amp: bool = False


# =============================================================================
# Trainer Class
# =============================================================================

class Trainer:
    """Training orchestration for gauge transformer."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[TrainingConfig] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or TrainingConfig()

        # Use -100 for loss masking (PyTorch cross_entropy default ignore_index).
        # Target tensors use -100 for padding positions to avoid masking valid
        # token IDs (e.g., tiktoken token 0 = "!" would be incorrectly ignored).
        self.pad_token_id = -100

        # Move model to device
        self.device = torch.device(self.config.device)
        self.model = self.model.to(self.device)

        # Optimizer
        self.optimizer = self._create_optimizer()

        # Learning rate scheduler
        self.scheduler = self._create_scheduler()

        # Mixed precision scaler (using modern AMP API for PyTorch 2.x / CUDA 12+)
        if self.config.use_amp and self.config.device == 'cuda':
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None

        # Training state
        self.step = 0
        self.epoch = 0
        self.best_val_ce = float('inf')  # Track CE loss (not total loss) for best model

        # Create checkpoint directory
        if self.config.checkpoint_dir is not None:
            self.config.checkpoint_dir = Path(self.config.checkpoint_dir)
            self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize W&B
        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name,
                config=vars(self.config),
            )
            wandb.watch(self.model, log='all', log_freq=1000)

        print("Trainer initialized")
        print(f"  Device: {self.device}")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  Optimizer: AdamW (lr={self.config.learning_rate})")
        print(f"  λ_β (attention-weighted KL): {self.config.lambda_beta}")
        print(f"  Max steps: {self.config.max_steps:,}")

        # Resume from checkpoint if specified
        if self.config.resume_from is not None:
            print(f"\n  Resuming from checkpoint: {self.config.resume_from}")
            self.load_checkpoint(self.config.resume_from)

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """
        Create AdamW optimizer with configurable parameter grouping.

        Modes:
        - Simple (use_param_groups=False): 2 groups (decay vs no-decay) with single LR
        - Multi-group (use_param_groups=True): 6 groups (mu, sigma, phi, attention, ffn, output)
        """
        if self.config.use_param_groups:
            # Multi-group mode: Natural gradients with per-parameter-type learning rates
            return self._create_multigroup_optimizer()
        else:
            # Simple mode: Traditional 2-group optimizer (decay vs no-decay)
            return self._create_simple_optimizer()

    def _create_simple_optimizer(self) -> torch.optim.Optimizer:
        """Create simple 2-group optimizer (decay vs no-decay)."""
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # No weight decay for biases, norms, and optionally embeddings
                no_decay_match = 'bias' in name or 'norm' in name
                if self.config.embed_no_decay:
                    no_decay_match = no_decay_match or 'embed' in name
                if no_decay_match:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)

        optimizer = torch.optim.AdamW([
            {'params': decay_params, 'weight_decay': self.config.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ], lr=self.config.learning_rate, betas=(self.config.beta1, self.config.beta2))

        return optimizer

    def _create_multigroup_optimizer(self) -> torch.optim.Optimizer:
        """
        Create optimizer with per-parameter group learning rates.

        Parameter Groups:
            1. mu_embed: Mean embeddings
            2. sigma_embed: Covariance embeddings
            3. phi_embed: Gauge frame embeddings
            4. attention: Attention mechanism
            5. ffn: Feed-forward networks
            6. output: Output projection

        This exploits natural gradient structure on statistical manifolds!
        """
        # Collect parameters by type
        mu_params = []
        sigma_params = []
        phi_params = []
        attention_params = []
        ffn_params = []
        output_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            # Mean embeddings (matches both mu_prior and prior_mu naming conventions)
            if 'mu_embed' in name or 'mu_prior' in name or 'prior_mu' in name:
                mu_params.append(param)
            # Covariance embeddings (matches log_sigma, sigma_prior, log_prior_sigma, etc.)
            elif 'sigma_embed' in name or 'log_sigma' in name or 'sigma_prior' in name or 'prior_sigma' in name or 'log_prior' in name:
                sigma_params.append(param)
            # Gauge frame embeddings
            elif 'phi_embed' in name or 'phi_prior' in name:
                phi_params.append(param)
            # Positional encoding (treat as gauge frames)
            elif 'pos_encoding' in name or 'position' in name:
                phi_params.append(param)
            # Attention mechanism
            elif 'attention' in name or 'attn' in name:
                attention_params.append(param)
            # Output projection
            elif 'out_proj' in name or 'lm_head' in name:
                output_params.append(param)
            # FFN (default for everything else)
            else:
                ffn_params.append(param)

        # Create parameter groups
        param_groups = []

        embed_wd = 0.0 if self.config.embed_no_decay else self.config.weight_decay

        if mu_params:
            param_groups.append({
                'params': mu_params,
                'lr': self.config.mu_lr,
                'weight_decay': embed_wd,
                'name': 'mu_embed',
            })
            print(f"  Parameter group 'mu_embed': {len(mu_params)} tensors @ lr={self.config.mu_lr}, wd={embed_wd}")

        if sigma_params:
            param_groups.append({
                'params': sigma_params,
                'lr': self.config.sigma_lr,
                'weight_decay': embed_wd,
                'name': 'sigma_embed',
            })
            print(f"  Parameter group 'sigma_embed': {len(sigma_params)} tensors @ lr={self.config.sigma_lr}, wd={embed_wd}")

        if phi_params:
            param_groups.append({
                'params': phi_params,
                'lr': self.config.phi_lr,
                'weight_decay': embed_wd,
                'name': 'phi_embed',
            })
            print(f"  Parameter group 'phi_embed': {len(phi_params)} tensors @ lr={self.config.phi_lr}")

        if attention_params:
            param_groups.append({
                'params': attention_params,
                'lr': self.config.attention_lr,
                'weight_decay': self.config.weight_decay,
                'name': 'attention',
            })
            print(f"  Parameter group 'attention': {len(attention_params)} tensors @ lr={self.config.attention_lr}")

        if ffn_params:
            param_groups.append({
                'params': ffn_params,
                'lr': self.config.ffn_lr,
                'weight_decay': self.config.weight_decay,
                'name': 'ffn',
            })
            print(f"  Parameter group 'ffn': {len(ffn_params)} tensors @ lr={self.config.ffn_lr}")

        if output_params:
            param_groups.append({
                'params': output_params,
                'lr': self.config.output_lr,
                'weight_decay': 0.0,  # Often tied to embeddings
                'name': 'output',
            })
            print(f"  Parameter group 'output': {len(output_params)} tensors @ lr={self.config.output_lr}")

        optimizer = torch.optim.AdamW(
            param_groups,
            betas=(self.config.beta1, self.config.beta2),
            eps=self.config.eps,
        )

        return optimizer

    def _create_scheduler(self):
        """Create learning rate scheduler."""
        if self.config.lr_decay == 'constant':
            return None

        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return step / max(1, self.config.warmup_steps)

            if self.config.lr_decay == 'cosine':
                progress = min(1.0, (step - self.config.warmup_steps) / max(1, self.config.max_steps - self.config.warmup_steps))
                return self.config.min_lr / self.config.learning_rate + \
                       0.5 * (1 - self.config.min_lr / self.config.learning_rate) * \
                       (1 + math.cos(progress * math.pi))
            elif self.config.lr_decay == 'linear':
                min_ratio = self.config.min_lr / self.config.learning_rate
                return max(min_ratio, (self.config.max_steps - step) / (self.config.max_steps - self.config.warmup_steps))
            else:
                return 1.0

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        # model.train() is called once in train() method, not per-step

        token_ids, targets = batch
        token_ids = token_ids.to(self.device)
        targets = targets.to(self.device)

        # Forward + loss
        # NOTE: Do NOT wrap model forward in autocast. This gauge transformer
        # has no W_Q/W_K/W_V linear layers — it's all analytical VFE with
        # eigendecomposition, Cholesky, log, exp, matrix inversion. Float16
        # destroys these operations. AMP only helps models with large matmuls
        # (standard transformers). GradScaler is still used for loss scaling.
        loss, metrics = compute_free_energy_loss(
            self.model,
            token_ids,
            targets,
            alpha=self.config.alpha,
            lambda_beta=self.config.lambda_beta,
            lambda_gamma=self.config.lambda_gamma,
            kappa_gamma=self.config.kappa_gamma,
            lambda_hyper=getattr(self.config, 'lambda_hyper', 0.0),
            pad_token_id=self.pad_token_id,
            use_obs_in_vfe=getattr(self.config, 'use_obs_in_vfe', False),
            alpha_phi=getattr(self.config, 'alpha_phi', 0.0),
            detach_sigma_kl=getattr(self.config, 'detach_sigma_kl', True),
        )

        # Scale loss for gradient accumulation
        loss = loss / self.config.accumulation_steps

        # Backward
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # =================================================================
        # Gradient Monitoring (only on logging steps to avoid GPU sync overhead)
        # =================================================================
        if self.step % self.config.log_every == 0:
            grad_norms = {}
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    if 'mu_embed' in name:
                        grad_norms['grad/mu_embed'] = grad_norm
                    elif 'sigma_embed' in name or 'log_sigma' in name:
                        grad_norms['grad/sigma_embed'] = grad_norm
                    elif 'phi_embed' in name:
                        grad_norms['grad/phi_embed'] = grad_norm
                    elif 'out_proj' in name:
                        grad_norms['grad/out_proj'] = grad_norm
            metrics.update(grad_norms)

            if 'grad/mu_embed' in grad_norms and grad_norms['grad/mu_embed'] == 0.0:
                print("[WARNING] mu_embed gradient is ZERO - gradient flow may be broken!")

        # Optimizer step (if accumulation complete)
        if (self.step + 1) % self.config.accumulation_steps == 0:
            # Gradient clipping
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)

            # Per-group gradient clipping for large gauge groups.
            # With SO(100), phi_embed has 4950 dims per token vs 100 for mu.
            # Global clipping at grad_clip=1.0 means phi dominates the norm,
            # starving mu/sigma of learning signal. Clip each param group
            # independently so all parameter types get sufficient gradients.
            if self.config.use_param_groups:
                for group in self.optimizer.param_groups:
                    if group['params']:
                        torch.nn.utils.clip_grad_norm_(
                            group['params'],
                            self.config.grad_clip
                        )
            else:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip
                )

            # Optimizer step
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()

            # Zero gradients
            self.optimizer.zero_grad()

        # Add learning rate to metrics
        metrics['lr'] = self.optimizer.param_groups[0]['lr']

        return metrics

    @torch.no_grad()
    def validate(self, max_batches: int = 200) -> Dict[str, float]:
        """Validation pass.

        Args:
            max_batches: Maximum number of validation batches to evaluate.
                Prevents long stalls on large datasets like WikiText-103.
        """
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        total_ce_loss = 0.0
        n_batches = 0

        for batch in self.val_loader:
            token_ids, targets = batch
            token_ids = token_ids.to(self.device)
            targets = targets.to(self.device)

            loss, metrics = compute_free_energy_loss(
                self.model, token_ids, targets,
                alpha=self.config.alpha,
                lambda_beta=self.config.lambda_beta,
                lambda_gamma=self.config.lambda_gamma,
                kappa_gamma=self.config.kappa_gamma,
                lambda_hyper=getattr(self.config, 'lambda_hyper', 0.0),
                pad_token_id=self.pad_token_id,
                use_obs_in_vfe=getattr(self.config, 'use_obs_in_vfe', False),
                alpha_phi=getattr(self.config, 'alpha_phi', 0.0),
                detach_sigma_kl=getattr(self.config, 'detach_sigma_kl', True),
            )

            total_loss += loss.item()
            total_ce_loss += metrics['loss/ce']
            n_batches += 1

            if n_batches >= max_batches:
                break

        avg_loss = total_loss / max(n_batches, 1)
        avg_ce_loss = total_ce_loss / max(n_batches, 1)
        perplexity = math.exp(min(avg_ce_loss, 20.0))  # Cap to avoid overflow

        return {
            'val/loss': avg_loss,
            'val/ce_loss': avg_ce_loss,
            'val/perplexity': perplexity,
        }

    def train(self):
        """Main training loop."""
        print("\n" + "="*70)
        print("STARTING TRAINING (Attention-Weighted Free Energy)")
        print("="*70)

        if TQDM_AVAILABLE:
            pbar = tqdm(total=self.config.max_steps, desc="Training")
        else:
            pbar = None

        start_time = time.time()
        self.model.train()

        try:
            while self.step < self.config.max_steps:
                for batch in self.train_loader:
                    # Training step
                    metrics = self.train_step(batch)

                    # Logging
                    if self.step % self.config.log_every == 0:
                        elapsed = time.time() - start_time
                        tokens_per_sec = (self.step * self.config.batch_size * batch[0].shape[1]) / elapsed

                        # Basic metrics
                        print(f"\nStep {self.step:6d} | Loss: {metrics['loss/total']:.4f} | "
                              f"CE: {metrics['loss/ce']:.4f} | Align: {metrics['loss/belief_align']:.4f} | "
                              f"LR: {metrics['lr']:.2e}")

                        # Gradient norms (verify gradients are flowing!)
                        grad_mu = metrics.get('grad/mu_embed', 0.0)
                        grad_out = metrics.get('grad/out_proj', 0.0)
                        grad_phi = metrics.get('grad/phi_embed', 0.0)
                        print(f"         Grads | μ_embed: {grad_mu:.4f} | out_proj: {grad_out:.4f} | φ_embed: {grad_phi:.4f}")

                        # Report numerical fallback events since last log
                        num_events = _nflush()
                        if num_events:
                            parts = " | ".join(f"{k}: {v}" for k, v in num_events.items())
                            print(f"\n         [NUMERICAL] {parts}\n")

                        if self.config.use_wandb and WANDB_AVAILABLE:
                            wandb.log(metrics, step=self.step)

                    # Validation
                    if self.step % self.config.eval_every == 0 and self.step > 0:
                        val_metrics = self.validate()
                        self.model.train()  # Restore training mode after validation
                        if val_metrics:
                            print(f"\nValidation | Loss: {val_metrics['val/loss']:.4f} | "
                                  f"PPL: {val_metrics['val/perplexity']:.2f}")

                            if self.config.use_wandb and WANDB_AVAILABLE:
                                wandb.log(val_metrics, step=self.step)

                            # Save best model based on CE loss (not total loss)
                            # CE loss is the proper metric since PPL = exp(CE)
                            if val_metrics['val/ce_loss'] < self.best_val_ce:
                                self.best_val_ce = val_metrics['val/ce_loss']
                                self.save_checkpoint('best_model.pt')

                    # Checkpointing
                    if self.step % self.config.save_every == 0 and self.step > 0:
                        self.save_checkpoint(f'checkpoint_step_{self.step}.pt')

                    # Update progress
                    if pbar is not None:
                        pbar.update(1)
                        pbar.set_postfix({'loss': f"{metrics['loss/total']:.4f}"})

                    self.step += 1

                    if self.step >= self.config.max_steps:
                        break

                self.epoch += 1

        except KeyboardInterrupt:
            print("\n⚠ Training interrupted by user")

        finally:
            if pbar is not None:
                pbar.close()

        # Final validation
        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)

        final_metrics = self.validate()
        if final_metrics:
            print(f"Final Validation Loss: {final_metrics['val/loss']:.4f}")
            print(f"Final Perplexity: {final_metrics['val/perplexity']:.2f}")

        # Save final model
        self.save_checkpoint('final_model.pt')

        print("="*70)

    def save_checkpoint(self, filename: str):
        """Save training checkpoint."""
        if self.config.checkpoint_dir is None:
            return

        checkpoint_path = self.config.checkpoint_dir / filename

        checkpoint = {
            'step': self.step,
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'best_val_ce': self.best_val_ce,
            'config': self.model.config,
        }

        if self.config.save_optimizer:
            checkpoint['optimizer_state'] = self.optimizer.state_dict()
            if self.scheduler is not None:
                checkpoint['scheduler_state'] = self.scheduler.state_dict()

        torch.save(checkpoint, checkpoint_path)
        print(f"  💾 Saved checkpoint: {checkpoint_path.name}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        state_key = 'model_state_dict' if 'model_state_dict' in checkpoint else 'model_state'
        self.model.load_state_dict(checkpoint[state_key])

        if 'optimizer_state' in checkpoint and self.config.save_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])

        if 'scheduler_state' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state'])

        self.step = checkpoint.get('step', 0)
        self.epoch = checkpoint.get('epoch', 0)
        # Backward compatible: try new key first, fall back to old key
        self.best_val_ce = checkpoint.get('best_val_ce', checkpoint.get('best_val_loss', float('inf')))

        print(f"✓ Loaded checkpoint from step {self.step}")


# =============================================================================
# PURE FEP MODE: Backprop-Free Training via Prior Evolution
# =============================================================================
# This implements the BELIEF (Backprop-free Evolving Local Inference via Free Energy)
# training paradigm where learning happens through prior evolution, not backprop.
#
# Key differences from standard training:
# 1. Forward pass wrapped in torch.no_grad() - no gradient tracking
# 2. Targets ARE passed to VFE dynamics (observation term active)
# 3. Learning via update_priors_from_beliefs() based on prediction errors
# 4. Persistent priors in each FFN layer consolidate successful beliefs
#
# The theoretical basis:
# - VFE includes observation term: F = KL(q||p) + alignment + CE
# - Beliefs adjust to minimize CE during forward pass
# - Low-error beliefs update persistent priors (soft EM)
# - Priors consolidate knowledge without backprop
# - Embeddings update toward successful beliefs (tied with output projection)
# =============================================================================


def update_output_embeddings_pflow(
    model,
    targets: torch.Tensor,         # (B, N) target token IDs
    mu_beliefs: torch.Tensor,      # (B, N, K) final beliefs (posteriors)
    prediction_errors: torch.Tensor,  # (B, N) per-position CE
    lr: float = 0.1,
):
    """
    P-FLOW: Update OUTPUT embeddings toward beliefs that predict each target.

    CRITICAL FOR LEARNING in pure FEP mode with untied embeddings!

    When embeddings are untied:
    - Input embeddings (priors) = what we expect for input tokens
    - Output embeddings (W_out) = what beliefs should look like to predict tokens

    If we only update input embeddings, the output projection never learns
    which belief patterns correspond to which tokens -> random predictions!

    This function updates W_out[target_v] toward the belief at positions
    that should predict token v.

    Args:
        model: GaugeTransformerLM with out_proj
        targets: (B, N) target token IDs (which tokens to predict)
        mu_beliefs: (B, N, K) evolved belief means (posteriors)
        prediction_errors: (B, N) per-position cross-entropy loss
        lr: Base learning rate for output embedding updates
    """
    B, N, K = mu_beliefs.shape

    # Get OUTPUT projection weight
    if not hasattr(model, 'out_proj') or model.out_proj is None:
        return {'out_embed_updates': 0, 'out_embed_mode': 'none'}

    out_weight = model.out_proj.weight  # (V, K)

    with torch.no_grad():
        # Compute weights from prediction errors (low CE = high weight)
        errors_clamped = prediction_errors.clamp(min=1e-6, max=20.0)
        weights = 1.0 / (1.0 + errors_clamped)  # (B, N) in range [0.05, 1]

        # Flatten for processing
        flat_targets = targets.view(-1)  # (B*N,) - TARGET token IDs
        flat_beliefs = mu_beliefs.view(-1, K)  # (B*N, K)
        flat_weights = weights.view(-1)  # (B*N,)

        # Mask out padding tokens (typically -100)
        valid_mask = flat_targets >= 0
        flat_targets = flat_targets[valid_mask]
        flat_beliefs = flat_beliefs[valid_mask]
        flat_weights = flat_weights[valid_mask]

        if len(flat_targets) == 0:
            return {'out_embed_updates': 0, 'out_embed_mode': 'no_valid_targets'}

        # For each unique TARGET token, compute weighted average posterior
        unique_tokens = flat_targets.unique()
        n_updates = 0

        for token_id in unique_tokens:
            mask = flat_targets == token_id
            token_beliefs = flat_beliefs[mask]  # (n_occurrences, K)
            token_weights = flat_weights[mask]  # (n_occurrences,)

            # Weighted average of posteriors for this target token
            weight_sum = token_weights.sum()
            if weight_sum > 0:
                weighted_posterior = (token_beliefs * token_weights.unsqueeze(-1)).sum(dim=0) / weight_sum

                # P-FLOW: output_embed <- (1-lr)*output_embed + lr*posterior
                # Scale lr by confidence (weight_sum / count)
                effective_lr = lr * (weight_sum / mask.sum()).item()
                effective_lr = min(effective_lr, 0.1)  # Cap for stability

                out_weight[token_id] = (
                    (1.0 - effective_lr) * out_weight[token_id] +
                    effective_lr * weighted_posterior
                )
                n_updates += 1

    return {
        'out_embed_updates': n_updates,
        'out_embed_mode': 'out_proj',
    }


def update_input_embeddings_pflow(
    model,
    input_ids: torch.Tensor,       # (B, N) input token IDs
    mu_beliefs: torch.Tensor,      # (B, N, K) final beliefs (posteriors)
    prediction_errors: torch.Tensor,  # (B, N) per-position CE
    lr: float = 0.1,
):
    """
    P-FLOW: Update INPUT embeddings toward posteriors (beliefs).

    This is the correct FEP learning rule:
    - Prior (input embedding) moves toward posterior (belief)
    - μ_p ← (1-lr) * μ_p + lr * μ_q
    - Weighted by prediction success (low CE = stronger update)

    With UNTIED embeddings:
    - Input embeddings (W_in) = priors, updated here
    - Output embeddings (W_out) = observation anchors, stay fixed

    Args:
        model: GaugeTransformerLM with untied embeddings
        input_ids: (B, N) input token IDs (which tokens to update)
        mu_beliefs: (B, N, K) evolved belief means (posteriors)
        prediction_errors: (B, N) per-position cross-entropy loss
        lr: Base learning rate for p-flow updates
    """
    B, N, K = mu_beliefs.shape

    # Get INPUT embedding weight (priors)
    token_embed = model.token_embed

    if hasattr(token_embed, 'mu_embed'):
        embed_weight = token_embed.mu_embed.weight  # (V, K) - INPUT embeddings
    elif hasattr(token_embed, 'base_mu'):
        # Gauge-fixed priors: update base_mu toward average belief
        with torch.no_grad():
            errors_clamped = prediction_errors.clamp(min=1e-6, max=20.0)
            weights = 1.0 / (1.0 + errors_clamped)
            weight_sum = weights.sum()
            if weight_sum > 0:
                weighted_belief = (mu_beliefs * weights.unsqueeze(-1)).sum(dim=(0, 1)) / weight_sum
                effective_lr = min(lr * 0.1, 0.01)
                token_embed.base_mu.data = (
                    (1.0 - effective_lr) * token_embed.base_mu.data +
                    effective_lr * weighted_belief
                )
        return {'embed_updates': 1, 'embed_mode': 'base_mu'}
    else:
        return {'embed_updates': 0, 'embed_mode': 'none'}

    with torch.no_grad():
        # Compute weights from prediction errors (low CE = high weight)
        errors_clamped = prediction_errors.clamp(min=1e-6, max=20.0)
        weights = 1.0 / (1.0 + errors_clamped)  # (B, N) in range [0.05, 1]

        # Flatten for processing
        flat_inputs = input_ids.view(-1)  # (B*N,) - INPUT token IDs
        flat_beliefs = mu_beliefs.view(-1, K)  # (B*N, K)
        flat_weights = weights.view(-1)  # (B*N,)

        # For each unique INPUT token, compute weighted average posterior
        unique_tokens = flat_inputs.unique()
        n_updates = 0

        for token_id in unique_tokens:
            mask = flat_inputs == token_id
            token_beliefs = flat_beliefs[mask]  # (n_occurrences, K)
            token_weights = flat_weights[mask]  # (n_occurrences,)

            # Weighted average of posteriors for this input token
            weight_sum = token_weights.sum()
            if weight_sum > 0:
                weighted_posterior = (token_beliefs * token_weights.unsqueeze(-1)).sum(dim=0) / weight_sum

                # P-FLOW: prior <- (1-lr)*prior + lr*posterior
                effective_lr = lr * (weight_sum / mask.sum()).item()
                effective_lr = min(effective_lr, 0.1)  # Cap for stability

                embed_weight[token_id] = (
                    (1.0 - effective_lr) * embed_weight[token_id] +
                    effective_lr * weighted_posterior
                )
                n_updates += 1

        return {'embed_updates': n_updates, 'embed_unique_tokens': len(unique_tokens)}


def pure_fep_train_step(
    model,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
) -> Dict[str, float]:
    """
    Single training step using pure FEP (no backprop).

    Learning happens through prior evolution:
    1. Forward pass WITH targets (CE inside VFE)
    2. Beliefs adjust to minimize prediction error
    3. Priors update toward successful beliefs

    Args:
        model: GaugeTransformerLM with VFE_dynamic FFN in pure_fep_mode
        input_ids: (B, N) input token IDs
        targets: (B, N) target token IDs
        device: Target device

    Returns:
        Dict of training metrics
    """
    model.eval()  # No dropout etc during pure FEP

    input_ids = input_ids.to(device)
    targets = targets.to(device)

    B, N = input_ids.shape

    with torch.no_grad():
        # Forward pass WITH targets - observation term is now active!
        # This is the key change: targets flow into VFE dynamics
        logits, attn_info = model.forward_with_attention(input_ids, targets=targets)

        # Per-position cross-entropy for prior weighting
        ce_per_position = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            reduction='none',
            ignore_index=-100,
        ).view(B, N)

        # Overall loss for logging
        valid_mask = targets != -100
        ce_loss = ce_per_position[valid_mask].mean() if valid_mask.any() else ce_per_position.mean()

        # Get evolved beliefs for prior updates
        mu_beliefs = attn_info['mu']  # (B, N, K)
        sigma_beliefs = attn_info['sigma']  # (B, N, K, K) or (B, N, K)

        # Convert full covariance to diagonal if needed
        if sigma_beliefs is not None and sigma_beliefs.dim() == 4:
            sigma_beliefs = torch.diagonal(sigma_beliefs, dim1=-2, dim2=-1)

        # Update priors in each transformer block's FFN
        prior_stats = {}
        for layer_idx, block in enumerate(model.transformer.blocks):
            if hasattr(block, 'ffn') and hasattr(block.ffn, 'update_priors_from_beliefs'):
                block.ffn.update_priors_from_beliefs(
                    mu_beliefs=mu_beliefs,
                    sigma_beliefs=sigma_beliefs if sigma_beliefs is not None else torch.ones_like(mu_beliefs),
                    prediction_errors=ce_per_position,
                )

                # Collect prior stats from last layer
                if layer_idx == len(model.transformer.blocks) - 1:
                    if hasattr(block.ffn, 'get_prior_stats'):
                        prior_stats = block.ffn.get_prior_stats()

        # P-FLOW: Update INPUT embeddings toward posteriors (beliefs)
        # With untied embeddings:
        # - Input embeddings (priors) get updated here
        embed_stats = update_input_embeddings_pflow(
            model=model,
            input_ids=input_ids,
            mu_beliefs=mu_beliefs,
            prediction_errors=ce_per_position,
            lr=0.1,
        )

        # P-FLOW: Update OUTPUT embeddings toward beliefs that predict targets
        # CRITICAL: With untied embeddings, output projection must also learn!
        # Otherwise logits stay random because W_out never changes.
        out_embed_stats = update_output_embeddings_pflow(
            model=model,
            targets=targets,
            mu_beliefs=mu_beliefs,
            prediction_errors=ce_per_position,
            lr=0.1,
        )
        embed_stats.update(out_embed_stats)

    # Compute metrics (cap loss to prevent overflow: exp(20) ≈ 485M)
    perplexity = torch.exp(ce_loss.clamp(max=20.0)).item()

    metrics = {
        'loss/ce': ce_loss.item(),
        'loss/total': ce_loss.item(),  # No other loss terms in pure FEP
        'perplexity': perplexity,
        'attention/beta_mean': attn_info['beta'][-1].mean().item(),  # Final layer
        'attention/kl_mean': attn_info['kl'][-1].mean().item(),    # Final layer
        **{f'prior/{k}': v for k, v in prior_stats.items()},
        **{f'embed/{k}': v for k, v in embed_stats.items()},
    }

    return metrics


def pure_fep_validate(
    model,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Validate model in pure FEP mode.

    Note: Validation does NOT update priors - just measures performance.

    Args:
        model: GaugeTransformerLM
        dataloader: Validation data
        device: Target device

    Returns:
        Dict of validation metrics
    """
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            # Unpack batch (tuple format from dataloader)
            input_ids, targets = batch
            input_ids = input_ids.to(device)
            targets = targets.to(device)

            # Forward WITHOUT targets for fair evaluation
            # (beliefs don't get to see answers during eval)
            logits, _ = model.forward_with_attention(input_ids, targets=None)

            # Compute CE loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                reduction='sum',
                ignore_index=-100,
            )

            valid_tokens = (targets != -100).sum().item()
            total_loss += loss.item()
            total_tokens += valid_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = np.exp(min(avg_loss, 20.0))  # Cap to prevent overflow

    return {
        'val/loss': avg_loss,
        'val/perplexity': perplexity,
    }


# NOTE: PureFEPConfig and PureFEPTrainer have been moved to transformer/experimental/
# For pure FEP training, use:
#   from transformer.experimental.pure_fep_transformer import PureFEPConfig, PureFEPTrainer
# Or use the pure_fep_train_step() and pure_fep_validate() functions above directly.