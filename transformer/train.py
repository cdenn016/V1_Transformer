"""
Training Loss Functions for Gauge-Theoretic Transformer
=======================================================

Core loss computation and RG metrics used by training scripts.

Exports:
    - compute_free_energy_loss(): M-step training loss (CE + KL regularizers)
    - compute_rg_metrics_from_attention(): RG flow analysis from attention info
    - compute_dynamic_rg_metrics(): Dynamic RG tracking across VFE iterations
    - gaussian_kl_divergence(): KL(N(μ_q,Σ_q) || N(μ_p,Σ_p)) for Gaussians

The actual training loop is in train_publication.py (PublicationTrainer).
"""

# Suppress Triton and CuPy warnings BEFORE torch import (torch may trigger imports)
import warnings
warnings.filterwarnings("ignore", message="Failed to find cuobjdump", module="triton")
warnings.filterwarnings("ignore", message="Failed to find nvdisasm", module="triton")
warnings.filterwarnings("ignore", message="CUDA path could not be detected", module="cupy")

from math_utils.numerical_monitor import record as _nr

import math
import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any
from transformer.analysis.rg_metrics import compute_rg_diagnostics
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
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute training loss (M-step objective in the hierarchical VFE).

    This computes the Level 1-2 loss terms. The full Bayesian hierarchy is:
        Level 3: N(0, 1/(2·wd)) hyper-prior on embeddings  [optimizer weight decay]
        Level 2: p_i = N(μ_p, Σ_p) priors                  [α · KL(q||p) below]
        Level 1: q_i = N(μ_q, Σ_q) beliefs                 [inferred in E-step]
        Level 0: x_i observations                           [CE loss below]

    The E-step (belief inference via ∂F/∂q) happens INSIDE VariationalFFN during
    the forward pass. This function computes the M-step training loss:

        L = CE + α · Σ_i KL(q_i || p_i)

    The α term (default 0.1) couples Level 1→2, pulling beliefs toward priors.
    Weight decay on embedding parameters (Level 2→3) is applied by the optimizer.
    Remaining auxiliary terms are off by default:
        + λ_β · Σ_{i,j} β_ij · KL(q_i || Ω_{ij}q_j)  [Belief coupling]
        + λ_γ · Σ_{i,j} γ_ij · KL(s_i || Ω_{ij}s_j)  [Model coupling]
        + λ_h · Σ_i KL(s_i || h)                      [Hyper-prior on models]
        + (α_φ/2) Σ_i ||φ_i||²                        [Gauge prior on φ]

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
        # Sigma gradients flow through KL(q||p): ∂KL/∂Σ_q = 0.5(Σ_p⁻¹ - Σ_q⁻¹)
        # pulls sigma embeddings toward consistency with evolved beliefs.
        kl_per_agent = gaussian_kl_divergence(
            mu_q=mu_q,
            sigma_q=sigma_q,
            mu_p=mu_p,        # PRIORS (not models directly)
            sigma_p=sigma_p,
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
            vffn = block.ffn if hasattr(block.ffn, 'learnable_alpha') else getattr(block.ffn, 'variational_ffn', None)
            if vffn is not None and vffn.learnable_alpha and mu_q is not None and mu_p is not None:
                import torch.nn.functional as _F
                alpha_vals = vffn.get_bayesian_alpha(mu_q, mu_p, sigma_p, sigma_q)
                c0 = _F.softplus(vffn.raw_c0)
                b0 = _F.softplus(vffn.raw_b0)
                # Compute full KL(q||p) for diagnostics
                delta = mu_q - mu_p
                K_dim = mu_q.shape[-1]
                if sigma_p.dim() == 3:
                    sp_safe = sigma_p.clamp(min=1e-6)
                    sq_safe = sigma_q.clamp(min=1e-6)
                    trace_t = (sq_safe / sp_safe).sum(dim=-1)
                    mahal_t = (delta ** 2 / sp_safe).sum(dim=-1)
                    logdet_t = (torch.log(sp_safe) - torch.log(sq_safe)).sum(dim=-1)
                else:
                    sigma_p_metric = sigma_p + 1e-6 * torch.eye(K_dim, device=mu_q.device)
                    try:
                        sp_inv = torch.linalg.inv(sigma_p_metric)
                    except (torch.linalg.LinAlgError, RuntimeError):
                        _nr("inv_pinv")
                        sp_inv = torch.linalg.pinv(sigma_p_metric)
                    prod = torch.matmul(sp_inv, sigma_q)
                    trace_t = prod.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
                    mahal_t = torch.einsum('bni,bnij,bnj->bn', delta, sp_inv, delta)
                    logdet_p = torch.linalg.slogdet(sigma_p_metric.float())[1]
                    logdet_q = torch.linalg.slogdet(sigma_q.float())[1]
                    logdet_t = logdet_p - logdet_q
                kl_qp = 0.5 * (trace_t + mahal_t - K_dim + logdet_t).clamp(min=0.0)
                metrics['bayesian/alpha_mean'] = alpha_vals.mean().item()
                metrics['bayesian/alpha_std'] = alpha_vals.std().item()
                metrics['bayesian/alpha_min'] = alpha_vals.min().item()
                metrics['bayesian/alpha_max'] = alpha_vals.max().item()
                metrics['bayesian/c0'] = c0.mean().item()
                metrics['bayesian/b0'] = b0.mean().item()
                metrics['bayesian/c0_std'] = c0.std().item()
                metrics['bayesian/b0_std'] = b0.std().item()
                metrics['bayesian/kl_qp_mean'] = kl_qp.mean().item()
                metrics['bayesian/kl_qp_std'] = kl_qp.std().item()
                metrics['bayesian/mahal_sq_mean'] = mahal_t.mean().item()
                metrics['bayesian/mahal_sq_std'] = mahal_t.std().item()
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
    if sigma_q is not None:
        metrics['p_flow/sigma_q'] = sigma_q.detach()  # (B, N, K) or (B, N, K, K) belief variances
    phi_evolved = attn_info.get('phi')
    if phi_evolved is not None:
        metrics['p_flow/phi_evolved'] = phi_evolved.detach()  # (B, N, phi_dim) VFE-evolved phi

    # Store attention info for RG metrics computation (detached)
    metrics['attention_info'] = {
        'beta': beta.detach(),
        'kl': kl.detach(),
        'mu': mu_q.detach(),
        'sigma': sigma_q.detach() if sigma_q is not None else None,
    }

    return total_loss, metrics
