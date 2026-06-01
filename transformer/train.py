"""
Training Loss Functions for Gauge-Theoretic Transformer
=======================================================

Core loss computation used by training scripts.

Exports:
    - compute_free_energy_loss(): M-step training loss (CE + KL regularizers)
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
from typing import Dict, Tuple

from transformer.core.vfe_utils import KL_CEIL_BASE, KL_CEIL_SCALE

# Import attention computation for gamma term
from transformer.core.attention import compute_attention_weights


# =============================================================================
# Gaussian KL Divergence (Proper Implementation)
# =============================================================================
# NOTE: parallel implementation of `transformer.core.kl_computation._kl_kernel_*`.
# The canonical version (with Rényi-alpha, SPD eigfloor, Cholesky stability) is
# in `transformer.core.kl_computation.compute_kl_matrix`. This local version is
# tied to the legacy `train.py` loss-composition contract; any numerical drift
# between the two should be resolved in favour of `kl_computation`.

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
                # Total Cholesky failure: return NaN KL to signal corruption
                # (do NOT return KL≈0 via identity fallback — that silently drops the term)
                _nr("chol_fail")
                return torch.full(mu_q.shape[:-1], float('nan'), device=device, dtype=dtype)

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
    kl_ceil = max(KL_CEIL_BASE, KL_CEIL_SCALE * K)
    kl = torch.clamp(kl, min=0.0, max=kl_ceil)
    # NaN/Inf safety: replace any residual numerical failures
    bad_mask = torch.isnan(kl) | torch.isinf(kl)
    if bad_mask.any():
        _nr("nan_replace")
    # NaN → kl_ceil (repulsive): a NaN KL should be treated as maximum distance,
    # not zero distance. Zero would make the pair maximally attractive in softmax.
    kl = kl.nan_to_num(nan=kl_ceil, posinf=kl_ceil, neginf=0.0)
    return kl


def _get_sigma_target(
    model: torch.nn.Module,
    sigma_s: torch.Tensor,
) -> torch.Tensor:
    r"""Get frozen sigma hyperprior target Σ_h, broadcast to match sigma_s shape.

    Uses the frozen ``sigma_target`` buffer from GaugeTokenEmbedding (registered
    at model creation with initial sigma values). Falls back to broadened
    centroid ``2 \times \mathrm{mean}(\sigma_s)`` if no buffer is available
    (backward compatibility with old checkpoints).

    Handles both diagonal (B, N, K) and full covariance (B, N, K, K).

    Returns:
        sigma_h: Same shape as sigma_s, detached (fixed target).
    """
    # Try to get frozen sigma_target from embedding.
    # Check PriorBank FIRST: when active, it owns the sigma parameters
    # (token_embed.sigma_target also exists but is stale/unused).
    sigma_target = None
    if hasattr(model, 'prior_bank') and model.prior_bank is not None:
        if hasattr(model.prior_bank, 'sigma_target'):
            sigma_target = model.prior_bank.sigma_target  # (K,)
    if sigma_target is None and hasattr(model, 'token_embed') and hasattr(model.token_embed, 'sigma_target'):
        sigma_target = model.token_embed.sigma_target  # (K,)

    if sigma_target is not None:
        # sigma_target is (K,) — expand to match sigma_s shape
        if sigma_s.dim() == 3:
            # Diagonal: sigma_s is (B, N, K)
            sigma_h = sigma_target.unsqueeze(0).unsqueeze(0).expand_as(sigma_s)
        elif sigma_s.dim() == 4:
            # Full covariance: sigma_s is (B, N, K, K)
            # Build diag matrix from sigma_target
            K = sigma_target.shape[0]
            sigma_h_mat = torch.diag(sigma_target)  # (K, K)
            sigma_h = sigma_h_mat.unsqueeze(0).unsqueeze(0).expand_as(sigma_s)
        else:
            sigma_h = sigma_target.expand_as(sigma_s)
        return sigma_h.detach()
    else:
        # Fallback: broadened centroid (old behavior, backward compatible).
        # Cache on first call to avoid a moving hyper-prior target — the
        # "frozen at init" contract requires sigma_h to be constant.
        if not hasattr(_get_sigma_target, '_cached_sigma_h') or _get_sigma_target._cached_sigma_h is None:
            import logging
            logging.warning(
                "_get_sigma_target: sigma_target buffer not found on model. "
                "Falling back to broadened centroid (computed once and cached). "
                "This happens with old checkpoints missing the sigma_target buffer."
            )
            _get_sigma_target._cached_sigma_h = sigma_s.mean(dim=1, keepdim=True).detach() * 2.0
        return _get_sigma_target._cached_sigma_h.expand_as(sigma_s)


# =============================================================================
# Free Energy Loss Computation (ATTENTION-WEIGHTED)
# =============================================================================

def compute_free_energy_loss(
    model,
    token_ids: torch.Tensor,
    targets: torch.Tensor,
    M_alpha: float = 0.0,         # Self-coupling: KL(q_i || p_i) — beliefs to priors
    M_beta: float = 1.0,          # Belief coupling weight
    lambda_gamma: float = 0.0,    # Model coupling weight
    kappa_gamma: float = 1.0,     # Temperature for γ_ij coupling weights
    lambda_hyper: float = 0.0,    # Hyper-prior: KL(s_i || h) — models to centroid
    pad_token_id: int = -100,     # Token ID to ignore in loss (padding)
    mass_phi: float = 0.0,        # Gauge prior weight: (mass_φ/2) Σ_i ||φ_i||²
    omega_det_penalty: float = 0.0,  # λ · mean_{i,h}(log|det Ω_{i,h}|)² regularizer (gauge_param='omega' only)
    detach_beta_m_step: bool = True,  # True = correct EM (detach β). False = old behavior
    include_attention_entropy: bool = True,  # Add κ·Σβ·log(β/π) to F (manuscript eq:free_energy_functional_final). (grad through softmax)
    normalize_ce_by_dim: bool = False,  # Divide CE by sqrt(K) to match VFE dim_scale
    ce_label_smoothing: float = 0.0,  # Label smoothing on loss-path CE only; ce_loss_raw stays un-smoothed for PPL
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
        M_alpha: Weight for self-coupling KL(q||p) (default: 0.0)
        M_beta: Weight for belief coupling (default: 1.0)
        lambda_gamma: Weight for model coupling (default: 0.0)
        kappa_gamma: Temperature for γ_ij model coupling weights (default: 1.0)
        lambda_hyper: Weight for hyper-prior KL(s_i||h) (default: 0.0)
        mass_phi: Gauge prior weight: (mass_φ/2) Σ_i ||φ_i||² (default: 0.0)
        pad_token_id: Token ID for padding (ignored in loss). Default -100.

    Returns:
        total_loss: Scalar loss for backprop
        metrics: Dict with loss components
    """
    # =================================================================
    # Forward pass with attention weights and KL matrices
    # =================================================================
    # The E-step does not see target tokens — CE below is the outer M-step
    # observation likelihood term of the free energy.
    logits, attn_info = model.forward_with_attention(token_ids, targets=targets)

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
    ce_loss_raw = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),  # (B*N, V)
        targets.reshape(-1),                   # (B*N,)
        reduction='mean',
        ignore_index=pad_token_id,
    )
    # Label smoothing applied to the LOSS path only; ce_loss_raw stays un-smoothed
    # for PPL/BPC metrics so reported PPL stays directly comparable across runs.
    if ce_label_smoothing > 0.0:
        ce_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction='mean',
            ignore_index=pad_token_id,
            label_smoothing=ce_label_smoothing,
        )
    else:
        ce_loss = ce_loss_raw  # zero-overhead bit-identical fast path when smoothing is off
    # VFE terms divide by sqrt(K) (dim_scale) but CE does not by default.
    # When normalize_ce_by_dim=True, apply the same normalization so relative
    # VFE-vs-CE weighting is independent of embed_dim.
    if normalize_ce_by_dim:
        embed_K = model.config.get('embed_dim', 128) if hasattr(model, 'config') else 128
        ce_loss = ce_loss / (embed_K ** 0.5)

    # =================================================================
    # 2. Belief Coupling: λ_β · Σ_ij β_ij · KL(q_i || Ω_ij q_j)
    # =================================================================
    if M_beta > 0.0:
        if beta is None or kl is None:
            # Pure-VFE block (attention sublayer removed 2026-06-01): the FFN
            # E-step surfaces its own β via _last_beta but does NOT retain the
            # per-iteration KL matrix (kl is None). The M-step belief-coupling
            # loss needs both β and KL, so it has no well-defined surface here.
            # The VFE E-step already handles belief coupling internally; use
            # M_beta=0. Guard on BOTH so a surfaced β with kl=None can't crash.
            import warnings
            warnings.warn(
                "M_beta > 0 but the attention KL matrix is unavailable (pure-VFE "
                "block). Setting belief_align_loss=0. Use M_beta=0.",
                stacklevel=2,
            )
            belief_align_loss = torch.tensor(0.0, device=ce_loss.device)
            # belief_entropy_loss is otherwise set only in the β-available branch;
            # set it here too so the total-loss sum (train.py downstream) cannot
            # hit an UnboundLocalError when β/KL are unavailable.
            belief_entropy_loss = torch.tensor(0.0, device=ce_loss.device)
        else:
            # Detach β: correct EM formulation. In the M-step, β is held fixed at
            # its E-step value. At convergence ∂F/∂β = 0 (β* is optimal), so the
            # β-gradient contribution vanishes by the envelope theorem. Detaching
            # is the finite-iteration approximation of this.
            #
            # Note: the non-detached softmax gradient (β_ij/κ)(E_β[KL] - KL_ij)
            # actually sharpens, not uniformizes. But it's not part of the M-step.
            beta_final = beta[-1].detach() if detach_beta_m_step else beta[-1]
            kl_final = kl[-1]               # (B, n_heads, N, N) — gradient flows through
            weighted_kl = beta_final * kl_final
            belief_align_loss = weighted_kl.sum(dim=(-2, -1)).mean()
            K = mu_q.shape[-1]
            dim_scale = math.sqrt(max(K, 1))
            belief_align_loss = M_beta * belief_align_loss / dim_scale
            # Manuscript Eq. eq:free_energy_functional_final alignment-block entropy:
            #   F_H = τ · Σ_ij β_ij log(β_ij / π_ij),   τ_eff = κ · √K (softmax temp).
            # belief_align_loss above is normalized by /dim_scale (1/√K); for τ-consistent
            # entropy in the SAME normalized space, F_H_normalized = κ · (β·log β).sum().
            # Uniform prior π = 1/N (constant log N dropped). β.detach (envelope theorem
            # at softmax fixed point → zero (μ,Σ,φ)-grad). Scalar κ here — κ-gradient flow
            # for learnable κ lives in the E-step path (variational_ffn.py per-head).
            if include_attention_entropy:
                beta_safe = beta_final.clamp(min=1e-30)
                entropy_per_row = (beta_safe * beta_safe.log()).sum(dim=(-2, -1)).mean()
                kappa_attn_scalar = 1.0
                try:
                    _ffn0 = model.blocks[0].ffn
                    _k = getattr(_ffn0, 'kappa', None)
                    if _k is not None:
                        kappa_attn_scalar = (
                            float(_k.detach().item()) if hasattr(_k, 'detach')
                            else float(_k)
                        )
                except (AttributeError, IndexError, TypeError):
                    pass
                # τ_eff · entropy / dim_scale = (κ·√K) · entropy / √K = κ · entropy
                belief_entropy_loss = M_beta * kappa_attn_scalar * entropy_per_row
            else:
                belief_entropy_loss = torch.tensor(0.0, device=ce_loss.device)
    else:
        belief_align_loss = torch.tensor(0.0, device=ce_loss.device)
        belief_entropy_loss = torch.tensor(0.0, device=ce_loss.device)

    # =================================================================
    # 3. Self-Coupling: α_i · KL(q_i || p_i) — beliefs toward PRIORS
    # =================================================================
    # This pulls evolved beliefs (fast) back toward priors (derived from
    # models). This is NOT KL(q||s) — it's KL(q||p) where p = f(s).
    # Currently p = s, so they're numerically identical, but the
    # conceptual distinction matters for the hierarchy.
    #
    # When learnable_alpha is enabled, α_i = c0/(b0 + KL(q||p)) is
    # per-agent, per-dimension (from E-step). We use it here in the
    # M-step loss for consistency: same α weights the self-coupling
    # in both E-step gradient and M-step loss.
    # =================================================================
    kl_per_agent = None  # Computed below when M_alpha > 0; reused in diagnostics
    if M_alpha > 0.0:
        K = mu_q.shape[-1]
        # Detach E-step covariances: backprop through the E-step sigma evolution
        # produces NaN from numerically unstable second derivatives (matrix_exp
        # backward through Omega @ Sigma @ Omega.T). Sigma learning is handled by
        # the E-step dynamics — same EM principle as beta/gamma.
        # mu_q keeps gradient (flows through the IFT-scaled path to mu_embed).
        sigma_q_for_kl = sigma_q.detach() if sigma_q is not None else None
        # sigma_p kept LIVE: the M-step KL(q||p) is the correct gradient source
        # for sigma_p when M_alpha > 0. The positive feedback concern (smaller σ_p →
        # larger ∂KL/∂σ_p) applies to the E-step's natural gradient (where 1/σ_p
        # amplifies), but in the M-step with AdamW the gradient is bounded by
        # adaptive learning rates and gradient clipping. Detaching here starves
        # sigma_p of gradient when the E-step also detaches it (variational_ffn.py)
        # and sigma_ce_scale dampens the decode path.
        kl_per_agent = gaussian_kl_divergence(
            mu_q=mu_q,
            sigma_q=sigma_q_for_kl,
            mu_p=mu_p,
            sigma_p=sigma_p,
        )  # (B, N)
        dim_scale = math.sqrt(max(K, 1))

        # Use adaptive α_i from E-step if available (learnable_alpha mode)
        alpha_i = attn_info.get('alpha_i', None)
        if alpha_i is not None:
            # alpha_i is (B, N, K) per-dimension; kl_per_agent is (B, N) scalar.
            # Use mean over dimensions for scalar KL weighting, then scale by
            # M_alpha as an overall loss weight.  This gives consistent
            # semantics with the fixed-α branch below: setting M_alpha=0
            # disables the term in both modes, and M_alpha acts as a
            # multiplicative scale on top of the learnable per-position α.
            alpha_scalar = alpha_i.mean(dim=-1)  # (B, N)
            self_consistency_loss = M_alpha * (alpha_scalar * kl_per_agent).mean() / dim_scale
        else:
            self_consistency_loss = M_alpha * kl_per_agent.mean() / dim_scale
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
            return_kl=True,
            diagonal_covariance=diagonal_cov,
        )

        # Detach gamma: same EM justification as beta. γ is an E-step
        # quantity, held fixed in the M-step.
        weighted_kl_model = gamma.detach() * kl_model  # (B, N, N)
        K = mu_s.shape[-1]
        dim_scale = math.sqrt(max(K, 1))
        model_align_loss = lambda_gamma * weighted_kl_model.sum(dim=(-2, -1)).mean() / dim_scale
    else:
        model_align_loss = torch.tensor(0.0, device=ce_loss.device)

    # =================================================================
    # 5. Hyper-Prior: λ_h · Σ_i KL(s_i || h) — models toward centroid
    # =================================================================
    # h = (μ_h, Σ_h) is the Level 3 hyperprior target.
    #   μ_h = centroid of all models (detached, anti-memorization)
    #   Σ_h = frozen initial sigma (fixed anchor, prevents collective drift)
    #
    # Previous bug: sigma_s was .detach()'d in the KL, giving zero gradient
    # to sigma_embed. Now sigma_s flows through, providing bidirectional
    # gradient: pulls sigma toward Σ_h if it inflates OR deflates.
    # =================================================================
    if lambda_hyper > 0.0:
        batch_size, num_agents, K = mu_s.shape
        dim_scale = math.sqrt(max(K, 1))

        # μ centroid: mean of all models (detach: treat as fixed target)
        mu_h = mu_s.mean(dim=1, keepdim=True).detach()  # (B, 1, K)

        if sigma_s is not None:
            # Σ_h: frozen initial sigma from embedding buffer (fixed anchor).
            # This replaces the old moving target (2×mean(sigma_s)) which
            # inflated together with sigma_s, providing no downward pressure.
            sigma_target = _get_sigma_target(model, sigma_s)  # (B, N, K) or (B, N, K, K)
            sigma_h = sigma_target
        else:
            sigma_h = None

        # Expand centroid to match all agents
        mu_h_expanded = mu_h.expand_as(mu_s)
        sigma_h_expanded = sigma_h  # already (B, N, ...) from _get_sigma_target

        # KL(s_i || h) for each agent position
        # sigma_s NOT detached: gradient flows to sigma_embed (bidirectional)
        kl_hyper = gaussian_kl_divergence(
            mu_q=mu_s,
            sigma_q=sigma_s,
            mu_p=mu_h_expanded,
            sigma_p=sigma_h_expanded,
        )  # (B, N)

        hyper_prior_loss = lambda_hyper * kl_hyper.mean() / dim_scale
    else:
        hyper_prior_loss = torch.tensor(0.0, device=ce_loss.device)

    # =================================================================
    # 6. Gauge Prior: (α_φ/2) Σ_i ||φ_i||² — mass term for gauge field
    # =================================================================
    if mass_phi > 0.0:
        K = mu_q.shape[-1]
        dim_scale = math.sqrt(max(K, 1))
        # phi_s is the model gauge frame from embeddings (B, N, n_gen)
        phi_norm_sq = (phi_s ** 2).sum(dim=-1).mean()  # Mean over batch and tokens
        gauge_prior_loss = (mass_phi / 2.0) * phi_norm_sq / dim_scale
    else:
        gauge_prior_loss = torch.tensor(0.0, device=ce_loss.device)

    # =================================================================
    # 8. Holonomy Penalty: λ_H · E[‖C_ijk − I‖²_F]
    # =================================================================
    # Non-flat transport regulariser.  When any block has non_flat_transport
    # enabled with holonomy_penalty > 0, compute the mean squared connection
    # holonomy across sampled triples and add to the loss.  Previously this
    # value was stored per-block but never read by the loss function — users
    # who set holonomy_penalty > 0 got zero regularisation.
    #
    # The per-block attribute `_last_exp_delta` is populated by
    # GaugeTransformerBlock.forward (blocks.py:378) and
    # GaugeTransformerLM.forward_with_attention (model.py:1090) when
    # non_flat_transport=True.  We compute the penalty on each block that
    # has a non-None exp_delta and a positive weight, and sum across blocks.
    holonomy_loss = torch.tensor(0.0, device=ce_loss.device)
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'blocks'):
        from transformer.analysis.holonomy import holonomy_penalty_loss as _h_pen
        for _block in model.transformer.blocks:
            _weight = getattr(_block, 'holonomy_penalty', 0.0)
            _exp_delta = getattr(_block, '_last_exp_delta', None)
            if _weight > 0.0 and _exp_delta is not None:
                # Sample-based average of ‖C_ijk − I‖²_F; gradient flows back
                # to the GaugeConnection weights via _exp_delta.  Clear the
                # attribute after use so the next forward pass starts fresh.
                holonomy_loss = holonomy_loss + _weight * _h_pen(_exp_delta)
                _block._last_exp_delta = None

    # =================================================================
    # Omega determinant regularizer (gauge_param='omega' only)
    # =================================================================
    # Direct-Omega parameterization has no exp-map guarantee that Ω ∈ GL(K).
    # Without regularization, Ω can drift toward singular (det→0) or blow up
    # (det→∞), breaking the sandwich transport and the cocycle identity.
    # Penalize (log|det Ω_h|)² per block per token. The minimum is |det|=1,
    # so this drives |det Ω_h| toward unity while allowing det<0 (reflections).
    omega_det_loss = torch.tensor(0.0, device=ce_loss.device)
    _omega_init = attn_info.get('omega_initial', None) if isinstance(attn_info, dict) else None
    if omega_det_penalty > 0.0 and _omega_init is not None:
        # Block structure lives on model.omega_head_dims (set by
        # GaugeTransformerLM.__init__ when gauge_param='omega'). Falling back
        # to a single full-K block would compute (sum_h log|det Ω_h|)² — a
        # looser penalty that lets positive- and negative-log|det| blocks
        # cancel. Prefer per-block Σ_h (log|det Ω_h|)² so each block is
        # regularized independently.
        _head_dims = getattr(model, 'omega_head_dims', None)
        if _head_dims is None:
            _head_dims = [_omega_init.shape[-1]]
        _det_acc = torch.tensor(0.0, device=ce_loss.device, dtype=_omega_init.dtype)
        _block_start = 0
        for _d in _head_dims:
            _block_end = _block_start + _d
            _omega_h = _omega_init[:, :,
                                   _block_start:_block_end,
                                   _block_start:_block_end]
            _, _logabsdet = torch.linalg.slogdet(_omega_h)  # (B, N)
            _det_acc = _det_acc + (_logabsdet ** 2).mean()
            _block_start = _block_end
        omega_det_loss = omega_det_penalty * _det_acc

    # =================================================================
    # Total Training Loss (CE + optional auxiliary VFE regularizers)
    # =================================================================
    total_loss = (ce_loss + belief_align_loss + belief_entropy_loss + self_consistency_loss
                  + model_align_loss + hyper_prior_loss + gauge_prior_loss
                  + holonomy_loss + omega_det_loss)

    # Compute attention metrics outside the computation graph
    # When skip_attention=True, beta/kl from the attention sublayer are None.
    # Fall back to the VFE E-step's internally computed beta (stored on the FFN).
    with torch.no_grad():
        if beta is not None and beta[-1] is not None:
            beta_avg = beta[-1].mean(dim=1)  # (B, N, N) - last layer, average over heads
            _beta_mean = beta.mean().item()
            _kl_mean = kl.mean().item() if kl is not None else 0.0
        else:
            # Retrieve beta from VFE E-step (last block's FFN stores it)
            _vfe_beta = None
            if hasattr(model, 'transformer'):
                last_ffn = model.transformer.blocks[-1].ffn
                _bh = getattr(last_ffn, '_last_beta_history', None)
                if _bh and len(_bh) > 0:
                    _vfe_beta = _bh[-1]
            if _vfe_beta is not None:
                beta_avg = _vfe_beta.mean(dim=1) if _vfe_beta.dim() == 4 else _vfe_beta
                _beta_mean = _vfe_beta.mean().item()
            else:
                # No beta available at all — use uniform
                N = logits.shape[1]
                beta_avg = torch.ones(1, N, N, device=logits.device) / N
                _beta_mean = 1.0 / N
            _kl_mean = 0.0

        beta_safe = beta_avg.clamp(min=1e-10)
        attn_entropy = -(beta_safe * beta_safe.log()).sum(dim=-1).mean()
        attn_concentration = beta_avg.max(dim=-1)[0].mean()

    # Metrics
    metrics = {
        'loss/total': total_loss.item(),
        'loss/ce': ce_loss.item(),
        'loss/ce_raw': ce_loss_raw.item(),
        'loss/belief_align': belief_align_loss.item(),
        'loss/belief_entropy': belief_entropy_loss.item(),
        'loss/self_consistency': self_consistency_loss.item() if M_alpha > 0 else 0.0,
        'loss/model_coupling': model_align_loss.item() if lambda_gamma > 0 else 0.0,
        'loss/hyper_prior': hyper_prior_loss.item() if lambda_hyper > 0 else 0.0,
        'loss/gauge_prior': gauge_prior_loss.item() if mass_phi > 0 else 0.0,
        'loss/holonomy': holonomy_loss.item() if isinstance(holonomy_loss, torch.Tensor) else float(holonomy_loss),
        'loss/omega_det': omega_det_loss.item() if omega_det_penalty > 0 else 0.0,
        'attention/beta_mean': _beta_mean,
        'attention/kl_mean': _kl_mean,
        'attention/entropy': attn_entropy.item(),
        'attention/concentration': attn_concentration.item(),
    }

    # Bayesian alpha diagnostics
    _blocks = getattr(model, 'transformer', None)
    _blocks = _blocks.blocks if _blocks is not None else getattr(model, 'blocks', [])
    with torch.no_grad():
        for block in _blocks:
            vffn = block.ffn if hasattr(block.ffn, 'learnable_alpha') else getattr(block.ffn, 'variational_ffn', None)
            if vffn is not None and vffn.learnable_alpha and mu_q is not None and mu_p is not None:
                import torch.nn.functional as _F
                alpha_vals = vffn.get_bayesian_alpha(mu_q, mu_p, sigma_p, sigma_q)
                c0 = _F.softplus(vffn.raw_c0)
                b0 = _F.softplus(vffn.raw_b0)
                # Reuse kl_per_agent if available (computed when M_alpha > 0),
                # otherwise compute for diagnostics only.
                if kl_per_agent is not None:
                    kl_qp = kl_per_agent.detach()
                else:
                    kl_qp = gaussian_kl_divergence(
                        mu_q=mu_q, sigma_q=sigma_q,
                        mu_p=mu_p, sigma_p=sigma_p,
                    ).detach()  # (B, N)
                # Mahalanobis term only (cheap: just the quadratic part)
                delta = mu_q - mu_p
                if sigma_p.dim() == 3:
                    sp_safe = sigma_p.clamp(min=1e-6)
                    mahal_t = (delta ** 2 / sp_safe).sum(dim=-1)
                else:
                    K_dim = mu_q.shape[-1]
                    sigma_p_metric = sigma_p + 1e-6 * torch.eye(K_dim, device=mu_q.device)
                    try:
                        sp_inv = torch.linalg.inv(sigma_p_metric)
                    except (torch.linalg.LinAlgError, RuntimeError):
                        sp_inv = torch.linalg.pinv(sigma_p_metric)
                    mahal_t = torch.einsum('bni,bnij,bnj->bn', delta, sp_inv, delta)
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

    # Per-head learnable kappa diagnostics (first layer only)
    for block in _blocks:
        vffn = block.ffn if hasattr(block.ffn, 'learnable_head_kappa') else None
        if vffn is None or not vffn.learnable_head_kappa:
            continue
        # Resolve kappa: when shared with attention (blocks.py kappa sharing),
        # ffn.log_kappa_per_head is None and the actual parameter lives on
        # the attention module via _kappa_attn_ref.
        log_kappa = vffn.log_kappa_per_head
        k0 = getattr(vffn, '_kappa_init', None)
        if log_kappa is None:
            attn_ref = getattr(vffn, '_kappa_attn_ref', None)
            if attn_ref is not None:
                log_kappa = getattr(attn_ref, 'log_kappa_per_head', None)
                k0 = getattr(attn_ref, '_kappa_init', k0)
        if log_kappa is not None:
            with torch.no_grad():
                kappa_vals = torch.exp(log_kappa)  # (n_heads,)
                if k0 is not None:
                    kappa_vals = kappa_vals.clamp(
                        min=0.5 * k0, max=1.5 * k0
                    )
                metrics['kappa/per_head_mean'] = kappa_vals.mean().item()
                metrics['kappa/per_head_std'] = kappa_vals.std().item()
                metrics['kappa/per_head_min'] = kappa_vals.min().item()
                metrics['kappa/per_head_max'] = kappa_vals.max().item()
                # Log each head individually
                for h_idx, kv in enumerate(kappa_vals):
                    metrics[f'kappa/head_{h_idx}'] = kv.item()
            break  # Only first layer

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

    # =================================================================
    # VFE Gradient Decomposition & Dynamics Metrics
    # =================================================================
    # Surface VFE debug dict from E-step (gradient component breakdown)
    vfe_debug = attn_info.get('vfe_debug')
    if vfe_debug is not None:
        for key, val in vfe_debug.items():
            if isinstance(val, (int, float)):
                metrics[f'vfe/{key}'] = val

    # Surface transport operator and covariance health metrics
    transport_m = attn_info.get('transport_metrics', {})
    for key, val in transport_m.items():
        metrics[f'transport/{key}'] = val
    covariance_m = attn_info.get('covariance_metrics', {})
    for key, val in covariance_m.items():
        metrics[f'cov/{key}'] = val

    return total_loss, metrics
