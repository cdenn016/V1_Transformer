"""
Analytic Gaussian operations for the Pure VFE Transformer.

All KL divergences and their exact gradients for full covariance Σ ∈ SPD(K).
No autograd anywhere. CUDA-accelerated pairwise operations when available.

Mathematical reference:
  - Main paper Eq. 21 (mean gradient), Eq. 24 (softmax correction)
  - Supplementary Eq. B.1 (covariance gradient), Appendix D (SPD retraction)
  - derivations/constant_gauge_kl_derivation.md (KL decomposition)
"""

import torch
import torch.nn.functional as F

from .cuda_ext import get_cuda_ext


# ---------------------------------------------------------------------------
# Safe linear algebra helpers
# ---------------------------------------------------------------------------

def safe_inverse(M, eps=1e-6):
    """Robust matrix inverse with conditioning check."""
    try:
        return torch.linalg.inv(M)
    except (torch.linalg.LinAlgError, RuntimeError):
        # Add regularization (RuntimeError can occur on CUDA for singular matrices)
        eye = torch.eye(M.shape[-1], device=M.device, dtype=M.dtype)
        return torch.linalg.inv(M + eps * eye.expand_as(M))


def safe_logdet(M):
    """Log-determinant via Cholesky for SPD matrices, slogdet fallback."""
    try:
        L = torch.linalg.cholesky(M)
        return 2.0 * torch.diagonal(L, dim1=-2, dim2=-1).log().sum(-1)
    except (torch.linalg.LinAlgError, RuntimeError):
        # RuntimeError can occur on CUDA for non-PD matrices
        sign, logabsdet = torch.linalg.slogdet(M)
        # For SPD matrices sign should be +1; clamp to 0 if negative (non-PD)
        return torch.where(sign > 0, logabsdet, torch.zeros_like(logabsdet))


def symmetrize(M):
    """Ensure matrix is symmetric: sym(M) = (M + Mᵀ)/2."""
    return 0.5 * (M + M.transpose(-2, -1))


# ---------------------------------------------------------------------------
# Core KL divergence (full covariance)
# ---------------------------------------------------------------------------

def kl_divergence(mu_p, Sigma_p, mu_q, Sigma_q):
    """
    KL(P || Q) for multivariate Gaussians with full covariance.

    P = N(mu_p, Sigma_p), Q = N(mu_q, Sigma_q).

    = ½[tr(Σ_q⁻¹ Σ_p) + (μ_q - μ_p)ᵀ Σ_q⁻¹ (μ_q - μ_p) - K + ln(det Σ_q / det Σ_p)]

    Args:
        mu_p, mu_q: [..., K]
        Sigma_p, Sigma_q: [..., K, K]
    Returns:
        kl: [...]
    """
    K = mu_p.shape[-1]
    Sigma_q_inv = safe_inverse(Sigma_q)

    # Trace term: tr(Σ_q⁻¹ Σ_p)
    trace_term = torch.diagonal(
        Sigma_q_inv @ Sigma_p, dim1=-2, dim2=-1
    ).sum(-1)

    # Mahalanobis term
    delta = mu_q - mu_p  # [..., K]
    mahal = torch.einsum('...i,...ij,...j->...', delta, Sigma_q_inv, delta)

    # Log-det term
    logdet_q = safe_logdet(Sigma_q)
    logdet_p = safe_logdet(Sigma_p)

    return 0.5 * (trace_term + mahal - K + logdet_q - logdet_p)


# ---------------------------------------------------------------------------
# Precomputation for efficient pairwise KL
# ---------------------------------------------------------------------------

def precompute_tokens(mu_h, Sigma_h, Omega, Sigma_h_inv=None):
    """
    Precompute per-token quantities for O(N² K²) pairwise KL.

    For each token i:
      ρ_i = Ω_i⁻¹ μ_i           (pulled-back mean)
      Q_i = Ω_i⁻¹ Σ_i Ω_i⁻ᵀ    (pulled-back covariance)
      P_i = Ω_iᵀ Σ_i⁻¹ Ω_i      (rotated precision)
      ν_i = Ω_i⁻¹ μ_i           (same as ρ — token is both query and key)
      ldc_q_i = 2 ln|det Ω_i| - ln det Σ_i
      ldc_k_i = -2 ln|det Ω_i| + ln det Σ_i

    Args:
        mu_h: [B, N, H, K_h]
        Sigma_h: [B, N, H, K_h, K_h]
        Omega: [B, N, H, K_h, K_h]
        Sigma_h_inv: optional precomputed inverse [B, N, H, K_h, K_h]

    Returns: dict of precomputed tensors, all [B, H, N, ...]
    """
    B, N, H, K_h = mu_h.shape

    # Transpose to [B, H, N, ...] for contiguous head-first layout
    mu = mu_h.permute(0, 2, 1, 3).contiguous()            # [B, H, N, K_h]
    Sig = Sigma_h.permute(0, 2, 1, 3, 4).contiguous()     # [B, H, N, K_h, K_h]
    Om = Omega.permute(0, 2, 1, 3, 4).contiguous()        # [B, H, N, K_h, K_h]

    # Inverse of Omega
    Om_inv = torch.linalg.inv(Om)                          # [B, H, N, K_h, K_h]

    # ρ = ν = Ω⁻¹ μ
    rho = torch.einsum('bhnpq,bhnq->bhnp', Om_inv, mu)     # [B, H, N, K_h]

    # Q = Ω⁻¹ Σ Ω⁻ᵀ
    temp = Om_inv @ Sig                                     # [B, H, N, K_h, K_h]
    Q = temp @ Om_inv.transpose(-2, -1)                     # [B, H, N, K_h, K_h]

    # Σ⁻¹
    if Sigma_h_inv is not None:
        Sig_inv = Sigma_h_inv.permute(0, 2, 1, 3, 4).contiguous()
    else:
        Sig_inv = safe_inverse(Sig)

    # P = Ωᵀ Σ⁻¹ Ω
    P = Om.transpose(-2, -1) @ Sig_inv @ Om                # [B, H, N, K_h, K_h]

    # Log-determinants
    _, ln_det_Om = torch.linalg.slogdet(Om)                # [B, H, N]
    ln_det_Sig = safe_logdet(Sig)                           # [B, H, N]

    ldc_q = 2.0 * ln_det_Om - ln_det_Sig                   # [B, H, N]
    ldc_k = -2.0 * ln_det_Om + ln_det_Sig                  # [B, H, N]

    return {
        'Q': Q, 'rho': rho, 'P': P, 'nu': rho.clone(),
        'ldc_q': ldc_q, 'ldc_k': ldc_k,
        'Omega_inv': Om_inv,
        'Sigma_inv': Sig_inv,
        'ln_det_Sigma': ln_det_Sig,
    }


# ---------------------------------------------------------------------------
# Pairwise KL computation (CUDA-accelerated)
# ---------------------------------------------------------------------------

def pairwise_kl(precomp, causal=True):
    """
    Compute KL(q_i || Ω_ij · q_j) for all pairs using precomputed quantities.

    KL_ij = ½[tr(P_j Q_i) + (ρ_i - ν_j)ᵀ P_j (ρ_i - ν_j) - K + ldc_q_i + ldc_k_j]

    Args:
        precomp: dict from precompute_tokens
        causal: if True, mask future tokens with +inf

    Returns:
        kl: [B, H, N, N] pairwise KL values
    """
    Q = precomp['Q']          # [B, H, N, K, K]
    rho = precomp['rho']      # [B, H, N, K]
    P = precomp['P']          # [B, H, N, K, K]
    nu = precomp['nu']        # [B, H, N, K]
    ldc_q = precomp['ldc_q']  # [B, H, N]
    ldc_k = precomp['ldc_k']  # [B, H, N]

    # Try CUDA kernel
    cuda_ext = get_cuda_ext()
    if cuda_ext is not None and Q.is_cuda:
        return cuda_ext.pairwise_kl(Q, rho, P, nu, ldc_q, ldc_k, causal)

    # Pure PyTorch fallback
    return _pairwise_kl_torch(Q, rho, P, nu, ldc_q, ldc_k, causal)


def _pairwise_kl_torch(Q, rho, P, nu, ldc_q, ldc_k, causal):
    """Pure PyTorch pairwise KL. [B, H, N, N]."""
    B, H, N, K = rho.shape

    # Trace term: tr(P_j Q_i) for all (i, j)
    # tr(P_j Q_i) = Σ_{a,b} P_j[a,b] Q_i[b,a] = vec(P_j) · vec(Q_iᵀ)
    P_flat = P.reshape(B, H, N, K * K)         # [B, H, N, K²]
    Qt_flat = Q.transpose(-2, -1).reshape(B, H, N, K * K)  # Q transposed, then flattened
    # Contract over flattened K² dim: result[b,h,j,i] = Σ_k P_flat[j,k] * Qt_flat[i,k] = tr(P_j Q_i)
    trace_term = torch.einsum('bhjk,bhik->bhij', P_flat, Qt_flat)  # [B, H, N_j, N_i]
    trace_term = trace_term.transpose(-2, -1)  # [B, H, N_i, N_j]

    # Mahalanobis term: (ρ_i - ν_j)ᵀ P_j (ρ_i - ν_j)
    # d_ij = ρ_i - ν_j: [B, H, N, 1, K] - [B, H, 1, N, K] = [B, H, N, N, K]
    d = rho.unsqueeze(3) - nu.unsqueeze(2)     # [B, H, N_i, N_j, K]

    # P_j @ d_ij: [B, H, 1, N_j, K, K] @ [B, H, N_i, N_j, K, 1]
    Pd = torch.einsum('bhjpq,bhijq->bhijp', P, d)  # [B, H, N_i, N_j, K]
    mahal = (d * Pd).sum(-1)                     # [B, H, N_i, N_j]

    # Log-det contributions
    logdet = ldc_q.unsqueeze(-1) + ldc_k.unsqueeze(-2)  # [B, H, N_i, N_j]

    kl = 0.5 * (trace_term + mahal - K + logdet)
    kl = kl.clamp(min=0.0)  # Numerical safety

    if causal:
        mask = torch.triu(torch.ones(N, N, device=kl.device, dtype=torch.bool), diagonal=1)
        kl.masked_fill_(mask.unsqueeze(0).unsqueeze(0), 1e9)

    return kl


# ---------------------------------------------------------------------------
# Attention weights from KL
# ---------------------------------------------------------------------------

def kl_attention(kl_ij, tau, causal=True):
    """
    β_ij = softmax_j(-KL_ij / τ)

    Args:
        kl_ij: [B, H, N, N] pairwise KL
        tau: temperature scalar
        causal: if True, entries already masked with large KL
    Returns:
        beta: [B, H, N, N] attention weights
    """
    logits = -kl_ij / tau
    return torch.softmax(logits, dim=-1)


# ---------------------------------------------------------------------------
# Analytic gradients of KL (per-pair)
# ---------------------------------------------------------------------------

def grad_kl_mu_i_perpair(precomp):
    """
    ∂KL(q_i || Ω_ij · q_j)/∂μ_i  for all pairs.

    = (Ω_ij Σ_j Ω_ijᵀ)⁻¹ (μ_i - Ω_ij μ_j) = Ω_i⁻ᵀ P_j (ρ_i - ν_j)

    Returns: [B, H, N_i, N_j, K]
    """
    rho = precomp['rho']           # [B, H, N, K]
    nu = precomp['nu']             # [B, H, N, K]
    P = precomp['P']               # [B, H, N, K, K]
    Om_inv = precomp['Omega_inv']  # [B, H, N, K, K]

    # d_ij = ρ_i - ν_j
    d = rho.unsqueeze(3) - nu.unsqueeze(2)  # [B, H, N_i, N_j, K]

    # P_j @ d_ij in pulled-back frame
    Pd = torch.einsum('bhjpq,bhijq->bhijp', P, d)  # [B, H, N_i, N_j, K]

    # Push forward: Ω_i⁻ᵀ @ Pd (per query i, broadcast over j)
    Om_invT = Om_inv.transpose(-2, -1)  # [B, H, N, K, K]
    grad = torch.einsum('bhipq,bhijq->bhijp', Om_invT, Pd)  # [B, H, N_i, N_j, K]

    return grad


# ---------------------------------------------------------------------------
# Aggregated VFE gradients (attention-weighted sums)
# ---------------------------------------------------------------------------

def vfe_grad_mu_alignment(precomp, beta, kl_ij, tau):
    """
    Alignment contribution to ∂F/∂μ_i (Eq. 21 alignment terms + Eq. 24 correction).

    = Σ_j β_ij [1 + (E_β[KL] - KL_ij)/τ] · (Ω_i⁻ᵀ P_j (ρ_i - ν_j))

    Args:
        precomp: dict from precompute_tokens
        beta: [B, H, N, N] attention weights
        kl_ij: [B, H, N, N] pairwise KL
        tau: temperature

    Returns: [B, H, N, K] gradient
    """
    # Try CUDA kernel
    P = precomp['P']
    rho = precomp['rho']
    nu = precomp['nu']
    Om_inv = precomp['Omega_inv']
    Om_invT = Om_inv.transpose(-2, -1)

    cuda_ext = get_cuda_ext()
    if cuda_ext is not None and P.is_cuda:
        return cuda_ext.grad_mu_alignment(P, rho, nu, Om_invT, beta, kl_ij, tau)

    # PyTorch fallback
    B, H, N, K = rho.shape

    # Per-pair gradient in original coords
    g_ij = grad_kl_mu_i_perpair(precomp)  # [B, H, N_i, N_j, K]

    # E_β[KL] = Σ_j β_ij KL_ij
    e_kl = (beta * kl_ij).sum(-1, keepdim=True)  # [B, H, N, 1]

    # Softmax-corrected weights: w_ij = β_ij [1 + (E_β[KL] - KL_ij)/τ]
    w = beta * (1.0 + (e_kl - kl_ij) / tau)  # [B, H, N, N]

    # Aggregate: Σ_j w_ij g_ij
    grad = torch.einsum('bhij,bhijk->bhik', w, g_ij)  # [B, H, N, K]

    return grad


def vfe_grad_Sigma_alignment(precomp, beta):
    """
    Alignment contribution to ∂F/∂Σ_i (Eq. B.1, ignoring softmax correction).

    ∂F_align/∂Σ_i = ½[Σ_j β_ij (Ω_i⁻ᵀ P_j Ω_i⁻¹) - Σ_i⁻¹]

    Returns the attention-weighted transported precision: [B, H, N, K, K]
    (The -Σ_i⁻¹ part and the ½ factor are applied in the caller.)
    """
    P = precomp['P']
    Om_inv = precomp['Omega_inv']

    cuda_ext = get_cuda_ext()
    if cuda_ext is not None and P.is_cuda:
        return cuda_ext.grad_sigma_alignment(P, Om_inv, beta)

    # PyTorch fallback: Σ_j β_ij · Ω_i⁻ᵀ P_j Ω_i⁻¹
    B, H, N, K, _ = P.shape

    # Weighted sum of P_j: [B, H, N_i, K, K] = Σ_j β_ij P_j
    wP = torch.einsum('bhij,bhjpq->bhipq', beta, P)  # [B, H, N, K, K]

    # Transform: Ω_i⁻ᵀ wP Ω_i⁻¹
    Om_invT = Om_inv.transpose(-2, -1)
    result = Om_invT @ wP @ Om_inv  # [B, H, N, K, K]

    return result


# ---------------------------------------------------------------------------
# VFE gradient for prior terms
# ---------------------------------------------------------------------------

def vfe_grad_mu_prior(mu, Sigma, prior_mu, prior_Sigma, alpha):
    """
    Prior contribution to ∂F/∂μ_i.

    = α_i · Σ_prior⁻¹ (μ_i - μ_prior)

    Args:
        mu: [B, N, K]
        Sigma: [B, N, K, K]
        prior_mu: [B, N, K]
        prior_Sigma: [B, N, K, K]
        alpha: [B, N] state-dependent precision

    Returns: [B, N, K]
    """
    prior_Sigma_inv = safe_inverse(prior_Sigma)
    delta = mu - prior_mu
    grad = torch.einsum('bnij,bnj->bni', prior_Sigma_inv, delta)
    return alpha.unsqueeze(-1) * grad


def vfe_grad_Sigma_prior(Sigma, prior_Sigma, alpha):
    """
    Prior contribution to ∂F/∂Σ_i.

    = ½ α_i · (Σ_prior⁻¹ - Σ_i⁻¹)

    Returns the prior precision: [B, N, K, K] (α and ½ applied by caller).
    """
    return safe_inverse(prior_Sigma)


# ---------------------------------------------------------------------------
# Natural gradient on SPD manifold
# ---------------------------------------------------------------------------

def natural_grad_sigma(grad_Sigma, Sigma):
    """
    Fisher-Rao natural gradient on SPD(K).

    ΔΣ = -2 Σ sym(∂F/∂Σ) Σ

    (Manuscript Appendix D, Eq. D.2)

    Args:
        grad_Sigma: [..., K, K] Euclidean gradient ∂F/∂Σ
        Sigma: [..., K, K] current covariance

    Returns: [..., K, K] natural gradient direction
    """
    sym_grad = symmetrize(grad_Sigma)
    return -2.0 * Sigma @ sym_grad @ Sigma


def retract_spd(Sigma, nat_grad, step_size, eps_min=1e-4, kappa_max=1e4, exp_clip=50.0):
    """
    Affine-invariant exponential map retraction on SPD(K).

    Algorithm (Appendix D):
      1. Eigendecompose Σ = V Λ Vᵀ
      2. Whiten: B = Λ⁻¹/² Vᵀ (nat_grad) V Λ⁻¹/²
      3. Clip ||B||_F (trust region)
      4. Diagonalize B = U Λ_B Uᵀ
      5. Retract: Σ_new = V Λ¹/² U exp(τ·Λ_B) Uᵀ Λ¹/² Vᵀ
      6. Enforce spectral floor and condition cap

    Args:
        Sigma: [..., K, K] current SPD matrix
        nat_grad: [..., K, K] natural gradient direction (should be symmetric)
        step_size: scalar step size τ

    Returns: [..., K, K] updated SPD matrix
    """
    nat_grad = symmetrize(nat_grad)

    # 1. Eigendecompose Σ
    Lambda, V = torch.linalg.eigh(Sigma)  # Lambda: [..., K], V: [..., K, K]
    Lambda = Lambda.clamp(min=eps_min)

    # 2. Whiten: B = Λ⁻¹/² Vᵀ δ V Λ⁻¹/²
    Lambda_inv_sqrt = 1.0 / Lambda.sqrt()
    # Vᵀ @ nat_grad @ V
    VtdV = V.transpose(-2, -1) @ nat_grad @ V
    # Scale: diag(Λ⁻¹/²) @ VtdV @ diag(Λ⁻¹/²)
    B = Lambda_inv_sqrt.unsqueeze(-1) * VtdV * Lambda_inv_sqrt.unsqueeze(-2)
    B = symmetrize(B)

    # 3. Diagonalize B
    Lambda_B, U = torch.linalg.eigh(B)

    # 4. Clip eigenvalue exponents
    scaled = step_size * Lambda_B
    scaled = scaled.clamp(-exp_clip, exp_clip)

    # 5. Retract
    exp_Lambda_B = torch.exp(scaled)
    # Σ_new = V Λ¹/² U diag(exp(τ·λ_B)) Uᵀ Λ¹/² Vᵀ
    Lambda_sqrt = Lambda.sqrt()
    # VΛ¹/² U
    VLU = V * Lambda_sqrt.unsqueeze(-2) @ U  # [..., K, K]

    Sigma_new = VLU * exp_Lambda_B.unsqueeze(-2) @ VLU.transpose(-2, -1)
    Sigma_new = symmetrize(Sigma_new)

    # 6. Enforce spectral safeguards
    eigs_new = torch.linalg.eigvalsh(Sigma_new)
    min_eig = eigs_new[..., 0:1]
    max_eig = eigs_new[..., -1:]

    # Spectral floor
    needs_floor = (min_eig < eps_min).squeeze(-1)
    if needs_floor.any():
        eye = torch.eye(Sigma.shape[-1], device=Sigma.device, dtype=Sigma.dtype)
        Sigma_new = torch.where(
            needs_floor.unsqueeze(-1).unsqueeze(-1),
            Sigma_new + eps_min * eye,
            Sigma_new
        )

    # Condition cap
    condition = max_eig / min_eig.clamp(min=eps_min)
    needs_cap = (condition > kappa_max).squeeze(-1)
    if needs_cap.any():
        # Shrink toward identity
        eye = torch.eye(Sigma.shape[-1], device=Sigma.device, dtype=Sigma.dtype)
        trace = torch.diagonal(Sigma_new, dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1)
        K = Sigma.shape[-1]
        Sigma_new = torch.where(
            needs_cap.unsqueeze(-1).unsqueeze(-1),
            0.9 * Sigma_new + 0.1 * (trace / K) * eye,
            Sigma_new
        )

    return Sigma_new


# ---------------------------------------------------------------------------
# State-dependent prior precision (Eq. 16)
# ---------------------------------------------------------------------------

def state_dependent_alpha(mu, Sigma, prior_mu, prior_Sigma, b0, c0):
    """
    α_i = c₀ / (b₀ + KL(q_i || p_i))

    Returns: [B, N]
    """
    kl = kl_divergence(mu, Sigma, prior_mu, prior_Sigma)  # [B, N]
    return c0 / (b0 + kl.clamp(min=0.0))


# ---------------------------------------------------------------------------
# KL-based decoding (logits from prior bank)
# ---------------------------------------------------------------------------

def kl_decode_logits(mu, Sigma, prior_mu_bank, prior_Sigma_bank):
    """
    Compute logits for each vocabulary token via negative KL to prior.

    logit_v(i) = -KL(q_i || π_v)

    Args:
        mu: [B, N, K]
        Sigma: [B, N, K, K]
        prior_mu_bank: [V, K]
        prior_Sigma_bank: [V, K, K]

    Returns: [B, N, V]
    """
    B, N, K = mu.shape
    V = prior_mu_bank.shape[0]

    # Process in chunks to manage memory: V can be ~50k
    chunk_size = min(V, 1024)
    logits = torch.empty(B, N, V, device=mu.device, dtype=mu.dtype)

    logdet_Sigma = safe_logdet(Sigma)  # [B, N]

    for v_start in range(0, V, chunk_size):
        v_end = min(v_start + chunk_size, V)
        chunk_V = v_end - v_start

        mu_v = prior_mu_bank[v_start:v_end]        # [cV, K]
        Sig_v = prior_Sigma_bank[v_start:v_end]     # [cV, K, K]
        Sig_v_inv = safe_inverse(Sig_v)             # [cV, K, K]
        logdet_v = safe_logdet(Sig_v)               # [cV]

        # KL(q_i || π_v) for each (i, v) pair
        # = ½[tr(Σ_v⁻¹ Σ_i) + (μ_v - μ_i)ᵀ Σ_v⁻¹ (μ_v - μ_i) - K + ln det Σ_v - ln det Σ_i]

        # Trace: tr(Σ_v⁻¹ Σ_i) — same for all v given i
        # [B, N, cV]: need Σ_v⁻¹[v] and Σ_i
        trace_term = torch.einsum('vpq,bnqp->bnv', Sig_v_inv, Sigma)  # [B, N, cV]

        # Mahalanobis: (μ_v - μ_i)ᵀ Σ_v⁻¹ (μ_v - μ_i)
        delta = mu_v.unsqueeze(0).unsqueeze(0) - mu.unsqueeze(2)  # [B, N, cV, K]
        Sinv_d = torch.einsum('vpq,bnvq->bnvp', Sig_v_inv, delta)  # [B, N, cV, K]
        mahal = (delta * Sinv_d).sum(-1)  # [B, N, cV]

        # Log-det
        logdet_term = logdet_v.unsqueeze(0).unsqueeze(0) - logdet_Sigma.unsqueeze(-1)

        kl = 0.5 * (trace_term + mahal - K + logdet_term)
        logits[:, :, v_start:v_end] = -kl

    return logits


# ---------------------------------------------------------------------------
# Cross-entropy gradient (analytic, no autograd)
# ---------------------------------------------------------------------------

def softmax_ce_gradient(logits, targets):
    """
    Gradient of cross-entropy loss w.r.t. logits.

    ∂CE/∂logits = softmax(logits) - one_hot(targets)

    Args:
        logits: [B, N, V]
        targets: [B, N] long tensor

    Returns: [B, N, V]
    """
    probs = torch.softmax(logits, dim=-1)
    one_hot = torch.zeros_like(probs)
    one_hot.scatter_(-1, targets.unsqueeze(-1), 1.0)
    return probs - one_hot


# ---------------------------------------------------------------------------
# Frobenius norm clipping (trust region)
# ---------------------------------------------------------------------------

def clip_norm(x, max_norm):
    """Clip tensor to have Frobenius norm ≤ max_norm along last dims."""
    norm = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = torch.clamp(max_norm / norm, max=1.0)
    return x * scale


def clip_matrix_norm(M, max_norm):
    """Clip matrix tensor Frobenius norm along last two dims."""
    norm = M.flatten(-2).norm(dim=-1, keepdim=True).unsqueeze(-1).clamp(min=1e-8)
    scale = torch.clamp(max_norm / norm, max=1.0)
    return M * scale
