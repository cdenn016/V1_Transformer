"""
Direct GL(K) gauge transport for the Pure VFE Transformer.

No Lie algebra parameterization. Ω_i ∈ GL⁺(K_h) stored directly per head.
Transport: Ω_ij = Ω_i · Ω_j⁻¹ (automatic cocycle condition).
Natural gradient: left-invariant metric on GL(K).

Mathematical reference:
  - Supplementary Appendix C (gauge frame gradients)
  - derivations/constant_gauge_kl_derivation.md §9 (GL(K) invariance)
"""

import torch

from .gaussians import safe_inverse, symmetrize


# ---------------------------------------------------------------------------
# Transport operations
# ---------------------------------------------------------------------------

def compute_transport(Omega_i, Omega_j, connection_delta=None, generators=None,
                      cocycle_relaxation=0.0):
    """
    Compute pairwise gauge transport Ω_ij.

    Flat:      Ω_ij = Ω_i · Ω_j⁻¹  (cocycle condition automatic)
    Non-flat:  Ω_ij = Ω_i · exp(α·δ_ij·G) · Ω_j⁻¹

    The cocycle condition Ω_ij · Ω_jk = Ω_ik is automatically satisfied
    in the flat case: (Ω_i Ω_j⁻¹)(Ω_j Ω_k⁻¹) = Ω_i Ω_k⁻¹.

    Args:
        Omega_i: [..., K, K]
        Omega_j: [..., K, K]
        connection_delta: Optional [..., n_gen] edge-local Lie algebra elements
        generators: Optional (n_gen, K, K) generators for connection
        cocycle_relaxation: Scale for connection (0=flat, 1=fully non-flat)

    Returns: [..., K, K]
    """
    Omega_j_inv = torch.linalg.inv(Omega_j)

    if connection_delta is not None and generators is not None and cocycle_relaxation > 0:
        # Non-flat: Ω_ij = Ω_i · exp(α·δ_ij·G) · Ω_j⁻¹
        scaled_delta = cocycle_relaxation * connection_delta
        delta_matrix = torch.einsum('...a,aij->...ij', scaled_delta, generators)
        exp_delta = torch.linalg.matrix_exp(delta_matrix.float()).to(Omega_i.dtype)
        return Omega_i @ exp_delta @ Omega_j_inv

    return Omega_i @ Omega_j_inv


def transport_gaussian(mu, Sigma, Omega):
    """
    Pushforward of Gaussian N(μ, Σ) through Ω:
      Ω · N(μ, Σ) = N(Ω μ, Ω Σ Ωᵀ)

    Args:
        mu: [..., K]
        Sigma: [..., K, K]
        Omega: [..., K, K]

    Returns: (mu_transported, Sigma_transported)
    """
    mu_t = torch.einsum('...ij,...j->...i', Omega, mu)
    Sigma_t = Omega @ Sigma @ Omega.transpose(-2, -1)
    return mu_t, Sigma_t


# ---------------------------------------------------------------------------
# Gradient of KL w.r.t. Ω_ij
# ---------------------------------------------------------------------------

def grad_kl_Omega_ij(mu_i, Sigma_i, mu_j, Sigma_j, Omega_ij):
    """
    ∂KL(q_i || Ω_ij · q_j)/∂Ω_ij

    From the KL formula with Ω_ij transport, the gradient involves:
    - Mean term: ∂/∂Ω [ (μ_i - Ω μ_j)ᵀ (ΩΣ_jΩᵀ)⁻¹ (μ_i - Ω μ_j) ]
    - Trace term: ∂/∂Ω [ tr((ΩΣ_jΩᵀ)⁻¹ Σ_i) ]
    - Log-det term: ∂/∂Ω [ ln det(ΩΣ_jΩᵀ) ]

    Using the identity (ΩΣΩᵀ)⁻¹ = Ω⁻ᵀΣ⁻¹Ω⁻¹, and defining
    Λ = (ΩΣ_jΩᵀ)⁻¹, δ = μ_i - Ωμ_j:

    ∂KL/∂Ω = -Λ δ μ_jᵀ                          (mean: numerator)
              - Λ (Σ_i + δδᵀ) Ω⁻ᵀ                (mean+trace: precision change)
              + Ω⁻ᵀ                                (log-det)

    Simplified:
    ∂KL/∂Ω_ij = Λ[-δ μ_jᵀ - (Σ_i + δδᵀ) Ω_ij⁻ᵀ] + Ω_ij⁻ᵀ

    where Λ = Ω_ij⁻ᵀ Σ_j⁻¹ Ω_ij⁻¹, δ = μ_i - Ω_ij μ_j.

    Args:
        mu_i: [..., K]
        Sigma_i: [..., K, K]
        mu_j: [..., K]
        Sigma_j: [..., K, K]
        Omega_ij: [..., K, K]

    Returns: [..., K, K] gradient ∂KL/∂Ω_ij
    """
    Omega_ij_inv = torch.linalg.inv(Omega_ij)
    Omega_ij_invT = Omega_ij_inv.transpose(-2, -1)
    Sigma_j_inv = safe_inverse(Sigma_j)

    # Transported precision: Λ = Ω⁻ᵀ Σ_j⁻¹ Ω⁻¹
    Lambda = Omega_ij_invT @ Sigma_j_inv @ Omega_ij_inv

    # Mean difference
    Omega_mu_j = torch.einsum('...ij,...j->...i', Omega_ij, mu_j)
    delta = mu_i - Omega_mu_j  # [..., K]

    # Term 1: -Λ δ μ_jᵀ
    term1 = -torch.einsum('...ij,...j->...i', Lambda, delta).unsqueeze(-1) * mu_j.unsqueeze(-2)

    # Term 2: -Λ (Σ_i + δδᵀ) Ω⁻ᵀ
    outer = delta.unsqueeze(-1) * delta.unsqueeze(-2)  # [..., K, K]
    inner = Sigma_i + outer
    term2 = -(Lambda @ inner @ Omega_ij_invT)

    # Term 3: Ω⁻ᵀ (from log-det)
    term3 = Omega_ij_invT

    return 0.5 * (term1 + term2) + 0.5 * term3  # ½ factor from KL formula


def grad_kl_Omega_i(mu_i, Sigma_i, mu_j, Sigma_j, Omega_i, Omega_j, beta_ij=None):
    """
    ∂KL(q_i || Ω_ij · q_j)/∂Ω_i via chain rule through Ω_ij = Ω_i · Ω_j⁻¹.

    ∂KL/∂Ω_i[a,b] = Σ_{c,d} (∂KL/∂Ω_ij[a,c]) · (∂Ω_ij/∂Ω_i[a,b])[a,c]

    Since Ω_ij = Ω_i Ω_j⁻¹, ∂Ω_ij[a,c]/∂Ω_i[a,b] = δ_{ac} (Ω_j⁻¹)[b,c]
    Actually: ∂(Ω_i M)[a,c]/∂Ω_i[a',b] = δ_{aa'} M[b,c]

    So: ∂KL/∂Ω_i = (∂KL/∂Ω_ij) @ Ω_j⁻ᵀ  (right multiply by Ω_j⁻ᵀ)

    Wait, let me be more careful.
    Ω_ij = Ω_i @ Ω_j⁻¹ = Ω_i @ M where M = Ω_j⁻¹.
    ∂(Ω_i M)/∂Ω_i = I ⊗ Mᵀ in Kronecker notation.
    So ∂KL/∂Ω_i = ∂KL/∂Ω_ij @ Mᵀ = ∂KL/∂Ω_ij @ (Ω_j⁻¹)ᵀ = ∂KL/∂Ω_ij @ Ω_j⁻ᵀ

    Args:
        mu_i, Sigma_i: query Gaussian [..., K], [..., K, K]
        mu_j, Sigma_j: key Gaussian [..., K], [..., K, K]
        Omega_i, Omega_j: gauge frames [..., K, K]

    Returns: [..., K, K] gradient ∂KL/∂Ω_i
    """
    Omega_j_inv = torch.linalg.inv(Omega_j)
    Omega_ij = Omega_i @ Omega_j_inv

    dKL_dOij = grad_kl_Omega_ij(mu_i, Sigma_i, mu_j, Sigma_j, Omega_ij)

    # Chain rule: ∂KL/∂Ω_i = ∂KL/∂Ω_ij @ Ω_j⁻ᵀ
    Omega_j_invT = Omega_j_inv.transpose(-2, -1)
    return dKL_dOij @ Omega_j_invT


# ---------------------------------------------------------------------------
# Aggregated gauge gradient (attention-weighted)
# ---------------------------------------------------------------------------

def vfe_grad_Omega(mu_h, Sigma_h, Omega, beta, kl_ij, precomp):
    """
    Full VFE gradient ∂F/∂Ω_i aggregated over attention-weighted pairs.

    ∂F/∂Ω_i = Σ_j β_ij · ∂KL_ij/∂Ω_i  (forward contribution)

    Fully vectorized over all (b, i, j) pairs per head.

    Args:
        mu_h: [B, N, H, K_h] per-head means
        Sigma_h: [B, N, H, K_h, K_h] per-head covariances
        Omega: [B, N, H, K_h, K_h] gauge frames
        beta: [B, H, N, N] attention weights
        kl_ij: [B, H, N, N] pairwise KL
        precomp: precomputed quantities dict

    Returns: [B, N, H, K_h, K_h] gradient ∂F/∂Ω_i
    """
    B, N, H, K_h = mu_h.shape
    device = mu_h.device
    dtype = mu_h.dtype

    grad_Omega = torch.zeros(B, N, H, K_h, K_h, device=device, dtype=dtype)

    # Transpose to head-first for consistency with beta
    mu = mu_h.permute(0, 2, 1, 3)          # [B, H, N, K_h]
    Sig = Sigma_h.permute(0, 2, 1, 3, 4)   # [B, H, N, K_h, K_h]
    Om = Omega.permute(0, 2, 1, 3, 4)      # [B, H, N, K_h, K_h]

    # Iterate over heads only (not over batch or positions)
    for h in range(H):
        mu_h_s = mu[:, h]           # [B, N, K_h]
        Sig_h_s = Sig[:, h]         # [B, N, K_h, K_h]
        Om_h_s = Om[:, h]           # [B, N, K_h, K_h]
        beta_h = beta[:, h]         # [B, N, N]

        # Compute Ω_j⁻¹ for all j: [B, N, K_h, K_h]
        Om_inv = torch.linalg.inv(Om_h_s)

        # Compute Ω_ij = Ω_i @ Ω_j⁻¹ for all (i,j) pairs
        # Om_h_s[:, :, None] is [B, N_i, 1, K_h, K_h]
        # Om_inv[:, None, :] is [B, 1, N_j, K_h, K_h]
        Om_ij = Om_h_s[:, :, None] @ Om_inv[:, None, :]  # [B, N, N, K_h, K_h]
        Om_ij_inv = torch.linalg.inv(Om_ij)               # [B, N, N, K_h, K_h]
        Om_ij_invT = Om_ij_inv.transpose(-2, -1)

        # Σ_j⁻¹ for all j
        Sig_j_inv = safe_inverse(Sig_h_s)  # [B, N, K_h, K_h]

        # Λ_ij = Ω_ij⁻ᵀ Σ_j⁻¹ Ω_ij⁻¹: [B, N_i, N_j, K_h, K_h]
        Lambda_ij = Om_ij_invT @ Sig_j_inv[:, None, :] @ Om_ij_inv

        # δ_ij = μ_i - Ω_ij μ_j: [B, N_i, N_j, K_h]
        Om_ij_mu_j = torch.einsum('bijnk,bjk->bijn', Om_ij, mu_h_s)
        delta_ij = mu_h_s[:, :, None] - Om_ij_mu_j  # [B, N_i, N_j, K_h]

        # ∂KL/∂Ω_ij:
        # Term 1: -Λ δ μ_jᵀ
        Lam_delta = torch.einsum('bijpq,bijq->bijp', Lambda_ij, delta_ij)
        term1 = -Lam_delta.unsqueeze(-1) * mu_h_s[:, None, :].unsqueeze(-2)
        # term1: [B, N_i, N_j, K_h, 1] * [B, 1, N_j, 1, K_h] = [B, N_i, N_j, K_h, K_h]

        # Term 2: -Λ (Σ_i + δδᵀ) Ω_ij⁻ᵀ
        outer_ij = delta_ij.unsqueeze(-1) * delta_ij.unsqueeze(-2)
        inner_ij = Sig_h_s[:, :, None] + outer_ij  # [B, N_i, N_j, K_h, K_h]
        term2 = -(Lambda_ij @ inner_ij @ Om_ij_invT)

        # Term 3: Ω_ij⁻ᵀ
        term3 = Om_ij_invT

        dKL_dOij = 0.5 * (term1 + term2) + 0.5 * term3  # [B, N_i, N_j, K_h, K_h]

        # Chain rule: ∂KL/∂Ω_i = ∂KL/∂Ω_ij @ Ω_j⁻ᵀ
        Om_j_invT = Om_inv.transpose(-2, -1)  # [B, N_j, K_h, K_h]
        dKL_dOi = dKL_dOij @ Om_j_invT[:, None, :]  # [B, N_i, N_j, K_h, K_h]

        # Weighted sum: Σ_j β_ij · ∂KL/∂Ω_i
        # beta_h: [B, N_i, N_j], dKL_dOi: [B, N_i, N_j, K_h, K_h]
        grad_h = torch.einsum('bij,bijpq->bipq', beta_h, dKL_dOi)  # [B, N, K_h, K_h]

        grad_Omega[:, :, h] = grad_h

    return grad_Omega


# ---------------------------------------------------------------------------
# Natural gradient on GL(K)
# ---------------------------------------------------------------------------

def natural_grad_omega(grad_Omega, Omega):
    """
    Left-invariant natural gradient on GL(K).

    ΔΩ = Ω · (Ω⁻ᵀ · ∂F/∂Ω)

    This left-translates the Euclidean gradient to the identity,
    giving the bi-invariant metric natural gradient.

    Args:
        grad_Omega: [..., K, K] Euclidean gradient ∂F/∂Ω
        Omega: [..., K, K] current gauge frame

    Returns: [..., K, K] natural gradient direction
    """
    # Lie algebra element via left-invariant metric on GL(K):
    # g_Ω(X,Y) = tr((Ω⁻¹X)ᵀ(Ω⁻¹Y)), so natural gradient = Ω Ωᵀ ∂F/∂Ω
    OmegaT = Omega.transpose(-2, -1)
    return Omega @ OmegaT @ grad_Omega


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def init_omega(shape, scale=0.01, device='cuda'):
    """
    Initialize GL⁺(K) gauge frames near identity.

    Ω = I + scale · randn(K, K), ensuring det > 0.

    Args:
        shape: tuple, e.g. (V, H, K_h, K_h) or (N, H, K_h, K_h)
        scale: perturbation magnitude
        device: torch device

    Returns: tensor of shape `shape`
    """
    assert shape[-1] == shape[-2], "Last two dims must be K×K"
    K = shape[-1]
    eye = torch.eye(K, device=device)
    Omega = eye.expand(shape).clone()
    Omega = Omega + scale * torch.randn(shape, device=device)

    # Ensure positive determinant
    dets = torch.linalg.det(Omega)
    neg_mask = dets < 0
    if neg_mask.any():
        # Flip first column to fix determinant sign
        Omega[neg_mask, :, 0] *= -1

    return Omega


def monitor_omega_health(Omega, name="Omega"):
    """Log condition numbers and determinants for debugging."""
    with torch.no_grad():
        dets = torch.linalg.det(Omega)
        svs = torch.linalg.svdvals(Omega)
        cond = svs[..., 0] / svs[..., -1].clamp(min=1e-8)
        return {
            f'{name}/det_min': dets.abs().min().item(),
            f'{name}/det_max': dets.abs().max().item(),
            f'{name}/cond_mean': cond.mean().item(),
            f'{name}/cond_max': cond.max().item(),
        }
