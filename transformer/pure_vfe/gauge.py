"""
Gauge transport for the Pure VFE Transformer.

Supports two parameterizations (togglable via config.gauge_param):
  'omega': Direct GL⁺(K_h) storage per head. Transport Ω_ij = Ω_i · Ω_j⁻¹.
  'phi':   Lie algebra φ ∈ gl(K_h). Transport Ω_ij = exp(φ_i) · exp(-φ_j).

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

    ∂KL/∂Ω_i = ∂KL/∂Ω_ij @ Ω_j⁻ᵀ

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
        grad_h = torch.einsum('bij,bijpq->bipq', beta_h, dKL_dOi)  # [B, N, K_h, K_h]

        grad_Omega[:, :, h] = grad_h

    return grad_Omega


# ---------------------------------------------------------------------------
# Natural gradient on GL(K)
# ---------------------------------------------------------------------------

def natural_grad_omega(grad_Omega, Omega):
    """
    Left-invariant natural gradient on GL(K).

    ΔΩ = Ω · Ωᵀ · ∂F/∂Ω

    For metric g_Ω(X,Y) = tr((Ω⁻¹X)ᵀ(Ω⁻¹Y)), the natural gradient is Ω Ωᵀ ∂F/∂Ω.

    Args:
        grad_Omega: [..., K, K] Euclidean gradient ∂F/∂Ω
        Omega: [..., K, K] current gauge frame

    Returns: [..., K, K] natural gradient direction
    """
    OmegaT = Omega.transpose(-2, -1)
    return Omega @ OmegaT @ grad_Omega


# ---------------------------------------------------------------------------
# Trust region and conditioning (ported from VFE dynamic)
# ---------------------------------------------------------------------------

def relative_trust_clip(nat_grad, Omega, trust_region=0.3, max_norm=None):
    """
    Two-level trust region clip for Omega natural gradient.

    Level 1 (relative): ||nat_grad|| ≤ trust_region × ||Omega||_F
        Scales step size with parameter magnitude.
    Level 2 (absolute): ||nat_grad|| ≤ max_norm
        Hard cap that prevents runaway when Omega grows large.
        This breaks the positive feedback loop where large Omega → large
        natural gradient → large step → even larger Omega.

    The tighter of the two bounds is applied.

    Args:
        nat_grad: [..., K, K] natural gradient
        Omega: [..., K, K] current gauge frame
        trust_region: max relative step size
        max_norm: absolute Frobenius norm cap (None = no absolute cap)

    Returns: [..., K, K] clipped natural gradient
    """
    nat_norm = nat_grad.flatten(-2).norm(dim=-1, keepdim=True).unsqueeze(-1)
    om_norm = Omega.flatten(-2).norm(dim=-1, keepdim=True).unsqueeze(-1).clamp(min=1e-6)
    max_upd = trust_region * om_norm

    # Apply absolute cap if specified (takes the tighter bound)
    if max_norm is not None:
        abs_cap = torch.tensor(max_norm, device=nat_grad.device, dtype=nat_grad.dtype)
        max_upd = torch.minimum(max_upd, abs_cap)

    scale = torch.clamp(max_upd / (nat_norm + 1e-8), max=1.0)
    return nat_grad * scale


def regularize_omega_conditioning(Omega, cond_max=50.0):
    """
    Progressive regularization of ill-conditioned Omega toward identity.

    When condition number exceeds cond_max, blend Omega toward I with
    strength proportional to the excess:
      blend = clamp(0.1 × (cond/cond_max - 1), 0, 0.5)
      Omega_new = (1 - blend) × Omega + blend × I

    This is more aggressive than the previous fixed 5% blend: matrices
    at 2× the threshold get a 10% blend, at 6× they get the maximum 50%.

    Args:
        Omega: [..., K, K] gauge frames
        cond_max: maximum allowed condition number

    Returns: [..., K, K] regularized gauge frames
    """
    K = Omega.shape[-1]
    svs = torch.linalg.svdvals(Omega)
    cond = svs[..., 0] / svs[..., -1].clamp(min=1e-8)
    needs_reg = cond > cond_max
    if needs_reg.any():
        eye = torch.eye(K, device=Omega.device, dtype=Omega.dtype)
        # Progressive blend: stronger for worse conditioning
        excess = (cond / cond_max).clamp(min=1.0)
        blend = torch.clamp(0.1 * (excess - 1.0), min=0.0, max=0.5)
        blend = blend.unsqueeze(-1).unsqueeze(-1)
        Omega = torch.where(
            needs_reg.unsqueeze(-1).unsqueeze(-1),
            (1.0 - blend) * Omega + blend * eye,
            Omega
        )
    return Omega


# ---------------------------------------------------------------------------
# Phi (Lie algebra) path utilities
# ---------------------------------------------------------------------------

def make_gl_generators(K, device='cpu'):
    """
    Construct GL(K) generators: K² basis matrices E_{ab} with (E_{ab})_{ij} = δ_{ai}δ_{bj}.

    Args:
        K: dimension
        device: torch device

    Returns: (K², K, K) generators
    """
    n_gen = K * K
    generators = torch.zeros(n_gen, K, K, device=device)
    idx = 0
    for a in range(K):
        for b in range(K):
            generators[idx, a, b] = 1.0
            idx += 1
    return generators


def phi_to_omega(phi, generators):
    """
    Convert Lie algebra element φ to group element Ω = exp(Σ_a φ^a T_a).

    Args:
        phi: [..., n_gen] Lie algebra coordinates
        generators: (n_gen, K, K) basis matrices

    Returns: [..., K, K] group element
    """
    X = torch.einsum('...a,aij->...ij', phi, generators)
    return torch.linalg.matrix_exp(X)


def retract_phi(phi, delta_phi, max_norm=3.14159):
    """
    Retract phi update: clamp ||phi|| to max_norm.

    Args:
        phi: [..., n_gen] current Lie algebra element
        delta_phi: [..., n_gen] update direction
        max_norm: maximum norm (default π)

    Returns: [..., n_gen] updated phi
    """
    phi_new = phi + delta_phi
    norm = phi_new.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = torch.clamp(max_norm / norm, max=1.0)
    return phi_new * scale


def init_phi(shape, scale=0.01, device='cpu'):
    """
    Initialize Lie algebra elements near zero (identity group element).

    Args:
        shape: tuple, e.g. (V, H, n_gen_h) or (N, H, n_gen_h)
        scale: perturbation magnitude
        device: torch device

    Returns: tensor of shape `shape`
    """
    return scale * torch.randn(shape, device=device)


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
