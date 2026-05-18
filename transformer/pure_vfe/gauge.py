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

    # The ½ from KL = ½[...] is already absorbed into each term's derivation.
    # term1, term2, term3 ARE the full ∂KL/∂Ω — no additional factor needed.
    return term1 + term2 + term3


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

    Fully vectorized over all (b, h, i, j). Reuses ``precomp['Omega_inv']``
    and ``precomp['Sigma_inv']`` to skip two redundant batched inversions
    per head, and computes ``Om_ij⁻¹ = Om_j · Om_i⁻¹`` from the (already
    available) per-token inverses instead of inverting the full pairwise
    ``[B, H, N, N, K_h, K_h]`` Om_ij tensor.

    Args:
        mu_h: [B, N, H, K_h] per-head means
        Sigma_h: [B, N, H, K_h, K_h] per-head covariances
        Omega: [B, N, H, K_h, K_h] gauge frames
        beta: [B, H, N, N] attention weights
        kl_ij: [B, H, N, N] pairwise KL (unused — present for API stability)
        precomp: precomputed quantities dict (must include 'Omega_inv' and 'Sigma_inv')

    Returns: [B, N, H, K_h, K_h] gradient ∂F/∂Ω_i
    """
    _ = kl_ij  # accepted for API stability, not consumed
    B, N, H, K_h = mu_h.shape

    # Head-first layouts (matching precomp's [B, H, N, ...] convention).
    mu = mu_h.permute(0, 2, 1, 3)                # [B, H, N, K_h]
    Sig = Sigma_h.permute(0, 2, 1, 3, 4)         # [B, H, N, K_h, K_h]
    Om = Omega.permute(0, 2, 1, 3, 4)            # [B, H, N, K_h, K_h]
    Om_inv = precomp['Omega_inv']                # [B, H, N, K_h, K_h] — reused
    Sig_inv = precomp['Sigma_inv']               # [B, H, N, K_h, K_h] — reused

    # Om_ij = Om_i @ Om_j⁻¹  and  Om_ij⁻¹ = Om_j @ Om_i⁻¹  (algebraic identity).
    # Materialize both as [B, H, N_i, N_j, K_h, K_h] without explicit inversion.
    Om_i = Om.unsqueeze(3)                       # [B, H, N, 1, K_h, K_h]
    Om_j = Om.unsqueeze(2)                       # [B, H, 1, N, K_h, K_h]
    Om_i_inv = Om_inv.unsqueeze(3)               # [B, H, N, 1, K_h, K_h]
    Om_j_inv = Om_inv.unsqueeze(2)               # [B, H, 1, N, K_h, K_h]
    Om_ij = Om_i @ Om_j_inv                      # [B, H, N, N, K_h, K_h]
    Om_ij_inv = Om_j @ Om_i_inv                  # [B, H, N, N, K_h, K_h]
    Om_ij_invT = Om_ij_inv.transpose(-2, -1)

    # Λ_ij = Om_ij⁻ᵀ Σ_j⁻¹ Om_ij⁻¹
    Sig_j_inv = Sig_inv.unsqueeze(2)             # [B, H, 1, N, K_h, K_h]
    Lambda_ij = Om_ij_invT @ Sig_j_inv @ Om_ij_inv

    # δ_ij = μ_i - Om_ij μ_j
    Om_ij_mu_j = torch.einsum('bhijpq,bhjq->bhijp', Om_ij, mu)
    delta_ij = mu.unsqueeze(3) - Om_ij_mu_j      # [B, H, N, N, K_h]

    # Term 1: -Λ_ij δ_ij μ_jᵀ
    Lam_delta = torch.einsum('bhijpq,bhijq->bhijp', Lambda_ij, delta_ij)
    term1 = -Lam_delta.unsqueeze(-1) * mu.unsqueeze(2).unsqueeze(-2)

    # Term 2: -Λ_ij (Σ_i + δ_ij δ_ijᵀ) Om_ij⁻ᵀ
    outer_ij = delta_ij.unsqueeze(-1) * delta_ij.unsqueeze(-2)
    Sig_i = Sig.unsqueeze(3)                     # [B, H, N, 1, K_h, K_h]
    inner_ij = Sig_i + outer_ij
    term2 = -(Lambda_ij @ inner_ij @ Om_ij_invT)

    # Term 3: Om_ij⁻ᵀ
    term3 = Om_ij_invT

    dKL_dOij = term1 + term2 + term3
    # NaN propagation guard matching the legacy per-head loop.
    dKL_dOij = torch.clamp(dKL_dOij, -1e6, 1e6)

    # Chain rule: ∂KL/∂Ω_i = ∂KL/∂Ω_ij @ Ω_j⁻ᵀ
    Om_j_invT = Om_inv.transpose(-2, -1).unsqueeze(2)  # [B, H, 1, N, K_h, K_h]
    dKL_dOi = dKL_dOij @ Om_j_invT

    # Weighted sum over j: ∂F/∂Ω_i = Σ_j β_ij · ∂KL/∂Ω_i
    grad_bh = torch.einsum('bhij,bhijpq->bhipq', beta, dKL_dOi)  # [B, H, N, K_h, K_h]
    return grad_bh.permute(0, 2, 1, 3, 4).contiguous()           # [B, N, H, K_h, K_h]


def vfe_grad_Omega_full(mu_h, Sigma_h, Omega, beta, kl_ij, precomp):
    r"""Full VFE gradient :math:`\partial F / \partial \Omega_i` including both
    forward (i as query) and backward (i as key) contributions.

    .. math::
        \frac{\partial F}{\partial \Omega_i}
        = \underbrace{\sum_j \beta_{ij} \frac{\partial \mathrm{KL}_{ij}}{\partial \Omega_i}}_{\text{forward}}
        + \underbrace{\sum_j \beta_{ji} \frac{\partial \mathrm{KL}_{ji}}{\partial \Omega_i}}_{\text{backward}}

    The forward term is :math:`(\partial \mathrm{KL}/\partial \Omega_{ij}) \Omega_j^{-\top}`.
    The backward term uses :math:`\Omega_{ji} = \Omega_j \Omega_i^{-1}`, giving
    :math:`\partial \mathrm{KL}_{ji}/\partial \Omega_i = -\Omega_{ji}^\top
    (\partial \mathrm{KL}/\partial \Omega_{ji}) \Omega_i^{-\top}`.

    This is needed for the M-step, where ``Omega_i`` changes affect ALL pairs
    involving position i. The E-step uses the forward-only version since each
    position updates independently.

    Args:
        mu_h: [B, N, H, K_h] per-head means
        Sigma_h: [B, N, H, K_h, K_h] per-head covariances
        Omega: [B, N, H, K_h, K_h] gauge frames
        beta: [B, H, N, N] attention weights
        kl_ij: [B, H, N, N] pairwise KL
        precomp: precomputed quantities dict

    Returns: [B, N, H, K_h, K_h] full gradient ∂F/∂Ω_i
    """
    B, N, H, K_h = mu_h.shape
    device = mu_h.device
    dtype = mu_h.dtype

    grad_Omega = torch.zeros(B, N, H, K_h, K_h, device=device, dtype=dtype)

    mu = mu_h.permute(0, 2, 1, 3)          # [B, H, N, K_h]
    Sig = Sigma_h.permute(0, 2, 1, 3, 4)   # [B, H, N, K_h, K_h]
    Om = Omega.permute(0, 2, 1, 3, 4)      # [B, H, N, K_h, K_h]

    for h in range(H):
        mu_h_s = mu[:, h]           # [B, N, K_h]
        Sig_h_s = Sig[:, h]         # [B, N, K_h, K_h]
        Om_h_s = Om[:, h]           # [B, N, K_h, K_h]
        beta_h = beta[:, h]         # [B, N, N]

        Om_inv = torch.linalg.inv(Om_h_s)  # [B, N, K_h, K_h]

        # Ω_ij = Ω_i @ Ω_j⁻¹
        Om_ij = Om_h_s[:, :, None] @ Om_inv[:, None, :]  # [B, N, N, K_h, K_h]
        Om_ij_inv = torch.linalg.inv(Om_ij)
        Om_ij_invT = Om_ij_inv.transpose(-2, -1)

        Sig_j_inv = safe_inverse(Sig_h_s)

        # Λ_ij, δ_ij
        Lambda_ij = Om_ij_invT @ Sig_j_inv[:, None, :] @ Om_ij_inv
        Om_ij_mu_j = torch.einsum('bijnk,bjk->bijn', Om_ij, mu_h_s)
        delta_ij = mu_h_s[:, :, None] - Om_ij_mu_j

        # ∂KL/∂Ω_ij (three terms)
        Lam_delta = torch.einsum('bijpq,bijq->bijp', Lambda_ij, delta_ij)
        term1 = -Lam_delta.unsqueeze(-1) * mu_h_s[:, None, :].unsqueeze(-2)
        outer_ij = delta_ij.unsqueeze(-1) * delta_ij.unsqueeze(-2)
        inner_ij = Sig_h_s[:, :, None] + outer_ij
        term2 = -(Lambda_ij @ inner_ij @ Om_ij_invT)
        term3 = Om_ij_invT

        dKL_dOij = term1 + term2 + term3  # [B, N_i, N_j, K_h, K_h]
        dKL_dOij = torch.clamp(dKL_dOij, -1e6, 1e6)

        # --- Forward: ∂KL_ij/∂Ω_i = ∂KL/∂Ω_ij @ Ω_j⁻ᵀ ---
        Om_j_invT = Om_inv.transpose(-2, -1)  # [B, N, K_h, K_h]
        dKL_dOi_fwd = dKL_dOij @ Om_j_invT[:, None, :]
        grad_fwd = torch.einsum('bij,bijpq->bipq', beta_h, dKL_dOi_fwd)

        # --- Backward: ∂KL_ji/∂Ω_i = -Ω_jiᵀ @ ∂KL/∂Ω_ji @ Ω_i⁻ᵀ ---
        # dKL_dOij[j, i] = ∂KL_ji/∂Ω_ji
        dKL_dOji = dKL_dOij.transpose(1, 2)  # [B, N_j, N_i, K_h, K_h] → swap i↔j
        Om_ji = Om_ij.transpose(1, 2)        # [B, N_j, N_i, K_h, K_h]
        Om_jiT = Om_ji.transpose(-2, -1)
        Om_i_invT = Om_inv.transpose(-2, -1)  # [B, N_i, K_h, K_h]
        # dKL_ji/dΩ_i = -Ω_jiᵀ @ dKL/dΩ_ji @ Ω_i⁻ᵀ
        # Shape: [B, N_j, N_i, K_h, K_h] with i in dim 2
        dKL_dOi_bwd = -(Om_jiT @ dKL_dOji @ Om_i_invT[:, None, :])
        dKL_dOi_bwd = torch.clamp(dKL_dOi_bwd, -1e6, 1e6)
        # beta_ji is beta_h[:, j, i] → sum over j (dim 1)
        # We need sum_j beta_ji * dKL_ji/dOmega_i, with j in dim 1, i in dim 2
        beta_ji = beta_h.transpose(1, 2)  # [B, N_i, N_j] → swap to [B, j, i]? No.
        # beta_h[b, j, i] with j in dim1, i in dim2
        # dKL_dOi_bwd[b, j, i, p, q] — sum over j
        grad_bwd = torch.einsum('bji,bjipq->bipq', beta_h, dKL_dOi_bwd)

        grad_Omega[:, :, h] = grad_fwd + grad_bwd

    return grad_Omega


# ---------------------------------------------------------------------------
# Natural gradient on GL(K)
# ---------------------------------------------------------------------------

def natural_grad_omega(grad_Omega, Omega):
    """
    Left-invariant natural gradient on GL(K).

    ΔΩ = Ω · Ωᵀ · ∂F/∂Ω

    For metric g_Ω(X,Y) = tr((Ω⁻¹X)ᵀ(Ω⁻¹Y)), the natural gradient is Ω Ωᵀ ∂F/∂Ω.

    Note: prefer lie_algebra_clip_grad which computes and clips the natural
    gradient in the Lie algebra for geometric consistency.

    Args:
        grad_Omega: [..., K, K] Euclidean gradient ∂F/∂Ω
        Omega: [..., K, K] current gauge frame

    Returns: [..., K, K] natural gradient direction
    """
    OmegaT = Omega.transpose(-2, -1)
    return Omega @ OmegaT @ grad_Omega


# ---------------------------------------------------------------------------
# Trust region and conditioning
# ---------------------------------------------------------------------------

def lie_algebra_clip_grad(grad_Omega, Omega, trust_radius=0.3):
    """
    Compute and clip the natural gradient via the Lie algebra of GL(K).

    Instead of forming Ω·Ωᵀ·g in the ambient space (which amplifies by σ(Ω)²)
    and then clipping in Euclidean norm (which depends on where Ω sits), we:

      1. Pull back to the Lie algebra: ξ = Ωᵀ · ∂F/∂Ω
      2. Clip ||ξ||_F ≤ trust_radius
      3. Push forward: ΔΩ = Ω · ξ_clipped

    Why this works: ||ξ||_F = ||Ω⁻¹ΔΩ||_F is the Riemannian step size
    under the left-invariant metric. It is invariant under left translation
    (Ω → A·Ω for any A ∈ GL(K)), so the trust region has the same intrinsic
    size everywhere on the manifold. No feedback loop is possible because
    the clip threshold is a *constant* in the Riemannian geometry.

    The subsequent Euler retraction Ω - η·ΔΩ = Ω·(I - η·ξ) approximates
    the geodesic retraction Ω·exp(-η·ξ) to first order, which is accurate
    when η·||ξ|| is small — guaranteed by the trust region.

    Args:
        grad_Omega: [..., K, K] Euclidean gradient ∂F/∂Ω
        Omega: [..., K, K] current gauge frame
        trust_radius: max Riemannian step size ||ξ||_F

    Returns: [..., K, K] natural gradient direction (clipped)
    """
    # Step 1: Lie algebra element ξ = Ωᵀ · g
    # (This is Ω⁻¹ · (Ω·Ωᵀ·g) = Ωᵀ·g, the pullback of the Riemannian
    # gradient to the identity, i.e. the gradient in gl(K) coordinates.)
    OmegaT = Omega.transpose(-2, -1)
    xi = OmegaT @ grad_Omega  # [..., K, K]

    # Step 2: Clip in Lie algebra norm (= Riemannian norm)
    xi_norm = xi.flatten(-2).norm(dim=-1, keepdim=True).unsqueeze(-1)
    scale = torch.clamp(trust_radius / (xi_norm + 1e-8), max=1.0)
    xi = xi * scale

    # Step 3: Push forward to tangent space at Ω
    return Omega @ xi


def relative_trust_clip(nat_grad, Omega, trust_region=0.3, max_norm=None):
    """
    Legacy Euclidean trust region clip (kept for backward compatibility).

    Prefer lie_algebra_clip_grad for geometrically consistent clipping.
    """
    nat_norm = nat_grad.flatten(-2).norm(dim=-1, keepdim=True).unsqueeze(-1)
    om_norm = Omega.flatten(-2).norm(dim=-1, keepdim=True).unsqueeze(-1).clamp(min=1e-6)
    max_upd = trust_region * om_norm
    if max_norm is not None:
        abs_cap = torch.tensor(max_norm, device=nat_grad.device, dtype=nat_grad.dtype)
        max_upd = torch.minimum(max_upd, abs_cap)
    scale = torch.clamp(max_upd / (nat_norm + 1e-8), max=1.0)
    return nat_grad * scale


def regularize_omega_conditioning(Omega, cond_max=50.0):
    """
    Progressive regularization of ill-conditioned Omega toward its polar factor.

    When condition number exceeds cond_max, blend Omega toward Q = U·Vᵀ
    (the nearest orthogonal matrix from polar decomposition) with strength
    proportional to the excess:
      blend = clamp(0.1 × (cond/cond_max - 1), 0, 0.5)
      Omega_new = (1 - blend) × Omega + blend × Q

    Using the polar factor instead of the identity matrix is critical for
    GL(K) support: Q preserves the sign of det(Omega), so frames in GL⁻(K)
    (det < 0) are regularized toward O⁻(K), not pushed through det = 0
    toward I. For frames already in GL⁺(K), Q ∈ SO(K) ⊂ GL⁺(K).

    Geometrically, the polar decomposition Ω = Q·P separates the "rotation"
    Q ∈ O(K) from the "stretch" P ∈ SPD(K). Regularizing toward Q shrinks
    P toward I (reducing condition number) while preserving Q.

    Args:
        Omega: [..., K, K] gauge frames
        cond_max: maximum allowed condition number

    Returns: [..., K, K] regularized gauge frames
    """
    svs = torch.linalg.svdvals(Omega)
    cond = svs[..., 0] / svs[..., -1].clamp(min=1e-8)
    needs_reg = cond > cond_max
    if needs_reg.any():
        # Full SVD only when regularization is needed (K_h typically 3-8, cheap)
        U, S, Vh = torch.linalg.svd(Omega)
        Q = U @ Vh  # polar factor: nearest orthogonal, preserves det sign
        # Progressive blend: stronger for worse conditioning
        excess = (cond / cond_max).clamp(min=1.0)
        blend = torch.clamp(0.1 * (excess - 1.0), min=0.0, max=0.5)
        blend = blend.unsqueeze(-1).unsqueeze(-1)
        Omega = torch.where(
            needs_reg.unsqueeze(-1).unsqueeze(-1),
            (1.0 - blend) * Omega + blend * Q,
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


def retract_phi(phi, delta_phi, max_norm=None):
    """
    Retract phi update: clamp ||phi|| to max_norm.

    Args:
        phi: [..., n_gen] current Lie algebra element
        delta_phi: [..., n_gen] update direction
        max_norm: maximum norm; None = auto-select based on n_gen
            (π for SO(N) i.e. n_gen is triangular, 5.0 for GL(K) i.e. n_gen is a perfect square)

    Returns: [..., n_gen] updated phi
    """
    if max_norm is None:
        import math
        n_gen = phi.shape[-1]
        K = int(math.isqrt(n_gen))
        is_glk = (K * K == n_gen and K > 0)
        max_norm = 5.0 if is_glk else math.pi
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

def init_omega(shape, scale=0.01, device='cuda', negative_det_fraction=0.0):
    """
    Initialize gauge frames near identity.

    Ω = I + scale · randn(K, K)

    For the omega path (gauge_param='omega'), frames live in full GL(K).
    Setting negative_det_fraction > 0 seeds a fraction of frames in GL⁻(K)
    (det < 0) by flipping the first column. This is necessary because the
    Lie algebra retraction preserves det sign — frames cannot cross between
    GL⁺(K) and GL⁻(K) during training.

    For the phi path (gauge_param='phi'), this function is not used (phi
    uses init_phi + exp → GL⁺(K) automatically).

    Args:
        shape: tuple, e.g. (V, H, K_h, K_h) or (N, H, K_h, K_h)
        scale: perturbation magnitude
        device: torch device
        negative_det_fraction: fraction of frames to initialize with det < 0.
            0.0 = all positive det (GL⁺(K), backward compatible default).
            0.5 = half negative (seeds both components of GL(K)).

    Returns: tensor of shape `shape`
    """
    assert shape[-1] == shape[-2], "Last two dims must be K×K"
    K = shape[-1]
    eye = torch.eye(K, device=device)
    Omega = eye.expand(shape).clone()
    Omega = Omega + scale * torch.randn(shape, device=device)

    if negative_det_fraction <= 0.0:
        # GL⁺(K): ensure all frames have positive determinant
        dets = torch.linalg.det(Omega)
        neg_mask = dets < 0
        if neg_mask.any():
            Omega[neg_mask, :, 0] *= -1
    else:
        # GL(K): first ensure all positive, then flip a fraction to negative
        dets = torch.linalg.det(Omega)
        neg_mask = dets < 0
        if neg_mask.any():
            Omega[neg_mask, :, 0] *= -1

        # Randomly select frames to place in GL⁻(K)
        flip_mask = torch.rand(shape[:-2], device=device) < negative_det_fraction
        Omega[flip_mask, :, 0] *= -1

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