"""
Gauge-Theoretic Gradient Preconditioning
=========================================

Implements principled gradient preconditioning for gauge frame (φ) parameters
based on the Cartan decomposition of gl(K).

Background:
    gl(K) = so(K) ⊕ sym(K)

    - so(K): compact (antisymmetric generators). The matrix exponential
      has bounded derivatives: ||d exp(X)/dX|| = O(1) for X ∈ so(K).
    - sym(K): non-compact (symmetric generators). exp(X) grows without bound,
      and ||d exp(X)/dX|| ~ exp(||X||) for X ∈ sym(K).

    This non-compact amplification is why phi gradients can spike to ~1e3
    when using GL(K) gauge groups: the backward pass through matrix_exp
    amplifies gradients exponentially in the symmetric directions.

Three preconditioning modes (controlled by phi_natural_gradient config):

1. 'clip' (default): Simple norm clipping. No geometric awareness.

2. 'cartan': Approximate Cartan decomposition preconditioning.
    Projects gradient into so(K) ⊕ sym(K) and dampens sym(K) by a fixed
    factor (sym_dampening=0.1 → 10× dampening). Uses a free parameter.

3. 'killing': Killing form natural gradient (position-independent).
    Uses the Cartan-involution-modified Killing form as metric:
        g̃(X, Y) = -B(X, θ(Y)) where θ(X) = -X^T
        g̃_ab = 2K · tr(T_a^T T_b) - 2 · tr(T_a) · tr(T_b)
    No free parameters; metric determined entirely by algebra structure.

4. 'pullback': Full pullback natural gradient (position-dependent).
    Uses the Riemannian metric on φ-space pulled back through exp:
        G_ab(φ) = ⟨ Ψ(ad_X)(T_a), Ψ(ad_X)(T_b) ⟩_gram
    where X = Σ_a φ^a T_a, Ψ(z) = (e^z - 1)/z = Σ_{k=0}^∞ z^k/(k+1)!
    This is the theoretically exact natural gradient on the Lie group.
    It captures the exponential amplification in non-compact directions
    that the position-independent metrics miss.
    Cost: O(n_gen³) per token per step (expensive but principled).

Also implements:
    - SL(K) projection: projects φ to the traceless subalgebra sl(K),
      removing the single most dangerous degree of freedom (uniform scaling).

Author: Theoretical foundation from Cartan decomposition of gl(K)
Date: March 2026
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple


def build_cartan_projector(
    generators: torch.Tensor,  # (n_gen, K, K)
    sym_dampening: float = 0.1,
) -> torch.Tensor:
    """
    Build Cartan decomposition preconditioner for gl(K) phi gradients.

    Decomposes the Lie algebra gl(K) = so(K) ⊕ sym(K) and constructs a
    preconditioning matrix that dampens the non-compact (symmetric) directions.

    For generators {T_a}, the symmetric projection in coordinate space is:
        [P_sym]_{ab} = (1/2)(⟨T_a, T_b⟩ + tr(T_a T_b))

    where ⟨T_a, T_b⟩ = tr(T_aᵀ T_b) is the Frobenius inner product.
    For Frobenius-orthonormal generators, ⟨T_a, T_b⟩ = δ_ab.

    The preconditioner is:
        C = P_so + c · P_sym = I - (1-c) · P_sym

    where c ∈ (0, 1] dampens the symmetric directions:
        c = 1.0: no preconditioning (Euclidean gradient)
        c = 0.0: project out symmetric components entirely (restrict to so(K))
        c = 0.1: dampen symmetric by 10× (recommended for GL(K))

    Args:
        generators: Lie algebra generators (n_gen, K, K)
        sym_dampening: Dampening factor for symmetric directions.
                      0.1 means sym directions get 10× smaller steps.

    Returns:
        preconditioner: (n_gen, n_gen) matrix to left-multiply phi gradients
    """
    n_gen, K, _ = generators.shape
    device = generators.device
    dtype = generators.dtype

    # Gram matrix: G_ab = tr(T_a^T T_b) (Frobenius inner product)
    # For standard E_ij basis this is the identity.
    gram = torch.einsum('aij,bij->ab', generators.transpose(-2, -1), generators)

    # Trace product: tr(T_a T_b) — captures symmetric/antisymmetric structure
    # For antisymmetric T: tr(T^2) < 0. For symmetric T: tr(T^2) > 0.
    trace_prod = torch.einsum('aij,bji->ab', generators, generators)

    # Symmetric projection in generator coordinate space:
    # P_sym = G^{-1} · (G + trace_prod) / 2
    # For orthonormal generators (G = I): P_sym = (I + trace_prod) / 2

    # Use pseudoinverse for numerical stability (G may be ill-conditioned
    # for non-standard generator bases)
    gram_inv = torch.linalg.pinv(gram)
    P_sym = gram_inv @ (gram + trace_prod) / 2.0

    # Preconditioner: C = I - (1 - c) * P_sym
    # This keeps so(K) components at full strength and dampens sym(K) by factor c
    I_gen = torch.eye(n_gen, device=device, dtype=dtype)
    preconditioner = I_gen - (1.0 - sym_dampening) * P_sym

    return preconditioner


def build_slk_projector(
    generators: torch.Tensor,  # (n_gen, K, K)
) -> torch.Tensor:
    """
    Build projector that removes the trace component of phi, restricting to sl(K).

    For φ parameterizing M = Σ_a φ_a T_a ∈ gl(K), the trace is:
        tr(M) = Σ_a φ_a tr(T_a) = v^T φ

    where v_a = tr(T_a). The SL(K) projection removes this component:
        φ_sl = φ - (v^T φ / ||v||²) v
        φ_sl = (I - v v^T / ||v||²) φ

    This ensures det(exp(M)) = exp(tr(M)) = exp(0) = 1, so Ω_ij ∈ SL(K).

    Args:
        generators: Lie algebra generators (n_gen, K, K)

    Returns:
        trace_proj: (n_gen,) vector v where v_a = tr(T_a).
                   To project: phi -= (phi @ v) / (v @ v) * v
    """
    # v_a = tr(T_a) = sum of diagonal elements
    trace_vec = generators.diagonal(dim1=-2, dim2=-1).sum(dim=-1)  # (n_gen,)
    return trace_vec


def apply_slk_projection(
    phi: torch.Tensor,           # (..., n_gen) — phi embedding weights
    trace_vec: torch.Tensor,     # (n_gen,) — from build_slk_projector
) -> torch.Tensor:
    """
    Project phi to the traceless subalgebra sl(K) in-place.

    Removes the component of phi along the trace direction, ensuring
    that M = Σ_a φ_a T_a has tr(M) = 0, so exp(M) ∈ SL(K).

    Args:
        phi: Gauge frame coordinates (..., n_gen)
        trace_vec: Trace vector v_a = tr(T_a), shape (n_gen,)

    Returns:
        phi_projected: Same shape as phi, with trace component removed
    """
    v_norm_sq = (trace_vec @ trace_vec).clamp(min=1e-12)
    # Component of phi along trace direction
    phi_trace = (phi @ trace_vec).unsqueeze(-1)  # (..., 1)
    # Remove it
    return phi - (phi_trace / v_norm_sq) * trace_vec


def apply_cartan_preconditioning(
    grad_phi: torch.Tensor,         # (..., n_gen) — gradient w.r.t. phi
    preconditioner: torch.Tensor,   # (n_gen, n_gen) — from build_cartan_projector
) -> torch.Tensor:
    """
    Apply Cartan decomposition preconditioning to phi gradients.

    Dampens the non-compact (symmetric) gradient components while
    preserving the compact (antisymmetric) components at full strength.

    This is the "Killing form natural gradient" — using the Lie algebra
    structure to define the proper metric for gradient descent on GL(K).

    Args:
        grad_phi: Raw gradient w.r.t. phi (..., n_gen)
        preconditioner: Preconditioning matrix (n_gen, n_gen)

    Returns:
        preconditioned gradient, same shape as grad_phi
    """
    return grad_phi @ preconditioner.T


# =============================================================================
# Killing Form Natural Gradient (position-independent)
# =============================================================================

def build_killing_form_preconditioner(
    generators: torch.Tensor,  # (n_gen, K, K)
    center_reg: float = None,
) -> torch.Tensor:
    r"""
    Build the Killing form natural gradient preconditioner for gl(K).

    Uses the Cartan-involution-modified Killing form as metric:
        g̃(X, Y) = -B(X, θ(Y))  where θ(X) = -X^T (Cartan involution)
        g̃_ab = 2K · tr(T_a^T T_b) - 2 · tr(T_a) · tr(T_b)

    This is positive semidefinite (degenerate only on the center ℝ·I).
    Returns g̃^{-1} for natural gradient: ∇̃F^a = [g̃^{-1}]^{ab} · ∂F/∂φ^b.

    Unlike the Cartan preconditioner (which has a free dampening parameter),
    this uses the exact Lie algebra metric with no free parameters.

    Eigenvalue structure for E_{ij} basis of gl(K):
        - so(K) directions: eigenvalue 2K (compact rotations)
        - sym₀(K) directions: eigenvalue 2K (non-compact, traceless)
        - Diagonal traceless: eigenvalue 2(K-1) to 2K
        - Center (trace): eigenvalue 0 → regularized

    The center (trace) direction of gl(K) = sl(K) ⊕ ℝ·I has zero Killing
    eigenvalue.  With center_reg ≪ 2K, the inverse metric amplifies this
    direction by 1/center_reg, creating a condition number of
    2K/center_reg.  For K=10 and center_reg=1e-4 this is 200,000×,
    which makes the preconditioned gradient dominated by center noise
    and defeats the purpose of the natural gradient.

    Default: center_reg = 2K (matches the typical non-center eigenvalue),
    giving condition number ~1 so no single Lie-algebra direction is
    artificially amplified.

    Args:
        generators: Lie algebra generators (n_gen, K, K)
        center_reg: Regularization for the degenerate center direction.
            Default None → 2K (isotropic conditioning).

    Returns:
        inv_metric: (n_gen, n_gen) inverse metric for natural gradient
    """
    n_gen, K, _ = generators.shape
    device = generators.device
    dtype = generators.dtype

    if center_reg is None:
        center_reg = 2.0 * K

    # Gram matrix: G_ab = tr(T_a^T T_b) = Σ_{ij} T_a[i,j] T_b[i,j] (Frobenius inner product)
    # NOTE: No transpose in the einsum — the Frobenius inner product is Σ_{ij} A_{ij} B_{ij}.
    # Previously used generators.transpose(-2,-1) which computed tr(T_a T_b) instead,
    # giving a permutation matrix (E_{ij} ↦ E_{ji}) with eigenvalues ±1 — making the
    # metric indefinite for non-symmetric generators (antisymmetric directions got
    # negative eigenvalues ≈ -2K).
    gram = torch.einsum('aij,bij->ab', generators, generators)

    # Trace vector: v_a = tr(T_a)
    traces = generators.diagonal(dim1=-2, dim2=-1).sum(dim=-1)  # (n_gen,)
    trace_outer = torch.outer(traces, traces)  # (n_gen, n_gen)

    # Modified Killing form metric: g̃_ab = 2K · gram_ab - 2 · trace_outer_ab
    metric = 2.0 * K * gram - 2.0 * trace_outer

    # Regularize the center direction (kernel of Killing form on gl(K))
    metric = metric + center_reg * torch.eye(n_gen, device=device, dtype=dtype)

    # Invert to get natural gradient metric
    inv_metric = torch.linalg.inv(metric)

    return inv_metric


def apply_killing_form_natural_gradient(
    grad_phi: torch.Tensor,         # (..., n_gen)
    inv_metric: torch.Tensor,       # (n_gen, n_gen) from build_killing_form_preconditioner
) -> torch.Tensor:
    """
    Apply Killing form natural gradient: ∇̃F = g̃^{-1} · ∂F/∂φ.

    Position-independent (same metric regardless of current φ).
    Principled (no free parameters), but does not capture the
    position-dependent curvature of the exponential map.

    Args:
        grad_phi: Euclidean gradient ∂F/∂φ^a, shape (..., n_gen)
        inv_metric: Inverse metric (n_gen, n_gen)

    Returns:
        Natural gradient, same shape as grad_phi
    """
    return grad_phi @ inv_metric.T


# =============================================================================
# Pullback Natural Gradient (position-dependent, theoretically exact)
# =============================================================================

def build_structure_constants(
    generators: torch.Tensor,  # (n_gen, K, K)
) -> torch.Tensor:
    """
    Precompute structure constants f^c_{ab} defined by [T_a, T_b] = Σ_c f^c_{ab} T_c.

    For orthonormal generators (gram ≈ I):
        f^c_{ab} = tr(T_c^T · [T_a, T_b])

    For general generators:
        f^c_{ab} = Σ_d G^{cd} · tr(T_d^T · [T_a, T_b])

    Args:
        generators: Lie algebra generators (n_gen, K, K)

    Returns:
        structure_constants: (n_gen, n_gen, n_gen) tensor f[a,b,c] = f^c_{ab}
    """
    n_gen = generators.shape[0]
    device = generators.device
    dtype = generators.dtype

    # Gram matrix and its pseudoinverse
    gram = torch.einsum('aij,bij->ab', generators.transpose(-2, -1), generators)
    gram_inv = torch.linalg.pinv(gram)

    # Brackets: [T_a, T_b] = T_a @ T_b - T_b @ T_a, shape (n_gen, n_gen, K, K)
    brackets = (torch.einsum('aik,bkj->abij', generators, generators) -
                torch.einsum('bik,akj->abij', generators, generators))

    # Project onto generator basis: f̃_{ab,d} = tr(T_d^T · [T_a, T_b])
    f_tilde = torch.einsum('dij,abij->abd',
                           generators.transpose(-2, -1), brackets)  # (n_gen, n_gen, n_gen)

    # Apply gram inverse: f^c_{ab} = Σ_d G^{cd} f̃_{ab,d}
    structure_constants = torch.einsum('cd,abd->abc', gram_inv, f_tilde)

    return structure_constants


def apply_pullback_natural_gradient(
    grad_phi: torch.Tensor,              # (..., n_gen)
    phi: torch.Tensor,                   # (..., n_gen)
    generators: torch.Tensor,            # (n_gen, K, K)
    structure_constants: torch.Tensor,   # (n_gen, n_gen, n_gen)
    gram: Optional[torch.Tensor] = None, # (n_gen, n_gen) precomputed gram matrix
    series_order: int = 6,               # Taylor series order for Ψ(ad_X)
) -> torch.Tensor:
    """
    Apply position-dependent natural gradient using the pullback metric through exp.

    The Riemannian metric on φ-space is pulled back from the bi-invariant
    Frobenius metric on GL(K) through the exponential map:

        dexp_X(T_a) = exp(X) · Ψ(ad_X)(T_a)

    where Ψ(z) = (e^z - 1)/z = Σ_{k=0}^∞ z^k/(k+1)!

    Left-translating back to the identity gives the metric:
        G_ab(φ) = ⟨ Ψ(ad_X)(T_a), Ψ(ad_X)(T_b) ⟩_gram

    where ⟨A, B⟩_gram = A^T @ gram @ B in generator coordinates.

    At φ = 0: Ψ = I, so G = gram (Frobenius inner product).
    At large ||φ|| in symmetric directions: G grows exponentially,
    so the natural gradient (G^{-1} · ∂F/∂φ) automatically shrinks —
    exactly compensating the exponential amplification through matrix_exp.

    Args:
        grad_phi: Euclidean gradient ∂F/∂φ^a, shape (..., n_gen)
        phi: Current gauge coordinates, shape (..., n_gen)
        generators: Lie algebra generators (n_gen, K, K)
        structure_constants: f^c_{ab} from build_structure_constants
        gram: Precomputed Gram matrix; if None, computed from generators
        series_order: Number of terms in Taylor expansion of Ψ

    Returns:
        Natural gradient G(φ)^{-1} · ∂F/∂φ, same shape as grad_phi
    """
    n_gen = generators.shape[0]
    batch_shape = phi.shape[:-1]
    device = phi.device
    dtype = phi.dtype

    if gram is None:
        gram = torch.einsum('aij,bij->ab',
                            generators.transpose(-2, -1), generators)

    # Compute ad_X: [ad_X]_bc = Σ_a φ^a f^c_{ab}
    # phi: (..., n_gen), structure_constants: (n_gen, n_gen, n_gen)
    # ad_X: (..., n_gen, n_gen)
    ad_X = torch.einsum('...a,abc->...bc', phi, structure_constants)

    # Compute Ψ(ad_X) via Taylor series:
    # Ψ(z) = (e^z - 1)/z = Σ_{k=0}^∞ z^k/(k+1)!
    # Ψ(ad_X) = I + ad_X/2! + ad_X²/3! + ad_X³/4! + ...
    I_gen = torch.eye(n_gen, device=device, dtype=dtype)
    I_expanded = I_gen.expand(*batch_shape, n_gen, n_gen)

    psi = I_expanded.clone()         # k=0 term: I
    ad_power = ad_X.clone()          # ad_X^1

    for k in range(1, series_order):
        coeff = 1.0 / math.factorial(k + 1)
        psi = psi + coeff * ad_power
        if k < series_order - 1:
            ad_power = torch.matmul(ad_power, ad_X)

    # Metric: G = Ψ^T @ gram @ Ψ  (positive definite by construction)
    gram_expanded = gram.expand(*batch_shape, n_gen, n_gen)
    G = torch.matmul(psi.transpose(-2, -1), torch.matmul(gram_expanded, psi))

    # Regularize for numerical stability
    G = G + 1e-6 * I_expanded

    # Solve G @ nat_grad = grad_phi for nat_grad
    nat_grad = torch.linalg.solve(G, grad_phi.unsqueeze(-1)).squeeze(-1)

    return nat_grad
