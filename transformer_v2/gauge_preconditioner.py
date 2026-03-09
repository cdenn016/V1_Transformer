# -*- coding: utf-8 -*-
"""
Gauge-Theoretic Gradient Preconditioning
=========================================

Principled gradient preconditioning for gauge frame (φ) parameters
based on the Cartan decomposition of gl(K).

Background:
    gl(K) = so(K) ⊕ sym(K)
    - so(K): compact. ||d exp(X)/dX|| = O(1).
    - sym(K): non-compact. ||d exp(X)/dX|| ~ exp(||X||).

Four preconditioning modes (phi_natural_gradient config):
    1. 'clip': Simple norm clipping. No geometric awareness.
    2. 'cartan': Approximate Cartan decomposition. Dampens sym(K) by fixed factor.
    3. 'killing': Killing form natural gradient. No free parameters.
    4. 'pullback': Full pullback natural gradient (position-dependent, exact).
"""

import math
import torch
from typing import Optional


def build_cartan_projector(
    generators: torch.Tensor,
    sym_dampening: float = 0.1,
) -> torch.Tensor:
    """Build Cartan decomposition preconditioner for gl(K) phi gradients.

    Decomposes gl(K) = so(K) ⊕ sym(K) and dampens sym(K) directions.

    Args:
        generators: (n_gen, K, K) Lie algebra generators
        sym_dampening: Dampening for symmetric directions (0.1 = 10× dampening)

    Returns:
        preconditioner: (n_gen, n_gen) matrix
    """
    n_gen, K, _ = generators.shape
    device = generators.device
    dtype = generators.dtype

    gram = torch.einsum('aij,bij->ab', generators.transpose(-2, -1), generators)
    trace_prod = torch.einsum('aij,bji->ab', generators, generators)

    gram_inv = torch.linalg.pinv(gram)
    P_sym = gram_inv @ (gram + trace_prod) / 2.0

    I_gen = torch.eye(n_gen, device=device, dtype=dtype)
    return I_gen - (1.0 - sym_dampening) * P_sym


def build_slk_projector(generators: torch.Tensor) -> torch.Tensor:
    """Build projector to remove trace component of phi (restrict to sl(K)).

    Returns:
        trace_vec: (n_gen,) where v_a = tr(T_a)
    """
    return generators.diagonal(dim1=-2, dim2=-1).sum(dim=-1)


def apply_slk_projection(
    phi: torch.Tensor,
    trace_vec: torch.Tensor,
) -> torch.Tensor:
    """Project phi to traceless subalgebra sl(K)."""
    v_norm_sq = (trace_vec @ trace_vec).clamp(min=1e-12)
    phi_trace = (phi @ trace_vec).unsqueeze(-1)
    return phi - (phi_trace / v_norm_sq) * trace_vec


def apply_cartan_preconditioning(
    grad_phi: torch.Tensor,
    preconditioner: torch.Tensor,
) -> torch.Tensor:
    """Apply Cartan decomposition preconditioning to phi gradients."""
    return grad_phi @ preconditioner.T


def build_killing_form_preconditioner(
    generators: torch.Tensor,
    center_reg: float = 1e-4,
) -> torch.Tensor:
    """Build Killing form natural gradient preconditioner for gl(K).

    Uses the Cartan-involution-modified Killing form:
        g̃_ab = 2K · tr(T_a^T T_b) - 2 · tr(T_a) · tr(T_b)

    No free parameters — metric determined by algebra structure.

    Returns:
        inv_metric: (n_gen, n_gen) inverse metric for natural gradient
    """
    n_gen, K, _ = generators.shape
    device = generators.device
    dtype = generators.dtype

    gram = torch.einsum('aij,bij->ab', generators.transpose(-2, -1), generators)
    traces = generators.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    trace_outer = torch.outer(traces, traces)

    metric = 2.0 * K * gram - 2.0 * trace_outer
    metric = metric + center_reg * torch.eye(n_gen, device=device, dtype=dtype)

    return torch.linalg.inv(metric)


def apply_killing_form_natural_gradient(
    grad_phi: torch.Tensor,
    inv_metric: torch.Tensor,
) -> torch.Tensor:
    """Apply Killing form natural gradient: ∇̃F = g̃⁻¹ · ∂F/∂φ."""
    return grad_phi @ inv_metric.T


def build_structure_constants(generators: torch.Tensor) -> torch.Tensor:
    """Precompute structure constants f^c_{ab} from [T_a, T_b] = Σ_c f^c_{ab} T_c.

    Returns:
        structure_constants: (n_gen, n_gen, n_gen) tensor
    """
    n_gen = generators.shape[0]

    gram = torch.einsum('aij,bij->ab', generators.transpose(-2, -1), generators)
    gram_inv = torch.linalg.pinv(gram)

    brackets = (torch.einsum('aik,bkj->abij', generators, generators) -
                torch.einsum('bik,akj->abij', generators, generators))

    f_tilde = torch.einsum('dij,abij->abd', generators.transpose(-2, -1), brackets)
    return torch.einsum('cd,abd->abc', gram_inv, f_tilde)


def apply_pullback_natural_gradient(
    grad_phi: torch.Tensor,
    phi: torch.Tensor,
    generators: torch.Tensor,
    structure_constants: torch.Tensor,
    gram: Optional[torch.Tensor] = None,
    series_order: int = 6,
) -> torch.Tensor:
    """Apply position-dependent natural gradient using pullback metric through exp.

    The metric on φ-space is:
        G_ab(φ) = ⟨ Ψ(ad_X)(T_a), Ψ(ad_X)(T_b) ⟩_gram

    where Ψ(z) = (e^z - 1)/z = Σ_{k=0}^∞ z^k/(k+1)!

    At φ = 0: G = gram. At large ||φ|| in sym directions: G grows
    exponentially, automatically compensating matrix_exp amplification.

    Args:
        grad_phi: Euclidean gradient (..., n_gen)
        phi: Current gauge coordinates (..., n_gen)
        generators: (n_gen, K, K) generators
        structure_constants: f^c_{ab} from build_structure_constants
        gram: Optional precomputed Gram matrix
        series_order: Taylor series terms for Ψ

    Returns:
        Natural gradient G(φ)⁻¹ · ∂F/∂φ
    """
    n_gen = generators.shape[0]
    batch_shape = phi.shape[:-1]
    device = phi.device
    dtype = phi.dtype

    if gram is None:
        gram = torch.einsum('aij,bij->ab', generators.transpose(-2, -1), generators)

    ad_X = torch.einsum('...a,abc->...bc', phi, structure_constants)

    I_gen = torch.eye(n_gen, device=device, dtype=dtype)
    I_expanded = I_gen.expand(*batch_shape, n_gen, n_gen)

    psi = I_expanded.clone()
    ad_power = ad_X.clone()

    for k in range(1, series_order):
        coeff = 1.0 / math.factorial(k + 1)
        psi = psi + coeff * ad_power
        if k < series_order - 1:
            ad_power = torch.matmul(ad_power, ad_X)

    gram_expanded = gram.expand(*batch_shape, n_gen, n_gen)
    G = torch.matmul(psi.transpose(-2, -1), torch.matmul(gram_expanded, psi))
    G = G + 1e-6 * I_expanded

    return torch.linalg.solve(G, grad_phi.unsqueeze(-1)).squeeze(-1)
