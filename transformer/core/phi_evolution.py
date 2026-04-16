"""
Phi (gauge frame) and Omega (direct group element) evolution utilities.

Extracted from variational_ffn.py to decouple gauge-frame math from the
VFE E-step orchestrator. These functions are stateless — they take tensors
and config scalars, not the VariationalFFNDynamic instance.

Used by both transformer/core/variational_ffn.py and transformer/vfe/e_step.py.
"""

from typing import Optional, List
import torch

from transformer.core.gauge_preconditioner import (
    apply_cartan_preconditioning,
    apply_killing_form_natural_gradient,
    apply_pullback_natural_gradient,
)


def precondition_phi_gradient(
    grad_phi: torch.Tensor,
    phi: torch.Tensor,
    mode: str,
    preconditioner: Optional[torch.Tensor] = None,
    generators: Optional[torch.Tensor] = None,
    structure_constants: Optional[torch.Tensor] = None,
    gram: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Apply phi gradient preconditioning based on selected mode.

    Modes:
        'clip': Simple norm clipping to 10.0 (no geometric awareness)
        'cartan': Cartan decomposition with fixed sym_dampening=0.1
        'killing': Killing form natural gradient (position-independent, no free params)
        'pullback': Full pullback metric through exp (position-dependent, exact)

    Args:
        grad_phi: Raw Euclidean gradient :math:`\partial F/\partial\phi^a`, shape ``(..., n_gen)``.
        phi: Current gauge frame coordinates, shape ``(..., n_gen)``.
            Needed only for 'pullback' mode.
        mode: Preconditioning mode string.
        preconditioner: Precomputed preconditioner matrix for 'cartan' or 'killing' modes.
        generators: Lie algebra generators ``(n_gen, K, K)``. Needed for 'pullback'.
        structure_constants: Structure constants. Needed for 'pullback'.
        gram: Gram matrix. Needed for 'pullback'.

    Returns:
        Preconditioned gradient, same shape as grad_phi.
    """
    if mode == 'cartan':
        return apply_cartan_preconditioning(grad_phi, preconditioner)

    elif mode == 'killing':
        return apply_killing_form_natural_gradient(grad_phi, preconditioner)

    elif mode == 'pullback':
        return apply_pullback_natural_gradient(
            grad_phi, phi, generators,
            structure_constants, gram,
        )

    else:  # 'clip' (default)
        grad_phi_norm = torch.norm(grad_phi, dim=-1, keepdim=True)
        return torch.where(
            grad_phi_norm > 10.0,
            grad_phi * 10.0 / (grad_phi_norm + 1e-6),
            grad_phi
        )


def retract_omega(
    omega: torch.Tensor,
    grad_omega: torch.Tensor,
    step_size: float,
    trust_region: float = 0.3,
    irrep_dims: Optional[List[int]] = None,
) -> torch.Tensor:
    r"""Retract Omega update on GL(K) via left-invariant Lie algebra step.

    Computes the natural gradient in the Lie algebra gl(K) using the
    left-invariant pullback:

    .. math::

        \xi = \Omega^{-1} \cdot \frac{\partial F}{\partial\Omega}

    then clips :math:`\|\xi\|_F \leq` ``trust_region`` and applies
    first-order Euler retraction:

    .. math::

        \Omega_{\text{new}} = \Omega \cdot \exp(-\eta\,\xi)
                            \approx \Omega - \eta\,(\Omega \cdot \xi)

    The Riemannian norm :math:`\|\xi\|_F = \|\Omega^{-1}\,\text{grad}\|_F`
    is left-invariant under :math:`\Omega \to A\Omega`, so the trust region
    bounds intrinsic step size.

    When ``irrep_dims`` is set, processes each head block independently to
    avoid O(K^3) matmuls on the full K x K matrix.

    Args:
        omega: Current group elements ``(B, N, K, K)``.
        grad_omega: Euclidean gradient ``(B, N, K, K)``.
        step_size: Learning rate.
        trust_region: Max Riemannian step size.
        irrep_dims: Block dimensions for block-diagonal processing.

    Returns:
        Updated group elements ``(B, N, K, K)``.
    """
    _ridge = 1e-6

    if irrep_dims is not None:
        omega_new = omega.clone()
        block_start = 0
        for d in irrep_dims:
            block_end = block_start + d
            om_blk = omega[:, :, block_start:block_end, block_start:block_end]
            gr_blk = grad_omega[:, :, block_start:block_end, block_start:block_end]

            _eye = torch.eye(d, device=om_blk.device, dtype=om_blk.dtype)
            om_reg = om_blk + _ridge * _eye
            try:
                xi_blk = torch.linalg.solve(om_reg, gr_blk)
            except (torch.linalg.LinAlgError, RuntimeError):
                xi_blk = torch.linalg.pinv(om_reg) @ gr_blk

            xi_norm = xi_blk.flatten(-2).norm(dim=-1, keepdim=True).unsqueeze(-1)
            scale = torch.clamp(trust_region / (xi_norm + 1e-8), max=1.0)
            xi_blk = xi_blk * scale

            omega_new[:, :, block_start:block_end, block_start:block_end] = (
                om_blk - step_size * (om_blk @ xi_blk)
            )
            block_start = block_end
        return omega_new

    # Fallback: full K×K retraction
    K = omega.shape[-1]
    _eye = torch.eye(K, device=omega.device, dtype=omega.dtype)
    om_reg = omega + _ridge * _eye
    try:
        xi = torch.linalg.solve(om_reg, grad_omega)
    except (torch.linalg.LinAlgError, RuntimeError):
        xi = torch.linalg.pinv(om_reg) @ grad_omega

    xi_norm = xi.flatten(-2).norm(dim=-1, keepdim=True).unsqueeze(-1)
    scale = torch.clamp(trust_region / (xi_norm + 1e-8), max=1.0)
    xi = xi * scale

    return omega - step_size * (omega @ xi)
