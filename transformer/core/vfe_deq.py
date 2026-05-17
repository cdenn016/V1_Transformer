"""
DEQ Fixed-Point Implicit Differentiation
=========================================

Extracted from ``variational_ffn.py`` to keep the "side-quest" DEQ
machinery in its own focused module.  Activated by ``use_deq=True`` on
``VariationalFFNDynamic``.

Mathematical background
-----------------------
In a standard truncated E-step with N iterations, the backward pass
computes ``∂z_N/∂θ`` where ``z_N`` is the N-th iterate of the natural
gradient descent.  This is the "straight-through" M-step gradient: it
ignores the fact that at the fixed point ``z* = T(z*)``, the true
derivative is

    ``∂z*/∂θ = (I − J_T)^{−1} · ∂T/∂θ``

where ``J_T = ∂T/∂z`` is the Jacobian of the E-step operator evaluated
at the fixed point.  The DEQ forward is an identity pass (the E-step
loop has already converged); the DEQ backward replaces the
straight-through gradient with a Neumann-series approximation of
``(I − J_T)^{−1}``:

    ``(I − J_T^T)^{−1} v ≈ v + J_T^T v + (J_T^T)^2 v + …  (K terms)``

Each term is one vjp through the E-step closure evaluated at the fixed
point.  The VJPs are computed with ``torch.autograd.grad`` on a freshly
detached leaf for ``(mu, sigma[, phi])``.

Divergence safeguards
---------------------
The Neumann series converges only when ``||J_T|| < 1``.  In practice the
trust-region-clipped natural gradient keeps this true near the fixed
point, but the backward loop includes two guards:

1. **NaN / non-finite check**: if any vjp produces a non-finite value,
   abort the sum.  A single NaN would otherwise poison every downstream
   M-step parameter.
2. **Norm cap**: if the cumulative vjp exceeds ``_DEQ_VJP_NORM_CAP``
   (default ``1e4``), rescale it back to the cap.  This biases the
   backward when truly needed but prevents silent geometric divergence.

Public API
----------
- :class:`DEQFixedPoint` — autograd Function for (μ, Σ) fixed points
- :class:`DEQFixedPointFull` — autograd Function for joint (μ, Σ, φ)
  fixed points (used when ``deq_include_phi=True``)
- :func:`make_deq_step_fn` — factory that builds the (μ, Σ) step closure
  given an FFN instance
- :func:`make_deq_step_fn_with_phi` — factory that builds the joint
  (μ, Σ, φ) step closure including a differentiable phi retraction

The two Function classes are re-exported from ``variational_ffn`` for
backward compatibility.

Mutual exclusions
-----------------
``use_deq=True`` is incompatible with:

- ``active_inference=True`` — the DEQ Jacobian is built from the VFE-only
  step operator, not the VFE+EFE composite.  Enforced in
  ``wire_readout_references``.
- ``closed_form_e_step=True`` — DEQ requires an iterative E-step to
  build the step closure; closed-form bypasses the loop.  Not strictly
  enforced but functionally incompatible.
"""

from typing import TYPE_CHECKING, Optional

import torch
import torch.nn.functional as F

from transformer.core.attention import compute_attention_weights
from transformer.core.transport_ops import compute_transport_operators
from transformer.core.vfe_gradients import (
    compute_vfe_gradients_gpu,
    compute_natural_gradient_gpu,
)
from math_utils.numerical_monitor import record as _nr
from transformer.core.vfe_utils import (
    retract_spd_torch,
    retract_spd_diagonal_torch,
)

if TYPE_CHECKING:
    # Forward reference only — avoids a circular import at runtime
    from transformer.core.variational_ffn import VariationalFFNDynamic


__all__ = [
    "DEQFixedPoint",
    "DEQFixedPointFull",
    "make_deq_step_fn",
    "make_deq_step_fn_with_phi",
]


# Divergence cap for Neumann series terms.  The forward E-step clips its
# natural gradient to ||·|| ≤ 500 inside e_step_fn, so any vjp term with a
# norm much larger than that signals the series is growing.  We cap each
# cumulative v_* at _DEQ_VJP_NORM_CAP and abort if NaN appears.
_DEQ_VJP_NORM_CAP: float = 1e4


# =============================================================================
# Autograd Functions
# =============================================================================

class DEQFixedPoint(torch.autograd.Function):
    """Implicit differentiation through E-step fixed point via Neumann series.

    Forward: identity (fixed point already computed by the E-step loop).
    Backward: corrects gradients using (I - J)^{-1} ≈ I + J + J² + ... (K terms)
    where J is the Jacobian of one E-step evaluated at the fixed point.
    """

    # Re-exported so external code that reads DEQFixedPoint._DEQ_VJP_NORM_CAP
    # (as the previous implementation did) continues to work.
    _DEQ_VJP_NORM_CAP = _DEQ_VJP_NORM_CAP

    @staticmethod
    def forward(ctx, mu_star, sigma_star, e_step_fn, n_steps, neumann_terms):
        ctx.save_for_backward(mu_star.detach(), sigma_star.detach())
        ctx.e_step_fn = e_step_fn
        ctx.neumann_terms = neumann_terms
        return mu_star, sigma_star

    @staticmethod
    def backward(ctx, grad_mu, grad_sigma):
        mu_star, sigma_star = ctx.saved_tensors
        e_step_fn = ctx.e_step_fn
        K = ctx.neumann_terms
        _cap = _DEQ_VJP_NORM_CAP

        # Neumann series: (I - J^T)^{-1} v ≈ v + J^T v + (J^T)^2 v + ...
        v_mu = grad_mu.clone()
        v_sigma = grad_sigma.clone()
        total_mu = grad_mu.clone()
        total_sigma = grad_sigma.clone()

        for _ in range(K):
            # One vjp through the E-step at the fixed point
            mu_in = mu_star.detach().requires_grad_(True)
            sigma_in = sigma_star.detach().requires_grad_(True)
            with torch.enable_grad():
                mu_out, sigma_out = e_step_fn(mu_in, sigma_in)
            jt_v = torch.autograd.grad(
                outputs=[mu_out, sigma_out],
                inputs=[mu_in, sigma_in],
                grad_outputs=[v_mu, v_sigma],
                retain_graph=False,
                allow_unused=True,
            )
            v_mu = jt_v[0] if jt_v[0] is not None else torch.zeros_like(grad_mu)
            v_sigma = jt_v[1] if jt_v[1] is not None else torch.zeros_like(grad_sigma)

            # Divergence guard: abort the Neumann sum if any vjp produces
            # non-finite values.  Without this, a single NaN contaminates the
            # rest of the backward and poisons every M-step parameter.
            if not (torch.isfinite(v_mu).all() and torch.isfinite(v_sigma).all()):
                _nr("deq_neumann_nonfinite")
                break

            # Norm cap: if ||J^T|| > 1 the Neumann series diverges and the
            # cumulative term grows geometrically.  Rescale any iterate that
            # exceeds the cap so the backward pass remains bounded.  This
            # biases the backward when truly needed but is preferable to
            # silent divergence.
            mu_n = torch.linalg.norm(v_mu)
            if mu_n > _cap:
                _nr("deq_neumann_norm_cap")
                import logging
                logging.getLogger(__name__).warning(
                    f"DEQ Neumann norm cap activated: ||v_mu||={mu_n:.2e} > {_cap:.0e}. "
                    f"rho(J) likely >= 1; M-step gradient is biased this step."
                )
                v_mu = v_mu * (_cap / mu_n)
            sig_n = torch.linalg.norm(v_sigma)
            if sig_n > _cap:
                _nr("deq_neumann_norm_cap")
                import logging
                logging.getLogger(__name__).warning(
                    f"DEQ Neumann norm cap activated: ||v_sigma||={sig_n:.2e} > {_cap:.0e}. "
                    f"rho(J) likely >= 1; M-step gradient is biased this step."
                )
                v_sigma = v_sigma * (_cap / sig_n)

            total_mu = total_mu + v_mu
            total_sigma = total_sigma + v_sigma

        return total_mu, total_sigma, None, None, None


class DEQFixedPointFull(torch.autograd.Function):
    r"""Implicit differentiation through joint (μ, Σ, φ) fixed point.

    Extends DEQFixedPoint to include gauge frame φ in the fixed-point
    variables.  At the E-step fixed point, all VFE gradients vanish:
        ∂F/∂μ = 0,  ∂F/∂Σ = 0,  ∂F/∂φ = 0.

    The IFT gives the exact M-step gradient:
        ∂z*/∂θ = −(∂²F/∂z²)⁻¹ · ∂²F/∂z∂θ
    where z = (μ, Σ, φ) and θ are model parameters (embeddings, etc.).

    We approximate (I − J)⁻¹ via Neumann series:
        (I − J^T)⁻¹ v ≈ v + J^T v + (J^T)² v + ⋯  (K terms)
    where J is the Jacobian of one full E-step (μ, Σ, φ) → (μ', Σ', φ')
    evaluated at the fixed point.

    This corrects the straight-through bias in the M-step φ gradient:
    instead of ∂φ*/∂φ_init ≈ I, we get the IFT-corrected Jacobian that
    accounts for how the E-step trajectory depends on initial conditions.
    """

    @staticmethod
    def forward(ctx, mu_star, sigma_star, phi_star, e_step_fn, n_steps, neumann_terms):
        ctx.save_for_backward(
            mu_star.detach(), sigma_star.detach(), phi_star.detach(),
        )
        ctx.e_step_fn = e_step_fn
        ctx.neumann_terms = neumann_terms
        return mu_star, sigma_star, phi_star

    @staticmethod
    def backward(ctx, grad_mu, grad_sigma, grad_phi):
        mu_star, sigma_star, phi_star = ctx.saved_tensors
        e_step_fn = ctx.e_step_fn
        K = ctx.neumann_terms
        _cap = _DEQ_VJP_NORM_CAP

        # Neumann series: (I - J^T)^{-1} v ≈ v + J^T v + (J^T)^2 v + ...
        v_mu = grad_mu.clone()
        v_sigma = grad_sigma.clone()
        v_phi = grad_phi.clone()
        total_mu = grad_mu.clone()
        total_sigma = grad_sigma.clone()
        total_phi = grad_phi.clone()

        for _ in range(K):
            mu_in = mu_star.detach().requires_grad_(True)
            sigma_in = sigma_star.detach().requires_grad_(True)
            phi_in = phi_star.detach().requires_grad_(True)
            with torch.enable_grad():
                mu_out, sigma_out, phi_out = e_step_fn(mu_in, sigma_in, phi_in)
            jt_v = torch.autograd.grad(
                outputs=[mu_out, sigma_out, phi_out],
                inputs=[mu_in, sigma_in, phi_in],
                grad_outputs=[v_mu, v_sigma, v_phi],
                retain_graph=False,
                allow_unused=True,
            )
            v_mu = jt_v[0] if jt_v[0] is not None else torch.zeros_like(grad_mu)
            v_sigma = jt_v[1] if jt_v[1] is not None else torch.zeros_like(grad_sigma)
            v_phi = jt_v[2] if jt_v[2] is not None else torch.zeros_like(grad_phi)

            # Divergence guard + norm cap (see DEQFixedPoint.backward above
            # for the rationale).  A single non-finite vjp poisons every
            # downstream M-step parameter; rescale any iterate that exceeds
            # the cap to prevent silent geometric growth when ||J^T|| > 1.
            if not (torch.isfinite(v_mu).all()
                    and torch.isfinite(v_sigma).all()
                    and torch.isfinite(v_phi).all()):
                _nr("deq_neumann_nonfinite")
                break
            mu_n = torch.linalg.norm(v_mu)
            if mu_n > _cap:
                _nr("deq_neumann_norm_cap")
                v_mu = v_mu * (_cap / mu_n)
            sig_n = torch.linalg.norm(v_sigma)
            if sig_n > _cap:
                _nr("deq_neumann_norm_cap")
                v_sigma = v_sigma * (_cap / sig_n)
            phi_n = torch.linalg.norm(v_phi)
            if phi_n > _cap:
                _nr("deq_neumann_norm_cap")
                v_phi = v_phi * (_cap / phi_n)

            total_mu = total_mu + v_mu
            total_sigma = total_sigma + v_sigma
            total_phi = total_phi + v_phi

        return total_mu, total_sigma, total_phi, None, None, None


# =============================================================================
# Step-function factories
# =============================================================================
# These build closures that perform one VFE natural gradient iteration with
# fully autograd-tracked operations.  The closures are handed to
# DEQFixedPoint.apply / DEQFixedPointFull.apply which VJP through them to
# compute the Neumann-series IFT correction.
#
# Both factories take the FFN instance as first argument so the closures can
# access the ~20 instance attributes that drive a single E-step iteration
# (irrep_dims, gauge_mode, lambda_belief, generators, etc.).  This mirrors
# the pattern used by active_inference.py for compute_ai_gradients(ffn, ...).

def make_deq_step_fn(
    ffn: "VariationalFFNDynamic",
    phi_current: torch.Tensor,
    mu_p_current: torch.Tensor,
    sigma_p: torch.Tensor,
    mask: Optional[torch.Tensor],
    is_diagonal: bool,
    eps: float,
    dtype: torch.dtype,
):
    """Create a differentiable (μ, Σ) E-step closure for DEQ backward.

    Returns a function ``(mu, sigma) -> (mu', sigma')`` that performs one
    VFE natural gradient step with autograd-tracked operations.  The
    closure captures ``ffn`` by reference; every attribute access uses
    ``ffn.attr`` rather than ``self.attr``.
    """
    def step_fn(mu_in, sigma_in):
        # Compute transport
        cached_transport = None

        alpha_eff = ffn.get_bayesian_alpha(mu_in, mu_p_current, sigma_p, sigma_in, eps=eps) if ffn.learnable_alpha else ffn.alpha
        _alpha_c0 = F.softplus(ffn.raw_c0) if ffn.learnable_alpha else None

        if ffn.multihead_vfe:
            # Differentiable multihead step (no detach)
            grad_mu = torch.zeros_like(mu_in)
            grad_sigma = torch.zeros_like(sigma_in)
            block_start = 0
            for h, d_h in enumerate(ffn.irrep_dims):
                block_end = block_start + d_h
                mu_h = mu_in[:, :, block_start:block_end].contiguous()
                mu_p_h = mu_p_current[:, :, block_start:block_end].contiguous()
                if is_diagonal:
                    sigma_h = sigma_in[:, :, block_start:block_end].contiguous()
                    sigma_p_h = sigma_p[:, :, block_start:block_end].contiguous()
                else:
                    sigma_h = sigma_in[:, :, block_start:block_end, block_start:block_end].contiguous()
                    sigma_p_h = sigma_p[:, :, block_start:block_end, block_start:block_end].contiguous()
                gen_h = ffn.generators[:, block_start:block_end, block_start:block_end]
                kappa_h = ffn._get_kappa_h(h, d_h)  # Match main path scaling

                beta_h = compute_attention_weights(
                    mu_q=mu_h, sigma_q=sigma_h,
                    phi=phi_current, generators=gen_h,
                    kappa=kappa_h, epsilon=eps, mask=mask,
                    return_kl=False,
                    diagonal_covariance=is_diagonal,
                        mask_self_attention=ffn.mask_self_attention,
                    gauge_mode=ffn.gauge_mode,
                    exact_diagonal_transport=ffn.exact_diagonal_transport,
                )
                # Slice alpha per head block if per-dim tensor
                alpha_h = alpha_eff[:, :, block_start:block_end] if isinstance(alpha_eff, torch.Tensor) and alpha_eff.dim() == 3 else alpha_eff
                c0_h = _alpha_c0[block_start:block_end] if _alpha_c0 is not None else None
                grad_mu_h, grad_sigma_h = compute_vfe_gradients_gpu(
                    mu_q=mu_h, sigma_q=sigma_h,
                    mu_p=mu_p_h, sigma_p=sigma_p_h,
                    beta=beta_h, phi=phi_current, generators=gen_h,
                    alpha=alpha_h, lambda_belief=ffn.lambda_belief, lambda_softmax=ffn.lambda_softmax,
                    kappa=kappa_h, eps=eps,
                    alpha_c0=c0_h,
                    compute_sigma_align_grad=ffn.compute_sigma_align_grad,
                    exact_diagonal_transport=ffn.exact_diagonal_transport,
                )
                grad_mu[:, :, block_start:block_end] = grad_mu_h
                if is_diagonal:
                    grad_sigma[:, :, block_start:block_end] = grad_sigma_h
                else:
                    if grad_sigma_h.dim() == 3 and d_h == 1:
                        grad_sigma_h = grad_sigma_h.unsqueeze(-1)
                    grad_sigma[:, :, block_start:block_end, block_start:block_end] = grad_sigma_h
                block_start = block_end
        grad_mu = torch.clamp(grad_mu, min=-1e3, max=1e3)
        grad_sigma = torch.clamp(grad_sigma, min=-1e3, max=1e3)

        nat_grad_mu, nat_grad_sigma = compute_natural_gradient_gpu(
            grad_mu, grad_sigma, sigma_in, eps=eps,
        )

        # Norm clipping for nat_grad_mu (was missing in DEQ path)
        nat_grad_mu_norm = torch.linalg.norm(nat_grad_mu, dim=-1, keepdim=True)
        nat_grad_mu = nat_grad_mu * torch.clamp(
            500.0 / (nat_grad_mu_norm + eps), max=1.0
        )

        # Norm clipping for nat_grad_sigma (matching main path Fix 1)
        if is_diagonal:
            nat_grad_sigma_norm = torch.linalg.norm(nat_grad_sigma, dim=-1, keepdim=True)
        else:
            nat_grad_sigma_norm = torch.linalg.norm(
                nat_grad_sigma.flatten(-2), dim=-1, keepdim=True
            ).unsqueeze(-1)
        nat_grad_sigma = nat_grad_sigma * torch.clamp(
            500.0 / (nat_grad_sigma_norm + eps), max=1.0
        )

        # mu update with trust region
        delta_mu = -ffn.lr * nat_grad_mu
        if is_diagonal:
            sigma_sqrt = torch.sqrt(sigma_in.float().clamp(min=eps)).to(dtype)
            whitened = delta_mu / sigma_sqrt
        else:
            sigma_diag = torch.diagonal(sigma_in, dim1=-2, dim2=-1).clone().float().clamp(min=eps)
            whitened = delta_mu / torch.sqrt(sigma_diag).to(dtype)

        w_norm = torch.linalg.norm(whitened, dim=-1, keepdim=True)
        scale = torch.clamp(2.0 / (w_norm + eps), max=1.0)
        mu_out = mu_in + scale * delta_mu

        # sigma update — calibrated step: ~0.1% max change per iter
        if ffn.update_sigma:
            sigma_trust = ffn._get_sigma_trust(ffn.lr)
            if is_diagonal:
                sigma_out = retract_spd_diagonal_torch(
                    sigma_diag=sigma_in, delta_sigma=-nat_grad_sigma,
                    step_size=1.0, trust_region=sigma_trust, eps=eps,
                    sigma_max=ffn.sigma_max,
                )
            else:
                sigma_out = retract_spd_torch(
                    Sigma=sigma_in, delta_Sigma=-nat_grad_sigma,
                    step_size=1.0, trust_region=sigma_trust * 0.5, eps=eps,
                    sigma_max=ffn.sigma_max,
                )
        else:
            sigma_out = sigma_in

        return mu_out, sigma_out

    return step_fn


def make_deq_step_fn_with_phi(
    ffn: "VariationalFFNDynamic",
    mu_p_current: torch.Tensor,
    sigma_p: torch.Tensor,
    mask: Optional[torch.Tensor],
    is_diagonal: bool,
    eps: float,
    dtype: torch.dtype,
):
    r"""Create a differentiable joint (μ, Σ, φ) E-step closure for DEQ backward.

    Returns a function ``(mu, sigma, phi) -> (mu', sigma', phi')`` that
    performs one VFE natural gradient step with full autograd tracking,
    including a differentiable phi update via Euclidean gradient descent
    on F_align.

    The phi update is a differentiable Euclidean step:
        φ' = φ − η_φ · ∂F_align/∂φ
    rather than the Lie group retraction used in the forward E-step.
    At the fixed point (where ∂F_align/∂φ ≈ 0), the Euclidean and
    retraction steps coincide to first order, so the IFT Jacobian is
    correct regardless of which retraction is used forward.
    """
    def step_fn(mu_in: torch.Tensor, sigma_in: torch.Tensor,
                 phi_in: torch.Tensor) -> tuple:
        # --- mu/sigma update (same as make_deq_step_fn) ---
        cached_transport = None

        alpha_eff = (
            ffn.get_bayesian_alpha(mu_in, mu_p_current, sigma_p, sigma_in, eps=eps)
            if ffn.learnable_alpha else ffn.alpha
        )
        _alpha_c0 = F.softplus(ffn.raw_c0) if ffn.learnable_alpha else None

        if ffn.multihead_vfe:
            grad_mu = torch.zeros_like(mu_in)
            grad_sigma = torch.zeros_like(sigma_in)
            block_start = 0
            for h, d_h in enumerate(ffn.irrep_dims):
                block_end = block_start + d_h
                mu_h = mu_in[:, :, block_start:block_end].contiguous()
                mu_p_h = mu_p_current[:, :, block_start:block_end].contiguous()
                if is_diagonal:
                    sigma_h = sigma_in[:, :, block_start:block_end].contiguous()
                    sigma_p_h = sigma_p[:, :, block_start:block_end].contiguous()
                else:
                    sigma_h = sigma_in[:, :, block_start:block_end, block_start:block_end].contiguous()
                    sigma_p_h = sigma_p[:, :, block_start:block_end, block_start:block_end].contiguous()
                gen_h = ffn.generators[:, block_start:block_end, block_start:block_end]
                kappa_h = ffn._get_kappa_h(h, d_h)  # Match main path scaling

                beta_h = compute_attention_weights(
                    mu_q=mu_h, sigma_q=sigma_h,
                    phi=phi_in, generators=gen_h,
                    kappa=kappa_h, epsilon=eps, mask=mask,
                    return_kl=False,
                    diagonal_covariance=is_diagonal,
                        mask_self_attention=ffn.mask_self_attention,
                    gauge_mode=ffn.gauge_mode,
                    exact_diagonal_transport=ffn.exact_diagonal_transport,
                )
                alpha_h = (
                    alpha_eff[:, :, block_start:block_end]
                    if isinstance(alpha_eff, torch.Tensor) and alpha_eff.dim() == 3
                    else alpha_eff
                )
                c0_h = _alpha_c0[block_start:block_end] if _alpha_c0 is not None else None
                grad_mu_h, grad_sigma_h = compute_vfe_gradients_gpu(
                    mu_q=mu_h, sigma_q=sigma_h,
                    mu_p=mu_p_h, sigma_p=sigma_p_h,
                    beta=beta_h, phi=phi_in, generators=gen_h,
                    alpha=alpha_h, lambda_belief=ffn.lambda_belief, lambda_softmax=ffn.lambda_softmax,
                    kappa=kappa_h, eps=eps,
                    alpha_c0=c0_h,
                    compute_sigma_align_grad=ffn.compute_sigma_align_grad,
                    exact_diagonal_transport=ffn.exact_diagonal_transport,
                )
                grad_mu[:, :, block_start:block_end] = grad_mu_h
                if is_diagonal:
                    grad_sigma[:, :, block_start:block_end] = grad_sigma_h
                else:
                    if grad_sigma_h.dim() == 3 and d_h == 1:
                        grad_sigma_h = grad_sigma_h.unsqueeze(-1)
                    grad_sigma[:, :, block_start:block_end, block_start:block_end] = grad_sigma_h
                block_start = block_end

        grad_mu = torch.clamp(grad_mu, min=-1e3, max=1e3)
        grad_sigma = torch.clamp(grad_sigma, min=-1e3, max=1e3)

        nat_grad_mu, nat_grad_sigma = compute_natural_gradient_gpu(
            grad_mu, grad_sigma, sigma_in, eps=eps,
        )

        nat_grad_mu_norm = torch.linalg.norm(nat_grad_mu, dim=-1, keepdim=True)
        nat_grad_mu = nat_grad_mu * torch.clamp(
            500.0 / (nat_grad_mu_norm + eps), max=1.0
        )

        if is_diagonal:
            nat_grad_sigma_norm = torch.linalg.norm(nat_grad_sigma, dim=-1, keepdim=True)
        else:
            nat_grad_sigma_norm = torch.linalg.norm(
                nat_grad_sigma.flatten(-2), dim=-1, keepdim=True
            ).unsqueeze(-1)
        nat_grad_sigma = nat_grad_sigma * torch.clamp(
            500.0 / (nat_grad_sigma_norm + eps), max=1.0
        )

        # mu update with trust region
        delta_mu = -ffn.lr * nat_grad_mu
        if is_diagonal:
            sigma_sqrt = torch.sqrt(sigma_in.float().clamp(min=eps)).to(dtype)
            whitened = delta_mu / sigma_sqrt
        else:
            sigma_diag = torch.diagonal(sigma_in, dim1=-2, dim2=-1).clone().float().clamp(min=eps)
            whitened = delta_mu / torch.sqrt(sigma_diag).to(dtype)

        w_norm = torch.linalg.norm(whitened, dim=-1, keepdim=True)
        scale = torch.clamp(2.0 / (w_norm + eps), max=1.0)
        mu_out = mu_in + scale * delta_mu

        # sigma update — calibrated step: ~0.1% max change per iter
        if ffn.update_sigma:
            sigma_trust = ffn._get_sigma_trust(ffn.lr)
            if is_diagonal:
                sigma_out = retract_spd_diagonal_torch(
                    sigma_diag=sigma_in, delta_sigma=-nat_grad_sigma,
                    step_size=1.0, trust_region=sigma_trust, eps=eps,
                    sigma_max=ffn.sigma_max,
                )
            else:
                sigma_out = retract_spd_torch(
                    Sigma=sigma_in, delta_Sigma=-nat_grad_sigma,
                    step_size=1.0, trust_region=sigma_trust * 0.5, eps=eps,
                    sigma_max=ffn.sigma_max,
                )
        else:
            sigma_out = sigma_in

        # --- phi update: differentiable Euclidean descent on F_align ---
        # Compute alignment loss with autograd tracking through phi_in.
        # At the fixed point ∂F_align/∂φ ≈ 0, so the Euclidean step and
        # Lie group retraction agree to first order (both give φ' ≈ φ).
        #
        # Split the product rule d/dφ[β·KL] = β·dKL/dφ + KL·dβ/dφ into
        # separately weighted direct + softmax-coupling terms, matching
        # the forward-path _compute_phi_grad.  This ensures the DEQ
        # closure's phi Jacobian matches the forward E-step Jacobian when
        # lambda_belief != lambda_softmax.  When the two weights are equal,
        # the result is identical to the original unified form.
        if ffn.multihead_vfe:
            alignment_loss = torch.tensor(0.0, device=mu_in.device, dtype=mu_in.dtype)
            block_start = 0
            for h, d_h in enumerate(ffn.irrep_dims):
                block_end = block_start + d_h
                mu_h = mu_in[:, :, block_start:block_end].detach()
                if sigma_in is None:
                    sigma_h = None
                elif is_diagonal:
                    sigma_h = sigma_in[:, :, block_start:block_end].detach()
                else:
                    sigma_h = sigma_in[:, :, block_start:block_end, block_start:block_end].detach()
                gen_h = ffn.generators[:, block_start:block_end, block_start:block_end]
                kappa_h = ffn._get_kappa_h(h, d_h)

                beta_phi_h_result = compute_attention_weights(
                    mu_q=mu_h, sigma_q=sigma_h,
                    phi=phi_in, generators=gen_h,
                    kappa=kappa_h, epsilon=eps, mask=mask,
                    return_kl=True,
                    diagonal_covariance=is_diagonal,
                    irrep_dims=[d_h],
                        mask_self_attention=ffn.mask_self_attention,
                    gauge_mode=ffn.gauge_mode,
                    exact_diagonal_transport=ffn.exact_diagonal_transport,
                )
                beta_phi_h, kl_h = beta_phi_h_result
                alignment_loss = alignment_loss + (
                    ffn.lambda_belief * (beta_phi_h.detach() * kl_h).sum()
                    + ffn.lambda_softmax * (beta_phi_h * kl_h.detach()).sum()
                )
                block_start = block_end

        # Differentiable Euclidean phi step (autograd tracks through alignment_loss).
        #
        # retain_graph=True is REQUIRED here alongside create_graph=True.  The
        # Neumann-series loop in DEQFixedPointFull.backward calls step_fn K
        # times, and the outer autograd.grad(outputs=[mu_out, sigma_out, phi_out])
        # must be able to trace through the shared intermediate values that
        # alignment_loss's inner computation uses (attention, KL, etc.).  With
        # retain_graph=False the inner call frees those intermediates, and on
        # Neumann iteration K > 0 the outer VJP crashes with
        # "Trying to backward through the graph a second time".  This was a
        # pre-existing bug in the joint-phi DEQ path that was never caught
        # because the test suite does not exercise deq_include_phi=True with
        # deq_neumann_terms > 1.  Audit finding #46.
        phi_lr_step = ffn.phi_lr
        if alignment_loss.grad_fn is not None:
            grad_phi_align = torch.autograd.grad(
                alignment_loss, phi_in,
                create_graph=True,  # Keep graph for DEQ backward VJP
                retain_graph=True,  # Share intermediates with outer VJP
            )[0]
            # Norm clipping (differentiable)
            grad_phi_norm = torch.linalg.norm(grad_phi_align, dim=-1, keepdim=True)
            grad_phi_align = grad_phi_align * torch.clamp(
                10.0 / (grad_phi_norm + 1e-6), max=1.0
            )
            phi_out = phi_in - phi_lr_step * grad_phi_align
        else:
            phi_out = phi_in

        return mu_out, sigma_out, phi_out

    return step_fn
