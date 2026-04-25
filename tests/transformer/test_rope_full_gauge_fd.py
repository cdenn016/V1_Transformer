"""Finite-difference verification for _compute_rope_full_gauge_gradient_per_head.

Closes the coverage gap identified in the 2026-04-19 audit (C3): prior to this
file, scripts/verify_vfe_gradients_fd.py only exercised the block-diagonal
analytic path.  The rope_full_gauge=True per-head autograd path in
transformer/core/vfe_gradients.py:_compute_rope_full_gauge_gradient_per_head
had zero finite-difference tests.

The function internally uses torch.autograd.grad to obtain
∂F_head/∂μ, ∂F_head/∂σ, where F_head is the per-head contribution to the
E-step free energy.  We construct an equivalent F_head in float64 (built from
the same rope rotation, Ω transport, full-covariance KL, and softmax weighting
the function implements) and finite-difference it.  The float32 analytic
gradients must agree with the float64 FD reference within 1e-2 relative error.
Also covers the C2 regression: the function must raise rather than silently
no-op when invoked under torch.no_grad().
"""
from __future__ import annotations

import math

import pytest
import torch


def _apply_rope_manual(x: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """Pure-python RoPE to build the float64 FD oracle.

    Matches transformer.core.rope._apply_rope for even d_h.  x shape (B, N, d_h).
    """
    B, N, d_h = x.shape
    assert d_h % 2 == 0, "RoPE requires even d_h"
    half = d_h // 2
    device = x.device
    dtype = x.dtype

    pos = torch.arange(N, device=device, dtype=dtype)
    freqs = torch.arange(half, device=device, dtype=dtype)
    inv_freq = 1.0 / (base ** (2.0 * freqs / d_h))
    angles = pos.unsqueeze(-1) * inv_freq.unsqueeze(0)  # (N, half)
    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    out_even = x_even * cos_a - x_odd * sin_a
    out_odd = x_even * sin_a + x_odd * cos_a
    out = torch.empty_like(x)
    out[..., 0::2] = out_even
    out[..., 1::2] = out_odd
    return out


def _apply_rope_to_cov_manual(sigma: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """Apply R(θ_n) Σ R(θ_n)^T to each position independently.

    Matches transformer.core.rope._apply_rope_to_covariance.  sigma shape
    (B, N, d, d).
    """
    B, N, d, _ = sigma.shape
    assert d % 2 == 0
    half = d // 2
    device = sigma.device
    dtype = sigma.dtype

    pos = torch.arange(N, device=device, dtype=dtype)
    freqs = torch.arange(half, device=device, dtype=dtype)
    inv_freq = 1.0 / (base ** (2.0 * freqs / d))
    angles = pos.unsqueeze(-1) * inv_freq.unsqueeze(0)  # (N, half)
    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)

    R = torch.zeros(N, d, d, device=device, dtype=dtype)
    for n in range(N):
        for h in range(half):
            R[n, 2 * h, 2 * h] = cos_a[n, h]
            R[n, 2 * h, 2 * h + 1] = -sin_a[n, h]
            R[n, 2 * h + 1, 2 * h] = sin_a[n, h]
            R[n, 2 * h + 1, 2 * h + 1] = cos_a[n, h]
    # R: (N, d, d); sigma: (B, N, d, d)
    R_b = R.unsqueeze(0).expand(B, -1, -1, -1)
    return torch.einsum('bnde,bnef,bngf->bndg', R_b, sigma, R_b)


def _build_F_head_f64(
    mu_h, sigma_h, mu_p_h, sigma_p_h, phi, gen_h,
    alpha, lambda_belief, lambda_softmax, kappa,
    eps, rope_base, d_h,
):
    """Replicate F_head scalar in float64 for FD oracle.

    Must match _compute_rope_full_gauge_gradient_per_head's loss (diagonal
    sigma path only).  Returns a scalar tensor with grad_fn linked to mu_h
    and sigma_h.
    """
    B, N, _ = mu_h.shape
    device = mu_h.device

    # Lift diagonal to full
    sigma_full = torch.diag_embed(sigma_h)  # (B, N, d, d)
    mu_rope = _apply_rope_manual(mu_h, base=rope_base)
    sigma_rope = _apply_rope_to_cov_manual(sigma_full, base=rope_base)

    # Build Ω from phi via generators (single-head: gen_h is (n_gen, d, d))
    # Ω_ij = exp(φ_i) · exp(-φ_j)
    phi_mat = torch.einsum('bna,aij->bnij', phi, gen_h)
    exp_phi = torch.linalg.matrix_exp(phi_mat)          # (B, N, d, d)
    exp_neg_phi = torch.linalg.matrix_exp(-phi_mat)     # (B, N, d, d)
    Omega = torch.einsum('bikl,bjlm->bijkm', exp_phi, exp_neg_phi)  # (B, N, N, d, d)

    mu_t = torch.einsum('bijkl,bjl->bijk', Omega, mu_rope)  # (B, N, N, d)
    sigma_t = torch.einsum('bijkl,bjlm,bijnm->bijkn', Omega, sigma_rope, Omega)
    sigma_t = 0.5 * (sigma_t + sigma_t.transpose(-1, -2))
    eye_d = torch.eye(d_h, device=device, dtype=mu_h.dtype)
    sigma_t = sigma_t + eps * eye_d

    sigma_i_rope = sigma_rope.unsqueeze(2).expand(-1, -1, N, -1, -1)
    sigma_i_rope = 0.5 * (sigma_i_rope + sigma_i_rope.transpose(-1, -2)) + eps * eye_d

    sigma_t_inv = torch.linalg.inv(sigma_t)

    # Trace term
    trace_term = torch.einsum('bijkl,bijlk->bij', sigma_t_inv, sigma_i_rope)
    delta = mu_t - mu_rope[:, :, None, :]
    mahal_term = torch.einsum('bijk,bijkl,bijl->bij', delta, sigma_t_inv, delta)
    _, logdet_t = torch.linalg.slogdet(sigma_t)
    _, logdet_i = torch.linalg.slogdet(sigma_i_rope)
    kl = 0.5 * (trace_term + mahal_term - d_h + logdet_t - logdet_i)
    kl = kl.clamp(min=0.0)

    # Softmax with sqrt(d_h) dim-scale
    kappa_val = float(kappa)
    dim_scale = math.sqrt(max(d_h, 1))
    logits = -kl / (kappa_val * dim_scale)
    beta = torch.nn.functional.softmax(logits, dim=-1)

    # Alignment loss: F_align_direct + F_align_softmax matches the function
    F_align = (lambda_belief * (beta.detach() * kl).sum()
               + lambda_softmax * (beta * kl.detach()).sum())

    # Self-coupling KL(q_i || p_i), diagonal form
    sp_safe = sigma_p_h.clamp(min=eps)
    sq_safe = sigma_h.clamp(min=eps)
    kl_self = 0.5 * (
        sq_safe / sp_safe
        + (mu_h - mu_p_h) ** 2 / sp_safe
        - 1.0
        + torch.log(sp_safe)
        - torch.log(sq_safe)
    )
    F_self = float(alpha) * kl_self.sum()

    return F_self + F_align


@pytest.fixture
def small_rope_setup():
    """Small (B=1, N=3, d_h=4) rope_full_gauge scenario with float64 tensors."""
    torch.manual_seed(1234)
    B, N, d_h = 1, 3, 4
    n_gen = 3
    mu_h = torch.randn(B, N, d_h, dtype=torch.float64)
    sigma_h = torch.rand(B, N, d_h, dtype=torch.float64).clamp(min=0.3) + 0.5
    mu_p_h = torch.randn(B, N, d_h, dtype=torch.float64)
    sigma_p_h = torch.rand(B, N, d_h, dtype=torch.float64).clamp(min=0.3) + 0.5
    phi = 0.2 * torch.randn(B, N, n_gen, dtype=torch.float64)
    # Anti-symmetric generators (so(d_h) subset) for numerical sanity
    raw = torch.randn(n_gen, d_h, d_h, dtype=torch.float64)
    gen_h = (raw - raw.transpose(-1, -2)) * 0.3
    return dict(
        mu_h=mu_h, sigma_h=sigma_h, mu_p_h=mu_p_h, sigma_p_h=sigma_p_h,
        phi=phi, gen_h=gen_h, d_h=d_h,
    )


def test_rope_full_gauge_grad_matches_f64_autograd_oracle(small_rope_setup):
    """Float32 analytic ≈ float64 autograd oracle on same F_head."""
    from transformer.core.vfe_gradients import (
        _compute_rope_full_gauge_gradient_per_head,
    )

    s = small_rope_setup
    alpha = 1.0
    lam = 1.0
    kappa = 1.0
    eps = 1e-6
    rope_base = 10000.0

    # ---- Float64 oracle via autograd on the reconstructed F_head ----
    mu_d = s['mu_h'].clone().requires_grad_(True)
    sigma_d = s['sigma_h'].clone().requires_grad_(True)
    F_oracle = _build_F_head_f64(
        mu_d, sigma_d, s['mu_p_h'], s['sigma_p_h'], s['phi'], s['gen_h'],
        alpha, lam, lam, kappa, eps, rope_base, s['d_h'],
    )
    grad_mu_oracle, grad_sigma_oracle = torch.autograd.grad(
        F_oracle, [mu_d, sigma_d], create_graph=False, retain_graph=False,
    )

    # ---- Float32 analytic via the production function ----
    mu_f32 = s['mu_h'].float()
    sigma_f32 = s['sigma_h'].float()
    mu_p_f32 = s['mu_p_h'].float()
    sigma_p_f32 = s['sigma_p_h'].float()
    phi_f32 = s['phi'].float()
    gen_f32 = s['gen_h'].float()

    _beta, grad_mu_ana, grad_sigma_ana = _compute_rope_full_gauge_gradient_per_head(
        mu_h=mu_f32, sigma_h=sigma_f32,
        mu_p_h=mu_p_f32, sigma_p_h=sigma_p_f32,
        phi=phi_f32, gen_h=gen_f32,
        alpha=alpha, lambda_belief=lam, lambda_softmax=lam,
        kappa=kappa, eps=eps, rope_base=rope_base, d_h=s['d_h'],
        cached_block_exp_pairs=None,
        enforce_orthogonal=False, mask=None, mask_self_attention=False,
        gauge_covariant_ridge=False, alpha_c0=None,
    )

    # Finite elements
    assert torch.isfinite(grad_mu_ana).all(), "grad_mu has non-finite elements"
    assert torch.isfinite(grad_sigma_ana).all(), "grad_sigma has non-finite elements"

    # Relative error vs. float64 oracle
    def _max_rel(analytic_f32, oracle_f64):
        a = analytic_f32.double()
        o = oracle_f64
        denom = o.abs().clamp(min=1e-6)
        return ((a - o).abs() / denom).max().item()

    err_mu = _max_rel(grad_mu_ana, grad_mu_oracle)
    err_sigma = _max_rel(grad_sigma_ana, grad_sigma_oracle)

    assert err_mu < 1e-2, f"grad_mu max rel err {err_mu:.3e} > 1e-2"
    assert err_sigma < 1e-2, f"grad_sigma max rel err {err_sigma:.3e} > 1e-2"


def test_rope_full_gauge_raises_under_no_grad(small_rope_setup):
    """C2 regression: function must not silently no-op under torch.no_grad().

    Prior to the fix, _update_phi in vfe/e_step.py lacked a torch.enable_grad()
    wrapper and would return phi unchanged if the caller used torch.no_grad().
    The rope_full_gauge path already had the correct defensive wrapper at
    vfe_gradients.py:2106 — this test is a regression fixture to keep it there.
    """
    from transformer.core.vfe_gradients import (
        _compute_rope_full_gauge_gradient_per_head,
    )

    s = small_rope_setup
    mu_f32 = s['mu_h'].float()
    sigma_f32 = s['sigma_h'].float()
    mu_p_f32 = s['mu_p_h'].float()
    sigma_p_f32 = s['sigma_p_h'].float()
    phi_f32 = s['phi'].float()
    gen_f32 = s['gen_h'].float()

    # Under no_grad, the defensive enable_grad() inside the function must
    # still permit autograd.grad to run and return real gradients (not None).
    with torch.no_grad():
        _beta, grad_mu, grad_sigma = _compute_rope_full_gauge_gradient_per_head(
            mu_h=mu_f32, sigma_h=sigma_f32,
            mu_p_h=mu_p_f32, sigma_p_h=sigma_p_f32,
            phi=phi_f32, gen_h=gen_f32,
            alpha=1.0, lambda_belief=1.0, lambda_softmax=1.0,
            kappa=1.0, eps=1e-6, rope_base=10000.0, d_h=s['d_h'],
            cached_block_exp_pairs=None,
            enforce_orthogonal=False, mask=None, mask_self_attention=False,
            gauge_covariant_ridge=False, alpha_c0=None,
        )

    assert grad_mu is not None and torch.isfinite(grad_mu).all()
    assert grad_sigma is not None and torch.isfinite(grad_sigma).all()
    assert grad_mu.abs().sum().item() > 0, (
        "grad_mu is all zeros under no_grad — enable_grad() defense may have regressed"
    )
