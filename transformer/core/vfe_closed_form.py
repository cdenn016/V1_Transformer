"""
Closed-Form E-Step + Picard Resolve
====================================

Extracted from ``variational_ffn.py`` to keep the "side-quest" closed-form
E-step machinery in its own focused module.  Activated by
``closed_form_e_step=True`` on ``VariationalFFNDynamic``.

Mathematical background
-----------------------
Instead of running gradient descent in the E-step, compute the
precision-weighted fixed point analytically by solving a linear system:

    ``A μ* = b``

where

    ``A = α·Σ_p^{−1} + λ·Σ_j β_ij·Σ_j_t^{−1}``
    ``b = α·Σ_p^{−1}·μ_p + λ·Σ_j β_ij·Σ_j_t^{−1}·Ω_ij·μ_j``

and ``Σ_j_t = Ω_ij·Σ_j·Ω_ij^T`` is the transported covariance.  The
covariance update is

    ``Σ* = (α + λ)·A^{−1}``

which comes from setting ``∂F/∂σ_q = 0`` with ``Σ_j β_ij = 1``
(softmax normalisation).

The diagonal branch additionally absorbs the softmax-coupling correction
(``S_mu``, ``c_mu``, ``S_sigma`` terms) into the fixed point directly.
The full-covariance branch uses a linear-only closed form and delegates
the softmax coupling to the optional Picard re-solve.

Picard re-solve
---------------
When ``n_picard_steps > 0``, the closed-form solution is iteratively
refined to account for the softmax coupling nonlinearity.  The diagonal
branch re-solves the enhanced closed form; the full-cov branch applies a
gradient-based correction using ``picard_trust_region``.

Public API
----------
- :func:`run_closed_form_e_step` — dispatches to the diagonal or
  full-covariance branch, applies Picard resolve, and runs optional phi
  evolution via the VFE gradient.  This is the function
  ``VariationalFFNDynamic.forward`` delegates to when
  ``closed_form_e_step=True``.

Mutual exclusions
-----------------
``closed_form_e_step=True`` is incompatible with:

- ``active_inference=True`` — the closed-form E-step bypasses the
  iterative VFE loop where the active-inference gradient is applied.
  Enforced in ``wire_readout_references`` (raises ``ValueError``).

``n_picard_steps > 0`` requires ``closed_form_e_step=True`` (enforced
in ``BlockConfig.__post_init__``).
"""

from typing import TYPE_CHECKING, Optional, Tuple

import math
import torch

from transformer.core.attention import compute_attention_weights
from transformer.core.gauge_utils import fused_block_matrix_exp_pairs
from transformer.core.vfe_utils import _retract_phi, _safe_spd_inv

if TYPE_CHECKING:
    # Forward reference only — avoids a circular import at runtime
    from transformer.core.variational_ffn import VariationalFFNDynamic


__all__ = [
    "run_closed_form_e_step",
]


def run_closed_form_e_step(
    ffn: "VariationalFFNDynamic",
    mu_current: torch.Tensor,
    sigma_current: Optional[torch.Tensor],
    phi_current: torch.Tensor,
    omega_current: Optional[torch.Tensor],
    mu_p_current: torch.Tensor,
    sigma_p: torch.Tensor,
    alpha_effective,
    _alpha_c0,
    is_diagonal: bool,
    B: int,
    N: int,
    device: torch.device,
    dtype: torch.dtype,
    eps: float,
    mask: Optional[torch.Tensor],
    return_beta_history: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor,
           Optional[torch.Tensor], list, Optional[list]]:
    r"""Compute the precision-weighted closed-form VFE fixed point.

    Implements:
        μ_i* = [α·μ_p/σ_p + λ·Σ_j β_ij·(Ω_ij μ_j)/σ_j] / [α/σ_p + λ·Σ_j β_ij/σ_j]
        σ_i* = (α + λ) / [α/σ_p + λ·Σ_j β_ij/σ_j]

    The (α + λ) numerator in σ_i* comes from setting ∂F/∂σ_q = 0 for a
    Gaussian q with Σ_j β_ij = 1 (softmax normalisation):

        ∂/∂σ_q [α·KL(q||p) + λ·Σ_j β_ij·KL(q||Ω_ij q_j)]
            = ½·[−(α + λ·Σ_j β_ij)/σ_q + (α/σ_p + λ·Σ_j β_ij/σ_j^t)]
            = 0  ⟹  σ_q = (α + λ) / (α/σ_p + λ·Σ_j β_ij/σ_j^t)

    The code applies this as ``entropy_scale = alpha_h + lambda_belief``
    at the σ update below.  Also applies Picard re-solve (n_picard_steps)
    and optional phi evolution.

    Returns:
        (mu_current, sigma_current, phi_current, omega_current, beta_heads, beta_history_list)
    """
    # 0. Exact diagonal transport: lift sigma to full (B,N,K,K) and route
    # through the full-cov branch below. The diagonal CF branch extracts
    # diag(Omega @ diag(sigma) @ Omega^T) correctly but then takes its
    # element-wise inverse at line 184 — which discards off-diagonal
    # information that the full-cov branch's Cholesky inverse preserves.
    # When the caller sets `exact_diagonal_transport=True` on the FFN,
    # honour the documented contract: lift, run the matrix-inverse path,
    # then extract the diagonal at exit. Mirrors the iterative dispatch in
    # `attention.py:283-285`.
    _lifted_for_exact = False
    if is_diagonal and getattr(ffn, 'exact_diagonal_transport', False) and sigma_current is not None:
        sigma_current = torch.diag_embed(sigma_current)        # (B, N, K) -> (B, N, K, K)
        sigma_p = torch.diag_embed(sigma_p)
        is_diagonal = False
        _lifted_for_exact = True

    # 1. Compute block exp pairs for transport
    _cf_bep = ffn._build_block_exp_pairs(
        phi_current, omega_current, B, N, device, dtype,
    )

    # 2. Per-head: compute β_h, then closed-form fixed point
    beta_heads: list = []

    if is_diagonal:
        # ─────────────────────────────────────────────────────────────────
        # DIAGONAL CLOSED-FORM: element-wise precision-weighted average
        # ─────────────────────────────────────────────────────────────────
        mu_star = torch.zeros_like(mu_current)
        sigma_star = torch.zeros_like(sigma_current)
        block_start = 0

        for h, d_h in enumerate(ffn.irrep_dims or [ffn.embed_dim]):
            block_end = block_start + d_h

            mu_h = mu_current[:, :, block_start:block_end].detach().contiguous()
            mu_p_h = mu_p_current[:, :, block_start:block_end].contiguous()
            sigma_h = sigma_current[:, :, block_start:block_end].detach().contiguous()
            sigma_p_h = sigma_p[:, :, block_start:block_end].contiguous()
            gen_h = ffn.generators[:, block_start:block_end, block_start:block_end]

            kappa_h = ffn._get_kappa_h(h, d_h) if ffn.irrep_dims else ffn.kappa
            alpha_h = (alpha_effective[:, :, block_start:block_end]
                       if isinstance(alpha_effective, torch.Tensor) and alpha_effective.dim() == 3
                       else alpha_effective)
            _head_bep = [_cf_bep[h]] if _cf_bep is not None else None

            # Compute β_h AND pairwise KL (need KL for softmax coupling)
            beta_kl_result = compute_attention_weights(
                mu_q=mu_h, sigma_q=sigma_h,
                phi=phi_current, generators=gen_h,
                kappa=kappa_h, epsilon=eps, mask=mask,
                return_kl=True,
                diagonal_covariance=True,
                irrep_dims=[d_h] if ffn.irrep_dims else None,
                cached_block_exp_pairs=_head_bep,
                mask_self_attention=ffn.mask_self_attention,
                gauge_mode=ffn.gauge_mode,
                use_rope=ffn._use_rope_vfe,
                rope_base=ffn._rope_base_vfe,
            )
            beta_h, kl_h = beta_kl_result
            beta_heads.append(beta_h)

            # Transport operators for this head
            exp_phi_h, exp_neg_phi_h = (_cf_bep[h] if _cf_bep is not None else (
                torch.eye(d_h, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1),
                torch.eye(d_h, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1),
            ))

            # Prior precision and information vector
            inv_sigma_p_h = 1.0 / sigma_p_h.clamp(min=eps)       # (B, N, d_h)
            prior_prec_h = alpha_h * inv_sigma_p_h                # (B, N, d_h)
            prior_info_h = alpha_h * mu_p_h * inv_sigma_p_h       # (B, N, d_h)

            # Alignment: precision-weighted transported aggregation.
            #
            # Both mu_j_t = Omega_ij @ mu_j and sigma_j_t_diag = diag(Omega_ij @
            # diag(sigma_j) @ Omega_ij^T) are computed without materialising the
            # full (B, N, N, d_h, d_h) Omega tensor. The mu path uses two
            # sequential contractions; the sigma path uses a (m1, m2) outer
            # product `S` that is O(B, N, d_h^2) instead. Peak memory in either
            # path is (B, N, N, d_h) rather than (B, N, N, d_h^2), saving a
            # factor of d_h vs. forming Omega explicitly.

            # Transported means: rotate j into neutral frame, then into frame i.
            rotated_mu_h = torch.einsum('bjkl,bjl->bjk', exp_neg_phi_h, mu_h)  # (B, N, d_h)
            mu_j_t_cf = torch.einsum('bikl,bjl->bijk', exp_phi_h, rotated_mu_h)  # (B, N, N, d_h)

            # Transported covariance diagonal:
            #   diag(Omega_ij @ diag(sigma_j) @ Omega_ij^T)_k
            #   = sum_{m1,m2} exp_phi_h[i,k,m1] * exp_phi_h[i,k,m2] *
            #                 (sum_l exp_neg_phi_h[j,m1,l] * exp_neg_phi_h[j,m2,l] * sigma_j[l])
            # Reduce over l first into the per-token (d_h, d_h) tensor S_j.
            S_h = torch.einsum(
                'bjml,bjnl,bjl->bjmn', exp_neg_phi_h, exp_neg_phi_h, sigma_h,
            )  # (B, N, d_h, d_h)
            sigma_j_t_diag = torch.einsum(
                'bikm,bikn,bjmn->bijk', exp_phi_h, exp_phi_h, S_h,
            ).clamp(min=eps)  # (B, N, N, d_h)
            inv_sigma_j_t = 1.0 / sigma_j_t_diag  # (B, N, N, d_h)

            # Information per pair: (Omega mu_j) / sigma_j_transported
            info_per_pair = mu_j_t_cf * inv_sigma_j_t  # (B, N, N, d_h)

            # Linear terms: attention-weighted precision and information
            align_info_h = ffn.lambda_belief * torch.einsum('bij,bijk->bik', beta_h, info_per_pair)  # (B, N, d_h)
            align_prec_h = ffn.lambda_belief * torch.einsum('bij,bijk->bik', beta_h, inv_sigma_j_t)  # (B, N, d_h)

            # ENHANCED CLOSED FORM: Include softmax coupling in fixed point
            kappa_h_val = kappa_h.item() if isinstance(kappa_h, torch.Tensor) else kappa_h
            kappa_h_scaled = max(kappa_h_val * math.sqrt(max(d_h, 1)), eps)

            if ffn.lambda_softmax > 0 and kappa_h_scaled > 0:
                w_j_scalar = kl_h * beta_h  # (B, N, N)
                w_bar = w_j_scalar.sum(dim=-1)  # (B, N)

                kl_weighted_prec = torch.einsum('bij,bijk->bik', w_j_scalar, inv_sigma_j_t)  # (B, N, d_h)
                avg_prec_h = align_prec_h / max(ffn.lambda_belief, eps)
                S_mu_h = -(ffn.lambda_softmax / kappa_h_scaled) * (
                    kl_weighted_prec - w_bar.unsqueeze(-1) * avg_prec_h
                )

                kl_weighted_info = torch.einsum('bij,bijk->bik', w_j_scalar, info_per_pair)  # (B, N, d_h)
                avg_info_h = align_info_h / max(ffn.lambda_belief, eps)
                c_mu_h = (ffn.lambda_softmax / kappa_h_scaled) * (
                    kl_weighted_info - w_bar.unsqueeze(-1) * avg_info_h
                )

                p_bar_h = avg_prec_h
                S_sigma_h = -(ffn.lambda_softmax / (2.0 * kappa_h_scaled)) * torch.einsum(
                    'bij,bijk->bik', w_j_scalar, inv_sigma_j_t - p_bar_h[:, :, None, :]
                )
            else:
                S_mu_h = 0.0
                c_mu_h = 0.0
                S_sigma_h = 0.0

            del S_h, sigma_j_t_diag, inv_sigma_j_t, mu_j_t_cf, info_per_pair

            # Enhanced fixed point
            total_prec_h = prior_prec_h + align_prec_h + S_mu_h    # A + S
            total_info_h = prior_info_h + align_info_h - c_mu_h    # b - c
            mu_star[:, :, block_start:block_end] = total_info_h / total_prec_h.clamp(min=eps)

            entropy_scale = alpha_h + ffn.lambda_belief
            sigma_prec_h = prior_prec_h + align_prec_h + 2.0 * S_sigma_h
            sigma_star[:, :, block_start:block_end] = (entropy_scale / sigma_prec_h.clamp(min=eps)).clamp(max=ffn.sigma_max)

            block_start = block_end

        mu_current = mu_star
        if ffn.update_sigma:
            sigma_current = sigma_star
            if ffn.isotropic_covariance:
                scalar_var = sigma_current.mean(dim=-1, keepdim=True)
                sigma_current = scalar_var.expand_as(sigma_current)

    else:
        # ─────────────────────────────────────────────────────────────────
        # FULL-COVARIANCE CLOSED-FORM: Q_j factorization + Cholesky solve
        # ─────────────────────────────────────────────────────────────────
        K_full = ffn.embed_dim
        mu_star = torch.zeros_like(mu_current)
        sigma_star_full = torch.zeros(B, N, K_full, K_full, device=device, dtype=dtype)
        block_start = 0

        for h, d_h in enumerate(ffn.irrep_dims or [ffn.embed_dim]):
            block_end = block_start + d_h

            mu_h = mu_current[:, :, block_start:block_end].detach().contiguous()
            mu_p_h = mu_p_current[:, :, block_start:block_end].contiguous()
            sigma_h = sigma_current[:, :, block_start:block_end, block_start:block_end].detach().contiguous()
            sigma_p_h = sigma_p[:, :, block_start:block_end, block_start:block_end].contiguous()
            gen_h = ffn.generators[:, block_start:block_end, block_start:block_end]

            kappa_h = ffn._get_kappa_h(h, d_h) if ffn.irrep_dims else ffn.kappa
            alpha_h = (alpha_effective[:, :, block_start:block_end]
                       if isinstance(alpha_effective, torch.Tensor) and alpha_effective.dim() == 3
                       else alpha_effective)
            _head_bep = [_cf_bep[h]] if _cf_bep is not None else None

            beta_h = compute_attention_weights(
                mu_q=mu_h, sigma_q=sigma_h,
                phi=phi_current, generators=gen_h,
                kappa=kappa_h, epsilon=eps, mask=mask,
                return_kl=False,
                diagonal_covariance=False,
                irrep_dims=[d_h] if ffn.irrep_dims else None,
                cached_block_exp_pairs=_head_bep,
                mask_self_attention=ffn.mask_self_attention,
                gauge_mode=ffn.gauge_mode,
                use_rope=ffn._use_rope_vfe,
                rope_base=ffn._rope_base_vfe,
            )
            beta_heads.append(beta_h)

            exp_phi_h, exp_neg_phi_h = (_cf_bep[h] if _cf_bep is not None else (
                torch.eye(d_h, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1),
                torch.eye(d_h, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1),
            ))
            E_i = exp_neg_phi_h

            sigma_p_h_safe = sigma_p_h + eps * torch.eye(d_h, device=device, dtype=dtype)
            L_p = torch.linalg.cholesky(sigma_p_h_safe)
            Sigma_p_inv_h = torch.cholesky_inverse(L_p)

            if isinstance(alpha_h, torch.Tensor) and alpha_h.dim() == 3:
                # Per-dimension alpha: use sandwich product to preserve SPD structure
                # A = diag(√α) Σ_p^{-1} diag(√α) — symmetric when Σ_p^{-1} is symmetric
                sqrt_alpha_h = alpha_h.sqrt().unsqueeze(-1)  # (B, N, d_h, 1)
                A_prior_h = sqrt_alpha_h * Sigma_p_inv_h * sqrt_alpha_h.transpose(-1, -2)
                b_prior_h = torch.einsum('bijk,bik->bij', A_prior_h, mu_p_h)
            else:
                A_prior_h = alpha_h * Sigma_p_inv_h
                b_prior_h = torch.einsum('bijk,bik->bij', A_prior_h, mu_p_h)

            sigma_h_safe = sigma_h + eps * torch.eye(d_h, device=device, dtype=dtype)
            L_j = torch.linalg.cholesky(sigma_h_safe)
            Sigma_j_inv_h = torch.cholesky_inverse(L_j)

            Q_j = torch.einsum('bjlk,bjlm,bjmn->bjkn', exp_phi_h, Sigma_j_inv_h, exp_phi_h)
            r_j = torch.einsum('bjkl,bjl->bjk', exp_neg_phi_h, mu_h)
            Q_agg = torch.einsum('bij,bjkl->bikl', beta_h, Q_j)
            Qr_j = torch.einsum('bjkl,bjl->bjk', Q_j, r_j)
            Qr_agg = torch.einsum('bij,bjk->bik', beta_h, Qr_j)

            A_align_h = ffn.lambda_belief * torch.einsum('bikl,bikm,bimn->biln', E_i, Q_agg, E_i)
            b_align_h = ffn.lambda_belief * torch.einsum('bikl,bik->bil', E_i, Qr_agg)

            A_h = A_prior_h + A_align_h + eps * torch.eye(d_h, device=device, dtype=dtype)
            b_h = b_prior_h + b_align_h

            L_A = torch.linalg.cholesky(A_h)
            mu_star_h = torch.cholesky_solve(b_h.unsqueeze(-1), L_A).squeeze(-1)
            A_inv_h = torch.cholesky_inverse(L_A)

            if isinstance(alpha_h, torch.Tensor) and alpha_h.dim() == 3:
                # Per-dimension entropy scale: sandwich product preserves SPD
                # Σ* = diag(√s) A^{-1} diag(√s) where s = α + λ
                sqrt_entropy = (alpha_h + ffn.lambda_belief).sqrt().unsqueeze(-1)  # (B, N, d_h, 1)
                Sigma_star_h = sqrt_entropy * A_inv_h * sqrt_entropy.transpose(-1, -2)
            else:
                Sigma_star_h = (alpha_h + ffn.lambda_belief) * A_inv_h

            # Spectral clamping for full covariance: element-wise clamp would
            # destroy SPD structure (off-diagonal entries clamped to sigma_max
            # can make the matrix singular).  Instead, clamp eigenvalues.
            if Sigma_star_h.dim() == 4 and Sigma_star_h.shape[-1] > 1:
                # Full covariance: spectral clamp
                eigvals, eigvecs = torch.linalg.eigh(
                    0.5 * (Sigma_star_h + Sigma_star_h.transpose(-1, -2))
                )
                eigvals_clamped = eigvals.clamp(min=1e-6, max=ffn.sigma_max ** 2)
                Sigma_star_h = eigvecs @ torch.diag_embed(eigvals_clamped) @ eigvecs.transpose(-1, -2)
            else:
                Sigma_star_h = Sigma_star_h.clamp(max=ffn.sigma_max)
            mu_star[:, :, block_start:block_end] = mu_star_h
            sigma_star_full[:, :, block_start:block_end, block_start:block_end] = Sigma_star_h

            block_start = block_end

        mu_current = mu_star
        if ffn.update_sigma:
            sigma_current = sigma_star_full

    beta_current = beta_heads[-1] if beta_heads else None

    # ── Iterative re-solve: Picard steps ─────────────────────────────
    if ffn.n_picard_steps > 0 and ffn.lambda_softmax > 0 and _cf_bep is not None:
        if is_diagonal:
            for _resolve_iter in range(ffn.n_picard_steps):
                mu_prev = mu_current.clone()
                mu_star = torch.zeros_like(mu_current)
                sigma_star = torch.zeros_like(sigma_current)
                beta_heads_new = []
                block_start = 0

                for h, d_h in enumerate(ffn.irrep_dims or [ffn.embed_dim]):
                    block_end = block_start + d_h

                    mu_h = mu_current[:, :, block_start:block_end].detach().contiguous()
                    mu_p_h = mu_p_current[:, :, block_start:block_end].contiguous()
                    sigma_h = sigma_current[:, :, block_start:block_end].detach().contiguous()
                    sigma_p_h = sigma_p[:, :, block_start:block_end].contiguous()
                    gen_h = ffn.generators[:, block_start:block_end, block_start:block_end]

                    kappa_h = ffn._get_kappa_h(h, d_h) if ffn.irrep_dims else ffn.kappa
                    alpha_h_iter = alpha_effective
                    if isinstance(alpha_effective, torch.Tensor) and alpha_effective.dim() == 3:
                        alpha_h_iter = alpha_effective[:, :, block_start:block_end]
                    if ffn.learnable_alpha and hasattr(ffn, 'get_bayesian_alpha'):
                        alpha_n = ffn.get_bayesian_alpha(
                            mu_current, mu_p_current, sigma_p, sigma_current, eps=eps
                        )
                        if isinstance(alpha_n, torch.Tensor) and alpha_n.dim() == 3:
                            alpha_h_iter = alpha_n[:, :, block_start:block_end]
                        else:
                            alpha_h_iter = alpha_n

                    _head_bep = [_cf_bep[h]] if _cf_bep is not None else None

                    beta_h, kl_h = compute_attention_weights(
                        mu_q=mu_h, sigma_q=sigma_h,
                        phi=phi_current, generators=gen_h,
                        kappa=kappa_h, epsilon=eps, mask=mask,
                        return_kl=True,
                        diagonal_covariance=True,
                        irrep_dims=[d_h] if ffn.irrep_dims else None,
                        cached_block_exp_pairs=_head_bep,
                        mask_self_attention=ffn.mask_self_attention,
                        gauge_mode=ffn.gauge_mode,
                        use_rope=ffn._use_rope_vfe,
                        rope_base=ffn._rope_base_vfe,
                    )
                    beta_heads_new.append(beta_h)

                    exp_phi_h, exp_neg_phi_h = (_cf_bep[h] if _cf_bep is not None else (
                        torch.eye(d_h, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1),
                        torch.eye(d_h, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1),
                    ))

                    inv_sigma_p_h = 1.0 / sigma_p_h.clamp(min=eps)
                    prior_prec_h = alpha_h_iter * inv_sigma_p_h
                    prior_info_h = alpha_h_iter * mu_p_h * inv_sigma_p_h

                    # Same Omega-free contraction pattern as the outer block;
                    # see comments at the diagonal CF construction above.
                    rotated_mu_h = torch.einsum('bjkl,bjl->bjk', exp_neg_phi_h, mu_h)
                    mu_j_t_cf = torch.einsum('bikl,bjl->bijk', exp_phi_h, rotated_mu_h)
                    S_h = torch.einsum(
                        'bjml,bjnl,bjl->bjmn', exp_neg_phi_h, exp_neg_phi_h, sigma_h,
                    )
                    sigma_j_t_diag = torch.einsum(
                        'bikm,bikn,bjmn->bijk', exp_phi_h, exp_phi_h, S_h,
                    ).clamp(min=eps)
                    inv_sigma_j_t = 1.0 / sigma_j_t_diag
                    info_per_pair = mu_j_t_cf * inv_sigma_j_t

                    align_info_h = ffn.lambda_belief * torch.einsum('bij,bijk->bik', beta_h, info_per_pair)
                    align_prec_h = ffn.lambda_belief * torch.einsum('bij,bijk->bik', beta_h, inv_sigma_j_t)

                    kappa_h_val = kappa_h.item() if isinstance(kappa_h, torch.Tensor) else kappa_h
                    kappa_h_scaled = max(kappa_h_val * math.sqrt(max(d_h, 1)), eps)

                    if ffn.lambda_softmax > 0 and kappa_h_scaled > 0:
                        w_j_scalar = kl_h * beta_h
                        w_bar = w_j_scalar.sum(dim=-1)
                        kl_weighted_prec = torch.einsum('bij,bijk->bik', w_j_scalar, inv_sigma_j_t)
                        avg_prec_h = align_prec_h / max(ffn.lambda_belief, eps)
                        S_mu_h = -(ffn.lambda_softmax / kappa_h_scaled) * (
                            kl_weighted_prec - w_bar.unsqueeze(-1) * avg_prec_h
                        )
                        kl_weighted_info = torch.einsum('bij,bijk->bik', w_j_scalar, info_per_pair)
                        avg_info_h = align_info_h / max(ffn.lambda_belief, eps)
                        c_mu_h = (ffn.lambda_softmax / kappa_h_scaled) * (
                            kl_weighted_info - w_bar.unsqueeze(-1) * avg_info_h
                        )
                        p_bar_h = avg_prec_h
                        S_sigma_h = -(ffn.lambda_softmax / (2.0 * kappa_h_scaled)) * torch.einsum(
                            'bij,bijk->bik', w_j_scalar, inv_sigma_j_t - p_bar_h[:, :, None, :]
                        )
                    else:
                        S_mu_h = 0.0
                        c_mu_h = 0.0
                        S_sigma_h = 0.0

                    del S_h, sigma_j_t_diag, inv_sigma_j_t, mu_j_t_cf, info_per_pair

                    total_prec_h = prior_prec_h + align_prec_h + S_mu_h
                    total_info_h = prior_info_h + align_info_h - c_mu_h
                    mu_star[:, :, block_start:block_end] = total_info_h / total_prec_h.clamp(min=eps)

                    entropy_scale = alpha_h_iter + ffn.lambda_belief
                    sigma_prec_h = prior_prec_h + align_prec_h + 2.0 * S_sigma_h
                    sigma_star[:, :, block_start:block_end] = (
                        entropy_scale / sigma_prec_h.clamp(min=eps)
                    ).clamp(max=ffn.sigma_max)

                    block_start = block_end

                mu_current = mu_star
                if ffn.update_sigma:
                    sigma_current = sigma_star
                    if ffn.isotropic_covariance:
                        scalar_var = sigma_current.mean(dim=-1, keepdim=True)
                        sigma_current = scalar_var.expand_as(sigma_current)

                beta_heads = beta_heads_new

                rel_change = (mu_current - mu_prev).norm() / mu_prev.norm().clamp(min=eps)
                if rel_change < 1e-4:
                    break

        else:
            # FULL COVARIANCE: Original Picard (linear-only CF + grad)
            mu_0 = mu_current.clone()
            sigma_0 = sigma_current

            for _picard_iter in range(ffn.n_picard_steps):
                grad_softmax_full = torch.zeros_like(mu_current)
                beta_heads_new: list = []
                block_start = 0

                for h, d_h in enumerate(ffn.irrep_dims or [ffn.embed_dim]):
                    block_end = block_start + d_h
                    # Refresh β with the latest μ so the Picard step
                    # iterates the joint (μ, β) fixed point rather than the
                    # frozen-β linearisation.  σ stays at the pre-loop value
                    # (σ_0) — full-cov Picard does not update σ inside the
                    # loop, so β is refreshed using (μ_current, σ_0).
                    mu_h_refresh = mu_current[:, :, block_start:block_end].detach().contiguous()
                    sigma_h_refresh = sigma_0[:, :, block_start:block_end, block_start:block_end].detach().contiguous()
                    gen_h = ffn.generators[:, block_start:block_end, block_start:block_end]
                    kappa_h_refresh = ffn._get_kappa_h(h, d_h) if ffn.irrep_dims else ffn.kappa
                    _head_bep = [_cf_bep[h]] if _cf_bep is not None else None

                    beta_h = compute_attention_weights(
                        mu_q=mu_h_refresh, sigma_q=sigma_h_refresh,
                        phi=phi_current, generators=gen_h,
                        kappa=kappa_h_refresh, epsilon=eps, mask=mask,
                        return_kl=False,
                        diagonal_covariance=False,
                        irrep_dims=[d_h] if ffn.irrep_dims else None,
                        cached_block_exp_pairs=_head_bep,
                        mask_self_attention=ffn.mask_self_attention,
                        gauge_mode=ffn.gauge_mode,
                        use_rope=ffn._use_rope_vfe,
                        rope_base=ffn._rope_base_vfe,
                    )
                    beta_heads_new.append(beta_h)
                    exp_phi_h, exp_neg_phi_h = _cf_bep[h]

                    mu_h = mu_current[:, :, block_start:block_end]

                    Omega_h = torch.einsum('bikl,bjlm->bijkm', exp_phi_h, exp_neg_phi_h)
                    mu_j_t = torch.einsum('bijkl,bjl->bijk', Omega_h, mu_h)
                    delta = mu_h[:, :, None, :] - mu_j_t

                    sigma_h_blk = sigma_0[:, :, block_start:block_end, block_start:block_end]

                    Sigma_j_t = torch.einsum('bijkl,bjlm,bijnm->bijkn', Omega_h, sigma_h_blk, Omega_h)
                    # Symmetrize and regularize: sandwich products can leave
                    # O(eps_machine) off-diagonal asymmetry that breaks Cholesky.
                    Sigma_j_t = 0.5 * (Sigma_j_t + Sigma_j_t.transpose(-1, -2))
                    Sigma_j_t = Sigma_j_t + eps * torch.eye(d_h, device=device, dtype=dtype)
                    # Cholesky-backed inverse (via _safe_spd_inv) for numerical
                    # stability inside the Picard refinement loop where repeated
                    # inversions amplify roundoff.
                    Sigma_j_t_inv = _safe_spd_inv(Sigma_j_t, eps=eps)

                    grad_kl_pair = torch.einsum('bijkl,bijl->bijk', Sigma_j_t_inv, delta)

                    sigma_i_h = sigma_h_blk
                    trace_h = torch.einsum('bijkl,bilk->bij', Sigma_j_t_inv, sigma_i_h)
                    mahal_h = torch.einsum('bijk,bijk->bij', delta, grad_kl_pair)
                    logdet_jt = torch.linalg.slogdet(Sigma_j_t)[1]
                    logdet_i = torch.linalg.slogdet(
                        sigma_i_h + eps * torch.eye(d_h, device=device, dtype=dtype)
                    )[1]
                    logdet_h = logdet_jt - logdet_i.unsqueeze(2)
                    kl_h = (0.5 * (trace_h + mahal_h - d_h + logdet_h)).clamp(min=0.0)

                    kappa_h = ffn._get_kappa_h(h, d_h) if ffn.irrep_dims else ffn.kappa
                    kappa_h_val = kappa_h.item() if isinstance(kappa_h, torch.Tensor) else kappa_h
                    kappa_h_scaled = max(kappa_h_val * math.sqrt(max(d_h, 1)), eps)
                    avg_grad_h = torch.einsum('bij,bijk->bik', beta_h, grad_kl_pair)
                    grad_dev = avg_grad_h.unsqueeze(2) - grad_kl_pair
                    d_beta_d_mu_h = beta_h.unsqueeze(-1) * grad_dev / kappa_h_scaled
                    grad_softmax_h = ffn.lambda_softmax * torch.einsum(
                        'bij,bijk->bik', kl_h, d_beta_d_mu_h
                    )
                    grad_softmax_full[:, :, block_start:block_end] = grad_softmax_h

                    block_start = block_end

                if ffn.learnable_alpha and _alpha_c0 is not None:
                    alpha_n = ffn.get_bayesian_alpha(
                        mu_current, mu_p_current, sigma_p, sigma_current, eps=eps
                    )
                    sigma_p_diag = torch.diagonal(sigma_p, dim1=-2, dim2=-1).clamp(min=eps)
                    sigma_q_diag = torch.diagonal(sigma_current, dim1=-2, dim2=-1).clamp(min=eps)
                    delta_mu_p = mu_current - mu_p_current
                    alpha_div = getattr(ffn, 'alpha_divergence', 1.0)

                    # α_mismatch term uses per-dim ∂KL/∂μ proxy: δμ/σ_p for KL,
                    # α_d·δμ/σ_blend for Rényi.  The divergence value (kl_k or
                    # D_α,k) is the same one feeding both α_effective and α_n
                    # via get_bayesian_alpha — now consistent under D_α.
                    if abs(alpha_div - 1.0) < 1e-6:
                        div_grad_mu = delta_mu_p / sigma_p_diag
                        div_k = 0.5 * (sigma_q_diag / sigma_p_diag + delta_mu_p ** 2 / sigma_p_diag
                                       - 1.0 + torch.log(sigma_p_diag) - torch.log(sigma_q_diag))
                    else:
                        sigma_blend = (
                            (1.0 - alpha_div) * sigma_q_diag + alpha_div * sigma_p_diag
                        ).clamp(min=eps)
                        div_grad_mu = alpha_div * delta_mu_p / sigma_blend
                        mahal_k = alpha_div * delta_mu_p ** 2 / sigma_blend
                        logdet_k = (
                            (1.0 - alpha_div) * torch.log(sigma_q_diag)
                            + alpha_div * torch.log(sigma_p_diag)
                            - torch.log(sigma_blend)
                        ) / (alpha_div - 1.0)
                        div_k = 0.5 * (mahal_k + logdet_k)
                    div_k = div_k.clamp(min=0.0)

                    alpha_mismatch = (alpha_n - alpha_effective) * div_grad_mu
                    product_rule = -(alpha_n ** 2 / _alpha_c0) * div_k * div_grad_mu
                    grad_softmax_full = grad_softmax_full + alpha_mismatch + product_rule

                correction = torch.zeros_like(mu_current)
                block_start = 0
                for h, d_h in enumerate(ffn.irrep_dims or [ffn.embed_dim]):
                    block_end = block_start + d_h
                    Sigma_0_h = sigma_0[:, :, block_start:block_end, block_start:block_end]
                    grad_h = grad_softmax_full[:, :, block_start:block_end]
                    correction[:, :, block_start:block_end] = torch.einsum(
                        'bijk,bik->bij', Sigma_0_h, grad_h
                    )
                    block_start = block_end
                w_norm = (grad_softmax_full * correction).sum(
                    dim=-1, keepdim=True
                ).clamp(min=0.0).sqrt()

                scale = (ffn.picard_trust_region / w_norm.clamp(min=eps)).clamp(max=1.0)
                mu_current = mu_0 - scale * correction
                # Rebind β so subsequent iterations (and the post-loop
                # beta_history return) see the refreshed weights.
                beta_heads = beta_heads_new

    # ── Beta history for return ───────────────────────────────────────
    beta_history_out: Optional[list] = None
    if return_beta_history:
        beta_stacked = (torch.stack(beta_heads, dim=1) if len(beta_heads) > 1
                        else beta_heads[0].unsqueeze(1))
        beta_history_out = [beta_stacked.detach().clone()]

    # ── Phi/Omega evolution via gradient (gauge enters nonlinearly) ──
    # em_phi_mode == 'M_phi_p' forbids any E-step evolution of the gauge
    # parameter; this guard matches the iterative path in variational_ffn.py.
    if (ffn.update_phi and torch.is_grad_enabled()
            and ffn.gauge_mode not in ('trivial', 'constant')
            and ffn.em_phi_mode != 'M_phi_p'):
        _use_omega = omega_current is not None and ffn.gauge_param == 'omega'
        if _use_omega:
            grad_omega = ffn._compute_omega_grad_direct(
                omega_current, mu_current, sigma_current,
                is_diagonal, mask, eps,
            )
            if grad_omega is not None:
                omega_current = ffn._retract_omega(
                    omega_current, grad_omega, ffn.phi_lr,
                    trust_region=getattr(ffn, 'omega_trust_region', 0.3),
                )
        else:
            # `_cf_bep` was computed at the top of the function from the
            # same `phi_current` (phi has not been retracted yet — that
            # happens below at `_retract_phi`), so it is identical to the
            # pairs we would otherwise rebuild here. Reuse it to avoid a
            # second per-head matrix-exponential pass.
            grad_phi = ffn._compute_phi_grad(
                phi_current, mu_current, sigma_current,
                is_diagonal, mask, eps,
                cached_block_exp_pairs=_cf_bep,
            )
            if grad_phi is not None:
                phi_current = _retract_phi(
                    phi=phi_current,
                    delta_phi=-grad_phi,
                    generators=ffn.generators,
                    step_size=ffn.phi_lr,
                    max_norm=ffn.phi_max_norm,
                    project_slk=getattr(ffn, 'phi_project_slk', False),
                    trace_clamp=getattr(ffn, 'phi_trace_clamp', None),
                    irrep_dims=getattr(ffn, 'irrep_dims', None),
                )

    # If we lifted diagonal sigma to a full (K,K) matrix at entry to honour
    # `exact_diagonal_transport=True`, extract the diagonal so the caller's
    # contract is preserved (the FFN expects diagonal sigma in diagonal mode).
    if _lifted_for_exact and sigma_current is not None:
        sigma_current = torch.diagonal(sigma_current, dim1=-2, dim2=-1).clamp(min=eps)

    return mu_current, sigma_current, phi_current, omega_current, beta_heads, beta_history_out
