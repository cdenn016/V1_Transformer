"""
VFE Diagnostic Functions
========================

Diagnostic and recording helpers extracted from VariationalFFNDynamic.
These are pure side-effect functions — they read tensor state and write
to diagnostic buffers but never modify the forward computation graph.

Extracted to reduce the god-file variational_ffn.py by ~270 lines.
"""

import logging
import torch
import numpy as np
from typing import Dict, Optional, Tuple

import transformer.core.vfe_utils as _vfe_utils_mod
from transformer.core.vfe_utils import (
    _grad_norm,
    _per_pos_stats,
    _aggregate_multihead_vfe_debug,
    IterationSnapshot,
)

logger = logging.getLogger(__name__)


# =============================================================================
# VFE Gradient Debug Printing
# =============================================================================

def print_vfe_grad_debug(
    ffn,
    iteration: int,
    is_diagonal: bool,
    mu_current: torch.Tensor,
    grad_mu: torch.Tensor,
    grad_sigma: torch.Tensor,
    nat_grad_mu: torch.Tensor,
    nat_grad_sigma: torch.Tensor,
    nat_grad_mu_norm: torch.Tensor,
    nat_grad_sigma_norm: torch.Tensor,
    max_nat_grad_norm: float,
    max_nat_grad_sigma_norm: float,
) -> None:
    r"""Print per-component gradient breakdown for VFE debug mode.

    Reads from ``_vfe_utils_mod._VFE_GRAD_DEBUG`` and writes
    ``ffn.last_vfe_debug`` (when ``ffn._collect_vfe_metrics`` is True).
    Resets ``_vfe_utils_mod._VFE_GRAD_DEBUG`` to ``None`` on exit.

    Pure side effect: no tensors are returned or modified.

    Args:
        ffn: The VariationalFFNDynamic instance (duck-typed — only
            ``_debug_vfe_gradients``, ``_collect_vfe_metrics``,
            ``n_iterations``, ``_e_step_grad_norms``, ``irrep_dims``,
            ``last_vfe_debug`` are accessed).
    """
    if _vfe_utils_mod._VFE_GRAD_DEBUG is not None and ffn._debug_vfe_gradients:
        d = _vfe_utils_mod._VFE_GRAD_DEBUG

        # Detect multihead mode: keys have 'headN(d=M)/' prefix
        _is_multihead = any('/' in k for k in d)

        # Euclidean totals computed on the full (already assembled) grad tensors
        _eu_mu = _grad_norm(grad_mu)
        _eu_sig = _grad_norm(grad_sigma)
        _ps_mu = _per_pos_stats(grad_mu)
        _ps_sig = _per_pos_stats(grad_sigma)

        # Nat_grad amplification factors — batch GPU→CPU transfer to minimize sync points
        _sig_clip_cond = (
            nat_grad_sigma_norm.squeeze(-1) if is_diagonal
            else nat_grad_sigma_norm.squeeze(-1).squeeze(-1)
        )
        _debug_scalars = torch.stack([
            nat_grad_mu.detach().norm(),
            nat_grad_sigma.detach().norm(),
            (nat_grad_mu_norm.squeeze(-1) >= max_nat_grad_norm * 0.99).float().mean(),
            (_sig_clip_cond >= max_nat_grad_sigma_norm * 0.99).float().mean(),
        ]).cpu().tolist()
        _raw_ng_mu, _raw_ng_sig, _mu_clip_frac, _sig_clip_frac = _debug_scalars
        _amp_mu = _raw_ng_mu / max(_eu_mu, 1e-12)
        _amp_sig = _raw_ng_sig / max(_eu_sig, 1e-12)

        logger.debug(f"\n{'='*80}")
        logger.debug(
            f"  [VFE GRAD DEBUG] iter {iteration}/{ffn.n_iterations}"
            f"  diag={is_diagonal}  K={mu_current.shape[-1]}"
            f"  B×N={mu_current.shape[0]}×{mu_current.shape[1]}"
            f"  multihead={_is_multihead}"
        )
        logger.debug(f"{'='*80}")

        if _is_multihead:
            # Extract unique head prefixes
            _heads = sorted(set(k.split('/')[0] for k in d if '/' in k),
                            key=lambda x: int(x.split('head')[1].split('(')[0]))
            logger.debug(f"  --- Per-head breakdown ({len(_heads)} heads) ---")
            logger.debug(
                f"  {'Head':<16} {'s_self':>8} {'s_align':>8} {'s_smx':>8}"
                f" {'mu_self':>8} {'mu_dir':>8} {'mu_smx':>8}"
                f" {'KL_avg':>8} {'KL_max':>8} {'sp_min':>8} {'sq_max':>8}"
            )
            logger.debug(
                f"  {'-'*16} {'-'*8} {'-'*8} {'-'*8}"
                f" {'-'*8} {'-'*8} {'-'*8}"
                f" {'-'*8} {'-'*8} {'-'*8} {'-'*8}"
            )
            for hp in _heads:
                def _hget(key, default=0):
                    return d.get(f'{hp}/{key}', default)
                logger.debug(
                    f"  {hp:<16}"
                    f" {_hget('grad_sigma_self'):>8.1f}"
                    f" {_hget('grad_sigma_align_direct'):>8.1f}"
                    f" {_hget('grad_sigma_softmax'):>8.1f}"
                    f" {_hget('grad_mu_self'):>8.1f}"
                    f" {_hget('grad_mu_direct'):>8.1f}"
                    f" {_hget('grad_mu_softmax'):>8.1f}"
                    f" {_hget('kl_pairwise_mean'):>8.2f}"
                    f" {_hget('kl_pairwise_max'):>8.1f}"
                    f" {_hget('sigma_p_min'):>8.4f}"
                    f" {_hget('sigma_q_eig_max'):>8.4f}"
                )
        else:
            # Single-beta mode
            logger.debug("  --- Covariance state ---")
            logger.debug(
                f"  sigma_p  range:  [{d.get('sigma_p_min', 0):.4f}, {d.get('sigma_p_max', 0):.4f}]"
                f"  ->  1/sigma_p range: [{1/max(d.get('sigma_p_max', 1), 1e-12):.2f},"
                f" {1/max(d.get('sigma_p_min', 1e-12), 1e-12):.2f}]"
            )
            logger.debug(f"  sigma_q  eig range: [{d.get('sigma_q_eig_min', 0):.4f}, {d.get('sigma_q_eig_max', 0):.4f}]")
            logger.debug("  --- Euclidean gradient components (global norms) ---")
            logger.debug(
                f"  {'Component':<30} {'mu':>12} {'sigma':>12} {'s pos_mean':>12} {'s pos_max':>12}"
            )
            logger.debug(
                f"  {'-'*30} {'-'*12} {'-'*12} {'-'*12} {'-'*12}"
            )
            logger.debug(
                f"  {'self-coupling (a*dKL/dtheta)':<30}"
                f" {d.get('grad_mu_self', 0):>12.1f}"
                f" {d.get('grad_sigma_self', 0):>12.1f}"
                f" {d.get('grad_sigma_self_pos_mean', 0):>12.2f}"
                f" {d.get('grad_sigma_self_pos_max', 0):>12.2f}"
            )
            logger.debug(
                f"  {'align direct (l*beta*dKL/dtheta)':<30}"
                f" {d.get('grad_mu_direct', 0):>12.1f}"
                f" {d.get('grad_sigma_align_direct', 0):>12.1f}"
                f" {d.get('grad_sigma_align_pos_mean', 0):>12.2f}"
                f" {d.get('grad_sigma_align_pos_max', 0):>12.2f}"
            )
            logger.debug(
                f"  {'softmax (KL*dbeta/dtheta)':<30}"
                f" {d.get('grad_mu_softmax', 0):>12.1f}"
                f" {d.get('grad_sigma_softmax', 0):>12.1f}"
                f" {d.get('grad_sigma_softmax_pos_mean', 0):>12.2f}"
                f" {d.get('grad_sigma_softmax_pos_max', 0):>12.2f}"
            )

        logger.debug("  --- Euclidean total (assembled, after obs) ---")
        logger.debug(f"  grad_mu:    {_eu_mu:>10.1f}  (pos mean: {_ps_mu[0]:.2f}, max: {_ps_mu[1]:.2f})")
        logger.debug(f"  grad_sigma: {_eu_sig:>10.1f}  (pos mean: {_ps_sig[0]:.2f}, max: {_ps_sig[1]:.2f})")
        logger.debug("  --- Natural gradient (Fisher projection) ---")
        logger.debug(
            f"  nat_grad_mu:    {_raw_ng_mu:>10.1f}  (amplification: {_amp_mu:.2f}x)"
            f"  clip: {ffn._e_step_grad_norms['nat_grad_mu_clipped']:.1f}"
            f"  ({_mu_clip_frac*100:.0f}% positions at cap)"
        )
        logger.debug(
            f"  nat_grad_sigma: {_raw_ng_sig:>10.1f}  (amplification: {_amp_sig:.2f}x)"
            f"  clip: {ffn._e_step_grad_norms['nat_grad_sigma_clipped']:.1f}"
            f"  ({_sig_clip_frac*100:.0f}% positions at cap)"
        )
        logger.debug(f"{'='*80}")
        # Store before resetting (debug mode may coexist with metrics collection)
        if ffn._collect_vfe_metrics:
            _aggregate_multihead_vfe_debug(_vfe_utils_mod._VFE_GRAD_DEBUG, ffn.irrep_dims)
            ffn.last_vfe_debug = dict(_vfe_utils_mod._VFE_GRAD_DEBUG)
        _vfe_utils_mod._VFE_GRAD_DEBUG = None  # Reset for next iteration

    # Store lightweight copy for external consumption (no printing overhead)
    elif ffn._collect_vfe_metrics and _vfe_utils_mod._VFE_GRAD_DEBUG is not None:
        _aggregate_multihead_vfe_debug(_vfe_utils_mod._VFE_GRAD_DEBUG, ffn.irrep_dims)
        ffn.last_vfe_debug = dict(_vfe_utils_mod._VFE_GRAD_DEBUG)
        _vfe_utils_mod._VFE_GRAD_DEBUG = None


# =============================================================================
# Per-iteration convergence diagnostics and fiber snapshots
# =============================================================================

def record_iteration_diagnostics(
    ffn,
    iteration: int,
    grad_mu: torch.Tensor,
    grad_sigma: torch.Tensor,
    nat_grad_mu: torch.Tensor,
    nat_grad_sigma: torch.Tensor,
    delta_mu: torch.Tensor,
    mu_current: torch.Tensor,
    sigma_current: torch.Tensor,
    mu_p_current: Optional[torch.Tensor],
    is_diagonal: bool,
    eps: float,
    effective_lr,
    scale: torch.Tensor,
    beta_current: Optional[torch.Tensor],
    beta_heads: Optional[list],
) -> None:
    r"""Collect per-iteration convergence diagnostics and fiber snapshots.

    Appends one entry to ``ffn._iteration_diagnostics`` when
    ``ffn._collect_iteration_diagnostics`` is ``True``, and one
    ``IterationSnapshot`` to ``ffn._fiber_snapshots`` when
    ``ffn._record_fiber_trajectory`` is ``True``.

    Pure side effect: nothing is returned.

    Args:
        ffn: The VariationalFFNDynamic instance (duck-typed — only
            ``_collect_iteration_diagnostics``, ``_record_fiber_trajectory``,
            ``_fiber_token_indices``, ``multihead_vfe``,
            ``_e_step_grad_norms``, ``_diag_prev_mu``,
            ``_iteration_diagnostics``, ``_fiber_snapshots`` are accessed).
    """
    # =============================================================
    # DIAGNOSTIC: Per-iteration convergence data
    # =============================================================
    if ffn._collect_iteration_diagnostics:
        # Sigma condition number for diagnostics
        if is_diagonal:
            _s_det = sigma_current.detach()
            _s_cond = (_s_det.max(dim=-1).values / _s_det.min(dim=-1).values.clamp(min=eps)).mean().item()
        else:
            _s_diag_det = torch.diagonal(sigma_current.detach(), dim1=-2, dim2=-1)
            _s_cond = (_s_diag_det.max(dim=-1).values / _s_diag_det.min(dim=-1).values.clamp(min=eps)).mean().item()
        _diag = {
            'iteration': iteration,
            'grad_mu_norm': grad_mu.detach().norm().item(),
            'grad_sigma_norm': grad_sigma.detach().norm().item(),
            'nat_grad_mu_norm': nat_grad_mu.detach().norm().item(),
            'nat_grad_mu_raw_norm': ffn._e_step_grad_norms.get('nat_grad_mu', 0.0),
            'nat_grad_sigma_norm': nat_grad_sigma.detach().norm().item(),
            'nat_grad_sigma_raw_norm': ffn._e_step_grad_norms.get('nat_grad_sigma', 0.0),
            'nat_grad_sigma_max': nat_grad_sigma.detach().abs().max().item(),
            'delta_mu_norm': delta_mu.detach().norm().item(),
            'mu_norm': mu_current.detach().norm().item(),
            'sigma_mean': sigma_current.detach().mean().item(),
            'sigma_max': sigma_current.detach().max().item(),
            'sigma_min': sigma_current.detach().min().item(),
            'sigma_std': sigma_current.detach().std().item(),
            'sigma_condition': _s_cond,
            'effective_lr': effective_lr.detach().item() if isinstance(effective_lr, torch.Tensor) else float(effective_lr),
            'scale_mean': scale.detach().mean().item(),
        }
        if mu_p_current is not None:
            _diag['mu_diff_to_prior_norm'] = (mu_current - mu_p_current).detach().norm().item()
        # Beta entropy from last computed beta
        try:
            if beta_heads:
                _b_diag = beta_heads[-1].detach().clamp(min=1e-10)
            elif beta_current is not None:
                _b_diag = beta_current.detach().clamp(min=1e-10)
            else:
                _b_diag = None
            if _b_diag is not None:
                _diag['beta_entropy'] = -(_b_diag * _b_diag.log()).sum(dim=-1).mean().item()
        except Exception:
            pass
        # Relative belief change from previous iteration
        if iteration == 0:
            ffn._diag_prev_mu = mu_current.detach().clone()
        else:
            _diag['mu_change_rel'] = (
                (mu_current - ffn._diag_prev_mu).detach().norm().item()
                / (mu_current.detach().norm().item() + 1e-8)
            )
            ffn._diag_prev_mu = mu_current.detach().clone()
        ffn._iteration_diagnostics.append(_diag)

    # =============================================================
    # FIBER TRAJECTORY: Record per-iteration (mu, sigma) snapshot
    # =============================================================
    if ffn._record_fiber_trajectory:
        _tok_idx = ffn._fiber_token_indices
        _beta_ent = 0.0
        try:
            if beta_current is not None:
                _b = beta_current.detach().clamp(min=1e-10)
                _beta_ent = -(_b * _b.log()).sum(dim=-1).mean().item()
        except Exception:
            pass
        ffn._fiber_snapshots.append(IterationSnapshot(
            iteration=iteration,
            mu=mu_current[0, _tok_idx, :].detach().cpu().numpy(),
            sigma_diag=(sigma_current[0, _tok_idx, :].detach().cpu().numpy()
                        if is_diagonal else
                        torch.diagonal(sigma_current[0, _tok_idx], dim1=-2, dim2=-1).detach().cpu().numpy()),
            beta_entropy=_beta_ent,
            grad_mu_norm=grad_mu.detach().norm().item(),
            grad_sigma_norm=grad_sigma.detach().norm().item(),
        ))
