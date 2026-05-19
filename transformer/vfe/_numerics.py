r"""Numerical-conditioning helpers for the /vfe E-step.

Package-internal utilities (underscore-prefixed module) used by
:mod:`transformer.vfe.e_step`, :mod:`transformer.vfe.omega_direct`, and
:mod:`transformer.vfe.non_flat`. Three helpers:

* :func:`check_finite` — NaN/Inf sentinel with selectable response policy.
* :func:`apply_mu_trust_region` — :math:`\sigma`-whitened per-component
  trust-region clamp on the mean update :math:`\delta\mu`.
* :func:`pre_exp_frobenius_clamp` — pre-exponential Frobenius rescale on a
  Lie-algebra tensor so that
  :math:`\|X\|_F \le \mathrm{phi\_spec\_max}` before
  :func:`torch.linalg.matrix_exp`.

All three are opt-in via :class:`transformer.vfe.config.VFEConfig` fields
(``e_nan_check``, ``e_mu_q_trust``, ``phi_spec_max``). The default
configuration leaves all three helpers inactive, so the pure mathematical
path is bitwise unchanged.
"""

from __future__ import annotations

import warnings
from typing import Dict, Literal, Optional, Tuple

import torch


NanCheckMode = Literal['off', 'warn', 'revert', 'abort']


class VFENonFiniteError(RuntimeError):
    r"""Raised by :func:`check_finite` under ``mode='abort'`` when one or more
    E-step iterate tensors contain a non-finite value.

    Carries the ``step_label`` (which checkpoint inside the inner loop fired)
    and ``field`` (which tensor was non-finite) so the trainer log identifies
    the originating site without re-deriving it from a stack trace.
    """

    def __init__(self, step_label: str, field: str) -> None:
        super().__init__(
            f"VFE E-step '{step_label}': non-finite value in tensor '{field}'"
        )
        self.step_label = step_label
        self.field = field


def check_finite(
    tensors: Dict[str, torch.Tensor],
    *,
    mode: NanCheckMode,
    step_label: str,
) -> bool:
    r"""Detect NaN / Inf in a set of E-step iterate tensors.

    Reduces ``torch.isfinite(t).all()`` for each entry. Each reduction is a
    host sync (``.item()``) — opt-in behavior, off by default.

    Args:
        tensors: Mapping ``{name: tensor}`` of iterate tensors to check.
        mode: Response policy.

            * ``'off'``: no-op (function returns ``True``).
            * ``'warn'``: log a :class:`RuntimeWarning`; return ``False``
              on detection but do not raise.
            * ``'revert'`` / ``'abort'``: raise :class:`VFENonFiniteError`
              on detection. Callers handle ``'revert'`` by catching the
              exception and restoring a pre-iteration snapshot before
              breaking the loop.

        step_label: Diagnostic label identifying which checkpoint fired
            (e.g. ``'nat_grad'``, ``'mu_update'``, ``'sigma_retract'``,
            ``'phi_update'``).

    Returns:
        ``True`` if all tensors are finite (or ``mode == 'off'``).
        ``False`` if a non-finite value was detected under ``'warn'`` mode.
        Never returns ``False`` under ``'revert'`` / ``'abort'`` — those
        modes raise on detection instead.
    """
    if mode == 'off':
        return True
    for name, t in tensors.items():
        if not torch.isfinite(t).all().item():
            msg = (
                f"VFE E-step '{step_label}': non-finite value in '{name}' "
                f"(shape={tuple(t.shape)})"
            )
            if mode == 'warn':
                warnings.warn(msg, RuntimeWarning, stacklevel=2)
                return False
            raise VFENonFiniteError(step_label, name)
    return True


def apply_mu_trust_region(
    delta_mu: torch.Tensor,
    sigma_q: torch.Tensor,
    *,
    trust: float,
    is_diagonal: bool,
    eps: float = 1e-8,
) -> torch.Tensor:
    r"""Per-component :math:`\sigma`-whitened trust-region clamp on
    :math:`\delta\mu`.

    Clamps each component
    :math:`|\delta\mu_k / \sqrt{\sigma_{q,kk}}| \le \mathrm{trust}`. The
    inf-norm form matches the existing diagonal-:math:`\sigma` retraction
    (``retract_spd_diagonal_torch`` clamps the whitened
    :math:`\delta\sigma / \sigma` element-wise to
    :math:`\pm\,\mathrm{e\_sigma\_q\_trust}`); the symmetric treatment
    avoids introducing a second trust-region geometry into the E-step. The
    Fisher-canonical 2-norm form (Mahalanobis ball,
    :math:`\delta\mu^\top \Sigma_q^{-1} \delta\mu \le \mathrm{trust}^2`)
    is the theoretical alternative; it can be added later via a `_norm`
    Literal without ABI change.

    Args:
        delta_mu: ``(..., K)`` proposed mean update.
        sigma_q: ``(..., K)`` diagonal variances when ``is_diagonal=True``,
            otherwise ``(..., K, K)`` SPD covariance (only the diagonal is
            used for whitening — matches the diagonal-only treatment that
            ``retract_sigma_e_step`` applies under full-cov inputs).
        trust: Inf-norm bound in whitened coordinates. Must be positive
            and finite; the caller is responsible for guarding ``None``.
        is_diagonal: Whether ``sigma_q`` is in diagonal form.
        eps: Floor on the whitening factor :math:`\sqrt{\sigma_{q,kk}}`
            before division. Defaults to ``1e-8``, matching the
            :func:`transformer.vfe.e_step._diag_kl` floor.

    Returns:
        Clamped ``delta_mu`` of the same shape and dtype as the input.
    """
    if is_diagonal:
        sigma_diag = sigma_q
    else:
        sigma_diag = sigma_q.diagonal(dim1=-2, dim2=-1)
    scale = sigma_diag.clamp(min=eps).sqrt()
    whitened = delta_mu / scale
    whitened_clamped = whitened.clamp(-trust, trust)
    return whitened_clamped * scale


def pre_exp_frobenius_clamp(
    algebra: torch.Tensor,
    *,
    max_fro: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Rescale a Lie-algebra tensor so its per-slice Frobenius norm is
    bounded before :func:`torch.linalg.matrix_exp`.

    For each ``(..., d, d)`` matrix slice ``X``::

        X <- X * min(1, max_fro / ||X||_F)

    The scale is a per-slice multiplier broadcast over the trailing
    ``(d, d)`` axes. The clamp is straight-through with respect to autograd
    where it doesn't bind (scale = 1) and shrinks the gradient by the same
    factor where it does — same semantics as standard gradient clipping.

    Theoretical caveat: clamping the algebra without writing back to the
    stored :math:`\phi` parameter modifies the forward-pass :math:`\Omega`
    while leaving the stored :math:`\phi` larger, so :math:`\Omega` and
    :math:`\phi` are temporarily inconsistent on iterations where the
    clamp binds. Acceptable as an opt-in defensive layer; not the default.

    Rationale: :math:`\|\exp(X)\|_2 \le \exp(\|X\|_2) \le \exp(\|X\|_F)`
    (Hall, *Lie Groups, Lie Algebras, and Representations*, on bounds for
    the matrix exponential), so Frobenius is a tractable upper bound on
    the spectral norm of :math:`\exp(X)` and bounding the former bounds
    the condition number of the sandwich transport
    :math:`\Sigma \mapsto \Omega \Sigma \Omega^\top`.

    Args:
        algebra: ``(..., d, d)`` Lie-algebra tensor (already constructed,
            e.g. ``einsum('bija,akl->bijkl', delta, G_h)``).
        max_fro: Per-slice Frobenius norm cap. Must be positive.

    Returns:
        Tuple ``(clamped_algebra, scale)`` where ``scale`` has shape
        ``(..., 1, 1)`` and contains the per-slice rescaling factor (1.0
        where the clamp did not bind, ``< 1.0`` where it did). The
        diagnostic ``(scale < 1.0).any()`` lets callers count clamp events
        without an extra reduction.
    """
    fro = algebra.norm(p='fro', dim=(-2, -1), keepdim=True)
    scale = (max_fro / fro.clamp(min=1e-12)).clamp(max=1.0)
    return algebra * scale, scale


__all__ = [
    'NanCheckMode',
    'VFENonFiniteError',
    'apply_mu_trust_region',
    'check_finite',
    'pre_exp_frobenius_clamp',
]
