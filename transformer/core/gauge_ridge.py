r"""Gauge-covariant numerical ridge for covariance regularization.

Under a gauge transformation :math:`h \in GL(K)`, a covariance matrix
transforms as the sandwich :math:`\Sigma \to h\,\Sigma\,h^{T}`.  The standard
regularizer :math:`\Sigma + \varepsilon I` does not transform covariantly
because :math:`I` is gauge-invariant as a raw tensor, so the regularized
matrix drifts from the orbit by :math:`\varepsilon\,(hh^{T} - I)`.

This module builds :math:`\varepsilon\,(gg^{T})` from the local gauge frame
:math:`g = \exp(\phi)`.  Under a global gauge :math:`h`, :math:`g \to hg`,
so :math:`gg^{T} \to h(gg^{T})h^{T}` and :math:`\Sigma + \varepsilon\,gg^{T}`
transforms exactly as :math:`\Sigma`.

Optional trace-normalization (``normalize=True``) rescales so
:math:`\mathrm{tr}(gg^{T}) = K` to keep ridge magnitude stable when
:math:`\|g\|` drifts.  This is *not* exactly gauge-covariant under
non-orthogonal :math:`h` because :math:`\mathrm{tr}(hMh^{T})\neq\mathrm{tr}(M)`
in general; it is exact only under orthogonal frame changes.  Default is
``normalize=False`` to preserve exact covariance — keep
``phi_max_norm`` / ``phi_project_slk`` active to bound :math:`\|g\|` if
ridge magnitude drift is a concern.

Intended use: opt-in replacement for ``eps * torch.eye(K)`` at Bucket-A
conditioning sites (see plan ``carefully-ultrathink-through-the-curious-narwhal.md``).
Gated by ``BlockConfig.gauge_covariant_ridge``.
"""

from __future__ import annotations

import torch
from torch import Tensor


def gauge_covariant_eye(
    exp_phi: Tensor,
    eps: float,
    normalize: bool = False,
) -> Tensor:
    r"""Return :math:`\varepsilon\,(gg^{T})` for local frame :math:`g = \exp(\phi)`.

    Args:
        exp_phi: Local gauge frame, shape ``(..., K, K)``. Typically the cached
            ``exp(phi)`` from the E-step.
        eps: Scalar ridge magnitude.
        normalize: If True, rescale so ``tr(gg^T) == K`` before multiplying by
            ``eps``.  Breaks exact gauge covariance under non-orthogonal
            :math:`h` (see module docstring).  Default False.

    Returns:
        Tensor of shape ``(..., K, K)`` that (with ``normalize=False``)
        transforms exactly as :math:`h(gg^{T})h^{T}` under :math:`g \to hg`.
    """
    M = exp_phi @ exp_phi.transpose(-1, -2)
    if normalize:
        K = M.shape[-1]
        trace = M.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1)
        M = M * (K / trace.clamp_min(1e-12))
    return eps * M


def make_ridge(
    K: int,
    eps: float,
    *,
    exp_phi: Tensor | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    batch_shape: tuple[int, ...] = (),
) -> Tensor:
    r"""Unified ridge constructor: covariant when frame provided, else ``eps I``.

    Usage pattern at call sites::

        frame = cached_exp_phi if cfg.gauge_covariant_ridge else None
        R = make_ridge(K, eps, exp_phi=frame, device=Sigma.device, dtype=Sigma.dtype)
        Sigma_reg = Sigma + R

    When ``exp_phi is None`` this returns ``eps * torch.eye(K)`` — bitwise
    identical to the pre-change path — so flag-off numerics are preserved.

    Args:
        K: Covariance dimension.
        eps: Ridge magnitude.
        exp_phi: Local frame ``g = exp(phi)`` if the caller has it and wants
            the covariant path; ``None`` to use the ``eps I`` fallback.
        device, dtype: Used only for the fallback path; when ``exp_phi`` is
            supplied these are taken from the frame.
        batch_shape: Leading dims for the fallback ``eps I`` tensor. Ignored
            when a frame is provided (broadcast shape follows ``exp_phi``).

    Returns:
        Tensor broadcastable against ``Sigma`` for ``Sigma + ridge`` addition.
    """
    if exp_phi is not None:
        return gauge_covariant_eye(exp_phi, eps)
    eye = torch.eye(K, device=device, dtype=dtype)
    if batch_shape:
        eye = eye.expand(*batch_shape, K, K)
    return eps * eye
