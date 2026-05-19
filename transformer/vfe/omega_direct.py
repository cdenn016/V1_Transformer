r"""
Pure-:math:`\Omega` gauge parameterization for the /vfe path.

In the default (:math:`\phi`) parameterization, the gauge state of each token is
a Lie-algebra vector :math:`\phi_i \in \mathfrak{g}` and the group element is
recomputed at every use via :math:`\Omega_i = \exp(\phi_i \cdot G)`. In the
omega-direct parameterization, :math:`\Omega_i \in G` (block-diagonal
:math:`GL_+(K)` in /vfe) is itself the state. Gradients flow on the group
manifold and updates use the Lie-group exponential retraction.

# Update rule (right-invariant natural gradient)

Given the Euclidean gradient :math:`\partial F / \partial \Omega_i \in
\mathbb{R}^{d_h \times d_h}` (per block), the tangent vector at
:math:`\Omega_i` is :math:`v = \Omega_i \cdot X` for :math:`X \in
\mathfrak{g}`. The right-invariant Riemannian gradient is then

.. math::
    X = \Omega_i^{-1} \cdot \tfrac{\partial F}{\partial \Omega_i},

projected onto :math:`\mathrm{span}(G^a)` (the generator subspace, preserving
block structure) and Killing-preconditioned to match the existing
:math:`\phi`-mode metric. The retraction is

.. math::
    \Omega_i^{\text{new}} = \Omega_i \cdot \exp(-\eta \cdot X),
    \quad \Omega_i^{-1, \text{new}} = \exp(\eta \cdot X) \cdot \Omega_i^{-1}.

Two batched matrix exponentials per step — no explicit ``torch.linalg.inv``
in the hot path, no condition-number worry, and the
``(\Omega_i, \Omega_i^{-1})`` pair stays consistent without a separate
inversion step.

# Determinant drift control

The Killing form on :math:`\mathfrak{gl}(K) = \mathfrak{sl}(K) \oplus \mathbb{R}
\cdot I` is degenerate on the :math:`\mathbb{R}\cdot I` (trace/det) direction —
the Riemannian metric provides no restoring force. Without periodic
renormalization, :math:`\det(\Omega_i)` can wander to arbitrary scales over
many steps. :func:`project_omega_to_slk` rescales each block to
:math:`\det = 1`; call it on a cadence (typically every ``n`` steps, not every
step) controlled by :attr:`VFEConfig.omega_normalize_every`.

# Equivalence to φ-mode

In the :math:`\phi`-mode regime the update is :math:`\Omega = \exp((\phi - \eta
g)\cdot G)`; omega-direct applies :math:`\Omega \cdot \exp(-\eta X)`. These
agree iff :math:`[\phi \cdot G,\, X] = 0`. For non-abelian gauge groups
BCH corrections at order :math:`\eta^2[\phi \cdot G,\, X]` accumulate and the
two modes diverge per step. **Don't bill omega-direct as "equivalent up to
BCH"**: it's "no chart constraint near identity, exact retraction on the group
manifold."
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from transformer.core.gauge_utils import fused_block_matrix_exp_pairs


# -----------------------------------------------------------------------------
# Initialization
# -----------------------------------------------------------------------------


def init_omega_from_phi(
    phi: torch.Tensor,
    generators: torch.Tensor,
    irrep_dims: List[int],
    skew_symmetric: Optional[bool] = None,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    r"""Build per-block :math:`(\Omega_i, \Omega_i^{-1})` pairs from a Lie-algebra
    :math:`\phi`.

    Used at encode time (the PriorBank path) so omega-direct starts at the same
    point that :math:`\phi`-mode would start at. After this initialization the
    gradient flows on :math:`\Omega` directly and :math:`\phi` is no longer
    consulted by the E-step.

    Args:
        phi: ``(B, N, n_gen)``.
        generators: ``(n_gen, K, K)``.
        irrep_dims: Per-block dims.
        skew_symmetric: Optional override for the SO(N) fast-exp detection.

    Returns:
        ``[(Omega_h, Omega_h_inv)] × len(irrep_dims)``, each of shape
        ``(B, N, d_h, d_h)``. ``Omega_h_inv`` is the second element of the
        :func:`fused_block_matrix_exp_pairs` return tuple — exactly the
        :math:`\exp(-\phi \cdot G^{(h)})` for the same block.
    """
    phi_3d = phi if phi.dim() == 3 else phi.unsqueeze(0)
    pairs = fused_block_matrix_exp_pairs(
        phi_3d, generators, irrep_dims,
        skew_symmetric=skew_symmetric,
        only_forward=False,
    )
    if phi.dim() == 2:
        # The original wasn't batched; squeeze the leading axis back out.
        pairs = [
            (p[0].squeeze(0), p[1].squeeze(0) if p[1] is not None else None)
            for p in pairs
        ]
    # Normalize the SO(N) fast-exp shape (which can return None for the
    # inverse half) into a concrete (Omega, Omega_inv) tuple by transposing.
    out = []
    for h, (omega_h, omega_inv_h) in enumerate(pairs):
        if omega_inv_h is None:
            omega_inv_h = omega_h.transpose(-1, -2)
        out.append((omega_h, omega_inv_h))
    return out


# -----------------------------------------------------------------------------
# Pairwise Omega from per-token Omega
# -----------------------------------------------------------------------------


def compute_pairwise_omega_from_endpoints(
    omega_per_token: List[Tuple[torch.Tensor, torch.Tensor]],
    irrep_dims: List[int],
    delta: Optional[torch.Tensor] = None,
    generators: Optional[torch.Tensor] = None,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    r"""Build pairwise :math:`(\Omega_{ij}, \Omega_{ij}^{-1})` from per-token
    endpoints, with optional non-flat correction :math:`\exp(\delta_{ij} \cdot G)`
    sandwiched between :math:`\Omega_i` and :math:`\Omega_j^{-1}`.

    Flat (no :math:`\delta`):

    .. math::
        \Omega^{(h)}_{ij} = \Omega^{(h)}_i \cdot \Omega^{(h)-1}_j,
        \quad \Omega^{(h)-1}_{ij} = \Omega^{(h)}_j \cdot \Omega^{(h)-1}_i.

    Non-flat (with :math:`\delta`):

    .. math::
        \Omega^{(h)}_{ij} = \Omega^{(h)}_i \cdot \exp(\delta_{ij}\cdot G^{(h)})
            \cdot \Omega^{(h)-1}_j.

    Args:
        omega_per_token: Per-block ``(Omega_i, Omega_i^{-1})`` pairs each of
            shape ``(B, N, d_h, d_h)``.
        irrep_dims: Per-block dims.
        delta: Optional ``(B, N, N, n_gen)`` connection.
        generators: Required when ``delta is not None``.

    Returns:
        ``[(Omega_ij, Omega_ij^{-1})] × len(irrep_dims)``, each
        ``(B, N, N, d_h, d_h)``.
    """
    use_delta = delta is not None
    if use_delta and generators is None:
        raise ValueError(
            "compute_pairwise_omega_from_endpoints: delta is given but "
            "generators is None — generators are required to build "
            "exp(delta · G) per irrep block."
        )

    out = []
    block_start = 0
    for h, d_h in enumerate(irrep_dims):
        block_end = block_start + d_h
        Om_i, Om_i_inv = omega_per_token[h]   # (B, N, d_h, d_h)

        if use_delta:
            G_h = generators[:, block_start:block_end, block_start:block_end]
            algebra = torch.einsum('bija,akl->bijkl', delta, G_h)
            with torch.amp.autocast('cuda', enabled=False):
                exp_delta = torch.linalg.matrix_exp(algebra.float())
                exp_neg_delta = torch.linalg.matrix_exp(-algebra.float())
            # Cast back to original dtype if needed (matrix_exp forces fp32).
            target_dtype = Om_i.dtype
            exp_delta = exp_delta.to(target_dtype)
            exp_neg_delta = exp_neg_delta.to(target_dtype)
            Omega_ij = (
                Om_i.unsqueeze(2) @ exp_delta @ Om_i_inv.unsqueeze(1)
            )  # (B, N, N, d_h, d_h)
            # Inverse: (Ω_i exp(δ) Ω_j^{-1})^{-1} = Ω_j exp(-δ) Ω_i^{-1}
            Omega_ij_inv = (
                Om_i.unsqueeze(1) @ exp_neg_delta @ Om_i_inv.unsqueeze(2)
            )
        else:
            # Flat: Ω_ij = Ω_i Ω_j^{-1}
            # Use broadcasting: Om_i (B, N, d, d) on axis i; Om_i_inv (B, N, d, d) on axis j.
            Omega_ij = Om_i.unsqueeze(2) @ Om_i_inv.unsqueeze(1)
            Omega_ij_inv = Om_i.unsqueeze(1) @ Om_i_inv.unsqueeze(2)

        out.append((Omega_ij, Omega_ij_inv))
        block_start = block_end
    return out


# -----------------------------------------------------------------------------
# Riemannian update on Omega
# -----------------------------------------------------------------------------


def _build_killing_matrix_per_block(
    generators: torch.Tensor,
    irrep_dims: List[int],
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    r"""Per-block (Frobenius-Gram, Frobenius-Gram-inverse) of the generators
    restricted to that block. Used to decompose a ``d_h × d_h`` algebra
    direction back onto the generator basis.

    .. note::
       Name is historical. The implementation builds the **Frobenius gram**
       :math:`M_{ab} = \langle G^a, G^b \rangle_F`, not the Killing form
       :math:`B(X,Y) = 2n \cdot \mathrm{tr}(XY) - 2 \mathrm{tr}(X)\mathrm{tr}(Y)`
       on :math:`\mathfrak{gl}(n)`. The two coincide up to a positive factor
       only for trace-free skew bases — neither holds for ``GL(K)``. The
       function is suitable as a projection metric (which is what callers
       need), not as a natural-gradient preconditioner derived from the
       Killing form. The companion ``core/gauge_preconditioner.py`` does
       implement a Killing-style metric.

    Frobenius inner product :math:`\langle X, G^a \rangle_F = \mathrm{tr}(X^\top
    G^a)` defines the projection coefficient :math:`c_a = (M^{-1})_{ab}
    \langle X, G^b \rangle_F` where :math:`M_{ab} = \langle G^a, G^b \rangle_F`.

    Block-diagonal generators have :math:`M` that is block-diagonal in the
    generator index too (generators of different blocks are Frobenius-
    orthogonal), so we restrict to per-block subsets of the generator basis.

    Returns:
        For each irrep block, ``(idx_for_block, M_inv_block)`` where
        ``idx_for_block`` is a 1-D tensor of generator indices that act on
        this block, and ``M_inv_block`` is the inverse Gram restricted to
        those indices, shape ``(n_a_h, n_a_h)``.
    """
    n_gen, K, _ = generators.shape
    out = []
    block_start = 0
    for h, d_h in enumerate(irrep_dims):
        block_end = block_start + d_h
        # Identify generators acting on this block (Frobenius mass localized).
        gens_block = generators[:, block_start:block_end, block_start:block_end]
        mass_in = (gens_block ** 2).sum(dim=(-2, -1))           # (n_gen,)
        mass_full = (generators ** 2).sum(dim=(-2, -1)).clamp(min=1e-12)
        in_block = (mass_in / mass_full) > 0.9                  # threshold for "lives in this block"
        idx = in_block.nonzero(as_tuple=True)[0]                # (n_a_h,)
        if idx.numel() == 0:
            # No generators act on this block; the algebra direction is empty.
            out.append((idx, torch.zeros(0, 0, device=generators.device, dtype=generators.dtype)))
            block_start = block_end
            continue
        G_block_subset = gens_block[idx]                        # (n_a_h, d_h, d_h)
        # Gram matrix: M[a, b] = <G^a, G^b>_F.
        gram = torch.einsum('akl,bkl->ab', G_block_subset, G_block_subset)
        # Add a tiny diagonal jitter for safe inversion when generators are
        # not orthonormal but linearly independent.
        eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        gram_inv = torch.linalg.inv(gram + 1e-10 * eye)
        out.append((idx, gram_inv))
        block_start = block_end
    return out


def project_grad_to_algebra(
    grad_X: torch.Tensor,
    block_idx: torch.Tensor,
    block_gram_inv: torch.Tensor,
    generators_block: torch.Tensor,
) -> torch.Tensor:
    r"""Project an arbitrary :math:`d_h \times d_h` direction onto
    :math:`\mathrm{span}(G^a_{\text{block}})`.

    Decomposes :math:`X = \sum_a c_a G^a` where :math:`c_a = (M^{-1})_{ab}
    \langle X, G^b \rangle_F`. Returns the reconstructed :math:`\hat X
    = \sum_a c_a G^a`, which lies in the block-respecting subspace of
    :math:`\mathfrak{gl}(d_h)`.

    Args:
        grad_X: ``(..., d_h, d_h)``.
        block_idx: 1-D tensor of generator indices (within the block subset).
            Unused here directly — passed for symmetry with the cache.
        block_gram_inv: ``(n_a_h, n_a_h)``.
        generators_block: ``(n_a_h, d_h, d_h)`` — the generators restricted to
            this block, already subset by block_idx.

    Returns:
        :math:`(..., d_h, d_h)` projected direction.
    """
    if generators_block.shape[0] == 0:
        return torch.zeros_like(grad_X)
    # coeffs[..., b] = <X, G^b>_F
    coeffs = torch.einsum('...kl,bkl->...b', grad_X, generators_block)
    # Apply Gram inverse.
    coeffs_metric = torch.einsum('ab,...b->...a', block_gram_inv, coeffs)
    # Reconstitute.
    X_proj = torch.einsum('...a,akl->...kl', coeffs_metric, generators_block)
    return X_proj


def omega_natural_grad_step(
    omega_per_token: List[Tuple[torch.Tensor, torch.Tensor]],
    grad_F_omega: List[torch.Tensor],
    generators: torch.Tensor,
    irrep_dims: List[int],
    killing_cache: List[Tuple[torch.Tensor, torch.Tensor]],
    lr: float,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    r"""One right-invariant Riemannian step.

    For each block:

    .. math::
        X = \mathrm{proj}_{\mathrm{span}(G^a)} \bigl( \Omega^{-1} \cdot \partial F / \partial \Omega \bigr),
        \quad \Omega_{\text{new}} = \Omega \cdot \exp(-\eta X),
        \quad \Omega_{\text{new}}^{-1} = \exp(\eta X) \cdot \Omega^{-1}.

    Args:
        omega_per_token: ``[(Omega_h, Omega_h_inv)] × len(irrep_dims)``.
        grad_F_omega: ``[dF/dOmega_h] × len(irrep_dims)``. Each
            ``(B, N, d_h, d_h)``.
        generators: ``(n_gen, K, K)``.
        irrep_dims: Per-block dims.
        killing_cache: Output of :func:`_build_killing_matrix_per_block`.
        lr: Step size.

    Returns:
        Updated ``[(Omega_h, Omega_h_inv)]``.
    """
    out = []
    block_start = 0
    for h, d_h in enumerate(irrep_dims):
        block_end = block_start + d_h
        Om, Om_inv = omega_per_token[h]
        dF_dOm = grad_F_omega[h]

        # X = Ω^{-1} · dF/dΩ  — right-invariant translate.
        X_unprojected = torch.einsum('...kl,...lm->...km', Om_inv, dF_dOm)

        # Project onto generator span for the block, preserving block structure.
        block_idx, gram_inv = killing_cache[h]
        if block_idx.numel() == 0:
            # Empty algebra direction for this block — no update.
            out.append((Om, Om_inv))
            block_start = block_end
            continue
        G_block = generators[block_idx][:, block_start:block_end, block_start:block_end]
        X = project_grad_to_algebra(X_unprojected, block_idx, gram_inv, G_block)

        # Retract.
        with torch.amp.autocast('cuda', enabled=False):
            step = (-lr * X).float()
            neg_step = (lr * X).float()
            exp_neg = torch.linalg.matrix_exp(step)
            exp_pos = torch.linalg.matrix_exp(neg_step)
        target_dtype = Om.dtype
        exp_neg = exp_neg.to(target_dtype)
        exp_pos = exp_pos.to(target_dtype)

        Om_new = torch.einsum('...kl,...lm->...km', Om, exp_neg)
        Om_inv_new = torch.einsum('...kl,...lm->...km', exp_pos, Om_inv)

        out.append((Om_new, Om_inv_new))
        block_start = block_end
    return out


# -----------------------------------------------------------------------------
# Determinant renormalization
# -----------------------------------------------------------------------------


def project_omega_to_slk(
    omega_per_token: List[Tuple[torch.Tensor, torch.Tensor]],
    irrep_dims: List[int],
    eps: float = 1e-12,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    r"""Rescale each per-block :math:`\Omega_h` so that :math:`\det(\Omega_h)
    = 1`.

    For per-token :math:`(B, N, d_h, d_h)` blocks::

        det_h = det(Omega_h)
        scale = det_h.sign() · |det_h|^(1/d_h)
        Omega_h  ← Omega_h / scale
        Omega_inv_h ← Omega_inv_h · scale

    The sign factor preserves orientation (keeps :math:`\Omega \in GL_+(d_h)`).
    Apply on a cadence — every step is wasteful (the natural-gradient direction
    is partially undone by the rescale) and noise is amplified at fp16.

    Args:
        omega_per_token: ``[(Omega_h, Omega_h_inv)]``.
        irrep_dims: Per-block dims.
        eps: Floor for |det|.

    Returns:
        Renormalized pairs.
    """
    out = []
    for h, d_h in enumerate(irrep_dims):
        Om, Om_inv = omega_per_token[h]
        with torch.amp.autocast('cuda', enabled=False):
            det = torch.linalg.det(Om.float())                      # (B, N)
        scale = det.sign() * det.abs().clamp(min=eps).pow(1.0 / d_h)  # (B, N)
        # Avoid zero / sign-zero issues at exact zero.
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        scale_view = scale.unsqueeze(-1).unsqueeze(-1)
        Om_new = Om / scale_view
        Om_inv_new = Om_inv * scale_view
        out.append((Om_new.to(Om.dtype), Om_inv_new.to(Om_inv.dtype)))
    return out
