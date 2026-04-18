r"""
Gauge Geometry: Curvature, Yang-Mills Energy, and Gauge Orbits
================================================================

Computes the curvature tensor $F$ from holonomy matrices $C_{ijk}$,
the Yang-Mills energy $E_{\text{YM}} = \frac{1}{2}\sum \|F\|_F^2$,
and characterizes gauge orbits under $\mathrm{GL}^+(K)$ transformations.

For **flat transport** ($\Omega_{ij} = e^{\phi_i} e^{-\phi_j}$, cocycle),
holonomy $C_{ijk} = I$ exactly and $E_{\text{YM}} = 0$. In this regime
the gauge field Dirichlet energy $E_D = \sum \beta_{ij}\|\phi_i - \phi_j\|^2$
serves as the meaningful geometric diagnostic.

For **non-flat transport** (with learned connection $\delta_{ij}$),
the curvature is non-trivial and is extracted via the matrix logarithm
of the holonomy: $F_{ijk} = \log(C_{ijk})$.
"""

from __future__ import annotations

import torch
from torch import Tensor
from typing import Dict, List, Tuple

from transformer.analysis.holonomy import compute_holonomy
from transformer.core.gauge_utils import stable_matrix_exp_pair, newton_schulz_orthogonalize


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_float32(t: Tensor) -> Tensor:
    r"""Cast tensor to float32; no-op if already float32 or float64."""
    if t.dtype in (torch.float32, torch.float64):
        return t.float()
    return t.float()


def _build_phi_matrix(phi_vec: Tensor, generators: Tensor) -> Tensor:
    r"""Contract gauge-field coefficients with generators.

    Args:
        phi_vec: ``(\ldots, n\_gen)`` coefficient vector.
        generators: ``(n\_gen, K, K)`` Lie algebra basis.

    Returns:
        ``(\ldots, K, K)`` Lie algebra element $\phi \cdot G =
        \sum_a \phi^a T_a$.
    """
    # einsum: ...a, aij -> ...ij
    return torch.einsum("...a,aij->...ij", phi_vec, generators)


# ---------------------------------------------------------------------------
# 1. Curvature tensor extraction
# ---------------------------------------------------------------------------

def extract_curvature_tensor(
    C: Tensor,
    method: str = "logm",
    series_order: int = 4,
) -> Tensor:
    r"""Extract the curvature 2-form $F_{ijk}$ from holonomy matrices.

    The curvature tensor is defined as the matrix logarithm of the holonomy:

    .. math::

        F_{ijk} = \log C_{ijk}

    so that $C_{ijk} = e^{F_{ijk}}$ and $F_{ijk} = 0$ if and only if transport
    is flat ($C_{ijk} = I$).

    Two numerical strategies are provided:

    **logm** (default): eigendecompose $C = V \Lambda V^{-1}$, apply the
    complex logarithm to each eigenvalue (clamping the modulus away from zero
    to prevent $\log 0$), then reconstruct
    $F = \mathrm{Re}(V \operatorname{diag}(\log \lambda) V^{-1})$.
    This is the standard principal matrix logarithm for matrices with no
    negative real eigenvalues; the real part discards imaginary rounding
    noise for real input matrices.

    **series**: Mercator (Taylor) series
    $\log(I + X) = X - X^2/2 + X^3/3 - \ldots$ where $X = C - I$.
    Accurate only when $\|C - I\|_F \ll 1$ (near-flat holonomy).

    Args:
        C: ``(B, n_triples, K, K)`` holonomy matrices from
            :func:`~transformer.analysis.holonomy.compute_holonomy`.
        method: ``'logm'`` (eigendecomposition, stable) or
            ``'series'`` (Taylor, only near-flat).
        series_order: Truncation order for the series method. Default 4.

    Returns:
        ``(B, n_triples, K, K)`` curvature tensors $F_{ijk}$.
    """
    with torch.no_grad():
        C_f32 = _to_float32(C)
        B, T, K, _ = C_f32.shape
        C_flat = C_f32.reshape(B * T, K, K)  # (B*T, K, K)

        if method == "series":
            # Mercator series: log(I + X) = sum_{n=1}^{order} (-1)^{n+1} X^n / n
            X = C_flat - torch.eye(K, device=C_flat.device, dtype=C_flat.dtype).unsqueeze(0)
            F_flat = torch.zeros_like(X)
            X_power = X.clone()
            for n in range(1, series_order + 1):
                sign = (-1.0) ** (n + 1)
                F_flat = F_flat + (sign / n) * X_power
                if n < series_order:
                    X_power = torch.bmm(X_power, X)
            return F_flat.reshape(B, T, K, K)

        # method == 'logm': eigendecomposition path
        # torch.linalg.eig returns complex eigenvalues/vectors for real input
        # Shape: eigenvalues (B*T, K) complex, eigenvectors (B*T, K, K) complex
        eigenvalues, V = torch.linalg.eig(C_flat)

        # Complex log of eigenvalues.
        # Clamp modulus to [eps, inf) to avoid log(0).
        eps = 1e-6  # Above float32 machine epsilon (~1.2e-7)
        ev_mod = eigenvalues.abs().clamp(min=eps)  # (B*T, K)
        ev_phase = eigenvalues.angle()              # (B*T, K)
        log_ev = torch.log(ev_mod) + 1j * ev_phase  # (B*T, K) complex

        # Reconstruct: F = V @ diag(log_ev) @ V^{-1}
        # For numerical safety use linalg.solve instead of explicit inverse:
        #   V @ diag(log_ev) @ V^{-1} = (solve(V^T, (V @ diag(log_ev))^T))^T
        log_ev_diag = torch.diag_embed(log_ev)  # (B*T, K, K) complex
        V_log = torch.bmm(V, log_ev_diag)       # (B*T, K, K) complex

        try:
            # Solve V^T X = (V_log)^T  →  X = F^T  →  F = X^T
            F_complex = torch.linalg.solve(
                V.transpose(-1, -2), V_log.transpose(-1, -2)
            ).transpose(-1, -2)
        except RuntimeError:
            # Fallback: explicit pseudo-inverse
            V_inv = torch.linalg.pinv(V)
            F_complex = torch.bmm(V_log, V_inv)

        # Discard imaginary rounding noise for real matrices
        F_flat = F_complex.real.to(dtype=C_f32.dtype)  # (B*T, K, K)
        return F_flat.reshape(B, T, K, K)


# ---------------------------------------------------------------------------
# 2. Yang-Mills energy
# ---------------------------------------------------------------------------

def compute_yang_mills_energy(F: Tensor) -> Tensor:
    r"""Yang-Mills energy from curvature tensors.

    .. math::

        E_{\text{YM}} = \frac{1}{2} \sum_{t} \|F_t\|_F^2

    where the sum is over all holonomy triples $t = (i,j,k)$.

    Args:
        F: ``(B, n_triples, K, K)`` curvature tensors from
            :func:`extract_curvature_tensor`.

    Returns:
        ``(B,)`` Yang-Mills energy per batch element.
    """
    with torch.no_grad():
        # Frobenius norm squared: sum over last two dims, then over triples
        frob_sq = (F * F).sum(dim=(-2, -1))  # (B, n_triples)
        return 0.5 * frob_sq.sum(dim=-1)     # (B,)


# ---------------------------------------------------------------------------
# 3. End-to-end Yang-Mills from holonomy
# ---------------------------------------------------------------------------

def compute_yang_mills_energy_from_holonomy(
    exp_delta: Tensor,
    sample_size: int = 1000,
    method: str = "logm",
) -> Tuple[Tensor, Dict[str, float]]:
    r"""End-to-end pipeline: $\exp(\delta) \to C_{ijk} \to F \to E_{\text{YM}}$.

    Calls :func:`~transformer.analysis.holonomy.compute_holonomy` to sample
    holonomy matrices, extracts the curvature tensor via
    :func:`extract_curvature_tensor`, and sums to produce the Yang-Mills
    energy.  Also computes a diagnostics dictionary characterising how
    non-flat the learned connection is.

    Args:
        exp_delta: ``(B, N, N, K, K)`` — $\exp(\delta_{ij} \cdot G)$ per edge,
            as produced by the non-flat transport branch.
        sample_size: Number of random triples $(i,j,k)$ to sample.
        method: Matrix-log method passed to :func:`extract_curvature_tensor`.

    Returns:
        A tuple ``(energy, diagnostics)`` where:

        - ``energy`` is ``(B,)`` Yang-Mills energy per batch element.
        - ``diagnostics`` is a :class:`dict` with float scalars:

          - ``mean_F_norm``: mean $\|F_t\|_F$ over batch and triples.
          - ``max_F_norm``: maximum $\|F_t\|_F$.
          - ``mean_holonomy_norm``: mean $\|C_t - I\|_F$.
          - ``abelian_fraction``: mean fraction of curvature energy in the
            abelian (trace) sector, from :func:`decompose_curvature`.
    """
    with torch.no_grad():
        C, norms, _ = compute_holonomy(exp_delta, sample_size=sample_size)
        F = extract_curvature_tensor(C, method=method)
        energy = compute_yang_mills_energy(F)

        # Diagnostics
        frob_F = (F * F).sum(dim=(-2, -1)).sqrt()  # (B, n_triples)
        decomp = decompose_curvature(F)

        abelian_frac = decomp["abelian_fraction"].mean().item()

        diagnostics: Dict[str, float] = {
            "mean_F_norm": float(frob_F.mean().item()),
            "max_F_norm": float(frob_F.max().item()),
            "mean_holonomy_norm": float(norms.detach().cpu().float().mean().item()),
            "abelian_fraction": abelian_frac,
        }
        return energy, diagnostics


# ---------------------------------------------------------------------------
# 4. Gauge field Dirichlet energy
# ---------------------------------------------------------------------------

def compute_gauge_field_energy(
    phi: Tensor,
    beta: Tensor,
    generators: Tensor,
    metric: str = "frobenius",
) -> Tensor:
    r"""Gauge field Dirichlet energy on the token graph.

    .. math::

        E_D = \sum_{i,j} \beta_{ij} \|\phi_i - \phi_j\|^2_g

    where $g$ is either the Frobenius or Killing form metric on
    $\mathfrak{gl}(K)$.

    **Frobenius metric**: $\|\Delta\phi\|^2 = \Delta\phi^T \Delta\phi =
    \sum_a (\phi^a_i - \phi^a_j)^2$.

    **Killing metric**: $\|\Delta\phi\|^2_{\mathrm{Kill}} =
    \Delta\phi^T \, G_{\mathrm{Kill}} \, \Delta\phi$ where the Killing form
    on $\mathfrak{gl}(K)$ is

    .. math::

        G_{ab} = 2K \operatorname{tr}(T_a^T T_b)
                 - 2 \operatorname{tr}(T_a) \operatorname{tr}(T_b)

    i.e.\ $\mathrm{tr}(\mathrm{ad}(T_a) \circ \mathrm{ad}(T_b))$ evaluated
    for the defining representation.

    Args:
        phi: ``(B, N, n_gen)`` gauge-field coefficients.
        beta: ``(B, N, N)`` attention weights $\beta_{ij}$.
        generators: ``(n_gen, K, K)`` Lie algebra generators.
        metric: ``'frobenius'`` or ``'killing'``.

    Returns:
        ``(B,)`` Dirichlet energy per batch element.
    """
    with torch.no_grad():
        phi_f32 = _to_float32(phi)
        beta_f32 = _to_float32(beta)

        # Delta phi: (B, N, N, n_gen)  — phi_i - phi_j for all pairs
        delta_phi = phi_f32.unsqueeze(2) - phi_f32.unsqueeze(1)  # (B, N, N, n_gen)

        if metric == "frobenius":
            # ||Δφ||² = sum_a (Δφ^a)²
            delta_sq = (delta_phi * delta_phi).sum(dim=-1)  # (B, N, N)

        elif metric == "killing":
            # Build Killing form gram matrix: G_ab = 2K tr(T_a^T T_b) - 2 tr(T_a)tr(T_b)
            gen_f32 = _to_float32(generators)  # (n_gen, K, K)
            n_gen, K, _ = gen_f32.shape

            # Killing form uses tr(T_a^T T_b) (Frobenius inner product),
            # NOT tr(T_a T_b) (matrix product trace). For skew-symmetric
            # generators these coincide, but for GL(K) they differ.
            # The Cartan-involution-modified Killing form is:
            #   g̃(X,Y) = 2K tr(X^T Y) - 2 tr(X) tr(Y)
            # which requires tr(T_a^T T_b) = Σ_{k,l} T_a[k,l] T_b[k,l].
            gen_flat = gen_f32.reshape(n_gen, K * K)
            gram = torch.mm(gen_flat, gen_flat.t())  # tr(T_a^T T_b), (n_gen, n_gen)

            # Traces: tr(T_a) = sum_k T_a[k,k]
            traces = torch.diagonal(gen_f32, dim1=-2, dim2=-1).sum(dim=-1)  # (n_gen,)
            trace_outer = traces.unsqueeze(1) * traces.unsqueeze(0)         # (n_gen, n_gen)

            killing_metric = 2.0 * K * gram - 2.0 * trace_outer             # (n_gen, n_gen)

            # ||Δφ||²_Kill = Δφ^T @ killing_metric @ Δφ
            # delta_phi: (B, N, N, n_gen) → batch matmul
            # delta_phi @ killing_metric: (B, N, N, n_gen)
            dphi_K = torch.einsum("bija,ac->bijc", delta_phi, killing_metric)  # (B, N, N, n_gen)
            delta_sq = (dphi_K * delta_phi).sum(dim=-1)  # (B, N, N)

        else:
            raise ValueError(
                f"compute_gauge_field_energy: unknown metric '{metric}'. "
                "Choose 'frobenius' or 'killing'."
            )

        # Weighted sum: E_D = sum_{i,j} beta_ij * ||phi_i - phi_j||^2
        energy = (beta_f32 * delta_sq).sum(dim=(-2, -1))  # (B,)
        return energy


# ---------------------------------------------------------------------------
# 5. Curvature decomposition
# ---------------------------------------------------------------------------

def decompose_curvature(F: Tensor) -> Dict[str, Tensor]:
    r"""Decompose curvature into abelian (trace) and non-abelian (traceless) parts.

    Every element of $\mathfrak{gl}(K)$ splits as

    .. math::

        F = F_{\text{ab}} + F_{\text{nab}}, \qquad
        F_{\text{ab}} = \frac{\operatorname{tr}(F)}{K} I_K, \qquad
        F_{\text{nab}} = F - F_{\text{ab}}.

    $F_{\text{ab}}$ lives in $\mathfrak{u}(1) \subset \mathfrak{gl}(K)$ (the
    centre) and measures the $U(1)$ electromagnetic-like curvature; $F_{\text{nab}}$
    is the non-abelian (traceless) remainder carrying $\mathfrak{sl}(K)$ content.

    Args:
        F: ``(B, n_triples, K, K)`` curvature tensors.

    Returns:
        Dictionary with keys:

        - ``'F_abelian'``: ``(B, n_triples, K, K)`` abelian component.
        - ``'F_nonabelian'``: ``(B, n_triples, K, K)`` non-abelian component.
        - ``'abelian_energy'``: ``(B,)`` $\frac{1}{2}\|F_{\text{ab}}\|_F^2$.
        - ``'nonabelian_energy'``: ``(B,)`` $\frac{1}{2}\|F_{\text{nab}}\|_F^2$.
        - ``'abelian_fraction'``: ``(B,)`` fraction of total energy that is abelian.
    """
    with torch.no_grad():
        F_f32 = _to_float32(F)
        B, T, K, _ = F_f32.shape

        # tr(F) / K  → scalar per (B, T)
        trace_F = torch.diagonal(F_f32, dim1=-2, dim2=-1).sum(dim=-1)  # (B, T)
        trace_coeff = trace_F / K                                        # (B, T)

        I_K = torch.eye(K, device=F_f32.device, dtype=F_f32.dtype)      # (K, K)
        # F_abelian: (B, T, K, K)
        F_abelian = trace_coeff.unsqueeze(-1).unsqueeze(-1) * I_K        # broadcast
        F_nonabelian = F_f32 - F_abelian

        # Energies
        E_ab = 0.5 * (F_abelian * F_abelian).sum(dim=(-2, -1)).sum(dim=-1)   # (B,)
        E_nab = 0.5 * (F_nonabelian * F_nonabelian).sum(dim=(-2, -1)).sum(dim=-1)  # (B,)
        E_total = E_ab + E_nab

        # Guard against division by zero (all-flat case where E_total = 0)
        ab_frac = E_ab / E_total.clamp(min=1e-12)  # (B,)

        return {
            "F_abelian": F_abelian,
            "F_nonabelian": F_nonabelian,
            "abelian_energy": E_ab,
            "nonabelian_energy": E_nab,
            "abelian_fraction": ab_frac,
        }


# ---------------------------------------------------------------------------
# 6. Axial gauge fixing
# ---------------------------------------------------------------------------

def gauge_fix_axial(
    mu: Tensor,
    sigma: Tensor,
    phi: Tensor,
    generators: Tensor,
    anchor_idx: int = 0,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    r"""Fix the gauge by setting $\phi_{\text{anchor}} = 0$.

    The axial gauge condition selects a canonical representative of the gauge
    orbit by applying the group element
    $g = \exp(-\phi_{\text{anchor}} \cdot G)$ to every belief:

    .. math::

        \mu_i \to g\,\mu_i, \qquad
        \Sigma_i \to g\,\Sigma_i\,g^T, \qquad
        \phi_i \to \phi_i - \phi_{\text{anchor}}.

    After the transform the anchor token satisfies $\phi_{\text{anchor}} = 0$,
    i.e.\ its gauge frame is the identity.  The KL-attention matrix is
    invariant under this global gauge transformation by construction.

    For **diagonal** covariance $\Sigma = \operatorname{diag}(\sigma)$, the
    transported diagonal variance is

    .. math::

        (\tilde{\sigma}_i)_k = \sum_l g_{kl}^2 \, \sigma_{il}
        = [g \operatorname{diag}(\sigma_i) g^T]_{kk}.

    Args:
        mu: ``(B, N, K)`` belief means.
        sigma: ``(B, N, K)`` diagonal variances.
        phi: ``(B, N, n_gen)`` gauge-field coefficients.
        generators: ``(n_gen, K, K)`` Lie algebra generators.
        anchor_idx: Index of the token to set as gauge origin. Default 0.

    Returns:
        Tuple ``(mu_fixed, sigma_fixed, phi_fixed, g)`` where:

        - ``mu_fixed``: ``(B, N, K)`` gauge-fixed means.
        - ``sigma_fixed``: ``(B, N, K)`` gauge-fixed diagonal variances.
        - ``phi_fixed``: ``(B, N, n_gen)`` gauge-fixed coefficients.
        - ``g``: ``(B, K, K)`` the gauge transformation applied.
    """
    with torch.no_grad():
        mu_f32 = _to_float32(mu)
        sigma_f32 = _to_float32(sigma)
        phi_f32 = _to_float32(phi)
        gen_f32 = _to_float32(generators)

        B, N, K = mu_f32.shape

        # phi_anchor: (B, n_gen)
        phi_anchor = phi_f32[:, anchor_idx, :]  # (B, n_gen)

        # Build Lie algebra element: -phi_anchor · G  →  (B, K, K)
        neg_phi_matrix = _build_phi_matrix(-phi_anchor, gen_f32)  # (B, K, K)

        # g = exp(-phi_anchor · G), g_inv not needed (for analysis only)
        g, _ = stable_matrix_exp_pair(neg_phi_matrix, only_forward=True)  # (B, K, K)

        # mu_fixed: g @ mu_i  →  (B, N, K)
        mu_fixed = torch.einsum("bkl,bnl->bnk", g, mu_f32)

        # sigma_fixed: diagonal of g @ diag(sigma_i) @ g^T
        # (g @ diag(σ) @ g^T)_{kk} = sum_l g_{kl}^2 * sigma_l
        g_sq = g * g  # (B, K, K) — element-wise square
        sigma_fixed = torch.einsum("bkl,bnl->bnk", g_sq, sigma_f32)

        # phi_fixed: exact gauge transformation log(exp(-φ_anchor·G) · exp(φ_i·G))
        # The first-order approximation φ_i - φ_anchor misses commutator
        # corrections [φ_anchor, φ_i]/2 + ... which matter for non-abelian
        # groups with large φ.  Use exact matrix logarithm instead.
        phi_anchor_matrix = _build_phi_matrix(phi_anchor, gen_f32)  # (B, K, K)
        neg_anchor_matrix = -phi_anchor_matrix  # (B, K, K)
        phi_i_matrices = torch.einsum('bna,aij->bnij', phi_f32, gen_f32)  # (B, N, K, K)

        # exp(-φ_anchor) · exp(φ_i) for each token
        exp_neg_anchor = torch.linalg.matrix_exp(neg_anchor_matrix)  # (B, K, K)
        exp_phi_i = torch.linalg.matrix_exp(phi_i_matrices)  # (B, N, K, K)
        composed = torch.einsum('bij,bnjk->bnik', exp_neg_anchor, exp_phi_i)  # (B, N, K, K)

        # Extract φ_fixed via matrix logarithm: φ_fixed·G = log(composed)
        # For near-identity composed matrices, logm is well-conditioned.
        B_size, N_size = composed.shape[:2]
        composed_flat = composed.reshape(B_size * N_size, K, K)
        # Use eigendecomposition for matrix log: log(V diag(λ) V^{-1}) = V diag(log λ) V^{-1}
        eigvals, eigvecs = torch.linalg.eig(composed_flat)
        log_eigvals = torch.log(eigvals.abs().clamp(min=1e-12)) + 1j * eigvals.angle()
        try:
            eigvecs_inv = torch.linalg.solve(eigvecs.transpose(-2, -1).conj(), torch.eye(
                eigvecs.shape[-1], device=eigvecs.device, dtype=eigvecs.dtype
            ).expand_as(eigvecs)).transpose(-2, -1).conj()
        except (torch.linalg.LinAlgError, RuntimeError):
            eigvecs_inv = torch.linalg.pinv(eigvecs)
        log_matrix = eigvecs @ torch.diag_embed(log_eigvals) @ eigvecs_inv
        log_matrix = log_matrix.real.reshape(B_size, N_size, K, K)  # (B, N, K, K)

        # Extract coordinates: φ_a = tr(G_a^T · log_matrix) / ||G_a||²_F
        gen_norms_sq = (gen_f32 * gen_f32).sum(dim=(-2, -1))  # (n_gen,)
        phi_fixed = torch.einsum('aij,bnij->bna', gen_f32, log_matrix) / gen_norms_sq.unsqueeze(0).unsqueeze(0).clamp(min=1e-12)  # (B, N, n_gen)

        return (
            mu_fixed.to(mu.dtype),
            sigma_fixed.to(sigma.dtype),
            phi_fixed.to(phi.dtype),
            g.to(mu.dtype),
        )


# ---------------------------------------------------------------------------
# 7. Coulomb gauge fixing
# ---------------------------------------------------------------------------

def gauge_fix_coulomb(
    mu: Tensor,
    sigma: Tensor,
    phi: Tensor,
    generators: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    r"""Fix the gauge by setting $\bar{\phi} = \frac{1}{N}\sum_i \phi_i = 0$.

    The Coulomb (or Lorenz) gauge condition removes the global degree of
    freedom by centring the gauge field.  Equivalent to applying the axial
    gauge fix with anchor $\phi_{\text{anchor}} = \bar{\phi}$.

    After the transform $\sum_i \phi^{\text{fixed}}_i = 0$, distributing the
    gauge-frame information uniformly across tokens.

    Args:
        mu: ``(B, N, K)`` belief means.
        sigma: ``(B, N, K)`` diagonal variances.
        phi: ``(B, N, n_gen)`` gauge-field coefficients.
        generators: ``(n_gen, K, K)`` Lie algebra generators.

    Returns:
        Tuple ``(mu_fixed, sigma_fixed, phi_fixed, g)`` — same convention as
        :func:`gauge_fix_axial`.
    """
    with torch.no_grad():
        phi_f32 = _to_float32(phi)
        gen_f32 = _to_float32(generators)

        B, N, n_gen = phi_f32.shape

        # Mean gauge field: (B, n_gen)
        phi_mean = phi_f32.mean(dim=1)  # (B, n_gen)

        # Build Lie algebra element for the mean: (B, K, K)
        neg_mean_matrix = _build_phi_matrix(-phi_mean, gen_f32)  # (B, K, K)
        g, _ = stable_matrix_exp_pair(neg_mean_matrix, only_forward=True)

        mu_f32 = _to_float32(mu)
        sigma_f32 = _to_float32(sigma)

        mu_fixed = torch.einsum("bkl,bnl->bnk", g, mu_f32)

        g_sq = g * g
        sigma_fixed = torch.einsum("bkl,bnl->bnk", g_sq, sigma_f32)

        # phi_fixed: subtract mean, so sum becomes zero
        phi_fixed = phi_f32 - phi_mean.unsqueeze(1)  # (B, N, n_gen)

        return (
            mu_fixed.to(mu.dtype),
            sigma_fixed.to(sigma.dtype),
            phi_fixed.to(phi.dtype),
            g.to(mu.dtype),
        )


# ---------------------------------------------------------------------------
# 8. Gauge-invariant quantities
# ---------------------------------------------------------------------------

def compute_gauge_invariants(
    mu: Tensor,
    sigma: Tensor,
    phi: Tensor,
    generators: Tensor,
    beta: Tensor,
    enforce_orthogonal: bool = False,
) -> Dict[str, Tensor]:
    r"""Compute gauge-invariant quantities under $\mathrm{GL}^+(K)$.

    A quantity $Q$ is gauge-invariant if $Q(g \cdot \mu, g \Sigma g^T,
    \phi + \epsilon) = Q(\mu, \Sigma, \phi)$ for all $g \in \mathrm{GL}^+(K)$.
    The attention weight matrix $\beta_{ij}$ is invariant by construction
    (it depends on $\mathrm{KL}(q_i \| \Omega_{ij} q_j)$ which is unchanged
    by a global gauge transformation).

    Quantities returned:

    - **beta** (passed through): already gauge-invariant attention weights.
    - **det_omega**: $\det(\Omega_{ij}) = \exp(\operatorname{tr}(\phi_i - \phi_j))$
      for all pairs, exploiting $\det(\exp M) = \exp(\operatorname{tr} M)$ and
      $\Omega_{ij} = \exp(\phi_i \cdot G) \exp(-\phi_j \cdot G)$.
    - **gauge_frame_spectrum**: eigenvalues of $\exp(\phi_i \cdot G)$ for each
      token (spectrally invariant under conjugation by a fixed $g$).
    - **gauge_field_energy**: scalar $E_D$ from :func:`compute_gauge_field_energy`
      (invariant under global but not local gauge transforms in general).

    Args:
        mu: ``(B, N, K)`` belief means.
        sigma: ``(B, N, K)`` diagonal variances.
        phi: ``(B, N, n_gen)`` gauge-field coefficients.
        generators: ``(n_gen, K, K)`` Lie algebra generators.
        beta: ``(B, N, N)`` attention weights.

    Returns:
        Dictionary with keys ``'beta'``, ``'det_omega'``, ``'gauge_frame_spectrum'``,
        ``'gauge_field_energy'``.
    """
    with torch.no_grad():
        phi_f32 = _to_float32(phi)
        gen_f32 = _to_float32(generators)
        B, N, n_gen = phi_f32.shape
        K = gen_f32.shape[-1]

        # -- det(Omega_ij) = exp(tr(phi_i - phi_j))
        # tr(phi_i · G) = phi_i · tr(G)  — linearity of trace
        # tr_gen[a] = tr(T_a)
        # Compute in fp64: torch.exp overflows fp32 at argument ≈ 88.7, which
        # maps to |tr Δφ| ≈ 88. With H-block GL(K) + per-block trace clamp T,
        # the worst-case |tr Δφ| is 2·H·T — exceedable for H=6, T>7.3. fp64
        # overflows at ≈ 709, giving enough headroom that finite-value guard
        # below is rarely triggered.
        tr_gen = torch.diagonal(gen_f32, dim1=-2, dim2=-1).sum(dim=-1).double()  # (n_gen,)
        # tr(phi_i · G) = sum_a phi^a_i * tr(T_a): (B, N)
        trace_phi = torch.einsum("bna,a->bn", phi_f32.double(), tr_gen)  # (B, N) f64
        # tr(phi_i - phi_j): (B, N, N)
        trace_diff = trace_phi.unsqueeze(2) - trace_phi.unsqueeze(1)     # (B, N, N) f64
        det_omega = torch.exp(trace_diff).float()                         # back to f32
        # Finite-value guard: if any det overflowed even in fp64, zero them
        # and warn — downstream aggregators (mean/std/max in publication_metrics)
        # would otherwise silently propagate inf.
        if not torch.isfinite(det_omega).all():
            import warnings as _warnings
            _n_bad = int((~torch.isfinite(det_omega)).sum().item())
            _max_abs = float(trace_diff.abs().max().item())
            _warnings.warn(
                f"gauge_geometry.compute_gauge_invariants: {_n_bad} non-finite "
                f"det(Omega) values (max |tr Δφ|={_max_abs:.1f}); zeroed. "
                f"Consider phi_project_slk=True or a tighter phi_trace_clamp.",
                RuntimeWarning,
            )
            det_omega = torch.where(
                torch.isfinite(det_omega), det_omega, torch.zeros_like(det_omega)
            )

        # -- Gauge frame spectrum: eigenvalues of exp(phi_i · G) for each token
        phi_matrix = _build_phi_matrix(phi_f32, gen_f32)  # (B, N, K, K)
        phi_flat = phi_matrix.reshape(B * N, K, K)
        exp_phi_flat, _ = stable_matrix_exp_pair(phi_flat, only_forward=True)  # (B*N, K, K)

        # When enforce_orthogonal is active, project to O(K) and recompute
        # det(Omega) from the actual orthogonalized matrices (det ≈ ±1).
        if enforce_orthogonal and K >= 2:
            exp_phi_flat = newton_schulz_orthogonalize(exp_phi_flat)
            det_per_token = torch.linalg.det(exp_phi_flat).reshape(B, N)  # (B, N)
            # det(Omega_ij) = det(exp_phi_i) / det(exp_phi_j)
            det_omega = det_per_token.unsqueeze(2) / det_per_token.unsqueeze(1).clamp(min=1e-30)

        # Eigenvalues (complex) → modulus as a gauge-invariant spectral quantity
        eigvals_complex = torch.linalg.eigvals(exp_phi_flat.float())   # (B*N, K) complex
        gauge_frame_spectrum = eigvals_complex.abs().reshape(B, N, K)  # (B, N, K) real

        # -- Gauge field Dirichlet energy (Frobenius)
        gauge_field_energy = compute_gauge_field_energy(
            phi, beta, generators, metric="frobenius"
        )  # (B,)

        # -- Pairwise KL divergence (gauge-invariant by construction)
        # KL(q_i || Omega_ij q_j) is what drives attention weights.
        # Compute via the standard diagonal KL path with transported beliefs.
        kl_values = None
        try:
            from transformer.core.kl_computation import compute_kl_matrix, KLMode
            from transformer.core.gauge_utils import fused_block_matrix_exp_pairs

            mu_f32 = _to_float32(mu)
            sigma_f32 = _to_float32(sigma).clamp(min=1e-6)
            irrep_dims = [K]  # single block
            bep = fused_block_matrix_exp_pairs(
                phi_f32, gen_f32, irrep_dims,
                enforce_orthogonal=enforce_orthogonal,
            )
            kl_matrix = compute_kl_matrix(
                mu_f32, sigma_f32, None, None,
                mode=KLMode.BLOCK_DIAGONAL,
                block_exp_pairs=bep,
                irrep_dims=irrep_dims,
            )  # (B, N, N)
            kl_values = kl_matrix
        except Exception:
            pass  # KL computation is optional; fail silently

        return {
            "beta": beta,
            "det_omega": det_omega,
            "gauge_frame_spectrum": gauge_frame_spectrum,
            "gauge_field_energy": gauge_field_energy,
            "kl_values": kl_values,
        }


# ---------------------------------------------------------------------------
# 9. Gauge orbit sampling
# ---------------------------------------------------------------------------

def gauge_orbit_sample(
    mu: Tensor,
    sigma: Tensor,
    phi: Tensor,
    generators: Tensor,
    n_samples: int = 50,
    perturbation_scale: float = 0.5,
) -> List[Dict[str, Tensor]]:
    r"""Sample random points on the gauge orbit.

    For each sample draws random coefficients $\epsilon \sim \mathcal{N}(0,
    \sigma_\epsilon^2 I)$ and applies the group element
    $g = \exp(\epsilon \cdot G)$ to all beliefs:

    .. math::

        \mu_i \to g\,\mu_i, \qquad
        \Sigma_i \to g\,\Sigma_i\,g^T, \qquad
        \phi_i \to \phi_i + \epsilon.

    All sampled points are physically equivalent to the input
    $(\mu, \Sigma, \phi)$ up to a gauge transformation; gauge-invariant
    observables (KL matrix, $\det \Omega_{ij}$, spectra) are unchanged.

    Args:
        mu: ``(B, N, K)`` belief means.
        sigma: ``(B, N, K)`` diagonal variances.
        phi: ``(B, N, n_gen)`` gauge-field coefficients.
        generators: ``(n_gen, K, K)`` Lie algebra generators.
        n_samples: Number of orbit points to return. Default 50.
        perturbation_scale: Standard deviation $\sigma_\epsilon$ of the
            random Lie algebra coefficients. Default 0.5.

    Returns:
        List of length ``n_samples``, each element a dict with keys:

        - ``'mu'``: ``(B, N, K)``
        - ``'sigma'``: ``(B, N, K)``
        - ``'phi'``: ``(B, N, n_gen)``
        - ``'g'``: ``(B, K, K)`` the transformation applied.
    """
    with torch.no_grad():
        mu_f32 = _to_float32(mu)
        sigma_f32 = _to_float32(sigma)
        phi_f32 = _to_float32(phi)
        gen_f32 = _to_float32(generators)

        B, N, K = mu_f32.shape
        n_gen = gen_f32.shape[0]

        orbit_samples: List[Dict[str, Tensor]] = []

        for _ in range(n_samples):
            # Random global Lie algebra coefficients: (B, n_gen)
            eps = torch.randn(B, n_gen, device=mu_f32.device, dtype=mu_f32.dtype)
            eps = eps * perturbation_scale

            # Build K×K matrix: (B, K, K)
            eps_matrix = _build_phi_matrix(eps, gen_f32)  # (B, K, K)
            g, _ = stable_matrix_exp_pair(eps_matrix, only_forward=True)  # (B, K, K)

            # Apply to all tokens
            mu_new = torch.einsum("bkl,bnl->bnk", g, mu_f32)      # (B, N, K)
            g_sq = g * g
            sigma_new = torch.einsum("bkl,bnl->bnk", g_sq, sigma_f32)  # (B, N, K)
            phi_new = phi_f32 + eps.unsqueeze(1)                    # (B, N, n_gen)

            orbit_samples.append({
                "mu": mu_new.to(mu.dtype),
                "sigma": sigma_new.to(sigma.dtype),
                "phi": phi_new.to(phi.dtype),
                "g": g.to(mu.dtype),
            })

        return orbit_samples


# ---------------------------------------------------------------------------
# 10. Gauge orbit dimension
# ---------------------------------------------------------------------------

def compute_gauge_orbit_dimension(
    phi: Tensor,
    generators: Tensor,
    eps: float = 1e-6,
) -> int:
    r"""Effective dimension of the gauge orbit at the current point.

    The gauge action of $\mathrm{GL}^+(K)$ on belief space sends
    $(\mu, \phi) \mapsto (g\,\mu, \phi + \epsilon)$ for $g = e^{\epsilon G}$.
    Its Jacobian with respect to the Lie algebra parameters $\epsilon$ has
    rank equal to the number of independent gauge directions actually used at
    the current configuration.

    For $\mathrm{GL}^+(K)$ with $K^2$ generators the theoretical maximum is
    $K^2$; the effective rank may be lower if the phi coefficients lie in a
    lower-dimensional sub-algebra.

    Numerically, this is approximated by constructing the matrix

    .. math::

        J_{ia} = \frac{\partial}{\partial \epsilon^a}
                 \bigl[\exp(\epsilon \cdot G)\,\mu_i\bigr]_{\epsilon=0}
               = (T_a \,\mu_i),

    collecting all token directions into a $(N \cdot K) \times n\_gen$ matrix
    and counting the rank (singular values above ``eps``).

    Args:
        phi: ``(B, N, n_gen)`` gauge-field coefficients. Only the first
            batch element is used.
        generators: ``(n_gen, K, K)`` Lie algebra generators.
        eps: Singular-value threshold for rank counting. Default ``1e-6``.

    Returns:
        Effective gauge orbit dimension (integer in $[0, n\_gen]$).
    """
    with torch.no_grad():
        phi_f32 = _to_float32(phi)
        gen_f32 = _to_float32(generators)

        # Use first batch element only; shape (N, n_gen)
        phi_0 = phi_f32[0]  # (N, n_gen)
        B, N, n_gen = phi_f32.shape
        K = gen_f32.shape[-1]

        # Evaluate exp(phi_0 · G) per token: (N, K, K)
        phi_matrix = _build_phi_matrix(phi_0, gen_f32)  # (N, K, K)
        exp_phi, _ = stable_matrix_exp_pair(phi_matrix, only_forward=True)  # (N, K, K)

        # Jacobian columns: for each generator T_a, the gauge-action direction on all
        # tokens is  [T_a @ exp(phi_i) @ mu_i]  — but at epsilon=0 the group element
        # is exp(phi_i · G) already, so the infinitesimal action of a perturbation
        # delta_epsilon^a is:
        #   d/d(delta_epsilon^a) exp((phi + delta_eps*e_a)·G)|_{delta_eps=0}
        # = exp(phi·G) @ T_a  (right-invariant vector field)
        #
        # The column of J corresponding to generator a is:
        #   J[:, a] = flatten over tokens of (exp(phi_i) @ T_a)  — (N*K, 1) per a
        #
        # But a more direct observable is simply whether the generators span
        # independent directions at the current phi.  We use:
        #   J[i*K:(i+1)*K, a] = (exp(phi_i) @ T_a)[:, 0]  ... actually the full matrix
        # which flattens to (N*K, n_gen).

        # gen_f32: (n_gen, K, K)
        # exp_phi: (N, K, K)
        # action[i, a] = exp_phi[i] @ gen_f32[a]  →  shape (N, n_gen, K, K)
        action = torch.einsum("nij,ajk->naik", exp_phi, gen_f32)  # (N, n_gen, K, K)

        # Flatten to (N*K*K, n_gen) — then SVD to find rank
        J = action.permute(0, 2, 3, 1).reshape(N * K * K, n_gen)  # (N*K*K, n_gen)
        J = J.float()

        # SVD: singular values in descending order
        try:
            sv = torch.linalg.svdvals(J)  # (min(N*K*K, n_gen),)
        except RuntimeError:
            # Fallback: QR-based rank estimate
            sv = torch.linalg.svdvals(J.t() @ J).sqrt()

        rank = int((sv > eps * sv[0].clamp(min=1e-12)).sum().item())
        return rank
