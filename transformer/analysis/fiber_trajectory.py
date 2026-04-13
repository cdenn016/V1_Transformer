"""
Intra-Fiber VFE Trajectory Analysis
====================================

Fisher-Rao geometry on the Gaussian belief manifold for analyzing
VFE E-step iteration trajectories. The fiber at each token position
is the space of diagonal Gaussians G_K = {N(mu, diag(sigma^2))},
equipped with the Fisher-Rao metric:

    ds^2 = sum_k [dmu_k^2 / sigma_k^2 + (d(sigma_k^2))^2 / (2 sigma_k^4)]

All functions operate on numpy arrays for offline analysis.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List

from transformer.analysis.trajectory import IterationSnapshot


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_std(sigma_diag: np.ndarray) -> np.ndarray:
    r"""Convert diagonal variance array to standard deviations.

    Args:
        sigma_diag: (..., K) array of per-dimension variances
            :math:`\sigma_k^2`.

    Returns:
        (..., K) array of standard deviations :math:`\sigma_k \ge 0`.
    """
    return np.sqrt(np.maximum(sigma_diag, 0.0))


def _extract_token(arr: np.ndarray, token_idx: int) -> np.ndarray:
    r"""Return the row corresponding to *token_idx* from an (N, K) array.

    Args:
        arr: (N, K) snapshot array.
        token_idx: Row index into the N_recorded token dimension.

    Returns:
        (K,) array for the requested token.

    Raises:
        IndexError: If *token_idx* is out of range for *arr*.
    """
    return arr[token_idx]  # (K,)


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def fisher_rao_infinitesimal(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
) -> np.ndarray:
    r"""Per-token Fisher-Rao infinitesimal distance between two adjacent states.

    Evaluates the diagonal Gaussian Fisher-Rao metric at the midpoint of the
    two states (trapezoidal rule) and returns

    .. math::

        ds_i = \sqrt{
            \sum_k \left[
                \frac{(\Delta\mu_k)^2}{\bar\sigma_k^2}
                + \frac{(\Delta(\sigma_k^2))^2}{2\,\bar\sigma_k^4}
            \right]
        }

    where :math:`\bar\sigma_k` is the midpoint standard deviation
    :math:`\tfrac{1}{2}(\sigma_{1,k}+\sigma_{2,k})` and the inputs carry
    variances :math:`\sigma_k^2`.  The two metric terms are equivalent to
    :math:`d\mu_k^2/\bar\sigma_k^2 + 2\,d\sigma_k^2/\bar\sigma_k^2` when
    expressed in terms of standard deviations.

    Args:
        mu1: (N, K) means at state 1.
        sigma1: (N, K) diagonal variances at state 1.
        mu2: (N, K) means at state 2.
        sigma2: (N, K) diagonal variances at state 2.

    Returns:
        (N,) Fisher-Rao infinitesimal distances, one per token.
    """
    # Convert variances to std devs; midpoint for metric evaluation
    std1 = _to_std(sigma1)   # (N, K)
    std2 = _to_std(sigma2)   # (N, K)
    std_mid = 0.5 * (std1 + std2)  # (N, K)
    std_mid = np.maximum(std_mid, 1e-12)  # numerical guard

    # Δμ term:  (Δμ_k)² / σ̄_k²
    d_mu = mu2 - mu1  # (N, K)
    term_mu = (d_mu ** 2) / (std_mid ** 2)  # (N, K)

    # Δ(σ²) term:  (Δσ²_k)² / (2 σ̄_k⁴)
    d_var = sigma2 - sigma1  # (N, K), differences of variances
    term_var = (d_var ** 2) / (2.0 * std_mid ** 4)  # (N, K)

    ds_per_dim = term_mu + term_var  # (N, K)
    ds = np.sqrt(np.sum(ds_per_dim, axis=-1))  # (N,)
    return ds


def fisher_rao_distance(
    mu1: np.ndarray,
    sig1: np.ndarray,
    mu2: np.ndarray,
    sig2: np.ndarray,
) -> np.ndarray:
    r"""Per-token closed-form Fisher-Rao geodesic distance on the diagonal
    Gaussian manifold.

    Each dimension :math:`k` is treated as an independent 1-D Gaussian
    equipped with the Fisher-Rao metric on the Poincaré upper half-plane
    :math:`\mathcal{H} = \{(\mu, \sigma) : \sigma > 0\}` with metric
    :math:`ds^2 = d\mu^2/\sigma^2 + 2\,d\sigma^2/\sigma^2`.

    The closed-form geodesic distance for one dimension is derived from
    the Poincaré half-plane isometry :math:`(\\mu, \\sigma) \\mapsto
    (\\mu,\\, \\sigma\\sqrt{2})`, which maps the Fisher-Rao metric to
    :math:`2(du^2 + dv^2)/v^2`:

    .. math::

        d_k = \\sqrt{2}\\,\\operatorname{arccosh}\\!\\left(
            1 + \\frac{(\\Delta\\mu_k)^2 + 2\\,(\\Delta\\sigma_k)^2}
                     {4\\,\\sigma_{1,k}\\,\\sigma_{2,k}}
        \\right)

    where :math:`\\sigma_k = \\sqrt{\\text{variance}_k}` and
    :math:`\\Delta\\sigma_k = \\sigma_{2,k} - \\sigma_{1,k}`.  The total K-
    dimensional distance combines dimensions in quadrature:

    .. math::

        d_{\text{total}} = \sqrt{\sum_k d_k^2}

    Reference: Atkinson, C. & Mitchell, A. F. S. (1981). Rao's distance
    measure. *Sankhya*, 43(A), 345–365.

    Args:
        mu1: (N, K) means at state 1.
        sig1: (N, K) diagonal variances at state 1.
        mu2: (N, K) means at state 2.
        sig2: (N, K) diagonal variances at state 2.

    Returns:
        (N,) total Fisher-Rao geodesic distances per token.
    """
    std1 = _to_std(sig1)   # (N, K)
    std2 = _to_std(sig2)   # (N, K)

    d_mu = mu2 - mu1           # (N, K)
    d_std = std2 - std1        # (N, K)

    denom = 2.0 * std1 * std2  # (N, K)
    denom = np.maximum(denom, 1e-24)  # guard against zero std

    acosh_arg = 1.0 + (d_mu ** 2 + 2.0 * d_std ** 2) / (2.0 * denom)  # (N, K)
    # Must be ≥ 1 for arccosh; numerical noise can push it just below
    acosh_arg = np.maximum(acosh_arg, 1.0)

    d_k = np.sqrt(2.0) * np.arccosh(acosh_arg)  # (N, K)
    d_total = np.sqrt(np.sum(d_k ** 2, axis=-1))  # (N,)
    return d_total


# ---------------------------------------------------------------------------
# Trajectory-level functions
# ---------------------------------------------------------------------------

def compute_arc_length(
    snapshots: List[IterationSnapshot],
    token_idx: int,
) -> np.ndarray:
    r"""Cumulative Fisher-Rao arc length along the VFE iteration trajectory.

    Computes

    .. math::

        L(t) = \sum_{s=0}^{t-1} d_{\text{FR}}\!\left(
            q_{s},\, q_{s+1}
        \right), \quad L(0) = 0

    where each :math:`q_s` is the Gaussian belief state at iteration :math:`s`
    and :math:`d_{\text{FR}}` is the closed-form Fisher-Rao geodesic distance.

    Args:
        snapshots: Ordered list of :class:`IterationSnapshot` objects
            representing consecutive VFE iterations.
        token_idx: Index into the N_recorded axis of each snapshot.

    Returns:
        (T,) cumulative arc lengths; first entry is 0.0.  Returns
        ``np.array([0.0])`` for a single-snapshot trajectory.
    """
    T = len(snapshots)
    cumulative = np.zeros(T)

    for t in range(1, T):
        s_prev = snapshots[t - 1]
        s_curr = snapshots[t]

        mu1 = _extract_token(s_prev.mu, token_idx)[np.newaxis, :]        # (1, K)
        var1 = _extract_token(s_prev.sigma_diag, token_idx)[np.newaxis, :]
        mu2 = _extract_token(s_curr.mu, token_idx)[np.newaxis, :]
        var2 = _extract_token(s_curr.sigma_diag, token_idx)[np.newaxis, :]

        step_dist = fisher_rao_distance(mu1, var1, mu2, var2)[0]  # scalar
        cumulative[t] = cumulative[t - 1] + step_dist

    return cumulative


def compute_velocity_profile(
    snapshots: List[IterationSnapshot],
    token_idx: int,
) -> np.ndarray:
    r"""Fisher-Rao speed :math:`\|ds/dt\|` at each VFE iteration step.

    Returns the per-step Fisher-Rao distances, i.e., the discrete derivative
    of the arc-length curve.

    Args:
        snapshots: Ordered list of :class:`IterationSnapshot` objects.
        token_idx: Index into the N_recorded axis.

    Returns:
        (T-1,) speed values.  Returns an empty array for a single-snapshot
        trajectory.
    """
    T = len(snapshots)
    if T < 2:
        return np.array([], dtype=np.float64)

    velocities = np.empty(T - 1)
    for t in range(T - 1):
        s0 = snapshots[t]
        s1 = snapshots[t + 1]

        mu1 = _extract_token(s0.mu, token_idx)[np.newaxis, :]
        var1 = _extract_token(s0.sigma_diag, token_idx)[np.newaxis, :]
        mu2 = _extract_token(s1.mu, token_idx)[np.newaxis, :]
        var2 = _extract_token(s1.sigma_diag, token_idx)[np.newaxis, :]

        velocities[t] = fisher_rao_distance(mu1, var1, mu2, var2)[0]

    return velocities


def compute_convergence_curve(
    snapshots: List[IterationSnapshot],
    token_idx: int,
) -> np.ndarray:
    r"""Fisher-Rao distance from each iteration state to the final state.

    Measures how far the belief trajectory is from convergence at each step:

    .. math::

        c(t) = d_{\text{FR}}\!\left(q_t,\, q_T\right)

    where :math:`q_T` is the belief at the last snapshot.  A well-converged
    trajectory should have :math:`c(t)` decreasing monotonically to zero.

    Args:
        snapshots: Ordered list of :class:`IterationSnapshot` objects.
        token_idx: Index into the N_recorded axis.

    Returns:
        (T,) distances from each snapshot to the final snapshot.  For a
        single-snapshot trajectory, returns ``np.array([0.0])``.
    """
    T = len(snapshots)
    if T == 1:
        return np.array([0.0])

    final = snapshots[-1]
    mu_f = _extract_token(final.mu, token_idx)[np.newaxis, :]
    var_f = _extract_token(final.sigma_diag, token_idx)[np.newaxis, :]

    curve = np.empty(T)
    for t in range(T):
        s = snapshots[t]
        mu_t = _extract_token(s.mu, token_idx)[np.newaxis, :]
        var_t = _extract_token(s.sigma_diag, token_idx)[np.newaxis, :]
        curve[t] = fisher_rao_distance(mu_t, var_t, mu_f, var_f)[0]

    return curve


def compute_geodesic_deviation(
    snapshots: List[IterationSnapshot],
    token_idx: int,
) -> float:
    r"""Ratio of trajectory arc length to geodesic distance.

    The geodesic deviation ratio

    .. math::

        \rho = \frac{L(T)}{d_{\text{FR}}(q_0,\, q_T)} \ge 1

    equals 1.0 when the VFE iteration follows the Fisher-Rao geodesic exactly
    and exceeds 1.0 whenever the path curves through the belief manifold.  A
    large :math:`\rho` indicates oscillation or over-shooting during
    convergence.

    Args:
        snapshots: Ordered list of :class:`IterationSnapshot` objects.
        token_idx: Index into the N_recorded axis.

    Returns:
        Deviation ratio :math:`\rho \ge 1.0`.  Returns 1.0 when both arc
        length and geodesic distance are effectively zero (trivial trajectory).
    """
    if len(snapshots) < 2:
        return 1.0

    arc = compute_arc_length(snapshots, token_idx)[-1]  # total arc length

    first = snapshots[0]
    last = snapshots[-1]
    mu0 = _extract_token(first.mu, token_idx)[np.newaxis, :]
    var0 = _extract_token(first.sigma_diag, token_idx)[np.newaxis, :]
    mu_f = _extract_token(last.mu, token_idx)[np.newaxis, :]
    var_f = _extract_token(last.sigma_diag, token_idx)[np.newaxis, :]

    geo = fisher_rao_distance(mu0, var0, mu_f, var_f)[0]

    if geo < 1e-12:
        # Both numerator and denominator are effectively zero
        return 1.0

    return float(arc / geo)


def fit_convergence_rate(convergence_curve: np.ndarray) -> float:
    r"""Fit exponential decay :math:`d(t) = A e^{-\lambda t}` to the
    convergence curve.

    Performs an ordinary least-squares fit in log space:

    .. math::

        \log d(t) \approx \log A - \lambda\,t

    The decay rate :math:`\lambda > 0` describes how rapidly the trajectory
    approaches its fixed point in Fisher-Rao distance.

    Edge-case handling:

    * Values :math:`\le 0` are masked out before fitting (they correspond to
      exact convergence or numerical noise).
    * If fewer than two positive values remain, returns 0.0.
    * If the curve is oscillatory (more than 50% sign changes in the
      log-space first differences), the exponential decay model is a poor
      fit — returns ``float('nan')`` so callers can detect this case.
    * A non-positive fitted slope (non-decaying curve) is clamped to 0.0.

    Args:
        convergence_curve: (T,) array of distances to the final state, as
            returned by :func:`compute_convergence_curve`.

    Returns:
        :math:`\lambda \ge 0` (the exponential decay rate), or ``nan`` if the
        curve is oscillatory and the exponential model does not fit.
    """
    curve = np.asarray(convergence_curve, dtype=np.float64)
    positive_mask = curve > 0.0
    if positive_mask.sum() < 2:
        return 0.0

    t_vals = np.where(positive_mask)[0].astype(np.float64)
    log_d = np.log(curve[positive_mask])

    # Monotonicity check: if log_d has many sign changes in its first
    # differences, the convergence is oscillatory and the exponential
    # decay model is a poor fit.
    diffs = np.diff(log_d)
    if len(diffs) >= 3:
        signs = np.sign(diffs)
        sign_changes = np.sum(signs[:-1] != signs[1:])
        if sign_changes > len(diffs) * 0.5:
            return float('nan')

    # Least-squares: [1, t] @ [log_A, -lambda] = log_d
    A_mat = np.column_stack([np.ones_like(t_vals), t_vals])
    coeffs, _, _, _ = np.linalg.lstsq(A_mat, log_d, rcond=None)
    # coeffs[1] = -lambda; negate and clamp to non-negative
    lam = float(-coeffs[1])
    return max(lam, 0.0)


# ---------------------------------------------------------------------------
# Statistics dataclass
# ---------------------------------------------------------------------------

@dataclass
class FiberTrajectoryStats:
    r"""Summary statistics for a single token's VFE iteration trajectory.

    All distances are in Fisher-Rao units on the diagonal Gaussian manifold
    :math:`\mathcal{G}_K`.

    Attributes:
        token_idx: Index into the N_recorded axis of the snapshot arrays.
        arc_length: Total Fisher-Rao arc length :math:`L(T)` along the
            trajectory.
        geodesic_distance: Straight-line Fisher-Rao distance between the
            initial and final belief states.
        deviation_ratio: :math:`\rho = L(T) / d_{\text{FR}}(q_0, q_T) \ge 1`.
        convergence_rate: Exponential decay rate :math:`\lambda` from
            :func:`fit_convergence_rate`.
        mean_velocity: Mean per-step Fisher-Rao speed (arc_length / (T-1)).
        final_velocity: Fisher-Rao speed at the last step.
        mu_displacement: Euclidean :math:`\ell_2` displacement of the mean:
            :math:`\|\mu_T - \mu_0\|_2`.
        sigma_ratio: Geometric mean of per-dimension std-dev ratios
            :math:`\sigma_{T,k} / \sigma_{0,k}`, a compact measure of the
            net covariance change.
    """
    token_idx: int
    arc_length: float
    geodesic_distance: float
    deviation_ratio: float
    convergence_rate: float
    mean_velocity: float
    final_velocity: float
    mu_displacement: float
    sigma_ratio: float


# ---------------------------------------------------------------------------
# High-level analysis entry points
# ---------------------------------------------------------------------------

def analyze_fiber_trajectory(
    snapshots: List[IterationSnapshot],
    token_idx: int,
) -> FiberTrajectoryStats:
    r"""Compute all Fisher-Rao trajectory statistics for one token.

    Assembles :class:`FiberTrajectoryStats` by calling the individual
    analysis functions.  Handles degenerate cases (single snapshot, zero
    displacement) gracefully.

    Args:
        snapshots: Ordered list of :class:`IterationSnapshot` objects from
            consecutive VFE E-step iterations.
        token_idx: Index into the N_recorded token dimension.

    Returns:
        :class:`FiberTrajectoryStats` populated for *token_idx*.
    """
    T = len(snapshots)

    # Arc length and velocity
    arc_lengths = compute_arc_length(snapshots, token_idx)  # (T,)
    total_arc = float(arc_lengths[-1])

    velocity_profile = compute_velocity_profile(snapshots, token_idx)  # (T-1,)
    if len(velocity_profile) > 0:
        mean_vel = float(np.mean(velocity_profile))
        final_vel = float(velocity_profile[-1])
    else:
        mean_vel = 0.0
        final_vel = 0.0

    # Geodesic distance: first → last
    first = snapshots[0]
    last = snapshots[-1]
    mu0 = _extract_token(first.mu, token_idx)[np.newaxis, :]
    var0 = _extract_token(first.sigma_diag, token_idx)[np.newaxis, :]
    mu_f = _extract_token(last.mu, token_idx)[np.newaxis, :]
    var_f = _extract_token(last.sigma_diag, token_idx)[np.newaxis, :]

    geo_dist = float(fisher_rao_distance(mu0, var0, mu_f, var_f)[0])

    # Deviation ratio
    if geo_dist < 1e-12:
        dev_ratio = 1.0
    else:
        dev_ratio = total_arc / geo_dist

    # Convergence curve and exponential decay rate
    conv_curve = compute_convergence_curve(snapshots, token_idx)  # (T,)
    conv_rate = fit_convergence_rate(conv_curve)

    # Euclidean mean displacement
    mu_disp = float(np.linalg.norm(mu_f[0] - mu0[0]))

    # Geometric mean std-dev ratio  σ_T / σ_0
    std0 = _to_std(var0[0])   # (K,)
    std_f = _to_std(var_f[0])  # (K,)
    K = std0.shape[0]
    # Mask dimensions where initial std is effectively zero
    valid = std0 > 1e-12
    if valid.any():
        log_ratios = np.log(np.maximum(std_f[valid], 1e-12)) - np.log(std0[valid])
        geom_ratio = float(np.exp(np.mean(log_ratios)))
    else:
        geom_ratio = 1.0

    return FiberTrajectoryStats(
        token_idx=token_idx,
        arc_length=total_arc,
        geodesic_distance=geo_dist,
        deviation_ratio=dev_ratio,
        convergence_rate=conv_rate,
        mean_velocity=mean_vel,
        final_velocity=final_vel,
        mu_displacement=mu_disp,
        sigma_ratio=geom_ratio,
    )


def analyze_all_tokens(
    snapshots: List[IterationSnapshot],
    n_tokens: int,
) -> List[FiberTrajectoryStats]:
    r"""Compute Fisher-Rao trajectory statistics for every recorded token.

    Calls :func:`analyze_fiber_trajectory` for token indices
    :math:`0, 1, \ldots, n\_tokens - 1`.  All snapshots must have at least
    *n_tokens* rows in their ``mu`` / ``sigma_diag`` arrays.

    Args:
        snapshots: Ordered list of :class:`IterationSnapshot` objects.
        n_tokens: Number of recorded tokens (N_recorded dimension).

    Returns:
        List of :class:`FiberTrajectoryStats`, one per token, in order of
        token index.
    """
    return [analyze_fiber_trajectory(snapshots, i) for i in range(n_tokens)]
