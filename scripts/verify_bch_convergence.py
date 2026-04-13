"""
BCH Convergence Test for soN_compose_bch_torch.

Verifies:
  1. Degree-4 fix (-(1/24) coefficient) gives O(eps^5) error at order=3.
  2. Whether degree-5 term (order >= 4) gives O(eps^6) or is incomplete.

Click-to-run: edit config dict below, then run.

Mathematical reference
----------------------
Standard BCH (Dynkin form), Z = log(exp(X) exp(Y)):

  Degree 1:  X + Y
  Degree 2:  (1/2)[X,Y]
  Degree 3:  (1/12)[X,[X,Y]] - (1/12)[Y,[X,Y]]
  Degree 4:  -(1/24)[Y,[X,[X,Y]]]
  Degree 5:  -(1/720)([Y,[Y,[Y,[Y,X]]]] + [X,[X,[X,[X,Y]]]])
             +(1/360)([X,[Y,[Y,[Y,X]]]] + [Y,[X,[X,[X,Y]]]])
             +(1/120)([Y,[X,[Y,[X,Y]]]] + [X,[Y,[X,[Y,X]]]])

The code's order >= 4 uses only -(1/24)[X,[Y,[Y,[X,Y]]]].
The convergence rate will reveal whether this single term captures
the full degree-5 contribution for SO(3).

SO(3) matrix log
----------------
For R in SO(3), the Rodrigues inverse formula:

  theta = arccos((tr(R) - 1) / 2)
  log(R) = (theta / (2 sin(theta))) * (R - R^T)    for theta != 0
  log(R) = 0                                         for theta = 0

This gives a 3x3 skew-symmetric matrix (an element of so(3)).

Coordinate projection
---------------------
Canonical so(3) generators L_{ij} (i < j) with (L_{ij})_{ab} = delta_{ia}delta_{jb} - delta_{ib}delta_{ja}.
Ordering: a=0 -> (0,1), a=1 -> (0,2), a=2 -> (1,2).
Coordinates: phi[a] = Z_mat[i, j]  (upper-triangular element).
"""

import sys
import os
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from math_utils.generators import generate_so3_generators, soN_compose_bch_torch

# ---------------------------------------------------------------------------
# Config (click-to-run)
# ---------------------------------------------------------------------------
CFG = {
    "seed": 42,
    "epsilons": [0.005, 0.01, 0.02, 0.05, 0.1, 0.2],
    "orders": [1, 2, 3, 4],
}

torch.set_default_dtype(torch.float64)


# ---------------------------------------------------------------------------
# SO(3) matrix log via Rodrigues inverse
# ---------------------------------------------------------------------------

def matrix_log_so3(R: torch.Tensor) -> torch.Tensor:
    r"""Compute log(R) for R in SO(3) via the Rodrigues inverse formula.

    For R in SO(3):
        theta = arccos((tr(R) - 1) / 2)
        log(R) = (theta / (2 sin(theta))) * (R - R^T)

    Special cases:
        theta ~ 0: log(R) ~ 0  (use first-order: (R - R^T)/2)
        theta ~ pi: use eigendecomposition fallback

    Args:
        R: Rotation matrix (3, 3), dtype float64

    Returns:
        log_R: Skew-symmetric matrix (3, 3) in so(3)
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    # Clamp to valid range for arccos
    cos_theta = torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0)
    theta = torch.acos(cos_theta)

    skew = (R - R.T) / 2.0  # = sin(theta) * K  where K is unit skew axis

    sin_theta = torch.sin(theta)
    if sin_theta.abs() < 1e-12:
        # Near identity: log(R) ~ R - I to first order in skew part
        return skew
    else:
        return (theta / sin_theta) * skew


def project_so3_coords(Z_mat: torch.Tensor) -> torch.Tensor:
    r"""Extract so(3) coordinates from a 3x3 skew-symmetric matrix.

    Canonical ordering: a=0 -> (i=0,j=1), a=1 -> (i=0,j=2), a=2 -> (i=1,j=2).
    Coordinate: phi[a] = Z_mat[i, j]  (upper triangle).

    Args:
        Z_mat: Skew-symmetric matrix (3, 3)

    Returns:
        coords: Lie algebra coordinates (3,)
    """
    coords = torch.zeros(3, dtype=Z_mat.dtype, device=Z_mat.device)
    coords[0] = Z_mat[0, 1]   # (0,1)
    coords[1] = Z_mat[0, 2]   # (0,2)
    coords[2] = Z_mat[1, 2]   # (1,2)
    return coords


# ---------------------------------------------------------------------------
# Build so(3) Lie algebra element from coordinate vector
# ---------------------------------------------------------------------------

def coords_to_matrix_so3(phi: torch.Tensor) -> torch.Tensor:
    r"""Build 3x3 skew-symmetric matrix from so(3) coordinate vector.

    Inverse of project_so3_coords.

    Args:
        phi: Coordinates (3,): [phi_01, phi_02, phi_12]

    Returns:
        A: Skew-symmetric (3, 3)
    """
    A = torch.zeros(3, 3, dtype=phi.dtype, device=phi.device)
    A[0, 1] =  phi[0];  A[1, 0] = -phi[0]
    A[0, 2] =  phi[1];  A[2, 0] = -phi[1]
    A[1, 2] =  phi[2];  A[2, 1] = -phi[2]
    return A


# ---------------------------------------------------------------------------
# Compute exact BCH result for phi1, phi2 in so(3)
# ---------------------------------------------------------------------------

def exact_bch_so3(phi1: torch.Tensor, phi2: torch.Tensor) -> torch.Tensor:
    r"""Compute Z = log(exp(phi1) exp(phi2)) exactly in SO(3).

    Steps:
      1. Build 3x3 matrices M1 = phi1[a] * G_a,  M2 = phi2[a] * G_a
      2. R = exp(M1) @ exp(M2)
      3. Z_mat = log(R)  (Rodrigues inverse)
      4. Project Z_mat to coordinates

    Args:
        phi1: so(3) coordinates (3,)
        phi2: so(3) coordinates (3,)

    Returns:
        z_coords: so(3) coordinates of the exact BCH result (3,)
    """
    M1 = coords_to_matrix_so3(phi1)
    M2 = coords_to_matrix_so3(phi2)
    R = torch.linalg.matrix_exp(M1) @ torch.linalg.matrix_exp(M2)
    Z_mat = matrix_log_so3(R)
    return project_so3_coords(Z_mat)


# ---------------------------------------------------------------------------
# Convergence rate helper
# ---------------------------------------------------------------------------

def convergence_rate(errors: list, epsilons: list, i: int) -> float:
    r"""Compute log-log slope between consecutive epsilon pairs.

    rate = log(err[i+1] / err[i]) / log(eps[i+1] / eps[i])

    A rate of k means error ~ eps^k.
    """
    if errors[i] < 1e-300 or errors[i + 1] < 1e-300:
        return float('nan')
    return np.log(errors[i + 1] / errors[i]) / np.log(epsilons[i + 1] / epsilons[i])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_convergence_test() -> None:
    r"""Run BCH convergence test and report rates per order."""

    seed = CFG["seed"]
    epsilons = CFG["epsilons"]
    orders = CFG["orders"]

    # Fixed random directions (unit-ish, not normalized — matches user spec)
    rng = torch.Generator()
    rng.manual_seed(seed)
    dir1 = torch.randn(3, generator=rng, dtype=torch.float64)
    dir2 = torch.randn(3, generator=rng, dtype=torch.float64)

    # Generators: only shape[0]=3 is used by soN_bracket_torch (n_gen -> N=3)
    # We pass the canonical SO(3) generators as (3,3,3) torch tensor
    gens_np = generate_so3_generators(3)       # (3, 3, 3) numpy
    gens = torch.from_numpy(gens_np).double()  # (3, 3, 3)

    # Header
    print("=" * 72)
    print("BCH Convergence Test  |  soN_compose_bch_torch  |  SO(3)")
    print("=" * 72)
    print(f"seed={seed},  dir1 norm={dir1.norm():.4f},  dir2 norm={dir2.norm():.4f}")
    print(f"epsilons: {epsilons}")
    print()

    all_errors = {}  # order -> list of errors

    for order in orders:
        errors = []
        for eps in epsilons:
            phi1 = eps * dir1
            phi2 = eps * dir2

            # Exact BCH result
            z_exact = exact_bch_so3(phi1, phi2)

            # Approximate BCH
            z_approx = soN_compose_bch_torch(phi1, phi2, gens, order=order)

            err = (z_approx - z_exact).norm().item()
            errors.append(err)

        all_errors[order] = errors

        # Print error table for this order
        print(f"Order {order}:")
        print(f"  {'eps':>10}   {'|error|':>14}   {'rate':>8}")
        print(f"  {'-'*10}   {'-'*14}   {'-'*8}")
        for i, (eps, err) in enumerate(zip(epsilons, errors)):
            if i == 0:
                rate_str = "     ---"
            else:
                rate = convergence_rate(errors, epsilons, i - 1)
                rate_str = f"{rate:8.2f}"
            print(f"  {eps:10.4f}   {err:14.3e}   {rate_str}")
        print()

    # Summary table: median rate per order (over the small-epsilon regime)
    print("=" * 72)
    print("Summary: median convergence rate (small-eps pairs)")
    print(f"  {'Order':>5}   {'median rate':>12}   {'expected':>10}   {'diagnosis':>25}")
    print(f"  {'-'*5}   {'-'*12}   {'-'*10}   {'-'*25}")

    expected = {1: 3, 2: 4, 3: 5, 4: 6}
    for order in orders:
        errors = all_errors[order]
        # Use first 4 epsilon pairs (small-eps) to avoid near-machine-eps noise
        n_pairs = min(4, len(epsilons) - 1)
        rates = [convergence_rate(errors, epsilons, i) for i in range(n_pairs)
                 if not np.isnan(convergence_rate(errors, epsilons, i))]
        if rates:
            median_rate = float(np.median(rates))
        else:
            median_rate = float('nan')

        exp = expected.get(order, '?')
        diff = abs(median_rate - exp) if not np.isnan(median_rate) else 99
        if diff < 0.3:
            diagnosis = "CORRECT"
        elif diff < 0.8:
            diagnosis = "MARGINAL (check)"
        else:
            diagnosis = f"WRONG (expected {exp})"

        print(f"  {order:>5}   {median_rate:12.2f}   {str(exp):>10}   {diagnosis:>25}")

    print()

    # Specific diagnoses
    print("=" * 72)
    print("Diagnoses:")
    print()

    # Order 3 (degree-4 fix)
    errors3 = all_errors[3]
    n_pairs = min(4, len(epsilons) - 1)
    rates3 = [convergence_rate(errors3, epsilons, i) for i in range(n_pairs)]
    median3 = float(np.median([r for r in rates3 if not np.isnan(r)]))
    print(f"  Degree-4 term (-(1/24)[Y,[X,[X,Y]]]):  order=3 median rate = {median3:.2f}")
    if abs(median3 - 5.0) < 0.3:
        print("  -> CORRECT: rate ~5 confirms the -(1/24) coefficient is right.")
    elif abs(median3 - 4.0) < 0.3:
        print("  -> WRONG: rate ~4 means degree-4 term is absent or zero.")
    else:
        print(f"  -> UNEXPECTED rate {median3:.2f}: manual inspection needed.")

    print()

    # Order 4 (degree-5 term)
    errors4 = all_errors[4]
    rates4 = [convergence_rate(errors4, epsilons, i) for i in range(n_pairs)]
    median4 = float(np.median([r for r in rates4 if not np.isnan(r)]))
    print(f"  Degree-5 term (-(1/24)[X,[Y,[Y,[X,Y]]]]):  order=4 median rate = {median4:.2f}")
    if abs(median4 - 6.0) < 0.3:
        print("  -> CORRECT: rate ~6 confirms the single degree-5 term is sufficient.")
        print("     (remaining degree-5 terms vanish for SO(3) by Jacobi/structure).")
    elif abs(median4 - 5.0) < 0.3:
        print("  -> INCOMPLETE: rate ~5 means the degree-5 contribution was not zeroed.")
        print("     The code's comment that other degree-5 terms 'vanish by Jacobi' is WRONG.")
        print("     Need to add the remaining Dynkin degree-5 terms:")
        print("       -(1/720)([Y,[Y,[Y,[Y,X]]]] + [X,[X,[X,[X,Y]]]])")
        print("       +(1/360)([X,[Y,[Y,[Y,X]]]] + [Y,[X,[X,[X,Y]]]])")
        print("       +(1/120)([Y,[X,[Y,[X,Y]]]] + [X,[Y,[X,[Y,X]]]])")
    else:
        print(f"  -> UNEXPECTED rate {median4:.2f}: manual inspection needed.")

    print()

    # Cross-check: compare order=3 and order=4 errors at smallest eps
    eps_small = epsilons[0]
    err3_small = errors3[0]
    err4_small = errors4[0]
    improvement = err3_small / err4_small if err4_small > 1e-300 else float('inf')
    print(f"  Error ratio  order=3 / order=4  at eps={eps_small}:  {improvement:.2f}x")
    print(f"  (expect ~{eps_small:.3f}^1 = {eps_small:.3f} ratio if degree-5 adds one more power)")

    print()
    print("=" * 72)
    print("Done.")


if __name__ == "__main__":
    run_convergence_test()
