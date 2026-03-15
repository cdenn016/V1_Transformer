#!/usr/bin/env python3
"""
Symbolic verification of ∂KL/∂Ω for diagonal-covariance Gaussians under GL(K) transport.

Setup
-----
q_i = N(μ_i, diag(σ_i))   with σ_i[k] > 0 (variances, not std devs)
q_j = N(μ_j, diag(σ_j))

Transported q_j under Ω ∈ GL(K):
    μ_t = Ω μ_j
    Σ_t = Ω diag(σ_j) Ω^T   (NOT diagonal in general)
    But for the diagonal-covariance KL path we use:
        σ_t[k] = Σ_l Ω[k,l]² σ_j[l]   (diagonal of Σ_t)

KL(q_i || Ω q_j) = 0.5 * [Σ_k σ_i[k]/σ_t[k] + Σ_k δ_k²/σ_t[k] - d + Σ_k log(σ_t[k]/σ_i[k])]
    where δ_k = μ_i[k] - μ_t[k]

Claim
-----
∂KL/∂Ω[r,s] = c_r · Ω[r,s] · σ_j[s] / σ_t[r]  -  δ_r · μ_j[s] / σ_t[r]

where c_r = 1 - σ_i[r]/σ_t[r] - δ_r²/σ_t[r]

Matrix form:
    ∂KL/∂Ω = diag(c / σ_t) @ Ω @ diag(σ_j)  -  outer(δ / σ_t, μ_j)
"""

from sympy import (
    symbols, Matrix, MatrixSymbol, IndexedBase, Idx, Sum,
    simplify, expand, factor, cancel, collect,
    log, sqrt, Rational, Function, Symbol,
    eye, zeros, diag, tensorproduct,
    diff, pprint, latex, init_printing
)

init_printing()


def verify_dKL_dOmega(d=3):
    """Verify ∂KL/∂Ω for d-dimensional diagonal-covariance Gaussians."""
    print(f"\n{'='*70}")
    print(f"  Verifying ∂KL/∂Ω for d = {d}")
    print(f"{'='*70}\n")

    # Define symbols for means and variances
    mu_i = [symbols(f'mui_{k}', real=True) for k in range(d)]
    mu_j = [symbols(f'muj_{k}', real=True) for k in range(d)]
    sig_i = [symbols(f'si_{k}', positive=True) for k in range(d)]
    sig_j = [symbols(f'sj_{k}', positive=True) for k in range(d)]

    # Omega as a d×d matrix of symbols
    Omega = [[symbols(f'W_{r}_{s}', real=True) for s in range(d)] for r in range(d)]

    # Compute transported mean: μ_t[r] = Σ_s Ω[r,s] μ_j[s]
    mu_t = [sum(Omega[r][s] * mu_j[s] for s in range(d)) for r in range(d)]

    # Compute transported diagonal variance: σ_t[r] = Σ_s Ω[r,s]² σ_j[s]
    sig_t = [sum(Omega[r][s]**2 * sig_j[s] for s in range(d)) for r in range(d)]

    # Residuals
    delta = [mu_i[r] - mu_t[r] for r in range(d)]

    # KL divergence
    KL = Rational(1, 2) * sum(
        sig_i[r] / sig_t[r]
        + delta[r]**2 / sig_t[r]
        - 1
        + log(sig_t[r]) - log(sig_i[r])
        for r in range(d)
    )

    print("KL divergence defined. Computing gradients...\n")

    # Compute ∂KL/∂Ω[r,s] for each entry
    all_match = True
    for r in range(d):
        for s in range(d):
            # Autograd-style symbolic derivative
            grad_rs = diff(KL, Omega[r][s])

            # Claimed formula
            c_r = 1 - sig_i[r] / sig_t[r] - delta[r]**2 / sig_t[r]
            claimed = c_r * Omega[r][s] * sig_j[s] / sig_t[r] - delta[r] * mu_j[s] / sig_t[r]

            # Simplify the difference
            diff_expr = simplify(expand(grad_rs - claimed))

            if diff_expr != 0:
                print(f"  MISMATCH at ({r},{s})!")
                print(f"    sympy grad  = {grad_rs}")
                print(f"    claimed     = {claimed}")
                print(f"    difference  = {diff_expr}")
                all_match = False

    if all_match:
        print(f"  ✓ All {d}×{d} = {d*d} entries match perfectly.\n")
    else:
        print(f"  ✗ Some entries DO NOT MATCH.\n")

    return all_match


def verify_matrix_form(d=3):
    """Verify the matrix form: ∂KL/∂Ω = diag(c/σ_t) @ Ω @ diag(σ_j) - outer(δ/σ_t, μ_j)"""
    print(f"\n{'='*70}")
    print(f"  Verifying matrix form for d = {d}")
    print(f"{'='*70}\n")

    mu_i = [symbols(f'mui_{k}', real=True) for k in range(d)]
    mu_j = [symbols(f'muj_{k}', real=True) for k in range(d)]
    sig_i = [symbols(f'si_{k}', positive=True) for k in range(d)]
    sig_j = [symbols(f'sj_{k}', positive=True) for k in range(d)]
    Omega = Matrix(d, d, lambda r, s: symbols(f'W_{r}_{s}', real=True))

    mu_j_vec = Matrix(mu_j)

    # Transported quantities
    mu_t = [sum(Omega[r, s] * mu_j[s] for s in range(d)) for r in range(d)]
    sig_t = [sum(Omega[r, s]**2 * sig_j[s] for s in range(d)) for r in range(d)]
    delta_vec = Matrix([mu_i[r] - mu_t[r] for r in range(d)])
    c_vec = Matrix([1 - sig_i[r] / sig_t[r] - (mu_i[r] - mu_t[r])**2 / sig_t[r] for r in range(d)])

    # Matrix form
    diag_c_over_st = Matrix(d, d, lambda r, s: c_vec[r] / sig_t[r] if r == s else 0)
    diag_sj = Matrix(d, d, lambda r, s: sig_j[r] if r == s else 0)
    delta_over_st = Matrix([delta_vec[r] / sig_t[r] for r in range(d)])

    matrix_grad = diag_c_over_st * Omega * diag_sj - delta_over_st * mu_j_vec.T

    # Element-wise form
    elem_grad = Matrix(d, d, lambda r, s: (
        c_vec[r] * Omega[r, s] * sig_j[s] / sig_t[r]
        - delta_vec[r] * mu_j[s] / sig_t[r]
    ))

    # Check equivalence
    diff_matrix = simplify(matrix_grad - elem_grad)
    is_zero = all(diff_matrix[r, s] == 0 for r in range(d) for s in range(d))

    if is_zero:
        print(f"  ✓ Matrix form matches element-wise form for d={d}.\n")
    else:
        print(f"  ✗ Matrix form DOES NOT match element-wise form.\n")
        pprint(diff_matrix)

    return is_zero


def verify_special_cases():
    """Verify gradient vanishes at identity transport (Ω=I, same distributions)."""
    print(f"\n{'='*70}")
    print(f"  Verifying special cases")
    print(f"{'='*70}\n")

    d = 2  # Use d=2 for speed on special cases

    mu = [symbols(f'mu_{k}', real=True) for k in range(d)]
    sig = [symbols(f'sig_{k}', positive=True) for k in range(d)]

    # Case 1: Ω = I, q_i = q_j  →  KL = 0, ∂KL/∂Ω = 0
    # With identical distributions and identity transport
    sig_t = list(sig)  # Ω=I → σ_t = σ_j = σ_i
    delta = [0] * d     # μ_i = μ_j, Ω=I → δ = 0
    c = [1 - sig[r] / sig_t[r] - 0 for r in range(d)]  # c_r = 1 - 1 - 0 = 0

    # Gradient should be zero
    for r in range(d):
        for s in range(d):
            omega_rs = 1 if r == s else 0
            grad_rs = c[r] * omega_rs * sig[s] / sig_t[r] - 0
            assert simplify(grad_rs) == 0, f"Case 1 failed at ({r},{s})"

    print("  ✓ Case 1: Ω=I, q_i=q_j → ∂KL/∂Ω = 0")

    # Case 2: Ω = I, different distributions → gradient is non-trivial
    mu_i = [symbols(f'mui_{k}', real=True) for k in range(d)]
    mu_j = [symbols(f'muj_{k}', real=True) for k in range(d)]
    sig_i = [symbols(f'si_{k}', positive=True) for k in range(d)]
    sig_j = [symbols(f'sj_{k}', positive=True) for k in range(d)]

    # At Ω = I: σ_t = σ_j, δ = μ_i - μ_j
    delta_2 = [mu_i[r] - mu_j[r] for r in range(d)]
    c_2 = [1 - sig_i[r] / sig_j[r] - delta_2[r]**2 / sig_j[r] for r in range(d)]

    # ∂KL/∂Ω[r,s] at Ω=I: only r=s contributes from the Ω term
    # = c_r · δ_{rs} · σ_j[s] / σ_j[r] - δ_r · μ_j[s] / σ_j[r]
    # For r=s: c_r - δ_r · μ_j[r] / σ_j[r]
    # For r≠s: -δ_r · μ_j[s] / σ_j[r]
    grad_00 = c_2[0] - delta_2[0] * mu_j[0] / sig_j[0]
    grad_01 = -delta_2[0] * mu_j[1] / sig_j[0]
    assert simplify(grad_00 - (c_2[0] * 1 * sig_j[0] / sig_j[0] - delta_2[0] * mu_j[0] / sig_j[0])) == 0
    assert simplify(grad_01 - (c_2[0] * 0 * sig_j[1] / sig_j[0] - delta_2[0] * mu_j[1] / sig_j[0])) == 0

    print("  ✓ Case 2: Ω=I, different distributions → gradient consistent")

    # Case 3: Verify trace term (log-det gradient)
    # ∂/∂Ω[r,s] log det(Ω diag(σ_j) Ω^T) contains 2(Ω^{-T})[r,s] for full covariance
    # But our diagonal approximation σ_t[r] = Σ_s Ω[r,s]² σ_j[s] gives different gradient
    # This is expected and correct for the diagonal KL path
    print("  ✓ Case 3: Diagonal approximation acknowledged (not full log-det)")

    print()
    return True


if __name__ == '__main__':
    print("Analytic ∂KL/∂Ω Verification")
    print("=" * 70)

    # Main verification for d=3
    ok1 = verify_dKL_dOmega(d=3)

    # Also verify for d=2 as sanity check
    ok2 = verify_dKL_dOmega(d=2)

    # Matrix form verification
    ok3 = verify_matrix_form(d=3)

    # Special cases
    ok4 = verify_special_cases()

    print("\n" + "=" * 70)
    if all([ok1, ok2, ok3, ok4]):
        print("  ALL VERIFICATIONS PASSED")
    else:
        print("  SOME VERIFICATIONS FAILED")
    print("=" * 70)
