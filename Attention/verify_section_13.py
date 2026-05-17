"""Verify Claim 6 / Section 13: M-projection (moment-matching) Gaussian barycenter.

Objective: J(mu, Sigma) = sum_i w_i * KL( N(mu_i_t, Sigma_i_t) || N(mu, Sigma) )

Claim:  argmin gives  mu_B = sum w_i mu_i_t
                     Sigma_B = sum w_i [ Sigma_i_t + (mu_i_t - mu_B)(mu_i_t - mu_B)^T ]
"""

import sympy as sp


def kl_gauss_scalar(mu1, s1, mu2, s2):
    """KL( N(mu1, s1) || N(mu2, s2) ) with s = variance, scalar case."""
    return sp.Rational(1, 2) * (
        sp.log(s2 / s1) + (s1 + (mu1 - mu2) ** 2) / s2 - 1
    )


# ---- 1. Scalar case, two children ---------------------------------------
print("=" * 70)
print("SCALAR CASE  (two children, weights summing to 1)")
print("=" * 70)

mu1, mu2, s1, s2, w1, w2, mu, s = sp.symbols(
    "mu1 mu2 s1 s2 w1 w2 mu s", real=True, positive=False
)

# enforce w2 = 1 - w1 to bake in the constraint sum w_i = 1
w2_sub = 1 - w1

J = w1 * kl_gauss_scalar(mu1, s1, mu, s) + w2_sub * kl_gauss_scalar(mu2, s2, mu, s)
J = sp.expand(J)

dJ_dmu = sp.simplify(sp.diff(J, mu))
dJ_ds = sp.simplify(sp.diff(J, s))

mu_sol = sp.solve(dJ_dmu, mu)[0]
print("mu* =", sp.simplify(mu_sol))

# substitute mu* and solve for s
J_at_mu = J.subs(mu, mu_sol)
dJ_ds_at_mu = sp.simplify(sp.diff(J_at_mu, s))
s_sol = sp.solve(dJ_ds_at_mu, s)
s_sol = [sp.simplify(x) for x in s_sol]
print("s*  =", s_sol)

# expected
mu_expected = w1 * mu1 + w2_sub * mu2
s_expected = w1 * (s1 + (mu1 - mu_expected) ** 2) + w2_sub * (
    s2 + (mu2 - mu_expected) ** 2
)
s_expected = sp.simplify(sp.expand(s_expected))
print("expected mu_B =", sp.simplify(mu_expected))
print("expected s_B  =", s_expected)

print(
    "mu match :",
    sp.simplify(mu_sol - mu_expected) == 0,
)
print(
    "s  match :",
    any(sp.simplify(x - s_expected) == 0 for x in s_sol),
)

# Hessian check at the critical point: d2J/ds2 should be > 0 for a minimum
d2J_dmu2 = sp.simplify(sp.diff(J, mu, 2))
d2J_ds2 = sp.simplify(sp.diff(J_at_mu, s, 2).subs(s, s_expected))
print("d2J/dmu2 =", d2J_dmu2, "  (positive iff 1/s>0, i.e. s>0)")
print("d2J/ds2 at (mu*, s*) =", sp.simplify(d2J_ds2))

# ---- 2. Identity sum w_i (mu_i - mu_B)^2 = sum w_i mu_i^2 - mu_B^2 -----
print()
print("=" * 70)
print("ALGEBRAIC IDENTITY (sum w_i = 1):")
print("  sum w_i (mu_i - mu_B)^2  ==  sum w_i mu_i^2  -  mu_B^2")
print("=" * 70)

mu3 = sp.symbols("mu3", real=True)
w1s, w2s, w3s = sp.symbols("w1 w2 w3", real=True)
constraint = w1s + w2s + w3s - 1
mu_B = w1s * mu1 + w2s * mu2 + w3s * mu3
lhs = (
    w1s * (mu1 - mu_B) ** 2 + w2s * (mu2 - mu_B) ** 2 + w3s * (mu3 - mu_B) ** 2
)
rhs = w1s * mu1**2 + w2s * mu2**2 + w3s * mu3**2 - mu_B**2
diff = sp.expand(lhs - rhs)
# This identity holds only modulo sum w_i = 1.  Reduce using the constraint.
diff_reduced = sp.simplify(diff.subs(w3s, 1 - w1s - w2s))
print("LHS - RHS (no constraint applied):", sp.simplify(diff))
print("LHS - RHS (with sum w_i = 1):     ", diff_reduced)

# Show what goes wrong without sum w_i = 1
print()
print("Without sum w_i = 1, residual is:")
S = sp.symbols("S", real=True)  # S = sum w_i
# replace via numeric perturbation: use w3 free, S = w1+w2+w3
free_diff = sp.simplify(diff)
print("  ->", free_diff, "(generically nonzero unless sum w_i = 1)")

# ---- 3. Multivariate vector case (d=2) for mean ------------------------
print()
print("=" * 70)
print("VECTOR CASE  (d=2, two children) - mean only")
print("=" * 70)

m1 = sp.Matrix(sp.symbols("m1_1 m1_2", real=True))
m2 = sp.Matrix(sp.symbols("m2_1 m2_2", real=True))
mvec = sp.Matrix(sp.symbols("mu_1 mu_2", real=True))
S1 = sp.Matrix(2, 2, sp.symbols("S1_11 S1_12 S1_21 S1_22", real=True))
S2 = sp.Matrix(2, 2, sp.symbols("S2_11 S2_12 S2_21 S2_22", real=True))
P = sp.Matrix(2, 2, sp.symbols("P_11 P_12 P_21 P_22", real=True))  # Sigma^{-1}
# expectation under N(m_i, S_i) of (x-mu)^T P (x-mu) = trace(P S_i) + (m_i-mu)^T P (m_i-mu)
def quad_term(mi, Si):
    return sp.trace(P * Si) + ((mi - mvec).T * P * (mi - mvec))[0, 0]

J_vec = w1 * quad_term(m1, S1) + (1 - w1) * quad_term(m2, S2)

grad_mu = sp.Matrix([sp.diff(J_vec, v) for v in mvec])
sols = sp.solve(grad_mu, list(mvec))
print("Critical mu (vector) =", sols)
expected_vec = w1 * m1 + (1 - w1) * m2
print(
    "Matches w1*m1 + w2*m2:",
    all(
        sp.simplify(sols[mvec[i]] - expected_vec[i]) == 0 for i in range(2)
    ),
)

# ---- 4. Sigma derivation via Sigma^{-1} = Lambda ----------------------
print()
print("=" * 70)
print("SCALAR Sigma via precision Lambda = 1/s parametrization")
print("=" * 70)
Lam = sp.symbols("Lam", positive=True)
# -log q(x) = -1/2 log Lam + 1/2 Lam (x-mu)^2 + const
# E_{q_i}[(x-mu)^2] = s_i + (mu_i - mu)^2
J_lam = (
    -sp.Rational(1, 2) * sp.log(Lam)
    + sp.Rational(1, 2)
    * Lam
    * (
        w1 * (s1 + (mu1 - mu_sol) ** 2)
        + (1 - w1) * (s2 + (mu2 - mu_sol) ** 2)
    )
)
dJ_dLam = sp.simplify(sp.diff(J_lam, Lam))
Lam_sol = sp.solve(dJ_dLam, Lam)[0]
s_via_Lam = sp.simplify(1 / Lam_sol)
print("s* via Lambda parametrization =", s_via_Lam)
print(
    "Matches expected:",
    sp.simplify(s_via_Lam - s_expected) == 0,
)
