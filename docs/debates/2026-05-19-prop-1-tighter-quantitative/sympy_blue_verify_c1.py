"""
Blue-team follow-up sympy: nail down the C_1 = 1/12 derivation.

Two questions:
(a) What is the precise Pinsker factor in the chain that gives V^{3/2}?
(b) Are the relations 1/8 = 3 * (1/24) and 5/24 = 15 * (1/72) exact, and
    do they match |H_4(0)| = 3 and |H_6(0)| = 15?
"""
import sympy as sp

xi, V, k3, k4, tau, m, R = sp.symbols('xi V k3 k4 tau m R', positive=True, real=True)

# (a) Pinsker chain in detail.
# KL = V. Pinsker: ||P - Q||_TV <= sqrt((1/2) KL).
# So |E_P[f] - E_Q[f]| <= 2 ||f||_inf ||P-Q||_TV <= 2 ||f||_inf sqrt(V/2)
# But for the Edgeworth cubic correction we want a more direct route:
# at small V, kappa_3 (third cumulant) ~ V^{3/2} for a Gaussian-perturbed measure,
# because cumulants of order n scale like (KL)^{n/2} via Cramer/Edgeworth scaling.
# More precisely, in canonical coords with KL = (1/2) ||delta||^2_F:
#   ||delta||_F ~ sqrt(2V),  k_3 = <delta^3>_F ~ ||delta||^3 = (2V)^{3/2}
#
# The Edgeworth coefficient on Hermite-3 is 1/6, giving density correction
# h1_cubic = (1/6) k3 H3(xi) (in canonical normal coords).
# Apply Pinsker-style L^infty closure-norm bound at the free-energy-functional level:
#   |F_exact - F_agent|_{L^infty(N_I)} = tau * sup |h1| <= tau * (1/6) |k3| * sup_{N_I} |H_3|
# But the claim packages this into a V^{3/2} bound with C_1 = 1/12.
# So we need: (1/6) * |k3| with |k3| <= (2V)^{3/2}/2^{3/2} = V^{3/2}?
#
# Check: if cubic cumulant in normal coordinates is k3 ~ V^{3/2} * 2^{3/2},
# then (1/6) k3 ~ (2^{3/2}/6) V^{3/2}.   2^{3/2}/6 = 2.828/6 = 0.471, not 1/12 = 0.083.
# So the naive product gives a coefficient ~ 0.47, NOT 1/12.
#
# The correct chain: claim's C_1 = 1/12 comes from a different identification.
# It is *not* simply (1/6)*(1/2). It is the V^{3/2} coefficient appearing in
# the Edgeworth-Cornish-Fisher inversion when one bounds the L^infty norm of
# the residual divergence on N_I after matching the first two moments.

# The honest statement: the numeric match (1/6) * (1/2) = 1/12 is *suggestive*
# but not a complete derivation. The 1/2 factor would have to come from
# *somewhere* (Pinsker has 1/2 inside a square root, giving 1/sqrt(2) net).
# Let's check the explicit composition both ways.

print("=== Test: Pinsker-cubic Edgeworth composition for V^{3/2} ===")
# Option A: claim's intended composition (1/6) * (1/2) = 1/12 (multiplicative, no sqrt)
opt_A = sp.Rational(1, 6) * sp.Rational(1, 2)
print("Option A: (1/6) * (1/2) =", opt_A, "decimal", float(opt_A))

# Option B: (1/6) * (1/sqrt(2)) -- the Pinsker factor under the square root chain
opt_B = sp.Rational(1, 6) / sp.sqrt(2)
print("Option B: (1/6) * (1/sqrt(2)) =", opt_B, "decimal", float(opt_B))

# Option C: derivation through scale of cumulants in canonical normal coords.
# k3 ~ (sqrt(2 V))^3 = 2^{3/2} V^{3/2}. Multiplied by 1/6 = 2^{3/2}/6 = sqrt(8)/6 = sqrt(2)/3.
opt_C = sp.sqrt(2) / 3
print("Option C: 2^{3/2} / 6 = sqrt(2)/3 =", opt_C, "decimal", float(opt_C))

# Option D: the "Hermite-3 at center" identification:
#   |H_3(R)| ~ R^3 for large R, but central value H_3(0) = 0 (it's odd).
# Half-oscillation of H_3 on [-R, R] in the m=1 normalization:
H3 = xi**3 - 3*xi
H3prime = sp.diff(H3, xi)
extrema = sp.solve(H3prime, xi)
print("H_3 extrema at xi =", extrema)
for e in extrema:
    print(f"  H_3({e}) = {sp.simplify(H3.subs(xi, e))}")
# Max of |H_3| in [-1, 1] is 2 (at xi = +/- 1):
print("|H_3(1)| =", sp.Abs(H3.subs(xi, 1)))

# The 1/12 coefficient is the *Edgeworth-Cornish-Fisher* coefficient bounding
# the supremum of (1/6) k3 H_3(xi) over xi in [-1, 1] in canonical-normal coords:
#   sup_{xi in [-1,1]} |(1/6) k3 H_3(xi)| = (1/6) |k3| * |H_3(1)| = (2/6) |k3| = (1/3) |k3|
# Substituting |k3| <= (2V)^{3/2}: bound = (1/3) * 2^{3/2} V^{3/2} = (2 sqrt(2)/3) V^{3/2}
# Still not 1/12.

# Honest conclusion: the chain leading to 1/12 requires specifying *which*
# composition is used. There is no single "standard" derivation that
# canonically yields 1/12 -- it's a convention-dependent normalization.

print()
print("=== Honest verdict on C_1 = 1/12 ===")
print("The product (1/6) * (1/2) = 1/12 is arithmetically correct, but its")
print("derivation requires specifying:")
print("  - Whether the 1/2 is Pinsker (inside a sqrt) or a different factor.")
print("  - How the Hermite-3 polynomial's L^infty norm on N_I is absorbed.")
print("  - The exact relation between k_3 and V (variance scaling of cumulant).")
print("These specifications are not in the claim. Red can demand them.")

# (b) Verify the 1/8 = 3 * (1/24) and 5/24 = 15 * (1/72) relations.
print()
print("=== Central-Hermite identity for Edgeworth-Wick conversion ===")
H4 = xi**4 - 6*xi**2 + 3
H6 = xi**6 - 15*xi**4 + 45*xi**2 - 15

H4_at_0 = H4.subs(xi, 0)
H6_at_0 = H6.subs(xi, 0)
print(f"H_4(0) = {H4_at_0},  |H_4(0)| = 3")
print(f"H_6(0) = {H6_at_0},  |H_6(0)| = 15")

print()
print("Wick log-Z coefficient on T_4 / H^2 :  1/8")
print("Edgeworth density coefficient on T_4 :  1/24")
print("Ratio: (1/8) / (1/24) =", sp.Rational(1, 8) / sp.Rational(1, 24), "= 3 = |H_4(0)|")
print()
print("Wick log-Z coefficient on T_3^2 / H^3 :  5/24")
print("Edgeworth density coefficient on T_3^2 :  1/72")
print("Ratio: (5/24) / (1/72) =", sp.Rational(5, 24) / sp.Rational(1, 72), "= 15 = |H_6(0)|")

print()
print("So: 1/8  = 3  * (1/24) = |H_4(0)| * (1/24)")
print("    5/24 = 15 * (1/72) = |H_6(0)| * (1/72)")
print()
print("This is exact. The two sets of coefficients are related by the")
print("central Hermite values. The Wick form integrates against the Gaussian")
print("measure (picking up the Hermite-at-zero values via <H_n(xi)>_phi-style")
print("integrals); the Edgeworth form keeps the Hermite polynomial explicit.")

# What this means for the claim:
# The claim's 1/24, 1/72 are the *Edgeworth density* coefficients on T_4 and T_3^2.
# Under the L^infty norm on N_I = {xi : ||xi||_F <= R}, the closure bound
# becomes:
#   ||F_exact - F_agent||_inf <= tau * (1/24) ||T_4||_op * sup|H_4|/m_I^2 + ...
# At the *center* xi = 0, |H_4| = 3, so the central contribution recovers 1/8.
# At the boundary |xi| = R, |H_4(R)| ~ R^4 for large R, blowing up.
# At |xi| = sqrt(3) ~ 1.73 (where H_4 = 0), the contribution vanishes.

# So the claim's coefficients are correct *at xi = 0* (the parent saddle)
# multiplied by an extra factor of 3 (resp. 15) coming from H_n(0).
# If the operator norm of T_n is defined to absorb the Hermite-n at zero,
# the claim is correct as stated.
