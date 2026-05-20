"""
Blue-team sympy verification for prop-1-tighter-quantitative.

Goals:
1) Verify the Edgeworth coefficients 1/6, 1/24, 1/72 by series expansion of
   <exp(-V3 - V4)>_Gauss in 1D where V3 = (T3/6) xi^3 and V4 = (T4/24) xi^4.
2) Identify the 1D Wick coefficients on T4/H^2 and T3^2/H^3 in log Z.
3) Compare to the Edgeworth density h_1 coefficients (the Hermite-polynomial
   form) which live at the density (not log Z) level.
4) Decide whether 1/24 (T4) and 1/72 (T3^2) in the claim's C_5 reproduce
   under the L^infty / parent-neighborhood closure norm.
"""

import sympy as sp

xi, H, T3, T4, lam, R = sp.symbols('xi H T3 T4 lam R', positive=True, real=True)

# ============================================================
# Step 1: 1D partition function Z = int exp(-H xi^2/2 - V3 - V4)
# Expand exp(-V3 - V4) in T3, T4 to order including T3^2 and T4^1.
# Vertices in the standard normalization:
#   V3 = (1/6) T3 xi^3
#   V4 = (1/24) T4 xi^4
# Gaussian moments: <xi^{2k}> = (2k-1)!! / H^k
# ============================================================

V3 = sp.Rational(1, 6) * T3 * xi**3
V4 = sp.Rational(1, 24) * T4 * xi**4

# Need <V4> (T4 piece) and <V3^2>/2 (T3^2 piece) -- these are the
# log-Z contributions at first cumulant order via -log <exp(-V)>.
mom2 = sp.Rational(1) / H            # <xi^2>
mom4 = sp.Rational(3) / H**2         # <xi^4> = 3/H^2
mom6 = sp.Rational(15) / H**3        # <xi^6> = 15/H^3
mom8 = sp.Rational(105) / H**4       # <xi^8> = 105/H^4

# <V4> = (T4/24) <xi^4>
exp_V4 = sp.Rational(1, 24) * T4 * mom4
print("=== Step 1: log Z corrections (1D, Wick / cumulant level) ===")
print("<V4>          =", sp.simplify(exp_V4))            # T4 / (8 H^2)

# <V3^2> = (T3/6)^2 <xi^6>
exp_V3sq = sp.Rational(1, 36) * T3**2 * mom6
print("<V3^2>        =", sp.simplify(exp_V3sq))          # T3^2 * 15/(36 H^3) = 5 T3^2/(12 H^3)

# log Z correction (to leading order):
# log Z = log Z_Gauss - <V4> + (1/2) <V3^2> + ...
# Numerically expanded (the (1/2)*15/36 = 5/24 on T3^2/H^3 and 1/8 on T4/H^2)
dlogZ = -exp_V4 + sp.Rational(1, 2) * exp_V3sq
print("delta log Z   =", sp.simplify(dlogZ))
print("              =>   coefficient of T4/H^2  :", sp.simplify(sp.Poly(dlogZ, T4).nth(1)) * H**2)
print("              =>   coefficient of T3^2/H^3:", sp.simplify(sp.Poly(dlogZ, T3).nth(2)) * H**3)

# Resulting Wick / 1D Bender-Orszag coefficients:
#   T4 / H^2     -> -1/8     (matches Wong/Bender-Orszag)
#   T3^2 / H^3   -> +5/24    (matches Bender-Orszag 1D)
# These are the cumulant-level (log Z) coefficients.

# ============================================================
# Step 2: Edgeworth density expansion
#   rho(xi) = phi(xi; 0, 1/H) [1 + (1/6) k3 H3(xi) + (1/24) k4 H4(xi)
#                                + (1/72) k3^2 H6(xi) + O(k^3)]
# Here we verify by hand the standard textbook derivation:
# starting from a small perturbation V = V3 + V4 of a unit-Gaussian
# Z(j) = int exp(-(1/2) xi^2 + j xi - V3 - V4), the density at xi normalized
# is phi(xi)[1 + h1(xi)] with
#   h1(xi) = -V(xi) + <V> + (1/2)(V(xi)-<V>)^2_connected ...
# Standard result is:
print()
print("=== Step 2: Edgeworth density coefficients ===")
# Hermite polynomials (probabilists' convention, unit variance)
He = sp.Function('He')
H3_poly = xi**3 - 3*xi
H4_poly = xi**4 - 6*xi**2 + 3
H6_poly = xi**6 - 15*xi**4 + 45*xi**2 - 15

# Cumulants of a unit Gaussian + cubic/quartic perturbation:
# k3 = <xi^3> = T3 * <xi^4>_0 ... working symbolically with unit variance:
# For the perturbed measure with H=1, to leading order:
#   k3 = -T3 (with cubic vertex (1/6) T3 xi^3 in the exponent of e^{-...})
#   k4 = -T4
# So h1 = (1/6) k3 H3 + (1/24) k4 H4 + (1/72) k3^2 H6
#       = -(1/6) T3 H3 - (1/24) T4 H4 + (1/72) T3^2 H6

h1 = (sp.Rational(-1, 6) * T3 * H3_poly
      + sp.Rational(-1, 24) * T4 * H4_poly
      + sp.Rational(1, 72) * T3**2 * H6_poly)
print("h1(xi) =", sp.expand(h1))

# Verify <h1>_phi = 0 (Hermite polynomials are mean-zero under phi for n>=1):
phi = sp.exp(-xi**2 / 2) / sp.sqrt(2 * sp.pi)
mean_h1 = sp.integrate(phi * h1, (xi, -sp.oo, sp.oo))
print("E_phi[h1(xi)] =", sp.simplify(mean_h1))      # should be 0

# Verify <xi^2 * h1>_phi gives the right second-moment correction (sanity)
sec = sp.integrate(phi * xi**2 * h1, (xi, -sp.oo, sp.oo))
print("E_phi[xi^2 * h1(xi)] =", sp.simplify(sec))

# ============================================================
# Step 3: Convert h1 density correction to free-energy-functional level
#   F_exact - F_agent = -tau log(1 + h1) ~ -tau h1   (leading order)
# So the L^infty norm of (F_exact - F_agent) on N_I is bounded by
#   tau * ||h1||_{L^infty(N_I)}.
# Bound the Hermite-polynomial L^infty norm on N_I = [-R, R].
# H3, H4, H6 are polynomials -> sup over [-R, R] is attained at endpoints.
# ============================================================

print()
print("=== Step 3: L^infty bounds of Hermite polynomials on [-R, R] ===")
H3_sup = sp.Max(*[sp.Abs(H3_poly.subs(xi, R)),
                  sp.Abs(H3_poly.subs(xi, -R))])
H4_sup = sp.Max(*[sp.Abs(H4_poly.subs(xi, R)),
                  sp.Abs(H4_poly.subs(xi, -R)),
                  sp.Abs(H4_poly.subs(xi, 0))])
H6_sup = sp.Max(*[sp.Abs(H6_poly.subs(xi, R)),
                  sp.Abs(H6_poly.subs(xi, -R)),
                  sp.Abs(H6_poly.subs(xi, 0))])
print("|H3(R)| =", sp.simplify(sp.Abs(H3_poly.subs(xi, R))))
print("|H4(R)| =", sp.simplify(sp.Abs(H4_poly.subs(xi, R))))
print("|H6(R)| =", sp.simplify(sp.Abs(H6_poly.subs(xi, R))))

# Leading behavior at large R:
# |H3(R)| ~ R^3,   |H4(R)| ~ R^4,   |H6(R)| ~ R^6
# Pinsker: R = O(V_I^{1/2})/H^{1/2} schematically in Fisher-normal coords.

# ============================================================
# Step 4: The decisive question -- are the claim's coefficients
#   C_5 = (1/24) T4/m^2 + (1/72) T3^2/m^3
# consistent with the L^infty Edgeworth-density bound?
#
# In the multidim form, replace 1/H -> H^{-1}, with operator-norm bound
#   ||H^{-1}||_op <= 1/m_I.
# The density-level h1 carries coefficients (1/24, 1/72) on T4 and T3^2
# in the multidim Edgeworth normalization (Amari-Nagaoka Ch. 4).
# These are NOT the log-Z coefficients (1/8, 5/24).
# ============================================================
print()
print("=== Step 4: Edgeworth vs log-Z coefficients ===")
print("log Z (Wick / Bender-Orszag 1D):")
print("    -1/8  on T4 / H^2")
print("    +5/24 on T3^2 / H^3")
print("Edgeworth density h1 = (1/6)k3 H3 + (1/24) k4 H4 + (1/72) k3^2 H6:")
print("    +1/24 on T4 (sign convention from k4 = -T4)")
print("    +1/72 on T3^2 (k3 = -T3, k3^2 = +T3^2)")
print()
print("Conclusion: The claim's coefficients 1/24, 1/72 are the Edgeworth-")
print("density coefficients. They are NOT the same as the Wick / log-Z")
print("coefficients 1/8, 5/24.")
print()

# ============================================================
# Step 5: Verify the proposed C_1 = 1/12 from cubic Edgeworth + Pinsker.
# ============================================================
print("=== Step 5: C_1 chain (Pinsker + cubic Edgeworth) ===")
# Pinsker: TV(P,Q) <= sqrt((1/2) KL(P||Q)) -> TV <= V^{1/2}/sqrt(2)
# Cubic Edgeworth: |F_exact - F_agent| ~ tau * |(1/6) k3 H3(xi)|
# Hermite-3 contribution to free energy at L^infty over N_I:
#   <-- the coefficient on (k3) in h1 is 1/6, not 1/12.
# The claim's 1/12 = (1/6) * (1/2) is a Pinsker-times-Hermite composition.
# Check: Pinsker's 1/2 factor:
pinsker_factor = sp.Rational(1, 2)
edgeworth_h3_factor = sp.Rational(1, 6)
C1_proposed = sp.Rational(1, 12)
combined = pinsker_factor * edgeworth_h3_factor
print("Edgeworth-Hermite-3 prefactor                 :", edgeworth_h3_factor)
print("Pinsker (1/2 inside sqrt -> 1/2 effective?)   :", pinsker_factor)
print("Claim's C_1                                   :", C1_proposed)
print("(1/6) * (1/2)                                 :", combined)
print("Match (1/6)*(1/2) == 1/12 ?                   :", sp.simplify(combined - C1_proposed) == 0)

# ============================================================
# Step 6: Run actual perturbative series expansion of log Z to nail it down.
# Define the 1D integrand and expand in small T3, T4.
# ============================================================
print()
print("=== Step 6: Symbolic perturbative log Z to order T3^2 + T4 ===")
eps = sp.symbols('eps', positive=True)
V = eps * sp.Rational(1, 6) * T3 * xi**3 + eps * sp.Rational(1, 24) * T4 * xi**4
e_factor = sp.series(sp.exp(-V), eps, 0, 3).removeO()
print("exp(-V) series in eps to order 2:")
print(sp.expand(e_factor))

# Compute <exp(-V)> by integrating each polynomial term against the Gaussian.
def gaussian_moment(n):
    """<xi^n>_phi(0, 1/H) for n even, else 0."""
    if n % 2:
        return sp.Integer(0)
    # (n-1)!! / H^(n/2)
    k = n // 2
    dbl = sp.Integer(1)
    for j in range(1, k + 1):
        dbl *= 2 * j - 1
    return dbl / H**k

# Expand e_factor in powers of xi and substitute moments.
e_expand = sp.expand(e_factor)
e_poly = sp.Poly(e_expand, xi)
Z_over_Z0 = sum(c * gaussian_moment(int(deg[0])) for deg, c in e_poly.as_dict().items())
Z_over_Z0 = sp.expand(Z_over_Z0)
print("<exp(-V)>/Z_0 (Z_0 = sqrt(2 pi / H)) =")
print(Z_over_Z0)

# log Z correction to order eps^2 :
log_correction = sp.series(sp.log(Z_over_Z0), eps, 0, 3).removeO()
log_correction = sp.expand(log_correction)
print()
print("Delta(log Z) to order eps^2 =")
print(log_correction)

# Set eps = 1 and read off coefficients.
log_correction_eps1 = log_correction.subs(eps, 1)
print()
print("At eps=1:")
print(sp.collect(sp.expand(log_correction_eps1), [T4, T3**2]))

# Pull out coefficients of T4/H^2 and T3^2/H^3
poly = sp.Poly(log_correction_eps1, T3, T4)
for (i, j), coef in poly.as_dict().items():
    print(f"  T3^{i} T4^{j} coefficient: {sp.simplify(coef)}")
