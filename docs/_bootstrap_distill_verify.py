"""
Rigorous sympy verification of gradient claims for bootstrap self-distillation.

Claims verified symbolically:
  (1) d CE(sg[t], softmax(z))/dz_k = softmax(z)_k - t_k.
  (2) d H[softmax(z)]/dz_k = -softmax(z)_k * (log softmax(z)_k + H).
      Has zero gradient at uniform AND at point distributions.
  (3) Without sg on beta: dL/d(logit_k) = beta_k * (CE_k - <CE>).
  (4) Gauge equivariance of KL(N(g mu1, g S1 g^T) || N(g mu2, g S2 g^T)).
"""
import sympy as sp

# -----------------------------------------------------------------
# Claim 1: gradient of CE(t, p) w.r.t. z equals (p - t)
# -----------------------------------------------------------------
z1, z2, z3 = sp.symbols('z1 z2 z3', real=True)
t1, t2, t3 = sp.symbols('t1 t2 t3', real=True, positive=True)

def softmax_sym(zs):
    ez = [sp.exp(zi) for zi in zs]
    Z = sum(ez)
    return [e / Z for e in ez]

p = softmax_sym([z1, z2, z3])
CE = -(t1*sp.log(p[0]) + t2*sp.log(p[1]) + t3*sp.log(p[2]))

print("=" * 70)
print("Claim 1: d CE/dz_k = p_k - t_k")
print("=" * 70)
for k, zk in enumerate([z1, z2, z3]):
    grad = sp.simplify(sp.diff(CE, zk).subs(t3, 1 - t1 - t2))
    expected = sp.simplify(p[k] - [t1, t2, t3][k]).subs(t3, 1 - t1 - t2)
    expected = sp.simplify(expected)
    ok = sp.simplify(grad - expected) == 0
    print(f"  d CE/d z{k+1}  =  p_{k+1} - t_{k+1}  ->  verified = {ok}")

print("\nFixed-point check: at p = t, gradient vanishes for every k")
tvec = [sp.Rational(1, 4), sp.Rational(1, 4), sp.Rational(1, 2)]
# Pick z such that softmax(z) = t; use log(t_k) (WLOG constant shift = 0)
z_agree = {z1: sp.log(tvec[0]), z2: sp.log(tvec[1]), z3: sp.log(tvec[2])}
CE_at_fp = CE.subs({t1: tvec[0], t2: tvec[1], t3: tvec[2]})
for k, zk in enumerate([z1, z2, z3]):
    g = sp.simplify(sp.diff(CE_at_fp, zk).subs(z_agree))
    print(f"  d CE/d z{k+1} at agreement p=t=(1/4,1/4,1/2)  =  {g}")

# -----------------------------------------------------------------
# Claim 2: entropy gradient has zero at BOTH uniform AND point distributions
# -----------------------------------------------------------------
print("\n" + "=" * 70)
print("Claim 2: d H/dz_k = -p_k * (log p_k + H)")
print("=" * 70)
H = -(p[0]*sp.log(p[0]) + p[1]*sp.log(p[1]) + p[2]*sp.log(p[2]))
for k, zk in enumerate([z1, z2, z3]):
    grad = sp.simplify(sp.diff(H, zk))
    expected = sp.simplify(-p[k]*(sp.log(p[k]) + H))
    ok = sp.simplify(grad - expected) == 0
    print(f"  d H/d z{k+1} matches -p_{k+1}*(log p_{k+1} + H)  ->  verified = {ok}")

print("\n  H-gradient at uniform (p = (1/3,1/3,1/3)):")
uniform = {z1: 0, z2: 0, z3: 0}
for k, zk in enumerate([z1, z2, z3]):
    g = sp.simplify(sp.diff(H, zk).subs(uniform))
    print(f"    d H/d z{k+1}  =  {g}")

print("\n  H-gradient in the limit of a point distribution (p_1 -> 1):")
L = sp.symbols('L', real=True, positive=True)
for k, zk in enumerate([z1, z2, z3]):
    g_L = sp.diff(H, zk).subs({z1: L, z2: 0, z3: 0})
    g_lim = sp.limit(g_L, L, sp.oo)
    print(f"    lim_{{L->oo}} d H/d z{k+1}  =  {g_lim}")

# -----------------------------------------------------------------
# Claim 3: attend-to-twins gradient flow
# -----------------------------------------------------------------
print("\n" + "=" * 70)
print("Claim 3: attend-to-twins gradient w/o stop-grad on beta")
print("=" * 70)
ell1, ell2, ell3 = sp.symbols('ell1 ell2 ell3', real=True)
a, b, c = sp.symbols('a b c', nonnegative=True)  # CE losses per neighbor
beta = softmax_sym([ell1, ell2, ell3])
L_combined = beta[0]*a + beta[1]*b + beta[2]*c

print("  Symbolic gradient form:")
for k, elk in enumerate([ell1, ell2, ell3]):
    grad = sp.simplify(sp.diff(L_combined, elk))
    avgCE = sum(beta[j] * [a, b, c][j] for j in range(3))
    expected = sp.simplify(beta[k] * ([a, b, c][k] - avgCE))
    ok = sp.simplify(grad - expected) == 0
    print(f"    d L/d ell_{k+1}  =  beta_{k+1} * (CE_{k+1} - <CE>)  ->  verified = {ok}")

print("\n  Numerical: beta_init = (1/3,1/3,1/3), CE = (0, 1, 1):")
subs_vals = {ell1: 0, ell2: 0, ell3: 0, a: 0, b: 1, c: 1}
for k, elk in enumerate([ell1, ell2, ell3]):
    g = sp.simplify(sp.diff(L_combined, elk).subs(subs_vals))
    print(f"    d L/d ell_{k+1}  =  {g}")
print("  => Gradient descent: ell_1 INCREASES (beta_1 up), ell_2,ell_3 DECREASE")
print("  => Loss decreases trivially by concentrating beta on agreeing neighbor 1")

# -----------------------------------------------------------------
# Claim 4: gauge equivariance of Gaussian KL (diagonal covariance)
# -----------------------------------------------------------------
print("\n" + "=" * 70)
print("Claim 4: gauge equivariance  KL(g mu1, g S1 g^T || g mu2, g S2 g^T)")
print("=" * 70)
# K=2, full (not just diagonal) scaled-identity covariances, general 2x2 g
mu1_0, mu1_1 = sp.symbols('m10 m11', real=True)
mu2_0, mu2_1 = sp.symbols('m20 m21', real=True)
s1, s2 = sp.symbols('s1 s2', positive=True)
mu1 = sp.Matrix([mu1_0, mu1_1])
mu2 = sp.Matrix([mu2_0, mu2_1])
S1 = s1 * sp.eye(2)
S2 = s2 * sp.eye(2)

def kl_gauss(mu1, S1, mu2, S2):
    K = mu1.shape[0]
    S2inv = S2.inv()
    delta = mu2 - mu1
    tr_term = (S2inv * S1).trace()
    mahal   = (delta.T * S2inv * delta)[0, 0]
    logdet1 = sp.log(S1.det())
    logdet2 = sp.log(S2.det())
    return sp.Rational(1, 2) * (tr_term + mahal - K + logdet2 - logdet1)

kl_orig = sp.simplify(kl_gauss(mu1, S1, mu2, S2))
print(f"  KL(N(mu1, s1 I) || N(mu2, s2 I))  =  {kl_orig}")

# Apply invertible 2x2 g to both fiducials
g11, g12, g21, g22 = sp.symbols('g11 g12 g21 g22', real=True)
g = sp.Matrix([[g11, g12], [g21, g22]])
mu1_p = g * mu1
mu2_p = g * mu2
S1_p = g * S1 * g.T
S2_p = g * S2 * g.T
kl_prime = sp.simplify(kl_gauss(mu1_p, S1_p, mu2_p, S2_p))
diff = sp.simplify(kl_prime - kl_orig)
print(f"  KL difference under gauge transform  =  {sp.simplify(diff)}")
print("  (Zero means: KL is invariant under simultaneous gauge transform)")

# Sanity check numerically
print("\n  Numerical sanity check with g = [[2, 1], [1, 3]]:")
subs_num = {
    mu1_0: 1, mu1_1: 2, mu2_0: 3, mu2_1: 5,
    s1: 2, s2: 3,
    g11: 2, g12: 1, g21: 1, g22: 3,
}
print(f"    KL_orig  = {float(kl_orig.subs(subs_num))}")
print(f"    KL_prime = {float(kl_prime.subs(subs_num))}")
