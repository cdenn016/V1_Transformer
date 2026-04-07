"""
Symbolic verification of the Section 11 curvature claims.

Claims to verify:
  (5a) At near-uniform p = 1/V + eps (with sum eps = 0), the entropy gradient
       reduces to  d H / d z_k = -eps_k + O(eps^2)  to leading order.
       This directly contradicts the "o(1/V)" language in the original
       Section 11 — the entropy gradient is O(eps), the same order as the
       CE gradient at random init.

  (5b) The entropy Hessian at uniform is  -(1/V) I + (1/V^2) 1 1^T,
       which on the simplex tangent space has eigenvalue -1/V.  Fisher
       information of the uniform categorical is (1/V) I on the tangent.
       Both terms have curvature O(1/V) at near-uniform init, not O(1)
       for CE vs O(1/V) for entropy.

  (5c) The CE Hessian at the current point p is diag(p) - p p^T, which
       at near-uniform is also -(1/V) I + (1/V^2) 1 1^T (same magnitude
       as entropy Hessian).  The curvature only becomes O(1) for directions
       where the target concentrates mass, which requires the readout to
       have developed structure through training.

  (5d) Numerical check with V=6: ||grad H||_2 and ||grad CE||_2 at a
       specific near-uniform perturbation are the same order of magnitude.
       Both scale as sigma_eps / sqrt(V) in the l2 norm.
"""
import sympy as sp

# -----------------------------------------------------------------
# 5a. Entropy gradient at near-uniform equals -eps to leading order
# -----------------------------------------------------------------
print("=" * 70)
print("Claim 5a: d H / d z_k = -eps_k + O(eps^2) at near-uniform")
print("=" * 70)

# V=3 for tractability
z1, z2, z3 = sp.symbols('z1 z2 z3', real=True)

def softmax_sym(zs):
    ez = [sp.exp(zi) for zi in zs]
    Z = sum(ez)
    return [e / Z for e in ez]

p = softmax_sym([z1, z2, z3])
H = -(p[0]*sp.log(p[0]) + p[1]*sp.log(p[1]) + p[2]*sp.log(p[2]))

# Parameterize z_k = t * eps_k where eps are small perturbations summing to zero.
# Use z1 = t*e1, z2 = t*e2, z3 = -t*(e1+e2) to enforce the constraint.
t, e1, e2 = sp.symbols('t e1 e2', real=True)
z_param = {z1: t*e1, z2: t*e2, z3: -t*(e1 + e2)}

# Expand the gradient as a power series in t
for k, zk in enumerate([z1, z2, z3]):
    grad_k = sp.diff(H, zk).subs(z_param)
    series = sp.series(grad_k, t, 0, 3).removeO()
    expanded = sp.expand(series)
    print(f"  d H / d z{k+1} near uniform = {expanded}")

# Check: leading order of d H / d z_k equals -eps_k (probability perturbation).
# CAREFUL: the logit-space e_k is NOT the same as the probability-space eps_k.
# With z_k = t*e_k (and mean-zero e's), the softmax expansion gives
#   p_k = 1/V + (t/V)*e_k + O(t^2)
# so eps_k = (t/V)*e_k, i.e., eps_k is smaller than e_k by a factor of V.
# The user's claim is d H / d z_k = -eps_k = -(t/V)*e_k, so the coefficient
# of t in the series expansion of d H / d z_k should be -e_k/V.
print("\n  Check: leading order of d H / d z_k equals -eps_k = -(1/V) * e_k")
V_sym = 3
for k, zk in enumerate([z1, z2, z3]):
    grad_k = sp.diff(H, zk).subs(z_param)
    coef_t = sp.series(grad_k, t, 0, 2).removeO().coeff(t)
    if k == 0:
        e_raw = e1
    elif k == 1:
        e_raw = e2
    else:
        e_raw = -(e1 + e2)
    expected_coef = -e_raw / V_sym
    match = sp.simplify(coef_t - expected_coef) == 0
    print(f"    coef of t in d H / d z{k+1}: {sp.simplify(coef_t)}   "
          f"expected -e{k+1}/V = {sp.simplify(expected_coef)}   match = {match}")

# -----------------------------------------------------------------
# 5b. Entropy Hessian at uniform
# -----------------------------------------------------------------
print("\n" + "=" * 70)
print("Claim 5b: Entropy Hessian at uniform = -(1/V) I + (1/V^2) 1 1^T")
print("=" * 70)

uniform = {z1: 0, z2: 0, z3: 0}
V = 3
H_hess = sp.Matrix([[sp.diff(H, zi, zj) for zj in [z1, z2, z3]] for zi in [z1, z2, z3]])
H_hess_uniform = H_hess.subs(uniform)
print(f"  Hessian of H at uniform (V={V}):")
sp.pprint(H_hess_uniform)

# Expected: -(1/V) I + (1/V^2) J where J is all-ones
expected_hess = sp.Rational(-1, V) * sp.eye(V) + sp.Rational(1, V**2) * sp.ones(V, V)
print(f"\n  Expected: -(1/{V}) I + (1/{V**2}) J")
sp.pprint(expected_hess)

diff = sp.simplify(H_hess_uniform - expected_hess)
print(f"\n  Difference: {diff}  (should be zero matrix)")

# Eigenvalues — on the tangent space (orthogonal to 1), eigenvalue should be -1/V
eigenvals = H_hess_uniform.eigenvals()
print(f"\n  Eigenvalues of H Hessian at uniform: {eigenvals}")
print(f"  (Expected: -1/{V} with multiplicity V-1 = 2, and 0 with multiplicity 1 for the 1-direction)")

# -----------------------------------------------------------------
# 5c. CE Hessian at target = uniform
# -----------------------------------------------------------------
print("\n" + "=" * 70)
print("Claim 5c: CE Hessian at p = diag(p) - p p^T.  At p ~ uniform,")
print("          curvature matches entropy's up to sign.")
print("=" * 70)

t1, t2, t3 = sp.symbols('t1 t2 t3', positive=True)
CE = -(t1*sp.log(p[0]) + t2*sp.log(p[1]) + t3*sp.log(p[2]))

CE_hess = sp.Matrix([[sp.diff(CE, zi, zj) for zj in [z1, z2, z3]] for zi in [z1, z2, z3]])
CE_hess_at_uniform = sp.simplify(CE_hess.subs(uniform))
print(f"  CE Hessian at p=uniform (any target t, since second deriv is target-free):")
sp.pprint(CE_hess_at_uniform)

# Compare with +(1/V) I - (1/V^2) J
expected_CE_hess = sp.Rational(1, V) * sp.eye(V) - sp.Rational(1, V**2) * sp.ones(V, V)
print(f"\n  Expected: +(1/{V}) I - (1/{V**2}) J")
sp.pprint(expected_CE_hess)

diff_CE = sp.simplify(CE_hess_at_uniform - expected_CE_hess)
print(f"\n  Difference: {diff_CE}  (should be zero matrix)")

ce_eigenvals = CE_hess_at_uniform.eigenvals()
print(f"\n  Eigenvalues of CE Hessian at p=uniform: {ce_eigenvals}")
print(f"  (Expected: +1/{V} with multiplicity V-1 = 2, and 0 with multiplicity 1)")

print("\n  CONCLUSION: both Hessians have the SAME curvature magnitude 1/V on")
print("  the simplex tangent space at random init.  The user's correction is")
print("  valid: both objectives have the same O(1/V) curvature at uniform,")
print("  and the 'CE has O(1) curvature' claim only holds once the target t")
print("  concentrates mass (requires training).")

# -----------------------------------------------------------------
# 5d. Numerical check: ||grad H||_2 vs ||grad CE||_2 at random near-uniform
# -----------------------------------------------------------------
print("\n" + "=" * 70)
print("Claim 5d: ||grad H||_2 and ||grad CE||_2 at near-uniform have same order")
print("=" * 70)

import random
random.seed(42)

V_num = 6
sigma_ell = sp.Rational(1, 10)  # small logit perturbation

# Draw random logits with std sigma_ell
ell = [sp.Rational(random.randint(-100, 100), 1000) for _ in range(V_num)]
# Center them so the mean is zero (WLOG for softmax)
mean_ell = sum(ell) / V_num
ell = [e - mean_ell for e in ell]

# Compute softmax
ez_vals = [sp.exp(e) for e in ell]
Z_val = sum(ez_vals)
p_vals = [e / Z_val for e in ez_vals]

# Entropy gradient at these logits
# d H / d z_k = -p_k (log p_k + H)
H_val = -sum(p_vals[v] * sp.log(p_vals[v]) for v in range(V_num))
grad_H = [-(p_vals[v] * (sp.log(p_vals[v]) + H_val)) for v in range(V_num)]
grad_H_l2 = sp.sqrt(sum(g**2 for g in grad_H))

# CE gradient with a DIFFERENT random near-uniform target
random.seed(100)
ell_tgt = [sp.Rational(random.randint(-100, 100), 1000) for _ in range(V_num)]
mean_tgt = sum(ell_tgt) / V_num
ell_tgt = [e - mean_tgt for e in ell_tgt]
ez_tgt = [sp.exp(e) for e in ell_tgt]
Z_tgt = sum(ez_tgt)
t_vals = [e / Z_tgt for e in ez_tgt]

grad_CE = [p_vals[v] - t_vals[v] for v in range(V_num)]
grad_CE_l2 = sp.sqrt(sum(g**2 for g in grad_CE))

print(f"  V = {V_num}, sigma_ell ~ 0.058 (random draws)")
print(f"  ||grad H||_2  at near-uniform p:      {float(grad_H_l2):.6f}")
print(f"  ||grad CE||_2 at near-uniform (p,t):  {float(grad_CE_l2):.6f}")
ratio = float(grad_CE_l2) / float(grad_H_l2)
print(f"  ratio CE/H = {ratio:.3f}  (expected: O(sqrt(2)) ~ 1.4, NOT V-scaled)")

# Sanity check: predicted scale is sigma_ell / sqrt(V) for both
import math
sigma_ell_num = sum(float(e)**2 for e in ell) / V_num  # variance of ell
sigma_ell_std = math.sqrt(sigma_ell_num)
predicted_scale = sigma_ell_std / math.sqrt(V_num)
print(f"  predicted scale sigma_ell / sqrt(V) = {predicted_scale:.6f}")
print(f"  ||grad H|| / predicted = {float(grad_H_l2) / predicted_scale:.3f}")
print(f"  ||grad CE||/ predicted = {float(grad_CE_l2) / predicted_scale:.3f}")
