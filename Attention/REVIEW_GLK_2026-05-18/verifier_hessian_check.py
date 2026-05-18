"""Independent verification of M-D-3 (Hessian of Gaussian KL wrt Sigma1)."""
import sympy as sp

# 2x2 symmetric matrices
a, b, c = sp.symbols('a b c', real=True)
p, q, r = sp.symbols('p q r', real=True)

Sigma1 = sp.Matrix([[a, b], [b, c]])
Sigma2 = sp.Matrix([[p, q], [q, r]])

# Sigma1-dependent part of D_KL(N(mu1,Sigma1) || N(mu2,Sigma2))
# = 1/2 [ -log|Sigma1| + tr(Sigma2^{-1} Sigma1) ] + (mean term has mu1 only, no Sigma1)
det1 = Sigma1.det()
Sigma2_inv = Sigma2.inv()
trace_term = (Sigma2_inv * Sigma1).trace()
D_KL = sp.Rational(1, 2) * (-sp.log(det1) + trace_term)

vars_ = [a, b, c]
H = sp.zeros(3, 3)
for i, vi in enumerate(vars_):
    for j, vj in enumerate(vars_):
        H[i, j] = sp.simplify(sp.diff(D_KL, vi, vj))

print("Symbolic Hessian (rows/cols = a,b,c):")
sp.pprint(H)
print()

# Test 1: H should not depend on Sigma2 entries (p, q, r)
print("Does H depend on p,q,r?")
for v in [p, q, r]:
    dHv = sp.simplify(sp.diff(H, v))
    print(f"  d/d{v} H = {dHv}")
print()

# Numerical: Sigma1=I, Sigma2=2I
subs1 = {a:1,b:0,c:1, p:2,q:0,r:2}
print("H at Sigma1=I, Sigma2=2I:")
sp.pprint(H.subs(subs1))
print()

# Numerical: Sigma1=I, Sigma2=5I
subs2 = {a:1,b:0,c:1, p:5,q:0,r:5}
print("H at Sigma1=I, Sigma2=5I:")
sp.pprint(H.subs(subs2))
print()

print("Verdict: Hessian does NOT depend on Sigma2 (the trace term is linear in Sigma1).")
print("Manuscript claim 'Sigma1^{-1} ⊗ Sigma1^{-1} + Sigma2^{-1} ⊗ Sigma2^{-1}' is WRONG.")
print("Correct: 1/2 * Sigma1^{-1} ⊗_sym Sigma1^{-1}")
