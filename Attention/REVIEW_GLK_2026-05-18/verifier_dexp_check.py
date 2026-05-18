"""Independent verification of Eqs. 530 / 534 of GL(K)_supplementary.tex.

Eq. 441: dexp_phi(xi) = (e^{ad_phi} - I)/ad_phi (xi)   [right-trivialized]
Eq. 445: D_phi(exp)[xi] = dexp_phi(xi) * e^phi
Eq. 469: dexp_phi(xi) = int_0^1 e^{t phi} xi e^{(1-t) phi} dt   [Frechet form]

But Eq. 530 claims:
  d(Omega_ij mu_j)/d phi_i^a = dexp_{phi_i}(T_a) * exp(-phi_j) * mu_j
                            ≡ Q_a^{(i)} * exp(-phi_j) * mu_j

Given Omega_ij = exp(phi_i) * exp(-phi_j), chain rule says
  d Omega_ij / d phi_i^a = D_{phi_i}(exp)[T_a] * exp(-phi_j)
                         = (dexp_{phi_i}(T_a) * exp(phi_i)) * exp(-phi_j)    [Frechet/Eq.445]
                         = dexp_{phi_i}(T_a) * Omega_ij

So:
  Eq. 530 is correct ONLY if Q_a means the Frechet object D(exp)[T_a].
  But Eq. 441 defines Q_a (or dexp) as right-trivialized, which lacks the exp(phi_i) factor.

Let me also check Eq. 534 with right-trivialized convention:
  d Sigma_tilde / d phi_i^a = (d Omega/d phi_i^a) Sigma_j Omega^T + Omega Sigma_j (d Omega/d phi_i^a)^T
                            = dexp_{phi_i}(T_a) * Omega_ij * Sigma_j * Omega_ij^T + sym
"""
import numpy as np
from scipy.linalg import expm

np.random.seed(0)
K = 2

# random small non-commuting matrices
phi_i = np.random.randn(K, K) * 0.3
phi_j = np.random.randn(K, K) * 0.3
mu_j = np.random.randn(K)
Sigma_j = np.eye(K) + 0.2 * np.random.randn(K, K)
Sigma_j = Sigma_j @ Sigma_j.T  # make SPD

# generator T_a: pick one for derivative direction
T_a = np.zeros((K, K))
T_a[0, 1] = 1.0  # one specific generator

# Numerical derivative d/dphi_i_a of Omega_ij * mu_j
def Omega(phi_i_val, phi_j_val):
    return expm(phi_i_val) @ expm(-phi_j_val)

eps = 1e-6
# perturb phi_i in direction T_a
phi_i_plus = phi_i + eps * T_a
phi_i_minus = phi_i - eps * T_a
num_d_mu = (Omega(phi_i_plus, phi_j) @ mu_j - Omega(phi_i_minus, phi_j) @ mu_j) / (2 * eps)

print("Numerical d(Omega mu_j)/d phi_i^a (direction T_a):")
print(num_d_mu)
print()

# Right-trivialized dexp series: dexp_phi(xi) = sum_n 1/(n+1)! ad_phi^n(xi)
def ad_pow(phi_val, xi, n):
    out = xi.copy()
    for _ in range(n):
        out = phi_val @ out - out @ phi_val
    return out

def dexp_right_triv(phi_val, xi, n_terms=30):
    s = np.zeros_like(xi)
    fact = 1.0
    for n in range(n_terms):
        if n > 0:
            fact *= (n + 1)
        else:
            fact = 1.0  # (0+1)! = 1
        s = s + ad_pow(phi_val, xi, n) / fact
    # Reset: term is 1/(n+1)! * ad^n(xi)
    s = np.zeros_like(xi)
    cumfact = 1
    for n in range(n_terms):
        cumfact = cumfact * (n + 1) if n > 0 else 1
        s = s + ad_pow(phi_val, xi, n) / float(np.math.factorial(n + 1))
    return s

# More robust: just use the integral form for Frechet
def dexp_frechet(phi_val, xi, n_steps=200):
    """D_phi(exp)[xi] = int_0^1 e^{t phi} xi e^{(1-t) phi} dt (Frechet)"""
    s = np.zeros_like(xi)
    for k in range(n_steps):
        t = (k + 0.5) / n_steps
        s = s + expm(t * phi_val) @ xi @ expm((1 - t) * phi_val) / n_steps
    return s

# right-trivialized: dexp_phi(xi) such that D(exp)[xi] = dexp_phi(xi) * exp(phi)
# Therefore dexp_phi_right(xi) = D_phi(exp)[xi] @ exp(-phi)
D_exp_T_a = dexp_frechet(phi_i, T_a)
exp_phi_i = expm(phi_i)
dexp_right = D_exp_T_a @ np.linalg.inv(exp_phi_i)

print("Right-trivialized dexp_{phi_i}(T_a):")
print(dexp_right)
print()
print("Frechet D_{phi_i}(exp)[T_a]:")
print(D_exp_T_a)
print()

# Manuscript Eq. 530 RHS with right-trivialized Q_a (Eq. 441 reading):
exp_mphi_j = expm(-phi_j)
rhs_eq530_right_triv = dexp_right @ exp_mphi_j @ mu_j
print("Eq. 530 RHS with Q_a = right-trivialized dexp:")
print(rhs_eq530_right_triv)
print()

# Compare to numerical:
print("Numerical truth:")
print(num_d_mu)
print()

# Correct expression: Q_a (right-triv) * Omega_ij * mu_j
Omega_ij = expm(phi_i) @ expm(-phi_j)
correct_right_triv = dexp_right @ Omega_ij @ mu_j
print("Correct: dexp_right * Omega_ij * mu_j:")
print(correct_right_triv)
print()

# Or equivalently Frechet form: D(exp)[T_a] * exp(-phi_j) * mu_j
correct_frechet = D_exp_T_a @ exp_mphi_j @ mu_j
print("Correct (Frechet form): D(exp)[T_a] * exp(-phi_j) * mu_j:")
print(correct_frechet)
print()

print("=== Summary ===")
print(f"Numerical:                {num_d_mu}")
print(f"Eq.530 (right-triv Q_a):  {rhs_eq530_right_triv}  -- MATCHES NUM? {np.allclose(rhs_eq530_right_triv, num_d_mu, atol=1e-4)}")
print(f"Correct (right-triv):     {correct_right_triv}    -- MATCHES NUM? {np.allclose(correct_right_triv, num_d_mu, atol=1e-4)}")
print(f"Correct (Frechet form):   {correct_frechet}       -- MATCHES NUM? {np.allclose(correct_frechet, num_d_mu, atol=1e-4)}")

# Now Eq. 534 / Sigma derivative
print()
print("=== Eq. 534 Sigma derivative ===")
# Numerical
def Sigma_tilde(phi_i_val, phi_j_val):
    O = Omega(phi_i_val, phi_j_val)
    return O @ Sigma_j @ O.T
num_d_Sigma = (Sigma_tilde(phi_i_plus, phi_j) - Sigma_tilde(phi_i_minus, phi_j)) / (2 * eps)
print("Numerical d Sigma_tilde / d phi_i^a:")
print(num_d_Sigma)
print()

# Manuscript Eq. 534 with right-trivialized Q_a:
# Q_a * exp(-phi_j) * Sigma_j * exp(-phi_j)^T * Omega^T + Omega * exp(-phi_j) * Sigma_j * exp(-phi_j)^T * Q_a^T
term1 = dexp_right @ exp_mphi_j @ Sigma_j @ exp_mphi_j.T @ Omega_ij.T
term2 = Omega_ij @ exp_mphi_j @ Sigma_j @ exp_mphi_j.T @ dexp_right.T
eq534_rhs = term1 + term2
print("Eq. 534 RHS with Q_a = right-trivialized:")
print(eq534_rhs)
print()

# Correct form: (dOmega/dphi) * Sigma_j * Omega^T + Omega * Sigma_j * (dOmega/dphi)^T
# with dOmega/dphi = D_phi(exp)[T_a] * exp(-phi_j) (Frechet) = dexp_right * Omega_ij (right-triv form)
dOmega = D_exp_T_a @ exp_mphi_j  # Frechet form
correct_dSigma = dOmega @ Sigma_j @ Omega_ij.T + Omega_ij @ Sigma_j @ dOmega.T
print("Correct: (dOmega) Sigma_j Omega^T + Omega Sigma_j (dOmega)^T:")
print(correct_dSigma)
print()
print(f"Eq.534 matches numerical? {np.allclose(eq534_rhs, num_d_Sigma, atol=1e-4)}")
print(f"Correct matches numerical? {np.allclose(correct_dSigma, num_d_Sigma, atol=1e-4)}")
