"""Finite-difference verification of VFE gradient computation.

Click-to-run: no argparse. Compares analytic VFE gradients from
_compute_vfe_gradients_block_diagonal_diag against central-difference
numerical gradients of the scalar VFE.

Design note on the VFE gradient convention
-------------------------------------------
The analytic gradient function implements a mean-field / coordinate-descent
convention: when computing the gradient for position i, it treats i strictly
as the *query* — i.e. it computes

    dF_i/dmu_i = alpha * dKL(q_i||p_i)/dmu_i
               + lambda * sum_j beta_ij * dKL(q_i || Omega_ij q_j)/dmu_i
               + lambda * sum_j KL_ij * d(beta_ij)/dmu_i   [softmax coupling]

The *key-side* contribution — that mu_i also appears inside
KL(q_j || Omega_ji q_i) for j != i — is deliberately excluded.
This is the mean-field approximation standard in coordinate-descent
variational inference: each factor's update is performed as if its
influence on other factors' KLs is ignored.

FD oracle construction
-----------------------
To match the analytic gradient (not the full joint gradient), the FD
oracle differentiates the *per-position query-side VFE*:

    F_i(mu_i, sigma_i) = alpha * KL(q_i || p_i)
                        + lambda * sum_j beta_ij(mu_i, sigma_i) * KL_ij(mu_i, sigma_i)

where:
- Omega is held fixed at the operating point (phi is not a variable here),
- KL_ij is computed with the perturbed (mu_i, sigma_i) as the query and
  fixed (mu_q, sigma_q) as the keys (key-side dependence excluded),
- beta_ij is recomputed live from the full KL row for query i (so that
  d(beta)/d(mu_i) is captured — the softmax coupling term).

This oracle exactly matches the function the analytic gradient
differentiates, making the FD check a precise unit test.

Double-precision FD
--------------------
The FD oracle runs in float64 to avoid catastrophic cancellation at
near-zero gradient elements.  Float32 machine epsilon (~1.2e-7) times a
typical F0 (~10) gives a noise floor of ~1.2e-6; for elements with
|grad| ~ 1e-3 and eps_fd=1e-4 the signal-to-noise ratio is only ~8,
producing ~10% relative error purely from float32 rounding.  Float64
raises this ratio to ~10^8, making the FD reference reliable.

The analytic gradient is still computed in float32 (matching its
production code path), and compared against the float64 FD reference.

PASS criterion: max relative error < 1e-2 (float64 FD vs float32 analytic).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import math
from math_utils.generators import generate_so3_generators
from transformer.core.gauge_utils import fused_block_matrix_exp_pairs
from transformer.core.vfe_utils import KL_CEIL_BASE, KL_CEIL_SCALE

torch.manual_seed(42)

# ── Config ──────────────────────────────────────────────────────────
B, N, K = 1, 4, 3
irrep_dims = [3]
alpha = 1.0
lam = 1.0          # lambda_belief = lambda_softmax = lam
kappa = 1.0
eps_kl = 1e-6
eps_fd = 1e-5       # FD step (float64 allows smaller step without cancellation)
kl_ceil = max(KL_CEIL_BASE, KL_CEIL_SCALE * K)

# ── Data ────────────────────────────────────────────────────────────
gens_np = generate_so3_generators(K)
gens_f32 = torch.from_numpy(gens_np).float()   # (3, K, K) float32 for analytic
gens_f64 = gens_f32.double()                    # (3, K, K) float64 for FD oracle

mu_q    = torch.randn(B, N, K)
mu_p    = torch.randn(B, N, K)
sigma_q = torch.rand(B, N, K).clamp(min=0.3) + 0.5   # in [0.5, 1.5]
sigma_p = torch.rand(B, N, K).clamp(min=0.3) + 0.5
phi     = 0.3 * torch.randn(B, N, 3)  # small phi for stability

kappa_scaled = kappa * math.sqrt(K)

# ── Transport helper ─────────────────────────────────────────────────
def compute_transport(phi_tensor: torch.Tensor, gens: torch.Tensor) -> torch.Tensor:
    """Return Omega (B, N, N, K, K) from phi (B, N, 3) and generators."""
    phi_mat = torch.einsum('bna,aij->bnij', phi_tensor, gens)
    exp_phi     = torch.linalg.matrix_exp(phi_mat)
    exp_neg_phi = torch.linalg.matrix_exp(-phi_mat)
    return torch.einsum('bikl,bjlm->bijkm', exp_phi, exp_neg_phi)


# ── Pre-compute operating-point quantities in float32 for analytic call ──
# Omega, beta, block_exp_pairs are all fixed at the operating point.
with torch.no_grad():
    Omega0_f32 = compute_transport(phi, gens_f32)   # (B, N, N, K, K)

    sig_t0_f32 = torch.einsum('bijkl,bijkl,bjl->bijk',
                               Omega0_f32, Omega0_f32, sigma_q).clamp(min=eps_kl)
    mu_t0_f32  = torch.einsum('bijkl,bjl->bijk', Omega0_f32, mu_q)
    delta0_f32 = mu_q[:, :, None, :] - mu_t0_f32
    kl0_f32    = (0.5 * (sigma_q[:, :, None, :] / sig_t0_f32
                          + delta0_f32 ** 2 / sig_t0_f32
                          - 1.0
                          + torch.log(sig_t0_f32)
                          - torch.log(sigma_q[:, :, None, :].clamp(min=eps_kl)))
                  ).sum(dim=-1).clamp(min=0.0, max=kl_ceil)   # (B, N, N)
    beta0_f32  = torch.softmax(-kl0_f32 / kappa_scaled, dim=-1)  # (B, N, N)

    block_exp_pairs = fused_block_matrix_exp_pairs(phi, gens_f32, irrep_dims)


# ── Pre-compute operating-point Omega in float64 for FD oracle ──────
with torch.no_grad():
    Omega0_f64 = compute_transport(phi.double(), gens_f64)   # (B, N, N, K, K) float64

    # Pre-compute fixed key-side quantities (never perturbed in the FD oracle)
    mu_q_d    = mu_q.double()
    mu_p_d    = mu_p.double()
    sigma_q_d = sigma_q.double()
    sigma_p_d = sigma_p.double()

    # For each query position i: Omega_i = Omega0_f64[:, i, :, :, :]  (B, N, K, K)
    # mu_j_t_all[b, i, j, k]  = sum_l Omega[b,i,j,k,l] * mu_q[b,j,l]
    mu_j_t_all  = torch.einsum('bijkl,bjl->bijk', Omega0_f64, mu_q_d)  # (B, N, N, K)
    # sig_j_t_all[b, i, j, k] = sum_l Omega[b,i,j,k,l]^2 * sigma_q[b,j,l]
    sig_j_t_all = torch.einsum('bijkl,bijkl,bjl->bijk',
                                Omega0_f64, Omega0_f64, sigma_q_d).clamp(min=1e-15)  # (B, N, N, K)


def query_vfe_i_f64(mu_i_val: torch.Tensor, sig_i_val: torch.Tensor, pos: int) -> torch.Tensor:
    r"""Per-position query-side VFE in float64.

    Evaluates

        F_i(mu_i, sigma_i) = alpha * KL(q_i || p_i)
                            + lambda * sum_j beta_ij(mu_i, sigma_i) * KL_ij(mu_i, sigma_i)

    with Omega held fixed at the operating point and the KL row for query i
    recomputed live (so that d(beta_ij)/d(mu_i) is captured).  Keys mu_j,
    sigma_j are fixed — key-side dependence on mu_i is excluded, matching the
    mean-field convention of the analytic gradient.

    Args:
        mu_i_val:  (K,) float64 mean for position *pos*.
        sig_i_val: (K,) float64 diagonal variance for position *pos*.
        pos:       Integer position index.

    Returns:
        Scalar float64 VFE contribution.
    """
    eps_d = 1e-15   # double-precision floor

    # Self-coupling KL(q_i || p_i)
    mp_i  = mu_p_d[0, pos]    # (K,)
    sp_i  = sigma_p_d[0, pos] # (K,)
    kl_self_pd = 0.5 * (sig_i_val / sp_i + (mp_i - mu_i_val) ** 2 / sp_i
                         - 1.0 + torch.log(sp_i.clamp(min=eps_d))
                         - torch.log(sig_i_val.clamp(min=eps_d)))
    kl_self = kl_self_pd.sum().clamp(min=0.0, max=kl_ceil)

    # Alignment KL_{i,j}(mu_i, sigma_i) with fixed keys
    # Fixed transported quantities for this row (pre-computed above)
    mu_j_t  = mu_j_t_all[0, pos]    # (N, K)
    sig_j_t = sig_j_t_all[0, pos]   # (N, K)

    # delta = mu_i - Omega_{i,j} mu_j  (query-side: only mu_i varies)
    delta_ij = mu_i_val[None, :] - mu_j_t    # (N, K)

    kl_ij_pd = 0.5 * (sig_i_val[None, :] / sig_j_t
                       + delta_ij ** 2 / sig_j_t
                       - 1.0
                       + torch.log(sig_j_t)
                       - torch.log(sig_i_val[None, :].clamp(min=eps_d)))
    kl_ij = kl_ij_pd.sum(dim=-1).clamp(min=0.0, max=kl_ceil)  # (N,)

    # Live beta for position i (softmax coupling is captured here)
    beta_i = torch.softmax(-kl_ij / kappa_scaled, dim=0)   # (N,)

    return alpha * kl_self + lam * (beta_i * kl_ij).sum()


# ── Analytic gradient (float32) ───────────────────────────────────────
from transformer.core.vfe_gradients import _compute_vfe_gradients_block_diagonal_diag

grad_mu_analytic, grad_sigma_analytic = _compute_vfe_gradients_block_diagonal_diag(
    mu_q=mu_q, sigma_q=sigma_q,
    mu_p=mu_p, sigma_p=sigma_p,
    beta=beta0_f32,
    phi=phi,
    generators=gens_f32,
    alpha=alpha,
    lambda_belief=lam,
    lambda_softmax=lam,   # Must equal lambda_belief for FD consistency
    kappa=kappa,
    eps=eps_kl,
    irrep_dims=irrep_dims,
    compute_sigma_align_grad=True,
    cached_block_exp_pairs=block_exp_pairs,
)

# ── Double-precision FD gradient ──────────────────────────────────────
# Each position i is perturbed independently.  Omega and key quantities are
# fixed; only the query's (mu_i, sigma_i) varies.
print("Computing double-precision finite-difference gradients...")
grad_mu_fd    = torch.zeros(N, K, dtype=torch.float64)
grad_sigma_fd = torch.zeros(N, K, dtype=torch.float64)

with torch.no_grad():
    for n in range(N):
        mu_i_base  = mu_q_d[0, n].clone()    # (K,) float64
        sig_i_base = sigma_q_d[0, n].clone()  # (K,) float64

        for k in range(K):
            # FD for mu[n, k]
            mu_plus  = mu_i_base.clone(); mu_plus[k]  += eps_fd
            mu_minus = mu_i_base.clone(); mu_minus[k] -= eps_fd
            grad_mu_fd[n, k] = (query_vfe_i_f64(mu_plus,  sig_i_base, n)
                                - query_vfe_i_f64(mu_minus, sig_i_base, n)) / (2 * eps_fd)

            # FD for sigma[n, k]
            sig_plus  = sig_i_base.clone(); sig_plus[k]  += eps_fd
            sig_minus = sig_i_base.clone(); sig_minus[k] -= eps_fd
            grad_sigma_fd[n, k] = (query_vfe_i_f64(mu_i_base, sig_plus,  n)
                                   - query_vfe_i_f64(mu_i_base, sig_minus, n)) / (2 * eps_fd)

# ── Compare ───────────────────────────────────────────────────────────
def report(name: str, analytic_f32: torch.Tensor, fd_f64: torch.Tensor) -> bool:
    """Report relative error between float32 analytic and float64 FD gradients."""
    an  = analytic_f32.squeeze().float()
    fd  = fd_f64.float()
    abs_err = (an - fd).abs()
    denom   = fd.abs().clamp(min=1e-8)
    rel_err = abs_err / denom
    max_abs  = abs_err.max().item()
    max_rel  = rel_err.max().item()
    mean_rel = rel_err.mean().item()
    passed   = max_rel < 0.01
    status   = "PASS" if passed else "FAIL"
    print(f"\n{name}:")
    print(f"  Max absolute error:  {max_abs:.2e}")
    print(f"  Max relative error:  {max_rel:.2e}")
    print(f"  Mean relative error: {mean_rel:.2e}")
    print(f"  Status: {status}")
    if not passed:
        idx = rel_err.argmax()
        coords: list = []
        temp = idx.item()
        for s in reversed(an.shape):
            coords.append(temp % s)
            temp //= s
        coords.reverse()
        print(f"  Worst at {coords}: analytic={an.flatten()[idx]:.6f}, fd={fd.flatten()[idx]:.6f}")
    return passed

with torch.no_grad():
    F0 = sum(query_vfe_i_f64(mu_q_d[0, n], sigma_q_d[0, n], n) for n in range(N))

print(f"\nVFE Gradient Finite-Difference Verification")
print(f"B={B}, N={N}, K={K}, irrep_dims={irrep_dims}")
print(f"alpha={alpha}, lambda={lam}, kappa={kappa}, eps_fd={eps_fd}")
print(f"FD precision: float64    Analytic precision: float32")
print(f"F0 (sum over positions, float64) = {float(F0):.6f}")
print(f"\nOracle: query-side per-position VFE with live beta and fixed keys.")
print(f"Matches the mean-field coordinate-descent convention of the analytic function.")

p1 = report("grad_mu",    grad_mu_analytic,    grad_mu_fd)
p2 = report("grad_sigma", grad_sigma_analytic, grad_sigma_fd)

if p1 and p2:
    print("\n=== ALL PASS ===")
else:
    print("\n=== SOME FAIL ===")
