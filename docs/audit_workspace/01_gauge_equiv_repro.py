"""Empirical gauge-equivariance audit for the skip_attention=True + ift_phi E-step.

Tests, on CPU, the LIVE diagonal+layernorm+rope path:
  (A) diagonal sigma transport == diag(Omega @ diag(sigma) @ Omega^T)  (sandwich, diagonal approx)
  (B) per-head block-diagonal Omega (GL(10) x 2 heads, not full K=20)
  (C) fused kernel KL/beta invariance + grad covariance under a GLOBAL per-head frame g
      with phi shifted so Omega_ij is unchanged (relabeling invariance)
  (D) whether nn.LayerNorm in the live block breaks covariance
"""
import os
import sys
# Repo root is two levels up from docs/audit_workspace/.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import math

torch.manual_seed(0)
# The fused kernel internally forces float32; match it so einsums don't dtype-clash.
torch.set_default_dtype(torch.float32)

from transformer.core.vfe_gradients import _fused_attention_and_vfe_gradients_block_diag
from transformer.core.gauge_utils import fused_block_matrix_exp_pairs

# ---------------------------------------------------------------------------
# Build a GL(10) x 2-head generator basis: per head, the 100 elementary E_ab
# matrices placed block-diagonally. n_gen = 2*100 = 200, K=20, irrep_dims=[10,10].
# ---------------------------------------------------------------------------
K = 20
d = 10
n_heads = 2
irrep_dims = [d, d]

gens = []
for h in range(n_heads):
    off = h * d
    for a in range(d):
        for b in range(d):
            G = torch.zeros(K, K)
            G[off + a, off + b] = 1.0
            gens.append(G)
generators = torch.stack(gens, dim=0)  # (200, K, K)
n_gen = generators.shape[0]
print(f"K={K} d={d} n_heads={n_heads} n_gen={n_gen}")

B, N = 2, 5
mu_q = torch.randn(B, N, K)
sigma_q = torch.rand(B, N, K) + 0.3
mu_p = torch.randn(B, N, K)
sigma_p = torch.rand(B, N, K) + 0.3
# Small phi so exp is well-conditioned; per head.
phi = 0.15 * torch.randn(B, N, n_gen)

kappa = 1.0
eps = 1e-6
alpha = 1.0
lambda_belief = 10.0
lambda_softmax = 0.0  # live: include_attention_entropy -> lambda_softmax_eff=0
ALPHA_DIV = 0.3        # LIVE config: alpha_divergence=0.3 (Renyi branch, NOT KL)

# ---------------------------------------------------------------------------
# (A) + (B): verify per-head Omega is block-diagonal and sigma transport is the
#            diagonal of the sandwich.
# ---------------------------------------------------------------------------
pairs = fused_block_matrix_exp_pairs(phi, generators, irrep_dims)
# Build full Omega_ij for head 0 and check it lives in the 10x10 block.
exp_phi0, exp_neg_phi0 = pairs[0]   # (B,N,10,10)
exp_phi1, exp_neg_phi1 = pairs[1]
# Omega for head 0: exp(phi_i)[h0] @ exp(-phi_j)[h0]
i, j = 1, 3  # pick a pair
Om0 = exp_phi0[0, i] @ exp_neg_phi0[0, j]   # (10,10)
Om1 = exp_phi1[0, i] @ exp_neg_phi1[0, j]
# Diagonal-of-sandwich for head-0 sigma_j block:
sig_j0 = sigma_q[0, j, 0:d]
sandwich0 = Om0 @ torch.diag(sig_j0) @ Om0.T
diag_sandwich0 = torch.diagonal(sandwich0)
# kernel's diagonal transport formula: sum_l Om0[k,l]^2 * sig_j0[l]
kernel_diag0 = torch.einsum('kl,kl,l->k', Om0, Om0, sig_j0)
errA = (diag_sandwich0 - kernel_diag0).abs().max().item()
print(f"(A) max|diag(OmSigOm^T) - kernel_einsum_diag| (head0) = {errA:.3e}")

# Check generators are block-diagonal -> Omega is block diagonal by construction:
# verify head-1 generators are zero in head-0 block and vice versa.
g_h1_in_h0 = generators[d*d:, 0:d, 0:d].abs().max().item()
g_h0_in_h1 = generators[:d*d, d:2*d, d:2*d].abs().max().item()
print(f"(B) head-1 gens leaking into head-0 block: {g_h1_in_h0:.3e} ; head-0 into head-1: {g_h0_in_h1:.3e}")

# ---------------------------------------------------------------------------
# (C) Relabeling invariance of the fused kernel under a global per-head frame g.
#   Action: mu_i -> G mu_i, sigma_i -> diag(G diag(sigma_i) G^T) (diagonal analog),
#           and shift phi so that exp(phi_i') = exp(phi_i) g^{-1} per head, i.e.
#           Omega_ij' = exp(phi_i)g^{-1} g exp(-phi_j) = Omega_ij  (UNCHANGED).
#   With Omega unchanged and mu->G mu, sigma->diag(G sig G^T):
#       transported mu_j' = Omega (G mu_j),  transported sigma' = diag-of-sandwich
#   For TRUE covariance we'd need Sigma_i -> G Sigma_i G^T (full). With diagonal
#   approx the KL is NOT exactly invariant unless g is a permutation/diagonal
#   that preserves diagonality. We test g = a per-head DIAGONAL scaling (which
#   preserves the diagonal covariance structure) -> KL/beta MUST be invariant.
# ---------------------------------------------------------------------------
def run_kernel(mu, sig, ph):
    return _fused_attention_and_vfe_gradients_block_diag(
        mu_q=mu, sigma_q=sig, mu_p=mu_p, sigma_p=sigma_p,
        phi=ph, generators=generators,
        alpha=alpha, lambda_belief=lambda_belief, lambda_softmax=lambda_softmax,
        kappa=kappa, eps=eps, irrep_dims=irrep_dims,
        compute_sigma_align_grad=True,
        use_rope=False, return_kl=True, alpha_div=ALPHA_DIV,
    )

# Baseline
beta0, gmu0, gsig0, kl0 = run_kernel(mu_q, sigma_q, phi)

# Per-head DIAGONAL frame g = diag(s) with s>0 (preserves diagonal covariance).
# To keep Omega_ij unchanged we need exp(phi_i') = exp(phi_i) g^{-1}.
# For a diagonal g = exp(D) with D = sum_a (log s_a) E_aa, and since the E_aa
# generators commute with a diagonal exp(phi) ONLY when phi is also diagonal —
# in general exp(phi)g^{-1} != exp(phi - D). So we cannot simply add to phi.
# Instead we directly rebuild the frame: we want a NEW phi' whose per-head
# exp equals exp(phi)[head] @ g_head^{-1}. We construct g as a diagonal scaling
# and absorb it by LEFT-multiplying mu and the transport. Cleanest test:
#   Define transformed inputs in the SAME frame but rescale the whole problem.
# Simpler exact test: use g = orthogonal? Diagonal sigma only invariant under
# permutation+sign. So test g = a per-head PERMUTATION matrix P (preserves
# diagonal covariance exactly: P diag(s) P^T = diag(P s)).
g_list = []
perm = torch.randperm(d)
P = torch.eye(d)[perm]  # permutation matrix, orthogonal, det=+-1
# Build full K block-diagonal G = blkdiag(P, P)
G = torch.zeros(K, K)
G[0:d, 0:d] = P
G[d:2*d, d:2*d] = P

# Transform mu: mu -> G mu
mu_t = torch.einsum('kl,bnl->bnk', G, mu_q)
mu_p_t = torch.einsum('kl,bnl->bnk', G, mu_p)
# Transform diagonal sigma under permutation: sigma -> P-permuted entries per head
sigma_t = torch.zeros_like(sigma_q)
sigma_t[:, :, 0:d] = sigma_q[:, :, perm]
sigma_t[:, :, d:2*d] = sigma_q[:, :, d + perm]
sigma_p_t = torch.zeros_like(sigma_p)
sigma_p_t[:, :, 0:d] = sigma_p[:, :, perm]
sigma_p_t[:, :, d:2*d] = sigma_p[:, :, d + perm]

# To keep Omega unchanged: phi' such that exp(phi_i')[h] = P @ exp(phi_i)[h] @ P^T
# (conjugation), then Omega'_ij = P exp(phi_i) P^T P exp(-phi_j) P^T
#                                = P Omega_ij P^T. Then transported mu_j' =
#   Omega'(G mu_j) = P Omega_ij P^T P mu_j... wait P^T P = I -> = P Omega_ij mu_j
#   = G (Omega_ij mu_j). And sigma transports as P-permutation of diag-sandwich.
# Conjugating phi by P: since generators are E_ab, phi = sum phi_ab E_ab, and
# P E_ab P^T = E_{perm^{-1}(a) perm^{-1}(b)} ... we permute the phi coefficients.
# Build the index map for the conjugated coefficients.
inv_perm = torch.argsort(perm)
phi_t = torch.zeros_like(phi)
for h in range(n_heads):
    base = h * d * d
    block = phi[:, :, base:base + d * d].reshape(B, N, d, d)
    # P block P^T : row a -> inv_perm? E_{ab} -> E_{perm[a],perm[b]} under P (.)P^T
    # (P M P^T)[i,j] = M[perm^{-1}? ] ; do it numerically to avoid index errors:
    block_conj = torch.einsum('pa,xnab,qb->xnpq', P, block, P)  # P @ block @ P^T
    phi_t[:, :, base:base + d * d] = block_conj.reshape(B, N, d * d)

beta1, gmu1, gsig1, kl1 = run_kernel(mu_t, sigma_t, phi_t)

# KL and beta must be invariant (permutation frame preserves diagonal Gaussians)
errKL = (kl0 - kl1).abs().max().item()
errBeta = (beta0 - beta1).abs().max().item()
# grad_mu should transform covariantly: gmu1 == G @ gmu0
gmu0_t = torch.einsum('kl,bnl->bnk', G, gmu0)
errGmu = (gmu1 - gmu0_t).abs().max().item()
# grad_sigma should permute like sigma
gsig0_t = torch.zeros_like(gsig0)
gsig0_t[:, :, 0:d] = gsig0[:, :, perm]
gsig0_t[:, :, d:2*d] = gsig0[:, :, d + perm]
errGsig = (gsig1 - gsig0_t).abs().max().item()
print(f"(C) permutation-frame: max|dKL|={errKL:.3e} max|dBeta|={errBeta:.3e} "
      f"max|grad_mu covar err|={errGmu:.3e} max|grad_sigma covar err|={errGsig:.3e}")
# DEBUG: locate where grad_mu covariance breaks. Decompose by recomputing the
# self term alone (lambda_belief=0) vs direct term (alpha=0).
def run_kernel_d(mu, sig, ph, mp, sp, lb, al):
    return _fused_attention_and_vfe_gradients_block_diag(
        mu_q=mu, sigma_q=sig, mu_p=mp, sigma_p=sp, phi=ph, generators=generators,
        alpha=al, lambda_belief=lb, lambda_softmax=0.0, kappa=kappa, eps=eps,
        irrep_dims=irrep_dims, compute_sigma_align_grad=True, use_rope=False, return_kl=True, alpha_div=ALPHA_DIV)
# self only
_,gmu0_self,_,_ = run_kernel_d(mu_q,sigma_q,phi,mu_p,sigma_p,0.0,1.0)
_,gmu1_self,_,_ = run_kernel_d(mu_t,sigma_t,phi_t,mu_p_t,sigma_p_t,0.0,1.0)
err_self = (gmu1_self - torch.einsum('kl,bnl->bnk',G,gmu0_self)).abs().max().item()
# direct only
_,gmu0_dir,_,_ = run_kernel_d(mu_q,sigma_q,phi,mu_p,sigma_p,10.0,0.0)
_,gmu1_dir,_,_ = run_kernel_d(mu_t,sigma_t,phi_t,mu_p_t,sigma_p_t,10.0,0.0)
err_dir = (gmu1_dir - torch.einsum('kl,bnl->bnk',G,gmu0_dir)).abs().max().item()
print(f"(C-debug) grad_mu self-term covar err={err_self:.3e}  direct(align)-term covar err={err_dir:.3e}")

# ---------------------------------------------------------------------------
# (C2) Non-permutation diagonal scaling g=diag(s). This BREAKS diagonal-sigma
#   covariance in general because Omega @ diag(sigma) @ Omega^T is not diagonal
#   and the kernel only keeps its diagonal. We quantify the residual to show the
#   diagonal-approx limit (this is the ALLOWED approximation per CLAUDE.md).
#   Use g = exp(D) diagonal, conjugate phi by g (block), scale mu by g.
# ---------------------------------------------------------------------------
s = torch.exp(0.3 * torch.randn(d))  # per-coordinate positive scaling (shared across heads for simplicity)
Dg = torch.diag(s)
Ginv = torch.diag(1.0 / s)
Gs = torch.zeros(K, K)
Gs[0:d, 0:d] = Dg
Gs[d:2*d, d:2*d] = Dg
mu_s = torch.einsum('kl,bnl->bnk', Gs, mu_q)
mu_p_s = mu_p  # keep prior fixed to isolate the alignment term? No—must transform prior too
mu_p_s = torch.einsum('kl,bnl->bnk', Gs, mu_p)
# diagonal sigma under diagonal g: sigma_k -> s_k^2 sigma_k (exact, stays diagonal!)
sig_s = torch.zeros_like(sigma_q)
sig_s[:, :, 0:d] = (s ** 2) * sigma_q[:, :, 0:d]
sig_s[:, :, d:2*d] = (s ** 2) * sigma_q[:, :, d:2*d]
sig_p_s = torch.zeros_like(sigma_p)
sig_p_s[:, :, 0:d] = (s ** 2) * sigma_p[:, :, 0:d]
sig_p_s[:, :, d:2*d] = (s ** 2) * sigma_p[:, :, d:2*d]
# conjugate phi by diagonal g: g E_ab g^{-1} = (s_a/s_b) E_ab -> scale coeff
phi_s = torch.zeros_like(phi)
scale_ab = (s.view(d, 1) / s.view(1, d)).reshape(d * d)
for h in range(n_heads):
    base = h * d * d
    phi_s[:, :, base:base + d * d] = phi[:, :, base:base + d * d] * scale_ab

# Need mu_p_s and sig_p_s passed in: rerun kernel with explicit prior override.
def run_kernel2(mu, sig, ph, mp, sp):
    return _fused_attention_and_vfe_gradients_block_diag(
        mu_q=mu, sigma_q=sig, mu_p=mp, sigma_p=sp,
        phi=ph, generators=generators,
        alpha=alpha, lambda_belief=lambda_belief, lambda_softmax=lambda_softmax,
        kappa=kappa, eps=eps, irrep_dims=irrep_dims,
        compute_sigma_align_grad=True, use_rope=False, return_kl=True, alpha_div=ALPHA_DIV,
    )

beta0b, gmu0b, gsig0b, kl0b = run_kernel2(mu_q, sigma_q, phi, mu_p, sigma_p)
beta2, gmu2, gsig2, kl2 = run_kernel2(mu_s, sig_s, phi_s, mu_p_s, sig_p_s)
errKL2 = (kl0b - kl2).abs().max().item()
errBeta2 = (beta0b - beta2).abs().max().item()
print(f"(C2) diagonal-scaling frame: max|dKL|={errKL2:.3e} max|dBeta|={errBeta2:.3e}")
print("     (~0 => diagonal-KL is exactly invariant under the MONOMIAL subgroup")
print("      (signed-perm x diagonal) that preserves diagonal covariance.)")

# ---------------------------------------------------------------------------
# (C3) GENERAL (non-monomial) per-head GL(10) frame g. This maps diagonal
#   covariance OFF the diagonal, so the diagonal-covariance kernel (which keeps
#   only diag(Omega Sigma Omega^T)) CANNOT be strictly invariant. Quantify the
#   break to confirm the documented diagonal-approx limitation.
#   We transform mu->Gg mu, prior likewise, and we must lift sigma to full and
#   take only its diagonal after the sandwich (the closest diagonal input).
# ---------------------------------------------------------------------------
Mh = 0.4 * torch.randn(d, d)
gg = torch.matrix_exp(Mh)          # general GL(10) element, det>0
gg_inv = torch.linalg.inv(gg)
Gg = torch.zeros(K, K)
Gg[0:d, 0:d] = gg
Gg[d:2*d, d:2*d] = gg
mu_g = torch.einsum('kl,bnl->bnk', Gg, mu_q)
mu_p_g = torch.einsum('kl,bnl->bnk', Gg, mu_p)
# diagonal sigma under general g: g diag(sigma) g^T is NOT diagonal. The kernel
# input must stay diagonal, so we feed diag(g diag(sigma) g^T) as the best
# diagonal proxy (this is the lossy step the diagonal approx forces).
def diag_sandwich_block(sig_blk, gmat):
    full = torch.einsum('kl,bnl,ml->bnkm', gmat, sig_blk, gmat)
    return torch.diagonal(full, dim1=-2, dim2=-1)
sig_g = torch.zeros_like(sigma_q)
sig_g[:, :, 0:d] = diag_sandwich_block(sigma_q[:, :, 0:d], gg)
sig_g[:, :, d:2*d] = diag_sandwich_block(sigma_q[:, :, d:2*d], gg)
sig_p_g = torch.zeros_like(sigma_p)
sig_p_g[:, :, 0:d] = diag_sandwich_block(sigma_p[:, :, 0:d], gg)
sig_p_g[:, :, d:2*d] = diag_sandwich_block(sigma_p[:, :, d:2*d], gg)
# conjugate phi by gg: phi' coeff s.t. exp(phi')[h] = gg exp(phi)[h] gg^{-1}.
# Conjugation of the algebra element: phi.G -> gg (phi.G) gg^{-1}. Build per
# (B,N) the 10x10 algebra element, conjugate, then project back to E_ab coeffs
# (the coeff of E_ab in a matrix M is just M[a,b]).
phi_g = torch.zeros_like(phi)
for h in range(n_heads):
    base = h * d * d
    coeff = phi[:, :, base:base + d * d].reshape(B, N, d, d)  # algebra element directly
    conj = torch.einsum('pa,xnab,qb->xnpq', gg, coeff, gg_inv)
    phi_g[:, :, base:base + d * d] = conj.reshape(B, N, d * d)

beta3, gmu3, gsig3, kl3 = run_kernel2(mu_g, sig_g, phi_g, mu_p_g, sig_p_g)
errKL3 = (kl0b - kl3).abs().max().item()
errBeta3 = (beta0b - beta3).abs().max().item()
print(f"(C3) GENERAL GL(10) frame: max|dKL|={errKL3:.3e} max|dBeta|={errBeta3:.3e}")
print("     (nonzero => documented diagonal-covariance approximation: strict")
print("      SE(K)/GL(K) covariance holds only for the monomial subgroup.)")

# ---------------------------------------------------------------------------
# (D) LayerNorm break: nn.LayerNorm(mu) vs a frame change. The live block applies
#     mu_normalized = nn.LayerNorm(mu) BEFORE the FFN. LayerNorm subtracts the
#     per-vector mean over K and divides by std over K -> not gauge-equivariant.
# ---------------------------------------------------------------------------
import torch.nn as nn
ln = nn.LayerNorm(K)
with torch.no_grad():
    ln.weight.fill_(1.0); ln.bias.zero_()
x = torch.randn(B, N, K)
# frame g = permutation G (orthogonal). LayerNorm commutes with permutation only.
gx = torch.einsum('kl,bnl->bnk', G, x)
err_ln_perm = (ln(gx) - torch.einsum('kl,bnl->bnk', G, ln(x))).abs().max().item()
# diagonal scaling g=Gs: LayerNorm should NOT commute
gsx = torch.einsum('kl,bnl->bnk', Gs, x)
err_ln_scale = (ln(gsx) - torch.einsum('kl,bnl->bnk', Gs, ln(x))).abs().max().item()
print(f"(D) LayerNorm vs permutation frame: max err = {err_ln_perm:.3e}  (commutes ~ ok)")
print(f"(D) LayerNorm vs diagonal-scaling frame: max err = {err_ln_scale:.3e}  (does NOT commute -> break)")

# ---------------------------------------------------------------------------
# (E) RoPE interaction. Re-run the fused kernel with use_rope=True under the
#   monomial frames and check invariance. RoPE rotates position-pair dims with
#   position-dependent angles, so it commutes with a DIAGONAL frame only if the
#   scaling is constant within each (2k,2k+1) pair; a generic per-coordinate
#   permutation/scaling does NOT commute with RoPE.
# ---------------------------------------------------------------------------
def run_rope(mu, sig, ph, mp, sp):
    return _fused_attention_and_vfe_gradients_block_diag(
        mu_q=mu, sigma_q=sig, mu_p=mp, sigma_p=sp, phi=ph, generators=generators,
        alpha=alpha, lambda_belief=lambda_belief, lambda_softmax=0.0, kappa=kappa,
        eps=eps, irrep_dims=irrep_dims, compute_sigma_align_grad=True,
        use_rope=True, rope_base=100.0, return_kl=True, alpha_div=ALPHA_DIV)
kb0,_,_,klr0 = run_rope(mu_q, sigma_q, phi, mu_p, sigma_p)
# diagonal frame where scaling is CONSTANT within each rope pair (commutes w/ RoPE)
s_pair = torch.exp(0.3*torch.randn(d//2)).repeat_interleave(2)  # same scale on (2k,2k+1)
Dp = torch.diag(s_pair); Gp = torch.zeros(K,K); Gp[0:d,0:d]=Dp; Gp[d:2*d,d:2*d]=Dp
mu_rp = torch.einsum('kl,bnl->bnk', Gp, mu_q); mu_p_rp = torch.einsum('kl,bnl->bnk', Gp, mu_p)
sig_rp = torch.zeros_like(sigma_q); sig_rp[:,:,0:d]=(s_pair**2)*sigma_q[:,:,0:d]; sig_rp[:,:,d:2*d]=(s_pair**2)*sigma_q[:,:,d:2*d]
sigp_rp = torch.zeros_like(sigma_p); sigp_rp[:,:,0:d]=(s_pair**2)*sigma_p[:,:,0:d]; sigp_rp[:,:,d:2*d]=(s_pair**2)*sigma_p[:,:,d:2*d]
phi_rp = torch.zeros_like(phi); scl=(s_pair.view(d,1)/s_pair.view(1,d)).reshape(d*d)
for h in range(n_heads):
    base=h*d*d; phi_rp[:,:,base:base+d*d]=phi[:,:,base:base+d*d]*scl
kb1,_,_,klr1 = run_rope(mu_rp, sig_rp, phi_rp, mu_p_rp, sigp_rp)
err_rope_pair = (klr0-klr1).abs().max().item()
print(f"(E) RoPE + pair-constant diagonal frame: max|dKL|={err_rope_pair:.3e}  (RoPE commutes -> invariant)")
# now a generic permutation that scrambles rope pairs (should break under RoPE)
kbg0,_,_,klrg0 = run_rope(mu_q, sigma_q, phi, mu_p, sigma_p)
kbg1,_,_,klrg1 = run_rope(mu_t, sigma_t, phi_t, mu_p_t, sigma_p_t)  # permutation from (C)
err_rope_perm = (klrg0-klrg1).abs().max().item()
print(f"(E) RoPE + coordinate-permutation frame: max|dKL|={err_rope_perm:.3e}  (perm scrambles RoPE pairs -> breaks)")

# ---------------------------------------------------------------------------
# (G) Sign-flip frame S=diag(-1,+1,...) per head. Pins down kernel=monomial vs
#     block=trivial: the KERNEL should be invariant (s^2=1 so sigma unchanged,
#     Delta mu^2 invariant) -> signs ARE in the monomial subgroup the kernel
#     preserves. nn.LayerNorm should BREAK under a sign flip (it subtracts the
#     global mean, which is not sign-equivariant) -> signs do NOT survive LN.
# ---------------------------------------------------------------------------
sgn = torch.ones(d); sgn[0] = -1.0; sgn[3] = -1.0  # flip a couple coords per head
S = torch.zeros(K, K)
S[0:d, 0:d] = torch.diag(sgn); S[d:2*d, d:2*d] = torch.diag(sgn)
mu_S = torch.einsum('kl,bnl->bnk', S, mu_q); mu_p_S = torch.einsum('kl,bnl->bnk', S, mu_p)
# diagonal sigma under sign flip: unchanged (s^2 = 1)
sig_S = sigma_q.clone(); sig_p_S = sigma_p.clone()
# conjugate phi by S (diagonal +-1): S E_ab S = sgn_a*sgn_b E_ab
sgn_ab = (sgn.view(d, 1) * sgn.view(1, d)).reshape(d * d)
phi_S = torch.zeros_like(phi)
for h in range(n_heads):
    base = h * d * d
    phi_S[:, :, base:base + d * d] = phi[:, :, base:base + d * d] * sgn_ab
_, _, _, kl_S0 = run_kernel2(mu_q, sigma_q, phi, mu_p, sigma_p)
_, _, _, kl_S1 = run_kernel2(mu_S, sig_S, phi_S, mu_p_S, sig_p_S)
err_sign_kernel = (kl_S0 - kl_S1).abs().max().item()
print(f"(G) kernel + sign-flip frame: max|dKL|={err_sign_kernel:.3e}  (invariant -> signs are IN the kernel's monomial subgroup)")
xS = torch.randn(B, N, K)
SxS = torch.einsum('kl,bnl->bnk', S, xS)
err_sign_ln = (ln(SxS) - torch.einsum('kl,bnl->bnk', S, ln(xS))).abs().max().item()
print(f"(G) LayerNorm + sign-flip frame: max err={err_sign_ln:.3e}  (break -> signs do NOT survive LayerNorm)")

# ---------------------------------------------------------------------------
# (F) det(Omega) bound under phi_trace_clamp=0.75, phi_project_slk=False.
#   det(Omega_h) = exp(tr(phi_i.G_h) - tr(phi_j.G_h)); with |s_h|<=0.75 per
#   token, the pairwise det is in [exp(-1.5), exp(1.5)].
# ---------------------------------------------------------------------------
from transformer.core.vfe_utils import _apply_det_control
phi_big = 3.0 * torch.randn(B, N, n_gen)  # deliberately large to stress the clamp
phi_clamped = _apply_det_control(phi_big, generators, is_glk=True,
                                 project_slk=False, trace_clamp=0.75,
                                 irrep_dims=irrep_dims)
pc = fused_block_matrix_exp_pairs(phi_clamped, generators, irrep_dims)
dets = []
for h in range(n_heads):
    ep, _ = pc[h]
    dets.append(torch.linalg.det(ep))  # det(exp(phi_h)) = exp(s_h)
dets = torch.stack(dets)  # (n_heads, B, N)
log_dets = torch.log(dets.clamp(min=1e-12))
print(f"(F) clamp=0.75: per-token log det(exp(phi_h)) in [{log_dets.min().item():.3f}, {log_dets.max().item():.3f}] "
      f"(should be within +-0.75); pairwise det(Omega) in "
      f"[{math.exp(2*log_dets.min().item()):.3f}, {math.exp(2*log_dets.max().item()):.3f}]")
