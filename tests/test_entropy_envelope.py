"""Envelope-theorem unit test for the attention-entropy fix.

Verifies that at the softmax stationary point β* = softmax(-KL/(κ·√K)), the
(μ, Σ, κ) gradients of the manuscript free energy

    F = Σ_ij β_ij · KL(q_i || q_j) + τ_eff · Σ_ij β_ij · log(β_ij / π_ij)

with τ_eff = κ·√K, reduce to the envelope-theorem predictions:

    ∂F/∂μ    = Σ_j β_j · ∂KL_j/∂μ                (no implicit β-dependence)
    ∂F/∂κ    = √K · Σ_j β_j · log(β_j / π_j)

We then compare these analytical envelope predictions against the gradient
that the *code path* actually produces under various combinations of:
  (a) lambda_soft ∈ {0, 1}   — the (β · KL.detach).sum() product-rule term
  (b) entropy_attached ∈ {True, False}
        — whether the entropy term lets β flow through

The four combinations expose which (λ_soft, attach) configurations match the
manuscript envelope gradient at the softmax fixed point.

No gauge transport — Ω = I, single block, K = 4, two tokens. The test isolates
the entropy-temperature consistency question from the transport machinery.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch


# -----------------------------------------------------------------------------
# Toy KL between Gaussians (no gauge transport)
# -----------------------------------------------------------------------------

def kl_gaussian_pair(mu_i: torch.Tensor, sigma_i: torch.Tensor,
                     mu_j: torch.Tensor, sigma_j: torch.Tensor) -> torch.Tensor:
    """KL(N(μ_i, Σ_i) || N(μ_j, Σ_j)) for K-dim Gaussians (full covariance)."""
    K = mu_i.shape[-1]
    sigma_j_inv = torch.linalg.inv(sigma_j)
    diff = mu_j - mu_i
    quad = diff @ sigma_j_inv @ diff
    trace_term = torch.trace(sigma_j_inv @ sigma_i)
    logdet_term = torch.logdet(sigma_j) - torch.logdet(sigma_i)
    return 0.5 * (trace_term + quad - K + logdet_term)


def kl_matrix_all_pairs(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Compute the (N, N) KL matrix KL_ij = KL(q_i || q_j)."""
    N = mu.shape[0]
    kl = torch.zeros((N, N), dtype=mu.dtype)
    for i in range(N):
        for j in range(N):
            kl[i, j] = kl_gaussian_pair(mu[i], sigma[i], mu[j], sigma[j])
    return kl


def softmax_attention(kl: torch.Tensor, kappa: torch.Tensor, K: int) -> torch.Tensor:
    """β_ij = softmax_j(-KL_ij / (κ·√K))."""
    tau_eff = kappa * math.sqrt(K)
    return torch.softmax(-kl / tau_eff, dim=-1)


# -----------------------------------------------------------------------------
# Loss assembly under the four configurations
# -----------------------------------------------------------------------------

def assemble_code_loss(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    kappa: torch.Tensor,
    K: int,
    lambda_align: float = 1.0,
    lambda_soft: float = 1.0,
    include_entropy: bool = True,
    entropy_attached: bool = False,
) -> torch.Tensor:
    """Mimic the code's alignment_loss expression."""
    kl = kl_matrix_all_pairs(mu, sigma)
    beta = softmax_attention(kl, kappa, K)
    tau_eff = kappa * math.sqrt(K)

    # The product-rule split used in e_step.py / variational_ffn.py:
    align_term = lambda_align * (beta.detach() * kl).sum()
    soft_term = lambda_soft * (beta * kl.detach()).sum()
    loss = align_term + soft_term

    if include_entropy:
        if entropy_attached:
            # Attached β — autograd flows through softmax into μ, Σ, κ
            beta_for_h = beta
        else:
            # Detached β — envelope-theorem path
            beta_for_h = beta.detach()
        beta_safe = beta_for_h.clamp(min=1e-30)
        h_term = tau_eff * (beta_safe * beta_safe.log()).sum()
        loss = loss + h_term

    return loss


def manuscript_envelope_prediction(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    kappa: torch.Tensor,
    K: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (∂F/∂μ, ∂F/∂Σ, ∂F/∂κ) per envelope theorem at softmax β.

    At softmax β, the full F functional has gradient:
        ∂F/∂θ = Σ_ij β_ij · ∂KL_ij/∂θ                    for θ ∈ (μ, Σ)
        ∂F/∂κ = √K · Σ_ij β_ij · log(β_ij / π_ij)        (π = 1/N uniform)
    """
    N = mu.shape[0]
    # Recompute β fresh on a detached graph
    with torch.no_grad():
        kl = kl_matrix_all_pairs(mu, sigma)
        beta = softmax_attention(kl, kappa, K)

    # Manuscript: gradient = Σ β·∂KL/∂μ.  Compute by autograd on (β.detach · KL).sum()
    mu_g = mu.detach().clone().requires_grad_(True)
    sigma_g = sigma.detach().clone().requires_grad_(True)
    kl_g = kl_matrix_all_pairs(mu_g, sigma_g)
    envelope_loss = (beta.detach() * kl_g).sum()
    g_mu, g_sigma = torch.autograd.grad(envelope_loss, [mu_g, sigma_g])

    # κ gradient: envelope-theorem prediction = √K · Σ β·log β
    with torch.no_grad():
        beta_safe = beta.clamp(min=1e-30)
        envelope_kappa_grad = math.sqrt(K) * (beta_safe * beta_safe.log()).sum()
    return g_mu, g_sigma, envelope_kappa_grad


def autograd_code_gradients(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    kappa: torch.Tensor,
    K: int,
    **loss_kwargs,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute code-path gradients via autograd."""
    mu_g = mu.detach().clone().requires_grad_(True)
    sigma_g = sigma.detach().clone().requires_grad_(True)
    kappa_g = kappa.detach().clone().requires_grad_(True)
    loss = assemble_code_loss(mu_g, sigma_g, kappa_g, K, **loss_kwargs)
    g_mu, g_sigma, g_kappa = torch.autograd.grad(loss, [mu_g, sigma_g, kappa_g])
    return g_mu, g_sigma, g_kappa


# -----------------------------------------------------------------------------
# Test
# -----------------------------------------------------------------------------

def run_test() -> None:
    torch.manual_seed(20260512)
    torch.set_default_dtype(torch.float64)

    K = 4
    N = 2

    # Random Gaussian beliefs
    mu = torch.randn(N, K)
    # SPD Σ: A·A^T + ε·I
    A = torch.randn(N, K, K)
    sigma = torch.matmul(A, A.transpose(-2, -1)) + 0.1 * torch.eye(K).unsqueeze(0).expand(N, K, K)
    kappa = torch.tensor(0.7)

    # Manuscript envelope-theorem prediction at softmax β
    env_mu, env_sigma, env_kappa = manuscript_envelope_prediction(mu, sigma, kappa, K)

    print("=" * 78)
    print("Envelope-theorem unit test — entropy fix")
    print("=" * 78)
    print(f"\nManuscript envelope prediction at softmax β*:")
    print(f"  ||∂F/∂μ||_F           = {env_mu.norm().item():.6f}")
    print(f"  ||∂F/∂Σ||_F           = {env_sigma.norm().item():.6f}")
    print(f"  ∂F/∂κ                 = {env_kappa.item():.6f}")
    print()

    configs = [
        # PRODUCTION CODE (post fix B 2026-05-12) is λ_soft default × β.attached:
        ("λ_soft=1, β.attached (PROD)",  dict(lambda_soft=1.0, entropy_attached=True)),
        # Other valid manuscript-matching alternative:
        ("λ_soft=0, β.detach",           dict(lambda_soft=0.0, entropy_attached=False)),
        # Inconsistent crosses (one term provides a correction the other doesn't cancel):
        ("λ_soft=1, β.detach (pre-B)",   dict(lambda_soft=1.0, entropy_attached=False)),
        ("λ_soft=0, β.attached",         dict(lambda_soft=0.0, entropy_attached=True)),
        # Pre-entropy-fix baselines (manuscript line 1261's "entropy-suppressed surrogate"):
        ("λ_soft=1, NO entropy",         dict(lambda_soft=1.0, include_entropy=False)),
        ("λ_soft=0, NO entropy",         dict(lambda_soft=0.0, include_entropy=False)),
    ]

    header = f"{'config':<35} {'||Δ μ||':<12} {'||Δ Σ||':<12} {'|Δ κ|':<12}"
    print(header)
    print("-" * len(header))
    for name, kwargs in configs:
        g_mu, g_sigma, g_kappa = autograd_code_gradients(mu, sigma, kappa, K, **kwargs)
        d_mu = (g_mu - env_mu).norm().item()
        d_sigma = (g_sigma - env_sigma).norm().item()
        d_kappa = (g_kappa - env_kappa).abs().item()
        match_mark = "  MATCH" if max(d_mu, d_sigma, d_kappa) < 1e-8 else "  MISMATCH"
        print(f"{name:<35} {d_mu:<12.2e} {d_sigma:<12.2e} {d_kappa:<12.2e}{match_mark}")

    print()
    # Hard assertion: production config must match envelope theorem
    prod_g_mu, prod_g_sigma, prod_g_kappa = autograd_code_gradients(
        mu, sigma, kappa, K, lambda_soft=1.0, entropy_attached=True,
    )
    tol = 1e-10
    d_mu = (prod_g_mu - env_mu).norm().item()
    d_sigma = (prod_g_sigma - env_sigma).norm().item()
    d_kappa = (prod_g_kappa - env_kappa).abs().item()
    print(f"Production-config assertion (λ_soft=1, β.attached) tolerance {tol:.0e}:")
    print(f"  ||Δ μ|| = {d_mu:.2e}    ||Δ Σ|| = {d_sigma:.2e}    |Δ κ| = {d_kappa:.2e}")
    assert d_mu < tol and d_sigma < tol and d_kappa < tol, (
        f"PRODUCTION CONFIG MISMATCH: μ={d_mu:.2e}, Σ={d_sigma:.2e}, κ={d_kappa:.2e}"
    )
    print("  PASS — production code matches manuscript envelope-theorem gradient.")


if __name__ == "__main__":
    run_test()
