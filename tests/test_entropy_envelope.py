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
    direct_form: bool = False,
) -> torch.Tensor:
    """Mimic the code's alignment_loss expression.

    direct_form=True replicates the post-2026-05-12 production path: when entropy
    is enabled, replace the product-rule split with the manuscript F functional
    scaled by lambda_align. lambda_soft is ignored.
    """
    kl = kl_matrix_all_pairs(mu, sigma)
    beta = softmax_attention(kl, kappa, K)
    tau_eff = kappa * math.sqrt(K)

    if direct_form and include_entropy:
        beta_safe = beta.clamp(min=1e-30)
        F_manuscript = (beta * kl).sum() + tau_eff * (beta_safe * beta_safe.log()).sum()
        return lambda_align * F_manuscript

    align_term = lambda_align * (beta.detach() * kl).sum()
    soft_term = lambda_soft * (beta * kl.detach()).sum()
    loss = align_term + soft_term

    if include_entropy:
        beta_for_h = beta if entropy_attached else beta.detach()
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

def test_entropy_envelope() -> None:
    torch.manual_seed(20260512)
    # Save and restore the global default dtype: this test needs float64
    # for envelope-theorem precision, but mutating the global default
    # leaked into every downstream test in the suite (~128 failures in
    # test_vfe_package.py that assume float32 default).
    _prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        _run_entropy_envelope_body()
    finally:
        torch.set_default_dtype(_prev_dtype)


def _run_entropy_envelope_body() -> None:
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
    # Hard assertion: production direct-form is λ_soft-independent and gives
    # λ_align × manuscript-envelope gradient for any λ_soft.
    print("Production direct-form assertion (λ_align=1, any λ_soft) tolerance 1e-10:")
    tol = 1e-10
    for lam_soft in (0.0, 0.5, 1.0, 7.3):
        g_mu, g_sigma, g_kappa = autograd_code_gradients(
            mu, sigma, kappa, K,
            lambda_align=1.0, lambda_soft=lam_soft,
            direct_form=True,
        )
        d_mu = (g_mu - env_mu).norm().item()
        d_sigma = (g_sigma - env_sigma).norm().item()
        d_kappa = (g_kappa - env_kappa).abs().item()
        print(f"  λ_soft={lam_soft}: ||Δμ||={d_mu:.2e}  ||ΔΣ||={d_sigma:.2e}  |Δκ|={d_kappa:.2e}")
        assert d_mu < tol and d_sigma < tol and d_kappa < tol, (
            f"DIRECT-FORM MISMATCH at λ_soft={lam_soft}: μ={d_mu:.2e}, Σ={d_sigma:.2e}, κ={d_kappa:.2e}"
        )

    # Also verify λ_align scaling: gradient should be λ_align × envelope.
    print("\nλ_align scaling assertion (direct-form should produce λ_align × envelope):")
    for lam_align in (0.5, 2.0, 10.0):
        g_mu, g_sigma, g_kappa = autograd_code_gradients(
            mu, sigma, kappa, K,
            lambda_align=lam_align, lambda_soft=0.0,
            direct_form=True,
        )
        scaled_env_mu = lam_align * env_mu
        scaled_env_sigma = lam_align * env_sigma
        scaled_env_kappa = lam_align * env_kappa
        d_mu = (g_mu - scaled_env_mu).norm().item()
        d_sigma = (g_sigma - scaled_env_sigma).norm().item()
        d_kappa = (g_kappa - scaled_env_kappa).abs().item()
        print(f"  λ_align={lam_align}: ||Δμ||={d_mu:.2e}  ||ΔΣ||={d_sigma:.2e}  |Δκ|={d_kappa:.2e}")
        assert d_mu < tol and d_sigma < tol and d_kappa < tol, (
            f"SCALING MISMATCH at λ_align={lam_align}"
        )

    print("\nPASS — direct-form production code is λ_soft-independent and λ_align-scaled.")


if __name__ == "__main__":
    test_entropy_envelope()
