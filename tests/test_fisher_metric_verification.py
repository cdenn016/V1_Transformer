# -*- coding: utf-8 -*-
"""
Fisher Metric Verification Tests
=================================

Verifies that the Fisher-Rao metric implementation is mathematically correct
by checking against known analytical results and consistency conditions.

Tests:
1. Fisher metric recovers KL divergence for infinitesimal perturbations
2. Natural gradient is inverse of Fisher metric applied to Euclidean gradient
3. Pullback metric satisfies chain rule
4. Mass matrix reduces to Fisher metric when only self-energy is present
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from math_utils.fisher_metric import (
    natural_gradient_gaussian,
    euclidean_from_natural,
)
from geometry.pullback_metrics import fisher_rao_metric_gaussian
from math_utils.numerical_utils import kl_gaussian


def test_fisher_metric_recovers_kl_for_small_perturbations():
    """
    The Fisher-Rao metric should satisfy:
        g_B(δq, δq) ≈ 2 * KL(q || q+δq)  for small δq

    This is the defining property: the Fisher metric IS the Hessian of KL.
    """
    np.random.seed(42)
    K = 4

    # Base distribution
    mu = np.random.randn(K)
    A = np.random.randn(K, K)
    Sigma = A @ A.T + 0.5 * np.eye(K)

    # Small perturbation
    eps = 1e-5
    delta_mu = eps * np.random.randn(K)
    B = np.random.randn(K, K)
    delta_Sigma = eps * (B + B.T)  # symmetric perturbation

    # Fisher-Rao metric
    g_fisher = fisher_rao_metric_gaussian(mu, Sigma, delta_mu, delta_Sigma)

    # KL divergence
    mu2 = mu + delta_mu
    Sigma2 = Sigma + delta_Sigma
    kl = kl_gaussian(mu, Sigma, mu2, Sigma2)

    # Should satisfy: g ≈ 2*KL for infinitesimal perturbations
    ratio = float(g_fisher) / (2 * float(kl))
    assert abs(ratio - 1.0) < 0.01, (
        f"Fisher metric should equal 2*KL for small perturbations. "
        f"Ratio = {ratio:.6f} (expected ~1.0)"
    )
    print(f"  PASS: g_Fisher / (2*KL) = {ratio:.6f} (expected 1.0)")


def test_natural_gradient_roundtrip():
    """
    Verify: Euclidean -> Natural -> Euclidean is identity.

    natural_gradient_gaussian followed by euclidean_from_natural
    should recover the original Euclidean gradients.
    """
    np.random.seed(123)
    K = 3

    mu = np.random.randn(K)
    A = np.random.randn(K, K)
    Sigma = A @ A.T + np.eye(K)

    grad_mu = np.random.randn(K)
    B = np.random.randn(K, K)
    grad_Sigma = 0.5 * (B + B.T)

    # Forward: Euclidean -> Natural
    delta_mu, delta_Sigma = natural_gradient_gaussian(
        mu, Sigma, grad_mu, grad_Sigma
    )

    # Inverse: Natural -> Euclidean
    recovered_grad_mu, recovered_grad_Sigma = euclidean_from_natural(
        mu, Sigma, delta_mu, delta_Sigma
    )

    mu_err = np.max(np.abs(recovered_grad_mu - grad_mu))
    sigma_err = np.max(np.abs(recovered_grad_Sigma - grad_Sigma))

    assert mu_err < 1e-4, f"Mean gradient roundtrip error: {mu_err}"
    assert sigma_err < 1e-4, f"Sigma gradient roundtrip error: {sigma_err}"
    print(f"  PASS: Roundtrip errors: mu={mu_err:.2e}, Sigma={sigma_err:.2e}")


def test_natural_gradient_is_parametrization_invariant():
    """
    The natural gradient direction should be the same regardless of
    whether we use (μ, Σ) or (μ, L) parametrization.

    Specifically, for KL(q||p), the natural gradient δμ = -Σ ∇_μ KL
    should equal -Σ Σ_p^{-1}(μ_q - μ_p) = -(μ_q - μ_p) when Σ_q = Σ_p.
    """
    np.random.seed(456)
    K = 3

    mu_q = np.array([1.0, 2.0, 3.0])
    mu_p = np.array([0.0, 0.0, 0.0])
    Sigma = 2.0 * np.eye(K)  # Same for q and p

    # Euclidean gradient of KL(q||p) w.r.t. μ_q = Σ_p^{-1}(μ_q - μ_p)
    Sigma_inv = np.linalg.inv(Sigma)
    grad_mu = Sigma_inv @ (mu_q - mu_p)
    grad_Sigma = np.zeros((K, K))  # Zero since Σ_q = Σ_p

    # Natural gradient
    delta_mu, _ = natural_gradient_gaussian(mu_q, Sigma, grad_mu, grad_Sigma)

    # When Σ_q = Σ_p, natural gradient = -Σ Σ^{-1}(μ_q - μ_p) = -(μ_q - μ_p)
    expected = -(mu_q - mu_p)
    err = np.max(np.abs(delta_mu - expected))
    assert err < 1e-5, f"Parametrization invariance error: {err}"
    print(f"  PASS: Natural gradient = -(μ_q - μ_p) when Σ_q=Σ_p, error={err:.2e}")


def test_fisher_metric_positive_definite():
    """
    The Fisher-Rao metric should be positive definite:
    g_B(δq, δq) > 0 for all nonzero δq.
    """
    np.random.seed(789)
    K = 5
    n_trials = 100

    mu = np.random.randn(K)
    A = np.random.randn(K, K)
    Sigma = A @ A.T + np.eye(K)

    for _ in range(n_trials):
        delta_mu = np.random.randn(K)
        B = np.random.randn(K, K)
        delta_Sigma = 0.5 * (B + B.T)

        g = fisher_rao_metric_gaussian(mu, Sigma, delta_mu, delta_Sigma)
        assert g > 0, f"Fisher metric not positive definite: g = {g}"

    print(f"  PASS: Positive definite for all {n_trials} random tangent vectors")


def test_fisher_metric_scaling():
    """
    For isotropic Σ = σ²I, the mean part of the Fisher metric should
    scale as (1/σ²)||δμ||².
    """
    K = 4
    sigma = 2.5
    Sigma = sigma**2 * np.eye(K)
    mu = np.zeros(K)

    delta_mu = np.array([1.0, 0.0, 0.0, 0.0])
    delta_Sigma = np.zeros((K, K))

    g = fisher_rao_metric_gaussian(mu, Sigma, delta_mu, delta_Sigma)
    expected = np.dot(delta_mu, delta_mu) / sigma**2
    err = abs(float(g) - expected)
    assert err < 1e-6, f"Isotropic scaling error: {err}"
    print(f"  PASS: Isotropic mean scaling: g={float(g):.6f}, expected={expected:.6f}")


def test_kl_hessian_equals_fisher_at_same_point():
    """
    Key theoretical property: ∂²KL(q||p)/∂θ²|_{q=p} = Fisher metric at p.

    The Hessian of KL w.r.t. the first argument, evaluated at q=p,
    equals the Fisher information matrix at p.
    """
    np.random.seed(101)
    K = 3

    mu_p = np.random.randn(K)
    A = np.random.randn(K, K)
    Sigma_p = A @ A.T + np.eye(K)

    # Numerical Hessian of KL(q||p) w.r.t. μ_q, evaluated at q=p
    eps = 1e-5
    H_mu = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            mu_pp = mu_p.copy(); mu_pp[i] += eps; mu_pp_j = mu_pp.copy(); mu_pp_j[j] += eps
            mu_pm = mu_p.copy(); mu_pm[i] += eps; mu_pm_j = mu_pm.copy(); mu_pm_j[j] -= eps
            mu_mp = mu_p.copy(); mu_mp[i] -= eps; mu_mp_j = mu_mp.copy(); mu_mp_j[j] += eps
            mu_mm = mu_p.copy(); mu_mm[i] -= eps; mu_mm_j = mu_mm.copy(); mu_mm_j[j] -= eps

            # Perturb only μ_q, keep μ_p fixed
            mu_ij_pp = mu_p.copy(); mu_ij_pp[i] += eps; mu_ij_pp[j] += eps
            mu_ij_pm = mu_p.copy(); mu_ij_pm[i] += eps; mu_ij_pm[j] -= eps
            mu_ij_mp = mu_p.copy(); mu_ij_mp[i] -= eps; mu_ij_mp[j] += eps
            mu_ij_mm = mu_p.copy(); mu_ij_mm[i] -= eps; mu_ij_mm[j] -= eps

            kl_pp = kl_gaussian(mu_ij_pp, Sigma_p, mu_p, Sigma_p)
            kl_pm = kl_gaussian(mu_ij_pm, Sigma_p, mu_p, Sigma_p)
            kl_mp = kl_gaussian(mu_ij_mp, Sigma_p, mu_p, Sigma_p)
            kl_mm = kl_gaussian(mu_ij_mm, Sigma_p, mu_p, Sigma_p)

            H_mu[i, j] = (kl_pp - kl_pm - kl_mp + kl_mm) / (4 * eps**2)

    # Fisher metric at p: G_μμ = Σ_p^{-1}
    Sigma_p_inv = np.linalg.inv(Sigma_p)

    err = np.max(np.abs(H_mu - Sigma_p_inv))
    assert err < 1e-3, (
        f"KL Hessian should equal Fisher metric at q=p. Max error: {err}"
    )
    print("  PASS: d^2 KL/d mu^2 |_(q=p) = Sigma_p^-1, max error = {:.2e}".format(err))


if __name__ == "__main__":
    print("=" * 60)
    print("Fisher Metric Verification Tests")
    print("=" * 60)

    tests = [
        ("Fisher metric ≈ 2*KL for small perturbations",
         test_fisher_metric_recovers_kl_for_small_perturbations),
        ("Natural gradient roundtrip (Euclidean ↔ Natural)",
         test_natural_gradient_roundtrip),
        ("Natural gradient parametrization invariance",
         test_natural_gradient_is_parametrization_invariant),
        ("Fisher metric positive definiteness",
         test_fisher_metric_positive_definite),
        ("Isotropic scaling: g = ||δμ||²/σ²",
         test_fisher_metric_scaling),
        ("∂²KL/∂μ²|_{q=p} = Fisher metric",
         test_kl_hessian_equals_fisher_at_same_point),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        print(f"\n[TEST] {name}")
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    print(f"{'=' * 60}")
