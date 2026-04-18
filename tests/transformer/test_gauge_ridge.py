"""Tests for transformer/core/gauge_ridge.py

Verify that the covariant ridge transforms as Sigma under gauge h ∈ GL(K):
    ridge(h @ exp_phi) == h @ ridge(exp_phi) @ h.T

And contrast with the standard eps * I ridge, which does NOT transform covariantly.
"""
import pytest
import torch

from transformer.core.gauge_ridge import gauge_covariant_eye, make_ridge


def _rand_gl(K: int, seed: int = 0, scale: float = 0.3) -> torch.Tensor:
    """Random GL(K) element via matrix exponential of a random dense matrix."""
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(K, K, generator=g, dtype=torch.float64) * scale
    return torch.linalg.matrix_exp(X)


def _rand_spd(K: int, seed: int = 1) -> torch.Tensor:
    """Random SPD matrix with controlled conditioning."""
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(K, K, generator=g, dtype=torch.float64)
    return A @ A.T + torch.eye(K, dtype=torch.float64)


@pytest.mark.parametrize("K", [3, 8, 16])
def test_covariant_ridge_transforms_as_sandwich(K):
    """ridge(h @ g) == h @ ridge(g) @ h.T (exact, normalize=False)."""
    g = _rand_gl(K, seed=42)
    h = _rand_gl(K, seed=7)
    eps = 1e-4

    R_g = gauge_covariant_eye(g, eps, normalize=False)
    R_hg = gauge_covariant_eye(h @ g, eps, normalize=False)
    R_hg_expected = h @ R_g @ h.T

    torch.testing.assert_close(R_hg, R_hg_expected, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("K", [3, 8])
def test_sigma_plus_ridge_is_gauge_covariant(K):
    """(Σ + ε gg^T) under h transforms to h(Σ + ε gg^T)h^T."""
    g = _rand_gl(K, seed=1)
    h = _rand_gl(K, seed=2)
    Sigma = _rand_spd(K, seed=3)
    eps = 1e-3

    lhs_R = gauge_covariant_eye(g, eps, normalize=False)
    lhs = Sigma + lhs_R

    Sigma_gauged = h @ Sigma @ h.T
    rhs_R = gauge_covariant_eye(h @ g, eps, normalize=False)
    rhs = Sigma_gauged + rhs_R

    expected = h @ lhs @ h.T
    torch.testing.assert_close(rhs, expected, rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("K", [4, 8])
def test_eye_ridge_fails_gauge_covariance(K):
    """Standard ε I ridge is NOT covariant — this test documents the bug we fix."""
    h = _rand_gl(K, seed=5, scale=0.5)
    eps = 1e-3
    I = torch.eye(K, dtype=torch.float64)

    lhs = h @ (eps * I) @ h.T
    rhs = eps * I

    diff = (lhs - rhs).norm()
    assert diff > 1e-5, f"expected εI sandwich to differ from εI, got diff={diff}"


@pytest.mark.parametrize("K", [3, 8])
def test_ridge_reduces_to_eye_at_identity_frame(K):
    """When exp_phi = I, ridge(I) = eps * I exactly."""
    I = torch.eye(K, dtype=torch.float64)
    eps = 1e-3
    R = gauge_covariant_eye(I, eps, normalize=False)
    torch.testing.assert_close(R, eps * I, rtol=0, atol=1e-15)


@pytest.mark.parametrize("K", [4, 8])
def test_ridge_with_batch_shape(K):
    """Helper broadcasts over leading batch dims."""
    B, N = 2, 5
    g = torch.stack(
        [torch.stack([_rand_gl(K, seed=b * 100 + n) for n in range(N)]) for b in range(B)]
    )  # (B, N, K, K)
    eps = 1e-4
    R = gauge_covariant_eye(g, eps, normalize=False)
    assert R.shape == (B, N, K, K)

    h = _rand_gl(K, seed=99)
    R_gauged = gauge_covariant_eye(h @ g, eps, normalize=False)
    expected = h @ R @ h.T
    torch.testing.assert_close(R_gauged, expected, rtol=1e-9, atol=1e-11)


def test_make_ridge_fallback_to_eye():
    """make_ridge collapses to eps * I when exp_phi is None."""
    K = 4
    eps = 1e-3

    R_off = make_ridge(K, eps, exp_phi=None,
                       device=torch.device("cpu"), dtype=torch.float64)
    torch.testing.assert_close(R_off, eps * torch.eye(K, dtype=torch.float64), rtol=0, atol=1e-15)

    g = _rand_gl(K, seed=11)
    R_on = make_ridge(K, eps, exp_phi=g)
    torch.testing.assert_close(R_on, eps * (g @ g.T), rtol=1e-12, atol=1e-14)


def test_make_ridge_batch_shape_for_fallback():
    """Fallback path respects batch_shape for broadcasting."""
    K = 3
    eps = 1e-3
    R = make_ridge(K, eps, exp_phi=None,
                   device=torch.device("cpu"), dtype=torch.float32,
                   batch_shape=(2, 5))
    assert R.shape == (2, 5, K, K)
    torch.testing.assert_close(
        R[0, 0], eps * torch.eye(K, dtype=torch.float32), rtol=0, atol=1e-7
    )


@pytest.mark.parametrize("K", [4, 8])
def test_kl_invariance_under_gauge(K):
    """KL(q||p) with covariant ridge is gauge-invariant; with εI it drifts by O(ε).

    This test is the proof-of-value: it shows the new ridge preserves an
    invariance that the existing εI path breaks.
    """
    torch.manual_seed(0)
    Sigma_q = _rand_spd(K, seed=10)
    Sigma_p = _rand_spd(K, seed=11)
    mu_q = torch.randn(K, dtype=torch.float64)
    mu_p = torch.randn(K, dtype=torch.float64)
    g = _rand_gl(K, seed=12)   # local frame at this token
    h = _rand_gl(K, seed=13, scale=0.4)  # global gauge transformation
    eps = 1e-3

    def kl(mu_q, Sigma_q, mu_p, Sigma_p, R_q, R_p):
        Sq, Sp = Sigma_q + R_q, Sigma_p + R_p
        Sp_inv = torch.linalg.inv(Sp)
        d = mu_p - mu_q
        trace = (Sp_inv @ Sq).diagonal(dim1=-2, dim2=-1).sum(-1)
        mah = d @ Sp_inv @ d
        _, logdet_p = torch.linalg.slogdet(Sp)
        _, logdet_q = torch.linalg.slogdet(Sq)
        return 0.5 * (trace + mah - K + logdet_p - logdet_q)

    R_q_cov = gauge_covariant_eye(g, eps, normalize=False)
    R_p_cov = gauge_covariant_eye(g, eps, normalize=False)
    kl_pre = kl(mu_q, Sigma_q, mu_p, Sigma_p, R_q_cov, R_p_cov)

    mu_q_h = h @ mu_q
    mu_p_h = h @ mu_p
    Sq_h = h @ Sigma_q @ h.T
    Sp_h = h @ Sigma_p @ h.T
    g_h = h @ g
    R_q_cov_h = gauge_covariant_eye(g_h, eps, normalize=False)
    R_p_cov_h = gauge_covariant_eye(g_h, eps, normalize=False)
    kl_post = kl(mu_q_h, Sq_h, mu_p_h, Sp_h, R_q_cov_h, R_p_cov_h)

    assert abs(kl_pre.item() - kl_post.item()) < 1e-8, \
        f"covariant ridge should yield gauge-invariant KL; drift={kl_pre - kl_post}"

    I = torch.eye(K, dtype=torch.float64)
    kl_pre_eye = kl(mu_q, Sigma_q, mu_p, Sigma_p, eps * I, eps * I)
    kl_post_eye = kl(mu_q_h, Sq_h, mu_p_h, Sp_h, eps * I, eps * I)
    drift_eye = abs(kl_pre_eye.item() - kl_post_eye.item())
    assert drift_eye > 1e-7, \
        f"εI ridge expected to drift under gauge; drift={drift_eye}"
