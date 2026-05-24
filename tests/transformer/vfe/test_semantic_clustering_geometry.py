# tests/transformer/vfe/test_semantic_clustering_geometry.py
import numpy as np
import torch
import pytest
from transformer.vfe.semantic_clustering import geometry as geo


def _props(D, n):
    assert D.shape == (n, n)
    assert np.allclose(D, D.T, atol=1e-6)          # symmetric
    assert np.allclose(np.diag(D), 0.0, atol=1e-6) # zero diagonal
    assert (D >= -1e-9).all()                       # non-negative


def test_mu_euclidean_props_and_value():
    mu = torch.tensor([[0.0, 0.0], [3.0, 4.0], [0.0, 1.0]])
    D = geo.mu_distances(mu, metric="euclidean")
    _props(D, 3)
    assert np.isclose(D[0, 1], 5.0)


def test_mu_mahalanobis_identity_cov_equals_euclidean():
    mu = torch.randn(6, 4)
    sigma = torch.ones(6, 4)  # diagonal, unit
    De = geo.mu_distances(mu, metric="euclidean")
    Dm = geo.mu_distances(mu, sigma=sigma, metric="mahalanobis", diagonal=True)
    assert np.allclose(De, Dm, atol=1e-5)


def test_sigma_bhattacharyya_zero_for_identical():
    mu = torch.randn(5, 3)
    sigma = torch.rand(5, 3) + 0.5
    D = geo.sigma_distances(sigma, mu=mu, metric="bhattacharyya", diagonal=True)
    _props(D, 5)


def test_sigma_logeuclidean_diag_matches_full_when_full_is_diagonal():
    n, K = 4, 3
    diag = torch.rand(n, K) + 0.5
    full = torch.stack([torch.diag(diag[i]) for i in range(n)])
    Dd = geo.sigma_distances(diag, metric="logeuclidean", diagonal=True)
    Df = geo.sigma_distances(full, metric="logeuclidean", diagonal=False)
    assert np.allclose(Dd, Df, atol=1e-4)


def test_sigma_bhattacharyya_diag_matches_full_when_full_is_diagonal():
    n, K = 4, 3
    mu = torch.randn(n, K)
    diag = torch.rand(n, K) + 0.5
    full = torch.stack([torch.diag(diag[i]) for i in range(n)])
    Dd = geo.sigma_distances(diag, mu=mu, metric="bhattacharyya", diagonal=True)
    Df = geo.sigma_distances(full, mu=mu, metric="bhattacharyya", diagonal=False)
    _props(Dd, n)
    assert np.allclose(Dd, Df, atol=1e-5)


def test_mu_mahalanobis_diag_matches_full_when_full_is_diagonal():
    n, K = 5, 3
    mu = torch.randn(n, K)
    diag = torch.rand(n, K) + 0.5
    full = torch.stack([torch.diag(diag[i]) for i in range(n)])
    Dd = geo.mu_distances(mu, sigma=diag, metric="mahalanobis", diagonal=True)
    Df = geo.mu_distances(mu, sigma=full, metric="mahalanobis", diagonal=False)
    assert np.allclose(Dd, Df, atol=1e-5)


def test_phi_vector_zero_distance_identical_rows():
    phi = torch.randn(5, 12)
    phi[2] = phi[0]
    D = geo.phi_vector_distances(phi)
    _props(D, 5)
    assert np.isclose(D[0, 2], 0.0, atol=1e-5)


def test_omega_geodesic_equals_quadrature_of_per_head():
    # 2 heads of dim 2, full gl(2) generators per head (4 each) -> n_gen=8
    torch.manual_seed(0)
    d = 2
    eye = np.eye(d)
    basis = [np.zeros((d, d)) for _ in range(d * d)]
    for idx in range(d * d):
        basis[idx].flat[idx] = 1.0
    # block-diagonal generator bank (8, 4, 4)
    K = 2 * d
    G = np.zeros((8, K, K))
    for h in range(2):
        for c in range(4):
            G[h * 4 + c, h * d:(h + 1) * d, h * d:(h + 1) * d] = basis[c]
    G = torch.tensor(G, dtype=torch.float64)
    phi = torch.randn(3, 8, dtype=torch.float64) * 0.1
    D = geo.omega_geodesic_distances(phi, generators=G, irrep_dims=[d, d])
    _props(D, 3)
    # identical phi -> zero
    phi2 = phi.clone(); phi2[1] = phi2[0]
    D2 = geo.omega_geodesic_distances(phi2, generators=G, irrep_dims=[d, d])
    assert np.isclose(D2[0, 1], 0.0, atol=1e-6)
    # the total distance IS the quadrature of independent per-head geodesics
    from scipy.linalg import expm, logm
    def head_geo(k, l, h):
        A_k = sum(phi[k, h*4 + c].item() * G[h*4 + c, h*d:(h+1)*d, h*d:(h+1)*d].numpy() for c in range(4))
        A_l = sum(phi[l, h*4 + c].item() * G[h*4 + c, h*d:(h+1)*d, h*d:(h+1)*d].numpy() for c in range(4))
        Ok, Ol = expm(A_k), expm(A_l)
        return np.linalg.norm(logm(np.linalg.inv(Ok) @ Ol).real, "fro")
    expected_01 = np.sqrt(head_geo(0, 1, 0)**2 + head_geo(0, 1, 1)**2)
    assert np.isclose(D[0, 1], expected_01, atol=1e-5)
