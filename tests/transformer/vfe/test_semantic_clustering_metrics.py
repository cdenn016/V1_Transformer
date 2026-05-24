import numpy as np, torch
from transformer.vfe.semantic_clustering import metrics as M

def test_silhouette_in_range():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0,0.1,(15,2)), rng.normal(4,0.1,(15,2))])
    D = np.linalg.norm(X[:,None]-X[None], axis=-1)
    labels = np.array([0]*15 + [1]*15)
    out = M.common_metrics(D, labels, precomputed=True)
    assert -1.0 <= out["silhouette"] <= 1.0
    assert out["n_clusters"] == 2

def test_effective_rank_bounds():
    sigma = torch.rand(10, 6) + 0.1
    out = M.sigma_metrics(sigma, diagonal=True)
    assert 1.0 <= out["effective_rank_mean"] <= 6.0

def test_phi_energy_partition_sums_to_one():
    phi = torch.randn(8, 8)          # 2 heads x gl(2) = 8 gens
    out = M.phi_metrics(phi, irrep_dims=[2, 2])
    assert np.isclose(out["energy_frac_diagonal"] + out["energy_frac_offdiag"], 1.0, atol=1e-6)
