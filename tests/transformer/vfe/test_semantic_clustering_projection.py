# tests/transformer/vfe/test_semantic_clustering_projection.py
import numpy as np
from transformer.vfe.semantic_clustering.projection import project

def test_project_precomputed_returns_2d():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(15, 5))
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    Y = project(D, method="umap", n_components=2, precomputed=True)
    assert Y.shape == (15, 2)
    assert np.isfinite(Y).all()

def test_project_tiny_n_safe():
    D = np.zeros((2, 2))
    Y = project(D, method="umap", n_components=2, precomputed=True)
    assert Y.shape == (2, 2)
