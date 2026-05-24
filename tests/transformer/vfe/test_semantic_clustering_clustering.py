# tests/transformer/vfe/test_semantic_clustering_clustering.py
import numpy as np
from transformer.vfe.semantic_clustering.clustering import cluster

def test_recovers_two_obvious_blobs():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 0.05, size=(20, 2))
    b = rng.normal(5, 0.05, size=(20, 2))
    X = np.vstack([a, b])
    D = np.linalg.norm(X[:, None] - X[None], axis=-1)
    labels = cluster(D, method="agglomerative", precomputed=True, k="auto")
    assert labels.shape == (40,)
    # the two blobs end up in different clusters
    assert labels[0] != labels[-1]

def test_tiny_n_single_cluster():
    D = np.zeros((3, 3))
    labels = cluster(D, precomputed=True, k="auto")
    assert labels.shape == (3,)
