"""Tests for T8: Gauge Frame Spectral Analysis."""

import numpy as np
import pytest
from scipy.linalg import expm

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from gauge_frame_spectral_analysis import (
    compute_per_head_M,
    compute_spectral_features,
    analyze_determinants,
    analyze_spectral_entropy_trend,
    cluster_heads_by_spectrum,
    generate_synthetic_gauge_weights,
    run_synthetic_validation,
    SAVE_DIR,
)
import torch


# ---------------------------------------------------------------------------
# compute_per_head_M
# ---------------------------------------------------------------------------
class TestComputePerHeadM:
    def test_output_shapes(self):
        hidden_dim, num_heads, head_dim = 64, 4, 16
        W_Q = torch.randn(hidden_dim, hidden_dim)
        W_K = torch.randn(hidden_dim, hidden_dim)
        heads = compute_per_head_M(W_Q, W_K, num_heads, head_dim)
        assert len(heads) == num_heads
        for M in heads:
            assert M.shape == (head_dim, head_dim)

    def test_identity_projection(self):
        """If W_Q = W_K = I, then M^h = I_{d_h}."""
        hidden_dim, num_heads, head_dim = 32, 4, 8
        W_Q = torch.eye(hidden_dim)
        W_K = torch.eye(hidden_dim)
        heads = compute_per_head_M(W_Q, W_K, num_heads, head_dim)
        for M in heads:
            np.testing.assert_allclose(M, np.eye(head_dim), atol=1e-6)

    def test_reproduces_bilinear_form(self):
        """M^h should satisfy h_i^T (W_Q^h)^T W_K^h h_j = h_i^T M^h h_j
        ... but actually M^h = W_Q^h @ (W_K^h)^T, so
        Q_i^T K_j = (W_Q^h h_i)^T (W_K^h h_j) = h_i^T M^{h,bilinear} h_j
        where M^{h,bilinear} = (W_Q^h)^T @ W_K^h.

        Our M^h = W_Q^h @ (W_K^h)^T is the head-space kernel:
        M^h_{ab} gives the (a,b) entry of the attention kernel in head space.
        """
        hidden_dim, num_heads, head_dim = 32, 2, 16
        W_Q = torch.randn(hidden_dim, hidden_dim)
        W_K = torch.randn(hidden_dim, hidden_dim)
        heads = compute_per_head_M(W_Q, W_K, num_heads, head_dim)

        # Verify: M^h = W_Q^h @ (W_K^h)^T
        for h in range(num_heads):
            s, e = h * head_dim, (h + 1) * head_dim
            Wq_h = W_Q[s:e, :].numpy()
            Wk_h = W_K[s:e, :].numpy()
            expected = Wq_h @ Wk_h.T
            np.testing.assert_allclose(heads[h], expected, atol=1e-5)


# ---------------------------------------------------------------------------
# compute_spectral_features
# ---------------------------------------------------------------------------
class TestSpectralFeatures:
    def test_identity_matrix(self):
        M = np.eye(8)
        feat = compute_spectral_features(M)
        # Identity has all eigenvalues = 1 -> max entropy
        assert feat["norm_spectral_entropy"] == pytest.approx(1.0, abs=1e-6)
        assert feat["effective_rank"] == pytest.approx(8.0, abs=0.1)
        assert feat["asymmetry"] == pytest.approx(0.0, abs=1e-10)
        assert feat["frac_positive_real"] == pytest.approx(1.0)

    def test_rank_one_matrix(self):
        v = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=float)
        M = np.outer(v, v)
        feat = compute_spectral_features(M)
        assert feat["effective_rank"] == pytest.approx(1.0, abs=0.1)

    def test_symmetric_has_zero_asymmetry(self):
        M = np.random.randn(10, 10)
        M = (M + M.T) / 2
        feat = compute_spectral_features(M)
        assert feat["asymmetry"] < 1e-10

    def test_gl_plus_matrix(self):
        """exp(A) always has positive determinant."""
        A = np.random.randn(16, 16) * 0.5
        M = expm(A)
        feat = compute_spectral_features(M)
        sign, _ = np.linalg.slogdet(M)
        assert sign > 0


# ---------------------------------------------------------------------------
# analyze_determinants
# ---------------------------------------------------------------------------
class TestDeterminants:
    def test_all_gl_plus(self):
        """Matrices from exp(A) should all have det > 0."""
        rng = np.random.RandomState(0)
        layers_M = []
        for _ in range(3):
            heads = [expm(rng.randn(8, 8) * 0.3) for _ in range(4)]
            layers_M.append(heads)

        res = analyze_determinants({"test": layers_M})["test"]
        assert res["n_positive"] == 12
        assert res["n_negative"] == 0
        assert res["gl_plus_holds"] is True

    def test_mixed_determinants(self):
        rng = np.random.RandomState(0)
        layers_M = []
        for _ in range(2):
            heads = []
            for h in range(4):
                M = expm(rng.randn(8, 8) * 0.3)
                if h % 2 == 1:
                    M[0, :] = -M[0, :]  # flip det sign
                heads.append(M)
            layers_M.append(heads)

        res = analyze_determinants({"test": layers_M})["test"]
        assert res["n_positive"] == 4
        assert res["n_negative"] == 4
        assert res["gl_plus_holds"] is False


# ---------------------------------------------------------------------------
# analyze_spectral_entropy_trend
# ---------------------------------------------------------------------------
class TestSpectralEntropyTrend:
    def test_monotone_decreasing_entropy(self):
        """Construct spectral features with decreasing entropy."""
        spectral = []
        for l in range(6):
            feats = []
            for _ in range(4):
                # Create matrix with decreasing effective rank
                d = 16
                s = np.exp(-np.arange(d) * (0.1 + 0.05 * l))
                U, _ = np.linalg.qr(np.random.randn(d, d))
                M = U @ np.diag(s) @ U.T
                feats.append(compute_spectral_features(M))
            spectral.append(feats)

        rg = analyze_spectral_entropy_trend(spectral)
        assert rg["entropy_regression"]["decreasing"]
        assert rg["entropy_regression"]["slope"] < 0


# ---------------------------------------------------------------------------
# Synthetic validation
# ---------------------------------------------------------------------------
class TestSyntheticValidation:
    def test_gl_plus_all_positive(self):
        data = generate_synthetic_gauge_weights(
            n_layers=4, n_heads=4, head_dim=16, hidden_dim=64
        )
        layers_M = data["synthetic_GL+"]
        res = analyze_determinants({"test": layers_M})["test"]
        assert res["gl_plus_holds"] is True

    def test_mixed_has_negatives(self):
        data = generate_synthetic_gauge_weights(
            n_layers=4, n_heads=4, head_dim=16, hidden_dim=64
        )
        layers_M = data["synthetic_mixed"]
        res = analyze_determinants({"test": layers_M})["test"]
        assert res["n_negative"] > 0

    def test_random_approx_half(self):
        data = generate_synthetic_gauge_weights(
            n_layers=6, n_heads=12, head_dim=16, hidden_dim=192
        )
        layers_M = data["synthetic_random"]
        res = analyze_determinants({"test": layers_M})["test"]
        # For random matrices, ~50% should have positive det
        assert 0.3 < res["fraction_positive"] < 0.7

    def test_clustering_returns_correct_shape(self):
        data = generate_synthetic_gauge_weights(
            n_layers=4, n_heads=4, head_dim=16, hidden_dim=64
        )
        layers_M = data["synthetic_GL+"]
        spectral = []
        for heads_M in layers_M:
            spectral.append([compute_spectral_features(M) for M in heads_M])

        labels, feat_mat, Z = cluster_heads_by_spectrum(spectral, n_clusters=3)
        assert len(labels) == 4 * 4  # n_layers * n_heads
        assert feat_mat.shape[0] == 16
        assert len(np.unique(labels)) <= 3
