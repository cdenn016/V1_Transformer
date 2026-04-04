"""
Tests for non-flat gauge transport and holonomy computation.

Covers GaugeConnection (bilinear and MLP variants that produce
connection coefficients delta_ij of shape (B, N, N, n_gen)),
holonomy computation around triples of positions, and the
synthetic gauge language dataset with controlled path-dependence.
"""

import pytest
import torch
import numpy as np
from transformer.core.connection import GaugeConnection
from transformer.analysis.holonomy import (
    compute_holonomy,
    holonomy_penalty_loss,
    holonomy_statistics,
)


# =============================================================================
# GaugeConnection Tests
# =============================================================================

class TestGaugeConnection:
    """Tests for GaugeConnection: learned connection coefficients delta_ij.

    GaugeConnection maps pairs of belief means (mu_i, mu_j) to Lie-algebra-valued
    connection coefficients delta of shape (B, N, N, n_gen). Supports bilinear
    and MLP architectures, both zero-initialized (flat at init).
    """

    @pytest.fixture
    def bilinear_conn(self):
        return GaugeConnection(d_head=8, n_gen=3, connection_type='bilinear')

    @pytest.fixture
    def mlp_conn(self):
        return GaugeConnection(d_head=8, n_gen=3, connection_type='mlp', hidden_dim=16)

    def test_bilinear_output_shape(self, bilinear_conn):
        mu_i = torch.randn(2, 10, 8)  # (B=2, N=10, d_head=8)
        mu_j = torch.randn(2, 10, 8)
        delta = bilinear_conn(mu_i, mu_j)
        assert delta.shape == (2, 10, 10, 3)  # (B, N, N, n_gen)

    def test_mlp_output_shape(self, mlp_conn):
        mu_i = torch.randn(2, 10, 8)
        mu_j = torch.randn(2, 10, 8)
        delta = mlp_conn(mu_i, mu_j)
        assert delta.shape == (2, 10, 10, 3)

    def test_zero_init_bilinear(self, bilinear_conn):
        """At initialization, W=0 so δ_ij should be zero (flat)."""
        mu_i = torch.randn(2, 10, 8)
        mu_j = torch.randn(2, 10, 8)
        delta = bilinear_conn(mu_i, mu_j)
        assert torch.allclose(delta, torch.zeros_like(delta), atol=1e-7)

    def test_zero_init_mlp(self, mlp_conn):
        """At initialization, output layer is zero so δ_ij should be zero."""
        mu_i = torch.randn(2, 10, 8)
        mu_j = torch.randn(2, 10, 8)
        delta = mlp_conn(mu_i, mu_j)
        assert torch.allclose(delta, torch.zeros_like(delta), atol=1e-7)

    def test_antisymmetric_bilinear(self):
        """With antisymmetrize=True, δ_ij = -δ_ji."""
        conn = GaugeConnection(d_head=8, n_gen=3, connection_type='bilinear',
                               antisymmetrize=True)
        # Set non-zero weights
        conn.W.data = torch.randn(3, 8, 8)
        mu = torch.randn(2, 10, 8)
        delta = conn(mu, mu)
        # Check δ_ij = -δ_ji (rtol for numerical stability with large values)
        assert torch.allclose(delta, -delta.transpose(1, 2), atol=1e-5, rtol=1e-5)

    def test_gradient_flow(self, bilinear_conn):
        """Gradients flow through the connection."""
        mu_i = torch.randn(2, 10, 8, requires_grad=True)
        mu_j = torch.randn(2, 10, 8)
        # Set non-zero W to get non-zero gradients
        bilinear_conn.W.data = torch.randn(3, 8, 8) * 0.1
        delta = bilinear_conn(mu_i, mu_j)
        loss = delta.sum()
        loss.backward()
        assert mu_i.grad is not None
        assert bilinear_conn.W.grad is not None


# =============================================================================
# Holonomy Computation Tests
# =============================================================================

class TestHolonomy:
    """Tests for holonomy computation around position triples.

    Holonomy H_{ijk} = exp(delta_ij) exp(delta_jk) exp(delta_ki) measures
    curvature of the gauge connection. For flat transport H = I.
    """

    def test_identity_exp_delta_gives_zero_holonomy(self):
        """exp(δ_ij) = I for all edges -> holonomy = 0."""
        B, N, K = 2, 10, 3
        # Identity transport on all edges
        exp_delta = torch.eye(K).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, N, N, K, K).clone()
        C, norms, triples = compute_holonomy(exp_delta, sample_size=100)
        assert torch.allclose(norms, torch.zeros_like(norms), atol=1e-6)

    def test_random_exp_delta_gives_nonzero_holonomy(self):
        """Random non-flat transport -> non-trivial holonomy."""
        B, N, K = 2, 10, 3
        # Random transport (not satisfying cocycle condition)
        exp_delta = torch.eye(K).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, N, N, K, K).clone()
        exp_delta += torch.randn(B, N, N, K, K) * 0.1
        _, norms, _ = compute_holonomy(exp_delta, sample_size=100)
        assert norms.mean() > 0.01  # Should be non-trivial

    def test_holonomy_shape(self):
        """Check output shapes."""
        B, N, K = 2, 10, 3
        exp_delta = torch.eye(K).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, N, N, K, K).clone()
        C, norms, triples = compute_holonomy(exp_delta, sample_size=50)
        assert C.shape == (B, 50, K, K)
        assert norms.shape == (B, 50)
        assert triples.shape == (50, 3)

    def test_specific_triples(self):
        """Test with specific triple indices."""
        B, N, K = 1, 5, 2
        exp_delta = torch.eye(K).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, N, N, K, K).clone()
        triples = torch.tensor([[0, 1, 2], [1, 3, 4]])
        C, norms, returned_triples = compute_holonomy(exp_delta, triples=triples)
        assert C.shape == (1, 2, K, K)
        assert torch.equal(returned_triples, triples)

    def test_penalty_loss_zero_for_flat(self):
        """Holonomy penalty should be ~0 for identity transport."""
        B, N, K = 2, 10, 3
        exp_delta = torch.eye(K).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, N, N, K, K).clone()
        loss = holonomy_penalty_loss(exp_delta, sample_size=100)
        assert loss.item() < 1e-10

    def test_penalty_loss_positive_for_nonflat(self):
        """Holonomy penalty should be > 0 for random transport."""
        B, N, K = 2, 10, 3
        exp_delta = torch.eye(K).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, N, N, K, K).clone()
        exp_delta += torch.randn(B, N, N, K, K) * 0.1
        loss = holonomy_penalty_loss(exp_delta, sample_size=100)
        assert loss.item() > 0

    def test_statistics_keys(self):
        """Holonomy statistics returns expected keys."""
        B, N, K = 2, 10, 3
        exp_delta = torch.eye(K).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, N, N, K, K).clone()
        stats = holonomy_statistics(exp_delta, sample_size=50)
        expected_keys = {'mean', 'max', 'std', 'median', 'frac_gt_0.01', 'frac_gt_0.1', 'frac_gt_1.0'}
        assert set(stats.keys()) == expected_keys


# =============================================================================
# Synthetic Language Tests
# =============================================================================

class TestSyntheticLanguage:
    """Tests for the synthetic gauge language with controlled holonomy.

    SyntheticGaugeLanguage generates sequences where token-to-token transport
    has tunable path-dependence controlled by epsilon. At epsilon=0 transport
    is flat (path-independent); at epsilon>0 holonomy is non-trivial.
    """

    def test_flat_language_is_path_independent(self):
        """At ε=0, transport depends only on endpoints."""
        from transformer.data.synthetic_gauge import SyntheticGaugeLanguage

        lang = SyntheticGaugeLanguage(vocab_size=20, K=3, epsilon=0.0, seq_len=8, seed=42)

        # Same endpoints, different intermediate tokens
        rng = np.random.RandomState(123)
        first, last = 5, 12
        T_values = []
        for _ in range(10):
            middle = rng.randint(0, 20, size=6)
            seq = np.concatenate([[first], middle, [last]])
            T = lang.compute_transport(seq)
            T_values.append(T)

        # All should be the same (path-independent)
        for T in T_values[1:]:
            assert np.allclose(T_values[0], T, atol=1e-6), \
                "Flat language should be path-independent!"

    def test_nonflat_language_is_path_dependent(self):
        """At ε>0, different paths give different transport."""
        from transformer.data.synthetic_gauge import SyntheticGaugeLanguage

        lang = SyntheticGaugeLanguage(vocab_size=20, K=3, epsilon=1.0, seq_len=8, seed=42)
        path_dep = lang.measure_path_dependence(n_samples=500)
        assert path_dep > 0.1, f"Non-flat language should show path-dependence, got {path_dep}"

    def test_dataset_shapes(self):
        """SyntheticGaugeDataset returns correct shapes."""
        from transformer.data.synthetic_gauge import SyntheticGaugeLanguage, SyntheticGaugeDataset

        lang = SyntheticGaugeLanguage(vocab_size=20, K=3, epsilon=0.0, seq_len=8, n_classes=4)
        dataset = SyntheticGaugeDataset(lang, n_samples=100)

        sample = dataset[0]
        assert sample['input_ids'].shape == (8 + 1,)  # seq_len + SEP
        assert sample['target_ids'].shape == (8 + 1,)  # shifted
        assert 0 <= sample['label'] < 4

    def test_dataloader_creation(self):
        """create_synthetic_dataloaders returns working dataloaders."""
        from transformer.data.synthetic_gauge import create_synthetic_dataloaders

        train_loader, val_loader, vocab_size = create_synthetic_dataloaders(
            epsilon=0.0, vocab_size=20, K=3, n_classes=4,
            seq_len=8, n_train=100, n_val=20, batch_size=16,
        )
        batch = next(iter(train_loader))
        assert batch['input_ids'].shape[0] == 16
        assert vocab_size == 20 + 1 + 4  # vocab + SEP + labels
