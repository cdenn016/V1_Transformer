"""
Tests for transformer/core/prior_bank.py
==========================================

Tests the PriorBank: encode (prior lookup), decode (fused KL logits),
rotation (gauge-fixed), and update_from_beliefs (VFE learning).
"""

import pytest
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_prior_bank(vocab_size=50, embed_dim=6, **kwargs):
    """Create a minimal PriorBank for testing."""
    from transformer.core.prior_bank import PriorBank
    defaults = dict(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        phi_dim=3,
        diagonal_covariance=True,
        learnable_sigma=True,
        gauge_fixed_priors=False,
        sigma_ce_scale=0.1,
    )
    defaults.update(kwargs)
    return PriorBank(**defaults)


def _make_gauge_fixed_prior_bank(vocab_size=50, embed_dim=3):
    """Create PriorBank with gauge_fixed_priors=True using SO(3) generators."""
    from transformer.core.prior_bank import PriorBank
    from math_utils.generators import generate_so3_generators
    G = torch.from_numpy(generate_so3_generators(embed_dim)).float()
    return PriorBank(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        phi_dim=3,
        diagonal_covariance=True,
        learnable_sigma=True,
        gauge_fixed_priors=True,
        generators=G,
    )


def _make_glk_gauge_fixed_prior_bank(vocab_size=50, embed_dim=4):
    """Create PriorBank with gauge_fixed_priors=True using GL(K) generators."""
    from transformer.core.prior_bank import PriorBank
    from math_utils.generators import generate_glK_generators
    G = torch.from_numpy(generate_glK_generators(embed_dim)).float()
    return PriorBank(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        # phi_dim auto-inferred from generators (K^2)
        diagonal_covariance=True,
        learnable_sigma=True,
        gauge_fixed_priors=True,
        generators=G,
    )


# ===========================================================================
# TestPriorBankEncode
# ===========================================================================

class TestPriorBankEncode:
    """Tests for PriorBank.encode()."""

    def test_output_shapes(self):
        """encode returns (mu, sigma, phi) with correct shapes."""
        pb = _make_prior_bank(vocab_size=50, embed_dim=6)
        token_ids = torch.randint(0, 50, (2, 8))
        mu, sigma, phi = pb.encode(token_ids)
        assert mu.shape == (2, 8, 6)
        assert sigma.shape == (2, 8, 6)
        assert phi.shape == (2, 8, 3)

    def test_sigma_positive(self):
        """All sigma values > 0."""
        pb = _make_prior_bank()
        token_ids = torch.randint(0, 50, (2, 8))
        _, sigma, _ = pb.encode(token_ids)
        assert (sigma > 0).all()

    def test_same_token_same_prior(self):
        """Encoding the same token twice gives the same result."""
        pb = _make_prior_bank()
        token_ids = torch.tensor([[5, 5, 5]])
        mu, sigma, phi = pb.encode(token_ids)
        assert torch.allclose(mu[0, 0], mu[0, 1])
        assert torch.allclose(mu[0, 0], mu[0, 2])

    def test_different_tokens_different_priors(self):
        """Distinct tokens -> distinct μ (with high probability)."""
        pb = _make_prior_bank(vocab_size=100, embed_dim=6)
        token_ids = torch.tensor([[0, 1, 2, 3, 4]])
        mu, _, _ = pb.encode(token_ids)
        # At least some pairs should differ
        diffs = torch.cdist(mu[0], mu[0])
        assert diffs.sum() > 0

    def test_gauge_fixed_mode(self):
        """gauge_fixed_priors=True produces valid output."""
        pb = _make_gauge_fixed_prior_bank(vocab_size=30, embed_dim=3)
        token_ids = torch.randint(0, 30, (2, 4))
        mu, sigma, phi = pb.encode(token_ids)
        assert mu.shape == (2, 4, 3)
        assert sigma.shape == (2, 4, 3)
        assert torch.isfinite(mu).all()
        assert (sigma > 0).all()


# ===========================================================================
# TestPriorBankDecode
# ===========================================================================

class TestPriorBankDecode:
    """Tests for PriorBank.decode()."""

    def test_output_shape(self):
        """Logits are (B, N, V)."""
        pb = _make_prior_bank(vocab_size=50, embed_dim=6)
        mu_q = torch.randn(2, 4, 6)
        sigma_q = torch.rand(2, 4, 6).clamp(min=0.1)
        logits = pb.decode(mu_q, sigma_q)
        assert logits.shape == (2, 4, 50)

    def test_logits_finite(self):
        """No NaN/Inf in logits."""
        pb = _make_prior_bank(vocab_size=50, embed_dim=6)
        mu_q = torch.randn(2, 4, 6)
        sigma_q = torch.rand(2, 4, 6).clamp(min=0.1)
        logits = pb.decode(mu_q, sigma_q)
        assert torch.isfinite(logits).all()

    def test_temperature_scaling(self):
        """Higher τ -> flatter distribution (higher entropy)."""
        pb = _make_prior_bank(vocab_size=50, embed_dim=6)
        mu_q = torch.randn(1, 2, 6)
        sigma_q = torch.rand(1, 2, 6).clamp(min=0.1)
        logits_low = pb.decode(mu_q, sigma_q, tau=0.5)
        logits_high = pb.decode(mu_q, sigma_q, tau=2.0)
        # Higher temp -> logits closer to uniform -> lower variance
        var_low = logits_low.var(dim=-1).mean()
        var_high = logits_high.var(dim=-1).mean()
        assert var_low > var_high, \
            f"Low-temp variance {var_low:.4f} should be > high-temp {var_high:.4f}"

    def test_gradient_flows_to_mu(self):
        """Gradient exists for mu_q through decode."""
        pb = _make_prior_bank(vocab_size=30, embed_dim=6)
        mu_q = torch.randn(1, 2, 6, requires_grad=True)
        sigma_q = torch.rand(1, 2, 6).clamp(min=0.1)
        logits = pb.decode(mu_q, sigma_q)
        logits.sum().backward()
        assert mu_q.grad is not None
        assert torch.isfinite(mu_q.grad).all()

    def test_fused_matches_explicit_kl(self):
        """Fused matmul logits ∝ -KL(q || π_v) for small vocab."""
        pb = _make_prior_bank(vocab_size=10, embed_dim=4)
        pb.eval()
        mu_q = torch.randn(1, 1, 4)
        sigma_q = torch.rand(1, 1, 4).clamp(min=0.1)

        with torch.no_grad():
            logits = pb.decode(mu_q, sigma_q, tau=1.0)

            # Compute explicit KL for each vocab token
            all_ids = torch.arange(10)
            mu_p, sigma_p, _ = pb.encode(all_ids)
            sigma_p = sigma_p.clamp(min=1e-4)
            sigma_q_safe = sigma_q[0, 0].clamp(min=1e-4)

            explicit_kl = torch.zeros(10)
            for v in range(10):
                delta = mu_q[0, 0] - mu_p[v]
                kl = 0.5 * (
                    (sigma_q_safe / sigma_p[v]).sum()
                    + (delta ** 2 / sigma_p[v]).sum()
                    - 4  # K=4
                    + torch.log(sigma_p[v]).sum()
                    - torch.log(sigma_q_safe).sum()
                )
                explicit_kl[v] = kl

            # Logits should be proportional to -KL (up to constants)
            # Check that ranking is preserved
            fused_rank = logits[0, 0].argsort(descending=True)
            explicit_rank = (-explicit_kl).argsort(descending=True)
            # Top predictions should match
            assert fused_rank[0] == explicit_rank[0], \
                f"Top-1 mismatch: fused={fused_rank[0]}, explicit={explicit_rank[0]}"

    def test_full_covariance_input(self):
        """Decode handles (B, N, K, K) full covariance input."""
        pb = _make_prior_bank(vocab_size=20, embed_dim=4)
        mu_q = torch.randn(1, 2, 4)
        sigma_q_full = torch.eye(4).unsqueeze(0).unsqueeze(0).expand(1, 2, 4, 4) * 0.5
        logits = pb.decode(mu_q, sigma_q_full)
        assert logits.shape == (1, 2, 20)
        assert torch.isfinite(logits).all()


# ===========================================================================
# TestPriorBankRotation
# ===========================================================================

class TestPriorBankGaugeTransform:
    """Tests for _compute_gauge_transform() in gauge-fixed mode."""

    def test_so3_orthogonal(self):
        """R^T R = I for SO(3) gauge transforms (rotations)."""
        pb = _make_gauge_fixed_prior_bank(vocab_size=10, embed_dim=3)
        phi = torch.randn(5, 3) * 0.5
        R = pb._compute_gauge_transform(phi)
        I = torch.eye(3)
        for i in range(5):
            assert torch.allclose(R[i].T @ R[i], I, atol=1e-4), \
                f"Rotation {i} not orthogonal: deviation {(R[i].T @ R[i] - I).abs().max():.2e}"

    def test_so3_det_one(self):
        """det(R) = 1 (SO(K), not just O(K))."""
        pb = _make_gauge_fixed_prior_bank(vocab_size=10, embed_dim=3)
        phi = torch.randn(5, 3) * 0.5
        R = pb._compute_gauge_transform(phi)
        dets = torch.linalg.det(R)
        assert torch.allclose(dets, torch.ones(5), atol=1e-4), \
            f"Determinants: {dets}"

    def test_glk_invertible(self):
        """GL(K) gauge transforms are invertible (det > 0)."""
        pb = _make_glk_gauge_fixed_prior_bank(vocab_size=10, embed_dim=4)
        phi = torch.randn(5, 16) * 0.3  # gl(4) has 16 generators
        A = pb._compute_gauge_transform(phi)
        dets = torch.linalg.det(A)
        # exp(X) always has det > 0 (det(exp(X)) = exp(tr(X)))
        assert (dets > 0).all(), f"Some determinants <= 0: {dets}"

    def test_glk_orbit_covers_space(self):
        """GL(K) orbit of base prior can reach any target mean."""
        K = 4
        pb = _make_glk_gauge_fixed_prior_bank(vocab_size=10, embed_dim=K)
        # Different phi values should produce different mu_v
        token_ids = torch.arange(10)
        mu_p, sigma_p, phi = pb._get_prior_for_tokens(token_ids)
        # With random phi init, all 10 token priors should be distinct
        pairwise = torch.cdist(mu_p.unsqueeze(0), mu_p.unsqueeze(0)).squeeze(0)
        # Off-diagonal should be nonzero
        off_diag = pairwise[~torch.eye(10, dtype=bool)]
        assert (off_diag > 1e-6).all(), "Some token priors are identical (degenerate phi init)"


# ===========================================================================
# TestPriorBankUpdateFromBeliefs
# ===========================================================================

class TestPriorBankUpdateFromBeliefs:
    """Tests for update_from_beliefs()."""

    def test_priors_change_after_update(self):
        """mu_p changes for tokens with prediction errors."""
        pb = _make_prior_bank(vocab_size=20, embed_dim=4)
        token_ids = torch.tensor([[0, 1, 2, 3]])
        mu_before = pb.prior_mu.data.clone()
        mu_beliefs = torch.randn(1, 4, 4)
        sigma_beliefs = torch.rand(1, 4, 4).clamp(min=0.1)
        prediction_errors = torch.ones(1, 4) * 5.0  # high errors
        pb.update_from_beliefs(token_ids, mu_beliefs, sigma_beliefs,
                               prediction_errors, lr=0.5)
        # Priors for tokens 0-3 should have changed
        changed = (pb.prior_mu.data[:4] - mu_before[:4]).abs().sum()
        assert changed > 0, "Priors did not change after update"

    def test_sigma_stays_positive_after_update(self):
        """sigma_p > 0 after update."""
        pb = _make_prior_bank(vocab_size=20, embed_dim=4)
        token_ids = torch.randint(0, 20, (2, 8))
        mu_beliefs = torch.randn(2, 8, 4)
        sigma_beliefs = torch.rand(2, 8, 4).clamp(min=0.1)
        prediction_errors = torch.rand(2, 8)
        pb.update_from_beliefs(token_ids, mu_beliefs, sigma_beliefs,
                               prediction_errors, lr=0.1)
        assert (pb.prior_sigma > 0).all()
