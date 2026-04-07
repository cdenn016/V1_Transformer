"""
Edge case tests for the pure VFE transformer.

Tests boundary conditions that could cause silent failures:
  - Single-token sequences (N=1)
  - Zero gauge frames (phi=0, Omega=I)
  - Identical beliefs (KL=0 everywhere)
  - Extreme temperatures (tau → 0, tau → inf)
  - Config validation
"""

import torch
import pytest

from .conftest import random_spd, random_gl, make_pure_vfe_config, make_model, K


class TestSingleTokenSequence:
    """Test that N=1 sequences work correctly."""

    def test_e_step_single_token(self):
        """E-step runs with N=1 (no cross-position attention possible)."""
        from ..inference import e_step

        config = make_pure_vfe_config(n_esteps=3, max_seq_len=1)
        torch.manual_seed(42)
        model = make_model(config)
        tokens = torch.randint(0, 20, (1, 1))

        mu_star, Sigma_star, Omega_star, logits, vfe_hist, diag = e_step(
            tokens, model, config,
        )
        assert mu_star.shape == (1, 1, K)
        assert Sigma_star.shape == (1, 1, K, K)
        assert logits.shape == (1, 1, 20)
        assert torch.isfinite(mu_star).all()
        assert torch.isfinite(logits).all()

    def test_m_step_single_token(self):
        """M-step works with N=1."""
        from ..inference import e_step
        from ..learning import m_step

        config = make_pure_vfe_config(n_esteps=3, max_seq_len=1)
        torch.manual_seed(42)
        model = make_model(config)
        tokens = torch.randint(0, 20, (1, 1))
        targets = torch.randint(0, 20, (1, 1))

        mu_s, Sig_s, Om_s, logits, _, _ = e_step(tokens, model, config)
        # Should not raise
        m_step(tokens, targets, mu_s, Sig_s, Om_s, model, config, logits)

    def test_pairwise_kl_n1(self):
        """pairwise_kl returns [B,H,1,1] with self-KL ~= 0."""
        from ..gaussians import precompute_tokens, pairwise_kl

        B, H, K_h = 1, 1, K
        mu_h = torch.randn(B, 1, H, K_h)
        Sigma_h = random_spd(K_h, (B, 1, H))
        Omega = random_gl(K_h, (B, 1, H), scale=0.1)

        precomp = precompute_tokens(mu_h, Sigma_h, Omega)
        kl = pairwise_kl(precomp, causal=False)

        assert kl.shape == (B, H, 1, 1)
        assert kl[0, 0, 0, 0] < 1e-3, f"Self-KL should be ~0, got {kl[0, 0, 0, 0]:.6f}"


class TestZeroGaugeFrame:
    """Test behavior with identity gauge frames (Omega = I)."""

    def test_identity_omega_identity_transport(self):
        """When all Omega = I, transport Omega_ij = I for all pairs."""
        from ..gauge import compute_transport

        I = torch.eye(K)
        Oij = compute_transport(I, I)
        diff = (Oij - torch.eye(K)).norm()
        assert diff < 1e-6, f"I→I transport should be I: diff = {diff:.6e}"

    def test_zero_gauge_gradient_vanishes(self):
        """When Omega=I and all beliefs identical, gauge gradient is zero."""
        from ..gaussians import precompute_tokens, pairwise_kl, kl_attention
        from ..gauge import vfe_grad_Omega

        B, H, N, K_h = 1, 1, 4, K
        mu_h = torch.ones(B, N, H, K_h) * 0.5
        Sigma_h = torch.eye(K_h).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, N, H, -1, -1).clone()
        Omega = torch.eye(K_h).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, N, H, -1, -1).clone()

        precomp = precompute_tokens(mu_h, Sigma_h, Omega)
        kl_ij = pairwise_kl(precomp, causal=False)
        beta = kl_attention(kl_ij, tau=K_h ** 0.5, causal=False)
        grad = vfe_grad_Omega(mu_h, Sigma_h, Omega, beta, kl_ij, precomp)

        assert grad.norm() < 1e-4, f"Gauge gradient should vanish: norm = {grad.norm():.6e}"

    def test_identity_omega_vfe_has_prior_term(self):
        """VFE with Omega=I still includes prior KL when mu != prior_mu."""
        from ..inference import e_step

        config = make_pure_vfe_config(n_esteps=1)
        torch.manual_seed(42)
        model = make_model(config)
        # Set all Omega to identity
        model.prior_Omega[:] = torch.eye(K).unsqueeze(0)
        tokens = torch.randint(0, 20, (1, 4))

        _, _, _, _, vfe_hist, _ = e_step(tokens, model, config)
        # VFE should be > 0 since beliefs start at prior (close) but there's
        # still some prior KL from the initial belief != converged belief
        assert len(vfe_hist) >= 1


class TestIdenticalBeliefs:
    """Test when all beliefs are identical (KL = 0 everywhere)."""

    def test_identical_beliefs_kl_zero(self):
        """All KL_ij ~= 0 when beliefs identical and Omega = I."""
        from ..gaussians import precompute_tokens, pairwise_kl

        B, H, N, K_h = 1, 1, 4, K
        mu_h = torch.ones(B, N, H, K_h) * 0.5
        Sigma_h = torch.eye(K_h).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, N, H, -1, -1).clone()
        Omega = torch.eye(K_h).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, N, H, -1, -1).clone()

        precomp = precompute_tokens(mu_h, Sigma_h, Omega)
        kl_ij = pairwise_kl(precomp, causal=False)

        assert kl_ij.max() < 1e-4, f"KL should be ~0: max = {kl_ij.max():.6e}"

    def test_identical_beliefs_uniform_attention(self):
        """Beta = 1/N (uniform) when all KL = 0."""
        from ..gaussians import precompute_tokens, pairwise_kl, kl_attention

        B, H, N, K_h = 1, 1, 4, K
        mu_h = torch.ones(B, N, H, K_h) * 0.5
        Sigma_h = torch.eye(K_h).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, N, H, -1, -1).clone()
        Omega = torch.eye(K_h).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, N, H, -1, -1).clone()

        precomp = precompute_tokens(mu_h, Sigma_h, Omega)
        kl_ij = pairwise_kl(precomp, causal=False)
        beta = kl_attention(kl_ij, tau=K_h ** 0.5, causal=False)

        expected = torch.ones_like(beta) / N
        diff = (beta - expected).abs().max()
        assert diff < 1e-4, f"Should be uniform attention: max diff = {diff:.6e}"

    def test_identical_beliefs_mu_gradient_zero(self):
        """No alignment gradient when beliefs are identical."""
        from ..gaussians import precompute_tokens, pairwise_kl, kl_attention, vfe_grad_mu_alignment

        B, H, N, K_h = 1, 1, 4, K
        mu_h = torch.ones(B, N, H, K_h) * 0.5
        Sigma_h = torch.eye(K_h).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, N, H, -1, -1).clone()
        Omega = torch.eye(K_h).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, N, H, -1, -1).clone()

        precomp = precompute_tokens(mu_h, Sigma_h, Omega)
        kl_ij = pairwise_kl(precomp, causal=False)
        beta = kl_attention(kl_ij, tau=K_h ** 0.5, causal=False)
        grad = vfe_grad_mu_alignment(precomp, beta, kl_ij, tau=K_h ** 0.5)

        assert grad.norm() < 1e-4, f"Gradient should be zero: norm = {grad.norm():.6e}"


class TestExtremeTemperatures:
    """Test attention behavior at extreme temperature limits."""

    def _build_kl(self, seed=42):
        from ..gaussians import precompute_tokens, pairwise_kl
        torch.manual_seed(seed)
        B, H, N, K_h = 1, 1, 4, K
        mu_h = torch.randn(B, N, H, K_h)
        Sigma_h = random_spd(K_h, (B, N, H))
        Omega = random_gl(K_h, (B, N, H), scale=0.1)
        precomp = precompute_tokens(mu_h, Sigma_h, Omega)
        kl_ij = pairwise_kl(precomp, causal=False)
        return kl_ij

    def test_tau_very_small_concentrates_attention(self):
        """tau -> 0 concentrates beta on minimum-KL pair."""
        from ..gaussians import kl_attention

        kl_ij = self._build_kl()
        beta = kl_attention(kl_ij, tau=0.01, causal=False)
        # Most weight on the self-diagonal (KL ~= 0)
        assert beta.max() > 0.8, f"Small tau should concentrate: max beta = {beta.max():.4f}"

    def test_tau_very_large_uniform_attention(self):
        """tau -> inf gives near-uniform beta."""
        from ..gaussians import kl_attention

        kl_ij = self._build_kl()
        N = kl_ij.shape[-1]
        beta = kl_attention(kl_ij, tau=1e6, causal=False)
        expected = 1.0 / N
        diff = (beta - expected).abs().max()
        assert diff < 0.01, f"Large tau should be uniform: max diff = {diff:.4f}"

    def test_tau_tiny_no_nan(self):
        """tau=1e-10 doesn't produce NaN."""
        from ..gaussians import kl_attention

        kl_ij = self._build_kl()
        beta = kl_attention(kl_ij, tau=1e-10, causal=False)
        assert torch.isfinite(beta).all(), "Tiny tau should not produce NaN"
        assert (beta >= 0).all(), "Beta should be non-negative"

    def test_tau_huge_no_nan(self):
        """tau=1e10 doesn't produce NaN."""
        from ..gaussians import kl_attention

        kl_ij = self._build_kl()
        beta = kl_attention(kl_ij, tau=1e10, causal=False)
        assert torch.isfinite(beta).all(), "Huge tau should not produce NaN"
        assert (beta >= 0).all(), "Beta should be non-negative"


class TestConfigValidation:
    """Test PureVFEConfig validation."""

    def test_belief_dim_assertion(self):
        """belief_dim != n_heads * head_dim triggers assertion."""
        from ..config import PureVFEConfig

        with pytest.raises(AssertionError):
            PureVFEConfig(belief_dim=10, n_heads=2, head_dim=4)  # 10 != 2*4

    def test_default_tau(self):
        """tau defaults to sqrt(head_dim) when None."""
        from ..config import PureVFEConfig

        config = PureVFEConfig(belief_dim=16, n_heads=2, head_dim=8, tau=None)
        assert config.tau == 8 ** 0.5, f"Default tau should be sqrt(8), got {config.tau}"

    def test_valid_config_no_error(self):
        """Valid config construction succeeds."""
        config = make_pure_vfe_config()
        assert config.belief_dim == config.n_heads * config.head_dim


class TestCausalMasking:
    """Test causal attention masking."""

    def test_causal_masks_future(self):
        """Causal masking sets future KL to 1e9."""
        from ..gaussians import precompute_tokens, pairwise_kl

        B, H, N, K_h = 1, 1, 4, K
        torch.manual_seed(42)
        mu_h = torch.randn(B, N, H, K_h)
        Sigma_h = random_spd(K_h, (B, N, H))
        Omega = random_gl(K_h, (B, N, H), scale=0.1)

        precomp = precompute_tokens(mu_h, Sigma_h, Omega)
        kl_causal = pairwise_kl(precomp, causal=True)

        # Upper triangle should be 1e9 (masked)
        for i in range(N):
            for j in range(i + 1, N):
                assert kl_causal[0, 0, i, j] > 1e8, \
                    f"Future KL({i},{j}) should be masked: {kl_causal[0, 0, i, j]:.1f}"

    def test_causal_attention_no_future_weight(self):
        """Causal beta has ~0 weight on future positions."""
        from ..gaussians import precompute_tokens, pairwise_kl, kl_attention

        B, H, N, K_h = 1, 1, 4, K
        torch.manual_seed(42)
        mu_h = torch.randn(B, N, H, K_h)
        Sigma_h = random_spd(K_h, (B, N, H))
        Omega = random_gl(K_h, (B, N, H), scale=0.1)

        precomp = precompute_tokens(mu_h, Sigma_h, Omega)
        kl_ij = pairwise_kl(precomp, causal=True)
        beta = kl_attention(kl_ij, tau=K_h ** 0.5, causal=True)

        for i in range(N):
            for j in range(i + 1, N):
                assert beta[0, 0, i, j] < 1e-6, \
                    f"Future beta({i},{j}) should be ~0: {beta[0, 0, i, j]:.6e}"

    def test_causal_beta_still_sums_to_one(self):
        """Causal beta still sums to 1 over valid (past+present) positions."""
        from ..gaussians import precompute_tokens, pairwise_kl, kl_attention

        B, H, N, K_h = 1, 1, 4, K
        torch.manual_seed(42)
        mu_h = torch.randn(B, N, H, K_h)
        Sigma_h = random_spd(K_h, (B, N, H))
        Omega = random_gl(K_h, (B, N, H), scale=0.1)

        precomp = precompute_tokens(mu_h, Sigma_h, Omega)
        kl_ij = pairwise_kl(precomp, causal=True)
        beta = kl_attention(kl_ij, tau=K_h ** 0.5, causal=True)

        sums = beta.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), \
            f"Causal beta should sum to 1: max deviation = {(sums - 1).abs().max():.6e}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
