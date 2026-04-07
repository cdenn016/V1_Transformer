"""
Mathematical invariant tests for the pure VFE transformer.

Tests properties that MUST hold across the system:
  - SPD preservation (covariance stays positive-definite)
  - KL non-negativity
  - Transport cocycle (Omega_ij @ Omega_jk = Omega_ik)
  - Gauge covariance (KL invariant under GL(K))
  - Attention normalization (beta sums to 1)
  - Connection flatness at initialization
  - VFE monotonicity during E-step
"""

import torch
import pytest

from .conftest import random_spd, random_gl, make_pure_vfe_config, make_model, K


class TestSPDPreservation:
    """Test that covariance stays SPD through the system."""

    def test_sigma_positive_through_estep(self):
        """Sigma eigenvalues > 0 after full E-step."""
        from ..inference import e_step

        config = make_pure_vfe_config(n_esteps=5, mu_q_lr=0.05)
        torch.manual_seed(42)
        model = make_model(config)
        tokens = torch.randint(0, 20, (1, 8))

        _, Sigma_star, _, _, _, _ = e_step(tokens, model, config)

        eigs = torch.linalg.eigvalsh(Sigma_star)
        assert eigs.min() > 0, f"Sigma not SPD: min eigenvalue = {eigs.min():.6e}"

    def test_sigma_symmetric_after_estep(self):
        """Sigma remains symmetric after E-step."""
        from ..inference import e_step

        config = make_pure_vfe_config(n_esteps=5)
        torch.manual_seed(42)
        model = make_model(config)
        tokens = torch.randint(0, 20, (1, 8))

        _, Sigma_star, _, _, _, _ = e_step(tokens, model, config)

        asym = (Sigma_star - Sigma_star.transpose(-2, -1)).norm()
        assert asym < 1e-5, f"Sigma not symmetric: asymmetry norm = {asym:.6e}"

    def test_sigma_bounded_condition_number(self):
        """Condition number doesn't explode through E-step."""
        from ..inference import e_step

        config = make_pure_vfe_config(n_esteps=5)
        torch.manual_seed(42)
        model = make_model(config)
        tokens = torch.randint(0, 20, (1, 8))

        _, Sigma_star, _, _, _, _ = e_step(tokens, model, config)

        eigs = torch.linalg.eigvalsh(Sigma_star)
        cond = eigs[..., -1] / eigs[..., 0].clamp(min=1e-8)
        assert cond.max() < 1e6, f"Condition number too large: max = {cond.max():.1f}"

    def test_prior_sigma_stays_spd_after_mstep(self):
        """Prior Sigma stays SPD after M-step."""
        from ..inference import e_step
        from ..learning import m_step

        config = make_pure_vfe_config(n_esteps=3)
        torch.manual_seed(42)
        model = make_model(config)
        tokens = torch.randint(0, 20, (1, 8))
        targets = torch.randint(0, 20, (1, 8))

        mu_s, Sig_s, Om_s, logits, _, _ = e_step(tokens, model, config)
        m_step(tokens, targets, mu_s, Sig_s, Om_s, model, config, logits)

        eigs = torch.linalg.eigvalsh(model.prior_Sigma)
        assert eigs.min() > 0, f"Prior Sigma not SPD after M-step: min eig = {eigs.min():.6e}"


class TestKLNonNegativity:
    """Test KL(P || Q) >= 0 for all Gaussian pairs."""

    def test_kl_nonnegative_random(self):
        """KL >= 0 for 50 random Gaussian pairs."""
        from ..gaussians import kl_divergence

        torch.manual_seed(42)
        for _ in range(50):
            K_ = 4
            mu_p = torch.randn(K_)
            mu_q = torch.randn(K_)
            Sigma_p = random_spd(K_)
            Sigma_q = random_spd(K_)
            kl = kl_divergence(mu_p, Sigma_p, mu_q, Sigma_q)
            assert kl >= -1e-6, f"KL negative: {kl:.6f}"

    def test_kl_zero_identical(self):
        """KL = 0 when P = Q exactly."""
        from ..gaussians import kl_divergence

        torch.manual_seed(42)
        mu = torch.randn(K)
        Sigma = random_spd(K)
        kl = kl_divergence(mu, Sigma, mu, Sigma)
        assert abs(kl.item()) < 1e-5, f"KL should be 0 for identical: {kl:.6e}"

    def test_kl_nonnegative_after_transport(self):
        """KL(q_i || Omega_ij * q_j) >= 0 for random transport."""
        from ..gaussians import kl_divergence
        from ..gauge import transport_gaussian, compute_transport

        torch.manual_seed(42)
        for _ in range(20):
            mu_i = torch.randn(K)
            mu_j = torch.randn(K)
            Sigma_i = random_spd(K)
            Sigma_j = random_spd(K)
            Omega_i = random_gl(K, scale=0.1)
            Omega_j = random_gl(K, scale=0.1)
            Omega_ij = compute_transport(Omega_i, Omega_j)
            mu_t, Sig_t = transport_gaussian(mu_j, Sigma_j, Omega_ij)
            kl = kl_divergence(mu_i, Sigma_i, mu_t, Sig_t)
            assert kl >= -1e-4, f"KL negative after transport: {kl:.6f}"


class TestTransportCocycle:
    """Test Omega_ij @ Omega_jk = Omega_ik (flat transport cocycle)."""

    def test_cocycle_flat_batched(self):
        """Cocycle condition for multiple random triples."""
        from ..gauge import compute_transport

        torch.manual_seed(42)
        for _ in range(10):
            Oi = random_gl(K, scale=0.2)
            Oj = random_gl(K, scale=0.2)
            Ok = random_gl(K, scale=0.2)

            Oij = compute_transport(Oi, Oj)
            Ojk = compute_transport(Oj, Ok)
            Oik = compute_transport(Oi, Ok)

            composed = Oij @ Ojk
            diff = (composed - Oik).norm()
            assert diff < 1e-4, f"Cocycle violated: diff = {diff:.6e}"

    def test_self_transport_is_identity(self):
        """Omega_ii = I (transport from self to self)."""
        from ..gauge import compute_transport

        torch.manual_seed(42)
        Omega = random_gl(K, scale=0.3)
        Oii = compute_transport(Omega, Omega)
        diff = (Oii - torch.eye(K)).norm()
        assert diff < 1e-5, f"Self-transport should be I: diff = {diff:.6e}"

    def test_inverse_transport(self):
        """Omega_ij @ Omega_ji = I (inverse)."""
        from ..gauge import compute_transport

        torch.manual_seed(42)
        Oi = random_gl(K, scale=0.2)
        Oj = random_gl(K, scale=0.2)
        Oij = compute_transport(Oi, Oj)
        Oji = compute_transport(Oj, Oi)
        product = Oij @ Oji
        diff = (product - torch.eye(K)).norm()
        assert diff < 1e-4, f"Inverse transport should give I: diff = {diff:.6e}"


class TestGaugeCovariance:
    """Test gauge covariance: KL invariant under global GL(K) transformation."""

    def test_kl_gauge_invariant(self):
        """KL(G*P || G*Omega*Q) = KL(P || Omega*Q) for random G."""
        from ..gaussians import kl_divergence
        from ..gauge import transport_gaussian

        torch.manual_seed(42)
        for _ in range(10):
            mu_i = torch.randn(K)
            mu_j = torch.randn(K)
            Sigma_i = random_spd(K)
            Sigma_j = random_spd(K)
            Omega = random_gl(K, scale=0.1)
            G = random_gl(K, scale=0.2)

            # Original
            mu_t, Sig_t = transport_gaussian(mu_j, Sigma_j, Omega)
            kl_orig = kl_divergence(mu_i, Sigma_i, mu_t, Sig_t)

            # Gauge-transformed
            mu_ig, Sig_ig = transport_gaussian(mu_i, Sigma_i, G)
            Omega_g = G @ Omega @ torch.linalg.inv(G)
            mu_jg, Sig_jg = transport_gaussian(mu_j, Sigma_j, G)
            mu_tg, Sig_tg = transport_gaussian(mu_jg, Sig_jg, Omega_g)
            kl_gauged = kl_divergence(mu_ig, Sig_ig, mu_tg, Sig_tg)

            diff = abs(kl_orig.item() - kl_gauged.item())
            assert diff < 1e-3, f"Gauge invariance violated: diff = {diff:.6e}"

    def test_attention_weights_gauge_invariant(self):
        """beta_ij unchanged under global gauge transformation."""
        from ..gaussians import precompute_tokens, pairwise_kl, kl_attention

        torch.manual_seed(42)
        B, H, N, K_h = 1, 1, 4, K
        mu_h = torch.randn(B, N, H, K_h)
        Sigma_h = random_spd(K_h, (B, N, H))
        Omega = random_gl(K_h, (B, N, H), scale=0.1)

        # Original attention
        precomp = precompute_tokens(mu_h, Sigma_h, Omega)
        kl_ij = pairwise_kl(precomp, causal=False)
        tau = K_h ** 0.5
        beta_orig = kl_attention(kl_ij, tau, causal=False)

        # Gauge-transform: multiply all Omega by same G
        G = random_gl(K_h, scale=0.15)
        # G transforms Omega_i -> G @ Omega_i for all i
        # But Omega_ij = Omega_i @ Omega_j^{-1} -> G @ Omega_ij @ G^{-1}
        # This changes the transported covariance: Omega_ij Sigma_j Omega_ij^T
        # To preserve KL, we must also transform beliefs: mu_i -> G @ mu_i, Sigma_i -> G @ Sigma_i @ G^T
        G_expand = G.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        mu_h_g = torch.einsum('...ij,...j->...i', G_expand.expand_as(Omega), mu_h)
        Sigma_h_g = G_expand @ Sigma_h @ G_expand.transpose(-2, -1)
        Omega_g = G_expand @ Omega

        precomp_g = precompute_tokens(mu_h_g, Sigma_h_g, Omega_g)
        kl_ij_g = pairwise_kl(precomp_g, causal=False)
        beta_g = kl_attention(kl_ij_g, tau, causal=False)

        diff = (beta_orig - beta_g).abs().max()
        assert diff < 1e-3, f"Attention not gauge-invariant: max diff = {diff:.6e}"

    def test_vfe_gauge_invariant(self):
        """VFE value unchanged under GL(K) transformation of beliefs + frames."""
        from ..inference import compute_vfe
        from ..gaussians import (
            precompute_tokens, pairwise_kl, kl_attention,
            state_dependent_alpha,
        )

        torch.manual_seed(42)
        B, N_, K_ = 1, 4, K
        H, K_h = 1, K
        mu = torch.randn(B, N_, K_)
        Sigma = random_spd(K_, (B, N_))
        prior_mu = torch.randn(B, N_, K_)
        prior_Sigma = random_spd(K_, (B, N_))
        alpha = torch.ones(B, N_)

        mu_h = mu.view(B, N_, H, K_h)
        Sigma_h_diag = torch.zeros(B, N_, H, K_h, K_h)
        for i in range(N_):
            Sigma_h_diag[0, i, 0] = Sigma[0, i]
        Omega = random_gl(K_h, (B, N_, H), scale=0.1)

        precomp = precompute_tokens(mu_h, Sigma_h_diag, Omega)
        kl_ij = pairwise_kl(precomp, causal=False)
        tau = K_h ** 0.5
        beta = kl_attention(kl_ij, tau, causal=False)

        vfe_orig = compute_vfe(mu, Sigma, prior_mu, prior_Sigma, alpha, beta, kl_ij)

        # Gauge transform
        G = random_gl(K_, scale=0.1)
        mu_g = (G @ mu.unsqueeze(-1)).squeeze(-1)
        Sigma_g = G @ Sigma @ G.T
        prior_mu_g = (G @ prior_mu.unsqueeze(-1)).squeeze(-1)
        prior_Sigma_g = G @ prior_Sigma @ G.T
        alpha_g = alpha  # alpha depends on KL which is gauge-invariant

        mu_h_g = mu_g.view(B, N_, H, K_h)
        Sigma_h_g = torch.zeros_like(Sigma_h_diag)
        for i in range(N_):
            Sigma_h_g[0, i, 0] = Sigma_g[0, i]
        Omega_g = G.unsqueeze(0).unsqueeze(0).unsqueeze(0) @ Omega

        precomp_g = precompute_tokens(mu_h_g, Sigma_h_g, Omega_g)
        kl_ij_g = pairwise_kl(precomp_g, causal=False)
        beta_g = kl_attention(kl_ij_g, tau, causal=False)

        vfe_g = compute_vfe(mu_g, Sigma_g, prior_mu_g, prior_Sigma_g, alpha_g, beta_g, kl_ij_g)

        vfe_o = vfe_orig.item() if hasattr(vfe_orig, 'item') else float(vfe_orig)
        vfe_gv = vfe_g.item() if hasattr(vfe_g, 'item') else float(vfe_g)
        diff = abs(vfe_o - vfe_gv)
        assert diff < 0.1, f"VFE not gauge-invariant: diff = {diff:.4f}"


class TestAttentionNormalization:
    """Test that attention weights are valid probabilities."""

    def test_beta_sums_to_one(self):
        """sum_j beta_ij = 1 for all i."""
        from ..gaussians import precompute_tokens, pairwise_kl, kl_attention

        torch.manual_seed(42)
        B, H, N, K_h = 2, 2, 8, K
        mu_h = torch.randn(B, N, H, K_h)
        Sigma_h = random_spd(K_h, (B, N, H))
        Omega = random_gl(K_h, (B, N, H), scale=0.1)

        precomp = precompute_tokens(mu_h, Sigma_h, Omega)
        kl_ij = pairwise_kl(precomp, causal=False)
        beta = kl_attention(kl_ij, tau=K_h ** 0.5, causal=False)

        sums = beta.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), \
            f"Beta doesn't sum to 1: max deviation = {(sums - 1).abs().max():.6e}"

    def test_beta_nonnegative(self):
        """All beta_ij >= 0."""
        from ..gaussians import precompute_tokens, pairwise_kl, kl_attention

        torch.manual_seed(42)
        B, H, N, K_h = 2, 2, 8, K
        mu_h = torch.randn(B, N, H, K_h)
        Sigma_h = random_spd(K_h, (B, N, H))
        Omega = random_gl(K_h, (B, N, H), scale=0.1)

        precomp = precompute_tokens(mu_h, Sigma_h, Omega)
        kl_ij = pairwise_kl(precomp, causal=False)
        beta = kl_attention(kl_ij, tau=K_h ** 0.5, causal=False)

        assert (beta >= 0).all(), f"Negative beta: min = {beta.min():.6e}"


class TestConnectionFlatnessAtInit:
    """Test that gauge connection is flat at initialization."""

    def test_flat_transport_at_init(self):
        """Omega initialized near identity → transport near identity."""
        from ..gauge import init_omega, compute_transport

        torch.manual_seed(42)
        Omega = init_omega((10, 1, K, K), scale=0.01, device='cpu')
        # Transport between adjacent tokens
        for i in range(9):
            O_ij = compute_transport(Omega[i, 0], Omega[i + 1, 0])
            diff_from_I = (O_ij - torch.eye(K)).norm()
            assert diff_from_I < 0.1, \
                f"Transport({i},{i+1}) not near I: diff = {diff_from_I:.4f}"

    def test_zero_init_phi_gives_identity_omega(self):
        """phi=0 → Omega = I for all tokens."""
        from ..gauge import init_phi, phi_to_omega, make_gl_generators

        generators = make_gl_generators(K)
        phi = init_phi((5, 1, K * K), scale=0.0)
        Omega = phi_to_omega(phi, generators)
        I = torch.eye(K).unsqueeze(0).unsqueeze(0).expand(5, 1, K, K)
        diff = (Omega - I).norm()
        assert diff < 1e-6, f"Zero phi should give I: diff = {diff:.6e}"

    def test_holonomy_vanishes_for_flat(self):
        """Flat transport has zero holonomy: Omega_ij @ Omega_jk @ Omega_ki = I."""
        from ..gauge import init_omega, compute_transport

        torch.manual_seed(42)
        Omega = init_omega((5, 1, K, K), scale=0.01, device='cpu')
        # Wilson loop around triangle (0,1,2)
        O01 = compute_transport(Omega[0, 0], Omega[1, 0])
        O12 = compute_transport(Omega[1, 0], Omega[2, 0])
        O20 = compute_transport(Omega[2, 0], Omega[0, 0])
        holonomy = O01 @ O12 @ O20
        diff = (holonomy - torch.eye(K)).norm()
        assert diff < 1e-4, f"Flat transport should have zero holonomy: diff = {diff:.6e}"


class TestVFEMonotonicity:
    """Test that VFE is non-increasing during E-step."""

    def test_vfe_decreases_default_config(self):
        """VFE non-increasing with default config (K=4, H=1)."""
        from ..inference import e_step

        config = make_pure_vfe_config(n_esteps=8, mu_q_lr=0.05)
        torch.manual_seed(42)
        model = make_model(config)
        tokens = torch.randint(0, 20, (1, 8))

        _, _, _, _, vfe_history, _ = e_step(tokens, model, config)
        for t in range(1, len(vfe_history)):
            assert vfe_history[t] <= vfe_history[t - 1] + 1e-2, \
                f"VFE increased at step {t}: {vfe_history[t - 1]:.4f} -> {vfe_history[t]:.4f}"

    def test_vfe_decreases_multihead(self):
        """VFE non-increasing with H=2 heads."""
        from ..inference import e_step

        config = make_pure_vfe_config(
            belief_dim=8, n_heads=2, head_dim=4,
            n_esteps=5, mu_q_lr=0.05,
        )
        torch.manual_seed(42)
        model = make_model(config)
        tokens = torch.randint(0, 20, (1, 8))

        _, _, _, _, vfe_history, _ = e_step(tokens, model, config)
        for t in range(1, len(vfe_history)):
            assert vfe_history[t] <= vfe_history[t - 1] + 1e-2, \
                f"VFE increased at step {t}: {vfe_history[t - 1]:.4f} -> {vfe_history[t]:.4f}"

    def test_vfe_decreases_with_rope(self):
        """VFE non-increasing with RoPE enabled."""
        from ..inference import e_step

        config = make_pure_vfe_config(n_esteps=5, mu_q_lr=0.05, use_rope=True)
        torch.manual_seed(42)
        model = make_model(config)
        tokens = torch.randint(0, 20, (1, 8))

        _, _, _, _, vfe_history, _ = e_step(tokens, model, config)
        for t in range(1, len(vfe_history)):
            assert vfe_history[t] <= vfe_history[t - 1] + 1e-2, \
                f"VFE increased at step {t}: {vfe_history[t - 1]:.4f} -> {vfe_history[t]:.4f}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
