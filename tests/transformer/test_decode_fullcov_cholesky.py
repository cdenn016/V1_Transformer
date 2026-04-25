"""
Tests for PriorBank._decode_full_cov Cholesky hardening (Phase 1).

Covers three cases:
  1. Near-singular sigma_p (cond ~500): must not raise; logits finite.
  2. Indefinite sigma_p (negative eigenvalue): must raise RuntimeError
     after 5-round ridge escalation.
  3. Well-conditioned sigma_p: logit differences across vocab entries
     match the textbook Gaussian KL up to the softmax-invariant terms
     (-K and -log|Sigma_q|) that the implementation drops.
"""

import pytest
import torch

from transformer.core.prior_bank import PriorBank


def _make_fullcov_prior_bank(vocab_size=4, embed_dim=8):
    """Minimal PriorBank configured for the full-cov decode path."""
    return PriorBank(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        phi_dim=3,
        diagonal_covariance=False,
        full_cov_decode=True,
        learnable_sigma=True,
        gauge_fixed_priors=False,
        sigma_ce_scale=0.7,
    )


def _patch_priors(bank, mu_p, sigma_p, phi_p=None):
    """Monkey-patch _get_prior_for_tokens to return controlled priors.

    The real method returns (mu, sigma, phi). _decode_full_cov unpacks
    only the first two, so phi can be zeros of the right shape.
    """
    if phi_p is None:
        phi_p = torch.zeros(mu_p.shape[0], 3)

    def _fake(token_ids, only_forward=False):
        return mu_p, sigma_p, phi_p

    bank._get_prior_for_tokens = _fake


class TestFullCovDecodeCholeskyEscalation:
    """Phase 1: decode must survive near-singular sigma_p."""

    def test_near_singular_sigma_p_no_raise(self):
        torch.manual_seed(0)
        V, K = 4, 8
        bank = _make_fullcov_prior_bank(V, K)
        bank.eval()

        # Build (V, K, K) sigma_p where vocab entry 1 has cond ~500.
        base = torch.eye(K).unsqueeze(0).expand(V, K, K).clone()
        eig_sharp = torch.ones(K)
        eig_sharp[0] = 0.002  # cond = 1 / 0.002 = 500
        Q, _ = torch.linalg.qr(torch.randn(K, K))
        sharp = Q @ torch.diag(eig_sharp) @ Q.T
        base[1] = 0.5 * (sharp + sharp.T)

        mu_p = torch.zeros(V, K)
        _patch_priors(bank, mu_p, base)

        B, N = 2, 3
        mu_q = torch.randn(B, N, K) * 0.1
        sigma_q = torch.eye(K).expand(B, N, K, K).clone()

        logits = bank._decode_full_cov(mu_q, sigma_q, tau=1.0)

        assert logits.shape == (B, N, V)
        assert torch.isfinite(logits).all(), "decode produced non-finite logits"

    def test_indefinite_sigma_p_raises_after_escalation(self):
        torch.manual_seed(0)
        V, K = 4, 8
        bank = _make_fullcov_prior_bank(V, K)
        bank.eval()

        # Inject sigma_p entry with a deeply negative eigenvalue that
        # 5-round 10x escalation cannot lift past zero. Starting ridge
        # 0.01, max escalated = 0.01 * 10^5 = 1e3, so eigenvalue must
        # be worse than -1e3 to guarantee all 5 rounds fail.
        base = torch.eye(K).unsqueeze(0).expand(V, K, K).clone()
        eig_bad = torch.ones(K)
        eig_bad[0] = -2.0e3  # defeats escalation ceiling
        Q, _ = torch.linalg.qr(torch.randn(K, K))
        bad = Q @ torch.diag(eig_bad) @ Q.T
        base[2] = 0.5 * (bad + bad.T)

        mu_p = torch.zeros(V, K)
        _patch_priors(bank, mu_p, base)

        B, N = 1, 1
        mu_q = torch.zeros(B, N, K)
        sigma_q = torch.eye(K).expand(B, N, K, K).clone()

        with pytest.raises(RuntimeError, match="Cholesky failed after 5"):
            bank._decode_full_cov(mu_q, sigma_q, tau=1.0)

    def test_logits_match_textbook_kl_on_wellconditioned_input(self):
        """Well-conditioned sigma_p: logit diffs = -KL diffs / tau.

        The implementation drops -K and -log|Sigma_q| because both are
        softmax-invariant. Differences across vocab entries therefore
        must equal -(KL_v1 - KL_v0) / tau exactly.
        """
        torch.manual_seed(0)
        V, K = 4, 4
        bank = _make_fullcov_prior_bank(V, K)
        bank.eval()

        # Build distinct well-conditioned sigma_p per vocab entry.
        mats = []
        for v in range(V):
            A = torch.randn(K, K) * 0.3 + torch.eye(K)
            M = A @ A.T + 0.5 * torch.eye(K)
            mats.append(M)
        sigma_p = torch.stack(mats, dim=0)          # (V, K, K)
        mu_p = torch.randn(V, K) * 0.2

        _patch_priors(bank, mu_p, sigma_p)

        B, N = 1, 1
        mu_q = torch.randn(B, N, K) * 0.3
        sigma_q = torch.eye(K).unsqueeze(0).unsqueeze(0).expand(B, N, K, K).clone()
        sigma_q = sigma_q * 1.5

        tau = 1.0
        logits = bank._decode_full_cov(mu_q, sigma_q, tau=tau)  # (1, 1, V)

        # Compute textbook Gaussian KL for each vocab entry.
        # The PriorBank decode wraps everything in float32 autocast-off and
        # adds variance_floor * I to sigma_p before the Cholesky. Reproduce
        # that here so the ground truth matches what the decode actually
        # sees.
        variance_floor = max(1e-6, 0.01)
        sigma_p_eff = (sigma_p + variance_floor * torch.eye(K)).float()
        mu_p_f = mu_p.float()
        mu_q_v = mu_q[0, 0].float()
        sigma_q_v = sigma_q[0, 0].float()

        kls = []
        for v in range(V):
            dist_q = torch.distributions.MultivariateNormal(mu_q_v, sigma_q_v)
            dist_p = torch.distributions.MultivariateNormal(mu_p_f[v], sigma_p_eff[v])
            kls.append(torch.distributions.kl.kl_divergence(dist_q, dist_p))
        kls = torch.stack(kls)                      # (V,)

        # Logit differences across vocab entries should equal -ΔKL / tau.
        logit_diffs = logits[0, 0] - logits[0, 0, 0]        # (V,)
        kl_diffs = -(kls - kls[0]) / tau                    # (V,)

        assert torch.allclose(logit_diffs, kl_diffs, rtol=1e-4, atol=1e-5), (
            f"logit diffs {logit_diffs.tolist()} "
            f"!= -ΔKL/tau {kl_diffs.tolist()}"
        )
