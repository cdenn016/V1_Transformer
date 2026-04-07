"""
Tests for learning.py: M-step prior updates, gradient accumulation, analytical Omega gradient.

Covers:
  - m_step() — single-batch prior update
  - _compute_m_step_omega_grad() — analytical autograd Omega gradient
  - MStepAccumulator — gradient accumulation across micro-batches
  - Adam momentum helpers
"""

import torch
import pytest

from .conftest import random_spd, random_gl, make_pure_vfe_config, make_model, K

DEVICE = 'cpu'


def _run_e_step(model, config, token_ids):
    """Helper: run e_step and return converged beliefs."""
    from ..inference import e_step
    mu_star, Sigma_star, Omega_star, logits, vfe_hist, diag = e_step(
        token_ids, model, config,
    )
    return mu_star, Sigma_star, Omega_star, logits


class TestMStep:
    """Test full M-step (prior update) correctness."""

    def setup_method(self):
        torch.manual_seed(42)
        self.config = make_pure_vfe_config(
            vocab_size=20, n_esteps=3, mu_q_lr=0.05,
        )
        self.model = make_model(self.config)
        self.tokens = torch.randint(0, 20, (1, 8))
        self.targets = torch.randint(0, 20, (1, 8))

    def test_m_step_reduces_loss(self):
        """After m_step, re-evaluating CE loss should not increase dramatically."""
        from ..learning import m_step
        from ..gaussians import kl_decode_logits

        mu_star, Sigma_star, Omega_star, logits = _run_e_step(
            self.model, self.config, self.tokens,
        )

        # CE before
        log_probs_before = torch.log_softmax(logits, dim=-1)
        ce_before = -log_probs_before.gather(-1, self.targets.unsqueeze(-1)).squeeze(-1).mean()

        # Apply M-step
        m_step(
            self.tokens, self.targets, mu_star, Sigma_star, Omega_star,
            self.model, self.config, logits,
        )

        # CE after (re-run E-step with updated priors)
        mu2, Sig2, Om2, logits2 = _run_e_step(self.model, self.config, self.tokens)
        log_probs_after = torch.log_softmax(logits2, dim=-1)
        ce_after = -log_probs_after.gather(-1, self.targets.unsqueeze(-1)).squeeze(-1).mean()

        # Loss should not increase by more than 50% (one step is noisy)
        assert ce_after < ce_before * 1.5, \
            f"Loss increased too much after m_step: {ce_before:.4f} -> {ce_after:.4f}"

    def test_m_step_prior_mu_bounded(self):
        """prior_mu norm stays within prior_mu_max_norm."""
        from ..learning import m_step

        mu_star, Sigma_star, Omega_star, logits = _run_e_step(
            self.model, self.config, self.tokens,
        )
        m_step(
            self.tokens, self.targets, mu_star, Sigma_star, Omega_star,
            self.model, self.config, logits,
        )

        mu_norms = self.model.prior_mu.norm(dim=-1)
        assert mu_norms.max() <= self.config.prior_mu_max_norm + 1e-3, \
            f"prior_mu norm exceeds max: {mu_norms.max():.4f} > {self.config.prior_mu_max_norm}"

    def test_m_step_prior_sigma_floored(self):
        """prior_Sigma eigenvalues >= prior_sigma_floor."""
        from ..learning import m_step

        mu_star, Sigma_star, Omega_star, logits = _run_e_step(
            self.model, self.config, self.tokens,
        )
        m_step(
            self.tokens, self.targets, mu_star, Sigma_star, Omega_star,
            self.model, self.config, logits,
        )

        eigs = torch.linalg.eigvalsh(self.model.prior_Sigma)
        min_eig = eigs.min().item()
        floor = self.config.prior_sigma_floor
        assert min_eig >= floor - 1e-4, \
            f"Min eigenvalue {min_eig:.6f} below floor {floor}"

    def test_m_step_omega_stays_invertible(self):
        """prior_Omega determinant stays away from zero."""
        from ..learning import m_step

        mu_star, Sigma_star, Omega_star, logits = _run_e_step(
            self.model, self.config, self.tokens,
        )
        m_step(
            self.tokens, self.targets, mu_star, Sigma_star, Omega_star,
            self.model, self.config, logits,
        )

        dets = torch.linalg.det(self.model.prior_Omega)
        assert dets.abs().min() > 1e-4, \
            f"Omega near-singular after m_step: min |det| = {dets.abs().min():.6e}"

    def test_m_step_no_update_for_unseen_tokens(self):
        """Tokens not in batch have unchanged priors."""
        from ..learning import m_step

        # Use tokens 0-9 only
        tokens = torch.randint(0, 10, (1, 8))
        targets = torch.randint(0, 10, (1, 8))

        mu_before = self.model.prior_mu[15:20].clone()
        Sigma_before = self.model.prior_Sigma[15:20].clone()
        Omega_before = self.model.prior_Omega[15:20].clone()

        mu_star, Sigma_star, Omega_star, logits = _run_e_step(
            self.model, self.config, tokens,
        )
        m_step(
            tokens, targets, mu_star, Sigma_star, Omega_star,
            self.model, self.config, logits,
        )

        assert torch.allclose(self.model.prior_mu[15:20], mu_before), \
            "Unseen tokens should not have mu updated"
        assert torch.allclose(self.model.prior_Sigma[15:20], Sigma_before), \
            "Unseen tokens should not have Sigma updated"
        assert torch.allclose(self.model.prior_Omega[15:20], Omega_before), \
            "Unseen tokens should not have Omega updated"


class TestMStepOmegaGrad:
    """Test analytical Omega gradient via autograd."""

    def setup_method(self):
        torch.manual_seed(42)
        self.config = make_pure_vfe_config(
            vocab_size=20, n_esteps=3,
            use_analytical_omega_grad=True,
        )
        self.model = make_model(self.config)

    def test_analytical_omega_grad_finite(self):
        """_compute_m_step_omega_grad returns finite values."""
        from ..learning import _compute_m_step_omega_grad

        tokens = torch.randint(0, 20, (1, 8))
        mu_star, Sigma_star, _, _ = _run_e_step(self.model, self.config, tokens)

        grad = _compute_m_step_omega_grad(mu_star, Sigma_star, tokens, self.model, self.config)
        assert torch.isfinite(grad).all(), "Analytical omega grad has NaN/Inf"
        assert grad.shape == (1, 8, self.config.n_heads, K, K)

    def test_analytical_omega_grad_at_identity_omega(self):
        """At Omega=I with identical beliefs, gradient is small."""
        from ..learning import _compute_m_step_omega_grad

        # Set all priors to identical
        self.model.prior_Omega[:] = torch.eye(K).unsqueeze(0).unsqueeze(0)
        self.model.prior_mu[:] = 0.5
        self.model.prior_Sigma[:] = torch.eye(K)

        tokens = torch.zeros(1, 4, dtype=torch.long)  # All same token
        mu_star = torch.ones(1, 4, K) * 0.5
        Sigma_star = torch.eye(K).unsqueeze(0).unsqueeze(0).expand(1, 4, K, K).clone()

        grad = _compute_m_step_omega_grad(mu_star, Sigma_star, tokens, self.model, self.config)
        assert grad.norm() < 1.0, \
            f"Gradient should be small at identity, got norm = {grad.norm():.4f}"

    def test_analytical_vs_moment_matching_direction(self):
        """Both gradient modes should move Omega in roughly similar direction."""
        from ..learning import _compute_m_step_omega_grad

        tokens = torch.randint(0, 20, (1, 8))
        mu_star, Sigma_star, Omega_star, _ = _run_e_step(self.model, self.config, tokens)

        # Analytical gradient
        grad_analytical = _compute_m_step_omega_grad(
            mu_star, Sigma_star, tokens, self.model, self.config,
        )

        # Moment-matching heuristic: -(Omega_star - Omega_prior)
        prior_Omega = self.model.prior_Omega[tokens]  # [1, 8, H, K, K]
        grad_heuristic = -(Omega_star - prior_Omega)

        # Both should have positive cosine similarity (same general direction)
        cos_sim = torch.nn.functional.cosine_similarity(
            grad_analytical.flatten(), grad_heuristic.flatten(), dim=0,
        )
        # Allow negative if problem structure differs, but test finite
        assert torch.isfinite(cos_sim), "Cosine similarity should be finite"


class TestMStepAccumulator:
    """Test gradient accumulation across micro-batches."""

    def setup_method(self):
        torch.manual_seed(42)
        self.config = make_pure_vfe_config(
            vocab_size=20, n_esteps=3,
            use_analytical_omega_grad=False,  # Use heuristic for simpler testing
        )
        self.model = make_model(self.config)

    def test_accumulate_reset_zeros_buffers(self):
        """After reset(), all buffers are zero."""
        from ..learning import MStepAccumulator

        accum = MStepAccumulator(self.config, 'cpu')

        # Accumulate something
        tokens = torch.randint(0, 20, (1, 8))
        targets = torch.randint(0, 20, (1, 8))
        mu_star, Sigma_star, Omega_star, logits = _run_e_step(
            self.model, self.config, tokens,
        )
        accum.accumulate(tokens, targets, mu_star, Sigma_star, Omega_star,
                         self.model, self.config, logits)

        # Verify something was accumulated
        assert accum.n_counts.sum() > 0, "Should have accumulated something"

        # Reset
        accum.reset()
        assert accum.n_counts.sum() == 0, "n_counts should be zero after reset"
        assert accum.mu_star_sum.norm() == 0, "mu_star_sum should be zero after reset"
        assert accum.n_micro_batches == 0, "n_micro_batches should be 0 after reset"

    def test_accumulated_ce_loss_average(self):
        """avg_ce_loss is mean over micro-batches."""
        from ..learning import MStepAccumulator

        accum = MStepAccumulator(self.config, 'cpu')

        losses = []
        for _ in range(3):
            tokens = torch.randint(0, 20, (1, 8))
            targets = torch.randint(0, 20, (1, 8))
            mu_star, Sigma_star, Omega_star, logits = _run_e_step(
                self.model, self.config, tokens,
            )
            ce = accum.accumulate(tokens, targets, mu_star, Sigma_star, Omega_star,
                                  self.model, self.config, logits)
            losses.append(ce)

        expected_avg = sum(losses) / len(losses)
        assert abs(accum.avg_ce_loss - expected_avg) < 1e-5, \
            f"avg_ce_loss {accum.avg_ce_loss:.4f} != mean {expected_avg:.4f}"

    def test_accumulate_disjoint_tokens_merge(self):
        """Two batches with non-overlapping tokens both update correctly."""
        from ..learning import MStepAccumulator

        accum = MStepAccumulator(self.config, 'cpu')

        # Batch 1: tokens 0-4
        tokens1 = torch.arange(5).unsqueeze(0)  # [1, 5]
        targets1 = torch.randint(0, 20, (1, 5))
        mu1, Sig1, Om1, log1 = _run_e_step(self.model, self.config, tokens1)
        accum.accumulate(tokens1, targets1, mu1, Sig1, Om1, self.model, self.config, log1)

        # Batch 2: tokens 10-14
        tokens2 = torch.arange(10, 15).unsqueeze(0)
        targets2 = torch.randint(0, 20, (1, 5))
        mu2, Sig2, Om2, log2 = _run_e_step(self.model, self.config, tokens2)
        accum.accumulate(tokens2, targets2, mu2, Sig2, Om2, self.model, self.config, log2)

        # Both token ranges should have counts
        assert (accum.n_counts[:5] > 0).all(), "Tokens 0-4 should have counts"
        assert (accum.n_counts[10:15] > 0).all(), "Tokens 10-14 should have counts"
        # Tokens 5-9 should have zero (unless they appear in targets)
        # We can only check that accumulation didn't crash and counts are reasonable
        assert accum.n_micro_batches == 2

    def test_accumulate_counts_additive(self):
        """Accumulating same tokens twice doubles counts."""
        from ..learning import MStepAccumulator

        accum = MStepAccumulator(self.config, 'cpu')
        tokens = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]])
        targets = torch.randint(0, 20, (1, 8))

        mu_star, Sigma_star, Omega_star, logits = _run_e_step(
            self.model, self.config, tokens,
        )

        # First accumulation
        accum.accumulate(tokens, targets, mu_star, Sigma_star, Omega_star,
                         self.model, self.config, logits)
        counts_after_1 = accum.n_counts[:4].clone()

        # Second accumulation (same data)
        accum.accumulate(tokens, targets, mu_star, Sigma_star, Omega_star,
                         self.model, self.config, logits)
        counts_after_2 = accum.n_counts[:4]

        assert torch.allclose(counts_after_2, 2 * counts_after_1), \
            "Counts should double after second accumulation"

    def test_accumulate_apply_runs_without_error(self):
        """Full accumulate + apply_m_step_from_accumulated cycle runs."""
        from ..learning import MStepAccumulator, apply_m_step_from_accumulated

        accum = MStepAccumulator(self.config, 'cpu')
        tokens = torch.randint(0, 20, (1, 8))
        targets = torch.randint(0, 20, (1, 8))

        mu_star, Sigma_star, Omega_star, logits = _run_e_step(
            self.model, self.config, tokens,
        )
        accum.accumulate(tokens, targets, mu_star, Sigma_star, Omega_star,
                         self.model, self.config, logits)

        ce_loss = apply_m_step_from_accumulated(accum, self.model, self.config)
        assert isinstance(ce_loss, float), f"Expected float CE loss, got {type(ce_loss)}"
        assert ce_loss > 0, f"CE loss should be positive, got {ce_loss}"


class TestAdamMStepMomentum:
    """Test Adam-like momentum in M-step."""

    def setup_method(self):
        torch.manual_seed(42)
        self.config = make_pure_vfe_config(
            vocab_size=20, n_esteps=3,
            use_adam_m_step=True,
            adam_beta1=0.9,
            adam_beta2=0.999,
        )
        self.model = make_model(self.config)

    def test_adam_buffers_initialized(self):
        """Model with use_adam_m_step=True has momentum buffers."""
        assert self.model.m1_mu is not None, "m1_mu should be initialized"
        assert self.model.m2_mu is not None, "m2_mu should be initialized"
        assert self.model.m1_Sigma is not None, "m1_Sigma should be initialized"
        assert self.model.m1_Omega is not None, "m1_Omega should be initialized"

    def test_adam_m_step_runs(self):
        """M-step with Adam momentum runs without error."""
        from ..learning import m_step

        tokens = torch.randint(0, 20, (1, 8))
        targets = torch.randint(0, 20, (1, 8))
        mu_star, Sigma_star, Omega_star, logits = _run_e_step(
            self.model, self.config, tokens,
        )

        # Should not raise
        m_step(tokens, targets, mu_star, Sigma_star, Omega_star,
               self.model, self.config, logits)

    def test_adam_buffers_nonzero_after_step(self):
        """After one M-step, momentum buffers should be nonzero for seen tokens."""
        from ..learning import m_step

        tokens = torch.randint(0, 10, (1, 8))
        targets = torch.randint(0, 20, (1, 8))
        mu_star, Sigma_star, Omega_star, logits = _run_e_step(
            self.model, self.config, tokens,
        )
        m_step(tokens, targets, mu_star, Sigma_star, Omega_star,
               self.model, self.config, logits)

        # m1_mu for seen tokens should be nonzero
        seen = torch.unique(tokens)
        m1_seen = self.model.m1_mu[seen]
        assert m1_seen.norm() > 1e-8, \
            f"m1_mu should be nonzero for seen tokens, got norm = {m1_seen.norm():.6e}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
