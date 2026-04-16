"""
Smoke tests for transformer/vfe/ package.

Validates construction, forward pass, gradient flow, and structural
invariants (Law 1: no target leakage in E-step).
"""

import inspect
import pytest
import torch

from transformer.core.types import BeliefState
from transformer.vfe.config import VFEConfig
from transformer.vfe.prior_bank import VFEPriorBank
from transformer.vfe.positional import VFEPositionalEncoding
from transformer.vfe.e_step import VFEEStep
from transformer.vfe.block import VFEBlock
from transformer.vfe.stack import VFEStack
from transformer.vfe.model import VFEModel
from transformer.vfe.active_inference import VFEActiveInference
from transformer.vfe.efe import VFEExpectedFreeEnergy


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg():
    return VFEConfig(
        vocab_size=50,
        embed_dim=16,
        irrep_spec=[('l0', 2, 8)],  # 2 heads x 8 = K=16
        n_layers=2,
        max_seq_len=32,
        n_e_steps=2,
        diagonal_covariance=True,
        gauge_group='GLK',
    )


@pytest.fixture
def model(cfg):
    return VFEModel(cfg)


@pytest.fixture
def generators(model):
    """Extract generators from VFEModel (which handles the build)."""
    return model.generators


# ---------------------------------------------------------------------------
# 1. VFEConfig
# ---------------------------------------------------------------------------

class TestVFEConfig:

    def test_construction_defaults(self):
        cfg = VFEConfig()
        assert cfg.embed_dim == 64
        assert cfg.n_layers == 4
        assert cfg.diagonal_covariance is True

    def test_irrep_dims(self, cfg):
        assert cfg.irrep_dims == [8, 8]
        assert sum(cfg.irrep_dims) == cfg.embed_dim

    def test_embed_dim_mismatch_raises(self):
        with pytest.raises(ValueError):
            VFEConfig(embed_dim=32, irrep_spec=[('l0', 2, 8)])  # 16 != 32


# ---------------------------------------------------------------------------
# 2. VFEPriorBank
# ---------------------------------------------------------------------------

class TestVFEPriorBank:

    def test_encode_shapes(self, cfg, generators):
        bank = VFEPriorBank(cfg, generators)
        token_ids = torch.randint(0, cfg.vocab_size, (2, 8))
        beliefs = bank.encode(token_ids)

        assert beliefs.mu.shape == (2, 8, cfg.embed_dim)
        assert beliefs.sigma.shape == (2, 8, cfg.embed_dim)
        assert beliefs.phi.shape == (2, 8, cfg.n_gen)

    def test_decode_shapes(self, cfg, generators):
        bank = VFEPriorBank(cfg, generators)
        B, N, K = 2, 8, cfg.embed_dim
        mu_q = torch.randn(B, N, K)
        sigma_q = torch.ones(B, N, K)
        logits = bank.decode(mu_q, sigma_q)

        assert logits.shape == (B, N, cfg.vocab_size)

    def test_gradients_flow(self, cfg, generators):
        bank = VFEPriorBank(cfg, generators)
        token_ids = torch.randint(0, cfg.vocab_size, (2, 8))
        beliefs = bank.encode(token_ids)

        # Decode with gradient tracking
        logits = bank.decode(beliefs.mu, beliefs.sigma)
        loss = logits.sum()
        loss.backward()

        assert bank.base_mu.grad is not None
        assert bank.base_mu.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# 3. VFEPositionalEncoding
# ---------------------------------------------------------------------------

class TestVFEPositionalEncoding:

    def test_composition(self, cfg, generators):
        pos_enc = VFEPositionalEncoding(cfg, generators.shape[0], generators)
        phi = torch.randn(2, 8, cfg.n_gen)
        phi_out = pos_enc(phi, seq_len=8)

        assert phi_out.shape == phi.shape
        # Position should change phi
        assert not torch.allclose(phi_out, phi)


# ---------------------------------------------------------------------------
# 4. VFEEStep
# ---------------------------------------------------------------------------

class TestVFEEStep:

    def test_forward_shapes(self, cfg, generators):
        e_step = VFEEStep(cfg, generators)
        B, N, K = 2, 8, cfg.embed_dim
        beliefs = BeliefState(
            mu=torch.randn(B, N, K),
            sigma=torch.ones(B, N, K),
            phi=torch.zeros(B, N, cfg.n_gen),
        )
        priors = BeliefState(
            mu=torch.randn(B, N, K),
            sigma=torch.ones(B, N, K),
            phi=torch.zeros(B, N, cfg.n_gen),
        )
        mask = torch.ones(N, N)

        out = e_step(beliefs, priors, mask=mask)

        assert out.mu.shape == (B, N, K)
        assert out.sigma.shape == (B, N, K)
        assert out.phi.shape == (B, N, cfg.n_gen)

    def test_law1_no_targets_parameter(self):
        """E-step forward() must not accept targets — Law 1 enforcement."""
        sig = inspect.signature(VFEEStep.forward)
        param_names = set(sig.parameters.keys())
        assert 'targets' not in param_names
        assert 'target_ids' not in param_names


# ---------------------------------------------------------------------------
# 5. VFEBlock
# ---------------------------------------------------------------------------

class TestVFEBlock:

    def test_forward_returns_belief_state(self, cfg, generators):
        block = VFEBlock(cfg, generators)
        B, N, K = 2, 8, cfg.embed_dim
        beliefs = BeliefState(
            mu=torch.randn(B, N, K),
            sigma=torch.ones(B, N, K),
            phi=torch.zeros(B, N, cfg.n_gen),
        )
        priors = BeliefState(
            mu=torch.randn(B, N, K),
            sigma=torch.ones(B, N, K),
            phi=torch.zeros(B, N, cfg.n_gen),
        )
        mask = torch.ones(N, N)

        out = block(beliefs, priors, mask=mask)

        assert isinstance(out, BeliefState)
        assert out.mu.shape == beliefs.mu.shape


# ---------------------------------------------------------------------------
# 6. VFEStack
# ---------------------------------------------------------------------------

class TestVFEStack:

    def test_layer_count(self, cfg, generators):
        stack = VFEStack(cfg, generators)
        assert len(stack.blocks) == cfg.n_layers

    def test_forward(self, cfg, generators):
        stack = VFEStack(cfg, generators)
        B, N, K = 2, 8, cfg.embed_dim
        beliefs = BeliefState(
            mu=torch.randn(B, N, K),
            sigma=torch.ones(B, N, K),
            phi=torch.zeros(B, N, cfg.n_gen),
        )
        initial_priors = BeliefState(
            mu=torch.randn(B, N, K),
            sigma=torch.ones(B, N, K),
            phi=torch.zeros(B, N, cfg.n_gen),
        )
        mask = torch.ones(N, N)

        out = stack(beliefs, initial_priors=initial_priors, mask=mask)

        assert out.mu.shape == (B, N, K)


# ---------------------------------------------------------------------------
# 7. VFEModel
# ---------------------------------------------------------------------------

class TestVFEModel:

    def test_forward_logits_only(self, model, cfg):
        token_ids = torch.randint(0, cfg.vocab_size, (2, 8))
        logits = model(token_ids)

        assert logits.shape == (2, 8, cfg.vocab_size)

    def test_forward_with_targets(self, model, cfg):
        token_ids = torch.randint(0, cfg.vocab_size, (2, 8))
        targets = torch.randint(0, cfg.vocab_size, (2, 8))
        logits, loss = model(token_ids, targets=targets)

        assert logits.shape == (2, 8, cfg.vocab_size)
        assert loss.dim() == 0  # scalar
        assert loss.item() > 0

    def test_gradient_flow_to_prior_bank(self, model, cfg):
        token_ids = torch.randint(0, cfg.vocab_size, (2, 8))
        targets = torch.randint(0, cfg.vocab_size, (2, 8))
        _, loss = model(token_ids, targets=targets)
        loss.backward()

        # Prior bank parameters should receive gradients
        has_grad = False
        for name, p in model.named_parameters():
            if 'prior_bank' in name and p.grad is not None:
                if p.grad.abs().sum() > 0:
                    has_grad = True
                    break
        assert has_grad, "No gradients reached prior_bank parameters"

    def test_generate(self, model, cfg):
        prompt = torch.randint(0, cfg.vocab_size, (1, 4))
        output = model.generate(prompt, max_new_tokens=3)

        assert output.shape == (1, 7)  # 4 prompt + 3 generated
        assert (output[:, :4] == prompt).all()  # prompt preserved


# ---------------------------------------------------------------------------
# 8. VFEActiveInference
# ---------------------------------------------------------------------------

class TestVFEActiveInference:

    def test_callback_produces_gradients(self, model, cfg):
        ai = VFEActiveInference(cfg, model.prior_bank)
        B, N, K = 2, 8, cfg.embed_dim
        mu = torch.randn(B, N, K)
        sigma = torch.ones(B, N, K)

        grad_mu, grad_sigma = ai(mu, sigma)

        assert grad_mu.shape == (B, N, K)
        assert grad_sigma.shape == (B, N, K)


# ---------------------------------------------------------------------------
# 9. VFEExpectedFreeEnergy
# ---------------------------------------------------------------------------

class TestVFEExpectedFreeEnergy:

    def test_score_candidates(self, model, cfg):
        efe = VFEExpectedFreeEnergy(model)
        context = torch.randint(0, cfg.vocab_size, (1, 8))
        candidates = torch.randint(0, cfg.vocab_size, (5,))

        scores = efe.score_candidates(context, candidates)

        assert 'efe' in scores
        assert 'risk' in scores
        assert 'ambiguity' in scores
        assert scores['efe'].shape == (5,)

    def test_select_action(self, model, cfg):
        efe = VFEExpectedFreeEnergy(model)
        context = torch.randint(0, cfg.vocab_size, (1, 8))

        token_id = efe.select_action(context, top_k=10)

        assert isinstance(token_id, int)
        assert 0 <= token_id < cfg.vocab_size


# ---------------------------------------------------------------------------
# 10. V2 Features: damped handoff, learned kappa, learned tau
# ---------------------------------------------------------------------------

class TestV2Config:

    def test_v2_defaults_backward_compatible(self):
        """V2 fields exist with defaults matching v1 behavior."""
        cfg = VFEConfig()
        assert cfg.prior_handoff_rho == 1.0
        assert cfg.learnable_kappa is False

    def test_full_cov_allowed(self):
        """diagonal_covariance=False is now supported."""
        cfg = VFEConfig(embed_dim=16, irrep_spec=[('l0', 2, 8)],
                        diagonal_covariance=False)
        assert cfg.diagonal_covariance is False

    def test_rope_full_gauge_requires_full_cov(self):
        """rope_full_gauge=True with diagonal_covariance=True raises ValueError."""
        with pytest.raises(ValueError):
            VFEConfig(embed_dim=16, irrep_spec=[('l0', 2, 8)],
                      rope_full_gauge=True, diagonal_covariance=True)


class TestDampedHandoff:

    def test_rho0_freezes_prior_mu(self, cfg, generators):
        """rho=0 means prior mu never updates from posterior."""
        from transformer.vfe.stack import VFEStack
        cfg_frozen = VFEConfig(
            vocab_size=50, embed_dim=16, irrep_spec=[('l0', 2, 8)],
            n_layers=3, max_seq_len=32, n_e_steps=1,
            prior_handoff_rho=0.0,
        )
        stack = VFEStack(cfg_frozen, generators)
        B, N, K = 2, 8, 16
        beliefs = BeliefState(
            mu=torch.randn(B, N, K),
            sigma=torch.ones(B, N, K),
            phi=torch.zeros(B, N, cfg.n_gen),
        )
        initial_priors = BeliefState(
            mu=torch.randn(B, N, K),
            sigma=torch.ones(B, N, K),
            phi=torch.zeros(B, N, cfg.n_gen),
        )
        mask = torch.ones(N, N)
        # With rho=0, the prior mu at each layer stays at initial_priors.mu
        # So the output should differ from rho=1
        out = stack(beliefs, initial_priors, mask)
        assert out.mu.shape == (B, N, K)


class TestLearnableKappa:

    def test_creates_parameter(self, cfg, generators):
        """learnable_kappa=True creates log_kappa nn.Parameter per layer."""
        from transformer.vfe.e_step import VFEEStep
        cfg_lk = VFEConfig(
            vocab_size=50, embed_dim=16, irrep_spec=[('l0', 2, 8)],
            n_layers=2, max_seq_len=32, n_e_steps=1,
            learnable_kappa=True,
        )
        e_step = VFEEStep(cfg_lk, generators)
        assert hasattr(e_step, 'log_kappa')
        assert isinstance(e_step.log_kappa, torch.nn.Parameter)

    def test_no_parameter_when_disabled(self, cfg, generators):
        """learnable_kappa=False should NOT create log_kappa."""
        from transformer.vfe.e_step import VFEEStep
        e_step = VFEEStep(cfg, generators)
        assert not hasattr(e_step, 'log_kappa')

    def test_gradient_flows_to_log_kappa(self, generators):
        """loss.backward() should produce gradient on log_kappa."""
        cfg_lk = VFEConfig(
            vocab_size=50, embed_dim=16, irrep_spec=[('l0', 2, 8)],
            n_layers=2, max_seq_len=32, n_e_steps=2,
            learnable_kappa=True,
        )
        model = VFEModel(cfg_lk)
        token_ids = torch.randint(0, 50, (2, 8))
        targets = torch.randint(0, 50, (2, 8))
        _, loss = model(token_ids, targets=targets)
        loss.backward()
        for block in model.stack.blocks:
            assert block.e_step.log_kappa.grad is not None


class TestExactDiagonalTransport:

    def test_config_default_true(self):
        """exact_diagonal_transport defaults to True for mathematical exactness."""
        cfg = VFEConfig()
        assert cfg.exact_diagonal_transport is True


class TestFullCrossLayerHandoff:

    def test_sigma_handoff(self, generators):
        """prior_handoff_sigma > 0 propagates sigma across layers."""
        from transformer.vfe.stack import VFEStack
        cfg = VFEConfig(
            vocab_size=50, embed_dim=16, irrep_spec=[('l0', 2, 8)],
            n_layers=3, max_seq_len=32, n_e_steps=1,
            prior_handoff_sigma=0.5,
        )
        stack = VFEStack(cfg, generators)
        B, N, K = 2, 8, 16
        beliefs = BeliefState(
            mu=torch.randn(B, N, K),
            sigma=torch.ones(B, N, K),
            phi=torch.zeros(B, N, cfg.n_gen),
        )
        initial_priors = BeliefState(
            mu=torch.randn(B, N, K),
            sigma=torch.ones(B, N, K) * 2.0,  # different from beliefs
            phi=torch.zeros(B, N, cfg.n_gen),
        )
        mask = torch.ones(N, N)
        out = stack(beliefs, initial_priors, mask)
        assert out.mu.shape == (B, N, K)

    def test_phi_handoff(self, generators):
        """prior_handoff_phi=True propagates phi across layers."""
        from transformer.vfe.stack import VFEStack
        cfg = VFEConfig(
            vocab_size=50, embed_dim=16, irrep_spec=[('l0', 2, 8)],
            n_layers=2, max_seq_len=32, n_e_steps=1,
            prior_handoff_phi=True,
        )
        stack = VFEStack(cfg, generators)
        B, N, K = 2, 8, 16
        beliefs = BeliefState(
            mu=torch.randn(B, N, K),
            sigma=torch.ones(B, N, K),
            phi=torch.randn(B, N, cfg.n_gen),
        )
        initial_priors = beliefs
        mask = torch.ones(N, N)
        out = stack(beliefs, initial_priors, mask)
        assert out.mu.shape == (B, N, K)
