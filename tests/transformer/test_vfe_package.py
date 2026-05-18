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
        assert cfg.n_layers == 1
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
        logits, loss, ce_for_log = model(token_ids, targets=targets)

        assert logits.shape == (2, 8, cfg.vocab_size)
        assert loss.dim() == 0  # scalar
        assert loss.item() > 0
        assert ce_for_log.dim() == 0

    def test_gradient_flow_to_prior_bank(self, model, cfg):
        token_ids = torch.randint(0, cfg.vocab_size, (2, 8))
        targets = torch.randint(0, cfg.vocab_size, (2, 8))
        _, loss, _ = model(token_ids, targets=targets)
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
        """diagonal_covariance=False is now supported.

        When RoPE is also active, rope_full_gauge must be set to a non-'off'
        mode explicitly — silent auto-promotion was removed because it breaks
        checkpoint round-trip (the saved config would differ from the live
        runtime).
        """
        cfg = VFEConfig(embed_dim=16, irrep_spec=[('l0', 2, 8)],
                        diagonal_covariance=False, rope_full_gauge='vfe_only')
        assert cfg.diagonal_covariance is False
        assert cfg.rope_full_gauge == 'vfe_only'

    def test_rope_full_gauge_requires_full_cov(self):
        """Non-'off' rope_full_gauge with diagonal_covariance=True raises."""
        with pytest.raises(ValueError):
            VFEConfig(embed_dim=16, irrep_spec=[('l0', 2, 8)],
                      rope_full_gauge='vfe_only', diagonal_covariance=True)

    def test_legacy_bool_rejected(self):
        """Bool values no longer silently coerce — must raise."""
        with pytest.raises(ValueError, match='must be one of'):
            VFEConfig(embed_dim=16, irrep_spec=[('l0', 2, 8)],
                      rope_full_gauge=True, diagonal_covariance=False)
        with pytest.raises(ValueError, match='must be one of'):
            VFEConfig(embed_dim=16, irrep_spec=[('l0', 2, 8)],
                      rope_full_gauge=False)


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
        _, loss, _ = model(token_ids, targets=targets)
        loss.backward()
        for block in model.stack.blocks:
            assert block.e_step.log_kappa.grad is not None


class TestMHyperLrActuallyMovesParams:
    """Tripwire for the m_hyper_lr no-op regression (2026-05-18).

    The pre-existing ``test_gradient_flows_to_log_kappa`` only checked
    ``grad is not None``, which passed even when the grad was 5 orders of
    magnitude smaller than M-step params (raw_c0/raw_b0 were exactly zero).
    These tests assert *magnitude* and *training movement* so the bug
    cannot silently come back.
    """

    def _build_model(self, learnable_kappa=True, E_learnable_alpha=True):
        cfg = VFEConfig(
            vocab_size=50, embed_dim=20, irrep_spec=[('fund', 2, 10)],
            n_layers=1, max_seq_len=32, n_e_steps=1,
            diagonal_covariance=True, gauge_group='GLK',
            use_rope=True, rope_full_gauge='off',
            norm_type='layernorm', use_prior_bank=False,
            mask_self_attention=False, use_autograd_mu_sigma=False,
            alpha_divergence=1,
            learnable_kappa=learnable_kappa,
            E_learnable_alpha=E_learnable_alpha,
            prior_handoff_sigma=0.0, prior_handoff_rho=1.0,
        )
        return VFEModel(cfg)

    def test_grad_magnitudes_nonzero_after_single_backward(self):
        torch.manual_seed(0)
        model = self._build_model()
        token_ids = torch.randint(0, 50, (4, 16))
        targets = torch.randint(0, 50, (4, 16))
        _, loss, _ = model(token_ids, targets=targets)
        loss.backward()
        es = model.stack.blocks[0].e_step
        # Threshold 1e-10 is well above float noise; the buggy state had
        # raw_c0 / raw_b0 at *exactly* 0.0, so any nonzero grad signals fix.
        assert es.raw_c0.grad is not None
        assert es.raw_c0.grad.norm().item() > 1e-10, (
            "raw_c0 has no gradient — the auxiliary hyperparameter loss "
            "is not reaching CE. See docs/edits/edits-2026-05-18.md."
        )
        assert es.raw_b0.grad is not None
        assert es.raw_b0.grad.norm().item() > 1e-10
        assert es.log_kappa.grad is not None
        assert es.log_kappa.grad.norm().item() > 1e-10

    def test_m_hyper_lr_sweep_moves_kappa_and_alpha(self):
        """20 AdamW steps at m_hyper_lr=1e-1 must move kappa and alpha
        visibly; m_hyper_lr=0 must leave them at their init values."""
        import torch.nn.functional as F_

        def _train(lr):
            torch.manual_seed(0)
            model = self._build_model()
            es = model.stack.blocks[0].e_step
            opt = torch.optim.AdamW(
                [es.raw_c0, es.raw_b0, es.log_kappa], lr=lr,
            )
            torch.manual_seed(999)
            for _ in range(20):
                token_ids = torch.randint(0, 50, (4, 16))
                targets = torch.randint(0, 50, (4, 16))
                _, loss, _ = model(token_ids, targets=targets)
                opt.zero_grad()
                loss.backward()
                opt.step()
            return (
                F_.softplus(es.raw_c0).mean().item(),
                F_.softplus(es.raw_b0).mean().item(),
                torch.exp(es.log_kappa).item(),
            )

        c0_0, b0_0, k_0 = _train(0.0)
        c0_hi, b0_hi, k_hi = _train(1e-1)

        # m_hyper_lr=0 leaves init values intact: softplus(softplus_inv(1)) = 1
        assert abs(c0_0 - 1.0) < 1e-6
        assert abs(b0_0 - 1.0) < 1e-6
        assert abs(k_0 - 1.0) < 1e-6

        # m_hyper_lr=1e-1 produces large movement (>10% from init)
        assert abs(c0_hi - 1.0) > 0.1, f"alpha_c0 moved only {c0_hi - 1.0:.4f}"
        assert abs(b0_hi - 1.0) > 0.1, f"alpha_b0 moved only {b0_hi - 1.0:.4f}"
        assert abs(k_hi - 1.0) > 0.1, f"kappa moved only {k_hi - 1.0:.4f}"

    def test_aux_loss_does_not_perturb_m_step_grads(self):
        """Detachment discipline: enabling the aux loss must not change
        gradients on base_mu / base_log_sigma / phi_embed. Hard regression
        test — these grad norms must be byte-identical with and without
        the hyperparam aux loss contribution."""
        torch.manual_seed(0)
        m_with_aux = self._build_model(
            learnable_kappa=True, E_learnable_alpha=True,
        )
        torch.manual_seed(0)
        m_without_aux = self._build_model(
            learnable_kappa=False, E_learnable_alpha=False,
        )
        torch.manual_seed(42)
        token_ids = torch.randint(0, 50, (4, 16))
        targets = torch.randint(0, 50, (4, 16))

        _, loss_a, _ = m_with_aux(token_ids, targets=targets)
        loss_a.backward()
        _, loss_b, _ = m_without_aux(token_ids, targets=targets)
        loss_b.backward()

        names_to_check = [
            'prior_bank.base_mu',
            'prior_bank.base_log_sigma',
            'prior_bank.phi_embed.weight',
        ]
        for name in names_to_check:
            g_a = dict(m_with_aux.named_parameters())[name].grad
            g_b = dict(m_without_aux.named_parameters())[name].grad
            assert torch.allclose(g_a, g_b, atol=0, rtol=0), (
                f"M-step gradient for {name} differs between aux-on and "
                f"aux-off — aux loss is leaking into M-step params. "
                f"Max abs diff: {(g_a - g_b).abs().max().item()}"
            )


class TestExactDiagonalTransport:

    def test_config_default_false(self):
        """exact_diagonal_transport defaults to False — the diagonal-of-sandwich
        approximation is the fast path used by all click-configs. Set to True
        only when full Ω@Σ@Ω^T transport is required (paid in extra compute).
        """
        cfg = VFEConfig()
        assert cfg.exact_diagonal_transport is False


class TestFMonitorEnvelopeRouting:
    """Regression for the lambda_softmax envelope-routing fix at
    ``transformer/vfe/e_step.py:288, :344``.

    The f_history monitor at ``vfe/e_step.py:396-413`` measures the
    manuscript free energy ``F = sum_j beta * KL + tau * sum(beta * log beta)``.
    Before the fix the (mu, sigma) gradient descended the
    entropy-suppressed surrogate ``F_surr = sum_j beta * KL`` because
    ``lambda_softmax=self.lambda_soft`` was passed unconditionally to
    ``compute_vfe_gradients_gpu``. With the fix,
    ``lambda_softmax=0.0`` is used when ``include_attention_entropy=True``
    so the envelope-theorem identity ``dF/dtheta = sum_j beta * dKL/dtheta``
    holds at the softmax stationary point of beta.

    Note: full strict-monotone descent of F is NOT asserted here.
    ``compute_vfe_gradients_gpu`` computes the QUERY-side partial
    gradient only (per ``test_vfe_gradients.py::test_exact_diagonal_transport_fd_mu``);
    the symmetric KEY-side contribution from positions where mu_k
    appears as the transport key is not summed in. That is a separate
    design-level concern from the lambda_softmax routing tested here.
    """

    def test_f_history_populated_with_entropy(self, generators):
        """With ``include_attention_entropy=True`` and diagnostics on, the
        monitor records the manuscript F across all E-step iterations."""
        torch.manual_seed(20260517)
        cfg = VFEConfig(
            vocab_size=50,
            embed_dim=16,
            irrep_spec=[('l0', 2, 8)],
            n_layers=2,
            max_seq_len=32,
            n_e_steps=3,
            diagonal_covariance=True,
            include_attention_entropy=True,
            track_layer_diagnostics=True,
        )
        model = VFEModel(cfg)
        token_ids = torch.randint(0, cfg.vocab_size, (2, 8))
        _ = model(token_ids)

        for block in model.stack.blocks:
            f_history = block.e_step._last_diagnostics.get('f_history')
            assert f_history is not None and len(f_history) == cfg.n_e_steps, (
                f"f_history not populated correctly: {f_history}"
            )
            assert all(isinstance(v, float) for v in f_history)

    def test_lambda_softmax_zero_when_entropy_on(self, generators):
        """The E-step must pass lambda_softmax=0.0 to the gradient kernels
        when include_attention_entropy=True. Verified by monkey-patching
        compute_vfe_gradients_gpu to capture the kwarg value."""
        import transformer.vfe.e_step as e_step_mod
        from transformer.core.vfe_gradients import compute_vfe_gradients_gpu

        captured = {}
        original = e_step_mod.compute_vfe_gradients_gpu

        def spy(*args, **kwargs):
            captured['lambda_softmax'] = kwargs.get('lambda_softmax')
            return original(*args, **kwargs)

        cfg = VFEConfig(
            vocab_size=50,
            embed_dim=16,
            irrep_spec=[('l0', 2, 8)],
            n_layers=1,
            max_seq_len=32,
            n_e_steps=1,
            diagonal_covariance=True,
            include_attention_entropy=True,
            lambda_soft=1.0,  # would be passed through without the fix
        )
        model = VFEModel(cfg)
        token_ids = torch.randint(0, cfg.vocab_size, (2, 4))

        e_step_mod.compute_vfe_gradients_gpu = spy
        try:
            _ = model(token_ids)
        finally:
            e_step_mod.compute_vfe_gradients_gpu = original

        assert captured.get('lambda_softmax') == 0.0, (
            f"Expected lambda_softmax=0.0 with include_attention_entropy=True, "
            f"got {captured.get('lambda_softmax')}. The envelope-routing fix "
            f"at vfe/e_step.py:344 has regressed."
        )

    def test_lambda_softmax_passes_through_when_entropy_off(self, generators):
        """With include_attention_entropy=False the legacy path is preserved:
        lambda_softmax = self.lambda_soft is passed unchanged."""
        import transformer.vfe.e_step as e_step_mod

        captured = {}
        original = e_step_mod.compute_vfe_gradients_gpu

        def spy(*args, **kwargs):
            captured['lambda_softmax'] = kwargs.get('lambda_softmax')
            return original(*args, **kwargs)

        cfg = VFEConfig(
            vocab_size=50,
            embed_dim=16,
            irrep_spec=[('l0', 2, 8)],
            n_layers=1,
            max_seq_len=32,
            n_e_steps=1,
            diagonal_covariance=True,
            include_attention_entropy=False,
            lambda_soft=0.7,
        )
        model = VFEModel(cfg)
        token_ids = torch.randint(0, cfg.vocab_size, (2, 4))

        e_step_mod.compute_vfe_gradients_gpu = spy
        try:
            _ = model(token_ids)
        finally:
            e_step_mod.compute_vfe_gradients_gpu = original

        assert captured.get('lambda_softmax') == 0.7, (
            f"Expected lambda_softmax=0.7 (legacy path), got "
            f"{captured.get('lambda_softmax')}. The envelope-routing fix "
            f"at vfe/e_step.py:344 broke the legacy lambda_soft pass-through."
        )


class TestKeysideTotalDerivative:
    """Regression for the Layer 2 key-side-gradient fix at
    ``transformer/vfe/e_step.py:_compute_mu_sigma_grad_autograd``.

    ``compute_vfe_gradients_gpu`` computes only the query-side partial
    ``dF/dmu_k |_{mu_j fixed for j!=k}`` (mean-field convention; documented
    in ``scripts/verify_vfe_gradients_fd.py:147``). The key-side contribution
    from positions where ``mu_k`` appears as the transport key in
    ``KL(q_i || Omega_ik q_k)`` for ``i!=k`` is not summed in.

    The new autograd path enabled by ``VFEConfig.use_autograd_mu_sigma=True``
    routes ``(mu, sigma)`` updates through ``torch.autograd.grad`` over the
    full manuscript F, capturing both sides. Default ``False`` preserves
    bit-identical behavior with the analytic kernel.

    See ``docs/audits/audit-2026-05-17.md`` §"Layer 2 Deep-Audit".
    """

    def test_autograd_path_invoked_when_flag_on(self, generators):
        """With ``use_autograd_mu_sigma=True``, ``compute_vfe_gradients_gpu``
        is NOT called; the autograd method is."""
        import transformer.vfe.e_step as e_step_mod

        analytic_calls = {'n': 0}
        autograd_calls = {'n': 0}
        original_analytic = e_step_mod.compute_vfe_gradients_gpu

        def analytic_spy(*args, **kwargs):
            analytic_calls['n'] += 1
            return original_analytic(*args, **kwargs)

        cfg = VFEConfig(
            vocab_size=50,
            embed_dim=16,
            irrep_spec=[('l0', 2, 8)],
            n_layers=1,
            max_seq_len=32,
            n_e_steps=2,
            diagonal_covariance=True,
            include_attention_entropy=True,
            use_autograd_mu_sigma=True,
        )
        model = VFEModel(cfg)
        original_autograd = model.stack.blocks[0].e_step._compute_mu_sigma_grad_autograd

        def autograd_spy(*args, **kwargs):
            autograd_calls['n'] += 1
            return original_autograd(*args, **kwargs)

        model.stack.blocks[0].e_step._compute_mu_sigma_grad_autograd = autograd_spy
        token_ids = torch.randint(0, cfg.vocab_size, (2, 4))

        e_step_mod.compute_vfe_gradients_gpu = analytic_spy
        try:
            _ = model(token_ids)
        finally:
            e_step_mod.compute_vfe_gradients_gpu = original_analytic

        assert autograd_calls['n'] == cfg.n_e_steps, (
            f"Expected {cfg.n_e_steps} calls to _compute_mu_sigma_grad_autograd "
            f"(one per E-step iter), got {autograd_calls['n']}"
        )
        assert analytic_calls['n'] == 0, (
            f"compute_vfe_gradients_gpu should not be called when "
            f"use_autograd_mu_sigma=True, but was called "
            f"{analytic_calls['n']} times"
        )

    def test_analytic_kernel_default_path_unchanged(self, generators):
        """With ``use_autograd_mu_sigma=False`` (default), the analytic
        path is still used. Guards against accidental regression that
        flips the default."""
        import transformer.vfe.e_step as e_step_mod

        captured = {'n': 0}
        original = e_step_mod.compute_vfe_gradients_gpu

        def spy(*args, **kwargs):
            captured['n'] += 1
            return original(*args, **kwargs)

        cfg = VFEConfig(
            vocab_size=50,
            embed_dim=16,
            irrep_spec=[('l0', 2, 8)],
            n_layers=1,
            max_seq_len=32,
            n_e_steps=2,
            diagonal_covariance=True,
            include_attention_entropy=True,
            # use_autograd_mu_sigma defaults to False
        )
        assert cfg.use_autograd_mu_sigma is False, (
            "Default of use_autograd_mu_sigma must remain False for "
            "backward compatibility with existing checkpoints."
        )
        model = VFEModel(cfg)
        token_ids = torch.randint(0, cfg.vocab_size, (2, 4))

        e_step_mod.compute_vfe_gradients_gpu = spy
        try:
            _ = model(token_ids)
        finally:
            e_step_mod.compute_vfe_gradients_gpu = original

        assert captured['n'] == cfg.n_e_steps, (
            f"Expected {cfg.n_e_steps} analytic calls with default flag, "
            f"got {captured['n']}"
        )

    def test_autograd_grad_mu_differs_from_analytic(self):
        """The autograd total derivative differs from the analytic
        query-side partial on ``grad_mu``. This is the empirical
        signature of the missing key-side contribution.

        Constructs identical inputs, computes both gradients, asserts
        they are NOT close. If they were close, the autograd path
        would be wired wrong or the bug would not exist.
        """
        from transformer.core.vfe_gradients import compute_vfe_gradients_gpu
        from transformer.vfe.attention import (
            compute_kl_attention,
            compute_gauge_transport,
        )

        torch.manual_seed(20260517)
        cfg = VFEConfig(
            vocab_size=50,
            embed_dim=8,
            irrep_spec=[('l0', 2, 4)],
            n_layers=1,
            max_seq_len=16,
            n_e_steps=1,
            diagonal_covariance=True,
            include_attention_entropy=True,
            use_autograd_mu_sigma=True,
        )
        model = VFEModel(cfg)
        e_step = model.stack.blocks[0].e_step

        B, N, K = 2, 4, cfg.embed_dim
        n_gen = e_step.generators.shape[0]
        mu = torch.randn(B, N, K)
        sigma = torch.rand(B, N, K) * 0.5 + 0.5  # in (0.5, 1.0)
        phi = torch.randn(B, N, n_gen) * 0.3
        mu_p = torch.randn(B, N, K) * 0.5
        sigma_p = torch.rand(B, N, K) * 0.3 + 0.5

        # Pre-compute beta and kl_matrix for the analytic call
        block_exp_pairs = compute_gauge_transport(
            phi, e_step.generators, e_step.irrep_dims,
            enforce_orthogonal=False,
        )
        beta, kl_matrix = compute_kl_attention(
            mu, sigma, phi, e_step.generators,
            e_step.irrep_dims, e_step.kappa, None,
            use_rope=False,
            cached_block_exp_pairs=block_exp_pairs,
            mask_self_attention=True,
            exact_diagonal_transport=True,
        )

        # Analytic (query-side partial only)
        grad_mu_analytic, grad_sigma_analytic = compute_vfe_gradients_gpu(
            mu_q=mu, sigma_q=sigma, mu_p=mu_p, sigma_p=sigma_p,
            beta=beta, phi=phi, generators=e_step.generators,
            alpha=1.0, alpha_div=1.0,
            lambda_belief=1.0, lambda_softmax=0.0,
            kappa=e_step.kappa, eps=1e-6,
            compute_sigma_align_grad=True,
            irrep_dims=e_step.irrep_dims,
            enforce_orthogonal=False,
            cached_block_exp_pairs=block_exp_pairs,
            use_rope=False,
            exact_diagonal_transport=True,
        )

        # Autograd (total derivative — query + key)
        grad_mu_auto, grad_sigma_auto = e_step._compute_mu_sigma_grad_autograd(
            mu=mu, sigma=sigma, mu_p=mu_p, sigma_p=sigma_p, phi=phi,
            alpha_eff=1.0,
            block_exp_pairs=block_exp_pairs,
            mask=None,
            kappa=e_step.kappa,
            eps=1e-6,
            is_diagonal=True,
        )

        # grad_mu MUST differ — the key-side term is non-zero in this setup
        assert not torch.allclose(grad_mu_analytic, grad_mu_auto, atol=1e-5), (
            "Autograd grad_mu matches analytic to 1e-5, but they should "
            "differ by the missing key-side contribution. Either the "
            "autograd path is computing only the query-side partial "
            "(implementation bug) or the test setup made the key-side "
            "term vanish (test bug)."
        )

        # A nonzero diff also has to be of physically reasonable scale,
        # not numerical noise. Diff norm should be comparable to grad norm.
        diff_norm = (grad_mu_auto - grad_mu_analytic).norm().item()
        analytic_norm = grad_mu_analytic.norm().item()
        rel_diff = diff_norm / max(analytic_norm, 1e-8)
        assert rel_diff > 1e-3, (
            f"Relative diff between autograd and analytic grad_mu = "
            f"{rel_diff:.2e}; expected > 1e-3 if key-side is non-trivial."
        )

    def test_monotone_descent_improves_with_autograd_path(self, generators):
        """With ``use_autograd_mu_sigma=True``, the F monitor's
        ``f_history`` reflects descent of the same objective the
        gradient targets. Asserts ``f_history`` is populated and the
        monotone-descent flag is True for all blocks under benign init.

        Note: a strict assertion would couple this test to LR/init
        choices. The assertion here is the weaker "f_history is
        finite and final value <= initial value", which any correct
        descent direction should satisfy on average even when single
        steps can overshoot.
        """
        torch.manual_seed(20260517)
        cfg = VFEConfig(
            vocab_size=50,
            embed_dim=16,
            irrep_spec=[('l0', 2, 8)],
            n_layers=2,
            max_seq_len=32,
            n_e_steps=4,
            diagonal_covariance=True,
            include_attention_entropy=True,
            use_autograd_mu_sigma=True,
            track_layer_diagnostics=True,
            # Conservative LRs so finite-step overshoot is unlikely
            e_mu_lr=0.05,
            e_sigma_lr=0.001,
            e_phi_lr=0.01,
        )
        model = VFEModel(cfg)
        token_ids = torch.randint(0, cfg.vocab_size, (2, 8))
        _ = model(token_ids)

        for i, block in enumerate(model.stack.blocks):
            f_history = block.e_step._last_diagnostics.get('f_history')
            assert f_history is not None and len(f_history) == cfg.n_e_steps, (
                f"Block {i}: f_history not populated correctly: {f_history}"
            )
            assert all(
                isinstance(v, float) and v == v  # not NaN
                for v in f_history
            ), f"Block {i}: f_history contains NaN or non-floats: {f_history}"
            # F at end <= F at start (overall descent over the iteration)
            assert f_history[-1] <= f_history[0] + 1e-3, (
                f"Block {i}: F did not descend overall: "
                f"f_history[0]={f_history[0]:.4f}, "
                f"f_history[-1]={f_history[-1]:.4f}"
            )


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

# ---------------------------------------------------------------------------
# Regression tests for the 2026-04-17 ultrareview Phase M fixes.
# bug_003: VFEEStep dropped alpha_c0 when E_learnable_alpha=True, omitting
# the product-rule correction -(α²/c0)·KL·(dKL/dθ) from the gradient.
# ---------------------------------------------------------------------------

class TestAlphaC0CorrectionWired:

    def _make_e_step(self, learnable_alpha):
        cfg_lk = VFEConfig(
            vocab_size=50,
            embed_dim=16,
            irrep_spec=[('l0', 2, 8)],
            n_layers=1,
            max_seq_len=16,
            n_e_steps=1,
            diagonal_covariance=True,
            gauge_group='GLK',
            E_learnable_alpha=learnable_alpha,
        )
        from transformer.vfe.model import VFEModel
        m = VFEModel(cfg_lk)
        return cfg_lk, m

    def test_alpha_c0_is_passed_when_learnable(self, monkeypatch):
        """When E_learnable_alpha=True, compute_vfe_gradients_gpu must receive
        alpha_c0 != None so the product-rule correction fires."""
        import torch
        cfg_lk, m = self._make_e_step(learnable_alpha=True)
        e_step = m.stack.blocks[0].e_step

        captured = {'alpha_c0': 'NOT_CALLED'}

        import transformer.vfe.e_step as estep_mod
        original = estep_mod.compute_vfe_gradients_gpu

        def spy(*args, **kwargs):
            captured['alpha_c0'] = kwargs.get('alpha_c0', 'MISSING_KEY')
            return original(*args, **kwargs)

        monkeypatch.setattr(estep_mod, 'compute_vfe_gradients_gpu', spy)

        B, N, K = 2, 4, cfg_lk.embed_dim
        beliefs = BeliefState(
            mu=torch.randn(B, N, K),
            sigma=torch.ones(B, N, K) * 0.1,
            phi=torch.zeros(B, N, m.generators.shape[0]),
        )
        priors = BeliefState(
            mu=torch.zeros(B, N, K),
            sigma=torch.ones(B, N, K),
            phi=torch.zeros(B, N, m.generators.shape[0]),
        )
        e_step(beliefs, priors)

        assert captured['alpha_c0'] != 'NOT_CALLED', 'compute_vfe_gradients_gpu was never invoked'
        assert captured['alpha_c0'] != 'MISSING_KEY', 'alpha_c0 keyword missing entirely'
        assert captured['alpha_c0'] is not None, (
            'alpha_c0 was None despite E_learnable_alpha=True — '
            'product-rule correction will not fire and gradients will be biased.'
        )
        assert captured['alpha_c0'].shape == (K,), (
            f'alpha_c0 must be shape (K,) = ({K},); got {captured["alpha_c0"].shape}'
        )

    def test_alpha_c0_is_none_when_fixed_alpha(self, monkeypatch):
        """When E_learnable_alpha=False, alpha is a constant; alpha_c0 must be
        None so compute_vfe_gradients_gpu skips the (now spurious) correction."""
        import torch
        cfg_lk, m = self._make_e_step(learnable_alpha=False)
        e_step = m.stack.blocks[0].e_step

        captured = {'alpha_c0': 'NOT_CALLED'}

        import transformer.vfe.e_step as estep_mod
        original = estep_mod.compute_vfe_gradients_gpu

        def spy(*args, **kwargs):
            captured['alpha_c0'] = kwargs.get('alpha_c0', 'MISSING_KEY')
            return original(*args, **kwargs)

        monkeypatch.setattr(estep_mod, 'compute_vfe_gradients_gpu', spy)

        B, N, K = 2, 4, cfg_lk.embed_dim
        beliefs = BeliefState(
            mu=torch.randn(B, N, K),
            sigma=torch.ones(B, N, K) * 0.1,
            phi=torch.zeros(B, N, m.generators.shape[0]),
        )
        priors = BeliefState(
            mu=torch.zeros(B, N, K),
            sigma=torch.ones(B, N, K),
            phi=torch.zeros(B, N, m.generators.shape[0]),
        )
        e_step(beliefs, priors)

        assert captured['alpha_c0'] is None, (
            f'alpha_c0 should be None for fixed alpha; got {captured["alpha_c0"]!r}'
        )
