# -*- coding: utf-8 -*-
"""
Model Tests
===========

Tests for transformer.core.model.GaugeTransformerLM.

Covers model creation, forward pass, gradient flow, eval-mode determinism,
configuration variants (evolve_sigma, evolve_phi, kappa, covariance mode),
and state-dict save/load. The gauge group and phi_dim are derived from the
irrep_spec and generators; these tests exercise the default SO(N) path.
"""

import pytest
import torch
import torch.nn as nn


class TestGaugeTransformerLMCreation:
    """Test GaugeTransformerLM creation with various configs.

    Verifies that model construction succeeds with different irrep_spec
    layouts, FFN modes, embedding tying, and covariance settings.
    """

    def test_create_minimal_model(self, minimal_config, cpu_device):
        """Test creating model with minimal config."""
        from transformer.core.model import GaugeTransformerLM

        model = GaugeTransformerLM(minimal_config)
        model = model.to(cpu_device)

        assert model is not None
        assert isinstance(model, nn.Module)

    def test_model_has_required_components(self, minimal_config, cpu_device):
        """Test model has all required submodules."""
        from transformer.core.model import GaugeTransformerLM

        model = GaugeTransformerLM(minimal_config)

        # Check core components exist
        assert hasattr(model, 'token_embed')
        assert hasattr(model, 'pos_encoding')
        assert hasattr(model, 'transformer')
        assert hasattr(model, 'out_proj')

    def test_model_config_stored(self, minimal_config, cpu_device):
        """Test model stores config correctly."""
        from transformer.core.model import GaugeTransformerLM

        model = GaugeTransformerLM(minimal_config)

        assert model.config == minimal_config

    def test_model_with_different_ffn_modes(self, cpu_device):
        """Test model creation with different FFN modes."""
        from transformer.core.model import GaugeTransformerLM

        for ffn_mode in ['VFE_dynamic']:
            config = {
                'vocab_size': 100,
                'embed_dim': 15,
                'n_layers': 1,
                'irrep_spec': [('l0', 6, 1), ('l1', 3, 3)],
                'hidden_dim': 32,
                'max_seq_len': 32,
                'kappa_beta': 1.0,
                'dropout': 0.0,
                'use_diagonal_covariance': True,
                'ffn_mode': ffn_mode,
            }
            model = GaugeTransformerLM(config)
            assert model is not None

    def test_model_with_tied_embeddings(self, cpu_device):
        """Test model with tied embeddings."""
        from transformer.core.model import GaugeTransformerLM

        config = {
            'vocab_size': 100,
            'embed_dim': 15,
            'n_layers': 1,
            'irrep_spec': [('l0', 6, 1), ('l1', 3, 3)],
            'hidden_dim': 32,
            'max_seq_len': 32,
            'kappa_beta': 1.0,
            'use_diagonal_covariance': True,
            'tie_embeddings': True,
        }
        model = GaugeTransformerLM(config)

        # Check output projection exists
        assert model.out_proj is not None

    def test_model_parameter_count(self, minimal_config, cpu_device):
        """Test model has reasonable parameter count."""
        from transformer.core.model import GaugeTransformerLM

        model = GaugeTransformerLM(minimal_config)

        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0
        # Minimal model should have < 1M parameters
        assert n_params < 1_000_000


class TestGaugeTransformerLMForward:
    """Test GaugeTransformerLM forward pass.

    Forward maps input_ids -> logits (B, N, V), propagating belief states
    (mu, sigma, phi) through the gauge transformer stack.
    """

    def test_forward_basic(self, gauge_model, batch_tensors, cpu_device):
        """Test basic forward pass."""
        input_ids = batch_tensors['input_ids'].to(cpu_device)

        with torch.no_grad():
            logits = gauge_model(input_ids)

        # Check output shape
        B, N = input_ids.shape
        V = gauge_model.config['vocab_size']
        assert logits.shape == (B, N, V)

    def test_forward_output_finite(self, gauge_model, batch_tensors, cpu_device):
        """Test forward pass produces finite outputs."""
        input_ids = batch_tensors['input_ids'].to(cpu_device)

        with torch.no_grad():
            logits = gauge_model(input_ids)

        assert torch.isfinite(logits).all(), "Output contains NaN or Inf"

    def test_forward_with_agents(self, gauge_model, batch_tensors, cpu_device):
        """Test forward pass returning agent states."""
        input_ids = batch_tensors['input_ids'].to(cpu_device)

        with torch.no_grad():
            logits, agents = gauge_model(input_ids, return_agents=True)

        # Check logits
        B, N = input_ids.shape
        V = gauge_model.config['vocab_size']
        assert logits.shape == (B, N, V)

        # Check agent states
        assert 'mu' in agents
        assert 'sigma' in agents
        assert agents['mu'].shape[0] == B
        assert agents['mu'].shape[1] == N

    def test_forward_different_batch_sizes(self, minimal_config, cpu_device):
        """Test forward with different batch sizes."""
        from transformer.core.model import GaugeTransformerLM

        model = GaugeTransformerLM(minimal_config)
        model = model.to(cpu_device)
        model.eval()

        V = minimal_config['vocab_size']
        N = 16

        for B in [1, 2, 4, 8]:
            input_ids = torch.randint(0, V, (B, N), device=cpu_device)
            with torch.no_grad():
                logits = model(input_ids)
            assert logits.shape == (B, N, V)

    def test_forward_different_sequence_lengths(self, minimal_config, cpu_device):
        """Test forward with different sequence lengths."""
        from transformer.core.model import GaugeTransformerLM

        model = GaugeTransformerLM(minimal_config)
        model = model.to(cpu_device)
        model.eval()

        V = minimal_config['vocab_size']
        B = 2
        max_len = minimal_config['max_seq_len']

        for N in [4, 8, 16, max_len]:
            input_ids = torch.randint(0, V, (B, N), device=cpu_device)
            with torch.no_grad():
                logits = model(input_ids)
            assert logits.shape == (B, N, V)

    def test_forward_with_attention(self, minimal_config, cpu_device):
        """Test forward_with_attention method."""
        from transformer.core.model import GaugeTransformerLM

        model = GaugeTransformerLM(minimal_config)
        model = model.to(cpu_device)
        model.eval()

        V = minimal_config['vocab_size']
        B, N = 2, 16
        input_ids = torch.randint(0, V, (B, N), device=cpu_device)

        with torch.no_grad():
            logits, attn_info = model.forward_with_attention(input_ids)

        assert logits.shape == (B, N, V)
        assert 'beta_layers' in attn_info or 'beta' in attn_info


class TestGaugeTransformerLMGradients:
    """Test gradient flow through the full gauge transformer stack."""

    def test_gradients_flow(self, minimal_config, cpu_device):
        """Test gradients flow through model."""
        from transformer.core.model import GaugeTransformerLM

        model = GaugeTransformerLM(minimal_config)
        model = model.to(cpu_device)
        model.train()

        V = minimal_config['vocab_size']
        B, N = 2, 16
        input_ids = torch.randint(0, V, (B, N), device=cpu_device)
        targets = torch.randint(0, V, (B, N), device=cpu_device)

        logits = model(input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, V),
            targets.view(-1)
        )
        loss.backward()

        # Check some parameters have gradients
        has_grad = False
        for p in model.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "No parameters received gradients"

    def test_gradients_finite(self, minimal_config, cpu_device):
        """Test gradients are finite."""
        from transformer.core.model import GaugeTransformerLM

        model = GaugeTransformerLM(minimal_config)
        model = model.to(cpu_device)
        model.train()

        V = minimal_config['vocab_size']
        B, N = 2, 16
        input_ids = torch.randint(0, V, (B, N), device=cpu_device)
        targets = torch.randint(0, V, (B, N), device=cpu_device)

        logits = model(input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, V),
            targets.view(-1)
        )
        loss.backward()

        # Check all gradients are finite
        for name, p in model.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"Gradient for {name} contains NaN/Inf"


class TestGaugeTransformerLMEvalMode:
    """Test model behavior in eval mode."""

    def test_eval_mode_deterministic(self, gauge_model, batch_tensors, cpu_device):
        """Test model is deterministic in eval mode."""
        input_ids = batch_tensors['input_ids'].to(cpu_device)

        gauge_model.eval()

        with torch.no_grad():
            out1 = gauge_model(input_ids)
            out2 = gauge_model(input_ids)

        assert torch.allclose(out1, out2), "Model not deterministic in eval mode"

    def test_train_eval_toggle(self, gauge_model, batch_tensors, cpu_device):
        """Test switching between train and eval mode."""
        input_ids = batch_tensors['input_ids'].to(cpu_device)

        # Start in eval mode
        gauge_model.eval()
        with torch.no_grad():
            eval_out = gauge_model(input_ids)

        # Switch to train and back
        gauge_model.train()
        gauge_model.eval()

        with torch.no_grad():
            eval_out2 = gauge_model(input_ids)

        assert torch.allclose(eval_out, eval_out2)


class TestGaugeTransformerLMConfigurations:
    """Test model behavior under different configuration knobs.

    Exercises evolve_sigma, evolve_phi, kappa_beta temperature,
    layer count, and diagonal vs full covariance mode.
    """

    def test_config_evolve_sigma(self, cpu_device):
        """Test evolve_sigma configuration."""
        from transformer.core.model import GaugeTransformerLM

        for evolve_sigma in [True, False]:
            config = {
                'vocab_size': 100,
                'embed_dim': 15,
                'n_layers': 1,
                'hidden_dim': 32,
                'max_seq_len': 32,
                'kappa_beta': 1.0,
                'evolve_sigma': evolve_sigma,
                'irrep_spec': [('l0', 6, 1), ('l1', 3, 3)],
                'use_diagonal_covariance': True,
                'ffn_mode': 'VFE_dynamic',
            }
            model = GaugeTransformerLM(config)
            input_ids = torch.randint(0, 100, (2, 16))

            with torch.no_grad():
                logits = model(input_ids)

            assert torch.isfinite(logits).all()

    def test_config_evolve_phi(self, cpu_device):
        """Test evolve_phi configuration."""
        from transformer.core.model import GaugeTransformerLM

        for evolve_phi in [True, False]:
            config = {
                'vocab_size': 100,
                'embed_dim': 15,
                'n_layers': 1,
                'hidden_dim': 32,
                'max_seq_len': 32,
                'kappa_beta': 1.0,
                'evolve_phi': evolve_phi,
                'irrep_spec': [('l0', 6, 1), ('l1', 3, 3)],
                'use_diagonal_covariance': True,
                'ffn_mode': 'VFE_dynamic',
            }
            model = GaugeTransformerLM(config)
            input_ids = torch.randint(0, 100, (2, 16))

            with torch.no_grad():
                logits = model(input_ids)

            assert torch.isfinite(logits).all()

    def test_config_different_kappa(self, cpu_device):
        """Test different kappa_beta values."""
        from transformer.core.model import GaugeTransformerLM

        for kappa in [0.1, 1.0, 10.0]:
            config = {
                'vocab_size': 100,
                'embed_dim': 15,
                'n_layers': 1,
                'hidden_dim': 32,
                'max_seq_len': 32,
                'kappa_beta': kappa,
                'irrep_spec': [('l0', 6, 1), ('l1', 3, 3)],
                'use_diagonal_covariance': True,
                'ffn_mode': 'VFE_dynamic',
            }
            model = GaugeTransformerLM(config)
            input_ids = torch.randint(0, 100, (2, 16))

            with torch.no_grad():
                logits = model(input_ids)

            assert torch.isfinite(logits).all()

    def test_config_multiple_layers(self, cpu_device):
        """Test model with multiple layers."""
        from transformer.core.model import GaugeTransformerLM

        for n_layers in [1, 2, 3]:
            config = {
                'vocab_size': 100,
                'embed_dim': 15,
                'n_layers': n_layers,
                'hidden_dim': 32,
                'max_seq_len': 32,
                'kappa_beta': 1.0,
                'irrep_spec': [('l0', 6, 1), ('l1', 3, 3)],
                'use_diagonal_covariance': True,
                'ffn_mode': 'VFE_dynamic',
            }
            model = GaugeTransformerLM(config)
            input_ids = torch.randint(0, 100, (2, 16))

            with torch.no_grad():
                logits = model(input_ids)

            assert torch.isfinite(logits).all()

    def test_config_diagonal_covariance(self, cpu_device):
        """Test diagonal covariance mode."""
        from transformer.core.model import GaugeTransformerLM

        config = {
            'vocab_size': 100,
            'embed_dim': 15,
            'n_layers': 1,
            'hidden_dim': 32,
            'max_seq_len': 32,
            'kappa_beta': 1.0,
            'diagonal_covariance': True,
            'irrep_spec': [('l0', 6, 1), ('l1', 3, 3)],
            'ffn_mode': 'VFE_dynamic',
        }
        model = GaugeTransformerLM(config)
        input_ids = torch.randint(0, 100, (2, 16))

        with torch.no_grad():
            logits = model(input_ids)

        assert torch.isfinite(logits).all()


class TestGaugeTransformerLMSaveLoad:
    """Test model save/load functionality."""

    def test_state_dict_roundtrip(self, minimal_config, cpu_device, tmp_path):
        """Test saving and loading state dict."""
        from transformer.core.model import GaugeTransformerLM

        # Create and run model
        model1 = GaugeTransformerLM(minimal_config)
        model1 = model1.to(cpu_device)

        input_ids = torch.randint(0, 100, (2, 16), device=cpu_device)
        with torch.no_grad():
            out1 = model1(input_ids)

        # Save state dict
        state_dict = model1.state_dict()
        save_path = tmp_path / "model.pt"
        torch.save(state_dict, save_path)

        # Create new model and load
        model2 = GaugeTransformerLM(minimal_config)
        model2 = model2.to(cpu_device)
        model2.load_state_dict(torch.load(save_path, weights_only=True))

        # Check outputs match
        with torch.no_grad():
            out2 = model2(input_ids)

        assert torch.allclose(out1, out2)


class TestPFlowAndDeltaRule:
    """Test P-flow EMA embedding updates and delta rule W_out updates."""

    @pytest.fixture
    def model_for_pflow(self, minimal_config, cpu_device):
        """Create a model with tie_embeddings=False for clean P-flow/delta rule testing."""
        from transformer.core.model import GaugeTransformerLM

        config = minimal_config.copy()
        config['tie_embeddings'] = False
        model = GaugeTransformerLM(config).to(cpu_device)
        return model

    def test_p_flow_update_modifies_embeddings(self, model_for_pflow, cpu_device):
        """Test that P-flow EMA updates modify token embeddings."""
        model = model_for_pflow
        B, N, K = 2, 8, model.config['embed_dim']
        V = model.config['vocab_size']

        token_ids = torch.randint(0, V, (B, N))
        mu_beliefs = torch.randn(B, N, K) * 0.1
        prediction_errors = torch.rand(B, N) * 5.0  # CE losses

        # Save original embeddings for comparison
        original_weight = model.token_embed.mu_embed.weight.data.clone()

        model.p_flow_update(
            token_ids=token_ids,
            mu_beliefs=mu_beliefs,
            prediction_errors=prediction_errors,
            ema_decay=0.9,
        )

        # Embeddings for tokens in the batch should have changed
        changed = (model.token_embed.mu_embed.weight.data != original_weight).any(dim=-1)
        assert changed.any(), "P-flow should modify at least some token embeddings"

    def test_p_flow_ignores_padding(self, model_for_pflow, cpu_device):
        """Test that P-flow does NOT update pad token embeddings."""
        model = model_for_pflow
        B, N, K = 2, 8, model.config['embed_dim']
        pad_id = 0  # Use token 0 as pad

        token_ids = torch.randint(1, 50, (B, N))
        token_ids[:, -2:] = pad_id  # Last 2 positions are padding
        mu_beliefs = torch.randn(B, N, K) * 0.1
        prediction_errors = torch.rand(B, N) * 5.0
        prediction_errors[:, -2:] = 0.0  # Pad positions have CE=0

        original_pad_embed = model.token_embed.mu_embed.weight.data[pad_id].clone()

        model.p_flow_update(
            token_ids=token_ids,
            mu_beliefs=mu_beliefs,
            prediction_errors=prediction_errors,
            ema_decay=0.9,
            pad_token_id=pad_id,
        )

        # Pad token embedding should NOT have changed
        assert torch.allclose(
            model.token_embed.mu_embed.weight.data[pad_id],
            original_pad_embed,
        ), "P-flow should not update padding token embeddings"

    def test_delta_rule_modifies_w_out(self, model_for_pflow, cpu_device):
        """Test that delta rule updates modify W_out."""
        model = model_for_pflow
        B, N, K = 2, 8, model.config['embed_dim']
        V = model.config['vocab_size']

        mu_beliefs = torch.randn(B, N, K) * 0.1
        targets = torch.randint(0, V, (B, N))

        original_weight = model.out_proj.weight.data.clone()

        model.delta_rule_update_w_out(
            mu_beliefs=mu_beliefs,
            targets=targets,
            lr=0.1,
        )

        assert not torch.allclose(
            model.out_proj.weight.data, original_weight
        ), "Delta rule should modify W_out"

    def test_delta_rule_ignores_padding(self, model_for_pflow, cpu_device):
        """Test that delta rule excludes padding positions from update."""
        model = model_for_pflow
        B, N, K = 2, 8, model.config['embed_dim']
        V = model.config['vocab_size']
        pad_id = 0

        mu_beliefs = torch.randn(B, N, K) * 0.1

        # All-padding batch should produce no update
        targets_all_pad = torch.full((B, N), pad_id, dtype=torch.long)

        original_weight = model.out_proj.weight.data.clone()
        model.delta_rule_update_w_out(
            mu_beliefs=mu_beliefs,
            targets=targets_all_pad,
            lr=0.1,
            pad_token_id=pad_id,
        )

        assert torch.allclose(
            model.out_proj.weight.data, original_weight
        ), "Delta rule should not update W_out when all positions are padding"

    def test_delta_rule_tied_embeddings_safe(self, minimal_config, cpu_device):
        """Test that delta rule with tied embeddings doesn't crash."""
        from transformer.core.model import GaugeTransformerLM

        config = minimal_config.copy()
        config['tie_embeddings'] = True
        model = GaugeTransformerLM(config).to(cpu_device)

        B, N, K = 2, 8, config['embed_dim']
        V = config['vocab_size']

        mu_beliefs = torch.randn(B, N, K) * 0.1
        targets = torch.randint(0, V, (B, N))

        # Should not crash even with tied weights
        model.delta_rule_update_w_out(
            mu_beliefs=mu_beliefs,
            targets=targets,
            lr=0.1,
        )

        # Verify weight is still shared
        assert model.out_proj.weight is model.token_embed.mu_embed.weight
