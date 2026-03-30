"""
Tests for transformer/utils/checkpoint.py
==========================================

Tests checkpoint save/load roundtrip, config preservation, and legacy migration.
"""

import pytest
import torch
import tempfile
import os


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model_and_config():
    """Create minimal model + config for checkpoint testing."""
    from transformer.core.model import GaugeTransformerLM
    config = {
        'vocab_size': 100,
        'embed_dim': 15,
        'n_layers': 1,
        'irrep_spec': [('l0', 6, 1), ('l1', 3, 3)],
        'hidden_dim': 32,
        'max_seq_len': 32,
        'kappa_beta': 1.0,
        'dropout': 0.0,
        'pos_encoding_mode': 'learned',
        'evolve_sigma': True,
        'evolve_phi': False,
        'tie_embeddings': True,
        'diagonal_covariance': True,
        'ffn_mode': 'VFE_dynamic',
    }
    model = GaugeTransformerLM(config)
    return model, config


# ===========================================================================
# TestCheckpointRoundtrip
# ===========================================================================

class TestCheckpointRoundtrip:
    """Tests for save_checkpoint + load_checkpoint roundtrip."""

    def test_save_load_roundtrip(self):
        """save → load → model produces same output."""
        from transformer.utils.checkpoint import save_checkpoint, load_checkpoint
        model, config = _make_model_and_config()
        model.eval()
        x = torch.randint(0, 100, (1, 8))

        with torch.no_grad():
            out_before = model(x)

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            save_path = f.name

        try:
            save_checkpoint(model, None, config, epoch=5, step=1000,
                            save_path=save_path)
            ckpt = load_checkpoint(save_path)
            assert 'model_state_dict' in ckpt
            assert 'config' in ckpt

            # Reload into fresh model
            from transformer.core.model import GaugeTransformerLM
            model2 = GaugeTransformerLM(ckpt['config'])
            model2.load_state_dict(ckpt['model_state_dict'])
            model2.eval()

            with torch.no_grad():
                out_after = model2(x)

            assert torch.allclose(out_before, out_after, atol=1e-5), \
                f"Output mismatch after roundtrip: max diff {(out_before - out_after).abs().max():.2e}"
        finally:
            os.unlink(save_path)

    def test_config_preserved(self):
        """Config dict survives save/load."""
        from transformer.utils.checkpoint import save_checkpoint, load_checkpoint
        model, config = _make_model_and_config()

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            save_path = f.name

        try:
            save_checkpoint(model, None, config, epoch=3, step=500,
                            save_path=save_path)
            ckpt = load_checkpoint(save_path)
            for key in ['vocab_size', 'embed_dim', 'n_layers']:
                assert ckpt['config'][key] == config[key], \
                    f"Config key {key} mismatch"
        finally:
            os.unlink(save_path)

    def test_epoch_step_preserved(self):
        """Epoch and step metadata correct."""
        from transformer.utils.checkpoint import save_checkpoint, load_checkpoint
        model, config = _make_model_and_config()

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            save_path = f.name

        try:
            save_checkpoint(model, None, config, epoch=7, step=2000,
                            save_path=save_path)
            ckpt = load_checkpoint(save_path)
            assert ckpt['epoch'] == 7
            assert ckpt['step'] == 2000
        finally:
            os.unlink(save_path)

    def test_extra_kwargs_preserved(self):
        """Extra kwargs (e.g. best_val_loss) are saved."""
        from transformer.utils.checkpoint import save_checkpoint, load_checkpoint
        model, config = _make_model_and_config()

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            save_path = f.name

        try:
            save_checkpoint(model, None, config, epoch=1, step=100,
                            save_path=save_path, best_val_loss=2.5)
            ckpt = load_checkpoint(save_path)
            assert ckpt['best_val_loss'] == 2.5
        finally:
            os.unlink(save_path)

    def test_file_not_found_raises(self):
        """Loading nonexistent checkpoint raises FileNotFoundError."""
        from transformer.utils.checkpoint import load_checkpoint
        with pytest.raises(FileNotFoundError):
            load_checkpoint('/nonexistent/path/checkpoint.pt')


# ===========================================================================
# TestCheckpointInfo
# ===========================================================================

class TestCheckpointInfo:
    """Tests for load_checkpoint_info()."""

    def test_info_fields(self):
        """Info dict has expected keys."""
        from transformer.utils.checkpoint import save_checkpoint, load_checkpoint_info
        model, config = _make_model_and_config()

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            save_path = f.name

        try:
            save_checkpoint(model, None, config, epoch=2, step=300,
                            save_path=save_path)
            info = load_checkpoint_info(save_path)
            assert info['epoch'] == 2
            assert info['step'] == 300
            assert 'config' in info
            # Note: save_checkpoint stores 'optimizer_state_dict': None,
            # so the key exists but value is None
            assert 'has_optimizer' in info
            assert 'n_parameters' in info
            assert info['n_parameters'] > 0
        finally:
            os.unlink(save_path)
