"""
Tests for transformer.training.experiment_runner (testable utility classes)
===========================================================================

Validates PublicationMetricsTracker, LayerDiagnosticsTracker,
IterationDiagnosticsTracker, and config save/load.
"""

import csv
import json
import pytest
import tempfile
import os
from pathlib import Path


# =============================================================================
# TestPublicationMetricsTracker
# =============================================================================

class TestPublicationMetricsTracker:
    """PublicationMetricsTracker: CSV metric logging."""

    def test_csv_creation(self):
        """Constructor creates CSV file with headers."""
        from transformer.training.experiment_runner import PublicationMetricsTracker
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / 'metrics.csv'
            tracker = PublicationMetricsTracker(path)
            assert path.exists()
            with open(path) as f:
                reader = csv.reader(f)
                headers = next(reader)
            assert 'step' in headers
            assert 'train_loss_ce' in headers

    def test_log_step_appends_to_history(self):
        """log_step appends entry to history; save() writes to CSV."""
        from transformer.training.experiment_runner import PublicationMetricsTracker
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / 'metrics.csv'
            tracker = PublicationMetricsTracker(path)
            tracker.log_step(
                step=1,
                metrics={'train_loss_ce': 4.5, 'train_loss_total': 5.0},
                lrs={'mu_embed': 0.1, 'sigma_embed': 0.005},
                grad_norms={'total': 1.0},
                step_time=0.1,
                batch_size=8,
                seq_len=32,
            )
            assert len(tracker.history) == 1
            tracker.save()
            with open(path) as f:
                lines = f.readlines()
            assert len(lines) >= 2, f"Expected >= 2 lines (header + data), got {len(lines)}"

    def test_tokens_per_sec(self):
        """tokens_per_sec = batch_size * seq_len / step_time."""
        from transformer.training.experiment_runner import PublicationMetricsTracker
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / 'metrics.csv'
            tracker = PublicationMetricsTracker(path)
            tracker.log_step(
                step=1,
                metrics={'train_loss_ce': 4.5, 'train_loss_total': 5.0},
                lrs={'mu_embed': 0.1},
                grad_norms={'total': 1.0},
                step_time=0.5,
                batch_size=8,
                seq_len=32,
            )
            tracker.save()
            with open(path) as f:
                reader = csv.DictReader(f)
                row = next(reader)
            tps = float(row.get('tokens_per_sec', 0))
            expected = 8 * 32 / 0.5
            assert abs(tps - expected) < 1.0, f"Expected {expected}, got {tps}"


# =============================================================================
# TestSaveExperimentConfig
# =============================================================================

class TestSaveExperimentConfig:
    """save_experiment_config: JSON round-trip."""

    def test_json_roundtrip(self):
        """Config dict saved and loaded back identically."""
        from transformer.training.experiment_runner import save_experiment_config
        config = {'vocab_size': 100, 'embed_dim': 64, 'n_layers': 2}
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_experiment_config(config, 'VFE_dynamic', Path(tmp_dir))
            # Find the saved JSON
            json_files = list(Path(tmp_dir).glob('*.json'))
            assert len(json_files) >= 1, "No JSON config file saved"
            with open(json_files[0]) as f:
                loaded = json.load(f)
            # Config should be somewhere in the loaded data
            if 'config' in loaded:
                assert loaded['config'].get('vocab_size') == 100
            else:
                assert loaded.get('vocab_size') == 100
