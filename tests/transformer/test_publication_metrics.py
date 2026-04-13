"""
Tests for transformer.analysis.publication_metrics (data structures)
=====================================================================

Validates TrainingSnapshot, ExperimentResult, and TrainingTracker
data structures and their serialization.
"""

import math
import pytest
import tempfile
from pathlib import Path

from transformer.analysis.publication_metrics import (
    TrainingSnapshot,
    ExperimentResult,
    TrainingTracker,
)


# =============================================================================
# TestTrainingSnapshot
# =============================================================================

class TestTrainingSnapshot:
    """TrainingSnapshot dataclass."""

    def test_creation(self):
        """Dataclass instantiates with required fields."""
        snap = TrainingSnapshot(
            step=10, epoch=0.5,
            train_loss=5.0, train_ce=4.5,
            train_ppl=90.0, train_bpc=6.5,
        )
        assert snap.step == 10
        assert snap.train_ce == 4.5

    def test_to_dict(self):
        """to_dict returns serializable dictionary."""
        snap = TrainingSnapshot(
            step=10, epoch=0.5,
            train_loss=5.0, train_ce=4.5,
            train_ppl=90.0, train_bpc=6.5,
        )
        d = snap.to_dict()
        assert isinstance(d, dict)
        assert d['step'] == 10
        assert d['train_ce'] == 4.5

    def test_default_values(self):
        """Optional fields have defaults."""
        snap = TrainingSnapshot(
            step=0, epoch=0.0,
            train_loss=0.0, train_ce=0.0,
            train_ppl=0.0, train_bpc=0.0,
        )
        assert snap.val_loss is None
        assert snap.grad_norm_total == 0.0


# =============================================================================
# TestExperimentResult
# =============================================================================

class TestExperimentResult:
    """ExperimentResult dataclass."""

    def test_creation(self):
        """Instantiates with all required fields."""
        result = ExperimentResult(
            name='test_run',
            config={'embed_dim': 64},
            final_val_ppl=50.0,
            final_val_bpc=5.5,
            best_val_ppl=45.0,
            total_params=100000,
            training_time=3600.0,
            tokens_per_sec=5000.0,
        )
        assert result.name == 'test_run'
        assert result.best_val_ppl == 45.0

    def test_to_dict(self):
        """to_dict returns serializable dict."""
        result = ExperimentResult(
            name='test', config={},
            final_val_ppl=50.0, final_val_bpc=5.5,
            best_val_ppl=45.0, total_params=100000,
            training_time=3600.0, tokens_per_sec=5000.0,
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d['name'] == 'test'


# =============================================================================
# TestTrainingTracker
# =============================================================================

class TestTrainingTracker:
    """TrainingTracker: step recording and CSV output."""

    def test_record_snapshot(self):
        """record() adds TrainingSnapshot to history."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tracker = TrainingTracker(save_dir=tmp_dir)
            tracker.record(
                step=1, epoch=0.1,
                train_metrics={'loss': 5.0, 'ce_loss': 4.5},
                step_time=0.1, batch_size=8, seq_len=32,
            )
            assert len(tracker.history) == 1
            assert tracker.history[0].step == 1

    def test_bpc_computation(self):
        """BPC = CE / ln(2)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tracker = TrainingTracker(save_dir=tmp_dir)
            ce = 3.0
            tracker.record(
                step=1, epoch=0.1,
                train_metrics={'loss': 3.0, 'ce_loss': ce},
                step_time=0.1, batch_size=8, seq_len=32,
            )
            expected_bpc = ce / math.log(2)
            assert abs(tracker.history[0].train_bpc - expected_bpc) < 1e-4

    def test_ppl_computation(self):
        """PPL = exp(min(CE, 20))."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tracker = TrainingTracker(save_dir=tmp_dir)
            ce = 3.0
            tracker.record(
                step=1, epoch=0.1,
                train_metrics={'loss': 3.0, 'ce_loss': ce},
                step_time=0.1, batch_size=8, seq_len=32,
            )
            expected_ppl = math.exp(min(ce, 20))
            assert abs(tracker.history[0].train_ppl - expected_ppl) < 1e-2
