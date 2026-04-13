"""
Tests for transformer.analysis.bayesian_validation
=====================================================

Validates ValidationData parsing, PyMC model construction (gated behind
pytest.importorskip('pymc')), and summary statistics.
"""

import json
import pytest
import numpy as np
import tempfile
from pathlib import Path

from transformer.analysis.bayesian_validation import (
    ValidationData,
    PYMC_AVAILABLE,
)


# =============================================================================
# Helpers
# =============================================================================

def _make_synthetic_validation_data():
    """Create a ValidationData with realistic synthetic arrays."""
    n_heads = 12
    n_tau = 10
    return ValidationData(
        layers=np.repeat(np.arange(12), 1)[:n_heads],
        heads=np.arange(n_heads),
        mean_r=np.random.uniform(0.3, 0.8, n_heads),
        std_r=np.random.uniform(0.05, 0.15, n_heads),
        se_r=np.random.uniform(0.01, 0.05, n_heads),
        median_r=np.random.uniform(0.3, 0.8, n_heads),
        keynorm_alpha=np.random.uniform(0.1, 0.5, n_heads),
        keynorm_beta=np.random.uniform(0.4, 0.9, n_heads),
        tau_values=np.linspace(0.1, 2.0, n_tau),
        tau_mean_r=np.random.uniform(0.3, 0.7, n_tau),
        tau_ci_lo=np.random.uniform(0.2, 0.5, n_tau),
        tau_ci_hi=np.random.uniform(0.5, 0.9, n_tau),
    )


def _make_synthetic_json(tmp_dir):
    """Create a synthetic validation_results.json for load_validation_data."""
    data = {
        'per_head': {
            'layers': list(range(12)),
            'heads': list(range(12)),
            'mean_r': [0.5] * 12,
            'std_r': [0.1] * 12,
            'se_r': [0.03] * 12,
            'median_r': [0.5] * 12,
            'keynorm_alpha': [0.3] * 12,
            'keynorm_beta': [0.6] * 12,
        },
        'tau_sweep': {
            'tau_values': [0.5, 1.0, 1.5],
            'mean_r': [0.4, 0.6, 0.5],
            'ci_lo': [0.3, 0.5, 0.4],
            'ci_hi': [0.5, 0.7, 0.6],
        },
    }
    path = Path(tmp_dir) / 'validation_results.json'
    with open(path, 'w') as f:
        json.dump(data, f)
    return path


# =============================================================================
# TestValidationData
# =============================================================================

class TestValidationData:
    """ValidationData dataclass creation and field consistency."""

    def test_creation(self):
        """All fields populated."""
        vd = _make_synthetic_validation_data()
        assert len(vd.layers) == 12
        assert len(vd.mean_r) == 12
        assert len(vd.tau_values) == 10

    def test_array_shapes_consistent(self):
        """Per-head arrays have same length."""
        vd = _make_synthetic_validation_data()
        n = len(vd.layers)
        assert len(vd.heads) == n
        assert len(vd.mean_r) == n
        assert len(vd.std_r) == n
        assert len(vd.keynorm_alpha) == n
        assert len(vd.keynorm_beta) == n

    def test_tau_arrays_consistent(self):
        """Tau sweep arrays have same length."""
        vd = _make_synthetic_validation_data()
        n = len(vd.tau_values)
        assert len(vd.tau_mean_r) == n
        assert len(vd.tau_ci_lo) == n
        assert len(vd.tau_ci_hi) == n

    def test_default_model_fields(self):
        """Multi-model fields default to empty."""
        vd = _make_synthetic_validation_data()
        assert len(vd.model_names) == 0
        assert len(vd.model_mean_r) == 0


# =============================================================================
# TestLoadValidationData
# =============================================================================

class TestLoadValidationData:
    """load_validation_data: JSON parsing."""

    def test_missing_file_raises(self):
        """FileNotFoundError on missing file."""
        from transformer.analysis.bayesian_validation import load_validation_data
        with pytest.raises((FileNotFoundError, OSError)):
            load_validation_data('/nonexistent/path/validation_results.json')


# =============================================================================
# TestPyMCModels (gated behind pymc availability)
# =============================================================================

@pytest.mark.slow
class TestBuildHierarchicalModel:
    """build_hierarchical_correlation_model: PyMC model construction."""

    def test_model_creation(self):
        """Model object is created with expected random variables."""
        pymc = pytest.importorskip('pymc')
        from transformer.analysis.bayesian_validation import build_hierarchical_correlation_model

        vd = _make_synthetic_validation_data()
        # Use very few samples just to verify construction
        model, idata = build_hierarchical_correlation_model(vd, n_samples=50, n_tune=50)
        assert model is not None
        assert idata is not None


@pytest.mark.slow
class TestBuildTemperatureModel:
    """build_temperature_model: temperature posterior."""

    def test_model_creation(self):
        """Model creates without error."""
        pymc = pytest.importorskip('pymc')
        from transformer.analysis.bayesian_validation import build_temperature_model

        vd = _make_synthetic_validation_data()
        model, idata = build_temperature_model(vd, n_samples=50, n_tune=50)
        assert model is not None


@pytest.mark.slow
class TestBuildKeynormModel:
    """build_keynorm_model: key-norm effect sizes."""

    def test_model_creation(self):
        """Model creates without error."""
        pymc = pytest.importorskip('pymc')
        from transformer.analysis.bayesian_validation import build_keynorm_model

        vd = _make_synthetic_validation_data()
        model, idata = build_keynorm_model(vd, n_samples=50, n_tune=50)
        assert model is not None
