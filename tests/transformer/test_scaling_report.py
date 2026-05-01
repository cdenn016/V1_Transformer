"""Tests for transformer.analysis.scaling_report."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from transformer.analysis.scaling_report import (
    BANNED_PHRASES,
    _lint,
    write_aggregated_csv,
    write_methods_paragraph,
)
from transformer.analysis.scaling_stats import PowerLawFit


def _stub_fit() -> PowerLawFit:
    grid = np.logspace(1, 2, 50)
    pred = 200.0 / (grid ** 0.5) + 50.0
    return PowerLawFit(
        a=200.0, b=-0.5, c=50.0,
        a_ci=(180.0, 220.0), b_ci=(-0.55, -0.45), c_ci=(48.0, 52.0),
        r_squared=0.99,
        n_bootstrap=2000, n_bootstrap_succeeded=2000,
        seeds_used=[6, 23, 111],
        axis_grid=grid, pred_mean=pred,
        pred_lo=pred * 0.95, pred_hi=pred * 1.05,
        method='nonlinear',
    )


def _stub_terminal() -> pd.DataFrame:
    rows = []
    for K in (10, 40, 80, 120):
        for seed in (6, 23, 111):
            rows.append({
                'K': float(K), 'seed': seed,
                'test_ppl': 100.0, 'val_ppl': 110.0,
                'total_params': K * 1_000_000,
                'flops_per_step': K * 1e9,
                'total_flops': K * 1e9 * 30000,
                'tokens_seen': 122_880_000, 'seq_len': 128,
            })
    return pd.DataFrame(rows)


def test_lint_passes_clean_text():
    _lint("This is a clean methods paragraph describing the protocol.")


@pytest.mark.parametrize('phrase', BANNED_PHRASES)
def test_lint_rejects_banned_phrase(phrase):
    with pytest.raises(ValueError):
        _lint(f"This text contains the phrase {phrase} somewhere.")


def test_lint_rejects_horizontal_rule():
    with pytest.raises(ValueError):
        _lint("Some text\n---\nMore text")


def test_write_aggregated_csv(tmp_path: Path):
    df = _stub_terminal()
    out = write_aggregated_csv(df, tmp_path / 'agg.csv', axis_col='K')
    assert out.exists()
    loaded = pd.read_csv(out)
    assert len(loaded) == 4  # 4 axis values
    assert 'mean_test_ppl' in loaded.columns
    assert 'n_seeds' in loaded.columns
    assert (loaded['n_seeds'] == 3).all()


def test_write_methods_paragraph_renders(tmp_path: Path):
    df = _stub_terminal()
    fit = _stub_fit()
    out = write_methods_paragraph(
        df, fit, tmp_path / 'methods.md',
        axis_name='K', axis_symbol='K', dataset='WikiText-103',
    )
    assert out.exists()
    text = out.read_text(encoding='utf-8')
    # Headline numbers must appear in the rendered paragraph
    assert 'b = -0.500' in text
    assert 'c = 50.00' in text
    assert 'R^2 = 0.990' in text
    # Methods should not contain any banned phrase
    for phrase in BANNED_PHRASES:
        assert phrase not in text.lower(), f"banned phrase leaked: {phrase!r}"
    # Word count under 250 (target ceiling per CLAUDE.md prose preference)
    word_count = len(text.split())
    assert word_count <= 260, f'unexpectedly long: {word_count} words'


def test_methods_handles_missing_seeds_clause(tmp_path: Path):
    df = _stub_terminal()
    df = df[df['K'] != 90]
    fit = _stub_fit()
    out = write_methods_paragraph(df, fit, tmp_path / 'm.md', axis_name='K')
    assert out.exists()
