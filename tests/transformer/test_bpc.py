"""Tests for transformer.training.bpc."""
from __future__ import annotations

import math

import pytest

from transformer.training.bpc import (
    bpc_from_dataset,
    compute_bpc,
    tokens_per_char_from_dataset,
)


class _StubDataset:
    def __init__(self, tpc):
        self.tokens_per_char = tpc


class _StubLoader:
    def __init__(self, dataset):
        self.dataset = dataset


def test_compute_bpc_with_tpc():
    # CE = ln(2) nats per token, tpc = 0.25 -> 1 bit per token, 0.25 bits per char.
    assert compute_bpc(math.log(2.0), 0.25) == pytest.approx(0.25)


def test_compute_bpc_returns_bits_per_token_when_tpc_none(caplog):
    # ln 2 nats per token -> 1.0 when no correction (bits-per-token).
    val = compute_bpc(math.log(2.0), None)
    assert val == pytest.approx(1.0)


def test_compute_bpc_returns_bits_per_token_when_tpc_zero():
    val = compute_bpc(math.log(2.0), 0.0)
    assert val == pytest.approx(1.0)


def test_compute_bpc_returns_bits_per_token_when_tpc_negative():
    val = compute_bpc(math.log(2.0), -0.1)
    assert val == pytest.approx(1.0)


def test_compute_bpc_handles_none_input():
    assert math.isnan(compute_bpc(None, 0.5))


def test_japanese_correction_magnitude():
    """cl100k_base on Japanese: ~1.1 tok/char -> BPC ≈ bits-per-token * 1.1."""
    ce = 5.32 * math.log(2.0)  # roughly the wiki-ja BPC reported
    val = compute_bpc(ce, 1.1)
    assert val == pytest.approx(5.32 * 1.1, rel=1e-6)


def test_english_correction_magnitude():
    """GPT-2 BPE on English: ~0.23 tok/char -> BPC much smaller than bits-per-token."""
    ce = 5.0 * math.log(2.0)  # 5 bits per token
    val = compute_bpc(ce, 0.23)
    assert val == pytest.approx(5.0 * 0.23, rel=1e-6)


def test_tokens_per_char_from_dataset_direct():
    ds = _StubDataset(0.5)
    assert tokens_per_char_from_dataset(ds) == 0.5


def test_tokens_per_char_from_loader_unwraps():
    ds = _StubDataset(0.42)
    loader = _StubLoader(ds)
    assert tokens_per_char_from_dataset(loader) == 0.42


def test_tokens_per_char_from_dataset_none_when_missing():
    class _NoTPC:
        pass
    assert tokens_per_char_from_dataset(_NoTPC()) is None


def test_tokens_per_char_returns_none_for_none_input():
    assert tokens_per_char_from_dataset(None) is None


def test_bpc_from_dataset_uses_loader():
    loader = _StubLoader(_StubDataset(0.25))
    val = bpc_from_dataset(math.log(2.0), loader)
    assert val == pytest.approx(0.25)


def test_fallback_warning_keyed_per_call_site(caplog):
    import logging
    from transformer.training import bpc as bpc_mod
    bpc_mod._FALLBACK_WARNED.clear()  # reset for this test
    with caplog.at_level(logging.WARNING, logger=bpc_mod.logger.name):
        compute_bpc(1.0, None, fallback_key='site_A')
        compute_bpc(1.0, None, fallback_key='site_A')  # second call same key — should not log
        compute_bpc(1.0, None, fallback_key='site_B')
    warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    # Two unique keys -> two warnings; repeated key suppressed.
    assert len(warning_messages) == 2
