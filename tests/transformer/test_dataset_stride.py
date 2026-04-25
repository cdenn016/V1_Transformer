"""Tests for stride-based windowing + random_offset_per_epoch in the dataset
factory layer. Non-negotiable invariant: stride=None preserves legacy behavior
bit-identically (no new RNG consumption, identical num_sequences formula,
identical __getitem__ output)."""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import Dataset

from transformer.data.datasets import (
    WikiText2Dataset,
    WikiText2TiktokenDataset,
    WikiText2CharDataset,
    WikiText2ByteDataset,
    create_dataloaders,
    create_char_dataloaders,
)


# --------------------------------------------------------------------------- #
# Pure-formula tests (no data loading required)
# --------------------------------------------------------------------------- #

def _compute_num_sequences(N: int, T: int, stride):
    """Re-implementation of the formula inside each dataset class."""
    if stride is None:
        return max(1, N - T)
    S = int(stride)
    assert S >= 1
    k_max = max(0, (N - T - S) // S)
    return max(1, k_max + 1)


class TestNumSequencesTable:
    """Lock down the num_sequences formula across the stride parameter."""

    @pytest.mark.parametrize("N,T,stride,expected", [
        (1000, 64, None, 936),   # legacy: max(1, N - T)
        (1000, 64, 1,   936),    # stride=1: (N-T-1)//1 + 1 = 935 + 1 = 936
        (1000, 64, 64,  14),     # stride=T: (1000-64-64)//64 + 1 = 13 + 1
        (128,  64, 64,  1),      # degenerate: (128-64-64)//64 + 1 = 0 + 1
        (64,   64, None, 1),     # N == T legacy: max(1, 0) = 1
        (64,   64, 64,   1),     # N == T strided: clamped to 1
    ])
    def test_formula(self, N, T, stride, expected):
        assert _compute_num_sequences(N, T, stride) == expected


# --------------------------------------------------------------------------- #
# set_epoch contract tests (no data loading)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("cls", [
    WikiText2TiktokenDataset,
    WikiText2Dataset,
    WikiText2CharDataset,
    WikiText2ByteDataset,
])
class TestSetEpochContract:
    """Each dataset class's set_epoch must:
    - be a no-op when random_offset_per_epoch=False (zero RNG consumed)
    - consume ZERO global torch RNG even when flag=True (uses isolated Generator)
    - produce deterministic offsets given fixed base_epoch_seed
    """

    def _mock(self, cls):
        inst = cls.__new__(cls)
        inst.random_offset_per_epoch = False
        inst.stride = None
        inst.base_epoch_seed = 0
        inst._epoch_offset = 0
        return inst

    def test_flag_false_zero_rng(self, cls):
        inst = self._mock(cls)
        torch.manual_seed(6)
        state_before = torch.get_rng_state()
        inst.set_epoch(0)
        inst.set_epoch(1)
        inst.set_epoch(42)
        assert torch.equal(torch.get_rng_state(), state_before)
        assert inst._epoch_offset == 0

    def test_flag_true_preserves_global_rng(self, cls):
        inst = self._mock(cls)
        inst.random_offset_per_epoch = True
        inst.stride = 64
        inst.base_epoch_seed = 6

        torch.manual_seed(6)
        state_before = torch.get_rng_state()
        inst.set_epoch(0)
        inst.set_epoch(1)
        inst.set_epoch(5)
        # Isolated Generator must not leak into global RNG state.
        assert torch.equal(torch.get_rng_state(), state_before)
        assert 0 <= inst._epoch_offset < 64

    def test_deterministic_across_runs(self, cls):
        inst_a = self._mock(cls)
        inst_a.random_offset_per_epoch = True
        inst_a.stride = 64
        inst_a.base_epoch_seed = 123
        inst_a.set_epoch(7)

        inst_b = self._mock(cls)
        inst_b.random_offset_per_epoch = True
        inst_b.stride = 64
        inst_b.base_epoch_seed = 123
        inst_b.set_epoch(7)

        assert inst_a._epoch_offset == inst_b._epoch_offset

    def test_different_epochs_differ(self, cls):
        """With high probability, different epochs produce different offsets.
        (Pigeonhole: across 10 epochs with stride=64, collision rate is low.)"""
        inst = self._mock(cls)
        inst.random_offset_per_epoch = True
        inst.stride = 64
        inst.base_epoch_seed = 42

        offsets = set()
        for epoch in range(10):
            inst.set_epoch(epoch)
            offsets.add(inst._epoch_offset)
        # At least 3 distinct offsets across 10 epochs (very loose bound).
        assert len(offsets) >= 3


# --------------------------------------------------------------------------- #
# End-to-end behavior tests using a minimal synthetic dataset
# --------------------------------------------------------------------------- #

class _SyntheticStridedDataset(Dataset):
    """Minimal dataset mirroring the stride/set_epoch contract. Used to
    exercise end-to-end behavior (num_sequences, __getitem__, set_epoch)
    without requiring WikiText data files."""

    def __init__(self, N: int, T: int, stride=None,
                 random_offset_per_epoch: bool = False,
                 base_epoch_seed: int = 0):
        self.tokens = torch.arange(N, dtype=torch.long)
        self.max_seq_len = T
        self.stride = stride
        self.random_offset_per_epoch = (
            bool(random_offset_per_epoch) and (stride is not None) and (stride > 1)
        )
        self.base_epoch_seed = int(base_epoch_seed)
        self._epoch_offset = 0
        if self.stride is None:
            self.num_sequences = max(1, N - T)
        else:
            S = int(self.stride)
            k_max = max(0, (N - T - S) // S)
            self.num_sequences = max(1, k_max + 1)

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        if self.stride is None:
            start = idx
        else:
            start = self._epoch_offset + idx * int(self.stride)
        return self.tokens[start : start + self.max_seq_len]

    def set_epoch(self, epoch: int) -> None:
        if not self.random_offset_per_epoch:
            return
        g = torch.Generator()
        g.manual_seed(int(self.base_epoch_seed) + int(epoch))
        self._epoch_offset = int(torch.randint(
            low=0, high=int(self.stride), size=(1,), generator=g,
        ).item())


class TestStrideNoneIdentity:
    """stride=None must produce identical indexing as the legacy formula."""

    def test_num_sequences(self):
        ds = _SyntheticStridedDataset(N=1000, T=64, stride=None)
        assert len(ds) == 1000 - 64

    def test_item_matches_legacy(self):
        N, T = 1000, 64
        ds = _SyntheticStridedDataset(N=N, T=T, stride=None)
        for i in [0, 1, 100, len(ds) - 1]:
            expected = torch.arange(i, i + T, dtype=torch.long)
            assert torch.equal(ds[i], expected)


class TestStrideTReproducibility:
    """stride=T with fixed base_seed and random_offset=True must be
    reproducible across independent constructions."""

    def test_offset_stream_reproducible(self):
        ds_a = _SyntheticStridedDataset(
            N=10_000, T=64, stride=64,
            random_offset_per_epoch=True, base_epoch_seed=6,
        )
        ds_b = _SyntheticStridedDataset(
            N=10_000, T=64, stride=64,
            random_offset_per_epoch=True, base_epoch_seed=6,
        )
        for epoch in range(5):
            ds_a.set_epoch(epoch)
            ds_b.set_epoch(epoch)
            assert ds_a._epoch_offset == ds_b._epoch_offset

    def test_batch_data_matches_under_fixed_offset(self):
        ds = _SyntheticStridedDataset(N=10_000, T=64, stride=64)
        # offset=0 default at construction
        item0 = ds[0].clone()
        item1 = ds[1].clone()
        assert torch.equal(item0, torch.arange(0, 64))
        assert torch.equal(item1, torch.arange(64, 128))


class TestWorstCaseOffsetBounds:
    """Last idx × last offset must not index past the token tensor."""

    def test_no_padding_at_worst_case_offset(self):
        N, T, S = 10_000, 64, 64
        ds = _SyntheticStridedDataset(
            N=N, T=T, stride=S,
            random_offset_per_epoch=True, base_epoch_seed=0,
        )
        # Force worst-case offset.
        ds._epoch_offset = S - 1
        last = ds[len(ds) - 1]
        assert last.shape == (T,), "worst-case offset produced short window"
        start_idx = ds._epoch_offset + (len(ds) - 1) * S
        # start + T + 1 must fit within N (for target: tokens[start+1:start+T+1])
        assert start_idx + T + 1 <= N, (
            f"start_idx + T + 1 = {start_idx + T + 1} > N = {N}"
        )


class TestEpochZeroUnderOptionB:
    """Option B: epoch 0 draws from the generator seeded by base_epoch_seed+0.
    Cross-seed first-epoch trajectories are NOT artificially correlated."""

    def test_epoch_zero_depends_on_base_seed(self):
        ds_a = _SyntheticStridedDataset(
            N=10_000, T=64, stride=64,
            random_offset_per_epoch=True, base_epoch_seed=6,
        )
        ds_b = _SyntheticStridedDataset(
            N=10_000, T=64, stride=64,
            random_offset_per_epoch=True, base_epoch_seed=7,
        )
        ds_a.set_epoch(0)
        ds_b.set_epoch(0)
        # Different seeds at epoch 0 should (almost surely) give different
        # offsets. For stride=64, collision probability is 1/64.
        # We seed with 6 and 7 — known distinct via test run.
        assert ds_a._epoch_offset != ds_b._epoch_offset


# --------------------------------------------------------------------------- #
# Factory assertion tests (no data loading)
# --------------------------------------------------------------------------- #

class TestFactoryAssertion:
    """create_dataloaders / create_char_dataloaders must reject
    stride-active + num_workers>0 before constructing any dataset."""

    def test_create_dataloaders_rejects_workers_with_stride(self):
        with pytest.raises(ValueError, match="num_workers=0"):
            create_dataloaders(
                max_seq_len=64, batch_size=64, num_workers=2,
                stride=64,
            )

    def test_create_dataloaders_rejects_workers_with_eval_stride(self):
        with pytest.raises(ValueError, match="num_workers=0"):
            create_dataloaders(
                max_seq_len=64, batch_size=64, num_workers=2,
                eval_stride=64,
            )

    def test_create_char_dataloaders_rejects_workers_with_stride(self):
        with pytest.raises(ValueError, match="num_workers=0"):
            create_char_dataloaders(
                max_seq_len=32, batch_size=16, num_workers=2,
                stride=32,
            )
