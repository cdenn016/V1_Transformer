"""
Tests for transformer/analysis/trajectory.py
==============================================

Tests trajectory recording infrastructure: LayerTrajectory, ForwardTrajectory,
TrajectoryRecorder, and global recorder management.
"""

import pytest
import numpy as np


# ===========================================================================
# TestLayerTrajectory
# ===========================================================================

class TestLayerTrajectory:
    """Tests for LayerTrajectory dataclass."""

    def test_creation(self):
        """LayerTrajectory can be created with required fields."""
        from transformer.analysis.trajectory import LayerTrajectory
        B, N, K, phi_dim = 2, 4, 6, 3
        lt = LayerTrajectory(
            layer_idx=0,
            mu_in=np.random.randn(B, N, K),
            Sigma_diag_in=np.random.rand(B, N, K),
            phi_in=np.random.randn(B, N, phi_dim),
            mu_out=np.random.randn(B, N, K),
            Sigma_diag_out=np.random.rand(B, N, K),
            phi_out=np.random.randn(B, N, phi_dim),
        )
        assert lt.layer_idx == 0
        assert lt.mu_in.shape == (B, N, K)

    def test_to_dict(self):
        """to_dict returns JSON-serializable dictionary."""
        from transformer.analysis.trajectory import LayerTrajectory
        lt = LayerTrajectory(
            layer_idx=1,
            mu_in=np.zeros((1, 2, 3)),
            Sigma_diag_in=np.ones((1, 2, 3)),
            phi_in=np.zeros((1, 2, 2)),
            mu_out=np.zeros((1, 2, 3)),
            Sigma_diag_out=np.ones((1, 2, 3)),
            phi_out=np.zeros((1, 2, 2)),
        )
        d = lt.to_dict()
        assert d['layer_idx'] == 1
        assert 'mu_in' in d
        assert 'mu_out' in d


# ===========================================================================
# TestForwardTrajectory
# ===========================================================================

class TestForwardTrajectory:
    """Tests for ForwardTrajectory dataclass."""

    def test_creation(self):
        """ForwardTrajectory can be created."""
        from transformer.analysis.trajectory import ForwardTrajectory
        B, N, K, phi_dim = 1, 4, 6, 3
        ft = ForwardTrajectory(
            batch_size=B,
            seq_len=N,
            mu_embed=np.random.randn(B, N, K),
            Sigma_diag_embed=np.random.rand(B, N, K),
            phi_embed=np.random.randn(B, N, phi_dim),
        )
        assert ft.batch_size == 1
        assert ft.seq_len == 4
        assert len(ft.layer_trajectories) == 0

    def test_add_layer_trajectory(self):
        """Can add LayerTrajectory objects."""
        from transformer.analysis.trajectory import ForwardTrajectory, LayerTrajectory
        B, N, K, phi_dim = 1, 4, 6, 3
        ft = ForwardTrajectory(
            batch_size=B, seq_len=N,
            mu_embed=np.zeros((B, N, K)),
            Sigma_diag_embed=np.ones((B, N, K)),
            phi_embed=np.zeros((B, N, phi_dim)),
        )
        lt = LayerTrajectory(
            layer_idx=0,
            mu_in=np.zeros((B, N, K)),
            Sigma_diag_in=np.ones((B, N, K)),
            phi_in=np.zeros((B, N, phi_dim)),
            mu_out=np.zeros((B, N, K)),
            Sigma_diag_out=np.ones((B, N, K)),
            phi_out=np.zeros((B, N, phi_dim)),
        )
        ft.layer_trajectories.append(lt)
        assert len(ft.layer_trajectories) == 1


# ===========================================================================
# TestTrajectoryRecorder
# ===========================================================================

class TestTrajectoryRecorder:
    """Tests for TrajectoryRecorder class."""

    def test_recorder_creation(self):
        """TrajectoryRecorder can be instantiated."""
        from transformer.analysis.trajectory import TrajectoryRecorder
        rec = TrajectoryRecorder(enabled=True)
        assert rec is not None
        assert rec.enabled is True

    def test_recorder_disabled_by_default(self):
        """TrajectoryRecorder is disabled by default."""
        from transformer.analysis.trajectory import TrajectoryRecorder
        rec = TrajectoryRecorder()
        assert rec.enabled is False

    def test_global_recorder_set_get(self):
        """set/get global recorder works."""
        from transformer.analysis.trajectory import (
            get_global_recorder, set_global_recorder, TrajectoryRecorder,
        )
        orig = get_global_recorder()
        try:
            rec = TrajectoryRecorder(enabled=True)
            set_global_recorder(rec)
            assert get_global_recorder() is rec
        finally:
            set_global_recorder(orig)

    def test_enable_disable_tracking(self):
        """enable/disable trajectory tracking functions work."""
        from transformer.analysis.trajectory import (
            get_global_recorder, set_global_recorder,
            enable_trajectory_tracking, disable_trajectory_tracking,
        )
        orig = get_global_recorder()
        try:
            enable_trajectory_tracking()
            rec = get_global_recorder()
            assert rec is not None
            assert rec.enabled is True

            disable_trajectory_tracking()
            rec = get_global_recorder()
            assert rec is not None
            assert rec.enabled is False
        finally:
            set_global_recorder(orig)

    def test_disabled_recorder_skips_recording(self):
        """Disabled recorder's start_forward is a no-op."""
        from transformer.analysis.trajectory import TrajectoryRecorder
        rec = TrajectoryRecorder(enabled=False)
        # start_forward on a disabled recorder should not crash
        rec.start_forward(batch_size=2, seq_len=4, ffn_mode='VFE_dynamic')
        assert rec._current_forward is None
