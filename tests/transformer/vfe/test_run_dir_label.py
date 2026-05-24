"""The run-directory gauge-group tag must report GL(d_h) — the per-head block
dimension — not GL(embed_dim). A 2-head GL config of head-dim 10 (embed_dim=20)
must label as GL(10), not GL(20) (bug report 2026-05-24)."""
import pytest

from transformer.vfe.config import VFEConfig
from transformer.vfe.trainer import _gauge_group_label


def test_glk_label_uses_head_dim_not_embed_dim():
    cfg = VFEConfig(embed_dim=20, irrep_spec=[("fund", 2, 10)], gauge_group="GLK")
    assert _gauge_group_label(cfg) == "GL(10)"


def test_glk_label_single_head_equals_embed_dim():
    cfg = VFEConfig(embed_dim=16, irrep_spec=[("fund", 1, 16)], gauge_group="GLK")
    assert _gauge_group_label(cfg) == "GL(16)"


def test_glk_label_mixed_block_dims():
    # embed_dim = 2*1 + 1*3 = 5; block dims {1, 3}.
    cfg = VFEConfig(embed_dim=5, irrep_spec=[("l0", 2, 1), ("l1", 1, 3)],
                    gauge_group="GLK")
    assert _gauge_group_label(cfg) == "GL(1,3)"


def test_son_label_uses_embed_dim():
    cfg = VFEConfig(embed_dim=12, irrep_spec=[("fund", 1, 12)], gauge_group="SON")
    assert _gauge_group_label(cfg) == "SO(12)"


def test_so3_label():
    cfg = VFEConfig(embed_dim=3, irrep_spec=[("l1", 1, 3)], gauge_group="SO3")
    assert _gauge_group_label(cfg) == "SO(3)"
