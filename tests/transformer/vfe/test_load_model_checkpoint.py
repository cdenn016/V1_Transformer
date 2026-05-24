"""Regression guard for the hardened checkpoint loader (audit 2026-05-24).

``run_semantic_clustering._load_model`` now loads with ``weights_only=True``.
A trainer-format checkpoint (only tensors / dicts / numbers) must still
round-trip cleanly under the stricter flag.
"""
from pathlib import Path

import torch

from transformer.vfe.config import VFEConfig
from transformer.vfe.model import VFEModel
from transformer.vfe.run_semantic_clustering import _load_model


_MODEL_CONFIG = dict(
    vocab_size=64,
    embed_dim=8,
    irrep_spec=[("fund", 2, 4)],
    diagonal_covariance=True,
    n_layers=1,
)


def test_load_model_weights_only_roundtrip(tmp_path):
    model = VFEModel(VFEConfig(**_MODEL_CONFIG))
    # Mirror the trainer's checkpoint layout (trainer.py::_save_checkpoint).
    ckpt = {
        "step": 5,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {"state": {}, "param_groups": [{"lr": 1e-3}]},
        "scheduler_state_dict": {"last_epoch": 5},
    }
    path = Path(tmp_path) / "step_5.pt"
    torch.save(ckpt, path)

    loaded = _load_model(str(path), _MODEL_CONFIG)

    # Weights must match the saved model bit-for-bit.
    ref = model.state_dict()
    got = loaded.state_dict()
    assert ref.keys() == got.keys()
    for k in ref:
        assert torch.equal(ref[k], got[k]), f"mismatch in {k}"
