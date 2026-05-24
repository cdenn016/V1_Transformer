import json
from pathlib import Path

import torch

from transformer.vfe.config import VFEConfig
from transformer.vfe.model import VFEModel
from transformer.vfe.semantic_clustering.pipeline import run_clustering


def _flatten_floats(d):
    for v in (d.values() if isinstance(d, dict) else []):
        if isinstance(v, dict):
            yield from _flatten_floats(v)
        elif isinstance(v, (int, float)):
            yield float(v)


def _tiny_model():
    cfg = VFEConfig(
        vocab_size=64,
        embed_dim=8,
        irrep_spec=[("fund", 2, 4)],
        diagonal_covariance=True,
        n_layers=1,
    )
    model = VFEModel(cfg)
    model.eval()
    return model


def test_vocab_pipeline_end_to_end(tmp_path):
    model = _tiny_model()
    out = run_clustering(
        model,
        source="vocab",
        layer="final",
        token_ids=torch.arange(40),
        dataset=None,
        methods={"projection": "umap", "clustering": "agglomerative"},
        outdir=tmp_path,
    )
    for name in [
        "mu_clustering",
        "sigma_clustering",
        "phi_vector_clustering",
        "omega_clustering",
    ]:
        assert (Path(tmp_path) / f"{name}.pdf").exists(), f"missing {name}.pdf"
        assert (Path(tmp_path) / f"{name}.png").exists(), f"missing {name}.png"
    assert (Path(tmp_path) / "metrics.json").exists()
    assert (Path(tmp_path) / "metrics.csv").exists()

    m = json.loads((Path(tmp_path) / "metrics.json").read_text())
    # no NaNs in any numeric metric (NaN != NaN)
    assert all(v == v for v in _flatten_floats(m))
    assert out["meta"]["source"] == "vocab"


def test_contextual_pipeline_end_to_end(tmp_path):
    model = _tiny_model()
    ids = torch.randint(0, 64, (2, 6))
    out = run_clustering(
        model,
        source="contextual",
        layer="final",
        token_ids=ids,
        dataset=None,
        outdir=tmp_path,
    )
    assert (Path(tmp_path) / "omega_clustering.png").exists()
    assert out["meta"]["source"] == "contextual"
    assert out["meta"]["n_tokens"] == 12
