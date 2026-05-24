from pathlib import Path

import torch

from transformer.vfe.config import VFEConfig
from transformer.vfe.model import VFEModel
from transformer.vfe.train_vfe import run_post_training_semantic_clustering


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


def test_hook_writes_both_views(tmp_path):
    model = _tiny_model()
    # Fake loader: a list with one (input_ids, target_ids) batch, mirroring the
    # tuple format the real dataloaders yield. The hook must use inputs only.
    loader = [(torch.randint(0, 64, (2, 6)), torch.randint(0, 64, (2, 6)))]

    sc_root = run_post_training_semantic_clustering(
        model, str(tmp_path), loader, dataset=None,
        device="cpu", max_points=20, seed=0,
    )

    for view in ["vocab", "contextual"]:
        for name in [
            "mu_clustering",
            "sigma_clustering",
            "phi_vector_clustering",
            "omega_clustering",
        ]:
            assert (Path(sc_root) / view / f"{name}.png").exists(), (
                f"missing {view}/{name}.png"
            )
        assert (Path(sc_root) / view / "metrics.json").exists()
