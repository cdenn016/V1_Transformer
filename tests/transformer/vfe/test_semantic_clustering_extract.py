import torch
from transformer.vfe.config import VFEConfig
from transformer.vfe.model import VFEModel
from transformer.vfe.semantic_clustering.extract import extract_contextual, extract_vocab


def _tiny_model():
    cfg = VFEConfig(
        vocab_size=64, embed_dim=8, irrep_spec=[("fund", 2, 4)],
        diagonal_covariance=True, n_layers=1,
    )
    return VFEModel(cfg), cfg


def test_contextual_final_shapes():
    model, cfg = _tiny_model()
    model.eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 5))
    b = extract_contextual(model, ids, layer="final")
    assert b.mu.shape == (10, cfg.embed_dim)
    assert b.sigma.shape == (10, cfg.embed_dim)   # diagonal active
    assert b.phi.shape[0] == 10
    assert b.source == "contextual"


def test_vocab_shapes():
    model, cfg = _tiny_model()
    model.eval()
    b = extract_vocab(model, token_ids=torch.arange(20))
    assert b.mu.shape == (20, cfg.embed_dim)
    assert b.phi.shape[0] == 20
    assert b.source == "vocab"
