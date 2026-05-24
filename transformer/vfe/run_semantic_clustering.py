r"""Click-to-run entry point for VFE semantic-clustering visualization.

Edit the ``CONFIG`` dict below, then run this file (no CLI arguments, per the
project's click-to-run convention). It loads a trained ``VFEModel`` (or builds a
fresh one for a pipeline smoke run when ``checkpoint_path`` is ``None``) and
produces, for the requested views, four separate publication-quality figures
(``mu_clustering``, ``sigma_clustering``, ``phi_vector_clustering``,
``omega_clustering``) plus ``metrics.json`` / ``metrics.csv`` sidecars.

Checkpoint note: VFE checkpoints store only ``model_state_dict`` (the trainer's
``best`` path stores a bare ``state_dict``); the architecture config is NOT in
the ``.pt``. You must therefore set ``model_config`` below to match the trained
run (the same keys you used in ``train_vfe.py`` / the run's ``system_info.json``).
A mismatch will surface as a ``load_state_dict`` error.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import torch

from transformer.vfe.config import VFEConfig
from transformer.vfe.model import VFEModel
from transformer.vfe.semantic_clustering.pipeline import run_clustering


# =============================================================================
# CONFIG — edit these values, then press Run.
# =============================================================================
CONFIG = {
    # Path to a trained checkpoint (.pt). None => fresh untrained model
    # (pipeline smoke run only; clusters will be near-random).
    "checkpoint_path": None,

    # Must match the trained checkpoint's architecture. Mirrors train_vfe.py.
    "model_config": {
        "vocab_size": 50257,
        "embed_dim": 200,
        "irrep_spec": [("fund", 20, 10)],
        "diagonal_covariance": True,
        "n_layers": 1,
        "use_prior_bank": False,
    },

    # Output root. None => <checkpoint_dir>/semantic_clustering, or
    # ./outputs/semantic_clustering when no checkpoint.
    "output_dir": None,

    "do_contextual": True,
    "do_vocab": True,
    "layer": "final",                       # 'final' or an int layer index
    "methods": {"projection": "umap", "clustering": "agglomerative"},
    "max_points": 200,                      # cap on tokens clustered (O(n^2))
    "seed": 0,

    # Contextual view inputs.
    "text": None,                           # raw text to encode for the context view
    "tokenizer": "cl100k_base",             # tiktoken encoding name, or None
    "n_context_tokens": 128,                # used when text is None (random ids)
    "seq_len": 32,

    # Vocab view inputs.
    "vocab_sample": 256,                    # number of token types to cluster
}


class _TiktokenDataset:
    """Minimal adapter exposing ``decode`` so the pipeline can label tokens."""

    def __init__(self, enc):
        self._enc = enc

    def decode(self, ids) -> str:
        return self._enc.decode(list(ids))


def _load_model(checkpoint_path: Optional[str], model_config: dict) -> VFEModel:
    """Build a ``VFEModel`` from ``model_config`` and optionally load weights.

    Handles both checkpoint layouts the trainer writes: a dict with a
    ``model_state_dict`` key, and a bare ``state_dict`` (the ``best`` path).
    """
    cfg = VFEConfig(**model_config)
    model = VFEModel(cfg)
    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state)
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        warnings.warn(
            "checkpoint_path is None: using a fresh untrained model. "
            "Clusters will be near-random — for a real analysis set a checkpoint."
        )
    model.eval()
    return model


def _resolve_output_root(config: dict) -> Path:
    if config["output_dir"] is not None:
        return Path(config["output_dir"])
    if config["checkpoint_path"] is not None:
        return Path(config["checkpoint_path"]).parent / "semantic_clustering"
    return Path("./outputs/semantic_clustering")


def _make_context_ids(config: dict, cfg: VFEConfig):
    """Build ``(B, N)`` context token ids and an optional decode dataset.

    Prefers encoding ``config['text']`` with the named tiktoken encoding;
    otherwise falls back to random ids (warned — not semantically meaningful).
    """
    text = config["text"]
    tok_name = config["tokenizer"]
    if text is not None and tok_name is not None:
        try:
            import tiktoken

            enc = tiktoken.get_encoding(tok_name)
            ids = enc.encode(text)
            ids = [i for i in ids if i < cfg.vocab_size]
            seq_len = config["seq_len"]
            n_full = (len(ids) // seq_len) * seq_len
            if n_full < seq_len:
                raise ValueError("text too short for one sequence")
            arr = torch.tensor(ids[:n_full], dtype=torch.long).reshape(-1, seq_len)
            return arr, _TiktokenDataset(enc)
        except Exception as exc:  # noqa: BLE001 - fall back to random ids
            warnings.warn(f"tokenizer/text path failed ({exc!r}); using random ids.")

    warnings.warn(
        "No text/tokenizer supplied for the contextual view: using random token "
        "ids. The resulting clusters are NOT semantically meaningful."
    )
    n = config["n_context_tokens"]
    seq_len = config["seq_len"]
    rows = max(1, n // seq_len)
    g = torch.Generator().manual_seed(config["seed"])
    return torch.randint(0, cfg.vocab_size, (rows, seq_len), generator=g), None


def main() -> None:
    config = CONFIG
    model = _load_model(config["checkpoint_path"], config["model_config"])
    cfg = model.cfg
    root = _resolve_output_root(config)

    if config["do_vocab"]:
        n = min(config["vocab_sample"], cfg.vocab_size)
        vocab_ids = torch.arange(n, dtype=torch.long)
        out = run_clustering(
            model,
            source="vocab",
            layer="final",
            token_ids=vocab_ids,
            dataset=_vocab_dataset(config, cfg),
            methods=config["methods"],
            outdir=root / "vocab",
            max_points=config["max_points"],
            seed=config["seed"],
        )
        print(f"[vocab]      wrote {root / 'vocab'} | "
              f"sigma eff-rank={out['sigma'].get('effective_rank_mean'):.3f}")

    if config["do_contextual"]:
        ctx_ids, dataset = _make_context_ids(config, cfg)
        out = run_clustering(
            model,
            source="contextual",
            layer=config["layer"],
            token_ids=ctx_ids,
            dataset=dataset,
            methods=config["methods"],
            outdir=root / "contextual",
            max_points=config["max_points"],
            seed=config["seed"],
        )
        print(f"[contextual] wrote {root / 'contextual'} | "
              f"n_tokens={out['meta']['n_tokens']}")

    print(f"Done. Figures + metrics under: {root}")


def _vocab_dataset(config: dict, cfg: VFEConfig):
    """A decode adapter for vocab-view labels, when a tokenizer is configured."""
    tok_name = config["tokenizer"]
    if tok_name is None:
        return None
    try:
        import tiktoken

        return _TiktokenDataset(tiktoken.get_encoding(tok_name))
    except Exception:  # noqa: BLE001 - labels are optional
        return None


if __name__ == "__main__":
    main()
