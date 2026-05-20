"""
Click-to-run entry point: load a trained VFEModel and emit AIF-scored tokens.

Edit the AIF_CONFIG dict at the top of this file, then press Run. No CLI
arguments per CLAUDE.md hard constraint.

Phase 1 default (``horizon_D=1, beam_width=16``) reduces to the existing
depth-1 EFE behaviour. To exercise the Phase 2 beam-search tree (once it
ships), set ``horizon_D=2, beam_width=4``. The full-cov runtime guard in
``AIFConfig.validate_against_model`` rejects depth > 1 with full
covariance.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import sys
from typing import Optional

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from transformer.aif.config import AIFConfig
from transformer.aif.generator import AIFGenerator
from transformer.aif.preferences import build_preference

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


# === AIF generation config — click to run ===
AIF_CONFIG = {
    # Planning
    'horizon_D':            1,        # Phase 1 anchor. Phase 2 enables D=2.
    'beam_width':           16,       # Top-k candidates scored per step.
    'branching_strategy':   'beam',

    # EFE weights and sampling
    'gamma':                1.0,
    'decode_tau':           1.0,
    'epistemic_samples':    4,
    'epistemic_weight':     1.0,
    'discount':             1.0,

    # Preferences
    'preference_type':      'empirical_marginal',
    'preference_path':      None,  # SET ME — path to (V,) log-frequency tensor
    'low_entropy_beta':     1.0,

    # Habit prior
    'habit_prior_path':     None,  # None = uniform.

    # Sampling at the root
    'sampling_strategy':    'multinomial',
    'sampling_temperature': 1.0,

    # Caching
    'belief_cache_max_entries': 4096,

    # Training-time
    'training_objective':   'standard_vfe',
}

# === Checkpoint and prompt — click to run ===
CHECKPOINT_PATH: Optional[str] = None  # SET ME — path to a trained VFEModel checkpoint
PROMPT_TEXT: str = "The quick brown fox"
MAX_NEW_TOKENS: int = 50


def main() -> None:
    if CHECKPOINT_PATH is None:
        raise RuntimeError(
            "train_aif.py: set CHECKPOINT_PATH at the top of this file to a "
            "trained VFEModel checkpoint. The generator wraps the checkpoint by "
            "composition; it does not train the model."
        )

    cfg = AIFConfig(**AIF_CONFIG)

    # Load the trained VFEModel from the checkpoint. The checkpoint format is
    # the one VFETrainer emits: a dict with 'model_state_dict' and 'cfg'.
    checkpoint = torch.load(
        Path(CHECKPOINT_PATH), map_location='cpu', weights_only=False,
    )
    from transformer.vfe.config import VFEConfig
    from transformer.vfe.model import VFEModel

    vfe_cfg_state = checkpoint.get('cfg', None)
    if vfe_cfg_state is None:
        raise RuntimeError(
            f"Checkpoint at {CHECKPOINT_PATH} has no 'cfg' key. "
            "Expected a dict written by VFETrainer."
        )
    if isinstance(vfe_cfg_state, dict):
        vfe_cfg = VFEConfig(**vfe_cfg_state)
    else:
        vfe_cfg = vfe_cfg_state

    model = VFEModel(vfe_cfg)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # The generator builds its preference internally from cfg.preference_type
    # / cfg.preference_path. For the demo we accept an in-memory preference
    # too — pass it as the third arg of AIFGenerator if you want to override.
    generator = AIFGenerator(model=model, cfg=cfg)

    # Tokenize the prompt with the same tokenizer the model was trained on.
    # The checkpoint should carry a 'tokenizer' field; fall back to the
    # default GPT-2 tokenizer if not present.
    tokenizer = checkpoint.get('tokenizer', None)
    if tokenizer is None:
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    prompt_ids = torch.tensor(
        tokenizer.encode(PROMPT_TEXT), dtype=torch.long, device=device,
    ).unsqueeze(0)

    print(f"Prompt: {PROMPT_TEXT!r}")
    print(f"Generating {MAX_NEW_TOKENS} tokens with horizon_D={cfg.horizon_D}, "
          f"beam_width={cfg.beam_width}, gamma={cfg.gamma}...")

    output_ids = generator.generate(prompt_ids, max_new_tokens=MAX_NEW_TOKENS)
    output_text = tokenizer.decode(output_ids[0].tolist())
    print(f"Output: {output_text!r}")
    print(
        f"Belief cache stats: {generator.cache.hits} hits, "
        f"{generator.cache.misses} misses, {len(generator.cache)} entries."
    )


if __name__ == '__main__':
    main()
