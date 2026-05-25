"""
Click-to-run AIF generation: wrap a trained VFEModel with canonical EFE.

This script does NOT train. It loads an already-trained ``VFEModel``
checkpoint (from ``train_vfe.py`` or ``train_aif_augmented.py`` — both
formats accepted) and emits tokens using canonical Expected Free Energy
policy selection. The trained model can be either a standard VFE model
or one trained with the EFE-augmented objective.

Edit the three sections below, then press Run. No CLI arguments
(per CLAUDE.md hard constraint).

Default planning preset (``horizon_D=1, beam_width=16``) is the depth-1
EFE anchor. To use Phase-2 beam search, set ``horizon_D=2, beam_width=4``.
For Phase-3 sophisticated inference set ``branching_strategy='sophisticated'``.
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
from transformer.vfe.config import VFEConfig
from transformer.vfe.model import VFEModel

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


# ============================================================================
# 1. VFE MODEL CONFIG — must match the trained checkpoint exactly
# ============================================================================
# Copy this dict from the ``train_vfe.py`` / ``train_aif_augmented.py``
# config that produced your checkpoint. Field-by-field equality is required
# because PyTorch's ``load_state_dict`` checks parameter shapes; a mismatched
# embed_dim, irrep_spec, or norm_type will error at load time.

vfe_config = {
    # === Structure ===
    'vocab_size':               50257,            # set to your trained vocab_size
    'embed_dim':                20,
    'irrep_spec':               [('fund', 2, 10)],

    'max_seq_len':              128,

    'use_prior_bank':           False,
    'mask_self_attention':      False,
    'causal_lower_triangle':    True,

    'gauge_fixed_priors':       False,

    'E_learnable_alpha':        True,
    'learnable_kappa':          False,

    'use_autograd_mu_sigma':       False,
    'use_equivariant_head_mixer':  True,
    'gauge_covariant_ridge':       True,

    # === E-step dynamics ===
    'n_e_steps':                1,
    'n_layers':                 1,

    'alpha_divergence':         1.0,
    'include_attention_entropy': True,

    'e_mu_lr':                  0.5,
    'e_sigma_lr':               0.015,
    'e_phi_lr':                 0.05,

    'alpha':                    1.0,
    'lambda_align':             2.45,
    'lambda_soft':              0.0,
    'mass_phi':                 0.0,

    'kappa':                    1.0,

    'prior_handoff_rho':        1.0,
    'prior_handoff_sigma':      0.0,

    'diagonal_covariance':      True,
    'isotropic_covariance':     False,
    'exact_diagonal_transport': False,
    'enforce_orthogonal':       False,

    'gauge_group':              'GLK',
    'phi_project_slk':          False,
    'phi_trace_clamp':          0.75,
    'phi_preconditioner':       'killing',

    'use_rope':                 True,
    'rope_full_gauge':          'off',
    'rope_base':                150,

    'mu_init_std':              0.001,
    'phi_scale':                0.001,
    'sigma_init':               0.4,

    'epistemic_weight':         0.5,
    'epistemic_samples':        4,
    'decode_tau':               1.0,

    'use_non_flat_transport':       False,
    'cross_couplings':              [],

    'norm_type':                'layernorm',
    'normalize_ce_by_dim':      True,

    'sigma_max':                12.0,
    'bch_order':                4,
}


# ============================================================================
# 2. AIF GENERATION CONFIG — planning + EFE weights + preference
# ============================================================================
# This controls how the generator scores futures. Independent of how the
# model was trained.

AIF_CONFIG = {
    # === Planning ===
    'horizon_D':            1,          # 1 = depth-1 anchor; 2-3 = beam tree
    'beam_width':           16,
    'branching_strategy':   'beam',     # 'beam' | 'top_k' | 'sophisticated'

    # === EFE weights and sampling ===
    'gamma':                1.0,
    'decode_tau':           1.0,
    'epistemic_samples':    4,
    'epistemic_weight':     1.0,
    'discount':             1.0,

    # === Preferences ===
    # 'low_entropy' (default) needs no file — uses p* ∝ exp(-β H[p(o|s)]).
    # 'empirical_marginal' is the canonical default and requires
    #   preference_path → (V,) log-frequency tensor. Running
    #   train_aif_augmented.py once auto-builds + caches this at
    #   aif_runs/preferences/{DATASET}_empirical_marginal.pt — point
    #   preference_path at that file to use the canonical mode.
    # 'task_conditioned' takes any (V,) log-pref tensor (RLHF-style).
    'preference_type':      'low_entropy',
    'preference_path':      None,       # required for empirical_marginal / task_conditioned
    'low_entropy_beta':     1.0,

    'habit_prior_path':     None,       # None = uniform habit

    # === Root-level action sampling ===
    'sampling_strategy':    'multinomial',     # or 'argmin'
    'sampling_temperature': 1.0,

    # === Caching ===
    'belief_cache_max_entries': 4096,

    # === Training-time (irrelevant for this generation script) ===
    'training_objective':   'standard_vfe',
}


# ============================================================================
# 3. CHECKPOINT + PROMPT — click to run
# ============================================================================
# CHECKPOINT_PATH should point to a specific .pt FILE, not a directory.
# Two file types are accepted automatically:
#   - best_model.pt          (bare state_dict, saved at run root by VFETrainer)
#   - checkpoints/step_N.pt  (wrapped dict with 'model_state_dict')
# Use a raw string (r"...") or forward slashes to avoid the
# "unicodeescape" error from backslashes in Windows paths.

CHECKPOINT_PATH: Optional[str] = r"C:\Users\chris and christine\Desktop\V13_Gauge_Transformer\transformer\vfe\vfe_runs\136.25=test-PPL_K=20_GL(10)_baseline-best\checkpoints\step_15000.pt"      # e.g. r"C:\...\best_model.pt"
PROMPT_TEXT: str = "The quick brown fox"
MAX_NEW_TOKENS: int = 50


def _load_state_dict(checkpoint_path: Path) -> dict:
    """Load either format the trainer emits:

    - Bare state_dict (``best_model.pt`` — produced by VFETrainer at the
      best-validation step).
    - Wrapped dict with ``'model_state_dict'`` (``checkpoints/step_N.pt``
      — produced at every ``checkpoint_interval``).

    Returns the bare state_dict in both cases.
    """
    blob = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if isinstance(blob, dict) and 'model_state_dict' in blob:
        return blob['model_state_dict']
    return blob


def main() -> None:
    if CHECKPOINT_PATH is None:
        raise RuntimeError(
            "train_aif.py: set CHECKPOINT_PATH at the top of this file to a "
            "specific .pt file (NOT a directory). For paths with backslashes "
            "on Windows, use a raw string r\"...\" or forward slashes."
        )
    ckpt_path = Path(CHECKPOINT_PATH)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if ckpt_path.is_dir():
        raise IsADirectoryError(
            f"CHECKPOINT_PATH points to a directory: {ckpt_path}\n"
            f"Pass a specific .pt file inside it, e.g. "
            f"{ckpt_path / 'step_15000.pt'} or the sibling best_model.pt."
        )

    cfg = AIFConfig(**AIF_CONFIG)
    vfe_cfg = VFEConfig(**vfe_config)
    model = VFEModel(vfe_cfg)
    state_dict = _load_state_dict(ckpt_path)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logging.warning(
            f"Missing keys in checkpoint ({len(missing)}): "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
        )
    if unexpected:
        logging.warning(
            f"Unexpected keys in checkpoint ({len(unexpected)}): "
            f"{unexpected[:5]}{'...' if len(unexpected) > 5 else ''}"
        )
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    logging.info(
        f"Loaded VFEModel from {ckpt_path.name} "
        f"({sum(p.numel() for p in model.parameters()):,} params, device={device})"
    )

    generator = AIFGenerator(model=model, cfg=cfg)

    # Use GPT-2 tokenizer (matches the default in transformer/data/datasets.py).
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    prompt_ids = torch.tensor(
        tokenizer.encode(PROMPT_TEXT), dtype=torch.long, device=device,
    ).unsqueeze(0)

    print(f"\nPrompt: {PROMPT_TEXT!r}")
    print(
        f"Generating {MAX_NEW_TOKENS} tokens with "
        f"horizon_D={cfg.horizon_D}, beam_width={cfg.beam_width}, "
        f"branching={cfg.branching_strategy}, gamma={cfg.gamma}..."
    )

    output_ids = generator.generate(prompt_ids, max_new_tokens=MAX_NEW_TOKENS)
    output_text = tokenizer.decode(output_ids[0].tolist())
    print(f"\nOutput: {output_text!r}")
    print(
        f"\nBelief cache: {generator.cache.hits} hits, "
        f"{generator.cache.misses} misses, {len(generator.cache)} entries."
    )


if __name__ == '__main__':
    main()
