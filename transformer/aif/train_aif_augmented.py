"""
Click-to-run EFE-augmented training experiment.

Edit the ``vfe_config`` and ``aif_config`` dicts below, then press Run.
No CLI arguments (per CLAUDE.md).

Trains a ``VFEModel`` end-to-end with the Phase-4 trajectory-as-policy
augmentation :math:`L_{\\text{total}} = L_{\\text{CE}} +
\\lambda_{\\text{AIF}}\\, L_{\\text{AIF}}` per
``docs/plans/2026-05-19-aif-transformer-buildout/06_plan.md`` §6 Phase 4.
The preference distribution is built from the training-corpus token
frequencies on first run and cached to disk for subsequent runs (see
``_build_or_load_empirical_marginal`` below).

Drop-in replacement for ``train_vfe.py`` — the ``vfe_config`` keys
match ``VFEConfig``'s fields exactly. Setting
``aif_config['aif_loss_weight'] = 0.0`` recovers standard VFE training
(modulo a small per-step BALD MC overhead from the still-evaluated
ambiguity/epistemic terms; set
``train_include_ambiguity=False, train_include_epistemic=False`` to
disable that and recover bitwise standard VFE).
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from transformer.aif.config import AIFConfig
from transformer.aif.preferences import EmpiricalMarginalPreference
from transformer.aif.trainer import AIFAugmentedTrainer
from transformer.data.datasets import create_dataloaders
from transformer.vfe.config import VFEConfig
from transformer.vfe.model import VFEModel

logging.basicConfig(level=logging.INFO, format='%(message)s')


# ============================================================================
# VFE CONFIGURATION — edit this dict, then press Run
# ============================================================================
# Mirrors transformer/vfe/train_vfe.py defaults so the AIF augmentation runs
# against the same model spec. Adjust freely.

vfe_config = {
    # === Structure ===
    'vocab_size':               None,      # populated after dataloader build
    'embed_dim':                20,
    'irrep_spec':               [('fund', 2, 10)],

    'batch_size':               64,

    'max_seq_len':              128,
    'max_steps':                15000,

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

    # === Cross-layer prior handoff ===
    'prior_handoff_rho':        1.0,
    'prior_handoff_sigma':      0.0,

    # === Covariance ===
    'diagonal_covariance':      True,
    'isotropic_covariance':     False,
    'exact_diagonal_transport': False,
    'enforce_orthogonal':       False,

    # === Gauge geometry ===
    'gauge_group':              'GLK',

    'phi_project_slk':          False,
    'phi_trace_clamp':          0.75,

    'phi_preconditioner':       'killing',

    # === Positional encoding ===
    'use_rope':                 True,
    'rope_full_gauge':          'off',
    'rope_base':                150,

    # === Embedding init ===
    'mu_init_std':              0.001,
    'phi_scale':                0.001,
    'sigma_init':               0.4,

    # === Decode and generation-time EFE ===
    'epistemic_weight':         0.5,
    'epistemic_samples':        4,
    'decode_tau':               1.0,

    'use_non_flat_transport':       False,
    'non_flat_max_strength':        1.0,
    'non_flat_per_edge_delta_max':  1.0,

    'cross_couplings':                  [],
    'auto_close_cross_head_basis':      False,
    'validate_cross_head_closure':      True,

    # === Normalization ===
    'norm_type':                'layernorm',
    'normalize_ce_by_dim':      True,

    # === Training ===
    'm_mu_lr':                  0.015,
    'm_sigma_lr':               0.004,
    'm_phi_lr':                 0.015,
    'm_hyper_lr':               0.001,
    'm_other_lr':               0.035,
    'weight_decay':             0.075,

    'warmup_steps':             100,
    'grad_clip':                50.0,
    'sigma_max':                12.0,

    'bch_order':                4,

    # === Logging / evaluation ===
    'log_interval':             200,
    'eval_interval':            2000,
    'checkpoint_interval':      25000,

    'track_layer_diagnostics':      False,
    'monitor_monotonicity':         False,
}


# ============================================================================
# AIF CONFIGURATION — edit this dict, then press Run
# ============================================================================
# Phase-4 trajectory-as-policy augmentation. The preference distribution is
# built automatically from the training corpus on first run and cached to disk
# (set preference_path = None to auto-build).

aif_config = {
    # === Training-time augmentation strength ===
    'training_objective':        'efe_augmented',   # required for this script
    'aif_loss_weight':           0.1,               # lambda_AIF
    'train_include_pragmatic':   True,
    'train_include_ambiguity':   True,
    'train_include_epistemic':   True,

    # === Preference distribution ===
    'preference_type':           'empirical_marginal',
    'preference_path':           None,              # auto-build + cache on first run
    'low_entropy_beta':          1.0,

    # === EFE sampling (used by the BALD term) ===
    'gamma':                     1.0,
    'decode_tau':                1.0,
    'epistemic_samples':         4,
    'epistemic_weight':          1.0,
    'discount':                  1.0,

    # === Generation-time (unused at training; kept for AIFConfig schema) ===
    'horizon_D':                 1,
    'beam_width':                4,
    'branching_strategy':        'beam',
    'sampling_strategy':         'multinomial',
    'sampling_temperature':      1.0,
    'belief_cache_max_entries':  4096,
    'habit_prior_path':          None,
}


# ============================================================================
# DATASET + driver knobs
# ============================================================================

SEED = 6
DATASET = 'wikitext-103'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def _build_or_load_empirical_marginal(
    train_loader,
    vocab_size: int,
    cache_path: Path,
    add_one: float = 1.0,
) -> EmpiricalMarginalPreference:
    r"""Build a (V,) log-frequency tensor from the training corpus.

    Caches the result to ``cache_path`` so subsequent runs skip the
    counting pass. Uses add-one (Laplace) smoothing controlled by
    ``add_one`` so log-probabilities are finite for tokens that did not
    appear in the count.
    """
    if cache_path.exists():
        logging.info(f"Loading cached empirical marginal from {cache_path}")
        return EmpiricalMarginalPreference.from_path(str(cache_path))

    logging.info(
        f"Building empirical marginal from training corpus "
        f"(vocab_size={vocab_size}, add_one={add_one})..."
    )
    counts = torch.full((vocab_size,), float(add_one), dtype=torch.float64)
    for batch in train_loader:
        if isinstance(batch, (list, tuple)):
            ids = batch[0]
        else:
            ids = batch['input_ids']
        flat = ids.view(-1)
        flat = flat[flat >= 0]  # drop padding (-100, etc.)
        counts.scatter_add_(
            0, flat.long(), torch.ones_like(flat, dtype=torch.float64),
        )

    probs = counts / counts.sum()
    log_pref = probs.log().to(torch.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(log_pref, cache_path)
    logging.info(
        f"Saved log-probability tensor to {cache_path} "
        f"(min={log_pref.min().item():.2f}, max={log_pref.max().item():.2f})"
    )
    return EmpiricalMarginalPreference(log_pref)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    import random
    from datetime import datetime
    import numpy as np

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    logging.info(f"Seed set to {SEED}")

    # Build dataloaders.
    logging.info(f"Loading dataset: {DATASET}")
    train_loader, val_loader, test_loader, vocab_size = create_dataloaders(
        max_seq_len=vfe_config['max_seq_len'],
        batch_size=vfe_config['batch_size'],
        vocab_size=vfe_config.get('vocab_size'),
        dataset=DATASET,
        include_test=True,
        num_workers=2,
    )
    logging.info(
        f"Dataset loaded: vocab_size={vocab_size}, "
        f"train_batches={len(train_loader)}, val_batches={len(val_loader)}, "
        f"test_batches={len(test_loader)}"
    )

    vfe_config['vocab_size'] = vocab_size
    cfg = VFEConfig(**vfe_config)
    model = VFEModel(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    logging.info(f"VFEModel: {n_params:,} parameters, device={DEVICE}")

    # Build preference. Auto-construct from training corpus if not provided.
    if aif_config['preference_path'] is None:
        cache_dir = Path('aif_runs') / 'preferences'
        preference_cache = cache_dir / f"{DATASET}_empirical_marginal.pt"
        preference = _build_or_load_empirical_marginal(
            train_loader=train_loader,
            vocab_size=vocab_size,
            cache_path=preference_cache,
        )
        # Point the AIFConfig at the cached file so the schema-required
        # field is populated for the validator.
        aif_config['preference_path'] = str(preference_cache)
    else:
        preference = None  # AIFAugmentedTrainer will build it from cfg

    aif_cfg = AIFConfig(**aif_config)
    logging.info(
        f"AIFConfig: training_objective={aif_cfg.training_objective}, "
        f"aif_loss_weight={aif_cfg.aif_loss_weight}, "
        f"epistemic_samples={aif_cfg.epistemic_samples}, "
        f"include=(prag={aif_cfg.train_include_pragmatic}, "
        f"amb={aif_cfg.train_include_ambiguity}, "
        f"epi={aif_cfg.train_include_epistemic})"
    )

    # Output directory.
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = (
        f"aif_runs/{DATASET}_aif{aif_cfg.aif_loss_weight}_{timestamp}"
    )

    trainer = AIFAugmentedTrainer(
        model=model,
        vfe_cfg=cfg,
        aif_cfg=aif_cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=DEVICE,
        output_dir=output_dir,
        preference=preference,
    )
    trainer.train(num_steps=cfg.max_steps)
