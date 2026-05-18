"""
Click-to-run VFE transformer training.

Edit the config dict below, then press Run. No CLI arguments (per CLAUDE.md).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import logging
import torch

from transformer.vfe.config import VFEConfig
from transformer.vfe.model import VFEModel
from transformer.vfe.trainer import VFETrainer
from transformer.data.datasets import create_dataloaders

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# ============================================================================
# CONFIGURATION — edit this dict, then press Run
# ============================================================================

config = {
    # === Structure ===
    'vocab_size':               None,      # populated after dataloader build
    'embed_dim':                20,
    'irrep_spec':               [('fund', 2, 10)],
    
    'batch_size':               64,
    
    'max_seq_len':              64,
    'max_steps':                15000,

    'use_prior_bank':           False,
    'mask_self_attention':      False,
    
    'E_learnable_alpha':        True,
    'learnable_kappa':          False,

    'use_autograd_mu_sigma':       False,
    'use_equivariant_head_mixer':  True,
    'gauge_covariant_ridge':       False,

    # === E-step dynamics ===
    'n_e_steps':                1,
    'n_layers':                 1,

    'alpha_divergence':         1.0,
    'include_attention_entropy': True,

    'e_mu_lr':                  0.4,
    'e_sigma_lr':               0.015,
    'e_phi_lr':                 0.05,
   
    'alpha':                    1.0,
    'lambda_align':             4,
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
    
    'phi_preconditioner':       'killing',  # 'clip', 'cartan', 'killing', 'pullback'

    # === Positional encoding ===
    'use_rope':                 True,
    'rope_full_gauge':          'off',
    'rope_base':                150,

    # === Embedding init ===
    'mu_init_std':              0.001,
    'phi_scale':                0.001,
    'sigma_init':               0.4,

    # === Active inference ===
    'active_inference':         False,
    'pragmatic_weight':         1.0,
    'epistemic_weight':         0.5,
    'epistemic_samples':        4,
    'decode_tau':               1.0,

    'use_non_flat_transport':       False,
    'non_flat_max_strength':        1.0,  # s_max in s = s_max·tanh(ρ)
    'non_flat_per_edge_delta_max':  1.0,  # δ_max bound on ‖δ_ij·G‖_F

    # === Normalization ===
    'norm_type':                'layernorm',
    'normalize_ce_by_dim':      True,

    # === Training ===
    # Per-group M-step LRs (see vfe/config.py for what each touches).
    'm_mu_lr':                  0.2,
    'm_sigma_lr':               0.015,
    'm_phi_lr':                 0.025,
    'm_hyper_lr':               0.001,
    'm_other_lr':               0.05,
    'weight_decay':             0.01,
    
    'warmup_steps':             100,
    'grad_clip':                50.0,
    'sigma_max':                12.0,
    
    'bch_order':                3,

    # === Logging / evaluation ===
    'log_interval':             200,
    'eval_interval':            2000,
    'checkpoint_interval':      25000,

    'track_layer_diagnostics':      True,
    'monitor_monotonicity':         False,

}

# ============================================================================
# DATASET — select one: 'wikitext-2', 'wikitext-103', 'wiki-ja', 'wiki-en'
# ============================================================================

SEED = 6                   # Reproducibility seed for torch / numpy / dataloader workers

DATASET = 'wikitext-103'      # ~103M tokens, full-scale training
# DATASET = 'wikitext-2'      # ~2M tokens, fast iteration
# DATASET = 'wiki-ja'         # Japanese Wikipedia, ~190M tokens at default 100K-article cap
# DATASET = 'wiki-en'         # English Wikipedia, ~5B cl100k tokens at full dump (no cap)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    import random
    import numpy as np
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    logging.info(f"Seed set to {SEED}")

    # Build dataloaders
    logging.info(f"Loading dataset: {DATASET}")
    train_loader, val_loader, vocab_size = create_dataloaders(
        max_seq_len=config['max_seq_len'],
        batch_size=config['batch_size'],
        vocab_size=config.get('vocab_size'),
        dataset=DATASET,
    )
    logging.info(f"Dataset loaded: vocab_size={vocab_size}, "
                 f"train_batches={len(train_loader)}, val_batches={len(val_loader)}")

    # Override vocab_size from tokenizer
    config['vocab_size'] = vocab_size

    cfg = VFEConfig(**config)
    model = VFEModel(cfg)

    n_params = sum(p.numel() for p in model.parameters())
    logging.info(f"VFEModel: {n_params:,} parameters, device={DEVICE}")

    # Output directory for metrics, checkpoints, figures
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'vfe_runs/{DATASET}_{timestamp}'

    # Train
    trainer = VFETrainer(
        model, cfg, train_loader,
        val_loader=val_loader, device=DEVICE,
        output_dir=output_dir,
    )
    trainer.train(num_steps=cfg.max_steps)
