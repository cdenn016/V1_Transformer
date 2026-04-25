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
    'vocab_size':               None,      # None = full GPT-2 vocabulary (50257)
    'embed_dim':                20,
    'irrep_spec':     [('fund', 2, 10)],   # 2 heads x dim 10 = K=20
    
    
    'batch_size':               16,    
    'max_seq_len':              32,
    'max_steps':                20000,
 
    
    # === E-step dynamics ===
    'n_e_steps':                2,         # Inner-loop iterations per layer
    'n_layers':                 2,
    
    'e_mu_lr':                  0.1,       # Mean natural gradient step size
    'e_sigma_lr':               0.001,     # Covariance retraction step size
    'e_phi_lr':                 0.05,      # Gauge frame step size
    
    'alpha':                    1.0,       # KL(q||p) prior self-coupling weight
    'alpha_divergence':         1.0,       # Rényi α (1.0=KL, 0.5=Bhattacharyya)
    
    'E_learnable_alpha':        True,     # Bayesian adaptive α_i = c0/(b0+KL)
    'lambda_align':             1.0,       # Direct attention coupling (β · ∂KL/∂θ)
    'lambda_soft':              1.0,       # Softmax coupling (∂β/∂θ · KL) — matches VFEConfig default
    'kappa':                    1.0,       # Attention temperature
    'learnable_kappa':          False,     # Learn per-layer kappa


    # === Cross-layer prior handoff ===
    'prior_handoff_rho':        1.0,       # μ damping (1.0 = full flow, <1 = smoother)
    'prior_handoff_sigma':      0.0,       # Σ handoff (0.0 = frozen, >0 = blends posterior)
    'prior_handoff_phi':        False,     # Deprecated/no-op — phi flows via beliefs, not priors


    # === Covariance ===
    'diagonal_covariance':      True,      # True = (B,N,K), False = (B,N,K,K) full
    'isotropic_covariance':     False,     # Force Σ = σ²I
    'exact_diagonal_transport': True,      # Lift diagonal for exact Ω@Σ@Ω^T
    
    
    

    # === Gauge geometry ===
    'gauge_group':              'GLK',      # 'SO3', 'SON', 'GLK'
    'phi_project_slk':          False,      # GL(K) only: hard project φ → sl(K) ⇒ det(Ω) ≡ 1
    'phi_trace_clamp':          None,       # GL(K) only: soft cap |tr(φ·G)| ≤ T (e.g., 0.35)
    'phi_preconditioner':       'killing',  # 'clip', 'cartan', 'killing', 'pullback'
    'enforce_orthogonal':       False,      # Project Ω to SO(K)
    'mask_self_attention':      False,      # Mask diagonal (prevents KL=0 self-attention)
    'mass_phi':                 0.0,        # Gauge prior: (mass_φ/2) mean(||φ||²)


    # === Positional encoding ===
    'use_rope':                 True,
    'rope_full_gauge':          'off',      # 'off' | 'vfe_only' | 'both'. Requires diagonal_covariance=False when != 'off'.
    'rope_base':                100.0,
    
    
    # === Embedding init ===
    'mu_init_std':              0.5,        # Std for base prior mean
    'phi_scale':                0.1,        # Scale for per-token gauge frames
    'sigma_init':               0.5,
    
    
    
    # === Active inference ===
    'active_inference':         False,      # Target-free E-step shaping
    'pragmatic_weight':         1.0,        # Entropy minimization weight
    'epistemic_weight':         0.5,        # BALD MI weight (counterpressure)
    'epistemic_samples':        4,          # MC samples for MI estimate
    'decode_tau':               1.0,        # Temperature for AI readout


    # === Normalization ===
    'norm_type':                'rmsnorm',  # 'mahalnorm', 'rmsnorm', 'none'
    'normalize_ce_by_dim':      False,      # Divide CE by sqrt(K)


    # === Training ===
    'learning_rate':            0.02,
    'weight_decay':             0.001,
    'warmup_steps':             100,
    
    'grad_clip':                15,
    'sigma_max':                12.0,
    'bch_order':                3,          # BCH truncation (1=additive, ≥2=commutator terms)
    
    
    # === Logging / evaluation ===
    'log_interval':             100,
    'eval_interval':            1000,
    'checkpoint_interval':      25000,
}

# ============================================================================
# DATASET — select one: 'wikitext-2', 'wikitext-103', 'wiki-ja'
# ============================================================================

SEED = 1234                   # Reproducibility seed for torch / numpy / dataloader workers

DATASET = 'wikitext-103'      # ~103M tokens, full-scale training
# DATASET = 'wikitext-2'      # ~2M tokens, fast iteration
# DATASET = 'wiki-ja'         # Japanese Wikipedia, ~1B chars

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
