#!/usr/bin/env python3
"""
Click-to-run Pure VFE Transformer.

Usage: Just press Run (F5 / Shift+Enter / python run.py).
Edit PURE_VFE_CONFIG below to change settings -- no CLI needed.

No nn.Module. No autograd. No loss.backward().
Just variational free energy descent on a gauge-covariant prior bank.
"""

import time
import sys

import torch

from transformer.pure_vfe.config import PureVFEConfig
from transformer.pure_vfe.model import PureVFETransformer
from transformer.pure_vfe.gauge import monitor_omega_health
from transformer.data.datasets import create_dataloaders


# ═══════════════════════════════════════════════════════════════════
#  CONFIGURATION -- edit this dict, then hit Run
# ═══════════════════════════════════════════════════════════════════

PURE_VFE_CONFIG = {
    # ── Data ──────────────────────────────────────────────────────
    'dataset': 'wikitext-2',          # 'wikitext-2', 'wikitext-103', or 'wiki-ja'
    'vocab_size': 50257,              # 50257 for English (GPT-2), 100277 for wiki-ja (cl100k)
    'num_workers': 4,                 # DataLoader CPU workers

    # ── Belief geometry ───────────────────────────────────────────
    'belief_dim': 16,                 # K: full belief dimension
    'n_heads': 4,                     # H: number of attention heads
    # head_dim is derived: belief_dim // n_heads

    # ── VFE descent ─────────────────────────────────────────────────
    'n_esteps': 6,                    # VFE descent iterations (depth)
    'tau': None,                      # Attention temperature (None -> √head_dim)

    # ── Per-variable natural gradient learning rates ──────────────
    'mu_q_lr': 0.1,                   # Belief mean step size
    'sigma_q_lr': 0.005,              # Belief covariance step size
    'phi_lr': 0.1,                    # Gauge connection step size
    'mu_p_lr': 0.001,                 # Prior mean step size
    'sigma_p_lr': 0.0002,             # Prior covariance step size

    # ── Prior precision ───────────────────────────────────────────
    'alpha_b0': 1.0,                  # Denominator offset
    'alpha_c0': 1.0,                  # Numerator scale

    # ── Hyper-prior ───────────────────────────────────────────────
    'hyper_var': 1.0,                 # Variance of hyper-prior on means

    # ── Sequence / batching ───────────────────────────────────────
    'max_seq_len': 32,                # N: maximum sequence length
    'batch_size': 4,                  # Batch size

    # ── Initialization ────────────────────────────────────────────
    'sigma_init': 1.0,                # Initial covariance scale
    'omega_init_scale': 0.01,         # GL(K) frame perturbation from I

    # ── Trust regions ─────────────────────────────────────────────
    'trust_region_mu': 1.0,
    'trust_region_sigma': 0.3,
    'trust_region_omega': 0.3,

    # ── SPD retraction safeguards ─────────────────────────────────
    'spd_eps_min': 1e-4,              # Spectral floor
    'spd_kappa_max': 1e4,             # Condition number cap
    'spd_exp_clip': 50.0,             # Eigenvalue exponent clip

    # ── Causal masking ────────────────────────────────────────────
    'causal': True,

    # ── Device ────────────────────────────────────────────────────
    'device': 'cpu',                  # 'cpu' or 'cuda'
    'use_cuda_kernels': False,        # Custom CUDA kernels (needs ninja + CUDA)

    # ── Training ──────────────────────────────────────────────────
    'epochs': 3,                      # Number of training epochs
    'log_interval': 5,                # Print every N steps
    'save_path': None,                # Path to save best checkpoint (None = don't save)
}


# ═══════════════════════════════════════════════════════════════════
#  BUILD CONFIG FROM DICT
# ═══════════════════════════════════════════════════════════════════

def build_config(cfg):
    """Convert PURE_VFE_CONFIG dict -> PureVFEConfig dataclass."""
    head_dim = cfg['belief_dim'] // cfg['n_heads']

    config_kwargs = {
        'vocab_size':         cfg['vocab_size'],
        'belief_dim':         cfg['belief_dim'],
        'n_heads':            cfg['n_heads'],
        'head_dim':           head_dim,
        'n_esteps':           cfg['n_esteps'],
        'tau':                cfg['tau'],
        'mu_q_lr':            cfg['mu_q_lr'],
        'sigma_q_lr':         cfg['sigma_q_lr'],
        'phi_lr':             cfg['phi_lr'],
        'mu_p_lr':            cfg['mu_p_lr'],
        'sigma_p_lr':         cfg['sigma_p_lr'],
        'alpha_b0':           cfg['alpha_b0'],
        'alpha_c0':           cfg['alpha_c0'],
        'hyper_var':          cfg['hyper_var'],
        'max_seq_len':        cfg['max_seq_len'],
        'batch_size':         cfg['batch_size'],
        'sigma_init':         cfg['sigma_init'],
        'omega_init_scale':   cfg['omega_init_scale'],
        'trust_region_mu':    cfg['trust_region_mu'],
        'trust_region_sigma': cfg['trust_region_sigma'],
        'trust_region_omega': cfg['trust_region_omega'],
        'spd_eps_min':        cfg['spd_eps_min'],
        'spd_kappa_max':      cfg['spd_kappa_max'],
        'spd_exp_clip':       cfg['spd_exp_clip'],
        'causal':             cfg['causal'],
        'device':             cfg['device'],
        'use_cuda_kernels':   cfg['use_cuda_kernels'],
    }

    return PureVFEConfig(**config_kwargs)


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main(cfg=None):
    """
    Train the Pure VFE Transformer using the provided config dict.
    Falls back to PURE_VFE_CONFIG if none given.
    """
    if cfg is None:
        cfg = PURE_VFE_CONFIG

    # ── Resolve device ──
    device = cfg['device']
    if device == 'cuda' and not torch.cuda.is_available():
        print("[WARN] CUDA not available, falling back to CPU")
        device = 'cpu'
        cfg['device'] = 'cpu'
        cfg['use_cuda_kernels'] = False

    # ── Load data via the shared pipeline ──
    dataset_name = cfg['dataset']

    # Auto-adjust vocab for Japanese
    if dataset_name == 'wiki-ja' and cfg['vocab_size'] == 50257:
        cfg['vocab_size'] = 100277

    print(f"Loading {dataset_name}...")
    train_loader, val_loader, actual_vocab_size = create_dataloaders(
        max_seq_len=cfg['max_seq_len'],
        batch_size=cfg['batch_size'],
        vocab_size=cfg['vocab_size'],
        num_workers=cfg['num_workers'],
        dataset=dataset_name,
    )
    cfg['vocab_size'] = actual_vocab_size
    print(f"  vocab_size={actual_vocab_size}, seq_len={cfg['max_seq_len']}")

    # ── Build config and model ──
    config = build_config(cfg)
    model = PureVFETransformer(config)
    params = model.param_count()

    print()
    print("=" * 60)
    print("  Pure VFE Transformer")
    print("=" * 60)
    print(f"  dataset={dataset_name}")
    print(f"  belief_dim={config.belief_dim}  n_heads={config.n_heads}  head_dim={config.head_dim}")
    print(f"  n_esteps={config.n_esteps}  tau={config.tau if config.tau is None else f'{config.tau:.2f}'}")
    print(f"  mu_q_lr={config.mu_q_lr}  sigma_q_lr={config.sigma_q_lr}  phi_lr={config.phi_lr}")
    print(f"  mu_p_lr={config.mu_p_lr}  sigma_p_lr={config.sigma_p_lr}")
    print(f"  device={device}  params={params['total']:,}")
    for k, v in params.items():
        if k != "total":
            print(f"    {k}: {v:,}")
    print("=" * 60)
    print()
    print(f"{'Epoch':>5} {'Step':>6} {'Loss':>8} {'PPL':>10} "
          f"{'VFE_0':>10} {'VFE_f':>10} {'s/step':>8}")
    print("-" * 60)

    # ── Train ──
    best_loss = float("inf")
    global_step = 0
    log_interval = cfg['log_interval']
    save_path = cfg['save_path']

    for epoch in range(cfg['epochs']):
        epoch_loss = 0.0
        epoch_steps = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            t0 = time.time()
            logits, ce_loss, vfe_history, _diag = model.update(inputs, targets)
            dt = time.time() - t0

            epoch_loss += ce_loss
            epoch_steps += 1
            global_step += 1

            if global_step % log_interval == 0:
                ppl = torch.exp(torch.tensor(ce_loss)).item()
                vfe_0 = vfe_history[0] if vfe_history else 0.0
                vfe_f = vfe_history[-1] if vfe_history else 0.0
                print(
                    f"{epoch:5d} {global_step:6d} {ce_loss:8.3f} {ppl:10.1f} "
                    f"{vfe_0:10.1f} {vfe_f:10.1f} {dt:8.3f}"
                )

                # Gauge health check
                health = monitor_omega_health(model.prior_Omega[:100], "Omega")
                if health["Omega/cond_max"] > 100:
                    print(f"  [WARN] condition number high: {health['Omega/cond_max']:.1f}")

        avg_loss = epoch_loss / max(epoch_steps, 1)
        avg_ppl = torch.exp(torch.tensor(avg_loss)).item()
        print(f"\n  Epoch {epoch} summary: loss={avg_loss:.3f}  ppl={avg_ppl:.1f}\n")

        if save_path and avg_loss < best_loss:
            best_loss = avg_loss
            model.save(save_path)
            print(f"  Saved best model -> {save_path}\n")

    print("Done.")
    return model


if __name__ == "__main__":
    main()
