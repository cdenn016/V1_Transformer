r"""
Gauge Geometry Analysis — Standalone Script
=============================================

Analyzes the gauge field geometry of a trained (or freshly initialized) model:
Dirichlet energy of the $\phi$ field on the attention-weighted token graph,
gauge-invariant quantities ($\det\Omega$, spectral structure), gauge orbit
sampling, and effective orbit dimension.

Click-to-run: edit CONFIG below, then press Run.
No CLI arguments (per CLAUDE.md).

Outputs:
    - Gauge geometry metrics per batch element (CSV)
    - Publication-quality figures (PNG + PDF)
    - Gauge field energy map, invariant scatter, orbit PCA

Author: Chris
Date: April 2026
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from pathlib import Path
import csv
import json

# ═══════════════════════════════════════════════════════════════════════
# CONFIG — edit here, then press Run
# ═══════════════════════════════════════════════════════════════════════

CONFIG = {
    # Model source: path to checkpoint directory, or None for fresh init
    'checkpoint_dir':       None,  # e.g. 'checkpoints/run_42'

    # Dataset
    'dataset':              'wikitext2',
    'seq_len':              64,
    'batch_size':           1,      # single batch element for analysis

    # Gauge orbit sampling
    'orbit_n_samples':      50,
    'orbit_perturbation_scale': 0.5,

    # Output
    'output_dir':           'analysis/gauge_geometry_output',
    'device':               'cuda' if torch.cuda.is_available() else 'cpu',
    'seed':                 42,
}

# If no checkpoint, use a matching EM_CONFIG for fresh init
FRESH_CONFIG = {
    'n_layers':                   1,
    'embed_dim':                  20,
    'n_heads':                    2,
    'vocab_size':                 50257,
    'max_seq_len':                64,
    'batch_size':                 4,
    'gauge_group':                'GLK',
    'gauge_mode':                 'learned',
    'gauge_param':                'phi',
    'diagonal_covariance':        True,
    'ffn_n_iterations':           1,
    'E_alpha':                    1.0,
    'E_lambda_belief':            1.0,
    'E_lambda_softmax':           1.0,
    'kappa_beta':                 1.0,
    'use_rope':                   True,
    'rope_base':                  10,
    'phi_natural_gradient':       'killing',
    'E_mu_q_lr':                  0.3,
    'E_sigma_q_lr':               0.05,
    'alpha_divergence':           0.25,
    'irrep_spec':                 [('fund', 2, 10)],
}


def main() -> None:
    """Run gauge geometry analysis."""
    import warnings
    warnings.filterwarnings("ignore", message="CUDA path could not be detected")

    device = torch.device(CONFIG['device'])
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])

    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # ═══════════════════════════════════════════════════════════════════
    # 1. Load or build model
    # ═══════════════════════════════════════════════════════════════════
    from transformer.core.model import GaugeTransformerLM

    if CONFIG['checkpoint_dir'] is not None:
        print(f"Loading model from {CONFIG['checkpoint_dir']}...")
        ckpt_dir = Path(CONFIG['checkpoint_dir'])
        config_path = ckpt_dir / 'config.json'
        if config_path.exists():
            with open(config_path) as f:
                model_config = json.load(f)
        else:
            raise FileNotFoundError(f"No config.json in {ckpt_dir}")

        model = GaugeTransformerLM(model_config).to(device)

        ckpt_files = sorted(ckpt_dir.glob('*.pt'))
        if ckpt_files:
            ckpt = torch.load(ckpt_files[-1], map_location=device, weights_only=False)
            state = ckpt.get('model_state_dict', ckpt)
            model.load_state_dict(state, strict=False)
            print(f"Loaded checkpoint: {ckpt_files[-1].name}")
    else:
        print("Building fresh model from FRESH_CONFIG...")
        model = GaugeTransformerLM(FRESH_CONFIG).to(device)

    model.eval()
    K = FRESH_CONFIG['embed_dim'] if CONFIG['checkpoint_dir'] is None else model_config.get('embed_dim', 20)
    print(f"Model: K={K}")

    # ═══════════════════════════════════════════════════════════════════
    # 2. Get data batch
    # ═══════════════════════════════════════════════════════════════════
    seq_len = CONFIG['seq_len']
    try:
        from transformer.data.datasets import get_dataloader
        train_loader = get_dataloader(
            CONFIG['dataset'], batch_size=CONFIG['batch_size'],
            seq_len=seq_len, split='train',
        )
        batch = next(iter(train_loader))
        if isinstance(batch, (list, tuple)):
            input_ids = batch[0].to(device)
        else:
            input_ids = batch.to(device)
    except Exception as e:
        print(f"Dataset loading failed ({e}), using random tokens")
        vocab_size = 50257
        input_ids = torch.randint(0, vocab_size, (CONFIG['batch_size'], seq_len), device=device)

    print(f"Input shape: {input_ids.shape}")

    # ═══════════════════════════════════════════════════════════════════
    # 3. Forward pass with attention info
    # ═══════════════════════════════════════════════════════════════════
    print("Running forward pass...")
    with torch.no_grad():
        targets = input_ids[:, 1:].contiguous()
        input_trimmed = input_ids[:, :-1].contiguous()
        logits, attn_info = model.forward_with_attention(input_trimmed, targets=targets)
    print("Forward pass complete.")

    # ═══════════════════════════════════════════════════════════════════
    # 4. Extract geometry data
    # ═══════════════════════════════════════════════════════════════════
    phi = attn_info['phi']              # (B, N, gauge_dim)
    mu = attn_info['mu']                # (B, N, K)
    sigma = attn_info['sigma']          # (B, N, K)
    beta_all = attn_info['beta']        # (n_layers, B, n_heads, N, N)
    generators = model.generators       # (n_gen, K, K)

    # Average beta across heads, take final layer
    beta = beta_all[-1].mean(dim=1)     # (B, N, N)

    B, N = mu.shape[:2]
    n_gen = phi.shape[-1]
    print(f"Extracted: B={B}, N={N}, K={mu.shape[-1]}, n_gen={n_gen}")

    # ═══════════════════════════════════════════════════════════════════
    # 5. Compute gauge geometry metrics
    # ═══════════════════════════════════════════════════════════════════
    from transformer.analysis.gauge_geometry import (
        compute_gauge_field_energy,
        compute_gauge_invariants,
        gauge_orbit_sample,
        compute_gauge_orbit_dimension,
    )

    print("\nComputing gauge field Dirichlet energy...")
    field_energy = compute_gauge_field_energy(phi, beta, generators)
    for b in range(B):
        print(f"  Batch {b}: E_Dirichlet = {field_energy[b].item():.6f}")

    print("\nComputing gauge invariants...")
    invariants = compute_gauge_invariants(mu, sigma, phi, generators, beta)
    det_omega = invariants['det_omega']     # (B, N, N)
    spectrum = invariants['gauge_frame_spectrum']  # (B, N, K)
    print(f"  det(Omega) mean={det_omega.mean().item():.4f}, "
          f"std={det_omega.std().item():.4f}")
    print(f"  Spectrum mean={spectrum.mean().item():.4f}, "
          f"std={spectrum.std().item():.4f}")

    print("\nComputing gauge orbit dimension...")
    orbit_dim = compute_gauge_orbit_dimension(phi, generators)
    print(f"  Effective orbit dimension: {orbit_dim} / {n_gen}")

    print(f"\nSampling {CONFIG['orbit_n_samples']} gauge orbit points...")
    orbit_samples = gauge_orbit_sample(
        mu, sigma, phi, generators,
        n_samples=CONFIG['orbit_n_samples'],
        perturbation_scale=CONFIG['orbit_perturbation_scale'],
    )
    print(f"  Sampled {len(orbit_samples)} orbit points")

    # ═══════════════════════════════════════════════════════════════════
    # 6. Save metrics CSV
    # ═══════════════════════════════════════════════════════════════════
    csv_path = output_dir / 'gauge_geometry_stats.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'batch_idx', 'field_energy', 'det_omega_mean', 'det_omega_std',
            'spectrum_mean', 'spectrum_std', 'orbit_dim',
        ])
        for b in range(B):
            writer.writerow([
                b,
                field_energy[b].item(),
                det_omega[b].mean().item(),
                det_omega[b].std().item(),
                spectrum[b].mean().item(),
                spectrum[b].std().item(),
                orbit_dim,
            ])
    print(f"\nMetrics saved to {csv_path}")

    # ═══════════════════════════════════════════════════════════════════
    # 7. Generate figures
    # ═══════════════════════════════════════════════════════════════════
    from transformer.visualization.gauge_geometry_plots import (
        generate_all_gauge_geometry_figures,
    )

    # Build phi_diff_sq matrix for energy map: ||phi_i - phi_j||^2
    phi_0 = phi[0].cpu().numpy()  # (N, n_gen)
    phi_diff = phi_0[:, None, :] - phi_0[None, :, :]  # (N, N, n_gen)
    phi_diff_sq = (phi_diff ** 2).sum(axis=-1)  # (N, N)

    # Convert orbit samples to numpy (batch element 0)
    orbit_samples_np = []
    for sample in orbit_samples:
        orbit_samples_np.append({
            'phi': sample['phi'][0].cpu().numpy(),   # (N, n_gen)
            'mu': sample['mu'][0].cpu().numpy(),     # (N, K)
        })
    # Add original point
    orbit_samples_np.insert(0, {
        'phi': phi[0].cpu().numpy(),
        'mu': mu[0].cpu().numpy(),
        'is_original': True,
    })

    data = {
        'ym_steps': [0],
        'ym_energies': [0.0],  # flat transport => YM=0
        'ym_dirichlet_energies': [field_energy[0].item()],
        'phi_diff_sq': phi_diff_sq,
        'beta': beta[0].cpu().numpy(),
        'invariants': {
            'det_omega': det_omega[0].cpu().numpy().flatten(),
            'kl_values': invariants['kl_values'][0].cpu().numpy().flatten() if invariants.get('kl_values') is not None else None,
            'field_energy': field_energy.cpu().numpy(),
        },
        'orbit_samples': orbit_samples_np,
    }

    print(f"\nGenerating figures...")
    saved = generate_all_gauge_geometry_figures(data, output_dir)
    for name, path in saved.items():
        print(f"  {name}: {path}")

    print(f"\nAnalysis complete. All outputs in {output_dir}/")


if __name__ == '__main__':
    main()
