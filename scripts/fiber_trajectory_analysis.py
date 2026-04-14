r"""
Fiber Trajectory Analysis — Standalone Script
================================================

Analyzes how beliefs $(\mu, \Sigma)$ evolve during VFE E-step iterations
on the Gaussian belief manifold $\mathcal{G}_K$ with Fisher-Rao metric.

Click-to-run: edit CONFIG below, then press Run.
No CLI arguments (per CLAUDE.md).

Outputs:
    - Fisher-Rao trajectory statistics per token per layer (CSV)
    - Publication-quality figures (PNG + PDF)
    - Fiber trajectory dashboard

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
    'dataset':              'wikitext-2',
    'seq_len':              128,
    'batch_size':           1,      # single batch element for analysis

    # Recording
    'n_tokens_to_record':   16,     # how many token positions to track
    'token_selection':      'uniform',  # 'uniform', 'random', 'first'
    'layers_to_analyze':    None,   # None = all layers, or list of ints

    # Output
    'output_dir':           'analysis/fiber_trajectory_output',
    'device':               'cuda' if torch.cuda.is_available() else 'cpu',
    'seed':                 42,
}

# If no checkpoint, use a matching EM_CONFIG for fresh init
FRESH_CONFIG = {
    'n_layers':                   2,
    'embed_dim':                  20,
    'vocab_size':                 50257,
    'max_seq_len':                128,
    'batch_size':                 4,
    'gauge_group':                'GLK',
    'gauge_mode':                 'learned',
    'gauge_param':                'phi',
    'diagonal_covariance':        True,
    'ffn_n_iterations':           8,
    'E_alpha':                    1.0,
    'E_lambda_belief':            1.0,
    'E_lambda_softmax':           1.0,
    'kappa_beta':                 3.16,
    'use_rope':                   True,
    'rope_base':                  10,
    'phi_natural_gradient':       'killing',
    'E_mu_q_lr':                  0.3,
    'E_sigma_q_lr':               0.05,
}


def main() -> None:
    """Run fiber trajectory analysis."""
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
    if CONFIG['checkpoint_dir'] is not None:
        print(f"Loading model from {CONFIG['checkpoint_dir']}...")
        ckpt_dir = Path(CONFIG['checkpoint_dir'])
        # Load config
        config_path = ckpt_dir / 'config.json'
        if config_path.exists():
            with open(config_path) as f:
                model_config = json.load(f)
        else:
            raise FileNotFoundError(f"No config.json in {ckpt_dir}")

        from transformer.core.model import GaugeTransformerLM
        model = GaugeTransformerLM(model_config).to(device)

        # Load weights
        ckpt_files = sorted(ckpt_dir.glob('*.pt'))
        if ckpt_files:
            ckpt = torch.load(ckpt_files[-1], map_location=device, weights_only=False)
            state = ckpt.get('model_state_dict', ckpt)
            model.load_state_dict(state, strict=False)
            print(f"Loaded checkpoint: {ckpt_files[-1].name}")
    else:
        print("Building fresh model from FRESH_CONFIG...")
        from transformer.core.model import GaugeTransformerLM
        model = GaugeTransformerLM(FRESH_CONFIG).to(device)

    model.eval()
    n_layers = len(model.blocks)
    K = model.blocks[0].ffn.embed_dim
    print(f"Model: {n_layers} layers, K={K}")

    # ═══════════════════════════════════════════════════════════════════
    # 2. Get data batch
    # ═══════════════════════════════════════════════════════════════════
    seq_len = CONFIG['seq_len']
    try:
        from transformer.data.datasets import create_dataloaders
        train_loader, _, _ = create_dataloaders(
            dataset_name=CONFIG['dataset'], batch_size=CONFIG['batch_size'],
            seq_len=seq_len,
        )
        batch = next(iter(train_loader))
        if isinstance(batch, (list, tuple)):
            input_ids = batch[0].to(device)
        else:
            input_ids = batch.to(device)
    except Exception as e:
        print(f"Dataset loading failed ({e}), using random tokens")
        vocab_size = getattr(model, 'vocab_size', 50257)
        input_ids = torch.randint(0, vocab_size, (CONFIG['batch_size'], seq_len), device=device)

    print(f"Input shape: {input_ids.shape}")

    # ═══════════════════════════════════════════════════════════════════
    # 3. Select token indices to record
    # ═══════════════════════════════════════════════════════════════════
    n_tok = CONFIG['n_tokens_to_record']
    N = input_ids.shape[1]
    if CONFIG['token_selection'] == 'uniform':
        token_indices = np.linspace(0, N - 1, n_tok, dtype=np.int64)
    elif CONFIG['token_selection'] == 'random':
        token_indices = np.sort(np.random.choice(N, n_tok, replace=False))
    else:  # 'first'
        token_indices = np.arange(min(n_tok, N), dtype=np.int64)

    print(f"Recording {len(token_indices)} tokens: {token_indices.tolist()}")

    # ═══════════════════════════════════════════════════════════════════
    # 4. Enable fiber recording on all layers
    # ═══════════════════════════════════════════════════════════════════
    layers = CONFIG['layers_to_analyze'] or list(range(n_layers))
    for layer_idx in layers:
        ffn = model.blocks[layer_idx].ffn
        ffn.enable_fiber_recording(
            token_indices=token_indices,
            n_tokens=n_tok,
            seq_len=N,
        )
    print(f"Fiber recording enabled on layers {layers}")

    # ═══════════════════════════════════════════════════════════════════
    # 5. Forward pass
    # ═══════════════════════════════════════════════════════════════════
    print("Running forward pass...")
    with torch.no_grad():
        try:
            output = model(input_ids)
        except Exception as e:
            # Some models need targets
            targets = input_ids[:, 1:].contiguous()
            input_trimmed = input_ids[:, :-1].contiguous()
            output = model(input_trimmed, targets=targets)
    print("Forward pass complete.")

    # ═══════════════════════════════════════════════════════════════════
    # 6. Extract snapshots and analyze
    # ═══════════════════════════════════════════════════════════════════
    from transformer.analysis.fiber_trajectory import (
        analyze_all_tokens, FiberTrajectoryStats,
    )

    all_layer_stats: dict = {}
    all_layer_snapshots: dict = {}

    for layer_idx in layers:
        ffn = model.blocks[layer_idx].ffn
        snapshots = ffn.get_fiber_snapshots()
        ffn.disable_fiber_recording()

        if not snapshots:
            print(f"  Layer {layer_idx}: no snapshots (closed-form E-step?)")
            continue

        n_recorded = snapshots[0].mu.shape[0]
        print(f"  Layer {layer_idx}: {len(snapshots)} iterations, "
              f"{n_recorded} tokens, K={snapshots[0].mu.shape[1]}")

        stats = analyze_all_tokens(snapshots, n_tokens=n_recorded)
        all_layer_stats[layer_idx] = stats
        all_layer_snapshots[layer_idx] = snapshots

        # Print summary
        for s in stats:
            print(f"    Token {s.token_idx:>3d}: arc_len={s.arc_length:.4f}, "
                  f"geodesic={s.geodesic_distance:.4f}, "
                  f"deviation={s.deviation_ratio:.3f}, "
                  f"conv_rate={s.convergence_rate:.4f}")

    if not all_layer_stats:
        print("No fiber trajectory data collected. "
              "Ensure the model uses iterative E-step (closed_form_e_step=False).")
        return

    # ═══════════════════════════════════════════════════════════════════
    # 7. Save metrics CSV
    # ═══════════════════════════════════════════════════════════════════
    csv_path = output_dir / 'fiber_trajectory_stats.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'layer', 'token_idx', 'arc_length', 'geodesic_distance',
            'deviation_ratio', 'convergence_rate', 'mean_velocity',
            'final_velocity', 'mu_displacement', 'sigma_ratio',
        ])
        for layer_idx, stats in all_layer_stats.items():
            for s in stats:
                writer.writerow([
                    layer_idx, s.token_idx, s.arc_length, s.geodesic_distance,
                    s.deviation_ratio, s.convergence_rate, s.mean_velocity,
                    s.final_velocity, s.mu_displacement, s.sigma_ratio,
                ])
    print(f"\nMetrics saved to {csv_path}")

    # ═══════════════════════════════════════════════════════════════════
    # 8. Generate figures
    # ═══════════════════════════════════════════════════════════════════
    from transformer.visualization.fiber_plots import generate_all_fiber_figures

    for layer_idx in all_layer_snapshots:
        snapshots = all_layer_snapshots[layer_idx]
        stats = all_layer_stats[layer_idx]
        layer_dir = output_dir / f'layer_{layer_idx}'

        print(f"\nGenerating figures for layer {layer_idx}...")
        saved = generate_all_fiber_figures(
            snapshots=snapshots,
            stats_list=stats,
            token_indices=list(range(len(stats))),
            output_dir=layer_dir,
        )
        for name, path in saved.items():
            print(f"  {name}: {path}")

    # Arc length heatmap across layers (if multiple layers)
    if len(all_layer_stats) > 1:
        try:
            from transformer.visualization.fiber_plots import plot_arc_length_heatmap
            n_layers_analyzed = len(all_layer_stats)
            n_tokens_analyzed = len(next(iter(all_layer_stats.values())))
            arc_map = np.zeros((n_layers_analyzed, n_tokens_analyzed))
            for i, (layer_idx, stats) in enumerate(sorted(all_layer_stats.items())):
                for j, s in enumerate(stats):
                    arc_map[i, j] = s.arc_length
            fig = plot_arc_length_heatmap(
                arc_map,
                token_labels=[f"t{idx}" for idx in token_indices],
                save_path=output_dir / 'arc_length_heatmap.png',
            )
            if fig is not None:
                import matplotlib.pyplot as plt
                plt.close(fig)
                print(f"  arc_length_heatmap: {output_dir / 'arc_length_heatmap.png'}")
        except Exception as e:
            print(f"  arc_length_heatmap failed: {e}")

    print(f"\nAnalysis complete. All outputs in {output_dir}/")


if __name__ == '__main__':
    main()
