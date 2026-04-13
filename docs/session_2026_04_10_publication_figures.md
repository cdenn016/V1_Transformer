# Publication Figure Gap Analysis & Implementation — 2026-04-10

## New File

`transformer/visualization/publication_gap_figures.py` — 10 publication-quality figure functions addressing the 10 gaps identified in the diagnostics audit.

## Functions Implemented

| Gap ID | Function | Input | Output |
|--------|----------|-------|--------|
| C2 | `plot_vfe_decomposition()` | metrics CSV | Stacked area: CE + α·KL + β·align + γ·model, with fractional panel |
| C5 | `plot_equivariance_error()` | model + dataloader | Histogram of gauge equivariance error under random GL(K) transforms |
| C6 | `plot_posterior_collapse()` | metrics CSV | 3-panel: σ_q evolution, prior-belief KL, effective dimensionality |
| H1 | `plot_kl_vs_dotproduct_attention()` | model + input_ids | 3-panel: KL heatmap, dot-product heatmap, difference + Pearson r |
| H4 | `plot_estep_convergence()` | iteration diagnostics CSV | ‖∇μ F‖ and ‖∇σ F‖ vs E-step iteration at multiple training steps |
| H6 | `plot_phi_spectral_evolution()` | metrics CSV | 3-panel: effective rank, variance concentration, spectral gap + ‖φ‖ |
| H7 | `plot_attention_entropy_per_head()` | metrics CSV | Time series of min/mean/max entropy + final summary bar |
| H8 | `evaluate_linear_probe()` | model + dataloader | Dict of top-1, top-5 accuracy (scikit-learn logistic regression) |
| H8 | `plot_linear_probe_results()` | results dict | Bar chart with chance level baseline |
| M6 | `plot_vfe_gradient_components()` | metrics CSV | 2-panel log-scale: μ and σ gradient decomposition (self/direct/softmax/total) |

## Data Sources

Most figures load from the existing 82-column metrics CSV at `checkpoints_publication/ffn_VFE_dynamic/metrics.csv`. Two figures (H4) require `track_iteration_diagnostics=True` to populate the iteration diagnostics CSV. Two figures (C5, H1, H8) require a trained model and dataloader.

## Remaining Gaps Not Implemented Here

| Gap | Reason |
|-----|--------|
| H3 (Natural gradient vs Euclidean) | Requires a separate training run with `optimizer_type='adamw'` — config change, not code |
| C3 (Standard transformer comparison) | Requires training STANDARD_CONFIG — existing ablation_plots.py handles the figure once data exists |
| C4 (Ablation table) | run_ablation_suite.py already generates this — needs specific sweep configs run |

## Usage

```python
from transformer.visualization.publication_gap_figures import *

# From metrics CSV (no model needed):
csv = Path('checkpoints_publication/ffn_VFE_dynamic/metrics.csv')
plot_vfe_decomposition(csv, save_path=Path('figures/vfe_decomposition.png'))
plot_posterior_collapse(csv, save_path=Path('figures/posterior_collapse.png'))
plot_phi_spectral_evolution(csv, save_path=Path('figures/phi_spectral.png'))
plot_attention_entropy_per_head(csv, save_path=Path('figures/attention_entropy.png'))
plot_vfe_gradient_components(csv, save_path=Path('figures/vfe_gradients.png'))

# From iteration diagnostics CSV (requires track_iteration_diagnostics=True):
iter_csv = Path('checkpoints_publication/ffn_VFE_dynamic/iteration_diagnostics.csv')
plot_estep_convergence(iter_csv, save_path=Path('figures/estep_convergence.png'))

# From trained model:
plot_equivariance_error(model, val_loader, save_path=Path('figures/equivariance.png'))
plot_kl_vs_dotproduct_attention(model, sample_input, tokenizer,
                                 save_path=Path('figures/kl_vs_dp.png'))
results = evaluate_linear_probe(model, val_loader)
plot_linear_probe_results(results, save_path=Path('figures/linear_probe.png'))
```
