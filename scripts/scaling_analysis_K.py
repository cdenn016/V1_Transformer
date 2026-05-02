r"""
Scaling Analysis Entry Point: Embedding-Dimension (K) Sweep on WikiText-103
===========================================================================

Click-to-run. Edit the CONFIG dict below and press Run.

Walks ``CONFIG['sweep_root']``, computes per-axis aggregates with
multi-seed bootstrap-CI power-law fits, renders four publication-quality
figures, and writes a methods paragraph plus an aggregated CSV.

For the upcoming K=48 / variable-irrep-dim sweep, copy this file to
``scaling_analysis_irrep.py`` and edit the dict only:

    'sweep_root': Path(r'.../scaling_data_irrep'),
    'sweep_axis_name': 'irrep_dim',
    'sweep_axis_regex': r'(?:^|_)irrep=(\d+)(?:$|_)',  # if in dir name
    'sweep_axis_from_config': lambda cfg: _extract_irrep_dim(cfg['irrep_spec']),
    'axis_label': r'$d_\mathrm{irrep}$',

The four analysis modules
(``scaling_ingest``, ``scaling_stats``, ``scaling_plots``, ``scaling_report``)
do not change.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from transformer.analysis.scaling_ingest import aggregate_sweep
from transformer.analysis.scaling_plots import render_all
from transformer.analysis.scaling_report import (
    write_aggregated_csv,
    write_methods_paragraph,
)
from transformer.analysis.scaling_stats import fit_power_law

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIG — edit and press Run
# ============================================================================

CONFIG = {
    'sweep_root': Path(
        r'C:\Users\chris and christine\Desktop\V13_Gauge_Transformer'
        r'\transformer\checkpoints_publication\0_scaling_data'
    ),
    'sweep_axis_name': 'K',
    'sweep_axis_regex': r'(?:^|_)K=(\d+)(?:$|_)',
    'sweep_axis_from_config': None,  # use dir-name regex only
    'output_dir': Path(
        r'C:\Users\chris and christine\Desktop\V13_Gauge_Transformer'
        r'\publication_outputs\scaling_analysis'
    ),
    'axis_label': r'$K$',
    'axis_symbol': 'K',
    'dataset_name': 'WikiText-103',
    'n_bootstrap': 2000,
    'fit_method': 'nonlinear',  # 'nonlinear' or 'loglog'
    'smoothing_window': 50,
    'rng_seed': 0,
    # Optional second paragraph appended to methods.md as 'discussion'. Edit
    # this string in place; lint applies to the combined methods + discussion.
    'discussion': (
        "The fitted floor c is not an architectural ceiling for this model "
        "class. The same single-layer gauge-theoretic transformer trained on "
        "the larger Japanese Wikipedia corpus (about one billion tokens, "
        "roughly an order of magnitude more data than the iso-token budget "
        "used here) reaches test perplexities in the 15 to 30 range across "
        "comparable embedding dimensions, indicating that the WikiText-103 "
        "floor reflects an undertrained-convergence regime at 1.2 epochs "
        "rather than a structural limit of the architecture. Direct cross-"
        "dataset comparison is qualified: WikiText-103 and Japanese "
        "Wikipedia differ in tokenizer (gpt2 BPE 50,257 vocabulary versus "
        "cl100k_base 100,277 vocabulary) and in language, both of which "
        "shift the absolute perplexity even before model-quality differences "
        "enter. With those caveats in place, the order-of-magnitude gap "
        "between the WikiText-103 floor and the Japanese Wikipedia results "
        "is large enough that data quantity dominates the difference, "
        "leaving the b approximately -1 exponent as the primary "
        "architectural fingerprint that survives across both regimes. The b "
        "exponent contrasts with the Chinchilla-style cross-entropy "
        "exponent of approximately 0.34 against total parameter count "
        "(Hoffmann et al., 2022), which corresponds to roughly 0.7 nats per "
        "decade of N at typical excess-loss values; the gauge-theoretic "
        "model loses about 2.4 nats per decade of K. Two structural facts "
        "make the comparison non-head-to-head: K is embedding dimension "
        "rather than total parameter count, and the gauge model has no "
        "learned attention projections (W_Q, W_K, W_V are absent), so the "
        "K-to-N map is approximately linear here while in a standard "
        "transformer it is approximately quadratic."
    ),
}


def main(cfg: dict) -> None:
    output_dir = Path(cfg['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info('Aggregating sweep at: %s', cfg['sweep_root'])
    terminal_df, trajectory_df = aggregate_sweep(
        sweep_root=cfg['sweep_root'],
        sweep_axis_name=cfg['sweep_axis_name'],
        sweep_axis_regex=cfg['sweep_axis_regex'],
        sweep_axis_from_config=cfg.get('sweep_axis_from_config'),
    )
    logger.info(
        'Terminal: %d runs, trajectory: %d rows', len(terminal_df), len(trajectory_df),
    )

    logger.info('Fitting power law (%s, n_bootstrap=%d)',
                cfg['fit_method'], cfg['n_bootstrap'])
    fit = fit_power_law(
        terminal_df,
        axis_col=cfg['sweep_axis_name'],
        y_col='test_ppl',
        n_bootstrap=cfg['n_bootstrap'],
        method=cfg['fit_method'],
        rng_seed=cfg['rng_seed'],
    )
    logger.info(
        'Fit: a=%.3f, b=%.3f, c=%.3f, R^2=%.3f, bootstrap %d/%d succeeded',
        fit.a, fit.b, fit.c, fit.r_squared,
        fit.n_bootstrap_succeeded, fit.n_bootstrap,
    )

    logger.info('Rendering figures to %s', output_dir)
    figure_paths = render_all(
        terminal_df, trajectory_df, fit, output_dir,
        axis_col=cfg['sweep_axis_name'],
        axis_label=cfg['axis_label'],
        smoothing_window=cfg['smoothing_window'],
    )
    for name, path in figure_paths.items():
        logger.info('  %s -> %s', name, path)

    csv_path = output_dir / f'aggregated_{cfg["sweep_axis_name"]}_sweep.csv'
    write_aggregated_csv(terminal_df, csv_path, axis_col=cfg['sweep_axis_name'])

    methods_path = output_dir / 'methods.md'
    write_methods_paragraph(
        terminal_df, fit, methods_path,
        axis_name=cfg['sweep_axis_name'],
        axis_symbol=cfg['axis_symbol'],
        dataset=cfg['dataset_name'],
        discussion=cfg.get('discussion'),
    )

    logger.info('Done.')


if __name__ == '__main__':
    main(CONFIG)
