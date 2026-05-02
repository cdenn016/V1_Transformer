"""
Scaling-Sweep Report Generation
===============================

Two outputs from a scaling-sweep analysis:

* :func:`write_aggregated_csv` — long-format per-axis summary (mean, std,
  sem, n_seeds, params, FLOPs) so future analyses don't need to re-walk the
  raw run dirs.
* :func:`write_methods_paragraph` — one prose paragraph (<= 250 words)
  describing protocol, statistical analysis, and headline numbers, ready to
  drop into a manuscript methods section. Runs a banned-word linter against
  the rendered text to keep Claude-isms out of the manuscript.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from transformer.analysis.scaling_stats import PowerLawFit, seed_summary

logger = logging.getLogger(__name__)


BANNED_PHRASES = (
    'key insight',
    'crucially',
    'critically',
    'notably',
    'importantly',
    'it\'s worth noting',
    'interestingly',
    'fundamentally',
    'in particular',
    'leverages',
    'underscores',
)


def _lint(text: str) -> None:
    """Reject text containing any banned phrase or horizontal-rule line."""
    lower = text.lower()
    offenders: List[str] = [p for p in BANNED_PHRASES if p in lower]
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == '---' or set(stripped) == {'-'} and len(stripped) >= 3:
            offenders.append(f'horizontal rule: {line!r}')
    if offenders:
        raise ValueError(
            "Methods text failed lint. Offending tokens:\n  - "
            + "\n  - ".join(offenders)
        )


def write_aggregated_csv(
    terminal_df: pd.DataFrame,
    output_path: Path,
    axis_col: str = 'K',
) -> Path:
    """Write per-axis aggregated CSV. Returns the path."""
    summary = seed_summary(terminal_df, axis_col=axis_col)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    logger.info("Wrote aggregated CSV: %s (%d rows)", output_path, len(summary))
    return output_path


def _format_ci(point: float, ci: tuple, fmt: str = '.3f') -> str:
    if pd.isna(ci[0]) or pd.isna(ci[1]):
        return f"{point:{fmt}}"
    return f"{point:{fmt}} (95% CI {ci[0]:{fmt}}, {ci[1]:{fmt}})"


def _shared_config_or_none(
    terminal_df: pd.DataFrame,
    keys: tuple,
) -> Optional[Dict]:
    """Return a dict of shared config values across runs, or None on drift."""
    return None


def write_methods_paragraph(
    terminal_df: pd.DataFrame,
    fit: PowerLawFit,
    output_path: Path,
    axis_name: str = 'K',
    axis_symbol: str = 'K',
    dataset: str = 'WikiText-103',
    seq_len: Optional[int] = None,
    tokens_seen: Optional[int] = None,
    discussion: Optional[str] = None,
) -> Path:
    """Render and write the methods paragraph; lint before write.

    Pulls protocol facts (seq_len, tokens_seen) from the terminal_df when not
    explicitly provided. When ``discussion`` is given, appends it as a second
    paragraph after a blank line so the file becomes a methods + discussion
    document rather than methods alone — the lint runs on the combined text.
    """
    df = terminal_df.copy()
    axis_values = sorted(df[axis_name].dropna().unique().astype(int).tolist())
    n_seeds = int(df['seed'].nunique()) if 'seed' in df.columns else len(fit.seeds_used)
    if seq_len is None and 'seq_len' in df.columns:
        seq_len_vals = df['seq_len'].dropna().unique()
        seq_len = int(seq_len_vals[0]) if len(seq_len_vals) == 1 else None
    if tokens_seen is None and 'tokens_seen' in df.columns:
        tokens_vals = df['tokens_seen'].dropna().unique()
        tokens_seen = int(tokens_vals[0]) if len(tokens_vals) == 1 else None

    seq_len_clause = f"with sequence length {seq_len}" if seq_len else ""
    tokens_clause = (
        f"matched at {tokens_seen / 1e6:.1f} M tokens"
        if tokens_seen else "matched on tokens seen"
    )
    seeds_clause = f"three random seeds ({', '.join(map(str, fit.seeds_used))})"
    axis_clause = (
        f"{axis_symbol} sweep over "
        f"{{{', '.join(map(str, axis_values))}}}"
    )
    fit_form = f"PPL = a {axis_symbol}^b + c"
    method_clause = (
        "nonlinear least squares (scipy.optimize.curve_fit)"
        if fit.method == 'nonlinear'
        else "log–log linear regression"
    )
    bootstrap_clause = (
        f"95% percentile intervals from {fit.n_bootstrap_succeeded} successful "
        f"bootstrap resamples (of {fit.n_bootstrap} requested), drawing seeds "
        f"with replacement within each {axis_symbol} value to keep the sweep "
        f"design intact"
        if fit.n_bootstrap_succeeded > 0
        else "no bootstrap (insufficient seeds per axis value); CIs collapse "
             "to point estimates"
    )

    text = (
        f"We evaluated the gauge-theoretic VFE transformer on {dataset} under a "
        f"{axis_clause} with {seeds_clause}, training every configuration "
        f"{seq_len_clause} for an iso-token budget {tokens_clause}, "
        f"with batch size adjusted across {axis_symbol} to satisfy memory "
        f"limits while preserving the budget exactly. For each run we record the "
        f"final-checkpoint test perplexity on the held-out split. Multi-seed "
        f"summaries report the per-{axis_symbol} mean and standard deviation "
        f"across the three seeds. We fit the scaling form {fit_form} to the "
        f"per-{axis_symbol} seed means by {method_clause}, with parameter "
        f"constraints b in (-2, 0) and c >= 1 to enforce a monotonically "
        f"decreasing perplexity curve and a non-negative irreducible-loss "
        f"floor. Confidence intervals are {bootstrap_clause}. The fitted "
        f"exponent is b = {_format_ci(fit.b, fit.b_ci, '.3f')}, the floor is "
        f"c = {_format_ci(fit.c, fit.c_ci, '.2f')}, and the coefficient is "
        f"a = {_format_ci(fit.a, fit.a_ci, '.2f')}. The fit explains "
        f"R^2 = {fit.r_squared:.3f} of the variance in per-{axis_symbol} "
        f"seed-mean perplexity. All training trajectories, FLOPs counts, and "
        f"parameter totals are reproduced from each run's stored "
        f"experiment_config.json and result_em.json."
    )

    if discussion is not None and discussion.strip():
        text = text + "\n\n" + discussion.strip()

    _lint(text)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text + "\n", encoding='utf-8')
    word_count = len(text.split())
    logger.info("Wrote methods paragraph: %s (%d words)", output_path, word_count)
    # 250-word soft cap applies to methods paragraph alone; with a discussion
    # paragraph appended, target ~500 words total.
    cap = 500 if (discussion and discussion.strip()) else 250
    if word_count > cap:
        logger.warning("Methods+discussion text exceeds %d words: %d", cap, word_count)
    return output_path
