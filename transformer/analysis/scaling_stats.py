r"""
Scaling-Law Statistics
======================

Power-law fit for scaling sweeps, with seed-resampled bootstrap CIs.

We fit

    PPL(x) = a \cdot x^{b} + c

on the per-axis seed means by default, where ``x`` is the swept axis (e.g.
``K``) and ``PPL`` is the held-out test perplexity. ``b`` is the scaling
exponent; ``c`` is the irreducible-loss floor. A log-log linear fit is
exposed as a second method (and as fallback when the nonlinear fit fails).

Bootstrap protocol: for each iteration, resample seeds independently within
each axis value (preserves the design matrix; does not conflate seed
variability with axis effects), recompute the per-axis mean, refit. CIs are
percentile intervals across successful fits.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PowerLawFit:
    """Result of a power-law fit on a scaling sweep.

    Point estimates are the central fit on the observed data. CIs are
    percentile intervals from the seed-resampled bootstrap. ``axis_grid``
    and the ``pred_*`` arrays are dense values for ribbon rendering.
    """

    a: float
    b: float
    c: float
    a_ci: Tuple[float, float]
    b_ci: Tuple[float, float]
    c_ci: Tuple[float, float]
    r_squared: float
    n_bootstrap: int
    n_bootstrap_succeeded: int
    seeds_used: List[int]
    axis_grid: np.ndarray
    pred_mean: np.ndarray
    pred_lo: np.ndarray
    pred_hi: np.ndarray
    method: Literal['nonlinear', 'loglog']


def _power_law(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return a * np.power(x, b) + c


def _initial_guess(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    y_min = float(np.nanmin(y))
    y_max = float(np.nanmax(y))
    return (max(y_max - y_min, 1.0), -0.3, max(y_min, 1.0))


def _fit_nonlinear(
    x: np.ndarray, y: np.ndarray,
) -> Optional[Tuple[float, float, float]]:
    if len(x) < 3:
        return None
    try:
        p0 = _initial_guess(x, y)
        bounds = ([0.0, -2.0, 1.0], [np.inf, 0.0, max(float(np.nanmax(y)), 2.0)])
        popt, _ = curve_fit(_power_law, x, y, p0=p0, bounds=bounds, maxfev=20000)
        return float(popt[0]), float(popt[1]), float(popt[2])
    except (RuntimeError, ValueError) as exc:
        logger.debug("Nonlinear fit failed: %s", exc)
        return None


def _fit_loglog(
    x: np.ndarray, y: np.ndarray,
) -> Optional[Tuple[float, float, float]]:
    """Fit log(y) = log(a) + b * log(x); equivalent to a * x^b with c=0."""
    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return None
    lx = np.log(x[mask])
    ly = np.log(y[mask])
    slope, intercept = np.polyfit(lx, ly, deg=1)
    return float(np.exp(intercept)), float(slope), 0.0


def _r_squared(y: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot <= 0:
        return float('nan')
    return 1.0 - ss_res / ss_tot


def _resample_seeds_per_axis(
    df: pd.DataFrame, axis_col: str, rng: np.random.Generator,
) -> pd.DataFrame:
    """Sample seeds with replacement within each axis value, return per-axis means."""
    rows = []
    for axis_val, group in df.groupby(axis_col):
        seeds = group['seed'].to_numpy()
        if len(seeds) == 0:
            continue
        sampled = rng.choice(seeds, size=len(seeds), replace=True)
        sampled_rows = pd.concat(
            [group[group['seed'] == s] for s in sampled], ignore_index=True,
        )
        rows.append({axis_col: axis_val, 'y': sampled_rows['y'].mean()})
    return pd.DataFrame(rows)


def fit_power_law(
    terminal_df: pd.DataFrame,
    axis_col: str = 'K',
    y_col: str = 'test_ppl',
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    method: Literal['nonlinear', 'loglog'] = 'nonlinear',
    rng_seed: int = 0,
    n_grid: int = 200,
) -> PowerLawFit:
    """Fit ``y = a*x^b + c`` (or log-log linear) with bootstrap CIs.

    Falls back to log-log linear if the nonlinear fit on the central data
    fails. Bootstrap iterations that fail are excluded; ``n_bootstrap_succeeded``
    is reported separately from the requested ``n_bootstrap``.
    """
    df = terminal_df.dropna(subset=[axis_col, y_col, 'seed']).copy()
    df = df.rename(columns={y_col: 'y'})
    if df.empty:
        raise ValueError(f"No non-null rows for ({axis_col}, {y_col})")

    seed_means = df.groupby(axis_col)['y'].mean().reset_index()
    x_obs = seed_means[axis_col].to_numpy(dtype=float)
    y_obs = seed_means['y'].to_numpy(dtype=float)

    central = None
    used_method: Literal['nonlinear', 'loglog'] = method
    if method == 'nonlinear':
        central = _fit_nonlinear(x_obs, y_obs)
        if central is None:
            logger.warning("Nonlinear fit failed on central data; falling back to log-log")
            used_method = 'loglog'
    if central is None:
        central = _fit_loglog(x_obs, y_obs)
    if central is None:
        raise RuntimeError("Both nonlinear and log-log fits failed on central data")

    a_hat, b_hat, c_hat = central
    y_pred_obs = _power_law(x_obs, a_hat, b_hat, c_hat)
    r2 = _r_squared(y_obs, y_pred_obs)

    rng = np.random.default_rng(rng_seed)
    a_samples: List[float] = []
    b_samples: List[float] = []
    c_samples: List[float] = []
    n_succeeded = 0
    seeds_per_axis_min = int(df.groupby(axis_col)['seed'].nunique().min())
    can_bootstrap = seeds_per_axis_min >= 2 and n_bootstrap > 0

    if can_bootstrap:
        for _ in range(n_bootstrap):
            sampled_means = _resample_seeds_per_axis(df, axis_col, rng)
            xs = sampled_means[axis_col].to_numpy(dtype=float)
            ys = sampled_means['y'].to_numpy(dtype=float)
            fit = (_fit_nonlinear(xs, ys) if used_method == 'nonlinear'
                   else _fit_loglog(xs, ys))
            if fit is None:
                continue
            a_samples.append(fit[0])
            b_samples.append(fit[1])
            c_samples.append(fit[2])
            n_succeeded += 1
    else:
        logger.warning(
            "Skipping bootstrap (min seeds per axis = %d); CIs collapse to point estimate",
            seeds_per_axis_min,
        )

    alpha = (1.0 - ci) / 2.0
    if n_succeeded >= 2:
        a_ci = (float(np.percentile(a_samples, 100 * alpha)),
                float(np.percentile(a_samples, 100 * (1 - alpha))))
        b_ci = (float(np.percentile(b_samples, 100 * alpha)),
                float(np.percentile(b_samples, 100 * (1 - alpha))))
        c_ci = (float(np.percentile(c_samples, 100 * alpha)),
                float(np.percentile(c_samples, 100 * (1 - alpha))))
    else:
        a_ci = b_ci = c_ci = (float('nan'), float('nan'))

    x_grid = np.logspace(np.log10(x_obs.min()), np.log10(x_obs.max()), n_grid)
    pred_mean = _power_law(x_grid, a_hat, b_hat, c_hat)
    if n_succeeded >= 2:
        ribbon_samples = np.array([
            _power_law(x_grid, a_samples[i], b_samples[i], c_samples[i])
            for i in range(n_succeeded)
        ])
        pred_lo = np.percentile(ribbon_samples, 100 * alpha, axis=0)
        pred_hi = np.percentile(ribbon_samples, 100 * (1 - alpha), axis=0)
    else:
        pred_lo = pred_mean.copy()
        pred_hi = pred_mean.copy()

    return PowerLawFit(
        a=a_hat, b=b_hat, c=c_hat,
        a_ci=a_ci, b_ci=b_ci, c_ci=c_ci,
        r_squared=r2,
        n_bootstrap=n_bootstrap,
        n_bootstrap_succeeded=n_succeeded,
        seeds_used=sorted(df['seed'].unique().astype(int).tolist()),
        axis_grid=x_grid,
        pred_mean=pred_mean,
        pred_lo=pred_lo,
        pred_hi=pred_hi,
        method=used_method,
    )


def seed_summary(
    terminal_df: pd.DataFrame,
    axis_col: str = 'K',
) -> pd.DataFrame:
    """Per-axis mean, std, sem, n_seeds for the headline metrics."""
    grouped = terminal_df.groupby(axis_col)
    out = pd.DataFrame({
        axis_col: list(grouped.groups.keys()),
    })

    def _agg(col: str, fn) -> List[float]:
        return [float(fn(grouped.get_group(k)[col].dropna()))
                if col in terminal_df.columns else float('nan')
                for k in out[axis_col]]

    for col in ('test_ppl', 'val_ppl'):
        out[f'mean_{col}'] = _agg(col, np.mean)
        out[f'std_{col}'] = _agg(col, np.std)
        out[f'sem_{col}'] = [s / max(len(grouped.get_group(k)), 1) ** 0.5
                              for s, k in zip(out[f'std_{col}'], out[axis_col])]
    out['n_seeds'] = [grouped.get_group(k)['seed'].nunique() for k in out[axis_col]]
    for col in ('total_params', 'flops_per_step', 'total_flops', 'tokens_seen'):
        out[f'mean_{col}'] = _agg(col, np.mean)
    return out.sort_values(axis_col).reset_index(drop=True)
