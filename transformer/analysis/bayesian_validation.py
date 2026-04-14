"""
Bayesian Validation of Gauge-Transformer Empirical Claims
==========================================================

Uses PyMC to build independent Bayesian reference models that validate
the manuscript's empirical claims with proper posterior uncertainty.

Models:
1. Hierarchical correlation model (144 heads × 105 passages)
2. Temperature τ posterior with theoretical comparison
3. Key-norm bias Bayesian effect sizes (α vs β)

Each model loads data from validation_results.json and produces:
- ArviZ InferenceData with full posterior traces
- Publication-ready summary statistics
- Diagnostics (R-hat, ESS, posterior predictive checks)

Author: Claude (Bayesian validation for JMLR submission)
Date: March 2026
"""

import json
import warnings
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, List, Tuple, Any

import numpy as np

try:
    import pymc as pm
    PYMC_AVAILABLE = True
except ImportError:
    pm = None
    PYMC_AVAILABLE = False


# =============================================================================
# Data Loading
# =============================================================================

@dataclass
class ValidationData:
    """Parsed validation results for Bayesian modeling."""
    # Per-head summaries (144 heads)
    layers: np.ndarray          # (144,) int — layer index 0..11
    heads: np.ndarray           # (144,) int — head index 0..11
    mean_r: np.ndarray          # (144,) — mean correlation across passages
    std_r: np.ndarray           # (144,) — std across passages
    se_r: np.ndarray            # (144,) — standard error
    median_r: np.ndarray        # (144,)
    keynorm_alpha: np.ndarray   # (144,) — dot-product key-norm correlation
    keynorm_beta: np.ndarray    # (144,) — KL-distance key-norm correlation

    # Tau sweep
    tau_values: np.ndarray      # (n_tau,)
    tau_mean_r: np.ndarray      # (n_tau,)
    tau_ci_lo: np.ndarray       # (n_tau,)
    tau_ci_hi: np.ndarray       # (n_tau,)

    # Multi-model
    model_names: List[str] = field(default_factory=list)
    model_mean_r: np.ndarray = field(default_factory=lambda: np.array([]))
    model_std_r: np.ndarray = field(default_factory=lambda: np.array([]))
    model_head_dim: np.ndarray = field(default_factory=lambda: np.array([]))
    model_n_params: np.ndarray = field(default_factory=lambda: np.array([]))
    model_n_heads: np.ndarray = field(default_factory=lambda: np.array([]))

    # Sequence length
    seq_lens: np.ndarray = field(default_factory=lambda: np.array([]))
    seqlen_mean_r: np.ndarray = field(default_factory=lambda: np.array([]))
    seqlen_std_r: np.ndarray = field(default_factory=lambda: np.array([]))

    # Entropy
    entropy_alpha: float = 0.0
    entropy_beta: float = 0.0
    entropy_ratio: float = 0.0

    # Metadata
    n_passages: int = 105
    head_dim: int = 64
    tau_opt: float = 19.0


def load_validation_data(json_path: str | Path) -> ValidationData:
    """Load validation_results.json into structured arrays."""
    with open(json_path) as f:
        data = json.load(f)

    # Per-head data
    per_head = data['phase2_multi_passage']['per_head']
    n_heads = len(per_head)
    layers = np.array([h['layer'] for h in per_head])
    heads = np.array([h['head'] for h in per_head])
    mean_r = np.array([h['mean_r'] for h in per_head])
    std_r = np.array([h['std_r'] for h in per_head])
    se_r = np.array([h['se_r'] for h in per_head])
    median_r = np.array([h['median_r'] for h in per_head])
    keynorm_alpha = np.array([h['mean_keynorm_alpha'] for h in per_head])
    keynorm_beta = np.array([h['mean_keynorm_beta'] for h in per_head])

    # Tau sweep
    tau_sweep = data['phase3_tau_sweep']['all_taus']
    tau_values = np.array([float(k) for k in tau_sweep.keys()])
    tau_mean_r = np.array([v['mean_r'] for v in tau_sweep.values()])
    tau_ci_lo = np.array([v['ci_lo'] for v in tau_sweep.values()])
    tau_ci_hi = np.array([v['ci_hi'] for v in tau_sweep.values()])

    # Multi-model
    multi = data['phase4_multi_model']
    model_names = list(multi.keys())
    model_mean_r = np.array([multi[m]['grand_mean_r'] for m in model_names])
    model_std_r = np.array([multi[m]['grand_std_r'] for m in model_names])
    model_head_dim = np.array([multi[m]['head_dim'] for m in model_names])
    model_n_params = np.array([multi[m]['n_params'] for m in model_names])
    model_n_heads = np.array([multi[m]['n_heads'] for m in model_names])

    # Sequence length
    seqlen = data['phase7_seqlen']
    seq_lens = np.array([s['seq_len'] for s in seqlen])
    seqlen_mean_r = np.array([s['mean_r'] for s in seqlen])
    seqlen_std_r = np.array([s['std_r'] for s in seqlen])

    # Entropy
    ent = data['phase6_entropy']

    corpus = data['phase2_multi_passage']['corpus_summary']

    return ValidationData(
        layers=layers, heads=heads,
        mean_r=mean_r, std_r=std_r, se_r=se_r, median_r=median_r,
        keynorm_alpha=keynorm_alpha, keynorm_beta=keynorm_beta,
        tau_values=tau_values, tau_mean_r=tau_mean_r,
        tau_ci_lo=tau_ci_lo, tau_ci_hi=tau_ci_hi,
        model_names=model_names,
        model_mean_r=model_mean_r, model_std_r=model_std_r,
        model_head_dim=model_head_dim, model_n_params=model_n_params,
        model_n_heads=model_n_heads,
        seq_lens=seq_lens, seqlen_mean_r=seqlen_mean_r,
        seqlen_std_r=seqlen_std_r,
        entropy_alpha=ent['mean_entropy_alpha'],
        entropy_beta=ent['mean_entropy_beta'],
        entropy_ratio=ent['mean_entropy_ratio'],
        n_passages=corpus['n_passages'],
        head_dim=64,
        tau_opt=data['phase3_tau_sweep']['best_tau'],
    )


# =============================================================================
# Model 1: Hierarchical Correlation Model
# =============================================================================

def build_hierarchical_correlation_model(
    vd: ValidationData,
    *,
    n_samples: int = 2000,
    n_tune: int = 1000,
    random_seed: int = 42,
) -> Tuple[Any, Any]:
    """
    Hierarchical Bayesian model for BERT head correlations.

    Structure:
        grand_mu ~ Normal(0.7, 0.3)           # population mean (Fisher-z)
        layer_sigma ~ HalfNormal(0.5)         # between-layer variance
        layer_offset[12] ~ Normal(0, layer_sigma)  # layer random effects
        head_sigma ~ HalfNormal(0.3)          # within-layer, between-head variance
        head_offset[144] ~ Normal(0, head_sigma)   # head random effects
        z_hat[h] = grand_mu + layer_offset[layer[h]] + head_offset[h]
        r_hat[h] = tanh(z_hat[h])             # back to correlation scale
        obs_r[h] ~ Normal(r_hat[h], se_r[h])  # observed with known SE

    Uses Fisher-z transform for proper correlation modeling.

    Returns (model, idata).
    """
    # Fisher-z transform observed correlations
    # Clip to avoid infinities at ±1. Use 0.99 (not 0.999) because
    # se_z ≈ se_r / (1 - r²) inflates ~500x near |r|=0.999, which
    # destabilizes MCMC. At 0.99 the max inflation is ~50x.
    r_clipped = np.clip(vd.mean_r, -0.99, 0.99)
    z_obs = np.arctanh(r_clipped)

    # SE in z-space: se_z ≈ se_r / (1 - r²)  (delta method)
    se_z = vd.se_r / (1 - r_clipped**2)

    with pm.Model() as model:
        # Hyperpriors
        grand_mu = pm.Normal('grand_mu', mu=0.8, sigma=0.5)
        layer_sigma = pm.HalfNormal('layer_sigma', sigma=0.5)
        head_sigma = pm.HalfNormal('head_sigma', sigma=0.3)

        # Layer random effects (12 layers)
        layer_offset = pm.Normal('layer_offset', mu=0, sigma=layer_sigma, shape=12)

        # Head random effects (144 heads)
        head_offset = pm.Normal('head_offset', mu=0, sigma=head_sigma, shape=144)

        # Linear predictor in z-space
        z_hat = grand_mu + layer_offset[vd.layers] + head_offset

        # Back-transform to r-space for derived quantities
        r_hat = pm.Deterministic('r_hat', pm.math.tanh(z_hat))

        # Grand mean on correlation scale
        grand_r = pm.Deterministic('grand_r', pm.math.tanh(grand_mu))

        # Layer means on correlation scale
        layer_r = pm.Deterministic(
            'layer_r', pm.math.tanh(grand_mu + layer_offset)
        )

        # Fraction of heads with r > 0.8 (posterior predictive)
        frac_gt_08 = pm.Deterministic(
            'frac_gt_08', pm.math.sum(r_hat > 0.8) / 144.0
        )

        # Likelihood
        pm.Normal('obs_z', mu=z_hat, sigma=se_z, observed=z_obs)

        # Sample
        idata = pm.sample(
            draws=n_samples, tune=n_tune,
            random_seed=random_seed,
            return_inferencedata=True,
            progressbar=True,
        )

    return model, idata


# =============================================================================
# Model 2: Temperature τ Posterior
# =============================================================================

def build_temperature_model(
    vd: ValidationData,
    *,
    n_samples: int = 2000,
    n_tune: int = 1000,
    random_seed: int = 42,
) -> Tuple[Any, Any]:
    """
    Bayesian model for optimal temperature τ.

    The theoretical prediction is τ = 2√d_k where d_k is head dimension.
    We directly parametrize the peak location and shape:

        r(τ) = r_max - κ · (log(τ) - log(τ_opt))²

    where κ controls curvature. This is a concave quadratic in log-space
    centered at τ_opt, which naturally captures the asymmetric shape in
    linear-τ space (slow rise from small τ, faster fall at large τ).

    Priors:
        τ_opt ~ LogNormal(log(16), 0.3)   # centered on theory: 2√64=16
        r_max ~ Normal(0.81, 0.05)         # peak correlation
        κ ~ HalfNormal(0.02)              # curvature
        σ ~ HalfNormal(0.005)             # observation noise

    Returns (model, idata).
    """
    tau = vd.tau_values
    r_obs = vd.tau_mean_r
    log_tau = np.log(tau)

    # Observation uncertainty from CI width
    se_obs = (vd.tau_ci_hi - vd.tau_ci_lo) / (2 * 1.96)

    with pm.Model() as model:
        # Direct parametrization of peak
        log_tau_opt = pm.Normal('log_tau_opt', mu=np.log(16.0), sigma=0.3)
        tau_opt = pm.Deterministic('tau_opt', pm.math.exp(log_tau_opt))

        r_max = pm.Normal('r_max', mu=0.81, sigma=0.05)
        kappa = pm.HalfNormal('kappa', sigma=0.02)
        sigma_obs = pm.HalfNormal('sigma_obs', sigma=0.005)

        # Concave quadratic in log-space
        r_pred = r_max - kappa * (log_tau - log_tau_opt)**2

        # Deviation from theoretical prediction
        tau_theory = 2.0 * np.sqrt(vd.head_dim)  # = 16.0
        deviation_ratio = pm.Deterministic(
            'deviation_ratio', tau_opt / tau_theory
        )
        deviation_pct = pm.Deterministic(
            'deviation_pct', (tau_opt - tau_theory) / tau_theory * 100
        )

        # Probability that τ_opt > τ_theory
        prob_gt_theory = pm.Deterministic(
            'prob_gt_theory', (tau_opt > tau_theory).astype('float64')
        )

        # Likelihood
        pm.Normal('obs_r', mu=r_pred, sigma=pm.math.sqrt(sigma_obs**2 + se_obs**2),
                  observed=r_obs)

        idata = pm.sample(
            draws=n_samples, tune=n_tune,
            random_seed=random_seed,
            return_inferencedata=True,
            progressbar=True,
        )

    return model, idata


# =============================================================================
# Model 3: Key-Norm Bias Effect Sizes
# =============================================================================

def build_keynorm_model(
    vd: ValidationData,
    *,
    n_samples: int = 2000,
    n_tune: int = 1000,
    random_seed: int = 42,
) -> Tuple[Any, Any]:
    """
    Bayesian comparison of key-norm bias between α (dot-product) and β (KL).

    The theory predicts that KL-distance attention (β) shows stronger key-norm
    bias than dot-product attention (α), because β ∝ exp(-||q-k||²/τ)
    explicitly depends on ||k||².

    Model:
        For each attention type t ∈ {α, β} and head h:
            ρ_t ~ HierarchicalNormal(μ_t, σ_t)

        Population means:
            μ_α ~ Normal(0, 0.5)
            μ_β ~ Normal(-0.5, 0.5)
            σ_α ~ HalfNormal(0.3)
            σ_β ~ HalfNormal(0.3)

        Effect size (Cohen's d):
            d = (|μ_β| - |μ_α|) / pooled_σ

    Returns (model, idata).
    """
    rho_alpha = vd.keynorm_alpha  # (144,)
    rho_beta = vd.keynorm_beta    # (144,)

    with pm.Model() as model:
        # Population parameters for dot-product key-norm correlation
        mu_alpha = pm.Normal('mu_alpha', mu=0, sigma=0.5)
        sigma_alpha = pm.HalfNormal('sigma_alpha', sigma=0.3)

        # Population parameters for KL-distance key-norm correlation
        mu_beta = pm.Normal('mu_beta', mu=-0.5, sigma=0.5)
        sigma_beta = pm.HalfNormal('sigma_beta', sigma=0.3)

        # Per-head random effects
        alpha_h = pm.Normal('alpha_h', mu=mu_alpha, sigma=sigma_alpha, shape=144)
        beta_h = pm.Normal('beta_h', mu=mu_beta, sigma=sigma_beta, shape=144)

        # Observation noise (measurement uncertainty from cross-passage averaging)
        sigma_obs = pm.HalfNormal('sigma_obs', sigma=0.1)

        # Observations
        pm.Normal('obs_alpha', mu=alpha_h, sigma=sigma_obs, observed=rho_alpha)
        pm.Normal('obs_beta', mu=beta_h, sigma=sigma_obs, observed=rho_beta)

        # Derived quantities
        abs_mu_alpha = pm.Deterministic('abs_mu_alpha', pm.math.abs(mu_alpha))
        abs_mu_beta = pm.Deterministic('abs_mu_beta', pm.math.abs(mu_beta))

        # Effect size: difference in absolute key-norm correlation
        diff_abs = pm.Deterministic('diff_abs', abs_mu_beta - abs_mu_alpha)

        # Pooled SD for Cohen's d
        pooled_sigma = pm.math.sqrt(
            (sigma_alpha**2 + sigma_beta**2) / 2
        )
        cohens_d = pm.Deterministic('cohens_d', diff_abs / pooled_sigma)

        # Probability of direction (β shows stronger bias)
        prob_beta_stronger = pm.Deterministic(
            'prob_beta_stronger',
            (abs_mu_beta > abs_mu_alpha).astype('float64')
        )

        idata = pm.sample(
            draws=n_samples, tune=n_tune,
            random_seed=random_seed,
            return_inferencedata=True,
            progressbar=True,
        )

    return model, idata
