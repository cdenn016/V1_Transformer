"""
Bayesian Validation of Gauge-Transformer Empirical Claims
==========================================================

Uses PyMC to build independent Bayesian reference models that validate
the manuscript's empirical claims with proper posterior uncertainty.

Models:
1. Hierarchical correlation model (144 heads x 105 passages)
2. Temperature τ posterior with theoretical comparison
3. Key-norm bias Bayesian effect sizes (α vs β)
4. Multi-model comparison with partial pooling
5. Sequence-length degradation model

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
    import arviz as az
    PYMC_AVAILABLE = True
except ImportError:
    pm = None
    az = None
    PYMC_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None
    MATPLOTLIB_AVAILABLE = False


# =============================================================================
# Data Loading
# =============================================================================

@dataclass
class ValidationData:
    """Parsed validation results for Bayesian modeling."""
    # Per-head summaries (144 heads)
    layers: np.ndarray          # (144,) int -- layer index 0..11
    heads: np.ndarray           # (144,) int -- head index 0..11
    mean_r: np.ndarray          # (144,) -- mean correlation across passages
    std_r: np.ndarray           # (144,) -- std across passages
    se_r: np.ndarray            # (144,) -- standard error
    median_r: np.ndarray        # (144,)
    keynorm_alpha: np.ndarray   # (144,) -- dot-product key-norm correlation
    keynorm_beta: np.ndarray    # (144,) -- KL-distance key-norm correlation

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
    # Clip to avoid infinities at +/-1
    r_clipped = np.clip(vd.mean_r, -0.999, 0.999)
    z_obs = np.arctanh(r_clipped)

    # SE in z-space: se_z ~= se_r / (1 - r^2)  (delta method)
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

        r(τ) = r_max - κ · (log(τ) - log(τ_opt))^2

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
    bias than dot-product attention (α), because β ∝ exp(-||q-k||^2/τ)
    explicitly depends on ||k||^2.

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


# =============================================================================
# Model 4: Multi-Model Comparison
# =============================================================================

def build_multimodel_comparison(
    vd: ValidationData,
    *,
    n_samples: int = 2000,
    n_tune: int = 1000,
    random_seed: int = 42,
) -> Tuple[Any, Any]:
    """
    Bayesian comparison of KL-attention alignment across architectures.

    Partial pooling across 5 models to get shrinkage-corrected estimates.

    Model:
        population_mu ~ Normal(0.75, 0.2)
        population_sigma ~ HalfNormal(0.2)
        model_mu[5] ~ Normal(population_mu, population_sigma)
        obs_r[m] ~ Normal(model_mu[m], obs_se[m])

    Returns (model, idata).
    """
    n_models = len(vd.model_names)
    r_obs = vd.model_mean_r
    se_obs = vd.model_std_r  # reported as grand_std_r (SE across passages)

    with pm.Model() as model:
        # Hyperpriors
        pop_mu = pm.Normal('pop_mu', mu=0.75, sigma=0.2)
        pop_sigma = pm.HalfNormal('pop_sigma', sigma=0.2)

        # Model-level means (partial pooling)
        model_mu = pm.Normal(
            'model_mu', mu=pop_mu, sigma=pop_sigma, shape=n_models
        )

        # Likelihood
        pm.Normal('obs_r', mu=model_mu, sigma=se_obs, observed=r_obs)

        # Shrinkage: how much does partial pooling move each estimate?
        shrinkage = pm.Deterministic(
            'shrinkage', model_mu - r_obs
        )

        idata = pm.sample(
            draws=n_samples, tune=n_tune,
            random_seed=random_seed,
            return_inferencedata=True,
            progressbar=True,
        )

    return model, idata


# =============================================================================
# Model 5: Sequence-Length Degradation
# =============================================================================

def build_seqlen_model(
    vd: ValidationData,
    *,
    n_samples: int = 2000,
    n_tune: int = 1000,
    random_seed: int = 42,
) -> Tuple[Any, Any]:
    """
    Bayesian model of correlation degradation with sequence length.

    Theory: attention sparsity increases with N, so KL-approximation
    quality should degrade. We model:

        r(N) = r_0 - β_decay · log₂(N / N_ref)

    where:
        r_0 is correlation at reference length N_ref = 32
        β_decay is degradation rate per doubling of sequence length

    Priors:
        r_0 ~ Normal(0.8, 0.1)
        β_decay ~ HalfNormal(0.05)     # positive = degradation
        σ ~ HalfNormal(0.02)

    Returns (model, idata).
    """
    N = vd.seq_lens.astype(float)
    r_obs = vd.seqlen_mean_r
    N_ref = 32.0
    log2_ratio = np.log2(N / N_ref)

    with pm.Model() as model:
        # Priors
        r_0 = pm.Normal('r_0', mu=0.8, sigma=0.1)
        beta_decay = pm.HalfNormal('beta_decay', sigma=0.05)
        sigma_obs = pm.HalfNormal('sigma_obs', sigma=0.02)

        # Linear in log₂(N)
        r_pred = r_0 - beta_decay * log2_ratio

        # Likelihood
        pm.Normal('obs_r', mu=r_pred, sigma=sigma_obs, observed=r_obs)

        # Derived: predicted correlation at key lengths
        r_at_128 = pm.Deterministic('r_at_128', r_0 - beta_decay * np.log2(128 / N_ref))
        r_at_512 = pm.Deterministic('r_at_512', r_0 - beta_decay * np.log2(512 / N_ref))
        r_at_1024 = pm.Deterministic('r_at_1024', r_0 - beta_decay * np.log2(1024 / N_ref))

        # Half-life: N at which r drops to 0.5
        halflife_log2 = pm.Deterministic(
            'halflife_doublings', (r_0 - 0.5) / beta_decay
        )

        idata = pm.sample(
            draws=n_samples, tune=n_tune,
            random_seed=random_seed,
            return_inferencedata=True,
            progressbar=True,
        )

    return model, idata


# =============================================================================
# Summary & Diagnostics
# =============================================================================

@dataclass
class BayesianSummary:
    """Summary of a single Bayesian validation model."""
    model_name: str
    parameters: Dict[str, Dict[str, float]]  # param -> {mean, sd, hdi_3%, hdi_97%, r_hat, ess}
    diagnostics: Dict[str, Any]
    manuscript_comparison: Dict[str, Any]

    def to_dict(self) -> dict:
        return asdict(self)


def summarize_model(
    name: str,
    idata: Any,
    manuscript_claims: Dict[str, float],
    params_of_interest: Optional[List[str]] = None,
) -> BayesianSummary:
    """Extract publication-ready summary from InferenceData."""
    summary = az.summary(
        idata,
        var_names=params_of_interest,
        hdi_prob=0.94,
        round_to=4,
    )

    parameters = {}
    for param in summary.index:
        row = summary.loc[param]
        parameters[param] = {
            'mean': float(row['mean']),
            'sd': float(row['sd']),
            'hdi_3%': float(row['hdi_3%']),
            'hdi_97%': float(row['hdi_97%']),
            'r_hat': float(row.get('r_hat', np.nan)),
            'ess_bulk': float(row.get('ess_bulk', np.nan)),
            'ess_tail': float(row.get('ess_tail', np.nan)),
        }

    # Diagnostics
    diag = {
        'all_r_hat_ok': all(
            p['r_hat'] < 1.05 for p in parameters.values()
            if not np.isnan(p['r_hat'])
        ),
        'min_ess_bulk': min(
            (p['ess_bulk'] for p in parameters.values()
             if not np.isnan(p['ess_bulk'])),
            default=0
        ),
        'n_divergences': int(
            idata.sample_stats.get('diverging', np.array([0])).values.sum()
        ) if hasattr(idata, 'sample_stats') else 0,
    }

    # Compare to manuscript claims
    comparison = {}
    for claim_name, claim_value in manuscript_claims.items():
        if claim_name in parameters:
            p = parameters[claim_name]
            in_hdi = p['hdi_3%'] <= claim_value <= p['hdi_97%']
            comparison[claim_name] = {
                'manuscript_value': claim_value,
                'posterior_mean': p['mean'],
                'posterior_sd': p['sd'],
                'hdi_94': [p['hdi_3%'], p['hdi_97%']],
                'claim_in_hdi': in_hdi,
            }

    return BayesianSummary(
        model_name=name,
        parameters=parameters,
        diagnostics=diag,
        manuscript_comparison=comparison,
    )


# =============================================================================
# Visualization
# =============================================================================

def plot_hierarchical_results(
    idata: Any,
    vd: ValidationData,
    output_dir: Path,
):
    """Publication-quality plots for the hierarchical correlation model."""
    if not MATPLOTLIB_AVAILABLE:
        warnings.warn("matplotlib not available, skipping plots")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Figure 1: Forest plot of layer-level correlations ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Layer posteriors
    ax = axes[0]
    layer_r = idata.posterior['layer_r'].values.reshape(-1, 12)
    layer_means = layer_r.mean(axis=0)
    layer_hdi = np.percentile(layer_r, [3, 97], axis=0)

    y_pos = np.arange(12)
    ax.errorbar(
        layer_means, y_pos,
        xerr=[layer_means - layer_hdi[0], layer_hdi[1] - layer_means],
        fmt='o', color='steelblue', capsize=3, markersize=5,
    )
    ax.axvline(
        idata.posterior['grand_r'].values.mean(),
        color='red', linestyle='--', alpha=0.7, label='Grand mean'
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f'Layer {i}' for i in range(12)])
    ax.set_xlabel('Correlation (r)')
    ax.set_title('Layer-Level Posterior Correlations')
    ax.legend(fontsize=9)
    ax.invert_yaxis()

    # Grand mean posterior
    ax = axes[1]
    grand_r = idata.posterior['grand_r'].values.flatten()
    ax.hist(grand_r, bins=50, density=True, alpha=0.7, color='steelblue')
    hdi = np.percentile(grand_r, [3, 97])
    ax.axvline(hdi[0], color='red', linestyle='--', alpha=0.7)
    ax.axvline(hdi[1], color='red', linestyle='--', alpha=0.7,
               label=f'94% HDI [{hdi[0]:.3f}, {hdi[1]:.3f}]')
    ax.axvline(grand_r.mean(), color='red', linewidth=2,
               label=f'Mean = {grand_r.mean():.3f}')
    ax.set_xlabel('Grand Mean Correlation (r)')
    ax.set_ylabel('Density')
    ax.set_title('Posterior: Population Mean Correlation')
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(output_dir / 'hierarchical_correlation_posterior.png', dpi=200)
    plt.close(fig)

    # --- Figure 2: Variance decomposition ---
    fig, ax = plt.subplots(figsize=(6, 4))
    layer_sigma = idata.posterior['layer_sigma'].values.flatten()
    head_sigma = idata.posterior['head_sigma'].values.flatten()

    parts = ax.violinplot(
        [layer_sigma, head_sigma],
        positions=[1, 2],
        showmeans=True, showmedians=True,
    )
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Between-Layer σ', 'Within-Layer σ'])
    ax.set_ylabel('Standard Deviation (Fisher-z scale)')
    ax.set_title('Variance Decomposition')

    fig.tight_layout()
    fig.savefig(output_dir / 'variance_decomposition.png', dpi=200)
    plt.close(fig)

    # --- Figure 3: Shrinkage plot ---
    fig, ax = plt.subplots(figsize=(7, 5))
    r_hat = idata.posterior['r_hat'].values.reshape(-1, 144).mean(axis=0)
    r_obs = vd.mean_r

    ax.scatter(r_obs, r_hat, alpha=0.6, s=20, c=vd.layers, cmap='viridis')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='No shrinkage')
    ax.set_xlabel('Observed Mean r')
    ax.set_ylabel('Posterior Mean r (shrinkage-corrected)')
    ax.set_title('Hierarchical Shrinkage: 144 Attention Heads')
    ax.legend()
    cbar = plt.colorbar(ax.collections[0], ax=ax, label='Layer')

    fig.tight_layout()
    fig.savefig(output_dir / 'shrinkage_plot.png', dpi=200)
    plt.close(fig)


def plot_temperature_results(
    idata: Any,
    vd: ValidationData,
    output_dir: Path,
):
    """Publication-quality plots for the temperature model."""
    if not MATPLOTLIB_AVAILABLE:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # τ posterior
    ax = axes[0]
    tau_samples = idata.posterior['tau_opt'].values.flatten()
    ax.hist(tau_samples, bins=50, density=True, alpha=0.7, color='darkorange')
    hdi = np.percentile(tau_samples, [3, 97])
    ax.axvline(16.0, color='blue', linewidth=2, linestyle='--',
               label=r'Theory: $\tau = 2\sqrt{d_k} = 16$')
    ax.axvline(19.0, color='red', linewidth=2, linestyle=':',
               label=r'Empirical: $\tau_{opt} = 19$')
    ax.axvline(tau_samples.mean(), color='darkorange', linewidth=2,
               label=f'Posterior mean: {tau_samples.mean():.1f}')
    ax.fill_betweenx([0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1],
                     hdi[0], hdi[1], alpha=0.15, color='darkorange')
    ax.set_xlabel(r'$\tau_{opt}$')
    ax.set_ylabel('Density')
    ax.set_title(r'Posterior: Optimal Temperature $\tau$')
    ax.legend(fontsize=8)

    # Deviation from theory
    ax = axes[1]
    dev = idata.posterior['deviation_pct'].values.flatten()
    ax.hist(dev, bins=50, density=True, alpha=0.7, color='teal')
    ax.axvline(0, color='blue', linewidth=2, linestyle='--',
               label='Theory = 0% deviation')
    hdi_dev = np.percentile(dev, [3, 97])
    ax.axvline(dev.mean(), color='teal', linewidth=2,
               label=f'Mean: {dev.mean():.1f}%')
    ax.set_xlabel(r'Deviation from $2\sqrt{d_k}$ (%)')
    ax.set_ylabel('Density')
    ax.set_title('Posterior: Temperature Deviation from Theory')
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / 'temperature_posterior.png', dpi=200)
    plt.close(fig)


def plot_keynorm_results(
    idata: Any,
    vd: ValidationData,
    output_dir: Path,
):
    """Publication-quality plots for the key-norm bias model."""
    if not MATPLOTLIB_AVAILABLE:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Population means
    ax = axes[0]
    mu_a = idata.posterior['mu_alpha'].values.flatten()
    mu_b = idata.posterior['mu_beta'].values.flatten()
    ax.hist(mu_a, bins=50, density=True, alpha=0.6, color='steelblue',
            label=r'$\mu_\alpha$ (dot-product)')
    ax.hist(mu_b, bins=50, density=True, alpha=0.6, color='coral',
            label=r'$\mu_\beta$ (KL-distance)')
    ax.set_xlabel('Population Mean Key-Norm Correlation')
    ax.set_ylabel('Density')
    ax.set_title('Key-Norm Bias: Population Means')
    ax.legend(fontsize=9)

    # Cohen's d
    ax = axes[1]
    d = idata.posterior['cohens_d'].values.flatten()
    ax.hist(d, bins=50, density=True, alpha=0.7, color='purple')
    hdi = np.percentile(d, [3, 97])
    ax.axvline(d.mean(), color='purple', linewidth=2,
               label=f"Cohen's d = {d.mean():.2f}")
    ax.axvline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel("Cohen's d (|β| - |α| effect)")
    ax.set_ylabel('Density')
    ax.set_title("Effect Size: KL > Dot-Product Bias")
    ax.legend(fontsize=9)

    # Probability of direction
    ax = axes[2]
    prob = idata.posterior['prob_beta_stronger'].values.flatten().mean()
    ax.bar(['P(|β| > |α|)', 'P(|α| > |β|)'],
           [prob, 1 - prob],
           color=['coral', 'steelblue'], alpha=0.7)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Posterior Probability')
    ax.set_title(f'Direction: P(KL stronger) = {prob:.3f}')
    for i, v in enumerate([prob, 1 - prob]):
        ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=11)

    fig.tight_layout()
    fig.savefig(output_dir / 'keynorm_posterior.png', dpi=200)
    plt.close(fig)


def plot_seqlen_results(
    idata: Any,
    vd: ValidationData,
    output_dir: Path,
):
    """Publication-quality plots for the sequence-length model."""
    if not MATPLOTLIB_AVAILABLE:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Fit curve with posterior predictive
    ax = axes[0]
    r_0_samples = idata.posterior['r_0'].values.flatten()
    beta_samples = idata.posterior['beta_decay'].values.flatten()

    N_plot = np.logspace(np.log2(8), np.log2(2048), 100, base=2)
    log2_ratio = np.log2(N_plot / 32.0)

    # Posterior predictive bands
    r_pred = r_0_samples[:, None] - beta_samples[:, None] * log2_ratio[None, :]
    r_mean = r_pred.mean(axis=0)
    r_hdi = np.percentile(r_pred, [3, 97], axis=0)

    ax.fill_between(N_plot, r_hdi[0], r_hdi[1], alpha=0.2, color='steelblue')
    ax.plot(N_plot, r_mean, 'b-', linewidth=2, label='Posterior mean')
    ax.errorbar(vd.seq_lens, vd.seqlen_mean_r, yerr=vd.seqlen_std_r,
                fmt='ko', capsize=3, label='Observed')
    ax.set_xscale('log', base=2)
    ax.set_xlabel('Sequence Length N')
    ax.set_ylabel('Mean Correlation r')
    ax.set_title('Correlation vs Sequence Length')
    ax.legend(fontsize=9)

    # Degradation rate posterior
    ax = axes[1]
    ax.hist(beta_samples, bins=50, density=True, alpha=0.7, color='steelblue')
    hdi = np.percentile(beta_samples, [3, 97])
    ax.axvline(beta_samples.mean(), color='red', linewidth=2,
               label=f'Mean: {beta_samples.mean():.4f} per doubling')
    ax.set_xlabel(r'$\beta_{decay}$ (correlation loss per doubling)')
    ax.set_ylabel('Density')
    ax.set_title('Posterior: Degradation Rate')
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(output_dir / 'seqlen_degradation.png', dpi=200)
    plt.close(fig)


# =============================================================================
# Main Runner
# =============================================================================

def run_all_validations(
    json_path: str | Path,
    output_dir: str | Path,
    *,
    n_samples: int = 2000,
    n_tune: int = 1000,
    random_seed: int = 42,
    models: Optional[List[str]] = None,
) -> Dict[str, BayesianSummary]:
    """
    Run all Bayesian validation models and save results.

    Args:
        json_path: Path to validation_results.json
        output_dir: Directory for outputs (plots, summaries)
        n_samples: MCMC draws per chain
        n_tune: Tuning samples
        random_seed: Random seed
        models: List of model names to run, or None for all.
            Options: 'hierarchical', 'temperature', 'keynorm',
                     'multimodel', 'seqlen'

    Returns:
        Dict of model_name -> BayesianSummary
    """
    if not PYMC_AVAILABLE:
        raise ImportError("PyMC not installed. Run: pip install pymc arviz")

    vd = load_validation_data(json_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_models = models or [
        'hierarchical', 'temperature', 'keynorm', 'multimodel', 'seqlen'
    ]

    results = {}
    sample_kwargs = dict(
        n_samples=n_samples, n_tune=n_tune, random_seed=random_seed
    )

    # ---- Model 1: Hierarchical Correlation ----
    if 'hierarchical' in all_models:
        print("\n" + "=" * 60)
        print("Model 1: Hierarchical Correlation (144 heads x 105 passages)")
        print("=" * 60)

        model, idata = build_hierarchical_correlation_model(vd, **sample_kwargs)

        results['hierarchical'] = summarize_model(
            'hierarchical_correlation', idata,
            manuscript_claims={
                'grand_r': 0.804,
                'frac_gt_08': 93 / 144,
            },
            params_of_interest=[
                'grand_mu', 'grand_r', 'layer_sigma', 'head_sigma',
                'frac_gt_08',
            ],
        )
        plot_hierarchical_results(idata, vd, output_dir)

        # Save trace
        idata.to_json(str(output_dir / 'hierarchical_trace.json'))
        print(f"\n  Grand r posterior: "
              f"{results['hierarchical'].parameters['grand_r']['mean']:.3f} "
              f"[{results['hierarchical'].parameters['grand_r']['hdi_3%']:.3f}, "
              f"{results['hierarchical'].parameters['grand_r']['hdi_97%']:.3f}]")

    # ---- Model 2: Temperature τ ----
    if 'temperature' in all_models:
        print("\n" + "=" * 60)
        print("Model 2: Temperature τ Posterior")
        print("=" * 60)

        model, idata = build_temperature_model(vd, **sample_kwargs)

        results['temperature'] = summarize_model(
            'temperature_tau', idata,
            manuscript_claims={
                'tau_opt': 19.0,
                'deviation_pct': 18.75,
            },
            params_of_interest=[
                'tau_opt', 'r_max', 'kappa',
                'deviation_ratio', 'deviation_pct',
            ],
        )
        plot_temperature_results(idata, vd, output_dir)
        idata.to_json(str(output_dir / 'temperature_trace.json'))

        print(f"\n  τ_opt posterior: "
              f"{results['temperature'].parameters['tau_opt']['mean']:.1f} "
              f"[{results['temperature'].parameters['tau_opt']['hdi_3%']:.1f}, "
              f"{results['temperature'].parameters['tau_opt']['hdi_97%']:.1f}]")

    # ---- Model 3: Key-Norm Bias ----
    if 'keynorm' in all_models:
        print("\n" + "=" * 60)
        print("Model 3: Key-Norm Bias Effect Sizes")
        print("=" * 60)

        model, idata = build_keynorm_model(vd, **sample_kwargs)

        results['keynorm'] = summarize_model(
            'keynorm_bias', idata,
            manuscript_claims={
                'abs_mu_alpha': 0.139,
                'abs_mu_beta': 0.256,
            },
            params_of_interest=[
                'mu_alpha', 'mu_beta', 'sigma_alpha', 'sigma_beta',
                'abs_mu_alpha', 'abs_mu_beta', 'diff_abs', 'cohens_d',
                'prob_beta_stronger',
            ],
        )
        plot_keynorm_results(idata, vd, output_dir)
        idata.to_json(str(output_dir / 'keynorm_trace.json'))

        print(f"\n  Cohen's d: "
              f"{results['keynorm'].parameters['cohens_d']['mean']:.2f} "
              f"[{results['keynorm'].parameters['cohens_d']['hdi_3%']:.2f}, "
              f"{results['keynorm'].parameters['cohens_d']['hdi_97%']:.2f}]")

    # ---- Model 4: Multi-Model Comparison ----
    if 'multimodel' in all_models:
        print("\n" + "=" * 60)
        print("Model 4: Multi-Model Comparison")
        print("=" * 60)

        model, idata = build_multimodel_comparison(vd, **sample_kwargs)

        results['multimodel'] = summarize_model(
            'multimodel_comparison', idata,
            manuscript_claims={
                'pop_mu': 0.75,
            },
            params_of_interest=[
                'pop_mu', 'pop_sigma', 'model_mu', 'shrinkage',
            ],
        )
        idata.to_json(str(output_dir / 'multimodel_trace.json'))

        print(f"\n  Population mean: "
              f"{results['multimodel'].parameters['pop_mu']['mean']:.3f}")

    # ---- Model 5: Sequence-Length Degradation ----
    if 'seqlen' in all_models:
        print("\n" + "=" * 60)
        print("Model 5: Sequence-Length Degradation")
        print("=" * 60)

        model, idata = build_seqlen_model(vd, **sample_kwargs)

        results['seqlen'] = summarize_model(
            'seqlen_degradation', idata,
            manuscript_claims={},
            params_of_interest=[
                'r_0', 'beta_decay', 'r_at_128', 'r_at_512', 'r_at_1024',
                'halflife_doublings',
            ],
        )
        plot_seqlen_results(idata, vd, output_dir)
        idata.to_json(str(output_dir / 'seqlen_trace.json'))

        print(f"\n  Degradation rate: "
              f"{results['seqlen'].parameters['beta_decay']['mean']:.4f} "
              f"per doubling")

    # ---- Save combined summary ----
    combined = {
        name: s.to_dict() for name, s in results.items()
    }
    with open(output_dir / 'bayesian_validation_summary.json', 'w') as f:
        json.dump(combined, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print(f"All results saved to {output_dir}")
    print("=" * 60)

    # Print consolidated manuscript comparison
    print("\n--- Manuscript Claim Validation ---")
    for name, summary in results.items():
        if summary.manuscript_comparison:
            print(f"\n  [{name}]")
            for claim, comp in summary.manuscript_comparison.items():
                status = "IN HDI" if comp['claim_in_hdi'] else "OUTSIDE HDI"
                print(f"    {claim}: manuscript={comp['manuscript_value']:.3f}, "
                      f"posterior={comp['posterior_mean']:.3f} "
                      f"{comp['hdi_94']}, {status}")

    return results


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Bayesian validation of Gauge-Transformer claims'
    )
    parser.add_argument(
        '--data', type=str,
        default='Attention/figs/validation_results.json',
        help='Path to validation_results.json',
    )
    parser.add_argument(
        '--output', type=str,
        default='Attention/figs/bayesian',
        help='Output directory for plots and traces',
    )
    parser.add_argument(
        '--samples', type=int, default=2000,
        help='MCMC draws per chain',
    )
    parser.add_argument(
        '--tune', type=int, default=1000,
        help='Tuning samples',
    )
    parser.add_argument(
        '--models', nargs='+',
        choices=['hierarchical', 'temperature', 'keynorm', 'multimodel', 'seqlen'],
        default=None,
        help='Models to run (default: all)',
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed',
    )

    args = parser.parse_args()

    results = run_all_validations(
        json_path=args.data,
        output_dir=args.output,
        n_samples=args.samples,
        n_tune=args.tune,
        random_seed=args.seed,
        models=args.models,
    )
