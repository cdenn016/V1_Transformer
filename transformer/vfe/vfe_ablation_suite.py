# -*- coding: utf-8 -*-
"""
VFE Ablation Suite (transformer/vfe/)
=====================================

One-at-a-time hyperparameter sweeps over the clean ``transformer/vfe/``
training stack. Mirrors ``scripts/run_ablation_suite.py`` in structure
(``BASELINE_CONFIG`` + ``SWEEPS`` registry + per-sweep ``run_sweep`` +
``generate_plots`` + ``analyze_all``) but targets ``VFEConfig`` /
``VFEModel`` / ``VFETrainer`` directly — no detour through
``transformer.training.experiment_runner``.

Click-to-run: edit ``CONFIG`` near the bottom, then press Run. No CLI
arguments (per CLAUDE.md).

Sweep coverage (per user request, 2026-05-17):
    e_mu_lr, e_sigma_lr, e_phi_lr   (E-step learning rates)
    lambda_align, lambda_soft       (E-step alignment / softmax couplings)
    kappa                           (attention temperature)
    phi_trace_clamp                 (GL(K) determinant control)
    rope_base                       (RoPE frequency base)
    mass_phi                        (gauge-frame L2 prior)
    mu_init_std, phi_scale, sigma_init  (embedding init)
    m_mu_lr, m_sigma_lr, m_phi_lr, m_hyper_lr, m_other_lr, weight_decay  (M-step optimizer)

Each sweep varies ONE field. Output layout::

    {output_dir}/
      {sweep_name}/
        {sanitized_label}/                  # full VFETrainer outputs
          metrics.csv
          best_model.pt
          checkpoints/
          attention/
          figures/
          vfe_dynamics_figures/
          ablation_result.json              # summary used by this suite
          run_result.csv
        sweep_results.csv
        sweep_meta.json
      figures/
        {sweep_name}.png
        sensitivity_summary.png

Primary metric is ``final_ppl`` (best validation perplexity) — keeps the
plot/analysis code identical to the core-package suite.
"""

import copy
import gc
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

# Add project root to sys.path so the absolute imports below resolve when
# the file is run as a script from inside transformer/vfe/.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from transformer.vfe.config import VFEConfig
from transformer.vfe.model import VFEModel
from transformer.vfe.trainer import VFETrainer
from transformer.data.datasets import create_dataloaders


# =============================================================================
# BASELINE CONFIG (mirrors transformer/vfe/train_vfe.py defaults, 2026-05-17)
# =============================================================================
# Every field that VFEConfig accepts. The ablation driver builds a
# VFEConfig per run via VFEConfig(**{k: v for k, v in cfg.items()
# if k in VFE_CONFIG_FIELDS}), so additional bookkeeping keys (e.g. 'dataset')
# can live alongside the VFEConfig kwargs without raising TypeError.

BASELINE_CONFIG: Dict[str, Any] = {
    # === Structure ===
    'vocab_size':               None,      # populated after dataloader build
    'embed_dim':                20,
    'irrep_spec':               [('fund', 2, 10)],
    
    'batch_size':               128,
    
    'max_seq_len':              32,
    'max_steps':                2000,

    'use_prior_bank':           True,
    'mask_self_attention':      False,
    # Match train_vfe.py: per-token (μ_v, σ_v) lookup priors. Without this
    # key the dataclass default (True = shared-base gauge-orbit) silently
    # selects a structurally different prior than the live entry point uses.
    'gauge_fixed_priors':       False,
    'E_learnable_alpha':        True,
    'learnable_kappa':          False,

    'use_autograd_mu_sigma':       False,
    'use_equivariant_head_mixer':  False,
    'gauge_covariant_ridge':       False,

    # === E-step dynamics ===
    'n_e_steps':                1,
    'n_layers':                 1,

    'alpha_divergence':         1.0,
    'include_attention_entropy': True,

    'e_mu_lr':                  0.4,
    'e_sigma_lr':               0.015,
    'e_phi_lr':                 0.05,
   
    'alpha':                    1.0,
    'lambda_align':             4,
    'lambda_soft':              0.0,
    'mass_phi':                 0.0,

    'kappa':                    1.0,



    # === Cross-layer prior handoff ===
    'prior_handoff_rho':        1.0,
    'prior_handoff_sigma':      0.0,

    # === Covariance ===
    'diagonal_covariance':      True,
    'isotropic_covariance':     False,
    'exact_diagonal_transport': False,
    'enforce_orthogonal':       False,
    
    
    # === Gauge geometry ===
    'gauge_group':              'GLK',
    
    'phi_project_slk':          False,
    'phi_trace_clamp':          0.75,
    
    'phi_preconditioner':       'killing',  # 'clip', 'cartan', 'killing', 'pullback'

    # === Positional encoding ===
    'use_rope':                 True,
    'rope_full_gauge':          'off',
    'rope_base':                150,

    # === Embedding init ===
    'mu_init_std':              0.001,
    'phi_scale':                0.001,
    'sigma_init':               0.4,

    # === Active inference ===
    'active_inference':         False,
    'pragmatic_weight':         1.0,
    'epistemic_weight':         0.5,
    'epistemic_samples':        4,
    'decode_tau':               1.0,

    'use_non_flat_transport':       False,
    'non_flat_max_strength':        1.0,  # s_max in s = s_max·tanh(ρ)
    'non_flat_per_edge_delta_max':  1.0,  # δ_max bound on ‖δ_ij·G‖_F

    # === Normalization ===
    'norm_type':                'layernorm',
    'normalize_ce_by_dim':      True,

    # === Training ===
    # Per-group M-step LRs (see vfe/config.py for what each touches).
    'm_mu_lr':                  0.2,
    'm_sigma_lr':               0.015,
    'm_phi_lr':                 0.025,
    'm_hyper_lr':               0.001,
    'm_other_lr':               0.05,
    'weight_decay':             0.01,
    
    'warmup_steps':             100,
    'grad_clip':                50.0,
    'sigma_max':                12.0,
    
    'bch_order':                3,

    # === Logging / evaluation ===
    'log_interval':             200,
    'eval_interval':            2000,
    'checkpoint_interval':      25000,

    'track_layer_diagnostics':      False,
    'monitor_monotonicity':         False,

    # === Driver-only (not VFEConfig fields) ===
    'dataset':                  'wikitext-103',
}


# =============================================================================
# SWEEP DEFINITIONS
# =============================================================================
# Same schema as scripts/run_ablation_suite.py:
#   'values': [v1, v2, ...]            explicit list
#   'range':  [start, stop, step]      inclusive arithmetic range
#   'configs': [{...,'label': str}]    multi-param categorical

SWEEPS: Dict[str, Dict[str, Any]] = {

    # --- E-step learning rates -----------------------------------------------
    'alpha_divergence': {
        'description': 'alpha_divergence',
        'param': 'alpha_divergence',
        'values': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
        'baseline_value': 0.1,
    },
    
    
    'e_mu_lr': {
        'description': 'E-step natural-gradient step size for mu_q',
        'param': 'e_mu_lr',
        'values': [0.01, 0.1, 0.2, 0.3, 0.4, 0.5],
        'baseline_value': 0.1,
    },

    'e_sigma_lr': {
        'description': 'E-step retraction step size for sigma_q (decoupled from mu LR)',
        'param': 'e_sigma_lr',
        'values': [0.0, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1],
        'baseline_value': 0.015,
        # Lift retracted sigma_q from autograd-orphan status: with the
        # BASELINE_CONFIG defaults (n_e_steps=1, n_layers=1,
        # use_prior_bank=False, norm_type='layernorm'), the retracted sigma
        # has no downstream consumer (see VFEConfig.__post_init__ orphan
        # warning). Bumping n_e_steps to 2 gives iteration #2's
        # nat_grad_mu a chance to read the updated sigma, making this LR
        # observable in val PPL.
        'requires': {'n_e_steps': 2},
    },

    'e_phi_lr': {
        'description': 'E-step gauge-frame step size for phi (Killing-preconditioned)',
        'param': 'e_phi_lr',
        'values': [0.0, 0.001, 0.005, 0.01, 0.05, 0.1],
        'baseline_value': 0.05,
        # Same orphan-lift as e_sigma_lr: phi only matters across
        # iterations or layers (no decoder/norm reads phi directly).
        # n_e_steps=2 gives iteration #2's attention/transport a chance
        # to read the updated phi.
        'requires': {'n_e_steps': 2},
    },

    # --- E-step coupling weights ---------------------------------------------
    'lambda_align': {
        'description': 'Boltzmann GLU weight in F: beta_ij * grad_theta KL(q_i || Omega_ij q_j)',
        'param': 'lambda_align',
        'values': [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
        'baseline_value': 1.0,
    },

    'lambda_softmax': {
        'description': 'Attention-variance coupling in F: KL * grad_theta beta_ij (vfe field: lambda_soft)',
        'param': 'lambda_soft',
        'values': [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
        'baseline_value': 0.0,
        # lambda_soft only enters the φ-update through the entropy-suppressed
        # surrogate path (e_step.py: legacy product-rule decomposition).
        # When include_attention_entropy=True (the BASELINE default), every
        # gradient gate forces lambda_softmax=0.0 — so all sweep values
        # collapse to bitwise-identical loss. Disable the entropy term for
        # this sweep so the surrogate path is exercised and lambda_soft is
        # observable in val PPL.
        'requires': {'include_attention_entropy': False},
    },

    # --- Attention temperature -----------------------------------------------
    'kappa': {
        'description': 'Attention temperature: beta = softmax(-KL / (kappa * sqrt(K)))',
        'param': 'kappa',
        'values': [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0, 8.0],
        'baseline_value': 1.0,
    },

    # --- GL(K) determinant control -------------------------------------------
    'phi_trace_clamp': {
        'description': 'Soft cap |tr(phi.G)| <= T => det(Omega_ij) in [exp(-2T), exp(2T)]',
        'param': 'phi_trace_clamp',
        'values': [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
        'baseline_value': 0.75,
    },

    # --- RoPE base frequency -------------------------------------------------
    'rope_base': {
        'description': 'RoPE frequency base: theta_n = 1 / base^(2n/K)',
        'param': 'rope_base',
        'values': [25, 50, 75, 100, 150, 200],
        'baseline_value': 100,
    },

    # --- Gauge frame L2 prior ------------------------------------------------
    'mass_phi': {
        'description': 'Gauge prior weight: (mass_phi / 2) * mean(||phi||^2)',
        'param': 'mass_phi',
        'values': [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        'baseline_value': 0.0,
    },

    # --- Embedding initialization (three knobs) ------------------------------
    'mu_init_std': {
        'description': 'Embedding init: std of base_mu = randn(K) * mu_init_std',
        'param': 'mu_init_std',
        'values': [0, 1e-3, 1e-2, 1e-1, 1],
        'baseline_value': 0.4,
    },

    'phi_scale': {
        'description': 'Embedding init: per-token phi_embed ~ N(0, phi_scale)',
        'param': 'phi_scale',
        'values': [0.0, 1e-3, 1e-2, 0.05, 0.1, 0.25, 0.5, 1.0],
        'baseline_value': 0.05,
    },

    'sigma_init': {
        'description': 'Embedding init: initial covariance scale (base_log_sigma = log(sigma_init))',
        'param': 'sigma_init',
        'values': [0.1, 0.25, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0],
        'baseline_value': 0.4,
    },

    # --- M-step optimizer ----------------------------------------------------
    'm_mu_lr': {
        'description': 'M-step LR for PriorBank.base_mu',
        'param': 'm_mu_lr',
        'values': [0.15, 0.25, 0.3, 0.4, 0.5],
        'baseline_value': 0.2,
    },

    'm_sigma_lr': {
        'description': 'M-step LR for PriorBank.base_log_sigma and decode_log_scale',
        'param': 'm_sigma_lr',
        'values': [0.015, 0.025, 0.035, 0.045, 0.055, 0.075],
        'baseline_value': 0.05,
    },

    'm_phi_lr': {
        'description': 'M-step LR for PriorBank.phi_embed and Positional.pos_phi',
        'param': 'm_phi_lr',
        'values': [0.005, 0.015, 0.025, 0.035, 0.045],
        'baseline_value': 0.01,
    },

    'm_hyper_lr': {
        'description': 'M-step LR for E-step learnable hyperparams (raw_c0, raw_b0, log_kappa, _phi_preconditioner)',
        'param': 'm_hyper_lr',
        'values': [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2],
        'baseline_value': 0.001,
    },

    'm_other_lr': {
        'description': 'M-step LR for catch-all group (head_mixer, non_flat W_raw, output projection, ...)',
        'param': 'm_other_lr',
        'values': [0.025, 0.035, 0.045, 0.055, 0.065, 0.075],
        'baseline_value': 0.001,
    },

    'weight_decay': {
        'description': 'M-step AdamW weight decay',
        'param': 'weight_decay',
        'values': [0.0, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1],
        'baseline_value': 0.001,
    },
}


# Sweep execution order (cheap-to-expensive; keep in sync with user priorities).
SWEEP_ORDER: List[str] = [
  #  'e_mu_lr',
   # 'e_sigma_lr',
   # 'e_phi_lr',
   'alpha_divergence',
   # 'm_mu_lr',
   # 'm_sigma_lr',
    #'m_phi_lr',
   # 'm_hyper_lr',
   # 'm_other_lr',

   # 'weight_decay',
    
  #  'lambda_align',
   # 'lambda_softmax',
   
   # 'rope_base',
  #  'mass_phi',
    
   # 'mu_init_std',
   # 'phi_scale',
   # 'sigma_init',
    # 'kappa',
   # 'phi_trace_clamp',
]


# =============================================================================
# CONFIG FIELD WHITELIST (built once from VFEConfig dataclass fields)
# =============================================================================

from dataclasses import fields as _dc_fields
_VFE_FIELDS = {f.name for f in _dc_fields(VFEConfig)}


def _strip_to_vfe_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Drop driver-only keys (e.g. 'dataset') before constructing VFEConfig."""
    return {k: v for k, v in cfg.items() if k in _VFE_FIELDS}


# =============================================================================
# RANGE / VALUE EXPANSION (verbatim from scripts/run_ablation_suite.py)
# =============================================================================

def _expand_range(
    range_spec: "Union[Dict[str, Union[int, float]], List[Union[int, float]]]",
) -> "List[Union[int, float]]":
    """Expand a ``'range': [start, stop, step]`` spec into an explicit list."""
    if isinstance(range_spec, dict):
        start = range_spec['start']
        stop = range_spec['stop']
        step = range_spec['step']
    else:
        if len(range_spec) != 3:
            raise ValueError(
                f"'range' spec must be [start, stop, step], got {range_spec!r}"
            )
        start, stop, step = range_spec
    if step == 0:
        raise ValueError("'range' step must be non-zero")

    all_int = all(isinstance(v, int) and not isinstance(v, bool)
                  for v in (start, stop, step))
    values = []
    n_steps = int(round((stop - start) / step))
    tol = abs(step) * 1e-9
    for i in range(n_steps + 2):
        v = start + i * step
        if step > 0 and v > stop + tol:
            break
        if step < 0 and v < stop - tol:
            break
        values.append(v if all_int else round(v, 10))
    return values


def _sweep_values(sweep: Dict[str, Any]) -> "List[Union[int, float]]":
    if 'configs' in sweep:
        return []
    if 'values' in sweep:
        return list(sweep['values'])
    if 'range' in sweep:
        return _expand_range(sweep['range'])
    raise KeyError(
        f"Single-param sweep must define 'values' or 'range' (sweep: {sweep!r})"
    )


def _sweep_num_runs(sweep: Dict[str, Any]) -> int:
    if 'configs' in sweep:
        return len(sweep['configs'])
    return len(_sweep_values(sweep))


def make_run_configs(sweep_name: str, base_config: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    """Generate (label, config) pairs for a sweep.

    Honors an optional per-sweep ``'requires'`` dict: prerequisite overrides
    merged into the base config BEFORE the swept param is set. Used by
    sweeps whose target field is autograd-orphaned under BASELINE_CONFIG
    (see e_sigma_lr / e_phi_lr — both require n_e_steps>=2 to lift the
    retracted tensors out of "computed-then-discarded" status).
    """
    sweep = SWEEPS[sweep_name]
    requires = sweep.get('requires', {})
    runs: List[Tuple[str, Dict[str, Any]]] = []

    if 'configs' in sweep:
        for entry in sweep['configs']:
            cfg = copy.deepcopy(base_config)
            cfg.update(requires)  # prerequisites first
            label = entry.pop('label')
            cfg.update(entry)
            entry['label'] = label  # restore for next call
            runs.append((label, cfg))
    else:
        param = sweep['param']
        for val in _sweep_values(sweep):
            cfg = copy.deepcopy(base_config)
            cfg.update(requires)  # prerequisites first
            cfg[param] = val
            label = f"{param}={val}"
            runs.append((label, cfg))

    return runs


def _cleanup_after_experiment() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _set_all_seeds(seed: int) -> None:
    """Local equivalent of transformer.training.utils.set_all_seeds (kept local
    so this script has no dependency on the core training package)."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# SINGLE-RUN EXECUTOR
# =============================================================================

def run_single_vfe_experiment(
    cfg: Dict[str, Any],
    device: torch.device,
    run_dir: Path,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    vocab_size: int,
    seed: int,
) -> Dict[str, Any]:
    """Build a fresh VFEModel + VFETrainer from `cfg` and train it.

    Returns a result dict with `final_ppl` (= best val PPL) and per-run
    diagnostics extracted from the trainer's PublicationMetricsTracker.
    """
    _set_all_seeds(seed)

    # VFEConfig expects vocab_size to be a concrete int.
    cfg = copy.deepcopy(cfg)
    cfg['vocab_size'] = vocab_size

    vfe_kwargs = _strip_to_vfe_kwargs(cfg)
    vcfg = VFEConfig(**vfe_kwargs)
    model = VFEModel(vcfg)
    n_params = sum(p.numel() for p in model.parameters())

    run_dir.mkdir(parents=True, exist_ok=True)

    trainer = VFETrainer(
        model, vcfg, train_loader,
        val_loader=val_loader,
        device=str(device),
        output_dir=str(run_dir),
        # Per-run training_curves / gradient_norms / vfe_dynamics figures
        # (and periodic attention β heatmaps) are skipped: the ablation
        # suite produces its own sweep-level plots in generate_plots().
        # Suppressing them also avoids the "More than 20 figures open"
        # RuntimeWarning that accumulated across runs (~4 leaked figures
        # per call to trainer._generate_figures()).
        generate_figures=False,
    )
    trainer.train(num_steps=vcfg.max_steps)

    # Pull summary from PublicationMetricsTracker (filled by trainer.train).
    summary: Dict[str, Any] = {}
    if trainer._pub_tracker is not None:
        try:
            summary = trainer._pub_tracker.get_summary() or {}
        except (AttributeError, KeyError, ValueError) as exc:
            # Tracker can return empty/partial state if a run aborted before
            # the first eval. Log loudly so a genuine bug doesn't get silently
            # masked by an empty summary downstream.
            print(f"  WARNING: _pub_tracker.get_summary() failed: {exc!r}")
            summary = {}

    best_val_ppl = summary.get('best_val_ppl')
    final_val_ppl = summary.get('final_val_ppl')
    final_train_ppl = summary.get('final_train_ppl')
    final_val_bpc = summary.get('final_val_bpc')
    best_val_bpc = summary.get('best_val_bpc')
    avg_tokens_per_sec = summary.get('avg_tokens_per_sec')

    # `final_ppl` is the primary plotting metric; mirrors the core suite.
    # Prefer best-val (the historical convention in run_ablation_suite.py),
    # fall back to final-val, then to final-train if validation never ran.
    primary = best_val_ppl if best_val_ppl is not None else final_val_ppl
    if primary is None:
        primary = final_train_ppl
    if primary is None:
        primary = float('inf')

    metrics_csv = run_dir / 'metrics.csv'

    return {
        'final_ppl':           float(primary),
        'best_val_ppl':        float(best_val_ppl) if best_val_ppl is not None else None,
        'final_val_ppl':       float(final_val_ppl) if final_val_ppl is not None else None,
        'final_train_ppl':     float(final_train_ppl) if final_train_ppl is not None else None,
        'best_val_bpc':        float(best_val_bpc) if best_val_bpc is not None else None,
        'final_val_bpc':       float(final_val_bpc) if final_val_bpc is not None else None,
        'avg_tokens_per_sec':  float(avg_tokens_per_sec) if avg_tokens_per_sec is not None else None,
        'total_params':        int(n_params),
        'metrics_csv':         str(metrics_csv) if metrics_csv.exists() else None,
    }


# =============================================================================
# SWEEP DRIVER
# =============================================================================

def _sanitize_label(label: str) -> str:
    """Make `label` safe as a single filesystem path component.

    Strips path separators (``/``, ``\\``), parent-directory tokens (``..``),
    drive prefixes, and whitespace so a malformed sweep label cannot escape
    its sweep directory via Path concatenation.
    """
    out = (
        label.replace('=', '_')
             .replace(' ', '_')
             .replace('/', '_')
             .replace('\\', '_')
             .replace('..', '_')
             .replace(':', '_')
    )
    # Strip leading separators so Path() does not treat the label as absolute.
    return out.lstrip('._') or '_'


def run_sweep(
    sweep_name: str,
    base_config: Dict[str, Any],
    device: torch.device,
    output_dir: Path,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    vocab_size: int,
    seed: int = 6,
    max_steps_override: Optional[int] = None,
    resume: bool = False,
) -> pd.DataFrame:
    """Run all configs in a sweep, return results DataFrame."""
    sweep = SWEEPS[sweep_name]
    sweep_dir = output_dir / sweep_name
    sweep_dir.mkdir(parents=True, exist_ok=True)

    runs = make_run_configs(sweep_name, base_config)

    print(f"\n{'=' * 70}")
    print(f"SWEEP: {sweep_name} ({len(runs)} configs)")
    print(f"  {sweep['description']}")
    print(f"  Baseline: {sweep['baseline_value']}")
    print(f"  Output:   {sweep_dir}")
    if resume:
        print(f"  Resume:   ON (skipping completed runs)")
    print(f"{'=' * 70}\n")

    results: List[Dict[str, Any]] = []

    # Preload any previously completed results (so post-sweep CSV is whole).
    if resume:
        for label, _cfg in runs:
            run_dir = sweep_dir / _sanitize_label(label)
            result_file = run_dir / 'ablation_result.json'
            if result_file.exists():
                with open(result_file, encoding='utf-8') as f:
                    results.append(json.load(f))

    for i, (label, cfg) in enumerate(runs):
        run_dir = sweep_dir / _sanitize_label(label)

        if resume and (run_dir / 'ablation_result.json').exists():
            print(f"\n--- Run {i + 1}/{len(runs)}: {label} [CACHED] ---")
            continue

        print(f"\n--- Run {i + 1}/{len(runs)}: {label} ---")

        if max_steps_override is not None:
            cfg['max_steps'] = int(max_steps_override)

        try:
            t0 = time.time()
            result = run_single_vfe_experiment(
                cfg=cfg,
                device=device,
                run_dir=run_dir,
                train_loader=train_loader,
                val_loader=val_loader,
                vocab_size=vocab_size,
                seed=seed,
            )
            elapsed = time.time() - t0

            result['sweep'] = sweep_name
            result['label'] = label
            result['elapsed_sec'] = elapsed
            result['seed'] = seed
            results.append(result)

            with open(run_dir / 'ablation_result.json', 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)

            run_df = pd.DataFrame([result])
            run_df.to_csv(run_dir / 'run_result.csv', index=False)

            sweep_csv = sweep_dir / 'sweep_results.csv'
            run_df.to_csv(
                sweep_csv,
                mode='a',
                header=not sweep_csv.exists(),
                index=False,
            )

            _completed = i + 1
            _remaining = len(runs) - _completed
            if _completed == 1 and _remaining > 0:
                _est_total = elapsed * (_remaining + 1)
                _est_remain = elapsed * _remaining
                print(f"  -> Val PPL: {result['final_ppl']:.2f},  Time: {elapsed:.0f}s")
                print(f"  ** Estimated sweep total: {_est_total / 60:.0f} min "
                      f"({_remaining} remaining x {elapsed:.0f}s = ~{_est_remain / 60:.0f} min left)")
            else:
                print(f"  -> Val PPL: {result['final_ppl']:.2f},  Time: {elapsed:.0f}s")

        except Exception as e:  # noqa: BLE001 - sweep driver must survive per-run failures
            logger.exception("sweep %s/%s failed", sweep_name, label)
            results.append({
                'sweep': sweep_name,
                'label': label,
                'error': str(e),
                'final_ppl': float('inf'),
            })
        finally:
            _cleanup_after_experiment()

    # Final sweep CSV — overwrites the appended one with the complete frame.
    df = pd.DataFrame(results)
    df.to_csv(sweep_dir / 'sweep_results.csv', index=False)

    meta = {
        'sweep_name':    sweep_name,
        'description':   sweep['description'],
        'baseline_value': str(sweep['baseline_value']),
        'n_runs':        len(runs),
        'seed':          seed,
        'timestamp':     time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(sweep_dir / 'sweep_meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"SWEEP COMPLETE: {sweep_name}")
    print(f"{'=' * 70}")
    if not df.empty and 'final_ppl' in df.columns:
        valid = df[df['final_ppl'] < float('inf')]
        if not valid.empty:
            best = valid.loc[valid['final_ppl'].idxmin()]
            print(f"  Best: {best.get('label', '?')} -> Val PPL {best['final_ppl']:.2f}")
    print()

    return df


# =============================================================================
# ANALYSIS
# =============================================================================

def analyze_sweep(sweep_dir: Path) -> Dict[str, Any]:
    csv_path = sweep_dir / 'sweep_results.csv'
    if not csv_path.exists():
        print(f"No results found in {sweep_dir}")
        return {}

    df = pd.read_csv(csv_path)
    meta_path = sweep_dir / 'sweep_meta.json'
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
    else:
        meta = {}

    print(f"\n{'=' * 70}")
    print(f"ANALYSIS: {meta.get('sweep_name', sweep_dir.name)}")
    print(f"  {meta.get('description', '')}")
    print(f"{'=' * 70}\n")

    valid = df[df['final_ppl'] < float('inf')].copy()
    valid = valid.sort_values('final_ppl')

    print(f"{'Label':<32} {'Val PPL':>10} {'Params':>12} {'Time':>8}")
    print('-' * 65)
    for _, row in valid.iterrows():
        params = f"{int(row.get('total_params', 0)):,}" if pd.notna(row.get('total_params')) else 'N/A'
        elapsed = f"{row.get('elapsed_sec', 0):.0f}s" if pd.notna(row.get('elapsed_sec')) else 'N/A'
        print(f"{row['label']:<32} {row['final_ppl']:>10.2f} {params:>12} {elapsed:>8}")

    if len(valid) > 1:
        baseline_ppl = valid.iloc[0]['final_ppl']
        print(f"\nRelative to best ({valid.iloc[0]['label']}):")
        for _, row in valid.iterrows():
            delta = ((row['final_ppl'] - baseline_ppl) / baseline_ppl) * 100
            print(f"  {row['label']:<32} {delta:+.1f}%")

    return {'df': df, 'meta': meta}


def analyze_all(output_dir: Path) -> None:
    print(f"\n{'=' * 70}")
    print(f"VFE ABLATION SUITE SUMMARY")
    print(f"  Directory: {output_dir}")
    print(f"{'=' * 70}\n")

    all_results = []
    for sweep_dir in sorted(output_dir.iterdir()):
        if sweep_dir.is_dir() and (sweep_dir / 'sweep_results.csv').exists():
            result = analyze_sweep(sweep_dir)
            if result:
                all_results.append(result)

    if not all_results:
        print("No completed sweeps found.")
        return

    print(f"\n{'=' * 70}")
    print(f"CROSS-SWEEP SUMMARY (best per sweep)")
    print(f"{'=' * 70}\n")
    print(f"{'Sweep':<25} {'Best Config':<30} {'Val PPL':>10}")
    print('-' * 70)
    for r in all_results:
        df = r['df']
        valid = df[df['final_ppl'] < float('inf')]
        if valid.empty:
            continue
        best = valid.loc[valid['final_ppl'].idxmin()]
        sweep = r['meta'].get('sweep_name', '?')
        print(f"{sweep:<25} {best['label']:<30} {best['final_ppl']:>10.2f}")


# =============================================================================
# PLOTS
# =============================================================================

def generate_plots(output_dir: Path) -> None:
    """Per-sweep PPL line/bar plots + cross-sweep sensitivity summary."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib/seaborn not available, skipping plots")
        return

    sns.set_theme(style='whitegrid', font_scale=1.2)

    sweep_dirs = [d for d in sorted(output_dir.iterdir())
                  if d.is_dir() and (d / 'sweep_results.csv').exists()]
    if not sweep_dirs:
        print("No sweeps to plot.")
        return

    fig_dir = output_dir / 'figures'
    fig_dir.mkdir(exist_ok=True)

    for sweep_dir in sweep_dirs:
        df = pd.read_csv(sweep_dir / 'sweep_results.csv')
        meta_path = sweep_dir / 'sweep_meta.json'
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
        else:
            meta = {}
        sweep_name = meta.get('sweep_name', sweep_dir.name)

        valid = df[df['final_ppl'] < float('inf')].copy()
        if valid.empty:
            continue

        ppl_col = 'final_ppl'
        ppl_label = 'Validation Perplexity'

        numeric_values = []
        for label in valid['label']:
            try:
                val = str(label).split('=')[-1]
                numeric_values.append(float(val))
            except (ValueError, IndexError):
                numeric_values.append(None)

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))

        if all(v is not None for v in numeric_values):
            valid['param_value'] = numeric_values
            valid = valid.sort_values('param_value')
            ax.plot(valid['param_value'], valid[ppl_col], 'o-', linewidth=2, markersize=8)
            ax.set_xlabel(sweep_name)

            baseline_val = meta.get('baseline_value')
            if baseline_val:
                try:
                    bv = float(baseline_val)
                    ax.axvline(bv, color='red', linestyle='--', alpha=0.5, label=f'baseline={bv}')
                    ax.legend()
                except (ValueError, TypeError):
                    pass
        else:
            valid = valid.sort_values(ppl_col)
            colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(valid))]
            ax.barh(range(len(valid)), valid[ppl_col], color=colors)
            ax.set_yticks(range(len(valid)))
            ax.set_yticklabels(valid['label'])
            ax.invert_yaxis()

        ax.set_ylabel(ppl_label)
        ax.set_title(f"VFE Ablation: {meta.get('description', sweep_name)}")
        fig.tight_layout()
        fig.savefig(fig_dir / f'{sweep_name}.png', dpi=150)
        plt.close(fig)

    # Cross-sweep sensitivity summary
    all_bests = []
    for sweep_dir in sweep_dirs:
        df = pd.read_csv(sweep_dir / 'sweep_results.csv')
        meta_path = sweep_dir / 'sweep_meta.json'
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
        else:
            meta = {}
        valid = df[df['final_ppl'] < float('inf')]
        if valid.empty:
            continue
        best = valid.loc[valid['final_ppl'].idxmin()]
        worst = valid.loc[valid['final_ppl'].idxmax()]
        all_bests.append({
            'sweep':     meta.get('sweep_name', sweep_dir.name),
            'best_ppl':  best['final_ppl'],
            'worst_ppl': worst['final_ppl'],
            'range':     worst['final_ppl'] - best['final_ppl'],
            'best_label': best['label'],
        })

    if all_bests:
        bdf = pd.DataFrame(all_bests).sort_values('range', ascending=False)
        fig, ax = plt.subplots(1, 1, figsize=(10, max(4, len(bdf) * 0.5)))
        y_pos = range(len(bdf))
        ax.barh(y_pos, bdf['range'], color='#e74c3c', alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{r['sweep']}\n(best: {r['best_label']})" for _, r in bdf.iterrows()])
        ax.set_xlabel('Val PPL Range (worst - best)')
        ax.set_title('VFE Hyperparameter Sensitivity (Val PPL range per sweep)')
        ax.invert_yaxis()
        fig.tight_layout()
        fig.savefig(fig_dir / 'sensitivity_summary.png', dpi=150)
        plt.close(fig)
        print(f"  Saved: {fig_dir / 'sensitivity_summary.png'}")

    print(f"\nAll figures saved to: {fig_dir}")


# =============================================================================
# DATASET CONSTRUCTION (one set of loaders per sweep, reused across runs)
# =============================================================================

def _build_loaders(
    base_config: Dict[str, Any],
) -> Tuple[DataLoader, Optional[DataLoader], int]:
    """Build (train_loader, val_loader, vocab_size) from base_config."""
    return create_dataloaders(
        max_seq_len=int(base_config['max_seq_len']),
        batch_size=int(base_config['batch_size']),
        vocab_size=base_config.get('vocab_size'),
        dataset=base_config.get('dataset', 'wikitext-103'),
    )


# =============================================================================
# MAIN  (click-to-run; edit CONFIG dict, press Run)
# =============================================================================

CONFIG: Dict[str, Any] = {
    # Action mode: 'train', 'analyze', 'plot', 'list'
    'mode':        'train',

    # Train-mode settings
    'sweep':       None,                       # name of one sweep, or None -> all in SWEEP_ORDER
    'device':      'auto',                     # 'auto', 'cuda', 'cpu'
    'dataset':     'wikitext-103',
    'output_dir':  'vfe_ablation_results',
    'max_steps':   None,                       # override BASELINE_CONFIG['max_steps']
    'seed':        6,
    'resume':      False,

    # Analyze / plot mode
    'results_dir': 'vfe_ablation_results',
}


def main() -> None:
    mode = CONFIG['mode']

    if mode == 'list':
        print(f"\nAvailable VFE sweeps ({len(SWEEPS)}):\n")
        print(f"{'Name':<25} {'# Configs':>10}  Description")
        print('-' * 80)
        for name in SWEEP_ORDER:
            s = SWEEPS[name]
            n = _sweep_num_runs(s)
            print(f"{name:<25} {n:>10}  {s['description']}")
        total = sum(_sweep_num_runs(SWEEPS[n]) for n in SWEEP_ORDER)
        print(f"\nTotal runs: {total}")
        return

    if mode == 'analyze':
        analyze_all(Path(CONFIG['results_dir']))
        return

    if mode == 'plot':
        generate_plots(Path(CONFIG['results_dir']))
        return

    if mode != 'train':
        raise ValueError(
            f"CONFIG['mode']={mode!r} not recognized; expected 'train', "
            "'analyze', 'plot', or 'list'."
        )

    # ---- Train mode --------------------------------------------------------
    if CONFIG['device'] == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(CONFIG['device'])

    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    base_config = copy.deepcopy(BASELINE_CONFIG)
    base_config['dataset'] = CONFIG['dataset']

    sweep_arg = CONFIG['sweep']
    if sweep_arg:
        if sweep_arg not in SWEEPS:
            print(f"Unknown sweep: {sweep_arg}")
            print(f"Available: {', '.join(SWEEP_ORDER)}")
            return
        sweep_names = [sweep_arg]
    else:
        sweep_names = SWEEP_ORDER

    print(f"\nVFE Ablation Suite")
    print(f"  Device:    {device}")
    print(f"  Dataset:   {CONFIG['dataset']}")
    print(f"  Output:    {output_dir}")
    print(f"  Seed:      {CONFIG['seed']}")
    print(f"  Sweeps:    {', '.join(sweep_names)}")
    if CONFIG['max_steps']:
        print(f"  Max steps: {CONFIG['max_steps']} (override)")
    if CONFIG['resume']:
        print(f"  Resume:    ON (skipping completed runs)")
    print()

    # Build dataloaders ONCE for the whole suite — all sweeps share dataset,
    # vocab_size, max_seq_len, batch_size, so the (multi-hundred-MB) token
    # cache and DataLoader workers don't need to spawn per run.
    print("Building shared dataloaders (one-time cost)...")
    train_loader, val_loader, vocab_size = _build_loaders(base_config)
    print(f"  vocab_size={vocab_size}  "
          f"train_batches={len(train_loader)}  val_batches={len(val_loader)}")
    base_config['vocab_size'] = vocab_size

    all_dfs: Dict[str, pd.DataFrame] = {}
    for sweep_name in sweep_names:
        df = run_sweep(
            sweep_name=sweep_name,
            base_config=base_config,
            device=device,
            output_dir=output_dir,
            train_loader=train_loader,
            val_loader=val_loader,
            vocab_size=vocab_size,
            seed=CONFIG['seed'],
            max_steps_override=CONFIG['max_steps'],
            resume=CONFIG['resume'],
        )
        all_dfs[sweep_name] = df

        # Regenerate plots after every sweep so partial results are visible.
        generate_plots(output_dir)

    analyze_all(output_dir)


if __name__ == '__main__':
    main()
