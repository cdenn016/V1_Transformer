"""
Flat Bundle Experiment Configurations.
=======================================

Training configs for each flat bundle hypothesis experiment.
Each config extends the base Gauge-Transformer config with non-flat transport settings.
"""

from typing import Dict, List, Optional


def get_base_experiment_config() -> dict:
    """Base config shared by all flat bundle experiments.

    Small-to-medium model suitable for hypothesis testing
    (not publication-scale, optimize later).
    """
    return {
        # Model
        'embed_dim': 64,
        'n_layers': 4,
        'hidden_dim': 256,
        'irrep_spec': [('ℓ0', 4, 1), ('ℓ1', 4, 3)],  # 4 scalar + 4 vector heads
        'kappa_beta': 1.0,
        'phi_dim': 3,
        'gauge_mode': 'learned',
        'diagonal_covariance': True,
        'evolve_sigma': True,
        'evolve_phi': True,
        'ffn_mode': 'VFE_dynamic',
        'ffn_n_iterations': 1,
        'ffn_alpha': 0.001,
        'ffn_learnable_lr': True,

        # Training
        'max_seq_len': 128,
        'batch_size': 32,
        'max_steps': 20000,
        'learning_rate': 3e-4,
        'weight_decay': 0.1,
        'grad_clip': 1.0,
        'warmup_steps': 500,

        # VFE loss
        'alpha': 0.1,
        'lambda_beta': 1.0,

        # Non-flat defaults (flat by default)
        'non_flat_transport': False,
        'cocycle_relaxation': 0.0,
        'per_head_flatness_gate': False,
        'connection_type': 'bilinear',
        'connection_hidden_dim': 64,
        'holonomy_penalty': 0.0,
    }


def get_small_config() -> dict:
    """Smaller config for synthetic language experiments (faster iteration)."""
    config = get_base_experiment_config()
    config.update({
        'embed_dim': 32,
        'n_layers': 3,
        'hidden_dim': 128,
        'irrep_spec': [('ℓ0', 4, 1), ('ℓ1', 2, 3)],
        'max_seq_len': 32,
        'batch_size': 64,
        'max_steps': 10000,
    })
    return config


# =============================================================================
# HF4.1: Cocycle Relaxation Ablation
# =============================================================================

def get_cocycle_relaxation_configs(
    alpha_values: Optional[List[float]] = None,
) -> Dict[str, dict]:
    """Sweep cocycle_relaxation from 0 (flat) to 1 (fully non-flat).

    Tests: COGS accuracy peaks at α≈0; sarcasm detection peaks at α>0;
    LM perplexity has intermediate optimum.
    """
    if alpha_values is None:
        alpha_values = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]

    configs = {}
    for alpha in alpha_values:
        name = f'cocycle_alpha_{alpha:.2f}'
        config = get_base_experiment_config()
        config.update({
            'non_flat_transport': alpha > 0,
            'cocycle_relaxation': alpha,
            'connection_type': 'bilinear',
        })
        configs[name] = config
    return configs


# =============================================================================
# HF4.2: Per-Head Flatness Gating
# =============================================================================

def get_per_head_gating_config() -> dict:
    """Per-head learnable flatness gates.

    Each head gets a gate g_h ∈ [0,1] controlling its non-flatness.
    After training, flat heads (g_h≈0) should handle compositional structure;
    non-flat heads (g_h≈1) should handle pragmatic/discourse structure.
    """
    config = get_base_experiment_config()
    config.update({
        'non_flat_transport': True,
        'cocycle_relaxation': 1.0,  # Gates control per-head
        'per_head_flatness_gate': True,
        'connection_type': 'bilinear',
    })
    return config


# =============================================================================
# HF2.3: Holonomy Penalty Scaling
# =============================================================================

def get_holonomy_penalty_configs(
    lambda_values: Optional[List[float]] = None,
) -> Dict[str, dict]:
    """Sweep holonomy penalty λ_H for non-flat model.

    Adding λ_H · E[‖H_ijk - I‖²] to the loss pushes toward flatness.
    Tests whether flatness is beneficial (optimal λ* > 0).
    """
    if lambda_values is None:
        lambda_values = [0.0, 0.001, 0.01, 0.1, 1.0, 10.0]

    configs = {}
    for lam in lambda_values:
        name = f'holonomy_penalty_{lam:.3f}'
        config = get_base_experiment_config()
        config.update({
            'non_flat_transport': True,
            'cocycle_relaxation': 1.0,
            'holonomy_penalty': lam,
        })
        configs[name] = config
    return configs


# =============================================================================
# HF5.3: Synthetic Language with Controlled Holonomy
# =============================================================================

def get_synthetic_language_configs(
    epsilon_values: Optional[List[float]] = None,
) -> Dict[str, dict]:
    """Sweep holonomy strength ε for synthetic gauge language.

    For each ε, trains both flat and non-flat models.
    Expected: crossover at ε* where non-flat surpasses flat.
    """
    if epsilon_values is None:
        epsilon_values = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0]

    configs = {}
    for eps in epsilon_values:
        for model_type in ['flat', 'non_flat']:
            name = f'synthetic_eps{eps:.2f}_{model_type}'
            config = get_small_config()
            config.update({
                'dataset': 'synthetic_gauge',
                'synthetic_epsilon': eps,
                'synthetic_vocab_size': 64,
                'synthetic_K': 3,
                'synthetic_n_classes': 4,
                'synthetic_seq_len': 16,
                'non_flat_transport': model_type == 'non_flat',
                'cocycle_relaxation': 1.0 if model_type == 'non_flat' else 0.0,
            })
            configs[name] = config
    return configs


# =============================================================================
# HF5.4: Developmental Trajectory (Flatness Emerges During Training)
# =============================================================================

def get_developmental_config() -> dict:
    """Config for tracking holonomy evolution during training.

    Trains a non-flat model on natural language, tracking whether
    holonomy decreases as the model learns (discovering flatness).
    """
    config = get_base_experiment_config()
    config.update({
        'non_flat_transport': True,
        'cocycle_relaxation': 1.0,
        'max_steps': 100000,
        'dataset': 'wikitext-103',
        # Log holonomy at frequent checkpoints
        'holonomy_log_interval': 500,
        'checkpoint_steps': [1000, 5000, 10000, 25000, 50000, 100000],
    })
    return config


# =============================================================================
# HF1.1 / HF2.2: Compositionality Benchmarks
# =============================================================================

def get_compositionality_configs() -> Dict[str, dict]:
    """Configs for COGS/SCAN evaluation.

    Trains flat Gauge-Transformer and standard transformer on compositional
    generalization benchmarks.
    """
    configs = {}

    # Flat Gauge-Transformer
    flat_config = get_base_experiment_config()
    flat_config.update({'dataset': 'cogs', 'max_seq_len': 256})
    configs['flat_gauge_cogs'] = flat_config

    # Non-flat Gauge-Transformer
    non_flat_config = get_base_experiment_config()
    non_flat_config.update({
        'dataset': 'cogs',
        'max_seq_len': 256,
        'non_flat_transport': True,
        'cocycle_relaxation': 1.0,
    })
    configs['non_flat_gauge_cogs'] = non_flat_config

    # Trivial gauge (closest to standard transformer)
    trivial_config = get_base_experiment_config()
    trivial_config.update({
        'dataset': 'cogs',
        'max_seq_len': 256,
        'gauge_mode': 'trivial',
    })
    configs['trivial_gauge_cogs'] = trivial_config

    return configs


# =============================================================================
# HF1.2: Non-Flat on Pragmatic Tasks
# =============================================================================

def get_pragmatic_configs() -> Dict[str, dict]:
    """Configs for sarcasm detection and word sense disambiguation.

    Tests whether non-flat model learns holonomy on context-dependent inputs.
    """
    configs = {}
    for dataset in ['sarcasm', 'wic']:
        for model_type in ['flat', 'non_flat']:
            name = f'{dataset}_{model_type}'
            config = get_base_experiment_config()
            config.update({
                'dataset': dataset,
                'max_seq_len': 128,
                'non_flat_transport': model_type == 'non_flat',
                'cocycle_relaxation': 1.0 if model_type == 'non_flat' else 0.0,
            })
            configs[name] = config
    return configs
