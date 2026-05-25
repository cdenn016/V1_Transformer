"""
Click-to-run VFE transformer training.

Edit the config dict below, then press Run. No CLI arguments (per CLAUDE.md).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import logging
import torch

from transformer.vfe.config import VFEConfig
from transformer.vfe.model import VFEModel
from transformer.vfe.trainer import VFETrainer
from transformer.data.datasets import create_dataloaders

logging.basicConfig(level=logging.INFO, format='%(message)s')

# ============================================================================
# CONFIGURATION — edit this dict, then press Run
# ============================================================================

config = {
    # === Structure ===
    'vocab_size':               None,      # populated after dataloader build
    'embed_dim':                40,
    'irrep_spec':               [('fund', 4, 10)],
    
    'batch_size':               64,
    
    'max_seq_len':              128,
    'max_steps':                15000,

    'use_prior_bank':           False,
    'exact_full_cov_decode':    False,
    
    'mask_self_attention':      False,
    'causal_lower_triangle':    True,
    
    # Prior parameterization:
    #   True  = shared base + per-token gauge-orbit (μ_v = A_v @ μ_0). Pure form;
    #           per-token capacity is V·n_gen (phi_embed only).
    #   False = per-token (μ_v, σ_v) lookup; phi retained for transport only.
    #           Per-token capacity is V·(2K + n_gen).
    'gauge_fixed_priors':       False,

    'E_learnable_alpha':        True,
    'learnable_kappa':          False,

    
    'use_equivariant_head_mixer':  True,
    'gauge_covariant_ridge':       True,

    # === E-step dynamics ===
    'n_e_steps':                1,
    'n_layers':                 1,

    'alpha_divergence':         1,
    'include_attention_entropy': True,

    'e_mu_lr':                  0.5,
    'e_sigma_lr':               0.015,
    'e_phi_lr':                 0.00,
   
    'alpha':                    1.0,
    'lambda_align':             2.45,
    'lambda_soft':              0.0,
    'mass_phi':                 0.0,

    'kappa':                    0.5,



    # === Cross-layer prior handoff ===
    'prior_handoff_rho':        0,
    'prior_handoff_sigma':      0,

    # === Covariance ===
    'diagonal_covariance':      True,
    'isotropic_covariance':     False,
    'exact_diagonal_transport': False,
    'enforce_orthogonal':       False,
    
    
    # === Gauge geometry ===
    'gauge_group':              'GLK',
    
    'phi_project_slk':          False,
    'phi_trace_clamp':          0.75,
    
    'phi_preconditioner':       'killing',  # 'clip', 'cartan', 'killing', 'killing_per_block', 'pullback'

    # === Positional encoding ===
    'use_rope':                 False,
    'rope_full_gauge':          'off',
    'rope_base':                150,

    # === Embedding init ===
    'mu_init_std':              0.001,
    'phi_scale':                0.001,
    'sigma_init':               1,

    # === Decode and generation-time EFE (canonical path; consumed by vfe/efe.py) ===
    'epistemic_weight':         0.5,
    'epistemic_samples':        4,
    'decode_tau':               1.0,

    'use_non_flat_transport':       False,
    'non_flat_max_strength':        1.0,  # s_max in s = s_max·tanh(ρ)
    'non_flat_per_edge_delta_max':  1.0,  # δ_max bound on ‖δ_ij·G‖_F
    'nonflat_delta_through_mu':     True, # False = ~17% faster: drop δ-through-μ E-step grad (matrix_exp bwd)

    # === Cross-head coupling (GL(K) multi-head) ===
    # Off-diagonal gauge generators sparsely connecting selected head pairs.
    # Each (a, b) entry adds d_head² generators that span head-a → head-b's
    # subspace; the merged subspace becomes a single super-block with full
    # GL(d_super) gauge. Empty default = standard block-diagonal gl(d_head)^H.
    # See transformer/vfe/cross_coupling_metrics.py and
    # transformer/vfe/cross_coupling_viz.py for diagnostics.
    'cross_couplings':                  [],
    'auto_close_cross_head_basis':      False,
    'validate_cross_head_closure':      True,

    # === Normalization ===
    'norm_type':                'none', #'mahalnorm', 'centered_mahalnorm', 'rmsnorm', 'layernorm'
    'normalize_ce_by_dim':      True,

    # === Training ===
    # Per-group M-step LRs (see vfe/config.py for what each touches).
    'm_mu_lr':                  0.025,
    'm_sigma_lr':               0.0025,
    'm_phi_lr':                 0.015,
    
    'm_hyper_lr':               0.0125,
    'm_other_lr':               0.02,
    'weight_decay':             0.05,
    
    'warmup_steps':             100,
    'grad_clip':                50.0,
    'sigma_max':                12.0,
    
    'bch_order':                3,

    # === Logging / evaluation ===
    'log_interval':             100,
    'eval_interval':            2000,
    'checkpoint_interval':      25000,

    'track_layer_diagnostics':      False,
    'monitor_monotonicity':         False,

}

# ============================================================================
# DATASET — select one: 'wikitext-2', 'wikitext-103', 'wiki-ja', 'wiki-en'
# ============================================================================

SEED = 6                   # Reproducibility seed for torch / numpy / dataloader workers

DATASET = 'wikitext-103'      # ~103M tokens, full-scale training
# DATASET = 'wikitext-2'      # ~2M tokens, fast iteration
# DATASET = 'wiki-ja'         # Japanese Wikipedia, ~190M tokens at default 100K-article cap
# DATASET = 'wiki-en'         # English Wikipedia, ~5B cl100k tokens at full dump (no cap)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Post-training semantic-clustering figures (mu / Sigma / phi / Omega), written
# to <output_dir>/semantic_clustering/{vocab,contextual}/. Opt-in extra compute
# (O(n^2) geometry-faithful distances over up to MAX_POINTS tokens); set
# RUN_SEMANTIC_CLUSTERING = False to skip. See transformer/vfe/semantic_clustering/.
RUN_SEMANTIC_CLUSTERING = True
SEMANTIC_CLUSTERING_MAX_POINTS = 200


def run_post_training_semantic_clustering(
    model, output_dir, loader, dataset=None, device='cpu',
    max_points=200, seed=0,
):
    """Generate post-training semantic-clustering figures for a trained model.

    Writes separate mu/Sigma/phi/Omega cluster figures + metrics for a
    vocab-level view (the learned per-token-type priors) and a contextual view
    (one real batch of token ids) into
    ``<output_dir>/semantic_clustering/{vocab,contextual}/``. Never passes
    targets — the E-step stays blind to labels (Law 1). Returns the output root.
    """
    from pathlib import Path as _Path
    from transformer.vfe.semantic_clustering.pipeline import run_clustering

    sc_root = _Path(output_dir) / 'semantic_clustering'
    model.eval()

    # Vocab view: cluster the learned per-token-type priors.
    run_clustering(
        model, source='vocab', token_ids=None, dataset=dataset,
        outdir=sc_root / 'vocab', max_points=max_points, seed=seed,
    )

    # Contextual view: one real batch of token ids (inputs only — Law 1).
    batch = next(iter(loader))
    if isinstance(batch, dict):
        ctx_ids = batch['input_ids']
    elif isinstance(batch, (tuple, list)):
        ctx_ids = batch[0]
    else:
        ctx_ids = batch
    ctx_ids = ctx_ids.to(device)
    run_clustering(
        model, source='contextual', layer='final', token_ids=ctx_ids,
        dataset=dataset, outdir=sc_root / 'contextual',
        max_points=max_points, seed=seed,
    )
    return sc_root


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    import random
    import numpy as np
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # Build dataloaders — suppress verbose per-split prints to match the
    # compact init banner emitted below (mirrors experiment_runner.py's
    # _datasets_mod.QUIET = True block).
    from transformer.data import datasets as _datasets_mod
    _prev_quiet = _datasets_mod.QUIET
    _datasets_mod.QUIET = True
    try:
        # num_workers=2 keeps CPU tokenization pipelined with the GPU step;
        # process-spawn overhead on Windows makes 4+ workers a poor trade.
        train_loader, val_loader, test_loader, vocab_size = create_dataloaders(
            max_seq_len=config['max_seq_len'],
            batch_size=config['batch_size'],
            vocab_size=config.get('vocab_size'),
            dataset=DATASET,
            include_test=True,
            num_workers=2,
        )
    finally:
        _datasets_mod.QUIET = _prev_quiet

    # Override vocab_size from tokenizer
    config['vocab_size'] = vocab_size

    cfg = VFEConfig(**config)
    model = VFEModel(cfg)
    model = model.to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())

    # Output directory for metrics, checkpoints, figures
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'vfe_runs/{DATASET}_{timestamp}'

    # =================================================================
    # Compact init banner — mirrors transformer/training/experiment_runner.py
    # run_single_experiment() banner so /vfe and the legacy path read the same.
    # =================================================================
    from transformer.baselines.flops_counter import (
        count_gauge_transformer_flops, format_flops,
    )

    seq_len = cfg.max_seq_len
    batch_size = config['batch_size']
    max_steps = cfg.max_steps
    n_heads = cfg.irrep_spec[0][1] if cfg.irrep_spec else 1
    head_dim = cfg.irrep_spec[0][2] if cfg.irrep_spec else cfg.embed_dim
    phi_dim = cfg.embed_dim * cfg.embed_dim

    flops_result = count_gauge_transformer_flops(
        vocab_size=cfg.vocab_size,
        embed_dim=cfg.embed_dim,
        n_layers=cfg.n_layers,
        n_heads=n_heads,
        head_dim=head_dim,
        seq_len=seq_len,
        batch_size=batch_size,
        phi_dim=phi_dim,
        ffn_n_iterations=cfg.n_e_steps,
        use_rope=cfg.use_rope,
        diagonal_covariance=cfg.diagonal_covariance,
    )
    step_flops = flops_result['step_total']
    total_flops = step_flops * max_steps
    total_tokens = max_steps * batch_size * seq_len

    try:
        dataset_tokens = len(train_loader.dataset.tokens)
    except AttributeError:
        dataset_tokens = None

    params_str = f"{n_params/1e6:.2f}M" if n_params >= 1e6 else f"{n_params/1e3:.1f}K"
    dataset_name = DATASET.upper()
    coverage_str = ""
    if dataset_tokens and dataset_tokens > 0:
        coverage_str = f" ~ {total_tokens / dataset_tokens * 100:.0f}% {DATASET.lower()}"

    logger = logging.getLogger()
    logger.info("=" * 70)
    logger.info(f"  Gauge VFE Transformer | {params_str} params | {DEVICE}")
    logger.info(f"  K={cfg.embed_dim}, N={seq_len}, L={cfg.n_layers}, "
                f"heads={n_heads} | {dataset_name} ({total_tokens/1e6:.0f}M tokens)")
    logger.info(f"  {max_steps:,} steps | B={batch_size}{coverage_str} | "
                f"FLOPs/step: {format_flops(step_flops)} | Total: {format_flops(total_flops)}")
    logger.info(f"  LR: mu={cfg.m_mu_lr}, sigma={cfg.m_sigma_lr}, "
                f"phi={cfg.m_phi_lr}, out={cfg.m_other_lr}")
    logger.info(f"  VFE weights: alpha={cfg.alpha}, lambda_align={cfg.lambda_align}, "
                f"lambda_soft={cfg.lambda_soft} | kappa={cfg.kappa}")
    extras = []
    if cfg.use_non_flat_transport:
        extras.append("non-flat")
    if cfg.use_prior_bank:
        extras.append("prior-bank")
    if cfg.learnable_kappa:
        extras.append("learnable-kappa")
    if cfg.E_learnable_alpha:
        extras.append("E-learnable-alpha")
    if extras:
        logger.info(f"  Features: {', '.join(extras)}")
    logger.info(f"  Seed: {SEED} | Dataset: vocab={vocab_size}, "
                f"train_batches={len(train_loader)}, val_batches={len(val_loader)}, "
                f"test_batches={len(test_loader)}")
    logger.info(f"  Output: {output_dir}")
    logger.info("=" * 70)

    # Train
    trainer = VFETrainer(
        model, cfg, train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=DEVICE,
        output_dir=output_dir,
    )
    trainer.train(num_steps=cfg.max_steps)

    # trainer.train() may rename the run directory to the measured
    # <test_ppl>=test-PPL_K=<K>_<gauge_label> form (trainer.py:1816-1831),
    # updating trainer.output_dir but NOT this local variable. Refresh it so
    # post-training artifacts (semantic-clustering figures) land in the renamed
    # directory instead of recreating the stale timestamped one. Safe in every
    # branch: when the rename is skipped, trainer.output_dir is the original Path.
    output_dir = trainer.output_dir

    # Unconditional final validation pass. The trainer's periodic eval is
    # gated on (step+1) % eval_interval == 0 (trainer.py:1695); under defaults
    # (eval_interval=2000, max_steps=15000) it fires 7 times, but if the user
    # lowers max_steps below eval_interval no val PPL is ever recorded. The
    # trainer's own final-pass branch at trainer.py:1785 runs `test_loader`,
    # not `val_loader`, so it does not cover this case.
    if val_loader is not None:
        try:
            final_val_metrics = trainer.evaluate()
            logger.info(
                f"Final val eval | PPL={final_val_metrics.get('val_ppl', float('nan')):.4f}"
                f"  CE={final_val_metrics.get('val_loss', float('nan')):.4f}"
                f"  BPC={final_val_metrics.get('val_bpc', float('nan')):.4f}"
            )
        except (RuntimeError, ValueError) as exc:
            logger.warning(f"Final trainer.evaluate() on val_loader failed: {exc!r}")

    # Post-training semantic-clustering figures (non-fatal: never break a
    # completed run if visualization fails).
    if RUN_SEMANTIC_CLUSTERING:
        try:
            sc_loader = val_loader if val_loader is not None else train_loader
            sc_dataset = getattr(sc_loader, 'dataset', None)
            sc_root = run_post_training_semantic_clustering(
                model, output_dir, sc_loader, dataset=sc_dataset,
                device=DEVICE, max_points=SEMANTIC_CLUSTERING_MAX_POINTS, seed=SEED,
            )
            logger.info(f"Semantic clustering figures written to {sc_root}")
        except Exception as exc:
            logger.warning(f"Semantic clustering step failed (non-fatal): {exc!r}")
