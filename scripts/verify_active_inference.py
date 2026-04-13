"""
Verify the active-inference EFE path is firing in your training config.

Usage:
    1. Edit TRAIN_CONFIG below to match your real EM_CONFIG (or import it
       directly from train_publication.py if that is cleaner).
    2. python scripts/verify_active_inference.py
    3. Read the diagnostic output: it will tell you (a) whether the master
       toggle reached the FFN, (b) whether a readout path was wired in,
       (c) whether forward passes differ with/without the toggle, and
       (d) the per-iteration EFE gradient norm.
"""
import torch
import torch.nn.functional as F
import logging
import sys
import os

# Make sure the project root is on sys.path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

from transformer.core.model import GaugeTransformerLM
import transformer.core.vfe_utils as _vfe_utils_mod


# ---- EDIT THIS: match your actual EM_CONFIG ----
# Default is a small, self-contained config that loads without your data.
# For the real diagnostic, copy your EM_CONFIG verbatim (and bump vocab_size,
# embed_dim, etc. to match).
TRAIN_CONFIG = {
    # === Architecture ===
    'vocab_size':            50257,
    'embed_dim':             20,
    'max_seq_len':           64,
    
    'batch_size':            64, 
    'max_steps':             15000,
    
    'n_layers':              1,
    'ffn_n_iterations':      1,
    
       
    'gauge_dim':                          10,
    'irrep_spec':            [('fund', 2, 10)],

    'use_prior_bank':           True,
    'learnable_pb_temperature': True,
    'mask_self_attention':      True,  #Prevent attention collapse?
  
    'hierarchical_priors':      True,
    'gauge_fixed_priors':       True,    
  
    'active_inference':         True,    #requires priorbank true
    
    'kappa_beta':               1,
    'kappa_warmup_steps':       7500,  # freeze kappa for first n steps
    'learnable_head_kappa':     True, # If True, learn per-head κ_h via log_kappa_per_head
    
    # === M-step: implicit differentiation ===
    'implicit_em':           False,
    'amortized_inference':   True,
    'use_obs_in_vfe':        False,  #cheats when true
       
    # === M-step: Optimizer ===  
    'optimizer_type':        'riemannian_adam',# or 'natural_gradient' or 'adamw' or 'riemannian_adam'
    'fisher_ema_decay':      0.95,            # for natural_gradient
    'fisher_damping':        1e-2,              # for natural_gradient


    'use_layernorm':         True,
    'norm_type':             'layernorm',  # 'layernorm' | 'rmsnorm' | 'none'
    'use_residual':          True,
    'use_output_projection': True,
    'multihead_vfe':         True,
    
    'evolve_sigma':          True,
    'evolve_phi':            True,  #M-step phi evolution
    'evolve_phi_e_step':     True,
    
    'obs_sigma_gradient':    True, # ∂E_q[CE]/∂σ via Hessian diagonal of expected CE
    'e_step_sigma_floor':    0.01,   # Floor on σ_p inside E-step (caps 1/σ_p at 1/floor)
    # === E-step dynamics ===
    
    
    'E_alpha':               1,      # E-step prior coupling weight
    'E_lambda_belief':       1,    # E-step belief alignment weight
    'E_lambda_softmax':      1,
    
    'detach_beta_m_step':    True, #if false need M_beta >0
    
    'E_learnable_alpha':     True,   # Adaptive α_i = c0/(b0 + KL) per dimension

    
    'E_learnable_lr':        True,   # Learnable E-step LR
    'lr_decay':              'linear',
    
    'E_mu_q_lr':             0.1,    # E-step μ step size (whitened, within trust=2.0)
    'E_sigma_q_lr':          0.05,   # E-step σ step size (conservative)
    
    'E_phi_lr':              0.05,   # E-step φ step size

    # === Gauge group: GL(K) with multi-head block-diagonal structure ===
    'gauge_group':              'GLK',
    'gauge_mode':               'learned',
    'gauge_param':              'phi',

    'skip_attention':           False,   #skips ad hoc attention sublayer
    'closed_form_e_step':       False,   #closed form...ignores non-linear softmax gradient


    
    'diagonal_covariance':      True,
    'exact_diagonal_transport': False,  # exact diagonal transport - more expensive
                                        # If True, force Σ = σ²I (scalar variance × identity)
    'isotropic_covariance':     False,
    'enforce_orthogonal':       False,    
    'learnable_reflection':     False,  # Per-token s_i ∈ {±1}^K → O(K)  - enforce orthogonal=true with glk
                                        # Set gauge-mode=constant and the above 3 = true for transf limit
 
    # === VFE loss weights (M-step objective) ===
    # E-step: prior + alignment (no observations with n_iterations=1).
    # CE enters through M-step via IFT (s_k ≈ 0.5 from fixed ffn_alpha=1).
    # alpha=0: KL(q*||p) homogenizes (q* is smoothed, not data-grounded).
    # beta=0: alignment term is vacuum-seeking. E-step handles it internally.
    
    'M_alpha':             0.00,   # M-step KL(q||p) self-consistency
    'M_beta':              0.0,    # M-step belief alignment
    'mass_phi':            0.01,    # Gauge prior: (mass_φ/2)||φ||²
    'lambda_hyper':        0.0,    # KL(s||h) explicit loss (pulls tokens toward centroid)
    'lambda_gamma':        0.0,
    'kappa_gamma':         1.0,

    'embed_weight_decay':     0.05,   # L2 hyper-prior on embeddings (μ_p, σ_p, φ) via AdamW
    'non_embed_weight_decay': 0.01,  # L2 on non-embedding params (attention, output)
    
    # === Phi gradient geometry ===
    'phi_natural_gradient':       'killing',
    'use_killing_form':           True,
    'killing_form_sym_dampening': 0.5,

    # === Position encoding ===
    'rope_full_gauge':    True,    #requires diagonal cov
    'use_rope':           True,
    'rope_base':          50, 
    'pos_encoding_mode': 'none',

    # === Embedding init ===
    'mu_init_std':     1.0,
    'phi_scale':       1.0,
    
    'mu_normalize':    False,
    'mu_max_norm':     None,


    # === M-step learning rates (AdamW parameter groups) ===
    # These update nn.Parameter objects via backprop. The E-step (inner VFE
    # loop) uses e_step_mu_lr / e_step_sigma_lr / e_step_phi_lr above.
    # mu_embed and log_sigma_diag have dual roles: they initialize E-step
    # beliefs (q₀) AND serve as prior parameters (μ_p, σ_p), so these rates
    # indirectly affect E-step initialization speed.
    'M_mu_p_lr':           0.05,   # M-step prior mean embeddings (μ_p) 0.05
    'M_sigma_p_lr':        0.015,  # M-step prior covariance embeddings (log σ_p) 0.015
    'M_phi_lr':            0.0075, # M-step gauge frame embeddings (φ) 0.0075
    'M_vfe_hyperparam_lr': 0.075,  # M-step VFE hyperparams (raw_c0, raw_b0, raw_lr) 0.05
    'M_attention_lr':      0.005,  # M-step attention params (W_O, constant_omega)0.005
    'M_output_lr':         0.05,  # M-step output projection (vocab logits) 0.05
    
    # === Logging ===
    'log_interval':               100,
    'eval_interval':              1000,
    'checkpoint_interval':        25000,
    'semantic_analysis_interval': 10000,

    'use_deq':           False,
    'deq_include_phi':   True,    # Corrects M-step phi gradient
    'deq_neumann_terms': 0,

    # =================================================================
    # NON-FLAT GAUGE TRANSPORT (holonomy)
    # =================================================================
    # When enabled, transport acquires an edge-local connection δ_ij:
    #   phi path:  Ω_ij = exp(φ_i·G) · exp(α·δ_ij·G) · exp(-φ_j·G)
    #   omega path: Ω_ij = Ω_i · exp(α·δ_ij·G) · Ω_j⁻¹
    # δ_ij is zero-initialized so the model starts flat and learns
    # curvature only where the data warrants it.
    # Holonomy H_ijk = Ω_ij·Ω_jk·Ω_ki ≠ I when δ ≠ 0.
    
    'non_flat_transport':    False,        # Enable edge-dependent connection δ_ij
    'cocycle_relaxation':    0.5,          # Scale for δ_ij: 0=flat, 1=fully non-flat    
    'connection_type':       'bilinear',  # 'bilinear' (δ_ij^a = μ_i^T W^a μ_j) | 'mlp'   
    'connection_hidden_dim': 64,   # Hidden dim for MLP connection (ignored for bilinear)   
    'connection_init_scale': 0.01,   # W init scale (0=flat saddle point, 0.01 recommended)    
    'holonomy_penalty':      0.0,  # λ_H · E[‖C_ijk - I‖²_F] regularizer (0 = off)

    'active_inference_pragmatic_weight': 1,   # start small
    'active_inference_epistemic_weight': 1,   # keep both ON to avoid feedback loop
    'active_inference_epistemic_samples': 4,     # MC samples for BALD



    # Option A: couple just 0↔1, head 2 stays independent
    # 'cross_couplings': [(0, 1), (1, 0)],
    # → super-blocks: [20, 10]  (heads 0,1 merged into GL(20), head 2 alone)
    
    # === Layer/iteration diagnostics ===
    'track_layer_diagnostics':     False,
    'track_iteration_diagnostics': False,
    'diagnostics_interval':        25,
    
    
    'tie_embeddings':              False,
    'ffn_mode':                    'VFE_dynamic',
    
    'debug_vfe_grads':             False,
    'verbose_diagnostics':         False,
    
    # === Multi-layer depth signal ===
    'aux_layer_loss':  True,   # Enable for multi-layer: per-layer M-step CE loss
    'aux_loss_weight': 0.3,     # Weight for auxiliary per-layer CE losses
    
    
    # === Regularization ===
    'sigma_ce_scale':  1,
    'sigma_max':       12.0,
    'grad_clip':       5,
    'hidden_dim':      508,
    'warmup_steps':    100,
    'num_workers':     10,
    
    'use_amp':         False, 
    'use_compile':     False,
    'compile_mode':    'reduce-overhead'  # 'default', 'reduce-overhead', 'max-autotune'

}
# --------------------------------------------------


def inspect_model(cfg, label):
    """Build a model and report the internal state of the EFE wiring."""
    torch.manual_seed(0)
    model = GaugeTransformerLM(cfg).eval()
    print(f'\n=== {label} ===')
    ffn = model.transformer.blocks[0].ffn
    print(f'  block[0].ffn._ai_enabled            = {getattr(ffn, "_ai_enabled", "MISSING")}')
    print(f'  block[0].ffn._ai_pragmatic_weight   = {getattr(ffn, "_ai_pragmatic_weight", "MISSING")}')
    print(f'  block[0].ffn._ai_epistemic_weight   = {getattr(ffn, "_ai_epistemic_weight", "MISSING")}')
    print(f'  block[0].ffn._ai_epistemic_samples  = {getattr(ffn, "_ai_epistemic_samples", "MISSING")}')
    bank_ref = ffn.__dict__.get('_prior_bank_ref', None)
    wout_ref = ffn.__dict__.get('_ai_w_out_ref', None)
    print(f'  block[0].ffn._prior_bank_ref        = {type(bank_ref).__name__ if bank_ref is not None else "None"}')
    print(f'  block[0].ffn._ai_w_out_ref          = {"set" if wout_ref is not None else "None"}')
    if bank_ref is None and wout_ref is None and getattr(ffn, '_ai_enabled', False):
        print('  ** WARNING: EFE enabled but no readout path available — will silently skip **')
    return model


def compare_forwards(cfg_off, cfg_on, input_ids, label):
    """Build two models differing only in the master toggle and compare."""
    torch.manual_seed(0)
    m_off = GaugeTransformerLM(cfg_off).eval()
    with torch.no_grad():
        l_off = m_off(input_ids)
    torch.manual_seed(0)
    m_on = GaugeTransformerLM(cfg_on).eval()
    with torch.no_grad():
        l_on = m_on(input_ids)
    max_diff = (l_off - l_on).abs().max().item()
    mean_diff = (l_off - l_on).abs().mean().item()
    print(f'\n=== Forward-pass comparison: {label} ===')
    print(f'  max  |logit_off - logit_on| = {max_diff:.6e}')
    print(f'  mean |logit_off - logit_on| = {mean_diff:.6e}')
    if max_diff < 1e-6:
        print('  FAIL: forward passes are identical — EFE path is NOT firing')
    else:
        print('  OK:   forward passes differ — EFE path IS firing')
    return max_diff


def measure_efe_grad_norm(cfg, input_ids):
    """Directly call the EFE helper on a representative belief to measure
    its gradient contribution.  Avoids the internal _VFE_GRAD_DEBUG dict
    which gets reset between iterations."""
    from transformer.core.variational_ffn import _compute_active_inference_gradient

    torch.manual_seed(0)
    model = GaugeTransformerLM(cfg).eval()
    ffn0 = model.transformer.blocks[0].ffn

    # Representative belief: embedding lookup for the first token at each position
    # then pass through normalization to simulate the state the FFN would see.
    B, N, K = 2, 16, cfg['embed_dim']
    mu = torch.randn(B, N, K) * 0.5
    sigma = torch.ones(B, N, K) * 0.5

    bank = ffn0.__dict__.get('_prior_bank_ref', None)
    wout_ref = ffn0.__dict__.get('_ai_w_out_ref', None)
    wout_tensor = None
    if wout_ref is not None:
        proj = wout_ref[0] if isinstance(wout_ref, list) else wout_ref
        wout_tensor = proj.weight

    print(f'\n=== Direct EFE gradient magnitude (on a sample belief) ===')
    if bank is None and wout_tensor is None:
        print('  FAIL: no readout path is wired (both prior_bank and w_out are None)')
        return

    prag = cfg.get('active_inference_pragmatic_weight', 0.0)
    epi  = cfg.get('active_inference_epistemic_weight', 0.0)
    samples = cfg.get('active_inference_epistemic_samples', 4)
    tau  = cfg.get('active_inference_decode_tau', 1.0)

    grad_efe = _compute_active_inference_gradient(
        mu_current=mu, sigma_current=sigma, prior_bank=bank,
        pragmatic_weight=prag, epistemic_weight=epi,
        epistemic_samples=samples, decode_tau=tau, w_out=wout_tensor,
    )
    if grad_efe is None:
        print('  FAIL: helper returned None (toggle off, or inference-mode context)')
        return
    norm = grad_efe.norm().item()
    max_abs = grad_efe.abs().max().item()
    print(f'  ||grad_EFE||_F                = {norm:.6e}')
    print(f'  max |grad_EFE|                = {max_abs:.6e}')
    print(f'  readout path used             = {"PriorBank" if bank is not None else "W_out fallback"}')
    print(f'  pragmatic_weight              = {prag}')
    print(f'  epistemic_weight              = {epi}')
    if norm < 1e-4:
        print('  ** WARNING: EFE gradient is tiny; may be overwhelmed by other terms **')
        print('     Try bumping active_inference_pragmatic_weight / active_inference_epistemic_weight')


def main():
    print('Active Inference / EFE Diagnostic')
    print('=' * 60)

    # Build two configs differing only in the master toggle
    cfg_off = dict(TRAIN_CONFIG, active_inference=False)
    cfg_on  = dict(TRAIN_CONFIG, active_inference=True)

    # 1. Inspect the two models
    m_off = inspect_model(cfg_off, 'active_inference=False')
    m_on  = inspect_model(cfg_on, 'active_inference=True')

    # 2. Compare forward passes
    ids = torch.randint(0, TRAIN_CONFIG['vocab_size'], (2, 16),
                        generator=torch.Generator().manual_seed(123))

    # Small weights (user's default)
    compare_forwards(cfg_off, cfg_on, ids,
                     f'defaults (prag={cfg_on["active_inference_pragmatic_weight"]}, '
                     f'epi={cfg_on["active_inference_epistemic_weight"]})')

    # Large weights (to confirm the path is wired)
    cfg_on_loud = dict(cfg_on, active_inference_pragmatic_weight=2.0,
                        active_inference_epistemic_weight=1.0)
    compare_forwards(cfg_off, cfg_on_loud, ids,
                     'large weights (prag=2.0, epi=1.0)')

    # 3. Measure the per-iteration gradient contribution
    measure_efe_grad_norm(cfg_on, ids)
    measure_efe_grad_norm(cfg_on_loud, ids)

    print('\n' + '=' * 60)
    print('Done.  If "forward passes are identical" above, the EFE path is')
    print('not firing — check the inspection output for which field is missing.')


if __name__ == '__main__':
    main()
