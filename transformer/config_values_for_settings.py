# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 17:45:31 2026

@author: chris and christine
"""

em_phi_p_CONFIG = {
    # === Architecture ===
    'vocab_size':                 50257,
    'embed_dim':                  20,
    'max_seq_len':                128,
    
    'batch_size':                 32, 
    'max_steps':                  30000,
    
    'stride':                     128,                                                                                                'random_offset_per_epoch': True,
    'stride_base_seed':           6,
    
    'n_layers':                   1,
    'ffn_n_iterations':           1,
    
    'alpha_divergence':           0.2,
    #'grad_accumulation_steps': 1,
    #'gradient_checkpoint_vfe': False,
    
    'gauge_dim':                   10,
    'irrep_spec':       [('fund', 2, 10)],

    'use_prior_bank':             False,
    'gauge_fixed_priors':         False,

    'learnable_pb_temperature':   False,    #prior bank temperature
    'mask_self_attention':        False,  # Prevent attention collapse?
  

    'kappa_beta':                 1,
    'kappa_warmup_steps':         7500,  # freeze kappa for first n steps
    'learnable_head_kappa':       False, # If True, learn per-head κ_h via log_kappa_per_head

    'e_step_sigma_floor':         0.01,   # Floor on σ_p inside E-step (caps 1/σ_p at 1/floor)
    
    # === EM gradient-flow mode ===
    'em_mode':                    'em_phi_p',  # - 'ift_phi' (default) — mu_p, sigma_p attached, full IFT phi gradient
                                                # - 'em_phi_q' — clean EM, phi in q, all detached at boundary
                                                # - 'em_phi_p' — clean EM, phi frozen in E-step

    # === GL(K) determinant control (off by default; pick at most one) ===
    # GL(K) has an unbounded trace direction that L2-norm clamping does not
    # constrain — det(Ω_ij) = exp(tr(φ_i − φ_j)) blows up on outlier tokens.
    # Recommended for gauge_group='GLK' with phi_max_norm > ~3.
    'phi_project_slk':            False,   # Hard project φ → sl(K): det(Ω) ≡ 1 always
    'phi_trace_clamp':            0.75,    # Soft cap |tr(φ·G)| ≤ T (e.g., 0.35 → det ∈ [0.5, 2])


    'active_inference':           False,   #requires priorbank true
    
    'cache_decode_priors':        False,
    'skip_attention':             False,   #skips ad hoc attention sublayer
    
    # === M-step: Optimizer ===  
    'optimizer_type':             'riemannian_adam',# or 'natural_gradient' or 'adamw' or 'riemannian_adam'
    'phi_optimizer_metric':       'killing',
    'fisher_ema_decay':           0.90,            # for natural_gradient
    'fisher_damping':             1e-2,              # for natural_gradient


    'use_layernorm':              True,   #breaks gauge equivariance
    'use_residual':               True,  #set False if skip-attention=True
    'use_output_projection':      True,
    'evolve_sigma':               True,
    'evolve_phi':                 True,  #M-step phi evolution
    'evolve_phi_e_step':          True,
    'normalize_ce_by_dim':        True, 
    
    'E_learnable_alpha':          True,   # Adaptive α_i = c0/(b0 + KL) per dimension   
    'E_learnable_lr':             True,   # Learnable E-step LR
    
    'min_lr_ratio':               0.1,
    'lr_decay':                   'cosine',   #'linear', 'cosine', 'constant'
    
    'norm_type':                  'layernorm',  # 'layernorm' | 'rmsnorm' | 'mahalnorm' | 'none'
    'residual_type':              'additive',    # 'additive': mu_q = mu_q + mu_sub 
                                         # 'delta':    mu_q = mu_q + (mu_sub - mu_normalized),
    
    # === E-step Weights ===
 
    'E_alpha':                    1,      # E-step prior coupling weight
    'E_lambda_belief':            10,    # E-step belief alignment weight
    'E_lambda_softmax':           0,
       
    # === E-step Learning Rates ===
    
    'E_mu_q_lr':                  0.3,    # E-step μ step size (whitened, within trust=2.0)
    'E_sigma_q_lr':               0.015,   # E-step σ step size (conservative)    
    'E_phi_lr':                   0.05,   # E-step φ step size

    # === M-step Weights ===        
    
    'M_alpha':                    0.00,   # M-step KL(q||p) self-consistency
    'M_beta':                     0.0,    # M-step belief alignment
    'mass_phi':                   0.00,    # Gauge prior: (mass_φ/2)||φ||²
    'lambda_hyper':               0.0,    # KL(s||h) explicit loss (pulls tokens toward centroid)
    'lambda_gamma':               0,
    # === M-step Learning Rates (AdamW parameter groups) ===
    
    'M_mu_p_lr':                  0.07,   # M-step prior mean embeddings (μ_p) 0.05
    'M_sigma_p_lr':               0.015,     # M-step prior covariance embeddings (log σ_p) 0.015
    'M_phi_lr':                   0.0036,    # M-step gauge frame embeddings (φ) 0.0075
    
    # === M-step Other LR's (AdamW parameter groups) ===
    'M_vfe_hyperparam_lr':        0.095,  # M-step VFE hyperparams (raw_c0, raw_b0, raw_lr) 0.05
    'M_attention_lr':             0.013,  # M-step attention params (W_O, constant_omega) was0.06
    'M_output_lr':                0.05,  # M-step output projection (vocab logits) 0.05
    'embed_weight_decay':         0.0016,   # L2 hyper-prior on embeddings (μ_p, σ_p, φ) via AdamW
    'non_embed_weight_decay':     0.0043,  # L2 on non-embedding params (attention, output)

    # === Gauge group: GL(K) with multi-head block-diagonal structure ===
    'gauge_group':                'GLK',
    'gauge_mode':                 'learned',
    'gauge_param':                'phi',

    

    'diagonal_covariance':        True,
     
    'exact_diagonal_transport':   False,  # exact diagonal transport - more expensive                                        
    'isotropic_covariance':       False, # If True, force Σ = σ²I (scalar variance × identity)
    'enforce_orthogonal':         False,    
    'learnable_reflection':       False,# Per-token s_i ∈ {±1}^K → O(K)  - enforce orthogonal=true with glk
                                        # Set gauge-mode=constant and the above 3 = true for transf limit

    # === Phi gradient geometry ===
    'phi_natural_gradient':       'killing',

    # === Position encoding ===
    'use_rope':                   True,
    'rope_base':                  100,   
    'rope_full_gauge':            'off', # 'off', 'both', 'vfe_only'
    'pos_encoding_mode':          'none',

    # === Embedding init ===
    'mu_init_std':                0.4,
    'phi_scale':                  0.05,
    
    'mu_normalize':               False,
    'mu_max_norm':                None,


    # === Logging ===
    'log_interval':               200,
    'eval_interval':              2000,
    'checkpoint_interval':        25000,
    'semantic_analysis_interval': 4000,
    'gauge_geometry_interval':    4000,   # Gauge field Dirichlet energy + invariants
    'fiber_trajectory_interval':  4000,   # Fisher-Rao E-step trajectory (requires ffn_n_iterations > 1)

    # =================================================================
    # NON-FLAT GAUGE TRANSPORT (holonomy)
    # =================================================================
    # When enabled, transport acquires an edge-local connection δ_ij:
    #   phi path:  Ω_ij = exp(φ_i·G) · exp(α·δ_ij·G) · exp(-φ_j·G)
    #   omega path: Ω_ij = Ω_i · exp(α·δ_ij·G) · Ω_j⁻¹
    # δ_ij is zero-initialized so the model starts flat and learns
    # curvature only where the data warrants it.
    # Holonomy H_ijk = Ω_ij·Ω_jk·Ω_ki ≠ I when δ ≠ 0.
    
    'non_flat_transport':         False,        # Enable edge-dependent connection δ_ij
    'cocycle_relaxation':         0.5,          # Scale for δ_ij: 0=flat, 1=fully non-flat    
    'connection_type':            'bilinear',  # 'bilinear' (δ_ij^a = μ_i^T W^a μ_j) | 'mlp'   
    'connection_hidden_dim':      64,   # Hidden dim for MLP connection (ignored for bilinear)   
    'connection_init_scale':      0.01,   # W init scale (0=flat saddle point, 0.01 recommended)    
    'holonomy_penalty':           0.0,  # λ_H · E[‖C_ijk - I‖²_F] regularizer (0 = off)

    # === Cross Head Gauge Couplings ====
    #Option A: couple just 0↔1, head 2 stays independent
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
    

    # === Regularization ===
    'sigma_ce_scale':              0.7,
    'sigma_max':                   12.0,
    'grad_clip':                   50.0,
    'hidden_dim':                  508,
    
    
    'warmup_steps':                100,
    'num_workers':                 0,   # 0 is faster on Windows (spawn multiprocessing overhead)
    
    'use_amp':                     False, 
    'use_compile':                 False,
    'compile_mode':                'default',  # 'default', 'reduce-overhead', 'max-autotune'


    # ===== Active Inference =======
    'active_inference_pragmatic_weight':  2,   # start small
    'active_inference_epistemic_weight':  5,   # keep both ON to avoid feedback loop
    'active_inference_epistemic_samples': 6,     # MC samples for BALD
}
