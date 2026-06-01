# -*- coding: utf-8 -*-
"""Adversarial perf/numerics verification probe (CPU, torch 2.x).

Confirms/refutes:
  PERF-1  fused_block_matrix_exp_pairs called 2x per forward (phi evolve ON);
          1x when evolve_phi_e_step=False.
  PERF-1b _compute_phi_grad accepts cached_block_exp_pairs but never reads it.
  PERF-2  track_iteration_diagnostics forces host syncs every forward.
  PERF-4  IrrepMultiHeadAttention under skip: 0 trainable params, ~40k buffer elems.
  NUM     alpha_divergence=0.3 path: 0 nonfinite grads nominal + phi-stress.
"""
import os, sys, inspect, types
import torch

torch.manual_seed(0)
ROOT = r"C:\Users\chris and christine\Desktop\VFE_1.0"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from transformer.core.model import GaugeTransformerLM
import transformer.core.variational_ffn as vffn

DEVICE = torch.device("cpu")

# Tiny config mirroring EM_CONFIG live values (smaller dims/seq for speed).
def make_cfg(evolve_phi_e_step=True, track_diag=True, evolve_phi=True):
    return {
        'vocab_size': 128, 'embed_dim': 20, 'max_seq_len': 16, 'batch_size': 2,
        'n_layers': 1, 'ffn_n_iterations': 1,
        'alpha_divergence': 0.3,
        'gauge_dim': 10, 'irrep_spec': [('fund', 2, 10)],
        'gauge_group': 'GLK', 'gauge_mode': 'learned', 'gauge_param': 'phi',
        'use_prior_bank': False, 'gauge_fixed_priors': False,
        'mask_self_attention': False, 'causal_lower_triangle': False,
        'kappa_beta': 1, 'learnable_head_kappa': False,
        'include_attention_entropy': True,
        'e_step_sigma_floor': 0.01,
        'em_mode': 'ift_phi',
        'phi_project_slk': False, 'phi_trace_clamp': 0.75,
        'active_inference': False, 'skip_attention': True,
        'use_layernorm': True, 'use_residual': False,
        'use_output_projection': False, 'use_equivariant_head_mixer': False,
        'evolve_sigma': True, 'evolve_phi': evolve_phi,
        'evolve_phi_e_step': evolve_phi_e_step,
        'E_learnable_alpha': True, 'E_learnable_lr': True,
        'norm_type': 'layernorm',
        'closed_form_e_step': False, 'n_picard_steps': 0,
        'E_alpha': 1, 'E_lambda_belief': 10, 'E_lambda_softmax': 0,
        'E_mu_q_lr': 0.3, 'E_sigma_q_lr': 0.015, 'E_sigma_q_trust': 5.0, 'E_phi_lr': 0.05,
        'diagonal_covariance': True, 'exact_diagonal_transport': False,
        'isotropic_covariance': False, 'enforce_orthogonal': False,
        'phi_natural_gradient': 'killing',
        'use_rope': True, 'rope_base': 100, 'rope_full_gauge': 'off',
        'pos_encoding_mode': 'none',
        'mu_init_std': 0.4, 'phi_scale': 0.05,
        'non_flat_transport': False,
        'track_layer_diagnostics': True,
        'track_iteration_diagnostics': track_diag,
        'diagnostics_interval': 25,
        'tie_embeddings': False, 'ffn_mode': 'VFE_dynamic',
        'spd_floor_mode': 'eigclamp',
        'sigma_max': 12.0, 'sigma_ce_scale': 0.7,
        'hidden_dim': 64,
    }


def count_matexp_calls(cfg, label):
    """Monkeypatch fused_block_matrix_exp_pairs to count calls in one forward."""
    orig = vffn.fused_block_matrix_exp_pairs
    counter = {'n': 0}
    def wrapped(*a, **k):
        counter['n'] += 1
        return orig(*a, **k)
    vffn.fused_block_matrix_exp_pairs = wrapped
    try:
        m = GaugeTransformerLM(cfg).to(DEVICE)
        m.train()
        B, N = cfg['batch_size'], cfg['max_seq_len']
        ids = torch.randint(0, cfg['vocab_size'], (B, N))
        out = m(ids)
        counter['after_forward'] = counter['n']
    finally:
        vffn.fused_block_matrix_exp_pairs = orig
    print(f"[PERF-1] {label}: fused_block_matrix_exp_pairs calls in one forward = {counter['after_forward']}")
    return counter['after_forward']


def check_dead_param():
    sig = inspect.signature(vffn.VariationalFFNDynamic._compute_phi_grad)
    has = 'cached_block_exp_pairs' in sig.parameters
    src = inspect.getsource(vffn.VariationalFFNDynamic._compute_phi_grad)
    # Drop the def...): header lines (up to and incl. the docstring-less colon),
    # then count remaining references to the name in the executable body.
    lines = src.split('\n')
    # find first line that is the end of the signature (the param appears at def)
    body_lines = [ln for ln in lines if 'cached_block_exp_pairs' in ln]
    # exclude the signature declaration line
    body_refs = [ln.strip() for ln in body_lines
                 if 'cached_block_exp_pairs:' not in ln]
    print(f"[PERF-1b] _compute_phi_grad has param cached_block_exp_pairs={has}; "
          f"body references (excl. signature decl)={len(body_refs)} -> {body_refs}")
    return has, len(body_refs)


def count_diag_syncs(cfg, force_ffn_flag=False):
    """Count host-sync ops (.cpu()/.item()) hit during one forward by patching Tensor."""
    import torch as _t
    counts = {'cpu': 0, 'item': 0, 'tolist': 0}
    orig_cpu = _t.Tensor.cpu
    orig_item = _t.Tensor.item
    orig_tolist = _t.Tensor.tolist
    def c_cpu(self, *a, **k):
        counts['cpu'] += 1; return orig_cpu(self, *a, **k)
    def c_item(self, *a, **k):
        counts['item'] += 1; return orig_item(self, *a, **k)
    def c_tolist(self, *a, **k):
        counts['tolist'] += 1; return orig_tolist(self, *a, **k)
    m = GaugeTransformerLM(cfg).to(DEVICE)
    m.train()
    if force_ffn_flag:
        _first_block(m).ffn.track_iteration_diagnostics = True
    B, N = cfg['batch_size'], cfg['max_seq_len']
    ids = torch.randint(0, cfg['vocab_size'], (B, N))
    _t.Tensor.cpu = c_cpu; _t.Tensor.item = c_item; _t.Tensor.tolist = c_tolist
    try:
        m(ids)
    finally:
        _t.Tensor.cpu = orig_cpu; _t.Tensor.item = orig_item; _t.Tensor.tolist = orig_tolist
    return counts


def _first_block(m):
    for mod in m.modules():
        if mod.__class__.__name__ == 'GaugeTransformerBlock':
            return mod
    return None


def check_attention_buffers(cfg):
    m = GaugeTransformerLM(cfg).to(DEVICE)
    block = _first_block(m)
    attn = block.attention
    trainable = sum(p.numel() for p in attn.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in attn.parameters())
    buf_elems = sum(b.numel() for b in attn.buffers())
    buf_detail = {n: tuple(b.shape) for n, b in attn.named_buffers()}
    # ffn generators for comparison
    ffn = block.ffn
    ffn_gen = getattr(ffn, 'generators', None)
    ffn_gen_shape = tuple(ffn_gen.shape) if ffn_gen is not None else None
    print(f"[PERF-4] attention trainable params={trainable}, total params={total_params}, "
          f"buffer elems={buf_elems}")
    print(f"[PERF-4] attention buffers: {buf_detail}")
    print(f"[PERF-4] ffn.generators shape={ffn_gen_shape}")
    return trainable, buf_elems, buf_detail, ffn_gen_shape


def nan_probe(cfg, stress=None, label="nominal"):
    m = GaugeTransformerLM(cfg).to(DEVICE)
    m.train()
    if stress is not None:
        stress(m)
    B, N = cfg['batch_size'], cfg['max_seq_len']
    ids = torch.randint(0, cfg['vocab_size'], (B, N))
    out = m(ids)
    logits = out[0] if isinstance(out, (tuple, list)) else out
    logits_finite = bool(torch.isfinite(logits).all())
    # Build a standard next-token CE loss from logits for the backward NaN probe.
    tgt = torch.randint(0, cfg['vocab_size'], (B, N))
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1)
    )
    nonfinite_grads = None
    if loss.requires_grad:
        m.zero_grad(set_to_none=True)
        loss.backward()
        nonfinite_grads = sum(
            1 for p in m.parameters()
            if p.grad is not None and not torch.isfinite(p.grad).all()
        )
    print(f"[NUM] {label}: loss={None if loss is None else float(loss):.4f}, "
          f"logits_finite={logits_finite}, nonfinite_grad_params={nonfinite_grads}")
    return logits_finite, nonfinite_grads


def stress_phi(m):
    with torch.no_grad():
        for name, p in m.named_parameters():
            if 'phi' in name.lower():
                p.mul_(30.0)


def check_ffn_diag_flag(cfg, label):
    m = GaugeTransformerLM(cfg).to(DEVICE)
    ffn = _first_block(m).ffn
    val = getattr(ffn, 'track_iteration_diagnostics', 'MISSING')
    print(f"[PERF-2] {label}: FFN.track_iteration_diagnostics = {val}")
    return val


if __name__ == "__main__":
    print("=" * 70)
    print("PERF-1: matrix_exp call count")
    n_live = count_matexp_calls(make_cfg(evolve_phi=True, evolve_phi_e_step=True), "LIVE evolve_phi=T evolve_phi_e_step=T")
    n_estepoff = count_matexp_calls(make_cfg(evolve_phi=True, evolve_phi_e_step=False), "evolve_phi=T evolve_phi_e_step=F (post-loop)")
    n_phioff = count_matexp_calls(make_cfg(evolve_phi=False, evolve_phi_e_step=False), "evolve_phi=F (update_phi=False)")
    print("=" * 70)
    print("PERF-1b: dead parameter")
    check_dead_param()
    print("=" * 70)
    print("PERF-2: does config flag reach the FFN?")
    f_on = check_ffn_diag_flag(make_cfg(track_diag=True), "cfg track_iteration_diagnostics=True")
    f_off = check_ffn_diag_flag(make_cfg(track_diag=False), "cfg track_iteration_diagnostics=False")
    print("PERF-2: diagnostic host syncs (delta isolates the gated block)")
    c_on = count_diag_syncs(make_cfg(track_diag=True))
    c_off = count_diag_syncs(make_cfg(track_diag=False))
    c_forced = count_diag_syncs(make_cfg(track_diag=True), force_ffn_flag=True)
    print(f"[PERF-2] cfg flag True (live plumbing) : {c_on}")
    print(f"[PERF-2] cfg flag False               : {c_off}")
    print(f"[PERF-2] FFN flag FORCED True on module: {c_forced}  <- shows the block's real sync cost when enabled")
    print("=" * 70)
    print("PERF-4: attention buffers under skip")
    check_attention_buffers(make_cfg())
    print("=" * 70)
    print("NUM: NaN probe")
    nan_probe(make_cfg(), label="nominal")
    nan_probe(make_cfg(), stress=stress_phi, label="phi x30 stress")
    print("=" * 70)
    print(f"SUMMARY: matexp LIVE={n_live}  estep_off={n_estepoff}  phi_off={n_phioff}")
