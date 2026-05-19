"""
Diagnostic probe: numerical scale of analytic vs autograd (mu, sigma) gradient.

Investigates WHY VFEConfig.use_autograd_mu_sigma=True produces worse training
results than =False at the same E-step learning rates.

Hypothesis: the autograd path includes the key-side contribution to dF/dmu_k,
dF/dsigma_k (from KL(q_j || Omega_ji q_i), j != k), which the analytic kernel
ignores (mean-field convention). At the same (e_mu_lr, e_sigma_lr) tuned for
the analytic path, the autograd path produces a larger gradient that overshoots.

Click-to-run from repo root:
    python scripts/diagnostic_autograd_gradient_scale_probe.py
"""

from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch

from transformer.vfe.config import VFEConfig
from transformer.vfe.e_step import VFEEStep
from transformer.vfe.attention import compute_kl_attention, compute_gauge_transport
from transformer.core.vfe_gradients import (
    compute_vfe_gradients_gpu,
    compute_natural_gradient_gpu,
)


def make_config(use_autograd):
    return VFEConfig(
        vocab_size=1000,
        embed_dim=20,
        irrep_spec=[("fund", 2, 10)],
        n_layers=1,
        max_seq_len=128,
        n_e_steps=1,
        e_mu_lr=0.4,
        e_sigma_lr=0.015,
        e_phi_lr=0.05,
        alpha=1.0,
        alpha_divergence=1.0,
        E_learnable_alpha=False,
        lambda_align=4.0,
        lambda_soft=0.0,
        kappa=1.0,
        include_attention_entropy=True,
        learnable_kappa=False,
        diagonal_covariance=True,
        isotropic_covariance=False,
        exact_diagonal_transport=False,
        sigma_init=0.4,
        sigma_max=12.0,
        gauge_group="GLK",
        phi_preconditioner="killing",
        phi_project_slk=False,
        phi_trace_clamp=0.75,
        enforce_orthogonal=False,
        mask_self_attention=True,
        use_rope=True,
        rope_full_gauge="off",
        rope_base=150.0,
        bch_order=3,
        mu_init_std=0.001,
        phi_scale=0.001,
        use_prior_bank=False,
        gauge_fixed_priors=False,
        norm_type="layernorm",
        gauge_parameterization="phi",
        use_non_flat_transport=False,
        active_inference=False,
        batch_size=8,
        max_steps=1,
        warmup_steps=1,
        use_autograd_mu_sigma=use_autograd,
        device="cpu",
    )


def build_generators(cfg):
    from math_utils.generators import (
        generate_glK_multihead_generators,
        generate_glK_generators,
        generate_multi_irrep_generators,
        generate_multi_irrep_soN_generators,
    )
    K = cfg.embed_dim
    irrep_spec = cfg.irrep_spec
    if cfg.gauge_group == "GLK":
        if len(irrep_spec) == 1 and irrep_spec[0][1] > 1:
            _, n_heads, _ = irrep_spec[0]
            gens = generate_glK_multihead_generators(K, n_heads)
        else:
            gens = generate_glK_generators(K)
    elif cfg.gauge_group == "SO3":
        gens = generate_multi_irrep_generators(irrep_spec)
    else:
        gens = generate_multi_irrep_soN_generators(irrep_spec, K)
    if not isinstance(gens, torch.Tensor):
        gens = torch.from_numpy(gens).float()
    return gens.float()


def build_causal_mask(B, N):
    # Convention used throughout transformer/core/attention.py: mask is a (B,N,N)
    # multiplicative gate, mask == 0 means "this (i,j) pair is disallowed".
    # We build a strict lower-triangular causal mask: allowed where j <= i.
    allowed = torch.tril(torch.ones(N, N, dtype=torch.float32))
    return allowed.unsqueeze(0).expand(B, -1, -1).contiguous()


def run_analytic(estep, mu, sigma, phi, mu_p, sigma_p, mask, kappa, eps=1e-6):
    block_exp_pairs = compute_gauge_transport(
        phi, estep.generators, estep.irrep_dims,
        enforce_orthogonal=estep.enforce_orthogonal,
    )
    beta, kl_matrix = compute_kl_attention(
        mu, sigma, phi, estep.generators,
        estep.irrep_dims, kappa, mask,
        use_rope=estep.use_rope,
        rope_base=estep.rope_base,
        cached_block_exp_pairs=block_exp_pairs,
        enforce_orthogonal=estep.enforce_orthogonal,
        mask_self_attention=estep.mask_self_attention,
        exact_diagonal_transport=estep.exact_diagonal_transport,
    )
    grad_mu, grad_sigma = compute_vfe_gradients_gpu(
        mu_q=mu,
        sigma_q=sigma,
        mu_p=mu_p,
        sigma_p=sigma_p,
        beta=beta,
        phi=phi,
        generators=estep.generators,
        alpha=estep.alpha,
        alpha_c0=None,
        alpha_div=estep.alpha_divergence,
        lambda_belief=estep.lambda_align,
        lambda_softmax=0.0 if estep.include_attention_entropy else estep.lambda_soft,
        kappa=kappa,
        eps=eps,
        compute_sigma_align_grad=True,
        irrep_dims=estep.irrep_dims,
        enforce_orthogonal=estep.enforce_orthogonal,
        cached_block_exp_pairs=block_exp_pairs,
        use_rope=estep.use_rope,
        rope_base=estep.rope_base,
        exact_diagonal_transport=estep.exact_diagonal_transport,
        gauge_covariant_ridge=estep.gauge_covariant_ridge,
    )
    return grad_mu, grad_sigma, beta


def run_autograd(estep, mu, sigma, phi, mu_p, sigma_p, mask, kappa, eps=1e-6):
    block_exp_pairs = compute_gauge_transport(
        phi, estep.generators, estep.irrep_dims,
        enforce_orthogonal=estep.enforce_orthogonal,
    )
    grad_mu, grad_sigma = estep._compute_mu_sigma_grad_autograd(
        mu=mu, sigma=sigma,
        mu_p=mu_p, sigma_p=sigma_p,
        phi=phi,
        alpha_eff=estep.alpha,
        block_exp_pairs=block_exp_pairs,
        mask=mask,
        kappa=kappa,
        eps=eps,
        is_diagonal=True,
    )
    return grad_mu, grad_sigma


def rms(x):
    return float(x.pow(2).mean().sqrt())


def per_token_cosine(g1, g2):
    g1f = g1.reshape(-1, g1.shape[-1])
    g2f = g2.reshape(-1, g2.shape[-1])
    n1 = g1f.norm(dim=-1).clamp_min(1e-30)
    n2 = g2f.norm(dim=-1).clamp_min(1e-30)
    return (g1f * g2f).sum(-1) / (n1 * n2)


def summarize(g_an, g_au):
    diff = g_au - g_an
    cos_pt = per_token_cosine(g_an, g_au)
    return {
        "rms_an": rms(g_an),
        "rms_au": rms(g_au),
        "rms_ratio": rms(g_au) / max(rms(g_an), 1e-30),
        "max_an": float(g_an.abs().max()),
        "max_au": float(g_au.abs().max()),
        "max_ratio": float(g_au.abs().max()) / max(float(g_an.abs().max()), 1e-30),
        "rms_diff": rms(diff),
        "rms_diff_rel": rms(diff) / max(rms(g_an), 1e-30),
        "mean_diff": float(diff.mean()),
        "cos_q05": float(torch.quantile(cos_pt, 0.05)),
        "cos_q50": float(torch.quantile(cos_pt, 0.50)),
        "cos_q95": float(torch.quantile(cos_pt, 0.95)),
        "cos_min": float(cos_pt.min()),
    }


def production_step(estep, mu, sigma, phi, mu_p, sigma_p, mask, kappa):
    from transformer.core.vfe_utils import retract_sigma_e_step
    grad_mu, grad_sigma, _ = run_analytic(
        estep, mu, sigma, phi, mu_p, sigma_p, mask, kappa,
    )
    nat_mu, nat_sigma = compute_natural_gradient_gpu(
        grad_mu, grad_sigma, sigma, eps=1e-6,
    )
    mu_new = mu - estep.e_mu_lr * nat_mu
    sigma_new = retract_sigma_e_step(
        sigma, nat_sigma,
        effective_lr=estep.e_sigma_lr,
        is_diagonal=True,
        eps=1e-6,
        update_sigma=True,
        sigma_trust=estep.e_sigma_q_trust,
        sigma_max=estep.sigma_max,
        isotropic_covariance=False,
    )
    return mu_new.detach(), sigma_new.detach()


def main():
    print("# Autograd vs Analytic (mu, sigma) Gradient Scale Probe")
    print()
    print("Repo:", _REPO_ROOT)
    print("Torch:", torch.__version__, "CUDA:", torch.cuda.is_available())

    cfg_an = make_config(use_autograd=False)
    cfg_au = make_config(use_autograd=True)

    generators = build_generators(cfg_an)
    cfg_an.generators = generators
    cfg_au.generators = generators

    print()
    print("Config: K={}, irrep_spec={!r}, n_gen={}".format(
        cfg_an.embed_dim, cfg_an.irrep_spec, generators.shape[0]))
    print("  alpha={}, lambda_align={}, lambda_soft={}, kappa={}, sigma_init={}".format(
        cfg_an.alpha, cfg_an.lambda_align, cfg_an.lambda_soft, cfg_an.kappa, cfg_an.sigma_init))
    print("  include_attention_entropy={}, E_learnable_alpha={}, alpha_divergence={}".format(
        cfg_an.include_attention_entropy, cfg_an.E_learnable_alpha, cfg_an.alpha_divergence))
    print()

    estep_an = VFEEStep(cfg_an, generators)
    estep_au = VFEEStep(cfg_au, generators)

    torch.manual_seed(0)
    B, N, K = 4, 16, cfg_an.embed_dim
    mu_p = (torch.randn(B, N, K) * 0.5).detach()
    sigma_p = torch.full((B, N, K), cfg_an.sigma_init).mul_(
        torch.exp(0.1 * torch.randn(B, N, K))
    ).detach()

    mu = (mu_p.clone() + 0.1 * torch.randn_like(mu_p)).detach()
    sigma = sigma_p.clone().detach()
    phi = (cfg_an.phi_scale * torch.randn(B, N, generators.shape[0])).detach()

    mask = build_causal_mask(B, N)
    kappa = cfg_an.kappa

    rows = []
    for iter_idx in range(0, 6):
        grad_mu_an, grad_sigma_an, beta = run_analytic(
            estep_an, mu, sigma, phi, mu_p, sigma_p, mask, kappa,
        )
        grad_mu_au, grad_sigma_au = run_autograd(
            estep_au, mu, sigma, phi, mu_p, sigma_p, mask, kappa,
        )
        with torch.no_grad():
            row = beta[0, N // 2].clamp_min(1e-30)
            beta_entropy = -(row * row.log()).sum().item()
            beta_max = float(row.max())
        nm_an, ns_an = compute_natural_gradient_gpu(grad_mu_an, grad_sigma_an, sigma, eps=1e-6)
        nm_au, ns_au = compute_natural_gradient_gpu(grad_mu_au, grad_sigma_au, sigma, eps=1e-6)
        rows.append({
            "iter": iter_idx,
            "beta_H": beta_entropy,
            "beta_max": beta_max,
            "mu": summarize(grad_mu_an, grad_mu_au),
            "sigma": summarize(grad_sigma_an, grad_sigma_au),
            "nat_mu_an": nm_an, "nat_mu_au": nm_au,
            "nat_sigma_an": ns_an, "nat_sigma_au": ns_au,
        })
        mu, sigma = production_step(
            estep_an, mu, sigma, phi, mu_p, sigma_p, mask, kappa,
        )

    print("## beta concentration trajectory (row N/2, batch 0)")
    print()
    print("| iter | H(beta_row) | max(beta_row) |")
    print("|-----:|------------:|--------------:|")
    for r in rows:
        print("| {} | {:.4f} | {:.4f} |".format(r["iter"], r["beta_H"], r["beta_max"]))
    print()

    def emit(key, label):
        print("## {} gradient (autograd vs analytic)".format(label))
        print()
        print("| iter | rms(an) | rms(au) | rms ratio | max ratio "
              "| rms(diff) | rms(diff)/rms(an) | mean(diff) "
              "| cos q05 | cos q50 | cos q95 | cos min |")
        print("|-----:|--------:|--------:|----------:|----------:"
              "|----------:|------------------:|-----------:"
              "|--------:|--------:|--------:|--------:|")
        for r in rows:
            s = r[key]
            print(("| {} | {:.3e} | {:.3e} | {:.3f} | {:.3f} | {:.3e} | {:.3f} "
                  "| {:+.3e} | {:+.4f} | {:+.4f} | {:+.4f} | {:+.4f} |").format(
                r["iter"], s["rms_an"], s["rms_au"],
                s["rms_ratio"], s["max_ratio"],
                s["rms_diff"], s["rms_diff_rel"], s["mean_diff"],
                s["cos_q05"], s["cos_q50"], s["cos_q95"], s["cos_min"],
            ))
        print()

    emit("mu", "mu")
    emit("sigma", "sigma")

    print("## Natural-gradient (actual update direction) RMS trajectory")
    print()
    print("| iter | nat_mu rms(an) | nat_mu rms(au) | ratio "
          "| nat_sig rms(an) | nat_sig rms(au) | ratio |")
    print("|-----:|---------------:|---------------:|------:"
          "|----------------:|----------------:|------:|")
    for r in rows:
        nm_an = rms(r["nat_mu_an"])
        nm_au = rms(r["nat_mu_au"])
        ns_an = rms(r["nat_sigma_an"])
        ns_au = rms(r["nat_sigma_au"])
        print("| {} | {:.3e} | {:.3e} | {:.3f} | {:.3e} | {:.3e} | {:.3f} |".format(
            r["iter"], nm_an, nm_au, nm_au / max(nm_an, 1e-30),
            ns_an, ns_au, ns_au / max(ns_an, 1e-30),
        ))
    print()

    print("## Effective update step magnitude (LR * RMS of natural grad)")
    print()
    e_mu_lr = float(estep_an.e_mu_lr)
    e_sigma_lr = float(estep_an.e_sigma_lr)
    print("e_mu_lr={}, e_sigma_lr={}".format(e_mu_lr, e_sigma_lr))
    print()
    print("| iter | dmu step (an) | dmu step (au) | dsig step (an) | dsig step (au) |")
    print("|-----:|--------------:|--------------:|---------------:|---------------:|")
    for r in rows:
        nm_an = e_mu_lr * rms(r["nat_mu_an"])
        nm_au = e_mu_lr * rms(r["nat_mu_au"])
        ns_an = e_sigma_lr * rms(r["nat_sigma_an"])
        ns_au = e_sigma_lr * rms(r["nat_sigma_au"])
        print("| {} | {:.3e} | {:.3e} | {:.3e} | {:.3e} |".format(
            r["iter"], nm_an, nm_au, ns_an, ns_au,
        ))
    print()


if __name__ == "__main__":
    main()
