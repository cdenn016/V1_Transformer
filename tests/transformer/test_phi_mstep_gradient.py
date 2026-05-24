"""Regression test: phi_embed / pos_phi must receive an M-step gradient.

In the vfe/ package, phi_embed learns by backprop-through-the-E-step: the
analytic VFE-gradient kernels write dF/dmu, dF/dsigma as differentiable
functions of phi (through Omega = exp(phi.G)), so the retracted beliefs carry
phi and the outer CE loss backprops to phi_embed.

The rope_full_gauge != 'off' branch routes through
``_compute_rope_full_gauge_gradient_per_head`` (core/vfe_gradients.py), which
historically detached phi and returned detached grads — silently severing the
M-step gradient to phi_embed (and pos_phi_free), freezing them at init for the
entire run. This test pins the connectivity: phi_embed and pos_phi_free must
have a non-None, nonzero gradient after one forward+backward, across the
covariance x rope_full_gauge matrix.
"""
from __future__ import annotations

import pytest
import torch

from transformer.vfe.config import VFEConfig
from transformer.vfe.model import VFEModel


def _build(**overrides):
    cfg_kwargs = dict(
        vocab_size=64,
        embed_dim=4,
        irrep_spec=[("fund", 2, 2)],   # 2 heads, d_h=2 (even -> RoPE-valid)
        n_layers=1,
        n_e_steps=1,
        max_seq_len=16,
        e_phi_lr=0.0,                  # mirror the reported run: phi frozen in E-step
        mass_phi=0.0,                  # no direct phi regularizer
        gauge_fixed_priors=False,      # direct per-token priors: decode does NOT use phi
        use_prior_bank=True,
        per_head_softmax=True,
    )
    cfg_kwargs.update(overrides)
    return VFEModel(VFEConfig(**cfg_kwargs))


def _phi_grad_norms(model: VFEModel) -> tuple[float | None, float | None]:
    torch.manual_seed(0)
    ids = torch.randint(0, model.cfg.vocab_size, (2, 8))
    tgt = torch.randint(0, model.cfg.vocab_size, (2, 8))
    _, loss, _ = model(ids, tgt)
    model.zero_grad(set_to_none=True)
    loss.backward()
    pe = model.prior_bank.phi_embed.weight.grad
    pp = model.pos_enc.pos_phi_free.grad
    return (None if pe is None else float(pe.norm()),
            None if pp is None else float(pp.norm()))


@pytest.mark.parametrize(
    "label,overrides",
    [
        ("full_cov_rope_both", dict(diagonal_covariance=False, use_rope=True, rope_full_gauge="both")),
        ("full_cov_no_rope", dict(diagonal_covariance=False, use_rope=False)),
        ("diagonal_rope_off", dict(diagonal_covariance=True, use_rope=True, rope_full_gauge="off")),
    ],
)
def test_phi_embed_receives_mstep_gradient(label, overrides):
    model = _build(**overrides)
    model.train()
    pe_norm, pp_norm = _phi_grad_norms(model)
    assert pe_norm is not None, f"[{label}] phi_embed.grad is None — M-step path severed"
    assert pe_norm > 0.0, f"[{label}] phi_embed.grad is exactly zero — M-step path severed"
    assert pp_norm is not None, f"[{label}] pos_phi_free.grad is None — M-step path severed"
    assert pp_norm > 0.0, f"[{label}] pos_phi_free.grad is exactly zero — M-step path severed"
