# -*- coding: utf-8 -*-
"""
Regression test for skip_attention + detaching-EM gradient flow.

Bug: when skip_attention=True AND em_mode is a detaching mode (em_phi_p,
em_phi_q, implicit_ift), the FFN's E-step detaches sigma_p and (for
em_phi_p) phi. With the attention sublayer also bypassed, log_sigma_diag
and phi_embed.weight receive no M-step gradient — their .grad stays None
through training.

Fix: a straight-through estimator (STE) bridge in
GaugeTransformerBlock.forward injects a "ghost" attention output under
autograd before the skip_attention bypass, restoring the M-step gradient
path while leaving the forward value bitwise unchanged.

Tests:
    (a) grad presence under em_phi_p (phi/sigma inputs receive non-None,
        non-zero gradients)
    (b) forward bitwise invariance (STE is a true no-op on forward output)
    (c) negative control under straight_through (bridge does NOT fire;
        attention is called once, not twice)
"""

import pytest
import torch

from transformer.core.block_config import BlockConfig
from transformer.core.blocks import GaugeTransformerBlock


def _make_glk_generators(K):
    """Standard E_{ij} basis for gl(K): K^2 generators, one nonzero entry each.

    Using the canonical basis (rather than random) keeps the configuration
    deterministic and matches the convention used in the gauge-utils tests.
    Random generators would also work but make the test more brittle.
    """
    G = torch.zeros(K * K, K, K)
    idx = 0
    for i in range(K):
        for j in range(K):
            G[idx, i, j] = 1.0
            idx += 1
    return G


def _make_cfg(em_mode='em_phi_p', skip_attention=True):
    # GL(K) gauge: K^2 generators, no irrep-dimension constraint, and the
    # gauge action on every head is non-trivial — phi rotation moves μ
    # through GL+(K), so dF/dphi != 0 in general. Pure SO(3) scalar irreps
    # would make phi a no-op and phi.grad would be exactly zero by symmetry,
    # masking the bug rather than testing the bridge.
    K = 8
    n_heads, d_head = 2, 4
    generators = _make_glk_generators(K)  # (K^2, K, K)
    cfg = BlockConfig(
        embed_dim=K,
        hidden_dim=2 * K,
        n_layers=1,
        kappa_beta=1.0,
        irrep_spec=[('glk', n_heads, d_head)],
        diagonal_covariance=True,
        generators=generators,
        ffn_mode='VFE_dynamic',
        ffn_irrep_dims=[d_head] * n_heads,
        em_mode=em_mode,
        skip_attention=skip_attention,
    )
    return cfg


def _make_inputs(B=2, N=4, K=8, phi_dim=None, device='cpu', seed=0):
    """Build a minimal (mu, sigma_diag, phi, mu_prior, mask) tuple with
    phi/sigma as leaf tensors carrying requires_grad so we can probe
    gradients flowing back from loss to the gauge / scale degrees of
    freedom that sigma_embed / phi_embed would normally provide."""
    if phi_dim is None:
        phi_dim = K * K  # GL(K) default
    g = torch.Generator(device=device).manual_seed(seed)
    mu = torch.randn(B, N, K, device=device, generator=g)
    sigma = torch.abs(torch.randn(B, N, K, device=device, generator=g)) + 0.1
    phi = torch.randn(B, N, phi_dim, device=device, generator=g) * 0.1
    mu_prior = mu.detach().clone()
    mask = torch.tril(torch.ones(B, N, N, device=device))
    return mu, sigma, phi, mu_prior, mask


def test_skip_attention_em_phi_p_phi_and_sigma_grads_present():
    """(a) Under skip_attention=True + em_mode='em_phi_p', phi and sigma
    inputs must receive non-None, non-zero gradients via the STE bridge."""
    torch.manual_seed(42)
    B, N, K = 2, 4, 8
    cfg = _make_cfg(em_mode='em_phi_p', skip_attention=True)
    block = GaugeTransformerBlock(cfg)
    block.train()

    mu, sigma, phi, mu_prior, mask = _make_inputs(B=B, N=N, K=K, seed=1)
    # Make phi and sigma leaf tensors with requires_grad — these stand in
    # for phi_embed.weight and log_sigma_diag in the full model. The STE
    # bridge must thread M-step gradients into these.
    phi = phi.detach().clone().requires_grad_(True)
    sigma = sigma.detach().clone().requires_grad_(True)

    mu_out, sigma_out, phi_out = block(
        mu, sigma, phi, block.generators, mask=mask, mu_prior=mu_prior
    )
    loss = mu_out.pow(2).mean()
    loss.backward()

    assert phi.grad is not None, "phi.grad is None — STE bridge failed for phi"
    assert (phi.grad != 0).any(), "phi.grad is all-zero — STE bridge produced no signal for phi"
    assert sigma.grad is not None, "sigma.grad is None — STE bridge failed for sigma"
    assert (sigma.grad != 0).any(), "sigma.grad is all-zero — STE bridge produced no signal for sigma"

    # Surface gradient magnitudes for diagnostic visibility.
    phi_grad_norm = phi.grad.norm().item()
    sigma_grad_norm = sigma.grad.norm().item()
    assert phi_grad_norm > 1e-10, f"phi.grad norm {phi_grad_norm:.3e} is below numerical floor"
    assert sigma_grad_norm > 1e-10, f"sigma.grad norm {sigma_grad_norm:.3e} is below numerical floor"


def test_skip_attention_em_phi_p_forward_bitwise_invariant():
    """(b) The STE bridge must be a true no-op on the forward output.
    Compare forward output with bridge enabled (train mode) vs disabled
    (eval mode — the bridge is gated on self.training)."""
    torch.manual_seed(123)
    B, N, K = 2, 4, 8
    cfg = _make_cfg(em_mode='em_phi_p', skip_attention=True)
    block = GaugeTransformerBlock(cfg)

    mu, sigma, phi, mu_prior, mask = _make_inputs(B=B, N=N, K=K, seed=2)

    block.eval()
    with torch.no_grad():
        mu_out_no_bridge, _, _ = block(
            mu, sigma, phi, block.generators, mask=mask, mu_prior=mu_prior
        )

    block.train()
    with torch.no_grad():
        mu_out_with_bridge, _, _ = block(
            mu, sigma, phi, block.generators, mask=mask, mu_prior=mu_prior
        )

    assert torch.equal(mu_out_no_bridge, mu_out_with_bridge), (
        "STE bridge changed forward output — "
        f"max |diff| = {(mu_out_no_bridge - mu_out_with_bridge).abs().max().item():.3e}"
    )


def test_negative_control_straight_through_does_not_double_call_attention():
    """(c) Under em_mode='straight_through' (non-detaching), the bridge
    predicate must evaluate False. We probe this by wrapping
    block.attention.forward in a call counter and confirming it is NOT
    called from the skip_attention branch (since skip_attention=True
    bypasses the real attention and no ghost should fire either)."""
    torch.manual_seed(7)
    B, N, K = 2, 4, 8
    cfg = _make_cfg(em_mode='straight_through', skip_attention=True)
    block = GaugeTransformerBlock(cfg)
    block.train()

    mu, sigma, phi, mu_prior, mask = _make_inputs(B=B, N=N, K=K, seed=3)

    call_count = {'n': 0}
    orig_forward = block.attention.forward

    def counting_forward(*args, **kwargs):
        call_count['n'] += 1
        return orig_forward(*args, **kwargs)

    block.attention.forward = counting_forward
    _ = block(
        mu, sigma, phi, block.generators, mask=mask, mu_prior=mu_prior
    )
    block.attention.forward = orig_forward

    assert call_count['n'] == 0, (
        f"Negative control failed: attention was called {call_count['n']} time(s) "
        f"under em_mode='straight_through' + skip_attention=True. The STE bridge "
        f"predicate should be False here (no detaching modes active)."
    )


def test_negative_control_em_phi_p_bridge_calls_attention_once():
    """(c-companion) Under em_mode='em_phi_p' + skip_attention=True, the
    bridge fires exactly once (the ghost call) and the real attention
    sublayer is bypassed — total attention.forward calls = 1."""
    torch.manual_seed(7)
    B, N, K = 2, 4, 8
    cfg = _make_cfg(em_mode='em_phi_p', skip_attention=True)
    block = GaugeTransformerBlock(cfg)
    block.train()

    mu, sigma, phi, mu_prior, mask = _make_inputs(B=B, N=N, K=K, seed=4)

    call_count = {'n': 0}
    orig_forward = block.attention.forward

    def counting_forward(*args, **kwargs):
        call_count['n'] += 1
        return orig_forward(*args, **kwargs)

    block.attention.forward = counting_forward
    _ = block(
        mu, sigma, phi, block.generators, mask=mask, mu_prior=mu_prior
    )
    block.attention.forward = orig_forward

    assert call_count['n'] == 1, (
        f"Bridge call count mismatch: expected 1 (the ghost) under em_phi_p + "
        f"skip_attention, got {call_count['n']}."
    )
