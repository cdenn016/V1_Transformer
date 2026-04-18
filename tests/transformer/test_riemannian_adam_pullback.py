"""Tests for RiemannianAdamW metric='pullback' option.

Covers CS-5 invariants:
  1. At φ=0, pullback reduces to Gram inverse.
  2. For non-zero φ, pullback ≠ Killing.
  3. Per-token clipping under pullback uses sqrt(ξᵀ G(φ) ξ) per row.
  4. metric='killing' default preserves legacy behavior bit-exactly.
  5. Training smoke: 3 steps, loss finite.
"""

import pytest
import torch
import torch.nn as nn

from transformer.core.gauge_preconditioner import (
    build_killing_form_preconditioner,
    build_pullback_metric_tensor,
    build_structure_constants,
)
from transformer.training.optimizer import RiemannianAdamW


def _so3_generators() -> torch.Tensor:
    g = torch.zeros(3, 3, 3)
    # Standard so(3) basis with [L_a, L_b] = ε_{abc} L_c
    g[0, 1, 2] = 1.0; g[0, 2, 1] = -1.0
    g[1, 0, 2] = -1.0; g[1, 2, 0] = 1.0
    g[2, 0, 1] = 1.0; g[2, 1, 0] = -1.0
    return g


def _glk_generators(K: int = 4) -> torch.Tensor:
    """Elementary E_ij basis of gl(K), n_gen = K²."""
    gens = []
    for i in range(K):
        for j in range(K):
            G = torch.zeros(K, K)
            G[i, j] = 1.0
            gens.append(G)
    return torch.stack(gens)


def test_pullback_at_zero_phi_equals_gram_inverse():
    """At φ=0, Ψ(ad_0) = I, so G(0) = gram → nat_grad = grad @ gram^{-1}."""
    so3 = _so3_generators()
    struct = build_structure_constants(so3)
    gram = torch.einsum('aij,bij->ab', so3, so3)

    phi = nn.Parameter(torch.zeros(1, 3))
    phi.grad = torch.tensor([[1.0, 0.5, -0.3]])
    grad_before = phi.grad.clone()

    opt = RiemannianAdamW(
        [{'params': [phi], 'name': 'phi_embed'}],
        lr=0.0,  # prevent any actual update — we inspect grad
        metric='pullback',
        generators=so3,
        structure_constants=struct,
        gram=gram,
        grad_clip=0.0,
    )
    # Call _precondition_phi directly to inspect the preconditioned grad
    opt._precondition_phi(opt.param_groups[0])

    expected = grad_before @ torch.linalg.inv(gram + 1e-6 * torch.eye(3))
    torch.testing.assert_close(phi.grad, expected, atol=1e-4, rtol=1e-4)


def test_pullback_differs_from_killing_at_nonzero_phi():
    """For non-zero φ, pullback and Killing give different preconditioned grads."""
    so3 = _so3_generators()
    struct = build_structure_constants(so3)
    K_inv, K_metric = build_killing_form_preconditioner(so3, return_both=True)

    torch.manual_seed(0)
    phi_val = torch.randn(1, 3) * 0.5  # non-trivial φ
    g = torch.randn(1, 3)

    # Killing path
    phi_k = nn.Parameter(phi_val.clone())
    phi_k.grad = g.clone()
    opt_k = RiemannianAdamW(
        [{'params': [phi_k], 'name': 'phi_embed'}],
        lr=0.0, metric='killing', killing_inv=K_inv, killing_metric=K_metric,
    )
    opt_k._precondition_phi(opt_k.param_groups[0])

    # Pullback path
    phi_p = nn.Parameter(phi_val.clone())
    phi_p.grad = g.clone()
    opt_p = RiemannianAdamW(
        [{'params': [phi_p], 'name': 'phi_embed'}],
        lr=0.0, metric='pullback', generators=so3, structure_constants=struct,
    )
    opt_p._precondition_phi(opt_p.param_groups[0])

    # Should differ
    diff = (phi_k.grad - phi_p.grad).abs().max().item()
    assert diff > 1e-3, (
        f"Pullback and Killing gave near-identical results (diff={diff}); "
        f"expected difference at non-zero φ."
    )


def test_killing_default_preserves_legacy_behavior():
    """metric='killing' (default) produces bit-identical output to the pre-refactor path."""
    so3 = _so3_generators()
    K_inv, K_metric = build_killing_form_preconditioner(so3, return_both=True)

    torch.manual_seed(1)
    phi = nn.Parameter(torch.randn(5, 3) * 0.1)
    g = torch.randn_like(phi)
    phi.grad = g.clone()

    opt = RiemannianAdamW(
        [{'params': [phi], 'name': 'phi_embed'}],
        lr=0.0,
        killing_inv=K_inv,
        killing_metric=K_metric,
        # metric defaults to 'killing'
    )
    opt._precondition_phi(opt.param_groups[0])

    expected = g @ K_inv
    torch.testing.assert_close(phi.grad, expected, atol=1e-6, rtol=1e-6)


def test_pullback_per_token_clip_uses_riemannian_norm():
    """Per-token clipping under pullback scales by sqrt(ξᵀ G(φ) ξ) per row."""
    so3 = _so3_generators()
    struct = build_structure_constants(so3)
    gram = torch.einsum('aij,bij->ab', so3, so3)

    # Two tokens with very different φ — clip should behave differently per row.
    phi_data = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    phi = nn.Parameter(phi_data.clone())
    # Large gradient on token 1 (big φ), small on token 0 (zero φ).
    phi.grad = torch.tensor([[10.0, 0.0, 0.0], [10.0, 0.0, 0.0]])

    opt = RiemannianAdamW(
        [{'params': [phi], 'name': 'phi_embed'}],
        lr=0.0,
        metric='pullback',
        generators=so3,
        structure_constants=struct,
        gram=gram,
        grad_clip=1.0,
    )
    # Skip precondition (we're testing the clip path directly with raw ξ).
    opt._riemannian_clip()

    # Compute analytical expected clipped grad per token.
    G_phi = build_pullback_metric_tensor(phi_data, so3, struct, gram)  # (2, 3, 3)
    xi_orig = torch.tensor([[10.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    norm_sq = torch.einsum('ti,tij,tj->t', xi_orig, G_phi, xi_orig)
    norm = norm_sq.clamp(min=0.0).sqrt()
    scale = (1.0 / (norm + 1e-8)).clamp(max=1.0)
    expected = xi_orig * scale.unsqueeze(-1)
    torch.testing.assert_close(phi.grad, expected, atol=1e-5, rtol=1e-5)


def test_pullback_requires_generators():
    """metric='pullback' without generators raises ValueError."""
    phi = nn.Parameter(torch.zeros(3))
    with pytest.raises(ValueError, match="generators"):
        RiemannianAdamW(
            [{'params': [phi], 'name': 'phi_embed'}],
            lr=1e-3,
            metric='pullback',
        )


def test_invalid_metric_raises():
    """Unknown metric value raises at construction."""
    phi = nn.Parameter(torch.zeros(3))
    with pytest.raises(ValueError, match="metric must be"):
        RiemannianAdamW(
            [{'params': [phi], 'name': 'phi_embed'}],
            lr=1e-3,
            metric='bogus',
            generators=_so3_generators(),
        )


def test_pullback_training_smoke_3_steps():
    """A small 3-step loop under metric='pullback' runs to completion without NaN."""
    so3 = _so3_generators()
    struct = build_structure_constants(so3)
    gram = torch.einsum('aij,bij->ab', so3, so3)

    phi = nn.Parameter(torch.randn(5, 3) * 0.1)
    opt = RiemannianAdamW(
        [{'params': [phi], 'name': 'phi_embed'}],
        lr=1e-2,
        metric='pullback',
        generators=so3,
        structure_constants=struct,
        gram=gram,
        grad_clip=1.0,
    )
    for step in range(3):
        opt.zero_grad()
        loss = (phi ** 2).sum()
        loss.backward()
        opt.step()
        assert torch.isfinite(phi).all(), f"Non-finite at step {step}"
        assert torch.isfinite(phi.grad if phi.grad is not None else torch.tensor(0.0)).all()


def test_pullback_glk_4d():
    """Pullback works on gl(4) = 16 generators, verifies shape invariants."""
    gens = _glk_generators(K=4)
    struct = build_structure_constants(gens)

    V = 7
    phi = nn.Parameter(torch.randn(V, 16) * 0.05)
    phi.grad = torch.randn_like(phi)
    opt = RiemannianAdamW(
        [{'params': [phi], 'name': 'phi_embed'}],
        lr=0.0,
        metric='pullback',
        generators=gens,
        structure_constants=struct,
    )
    opt._precondition_phi(opt.param_groups[0])
    assert phi.grad.shape == (V, 16)
    assert torch.isfinite(phi.grad).all()
