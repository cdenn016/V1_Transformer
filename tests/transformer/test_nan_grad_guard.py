"""
Tests for RiemannianAdamW NaN-gradient guard
============================================

Validates that ``RiemannianAdamW.step()`` skips the AdamW update
(and therefore does not corrupt the ``exp_avg`` / ``exp_avg_sq`` state)
when any parameter has a non-finite gradient.
"""

import pytest
import torch
import torch.nn as nn

from transformer.training.optimizer import RiemannianAdamW


def test_nan_grad_skips_step():
    """Inject NaN grad on one parameter; optimizer skips step and records."""
    torch.manual_seed(0)
    p_ok = nn.Parameter(torch.randn(4))
    p_bad = nn.Parameter(torch.randn(4))

    opt = RiemannianAdamW(
        [
            {'params': [p_ok], 'name': 'attention'},
            {'params': [p_bad], 'name': 'attention'},
        ],
        lr=0.01,
        grad_clip=0.0,
    )

    # Snapshot initial parameter values
    ok_before = p_ok.detach().clone()
    bad_before = p_bad.detach().clone()

    # Assign a finite grad to p_ok and a NaN grad to p_bad
    p_ok.grad = torch.randn(4)
    p_bad.grad = torch.full((4,), float('nan'))

    opt.step()

    # Parameters unchanged (step was skipped).
    assert torch.allclose(p_ok, ok_before)
    assert torch.allclose(p_bad, bad_before)
    # Telemetry updated.
    assert opt._nan_step_count == 1
    assert opt._last_nan_param is not None
    assert 'attention' in opt._last_nan_param
    # AdamW internal state: no exp_avg entry yet (step never ran).
    assert len(opt.state.get(p_ok, {})) == 0
    assert len(opt.state.get(p_bad, {})) == 0


def test_finite_grad_runs_step():
    """With finite grads, step runs normally."""
    torch.manual_seed(1)
    p = nn.Parameter(torch.randn(4))

    opt = RiemannianAdamW(
        [{'params': [p], 'name': 'attention'}],
        lr=0.01,
        grad_clip=0.0,
    )
    p_before = p.detach().clone()
    p.grad = torch.randn(4)
    opt.step()

    # Parameter updated by one Adam step.
    assert not torch.allclose(p, p_before)
    assert opt._nan_step_count == 0
    # AdamW state populated.
    assert 'exp_avg' in opt.state[p]
    assert 'exp_avg_sq' in opt.state[p]


def test_inf_grad_skips_step():
    """Inf gradients are treated the same as NaN gradients."""
    p = nn.Parameter(torch.randn(4))
    opt = RiemannianAdamW(
        [{'params': [p], 'name': 'attention'}],
        lr=0.01,
        grad_clip=0.0,
    )
    p_before = p.detach().clone()
    p.grad = torch.full((4,), float('inf'))
    opt.step()
    assert torch.allclose(p, p_before)
    assert opt._nan_step_count == 1


def test_nan_step_does_not_corrupt_subsequent_finite_step():
    """Critical: after a NaN-triggered skip, the next finite step must
    produce a clean Adam update (no residual NaN in exp_avg / exp_avg_sq)."""
    torch.manual_seed(2)
    p = nn.Parameter(torch.randn(4))
    opt = RiemannianAdamW(
        [{'params': [p], 'name': 'attention'}],
        lr=0.01,
        grad_clip=0.0,
    )

    # Step 1: NaN grad -> skip
    p.grad = torch.full((4,), float('nan'))
    opt.step()
    assert opt._nan_step_count == 1

    # Step 2: finite grad -> should produce finite parameters
    p.grad = torch.randn(4)
    opt.step()
    assert torch.isfinite(p).all()
    assert torch.isfinite(opt.state[p]['exp_avg']).all()
    assert torch.isfinite(opt.state[p]['exp_avg_sq']).all()
