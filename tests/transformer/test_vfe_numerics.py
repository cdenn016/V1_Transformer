"""Smoke tests for the opt-in numerical-conditioning helpers added 2026-05-19.

Covers the four config flags from VFEConfig:
* ``e_nan_check``  (NaN/Inf sentinel)
* ``e_mu_q_trust`` (mu trust region)
* ``phi_spec_max`` (pre-exp Frobenius clamp)
* ``phi_strict_glplus`` (negative-det Omega abort mode)

Each test exercises a single helper or call path. The full E-step is not
re-tested here — existing suites cover the integrated path; these tests
verify the helpers in isolation and confirm the pure path is bitwise
unchanged when the flags are at their defaults.
"""

from __future__ import annotations

import pytest
import torch

from transformer.vfe._numerics import (
    VFENonFiniteError,
    apply_mu_trust_region,
    check_finite,
    pre_exp_frobenius_clamp,
)
from transformer.vfe.omega_direct import (
    VFEGaugeOrientationError,
    project_omega_to_slk,
)


# ---------------------------------------------------------------------------
# Item 1 — check_finite
# ---------------------------------------------------------------------------


def test_check_finite_off_is_noop():
    """``mode='off'`` returns True without inspecting tensors."""
    nan_t = torch.tensor([float('nan')])
    assert check_finite({'x': nan_t}, mode='off', step_label='test') is True


def test_check_finite_warn_returns_false_on_nan():
    """``mode='warn'`` logs and returns False; does not raise."""
    nan_t = torch.tensor([1.0, float('nan'), 3.0])
    with pytest.warns(RuntimeWarning):
        result = check_finite({'x': nan_t}, mode='warn', step_label='test')
    assert result is False


def test_check_finite_abort_raises_on_inf():
    """``mode='abort'`` raises VFENonFiniteError on +inf."""
    inf_t = torch.tensor([1.0, float('inf'), 3.0])
    with pytest.raises(VFENonFiniteError) as excinfo:
        check_finite({'phi': inf_t}, mode='abort', step_label='iter_2')
    assert excinfo.value.step_label == 'iter_2'
    assert excinfo.value.field == 'phi'


def test_check_finite_clean_input_returns_true():
    """No-NaN inputs return True under every non-off mode."""
    t = torch.randn(3, 4)
    for mode in ('warn', 'revert', 'abort'):
        assert check_finite({'x': t}, mode=mode, step_label='clean') is True


# ---------------------------------------------------------------------------
# Item 2 — apply_mu_trust_region
# ---------------------------------------------------------------------------


def test_mu_trust_region_clamps_large_delta_diagonal():
    """A delta_mu 10x larger than sigma should be clamped to trust*sigma."""
    sigma = torch.full((2, 4, 8), 0.25)
    sqrt_sigma = sigma.sqrt()
    # delta_mu / sqrt(sigma) = 10 ⇒ should clamp to trust = 3.0
    delta_mu = 10.0 * sqrt_sigma
    out = apply_mu_trust_region(delta_mu, sigma, trust=3.0, is_diagonal=True)
    whitened = out / sqrt_sigma
    assert torch.all(whitened.abs() <= 3.0 + 1e-6)
    # And it should saturate at exactly the trust value.
    assert torch.allclose(whitened, torch.full_like(whitened, 3.0), atol=1e-6)


def test_mu_trust_region_passes_small_delta_unchanged():
    """When the clamp does not bind, delta_mu is returned unchanged."""
    sigma = torch.full((1, 1, 4), 1.0)
    delta_mu = torch.tensor([[[0.1, -0.2, 0.05, -0.3]]])
    out = apply_mu_trust_region(delta_mu, sigma, trust=5.0, is_diagonal=True)
    assert torch.allclose(out, delta_mu)


def test_mu_trust_region_full_cov_uses_diagonal_whitening():
    """For (..., K, K) sigma, only the diagonal is used for whitening."""
    K = 4
    sigma_diag = torch.tensor([0.25, 1.0, 4.0, 0.16])
    sigma_full = torch.diag(sigma_diag).expand(2, 3, K, K).clone()
    # Inject off-diagonal noise that should be IGNORED.
    sigma_full[..., 0, 1] = 0.5
    sigma_full[..., 1, 0] = 0.5
    delta_mu = 10.0 * sigma_diag.sqrt().expand(2, 3, K)
    out_full = apply_mu_trust_region(delta_mu, sigma_full, trust=2.0, is_diagonal=False)
    out_diag = apply_mu_trust_region(
        delta_mu, sigma_diag.expand(2, 3, K), trust=2.0, is_diagonal=True,
    )
    assert torch.allclose(out_full, out_diag)


# ---------------------------------------------------------------------------
# Item 4 — pre_exp_frobenius_clamp
# ---------------------------------------------------------------------------


def test_frobenius_clamp_binds_on_large_norm():
    """Algebra with ||X||_F = 10 should be clamped to ||X||_F = max_fro=1."""
    X = torch.eye(4).expand(2, 3, 4, 4).clone() * 5.0  # ||X||_F = 10
    out, scale = pre_exp_frobenius_clamp(X, max_fro=1.0)
    fro = out.norm(p='fro', dim=(-2, -1))
    assert torch.all(fro <= 1.0 + 1e-5)
    assert torch.all(scale < 1.0)


def test_frobenius_clamp_passthrough_below_bound():
    """Small-norm algebra returns unchanged (scale=1)."""
    X = torch.randn(2, 3, 4, 4) * 0.01
    out, scale = pre_exp_frobenius_clamp(X, max_fro=10.0)
    assert torch.allclose(out, X)
    assert torch.allclose(scale, torch.ones_like(scale))


# ---------------------------------------------------------------------------
# Item 5 — phi_strict_glplus / project_omega_to_slk strict mode
# ---------------------------------------------------------------------------


def test_project_omega_to_slk_default_preserves_sign():
    """Default ``strict=False`` preserves the sign factor (current contract)."""
    torch.manual_seed(0)
    d = 4
    O = torch.eye(d).expand(2, 3, d, d).clone()
    # Negate one row of one block to flip the sign of det.
    O[0, 1, 0, :] = -O[0, 1, 0, :]
    Oi = torch.linalg.inv(O)
    op = project_omega_to_slk([(O, Oi)], [d], strict=False)
    O_new, _ = op[0]
    det = torch.linalg.det(O_new.float())
    # |det| ≈ 1 everywhere (the rescale).
    assert torch.allclose(det.abs(), torch.ones_like(det), atol=1e-4)
    # And the sign of the originally-negative block is preserved.
    assert det[0, 1] < 0


def test_project_omega_to_slk_strict_raises_on_negative_det():
    """``strict=True`` raises VFEGaugeOrientationError on det<0 blocks."""
    d = 4
    O = torch.eye(d).expand(2, 3, d, d).clone()
    O[0, 0, 0, :] = -O[0, 0, 0, :]
    Oi = torch.linalg.inv(O)
    with pytest.raises(VFEGaugeOrientationError) as excinfo:
        project_omega_to_slk([(O, Oi)], [d], strict=True)
    assert excinfo.value.n_negative_blocks == 1


def test_project_omega_to_slk_strict_passes_positive_det():
    """``strict=True`` is a no-op on all-positive-det inputs."""
    torch.manual_seed(0)
    d = 4
    O = torch.eye(d).expand(2, 3, d, d).clone() + 0.05 * torch.randn(2, 3, d, d)
    Oi = torch.linalg.inv(O)
    op = project_omega_to_slk([(O, Oi)], [d], strict=True)
    O_new, _ = op[0]
    det = torch.linalg.det(O_new.float())
    assert torch.allclose(det.abs(), torch.ones_like(det), atol=1e-4)
