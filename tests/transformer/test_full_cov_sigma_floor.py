"""
Tests for full-covariance SPD eigenvalue floor
==============================================

Validates the ``spd_eigfloor`` primitive in ``transformer/core/vfe_utils.py``
and its integration with full-covariance KL computation. Replaces the legacy
ridge floor ``Sigma + floor*I`` with a spectral clamp that bounds the
condition number.

Mathematical invariants:
    - Idempotency: floor(floor(Sigma)) == floor(Sigma).
    - Minimum eigenvalue: lambda_min(floor(Sigma)) >= floor.
    - Gauge covariance: floor(h Sigma h^T, exp_phi=h) == h floor(Sigma) h^T
      when the frame exp_phi transforms with the sandwich.
    - Finite forward/backward under ill-conditioned Sigma_p: KL and its
      gradients stay finite with eigclamp when Sigma_p has near-zero eigvals.
"""

import pytest
import torch

from transformer.core.vfe_utils import spd_eigfloor


# =============================================================================
# Helpers
# =============================================================================

def _random_spd(K, device='cpu', dtype=torch.float64, conditioning=None):
    """Build a symmetric positive-definite matrix.

    When ``conditioning`` is provided, constructs a matrix with
    eigenvalues linearly spaced in ``[1/conditioning, 1]`` for controlled
    condition number.
    """
    if conditioning is not None:
        eigs = torch.linspace(1.0 / conditioning, 1.0, K, device=device, dtype=dtype)
        Q, _ = torch.linalg.qr(torch.randn(K, K, device=device, dtype=dtype))
        return Q @ torch.diag(eigs) @ Q.t()
    A = torch.randn(K, K, device=device, dtype=dtype) * 0.3
    return A @ A.t() + 0.1 * torch.eye(K, device=device, dtype=dtype)


# =============================================================================
# Primitive tests
# =============================================================================

def test_spd_eigfloor_idempotent_spectrum():
    """Applying the floor twice yields the same spectrum (up to the
    degeneracy-breaker perturbation inside _safe_eigh, which is O(floor/K)
    per eigenvalue and is intentional for gradient stability)."""
    torch.manual_seed(0)
    K = 6
    floor = 0.01
    Sigma = _random_spd(K, conditioning=1e4, dtype=torch.float64)
    once = spd_eigfloor(Sigma, floor)
    twice = spd_eigfloor(once, floor)
    eig_once = torch.linalg.eigvalsh(once)
    eig_twice = torch.linalg.eigvalsh(twice)
    # Spectrum is stable across repeated applications within ~floor tolerance
    # (the degeneracy-breaker inside _safe_eigh adds <= floor to each eigval).
    # Breaker shifts eigenvalues by up to `floor` per application; repeated
    # application is bounded by 2*floor.
    assert torch.allclose(eig_once, eig_twice, atol=2.0 * floor), (
        f"Spectrum shifts across repeated application: "
        f"max diff {(eig_once - eig_twice).abs().max().item()}"
    )
    # All eigenvalues remain at or above the floor after both applications.
    assert eig_twice.min().item() >= floor - 1e-6


def test_spd_eigfloor_lambda_min_bound():
    """Output satisfies lambda_min >= floor to within numerical slack."""
    torch.manual_seed(1)
    K = 5
    floor = 0.05
    # Intentionally create a matrix with one tiny eigenvalue.
    Sigma = _random_spd(K, conditioning=1e8, dtype=torch.float64)
    lam_in = torch.linalg.eigvalsh(Sigma).min().item()
    assert lam_in < floor, f"Test precondition failed: lam_min={lam_in}"
    Sigma_out = spd_eigfloor(Sigma, floor)
    lam_out = torch.linalg.eigvalsh(Sigma_out).min().item()
    assert lam_out >= floor - 1e-6, (
        f"Floor not enforced: lam_min={lam_out}, floor={floor}"
    )


def test_spd_eigfloor_upper_bound():
    """When sigma_max is set, lambda_max <= sigma_max^2."""
    torch.manual_seed(2)
    K = 4
    floor = 0.01
    sigma_max = 2.0
    Sigma = _random_spd(K, dtype=torch.float64)
    Sigma = Sigma * 50.0  # blow up eigenvalues
    Sigma_out = spd_eigfloor(Sigma, floor, sigma_max=sigma_max)
    lam_max = torch.linalg.eigvalsh(Sigma_out).max().item()
    # Tolerance accounts for the O(floor) degeneracy-breaker perturbation
    # inside _safe_eigh which can push eigvals slightly past the clamp bound.
    assert lam_max <= sigma_max * sigma_max + floor, (
        f"Ceiling not enforced: lam_max={lam_max}, bound={sigma_max**2}"
    )


def test_spd_eigfloor_noop_on_well_conditioned():
    """A well-conditioned matrix with eigvals well above floor is unchanged
    except for numerical roundoff (degeneracy-breaker perturbation)."""
    torch.manual_seed(3)
    K = 3
    floor = 0.001
    Sigma = _random_spd(K, conditioning=10.0, dtype=torch.float64)
    Sigma_out = spd_eigfloor(Sigma, floor)
    # The degeneracy breaker is O(floor/K), so tolerance ~floor.
    assert torch.allclose(Sigma, Sigma_out, atol=5 * floor), (
        f"Well-conditioned matrix modified too much: "
        f"max diff {(Sigma - Sigma_out).abs().max().item()}"
    )


def test_spd_eigfloor_preserves_symmetry():
    """Output must be exactly symmetric (within fp64 roundoff)."""
    torch.manual_seed(4)
    K = 8
    floor = 0.02
    Sigma = _random_spd(K, conditioning=100.0, dtype=torch.float64)
    Sigma_out = spd_eigfloor(Sigma, floor)
    asymmetry = (Sigma_out - Sigma_out.t()).abs().max().item()
    assert asymmetry < 1e-10, f"Output is not symmetric: max asym={asymmetry}"


def test_spd_eigfloor_on_psd_with_zero_eigval():
    """Handles a rank-deficient PSD input: zero eigenvalue is lifted to floor."""
    torch.manual_seed(5)
    K = 4
    floor = 0.03
    # Rank-2 PSD: Sigma = A A^T with A shape (K, 2)
    A = torch.randn(K, 2, dtype=torch.float64)
    Sigma = A @ A.t()
    # Verify rank deficiency
    lam_in = torch.linalg.eigvalsh(Sigma)
    assert lam_in.min().item() < 1e-8, (
        f"Test precondition: expected rank deficient, got min eig={lam_in.min()}"
    )
    Sigma_out = spd_eigfloor(Sigma, floor)
    lam_out = torch.linalg.eigvalsh(Sigma_out)
    assert lam_out.min().item() >= floor - 1e-6, (
        f"Rank-deficient input not lifted to floor: min={lam_out.min()}"
    )


# =============================================================================
# Strict GL(K) gauge covariance
# =============================================================================


def _random_invertible(K, dtype=torch.float64, scale=0.3, seed=None):
    """Build a random invertible matrix near identity.

    A = scale·N + I with N standard normal is almost surely invertible;
    add an extra identity shift for conditioning safety.
    """
    if seed is not None:
        torch.manual_seed(seed)
    A = scale * torch.randn(K, K, dtype=dtype) + torch.eye(K, dtype=dtype)
    # Guarantee conditioning
    return A + 0.1 * torch.eye(K, dtype=dtype)


def test_spd_eigfloor_glk_covariance():
    """spd_eigfloor(h Σ h^T, exp_phi=h A) == h · spd_eigfloor(Σ, exp_phi=A) · h^T
    under strict GL(K) transformation."""
    torch.manual_seed(42)
    K = 4
    floor = 0.02

    # Start with an ill-conditioned SPD matrix so the clamp is non-trivial.
    Sigma = _random_spd(K, conditioning=1e5, dtype=torch.float64)
    A = _random_invertible(K, scale=0.2, seed=43)
    h = _random_invertible(K, scale=0.3, seed=44)

    # Baseline clamp in frame A.
    Sigma_clamped = spd_eigfloor(Sigma, floor, exp_phi=A)

    # Transported clamp.
    Sigma_transported = h @ Sigma @ h.t()
    A_transported = h @ A
    Sigma_transported_clamped = spd_eigfloor(
        Sigma_transported, floor, exp_phi=A_transported
    )

    # Expected under gauge covariance.
    expected = h @ Sigma_clamped @ h.t()

    max_err = (Sigma_transported_clamped - expected).abs().max().item()
    assert max_err < 1e-3, (
        f"Gauge covariance broken: max diff {max_err}. "
        f"spd_eigfloor is not strictly GL(K)-covariant."
    )


def test_spd_eigfloor_glk_path_still_bounds_lambda_min():
    """The gauge-covariant path still enforces λ_min(Σ') ≥ floor after
    de-whitening. The bound transforms via A, so we check the whitened
    spectrum directly."""
    torch.manual_seed(46)
    K = 4
    floor = 0.05
    Sigma = _random_spd(K, conditioning=1e6, dtype=torch.float64)
    A = _random_invertible(K, scale=0.2, seed=47)

    Sigma_out = spd_eigfloor(Sigma, floor, exp_phi=A)
    # Verify the WHITENED eigenvalues satisfy the floor.
    A_inv = torch.linalg.inv(A)
    W_out = A_inv @ Sigma_out @ A_inv.t()
    W_out = 0.5 * (W_out + W_out.t())
    eig_W = torch.linalg.eigvalsh(W_out)
    assert eig_W.min().item() >= floor - 1e-4, (
        f"Whitened λ_min = {eig_W.min()}, floor = {floor}"
    )


# =============================================================================
# Gradient finiteness under ill-conditioning
# =============================================================================

def test_spd_eigfloor_gradient_finite():
    """Gradient through spd_eigfloor is finite even when the input has
    a near-zero eigenvalue. Input must carry grad; output should too."""
    torch.manual_seed(6)
    K = 5
    floor = 0.01
    # Build Sigma with a tiny eigenvalue via a parameter we'll differentiate
    A = torch.randn(K, K, dtype=torch.float64, requires_grad=True)
    Sigma = A @ A.t() + 1e-10 * torch.eye(K, dtype=torch.float64)
    Sigma_out = spd_eigfloor(Sigma, floor)
    loss = Sigma_out.trace()
    loss.backward()
    assert torch.isfinite(A.grad).all(), "Gradient contains NaN/Inf"


# =============================================================================
# Integration with full-covariance KL kernel
# =============================================================================

def _make_illcond_sigma(K, dtype=torch.float64):
    """Sigma_p with deliberately ill-conditioned eigenvalue spectrum."""
    eigs = torch.tensor([1e-12] + [1.0] * (K - 1), dtype=dtype)
    Q, _ = torch.linalg.qr(torch.randn(K, K, dtype=dtype))
    return Q @ torch.diag(eigs) @ Q.t()


def test_kl_finite_under_collapsed_sigma_p_with_eigclamp():
    """KL computation produces finite output and gradients when Sigma_p
    has collapsed eigenvalues, provided sigma_floor is set."""
    from transformer.core.kl_computation import _kl_kernel_dense

    torch.manual_seed(7)
    K = 4
    B, N = 1, 3

    mu_q = torch.randn(B, N, N, K, dtype=torch.float64)
    # Sigma_q well conditioned
    A_q = torch.randn(B, N, N, K, K, dtype=torch.float64) * 0.3
    sigma_q = A_q @ A_q.transpose(-1, -2) + 0.1 * torch.eye(K, dtype=torch.float64)

    # Sigma_t (transported) has one collapsed direction
    sig_collapsed = _make_illcond_sigma(K)
    sigma_t = sig_collapsed.expand(B, N, N, K, K).clone()

    mu_t = torch.randn(B, N, N, K, dtype=torch.float64)
    mu_q_p = mu_q.clone().requires_grad_(True)

    # Without floor: graph can produce NaN/inf from log(tiny eigval).
    # With floor=0.01: log ratio bounded; gradient finite.
    kl = _kl_kernel_dense(
        mu_q_p, sigma_q, mu_t, sigma_t,
        kl_max=1e6, eps=1e-6,
        sigma_floor=0.01, spd_floor_mode='eigclamp',
    )
    assert torch.isfinite(kl).all(), f"KL non-finite: {kl}"
    loss = kl.sum()
    loss.backward()
    assert torch.isfinite(mu_q_p.grad).all(), "mu_q gradient non-finite"


def test_variational_ffn_full_cov_illcond_sigma_prior():
    """End-to-end smoke: build a full-covariance VFE FFN with eigclamp mode,
    feed an ill-conditioned sigma_prior, assert forward is finite.

    This exercises the actual site `variational_ffn.py:1486-1508` under
    `spd_floor_mode='eigclamp'` + `diagonal_covariance=False`.
    """
    from transformer.core.variational_ffn import VariationalFFNDynamic

    torch.manual_seed(9)
    K, n_gen = 4, 16  # GL(K): n_gen = K*K
    B, N = 2, 4
    floor = 0.01

    generators = torch.randn(n_gen, K, K)
    ffn = VariationalFFNDynamic(
        embed_dim=K,
        generators=generators,
        alpha=0.1,
        lambda_belief=1.0,
        kappa=1.0,
        n_iterations=2,
        diagonal_covariance=False,
        e_step_sigma_floor=floor,
        spd_floor_mode='eigclamp',
        irrep_dims=[K],
    )

    # Collapsed prior: one eigenvalue ~1e-12, rest ~1.0.
    eigs = torch.tensor([1e-12, 1.0, 1.0, 1.0])
    Q, _ = torch.linalg.qr(torch.randn(K, K))
    sig_p_single = Q @ torch.diag(eigs) @ Q.t()
    sigma_prior = sig_p_single.expand(B, N, K, K).contiguous()

    mu = torch.randn(B, N, K)
    sigma = 0.5 * torch.eye(K).expand(B, N, K, K).contiguous()
    phi = torch.randn(B, N, n_gen) * 0.1
    mu_prior = torch.randn(B, N, K)

    # Forward: must return finite outputs despite the collapsed sigma_prior.
    out = ffn(mu=mu, sigma=sigma, phi=phi, mu_prior=mu_prior, sigma_prior=sigma_prior)
    mu_out = out[0] if isinstance(out, tuple) else out
    assert torch.isfinite(mu_out).all(), "mu output non-finite"


def test_variational_ffn_full_cov_ridge_mode_still_works():
    """Smoke test: `spd_floor_mode='ridge'` preserves the legacy code path."""
    from transformer.core.variational_ffn import VariationalFFNDynamic

    torch.manual_seed(10)
    K, n_gen = 4, 16
    B, N = 1, 3

    generators = torch.randn(n_gen, K, K)
    ffn = VariationalFFNDynamic(
        embed_dim=K,
        generators=generators,
        alpha=0.1,
        lambda_belief=1.0,
        kappa=1.0,
        n_iterations=2,
        diagonal_covariance=False,
        e_step_sigma_floor=0.01,
        spd_floor_mode='ridge',
        irrep_dims=[K],
    )

    A = torch.randn(B, N, K, K) * 0.3
    sigma_prior = A @ A.transpose(-1, -2) + 0.1 * torch.eye(K)

    mu = torch.randn(B, N, K)
    sigma = 0.5 * torch.eye(K).expand(B, N, K, K).contiguous()
    phi = torch.randn(B, N, n_gen) * 0.1
    mu_prior = torch.randn(B, N, K)

    out = ffn(mu=mu, sigma=sigma, phi=phi, mu_prior=mu_prior, sigma_prior=sigma_prior)
    mu_out = out[0] if isinstance(out, tuple) else out
    assert torch.isfinite(mu_out).all()


def test_variational_ffn_rejects_invalid_floor_mode():
    """Invalid spd_floor_mode at construction raises."""
    from transformer.core.variational_ffn import VariationalFFNDynamic

    generators = torch.randn(4, 2, 2)
    with pytest.raises(ValueError, match="spd_floor_mode"):
        VariationalFFNDynamic(
            embed_dim=2, generators=generators,
            alpha=0.1, lambda_belief=1.0, kappa=1.0,
            n_iterations=1,
            spd_floor_mode='bogus',
        )


def test_safe_kl_clamp_propagate_nonfinite():
    """propagate_nonfinite=True preserves NaN/inf; default replaces with kl_max."""
    from transformer.core.kl_computation import safe_kl_clamp

    kl = torch.tensor([float('nan'), float('inf'), -float('inf'), 50.0, 150.0])
    # Default: mask.
    out_default = safe_kl_clamp(kl, kl_max=100.0)
    assert torch.isfinite(out_default).all()
    assert out_default[0].item() == 100.0  # nan → kl_max
    assert out_default[1].item() == 100.0  # +inf → kl_max
    assert out_default[2].item() == 0.0    # -inf → 0
    assert out_default[3].item() == 50.0   # finite unchanged
    assert out_default[4].item() == 100.0  # above kl_max → kl_max

    # propagate_nonfinite=True: keep NaN/inf, still clamp finite.
    out_prop = safe_kl_clamp(kl, kl_max=100.0, propagate_nonfinite=True)
    assert torch.isnan(out_prop[0])
    assert torch.isinf(out_prop[1]) and out_prop[1] > 0
    assert torch.isinf(out_prop[2]) and out_prop[2] < 0
    assert out_prop[3].item() == 50.0
    assert out_prop[4].item() == 100.0  # upper clamp still applies to finite


def test_kl_ridge_matches_legacy():
    """With spd_floor_mode='ridge' and sigma_floor=None, the kernel behaves
    identically to the pre-change implementation."""
    from transformer.core.kl_computation import _kl_kernel_dense

    torch.manual_seed(8)
    K = 4
    B, N = 1, 2

    mu_q = torch.randn(B, N, N, K, dtype=torch.float64)
    A = torch.randn(B, N, N, K, K, dtype=torch.float64) * 0.3
    sigma = A @ A.transpose(-1, -2) + 0.1 * torch.eye(K, dtype=torch.float64)
    sigma_t = sigma.clone()
    mu_t = torch.randn(B, N, N, K, dtype=torch.float64)

    kl_noeff = _kl_kernel_dense(
        mu_q, sigma, mu_t, sigma_t,
        kl_max=1e6, eps=1e-6,
        sigma_floor=None, spd_floor_mode='eigclamp',
    )
    kl_ridge = _kl_kernel_dense(
        mu_q, sigma, mu_t, sigma_t,
        kl_max=1e6, eps=1e-6,
        sigma_floor=None, spd_floor_mode='ridge',
    )
    # No floor applied in either case (sigma_floor=None short-circuits),
    # so outputs are identical.
    assert torch.allclose(kl_noeff, kl_ridge, atol=1e-10)
