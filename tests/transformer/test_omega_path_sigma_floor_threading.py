"""
Phase 2 regression test: verify that _compute_omega_grad_direct threads
sigma_floor, spd_floor_mode, and propagate_nonfinite through every
compute_attention_weights call it issues.

Before Phase 2 (TODO at variational_ffn.py:1007-1010), the omega gradient
path passed sigma_floor=None unconditionally, so full-covariance under
em_mode='em_phi_p' * gauge_param='omega' would run the attention-KL
Cholesky without the E-step floor guarantee that sibling phi-path call
sites (:1196-1211, :1226-1244, :1934-1953) honor.
"""

import math
import torch
import pytest

import transformer.core.variational_ffn as vffn_mod
from transformer.core.variational_ffn import VariationalFFNDynamic
from math_utils.generators import generate_so3_generators


@pytest.fixture
def small_omega_config():
    return dict(B=2, N=4, K=6, K_h=3, irrep_dims=[3, 3])


def _make_omega_ffn(K, irrep_dims, spd_floor_mode, e_step_sigma_floor,
                    propagate_kl_nonfinite):
    """Minimal omega-path VariationalFFNDynamic matching
    test_compute_omega_grad_direct_full_covariance (test_omega_gradient.py:508)."""
    G_block = torch.from_numpy(generate_so3_generators(irrep_dims[0])).float()
    n_gen = G_block.shape[0]
    generators = torch.zeros(n_gen, K, K)
    generators[:, :irrep_dims[0], :irrep_dims[0]] = G_block
    generators[:, irrep_dims[0]:, irrep_dims[0]:] = G_block

    return VariationalFFNDynamic(
        embed_dim=K,
        generators=generators,
        alpha=0.001,
        kappa=1.0,
        n_iterations=1,
        diagonal_covariance=False,
        irrep_dims=irrep_dims,
        gauge_param='omega',
        spd_floor_mode=spd_floor_mode,
        e_step_sigma_floor=e_step_sigma_floor,
        propagate_kl_nonfinite=propagate_kl_nonfinite,
    )


def _run_omega_grad(ffn, B, N, K, captured):
    """Invoke _compute_omega_grad_direct once with a full-cov Σ, patching
    compute_attention_weights in the variational_ffn module so every call
    records its kwargs into `captured`."""
    torch.manual_seed(0)
    mu = torch.randn(B, N, K)
    A = torch.randn(B, N, K, K) * 0.1
    sigma_full = torch.einsum('bnij,bnkj->bnik', A, A) + 0.5 * torch.eye(K)
    eye_k = torch.eye(K).expand(B, N, -1, -1).clone()
    omega = eye_k + 0.05 * torch.randn(B, N, K, K)

    real_caw = vffn_mod.compute_attention_weights

    def _capture_caw(*args, **kwargs):
        captured.append(dict(kwargs))
        return real_caw(*args, **kwargs)

    vffn_mod.compute_attention_weights = _capture_caw
    try:
        ffn._compute_omega_grad_direct(
            omega_current=omega, mu_current=mu, sigma_current=sigma_full,
            is_diagonal=False, mask=None, eps=1e-8,
        )
    finally:
        vffn_mod.compute_attention_weights = real_caw


def test_omega_path_threads_sigma_floor_under_eigclamp(small_omega_config):
    """spd_floor_mode='eigclamp' + full-cov: sigma_floor must equal
    self.e_step_sigma_floor at every compute_attention_weights call issued
    by _compute_omega_grad_direct."""
    B, N, K = (small_omega_config[k] for k in ('B', 'N', 'K'))
    irrep_dims = small_omega_config['irrep_dims']

    ffn = _make_omega_ffn(
        K=K, irrep_dims=irrep_dims,
        spd_floor_mode='eigclamp',
        e_step_sigma_floor=0.01,
        propagate_kl_nonfinite=False,
    )

    captured = []
    _run_omega_grad(ffn, B, N, K, captured)

    assert len(captured) >= 1, "No compute_attention_weights calls captured"
    for i, kw in enumerate(captured):
        assert kw.get('sigma_floor') == 0.01, (
            f"call[{i}] sigma_floor={kw.get('sigma_floor')!r}, expected 0.01 "
            f"(all kwargs: {sorted(kw.keys())})"
        )
        assert kw.get('spd_floor_mode') == 'eigclamp', (
            f"call[{i}] spd_floor_mode={kw.get('spd_floor_mode')!r}"
        )
        assert kw.get('propagate_nonfinite') is False, (
            f"call[{i}] propagate_nonfinite={kw.get('propagate_nonfinite')!r}"
        )


def test_omega_path_sigma_floor_none_under_ridge_mode(small_omega_config):
    """spd_floor_mode='ridge' must leave sigma_floor=None to preserve
    legacy behavior (matching sibling sites :1207-1211)."""
    B, N, K = (small_omega_config[k] for k in ('B', 'N', 'K'))
    irrep_dims = small_omega_config['irrep_dims']

    ffn = _make_omega_ffn(
        K=K, irrep_dims=irrep_dims,
        spd_floor_mode='ridge',
        e_step_sigma_floor=0.01,
        propagate_kl_nonfinite=False,
    )

    captured = []
    _run_omega_grad(ffn, B, N, K, captured)

    assert len(captured) >= 1
    for i, kw in enumerate(captured):
        assert kw.get('sigma_floor') is None, (
            f"call[{i}] expected sigma_floor=None under ridge mode, "
            f"got {kw.get('sigma_floor')!r}"
        )
        assert kw.get('spd_floor_mode') == 'ridge'


def test_omega_path_propagate_nonfinite_flag_is_threaded(small_omega_config):
    """Setting propagate_kl_nonfinite=True on the module must surface in
    every compute_attention_weights call's propagate_nonfinite kwarg."""
    B, N, K = (small_omega_config[k] for k in ('B', 'N', 'K'))
    irrep_dims = small_omega_config['irrep_dims']

    ffn = _make_omega_ffn(
        K=K, irrep_dims=irrep_dims,
        spd_floor_mode='eigclamp',
        e_step_sigma_floor=0.01,
        propagate_kl_nonfinite=True,
    )

    captured = []
    _run_omega_grad(ffn, B, N, K, captured)

    assert len(captured) >= 1
    for i, kw in enumerate(captured):
        assert kw.get('propagate_nonfinite') is True, (
            f"call[{i}] propagate_nonfinite={kw.get('propagate_nonfinite')!r}"
        )
