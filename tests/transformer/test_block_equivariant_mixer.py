"""
Tests for the block-level gauge-equivariant mixer
(`BlockEquivariantMixer`, gated by `use_block_equivariant_mixer`).

Companion to test_equivariant_head_mixer.py, which exercises the
in-attention mixer. The math is the same Schur commutant; this module
tests the post-FFN insertion point used when skip_attention=True.

See `transformer/core/block_equivariant_mixer.py`.
"""

from __future__ import annotations

import warnings

import pytest
import torch

from transformer.core.block_equivariant_mixer import BlockEquivariantMixer


# =============================================================================
# T1 — Identity at initialization (bitwise no-op on μ, Σ for both cov modes)
# =============================================================================

def test_identity_init_diagonal_cov():
    torch.manual_seed(0)
    mixer = BlockEquivariantMixer(
        irrep_spec=[('fund', 2, 10)],
        embed_dim=20,
        diagonal_covariance=True,
    )
    B, N, K = 2, 5, 20
    mu = torch.randn(B, N, K)
    sigma = torch.rand(B, N, K) + 0.1
    mu_out, sigma_out = mixer(mu, sigma)
    assert torch.equal(mu_out, mu), (
        f"μ changed at init (max diff {(mu_out - mu).abs().max()})"
    )
    assert torch.equal(sigma_out, sigma), (
        f"σ (diag) changed at init (max diff {(sigma_out - sigma).abs().max()})"
    )


def test_identity_init_full_cov():
    torch.manual_seed(0)
    mixer = BlockEquivariantMixer(
        irrep_spec=[('fund', 2, 10)],
        embed_dim=20,
        diagonal_covariance=False,
    )
    B, N, K = 2, 5, 20
    mu = torch.randn(B, N, K)
    A = torch.randn(B, N, K, K) * 0.2
    sigma = A @ A.transpose(-1, -2) + 0.1 * torch.eye(K)
    sigma = 0.5 * (sigma + sigma.transpose(-1, -2))
    mu_out, sigma_out = mixer(mu, sigma)
    assert torch.equal(mu_out, mu)
    # Full-cov path explicitly symmetrizes; identity sandwich preserves Σ up
    # to that symmetrization which is a no-op when input is already symmetric.
    assert torch.allclose(sigma_out, sigma, atol=1e-6), (
        f"Σ (full) changed at init (max diff {(sigma_out - sigma).abs().max()})"
    )


# =============================================================================
# T2 — Parameter count matches Schur-Weyl dimension
# =============================================================================

@pytest.mark.parametrize(
    "irrep_spec,embed_dim,expected",
    [
        ([('fund', 2, 10)], 20, 4),     # one group of multiplicity 2 → 2² = 4
        ([('fund', 3, 5)], 15, 9),      # one group of multiplicity 3 → 3² = 9
        ([('fund', 4, 4)], 16, 16),     # one group of multiplicity 4 → 4² = 16
    ],
)
def test_param_count_schur_weyl(irrep_spec, embed_dim, expected):
    mixer = BlockEquivariantMixer(
        irrep_spec=irrep_spec,
        embed_dim=embed_dim,
        diagonal_covariance=False,
    )
    total = sum(p.numel() for p in mixer.mixer_delta)
    assert total == expected, (
        f"Expected {expected} scalar params (Schur-Weyl Σ n²), got {total}"
    )


# =============================================================================
# T3 — Gauge equivariance under tied gauge
# =============================================================================

def test_equivariance_under_tied_gauge():
    r"""Verify mixer(h μ, h Σ hᵀ) = h mixer(μ, Σ) when h = block_diag(h_d, h_d)
    is a tied gauge across heads in the same irrep type.
    """
    torch.manual_seed(1)
    mixer = BlockEquivariantMixer(
        irrep_spec=[('fund', 2, 10)],
        embed_dim=20,
        diagonal_covariance=False,
    )
    # Perturb A off identity (δ_t ≠ 0).
    with torch.no_grad():
        mixer.mixer_delta[0].data = torch.tensor(
            [[0.2, 0.3], [-0.2, -0.1]], dtype=torch.float32
        )

    B, N, K = 1, 3, 20
    d = 10
    mu = torch.randn(B, N, K)
    A = torch.randn(B, N, K, K) * 0.2
    sigma = A @ A.transpose(-1, -2) + 0.1 * torch.eye(K)
    sigma = 0.5 * (sigma + sigma.transpose(-1, -2))

    # Tied gauge h = block_diag(h_small, h_small).
    h_small = torch.eye(d) + 0.1 * torch.randn(d, d)
    h = torch.zeros(K, K)
    h[:d, :d] = h_small
    h[d:, d:] = h_small

    # Path A: mixer first, then gauge transform.
    mu_m, sigma_m = mixer(mu, sigma)
    mu_then_h = mu_m @ h.t()
    sigma_then_h = h @ sigma_m @ h.t()
    sigma_then_h = 0.5 * (sigma_then_h + sigma_then_h.transpose(-1, -2))

    # Path B: gauge transform first, then mixer.
    mu_h = mu @ h.t()
    sigma_h = h @ sigma @ h.t()
    sigma_h = 0.5 * (sigma_h + sigma_h.transpose(-1, -2))
    mu_h_m, sigma_h_m = mixer(mu_h, sigma_h)

    assert torch.allclose(mu_then_h, mu_h_m, atol=1e-4), (
        f"μ-equivariance failed (max diff {(mu_then_h - mu_h_m).abs().max()})"
    )
    assert torch.allclose(sigma_then_h, sigma_h_m, atol=1e-4), (
        f"Σ-equivariance failed (max diff {(sigma_then_h - sigma_h_m).abs().max()})"
    )


# =============================================================================
# T4 — Diagonal closed-form == diagonal of full sandwich
# =============================================================================

def test_diagonal_closed_form_matches_full_sandwich():
    """For the same input σ (interpreted both ways), the diagonal-mode closed
    form must equal the diagonal of the full sandwich `MΣMᵀ`.
    """
    torch.manual_seed(2)
    diag_mixer = BlockEquivariantMixer(
        irrep_spec=[('fund', 3, 4)], embed_dim=12, diagonal_covariance=True,
    )
    full_mixer = BlockEquivariantMixer(
        irrep_spec=[('fund', 3, 4)], embed_dim=12, diagonal_covariance=False,
    )
    # Same perturbation on both mixers' A_t so they realize the same M.
    delta = torch.tensor(
        [[0.1, 0.2, -0.1], [0.0, 0.3, 0.2], [-0.1, 0.05, 0.15]],
        dtype=torch.float32,
    )
    with torch.no_grad():
        diag_mixer.mixer_delta[0].data = delta.clone()
        full_mixer.mixer_delta[0].data = delta.clone()

    B, N, K = 2, 4, 12
    mu = torch.randn(B, N, K)
    diag_sigma = torch.rand(B, N, K) + 0.1
    full_sigma = torch.diag_embed(diag_sigma)

    _, sigma_diag_out = diag_mixer(mu, diag_sigma)
    _, sigma_full_out = full_mixer(mu, full_sigma)
    sigma_full_diag = torch.diagonal(sigma_full_out, dim1=-2, dim2=-1)

    assert torch.allclose(sigma_diag_out, sigma_full_diag, atol=1e-6), (
        f"diagonal closed-form != diag(MΣMᵀ); "
        f"max diff {(sigma_diag_out - sigma_full_diag).abs().max()}"
    )


# =============================================================================
# T5/T6 — Block integration under skip_attention=True and =False
# =============================================================================

def _make_block_cfg(*, skip_attention: bool, embed_dim: int = 20,
                    n_heads: int = 2, d_head: int = 10):
    """Build a minimal BlockConfig for a GaugeTransformerBlock smoke test."""
    from transformer.core.block_config import BlockConfig
    n_gen = n_heads * d_head * d_head  # GL(d) generators per head
    # Use block-diagonal-per-head generators (otherwise the VFE FFN may
    # blow up under arbitrary random K×K generators).
    generators = torch.zeros(n_gen, embed_dim, embed_dim)
    idx = 0
    for h in range(n_heads):
        for a in range(d_head):
            for b in range(d_head):
                generators[idx, h * d_head + a, h * d_head + b] = 1.0
                idx += 1
    return BlockConfig(
        embed_dim=embed_dim,
        irrep_spec=[('fund', n_heads, d_head)],
        hidden_dim=64,
        n_layers=1,
        skip_attention=skip_attention,
        use_block_equivariant_mixer=True,
        diagonal_covariance=True,
        ffn_irrep_dims=[d_head] * n_heads,
        generators=generators,
    )


def test_block_forward_skip_attention_true():
    """Build a block with `skip_attention=True, use_block_equivariant_mixer=True`
    and run a forward pass — output must be finite and shape-correct."""
    from transformer.core.blocks import GaugeTransformerBlock

    torch.manual_seed(3)
    cfg = _make_block_cfg(skip_attention=True)
    block = GaugeTransformerBlock(cfg)
    assert block.block_mixer is not None, "block_mixer should be constructed"

    B, N, K = 1, 4, cfg.embed_dim
    n_gen = cfg.generators.shape[0]
    mu_q = torch.randn(B, N, K)
    sigma_q = torch.rand(B, N, K) + 0.1
    phi = torch.zeros(B, N, n_gen)
    mu_prior = torch.randn(B, N, K)
    sigma_prior = torch.rand(B, N, K) + 0.1
    out = block(
        mu_q=mu_q,
        sigma_q=sigma_q,
        phi=phi,
        generators=cfg.generators,
        mu_prior=mu_prior,
        sigma_prior=sigma_prior,
    )
    mu_out = out[0]
    assert mu_out.shape == (B, N, K), f"got {mu_out.shape}"
    assert torch.isfinite(mu_out).all(), "non-finite μ_out"


def test_block_forward_skip_attention_false():
    """`skip_attention=False` is forced to True (attention sublayer removed
    2026-06-01); the post-FFN block mixer still runs in the pure-VFE block."""
    import warnings
    from transformer.core.blocks import GaugeTransformerBlock

    torch.manual_seed(4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = _make_block_cfg(skip_attention=False)
    assert cfg.skip_attention is True  # forced by BlockConfig.__post_init__
    block = GaugeTransformerBlock(cfg)
    assert block.block_mixer is not None
    # The attention sublayer no longer exists.
    assert not hasattr(block, 'attention')

    B, N, K = 1, 4, cfg.embed_dim
    n_gen = cfg.generators.shape[0]
    mu_q = torch.randn(B, N, K)
    sigma_q = torch.rand(B, N, K) + 0.1
    phi = torch.zeros(B, N, n_gen)
    mu_prior = torch.randn(B, N, K)
    sigma_prior = torch.rand(B, N, K) + 0.1
    out = block(
        mu_q=mu_q,
        sigma_q=sigma_q,
        phi=phi,
        generators=cfg.generators,
        mu_prior=mu_prior,
        sigma_prior=sigma_prior,
    )
    assert out[0].shape == (B, N, K)
    assert torch.isfinite(out[0]).all()


# =============================================================================
# T7/T8 — Mutual exclusion at BlockConfig level
# =============================================================================

def test_mutual_exclusion_with_attention_mixer():
    """use_block_equivariant_mixer + use_equivariant_head_mixer raises ValueError."""
    from transformer.core.block_config import BlockConfig
    with pytest.raises(ValueError, match="mutually exclusive"):
        BlockConfig(
            use_block_equivariant_mixer=True,
            use_equivariant_head_mixer=True,
        )


def test_mutual_exclusion_with_output_projection():
    """use_block_equivariant_mixer + use_output_projection raises ValueError."""
    from transformer.core.block_config import BlockConfig
    with pytest.raises(ValueError, match="mutually exclusive"):
        BlockConfig(
            use_block_equivariant_mixer=True,
            use_output_projection=True,
        )


# =============================================================================
# T9 — Independent-gauge warning fires (parallel to the attention-side check)
# =============================================================================

def test_independent_gauge_warning_fires():
    """GL(K) multi-head with independent per-head gauges (default config)
    emits a RuntimeWarning at construction matching the attention-side
    mixer's policy."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        BlockEquivariantMixer(
            irrep_spec=[('fund', 2, 10)],
            embed_dim=20,
            diagonal_covariance=True,
            gauge_group='GLK',
        )
    msgs = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
    assert any(
        "independent per-head" in m.lower() or "commutant collapses" in m.lower()
        for m in msgs
    ), f"expected independent-gauge RuntimeWarning, got: {msgs}"
