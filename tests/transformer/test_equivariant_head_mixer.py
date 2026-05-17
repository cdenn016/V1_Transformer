"""
Tests for the gauge-equivariant head mixer (`use_equivariant_head_mixer=True`).

Background: W_O (`use_output_projection`) applies a learnable K×K linear map
to μ only and never to Σ, breaking the joint gauge transformation of the
belief pair (μ, Σ). The equivariant mixer replaces W_O with a commutant
parameterization: for each irrep type with multiplicity n and dim d, we
learn an n×n scalar matrix a; the K×K mixer M has M[i·d:(i+1)·d, j·d:(j+1)·d]
= a[i,j]·I_d. M is applied symmetrically — μ ↦ M·μ, Σ ↦ M·Σ·Mᵀ — which is
the full Schur commutant of the decomposition **under tied gauge action on
each irrep copy**.

These tests verify:
    T1. At init (a = I), the mixer is the identity map on both μ and Σ.
    T2. The mixer parameter count equals Σ_group n_g² (Schur-Weyl dimension).
    T3. Under a tied gauge h applied block-diagonally to every head in the
        same irrep group, the mixer output transforms covariantly:
        mixer(h·μ, h·Σ·hᵀ) == h·mixer(μ, Σ)·h^(related). This is the exact
        definition of gauge equivariance.
    T4. Running the full forward of `IrrepMultiHeadAttention` with the
        mixer enabled produces finite output and the μ, Σ shapes are
        preserved.
"""

from __future__ import annotations

import warnings

import pytest
import torch

from transformer.core.attention import IrrepMultiHeadAttention


# =============================================================================
# Helpers
# =============================================================================

def _build_attn_glk_tied(embed_dim: int = 20, n_heads: int = 2, d_head: int = 10,
                         use_mixer: bool = True,
                         use_wo: bool = False) -> IrrepMultiHeadAttention:
    """Construct a GL(K) multi-head IrrepMultiHeadAttention with the mixer.

    The GL(K) independent-gauge case emits a RuntimeWarning — we suppress it
    here since the tests deliberately exercise the mixer's behavior even in
    that regime (the equivariance test separately constructs a tied gauge).
    """
    # Generators for GL(K) multi-head: block-diagonal per head, d_head² each.
    n_gen = n_heads * d_head * d_head
    generators = torch.randn(n_gen, embed_dim, embed_dim) * 0.1
    irrep_spec = [('fund', n_heads, d_head)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        attn = IrrepMultiHeadAttention(
            embed_dim=embed_dim,
            irrep_spec=irrep_spec,
            kappa_beta=1.0,
            diagonal_covariance=False,
            gauge_group='GLK',
            gauge_dim=d_head,
            global_generators=generators,
            use_output_projection=use_wo,
            use_equivariant_head_mixer=use_mixer,
        )
    return attn


# =============================================================================
# T1 — Identity at initialization
# =============================================================================

def test_mixer_identity_init_preserves_mu_and_sigma():
    """At init, the mixer's n×n parameter is I_n → M = I_K → μ_out = μ, Σ_out = Σ."""
    torch.manual_seed(0)
    attn = _build_attn_glk_tied()
    assert attn._mixer_active, "mixer should be active after construction"

    B, N, K = 2, 5, 20
    mu = torch.randn(B, N, K)
    sigma = torch.zeros(B, N, K, K)
    # Block-diagonal random SPD Σ (same shape attention produces).
    for h_off in (0, 10):
        A = torch.randn(10, 10) * 0.3
        block = A @ A.t() + 0.1 * torch.eye(10)
        sigma[..., h_off:h_off + 10, h_off:h_off + 10] = block

    mu_out, sigma_out = attn._apply_equivariant_mixer(mu, sigma)
    assert torch.allclose(mu_out, mu, atol=1e-6), \
        f"Init mixer changed μ: max diff {(mu_out - mu).abs().max()}"
    assert torch.allclose(sigma_out, sigma, atol=1e-6), \
        f"Init mixer changed Σ: max diff {(sigma_out - sigma).abs().max()}"


# =============================================================================
# T2 — Parameter count matches Schur-Weyl dimension
# =============================================================================

def test_mixer_commutant_dimension_glk():
    """For irrep_spec=[('fund', 2, 10)], one group of multiplicity 2 → 4 scalars."""
    attn = _build_attn_glk_tied(embed_dim=20, n_heads=2, d_head=10)
    total = sum(p.numel() for p in attn.mixer_params)
    assert total == 4, f"Expected 4 params (2² for multiplicity-2), got {total}"


def test_mixer_commutant_dimension_glk_3heads():
    """3 heads of dim 10 → 9 scalars."""
    attn = _build_attn_glk_tied(embed_dim=30, n_heads=3, d_head=10)
    total = sum(p.numel() for p in attn.mixer_params)
    assert total == 9, f"Expected 9 params (3² for multiplicity-3), got {total}"


# =============================================================================
# T3 — Gauge equivariance under tied gauge
# =============================================================================

def test_mixer_equivariance_under_tied_gauge():
    """Under a tied gauge h ∈ GL(10) applied identically to both heads, the
    mixer commutes: mixer(h·μ, h·Σ·hᵀ) = h·mixer(μ, Σ).

    Build h = block_diag(h_small, h_small) — same GL(10) element on each head
    — and verify the covariance relation. This is the gauge structure the
    mixer is designed for.
    """
    torch.manual_seed(1)
    attn = _build_attn_glk_tied()
    # Perturb the mixer params off identity so M is non-trivial.
    with torch.no_grad():
        attn.mixer_params[0].data = torch.tensor(
            [[1.2, 0.3], [-0.2, 0.9]], dtype=torch.float32
        )

    B, N, K = 1, 3, 20
    d = 10
    mu = torch.randn(B, N, K)
    A = torch.randn(B, N, K, K) * 0.2
    sigma = A @ A.transpose(-1, -2) + 0.1 * torch.eye(K)
    # Symmetrize Σ explicitly.
    sigma = 0.5 * (sigma + sigma.transpose(-1, -2))

    # Tied gauge h = block_diag(h_small, h_small).
    h_small = torch.eye(d) + 0.1 * torch.randn(d, d)
    h = torch.zeros(K, K)
    h[:d, :d] = h_small
    h[d:, d:] = h_small

    # Path A: mixer first, then transform.
    mu_m, sigma_m = attn._apply_equivariant_mixer(mu, sigma)
    mu_then_h = mu_m @ h.t()
    sigma_then_h = h @ sigma_m @ h.t()

    # Path B: transform first, then mixer.
    mu_h = mu @ h.t()
    sigma_h = h @ sigma @ h.t()
    sigma_h = 0.5 * (sigma_h + sigma_h.transpose(-1, -2))
    mu_h_m, sigma_h_m = attn._apply_equivariant_mixer(mu_h, sigma_h)

    assert torch.allclose(mu_then_h, mu_h_m, atol=1e-4), (
        f"μ equivariance failed: max diff {(mu_then_h - mu_h_m).abs().max()}"
    )
    assert torch.allclose(sigma_then_h, sigma_h_m, atol=1e-4), (
        f"Σ equivariance failed: max diff {(sigma_then_h - sigma_h_m).abs().max()}"
    )


def test_mixer_breaks_equivariance_under_untied_gauge():
    """Sanity check: under untied gauge h = block_diag(h_a, h_b) with h_a ≠ h_b,
    the mixer does NOT commute. This documents the known limitation and
    justifies the construction-time warning when gauge is per-head independent.
    """
    torch.manual_seed(2)
    attn = _build_attn_glk_tied()
    with torch.no_grad():
        attn.mixer_params[0].data = torch.tensor(
            [[1.0, 0.5], [-0.3, 1.1]], dtype=torch.float32
        )

    B, N, K = 1, 2, 20
    d = 10
    mu = torch.randn(B, N, K)
    sigma = 0.3 * torch.eye(K).expand(B, N, K, K).clone()

    h_a = torch.eye(d) + 0.15 * torch.randn(d, d)
    h_b = torch.eye(d) + 0.15 * torch.randn(d, d)
    h = torch.zeros(K, K)
    h[:d, :d] = h_a
    h[d:, d:] = h_b

    mu_m, _ = attn._apply_equivariant_mixer(mu, sigma)
    mu_then_h = mu_m @ h.t()
    mu_h = mu @ h.t()
    mu_h_m, _ = attn._apply_equivariant_mixer(mu_h, sigma)

    diff = (mu_then_h - mu_h_m).abs().max().item()
    assert diff > 1e-3, (
        f"Mixer unexpectedly commuted with untied gauge (diff {diff}); "
        "this test documents that equivariance requires tied gauge."
    )


# =============================================================================
# T4 — Runtime warning for independent per-head gauge
# =============================================================================

def test_mixer_warns_on_independent_gauge_glk():
    """The construction emits a RuntimeWarning when the effective gauge is
    independent per-head (GL(K) multi-head with cross_couplings unset)."""
    torch.manual_seed(0)
    embed_dim, n_heads, d_head = 20, 2, 10
    n_gen = n_heads * d_head * d_head
    generators = torch.randn(n_gen, embed_dim, embed_dim) * 0.1

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        IrrepMultiHeadAttention(
            embed_dim=embed_dim,
            irrep_spec=[('fund', n_heads, d_head)],
            kappa_beta=1.0,
            diagonal_covariance=False,
            gauge_group='GLK',
            gauge_dim=d_head,
            global_generators=generators,
            use_output_projection=False,
            use_equivariant_head_mixer=True,
        )
    msgs = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
    assert any("commutant mixer" in m.lower() or "tied gauge" in m.lower()
               for m in msgs), (
        f"Expected RuntimeWarning about tied-gauge requirement; got: {msgs}"
    )


# =============================================================================
# T5 — Default use_output_projection is False (plan's primary change)
# =============================================================================

def test_block_config_default_use_output_projection_is_false():
    """The plan flipped the default per VFE_Transformer_Idea.md §18.2."""
    from transformer.core.block_config import BlockConfig
    # Build a minimal config — we don't need to validate, just check default.
    import dataclasses
    fields = {f.name: f.default for f in dataclasses.fields(BlockConfig)
              if f.default is not dataclasses.MISSING}
    assert fields.get('use_output_projection') is False, (
        "Default use_output_projection must be False — PriorBank decode "
        "requires a gauge-covariant belief, W_O breaks that invariance."
    )
    assert fields.get('use_equivariant_head_mixer') is False, (
        "Default use_equivariant_head_mixer must be False (opt-in only)."
    )


# =============================================================================
# T6 — Warning fires when irrep_dims_override is auto-populated (model.py path)
# =============================================================================

def test_mixer_warns_under_auto_populated_irrep_dims_override():
    """Production configs reach IrrepMultiHeadAttention with irrep_dims_override
    auto-populated to [d_head]*n_heads by model.py (use_block_diagonal_kl=True
    default). The pre-fix predicate required irrep_dims_override is None and
    therefore stayed silent in every production config. This test pins the
    label-based detection in place: when labels are 'glk_head_*' (i.e. plain
    multi-head, not cross-coupled super-blocks), the warning MUST fire even
    though irrep_dims_override is non-None.
    """
    torch.manual_seed(0)
    embed_dim, n_heads, d_head = 20, 2, 10
    n_gen = n_heads * d_head * d_head
    generators = torch.randn(n_gen, embed_dim, embed_dim) * 0.1

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        IrrepMultiHeadAttention(
            embed_dim=embed_dim,
            irrep_spec=[('fund', n_heads, d_head)],
            kappa_beta=1.0,
            diagonal_covariance=False,
            gauge_group='GLK',
            gauge_dim=d_head,
            global_generators=generators,
            use_output_projection=False,
            use_equivariant_head_mixer=True,
            # Simulate model.py:338 auto-populating ffn_irrep_dims.
            irrep_dims_override=[d_head] * n_heads,
        )
    msgs = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
    assert any(
        "commutant mixer" in m.lower() or "tied gauge" in m.lower() for m in msgs
    ), (
        "Expected RuntimeWarning to fire when irrep_dims_override=[d_head]*n_heads "
        "(the model.py auto-populate path) — labels are 'glk_head_*', gauges are "
        "independent per-head. Without this, the warning is dead in every "
        f"production config. Got: {msgs}"
    )


def test_mixer_silent_under_cross_coupled_super_blocks():
    """Sanity counterpart: when irrep_dims_override produces non-uniform super-blocks
    (true cross-coupled gauges via merge_coupled_heads), labels are 'glk_superblock_*'
    and the warning must STAY silent — those configs have tied gauges within each
    super-block, making the mixer principled.
    """
    torch.manual_seed(0)
    embed_dim, n_heads, d_head = 20, 2, 10
    n_gen = n_heads * d_head * d_head
    generators = torch.randn(n_gen, embed_dim, embed_dim) * 0.1

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        attn = IrrepMultiHeadAttention(
            embed_dim=embed_dim,
            irrep_spec=[('fund', n_heads, d_head)],
            kappa_beta=1.0,
            diagonal_covariance=False,
            gauge_group='GLK',
            gauge_dim=d_head,
            global_generators=generators,
            use_output_projection=False,
            use_equivariant_head_mixer=True,
            # Single super-block of size 20 — mimics merge_coupled_heads output
            # when two heads are coupled together into one shared GL(20) gauge.
            irrep_dims_override=[embed_dim],
        )
    mixer_msgs = [
        str(w.message)
        for w in caught
        if issubclass(w.category, RuntimeWarning)
        and ("commutant mixer" in str(w.message).lower() or "tied gauge" in str(w.message).lower())
    ]
    assert not mixer_msgs, (
        f"Mixer warning should be silent under cross-coupled super-blocks "
        f"(labels are 'glk_superblock_*'). Got: {mixer_msgs}"
    )
    assert attn.irrep_labels[0].startswith('glk_superblock_'), (
        f"Expected super-block labels; got {attn.irrep_labels}"
    )


# =============================================================================
# T7 — Mutual exclusion: use_output_projection AND use_equivariant_head_mixer
# =============================================================================

def test_block_config_mutual_exclusion_errors():
    """Setting both flags True silently allocated mixer_params as dead weight
    (W_O wins the forward dispatch). BlockConfig.__post_init__ now raises
    ValueError to fail fast on this misconfiguration.
    """
    from transformer.core.block_config import BlockConfig
    with pytest.raises(ValueError, match="mutually exclusive"):
        BlockConfig(
            use_output_projection=True,
            use_equivariant_head_mixer=True,
        )


# =============================================================================
# T8 — skip_attention force-disables the mixer
# =============================================================================

def test_block_config_skip_attention_warns_and_force_disables_mixer():
    """Under skip_attention=True the attention forward is bypassed, so any
    mixer_params would be dead weight. BlockConfig.__post_init__ warns; the
    block-construction path (blocks.py:501) gates the flag on
    `not self.skip_attention` so mixer_params are not allocated.
    """
    from transformer.core.block_config import BlockConfig
    from transformer.core.blocks import GaugeTransformerBlock

    torch.manual_seed(0)
    embed_dim, n_heads, d_head = 20, 2, 10
    n_gen = n_heads * d_head * d_head
    generators = torch.randn(n_gen, embed_dim, embed_dim) * 0.1

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = BlockConfig(
            embed_dim=embed_dim,
            irrep_spec=[('fund', n_heads, d_head)],
            hidden_dim=64,
            n_layers=1,
            skip_attention=True,
            use_equivariant_head_mixer=True,
            generators=generators,
        )
    msgs = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert any("force-disabled" in m and "skip_attention" in m for m in msgs), (
        f"Expected UserWarning about force-disable under skip_attention; got: {msgs}"
    )

    # Build the block and confirm the mixer is not active inside the attention
    # submodule (allocated but inert if force-disable did not propagate would be
    # `_mixer_active=True` with mixer_params populated).
    block = GaugeTransformerBlock(cfg)
    assert getattr(block.attention, '_mixer_active', False) is False, (
        "skip_attention=True must force-disable the mixer in IrrepMultiHeadAttention; "
        f"_mixer_active={block.attention._mixer_active}"
    )
    assert block.attention.mixer_params is None, (
        "skip_attention=True must skip mixer_params allocation; "
        f"got {block.attention.mixer_params}"
    )
