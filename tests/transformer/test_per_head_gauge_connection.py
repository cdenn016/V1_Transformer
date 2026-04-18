"""Tests for PerHeadGaugeConnection + partition_generators_by_block.

Covers CS-1b invariants:
  1. partition_generators_by_block correctly splits block-diagonal generators.
  2. PerHeadGaugeConnection output shape matches (B, N, N, n_gen_total).
  3. Cross-head non-leakage: perturbing μ in head h does not change δ for head g≠h.
  4. Antisymmetry: δ(μ_j, μ_i) == -δ(μ_i, μ_j) when antisymmetrize=True.
  5. Flat-at-init: with connection_init_scale=0, all δ outputs are zero.
"""

import pytest
import torch

from transformer.core.connection import (
    GaugeConnection,
    PerHeadGaugeConnection,
    partition_generators_by_block,
)


def _glk_block_generators(K_total: int, d: int, start: int) -> torch.Tensor:
    """Build d² elementary matrix units E_ij embedded in the (start, start+d) block."""
    gens = []
    for i in range(d):
        for j in range(d):
            G = torch.zeros(K_total, K_total)
            G[start + i, start + j] = 1.0
            gens.append(G)
    return torch.stack(gens)


def _build_two_head_glk_generators(d_h: int = 5):
    """GL(5) × GL(5) block-diagonal generator set over K=10."""
    K = 2 * d_h
    gens_h0 = _glk_block_generators(K, d_h, start=0)
    gens_h1 = _glk_block_generators(K, d_h, start=d_h)
    return torch.cat([gens_h0, gens_h1], dim=0), [d_h, d_h]


def test_partition_generators_by_block_shapes():
    """partition_generators_by_block returns per-head (n_gen_h, d_h, d_h) tensors."""
    gens, irrep_dims = _build_two_head_glk_generators(d_h=5)
    per_head = partition_generators_by_block(gens, irrep_dims)
    assert len(per_head) == 2
    assert per_head[0].shape == (25, 5, 5)
    assert per_head[1].shape == (25, 5, 5)


def test_partition_generators_by_block_content():
    """Partitioned generators match the diagonal blocks of the input."""
    gens, irrep_dims = _build_two_head_glk_generators(d_h=5)
    per_head = partition_generators_by_block(gens, irrep_dims)
    # Head 0: generators 0..24 should have non-zero (5,5) top-left block.
    assert per_head[0].sum().item() == 25  # 25 generators × 1.0 each
    assert per_head[1].sum().item() == 25


def test_partition_rejects_non_block_diagonal():
    """Generators with off-block support trigger ValueError."""
    K = 10
    # A generator with mass split 50/50 across two blocks.
    G_bad = torch.zeros(K, K)
    G_bad[0, 0] = 1.0
    G_bad[5, 5] = 1.0
    gens = G_bad.unsqueeze(0)
    with pytest.raises(ValueError, match="does not localize"):
        partition_generators_by_block(gens, [5, 5])


def test_per_head_connection_output_shape():
    """Forward pass returns (B, N, N, n_gen_total)."""
    gens, irrep_dims = _build_two_head_glk_generators(d_h=5)
    per_head = partition_generators_by_block(gens, irrep_dims)
    phgc = PerHeadGaugeConnection(
        irrep_dims=irrep_dims,
        per_head_generators=per_head,
        connection_type='bilinear',
        init_scale=0.01,
    )
    B, N, K = 2, 4, 10
    mu = torch.randn(B, N, K)
    delta = phgc(mu, mu)
    assert delta.shape == (B, N, N, 50)


def test_cross_head_non_leakage():
    """Perturbing μ in head 0 must not change δ for head 1's generators."""
    gens, irrep_dims = _build_two_head_glk_generators(d_h=5)
    per_head = partition_generators_by_block(gens, irrep_dims)
    phgc = PerHeadGaugeConnection(
        irrep_dims=irrep_dims,
        per_head_generators=per_head,
        connection_type='bilinear',
        init_scale=0.01,
    )
    torch.manual_seed(0)
    B, N, K = 2, 4, 10
    mu = torch.randn(B, N, K)
    delta_ref = phgc(mu, mu)

    mu_pert = mu.clone()
    mu_pert[..., :5] += 100.0  # Perturb head 0 only
    delta_pert = phgc(mu_pert, mu_pert)

    # Head 1's generator block: indices [25:50]
    head1_ref = delta_ref[..., 25:50]
    head1_pert = delta_pert[..., 25:50]
    assert (head1_ref - head1_pert).abs().max().item() < 1e-6


def test_antisymmetry_per_head():
    """δ(μ_j, μ_i) == -δ(μ_i, μ_j) per head when antisymmetrize=True."""
    gens, irrep_dims = _build_two_head_glk_generators(d_h=5)
    per_head = partition_generators_by_block(gens, irrep_dims)
    phgc = PerHeadGaugeConnection(
        irrep_dims=irrep_dims,
        per_head_generators=per_head,
        connection_type='bilinear',
        antisymmetrize=True,
        init_scale=0.01,
    )
    torch.manual_seed(1)
    B, N, K = 2, 4, 10
    mu = torch.randn(B, N, K)
    delta_ij = phgc(mu, mu)
    # Swap i ↔ j: use the transposed (j, i) ordering
    delta_ji = delta_ij.transpose(1, 2)
    torch.testing.assert_close(delta_ij, -delta_ji, atol=1e-5, rtol=1e-5)


def test_flat_at_init_zero_scale():
    """connection_init_scale=0 yields δ ≡ 0 at init for every head."""
    gens, irrep_dims = _build_two_head_glk_generators(d_h=5)
    per_head = partition_generators_by_block(gens, irrep_dims)
    phgc = PerHeadGaugeConnection(
        irrep_dims=irrep_dims,
        per_head_generators=per_head,
        connection_type='bilinear',
        init_scale=0.0,
    )
    B, N, K = 2, 4, 10
    mu = torch.randn(B, N, K)
    delta = phgc(mu, mu)
    assert delta.abs().max().item() == 0.0


def test_mlp_connection_type_flat_at_init():
    """MLP connection type is also flat at init (zero-init output layer)."""
    gens, irrep_dims = _build_two_head_glk_generators(d_h=5)
    per_head = partition_generators_by_block(gens, irrep_dims)
    phgc = PerHeadGaugeConnection(
        irrep_dims=irrep_dims,
        per_head_generators=per_head,
        connection_type='mlp',
        hidden_dim=16,
    )
    B, N, K = 2, 4, 10
    mu = torch.randn(B, N, K)
    delta = phgc(mu, mu)
    assert delta.abs().max().item() == 0.0


def test_irrep_dims_mismatch_raises():
    """Mismatched irrep_dims and per_head_generators lengths raise."""
    gens = _glk_block_generators(10, 5, 0)
    per_head = [gens[:, :5, :5]]  # Only one head's data
    with pytest.raises(ValueError, match="length"):
        PerHeadGaugeConnection(
            irrep_dims=[5, 5],
            per_head_generators=per_head,
        )


def test_sum_irrep_dims_matches_K():
    """partition_generators_by_block enforces sum(irrep_dims) == K."""
    gens = _glk_block_generators(10, 5, 0)
    with pytest.raises(ValueError, match="does not match K"):
        partition_generators_by_block(gens, [3, 4])  # sum=7, K=10


# ----------------------------------------------------------------------------
# Regression tests for the 2026-04-17 ultrareview Phase M fixes.
# ----------------------------------------------------------------------------

class TestNonFlatTransportFallback:
    """bug_004: PerHeadGaugeConnection cannot handle shared multi-irrep
    SO(N) generators (one G_fund replicated across mult>=2 copies has
    per-block Frobenius mass 1/mult < 0.999). The fallback to a single
    global GaugeConnection must keep these configs constructible."""

    def _base_config(self):
        return {
            'vocab_size': 50,
            'embed_dim': 6,           # 2 fund copies × 3-dim
            'n_layers': 1,
            'irrep_spec': [('fund', 2, 3)],
            'hidden_dim': 12,
            'max_seq_len': 16,
            'kappa_beta': 1.0,
            'evolve_sigma': False,
            'evolve_phi': True,
            'gauge_group': 'SO_N',
            'gauge_dim': 3,
            'use_block_diagonal_kl': True,
            'diagonal_covariance': True,
            'use_layernorm': True,
            'use_residual': True,
            'non_flat_transport': True,
            'connection_type': 'bilinear',
            'connection_init_scale': 0.0,
        }

    def test_so3_multi_irrep_non_flat_transport_constructs(self, recwarn):
        """irrep_spec=[('fund', 2, 3)] with non_flat_transport=True used to
        raise ValueError at construction (regression from per-head split)."""
        import torch
        from transformer.core.model import GaugeTransformerLM

        cfg = self._base_config()
        # Construction must succeed via the global-GaugeConnection fallback.
        model = GaugeTransformerLM(cfg)

        # The fallback must announce itself via RuntimeWarning.
        assert any(issubclass(w.category, RuntimeWarning) and 'global GaugeConnection' in str(w.message)
                   for w in recwarn.list), (
            'Expected a RuntimeWarning announcing the GaugeConnection fallback.'
        )

        # _connection_mode tag is set on the block.
        block = model.transformer.blocks[0]
        assert getattr(block, '_connection_mode', None) == 'global', (
            f'_connection_mode should be "global" after fallback; '
            f'got {getattr(block, "_connection_mode", None)!r}'
        )

        # Forward + backward must succeed.
        x = torch.randint(0, 50, (2, 8))
        logits = model(x)
        assert logits.shape == (2, 8, 50)
        logits.sum().backward()

    def test_per_head_path_still_taken_for_block_local_generators(self):
        """A clean GLK config keeps the per-head path; no fallback warning."""
        import torch
        from transformer.core.model import GaugeTransformerLM

        cfg = {
            'vocab_size': 50,
            'embed_dim': 24,
            'n_layers': 1,
            'irrep_spec': [('fund', 4, 6)],
            'hidden_dim': 48,
            'max_seq_len': 16,
            'kappa_beta': 1.0,
            'evolve_sigma': False,
            'evolve_phi': True,
            'gauge_group': 'GLK',
            'use_block_diagonal_kl': True,
            'diagonal_covariance': True,
            'use_layernorm': True,
            'use_residual': True,
            'non_flat_transport': True,
            'connection_type': 'bilinear',
            'connection_init_scale': 0.0,
        }
        model = GaugeTransformerLM(cfg)
        block = model.transformer.blocks[0]
        assert getattr(block, '_connection_mode', None) == 'per_head'


class TestGaugeCovariantRidgeUnderNonFlat:
    """bug_005: Non-flat transport caches at blocks.py:566-569 used to be
    Omega-only. With gauge_covariant_ridge=True, downstream branches in
    aggregate_messages then silently fell back to eps*I. Verify the cache
    now carries 'exp_phi' so the covariant ridge actually fires."""

    def test_blocks_nonflat_cache_carries_exp_phi(self):
        import torch
        from transformer.core.model import GaugeTransformerLM

        cfg = {
            'vocab_size': 50,
            'embed_dim': 24,
            'n_layers': 1,
            'irrep_spec': [('fund', 4, 6)],
            'hidden_dim': 48,
            'max_seq_len': 16,
            'kappa_beta': 1.0,
            'evolve_sigma': False,
            'evolve_phi': True,
            'gauge_group': 'GLK',
            'use_block_diagonal_kl': True,
            'diagonal_covariance': True,
            'use_layernorm': True,
            'use_residual': True,
            'non_flat_transport': True,
            'gauge_covariant_ridge': True,
            'connection_type': 'bilinear',
            'connection_init_scale': 0.0,
        }
        model = GaugeTransformerLM(cfg)

        # The blocks.py:566-575 cache producer feeds ATTENTION (not the FFN
        # in VFE_dynamic mode). Hook attention to capture the dict keys.
        captured = {}
        block = model.transformer.blocks[0]

        def pre_hook(_module, args, kwargs):
            ct = kwargs.get('cached_head_transports')
            if ct is not None and len(ct) > 0:
                captured['keys'] = sorted(ct[0].keys())

        handle = block.attention.register_forward_pre_hook(pre_hook, with_kwargs=True)
        try:
            x = torch.randint(0, 50, (2, 8))
            model(x)
        finally:
            handle.remove()

        assert 'keys' in captured, (
            'pre_hook did not observe a non-None cached_head_transports — '
            'either the non-flat code path is not firing or the FFN was bypassed.'
        )
        assert 'Omega' in captured['keys'], f'Omega missing: {captured["keys"]}'
        assert 'exp_phi' in captured['keys'], (
            f"exp_phi missing from non-flat cache; "
            f"gauge_covariant_ridge would silently degrade to eps*I. "
            f"Got keys: {captured['keys']}"
        )


# ----------------------------------------------------------------------------
# Phase N: tri-state rope_full_gauge ('off' | 'vfe_only' | 'both') + no-irrep
# rope chain-rule hard-error.
# ----------------------------------------------------------------------------

class TestRopeFullGaugeTriState:

    def test_coerce_legacy_bool_true(self):
        from transformer.core.block_config import _coerce_rope_full_gauge
        assert _coerce_rope_full_gauge(True) == 'vfe_only'

    def test_coerce_legacy_bool_false(self):
        from transformer.core.block_config import _coerce_rope_full_gauge
        assert _coerce_rope_full_gauge(False) == 'off'

    def test_coerce_none(self):
        from transformer.core.block_config import _coerce_rope_full_gauge
        assert _coerce_rope_full_gauge(None) == 'off'

    def test_coerce_canonical_strings(self):
        from transformer.core.block_config import _coerce_rope_full_gauge
        for s in ('off', 'vfe_only', 'both'):
            assert _coerce_rope_full_gauge(s) == s

    def test_coerce_invalid_string_raises(self):
        import pytest
        from transformer.core.block_config import _coerce_rope_full_gauge
        with pytest.raises(ValueError, match='must be one of'):
            _coerce_rope_full_gauge('full')

    def test_coerce_invalid_type_raises(self):
        import pytest
        from transformer.core.block_config import _coerce_rope_full_gauge
        with pytest.raises(TypeError):
            _coerce_rope_full_gauge(1.5)

    def _base_config(self, **overrides):
        cfg = {
            'vocab_size': 50,
            'embed_dim': 24,
            'n_layers': 1,
            'irrep_spec': [('fund', 4, 6)],
            'hidden_dim': 48,
            'max_seq_len': 16,
            'kappa_beta': 1.0,
            'evolve_sigma': False,
            'evolve_phi': True,
            'gauge_group': 'GLK',
            'use_block_diagonal_kl': True,
            'diagonal_covariance': True,
            'use_layernorm': True,
            'use_residual': True,
            'use_rope': True,
        }
        cfg.update(overrides)
        return cfg

    def test_legacy_bool_true_maps_to_vfe_only(self):
        """rope_full_gauge=True (legacy) must coerce to 'vfe_only' end-to-end."""
        from transformer.core.model import GaugeTransformerLM
        cfg = self._base_config(
            rope_full_gauge=True,
            diagonal_covariance=False,  # avoid the diagonal-cov warning collision
        )
        model = GaugeTransformerLM(cfg)
        block = model.transformer.blocks[0]
        assert block.attention.rope_full_gauge_mode == 'vfe_only'
        assert block.ffn._rope_full_gauge_vfe == 'vfe_only'

    def test_legacy_bool_false_maps_to_off(self):
        from transformer.core.model import GaugeTransformerLM
        cfg = self._base_config(rope_full_gauge=False)
        model = GaugeTransformerLM(cfg)
        block = model.transformer.blocks[0]
        assert block.attention.rope_full_gauge_mode == 'off'
        assert block.ffn._rope_full_gauge_vfe == 'off'

    def test_string_off_default(self):
        """No rope_full_gauge in config → 'off'."""
        from transformer.core.model import GaugeTransformerLM
        cfg = self._base_config()
        model = GaugeTransformerLM(cfg)
        block = model.transformer.blocks[0]
        assert block.attention.rope_full_gauge_mode == 'off'
        assert block.ffn._rope_full_gauge_vfe == 'off'

    def test_both_requires_non_diagonal_covariance(self):
        """'both' under diagonal_covariance must raise at attention forward."""
        import pytest
        import torch
        from transformer.core.model import GaugeTransformerLM
        cfg = self._base_config(
            rope_full_gauge='both',
            diagonal_covariance=True,
        )
        model = GaugeTransformerLM(cfg)
        x = torch.randint(0, 50, (2, 8))
        with pytest.raises(ValueError, match="rope_full_gauge='both' requires diagonal_covariance=False"):
            model(x)

    def test_both_forward_backward_with_full_cov(self):
        """rope_full_gauge='both' + diagonal_covariance=False must construct,
        run forward, and produce gradients without error."""
        import torch
        from transformer.core.model import GaugeTransformerLM
        cfg = self._base_config(
            rope_full_gauge='both',
            diagonal_covariance=False,
            evolve_sigma=True,
        )
        model = GaugeTransformerLM(cfg)
        block = model.transformer.blocks[0]
        assert block.attention.rope_full_gauge_mode == 'both'
        assert block.ffn._rope_full_gauge_vfe == 'both'

        x = torch.randint(0, 50, (2, 8))
        logits = model(x)
        assert logits.shape == (2, 8, 50)
        logits.sum().backward()

    def test_invalid_string_in_config_raises_at_construction(self):
        import pytest
        from transformer.core.model import GaugeTransformerLM
        cfg = self._base_config(rope_full_gauge='partial')
        with pytest.raises(ValueError, match='must be one of'):
            GaugeTransformerLM(cfg)


class TestRopeNoIrrepDimsHardError:
    """Phase N: vfe_gradients.py used to warn on use_rope=True without
    irrep_dims and continue with a biased gradient. Now it hard-errors."""

    def test_use_rope_without_irrep_dims_raises(self):
        import pytest
        import torch
        from transformer.core.vfe_gradients import compute_vfe_gradients_gpu

        B, N, K, n_gen = 1, 4, 8, 8
        mu_q = torch.randn(B, N, K)
        sigma_q = torch.ones(B, N, K) * 0.1
        mu_p = torch.zeros(B, N, K)
        sigma_p = torch.ones(B, N, K)
        beta = torch.softmax(torch.randn(B, N, N), dim=-1)
        phi = torch.zeros(B, N, n_gen)
        generators = torch.eye(K).unsqueeze(0).repeat(n_gen, 1, 1) * 0.01

        with pytest.raises(ValueError, match='use_rope=True without irrep_dims'):
            compute_vfe_gradients_gpu(
                mu_q=mu_q, sigma_q=sigma_q, mu_p=mu_p, sigma_p=sigma_p,
                beta=beta, phi=phi, generators=generators,
                use_rope=True,
                irrep_dims=None,
            )
