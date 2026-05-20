"""
Tests for transformer/core/gauge_preconditioner.py
===================================================

Tests the four phi-gradient preconditioning modes (clip, cartan, killing,
pullback) and the SL(K) projection. These control gradient dynamics for
gauge frames; a singular metric would crash training.
"""

import pytest
import torch
import math


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_glk_generators(K):
    """Standard E_{ij} basis for gl(K): K² generators."""
    n_gen = K * K
    G = torch.zeros(n_gen, K, K)
    idx = 0
    for i in range(K):
        for j in range(K):
            G[idx, i, j] = 1.0
            idx += 1
    return G


def _make_so_generators(K):
    """Skew-symmetric generators for so(K): K(K-1)/2 generators."""
    n_gen = K * (K - 1) // 2
    G = torch.zeros(n_gen, K, K)
    idx = 0
    for i in range(K):
        for j in range(i + 1, K):
            G[idx, i, j] = 1.0
            G[idx, j, i] = -1.0
            idx += 1
    return G


def _make_block_diagonal_glk_generators(irrep_dims):
    """Block-diagonal `gl(d_1) ⊕ ... ⊕ gl(d_H)` generators.

    For each block ``h`` of dim ``d_h``, emits ``d_h ** 2`` ``E_{ij}``
    generators with all mass on block ``h``'s spatial support. Total
    generator count is ``sum(d_h ** 2)``; ambient dimension is
    ``K_full = sum(d_h)``.
    """
    K_full = sum(irrep_dims)
    n_gen = sum(d * d for d in irrep_dims)
    G = torch.zeros(n_gen, K_full, K_full)
    block_start = 0
    gen_idx = 0
    for d_h in irrep_dims:
        for i in range(d_h):
            for j in range(d_h):
                G[gen_idx, block_start + i, block_start + j] = 1.0
                gen_idx += 1
        block_start += d_h
    return G


# ===========================================================================
# TestBuildCartanProjector
# ===========================================================================

class TestBuildCartanProjector:
    """Tests for build_cartan_projector()."""

    def test_projector_shape(self):
        """Preconditioner is (n_gen, n_gen)."""
        from transformer.core.gauge_preconditioner import build_cartan_projector
        G = _make_glk_generators(3)
        C = build_cartan_projector(G)
        assert C.shape == (9, 9)

    def test_projector_symmetric(self):
        """Preconditioner is symmetric for orthonormal generators."""
        from transformer.core.gauge_preconditioner import build_cartan_projector
        G = _make_glk_generators(3)
        C = build_cartan_projector(G)
        assert torch.allclose(C, C.T, atol=1e-5)

    def test_dampening_one_is_identity(self):
        """sym_dampening=1 → C = I (no preconditioning)."""
        from transformer.core.gauge_preconditioner import build_cartan_projector
        G = _make_glk_generators(3)
        C = build_cartan_projector(G, sym_dampening=1.0)
        I = torch.eye(9)
        assert torch.allclose(C, I, atol=1e-5)

    def test_apply_changes_gradient(self):
        """Cartan preconditioning modifies the gradient (non-trivial)."""
        from transformer.core.gauge_preconditioner import (
            build_cartan_projector, apply_cartan_preconditioning,
        )
        G = _make_glk_generators(3)
        C = build_cartan_projector(G, sym_dampening=0.1)
        grad = torch.randn(5, 9)
        precond = apply_cartan_preconditioning(grad, C)
        # Should differ from input (unless grad is entirely in so(K))
        assert not torch.allclose(grad, precond, atol=1e-3)

    def test_so_generators_less_dampened(self):
        """For so(K) generators, Cartan preconditioner acts as ~identity."""
        from transformer.core.gauge_preconditioner import (
            build_cartan_projector, apply_cartan_preconditioning,
        )
        G = _make_so_generators(4)  # 6 generators for so(4)
        C = build_cartan_projector(G, sym_dampening=0.1)
        # For purely antisymmetric generators, trace_prod should be negative,
        # and P_sym should project out less. Apply to random grad:
        grad = torch.randn(6)
        precond = apply_cartan_preconditioning(grad, C)
        assert torch.isfinite(precond).all()


# ===========================================================================
# TestKillingFormPreconditioner
# ===========================================================================

class TestKillingFormPreconditioner:
    """Tests for build_killing_form_preconditioner()."""

    def test_metric_shape(self):
        """Inverse metric is (n_gen, n_gen)."""
        from transformer.core.gauge_preconditioner import build_killing_form_preconditioner
        G = _make_glk_generators(3)
        M = build_killing_form_preconditioner(G)
        assert M.shape == (9, 9)

    def test_metric_symmetric(self):
        """Inverse metric should be symmetric."""
        from transformer.core.gauge_preconditioner import build_killing_form_preconditioner
        G = _make_glk_generators(3)
        M = build_killing_form_preconditioner(G)
        assert torch.allclose(M, M.T, atol=1e-5)

    def test_metric_positive_definite(self):
        """Inverse metric has all eigenvalues > 0 (after center regularization)."""
        from transformer.core.gauge_preconditioner import build_killing_form_preconditioner
        G = _make_glk_generators(3)
        inv_metric = build_killing_form_preconditioner(G)
        # The metric (not inv_metric) should be PD; inv_metric PD iff metric PD
        eigvals = torch.linalg.eigvalsh(inv_metric)
        assert (eigvals > 0).all(), f"Non-positive eigenvalues: {eigvals[eigvals <= 0]}"

    def test_natural_gradient_finite(self):
        """Applying natural gradient produces finite output."""
        from transformer.core.gauge_preconditioner import (
            build_killing_form_preconditioner, apply_killing_form_natural_gradient,
        )
        G = _make_glk_generators(3)
        inv_metric = build_killing_form_preconditioner(G)
        grad = torch.randn(4, 9)
        nat_grad = apply_killing_form_natural_gradient(grad, inv_metric)
        assert torch.isfinite(nat_grad).all()
        assert nat_grad.shape == grad.shape

    def test_natural_gradient_nontrivial(self):
        """Natural gradient ≠ Euclidean gradient (non-identity metric)."""
        from transformer.core.gauge_preconditioner import (
            build_killing_form_preconditioner, apply_killing_form_natural_gradient,
        )
        G = _make_glk_generators(3)
        inv_metric = build_killing_form_preconditioner(G)
        grad = torch.randn(9)
        nat_grad = apply_killing_form_natural_gradient(grad, inv_metric)
        assert not torch.allclose(grad, nat_grad, atol=1e-3)


# ===========================================================================
# TestKillingFormPreconditionerPerBlock — direct-sum gl(d_1) ⊕ ... ⊕ gl(d_H)
# Added 2026-05-19 (audit-2026-05-18-v4, F6.1).
# ===========================================================================

class TestKillingFormPreconditionerPerBlock:
    """Tests for build_killing_form_preconditioner_per_block().

    Validates the direct-sum Killing form on a block-diagonal generator bank:
    block-diagonal metric in the generator index, no cross-block coupling,
    per-block scale ``2·d_h`` (not the ambient ``2·K_full``).
    """

    def test_metric_shape_matches_n_gen(self):
        from transformer.core.gauge_preconditioner import (
            build_killing_form_preconditioner_per_block,
        )
        irrep_dims = [3, 4]
        G = _make_block_diagonal_glk_generators(irrep_dims)
        n_gen = sum(d * d for d in irrep_dims)
        inv_metric, metric = build_killing_form_preconditioner_per_block(
            G, irrep_dims, return_both=True,
        )
        assert inv_metric.shape == (n_gen, n_gen)
        assert metric.shape == (n_gen, n_gen)

    def test_metric_symmetric(self):
        from transformer.core.gauge_preconditioner import (
            build_killing_form_preconditioner_per_block,
        )
        irrep_dims = [3, 4]
        G = _make_block_diagonal_glk_generators(irrep_dims)
        inv_metric, metric = build_killing_form_preconditioner_per_block(
            G, irrep_dims, return_both=True,
        )
        assert torch.allclose(metric, metric.T, atol=1e-5)
        assert torch.allclose(inv_metric, inv_metric.T, atol=1e-5)

    def test_metric_block_diagonal_in_generator_index(self):
        """Cross-block entries are zero. This is the headline property the
        ambient `build_killing_form_preconditioner` violates via -2·tr⊗tr."""
        from transformer.core.gauge_preconditioner import (
            build_killing_form_preconditioner_per_block,
        )
        irrep_dims = [3, 4]
        G = _make_block_diagonal_glk_generators(irrep_dims)
        _, metric = build_killing_form_preconditioner_per_block(
            G, irrep_dims, return_both=True, center_only=True,
        )
        # Generator indices 0..d_1^2-1 are block 0; d_1^2 .. d_1^2+d_2^2-1 are block 1
        n0 = irrep_dims[0] ** 2  # 9
        # Cross-block submatrix metric[0:n0, n0:] must be all zero
        cross_block = metric[:n0, n0:]
        assert torch.allclose(
            cross_block, torch.zeros_like(cross_block), atol=1e-7
        ), (
            f"Per-block Killing metric leaked cross-block coupling: "
            f"max(|cross|)={cross_block.abs().max().item():.3e}"
        )

    def test_within_block_scale_matches_2_dh(self):
        """For a single ``gl(d_h)`` block, the metric reduces to
        ``2·d_h·gram − 2·tr⊗tr + center_reg·(center proj)``. For non-trace
        generators the diagonal entry should be ``2·d_h`` (not ``2·K_full``)
        when ``gram`` is the identity (which it is for the orthonormal
        ``E_{ij}`` basis).
        """
        from transformer.core.gauge_preconditioner import (
            build_killing_form_preconditioner_per_block,
        )
        # Two blocks, dim 3 and dim 4. K_full = 7. The ambient builder would
        # give 2·K_full = 14 along non-trace diagonal; per-block builder gives
        # 2·d_h = 6 for block-0 generators and 2·d_h = 8 for block-1 generators.
        irrep_dims = [3, 4]
        G = _make_block_diagonal_glk_generators(irrep_dims)
        _, metric = build_killing_form_preconditioner_per_block(
            G, irrep_dims, return_both=True, center_only=True,
        )
        # Pick a non-trace generator from block 0: E_{0,1} is at index
        # 0*3 + 1 = 1 in the (3x3) basis.
        # For E_{i,j} with i!=j, tr(E_{i,j}) = 0, so the trace term vanishes
        # and metric[a,a] = 2·d_h·<E_{i,j}, E_{i,j}>_F = 2·d_h·1 = 2·d_h.
        assert math.isclose(metric[1, 1].item(), 2.0 * irrep_dims[0], abs_tol=1e-5)
        # Pick a non-trace generator from block 1: E_{0,1} in (4x4) basis is at
        # offset irrep_dims[0]**2 + 0*4 + 1 = 9 + 1 = 10.
        offset = irrep_dims[0] ** 2 + 1
        assert math.isclose(metric[offset, offset].item(), 2.0 * irrep_dims[1], abs_tol=1e-5)

    def test_metric_positive_definite(self):
        from transformer.core.gauge_preconditioner import (
            build_killing_form_preconditioner_per_block,
        )
        irrep_dims = [3, 4]
        G = _make_block_diagonal_glk_generators(irrep_dims)
        _, metric = build_killing_form_preconditioner_per_block(
            G, irrep_dims, return_both=True,
        )
        eigvals = torch.linalg.eigvalsh(metric)
        assert (eigvals > 0).all(), (
            f"Non-positive eigenvalues in per-block Killing metric: "
            f"{eigvals[eigvals <= 0]}"
        )

    def test_natural_gradient_finite(self):
        from transformer.core.gauge_preconditioner import (
            build_killing_form_preconditioner_per_block,
            apply_killing_form_natural_gradient,
        )
        irrep_dims = [3, 4]
        G = _make_block_diagonal_glk_generators(irrep_dims)
        inv_metric = build_killing_form_preconditioner_per_block(G, irrep_dims)
        n_gen = sum(d * d for d in irrep_dims)
        grad = torch.randn(5, n_gen)
        nat_grad = apply_killing_form_natural_gradient(grad, inv_metric)
        assert torch.isfinite(nat_grad).all()
        assert nat_grad.shape == grad.shape

    def test_natural_gradient_does_not_couple_blocks(self):
        """A gradient that is non-zero only on block-0 generators should
        produce a natural gradient that is also non-zero only on block-0
        generators — direct-sum structure preserves block independence."""
        from transformer.core.gauge_preconditioner import (
            build_killing_form_preconditioner_per_block,
            apply_killing_form_natural_gradient,
        )
        irrep_dims = [3, 4]
        G = _make_block_diagonal_glk_generators(irrep_dims)
        inv_metric = build_killing_form_preconditioner_per_block(G, irrep_dims)
        n_gen = sum(d * d for d in irrep_dims)
        n0 = irrep_dims[0] ** 2
        grad = torch.zeros(n_gen)
        grad[:n0] = torch.randn(n0)  # only block 0 has gradient
        nat_grad = apply_killing_form_natural_gradient(grad, inv_metric)
        # Block-1 components of the natural gradient must be zero.
        assert torch.allclose(
            nat_grad[n0:], torch.zeros(n_gen - n0), atol=1e-6
        ), (
            f"Direct-sum Killing-form natural gradient leaked into block 1: "
            f"max(|nat_grad[block1]|)={nat_grad[n0:].abs().max().item():.3e}"
        )

    def test_raises_on_straddling_generator(self):
        """A generator whose Frobenius mass spans multiple blocks must
        cause the builder to refuse — direct-sum Killing form is undefined."""
        from transformer.core.gauge_preconditioner import (
            build_killing_form_preconditioner_per_block,
        )
        irrep_dims = [3, 4]
        # Start with a clean block-diagonal bank, then add a straddling generator
        G = _make_block_diagonal_glk_generators(irrep_dims)
        K_full = sum(irrep_dims)
        straddler = torch.zeros(1, K_full, K_full)
        straddler[0, 0, irrep_dims[0]] = 1.0  # entry (0, d_1) — block 0 row, block 1 col
        G_bad = torch.cat([G, straddler], dim=0)
        with pytest.raises(ValueError, match="straddle blocks|do not have"):
            build_killing_form_preconditioner_per_block(G_bad, irrep_dims)

    def test_raises_on_irrep_dims_sum_mismatch(self):
        from transformer.core.gauge_preconditioner import (
            build_killing_form_preconditioner_per_block,
        )
        G = _make_block_diagonal_glk_generators([3, 4])
        with pytest.raises(ValueError, match="sum\\(irrep_dims\\)"):
            build_killing_form_preconditioner_per_block(G, [3, 5])  # 8 != 7

    def test_differs_from_ambient_killing_form(self):
        """Per-block metric must differ from the ambient `build_killing_form_
        preconditioner` on the same generator bank — the whole point of the
        toggle is that the two metrics are inequivalent."""
        from transformer.core.gauge_preconditioner import (
            build_killing_form_preconditioner,
            build_killing_form_preconditioner_per_block,
        )
        irrep_dims = [3, 4]
        G = _make_block_diagonal_glk_generators(irrep_dims)
        _, m_ambient = build_killing_form_preconditioner(G, return_both=True)
        _, m_perblock = build_killing_form_preconditioner_per_block(
            G, irrep_dims, return_both=True,
        )
        assert not torch.allclose(m_ambient, m_perblock, atol=1e-4), (
            "Per-block and ambient Killing metrics agreed numerically — the "
            "toggle is meaningless on this generator bank."
        )


# ===========================================================================
# TestPullbackNaturalGradient
# ===========================================================================

class TestPullbackNaturalGradient:
    """Tests for build_structure_constants() and apply_pullback_natural_gradient()."""

    def test_structure_constants_antisymmetry(self):
        """f^c_{ab} = -f^c_{ba} (antisymmetry in lower indices)."""
        from transformer.core.gauge_preconditioner import build_structure_constants
        G = _make_so_generators(3)  # 3 generators for so(3)
        f = build_structure_constants(G)
        # f[a,b,c] should equal -f[b,a,c]
        assert torch.allclose(f, -f.transpose(0, 1), atol=1e-5), \
            f"Structure constants not antisymmetric: max diff {(f + f.transpose(0, 1)).abs().max():.2e}"

    def test_metric_at_origin_equals_gram(self):
        """G(φ=0) = gram matrix (Ψ(0) = I)."""
        from transformer.core.gauge_preconditioner import (
            build_structure_constants, apply_pullback_natural_gradient,
        )
        G = _make_so_generators(3)
        f = build_structure_constants(G)
        gram = torch.einsum('aij,bij->ab', G, G)

        phi = torch.zeros(3)  # origin
        grad = torch.randn(3)

        # At φ=0, natural gradient = gram^{-1} @ grad
        nat_grad = apply_pullback_natural_gradient(grad, phi, G, f, gram=gram)
        expected = torch.linalg.solve(gram, grad)
        assert torch.allclose(nat_grad, expected, atol=1e-4), \
            f"max diff: {(nat_grad - expected).abs().max():.2e}"

    def test_metric_positive_definite_random_phi(self):
        """G(φ) has all eigenvalues > 0 for random φ."""
        from transformer.core.gauge_preconditioner import build_structure_constants
        G = _make_so_generators(3)
        f = build_structure_constants(G)
        gram = torch.einsum('aij,bij->ab', G, G)
        n_gen = G.shape[0]

        phi = torch.randn(n_gen) * 2.0
        ad_X = torch.einsum('a,abc->bc', phi, f)
        I = torch.eye(n_gen)
        psi = I.clone()
        ad_power = ad_X.clone()
        for k in range(1, 6):
            psi = psi + (1.0 / math.factorial(k + 1)) * ad_power
            if k < 5:
                ad_power = ad_power @ ad_X
        metric = psi.T @ gram @ psi + 1e-6 * I
        eigvals = torch.linalg.eigvalsh(metric)
        assert (eigvals > 0).all(), f"Non-positive eigenvalue: {eigvals.min():.2e}"

    def test_gradient_finite_for_large_phi(self):
        """Natural gradient doesn't explode for ||φ|| ~ 5."""
        from transformer.core.gauge_preconditioner import (
            build_structure_constants, apply_pullback_natural_gradient,
        )
        G = _make_so_generators(3)
        f = build_structure_constants(G)
        phi = torch.randn(3) * 5.0
        grad = torch.randn(3)
        nat_grad = apply_pullback_natural_gradient(grad, phi, G, f)
        assert torch.isfinite(nat_grad).all()

    def test_batched_pullback(self):
        """Batched phi (..., n_gen) works."""
        from transformer.core.gauge_preconditioner import (
            build_structure_constants, apply_pullback_natural_gradient,
        )
        G = _make_so_generators(3)
        f = build_structure_constants(G)
        phi = torch.randn(4, 3) * 1.0
        grad = torch.randn(4, 3)
        nat_grad = apply_pullback_natural_gradient(grad, phi, G, f)
        assert nat_grad.shape == (4, 3)
        assert torch.isfinite(nat_grad).all()

    def test_metric_matches_frobenius_pullback_noncompact(self):
        """The metric G(φ) matches the direct Frobenius pullback of D_φ exp.

        For non-compact φ ∈ gl(K), the position-dependent factor
        ``exp(φ) exp(φ)^T`` is non-trivial and must be included. This test
        verifies the corrected formula in `build_pullback_metric_tensor`
        against the canonical Frobenius pullback computed directly from
        Higham 2008's block-matrix identity for D_φ exp[T_a].

        Uses the full elementary basis ``{E_{ij}}`` of gl(K) so the basis is
        closed under the Lie bracket. A partial basis (e.g., sym + skew alone
        without sym off-diagonal) would leave commutators outside the
        spanning subspace and break the structure-constant projection.
        """
        from transformer.core.gauge_preconditioner import (
            build_structure_constants, build_pullback_metric_tensor,
        )
        K = 3
        # Full gl(K) elementary basis: E_{ij}, n_gen = K². Closed under [·,·].
        gens = []
        for i in range(K):
            for j in range(K):
                T = torch.zeros(K, K)
                T[i, j] = 1.0
                gens.append(T)
        gens = torch.stack(gens)  # (K², K, K) = (9, 3, 3) for K=3.
        n_gen = gens.shape[0]
        struct = build_structure_constants(gens)

        torch.manual_seed(42)
        # Small ||phi|| so the Taylor truncation in build_pullback_metric_tensor
        # converges to machine precision for the test comparison.
        phi = torch.randn(n_gen) * 0.1
        G_code = build_pullback_metric_tensor(
            phi, gens, struct, series_order=20,
        )

        # Direct Frobenius pullback via Higham 2008 block-matrix identity:
        #   exp([[phi, T_a], [0, phi]]) = [[exp(phi), D_phi exp[T_a]], [0, exp(phi)]]
        # The (1,2) block is the full Frechet derivative D_phi exp[T_a].
        phi_mat = torch.einsum('a,aij->ij', phi, gens)  # (K, K)
        D_exp = torch.zeros(n_gen, K, K)
        for a in range(n_gen):
            block = torch.zeros(2 * K, 2 * K)
            block[:K, :K] = phi_mat
            block[K:, K:] = phi_mat
            block[:K, K:] = gens[a]
            block_exp = torch.matrix_exp(block)
            D_exp[a] = block_exp[:K, K:]
        # Direct Frobenius metric: G_canon[a,b] = tr(D_exp[a]^T @ D_exp[b]).
        G_canon = torch.einsum('aij,bij->ab', D_exp, D_exp)
        # Compare without the 1e-6 * I regularizer.
        G_clean = G_code - 1e-6 * torch.eye(n_gen)
        torch.testing.assert_close(G_clean, G_canon, atol=1e-3, rtol=1e-3)

    def test_metric_compact_reduces_to_psi_gram_psi(self):
        """On the compact subalgebra so(K), the new metric reduces to Ψ^T gram Ψ.

        For skew-symmetric ``phi``, exp(phi) ∈ SO(K) so ``exp(phi) exp(phi)^T = I``
        and the position-dependent factor H(phi) = gram. The new metric must
        match the legacy ``Ψ^T gram Ψ`` form bit-exactly on this subalgebra.
        """
        from transformer.core.gauge_preconditioner import (
            build_structure_constants, build_pullback_metric_tensor,
        )
        gens = _make_so_generators(3)
        struct = build_structure_constants(gens)
        gram = torch.einsum('aij,bij->ab', gens, gens)

        torch.manual_seed(7)
        phi = torch.randn(3) * 0.4  # skew-only — so(3) only has skew generators
        G_code = build_pullback_metric_tensor(
            phi, gens, struct, series_order=10,
        )

        # Reconstruct the legacy form: Ψ^T gram Ψ + 1e-6 I.
        ad = torch.einsum('a,abc->bc', phi, struct)
        I = torch.eye(3)
        psi = I.clone()
        ad_power = ad.clone()
        for k in range(1, 10):
            psi = psi + (1.0 / math.factorial(k + 1)) * ad_power
            if k < 9:
                ad_power = ad_power @ ad
        G_legacy = psi.T @ gram @ psi + 1e-6 * I

        torch.testing.assert_close(G_code, G_legacy, atol=1e-5, rtol=1e-5)


# ===========================================================================
# TestSLKProjection
# ===========================================================================

class TestSLKProjection:
    """Tests for build_slk_projector() and apply_slk_projection()."""

    def test_trace_zero_after_projection(self):
        """tr(Σ φ_a T_a) = 0 after SL(K) projection."""
        from transformer.core.gauge_preconditioner import (
            build_slk_projector, apply_slk_projection,
        )
        G = _make_glk_generators(3)
        trace_vec = build_slk_projector(G)
        phi = torch.randn(5, 9)
        phi_proj = apply_slk_projection(phi, trace_vec)
        # tr(M) = v^T φ should be zero
        traces = phi_proj @ trace_vec
        assert torch.allclose(traces, torch.zeros_like(traces), atol=1e-6)

    def test_projection_idempotent(self):
        """project(project(φ)) = project(φ)."""
        from transformer.core.gauge_preconditioner import (
            build_slk_projector, apply_slk_projection,
        )
        G = _make_glk_generators(3)
        trace_vec = build_slk_projector(G)
        phi = torch.randn(9)
        phi_once = apply_slk_projection(phi, trace_vec)
        phi_twice = apply_slk_projection(phi_once, trace_vec)
        assert torch.allclose(phi_once, phi_twice, atol=1e-6)

    def test_det_exp_equals_one(self):
        """det(exp(projected_phi · G)) ≈ 1 since tr(M)=0 → det(exp(M))=exp(0)=1."""
        from transformer.core.gauge_preconditioner import (
            build_slk_projector, apply_slk_projection,
        )
        G = _make_glk_generators(3)
        trace_vec = build_slk_projector(G)
        phi = torch.randn(9) * 0.5
        phi_proj = apply_slk_projection(phi, trace_vec)
        # Construct M = Σ φ_a T_a
        M = torch.einsum('a,aij->ij', phi_proj, G)
        det_exp = torch.linalg.det(torch.linalg.matrix_exp(M))
        assert torch.allclose(det_exp, torch.ones(1), atol=1e-4), \
            f"det(exp(M)) = {det_exp.item():.6f}, expected 1.0"

    def test_traceless_generators_unchanged(self):
        """For so(K) generators (already traceless), projection is identity."""
        from transformer.core.gauge_preconditioner import (
            build_slk_projector, apply_slk_projection,
        )
        G = _make_so_generators(4)
        trace_vec = build_slk_projector(G)
        # All so(K) generators are traceless → trace_vec = 0
        assert torch.allclose(trace_vec, torch.zeros_like(trace_vec), atol=1e-6)
