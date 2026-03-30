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
