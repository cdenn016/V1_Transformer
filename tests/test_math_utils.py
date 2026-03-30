"""
Tests for math_utils/ modules
===============================

Tests generators.py, transport.py, push_pull.py, and numerical_utils.py.
These are the foundation layer; bugs here silently corrupt everything above.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose


# ===========================================================================
# TestSO3Generators
# ===========================================================================

class TestSO3Generators:
    """Tests for generate_so3_generators()."""

    def test_skew_symmetric(self):
        """G^T = -G for all generators."""
        from math_utils.generators import generate_so3_generators
        for K in [3, 5, 7]:
            G = generate_so3_generators(K)
            for a in range(3):
                assert_allclose(G[a].T, -G[a], atol=1e-6,
                                err_msg=f"K={K}, generator {a} not skew-symmetric")

    def test_commutation_relations_cyclic(self):
        """[G_x, G_y] = G_z, [G_y, G_z] = G_x, [G_z, G_x] = G_y."""
        from math_utils.generators import generate_so3_generators
        for K in [3, 5, 7]:
            G = generate_so3_generators(K)
            # [G_0, G_1] = G_2
            comm_01 = G[0] @ G[1] - G[1] @ G[0]
            assert_allclose(comm_01, G[2], atol=1e-5,
                            err_msg=f"K={K}: [G_x, G_y] ≠ G_z")
            # [G_1, G_2] = G_0
            comm_12 = G[1] @ G[2] - G[2] @ G[1]
            assert_allclose(comm_12, G[0], atol=1e-5,
                            err_msg=f"K={K}: [G_y, G_z] ≠ G_x")
            # [G_2, G_0] = G_1
            comm_20 = G[2] @ G[0] - G[0] @ G[2]
            assert_allclose(comm_20, G[1], atol=1e-5,
                            err_msg=f"K={K}: [G_z, G_x] ≠ G_y")

    def test_casimir_eigenvalue(self):
        """-Σ G_a² = ℓ(ℓ+1)·I where ℓ = (K-1)/2."""
        from math_utils.generators import generate_so3_generators
        for K in [3, 5, 7]:
            G = generate_so3_generators(K)
            ell = (K - 1) / 2.0
            C2 = -sum(G[a] @ G[a] for a in range(3))
            expected = ell * (ell + 1) * np.eye(K)
            assert_allclose(C2, expected, atol=1e-5,
                            err_msg=f"K={K}: Casimir ≠ {ell}({ell}+1)I")

    def test_even_K_raises(self):
        """Even K should raise ValueError."""
        from math_utils.generators import generate_so3_generators
        with pytest.raises(ValueError, match="odd"):
            generate_so3_generators(4)

    def test_cache_returns_copy(self):
        """Modifying returned generators doesn't corrupt cache."""
        from math_utils.generators import generate_so3_generators
        G1 = generate_so3_generators(3)
        G1[0, 0, 0] = 999.0  # corrupt the returned array
        G2 = generate_so3_generators(3)
        assert G2[0, 0, 0] != 999.0, "Cache was corrupted"

    def test_shape(self):
        """Output shape is (3, K, K)."""
        from math_utils.generators import generate_so3_generators
        for K in [3, 5, 7, 9]:
            G = generate_so3_generators(K)
            assert G.shape == (3, K, K)

    def test_orthonormality(self):
        """tr(G_a^T G_b) is diagonal (generators are orthogonal under Frobenius)."""
        from math_utils.generators import generate_so3_generators
        G = generate_so3_generators(5)
        gram = np.einsum('aij,bij->ab', G, G)
        # Off-diagonal should be ~zero
        off_diag = gram - np.diag(np.diag(gram))
        assert_allclose(off_diag, 0.0, atol=1e-5)


# ===========================================================================
# TestTransportOperators
# ===========================================================================

class TestTransportOperators:
    """Tests for compute_transport()."""

    def _get_generators(self, K=3):
        from math_utils.generators import generate_so3_generators
        return generate_so3_generators(K)

    def test_cocycle_condition(self):
        """Ω_ij · Ω_jk = Ω_ik (transitivity)."""
        from math_utils.transport import compute_transport
        G = self._get_generators(3)
        phi_i = np.random.randn(3) * 0.5
        phi_j = np.random.randn(3) * 0.5
        phi_k = np.random.randn(3) * 0.5
        Omega_ij = compute_transport(phi_i, phi_j, G)
        Omega_jk = compute_transport(phi_j, phi_k, G)
        Omega_ik = compute_transport(phi_i, phi_k, G)
        product = Omega_ij @ Omega_jk
        assert_allclose(product, Omega_ik, atol=1e-5,
                        err_msg="Cocycle condition violated")

    def test_self_transport_identity(self):
        """Ω_ii = I."""
        from math_utils.transport import compute_transport
        G = self._get_generators(3)
        phi = np.random.randn(3) * 0.5
        Omega = compute_transport(phi, phi, G)
        assert_allclose(Omega, np.eye(3), atol=1e-6)

    def test_det_positive(self):
        """det(Ω_ij) > 0 — stays in GL⁺(K)."""
        from math_utils.transport import compute_transport
        G = self._get_generators(5)
        for _ in range(10):
            phi_i = np.random.randn(3)
            phi_j = np.random.randn(3)
            Omega = compute_transport(phi_i, phi_j, G)
            det = np.linalg.det(Omega)
            assert det > 0, f"det(Ω) = {det:.4e} ≤ 0"

    def test_inverse_relation(self):
        """Ω_ij · Ω_ji = I."""
        from math_utils.transport import compute_transport
        G = self._get_generators(3)
        phi_i = np.random.randn(3) * 0.5
        phi_j = np.random.randn(3) * 0.5
        Omega_ij = compute_transport(phi_i, phi_j, G)
        Omega_ji = compute_transport(phi_j, phi_i, G)
        product = Omega_ij @ Omega_ji
        assert_allclose(product, np.eye(3), atol=1e-5)

    def test_zero_phi_gives_identity(self):
        """φ_i = φ_j = 0 → Ω = I."""
        from math_utils.transport import compute_transport
        G = self._get_generators(3)
        Omega = compute_transport(np.zeros(3), np.zeros(3), G)
        assert_allclose(Omega, np.eye(3), atol=1e-8)

    def test_so3_transport_is_orthogonal(self):
        """For SO(3) generators, Ω should be orthogonal: Ω^T Ω = I."""
        from math_utils.transport import compute_transport
        G = self._get_generators(3)
        phi_i = np.random.randn(3) * 1.0
        phi_j = np.random.randn(3) * 1.0
        Omega = compute_transport(phi_i, phi_j, G)
        assert_allclose(Omega.T @ Omega, np.eye(3), atol=1e-4,
                        err_msg="SO(3) transport not orthogonal")

    def test_batched_transport(self):
        """Batched phi (*S, n_gen) produces batched Ω."""
        from math_utils.transport import compute_transport
        G = self._get_generators(3)
        phi_i = np.random.randn(5, 3) * 0.5
        phi_j = np.random.randn(5, 3) * 0.5
        Omega = compute_transport(phi_i, phi_j, G)
        assert Omega.shape == (5, 3, 3)


# ===========================================================================
# TestGaussianPushforward
# ===========================================================================

class TestGaussianPushforward:
    """Tests for push_gaussian()."""

    def test_mean_transport(self):
        """μ_transported = Ω @ μ."""
        from math_utils.push_pull import push_gaussian, GaussianDistribution
        K = 3
        mu = np.array([1.0, 2.0, 3.0])
        Sigma = np.eye(K, dtype=np.float32)
        q = GaussianDistribution(mu, Sigma)
        # 90° rotation about z-axis
        angle = np.pi / 2
        Omega = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle),  np.cos(angle), 0],
            [0, 0, 1]
        ], dtype=np.float64)
        q_pushed = push_gaussian(q, Omega)
        expected_mu = Omega.astype(np.float32) @ mu.astype(np.float32)
        assert_allclose(q_pushed.mu, expected_mu, atol=1e-4)

    def test_covariance_sandwich(self):
        """Σ_transported = Ω @ Σ @ Ω^T (THE hard constraint)."""
        from math_utils.push_pull import push_gaussian, GaussianDistribution
        K = 3
        mu = np.zeros(K)
        A = np.random.randn(K, K).astype(np.float64) * 0.5
        Sigma = (A @ A.T + 0.5 * np.eye(K)).astype(np.float32)
        q = GaussianDistribution(mu, Sigma)

        Omega = np.array([
            [2.0, 0.5, 0.0],
            [0.0, 1.5, 0.3],
            [0.1, 0.0, 1.0]
        ], dtype=np.float64)
        q_pushed = push_gaussian(q, Omega)

        expected_Sigma = Omega @ Sigma.astype(np.float64) @ Omega.T
        # push_gaussian applies eigenvalue floor, so compare with tolerance
        assert_allclose(q_pushed.Sigma, expected_Sigma.astype(np.float32), atol=5e-3,
                        err_msg="Covariance sandwich product violated")

    def test_covariance_stays_spd(self):
        """Transported Σ has all eigenvalues > 0."""
        from math_utils.push_pull import push_gaussian, GaussianDistribution
        K = 4
        mu = np.zeros(K)
        A = np.random.randn(K, K).astype(np.float64)
        Sigma = (A @ A.T + 0.1 * np.eye(K)).astype(np.float32)
        q = GaussianDistribution(mu, Sigma)
        # Random invertible Omega
        Omega = np.random.randn(K, K).astype(np.float64)
        Omega = Omega + 2.0 * np.eye(K)  # ensure invertible
        q_pushed = push_gaussian(q, Omega)
        eigvals = np.linalg.eigvalsh(q_pushed.Sigma)
        assert (eigvals > 0).all(), f"Non-positive eigenvalues: {eigvals}"

    def test_identity_transport_unchanged(self):
        """Ω = I → μ and Σ unchanged."""
        from math_utils.push_pull import push_gaussian, GaussianDistribution
        K = 3
        mu = np.array([1.0, 2.0, 3.0])
        Sigma = np.diag([1.0, 2.0, 3.0]).astype(np.float32)
        q = GaussianDistribution(mu, Sigma)
        q_pushed = push_gaussian(q, np.eye(K))
        assert_allclose(q_pushed.mu, q.mu, atol=1e-5)
        assert_allclose(q_pushed.Sigma, q.Sigma, atol=1e-5)


# ===========================================================================
# TestKLGaussianNumpy
# ===========================================================================

class TestKLGaussianNumpy:
    """Tests for kl_gaussian() from numerical_utils.py."""

    def test_kl_non_negative(self):
        """KL(q||p) ≥ 0."""
        from math_utils.numerical_utils import kl_gaussian
        K = 3
        for _ in range(10):
            A = np.random.randn(K, K)
            Sigma_q = A @ A.T + 0.5 * np.eye(K)
            B = np.random.randn(K, K)
            Sigma_p = B @ B.T + 0.5 * np.eye(K)
            mu_q = np.random.randn(K)
            mu_p = np.random.randn(K)
            kl = kl_gaussian(mu_q, Sigma_q, mu_p, Sigma_p)
            assert kl >= 0, f"KL = {kl:.4e} < 0"

    def test_kl_zero_same_distribution(self):
        """KL(p||p) = 0."""
        from math_utils.numerical_utils import kl_gaussian
        K = 4
        mu = np.random.randn(K)
        A = np.random.randn(K, K)
        Sigma = A @ A.T + 0.5 * np.eye(K)
        kl = kl_gaussian(mu, Sigma, mu, Sigma)
        assert_allclose(kl, 0.0, atol=1e-5)

    def test_kl_term_breakdown(self):
        """return_terms gives trace + quad + logdet terms."""
        from math_utils.numerical_utils import kl_gaussian
        K = 3
        mu_q = np.random.randn(K)
        mu_p = np.random.randn(K)
        A = np.random.randn(K, K)
        Sigma_q = A @ A.T + 0.5 * np.eye(K)
        B = np.random.randn(K, K)
        Sigma_p = B @ B.T + 0.5 * np.eye(K)
        kl, terms = kl_gaussian(mu_q, Sigma_q, mu_p, Sigma_p, return_terms=True)
        assert 'term_trace' in terms
        assert 'term_quad' in terms
        assert 'term_logdet' in terms
        # Verify: KL = 0.5 * (trace + quad - K + logdet)
        reconstructed = 0.5 * (terms['term_trace'] + terms['term_quad'] - K + terms['term_logdet'])
        assert_allclose(kl, np.clip(reconstructed, 0, None).astype(np.float32), atol=1e-4)

    def test_known_value_1d(self):
        """Known analytic KL for 1D Gaussians."""
        from math_utils.numerical_utils import kl_gaussian
        # KL(N(0,1) || N(1,2)) = 0.5 * (1/2 + 1/2 - 1 + ln(2)) = 0.5 * ln(2) ≈ 0.3466
        mu_q = np.array([0.0])
        Sigma_q = np.array([[1.0]])
        mu_p = np.array([1.0])
        Sigma_p = np.array([[2.0]])
        kl = kl_gaussian(mu_q, Sigma_q, mu_p, Sigma_p)
        expected = 0.5 * (1.0/2.0 + 1.0/2.0 - 1.0 + np.log(2.0))
        assert_allclose(float(kl), expected, atol=1e-4)


# ===========================================================================
# TestSafeInv
# ===========================================================================

class TestSafeInv:
    """Tests for safe_inv()."""

    def test_inverse_correctness(self):
        """A @ A^{-1} ≈ I."""
        from math_utils.numerical_utils import safe_inv
        K = 4
        A = np.random.randn(K, K)
        Sigma = A @ A.T + 0.5 * np.eye(K)
        Sigma_inv = safe_inv(Sigma)
        product = Sigma @ Sigma_inv
        assert_allclose(product, np.eye(K), atol=1e-4)

    def test_batched_inverse(self):
        """Batched (..., K, K) input works."""
        from math_utils.numerical_utils import safe_inv
        K = 3
        batch = 5
        A = np.random.randn(batch, K, K)
        Sigma = A @ np.swapaxes(A, -1, -2) + 0.5 * np.eye(K)
        Sigma_inv = safe_inv(Sigma)
        assert Sigma_inv.shape == (batch, K, K)
        for i in range(batch):
            product = Sigma[i] @ Sigma_inv[i]
            assert_allclose(product, np.eye(K), atol=1e-4)


# ===========================================================================
# TestSanitizeSigma
# ===========================================================================

class TestSanitizeSigma:
    """Tests for sanitize_sigma()."""

    def test_output_is_spd(self):
        """Sanitized matrix has all eigenvalues > 0."""
        from math_utils.numerical_utils import sanitize_sigma
        K = 4
        # Create near-singular matrix
        Sigma = np.diag([1.0, 0.5, 1e-10, -0.01])
        Sigma_clean = sanitize_sigma(Sigma)
        eigvals = np.linalg.eigvalsh(Sigma_clean)
        assert (eigvals > 0).all(), f"Non-positive eigenvalues: {eigvals}"

    def test_output_is_symmetric(self):
        """Sanitized matrix is symmetric."""
        from math_utils.numerical_utils import sanitize_sigma
        K = 3
        A = np.random.randn(K, K)
        Sigma = A @ A.T + 0.1 * np.eye(K)
        # Add slight asymmetry
        Sigma[0, 1] += 0.01
        Sigma_clean = sanitize_sigma(Sigma)
        assert_allclose(Sigma_clean, Sigma_clean.T, atol=1e-10)

    def test_already_good_sigma_unchanged(self):
        """Well-conditioned SPD matrix is nearly unchanged."""
        from math_utils.numerical_utils import sanitize_sigma
        K = 3
        Sigma = np.diag([1.0, 2.0, 3.0])
        Sigma_clean = sanitize_sigma(Sigma)
        assert_allclose(Sigma_clean, Sigma, atol=1e-4)

    def test_batched_sanitize(self):
        """Batched input (..., K, K) works."""
        from math_utils.numerical_utils import sanitize_sigma
        K = 3
        batch = 4
        A = np.random.randn(batch, K, K)
        Sigma = A @ np.swapaxes(A, -1, -2) + 0.1 * np.eye(K)
        Sigma_clean = sanitize_sigma(Sigma)
        assert Sigma_clean.shape == (batch, K, K)
        for i in range(batch):
            eigvals = np.linalg.eigvalsh(Sigma_clean[i])
            assert (eigvals > 0).all()


# ===========================================================================
# TestFrechetExpm
# ===========================================================================

class TestFrechetExpm:
    """Tests for frechet_expm() — Fréchet derivative of matrix exponential."""

    def test_matches_finite_difference(self):
        """Fréchet derivative ≈ (exp(X+εH) - exp(X)) / ε."""
        from math_utils.transport import frechet_expm
        from scipy.linalg import expm
        K = 3
        X = np.random.randn(K, K) * 0.5
        H = np.random.randn(K, K) * 0.5
        eps = 1e-6
        # Finite difference
        fd = (expm(X + eps * H) - expm(X)) / eps
        # Fréchet
        frechet = frechet_expm(X, H, steps=16)
        assert_allclose(frechet, fd, atol=1e-3,
                        err_msg="Fréchet derivative doesn't match finite difference")

    def test_identity_direction(self):
        """dexp_0(H) = H (at origin, Fréchet is identity)."""
        from math_utils.transport import frechet_expm
        K = 3
        X = np.zeros((K, K))
        H = np.random.randn(K, K)
        result = frechet_expm(X, H, steps=16)
        assert_allclose(result, H, atol=1e-4)
