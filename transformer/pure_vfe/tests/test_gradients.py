"""
Finite-difference gradient validation for all analytic gradients.

Tests:
  1. ∂KL/∂μ_i — mean gradient
  2. ∂KL/∂Σ_i — covariance gradient
  3. ∂KL/∂Ω_ij — gauge transport gradient
  4. GL(K) invariance — KL(G·P || G·Ω·Q) = KL(P || Ω·Q)
  5. E-step monotonicity — VFE decreases
  6. SPD retraction — Σ stays SPD
  7. Pairwise KL precomputation — matches direct computation
"""

import torch
import pytest

# Use CPU for tests (no CUDA requirement)
DEVICE = 'cpu'
K = 4   # Small K for fast finite-diff
EPS = 1e-5
REL_TOL = 1e-3


def random_spd(K, batch_shape=()):
    """Generate random SPD matrix."""
    A = torch.randn(*batch_shape, K, K)
    return A @ A.transpose(-2, -1) + 0.1 * torch.eye(K)


def random_gl(K, batch_shape=(), scale=0.3):
    """Generate random GL(K) matrix near identity."""
    return torch.eye(K).expand(*batch_shape, K, K).clone() + scale * torch.randn(*batch_shape, K, K)


class TestKLGradients:
    """Test analytic KL gradients against finite differences."""

    def setup_method(self):
        torch.manual_seed(42)
        self.mu_i = torch.randn(K)
        self.mu_j = torch.randn(K)
        self.Sigma_i = random_spd(K)
        self.Sigma_j = random_spd(K)
        self.Omega_ij = random_gl(K)

    def test_kl_grad_mu_i(self):
        """∂KL(q_i || Ω_ij · q_j)/∂μ_i vs finite differences."""
        from ..gaussians import kl_divergence
        from ..gauge import transport_gaussian

        mu_t, Sigma_t = transport_gaussian(self.mu_j, self.Sigma_j, self.Omega_ij)
        Sigma_t_inv = torch.linalg.inv(Sigma_t)

        # Analytic: Σ_Q⁻¹(μ_i - μ_Q)
        analytic = Sigma_t_inv @ (self.mu_i - mu_t)

        # Finite difference
        fd = torch.zeros(K)
        for d in range(K):
            mu_plus = self.mu_i.clone()
            mu_plus[d] += EPS
            mu_minus = self.mu_i.clone()
            mu_minus[d] -= EPS
            kl_plus = kl_divergence(mu_plus, self.Sigma_i, mu_t, Sigma_t)
            kl_minus = kl_divergence(mu_minus, self.Sigma_i, mu_t, Sigma_t)
            fd[d] = (kl_plus - kl_minus) / (2 * EPS)

        rel_err = (analytic - fd).norm() / fd.norm().clamp(min=1e-8)
        assert rel_err < REL_TOL, f"mu_i grad rel_err = {rel_err:.6f}"

    def test_kl_grad_Sigma_i(self):
        """∂KL(q_i || Ω_ij · q_j)/∂Σ_i vs finite differences."""
        from ..gaussians import kl_divergence, safe_inverse
        from ..gauge import transport_gaussian

        mu_t, Sigma_t = transport_gaussian(self.mu_j, self.Sigma_j, self.Omega_ij)
        Sigma_t_inv = torch.linalg.inv(Sigma_t)
        Sigma_i_inv = torch.linalg.inv(self.Sigma_i)

        # Analytic: ½(Σ_Q⁻¹ - Σ_i⁻¹)
        analytic = 0.5 * (Sigma_t_inv - Sigma_i_inv)

        # Finite difference (symmetric perturbations)
        fd = torch.zeros(K, K)
        for a in range(K):
            for b in range(a, K):
                # Symmetric perturbation
                E = torch.zeros(K, K)
                E[a, b] = E[b, a] = EPS
                kl_plus = kl_divergence(self.mu_i, self.Sigma_i + E, mu_t, Sigma_t)
                kl_minus = kl_divergence(self.mu_i, self.Sigma_i - E, mu_t, Sigma_t)
                val = (kl_plus - kl_minus) / (2 * EPS)
                fd[a, b] = val
                fd[b, a] = val

        # For symmetric matrix gradient, need to account for the symmetric constraint
        # The gradient w.r.t. Σ when Σ is symmetric is: diag gets factor 1, off-diag gets 2
        analytic_sym = analytic.clone()
        mask = ~torch.eye(K, dtype=torch.bool)
        fd_sym = fd.clone()

        rel_err = (analytic_sym - fd_sym).norm() / fd_sym.norm().clamp(min=1e-8)
        assert rel_err < REL_TOL, f"Sigma_i grad rel_err = {rel_err:.6f}"

    def test_kl_grad_Omega_ij(self):
        """∂KL(q_i || Ω_ij · q_j)/∂Ω_ij vs finite differences."""
        from ..gaussians import kl_divergence
        from ..gauge import transport_gaussian, grad_kl_Omega_ij

        analytic = grad_kl_Omega_ij(
            self.mu_i, self.Sigma_i, self.mu_j, self.Sigma_j, self.Omega_ij
        )

        fd = torch.zeros(K, K)
        for a in range(K):
            for b in range(K):
                E = torch.zeros(K, K)
                E[a, b] = EPS
                Om_plus = self.Omega_ij + E
                Om_minus = self.Omega_ij - E
                mu_tp, Sig_tp = transport_gaussian(self.mu_j, self.Sigma_j, Om_plus)
                mu_tm, Sig_tm = transport_gaussian(self.mu_j, self.Sigma_j, Om_minus)
                kl_p = kl_divergence(self.mu_i, self.Sigma_i, mu_tp, Sig_tp)
                kl_m = kl_divergence(self.mu_i, self.Sigma_i, mu_tm, Sig_tm)
                fd[a, b] = (kl_p - kl_m) / (2 * EPS)

        rel_err = (analytic - fd).norm() / fd.norm().clamp(min=1e-8)
        assert rel_err < REL_TOL, f"Omega_ij grad rel_err = {rel_err:.6f}"


class TestGLInvariance:
    """Test GL(K) gauge invariance of KL divergence."""

    def test_gl_invariance(self):
        """KL(G·P || G·Ω·Q) = KL(P || Ω·Q) for random G ∈ GL(K)."""
        from ..gaussians import kl_divergence
        from ..gauge import transport_gaussian

        torch.manual_seed(123)
        mu_i = torch.randn(K)
        mu_j = torch.randn(K)
        Sigma_i = random_spd(K)
        Sigma_j = random_spd(K)
        Omega = random_gl(K)
        G = random_gl(K, scale=0.5)

        # Original KL
        mu_t, Sig_t = transport_gaussian(mu_j, Sigma_j, Omega)
        kl_orig = kl_divergence(mu_i, Sigma_i, mu_t, Sig_t)

        # Gauge-transformed KL
        mu_i_g, Sig_i_g = transport_gaussian(mu_i, Sigma_i, G)
        Omega_g = G @ Omega @ torch.linalg.inv(G)
        mu_j_g, Sig_j_g = transport_gaussian(mu_j, Sigma_j, G)
        mu_tg, Sig_tg = transport_gaussian(mu_j_g, Sig_j_g, Omega_g)
        kl_gauged = kl_divergence(mu_i_g, Sig_i_g, mu_tg, Sig_tg)

        diff = abs(kl_orig.item() - kl_gauged.item())
        assert diff < 1e-4, f"GL(K) invariance violated: diff = {diff:.6e}"

    def test_cocycle_condition(self):
        """Ω_ij · Ω_jk = Ω_ik (automatic from Ω_ij = Ω_i · Ω_j⁻¹)."""
        from ..gauge import compute_transport

        torch.manual_seed(456)
        Omega_i = random_gl(K)
        Omega_j = random_gl(K)
        Omega_k = random_gl(K)

        Omega_ij = compute_transport(Omega_i, Omega_j)
        Omega_jk = compute_transport(Omega_j, Omega_k)
        Omega_ik = compute_transport(Omega_i, Omega_k)

        composed = Omega_ij @ Omega_jk
        diff = (composed - Omega_ik).norm()
        assert diff < 1e-5, f"Cocycle condition violated: diff = {diff:.6e}"


class TestSPDRetraction:
    """Test that SPD retraction preserves positive-definiteness."""

    def test_stays_spd(self):
        """Σ remains SPD after retraction with random perturbation."""
        from ..gaussians import natural_grad_sigma, retract_spd

        torch.manual_seed(789)
        Sigma = random_spd(K, batch_shape=(5,))

        # Random gradient
        grad = torch.randn(5, K, K)
        grad = 0.5 * (grad + grad.transpose(-2, -1))  # Symmetrize

        nat = natural_grad_sigma(grad, Sigma)
        Sigma_new = retract_spd(Sigma, nat, step_size=0.1)

        # Check SPD
        for i in range(5):
            eigs = torch.linalg.eigvalsh(Sigma_new[i])
            assert eigs.min() > 0, f"Sigma[{i}] not SPD: min eig = {eigs.min():.6e}"

        # Check symmetry
        asym = (Sigma_new - Sigma_new.transpose(-2, -1)).norm()
        assert asym < 1e-6, f"Sigma not symmetric: asymmetry = {asym:.6e}"

    def test_identity_retraction(self):
        """Zero perturbation should return original Sigma."""
        from ..gaussians import retract_spd

        Sigma = random_spd(K, batch_shape=(3,))
        zero = torch.zeros_like(Sigma)
        Sigma_new = retract_spd(Sigma, zero, step_size=0.1)

        diff = (Sigma - Sigma_new).norm()
        assert diff < 1e-5, f"Identity retraction diff = {diff:.6e}"


class TestPairwisePrecomputation:
    """Test that precomputed pairwise KL matches direct computation."""

    def test_precomputed_matches_direct(self):
        """Pairwise KL from precomputation matches direct KL computation."""
        from ..gaussians import kl_divergence, precompute_tokens, pairwise_kl
        from ..gauge import transport_gaussian, compute_transport

        torch.manual_seed(101)
        B, N, H, K_h = 1, 4, 1, K

        mu_h = torch.randn(B, N, H, K_h)
        Sigma_h = random_spd(K_h, batch_shape=(B, N, H))
        Omega = random_gl(K_h, batch_shape=(B, N, H))

        # Direct computation
        direct_kl = torch.zeros(B, H, N, N)
        for i in range(N):
            for j in range(N):
                Om_ij = compute_transport(Omega[0, i, 0], Omega[0, j, 0])
                mu_t, Sig_t = transport_gaussian(
                    mu_h[0, j, 0], Sigma_h[0, j, 0], Om_ij
                )
                direct_kl[0, 0, i, j] = kl_divergence(
                    mu_h[0, i, 0], Sigma_h[0, i, 0], mu_t, Sig_t
                )

        # Precomputed
        precomp = precompute_tokens(mu_h, Sigma_h, Omega)
        precomp_kl = pairwise_kl(precomp, causal=False)

        diff = (direct_kl - precomp_kl).abs().max()
        assert diff < 1e-3, f"Precomputed KL max diff = {diff:.6e}"


class TestEStepMonotonicity:
    """Test that E-step VFE is non-increasing."""

    def test_vfe_decreases(self):
        """VFE should not increase during E-step."""
        from ..config import PureVFEConfig
        from ..model import PureVFETransformer
        from ..inference import e_step

        config = PureVFEConfig(
            vocab_size=100,
            belief_dim=K,
            n_heads=1,
            head_dim=K,
            n_esteps=5,
            max_seq_len=8,
            device='cpu',
            use_cuda_kernels=False,
            eta_E=0.05,  # Conservative step size for monotonicity
        )

        torch.manual_seed(202)
        model = PureVFETransformer(config)
        tokens = torch.randint(0, 100, (1, 8))

        _, _, _, _, vfe_history = e_step(tokens, model, config)

        # Check monotonicity (allowing small numerical tolerance)
        for t in range(1, len(vfe_history)):
            assert vfe_history[t] <= vfe_history[t - 1] + 1e-2, \
                f"VFE increased at step {t}: {vfe_history[t - 1]:.4f} -> {vfe_history[t]:.4f}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
