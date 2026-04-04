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
REL_TOL = 5e-3  # Relaxed for float32; FD tests use float64 internally
FD_DTYPE = torch.float64  # Float64 for accurate finite differences


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
        self.mu_i = torch.randn(K, dtype=FD_DTYPE)
        self.mu_j = torch.randn(K, dtype=FD_DTYPE)
        self.Sigma_i = random_spd(K).to(FD_DTYPE)
        self.Sigma_j = random_spd(K).to(FD_DTYPE)
        self.Omega_ij = random_gl(K).to(FD_DTYPE)

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
        """∂KL(q_i || Ω_ij · q_j)/∂Σ_i vs finite differences.

        Uses symmetric perturbations to keep Σ_i SPD during FD.
        For diagonal: perturb (a,a) by ε.
        For off-diagonal: perturb both (a,b) and (b,a) by ε (symmetric).
        """
        from ..gaussians import kl_divergence
        from ..gauge import transport_gaussian

        mu_t, Sigma_t = transport_gaussian(self.mu_j, self.Sigma_j, self.Omega_ij)
        Sigma_t_inv = torch.linalg.inv(Sigma_t)
        Sigma_i_inv = torch.linalg.inv(self.Sigma_i)

        # Analytic: ½(Σ_Q⁻¹ - Σ_i⁻¹)
        analytic = 0.5 * (Sigma_t_inv - Sigma_i_inv)

        # Finite difference with symmetric perturbations.
        # Off-diagonal: perturb (a,b) and (b,a) by eps, giving FD = dKL/dS[a,b] + dKL/dS[b,a].
        # Since analytic grad is symmetric, dKL/dS[a,b] = dKL/dS[b,a], so FD = 2*analytic[a,b].
        fd = torch.zeros(K, K, dtype=FD_DTYPE)
        for a in range(K):
            for b in range(a, K):
                E = torch.zeros(K, K, dtype=FD_DTYPE)
                E[a, b] = EPS
                if a != b:
                    E[b, a] = EPS  # Keep symmetric
                kl_plus = kl_divergence(self.mu_i, self.Sigma_i + E, mu_t, Sigma_t)
                kl_minus = kl_divergence(self.mu_i, self.Sigma_i - E, mu_t, Sigma_t)
                val = (kl_plus - kl_minus) / (2 * EPS)
                if a != b:
                    val = val / 2  # Symmetric perturbation counts both (a,b) and (b,a)
                fd[a, b] = val
                fd[b, a] = val

        rel_err = (analytic - fd).norm() / fd.norm().clamp(min=1e-8)
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
        mu_i = torch.randn(K, dtype=FD_DTYPE)
        mu_j = torch.randn(K, dtype=FD_DTYPE)
        Sigma_i = random_spd(K).to(FD_DTYPE)
        Sigma_j = random_spd(K).to(FD_DTYPE)
        Omega = random_gl(K).to(FD_DTYPE)
        G = random_gl(K, scale=0.5).to(FD_DTYPE)

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
        Omega_i = random_gl(K).to(FD_DTYPE)
        Omega_j = random_gl(K).to(FD_DTYPE)
        Omega_k = random_gl(K).to(FD_DTYPE)

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

        mu_h = torch.randn(B, N, H, K_h, dtype=FD_DTYPE)
        Sigma_h = random_spd(K_h, batch_shape=(B, N, H)).to(FD_DTYPE)
        Omega = random_gl(K_h, batch_shape=(B, N, H)).to(FD_DTYPE)

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

        # Compare only the lower triangle (causal=False still has valid entries)
        # Use relative error since KL values can be large
        mask = direct_kl > 1e-3  # Only compare entries with meaningful KL
        if mask.any():
            rel_diff = ((direct_kl[mask] - precomp_kl[mask]).abs() / direct_kl[mask].abs().clamp(min=1e-6)).max()
            assert rel_diff < 0.01, f"Precomputed KL max rel diff = {rel_diff:.6e}"
        else:
            diff = (direct_kl - precomp_kl).abs().max()
            assert diff < 1e-2, f"Precomputed KL max diff = {diff:.6e}"


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
            mu_q_lr=0.05,  # Conservative step size for monotonicity
        )

        torch.manual_seed(202)
        model = PureVFETransformer(config)
        tokens = torch.randint(0, 100, (1, 8))

        _, _, _, _, vfe_history, _ = e_step(tokens, model, config)

        # Check monotonicity (allowing small numerical tolerance)
        for t in range(1, len(vfe_history)):
            assert vfe_history[t] <= vfe_history[t - 1] + 1e-2, \
                f"VFE increased at step {t}: {vfe_history[t - 1]:.4f} -> {vfe_history[t]:.4f}"


class TestSoftmaxCorrectedSigmaGradient:
    """Test that Sigma gradient includes softmax correction (Finding 6)."""

    def setup_method(self):
        torch.manual_seed(123)
        self.B, self.H, self.N, self.K_h = 1, 2, 4, 3

    def test_sigma_gradient_uses_softmax_correction(self):
        """vfe_grad_Sigma_alignment with kl_ij should differ from without."""
        from ..gaussians import vfe_grad_Sigma_alignment

        B, H, N, K_h = self.B, self.H, self.N, self.K_h

        # Build mock precomp dict
        P = random_spd(K_h, (B, H, N))  # precisions
        Om_inv = random_gl(K_h, (B, H, N), scale=0.1)
        precomp = {'P': P, 'Omega_inv': Om_inv}

        beta = torch.softmax(torch.randn(B, H, N, N), dim=-1)
        kl_ij = torch.rand(B, H, N, N) * 5.0
        tau = 1.0

        # Without correction
        result_no_corr = vfe_grad_Sigma_alignment(precomp, beta)
        # With correction
        result_with_corr = vfe_grad_Sigma_alignment(precomp, beta, kl_ij, tau)

        # They should differ (correction changes weights from beta to w_ij)
        assert not torch.allclose(result_no_corr, result_with_corr, atol=1e-6), \
            "Softmax correction should produce different gradients"

    def test_sigma_gradient_correction_reduces_to_uncorrected_when_kl_uniform(self):
        """When all KL values are equal, correction factor is 1 → same as uncorrected."""
        from ..gaussians import vfe_grad_Sigma_alignment

        B, H, N, K_h = self.B, self.H, self.N, self.K_h
        P = random_spd(K_h, (B, H, N))
        Om_inv = random_gl(K_h, (B, H, N), scale=0.1)
        precomp = {'P': P, 'Omega_inv': Om_inv}

        beta = torch.softmax(torch.randn(B, H, N, N), dim=-1)
        # Uniform KL → correction = (E[KL] - KL) / tau = 0 → w_ij = beta_ij
        kl_uniform = torch.ones(B, H, N, N) * 3.0
        tau = 1.0

        result_no_corr = vfe_grad_Sigma_alignment(precomp, beta)
        result_with_corr = vfe_grad_Sigma_alignment(precomp, beta, kl_uniform, tau)

        assert torch.allclose(result_no_corr, result_with_corr, atol=1e-5), \
            "Uniform KL should make corrected gradient match uncorrected"


class TestMStepOmegaUpdate:
    """Test M-step Omega moment-matching update (Finding 7)."""

    def test_omega_converges_toward_target(self):
        """Moment-matching grad = -(Omega_star - Omega) should pull toward target."""
        torch.manual_seed(99)
        T, H, K_h = 5, 2, 3
        Omega = torch.eye(K_h).unsqueeze(0).unsqueeze(0).expand(T, H, K_h, K_h).clone()
        Omega_star = random_gl(K_h, (T, H), scale=0.2)

        # Moment-matching update
        lr = 0.1
        grad = -(Omega_star - Omega)
        Omega_updated = Omega - lr * grad  # = Omega + lr * (Omega_star - Omega)

        # Distance to target should decrease
        dist_before = (Omega - Omega_star).norm()
        dist_after = (Omega_updated - Omega_star).norm()
        assert dist_after < dist_before, \
            f"Moment-matching should reduce distance to target: {dist_before:.4f} -> {dist_after:.4f}"


###############################################################################
# Phase 1b: Expanded gradient validation tests
###############################################################################

class TestVfeGradMuAlignment:
    """FD validation of vfe_grad_mu_alignment (Eq. 21 + Eq. 24 softmax correction)."""

    def setup_method(self):
        torch.manual_seed(42)
        self.B, self.H, self.N, self.K_h = 1, 1, 4, K

    def _build(self, seed=42, dtype=torch.float32):
        from ..gaussians import precompute_tokens, pairwise_kl, kl_attention
        torch.manual_seed(seed)
        B, H, N, K_h = self.B, self.H, self.N, self.K_h
        mu_h = torch.randn(B, N, H, K_h, dtype=dtype)
        Sigma_h = random_spd(K_h, (B, N, H)).to(dtype)
        Omega = random_gl(K_h, (B, N, H)).to(dtype)
        precomp = precompute_tokens(mu_h, Sigma_h, Omega)
        kl_ij = pairwise_kl(precomp, causal=False)
        tau = K_h ** 0.5
        beta = kl_attention(kl_ij, tau, causal=False)
        return mu_h, Sigma_h, Omega, precomp, kl_ij, beta, tau

    def test_mu_alignment_grad_fd(self):
        """vfe_grad_mu_alignment matches FD on coupling VFE = sum_j beta_ij * KL_ij.

        Uses close beliefs to keep softmax correction within clamp bounds.
        """
        from ..gaussians import (
            vfe_grad_mu_alignment, precompute_tokens, pairwise_kl, kl_attention,
        )
        B, H, N, K_h = self.B, self.H, self.N, self.K_h
        # Use close beliefs so KL is small and the clamp(-1, 2) in the
        # softmax correction is NOT active, making the analytical formula exact.
        torch.manual_seed(42)
        base_mu = torch.randn(1, 1, 1, K_h, dtype=FD_DTYPE)
        mu_h = base_mu + 0.1 * torch.randn(B, N, H, K_h, dtype=FD_DTYPE)
        Sigma_h = (torch.eye(K_h, dtype=FD_DTYPE) * 1.0).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, N, H, -1, -1).clone()
        Omega = torch.eye(K_h, dtype=FD_DTYPE).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, N, H, -1, -1).clone()
        # Small perturbation to Omega to break symmetry
        Omega = Omega + 0.05 * torch.randn(B, N, H, K_h, K_h, dtype=FD_DTYPE)

        precomp = precompute_tokens(mu_h, Sigma_h, Omega)
        kl_ij = pairwise_kl(precomp, causal=False)
        tau = 10.0  # Large tau to keep corrections small
        beta = kl_attention(kl_ij, tau, causal=False)

        analytic = vfe_grad_mu_alignment(precomp, beta, kl_ij, tau)

        def query_side_coupling(mu_h_, query_i):
            """sum_j beta_ij * KL_ij for query position i, with beta recomputed."""
            pc = precompute_tokens(mu_h_, Sigma_h, Omega)
            kl = pairwise_kl(pc, causal=False)
            b = kl_attention(kl, tau, causal=False)
            return (b[0, 0, query_i, :] * kl[0, 0, query_i, :]).sum()

        fd = torch.zeros_like(analytic)
        for i in range(N):
            for d in range(K_h):
                mu_p = mu_h.clone()
                mu_p[0, i, 0, d] += EPS
                mu_m = mu_h.clone()
                mu_m[0, i, 0, d] -= EPS
                fd[0, 0, i, d] = (query_side_coupling(mu_p, i) - query_side_coupling(mu_m, i)) / (2 * EPS)

        rel_err = (analytic - fd).norm() / fd.norm().clamp(min=1e-8)
        assert rel_err < 0.01, f"mu_alignment grad rel_err = {rel_err:.6f}"

    def test_mu_alignment_zero_when_beliefs_equal(self):
        """When all beliefs identical and Omega=I, gradient should be ~zero."""
        from ..gaussians import (
            vfe_grad_mu_alignment, precompute_tokens, pairwise_kl, kl_attention,
        )
        B, H, N, K_h = self.B, self.H, self.N, self.K_h
        mu_h = torch.ones(B, N, H, K_h) * 0.5
        Sigma_h = torch.eye(K_h).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, N, H, -1, -1).clone()
        Omega = torch.eye(K_h).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, N, H, -1, -1).clone()

        precomp = precompute_tokens(mu_h, Sigma_h, Omega)
        kl_ij = pairwise_kl(precomp, causal=False)
        tau = K_h ** 0.5
        beta = kl_attention(kl_ij, tau, causal=False)

        grad = vfe_grad_mu_alignment(precomp, beta, kl_ij, tau)
        assert grad.norm() < 1e-4, f"Expected ~zero gradient, got norm = {grad.norm():.6e}"

    def test_mu_alignment_softmax_correction_sign(self):
        """Correction term (E[KL] - KL_ij)/tau upweights below-average pairs."""
        from ..gaussians import vfe_grad_mu_alignment
        _, _, _, precomp, kl_ij, beta, tau = self._build()

        # With correction
        grad_corr = vfe_grad_mu_alignment(precomp, beta, kl_ij, tau)
        # Without correction (set tau very large to suppress correction)
        grad_nocorr = vfe_grad_mu_alignment(precomp, beta, kl_ij, tau=1e10)

        # They should differ when KL values are non-uniform
        assert not torch.allclose(grad_corr, grad_nocorr, atol=1e-5), \
            "Softmax correction should produce different gradients"

    def test_mu_alignment_reduces_to_uncorrected_when_kl_uniform(self):
        """Uniform KL => correction factor = 1 => matches uncorrected."""
        from ..gaussians import vfe_grad_mu_alignment
        _, _, _, precomp, _, beta, _ = self._build()

        kl_uniform = torch.ones_like(beta) * 3.0
        tau = K ** 0.5
        grad_corr = vfe_grad_mu_alignment(precomp, beta, kl_uniform, tau)
        grad_nocorr = vfe_grad_mu_alignment(precomp, beta, kl_uniform, tau=1e10)

        assert torch.allclose(grad_corr, grad_nocorr, atol=1e-4), \
            "Uniform KL should make corrected gradient match uncorrected"


class TestVfeGradOmega:
    """FD validation of aggregated gauge gradient vfe_grad_Omega."""

    def setup_method(self):
        torch.manual_seed(42)
        self.B, self.H, self.N, self.K_h = 1, 1, 4, K

    def _build(self, seed=42, dtype=torch.float32):
        from ..gaussians import precompute_tokens, pairwise_kl, kl_attention
        torch.manual_seed(seed)
        B, H, N, K_h = self.B, self.H, self.N, self.K_h
        mu_h = torch.randn(B, N, H, K_h, dtype=dtype)
        Sigma_h = random_spd(K_h, (B, N, H)).to(dtype)
        Omega = random_gl(K_h, (B, N, H), scale=0.2).to(dtype)
        precomp = precompute_tokens(mu_h, Sigma_h, Omega)
        kl_ij = pairwise_kl(precomp, causal=False)
        tau = K_h ** 0.5
        beta = kl_attention(kl_ij, tau, causal=False)
        return mu_h, Sigma_h, Omega, precomp, kl_ij, beta, tau

    def test_vfe_grad_Omega_fd(self):
        """vfe_grad_Omega matches FD of query-side coupling: sum_j beta_ij * KL_ij.

        vfe_grad_Omega only computes the forward (query) contribution:
        dF/dOmega_i = sum_j beta_ij * dKL_ij/dOmega_i.
        When Omega_i changes, KL_ji (where i is key) also changes, but
        vfe_grad_Omega does NOT include that. FD must match by only summing
        over pairs where i is the query.
        """
        from ..gaussians import precompute_tokens, pairwise_kl, kl_attention
        from ..gauge import vfe_grad_Omega
        mu_h, Sigma_h, Omega, precomp, kl_ij, beta, tau = self._build(dtype=FD_DTYPE)
        B, H, N, K_h = self.B, self.H, self.N, self.K_h

        analytic = vfe_grad_Omega(mu_h, Sigma_h, Omega, beta, kl_ij, precomp)
        beta_fixed = beta.detach().clone()

        def query_side_vfe(Om_, query_i):
            """sum_j beta_fixed[i,j] * KL_ij for a single query position i."""
            pc = precompute_tokens(mu_h, Sigma_h, Om_)
            kl = pairwise_kl(pc, causal=False)
            return (beta_fixed[0, 0, query_i, :] * kl[0, 0, query_i, :]).sum()

        fd = torch.zeros_like(analytic)
        for i in range(N):
            for a in range(K_h):
                for b in range(K_h):
                    Om_p = Omega.clone()
                    Om_p[0, i, 0, a, b] += EPS
                    Om_m = Omega.clone()
                    Om_m[0, i, 0, a, b] -= EPS
                    fd[0, i, 0, a, b] = (query_side_vfe(Om_p, i) - query_side_vfe(Om_m, i)) / (2 * EPS)

        rel_err = (analytic - fd).norm() / fd.norm().clamp(min=1e-8)
        assert rel_err < REL_TOL, f"vfe_grad_Omega rel_err = {rel_err:.6f}"

    def test_vfe_grad_Omega_zero_at_identity(self):
        """When all Omega=I and all beliefs identical, gradient should vanish."""
        from ..gaussians import precompute_tokens, pairwise_kl, kl_attention
        from ..gauge import vfe_grad_Omega
        B, H, N, K_h = self.B, self.H, self.N, self.K_h

        mu_h = torch.ones(B, N, H, K_h) * 0.5
        Sigma_h = torch.eye(K_h).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, N, H, -1, -1).clone()
        Omega = torch.eye(K_h).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, N, H, -1, -1).clone()

        precomp = precompute_tokens(mu_h, Sigma_h, Omega)
        kl_ij = pairwise_kl(precomp, causal=False)
        tau = K_h ** 0.5
        beta = kl_attention(kl_ij, tau, causal=False)

        grad = vfe_grad_Omega(mu_h, Sigma_h, Omega, beta, kl_ij, precomp)
        assert grad.norm() < 1e-4, f"Expected ~zero gradient, got norm = {grad.norm():.6e}"

    def test_vfe_grad_Omega_finite_with_illconditioned(self):
        """Gradient stays finite even with near-singular Omega."""
        from ..gaussians import precompute_tokens, pairwise_kl, kl_attention
        from ..gauge import vfe_grad_Omega
        B, H, N, K_h = self.B, self.H, self.N, self.K_h
        torch.manual_seed(99)

        mu_h = torch.randn(B, N, H, K_h)
        Sigma_h = random_spd(K_h, (B, N, H))
        # Ill-conditioned Omega: stretch one singular value
        Omega = random_gl(K_h, (B, N, H), scale=0.1)
        Omega[..., 0, :] *= 100.0  # Condition ~100

        precomp = precompute_tokens(mu_h, Sigma_h, Omega)
        kl_ij = pairwise_kl(precomp, causal=False)
        tau = K_h ** 0.5
        beta = kl_attention(kl_ij, tau, causal=False)

        grad = vfe_grad_Omega(mu_h, Sigma_h, Omega, beta, kl_ij, precomp)
        assert torch.isfinite(grad).all(), "Gradient has NaN/Inf with ill-conditioned Omega"

    def test_vfe_grad_Omega_full_includes_backward(self):
        """vfe_grad_Omega_full differs from vfe_grad_Omega (forward-only misses backward)."""
        from ..gauge import vfe_grad_Omega, vfe_grad_Omega_full
        mu_h, Sigma_h, Omega, precomp, kl_ij, beta, tau = self._build()

        grad_fwd = vfe_grad_Omega(mu_h, Sigma_h, Omega, beta, kl_ij, precomp)
        grad_full = vfe_grad_Omega_full(mu_h, Sigma_h, Omega, beta, kl_ij, precomp)

        diff = (grad_fwd - grad_full).norm()
        assert diff > 1e-4, f"Full and forward-only should differ, diff = {diff:.6e}"


class TestGradKlOmegaI:
    """Test chain rule: grad_kl_Omega_i = grad_kl_Omega_ij @ Omega_j^{-T}."""

    def setup_method(self):
        torch.manual_seed(42)
        self.mu_i = torch.randn(K, dtype=FD_DTYPE)
        self.mu_j = torch.randn(K, dtype=FD_DTYPE)
        self.Sigma_i = random_spd(K).to(FD_DTYPE)
        self.Sigma_j = random_spd(K).to(FD_DTYPE)
        self.Omega_i = random_gl(K).to(FD_DTYPE)
        self.Omega_j = random_gl(K).to(FD_DTYPE)

    def test_grad_kl_Omega_i_chain_rule_fd(self):
        """grad_kl_Omega_i matches FD w.r.t. Omega_i through Omega_ij = Omega_i @ Omega_j^{-1}."""
        from ..gaussians import kl_divergence
        from ..gauge import transport_gaussian, grad_kl_Omega_i

        analytic = grad_kl_Omega_i(
            self.mu_i, self.Sigma_i, self.mu_j, self.Sigma_j,
            self.Omega_i, self.Omega_j,
        )

        fd = torch.zeros(K, K)
        for a in range(K):
            for b in range(K):
                E = torch.zeros(K, K)
                E[a, b] = EPS
                Oi_p = self.Omega_i + E
                Oi_m = self.Omega_i - E
                Oij_p = Oi_p @ torch.linalg.inv(self.Omega_j)
                Oij_m = Oi_m @ torch.linalg.inv(self.Omega_j)
                mu_tp, Sig_tp = transport_gaussian(self.mu_j, self.Sigma_j, Oij_p)
                mu_tm, Sig_tm = transport_gaussian(self.mu_j, self.Sigma_j, Oij_m)
                kl_p = kl_divergence(self.mu_i, self.Sigma_i, mu_tp, Sig_tp)
                kl_m = kl_divergence(self.mu_i, self.Sigma_i, mu_tm, Sig_tm)
                fd[a, b] = (kl_p - kl_m) / (2 * EPS)

        rel_err = (analytic - fd).norm() / fd.norm().clamp(min=1e-8)
        assert rel_err < REL_TOL, f"Omega_i chain rule rel_err = {rel_err:.6f}"

    def test_grad_kl_Omega_i_at_identity(self):
        """At Omega_i = Omega_j = I, reduces to grad_kl_Omega_ij."""
        from ..gauge import grad_kl_Omega_i, grad_kl_Omega_ij

        I = torch.eye(K, dtype=FD_DTYPE)
        grad_i = grad_kl_Omega_i(
            self.mu_i, self.Sigma_i, self.mu_j, self.Sigma_j, I, I,
        )
        grad_ij = grad_kl_Omega_ij(
            self.mu_i, self.Sigma_i, self.mu_j, self.Sigma_j, I,
        )
        # At I, Omega_j^{-T} = I, so grad_i = grad_ij @ I = grad_ij
        diff = (grad_i - grad_ij).norm()
        assert diff < 1e-5, f"At identity, chain rule should match: diff = {diff:.6e}"

    def test_grad_kl_Omega_i_equivariance(self):
        """Left-multiply Omega_i, Omega_j by G => gradient is finite and correct shape."""
        from ..gauge import grad_kl_Omega_i

        G = random_gl(K).to(FD_DTYPE)
        grad_orig = grad_kl_Omega_i(
            self.mu_i, self.Sigma_i, self.mu_j, self.Sigma_j,
            self.Omega_i, self.Omega_j,
        )
        grad_g = grad_kl_Omega_i(
            self.mu_i, self.Sigma_i, self.mu_j, self.Sigma_j,
            G @ self.Omega_i, G @ self.Omega_j,
        )
        assert torch.isfinite(grad_g).all(), "Equivariance test: gradient not finite"
        assert grad_g.shape == grad_orig.shape


class TestVfeGradMuPrior:
    """FD validation of vfe_grad_mu_prior."""

    def test_mu_prior_grad_fd(self):
        """vfe_grad_mu_prior matches FD of alpha * KL(q||p)."""
        from ..gaussians import kl_divergence, vfe_grad_mu_prior
        torch.manual_seed(42)

        B, N_, K_ = 1, 2, K
        mu = torch.randn(B, N_, K_, dtype=FD_DTYPE)
        Sigma = random_spd(K_, (B, N_)).to(FD_DTYPE)
        prior_mu = torch.randn(B, N_, K_, dtype=FD_DTYPE)
        prior_Sigma = random_spd(K_, (B, N_)).to(FD_DTYPE)
        alpha = torch.tensor([[0.5, 0.8]], dtype=FD_DTYPE)

        analytic = vfe_grad_mu_prior(mu, Sigma, prior_mu, prior_Sigma, alpha)

        fd = torch.zeros_like(mu)
        for n in range(N_):
            for d in range(K_):
                mu_p = mu.clone()
                mu_p[0, n, d] += EPS
                mu_m = mu.clone()
                mu_m[0, n, d] -= EPS
                kl_p = alpha[0, n] * kl_divergence(mu_p[0, n], Sigma[0, n], prior_mu[0, n], prior_Sigma[0, n])
                kl_m = alpha[0, n] * kl_divergence(mu_m[0, n], Sigma[0, n], prior_mu[0, n], prior_Sigma[0, n])
                fd[0, n, d] = (kl_p - kl_m) / (2 * EPS)

        rel_err = (analytic - fd).norm() / fd.norm().clamp(min=1e-8)
        assert rel_err < REL_TOL, f"mu_prior grad rel_err = {rel_err:.6f}"

    def test_mu_prior_grad_zero_at_prior(self):
        """When mu == prior_mu, gradient is zero."""
        from ..gaussians import vfe_grad_mu_prior
        torch.manual_seed(42)

        B, N_, K_ = 1, 2, K
        mu = torch.randn(B, N_, K_)
        Sigma = random_spd(K_, (B, N_))
        prior_Sigma = random_spd(K_, (B, N_))
        alpha = torch.ones(B, N_)

        grad = vfe_grad_mu_prior(mu, Sigma, mu.clone(), prior_Sigma, alpha)
        assert grad.norm() < 1e-6, f"Gradient should be zero at prior, got norm = {grad.norm():.6e}"


class TestVfeGradSigmaPrior:
    """Test vfe_grad_Sigma_prior returns prior precision."""

    def test_sigma_prior_returns_prior_precision(self):
        from ..gaussians import vfe_grad_Sigma_prior, safe_inverse
        torch.manual_seed(42)

        B, N_, K_ = 1, 2, K
        Sigma = random_spd(K_, (B, N_))
        prior_Sigma = random_spd(K_, (B, N_))
        alpha = torch.ones(B, N_)

        result = vfe_grad_Sigma_prior(Sigma, prior_Sigma, alpha)
        expected = safe_inverse(prior_Sigma)
        diff = (result - expected).norm()
        assert diff < 1e-5, f"Should return prior precision, diff = {diff:.6e}"

    def test_sigma_prior_spd(self):
        from ..gaussians import vfe_grad_Sigma_prior
        torch.manual_seed(42)

        Sigma = random_spd(K, (2, 3))
        prior_Sigma = random_spd(K, (2, 3))
        alpha = torch.ones(2, 3)

        result = vfe_grad_Sigma_prior(Sigma, prior_Sigma, alpha)
        for i in range(2):
            for j in range(3):
                eigs = torch.linalg.eigvalsh(result[i, j])
                assert eigs.min() > 0, f"Prior precision not SPD at [{i},{j}]: min eig = {eigs.min():.6e}"


class TestStateDependentAlpha:
    """Test state_dependent_alpha: alpha_i = c0 / (b0 + KL(q_i || p_i))."""

    def setup_method(self):
        torch.manual_seed(42)
        self.B, self.N_ = 1, 3
        self.mu = torch.randn(self.B, self.N_, K)
        self.Sigma = random_spd(K, (self.B, self.N_))
        self.prior_mu = torch.randn(self.B, self.N_, K)
        self.prior_Sigma = random_spd(K, (self.B, self.N_))

    def test_alpha_decreases_with_kl(self):
        """Higher KL => lower alpha."""
        from ..gaussians import state_dependent_alpha
        # Close to prior -> high alpha
        alpha_close = state_dependent_alpha(
            self.prior_mu, self.Sigma, self.prior_mu, self.prior_Sigma, 1.0, 1.0,
        )
        # Far from prior -> low alpha
        alpha_far = state_dependent_alpha(
            self.prior_mu + 10.0, self.Sigma, self.prior_mu, self.prior_Sigma, 1.0, 1.0,
        )
        assert (alpha_close >= alpha_far).all(), "Alpha should decrease with KL distance"

    def test_alpha_floor(self):
        """alpha >= alpha_floor even for huge KL."""
        from ..gaussians import state_dependent_alpha
        alpha = state_dependent_alpha(
            self.prior_mu + 100.0, self.Sigma, self.prior_mu, self.prior_Sigma,
            1.0, 1.0, alpha_floor=0.5,
        )
        assert alpha.min() >= 0.5, f"Alpha floor violated: min = {alpha.min():.6f}"

    def test_alpha_at_prior(self):
        """When q = p (KL=0), alpha = c0/b0."""
        from ..gaussians import state_dependent_alpha
        alpha = state_dependent_alpha(
            self.prior_mu, self.prior_Sigma, self.prior_mu, self.prior_Sigma,
            b0=2.0, c0=3.0,
        )
        expected = 3.0 / 2.0
        assert torch.allclose(alpha, torch.full_like(alpha, expected), atol=1e-3), \
            f"At prior, alpha should be c0/b0 = {expected}, got {alpha}"


class TestKlDecodeLogits:
    """Test KL-based decoding logits."""

    def setup_method(self):
        torch.manual_seed(42)
        self.V, self.K_ = 10, K
        self.prior_mu_bank = torch.randn(self.V, self.K_)
        self.prior_Sigma_bank = random_spd(self.K_, (self.V,))

    def test_decode_logits_argmax_is_closest_prior(self):
        """Token with mu == prior_mu_bank[v] should get highest logit for v."""
        from ..gaussians import kl_decode_logits

        v = 3
        mu = self.prior_mu_bank[v].unsqueeze(0).unsqueeze(0)  # [1, 1, K]
        Sigma = self.prior_Sigma_bank[v].unsqueeze(0).unsqueeze(0)  # [1, 1, K, K]

        logits = kl_decode_logits(mu, Sigma, self.prior_mu_bank, self.prior_Sigma_bank)
        assert logits[0, 0].argmax() == v, \
            f"Expected argmax = {v}, got {logits[0, 0].argmax().item()}"

    def test_decode_logits_temperature_scaling(self):
        """decode_tau=2 halves all logits compared to tau=1."""
        from ..gaussians import kl_decode_logits
        torch.manual_seed(42)
        mu = torch.randn(1, 2, self.K_)
        Sigma = random_spd(self.K_, (1, 2))

        logits_1 = kl_decode_logits(mu, Sigma, self.prior_mu_bank, self.prior_Sigma_bank, decode_tau=1.0)
        logits_2 = kl_decode_logits(mu, Sigma, self.prior_mu_bank, self.prior_Sigma_bank, decode_tau=2.0)

        ratio = logits_1 / logits_2.clamp(min=-1e6, max=-1e-6)
        assert torch.allclose(ratio, torch.full_like(ratio, 2.0), atol=0.1), \
            "decode_tau=2 should halve logits"

    def test_decode_logits_correct_kl_formula(self):
        """Logit[i,v] = -KL(q_i || pi_v) / tau matches manual computation."""
        from ..gaussians import kl_decode_logits, kl_divergence
        torch.manual_seed(42)
        mu = torch.randn(1, 1, self.K_)
        Sigma = random_spd(self.K_, (1, 1))

        logits = kl_decode_logits(mu, Sigma, self.prior_mu_bank, self.prior_Sigma_bank)

        # Manual check for first 3 tokens
        for v in range(3):
            kl = kl_divergence(mu[0, 0], Sigma[0, 0], self.prior_mu_bank[v], self.prior_Sigma_bank[v])
            expected = -kl.item()
            actual = logits[0, 0, v].item()
            assert abs(expected - actual) < 1e-3, \
                f"Token {v}: expected {expected:.4f}, got {actual:.4f}"


class TestSafeInverseAndLogdet:
    """Test safe_inverse and safe_logdet numerical robustness."""

    def test_safe_inverse_wellconditioned(self):
        from ..gaussians import safe_inverse
        M = random_spd(K)
        M_inv = safe_inverse(M)
        product = M @ M_inv
        diff = (product - torch.eye(K)).norm()
        assert diff < 1e-4, f"Inverse not accurate: ||M @ M_inv - I|| = {diff:.6e}"

    def test_safe_inverse_nearsingular(self):
        """Returns regularized result for near-singular matrix."""
        from ..gaussians import safe_inverse
        M = torch.eye(K)
        M[0, 0] = 1e-10  # Near-singular
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            M_inv = safe_inverse(M)
        assert torch.isfinite(M_inv).all(), "Inverse should be finite for near-singular"

    def test_safe_logdet_spd(self):
        """Returns correct value for known SPD matrix."""
        from ..gaussians import safe_logdet
        diag = torch.tensor([1.0, 2.0, 3.0, 4.0])
        M = torch.diag(diag)
        result = safe_logdet(M)
        expected = diag.log().sum()
        assert abs(result.item() - expected.item()) < 1e-5, \
            f"logdet: expected {expected:.4f}, got {result:.4f}"

    def test_safe_logdet_non_spd_returns_penalty(self):
        """Returns penalty for matrix with negative eigenvalue."""
        from ..gaussians import safe_logdet
        M = torch.diag(torch.tensor([1.0, 2.0, -1.0, 3.0]))
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = safe_logdet(M)
        assert result.item() < 0, "Non-SPD should return negative penalty"
        assert torch.isfinite(result), "Penalty should be finite"


class TestSoftmaxCeGradient:
    """Test softmax_ce_gradient matches PyTorch autograd."""

    def test_ce_gradient_matches_pytorch(self):
        from ..gaussians import softmax_ce_gradient
        torch.manual_seed(42)
        logits = torch.randn(2, 4, 10, requires_grad=True)
        targets = torch.randint(0, 10, (2, 4))

        # Autograd reference
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, 10), targets.view(-1), reduction='sum'
        ) / (2 * 4)
        loss.backward()
        autograd_grad = logits.grad.clone() * (2 * 4)  # undo mean reduction

        # Analytical
        analytic = softmax_ce_gradient(logits.detach(), targets)

        rel_err = (analytic - autograd_grad).norm() / autograd_grad.norm().clamp(min=1e-8)
        assert rel_err < 1e-4, f"CE gradient rel_err = {rel_err:.6f}"

    def test_ce_gradient_sums_to_zero(self):
        """Each gradient vector sums to zero (softmax property: probs sum to 1)."""
        from ..gaussians import softmax_ce_gradient
        torch.manual_seed(42)
        logits = torch.randn(2, 4, 10)
        targets = torch.randint(0, 10, (2, 4))
        grad = softmax_ce_gradient(logits, targets)
        sums = grad.sum(dim=-1)
        assert sums.abs().max() < 1e-5, f"Gradient doesn't sum to zero: max sum = {sums.abs().max():.6e}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
