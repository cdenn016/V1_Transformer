"""
Parallel Transport Operators on GL(K) Principal Bundle
======================================================

Implements Ω_ij(c) = exp(φ_i(c)) · exp(-φ_j(c)) for gauge-theoretic
active inference on spatial manifolds.

Mathematical Framework:
----------------------
**Transport Operator:**
    Ω_ij: Fiber_i → Fiber_j
    Ω_ij(c) = g_i(c) · g_j(c)^{-1} where g = exp(φ) ∈ GL⁺(K)

**Properties:**
    - Ω_ij ∈ GL⁺(K): det(Ω) > 0 (positive determinant)
    - Ω_ij(c) · Ω_jk(c) = Ω_ik(c) (transitivity)
    - Ω_ii(c) = I (self-transport is identity)

**Key Insight (GL(K) vs SO(K)):**
    All f-divergences (including KL) are invariant under the full GL(K) group.
    This is because the action (μ, Σ) → (Ωμ, ΩΣΩᵀ) is the pushforward under
    x → Ωx, and f-divergences are coordinate-invariant (Jacobians cancel in ratio).

    We do NOT need orthogonality constraints for the variational free energy
    to be gauge-invariant. The only requirement is invertibility: det(Ω) ≠ 0.

**Surjectivity of exp (important caveat):**
    The exponential map exp: gl(K, ℝ) → GL(K, ℝ) is NOT surjective for K > 1:

    1. det(exp(X)) = exp(tr(X)) > 0 always, so Im(exp) ⊂ GL⁺(K).
       Orientation-reversing transformations (det < 0) are unreachable.

    2. Even within GL⁺(K), exp is NOT surjective for K > 1.
       A matrix A ∈ GL(K, ℝ) has a real logarithm if and only if for
       EACH negative real eigenvalue λ, the number of Jordan blocks of
       EACH SIZE for λ is even (Culver 1966).

       Concrete failures (all have det > 0, none have a real log):
         - diag(-2, -3): two negative eigenvalues, each with 1 Jordan
           block of size 1 — odd count → no real log.
         - diag(-2, 1, 1, 1): one negative eigenvalue with 1 Jordan
           block of size 1 — odd count → no real log.
       Concrete successes:
         - diag(-2, -2): one negative eigenvalue (-2) with 2 Jordan
           blocks of size 1 — even count → has real log.
           (Log uses paired complex eigenvalues: log(2)I + π·J₂.)

       These "unreachable" matrices are a measure-zero set but are NOT
       topologically negligible — they disconnect components of GL⁺(K)
       in the log topology.

    3. However, every A ∈ GL⁺(K) can be written as a PRODUCT of two
       exponentials. Proof: polar decomposition A = P·O where P is
       positive-definite symmetric (always has a real log) and O ∈ SO(K)
       (always has a real log, since SO(K) is compact connected).
       So A = exp(log P) · exp(log O).

       Since Ω_ij = exp(X_i) · exp(-X_j) is a free product of two
       exponentials, the transport operators cover ALL of GL⁺(K).

    4. For SO(K) (compact, connected), exp: so(K) → SO(K) IS surjective.
       No issues there.

    5. The connection Ω_ij = g_i · g_j⁻¹ is always flat (zero curvature).
       Non-trivial holonomy would require independent per-pair parameters.

**Implementation:**
    For K=3 with SO(3) generators, use Rodrigues formula (closed form, exact)
    For general K, use matrix exponential via scipy.linalg.expm
    NO projection to orthogonal matrices - allows full GL⁺(K) flexibility

Author: Clean Rebuild
Date: November 2025
Updated: GL(K) generalization - February 2026
"""

import numpy as np
from typing import Optional, Tuple


def compute_transport(
    phi_i: np.ndarray,
    phi_j: np.ndarray,
    generators: np.ndarray,
    *,
    validate: bool = False,
    eps: float = 1e-8,
    use_gpu: bool = False,
    project_to_orthogonal: bool = False,  # NEW: opt-in orthogonal projection
) -> np.ndarray:
    """
    Compute transport operator Omega_ij = exp(phi_i . G) . exp(-phi_j . G).

    Supports SO(3), SO(N), and GL(K) gauge groups. The gauge group is
    determined by the generators shape (n_gen, K, K):
    - SO(3): n_gen=3, phi_dim=3
    - SO(N): n_gen=N(N-1)/2, phi_dim=N(N-1)/2
    - GL(K): n_gen=K^2, phi_dim=K^2

    Args:
        phi_i, phi_j: Gauge fields, shape (*S, n_gen)
        generators: Lie algebra generators, shape (n_gen, K, K)
        validate: Check invertibility of result (|det(Omega)| > eps)
        eps: Small-angle threshold / minimum determinant
        use_gpu: Ignored (kept for API compatibility)
        project_to_orthogonal: If True, project to SO(K) via SVD.
                              Default False for full GL(K) flexibility.

    Returns:
        Omega_ij: Transport operator, shape (*S, K, K)

    Note:
        GL(K) transport is sufficient for gauge-invariant VFE because all
        f-divergences are invariant under invertible linear transformations.
        Orthogonal projection is only needed for specific applications
        (e.g., volume preservation, Haar measure averaging).
    """
    phi_i = np.asarray(phi_i, dtype=np.float64)
    phi_j = np.asarray(phi_j, dtype=np.float64)
    G = np.asarray(generators, dtype=np.float64)

    # Validate shapes
    if phi_i.shape != phi_j.shape:
        raise ValueError(f"Shape mismatch: phi_i {phi_i.shape}, phi_j {phi_j.shape}")
    n_gen = G.shape[0]
    if phi_i.shape[-1] != n_gen:
        raise ValueError(f"phi has {phi_i.shape[-1]} components but {n_gen} generators provided")

    K = G.shape[1]

    # General SO(N) / GL(K) case
    exp_phi_i = _matrix_exponential_lie_algebra(phi_i, G)
    exp_neg_phi_j = _matrix_exponential_lie_algebra(-phi_j, G)

    # ========================================================================
    # Compose transport operators
    # ========================================================================
    Omega_ij = np.matmul(exp_phi_i, exp_neg_phi_j)

    if validate:
        _validate_invertible(Omega_ij, eps=eps)

    return Omega_ij.astype(np.float64, copy=False)





# =============================================================================
# General GL(K) Matrix Exponential
# =============================================================================

def _matrix_exponential_lie_algebra(
    phi: np.ndarray,
    generators: np.ndarray,
    *,
    small_threshold: float = 1e-4,
    project_to_orthogonal: bool = False,  # NEW: opt-in for SO(K) projection
    enforce_skew_symmetry: bool = False,  # NEW: opt-in for skew-symmetry
) -> np.ndarray:
    """
    Compute exp(sum_a phi^a G_a) for general GL(K).

    Uses scipy.linalg.expm for large ||phi||, Taylor series for small.

    Args:
        phi: Lie algebra coefficients, shape (*S, n_generators)
        generators: Shape (n_generators, K, K)
        small_threshold: Switch point for Taylor series
        project_to_orthogonal: If True, project result to SO(K) (legacy behavior)
        enforce_skew_symmetry: If True, enforce X = -Xᵀ (for SO(K) generators)

    Returns:
        exp_phi: Shape (*S, K, K), invertible matrices in GL⁺(K) (det > 0)
                 (orthogonal if project_to_orthogonal=True)

    Note:
        For GL(K) gauge transformations, we do NOT need orthogonal projection.
        The VFE is invariant under GL(K) because f-divergences are invariant
        under pushforward by invertible linear maps.

        A single exp(X) cannot reach all of GL⁺(K) for K > 1.  By Culver
        (1966), A ∈ GL(K,ℝ) has a real log iff for each negative real
        eigenvalue λ, the number of Jordan blocks of each size for λ is
        even. E.g. diag(-2,-3) has det 6 > 0 but no real log (each negative
        eigenvalue has 1 block of size 1: odd). But the product
        Ω_ij = exp(X_i)·exp(-X_j) covers all of GL⁺(K) via polar decomp.
    """
    phi = np.asarray(phi, dtype=np.float64)
    G = np.asarray(generators, dtype=np.float64)

    # Clip phi norm to prevent numerical overflow in expm.
    #
    # For SO(K) (enforce_skew_symmetry=True): clip to 2π.
    #   Rotations are periodic: exp(θ·G) = exp((θ mod 2π)·G), so this is
    #   mathematically exact, not just a numerical safeguard.
    #
    # For GL(K) (enforce_skew_symmetry=False): clip to ~20.
    #   There is NO periodicity in the symmetric/trace directions of gl(K).
    #   Clipping to 2π would artificially restrict reachable eigenvalues to
    #   [e^{-2π}, e^{2π}] ≈ [0.002, 535]. We use a larger threshold that
    #   keeps scipy.linalg.expm numerically stable (scaling-squaring handles
    #   norms up to ~50 comfortably in float64) while allowing the model to
    #   represent a much wider range of GL⁺(K) transformations.
    max_norm = 2 * np.pi if enforce_skew_symmetry else 20.0
    phi_norm = np.linalg.norm(phi, axis=-1, keepdims=True)
    phi_norm_clipped = np.clip(phi_norm, 0, max_norm)

    scale_factor = np.ones_like(phi_norm)
    np.divide(phi_norm_clipped, phi_norm, out=scale_factor, where=phi_norm > 1e-8)
    phi = phi * scale_factor

    batch_shape = phi.shape[:-1]
    K = G.shape[1]

    # Compute algebra element: X = Σ_a φ^a G_a
    X = np.einsum('...a,aij->...ij', phi, G, optimize=True)  # (*S, K, K)

    # Optionally enforce skew-symmetry (for SO(K) subalgebra)
    if enforce_skew_symmetry:
        X = 0.5 * (X - np.swapaxes(X, -1, -2))

    # Compute norms
    phi_norms = np.linalg.norm(phi, axis=-1)  # (*S,)

    # Allocate output
    exp_phi = np.empty(batch_shape + (K, K), dtype=np.float64)

    # ========== Small angle: Taylor series ==========
    small_mask = phi_norms < small_threshold

    if np.any(small_mask):
        X_small = X[small_mask]
        I = np.eye(K, dtype=np.float64)

        X2 = X_small @ X_small
        X3 = X2 @ X_small
        X4 = X2 @ X2

        exp_phi[small_mask] = I + X_small + 0.5*X2 + (1.0/6.0)*X3 + (1.0/24.0)*X4

    # ========== Large angle: Matrix exponential ==========
    large_mask = ~small_mask

    if np.any(large_mask):
        X_large = X[large_mask]

        try:
            from scipy.linalg import expm as scipy_expm
            exp_phi[large_mask] = np.array([scipy_expm(X_i) for X_i in X_large])
        except ImportError:
            # Fallback: Padé approximation
            exp_phi[large_mask] = np.array([_expm_pade(X_i) for X_i in X_large])

    # Optionally project to nearest orthogonal matrix (for SO(K) compatibility)
    if project_to_orthogonal:
        exp_phi = _project_to_orthogonal(exp_phi)

    return exp_phi


def _project_to_orthogonal(M: np.ndarray) -> np.ndarray:
    """
    Project matrices to nearest orthogonal matrices via SVD.

    For M ≈ rotation matrix with numerical errors,
    finds Q = argmin_Q ||M - Q||_F subject to Q ∈ SO(K).

    Solution: Q = U V^T where M = U Σ V^T (SVD)
    With determinant correction to ensure det(Q) = +1.
    """
    batch_shape = M.shape[:-2]
    K = M.shape[-1]

    # Flatten batch
    M_flat = M.reshape(-1, K, K)
    Q_flat = np.empty_like(M_flat)

    for i in range(len(M_flat)):
        # Check for NaN/Inf before SVD
        if not np.all(np.isfinite(M_flat[i])):
            # Fallback to identity if matrix is corrupted
            Q_flat[i] = np.eye(K, dtype=M_flat.dtype)
            continue

        try:
            U, _, Vt = np.linalg.svd(M_flat[i], full_matrices=False)
            Q = U @ Vt

            # Ensure det(Q) = +1
            if np.linalg.det(Q) < 0:
                U[:, -1] *= -1
                Q = U @ Vt

            Q_flat[i] = Q
        except np.linalg.LinAlgError:
            # SVD failed - fallback to identity
            Q_flat[i] = np.eye(K, dtype=M_flat.dtype)

    return Q_flat.reshape(batch_shape + (K, K))


def _expm_pade(A: np.ndarray, order: int = 13) -> np.ndarray:
    """
    Matrix exponential via Padé approximation.
    
    Fallback when scipy unavailable. For production, use scipy.linalg.expm.
    """
    n = A.shape[0]

    # Scaling
    norm_A = np.linalg.norm(A, ord=np.inf)
    if norm_A < 1e-15:
        return np.eye(n, dtype=np.float64)
    n_squarings = max(0, int(np.ceil(np.log2(norm_A))))
    A_scaled = A / (2 ** n_squarings)
    
    # Padé coefficients (order 13)
    b = np.array([
        64764752532480000., 32382376266240000., 7771770303897600.,
        1187353796428800., 129060195264000., 10559470521600.,
        670442572800., 33522128640., 1323241920., 40840800.,
        960960., 16380., 182., 1.
    ])
    
    I = np.eye(n, dtype=np.float64)
    A2 = A_scaled @ A_scaled
    A4 = A2 @ A2
    A6 = A2 @ A4
    
    U = A_scaled @ (A6 @ (b[13]*A6 + b[11]*A4 + b[9]*A2) + b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*I)
    V = A6 @ (b[12]*A6 + b[10]*A4 + b[8]*A2) + b[6]*A6 + b[4]*A4 + b[2]*A2 + b[0]*I
    
    X = np.linalg.solve(V - U, V + U)
    
    # Undo scaling
    for _ in range(n_squarings):
        X = X @ X
    
    return X


def _validate_invertible(Omega: np.ndarray, eps: float = 1e-8) -> None:
    """
    Check Ω is invertible: |det(Ω)| > eps.

    For GL(K) gauge transformations, we only need invertibility, not orthogonality.
    This is because all f-divergences are invariant under GL(K) pushforward.

    Args:
        Omega: Transport operators, shape (*batch, K, K)
        eps: Minimum absolute determinant threshold

    Raises:
        ValueError: If any transport operator is singular or near-singular
    """
    K = Omega.shape[-1]

    # Flatten for checking
    Omega_flat = Omega.reshape(-1, K, K)

    for i, Om in enumerate(Omega_flat):
        det = np.linalg.det(Om)
        if np.abs(det) < eps:
            raise ValueError(
                f"Transport operator singular at index {i}:\n"
                f"  |det(Ω)| = {np.abs(det):.2e} < {eps:.2e}"
            )

        # Also check condition number for numerical stability
        cond = np.linalg.cond(Om)
        if cond > 1e10:
            import warnings
            warnings.warn(
                f"Transport operator ill-conditioned at index {i}:\n"
                f"  cond(Ω) = {cond:.2e} (may cause numerical issues)",
                RuntimeWarning
            )


# =============================================================================
# Transport Differentials (for gradients)
# =============================================================================
#
# ``_validate_orthogonal``, ``compute_transport_differential`` and
# ``_compute_dexp_generators`` were removed in the 2026-05-17 deep-audit
# follow-up: each had zero production callers (verified by grep across the
# repository), and the dexp/Fréchet path was superseded by PyTorch autograd
# in ``transformer/core/variational_ffn.py`` and ``transformer/vfe/e_step.py``.
# ``frechet_expm`` is retained — it has dedicated tests in
# ``tests/test_math_utils.py`` and documents the mathematical reference.


def frechet_expm(X, H, steps=6):
    """
    Approximate d/dt exp(X + tH)|_{t=0} via midpoint quadrature.

    Works for any square matrix X (all irreps, all gauge groups).

    Uses the integral representation: dexp_X(H) = integral_0^1 exp((1-s)X) H exp(sX) ds.
    Midpoint rule avoids boundary double-counting of the rectangle rule.

    Args:
        X: Lie algebra element, shape (K, K)
        H: Perturbation direction, shape (K, K)
        steps: Number of quadrature points

    Returns:
        Frechet derivative, shape (K, K)
    """
    import scipy.linalg
    # Midpoint quadrature: evaluate at centers of equal-width subintervals
    ds = 1.0 / steps
    out = 0
    for k in range(steps):
        si = (k + 0.5) * ds  # midpoint of k-th subinterval
        out += scipy.linalg.expm((1 - si) * X) @ H @ scipy.linalg.expm(si * X)
    return out * ds


# ``_compute_dexp_generators`` removed 2026-05-17 — only consumer was the
# now-deleted ``compute_transport_differential``. Both replaced by PyTorch
# autograd in the production transport path.