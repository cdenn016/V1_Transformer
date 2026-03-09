# -*- coding: utf-8 -*-
"""
Lie algebra generators for gauge transport operators.

Supports SO(3), SO(N), GL(K), multi-irrep block-diagonal, and cross-head coupling.
Also provides torch-based Lie group operations (BCH, coordinate extraction, retractions).
"""

import numpy as np
from typing import Dict


def generate_so3_generators(
    K: int,
    *,
    cache: bool = True,
    validate: bool = True,
    eps: float = 1e-6,
) -> np.ndarray:
    """Generate SO(3) Lie algebra generators for dimension K.

    Args:
        K: Latent dimension (must be odd: K = 2l + 1)
        cache: If True, cache generators for reuse
        validate: If True, verify commutation relations
        eps: Tolerance for validation

    Returns:
        G: Generators array, shape (3, K, K), float32
    """
    if K % 2 == 0:
        raise ValueError(
            f"K must be odd for SO(3) irreps (K = 2ℓ + 1). Got K={K}."
        )

    if cache and K in _GENERATOR_CACHE:
        return _GENERATOR_CACHE[K].copy()

    ell = (K - 1) // 2

    G = _build_so3_irrep_generators(ell)

    if validate:
        _validate_so3_generators(G, eps=1e-5)

    if cache:
        _GENERATOR_CACHE[K] = G.copy()

    return G


def _build_so3_irrep_generators(ell: int) -> np.ndarray:
    """Build SO(3) generators for spin-l irrep in real tesseral basis.

    Constructs complex J operators, builds spherical-to-tesseral transform S,
    then computes G_a = Re(S (iJ_a) S^dagger) with skew-symmetry enforcement.

    Args:
        ell: Spin quantum number (l >= 0)

    Returns:
        G: (3, K, K) float32 generators where K = 2l + 1
    """
    K = 2 * ell + 1

    # Build J_+, J_-, J_z in complex basis
    J_plus = np.zeros((K, K), dtype=np.complex128)
    J_minus = np.zeros((K, K), dtype=np.complex128)
    J_z = np.zeros((K, K), dtype=np.complex128)

    for m in range(-ell, ell + 1):
        i = m + ell  # m in [-l, l] -> i in [0, K-1]

        J_z[i, i] = m

        if m < ell:
            a = np.sqrt((ell - m) * (ell + m + 1))
            J_plus[i, i + 1] = a

        if m > -ell:
            a = np.sqrt((ell + m) * (ell - m + 1))
            J_minus[i, i - 1] = a

    J_x = (J_plus + J_minus) / 2.0
    J_y = (J_plus - J_minus) / (2.0j)

    # Spherical -> tesseral transformation
    S = _build_tesseral_transform(ell)
    S_inv = S.conj().T

    def _to_real_skew(J_complex: np.ndarray) -> np.ndarray:
        """Transform complex operator to real skew-symmetric generator."""
        # G = Re(S (iJ) S^dagger) -- factor of i makes it skew-symmetric
        G_complex = S @ (1j * J_complex) @ S_inv
        G_real = G_complex.real
        G_skew = 0.5 * (G_real - G_real.T)
        return G_skew

    G_x = _to_real_skew(J_x)
    G_y = _to_real_skew(J_y)
    G_z = _to_real_skew(J_z)

    G = np.stack([G_x, G_y, G_z], axis=0)

    return G.astype(np.float32, copy=False)


def _build_tesseral_transform(ell: int) -> np.ndarray:
    """Construct unitary transformation from spherical to tesseral basis.

    Tesseral harmonics are real combinations of spherical harmonics:
        Y^c_{lm} = (Y_{lm} + (-1)^m Y_{l,-m}) / sqrt(2)   (cosine, m > 0)
        Y^s_{lm} = (Y_{lm} - (-1)^m Y_{l,-m}) / (i*sqrt(2)) (sine, m > 0)
        Y^0_{l0} = Y_{l0}                                    (m = 0)

    Args:
        ell: Spin quantum number

    Returns:
        S: (K, K) unitary matrix, complex128
    """
    K = 2 * ell + 1
    S = np.zeros((K, K), dtype=np.complex128)

    S[0, ell] = 1.0

    row = 1
    for m in range(1, ell + 1):
        phase = (-1) ** m
        sqrt2_inv = 1.0 / np.sqrt(2.0)

        # Cosine-like
        S[row, ell + m] = sqrt2_inv
        S[row, ell - m] = phase * sqrt2_inv
        row += 1

        # Sine-like
        S[row, ell + m] = -1j * sqrt2_inv
        S[row, ell - m] = 1j * phase * sqrt2_inv
        row += 1

    return S


def _validate_so3_generators(
    G: np.ndarray,
    *,
    eps: float = 1e-6,
    verbose: bool = False,
) -> None:
    """Validate SO(3) generators: skew-symmetry, commutation [G_x,G_y]=G_z, and Casimir.

    Args:
        G: (3, K, K) generators
        eps: Tolerance for checks
        verbose: If True, print validation details
    """
    if G.shape[0] != 3:
        raise ValueError(f"Expected 3 generators (x,y,z), got {G.shape[0]}")

    K = G.shape[1]
    if G.shape != (3, K, K):
        raise ValueError(f"Expected shape (3, K, K), got {G.shape}")

    G64 = G.astype(np.float64)
    G_x, G_y, G_z = G64[0], G64[1], G64[2]

    for a, name in enumerate(['x', 'y', 'z']):
        G_a = G64[a]
        skew_error = np.linalg.norm(G_a + G_a.T, ord='fro')
        if skew_error > eps:
            raise RuntimeError(
                f"Generator G_{name} not skew-symmetric: ||G + Gᵀ|| = {skew_error:.3e}"
            )

    comm_xy = G_x @ G_y - G_y @ G_x
    error_xy = np.linalg.norm(comm_xy - G_z, ord='fro')

    comm_yz = G_y @ G_z - G_z @ G_y
    error_yz = np.linalg.norm(comm_yz - G_x, ord='fro')

    comm_zx = G_z @ G_x - G_x @ G_z
    error_zx = np.linalg.norm(comm_zx - G_y, ord='fro')

    max_error = max(error_xy, error_yz, error_zx)

    # Scale tolerance by generator norm squared (commutator errors scale as norm^2)
    scale = max(np.linalg.norm(G64[a], ord='fro') for a in range(3))
    threshold = eps * max(scale * scale, 1.0)

    if max_error > threshold:
        raise RuntimeError(
            f"SO(3) commutation relations violated:\n"
            f"  [G_x, G_y] - G_z: {error_xy:.3e}\n"
            f"  [G_y, G_z] - G_x: {error_yz:.3e}\n"
            f"  [G_z, G_x] - G_y: {error_zx:.3e}\n"
            f"  threshold: {threshold:.3e}"
        )

    C_2 = -sum(G64[a] @ G64[a] for a in range(3))

    eigenvalues    = np.linalg.eigvalsh(C_2)
    casimir_value  = float(np.mean(eigenvalues))
    casimir_spread = float(np.std(eigenvalues))

    ell = (K - 1) // 2
    casimir_expected = ell * (ell + 1)
    casimir_error = abs(casimir_value - casimir_expected)

    # Scale tolerance by Casimir magnitude
    base = max(abs(casimir_expected), 1.0)
    tol  = eps * base

    if casimir_error > tol or casimir_spread > tol:
        raise RuntimeError(
            "Casimir operator check failed:\n"
            f"  Expected: {casimir_expected}\n"
            f"  Got: {casimir_value:.6f} ± {casimir_spread:.3e}\n"
            f"  Error: {casimir_error:.3e}"
        )


    if verbose:
        print("✓ SO(3) generator validation passed:")
        print(f"  Dimension: K = {K} (ℓ = {ell})")
        print(f"  Skew-symmetry: max error = {max([np.linalg.norm(G64[a] + G64[a].T) for a in range(3)]):.3e}")
        print(f"  Commutation: max error = {max_error:.3e}")
        print(f"  Casimir: C₂ = {casimir_value:.6f} (expected {casimir_expected})")


def generate_multi_irrep_generators(
    irrep_spec: list,
    *,
    validate: bool = True,
    eps: float = 1e-6,
) -> np.ndarray:
    """Generate block-diagonal SO(3) generators from a multi-irrep specification.

    Creates generators acting on V = direct_sum_l (V_l)^{n_l}.

    Args:
        irrep_spec: List of (label, multiplicity, dim) tuples. dim must be odd.
        validate: If True, verify the resulting generators
        eps: Tolerance for validation

    Returns:
        G: Block-diagonal generators, shape (3, K, K), where K = sum(mult * dim)
    """
    for label, mult, dim in irrep_spec:
        if dim % 2 == 0:
            raise ValueError(
                f"Irrep '{label}' has even dimension {dim}. "
                f"SO(3) irreps must have odd dimension (2ℓ+1)."
            )
        if mult < 0:
            raise ValueError(f"Irrep '{label}' has negative multiplicity {mult}.")

    K = sum(mult * dim for _, mult, dim in irrep_spec)

    G = np.zeros((3, K, K), dtype=np.float32)

    idx = 0
    for label, mult, dim in irrep_spec:
        if dim == 1:
            idx += mult * dim
        else:
            G_irrep = generate_so3_generators(dim, cache=True, validate=False)

            for _ in range(mult):
                G[:, idx:idx+dim, idx:idx+dim] = G_irrep
                idx += dim

    if validate and K > 1:
        _validate_block_diagonal_generators(G, irrep_spec, eps=eps)

    return G


def _validate_block_diagonal_generators(
    G: np.ndarray,
    irrep_spec: list,
    *,
    eps: float = 1e-6,
) -> None:
    """Validate block-diagonal multi-irrep generators (skew-symmetry, commutation, block structure).

    Args:
        G: (3, K, K) generators
        irrep_spec: The irrep specification used to create G
        eps: Tolerance for checks
    """
    K = G.shape[1]

    G64 = G.astype(np.float64)

    for a in range(3):
        skew_error = np.linalg.norm(G64[a] + G64[a].T, ord='fro')
        if skew_error > eps:
            raise RuntimeError(
                f"Block-diagonal generator G[{a}] not skew-symmetric: "
                f"||G + Gᵀ|| = {skew_error:.3e}"
            )

    G_x, G_y, G_z = G64[0], G64[1], G64[2]

    comm_xy = G_x @ G_y - G_y @ G_x
    error_xy = np.linalg.norm(comm_xy - G_z, ord='fro')

    comm_yz = G_y @ G_z - G_z @ G_y
    error_yz = np.linalg.norm(comm_yz - G_x, ord='fro')

    comm_zx = G_z @ G_x - G_x @ G_z
    error_zx = np.linalg.norm(comm_zx - G_y, ord='fro')

    max_error = max(error_xy, error_yz, error_zx)

    # Scale tolerance by generator norm squared (commutator errors scale as norm^2)
    scale = max(np.linalg.norm(G64[a], ord='fro') for a in range(3))
    threshold = eps * max(scale * scale, 1.0)

    if max_error > threshold:
        raise RuntimeError(
            f"Block-diagonal SO(3) commutation violated:\n"
            f"  [G_x, G_y] - G_z: {error_xy:.3e}\n"
            f"  [G_y, G_z] - G_x: {error_yz:.3e}\n"
            f"  [G_z, G_x] - G_y: {error_zx:.3e}\n"
            f"  threshold: {threshold:.3e}"
        )

    idx = 0
    block_starts = []
    for _, mult, dim in irrep_spec:
        for _ in range(mult):
            block_starts.append((idx, dim))
            idx += dim

    for i, (start_i, dim_i) in enumerate(block_starts):
        for j, (start_j, dim_j) in enumerate(block_starts):
            if i != j:
                for a in range(3):
                    block = G64[a, start_i:start_i+dim_i, start_j:start_j+dim_j]
                    block_norm = np.linalg.norm(block, ord='fro')
                    if block_norm > eps:
                        raise RuntimeError(
                            f"Off-diagonal block ({i},{j}) is non-zero: "
                            f"||block|| = {block_norm:.3e}"
                        )


def generate_soN_generators(
    N: int,
    *,
    validate: bool = True,
    eps: float = 1e-6,
) -> np.ndarray:
    """Generate SO(N) generators in the fundamental representation.

    Basis elements L_{ij} for i < j: (L_{ij})_{kl} = d_{ik}d_{jl} - d_{il}d_{jk}

    Args:
        N: Dimension of the fundamental representation (N >= 2)
        validate: If True, verify commutation relations
        eps: Tolerance for validation

    Returns:
        G: Generators array, shape (N(N-1)/2, N, N), float32
    """
    if N < 2:
        raise ValueError(f"N must be >= 2 for SO(N), got N={N}")

    n_generators = N * (N - 1) // 2
    G = np.zeros((n_generators, N, N), dtype=np.float32)

    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            G[idx, i, j] = 1.0
            G[idx, j, i] = -1.0
            idx += 1

    if validate:
        _validate_soN_generators(G, N, eps=eps)

    return G


def _validate_soN_generators(
    G: np.ndarray,
    N: int,
    *,
    eps: float = 1e-6,
) -> None:
    """Validate SO(N) generators (skew-symmetry and commutation relations).

    Args:
        G: (n_gen, N, N) generators where n_gen = N(N-1)/2
        N: Dimension of fundamental rep
        eps: Tolerance for checks
    """
    n_gen = G.shape[0]
    expected_n_gen = N * (N - 1) // 2

    if n_gen != expected_n_gen:
        raise ValueError(
            f"Expected {expected_n_gen} generators for SO({N}), got {n_gen}"
        )

    for a in range(n_gen):
        skew_error = np.linalg.norm(G[a] + G[a].T, ord='fro')
        if skew_error > eps:
            raise RuntimeError(
                f"SO({N}) generator G[{a}] not skew-symmetric: "
                f"||G + Gᵀ|| = {skew_error:.3e}"
            )

    idx_map = {}
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            idx_map[(i, j)] = idx
            idx += 1

    # Check [L_{ij}, L_{jk}] = L_{ik} for i < j < k
    max_error = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                a = idx_map[(i, j)]
                b = idx_map[(j, k)]
                c = idx_map[(i, k)]

                comm = G[a] @ G[b] - G[b] @ G[a]
                error = np.linalg.norm(comm - G[c], ord='fro')
                max_error = max(max_error, error)

    if max_error > eps:
        raise RuntimeError(
            f"SO({N}) commutation relations violated, max error: {max_error:.3e}"
        )


def generate_glK_generators(
    K: int,
    *,
    include_identity: bool = False,
) -> np.ndarray:
    """Generate gl(K) Lie algebra generators (full K^2 basis of elementary matrices E_ij).

    Args:
        K: Matrix dimension
        include_identity: If True, include trace component (K^2 generators).
                         If False, use sl(K) traceless basis (K^2-1 generators).

    Returns:
        G: Generators array, shape (K^2, K, K) or (K^2-1, K, K), float32
    """
    if K < 1:
        raise ValueError(f"K must be >= 1 for GL(K), got K={K}")

    n_generators = K * K
    G = np.zeros((n_generators, K, K), dtype=np.float32)

    idx = 0
    for i in range(K):
        for j in range(K):
            G[idx, i, j] = 1.0
            idx += 1

    if not include_identity:
        # Project out trace component to get sl(K)
        I_K = np.eye(K, dtype=np.float32)
        trace_dir = I_K / np.sqrt(K)
        projected = []
        for g in range(n_generators):
            overlap = np.sum(G[g] * trace_dir)
            G_proj = G[g] - overlap * trace_dir
            if np.linalg.norm(G_proj) > 1e-8:
                projected.append(G_proj)
        G = np.stack(projected, axis=0)

    return G


def generate_glK_multihead_generators(
    K: int,
    n_heads: int,
    *,
    include_identity: bool = True,
) -> np.ndarray:
    """Generate block-diagonal gl(d_head) generators for multi-head GL(K) attention.

    Each head gets independent GL(d_head) gauge structure where d_head = K // n_heads.

    Args:
        K: Total embedding dimension
        n_heads: Number of attention heads
        include_identity: If True, include trace component per head

    Returns:
        G: Block-diagonal generators, shape (n_heads * d_head^2, K, K)
    """
    if K % n_heads != 0:
        raise ValueError(
            f"K={K} must be divisible by n_heads={n_heads}. "
            f"Got K % n_heads = {K % n_heads}"
        )

    d_head = K // n_heads
    n_gen_per_head = d_head * d_head
    n_generators = n_heads * n_gen_per_head

    G = np.zeros((n_generators, K, K), dtype=np.float32)

    for h in range(n_heads):
        start = h * d_head
        end = (h + 1) * d_head
        gen_offset = h * n_gen_per_head

        idx = 0
        for i in range(d_head):
            for j in range(d_head):
                G[gen_offset + idx, start + i, start + j] = 1.0
                idx += 1

    return G


def generate_glK_cross_head_generators(
    K: int,
    n_heads: int,
    cross_couplings: 'List[Tuple[int, int]]',
) -> np.ndarray:
    """Generate GL(K) generators with sparse off-diagonal cross-head coupling.

    Adds generators for selected head pairs on top of per-head gl(d_head) blocks,
    enabling gauge transport that mixes information between heads.

    Args:
        K: Total embedding dimension
        n_heads: Number of attention heads
        cross_couplings: List of (head_a, head_b) directed pairs to couple (a != b).

    Returns:
        G: Generators, shape (n_heads*d^2 + len(couplings)*d^2, K, K)
    """
    if K % n_heads != 0:
        raise ValueError(f"K={K} not divisible by n_heads={n_heads}")

    d_head = K // n_heads
    n_gen_diag = n_heads * d_head * d_head
    n_gen_cross = len(cross_couplings) * d_head * d_head
    n_gen_total = n_gen_diag + n_gen_cross

    G = np.zeros((n_gen_total, K, K), dtype=np.float32)

    # Diagonal blocks
    for h in range(n_heads):
        start = h * d_head
        gen_offset = h * d_head * d_head
        idx = 0
        for i in range(d_head):
            for j in range(d_head):
                G[gen_offset + idx, start + i, start + j] = 1.0
                idx += 1

    # Off-diagonal blocks for each coupling pair
    for pair_idx, (a, b) in enumerate(cross_couplings):
        if a == b:
            raise ValueError(f"Self-coupling ({a},{a}) not allowed — already in diagonal")
        if not (0 <= a < n_heads and 0 <= b < n_heads):
            raise ValueError(f"Head indices ({a},{b}) out of range [0, {n_heads})")

        a_start = a * d_head
        b_start = b * d_head
        gen_offset = n_gen_diag + pair_idx * d_head * d_head
        idx = 0
        for i in range(d_head):
            for j in range(d_head):
                G[gen_offset + idx, a_start + i, b_start + j] = 1.0
                idx += 1

    return G


def merge_coupled_heads(
    n_heads: int,
    d_head: int,
    cross_couplings: 'List[Tuple[int, int]]',
) -> 'Tuple[List[int], List[List[int]]]':
    """Compute super-block structure from cross-head coupling via union-find.

    Transitively connected heads are merged into a single super-block for
    block-diagonal KL computation.

    Args:
        n_heads: Number of attention heads
        d_head: Dimension per head
        cross_couplings: List of (head_a, head_b) directed coupling pairs.

    Returns:
        super_block_dims: List of block dimensions (sum = K).
        super_block_head_groups: List of lists of head indices per super-block.
    """
    parent = list(range(n_heads))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for a, b in cross_couplings:
        union(a, b)

    from collections import defaultdict
    groups = defaultdict(list)
    for h in range(n_heads):
        groups[find(h)].append(h)

    sorted_groups = sorted(groups.values(), key=lambda g: g[0])

    super_block_dims = [len(g) * d_head for g in sorted_groups]
    super_block_head_groups = sorted_groups

    return super_block_dims, super_block_head_groups


def reorder_cross_head_generators(
    G: np.ndarray,
    n_heads: int,
    d_head: int,
    cross_couplings: 'List[Tuple[int, int]]',
    super_block_head_groups: 'List[List[int]]',
) -> 'Tuple[np.ndarray, List[int]]':
    """Reorder generators so that merged super-blocks are contiguous.

    Args:
        G: Generators from generate_glK_cross_head_generators, shape (n_gen, K, K)
        n_heads: Number of original heads
        d_head: Dimension per head
        cross_couplings: Coupling pairs (for reference)
        super_block_head_groups: From merge_coupled_heads

    Returns:
        G_reordered: Generators with permuted rows/cols, shape (n_gen, K, K)
        perm: Permutation vector of length K
    """
    K = n_heads * d_head

    perm = []
    for group in super_block_head_groups:
        for h in group:
            perm.extend(range(h * d_head, (h + 1) * d_head))

    perm = np.array(perm, dtype=np.intp)
    assert len(perm) == K and len(set(perm)) == K, "Permutation must be a bijection"

    G_reordered = G[:, perm][:, :, perm]

    return G_reordered, perm


def generate_multi_irrep_soN_generators(
    irrep_spec: list,
    N: int,
    *,
    validate: bool = True,
    eps: float = 1e-6,
) -> np.ndarray:
    """Generate block-diagonal SO(N) generators from a multi-irrep specification.

    Supported irrep types: scalar (dim=1), fund (dim=N),
    wedge2 (dim=N(N-1)/2), sym2 (dim=N(N+1)/2-1).

    Args:
        irrep_spec: List of (label, multiplicity, dim) tuples.
        N: The gauge group dimension (SO(N))
        validate: If True, verify the resulting generators
        eps: Tolerance for validation

    Returns:
        G: Block-diagonal generators, shape (N(N-1)/2, K, K)
    """
    expected_dims = {
        'scalar': 1,
        'fund': N,
        'fundamental': N,
        'vector': N,
        'wedge2': N * (N - 1) // 2,
        'antisym2': N * (N - 1) // 2,
        'exterior2': N * (N - 1) // 2,
        'sym2': N * (N + 1) // 2 - 1,
        'sym2_traceless': N * (N + 1) // 2 - 1,
        'symmetric2': N * (N + 1) // 2 - 1,
    }

    for label, mult, dim in irrep_spec:
        label_lower = label.lower()

        if label_lower in expected_dims:
            expected_dim = expected_dims[label_lower]
            if dim != expected_dim:
                raise ValueError(
                    f"Irrep '{label}' should have dim={expected_dim} for SO({N}), "
                    f"but got dim={dim}."
                )
        else:
            if dim == 1:
                pass
            elif dim == N:
                pass
            elif dim == N * (N - 1) // 2:
                pass
            elif dim == N * (N + 1) // 2 - 1:
                pass
            else:
                raise ValueError(
                    f"Irrep '{label}' has dimension {dim}, which doesn't match any "
                    f"implemented SO({N}) irrep. Supported dims: 1 (scalar), "
                    f"{N} (fund), {N*(N-1)//2} (wedge2), {N*(N+1)//2-1} (sym2)."
                )

        if mult < 0:
            raise ValueError(f"Irrep '{label}' has negative multiplicity {mult}.")

    K = sum(mult * dim for _, mult, dim in irrep_spec)

    n_gen = N * (N - 1) // 2

    G = np.zeros((n_gen, K, K), dtype=np.float32)

    G_fund = None
    G_wedge2 = None
    G_sym2 = None

    idx = 0
    for label, mult, dim in irrep_spec:
        if dim == 1:
            idx += mult * dim

        elif dim == N:
            if G_fund is None:
                G_fund = generate_soN_generators(N, validate=False)
            for _ in range(mult):
                G[:, idx:idx+dim, idx:idx+dim] = G_fund
                idx += dim

        elif dim == N * (N - 1) // 2:
            if G_wedge2 is None:
                G_wedge2 = generate_wedge2_generators(N, validate=False)
            for _ in range(mult):
                G[:, idx:idx+dim, idx:idx+dim] = G_wedge2
                idx += dim

        elif dim == N * (N + 1) // 2 - 1:
            if G_sym2 is None:
                G_sym2 = generate_sym2_traceless_generators(N, validate=False)
            for _ in range(mult):
                G[:, idx:idx+dim, idx:idx+dim] = G_sym2
                idx += dim

        else:
            raise RuntimeError(f"Unexpected dimension {dim} for irrep '{label}'")

    if validate and K > 1:
        _validate_block_diagonal_soN_generators(G, irrep_spec, N, eps=eps)

    return G


def _wedge2_index_to_pair(idx: int, N: int) -> tuple:
    """Map linear index to (i,j) pair with i < j for wedge-2 basis."""
    i = 0
    count = 0
    while count + (N - 1 - i) <= idx:
        count += N - 1 - i
        i += 1
    j = idx - count + i + 1
    return i, j


def _wedge2_pair_to_index(i: int, j: int, N: int) -> int:
    """Map (i,j) pair with i < j to linear index for wedge-2 basis."""
    return i * N - i * (i + 1) // 2 + (j - i - 1)


def generate_wedge2_generators(
    N: int,
    *,
    validate: bool = True,
    eps: float = 1e-6,
) -> np.ndarray:
    """Generate SO(N) generators for wedge-2 V (antisymmetric 2-tensor representation).

    The Lie algebra action is the commutator: G . X = [G, X] = GX - XG.

    Args:
        N: Dimension of the fundamental representation (N >= 2)
        validate: If True, verify the generators
        eps: Tolerance for validation

    Returns:
        G: Generators, shape (N(N-1)/2, N(N-1)/2, N(N-1)/2)
    """
    if N < 2:
        raise ValueError(f"N must be >= 2 for SO(N), got N={N}")

    n_gen = N * (N - 1) // 2
    dim = N * (N - 1) // 2

    G_fund = generate_soN_generators(N, validate=False)

    G_wedge2 = np.zeros((n_gen, dim, dim), dtype=np.float32)

    for a in range(n_gen):
        G_a = G_fund[a]

        for p in range(dim):
            i, j = _wedge2_index_to_pair(p, N)

            # Basis element: E_ij - E_ji (antisymmetric)
            X = np.zeros((N, N), dtype=np.float32)
            X[i, j] = 1.0
            X[j, i] = -1.0

            comm = G_a @ X - X @ G_a

            for q in range(dim):
                k, l = _wedge2_index_to_pair(q, N)
                G_wedge2[a, q, p] = comm[k, l]

    if validate:
        _validate_wedge2_generators(G_wedge2, N, eps=eps)

    return G_wedge2


def _validate_wedge2_generators(
    G: np.ndarray,
    N: int,
    *,
    eps: float = 1e-6,
) -> None:
    """Validate wedge-2 V generators (shape, skew-symmetry, sample commutation)."""
    n_gen, dim, _ = G.shape

    expected_n_gen = N * (N - 1) // 2
    expected_dim = N * (N - 1) // 2

    if n_gen != expected_n_gen:
        raise ValueError(f"Expected {expected_n_gen} generators, got {n_gen}")
    if dim != expected_dim:
        raise ValueError(f"Expected dim {expected_dim}, got {dim}")

    for a in range(n_gen):
        skew_error = np.linalg.norm(G[a] + G[a].T, ord='fro')
        if skew_error > eps:
            raise RuntimeError(
                f"∧² generator G[{a}] not skew-symmetric: ||G + Gᵀ|| = {skew_error:.3e}"
            )

    if n_gen >= 3:
        comm_01 = G[0] @ G[1] - G[1] @ G[0]
        if np.linalg.norm(comm_01 + comm_01.T, ord='fro') > eps:
            raise RuntimeError("Commutator [G_0, G_1] in ∧² rep not skew-symmetric")


def _sym2_traceless_basis_size(N: int) -> int:
    """Dimension of Sym^2_0 V (symmetric traceless 2-tensors)."""
    return N * (N + 1) // 2 - 1


def _sym2_traceless_index_to_components(idx: int, N: int) -> tuple:
    """Map linear index to symmetric traceless basis element type and data.

    Returns:
        (type, data): 'offdiag' with (i,j) or 'diag' with index k.
    """
    n_offdiag = N * (N - 1) // 2

    if idx < n_offdiag:
        i, j = _wedge2_index_to_pair(idx, N)
        return ('offdiag', (i, j))
    else:
        k = idx - n_offdiag
        return ('diag', k)


def _build_sym2_traceless_basis_element(idx: int, N: int) -> np.ndarray:
    """Build the idx-th orthonormal basis element of Sym^2_0 V as an N x N matrix."""
    n_offdiag = N * (N - 1) // 2
    X = np.zeros((N, N), dtype=np.float32)

    if idx < n_offdiag:
        # Off-diagonal: (E_ij + E_ji) / sqrt(2)
        i, j = _wedge2_index_to_pair(idx, N)
        X[i, j] = 1.0 / np.sqrt(2)
        X[j, i] = 1.0 / np.sqrt(2)
    else:
        # Diagonal traceless: Gell-Mann-like basis
        # Element k: (E_00 + ... + E_kk - (k+1)*E_{k+1,k+1}) / sqrt((k+1)(k+2))
        k = idx - n_offdiag
        norm = np.sqrt((k + 1) * (k + 2))
        for i in range(k + 1):
            X[i, i] = 1.0 / norm
        X[k + 1, k + 1] = -(k + 1) / norm

    return X


def generate_sym2_traceless_generators(
    N: int,
    *,
    validate: bool = True,
    eps: float = 1e-6,
) -> np.ndarray:
    """Generate SO(N) generators for Sym^2_0 V (symmetric traceless 2-tensor rep).

    The Lie algebra action is the commutator: G . X = [G, X] = GX - XG.

    Args:
        N: Dimension of the fundamental representation (N >= 2)
        validate: If True, verify the generators
        eps: Tolerance for validation

    Returns:
        G: Generators, shape (N(N-1)/2, N(N+1)/2-1, N(N+1)/2-1)
    """
    if N < 2:
        raise ValueError(f"N must be >= 2 for SO(N), got N={N}")

    n_gen = N * (N - 1) // 2
    dim = _sym2_traceless_basis_size(N)

    G_fund = generate_soN_generators(N, validate=False)

    basis = [_build_sym2_traceless_basis_element(p, N) for p in range(dim)]

    G_sym2 = np.zeros((n_gen, dim, dim), dtype=np.float32)

    for a in range(n_gen):
        G_a = G_fund[a]

        for p in range(dim):
            X = basis[p]

            # [G_a, X] is symmetric and traceless when G is skew and X is symmetric
            comm = G_a @ X - X @ G_a

            for q in range(dim):
                Y = basis[q]
                # Inner product: tr(Y^T comm)
                G_sym2[a, q, p] = np.sum(Y * comm)

    if validate:
        _validate_sym2_traceless_generators(G_sym2, N, eps=eps)

    return G_sym2


def _validate_sym2_traceless_generators(
    G: np.ndarray,
    N: int,
    *,
    eps: float = 1e-6,
) -> None:
    """Validate Sym^2_0 V generators (shape, skew-symmetry, sample commutation)."""
    n_gen, dim, _ = G.shape

    expected_n_gen = N * (N - 1) // 2
    expected_dim = _sym2_traceless_basis_size(N)

    if n_gen != expected_n_gen:
        raise ValueError(f"Expected {expected_n_gen} generators, got {n_gen}")
    if dim != expected_dim:
        raise ValueError(f"Expected dim {expected_dim}, got {dim}")

    for a in range(n_gen):
        skew_error = np.linalg.norm(G[a] + G[a].T, ord='fro')
        if skew_error > eps:
            raise RuntimeError(
                f"Sym²₀ generator G[{a}] not skew-symmetric: ||G + Gᵀ|| = {skew_error:.3e}"
            )

    if n_gen >= 3:
        comm_01 = G[0] @ G[1] - G[1] @ G[0]
        if np.linalg.norm(comm_01 + comm_01.T, ord='fro') > eps:
            raise RuntimeError("Commutator [G_0, G_1] in Sym²₀ rep not skew-symmetric")


def _validate_block_diagonal_soN_generators(
    G: np.ndarray,
    irrep_spec: list,
    N: int,
    *,
    eps: float = 1e-6,
) -> None:
    """Validate block-diagonal multi-irrep SO(N) generators (skew-symmetry, commutation, blocks)."""
    n_gen = G.shape[0]
    K = G.shape[1]

    expected_n_gen = N * (N - 1) // 2
    if n_gen != expected_n_gen:
        raise ValueError(
            f"Expected {expected_n_gen} generators for SO({N}), got {n_gen}"
        )

    for a in range(n_gen):
        skew_error = np.linalg.norm(G[a] + G[a].T, ord='fro')
        if skew_error > eps:
            raise RuntimeError(
                f"Block-diagonal SO({N}) generator G[{a}] not skew-symmetric: "
                f"||G + Gᵀ|| = {skew_error:.3e}"
            )

    if n_gen >= 3:
        G_0, G_1, G_2 = G[0], G[1], G[2]

        comm_01 = G_0 @ G_1 - G_1 @ G_0
        if np.linalg.norm(comm_01 + comm_01.T, ord='fro') > eps:
            raise RuntimeError("Commutator [G_0, G_1] not skew-symmetric")

    idx = 0
    block_starts = []
    for _, mult, dim in irrep_spec:
        for _ in range(mult):
            block_starts.append((idx, dim))
            idx += dim

    for i, (start_i, dim_i) in enumerate(block_starts):
        for j, (start_j, dim_j) in enumerate(block_starts):
            if i != j:
                for a in range(min(n_gen, 10)):
                    block = G[a, start_i:start_i+dim_i, start_j:start_j+dim_j]
                    block_norm = np.linalg.norm(block, ord='fro')
                    if block_norm > eps:
                        raise RuntimeError(
                            f"Off-diagonal block ({i},{j}) in generator {a} "
                            f"is non-zero: ||block|| = {block_norm:.3e}"
                        )


def _get_soN_gauge_generators(n_gen: int, device, dtype) -> 'torch.Tensor':
    """Get N x N canonical so(N) basis elements L_{ij} for BCH composition."""
    import torch
    import math

    N = int((1 + math.sqrt(1 + 8 * n_gen)) / 2)

    if N * (N - 1) // 2 != n_gen:
        raise ValueError(f"n_gen={n_gen} doesn't correspond to valid SO(N)")

    generators = torch.zeros(n_gen, N, N, device=device, dtype=dtype)
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            generators[idx, i, j] = 1.0
            generators[idx, j, i] = -1.0
            idx += 1

    return generators


def soN_bracket_torch(
    phi1: 'torch.Tensor',
    phi2: 'torch.Tensor',
    generators: 'torch.Tensor',
) -> 'torch.Tensor':
    """Compute Lie bracket [phi1.G, phi2.G] in so(N), returning coordinates.

    Args:
        phi1: First element coordinates (..., n_gen)
        phi2: Second element coordinates (..., n_gen)
        generators: Transport generators (n_gen, K, K) -- used only for n_gen count

    Returns:
        bracket_coords: Coordinates of the bracket (..., n_gen)
    """
    import torch

    n_gen = generators.shape[0]

    gauge_gens = _get_soN_gauge_generators(n_gen, phi1.device, phi1.dtype)

    A1 = torch.einsum('...a,aij->...ij', phi1, gauge_gens)
    A2 = torch.einsum('...a,aij->...ij', phi2, gauge_gens)

    bracket = A1 @ A2 - A2 @ A1

    bracket_coords = extract_soN_coords_torch(bracket, gauge_gens)

    return bracket_coords


def extract_soN_coords_torch(
    A: 'torch.Tensor',
    generators: 'torch.Tensor',
) -> 'torch.Tensor':
    """Extract so(N) coordinates from a skew-symmetric matrix (upper-triangular elements).

    Args:
        A: Skew-symmetric matrix (..., N, N)
        generators: Generators (n_gen, K, K)

    Returns:
        phi: Lie algebra coordinates (..., n_gen)
    """
    import torch
    import math

    n_gen = generators.shape[0]
    M = A.shape[-1]

    N = int((1 + math.sqrt(1 + 8 * n_gen)) / 2)

    if N * (N - 1) // 2 != n_gen:
        raise ValueError(f"n_gen={n_gen} doesn't correspond to valid SO(N). "
                        f"Expected N*(N-1)/2 for some integer N.")

    if M != N:
        raise ValueError(f"Matrix A has dimension {M}x{M} but gauge group is SO({N}). "
                        f"For BCH composition, need {N}x{N} matrices.")

    batch_shape = A.shape[:-2]
    phi = torch.zeros(*batch_shape, n_gen, device=A.device, dtype=A.dtype)

    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            phi[..., idx] = A[..., i, j]
            idx += 1

    return phi


def soN_compose_bch_torch(
    phi1: 'torch.Tensor',
    phi2: 'torch.Tensor',
    generators: 'torch.Tensor',
    order: int = 1,
) -> 'torch.Tensor':
    """Compose two so(N) elements via BCH: log(exp(phi1.G) exp(phi2.G)).

    Args:
        phi1: First so(N) element (..., n_gen)
        phi2: Second so(N) element (..., n_gen)
        generators: Lie algebra generators (n_gen, N, N)
        order: BCH order (0=addition, 1=first correction, 2=second)

    Returns:
        phi_composed: Composed element (..., n_gen)
    """
    if order == 0:
        return phi1 + phi2

    bracket_12 = soN_bracket_torch(phi1, phi2, generators)
    result = phi1 + phi2 + 0.5 * bracket_12

    if order >= 2:
        bracket_1_12 = soN_bracket_torch(phi1, bracket_12, generators)
        bracket_2_12 = soN_bracket_torch(phi2, bracket_12, generators)
        result = result + (1.0/12.0) * bracket_1_12 - (1.0/12.0) * bracket_2_12

    return result


def retract_soN_torch(
    phi: 'torch.Tensor',
    delta_phi: 'torch.Tensor',
    generators: 'torch.Tensor',
    step_size: float = 1.0,
    trust_region: float = 0.3,
    max_norm: float = 3.14159,
    bch_order: int = 1,
    eps: float = 1e-6,
) -> 'torch.Tensor':
    """Retract phi update onto SO(N) manifold with trust region and BCH composition.

    Args:
        phi: Current gauge frames (..., n_gen)
        delta_phi: Update direction (..., n_gen)
        generators: Lie algebra generators (n_gen, N, N)
        step_size: Learning rate
        trust_region: Max relative change ||delta|| / ||phi||
        max_norm: Max allowed norm (pi = 180 deg rotation)
        bch_order: BCH expansion order
        eps: Numerical stability constant

    Returns:
        phi_new: Updated gauge frames (..., n_gen)
    """
    import torch

    update = step_size * delta_phi

    phi_norm = torch.norm(phi, dim=-1, keepdim=True).clamp(min=0.1)
    update_norm = torch.norm(update, dim=-1, keepdim=True)

    scale = torch.clamp(trust_region * phi_norm / (update_norm + eps), max=1.0)
    update = scale * update

    phi_new = soN_compose_bch_torch(phi, update, generators, order=bch_order)

    phi_new_norm = torch.norm(phi_new, dim=-1, keepdim=True)
    phi_new = torch.where(
        phi_new_norm > max_norm,
        phi_new * (max_norm / (phi_new_norm + eps)),
        phi_new
    )

    return phi_new


def _get_glK_gauge_generators(
    n_gen: int,
    device: 'torch.device',
    dtype: 'torch.dtype',
) -> 'torch.Tensor':
    """Get K x K elementary matrices E_ij for GL(K) where n_gen = K^2.

    Args:
        n_gen: Number of generators (must be a perfect square)
        device: PyTorch device
        dtype: PyTorch dtype

    Returns:
        generators: (K^2, K, K) tensor
    """
    import torch
    import math

    K = int(math.sqrt(n_gen))

    if K * K != n_gen:
        raise ValueError(f"n_gen={n_gen} is not a perfect square (needed for GL(K))")

    generators = torch.zeros(n_gen, K, K, device=device, dtype=dtype)
    idx = 0
    for i in range(K):
        for j in range(K):
            generators[idx, i, j] = 1.0
            idx += 1

    return generators


def glK_bracket_torch(
    phi1: 'torch.Tensor',
    phi2: 'torch.Tensor',
    generators: 'torch.Tensor',
) -> 'torch.Tensor':
    """Compute Lie bracket [phi1.G, phi2.G] in gl(K), returning coordinates.

    Args:
        phi1: First element coordinates (..., n_gen) where n_gen = K^2
        phi2: Second element coordinates (..., n_gen)
        generators: Transport generators (n_gen, dim, dim) -- used only for n_gen count

    Returns:
        bracket_coords: Coordinates of the bracket (..., n_gen)
    """
    import torch

    n_gen = generators.shape[0]

    gauge_gens = _get_glK_gauge_generators(n_gen, phi1.device, phi1.dtype)

    A1 = torch.einsum('...a,aij->...ij', phi1, gauge_gens)
    A2 = torch.einsum('...a,aij->...ij', phi2, gauge_gens)

    bracket = A1 @ A2 - A2 @ A1

    bracket_coords = extract_glK_coords_torch(bracket, gauge_gens)

    return bracket_coords


def extract_glK_coords_torch(
    A: 'torch.Tensor',
    generators: 'torch.Tensor',
) -> 'torch.Tensor':
    """Extract gl(K) coordinates from a matrix (flattens K x K -> K^2).

    Args:
        A: Matrix (..., K, K)
        generators: Gauge generators (n_gen, K, K) -- used for shape only

    Returns:
        phi: Coordinates (..., n_gen) where n_gen = K^2
    """
    import torch

    K = A.shape[-1]
    batch_shape = A.shape[:-2]

    phi = A.reshape(batch_shape + (K * K,))

    return phi


def glK_compose_bch_torch(
    phi1: 'torch.Tensor',
    phi2: 'torch.Tensor',
    generators: 'torch.Tensor',
    order: int = 1,
) -> 'torch.Tensor':
    """Compose two gl(K) elements via BCH: log(exp(phi1.G) exp(phi2.G)).

    Args:
        phi1: First gl(K) element (..., n_gen) where n_gen = K^2
        phi2: Second gl(K) element (..., n_gen)
        generators: Lie algebra generators (n_gen, dim, dim)
        order: BCH order (0=addition, 1=first correction, 2=second)

    Returns:
        phi_composed: Composed element (..., n_gen)
    """
    if order == 0:
        return phi1 + phi2

    bracket_12 = glK_bracket_torch(phi1, phi2, generators)
    result = phi1 + phi2 + 0.5 * bracket_12

    if order >= 2:
        bracket_1_12 = glK_bracket_torch(phi1, bracket_12, generators)
        bracket_2_12 = glK_bracket_torch(phi2, bracket_12, generators)
        result = result + (1.0/12.0) * bracket_1_12 - (1.0/12.0) * bracket_2_12

    return result


def retract_glK_torch(
    phi: 'torch.Tensor',
    delta_phi: 'torch.Tensor',
    generators: 'torch.Tensor',
    step_size: float = 1.0,
    trust_region: float = 0.1,
    max_norm: float = 1.0,
    bch_order: int = 0,  # Simple addition -- BCH bracket can amplify noise in GL(K)
    eps: float = 1e-6,
    grad_clip: float = 10.0,
) -> 'torch.Tensor':
    """Retract phi update in GL(K) with trust region and gradient clipping.

    Uses conservative settings since GL(K) can produce ill-conditioned transport
    operators more easily than SO(K).

    Args:
        phi: Current gauge frames (..., n_gen) where n_gen = K^2
        delta_phi: Update direction (..., n_gen)
        generators: Lie algebra generators (n_gen, dim, dim)
        step_size: Learning rate
        trust_region: Max relative change
        max_norm: Max allowed norm
        bch_order: BCH order (0=add, 1=first correction)
        eps: Numerical stability constant
        grad_clip: Max gradient norm (per-element clipping)

    Returns:
        phi_new: Updated gauge frames (..., n_gen)
    """
    import torch

    delta_norm = torch.norm(delta_phi, dim=-1, keepdim=True)
    clip_scale = torch.clamp(grad_clip / (delta_norm + eps), max=1.0)
    delta_phi_clipped = clip_scale * delta_phi

    update = step_size * delta_phi_clipped

    phi_norm = torch.norm(phi, dim=-1, keepdim=True).clamp(min=0.1)
    update_norm = torch.norm(update, dim=-1, keepdim=True)

    scale = torch.clamp(trust_region * phi_norm / (update_norm + eps), max=1.0)
    update = scale * update

    # Absolute clipping on update magnitude
    update_norm_after = torch.norm(update, dim=-1, keepdim=True)
    max_update = 0.5
    update = torch.where(
        update_norm_after > max_update,
        update * max_update / (update_norm_after + eps),
        update
    )

    if bch_order == 0:
        phi_new = phi + update
    else:
        phi_new = glK_compose_bch_torch(phi, update, generators, order=bch_order)

    phi_new_norm = torch.norm(phi_new, dim=-1, keepdim=True)
    phi_new = torch.where(
        phi_new_norm > max_norm,
        phi_new * (max_norm / (phi_new_norm + eps)),
        phi_new
    )

    return phi_new


def is_glK_generators(n_gen: int) -> bool:
    """Check if n_gen corresponds to GL(K) (perfect square)."""
    import math
    K = int(math.sqrt(n_gen))
    return K * K == n_gen and K > 0


def is_soN_generators(n_gen: int) -> bool:
    """Check if n_gen corresponds to SO(N) (triangular number N(N-1)/2)."""
    import math
    discriminant = 1 + 8 * n_gen
    sqrt_disc = int(math.sqrt(discriminant))
    if sqrt_disc * sqrt_disc != discriminant:
        return False
    N = (1 + sqrt_disc) // 2
    return N * (N - 1) // 2 == n_gen and N >= 2


_GENERATOR_CACHE: Dict[int, np.ndarray] = {}
