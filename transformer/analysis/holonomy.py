"""
Holonomy Computation and Analysis for Non-Flat Gauge Transport.
================================================================

Computes the holonomy H_ijk = Ω_ij · Ω_jk · Ω_ki around closed loops
in the token graph. For the multiplicative-perturbation architecture:

    Ω_ij = exp(φ_i) · exp(δ_ij · G) · exp(-φ_j)

the holonomy factorizes as:

    H_ijk = exp(φ_i) · C_ijk · exp(-φ_i)

where C_ijk = exp(δ_ij·G) · exp(δ_jk·G) · exp(δ_ki·G) is the
"connection holonomy" -- the gauge-invariant (up to conjugation) part.

‖C_ijk - I‖_F measures the deviation from flatness for the triple (i,j,k).
When the transport is flat (cocycle condition holds), C_ijk = I for all triples.
"""

import torch
import numpy as np
from typing import Optional, Tuple, Dict, List


def compute_holonomy(
    exp_delta: torch.Tensor,
    triples: Optional[torch.Tensor] = None,
    sample_size: int = 1000,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute connection holonomy C_ijk = exp(δ_ij)·exp(δ_jk)·exp(δ_ki).

    We compute the connection holonomy (without gauge-frame conjugation) since:
    1. ‖C - I‖_F = 0 ⟺ ‖H - I‖_F = 0 (both vanish iff flat)
    2. C is the physically meaningful quantity (gauge-covariant)
    3. Avoids requiring exp(φ_i) which may not be stored

    Args:
        exp_delta: (B, N, N, K, K) -- exp(δ_ij · G) per edge.
            Output of the non-flat compute_transport_operators().
        triples: Optional (n_triples, 3) tensor of (i,j,k) indices.
            If None, samples random triples.
        sample_size: Number of random triples to sample if triples is None.
        seed: Random seed for reproducible triple sampling.

    Returns:
        C: (B, n_triples, K, K) -- connection holonomy matrices.
        norms: (B, n_triples) -- ‖C_ijk - I‖_F per triple.
        triple_indices: (n_triples, 3) -- the (i,j,k) indices used.
    """
    B, N, _, K, _ = exp_delta.shape
    device = exp_delta.device

    if triples is None:
        # Sample random triples (i, j, k) with distinct indices
        rng = np.random.RandomState(seed)
        triple_list = []
        attempts = 0
        while len(triple_list) < sample_size and attempts < sample_size * 10:
            ijk = rng.choice(N, size=3, replace=False)
            triple_list.append(ijk)
            attempts += 1
        triple_indices = torch.tensor(np.array(triple_list), device=device)
    else:
        triple_indices = triples.to(device)

    n_triples = triple_indices.shape[0]
    i = triple_indices[:, 0]
    j = triple_indices[:, 1]
    k = triple_indices[:, 2]

    # C_ijk = exp(δ_ij) · exp(δ_jk) · exp(δ_ki)
    # exp_delta[:, i, j] is (B, n_triples, K, K)
    C = torch.bmm(
        exp_delta[:, i, j].reshape(-1, K, K),
        exp_delta[:, j, k].reshape(-1, K, K),
    )
    C = torch.bmm(C, exp_delta[:, k, i].reshape(-1, K, K))
    C = C.reshape(B, n_triples, K, K)

    # Deviation from identity
    I_K = torch.eye(K, device=device, dtype=C.dtype).unsqueeze(0).unsqueeze(0)
    norms = torch.norm(C - I_K, p='fro', dim=(-2, -1))  # (B, n_triples)

    return C, norms, triple_indices


def holonomy_penalty_loss(
    exp_delta: torch.Tensor,
    sample_size: int = 500,
    seed: int = None,
) -> torch.Tensor:
    """Regularizer pushing the model toward flatness: E[‖C_ijk - I‖^2_F].

    Used for HF2.3 (holonomy penalty scaling) experiments.
    Adding λ_H · holonomy_penalty_loss to the total loss penalizes
    non-trivial holonomy, encouraging flat transport.

    Args:
        exp_delta: (B, N, N, K, K) -- exp(δ_ij · G) per edge.
        sample_size: Number of random triples to average over.
        seed: Random seed. If None, uses current step for variation.

    Returns:
        Scalar tensor: mean squared holonomy norm.
    """
    if seed is None:
        seed = int(torch.randint(0, 100000, (1,)).item())
    _, norms, _ = compute_holonomy(exp_delta, sample_size=sample_size, seed=seed)
    return (norms ** 2).mean()


def holonomy_statistics(
    exp_delta: torch.Tensor,
    sample_size: int = 2000,
    seed: int = 42,
) -> Dict[str, float]:
    """Compute summary statistics of holonomy across sampled triples.

    Args:
        exp_delta: (B, N, N, K, K) -- exp(δ_ij · G) per edge.
        sample_size: Number of random triples to sample.
        seed: Random seed for reproducibility.

    Returns:
        Dict with holonomy summary statistics.
    """
    _, norms, _ = compute_holonomy(exp_delta, sample_size=sample_size, seed=seed)
    norms_flat = norms.detach().cpu()

    return {
        'mean': norms_flat.mean().item(),
        'max': norms_flat.max().item(),
        'std': norms_flat.std().item(),
        'median': norms_flat.median().item(),
        'frac_gt_0.01': (norms_flat > 0.01).float().mean().item(),
        'frac_gt_0.1': (norms_flat > 0.1).float().mean().item(),
        'frac_gt_1.0': (norms_flat > 1.0).float().mean().item(),
    }


def holonomy_by_token_pairs(
    exp_delta: torch.Tensor,
    anchor_indices: List[int],
    sample_size_per_anchor: int = 100,
    seed: int = 42,
) -> Dict[int, Dict[str, float]]:
    """Compute holonomy statistics anchored at specific tokens.

    For each anchor token i, samples triples (i, j, k) and computes
    holonomy statistics. Useful for identifying which tokens participate
    in high-holonomy interactions (HF1.2, HF5.1).

    Args:
        exp_delta: (B, N, N, K, K) -- exp(δ_ij · G) per edge.
        anchor_indices: List of token indices to anchor triples at.
        sample_size_per_anchor: Number of triples per anchor.
        seed: Random seed.

    Returns:
        Dict mapping anchor_index -> holonomy statistics dict.
    """
    B, N, _, K, _ = exp_delta.shape
    rng = np.random.RandomState(seed)
    results = {}

    for anchor in anchor_indices:
        triples = []
        for _ in range(sample_size_per_anchor):
            jk = rng.choice([x for x in range(N) if x != anchor], size=2, replace=False)
            triples.append([anchor, jk[0], jk[1]])
        triples_tensor = torch.tensor(triples, device=exp_delta.device)
        _, norms, _ = compute_holonomy(exp_delta, triples=triples_tensor)
        norms_flat = norms.detach().cpu()
        results[anchor] = {
            'mean': norms_flat.mean().item(),
            'max': norms_flat.max().item(),
            'std': norms_flat.std().item(),
        }

    return results
