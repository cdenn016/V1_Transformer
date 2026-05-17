"""
Core type definitions for gauge-theoretic VFE transformer.

Shared between transformer/core/ and transformer/vfe/.
"""

from typing import List, NamedTuple, Optional, Tuple
import torch


class BeliefState(NamedTuple):
    r"""Gaussian belief tuple at each token position.

    Represents :math:`q_i(z) = \mathcal{N}(z;\, \mu_i, \Sigma_i)` with gauge
    frame :math:`\phi_i \in \mathfrak{g}`, and optionally a group-level gauge
    state :math:`\Omega_i \in G` for the "omega-direct" parameterization.

    Attributes:
        mu: Belief means ``(B, N, K)``.
        sigma: Belief covariances — ``(B, N, K)`` diagonal or ``(B, N, K, K)`` full.
        phi: Gauge frame coordinates ``(B, N, n_gen)`` in the Lie algebra.
        omega: Optional per-block group-level gauge state. When present, it is
            a list of length ``len(irrep_dims)`` whose elements are
            ``(Omega_h, Omega_h_inv)`` pairs each of shape ``(B, N, d_h, d_h)``.
            Populated only when ``VFEConfig.gauge_parameterization ==
            'omega_direct'``; ``None`` otherwise (preserves back-compatibility
            with all existing callers that construct
            ``BeliefState(mu=, sigma=, phi=)``).
    """
    mu: torch.Tensor
    sigma: torch.Tensor
    phi: torch.Tensor
    omega: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
