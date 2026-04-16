"""
Core type definitions for gauge-theoretic VFE transformer.

Shared between transformer/core/ and transformer/vfe/.
"""

from typing import NamedTuple, Optional
import torch


class BeliefState(NamedTuple):
    r"""Gaussian belief tuple at each token position.

    Represents :math:`q_i(z) = \mathcal{N}(z;\, \mu_i, \Sigma_i)` with gauge
    frame :math:`\phi_i \in \mathfrak{g}`.

    Attributes:
        mu: Belief means ``(B, N, K)``.
        sigma: Belief covariances — ``(B, N, K)`` diagonal or ``(B, N, K, K)`` full.
        phi: Gauge frame coordinates ``(B, N, n_gen)`` in the Lie algebra.
    """
    mu: torch.Tensor
    sigma: torch.Tensor
    phi: torch.Tensor
