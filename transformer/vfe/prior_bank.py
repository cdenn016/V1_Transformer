"""
VFEPriorBank: gauge-fixed orbit parameterization for encode and decode.

All V token priors are gauge transforms of a single base prior:
    pi_v = A_v . pi_0,  A_v = exp(psi_v . G)

Encode: token_ids -> BeliefState (mu, sigma, phi)
Decode: BeliefState -> logits = -KL(q* || pi_v) / tau

Law 3 enforced: same Gaussian manifold for encode, infer, and decode.
"""

from __future__ import annotations

import math
from typing import Optional, List, Tuple, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from transformer.vfe.config import VFEConfig
import torch.nn as nn
import torch.nn.functional as F

from transformer.core.types import BeliefState
from transformer.core.gauge_utils import (
    stable_matrix_exp_pair,
    fused_block_matrix_exp_pairs,
)


class VFEPriorBank(nn.Module):
    r"""Gauge-fixed prior bank: encode tokens to Gaussian beliefs, decode beliefs to logits.

    Learnable parameters:
        - ``base_mu`` :math:`(K,)` — shared base prior mean
        - ``base_log_sigma`` :math:`(K,)` — shared base prior log-variance
        - ``phi_embed`` :math:`(V, n_{\text{gen}})` — per-token Lie algebra coordinates

    The prior for token :math:`v` is:

    .. math::
        \pi_v = A_v \triangleright \pi_0, \quad
        A_v = \exp(\psi_v \cdot G), \quad
        \mu_v = A_v \mu_0, \quad
        \Sigma_v = A_v \,\text{diag}(\sigma_0)\, A_v^\top

    Args:
        cfg: VFEConfig with vocab_size, embed_dim, irrep_spec, etc.
        generators: ``(n_gen, K, K)`` Lie algebra generators.
    """

    def __init__(self, cfg: 'VFEConfig', generators: torch.Tensor) -> None:
        super().__init__()
        V = cfg.vocab_size
        K = cfg.embed_dim
        self.vocab_size = V
        self.embed_dim = K
        self.irrep_dims: List[int] = cfg.irrep_dims
        self.diagonal_covariance = cfg.diagonal_covariance
        self.sigma_max = cfg.sigma_max
        self.eps = 1e-6

        n_gen = generators.shape[0]
        if not isinstance(generators, torch.Tensor):
            generators = torch.from_numpy(generators).float()
        self.register_buffer('generators', generators)

        # Detect skew-symmetric generators (SO(N)) for optimized exp
        with torch.no_grad():
            _G0 = generators[0]
            self._generators_are_skew = bool(
                torch.allclose(_G0, -_G0.T, atol=1e-6)
            )

        # Base prior: shared across all tokens (gauge-orbit parameterization)
        mu_std = cfg.mu_init_std if cfg.mu_init_std > 0 else math.sqrt(math.log(max(V, 2)) / K)
        self.base_mu = nn.Parameter(torch.randn(K) * mu_std)
        sigma_init_log = math.log(cfg.sigma_init) if cfg.sigma_init > 0 else 0.0
        self.base_log_sigma = nn.Parameter(torch.full((K,), sigma_init_log))

        # Per-token gauge frames in Lie algebra
        self.phi_embed = nn.Embedding(V, n_gen)
        nn.init.normal_(self.phi_embed.weight, mean=0.0, std=cfg.phi_scale)

        # Learnable decode temperature
        self.decode_log_scale = nn.Parameter(torch.zeros(1))

        # Decode cache (invalidated each forward pass)
        self._decode_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    @property
    def base_sigma(self) -> torch.Tensor:
        """Base prior variance (always positive), shape ``(K,)``."""
        with torch.amp.autocast('cuda', enabled=False):
            p = self.base_log_sigma
            return torch.exp(p.float() if p.dtype != torch.float32 else p).clamp(0.01, self.sigma_max)

    def invalidate_cache(self) -> None:
        """Call at the start of each forward pass to clear decode cache."""
        self._decode_cache = None

    def _compute_block_exp_pairs(
        self,
        phi: torch.Tensor,
        only_forward: bool = False,
    ) -> List[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """Compute block-diagonal matrix exponentials from phi."""
        _phi_3d = phi if phi.dim() == 3 else phi.unsqueeze(0)
        pairs = fused_block_matrix_exp_pairs(
            _phi_3d, self.generators, self.irrep_dims,
            skew_symmetric=self._generators_are_skew,
            only_forward=only_forward,
        )
        if phi.dim() == 2:
            pairs = [
                (p[0].squeeze(0), p[1].squeeze(0) if p[1] is not None else None)
                for p in pairs
            ]
        return pairs

    def _apply_gauge_transform(
        self,
        block_exp_pairs: List[Tuple[torch.Tensor, Optional[torch.Tensor]]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Apply gauge transform to base prior: :math:`\mu_v = A_v \mu_0`, :math:`\Sigma_v = A_v \Sigma_0 A_v^\top`.

        When ``diagonal_covariance=True``, uses the diagonal approximation:
        :math:`\text{diag}(A \text{diag}(s) A^\top)`.

        When ``diagonal_covariance=False``, computes the exact sandwich product
        per block: :math:`A_h \text{diag}(s_h) A_h^\top`, assembling a
        block-diagonal :math:`(K, K)` matrix.

        Returns:
            mu_p: ``(..., K)`` prior means
            sigma_p: ``(..., K)`` diagonal or ``(..., K, K)`` full covariance
        """
        base_mu = self.base_mu
        base_sigma = self.base_sigma  # (K,)
        full_cov = not self.diagonal_covariance

        mu_parts = []
        sigma_parts = []
        block_start = 0

        for h, d_h in enumerate(self.irrep_dims):
            block_end = block_start + d_h
            exp_h = block_exp_pairs[h][0]  # (..., d_h, d_h)
            base_mu_h = base_mu[block_start:block_end]

            # mu_v = A_v @ mu_0 (per block)
            mu_parts.append(torch.einsum('...ij,j->...i', exp_h, base_mu_h))

            with torch.amp.autocast('cuda', enabled=False):
                base_sigma_h = base_sigma[block_start:block_end]
                exp_h_f32 = exp_h.float()

                if full_cov:
                    # Exact: Sigma_h = A_h @ diag(s_h) @ A_h^T  (sandwich product)
                    sigma_diag = torch.diag(base_sigma_h)  # (d_h, d_h)
                    sigma_h = exp_h_f32 @ sigma_diag @ exp_h_f32.transpose(-2, -1)
                    sigma_parts.append(sigma_h)
                else:
                    # Diagonal approx: diag(A diag(s) A^T) = sum_j A_{ij}^2 * s_j
                    sigma_parts.append(
                        (exp_h_f32 ** 2 @ base_sigma_h).clamp(min=self.eps)
                    )
            block_start = block_end

        mu_p = torch.cat(mu_parts, dim=-1)

        if full_cov:
            # Assemble block-diagonal (K, K)
            batch_shape = mu_p.shape[:-1]
            K = self.embed_dim
            sigma_p = torch.zeros(*batch_shape, K, K, device=mu_p.device, dtype=mu_p.dtype)
            block_start = 0
            for h, d_h in enumerate(self.irrep_dims):
                block_end = block_start + d_h
                sigma_p[..., block_start:block_end, block_start:block_end] = sigma_parts[h]
                block_start = block_end
            # Ensure SPD: clamp diagonal
            diag_idx = torch.arange(K, device=sigma_p.device)
            sigma_p[..., diag_idx, diag_idx] = sigma_p[..., diag_idx, diag_idx].clamp(min=self.eps)
        else:
            sigma_p = torch.cat(sigma_parts, dim=-1)

        return mu_p, sigma_p

    def encode(self, token_ids: torch.Tensor) -> BeliefState:
        r"""Encode tokens to initial Gaussian beliefs.

        Maps token IDs to :math:`(\mu_v, \Sigma_v, \phi_v)` via the gauge-fixed
        orbit parameterization.

        Args:
            token_ids: ``(B, N)`` input token IDs.

        Returns:
            BeliefState with mu ``(B, N, K)``, sigma ``(B, N, K)`` or ``(B, N, K, K)``,
            phi ``(B, N, n_gen)``.
        """
        phi = self.phi_embed(token_ids)  # (B, N, n_gen)
        block_exp_pairs = self._compute_block_exp_pairs(phi)
        mu_p, sigma_p = self._apply_gauge_transform(block_exp_pairs)
        return BeliefState(mu=mu_p, sigma=sigma_p, phi=phi)

    def decode(
        self,
        mu_q: torch.Tensor,
        sigma_q: torch.Tensor,
        tau: float = 1.0,
    ) -> torch.Tensor:
        r"""Compute logits via KL-to-prior: :math:`\ell_{i,v} = -\mathrm{KL}(q_i^\star \| \pi_v) / \tau`.

        When beliefs are full-covariance ``(B, N, K, K)``, the decode uses
        diagonal projection for O(V·K) efficiency. This is a documented
        approximation at the decode boundary — encode and infer operate on
        the full Gaussian manifold, but decode projects to diagonal KL.

        Uses the fused single-matmul diagonal KL implementation:

        .. math::
            \text{KL}(q \| \pi_v) = \tfrac{1}{2}\bigl[
                \text{tr}(\Sigma_q / \Sigma_p) + (\mu_q - \mu_p)^\top \Sigma_p^{-1} (\mu_q - \mu_p)
                - K + \log|\Sigma_p|/|\Sigma_q|
            \bigr]

        Terms constant across V (``-K``, ``log|Sigma_q|``) are dropped (cancel in softmax).

        Args:
            mu_q: ``(B, N, K)`` final belief means.
            sigma_q: ``(B, N, K)`` diagonal variances (or ``(B, N, K, K)`` — diagonal extracted).
            tau: Decode temperature.

        Returns:
            logits: ``(B, N, V)`` unnormalized log-probabilities.
        """
        V = self.vocab_size
        is_full_cov = sigma_q.dim() == 4

        # Materialize all V priors (uses diagonal even for full-cov model,
        # since decode is diagonal KL for efficiency)
        if self._decode_cache is not None:
            mu_p, sigma_p = self._decode_cache
        else:
            all_ids = torch.arange(V, device=mu_q.device)
            phi_all = self.phi_embed(all_ids)  # (V, n_gen)
            block_exp_pairs = self._compute_block_exp_pairs(phi_all, only_forward=True)
            mu_p, sigma_p = self._apply_gauge_transform(block_exp_pairs)
            self._decode_cache = (mu_p, sigma_p)

        # Extract diagonal for KL computation (decode always uses diagonal KL
        # for O(V*K) efficiency — full-cov KL over V tokens is O(V*K^3))
        sigma_q_diag = torch.diagonal(sigma_q, dim1=-2, dim2=-1) if is_full_cov else sigma_q
        sigma_p_diag = torch.diagonal(sigma_p, dim1=-2, dim2=-1) if sigma_p.dim() >= 3 and sigma_p.shape[-1] == sigma_p.shape[-2] else sigma_p

        # AMP guard: KL uses division and log
        with torch.amp.autocast('cuda', enabled=False):
            mu_q = mu_q.float()
            mu_p = mu_p.float()
            sigma_q_d = sigma_q_diag.float().clamp(min=1e-4)
            sigma_p_d = sigma_p_diag.float().clamp(min=1e-4)

            inv_sigma_p = 1.0 / sigma_p_d                    # (V, K)
            mu_p_inv_sigma_p = mu_p * inv_sigma_p             # (V, K)

            # Fused matmul: trace + quad_q - cross in ONE operation
            lhs = torch.cat([sigma_q_d + mu_q ** 2, -2.0 * mu_q], dim=-1)  # (B, N, 2K)
            rhs = torch.cat([inv_sigma_p, mu_p_inv_sigma_p], dim=-1)        # (V, 2K)
            combined = torch.matmul(lhs, rhs.T)               # (B, N, V)

            # Prior-side bias: quad_p + log_det_p
            prior_bias = (mu_p ** 2 * inv_sigma_p).sum(-1) + torch.log(sigma_p_d).sum(-1)  # (V,)

        scale = torch.exp(self.decode_log_scale.clamp(-3.0, 3.0))
        logits = -0.5 * scale / tau * (combined + prior_bias.unsqueeze(0).unsqueeze(0))

        return logits
