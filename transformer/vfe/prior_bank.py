"""
VFEPriorBank: per-token Gaussian prior bank for encode and decode.

Two parameterizations selected by ``cfg.gauge_fixed_priors``:

    gauge_fixed_priors=True (default):
        pi_v = A_v . pi_0,  A_v = exp(psi_v . G)
        Single universal Gaussian (base_mu, base_log_sigma); per-token gauge
        frame phi_v rotates the orbit. Per-token capacity = V * n_gen.

    gauge_fixed_priors=False:
        mu_v, sigma_v looked up directly per token. The gauge frame phi_v is
        retained per token for cross-token transport (Omega_ij used in
        attention and stack transport), not for prior construction.
        Per-token capacity = V * (2K + n_gen).

Encode: token_ids -> BeliefState (mu, sigma, phi)
Decode: BeliefState -> logits = -c * KL(q* || pi_v) / tau

Law 3 (same Gaussian manifold for encode, infer, decode) holds in both modes.

Implementation note — decode rescales the canonical KL by a learnable scalar.
=============================================================================
The manuscript-canonical decode is `logits = -KL(q || pi_v) / tau`. This
module instead computes `logits = -c * KL(q || pi_v) / tau` with a
learnable scalar `c = exp(decode_log_scale)`, clamped to
`[exp(-3), exp(3)] ~ [0.05, 20]` and initialised at 0 so that `c = 1` at
construction (untrained models match the documented decode bitwise).

This is equivalent to a second softmax temperature stacked multiplicatively
on `tau`: the entangled product `c / tau` controls the logit scale. The
parameter is grouped with `m_sigma_lr` in the optimizer (see
`transformer/training/config.py`), not with the temperature group with
`kappa`. Document and tune accordingly.
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
    r"""Per-token Gaussian prior bank: encode tokens to beliefs, decode beliefs to logits.

    Selected at construction by ``cfg.gauge_fixed_priors``.

    Gauge-fixed mode (``gauge_fixed_priors=True``):
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

    Direct mode (``gauge_fixed_priors=False``):
        Learnable parameters:
            - ``mu_embed`` :math:`(V, K)` — per-token prior mean
            - ``sigma_log_embed`` :math:`(V, K)` — per-token log-variance (diagonal)
            - ``phi_embed`` :math:`(V, n_{\text{gen}})` — per-token Lie algebra coordinates

        The prior for token :math:`v` is :math:`\mu_v = \mu_{\text{embed}}[v]`,
        :math:`\Sigma_v = \text{diag}(\exp(\sigma_{\log}[v]))`. The gauge frame
        :math:`\phi_v` is retained for cross-token transport but does NOT enter
        prior construction. Full-covariance mode embeds the diagonal into a
        block-diagonal :math:`(K, K)` matrix.

    Args:
        cfg: VFEConfig with vocab_size, embed_dim, irrep_spec, gauge_fixed_priors.
        generators: ``(n_gen, K, K)`` Lie algebra generators.
    """

    def __init__(self, cfg: 'VFEConfig', generators: torch.Tensor) -> None:
        super().__init__()
        V = cfg.vocab_size
        K = cfg.embed_dim
        self.vocab_size = V
        self.embed_dim = K
        # Gauge-block partition for trace vectors, sl(K) projection, and per-
        # block sandwich-transform iteration. Under cross_couplings this is
        # the super-block layout: phi_project_slk then targets one trace per
        # super-block (the determinant of each super-block GL(d_super)) which
        # is the correct constraint for the merged gauge group.
        self.irrep_dims: List[int] = cfg.effective_block_dims
        self._original_irrep_dims: List[int] = cfg.irrep_dims
        self.diagonal_covariance = cfg.diagonal_covariance
        self.gauge_covariant_ridge = getattr(cfg, 'gauge_covariant_ridge', False)
        self.sigma_max = cfg.sigma_max
        # Numerical floor for σ. The base-σ helper at `:base_sigma` uses
        # 0.01 — the historical floor for gauge-fixed mode — and the
        # direct-mode `_per_token_priors` uses the same constant so the
        # two parameterizations span the same reachable Gaussian manifold.
        # The smaller `1e-6` used elsewhere is retained as `self.eps_small`
        # for places that previously consumed `self.eps`.
        self.eps = 0.01
        self.eps_small = 1e-6
        self.phi_project_slk = cfg.phi_project_slk
        self.phi_trace_clamp = cfg.phi_trace_clamp

        # Normalize generators to torch tensor BEFORE consuming them for any
        # buffer setup (skew detection, per-block trace vectors).
        if not isinstance(generators, torch.Tensor):
            generators = torch.from_numpy(generators).float()
        n_gen = generators.shape[0]
        self.register_buffer('generators', generators)

        # Detect skew-symmetric generators (SO(N)) — check ALL generators
        # (previously only generators[0], which mis-classifies mixed sets and
        # silently selects an SO(N)-only fast exp path for non-skew banks).
        with torch.no_grad():
            self._generators_are_skew = bool(
                torch.allclose(generators, -generators.transpose(-1, -2), atol=1e-6)
            )

        # Per-block determinant-control trace vectors. For a block-diagonal
        # gauge group GL(K_1) ⊕ ... ⊕ GL(K_H), each block carries its own
        # unbounded trace direction; killing one summed direction is NOT
        # sufficient (compensating signs across blocks defeat the constraint).
        # Build V[h, a] = tr(G_a restricted to block h) — H independent
        # constraints, mutually orthogonal because block-diagonal generators
        # have disjoint generator-index supports per block.
        # Register det-control buffers in BOTH branches so state_dict
        # round-trips have consistent keys and `.to(device)` always
        # moves them. The else-branch registers zero-numel sentinels;
        # `_apply_phi_det_control` predicates on numel() == 0.
        if self.phi_project_slk or (
            self.phi_trace_clamp is not None and self.phi_trace_clamp > 0
        ):
            H = len(self.irrep_dims)
            V_blocks = torch.zeros(H, n_gen, dtype=generators.dtype, device=generators.device)
            start = 0
            for h, d_h in enumerate(self.irrep_dims):
                end = start + d_h
                V_blocks[h] = generators[:, start:end, start:end].diagonal(
                    dim1=-2, dim2=-1
                ).sum(dim=-1)
                start = end
            self.register_buffer('_phi_trace_vec', V_blocks)         # (H, n_gen)
            self.register_buffer(
                '_phi_trace_vec_norm_sq',
                (V_blocks * V_blocks).sum(dim=-1).clamp(min=1e-12),  # (H,)
            )
        else:
            self.register_buffer(
                '_phi_trace_vec',
                torch.empty(0, dtype=generators.dtype),
            )
            self.register_buffer(
                '_phi_trace_vec_norm_sq',
                torch.empty(0, dtype=generators.dtype),
            )

        # Prior parameterization: gauge-fixed (shared base + per-token gauge
        # orbit) vs direct (per-token (mu_v, sigma_v) lookup). Both modes keep
        # phi_embed as the per-token gauge frame used for cross-token transport
        # (Omega_ij in attention and stack handoff). Only the prior *value*
        # construction differs.
        # mu_init_std=0 is honored literally (degenerate prior at zero) — the
        # previous silent rescale to sqrt(log(V)/K) corrupted sweep results
        # because mu_init_std=0 in the ablation suite then mapped to ~0.74.
        self.gauge_fixed_priors = getattr(cfg, 'gauge_fixed_priors', True)
        sigma_init_log = math.log(cfg.sigma_init) if cfg.sigma_init > 0 else 0.0
        if self.gauge_fixed_priors:
            # Shared base prior (mu_0, sigma_0) lifted to per-token (mu_v, sigma_v)
            # via A_v = exp(phi_v . G).
            self.base_mu = nn.Parameter(torch.randn(K) * cfg.mu_init_std)
            self.base_log_sigma = nn.Parameter(torch.full((K,), sigma_init_log))
        else:
            # Per-token Gaussian prior directly. ~V*K + V*K = 2VK learnable
            # scalars on top of phi_embed's V*n_gen.
            self.mu_embed = nn.Embedding(V, K)
            nn.init.normal_(self.mu_embed.weight, mean=0.0, std=cfg.mu_init_std)
            self.sigma_log_embed = nn.Embedding(V, K)
            with torch.no_grad():
                self.sigma_log_embed.weight.fill_(sigma_init_log)

        # Per-token gauge frames in Lie algebra (used in both modes — drives
        # prior in gauge-fixed mode, drives transport-only in direct mode).
        self.phi_embed = nn.Embedding(V, n_gen)
        nn.init.normal_(self.phi_embed.weight, mean=0.0, std=cfg.phi_scale)

        # Learnable decode temperature. The canonical Gaussian-cluster
        # decode is `logits = -KL(q || pi_v) / tau`. This module instead
        # computes `logits = -c * KL(q || pi_v) / tau` with a learnable
        # `c = exp(decode_log_scale)` clamped to `[exp(-3), exp(3)] ~ [0.05, 20]`.
        # Equivalent to a learnable scalar softmax temperature on top of
        # the canonical form. Initialised at 0 (c=1), so untrained models
        # match the documented decode; the parameter drifts during training.
        self.decode_log_scale = nn.Parameter(torch.zeros(1))

        # Decode cache (invalidated each forward pass)
        self._decode_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        # Pre-built arange(V) so decode() doesn't rebuild the index tensor
        # every forward when the decode cache is cold (which is every step,
        # because model.py invalidates the cache at the top of forward).
        self.register_buffer('_all_token_ids', torch.arange(V, dtype=torch.long))

    @property
    def base_sigma(self) -> torch.Tensor:
        """Base prior variance (always positive), shape ``(K,)``.

        Only meaningful when ``gauge_fixed_priors=True``. Raises ``AttributeError``
        under direct-mode parameterization (use ``sigma_log_embed`` instead).
        """
        p = self.base_log_sigma
        return torch.exp(p.float() if p.dtype != torch.float32 else p).clamp(0.01, self.sigma_max)

    def _per_token_priors(
        self, token_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Direct-mode prior lookup. Returns (mu_v, sigma_v_diag).

        Used when ``gauge_fixed_priors=False``. mu_v is shape ``(..., K)``,
        sigma_v is diagonal shape ``(..., K)``. Caller is responsible for
        lifting sigma_v to full ``(..., K, K)`` if ``diagonal_covariance=False``.
        """
        mu = self.mu_embed(token_ids)                               # (..., K)
        log_sigma = self.sigma_log_embed(token_ids)                 # (..., K)
        # Compute σ in float32 for numerical stability of `exp` + `clamp`,
        # then cast back to μ's dtype so the (μ, σ) pair matches downstream.
        sigma_diag = (
            torch.exp(log_sigma.float()).clamp(self.eps, self.sigma_max).to(mu.dtype)
        )
        return mu, sigma_diag

    def _diag_to_full_block(self, sigma_diag: torch.Tensor) -> torch.Tensor:
        r"""Embed a diagonal sigma ``(..., K)`` into full block-diagonal ``(..., K, K)``.

        Used in direct mode when ``diagonal_covariance=False`` to match the
        full-cov tensor shape downstream consumers expect. Because the prior
        is diagonal in direct mode, the off-diagonal blocks are zero.
        """
        K = self.embed_dim
        batch_shape = sigma_diag.shape[:-1]
        sigma_full = torch.zeros(*batch_shape, K, K, device=sigma_diag.device, dtype=sigma_diag.dtype)
        diag_idx = torch.arange(K, device=sigma_diag.device)
        sigma_full[..., diag_idx, diag_idx] = sigma_diag
        return sigma_full

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
            # Ensure SPD. Default path clamps the diagonal (gauge-breaking under
            # non-diagonal Ω). When gauge_covariant_ridge is on, add the
            # block-diagonal sandwich ε · block_diag(g_h g_h^T) instead — this
            # is an SPD additive floor that transforms as h·(·)·h^T under a
            # block-diagonal gauge.
            if self.gauge_covariant_ridge:
                _block_start = 0
                for _h, _d_h in enumerate(self.irrep_dims):
                    _block_end = _block_start + _d_h
                    _exp_h = block_exp_pairs[_h][0].to(sigma_p.dtype)
                    _M_h = _exp_h @ _exp_h.transpose(-1, -2)
                    sigma_p[..., _block_start:_block_end, _block_start:_block_end] = (
                        sigma_p[..., _block_start:_block_end, _block_start:_block_end]
                        + self.eps_small * _M_h
                    )
                    _block_start = _block_end
            else:
                diag_idx = torch.arange(K, device=sigma_p.device)
                sigma_p[..., diag_idx, diag_idx] = sigma_p[..., diag_idx, diag_idx].clamp(min=self.eps)
        else:
            sigma_p = torch.cat(sigma_parts, dim=-1)

        return mu_p, sigma_p

    def _apply_phi_det_control(self, phi: torch.Tensor) -> torch.Tensor:
        r"""Project or clamp the per-block determinant direction(s) of φ.

        For block-diagonal gauge groups :math:`GL(K_1) \oplus \cdots \oplus GL(K_H)`,
        each block has an independent trace functional
        :math:`s_h = \operatorname{tr}(\phi \cdot G^{(h)})` and an independent
        determinant :math:`\det(\Omega_h) = \exp(s_h^{(i)} - s_h^{(j)})`. Killing
        only the summed trace lets compensating signs across blocks blow up
        per-block determinants.

        Per-block constraint: enforce :math:`s_h \equiv 0` for every h
        (project_slk) or :math:`|s_h| \le T` for every h (trace_clamp). Because
        block-diagonal generators have disjoint per-block trace supports, the
        per-block trace vectors are mutually orthogonal — projection collapses
        to a single ``phi - V_blocks.T @ (V_blocks @ phi / ||v_h||²)`` step.

        Returns φ unchanged when neither toggle is enabled or when generators
        are skew-symmetric (SO(N)) so all v_h ≡ 0.
        """
        if self._phi_trace_vec.numel() == 0:
            return phi
        V = self._phi_trace_vec            # (H, n_gen)
        v_norm_sq = self._phi_trace_vec_norm_sq  # (H,)
        # s = phi @ V.T → (..., H) per-block scalar trace
        s = phi @ V.transpose(-2, -1)
        if self.phi_project_slk:
            coeffs = s / v_norm_sq         # (..., H)
            return phi - torch.einsum('...h,hg->...g', coeffs, V)
        # Soft per-block cap on |s_h|
        T = float(self.phi_trace_clamp)
        s_clamped = s.clamp(min=-T, max=T)
        delta_coeffs = (s_clamped - s) / v_norm_sq   # (..., H)
        return phi + torch.einsum('...h,hg->...g', delta_coeffs, V)

    def encode(self, token_ids: torch.Tensor) -> BeliefState:
        r"""Encode tokens to initial Gaussian beliefs.

        Gauge-fixed mode: :math:`(\mu_v, \Sigma_v) = (A_v \mu_0, A_v \text{diag}(\sigma_0) A_v^\top)`.
        Direct mode: :math:`\mu_v, \sigma_v` looked up from per-token embeddings;
        :math:`\phi_v` is returned for transport but does not enter the prior.

        Args:
            token_ids: ``(B, N)`` input token IDs.

        Returns:
            BeliefState with mu ``(B, N, K)``, sigma ``(B, N, K)`` or ``(B, N, K, K)``,
            phi ``(B, N, n_gen)``.
        """
        phi = self.phi_embed(token_ids)  # (B, N, n_gen)
        phi = self._apply_phi_det_control(phi)
        if self.gauge_fixed_priors:
            block_exp_pairs = self._compute_block_exp_pairs(phi)
            mu_p, sigma_p = self._apply_gauge_transform(block_exp_pairs)
        else:
            mu_p, sigma_diag = self._per_token_priors(token_ids)
            if self.diagonal_covariance:
                sigma_p = sigma_diag
            else:
                sigma_p = self._diag_to_full_block(sigma_diag)
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
        # since decode is diagonal KL for efficiency). In direct mode, the
        # all-V materialization is just a parameter read; phi is not used.
        if self._decode_cache is not None:
            mu_p, sigma_p = self._decode_cache
        elif self.gauge_fixed_priors:
            phi_all = self.phi_embed(self._all_token_ids)  # (V, n_gen)
            phi_all = self._apply_phi_det_control(phi_all)
            block_exp_pairs = self._compute_block_exp_pairs(phi_all, only_forward=True)
            mu_p, sigma_p = self._apply_gauge_transform(block_exp_pairs)
            self._decode_cache = (mu_p, sigma_p)
        else:
            mu_p, sigma_p = self._per_token_priors(self._all_token_ids)  # (V,K), (V,K)
            self._decode_cache = (mu_p, sigma_p)

        # Extract diagonal for KL computation (decode always uses diagonal KL
        # for O(V*K) efficiency — full-cov KL over V tokens is O(V*K^3))
        sigma_q_diag = torch.diagonal(sigma_q, dim1=-2, dim2=-1) if is_full_cov else sigma_q
        sigma_p_diag = torch.diagonal(sigma_p, dim1=-2, dim2=-1) if sigma_p.dim() >= 3 and sigma_p.shape[-1] == sigma_p.shape[-2] else sigma_p

        # KL uses division and log — promote to float32 explicitly. The
        # previous torch.amp.autocast('cuda', enabled=False) guard was both
        # redundant (the .float() cast below already overrides AMP) and
        # fragile on CPU/MPS where the 'cuda' device_type string is wrong.
        mu_q = mu_q.float()
        mu_p = mu_p.float()
        # Use the same σ floor as encode (`self.eps`). Previously decode
        # floored at 1e-4 — 100× below encode/`_per_token_priors` floor —
        # so when the E-step drove σ_q below 0.01 the decode KL was computed
        # on (q, p) pairs that the encode path would never have produced.
        sigma_q_d = sigma_q_diag.float().clamp(min=self.eps)
        sigma_p_d = sigma_p_diag.float().clamp(min=self.eps)

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
