r"""Belief extraction for VFE semantic clustering.

Two views of per-token belief geometry :math:`(\mu, \Sigma, \phi)`:

``extract_contextual``
    Per-occurrence beliefs after the model's E-step has relaxed them in
    context. ``layer='final'`` reads the post-final-norm converged state via
    :meth:`VFEModel.forward_with_beliefs`; an integer ``layer`` reads that
    block's E-step snapshot (``_last_attention_state``).

``extract_vocab``
    One row per token *type*: the per-token Gaussian prior the model encodes,
    obtained from :meth:`VFEPriorBank.encode`. This dispatches both
    parameterizations (``gauge_fixed_priors`` True → gauge-orbit
    :math:`\mu_v = A_v \mu_0`; False → direct ``mu_embed``/``sigma_log_embed``
    lookup) so the extracted prior matches what the model actually uses.

Both views populate ``generators`` with the EXACT model generator bank
(``model.generators``, registered at ``prior_bank.py:132`` /
``model.py:82`` — the same tensor the bank exponentiates in
``_compute_block_exp_pairs``) so the :math:`\Omega` geodesic in
``geometry.py`` reproduces the model's own transport.

Law 1 (the E-step never sees targets) is preserved: neither function ever
passes ``targets`` into the model.
"""

from __future__ import annotations

import warnings
from typing import Optional, Union

import torch

from transformer.vfe.semantic_clustering.bundle import BeliefBundle


def _to_cpu(t: torch.Tensor) -> torch.Tensor:
    """Detach a tensor and move it to CPU (no-op gradient, host-resident)."""
    return t.detach().to("cpu")


def extract_contextual(
    model,
    token_ids: torch.Tensor,
    layer: Union[str, int] = "final",
) -> BeliefBundle:
    r"""Extract per-occurrence contextual beliefs from a VFE model.

    Never passes ``targets`` — Law 1 (the E-step is blind to targets) holds
    structurally.

    Args:
        model: A ``VFEModel``.
        token_ids: ``(B, N)`` input token ids.
        layer: ``'final'`` reads the post-final-norm converged beliefs via
            :meth:`VFEModel.forward_with_beliefs`. An ``int`` reads
            ``model.stack.blocks[layer].e_step._last_attention_state`` after a
            target-free ``model.forward`` pass.

    Returns:
        A :class:`BeliefBundle` with ``source='contextual'``, one row per
        ``(b, n)`` occurrence (flattened ``(B, N, *) -> (B * N, *)``).
    """
    cfg = model.cfg

    if cfg.gauge_parameterization == "omega_direct":
        warnings.warn(
            "gauge_parameterization='omega_direct': the per-layer E-step "
            "snapshot stores phi at its encode-time value (phi does not evolve "
            "when Omega evolves), so contextual phi is NOT the E-step-evolved "
            "gauge frame for this configuration.",
            stacklevel=2,
        )

    if layer == "final":
        with torch.no_grad():
            _, beliefs = model.forward_with_beliefs(token_ids)
        mu = beliefs.mu          # (B, N, K)
        sigma = beliefs.sigma    # (B, N, K) diagonal or (B, N, K, K) full
        phi = beliefs.phi        # (B, N, n_gen)
    else:
        layer_idx = int(layer)
        blocks = model.stack.blocks
        for block in blocks:
            block.e_step._capture_attention_state = True
        with torch.no_grad():
            model.forward(token_ids)  # no targets — Law 1
        state = blocks[layer_idx].e_step._last_attention_state
        mu = state["mu_q"]        # (B, N, K)
        sigma = state["sigma_q"]  # (B, N, K) — already diagonal-extracted
        phi = state["phi"]        # (B, N, n_gen)

    K = cfg.embed_dim
    mu = _to_cpu(mu).reshape(-1, K)
    if sigma.dim() == 4:
        sigma = _to_cpu(sigma).reshape(-1, K, K)
    else:
        sigma = _to_cpu(sigma).reshape(-1, K)
    n_gen = phi.shape[-1]
    phi = _to_cpu(phi).reshape(-1, n_gen)

    ids = _to_cpu(token_ids).reshape(-1)

    generators = _to_cpu(model.generators) if model.generators is not None else None

    return BeliefBundle(
        mu=mu,
        sigma=sigma,
        phi=phi,
        token_ids=ids,
        token_strings=None,
        generators=generators,
        irrep_dims=list(cfg.effective_block_dims),
        source="contextual",
        layer=layer,
        diagonal=bool(cfg.diagonal_covariance),
    )


def extract_vocab(
    model,
    token_ids: Optional[torch.Tensor] = None,
    max_tokens: int = 4096,
) -> BeliefBundle:
    r"""Extract one prior belief per token type from the encode bank.

    Reads the per-token Gaussian prior via :meth:`VFEPriorBank.encode`, which
    dispatches both parameterizations internally: ``gauge_fixed_priors=True``
    yields the gauge-orbit prior :math:`(\mu_v, \Sigma_v) = (A_v \mu_0, A_v
    \mathrm{diag}(\sigma_0) A_v^\top)`; ``gauge_fixed_priors=False`` yields the
    direct ``mu_embed`` / ``sigma_log_embed`` lookup. Using ``encode`` (rather
    than reading raw embedding weights) guarantees the extracted vocab prior is
    exactly what the model treats as the per-token prior, including
    determinant-control on :math:`\phi`.

    Never passes ``targets`` — Law 1 holds (encode reads token ids only).

    Args:
        model: A ``VFEModel``.
        token_ids: ``(n,)`` token ids to extract. When None, uses
            ``arange(min(vocab_size, max_tokens))``.
        max_tokens: Cap on the default id range when ``token_ids`` is None.

    Returns:
        A :class:`BeliefBundle` with ``source='vocab'``, one row per token type.
    """
    cfg = model.cfg
    pb = model.prior_bank

    if token_ids is None:
        n = min(cfg.vocab_size, max_tokens)
        token_ids = torch.arange(n, dtype=torch.long)

    device = next(model.parameters()).device
    ids = token_ids.to(device).reshape(-1)

    with torch.no_grad():
        # encode expects a (B, N) shape; add a batch dim then squeeze it.
        beliefs = pb.encode(ids.unsqueeze(0))

    K = cfg.embed_dim
    mu = _to_cpu(beliefs.mu).reshape(-1, K)
    if beliefs.sigma.dim() == 4:
        sigma = _to_cpu(beliefs.sigma).reshape(-1, K, K)
    else:
        sigma = _to_cpu(beliefs.sigma).reshape(-1, K)
    n_gen = beliefs.phi.shape[-1]
    phi = _to_cpu(beliefs.phi).reshape(-1, n_gen)

    generators = _to_cpu(model.generators) if model.generators is not None else None

    return BeliefBundle(
        mu=mu,
        sigma=sigma,
        phi=phi,
        token_ids=_to_cpu(ids),
        token_strings=None,
        generators=generators,
        irrep_dims=list(cfg.effective_block_dims),
        source="vocab",
        layer="final",
        diagonal=bool(cfg.diagonal_covariance),
    )
