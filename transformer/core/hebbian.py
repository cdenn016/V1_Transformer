"""
Hebbian / P-flow / Delta-Rule Learning
=======================================

Extracted from ``model.py``, ``embeddings.py``, ``prior_bank.py``, and
``training/experiment_runner.py`` to consolidate the "side-quest"
backprop-free learning machinery in a single focused module.

Activated when ``use_p_flow=True`` and/or ``use_delta_rule_w_out=True``
in the trainer config.

What it does
------------
- **P-flow** (Predictive Free-Energy flow): EMA-update token embeddings
  toward beliefs that predicted successfully (low CE).  Each token's
  prior moves toward the prediction-error-weighted average of beliefs
  observed at positions where that token appeared in the batch.
- **Phi P-flow**: same EMA update applied to gauge frame embeddings
  ``phi_embed`` when ``detach_phi=True``.  Provides a local learning rule
  for gauge frames without backprop.
- **Delta rule** (Widrow–Hoff): backprop-free Hebbian update of the output
  projection ``W_out``: ``ΔW = lr · (target − prediction) ⊗ μ^T``.

Combined with ``detach_phi=True`` these three mechanisms allow fully
backprop-free training: P-flow updates the embeddings, phi P-flow updates
the gauge frames, and the delta rule updates the output projection.

Public API
----------
- :func:`p_flow_update_model` — model-level dispatcher (PriorBank vs
  GaugeTokenEmbedding) for the embedding-mean P-flow update.
- :func:`phi_flow_update_model` — model-level dispatcher for the gauge-frame
  P-flow update.
- :func:`delta_rule_update_w_out_model` — model-level Widrow-Hoff update of
  the output projection.
- :func:`apply_p_flow_and_delta_rule` — trainer-level orchestration that
  reads the config and calls the three above in sequence.

Backward compatibility
----------------------
The original methods on ``GaugeTransformerLM``, ``GaugeTokenEmbedding``,
``PriorBank``, and ``PublicationTrainer`` remain as thin wrappers that
delegate to the free functions in this module.  External code that calls
``model.p_flow_update(...)`` continues to work unchanged.
"""

from typing import Optional, Dict

import torch
import torch.nn.functional as F


__all__ = [
    # Trainer-level orchestration
    "apply_p_flow_and_delta_rule",
    # Model-level dispatchers (replacements for GaugeTransformerLM methods)
    "p_flow_update_model",
    "phi_flow_update_model",
    "delta_rule_update_w_out_model",
    # GaugeTokenEmbedding helpers
    "compute_pflow_weights",
    "update_embeddings_from_beliefs",
    "update_phi_from_beliefs",
    # PriorBank helpers
    "update_prior_bank_from_beliefs",
    "update_gauge_fixed_base_prior",
]


# =============================================================================
# Low-level helpers for prediction-error-weighted segment softmax
# =============================================================================

def compute_pflow_weights(
    token_ids: torch.Tensor,           # (B, N)
    prediction_errors: torch.Tensor,   # (B, N)
    pad_token_id: int = -100,          # Default matches CE loss ignore_index
) -> tuple:
    r"""Compute per-token-type prediction-error-weighted averages.

    Returns ``(unique_tokens, inverse_idx, weights, n_unique)`` where
    ``weights`` are segment-wise ``softmax(-error)`` per token type.  Uses
    scatter ops for O(B*N) vectorized computation.

    Padding positions (where ``token_ids == pad_token_id``) are masked
    out by zeroing their softmax contributions.
    """
    flat_ids = token_ids.reshape(-1)            # (B*N,)
    flat_errors = prediction_errors.reshape(-1)  # (B*N,)

    # Mask padding
    valid = (flat_ids != pad_token_id)
    neg_errors = (-flat_errors.clamp(min=-10, max=10))
    # Set padding to very negative so it gets near-zero softmax weight
    neg_errors = neg_errors.masked_fill(~valid, -20.0)

    unique_tokens, inverse_idx = torch.unique(flat_ids, return_inverse=True)
    n_unique = unique_tokens.shape[0]

    # Segment-wise softmax: max per token type for numerical stability
    seg_max = torch.full((n_unique,), float('-inf'), device=flat_ids.device, dtype=neg_errors.dtype)
    seg_max.scatter_reduce_(0, inverse_idx, neg_errors, reduce='amax', include_self=False)
    shifted = neg_errors - seg_max[inverse_idx]
    exp_shifted = torch.exp(shifted)
    exp_shifted = exp_shifted * valid.float()  # Zero padding contributions

    # Normalize per token type
    seg_sum = torch.zeros(n_unique, device=flat_ids.device, dtype=neg_errors.dtype)
    seg_sum.scatter_add_(0, inverse_idx, exp_shifted)
    weights = exp_shifted / seg_sum[inverse_idx].clamp(min=1e-12)

    return unique_tokens, inverse_idx, weights, n_unique


# =============================================================================
# GaugeTokenEmbedding helpers
# =============================================================================

def update_embeddings_from_beliefs(
    embed,                              # GaugeTokenEmbedding instance
    token_ids: torch.Tensor,            # (B, N) token IDs in this batch
    mu_beliefs: torch.Tensor,           # (B, N, K) final beliefs after VFE
    prediction_errors: torch.Tensor,    # (B, N) per-position CE loss
    ema_decay: float = 0.99,            # EMA decay rate (higher = slower update)
    sigma_beliefs: Optional[torch.Tensor] = None,
    pad_token_id: int = -100,
):
    r"""P-flow: Update token embeddings toward successful beliefs via EMA.

    Uses vectorized scatter ops with segment-wise softmax per token type.
    Updates both mu and sigma embeddings.

    Formula:
        \mu_{\text{token}} ← (1 − η) μ_{\text{token}} + η \bar{μ}_{\text{belief}}

    where \bar{μ} is the prediction-error-weighted mean across all
    occurrences of this token in the batch.
    """
    if embed.gauge_fixed_priors:
        return

    B, N, K = mu_beliefs.shape
    lr = 1.0 - ema_decay

    with torch.no_grad():
        unique_tokens, inverse_idx, weights, n_unique = compute_pflow_weights(
            token_ids, prediction_errors, pad_token_id,
        )

        flat_mu = mu_beliefs.reshape(-1, K)  # (B*N, K)

        # Weighted mean mu per token type via scatter_add
        weighted_mu = torch.zeros(n_unique, K, device=flat_mu.device, dtype=flat_mu.dtype)
        weighted_mu.scatter_add_(
            0, inverse_idx.unsqueeze(-1).expand(-1, K),
            flat_mu * weights.unsqueeze(-1),
        )

        # Confidence-weighted LR: tokens with lower mean error get higher LR
        flat_errors = prediction_errors.reshape(-1)
        seg_error_sum = torch.zeros(n_unique, device=flat_mu.device, dtype=flat_mu.dtype)
        seg_error_sum.scatter_add_(0, inverse_idx, flat_errors.clamp(min=0))
        seg_count = torch.zeros(n_unique, device=flat_mu.device, dtype=flat_mu.dtype)
        seg_count.scatter_add_(0, inverse_idx, torch.ones_like(flat_errors))
        mean_errors = seg_error_sum / seg_count.clamp(min=1)
        confidence = 1.0 / (1.0 + mean_errors)
        effective_lr = lr * confidence  # (n_unique,)

        # Skip padding token if present
        pad_mask = (unique_tokens != pad_token_id)
        update_tokens = unique_tokens[pad_mask]
        update_mu = weighted_mu[pad_mask]
        update_lr = effective_lr[pad_mask]

        # Vectorized EMA update for mu
        embed.mu_embed.weight.data[update_tokens] = (
            (1.0 - update_lr.unsqueeze(-1)) * embed.mu_embed.weight.data[update_tokens]
            + update_lr.unsqueeze(-1) * update_mu
        )

        # Sigma P-flow: update log_sigma_diag if learnable
        if sigma_beliefs is not None and embed.learnable_sigma and hasattr(embed, 'log_sigma_diag'):
            # Handle full covariance (B, N, K, K) → extract diagonal
            if sigma_beliefs.dim() == 4:
                sigma_beliefs_diag = sigma_beliefs.diagonal(dim1=-2, dim2=-1)
            else:
                sigma_beliefs_diag = sigma_beliefs
            flat_sigma = sigma_beliefs_diag.reshape(-1, K)
            weighted_sigma = torch.zeros(n_unique, K, device=flat_mu.device, dtype=flat_mu.dtype)
            weighted_sigma.scatter_add_(
                0, inverse_idx.unsqueeze(-1).expand(-1, K),
                flat_sigma * weights.unsqueeze(-1),
            )
            update_sigma = weighted_sigma[pad_mask]
            sigma_lr = update_lr * 0.1  # Slower sigma updates for stability
            current_sigma = torch.exp(embed.log_sigma_diag.data[update_tokens])
            new_sigma = (
                (1.0 - sigma_lr.unsqueeze(-1)) * current_sigma
                + sigma_lr.unsqueeze(-1) * update_sigma
            )
            embed.log_sigma_diag.data[update_tokens] = torch.log(new_sigma.clamp(min=1e-6))


def update_phi_from_beliefs(
    embed,                              # GaugeTokenEmbedding instance
    token_ids: torch.Tensor,
    phi_evolved: torch.Tensor,          # (B, N, phi_dim) VFE-evolved phi
    prediction_errors: torch.Tensor,
    ema_decay: float = 0.99,
    pad_token_id: int = -100,
):
    r"""Phi P-flow: Update gauge frame embeddings toward VFE-evolved values.

    The E-step evolves phi within each forward pass to minimize transported
    KL.  This function persists those evolved values back to ``phi_embed``
    via EMA, providing a local learning rule for gauge frames without
    backprop.
    """
    if not hasattr(embed, 'phi_embed'):
        return

    phi_dim = phi_evolved.shape[-1]
    lr = 1.0 - ema_decay

    with torch.no_grad():
        unique_tokens, inverse_idx, weights, n_unique = compute_pflow_weights(
            token_ids, prediction_errors, pad_token_id,
        )

        flat_phi = phi_evolved.reshape(-1, phi_dim)  # (B*N, phi_dim)

        # Weighted mean phi per token type
        weighted_phi = torch.zeros(n_unique, phi_dim, device=flat_phi.device, dtype=flat_phi.dtype)
        weighted_phi.scatter_add_(
            0, inverse_idx.unsqueeze(-1).expand(-1, phi_dim),
            flat_phi * weights.unsqueeze(-1),
        )

        # Skip padding
        pad_mask = (unique_tokens != pad_token_id)
        update_tokens = unique_tokens[pad_mask]
        update_phi = weighted_phi[pad_mask]

        # Vectorized EMA update for phi
        embed.phi_embed.weight.data[update_tokens] = (
            (1.0 - lr) * embed.phi_embed.weight.data[update_tokens]
            + lr * update_phi
        )


# =============================================================================
# PriorBank helpers
# =============================================================================

def update_prior_bank_from_beliefs(
    prior_bank,                          # PriorBank instance
    token_ids: torch.Tensor,
    mu_beliefs: torch.Tensor,
    sigma_beliefs: torch.Tensor,
    prediction_errors: torch.Tensor,
    lr: float = 0.01,
):
    """Update token priors via prediction-error weighted EMA.

    Pure FEP learning mechanism:
    - Beliefs with low prediction error are "good" — priors move toward them
    - Beliefs with high prediction error are "bad" — priors ignore them
    - For each token, aggregates across all its occurrences in the batch

    CRITICAL: Updates priors by TOKEN ID, not position!
    """
    if prior_bank.gauge_fixed_priors:
        # Gauge-fixed M-step: update base prior (μ_0, σ_0) via de-rotation.
        update_gauge_fixed_base_prior(
            prior_bank, token_ids, mu_beliefs, sigma_beliefs, prediction_errors, lr,
        )
        return

    with torch.no_grad():
        # Full covariance (B, N, K, K) → diagonal variances (B, N, K)
        if sigma_beliefs.dim() == 4:
            sigma_beliefs = torch.diagonal(sigma_beliefs, dim1=-2, dim2=-1)

        B, N, K = mu_beliefs.shape

        # Flatten batch dimensions: (B*N,)
        flat_ids = token_ids.reshape(-1)
        flat_mu = mu_beliefs.reshape(-1, K)
        flat_sigma = sigma_beliefs.reshape(-1, K)
        flat_errors = prediction_errors.reshape(-1)

        # Filter padding tokens (pad_token_id=-100 wraps via negative indexing
        # and silently corrupts prior_mu[vocab_size - 100]).
        valid_mask = flat_ids >= 0
        if not valid_mask.any():
            return
        flat_ids = flat_ids[valid_mask]
        flat_mu = flat_mu[valid_mask]
        flat_sigma = flat_sigma[valid_mask]
        flat_errors = flat_errors[valid_mask]

        # Compute per-token-type weights via segment-wise softmax
        neg_errors = -flat_errors.clamp(min=-10, max=10)
        unique_tokens, inverse_idx = torch.unique(flat_ids, return_inverse=True)
        n_unique = unique_tokens.shape[0]

        seg_max = torch.full((n_unique,), float('-inf'), device=flat_ids.device, dtype=flat_mu.dtype)
        seg_max.scatter_reduce_(0, inverse_idx, neg_errors, reduce='amax', include_self=False)
        shifted = neg_errors - seg_max[inverse_idx]
        exp_shifted = torch.exp(shifted)

        seg_sum = torch.zeros(n_unique, device=flat_ids.device, dtype=flat_mu.dtype)
        seg_sum.scatter_add_(0, inverse_idx, exp_shifted)
        weights = exp_shifted / seg_sum[inverse_idx].clamp(min=1e-12)

        # Weighted means per token type
        weighted_mu = torch.zeros(n_unique, K, device=flat_ids.device, dtype=flat_mu.dtype)
        weighted_mu.scatter_add_(0, inverse_idx.unsqueeze(-1).expand(-1, K), flat_mu * weights.unsqueeze(-1))

        # Mean error per token type for confidence-weighted LR
        seg_error_sum = torch.zeros(n_unique, device=flat_ids.device, dtype=flat_mu.dtype)
        seg_error_sum.scatter_add_(0, inverse_idx, flat_errors)
        seg_count = torch.zeros(n_unique, device=flat_ids.device, dtype=flat_mu.dtype)
        seg_count.scatter_add_(0, inverse_idx, torch.ones_like(flat_errors))
        mean_errors = seg_error_sum / seg_count.clamp(min=1)
        confidence = 1.0 / (1.0 + mean_errors)
        effective_lr = lr * confidence

        # Vectorized EMA update for mu: prior ← (1-lr)*prior + lr*belief
        prior_bank.prior_mu.data[unique_tokens] = (
            (1.0 - effective_lr.unsqueeze(-1)) * prior_bank.prior_mu.data[unique_tokens]
            + effective_lr.unsqueeze(-1) * weighted_mu
        )

        # Vectorized EMA update for sigma
        if prior_bank.learnable_sigma:
            weighted_sigma = torch.zeros(n_unique, K, device=flat_ids.device, dtype=flat_mu.dtype)
            weighted_sigma.scatter_add_(0, inverse_idx.unsqueeze(-1).expand(-1, K), flat_sigma * weights.unsqueeze(-1))
            sigma_lr = effective_lr * 0.1
            current_sigma = torch.exp(prior_bank.log_prior_sigma.data[unique_tokens])
            new_sigma = (1.0 - sigma_lr.unsqueeze(-1)) * current_sigma + sigma_lr.unsqueeze(-1) * weighted_sigma
            prior_bank.log_prior_sigma.data[unique_tokens] = torch.log(new_sigma.clamp(min=prior_bank.eps))


def update_gauge_fixed_base_prior(
    prior_bank,
    token_ids: torch.Tensor,
    mu_beliefs: torch.Tensor,
    sigma_beliefs: torch.Tensor,
    prediction_errors: torch.Tensor,
    lr: float,
):
    r"""Update base prior (μ_0, σ_0) via de-rotation of evolved beliefs.

    For gauge-fixed priors, each token's prior is ``μ_v = A_v · μ_0``
    where ``A_v = exp(φ_v · G)``.  Given evolved beliefs ``μ_q`` at token
    v, the de-rotated target for the base prior is

        ``μ_0^{target} = A_v⁻¹ · μ_q``

    We compute prediction-error-weighted averages of these de-rotated
    beliefs and EMA-update the base prior toward them.

    φ_v updates are handled by backprop through the VFE loss — the
    gradient ∂F/∂φ_v flows through ``A_v = exp(φ_v · G)`` in the forward
    pass.  This function handles the base prior only.
    """
    with torch.no_grad():
        if sigma_beliefs.dim() == 4:
            sigma_beliefs = torch.diagonal(sigma_beliefs, dim1=-2, dim2=-1)

        B, N, K = mu_beliefs.shape

        # Filter padding tokens before gauge transform lookup.
        # pad_token_id=-100 would wrap via negative indexing in phi_embed,
        # silently corrupting the base prior.
        flat_token_ids = token_ids.reshape(-1)
        valid_mask = flat_token_ids >= 0
        if not valid_mask.any():
            return

        mu_flat_all = mu_beliefs.reshape(-1, K)
        sigma_flat_all = sigma_beliefs.reshape(-1, K)
        errors_flat_all = prediction_errors.reshape(-1)

        # Apply mask
        valid_ids = flat_token_ids[valid_mask]
        mu_valid = mu_flat_all[valid_mask]
        sigma_valid = sigma_flat_all[valid_mask]
        errors_valid = errors_flat_all[valid_mask]
        n_valid = valid_ids.shape[0]

        # Get gauge transforms for valid tokens only
        phi = prior_bank.phi_embed(valid_ids.unsqueeze(0)).squeeze(0)  # (n_valid, phi_dim)
        A = prior_bank._compute_gauge_transform(phi.unsqueeze(0)).squeeze(0)  # (n_valid, K, K)

        # Compute A_v⁻¹ via solving the linear system (more stable than .inverse())
        mu_flat = mu_valid.unsqueeze(-1)  # (n_valid, K, 1)
        A_flat = A  # (n_valid, K, K)
        derotated_mu = torch.linalg.solve(A_flat, mu_flat).squeeze(-1)  # (n_valid, K)

        # Similarly de-rotate sigma: diag(A⁻¹ diag(σ_q) A⁻ᵀ)
        A_inv = torch.linalg.solve(
            A_flat, torch.eye(K, device=A_flat.device).expand_as(A_flat),
        )  # (n_valid, K, K)
        sigma_flat = sigma_valid  # (n_valid, K)
        derotated_sigma = torch.einsum(
            'bkj,bj->bk', A_inv ** 2, sigma_flat,
        )  # (n_valid, K)

        # Prediction-error weighted aggregation
        flat_errors = errors_valid
        neg_errors = -flat_errors.clamp(min=-10, max=10)

        # Global softmax over all positions (all contribute to base prior)
        weights = torch.softmax(neg_errors, dim=0)  # (B*N,)

        # Weighted mean of de-rotated beliefs
        weighted_mu = (derotated_mu * weights.unsqueeze(-1)).sum(dim=0)  # (K,)
        weighted_sigma = (derotated_sigma * weights.unsqueeze(-1)).sum(dim=0)  # (K,)

        # Confidence-weighted learning rate
        mean_error = flat_errors.mean()
        confidence = 1.0 / (1.0 + mean_error)
        effective_lr = lr * confidence

        # EMA update for base prior mean
        prior_bank.base_prior_mu.data.lerp_(weighted_mu, effective_lr)

        # EMA update for base prior sigma
        if prior_bank.learnable_sigma:
            current_sigma = torch.exp(prior_bank.base_log_prior_sigma.data)
            sigma_lr = effective_lr * 0.1  # Slower sigma updates
            new_sigma = current_sigma.lerp(weighted_sigma, sigma_lr)
            prior_bank.base_log_prior_sigma.data.copy_(
                torch.log(new_sigma.clamp(min=prior_bank.eps)),
            )


# =============================================================================
# Model-level dispatchers
# =============================================================================

def p_flow_update_model(
    model,                              # GaugeTransformerLM instance
    token_ids: torch.Tensor,
    mu_beliefs: torch.Tensor,
    prediction_errors: torch.Tensor,
    ema_decay: float = 0.99,
    sigma_beliefs: Optional[torch.Tensor] = None,
    pad_token_id: int = -100,
):
    """P-flow dispatcher: route to PriorBank or GaugeTokenEmbedding.

    Updates token embeddings (mu + sigma) toward successful beliefs via
    EMA, weighted by inverse prediction error.
    """
    if model.use_prior_bank and model.prior_bank is not None:
        update_prior_bank_from_beliefs(
            prior_bank=model.prior_bank,
            token_ids=token_ids,
            mu_beliefs=mu_beliefs,
            sigma_beliefs=sigma_beliefs if sigma_beliefs is not None else torch.ones_like(mu_beliefs),
            prediction_errors=prediction_errors,
            lr=1.0 - ema_decay,
        )
    elif hasattr(model.token_embed, 'update_embeddings_from_beliefs'):
        # The embedding still owns the wrapper method; the body now lives
        # in update_embeddings_from_beliefs (this module).
        update_embeddings_from_beliefs(
            embed=model.token_embed,
            token_ids=token_ids,
            mu_beliefs=mu_beliefs,
            prediction_errors=prediction_errors,
            ema_decay=ema_decay,
            sigma_beliefs=sigma_beliefs,
            pad_token_id=pad_token_id,
        )


def phi_flow_update_model(
    model,                              # GaugeTransformerLM instance
    token_ids: torch.Tensor,
    phi_evolved: torch.Tensor,
    prediction_errors: torch.Tensor,
    ema_decay: float = 0.99,
    pad_token_id: int = -100,
):
    """Phi P-flow dispatcher: route to PriorBank.phi_embed or token embedding.

    Updates gauge frame embeddings toward VFE-evolved values.  Used when
    ``detach_phi=True`` to provide a local learning rule for phi without
    backprop.
    """
    if model.use_prior_bank and model.prior_bank is not None:
        # PriorBank owns phi_embed — update it directly via EMA.
        # Inline the EMA logic here because the destination is the
        # PriorBank's phi_embed nn.Embedding, not a GaugeTokenEmbedding.
        phi_embed = model.prior_bank.phi_embed
        phi_dim = phi_evolved.shape[-1]
        lr = 1.0 - ema_decay
        with torch.no_grad():
            flat_ids = token_ids.reshape(-1)
            flat_phi = phi_evolved.reshape(-1, phi_dim)
            flat_errors = prediction_errors.reshape(-1)
            # Prediction-error-weighted average per token type
            neg_errors = -flat_errors.clamp(min=-10, max=10)
            unique_tokens, inverse_idx = torch.unique(flat_ids, return_inverse=True)
            n_unique = unique_tokens.shape[0]
            seg_max = torch.full((n_unique,), float('-inf'),
                                 device=flat_ids.device, dtype=flat_phi.dtype)
            seg_max.scatter_reduce_(0, inverse_idx, neg_errors, reduce='amax', include_self=False)
            exp_shifted = torch.exp(neg_errors - seg_max[inverse_idx])
            seg_sum = torch.zeros(n_unique, device=flat_ids.device, dtype=flat_phi.dtype)
            seg_sum.scatter_add_(0, inverse_idx, exp_shifted)
            weights = exp_shifted / seg_sum[inverse_idx].clamp(min=1e-12)
            weighted_phi = torch.zeros(n_unique, phi_dim, device=flat_phi.device, dtype=flat_phi.dtype)
            weighted_phi.scatter_add_(
                0, inverse_idx.unsqueeze(-1).expand(-1, phi_dim),
                flat_phi * weights.unsqueeze(-1),
            )
            pad_mask = (unique_tokens != pad_token_id)
            update_tokens = unique_tokens[pad_mask]
            update_phi = weighted_phi[pad_mask]
            phi_embed.weight.data[update_tokens] = (
                (1.0 - lr) * phi_embed.weight.data[update_tokens]
                + lr * update_phi
            )
    elif hasattr(model.token_embed, 'update_phi_from_beliefs'):
        update_phi_from_beliefs(
            embed=model.token_embed,
            token_ids=token_ids,
            phi_evolved=phi_evolved,
            prediction_errors=prediction_errors,
            ema_decay=ema_decay,
            pad_token_id=pad_token_id,
        )


def delta_rule_update_w_out_model(
    model,                              # GaugeTransformerLM instance
    mu_beliefs: torch.Tensor,
    targets: torch.Tensor,
    lr: float = 0.1,
    pad_token_id: int = -100,
):
    r"""Delta rule update for ``W_out`` — backprop-free Hebbian learning.

    Instead of backpropagating through the full computation graph, update
    ``W_out`` using the local Widrow–Hoff delta rule:

        ``ΔW = lr · (target − prediction) ⊗ μ^T``

    This is biologically plausible and does not require storing
    intermediate activations for backprop.

    No-op when ``use_prior_bank=True`` (PriorBank decodes via KL — there
    is no W_out to update).
    """
    if model.use_prior_bank and model.prior_bank is not None:
        return

    with torch.no_grad():
        B, N, K = mu_beliefs.shape
        V = model.config['vocab_size']

        # Mask out padding positions
        valid_mask = (targets != pad_token_id)  # (B, N)
        n_valid = valid_mask.sum().item()
        if n_valid == 0:
            return

        # Get current predictions: softmax(W_out @ mu)
        # NOTE: when learnable_reflection=True, this should apply sign
        # vectors to W_out rows (as in forward()). Currently unsupported
        # in delta_rule_update_w_out — use standard backprop training instead.
        logits = model.out_proj(mu_beliefs)  # (B, N, V)
        predictions = F.softmax(logits, dim=-1)  # (B, N, V)

        # One-hot encode targets (clamp pad tokens to 0 for valid one-hot)
        targets_safe = targets.clone()
        targets_safe[~valid_mask] = 0
        targets_onehot = F.one_hot(targets_safe, num_classes=V).float()  # (B, N, V)

        # Prediction error: (target - prediction), zeroed at padding positions
        error = targets_onehot - predictions  # (B, N, V)
        error = error * valid_mask.unsqueeze(-1).float()  # Zero out pad positions

        # Delta rule: ΔW = error^T @ mu (outer product averaged over valid positions)
        # W_out shape is (V, K), so we need: (V, K) += (B*N, V)^T @ (B*N, K)
        error_flat = error.reshape(-1, V)  # (B*N, V)
        mu_flat = mu_beliefs.reshape(-1, K)  # (B*N, K)

        # Compute delta: (V, K) = (V, B*N) @ (B*N, K)
        delta_W = error_flat.t() @ mu_flat  # (V, K)
        delta_W /= n_valid  # Average over valid (non-padded) positions

        # Apply update to W_out
        model.out_proj.weight.add_(lr * delta_W)


# =============================================================================
# Trainer-level orchestration
# =============================================================================

def apply_p_flow_and_delta_rule(
    trainer,                            # PublicationTrainer instance
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    full_metrics: Dict,
    is_standard: bool,
    use_delta_rule: bool,
) -> None:
    """Apply P-flow EMA update and delta rule W_out update.

    This is the entry point called from
    ``PublicationTrainer.train_step`` after the backward pass.

    P-flow:
        EMA update of token embeddings (mu, sigma, optionally phi) toward
        beliefs that predicted successfully (low CE).  Disabled for the
        standard transformer.

    Delta rule:
        Backprop-free Hebbian update of W_out:
            ``ΔW = lr · (target − prediction) ⊗ μ^T``
        Combined with P-flow + ``detach_phi`` this makes learning fully
        backprop-free.
    """
    # --- P-FLOW ---
    use_p_flow = getattr(trainer.config, 'use_p_flow', False)
    if use_p_flow and not is_standard and 'p_flow/mu_q' in full_metrics:
        mu_beliefs = full_metrics['p_flow/mu_q']
        ce_per_position = full_metrics['p_flow/ce_per_position']
        ema_decay = getattr(trainer.config, 'p_flow_ema_decay', 0.99)

        sigma_beliefs = full_metrics.get('p_flow/sigma_q')
        if hasattr(trainer.model, 'p_flow_update'):
            trainer.model.p_flow_update(
                token_ids=input_ids,
                mu_beliefs=mu_beliefs,
                prediction_errors=ce_per_position,
                ema_decay=ema_decay,
                sigma_beliefs=sigma_beliefs,
                pad_token_id=trainer.pad_token_id,
            )

        # Phi P-flow: update gauge frames toward VFE-evolved values
        # Only when detach_phi=True (phi is detached from backprop)
        if (getattr(trainer.config, 'detach_phi', False) and
                'p_flow/phi_evolved' in full_metrics and
                hasattr(trainer.model, 'phi_flow_update')):
            trainer.model.phi_flow_update(
                token_ids=input_ids,
                phi_evolved=full_metrics['p_flow/phi_evolved'],
                prediction_errors=ce_per_position,
                ema_decay=ema_decay,
                pad_token_id=trainer.pad_token_id,
            )

    # --- DELTA RULE ---
    if use_delta_rule and 'p_flow/mu_q' in full_metrics:
        mu_beliefs = full_metrics['p_flow/mu_q']
        delta_lr = getattr(trainer.config, 'delta_rule_lr', 0.1)
        if hasattr(trainer.model, 'delta_rule_update_w_out'):
            trainer.model.delta_rule_update_w_out(
                mu_beliefs=mu_beliefs,
                targets=target_ids,
                lr=delta_lr,
                pad_token_id=trainer.pad_token_id,
            )
