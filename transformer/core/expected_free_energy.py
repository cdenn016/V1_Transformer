"""
Expected Free Energy for Canonical Active Inference
====================================================

Implements one-step expected free energy (EFE) over candidate token actions
for the Gauge-Theoretic VFE Transformer.  This is the canonical active
inference formulation from Parr, Pezzulo & Friston (2022), specialized to
next-token prediction.

For each candidate next-token action `a`, the expected free energy is:

.. math::

    G_t(a) = \\underbrace{E_{q(o|a)}[-\\log p^*(o)]}_{\\text{risk}}
           + \\underbrace{E_{q(z|a)}[H[p(o|z)]]}_{\\text{ambiguity}}
           - \\underbrace{I_q(z; o | a)}_{\\text{epistemic value}}

The policy posterior over next tokens is:

.. math::

    q_t(a) \\propto \\exp(-\\gamma\\, G_t(a))

Phase 1A: generation-time action selection via one-step rollout
Phase 1B: teacher-forced EFE proxy loss for training (surrogate, not
          genuine active inference — the "action" is the dataset token)

The transition model T_θ(x_t, C_t ⊕ a) reuses the model's own forward
pass: append candidate token a to context, run forward, extract beliefs
at the last position.  No neural network components are introduced.
"""

import logging
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from transformer.core.model import GaugeTransformerLM

logger = logging.getLogger(__name__)


# =============================================================================
# Pure math functions — Risk, Ambiguity, Epistemic Value
# =============================================================================

def compute_risk(
    predictive_probs: torch.Tensor,
    preference_mode: str = 'current_belief',
    preferences: Optional[torch.Tensor] = None,
    targets: Optional[torch.Tensor] = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    r"""Compute risk for each candidate action.

    Risk measures how much the predicted outcome after taking action `a`
    diverges from the preference distribution `p^*(o)`.

    Three modes:

    - ``'current_belief'``: Risk = :math:`-\sum_v q(v|a) \log p^*(v)`,
      standard cross-entropy.  Requires ``preferences`` with full support.
    - ``'target'``: Risk = :math:`-\log q(o = y | a)`, the negative
      log-likelihood of the target token under the rolled-out prediction.
      Avoids the :math:`-\log 0 = \infty` divergence that occurs when
      applying the general formula with delta preferences.
    - ``'uniform'``: Risk = :math:`\log |V| - H[q(o|a)]`.  The constant
      :math:`\log |V|` cancels across candidates.

    Args:
        predictive_probs: ``(K, V)`` predicted outcome distribution after
            rollout, one row per candidate action.
        preference_mode: One of ``'current_belief'``, ``'target'``,
            ``'uniform'``.
        preferences: ``(V,)`` or ``(K, V)`` preference distribution
            :math:`p^*(o)`.  Required for ``'current_belief'`` mode.
        targets: ``(K,)`` target token IDs.  Required for ``'target'`` mode.
        eps: Numerical floor for log.

    Returns:
        risk: ``(K,)`` risk value per candidate.
    """
    K, V = predictive_probs.shape

    if preference_mode == 'target':
        if targets is None:
            raise ValueError("targets required for preference_mode='target'")
        # Risk = -log q(o=y|a) — NLL at target token
        log_probs = predictive_probs.clamp(min=eps).log()  # (K, V)
        risk = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # (K,)
        return risk

    elif preference_mode == 'uniform':
        # Risk = log(V) - H[q]  (constant shift across candidates)
        log_probs = predictive_probs.clamp(min=eps).log()
        entropy = -(predictive_probs * log_probs).sum(dim=-1)  # (K,)
        return torch.log(torch.tensor(V, dtype=entropy.dtype, device=entropy.device)) - entropy

    elif preference_mode == 'current_belief':
        if preferences is None:
            raise ValueError("preferences required for preference_mode='current_belief'")
        # Risk = -sum_v q(v|a) * log p*(v)  — cross-entropy H(q, p*)
        log_pref = preferences.clamp(min=eps).log()  # (V,) or (K, V)
        risk = -(predictive_probs * log_pref).sum(dim=-1)  # (K,)
        return risk

    else:
        raise ValueError(f"Unknown preference_mode: {preference_mode}")


def compute_ambiguity(
    predictive_probs: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    r"""Compute ambiguity for each candidate action.

    Ambiguity = :math:`H[p(o | z_{t+1}(a))]`, the entropy of the readout
    distribution at the rolled-out belief state.  High ambiguity means the
    model is uncertain about what observation follows this action.

    With a deterministic rollout (delta transition), this is the entropy
    of the readout evaluated at the single rolled-out point.

    Args:
        predictive_probs: ``(K, V)`` readout probabilities after rollout.
        eps: Numerical floor for log.

    Returns:
        ambiguity: ``(K,)`` entropy per candidate.
    """
    log_probs = predictive_probs.clamp(min=eps).log()
    ambiguity = -(predictive_probs * log_probs).sum(dim=-1)  # (K,)
    return ambiguity


def compute_epistemic_value(
    mu_rollout: torch.Tensor,
    sigma_rollout: torch.Tensor,
    model: 'GaugeTransformerLM',
    n_samples: int = 4,
    eps: float = 1e-12,
) -> torch.Tensor:
    r"""Compute epistemic value (BALD mutual information) for each candidate.

    .. math::

        I_q(z; o | a) = H[\bar{p}] - \frac{1}{S}\sum_s H[p_s]

    where :math:`p_s = p(o | z_s)` with :math:`z_s \sim \mathcal{N}(\mu, \Sigma)`
    drawn from the rolled-out belief, and :math:`\bar{p} = \frac{1}{S}\sum_s p_s`.

    Uses the same streaming Welford-style two-pass pattern as
    ``active_inference._compute_active_inference_gradient`` for memory
    efficiency, but only computes the scalar MI (no autograd needed).

    **Structurally weaker for self-generated text** than with exogenous
    observations.  Off by default in generation.

    Args:
        mu_rollout: ``(K, K_dim)`` rolled-out belief means.
        sigma_rollout: ``(K, K_dim)`` rolled-out diagonal variances.
        model: GaugeTransformerLM for ``_compute_logits`` access.
        n_samples: Number of MC samples S.
        eps: Numerical floor.

    Returns:
        epistemic: ``(K,)`` mutual information per candidate.
    """
    K, K_dim = mu_rollout.shape
    device = mu_rollout.device

    std = sigma_rollout.clamp(min=1e-6).sqrt()  # (K, K_dim)

    # ----- Pass 1: streaming p_bar accumulation -----
    with torch.no_grad():
        probs_avg: Optional[torch.Tensor] = None
        sample_entropies: List[torch.Tensor] = []

        for _s in range(n_samples):
            noise = torch.randn_like(mu_rollout)  # (K, K_dim)
            mu_s = mu_rollout + noise * std  # (K, K_dim)

            # Readout at sampled belief — add N dimension for _compute_logits
            logits_s = model._compute_logits(
                mu_s.unsqueeze(1), sigma_rollout.unsqueeze(1), device,
            ).squeeze(1)  # (K, V)
            probs_s = F.softmax(logits_s, dim=-1)  # (K, V)

            # Per-sample entropy
            log_probs_s = probs_s.clamp(min=eps).log()
            H_s = -(probs_s * log_probs_s).sum(dim=-1)  # (K,)
            sample_entropies.append(H_s)

            # Streaming mean
            _contrib = probs_s / n_samples
            probs_avg = _contrib if probs_avg is None else probs_avg + _contrib

            del mu_s, logits_s, probs_s, log_probs_s, _contrib

    # ----- Compute MI -----
    # MI = H[p_bar] - mean_s(H[p_s])
    log_probs_avg = probs_avg.clamp(min=eps).log()
    H_bar = -(probs_avg * log_probs_avg).sum(dim=-1)  # (K,)
    mean_H_s = torch.stack(sample_entropies).mean(dim=0)  # (K,)
    epistemic = (H_bar - mean_H_s).clamp(min=0.0)  # (K,) — MI is non-negative

    return epistemic


# =============================================================================
# Rollout: batched one-step belief propagation
# =============================================================================

def rollout_candidates(
    model: 'GaugeTransformerLM',
    context_ids: torch.Tensor,
    candidate_ids: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    r"""Run batched one-step rollouts for K candidate next-token actions.

    For each candidate token ``a``:

    1. Append ``a`` to the context: ``rollout_ids = context || [a]``
    2. Run ``model.forward(rollout_ids, return_agents=True)``
    3. Extract beliefs and logits at the last position

    All K candidates are batched into a single forward pass for efficiency.

    **Compute note:** The batched ``(K, T+1)`` rollout recomputes the entire
    prefix for every candidate because there is no KV-cache (attention
    weights depend on pairwise KL divergences).  Cost: O(K * T^2 * L * I).

    Args:
        model: GaugeTransformerLM instance (should be in eval mode).
        context_ids: ``(1, T)`` current context token IDs.
        candidate_ids: ``(K,)`` candidate next-token IDs.

    Returns:
        Dict with:
            ``'mu'``: ``(K, K_dim)`` belief means at last position
            ``'sigma'``: ``(K, K_dim)`` diagonal variances at last position,
                or None if sigma not tracked
            ``'logits'``: ``(K, V)`` readout logits at last position
            ``'predictive_probs'``: ``(K, V)`` softmax of logits
    """
    K = candidate_ids.shape[0]
    device = context_ids.device
    max_seq_len = model.config['max_seq_len']

    # Build rollout batch: (K, T+1) — same context, different last token
    context_expanded = context_ids.expand(K, -1)  # (K, T)
    rollout_ids = torch.cat(
        [context_expanded, candidate_ids.unsqueeze(1)], dim=1,
    )  # (K, T+1)

    # Apply sliding window if needed
    if rollout_ids.shape[1] > max_seq_len:
        rollout_ids = rollout_ids[:, -max_seq_len:]

    # Forward pass — return_agents=True gives detached mu/sigma/phi
    with torch.no_grad():
        result = model.forward(rollout_ids, return_agents=True)
        logits_all = result[0]  # (K, T', V) where T' = min(T+1, max_seq_len)
        agents = result[1]

    # Extract last position only, free the rest
    logits_last = logits_all[:, -1, :].clone()  # (K, V)
    del logits_all

    mu_last = agents['mu'][:, -1, :]  # (K, K_dim)
    sigma_last = (
        agents['sigma'][:, -1, :]
        if agents['sigma'] is not None else None
    )  # (K, K_dim) or None

    predictive_probs = F.softmax(logits_last, dim=-1)  # (K, V)

    return {
        'mu': mu_last,
        'sigma': sigma_last,
        'logits': logits_last,
        'predictive_probs': predictive_probs,
    }


# =============================================================================
# Full EFE computation
# =============================================================================

def compute_efe(
    model: 'GaugeTransformerLM',
    context_ids: torch.Tensor,
    candidate_ids: torch.Tensor,
    gamma: float = 1.0,
    preference_mode: str = 'current_belief',
    preferences: Optional[torch.Tensor] = None,
    targets: Optional[torch.Tensor] = None,
    include_epistemic: bool = False,
    epistemic_samples: int = 4,
    eps: float = 1e-12,
) -> Dict[str, torch.Tensor]:
    r"""Compute one-step expected free energy for K candidate actions.

    Orchestrates: rollout → risk + ambiguity [- epistemic] → policy.

    .. math::

        G_t(a) = \mathrm{Risk}_t(a) + \mathrm{Amb}_t(a) - \mathrm{Epi}_t(a)
        q_t(a) \propto \exp(-\gamma\, G_t(a))

    Args:
        model: GaugeTransformerLM instance (eval mode).
        context_ids: ``(1, T)`` current context.
        candidate_ids: ``(K,)`` candidate next-token IDs.
        gamma: Policy precision (inverse temperature).  Higher = more
            greedy on lowest-EFE action.
        preference_mode: ``'current_belief'``, ``'target'``, ``'uniform'``.
        preferences: ``(V,)`` preference distribution for ``'current_belief'``
            mode.  If None and mode is ``'current_belief'``, computed from
            the model's current prediction.
        targets: ``(K,)`` target token IDs for ``'target'`` mode.
        include_epistemic: Whether to compute BALD MI.  Off by default —
            structurally weaker for self-generated text.
        epistemic_samples: MC samples for BALD MI.
        eps: Numerical stability floor.

    Returns:
        Dict with:
            ``'G'``: ``(K,)`` expected free energy per candidate
            ``'risk'``: ``(K,)`` risk component
            ``'ambiguity'``: ``(K,)`` ambiguity component
            ``'epistemic'``: ``(K,)`` epistemic value (0 if disabled)
            ``'policy'``: ``(K,)`` policy posterior q(a)
            ``'predictive_probs'``: ``(K, V)`` readout probs after rollout
            ``'candidate_ids'``: ``(K,)`` the candidate token IDs
    """
    device = context_ids.device

    # ----- Compute current-belief preferences if needed -----
    if preference_mode == 'current_belief' and preferences is None:
        with torch.no_grad():
            result = model.forward(context_ids)
            current_logits = result[0] if isinstance(result, tuple) else result
            preferences = F.softmax(current_logits[:, -1, :].squeeze(0), dim=-1)  # (V,)
            del current_logits

    # ----- Rollout: batched forward pass with each candidate appended -----
    rollout = rollout_candidates(model, context_ids, candidate_ids)
    pred_probs = rollout['predictive_probs']  # (K, V)

    # ----- Risk -----
    risk = compute_risk(
        pred_probs, preference_mode=preference_mode,
        preferences=preferences, targets=targets, eps=eps,
    )

    # ----- Ambiguity -----
    ambiguity = compute_ambiguity(pred_probs, eps=eps)

    # ----- Epistemic value (optional) -----
    if include_epistemic and rollout['sigma'] is not None:
        epistemic = compute_epistemic_value(
            rollout['mu'], rollout['sigma'], model,
            n_samples=epistemic_samples, eps=eps,
        )
    else:
        epistemic = torch.zeros_like(risk)

    # ----- G = Risk + Ambiguity - Epistemic -----
    G = risk + ambiguity - epistemic

    # ----- Policy posterior: q(a) ∝ exp(-γ G(a)) -----
    policy = F.softmax(-gamma * G, dim=0)  # (K,)

    return {
        'G': G,
        'risk': risk,
        'ambiguity': ambiguity,
        'epistemic': epistemic,
        'policy': policy,
        'predictive_probs': pred_probs,
        'candidate_ids': candidate_ids,
    }


# =============================================================================
# Generation loop
# =============================================================================

def generate_active_inference(
    model: 'GaugeTransformerLM',
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    gamma: float = 1.0,
    top_k: int = 50,
    preference_mode: str = 'current_belief',
    include_epistemic: bool = False,
    epistemic_samples: int = 4,
    temperature: float = 1.0,
    verbose: bool = False,
) -> torch.Tensor:
    r"""Autoregressive generation using EFE-based action selection.

    At each step:

    1. Forward pass on current context → current predictive distribution
    2. Select top-K candidates by probability
    3. Rollout each candidate (batched forward pass with candidate appended)
    4. Compute :math:`G_t(a)` = Risk + Ambiguity [- Epistemic] for each
    5. Form policy :math:`q(a) \propto \exp(-\gamma\, G(a) / T)`
    6. Sample from policy, append token, repeat

    **Not decorated with** ``@torch.inference_mode()`` — uses
    ``torch.no_grad()`` internally.  All computation is forward-only
    evaluation.

    **Compute cost:** Each generated token requires ~K+1 forward passes
    (1 for current prediction + K batched rollouts).  For K=50, T=128:
    ~50x slower than standard generation.

    Args:
        model: GaugeTransformerLM instance.
        prompt_ids: ``(1, prompt_len)`` initial tokens.
        max_new_tokens: Number of tokens to generate.
        gamma: Policy precision.  Higher = more greedy on lowest-EFE.
        top_k: Number of candidate actions to evaluate.
        preference_mode: ``'current_belief'`` or ``'uniform'``.
        include_epistemic: Whether to compute BALD MI term.
        epistemic_samples: MC samples for BALD MI.
        temperature: Additional temperature on policy (applied as
            :math:`q(a) \propto \exp(-\gamma\, G(a) / T)`).
        verbose: Print per-step diagnostics.

    Returns:
        generated: ``(1, prompt_len + max_new_tokens)`` full sequence.
    """
    was_training = model.training
    model.eval()

    try:
        generated = prompt_ids.clone()
        max_seq_len = model.config['max_seq_len']
        device = prompt_ids.device

        for step in range(max_new_tokens):
            # Sliding window
            context = (
                generated[:, -max_seq_len:]
                if generated.shape[1] > max_seq_len else generated
            )

            # ----- Step 1: current prediction for candidate selection -----
            with torch.no_grad():
                result = model.forward(context)
                current_logits = (
                    result[0] if isinstance(result, tuple) else result
                )  # (1, T, V)
                last_logits = current_logits[:, -1, :]  # (1, V)

            # ----- Step 2: select top-K candidates -----
            K_actual = min(top_k, last_logits.shape[-1])
            _, top_indices = torch.topk(last_logits.squeeze(0), K_actual)
            candidate_ids = top_indices  # (K,)

            # Current-belief preferences from this forward pass
            preferences = F.softmax(last_logits.squeeze(0), dim=-1)  # (V,)

            del current_logits, last_logits

            # ----- Steps 3-5: EFE computation -----
            efe_result = compute_efe(
                model=model,
                context_ids=context,
                candidate_ids=candidate_ids,
                gamma=gamma,
                preference_mode=preference_mode,
                preferences=preferences if preference_mode == 'current_belief' else None,
                include_epistemic=include_epistemic,
                epistemic_samples=epistemic_samples,
            )

            # Apply temperature to policy
            if temperature != 1.0 and temperature > 0.0:
                policy = F.softmax(-gamma * efe_result['G'] / temperature, dim=0)
            else:
                policy = efe_result['policy']

            # ----- Step 6: sample from policy -----
            if temperature <= 0.0:
                # Greedy: pick lowest-EFE action
                idx = efe_result['G'].argmin()
            else:
                idx = torch.multinomial(policy, 1).squeeze()

            next_token = candidate_ids[idx].unsqueeze(0).unsqueeze(0)  # (1, 1)
            generated = torch.cat([generated, next_token], dim=1)

            if verbose:
                token_id = candidate_ids[idx].item()
                logger.info(
                    f"Step {step}: token={token_id}, "
                    f"G={efe_result['G'][idx]:.4f}, "
                    f"risk={efe_result['risk'][idx]:.4f}, "
                    f"amb={efe_result['ambiguity'][idx]:.4f}, "
                    f"epi={efe_result['epistemic'][idx]:.4f}, "
                    f"policy_entropy={-(policy * policy.clamp(min=1e-12).log()).sum():.4f}"
                )

        return generated

    finally:
        if was_training:
            model.train()


# =============================================================================
# Teacher-forced EFE proxy loss (Phase 1B)
# =============================================================================

def efe_teacher_forced_proxy_loss(
    logits: torch.Tensor,
    mu_q: torch.Tensor,
    sigma_q: Optional[torch.Tensor],
    targets: torch.Tensor,
    model: 'GaugeTransformerLM',
    eta: float = 0.1,
    pad_token_id: int = -100,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    r"""Teacher-forced EFE proxy loss for training.

    **This is a surrogate, not genuine active inference.** The "action" at
    each position is the dataset token :math:`y_{t+1}`, not a token chosen
    by the model.  The "rollout" is the model's own beliefs at position
    :math:`t+1` from the teacher-forced forward pass — no separate rollout
    is needed.

    For each valid position :math:`t`:

    - **Risk:** :math:`-\log q(o = y_{t+2} \mid z_{t+1})` — how well the
      post-action belief predicts the *next-next* target.
    - **Ambiguity:** :math:`H[p(o \mid z_{t+1})]` — entropy of the readout
      at the post-action belief.

    .. math::

        L_{\text{proxy}} = \eta \cdot \frac{1}{|T_{\text{valid}}|}
                           \sum_{t \in T_{\text{valid}}} G_t(y_{t+1})

    Epistemic term is omitted (set to 0) — structurally weaker for
    teacher-forced training where "observations" are dataset tokens.

    **Consumes** ``mu_q`` and ``sigma_q`` from ``forward_with_attention()``
    (via ``attn_info``).  Do NOT call plain ``model.forward()`` during
    training — broken with ``implicit_em=True``.

    Args:
        logits: ``(B, N, V)`` logits from ``forward_with_attention()``.
        mu_q: ``(B, N, K)`` evolved belief means from ``attn_info['mu']``.
        sigma_q: ``(B, N, K)`` diagonal variances from ``attn_info['sigma']``,
            or None.
        targets: ``(B, N)`` target token IDs.
        model: GaugeTransformerLM for ``_compute_logits`` access.
        eta: Weight for the proxy loss.
        pad_token_id: Token ID to ignore in loss computation.
        eps: Numerical stability floor.

    Returns:
        (loss, metrics): scalar loss tensor and dict with diagnostic values.
    """
    B, N, V = logits.shape
    device = logits.device

    if eta <= 0.0 or N < 3:
        return (
            torch.tensor(0.0, device=device, requires_grad=True),
            {'efe_proxy_loss': 0.0, 'efe_proxy_risk_mean': 0.0,
             'efe_proxy_amb_mean': 0.0},
        )

    # ----- Shifted beliefs: position t+1 is the "rollout" for action y_{t+1} -----
    # We use beliefs at positions 1..N-2 as rollout states
    # Targets at positions 2..N-1 as the "next-next" tokens for risk
    mu_rollout = mu_q[:, 1:-1, :]         # (B, N-2, K) — beliefs at t+1
    sigma_rollout = (
        sigma_q[:, 1:-1, :] if sigma_q is not None else None
    )                                      # (B, N-2, K) or None
    next_next_targets = targets[:, 2:]     # (B, N-2) — y_{t+2}

    # ----- Compute logits at rolled-out beliefs -----
    # Use model._compute_logits for correct PriorBank/reflection handling
    rollout_logits = model._compute_logits(
        mu_rollout, sigma_rollout, device,
    )  # (B, N-2, V)
    rollout_probs = F.softmax(rollout_logits, dim=-1)  # (B, N-2, V)

    # ----- Risk: -log q(y_{t+2} | z_{t+1}) -----
    # Mask padded positions
    valid_mask = (next_next_targets != pad_token_id).float()  # (B, N-2)
    n_valid = valid_mask.sum().clamp(min=1.0)

    # Gather log-probs at target positions
    log_probs = rollout_probs.clamp(min=eps).log()  # (B, N-2, V)
    target_indices = next_next_targets.clamp(min=0).unsqueeze(-1)  # (B, N-2, 1)
    target_log_probs = log_probs.gather(2, target_indices).squeeze(-1)  # (B, N-2)
    risk = -target_log_probs * valid_mask  # (B, N-2)

    # ----- Ambiguity: H[p(o | z_{t+1})] -----
    ambiguity = -(rollout_probs * log_probs).sum(dim=-1) * valid_mask  # (B, N-2)

    # ----- G = Risk + Ambiguity (no epistemic) -----
    G = risk + ambiguity  # (B, N-2)

    # ----- Loss -----
    loss = eta * G.sum() / n_valid

    # ----- Metrics -----
    with torch.no_grad():
        metrics = {
            'efe_proxy_loss': loss.item(),
            'efe_proxy_risk_mean': (risk.sum() / n_valid).item(),
            'efe_proxy_amb_mean': (ambiguity.sum() / n_valid).item(),
            'efe_proxy_G_mean': (G.sum() / n_valid).item(),
            'efe_proxy_n_valid': n_valid.item(),
        }

    return loss, metrics
