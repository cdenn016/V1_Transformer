"""
VFEExpectedFreeEnergy: generation-time EFE policy over candidate next tokens.

This is the correct place for active inference in a language model:
generation-time action selection, not target-conditioned E-step inference.

    G_t(a) = Risk + Ambiguity - Epistemic_value
    q_t(a) ~ exp(-gamma * G_t(a))

See VFE_Transformer_Idea.md Section 11.
"""

from __future__ import annotations

from typing import Optional, Dict, Literal, Tuple, TYPE_CHECKING

PreferenceMode = Literal['current_belief', 'target', 'uniform']

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from transformer.vfe.model import VFEModel

from transformer.core.expected_free_energy import compute_risk


class VFEExpectedFreeEnergy:
    r"""Generation-time EFE policy for action (next-token) selection.

    For each candidate next token :math:`a`, scores:

    .. math::
        G_t(a) = \underbrace{\mathbb{E}_{q(o|a)}[-\log p^\star(o)]}_{\text{risk}}
               + \underbrace{\mathbb{E}_{q(z|a)}[H[p(o|z)]]}_{\text{ambiguity}}
               - \underbrace{I_q(z;\, o \mid a)}_{\text{epistemic value}}

    The policy posterior is :math:`q_t(a) \propto \exp(-\gamma\, G_t(a))`.

    The epistemic term uses BALD mutual information:
    :math:`I = H[\bar{p}] - \frac{1}{S}\sum_s H[p_s]` where
    :math:`p_s = p(o|z_s)` with :math:`z_s \sim \mathcal{N}(\mu, \Sigma)`.
    It is structurally weaker for self-generated text (no exogenous
    observations) and is controlled by ``epistemic_weight``.

    Args:
        model: VFEModel instance for rollout computation.
        gamma: Inverse temperature for action policy.
        preference_mode: Risk computation mode (``'current_belief'``, ``'target'``, or ``'uniform'``).
        epistemic_weight: Weight for epistemic value term (0 = disabled).
        epistemic_samples: Number of MC samples for BALD MI estimate.
    """

    def __init__(
        self,
        model: 'VFEModel',
        gamma: float = 1.0,
        preference_mode: PreferenceMode = 'uniform',
        epistemic_weight: float = 0.0,
        epistemic_samples: int = 4,
    ) -> None:
        if preference_mode not in ('current_belief', 'target', 'uniform'):
            raise ValueError(
                f"preference_mode must be 'current_belief', 'target', or "
                f"'uniform'; got {preference_mode!r}."
            )
        self.model = model
        self.gamma = gamma
        self.preference_mode = preference_mode
        self.epistemic_weight = epistemic_weight
        self.epistemic_samples = epistemic_samples

    def _compute_epistemic_value(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Compute BALD mutual information and mean predictive entropy.

        :math:`I(z; o | q) = H[\bar{p}] - \frac{1}{S}\sum_s H[p_s]`

        where :math:`p_s = \text{softmax}(\text{decode}(z_s))` with
        :math:`z_s = \mu + \sqrt{\sigma} \cdot \varepsilon_s`.

        Args:
            mu: ``(1, N, K)`` belief means from rollout.
            sigma: ``(1, N, K)`` diagonal variances from rollout.

        Returns:
            ``(epistemic_mi, mean_H)`` — the BALD mutual information
            :math:`H[\bar p] - \mathbb E_z[H[p(o|z)]]` AND the mean
            predictive entropy :math:`\mathbb E_z[H[p(o|z)]]`. The mean
            entropy is the canonical EFE "ambiguity" term per
            [ParrPezzuloFriston2022]; the marginal-predictive entropy
            ``H[\bar p]`` would double-count uncertainty.
        """
        S = self.epistemic_samples
        prior_bank = self.model.prior_bank
        eps = 1e-12

        # Extract diagonal if full covariance
        if sigma.dim() == 4:
            sigma = torch.diagonal(sigma, dim1=-2, dim2=-1)

        mu_last = mu[:, -1:, :]     # (1, 1, K)
        std_last = sigma[:, -1:, :].clamp(min=1e-6).sqrt()  # (1, 1, K)

        # Streaming computation of p_bar and per-sample entropies
        p_bar = None
        H_samples = []

        # MC sample z_s already absorbs σ via z_s = μ + √σ · ε.
        # Pass σ=0 to decode so the predictive treats z_s as a point estimate
        # — otherwise σ enters BALD MI twice and inflates predictive entropy.
        zero_sigma = torch.zeros_like(sigma[:, -1:, :])
        for s in range(S):
            noise = torch.randn_like(mu_last)
            z_s = mu_last + std_last * noise  # (1, 1, K)
            logits_s = prior_bank.decode(z_s, zero_sigma)  # (1, 1, V)
            p_s = F.softmax(logits_s[:, 0, :], dim=-1)  # (1, V)

            H_s = -(p_s * (p_s + eps).log()).sum(dim=-1)  # (1,)
            H_samples.append(H_s)

            if p_bar is None:
                p_bar = p_s / S
            else:
                p_bar = p_bar + p_s / S

        # H[p_bar] - mean(H[p_s])
        H_bar = -(p_bar * (p_bar + eps).log()).sum(dim=-1)  # (1,)
        mean_H = torch.stack(H_samples).mean(dim=0)  # (1,)

        return (H_bar - mean_H).squeeze(), mean_H.squeeze()

    @torch.no_grad()
    def score_candidates(
        self,
        context_ids: torch.Tensor,
        candidate_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        r"""Score candidate next tokens by expected free energy.

        For each candidate, appends it to the context, runs a forward pass,
        and computes risk + ambiguity (- epistemic) from the predictive distribution.

        Args:
            context_ids: ``(1, N)`` context token IDs.
            candidate_ids: ``(C,)`` candidate next-token IDs to score.

        Returns:
            Dict with keys ``'efe'``, ``'risk'``, ``'ambiguity'``,
            ``'epistemic'``, each ``(C,)``.
        """
        C = candidate_ids.shape[0]
        device = context_ids.device

        risks = []
        ambiguities = []
        epistemics = []

        max_len = self.model.cfg.max_seq_len

        for c in range(C):
            # Append candidate to context, truncate to max_seq_len
            trial_ids = torch.cat([
                context_ids,
                candidate_ids[c:c+1].unsqueeze(0)
            ], dim=1)  # (1, N+1)
            if trial_ids.shape[1] > max_len:
                trial_ids = trial_ids[:, -max_len:]

            # Forward pass
            logits = self.model(trial_ids)  # (1, <=max_len, V)
            probs = F.softmax(logits[:, -1, :], dim=-1)  # (1, V)

            # Risk: expected negative log probability under preferences
            risk = compute_risk(
                predictive_probs=probs,
                preference_mode=self.preference_mode,
                preferences=None,
                targets=None,
            )
            risks.append(risk.squeeze())

            # Ambiguity AND epistemic share a single BALD pass: ambiguity is
            # the mean predictive entropy E_q(z)[H[p(o|z)]], epistemic is the
            # mutual information I(z;o) = H[p_bar] - mean_H. Pre-fix the
            # ambiguity term was computed as H[q_marginal] which exactly
            # cancels risk in 'uniform' preference mode (sum = log V).
            beliefs = self.model.prior_bank.encode(trial_ids)
            beliefs = beliefs._replace(
                phi=self.model.pos_enc(beliefs.phi, trial_ids.shape[1])
            )
            ep_mi, mean_H = self._compute_epistemic_value(beliefs.mu, beliefs.sigma)
            ambiguities.append(mean_H)
            if self.epistemic_weight > 0:
                epistemics.append(ep_mi)
            else:
                _dtype = risks[0].dtype if risks else torch.float32
                epistemics.append(torch.tensor(0.0, device=device, dtype=_dtype))

        risk_tensor = torch.stack(risks)       # (C,)
        ambiguity_tensor = torch.stack(ambiguities)  # (C,)
        epistemic_tensor = torch.stack(epistemics)  # (C,)
        efe = risk_tensor + ambiguity_tensor - self.epistemic_weight * epistemic_tensor

        return {
            'efe': efe,
            'risk': risk_tensor,
            'ambiguity': ambiguity_tensor,
            'epistemic': epistemic_tensor,
        }

    @torch.no_grad()
    def select_action(
        self,
        context_ids: torch.Tensor,
        top_k: int = 50,
        temperature: float = 1.0,
    ) -> int:
        r"""Select next token via EFE-weighted sampling.

        Produces top-k candidates from the model's own distribution,
        then re-scores them by EFE and samples from
        :math:`q(a) \propto \exp(-\gamma \, G(a))`.

        Args:
            context_ids: ``(1, N)`` context token IDs.
            top_k: Number of candidates to consider.
            temperature: Sampling temperature applied to EFE scores.

        Returns:
            Selected token ID (int).
        """
        # Get top-k candidates from model's own distribution
        logits = self.model(context_ids)  # (1, N, V)
        last_logits = logits[:, -1, :]  # (1, V)
        top_k = min(top_k, last_logits.shape[-1])
        _, top_ids = torch.topk(last_logits, top_k, dim=-1)  # (1, top_k)
        candidate_ids = top_ids.squeeze(0)  # (top_k,)

        # Score by EFE
        scores = self.score_candidates(context_ids, candidate_ids)
        efe = scores['efe']  # (top_k,)

        # Sample from Gibbs distribution
        log_probs = -self.gamma * efe / temperature
        probs = F.softmax(log_probs, dim=-1)
        idx = torch.multinomial(probs, num_samples=1).item()

        return candidate_ids[idx].item()
