"""
Preference distributions :math:`p^*(o | C)` for canonical Expected Free Energy.

The pragmatic term in EFE is :math:`E_{q(o|\\pi)}[-\\log p^*(o | C)]`
where :math:`p^*` is the agent's preferred outcome distribution under
goal context :math:`C` [ParrPezzuloFriston2022 Ch. 2]. For a transformer
LM without an exogenous goal there is no natural :math:`p^*`; this module
provides three principled defaults:

- :class:`EmpiricalMarginalPreference`: :math:`p^*(o) = \\mathrm{freq_{train}}(o)`.
  Pragmatic value reduces to :math:`H(q(o|a), \\mathrm{freq_{train}})`,
  the cross-entropy of the model's predictive against the corpus
  marginal. This is the canonical default for the build-out — it ties
  the AIF objective to the empirical data distribution without requiring
  an external preference.
- :class:`LowEntropyPreference`: :math:`p^* \\propto \\exp(-\\beta H[p(o|s_*)])`.
  Strict self-evidencing per [Hohwy2016] (full citation pending in
  bibliography). Reproduces the surrogate behaviour that the deleted
  `transformer/vfe/active_inference.py` implemented at the E-step.
  Available here as a configurable preference at the policy level
  (generation-time only, never folded into the E-step).
- :class:`TaskConditionedPreference`: :math:`p^*` supplied externally as a
  ``(V,)`` log-probability tensor. RLHF-style preference for goal-directed
  generation.

All subclasses expose ``log_pref(predictive_probs) -> Tensor`` with
shape ``(K,)`` returning the pragmatic value (expected -log preference)
per candidate row of ``predictive_probs``.
"""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Optional

import torch


_EPS: float = 1e-12


class Preference(abc.ABC):
    r"""Abstract base class for preference distributions :math:`p^*(o | C)`.

    Concrete subclasses implement ``pragmatic`` which computes
    :math:`E_{q(o|a)}[-\log p^*(o | C)]` per row of the predictive.
    """

    @abc.abstractmethod
    def pragmatic(self, predictive_probs: torch.Tensor) -> torch.Tensor:
        r"""Compute the expected negative log-preference under each predictive row.

        Args:
            predictive_probs: ``(K, V)`` predictive distribution per candidate
                action.

        Returns:
            ``(K,)`` pragmatic value per candidate.
        """
        raise NotImplementedError


class EmpiricalMarginalPreference(Preference):
    r"""Empirical training-data marginal :math:`p^*(o) = \mathrm{freq_{train}}(o)`.

    Loads a ``(V,)`` log-probability tensor at construction. Pragmatic value
    becomes the cross-entropy of the model's predictive against the corpus
    marginal: :math:`-\sum_v q(v|a) \log \mathrm{freq_{train}}(v)`.

    This is the canonical default for the build-out per
    ``01_canonical_efe_for_lm.md`` §4: it grounds the pragmatic term in
    the data distribution without requiring an external goal, and closes
    the dark-room failure mode that pure self-evidencing exposes.
    """

    def __init__(self, log_pref: torch.Tensor) -> None:
        if log_pref.dim() != 1:
            raise ValueError(
                f"EmpiricalMarginalPreference.log_pref must be 1-D (V,); "
                f"got shape {tuple(log_pref.shape)}."
            )
        self.log_pref = log_pref

    @classmethod
    def from_path(cls, path: str) -> 'EmpiricalMarginalPreference':
        r"""Load a precomputed ``(V,)`` log-frequency tensor from disk."""
        log_pref = torch.load(Path(path), map_location='cpu', weights_only=True)
        return cls(log_pref)

    def pragmatic(self, predictive_probs: torch.Tensor) -> torch.Tensor:
        log_pref = self.log_pref.to(
            device=predictive_probs.device, dtype=predictive_probs.dtype,
        )
        return -(predictive_probs * log_pref).sum(dim=-1)


class LowEntropyPreference(Preference):
    r"""Self-evidencing surrogate :math:`p^*(o) \propto \exp(-\beta H[p(o|s)])`.

    Reproduces the substitution the deleted `transformer/vfe/active_inference.py`
    used at the E-step. Available here as a configurable preference at the
    policy level — at generation time only, never folded into the E-step
    (Law 1 / E-step blindness is preserved).

    Computes ``-beta * H[p(o|a)]`` as the pragmatic value; the proportionality
    constant cancels under the policy posterior softmin.
    """

    def __init__(self, beta: float = 1.0) -> None:
        if beta <= 0.0:
            raise ValueError(
                f"LowEntropyPreference.beta must be positive (got {beta})."
            )
        self.beta = beta

    def pragmatic(self, predictive_probs: torch.Tensor) -> torch.Tensor:
        log_probs = predictive_probs.clamp(min=_EPS).log()
        entropy = -(predictive_probs * log_probs).sum(dim=-1)
        return -self.beta * entropy


class TaskConditionedPreference(Preference):
    r"""External preference distribution :math:`p^*(o | C)` supplied as a tensor.

    Loads a ``(V,)`` log-probability tensor from disk; computes the
    cross-entropy `-sum_v q(v|a) log p^*(v|C)`. Identical signature to
    `EmpiricalMarginalPreference` but flagged as task-conditioned for
    clarity at the call site (a learned reward model, an RLHF preference
    output, or a hand-specified goal distribution).
    """

    def __init__(self, log_pref: torch.Tensor) -> None:
        if log_pref.dim() != 1:
            raise ValueError(
                f"TaskConditionedPreference.log_pref must be 1-D (V,); "
                f"got shape {tuple(log_pref.shape)}."
            )
        self.log_pref = log_pref

    @classmethod
    def from_path(cls, path: str) -> 'TaskConditionedPreference':
        log_pref = torch.load(Path(path), map_location='cpu', weights_only=True)
        return cls(log_pref)

    def pragmatic(self, predictive_probs: torch.Tensor) -> torch.Tensor:
        log_pref = self.log_pref.to(
            device=predictive_probs.device, dtype=predictive_probs.dtype,
        )
        return -(predictive_probs * log_pref).sum(dim=-1)


def build_preference(
    preference_type: str,
    preference_path: Optional[str] = None,
    low_entropy_beta: float = 1.0,
) -> Preference:
    r"""Factory: dispatch on ``preference_type`` and return a `Preference`.

    Used by `AIFGenerator.__init__` to convert the string config field into
    the concrete subclass instance. The user passes the canonical type
    string (no class-name or case-insensitive aliasing per the project's
    no-lenient-config-aliases convention).
    """
    if preference_type == 'empirical_marginal':
        if preference_path is None:
            raise ValueError(
                "build_preference('empirical_marginal') requires "
                "preference_path pointing to a (V,) log-probability tensor."
            )
        return EmpiricalMarginalPreference.from_path(preference_path)
    if preference_type == 'low_entropy':
        return LowEntropyPreference(beta=low_entropy_beta)
    if preference_type == 'task_conditioned':
        if preference_path is None:
            raise ValueError(
                "build_preference('task_conditioned') requires preference_path."
            )
        return TaskConditionedPreference.from_path(preference_path)
    raise ValueError(
        f"build_preference: unknown preference_type={preference_type!r}. "
        "Accepted: 'empirical_marginal', 'low_entropy', 'task_conditioned'."
    )
