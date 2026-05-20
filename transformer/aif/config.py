"""
AIFConfig: dataclass for the canonical active-inference policy machinery.

Defaults reduce bitwise to the existing depth-1 EFE path in
`transformer/vfe/efe.py:VFEExpectedFreeEnergy` so the new module ships
without behavioral surprise. Demo-preset defaults for the click-to-run
`train_aif.py` are documented in that file and require a one-line override.

The class is a dataclass with explicit field-level documentation and a
`__post_init__` validator that catches infeasible combinations early (see
`05_verifier_report.md` §8.7 and `04_compute_feasibility.md` §7).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from transformer.vfe.config import VFEConfig


@dataclass
class AIFConfig:
    r"""Configuration for `AIFGenerator` policy machinery.

    Field groupings match the manuscript decomposition:

    - **Planning horizon** controls policy length D and tree branching.
    - **EFE weights** control the relative weights of risk, ambiguity, and
      epistemic terms (the canonical decomposition per
      [ParrPezzuloFriston2022 Ch. 2]).
    - **Preferences** selects :math:`p^*(o | C)` from
      `transformer/aif/preferences.py`. Default is the empirical training-
      data marginal so the pragmatic value reduces to the cross-entropy of
      the model's predictive against the corpus marginal.
    - **Habit prior** is uniform under this build-out; the field is reserved
      for a learned or constructed habit prior in a research extension.
    - **Caching** budgets the belief-state cache used for prefix sharing
      across sibling policies in the tree.
    - **Training-time EFE** is Phase 1 disabled; setting
      ``training_objective='efe_augmented'`` raises ``NotImplementedError``.

    The ``__post_init__`` validator enforces the full-covariance guard
    (verifier §8.7): canonical tree search at depth > 1 is intractable for
    full-cov sigma because the attention grid cost scales as
    :math:`O(B N^2 K^2)`.
    """

    # === Planning horizon ===
    horizon_D: int = 1
    """Depth of policy lookahead. ``horizon_D=1`` reduces bitwise to the
    existing depth-1 EFE path; depth > 1 enables tree search."""

    beam_width: int = 16
    """Branching factor at each tree node. At ``horizon_D=1`` this is the
    number of top candidates scored at the root."""

    branching_strategy: Literal['beam', 'top_k', 'sophisticated'] = 'beam'
    """Tree expansion strategy. All three use the same top-`beam_width`
    candidate selection at every node; they differ only in the
    back-propagation aggregator. ``'beam'`` and ``'top_k'`` use a uniform
    child-action posterior (the mean over children's V — equivalent to
    the :math:`\\gamma \\to 0` limit of sophisticated inference).
    ``'sophisticated'`` implements the canonical
    [Friston2021SophisticatedInference] recursion with a softmax-weighted
    child posterior :math:`q(a' \\mid s) \\propto \\exp(-\\gamma\\, V(a'))`."""

    # === EFE weights and sampling ===
    gamma: float = 1.0
    r"""Policy precision. Used in two places: (i) the root-level action
    posterior :math:`q(\pi) \propto \exp(-\gamma G(\pi))` consumed by
    :meth:`AIFGenerator._commit_action` under
    ``sampling_strategy='multinomial'``, and (ii) the recursive
    child-action posterior :math:`q(a' \mid s) \propto \exp(-\gamma V(a'))`
    inside the sophisticated-inference back-propagation per
    [Friston2021SophisticatedInference]. Fixed at construction;
    Gamma-hyperprior inference deferred."""

    decode_tau: float = 1.0
    """Softmax temperature for the PriorBank decode used by AIF scoring.
    Independent of `VFEConfig.decode_tau` so AIF generation can be tuned
    separately from the training-time decode."""

    epistemic_samples: int = 4
    """Monte Carlo samples for the BALD MI estimate. At ``S=4`` the standard
    error on MI is roughly 50% per `04_compute_feasibility.md` §5; raise
    to 16 for low-noise scoring at the cost of S× more decode calls."""

    epistemic_weight: float = 1.0
    """Multiplier on the BALD MI term. The canonical decomposition has
    weight 1.0 (epistemic value enters with a unit coefficient and is
    subtracted from G). Allowing the user to dial this down corresponds
    to a goal-pursuit-vs-exploration tradeoff, not the canonical form."""

    discount: float = 1.0
    """Geometric discount applied to child G values during multi-step
    aggregation. Engineering hyperparameter, not canonical EFE quantity
    (per verifier §7.5)."""

    # === Preferences ===
    preference_type: Literal['empirical_marginal', 'low_entropy', 'task_conditioned'] = 'empirical_marginal'
    """Selects the `Preference` subclass. ``'empirical_marginal'`` requires
    `preference_path`. ``'low_entropy'`` is the strict self-evidencing
    surrogate `p* = exp(-β H[p(o|s)])`. ``'task_conditioned'`` accepts an
    arbitrary `(V,)` log-probability tensor at construction."""

    preference_path: Optional[str] = None
    """Filesystem path to a `(V,)` torch tensor of log preferences. Required
    when ``preference_type == 'empirical_marginal'`` or ``'task_conditioned'``."""

    low_entropy_beta: float = 1.0
    """Inverse-temperature for `LowEntropyPreference`. Only consulted when
    ``preference_type == 'low_entropy'``."""

    # === Habit prior ===
    habit_prior_path: Optional[str] = None
    r"""Filesystem path to a habit-prior tensor. ``None`` means uniform
    habit (the canonical posterior reduces to :math:`q(\pi) \propto \exp(-\gamma G)`)."""

    # === Sampling at the root ===
    sampling_strategy: Literal['argmin', 'multinomial'] = 'multinomial'
    """How to pick the committed action from the root policy posterior.
    ``'argmin'`` is the limit ``gamma -> inf`` (deterministic best-G).
    ``'multinomial'`` samples from `q(a) = softmax(-gamma * G)`."""

    sampling_temperature: float = 1.0
    """Temperature applied to the policy posterior at the root. Multiplies
    ``gamma`` for sampling but does not affect the recursive aggregation
    of G at internal nodes."""

    # === Caching ===
    belief_cache_max_entries: int = 4096
    """LRU cap on the prefix-keyed belief-state cache. At 125 KB per
    snapshot (K=20, N=128) the default cap holds ~500 MB of cached
    states — well above the 72 MB budget for the recommended demo
    preset (D=2, b=4)."""

    # === Training-time EFE ===
    training_objective: Literal['standard_vfe', 'efe_augmented'] = 'standard_vfe'
    """``'standard_vfe'`` is generation-only AIF; training stays cross-
    entropy. ``'efe_augmented'`` is the research extension that adds
    EFE-style loss to the training objective; raises
    ``NotImplementedError`` in Phase 1."""

    def __post_init__(self) -> None:
        if self.horizon_D < 1:
            raise ValueError(
                f"AIFConfig.horizon_D must be >= 1 (got {self.horizon_D}); "
                "depth 0 is the no-policy degenerate case."
            )
        if self.beam_width < 1:
            raise ValueError(
                f"AIFConfig.beam_width must be >= 1 (got {self.beam_width})."
            )
        if self.epistemic_samples < 1:
            raise ValueError(
                f"AIFConfig.epistemic_samples must be >= 1 (got {self.epistemic_samples})."
            )
        if self.gamma <= 0.0:
            raise ValueError(
                f"AIFConfig.gamma must be positive (got {self.gamma}); "
                "non-positive precision inverts the policy posterior."
            )
        if self.preference_type in ('empirical_marginal', 'task_conditioned') \
                and self.preference_path is None:
            raise ValueError(
                f"AIFConfig.preference_path is required when preference_type "
                f"is {self.preference_type!r} (path to a (V,) log-probability "
                "tensor on disk)."
            )
        if self.training_objective == 'efe_augmented':
            raise NotImplementedError(
                "AIFConfig.training_objective='efe_augmented' is deferred to "
                "Phase 4 of the build-out (see "
                "docs/plans/2026-05-19-aif-transformer-buildout/06_plan.md "
                "§6). Phase 1 ships generation-only AIF; training stays "
                "standard variational F."
            )

    def validate_against_model(self, vfe_cfg: 'VFEConfig') -> None:
        r"""Validate the AIF config against the wrapped `VFEModel`'s config.

        Enforces verifier §8.7: full-covariance Σ at depth > 1 is
        intractable because the attention grid cost scales as
        :math:`O(B N^2 K^2)` (see `04_compute_feasibility.md` §7).
        """
        if self.horizon_D > 1 and not vfe_cfg.diagonal_covariance:
            raise ValueError(
                f"AIFConfig.horizon_D={self.horizon_D} > 1 requires the "
                "wrapped VFEModel to use diagonal_covariance=True. Full-cov "
                "tree search blows up at K^2 per attention pair per node; "
                "see docs/plans/2026-05-19-aif-transformer-buildout/"
                "04_compute_feasibility.md §7."
            )
