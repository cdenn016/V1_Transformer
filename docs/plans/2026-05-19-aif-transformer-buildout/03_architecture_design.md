# Architecture Design — Canonical Active-Inference Transformer

**Agent:** 3 / 4 (architecture)
**Date:** 2026-05-19
**Scope:** Module layout, data structures, and control flow for a canonical active-inference generator that treats Friston-style policies as future token sequences. Math and citations belong to Agent 1 (`01_canonical_efe_for_lm.md`); compute and memory belong to Agent 4 (`04_compute_feasibility.md`); reuse audit belongs to Agent 2 (`02_codebase_audit.md`). This document specifies file boundaries, class signatures, configuration, and reduction invariants only.

The defining design decision is that active inference enters at generation time as a tree-search policy posterior over future token sequences, not as an E-step augmentation. The training loop of `transformer/vfe/train_vfe.py` stays as-is. The new module `transformer/aif/` is a parallel, opt-in generator that wraps an already-trained `VFEModel`.

The architecture must satisfy three reduction invariants that double as correctness anchors. First, with the new generator disabled, the existing `VFEModel.generate(use_efe=False)` path must be bitwise unchanged. Second, with `horizon_D = 1` and the default beam strategy, the new generator must reproduce `transformer/vfe/efe.py::VFEExpectedFreeEnergy.select_action` (`efe.py:215-252`) up to the same numerical noise the existing one-step path already incurs. Third, with `gamma -> infinity`, the generator must return `argmin_a G(a)` over root children. These are the unit tests in §10.


## §1. Module Layout

The package lives at `transformer/aif/`, parallel to `transformer/vfe/`. The choice to put it outside `transformer/vfe/` rather than inside is deliberate: `vfe/` owns variational inference of the past (the E-step over observed context), `aif/` owns expected-free-energy planning of the future (policy posterior over candidate sequences). The two operate on disjoint timelines and the manuscript treats them as separate sections; the file boundary should mirror that.

```
transformer/aif/
    __init__.py          # Public surface: AIFConfig, AIFGenerator, Preference subclasses
    config.py            # AIFConfig dataclass (matches VFEConfig conventions)
    preferences.py       # Preference base class + EmpiricalMarginalPreference,
                         #     LowEntropyPreference, TaskConditionedPreference
    policy.py            # PolicyNode (tree node, immutable action sequence + score)
    belief_cache.py      # BeliefStateCache (prefix-keyed lookup of cached beliefs)
    tree_search.py       # Beam / top-k / sophisticated-inference expansion
    efe_score.py         # Per-node G computation: pragmatic + ambiguity - epistemic
    generator.py         # AIFGenerator: orchestrates expand -> score -> commit
    train_aif.py         # Click-to-run entry point with config dict (mirrors train_vfe.py)
```

The package deliberately does NOT include a `trainer.py`. Recommended training is unchanged from `train_vfe.py`; the §5 discussion below settles this. A future option-B trainer would land here as `trainer.py` if the empirical case is ever made, but it is not part of the build-out specified here.

Per-file ownership. `config.py` holds the dataclass with knob defaults and `__post_init__` validation. `preferences.py` houses every `p*(o | C)` implementation behind a common `log_pref` interface; the generator never sees a concrete preference class, only the abstract base. `policy.py` is a pure data container; no torch ops live in it. `belief_cache.py` is the prefix cache that lets sibling tree nodes share computation on the path back to the prompt root. `tree_search.py` implements the expansion strategy as a strategy-pattern dispatch keyed on `AIFConfig.branching_strategy`, so swapping beam for MCTS does not touch `generator.py`. `efe_score.py` exposes a single `compute_G_at_node` function that consumes a `PolicyNode` and a `Preference` and returns the three EFE component scalars plus their sum. `generator.py` is the top-level loop: it owns the tree, the cache, and the per-token commit step. `train_aif.py` is the click-to-run example that builds a small `VFEModel`, loads a checkpoint, and runs `AIFGenerator.generate` on a prompt.


## §2. Data Structures

The tree machinery rests on four types: `PolicyNode`, `BeliefStateCache`, `Preference` (abstract), and `AIFConfig`. Each is laid out below in the form a Python source file would carry, with the constraint that pseudocode signatures are type-annotated but bodies are left as prose.

### `PolicyNode`

A `PolicyNode` represents one partial token sequence under consideration. Friston's discrete policy `pi = (a_{T+1}, ..., a_{T+D})` (Parr-Pezzulo-Friston 2022, Ch. 7) maps onto a path from the root to a node at depth D, with cumulative score `G(pi)` aggregated along the way. The node carries the action history that produced it, the EFE components for the action taken to reach it, a pointer to the cached belief state at that node, and a parent pointer for tree backtracking.

```python
@dataclass(frozen=True)
class PolicyNode:
    """One node in the policy-posterior search tree.

    Each node corresponds to a partial action sequence
    ``a_{T+1}, ..., a_{T+d}`` of depth ``d``. The root has depth 0 and
    no action. The cumulative score ``G_cum`` is the sum of per-step
    EFE values along the path from root to this node, with optional
    geometric discount ``gamma_discount ** d``.

    All scores are scalars on the model's device.
    """
    action_seq:        Tuple[int, ...]               # full path actions from root
    depth:             int                            # len(action_seq); 0 at root
    parent:            Optional[PolicyNode]           # None at root
    pragmatic_step:    torch.Tensor                   # () scalar, this-step pragmatic
    ambiguity_step:    torch.Tensor                   # () scalar, this-step ambiguity
    epistemic_step:    torch.Tensor                   # () scalar, this-step epistemic
    G_step:            torch.Tensor                   # () pragmatic + ambiguity - epistemic
    G_cum:             torch.Tensor                   # () running sum (or discounted sum)
    belief_key:        BeliefCacheKey                 # opaque hash into BeliefStateCache
```

`PolicyNode` is frozen because tree-search code passes nodes between expansion and scoring layers; immutability removes a class of bugs where a child node mutates a parent's score during back-propagation. The `belief_key` is an opaque `Tuple[int, ...]` that encodes the prompt suffix plus the action sequence; lookups go through `BeliefStateCache`.

### `BeliefStateCache`

The cache stores `(mu, sigma, phi)` keyed by the token-sequence suffix that produced it. Two sibling nodes that share a prefix share a cache entry for the parent's belief; only the extension forward pass differs. Without this, an order-`B^D` tree pays `O(B^D)` forward passes, but each pass would also redo the prefix; with the cache, the marginal cost per node is one forward step rather than an end-to-end forward over the full path.

```python
class BeliefStateCache:
    """Prefix-keyed cache of (mu, sigma, phi) belief states.

    Keyed by ``Tuple[int, ...]`` of the full token-id sequence (prompt
    concatenated with the action prefix that produced the belief).
    Values are detached belief tensors on the model's device.

    Lifetime: one cache per ``AIFGenerator.generate`` invocation.
    Memory budget is bounded by ``AIFConfig.belief_cache_max_entries``;
    on overflow, LRU eviction is used.

    The cache is invariant under re-execution: re-running the model's
    E-step on the same key MUST produce the same belief tensors up to
    numerical tolerance. The §10 test plan asserts this directly.
    """

    def __init__(self, max_entries: int) -> None: ...

    def get(self, key: BeliefCacheKey) -> Optional[BeliefState]: ...

    def put(self, key: BeliefCacheKey, belief: BeliefState) -> None: ...

    def evict_below_depth(self, min_depth: int) -> None:
        """Drop entries shallower than ``min_depth`` after committing a token."""
```

### `Preference` (abstract base + concrete subclasses)

The preference distribution `p*(o | C)` is the only point at which the agent's goals enter the EFE expression. The verdict's critique of the substitution `p* <- p_pred` (verdict §4) makes this the single most theoretically load-bearing knob. The abstract base exposes a `log_pref` method that takes either a token id or a categorical distribution and returns a log probability under the preference; concrete subclasses differ only in how `log p*(v)` is computed.

```python
class Preference(ABC):
    r"""Abstract preference distribution :math:`p^*(o \mid C)`.

    All subclasses MUST be stateless after construction: they may cache
    a precomputed log-probability vector but must not depend on the
    agent's current belief or on the search tree state.
    """

    @abstractmethod
    def log_pref(self, token_ids: torch.Tensor) -> torch.Tensor:
        r"""Return :math:`\log p^*(o \mid C)` evaluated at the given tokens.

        Args:
            token_ids: ``(K,)`` candidate token ids.

        Returns:
            ``(K,)`` tensor of log preferences.
        """
        ...

    def expected_cost(self, predictive_probs: torch.Tensor) -> torch.Tensor:
        r"""Expected negative log preference under predictive distribution.

        :math:`\mathbb{E}_{q(o \mid \pi)}[-\log p^*(o \mid C)]`.

        Args:
            predictive_probs: ``(V,)`` predicted token distribution.

        Returns:
            ``()`` scalar expected cost (the pragmatic term in G).
        """
        ...
```

Three concrete subclasses live in `preferences.py`. `EmpiricalMarginalPreference` precomputes `log p*(v) = log f(v)` from the training-set unigram frequencies; this is the default and §6 defends it. `LowEntropyPreference` constructs `p*(v) propto exp(-beta H[v])` where the exponent is a per-token confidence shaping bias; the implementation precomputes the normalized distribution at construction. `TaskConditionedPreference` accepts a `(V,)` log-probability vector at construction (the caller is responsible for producing it from a task description, e.g., RLHF reward model marginals); the class is purely a wrapper.

### `AIFConfig`

The config dataclass mirrors `VFEConfig` conventions: short field names, `# === Section ===` separators, type hints on every field, and a `__post_init__` that validates combinations. Defaults are chosen so that constructing `AIFConfig()` with no arguments and passing it to `AIFGenerator` produces behavior identical to `VFEModel.generate(use_efe=True)` at the existing one-step depth. The full field list is enumerated in §7 below; this section sketches only the structure.

```python
@dataclass
class AIFConfig:
    r"""Configuration for the active-inference generator.

    Defaults reduce to the existing depth-1 EFE path in
    ``transformer/vfe/efe.py``. The active-inference machinery becomes
    visible only when ``horizon_D > 1`` and ``branching_strategy`` is set.
    """
    # === Tree structure ===
    horizon_D:           int  = 1
    beam_width:          int  = 16
    branching_strategy:  Literal['beam', 'top_k', 'sophisticated'] = 'beam'
    # ... (see §7 for the full field list)
```


## §3. EFE Computation per Policy Node

The per-node EFE score is the canonical Parr-Pezzulo-Friston decomposition specialized to a language-model policy. For a partial policy `pi_<=d` ending at depth `d` with action sequence `a_{T+1:T+d}`, the running G is the discounted sum of per-step scores

```
G(pi_<=d) = sum_{k=1}^{d} discount^{k-1} * [pragmatic_k + ambiguity_k - epistemic_k]
```

with each per-step component evaluated using the model's belief at depth `k-1` and the action taken to reach depth `k`. The function `compute_G_at_node` returns the three components separately so the caller (the tree expansion in `tree_search.py`) can aggregate them on the path back to the root.

```python
def compute_G_at_node(
    node:              PolicyNode,
    model:             'VFEModel',
    preference:        Preference,
    cache:             BeliefStateCache,
    epistemic_samples: int,
    decode_tau:        float,
) -> EFEComponents:
    r"""Compute one-step EFE at a policy tree node.

    Given a node at depth ``d`` whose action sequence is ``a_{T+1:T+d}``,
    this evaluates :math:`G_d` for the single transition from depth
    ``d-1`` to depth ``d``:

    .. math::
        G_d = \mathbb{E}_{q(o \mid \pi)}[-\log p^*(o \mid C)]
            + \mathbb{E}_{q(z \mid \pi)}[H[p(o \mid z)]]
            - I_q(z; o \mid \pi).

    The first term is the pragmatic value (cost under preferences), the
    second is ambiguity (expected predictive entropy under latent
    samples), and the third is epistemic value (BALD mutual information).

    The node's belief state is retrieved from ``cache`` if present;
    otherwise the model is run on the cached parent belief extended by
    the new action and the result is stored back in ``cache``.

    Args:
        node: The leaf node to score.
        model: The trained ``VFEModel`` providing forward + decode.
        preference: The preference distribution :math:`p^*(o \mid C)`.
        cache: Prefix-keyed belief cache.
        epistemic_samples: Number of latent samples for BALD MI.
        decode_tau: Softmax temperature for ``prior_bank.decode``.

    Returns:
        ``EFEComponents(pragmatic, ambiguity, epistemic, G)`` scalars.
    """
```

The control flow is straightforward. The function first retrieves the parent belief from `cache`; if missing, it recursively recomputes by running `model.forward` on the parent's full token sequence (the recursion bottoms out at the root, whose belief was precomputed once at the start of generation). It then constructs the candidate context by appending the node's last action to the parent's sequence, runs `model.forward` to obtain the predictive distribution `q(o | pi)`, and stores the resulting belief in `cache` under the node's key.

The pragmatic term is `preference.expected_cost(predictive_probs)`. For `EmpiricalMarginalPreference`, this reduces to the cross-entropy between the predictive distribution and the training unigram, which is bounded and well-behaved. The ambiguity term is the mean over `epistemic_samples` latent draws `z_s ~ N(mu, Sigma)` of the predictive entropy `H[p(o | z_s)]`; the existing implementation pattern in `transformer/vfe/efe.py::_compute_epistemic_value` (lines 73-134) computes this and returns both the ambiguity and the BALD MI from one sampling pass, and the new code reuses the same kernel. The epistemic term is the BALD MI itself: `H[p_bar] - mean_s H[p_s]` with `p_bar = (1/S) sum_s p(o | z_s)`; the existing code at `efe.py:131-134` is the reference.

For nodes with `depth < horizon_D`, the tree search recurses on children before returning the final discounted G; this is handled in `tree_search.py` rather than inside `compute_G_at_node`, which is strictly a per-node operation. The separation is deliberate so the strategy-pattern dispatch in `tree_search.py` controls the recursion shape without `efe_score.py` needing to know about beam widths or strategy choices.


## §4. Generation Loop

The generator owns four resources for the duration of a `generate` call: the prompt prefix, the belief cache, the root node, and the tree-expansion strategy object. Each emitted token triggers one tree expansion to depth `horizon_D`, one back-propagation of cumulative G to the root's children, and one sample (or argmax) from the policy posterior over those children.

```python
class AIFGenerator:
    r"""Sophisticated active-inference generator.

    Wraps a trained :class:`VFEModel` and replaces its standard
    autoregressive ``generate`` with a tree-search policy posterior.
    The model is not modified.

    Args:
        model: A trained ``VFEModel`` (eval mode).
        aif_cfg: The :class:`AIFConfig` controlling tree shape and EFE.
        preference: The preference distribution :math:`p^*(o \mid C)`.
    """

    def __init__(
        self,
        model:       'VFEModel',
        aif_cfg:     AIFConfig,
        preference:  Preference,
    ) -> None: ...

    @torch.no_grad()
    def generate(
        self,
        prompt_ids:      torch.Tensor,
        max_new_tokens:  int,
    ) -> torch.Tensor:
        r"""Generate token by token under EFE-weighted policy selection.

        For each new token position the generator: (i) seeds the tree
        root with the current belief, (ii) expands to depth
        ``horizon_D`` using the configured branching strategy,
        (iii) scores every leaf via :func:`compute_G_at_node` and
        back-propagates the cumulative discounted G to the root's
        children, (iv) forms ``q(a) propto exp(-gamma * G_root(a))`` over
        the root's children, (v) samples or argmaxes an action, and
        (vi) commits the action, advancing the prompt and pruning the
        cache.

        Args:
            prompt_ids: ``(1, N)`` or ``(N,)`` initial context.
            max_new_tokens: Number of tokens to emit.

        Returns:
            ``(1, N + max_new_tokens)`` full sequence.
        """
```

Per-token control flow.

```
ids <- prompt_ids
cache <- BeliefStateCache(aif_cfg.belief_cache_max_entries)
seed_root_belief(cache, model, ids)              # one forward pass on the prompt
for step in range(max_new_tokens):
    root <- PolicyNode(action_seq=(), depth=0, parent=None,
                      G_step=0, G_cum=0, belief_key=key(ids))
    leaves <- tree_search.expand(
        root, model, preference, cache, aif_cfg,
    )                                            # branching to depth aif_cfg.horizon_D
    score_leaves(leaves, model, preference, cache, aif_cfg)
    back_propagate_to_root(leaves, root, discount=aif_cfg.discount)
    children <- root.children                    # depth-1 candidates
    q_pi <- softmax(-aif_cfg.gamma * stack(c.G_cum for c in children))
    if aif_cfg.gamma == infinity or aif_cfg.greedy:
        idx <- argmin(c.G_cum for c in children)
    else:
        idx <- multinomial(q_pi)
    a_t <- children[idx].action_seq[0]
    ids <- concat(ids, [a_t])
    cache.evict_below_depth(min_depth=1)         # prune now-orphaned branches
return ids
```

The commit step (the `cache.evict_below_depth` call) is what makes the cache safe across iterations: only entries on the committed path stay. Cache invariance under re-execution is asserted in §10.

Two implementation notes. First, the generator's `generate` is wrapped in `torch.no_grad()` because no training gradient flows through tree search; this also lets the cache hold detached tensors. Second, when `aif_cfg.horizon_D == 1` and `aif_cfg.branching_strategy == 'beam'`, the loop above degenerates exactly to the depth-1 EFE path in `transformer/vfe/efe.py:215-252`. The §10 invariant test pins this.


## §5. Training-Time Use of EFE

Recommendation: **Option A — EFE generation-only.** Training stays as it is in `transformer/vfe/train_vfe.py` (lines 25-152). The CE objective remains the M-step driver; the renamed `self_evidencing_regularizer` from verdict action item 1 remains available as an opt-in E-step shaping. Active inference enters exclusively at generation time through `AIFGenerator`.

The defense rests on three considerations. First, the verdict at `04_verdict.md:9-25` documents that the prior attempt to inject EFE into the E-step substituted `p* <- p_pred` and was struck down as non-canonical. Re-inserting an EFE-augmented loss into training risks repeating the same substitution at a different layer of the stack — the training-data sequence is not, in general, the agent's preferred future, and treating the observed sequence as the chosen policy presupposes a generative-model identification (training-data marginal = preferences) that the canon §10 pitfall list (Parr-Pezzulo-Friston 2022 Ch. 2) explicitly warns against overclaiming. Second, compute cost: every training step would multiply by `beam_width * horizon_D`, a regime where the existing optimizer settings (M-step LR ratios, cosine schedule with `warmup_steps=100`) are not validated. Third, separation of concerns: VFE training of the past and AIF planning of the future are distinct activities under the canon (`external_canon_inference.md` §2), and conflating them is the dark-room failure mode the verdict already flagged.

Option B (EFE-augmented training, where the loss is `CE + lambda_AIF * G(pi_observed)`) is enumerated as a future research-track extension but not built. The dataclass field `AIFConfig.training_objective` defaults to `'generation_only'` and the only other accepted value, `'efe_augmented'`, raises `NotImplementedError` from `__post_init__`. If a future ablation pushes for option B, it lands as `transformer/aif/trainer.py` with its own `AIFTrainer` class; the slot is enumerated in §1 but the file is not created in this build-out.


## §6. Default Preference Distribution

Agent 1's `01_canonical_efe_for_lm.md` has not yet landed; this section makes the architecture-side recommendation without depending on it and flags the cross-reference for the verifier.

Recommendation: **`EmpiricalMarginalPreference` as the default**, with `log p*(v) = log f(v)` where `f(v)` is the training-set unigram frequency vector (smoothed by add-one to avoid `log 0`). Under this preference the pragmatic value reduces to

```
E_{q(o | pi)}[-log p*(o)] = -sum_v q(v | pi) log f(v) = H(q(o | pi), f) = CE(q(o | pi), f),
```

which is the cross-entropy between the predictive distribution and the training unigram. This is a bounded, well-defined scalar with a clear interpretation: it penalizes the model for predicting tokens that the training distribution considers rare. It does NOT collapse onto the model's own prediction (the verdict-struck substitution); the preference is defined externally to the agent at construction time and held fixed across all generation calls. The choice grounds the EFE pragmatic term in observed data rather than in the agent's beliefs, which is the structural property the verdict's critique demanded.

The hierarchy in §2 admits three subclasses: `EmpiricalMarginalPreference` (default), `LowEntropyPreference` (per-token confidence shaping), and `TaskConditionedPreference` (externally supplied log-probability vector for RLHF-style goal conditioning). All three implement the same `log_pref` interface; the generator is preference-agnostic. The user selects via `AIFConfig.preference_type ∈ {'empirical_marginal', 'low_entropy', 'task_conditioned'}`. A future fourth subclass for delta-target preferences (the `compute_risk` `'target'` mode in `transformer/core/expected_free_energy.py:86-92`) is admissible but not built; it would be a `DeltaTargetPreference` for diagnostic teacher-forced evaluation only.


## §7. Configuration Knobs

Full `AIFConfig` field listing, in source-file order, with type and effect.

```python
@dataclass
class AIFConfig:
    # === Tree structure ===
    horizon_D:              int  = 1
        # Planning depth. 0 = score current-belief next-token distribution only
        # (matches VFEModel.generate(use_efe=False) up to one extra forward pass).
        # 1 = canonical depth-1 EFE (matches efe.py::VFEExpectedFreeEnergy).
        # >=2 = sophisticated active inference.
    beam_width:             int  = 16
        # Number of children kept at each tree level under beam strategy.
        # At depth_D=1 the value reads as the top-k cap for the candidate set.
    branching_strategy:     Literal['beam', 'top_k', 'sophisticated'] = 'beam'
        # Tree expansion strategy. See §8.

    # === EFE scoring ===
    gamma:                  float = 1.0
        # Inverse temperature of the policy posterior q(pi) propto exp(-gamma * G).
        # gamma -> inf reduces to argmin G; gamma = 0 reduces to uniform.
    discount:               float = 1.0
        # Per-step geometric discount on running G. 1.0 = undiscounted sum
        # (default; matches the canon when no terminal-value bootstrap is used).
    epistemic_weight:       float = 0.5
        # Multiplier on the BALD MI term. Matches VFEConfig.epistemic_weight default.
    epistemic_samples:      int   = 4
        # MC samples for the BALD pass. Matches VFEConfig.epistemic_samples.
    decode_tau:             float = 1.0
        # Softmax temperature for prior_bank.decode in the EFE rollout.
        # Matches VFEConfig.decode_tau.

    # === Preferences ===
    preference_type:        Literal['empirical_marginal', 'low_entropy',
                                    'task_conditioned'] = 'empirical_marginal'
        # Selects the concrete Preference subclass. See §6.
    preference_smoothing:   float = 1.0
        # Add-one (or alpha) smoothing for empirical_marginal. Avoids log(0).
    low_entropy_beta:       float = 1.0
        # Exponent for low_entropy preference. Ignored for other types.
    task_conditioned_log_p: Optional[torch.Tensor] = None
        # (V,) externally supplied log-probabilities for task_conditioned mode.

    # === Sampling ===
    greedy:                 bool  = False
        # If True, argmin G over root children instead of multinomial sampling.
    temperature:            float = 1.0
        # Additional temperature on the policy posterior (q ~ exp(-gamma * G / T)).

    # === Caching ===
    belief_cache_max_entries: int = 4096
        # LRU cap on BeliefStateCache. Order ~ beam_width**horizon_D plus the
        # path back to root. See Agent 4 for memory budget guidance.

    # === Training (gate, not yet implemented) ===
    training_objective:     Literal['generation_only', 'efe_augmented'] = 'generation_only'
        # 'efe_augmented' is reserved for the option-B research track. The
        # __post_init__ raises NotImplementedError on the latter; see §5.

    # === Diagnostics ===
    log_per_step_components: bool = False
        # If True, AIFGenerator emits pragmatic/ambiguity/epistemic per
        # committed token to the standard logger.
```

`__post_init__` validation. It enforces `horizon_D >= 0` (negative values are rejected). It enforces `beam_width >= 1` (a width of zero would produce an empty tree). It rejects `branching_strategy='sophisticated'` combined with `horizon_D < 2` (the strategy is meaningful only beyond depth 1; below that it collapses to beam). It rejects `training_objective='efe_augmented'` with `NotImplementedError` and the message `Option B (EFE-augmented training) is enumerated in 03_architecture_design.md §5 but not implemented; see verdict 04_verdict.md`. It also rejects `gamma < 0`, `discount` outside `[0, 1]`, and `epistemic_samples < 1`.


## §8. Sophisticated Inference vs Beam Search

Default: **beam search** at every depth, with width `beam_width` and `top_k` candidates per parent drawn from the model's own predictive distribution at that node. This is the existing one-step pattern in `transformer/vfe/efe.py:215-252` extended to depth `D` by recursion. It has bounded compute `O(beam_width^D)` per generated token, reduces to greedy at `beam_width=1`, reduces to the existing depth-1 EFE path at `horizon_D=1`, and admits a clean cache-reuse pattern because sibling nodes share a parent belief.

The alternative is sophisticated inference in the sense of `[Friston2021SophisticatedInference]` — the title is "Sophisticated Inference," arXiv:2006.04120, Friston, Da Costa, Hafner, Hesp & Parr 2021 — where each node's score is the expected free energy taken under the posterior over the agent's future beliefs and not merely over a sampled trajectory. The published version is in *Neural Computation* 33(3):713-763 (2021); this paper is NOT in `external_bibliography.md` and Agent 1 should add it.

The cost/quality tradeoff. Sophisticated inference is theoretically the correct full-depth recursion: G at depth `d` depends on what the agent will believe after observing the action at depth `d-1`, which in turn changes its EFE estimates for depth `d+1`. Beam search collapses this to a point estimate of beliefs at each node. The empirical evidence in Friston 2021 §3-4 is that the recursion matters most when the planning horizon is long and the predictive distribution is sharply peaked — the regime where beam pruning loses information by committing to a wrong branch early. For language modeling at `horizon_D <= 4` and `beam_width >= 8`, the gain is plausibly small relative to the compute cost, but this is empirical and not settled.

The architecture treats the choice as a strategy-pattern dispatch in `tree_search.py`. Adding the sophisticated-inference strategy is a localized change (one new function in that file plus one new accepted value of `branching_strategy`), so the build-out lands beam first and sophisticated as a follow-up if the depth-D ablation shows it.


## §9. Backwards Compatibility and Migration

The new `transformer/aif/` module and the renamed `self_evidencing_regularizer` (verdict action item 1, replacing the misnamed `active_inference` flag in `VFEConfig` at `transformer/vfe/config.py:236`) are fully orthogonal. The former is generation-time tree-search policy machinery; the latter is an E-step augmentation that adds a self-confidence + BALD-MI term to the belief-update gradient. They touch disjoint code paths and disjoint configuration flags.

The 2x2 interaction matrix.

|  | `self_evidencing_regularizer = False` | `self_evidencing_regularizer = True` |
|--|--|--|
| `AIFGenerator` not used | Pure VFE path: training and generation behave as in `train_vfe.py` today with the surrogate disabled. The CLAUDE.md "theoretically pure path under appropriate toggles" invariant lives here. | Training uses the renamed self-evidencing E-step augmentation. Generation uses `VFEModel.generate(use_efe=False)`. This is the existing `active_inference=True` config under its corrected name. |
| `AIFGenerator` used (eval only) | Training is pure VFE. Generation uses canonical EFE policy posterior over future token sequences. This is the default new configuration the build-out targets. | Training uses the self-evidencing surrogate. Generation uses canonical EFE policy posterior. Both shaping pressures are active but on separate phases (training E-step vs eval tree search) and do not interact at runtime. |

No combination is forbidden. The bottom-left cell is the recommended deployment; the top-left is the recovery path that matches the pure manuscript F functional at `Attention/GL(K)_attention.tex` `\label{eq:free_energy_functional_final}`.


## §10. Test Plan

Tests fall into two layers: reduction invariants (the module must reproduce known-good behavior under specific configurations) and unit tests of the new components.

### Reduction invariants

The first invariant pins backward compatibility. With `AIFGenerator` constructed but `VFEModel.generate(use_efe=False)` called instead, the output sequence must be bit-identical to the pre-existing path. This is a test of "no spillover": importing `transformer.aif` must not perturb `transformer.vfe`.

The second invariant pins the depth-1 anchor. `AIFGenerator.generate(prompt, max_new_tokens=N)` with `aif_cfg = AIFConfig(horizon_D=1, beam_width=top_k, branching_strategy='beam', preference_type='empirical_marginal', ...)` and the same RNG state must match the sequence produced by `VFEExpectedFreeEnergy.select_action` at `transformer/vfe/efe.py:215-252` token-by-token under matching `gamma` and `epistemic_weight`. Mismatches indicate either the recursion floor or the cache is wrong.

The third invariant pins the limit behavior. With `aif_cfg.gamma -> infinity` (use a large finite value like 1e6), `AIFGenerator` must emit the argmin-G action at every step over the root's children. With `aif_cfg.horizon_D = 0`, the generator must call the same path as `VFEModel.generate(use_efe=False)`.

The fourth invariant pins cache correctness. For any prefix `p` that has been cached during a search, running `model.forward(p)` from scratch and comparing the resulting `(mu, sigma, phi)` to the cached values must agree to floating-point tolerance (e.g., `1e-5` relative). This is the `BeliefStateCache` invariance test and is the most subtle: if E-step iterations carry hidden state (random seeds, dropout, etc.), the cache will silently disagree and downstream G values will drift.

### Unit tests

`PolicyNode` immutability and `G_cum` accumulation: build a small tree by hand, verify that `G_cum` at depth `d` is `sum_{k<=d} discount^{k-1} * G_step` along the path.

`BeliefStateCache` LRU eviction: insert `max_entries + 1` keys, verify the oldest is gone. Insert and then `evict_below_depth(d)`, verify all shallower entries are removed.

`EmpiricalMarginalPreference.expected_cost` correctness: construct from a known unigram, evaluate against a known predictive distribution, compare against the hand-computed cross-entropy.

`compute_G_at_node` agreement with `compute_efe` in `transformer/core/expected_free_energy.py:303-404` for depth-1 nodes under matching configurations. This is the math-purity check: the existing depth-1 implementation is the reference.

`AIFConfig.__post_init__` rejects every invalid combination listed in §7: negative `horizon_D`, zero `beam_width`, `branching_strategy='sophisticated'` with `horizon_D < 2`, `training_objective='efe_augmented'`, and the others.

### Architectural tests not in Agent 2's enumeration

`AIFGenerator` does not import or call any training-related code. A test that asserts `transformer.aif.generator` imports do not transitively touch `VFETrainer` keeps the option-A boundary durable across refactors.

`AIFGenerator` runs under `torch.no_grad()` end to end: a test that wraps an `AIFGenerator.generate` call in `torch.is_grad_enabled() == False` checks across all per-step belief updates protects against accidental autograd graph retention through the tree.

The combined `self_evidencing_regularizer = True` and `AIFGenerator` used path (bottom-right of the §9 matrix) runs end to end on a tiny config without error, even though it carries no theoretical interaction guarantee.

This document specifies the architecture and reduction invariants. The math behind `p*(o|C)` and the EFE decomposition is Agent 1's lane (`01_canonical_efe_for_lm.md`); the compute and memory budgets for `belief_cache_max_entries` and the deep-horizon regime are Agent 4's lane (`04_compute_feasibility.md`). The verifier (`05_verifier_report.md`) should cross-check that the strategy-pattern boundary in §1, the reduction invariants in §10, and the §9 orthogonality matrix all hold against the actual code paths in `transformer/vfe/efe.py` and `transformer/core/expected_free_energy.py`.
