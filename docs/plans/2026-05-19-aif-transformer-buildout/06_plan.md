# Plan — Canonical Active-Inference Transformer Build-Out

**Date:** 2026-05-19
**Verifier verdict:** YELLOW — math canonical, code references verified, no CLAUDE.md hard-constraint violations. Patches from `05_verifier_report.md` §8 are folded into this plan as first-tier work items.
**Origin:** Verdict at `docs/debates/2026-05-19-vfe-active-inference-impl/04_verdict.md` (RED_WINS) established that the current `transformer/vfe/active_inference.py` is a self-evidencing surrogate, not canonical active inference. The user's intuition that "Friston's policies = future strings of tokens" is exactly the `[ParrPezzuloFriston2022 Ch. 7]` discrete-state-space formulation and is the correct mapping for a transformer LM.

## 1. Goal

Build a separate `transformer/aif/` package implementing canonical active inference per `[ParrPezzuloFriston2022]`, `[FristonEtAl2017]`, and `[Friston2021SophisticatedInference]` (full citation in §10). Generation-time policy selection uses Expected Free Energy `G(π)` over horizon-D token sequences. Training-time stays standard variational F minimization — the renamed self-evidencing regularizer (verdict action item 1) is the existing path and remains opt-in. The build-out preserves all CLAUDE.md hard constraints (gauge equivariance, no neural networks beyond PriorBank decode, no CLI args, click-to-run config dicts, E-step blindness) and remains bitwise-equivalent to the current /vfe behavior when every AIF toggle is off.

## 2. Mapping — generative model to transformer LM

(Full derivation in `01_canonical_efe_for_lm.md`; this section restates the operative identifications.)

- State `s_t` = belief tuple `(μ_t, σ_t, φ_t)` at position t, as defined in `transformer/core/types.py:11`.
- Observation `o_t` = the token at position t, drawn from vocabulary V.
- Action `a_t ∈ V` = a choice of next token to emit. In an autoregressive LM, action and next observation are typographically the same object; the build-out separates them at the math level via `q(o_{t+1} | a_t)` (the model's predictive at the belief that follows committing to `a_t`).
- Policy `π = (a_{T+1}, ..., a_{T+D})` = an ordered sequence of token-emission actions over horizon D.
- Likelihood `p(o_t | s_t)` = `VFEPriorBank.decode(μ_t, σ_t)` softmax-over-V output, per `transformer/vfe/prior_bank.py:427`.
- Transition `p(s_{t+1} | s_t, a_t)` = deterministic. Committing to token `a_t` extends the context; re-running `VFEModel.forward` on the extended context computes the next belief state. Per `02_codebase_audit.md` §5 the safe path is to re-run the full E-step at each tree node rather than attempt incremental belief updates.
- Preferences `p*(o | C)` = the canonical default is the empirical training-data marginal (verifier §8.5 clarified: this gives the cross-entropy `H(q(o|a), freq_train)` of the model's predictive against the corpus marginal, NOT the standard CE loss against the realized next token; the two coincide only in the uninformative-model limit). Configurable alternatives: low-entropy (self-evidencing strict) and task-conditioned (RLHF-style preference distribution).

The canonical posterior is `q(π) ∝ exp(-γ G(π)) · E_prior(π)`, with the habit prior `E_prior(π)` defaulting to uniform under this build-out (verifier §8.9). Policy precision γ is fixed at construction; the Gamma-hyperprior γ-inference of `[FristonEtAl2017]` §3 is a research extension (verifier §8.10).

## 3. EFE decomposition for one policy step

Per `01_canonical_efe_for_lm.md` eq (11) (canonical Form-3 BALD decomposition):

```
G(π) = E_{q(o|π)}[ -log p*(o|C) ]    # pragmatic value (expected cost under preferences)
     + E_{q(s|π)}[ H[p(o|s)] ]       # ambiguity (mean predictive entropy over sampled states)
     - I_{q(s,o|π)}( s ; o ).        # epistemic value (BALD mutual information)
```

Sign convention: G is energy-to-minimize. The policy posterior softmins `-γG`. Ambiguity is the canonical mean-predictive-entropy form (matches `transformer/vfe/efe.py:78-95`, not the deprecated marginal-entropy form at `transformer/core/expected_free_energy.py:112-137` which carries a non-canonical disclosure).

Multi-step EFE per `[Friston2021SophisticatedInference]`: at depth d the recursive form is
```
G_d(π_{1:d}) = local_pragmatic_d + local_ambiguity_d - local_epistemic_d
             + γ_disc · E_{q(a_{d+1} | s_d)}[ G_{d+1}(π_{1:d}, a_{d+1}) ],
```
with the policy posterior at the child node weighting the recursion. The build-out defaults to a beam-search approximation that replaces the inner expectation with `sum_{a in beam} G_{d+1}(...)`; the full sophisticated-inference recursion is exposed as a configurable.

## 4. Module layout

```
transformer/aif/
  __init__.py
  config.py            # AIFConfig dataclass
  preferences.py       # Preference ABC + EmpiricalMarginal / LowEntropy / TaskConditioned
  policy.py            # PolicyNode (frozen dataclass with action_seq, depth, parent, G components)
  belief_cache.py      # BeliefStateCache (prefix-keyed; per-commit pruning + LRU)
  efe_score.py         # compute_G_at_node + helpers (delegates to existing risk/ambiguity/MI)
  tree_search.py       # Beam expansion + sophisticated-inference recursion variants
  generator.py         # AIFGenerator: full sophisticated-inference generation
  train_aif.py         # Click-to-run entry point; demo config + checkpoint load + generate
```

`AIFGenerator` composes a `VFEModel` (does not subclass). The wrapped model is constructed by the existing `/vfe` package; the AIF module never modifies model parameters at generation time. No `trainer.py` ships in Phase 1 — EFE-augmented training is explicitly deferred (§7).

## 5. AIFConfig — defaults and validation

```python
@dataclass
class AIFConfig:
    # === Planning horizon ===
    horizon_D:           int = 1            # depth of policy lookahead; D=1 reduces bitwise to existing vfe/efe.py
    beam_width:          int = 16           # branching factor at each tree node
    branching_strategy:  Literal['beam', 'top_k', 'sophisticated'] = 'beam'

    # === EFE weights and sampling ===
    gamma:               float = 1.0        # policy precision (fixed; Gamma-prior inference deferred)
    decode_tau:          float = 1.0        # PriorBank.decode temperature inside the AIF readout
    epistemic_samples:   int = 4            # MC samples for BALD MI estimate
    epistemic_weight:    float = 0.5        # multiplier on the BALD term (informational gain dial)
    discount:            float = 1.0        # geometric discount across tree depth (engineering hyperparameter)

    # === Preferences ===
    preference_type:     Literal['empirical_marginal', 'low_entropy', 'task_conditioned'] = 'empirical_marginal'
    preference_path:     Optional[str] = None  # required for 'empirical_marginal' or 'task_conditioned'

    # === Habit prior ===
    habit_prior_path:    Optional[str] = None  # None = uniform habit prior over policies

    # === Caching ===
    belief_cache_max_entries: int = 4096

    # === Training-time EFE ===
    training_objective:  Literal['standard_vfe', 'efe_augmented'] = 'standard_vfe'
    # 'efe_augmented' raises NotImplementedError in __post_init__ for Phase 1.
```

Validation rules (`__post_init__`):
1. `horizon_D >= 1`.
2. `beam_width >= 1`.
3. `branching_strategy == 'sophisticated' implies horizon_D >= 2` (no-op otherwise).
4. `preference_type == 'empirical_marginal'` requires `preference_path` to point to a `(V,)` log-probability tensor on disk.
5. **Full-cov guard (verifier §8.7):** when wrapping a `VFEModel`, validate `model.cfg.diagonal_covariance is True` whenever `horizon_D > 1`; raise `ValueError` otherwise. Full-cov tree search is intractable per `04_compute_feasibility.md` §7 ¶2.
6. `training_objective == 'efe_augmented'` raises `NotImplementedError` in Phase 1.

**Defaults pick the conservative anchor over the demo preset.** `horizon_D = 1, beam_width = 16` matches the existing `vfe/efe.py:VFEExpectedFreeEnergy` depth-1 behavior bitwise — turning `active_inference=False` in the wrapped VFEConfig and instantiating the AIFGenerator with these defaults must reproduce existing `VFEModel.generate(use_efe=True)` output. The `train_aif.py` demo header documents the alternative `horizon_D=2, beam_width=4` as a one-line edit for users wanting to exercise the tree search.

## 6. Phased implementation

### Phase 0 — Verdict action items from the prior debate

Before any new code, close the action items from `docs/debates/2026-05-19-vfe-active-inference-impl/05_action.md`:

1. Rename `transformer/vfe/active_inference.py:VFEActiveInference` → `VFESelfEvidencingRegularizer`. Rename the config flags: `active_inference` → `self_evidencing_regularizer`; `pragmatic_weight` → `self_confidence_weight`. Retain one-release deprecation aliases in `VFEConfig.__post_init__`.
2. Restrict user-facing strings in `transformer/vfe/train_vfe.py:104-108` comments. Update the verify-script `scripts/verify_active_inference.py` accordingly.
3. Add the manuscript appendix to `Attention/GL(K)_supplementary.tex` documenting the surrogate and labeling it research-track.
4. Add the finite-difference + sign + dark-room tests against the surrogate path.
5. Add the research-track note to `CLAUDE.md` alongside the existing RoPE × MahalanobisNorm exception.

### Phase 1 — Canonical depth-1 EFE (reduction anchor)

**Goal:** ship `transformer/aif/` with `AIFGenerator` configured at the defaults, behaving identically to the existing `vfe/efe.py` depth-1 path under `horizon_D=1`. This phase has zero new behavior; it is the scaffolding the rest of the build sits on.

**Tasks:**

1. **Add `VFEModel.forward_with_beliefs` accessor (verifier §8.8).** New optional method that returns `(logits, beliefs)` where `beliefs` is the converged `BeliefState` at the model output. The default `forward` continues to return `(logits, loss, ce_for_log)` to preserve the existing training loop. The new accessor closes the double-encode pattern at `transformer/vfe/efe.py:191-196` (Agent 4 §2 step 2 flagged this; Phase 1 must not silently inherit it).

2. **Build the module skeleton.** Create empty files under `transformer/aif/`. Add `__init__.py` exports for `AIFConfig`, `AIFGenerator`, `Preference`, `EmpiricalMarginalPreference`, `LowEntropyPreference`, `TaskConditionedPreference`.

3. **Implement `AIFConfig`** per §5, including the full-cov runtime guard (verifier §8.7).

4. **Implement `Preference` base class and the three subclasses.** Empirical marginal pre-computes `log_pref: torch.Tensor` of shape `(V,)` at construction from a tokenized training corpus. Low-entropy returns `-β · H[p(o|s)]` at evaluation time. Task-conditioned takes an arbitrary `(V,)` log-probability tensor at construction.

5. **Implement `PolicyNode` and `BeliefStateCache`.** `PolicyNode` is a frozen dataclass with `action_seq: Tuple[int, ...], depth: int, parent: Optional['PolicyNode'], G_pragmatic, G_ambiguity, G_epistemic, G_local, G_cum` scalar fields, plus a `belief_cache_key: bytes = tuple(action_seq).__hash__()` field. `BeliefStateCache` is a dict-backed LRU with `evict_below_depth(min_depth: int)` for per-commit pruning. Memory budget per `04_compute_feasibility.md` §4: 125 KB per snapshot at K=20, N=128; 4096 entries × 125 KB = 500 MB cap.

6. **Implement `efe_score.compute_G_at_node(node, model, preference, cache, cfg) -> EFEComponents`.** Delegates pragmatic / ambiguity / BALD MI to existing primitives: pragmatic uses `transformer/core/expected_free_energy.py:compute_risk`, ambiguity and BALD MI use the mean-predictive-entropy path verified in `transformer/vfe/efe.py:78-95`. Reuses the converged belief from `forward_with_beliefs` rather than re-encoding (closes the double-encode bug). Signature returns the four scalars individually so the diagnostic logging can decompose them.

7. **Implement `AIFGenerator.generate(prompt_ids, max_new_tokens) -> torch.Tensor`.** At `horizon_D=1` the control flow reduces to: for each step, expand the root by the top-`beam_width` candidates from the predictive distribution, score each with `compute_G_at_node`, softmin `-γ G` over the candidates, sample or argmax, commit, prune the cache. At `horizon_D=1` this matches `VFEExpectedFreeEnergy.select_action` semantics exactly.

8. **Implement the decode-prior cache hoist (verifier §8.6).** Add a `frozen_decode_prior: bool = False` parameter to `VFEModel.forward` (or to a new context manager) that suppresses the `prior_bank.invalidate_cache()` call at `transformer/vfe/model.py:156`. `AIFGenerator.generate` wraps the tree-search loop in this context so the `(V, K)` prior tensor is computed once per generation step rather than per tree node. Per Agent 4 §7 ¶3 this saves 2.3 GB of redundant compute at D=3, b=8.

9. **Add Phase-1 tests** (corresponding to `02_codebase_audit.md` §6 + `03_architecture_design.md` §10):
   - Reduction invariant: `AIFGenerator(horizon_D=1, beam_width=k).generate(prompt)` matches `VFEModel.generate(prompt, use_efe=True, top_k=k)` token-for-token.
   - Argmin invariant: with `gamma → ∞` the AIF generator emits the EFE-argmin token at every step.
   - Cache invariance: belief at a re-visited prefix matches a fresh forward pass on the same prefix.
   - Sign test: under the empirical-marginal preference, doubling `freq_train(v)` for some token v strictly reduces `G(π_with_v)` relative to `G(π_without_v)`, all else equal.
   - Law-1 regression: `AIFGenerator.generate(...)` never instantiates a non-None `targets` argument to any forward pass.
   - Style: banned-phrase grep across `transformer/aif/` returns zero hits.

**Exit criterion:** Phase-1 tests pass; `AIFGenerator(default_config).generate(prompt)` produces the same output as `VFEModel.generate(prompt, use_efe=True)` on a fixed seed; wall-clock per token within 1.2× of the existing depth-1 path.

### Phase 2 — Tree search at depth D > 1

**Goal:** enable canonical multi-step EFE with the beam approximation. Default config flips to `horizon_D=2, beam_width=4` (demo preset) for the click-to-run example; the conservative `horizon_D=1, beam_width=16` anchor remains as the test reduction target.

**Tasks:**

1. **Implement `tree_search.beam_expand(root, model, preference, cache, cfg) -> List[PolicyNode]`.** Build a depth-`D` tree by iteratively expanding the top-`beam_width` children at each node according to the predictive distribution; score each leaf with `compute_G_at_node`; back-propagate cumulative discounted G to root children.

2. **Wire `AIFGenerator` to `tree_search.beam_expand` when `horizon_D > 1`.** The depth-1 path stays the existing simple-scoring loop.

3. **Cache hit-rate diagnostics.** Add a one-line counter `BeliefStateCache.hits, BeliefStateCache.misses` that surfaces in `train_aif.py` output to confirm prefix sharing is working (Agent 4 §4 ¶5 budget assumes prefix sharing).

4. **Phase-2 tests:**
   - Depth-2 reduction: `beam_width=1, horizon_D=2` matches a hand-computed two-step EFE on a tiny toy LM (B=1, V=8, K=4).
   - Branching reduction: `beam_width=V` matches a brute-force depth-D enumeration on the same toy LM (within numerical tolerance).
   - Cache hit rate: under `horizon_D=2, beam_width=4` the cache reports ≥ 80% hit rate on the second commit step (siblings of the previous commit share the prefix belief).
   - Memory: peak GPU memory under the demo preset (D=2, b=4, K=20, N=128, B=1) stays under 8 GB (matches Agent 4 §6 estimate ±50%).

**Exit criterion:** Phase-2 tests pass; demo preset runs in under 5 seconds per generated token on a 24 GB consumer GPU (matches Agent 4 §6 wall-clock estimate).

### Phase 3 — Sophisticated-inference recursion

**Goal:** expose the canonical Friston-2021 recursive G evaluation as a configurable.

**Tasks:**

1. **Implement `tree_search.sophisticated_expand`** following `[Friston2021SophisticatedInference]` eqs in the cited paper (citation per verifier §8.1). At each internal node, compute the child action posterior `q(a | s) ∝ exp(-γ G_child(a))` and weight the recursion by it, rather than the flat policy-sum the beam approximation uses.

2. **Wire `branching_strategy='sophisticated'`** in `AIFGenerator`.

3. **Phase-3 tests:**
   - Recursion reduction: at `horizon_D=1` sophisticated recursion matches the beam approximation bitwise (the recursion has no internal nodes).
   - Action-posterior agreement: at depth 2 the sophisticated child posterior matches a direct softmax computation on a hand-computed toy LM.
   - Cost sanity: sophisticated recursion at (D=2, b=4) is within 2× the beam wall-clock at the same setting.

**Exit criterion:** Phase-3 tests pass; both branching strategies produce sensible outputs on a small held-out validation set (lower validation NLL than greedy under the empirical-marginal preference).

### Phase 4 — Optional EFE-augmented training (research extension)

**Goal:** investigate whether training-time EFE produces measurable gains over standard VFE training. Phase 4 is research-track and only proceeds if Phases 1-3 are stable.

**Tasks (gated by user approval):**

1. Treat the observed training trajectory as the chosen policy. Add a per-batch EFE-style policy loss `L_AIF = γ_train · G(π_observed)` to the standard CE objective.
2. Run a controlled ablation: validation perplexity under `(training_objective='standard_vfe')` vs `('efe_augmented')` at a fixed step budget.
3. Decide go / no-go based on the ablation. Agent 4 §9 estimates 1% overhead for the trajectory-only path (Regime A), 200-500× overhead for tree-expanded training-time EFE (Regime B, infeasible). Phase 4 commits to Regime A only.

**Exit criterion:** ablation results gated by user review. If validation perplexity does not improve or improves by less than the run-to-run noise, Phase 4 stays disabled and the build-out concludes with Phases 0-3.

## 7. Citation patches (from verifier §8)

Add to `.claude/agents/vfe-knowledge/external_bibliography.md`:

- `[Friston2021SophisticatedInference]` Friston, K., Da Costa, L., Hafner, D., Hesp, C., Parr, T. (2021). "Sophisticated Inference." *Neural Computation* 33(3): 713–763. arXiv:2006.04120.
- `[HoulsbyEtAl2011]` Houlsby, N., Huszár, F., Ghahramani, Z., Lengyel, M. (2011). "Bayesian Active Learning for Classification and Preference Learning." arXiv:1112.5745.
- `[Friston2012Darkroom]` Friston, K., Thornton, C., Clark, A. (2012). "Free-energy minimization and the dark-room problem." *Frontiers in Psychology* 3: 130.

The Hohwy 2016 self-evidencing reference and any RLHF preference reference remain `[full citation needed]` in `01_canonical_efe_for_lm.md` §6; before this build-out reaches manuscript stage, those should be resolved similarly.

## 8. Style and writing patches

- `04_compute_feasibility.md` line 220 contains the banned word `critically` (verifier §1.5, §8.4). Replace with `severely` / `acutely` / `expensively`.
- `01_canonical_efe_for_lm.md` §4 ¶4 second sentence overstates the NLL identification (verifier §6.6, §8.5). Rewrite to distinguish `H(q(o|a), freq_train)` (the empirical-marginal-preference pragmatic) from `-log q(y|a)` (the standard CE loss at the realized token); the two coincide only under `q(o|a) = freq_train(o)`.
- `01_canonical_efe_for_lm.md` §2 add a sentence after eq (14): the canonical policy posterior is `q(π) ∝ exp(-γG(π)) · E_prior(π)` with uniform habit prior assumed throughout the build-out (verifier §7.1, §8.9).
- `01_canonical_efe_for_lm.md` §2 or §5 add a paragraph noting γ is held fixed at construction; Gamma-hyperprior γ-inference deferred (verifier §7.2, §8.10).
- `01_canonical_efe_for_lm.md` §3 ¶3 add a sentence noting the depth-D discount factor is an engineering hyperparameter, not canonical (verifier §7.5).
- `01_canonical_efe_for_lm.md` §1 add one-line acknowledgement that the canonical EFE novelty term is zero by construction at generation time when the M-step parameters are frozen (verifier §7.4).

## 9. Interaction matrix with existing /vfe

The four toggle states behave as follows (verifier §5.5 reduction-invariance audit confirmed):

|                                   | `AIFGenerator` not used                   | `AIFGenerator` used                          |
|-----------------------------------|-------------------------------------------|----------------------------------------------|
| `self_evidencing_regularizer=False` | **Pure path.** Bitwise unchanged /vfe.    | Canonical AIF at generation; pure /vfe inside the wrapped `VFEModel`. |
| `self_evidencing_regularizer=True`  | Research-track surrogate at E-step; standard generate. | Two augmentations stack — the E-step uses the surrogate, generation uses canonical AIF. Documented as research-only; not the default. |

The top-left cell is the "pure path under appropriate toggles" CLAUDE.md invariant. The top-right cell is the recommended build-out target.

## 10. Risks and mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| Full-cov × tree search × depth > 1 is intractable. | High | Hard runtime guard in `AIFConfig.__post_init__` (§5 validation rule 5). |
| Decode-prior recomputation per tree node burns 2.3 GB at D=3, b=8. | High | `frozen_decode_prior` context manager around the tree-search loop (Phase 1 task 8). |
| Double-encode in `compute_G_at_node` perpetuates the existing `vfe/efe.py:191-196` bug. | Medium | `VFEModel.forward_with_beliefs` accessor (Phase 1 task 1). |
| BALD MI at S=4 produces noisy scoring. | Medium | Surface `epistemic_samples` in `AIFConfig`; document the 50% standard error at S=4; recommend S=16 for low-noise scoring (Agent 4 §5). |
| Sophisticated-inference recursion explodes in cost at deep horizons. | Medium | Beam approximation remains the default; SI recursion behind `branching_strategy='sophisticated'`. |
| User expects "active inference" naming on the surrogate path. | Low | Verdict action item 1 renames the surrogate to `self_evidencing_regularizer`; AIF naming is reserved for the canonical module. |
| Existing tests pass while AIF integration silently breaks. | Medium | Reduction-invariant tests at every Phase exit; Phase 1 includes the bitwise reduction to existing `vfe/efe.py` behavior. |

## 11. Non-goals

- **Differentiable policy rollouts** — the build-out does not require gradients to flow backward through the tree search. Backward is for training only (standard CE on the realized token); generation is `torch.no_grad()`.
- **Beam continuation across token commits** — each generation step starts with a fresh tree rooted at the current belief. Cross-step beam persistence is a future extension.
- **Per-token learned γ-precision** — γ is fixed at config-time. Gamma-hyperprior inference is deferred.
- **MCTS or nucleus sampling at the expansion step** — beam and sophisticated recursion are the two strategies in scope. Other strategies are future extensions.
- **Cross-language / cross-modality preferences** — `p*(o|C)` lives over the model's own vocabulary V.
- **Manuscript derivation of the augmented F functional** — `Attention/GL(K)_attention.tex` `\label{eq:free_energy_functional_final}` remains the five-term canonical F. The AIF module's G functional is a *separate* generation-time objective, not an augmentation of F.

## 12. Open questions

1. **Habit prior parameterization.** The build-out defaults to uniform. If the user wants a learned or constructed habit prior, what is the data source? An n-gram model over training data? A frozen "common token" distribution? Defer to user input before Phase 1 starts if the empirical-marginal default is insufficient.
2. **Validation metric for AIF quality.** Validation perplexity is the obvious metric but rewards exactly the depth-1 model when the preference is the empirical marginal. A better falsification metric: held-out generative-output evaluation against a separate reward model or perplexity under a stronger reference LM. Defer the metric choice to Phase 3 exit criteria discussion.
3. **Hohwy 2016 self-evidencing citation** for the surrogate-path manuscript appendix (verdict action item 3). Pending bibliography fill.

## 13. Estimated effort

- Phase 0 (rename + verdict items): ~1-2 sessions. Mechanical refactor; verdict has already produced the action list.
- Phase 1 (canonical depth-1 module + reduction anchor): ~3-5 sessions. The reduction-invariant test is the load-bearing exit criterion.
- Phase 2 (tree search): ~3-4 sessions. The cache hit-rate test is the load-bearing exit criterion.
- Phase 3 (sophisticated inference): ~2-3 sessions. Pure addition once Phase 2 lands.
- Phase 4 (optional EFE-augmented training): gated by user approval; ~2-3 sessions if proceeded with.

Total: ~11-17 working sessions for the full build-out through Phase 3. Phase 4 is gated.

## 14. First action on user approval

If the user approves this plan, the next action is to read `docs/debates/2026-05-19-vfe-active-inference-impl/05_action.md` and execute Phase 0 (the verdict's existing action items) before any new module work. This is sequential: Phase 0 closes the diagnosis from the prior debate, Phase 1 builds the canonical replacement. Skipping Phase 0 would leave a research-track surrogate named "active inference" in user-facing config while the canonical implementation also claims the name — a labeling collision the verdict already identified.
