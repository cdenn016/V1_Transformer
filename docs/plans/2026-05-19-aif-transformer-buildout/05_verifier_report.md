# Verifier Report — Canonical Active-Inference Transformer Build-Out

**Date:** 2026-05-19
**Role:** Verifier across Agents 1–4.
**Inputs verified:** `00_context.md`, `01_canonical_efe_for_lm.md`, `02_codebase_audit.md`, `03_architecture_design.md`, `04_compute_feasibility.md`, and `docs/debates/2026-05-19-vfe-active-inference-impl/04_verdict.md`. Cross-referenced against `.claude/agents/vfe-knowledge/external_canon_inference.md`, `.claude/agents/vfe-knowledge/external_bibliography.md`, `.claude/agents/vfe-knowledge/style_constraints.md`, and `CLAUDE.md`.

This report cross-checks math against canon, code references against the actual files, architecture against math, cost against architecture, and CLAUDE.md hard constraints against the proposed build. Every claim about an investigator carries a `<doc>:§<section>` citation. Every claim about the codebase carries a verified `path:line` reference.


## §1. Math Consistency — Agent 1 vs Canon

### 1.1 EFE decomposition

Agent 1 §2 writes the policy-conditioned EFE in Form 1 at eq (9):
```
G(π) = E_{q(o, s | π)} [ log q(s | π) - log p(o, s | π) ].
```
Agent 1's Form-3 rearrangement at eq (10) is
```
G(π) = E_{q(o | π)}[ -log p^*(o | C) ]
     - E_{q(o | π)}[ KL( q(s | o, π) || q(s | π) ) ].
```
The first term is labeled pragmatic value, the second epistemic value, with the minus sign in front of the KL flagged as the canonical convention because G is energy-to-minimize.

The canon `external_canon_inference.md` §2 contains an internal sign inconsistency. Its first display equation writes
```
G(π) = -E_{q(o|π)}[ log p(o|C) ] + E_{q(o|π)}[ KL( q(s|o,π) ‖ q(s|π) ) ]
```
(epistemic added), but its second display equation immediately below writes
```
G(π) = E_q[ -log p(o|C) ] - E_{q(o|π)}[ KL( q(s|o,π) ‖ q(s|π) ) ]
     = expected cost - expected information gain
```
(epistemic subtracted). The second form matches the textbook treatment in `[ParrPezzuloFriston2022 §2.4]` and Agent 1's eq (10). The first form in canon §2 is a typo in the canon source; Agent 1's choice is canonical. Agent 1 §2 ¶3 internally fixes the issue by stating "the minus sign in front of the KL is the canonical sign convention" and matching the second canon line.

The BALD rearrangement at Agent 1 eq (11),
```
G(π) = E_{q(o|π)}[ -log p^*(o|C) ]
     + E_{q(s|π)}[ H[p(o|s)] ]    # ambiguity
     - I_{q(s,o|π)}(s ; o),       # BALD MI
```
uses the identity `E_{q(o|π)}[KL(q(s|o,π)||q(s|π))] = I(s; o)`. This identity is a textbook result (the chain rule on `q(s, o | π)`); Agent 1's algebra at lines 75-78 is correct.

### 1.2 Sign convention is internally consistent

Agent 1 §2 eq (14) writes `q(π) ∝ exp(-γ G(π))`. Agent 3 §4 control flow uses `q_pi <- softmax(-aif_cfg.gamma * stack(c.G_cum for c in children))`. The existing codebase at `transformer/vfe/efe.py:248` computes `log_probs = -self.gamma * efe / temperature`. All three agree: G is energy-to-minimize and the policy posterior softmins it. No sign drift between math, architecture, and existing code.

### 1.3 Empirical-marginal preference reduces to NLL

Agent 1 §4 eq (19) sets `p*(o = v) = freq_train(v)`. Agent 1 §4 ¶4 claims that "when `q(o | a)` is treated as a Dirac at the observed training token `y`, the pragmatic term reduces exactly to the NLL `-log freq_train(y)`."

The algebra. The pragmatic term is `E_{q(o|a)}[-log p*(o)] = -sum_v q(v|a) log freq_train(v)`. If `q(o|a) = δ_{o=y}` (deterministic at the observed token), this collapses to `-log freq_train(y)`, which is the NLL of `y` under the empirical-marginal preference. This is **not** the model's own NLL — that would be `-log q(y|a)` under the model's predictive `q`. Agent 1's claim that this "recovers NLL" is correct as stated for the preference's NLL, but the build-out plan must distinguish the model's predictive NLL (the standard CE objective) from the preference's empirical-marginal NLL. The two coincide only under the substitution `q(o|a) = p*(o)`, which is not the operative claim. The plan documents should not assert that the EFE pragmatic term **is** the cross-entropy loss; it is the cross-entropy of the model's predictive against the corpus marginal, which differs from the standard CE loss against the realized next token. This is a clarity nit; Agent 1 §4 ¶4 second sentence — "combined with the complexity term in equation (8), one recovers the standard variational F = NLL + KL(q || p)" — overstates the identification. Mark for §8 patch.

### 1.4 Sophisticated-inference recursion

Agent 1 §3 eqs (15)-(17) write the recursion with leaf-EFE at `d = D` and an internal-node EFE that includes the discounted expectation over the policy posterior at the child node. The recursion matches the form in the canonical Friston 2021 paper, which Agent 1 §6 leaves as `[Friston2021SophisticatedInference: full citation needed]`. Agent 3 §8 ¶2 fills in the bibliographic record:

> "Sophisticated Inference," arXiv:2006.04120, Friston, Da Costa, Hafner, Hesp & Parr 2021 ... published version is in *Neural Computation* 33(3):713-763 (2021); this paper is NOT in `external_bibliography.md` and Agent 1 should add it.

I confirmed the bibliography lacks Friston2021 by grep against `external_bibliography.md` — only Friston2010, FristonEtAl2017, Friston2017Graphical, and ParrPezzuloFriston2022 are present. The citation Agent 3 supplied is consistent with what is widely catalogued for the paper (Neural Computation 33(3):713-763, 2021). The verifier supplies it formally in §8 below.

Agent 1's recursion in eq (16) is the standard SI form: local risk-plus-ambiguity-minus-MI plus the expected child EFE under the child action posterior. Agent 1 §3 ¶3 correctly distinguishes this from the **naive policy-sum** `sum_d G_d` (eq 18), which is the depth-D extension Agent 3 §8 ¶1 defaults to under the `'beam'` branching strategy. No drift on this distinction; both documents are explicit that beam search is the approximation and full SI is the recursion.

### 1.5 Banned-phrase audit

I ran the project banned-phrase list (`key insight`, `crucially`, `critically`, `notably`, `importantly`, `it's worth noting`, `interestingly`, `fundamentally`, `in particular`, `leverages`, `underscores`) across all four investigator documents.

- `01_canonical_efe_for_lm.md`: zero hits.
- `02_codebase_audit.md`: zero hits.
- `03_architecture_design.md`: zero hits.
- `04_compute_feasibility.md`: one hit. Line 220, "More **critically**, the attention grid cost scales as O(B × N² × K²) ..." This is the banned `critically` (used here mid-sentence rather than as a sentence opener, but `style_constraints.md` lists it without a positional qualifier). Patch is a one-word substitution ("More severely" / "More acutely" / "More expensively"); not a substantive issue but flagged for the §8 patch list.


## §2. Code-Reuse Consistency — Agent 2 vs Reality

I read each path:line Agent 2 cites and confirmed the lines exist and the API matches.

### 2.1 Verified path:line claims

- `transformer/core/types.py:11` — `BeliefState` NamedTuple. Verified. Lines 11-33 show the exact fields Agent 2 quotes (`mu`, `sigma`, `phi`, `omega: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None`).
- `transformer/vfe/prior_bank.py:400` — `VFEPriorBank.encode`. Verified. Signature `def encode(self, token_ids: torch.Tensor) -> BeliefState:` at line 400. Returns `BeliefState(mu=mu_p, sigma=sigma_p, phi=phi)` at line 425.
- `transformer/vfe/prior_bank.py:321` — sandwich product `sigma_h = exp_h_f32 @ sigma_diag @ exp_h_f32.transpose(-2, -1)`. Verified bitwise at line 321. The diagonal approximation Agent 2 cites at line 326 is also verified: `(exp_h_f32 ** 2 @ base_sigma_h).clamp(min=self.eps)`.
- `transformer/vfe/prior_bank.py:427` — `VFEPriorBank.decode`. Verified at line 427. The fused diagonal-KL matmul Agent 2 cites at "498-500" is verified: `combined = torch.matmul(lhs, rhs.T)` at line 500 with the `(B, N, 2K) @ (V, 2K).T` shape claim.
- `transformer/vfe/prior_bank.py:211` — `_decode_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None`. Verified.
- `transformer/vfe/prior_bank.py:22-34` — module-level implementation note on the learnable `c = exp(decode_log_scale)` and `c = 1` initialization. Verified.
- `transformer/vfe/e_step.py:725` — `VFEEStep.forward`. Verified. The signature at lines 725-731 has **no `targets` parameter**. Law 1 docstring at line 734 confirmed: "Law 1 enforced: No `targets` parameter exists. Target leakage is structurally impossible." `sigma_p` detach at line 790 confirmed.
- `transformer/vfe/e_step.py:1006-1028` — fused block-diag attention kernel. Verified; `_fused_attention_and_vfe_gradients_block_diag(...)` invocation at line 1007 with the expected arguments.
- `transformer/vfe/stack.py:76` — `VFEStack.forward`. Verified. Signature at lines 76-82 matches Agent 2's quote. Cross-layer handoff at lines 107-133 matches Agent 2's prose summary; `rho_sigma == 0.0` defaults to frozen embedding sigma (line 120), confirming the "point-estimate handoff" claim.
- `transformer/vfe/model.py:126` — `VFEModel.forward`. Verified. Signature at lines 126-130. `targets` parameter at line 129 is consumed only at line 217 (`F.cross_entropy(...)`), confirming Law 1 / E-step blindness.
- `transformer/vfe/model.py:272` — `VFEModel.generate`. Verified at line 273 (Agent 2's "272" is one line off because the `@torch.no_grad()` decorator sits on line 272 and `def generate` starts on line 273; immaterial). The signature at lines 273-281 matches: `prompt_ids, max_new_tokens, temperature, top_k, use_efe, efe_gamma`. The `use_efe=True` branch delegates to `VFEExpectedFreeEnergy.select_action` at line 319, which is depth-1.
- `transformer/vfe/efe.py:28` — `VFEExpectedFreeEnergy` class. Verified. Depth-1 implementation confirmed: `score_candidates` enumerates `C` candidates one at a time (lines 164-194), `select_action` does top-k filtering then EFE scoring (lines 215-252). Agent 2's claim that this is depth-1 is correct — there is no recursion or multi-step rollout in the class.
- `transformer/vfe/efe.py:191-196` — second encode pass per candidate. Verified at lines 191-196: `beliefs = self.model.prior_bank.encode(trial_ids); beliefs = beliefs._replace(phi=self.model.pos_enc(beliefs.phi, trial_ids.shape[1])); ep_mi, mean_H = self._compute_epistemic_value(beliefs.mu, beliefs.sigma)`. This is a redundant encode after the full forward pass at line 174 because `VFEModel.forward` does not currently return the converged belief tuple — only the logits. Agent 4 §2 step 2 flags this as a "redundant embedding lookup ... a correctness-relevant design gap since the two passes may see different positional encoding state." Agent 2 §1.8 mentions "re-encodes the entire context on every candidate" but does not label it a bug; the bug-flag attribution belongs to **Agent 4**, not Agent 2. The 06_plan.md must fold this finding in and ensure Agent 3's `compute_G_at_node` does not perpetuate the redundancy.
- `transformer/vfe/efe.py:215-252` — `select_action`. Verified.
- `transformer/core/expected_free_energy.py:48-109` — `compute_risk` with three preference modes. Verified. Lines 86-92 (`'target'`), 94-98 (`'uniform'`), 100-106 (`'current_belief'`).
- `transformer/core/expected_free_energy.py:112-137` — `compute_ambiguity` with the non-canonical-marginal-entropy disclosure at lines 117-126. Verified bitwise.
- `transformer/core/expected_free_energy.py:140-220` — `compute_epistemic_value` with `return_mean_H` parameter. Verified at lines 140-178.
- `transformer/core/active_inference.py:28-37` — NOVEL-CONSTRUCTION DISCLOSURE. Verified bitwise. The text matches the verdict's decisive evidence at `docs/debates/2026-05-19-vfe-active-inference-impl/04_verdict.md:9-12`.
- `transformer/core/active_inference.py:94` — `_compute_active_inference_gradient`. Verified.
- `transformer/core/active_inference.py:270-345` — streaming-`p̄` accumulation. Verified at lines 269-296 (the streaming Pass 1) and the Pass 2 gradient computation at lines 299-342.
- `transformer/vfe/active_inference.py:27` — `VFEActiveInference(nn.Module)`. Verified at line 27.
- `transformer/vfe/train_vfe.py:104-108` — `active_inference, pragmatic_weight, epistemic_weight, epistemic_samples, decode_tau`. Verified.
- `tests/transformer/test_vfe_package.py:265-280` — `TestVFEActiveInference.test_callback_produces_gradients`. Verified at lines 269-280; shape-only assertion confirmed.
- `tests/transformer/test_vfe_package.py:287-308` — `TestVFEExpectedFreeEnergy`. Verified at lines 287-308.

### 2.2 No stale or wrong path:line in Agent 2's audit

No Agent 2 reference resolves to a wrong line or a vanished symbol. The off-by-one on `VFEModel.generate` (Agent 2 says line 272 for the function; the `def` is at 273 with `@torch.no_grad()` at 272) is the only nit. Every other path:line Agent 2 cites is bitwise accurate.


## §3. Architecture-vs-Math Consistency — Agent 3 vs Agent 1

### 3.1 Sign and term breakdown in `compute_G_at_node`

Agent 3 §3 writes the per-step EFE as
```
G_d = E_{q(o | π)}[-log p*(o | C)]  +  E_{q(z | π)}[H[p(o | z)]]  -  I_q(z; o | π).
```
This is identical to Agent 1's eq (11) (BALD form with mean-predictive-entropy ambiguity and BALD MI subtracted) under the LM specialization. Pragmatic added, ambiguity added, epistemic subtracted, matching the canonical Form-3 BALD decomposition. No drift.

The policy posterior in Agent 3 §4 control flow uses `softmax(-aif_cfg.gamma * stack(c.G_cum for c in children))`. Agent 1 §2 eq (14) writes `q(π) ∝ exp(-γ G(π))`. Same convention. No drift.

### 3.2 Ambiguity is mean-predictive-entropy, not marginal entropy

Agent 3 §3 ¶3 explicitly states the ambiguity term is "the mean over `epistemic_samples` latent draws ... of the predictive entropy `H[p(o | z_s)]`". This is the canonical ambiguity, matching `transformer/vfe/efe.py:78-95` (the `mean_H` return path from `_compute_epistemic_value`). It is **not** the deprecated `compute_ambiguity` at `transformer/core/expected_free_energy.py:112-137` which returns `H[q_marginal]` and is disclosed at lines 117-126 as non-canonical. Agent 3 is correctly routed through the canonical estimator.

### 3.3 Preference default

Agent 3 §6 ¶2 recommends `EmpiricalMarginalPreference` as the default. Agent 1 §4 ¶6 ("**Default for the build-out:** option one, the empirical training-data marginal") agrees. No drift on the default preference. The list of alternative subclasses also agrees: `LowEntropyPreference` (Agent 1 §4 option 2 = Agent 3 §6 ¶3), `TaskConditionedPreference` (Agent 1 §4 option 3 = Agent 3 §6 ¶3). Both documents label the existing `transformer/vfe/active_inference.py` substitution `p* <- exp(-βH)` as the `self_evidencing` / `low_entropy` configurable option, not the default.

### 3.4 Training-time use

Agent 3 §5 recommends "Option A — EFE generation-only." Agent 1 §5 (pitfalls) flags the dark-room failure mode and the substitution risk if training-time EFE is added prematurely. Agent 1 does not directly recommend option A vs option B; the pitfall list is consistent with — but does not state — Agent 3's recommendation. No contradiction, but Agent 1's text could be tightened to explicitly endorse option A; this is a §8 patch item for Agent 1.

### 3.5 BeliefStateCache consistency with Agent 4's memory budget

Agent 3 §2 `BeliefStateCache` design uses prefix-keyed lookup of detached belief tensors with LRU eviction at `belief_cache_max_entries` (default 4096 in Agent 3 §7). Agent 4 §4 ¶6 estimates 72 MB total cache for D=3, b=8, K=20, N=128 at 585 tree nodes, with prefix sharing reducing this to 20-30 MB. At 122.9 KB per snapshot (Agent 4 §4 ¶3), the Agent 3 default cap of 4096 entries × 125 KB = 500 MB — well above the 72 MB budget. The default is loose enough to never bind under the recommended (D, b) settings; this is acceptable. The two documents agree that the dominant cost component is `phi` (200 floats per token, 10× larger than `mu` or `sigma` — Agent 4 §4 ¶4), and that the cache is structurally a `BeliefState` keyed by token-sequence prefix.

One alignment gap: Agent 3 §2 belief_cache.py describes "LRU eviction"; Agent 3 §4 control flow uses `cache.evict_below_depth(min_depth=1)` after each token commit. The two strategies coexist (per-commit pruning of stale branches, LRU eviction within the active tree under memory pressure), but the API surface should document this dual mechanism. Minor; §8 patch.


## §4. Cost-vs-Architecture Consistency — Agent 4 vs Agent 3

### 4.1 Default-config drift between Agent 3 and Agent 4

Agent 3 §7 defaults `AIFConfig` to `horizon_D=1, beam_width=16, branching_strategy='beam'`. The defense at Agent 3 §1 ¶2 is that this "reduces to the existing depth-1 EFE path in `transformer/vfe/efe.py`" — a reduction-invariant anchor that costs the same as the existing depth-1 EFE generation.

Agent 4 §6 recommends a "click-to-run demo" at `horizon_depth=2, beam_width=4`, with the cost estimate "1-5 s per token on a 24 GB consumer GPU."

These are not directly contradictory. Agent 3's defaults are conservative reduction anchors that match the existing path bitwise (its first reduction invariant in Agent 3 §10 ¶2 explicitly requires this). Agent 4's recommendation is for "a click-to-run demo that exercises the canonical machinery." The plan in `06_plan.md` must pick which configuration the user-facing `train_aif.py` ships with. Soft drift, not a contradiction; flag for the planner.

### 4.2 Decode dominates — does the architecture exploit this?

Agent 4 §1 ¶4 estimates the decode at ~33 GFLOPs / 1.65 GB logits — dominant over the entire E-step. Agent 4 §7 ¶3 warns: "Per-node decode cache materialization. If `VFEPriorBank._decode_cache` is invalidated and recomputed at every tree node ... the build-out pays V × K × 4 bytes = 4 MB to recompute per-node ... Mitigation: compute the (V, K) prior once at the start of the generation step and pass it as a frozen context to all tree nodes, bypassing `invalidate_cache()`." I verified `transformer/vfe/model.py:156` calls `self.prior_bank.invalidate_cache()` at every forward, and `transformer/vfe/prior_bank.py:211` defines the cache.

Agent 3 §2 `BeliefStateCache` design does **not** address the decode-prior cache. It addresses the per-node belief cache only. The decode-prior cache reuse is a different concern (the (V, K) tensor of prior parameters used by `prior_bank.decode`) and is necessary for Agent 4's cost estimates to hold. The 06_plan.md must add a build-out item that suppresses `invalidate_cache()` within the tree-search loop or otherwise caches the decode-prior across siblings. Architectural gap, not a contradiction; flag for the planner.

### 4.3 Belief-state cache exploits prefix sharing

Agent 3 §2 `belief_cache.py` keys are token-sequence prefixes. Agent 4 §4 ¶5 explicitly relies on prefix sharing: "Nodes at depth d share the belief state of their common ancestor prefix through depth d−1 ... the first encode (depth 0, the original context) is always shared." This is consistent with Agent 3's design. Both documents agree the cache key is the full token-sequence path.

### 4.4 Full-cov guard at depth > 1

Agent 4 §7 ¶2 mandates `diagonal_covariance=True` at depth > 1, citing the K² blow-up of the attention grid cost. Agent 3 §7 does not enumerate `diagonal_covariance` as an `AIFConfig` field — it inherits the underlying `VFEModel`'s configuration. Agent 3 §7's `__post_init__` validation list does not include a guard on full-cov at depth > 1. Architectural gap that the 06_plan.md must close; either the AIF entry point reads `model.cfg.diagonal_covariance` and raises if `False` with `horizon_D > 1`, or the documentation must call this out as an undefended combination.


## §5. CLAUDE.md Hard-Constraint Audit on the Proposed Build

### 5.1 No new neural networks beyond PriorBank decode + optional output linear

Agent 3 §1 module layout enumerates: `config.py, preferences.py, policy.py, belief_cache.py, tree_search.py, efe_score.py, generator.py, train_aif.py`. Reading the descriptions in §2, §3, and §4: `PolicyNode` is a frozen dataclass with scalar tensor fields; `BeliefStateCache` is a dict-backed LRU; `Preference` subclasses precompute a `(V,)` log-probability tensor at construction; `compute_G_at_node` calls existing `model.forward`, `compute_risk`, and `_compute_epistemic_value`; `AIFGenerator` orchestrates tree search using pure tensor ops. No `nn.Linear`, no MLP, no activation function appears in any class description. Constraint satisfied.

### 5.2 No CLI arguments

Agent 3 §1 names the entry point `train_aif.py` and §3 ¶1 states it is "the click-to-run example that builds a small `VFEModel`, loads a checkpoint, and runs `AIFGenerator.generate` on a prompt." Agent 2 §3.6 also calls out the constraint: "the new entry uses the click-to-run config dict pattern at the top of file, matching `train_vfe.py`." The existing `train_vfe.py:104-108` confirms the project's click-to-run convention. Constraint satisfied by Agent 3's design.

### 5.3 Gauge equivariance preserved

Agent 2 §5 lists the new compute paths and certifies each: re-encode delegates to `VFEPriorBank.encode` (sandwich at line 321 verified in §2.1); E-step uses the fused block-diag kernel (line 1006-1028 verified); tree-search frontier is data-structure only with no transport; `compute_risk` / `compute_epistemic_value` / `decode` consume already-transported beliefs. Agent 2 §5 ¶7 explicitly recommends against incremental belief updates: "the simpler — and safer — pattern is to *not* attempt incremental belief updates and to instead re-run the full E-step at each node. The cost is borne by the depth/branching factor, not by adding transport code paths." Agent 3 §3 ¶2 follows this recommendation: "the function first retrieves the parent belief from `cache`; if missing, it recursively recomputes by running `model.forward` on the parent's full token sequence." No new transport code is introduced. Constraint satisfied.

### 5.4 E-step blindness

`transformer/vfe/e_step.py:725-731` has no `targets` parameter (verified §2.1). `transformer/vfe/model.py:129` declares `targets` but uses it only at the CE-loss site at line 217, after `self.stack(...)` returns (line 193-196). Agent 3 §4 `AIFGenerator.generate` calls `model.forward(context_ids, targets=None)` exclusively. Agent 3 §5 explicitly defends option A (generation-only EFE) on the ground that training-time EFE would risk re-introducing the target / preference confusion. Constraint satisfied.

### 5.5 Pure path under appropriate toggles

Agent 3 §9 ¶3 maps a 2×2 interaction matrix between `AIFGenerator` use and the renamed `self_evidencing_regularizer` flag, and identifies the top-left cell (both off) as "the recovery path that matches the pure manuscript F functional at `Attention/GL(K)_attention.tex` `\label{eq:free_energy_functional_final}`." Agent 3 §10 ¶2 (first reduction invariant) requires that `VFEModel.generate(use_efe=False)` remain bitwise unchanged when the new `transformer/aif/` module is present but not used. Constraint satisfied.


## §6. Drift / Contradictions Between Agents

### 6.1 Default horizon: Agent 3 D=1 vs Agent 4 D=2

Agent 3 §7 default: `horizon_D = 1`. Agent 4 §6 recommended demo: `horizon_depth = 2`. These are not contradictory in scope (one is the conservative anchor, the other is the demo target) but the 06_plan.md must pick one configuration for the user-facing `train_aif.py`. Flag for the planner.

### 6.2 Default beam width: Agent 3 b=16 vs Agent 4 b=4

Agent 3 §7 default: `beam_width = 16`. Agent 4 §6 demo: `beam_width = 4`. Same scope distinction as §6.1; flag for the planner.

### 6.3 Decode-prior cache reuse — gap in Agent 3 design

Agent 4 §7 ¶3 mandates suppressing `prior_bank.invalidate_cache()` calls within the tree-search loop to avoid 2.3 GB of redundant compute at D=3, b=8. Agent 3 §2 `BeliefStateCache` design does not address this. Not a contradiction (Agent 3's cache covers a different object) but a missing architectural item Agent 3 should incorporate. Flag for §8 patch.

### 6.4 Full-cov runtime guard — gap in Agent 3 validation

Agent 4 §7 ¶2 requires a runtime guard that rejects `diagonal_covariance=False` at `horizon_D > 1`. Agent 3 §7 `__post_init__` validation list does not enumerate this guard. Flag for §8 patch.

### 6.5 Double-encode bug attribution

Agent 4 §2 step 2 flags the redundant encode at `efe.py:191` as a "correctness-relevant design gap." Agent 2 §1.8 mentions the re-encode but does not call it a bug. Agent 3 §3 ¶2 design has `compute_G_at_node` call `model.forward` (not a redundant `encode`), so the architecture does not perpetuate the bug — but only if the eventual implementation either (a) avoids the second `encode` entirely by reusing the belief tensors from the `forward` pass, or (b) modifies `VFEModel` to optionally return its converged beliefs. Agent 2 §4.2 ¶2 flags this exact accessor question: "If the AIF needs the converged belief state (for use as the new node's `leaf_belief`), it must either call a new accessor (e.g., `model.forward_with_beliefs(context_ids)`) or, if reluctant to add the accessor, monkey-patch the final-norm output capture." The 06_plan.md must commit to adding the accessor (or equivalent) and must not silently inherit the double-encode pattern from the existing `vfe/efe.py:191-196`.

### 6.6 NLL identification overstatement

Agent 1 §4 ¶4 second sentence claims the empirical-marginal-preference pragmatic term recovers "the standard variational F = NLL + KL(q || p)" under the Dirac collapse. This conflates the preference's NLL (`-log freq_train(y)`) with the model's predictive NLL (`-log q(y|a)`); the latter is what standard CE training optimizes. The two coincide only under the substitution `q(o|a) = p*(o)`, which is not the operative claim. Flag for §8 patch on Agent 1.

### 6.7 No other direct contradictions

The four documents agree on: the sign convention of G as energy-to-minimize, the BALD form of ambiguity as mean predictive entropy, the policy posterior as `softmax(-γG)`, the empirical-marginal preference as the default, the recommendation for option A (generation-only EFE) training, the depth-1 reduction anchor, the gauge-equivariance compliance through reuse of existing primitives, and the cache-by-prefix design.


## §7. Missing Claims — What the Canon Requires That No Document Discusses

### 7.1 Habit prior over policies

Canon (`[FristonEtAl2017]` §3, `[ParrPezzuloFriston2022]` §2.4) writes the policy posterior as `q(π) ∝ exp(-γG(π)) × E_prior(π)`, where `E_prior(π)` is a habit / a-priori prior over policies (often referred to as the "expected policy" prior, denoted `E` in Friston's notation). The factor is implicitly uniform in Agent 1 eq (14), Agent 3 §4 control flow, and Agent 4's cost estimates. No document states the uniform-habit assumption explicitly. Missing claim; flag for §8 patch on Agent 1 (math) and Agent 3 (config — exposable as `AIFConfig.habit_prior: Optional[torch.Tensor] = None`).

### 7.2 Precision γ — fixed, learned, or annealed?

Canon `[FristonEtAl2017]` §3 treats γ (the policy precision) as itself inferred under a Gamma hyperprior, with γ-updates coupled to the action posterior via belief propagation. Agent 1 §2 ¶6 calls γ a "policy precision" but does not specify whether it is fixed or learned. Agent 3 §7 sets `gamma: float = 1.0` as a fixed config field. Agent 4 §6 carries through with `'gamma': 1.0`. None of the four documents discusses γ-precision inference or annealing.

This is acceptable for a first build (fixed γ is the simplest reasonable choice), but the omission should be acknowledged. Missing claim; flag for Agent 1 §4 to add a paragraph (γ fixed at construction is a defensible default; the Gamma-prior inference of `[FristonEtAl2017]` is a research extension).

### 7.3 Sophisticated inference vs beam approximation

Agent 1 §3 ¶3 and Agent 3 §8 both correctly distinguish the beam-search default from full sophisticated-inference recursion. The two documents agree the beam approximation is the cheap starting point and SI recursion is the canonical full form. No drift on this; the documents are explicitly cautious not to conflate beam-search-over-cumulative-G with the SI recursion through child action posteriors. Good.

### 7.4 Policy novelty term

Canon `[ParrPezzuloFriston2022]` §9.3 and the SI paper expose a "novelty" term that quantifies expected information gain about model parameters (not just about latent states). In the LM specialization with frozen model parameters at generation time, this term is zero by construction — the action does not change the M-step parameters. None of the four documents derives this explicitly. Acceptable omission for a fixed-parameter generation-time build, but a one-line acknowledgement in Agent 1 §1 (where the generative model is fixed) would close the gap.

### 7.5 Discount across depth

Agent 3 §7 exposes `discount: float = 1.0` (undiscounted geometric sum, matching Agent 1 eq (18)). Agent 1 §3 ¶3 mentions the depth-D policy-sum but does not discuss discounting. Friston 2021 SI paper does not require discounting; the discount in Agent 3 is a hyperparameter without a canonical default beyond 1.0. No drift, but Agent 1 should add a one-paragraph note in §3 that the discount is an engineering hyperparameter, not a canonical EFE quantity.


## §8. Verifier-Supplied Corrections

This section enumerates the patches the investigators should accept before the 06_plan.md proceeds. For each item, the patch destination is named; the verifier supplies the corrected statement where one is available.

### 8.1 Friston 2021 sophisticated-inference citation

**Destination:** Agent 1 §6 (replace the `[full citation needed]` tag) and `external_bibliography.md` (add the entry).

**Correction:** `[Friston2021SophisticatedInference]` Friston, K., Da Costa, L., Hafner, D., Hesp, C., Parr, T. (2021). "Sophisticated Inference." *Neural Computation* 33(3): 713–763. arXiv:2006.04120.

Agent 3 §8 ¶2 already supplies this in prose; the citation must propagate to Agent 1's bibliography section and to `external_bibliography.md`.

### 8.2 Houlsby BALD citation

**Destination:** Agent 1 §6 (replace `[full citation needed]`).

**Correction:** `[HoulsbyEtAl2011]` Houlsby, N., Huszár, F., Ghahramani, Z., Lengyel, M. (2011). "Bayesian Active Learning for Classification and Preference Learning." arXiv:1112.5745.

### 8.3 Friston 2012 dark-room citation

**Destination:** Agent 1 §6.

**Correction:** `[Friston2012Darkroom]` Friston, K., Thornton, C., Clark, A. (2012). "Free-energy minimization and the dark-room problem." *Frontiers in Psychology* 3: 130.

### 8.4 Banned-phrase patch in Agent 4

**Destination:** Agent 4 §7 ¶2 line 220.

**Correction:** Replace "More critically, the attention grid cost scales ..." with "More severely, the attention grid cost scales ..." or any non-banned synonym.

### 8.5 NLL identification clarity nit

**Destination:** Agent 1 §4 ¶4.

**Correction:** The pragmatic term under the empirical-marginal preference is the cross-entropy of the model's predictive distribution against the corpus marginal `H(q(o|a), freq_train)`, not the standard CE loss `-log q(y|a)` at the realized next token. The two coincide only under the substitution `q(o|a) = freq_train(o)`, which is the limit of an uninformative model. Agent 1 should rewrite the second sentence of §4 ¶4 to avoid the conflation, e.g., "the pragmatic term becomes the cross-entropy of the model's predictive against the training marginal; the standard CE loss at a realized token is the special case `q(o|a) = δ_{o=y}`."

### 8.6 Decode-prior cache reuse in `AIFGenerator`

**Destination:** Agent 3 §3 ¶2 or §4 control flow; 06_plan.md must enumerate the build-out item.

**Correction:** The tree-search loop must suppress `prior_bank.invalidate_cache()` calls within a single generation step (or equivalently, hoist the decode-prior `(V, K)` computation out of the per-node forward). Agent 4 §7 ¶3 supplies the rationale; the architecture document does not address it. Recommended patch: add a `frozen_decode_prior` context manager around the tree-search loop in `AIFGenerator.generate`, or add a `model.forward(..., reuse_decode_cache=True)` flag.

### 8.7 Full-cov runtime guard

**Destination:** Agent 3 §7 `__post_init__` validation list.

**Correction:** Add a guard that reads the underlying `VFEModel.cfg.diagonal_covariance` and raises if `False` with `horizon_D > 1`. Agent 4 §7 ¶2 supplies the rationale; the architecture document elides it.

### 8.8 Double-encode in `compute_G_at_node`

**Destination:** Agent 3 §3 ¶2.

**Correction:** `compute_G_at_node` must reuse the converged belief from the `model.forward` pass rather than re-running `prior_bank.encode` separately. This requires either an accessor on `VFEModel` that returns the converged `BeliefState` (preferred) or capturing it via the existing `stack` output. Agent 2 §4.2 ¶2 flags the accessor question; the 06_plan.md must commit to adding it.

### 8.9 Habit-prior acknowledgement

**Destination:** Agent 1 §2 (after eq 14) and Agent 3 §7 (config schema).

**Correction:** Agent 1 should add a sentence after eq (14) noting that the canonical posterior is `q(π) ∝ exp(-γG(π)) × E_prior(π)` with `E_prior(π)` defaulting to uniform under this build-out. Agent 3 should expose `habit_prior: Optional[torch.Tensor] = None` (None means uniform) in `AIFConfig`.

### 8.10 γ-precision treatment

**Destination:** Agent 1 §2 or §5.

**Correction:** Agent 1 should add a paragraph noting that γ is held fixed at construction in this build-out, with the Gamma-prior precision inference of `[FristonEtAl2017]` deferred as a research extension.


## §9. Verdict on the Plan

**Color:** YELLOW.

**Justification:** The four documents are mutually consistent on every load-bearing claim — the sign convention of G, the BALD form of ambiguity, the policy-posterior softmin, the empirical-marginal preference default, the option-A training recommendation, the depth-1 reduction anchor, and the gauge-equivariance compliance through reuse. Math is canonical, code references are verified bitwise, and no CLAUDE.md hard constraint is violated by the proposed build. The drift is confined to:

- one banned-phrase use in Agent 4 (one-word fix),
- the default-horizon and beam-width gap between Agent 3 §7 (anchor) and Agent 4 §6 (demo) which the planner must resolve in 06_plan.md,
- two missing items in Agent 3's design that Agent 4 surfaced (decode-prior cache reuse, full-cov runtime guard),
- one architectural commitment Agent 2 surfaced and Agent 3 must close (the `VFEModel` accessor returning the converged belief),
- four citation fillings (Friston 2021, Houlsby 2011, Friston 2012; the Hohwy and RLHF references in Agent 1 §6 remain `[full citation needed]` and should be similarly resolved by Agent 1 before manuscript submission),
- one NLL-identification clarity nit in Agent 1 §4 ¶4,
- two missing-claim acknowledgements (habit prior, γ-precision treatment).

None of these block the planner. The 06_plan.md can proceed once the corrections in §8 are merged into Agents 1, 3, and the bibliography file; the build-out items in §8.6, §8.7, and §8.8 become first-tier phase requirements rather than rework. Re-verification after the corrections is a desk-check, not a re-run of the four investigators.
