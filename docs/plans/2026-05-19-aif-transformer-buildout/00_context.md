# Context — Canonical Active-Inference Transformer Build-Out

**Date:** 2026-05-19
**Origin:** Verdict of `docs/debates/2026-05-19-vfe-active-inference-impl/04_verdict.md` (RED_WINS) established that the existing `transformer/vfe/active_inference.py` is a self-evidencing surrogate, not canonical EFE. The user has now asked for a plan to build a *proper* active-inference transformer — separate module from `train_vfe.py` if needed.

**User intuition (load-bearing):** Friston's "policies" map to **future strings of tokens**.
This is exactly the canonical formulation in `[ParrPezzuloFriston2022 Ch. 7]` (discrete state-space active inference): a policy π is a sequence of actions over a finite planning horizon. For a transformer LM:

- **State** `s_t` = belief tuple `(μ_t, σ_t, φ_t)` over latent representation at position t.
- **Observation** `o_t` = the actual token emitted at position t.
- **Action** `a_t` = a choice of next token to emit, drawn from vocabulary V.
- **Policy** `π = (a_{T+1}, ..., a_{T+D})` = a sequence of token-emission actions over horizon D.
- **Preferences** `p^*(o | C)` = the agent's preferred outcome distribution under goal context C. For language modeling, the natural default is the empirical training-data marginal, in which case pragmatic value reduces to expected NLL.

## What this build-out must achieve

1. **Canonical EFE per the Parr-Pezzulo-Friston textbook**, with `G(π)` decomposed into pragmatic (expected cost under preferences) and epistemic (expected information gain) terms.
2. **Policy posterior** `q(π) ∝ exp(-γG(π))` over multi-step token sequences.
3. **Sophisticated inference** in the `[Friston2021SophisticatedInference]` sense: recursive EFE evaluation through a search tree of futures.
4. **Tractable policy expansion** under V=50257: beam search / top-k sampling at each tree node; deep horizons via belief-state caching.
5. **Preserve gauge-theoretic VFE invariants**:
   - Gauge equivariance: `Σ_transported = Ω @ Σ @ Ω.T`.
   - E-step blindness: no targets parameter on inference paths.
   - Natural gradients via Fisher metric per `[Amari1998]`.
6. **Clean separation from `/vfe`**: the training-time E-step minimizes the variational F (no EFE augmentation in the inner loop). The AIF module hooks in for generation-time policy selection and optionally for an EFE-augmented training objective.
7. **Honest theoretical reporting**: every term cites a canonical source; any deviation (e.g., preference-distribution choice for an LM) is labeled and justified.

## What this build-out must NOT do

- Re-introduce the non-canonical `H[p_pred]` substitution into the E-step. Per the verdict, that path stays available under its renamed `self_evidencing_regularizer` config — but it is not active inference.
- Add neural network components beyond the existing PriorBank decode and optional output linear projection (CLAUDE.md hard constraint).
- Add CLI arguments to new entry points (CLAUDE.md hard constraint; click-to-run config dict at top of file).
- Break gauge equivariance: any new transport must use the sandwich product.
- Pretend that canonical EFE for token-LM is a settled topic — there is real open research here, particularly around `p^*(o|C)` choice and policy expansion depth.

## Scope decisions deferred to investigation

- **Training-time use of EFE.** Two cleanly different choices:
  1. EFE is generation-time only; training stays standard VFE with cross-entropy.
  2. Training-time EFE-augmented objective where the loss IS `-log q(π_observed)` for the observed training-data sequence (treats the training trajectory as the chosen policy).
  Each has pros and cons; the architecture agent must propose a default and justify.
- **Preference distribution `p^*(o|C)`.** Natural defaults: training-data empirical marginal (recovers NLL), low-entropy preference (`p^* ∝ exp(-βH)`), task-conditioned distribution (RLHF-style). Pick one default; document the others as configurable.
- **Tree expansion strategy.** Beam vs MCTS vs nucleus sampling vs sophisticated-inference recursion. Each has different cost/quality tradeoffs.
- **Discount / horizon D.** Default depth, and how the agent evaluates beyond D (truncated G vs bootstrapped value).

## Constraints all agents must respect

- **No banned phrases** per `.claude/agents/vfe-knowledge/style_constraints.md`: key insight, crucially, critically, notably, importantly, it's worth noting, interestingly, fundamentally, in particular, leverages, underscores.
- **Citations required.** Every theoretical claim cites a primary source from `.claude/agents/vfe-knowledge/external_bibliography.md` (Friston 2010, Friston et al 2017, Parr-Pezzulo-Friston 2022, Amari 1998, Bogacz 2017, Houlsby et al 2011 if available; otherwise the surrogate canonical refs).
- **Code references with `path:line`** for any factual claim about the current codebase.
- **Math-purity rules** from CLAUDE.md: no LaTeX `\;` `\,` `\!` spacing macros; standard equation punctuation.

## Files the existing codebase exposes that may be reused

- `transformer/vfe/efe.py:VFEExpectedFreeEnergy` — generation-time, depth-1 EFE for single-token candidate scoring. Has canonical risk via `compute_risk` and canonical ambiguity = mean predictive entropy.
- `transformer/core/expected_free_energy.py:compute_risk` — risk computation.
- `transformer/vfe/prior_bank.py:VFEPriorBank` — encode/decode primitives.
- `transformer/core/types.py:BeliefState` — belief state container.
- `transformer/vfe/model.py:VFEModel` — full forward pass, generate method.

## Files that must NOT be modified (research-track path stays as-is, only renamed in a separate workstream)

- `transformer/vfe/active_inference.py` — the non-canonical surrogate. Rename per verdict action item 1.
- `transformer/core/active_inference.py` — same.

## Deliverables

- `01_canonical_efe_for_lm.md` (Agent 1: lit/math) — exact equations, citations, mapping to LM.
- `02_codebase_audit.md` (Agent 2: vfe-codebase-auditor) — reuse map and integration boundaries.
- `03_architecture_design.md` (Agent 3: module design) — file layout, classes, training/generation loops.
- `04_compute_feasibility.md` (Agent 4: performance) — cost/memory analysis.
- `05_verifier_report.md` (Verifier) — cross-checks math against code, flags drift.
- `06_plan.md` (Main Claude) — final executable plan with phased milestones.
