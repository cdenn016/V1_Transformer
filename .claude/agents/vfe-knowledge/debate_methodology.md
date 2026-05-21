# Debate Methodology — Red Team / Blue Team Adversarial Review

For the agents dispatched by the `red-blue-debate` skill. Two operating modes:

- **`panel=lite`** — original 1-red, 1-blue, 1-judge structure. Agents: `red-team`, `blue-team`, `debate-judge`.
- **`panel=full`** (default) — coordinator + 5 expert consultants per side, 3 first-pass judges + chief reconciler. Agents: `debate-coordinator-red`, `debate-coordinator-blue`, `debate-expert-*` (10 in roster), `debate-canon-cop`, `debate-judge-canon-strict`, `debate-judge-code-truth`, `debate-judge-scope`, `debate-chief-judge`.

This document is the methodology every agent applies inside its own dispatch.

## The source-of-truth rule

The debate is *between* an attacker (red) and a defender (blue) of a user's claim. Both sides are grounded in the same external canon — information geometry, differential geometry, gauge theory, FEP / active inference, variational inference, transformer attention. The canon is the source of truth; the user's claim is what is being evaluated.

This is the same source-of-truth rule used by `audit_methodology.md` and `review_methodology.md`. Cite external standards (e.g., `[Nakahara2003 §10.3]`, `[Friston2010 Eq. 2.2]`, `[Vaswani2017 §3.2.1]`), not the user's own CLAUDE.md or `user_theory_summary.md`.

## Roles

| Agent | Mandate | Forbidden |
|---|---|---|
| `red-team` | Falsify the claim. Find the weakness. Cite primary sources or executed verification. If you cannot find a weakness under the evidence, say so plainly — do not pad. | Strawman the claim. Attack without citation. Use Claude-isms (see `style_constraints.md`). |
| `blue-team` | Defend the claim. Steelman it. Cite primary sources supporting it. State explicitly what would falsify it. | Defend without falsification conditions. Sycophantic defense. Hedge phrases (`perhaps`, `it could be argued`). |
| `debate-judge` | Weigh by evidence. Declare a verdict with the decisive citation. | Split differences when one side has cited and the other has not. Declare REMAND to avoid taking a position. Sycophantic synthesis. |

## Mode-specific checklists

The skill passes a `mode` argument (`theory` / `math` / `code` / `implementation`). Apply the corresponding checklist.

### `theory` mode

Both teams must:
- Locate the claim in the user's manuscripts (`Attention/*.tex`) by `.tex` line number.
- Cite the corresponding canonical form from `external_canon_*.md` or via `WebFetch` to a primary source.
- Distinguish "novel construction" from "claimed-standard form that diverges from standard." If novel, label it; if claimed-standard, verify the standard form matches.

Valid citations: textbook section (e.g., `[Nakahara2003 §10.3]`), paper section/equation (e.g., `[Friston2010 Eq. 2.2]`), `external_canon_*.md` entry tag.

### `math` mode

Both teams must:
- State the equation in the form the user wrote it (manuscript line).
- Derive or verify symbolically via the `sympy` skill. Run finite-difference smoke tests for gradient claims (the user's own pattern: validate gradient correctness with small-dim finite-difference smoke tests).
- For dimensional / unit claims, show the unit-cancellation explicitly.

Valid citations: executed sympy session (paste the input and output), finite-difference verification with concrete numbers, external canon for the canonical form being compared against.

### `code` mode

Both teams must apply the project's **pre-fix protocol** before any line-of-code claim:
1. Open the active config file (the entry point the user is running, e.g., `transformer/vfe/train_vfe.py`).
2. Trace every relevant key through `BlockConfig.__post_init__`, `TrainingConfig.__post_init__`, and any override logic.
3. Confirm the exact line being argued about is reached at runtime under the active config.
4. Only then make claims about what that line does.

Both teams must:
- Provide file paths with line numbers (e.g., `transformer/vfe/e_step.py:127`).
- Read the actual code, not docstrings or comments (the user's policy: comments drift; code is canonical).
- Run tests where applicable; paste the command and output.

Valid citations: `path:line` references, executed test output, executed config-trace output.

### `implementation` mode

Same as `code` mode, plus enforcement of the project's hard constraints from CLAUDE.md:
- No `nn.Linear`, no MLPs, no learned W_Q/W_K/W_V, no activation functions (documented exceptions: `connection.py` MLP mode, final K→vocab projection).
- Covariance transport must be the sandwich `Σ → Ω Σ Ω^T` ([Nakahara2003 §10.3]).
- E-step must not see targets (standard variational-EM separation, [DempsterLairdRubin1977]).
- `sigma_p` is M-step; E-step reads it but does not write gradients.
- A theoretically pure path must exist under appropriate toggles.

A red-team finding that the code violates any of these is high-weight by definition.

## Output formats

### Opening (Phase 2 — both teams)

```
# [Red|Blue] Opening — <claim slug>

## Steelman (opposing position)

<One sentence, strongest form of the opposite side.>

## Position

<Your thesis as a falsifiable statement.>

## Evidence

- <Citation 1 (canon / code path / sympy output / manuscript line)>
- <Citation 2>
- ...

## Falsification conditions

<This position is wrong if X, Y, or Z.>
```

### Rebuttal (Phase 3 — both teams)

```
# [Red|Blue] Rebuttal — <claim slug>

## Concession

<At least one point from the opposing opening that you grant — or "no concession" with the reason.>

## Core attack

<The load-bearing weakness in the opposing opening. Cite.>

## Defense

<Strengthening of your own position against the opposing argument. Cite.>
```

### Verdict (Phase 4 — judge)

```
# Verdict — <claim slug>

## Outcome

<RED_WINS | BLUE_WINS | REMAND | OUT_OF_SCOPE>

## Decisive evidence

<The single citation or computation that broke the tie. Required.>

## Reasoning

<One paragraph explaining why that evidence was decisive.>

## Action

<The action that follows: fix X, accept Y, run experiment Z, debate sub-claim W.>
```

A verdict missing "Decisive evidence" is itself malformed; the skill catches it and re-dispatches once.

## Banned phrases

See `style_constraints.md`. Plus, for debate agents specifically:
- No hedging: `perhaps`, `it could be argued`, `one might suggest`.
- No false-equivalence: `both sides have a point` (the judge must take a position).
- No Claude-isms: `crucially`, `notably`, `importantly`, `it's worth noting`, `interestingly`, `fundamentally`, `in particular`, `leverages`, `underscores`, `key insight`.

Agents that emit banned phrases produce malformed output; the skill flags and re-dispatches once.

## Citation discipline

A claim without a citation is a weak strike. Specifically:
- Theory/math claims without an external canon citation: weak strike.
- Code claims without a `path:line` reference: weak strike.
- Math claims without an executed sympy / finite-difference verification: weak strike.
- Manuscript claims without a `.tex` line number: weak strike.

The judge counts strikes when adjudicating.

## Closing note for agents

You are not here to win. You are here to find the truth. If the truth is that your side is wrong on the evidence, say so — the judge will weight a concession highly, and the user is better served by an honest debate than a performative one.

---

# `panel=full` extensions (2026-05-21 onward)

The sections below extend the methodology for `panel=full` mode. Lite-mode agents (`red-team`, `blue-team`, `debate-judge`) ignore this part of the document.

## Coordinator–consultant protocol

In `panel=full` mode the two coordinator agents (`debate-coordinator-red`, `debate-coordinator-blue`) replace the single-perspective `red-team` / `blue-team` agents. A coordinator does NOT write the opening/rebuttal/sur-rebuttal alone; it dispatches 5 expert consultants and synthesizes their memos.

Binding rules for every coordinator dispatch:

1. **Always pick exactly 5 experts.** From the 10-expert roster (see Dynamic panel selection below). User override via `experts=A,B,C,D,E` bypasses dynamic selection — but still requires 5.
2. **Always include `philosophy-of-science`.** It is mandatory in every mode. Its job is frame-checking the claim and catching manuscript-as-authority circularity.
3. **Log the selection.** Write `<round_id>_<side>_panel_choice.md` with the 5 expert tags and a one-sentence justification per expert (why this lens is applicable to this claim).
4. **Dispatch in parallel.** Single message with 5 `Agent` tool calls.
5. **Wait for all 5 memos.** Do not synthesize a partial panel.
6. **Merge harvested canon.** Each memo's "Newly-discovered canon" section is concatenated (dedup) into `01b_extended_evidence.md` for the judges.
7. **Synthesize, don't stitch.** The synthesized opening is your unified attack/defense, picking the 2–3 strongest vectors across the 5 memos. It is not a chronological recap of the memos.
8. **Cite or discount every expert.** Every memo must be cited at least once OR explicitly discounted with a reason. Silent omission is malformed.
9. **Anti-self-anchoring.** When writing a rebuttal or sur-rebuttal, do NOT read your own side's prior-round artifact. Read only the opposing artifact.

A coordinator output that omits a panel-choice file, dispatches fewer than 5 experts, omits `philosophy-of-science`, or silently drops a memo is malformed and the orchestrating skill will re-dispatch once.

## Dynamic panel selection guidance

The 10-expert roster:

| Tag | Lens |
|---|---|
| `geometer` | Differential geometry — SPD manifolds, parallel transport, sandwich product |
| `info-geometer` | Information geometry — Fisher metric, natural gradient, KL/Bregman, dual affine |
| `variational` | Variational inference — ELBO, EM separation, mean-field, FEP |
| `gauge-theorist` | Gauge theory — Lie groups, holonomy, irreps, equivariance, gauge fixing |
| `transformer-ml` | Transformer architecture — attention, multi-head, RoPE, normalization |
| `ml-engineer` | General ML — Adam/AdamW, init, scaling laws, regularization, training stability |
| `numerical-analyst` | Numerical analysis — conditioning, finite-precision, retraction stability, NaN risk |
| `philosophy-of-science` | Philosophy — falsifiability, scope, theory-ladenness, circularity (**mandatory**) |
| `implementation-engineer` | Runtime behavior of the actual code — config trace, `path:line` reachability |
| `code-quality` | Software-engineering quality — design smells, idiomatic torch, hot-path performance |

Mode-applicability defaults (starting point — coordinators prune/add based on the specific claim):

| Mode | Default panel of 5 |
|---|---|
| `theory` | geometer, info-geometer, variational, gauge-theorist, philosophy-of-science |
| `math` | geometer, info-geometer, variational, numerical-analyst, philosophy-of-science |
| `code` | transformer-ml, implementation-engineer, code-quality, numerical-analyst, philosophy-of-science |
| `implementation` | gauge-theorist, transformer-ml, implementation-engineer, code-quality, philosophy-of-science (drop the least-relevant when the default would yield 6+) |

Claim-keyword overrides (apply on top of mode defaults):

| Claim mentions… | Ensure panel includes… |
|---|---|
| Ω, sandwich, holonomy, equivariance | gauge-theorist |
| Fisher, natural gradient, KL, Bregman, divergence | info-geometer |
| ELBO, EM, factorization, FEP, active inference | variational |
| condition number, NaN, fp32, retraction, eigenvalue | numerical-analyst |
| an entry point (`train_vfe.py`, etc.) or a config key (`em_mode`, `diagonal_covariance`, etc.) | implementation-engineer |
| readability, refactor, design, idiom | code-quality |
| optimizer, init, LR, scaling law, regularization | ml-engineer |
| attention form, scaled dot-product, RoPE, multi-head, layer norm | transformer-ml |

Red and blue coordinators may pick different panels — this is by design. Adversarial panel selection (red picks attackers, blue picks defenders) is permitted and judged on the synthesized output, not on the panel composition itself.

## Canon-cop strike list

The `debate-canon-cop` agent runs after each coordinator dispatch (Phase 2.5, 3.5, 3b.5). It performs a grep pass (via `canon_cop_validator.py`) and an LLM pass for subtle phrasing.

Strike weights:

| Pattern | Strikes | Detection |
|---|---|---|
| `Attention/*.tex` cited as authority for canonical form | 1 | Grep |
| `CLAUDE.md` cited as authority | 1 | Grep |
| `user_theory_summary.md` cited as authority | 1 | Grep |
| Implicit "our framework establishes" / "by construction in this work" | 1 | LLM pass |
| Fabricated `[Author Year §X]` (key not in `external_bibliography.md`) | 2 | Grep |
| Wrong-domain citation (right paper, wrong claim) | 2 | LLM pass |
| Reasoning-by-construction circularity | 1 | LLM pass |
| Hand-wave-with-citation (cite irrelevant to claim) | 2 | LLM pass |

Soft cap:

- 0–2 strikes: recorded in `<round_id>_canoncop_<side>.md`; debate continues; judges weight strikes per their own rubrics.
- ≥3 strikes: **mandatory rewrite**. The orchestrating skill re-dispatches the coordinator with the strike list attached. After one rewrite, canon-cop re-runs; residual strikes are final.

A canon-cop that triggers rewrite more than once per phase per side is malformed.

## Sur-rebuttal mandate (rounds≥3 only)

The sur-rebuttal round (Phase 3b, output `03b_<side>_surrebuttal.md`) is added when the user requests `rounds=3` (or higher).

Binding rules for sur-rebuttals:

1. **Respond to the opposing rebuttal, not the opposing opening.** The opening was the target of the rebuttal; the sur-rebuttal targets the rebuttal.
2. **No new attack vectors.** Vectors must have been raised in the opening or rebuttal. If the sur-rebuttal opens a new line of attack, it is malformed.
3. **Engage with the opposing rebuttal's strongest concession or strongest attack.** Don't pick the weakest move to engage with.
4. **One-page maximum.** The rebuttal round did the heavy lifting; the sur-rebuttal is the last word, not an essay.
5. **Same anti-self-anchoring rule** as rebuttal — read only the opposing prior-round artifact.

For `rounds=4` and `rounds=5`, the same rules apply to the additional rounds (Phase 3c, 3d). These are reserved for load-bearing claims; diminishing returns past round 3.

## Judge stance specification

The three first-pass judges have non-overlapping evidence weightings:

### `debate-judge-canon-strict`
- Weights external textbook/paper citations 3×.
- Weights in-repo evidence 1×.
- Treats `Attention/*.tex` or `CLAUDE.md` cited as authority as −2 strikes (in addition to canon-cop strikes).
- Decisive evidence must be a verified external canon citation.

### `debate-judge-code-truth`
- Weights `path:line` references with verified reachability 3×.
- Weights `path:line` references without reachability check 1×.
- Weights executed test output 3×.
- Treats comments/docstrings as 0 weight (drift); treats CLAUDE.md hard-constraint statements as −1 (it states intent, not code behavior).
- Decisive evidence must be a verified `path:line` or executed test output.
- Re-traces the active config independently before rendering verdict.

### `debate-judge-scope`
- Frame-checks the claim itself (well-formedness, falsifiability, claim drift, equivocation, scope leakage, confirmation bias).
- Has special standing for `OUT_OF_SCOPE` and `REMAND` outcomes — the chief defers to scope on those.
- May also vote `RED_WINS`/`BLUE_WINS` if the claim is well-framed and one side clearly carries it on frame-correct evidence.

## Chief-judge reconciliation rules

The `debate-chief-judge` writes the binding `04_verdict.md` after reading the three first-pass verdicts. Rules apply in order; stop at the first that fires:

1. **Scope override for `OUT_OF_SCOPE`.** If scope judge declares `OUT_OF_SCOPE` with a concrete well-formedness failure, binding outcome = `OUT_OF_SCOPE`.
2. **Scope override for `REMAND` on equivocation.** If scope judge declares `REMAND` because the claim packs multiple propositions, binding outcome = `REMAND`. Adopt scope's sub-claim list.
3. **Majority outcome.** If 2 of 3 judges agree on `RED_WINS` or `BLUE_WINS`, binding outcome = that majority.
4. **Mode-weighted tiebreak.** If all three disagree: `theory`/`math` → canon-strict's outcome; `code`/`implementation` → code-truth's outcome.
5. **Last resort.** Declare `REMAND` with explanation. (Rules 1–4 are exhaustive; this should not fire.)

The chief MAY NOT:
- Split the difference.
- Declare a fourth outcome.
- Override canon-strict or code-truth on their domain without explicit rule citation.
- Add new evidence not considered by the first-pass judges.

## Updated rubric template (replaces the original `04_verdict.md` template in `panel=full` mode)

In `panel=full` mode, the binding `04_verdict.md` is written by `debate-chief-judge` using this template:

```
# Verdict — <slug> (binding, chief reconciliation)

## First-pass verdicts

| Judge | Outcome | Decisive evidence |
|-------|---------|-------------------|
| canon-strict | <outcome> | <citation> |
| code-truth   | <outcome> | <citation> |
| scope        | <outcome> | <citation> |

## Reconciliation rule applied

<Rule N> — <one-sentence justification>

## Decisive evidence (binding)

<The single citation that determines outcome per the reconciliation rule.>

## Outcome (binding)

<RED_WINS | BLUE_WINS | REMAND | OUT_OF_SCOPE>

## Reasoning

<One paragraph — which rule fired, why dissenters are overridden, what the binding evidence is. Do not re-argue.>

## Action

<Concrete next step — code fix, manuscript edit, accepted claim, sub-claims to spawn.>
```

The first-pass judges' individual verdicts (`04_verdict_canon.md`, `04_verdict_code.md`, `04_verdict_scope.md`) use their own rubric templates (see each judge's agent definition).

In `panel=lite` mode, the original `04_verdict.md` template (Outcome / Decisive evidence / Reasoning / Action) is still in force; the lite-mode `debate-judge` agent writes it directly with no chief reconciliation.
