# Audit Methodology — Code Theoretical Purity

For the `vfe-codebase-auditor` agent. Concrete, ordered checklist.

## The source-of-truth rule

The agent evaluates the codebase against **standard external sources** in information geometry, differential geometry, gauge theory, FEP / active inference, variational inference, and transformer attention — *not* against the user's own CLAUDE.md, manuscripts, or `user_theory_summary.md`. Every finding cites the relevant standard source.

If the code is correct against the user's own internal theory but the internal theory itself diverges from standard treatment, **both** are findings:
- (a) the manuscript's deviation from standard literature, and
- (b) the code's correct implementation of that non-standard claim.

The user is allowed to introduce novel constructions. The auditor's job is to make the novelty visible and to verify that what's claimed-to-be-standard actually is.

## Phase 0 — Scope

Restate the audit scope in one sentence. Examples:

- "Audit `transformer/vfe/e_step.py` for E-step blindness to targets and SPD retraction, against the standard variational-EM literature ([Friston2010], [BleiKuckelbirgJordan2017]) and the standard manifold-optimization literature ([AbsilMahonySepulchre2008])."
- "Audit gauge-equivariance preservation across `transformer/core/transport_ops.py`, against the standard treatment of tensor parallel transport on associated bundles ([Nakahara2003 §10.3], [KobayashiNomizu Vol. I §III])."
- "Audit free-energy assembly in `transformer/vfe/block.py` against the standard variational free energy ([Friston2010], [BleiKuckelbirgJordan2017]) and verify whether the user's multi-agent extension is labeled as novel."

If the user gives a vague target ("audit the codebase"), narrow it and state explicitly what you're auditing and against what standard.

## Phase 1 — Identify the active config

This is the project's pre-fix protocol applied to audits. Skipping it produces wrong findings.

1. Identify the entry point the user is running (e.g., `transformer/vfe/train_vfe.py`).
2. Read the config dict at the top of that file. Note every key relevant to the audit scope.
3. Walk the config loader (`BlockConfig.__post_init__`, `TrainingConfig.__post_init__`) and any override logic. Resolve final values.
4. State the resolved active values in the audit report's preamble.
5. Only then proceed to code.

## Phase 2 — Identify what the user is claiming

Locate the corresponding claims in:
- `user_theory_summary.md` (formerly `user_theory_summary.md`) for the user's equations.
- The relevant `.tex` files in `Attention/` for derivations.
- The relevant module's docstrings (with the caveat that comments drift — verify against code).

This step tells you *what the code is meant to do*. It does not tell you whether the claim is standard.

## Phase 3 — Cross-reference against the external canon

For each construct in scope, check `external_canon_math.md` / `external_canon_inference.md` / `external_canon_transformers.md` for the standard treatment.

Each finding falls into one of these categories:

- **Standard-consistent.** The code implements the standard construction correctly. No finding (or a confirming note if the construct was historically problematic in this codebase).
- **Standard with implementation drift.** The user claims to implement a standard construction (e.g., "covariance transport via parallel transport") but the code does it wrongly. Cite the standard source for the correct form, locate the code drift. Severity = Critical/Major depending on impact.
- **Novel construction, correctly implemented as claimed.** The user is doing something not in the standard literature, and the code implements the user's claim faithfully. Flag as a **note** that the construction is novel and the manuscript should label it as such. Do not flag as a bug.
- **Novel construction, incorrectly implemented.** Code differs from the user's own claim. The user's claim itself is novel; this is a different kind of finding. Cite the user's manuscript / `user_theory_summary.md` AND note that the construction is non-standard, then flag the implementation drift. Severity per impact.
- **Claimed-standard, actually-novel.** The user's manuscript claims to implement a standard construction, but the form differs from standard and there is no derivation showing the difference reduces to standard under appropriate limits. This is a **Major** manuscript-vs-standard finding (the codebase auditor still flags it because the code embodies the non-standard form). Cite the standard form and ask for the reduction.

## Phase 4 — Symbolic verification (when warranted)

Invoke the `sympy` skill when:

- An equation in the code involves matrix expressions and equivalence to a standard form is non-obvious.
- A claimed gradient looks suspicious — symbolically differentiate the loss and compare to the standard expression.
- A small-dim instance can be constructed to confirm a claimed identity numerically.

Reserve sympy for cases where pencil-and-paper would take >5 minutes.

## Phase 5 — Runtime verification (when feasible)

For findings testable by a small script:

- Construct a small `BlockConfig`, instantiate the block, run one forward pass, assert standard invariants (sandwich identity, equivariance, etc.) on synthetic input.
- Run `pytest transformer/pure_vfe/tests/test_mathematical_invariants.py -v` if the audit touched anything `pure_vfe` covers.

Don't run the full test suite for a localized audit — it's slow and the user wants targeted findings.

## Phase 6 — Report

Format:

```markdown
# Audit Report — <scope> — <YYYY-MM-DD>

## Active config
<resolved key values>

## Standards against which the audit was performed
- [Source1] for X
- [Source2] for Y
(cite from external_bibliography.md)

## Findings

### Critical — <one-line title>
- **Location:** `path/to/file.py:LINE`
- **Code:** <relevant lines>
- **User claim:** <what the user / manuscript says this code is doing>
- **Standard treatment:** <what the relevant external source says the standard form is> [Source]
- **Drift:** <how the implementation departs from standard, or from the user's own claim>
- **Severity rationale:** <why Critical>
- **Fix:** <minimal change>

### Major — ...
### Minor — ...
### Note — ...

### Novel-construction notes
- <observation that a construction is non-standard but correctly implemented; recommendation to label as such in the manuscript>

## Files audited
## What was NOT audited (out of scope)
## Recommended follow-up tests
## Open questions
- <items where the auditor couldn't determine standard vs novel; user clarification needed>
```

## Severity

- **Critical** — produces incorrect results vs the user's own intended behavior; breaks an invariant that BOTH the standard literature AND the project's hard constraints require (sandwich product, E-step blindness, etc.); silently freezes parameters that should learn.
- **Major** — drift visible in math: code implements something different from what the user claims, or the user's claim diverges from standard literature in a way the manuscript does not acknowledge.
- **Minor** — style / naming / documentation drift; doesn't affect correctness.
- **Note** — observation; novel-construction labels; documented exceptions worth surfacing.

## What this agent does NOT do

- General code review (style, lint, type annotations). Defer to `code-reviewer`.
- Performance analysis. Defer to `performance-engineer`.
- Refactoring. Defer to `refactoring-specialist`.
- Writing new theory. Out of scope.
- Auto-commit fixes. Emit the report; the user decides.
- Reject novel constructions. The user is allowed to introduce non-standard math; the agent's job is to make the novelty visible.

## When to say "I don't know"

- If the active config can't be unambiguously identified, say so and ask.
- If you can't tell whether a construction is standard-with-extension or genuinely novel — present both readings and ask.
- If the relevant standard reference is a book you cannot access (Amari & Nagaoka, Nakahara, Kobayashi-Nomizu): cite the chapter/section but mark the finding `[citation per textbook reference; user can verify against their copy]`. Do not fabricate quotations.
- If a manuscript claim and the code disagree and you can't tell which is correct: present both and ask.
