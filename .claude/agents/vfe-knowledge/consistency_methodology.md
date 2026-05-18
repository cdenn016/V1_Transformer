# Consistency Methodology — Cross-Manuscript Notation, Equation, Citation, Style

For the `vfe-cross-manuscript-consistency` agent. Concrete, ordered checklist.

## The source-of-truth rule

The agent enforces *internal consistency* across the user's manuscripts. The reference for "what should this symbol mean" is `notation_dictionary.md` plus the standard literature (`external_canon_*.md`). The reference for "what should this equation be" is the standard literature first, then the project's own conventions if the construct is novel.

The agent's job is to make drift visible — not to rewrite the manuscripts. It emits a structured drift report; the user decides what to fix.

## Phase 0 — Scope

Restate the consistency target in one sentence. Examples:

- "Full cross-manuscript consistency check across `Attention/*.tex` — notation, equations, citations, style."
- "Verify `α_i` (per commit `89e7982d`) has a single consistent meaning across the four main manuscripts."
- "Find all citations to Friston 2010 across `Attention/` and verify bibkeys are identical."
- "Check that the sandwich-product convention `Σ → Ω Σ Ω^T` is used identically wherever it appears."

If the user gives a vague target ("check consistency"), run the full scan (Phases 1–6).

## Phase 1 — Inventory the manuscript set

List all `.tex` files in `Attention/`. As of 2026-05-18:
- `GL(K)_attention.tex` (main)
- `GL(K)_supplementary.tex` (supplementary to main)
- `belief_inertia_unified.tex` (related — sociological belief dynamics, uses same gauge math)
- `Participatory_it_from_bit.tex` (related — broader framework, currently dirty per `git status`)
- `jmlr_coverletter.tex` (cover letter for main submission)
- `tikz.tex` (figures-only; not subject to most consistency checks but figure labels should match manuscript notation)

Other `.tex` files (preambles, includes) — list and decide per scope.

## Phase 2 — Notation drift scan

For each symbol in `notation_dictionary.md`:

1. **Definition consistency.** Grep `\newcommand` and `\providecommand` for the symbol across all manuscripts. Report drift (e.g., `\KL` as `\operatorname{KL}` vs `\mathrm{KL}` — already documented).
2. **Usage consistency.** Grep for usages (`\KL`, `\beta`, `\Omega`, `\phi`, etc.) and verify the meaning is the same. If the same symbol is used for two different concepts in two manuscripts (e.g., `α` for self-coupling weight in one paper and `α` for learning rate in another), that's a Major finding.
3. **Subscript/superscript conventions.** `μ_q` vs `μ^q`, `Σ^p` vs `Σ_p` — verify consistency. Mixed conventions in the same paper are usually typos; cross-manuscript mixing is a drift.

Use the symbol table in `notation_dictionary.md` as the spec. New symbols not in the dictionary that appear in only one manuscript are flagged as "manuscript-local" — they're allowed, but note that if the symbol becomes load-bearing across manuscripts it should be added to the dictionary.

## Phase 3 — Equation drift scan

For each canonical equation form in `notation_dictionary.md` "Equation-form invariants":

1. Grep the manuscripts for the equation. Common patterns:
   - Covariance transport: search for `\Omega.*\Sigma.*\Omega` or any `\Sigma_{` neighbors of `\Omega_{`. Verify the sandwich.
   - Transport factorization: search for `\Omega_{ij}` and `\exp(\phi`. Verify the two-exponential form.
   - Attention: search for `\beta_{ij}` and `\softmax`. Verify the `−KL/τ` form.
   - Free energy: search for `F` definitions; verify the `τ β log(β/π)` entropy term.
2. For each variant found, report:
   - Manuscript and section/line.
   - The variant form.
   - The canonical form (from `notation_dictionary.md` / `external_canon_*.md`).
   - Severity: Critical if used as load-bearing math; Minor if a typo in a single line that the rest of the derivation corrects.
3. For derivations that should appear in multiple manuscripts (e.g., the closed-form Gaussian KL): verify the derivations match line-by-line, not just the final form.

## Phase 4 — Citation drift scan

1. Extract every `\cite{}`, `\citep{}`, `\citet{}` from each manuscript. Build a global citation list.
2. Group by author/year. Within each group, check if the same paper has different bibkeys in different manuscripts.
3. For each cross-manuscript reuse, verify the bibkey is identical. Common drift: `friston2010free` vs `Friston2010` vs `friston_free_2010` — all referring to the same paper.
4. Also verify each cited paper is actually in the relevant `.bib` files (or as inline `\bibitem`).
5. **Cross-manuscript text references** like "as we showed in [our companion paper]" or `\cite{belief_inertia_unified}` should resolve. If a manuscript references another by its filename rather than a bibkey, flag as a stylistic issue.

## Phase 5 — `MR-N` consistency (track via `git log`, not `.tex` source)

The repo's commit history uses `MR-N` (math reviewer numbered items) to track math-reviewer feedback. **`MR-N` markers do not appear in the manuscript source** — grep at 2026-05-18 found no matches in `Attention/*.tex`. They live exclusively in commit messages.

To track `MR-N` status:

1. `git log --grep="MR-[0-9]" --all -- Attention/` to list commits referencing `MR-N` items.
2. For each `MR-N` referenced, summarize its status: `partial`, `complete`, or `open` based on commit message wording.
3. If a manuscript section claims to address an `MR-N` item (e.g., in a "Response to reviewers" appendix), verify the resolution is consistent with what the commit messages claim was done.

If the user introduces `MR-N` markers into `.tex` source (e.g., as `\todo{MR-3: ...}` macros or comments), revisit this phase to scan source as well as commits.

## Phase 6 — Style drift scan

Apply `style_constraints.md` to all manuscripts simultaneously and report patterns:

- Banned phrases: `key insight`, `crucially`, `critically` (sentence-opener), `notably`, `importantly`, `it's worth noting`, `interestingly`, `fundamentally`, `in particular`, `leverages`, `underscores`. Report per-manuscript counts so the user sees where to clean up.
- Banned LaTeX: `\;`, `\,`, `\!` spacing macros. Horizontal-rule visual separators in body text.
- Equation punctuation: comma/period at end of display equations.
- Self-referential drafting language (`feedback_no_self_referential_history.md` memory): "earlier drafts", "the corrected reading", etc.

## Phase 7 — Cross-manuscript narrative consistency

Higher-level than notation/equations: do the manuscripts present a coherent story?

- **Claim alignment.** If `GL(K)_attention.tex` claims "X is a degenerate limit", does `Participatory_it_from_bit.tex` consistently treat X as a special case? Or does it implicitly claim X is the general framework?
- **Scope alignment.** Are the empirical claims in the main paper and supplementary consistent? (E.g., if the main paper claims PPL 71.6 at one K and the supplementary shows different numbers at the same K, flag.)
- **Limitation alignment.** If one manuscript discloses a limitation (e.g., the documented RoPE×Mahalanobis gap), do the others disclose it too where relevant?

This phase is more interpretive — use it sparingly and lead with concrete evidence from the manuscripts.

## Phase 8 — Write the consistency report

Format:

```markdown
# Cross-Manuscript Consistency Report — <date>

## Manuscripts in scope
- `Attention/GL(K)_attention.tex`
- `Attention/GL(K)_supplementary.tex`
- ...

## Notation drifts

### N1. <symbol> — drift between manuscripts
- **Definitions found:**
  - `GL(K)_attention.tex:27`: `\newcommand{\KL}{\operatorname{KL}}`
  - `belief_inertia_unified.tex:13`: `\newcommand{\KL}{\mathrm{KL}}`
- **Recommended canonical form:** <pick one and recommend>
- **Severity:** Minor / Major / Critical
- **Fix:** unify in a shared `preamble.tex` `\input`-ed by both.

### N2. ...

## Equation drifts

### E1. <equation> — drift between manuscripts or from canonical
- **Found:** `<manuscript>:<line>` writes `<variant>`
- **Canonical:** per `notation_dictionary.md` / `external_canon_*.md` should be `<canonical>`
- **Severity:**
- **Fix:**

### E2. ...

## Citation drifts

### C1. <paper> — different bibkeys
- **Paper:** Friston 2010 "The free-energy principle: a unified brain theory?"
- **Keys found:**
  - `GL(K)_attention.tex`: `friston2010free`
  - `belief_inertia_unified.tex`: `Friston2010`
- **Recommended canonical key:** <pick one>
- **Fix:** rename in the second; rebuild bibliography.

### C2. ...

## MR-N status (if applicable)
- MR-1: <status across manuscripts>
- MR-2: ...

## Style drifts (per-manuscript counts)

| Banned phrase / pattern | GL(K)_attention | supplementary | belief_inertia | Participatory |
|---|---|---|---|---|
| `key insight` | 0 | 2 | 1 | 0 |
| `\;` spacing | 0 | 0 | 0 | 4 |
| ... |

## Narrative-consistency findings
### NC1. <title>
- **Manuscript A claim:** ...
- **Manuscript B claim:** ...
- **Tension:** ...
- **Recommended resolution:**

## Open questions
- <items where the agent can't classify drift severity without user input>

## Summary
- Total notation drifts: N
- Total equation drifts: N
- Total citation drifts: N
- Highest-severity item: <one-line>
- Recommended next action: <e.g., extract shared preamble.tex; unify bibkeys; redo cross-references>
```

If a section has nothing to report, write "(none)" — don't manufacture findings.

## What this agent does NOT do

- Rewrite the manuscripts. Recommend fixes; let the user apply them.
- Apply fixes automatically unless the user explicitly asks for a specific surgical fix (e.g., "unify `\KL` definitions to `\operatorname{KL}` across all manuscripts").
- Comment on the substance of the math (defer to `vfe-manuscript-reviewer`).
- Audit code (defer to `vfe-codebase-auditor`).

## When to invoke other skills

- `sympy` — to verify two variant forms of an equation are algebraically equivalent (rare; usually obvious).
- `scientific-writing` — only if the user asks for actual rewrites of inconsistent passages.

## When to say "I don't know"

- A symbol drift could be intentional (e.g., the manuscripts cover overlapping but distinct subjects and use the same letter for different things on purpose). Present the finding as "drift detected; verify intent."
- An equation variant might be a deliberate alternative form. Present the canonical form, ask whether the variant is intentional.
- A cited paper can't be located in either manuscript's `.bib` — flag and ask the user to verify.
