---
name: red-blue-debate
description: Structured red-team/blue-team adversarial debate on a specific claim about the codebase, manuscripts, theory, or math. Dispatches isolated-context agents in parallel (red attacks, blue defends), then an evidence-weighing judge. Every agent dispatch is in ultrathink mode. Outputs durable artifacts in docs/debates/. Use when you want adversarial resolution of a non-trivial claim; do not use for quick reviews.
allowed-tools: Read Write Edit Bash Glob Grep Agent WebFetch
metadata:
    skill-author: cdenn016
---

# Red Team / Blue Team Debate

Structured adversarial debate on a specific claim. Three subagents (`red-team`, `blue-team`, `debate-judge`) dispatched in isolated context per phase. Every agent dispatch is in ultrathink mode.

## When to use

- Resolving a non-trivial claim about the codebase, manuscripts, or theory where a single auditor might miss an objection.
- Cross-checking the output of `vfe-codebase-auditor` or `vfe-manuscript-reviewer` when stakes warrant.
- Adjudicating a design decision where the user wants real adversarial pressure.

Do not use for:
- Quick reviews — use `vfe-codebase-auditor` or `code-reviewer` directly.
- Open-ended exploration — use the brainstorming skill.
- One-off lookups — use `Grep` / `Read`.

## Source of truth

The user's Gauge-Theoretic VFE construction — including `Attention/*.tex`, `CLAUDE.md`, `user_theory_summary.md`, in-repo derivations, and any project-specific equations or definitions — is a **work in progress that is being evaluated**, not the source of truth. The standard external literature is the source of truth:

- information geometry: Amari, Nielsen
- differential geometry: Nakahara, Lee, do Carmo
- gauge theory: Nakahara, Baez & Muniain
- variational inference / FEP / active inference: Friston, Beal, Bishop, Blei
- transformer attention: Vaswani et al. 2017 and the canonical attention literature

When a manuscript equation, a `CLAUDE.md` statement, or a `user_theory_summary.md` excerpt disagrees with the standard form, **the standard form wins**. The point of these debates is to surface errors in the user's construction by adversarially pressure-testing it against the canon. A team that a priori treats the user's framework as correct will miss exactly the errors the debate exists to find.

Operational rules for every red, blue, and judge dispatch:

1. The user's manuscripts and CLAUDE.md may be cited as the **claim under evaluation**, never as the standard.
2. Establishing the canonical form requires an external citation: `external_canon_*.md` entry, textbook section (e.g., `[Nakahara2003 §10.3]`), or paper equation (e.g., `[Friston2010 Eq. 2.2]`, `[Vaswani2017 §3.2.1]`). A WebFetch to a primary source is also valid.
3. "The manuscript says X, therefore X" is a malformed argument and must be flagged by the opposing team and by the judge.
4. Apparent agreement between the user's construction and the canon must be **verified**, not assumed. If the manuscript claims to derive a standard form, both teams check the derivation against the canonical derivation.
5. Novel constructions by the user (those without a canonical counterpart) are not free passes — they require an explicit derivation from canonical primitives. "Novel" labels what needs the most scrutiny, not the least.

These rules apply to all four modes (`theory`, `math`, `code`, `implementation`). For `code` and `implementation` modes the in-repo source code itself is canonical for *what the code does* (read the code, not the comments), but the in-repo *theory* — what the code is claimed to implement — is still subject to the rules above.

## Invocation

```
/red-blue-debate <claim>
  [mode=theory|math|code|implementation]   default: inferred
  [rounds=1|2]                              default: 2
  [judge=on|off]                            default: on
  [evidence=auto|paths:<glob>]              default: auto
```

If `<claim>` is empty, the skill prompts for it once and proceeds.

## Reasoning depth — ultrathink (mandatory)

Every `red-team`, `blue-team`, and `debate-judge` dispatch from this skill **must include the literal token `ultrathink` as the first line of the dispatch prompt**. This is not optional. See `protocol.md` for the exact dispatch templates.

The skill itself (orchestration, evidence gathering, validation) does not require ultrathink — only the three agents.

## Protocol summary

| Phase | Actor | Action |
|---|---|---|
| 0 | main Claude | Claim extraction → `00_claim.md` |
| 1 | main Claude | Shared evidence pack → `01_evidence.md` |
| 2 | red + blue (parallel, ultrathink) | Openings → `02_red_opening.md`, `02_blue_opening.md` |
| 3 | red + blue (parallel, ultrathink) | Rebuttals → `03_red_rebuttal.md`, `03_blue_rebuttal.md` |
| 4 | judge (ultrathink) OR user | Verdict → `04_verdict.md` |
| 5 | main Claude | Action extraction → `05_action.md` |

Detailed protocol in `protocol.md` (read it on every invocation — it has the dispatch templates and validation rules).

## On invocation

1. Read `protocol.md` (this directory, sibling file).
2. Resolve the canon location per the algorithm in `protocol.md` (look for `<cwd>/.claude/agents/vfe-knowledge/debate_methodology.md`, then any `*-knowledge/`, then fall back to embedded).
3. Parse arguments. Infer `mode` if not given:
   - Mention of `.tex` files or paper titles → `theory`.
   - Mention of `∂`, `KL`, equations, derivatives → `math`.
   - Mention of `.py` files or config keys → `code`.
   - Mention of the project's hard constraints (gauge equivariance, sandwich product, E-step blindness) → `implementation`.
   - Otherwise prompt the user once for the mode.
4. Execute phases 0–5 per `protocol.md`.
5. Present the final verdict path and a one-paragraph factual summary to the user.

## Outputs

All artifacts in `<cwd>/docs/debates/YYYY-MM-DD-<slug>/`:

```
00_claim.md
01_evidence.md
02_red_opening.md
02_blue_opening.md
03_red_rebuttal.md
03_blue_rebuttal.md
04_verdict.md
05_action.md
```

If `<cwd>/docs/` does not exist, it is created. If `<cwd>` is not a git repo, the skill warns and writes to `<cwd>/debates/` instead.

## Examples

### Code-mode debate (known-false claim)

```
/red-blue-debate "The codebase computes raw Euclidean gradients on the Lie algebra without preconditioning." mode=code
```

Expected: red wins, citing the natural-gradient preconditioning in `transport_ops.py` and related files.

### Theory-mode debate

```
/red-blue-debate "The attention β = softmax(-KL/τ) form is the stationary point of the free energy with the attention-entropy term and uniform prior." mode=theory
```

Expected: blue wins if the manuscript derivation matches the canonical Lagrangian form; red wins if there's a derivation gap.

### Implementation-mode debate (cheap mode)

```
/red-blue-debate "skip_attention=True with em_mode=em_phi_q silently freezes sigma_embed at initialization." mode=implementation rounds=1 judge=off
```

Expected: 2 dispatches total. User adjudicates.

## What this skill does not do

- Modify code. Verdicts produce action items; the user (or another skill/agent) acts on them.
- Synthesize sides when `judge=off`. The user adjudicates.
- Produce numerical quality scores. Verdicts are qualitative and evidence-grounded.
