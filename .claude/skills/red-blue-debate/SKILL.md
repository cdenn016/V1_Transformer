---
name: red-blue-debate
description: Structured red-team/blue-team adversarial debate on a specific claim about the codebase, manuscripts, theory, or math. Two modes — panel=full dispatches coordinators that each pick 5-of-10 expert consultants (philosophy-of-science mandatory) per side and a 3-judge panel + chief reconciler with a soft-cap canon-cop validator between rounds; panel=lite preserves the original 1-red, 1-blue, 1-judge structure. Every agent dispatch is in ultrathink mode. Outputs durable artifacts in docs/debates/.
allowed-tools: Read Write Edit Bash Glob Grep Agent WebFetch
metadata:
    skill-author: cdenn016
---

# Red Team / Blue Team Debate

Structured adversarial debate on a specific claim, with two operating modes:

- **`panel=full`** (default, 2026-05-21 onward) — each side is a coordinator + 5 expert consultants dynamically selected from a 10-expert roster (philosophy-of-science mandatory). Judging is a 3-judge panel (canon-strict, code-truth, scope) reconciled by a chief judge. A canon-cop validator runs between rounds with a soft-cap rewrite rule. ~32–36 agent dispatches in the default 2-round configuration.
- **`panel=lite`** — original structure: 1 red, 1 blue, 1 judge. 2–5 dispatches. Preserved for cheap mode and backwards-compatible reruns.

Every agent dispatch is in ultrathink mode.

## When to use

- Resolving a non-trivial claim about the codebase, manuscripts, or theory where a single auditor might miss an objection.
- Cross-checking the output of `vfe-codebase-auditor` or `vfe-manuscript-reviewer` when stakes warrant.
- Adjudicating a design decision where the user wants real adversarial pressure.

Do not use for:
- Quick reviews — use `vfe-codebase-auditor` or `code-reviewer` directly.
- Open-ended exploration — use the brainstorming skill.
- One-off lookups — use `Grep` / `Read`.

Choose `panel=lite` when you want the old fast path. Choose `panel=full` (default) when you want the best possible verdict and don't mind ~35 dispatches.

## Source of truth

The user's Gauge-Theoretic VFE construction — including `Attention/*.tex`, `CLAUDE.md`, `user_theory_summary.md`, in-repo derivations, and any project-specific equations or definitions — is a **work in progress that is being evaluated**, not the source of truth. The standard external literature is the source of truth:

- information geometry: Amari, Nielsen
- differential geometry: Nakahara, Lee, do Carmo
- gauge theory: Nakahara, Baez & Muniain
- variational inference / FEP / active inference: Friston, Beal, Bishop, Blei
- transformer attention: Vaswani et al. 2017 and the canonical attention literature
- general ML: Goodfellow/Bengio/Courville, Kingma, Loshchilov, He, Hinton, Kaplan, Hoffmann
- numerical analysis: Higham, Trefethen, Golub & Van Loan, Absil
- philosophy of science: Popper, Lakatos, Cartwright, Hacking

When a manuscript equation, a `CLAUDE.md` statement, or a `user_theory_summary.md` excerpt disagrees with the standard form, **the standard form wins**. The purpose of these debates is to surface errors in the user's construction by adversarially pressure-testing it against the canon. A team that a priori treats the user's framework as correct will miss exactly the errors the debate exists to find.

Operational rules for every coordinator, expert, canon-cop, judge, and chief dispatch:

1. The user's manuscripts and CLAUDE.md may be cited as the **claim under evaluation**, never as the standard.
2. Establishing the canonical form requires an external citation: `external_canon_*.md` entry, textbook section (e.g., `[Nakahara2003 §10.3]`), or paper equation (e.g., `[Friston2010 Eq. 2.2]`, `[Vaswani2017 §3.2.1]`). A `WebFetch`/`WebSearch` to a primary source is also valid.
3. "The manuscript says X, therefore X" is a malformed argument and **canon-cop will fire a strike**.
4. Apparent agreement between the user's construction and the canon must be **verified**, not assumed.
5. Novel constructions by the user (those without a canonical counterpart) are not free passes — they require an explicit derivation from canonical primitives. "Novel" labels what needs the most scrutiny, not the least.

These rules apply to all four modes (`theory`, `math`, `code`, `implementation`). For `code` and `implementation` modes the in-repo source code itself is canonical for *what the code does* (read the code, not the comments), but the in-repo *theory* — what the code is claimed to implement — is still subject to the rules above.

## Invocation

```
/red-blue-debate <claim>
  [mode=theory|math|code|implementation]   default: inferred
  [panel=full|lite]                         default: full
  [rounds=1..5]                             default: 2
  [judging=panel|rubric|off]                default (panel=full): panel; default (panel=lite): rubric
  [experts=<comma-separated 5 tags>]        default: dynamic per side (panel=full only)
  [evidence=auto|paths:<glob>]              default: auto
```

If `<claim>` is empty, the skill prompts for it once and proceeds.

Rounds semantics:
- `rounds=1`: opening only.
- `rounds=2` (default): opening + rebuttal.
- `rounds=3`: + sur-rebuttal.
- `rounds=4`/`5`: additional response rounds (diminishing returns).

Expert override (`panel=full` only): `experts=geometer,info-geometer,variational,numerical-analyst,philosophy-of-science` forces that exact 5-expert panel on both sides (suppresses dynamic per-side selection). `philosophy-of-science` is still mandatory and will be added if omitted.

## Reasoning depth — ultrathink (mandatory)

Every coordinator, expert, canon-cop, and judge dispatch from this skill **must include the literal token `ultrathink` as the first line of the dispatch prompt**. This is not optional. See `protocol.md` for the exact dispatch templates.

The skill itself (orchestration, evidence gathering, validation) does not require ultrathink — only the agents.

## Protocol summary (panel=full default, rounds=2, judging=panel)

| Phase | Actor | Action |
|---|---|---|
| 0 | main Claude | Claim extraction → `00_claim.md` |
| 1 | main Claude | Shared evidence pack → `01_evidence.md` |
| 2 | red-coord + blue-coord (parallel) | Each coordinator: picks 5-of-10 experts → `02_<side>_panel_choice.md`; dispatches 5 experts in parallel → `memo_<side>_<expert>.md`; merges discovered canon → `01b_extended_evidence.md`; synthesizes → `02_<side>_opening.md` |
| 2.5 | canon-cop × 2 | Validate openings → `02_canoncop_<side>.md`; trigger rewrite if ≥3 strikes |
| 3 | red-coord + blue-coord (parallel) | Same coordinator-consultant pattern → rebuttals |
| 3.5 | canon-cop × 2 | Validate rebuttals |
| 3b–3d | (only if rounds≥3) | Sur-rebuttals |
| 4a | 3 first-pass judges (parallel) | `04_verdict_canon.md`, `04_verdict_code.md`, `04_verdict_scope.md` |
| 4b | chief judge | Reconciles → binding `04_verdict.md` |
| 5 | main Claude | Action extraction → `05_action.md` |

Lite-mode preserves the original 5-phase shape (`red-team`, `blue-team`, `debate-judge`) — see `protocol.md` for the lite-mode protocol.

Detailed protocol in `protocol.md` (read it on every invocation — it has the dispatch templates and validation rules).

## On invocation

1. Read `protocol.md` (this directory, sibling file).
2. Resolve the canon location per the algorithm in `protocol.md`.
3. Parse arguments. Infer `mode` if not given:
   - Mention of `.tex` files or paper titles → `theory`.
   - Mention of `∂`, `KL`, equations, derivatives → `math`.
   - Mention of `.py` files or config keys → `code`.
   - Mention of the project's hard constraints (gauge equivariance, sandwich product, E-step blindness) → `implementation`.
   - Otherwise prompt the user once for the mode.
4. Execute phases 0–5 per `protocol.md`. Choose `panel=full` path or `panel=lite` path based on the `panel` argument.
5. Present the final verdict path and a one-paragraph factual summary to the user.

## Outputs

All artifacts in `<cwd>/docs/debates/YYYY-MM-DD-<slug>/`. The artifact set depends on `panel` and `rounds`.

### `panel=full rounds=2 judging=panel` (default)

```
00_claim.md
01_evidence.md
01b_extended_evidence.md
02_red_panel_choice.md       02_blue_panel_choice.md
02_red_opening.md            02_blue_opening.md
memo_red_<expert1..5>.md     memo_blue_<expert1..5>.md
02_canoncop_red.md           02_canoncop_blue.md
03_red_panel_choice.md       03_blue_panel_choice.md
03_red_rebuttal.md           03_blue_rebuttal.md
memo_red_<expert>_rebuttal.md memo_blue_<expert>_rebuttal.md
03_canoncop_red.md           03_canoncop_blue.md
04_verdict_canon.md          04_verdict_code.md          04_verdict_scope.md
04_verdict.md                (binding — chief reconciler)
05_action.md
```

`rounds≥3` adds `03b_*` artifacts. `rounds=4`/`5` adds `03c_*` / `03d_*`.

### `panel=lite rounds=2 judging=rubric` (original shape)

```
00_claim.md
01_evidence.md
02_red_opening.md            02_blue_opening.md
03_red_rebuttal.md           03_blue_rebuttal.md
04_verdict.md
05_action.md
```

If `<cwd>/docs/` does not exist, it is created. If `<cwd>` is not a git repo, the skill warns and writes to `<cwd>/debates/` instead.

## Examples

### Theory-mode debate (default, full panel)

```
/red-blue-debate "The attention β = softmax(-KL/τ) form is the stationary point of the free energy with the attention-entropy term and uniform prior." mode=theory
```

Expected: each coordinator picks 5 experts (likely both include variational, info-geometer, philosophy-of-science; red likely picks numerical-analyst to attack the Lagrangian setup, blue likely picks geometer/gauge-theorist for structural defense). Three judges render first-pass verdicts; chief reconciles.

### Code-mode debate, lite-mode (cheap)

```
/red-blue-debate "The codebase computes raw Euclidean gradients on the Lie algebra without preconditioning." mode=code panel=lite
```

Expected: 3 dispatches total (red-team, blue-team, debate-judge). Red wins, citing the natural-gradient preconditioning in `transport_ops.py`.

### Implementation-mode debate, deep (3 rounds)

```
/red-blue-debate "skip_attention=True with em_mode=em_phi_q silently freezes sigma_embed at initialization." mode=implementation rounds=3
```

Expected: ~50 dispatches. Sur-rebuttal round forces both sides to engage with the strongest opposing point. Chief judge likely defers to code-truth on the implementation question.

### User-specified panel

```
/red-blue-debate "The decode logits = -KL/tau formulation is equivalent to standard cross-entropy in the small-Sigma limit." mode=math experts=info-geometer,variational,transformer-ml,numerical-analyst,philosophy-of-science
```

Expected: both sides use the same 5 experts. Adversarial structure remains (red attacks, blue defends), but expert diversity per side is removed.

## What this skill does not do

- Modify code. Verdicts produce action items; the user (or another skill/agent) acts on them.
- Synthesize sides when `judging=off`. The user adjudicates.
- Produce numerical quality scores. Verdicts are qualitative and evidence-grounded.
