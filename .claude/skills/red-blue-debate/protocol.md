# Red Team / Blue Team Debate Protocol

Detailed phase-by-phase protocol for the `red-blue-debate` skill. The skill `SKILL.md` summarizes; this file is the full procedure.

## Arguments

```
/red-blue-debate <claim>
  [mode=theory|math|code|implementation]   default: inferred
  [rounds=1|2]                              default: 2
  [judge=on|off]                            default: on
  [evidence=auto|paths:<glob>]              default: auto
```

If `<claim>` is empty, prompt the user once for the claim, then proceed.

## Source-of-truth precedence (applies to every phase)

The user's Gauge-Theoretic VFE construction — `Attention/*.tex`, `CLAUDE.md`, `user_theory_summary.md`, in-repo derivations — is a **work in progress under evaluation**, not the source of truth. The source of truth is the standard external literature (information geometry, differential geometry, gauge theory, FEP / active inference, variational inference, transformer attention). When the user's construction disagrees with the canonical form, the canonical form wins.

The purpose of this skill is to surface errors in the user's construction by adversarially pressure-testing it against the canon. Both teams and the judge must treat the user's manuscripts and CLAUDE.md as the claim being evaluated, never as the standard. Establishing the canonical form requires an external citation (`external_canon_*.md` entry, textbook section, paper equation, or `WebFetch` to a primary source). "The manuscript says X, therefore X" is a malformed argument and must be flagged. Apparent agreement with the canon must be verified, not assumed; novel constructions without a canonical counterpart require an explicit derivation from canonical primitives, not a free pass.

This precedence rule is restated in every dispatch prompt below, and in the embedded fallback methodology at the bottom of this file. Do not omit it when constructing dispatch prompts.

## Canon-location resolution (run first)

Before any phase, resolve the canon location:

```
1. If `<cwd>/.claude/agents/vfe-knowledge/debate_methodology.md` exists → canon_location = `<cwd>/.claude/agents/vfe-knowledge/`.
2. Else if any `<cwd>/.claude/agents/*-knowledge/debate_methodology.md` exists → canon_location = that directory.
3. Else canon_location = "embedded" (use the minimal generic fallback below; pass excerpts inline to each agent dispatch).
```

Record `canon_location` at the top of `00_claim.md` so the agents know where to read from.

## Working directory

Create `<cwd>/docs/debates/YYYY-MM-DD-<slug>/` where `<slug>` is a kebab-case summary of the claim (max 40 chars). Use the current date.

If `<cwd>/docs/` does not exist, create it. If `<cwd>` is not a git repo, warn and use `<cwd>/debates/` instead.

## Phase 0 — Claim extraction (main Claude)

1. Parse the user-provided claim into a single declarative sentence.
2. If the claim is compound, identify the load-bearing proposition and note that sub-claims need separate debates.
3. Write `00_claim.md`:

```markdown
# Claim — <slug>

**Mode:** <theory|math|code|implementation>
**Rounds:** <1|2>
**Judge:** <on|off>
**Evidence scope:** <auto|paths:<glob>>
**Canon location:** <absolute path or "embedded">

## Claim

<One declarative sentence.>

## User context

<Any additional context the user provided.>
```

## Phase 1 — Shared evidence pack (main Claude)

Main Claude assembles a neutral fact pack. Output: `01_evidence.md`.

For `code` / `implementation` mode:
1. Identify the active entry point (e.g., `transformer/vfe/train_vfe.py`). If not given in context, ask the user.
2. Read the config dict at the top of that entry point.
3. Trace every relevant key through `BlockConfig.__post_init__`, `TrainingConfig.__post_init__`, and any override logic. Record resolved values.
4. Identify all code paths the claim references (Grep for relevant symbols).
5. Record `path:line` references for each.

For `theory` / `math` mode:
1. Locate the claim in `Attention/*.tex` if applicable; record `.tex` line numbers.
2. Pull the relevant canon excerpts from `<canon_location>/external_canon_*.md` if canon_location is not "embedded".
3. If primary sources need verification, use `WebFetch`.

Output template:

```markdown
# Evidence Pack — <slug>

## Active config (code/implementation mode only)

<entry point path, resolved values>

## Code references

- `path/to/file.py:LINE` — <one-line description>

## Manuscript references (theory/math mode)

- `Attention/file.tex:LINE` — <equation/passage>

## Canon excerpts

- `[Author Year §X]` — <quoted form>

## What this evidence does NOT settle

<List of questions the evidence pack alone cannot answer.>
```

If `evidence=paths:<glob>` is given, constrain search to those paths.

## Phase 2 — Openings (parallel agent dispatch)

In a single message, dispatch both `red-team` and `blue-team` with the `Agent` tool. The dispatch prompts:

**For red-team:**

```
ultrathink

You are dispatched as the red team in debate <slug>, Phase 2 (opening).

Working directory: <absolute path to docs/debates/<slug>>
Canon location: <absolute path, or "embedded">
Mode: <mode>

Read in order:
1. <working_dir>/00_claim.md
2. <working_dir>/01_evidence.md
3. <canon_location>/debate_methodology.md  (or use the inline excerpt below if canon_location is "embedded")
4. <canon_location>/style_constraints.md   (skip if embedded; observe the banned-phrase list in the agent definition)
5. The relevant <canon_location>/external_canon_*.md for citation forms.

Then write your opening to <working_dir>/02_red_opening.md using the Phase-2 template in debate_methodology.md.

Mandate: falsify the claim. Steelman first, then attack. At least one primary-source citation required.

Source-of-truth precedence (binding): The user's Gauge-Theoretic VFE construction — `Attention/*.tex`, `CLAUDE.md`, `user_theory_summary.md`, in-repo derivations — is a work in progress being evaluated, not the source of truth. The standard external literature (information geometry, differential geometry, gauge theory, FEP / active inference, variational inference, transformer attention) is the source of truth. Treat the user's manuscripts and CLAUDE.md as the claim under evaluation. Establishing the canonical form requires an external citation (`external_canon_*.md` entry, textbook section like `[Nakahara2003 §10.3]`, paper equation like `[Friston2010 Eq. 2.2]`, or `WebFetch` to a primary source). "The manuscript says X, therefore X" is malformed — flag it. Verify apparent agreement; do not assume it. Novel constructions require derivation from canonical primitives, not a free pass.

<If canon_location is "embedded": include the minimal-generic-fallback methodology excerpt here verbatim. The excerpt is in the "Embedded fallback methodology" section of this protocol file.>
```

**For blue-team:**

```
ultrathink

You are dispatched as the blue team in debate <slug>, Phase 2 (opening).

Working directory: <absolute path to docs/debates/<slug>>
Canon location: <absolute path, or "embedded">
Mode: <mode>

Read in order:
1. <working_dir>/00_claim.md
2. <working_dir>/01_evidence.md
3. <canon_location>/debate_methodology.md  (or use the inline excerpt below if canon_location is "embedded")
4. <canon_location>/style_constraints.md   (skip if embedded; observe the banned-phrase list in the agent definition)
5. The relevant <canon_location>/external_canon_*.md for citation forms.

Then write your opening to <working_dir>/02_blue_opening.md using the Phase-2 template in debate_methodology.md.

Mandate: defend the claim by steelmanning it. Cite primary sources. State falsification conditions explicitly. At least one primary-source citation required.

Source-of-truth precedence (binding): The user's Gauge-Theoretic VFE construction — `Attention/*.tex`, `CLAUDE.md`, `user_theory_summary.md`, in-repo derivations — is a work in progress being evaluated, not the source of truth. The standard external literature (information geometry, differential geometry, gauge theory, FEP / active inference, variational inference, transformer attention) is the source of truth. Defending the claim means showing it follows from the canon, not citing the manuscript back at itself. Establishing the canonical form requires an external citation (`external_canon_*.md` entry, textbook section like `[Nakahara2003 §10.3]`, paper equation like `[Friston2010 Eq. 2.2]`, or `WebFetch` to a primary source). The user wants errors found — sycophantic defense that relies on the manuscript as authority is malformed and will be flagged. Novel constructions require derivation from canonical primitives, not assertion.

<If canon_location is "embedded": include the minimal-generic-fallback methodology excerpt here verbatim. The excerpt is in the "Embedded fallback methodology" section of this protocol file.>
```

The literal `ultrathink` is the first token of each dispatch prompt. **This is load-bearing — every red/blue/judge dispatch must begin with `ultrathink`.**

After both return, validate:
- `02_red_opening.md` and `02_blue_opening.md` exist.
- Each contains all four required sections (Steelman, Position, Evidence, Falsification conditions).
- Each contains at least one citation.
- Neither contains banned phrases.

If any check fails, re-dispatch that side once with a corrective prompt naming the missing element. If the second attempt fails, append a malformed-output note ("malformed: missing element N, debate continues") to the file and proceed.

## Phase 3 — Rebuttals (parallel, isolated)

Skip this phase if `rounds=1`.

Dispatch both teams in parallel. Each receives the opposing team's opening only (not its own).

**For red-team:**

```
ultrathink

You are dispatched as the red team in debate <slug>, Phase 3 (rebuttal).

Working directory: <absolute path>
Canon location: <absolute path>
Mode: <mode>

Read:
1. <working_dir>/00_claim.md
2. <working_dir>/01_evidence.md
3. <working_dir>/02_blue_opening.md  (the opposing opening)
4. <canon_location>/debate_methodology.md (or embedded excerpt)

Then write your rebuttal to <working_dir>/03_red_rebuttal.md using the Phase-3 template.

Required: at least one concession, a core attack with citation, a defense with citation.

Source-of-truth precedence (binding, same as Phase 2): the user's manuscripts, CLAUDE.md, and in-repo derivations are the claim under evaluation, not the canon. The standard external literature is the source of truth. If blue's opening establishes the canonical form by citing the user's own manuscript or CLAUDE.md, that is a malformed argument and should be your core attack — demand an external citation. Apparent agreement with the canon must be verified, not asserted.

Do not read 02_red_opening.md — your rebuttal should be shaped by blue's opening, not by self-anchoring.
```

**For blue-team:**

```
ultrathink

You are dispatched as the blue team in debate <slug>, Phase 3 (rebuttal).

Working directory: <absolute path>
Canon location: <absolute path>
Mode: <mode>

Read:
1. <working_dir>/00_claim.md
2. <working_dir>/01_evidence.md
3. <working_dir>/02_red_opening.md  (the opposing opening)
4. <canon_location>/debate_methodology.md (or embedded excerpt)

Then write your rebuttal to <working_dir>/03_blue_rebuttal.md using the Phase-3 template.

Required: at least one concession, a core attack with citation, a defense with citation.

Source-of-truth precedence (binding, same as Phase 2): the user's manuscripts, CLAUDE.md, and in-repo derivations are the claim under evaluation, not the canon. The standard external literature is the source of truth. Defend the claim by deriving it from external canon, not by citing the manuscript as authority. If red's attack lands and the canon contradicts the user's construction, concede on the evidence — the user is better served by an honest debate than a performative defense.

Do not read 02_blue_opening.md — your rebuttal should be shaped by red's opening, not by self-anchoring.
```

Validation: each contains Concession, Core attack, Defense; each contains at least one citation; no banned phrases. Re-dispatch once on failure, then proceed with malformed note.

## Phase 4 — Judgment

Skip if `judge=off` (jump to "Phase 4-off").

Dispatch `debate-judge`:

```
ultrathink

You are dispatched as the judge in debate <slug>.

Working directory: <absolute path>
Canon location: <absolute path>
Mode: <mode>
Rebuttals: <yes|no — based on rounds>

Read:
1. 00_claim.md
2. 01_evidence.md
3. 02_red_opening.md, 02_blue_opening.md
4. 03_red_rebuttal.md, 03_blue_rebuttal.md  (if rebuttals = yes)
5. <canon_location>/debate_methodology.md
6. The relevant external_canon_*.md to verify any standard-form citations.

Write your verdict to 04_verdict.md using the Phase-4 template.

You may not split differences when one side has cited evidence and the other has not. Every verdict needs a Decisive evidence citation.

Source-of-truth precedence (binding): The user's Gauge-Theoretic VFE construction — `Attention/*.tex`, `CLAUDE.md`, `user_theory_summary.md`, in-repo derivations — is a work in progress under evaluation, not the source of truth. The standard external literature (information geometry, differential geometry, gauge theory, FEP / active inference, variational inference, transformer attention) is the source of truth. When the user's construction disagrees with the canonical form, the canonical form wins. A citation of the user's manuscript or CLAUDE.md as authority for the canonical form is a weak strike — discount it. The Decisive evidence must be an external canon citation, executed verification (`sympy`, finite-difference, test output), or a `path:line` reference to the actual code. The user is paying for errors to be found; a BLUE_WINS verdict that rests on the manuscript's own authority is malformed, and on close calls you should err toward RED_WINS or REMAND rather than absolving the construction on its own say-so.
```

Validation: verdict contains Outcome (one of RED_WINS, BLUE_WINS, REMAND, OUT_OF_SCOPE), Decisive evidence, Reasoning, Action. Re-dispatch once on failure.

## Phase 4-off — User adjudicates

If `judge=off`, main Claude writes `04_verdict.md`:

```markdown
# Verdict — <slug> (user-adjudicated)

The judge phase was disabled. Both sides' openings and rebuttals are above. Please declare a verdict.

## Red opening summary
<one-paragraph factual summary, no editorial framing>

## Blue opening summary
<one-paragraph factual summary>

## Red rebuttal summary
<one-paragraph factual summary — omit section if rounds=1>

## Blue rebuttal summary
<one-paragraph factual summary — omit section if rounds=1>

## Your verdict

(User to fill in.)
```

Main Claude must **not** weigh sides. Summaries are factual relays.

## Phase 5 — Action extraction

Read `04_verdict.md`. Extract the `Action` section into `05_action.md`:

```markdown
# Action — <slug>

**From verdict:** <RED_WINS | BLUE_WINS | REMAND | OUT_OF_SCOPE | user-adjudicated>

## Recommended action

<The action stated in the verdict.>

## Follow-up debates (if any)

<For REMAND or compound claims, list sub-claims that need their own debate.>
```

## Banned-phrase enforcement

After every agent dispatch, search the output file for:

Claude-isms (universal):
```
key insight, crucially, critically, notably, importantly,
it's worth noting, interestingly, fundamentally, in particular,
leverages, underscores
```

Debate-specific:
```
perhaps, it could be argued, one might suggest, both sides have a point
```

If matches found, re-dispatch the relevant agent with the explicit instruction to remove them. After one retry, proceed with a malformed note.

## Cost summary

| Configuration | Agent dispatches |
|---|---|
| Full (rounds=2, judge=on) | 5 (2 openings + 2 rebuttals + judge) |
| rounds=1, judge=on | 3 |
| rounds=2, judge=off | 4 |
| rounds=1, judge=off | 2 |

All agent dispatches are in ultrathink mode. Phases 0, 1, 4-off, 5 do not require ultrathink (orchestration only).

## Embedded fallback methodology (minimal)

Used only when no project canon is found. Pass this excerpt inline to each agent dispatch.

```
You are participating in a structured adversarial debate. The methodology:

Source-of-truth precedence (binding):
The user's own framework — manuscripts, project-internal notes (CLAUDE.md, design docs), in-repo derivations — is the claim under evaluation, not the canon. The standard external literature in the relevant field is the source of truth. When the user's construction disagrees with the canonical form, the canonical form wins. Establishing the canonical form requires an external citation (textbook section, paper equation, or primary-source fetch); "the user's document says X, therefore X" is a malformed argument. Apparent agreement with the canon must be verified, not assumed. Novel constructions require explicit derivation from canonical primitives, not a free pass. The user wants errors surfaced — sycophantic defense that treats the user's framework as self-authoritative defeats the purpose.

Roles:
- red-team: falsify the claim. Steelman first, then attack. Cite primary sources (external canon).
- blue-team: defend the claim by deriving it from external canon. State falsification conditions.
- debate-judge: weigh by external-canon evidence, declare a verdict with decisive citation.

Required output sections (Phase 2 opening): Steelman, Position, Evidence (≥1 external citation), Falsification conditions.
Required output sections (Phase 3 rebuttal): Concession, Core attack, Defense.
Required output sections (Phase 4 verdict): Outcome (RED_WINS|BLUE_WINS|REMAND|OUT_OF_SCOPE), Decisive evidence, Reasoning, Action.

Banned phrases: key insight, crucially, notably, importantly, it's worth noting, interestingly, fundamentally, in particular, leverages, underscores, perhaps, it could be argued, both sides have a point.

Citations:
- Theory/math claims: external textbook or paper citation, e.g., [Nakahara 2003 §10.3]. The user's own manuscript may be cited as the claim under evaluation, never as the standard.
- Code claims: file path with line number, e.g., src/file.py:127 (the in-repo code is canonical for what the code does; the in-repo theory is not canonical for what is correct).
- Math (formal): executed sympy session or finite-difference verification with concrete numbers.

A claim without an external citation is a weak strike. A claim that cites the user's own framework as the standard is also a weak strike. The judge counts strikes.
```

When the embedded fallback is in use, no project-specific citation discipline is applied — generic only.
