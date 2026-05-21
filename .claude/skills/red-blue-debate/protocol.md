# Red Team / Blue Team Debate Protocol

Detailed phase-by-phase protocol for the `red-blue-debate` skill. Two modes:

- **`panel=full`** (default, 2026-05-21 onward) — coordinator + 5 expert consultants per side, canon-cop validator between rounds, 3 first-pass judges + chief reconciler.
- **`panel=lite`** — original 1-red, 1-blue, 1-judge structure. See "Lite-mode protocol" section at the bottom.

The skill `SKILL.md` summarizes; this file is the full procedure.

## Arguments

```
/red-blue-debate <claim>
  [mode=theory|math|code|implementation]   default: inferred
  [panel=full|lite]                         default: full
  [rounds=1..5]                             default: 2
  [judging=panel|rubric|off]                default (panel=full): panel; default (panel=lite): rubric
  [experts=<comma-separated 5 tags>]        default: dynamic per side (panel=full only)
  [evidence=auto|paths:<glob>]              default: auto
```

If `<claim>` is empty, prompt the user once for the claim, then proceed.

## Source-of-truth precedence (applies to every phase, both modes)

The user's Gauge-Theoretic VFE construction is a **work in progress under evaluation**, not the source of truth. The source of truth is the standard external literature. When the user's construction disagrees with the canonical form, the canonical form wins.

This precedence rule is restated in every dispatch prompt below. Do not omit it.

## Canon-location resolution (run first)

Before any phase, resolve the canon location:

```
1. If `<cwd>/.claude/agents/vfe-knowledge/debate_methodology.md` exists → canon_location = `<cwd>/.claude/agents/vfe-knowledge/`.
2. Else if any `<cwd>/.claude/agents/*-knowledge/debate_methodology.md` exists → canon_location = that directory.
3. Else canon_location = "embedded" (use the minimal generic fallback at the bottom; pass excerpts inline to each agent dispatch).
```

Record `canon_location` at the top of `00_claim.md` so the agents know where to read from.

## Working directory

Create `<cwd>/docs/debates/YYYY-MM-DD-<slug>/` where `<slug>` is a kebab-case summary of the claim (max 40 chars). Use the current date.

If `<cwd>/docs/` does not exist, create it. If `<cwd>` is not a git repo, warn and use `<cwd>/debates/` instead.

# `panel=full` protocol (default)

## Phase 0 — Claim extraction (main Claude)

1. Parse the user-provided claim into a single declarative sentence.
2. If the claim is compound, identify the load-bearing proposition and note that sub-claims need separate debates.
3. Write `00_claim.md`:

```markdown
# Claim — <slug>

**Mode:** <theory|math|code|implementation>
**Panel:** full
**Rounds:** <1|2|3|4|5>
**Judging:** <panel|rubric|off>
**Experts override:** <none|comma-separated 5 tags>
**Evidence scope:** <auto|paths:<glob>>
**Canon location:** <absolute path or "embedded">

## Claim

<One declarative sentence.>

## User context

<Any additional context the user provided.>
```

## Phase 1 — Shared evidence pack (main Claude)

Same as before — main Claude assembles a neutral fact pack to `01_evidence.md`.

For `code` / `implementation` mode:
1. Identify the active entry point (e.g., `transformer/vfe/train_vfe.py`). If not given in context, ask the user.
2. Read the config dict at the top of that entry point.
3. Trace every relevant key through `BlockConfig.__post_init__`, `TrainingConfig.__post_init__`, `VFEConfig.__post_init__`, and any override logic. Record resolved values.
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

## Canon excerpts (initial — experts will extend during Phase 2)

- `[Author Year §X]` — <quoted form>

## What this evidence does NOT settle

<List of questions the evidence pack alone cannot answer.>
```

If `evidence=paths:<glob>` is given, constrain search to those paths.

## Phase 2 — Coordinated openings (parallel)

In a single message, dispatch both `debate-coordinator-red` and `debate-coordinator-blue` with the `Agent` tool.

**For debate-coordinator-red:**

```
ultrathink

You are dispatched as debate-coordinator-red in debate <slug>, Phase 2 (opening).

Working directory: <absolute path to docs/debates/<slug>>
Canon location: <absolute path, or "embedded">
Mode: <mode>
Round: opening
Experts override: <none or comma-separated 5 tags>

Read in order:
1. <working_dir>/00_claim.md
2. <working_dir>/01_evidence.md
3. <canon_location>/debate_methodology.md (especially the panel=full extensions)
4. <canon_location>/style_constraints.md
5. The relevant <canon_location>/external_canon_*.md

Your responsibilities:
Step 1 — Pick exactly 5 experts from the 10-expert roster (philosophy-of-science mandatory). Log to <working_dir>/02_red_panel_choice.md.
Step 2 — Dispatch all 5 experts in PARALLEL (single message, 5 Agent tool calls). Each dispatch begins with "ultrathink" on line 1. Each expert writes a memo to <working_dir>/memo_red_<expert>.md.
Step 3 — Merge the experts' "Newly-discovered canon" sections into <working_dir>/01b_extended_evidence.md (append if exists, create if not).
Step 4 — Synthesize the opening to <working_dir>/02_red_opening.md using the Phase-2 template in debate_methodology.md. Every memo must be cited or explicitly discounted.

Source-of-truth precedence (binding): The user's Gauge-Theoretic VFE construction is a work in progress under evaluation, not the source of truth. The standard external literature is the source of truth. Establishing the canonical form requires an external citation; "the manuscript says X, therefore X" is malformed and canon-cop will fire a strike on the synthesized output.
```

**For debate-coordinator-blue:**

```
ultrathink

You are dispatched as debate-coordinator-blue in debate <slug>, Phase 2 (opening).

Working directory: <absolute path>
Canon location: <absolute path>
Mode: <mode>
Round: opening
Experts override: <none or 5 tags>

(Same reading list as debate-coordinator-red.)

Your responsibilities:
Step 1 — Pick 5 experts (philosophy-of-science mandatory). Log to <working_dir>/02_blue_panel_choice.md.
Step 2 — Dispatch in parallel. Each expert writes memo to <working_dir>/memo_blue_<expert>.md.
Step 3 — Merge harvested canon to <working_dir>/01b_extended_evidence.md.
Step 4 — Synthesize defense to <working_dir>/02_blue_opening.md.

Mandate: defend the claim by steelmanning it. State falsification conditions. Cite external canon, not the manuscript-as-authority. If the panel's collective evidence cannot defend the claim, concede honestly — sycophantic defense is malformed.

Source-of-truth precedence (binding): Same as red coordinator above. Manuscript-as-authority citations get a double-weighted strike on the blue side; that's blue's signature failure mode.
```

After both coordinators return, validate:
- `02_red_opening.md` and `02_blue_opening.md` exist.
- `02_red_panel_choice.md` and `02_blue_panel_choice.md` exist, each listing exactly 5 experts including philosophy-of-science.
- 5 `memo_red_*.md` files and 5 `memo_blue_*.md` files exist.
- `01b_extended_evidence.md` exists with newly-discovered canon.

If any check fails, re-dispatch the relevant coordinator once with a corrective prompt naming the missing element. If the second attempt fails, append a malformed-output note and proceed.

## Phase 2.5 — Canon-cop validation (parallel)

In a single message, dispatch `debate-canon-cop` twice in parallel — once for each side's opening.

**For each canon-cop dispatch:**

```
ultrathink

You are dispatched as debate-canon-cop in debate <slug>, Phase 2.5, side=<red|blue>.

Working directory: <absolute path>
Canon location: <absolute path>
Target file: <working_dir>/02_<side>_opening.md
Output file: <working_dir>/02_canoncop_<side>.md

Step 1 — Run python <skill_dir>/canon_cop_validator.py --target <target> --bibliography <canon>/external_bibliography.md --canon-dir <canon>. Read the JSON output.
Step 2 — LLM pass for subtle phrasing (implicit manuscript-as-authority, reasoning-by-construction circularity, hand-wave-with-citation, wrong-domain citation).
Step 3 — Apply the soft cap: ≥3 total strikes → write strike list, signal MANDATORY_REWRITE; 0–2 strikes → RECORD.

Output JSON-plus-prose report to the target output file. See your agent definition for the exact format.
```

After both canon-cop dispatches return:
- If either signaled `MANDATORY_REWRITE`, re-dispatch the corresponding coordinator once with the strike list attached to the prompt: "Your previous opening had N source-of-truth violations: [paste strike list]. Rewrite <output_file> without them. Residual strikes after rewrite are recorded but not re-thrown back."
- After rewrite, re-run canon-cop once on the new opening. Final strikes recorded; debate continues.

## Phase 3 — Coordinated rebuttals (parallel)

Skip this phase if `rounds=1`.

Dispatch both coordinators in parallel. Each reads only the OPPOSING opening (not its own — anti-self-anchoring).

**For debate-coordinator-red (rebuttal):**

```
ultrathink

You are dispatched as debate-coordinator-red in debate <slug>, Phase 3 (rebuttal).

Working directory: <absolute path>
Canon location: <absolute path>
Mode: <mode>
Round: rebuttal
Experts override: <none or 5 tags>

Read:
1. <working_dir>/00_claim.md
2. <working_dir>/01_evidence.md
3. <working_dir>/01b_extended_evidence.md
4. <working_dir>/02_blue_opening.md  (the opposing opening — do NOT read 02_red_opening.md)
5. <canon_location>/debate_methodology.md

Your responsibilities (same as Phase 2, but now in rebuttal):
Step 1 — Pick 5 experts (philosophy-of-science mandatory). You MAY pick a different panel than your Phase-2 panel based on what the opposing opening raised. Log to <working_dir>/03_red_panel_choice.md.
Step 2 — Dispatch in parallel. Memos go to <working_dir>/memo_red_<expert>_rebuttal.md.
Step 3 — Append newly-discovered canon to <working_dir>/01b_extended_evidence.md (under a Phase-3 section).
Step 4 — Synthesize rebuttal to <working_dir>/03_red_rebuttal.md using the Phase-3 template: Concession, Core attack, Defense.

Required: at least one concession, a core attack with citation, a defense with citation.

Source-of-truth precedence (binding, same as Phase 2): If blue's opening cited the user's manuscript as authority, that is a malformed argument and should be your core attack — demand an external citation.
```

**For debate-coordinator-blue (rebuttal):**

```
ultrathink

You are dispatched as debate-coordinator-blue in debate <slug>, Phase 3 (rebuttal).

(Same reading list, with 02_red_opening.md as opposing opening; do NOT read 02_blue_opening.md.)

Same 4-step responsibility chain, with memos to memo_blue_<expert>_rebuttal.md and synthesis to 03_blue_rebuttal.md.

If red's attack lands and the canon contradicts the user's construction, concede on the evidence. Honest concession beats performative defense.
```

Validate as in Phase 2 (panel choices, memos, opening synthesis exist; 5 experts each; philosophy-of-science present).

## Phase 3.5 — Canon-cop validation (rebuttals)

Same as Phase 2.5, with `03_<side>_rebuttal.md` as target and `03_canoncop_<side>.md` as output.

## Phase 3b — Sur-rebuttals (rounds≥3 only)

If `rounds=2`, skip.

Dispatch both coordinators in parallel. Each reads the opposing rebuttal (not its own — anti-self-anchoring). Memos suffix `_surrebuttal`. Synthesis goes to `03b_<side>_surrebuttal.md`.

**Sur-rebuttal mandate (binding):**
1. Respond to the opposing rebuttal, not the opposing opening.
2. No new attack vectors.
3. Engage with the opposing rebuttal's strongest concession or strongest attack.
4. One-page maximum.

See `debate_methodology.md` "Sur-rebuttal mandate" section.

Phase 3b.5: canon-cop on sur-rebuttals (same pattern).

For `rounds=4`, `rounds=5`: repeat the pattern with Phase 3c, 3d. Diminishing returns past round 3.

## Phase 4a — First-pass judges (parallel)

Skip if `judging=off` (jump to Phase 4-off below). If `judging=rubric` (single judge with structured rubric), skip 4a and use 4b-rubric pattern.

If `judging=panel` (default), dispatch all three first-pass judges in a single message (3 parallel `Agent` tool calls):

**For debate-judge-canon-strict:**

```
ultrathink

You are dispatched as debate-judge-canon-strict in debate <slug>.

Working directory: <absolute path>
Canon location: <absolute path>
Mode: <mode>
Rounds completed: <2|3|4|5>

Read:
1. 00_claim.md
2. 01_evidence.md
3. 01b_extended_evidence.md
4. All 02_*_opening.md, 03_*_rebuttal.md, 03b_*_surrebuttal.md (etc.)
5. All memo_*.md
6. All *_canoncop_*.md
7. <canon_location>/debate_methodology.md
8. <canon_location>/external_bibliography.md
9. The relevant <canon_location>/external_canon_*.md

Write your verdict to 04_verdict_canon.md using your agent's rubric template.

Your weighting: external textbook/paper citations (verified) 3x; in-repo evidence 1x; Attention/*.tex or CLAUDE.md cited as authority = -2; canon-cop strikes = -1 each.

You may not split differences when one side has cited verified external canon and the other has not.
```

**For debate-judge-code-truth:**

```
ultrathink

You are dispatched as debate-judge-code-truth in debate <slug>.

(Same reading list.)

Step 1 — Re-trace the active config yourself (don't trust the openings).
Step 2 — For every path:line claim, verify reachability under the active config.
Step 3 — Write your verdict to 04_verdict_code.md using your agent's rubric template.

Your weighting: verified path:line under active config 3x; test outputs 3x; unverified path:line 1x; comments/docstrings 0; CLAUDE.md as authority for code behavior -1.
```

**For debate-judge-scope:**

```
ultrathink

You are dispatched as debate-judge-scope in debate <slug>.

(Same reading list.)

Step 1 — Check claim well-formedness, falsifiability, vocabulary anchoring.
Step 2 — Detect claim drift across rounds.
Step 3 — Detect false dichotomies / equivocations / scope leakage.
Step 4 — Write your verdict to 04_verdict_scope.md.

You have special standing for OUT_OF_SCOPE and REMAND outcomes — the chief defers to you on those.
```

After all three return, validate that each verdict exists and contains the required rubric sections (per each judge's agent definition). Re-dispatch any malformed verdict once.

## Phase 4b — Chief reconciliation

Dispatch `debate-chief-judge` after the three first-pass verdicts are written:

```
ultrathink

You are dispatched as debate-chief-judge in debate <slug>.

Working directory: <absolute path>
Canon location: <absolute path>
Mode: <mode>

Read:
1. 00_claim.md, 01_evidence.md, 01b_extended_evidence.md
2. All openings, rebuttals, sur-rebuttals
3. 04_verdict_canon.md, 04_verdict_code.md, 04_verdict_scope.md
4. All *_canoncop_*.md
5. <canon_location>/debate_methodology.md

Apply the 5-rule reconciliation algorithm (in order, stop at first rule that fires):
Rule 1 — Scope override for OUT_OF_SCOPE
Rule 2 — Scope override for REMAND on equivocation
Rule 3 — Majority outcome (2 of 3 agree)
Rule 4 — Mode-weighted tiebreak (theory/math → canon-strict; code/implementation → code-truth)
Rule 5 — Last resort REMAND

Write binding verdict to 04_verdict.md using the chief-judge template in debate_methodology.md.

You MAY NOT split differences, declare a fourth outcome, override a first-pass judge on their domain without explicit rule citation, or add new evidence.
```

Validate: `04_verdict.md` exists with First-pass verdicts table, Reconciliation rule applied, Decisive evidence, Outcome, Reasoning, Action.

## Phase 4-off — User adjudicates (judging=off)

If `judging=off`, main Claude writes `04_verdict.md`:

```markdown
# Verdict — <slug> (user-adjudicated)

The judge phase was disabled. Both sides' coordinator outputs are above. Please declare a verdict.

## Red opening summary
<one-paragraph factual summary of 02_red_opening.md, no editorial framing>

## Blue opening summary
<one-paragraph factual summary>

(Continue for each round that ran.)

## Your verdict

(User to fill in.)
```

Main Claude must **not** weigh sides. Summaries are factual relays.

## Phase 4b-rubric — Single judge (judging=rubric)

If `judging=rubric`, dispatch `debate-judge-canon-strict` ONLY (no panel, no chief). The judge writes directly to `04_verdict.md` using its own rubric. Cheaper but less robust to judge-style bias.

## Phase 5 — Action extraction (main Claude)

Read `04_verdict.md`. Extract the `Action` section into `05_action.md`:

```markdown
# Action — <slug>

**From verdict:** <RED_WINS | BLUE_WINS | REMAND | OUT_OF_SCOPE | user-adjudicated>
**Reconciliation rule (panel=full):** <Rule 1|2|3|4|5>

## Recommended action

<The action stated in the verdict.>

## Follow-up debates (if any)

<For REMAND or compound claims, list sub-claims that need their own debate.>
```

## Banned-phrase enforcement (universal, both modes)

After every agent dispatch, search the output file for:

Claude-isms:
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

## Cost summary (panel=full default)

| Configuration | Coordinator | Expert | Canon-cop | Judge | Total |
|---|---|---|---|---|---|
| rounds=1, judging=panel | 2 | 10 | 2 | 4 | 18 |
| rounds=2, judging=panel (default) | 4 | 20 | 4 | 4 | 32 |
| rounds=2, judging=rubric | 4 | 20 | 4 | 1 | 29 |
| rounds=3, judging=panel | 6 | 30 | 6 | 4 | 46 |
| rounds=5, judging=panel | 10 | 50 | 10 | 4 | 74 |

Plus 1–2 canon-cop rewrites per debate on average (≈30% trigger rate). All agent dispatches in ultrathink mode.

# `panel=lite` protocol (backwards compatible)

When `panel=lite` is specified, the skill reverts to the original 5-phase protocol:

| Phase | Actor | Action |
|---|---|---|
| 0 | main Claude | `00_claim.md` |
| 1 | main Claude | `01_evidence.md` |
| 2 | red-team + blue-team (parallel, ultrathink) | `02_red_opening.md`, `02_blue_opening.md` |
| 3 | red-team + blue-team (parallel, ultrathink) | `03_red_rebuttal.md`, `03_blue_rebuttal.md` (skip if rounds=1) |
| 4 | debate-judge (ultrathink) OR user | `04_verdict.md` |
| 5 | main Claude | `05_action.md` |

Lite-mode dispatch templates are exactly as they were before 2026-05-21 — see the pre-2026-05-21 `protocol.md` in git history for the original wording, or just use the standard templates with the source-of-truth precedence rule restated.

Lite-mode does not use canon-cop, expert consultants, or judge panels.

| rounds | judge | total |
|---|---|---|
| 1 | off | 2 |
| 1 | rubric/panel (single judge) | 3 |
| 2 | off | 4 |
| 2 | rubric/panel (single judge) | 5 |

# Embedded fallback methodology (minimal — both modes)

Used only when no project canon is found. Pass this excerpt inline to each agent dispatch.

```
You are participating in a structured adversarial debate. The methodology:

Source-of-truth precedence (binding):
The user's own framework — manuscripts, project-internal notes (CLAUDE.md, design docs), in-repo derivations — is the claim under evaluation, not the canon. The standard external literature in the relevant field is the source of truth. When the user's construction disagrees with the canonical form, the canonical form wins. Establishing the canonical form requires an external citation (textbook section, paper equation, or primary-source fetch); "the user's document says X, therefore X" is a malformed argument. Apparent agreement with the canon must be verified, not assumed. Novel constructions require explicit derivation from canonical primitives, not a free pass.

Roles:
- coordinator-red / coordinator-blue: dispatch 5 expert consultants (philosophy-of-science mandatory), synthesize opening/rebuttal/sur-rebuttal from their memos. (Or in panel=lite: red-team / blue-team write directly.)
- expert consultants: write a memo from their disciplinary lens with at least 3 external citations.
- canon-cop: validate that the synthesis does not cite the user's framework as authority.
- judges: weigh by external-canon evidence, declare a verdict with decisive citation.

Required output sections (Phase 2 opening): Steelman, Position, Evidence (≥3 external citations), Falsification conditions.
Required output sections (Phase 3 rebuttal): Concession, Core attack, Defense.
Required output sections (Phase 4 verdict): Outcome (RED_WINS|BLUE_WINS|REMAND|OUT_OF_SCOPE), Decisive evidence, Reasoning, Action.

Banned phrases: key insight, crucially, notably, importantly, it's worth noting, interestingly, fundamentally, in particular, leverages, underscores, perhaps, it could be argued, both sides have a point.

Citations:
- Theory/math claims: external textbook or paper citation, e.g., [Nakahara 2003 §10.3]. The user's own manuscript may be cited as the claim under evaluation, never as the standard.
- Code claims: file path with line number, e.g., src/file.py:127 (the in-repo code is canonical for what the code does; the in-repo theory is not canonical for what is correct).
- Math (formal): executed sympy session or finite-difference verification with concrete numbers.

A claim without an external citation is a weak strike. A claim that cites the user's own framework as the standard is also a weak strike.
```

When the embedded fallback is in use, no project-specific citation discipline is applied — generic only.
