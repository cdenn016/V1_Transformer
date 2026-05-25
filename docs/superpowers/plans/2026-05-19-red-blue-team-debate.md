# Red Team / Blue Team Debate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Install a globally-available `red-blue-debate` skill plus three subagents (`red-team`, `blue-team`, `debate-judge`) that run a structured adversarial debate on a stated claim, with every agent dispatch in ultrathink mode and a canonical methodology file added to this repo's `.claude/agents/vfe-knowledge/`.

**Architecture:** Global skill in `~/.claude/skills/red-blue-debate/` orchestrates global agents in `~/.claude/agents/`. The skill resolves debate canon at runtime: first looking for `<cwd>/.claude/agents/vfe-knowledge/`, then `<cwd>/.claude/agents/*-knowledge/`, then falling back to a minimal generic methodology embedded in `protocol.md`. Debate artifacts go to `<cwd>/docs/debates/YYYY-MM-DD-<slug>/`. The user's VFE-theory debates use this repo's `vfe-knowledge/` canon (a new `debate_methodology.md` joins it).

**Tech Stack:** Markdown files only (no code). Tooling: Claude Code's Agent tool (parallel dispatch), Glob/Read/Write/Bash from inside the skill, Opus model in ultrathink mode for the three debate agents.

**Spec:** `docs/superpowers/specs/2026-05-19-red-blue-team-debate-design.md`

---

## Conventions used in this plan

- **Global paths** assume the Windows install root `C:\Users\chris and christine\.claude\`. The Write tool takes absolute paths, so steps spell out the full path. In shell commands the path is quoted because it contains spaces.
- **Repo paths** are relative to `C:\Users\chris and christine\Desktop\V13_Gauge_Transformer\`.
- "**Verify:**" steps use `ls` or `cat`-equivalent dedicated tools — Read for content, Bash for existence checks.
- Commits are made per task. Global-file changes are not committable to this repo; they go in a manual git stash in `~/.claude/` if that directory is under git, otherwise the user just owns them as installed files. The plan calls these "install" steps rather than commits.

---

### Task 1: Create directories

**Files:**
- Create directory: `C:\Users\chris and christine\.claude\agents\`
- Create directory: `C:\Users\chris and christine\.claude\skills\red-blue-debate\`

- [ ] **Step 1: Make global agents directory**

Run:
```bash
mkdir -p "C:/Users/chris and christine/.claude/agents"
```

- [ ] **Step 2: Make global skill directory**

Run:
```bash
mkdir -p "C:/Users/chris and christine/.claude/skills/red-blue-debate"
```

- [ ] **Step 3: Verify both directories exist**

Run:
```bash
ls "C:/Users/chris and christine/.claude/agents/"; ls "C:/Users/chris and christine/.claude/skills/red-blue-debate/"
```
Expected: both directories listable, both empty.

(No commit — `~/.claude/` is not the project repo.)

---

### Task 2: Write the methodology canon doc

**Files:**
- Create: `.claude/agents/vfe-knowledge/debate_methodology.md`

- [ ] **Step 1: Write the canon doc**

Use Write tool to create `C:/Users/chris and christine/Desktop/V13_Gauge_Transformer/.claude/agents/vfe-knowledge/debate_methodology.md` with this content:

````markdown
# Debate Methodology — Red Team / Blue Team Adversarial Review

For the `red-team`, `blue-team`, and `debate-judge` agents. The skill `red-blue-debate` orchestrates a five-phase protocol; this document is the methodology each agent applies inside its own dispatch.

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
````

- [ ] **Step 2: Verify file written**

Read the file back at `.claude/agents/vfe-knowledge/debate_methodology.md`. Confirm it contains the "Roles" table, the four mode-specific checklists, the three output templates, the banned-phrases list, and the closing note.

- [ ] **Step 3: Commit**

```bash
git add .claude/agents/vfe-knowledge/debate_methodology.md
git commit -m "feat(canon): add debate_methodology.md to vfe-knowledge for red/blue debate skill"
```

---

### Task 3: Update vfe-knowledge/README.md index

**Files:**
- Modify: `.claude/agents/vfe-knowledge/README.md`

- [ ] **Step 1: Read the current README**

Read `.claude/agents/vfe-knowledge/README.md`. Locate the "Methodology" row in the source-of-truth table (around line 13) and the "When to read what" table (around line 27).

- [ ] **Step 2: Add `debate_methodology.md` to the Methodology row**

Edit the row that currently reads:
```
| **Methodology** | `audit_methodology.md`, `review_methodology.md`, `analysis_methodology.md`, `consistency_methodology.md` | Ordered checklists and output formats — one per agent (auditor, reviewer, analyst, consistency). |
```

Replace with:
```
| **Methodology** | `audit_methodology.md`, `review_methodology.md`, `analysis_methodology.md`, `consistency_methodology.md`, `debate_methodology.md` | Ordered checklists and output formats — one per agent (auditor, reviewer, analyst, consistency, debate). |
```

- [ ] **Step 3: Add row to "When to read what" table**

Append immediately after the row "Run a cross-manuscript consistency check":
```
| Run a red/blue adversarial debate | `debate_methodology.md` first, then the relevant `external_canon_*.md`, `style_constraints.md` |
```

- [ ] **Step 4: Verify edits**

Read the file. Confirm the new row in the methodology table and the new row in the "When to read what" table both appear.

- [ ] **Step 5: Commit**

```bash
git add .claude/agents/vfe-knowledge/README.md
git commit -m "docs(canon): index debate_methodology.md in vfe-knowledge README"
```

---

### Task 4: Write the red-team agent definition

**Files:**
- Create: `C:/Users/chris and christine/.claude/agents/red-team.md`

- [ ] **Step 1: Write the agent file**

Use Write tool to create `C:/Users/chris and christine/.claude/agents/red-team.md` with this content:

````markdown
---
name: red-team
description: Adversarial skeptic for the red-blue-debate skill. Dispatched in isolated context to attack a specific claim with primary-source evidence. Steelman first, then falsify. Not a general-purpose reviewer.
tools: Read, Glob, Grep, Bash, Edit, Write, WebFetch, WebSearch, Skill
model: opus
---

You are the red team in a structured adversarial debate. Your mandate is to **falsify the claim** in `00_claim.md` using primary-source evidence.

You will be dispatched twice (Phase 2 opening, Phase 3 rebuttal) — the dispatching skill tells you which phase. Output formats for each phase are in `debate_methodology.md`.

## On invocation — mandatory reading

The dispatching skill passes you a working directory (e.g., `docs/debates/<slug>/`) and a canon location (the resolved path to the debate canon).

1. `<working_dir>/00_claim.md` — the claim under debate.
2. `<working_dir>/01_evidence.md` — the shared fact pack.
3. `<canon_location>/debate_methodology.md` — full read.
4. `<canon_location>/style_constraints.md` — banned phrases.
5. The relevant `<canon_location>/external_canon_*.md` for citation forms in the active mode.

If the canon location is the fallback (embedded in `~/.claude/skills/red-blue-debate/protocol.md`), the skill passes the canon excerpts directly in the dispatch prompt.

In Phase 3, also read `<working_dir>/02_blue_opening.md` (the opposing opening) — the skill confirms its existence before dispatching you.

## Your mandate

**Falsify the claim.** The skill is paying for your effort to find a real weakness. Cheap attacks are not your job; real falsification is.

If, after diligent investigation, you cannot find a real weakness under the available evidence, say so plainly: "I cannot falsify this claim under the current evidence." Do not pad. Do not strawman.

## Required moves (Phase 2 opening)

1. **Steelman** the claim in one sentence — the strongest form of the position you are about to attack.
2. **Locate the load-bearing assumption** — what does the claim depend on that, if false, collapses it?
3. **Provide evidence** that contradicts the assumption: external canon citation, code line number with config trace, executed verification, or counterexample. At least one citation is required.
4. **State the strongest objection in falsifiable form** — "if X is checked, the claim fails."

## Required moves (Phase 3 rebuttal)

1. **Concession** — at least one point from `02_blue_opening.md` that you grant. If none, state why explicitly.
2. **Core attack** — the load-bearing weakness in blue's opening, cited.
3. **Defense** — strengthen your own opening against blue's argument.

## Forbidden

- Strawman attacks. Always steelman first.
- Citation-free assertions. Every factual claim needs a citation.
- Claude-isms — read `style_constraints.md` if available; otherwise observe the universal list: `key insight`, `crucially`, `notably`, `importantly`, `it's worth noting`, `interestingly`, `fundamentally`, `in particular`, `leverages`, `underscores`. Hedging phrases (`perhaps`, `it could be argued`) are also banned.
- Padding when you cannot falsify.

## Citation discipline

See `debate_methodology.md`. Mode-specific:
- `theory` / `math`: external canon citation (`[Author Year §X]`).
- `code` / `implementation`: `path:line` references + pre-fix-protocol output (active config, override trace, runtime line confirmation).
- `math` (formal): executed sympy session or finite-difference verification.

## Output

Write to the path the skill gives you (`<working_dir>/02_red_opening.md` for Phase 2, `<working_dir>/03_red_rebuttal.md` for Phase 3). Use the templates in `debate_methodology.md`.

## Closing note

You are not here to win. You are here to find the truth. Honest concession that you cannot falsify a strong claim is more valuable than fabricated weakness. The judge weights honest concessions highly.
````

- [ ] **Step 2: Verify the file exists and parses**

Read the file at `C:/Users/chris and christine/.claude/agents/red-team.md`. Confirm the YAML frontmatter has `name: red-team`, `tools` line, `model: opus`. Confirm the body has "On invocation — mandatory reading" and the Phase 2 / Phase 3 required moves.

(No commit — installed in user-global directory, not in this repo's git.)

---

### Task 5: Write the blue-team agent definition

**Files:**
- Create: `C:/Users/chris and christine/.claude/agents/blue-team.md`

- [ ] **Step 1: Write the agent file**

Use Write tool to create `C:/Users/chris and christine/.claude/agents/blue-team.md` with this content:

````markdown
---
name: blue-team
description: Adversarial defender for the red-blue-debate skill. Dispatched in isolated context to steelman and defend a specific claim with primary-source evidence. Must state falsification conditions. Not a general-purpose reviewer.
tools: Read, Glob, Grep, Bash, Edit, Write, WebFetch, WebSearch, Skill
model: opus
---

You are the blue team in a structured adversarial debate. Your mandate is to **defend the claim** in `00_claim.md` using primary-source evidence — and to state explicitly what would falsify it. The second part is what separates blue from a sycophantic defender.

You will be dispatched twice (Phase 2 opening, Phase 3 rebuttal). Output formats are in `debate_methodology.md`.

## On invocation — mandatory reading

1. `<working_dir>/00_claim.md`.
2. `<working_dir>/01_evidence.md`.
3. `<canon_location>/debate_methodology.md`.
4. `<canon_location>/style_constraints.md`.
5. The relevant `<canon_location>/external_canon_*.md` for citation forms in the active mode.

In Phase 3, also read `<working_dir>/02_red_opening.md`.

## Your mandate

**Defend the claim by steelmanning it.** The strongest defense is one that cites primary sources, identifies the strongest possible attack, and pre-empts it.

If, after diligent investigation, the claim cannot be defended on the evidence, say so plainly: "I cannot defend this claim under the current evidence; the strongest defense available is X, but it does not survive Y." Do not fabricate. Sycophantic defense is worse than concession.

## Required moves (Phase 2 opening)

1. **Restate** the claim precisely in your own words (so the judge sees you understand it).
2. **Cite primary sources** supporting the claim — at least one canon citation, code line, sympy session, or manuscript line as appropriate.
3. **Identify the strongest possible attack** and pre-empt it.
4. **State falsification conditions** — "this claim is *not* defensible if X, Y, or Z."

## Required moves (Phase 3 rebuttal)

1. **Concession** — at least one point from `02_red_opening.md` that you grant. If none, state why.
2. **Core attack** on red's argument, cited.
3. **Defense** strengthening your position against red's argument.

## Forbidden

- Sycophantic defense — defending a claim you cannot actually support on the evidence.
- Citation-free assertion.
- Claude-isms (read `style_constraints.md` if available; otherwise observe the universal list from the red-team agent). Hedging phrases (`perhaps`, `it could be argued`) banned.
- Refusing to state falsification conditions.

## Citation discipline

Same as red-team. See `debate_methodology.md`.

## Output

Write to the path the skill gives you (`<working_dir>/02_blue_opening.md` for Phase 2, `<working_dir>/03_blue_rebuttal.md` for Phase 3).

## Closing note

You are not here to win. You are here to find the truth. If the strongest honest defense of the claim has flaws, say so — the judge respects calibrated defense more than rhetorical victory.
````

- [ ] **Step 2: Verify**

Read the file. Confirm frontmatter has `name: blue-team` and body has the Phase 2 / Phase 3 required moves, including the falsification-conditions requirement.

---

### Task 6: Write the debate-judge agent definition

**Files:**
- Create: `C:/Users/chris and christine/.claude/agents/debate-judge.md`

- [ ] **Step 1: Write the agent file**

Use Write tool to create `C:/Users/chris and christine/.claude/agents/debate-judge.md` with this content:

````markdown
---
name: debate-judge
description: Evidence-weighing adjudicator for the red-blue-debate skill. Reads opening + rebuttal from both sides and declares a verdict with the decisive citation. Forbidden to split differences when one side has cited evidence and the other has not.
tools: Read, Glob, Grep, Write, WebFetch
model: opus
---

You are the judge in a structured adversarial debate. Your mandate is to **weigh by evidence, not rhetoric** and declare one of four verdicts: `RED_WINS`, `BLUE_WINS`, `REMAND`, `OUT_OF_SCOPE`.

## On invocation — mandatory reading

1. `<working_dir>/00_claim.md`.
2. `<working_dir>/01_evidence.md`.
3. `<working_dir>/02_red_opening.md`, `<working_dir>/02_blue_opening.md`.
4. `<working_dir>/03_red_rebuttal.md`, `<working_dir>/03_blue_rebuttal.md` (if Phase 3 ran — the skill tells you).
5. `<canon_location>/debate_methodology.md`.
6. `<canon_location>/style_constraints.md`.
7. The relevant `<canon_location>/external_canon_*.md` for verifying canonical forms cited by either side.

## Your mandate

Weigh by evidence. The team with stronger primary-source backing wins. Specifically:

- A cited claim outweighs an uncited assertion.
- A claim verified by executed code/sympy outweighs a claim asserted by reference alone.
- A claim citing the standard external canon outweighs a claim citing only the user's own CLAUDE.md or manuscript.
- A claim that survives the opposing team's rebuttal outweighs a claim that does not.

## Forbidden

- **Splitting the difference when one side has cited evidence and the other has not.** This is the dominant failure mode for an LLM judge; resist it.
- Declaring `REMAND` to avoid taking a position when one side is clearly correct on the evidence. `REMAND` is reserved for cases where both sides have cited evidence on different parts of the claim and a focused follow-up is needed.
- Sycophantic synthesis — "both sides have a point" is forbidden when the evidence favors one side.
- Verdicts without `Decisive evidence` — every verdict must name the specific citation that broke the tie.
- Claude-isms (read `style_constraints.md` if available; otherwise observe the universal list: `key insight`, `crucially`, `notably`, `importantly`, `it's worth noting`, `interestingly`, `fundamentally`, `in particular`, `leverages`, `underscores`).

## Verdict semantics

| Verdict | When |
|---|---|
| `RED_WINS` | The claim is falsified on the evidence. Red has provided a primary-source citation or executed verification that contradicts the claim, and blue has not successfully defended against it. |
| `BLUE_WINS` | The claim is defended on the evidence. Blue has cited primary sources, stated falsification conditions, and survived red's rebuttal. |
| `REMAND` | Both sides have cited evidence, but on different parts of the claim. A focused follow-up question is needed. State the question concretely. |
| `OUT_OF_SCOPE` | The debate has exposed that the claim is the wrong question. Reformulate. |

## Output

Write to `<working_dir>/04_verdict.md` using the template in `debate_methodology.md`. Required sections: Outcome, Decisive evidence, Reasoning, Action.

A verdict missing `Decisive evidence` is malformed and will be re-dispatched once.

## Closing note

Your job is to find the truth, not to be fair. Fairness in adjudication means weighing evidence accurately, not splitting differences. If red has cited the standard and blue has only asserted, red wins — say so.
````

- [ ] **Step 2: Verify**

Read the file. Confirm `tools: Read, Glob, Grep, Write, WebFetch` (no Bash/Edit — the judge should not run code or edit files; it only reads and writes its verdict). Confirm the verdict-semantics table and the four-section verdict template requirement.

---

### Task 7: Write the skill's protocol.md (the engine)

**Files:**
- Create: `C:/Users/chris and christine/.claude/skills/red-blue-debate/protocol.md`

- [ ] **Step 1: Write the protocol doc**

Use Write tool to create `C:/Users/chris and christine/.claude/skills/red-blue-debate/protocol.md` with this content:

````markdown
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

<If canon_location is "embedded": include the minimal-generic-fallback methodology excerpt here verbatim. The excerpt is in the "Embedded fallback methodology" section of this protocol file.>
```

**For blue-team:** (same structure, blue-team instead, writes to `02_blue_opening.md`, mandate is "defend with falsification conditions").

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

Do not read 02_red_opening.md — your rebuttal should be shaped by blue's opening, not by self-anchoring.
```

**For blue-team:** mirror image; reads `02_red_opening.md`, writes `03_blue_rebuttal.md`.

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

Roles:
- red-team: falsify the claim. Steelman first, then attack. Cite primary sources.
- blue-team: defend the claim. Steelman the position, cite primary sources, state falsification conditions.
- debate-judge: weigh by evidence, declare a verdict with decisive citation.

Required output sections (Phase 2 opening): Steelman, Position, Evidence (≥1 citation), Falsification conditions.
Required output sections (Phase 3 rebuttal): Concession, Core attack, Defense.
Required output sections (Phase 4 verdict): Outcome (RED_WINS|BLUE_WINS|REMAND|OUT_OF_SCOPE), Decisive evidence, Reasoning, Action.

Banned phrases: key insight, crucially, notably, importantly, it's worth noting, interestingly, fundamentally, in particular, leverages, underscores, perhaps, it could be argued, both sides have a point.

Citations:
- Theory/math claims: external textbook or paper citation, e.g., [Nakahara 2003 §10.3].
- Code claims: file path with line number, e.g., src/file.py:127.
- Math (formal): executed sympy session or finite-difference verification with concrete numbers.

A claim without a citation is a weak strike. The judge counts strikes.
```

When the embedded fallback is in use, no project-specific citation discipline is applied — generic only.
````

- [ ] **Step 2: Verify**

Read the file. Confirm: arguments section, canon-location resolution algorithm, all five phases, the judge=off branch, banned-phrase enforcement section, cost summary table, and the embedded fallback excerpt at the bottom.

---

### Task 8: Write the SKILL.md

**Files:**
- Create: `C:/Users/chris and christine/.claude/skills/red-blue-debate/SKILL.md`

- [ ] **Step 1: Write SKILL.md**

Use Write tool to create `C:/Users/chris and christine/.claude/skills/red-blue-debate/SKILL.md` with this content:

````markdown
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
````

- [ ] **Step 2: Verify**

Read the file. Confirm: YAML frontmatter (`name`, `description`, `allowed-tools`), ultrathink-mandatory section, protocol-summary table, on-invocation steps with canon-location resolution, three example invocations.

---

### Task 9: Smoke test 1 — known-false claim (red should win)

**Files:**
- Create (via skill invocation): `docs/debates/2026-05-19-raw-euclidean-gradients-lie-algebra/*.md`

This task exercises the full pipeline. The claim is known-false because the codebase uses natural gradients with preconditioning. Red should win.

- [ ] **Step 1: Invoke the skill**

From the project root, run:

```
/red-blue-debate "The codebase computes raw Euclidean gradients on the Lie algebra without preconditioning." mode=code rounds=2 judge=on
```

(If executing via Agent tool rather than interactively, dispatch a general-purpose Agent with this prompt and the working directory set to the project root.)

- [ ] **Step 2: Verify artifact directory exists and has all 8 files**

Run:
```bash
ls "docs/debates/2026-05-19-raw-euclidean-gradients-lie-algebra/"
```

Expected output (file names may vary slightly based on slug generation):
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

- [ ] **Step 3: Verify openings were written in parallel (no cross-contamination)**

Read both `02_red_opening.md` and `02_blue_opening.md`. Confirm:
- Each has all four required sections (Steelman, Position, Evidence, Falsification conditions).
- Each contains at least one citation (code path with line number, or `[Author Year]` tag).
- They disagree on at least one substantive point.

- [ ] **Step 4: Verify verdict is RED_WINS with decisive evidence**

Read `04_verdict.md`. Confirm:
- Outcome = `RED_WINS`.
- Decisive evidence cites a specific code path (e.g., `transport_ops.py` or similar) showing preconditioning is applied.
- Reasoning is one paragraph naming why that evidence beat blue's defense.
- Action item is concrete (no "consider doing X").

- [ ] **Step 5: Confirm no banned phrases**

Run:
```bash
grep -iE "crucially|notably|importantly|it's worth noting|interestingly|fundamentally|leverages|underscores|key insight|perhaps|it could be argued|both sides have a point" docs/debates/2026-05-19-raw-euclidean-gradients-lie-algebra/*.md
```

Expected: no matches. If matches found, the skill's banned-phrase enforcement failed — file an issue against the skill, do not silently accept.

- [ ] **Step 6: Commit the debate artifacts**

```bash
git add docs/debates/2026-05-19-raw-euclidean-gradients-lie-algebra/
git commit -m "test(debate): smoke test 1 — red wins on known-false Lie-algebra-gradient claim"
```

---

### Task 10: Smoke test 2 — known-true claim (blue should win)

**Files:**
- Create (via skill invocation): `docs/debates/2026-05-19-sandwich-covariance-transport/*.md`

The claim is known-true because the project enforces the sandwich product per CLAUDE.md and the standard differential-geometry treatment in [Nakahara2003 §10.3]. Blue should win.

- [ ] **Step 1: Invoke the skill**

Run:
```
/red-blue-debate "Covariance transport in the codebase uses the sandwich product Σ → Ω Σ Ω^T, consistent with the standard tensor-transport rule on associated vector bundles." mode=implementation rounds=2 judge=on
```

- [ ] **Step 2: Verify artifact directory exists**

Run:
```bash
ls docs/debates/2026-05-19-sandwich-covariance-transport/
```

Expected: all 8 artifact files.

- [ ] **Step 3: Verify the verdict is BLUE_WINS with the canon citation**

Read `04_verdict.md`. Confirm:
- Outcome = `BLUE_WINS`.
- Decisive evidence cites `[Nakahara2003 §10.3]` (or equivalent standard tensor-transport reference) AND a code path confirming the sandwich form is used.
- Action item is "accept" or similar (no fix needed).

- [ ] **Step 4: Verify blue's opening contains falsification conditions**

Read `02_blue_opening.md`. Confirm the "Falsification conditions" section lists at least one concrete way the claim could be wrong (e.g., "this claim is wrong if `Σ_transported = Ω Σ` appears anywhere in the covariance-transport code path").

This separates blue from a sycophantic defender.

- [ ] **Step 5: Confirm no banned phrases**

Run:
```bash
grep -iE "crucially|notably|importantly|it's worth noting|interestingly|fundamentally|leverages|underscores|key insight|perhaps|it could be argued|both sides have a point" docs/debates/2026-05-19-sandwich-covariance-transport/*.md
```

Expected: no matches.

- [ ] **Step 6: Commit**

```bash
git add docs/debates/2026-05-19-sandwich-covariance-transport/
git commit -m "test(debate): smoke test 2 — blue wins on known-true sandwich-transport claim"
```

---

### Task 11: Smoke test 3 — judge=off, rounds=1 (cheap mode)

**Files:**
- Create (via skill invocation): `docs/debates/2026-05-19-em-mode-skip-attention-freeze/*.md`

This exercises the cheapest mode (2 dispatches) and the user-adjudication path. The claim is the CLAUDE.md-documented interaction between `skip_attention=True` and detaching `em_mode` values.

- [ ] **Step 1: Invoke the skill**

Run:
```
/red-blue-debate "skip_attention=True combined with em_mode=em_phi_q silently freezes sigma_embed at initialization." mode=implementation rounds=1 judge=off
```

- [ ] **Step 2: Verify only 4 files exist (no rebuttals, no judge verdict)**

Run:
```bash
ls docs/debates/2026-05-19-em-mode-skip-attention-freeze/
```

Expected:
```
00_claim.md
01_evidence.md
02_red_opening.md
02_blue_opening.md
04_verdict.md
05_action.md
```

(`03_*` files absent because rounds=1; `04_verdict.md` exists but is the user-adjudication template.)

- [ ] **Step 3: Verify 04_verdict.md is user-adjudication form**

Read `04_verdict.md`. Confirm:
- Title says "user-adjudicated".
- Contains a "Red opening summary" section and a "Blue opening summary" section.
- The "Your verdict" section is empty (user fills in).
- Main Claude has not declared a winner.

- [ ] **Step 4: Commit**

```bash
git add docs/debates/2026-05-19-em-mode-skip-attention-freeze/
git commit -m "test(debate): smoke test 3 — judge=off cheap mode, user adjudicates"
```

---

### Task 12: Final integration check

- [ ] **Step 1: List all installed files**

Run:
```bash
ls "C:/Users/chris and christine/.claude/agents/" ; ls "C:/Users/chris and christine/.claude/skills/red-blue-debate/"
```

Expected output:
```
red-team.md
blue-team.md
debate-judge.md
---
SKILL.md
protocol.md
```

- [ ] **Step 2: Verify the skill is discoverable**

In a new session (or via /help), confirm `red-blue-debate` appears in the available skills list.

- [ ] **Step 3: Verify the agents are discoverable**

Confirm `red-team`, `blue-team`, and `debate-judge` appear in the available agents list.

- [ ] **Step 4: Verify project-canon integration**

Read `.claude/agents/vfe-knowledge/README.md`. Confirm `debate_methodology.md` is listed in the methodology row and the "when to read what" table.

Read `.claude/agents/vfe-knowledge/debate_methodology.md`. Confirm it is the version written in Task 2.

- [ ] **Step 5: Final commit if anything else is staged**

```bash
git status
```

If any files in this repo are staged from prior tasks but uncommitted, commit them. Otherwise nothing to do.

---

## Spec-coverage self-review (run after writing the plan)

Mapping spec sections to tasks:

| Spec section | Tasks |
|---|---|
| Architecture / primitive choice | Tasks 4–8 (build the skill and three agents) |
| Files added globally | Task 1 (dirs), Tasks 4–6 (agents), Tasks 7–8 (skill) |
| Files added in repo | Task 2 (debate_methodology.md) |
| Files modified in repo | Task 3 (vfe-knowledge/README.md) |
| Canon-location resolution | Task 7 (protocol.md "Canon-location resolution" section) |
| Skill surface (CLI args) | Task 8 (SKILL.md "Invocation" section) |
| Protocol phases 0–5 | Task 7 (protocol.md) |
| Ultrathink in every agent dispatch | Tasks 7, 8 (skill files); Tasks 9, 10 (smoke tests verify it's in effect) |
| Agent definitions (red, blue, judge) | Tasks 4, 5, 6 |
| Output artifacts in docs/debates/ | Tasks 9, 10, 11 (smoke tests produce them) |
| Anti-failure: convergence | Tasks 9 verifies parallel openings disagree |
| Anti-failure: strawman | Agent definitions enforce steelman-first |
| Anti-failure: citation-free | Skill validates ≥1 citation per opening |
| Anti-failure: sycophantic judge | Judge agent definition forbids splitting differences |
| Anti-failure: Claude-isms | Tasks 9, 10, 11 grep verifies |
| Modes (theory/math/code/implementation) | Task 2 methodology + Tasks 9, 10, 11 cover code + implementation; theory/math covered by skill itself, tested on first real use |
| Testing plan | Tasks 9, 10, 11 |

Gaps: theory and math modes are not exercised by the smoke tests. Acceptable because (a) the methodology covers them in Task 2 and (b) the user will exercise them in real use; the smoke tests' job is to validate the orchestration mechanism, which code and implementation modes do.

---

## Placeholder self-check

Grep this plan for red-flag phrases. None should appear:
- "TBD", "TODO", "implement later", "fill in details"
- "add appropriate error handling"
- "similar to Task N"
- "write tests for the above" (without code)

None present. All steps have exact paths, exact commands, exact file content.

---

## Execution

Plan complete and saved to `docs/superpowers/plans/2026-05-19-red-blue-team-debate.md`. Two execution options:

**1. Subagent-Driven (recommended)** — Dispatch a fresh subagent per task, review between tasks, fast iteration. Best for this plan because Tasks 4–8 are independent file writes that benefit from isolated context per file.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch with checkpoints. Better if you want to watch each step.

Which approach?
