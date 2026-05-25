# Red Team / Blue Team Debate — Design Spec

**Date:** 2026-05-19
**Status:** Draft for user review
**Owner:** cdenn016
**Scope:** New skill `red-blue-debate` + three new subagents (`red-team`, `blue-team`, `debate-judge`), all installed **globally** in `~/.claude/` so they are available across every repo. The methodology canon stays in this repo's `.claude/agents/vfe-knowledge/` (a new `debate_methodology.md` file is added there), because the user's actual debates are all about the VFE theory and that canon is the authoritative source for citation discipline, banned phrases, and external standards. The global skill and agents reference `vfe-knowledge/` by a path resolved at runtime relative to the current working directory; in other repos without a `vfe-knowledge/`, the skill falls back to a minimal generic methodology baked into the skill's `protocol.md`.

## Motivation

The codebase already has four specialist auditing agents (`vfe-codebase-auditor`, `vfe-manuscript-reviewer`, `vfe-cross-manuscript-consistency`, `vfe-experiment-analyst`), each with isolated context and a shared canon in `.claude/agents/vfe-knowledge/`. They are excellent at *unilateral* review against external standards but do not produce adversarial dialectic. A claim like "this E-step path is gauge-equivariant" or "the manuscript's free-energy derivation is canonical" is currently checked by one auditor at a time; if the auditor misses an objection, nothing catches it.

The goal of this skill is to produce **structured adversarial debate with isolated context per side**, so the failure mode "single Claude convinces itself" is structurally excluded. Outputs are durable Markdown artifacts that survive the session and can be revisited.

## Non-goals

- Best-of-N tournaments.
- Persistent multi-session debate state.
- LLM-as-evaluator numerical quality scores.
- Direct integration into `deep-audit` (composable later if useful).
- Replacement of the existing `vfe-*` auditors. Debate composes with them; it does not subsume them.

## Architecture

### Primitive choice

- **Skill** `red-blue-debate` is the global orchestrator. It owns the protocol — phase ordering, parallel dispatch, file layout, judge invocation, banned-phrase enforcement. Installed at `~/.claude/skills/red-blue-debate/` so it is callable from every repo.
- **Subagents** `red-team`, `blue-team`, `debate-judge` are dispatched from the skill. Each has isolated context per invocation. They do not see each other's reasoning except via the durable artifact files the skill hands them. Installed at `~/.claude/agents/` so they are callable from every repo.
- **Methodology canon stays in `.claude/agents/vfe-knowledge/`** (this repo). A new `debate_methodology.md` joins the existing canon files there. The user's debates are about the VFE theory; the existing `external_canon_*.md`, `style_constraints.md`, and the new `debate_methodology.md` together are the authoritative methodology.
- **Canon-location resolution at runtime.** The skill instructs main Claude to find a debate canon by looking, in order: (1) `<cwd>/.claude/agents/vfe-knowledge/` (the canonical location in this repo), (2) `<cwd>/.claude/agents/*-knowledge/` (any similarly-named project canon), (3) a minimal generic fallback methodology embedded inside `~/.claude/skills/red-blue-debate/protocol.md`. The fallback exists only so the skill does not break in a repo without a canon — it is **not** the path the user normally takes.

Rationale: a single-agent design fails because both sides share context and converge. A pure parallel-agents-only design (no skill orchestrator) leaves no place to enforce the protocol. A skill orchestrating isolated agents is the right factoring. Global installation makes the skill callable from every repo without re-installation. Keeping the canon in `vfe-knowledge/` preserves the user's existing source-of-truth infrastructure and avoids forking the canon into two places.

### Files added — globally (in `~/.claude/`)

```
~/.claude/skills/red-blue-debate/SKILL.md
~/.claude/skills/red-blue-debate/protocol.md
~/.claude/agents/red-team.md
~/.claude/agents/blue-team.md
~/.claude/agents/debate-judge.md
```

### Files added — in this repo (`.claude/agents/vfe-knowledge/`)

```
.claude/agents/vfe-knowledge/debate_methodology.md
```

### Files modified — in this repo

```
.claude/agents/vfe-knowledge/README.md
```

Add `debate_methodology.md` to the file inventory in the source-of-truth table and add a row to the "when to read what" table for "Run a red/blue debate" pointing to it.

### Output artifacts (per debate)

```
<cwd>/docs/debates/YYYY-MM-DD-<slug>/
  00_claim.md
  01_evidence.md
  02_red_opening.md
  02_blue_opening.md
  03_red_rebuttal.md
  03_blue_rebuttal.md
  04_verdict.md
  05_action.md
```

Artifacts live in `docs/debates/` under the **current working directory** (the repo the user invoked from), not globally. This keeps per-repo debate history with the repo it belongs to. If `<cwd>/docs/` does not exist, the skill creates it; if the cwd is not a git repo, the skill warns and writes to `<cwd>/debates/` instead.

`<slug>` is a short kebab-case summary of the claim (e.g., `e-step-blindness-vfe-default`).

## Skill surface

User invocation:

```
/red-blue-debate <claim>
  [mode=theory|math|code|implementation]   default: inferred from claim language
  [rounds=1|2]                              default: 2
  [judge=on|off]                            default: on
  [evidence=auto|paths:<glob>]              default: auto
```

The skill makes reasonable inferences when args are omitted (auto mode). If `<claim>` is empty, the skill prompts once for the claim and proceeds.

## Protocol

### Phase 0 — Claim extraction (skill, main Claude)

The skill parses `<claim>` into a single declarative sentence and writes `00_claim.md`. If the claim is compound, the skill picks the load-bearing proposition and warns the user that sub-claims will need separate debates. If the user agrees in advance (auto mode), the skill picks the strongest sub-claim and proceeds.

### Phase 1 — Shared evidence pack (main Claude, in the skill)

The skill instructs main Claude to assemble a neutral fact pack directly — no agent dispatch. Output: `01_evidence.md` containing:

- For `code` / `implementation` mode: file paths and line numbers of all code referenced by the claim, plus the pre-fix-protocol output (active config, override trace, resolved values).
- For `theory` / `math` mode: the relevant equation as it appears in the manuscript (with `.tex` line numbers), plus the relevant canon excerpts pulled from `external_canon_*.md` and (if needed) WebFetch'd primary sources.

Both `red-team` and `blue-team` receive this pack verbatim in Phase 2. This prevents the debate from becoming an argument about what the code or manuscript says.

If `evidence=paths:<glob>` is provided, the evidence pack is constrained to those paths.

Rationale for not using an agent: a fact pack is small (file reads + grep results) and benefits from main Claude's working knowledge of the user's active config and recent edits. The cost of context bloat is low; the benefit of avoiding a dispatch and an overloaded agent definition is high.

### Phase 2 — Openings (parallel, isolated)

Skill dispatches `red-team` and `blue-team` **in a single message with two `Agent` tool uses** (the parallel-agents pattern). Each agent receives:

- `00_claim.md`
- `01_evidence.md`
- Its role mandate (red attacks, blue defends).
- The mode-specific checklist from `debate_methodology.md`.
- **The literal keyword `ultrathink` in the dispatch prompt** (see "Reasoning depth" below).

Each writes its opening to `02_<side>_opening.md`. **Neither sees the other's output in this phase.**

Each opening must contain, in this order:

1. **Steelman** — one sentence stating the opposing position in its strongest form (anti-strawman).
2. **Position** — the team's own thesis as a falsifiable statement.
3. **Evidence** — at least one of: external canon citation (with full `[Author Year §X.Y]` tag), code line number with config trace, executed sympy/finite-diff verification, manuscript line number.
4. **Falsification conditions** — "this position is wrong if X, Y, or Z."

Openings missing any of (1)-(4) are flagged by the skill as malformed. The skill re-dispatches that side once with a corrective prompt naming the missing element. If the second attempt is still malformed, the skill proceeds with a warning written to `02_<side>_opening.md` ("malformed: missing element N, debate continues") rather than blocking — a malformed side is a strike against that team in Phase 4.

### Phase 3 — Rebuttals (parallel, isolated)

Skill dispatches `red-team` and `blue-team` again in parallel. Each now receives:

- `00_claim.md`, `01_evidence.md`
- The opposing team's opening only (not its own).
- Its role mandate.
- The literal keyword `ultrathink` in the dispatch prompt.

Each writes `03_<side>_rebuttal.md`. Required structure:

1. **Concession** — at least one point from the opposing opening that the team grants (or "no concession; here is why").
2. **Core attack** — the load-bearing weakness in the opposing opening, cited.
3. **Defense** — strengthening of own position against the opposing argument.

`rounds=1` skips Phase 3.

### Phase 4 — Judgment (optional, single agent)

If `judge=on`, skill dispatches `debate-judge` with all four artifacts and the literal keyword `ultrathink` in the dispatch prompt. The judge writes `04_verdict.md` containing:

- **Verdict**: one of `RED_WINS`, `BLUE_WINS`, `REMAND`, `OUT_OF_SCOPE`.
- **Decisive evidence**: the specific citation/code-line/computation that broke the tie. Verdicts without decisive evidence are forbidden.
- **Reasoning**: one paragraph explaining why the decisive evidence was decisive.
- **Action**: the action that follows from the verdict (fix X, accept Y, run experiment Z, debate sub-claim W).

If `judge=off`, the skill writes `04_verdict.md` itself by relaying both rebuttals verbatim and asking the user to declare the verdict. No main-Claude synthesis — the user adjudicates.

### Phase 5 — Synthesis to user

Skill writes `05_action.md` (extracted from the verdict) and presents the path to the user with a one-paragraph summary. The summary itself must not contain banned phrases.

## Reasoning depth — ultrathink (load-bearing)

Every `red-team`, `blue-team`, and `debate-judge` dispatch from the skill **must include the literal token `ultrathink` in the dispatch prompt**. This is not optional and not subject to a toggle.

Rationale: the entire value of adversarial dispatch is real falsification effort. Shallow per-agent reasoning produces performative disagreement — superficial attacks, reflexive defenses, sycophantic verdicts — which is exactly the failure mode this design exists to prevent. Ultrathink in the dispatch prompt is what converts the architecture into actual dialectic.

The skill itself need not be invoked with ultrathink (orchestration is mechanical). The evidence-gathering phase (main Claude) need not either (it is mechanical file-reading). It is the three agents — and only those three — where deep reasoning per side is structurally required.

Cost note: ultrathink increases per-dispatch token use. Combined with up-to-5 Opus dispatches per debate, a full debate is non-trivial in cost. This is acknowledged and accepted; the `rounds=1` and `judge=off` toggles exist for cheaper variants when the user wants a quicker pass.

Implementation: the skill's prompt-construction step for each Agent tool call appends `ultrathink` (case-insensitive, recognized by the harness keyword detector) to the dispatch prompt before the structured instructions. Agent definitions need not reference ultrathink internally — the directive lives in the calling prompt.

## Agent definitions (sketch — final wording belongs in the implementation plan)

### `red-team` agent

```yaml
---
name: red-team
description: Adversarial skeptic in the red-blue-debate skill. Dispatched in isolated context to attack a specific claim with primary-source evidence. Steelman first, then falsify. Not a general-purpose reviewer.
tools: Read, Glob, Grep, Bash, Edit, Write, WebFetch, WebSearch, Skill
model: opus
---
```

Body content:
- Mandate: falsify the claim. The skill is paying you to find the weakness. If you cannot find one, say "I cannot falsify this under the current evidence" — do not pad.
- Reading order on invocation: `00_claim.md`, `01_evidence.md`, the mode-specific section of `debate_methodology.md`, then the relevant `external_canon_*.md` for citation forms.
- Citation discipline: every claim of fact must cite a primary source (external canon for theory/math; code path with line number for code; sympy output or finite-diff smoke test for math; manuscript line for manuscript claims). Uncited claims are weak strikes.
- Banned phrases enforced (read `style_constraints.md`).
- Output format prescribed by phase (opening structure for Phase 2; rebuttal structure for Phase 3). The skill tells the agent which phase it is in.

### `blue-team` agent

Same skeleton, opposite mandate. Body: defend the claim by steelmanning it and producing the strongest primary-source-supported defense. Same citation discipline, same banned phrases. Required to state falsification conditions (the claim is *not* defensible if X) — this is what distinguishes blue-team from a sycophantic defender.

### `debate-judge` agent

```yaml
---
name: debate-judge
description: Evidence-weighing adjudicator for the red-blue-debate skill. Reads opening + rebuttal from both sides and declares a verdict with the decisive citation. Forbidden to split differences when one side has cited evidence and the other has not.
tools: Read, Glob, Grep, WebFetch
model: opus
---
```

Body: weigh by evidence, not rhetoric. Specifically forbidden:
- Splitting the difference when one side has cited and the other has not.
- Declaring `REMAND` to avoid taking a position when one side is clearly correct on the evidence.
- Sycophantic synthesis of both positions.

Required: cite the specific evidence that broke the tie. If both sides are equally evidenced but on different parts of the claim, the verdict is `OUT_OF_SCOPE` with a reformulation.

## Shared canon: `debate_methodology.md`

New file in this repo's `.claude/agents/vfe-knowledge/` (not in the global skill directory). The global skill resolves the canon location at runtime as described in "Canon-location resolution at runtime" above. Sections:

1. **The source-of-truth rule** — same as existing audit/review methodologies: external canon is the source of truth, user's claims are what's being evaluated. The debate is *between* a defender and an attacker of the user's claim, both grounded in external canon.
2. **Mode-specific checklists** — concrete checklists for `theory`, `math`, `code`, `implementation` modes. For example, `code` mode mandates the pre-fix protocol (active config, override trace, runtime line confirmation) before any line-of-code claim.
3. **Citation forms** — exactly what counts as a valid citation in each mode (matching the existing audit methodology).
4. **Output format** — phase-specific templates (opening, rebuttal, verdict).
5. **Banned phrases** — pointer to `style_constraints.md` plus a debate-specific addition: no "perhaps", no "it could be argued", no hedging phrases that dilute the adversarial position.

The file is read by all three debate agents on every invocation, the same way the existing canon files are read by the existing auditors.

## Anti-failure mechanisms

| Failure | Mechanism |
|---|---|
| Convergence (both sides reach the same conclusion) | Parallel isolated-context dispatch; openings written before either side sees the other |
| Strawman | Each opening + rebuttal must steelman / concede first |
| Citation-free assertion | Skill validates that openings contain at least one citation matching the mode-specific form; re-dispatches once if missing |
| Sycophantic judge | Judge forbidden to split differences without decisive evidence; must cite the specific tie-breaker |
| Shallow / performative adversarial reasoning | Every red/blue/judge dispatch is in ultrathink mode (not a toggle — see "Reasoning depth") |
| Stale-config bug in code mode | Pre-fix protocol required in evidence pack and in any line-of-code claim |
| Claude-isms | Banned-phrase list enforced via `style_constraints.md` in agent prompts |
| Main Claude reads both sides and synthesizes sycophantically | Synthesis is constrained to relaying the verdict; main Claude does not weigh sides in `judge=on` mode. In `judge=off` mode, the user adjudicates and main Claude relays verbatim. |

## Testing plan

The skill itself does not have unit tests in the conventional sense (it orchestrates LLM calls). Validation is empirical:

1. **Convergence test** — run the same claim through the skill twice; verify that openings are written before rebuttals (file timestamps), and that the two openings disagree on at least one substantive point.
2. **Citation test** — run a claim known to have strong canon support; verify both teams cite the canon, and the judge can identify which interpretation is consistent with it.
3. **Anti-sycophancy test** — run a claim that is clearly false (e.g., "the codebase uses raw Euclidean gradients on the Lie algebra without preconditioning"); verify that `RED_WINS` and that the judge cites the specific code path that contradicts the claim.
4. **Mode coverage** — at least one debate in each mode (`theory`, `math`, `code`, `implementation`).
5. **judge=off mode** — verify main Claude relays without synthesizing.

These tests are documented in the writing-plans output; the user runs them as part of acceptance.

## Risks and tradeoffs

- **Cost**: a full debate is 5 agent dispatches at most (2 openings + 2 rebuttals + judge) — Phase 1 evidence is done by main Claude, not an agent. Each agent dispatch is in ultrathink mode (mandatory — see "Reasoning depth" above), so per-dispatch cost is higher than a default Opus call. Mitigation: `rounds=1` skips rebuttals (down to 3 dispatches); `judge=off` skips the judge (down to 4 dispatches); both flags together (2 dispatches) is the cheapest mode. The ultrathink requirement itself is not negotiable — the cheapest mode is still ultrathink-on-2-dispatches, not "ultrathink off."
- **Latency**: parallel dispatch helps but does not eliminate the wait. Acceptable for high-stakes claims; overkill for low-stakes.
- **Judge quality ceiling**: the judge is itself an LLM. The mitigation is constraint-based (must cite decisive evidence) rather than aiming for an objective oracle.
- **Evidence pack quality**: if the evidence gatherer misses a relevant fact, both sides debate with bad evidence. Mitigation: `evidence=paths:<glob>` lets the user constrain the gatherer; the user can also edit `01_evidence.md` between Phase 1 and Phase 2 if running manually.

## Open questions (resolved by auto mode)

- **Should the judge cite external canon or just internal evidence?** Resolved: judge cites whatever is decisive — external canon if the debate is about whether the user's form matches the standard form, code/config trace if the debate is about what the code does at runtime.
- **Should rebuttal phase be three-way (red rebuts blue rebuts red)?** Resolved: no. Two rounds maximum. Diminishing returns and runaway cost.
- **Should the skill enforce that the claim is a single proposition?** Resolved: yes, in Phase 0. Compound claims are split (auto mode picks the strongest sub-claim).

## Implementation order (for writing-plans)

1. Create `.claude/agents/vfe-knowledge/debate_methodology.md` (in this repo) and update `vfe-knowledge/README.md` index.
2. Create three agent definitions globally at `~/.claude/agents/red-team.md`, `~/.claude/agents/blue-team.md`, `~/.claude/agents/debate-judge.md`.
3. Create the skill globally at `~/.claude/skills/red-blue-debate/SKILL.md` and `~/.claude/skills/red-blue-debate/protocol.md`. `protocol.md` includes the minimal-generic-fallback methodology used when no project canon is found.
4. Run a smoke debate on a known-false claim to verify the convergence and anti-sycophancy mechanisms.
5. Run a smoke debate on a known-true claim to verify blue can win and judge cites correctly.
