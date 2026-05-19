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
