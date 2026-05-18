# VFE Theory-Auditor Agents — Design Spec

**Date:** 2026-05-18
**Status:** Approved (single-pass; user said "build it")
**Author:** brainstorming + ultrathink session

## Purpose

Two project-level Claude Code subagents, sharing a knowledge base, that act as a resident domain expert in:

- Variational free energy (VFE) / active inference
- The Gauge-Transformer codebase (this repo)
- Information geometry (KL on SPD, natural gradients, retractions)
- Gauge theory (parallel transport, equivariance, GL(K) / Lie algebra)

They are *not* general code reviewers. They exist to catch mathematical/theoretical drift between intent and code, and to peer-review the manuscripts in `Attention/`.

## Two agents

### `vfe-codebase-auditor`

Invoked when the user wants a theoretical-purity audit of a module, config path, or surface area.

**Inputs:** a target (file, function, config preset, or natural-language scope like "audit the E-step under `em_mode='ift_phi'`").

**Outputs:** severity-tagged finding list — Critical / Major / Minor / Note — each with `file:line`, the canonical invariant violated, evidence (citation to manuscript equation or knowledge file), and the minimal fix.

**Workflow per the CLAUDE.md pre-fix protocol:**
1. Open the user's *active* config (not a default), trace every relevant key through the loader and any override logic.
2. Confirm which code path actually runs under that config.
3. Check the implementation against canonical equations in the knowledge base.
4. Where math is non-trivial, drop into the `sympy` skill to verify symbolically.
5. Emit the report.

**Tool surface:** Read, Glob, Grep, Bash, Edit, Write, Skill (uses `sympy`, `math-skills`, `topology-geometry-guide` as needed).

### `vfe-manuscript-reviewer`

Invoked when the user wants a structured peer review of a `.tex` file in `Attention/` or a Methods/derivation section.

**Inputs:** a manuscript path (or a question like "review the derivation in §3.2 of `GL(K)_attention.tex`").

**Outputs:** a peer review document following the `peer-review` skill structure, with these sections:
- Summary (5–8 sentences, no praise preambles)
- Major Issues (numbered, each: claim → evidence → required revision)
- Minor Issues
- Math Reviewer items (`MR-1`, `MR-2`, …) matching the commit-message convention seen in this repo's history
- Editorial (style violations per `style_constraints.md`)
- Citation Verification (claims that need external sourcing flagged with WebFetch/WebSearch evidence)

**Workflow:**
1. Parse the manuscript; pull out every load-bearing claim and every numbered equation.
2. Cross-check equations against the canonical free-energy form and against what the code actually implements (so manuscript ↔ implementation can't silently diverge).
3. For external claims, use `arxiv-database` / `literature-review` / WebFetch to verify the cited paper says what's claimed.
4. Apply style scanner (banned patterns, `\;`/`\,` spacing macros, em-dash horizontal rules).
5. Emit the review.

**Tool surface:** Read, Glob, Grep, Bash, Edit, Write, WebFetch, WebSearch, Skill (uses `peer-review`, `sympy`, `literature-review`, `arxiv-database`, `scientific-writing`).

## Shared knowledge base

Located at `.claude/agents/vfe-knowledge/`. Files:

| File | Contents |
|---|---|
| `README.md` | Index; tells agent which file to read when |
| `canonical_math.md` | Free energy F, attention β, Omega = exp(φᵢ)·exp(-φⱼ), sandwich product, τ = κ√K |
| `em_modes.md` | The 5-mode gradient-flow table from CLAUDE.md, plus the `transformer/vfe/` package's own profile |
| `e_step_constraints.md` | E-step blindness to targets, decoupled μ/σ LRs, retraction, SPD trust region |
| `codebase_map.md` | Where each math construct lives (file paths), which configs are active, `transformer/core/` vs `transformer/vfe/` differences |
| `manuscript_index.md` | One-paragraph synopsis of each `.tex` in `Attention/` + the load-bearing claims to verify |
| `audit_methodology.md` | Concrete checklist for code purity audits |
| `review_methodology.md` | Concrete checklist for manuscript peer reviews |
| `style_constraints.md` | Banned patterns, LaTeX rules, scientific writing conventions from CLAUDE.md |

## Model and personality

- **Model:** `opus` (4.7, 1M context). Math verification and cross-document reasoning benefit from the strongest model; 1M context lets the auditor hold a whole module + the canonical math + the relevant manuscript section without summarization loss.
- **Personality:** matches the project's communication style — direct, no praise preambles, no Claude-isms, willing to say "I don't know" and to push back. Encoded in the agent body.

## Non-goals

- These agents are not general code reviewers — defer style/lint/security to existing `code-reviewer`.
- They do not write new theory — they verify existing theory against existing code/manuscripts.
- They do not auto-commit — they emit reports; the user decides what to do.

## Files created by this spec

```
.claude/agents/vfe-codebase-auditor.md
.claude/agents/vfe-manuscript-reviewer.md
.claude/agents/vfe-knowledge/README.md
.claude/agents/vfe-knowledge/canonical_math.md
.claude/agents/vfe-knowledge/em_modes.md
.claude/agents/vfe-knowledge/e_step_constraints.md
.claude/agents/vfe-knowledge/codebase_map.md
.claude/agents/vfe-knowledge/manuscript_index.md
.claude/agents/vfe-knowledge/audit_methodology.md
.claude/agents/vfe-knowledge/review_methodology.md
.claude/agents/vfe-knowledge/style_constraints.md
docs/superpowers/specs/2026-05-18-vfe-theory-auditor-agents-design.md  (this file)
```
