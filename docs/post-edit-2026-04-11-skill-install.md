# Skill Installation — 2026-04-11

## Summary

Installed 20 new Claude Code skills into `.claude/skills/` from two upstream repositories, bringing the project skill count from 25 to 45. No project code, configuration, or documentation was modified. The goal is to give the assistant domain coverage for the pure mathematics (SPD manifolds, Lie theory, information geometry), low-level Python/concurrency debugging, and theoretical physics context needed for the Gauge-Theoretic VFE Transformer.

## Motivation

A survey of three candidate skill sources was requested: `rand/cc-polymath` (code-recipe math skills), `alirezarezvani/claude-skills` (engineering/meta skills), and `sandraschi/advanced-memory-mcp` (tutor-style math skills bundled with an MCP memory server). The pre-existing 25 project skills cover the scientific Python stack (PyTorch, Lightning, pandas, statsmodels, pymc, torch-geometric), visualization (plotly, seaborn, scientific-visualization, umap), research writing (scientific-writing, peer-review, literature-review, arxiv-database), and GPU optimization. They do not cover pure mathematics theory, low-level debugging of the VFE hot path, or physics context for gauge theory. The 20 installed skills fill these gaps along two axes: computational recipes (cc-polymath) and theoretical/pedagogical references (sandraschi).

## What Changed

### New skill directories under `.claude/skills/`

From `rand/cc-polymath` (12 computational skills, markdown-per-skill, ~600–800 lines each):

1. `linear-algebra` — NumPy/SciPy matrix operations, SVD, QR, Cholesky, eigendecomposition. Relevant to `transformer/core/vfe_utils.py`, `transformer/core/gauge_utils.py`, `math_utils/generators.py`.
2. `numerical-methods` — ODE solvers, convergence monitoring, VFE iteration support.
3. `optimization-algorithms` — AdamW, SGD, Adam, natural gradient, Fisher projection. Relevant to `transformer/training/optimizer.py` and `transformer/core/vfe_gradients.py`.
4. `probability-statistics` — VFE foundations, KL, ELBO, Gaussian algebra, hypothesis testing.
5. `differential-equations` — ODEs, PDEs, gradient flow, analytical and numerical solutions.
6. `abstract-algebra` — Groups, rings, fields. Foundation for SO(N), GL(K), block-diagonal irreps in `math_utils/generators.py`.
7. `category-theory` — Functors, natural transformations, limits, adjunctions.
8. `python-debugging` — pdb, ipdb, pytest, VSCode/PyCharm debuggers, remote debugging with debugpy, cProfile.
9. `memory-leak-debugging` — Heap profiling for multi-language memory leak detection.
10. `performance-profiling` — CPU/memory profilers (perf, pprof, py-spy, heaptrack, Valgrind), flame graphs.
11. `concurrency-debugging` — Race condition, deadlock, and data race analysis with ThreadSanitizer.
12. `distributed-systems-debugging` — Trace correlation, request replay, clock skew, chaos engineering for DDP/FSDP training.

From `sandraschi/advanced-memory-mcp` (8 tutor-style skills, SKILL.md plus modules/ subdirectory, ~100 lines per module):

13. `matrix-theory-specialist` — Spectral theorem, matrix norms, Rayleigh quotient, Cholesky, condition number. SPD-manifold directly relevant.
14. `topology-geometry-guide` — Point-set topology, algebraic topology, differential geometry. Foundation for gauge-theoretic framework.
15. `mathematical-proofs-mentor` — Proof techniques and rigorous argumentation. For verifying gauge equivariance claims.
16. `fourier-analysis-expert` — Fourier series, transforms, harmonic analysis. Relevant to RoPE positional encoding in `transformer/core/transport_ops.py`.
17. `complex-analysis-expert` — Contour integration, residue theory, conformal mappings.
18. `real-analysis-fundamentals` — Limits, continuity, measure theory. Foundation for measure-theoretic VFE formulations.
19. `optimization-theory-expert` — Convex optimization, gradient methods, constrained optimization. Theoretical complement to `optimization-algorithms`.
20. `quantum-mechanics-explainer` — Quantum formalism. Gauge theory originates from QFT/QED and this provides physics context.

### Frontmatter normalization

The cc-polymath source files ship with prefixed names like `math-linear-algebra-computation` and `debugging-python-debugging`. These were rewritten to bare names matching the target directory (e.g. `linear-algebra`, `python-debugging`) so the frontmatter `name:` field aligns with the Claude Code directory-based skill identifier. One sandraschi file (`topology-geometry-guide/SKILL.md`) had `name: topology-and-geometry-guide` which was normalized to `topology-geometry-guide` to match its directory. No other content was edited.

### Source strategy differences

The two source repositories serve complementary purposes and were both installed despite overlap on topics like linear algebra and optimization. cc-polymath skills are dense code recipes — 622-line `linear-algebra-computation.md` is full of NumPy/SciPy call patterns and numerical recipes. sandraschi skills are terse pedagogical references — 100-line `linear-algebra-expert/modules/core-guidance.md` is LaTeX definitions and a tutor persona. Where cc-polymath answers "how do I compute this", sandraschi answers "what is this and why". The assistant will pick the appropriate one per query. Sandraschi skills are flagged upstream as "🔴 LOW confidence — Legacy template awaiting research upgrade", so they should be used for quick theoretical reference rather than authoritative derivations.

### What was not installed

The `alirezarezvani/claude-skills` repository was surveyed but not installed. Its engineering skills (pr-review-expert, senior-architect, skill-security-auditor) overlap with the existing `code-reviewer`, `python-pro`, and `refactoring-specialist` agents in `.claude/agents/`. Its ML skills (DSPy, RAG, HuggingFace) conflict with the project's "no neural networks" hard constraint.

The `advanced-memory-mcp` server itself — a FastMCP service with LanceDB vector search, Zettelkasten-style note linking, and arXiv/GitHub integration — was not installed. It is a Python service that needs to be run and registered in the Claude Code MCP configuration, which is a separate scope from skill file installation. It could be a useful follow-up for persistent experiment/paper memory across sessions but requires explicit user approval before setup.

Skills in sandraschi's mathematics category that overlap cc-polymath one-to-one (`linear-algebra-expert`, `differential-equations-solver`, `abstract-algebra-specialist`, `numerical-methods-expert`, `probability-theory-expert`, `statistics-probability-guide`) were skipped to avoid redundancy. The `number-theory-explorer`, `discrete-mathematics-expert`, `mathematical-logic-expert`, `game-theory-strategist`, `calculus-tutor`, and `applied-mathematics-engineering` skills were skipped as not sufficiently relevant to the gauge-transformer codebase.

The `scripts/hooks/` directory and `.claude/settings.json` were deleted upstream on `main` during this session (commits `916b9e7` and `18968fe`) after hook-path issues blocked tool execution. The `claude/search-claude-skills-repos-RV7Kb` branch was fast-forwarded to include these deletions before the skill install proceeded. This is not part of the current post-edit but is the environmental precondition that made the install possible.

## Files Added

```
.claude/skills/linear-algebra/SKILL.md
.claude/skills/numerical-methods/SKILL.md
.claude/skills/optimization-algorithms/SKILL.md
.claude/skills/probability-statistics/SKILL.md
.claude/skills/differential-equations/SKILL.md
.claude/skills/abstract-algebra/SKILL.md
.claude/skills/category-theory/SKILL.md
.claude/skills/python-debugging/SKILL.md
.claude/skills/memory-leak-debugging/SKILL.md
.claude/skills/performance-profiling/SKILL.md
.claude/skills/concurrency-debugging/SKILL.md
.claude/skills/distributed-systems-debugging/SKILL.md
.claude/skills/matrix-theory-specialist/{SKILL.md, README.md, _toc.md, modules/}
.claude/skills/topology-geometry-guide/{SKILL.md, README.md, _toc.md, modules/}
.claude/skills/mathematical-proofs-mentor/{SKILL.md, README.md, _toc.md, modules/}
.claude/skills/fourier-analysis-expert/{SKILL.md, README.md, _toc.md, modules/}
.claude/skills/complex-analysis-expert/{SKILL.md, README.md, _toc.md, modules/}
.claude/skills/real-analysis-fundamentals/{SKILL.md, README.md, _toc.md, modules/}
.claude/skills/optimization-theory-expert/{SKILL.md, README.md, _toc.md, modules/}
.claude/skills/quantum-mechanics-explainer/{SKILL.md, README.md, _toc.md, modules/}
```

## Files Modified

None. All changes are additions under `.claude/skills/`. No source code, tests, training configs, or documentation outside this post-edit note were touched.

## Files Deleted

None.

## Verification

- `ls .claude/skills/ | wc -l` returns 45 (25 existing + 20 new).
- All 20 new skills are visible in the Claude Code in-session skill list with correct frontmatter and directory names.
- `.claude/skills/linear-algebra/SKILL.md` and the other 11 cc-polymath skills have normalized `name:` fields matching their directories.
- `.claude/skills/topology-geometry-guide/SKILL.md` frontmatter normalized from `topology-and-geometry-guide` to `topology-geometry-guide`.
- No existing skill directory was overwritten or touched.
- No hooks, settings, or project code was modified.

## Branch and Commit

Development branch: `claude/search-claude-skills-repos-RV7Kb`. The branch was fast-forwarded to `origin/main` at commit `18968fe` before the install, bringing in the user-initiated deletion of `.claude/settings.json` and `scripts/hooks/` that had previously been blocking tool execution. The skill install is a single commit on top of that.
