# 2026-04-08 — External Skills and Plugins Installation

## Summary

Installed 7 scientific skills from `K-Dense-AI/claude-scientific-skills` and 8
workflow plugins from `wshobson/agents` into the project's `.claude/`
directory to augment the Claude Code harness for VFE Transformer research.
No Python code, no tests, no CLAUDE.md, and no `settings.json` were modified.

## Motivation

A survey of four external repositories (contains-studio/agents, wshobson/agents,
VoltAgent/awesome-claude-code-subagents, K-Dense-AI/claude-scientific-skills)
identified a shortlist of items that fill real gaps in the existing 18 skills
and 6 agents: graph neural network support for attention-graph analysis,
HuggingFace transformers for the BERT/RoBERTa diagnostic entry points, GPU
kernel optimization guidance for the VFE gradient hot path, BibTeX/citation
management, publication-quality schematics, LaTeX poster generation, critical-
thinking rigor patterns, and workflow-orchestration plugins for managing
multi-week ablation studies. The `contains-studio/agents` repository was
excluded as product/startup-focused and not aligned with research workflows;
the `VoltAgent/awesome-claude-code-subagents` catalog was excluded as an
aggregator whose top picks duplicate existing built-in agents.

## K-Dense-AI Skills Installed

Source: `https://github.com/K-Dense-AI/claude-scientific-skills`
Source commit: `899a51bfbd63218153b272c9459dcb4a7ae88da6`
Install path: `.claude/skills/<name>/`

| Skill | Justification |
|---|---|
| `torch-geometric` | GNN/message-passing library; aligns with attention-graph analysis in `transformer/analysis/holonomy.py` and gauge-equivariant message passing experiments. |
| `transformers` | HuggingFace transformers required by `transformer_test.py` and `roberta_diagnostics.py` for BERT/RoBERTa KL-alignment validation. |
| `optimize-for-gpu` | CUDA/CuPy/Numba guidance for the fused VFE gradient kernels in `transformer/core/vfe_gradients.py`. |
| `citation-management` | BibTeX workflow complementing `arxiv-database` and `literature-review` for manuscript preparation. |
| `scientific-schematics` | Publication-quality diagrams for the VFE hierarchy (`h → s → p → q`), gauge transport, and attention graph. |
| `latex-posters` | Conference poster generation for presenting the gauge-theoretic framework. |
| `scientific-critical-thinking` | Rigor-enforcement patterns aligned with the CLAUDE.md "push back" / "no bullshit" communication style. |

After install the harness auto-discovered all seven; they appear in the
Skill tool's available skills list.

## wshobson Plugins Installed

Source: `https://github.com/wshobson/agents`
Source commit: `70444e5b1fae2237f3cb087c70db043ab633fe11`
Install path: `.claude/plugins/<name>/`

Each plugin preserves its full structure: `.claude-plugin/` (manifest),
`agents/`, `commands/`, `skills/`, `templates/`, `README.md`, and in the case
of `plugin-eval` also a full Python package (`src/`, `tests/`, `pyproject.toml`,
`uv.lock`).

| Plugin | Justification |
|---|---|
| `conductor` | "Context → Spec & Plan → Implement" workflow with state persistence, for managing ablation sweeps driven by `transformer/train_publication.py`. |
| `machine-learning-ops` | MLOps skills for experiment tracking, training pipelines, reproducibility; complements PyTorch Lightning + WandB. |
| `distributed-debugging` | Multi-GPU (DDP/FSDP) diagnostics for scaled VFE training runs. |
| `observability-monitoring` | Training monitoring beyond WandB defaults. |
| `plugin-eval` | Three-layer evaluation framework (static + LLM-judge + Monte Carlo) applicable to validating VFE gradient kernels and finite-difference tests in `transformer/pure_vfe/tests/test_gradients.py`. |
| `agent-teams` | Parallel multi-agent coordination for code review + math verification + doc generation. |
| `agent-orchestration` | General multi-agent orchestration primitives. |
| `data-validation-suite` | Data contract validation for dataset loaders in `transformer/data/datasets.py`. |

The `.claude/plugins/` directory did not exist prior to this install and was
created fresh. Whether the harness auto-discovers plugins placed at this path
is not yet confirmed for this runtime; if the plugin agents do not appear in
the next session's agent list, the fallback is to extract each plugin's
`agents/*.md` into `.claude/agents/` and `skills/*` into `.claude/skills/`
(see the plan file `/root/.claude/plans/serene-stirring-neumann.md` for the
Option B extraction script).

## What Was NOT Changed

- `.claude/settings.json` — hooks (`check_no_nn.py` on Write, pytest on git
  commit) and `subagentModel: haiku` preserved byte-for-byte.
- `CLAUDE.md` — unchanged.
- Any file under `transformer/`, `tests/`, `scripts/`, `math_utils/` — no
  Python code edits.
- The 18 pre-existing skills and 6 pre-existing agents — only additions.

## Verification

1. `ls .claude/skills/ | wc -l` → 25 (was 18, +7 new).
2. `ls .claude/plugins/ | wc -l` → 8 (new directory, +8 new).
3. Each new K-Dense-AI skill has a readable `SKILL.md`.
4. Each new wshobson plugin preserves `.claude-plugin/`, `agents/`, and
   `commands/` subdirs.
5. `diff /tmp/settings.json.pre-install .claude/settings.json` → empty
   (no changes).
6. Harness auto-discovery of K-Dense-AI skills confirmed via Skill tool
   availability list.

## Skipped Items

### K-Dense-AI skills NOT installed (tier 2 except #12, and tier 3)
`modal`, `dask`, `zarr-python`, `polars`, `matplotlib`, `pymatgen`, `astropy`,
`qiskit`, `cirq`, `qutip`, `scholar-evaluation`, `paper-lookup`,
`perplexity-search`, `research-lookup`. Candidates for a future install if
the matching workflow materialises.

### Entire repositories skipped
- `contains-studio/agents`: 40 agents targeted at 6-day commercial product
  sprints (marketing, UI, launch ops). No research alignment.
- `VoltAgent/awesome-claude-code-subagents`: curated catalog; top ML/research
  picks duplicate existing built-in agents. Kept as browse-only reference.

## References

- Plan file: `/root/.claude/plans/serene-stirring-neumann.md`
- K-Dense-AI source: `https://github.com/K-Dense-AI/claude-scientific-skills/tree/899a51bf/scientific-skills`
- wshobson source: `https://github.com/wshobson/agents/tree/70444e5b/plugins`
