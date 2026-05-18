# VFE Knowledge Base — Index

Shared reference for `vfe-codebase-auditor` and `vfe-manuscript-reviewer`.

## Source-of-truth principle (read first)

The agents have **two kinds of sources**, and the distinction matters:

| Source kind | Files | Role |
|---|---|---|
| **External canon** (sources of truth) | `external_canon_math.md`, `external_canon_inference.md`, `external_canon_transformers.md`, `external_bibliography.md` | The standard tools and math of information geometry, differential geometry, gauge theory, free energy principle / active inference, variational inference, and transformer attention. Cited as **canon** for findings. Drawn from textbooks (Amari & Nagaoka, Nakahara, Kobayashi-Nomizu, Frankel) and load-bearing papers (Friston 2010, Vaswani 2017, Amari 1998, Bai-Kolter-Koltun 2019, etc.) — not from this codebase or its manuscripts. |
| **User claims** (the thing being evaluated) | `user_theory_summary.md`, `em_modes.md`, `e_step_constraints.md`, `codebase_map.md`, `manuscript_index.md`, `diagnostics_glossary.md`, `notation_dictionary.md` | Descriptions of *what the user claims and builds*. These are evaluated against the external canon, not treated as canon themselves. `user_theory_summary.md` summarizes the user's theory; `diagnostics_glossary.md` enumerates the runtime metrics the codebase tracks (with standard interpretations); `notation_dictionary.md` catalogs the manuscripts' symbols and already-detected drifts. |
| **Methodology** | `audit_methodology.md`, `review_methodology.md`, `analysis_methodology.md`, `consistency_methodology.md` | Ordered checklists and output formats — one per agent (auditor, reviewer, analyst, consistency). |
| **Style** | `style_constraints.md` | Project-wide style rules: banned phrases, banned LaTeX, communication style. |

**Direction of evaluation:** standard literature → user. Never the reverse.

When the agents find a discrepancy:
- If the user's form differs from the standard form, cite the standard source (e.g., `[Nakahara2003 §10.3]`) and flag the drift.
- If the user's form is a generalization with no claim of standard-equivalence, label it "novel construction — requires independent justification."
- If the user's form claims standard-equivalence but the derivation is incomplete, flag and ask for the missing reduction.

The agents are not here to validate the user's theory by checking internal consistency. They are here to evaluate the user's theory against the standard tools of the relevant fields.

## When to read what

| Task | Required reading |
|---|---|
| Any audit, review, analysis, or consistency check | `README.md` (this file), the relevant `external_canon_*.md`, `external_bibliography.md` |
| Audit gauge / transport / Lie-algebra code | `external_canon_math.md`, then `codebase_map.md` |
| Audit free-energy assembly | `external_canon_inference.md`, then `user_theory_summary.md` (as claim summary), then `codebase_map.md` |
| Audit attention β / softmax derivation | `external_canon_transformers.md` + `external_canon_inference.md`, then `user_theory_summary.md`, then `codebase_map.md` |
| Audit E-step / IFT / amortized | `external_canon_inference.md` (EM + IFT), then `e_step_constraints.md`, then `em_modes.md` |
| Review any manuscript | `manuscript_index.md` (for orientation), all three `external_canon_*.md`, `review_methodology.md`, `style_constraints.md` |
| Analyze a training run / sweep / ablation | `analysis_methodology.md`, `diagnostics_glossary.md`, `external_canon_inference.md` + `external_canon_transformers.md` for interpretive priors |
| Check consistency across manuscripts | `consistency_methodology.md`, `notation_dictionary.md`, `style_constraints.md`, all three `external_canon_*.md` for canonical equation forms |
| Run a code-purity audit | `audit_methodology.md` first |
| Run a peer review | `review_methodology.md` first |
| Run an experiment analysis | `analysis_methodology.md` first |
| Run a cross-manuscript consistency check | `consistency_methodology.md` first |

## Project-specific hard constraints (operate within these, but they are NOT theoretical canon)

These come from this project's CLAUDE.md and the user's policy choices. They are *constraints on what the codebase commits to do*, not statements about standard mathematics. The agents respect them as project policy but do not treat them as theoretical truths.

1. **No neural networks** in the user's framework (project policy). The constraint is theoretical (the user wants representational capacity from VFE minimization alone, no learned MLPs). Audit findings should respect this — flag introductions of `nn.Linear` etc. Documented exceptions: `connection.py` MLP mode (opt-in research variant), final K→vocab projection.
2. **No CLI arguments on new entry points** (project policy — click-to-run pattern).
3. **`Σ_t = Ω Σ Ωᵀ` (sandwich) for covariance transport** — this *is* the standard rule from differential geometry [Nakahara2003 §10.3]; the project enforces it correctly. Audit findings cite the standard, not just the policy.
4. **E-step must not see targets** — this is the standard variational-EM separation of E and M steps [DempsterLairdRubin1977]. The project enforces it correctly.
5. **There must always exist a theoretically pure path** under appropriate toggles (project policy). Computationally extreme paths are opt-in.
6. **Code focus, not comments** — when reading code paths, the agent should not rely on inline comments; comments drift.

## The pre-fix protocol (project policy)

> Before you say the fix is done: (1) open my active config file, (2) trace every relevant key through the config loader and any override logic, (3) confirm the exact line you changed is reached at runtime under my config, (4) only then run tests and report.

Both agents follow this for any audit that proposes a fix.

## Communication style (project policy)

- Be direct. "This is wrong because X."
- State uncertainty plainly. "I don't know" is preferred over speculation.
- Push back. Don't capitulate under pressure — ask "what am I missing?"
- No praise preambles. No "great question", "excellent point".
- No Claude-isms: `key insight`, `crucially`, `notably`, `importantly`, `it's worth noting`, `interestingly`, `fundamentally`, `in particular`, `leverages`, `underscores` — banned in manuscripts and in agent output.
- No bullshit. Interpretive correspondences are not theorems; say so.
