# Action — pifb-discussion-constructivism

**From verdict:** RED_WINS (with per-sub-claim breakdown; see `04_verdict.md`)

## Recommended action

Apply two clerical citation patches in `Attention/Participatory_it_from_bit.tex`. Do not add a "constructivism" subsection. Do not cite Glasersfeld, Piaget, Maturana, Varela, or Heinz von Foerster in the Discussion — those attributions are inconsistent with the framework's stated positive ontology at lines 3344, 3386, 3442.

### Edit 1 — line 3285 (PP citation cluster)

Add Hohwy to the existing predictive-processing citation. Change

```latex
\cite{Clark2016, Seth2021, Hoffman2019}
```

to

```latex
\cite{Clark2016, Seth2021, Hoffman2019, hohwy2013predictive}
```

Discharges the staged-but-uncited bib entry at `Attention/references.bib:100`.

### Edit 2 — line 3207 (token-as-agent paragraph)

After the existing sentence on token-level priors as cross-scale shadows, insert:

> This reading of beliefs and frames as agent-internal constructs is the multi-agent realization, on a transformer architecture, of the enactive-inference reading of active inference advanced by Ramstead, Kirchhoff & Friston~\cite{Ramstead2019}, in which the recognition and generative densities are realized as the organism's action-perception coupling rather than as structural representations of an external reality.

Discharges the doubled but uncited bib entries at `Attention/references.bib:595` (`Ramstead2019`) and `:2459` (`ramstead2020variational`). Note that two bib keys point to the same paper — the editor should also de-duplicate the bib file, keeping `Ramstead2019` and removing the `ramstead2020variational` duplicate at `:2459` (or vice versa).

### What was explicitly declined

- A "Constructivism" subsection in the Discussion.
- Glasersfeld 1995 (radical constructivism) citations.
- Piaget 1970 (genetic epistemology) citations.
- Maturana & Varela 1980/1987 (autopoiesis) citations.
- Heinz von Foerster 1981 (second-order cybernetics) citations.

Reasons: the manuscript's substrate commitments at `:3344, :3386, :3442` are incompatible with radical constructivism's substrate-agnosticism; meta-agent formation is variational alignment of pre-existing tuples, not autopoietic component self-production; the manuscript cites Nagel for "no view from nowhere" and does not in fact make von Foerster's stronger observer-dissolution move.

## Follow-up debates (if any)

None required for this claim. Two narrower questions are admissible but optional:

1. **Whether the line-3442 noumenon-as-"structureless until accessed" formulation departs from Kant in a way that requires a more careful philosophical citation** (Hohwy 2013 covers part of this; a closer match may exist in the relational-physicalist or neo-Kantian-structural-realist literature). This is a small downstream question, not a major debate.

2. **Whether the multi-scale meta-agent emergence (sec:meta_agent_emergence) deserves a citation to the dynamical-systems lineage of self-organization (Haken, Kauffman, Bak) rather than to autopoiesis.** Also a narrow question, separate from the constructivism question debated here.

Neither follow-up is required by the present verdict.
