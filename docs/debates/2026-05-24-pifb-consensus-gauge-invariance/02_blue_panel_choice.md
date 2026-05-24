# Blue Panel Choice — pifb-consensus-gauge-invariance (Phase 2, opening)

Environment note: no `Agent` tool is available in this dispatch. Per the coordinator
fallback protocol I embody the five lenses below myself and log a per-lens memo for
each (`memo_blue_<expert>.md`). The synthesized opening is `02_blue_opening.md`.

## Selected experts (5 of 10)

| Expert | One-sentence justification |
|--------|----------------------------|
| `gauge-theorist` | The claim's load-bearing facts are gauge-theoretic — trace-cyclicity invariance of `tr(A_μ A_ν)` under constant `g`, the Maurer–Cartan inhomogeneous term under local `g(c)`, and Faddeev–Popov gauge-fixing of the orbit integral; this lens verifies each against connection canon. |
| `geometer` | The consensus metric is a (0,2)/(2,0) tensor pulled back across fibers, and the constant-`g` adjoint action `A → g⁻¹ A g` plus the orbit-average construction are differential-geometric objects whose stated properties must be checked against bundle-transport canon. |
| `info-geometer` | The GL(K,ℝ)-invariance of Gaussian KL that the whole "consensus" interpretation rests on is an information-geometry fact (Čencov/sufficient-statistic invariance); this lens establishes that the invariance is a *setup property* of the divergence, which is exactly what makes the "derivation" reading circular. |
| `variational` | "Gauge invariance arises as a consistency requirement for multi-agent consensus" is at root an inference-theoretic claim about agents agreeing; this lens polices whether "consensus" does any work beyond restating an invariance already imposed on the belief fiber. |
| `philosophy-of-science` | Mandatory. Frame-checks whether the claim is "the section is honestly characterized" (defensible) versus "the construction works" (not claimed), audits the self-flag for falsifiability and circularity, and catches any defense that cites the manuscript back at itself. |

## Experts not selected

- `transformer-ml`, `ml-engineer`, `numerical-analyst`, `implementation-engineer`,
  `code-quality` — the claim is theory/formal-math only (Haar measure, gauge transformation,
  functional integral); `code-truth` is marked N/A in `00_claim.md`. No config trace, no
  runtime path, no numerical-stability question is in scope.
