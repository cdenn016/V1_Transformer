# Red Memo — variational — Phase 3 rebuttal

Target: `02_blue_opening.md` placeholder-isolation defense (the "narrow editorial leak" framing of PIFB:2807 and PIFB:3157), and Falsification condition 1.

## Concession

Blue's invocation of the variational-inference literature's separation between structural-existence claims (a variational family exists; a posterior is well-defined) and empirical-fit claims (the approximation has small KL on a specific dataset) [BleiKucukelbirJordan 2017 §5; ParrPezzuloFriston 2022 Ch. 14] is canonically appropriate in the abstract. Structural-existence content can stand independently of empirical fit in canonical variational-inference practice. Granted as a general principle.

## Core attack

Blue's own Falsification Condition 1 reads (verbatim from `02_blue_opening.md`):

> Some sentence in §Speculative Extensions makes a present-tense empirical claim that depends on placeholder §Results data and is not explicitly qualified within the same paragraph or its parent subsection. The candidate sites are 2807 ("validated empirically"), 3070 (which is explicitly qualified within its own paragraph), and 3157 (which is explicitly qualified within its own sentence). If red can produce a sentence outside these three where a present-tense empirical claim depends on placeholder data without local qualification, the placeholder-isolation sub-claim (sub-claim 8) fails.

The trip happens *at site 2807 itself*, by blue's own rule applied to the actual text. Verbatim from `Attention/Participatory_it_from_bit.tex:2807-2808`:

> 1. **$\mathrm{GL}(K, \mathbb{R})$ with real Gaussians** (validated empirically in Section~\ref{sec:transformers}): Demonstrates that the full non-compact gauge symmetry produces meaningful dynamics. The Fisher-Rao metric remains Riemannian, but the gauge structure is richer than $\mathrm{SO}(3)$.

The 2807 line carries no within-paragraph qualifier, no parent-subsection qualifier ("Postulates Required for an Indefinite Pullback" at 2803 has no placeholder hedge), and no within-sentence qualifier. The qualification blue's defense reaches for ("the structural use is the step-1 statement 'full non-compact gauge symmetry produces meaningful dynamics' — a structural-existence claim about the framework") is not in the text; it is post-hoc rationalization in `02_blue_opening.md` itself.

Worse, the qualifier blue claims is *structural* — "produces meaningful dynamics" — does not save the leak. "Validated empirically" is a present-tense empirical claim; "meaningful dynamics" is a present-tense empirical claim; the cited section `sec:transformers` is the section the user has flagged at `01_evidence.md` line 32 as placeholder pending the operationally-independent $\omega^2 \propto \Sigma_p^{-1}$ test. The multi-seed scaling fit at `Participatory_it_from_bit.tex:2559-2577` ($b = -1.049$, 95% CI $[-1.103, -0.998]$, $R^2 \approx 0.9998$, nested-$F$ test $F(1,8)=9.73, p=0.014$ rejecting $b=-1$) is the evidence base the 2807 forward-reference inherits, and the user has registered that data as placeholder.

By blue's own Falsification Condition 1, blue's own defense, applied to the actual text, trips: placeholder-isolation sub-claim 8 fails at 2807. Blue's "structural use" rescue is not present in the manuscript text — only in blue's opening — and the canon-cop disallows reasoning from blue's own opening as if it amended the manuscript.

Compound at 3157: verbatim from `Participatory_it_from_bit.tex:3157`:

> This is a within-framework observation about the threshold-detector single-seed dynamics of Section~\ref{sec:results}.

The 3157 sentence is the second-to-last sentence of the closing paragraph "Sustained Non-Equilibrium under the Threshold-Detector Dynamics" at 3151–3157. The preceding sentence at 3155 ("The system instantiates a structurally analogous version of what Wheeler called a 'self-excited circuit'") attaches the Wheeler self-excited-circuit framing to "the system" — meaning the specific threshold-detector single-seed run. Blue's defense reads the 3157 single-seed registration as an honest qualifier; this is one valid reading, but the load-bearing content of 3155–3157 *attaches* the Wheeler-framing to the single-seed run, which is exactly what placeholder isolation requires not to happen at the load-bearing layer.

Two leak sites are not "narrow editorial leaks"; they are the structural verification that the placeholder-isolation sub-claim does not hold as stated. Sub-claim 8 fails on the canonical reading.

## Defense

The canonical variational-inference treatment blue invokes [BleiKucukelbirJordan 2017 §5; ParrPezzuloFriston 2022 Ch. 14] requires a *clean separation* between structural-existence content and empirical-fit content, deployed via local within-paragraph hedges that bind the empirical content to its actual empirical status. PIFB delivers this clean separation at PIFB:3070 (verified: "no operationally independent measurement is reported in this manuscript" appears within the same paragraph as the mass identification claim). PIFB does *not* deliver this clean separation at 2807 or at 3155–3157.

The within-paragraph standard is what makes structural-existence claims defensible under canonical variational-inference practice. A structural-existence claim that forward-references a placeholder empirical result without a local hedge inherits the placeholder's empirical status by direct citation. This is the canonical-VI failure mode that blue's pillar-3 defense names in the abstract but does not apply to the specific text under debate.

Sub-claim 8 fails at 2807 by blue's own falsification condition; this is the cleanest single-site take-down available in the present text. The corresponding action is RED_WINS_NARROW with a required editorial fix at 2807 (replace "validated empirically in Section~\ref{sec:transformers}" with the user-decision-honoring form "exhibited in the present scaling protocol of Section~\ref{sec:transformers}, whose specific exponent value is registered as placeholder pending the operationally-independent test"). Without that fix, the rock-solid claim does not hold at sub-claim 8.

## Newly-discovered canon

- **ParrPezzuloFriston 2022 *Active Inference* §3.3, §3.4** — variational-inference treatments require local registration of approximation status; the canonical practice is paragraph-level hedge attachment, not section-level cross-reference.
- **BleiKucukelbirJordan 2017 *J. Amer. Statist. Assoc.* 112 §5** — discussion of structural-vs-empirical content separation; the section discusses approximation gaps and interpretive choices, with the load-bearing operative point being that empirical claims inherit empirical-status registrations from their data, not from their structural roles.
- **`01_evidence.md` lines 28–47** — the placeholder-experiment context in the present debate's evidence pack. The 2807 forward-reference points into the placeholder-flagged scaling-validation subsection at PIFB:2559–2577 by direct cite; the user has registered all prior PIFB experimental data as placeholder per memory `pifb-mass-todo-plan`.
