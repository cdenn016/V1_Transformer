# Verdict — supplementary-covariance-dynamics

## Outcome

RED_WINS-narrow

## Decisive evidence

Five concrete, citeable editorial breaks in `Attention/GL(K)_supplementary.tex` lines 180-387, conceded as factual by Blue in `03_blue_rebuttal.md` lines 5-11 and verified by Red via direct file inspection in `03_red_rebuttal.md` lines 15-43:

1. `Attention/GL(K)_supplementary.tex:275` — first use of `α_i` in §B with no internal definition; the symbol appears with "unit coefficient α_i = 1" before §B has introduced it.
2. `Attention/GL(K)_supplementary.tex:337` — plain-text cross-reference to "main text, Section~3.6" for the state-dependent prior coupling, while the canonical location is `Attention/GL(K)_attention.tex:895` §3.7 "State-Dependent Prior Precision" carrying `\label{sec:state_dependent_precision}` (confirmed at `Attention/GL(K)_attention.tex:408` parameter table).
3. `Attention/GL(K)_supplementary.tex:385` — `Λ_o` used in the stability disclaimer with no defining equation anywhere in the file; semantically equivalent to `R^{-1}` introduced at line 253 but never identified.
4. `Attention/GL(K)_supplementary.tex:380` — `⊠_{sym}` notation used in the Hessian display with a one-line prose gloss but no formula and no citation to the SPD-manifold canon (Smith 2005 / Pennec 2006) where the notation is defined.
5. `Attention/GL(K)_supplementary.tex` lines 180-387 — zero `\cite/\citep/\citet` calls within §B's 207 lines, while four load-bearing canonical correspondences are stated as standard: Gaussian KL closed form at lines 203-221 (Bishop 2006 §2.3.6 / Murphy 2012 §2.3.2 / Cover-Thomas 2006 §8.6), matrix-calculus derivative at lines 227-232 (Petersen-Pedersen §9.1, §9.4), Gaussian linear pushforward at line 234 (Bishop 2006 §2.3.3), and SPD Hessian at line 380 (Smith 2005 / Pennec 2006). The editorial precedent at `Attention/GL(K)_attention.tex:921` (the §3.7 Gamma-prior fix carrying `\citep{bishop2006pattern,murphy2012machine}`) establishes that load-bearing canonical correspondences in this manuscript carry inline citations.

## Reasoning

The compound claim has two prongs. The mathematical-purity prong (sub-claims α through ζ) survives decisively — Red conceded all six in `03_red_rebuttal.md` lines 4-10 against primary-source canon (Bishop 2006 §2.3.6 Eq. 2.121, Murphy 2012 §2.3.2, Cover-Thomas 2006 §8.6 Eq. 8.69, Petersen-Pedersen §9.1 Eq. 75 and §9.4 Eq. 100, Smith 2005 / Pennec 2006). Blue's envelope-theorem defense of the F_i Hessian extension at `03_blue_rebuttal.md` lines 17-23, anchored to the explicit invocation at `Attention/GL(K)_supplementary.tex:273` referencing `Attention/GL(K)_attention.tex:859-874` Eq. `eq:autograd_envelope_gap`, closes Red's sub-claim ζ attack. The math is six-for-six against canon.

The self-containment prong fails. The compound claim's literal wording in `00_claim.md` line 11 is "complete and mathematically/theoretically pure as a self-contained supplementary chapter" with "no residual theoretical-purity issues comparable in magnitude to those identified and corrected in the twelve-debate audit series." Blue conceded the three Red-opening editorial defects as factual (`03_blue_rebuttal.md` lines 5-11) and Red supplied two additional concrete breaks in rebuttal (the §B-internal absence of an `α_i` definition at line 275; the §B.2.2 regime gated by `α_i ≪ 1` against the framework default `α_i = 1` at `Attention/GL(K)_attention.tex:948`). The aggregate is five-to-six concrete editorial breaks. Debate 3 (§3 Gauge-Covariant VFE) produced RED_WINS-narrow on three editorial gaps and Debate 12 (§A General Mathematical Framework) produced RED_WINS-narrow on four editorial gaps. The §B aggregate is at or above that established threshold.

Blue's defense that each break is "a single-token / single-line / citation-addition fix" is correct as a remediation-scope statement but does not refute the threshold question. The methodology forbids splitting differences when one side has cited and the other has not — both sides here cited primary evidence — so the verdict is calibrated by the binding wording of the compound claim. The "comparable in magnitude to those identified and corrected in the twelve-debate audit series" clause is the explicit comparison the claim invites the judge to apply. Five-to-six concrete editorial breaks exceed the three-gap (Debate 3) and four-gap (Debate 12) precedents.

The verdict reflects the worst load-bearing sub-claim per `00_claim.md` line 47. The math survives; the self-containment clause does not. RED_WINS-narrow is the appropriate verdict, restricted to the editorial self-containment prong. The mathematical content of sub-claims α through ζ is accepted as canon-verified and requires no derivation changes.

## Action

Apply six scoped edits to `Attention/GL(K)_supplementary.tex` lines 180-387, none of which touches the boxed equations, the regime equations, or the derivation chain:

1. At line 275, define `α_i` at first use in §B or insert `(see main text \ref{sec:state_dependent_precision})` so the symbol is grounded inside the chapter.
2. At line 337, replace `cf.\ main text, Section~3.6` with `cf.\ main text \ref{sec:state_dependent_precision}` (which resolves to §3.7).
3. At line 385, insert "where `\Lambda_o := R^{-1}` is the observation precision (cf. line 253)" or directly substitute `R^{-1}` to match §B-internal notation.
4. At line 380, either write the explicit Hessian action `\mathrm{Hess}[H,H] = (1/2)\,\mathrm{tr}(\Sigma_1^{-1} H \Sigma_1^{-1} H)` or append `\citep{smith2005covariance,pennec2006riemannian}` to ground the `⊠_{sym}` notation.
5. Append inline citations to load-bearing canonical correspondences: `\citep{bishop2006pattern,murphy2012machine,coverthomas2006elements}` at the Gaussian KL display (line 221); `\citep{petersen2012matrix}` at the matrix-calculus derivative (line 232); `\citep{bishop2006pattern}` at the linear pushforward (line 234).
6. At §B.2.2 (around line 337-348), flag the alignment-dominated regime as a theoretical limit not exercised under the canonical default `α_i = 1` per `Attention/GL(K)_attention.tex:948`. A single sentence suffices: "This regime is studied as a theoretical limit; the canonical configuration uses `α_i = 1` per main paper §3.7."

These edits close the self-containment prong without altering any of the six mathematical sub-claims, all of which have been verified against canon and stand intact.
