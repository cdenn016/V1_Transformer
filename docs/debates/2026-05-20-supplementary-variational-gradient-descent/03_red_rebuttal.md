# Red Rebuttal — supplementary-variational-gradient-descent

## Concession

I grant sub-claims α through ε. Blue's verification against [Amari 2016 §2.3, §4.3] for Eq. (623–626), [Pennec 2006 §3] and [Bhatia 2007 §6.1, Theorem 6.1.6] for the affine-invariant SPD exponential at Eq. (637–639), [Hall 2015 §2.7] for the SO(3) closed-form `Q_a^L = T_a - c_1(θ) ad_X(T_a) + c_2(θ) ad_X²(T_a)` at Eq. (661), and [Hall 2015 Proposition 2.25] (`Ad_{exp(X)} = e^{ad_X}`) for the trivialization identity `Q_a^R · exp(φ) = exp(φ) · Q_a^L` at line 659 — each holds line-for-line against the canon. The Frobenius trust-region clip on the whitened tangent at Eq. (631) is also standard [Absil–Mahony–Sepulchre 2008 §7]. I do not contest these.

I also grant blue's procedural point that §D.3 line 665 is a one-line preconditioner taxonomy inside a parenthetical, and that the substantive derivations live at §C.5. A reader who follows the cross-reference does land at §C.5 line 606, which carries the corrected pullback metric `G_ab(φ) = tr(Ψ(ad_φ)(T_a)^⊤ · exp(φ) exp(φ)^⊤ · Ψ(ad_φ)(T_b))`.

Consequently the contested ground reduces to sub-claim ζ at line 665, and the dispute is over whether the editorial defects there are below the §C substantive threshold or above the §A / §B narrow threshold. I will argue narrow.

## Core attack

Blue's load-bearing move is the "taxonomy summary that defers to §C.5" framing of line 665. That framing is selective. The actual prose at `Attention/GL(K)_supplementary.tex:665` reads:

> "(iii) the full \emph{pullback} natural gradient $G_{ab}(\phi) = \langle \Psi(\mathrm{ad}_X)(T_a), \Psi(\mathrm{ad}_X)(T_b) \rangle$, where $\Psi(z) = (e^z - 1)/z$, which captures the position-dependent curvature of the exponential map."

This is not a deferred-to-§C.5 placeholder. It is a displayed inline equation written as the definition of mode (iii), introduced by the noun phrase "the full pullback natural gradient" with no qualifier such as "see §C.5 for the corrected form." Two textual facts make this load-bearing rather than ornamental.

First, the formula at line 665 is materially different from the formula at line 606. Line 606 reads (verified via Read of `Attention/GL(K)_supplementary.tex` lines 605–612): `G_ab(φ) = tr(Ψ(ad_φ)(T_a)^⊤ · exp(φ) exp(φ)^⊤ · Ψ(ad_φ)(T_b))`. The `exp(φ) exp(φ)^⊤` factor is not optional — line 612 explains its role: "For non-compact (symmetric) φ, exp(φ) exp(φ)^⊤ = exp(2φ) grows exponentially, automatically compensating the exponential amplification of D exp in the non-compact sector." Strip that factor and the metric is no longer the pullback of the Frobenius inner product on Mat(K) through D exp; it is the metric appropriate to a compact subalgebra where `exp(φ) exp(φ)^⊤ = I`. The prior §C debate adjudicated this against the manuscript precisely because the omitted factor breaks the pullback identity on the non-compact GL(K) sector. Reproducing the omission verbatim at line 665, six weeks later, in a chapter purporting to be a self-contained implementation summary, is not deference — it is repetition.

Second, the parenthetical cross-reference at line 665 is broken. The text reads "(positive-definite on $\mathfrak{sl}(K)$ as derived in App.~C.3)". The §C subsection structure, verified via Grep on `\\subsection\{` in `Attention/GL(K)_supplementary.tex`, is: §C.1 "Structure of the Gauge Frame Gradient" (line 397), §C.2 "Differential of the Matrix Exponential" (line 434), §C.3 "KL Gradient Through Transport" (line 488), §C.4 "Retraction and Numerical Considerations" (line 543), §C.5 "Gauge Frame Preconditioning for GL(K)" (line 558). The Cartan-involution-modified preconditioner and its positive-definiteness on sl(K) live at §C.5, not §C.3. A reader following the cross-reference as written lands in "KL Gradient Through Transport" and finds no preconditioner derivation. Blue's "follows the cross-reference and lands on the canonical form" defense assumes a reader who knows in advance to ignore the section label printed on the page; that is not how cross-references function.

These two textual defects coexist within a single parenthetical clause on a single line. One is a stale formula propagated forward from the pre-§C-debate state of the manuscript; the other is a broken section label. Together they constitute exactly the pattern that produced RED_WINS-narrow on §B (line 337 "Section 3.6" → §3.7 drift) — a cross-reference that points to the wrong location combined with a forward-propagated formulation that does not match the corrected derivation. Calling line 665 a "taxonomy summary" does not make the printed formula or the printed section label disappear.

Blue's pre-emption argument adds that "no code path in `transformer/aif/`, `transformer/vfe/`, or the legacy `transformer/core/variational_ffn.py` reads §D's text rather than §C.5's derivation." That is true and I grant it. It is also irrelevant to the editorial threshold: the §A and §B narrow verdicts did not require a consuming code path either. The bar for narrow editorial scope is textual inconsistency that a reader is expected to detect and that the authors are expected to fix; the bar for substantive scope is a derivation that, taken on its own face, produces a wrong downstream result. Line 665 is the former.

## Defense

My falsification target is sub-claim ζ as stated in `00_claim.md`: "the three preconditioner modes referenced as '(i) Cartan-involution-modified', '(ii) Cartan decomposition', '(iii) full pullback natural gradient' point to the §C.5 preconditioning section." Two of the three textual referents at line 665 do not in fact "point to" §C.5: (a) the printed section label is "App.~C.3", which by the verified subsection structure (Grep output above) is "KL Gradient Through Transport", not "Gauge Frame Preconditioning"; (b) the inline formula for mode (iii) is the pre-corrected form `G_ab(φ) = ⟨Ψ(ad_X)(T_a), Ψ(ad_X)(T_b)⟩`, which is not the form at §C.5 line 606. Sub-claim ζ as written fails on both the cross-reference and the formula.

Against this, sub-claim ζ does not fail in the substantive sense that produced the §C verdict. The §C debate identified three formula errors that, when implemented, produced numerically wrong gradients — failures verified by finite-difference smoke tests. Line 665 in §D.3 is not implemented; the parallel code path in `transformer/aif/` reads §C.5's corrected derivation rather than §D.3's text, per blue's pre-emption (which I granted above). Sub-claims α, β, γ, δ, ε survive without revision. The chapter's mathematical content is canon-correct under blue's [Amari 2016], [Pennec 2006], [Bhatia 2007], [Absil–Mahony–Sepulchre 2008], [Hall 2015] citations.

The calibrated red verdict is therefore RED_WINS-narrow with three editorial fixes, matching the verdict profile of §A and §B in the audit series:

1. Update line 665 mode (iii) inline formula to `G_ab(φ) = tr(Ψ(ad_φ)(T_a)^⊤ · exp(φ) exp(φ)^⊤ · Ψ(ad_φ)(T_b))`, matching §C.5 line 606; or strip the inline formula entirely and write "(iii) the full pullback natural gradient (see App.~C.5 Eq. \ref{eq:pullback_metric})". Either resolves the formula inconsistency.

2. Fix the cross-reference at line 665 from "App.~C.3" to "App.~C.5", matching the subsection structure verified via Grep on `\\subsection\{` in `Attention/GL(K)_supplementary.tex` (§C.5 at line 558 is "Gauge Frame Preconditioning for GL(K)").

3. Add [Pennec 2006] and [Bhatia 2007] citations to the SPD retraction at Eq. (637–639), matching the canonical references blue used in defending sub-claim γ. These canonical sources are absent from the chapter as printed; their addition is a citation-gap fix comparable to the §B citation gaps that produced RED_WINS-narrow there.

The compound claim asserts the chapter is "complete and mathematically/theoretically pure as a self-contained supplementary chapter." Sub-claims α through ε confirm the math content is pure. Sub-claim ζ as written is not satisfied at line 665 — the printed text contains a stale formula and a broken cross-reference. Editorial fixes 1 and 2 restore the assertion; fix 3 closes the canonical citation gap that the §B verdict pattern flagged elsewhere.

I do not push for substantive scope. The §C debate's substantive verdict required FD-verified formula errors in active code paths; §D.3 line 665 is text-only, not consumed by code. Narrow scope, three concrete edits, matches §A and §B.
