# Blue Rebuttal — supplementary-variational-gradient-descent

## Concession

I grant two of red's three primary findings outright.

**Concession 1 — Issue 2 (broken cross-reference "App.~C.3").** Red is correct on the LaTeX. Grep over `\subsection{` in `Attention/GL(K)_supplementary.tex` returns §C.3 at line 488 as `KL Gradient Through Transport` and §C.5 at line 558 as `Gauge Frame Preconditioning for GL(K)` with label `sec:glk_preconditioning` (line 559). The Cartan-involution-modified bilinear form is derived in §C.5 (lines 584–594), not §C.3. The textual reference "App.~C.3" at line 665 is a plain-text pointer — there is no `\ref{}` macro — and it lands in the wrong subsection. This is a single-token fix: "App.~C.3" → "App.~C.5". The defect is real.

**Concession 2 — Issue 1 (stale pullback formula at line 665, internal inconsistency).** Red is correct that line 665 states `G_{ab}(φ) = ⟨Ψ(ad_X)(T_a), Ψ(ad_X)(T_b)⟩` without the `exp(φ)exp(φ)^⊤` factor, and that this is non-equivalent on the non-compact symmetric sector of `gl(K)` to the corrected §C.5 line 606–608 form `tr(Ψ(ad_φ)(T_a)^⊤ · exp(φ)exp(φ)^⊤ · Ψ(ad_φ)(T_b))`. The pullback-of-Frobenius derivation `df = D_φ exp = Ψ(ad_φ) · exp(φ)` (right-trivialised, [Hall 2015 §2.7, Theorem 5.4]) `⇒ f*h(T_a, T_b) = tr(Ψ(ad_φ)(T_a)^⊤ · exp(φ)exp(φ)^⊤ · Ψ(ad_φ)(T_b))` is the canonical pullback metric [Lee 2013 §13]. Line 665 is the pre-correction form and is internally inconsistent with §C.5 post-correction. The fix is bounded: replace the formula with the §C.5 form, or remove the explicit equation and refer to §C.5 directly.

These two concessions, plus Issue 3 (missing [Pennec 2006] / [Bhatia 2007] / [Absil-Mahony-Sepulchre 2008] citations for Eq. 637–639) which I also concede, give a defect set comparable to the §A and §B narrow verdicts (4 fixes each).

## Core attack

Red's load-bearing move is the scope escalation from narrow-editorial to substantive, justified by the claim that "the §D.3 formula is now internally inconsistent with the corrected §C.5 formula and the pointer to the derivation is dead, sub-claim ζ fails on its own terms, and the chapter is not pure as a self-contained unit." Red argues this is "substantive (not narrow-editorial)" and "comparable in magnitude to the §C verdict, not the narrow §A/§B verdicts."

This conflates two distinct questions: (a) is the line-665 formula a stale duplicate of a load-bearing equation that the codebase consumes, in which case a propagated bug is substantive — the §C precedent — or (b) is the line-665 formula a summary pointer to §C.5, in which case the residual inconsistency is editorial and reduces to "duplicate equation needs to be updated when its primary copy changes." Red asserts (a) by reading line 665 as "stated as a definition." The codebase falsifies (a).

`transformer/vfe/config.py:597-609` explicitly forbids `phi_preconditioner='pullback'` from the inner E-step path:

```
if self.phi_preconditioner not in (
    'clip', 'cartan', 'killing', 'killing_per_block'
):
    raise ValueError(
        f"phi_preconditioner={self.phi_preconditioner!r} is not supported. "
        ...
        f"'pullback' is gated here because the inner E-step call chain "
        f"in `VFEEStep._update_phi` does not thread the structure_constants "
        f"tensor that apply_pullback_natural_gradient requires. The metric "
        f"math itself was corrected on 2026-05-20; the corrected formula is "
        f"already available via the outer optimizer "
        f"RiemannianAdamW(metric='pullback')."
    )
```

The pullback preconditioner is reachable in the codebase only through `RiemannianAdamW(metric='pullback')`, which calls `build_pullback_metric_tensor` at `transformer/core/gauge_preconditioner.py:582` (definition at line 596). That function implements the corrected §C.5 form with the `exp(φ)exp(φ)^⊤` factor — verified in the §C debate (commit `e4481f7c`). No active code path in the repository consumes the line-665 formula. The bug is text-only: it lives in one paragraph of `GL(K)_supplementary.tex`, not in any callable.

This matters because the §C verdict went substantive on the strength of three FD-verified formula errors (lines 590, 599, 606) where the codebase consumed the stale forms — the bug existed in running code. §D inherits one stale formula at line 665, but the codebase already prevents it from being consumed. The aggregate scope is: one stale duplicate formula (line 665), one broken cross-reference (line 665), missing citations (Eq. 637–639), and the three minor editorial gaps (Issues 4–6). That is the §A/§B narrow profile, not the §C substantive profile.

Red's counter at Falsification (3) — "a summary that points to a non-existent derivation and states an outdated formula is not a complete and pure self-contained supplementary chapter" — is a stronger version of the editorial claim, not a substantive one. It says: "the textual presentation is broken in a way that matters." I grant the editorial brokenness. What I deny is the verdict-scope escalation: a broken pointer plus a stale duplicate of an equation derived elsewhere is the same defect class as the §B "Section 3.6 → §3.7" drift and the §A citation gaps, both of which earned narrow verdicts.

## Defense

The defensible verdict is RED_WINS-narrow, matching §A and §B, with a remediation set of three to five editorial fixes:

1. Update line 665 pullback formula to match §C.5 line 606–608, or remove the explicit equation and write "see App.~C.5, Eq. (...) for the position-dependent pullback metric." This is a single-equation edit. Justification: red's pullback-of-Frobenius derivation via [Lee 2013 §13] and [Hall 2015 §2.7, Theorem 5.4] is canonical, and §C.5 line 608 already states it correctly.

2. Update line 665 cross-reference "App.~C.3" → "App.~C.5". Single-token edit. Justification: §C.5 (label `sec:glk_preconditioning`) is the actual location of the Cartan-involution-modified derivation; §C.3 is `KL Gradient Through Transport`.

3. Add canonical citations to Eq. (637–639): [Pennec 2006 "Intrinsic statistics on Riemannian manifolds"], [Bhatia 2007 §6.1 "The Riemannian metric"], [Absil-Mahony-Sepulchre 2008 §5.4.6 "The set of symmetric positive-definite matrices"]. The affine-invariant exponential map is a textbook result and the chapter currently presents it citation-free.

4. (Optional) Cite [Higham 2008 §10.2, Algorithm 10.27] at line 663 for the symmetric-quadrature Fréchet derivative path. `Higham2008` is already in `references.bib` and cited at supplementary line 556.

5. (Optional) Provide a code reference for the FD-validation claim at line 670 (e.g., `tests/transformer/test_gauge_preconditioner.py`), and justify the `|τ λ_B^{(j)}| ≤ 50` clipping bound at line 640 against the spectral floor `ε_SPD = 10^{-4}` and condition cap `κ_max = 10^4`.

Sub-claims α through ε remain primary-source verified. Red concedes α via the sympy check (red opening, evidence bullet 6): `Trace((G + G.T)*v)/2 = Trace(((1/2)*G + (1/2)*G.T)*v)`, matching [Amari 2016 §4.3]. β through ε are verified against [Pennec 2006], [Bhatia 2007 §6.1], [Hall 2015 §2.7], and [Lee 2013 §13] respectively. The mathematics of the chapter is canonical. Only sub-claim ζ — the cross-reference summary at line 665 — is broken, and it is broken in the editorial way: the duplicate equation drifted from its primary copy when §C.5 was corrected, and the typed cross-reference points to the wrong subsection.

The scope question reduces to: when a chapter contains a summary pointer to material derived elsewhere, and the summary statement is incorrect while the primary statement is correct, is that substantive or editorial? The §A/§B precedent (manuscript cross-reference drift, citation gaps, undefined notation) treats this as editorial. The §C precedent (formula errors that the codebase consumes, FD-verifiable in running code) treats this as substantive. The line-665 defect sits squarely in the first category: the codebase does not consume it, the corrected formula is reachable in the codebase via `build_pullback_metric_tensor` (`transformer/core/gauge_preconditioner.py:596`), and the inner E-step explicitly raises if the user tries to instantiate the stale path. The remediation is three single-line edits to `Attention/GL(K)_supplementary.tex`. That is RED_WINS-narrow.

## Falsification of this rebuttal

This defense is wrong if any of the following is demonstrated:

1. The line-665 formula is consumed by an active code path that the §C debate did not fix. Verifiable by `Grep` over the repository for any caller that constructs `G_{ab}` without the `exp(φ)exp(φ)^⊤` factor and is reachable under a documented config. I have searched `transformer/core/gauge_preconditioner.py` and `transformer/vfe/` and found only the corrected form is reachable; the stale form is guarded out at `vfe/config.py:597-609`.

2. The §C verdict's "substantive" classification was justified specifically by manuscript-only inconsistency, not by codebase consumption. If the §C debate artifacts show the substantive scope was earned by the manuscript drift alone (not by FD-verified running-code errors), then the §D inherited drift is symmetric and the verdict should match §C.

3. A reasonable implementer reading §D.3 in isolation, without §C.5, would build a system whose runtime behavior differs from the corrected codebase. The standard manuscript reading order has §D referring back to §C (line 663 explicitly cites Appendix C; line 665 explicitly says "as derived in App.~C.[5]"); the chapter is structured to be read with §C.5 in scope. An implementer who reads only §D.3 and ignores the cross-references is constructing from an incomplete source, but the chapter does not claim to be a standalone reference, only "complete and mathematically/theoretically pure as a self-contained supplementary chapter" of a manuscript whose chapter ordering is fixed.
