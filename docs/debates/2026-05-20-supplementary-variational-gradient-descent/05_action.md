# Action — supplementary-variational-gradient-descent

**From verdict:** RED_WINS (narrow editorial scope)

## Summary of verdict

The compound claim that §D of `Attention/GL(K)_supplementary.tex` (lines 617–670) is "complete and mathematically/theoretically pure" fails on sub-claim ζ (preconditioner cross-references at line 665). Sub-claims α (natural-gradient projections), β (whitening + trust region), γ (SPD retraction), δ (SO(3) Ψ_L closed form), and ε (trivialization relation) all survive primary-source verification against [Amari 2016 §2.3/§4.3, Pennec 2006 §3, Bhatia 2007 §6.1, Absil-Mahony-Sepulchre 2008 §5.4.6, Hall 2015 §2.7 / Proposition 2.25]. The defects on sub-claim ζ are confined to textual cross-document drift — the corrected pullback formula at §C.5 line 606 (post commit `e4481f7c`) was not propagated to §D.3 line 665, leaving a stale inline formula plus a broken cross-reference. Both teams converged on RED_WINS-narrow as the appropriate verdict.

This matches the §A / §B narrow precedent, not the §C substantive precedent. Active code paths consuming the pullback metric (`RiemannianAdamW(metric='pullback')` → `build_pullback_metric_tensor`) use the corrected formula; the inner E-step pullback mode is gated by `vfe/config.py:591-600`. Manuscript-only drift.

## Recommended action

Three editorial corrections to `Attention/GL(K)_supplementary.tex` §D.3 and §D.2. No mathematical-content changes; the math of §D.1, §D.2, §D.3 all survives.

### Edit 1 — update stale pullback formula at line 665

Current text:

> "(iii) the full *pullback* natural gradient `G_{ab}(φ) = ⟨Ψ(ad_X)(T_a), Ψ(ad_X)(T_b)⟩`, where `Ψ(z) = (e^z - 1)/z`, which captures the position-dependent curvature of the exponential map."

Replace with either (a) the corrected formula matching §C.5 line 606:

> "(iii) the full *pullback* natural gradient `G_{ab}(φ) = tr(Ψ(ad_φ)(T_a)^⊤ · exp(φ) exp(φ)^⊤ · Ψ(ad_φ)(T_b))` (see Eq.~\ref{eq:pullback_metric}), where `Ψ(z) = (e^z - 1)/z`, which captures the position-dependent curvature of the exponential map through the right-trivialised Fréchet derivative `D_φ exp[T_a] = Ψ(ad_φ)(T_a) · exp(φ)`."

or (b) remove the inline formula and forward-reference §C.5:

> "(iii) the full *pullback* natural gradient (Eq.~\ref{eq:pullback_metric} in Appendix~\ref{sec:glk_preconditioning}), which captures the position-dependent curvature of the exponential map through the right-trivialised Fréchet derivative of `\exp`."

Option (b) is the lighter-weight remedy; option (a) keeps the formula visible at the cost of redundancy.

### Edit 2 — fix cross-reference "App.~C.3" → "App.~C.5"

Current text at line 665: "(positive-definite on `sl(K)` as derived in App.~C.3)".

Replace with: "(positive-definite on `sl(K)` as derived in Appendix~\ref{sec:glk_preconditioning})".

§C.5 is labelled `sec:glk_preconditioning` at line 559 of the supplementary.

### Edit 3 — add citations for SPD retraction at §D.2

Current text at Eq. (637–639) presents the affine-invariant SPD exponential map without citation. Add citations to the canonical sources:

After "is performed using the affine-invariant exponential map" at line 636, append: "\citep{Pennec2006,Bhatia2007,Absil2008}".

The retraction formula at Eq. (637–639) is the canonical [Pennec 2006] / [Bhatia 2007 §6.1] / [Absil-Mahony-Sepulchre 2008 §5.4.6] affine-invariant exponential map on SPD(K). This brings §D.2's citation density up to the editorial standard set by the §B Killing-form citation fix.

## Bib additions

Verify in `Attention/references.bib`:
- `Pennec2006`: not yet verified — likely needs to be added.
- `Bhatia2007`: not yet verified — likely needs to be added.
- `Absil2008`: not yet verified — likely needs to be added.

Standard entries:
```
@article{Pennec2006,
  author = {Pennec, Xavier},
  title  = {Intrinsic Statistics on {Riemannian} Manifolds: Basic Tools for Geometric Measurements},
  journal = {Journal of Mathematical Imaging and Vision},
  volume = {25}, number = {1}, pages = {127--154}, year = {2006}
}

@book{Bhatia2007,
  author    = {Bhatia, R.},
  title     = {Positive Definite Matrices},
  publisher = {Princeton University Press},
  series    = {Princeton Series in Applied Mathematics},
  year      = {2007}
}

@book{Absil2008,
  author    = {Absil, P.-A. and Mahony, R. and Sepulchre, R.},
  title     = {Optimization Algorithms on Matrix Manifolds},
  publisher = {Princeton University Press},
  year      = {2008}
}
```

## Sub-claims α-ε

No action required. All math content of §D.1, §D.2, §D.3 verified against canonical sources. The derivation chain is mathematically pure.

## Cumulative debate-series state

Fifteenth debate in the gauge-transformer audit series. Closed queue:

1. §5 transformer reduction (RED_WINS).
2. Softmax-β stationarity (RED_WINS).
3. Sub-claim A flat bundle (BLUE_WINS).
4. Sub-claim B degenerate Σ (BLUE_WINS).
5. Sub-claim C QK^T identification (BLUE_WINS).
6. Sub-claim D V identification (BLUE_WINS).
7. Canonical F vs surrogate (RED_WINS).
8. Multi-head block-diagonal (BLUE_WINS).
9. Route 1 untied carving (RED_WINS).
10. FFN softmax-gradient correction (RED_WINS).
11. §3 Gauge-Covariant VFE (RED_WINS narrow).
12. Supplementary §A General Mathematical Framework (RED_WINS narrow).
13. Supplementary §B Covariance Dynamics (RED_WINS narrow).
14. Supplementary §C Gauge Frame Gradients (RED_WINS substantive — 3 FD-verified math errors + codebase fix).
15. **Supplementary §D Variational Gradient Descent (RED_WINS narrow, this debate).**

The main paper §3–§5 and supplementary §A–§D are now fully audited. Optional follow-up: Participatory_it_from_bit.tex §Theory.

## Follow-up debates

None required from this verdict. The line-665 stale formula and the "App.~C.3" cross-reference are direct downstream consequences of the §C debate's corrections that were not propagated forward; this debate closes that propagation gap.
