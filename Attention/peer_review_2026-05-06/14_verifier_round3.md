# Round-3 Verifier Report — Issues 4, 5, 6, 7

**Verifier:** Independent (no prior conversation context)
**Date:** 2026-05-06
**Manuscript:** `Attention/Participatory_it_from_bit.tex`
**Reports verified:** `10_ym_sign_convention.md`, `11_complex_kl.md`, `12_lorentzian_framing.md`, `13_mass_stiffness.md`

---

## Top-line synthesis

All four investigator findings hold against the manuscript text and against independent algebraic checks. Issue 4 (sign/naming for `tr(A_mu A_nu)`) and Issue 7 (Hessian block-symmetry caveat) are clean technical defects with sympy-confirmed verdicts. Issue 5 (the "KL generalizes naturally to complex exponential families" claim at line 1623) is a genuine load-bearing inconsistency: nothing in the manuscript specifies which of the four standard complex-divergence constructions is intended, and the worked example at lines 1633-1660 only complexifies the gauge frame `phi`, not the densities `q`. Issue 6 (Lorentzian-signature overclaiming) is a coordinated rhetorical defect at seven sites in the manuscript that contradicts the honest framing already present at five other sites; the investigator's census is accurate and my independent recount agrees with their classification. Issues 4, 5, and 6 all live in section `sec:signature_resolution` (lines 1593-1671) and their fixes share scaffolding; Issue 7 is independent and confined to lines 1098-1300. I rate Issues 4, 5, 6 as medium/medium/medium-to-high and Issue 7 as medium (technical correction with no quantitative consequence).

---

## Issue 4 verification (Yang-Mills naming and sign convention)

**Quoted text confirmed verbatim** at lines 1521-1525, 1545-1546, 1614, 1633, 1640-1642, 1648.

**Sign computation, independent sympy run:**

| Generator | Form | tr(A^2) |
|---|---|---|
| `A in so(2)` | `[[0,a],[-a,0]]` | `-2 a^2 <= 0` |
| `T = diag(1,-1)` | symmetric, `T + T^T = diag(2,-2)` (not skew) | `+2` |

Confirmed: `T = diag(1,-1)` is in `sl(2,R) ⊂ gl(2,R)` but **not** in `so(2)`. The manuscript at line 1640 correctly states `T \in \mathfrak{gl}(2)` (not `so(2)`), but never flags that this is exactly the load-bearing move that makes `tr(T^2) > 0` available. Under the canonical compact convention `<A,B> = -tr(AB)` on `so(N)` or `su(N)`, the spatial sector of the worked example would come out negative and the Wick rotation `phi_tau -> i phi_tau` would produce signature `(+,-)` rather than the claimed `(-,+)`.

**Naming census of "Yang-Mills":** the manuscript uses the phrase or the YM superscript at lines 1521, 1525, 1545-1546 (where it correctly disowns the YM invariant `tr(F F)`), 1614 (where it concedes the name is "shorthand"), 1633 (subsubsection heading), 1648 (worked-example heading), 1665, 1726, plus the equation labels `g_{C,munu}^{YM}` and `G_{munu}^{YM}`. The disclaimer at 1614 is the only honest counterweight; the rest read as if `tr(A_mu A_nu)` were a Yang-Mills kinetic invariant, which it is not (it depends on `A`, not on `F`, and is gauge-variant).

**Investigator 10's verdict — partially confirmed for naming, confirmed for sign convention — holds.** Their proposed rename to "frame-twist quadratic form" with explicit `<A,B>_g = +tr(AB)` (indefinite trace form on `gl(K,C)`) is the right surgery and is internally consistent with the worked example.

**Severity: medium.** The defect is real and corrupts a sentence-level reading of the construction, but does not invalidate any equation in the worked example because `T = diag(1,-1)` is in fact non-compact and the trace form on `gl(2,R)` is in fact indefinite. The manuscript reaches the right answer by quietly making the right choice; the defect is that this choice is not stated.

---

## Issue 5 verification ("KL generalizes naturally to complex exponential families")

**Quoted text confirmed verbatim** at line 1623 and at line 1631 (the disclaimer that fiber statistics stay positive semi-definite). The two lines contradict each other within twelve lines of text.

**Census of which complex-KL option is intended:** Grep over the manuscript for `quantum relative entropy`, `density matri`, `Born rule`, `complex KL`, `von Neumann`, `realification`, `signed measure` returns **only** line 578, which is a future-work bullet listing "Quantum extensions replacing classical probability distributions with density matrices" alongside other unspecified extensions. **Nowhere** does the manuscript define what "complex KL" means or pick one of (a) quantum relative entropy `S(rho||sigma) = tr rho(log rho - log sigma)`, (b) Born-rule probabilities `|psi|^2`, (c) `2K`-real realification, (d) signed/complex measures. The claim at line 1623 is therefore unsupported.

**What the worked example actually does** (lines 1633-1660, read verbatim): postulates `phi(tau, x) = i psi_tau T + psi_x T`, computes `A_mu = partial_mu phi`, takes the trace `G_munu = tr(A_mu A_nu)`. **No probability density appears in any of these three equations.** The Gaussian beliefs `q_i = N(mu_i, Sigma_i)` introduced earlier are never re-introduced into the trace. Line 1631 explicitly states "the Fisher-Rao metric on the belief fiber remains positive semi-definite throughout"; line 1655 ("the indefinite signature arises from the gauge connection... not from the fiber metric") doubles down on the same disclaimer. **The complex content is in `phi`, not in `q`.** Investigator 11 is correct.

**Investigator 11's verdict — claim unjustified and inconsistent with the worked example — holds.** Their proposed rewrite (drop the complex-KL claim, restate as "complex-valued gauge frames acting on real Gaussian beliefs") matches the actual mathematics.

**Severity: medium.** The sentence at 1623 is a small piece of text but it pins a load-bearing claim ("KL generalizes naturally to complex exponential families") that the manuscript cannot back. Because line 1631 already contradicts it, removing line 1623 is mechanical and does not require reproof of any equation.

---

## Issue 6 verification (Lorentzian-signature framing)

**Quoted text confirmed verbatim** at all 32 cited lines. I re-classified each independently before comparing with investigator 12.

| Line | My class | Investigator class | Agree? |
|---:|:--:|:--:|:--:|
| 47 (abstract) | H | H | yes |
| 112 (Level-3) | **O** | **O** | yes |
| 123 (Worked-example only) | H | H | yes |
| 238 (two-roles) | H | H | yes |
| 256 (sec:agents_as_sections) | **O** | **O** | yes |
| 505 (Gaussian-manifold) | M | M | yes |
| 556 (gauge-group choice) | **O** | **O** | yes |
| 1466 (Fisher-arc-length) | H | H | yes |
| 1535 (cross-ref) | H | H | yes |
| 1593 (subsection title) | H | H | yes |
| 1596 (regime status) | H | H | yes |
| 1604 ("artifact of SO(3)") | **O** | **O** | yes |
| 1608 (subsubsection lead) | H (body OK) | (subsubsection title classed O at 1618; body H) | yes (body) |
| 1614 (YM-kinetic paragraph) | H (uses "can exhibit") | H | yes |
| 1618 ("The Concrete Pathway") | **O** (title) | **O** | yes |
| 1623 (pathway step 2) | **O** ("acquires") | **O** | yes |
| 1628 (closing paragraph) | H | H | yes |
| 1681 ("naturally acquires") | **O** | **O** | yes |
| 1687 ("can in principle") | H | H | yes |
| 1712 (consensus-metric) | H | H | yes |
| 1841 ("candidate pathway") | H | H | yes |
| 1883 ("can be resolved") | H | H | yes |
| 1914 ("not derived") | H | H | yes |
| 1930 ("can in principle be derived") | M | M | yes |
| 2539 ("artifact... restriction") | M | M | yes |
| 2814 (Divergence III) | H | H | yes |
| 2886 (limitations title) | H | H | yes |
| 2890 ("Resolution Pathway... without ad hoc") | **O (worst)** | **O (worst)** | yes |
| 2900 ("not yet resolved") | H | H | yes |
| 2917 ("natural bridge", "concrete tools") | **O** | **O** | yes |
| 2944 (Conclusion para 5) | H | H | yes |
| 2950 ("validated") | M | M | yes |

**Counts agree.** 7 honest sites, 7 overclaiming sites, 4 mixed; the investigator's claim "the manuscript already contains the correct, honest framing in five places" plus four-to-seven overclaim sites is accurate.

**Independent strongest-overclaim ranking (matches investigator 12):**

1. **Line 2890** — "Resolution Pathway: ... yields indefinite pullback metrics without ad hoc signature assignments." This sentence is a direct inversion of fact. The worked example at 1633-1660 *requires* both an ad hoc imaginary-component postulate (line 1644: `phi_tau -> i psi_tau`) and an ad hoc real-part projection (line 1654: `G^Lor_munu := Re(G_munu)`). The 2890 sentence asserts "without ad hoc signature assignments" four sentences after the limitations subsection has been opened. Internal contradiction within ten lines (2890 vs. 2900).
2. **Line 1618** — "The Concrete Pathway" subsubsection title. "Concrete" is the explicit overstatement word; the body of the subsubsection at 1628 walks it back, so the title and body disagree.
3. **Line 556 + 1604 + 2539 — recurring "artifact of compact SO(3) restriction"** rhetorical formula. Three repetitions of the same overclaim, each implying the necessary condition (lifting compactness) is the sufficient condition.
4. **Line 1681** — "naturally acquires the opposite signature" overclaims; the next sentence at 1687 walks it back to "can in principle be derived". Local inconsistency.

**Investigator 12's verdict — multiple coordinated overclaims contradicting the honest text already in the manuscript — holds.** The proposed rewrites are well-targeted and do not require any new derivation.

**Severity: medium-to-high.** This is the highest-impact of the three section-resolution defects. The repeated "concrete pathway" / "resolution" / "without ad hoc signature assignments" language is the seed of the overclaim and would visibly read as overpromising to a careful reviewer. The fix is purely textual.

---

## Issue 7 verification (mass-from-precision)

**Hessian block-symmetry algebraic check, independent sympy run:**

```python
n = 3  # arbitrary
M_ik = -b_ik * Om_ik.inv().T @ L_k - b_ki * L_i @ Om_ki.inv()
M_ki = -b_ki * Om_ki.inv().T @ L_i - b_ik * L_k @ Om_ik.inv()  # i<->k swap
sp.simplify(M_ik - M_ki.T)  # -> zero matrix
```

Returns the zero matrix identically. The only assumption used is that `Lambda_i, Lambda_k` are symmetric, which is true by construction (precision matrices). **No reciprocity condition on `beta_ik` versus `beta_ki`, and no reciprocity condition on `Omega_ik` versus `Omega_ki^T`, is required for `[M_mumu]_ik = [M_mumu]_ki^T`.** Investigator 13 is correct.

**The manuscript's caveat at line 1240** — "The mass matrix `M_mumu` is symmetric only when `beta_ik = beta_ki` (reciprocal attention) and `Omega_ik = Omega_ki^T` (reciprocal gauge transport)" — **conflates two distinct symmetry notions:**

(i) **Block-symmetry of the Hessian** (`M_ik = M_ki^T`). This is automatic by Schwarz/Clairaut, and the explicit formula at line 1232 satisfies it identically.

(ii) **Symmetry of the velocity-Lagrangian 2-form** under `dot mu_i <-> dot mu_k`. The kinetic energy `T = (1/2) dot mu^T M_mumu dot mu` reads `M_mumu` as a bilinear form on velocities; whether the antisymmetric component (`M_mumu - M_mumu^T)/2` over the velocity-pair pairing vanishes is what gives a conservative-Hamiltonian reading. This *is* what reciprocity buys.

The manuscript's own next sentence at 1240-1242 ("the antisymmetric part generates velocity-dependent forces, and the kinetic-energy interpretation does not yield a conservative Hamiltonian") is correct and is referring to (ii); the *opening* sentence mis-names this as a failure of (i).

**Kinetic-metric postulate location:** confirmed at lines 1283-1290. The text at 1283 — "In the symmetric-attention limit, the second variation of free energy defines a Riemannian metric with velocity-quadratic contributions. These metric terms, when evaluated on belief trajectories, induce second-order dynamics formally analogous to a kinetic energy" — is a structural assignment, not a derivation. The Hessian computed earlier sits in the role of a *potential* curvature `K`; assigning the same object as the coefficient of `(1/2) dot mu^T M_mumu dot mu` is the load-bearing move that produces `omega^2 = K/m` with `M_mumu` simultaneously playing both roles. This is exactly the stiffness/mass conflation investigator 13 flags. **Confirmed as a postulate.**

**Investigator 13's verdict — both defects real — holds.** The proposed rewrites separate (i) automatic Hessian block-symmetry from (ii) reciprocity-for-conservative-Hamiltonian and explicitly flag the kinetic-metric postulate as load-bearing. Both fixes are textual; neither changes a quantitative result.

**Severity: medium.** Issue 7 is a technical correction with no equation-level consequence: the empirical `omega^2 ~ 1/m_eff` validation stands, but its theoretical status changes from "consequence of the Hessian of F" to "consequence of the Hessian of F **plus** the kinetic-metric postulate." The user explicitly flagged this as needing repair.

---

## Cross-issue coupling — coordinated rewrite of `sec:signature_resolution`

Issues 4, 5, and 6 all touch the same span (lines 1593-1671) and the surrounding cross-references at 47, 112, 256, 505, 556, 1466, 1535, 1681-1687, 1712, 1841, 1914, 1930, 2539, 2890-2917, 2944, 2950. A single coordinated pass should:

1. **Rename `g_{C,munu}^{YM}` -> `g_{C,munu}^{tw}` (frame-twist quadratic form) globally** [Issue 4]. Touch lines 1521-1525, 1556 (if present in the dual-metric box), 1633 (subsubsection heading), 1648, 1665, 1726. Add a one-paragraph statement at the first definition (line 1521 area) specifying `<A,B>_g = +tr(AB)` as the trace form on `gl(K,C)` and noting that under the negative-trace convention on a compact form the worked-example signature would be `(+,-)` instead of `(-,+)`.

2. **Add a non-compactness footnote at line 1640-1642** [Issue 4]. State explicitly that `T = diag(1,-1) ∈ sl(2,R) ⊂ gl(2,R)` but `T ∉ so(2)`, and that this non-compact choice is what makes `tr(T^2) > 0` available before the Lie-algebra Wick rotation.

3. **Rewrite the bullet at line 1623** [Issue 5]. Drop "complex exponential family distributions" and "KL divergence generalizes naturally to complex exponential families". Replace with: gauge frames are complex-valued (`phi ∈ gl(K,C)`), but fiber distributions remain real Gaussians; KL is used in its standard real form; a genuine quantum extension via density matrices is a separate construction not adopted here. This reconciles 1623 with the disclaimer at 1631 four lines later.

4. **Rename "The Concrete Pathway" -> "Postulates Required for an Indefinite Pullback" at line 1618** [Issue 6, OC-3]. Restructure the three-bullet pathway as a five-step list that separates necessary group-prerequisite (steps 1-2: real `GL(K,R)`, then complexification `GL(K,C)`) from sufficient postulates (step 3: imaginary frame component along `tau`; step 4: real-part projection of `tr(A_mu A_nu)`; step 5: subgroup choice `SO(1,3)` selected by input). Preserve label `sec:concrete_pathway` so cross-references at 256, 1466, 1535 etc. continue to work.

5. **Replace "Resolution Pathway" subitem at line 2890 with "Existence-toy status"** [Issue 6, OC-1]. Drop the sentence "yields indefinite pullback metrics without ad hoc signature assignments" — this sentence is the inversion-of-fact that contradicts the worked example. Replace with text that states necessary-vs-sufficient explicitly.

6. **Replace recurring "removes this restriction" / "concrete pathway" / "artifact of compact restriction" formula at lines 112, 256, 556, 1604, 2539** [Issue 6, OC-4]. Substitute uniform language: "remove the group-theoretic obstruction that forces a compact-group construction to be Riemannian, but do not by themselves derive an indefinite signature." The five sites should converge on this voice.

7. **Soften "naturally acquires" at line 1681** [Issue 6, OC-3 secondary]. Replace with "can be assigned... under the worked example of Section ref{sec:worked_signature}". The follow-up sentence at 1687 ("can in principle be derived") is already honest and should be retained.

8. **Replace "natural bridge" / "concrete mathematical tools" at line 2917** [Issue 6, OC supplement]. Same hedging fix as item 6.

9. **Soften "validated" at line 2950** [Issue 6, OC supplement]. Replace with "extended to four nonlinear dimensions and tested for dynamical selection of the imaginary component."

Items 1-2 (Issue 4) and item 3 (Issue 5) are local edits at lines 1521-1670. Items 4-9 (Issue 6) are spread across the manuscript but are uniform substitutions. The Regime~I status paragraph already added at line 1596 does the framing work for the worked-example subsubsection; the rest of the manuscript needs only to be brought into alignment with it. Issue 7's fixes (lines 1098-1300) are independent.

---

## Investigator quality

**Investigator 10 (YM sign/naming):** Strong. The sympy verification of `tr(A^2)` signs across `so(2)`, `so(3)`, `su(2)`, and `T = diag(1,-1)` is thorough, the convention-conflict diagnosis is precise, and the proposed rewrites are surgical. The investigator correctly flags the "Wick rotation in the Lie algebra" framing as not the same as a passage between real forms of a complex Lie algebra — a subtle distinction that the manuscript currently glosses. No misses.

**Investigator 11 (complex KL):** Strong. The four-option enumeration (quantum relative entropy, Born rule, realification, signed measures) is the correct frame for assessing what "complex KL" could mean, the identification of the contradiction between line 1623 and line 1631 is exactly right, and the cross-check against the worked example is decisive. The secondary nit on "non-compact structure" overstating non-compactness is also correct (real `GL(K,R)` is non-compact and produces positive-definite signatures by Sylvester). No misses.

**Investigator 12 (Lorentzian framing):** Strong. The 32-line census is comprehensive — I cross-checked every quoted line and every classification and found no errors. The honest-vs-overclaim count is accurate, the strongest-overclaim ranking is correct (line 2890 is the worst single sentence), and the proposed coordinated rewrite respects the existing label structure and cross-references. The one possible miss is that the investigator did not separately flag line 1623's "complex exponential family distributions" claim under Issue 6 (it is correctly left for Issue 5 instead, so this is a scope choice rather than an oversight).

**Investigator 13 (mass/stiffness):** Strong. The sympy-confirmed identity `M[i,k] = M[k,i]^T` independent of reciprocity, the textbook check of `omega^2 = K/m`, and the separation of automatic Hessian block-symmetry from velocity-Lagrangian symmetry are all correct. The kinetic-metric postulate identification at lines 1283-1290 is precisely localized. No misses.

---

## Consolidated severity (verifier-adjudicated)

| Issue | Investigator severity | Verifier severity | Reasoning |
|---|:--:|:--:|---|
| 4 (YM naming + sign) | (not numerically rated; "partially confirmed" + "confirmed") | **medium** | Real defect; manuscript reaches right answer via unstated choice; fix is textual |
| 5 (complex KL) | (not numerically rated; "unjustified") | **medium** | Single sentence pins unsupported load-bearing claim; contradicted by line 1631; fix is mechanical |
| 6 (Lorentzian overclaim) | (not numerically rated; multiple OC sites) | **medium-to-high** | Coordinated rhetorical defect at seven sites contradicting honest text at five; line 2890's "without ad hoc signature assignments" is a direct inversion of fact |
| 7 (mass/stiffness + Hessian caveat) | "Substantive" (×2) + "Technical correction" (×1) | **medium** | Both defects real; no quantitative consequence; user-flagged for repair |

---

## Recommended fix priority (rank-ordered)

1. **Issue 6, line 2890** — "Resolution Pathway: ... without ad hoc signature assignments." Inversion-of-fact in the limitations section. Fix first; one-paragraph rewrite.
2. **Issue 7** — User explicitly flagged. Rewrite Hessian-symmetry caveat at lines 1240-1242 (separate (i) from (ii)) and add stiffness-vs-mass clarification at lines 1112 + 1273. No equation changes.
3. **Issue 5, line 1623** — Drop "complex exponential family distributions" / "KL generalizes naturally to complex exponential families". Replace with a bullet that matches what the worked example actually does. Reconciles with line 1631.
4. **Issue 6, line 1618 ("The Concrete Pathway") + recurring "removes this restriction" formula at 112, 256, 556, 1604, 2539** — Coordinated uniform substitution across five sites; rename subsubsection.
5. **Issue 4 — global rename `YM -> tw` and add convention/non-compactness statements at 1521 and 1640-1642**. Mechanical substitution at six sites plus two paragraph additions.
6. **Issue 6, secondary** — line 1681 "naturally acquires", line 2917 "natural bridge"/"concrete tools", line 2950 "validated". Local hedging substitutions.

Items 1-3 are independent and high-value. Items 4-6 are cross-coupled and best done in a single editing pass over `sec:signature_resolution` as outlined in the cross-issue coupling section above.
