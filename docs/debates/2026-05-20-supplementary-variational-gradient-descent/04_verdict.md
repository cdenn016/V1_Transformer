# Verdict — supplementary-variational-gradient-descent

## Outcome

RED_WINS (narrow editorial scope, comparable to §A / §B precedent)

## Decisive evidence

The decisive citation pair is the verbatim textual diff between two lines of `Attention/GL(K)_supplementary.tex` and the verified `\subsection{` structure of §C:

1. Line 665 (post-`e4481f7c` state, unchanged): `(iii) the full \emph{pullback} natural gradient $G_{ab}(\phi) = \langle \Psi(\mathrm{ad}_X)(T_a), \Psi(\mathrm{ad}_X)(T_b) \rangle$, where $\Psi(z) = (e^z - 1)/z$ ... (positive-definite on $\mathfrak{sl}(K)$ as derived in App.~C.3)`.

2. Line 606 (post-`e4481f7c` corrected state): `\mathcal{G}_{ab}(\phi) = \operatorname{tr}\bigl(\Psi(\mathrm{ad}_\phi)(T_a)^\top \cdot \exp(\phi)\exp(\phi)^\top \cdot \Psi(\mathrm{ad}_\phi)(T_b)\bigr)`.

3. Grep on `\subsection{` in the same file: §C.3 (line 488) is `KL Gradient Through Transport`; §C.5 (line 558, label `sec:glk_preconditioning`) is `Gauge Frame Preconditioning for GL(K)`. The Cartan-involution-modified derivation lives at §C.5 lines 584–594, not at §C.3.

4. Pullback identity derivation from [Lee 2013 §13] and [Hall 2015 §2.7, Theorem 5.4]: with right-trivialised differential `D_\phi \exp = \Psi(\mathrm{ad}_\phi) \cdot \exp(\phi)`, the Frobenius pullback yields `tr(\Psi(\mathrm{ad}_\phi)(T_a)^\top \cdot \exp(\phi)\exp(\phi)^\top \cdot \Psi(\mathrm{ad}_\phi)(T_b))`. Stripping the `\exp(\phi)\exp(\phi)^\top` factor reduces this to the metric at the identity `\phi = 0`, which on the symmetric (non-compact) sector of `gl(K)` is non-equivalent to the corrected form, since `\exp(\phi)\exp(\phi)^\top = \exp(2\phi) \neq I` for `\phi \in \mathrm{Sym}(K) \setminus \{0\}` (line 612 of the same file explicitly states this).

Sub-claims α through ε are conceded by both sides under [Amari 2016 §2.3, §4.3], [Pennec 2006 §3], [Bhatia 2007 §6.1, Theorem 6.1.6], [Absil–Mahony–Sepulchre 2008 §5.4.6], [Hall 2015 §2.7, Proposition 2.25] — verified by sympy substitution for α. Sub-claim ζ as written in `00_claim.md` ("the three preconditioner modes ... point to the §C.5 preconditioning section") is falsified on its own terms by (1)–(3): the printed cross-reference does not in fact point to §C.5, and the printed inline formula for mode (iii) is not the §C.5 form.

## Reasoning

Both sides converged on the same factual record and on the same verdict label. Red conceded sub-claims α through ε after blue's citation-by-citation verification against [Amari 2016], [Pennec 2006], [Bhatia 2007], [Absil–Mahony–Sepulchre 2008], and [Hall 2015]; blue conceded Issue 1 (stale pullback formula at line 665), Issue 2 (broken `App.~C.3` cross-reference), and Issue 3 (missing canonical citations on Eq. 637–639) after red's pullback-of-Frobenius derivation via [Lee 2013 §13] and the Grep-verified `\subsection{` structure. Red's rebuttal closes with "I do not push for substantive scope... Narrow scope, three concrete edits, matches §A and §B." Blue's rebuttal closes with "The defensible verdict is RED_WINS-narrow." There is no remaining dispute to adjudicate; the compound claim is falsified at sub-claim ζ on textual evidence both sides accept.

The scope distinction between this verdict and the §C substantive verdict is anchored by blue's codebase evidence at `transformer/vfe/config.py:597-609`: the inner E-step path raises `ValueError` when `phi_preconditioner='pullback'`, gating the stale form out of the active call chain; the only reachable pullback consumer is `RiemannianAdamW(metric='pullback')`, which calls `build_pullback_metric_tensor` at `transformer/core/gauge_preconditioner.py:596`, which implements the corrected §C.5 form with the `\exp(\phi)\exp(\phi)^\top` factor (verified in commit `e4481f7c`). The §D defect therefore propagates only at the manuscript level: it is a stale duplicate of an equation whose primary copy was corrected in the prior debate, plus a broken plain-text section label. The §C debate's substantive classification rested on three FD-verified formula errors in active code paths; §D inherits one of those errors as text-only forward drift. That pattern is the §A and §B narrow-verdict profile, not the §C substantive profile.

The compound claim asserts the chapter is "complete and mathematically/theoretically pure as a self-contained supplementary chapter." The "mathematically pure" prong survives — sub-claims α through ε hold against the canon. The "complete and self-contained" prong fails at line 665 on two textual defects within a single parenthetical clause, plus the citation gap at Eq. (637–639). Three concrete edits restore the assertion.

## Action

Apply three editorial fixes to `Attention/GL(K)_supplementary.tex` and re-verify the chapter against the canon.

1. **Line 665 inline formula for mode (iii).** Either replace the displayed equation `G_{ab}(\phi) = \langle \Psi(\mathrm{ad}_X)(T_a), \Psi(\mathrm{ad}_X)(T_b) \rangle` with the §C.5 corrected form `G_{ab}(\phi) = \operatorname{tr}\bigl(\Psi(\mathrm{ad}_\phi)(T_a)^\top \cdot \exp(\phi)\exp(\phi)^\top \cdot \Psi(\mathrm{ad}_\phi)(T_b)\bigr)`, or remove the inline formula and write "(iii) the full pullback natural gradient (see App.~C.5, Eq. (...) for the position-dependent form)." Either edit closes the internal inconsistency with §C.5 line 606.

2. **Line 665 cross-reference.** Replace `App.~C.3` with `App.~C.5` (label `sec:glk_preconditioning` at line 559). Single-token edit; the §C.3 subsection is `KL Gradient Through Transport` and contains no preconditioner derivation.

3. **Eq. (637–639) SPD-retraction citations.** Add `\citep{Pennec2006,Bhatia2007,AbsilMahonySepulchre2008}` (or the project's existing bibkeys for these works; confirm `Pennec2006` and `Bhatia2007` are present in `references.bib`, add if missing). Reference targets: [Pennec 2006 §3], [Bhatia 2007 §6.1 Theorem 6.1.6], [Absil–Mahony–Sepulchre 2008 §5.4.6].

Optional secondary fixes flagged in evidence but not load-bearing on this verdict:

4. Cite [Higham 2008 §10.2, Algorithm 10.27] at line 663 for the symmetric-quadrature Fréchet-derivative path. `Higham2008` is already in `references.bib` (cited at supplementary line 556).

5. Justify the clipping bound `|\tau \lambda_B^{(j)}| \leq 50` at line 640 against the spectral floor `\varepsilon_{\mathrm{SPD}} = 10^{-4}` and condition cap `\kappa_{\max} = 10^4`, or cite a numerical convention.

6. Provide a code reference (test file path / function name) for the FD-validation claim at line 670.

After fixes 1–3 are applied, sub-claim ζ holds line-for-line: line 665 mode (iii) matches §C.5 line 606, the cross-reference resolves to the correct subsection, and the SPD retraction at Eq. (637–639) is anchored to the canonical sources used to verify sub-claim γ.
