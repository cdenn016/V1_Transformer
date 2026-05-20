# Red Opening — supplementary-variational-gradient-descent

## Steelman (opposing position)

Sub-claims α through ε of the chapter are mathematically standard and primary-source verifiable: Eq. (623–626) is the textbook Fisher-Rao natural gradient on the Gaussian SPD manifold [Amari 2016 §4.3]; Eq. (637–639) is the [Pennec 2006] / [Bhatia 2007 §6.1] / [Absil-Mahony-Sepulchre 2008 §5.4.6] affine-invariant exponential map; Eq. (661) is the SO(3) closed form of the left-trivialised dexp series [Hall 2015 §2.7, Theorem 5.4]; the trivialization identity at line 659 reduces to `Ad_{exp(-X)} = e^{-ad_X}` and the operator identity `Ψ_R(z) = e^z Ψ_L(z)`. The chapter could therefore be accepted as a self-contained supplementary chapter on the strength of those five sub-claims, with the residual cross-reference at line 665 read as an informal pointer to §C.5.

## Position

The compound claim is FALSE because sub-claim ζ is broken in two distinct, primary-source-verifiable ways at line 665 of `Attention/GL(K)_supplementary.tex`, and the breakage is substantive (not narrow-editorial). Specifically: (a) the §D.3 statement of the pullback metric at line 665 is the OLD uncorrected formula `G_{ab}(φ) = ⟨Ψ(ad_X)(T_a), Ψ(ad_X)(T_b)⟩` which omits the position-dependent `exp(φ)exp(φ)^⊤` factor that was added to §C.5 line 606–608 during the §C debate; and (b) the cross-reference "App.~C.3" at line 665 lands in `\subsection{KL Gradient Through Transport}` rather than the `\subsection{Gauge Frame Preconditioning for GL(K)}` (label `sec:glk_preconditioning`) where the Killing form / Cartan-involution-modified metric is actually derived. Because the §D.3 formula is now internally inconsistent with the corrected §C.5 formula and the pointer to the derivation is dead, sub-claim ζ fails on its own terms, and the chapter is not pure as a self-contained unit.

## Evidence

- `Attention/GL(K)_supplementary.tex:665` reads `(iii) the full \emph{pullback} natural gradient $G_{ab}(\phi) = \langle \Psi(\mathrm{ad}_X)(T_a), \Psi(\mathrm{ad}_X)(T_b) \rangle$, where $\Psi(z) = (e^z - 1)/z$`. No `exp(φ)exp(φ)^⊤` factor.

- `Attention/GL(K)_supplementary.tex:606-608` reads `\mathcal{G}_{ab}(\phi) = \langle D_\phi(\exp)[T_a], D_\phi(\exp)[T_b]\rangle_F = \operatorname{tr}\bigl(\Psi(\mathrm{ad}_\phi)(T_a)^\top \cdot \exp(\phi)\exp(\phi)^\top \cdot \Psi(\mathrm{ad}_\phi)(T_b)\bigr)`. This is the corrected form. Line 612 of the same file explicitly notes "For non-compact (symmetric) $\phi$, $\exp(\phi)\exp(\phi)^\top = \exp(2\phi)$ grows exponentially, automatically compensating the exponential amplification of $D\exp$ in the non-compact sector." The factor is therefore not notational shorthand — it materially changes the metric on the non-compact (symmetric) directions of `gl(K)`, which is the entire motivation for the position-dependent preconditioner.

- §C subsection structure (verified via Grep on `\\subsection\{`):
  - §C.1 line 397 `Structure of the Gauge Frame Gradient`
  - §C.2 line 434 `Differential of the Matrix Exponential`
  - §C.3 line 488 `KL Gradient Through Transport`
  - §C.4 line 543 `Retraction and Numerical Considerations`
  - §C.5 line 558 `Gauge Frame Preconditioning for GL(K)` (label `sec:glk_preconditioning` at line 559)

- `Attention/GL(K)_supplementary.tex:584-594` (in §C.5) is the actual derivation of the Cartan-involution-modified bilinear form `g̃_{ab} = 2K tr(T_a^⊤ T_b) - 2 tr(T_a) tr(T_b)` and its positive-definiteness on `sl(K)`. The cross-reference at line 665 says "as derived in App.~C.3" — but §C.3 is `KL Gradient Through Transport`, which contains no such derivation. The pointer is dead.

- The canonical pullback metric of a Riemannian metric `h` on the target through a smooth map `f: M → N` is `f^*h(X, Y) = h(df(X), df(Y))` [Lee 2013 §13]. Here `f = exp`, `df = D_φ exp = Ψ(ad_φ) · exp(φ)` (right-trivialised, [Hall 2015 §2.7, Theorem 5.4], cited at supplementary line 597), and `h` is the Frobenius inner product on `Mat(K)`: `h(A, B) = tr(A^⊤ B)`. Plugging in: `f^*h(T_a, T_b) = tr((Ψ(ad_φ)(T_a) · exp(φ))^⊤ · (Ψ(ad_φ)(T_b) · exp(φ))) = tr(exp(φ)^⊤ · Ψ(ad_φ)(T_a)^⊤ · Ψ(ad_φ)(T_b) · exp(φ)) = tr(Ψ(ad_φ)(T_a)^⊤ · exp(φ) exp(φ)^⊤ · Ψ(ad_φ)(T_b))` (cyclic). This is exactly the §C.5 form at line 608. The §D.3 form at line 665 omits the `exp(φ)exp(φ)^⊤` factor and therefore does not correspond to the pullback of the Frobenius metric through `exp` — it is the bilinear form one would get from `tr(Ψ(ad_φ)(T_a)^⊤ Ψ(ad_φ)(T_b))`, which is the pullback only at the identity `φ = 0`.

- Sub-claim α verified via sympy: with `g(A,B) = (1/2) tr(Σ^{-1} A Σ^{-1} B)` and natural-gradient candidate `2 Σ sym(G) Σ`, the bilinear form `(1/2) tr(Σ^{-1} · 2 Σ sym(G) Σ · Σ^{-1} · v)` reduces to `Trace((G + G^⊤)/2 · v) = tr(sym(G) v)`, matching `⟨∇_Σ, v⟩` for symmetric `v` (sympy output `LHS = Trace((G + G.T)*v)/2`, `RHS = Trace(((1/2)*G + (1/2)*G.T)*v)`, identical). Eq. (623–626) is canonical [Amari 2016 §4.3]. Sub-claims α–ε are conceded.

- Issue 3 (additional editorial gap, not load-bearing alone): Eq. (637–639) is the affine-invariant SPD exponential map and is presented without citation. The canonical references are [Pennec 2006 "Intrinsic statistics on Riemannian manifolds"], [Bhatia 2007 "Positive Definite Matrices" §6.1], [Absil-Mahony-Sepulchre 2008 "Optimization Algorithms on Matrix Manifolds" §5.4.6]. Same standard the §3.7 Gamma-prior debate and §B Killing-form debate were held to.

- Issue 4 (additional editorial gap): line 663 says `K > 3` falls back to "the Fr\'{e}chet derivative of the matrix exponential using symmetric quadrature" with no citation. The canonical algorithm is [Higham 2008 §10.2, Algorithm 10.27 "Fr\'{e}chet derivative via Pad\'{e} approximation with scaling and squaring"] or the augmented-matrix identity due to [Mathias 1996]. `Higham2008` is already in `references.bib`; cited at supplementary line 556 for polar decomposition.

- Issue 6 (additional editorial gap): the clipping bound `|τ λ_B^{(j)}| ≤ 50` at line 640 is hard-coded with no justification. `exp(50) ≈ 5.18 × 10^21` is well below the float64 overflow at `exp(709)`, so the bound is conservative; but the prose offers no anchor for why 50 versus 30 or 100. Either justify (e.g., "to preserve the SPD condition cap `κ_max = 10^4` given the spectral floor `ε_SPD = 10^{-4}`") or cite a numerical convention.

## Falsification conditions

Red's position is wrong if any of the following are demonstrated:

1. **The §D.3 line 665 formula `G_{ab}(φ) = ⟨Ψ(ad_X)(T_a), Ψ(ad_X)(T_b)⟩` is equivalent to the §C.5 line 608 formula on the non-compact directions of `gl(K)`.** This would require `exp(φ)exp(φ)^⊤ = I` on the symmetric sector, which fails by direct computation: for `φ = diag(s, -s, 0, ..., 0)` ∈ Sym(K), `exp(φ)exp(φ)^⊤ = diag(e^{2s}, e^{-2s}, 1, ..., 1) ≠ I` for `s ≠ 0`. §C.5 line 612 itself states `exp(φ)exp(φ)^⊤ = exp(2φ)` on Sym. The two formulas are non-equivalent on the non-compact sector.

2. **The cross-reference "App.~C.3" at line 665 is a typesetter convention that LaTeX resolves to §C.5.** Verifiable false: §C.5 has the label `\label{sec:glk_preconditioning}` at line 559 and is the fifth subsection of `\section{Gauge Frame Gradients}`; "App.~C.3" is a hand-typed text reference to the third subsection of Appendix C, which is `\subsection{KL Gradient Through Transport}` at line 488. There is no LaTeX cross-reference command (`\ref{}`, `\autoref{}`) — it is plain text. The pointer is dead by inspection.

3. **§D.3 line 665 is intended only as an informal three-line summary of §C.5 modes (i), (ii), (iii), not as a load-bearing equation.** This defense is contradicted by the prose at line 665: the formula is stated as a definition (`uses the bilinear form $\tilde{g}_{ab} = ...$`, `the full pullback natural gradient $G_{ab}(\phi) = ...$`) and the cross-reference is to a derivation, not a definition. If §D.3 is just a summary, the broken pointer and stale formula still fail it — a summary that points to a non-existent derivation and states an outdated formula is not a "complete and pure self-contained supplementary chapter," which is what the compound claim asserts.

4. **The §C debate verdict already accounts for this propagation.** Verifiable from the §C debate artifacts: the §C action plan corrected §C.5 line 606–608 (commit `e4481f7c`) but did not touch line 665. The propagation is unfixed at the time of this debate.

If none of (1)–(4) is demonstrated, the compound claim fails on sub-claim ζ, and the verdict should be RED_WINS on substantive grounds (formula error, not narrow-editorial citation gap) — comparable in magnitude to the §C verdict, not the narrow §A/§B verdicts.
