# Red Rebuttal — pifb-spec-1-pullback

## Concession

Blue's defense of the gauge-non-invariance disclosure at 2722–2723 lands. The Maurer-Cartan transformation law `A -> g^{-1} A g + g^{-1} dg` is the textbook gauge-transformation rule for a principal connection [Nakahara2003 §10.4, Eq. 10.19] [KobayashiNomizu Vol. I §II.1], and the refusal of the `tr(FF)` escape via the pure-gauge identity (`A = U^{-1} dU` gives `F = dA + (1/2)[A,A] = 0` by Maurer-Cartan) is standard [KobayashiNomizu Vol. I §II.5, Maurer-Cartan structure equation]. Blue is right that this is the disciplined move: declare the non-invariance, refuse the spurious escape, route invariance elsewhere. I also grant the `−tr` vs `+tr` piecewise convention at 2697–2700 is mathematically correct under [Helgason 1978 Ch. III §6] (negative-definite Killing on compact real form so(N), so `−tr` becomes the positive-definite inner product, equivalent up to a positive scalar `(N−2)` per [Knapp 2002 Ch. II §1]). The Fisher-Rao form at 2673 matches [AmariNagaoka2000 Ch. 2] and the natural-gradient forms at 2617 match [Amari1998 §4].

## Core attack

Blue's evidence #6 (the BCH order claim) collapses under numerical verification, and blue's evidence #9 (Hoffman/Friston attribution at 2749–2751) overruns the cited literature. These are two independent failures of operational-reading criteria (1) and (2).

**Attack 1 — BCH order at line 2649 is wrong by one order under the obvious reading.** The manuscript writes
```
dphi_i/dt = -eta * tilde-grad F + O(|| eta * tilde-grad F ||^2)
```
with the parenthetical claim "the higher-order term is the Baker-Campbell-Hausdorff correction to the additive Lie-algebra step." Blue cites [Hall 2015 Ch. 5] for BCH. Let me check what BCH actually gives. The BCH expansion is
```
log(exp(X) exp(Y)) = X + Y + (1/2)[X,Y] + (1/12)([X,[X,Y]] − [Y,[X,Y]]) + …  [Hall 2015 Thm. 5.3]
```
The leading correction `(1/2)[X,Y]` is BILINEAR in X and Y. With X = phi_i (the chart-coordinate frame, generically O(1) when the chart is centered at the identity, which is what `U_i = exp(phi_i)` at line 2647 fixes) and Y = −eta * tilde-grad F (the step), the leading BCH correction is `−(eta/2)[phi_i, tilde-grad F]`, which is FIRST order in eta, not second. The manuscript's `O(||eta * tilde-grad F||^2)` is correct only in the special regime ||phi|| = O(eta * ||tilde-grad F||) — i.e., when the chart is also centered at the current frame so the chart-phi is O(eta) — but that re-centering is not declared and would contradict the global chart `U_i = exp(phi_i)` written at 2647.

Numerical verification in so(3) with phi_vec = (0.5, 0.3, -0.2) and tilde-grad-vec = (0.1, 0.05, -0.08), sweeping eta:

| eta    | ‖residual after −eta·G‖ | ratio vs prev | (eta·‖G‖)² (predicted O(eta²)) | eta·‖G‖·‖phi‖ (predicted O(eta)) |
|--------|---------------------------|---------------|----------------------------------|------------------------------------|
| 0.1    | 1.77e-3                   | —             | 3.78e-4                          | 1.69e-2                            |
| 0.01   | 1.77e-4                   | 9.996         | 3.78e-6                          | 1.69e-3                            |
| 0.001  | 1.77e-5                   | 10.00         | 3.78e-8                          | 1.69e-4                            |
| 0.0001 | 1.77e-6                   | 10.00         | 3.78e-10                         | 1.69e-5                            |

The residual decreases LINEARLY with eta (ratio 10×), not quadratically (ratio 100×). This is the empirical signature of an O(eta) leading correction. Adding the leading BCH term `−(eta/2)[phi,G]` reduces the residual by another order of magnitude to 1.86e-4 ≈ (eta·‖G‖)², confirming the residual structure is exactly `−eta·G − (eta/2)[phi,G] + O(eta²)`. The first-order correction `−(eta/2)[phi,G]` is non-trivial for generic phi (the chart coordinate of an actual gauge frame is not small a priori in any of `SO(3)`, `SU(N)`, `GL(K,R)`).

This is a math error under the operational-reading criterion (1) "the math is correct," not a disclaimer issue. The fix is two characters: replace `O(||eta · grad||^2)` with `O(||eta · grad|| · ||phi||)` (or equivalent), or explicitly state the chart is re-centered each step and that phi^t is reset to 0 (the right-trivialized update). Blue's citation [Hall 2015 Ch. 5] is the correct source for BCH, but blue does not perform the order check that source enables.

**Attack 2 — Hoffman/Friston citation at 2751 overruns the source.** The manuscript writes:
> "The Interface-Theory reading [Hoffman2019] sharpens this further: perceived space is an evolved interface optimized for fitness rather than a faithful depiction of an agent-independent geometry, and the carrier of that interface is precisely the slow-timescale generative model rather than the moment-to-moment posterior."

Blue defends this as "consistent with Hoffman's interface-theory thesis." This is mis-located. Hoffman 2019 *The Case Against Reality* argues fitness-beats-truth in evolutionary perception via the desktop/icon metaphor [Hoffman 2019 Ch. 4-6, "Interface Theory of Perception"]: the carrier of the interface is *evolved sensory machinery* (species genome, perceptual coding), shaped on the *evolutionary timescale* (generations), not the *learned slow generative model* of active-inference parameter learning (within-lifetime). The manuscript's "precisely the slow-timescale generative model rather than the moment-to-moment posterior" is borrowing the active-inference timescale separation [FristonEtAl2017] and back-projecting it onto Hoffman, who uses a different (evolutionary, not within-lifetime parameter) timescale separation. Hoffman never says — and the search of his work does not support — that the interface is carried by a slow-timescale Bayesian generative model as opposed to fast-timescale Bayesian belief updating. The two slow-vs-fast distinctions are different stories.

This is a citation-misrepresentation under operational-reading criterion (2). The text "the carrier of that interface is precisely the slow-timescale generative model rather than the moment-to-moment posterior" reads as a Hoffman attribution; it is the author's own bridge claim, not Hoffman's. Blue's own falsification condition #4 explicitly admits this: "If Hoffman 2019's interface-theory thesis does not in fact locate the interface at a slow-timescale evolved structure (vs. moment-to-moment perception), the 'sharpens this further' claim at 2751 fails." The condition fires: Hoffman 2019 locates the interface at the *evolutionary-timescale* genome, not the *within-lifetime slow-channel generative model*.

Together these two attacks land under criteria (1) and (2) of the operational reading. Operational-reading criterion failure on either is sufficient for a red strike per `00_claim.md`.

## Defense

Blue's strongest single point is evidence #5: the gauge non-invariance disclosure at 2722–2723 is honest, the Maurer-Cartan transformation law is correct, and the Yang-Mills escape is correctly refused. I do not contest this — the disclosure paragraph is the most disciplined passage in the section. The Maurer-Cartan structure equation `dA = −(1/2)[A,A]` gives `F ≡ 0` for any pure-gauge connection `A = U^{-1} dU` [KobayashiNomizu Vol. I §II.5], confirming blue's refusal of the `tr(FF)` escape.

What this defense does not rescue is the math error at 2649 or the Hoffman over-attribution at 2751. A section can have one honest paragraph and still fail "publication-ready and rock-solid" if another paragraph carries a wrong-order error term and another mis-attributes a load-bearing interpretive claim. The bundle metric construction is best-read; the gauge disclosure is well-disclosed; but the operational reading binds the judge to math-correctness AND citation-fidelity, not majority-rule across paragraphs.

The minimum repair: change `O(||eta * tilde-grad F||^2)` at 2649 to either `O(||eta * tilde-grad F|| * ||phi||)` (if the chart at 2647 is read as fixed at identity), or restate 2647 in right-trivialized form (the chart is re-centered at the current `U_i^t` each step, so the local chart-phi is 0) and keep the O(eta^2) order. At 2751, replace "the carrier of that interface is precisely the slow-timescale generative model" with either a Friston-only citation (where the slow-parameter/fast-state distinction is canonical [FristonEtAl2017]) or an acknowledgement that the bridge between Hoffman's evolutionary-interface and the active-inference structural tier is an interpretive proposal of the authors. Without these repairs, the section is not "publication-ready and rock-solid" under its own operational reading.

Sources:
- [The Case Against Reality - UC Irvine News](https://news.uci.edu/2019/07/22/the-case-against-reality/)
- [Interface Theory of Perception from the Case Against Reality - Medium](https://medium.com/@thecommonsapien/interface-theory-of-perception-from-the-case-against-reality-f4d09c655d92)
