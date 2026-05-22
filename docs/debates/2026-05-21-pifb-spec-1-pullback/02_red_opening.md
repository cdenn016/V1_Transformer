# Red Opening — pifb-spec-1-pullback

## Steelman (opposing position)

PIFB lines 2584-2772 are a self-aware Outlook section whose disclaimers ("toy model", "highly speculative", explicit dimensional caveat on `tau`, explicit non-identification of Fisher arc length with proper time, explicit gauge-invariance disclosure for the horizontal block, explicit `F_{munu} = 0` flag for Regime I) honestly fence off the speculative content from the empirically anchored portion of the manuscript, and the underlying math — block-Fisher Gaussian natural gradient, group-level retraction `U^{t+1} = U^t exp(-eta ~∇)`, Fisher-Rao fiber metric, horizontal/vertical bundle split — recapitulates standard information geometry and bundle geometry (`[Amari1998]`, `[Nakahara2003 §10.3]`, `[AmariNagaoka2000]`) without falsifiable additional commitments, so a reader looking for "math correctness, honest disclaimers, no overreach" finds all three.

## Position

The section is **not publication-ready** under the operational reading. Two of the four operational criteria fail in load-bearing ways, and a third is shakier than the manuscript admits.

1. **Operational rule (1) — the math is correct: FAILS at PIFB:2649.** The order estimate `O(||eta_phi ~∇_phi F||^2)` on the BCH correction term in the chart-coordinate flow `dphi/dt` is wrong by a full order in the gradient. The leading correction is `(1/2)[phi, eta ~∇]`, which is `O(||phi|| · ||eta ~∇||)` — first order in the gradient, scaled by `||phi||` (which is `O(1)`, not small). The "exact in the abelian sector" qualifier at PIFB:2651 contradicts the order estimate: a term vanishing in the abelian sector cannot be `O(||eta ~∇||^2)` in the non-abelian sector, because what makes it nonzero is `[phi, ~∇]`, which depends linearly on the gradient, not quadratically.

2. **Operational rule (2) — citations correctly invoked: FAILS at PIFB:2751 (Hoffman2019).** The manuscript attributes to Hoffman's Interface Theory of Perception the specific architectural claim that "the carrier of that interface is precisely the slow-timescale generative model rather than the moment-to-moment posterior." This is the user's own three-tier identification dressed in a Hoffman citation; Hoffman 2019 is an evolutionary fitness-beats-truth argument that does not adjudicate between state-level inference and parameter-level learning. This is wrong-domain citation under the canon-cop rubric (`external_canon_inference.md` §1, "FEP implies X" pitfall analogue).

3. **Operational rule (3) — disclaimers complete: SHAKY at PIFB:2680, 2701.** The defense against "this is L^2 pullback in disguise" at PIFB:2680 does not survive the gauge-non-invariance admission at PIFB:2722. A bundle metric whose horizontal block is built from a gauge-frame-dependent connection one-form `A^{(i)} = U_i^{-1} dU_i` is a metric on the *trivialized* bundle in the chosen gauge, not on the abstract associated bundle `E_q = P x_G B_state`. The two-line concession at PIFB:2722 ("within an agent's gauge fixing") defers the problem to `sec:consensus_metric` rather than defusing it; the section as written reads as a derivation when it is, structurally, still a section-dependent quadratic form.

## Evidence

- **BCH leading-correction order (canon).** The BCH expansion of `dexp_phi^{-1}(Y) = Y - (1/2) ad_phi(Y) + (1/12) ad_phi^2(Y) - ...` (`external_canon_math.md` §2 "Lie algebra exponential"; standard BCH formula, Frankel Ch. 14, Hall *Lie Groups* Ch. 5). Under right-trivialization `U^{-1} dU/dt = -eta ~∇`, the chart-coordinate velocity is `dphi/dt = dexp_phi^{-1}(-eta ~∇) = -eta ~∇ + (1/2)[phi, eta ~∇] - (1/12)[phi,[phi, eta ~∇]] + ...`. The leading correction `(1/2)[phi, eta ~∇]` is linear in `eta ~∇` and linear in `phi`, not quadratic in either.

- **Executed sympy verification (so(3), unit-norm `phi` and `g`).**

  ```
  phi = [[0,-1,0],[1,0,0],[0,0,0]],   g = [[0,0,0],[0,0,-1],[0,1,0]]
  ||phi||_F = sqrt(2),  ||g||_F = sqrt(2)
  corr = (1/2)[phi, g] = [[0, 0, 1/2], [0, 0, 0], [-1/2, 0, 0]]
  ||corr||_F = sqrt(2)/2
  ```

  The Frobenius norm of the correction scales linearly with `||g||` and linearly with `||phi||`, not quadratically with `||g||`. Substituting `g -> eta ~∇` confirms the correction is `Theta(eta · ||~∇||)`, contradicting `O(eta^2 ||~∇||^2)`.

- **Manuscript self-contradiction.** PIFB:2649 states `O(||eta ~∇||^2)`. PIFB:2651 states "exact in the abelian sector". For the correction to vanish in the abelian sector, it must contain a commutator `[phi, ·]`. Any commutator-based correction is linear in its arguments, not quadratic, so the two sentences cannot both be right.

- **Hoffman2019 scope (canon).** `external_canon_inference.md` §6 pitfall list, item 3 ("FEP implies X. FEP is a variational principle; specific implementations follow from specific generative-model choices. Claims that FEP alone implies an architectural choice ... require the explicit generative model that connects them"). Hoffman's argument is the analogous case for ITP: an evolutionary-decision-theoretic argument that perception tracks fitness, not a commitment to a slow-parameter vs fast-state carrier. The clause "the carrier of that interface is precisely the slow-timescale generative model rather than the moment-to-moment posterior" at PIFB:2751 attributes the user's three-tier architecture to Hoffman, who does not articulate it.

- **Bundle-metric construction (canon).** `[Nakahara2003 §10.3]` and `external_canon_math.md` §2 "Bundles" describe the standard Riemannian-submersion construction: a metric on the base `g_C` plus a metric on the typical fiber together induce a metric on `E = P x_G F` via horizontal lift of `g_C` through the connection. The user's construction at PIFB:2693 reverses this: it *defines* `g_C^tw` from the connection itself, using a section-determined `A^{(i)} = U_i^{-1} dU_i`. The result is gauge-non-invariant (admitted at PIFB:2722) — which is the diagnostic that the construction is not a metric on the abstract bundle `E_q`, only on the trivialization. The pre-emptive defense at PIFB:2680 ("does earn the name 'pullback of a fiber-bundle metric'") does not survive this admission.

- **PageWootters1983 (control).** Verified against Wikipedia summary and `[PageWootters1983]` primary: clock subsystem entangled with rest of globally stationary state. The manuscript's gloss at PIFB:2592 matches. This citation is correctly invoked, and is offered as evidence that the citation discipline elsewhere in the section is uneven rather than universally bad.

## Falsification conditions

This Red position is wrong if any of the following holds:

1. The BCH correction term `(1/2)[phi, eta ~∇]` can be shown to be `O(||eta ~∇||^2)` under the parameter regime the manuscript actually operates in. The blue defense would need to specify that regime explicitly (e.g., "we assume `||phi|| = O(eta ||~∇||)`") and demonstrate that PIFB:2651 ("exact in the abelian sector") remains consistent under that restriction. Without such a regime restriction, the order estimate at PIFB:2649 is wrong.

2. Hoffman 2019 (`The Case Against Reality` or his prior journal papers) contains a statement that the perceptual interface is constituted at the slow-timescale generative-model level rather than at the moment-to-moment posterior level. A direct page citation from Hoffman 2019 to that effect would defeat the wrong-domain charge.

3. The bundle-metric construction at PIFB:2693 can be shown to define a metric on the abstract bundle `E_q = P x_G B_state` (not merely on a chosen trivialization) — i.e., a metric whose value at a point `(c, q) in E_q` is independent of which `U_i` is used to trivialize. The manuscript's own admission at PIFB:2722 ("the connection acquires Maurer-Cartan cross terms ... different gauge fixings of the same agent yield different horizontal contributions") makes this a hard ask, but the formal claim is on the table.
