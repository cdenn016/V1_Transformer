# Verdict — pifb-spec-2-signature

## Outcome

RED_WINS

## Decisive evidence

The manuscript at `Attention/Participatory_it_from_bit.tex:2824` introduces two distinct generators in the frame-field decomposition,

```
phi(tau, x) = psi_tau(tau, x) . T_tau + psi_x(tau, x) . T_x,
```

and at line 2828 silently uses a single generator,

```
phi(tau, x) = i psi_tau(tau, x) . T + psi_x(tau, x) . T.
```

No sentence between 2824 and 2858 names the identification `T_tau = T_x = T` as a postulate, simplification, or input. The 2826 paragraph names only the imaginary-`psi_tau` postulate; the 2831 sentence names the separable ansatz; the 2846 paragraph names the real-part projection; the 2856 summary lists imaginary `phi_tau` and real-part projection. The single-generator identification (equivalently, the requirement `tr(T_x^2) > 0` selecting `T_x` from the non-compact part of `gl(2,C)`) is load-bearing: red's sympy verification with `T_tau = diag(1,-1)`, `T_x = [[0,1],[-1,0]]` — both admissible Lie-algebra elements in `gl(2,C)`, the latter compact — yields signature `(-, -)` not `(-, +)`. The Lorentzian conclusion of Eq. `lorentzian_metric` therefore requires a postulate the manuscript does not name. Blue's rebuttal concedes this verbatim: "Operational test 2 of `00_claim.md` ('each postulate explicitly flagged') fails on this specific point."

## Reasoning

The operational reading at line 21 of `00_claim.md` is binding: "Each postulate must be explicitly flagged as a postulate. These must not silently slip into a derivation." Line 23 makes the consequence concrete: "A red strike lands if any postulate is implicit." Red identified the `T_tau = T_x = T` collapse as the silent postulate, executed sympy to confirm that the Lorentzian-signature conclusion fails under a different admissible choice of `T_x`, and cited the manuscript line numbers showing no flag. Blue conceded the strike without qualification, granted red's sympy verification, and acknowledged that the real-part projection at 2846 is rank-changing rather than off-diagonal-discarding as the text currently paraphrases. Blue's defense narrowed to "a one-paragraph edit makes the section publication-ready" — but the claim under debate is that the section is publication-ready and rock-solid as written. Blue's remediation pathway is itself an admission that the as-written text does not meet the standard.

The cross-section incoherence at lines 2723 and 2928 reinforces the verdict. The signature construction in §sec:signature_resolution places its Lorentzian conclusion on `tr(A_mu A_nu)`, which §sec:pullback at 2723 labels "agent-frame-dependent" and routes gauge invariance through §sec:consensus_metric. §sec:consensus_metric at 2928 then admits explicitly: "the non-compact `SO(1,3) ⊂ GL(K, C)` case carries the additional obstruction that the Haar measure is infinite even for constant g. We therefore retain the construction below as a heuristic ... but explicitly do not claim it produces a finite, regulator-free gauge-invariant metric." Two consecutive subsections in the same chapter therefore assert (a) Lorentzian signature lives on the horizontal block, and (b) the horizontal block has no extractable gauge-invariant content under `SO(1,3)`. Blue did not rebut this composition. A rock-solid subsection does not contradict the next subsection on the existence of its own observable.

Blue defended the 2856 vs 2858 framing reading (structural locator vs dynamical selection) and the Wick framing (two compatible aspects of the same operation), and those defenses are plausible. They do not, however, restore the load-bearing strike. The verdict turns on the verified silent postulate plus the cross-section incoherence, both of which red supports with the primary source (the manuscript itself) and an executed sympy session, and neither of which blue successfully rebutted. The source-of-truth precedence rule applies: red's decisive evidence is a `path:line` reference to the actual manuscript text plus executed verification, not an appeal to the manuscript's own authority.

## Action

Edit `Attention/Participatory_it_from_bit.tex` lines 2822-2856 to add the single-generator postulate to the named-postulate list and reframe the real-part projection at 2841-2846 as a rank-changing operation rather than as discarding an off-diagonal imaginary piece. Concretely:

1. At line 2826, after "we now \emph{postulate} that the temporal component is imaginary," add a second postulate sentence naming the single-generator simplification: "we further postulate, for this two-dimensional worked example, that the same generator `T = diag(1,-1)` carries both base directions, equivalently `T_tau = T_x`. With two trace-orthogonal generators (e.g., `T_tau = diag(1,-1)`, `T_x = [[0,1],[1,0]]` in `sl(2,R)`) the off-diagonal entry of `G_{mu nu}` vanishes identically and the real-part projection step is not required; the projection step in the present worked example is therefore a consequence of the single-generator simplification rather than of the imaginary-`psi_tau` postulate alone."

2. At line 2841, reframe the real-part projection: "Under the single-generator simplification, `G_{mu nu} = c_mu c_nu tr(T^2)` is a rank-1 complex outer product (`det(G) = 0`); the real-part projection is therefore a rank-changing operation, mapping a degenerate rank-1 complex form to the non-degenerate rank-2 real form in Eq.~\eqref{eq:lorentzian_metric}."

3. At line 2856, extend the explicit-postulate list to four items: (i) imaginary `psi_tau` along the distinguished direction, (ii) `+tr(AB)` sign convention, (iii) single-generator (`T_tau = T_x`) simplification within the worked example, (iv) real-part projection of the resulting rank-1 complex frame-twist form.

4. Address the §sec:consensus_metric incoherence by adding a paragraph at the close of §sec:worked_signature (around line 2858) acknowledging that the gauge-invariant observable status of the constructed Lorentzian metric is conditional on the regulated consensus-metric construction of §sec:consensus_metric, which the manuscript itself flags as not regulator-free for non-compact `SO(1,3)`. Frame the signature construction as conditional on a separate regulation choice rather than as a standalone signature derivation.

After these edits, the subsection is publication-ready as an explicit-postulate structural-compatibility demonstration. Without them, the claim "publication-ready and rock-solid" does not hold.
