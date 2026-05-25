# Memo — debate-expert-gauge-theorist (red, sur-rebuttal)

## Position

Accept blue's correction on the simulator-transport reading. Red's Vector 2 in its strongest form ("identity-copy substitute, in which case $p_i^{(s)} = q_I^{(s+1)}$ verbatim with no gauge content") is empirically refuted by the transposed-solve algebra at `MAgent_Model-main/gauge_agent/meta_agents.py:226-227`: $\omega_{ij}^\top = (\omega_i^\top)^{-1} \omega_{\mathrm{ref}}^\top$ gives $\omega_{ij} = \omega_{\mathrm{ref}} \cdot \omega_i^{-1}$, which is the non-trivial product-of-exponentials form. The simulator is not identity-copy.

The wound on sub-claim 2 (canonical fidelity) nevertheless survives in its restated form: the cross-scale $\Omega_{i,I}$ is a *frame-change*, not a *parallel-transport-of-a-specified-connection*. Blue's own rebuttal §Core attack final sentence concedes this: "frame-change in canonical $U_I U_i^{-1}$ form is mathematically substantive content that red's identity-copy reading erases. Vector 2 should be re-stated as 'the cross-scale $\Omega_{i,I}$ is not a parallel-transport map of a connection,' not 'may be identity-copy.'" Blue's concession of the restated wound is on the record.

## Why this matters under sub-claim 2

The manuscript at line 2254 writes $\Omega_{i,I}$ as the product-of-exponentials form $U_i U_I^{-1}$ and invokes it as the cross-scale transport on top of the cross-scale-shadow priors at line 2247: $p_i^{(s)}(x) = \Omega_{i,I}[q_I^{(s+1)}](x)$. The Ouroboros fragment at line 2270 then chains these transports across $k$ generations, weighted by geometric discount $\rho^k$. The mathematical content of "transport" here is load-bearing for the participatory-loop interpretation: if the cross-scale map is merely a vertex-local frame-change (a section of the frame bundle, in the Bishop-Crittenden 1964 Ch. V §4 sense) rather than parallel transport along a curve relating scale-$s$ to scale-$(s+1)$ structures (in the Nakahara 2003 §10.3 / Kobayashi-Nomizu 1963 §II.7 sense), then the iterated $\rho^k$-discounted KL terms in the Ouroboros sum are *not* path-ordered holonomies of a connection on a principal bundle. They are products of vertex-local re-expressions of distributions.

This matters because the manuscript's interpretive payoff at lines 2243–2293 (Top-Down Participation: Closing the Loop; Self-Referential Closure) reads the cross-scale chain as a participatory feedback loop with gauge structure. Without a connection 1-form on a principal bundle whose base relates the scales, the "loop" is a graph traversal, not a holonomy. Atiyah 1979 *Geometry of Yang-Mills Fields* Ch. 2 is explicit on the global-gauge-transformation vs path-ordered-exponential-of-connection distinction. The manuscript does not provide the connection 1-form for the cross-scale identification — that is the load-bearing absence.

## Karcher non-compact

Blue's rebuttal §Defense on Vector 3c argues the manuscript's $\mathrm{SO}(N)$-in-simulations parenthetical at line 2160 discharges the non-compact $\mathrm{GL}^+(K)$ caveat. This defense is correct for the simulation regime [Karcher 1977; Moakher 2002 *SIAM J. Matrix Anal. Appl.* 24(1)]. It is not correct for the manuscript's *theoretical* claims, which are framed on the full $\mathrm{GL}^+(K)$ gauge group of §Theory. The non-compact caveat at 2160 reads, in full, "a modeling decision the present implementation does not adjudicate" — meaning the manuscript *itself* labels the non-compact case undischarged. Calling this "anticipatory" is reasonable; calling it discharged is not. The honest reading is: simulations are scoped to $\mathrm{SO}(N)$; theoretical claims at the full-$\mathrm{GL}^+(K)$ level are not.

## Concession

The identity-copy reading of Vector 2 was over-reach. The restated wound — frame-change is not parallel-transport-of-a-connection — is the correct form, and is on blue's record as conceded.

## Newly-discovered canon

None — Atiyah 1979 *Geometry of Yang-Mills Fields*, Bishop-Crittenden 1964 *Geometry of Manifolds* Ch. V, Nakahara 2003 §10.3, Kobayashi-Nomizu 1963 Vol. I §II.7, Karcher 1977, Moakher 2002 already on the record in the round 2 / round 3 extended-evidence pack.
