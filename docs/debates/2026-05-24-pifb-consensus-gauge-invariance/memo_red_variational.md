# Memo — Red / variational / opening

## Steelman
Read as an FEP/active-inference construction, the consensus metric answers a real question:
when N agents each run variational free-energy minimization over their own belief fiber, what
is the shared structure they converge on? The manuscript's answer — a Haar-averaged,
frame-quotiented metric — is the natural "intersubjective" object, and the manuscript is careful
to call it a correlate/heuristic ("a correlate of objective reality," :3309; "the closest analog
to objective reality... read in this conditional sense," :2986) rather than a finished observable.

## Falsify
The variational framing sharpens the circularity rather than rescuing it. In FEP, agents
minimize F = E_q[log q − log p] = KL(q ‖ p(s|o)) − log p(o) [Friston2010]. Consensus, in this
setting, is a stationary point of the inter-agent alignment terms — and those alignment terms are
themselves KL divergences between transported beliefs (the framework's own β_ij and γ_ij coupling
terms are KL(q_i ‖ Ω_ij q_j)). Because KL is GL(K)-invariant on the Gaussian fiber
(info-geometer memo), the alignment objective is frame-invariant *before any consensus forms*.
So the claim "for agents to agree, the shared structure must be gauge-invariant" (:2990) is not a
constraint that consensus dynamics discover; it is a property the alignment functional has at
initialization. The variational dynamics cannot select gauge invariance because there is no
frame-dependent competitor for them to suppress — every admissible frame yields the same KL.

This is where the "objective reality" prose, even hedged, still does work it has not earned. The
section presents gauge invariance as *emerging from* consensus formation ("it emerges from the
informational requirements of consensus formation among agents with diverse perspectives,"
:2992). Emergence implies a process whose endpoint differs from its start. Here the endpoint
(gauge-invariant shared structure) equals the start (a gauge-invariant comparison functional).
Nothing emerges; the property is conserved. The manuscript discharges the *ontological* claim
(it is a "correlate," conditional on a regulator) but not the *dynamical* claim (it "emerges").
The dynamical claim is the one the variational reading exposes as empty.

I concede the prose-discharge vector: :2986 does retroactively recast "closest analog to
objective reality" in the conditional sense, and the cross-reference locations (:3309, :3313)
are consistently hedged as "correlate" and "interpretive thesis... not a derivation." Attack the
*emergence* verb, not the *ontology* noun.

## External primary-source citation (not the manuscript)
- Friston, K. (2010), "The free-energy principle: a unified brain theory?" *Nat. Rev. Neurosci.*
  11(2):127–138 — F = E_q[log q − log p] = KL(q‖p(s|o)) − log p(o); minimization bounds surprise.
  The alignment/consensus terms are KL functionals, hence inherit GL(K)-invariance.

## Falsification condition
Wrong if the consensus dynamics can drive a frame-dependent shared structure to a frame-invariant
one — i.e., if there exists an admissible initialization whose alignment objective is NOT already
frame-invariant, so that invariance is a genuine attractor rather than a conserved quantity. The
GL(K)-invariance of KL (info-geometer) rules this out on the Gaussian belief fiber.

## Newly-discovered canon
- Friston (2010), *Nat. Rev. Neurosci.* 11:127–138 — canonical F decomposition; consensus
  alignment terms are KL functionals.
