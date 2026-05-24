# Blue Opening — pifb-consensus-gauge-invariance

## Steelman (opposing position)

The section dresses an ill-posed object in the language of geometry: it presents a
gauge-averaged "consensus metric" as a route to a gauge-invariant observable when the
construction provably cannot deliver one, and it props up an unfalsifiable "gauge invariance
from consensus" interpretation that is circular — the GL(K,ℝ)-invariance it claims to explain
is assumed into the belief fiber from the start — so no amount of hedging makes the treatment
honestly characterized.

## Position

The claim under evaluation is narrow and it is defensible: the §"Collective Geometry and Gauge
Invariance" treatment (Attention/Participatory_it_from_bit.tex:2954–2992) is *honestly
characterized*. Every gauge-theoretic obstruction it states is standard and correctly stated,
and the consensus metric and the consensus interpretation are explicitly downgraded — to "a
heuristic target rather than a completed observable" (:2986) and to "a metaphysical
interpretation rather than a derivation, and the hypothesis may not be falsifiable" (:2992).
The claim is NOT that the consensus metric works; the manuscript does not claim that, so the
blue position does not defend it. The claim is that the self-assessment is accurate. It is
accurate on three of four sub-points and partially incomplete on the fourth: two residual
overclaims (the earlier "closest analog to objective reality" prose, and the
U(1)/SU(2)/SU(3)/SO(1,3) "available as subgroups" listing) survive the in-place hedges and
require local repair. With those two trims, the treatment is sound as characterized.

## Evidence

**Sub-point 1 — constant-`g` cyclicity is correct (:2977).** Under a constant gauge
transformation the connection transforms by the adjoint action `A → g⁻¹ A g`
[Nakahara2003 Ch. 10; KobayashiNomizu Vol. I §III.2], so
`tr(g⁻¹ A_μ g · g⁻¹ A_ν g) = tr(g⁻¹ A_μ A_ν g) = tr(A_μ A_ν)` by trace cyclicity. The
manuscript's statement that `tr(A_μ A_ν)` is "already invariant by cyclicity of the trace" and
that the single-copy Haar average is "either trivial or unnecessary, and does not by itself
rescue the non-invariance of the horizontal block" is exactly right — and notice it withholds
the rescue rather than claiming it. (gauge-theorist, geometer.)

**Sub-point 2 — the local-`g(c)` functional-integral obstruction is correct (:2977).** Under a
local transformation the connection acquires the inhomogeneous Maurer–Cartan term
`A → g⁻¹ A g + g⁻¹ dg` [Nakahara2003 Ch. 10], the defining transformation law of a connection
1-form. An honest gauge-orbit average over local `g(c)` must integrate over the maps
`g : C → G`, an infinite-dimensional functional integral that requires gauge-fixing or a
regulator. This is the Faddeev–Popov canon: the naive integral over gauge-equivalent
configurations overcounts by the infinite volume of the gauge orbit, and a gauge-fixing
condition with the Faddeev–Popov determinant is needed to extract a finite answer
[PeskinSchroeder1995 Ch. 9; Weinberg QFT Vol. II]. The harvested canon strengthens the
manuscript's position rather than weakening it: the local gauge group `Map(C,G)` is not even
locally compact, so a Haar measure on it is not guaranteed to exist (arXiv:hep-th/0103160) —
the manuscript's "in general no finite gauge-orbit average over local `g` exists without such a
choice" is conservative. (gauge-theorist, geometer.)

**Sub-point 3 — non-compact Haar is correct (:2977).** A locally compact group has a finite
(normalizable) Haar measure if and only if it is compact; for non-compact groups every
left-invariant Haar measure is infinite [standard Haar-measure theorem, abstract harmonic
analysis]. `SO(1,3)` is non-compact, so its Haar measure is infinite even for constant `g`, as
the manuscript states. (gauge-theorist.)

**Sub-point 4 — the interpretation is correctly flagged, because the invariance is a setup
property (:2988–2992).** Gaussian KL on the belief fiber is GL(K,ℝ)-invariant under a common
frame change; the deeper reason is Čencov's theorem — the Fisher metric and the KL it generates
are the unique divergence invariant under sufficient statistics / invertible reparameterization
[Cencov1972; AmariNagaoka2000 Ch. 2]. The invariance is therefore intrinsic to the divergence,
present before any multi-agent story. It follows that "gauge invariance arises as a consistency
requirement for multi-agent consensus" cannot be a *derivation* of gauge invariance — read that
way it is circular, recovering an assumption as a conclusion — and can only be a re-description
of an invariance already imposed. The manuscript adopts exactly the non-derivational reading:
":2992" calls it "a metaphysical interpretation rather than a derivation, and the hypothesis
may not be falsifiable from within the framework." Disclosing an interpretation as untestable
and quarantining it from the empirical core is the correct epistemic act
[Popper, *Logic of Scientific Discovery*, demarcation], not a dodge. No empirical or
derivational claim is rested on the consensus interpretation; both "imposed on nature" and
"emergent from consensus" predict identical frame-independent observables, so the
unfalsifiability admission is accurate. (info-geometer, variational, philosophy-of-science.)

**The base inference layer is standard.** The per-agent objective is the ordinary FEP free
energy `F = KL(q‖p) − log p(o)` [Friston2010]; the consensus construction sits on top of a
single-agent free energy that reduces correctly at N=1 [external_canon_math.md §4]. Nothing
non-standard is smuggled into the per-agent inference. (variational.)

**Concessions (where the in-place hedges do not fully discharge the overclaim).** Two residual
overclaims survive and should be fixed; conceding them is the honest disposition.
First, the prose elsewhere calling the consensus metric "the closest analog to objective
reality" is retroactively conditionalized only at :2986 ("should be read in this conditional
sense"); a reader who reaches the earlier location first meets an unconditioned overclaim, so an
inline hedge or forward cross-reference is needed at the earlier location. Second, the listing of
U(1), SU(2), SU(3), SO(1,3) as "available as subgroups of the connection-sector gauge symmetry"
(:2992) carries no derivational content — every compact Lie group embeds in some GL(n), so
subgroup containment alone does not select or derive the Standard Model gauge group. The
manuscript does hedge "whether specific subgroups are dynamically selected … remains open," but
the sentence still gestures; trim it to the bare containment fact or delete it.
(philosophy-of-science, gauge-theorist.)

All five panel lenses are represented above: gauge-theorist and geometer carry sub-points 1–3,
info-geometer and variational carry sub-point 4 and the circularity analysis, and
philosophy-of-science carries the frame-check and the two concessions. No memo is discounted.

## Falsification conditions

This blue position — that the section is honestly characterized modulo two local trims — is
wrong if any of the following hold:

1. `tr(A_μ A_ν)` is NOT invariant under constant `g`. It is, by trace cyclicity of the adjoint
   action [Nakahara2003]; this falsifier is unmet.
2. The local-`g(c)` gauge-orbit average admits a finite, regulator-free value. It does not; the
   Faddeev–Popov canon requires gauge-fixing, and `Map(C,G)` may not even carry a Haar measure
   [PeskinSchroeder1995 Ch. 9; arXiv:hep-th/0103160]; unmet.
3. `SO(1,3)` has finite Haar measure. It does not — non-compact groups have infinite Haar
   measure [Haar-measure theorem]; unmet.
4. The manuscript at :2954–2992 does NOT contain the hedges the claim attributes to it. It does:
   :2977 "heuristic … requires a gauge fixing or a regulator … no finite gauge-orbit average
   over local `g` exists without such a choice"; :2986 "a heuristic target rather than a
   completed observable … conditional on a regulator whose construction is left to future work";
   :2992 "a metaphysical interpretation rather than a derivation … may not be falsifiable."
   Unmet.
5. The section rests an empirical or derivational claim on the unfalsifiable consensus
   interpretation. It does not — the interpretation is quarantined and labeled; unmet.

The position is PARTIALLY constrained by a sixth condition, which is met and which the blue side
concedes: if the in-place hedges fail to neutralize overclaims at *other* locations, the
"honestly characterized" claim is incomplete. This is met at two spots — the earlier "objective
reality" prose and the subgroup listing — and the remedy is two local edits, not a retraction of
the section. The blue side does not defend that the consensus metric is a finite gauge-invariant
observable; the manuscript does not claim it, and on the canon it is not one.
