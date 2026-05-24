# Blue memo — geometer

## Steelman of the attack
Red's strongest move: averaging a metric (a (0,2)-tensor) over a group orbit does not
produce a geometrically meaningful object unless the orbit and the measure are well-defined,
and the manuscript dresses an ill-posed integral in the language of a "metric."

## Defense from external canon

**The naive average is genuinely frame-dependent.** Each agent's pullback metric `G_i(c)`
depends on its gauge frame `φ_i`; a direct weighted sum `(1/N) Σ w_i G_i` inherits that
dependence. A physical (observable) geometric structure must be gauge-invariant — gauge
freedom is redundancy of description, not a physical degree of freedom [Nakahara2003 Ch. 10,
gauge invariance vs equivariance; external_canon_math.md §2 "Gauge invariance vs gauge
equivariance"]. The manuscript identifies this as the "critical flaw" of the naive average.
That diagnosis is correct.

**The orbit-average is the right *target* even though it is not yet a completed object.** A
quantity that is invariant under the full gauge action is the geometrically correct notion of
"observable." Averaging a tensor over a compact group with normalized Haar measure is the
standard way to project onto the invariant subspace (group averaging / the Reynolds operator):
for a compact group `K`, `P(T) = ∫_K ρ(k) T dk` lands in the invariant subspace and is the
canonical invariant-extraction map [standard representation theory; group averaging is exactly
the construction Faddeev–Popov regulates in the non-compact / local case]. So the *form* of
the construction — average over the orbit to kill frame dependence — is principled. What
breaks is finiteness, not the idea. The manuscript states exactly this division: the
construction is "what gauge-invariant content the horizontal block could be reduced to under a
chosen regulator," explicitly "not … a finite, regulator-free gauge-invariant metric."

**Constant-g adjoint action is the same fact the gauge-theorist verifies, geometrically:** for
constant `g`, group averaging of an already-invariant tensor is the identity, so the single-
copy average is trivial. Correct.

**Transfer to the model fiber (:2986).** The manuscript asserts the construction "transfers
verbatim from the state fiber to the model fiber" because `G` acts on `B_model` by the same
associated-bundle action and the obstructions are properties of the group, not the fiber.
Geometrically sound: associated-bundle parallel transport is determined by the structure group
action `ρ(g)` [Nakahara2003 §10.3; external_canon_math.md §2 associated-bundle transport], and
the compactness/Haar obstructions live in `G`, independent of which associated fiber `ρ` acts
on. The transfer claim is correctly scoped.

## External citations
- [Nakahara2003 Ch. 10, §10.3] gauge invariance of observables; associated-bundle transport.
- [external_canon_math.md §2] invariant vs equivariant; (0,2)/(2,0) tensor transport.

## Falsification condition (argued unmet)
The defense fails if the naive average is actually gauge-invariant (it is not — it depends on
`φ_i`), or if the orbit-average form is geometrically illegitimate as an invariant-extraction
target (it is the standard group-averaging / Reynolds construction; only finiteness fails in
the non-compact/local case, which the manuscript concedes).
