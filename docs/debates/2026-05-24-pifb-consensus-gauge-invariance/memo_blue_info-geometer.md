# Blue memo — info-geometer

## Steelman of the attack
Red's strongest move: the "gauge invariance from consensus" interpretation is circular — the
GL(K,ℝ)-invariance of Gaussian KL is a setup property of the divergence, so "consensus" does
not derive gauge invariance, it presupposes it.

## Defense from external canon

**The invariance is real and it is a property of the divergence, not of consensus.** Gaussian
KL `KL(N(μ_q,Σ_q) ‖ N(μ_p,Σ_p))` is invariant under a common affine reparameterization of both
arguments; the deeper statement is Čencov's theorem — the Fisher metric (and the KL it
locally generates) is the unique divergence invariant under sufficient statistics / invertible
reparameterization [Cencov1972; AmariNagaoka2000 Ch. 2; external_canon_math.md §1]. Under a
common GL(K,ℝ) change of frame applied to both `q` and `p`, KL is unchanged. This is a
mathematical fact about the divergence, present before any multi-agent story is told.

**Therefore the honest reading is that "consensus" is a *re-description* of an invariance
already present, not a derivation of it.** The manuscript says exactly this: the interpretation
is "a metaphysical interpretation rather than a derivation, and the hypothesis may not be
falsifiable from within the framework" (:2992). Because the GL(K,ℝ)-invariance is built into the
belief fiber by Čencov-type uniqueness, an agent maintaining a *different* frame computes the
*same* divergence — agreement is automatic, not an extra requirement that forces invariance.
The manuscript's downgrade to "metaphysical interpretation, not a derivation" is the
information-geometrically correct characterization. It is what the claim says it is.

CONCESSION the info-geometry lens forces: if one read the section as *deriving* gauge invariance
from consensus, that reading would be circular (invariance assumed in the construction is then
"recovered" as a consensus requirement). The defense does not defend that reading — it defends
that the manuscript already disowns it. The claim under evaluation is that the self-flag is the
honest characterization; on the information geometry, it is.

## External citations
- [Cencov1972] Fisher/KL is the unique sufficient-statistic-invariant divergence — the
  invariance is intrinsic to the divergence, not produced by consensus.
- [AmariNagaoka2000 Ch. 2] closed-form Gaussian KL and its reparameterization invariance.

## Falsification condition (argued unmet)
The defense fails if Gaussian KL is NOT GL(K,ℝ)-invariant under a common frame change (it is,
by Čencov), OR if the manuscript framed the interpretation as a derivation rather than a
metaphysical re-description (it explicitly does the latter at :2992). The self-flag therefore
correctly identifies the interpretation's status.
