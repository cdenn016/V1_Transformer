# Blue memo — gauge-theorist

## Steelman of the attack
Red's strongest move: the section is not "honestly characterized" because it presents a
gauge-averaged metric as a route to a gauge-invariant observable while the construction
provably cannot deliver one — so any framing short of "this does not work" is dressing-up.

## Defense from external canon

**Sub-point 1 — constant-g cyclicity (manuscript :2974).** Under a constant gauge
transformation the connection transforms by the adjoint action `A → g⁻¹ A g`
[Nakahara2003, Ch. 10 connection transformation; KobayashiNomizu Vol. I §III.2]. The
trace bilinear is then `tr(g⁻¹ A_μ g · g⁻¹ A_ν g) = tr(g⁻¹ A_μ A_ν g) = tr(A_μ A_ν)` by
cyclicity of the trace. The manuscript's statement that `tr(A_μ A_ν)` is "already invariant
by cyclicity of the trace" and that "the Haar average over a single copy of G is then
either trivial or unnecessary" is exactly correct. The manuscript does NOT use this to
claim the horizontal block is rescued; it says the opposite ("does not by itself rescue
the non-invariance"). Honest.

**Sub-point 2 — local-g Maurer–Cartan obstruction (:2974).** Under a local transformation
the connection picks up the inhomogeneous term `A → g⁻¹ A g + g⁻¹ dg` [Nakahara2003 Ch. 10;
KobayashiNomizu Vol. I, structure equation / connection-form transformation]. This is the
defining transformation law of a connection 1-form and is standard. An honest gauge-orbit
average over local `g(c)` must integrate over the space of maps `g : C → G`, i.e. over the
infinite-dimensional local gauge group. The manuscript says precisely this and that it
"requires a gauge fixing or a regulator to be well-defined." This matches the Faddeev–Popov
canon [PeskinSchroeder1995, Ch. 9 Faddeev–Popov procedure; Weinberg QFT Vol. II]: the naive
path integral over gauge-equivalent configurations overcounts by the (infinite) volume of
the gauge orbit, and a gauge-fixing condition plus the Faddeev–Popov determinant is required
to extract a finite answer.

Strengthening fact harvested at audit time: the local gauge group `Map(C,G)` is
infinite-dimensional and **not locally compact**, so it does not in general even possess a
Haar measure ([Mottola/Adler line of work, "A Note on Functional Integral over the Local
Gauge Group", arXiv:hep-th/0103160], and Faddeev–Popov literature). The manuscript's claim
is therefore conservative, not overclaiming: it says a regulator is *required*; the canon
says the measure may not exist at all without one.

**Sub-point 3 — available subgroups (:2992).** "U(1), SU(2), SU(3), SO(1,3) available as
subgroups of the connection-sector gauge symmetry" is, read literally, only a subgroup-
containment statement about GL(K,ℂ). Containment alone does not select or derive the Standard
Model gauge group — every compact Lie group embeds in some GL(n) — so this sentence carries
no derivational content. The manuscript does hedge ("whether specific subgroups are
dynamically selected … remains open"), but the listing still gestures at more than it
delivers. CONCESSION CANDIDATE: this line should be trimmed to the bare containment fact or
deleted.

## External citations
- [Nakahara2003] connection transformation law `A → g⁻¹ A g + g⁻¹ dg` (Ch. 10).
- [PeskinSchroeder1995] Faddeev–Popov gauge-fixing of the gauge-orbit integral (Ch. 9).
- [arXiv:hep-th/0103160] local gauge group not locally compact → Haar measure not guaranteed.

## Falsification condition (argued unmet)
The defense fails if `tr(A_μ A_ν)` is NOT invariant under constant `g` (it is, by cyclicity),
or if the local-`g` orbit average admits a finite regulator-free value (it does not; the
canon requires gauge-fixing). Neither holds, so the gauge-theoretic content of the self-flag
is accurate.
