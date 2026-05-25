# Red Memo — info-geometer — Phase 3 rebuttal

Target: `02_blue_opening.md` Pillar 3 closing paragraph (the Cencov-as-scale-fixing "editorial roughness" concession) and Falsification condition 3 ("if red can produce a contrary canonical citation, the corresponding pillar fails").

## Concession

Blue's deployment of [Amari-Nagaoka 2000 §2.1, §7.2, §7.4] on the block-Fisher form for the K-dimensional Gaussian family and on the SPD natural-gradient sandwich form `dΣ/dt = -2 η_Σ Σ (∇_Σ F) Σ` is canonically correct. PIFB:2670–2676 and PIFB:2623–2627 do match these standard treatments. The information-geometric primitives that drive the construction's mechanics are not where the load-bearing failure sits. Granted.

## Core attack

Blue concedes the Cencov-as-scale-fixing phrasing at PIFB:3084 is "technically loose (Cencov is uniqueness-up-to-scalar, not scale-fixing [AmariNagaoka 2000 §2.3])" and immediately reframes this as "editorial roughness rather than load-bearing failure, since the symmetric-success-failure criterion's substantive content is structural specification, not the technical precision of which theorem is invoked" (`02_blue_opening.md` Pillar 3 closing). The reframing is wrong on the structural-vs-editorial axis, and the correct read is the opposite of blue's.

The PIFB 3084 criterion reads (verified from `Attention/Participatory_it_from_bit.tex:3084`):

> Failure of the derivation under a fully developed information-geometric construction --- one in which the relevant statistical manifolds, comparison structure, and Cencov scale fixings are all specified --- would constitute substantive evidence against the phenomenological interpretation.

The criterion's falsifiability rests on three named primitives of the "fully developed construction": (i) the relevant statistical manifolds, (ii) comparison structure, (iii) Cencov scale fixings. Primitive (iii) is fictitious. Cencov's theorem [Cencov 1972; restatement at AmariNagaoka 2000 §2.3] is:

> The Fisher metric is the unique (up to a positive scalar multiple) Riemannian metric on a statistical manifold invariant under sufficient statistics.

This is a uniqueness-modulo-scalar theorem. Its content is to *remove* a one-parameter freedom (the overall scalar) by routing through the invariance principle; it does not *introduce* scale fixings. The phrase "Cencov scale fixings" inverts the theorem: Cencov leaves a one-parameter scalar undetermined; the load-bearing primitive that the 3084 criterion conditions on therefore does not exist under the theorem the manuscript names.

This is not editorial roughness. A falsifiability criterion that conditions on a deferred construction whose named primitive does not exist is structurally underspecified at the load-bearing layer. Two reformulations would have been admissible:

1. The 3084 criterion could have invoked an *additional* scale-fixing principle separate from Cencov (e.g., a normalization convention tied to a physical reference measurement). The manuscript does not do this.
2. The 3084 criterion could have been about dimensionless ratios *invariant under the Cencov scalar* (e.g., $m_e/m_p$), in which case the scalar is irrelevant and Cencov's uniqueness-up-to-scalar suffices. The manuscript names dimensionless ratios as the target at PIFB:3082 ("Dimensionless ratios between fundamental constants should be derivable...") — but then the 3084 criterion does not need "Cencov scale fixings" to be among the load-bearing primitives. Naming "Cencov scale fixings" at the falsifiability-criterion layer is therefore either redundant (if the target is dimensionless ratios) or incoherent (if the target requires scale fixing that Cencov cannot deliver).

Either reading collapses the criterion. The structural fact is that PIFB:3084 names a primitive that the cited theorem disclaims. This is not "rock-solid" by the present-tense reading; it is a load-bearing slip dressed as a research-program desideratum.

## Defense

The further problem with blue's "structural specification" rescue: the PIFB 3082–3084 commitment to derive dimensionless constants from "pure information geometry" sits outside the canonical scope of variational inference and information geometry. Amari-Nagaoka [AmariNagaoka 2000 Ch. 2–3] develops the Fisher metric, dual affine connections, and α-geometry as tools for statistical inference, parameter estimation, and exponential-family characterization. These tools have no canonical mechanism for fixing the values of physical dimensionless constants such as $\alpha \approx 1/137$ or $m_e/m_p \approx 1/1836$. Such constants live in the empirical content of physical theories with their own measured inputs (the Standard Model takes them as free parameters; no derivation from information-geometric first principles is known in the literature).

The PIFB framework therefore conditions its falsifiability on a derivation that:
- has no precedent in the canonical information-geometric literature [AmariNagaoka 2000; Amari 1998 *Neural Computation* 10, 251–276],
- names a foundational primitive (Cencov scale fixings) that the cited theorem cannot deliver,
- is admitted at 3060 to "make the framework compatible with any result by construction" if reinterpretation is permitted.

Blue's "burden-of-construction clause" assigns the burden to the research programme but provides no presently-achievable mechanism for discharging it. Under [AmariNagaoka 2000 §2.3] read strictly, the named mechanism cannot be discharged because the named primitive is fictitious. This is the structural compound of red's Phase-2 attack on sub-claim 7 that blue's Pillar 3 does not answer.

## Newly-discovered canon

- **AmariNagaoka 2000 §2.3** [via `external_canon_math.md §1` and `external_bibliography.md:17`] — Cencov uniqueness theorem: "the Fisher metric is the unique (up to a positive scalar multiple) Riemannian metric on a statistical manifold invariant under sufficient statistics." Confirms Cencov is scale-removing, not scale-fixing.
- **Amari 1998 *Neural Computation* 10, 251–276 §2** — natural gradient on Riemannian manifolds; the Fisher metric is normalized by the operational invariance principle, not by reference to an external scale-fixing convention. No mechanism for fixing physical constants is introduced.
- **AmariNagaoka 2000 Ch. 2–3** — scope of information geometry: statistical inference, parameter estimation, exponential-family characterization. No standard-model-style constant derivation appears in the canonical literature.
