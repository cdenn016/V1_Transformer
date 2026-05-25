# Blue memo — info-geometer — Phase 3 (rebuttal)

## Target attacks in `02_red_opening.md`

- Vector 2(b) (sub-claim 7): Red claims "the 3084 invocation of 'Cencov scale fixings' as a load-bearing primitive of the future construction is a category error — Cencov removes scale, it does not fix it" [Cencov 1972; Amari–Nagaoka 2000 §2.3; external_canon_math.md §1].
- Vector 2 supporting fire: Red invokes information geometry for the claim that the dimensionless-constants programme at 3082 names canonical primitives in a way the canonical literature does not authorize.

## Concession (one red argument that holds under canon)

Red is correct on the canon. [Cencov 1972 *Statistical Decision Rules and Optimal Inference*; Amari–Nagaoka 2000 *Methods of Information Geometry* §2.3] establish that the Fisher information metric is the unique (up to a *positive scalar multiple*) Riemannian metric on a statistical manifold invariant under sufficient statistics. The uniqueness is *up to scalar.* That scalar is precisely *not fixed* by Cencov's theorem — it is a remaining one-parameter freedom in the construction. The phrase "Cencov scale fixings" in PIFB:3084 names the canonical theorem as if it fixed scales, which is a misreading: canonically, Cencov delivers scale *invariance up to a positive multiplier,* not scale *fixing.* The closest canonical operation that *does* fix scales is the choice of a normalization convention (e.g., choosing $g_{\mathrm{Fisher}}^{(\text{normalized})}$ such that some reference distribution has $g_{11} = 1$, or pinning the metric to a physically meaningful invariant like the Bures metric on quantum states); none of these is Cencov's theorem itself. Blue's Phase 2 defense of sub-claim 7 cannot rest on a defense of the 3084 prose as written. **I concede the category-error wording at PIFB:3084.**

## Core attack (red's load-bearing weakness under canon)

Red's attack lands on *prose wording*, not on the *structural logic of the research programme.* The 3082 programme — "Dimensionless ratios between fundamental constants should be derivable from pure information geometry if this interpretation is correct. The fine structure constant $\alpha \approx 1/137$ is dimensionless. It should emerge from ratios of coupling strengths or Fisher information scales without reference to kilograms or meters" — is a structural research-programme statement, not a derivation. The Cencov-uniqueness-up-to-scalar result [Amari–Nagaoka 2000 §2.3] is *relevant* to such a programme precisely because dimensionless ratios are scale-invariants and Cencov shows the Fisher metric is the unique metric on a statistical manifold up to scalar. Ratios of Fisher information eigenvalues, ratios of canonical curvatures, ratios of dual-affine coordinates — these are all scale-invariants in the Cencov sense and are precisely the kind of object that *could* yield dimensionless physical constants if the programme succeeded.

What red elides is that the natural canonical primitive for the 3082 programme is *not* a "scale fixing" at all but a *scale-invariant ratio.* The right rewrite of 3084 is "a specified canonical normalization or, equivalently, an explicit scale-invariant ratio of canonical info-geometric quantities," which honors Cencov's uniqueness-up-to-scalar register correctly. This is an editorial fix at one line; it does not falsify the programme's logical structure.

[Amari 1998 *Neural Computation* 10:251 §2; Amari–Nagaoka 2000 §3.2 (α-connections); Lauritzen 1987 *Differential Geometry in Statistical Inference* IMS Lecture Notes 10] adds the second relevant canonical structure: the α-connection family on a statistical manifold (with α=0 the canonical mixture connection, α=1 the exponential connection, α=-1 the dual) provides a one-parameter family of canonical objects on the manifold. The 3082 programme's "ratios of coupling strengths or Fisher information scales" can be read as ratios within or between α-connections — a canonical structure that supplies the kind of dimensionless ratio the programme commits to.

[Bishop 2006 *Pattern Recognition and Machine Learning* §10.1.1; Skovgaard 1984 *Scand. J. Statist.* 11:211] adds the third relevant canonical structure: the variational free-energy Hessian and the posterior Fisher precision coincide for quadratic potentials. The PIFB:3070 within-framework structural identification of $M_{\mathrm{eff}}$ with $\Sigma_p^{-1}$ is canonical under this Hessian/Fisher coincidence. The 3082 programme commits to deriving dimensionless ratios from such canonical info-geometric structures; the structures exist canonically, even though the derivation does not exist in the present manuscript.

The narrow concession red wins on Vector 2(b) is the prose at 3084, not the programme structure. The information-geometric programme at 3082 invokes canonical primitives — Fisher scalar-invariants, α-connections, Hessian/Fisher precision coincidence — that exist in the canon; it does not deliver the derivation. The 3094–3102 "metaphysical postulate, not a derived result" admission and the 3100 "First, attempt to derive known dimensionless constants" research-program framing register this status honestly. Red's Vector 2 thus succeeds in a narrow editorial sense (rewrite 3084) and fails in the broader sense (the sub-claim 7 calibration mesh holds with the editorial fix).

## Defense (citation that strengthens blue's position)

The strongest info-geometric defense of the sub-claim 7 calibration mesh:

- [Amari–Nagaoka 2000 §2.3] — Cencov uniqueness up to positive scalar; this is what 3084 *should* invoke, and the rewrite "a specified canonical normalization, given Cencov uniqueness up to a positive scalar [Amari–Nagaoka 2000 §2.3]" makes the structure precise without category error.
- [Amari–Nagaoka 2000 §3.2] — α-connections as canonical one-parameter family on statistical manifolds. The 3082 "ratios of coupling strengths" reading is canonically grounded in α-connection structure.
- [Bishop 2006 §10.1.1; Skovgaard 1984] — Hessian/Fisher precision coincidence. The 3070 within-framework $M_{\mathrm{eff}} = \Sigma_p^{-1}$ identification is canonical under this result.
- [Cencov 1972 (the original)] — invariance theorem stated as: any sequence of statistical maps preserving sufficient statistics induces a Riemannian metric on the statistical manifold that agrees with the Fisher metric up to a positive multiplier. Cencov is the *invariance* theorem, not a *normalization* theorem; using its name as "scale fixing" reverses the canonical reading.

The sub-claim 7 calibration mesh holds *for the research-programme structure* under canonical info-geometric primitives; the calibration fails *only at the 3084 prose wording* and is curable by editorial rewrite. Blue's Phase 2 defense of sub-claim 7 should narrow to this position.

## Newly-discovered canon

- **Amari–Nagaoka 2000, *Methods of Information Geometry* §2.3, §3.2.** Cencov uniqueness up to positive scalar; α-connection family. Both are canonical primitives the 3082 research programme can canonically invoke; the 3084 prose mis-names the first.

- **Cencov 1972, *Statistical Decision Rules and Optimal Inference* §VII.** The original uniqueness theorem. Cencov: invariance under sufficient statistics. The theorem is *scale-removing in the sense of identifying the metric up to scalar,* not *scale-fixing.*

- **Lauritzen 1987, *Differential Geometry in Statistical Inference*, IMS Lecture Notes 10.** Canonical treatment of dual affine connections and the α-connection family in statistical inference. Provides the additional canonical structure the 3082 programme can invoke.

- **Bishop 2006 §10.1.1; Skovgaard 1984 *Scand. J. Statist.* 11:211.** Variational free-energy Hessian and posterior Fisher precision coincide for quadratic potentials. Canonically supports PIFB:3070 mass-from-Fisher identification as a within-framework structural identification.
