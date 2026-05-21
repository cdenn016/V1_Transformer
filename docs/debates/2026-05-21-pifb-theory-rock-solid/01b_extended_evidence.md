# Extended Evidence — pifb-theory-rock-solid

This file collects canon harvested by the expert panels during Phase 2 (and later phases). The base evidence pack is `01_evidence.md`. Append; do not overwrite. Dedup across red/blue.

## Phase 2 — blue panel harvest

### From `02_blue_memo_info-geometer.md`

- **`[Milgrom Segal 2002]`** — "Envelope theorems for arbitrary choice sets," Econometrica 70(2). Canonical statement of the parametric envelope theorem: for $V(x)=\max_\beta f(x,\beta)$, $\partial V/\partial x = \partial_x f$ at $\beta=\beta^*(x)$ with no $\partial\beta^*/\partial x$ contribution. Licenses the receiver-side gradient at line 1391 of `Participatory_it_from_bit.tex`. Candidate for inclusion in `external_canon_inference.md` §5 or §10 (pitfalls).

### From `02_blue_memo_variational.md`

- **`[Sønderby2016]`** — "Ladder Variational Autoencoders," NeurIPS 2016. The deterministic-decoder limit of the ladder-VAE hierarchical prior is the construction the manuscript reduces to at line 552 under the $\sigma^2\to 0$ Gibbs limit of the cross-scale shadow. Candidate for inclusion in `external_canon_inference.md` §3 (Hierarchical / nested formulations) as the canonical reference for the ladder-VAE / point-passing relation.

### From `02_blue_memo_gauge-theorist.md`

- **`[Wilson1974]`** — "Confinement of quarks," Phys. Rev. D 10. Canonical reference for lattice gauge theory under per-vertex transformations and the Wilson-line / trace-cyclicity argument that gives invariance under the residual constant-per-agent subgroup. Relevant to the dual-role gauge-frame treatment at lines 555–571 of `Participatory_it_from_bit.tex`. Candidate for inclusion in `external_canon_math.md` §2 (lattice gauge / Wilson loop).

### From `02_blue_memo_philosophy-of-science.md`

No new canon — citations are to existing `external_canon_*.md` entries (Nakahara, Bishop, Weinberg, Donnelly-Freidel, Bartlett-Rudolph-Spekkens, Vanrietvelde, Vaswani) and to venue-level reviewing practice (NeurIPS / ICML / JMLR editorial policies on companion papers).

### From `02_blue_memo_geometer.md`

No new canon — citations are to `[Nakahara2003 §10.3, Ch. 9–10]`, `[KobayashiNomizu]`, `[Frankel2011 Ch. 17]`, `[Hall LieGroups]`, all already in `external_bibliography.md` and `external_canon_math.md`.

## Phase 2 — red panel harvest

### From `02_red_memo_philosophy-of-science.md`

- **`[Arnold1989]`** — *Mathematical Methods of Classical Mechanics*, 2nd ed., Springer GTM 60, Ch. 5 §22–25. Small-oscillations theory with generalized eigenvalue problem $V v = \omega^2 T v$ requiring inertia tensor $T$ and potential Hessian $V$ to be independent positive-definite quadratic forms at the equilibrium configuration. The manuscript cites Arnold at line 1882 to flag its own failure to supply this independence. Candidate for inclusion in `external_canon_math.md` §2 (differential geometry / classical mechanics).
- **`[Popper1959]`** — *The Logic of Scientific Discovery*, Routledge. Falsifiability as the demarcation criterion. A predictive scaling that is "a definitional consequence of the postulate" (manuscript lines 1882, 2064) is unfalsifiable as stated. Candidate for inclusion in a philosophy-of-science canon entry if one is created.
- **`[GoldstoneSalamWeinberg1962]`** — "Broken symmetries," *Phys. Rev.* 127: 965–970. Canonical Goldstone theorem. Spontaneous breaking → massless boson per broken generator; explicit Zeeman-style breaking → textbook non-example. Already cited in `external_canon_math.md` (Weinberg 1995 Vol. II §19); the 1962 paper is the primary source.

### From `02_red_memo_variational.md`

- **`[RanganathTranBlei2016]`** — "Hierarchical Variational Models," [arXiv:1511.02386](https://arxiv.org/abs/1511.02386). Canonical modern hierarchical-VI treatment: hierarchical $q$ built by placing a *prior on variational parameters*, not by passing transported posteriors between levels. The closest published precedent for what a "principled hierarchical variational scheme" looks like; the manuscript's cross-scale shadow at line 547 does not match it. Candidate for inclusion in `external_canon_inference.md` §3 (Hierarchical / nested formulations) alongside Sønderby2016.
- **`[Beal2003]`** — Beal thesis, *Variational Algorithms for Approximate Bayesian Inference*, Gatsby Unit, UCL. Conjugate-prior mean-field treatment for Gaussian families; the full Gamma-Normal joint stationary equations that Bishop §10.2 builds on. The manuscript cites Bishop §10.2 at line 1336 but does not display the joint or its stationary equations. Candidate for inclusion in `external_canon_inference.md` §6 (Mean-field, structured, amortized).

### From `02_red_memo_gauge-theorist.md`

- **`[Witten2018]`** — "Notes on some entanglement properties of quantum field theory," *Rev. Mod. Phys.* 90: 045003 ([arXiv:1803.04993](https://arxiv.org/abs/1803.04993)). Modern reference on subregion algebras and edge modes in QFT, including the von Neumann algebra structure that Donnelly-Freidel-style constructions rest on. Benchmark for whether the manuscript's edge-mode appeal transfers to the multi-agent statistical-manifold setting.
- **`[CarrozzaHoehn2022]`** — "Edge modes as reference frames and boundary actions from post-selection," *JHEP* 02: 172 ([arXiv:2109.06184](https://arxiv.org/abs/2109.06184)). Explicit identification of edge modes with quantum reference frames; constructive boundary action. The combined-license cite for the manuscript's Donnelly-Freidel + QRF appeal at line 569, with the same caveat that the constructive content rests on a symplectic apparatus the manuscript does not exhibit.

### From `02_red_memo_info-geometer.md`

- **`[BauerBruverisMichor2016]`** — "Uniqueness of the Fisher–Rao metric on the space of smooth densities," *Bull. London Math. Soc.* 48: 499–506. Non-parametric extension of Cencov uniqueness. Establishes that the Cencov-style uniqueness is for *the Fisher metric*, not for any divergence or functional built on it. Load-bearing for the "uniqueness-of-metric ≠ uniqueness-of-functional" strike against the "geometric necessity" framing at manuscript line 1411.
- **`[Cuturi2013]`** — "Sinkhorn distances: lightspeed computation of optimal transport," NeurIPS 2013. Softmax from entropy-regularized optimal transport. Establishes that the entropy-regularized soft-assignment softmax is recovered from many starting energies and is *not* uniquely a consequence of KL-of-Gaussians. Candidate for inclusion in `external_canon_inference.md` §10 (Pitfalls) under "softmax-from-energy is not uniquely information-geometric."
- **`[PachecoSasai2024]`** — "f-divergences and their relations to functional inequalities and information geometry" ([arXiv:2402.05014](https://arxiv.org/abs/2402.05014)). Modern review of the f-divergence class. Uniqueness results within the class always carry assumption-list caveats; the manuscript's Appendix-C conditional uniqueness inherits this.
- **`[Amari2016]`** — *Information Geometry and Its Applications*, Springer AMS 194. The full chapter-and-section reference for the manuscript's "[Amari2016]" citation at line 1555: dually-flat structure on exponential families (Ch. 2), chart-dependent Fisher matrices on Gaussian families (§4.3). The manuscript cites the book without page or chapter; this is the locus.

### From `02_red_memo_geometer.md`

- **`[MarsdenRatiu2002]`** — *Introduction to Mechanics and Symmetry*, 2nd ed., Springer TAM 17. Symplectic-mechanics treatment; reduction theorem for systems with continuous symmetry; moment map. Benchmark for what a "Newtonian reading" of an information-geometric variational principle would have to display — none of which the manuscript's §1.19 exhibits.
- **`[Bloch2003]`** — *Nonholonomic Mechanics and Control*, Springer Interdisciplinary Applied Math 24. Non-conservative and dissipative mechanical systems. The natural literature for the manuscript's asymmetric-attention "Lyapunov / dissipative" fallback at line 1994; the manuscript does not cite this and instead gestures at Arnold's conservative-Hamiltonian framework.
- **`[GoldsteinPooleSafko2002]`** — *Classical Mechanics*, 3rd ed., Addison-Wesley, Ch. 6. Small oscillations; independent confirmation of Arnold §22–25 on the operational independence requirement for inertia and potential Hessian.

## Deduplicated cross-cutting references (red opening)

The following appear in multiple red memos and carry the load-bearing strikes:

- `[Arnold1989]` (philosophy-of-science + geometer) — mass-analogy "definitional consequence" strike.
- `[DonnellyFreidel2016]`, `[BartlettRudolphSpekkens2007]`, `[Vanrietvelde2020]` (philosophy-of-science + gauge-theorist) — edge-mode + QRF "language without machinery" strike.
- `[Sønderby2016]`, `[RanganathTranBlei2016]` (philosophy-of-science + variational) — cross-scale shadow "rigid-link $\sigma^2\to 0$ limit not endorsed by standard hierarchical-VI precedent" strike.
- `[Vaswani2017]` (gauge-theorist + geometer) — "asymmetric attention breaks the mass-analogy scope" strike.
- `[Cencov1982]`, `[BauerBruverisMichor2016]`, `[Cuturi2013]` (info-geometer + philosophy-of-science) — "uniqueness-of-metric ≠ uniqueness-of-functional" strike against the "geometric necessity" framing.

## Phase 3 — blue panel harvest

### From `03_blue_memo_info-geometer.md`

- **`[Csiszár-Shields 2004]`** — "Information Theory and Statistics: A Tutorial," *Foundations and Trends in Communications and Information Theory* 1(4): 417–528. Canonical modern tutorial on the f-divergence class and the *shape* of uniqueness theorems within it: uniqueness within a divergence class under assumption-list constraints (closure under linear coupling, exponential-family preservation, dual-interpretation consistency). This is the canonical lineage for the manuscript's conditional uniqueness theorem at Appendix~\ref{app:conditional_uniqueness}, against which the body-text statement at line 1252 should be measured. Candidate for inclusion in `external_canon_inference.md` §4 alongside `[Liese-Vajda 1987]`.
- Append candidate for `external_canon_inference.md` §10 (Pitfalls): "Softmax-from-energy is recoverable from multiple energy functionals: KL-of-Gaussians under f-divergence assumptions (Csiszár-Liese-Vajda-class conditional uniqueness); entropy-regularized OT (Cuturi 2013); maximum-entropy with linear-moment constraints (Jaynes). Uniqueness claims must specify the closure class."

### From `03_blue_memo_gauge-theorist.md`

- **`[Hall 2015]`** — *Lie Groups, Lie Algebras, and Representations: An Elementary Introduction*, 2nd ed., Springer GTM 222, §3.3. The block-diagonal Lie-subgroup identification $\mathrm{GL}(d_{\text{head}})^H \subset \mathrm{GL}(d_{\text{model}})$ is a one-line elementary consequence of the standard block-diagonal subgroup construction. Canonical reference for treating the multi-head extension as an "extension after the core reduction" rather than as a load-bearing derivational step requiring delegation to the companion. Already implicit in `external_canon_math.md` Lie-theory background. Append candidate as explicit §3.3 reference for block-diagonal subgroups.

### From `03_blue_memo_philosophy-of-science.md`, `03_blue_memo_geometer.md`, `03_blue_memo_variational.md`

No new canon. Citations are to `[Popper 1959]` (already in red harvest), `[Arnold 1989]` (already in red harvest), `[Marsden-Ratiu 2002]` (already in red harvest), `[Sønderby 2016]` (already in blue Phase 2 harvest), `[Wilson 1974]` (already in blue Phase 2 harvest), `[Bishop 2006]` and `[Kingma-Welling 2014]` (both already in `external_canon_inference.md`).

## Phase 3 — empirical findings (per-citation reclassification)

For the rebuttal-round resolution of Strike 2: the 14 in-section citations of `Dennis2025trans` in §Theory of `Participatory_it_from_bit.tex` classify by load-bearing criterion as **5 strict delegations** (lines 1209, 1607, 1615, 1818, 1875) and **9 provenance / adopts-form / see-also** references (lines 636, 943, 1042, 1294, 1352, 1365, 1623, 1676, 1702). Verbatim phrase classification at each line is documented in `03_blue_rebuttal.md` and `03_blue_memo_variational.md`. The strict-load-bearing count of 5 is below red's falsification-condition (ii) threshold of ≥10.

## Phase 3 — null finding

- WebSearch for a peer-reviewed precedent in NeurIPS / ICML / JMLR / *Information Geometry* / *J. Stat. Mech.* matching red's falsification condition (ii) literally (≥10 "follows the development in [companion]" / "adapted from [companion]" references at load-bearing reduction steps + literal `\textbf{TODO}` token in theory section) returned no match. Blue cannot supply such a precedent. Reported candidly in the rebuttal; the operative question becomes whether the threshold is community-recognized or red-constructed, not whether the threshold is met.

## Phase 3 — red panel harvest

### From `03_red_memo_philosophy-of-science.md`, `03_red_memo_variational.md`, `03_red_memo_gauge-theorist.md`, `03_red_memo_info-geometer.md`, `03_red_memo_geometer.md`

No new external-canon citations beyond what the Phase-2 red harvest recorded. The Phase-3 red strikes deploy citations already in `01b_extended_evidence.md` (Phase 2 red harvest) or in `external_bibliography.md` / `external_canon_*.md`: `[Popper1959]`, `[Arnold1989]` Ch. 5 §22–25, `[GoldsteinPooleSafko2002 Ch. 6]`, `[Sønderby2016]` (arXiv:1602.02282), `[RanganathTranBlei2016]` (arXiv:1511.02386), `[Friston2017Graphical]`, `[ParrPezzuloFriston2022 Ch. 9]`, `[DonnellyFreidel2016]`, `[Witten2018]` (arXiv:1803.04993), `[CarrozzaHoehn2022]` (arXiv:2109.06184), `[Wilson1974]`, `[BauerBruverisMichor2016]`, `[Cuturi2013]`, `[Vaswani2017 §3.2.2]`, `[Milgrom Segal 2002]`, `[Nakahara2003 §10.3]`.

### Phase 3 — primary manuscript evidence introduced by red

The Phase 3 red rebuttal `03_red_rebuttal.md` introduces three load-bearing manuscript-line cross-references not deployed in the Phase 2 red opening:

- **Manuscript line 1458** (inside §Theory at sec:env_agents): "$\mathrm{KL}(q_i \| \delta(c - c_k)) = +\infty$ for any non-degenerate $q_i$" — the manuscript's own statement that a Dirac measure yields an infinite KL.
- **Manuscript line 552** (sec:cross_scale_shadows): the cross-scale shadow's rigid-link $\sigma^2 \to 0$ limit of the Gibbs cross-scale factor $\chi \propto \exp[-\|k_i - \Omega k_{\pi(i)}\|^2/(2\sigma^2)]$ — which converges in distribution to $\delta(k_i - \Omega k_{\pi(i)})$ in the cited limit, producing the line-1458 singularity.
- **Manuscript line 2064** (sec:velocity_quadratic): "Under this identification the harmonic-oscillator scaling $\omega^2 \propto k/m$ is a definitional consequence of the postulate rather than an independent dynamical scaling: when $k$ and $m$ are both equal to $M_{\mu\mu}$ by construction, $\omega^2$ reduces to a per-direction unit relation and the analogy is structural, not empirical."

These three manuscript-line cross-references are cited as the location of statements being attacked (the manuscript itself supplies the load-bearing admissions), not as authority for canonical form.

## Phase 3 — appendix read (blue, primary manuscript evidence)

The Phase 3 blue rebuttal directly inspected the appendix referenced from §Theory line 1252:

- **Manuscript lines 4258–4407** (appendix `app:conditional_uniqueness`). The appendix is **titled "Conditional Representation Theorem for the Forward KL Divergence via Variational Duality"** (line 4258). Line 4261 opens: "The result is a representation theorem rather than an unconditional uniqueness theorem." Line 4268: "We are not claiming that forward KL is uniquely required by 'well-behaved coordination' in any broader sense." The three structural assumptions are displayed at line 4272: (i) locality / standard f-divergence form with second-argument measure; (ii) linear coupling; (iii) exponential-family closure of the minimizer. The theorem statement at line 4351 (Theorem `thm:uniqueness_app`) requires an additional real-analyticity hypothesis on the generator $f$ (line 4357) used in Step 3 of the proof to extend pointwise identification of $f'(t)$ to all of $(0, \infty)$ by analytic continuation. The richness lemma at line 4409 carries an acknowledged "global-normalizer caveat." The body-text statement at line 1252 enumerates "assumptions (i)–(iii)" without flagging the real-analyticity hypothesis as a fourth condition, producing a mild assumption-count discrepancy between body and appendix.

- **Manuscript line 4261** (appendix opening): "This appendix develops, in the present notation, the conditional representation theorem from the supplementary treatment of the gauge-theoretic transformer companion paper~\cite{Dennis2025trans}." This is a 15th `Dennis2025trans` citation in the manuscript, located in the appendix rather than in §Theory's 180–2070 range. It does not affect the in-section count of 14 that red's Strike 2 enumerated, but it does mean the appendix-conditional uniqueness theorem itself is sourced to the companion paper's supplementary treatment.

These appendix findings strengthen rather than weaken the body-text conditional-representation reading at line 1252: the appendix's own scope-self-labeling ("representation theorem rather than... uniqueness theorem") preempts the broader interpretive overreach that Figure 1's caption commits. The (i)–(iii) vs (i)–(iv) assumption-count discrepancy is a one-sentence body-text revision.
