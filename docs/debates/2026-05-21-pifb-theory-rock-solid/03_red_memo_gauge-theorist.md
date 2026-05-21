# Red Rebuttal Memo — gauge-theorist

Lens: gauge theory — Lie groups, representations, holonomy, equivariance, gauge-fixing, edge modes / quantum reference frames.

## One concession from blue's opening

Blue's defense item 5 correctly notes that the Goldstone caveat at line 1518 of §Theory is explicit, named, and load-bearing. The manuscript identifies the breaking as Zeeman-type explicit (external source field) and rules out the spontaneous-symmetry-breaking reading by name, matching `[Weinberg1995 Vol. II §19]` and `[Peskin Schroeder Ch. 11]`. The Goldstone discipline at line 1518 is publication-grade. Granted.

## Strongest attack on blue's core defense

Blue's defense item 5 makes a *combined-license* move: lines 567–571 separate Role A (gauge-redundant under global diagonal $U_i \to U_i g$) from Role B (physical state under the residual constant-per-agent subgroup) and cite `[DonnellyFreidel2016]` (edge modes), `[BartlettRudolphSpekkens2007]`, `[Vanrietvelde2020]` (quantum reference frames), and `[Rovelli1996]` (relational interpretation) as the "discrete analog of three well-developed mechanisms." Blue then asserts at line 23 of `02_blue_opening.md`: "This is rigorous, not ad hoc; the manuscript names the subgroup at every step."

The combined license is overclaimed in two specific ways.

(1) **Edge modes carry a symplectic apparatus the manuscript does not exhibit.** `[DonnellyFreidel2016]` *"Local subsystems in gauge theory and gravity"* (JHEP 09:102, [arXiv:1601.04744](https://arxiv.org/abs/1601.04744)) promotes would-be pure-gauge boundary data to physical degrees of freedom *via an explicit symplectic structure on the extended phase space*: the boundary modes appear as new symplectically-conjugate variables paired with the gauge transformations they break. The Witten 2018 review (`[Witten2018]` Rev. Mod. Phys. 90:045003, [arXiv:1803.04993](https://arxiv.org/abs/1803.04993)) and Carrozza-Höhn 2022 (`[CarrozzaHoehn2022]` JHEP 02:172, [arXiv:2109.06184](https://arxiv.org/abs/2109.06184)) both make this constructive content explicit. The manuscript's §Theory does not exhibit a symplectic form on the augmented $\phi_i$-space, does not construct a momentum conjugate to $\phi_i$, and does not derive the boundary action from the symplectic 2-form. What §Theory exhibits is *the language* of edge modes (Role A redundant, Role B physical) without *the machinery* (symplectic form, conjugate variables, boundary action). Per `external_canon_math.md` on what constitutes a rigorous edge-mode treatment, the combined-license cite is hand-wave-with-citation: the right paper, the wrong claim.

(2) **The residual-subgroup discipline at line 571 is broken downstream by the lattice-gauge cocycle at line 610.** The lattice gauge transformation $\Omega_{ij} \mapsto g_i \Omega_{ij} g_j^{-1}$ at manuscript line 610 is *not* the constant-per-agent residual subgroup of line 571. It is the *full* per-vertex local gauge — different $g_i, g_j$ at different vertices. The manuscript's "we attach this restriction to each occurrence below" at line 571 commits to the constant-$g$ subgroup throughout Role-B occurrences, but line 610 uses the per-vertex action without restating the constant-in-$c$ caveat. Either (a) the lattice-gauge transformation at line 610 is the full local gauge — in which case Role-B quantities like $\mathrm{tr}(A^{(i)}_\mu A^{(i)}_\nu)$ are *not* invariant under it (Maurer-Cartan derivatives appear) and the line-571 commitment is violated; or (b) the lattice-gauge transformation is implicitly restricted to constant-per-vertex elements that are *also* constant in $c$, in which case the cocycle structure is degenerate (a single $g$ shared across all vertices, equivalent to the line-571 global subgroup). The manuscript does not state which reading is in force. This is the line-571 falsification condition that blue's own falsification trigger (d) names: "A single later §Theory equation that asserts gauge-invariance without naming the residual restriction would falsify the gauge-theory defense at sub-claim 4."

## Strongest defense against blue's strongest attack

Blue does not mount a direct attack on red's opening from the gauge-theorist memo. The defense is the reinforcement above: blue's evidence item 5 is the strongest gauge-theory defense available and even it is undercut by the line-610 / line-571 discipline gap, which fires blue's own falsification trigger (d).

`[Wilson1974]` *"Confinement of quarks"* (Phys. Rev. D 10:2445) is the natural reference for the lattice-gauge treatment under per-vertex transformations and is what makes the constant-$g$ residual subgroup well-defined as a Wilson-line-type construct. The manuscript does not cite Wilson 1974, and the residual-subgroup treatment is not derived from the Wilson-line machinery — it is named.

## Newly-discovered canon

None beyond what the Phase-2 red harvest recorded. `[Witten2018]` (arXiv:1803.04993), `[CarrozzaHoehn2022]` (arXiv:2109.06184), and `[Wilson1974]` (Phys. Rev. D 10) are all in `01b_extended_evidence.md`.
