# Extended Evidence — harvested canon (Phase 2)

Canon discovered by the blue panel beyond `01_evidence.md`. Concatenated for the judges.

## Blue side (Phase 2 opening)

- **Local gauge group is not locally compact.** `Map(C, G)`, the group of gauge-group-valued
  maps over which an honest local-`g(c)` orbit average must integrate, is infinite-dimensional
  and not locally compact, so it does not in general possess a Haar measure at all. The
  Faddeev–Popov procedure implicitly assumes one; absent gauge-fixing the construction is
  ill-defined. Source: Faddeev–Popov literature and "A Note on Functional Integral over the
  Local Gauge Group" (arXiv:hep-th/0103160); standard treatment [PeskinSchroeder1995 Ch. 9].
  Effect on the debate: the manuscript's "requires a gauge fixing or a regulator" is
  conservative, not overclaiming — the measure may not exist without one.

- **Haar finiteness ⇔ compactness (confirmed against standard treatments).** A locally compact
  Hausdorff group has a finite (normalizable, probability) Haar measure if and only if it is
  compact; for non-compact groups every left-invariant Haar measure is infinite. Confirms the
  manuscript's `SO(1,3)` claim (:2977). Sources: Haar-measure theorem as stated in standard
  abstract harmonic analysis (Folland-type), Tao 254A Notes 3 on Haar measure / Peter–Weyl,
  Wolfram MathWorld "Haar Measure".

- **Čencov uniqueness as the source of the belief-fiber invariance.** The GL(K,ℝ)-invariance
  that the "gauge invariance from consensus" interpretation rests on is a property of the
  divergence itself: the Fisher metric (and the KL it generates) is the unique divergence
  invariant under sufficient statistics / invertible reparameterization [Cencov1972;
  AmariNagaoka2000 Ch. 2]. Effect: any reading in which consensus *derives* gauge invariance is
  circular, because the invariance is intrinsic to the divergence before any multi-agent story.

- **Group averaging / Reynolds operator as the legitimate orbit-average form.** For a compact
  group `K`, `P(T) = ∫_K ρ(k) T dk` projects a tensor onto the gauge-invariant subspace; this is
  the standard invariant-extraction construction the manuscript's orbit average instantiates.
  The non-compact / local case is exactly where this projector fails to be finite, which is the
  Faddeev–Popov regulator problem. Source: standard representation theory; [PeskinSchroeder1995
  Ch. 9] for the regulated non-compact analogue.

### Citation-hygiene note
`[Folland]` and `[PeskinSchroeder1995]` are named in `01_evidence.md` as canon to verify but are
not tagged in `external_bibliography.md` (which lists Peskin & Schroeder and Weinberg Vol. II
under "Coverage gaps — extend on demand"). They are cited here at source level only, with the
specific facts confirmed against the standard treatments and the web searches logged in the
Phase 2 transcript. `[Nakahara2003]`, `[Cencov1972]`, `[AmariNagaoka2000]`, `[Friston2010]` are
in the bibliography.

## Red side (Phase 2 opening)

- **Standard-Model gauge group is empirical, not derived.** The su(3)×su(2)×u(1) gauge algebra,
  the quark/lepton/Higgs quantum numbers, and the 19 free parameters of the SM are fixed by
  experiment; "a first-principles explanation for the three gauge groups could not be furnished
  so far." Effect on the debate: nothing in the consensus framework selects U(1), SU(2), SU(3),
  SO(1,3) from the uncountably many compact Lie subgroups of GL(K,ℂ); enumerating exactly the
  SM factors (:2992) borrows the SM's prestige without supplying the selecting content (anomaly
  cancellation, irrep/chirality structure, free-energy stationarity). Sources:
  "Mathematical formulation of the Standard Model" (su(3)×su(2)×u(1) experimentally determined,
  19 parameters fixed by experiment),
  https://en.wikipedia.org/wiki/Mathematical_formulation_of_the_Standard_Model ;
  [PeskinSchroeder1995] on the SM gauge structure as empirical input.

- **Cencov/Chentsov uniqueness — extended generalizations.** Beyond [Cencov1972]: Ay, Jost, Lê,
  Schwachhöfer, "Information geometry and sufficient statistics," *Probab. Theory Relat. Fields*
  (2015), arXiv:1207.6736, give the full generalization of Chentsov to infinite sample sizes —
  the Fisher metric and Amari–Chentsov tensor are uniquely characterized (up to scale) by
  invariance under sufficient statistics; Dowty (2018) for exponential families. Effect: the
  GL(K)-invariance of KL on the Gaussian belief fiber is not a coincidence the consensus story
  explains — it is forced by the choice of a Cencov-invariant objective. Source:
  https://arxiv.org/pdf/1207.6736 ; https://en.wikipedia.org/wiki/Chentsov's_theorem

- **Numerical confirmation of belief-fiber KL invariance (executed this round).** A direct
  finite-precision check: for q=N(μ_q,Σ_q), p=N(μ_p,Σ_p) on K=4 with random SPD Σ and the
  framework's frame action μ→gμ, Σ→gΣgᵀ for random g∈GL(4,ℝ), the closed-form Gaussian KL is
  unchanged to 4×10⁻¹⁵. Confirms the consensus requirement (frame-independence of shared
  structure) is auto-satisfied on the belief fiber. Command and output logged in the Phase 2
  red transcript.

- **Falsifiability / derivation-vs-redescription (philosophy of science).** Popper's demarcation:
  a thesis "compatible with all possible observations" lacks potential falsifiers and is
  tautological rather than empirical; a conclusion built into its premise is a redescription,
  not a derivation (the "survival of the fittest" tautology case). Effect: the manuscript's
  self-flag "may not be falsifiable" names a weaker defect than the actual one — the consensus
  thesis is circular (its conclusion, gauge invariance, is its premise, the Cencov-invariant
  objective). Sources: SEP "Karl Popper" https://plato.stanford.edu/entries/popper/ ;
  "Falsifiability" https://en.wikipedia.org/wiki/Falsifiability .
