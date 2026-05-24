# Verdict — pifb-dimensional-sectors (binding, chief reconciliation)

## First-pass verdicts

| Judge | Outcome | Decisive evidence |
|-------|---------|-------------------|
| canon-strict | RED_WINS | [Cencov1972] + [AmariNagaoka2000 Ch.2]: Fisher invariance is invariance of the metric *tensor*, not of pullback eigenvalue *magnitudes*; under base reparameterization $\tilde G = J^{-\top} G J^{-1}$, so absolute thresholds $\lambda_a>\Lambda_{\rm obs}$ at :3026/:3034/:3042 name a chart, not intrinsic structure. Blue conceded verbatim. |
| code-truth   | N/A | Theory + formal-math mode; no code judge dispatched per `00_claim.md`. |
| scope        | REMAND | The claim's central proposition "well-defined by eigenvalue magnitude" carries two non-equivalent parses (weak set-partition-existence vs. strong gap-stable, chart-invariant structure) whose truth values diverge; the sentence is two claims fused, and the debate split exactly along that seam. |

## Reconciliation rule applied

Rule 2 (scope override for REMAND on equivocation) — the scope judge declared REMAND because the claim packs multiple propositions joined by semicolons, with the load-bearing one ("well-defined by eigenvalue magnitude") admitting two opposite-truth-value parses; Rule 2 fires before Rule 3/Rule 4 can run, so the binding outcome is REMAND and the scope judge's sub-claims become the spawn list.

## Decisive evidence (binding)

[Cencov1972] + [AmariNagaoka2000 Ch.2] — both verified entries in `external_bibliography.md`. Fisher invariance under sufficient statistics is invariance of the metric *tensor*, not of the numerical eigenvalue *magnitudes* of a pullback under a base chart change. Under $c \mapsto \psi(c)$ with Jacobian $J$, the induced metric transforms as $\tilde G = J^{-\top} G J^{-1}$, so eigenvalues rescale by the squared singular values of $J$; only rank and eigenvalue ratios / relative gaps are reparameterization-invariant. The sectors at :3026/:3034/:3042 are defined by absolute thresholds on raw magnitudes ($\lambda_a > \Lambda_{\rm obs}$, etc.), and the :3061–3062 hierarchy is stated in absolute magnitudes. An absolute-magnitude threshold therefore names a chart, not an intrinsic geometric direction. Blue conceded this ("Red is right on this"; "the section should be read as defining sectors only up to a choice of base chart"). The granted bookkeeping baseline rests on [Lee2013 Ch.11/13] (pullback rank $\le \dim\mathcal{C}$; PSD pullback stays PSD) — external canon, accepted by both sides. Corroborating manuscript-internal finding (manuscript-as-object, not authority): :3057 states the hierarchy with the declarative verb "satisfy," while :3190 lists that same hierarchy under "What is postulated" ("None of (i)–(iv) is derived").

## Outcome (binding)

REMAND

## Reasoning

Rule 2 fired: the claim is a single compound sentence packing three independent propositions, and its central one — the spectral decomposition "into observable / subthreshold / internal sectors by eigenvalue magnitude is well-defined" — has two non-equivalent readings (weak: a partition exists for any threshold pair and reconstructs $G_i$; strong: the named sectors are reparameterization-invariant, gap-stable features of the geometry). Red carried the strong reading and Blue carried the weak reading, so neither carries the claim *as written*; this is the canonical equivocation signature scope is mandated to catch, and Rule 2 short-circuits both the canon-strict RED_WINS (which holds only under the strong parse) and the theory-mode Rule 4 tiebreak that would otherwise route to canon-strict. Canon-strict is not overruled on its domain — its RED_WINS on the strong reading stands as written and supplies the binding external canon ([Cencov1972; AmariNagaoka2000 Ch.2] on the chart-dependence of pullback eigenvalue magnitudes); it is subordinated to the higher-priority scope rule because a soundness verdict cannot adjudicate a claim whose predicate is ambiguous, nor an in-span/out-of-span manuscript inconsistency (:3057 "satisfy" vs. :3190 "postulated"). No BLUE_WINS is available, and would in any case be malformed if it rested on manuscript authority: the surviving positive content is granted on external canon ([Lee2013 Ch.11/13], [AmariNagaoka2000 Ch.2]), not on the manuscript's say-so.

## Action

Spawn two sub-debates (adopted verbatim from the scope judge's decomposition):

- **Sub-claim A** (expected to resolve fast — both sides already concede the answer is no): The named "observable / subthreshold / internal" sectors are reparameterization-invariant features of the induced geometry, not labels that depend on a choice of base chart and the two free thresholds $\Lambda_{\rm obs}, \Lambda_{\rm subthresh}$. Resolution path: under $\tilde G = J^{-\top} G J^{-1}$ the magnitudes rescale, so membership is chart-dependent; only rank and eigenvalue ratios are invariant [Cencov1972; AmariNagaoka2000 Ch.2].

- **Sub-claim B** (the live manuscript-fix question): The in-span text at :3055–3062 is consistent with the :3188–3192 "what is derived vs. what is postulated" box — i.e., the size hierarchy $|\mathcal{D}_{\rm obs}| \ll |\mathcal{D}_{\rm subthresh}| \ll |\mathcal{D}_{\rm internal}|$ and the magnitude hierarchy $\lambda_{\rm obs} \gg \lambda_{\rm subthresh} \gg \lambda_{\rm internal} \approx 0$ are presented as posits, not as properties the spectrum "satisfies." Resolution path: :3057 uses the declarative "satisfy" with no posit marker while :3190 lists the same hierarchy under "What is postulated"; the :3021 "$\dim(\mathcal{C})$ large enough" gate governs sector *existence*, not the *ordering*.

Manuscript fixes owed when the sub-debates resolve (all three follow from the binding evidence and Debate 2's inherited signature action):

1. Recast the sector definitions at :3026/:3034/:3042 and the :3061–3062 hierarchy in reparameterization-invariant quantities — eigenvalue *ratios* or *relative gaps* rather than absolute thresholds on raw magnitudes — OR explicitly mark the partition as defined only relative to a chosen base chart [Cencov1972; AmariNagaoka2000 Ch.2].
2. Change the :3055–3062 framing from the unconditional "The dimensional and eigenvalue hierarchies satisfy" to "are assumed to satisfy" / "we postulate," matching the manuscript's own :3190 ("What is postulated") and eliminating the in-span/out-of-span inconsistency.
3. Conditionalize the :3029 "(1+3) Lorentzian decomposition" per Debate 2's signature action: the PSD spectrum supplies a 4-dimensional subspace; the Lorentzian sign is imported from the (REMANDed) signature postulates and is not intrinsic to the decomposition [Nakahara2003 §7.1–7.2; Sylvester].

Survives unchanged (granted on external canon, not on manuscript authority; needs no sub-debate):

- Base-vs-fiber bookkeeping at :3019, :3045, :3073 — exact pullback algebra, $G_i(c)=\sigma_i^* g_{\mathcal{B}}$ is an $n\times n$ tensor on $T_c\mathcal{C}$ with at most $n=\dim\mathcal{C}$ eigenvalues [Lee2013 Ch.11/13].
- Arithmetic: $\dim(\mathcal{B}) = K + K(K+1)/2 = K(K+3)/2 = 296{,}064$ at $K=768$ [AmariNagaoka2000 Ch.2].
- "(Speculative)" header and disclaimers at :3067–3071 ("no quantitative predictions ... that could be tested experimentally") — honest labels.
- No downstream leakage of the (1+3) supposition: :3174, :3188–3192, :2858, :3598–3600 all re-hedge (verified by grep; Red withdrew the leakage charge).
