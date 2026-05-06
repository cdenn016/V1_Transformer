# Round-2 Verifier Report — Issues 6, 7, 8

**Manuscript:** `Participatory_it_from_bit.tex` (3738 lines, fix/holonomy-numerics-2026-05-05)
**Reports verified:** `06_freeenergy_entropy.md`, `07_curvature_consistency.md`, `08_gauge_redundancy.md`
**Method:** Independent re-derivation; verbatim quote checks against the .tex source.

## Top-line synthesis

All three investigators are essentially correct in their primary verdicts. Issue 1 ("free-energy entropy term") is genuinely refuted as a bug — Section 3.4 is present in the manuscript verbatim and contains exactly the disambiguation the report claims it does. Issue 2 ("curvature consistency") is genuinely confirmed: under the cocycle parameterization adopted by the working framework `F_{mu nu} = 0` is an identity (Maurer-Cartan), and approximately a dozen passages still write or rely on `F != 0` or on Yang-Mills kinetic structures that are vacuous under cocycle. The single correctly-resolved passage is the vanishing-holonomy theorem at lines 765–790, which is consistent. Issue 3 ("gauge redundancy vs frame-as-state") is genuinely confirmed at the sentence level (line 235 vs line 2747 are mutually inconsistent as written), with the manuscript already partially resolving it via the disclosure at 1546 and the Haar-averaging consensus metric at 1689–1715. Issues 2 and 3 are coupled: the gauge-invariant content the manuscript points to (`tr(F F)`) is identically zero under cocycle, so Issue 3's resolution cannot lean on `F` invariants and must instead lean on Haar averaging or on a Regime-II curved-bundle extension.

## Issue 1 verification — free-energy entropy term

**Verdict: refuted as bug, confirmed as presentational hazard. Investigator is correct.**

### Verbatim verification of Section 3.4 (lines 867–905)

Line 867 begins `\subsubsection{The Envelope Theorem and the Reduced Free Energy}` with `\label{sec:envelope_theorem}`. Lines 870–871 read verbatim:

> "The boxed functional~\eqref{eq:free_energy_functional_final} is the *full* (un-minimized) free energy: $\beta_{ij}$ and $\gamma_{ij}$ appear as variational attention parameters that are optimized jointly with the belief and model fields."

Lines 875–882 contain the boxed `eq:free_energy_reduced` with the `-tau sum_i log Z_i` term explicitly. Line 884 reads verbatim:

> "The log-partition function $-\tau\log Z_i$ is the result of substituting $\beta_{ij}^*$ into both the energy term $\sum_j \beta_{ij} E_{ij}$ and the entropy term $\tau\sum_j \beta_{ij}\log(\beta_{ij}/\pi_j)$ of $\mathcal{F}_{\text{align}}$. The full (un-minimized) alignment energy includes the attention entropy/prior term, which is essential: it is this term that makes the $\beta$-optimization yield the softmax rather than an arg-min."

The "Autograd versus reduced-free-energy gradients" paragraph at lines 897–902 reads verbatim:

> "This computes the gradient of the attention-weighted energy $\langle E \rangle_\beta = \sum_j \beta_{ij}^* E_{ij}$, which differs from the reduced free energy $-\tau\log Z_i$ by the entropy term $\tau\sum_j \beta_{ij}^* \log(\beta_{ij}^*/\pi_j)$. These two objectives share the same critical points but have different gradient fields away from equilibrium."

The disambiguation the investigator relies on is genuinely present.

### Algebra independently rechecked

For `F_align = sum_j beta_j (E_j + tau log beta_j - tau log pi_j)` with `beta*_j = pi_j exp(-E_j/tau) / Z`, `Z = sum_k pi_k exp(-E_k/tau)`:

- `F_align(beta*) = -tau log Z`. Substitute: `sum_j beta*_j (E_j + tau log beta*_j - tau log pi_j) = sum_j beta*_j (E_j - E_j - tau log Z) = -tau log Z` (using `log beta*_j = -E_j/tau + log pi_j - log Z`). Confirmed.
- `<E>_beta* = sum_j beta*_j E_j`. Then `F_align(beta*) - <E>_beta* = tau sum_j beta*_j (log beta*_j - log pi_j) = tau KL(beta* || pi)`. So `<E>_beta* = -tau log Z - tau KL(beta* || pi)`. Confirmed.

The investigator's two identities are algebraically correct.

### Verdict on report 6

The "refuted as bug" verdict is correct. The defect is strictly presentational: a reader who stops at the boxed Eq. 39 sees `sum beta KL(...)` (the energy term, no entropy) and may infer `F = <E>_beta*`. Section 3.4 corrects this within seven lines of the box, but the box itself is not self-flagging. The recommended editorial fix (option C — small clauses on Eqs. 32 and 39 cross-referencing `eq:free_energy_reduced` and Section 3.4) is the right disposition. Severity: **editorial / low**.

## Issue 2 verification — curvature consistency

**Verdict: confirmed. Investigator is correct.**

### F = 0 re-derivation (Maurer-Cartan)

For `A_mu := U^{-1} partial_mu U` with U single-valued:

Using `partial_mu(U^{-1}) = -U^{-1} (partial_mu U) U^{-1}`,

`partial_mu A_nu = -U^{-1}(partial_mu U) U^{-1} (partial_nu U) + U^{-1} partial_mu partial_nu U = -A_mu A_nu + U^{-1} partial_mu partial_nu U`.

Antisymmetrizing,

`partial_mu A_nu - partial_nu A_mu = -A_mu A_nu + A_nu A_mu + U^{-1}(partial_mu partial_nu U - partial_nu partial_mu U) = -[A_mu, A_nu]`,

where the second-derivatives commutator vanishes by Schwarz (U is C^2). Therefore

`F_{mu nu} = (partial_mu A_nu - partial_nu A_mu) + [A_mu, A_nu] = -[A_mu, A_nu] + [A_mu, A_nu] = 0`.

Identity, no smoothness assumption beyond C^2 and single-valuedness. The investigator's derivation is correct.

### Census of cited "F != 0" passages

I read each cited line range. All quotes verified against the .tex source:

- **Line 434** — connection form definition. Quote matches.
- **Line 438** — F_{mu nu} field-strength definition. Quote matches; this is benign as a definition.
- **Lines 443–445** — "When F != 0 (curved connection), beliefs accumulate holonomy"; "non-zero field strength indicates the agent's internal coordinate system is 'twisted'." Quotes match. Both implicitly require F to be non-zero to have content; under cocycle they describe an empty regime.
- **Lines 467–469** — "Non-zero F_{mu nu} creates path-dependent information transport... Here it measures 'information field strength' - how strongly the cognitive frame varies spatially." Quote matches. The conflation of "rate of frame variation" (`||grad phi||^2`, real and nonzero) with "curvature" (`F`, identically zero) is a real category error in the manuscript.
- **Line 482** — "...sacrificed includes path-dependent parallel transport (holonomy effects)... and strong gauge field curvature F_{mu nu}." Quote matches. As the investigator notes, this misrepresents structurally-zero as merely "small."
- **Line 758** — connection form on dim>=1 base. Quote matches.
- **Lines 765–790** — vanishing-holonomy theorem. Quote matches; this is the manuscript's correct treatment.
- **Line 865** — "Yang-Mills curvature penalties int tr(F_{mu nu} F^{mu nu}) sqrt(g) dc." Quote matches. Vacuous under cocycle.
- **Lines 1525, 1540, 1546** — bundle pullback metric uses tr(A_mu A_nu); explicit gauge-invariance disclosure follows. Quotes match. The disclosure is correct standard Yang-Mills bookkeeping (tr(A A) is gauge-noninvariant; tr(F F) is the invariant) — but tr(F F) is zero under cocycle, so the disclosed "genuinely gauge-invariant content" is empty.
- **Lines 1636, 1657** — Lorentzian-signature worked example. `G_{mu nu} = tr(A_mu A_nu)` followed by complex-phi postulate. Quotes match. Critical observation: the *manuscript itself* admits at line 1620 that "the construction is an existence demonstration, not a derivation" and at line 1641 that "the real-part projection is an additional choice" and "We flag this as a derivation gap." This is exculpatory but does not address the gauge-noninvariance of the constructed metric component.
- **Lines 2626–2636** — gauge curvature linguistic conjecture. Quote matches. The formula at 2628 is `F_{mu nu} = partial_mu phi_nu - partial_nu phi_mu + [phi_mu, phi_nu]` — written in terms of phi rather than A, which is **a separate error** beyond the cocycle issue: phi is a Lie-algebra element, not the connection one-form, and there is no consistent way to read this expression as the field strength of any connection. Even granting it as the curvature of the linear-order connection `A_mu approx partial_mu phi`, the conjecture minimizes a quantity that is identically zero under the implemented architecture.

### Independent gauge-transformation calculation

For Issue 2/3 cross-coupling: under `U -> U g(c)` with g non-constant,

`A_mu -> (Ug)^{-1} partial_mu (Ug) = g^{-1} U^{-1}(partial_mu U) g + g^{-1} U^{-1} U partial_mu g = g^{-1} A_mu g + g^{-1} partial_mu g`.

Then

`tr(A_mu A_nu) -> tr[(g^{-1} A_mu g + g^{-1} partial_mu g)(g^{-1} A_nu g + g^{-1} partial_nu g)]`.

Expand and use cyclicity of trace:
- `tr(g^{-1} A_mu g g^{-1} A_nu g) = tr(A_mu A_nu)`. (Survives.)
- `tr(g^{-1} A_mu g g^{-1} partial_nu g) = tr(A_mu (partial_nu g) g^{-1})`. Nonzero unless `partial_nu g = 0`.
- `tr(g^{-1} partial_mu g g^{-1} A_nu g) = tr((partial_mu g) g^{-1} A_nu)`. Nonzero similarly.
- `tr(g^{-1} partial_mu g g^{-1} partial_nu g)`. Pure Maurer-Cartan term.

`tr(A_mu A_nu)` is gauge-invariant only for *constant* g (rigid gauge). The manuscript reproduces the same conclusion at line 1546.

### Miscount check

The investigator's table is honest. Two minor observations:
1. Item #4 (lines 467–469) does in fact contain the conflation between `||grad phi||^2` and curvature, as I read the passage. Investigator captures this.
2. Item #14 (line 3261, "BCH when gauge field strengths are small") I did not directly verify but the investigator flags it as "pragmatic; consistent if reread." That is a defensible classification.
3. The investigator did not separately call out the **typo** at line 2628 where `F_{mu nu}` is written using `phi` arguments rather than `A` arguments. This is a separate small bug worth flagging in the recommended fix list.

### Verdict on report 7

Confirmed. Investigator's recommended Option C (introduce two named regimes — Regime I = flat-bundle, cocycle, F=0 identically; Regime II = curved-bundle extension, F can be nonzero) is the right disposition. It preserves the mathematical content while making dependency structure transparent and matches what the implementation actually does. Severity: **high**, because it touches multiple load-bearing claims (Lorentzian-signature mechanism, gauge-curvature linguistic conjecture, Yang-Mills kinetic block of the effective metric, four-curvatures synthesis). The investigator's severity verdict stands.

## Issue 3 verification — gauge redundancy vs frame-as-state

**Verdict: partially confirmed. Investigator is correct.**

### Quote-verification

- **Line 235**: "These frames are arbitrary choices with no physical content, yet they're necessary for representing probabilistic information." Verified verbatim.
- **Line 2747**: "There is, in this reading, something it is like to occupy gauge frame $\phi_i$ because that frame induces a specific metric $G_i$ defining the agent's phenomenal space." Verified verbatim.

The two sentences are in plain logical tension as written. Either frames have "no physical content" (Interp 1, redundancy view) or there is "something it is like to occupy" them (Interp 2, frame-as-state view). The qualia indeterminacy paragraph at 2734 attempts mitigation via "multi-scale constituent structure," but this does not retract line 235.

### Haar-averaging resolution at 1689–1715 — verified

Line 1689 contains the textbook "redundancy" framing verbatim:

> "Gauge freedom represents redundancy in description, not physical degrees of freedom, therefore the observable geometry must not depend on these arbitrary choices."

Lines 1693–1707 then construct the Haar-averaged single-agent metric `<G_i>(c) = int_G dg G_i(c; U_i -> U_i g)` and the consensus metric `bar G^{consensus}(c) = sum_i w_i(c) <G_i>(c)`. The construction is gauge-invariant by orbit-averaging — for compact G this is standard and correct. For non-compact G the manuscript explicitly flags that "the integral requires a regulator that we do not specify here" and labels the Lorentzian outcome "a plausibility argument... rather than an established result."

The investigator's claim — that the manuscript already partially resolves the tension via Haar averaging — is correct. The unresolved part is that the qualia/phenomenology section (2732–2752) and the per-agent pullback `G_i` at 1546 are deployed in the frame-as-state sense, where gauge-equivalent frames produce *distinguishable* phenomenologies. That use is genuinely incompatible with line 235.

### Gauge transformation of tr(A A) — verified

See Issue 2 calculation above. Confirmed gauge-noninvariant under non-constant g. The manuscript reproduces this correctly at line 1546.

### Verdict on report 8

Confirmed. The recommended fix (insert a "two roles for the gauge frame" paragraph after line 244, soften line 235, add one disambiguating sentence near line 2747) is appropriate and minimal. The investigator's severity assessment of **medium** is defensible. I would lean slightly higher — toward **medium-high** — because line 235 is in the third subsection of Section 2 and sets the philosophical tone for everything downstream, so a referee will hit the contradiction within five pages of opening the manuscript.

## Cross-issue coupling (Issues 2 and 3)

The two issues are connected through `tr(A_mu A_nu)` and `tr(F_{mu nu} F^{mu nu})`.

The manuscript's gauge-invariance disclosure at line 1546 says: "The genuinely gauge-invariant content lives in the field strength F_{mu nu}... and yields the gauge-invariant Yang-Mills action density tr(F_{mu nu} F^{mu nu}) upon integration." This is offered as the reason that the gauge-noninvariance of `tr(A A)` does not undermine the framework — the "real" gauge-invariant content is in F, A is just a non-invariant proxy.

Under the cocycle parameterization, **F = 0 identically**, hence `tr(F F) = 0` identically. The disclosure's resolution evaporates: there is no gauge-invariant content in F to point at, because the only gauge-invariant content of `A = U^{-1} dU` for single-valued U is trivially zero. So the manuscript's implicit defense of the per-agent metric — "we know it's gauge-noninvariant, but the invariant piece tr(F F) carries the real content" — is mathematically empty under the cocycle constraint.

This means **Issue 3's resolution must lean on Haar averaging or on a separate Regime-II extension; it cannot lean on field-strength invariants**, because Issue 2 establishes those invariants are zero. The recommended fix for Issue 3 should explicitly route gauge-invariant ontology through the consensus metric (Section 1689–1715) rather than through `tr(F F)`, and the recommended fix for Issue 2 should remove or quarantine the language at line 1546 that points readers toward `tr(F F)` as the invariant content.

A coordinated edit at lines 1525–1554 should: (a) state that the per-agent pullback `G_i` is a Role-B (frame-as-state) object whose `phi_i`-dependence is intentional ontological content, not an error; (b) state that the only gauge-invariant geometry on `C` in the implemented (Regime-I) framework is the Haar-averaged consensus metric of Section 1689–1715; (c) flag that the Yang-Mills invariant `tr(F F)` is identically zero under cocycle, so the "field strength carries the gauge-invariant content" sentence in the disclosure should be deferred to the curved-bundle Regime-II extension or rephrased to point at the consensus metric instead.

## Investigator quality assessment

- **Investigator 6 (free-energy entropy):** rigorous. Quote-checks accurate, algebra correct, classification (option C — additive editorial fix) appropriate. Did not over-claim. The analysis correctly identifies that the apparent contradiction is dissolved within Section 3.4 and that the residual hazard is a layout problem, not a math problem. No corrections needed.

- **Investigator 7 (curvature consistency):** rigorous and arguably the most thorough of the three. Quote-checks accurate, derivation of F = 0 correct, census of 12+ affected passages substantiated. One small miss: did not flag the typo at line 2628 (where `F_{mu nu}` is written with `phi` arguments rather than `A` arguments). Otherwise complete. The classification of severity as "high" is correct given the load-bearing nature of the affected claims.

- **Investigator 8 (gauge redundancy):** rigorous on the math (gauge transformation calculation correct) and on quote-checks. Slight under-claim: classified severity as medium, but the line-235 violation appears so early in the manuscript that a referee will encounter the contradiction before reaching any of the resolutions. Reasonable to bump to medium-high. Investigator did not flag the cross-coupling with Issue 2 (the implicit reliance on `tr(F F)` as invariant is undermined by Issue 2), but each investigator was bounded by their own scope and the cross-coupling is genuinely a synthesis-level observation.

## Consolidated severity and revision priority

| Issue | Original severity | Verifier severity | Priority |
|---|---|---|---|
| 7 (curvature consistency, F=0 vs F!=0) | High | **High** | 1 |
| 8 (gauge redundancy vs frame-as-state) | Medium | **Medium-high** | 2 |
| 6 (free-energy entropy term presentation) | Editorial | **Editorial / low** | 3 |

### Consolidated revision plan, ranked

1. **Issue 7 (high priority).** Adopt Option C (Regime I / Regime II naming). Specific edits per the investigator's list at items 1–10 of report 7. Critical that the edit at lines 1540–1546 be coordinated with Issue 3's edit (see point 2). Additional small edit: correct the line 2628 formula for F to use A rather than phi as arguments, or relabel as the linear-order approximation. Effort: section restructuring, ~half-day.

2. **Issue 8 (medium-high priority).** Insert the "Two roles for the gauge frame" paragraph after line 244, soften line 235, add the one-sentence disambiguator near line 2747. Coordinate with Issue 7: replace the line-1546 appeal to `tr(F F)` (which is zero under cocycle) with an appeal to the Haar-averaged consensus metric of Section 1689–1715. Effort: three short LaTeX patches, ~one hour.

3. **Issue 6 (low priority).** Add the small clauses at Eqs. 32 and 39 cross-referencing `eq:free_energy_reduced` and Section 3.4 (option C from report 6), or alternatively re-box Eq. 39 in terms of `-tau log Z_i` (option A / "optional stronger fix"). Effort: two short LaTeX patches, ~thirty minutes.

The order matters: Issue 7's resolution constrains Issue 8's resolution (because the "tr(F F) is the gauge-invariant content" framing in 1546 must be retired), so 7 should be drafted first and 8 immediately after to keep the disclosure paragraph internally consistent. Issue 6 is independent and can be done at any time.

No numerical results are corrupted. No retraction is required for any of the three issues; all are resolvable by editorial revision.
