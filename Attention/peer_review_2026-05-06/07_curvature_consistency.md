# 07. Curvature Consistency Audit

## Verdict

**CONFIRMED.** The manuscript is internally inconsistent on whether the gauge connection has nontrivial curvature. The connection is parameterized as a pure-gauge Maurer-Cartan form A = U^{-1} dU with U = exp(phi), which forces F_{mu nu} = 0 identically. The manuscript correctly states this at lines 765-790 (Theorem on Vanishing Holonomy), but in at least seven other passages it appeals to nonzero F_{mu nu}, Yang-Mills kinetic terms, holonomy around closed loops, and a Yang-Mills-derived effective metric, all of which require F != 0 to have nontrivial content. The defect is structural, not cosmetic, because the entire Lorentzian-signature mechanism in Section "Worked Example: Lorentzian Signature from GL(2,C)" rests on a Yang-Mills kinetic block that the cocycle parameterization makes identically zero up to gauge.

## Direct calculation: F = 0 for A = U^{-1} dU

Let U: C -> G be smooth and single-valued, and define A_mu := U^{-1} partial_mu U valued in g.

Compute the two terms of the field strength.

**Term 1.** partial_mu A_nu = partial_mu (U^{-1} partial_nu U). Using partial_mu(U^{-1}) = -U^{-1}(partial_mu U)U^{-1},

    partial_mu A_nu = -U^{-1}(partial_mu U) U^{-1} (partial_nu U) + U^{-1} partial_mu partial_nu U
                    = -A_mu A_nu + U^{-1} partial_mu partial_nu U.

**Term 2.** Antisymmetrizing in (mu, nu),

    partial_mu A_nu - partial_nu A_mu = -A_mu A_nu + A_nu A_mu + U^{-1}(partial_mu partial_nu U - partial_nu partial_mu U)
                                     = -[A_mu, A_nu]    (the second derivatives commute since U is C^2).

**Term 3.** Adding the commutator from the Yang-Mills field strength,

    F_{mu nu} = (partial_mu A_nu - partial_nu A_mu) + [A_mu, A_nu]
             = -[A_mu, A_nu] + [A_mu, A_nu]
             = 0.

This is the standard Maurer-Cartan identity: the curvature of a pure-gauge connection vanishes identically for any single-valued group-valued function U. No assumption about the gauge group, the dimension of C, or the smoothness of phi beyond C^2 is required. The result is an identity, not an approximation.

## Quoted passages from each cited region

### Region 1: Lines 425-475 (connection forms, field strength, parallel transport)

Line 434:
> "The gauge frame phi_i induces a connection one-form on agent i's domain defined by A_mu^{(i)}(c) = U_i^{-1}(c) partial_mu U_i(c) in g where U_i(c) = exp[phi_i(c)] in G."

Line 438 (the field strength definition, before the vanishing-holonomy theorem appears):
> "The field strength (curvature two-form) measures path-dependence of parallel transport: F_{mu nu}^{(i)}(c) = partial_mu A_nu^{(i)} - partial_nu A_mu^{(i)} + [A_mu^{(i)}, A_nu^{(i)}] in g."

Line 443:
> "When F^{(i)}_{mu nu} = 0 (flat connection), parallel transporting a belief around any closed loop returns it unchanged - information transport is path-independent. When F^{(i)}_{mu nu} != 0 (curved connection), beliefs accumulate holonomy around closed loops..."

Line 445:
> "The cognitive interpretation: non-zero field strength indicates the agent's internal coordinate system is 'twisted' across the base manifold."

Lines 467-469 (in the "Four Curvatures" subsection):
> "The gauge field curvature characterizes connection geometry. The field strength F_{mu nu}^{(i)} = partial_mu A_nu^{(i)} - partial_nu A_mu^{(i)} + [A_mu^{(i)}, A_nu^{(i)}] measures curvature of the connection... Non-zero F_{mu nu} creates path-dependent information transport - moving beliefs along different paths through C yields different results. In Yang-Mills theory, F_{mu nu} is the field strength of gauge bosons (photons, gluons, weak bosons). Here it measures 'information field strength' - how strongly the cognitive frame varies spatially."

Line 482 (working-framework simplifications):
> "What is sacrificed includes path-dependent parallel transport (holonomy effects)... and strong gauge field curvature F_{mu nu}."

This last quote is partially exculpatory: it concedes that the working framework operates in a regime where F_{mu nu} is small. But "small" is not the issue — the cocycle parameterization makes F_{mu nu} **identically zero**, not "small". Calling a structurally-zero quantity "negligible" misrepresents the situation.

### Region 2: Lines 750-800 (vanishing-holonomy theorem)

Line 758:
> "A_mu^{(i)}(c) = U_i^{-1}(c) partial_mu U_i(c) in gl(K)" [Eq. eq:connection_form]

Line 765:
> "For connections derived from a single-valued gauge function phi_i(c), the connection is pure gauge and the curvature vanishes identically: F_{mu nu}^{(i)} = 0. This is the flat bundle regime."

Line 776, Theorem (Vanishing Holonomy):
> "For gauge transport of the form Omega_{ij} = g_i g_j^{-1} with vertex-local group elements g_i in G, the holonomy around any closed loop vanishes: H_{ijk} = Omega_{ij} Omega_{jk} Omega_{ki} = g_i g_j^{-1} g_j g_k^{-1} g_k g_i^{-1} = I."

Line 790:
> "Architectures with non-trivial curvature and holonomy would require promoting Omega_{ij} to independent edge variables unconstrained by the cocycle condition; such architectures would exhibit path-dependent meaning transport, where the route through intermediate agents affects the result."

This passage is the manuscript's own statement of the resolution: F != 0 is unreachable without breaking Omega_{ij} = U_i U_j^{-1}.

### Region 3: Lines 855-880 (free energy with Yang-Mills terms)

Line 865:
> "On base manifolds of dimension >= 1, the gauge frame fields additionally induce a connection one-form A_mu^{(i)}(c) = U_i^{-1} partial_mu U_i (Eq. ref{eq:connection_form}) and gauge curvature F_{mu nu}^{(i)} (Eq. ref{eq:field_strength}). Optional regularizers include gauge field smoothness lambda_phi int ||grad phi_i||^2 sqrt(g) dc, Fisher metric mass terms int tr(G_{ij}) sqrt(g) dc, and Yang-Mills curvature penalties int tr(F_{mu nu} F^{mu nu}) sqrt(g) dc."

The Yang-Mills penalty integrates a quantity that the manuscript itself proves is identically zero. The penalty has no content under the cocycle parameterization.

### Region 4: Lines 1540-1620 (effective metric from gauge connection)

Line 1540 (boxed equation):
> "G_{i,mu nu}^{(q)}(c) = tr(A_mu^{(i)} A_nu^{(i)}) + E_{q_i(c)}[(grad_mu^{(i)} log q_i)(grad_nu^{(i)} log q_i)]."

Line 1546 (gauge-invariance disclosure):
> "Under a local gauge transformation U_i -> U_i g(c), the connection A^{(i)} = U_i^{-1} dU_i acquires a Maurer-Cartan piece A^{(i)} -> g^{-1} A^{(i)} g + g^{-1} dg, and tr(A_mu A_nu) is therefore not invariant... The genuinely gauge-invariant content lives in the field strength F_{mu nu} = partial_mu A_nu - partial_nu A_mu + [A_mu, A_nu], which transforms by conjugation F -> g^{-1} F g and yields the gauge-invariant Yang-Mills action density tr(F_{mu nu} F^{mu nu}) upon integration with a base metric."

This is the deepest contradiction: the manuscript states that the only gauge-invariant content of the connection is F_{mu nu}, then immediately uses this same connection (in a non-gauge-invariant way) to construct the effective metric. Under the cocycle constraint, the gauge-invariant content is **zero** — meaning the Yang-Mills part of G_{i,mu nu}^{(q)} carries no gauge-invariant information whatsoever.

### Region 5: Lines 1636-1670 (Worked example, Lorentzian signature mechanism)

Line 1636:
> "The Yang-Mills kinetic form G_{mu nu} = tr(A_mu A_nu) evaluates to..."

Line 1657:
> "the indefinite signature arises from the gauge connection (the kinetic form tr(A_mu A_nu)), not from the fiber metric, which remains the positive-definite Fisher-Rao metric throughout."

The Lorentzian-signature mechanism rests on tr(A_mu A_nu) acquiring a sign flip from imaginary phi_tau. But tr(A_mu A_nu) for A = U^{-1} dU is gauge-variant (it transforms with cross terms involving g^{-1} dg), so this metric component is not a physically meaningful object — the Yang-Mills kinetic block of the bundle metric is a gauge artifact, not a derived geometric structure of the system.

### Region 6: Lines 2624-2640 (Gauge curvature conjecture)

Line 2626:
> "We propose a potentially falsifiable conjecture: language is a gauge theory, and linguistic evolution is driven by minimization of gauge field curvature."

Line 2628:
> "The gauge field curvature measures path-dependence of information transport. For gauge frames phi_i(x) varying over the token sequence or discourse structure, the field strength tensor is F_{mu nu} = partial_mu phi_nu - partial_nu phi_mu + [phi_mu, phi_nu]."

This is materially incorrect even by the manuscript's own definitions: F_{mu nu} is built from A = U^{-1} dU, not from phi directly. More fatally, under the manuscript's parameterization F == 0, so the conjecture "linguistic evolution minimizes gauge field curvature" minimizes a quantity that is structurally zero — the conjecture is vacuous.

## Census of every claim requiring F != 0

| # | Line | Claim | Requires F != 0? |
|---|------|-------|------------------|
| 1 | 438 | F_{mu nu} measures path-dependence of parallel transport. | Substantively, yes. |
| 2 | 443 | "When F != 0, beliefs accumulate holonomy around closed loops." | Yes. |
| 3 | 445 | "Non-zero field strength indicates the agent's internal coordinate system is twisted." | Yes. |
| 4 | 467-469 | "Non-zero F_{mu nu} creates path-dependent information transport... Here it measures information field strength - how strongly the cognitive frame varies spatially." | Yes. The claim conflates rate-of-frame-variation (which can be nonzero — phi_i varies in c) with curvature (which under cocycle is identically zero). These are distinct concepts that the manuscript fuses. |
| 5 | 482 | "Strong gauge field curvature F_{mu nu}" sacrificed in working framework. | Concedes the issue but mislabels it as "small" rather than "structurally zero." |
| 6 | 555 | "Future work will explore curved base manifolds." | Distinct from the connection-curvature issue; consistent. |
| 7 | 758-790 | Connection form, F_{mu nu}, parallel transport, **vanishing-holonomy theorem**. | Theorem **establishes F = 0 holonomy** but the surrounding text continues to write F_{mu nu} as if it could be nonzero on the right-hand side. |
| 8 | 865 | "Yang-Mills curvature penalties int tr(F_{mu nu} F^{mu nu}) sqrt(g) dc." | Yes. **Vacuous as written.** |
| 9 | 1540-1546 | Effective metric uses A_mu A_nu; aside states only F_{mu nu} is gauge-invariant. | The aside contradicts the use. |
| 10 | 1620-1657 | Lorentzian signature mechanism rests on tr(A_mu A_nu) sign flip. | Requires the connection to be physically meaningful, which under cocycle it is only as a gauge-fixed quantity (acknowledged at line 1547, but the Lorentzian construction does not honor that caveat). |
| 11 | 2514-2524 | "Gravitational effects emerge as curvature in this pullback geometry." | Indirect: relies on the Yang-Mills metric being nontrivial. |
| 12 | 2626-2636 | Gauge-curvature minimization conjecture for linguistic evolution. | Yes. **Conjecture is vacuous since F == 0 by construction.** |
| 13 | 2685, 2797, 2874, 2926 | Various invocations of Yang-Mills kinetic term in the discussion sections. | Inherits the same defect. |
| 14 | 3261 | "Baker-Campbell-Hausdorff formula when gauge field strengths are small." | Pragmatic; consistent if reread as "phi gradients small," but it perpetuates the language confusion. |

Net: ~12 distinct passages claim or use nonzero F_{mu nu}; **1** passage (the theorem at 765-790) acknowledges F == 0 identically.

## Resolution options

### Option A: Promote A_mu to an independent connection variable (user's recommendation)

Replace Omega_{ij} = U_i U_j^{-1} (cocycle, automatically flat) with edge variables Omega_gamma = P exp(-int_gamma A_mu dx^mu) with A_mu carried as an independent g-valued field on C, not derived from a global single-valued U. Then F_{mu nu} can be nonzero, the cocycle is no longer automatic (and would need to be enforced as a separate constraint or relaxed), and the Yang-Mills kinetic term has nontrivial content.

**Cost:** This breaks two of the framework's most-cited features. (1) The vanishing-holonomy theorem ceases to hold — the manuscript's own argument at line 790 admits this. (2) Gauge covariance under the simultaneous global transformation U_i -> U_i g would have to be replaced by the local gauge transformation A -> g^{-1} A g + g^{-1} dg, which is the standard Yang-Mills story but does not preserve the "simultaneous frame change" interpretation that the manuscript uses to motivate gauge covariance from cognitive-consensus arguments.

**Benefit:** Restores all the curvature/holonomy/Yang-Mills/effective-metric language to mathematical content. Aligns the manuscript's physics analogies with the standard Yang-Mills construction.

**Compatibility with the implementation:** The companion code (`transformer/`) uses Omega_{ij} = exp(phi_i) exp(-phi_j), the cocycle form. So Option A is **inconsistent with the actual implementation**. Adopting Option A would make the manuscript a description of a different system than the one validated empirically.

### Option B: Drop all references to nonzero F_{mu nu}

Remove the field-strength-based discussion in Section "Gauge Field Strength," excise the Yang-Mills curvature penalty from line 865, drop the curvature-from-connection effective metric of Section 1540-1620, and drop the Lorentzian-signature worked example since its mechanism evaporates without a meaningful Yang-Mills kinetic term. Demote the gauge-curvature conjecture to a hypothetical extension.

**Cost:** Removes the entire "structural parallel to general relativity" thread. The Lorentzian-signature mechanism, which the conclusion holds up as a key contribution, must be retracted or completely rebuilt.

**Benefit:** Internally consistent; honest about what the cocycle parameterization can and cannot deliver.

**Compatibility with the implementation:** Faithful to the actual code.

### Option C: Distinguish flat-bundle regime from curved-bundle extension

Introduce two named regimes. Regime I (current implementation, cocycle Omega_{ij} = U_i U_j^{-1}): F == 0, no Yang-Mills metric, vanishing holonomy is a theorem, Lorentzian signature is unreachable. Regime II (curved-bundle extension): A_mu is independent, F can be nonzero, Yang-Mills metric is nontrivial, holonomy is generally nonzero, Lorentzian signature mechanism is operative. State that all empirical results are in Regime I and all general-relativity / Lorentzian-signature analogies live in Regime II.

**Cost:** Section restructuring; two parallel notations; explicit acknowledgment that the Lorentzian-signature pathway has no implemented evidence.

**Benefit:** Preserves all the mathematical content and ambition while making the dependency structure transparent. The reader sees clearly which results live in which regime.

**Compatibility with the implementation:** Faithful — Regime I matches the code; Regime II is flagged as an open extension.

## Recommended fix

**Option C, with these specific edits:**

1. **Lines 432-447** (Section "Gauge Field Strength: Path-Dependent Information Transport"). Open with a paragraph stating that the field-strength definition that follows applies to a curved-bundle extension and **vanishes identically** under the cocycle parameterization adopted in Section "Working Framework." Move the definition of F_{mu nu} (line 438) and the path-dependence discussion (lines 443-445) into this curved-bundle subsection.

2. **Lines 467-475** (Four-Curvatures section, "gauge field curvature" paragraph). Rewrite to state that under the cocycle parameterization the gauge field curvature is identically zero; nontrivial gauge field curvature requires the curved-bundle extension. Stop fusing "rate of frame variation" with "curvature" — the gradient ||grad phi_i||^2 measures the former without claiming F != 0.

3. **Line 482**. Replace "strong gauge field curvature F_{mu nu}" in the sacrificed-features list with "all gauge field curvature F_{mu nu}, which is identically zero under the cocycle parameterization."

4. **Line 765**. Already correct — keep verbatim.

5. **Line 790**. Keep verbatim, and add a forward reference: "We retain the field-strength notation in subsequent sections for use in the curved-bundle extension; the reader should treat all F_{mu nu} occurrences as identically zero in the implemented framework."

6. **Line 865** (free-energy functional). Move the Yang-Mills curvature penalty into a clearly-labeled "curved-bundle-extension regularizers" subsection. Note that under the implemented cocycle parameterization the penalty is identically zero and contributes nothing.

7. **Lines 1540-1620** (effective metric). The horizontal block tr(A_mu A_nu) is a gauge-variant quantity under the cocycle parameterization (the manuscript already acknowledges this at line 1547). The honest move is either (a) drop the horizontal block from the effective metric in the implemented regime and use only the score block, or (b) explicitly mark the horizontal block as a gauge-fixed quantity that requires choosing a reference frame. The current text does both incoherently. Pick (a) for the implemented regime; reserve (b) for the curved-bundle extension.

8. **Lines 1620-1670** (Worked example, Lorentzian signature). State up front that this entire construction lives in the curved-bundle extension, not the implemented framework. Note that the imaginary-phi mechanism produces nontrivial sign structure in tr(A_mu A_nu) only because tr(A_mu A_nu) is gauge-variant; the gauge-invariant content (tr(F F)) remains zero under cocycle, so the Wick-rotation argument has no gauge-invariant meaning in the implemented regime.

9. **Lines 2624-2636** (Gauge-curvature conjecture). Rewrite to state that the conjecture concerns gauge field curvature in the curved-bundle extension; under the cocycle parameterization the curvature is identically zero, so any "curvature minimization" interpretation must be of a different quantity (e.g., ||grad phi_i||^2 or Fisher-induced curvature on B). Replace F_{mu nu} = partial_mu phi_nu - partial_nu phi_mu + [phi_mu, phi_nu] with the corrected expression in terms of A, and note that this is identically zero under cocycle.

10. **Conclusion** (Lines 2870-2880 area, Lorentzian signature paragraph). State explicitly that the Lorentzian signature mechanism rests on the curved-bundle extension and is independent of the empirical results, which are entirely within the flat-bundle regime.

## Severity

High. The defect affects: (1) the framework's claim to a Yang-Mills kinetic structure (line 865, lines 1540, 1636); (2) the Lorentzian-signature mechanism (Sections "Signature Resolution" and "Worked Signature"), which is held up in the conclusion as a candidate resolution to the signature problem; (3) the gauge-curvature conjecture for language (line 2626), described as "the strongest testable prediction of our framework"; (4) the four-curvatures synthesis (lines 463-475). These are not incidental passages — they are load-bearing claims spread across the manuscript. The cocycle theorem at line 765-790 is mathematically correct and was added precisely to address this issue, but the rest of the manuscript was not made consistent with it. The fix is editorial/structural rather than mathematical (the math underlying both regimes is well-known), but it is extensive.
