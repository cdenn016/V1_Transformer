# Evidence Pack — softmax-beta-stationary-point

Neutral fact pack. Both teams work from this file.

## Manuscript references — `Attention/GL(K)_attention.tex` §4.6 "Deriving Attention from a Mixture-of-Sources Model"

The full derivation lives at lines 679–769. Six subsections:

### Generative model (lines 684–697)

> "Agent $i$ (the query) seeks to update its belief $q_i(k)$ by attending to a set of source agents $j$ (the keys). Agent $i$'s generative model assumes its latent state $k$ was drawn from one of the neighboring agents, selected by a categorical latent variable $z \in \{1, \ldots, N\}$:
> `P(k, z) = P(k | z) P(z)`  (Eq. eq:mixture_joint, line 688)
> where `P(z = j) = π_j` is the prior probability of attending to agent $j$ (the attention prior), and `P(k | z = j) = N(k; Ω_{ij}μ_j, Ω_{ij}Σ_jΩ_{ij}^T)` is agent $j$'s belief transported into agent $i$'s gauge frame via $Ω_{ij}$."

The manuscript at line 697 acknowledges the self-referential structure:
> "the component distributions $P(k | z=j) = Ω_{ij}q_j$ depend on the variational posteriors $q_j$ of other agents, making the generative model itself a function of variational quantities."

It then offers two complementary framings: (a) augmented joint model with mean-field decomposition (the "ELBO" reading), and (b) consensus energy (engineered functional). The manuscript states "in either framing, the resulting update equations are the same."

### Mean-field variational posterior (lines 699–708)

`Q(k, z) = q_i(k) · β(z)`  (Eq. eq:mixture_posterior, line 703)
with `β(z=j) ≡ β_{ij} ≥ 0`, `Σ_j β_{ij} = 1`.

### Alignment free energy (lines 710–725)

> "The free energy for the alignment component is the KL divergence between the variational posterior and the generative model:
> `F_align = KL[Q(k,z) ‖ P(k,z)] = Σ_j β_{ij} ∫ q_i(k) log(q_i(k) β_{ij} / (P(k|z=j) π_j)) dk`  (line 715)

Decomposition (line 720):
> `F_align = Σ_j β_{ij} [ ∫ q_i(k) log(q_i(k)/P(k|z=j)) dk + log β_{ij} - log π_j ]`
> = `Σ_j β_{ij} [ D_KL(q_i ‖ Ω_{ij} q_j) + log β_{ij} - log π_j ]`  (Eq. eq:mixture_free_energy)

The manuscript states (line 725): "The identification of the integral with `D_KL[q_i ‖ Ω_{ij} q_j]` follows directly from the definition of KL divergence and the generative model `P(k | z=j) = (Ω_{ij} q_j)(k)`, with no approximation."

Defining `E_{ij} ≡ D_KL(q_i ‖ Ω_{ij} q_j)` (line 727):
> `F_align = Σ_j β_{ij} ( E_{ij} + log β_{ij} - log π_j )`  (Eq. eq:mixture_energy_entropy, line 730)

Energy-entropy form (line 734):
> "This has the usual 'energy minus entropy' form `F_align = ⟨E⟩_β - H(β) + const`, where `H(β) = -Σ_j β_{ij} log β_{ij}` is the entropy of the attention distribution."

Note: the "+ const" term is `-Σ_j β_{ij} log π_j = ⟨-log π⟩_β`, which is β-dependent unless π is uniform. For uniform π = 1/N, this term equals `log N`, a true constant. Manuscript shorthand at line 734.

### Lagrangian minimization (lines 736–755)

> "Next, we minimize `F_align` with respect to `β_{ij}` subject to the constraint `Σ_j β_{ij} = 1` using Lagrange multipliers:
> `L = Σ_j β_{ij}(E_{ij} + log β_{ij} - log π_j) - λ(Σ_j β_{ij} - 1)`  (line 741)

Stationarity (line 747):
> "Requiring `∂L/∂β_{ik} = 0` produces:
> `E_{ik} + log β_{ik} + 1 - log π_k - λ = 0`"

Algebra: `log β_{ik} = log π_k - E_{ik} + (λ - 1)`, so `β_{ik} = π_k exp(λ-1) exp(-E_{ik})`. Normalization fixes `exp(λ-1) = 1 / Σ_m π_m exp(-E_{im})`.

Result (line 752):
> `β_{ik} = π_k exp(-E_{ik}) / Σ_m π_m exp(-E_{im})`  (Eq. eq:mixture_softmax_general, line 753)

Uniform-π specialization (line 757):
> "For a uniform attention prior `π_k = 1/N`, the prior factors cancel, yielding:
> `β_{ik} = softmax_k(-E_{ik}) = softmax_k(-D_KL[q_i ‖ Ω_{ik} q_k])`"  (Eq. eq:mixture_softmax, line 760, BOXED)

### Temperature rescaling (lines 764–769)

> "Without loss of generality we may include a temperature parameter `τ > 0` by rescaling the alignment energy `E_{ik} → E_{ik}/τ`. This rescaling, together with multiplication of `F_align` by `τ`, is equivalent to writing the un-minimized alignment free energy in the canonical row-Lagrangian form
> `F_align^(τ) = Σ_j [β_{ij} E_{ij} + τ β_{ij} log(β_{ij}/π_j)]`  (Eq. eq:F_align_canonical_tau, line 766)
> where τ couples explicitly to the attention entropy/prior term. The softmax stationary point `β_{ij}^* = π_j exp(-E_{ij}/τ)/Z_i` is unchanged, while the substituted value becomes `F_align^(τ)* = -τ log Z_i`, matching the `-τ log Z_i` reduction used in (eq:free_energy_final)."

The temperature-extended form combines `β log(β/π) = β log β - β log π` into a single entropy-prior term scaled by τ.

### Supplementary acknowledgement — `Attention/GL(K)_supplementary.tex` §B.1 (line 183)

> "For brevity we work with the entropy-suppressed surrogate `Σ_j β_{ij} D_KL(q_i ‖ Ω_{ij} q_j)` (i.e. holding the attention distribution β fixed and treating the alignment energy at this β); the canonical free energy of the main text adds the `τ β_{ij} log(β_{ij}/π_{ij})` entropy term to make the softmax form of β a stationary point."

This confirms the canonical-F vs surrogate distinction. The two differ by the attention-entropy term; gradients differ by `-τ⁻¹ Cov_β(KL, ∇KL)` per main-text line 866-871 (eq:autograd_envelope_gap).

### Forward-KL uniqueness (line 769; full proof in Appendix H of supplementary, line 1091)

> "the forward KL divergence is the unique f-divergence that preserves exponential-family closure under this linear coupling structure and yields a consistent dual interpretation for the attention weights (see Appendix A)."

Main-text reference is to Appendix A, but the actual location in the supplementary is Appendix H: `Attention/GL(K)_supplementary.tex` line 1091, "Conditional Uniqueness of the Forward KL Divergence via Variational Duality."

## Canon excerpts

### `external_canon_inference.md` §1 — Variational free energy standard form [Friston2010, BleiKuckelbirgJordan2017]

The standard variational free energy is:
```
F[q] = E_q[log q(s) - log p(o, s)]
     = KL(q(s) ‖ p(s | o)) - log p(o)
     = E_q[-log p(o | s)] + KL(q(s) ‖ p(s))
```

The manuscript's `F_align = KL[Q(k,z) ‖ P(k,z)]` at line 715 is the Form-1 version (expected log-ratio) applied to the joint over `(k, z)`. The mean-field factorization `Q = q_i(k) β(z)` is standard [BleiKuckelbirgJordan2017 §3].

### `external_canon_inference.md` §1 — Note on attention entropy term

> "Attention entropy term `τ β log(β/π)`. The user's claim that this is required for softmax to be stationary is *internally consistent* (it's the standard maximum-entropy regularization), but it is not part of standard single-agent FEP. It belongs to the entropy-regularized optimal-transport / soft-assignment literature (Sinkhorn divergences, etc.) and to the user's specific functional. Standard if presented as a Lagrangian for the soft-assignment problem; novel if claimed to follow from FEP alone."

The canon agrees that the entropy term is standard maxent regularization; the question is whether the manuscript presents it as a Lagrangian-for-soft-assignment (standard) or claims it follows from raw FEP (novel and would need separate justification).

### `external_canon_inference.md` §4 — Forward vs reverse KL

> "`KL(q ‖ p)` (forward / 'inclusive' KL): zero-forcing — q must be zero wherever p is zero. **Used in variational inference (ELBO).** ... The user's framework uses `KL(q ‖ Ω q')` — forward KL with the transported neighbor as the 'prior.' This is the variational direction."

Manuscript's choice of `KL(q_i ‖ Ω_{ij} q_j)` is the standard variational direction.

### `external_canon_math.md` §1 — KL between Gaussians, properties

```
KL(q ‖ p) = ½ [tr(Σ_p⁻¹ Σ_q) + (μ_p − μ_q)ᵀ Σ_p⁻¹ (μ_p − μ_q) − K + log(|Σ_p|/|Σ_q|)]
```
Properties: KL ≥ 0, KL = 0 iff q = p, asymmetric, not a metric.

### Mathematical canon — KKT conditions on the probability simplex

For the optimization problem
```
min_{β ∈ Δ^N} f(β),    Δ^N = {β : β_j ≥ 0, Σ_j β_j = 1}
```
with `f(β) = Σ_j β_j (E_j + log β_j - log π_j)` (the row-Lagrangian functional at line 730), the KKT conditions are:
- Stationarity: `∂f/∂β_k + Σ_j ν_j (-δ_{jk}) - λ · 1 = 0` for each k where β_k > 0, with `ν_j ≥ 0` complementarity-slack multipliers for β_j ≥ 0.
- Primal feasibility: `Σ β = 1`, `β ≥ 0`.
- Dual feasibility / complementary slackness: `ν_j β_j = 0`.

For this f, `∂f/∂β_k = E_k + log β_k + 1 - log π_k`. As β_k → 0⁺, `log β_k → -∞`, so the inequality constraint is strictly inactive at any interior stationary point and `ν_j = 0`. The interior KKT system reduces to the equality-constrained Lagrangian at line 741. **The Lagrangian as written is sufficient when the solution is interior; this is the standard treatment for entropy-regularized soft assignment** [Boyd & Vandenberghe *Convex Optimization* §5.5; Cuturi 2013 *Sinkhorn Distances*].

### Convexity of `f(β) = Σ β_j (E_j + log β_j - log π_j)`

`f` is strictly convex on the open simplex because:
- `β_j E_j` is linear (E_j is constant in the inner optimization over β with q fixed).
- `β_j log β_j` is strictly convex on `(0, ∞)`.
- `-β_j log π_j` is linear.
- Sum of strictly convex + linear is strictly convex.

Strict convexity on the open simplex implies: any interior stationary point is the unique global minimum. **The manuscript does not state this explicitly**, but it is a standard consequence cited e.g. in [Wainwright & Jordan 2008 *Graphical Models, Exponential Families, and Variational Inference* §3.6].

## What this evidence does NOT settle

1. **Self-referential generative model.** Manuscript line 697 admits the component distributions `P(k|z=j) = Ω_{ij}q_j` depend on `q_j` (other agents' beliefs), making the generative model itself a function of the variational quantities. The two framings (augmented joint with mean-field decomposition; consensus energy) are offered as resolutions. Whether one or both are mathematically tight, and whether either makes the row-Lagrangian a derivation rather than a definition-by-fiat, is a load-bearing question. The red side may argue this is a circularity that converts the derivation into a fixed-point definition; the blue side may argue this is standard coordinate-ascent VI (the q_j's are held fixed during the β-update, the β's during the q-update, with mutual consistency at the joint fixed point).

2. **"+const" sloppiness at line 734.** The energy-entropy form `F_align = ⟨E⟩_β - H(β) + const` is exact under uniform π (where `-Σ β log π = log N`); under non-uniform π, the "const" is `⟨-log π⟩_β`, which is β-dependent. The decomposition is still valid (the log π term enters the softmax as a prior bias at line 753) but the "energy minus entropy plus const" framing is uniform-π-specific. The claim specifies uniform π, so this is consistent with the headline, but it leaves the general-π case as a separate consideration.

3. **Second-order conditions.** The manuscript verifies first-order stationarity (the Lagrangian KKT equations) but does not explicitly verify that the stationary point is a minimum (let alone the unique global minimum). Convexity of `Σ β log β` on the open simplex gives this for free, but the manuscript does not state convexity. The claim says "exact stationary point" rather than "unique global minimum"; under a strict reading, first-order stationarity is what is claimed and what is shown.

4. **Boundary behavior.** The Lagrangian at line 741 treats only the equality constraint `Σ β = 1`, not the inequality constraints `β_j ≥ 0`. The standard KKT treatment includes the inequality constraints, but for entropy-regularized problems the `-log β_k → ∞` boundary blow-up makes the inequality constraints non-binding at any stationary point. The manuscript's choice to write only the equality Lagrangian is standard practice but not derived from first principles.

5. **The "+1" at line 747.** `∂(β log β)/∂β = log β + 1`. The "+1" cancels into the Lagrange multiplier λ during normalization; the manuscript writes the line correctly but does not explain the absorption explicitly.

6. **Temperature factor in the un-minimized vs minimized form.** Manuscript at line 764–769 says rescaling `E → E/τ` together with multiplying `F_align` by τ gives the canonical `F_align^(τ) = Σ [β E + τ β log(β/π)]`. This is correct: τ · [β · (E/τ) + β log β - β log π] = β E + τ β log β - τ β log π = β E + τ β log(β/π). The stationary point under this functional satisfies `E + τ(log β + 1 - log π) = λ`, giving `β = π exp(-E/τ)/Z`. Correct.

7. **The "energy minus entropy + const" identification matches the canonical maxent / Sinkhorn problem.** Cuturi 2013 *Sinkhorn Distances* §4 derives the exact same form: under entropy-regularized assignment with constraint `Σ β = 1`, the optimal `β = exp(-E/τ)/Z` is the unique solution. The manuscript's derivation is structurally identical to the Sinkhorn row-update.

8. **What is novel vs standard.** Per `external_canon_inference.md` §1 last paragraph: the τβ log(β/π) entropy term is "standard if presented as a Lagrangian for the soft-assignment problem; novel if claimed to follow from FEP alone." The manuscript presents it as a Lagrangian (line 715 `F_align = KL[Q‖P]`, decomposed and minimized). This is the standard framing.
