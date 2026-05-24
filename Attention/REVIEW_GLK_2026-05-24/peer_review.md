# Peer Review — *Attention as Gauge-Theoretic Variational Inference*

**Manuscripts.**
- `Attention/GL(K)_attention.tex` (main, ~2360 lines, modified 2026-05-21)
- `Attention/GL(K)_supplementary.tex` (supplementary, ~1437 lines, modified 2026-05-21)

**Scope of this review.** Theory, math, derivations, exposition, internal consistency, citation hygiene. Per the author's instruction, empirical results and numerical validation tables (BERT § Appendix E, GL(K) training § §7, RG synthetic check § Appendix F) are **not** assessed for substantive content — they are noted only when they appear as referents in textual claims that affect the theory's status.

---

## Summary Statement

The manuscript pair presents a substantively new derivation: standard transformer attention is recovered as a degenerate limit of a single variational principle on a statistical fiber bundle, with categorical source-selection variables yielding the softmax structure and the gauge invariance of the Gaussian KL divergence providing the unifying symmetry. The mathematical core — the mixture-of-sources generative model, the softmax derivation via row-Lagrangian KKT, the GL(K) invariance theorem, the untied query-key carving from per-token frames (Eq. `eq:full_kl_general`), and the closed-form covariance fixed-point dynamics — is correct where checked and represents a real contribution. The recovery of standard architectural features (causal masking, ALiBi, RoPE, layer normalization, multi-head structure, residual connections, GLU-type activations) from a single principle is the manuscript's main organizing claim and is presented with appropriate care to distinguish derived from interpretive correspondences (D/D♯/S markings in Table `tab:fep_nn_correspondence`).

The manuscript would benefit from a focused revision pass on three classes of issue: (i) two **theorem-status overstatements** — the "Conditional Uniqueness of forward KL" theorem (Supplementary App. H) and the "Theorem 6.x conditional CLT contraction" of the RG section — both of which contain a small but real gap or overreach; (ii) several **scope qualifications** where the framework's expressiveness is claimed to "explain" features that it in fact only structurally accommodates (multi-head subspace embeddings, GELU/SiLU activation forms, the universal-Rényi extension); and (iii) a **convention-consistency** issue between Appendix C and Appendix D on the gauge frame gradient (right- vs left-trivialized) that is internally undeclared.

**Strengths**
- The KL-as-attention-cost derivation from the mixture-of-sources model (§3.2) is clean, well-motivated, and the non-uniform-prior taxonomy unifying causal masking, sliding windows, ALiBi, and relative biases (§3.2.5) is one of the manuscript's more elegant contributions.
- The untied query/key carving `Q_i = U_i^{-1}\mu_i`, `K_j = U_j^\top \Sigma_j^{-1}\mu_j` (Eq. `eq:gauge_qk`) is a genuinely novel structural identification beyond the prior trivial-frame route; the explicit "verified symbolically to machine precision" claim at main:1138 was independently checked and holds.
- The D/D♯/S taxonomy at the bottom of Table 1 (main:1700) and the explicit disclosure of which steps are exact algebra vs. literal reparameterization vs. approximate softmax-shift cancellation (Complete Attention Formula, main:1330) shows uncommon care about epistemic status.
- Style hygiene is excellent: I found zero instances of the project's banned Claude-isms (`key insight`, `crucially`, `notably`, etc.) and zero instances of the banned spacing macros (`\;`, `\,`, `\!`) in either file.

**Weaknesses (decision-relevant)**
- Theorem H (Supp. App. H) has a step-3 algebra gap and lists an unused hypothesis; the χ² intuition paragraph mischaracterizes the divergence's stationary form; the relation to the main paper's "extends without change to Rényi" claim (main:1062) needs reconciliation.
- The RG Theorem (main:2258) is more honestly a Proposition: clauses (i)–(iii) are textbook CLT applied to a definition. The synthetic-CLT validation (Supp. Table at supp:990) is labeled as "validating the scaling predictions" but in fact validates the CLT on synthetic i.i.d. data. The supplementary at supp:1026 contradicts the main paper at main:2280 on the H1/H2 interpretation of measured deviations.
- The multi-head row of Table 1 and several positional/activation rows are marked D where D♯ or S would be more honest. The framework explains the per-head invertible bilinear `M_h^a` but does **not** derive the rectangular subspace embeddings `U_Q^a, U_K^a, U_V^a`.
- Appendix D §"Gauge Frame Dynamics on GL(K)" introduces a left-trivialized convention that conflicts with Appendix C's right-trivialized convention; the implementation's actual choice is not declared. If mixed, the preconditioned natural-gradient step is Ad-twisted away from φ = 0.

**Recommendation.** Major revisions on the items listed in §1 (Major Comments) and §2 (Minor Comments) below. None of the issues identified are fatal to the manuscript's central thesis; all are repairable within the existing scope. After revision the manuscript would be a solid contribution to the variational/geometric interpretation of transformer attention.

---

## 1. Major Comments

### M1. Conditional Uniqueness of forward KL (Supplementary Appendix H) — algebra gap in Step 3, unused hypothesis, χ² mischaracterization, and a latent tension with main §4.2.1

**Location.** `GL(K)_supplementary.tex:1196–1290` (Theorem statement + proof), with related claims at `GL(K)_attention.tex:1060–1074` (Rényi extension) and the assumption list at `GL(K)_supplementary.tex:1096`.

**M1.1 — Step 3 reverse direction has a missing specialization step.**
The proof's reverse implication subtracts the rearranged geometric-mean target (`supp:1265`) from the general stationarity condition (`supp:1230`) and concludes term-by-term that `f'(r_ij) = log r_ij + k`. As written, however, the difference of the two conditions is
```
sum_j beta_ij [ f'(r_ij) - log r_ij ] = const,
```
not a single algebraic identity in one variable. To collapse this to `f'(r) - log r = k`, one needs to specialize the system to configurations where all transported neighbor densities `(Ω_ij q_j)(c)` agree pointwise — at which point `Σ β_ij = 1` collapses the sum. Real-analyticity of `f` then extends the identity from the resulting "open subinterval of ratios" (correctly invoked at `supp:1276`) to all of `(0, ∞)`. The specialization step is not written in the proof, so the proof reads as if a sum is being matched term-by-term. *The conclusion is salvageable; the proof is not.*
**Required:** insert one paragraph in Step 3 constructing the specialization configuration, then collapse the β-sum.

**M1.2 — Assumption (iii) "exponential-family closure" (`supp:1096`) is unused.**
The Theorem's actually-invoked hypotheses are: convex f, `f(1)=0`, `f'(1)=0`, real-analyticity, `Σβ=1`, linear-in-β coupling, and admissible-ratio range containing an open subinterval. Exponential-family closure of `q_i*` is a *consequence* of `f'(t) = log t`, not a hypothesis; it never enters the algebra. Stating it as assumption (iii) inflates the conditions.
**Required:** delete (iii) or relabel as a remark: "the resulting `q*` automatically lies in the exponential family, by Step 2."

**M1.3 — χ² intuition (`supp:1090`) is incorrect.**
The intuition paragraph claims χ² "produces algebraic (polynomial-ratio) rather than exponential solutions." For χ², `f(t) = (t-1)^2/2`, `f'(t) = t - 1`. Substituting into the general stationarity condition gives `log(q_i/p_i) + 1 + Σ β [q_i/(Ω q_j) - 1] = λ`, which is **transcendental** (the `log p_i` term persists in the stationarity equation), not polynomial-ratio. The Theorem still excludes χ² correctly via the reverse direction (`f'(t) = t - 1` cannot equal `log t + k` on any open interval). The intuition paragraph thus mischaracterizes χ²'s solution structure; the theorem statement is unaffected.
**Required:** rewrite the χ² sentence to: "χ² stationarity is transcendental but mixes `q_i` and `log q_i` rather than producing the log-linear form required to remain in the exponential family."

**M1.4 — Latent inconsistency with main paper §4.2.1 Rényi extension (`main:1060–1074`).**
The main paper claims at line 1062 that the construction "extends without change to the Rényi family `D_α`." But under the same variational machinery, replacing forward KL with `D_α` no longer produces the geometric-mean Boltzmann form `q* ∝ p^{1/2} Π (Ωq_j)^{β/2}` — that closed form is what Theorem H proves to be **unique** to KL (i.e., to `f(t) = t log t - t + 1`). The Rényi softmax structure of `β` (line 1071) does extend, because it depends only on the row-Lagrangian KKT with `Σβ = 1` and an arbitrary divergence in the energy; the *belief stationary form*, however, does not.
**Required:** qualify `main:1062` to "the softmax structure of `β` extends to the Rényi family; the geometric-mean closed form of `q*` does not, and the Rényi-α stationarity admits no analogous closed form for `q*`."

---

### M2. The "Theorem 6.x" — "Conjecture 6.x" split (Main §6.4 and Supp. Appendix F) — Theorem is a Proposition, supplementary contradicts main on interpretation, K² → √K is asserted without derivation

**Location.** `GL(K)_attention.tex:2230–2282` (main §6.4); `GL(K)_supplementary.tex:917–1027` (Appendix F).

**M2.1 — Theorem `thm:rg_conditional` is a Proposition / Calculation, not a Theorem.**
Read the three clauses of `main:2258–2266`:
- Clause (i) "intrinsic fixed point `g_1 = g_2 = g_3 = 0`": this is `0 = 0` under the definitions of g_1, g_2, g_3 in `main:2239–2241`. Definitional.
- Clause (ii) "all `y_k < 0` ⇒ infrared-stable under i.i.d.": this is the textbook CLT contraction rate `n^{-1/2}` (for g_1) and `n^{-1}` (for g_2, g_3) applied to averages of i.i.d. perturbations [Cardy 1996; Goldenfeld 1992]. No gauge content enters the calculation; the same rates would hold for any quantity definable as an average of independent perturbations.
- Clause (iii) "R_n generates `Var_A(μ) > 0`": this is the definition of `Var_A(μ)` (`main:2249`) plus the observation that within-cluster mean variance is generically nonzero.

The non-trivial content sits entirely in the i.i.d. premise (`main:2260`), which is precisely the assumption the Conjecture has to relax. Calling this a Theorem creates a derivational impression the content does not carry. **Required:** downgrade to Proposition (or Calculation), and state explicitly that the i.i.d. premise is load-bearing and that the calculation does not itself require the gauge framework.

**M2.2 — Internal inconsistency between coarse-graining rate and reported holonomy exponent.**
At `supp:949`, the manuscript justifies the holonomy scaling `y_3 = -1` by stating "triangle sides each contract as `n^{-1/2}`." Under the coarse-graining map defined at `supp:940` (`Ω_{AB} = (1/|A||B|) Σ_{i∈A,j∈B} Ω_{ij}`, a bilinear average over `n²` edges by CLT), each inter-cluster transport contracts as `n^{-1}`, not `n^{-1/2}`. A linear-order expansion `H = (I + e_1)(I + e_2)(I + e_3) - I = e_1 + e_2 + e_3 + O(e²)` then gives `||H - I|| ∼ n^{-1}` ⇒ `y_3 = -1`, matching the reported synthetic value of −1.003 and the matrix equation at `supp:951`. If sides actually contracted as `n^{-1/2}` (as the prose at 949 states), the expansion would give `y_3 = -1/2`, contradicting both the table and the matrix. The prose at `supp:949` therefore mislabels the contraction rate but the conclusion is correct under the actual `n²`-averaging map.
**Required:** rewrite `supp:949` to "edges contract as `n^{-1}` under the bilinear average in `eq:meta_agent_beliefs_supp`; the linear-order product over a triangle then gives `||H - I|| ∼ n^{-1}`."

**M2.3 — "Validates the scaling predictions to three significant figures" (`supp:988`) overstates the synthetic test.**
The synthetic-CLT test (`supp:985–1000`) draws `n` i.i.d. samples from a Gaussian, averages them, and verifies the empirical standard deviation contracts as `n^{-1/2}`. This is a numerical confirmation of the CLT on synthetic i.i.d. data — a sanity check of the analytic prediction, not validation of the gauge-theoretic framework or any claim about transformer attention graphs. The main paper at `main:2280` is more careful ("this content does not itself require the gauge framework"); the supplementary text is not.
**Required:** replace `supp:988`'s "the results confirm the predicted exponents to three significant figures" with "the results confirm the CLT on i.i.d. synthetic data — a sanity check on the analytic prediction, not validation of the framework's claims about trained models." Reconcile with M2.4 below.

**M2.4 — Main paper and supplementary disagree on H1/H2 interpretation.**
At `main:2280` the manuscript gives finite-size bias (H1) and genuine cross-token correlation (H2) comparable weight, noting that the measured shift in `y_3` (+0.2 vs predicted −1) "would require the correction term to dominate the asymptotic … a regime standardly indicative of crossover behavior rather than small-`L` corrections to an infrared-stable fixed point." At `supp:1026` the same data is interpreted as solely H1: "the graph-based deviations quantify the finite-size corrections that would vanish in the `N → ∞` limit." The unconditional `N → ∞` extrapolation in the supplement is precisely the H1-presupposing claim the main paper warns against.
**Required:** align `supp:1026` with `main:2280`'s two-hypothesis framing; remove or qualify the unconditional `N → ∞` extrapolation.

**M2.5 — `O(K^2) → O(\sqrt K)` sample-efficiency advantage is asserted, not derived.**
Conjecture clause (b) at `main:2275` (restated at `supp:956` and listed as testable prediction at `supp:968`) claims the absorbed `O(K²)` degrees of freedom produce an `O(√K)` sample-efficiency advantage. No derivation appears in either file. Standard statistical learning theory does not give a √K rate from a K² parameter difference for free — the rate depends on the loss class, the data distribution, and the precise sample-complexity bound (VC, PAC-Bayes, Chinchilla). If a heuristic argument is intended (e.g., "extra K² DoF require √(K²) = K samples to learn, with a further reduction"), the heuristic should be written down.
**Required:** either add a brief derivation of the K² → √K step (citing the sample-complexity result invoked) or relabel as "we conjecture an `O(√K)`-class advantage without a quantitative derivation of the exponent."

---

### M3. Multi-head identification (Main §4.1.3 and §4.6) — framework explains the per-head invertible bilinear, not the rectangular subspace embeddings

**Location.** `GL(K)_attention.tex:1240–1256` (thin-SVD lift); `1700` (Table 1, multi-head row marked D); `1705–1745` (§4.6 multi-head).

The literal statement `W_Q W_K^\top = \sigma^{-2} \Omega^{-\top}` at `main:1243` reads as if the gauge factor `Ω` is identified with the standard `W_Q W_K^\top \in \mathrm{GL}(d_k)`. The patch at `main:1247–1256` correctly acknowledges that standard `W_Q^a, W_K^a \in \mathbb{R}^{d_{\mathrm{model}} \times d_{\mathrm{head}}}` are rectangular and the ambient kernel `W_Q^a (W_K^a)^\top` has rank `≤ d_{\mathrm{head}}` (not in `GL(d_{\mathrm{model}})`). The fix invokes a thin SVD: `W_Q^a = U_Q^a A_Q^a` with `U_Q^a \in \mathbb{R}^{d_{\mathrm{model}} \times d_{\mathrm{head}}}` isometric and `A_Q^a \in \mathrm{GL}(d_{\mathrm{head}})`. The identification then operates at the invertible head-space factor `M_h^a := A_Q^a (A_K^a)^\top \in \mathrm{GL}(d_{\mathrm{head}})`.

**This patch is mathematically correct** as a value-level statement on the head subspace. The remaining issue:

(a) The subspace embeddings `U_Q^a, U_K^a, U_V^a` are entirely free factors that the gauge framework does **not** predict, constrain, or interpret. They select which `d_{\mathrm{head}}`-dimensional subspace of the embedding space each head reads from — a structural choice with no analog in the `(\sigma, \Omega)` parameterization. The text at `main:1318` absorbs `U_V^a` into `W_O`, which is fair bookkeeping for the value path but leaves `U_Q^a, U_K^a` unidentified. The framework therefore **explains the per-head invertible bilinear, not the subspace selection**.

(b) The thin-SVD factorization is not unique; the table caption at `main:1700` (the `D^\sharp` footnote) correctly notes "lift fixed only up to right multiplication by elements of the isotropy group."

(c) The multi-head row of Table 1 at `main:1662` (`\mathrm{GL}(d_{\mathrm{head}})^H \subset \mathrm{GL}(d_k)` ↔ Multi-head attention) is marked `D`. By the analysis above, it should be `D^\sharp` (head-space derivation up to the U-isometry equivalence), or arguably `S` if one wants to acknowledge that the subspace-selection part is structural.

**Required:**
- Add one sentence after `main:1256` stating: "The framework's identification covers the invertible per-head bilinear `M_h^a` and the value aggregation up to `W_O` absorption; it does not derive the choice of subspace embeddings `U_Q^a, U_K^a, U_V^a`, which remain a structural design choice."
- Downgrade the multi-head row of Table 1 to `D^\sharp` (or split into two rows: the block-diagonal `GL(d_head)^H` structure as `D`, the rectangular subspace embeddings as `S`).
- Reconsider whether the per-head invertibility claim at `main:1729` ("Each per-head transport `\Omega^a \in \mathrm{GL}(d_{\mathrm{head}})` is full rank and invertible within its block") sufficiently distinguishes the head-space `\Omega^a` from the ambient rectangular projection.

---

### M4. Convention flip between Appendix C and Appendix D on the gauge-frame gradient — undeclared implementation choice

**Location.** `GL(K)_supplementary.tex:439–541` (App C, right-trivialized); `652–663` (App D, left-trivialized); `486` (autograd hook); `659` (symmetric-quadrature claim for K > 3).

Appendix C uses the **right-trivialized** form throughout: `∂Ω_ij/∂φ_i^a = Q_a^{(i)} Ω_ij` with `Q_a^{(i)} = \mathrm{dexp}_{\phi_i}(T_a)` (the right-trivialized differential), consistent within the section. Appendix D §"Gauge Frame Dynamics on GL(K)" (`supp:655–659`) then introduces the **left**-trivialized form `∂/∂φ^a \exp(X) = \exp(X) \cdot Q_a^L` with `Q_a^L = \mathrm{Ad}_{\exp(-φ)}(Q_a^R)`. Both are mathematically correct; both give the Fréchet derivative `D_φ(\exp)[T_a]` when paired with the appropriate group-element factor. They differ by an `\mathrm{Ad}` action.

The concern is that the components of the Euclidean gradient `∂F/∂φ_i^a` in the `{T_a}` basis depend on which convention is used. The Cartan-involution-modified preconditioner `\tilde g_{ab}` at `supp:587–590` acts on those components. If the implementation computes the gradient in one convention and feeds it to a preconditioner in the other (or mixes between them along the codepath), the resulting preconditioned natural-gradient step is **off by an Ad-twist** at points where `φ ≠ 0`. The convention is collapsed at `φ = 0` (where `Ad_{exp(0)} = \mathrm{id}`) but generically nonzero in training.

There is also an internal contradiction: `supp:486` says "automatic differentiation through PyTorch's `torch.matrix_exp` provides the GL(K) differential implicitly", which gives the **Fréchet** derivative `D_φ(\exp)[T_a]`. `supp:659` then says "for K > 3 ... we instead compute `Q_a^L` via the Fréchet derivative of the matrix exponential using symmetric quadrature." These two paragraphs describe different computational paths.

**Required:**
- Insert a single declarative sentence in App C and App D stating *exactly* which trivialization the implementation uses to define `∂F/∂φ^a` and feed the preconditioner, with the explicit equation (right- or left-trivialized).
- Reconcile `supp:486` and `supp:659`: is the implementation autograd-through-`matrix_exp` or symmetric-quadrature? Both routes work, but the manuscript should pick one and state it.
- A finite-difference cross-check of the preconditioned natural-gradient step at a non-trivial `φ_0 ≠ 0` would put this beyond doubt; the manuscript already documents an FD validation pass at `supp:666` ("relative error < 10^{-5}") — extend that pass to the preconditioned step specifically.

---

### M5. Bundle scaffolding (Supp. App. A) — leaked deferred content in main §4.6 "Per-head holonomy" and unused symbols in notation table

**Location.** `GL(K)_attention.tex:400` (notation table, `Φ_i, \tilde Φ_i`); `677` (explicit exclusion from equations); `1739–1745` (per-head holonomy invokes edge-relaxed form).

The author's disclosure that App A is largely scaffolding from the companion paper [Dennis2025it] is appropriate (`supp:106`, `supp:176–177`). Two leaks remain:

(a) The notation table at `main:400` lists `Φ_i, \tilde Φ_i` (cross-fiber morphisms) with the description "Cross-fiber morphisms (belief ↔ model)". The main paper then states at `main:677` that "the cross-fiber morphisms `Φ_i, \tilde Φ_i` named in Table 1 do not enter Eq. 3.21." Having symbols in the notation table that are explicitly unused is sloppy and confusing to readers.
**Required:** either remove `Φ_i, \tilde Φ_i` from Table 1, or footnote them as "introduced in App. A, not used in the main paper's equations; included for cross-reference with the companion paper [Dennis2025it]."

(b) The "Per-head holonomy" paragraph at `main:1739–1745` silently invokes the edge-relaxed Regime II form `Ω_ij = \exp(δ_ij^{(a)} G_a)` and writes the per-head holonomy `H_ijk^{(a)} = \exp(δ_{ij}^{(a)} G_a) \exp(δ_{jk}^{(a)} G_a) \exp(δ_{ki}^{(a)} G_a)` — without prefixing the paragraph with a "Under the Regime II extension (deferred to [Dennis2025it]) ..." conditional. The rest of the main paper lives under the flat-bundle Regime I assumption; this paragraph contradicts the assumption locally.
**Required:** open the "Per-head holonomy" paragraph at `main:1739` with an explicit conditional: "Under the edge-relaxed (Regime II) parameterization developed in the companion paper, if one were to introduce per-head edge connections `δ_{ij}^{(a)}`, the holonomy would factorize block-wise as follows…"

A milder version of the same issue appears at `main:2241` (`g_3 = ||H_{ijk} - I||`) where the RG analysis treats holonomy as a coupling. Since §6.4 is itself a speculative-extensions section, this is more defensible — but the section header should at least state "Throughout this section we work in the edge-relaxed (Regime II) parameterization."

---

## 2. Minor Comments

### Mathematical exposition

- **`main:902`**: "The hierarchical structure of the generative model (discussed in the appendix) implies that the effective prior precision manifestly depends on the model uncertainty." The derivation that follows is a closed-form Lagrangian on a single-scale problem; the appeal to "hierarchical structure" is rhetorical, not load-bearing. Either remove the appeal or label it as heuristic motivation that is not part of the derivation.

- **`main:1157`**: "polar decomposition shows that `M_ij` ranges over all of `GL(d_k)`." True per-pair, but the family `{M_ij}_{i,j}` is constrained by the shared factorization through `(U_i, Σ_j)` — it is **not** independent per pair. Add a sentence noting this constraint, parallel to the constraint on `{W_Q μ_i \cdot W_K μ_j}` through shared `W_Q, W_K`.

- **`main:1244`**: the boxed identification `W_Q W_K^\top = (1/σ²) Ω^{-\top} \in \mathrm{GL}(d_k)` (square bracket form) is in tension with the rectangular reality acknowledged 5 lines later at `main:1249`. The boxed equation, as the visual anchor of the dot-product reduction, would be more honest if rewritten as `M_h^a = A_Q^a (A_K^a)^\top = (1/σ²) (Ω^a)^{-\top} \in \mathrm{GL}(d_{\mathrm{head}})` from the outset, with one explanatory sentence about why this is the head-space invertible factor.

- **Table 1, `main:1692`**: the "Boltzmann GLU gate ↔ GELU/SiLU activation" row is marked `S`. The functional forms differ: GELU is `x·Φ(x)` (Gaussian CDF), SiLU is `x·σ(x)` (sigmoid), and the framework's VFE gate is `x·exp(-||x||²/τ)/Z` (Gaussian PDF inside a softmax). The manuscript correctly disclaims this at `main:1926–1934` ("not functionally identical, family membership only"). Consider marking this row `I` (interpretive) rather than `S` (structural), or annotating with the disclaimer.

- **Table 1, positional-structure rows (`main:1666–1670`)**: ALiBi/sliding window/T5 bias are marked `D`. The framework supplies the *interpretive frame* (the prior `π_k` admits these functional forms), but the specific choice of `exp(-m|i-j|)` for ALiBi vs. `\exp(b_{i-j})` for T5 is not predicted from first principles — it's the catalog of priors that practitioners have empirically chosen. These rows could honestly be marked `S` or `I`.

- **`supp:594` (regularization of the central direction)**: the manuscript regularizes the degenerate Killing form on `R·I` by `\tilde g \to \tilde g + ε I`. Add a sentence stating how `ε` is chosen and noting that the resulting natural gradient on the trace direction is `ε^{-1} \partial F/\partial(\mathrm{tr}\phi)` — an unconditioned step scaled by `ε^{-1}`. This is a small detail but it pins down the metric on a one-dimensional non-semisimple direction.

- **`main:858` (envelope theorem statement)**: the displayed equation uses `$$ ... $$` rather than the `equation` environment used everywhere else in the paper. Inconsistent with the project style and risks losing a label.

- **`main:1048, 1054, 1080, 1085, ...`**: many display equations end without punctuation (no comma or period). The CLAUDE.md style note specifies "Apply standard equation punctuation (comma/period at end of display equations) as part of any doc cleanup pass." A consistent sweep is overdue.

### Citation hygiene

- The supp. proof of Theorem H cites `Boyd2004` and `Wainwright2008` for the row-Lagrangian KKT argument (`main:744`), which is appropriate. Add a citation to Csiszár (1967) at the *Theorem statement* itself (`supp:1199`), since the conditional-uniqueness statement sits squarely in the f-divergence canon initiated by Csiszár.

- Eq. `eq:dexp_series` (`supp:445`) is the standard right-trivialized differential of the matrix exponential. The citation `\citep{Gallier2020,Hall2015}` is appropriate. The "differs from the Fréchet derivative" caveat at `supp:450` could profitably also cite Higham (2008, Ch. 10) which is already in the bibliography and is the standard reference for the integral/block-matrix Fréchet formulas.

- `main:60`: the introduction cites bahdanau2014neural, amari1998natural, bronstein2021geometric, foerster2016learning, wooldridge2009introduction as evidence for the convergence claim. Foerster (multi-agent RL) and Wooldridge (intro to MAS textbook) are out of place in this list; they are not load-bearing for the "convergence on inference under uncertainty" claim. Either drop them or replace with citations more directly on point (Friston 2010, 2017; Parr & Friston 2022 are already cited later).

- The PIFB companion paper `\citep{Dennis2025it}` is referenced ~10 times. Confirm that the bibliographic entry resolves to a stable identifier (arXiv or DOI) by the time of submission; "in preparation" or a placeholder will be a referee target.

### Structural / exposition

- The "An intuitive simplification for non-geometers" subsection at `main:422` is well-intentioned but stylistically a bit informal for the rest of the manuscript. It would be more naturally placed as a sidebar/box, or moved to a brief introductory paragraph before the formal §3.

- The Algorithm `alg:em_loop` (`main:2027`) is the first explicit pseudocode in the paper and lands without a connecting paragraph. A one-sentence pointer earlier (e.g., at `main:2011`) saying "the full pseudocode appears in Algorithm 1 below" would help.

- The notation table at `main:370–420` is comprehensive but two rows are missing: `η_q, η_s` (the timescale-separation learning rates introduced at `main:2317`) and `\hat\alpha` (the trained-from-objective version of `α` at `main:2058`).

- **`main:1346`** ("The limits are deliberately aggressive. They collapse the statistical manifold into a basic Euclidean space and absorb the gauge parameters into the learned matrices."): this sentence ends a paragraph and is followed by another lone sentence at `main:1348`. Both could be combined into one clear paragraph stating "the three-limit chain is summarized in Table 1; each row indexes a generalization that the framework opens by relaxing the corresponding limit."

- **`supp:953`**: the linearised RG matrix is displayed as a diagonal `3×3` matrix with entries `n^{-1/2}, n^{-1}, n^{-1}`. The "y₁" row mixes the intrinsic vs. total channel distinction made in the surrounding text. Either annotate (e.g., "where the (1,1) entry is the intrinsic-channel scaling; the total channel does not flow to zero, see clause (iii)") or split into two displays.

### Style

- Zero banned macros (`\;`, `\,`, `\!`) and zero banned Claude-isms detected. Hygiene is excellent.
- A few of the boxed equations (`main:444`, `main:847`, `main:1147`) contain the word "Box" implicitly through the `\boxed{}` macro; this is fine and consistent.

---

## 3. Questions for Authors

1. **Theorem H scope.** Is the conditional-uniqueness theorem intended to also apply to the model-channel KL `D_{KL}(s_i \| \tilde\Omega_{ij} s_j)` in the full free energy (`supp:1067`)? The proof structure transfers, but the manuscript does not state this explicitly.

2. **Rényi extension.** Given M1.4, do you intend the Rényi extension at `main:1062` as: (a) a softmax-form-only extension (β remains softmax but the closed form for `q*` is lost), or (b) a full extension under a different uniqueness theorem (with `D_α` having its own dual structure)? If (b), where is that theorem?

3. **Gauge convention.** Per M4, which trivialization (right or left) does the production implementation use to compute `∂F/∂φ^a`? Is the Cartan-modified preconditioner `\tilde g_{ab}` applied in the same convention?

4. **`O(\sqrt K)` scaling argument.** Per M2.5, can you sketch the sample-complexity argument that takes `O(K²)` absorbed DoF to a `√K` sample-efficiency advantage? If the argument is dimensional or heuristic, label it as such.

5. **Cocycle vs. edge-relaxed in §4.6.** Per M5(b), is the per-head holonomy paragraph at `main:1739–1745` intended as Regime I (vacuous, since `H_{ijk} = I`) or Regime II (per-head edge connections)? The block-diagonal factorization stated only has content under Regime II.

6. **Empirical scope.** The author's deferral of all empirical claims to a future iteration is noted. Will the planned multi-seed runs at K=90 (per the "Seed disclosure" paragraph at `main:2295`) be reported in the same manuscript or in the companion paper? This affects how the manuscript's empirical claims should be framed at submission.

---

## 4. Bottom Line

The manuscript pair presents a substantive and largely correct geometric reformulation of attention. The mathematical core — KL gauge invariance, mixture-of-sources softmax derivation, untied query-key carving, covariance fixed-point dynamics — is sound. The expository care taken to distinguish *derived* from *interpretive* correspondences is unusually good for a paper of this scope. The issues identified are all of the "tighten the theorem statements, qualify the scope claims, declare the conventions" variety; none undermine the central thesis. A focused major-revisions pass on M1–M5, with a lighter touch on the minor comments, would produce a manuscript ready for full empirical evaluation.

— *Reviewer*

---

## Appendix: Verification log

- `eq:full_kl_general` (`main:1131`): independently re-derived; matches the manuscript's symbolic-verification claim at `main:1138`. The auxiliary identification `r_j = μ_j^\top Σ_j^{-1} μ_j` is correct (cyclic-trace check on `Ω^\top Σ_t^{-1} Ω` with `Ω = U_i U_j^{-1}`, `Σ_t = Ω Σ_j Ω^\top`).
- Cartan-modified bilinear `\tilde g_{ab}` (`supp:590`): numerically positive-definite on `\mathfrak{sl}(K)` for small K (verified K=2). Citation to Knapp Prop. 1.93 is the right reference.
- Style scan: zero banned macros, zero banned Claude-isms.
- Bibliography spot-check: Knapp 2002, Hall 2015, Gallier 2020, Higham 2008, Knapp, Bishop 2006, Murphy 2012, Vaswani 2017, Su 2024, Friston 2010/2017, Amari 1998/2016, Nakahara 2003, Cardy 1996, Goldenfeld 1992, Csiszár 1967 — all present in `references.bib`.
