# Manuscript Index — `Attention/`

**Descriptive file.** Each entry below summarizes *what the manuscript claims*, drawn from the actual `.tex` content as of 2026-05-18. These summaries are claims to be evaluated against the external canon (`external_canon_*.md`), not statements of mathematical truth. The manuscripts evolve; **read the current `.tex` before trusting any of this.**

The agents evaluate manuscript claims against standard literature (Friston, Amari, Nakahara, Vaswani, etc.) — not against the user's own CLAUDE.md or against this file.

## `GL(K)_attention.tex`

**Title:** *Attention as Gauge-Theoretic Variational Inference*. Main JMLR submission. Author: Robert C. Dennis.

**Thesis:** Transformer attention is variational inference over information sources. Each token is a Gaussian agent on a statistical fiber bundle; the attention weight `β_ij = softmax(−KL(q_i ‖ Ω_ij q_j) / τ)` is derived from a single variational principle (minimizing the free energy of a mixture-of-sources generative model). The KL arises exactly; the softmax follows from constrained optimization over the categorical source-selection posterior.

**Recovered-as-limit claims:**
- The `1/√d_k` temperature scaling comes from dimensional concentration of the KL.
- Layer normalization is the geometric condition for frame-independent inference.
- Multi-head attention is a block-diagonal restriction of the gauge algebra.
- Causal masking and positional biases come from non-uniform attention priors `π_j`.
- Two successive limits (isotropic covariances, flat gauge connection) recover `β_ij ∝ softmax(Q_i K_jᵀ / √d_k)`.
- `W_Q, W_K` are gauge transformations with `W_Q W_Kᵀ = σ⁻² Ω⁻ᵀ`.

**Empirical claims to verify:**
- GL(K) gauge transformer on WikiText-103 achieves test PPL 71.6, *no* learned `W_Q/W_K/W_V`, no MLPs, no pointwise activations.
- 1.66× better than standard transformer at matched embedding dim (PPL 118.6 at `d_model = 90`).
- Approaches a parameter-matched standard transformer (PPL 48.5 at `d_model = 1280`).
- Frozen BERT consistency check across 105 passages: grand mean Pearson `r = 0.804`, 95% CI `[0.771, 0.838]`.

**What to verify:**
- F includes the `τ β log(β/π)` entropy term (per `canonical_math.md`).
- Softmax derivation uses canonical F, not the entropy-suppressed surrogate.
- Ω written as `exp(φ_i) exp(−φ_j)`, not `exp(φ_i − φ_j)`.
- Covariance transport uses sandwich `Ω Σ Ωᵀ`.
- `τ = κ√K` (or the standardized `1/√d_k` form) is justified, not asserted.
- Empirical numbers above are reproducible from the codebase and the configs cited.

## `GL(K)_supplementary.tex`

**Title:** *Attention as Gauge-Theoretic Variational Inference — Supplementary Material*. Same author.

**Eight appendices (listed in the opening paragraph):**
- **A.** Standard differential geometry review (for ML readers).
- **B.** Covariance gradient derivation + equilibrium analysis.
- **C.** Gauge-frame gradients for SO(N) and GL(K), including preconditioning strategies.
- **D.** Numerical methods: variational gradient descent on Gaussian and gauge-frame manifolds.
- **E.** Full BERT validation: protocol, results across 105 passages and 5 architectures, Bayesian uncertainty quantification.
- **F.** Renormalization-group universality conjecture + numerical validation.
- **G.** Symmetry-breaking simulations + model-channel formalism.
- **H.** Proof: conditional uniqueness of forward KL divergence via variational duality.

**What to verify per appendix:**
- **B, C, D:** all derivations match what `transformer/core/vfe_gradients.py`, `transformer/core/gauge_preconditioner.py`, and `transformer/core/transport_ops.py` actually compute. Manuscript ↔ code divergence is a finding.
- **E:** the 105-passage BERT validation must be tied to the actual analysis script (likely under `transformer/analysis/`). Configs reported must match what produced the numbers.
- **F:** the RG conjecture was moved to a new appendix section in commit `4bc4de5e` (`pass 5`). Check that the lift is clean — no orphan references in the main body.
- **H:** verify the uniqueness proof is correct (likely a candidate for `sympy` symbolic check on the variational dual).

The user's `project_em_config_best_results.md` memory notes best legacy config is `layernorm + no residual + skip_attention + n_layers=1`. If any "best results" table uses a different config without explanation, flag.

## `belief_inertia_unified.tex`

**Title:** *The Inertia of Belief*. **This is NOT about VFE iteration inertia in the codebase sense.** It's a separate manuscript applying the same gauge-theoretic VFE math to **sociological belief dynamics**.

**Thesis:** Several foundational sociological models (DeGroot social learning, Friedkin-Johnsen opinion dynamics, bounded confidence, echo chambers, Social Impact Theory) emerge as limiting cases of variational free energy minimization on statistical manifolds. Then proposes a Hamiltonian extension: precision as epistemic inertia, reducing to standard Bayesian free-energy descent in the overdamped limit but predicting oscillation, overshooting, and resonance in underdamped regimes.

**Three primary contributions stated in the manuscript:**
1. Second-order belief dynamics where the Fisher information metric supplies the inertial mass tensor: `M = Λ_prior + Λ_observation + Λ_social^in + Λ_social^out`.
2. Application of pullback geometry on statistical manifolds to multi-agent systems, with social coupling as `KL(q_i ‖ Ω_ij q_j)` — same gauge-transport pattern as the main paper.
3. (Stated structure in the manuscript — verify by reading.)

**What to verify:**
- "Inertia" definition is consistent throughout (Fisher-information-based per the abstract; check the body uses this consistently and doesn't drift to "operational" or "KL-second-derivative" without explicit identification).
- The Hamiltonian ansatz reduction to standard FEP in the overdamped limit is actually derived, not asserted.
- Limit-case derivations (DeGroot, Friedkin-Johnsen, etc.) actually reduce to those models under the claimed limits — sympy-verifiable for the linear cases.
- The social coupling `KL(q_i ‖ Ω_ij q_j)` matches the canonical form (`canonical_math.md`). Same hard rules: sandwich product, Ω factorization.

## `Participatory_it_from_bit.tex`

**Title:** *A Gauge-Theoretic Framework Toward a Participatory "It From Bit" Program: Mathematical Foundations and Computational Implementation*. Wheeler / Kant / Friston framing — heavier on philosophy of physics, but with explicit empirical and limitations sections.

**Currently dirty per `git status`** — actively being edited (recent commits `pass 4`–`pass 8`).

**Empirical claims from the abstract:**
- WikiText-103, iso-token budget 122.9M tokens (~1.19 epochs under GPT-2 BPE).
- `K ∈ [10, 120]` swept, 3 seeds per K (except K=90 has 2).
- Fit `PPL = aK^b + c`, per-K seed-mean: `b ≈ −1.05`, 95% bootstrap CI `[−1.10, −1.00]`, `R² ≈ 0.9998` (fit dominated by floor parameter c).
- Restricted `b = −1` model fits at `R² ≈ 0.9996` — exponent statistically indistinguishable from −1 within this sweep.
- Test PPL ≈ 73 at K = 120. *Explicitly noted as well above WikiText-103 multi-layer baselines (PPL 18–25)*; the exponent describes scaling within the single-layer architecture, not SOTA competition. **This honest qualification is good — preserve it.**
- Multi-agent simulation: single-seed, slow subsystem frozen (`γ_ij = 0`), threshold-based meta-agent formation. Multi-seed reproducibility deferred to follow-up.

**Speculative extensions (the manuscript explicitly labels them so — preserve this):**
- Pullback information-geometric construction for cognitive reference frames in the Lahav–Neemeh sense.
- Meta-agent renormalization with a continuous variational criterion not yet validated against the discrete detector.
- 2D linearized `GL(K, ℂ)` example yielding an `SO(1,1)`-compatible indefinite pullback metric under an imposed imaginary temporal generator and a real-part projection.

**Open / explicitly limited (per the manuscript's Section on scope_limitations):**
- 4D nonlinear extension.
- Dynamical selection of the imaginary generator.
- Dimensional analysis between informational and physical units.
- Rigorous quantum extension.
- The gauge-orbit-averaged consensus metric is conditional on a regulator not constructed in the manuscript.

**What to verify:**
- Any "is equivalent to" / "is the same as" claim is either provable or labeled interpretive. The manuscript already does this well in places — extend that discipline wherever it's missing.
- Recent commits `pass 4`–`pass 8` softened over-strong claims (e.g., `pass 7` mentions "measurement-problem softening"). Verify no remaining over-claiming.
- `pass 4` expanded Methods from 7 lines to 4 subsections — verify scope is preserved (no speculation added under "Methods").
- `pass 8` did "alpha_i disambiguation (math reviewer MR-1 partial)" — verify `α_i` is now consistently distinguished from any other α in the paper. Check if MR-1 still has open items.
- Empirical scaling fit parameters and CIs are reproducible from the scaling analysis pipeline in `transformer/analysis/scaling_*`.

## `jmlr_coverletter.tex`

**For:** the JMLR submission of `GL(K)_attention.tex` (or possibly a parallel/earlier draft — see below).

**What to verify:**
- The title in the cover letter is *"Attention, Transformers, and Backpropagation are Degenerate Limits of the Variational Free Energy Principle"* — this differs from the main manuscript title *"Attention as Gauge-Theoretic Variational Inference"*. Either (a) this is intentional (cover letter highlights broader contribution while title is sharper), or (b) one is stale. Flag for the user to resolve before submission.
- Suggested action editors and reviewers (Cuturi, Chiappa, Chang, Mahoney, Rosasco; Cohen, Finzi, ...) are current and reachable.
- "Original submission not based on prior conference work" claim is correct.

## `tikz.tex`

Standalone TikZ source for figures (principal bundle, etc.). Note line 1 is `documentclass{article}` — missing the leading `\` backslash. Likely intentional if this file is `\input{}`-ed elsewhere (the backslash would be supplied by the calling document), but verify. Otherwise it's a typo that would break standalone compilation.

Not subject to peer review proper. Verify:
- Figure captions match figures.
- Variable labels in figures match notation in the main body.

## Recent commit context

From `git log` on `fix/holonomy-numerics-2026-05-05`:

| Commit | Summary |
|---|---|
| `89e7982d` | `pass 8 - alpha_i disambiguation (math reviewer MR-1 partial)` |
| `c4729ddd` | `pass 7 - pan-agentic ontology, measurement-problem softening, Friedman engagement` |
| `ed65254f` | `pass 6 - bundled medium-priority cleanups (5 surgical edits)` |
| `4bc4de5e` | `pass 5 - RG-construction lifted to new appendix section` |
| `87621f54` | `pass 4 - Methods expansion (7 lines to 4 subsections)` |

The `MR-N` convention (math reviewer numbered items) is the user's preferred way to track math-reviewer feedback. Match it in any review output.
