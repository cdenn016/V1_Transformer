# Evidence Pack — ffn-softmax-gradient-correction

## Manuscript references — §5.3 FFN derivation (main paper)

- `Attention/GL(K)_attention.tex:1878` — `\subsubsection{Feed forward Networks and Non-linear Activations.}`. Headline: "the FFN corresponds to the iterative VFE dynamics that adjust beliefs between attention computations. The nonlinear activation function is not an independent architectural choice but a natural consequence of the information-geometric structure."

- `Attention/GL(K)_attention.tex:1885-1890` — Eq. (eq:glu_message): the per-edge mean update `Δμ_i|_j = -η̃ Σ_i β_ij (Ω_ij Σ_j Ω_ij^T)^{-1} (μ_i - Ω_ij μ_j)`. The residual is `e_ij = μ_i - Ω_ij μ_j`.

- `Attention/GL(K)_attention.tex:1900-1907` — **Boxed result Eq. (eq:vfe_glu):** per-edge contribution is a GLU `e_ij · gate(e_ij)`, with gate `= exp(-‖e_ij‖²_{Σ⁻¹}/(2τ) + ⋯) / Σ_k exp(-‖e_ik‖²_{Σ⁻¹}/(2τ) + ⋯)`. This is the headline derivation.

- `Attention/GL(K)_attention.tex:1909-1920` — Binary limit (n_s = 2): Boltzmann reduces to logistic sigmoid. "This recovers the SiLU/Swish structure f(x) = x · σ(g(x)) ... This binary case corresponds to the FFN sublayer."

- `Attention/GL(K)_attention.tex:1922-1934` — GELU/SiLU/ReLU family-membership claim and family-not-functional-identity hedging.

- `Attention/GL(K)_attention.tex:1936-1947` — **"Softmax-gradient correction" paragraph.** Verbatim relevant text:
  > "Beyond the GLU structure of the message-passing term itself, the autograd gradient of the attention-weighted energy `Σ_j β_ij E_ij` contains an additional nonlinear correction from differentiating `β_ij` with respect to `μ_i` (**this term is absent from the gradient of the reduced free energy `F_red` by the envelope theorem, but present in standard autograd implementations**; cf. Section sec:final_free_energy):
  > `∂β_ij/∂μ_i = -(β_ij/τ) [∂D_{KL,ij}/∂μ_i - Σ_k β_ik ∂D_{KL,ik}/∂μ_i]`
  > This centered gradient (deviation of the per-source KL gradient from its β-weighted mean) provides a higher-order non-linearity beyond the first-order GLU gate. ... This correction is unique to the VFE setting, where β depends on the same μ being updated."

- `Attention/GL(K)_attention.tex:1949-1950` — "FFN depth from VFE iterations": "In the full VFE implementation, multiple iterations of gradient descent with recomputed β at each step constitute the analogue of the FFN sub-layer. Each iteration applies the GLU map composed with the softmax-gradient correction."

## Manuscript references — §4.7 envelope-theorem treatment (main paper, the load-bearing precedent)

- `Attention/GL(K)_attention.tex:859-866` — Envelope theorem statement at the reduced free energy:
  > "The envelope theorem guarantees that the gradient of `F_red` with respect to any belief parameter `x ∈ {μ_i, Σ_i, φ_i}` takes the form `dF_red/dx = ∂F/∂x|_{β=β*} = (prior terms) + Σ_j β_ij* ∂E_ij/∂x - (observation terms)`, with no explicit `∂β*/∂x` terms, since the cross-term `Σ_j (∂F/∂β_ij)(∂β*_ij/∂x)` vanishes at the optimal attention where `∂F/∂β_ij = 0`."

- `Attention/GL(K)_attention.tex:868-874` — **Autograd-versus-reduced-free-energy paragraph** with the explicit gap formula. Eq. (eq:autograd_envelope_gap):
  > `∇_x ⟨E⟩_{β*} - ∇_x F_red = -τ⁻¹ Cov_{β*}(E_ij, ∂E_ij/∂x)`
  > "The two objectives therefore share critical points only where this covariance vanishes; this holds at the joint stationary point of `F_red` in `x` and in the high-temperature limit `τ → ∞`, but not generically off-equilibrium. The `∂β/∂x` terms, which vanish in the reduced-free-energy gradient by the envelope theorem, are present in the autograd gradient and provide a softmax-gradient nonlinearity (Section sec:ffn_nonlinearity). Standard transformer training follows the autograd convention (differentiating through the softmax), and we adopt the same convention in the gradient expressions and algorithm below."

- `Attention/GL(K)_attention.tex:967-973` — Belief Dynamics restatement post the canonical-F-vs-surrogate edit (the manuscript's adopted resolution):
  > "The belief dynamics follow gradient descent on the attention-weighted alignment energy at `β = β*`, the autograd-convention form adopted at the close of Section sec:final_free_energy: `∂q_i/∂t = -η_q ∇_{q_i} [Σ_j β_ij* E_ij + α_i D_{KL}(q_i ‖ p_i) - E_{q_i}[log p(o_i | k_i)]]`. This differs from the gradient of the reduced free energy `F_red` (Eq. eq:free_energy_final) by the covariance term `-τ⁻¹ Cov_{β*}(E_ij, ∂E_ij/∂q_i)` of Eq. eq:autograd_envelope_gap; the two vector fields coincide at joint stationary points and differ off-equilibrium by the softmax-gradient nonlinearity identified at Section sec:ffn_nonlinearity."

## Manuscript references — supplementary envelope statement

- `Attention/GL(K)_supplementary.tex:1189-1199` — Supplementary §"Dual Relation via the Envelope Theorem":
  > "At the stationary value `q_i*`, the envelope theorem implies `∂F_i/∂β_ij = D(q_i*, q_j) = D_{KL}(q_i* ‖ Ω_ij q_j)`."

  This is the *other* envelope identity used in the framework — derivative w.r.t. β at stationary `q_i*`. Distinct from the §5.3-relevant `∂F/∂β = 0` at stationary β.

## Code reference — E-step β recomputation

- `transformer/vfe/e_step.py:828` — `for t in range(self.n_e_steps):` — the inner E-step loop. Every iteration recomputes both `μ_{t+1}` and (via `compute_pairwise_omega` and downstream `_iter_nonflat` or `_fused_attention_and_vfe_gradients_block_diag`) a fresh `β_t = softmax(-KL(μ_t)/τ)`. There is no caching of `β` across `μ` updates within a forward pass.

- `transformer/vfe/e_step.py:899-905` — RoPE-full-gauge per-head branch documents the envelope identity in code:
  > `# Envelope identity: at the softmax fixed point of beta, the manuscript F gradient is just sum_j beta * dKL/dtheta — the softmax-coupling term sum_j KL * dbeta/dtheta cancels exactly against the entropy-gradient term tau * sum_j log(beta) * dbeta/dtheta. So when the entropy term is included in F, pass lambda_softmax=0.`
  > `lambda_softmax=0.0 if self.include_attention_entropy else self.lambda_soft,`

  Translation: the codebase's manual-gradient branch chooses to *implement* the envelope form (drops the `KL · dβ/dθ` correction term) when `include_attention_entropy=True`, on the grounds that the correction cancels exactly against the entropy-gradient term *at the softmax stationary point of β*. This is the same envelope-theorem move as §5.3 / §4.7, applied in code.

- `transformer/vfe/e_step.py:1883` — `beta_d_in_graph = beta_g.detach()  # envelope at softmax fixed pt` — comment confirms the trainer's autograd path explicitly detaches β to enforce the envelope-form gradient computationally.

## Canon excerpts (relevant external references)

- **Envelope theorem (standard form, [Milgrom-Segal 2002, Boyd-Vandenberghe 2004 §5.5]).** For a parametric optimization `V(x) = min_y f(x, y)` with interior minimizer `y*(x)`, `dV/dx = ∂f/∂x|_{y=y*}` *provided* `y*(x)` is a stationary point of `f` in `y`. The result is an identity *between gradients at the minimizer*; it makes no statement about the relation between `∇V(x)` and `∇_x f(x, y(x))` for `y ≠ y*`.

- **Mean-field variational inference / VBEM [Wainwright-Jordan 2008 §3, Bishop 2006 §10.1].** The standard coordinate-ascent VB algorithm iterates updates `q_i^{(t+1)} = argmin F[q_1, ..., q_n]` with all `q_{j≠i}` fixed. Each coordinate update *is* an envelope-theorem stationarity move: `q_i^{(t+1)}` is by construction the stationary point in the `i`-th block. The chain `q_i^{(t)} → q_i^{(t+1)}` is therefore a sequence of *block-stationary* iterates, each of which satisfies the envelope identity *for the variables being optimized at that step*. But the entire iteration sequence is not at the joint stationary point until convergence.

- **EM and "inner-loop" stationarity [Neal-Hinton 1998].** The "incremental EM" formulation treats each E-step coordinate update as a stationary block-update. The gradient with respect to the M-step parameters is identified via the envelope theorem *at the E-step stationary point*. The exact same logical structure applies to the user's framework: `β_t = softmax(-KL(μ_t)/τ)` is constructed to be the row-stationary point of the row-Lagrangian for the *current* `μ_t`; the envelope identity `∂F/∂β|_{β=β_t} = 0` holds at every iterate `t`, not just at convergence.

## What this evidence does NOT settle

1. **Whether "β is at its stationary point for the current μ" is the right notion of stationarity for invoking the envelope theorem in the FFN derivation.** The §5.3 derivation operates on per-iteration μ updates. At each iterate `t`, `β_t = softmax(-KL(μ_t)/τ)` *is* the stationary point of the row-Lagrangian for `μ_t`. So the envelope theorem applies at every inner-loop step, not just at convergence. If this reading is correct, the user's "envelope theorem is exact only at the joint stationary point" framing is conflating *joint* stationarity (the only place `(μ, β)` is jointly stationary, where `μ` is also at its fixed point) with *partial* stationarity (β is stationary in `q_i` for fixed `μ`, which holds at every iterate by construction).

2. **Whether the §5.3 derivation actually claims the GLU+Boltzmann-gate is the autograd gradient or the reduced-F gradient.** The manuscript at lines 1936–1947 explicitly states the correction is present in autograd and absent from `F_red`. The boxed Eq. (eq:vfe_glu, line 1903) is derived from Eq. (eq:glu_message, line 1888), which is the autograd-form per-edge contribution `Δμ_i|_j = -η̃ Σ_i β_ij ⋯`. The covariance correction is *also* an autograd-form term that the §5.3 derivation explicitly catalogues at line 1947 as "compositions of these channels within each VFE iteration produces a richer non-linear map than any single channel alone."

3. **Whether a "more careful formulation" that incorporates the covariance correction into the boxed GLU form would be different from what §5.3 already does.** The manuscript already presents BOTH the GLU form (lines 1900-1907) AND the softmax-gradient correction (lines 1940-1947) AND the Σ/φ channel analogues (line 1947) AND identifies the FFN with VFE iterations *that compose all of these* (lines 1949-1950).

4. **The judge will need to decide whether the user's reading — that the envelope theorem is being invoked to *suppress* the covariance correction from the FFN derivation — is supported by the manuscript text. The manuscript text does not suppress it; it labels it and includes it as an additional channel of nonlinearity.**
