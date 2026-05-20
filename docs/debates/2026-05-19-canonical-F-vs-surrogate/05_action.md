# Action — canonical-F-vs-surrogate

**From verdict:** RED_WINS

## Recommended action

The verdict's action item is **editorial reconciliation** of three loci in `Attention/GL(K)_attention.tex`:

1. **`:967` (Belief Dynamics paragraph).** "The global free energy is minimized by gradient descent on the beliefs `∂q_i/∂t = -η_q δF/δq_i`." This prose claims descent on F itself. The implementation at `:874` adopts the autograd convention, which descends on `⟨E⟩_{β*}`, not on F_red. The two are non-equivalent off-equilibrium.

2. **`:2008` (E-step/M-step paragraph).** "E-step (Belief Inference): Given current parameters, update agent beliefs `{q_i}` via natural gradient descent on the free energy: `q_i^(t+1) ← q_i^(t) - η_E ∇̃_{q_i} F|_{q_i = q_i^(t)}`." Same issue: prose says descent on F; implementation descends on `⟨E⟩_{β*}`.

3. **`:874` (autograd-convention adoption).** This locus is the disclosure that the implementation differentiates the surrogate, not F. The disclosure is honest but is editorially isolated from the two descent-on-F prose claims above.

### Two reconciliation options

**Option α — adopt the autograd-convention framing throughout.** Rewrite `:967` and `:2008` to describe the implementation as descent on `⟨E⟩_{β*}` (the attention-weighted alignment energy at fixed β = β*), with `F_red` reserved for the analytical stationary-point characterization. This matches what the implementation actually does and is consistent with `:874` and the supplementary §B.1 working form.

**Option β — supply a first-principles derivation that the surrogate IS the correct variational target.** This would require showing that `⟨E⟩_{β*}` is a variational objective whose minimization corresponds to a meaningful inference problem (e.g., a coordinate-ascent fixed point of a related joint objective). The previous `softmax-beta-stationary-point` debate verdict suggests this route is high-effort, since F_align^(τ) itself was concluded to be engineered rather than ELBO-derived.

**Recommended:** Option α. It matches the manuscript's own line-874 adoption, is consistent with the supplementary's working form, and avoids further FEP-derivation commitments.

### Specific edit proposals

**At `:967`** (Belief Dynamics), replace:

> "The global free energy \eqref{eq:free_energy_final} is minimized by gradient descent on the beliefs `{q_i}`: ∂q_i/∂t = -η_q δF/δq_i"

with:

> "The belief dynamics follow gradient descent on the attention-weighted alignment energy at fixed β = β*: ∂q_i/∂t = -η_q ∇_{q_i} ⟨E⟩_{β*} = -η_q ∇_{q_i} (Σ_j β_{ij}* E_{ij} + Σ_i α_i D_KL(q_i ‖ p_i) - E_{q_i}[log p(o_i)]). This is the autograd-convention gradient (cf.\ §\ref{sec:final_free_energy}); it differs from ∇F_red by the covariance term `-τ⁻¹ Cov_{β*}(E_{ij}, ∂E_{ij}/∂q_i)` of Eq.~\ref{eq:autograd_envelope_gap}, which is the softmax-gradient nonlinearity identified at §\ref{sec:ffn_nonlinearity}."

**At `:2008`** (E-step), replace:

> "E-step (Belief Inference): Given current parameters, update agent beliefs `{q_i}` via natural gradient descent on the free energy"

with:

> "E-step (Belief Inference): Given current parameters, compute the closed-form softmax β = β* at the current beliefs (the stationary point of F_align in β, Eq.~\ref{eq:mixture_softmax_general}), then update agent beliefs `{q_i}` via natural gradient descent on the attention-weighted alignment energy at β = β*"

with the same Eq.~\ref{eq:autograd_envelope_gap} cross-reference to make the autograd-convention adoption explicit.

### Companion editorial cleanup

The manuscript's `:874` disclosure is excellent; the editorial gap is at the descent-on-F prose elsewhere. After the two specific edits above, optionally add a one-paragraph reconciliation in §3.5 ("Full Variational Free Energy") that explicitly states: F_red is the variational objective characterized by Theorem~\ref{thm:glk_invariance} and the softmax-β stationarity at §4.6, but the implementation descends on `⟨E⟩_{β*}` per the autograd convention adopted at `:874`; the two coincide at joint stationary points and differ by the softmax-gradient nonlinearity off-equilibrium.

## Cross-debate observation

The verdict notes: "the previous softmax-beta-stationary-point verdict (RED_WINS) reinforces: F_align^(τ) is engineered, F_red is its reduction, and the implementation descends on a third functional `⟨E⟩_{β*}`."

The cumulative picture across the two related debates:
1. **Softmax-β debate:** F_align^(τ) is an engineered soft-assignment Lagrangian (not a joint-FEP-derived ELBO).
2. **Canonical-F-vs-surrogate debate:** the implementation descends on the surrogate `⟨E⟩_{β*}`, not on F_red = -τ log Z_i.

Combined: the framework's analytical machinery characterizes F_align^(τ) and F_red; the implementation runs on `⟨E⟩_{β*}`. The manuscript explicitly states this at `:874` but reverts to descent-on-F language at `:967` and `:2008`. The editorial reconciliation above addresses the second-debate finding directly.

## Follow-up debates (if any)

Two open queue items remain from the prior 05_action.md files:

1. **Multi-head = block-diagonal GL(K)** structural correspondence (§5.4). The rectangular-projection caveat at `:1720` intersects the first debate's verdict; a focused sub-debate would test whether the thin-SVD lift is a structural correspondence or a chosen factorization.

2. **Route 1 (untied carving) alone reduces to Vaswani §3.2.1** (§5.2.1). Blue's strongest unrefuted move in the first debate; remains open.
