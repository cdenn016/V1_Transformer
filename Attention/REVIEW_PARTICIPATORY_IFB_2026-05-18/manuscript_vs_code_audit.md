# Audit Report — `Participatory_it_from_bit.tex` claims vs. code — 2026-05-18

## Scope

Audit the claims in `Attention/Participatory_it_from_bit.tex` (4686 lines) that
purport to map to implementation in this codebase, against the standard
external literature (Friston 2010 / Parr-Pezzulo-Friston 2022; Cover-Thomas
2006; MacKay 2003; Bishop PRML 2006; Vaswani 2017; Nakahara 2003;
Amari-Nagaoka 2000; Blei-Kucukelbir-McAuliffe 2017; Wheeler 1990; Bai-Kolter-Koltun
2019 for IFT). The codebase and the manuscript are both evaluated against
that external standard; neither is treated as canon.

Five specific items in scope (per task):

1. Wheeler/IFB ("observer participation", "measurement collapses the prior",
   "attention is a participatory bit") vs. what the code actually does.
2. Gauge equivariance under observation/conditioning (`Ω Σ Ω^T` transport).
3. Attention as IFB unit: softmax-of-(-KL/τ) form and entropy regulariser.
4. No-NN constraint: do the participatory framings smuggle in MLPs/activations?
5. EM-mode honesty: does the manuscript's "observer updates the prior"
   correspond to a documented `em_mode`?

## Active config (resolved from `transformer/vfe/train_vfe.py`, top dict)

The user's actual click-to-run config differs in important ways from
defaults; every finding below assumes these resolved values:

- `n_layers = 1` (no cross-layer hierarchy exercised at runtime)
- `n_e_steps = 1` (single E-step iteration per forward pass)
- `use_prior_bank = False` (final `nn.Linear(K, vocab_size, bias=False)`
  decoder active — the documented neural exception)
- `norm_type = 'layernorm'` (gauge-blind boundary; emits `UserWarning`)
- `mask_self_attention = False`
- `learnable_kappa = False`, `kappa = 1.0` (constant temperature)
- `active_inference = False` (no pragmatic/epistemic gradient injection)
- `use_rope = True`, `rope_full_gauge = 'off'` (μ-only RoPE; the documented
  RoPE × MahalanobisNorm gap from CLAUDE.md is active, except norm is
  LayerNorm here so the gap is moot for this run)
- `use_non_flat_transport = False` (Regime I flat connection — vanishing
  holonomy by the cocycle identity, manuscript Lemma 3.10)
- `phi_preconditioner = 'killing'`, `gauge_group = 'GLK'`
- `include_attention_entropy = True` (default; the κ-log-β term IS added)
- Default `gauge_parameterization = 'phi'` (Lie algebra parameterisation)
- `irrep_spec = [('fund', 2, 10)]` so `K = 20`, two heads of size 10 each.

Per `EM_CONFIG Best Results` memory the user has empirically converged on
LayerNorm + `n_layers=1` + (legacy `skip_attention=True`); the `vfe/` package
described above does not have a separate attention sublayer at all, so the
`skip_attention` toggle no longer exists in this path.

## Standards used

- [Friston2010] for canonical variational free energy and the
  E-step / M-step factorisation
- [BleiKuckelbirgJordan2017] for the canonical
  "energy minus entropy" Lagrangian derivation of softmax-form variational
  posteriors
- [Vaswani2017] for the standard scaled dot-product
  `softmax(QKᵀ/√d_k)V`
- [Nakahara2003] for tensor parallel transport on
  associated bundles (the `Ω Σ Ωᵀ` sandwich)
- [AmariNagaoka2000] for Fisher information geometry and natural gradient
- [Wheeler1990] for the actual "it from bit" thesis
- [CoverThomas2006] / [MacKay2003] / [Bishop2006] for standard Bayesian
  conditioning (the "measurement" baseline)
- [BaiKolterKoltun2019] for IFT-style gradients through fixed points

## Findings

The verdict taxonomy used below: **exact** = code implements the claim as
stated; **partial** = code implements a documented restriction/specialisation;
**mismatch** = code does something measurably different from the claim;
**not-implemented** = no corresponding code path; **not-applicable** = the
manuscript itself disclaims the claim as interpretive only and there is no
mathematical assertion to audit.

---

### Finding 1 (exact) — Attention β as softmax(−KL / (κ√K))

**CLAIM** (`Participatory_it_from_bit.tex:1104, 1241`):
> "β_{ik} = softmax_k(−KL[q_i ‖ Ω_{ik} q_k] / τ) ... in the working
> implementation the temperature is factorised as τ = κ√K, with κ a learnable
> scalar and the √K factor the dimension scaling familiar from scaled
> dot-product attention"

**CODE EVIDENCE**:
- `transformer/core/attention.py:374-377` —
  `dim_scale = math.sqrt(max(K, 1))`,
  `logits = -kl_matrix / (kappa * dim_scale)`
- `transformer/vfe/attention.py:90-92` (docstring) and `:113-131` (delegate)
- `transformer/vfe/e_step.py:453` — `tau = kappa_attached * self._dim_scale`

**STANDARD REFERENCE**: [Vaswani2017] gives `softmax(QKᵀ/√d_k)`; the
manuscript's `τ = κ√K` is the standard `√d_k` (here `d_k = K`) with an
additional learnable scalar `κ`.

**VERDICT**: exact. The code matches the claim line for line.
The √K is the standard dimensional scaling [Vaswani2017]; the `κ` is the
gauge framework's added learnable temperature. Both are present.

(Caveat: in this user's active config `learnable_kappa = False` and
`kappa = 1.0`, so `κ` plays no role at runtime — `τ` is exactly `√K`.)

---

### Finding 2 (exact) — Attention entropy term `τ·β·log(β/π)`

**CLAIM** (`Participatory_it_from_bit.tex:1234, 1241, 1254`,
`eq:free_energy_functional_final`):
> "β_{ij}(c) ... + τ Σ_{i,j} β_{ij}(c) log(β_{ij}(c) / π_{ij}(c)) ...
> The canonical free energy in this manuscript is the full form above"

**CODE EVIDENCE**:
- `transformer/vfe/e_step.py:440-457`:
  ```
  tau = kappa_attached * self._dim_scale
  entropy_term = (
      tau * (beta_safe * beta_safe.log()).sum()
      + tau * log_N_const * beta.sum()
  )
  ```
  with `log_N_const = math.log(max(beta.shape[-1], 1))`, which is the
  `+τ·log(N)·Σβ` constant that arises from a uniform attention prior
  `π = 1/N`. The combined form is `τ·β·log(β·N) = τ·β·log(β/π)` with
  `π = 1/N`.
- `transformer/vfe/e_step.py:886-900`: when `include_attention_entropy=True`
  (the default), `_update_phi` assembles
  `_F = (β·KL).sum() + κ·√K·(β·log β).sum()` and backpropagates through
  it directly.
- `transformer/vfe/e_step.py:127-135` (in `_compute_aux_alignment_loss` family).

**STANDARD REFERENCE**: This is the standard energy-minus-entropy Lagrangian
form for a variational posterior over a categorical latent
[BleiKuckelbirgJordan2017]; differentiating the row-Lagrangian w.r.t. β with
unit-sum constraint produces the closed-form softmax. The manuscript
derivation at `:1081-1106` is correct and the code implements the resulting
free-energy functional.

**VERDICT**: exact, when `include_attention_entropy = True` (default).
Manuscript and code agree. The branch at `e_step.py:902-907` is an
"entropy-suppressed surrogate" with a documented gradient mismatch
(also discussed at `Participatory_it_from_bit.tex:1241` and `:1278`).

---

### Finding 3 (exact for full covariance, partial for diagonal) — `Ω Σ Ωᵀ` sandwich product

**CLAIM** (`Participatory_it_from_bit.tex:1044`):
> "P(k | z = j) = N(k; Ω_{ij}μ_j, Ω_{ij}Σ_jΩ_{ij}^⊤)"

(and recurring throughout — covariance is transported by the sandwich
product, repeatedly written as `Ω Σ Ωᵀ`)

**CODE EVIDENCE**:
- `transformer/core/transport_ops.py:11-15` (module docstring states the
  invariant)
- `transformer/core/attention.py:835-839` (full-covariance unchunked path):
  ```
  sig_t = torch.einsum(
      'bijkl,bjlm,bijmn->bijkn',
      _Om, _sq, _Om.transpose(-1, -2)
  )
  ```
- `transformer/core/attention.py:865-869` (full-covariance chunked path —
  identical sandwich).
- `transformer/core/attention.py:795-797` (diagonal path):
  `sig_tc = einsum('bijkl,bijkl,bijl->bijk', _Oc, _Oc, sig_j_exp)` ≡
  the diagonal of `Ω diag(σ) Ωᵀ`, which keeps only the diagonal of the
  sandwich (this is the documented diagonal approximation, allowed by
  CLAUDE.md for speed).
- `transformer/core/kl_computation.py:17-19` (kernel docstring states
  "kernels receive already-transported tensors").
- The KL kernel itself is the standard Cholesky-based Gaussian KL
  (`kl_computation.py:264-290`).

**STANDARD REFERENCE**: The standard transformation rule for a (2,0)-tensor
under change of frame is `T → g T gᵀ` (or `g⁻¹ T g⁻ᵀ` depending on the
tensor's covariance class) — [Nakahara2003], [KobayashiNomizu Vol. I §III];
a Gaussian covariance is a (2,0)-tensor on the fibre and transforms by
the sandwich product.

**VERDICT**: exact for the full-covariance path; partial-by-design for the
diagonal-covariance path (the diagonal of the sandwich is computed; the
off-diagonal entries are discarded). The user's active config has
`diagonal_covariance = True` and `exact_diagonal_transport = False`, so
the diagonal approximation is what runs.

---

### Finding 4 (exact) — Value aggregation `μ̂_i = Σ_j β_ij Ω_ij μ_j`

**CLAIM** (`Participatory_it_from_bit.tex:1593, 1815, 1824`):
> "μ̂_i = Σ_j β_{ij} Ω_{ij} μ_j ... identical to the standard transformer
> attention update z_i = Σ_j α_{ij} V_j"

**CODE EVIDENCE**: `transformer/core/attention.py:1010-1012`:
```
w = einsum('bjkl,bjl->bjk', _exp_neg_phi, mu_q)        # exp(-φ_j) μ_j
w_weighted = einsum('bij,bjk->bik', beta, w)            # Σ_j β_ij (...)
mu_aggregated = einsum('bikl,bil->bik', _exp_phi, w_weighted)  # exp(φ_i) (...)
```
This factorises `Σ_j β_ij · exp(φ_i)·exp(-φ_j)·μ_j = Σ_j β_ij Ω_ij μ_j`
without materialising the full pairwise Ω tensor.

**STANDARD REFERENCE**: [Vaswani2017] gives `Σ_j α_{ij} V_j` as the standard
value aggregation (no transport).

**VERDICT**: exact (when the attention sublayer is in use). Note however
that `transformer/vfe/block.py` does NOT call `aggregate_messages` — the
`vfe/` package operates purely through E-step updates to (μ, Σ, φ). The
manuscript's claim that the message-aggregation form is the "value
aggregation" of standard transformers is correct as a mathematical identity
but is exercised by the legacy `core/` attention path, not by the `vfe/`
block running in the user's active config.

---

### Finding 5 (mismatch) — Cross-scale shadow `p_i^{(s)} = Ω_{i,I}[q_I^{(s+1)}]`

**CLAIM** (`Participatory_it_from_bit.tex:2188-2195`,
Eq. `eq:topdown_priors`):
> "p_i^{(s)}(x) = Ω_{i,I}[q_I^{(s+1)}](x), r_i^{(s)}(x) =
> Ω̃_{i,I}[s_I^{(s+1)}](x), where Ω_{i,I} acts on the latent-state fiber ...
> the meta-agent's collective belief, formed through bottom-up aggregation
> from constituent statistics, directly becomes the updated expectation
> (prior) for individual agents"

and `:2218`:
> "Our implementation uses direct prior assignment p_i ← Ω_{i,I}[q_I]
> rather than gradual updates."

**CODE EVIDENCE**: `transformer/vfe/stack.py:85-94`:
```
rho_mu = self.prior_handoff_rho               # = 1.0 (default and user)
if rho_mu == 1.0:
    new_prior_mu = beliefs.mu                  # direct identity copy
else:
    new_prior_mu = (1 - rho_mu) * priors.mu + rho_mu * beliefs.mu
...
rho_sigma = self.prior_handoff_sigma          # = 0.0 (default and user)
if rho_sigma == 0.0:
    new_prior_sigma = initial_priors.sigma     # frozen at embedding
```

What the code does: at the end of each forward-pass *layer*, the posterior
μ becomes the next layer's prior μ; σ stays at the embedding value; φ is
not stored on `priors` at all and flows through `beliefs.phi` only.

What is missing relative to the manuscript claim:

1. **No transport `Ω_{i,I}`** is applied during the handoff. The code is a
   direct identity copy; the claim is `p ← Ω_{i,I}[q_I]`.
2. **The handoff is cross-LAYER (within one batch), not cross-SCALE
   (meta-agent → constituent).** No meta-agent is ever formed in this code
   path. There is no scale index `s`. `n_layers = 1` in the active config,
   so the handoff is not even exercised at runtime.
3. **No bottom-up aggregation** from constituent statistics to a meta-agent
   belief `q_I^{(s+1)}`. The claim presupposes an aggregator (the
   threshold-detector of §4.3, with `Γ_min = 0.5`, `N_min = 2`) which is
   not present in the repo (Finding 7).

**STANDARD REFERENCE**: [Friston2010] hierarchical message passing
specifies upward error and downward prediction with explicit precision
weighting; the manuscript's formula is internally consistent with that
schema, but the code's flat layer-to-layer μ pass is not.

**VERDICT**: mismatch. The "cross-scale shadow" structure described in
§3.5 and the "self-referential closure" of §4.5 (line 2211-2216) are not
realised by any code I located. The closest thing in the repo is the
within-batch cross-layer μ handoff in `vfe/stack.py`, and even that does
not apply the gauge transport `Ω_{i,I}` the manuscript prescribes.

(Open question: a parallel "non-Markovian Ouroboros Tower" pseudocode is
described at `:2197-2209`. If a private simulator exists outside the repo,
the user should attach it; if not, this whole section is theoretical.)

---

### Finding 6 (exact) — No NN in the default VFE path

**CLAIM** (`Participatory_it_from_bit.tex:120, 133`):
> "the resulting architecture is a working language model trained without
> learned attention projections, MLPs, or pointwise activation functions"

**CODE EVIDENCE**:
- `transformer/vfe/model.py:6`: docstring "No nn.Linear in default config —
  PriorBank IS the decoder."
- Grep for `nn.Linear|nn.MLP|nn.GELU|nn.ReLU|nn.Sequential|nn.LayerNorm`
  in `transformer/vfe/` returns three hits:
  - `block.py:47`: `nn.LayerNorm(dim)` — gauge-blind ablation
    (emits `UserWarning` on construction)
  - `model.py:75`: `nn.Linear(K, vocab_size, bias=False)` — the documented
    output-projection exception (active in this config because
    `use_prior_bank = False`)
  - `block.py:27`: the `_LayerNormSigmaAdapter` wrapper for LayerNorm.
- No MLPs. No GELU/ReLU/SiLU.

**STANDARD REFERENCE**: [Vaswani2017] uses `W_Q, W_K, W_V, W_O`, two-layer
FFN with GELU, and pre/post-LayerNorm. None of those (except the documented
LayerNorm exception and the documented linear output decoder) are in this
codebase's default config.

**VERDICT**: exact (with two documented exceptions). The user's active
config DOES use both exceptions: `nn.Linear` decoder (because
`use_prior_bank = False`) and `nn.LayerNorm` (because
`norm_type = 'layernorm'`). The manuscript's claim is "no learned attention
projections, no MLPs, no pointwise activations" — none of those are present.
The output linear projection and LayerNorm are explicitly disclaimed by the
manuscript as separate from the gauge-theoretic core.

(Note: the manuscript at `:133` writes "trained without learned attention
projections, MLPs, or pointwise activation functions" without listing the
linear output projection as an exception. The companion
`GL(K)_attention.tex` is more explicit. Manuscripts should harmonise on
this.)

---

### Finding 7 (not-implemented) — Ouroboros Tower simulation of §6

**CLAIM** (`Participatory_it_from_bit.tex:2239-2502`, §6 Results):
The entire §6 (Results) describes a single-seed simulation with:
- 8 initial agents, max 25 scales, hyperprior depth 5, decay γ=0.5
  (Table 2 at `:2266-2286`),
- η_μq=0.05, η_Σq=0.0075, η_μp=0.02, η_Σp=0.0075,
- A "reorganisation event around step 150" with ~520× variance spike, ~28×
  gradient variance spike, NE score crossing 0.5,
- Phase I (steps 0-140) "smooth descent",
- Phase II (steps 140-160) "explosive fluctuations",
- Phase III (steps 160-200) "hierarchical condensation",
- A final 13-scale hierarchy at step 200 with 173 agents,
- Figs 4-8 (energy flow, energy landscape, non-equilibrium indicators,
  condensation bubble chart, hierarchy graph).

**CODE EVIDENCE**: I searched the repo for:
- File patterns: `**/ouroboros*.py`, `**/meta_agent*.py`,
  `**/cross_scale*.py`, `**/hierarchical*.py`, `**/participat*.py` — no
  matches.
- Identifiers: `Ouroboros`, `meta_agent`, `MetaAgent`,
  `threshold_consensus`, `hyperprior_depth`, `hyperprior_decay`, `Γ_min`,
  `N_min` — no Python file matches (the strings appear only in
  `.tex` and reviewer-history `.md` files).
- Figure files: `Fig_4.png` through `Fig_8.png` are not present under
  `Attention/figs/` or anywhere in the repo. Only `fig_scaling_main.png/pdf`
  (the WikiText-103 scaling figure, Finding 8) exists.

**STANDARD REFERENCE**: not applicable — this is a missing-implementation
finding, not a standard-vs-novel question.

**VERDICT**: not-implemented (under the present repo). The §6 simulation
that the manuscript calls "Level 2: Mathematical Implementation" in its
own Epistemic Status (`:122`) and "demonstrated computationally that ..."
in its Summary (`:2492-2501`) has no source file or output artefact I
could locate. The manuscript's "we have constructed and computationally
implemented participatory dynamics" cannot be verified from this repo.

(Open question: does the Ouroboros simulator live in a separate repo? If
so, the manuscript's `\code` URL or the `Data Availability` /
`Code Availability` statement should point there. As of this audit, both
are placeholder text and the GitHub URL at `:173` points to a single
combined repo.)

---

### Finding 8 (exact) — WikiText-103 multi-seed scaling validation

**CLAIM** (`Participatory_it_from_bit.tex:2503-2519`):
> "multi-seed scaling study on WikiText-103 ... b = -1.049 ([-1.103, -0.998]),
> c = 61.17, R² ≈ 0.9998 ... Reproduced from
> publication_outputs/scaling_analysis/"

**CODE EVIDENCE**:
- `transformer/train_publication.py` exists and is referenced.
- `Attention/figs/fig_scaling_main.pdf` and `.png` exist.
- The manuscript's `\section{Methods}` (line 3603-3624) describes the
  protocol and the user's `transformer/vfe/train_vfe.py` is the
  click-to-run entry point.

**STANDARD REFERENCE**: [Kaplan2020], [Hoffmann2022] for the comparison
context.

**VERDICT**: exact for the claim of having run a scaling sweep
(the figure exists). I did not re-fit the perplexity numbers; that is
the experiment-analyst's territory, not a manuscript-vs-code audit.

---

### Finding 9 (not-applicable) — "Measurement as pre-consensus dynamics" / quantum collapse analogy

**CLAIM** (`Participatory_it_from_bit.tex:3203-3224`,
"An Inferential-Consensus Analogy for Measurement (Speculative)"):
> "Measurement is the dynamical process by which an apparatus ... couples
> to the particle and dominates the free energy landscape, forcing all
> observers into agreement. What physics calls 'wavefunction collapse' is
> modelled by the transition from pre-consensus ... to post-consensus."

**The manuscript itself disclaims this** at `:3224`:
> "The analogy is suggestive only. The framework contains no
> quantum-mechanical formalism — no Hilbert space, no Born rule, no
> superposition states — so it cannot derive the measurement problem's
> resolution. ... We do not claim a resolution of the measurement problem."

**VERDICT**: not-applicable. The manuscript explicitly labels this as
analogy, not derivation. The codebase has no Hilbert-space machinery, no
Born rule, no superposition representation; there is nothing to audit.

(This finding is recorded so it is on the record that the audit DID check.)

---

### Finding 10 (not-applicable) — "Participatory bootstrap" / "self-excited circuit"

**CLAIM** (`Participatory_it_from_bit.tex:2438-2454`, 2456,
"Reality Participates in Its Own Construction"):
> "(1) Agents form meta-agents through consensus (bottom-up)
> (2) Meta-agents define new priors for constituents (top-down)
> (3) Updated priors change free energy landscape
> (4) Agents respond to new landscape by updating beliefs
> (5) Updated beliefs change induced metric G(t) = σ*g_B
> (6) Changed metric alters geodesics and transport ..."

The same pages disclaim at `:2440`:
> "The participatory loop provides a mathematical model compatible with
> Wheeler's self-excited-circuit metaphor under the demonstration regime
> described above. Whether this loop bears on physical reality is
> interpretive and beyond the formal results."

and at `:2462`:
> "The participatory dynamics described here are a property of the
> mathematical structure under the demonstration regime ... we offer this
> as an interpretive reading rather than a derivation."

**VERDICT**: not-applicable as a code claim. The interpretive Wheeler/
self-excited-circuit framing is properly labeled in the manuscript. The
substantive code claims that DO appear (cross-scale shadow, meta-agent
formation, induced-metric back-action) are evaluated under Findings 5 and 7.

The induced-metric pullback step (5) — "G(t) = σ*g_B" — is the Fisher
pullback developed in §4.2. **That construction is also not implemented in
the code**: the `vfe/` package operates in the 0-D limit
(`Participatory_it_from_bit.tex:1575`: "All agents exist at a single point
x_0"), where the base manifold is a point and the pullback is degenerate.
This is not a hidden flaw — the manuscript explicitly says so at `:1575`
and `:991` — but it should be visible in any "Wheeler" interpretive
discussion: step (5) of the loop has no code.

---

### Finding 11 (mismatch) — Per-token frame `U_i` and RoPE identification

**CLAIM** (`Participatory_it_from_bit.tex:1708-1709`):
> "The natural identification of the per-token frame U_i with a real
> transformer architecture is the per-position rotational frame of rotary
> positional embeddings, in which U_i ∈ O(d_k) is a block-diagonal rotation
> depending on token position. In that case U_i^{-1} = U_i^T and the
> closure Σ_j = U_j C U_j^T holds automatically for any C commuting with
> the rotation block structure."

The claim is that RoPE acts as the gauge frame `U_i` and therefore that
covariance Σ_j is transported by `R(θ_j) Σ R(θ_j)^T` in the attention KL.

**CODE EVIDENCE**: `transformer/core/transport_ops.py:92-155` (`_apply_rope`)
applies the rotation to μ only. The function's own docstring (lines 101-122)
flags the mismatch explicitly:

> "INTERPRETATION (important). This implementation rotates only μ, not
> the covariance Σ. ... This function does *not* implement that — it
> follows the standard-transformer Q/K rotation pattern (rotate the means
> used for scoring, leave the covariance untouched). The result is that
> the forward-pass KL is computed with rotated μ but raw Σ, which is
> neither the manuscript's full gauge interpretation nor a textbook KL
> between rotated Gaussians."

The framework-consistent `R Σ Rᵀ` exists as an opt-in
(`_apply_rope_to_covariance` at `transport_ops.py:158-209`) gated by
`BlockConfig.rope_full_gauge`, which is `'off'` in the active config.

**STANDARD REFERENCE**: [Su2024RoPE] is the original RoPE paper; it
operates on Q and K (means) and does not touch covariance — RoPE has no
notion of covariance. The manuscript's identification of RoPE with the
gauge frame U_i extends RoPE beyond its original scope.

**VERDICT**: mismatch between the manuscript's identification of `U_i` with
the RoPE rotation (which requires `Σ → R Σ Rᵀ`) and the codepath used
under the active config (μ rotated, Σ raw). This mismatch is acknowledged
in the code docstring (so the engineering is honest about it) but not in
the manuscript section that asserts the identification (`:1708-1709`).
Recommend the manuscript add a sentence at `:1708` flagging that the
implementation rotates μ only.

---

### Finding 12 (partial / "claimed-standard, actually-novel") — Forward KL is not a "natural metric on a statistical manifold"

**CLAIM** (`Participatory_it_from_bit.tex:1108, 1241`):
> "within the f-divergence class satisfying assumptions (i)–(iii) of
> Appendix [conditional_uniqueness], the forward KL divergence is the
> unique f-divergence that yields a consistent dual interpretation for
> the attention weights"

This is described as a uniqueness theorem rather than the standard
information-geometric statement (which is [Cencov1972]:
the Fisher information metric is the unique Riemannian metric on a
statistical manifold invariant under sufficient statistics).

**STANDARD REFERENCE**: [Cencov1972], [AmariNagaoka2000]. The standard
uniqueness theorem in information geometry is about the Fisher metric,
not about choice of f-divergence. The manuscript's conditional
uniqueness theorem (`Participatory_it_from_bit.tex:4265-4453`) is a
novel construction with its own assumption set (locality, linear
coupling, exponential-family closure).

**VERDICT**: not a mismatch but a labelling concern. The forward KL choice
is properly disclaimed at `:1034` ("post-hoc justification") and at
`:1108`, and the code implements the forward KL consistently. The novelty
of the conditional-uniqueness theorem is correctly attributed to the
appendix. This is recorded as a note rather than a finding because the
manuscript is transparent about the post-hoc nature.

---

### Finding 13 (not-implemented) — EM-mode-honesty: "the observer updates only the prior" has no implementation

The task asks whether any manuscript claim of the form "the observer
updates only the prior" maps to a documented `em_mode` variant.

**MANUSCRIPT REVIEW**: I searched the manuscript for explicit claims of
this form. The closest are:

- `Participatory_it_from_bit.tex:2218`: "p_i ← Ω_{i,I}[q_I]" — direct prior
  assignment (covered under Finding 5).
- `:2222-2227`: tracking "prior change Δp_i" via top-down feedback.
- §3.5.6 "Hierarchical Structure" generally.

None of these specify a gradient-flow profile on the EM boundary in the
sense of CLAUDE.md's four `em_mode` variants (`ift_phi`, `em_phi_p`,
`em_phi_q`, `vfe_default`).

**CODE EVIDENCE**: CLAUDE.md notes the `vfe/` package "does not currently
honor `em_mode` switching — it is hardwired to this profile":
- μ_p attached (previous layer's posterior μ_q becomes next layer's μ_p)
- σ_p structurally frozen at embedding value (`prior_handoff_sigma = 0`)
- φ updated each E-step iteration via `_update_phi`, with a fresh
  detached leaf `phi_for_grad = phi.detach().requires_grad_(True)`.

So at runtime the user's `vfe/` configuration is **none of** the four
documented `em_mode` strings. It is a fifth, package-specific profile.

**VERDICT**: not-implemented as a manuscript-claim. The manuscript does not
make a precise EM-gradient-flow claim, and the codebase silently runs a
profile that does not match the `em_mode` enumeration in CLAUDE.md. This
is a documentation gap rather than an audit finding against the
manuscript, but it is worth surfacing.

---

## Summary table

| # | Manuscript claim | Code path | Verdict |
|---|---|---|---|
| 1 | β = softmax(-KL/(κ√K)) | `core/attention.py:374-377` | exact |
| 2 | F includes τ·β·log(β/π) entropy term | `vfe/e_step.py:440-457, 886-900` | exact |
| 3 | Σ transport via Ω Σ Ωᵀ sandwich | `core/attention.py:835-839, 865-869` | exact (full); diag-approximation (diagonal) |
| 4 | μ̂_i = Σ_j β_ij Ω_ij μ_j | `core/attention.py:1010-1012` | exact (when sublayer used) |
| 5 | Cross-scale shadow p ← Ω_{i,I}[q_I] | `vfe/stack.py:85-94` | mismatch |
| 6 | No NN in default | `vfe/model.py:75`, `block.py:47` | exact (with documented exceptions) |
| 7 | Ouroboros Tower simulation | no file | not-implemented |
| 8 | WikiText-103 scaling fit | `train_publication.py`, `fig_scaling_main.pdf` | exact |
| 9 | Measurement as pre-consensus | (analogy only) | not-applicable |
| 10 | Participatory bootstrap loop | (interpretive; cf. 5, 7) | not-applicable |
| 11 | RoPE = U_i with Σ → R Σ Rᵀ | `transport_ops.py:92-155` (μ only) | mismatch |
| 12 | Forward KL "uniquely consistent" | (manuscript-internal, properly disclaimed) | not-applicable |
| 13 | EM mode for "observer updates prior" | `vfe/` runs a 5th profile | not-implemented as claim |

---

## Severity tags (auditor's reading)

- **Critical (1)**: Finding 7 — the §6 Results section is unverifiable
  from this repo. The Epistemic Status at `:122` classifies §6 as
  "Level 2: Mathematical Implementation"; absent the simulator code, that
  classification cannot be sustained. The manuscript already softened the
  language to "single illustrative run" and added the multi-seed-not-yet
  caveat (an improvement over earlier drafts), but the load-bearing
  quantitative claims (Phases I/II/III, 520× variance, 13-scale hierarchy)
  cannot be replicated.
- **Major (2)**: Findings 5 and 11 — the cross-scale shadow `Ω_{i,I}` is
  not in the layer handoff, and the RoPE-as-`U_i` identification omits
  the `Σ → R Σ Rᵀ` half of the gauge transport. Both are not flagged in
  the manuscript at the points where the identification is asserted.
- **Note**: Findings 6 (the `nn.Linear` decoder is active in the user's
  config and the manuscript could be more explicit about it), 12, 13.
- **Confirmed-exact (5)**: Findings 1, 2, 3 (full-cov), 4, 8 — the core
  attention construction, the entropy regulariser, the sandwich product,
  the value aggregation, and the scaling experiment are implemented as
  claimed.

## What was NOT audited (out of scope)

- The Lorentzian-signature construction of §5 (speculative; out of audit
  scope per the user task).
- The pullback construction of §4 in any non-0D regime (the code is
  0-dimensional only).
- The `transformer/core/variational_ffn.py` legacy path (only the active
  `transformer/vfe/` path was traced).
- Numerical correctness of the actual scaling fit; that is an
  experiment-analyst question, not a manuscript-vs-code audit.
- Gradient correctness inside `_update_phi` beyond the autograd-envelope
  identity that is already audited by other reviewers
  (see `Attention/REVIEW_GLK_2026-05-18/`).

## Recommended follow-up

1. **Manuscript edit**: insert at `:2188` and `:2218` a one-sentence
   "Implementation note: in the present repository the cross-scale
   handoff is collapsed to the within-batch cross-layer μ identity
   (Section X of the companion paper); the Ω_{i,I} transport in
   Eq. (4.3) is not exercised at runtime." This is what the code
   actually does.
2. **Manuscript edit at `:1708`**: insert "in the present implementation
   only μ is rotated; the experimental `rope_full_gauge` flag enables
   the framework-consistent Σ rotation."
3. **Manuscript edit at §6**: either move the simulator into the public
   repo (so the Level-2 classification is supportable from artefacts the
   reader can inspect) or downgrade §6 to Level-3 ("speculative
   demonstration in private code, not reproducible from this repository").
4. **Manuscript edit at `:133`**: enumerate the two documented exceptions
   (the `nn.Linear` decoder under `use_prior_bank=False` and the
   `nn.LayerNorm` ablation under `norm_type='layernorm'`).
5. **CLAUDE.md edit (project policy, not manuscript)**: either add the
   `vfe/` package's hardwired gradient profile as a fifth `em_mode` entry,
   or document explicitly that `em_mode` only applies to the legacy
   `core/variational_ffn.py` path.

## Open questions for the user

- Where does the Ouroboros Tower simulator live? Is there a separate
  private repo, or has the simulation code been removed?
- Should §6 be retained at Level 2 (Mathematical Implementation) without
  the simulator code being publicly inspectable?
- Is the `vfe/stack.py` cross-layer handoff the intended implementation
  of the cross-scale shadow Eq. (`:2188`)? If so the manuscript should
  acknowledge that no transport is applied; if not, then the cross-scale
  shadow has no code in this repo and §3.5 needs the same Level-3
  reclassification as §6.
