# Fresh Code Audit — Four Manuscript-vs-Code Claims — 2026-05-18 (night)

## Scope and method

Targeted re-audit of four claims raised by prior verifier reports
(`verifier_report_2.md`, `manuscript_vs_code_audit.md`) where the
verification required reading the live codebase against the manuscript.
For each item below I name the file(s) inspected with line ranges, give
the literal code, and a verdict among:

- **CONFIRMS-MANUSCRIPT** — code path implements what the manuscript
  claims.
- **REFUTES-MANUSCRIPT** — code path implements something measurably
  different from what the manuscript claims.
- **CANNOT-VERIFY-NO-CODE** — the code path the manuscript describes
  does not exist in the repository in any form.

External standards used: [Vaswani2017] for scaled dot-product
attention, [BleiKuckelbirgJordan2017] / Bishop PRML for the standard
Gaussian KL closed form, [Friston2010] / [Parr-Pezzulo-Friston2022]
for variational free-energy and active-inference baselines.

Project working tree state at audit time: branch
`fix/holonomy-numerics-2026-05-05`; `transformer/vfe/train_vfe.py`
modified locally (active config dict resolved below where relevant).

---

## AUDIT 1 — Detector form §2115 vs §3613 in Participatory_it_from_bit.tex

### Claim being checked

Manuscript §2115 (line 2107-2114, with the boxed rule at line 2114):
the meta-agent consensus detector is
`Γ({i}, x) = P({i}, x) · C_q({i}, x) · C_s({i}, x) ∈ [0, 1]`, where
`C_q = exp(-V_q/τ_q)`, `V_q` is the average post-transport pairwise
KL, `Γ_min = 0.5`, `N_min = 2`. The detector is a multiplicative
bounded-exponential surrogate for the variational free-energy
improvement criterion.

Manuscript §3613 (line 3613, inside Methods):
"When a cluster of agents achieves both belief consensus
(`KL(q_i || Ω_ij[q_j]) < τ_KL = 0.05`) and prior consensus
(`KL(p_i || Ω_ij[p_j]) < τ_KL`), it is treated as having undergone
epistemic death."

These are two mathematically distinct detectors. §2115 is a smooth
multiplicative-Gibbs product over bounded coherences; §3613 is a hard
per-pair raw-KL threshold on `q` and `p` simultaneously.

### Files inspected and search results

The full codebase was searched for any implementation of either form:

- `Grep "tau_KL|Gamma_min|epistemic_death|consensus_detector|meta_agent"`
  on the entire repo: matches only inside the manuscript itself and
  inside reviewer reports under `Attention/REVIEW_*/`. ZERO code
  matches.
- `Grep "tau_KL|Gamma_min|N_min|gamma_min|n_min"` on all `.py` files:
  ZERO matches.
- `Grep "C_q|C_s|tau_q|tau_s"` on all `.py` files: ZERO matches.
- `Grep "epistemic_death|meta-agent|MetaAgent|epistemic death"` on
  `.py`: the only `meta-agent` matches are in `transformer/visualization/
  belief_space_viz.py` and `transformer/visualization/
  interactive_belief_viz.py`, which print the string "Evidence FOR
  meta-agent hypothesis" as a sanity-check banner; no detector code.
- `Grep "epistemic"` on `.py`: the only matches are EFE-related code
  (`epistemic_weight`, `epistemic_samples` in
  `transformer/core/active_inference.py`,
  `transformer/core/expected_free_energy.py`,
  `transformer/vfe/efe.py`, `transformer/vfe/active_inference.py`).
  These implement the BALD-style mutual-information epistemic-value
  term of standard active inference [Friston2017], NOT the
  "epistemic death" clustering rule of §2115 / §3613.
- `Glob "**/simulator*.py"`, `Glob "**/multi_agent*.py"`,
  `Glob "**/ouroboros*.py"`, `Glob "**/participatory*.py"`: NO files.
- `Grep "scale|s\+1|s\+\+|next_scale|parent_scale|cross_scale"` on
  `transformer/**/*.py`: matches are unrelated (axis scales for
  plotting, `connection_init_scale`, etc.). No multi-scale `s → s+1`
  code path.

The manuscript itself acknowledges this. Line 2217:

> The transformer codebase referenced in the abstract is a separate
> code path with its own cross-layer prior handoff (an identity-copy
> with damping, not a multi-scale transport) and should not be read
> as the simulator implementation of the present subsection.

And line 3641-3642:

> The implementation, including the gauge-theoretic active-inference
> simulator and the multi-seed scaling-validation pipeline, will be
> released at https://github.com/.../Participatory-It-From-Bit-Universe
> upon publication.

### Verdict

**CANNOT-VERIFY-NO-CODE.** Neither the §2115 form
(`Γ = P · C_q · C_s`) nor the §3613 form (raw-KL pair threshold) is
implemented anywhere in the repository. The §2115 / §3613 mismatch
cannot be reconciled by appeal to "code disambiguates" because no
code embodies either form. This finding is therefore a pure
manuscript-internal inconsistency, the resolution of which is a
modelling decision by the author, not a code-derived fact.

The "epistemic" code that does exist
(`transformer/core/active_inference.py`,
`transformer/core/expected_free_energy.py`,
`transformer/vfe/efe.py`) implements BALD mutual-information
epistemic-value gradients in the E-step; it shares no semantics
with the §2115 / §3613 cluster-formation rule.

### Recommended manuscript edit

Pick ONE detector form and use it consistently in both §2115 and
§3613. Recommended choice: the §2115 multiplicative form, because
(a) it is bounded in `[0, 1]` and links cleanly to the Gibbs form
the rest of the manuscript uses, (b) §2114 itself notes
"`Γ_min = 0.5, N_min = 2`" as the implementation constants. Then
rewrite §3613 to read:

> "Every two steps, the consensus detector of
> Section~\ref{sec:meta_agent_threshold} is evaluated: a cluster is
> retained as a candidate meta-agent when
> $\Gamma(\{i\}, x) > \Gamma_{\min} = 0.5$ and $|\{i\}| \geq N_{\min} = 2$.
> The slow subsystem $(s_i, r_i)$ is frozen, so the model-coherence
> factor $C_s$ is approximated by the prior-coherence proxy
> $C_p({\{i\}}, x) = \exp[-V_p({\{i\}}, x)/\tau_p]$ with
> $V_p = |\{i\}|^{-2} \sum_{i,j} \mathrm{KL}(p_i \| \Omega_{ij}[p_j])$.
> The hard pair-threshold reading
> $\mathrm{KL}(q_i \| \Omega_{ij} q_j) < \tau_{\mathrm{KL}}$
> appearing in earlier drafts of this section was a simplification
> and is not the rule used by the simulator; the simulator uses the
> multiplicative form above."

Alternatively, if the simulator code (when released) actually uses
the §3613 pair-threshold form, then §2115 should be downgraded from
"the simulations of this paper use" to "an alternative bounded
detector" and §3613 should be the canonical rule.

Either way, the manuscript should not present BOTH forms side-by-side
without flagging which one the released simulator implements,
because the verifier currently has no empirical means to discriminate
(repo is private at the GitHub URL given, returns HTTP 404).

---

## AUDIT 2 — Gaussian KL closed form (Disc MR-25)

### Claim being checked

Manuscript Participatory line 3923 (Appendix "Mathematical Details",
"Gaussian KL Divergence"):

```
KL(q || p)
  = (1/2)[log(|Σ_p|/|Σ_q|) + tr(Σ_p^{-1} Σ_q)
          + (μ_p - μ_q)^T Σ_p^{-1} (μ_p - μ_q) - K]
```

Manuscript line 4002-4014 (Appendix "Covariance Dynamics", restated):

```
KL(N(μ_1,Σ_1) || N(μ_2,Σ_2))
  = (1/2)[log(|Σ_2|/|Σ_1|) + tr(Σ_2^{-1} Σ_1)
          + (μ_2 - μ_1)^T Σ_2^{-1} (μ_2 - μ_1) - d]
```

These are the two displayed Gaussian KL formulas in the manuscript.
Question: does `transformer/core/kl_computation.py` (and any analog
in `transformer/vfe/`) implement the same formula literally?

### Files inspected

`transformer/core/kl_computation.py`:

- Docstring of `_kl_kernel_dense` at lines 141-145:
  ```
  KL(N(μ_q, Σ_q) || N(μ_t, Σ_t))
    = (1/2)(tr(Σ_t^{-1} Σ_q)
            + (μ_t - μ_q)^T Σ_t^{-1} (μ_t - μ_q)
            - K + log|Σ_t| - log|Σ_q|)
  ```
- Implementation at line 301:
  ```python
  kl = 0.5 * (trace_term + mahal_term - K + logdet_p - logdet_q)
  ```
  where `mahal_term = ‖L_p^{-1} (mu_t - mu_q)‖²`,
  `trace_term = tr(Σ_t^{-1} Σ_q)` via Cholesky-back-solve at lines
  281-283, `logdet_p = log|Σ_t|`, `logdet_q = log|Σ_q|`.
- `_kl_kernel_diagonal` at line 452 (the diagonal-σ closed form):
  ```python
  kl = 0.5 * (trace_term + mahal_term - K + logdet_term)
  ```
  with `trace_term = sum_k σ_q^k / σ_t^k`,
  `mahal_term = sum_k (μ_t^k - μ_q^k)² / σ_t^k`,
  `logdet_term = sum_k (log σ_t^k - log σ_q^k)`.

`transformer/vfe/e_step.py`, function `_diag_kl` at lines 123-154:

```python
return 0.5 * (
    _sq / _sp
    + (mu_q - mu_p) ** 2 / _sp
    - 1.0
    + _sp.log()
    - _sq.log()
)
```
This is the per-dim version: divisor and log-det target are `σ_p`
(the second-slot Gaussian, matching the convention `KL(q || p)`).

### Comparison to manuscript

Manuscript §3923 form (with `q = q, p = p`):
- log term: `log|Σ_p| - log|Σ_q|`
- trace: `tr(Σ_p^{-1} Σ_q)`
- Mahalanobis: `(μ_p - μ_q)^T Σ_p^{-1} (μ_p - μ_q)`
- subtract: `K`
- overall factor `1/2`

Code `_kl_kernel_dense` (with `q = q, t = "transported" = p`):
- log term: `logdet_p - logdet_q = log|Σ_t| - log|Σ_q|`. Match.
- trace: `tr(Σ_t^{-1} Σ_q)`. Match (Σ_t in the second slot).
- Mahalanobis: `(μ_t - μ_q)^T Σ_t^{-1} (μ_t - μ_q)`. Match
  (`(μ_p - μ_q)² = (μ_q - μ_p)²` so the order in the difference
  doesn't matter; the inverse-covariance Σ_t^{-1} is correctly on
  the second slot).
- `- K` and overall `1/2`. Match.

Code `_diag_kl` (per-dim) is the diagonal Gaussian specialisation of
the same form: divide by `σ_p`, subtract `1` per dimension, log-det
is `log σ_p - log σ_q`. Match.

Sign and factor conventions are consistent throughout. The convention
"second slot is the inversion target" (i.e., `Σ_p^{-1}` appears in
the trace and Mahalanobis, `log|Σ_p|` is positive) is uniform in both
the manuscript and both code paths.

### Verdict

**CONFIRMS-MANUSCRIPT.** The closed-form Gaussian KL displayed at
§3923 and §4014 is exactly what the code computes, both in the
dense full-covariance kernel (Cholesky-based) and in the diagonal
closed-form kernel. Standard form per [BleiKuckelbirgJordan2017]
and Bishop PRML §A.4 (formula reproduced in any standard treatment
of multivariate Gaussian KL).

### Recommended manuscript edit

Manuscript is correct, no edit needed for the KL formula itself.

One minor presentation note (Note-level, not Major): §3923 and §4014
display the same formula in two different orders (`log(|Σ_p|/|Σ_q|)`
first in §3923 vs trace first in §4014). This is purely cosmetic but
consistency across the two display equations would help readers
cross-reference. If a cleanup pass is being done, normalise both to
the same term ordering. No code change implied.

---

## AUDIT 3 — Top-down ablation presence

### Claim being checked

Manuscript §2184-2228 ("Top-Down Participation: Closing the Loop")
describes the participatory loop closing via cross-scale shadow
transport `p_i^{(s)}(x) = Ω_{i,I}[q_I^{(s+1)}](x)` and
`r_i^{(s)}(x) = \tilde Ω_{i,I}[s_I^{(s+1)}](x)`, with the
"Ouroboros Tower" extension propagating shadows from multiple
ancestral scales and the self-referential closure for top-scale
agents. The claim under audit: is there a controlled ablation
anywhere in the codebase that *disables* the upward / cross-scale
top-down path and measures the effect?

### Files inspected

- `Grep "top_down|topdown|top-down|disable_top|cross_scale|Lambda_top"`
  on all `.py` files: ZERO matches.
- `Grep "scale|s\+1|s\+\+|next_scale|parent_scale|hierarchical"` on
  `transformer/**/*.py`: matches are unrelated (`xscale`, `yscale`,
  `connection_init_scale`, `init_sigma_scale`, `cocycle_scale`,
  `perturbation_scale`, etc.); no `s → s+1` cross-scale code.
- `Grep "prior_handoff|prior_cascade|prior_update_from_post|prior_from_q"`:
  matches are confined to `transformer/vfe/config.py:53-54`,
  `transformer/vfe/stack.py:69-160`, `transformer/vfe/
  vfe_ablation_suite.py:128-129`, `transformer/vfe/train_vfe.py:74-75`,
  and the corresponding tests in `tests/transformer/test_vfe_package.py`.
- `transformer/vfe/stack.py` lines 1-163 implement the
  `prior_handoff_rho` / `prior_handoff_sigma` cross-LAYER
  posterior-to-prior handoff described by the manuscript itself at
  line 2217 as "an identity-copy with damping, not a multi-scale
  transport".

There is no toggle that, when set, disables a cross-scale path
`Λ^{s → s+1}` or `Ω_{i,I}[q_I^{(s+1)}]`. The code path the
manuscript describes (§2184-2228) does not exist in this
repository at all — the transformer codebase has cross-LAYER
handoff (one of `n_layers ≥ 1` legacy transformer blocks), not
cross-SCALE meta-agent hierarchical transport. The two are
different objects: the layer index `ℓ` in the transformer is not
the same as the manuscript's scale index `s` (which indexes
hierarchical RG levels with their own agents at each scale).

### Verdict

**CANNOT-VERIFY-NO-CODE.** The top-down cross-scale construction of
§2184-2228 is not implemented in the transformer codebase. There is
consequently no ablation that disables it. The closest analog —
`prior_handoff_rho` / `prior_handoff_sigma` in
`transformer/vfe/stack.py` — is explicitly disclaimed by the
manuscript at line 2217 as NOT the simulator's multi-scale top-down.

The simulator code that *would* host such an ablation is the same
deferred simulator addressed in AUDIT 1 (404 at the cited GitHub
URL).

### Recommended manuscript edit

§2228 (end of "Top-Down Participation" subsection) should explicitly
state that no top-down-disabled ablation is provided in the present
manuscript. Recommended new paragraph immediately before §2228:

> "An ablation that disables the top-down cross-scale shadow
> assignment ($p_i^{(s)} \leftarrow \Omega_{i,I}[q_I^{(s+1)}]$ set
> to the identity update $p_i^{(s)} \leftarrow p_i^{(s)}$) is not
> presented in this manuscript. The reported figures
> (Figs. 4-8) are from a single seed of the full top-down active
> dynamics; the contribution of the cross-scale shadow to those
> figures is therefore not quantified, and we flag this as a
> limitation of the present implementation report. A controlled
> ablation that toggles cross-scale shadow assignment is the
> natural next experiment and is deferred to the simulator-code
> release (Section~\ref{sec:methods_metagent})."

Additionally, the existing line 2217 disclaimer ("the transformer
codebase ... should not be read as the simulator implementation")
should be cross-referenced from §2228 so a reader following the
top-down discussion does not infer transformer-runtime support for
the §2184-2228 mechanism.

---

## AUDIT 4 — τ = κ √K canonical claim consistency

### Claim being checked

The CLAUDE.md (project root) states `τ = κ √K` as the canonical
effective softmax temperature. Manuscript Participatory line 1244
makes the same claim explicit:

> "In the working implementation the temperature is factorised as
> $\tau = \kappa\sqrt{K}$, with $\kappa$ a learnable scalar and the
> $\sqrt{K}$ factor the dimension scaling familiar from scaled
> dot-product attention."

Manuscript GL(K)_attention.tex body uses bare `τ` at line 766 / 855
in the Lagrangian derivation, then specialises to `τ = √d_k` at
line 1280 (recovery-of-standard-attention limit, where `σ⁻²` is
absorbed into `W_Q W_K^T`) and at line 1665 (Table 1, with the
"σ⁻² absorbed" condition stated in the same row). At line 1744
the multi-head form is given:

> $\beta_{ij}^{(a)} = \operatorname{softmax}_j\bigl(-\mathrm{KL}^{(a)}/(\kappa_a \sqrt{d_{\text{head}}})\bigr)$

with the surrounding text at line 1748 noting that "$\kappa_a$
serves as a convenient global scalar handle on the attention
sharpness".

### Files inspected

`transformer/core/attention.py` lines 377-383:

```python
# Effective temperature: τ_eff = κ · √K
# In the dot-product form (σ absorbed into W_Q W_K^T): τ = √d_k
# In the squared-distance form (½ from KL explicit):   τ = 2√d_k
dim_scale = math.sqrt(max(K, 1))

# Attention logits: -KL / (κ · √K)
logits = -kl_matrix / (kappa * dim_scale)  # (B, N, N)
```

The `dim_scale = sqrt(K)` factor is unconditional. The `kappa` is
threaded through from the caller. The actual softmax temperature is
the product `kappa * dim_scale = κ √K`. This matches the manuscript
Participatory §1244 statement and the manuscript GL(K) line 1744
multi-head form.

The same `κ · √K` pattern appears throughout the rest of the code
where attention temperatures are reconstructed:

- `transformer/vfe/non_flat.py:520-522`:
  ```python
  tau = kappa * math.sqrt(max(K, 1))
  ```
- `transformer/vfe/e_step.py:322`:
  `self._dim_scale = math.sqrt(max(cfg.embed_dim, 1))`,
  consumed at `e_step.py:555` as
  `tau = kappa_attached * self._dim_scale`, and at lines 1193 and
  1518 as the entropy-term coefficient `_kappa * self._dim_scale`.
- `transformer/core/vfe_gradients.py:501, 544, 894, 1343, 1795, 1919`:
  every kappa-using gradient kernel computes
  `kappa_scaled = max(kappa * math.sqrt(max(K, 1)), eps)`.
- `transformer/core/vfe_closed_form.py:231, 463, 585`: per-head
  `kappa_h_scaled = max(kappa_h_val * math.sqrt(max(d_h, 1)), eps)`.
- `transformer/core/variational_ffn.py:1038, 1241, 1302`: per-head
  entropy-coefficient `kappa_h * _sqrt_dh` (and likewise for `phi`
  per-head F evaluations at 1888).

Per-head multi-head agreement with manuscript line 1744:
`κ_h_scaled = κ_h · √d_h` — matches `κ_a √d_head` exactly.

`kappa` itself: `transformer/vfe/config.py:55` declares
`learnable_kappa: bool = False` as the default. The user's active
config (`transformer/vfe/train_vfe.py:47`) has
`'learnable_kappa': False, 'kappa': 1.0` — a constant scalar.
When `learnable_kappa = True`, `transformer/vfe/e_step.py:289-290`
registers `self.log_kappa = nn.Parameter(torch.tensor(math.log(cfg.kappa)))`
and `transformer/vfe/e_step.py:442-446` returns
`torch.exp(self.log_kappa)` as the effective `κ`. Both branches
multiply by `dim_scale = √K` in the softmax denominator.

### Comparison to manuscript

Manuscript Participatory §1244 claim: `τ = κ√K`, `κ` learnable.

Code: `τ_eff = kappa * sqrt(K)` everywhere it appears.
`learnable_kappa` is a config flag (default False); when False, `κ`
is a constant. The manuscript line 1244 claim "with $\kappa$ a
learnable scalar" describes the **available** form, not the
**active-run** form: the user's `train_vfe.py` has it off.

Manuscript GL(K) line 1280 (`τ = √d_k`): explicit recovery limit
("σ⁻² absorbed into the learned projections"). Standard-form
softmax-of-dot-product attention has no κ because κ was absorbed
into the unconstrained `W_Q W_K^T`. This is consistent with
[Vaswani2017].

Manuscript GL(K) line 1665 (Table 1, `τ = √d_k`): same context;
the row label includes "(dot-product form; σ⁻² absorbed)" — explicit
disclosure of the limit.

Manuscript GL(K) line 1744 (multi-head, `κ_a √d_head`): matches the
per-head code at `transformer/core/vfe_closed_form.py:231` exactly.

### Verdict

**CONFIRMS-MANUSCRIPT.** `τ = κ √K` is the canonical effective
softmax temperature in both single-head and multi-head paths; the
code threads `kappa` and multiplies by `dim_scale = √K` (or `√d_h`
per head) unconditionally. The manuscript GL(K) §1280 / §1665
`τ = √d_k` references are explicit recovery-of-standard limits
that do not contradict the canonical form (they correspond to
absorbing `κ` into `W_Q W_K^T`).

The one disclosure friction worth noting (Note-level): the
manuscript Participatory line 1244 says "$\kappa$ a learnable
scalar," but the user's active VFE config has `learnable_kappa =
False`. This is the same Theory M4 disclosure issue raised by the
prior verifier ("learnable κ undisclosed at canonical reduction"),
just from the opposite angle: the manuscript advertises learnable
κ at the canonical introduction but the active config does not
exercise that mode. This is a config-disclosure note, not a code
correctness issue.

### Recommended manuscript edit

Manuscript line 1244 is correct as a statement about the available
implementation. No edit required for AUDIT 4 specifically. The
broader Theory M4 finding (κ availability not flagged at §1797-1808
where the canonical reduction lives) is the appropriate place for an
edit and is already covered by the existing review.

Optional Note-level edit: at §1244 the parenthetical "with $\kappa$
a learnable scalar" could be sharpened to "with $\kappa$ a scalar
that the implementation supports as either constant or learnable;
the configurations reported in this manuscript use a constant
$\kappa = 1$". This makes the active-config choice transparent
without changing any equations.

---

## Summary table

| Audit | Verdict | Recommended edit needed |
|---|---|---|
| 1: Detector §2115 vs §3613 | CANNOT-VERIFY-NO-CODE | YES — pick one detector form, rewrite §3613 |
| 2: Gaussian KL §3923/§4014 | CONFIRMS-MANUSCRIPT | NO (optional cosmetic ordering) |
| 3: Top-down ablation §2184-2228 | CANNOT-VERIFY-NO-CODE | YES — add explicit "no ablation provided" note to §2228 |
| 4: τ = κ √K canonical | CONFIRMS-MANUSCRIPT | NO (already covered by Theory M4) |

## Audit hygiene notes

- AUDIT 1 and AUDIT 3 both reflect the same underlying gap: the
  participatory-it-from-bit simulator (which would host the meta-agent
  detector, the top-down shadow transport, and the cross-scale RG
  iteration) is not in the repository. The cited GitHub URL
  (`https://github.com/cdenn016/Participatory-It-From-Bit-Universe`)
  returned HTTP 404 at the time of the prior verifier report
  (`verifier_report_2.md` §Impl M2). The transformer codebase
  (`transformer/`) shares mathematical machinery (gauge transport,
  KL-based attention, EFE epistemic value) with the simulator but is
  not the simulator. The manuscript already discloses this at line
  2217 and line 3641-3642; the recommended edits above tighten that
  disclosure at the specific subsections that lead a reader to
  expect transformer-runtime evidence for §2115 / §2184-2228.
- AUDIT 2 and AUDIT 4 confirm long-standing core code paths
  (`kl_computation.py`, `attention.py`) implement standard forms;
  no surprises.

## Files inspected (absolute paths)

- `C:\Users\chris and christine\Desktop\V13_Gauge_Transformer\Attention\Participatory_it_from_bit.tex`
- `C:\Users\chris and christine\Desktop\V13_Gauge_Transformer\Attention\GL(K)_attention.tex`
- `C:\Users\chris and christine\Desktop\V13_Gauge_Transformer\transformer\core\kl_computation.py`
- `C:\Users\chris and christine\Desktop\V13_Gauge_Transformer\transformer\core\attention.py`
- `C:\Users\chris and christine\Desktop\V13_Gauge_Transformer\transformer\core\vfe_gradients.py`
- `C:\Users\chris and christine\Desktop\V13_Gauge_Transformer\transformer\core\vfe_closed_form.py`
- `C:\Users\chris and christine\Desktop\V13_Gauge_Transformer\transformer\core\variational_ffn.py`
- `C:\Users\chris and christine\Desktop\V13_Gauge_Transformer\transformer\core\active_inference.py`
- `C:\Users\chris and christine\Desktop\V13_Gauge_Transformer\transformer\core\expected_free_energy.py`
- `C:\Users\chris and christine\Desktop\V13_Gauge_Transformer\transformer\vfe\attention.py`
- `C:\Users\chris and christine\Desktop\V13_Gauge_Transformer\transformer\vfe\e_step.py`
- `C:\Users\chris and christine\Desktop\V13_Gauge_Transformer\transformer\vfe\config.py`
- `C:\Users\chris and christine\Desktop\V13_Gauge_Transformer\transformer\vfe\stack.py`
- `C:\Users\chris and christine\Desktop\V13_Gauge_Transformer\transformer\vfe\non_flat.py`
- `C:\Users\chris and christine\Desktop\V13_Gauge_Transformer\transformer\vfe\active_inference.py`
- `C:\Users\chris and christine\Desktop\V13_Gauge_Transformer\transformer\vfe\efe.py`
- `C:\Users\chris and christine\Desktop\V13_Gauge_Transformer\transformer\vfe\train_vfe.py` (active config)
