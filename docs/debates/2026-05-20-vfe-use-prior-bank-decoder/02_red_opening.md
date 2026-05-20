# Red Opening — vfe-use-prior-bank-decoder

## Steelman (opposing position)

`use_prior_bank=True` discharges the no-neural-networks constraint by routing logits through the Gaussian-cluster classifier `softmax_v(-KL(q_i || π_v)/τ)` of [Bishop2006 §4.2], with the bank's per-token priors living on the same statistical manifold that encode and the E-step traverse [Amari Information Geometry Ch. 2–3]; the algebraic identity `softmax_v(-0.5·(combined+bias)) = softmax_v(-KL(q || π_v))` (verified by sympy below) means the runtime formula reproduces the canonical decode bitwise at the construction-time value of the learnable scalar `c`, and the bank holds Embedding tables rather than the MLPs / learned QKV / activations the constraint actually prohibits.

## Position

The claim is false in three load-bearing senses, any one of which is sufficient:

1. The runtime decode is `logits = -c·KL/τ` with a *trained* scalar `c = exp(decode_log_scale)` clamped to `[exp(-3), exp(3)]`, not `logits = -KL/τ`; `softmax(-c·KL/τ)` differs from `softmax(-KL/τ)` for every `c ≠ 1`, so the claim "implementation matches the stated formula modulo softmax-invariant constants" fails at any optimizer step where `decode_log_scale ≠ 0`.
2. The Law-3 statement "same Gaussian manifold for encode/infer/decode" is broken in the full-covariance branch — encode and the E-step operate on the full `K×K` SPD manifold; decode unconditionally projects to the diagonal manifold via `torch.diagonal(...)` at `transformer/vfe/prior_bank.py:478-479`. These are different statistical manifolds (different Fisher metrics).
3. The decode KL is not a stationary-point term of the manuscript free-energy functional `F`; the training objective is `CE + 0.5·mass_phi·||φ||² + Σ aux` (`transformer/vfe/model.py:254-302`), and the `e_step.py:13-48` module docstring concedes that the framework is "structurally amortised inference, not classical variational EM where E and M alternate on the same F functional." Labeling the decode "VFE-native" treats a discriminative readout as if it were a variational update; canonical FEP / VI [Friston2010 Eq. 2.2; BleiKuckelbirgJordan2017 Eq. 3] requires the decode term to enter F.

## Evidence

**Code, runtime decode deviation (sub-proposition 1).**
- `transformer/vfe/prior_bank.py:208` — `self.decode_log_scale = nn.Parameter(torch.zeros(1))`. A trainable scalar.
- `transformer/vfe/prior_bank.py:505-506`:
  ```
  scale = torch.exp(self.decode_log_scale.clamp(-3.0, 3.0))
  logits = -0.5 * scale / tau * (combined + prior_bias.unsqueeze(0).unsqueeze(0))
  ```
  i.e. `logits = -(c/τ) · 0.5 · (combined + prior_bias)`, not `-(1/τ) · 0.5 · (combined + prior_bias)`. The "modulo softmax-invariant constants" caveat the manuscript hangs on does not absorb `c`: `c` multiplies the v-dependent logit, so the row-rescaling propagates into every softmax probability.
- `transformer/vfe/trainer.py:247-252` — `decode_log_scale` is collected into the `m_sigma_params` group with LR `m_sigma_lr = 5e-5` (`transformer/vfe/config.py:358`).
- `transformer/training/optimizer.py:252-261` — that group's gradient is then multiplied by 2 (`p.grad = p.grad * 2.0`) under the σ-Fisher preconditioner, treating `decode_log_scale` as if it were a log-variance coordinate. The parameter is therefore both trained and *natural-gradient-preconditioned* on a Fisher metric it does not live on.
- `transformer/vfe/prior_bank.py:23-34` — the module-level docstring concedes the deviation: "This module instead computes `logits = -c * KL(q || pi_v) / tau` with a learnable scalar `c = exp(decode_log_scale)`."

The clamp range `[exp(-3), exp(3)] ≈ [0.05, 20]` is a 400× window. CLAUDE.md, `transformer/vfe/config.py:338-348`, and the no-neural-networks constraint all advertise `logits = -KL/τ`; the runtime is `-c·KL/τ` with `c` learned. The two formulas do not agree under softmax for `c ≠ 1`.

**Sympy verification of the identity at `c = 1` (sub-proposition 1 baseline).**
```
import sympy as sp
sq, sp_v, mq, mp_v = sp.symbols('sq sp_v mq mp_v', positive=True, real=True)
two_KL = sq/sp_v + (mq - mp_v)**2/sp_v - 1 + sp.log(sp_v) - sp.log(sq)
combined_bias = (sq + mq**2)/sp_v + (-2*mq)*(mp_v/sp_v) + (mp_v**2/sp_v) + sp.log(sp_v)
sp.simplify(two_KL - combined_bias)   # => -log(sq) - 1   (v-invariant)
```
At `c = 1` the discarded constant per coordinate is `1 + log σ_q`, which is v-invariant and cancels in `softmax_v`. The identity is exact in that limit. With `c ≠ 1` the per-position rescaling of `-0.5·(2·KL + 1 + log σ_q)` by `c` makes the (1 + log σ_q) term no longer cancel as written — it becomes a position-only logit shift `-0.5·c·(1+log σ_q)` that still cancels in softmax over v (since v-independent), so the softmax-invariance survives `c`; however the *softmax* itself, i.e. the probability vector and its KL gradients during CE training, is `softmax_v(-c·KL/τ) ≠ softmax_v(-KL/τ)`. The claim "modulo softmax-invariant constants" therefore captures the dropped constants but not the multiplicative scalar.

**Code, manifold mismatch at decode (sub-proposition 2).**
- `transformer/vfe/prior_bank.py:478-479`:
  ```
  sigma_q_diag = torch.diagonal(sigma_q, dim1=-2, dim2=-1) if is_full_cov else sigma_q
  sigma_p_diag = torch.diagonal(sigma_p, ...) if sigma_p.dim() >= 3 ... else sigma_p
  ```
- `transformer/vfe/prior_bank.py:436-439` (docstring concession): "the decode uses diagonal projection for O(V·K) efficiency. This is a documented approximation at the decode boundary — encode and infer operate on the full Gaussian manifold, but decode projects to diagonal KL."

The full-cov manifold and the diagonal-cov manifold are distinct statistical manifolds with different Fisher metrics [Amari Information Geometry Ch. 2-3]; the canonical multivariate Gaussian KL [Bishop2006 App. B] is
```
KL(q||p) = 0.5[ tr(Σ_p⁻¹ Σ_q) + (μ_p - μ_q)ᵀ Σ_p⁻¹ (μ_p - μ_q) - K + log|Σ_p|/|Σ_q| ]
```
The diagonal projection drops every off-diagonal contribution from both `tr(Σ_p⁻¹ Σ_q)` and `log|Σ_p|/|Σ_q|`. The discarded mass is exactly the off-diagonal contribution that the sandwich product `Σ ← Ω Σ Ωᵀ` *generates* (`transformer/vfe/prior_bank.py:283-364`, `_apply_gauge_transform`). Under `diagonal_covariance=False` (the rigorous covariance-transport setting), Law 3 fails on the implementation side: encode and infer compute on `Sym⁺(K)`, decode computes on `(ℝ⁺)^K`. The user's active config has `diagonal_covariance=True`, so this branch is dormant in production, but the claim "Law 3 holds in both modes" (`transformer/vfe/prior_bank.py:18-20`) is broader than the active config and is falsified by the full-cov path.

**Code + canon, "VFE-native" mislabeling (sub-proposition 4 and Law 3 interpretation).**
- `transformer/vfe/model.py:251-304` — the optimizer-visible scalar is `loss = ce_loss + 0.5·mass_phi·||φ||² + Σ aux`. None of those terms is `-E_q[log p(o|s)]` taken as the observation-likelihood term of the manuscript F.
- `transformer/vfe/e_step.py:41-48`: "Outer M-step ... minimises cross-entropy plus a `mass_phi * ||phi||^2` regulariser plus `sum block._aux_hyperparam_loss`, NOT the converged-q value of F. ... this is structurally amortised inference ..., not classical variational EM where E and M alternate on the same F functional."
- `[Friston2010 Eq. 2.2]` and `[BleiKuckelbirgJordan2017 Eq. 3]`: `F = E_q[-log p(o|s)] + KL(q(s) || p(s))`. For the decode to be "VFE-native" in any non-terminological sense, `-KL(q || π_v)/τ` must arise as the log of `p(o=v | s) = softmax-target` derived from `E_q[log p(o|s)]`. The Bishop §4.2 generative-classifier derivation gives this only if (a) `log p(o=v) ∝ -KL(q || π_v)`, which requires uniform class prior + the entropy term `H(q)` being v-invariant (it is, but only by construction), and (b) the decode is interpreted *discriminatively*. The user is then not minimising F end-to-end; they are minimising CE on logits that *look like* `-KL/τ`. That is precisely a discriminative readout, exactly what the e_step.py docstring concedes.

**Canonical literature on the form itself (sub-proposition 4).**
- The transformer canon [Vaswani2017 §3.2.1; Bahdanau2015] uses `logits = W_O · h`; `softmax(-KL/τ)` is not a canonical language-modeling decoder.
- The closest precedent is Gaussian word embeddings for *retrieval* / metric learning [Vilnis-McCallum-2015 "Word Representations via Gaussian Embedding"], which uses KL-similarity but is not a language-model decoder per se. Using KL-to-prior as the LM logit is a user-specific construction.
- The novel-construction policy is explicit in `external_canon_inference.md §1`: "Multi-agent coupling terms ... and similar user constructions ... must be labeled as novel, requiring its own justification." The "VFE-native LM decoder" deserves the same labeling treatment and a derivation from canonical primitives, not a CLAUDE.md assertion.

**The "no-neural-networks" constraint is not fully discharged by the toggle (sub-proposition 3).**
- The constraint in CLAUDE.md prohibits `nn.Linear`, MLPs, activations, and learned QKV projections. The PriorBank holds `nn.Embedding` tables (`prior_bank.py:190-198`), `nn.Parameter` log-σ and µ vectors (`prior_bank.py:185-186`), and the trained `decode_log_scale` (`prior_bank.py:208`). None of those are MLPs/activations/QKV, so by the pure letter of the constraint the toggle does discharge it. But the framing "subsumed by the PriorBank decode, `logits = -KL(q || π_v)/τ`" specifically advertises the canonical formula; the runtime is `-c·KL/τ` with trained `c`. The constraint claim is *worded as if* the canonical formula were implemented; it isn't.

## Falsification conditions

This red position is wrong if, at runtime under the active config:

1. **`decode_log_scale` never moves from zero during training.** If the parameter's optimizer step is zero by construction (e.g., excluded from the optimizer, or `requires_grad=False`, or in a parameter group with `lr=0`), then `c ≡ 1` and the runtime decode equals the canonical decode bitwise. Falsification check: print `model.prior_bank.decode_log_scale.grad` after the first backward and inspect whether the AdamW step changes its value. Current evidence (`trainer.py:247-252` group routing, `optimizer.py:252-261` σ-preconditioner doubling the gradient) shows it is trained, so the falsification requires user verification that the optimizer state is in fact null for this parameter.

2. **The "modulo softmax-invariant constants" caveat is interpreted to include any v-independent multiplicative scalar.** This is a stretch reading — `c` multiplies the v-dependent KL value, not a v-independent constant — but if the project's canonical formulation is reinterpreted to permit any temperature reparameterization (so the manuscript's `τ` is implicitly understood to absorb `c`), then the runtime is equivalent to a manuscript with `τ_effective = τ/c` and the soundness claim survives at the cost of the `tau = κ√K` interpretation no longer being the only temperature in the system. The user must then pick one: (a) edit CLAUDE.md and the manuscript to read `logits = -c·KL/τ`, or (b) freeze `decode_log_scale = 0`.

3. **The diagonal-projection-at-decode is not a Law-3 break.** This requires arguing that the diagonal Gaussian and the full Gaussian are *the same statistical manifold for decode-purpose KL*. Concretely: a proof that `KL_diag(diag(Σ_q), diag(Σ_p))` equals `KL_full(Σ_q, Σ_p)` up to a v-invariant constant for every `(Σ_q, Σ_p)` produced by the encode + E-step pipeline under `diagonal_covariance=False`. Counterexample suffices to break this: any pair where `Σ_q` or `Σ_p` has nonzero off-diagonals (which the sandwich product `Ω Σ Ωᵀ` produces whenever `Ω` is not block-diagonal in the natural basis).

4. **CE+aux is the canonical observation-likelihood term of F under the user's `p(o|s)` generative model.** A clean derivation would write `-log p(o=v|s) = -log softmax_v(-KL(q||π_v)/τ)·(v-th element)` and show that integrating this against `q(s)` recovers the manuscript F's `-E_q[log p(o|s)]` term. The Bishop §4.2 argument provides a discriminative-classifier sketch but does not put F on the M-step loss line. Falsification check: locate, in `Attention/GL(K)_attention.tex` or `Attention/GL(K)_supplementary.tex`, the equation that derives `loss = CE + ...` *from* the manuscript F (not just by analogy). If found, this red position is over-stated; if not found, the "VFE-native" label is terminology, not theorem.

If none of these four falsifiers fires, the claim fails on at least one of the three sub-propositions stated in `00_claim.md`.
