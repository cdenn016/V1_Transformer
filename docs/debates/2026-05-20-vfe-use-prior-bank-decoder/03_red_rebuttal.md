# Red Rebuttal — vfe-use-prior-bank-decoder

## Concession

I grant blue's sympy result. The identity `2·KL(q || π_v) − (combined + prior_bias) = −1 − log σ_q` is v-independent on the diagonal-Gaussian sub-manifold, so at construction (`decode_log_scale = 0` ⇒ `c = 1`) the implemented logit row reduces exactly to `softmax_v(−KL(q_i || π_v)/τ)`. Sub-proposition 1 holds in the untrained regime. I also grant that no `nn.Linear`, MLP, activation, or learned QKV projection sits on the `use_prior_bank=True` decode path between `transformer/vfe/model.py:184` and the return of `decode()` (sub-proposition 3 in the structural sense). Finally, I grant that under blue's falsification condition 6 the strict parameter-frugality argument is config-dependent — blue retracted it explicitly in its own opening, so I capture that as a closed sub-claim: the "fewer parameters than `nn.Linear(K,V)`" argument is not load-bearing here.

## Core attack

Blue's defense relies on a c=1 reading that the runtime config does not produce. The deployed model trains `decode_log_scale` against the cross-entropy loss, so `c` drifts from 1, and the deviation enters the softmax distribution **multiplicatively on the KL spread across v**, not as a v-invariant additive shift. This invalidates sub-proposition 1 for the trained model — which is what CLAUDE.md's "Hard Constraint" footnote is documenting as the canonical decoder.

The algebraic structure is precise. The implementation computes (`transformer/vfe/prior_bank.py:505-506`):

```
logits_{i,v}^impl = −0.5 · c · (combined_{i,v} + prior_bias_v)         where c = exp(decode_log_scale.clamp(±3))
```

The canonical claim is `logits_{i,v}^canon = −KL(q_i || π_v)/τ = −0.5·(combined_{i,v} + prior_bias_v) + 0.5·(1 + log σ_q,i)`. The bracketed term is v-independent and drops in softmax. So:

```
logits_{i,v}^impl − logits_{i,w}^impl = c · (logits_{i,v}^canon − logits_{i,w}^canon)
```

Executed sympy session (1D diagonal, `mu_p ∈ {1, 2}`, `sigma_p = 1`):

```
canonical logit diff (v=1 vs v=2):   1.5 − 1.0·mu_q
impl     logit diff (v=1 vs v=2):  c·(1.5 − 1.0·mu_q)
```

Executed numpy session (two-class softmax, `KL = [1.0, 3.0]`, `τ = 1.0`):

```
c=1.0  : softmax = [0.881, 0.119]    p[0]/p[1] = 7.39
c=0.5  : softmax = [0.731, 0.269]    p[0]/p[1] = 2.72
c=2.0  : softmax = [0.982, 0.018]    p[0]/p[1] = 54.60
c=20.0 : softmax ≈ [1.000, 4.2e−18]   p[0]/p[1] ≈ 2.4e17
```

The trained distribution at `c = 0.5` or `c = 2.0` is not the same distribution as `softmax(−KL/τ)`. CLAUDE.md's Hard Constraints clause states the documented decoder is `logits = −KL(q || π_v)/τ`. The training-time decoder is `softmax(−c·KL/τ)`, and `c` is a free scalar grouped into the optimizer (`transformer/training/optimizer.py:247-252, 311-316`, `m_sigma_params`, `m_sigma_lr = 5e-5`). The module's own docstring (`transformer/vfe/prior_bank.py:22-34`) admits this: "the parameter drifts during training" and "Equivalent to a second softmax temperature stacked multiplicatively on `tau`."

Two corollaries.

First, blue's Vaswani / CLIP analogy is malformed. The `1/√d_k` factor in [Vaswani2017 §3.2.1] is a **fixed dimension-dependent constant**: "We suspect that for large values of d_k, the dot products grow large … To counteract this effect, we scale the dot products by 1/√d_k." It is not a learnable parameter trained against the loss. The learned-temperature variant blue cites belongs to CLIP [Radford2021 §2.5] — and CLIP's contrastive loss is over an `(image, text)` similarity matrix, not a Gaussian KL classifier, and Radford et al. explicitly document the learned scalar as a deviation from the fixed `1/√d` convention, not an instance of it. Conflating "fixed dimension scaling" with "free learnable scalar trained against CE" empties the structural distinction. The honest precedent for `decode_log_scale` is CLIP's `logit_scale`, which is documented in CLIP as a **learnable temperature**, not as a Vaswani-style scaling factor.

Second, sub-proposition 4 (Bishop derivation) is not the discharge blue claims it is. Bishop §4.2 derives `log p(C_k | x) ∝ log p(x | C_k) = −0.5·(x − μ_k)^T Σ_k^{−1}(x − μ_k) − 0.5·log|Σ_k| + const` for a **point observation `x`** under class-conditional Gaussians [Bishop2006 §4.2 eqs 4.65–4.69]. Substituting a recognition distribution `q(s)` for the point `x` and taking `E_q[log p(s | C_v)] = −H(q) − KL(q || π_v)` is a user extension; it does not appear in Bishop §4.2. Blue's evidence pack itself acknowledges this — `01_evidence.md` line 84 calls it a "Replacing the point observation `x` with a distribution `q(x)`" step, which is precisely the step Bishop does not perform. The closest external precedent for the actual operation (KL-as-classifier under a recognition distribution on the manifold of Gaussians) is [Vilnis-McCallum 2015, *Word Representations via Gaussian Embedding*] in a retrieval setting, not a language-model decode. I do not dispute that the form is mathematically sensible; I dispute the claim that it is "the standard form" derivable by direct substitution from Bishop. The construction is the user's own; calling it a "canonical decoder" is overstatement.

## Defense

My Phase-2 falsification condition stands: the soundness claim collapses unless either (a) the documentation states `logits = −c·KL/τ` with `c` learnable, or (b) `decode_log_scale` is structurally frozen at 0 in the deployed model. Neither holds. The CLAUDE.md text states `−KL/τ` (no `c`), and `decode_log_scale = nn.Parameter(torch.zeros(1))` (`prior_bank.py:208`) is in the trainable optimizer group. The module docstring at `prior_bank.py:22-34` already concedes this internally — the implementation knows it has drifted from the manuscript's stated form.

On Law 3 (sub-proposition 2): I grant blue's narrow defense under `diagonal_covariance=True` (the user's active config at `train_vfe.py:79`). I do not grant the unqualified soundness claim. The claim under debate is the global claim in `00_claim.md`, which does not condition on the toggle. If blue's defense requires `diagonal_covariance=True`, then the claim has a hidden config qualifier and the full-covariance branch (which is a supported, theoretically pure path per CLAUDE.md's "There should ALWAYS exist a theoretically/mathematically 'pure' path under appropriate toggles") is decoded by a diagonal projection (`prior_bank.py:478-479`) that drops off-diagonal Σ terms. On the statistical manifold of `K×K` SPD-augmented Gaussians [AmariNagaoka2000 §2.3], the diagonal sub-manifold is a strict submanifold; the Fisher-induced KL on the diagonal projection is not the Fisher KL of the embedding distribution. Either the soundness claim must be qualified to `diagonal_covariance=True`, or the full-cov decoder's diagonal projection is itself a Law-3 violation the claim must own. Blue's opening picks (a) implicitly by citing `train_vfe.py:79`; the claim in `00_claim.md` does not.

On blue's pre-emption (sub-proposition 6, "the decode is not a term of F"): I grant that "VFE-native" can be read as "lives on the same Gaussian manifold" rather than "is a summand of F." That reading is defensible. But blue's appeal to Form-3 FEP — "the accuracy term `E_q[−log p(o|s)]` is part of F when an observation is realised" — fails on the actual loss assembly. `transformer/vfe/model.py:11-32` documents that the training objective is `loss = ce_loss + 0.5·mass_phi·||phi||^2 + Σ block._aux_hyperparam_loss`. The CE loss against the labels is `−log softmax_v(−c·KL/τ)_{y}`, not `E_q[−log p(o = y | s)]`. The two differ by exactly the V-invariant `H(q)` shift **only when `c = 1`**. With `c ≠ 1`, the CE-on-implementation-logits is `c·KL(q || π_y) − log Z(c·KL/τ)`, and `Z(c·KL/τ)` is **not** the FEP partition function `Σ_v exp(−KL(q||π_v)/τ)`. The "observation-likelihood face of F" identification therefore requires `c = 1` to be exact — the same assumption falsified by training the scalar.

I cite the standard form against which the claim should be measured. The canonical Gaussian-cluster discriminant [Bishop2006 §4.2 eq. 4.66] and the canonical variational ELBO [BleiKuckelbergJordan2017 eq. 3, Friston2010 eq. 2.2] both contain no learnable scalar multiplier on the KL. The standard form is what CLAUDE.md documents. The deployed code is `c · KL`, with `c` trained. The claim as stated in `00_claim.md` — that the implementation "is correctly implemented as a VFE-native decoder via `logits = −KL(q || π_v)/τ`" — is falsified for any deployed checkpoint where `decode_log_scale ≠ 0`. The honest reading is the docstring at `prior_bank.py:22-34`: the documented formula and the runtime formula differ by a trained scalar; the soundness claim should either qualify itself to `c = 1` or rewrite CLAUDE.md to `−c·KL/τ`.
