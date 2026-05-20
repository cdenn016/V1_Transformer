# Verdict — vfe-use-prior-bank-decoder

## Outcome

RED_WINS

## Decisive evidence

`transformer/vfe/prior_bank.py:505-506` (runtime) vs CLAUDE.md "Hard Constraints" footnote (advertised formula), in light of `external_canon_transformers.md` §2 ("κ is user-introduced — it's a learnable scalar on top of the standard scaling. ... they are related but the κ is extra"):

```
# runtime (prior_bank.py:505-506):
scale = torch.exp(self.decode_log_scale.clamp(-3.0, 3.0))
logits = -0.5 * scale / tau * (combined + prior_bias.unsqueeze(0).unsqueeze(0))
```

The runtime formula is `logits = -c · KL(q || π_v) / τ` with `c = exp(decode_log_scale)` trainable (`prior_bank.py:208` declares `nn.Parameter(torch.zeros(1))`; `transformer/training/optimizer.py:247-261` routes it into the σ-Fisher group and applies the `p.grad = p.grad * 2.0` natural-gradient preconditioner). The advertised formula in CLAUDE.md and `transformer/vfe/config.py:338-348` is the bare `-KL(q || π_v) / τ`. The two coincide only at `c = 1`, i.e., at construction; under training they diverge multiplicatively on the v-dependent KL spread, and red's executed numpy session (`KL = [1.0, 3.0]`, τ=1.0) shows the probability ratio `p[0]/p[1]` swings from 7.39 at c=1 to 2.72 at c=0.5 and 54.6 at c=2.0. The "modulo softmax-invariant constants" caveat absorbs v-independent shifts but does not absorb a multiplicative rescaling of the v-dependent logit. The module's own docstring at `prior_bank.py:22-34` concedes the deviation ("This module instead computes `logits = -c * KL(q || pi_v) / tau` ... Equivalent to a second softmax temperature stacked multiplicatively on `tau`").

## Reasoning

The claim in `00_claim.md` is compound: the decoder is sound, correctly implemented as `logits = -KL(q || π_v)/τ`, constraint-compliant, and derivable from canonical primitives. A conjunctive claim falls when any load-bearing sub-proposition falls.

Grading each sub-proposition:

**P1 (formula match) — LOST for blue.** Blue's sympy verified the algebraic identity at `c = 1`: `2·KL(q||π_v) − (combined + prior_bias) = −1 − log σ_q`, a row-only (v-invariant) shift that drops in softmax. Red conceded this baseline. Red then attacked at the deployed condition `c ≠ 1`: `c` multiplies the v-dependent KL, so `softmax(-c·KL/τ) ≠ softmax(-KL/τ)`. Red's executed numpy demonstration is decisive on the numerics. Blue's Phase-3 rebuttal pivoted to CLIP precedent ([Radford2021 §2.5]) and conceded the documentation gap explicitly: "The documentation is wrong in a load-bearing way (the formula advertised is not the formula executed)." The CLIP defense does not promote `c` to canonical status: `external_canon_transformers.md` §2 explicitly states that learnable temperature scalars on top of the standard scaling are user-introduced additions, not part of the canonical form (the canon's identical treatment of κ in `τ = κ√K`). The Vaswani `1/√d_k` precedent is a fixed dimension-dependent constant, not a learnable scalar; the canon §2 calls out this exact conflation as a pitfall. CLIP's `logit_scale` belongs to the contrastive-learning literature and does not appear in the project's external canon files — citing it does not invoke a canonical form. The CLAUDE.md wording "subsumed by the PriorBank decode, `logits = -KL(q || π_v)/τ`" is what the claim binds to; the runtime is `-c·KL/τ` with `c` trained; the two formulas disagree under softmax for `c ≠ 1`.

**P2 (Law 3 / same manifold) — PARTIAL.** Blue's defense holds under the active config `diagonal_covariance=True` (`train_vfe.py:79`), where `torch.diagonal(...)` at `prior_bank.py:478-479` is a no-op and encode/E-step/decode all read the same `(ℝ⁺)^K` representation. Blue conceded the full-covariance branch in Phase 3: "Under `diagonal_covariance=False`, encode and the E-step compute on `Sym⁺(K)` while decode reads only the diagonal." The claim in `00_claim.md` carries no config qualifier and CLAUDE.md mandates a theoretically pure path under appropriate toggles; the full-cov toggle's decode path discards off-diagonal mass that the sandwich `Ω Σ Ωᵀ` generates. P2 holds in the live config but breaks in a supported configuration; the unqualified claim is not defended.

**P3 (no-neural-networks) — CONCEDED to blue.** Red's opening states "by the pure letter of the constraint the toggle does discharge it"; Phase-3 rebuttal repeats the structural concession. The `use_prior_bank=True` path between `model.py:184` and the return of `decode()` contains no `nn.Linear`, no MLP, no activation, no learned QKV projection. The Embedding tables and the single `decode_log_scale` scalar are not what the constraint prohibits.

**P4 (canonical derivation) — PARTIAL for blue.** Blue's two-step composition — Bishop §4.2 discriminant `log p(C_v | x) ∝ -0.5 (x-μ_v)ᵀ Σ_v⁻¹ (x-μ_v) - 0.5 log|Σ_v| + const`, then substitute `E_q[·]` for the point observation — yields `E_q[log p(s | C_v)] = -KL(q || π_v) - H(q) + const_v-invariant`. The algebra is canonical and the substitution is a standard variational move; plugged into Friston Form-3 `F = E_q[-log p(o|s)] + KL(q||p(s))` (canon §1) it recovers the user's logit form modulo v-invariant terms. Red's strike — that Bishop §4.2 treats only point observations and the q-substitution is a user extension — is technically accurate but does not unmake the canonical composition; the composition is two-step canonical, and the `q`-for-`x` substitution is the canonical variational plug-in (Form-3 is *defined* as `E_q[-log p(o|s)]` for a recognition `q`). Blue conceded the derivation is not present as a labelled theorem in any `.tex` file. The labeling gap is real but does not falsify derivability.

Weighing: P3 is conceded; P4 lands for blue mathematically; P2 is partial; P1 is falsified at the deployed condition. The compound claim binds the implementation to a specific formula, and the runtime does not implement that formula. Blue's Phase-3 concession is explicit on this point. Per the source-of-truth precedence (canonical form wins when the user's construction disagrees, and CLAUDE.md's own canonical-form advertising is what the claim binds to), the compound claim is falsified on P1 alone. The remaining wins for blue (P3, P4) do not rescue a compound that asserts the implementation matches the stated formula when it does not.

This is not a "documentation is slightly off" tie. The advertised formula and the runtime formula define different probability distributions for every `c ≠ 1` over a 400× range, and `c` is trained against the cross-entropy loss with a natural-gradient preconditioner. The deployed model is the runtime, not the documentation.

## Action

Two acceptable resolutions, both already named in blue's Phase-3 rebuttal:

1. Edit CLAUDE.md "Hard Constraints" footnote and `transformer/vfe/config.py:338-348` to read `logits = -c · KL(q || π_v) / τ` with `c = exp(decode_log_scale)` documented as a CLIP-style learnable temperature stacked multiplicatively on τ. This aligns the advertised formula to the runtime. The canon-§2 treatment of κ in `τ = κ√K` is the precedent for how to document this: label `c` as the user's addition on top of the canonical scaling, not as part of the canonical form.

2. Freeze `decode_log_scale = 0` (set `requires_grad=False` at `prior_bank.py:208`, or remove the parameter entirely and replace `scale` with `1.0` at `prior_bank.py:505`). This restores bitwise agreement with the advertised formula at all training steps.

Independently, P2 is a separate documentation strike: either qualify the Law-3 claim in `config.py:338-348` and `prior_bank.py:18-20` to `diagonal_covariance=True` only, or own the full-cov branch's diagonal-projection-at-decode as a documented approximation rather than as Law 3.

P4's labeling gap (the Bishop §4.2 + Friston Form-3 composition is not in the manuscripts as a labelled derivation) is a manuscript-correctness item, not a soundness item. Adding the two-step derivation as a short appendix would discharge the canon-§1 policy ("similar user constructions ... must be labeled as novel, requiring its own justification") and convert P4 from PARTIAL to WON for any future debate.
