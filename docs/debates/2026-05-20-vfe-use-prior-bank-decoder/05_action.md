# Action — vfe-use-prior-bank-decoder

**From verdict:** RED_WINS

## Sub-proposition outcomes

| Sub-proposition | Outcome |
|---|---|
| P1 — runtime matches `logits = -KL(q || π_v)/τ` | **LOST for blue** (runtime is `-c · KL/τ` with `c = exp(decode_log_scale)` trainable; coincides only at construction) |
| P2 — same Gaussian manifold for encode/infer/decode (Law 3) | **PARTIAL** (holds in `diagonal_covariance=True`, breaks in full-cov branch) |
| P3 — no-neural-networks constraint satisfied | **CONCEDED to blue** (no `nn.Linear`/MLP/activation on the True path) |
| P4 — decode derivable from canonical primitives | **PARTIAL for blue** (Bishop §4.2 + Friston Form-3 composition is canonical; derivation absent from manuscripts) |

## Recommended action

Two acceptable resolutions for P1 (the falsifying sub-proposition):

1. **Documentation fix.** Edit `CLAUDE.md` "Hard Constraints" footnote and `transformer/vfe/config.py:338-348` to advertise `logits = -c · KL(q || π_v) / τ`, with `c = exp(decode_log_scale)` labelled as a CLIP-style learnable softmax temperature stacked multiplicatively on `τ` (precedent: the canon's treatment of `κ` in `τ = κ√K` at `external_canon_transformers.md` §2). Aligns advertised formula to runtime.

2. **Code fix.** Freeze `decode_log_scale` at 0 (set `requires_grad=False` at `transformer/vfe/prior_bank.py:208`, or remove the parameter entirely and replace `scale` with `1.0` at `prior_bank.py:505`). Restores bitwise agreement with the advertised formula across training.

Either resolution discharges P1. The runtime currently fails the stated formula because `c` drifts during training (~400× range under the `[-3, 3]` clamp), so doing nothing leaves the claim falsified.

### Separate action for P2

Qualify the Law-3 claim in `transformer/vfe/config.py:338-348` and `transformer/vfe/prior_bank.py:18-20` to `diagonal_covariance=True` only, or document the full-cov branch's `torch.diagonal` projection at `prior_bank.py:478-479` as a documented approximation rather than as Law 3. This is independent of the P1 fix.

### Optional action for P4

Add a labelled appendix to `Attention/GL(K)_attention.tex` (or `GL(K)_supplementary.tex`) deriving the decode form as the Bishop §4.2 + Friston Form-3 composition: substitute the recognition distribution `q` for the point observation in the Bayes-optimal Gaussian discriminant, take expected log-likelihood under `q`, identify the result as the F accuracy summand under the per-class Gaussian likelihood model. Converts P4 from PARTIAL to WON for any future debate; discharges the canon-§1 policy that novel constructions require explicit derivation from canonical primitives.

## Follow-up debates (if any)

None mandatory. The Law-3 partial (P2) is a clean documentation strike; a follow-up debate is only warranted if the user wants to defend the full-cov decode-as-canonical claim under load.
