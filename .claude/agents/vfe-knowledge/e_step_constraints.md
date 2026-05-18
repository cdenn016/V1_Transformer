# E-step Constraints — User's Implementation Choices

**Descriptive file.** Documents what the user's code commits to do. Agents evaluate these choices against the standard variational-EM literature (see `external_canon_inference.md` §5 and §6) and the standard manifold-optimization literature (see `external_canon_transformers.md` §8). Most of the constraints below are *standard practice* and the user's code follows them correctly — but the agents must verify that against the standard, not just against this file.

# E-step Constraints — Inference Hygiene and SPD Retraction

## Constraint 1 — E-step must not see targets

Inference computes `q` (beliefs) from inputs alone. The likelihood term `−E_q[log p(o|x)]` enters F at training, but the E-step iterations themselves use only the prior, transport, and cross-site couplings — **never the labels**.

**Audit grep targets:** in any function that runs the E-step iterations (e.g., `vfe/e_step.py::EStep.run`, `core/variational_ffn.py`), search for `targets`, `labels`, `y`, `ground_truth`. None should appear inside the iteration loop. The likelihood term may be added *after* iterations complete (M-step gradient) but must not influence the iterates themselves.

## Constraint 2 — σ_p is M-step parameter; E-step reads but does not write

`sigma_p` is updated by backprop on F (M-step). During E-step iterations, gradients computed for `μ_q, σ_q, φ` must not flow into `sigma_p`. Implementations typically `.detach()` `sigma_p` at the start of each E-step iteration.

`sigma_ce_scale` controls a residual CE→σ_p gradient in decode:

- `0.0` ⇒ fully detached (the theoretically clean default).
- `> 0.0` ⇒ partial gradient. Opt-in. Should be documented in the active config.

**Audit check:** if `sigma_ce_scale > 0.0`, verify the user knows. If `sigma_ce_scale == 0.0`, verify the detach is actually applied in code (a missing `.detach()` is a regression).

## Constraint 3 — decoupled learning rates (post 2026-05-13)

- `E_mu_q_lr` — μ retraction LR. Independent of σ LR.
- `E_sigma_q_lr` — σ retraction LR (multiplicative on `σ`).
- `E_sigma_q_trust` — trust-region clamp on whitened tangent `δσ/σ`. Default `5.0` (matches the historical `retract_spd_diagonal_torch` default).

σ retraction step:
```
σ_new = σ · exp( E_sigma_q_lr · decay_t · clamp( δσ/σ, ±E_sigma_q_trust ) )
```

`decay_t` is the same cosine factor used for the μ LR. The trust-region clamp is now a **separate field**, not the same scalar.

**Pre-2026-05-13 behavior (regression to flag):** `sigma_lr` did double duty as the trust-region clamp; sweeping σ LR had no visible effect unless the clamp was binding. If code reuses `sigma_lr` for both purposes, that is stale.

## Constraint 4 — SPD retraction on full Σ

For full covariance, retraction must keep `Σ` positive definite. Standard recipe: parameterize Σ via its Cholesky factor `L` and retract on `L`, or work in the log-Cholesky parameterization. Naive Euclidean steps on `Σ` can break PSD.

For diagonal Σ (the common path here), the exponential retraction above keeps σ > 0 automatically.

**Audit check:** wherever full covariance is updated, look for the retraction. If a step like `Sigma_new = Sigma + lr * grad` exists without subsequent PSD projection, that is a Critical bug.

## Constraint 5 — fixed-point vs amortized vs IFT (don't conflate)

Three approximations to ∇_θ F (where θ are M-step params):

- **Amortized:** compute one E-step forward pass, backprop through it. Cheap, biased (the E-step is incomplete).
- **Fixed-point:** iterate E-step to convergence, then backprop through the unrolled trajectory. Memory-expensive.
- **IFT (Implicit Function Theorem):** iterate E-step to convergence, then compute the M-step gradient via implicit differentiation at the fixed point. Memory-cheap, asymptotically exact.

The `em_mode='ift_phi'` label means **IFT applied specifically to φ** — not "implicit FFN" or "implicit transformer." A single-step approximation labeled `ift_phi` is **drift**.

**Audit check:** in `core/variational_ffn.py` or wherever `ift_phi` is implemented, verify the implicit gradient is solved (typically via a linear system involving the Jacobian of the E-step fixed-point map), not just a single backprop through one iteration.

## Natural gradient on Lie algebra

φ lives in `gl(K)` (or a chosen Lie subalgebra). Natural gradient on a Lie algebra requires the appropriate metric. Raw Euclidean gradient on φ without preconditioning is theoretically incorrect — it ignores the curvature of the manifold.

Project CLAUDE.md `## Contributing` rule 2: "Use natural gradients — never raw Euclidean gradients on Lie algebra without preconditioning."

**Audit check:** in `core/gauge_preconditioner.py`, verify a preconditioner is applied to φ gradients. Common preconditioners: Fisher info, identity (only if explicitly justified), or a learned metric.

## RoPE × MahalanobisNorm known gap

Documented limitation, not a bug: when `diagonal_covariance=True` AND `use_rope=True` AND `rope_full_gauge='off'`, RoPE rotates μ but not σ. Downstream `MahalanobisNorm(μ, σ)` then divides rotated μ by un-rotated σ, breaking strict SE(K) covariance for that combination.

`vfe/config.py::__post_init__` forbids non-`'off'` values of `rope_full_gauge` in the diagonal-σ path. **Audit check:** verify the forbid is still present.

**Manuscript check:** any claim of strict SE(K) covariance for the diagonal+RoPE path is wrong. Either the manuscript must disclaim this combination or it must derive a corrected MahalanobisNorm that uses rotated σ.
