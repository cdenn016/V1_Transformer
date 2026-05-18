# Gauge-Theoretic VFE Transformer

Gauge-covariant variational free energy transformer for language modeling. No neural network components — all representational capacity from iterative VFE minimization over Gaussian belief tuples `(mu, Sigma, phi)`. See `README.md` for theoretical framework, experimental results, and installation.

## Hard Constraints

**NO NEURAL NETWORKS**: No `nn.Linear`, no MLPs, no learned W_Q/W_K/W_V projections, no activation functions (GELU, ReLU, etc.). The only retained neural component is a linear output projection from K dimensions to vocabulary size (subsumed by the PriorBank decode, `logits = -KL(q || π_v)/τ`, when `use_prior_bank=True`). If you are tempted to add an MLP or activation function, you are violating the core thesis. **Documented exceptions**: `connection.py` MLP mode (optional non-flat transport research variant; bilinear default is constraint-compliant)

**NO CLI ARGUMENTS (with one documented exception)**: Entry points use the click-to-run pattern. Edit config dicts directly in the file, then press Run. Do not add `argparse`, `click`, `typer`, or any CLI flag parsing to *new* entry points.

**PRESERVE GAUGE EQUIVARIANCE**: Covariance transport must always use the sandwich product: `Sigma_transported = Omega @ Sigma @ Omega.T`. Never transport covariance without the conjugation. This is the single most common correctness bug. diagonal approximation is allowable for speed

**E-STEP MUST NOT SEE TARGETS**

**KNOWN GAP — RoPE × MahalanobisNorm**: When `diagonal_covariance=True` AND `use_rope=True` AND `rope_full_gauge='off'` (the diagonal-σ path forbids non-`'off'` values in `vfe/config.py::__post_init__`), RoPE rotates μ but not σ. Downstream `MahalanobisNorm(μ, σ)` in `vfe/block.py` then divides rotated μ by un-rotated σ, breaking strict SE(K) covariance for that combination. Acceptable as documented research limitation.

**Figures**: ALL Figures should be publication quality by default.

**LOCALLY DEFINED CONFIGS**: User may not be running the config values which match the repo.  always double check what values the user is using!

**Post Edit Policy**:  Always write a post-edit description of all changes made to the codebase as a .md.  The date the edits were made should be in the naming convention of the document.  there should be only one document per day.  you should update the same document as edits are made

**ALWAYS PLAN MODE FIRST**

**There should ALWAYS exist a theoretically/mathematically "pure" path under appropriate toggles.**  Computationally extreme paths should be 'opt in' toggles and clearly documented.

**Check for sub-agents, skills, and plug-ins before deploying your own. 

**check claude-mem for prior session context when resuming work on a topic**

Before you say the fix is done: (1) open my active config file, (2) trace every relevant key through the config loader and any override logic, (3) confirm the exact line you changed is reached at runtime under my config, (4) only then run tests and report.

**CODE FOCUS** when investigating and/or auditing the codebase do NOT rely on code comments....focus on the actual code and paths


##Agent policy

-always deploy parallel investigation agents first and then a separate 'verifier' agent. The verifier will check the discoveries of the investigators  

## Code Conventions

- Type hints on all function signatures
- Docstrings with LaTeX math where relevant (e.g., `r"""Computes KL(q || p) = ..."""`)
- Variable names match paper notation: `mu_q`, `Sigma`, `phi`, `Omega`, `kappa`, `alpha`, `beta_ij`, `gamma`
- Tensor shape comments at critical points in attention, transport, and KL computations
- `BlockConfig` is the single source of truth for gauge group structure (generators, irrep dims, head count)
- `TrainingConfig` is the single source of truth for training hyperparameters (learning rates, schedules, VFE weights)
- Validate gradient correctness with small-dim finite-difference smoke tests

## Mathematical Reference

Minimal equations for code review — see `README.md` for full derivations.

**VFE hierarchy**: `h → s → p → q → observations` (hyper-prior → models → priors → beliefs → data)

**Free energy** (canonical form, manuscript `\label{eq:free_energy_functional_final}`):
```
F = alpha * KL(q_i || p_i)                                          # self-coupling: beliefs to priors
  + lambda_h * KL(s_i || h)                                         # hyper-prior: models to centroid
  + sum_ij [ beta_ij  * KL(q_i || Omega_ij * q_j)
             + tau * beta_ij  * log(beta_ij  / pi_ij) ]             # belief coupling + attention entropy
  + sum_ij [ gamma_ij * KL(s_i || Omega_ij * s_j)
             + tau * gamma_ij * log(gamma_ij / pi^(s)_ij) ]         # model coupling + meta entropy
  - E_q[log p(o | x)]                                               # observation likelihood
```

`tau = kappa * sqrt(K)` is the effective softmax temperature. The `tau * beta_ij * log(beta_ij/pi_ij)` term is the attention-distribution entropy with uniform prior `pi_ij = 1/N`; it is required for the softmax β to be a stationary point of F (without it the row-Lagrangian gives a delta, not softmax). The canonical F vs "entropy-suppressed surrogate" `sum β KL` distinction is made explicitly in `Attention/GL(K)_attention.tex` at lines 766, 855, and 1354 (the surrogate is acknowledged again in `Attention/GL(K)_supplementary.tex` §B.1) — their gradients differ by `-tau^{-1} Cov_β(KL, ∇KL)`.

**Attention**: `beta_ij = softmax(-KL(q_i || Omega_ij * q_j) / (kappa * sqrt(K)))` — `kappa` is a learnable hyperparameter; the `sqrt(K)` factor is intentional dimension scaling on top of `kappa` (analogous to scaled dot-product attention).

**Transport**: `Omega_ij = exp(phi_i) * exp(-phi_j)` — covers GL+(K) via product of two exponentials

**Covariance transport**: `Sigma_transported = Omega @ Sigma @ Omega.T` — the sandwich product


**Timescales**: Fast E-step (belief q inference per forward pass) / Slow M-step (prior/model s,p parameter learning via backprop) / Static hyper-prior h (frozen at init, never learned). sigma_p is an M-step parameter — the E-step reads it but must not write gradients to it (detached in VFE iterations). sigma_ce_scale controls the residual CE→sigma_p gradient in decode (0.0 = fully detached).

**E-step LRs are decoupled (2026-05-13 onward).** `E_mu_q_lr` controls the μ retraction step size; `E_sigma_q_lr` controls the σ retraction step size — both modulated by the same cosine decay factor across iterations but independent in magnitude. Earlier code (pre-2026-05-13) used `sigma_lr` only as a trust-region clamp on the whitened tangent δσ/σ, leaving the μ LR (`self.lr`) as the actual σ step — sweeping `E_sigma_q_lr` produced no visible effect when the clamp wasn't binding. The trust-region clamp is now a separate field `E_sigma_q_trust` (default 5.0, matches the historical `retract_spd_diagonal_torch` default). The σ retraction is `σ_new = σ · exp(E_sigma_q_lr · decay_t · clamp(δσ/σ, ±E_sigma_q_trust))`.

**Attention sublayer is OPTIONAL.** The pure VFE architecture (`skip_attention=True`) is the theoretically clean form: the FFN's E-step computes its own β internally and updates beliefs. The separate attention sublayer at the top of `GaugeTransformerBlock.forward` is an engineering heuristic (β-weighted message aggregation through `W_O · μ_agg` residual). `skip_attention=True` works cleanly with `em_mode='ift_phi'` (default). It is INCOMPATIBLE with the detaching EM modes `em_phi_p`, `em_phi_q` — those modes detach σ_p and/or φ inside the FFN, so the attention sublayer is their sole autograd path back to `sigma_embed` and `phi_embed`. With `skip_attention=True` AND a detaching mode, σ_embed and φ_embed silently stay frozen at initialization. `BlockConfig.__post_init__` warns when this combination is detected.

**EM modes** (`em_mode`): Single string selector controlling gradient flow at the EM boundary. Replaces the old 5-flag system (`amortized_inference`, `amortize_sigma`, `exact_phi_grad`, `implicit_em`, `em_phi_mode`). Prior covariance sigma_p is always attached in amortized modes. Cross-layer cascade: mu_q flows to next layer's mu_prior; sigma_prior stays at embedding value.

| `em_mode` | mu_p | sigma_p | phi | At EM exit |
|-----------|------|---------|-----|------------|
| `'ift_phi'` (default) | attached | attached | full IFT | attached |
| `'em_phi_q'` | detached | detached | evolves in E-step | all detached |
| `'em_phi_p'` | detached | detached | frozen in E-step | mu,sigma detached |
| `'vfe_default'` (transformer/vfe/) | attached | frozen at embedding¹ | full autograd | attached |

¹ The `transformer/vfe/` package operates in its own gradient profile that does not correspond to any of the three `em_mode` values above. `mu_p` is attached (the previous layer's posterior `mu_q` becomes the next layer's `mu_p` via `vfe/stack.py`); `sigma_p` is structurally frozen at the embedding value when `prior_handoff_sigma=0` (the default); `phi` is updated each E-step iteration via `_update_phi` in `vfe/e_step.py`, which constructs a fresh detached leaf `phi_for_grad = phi.detach().requires_grad_(True)` for the local alignment-loss backward, then retracts `phi` and reassigns it. The iteration sequence as a whole is autograd-tracked, but `phi` is not globally cloned per outer iteration. The `vfe/` package does not currently honor `em_mode` switching — it is hardwired to this profile. Selecting any other `em_mode` requires routing through the legacy `transformer/core/variational_ffn.py` path.

## Communication Style

**Humility** say "i don't know" whenever you are unsure about anything.

**Verify** Use citations to verify theoretical and mathematical suggestions and responses

**Be direct.** State errors and concerns plainly. "This is wrong because X" not "This might potentially be slightly off." Always ultra-think and double check.

**Push back.** Challenge gaps in derivations, ask for justification. If a claim needs proof, ask for it. Maintain position under pushback — ask "What am I missing?" rather than capitulating.

**Skip praise preambles.** No "Great question!" openers. No "Excellent point!" — engage with the substance.

**Flag simpler alternatives.** Call out over-engineering. Ask what the complexity buys if something seems unnecessarily elaborate.

**Honest uncertainty.** "I'm not sure this is right" beats confident speculation. Acknowledge when something needs verification.

**No bullshit.** If a correspondence is interpretive rather than mathematically exact, say so explicitly. If something doesn't connect, admit the gap. Remove content that doesn't earn its place through rigorous derivation. Never dress up hand-waving as theorem. When asked "what does X have to do with anything?" — if the answer is "not much", say that.

## Style

Add under an existing ## Documentation or ## Writing Style section, or create one\n\n## Scientific Writing Rules
- Do NOT use LaTeX spacing macros like `\;`, `\,`, `\!` in equations — these are banned in this project's docs.
- Apply standard equation punctuation (comma/period at end of display equations) as part of any doc cleanup pass.

Write in academic prose, not bullet points. Use flowing paragraphs with clear logical progression. Use /literature-review, /scientific-writing, /sympy, and other relavent skills

**Banned patterns:** horizontal rules (`---`), "key insight", "crucially", "critically", "notably", "importantly", "it's worth noting", "interestingly", "fundamentally", "in particular", "leverages", "underscores". These are Claude-isms — never use them in manuscripts.

Minimize itemizations, lists, and enumerations. If content can be expressed as a paragraph, express it as a paragraph. Remove content that doesn't earn its place through rigorous derivation.

1. Think Before Coding
Don't assume. Don't hide confusion. Surface tradeoffs.

Before implementing:

State your assumptions explicitly. If uncertain, ask.
If multiple interpretations exist, present them - don't pick silently.
If a simpler approach exists, say so. Push back when warranted.
If something is unclear, stop. Name what's confusing. Ask.

2. Simplicity First
Minimum code that solves the problem. Nothing speculative.

No features beyond what was asked.
No abstractions for single-use code.
No "flexibility" or "configurability" that wasn't requested.
No error handling for impossible scenarios.
If you write 200 lines and it could be 50, rewrite it.
Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

3. Surgical Changes
Touch only what you must. Clean up only your own mess.

When editing existing code:

Don't "improve" adjacent code, comments, or formatting.
Don't refactor things that aren't broken.
Match existing style, even if you'd do it differently.
If you notice unrelated dead code, mention it - don't delete it.
When your changes create orphans:

Remove imports/variables/functions that YOUR changes made unused.
Don't remove pre-existing dead code unless asked.
The test: Every changed line should trace directly to the user's request.

4. Goal-Driven Execution
Define success criteria. Loop until verified.

Transform tasks into verifiable goals:

"Add validation" → "Write tests for invalid inputs, then make them pass"
"Fix the bug" → "Write a test that reproduces it, then make it pass"
"Refactor X" → "Ensure tests pass before and after"
For multi-step tasks, state a brief plan:

1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification


## Contributing

1. **Preserve gauge equivariance** — covariance transport is always `Omega @ Sigma @ Omega.T`
2. **Use natural gradients**  never raw Euclidean gradients on Lie algebra without preconditioning
3. **Test spectral behavior** — new features should maintain or improve spectral trends (effective rank, entropy)
4. **Document math** — include LaTeX notation in docstrings for any non-trivial formula
5. **Domain expertise areas**: differential geometry (SPD manifolds, Lie theory), variational inference (KL, ELBO, information geometry), gauge theory (equivariance, parallel transport, irreps), matrix algebra (eigendecomposition, matrix exponentials)
