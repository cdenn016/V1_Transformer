# Gauge-Theoretic VFE Transformer

Gauge-covariant variational free energy transformer for language modeling. No neural network components — all representational capacity from iterative VFE minimization over Gaussian belief tuples `(mu, Sigma, phi)`. See `README.md` for theoretical framework, experimental results, and installation.

## Hard Constraints

**NO NEURAL NETWORKS**: No `nn.Linear`, no MLPs, no learned W_Q/W_K/W_V projections, no activation functions (GELU, ReLU, etc.). The only retained neural component is a linear output projection from K dimensions to vocabulary size. If you are tempted to add an MLP or activation function, you are violating the core thesis. **Documented exceptions**: `connection.py` MLP mode (optional non-flat transport research variant, bilinear default is constraint-compliant), `attention.py` `use_output_projection` (off by default, ablation-only option).

**NO CLI ARGUMENTS (with one documented exception)**: Entry points use the click-to-run pattern. Edit config dicts directly in the file, then press Run. Do not add `argparse`, `click`, `typer`, or any CLI flag parsing to *new* entry points. **Exception**: `transformer/train_publication.py` uses `argparse` for `--mode` / `--ffn_mode` / `--device` / `--checkpoint_dir` / `--seed` / `--dataset` / `--semantic_analysis_interval` because it dispatches between many preset configurations defined as dicts inside the same file. The CLI here selects which config dict to run, it does not expose tunable hyperparameters — those still live in the file-level dicts. Do not extend this exception to other entry points, and do not add new `add_argument` calls that bypass the dict-based config contract.

**PRESERVE GAUGE EQUIVARIANCE**: Covariance transport must always use the sandwich product: `Sigma_transported = Omega @ Sigma @ Omega.T`. Never transport covariance without the conjugation. This is the single most common correctness bug. diagonal approximation is allowable for speed

**Figures**: ALL Figures should be publication quality by default.

**LOCALLY DEFINED CONFIGS**: User may not be running the config values which match the repo.  always double check what values the user is using!

**Post Edit Policy**:  Always write a post-edit description of all changes made to the codebase as a .md.  The date the edits were made should be in the naming convention of the document.  there should be only one document per day.  you should update the same document as edits are made

**ALWAYS PLAN MODE FIRST**

**ignore DEQ, closed-form, and hebbian paths unless otherwise instructed**

**LOCAL CODEBASE IS THE SOURCE OF TRUTH UNLESS OTHERWISE INSTRUCTED**

**Check for sub-agents, skills, and plug-ins before deploying your own**

## Codebase Map


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

**Free energy**:
```
F = alpha * KL(q_i || p_i)                    # self-coupling: beliefs to priors
  + lambda_h * KL(s_i || h)                   # hyper-prior: models to centroid
  + beta_ij * KL(q_i || Omega_ij * q_j)       # belief coupling (attention)
  + gamma_ij * KL(s_i || Omega_ij * s_j)      # model coupling (meta-cognition)
  - E_q[log p(o | x)]                         # observation likelihood
```

**Attention**: `beta_ij = softmax(-KL(q_i || Omega_ij * q_j) / kappa)`

**Transport**: `Omega_ij = exp(phi_i) * exp(-phi_j)` — covers GL+(K) via product of two exponentials

**Covariance transport**: `Sigma_transported = Omega @ Sigma @ Omega.T` — the sandwich product

**Phi preconditioning**: clip (default, O(n_gen)), cartan (O(n_gen²)), killing (O(n_gen²)), pullback (O(n_gen³)) — see `transformer/core/gauge_preconditioner.py`

**Timescales**: Fast E-step (belief q inference per forward pass) / Slow M-step (prior/model s,p parameter learning via backprop) / Static hyper-prior h (frozen at init, never learned). sigma_p is an M-step parameter — the E-step reads it but must not write gradients to it (detached in VFE iterations). sigma_ce_scale controls the residual CE→sigma_p gradient in decode (0.0 = fully detached).

**Amortization scope**: `amortized_inference=True` flows gradients through prior means (mu_p) only by default. Prior covariances (sigma_p) are detached (`amortize_sigma=False`). The post-loop phi gradient (`_compute_phi_grad`) detaches beliefs (`exact_phi_grad=False`), producing a semi-gradient. Cross-layer cascade is also mean-only: mu_q flows to next layer's mu_prior; sigma_prior stays at embedding value. Set `amortize_sigma=True` for full prior amortization through covariances. Set `exact_phi_grad=True` for the IFT-correct total derivative dF/dphi through the E-step iteration graph.

**EM modes** (`em_phi_mode`): `'amortized'` (default) is a straight-through estimator — gradients flow through the E-step update map. `'E_phi_q'` treats phi as a belief variable: E-step optimizes (mu_q, Sigma_q, phi_q), all detached at the EM boundary. `'M_phi_p'` treats phi as a model parameter: E-step optimizes (mu_q, Sigma_q) only with phi frozen; M-step optimizes phi alongside priors/readout via backprop through the attention coupling term.

## Communication Style

**Humility** You are free to say "i don't know" whenever you are unsure about a response.

**Verify** Use citations to verify theoretical and mathematical suggestions and responses

**quote** use direct quotes for factual grounding

**Be direct.** State errors and concerns plainly. "This is wrong because X" not "This might potentially be slightly off." Always ultra-think and double check.

**Push back.** Challenge gaps in derivations, ask for justification. If a claim needs proof, ask for it. Maintain position under pushback — ask "What am I missing?" rather than capitulating.

**Skip praise preambles.** No "Great question!" openers. No "Excellent point!" — engage with the substance.

**Flag simpler alternatives.** Call out over-engineering. Ask what the complexity buys if something seems unnecessarily elaborate.

**Honest uncertainty.** "I'm not sure this is right" beats confident speculation. Acknowledge when something needs verification.

**No bullshit.** If a correspondence is interpretive rather than mathematically exact, say so explicitly. If something doesn't connect, admit the gap. Remove content that doesn't earn its place through rigorous derivation. Never dress up hand-waving as theorem. When asked "what does X have to do with anything?" — if the answer is "not much", say that.

## Manuscript Style

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
2. **Use natural gradients** for phi parameters — never raw Euclidean gradients on Lie algebra without preconditioning
3. **Test spectral behavior** — new features should maintain or improve spectral trends (effective rank, entropy)
4. **Document math** — include LaTeX notation in docstrings for any non-trivial formula
5. **Domain expertise areas**: differential geometry (SPD manifolds, Lie theory), variational inference (KL, ELBO, information geometry), gauge theory (equivariance, parallel transport, irreps), matrix algebra (eigendecomposition, matrix exponentials)
