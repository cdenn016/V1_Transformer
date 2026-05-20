# Codebase Reuse Audit — Canonical Active-Inference Transformer

**Date:** 2026-05-19
**Author:** Investigation Agent 2 (codebase-auditor)
**Source-of-truth for verdict:** `docs/debates/2026-05-19-vfe-active-inference-impl/04_verdict.md` (RED_WINS).
**Citations in this document:** `path:line` for codebase claims; `[ParrPezzuloFriston2022]`, `[Friston2010]`, `[FristonEtAl2017]`, `[Friston2021SophisticatedInference]`, `[Amari1998]` for theoretical claims.

## §1. Existing primitives — REUSE candidates

### 1.1 `transformer/core/types.py:11 — BeliefState`

```python
class BeliefState(NamedTuple):
    mu: torch.Tensor
    sigma: torch.Tensor
    phi: torch.Tensor
    omega: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
```

Container for `(mu, sigma, phi[, omega])`. `mu` is `(B, N, K)`; `sigma` is `(B, N, K)` diagonal or `(B, N, K, K)` full; `phi` is `(B, N, n_gen)` Lie-algebra coordinates; `omega` is an optional per-block group-level state used only under `cfg.gauge_parameterization == 'omega_direct'`.

**Reusable AS IS.** Holds the full Gaussian belief tuple per token position. The omega field is already engineered to be optional and back-compatible. For an AIF tree-search, each node will store a `BeliefState` pointer — the existing NamedTuple is the right unit.

E-step blindness: this is a pure data container, no targets channel. Sandwich-product compliance is enforced by *consumers*, not by the container itself; the container itself is gauge-neutral.

### 1.2 `transformer/vfe/prior_bank.py:400 — VFEPriorBank.encode`

```python
def encode(self, token_ids: torch.Tensor) -> BeliefState:
    phi = self.phi_embed(token_ids)            # (B, N, n_gen)
    phi = self._apply_phi_det_control(phi)
    if self.gauge_fixed_priors:
        block_exp_pairs = self._compute_block_exp_pairs(phi)
        mu_p, sigma_p = self._apply_gauge_transform(block_exp_pairs)
    else:
        mu_p, sigma_diag = self._per_token_priors(token_ids)
        ...
    return BeliefState(mu=mu_p, sigma=sigma_p, phi=phi)
```

Maps `token_ids (B, N)` to a `BeliefState`. Two modes: gauge-fixed (shared base prior lifted per-token by `A_v = exp(phi_v . G)`, sandwich-product covariance at `transformer/vfe/prior_bank.py:321 sigma_h = exp_h_f32 @ sigma_diag @ exp_h_f32.transpose(-2, -1)`); direct (per-token `(mu_v, sigma_v)` lookup).

**Reusable AS IS** for the AIF build-out. Every node expansion that commits to a candidate token `a` will call `encode([..., a])` (or, more efficiently, append `a` to the prior cache and re-encode just position N+1). The sandwich product at line 321 is explicit and correct. E-step blindness holds — the only input is `token_ids`, no targets channel.

Gauge equivariance: confirmed at line 321 (full-cov path) and at the diagonal approximation line 326 `(exp_h_f32 ** 2 @ base_sigma_h).clamp(min=self.eps)` which is the CLAUDE.md-permitted diagonal speed approximation.

### 1.3 `transformer/vfe/prior_bank.py:427 — VFEPriorBank.decode`

```python
def decode(self, mu_q, sigma_q, tau=1.0) -> torch.Tensor:
    # KL-to-prior: logits[i, v] = -c * KL(q_i || pi_v) / tau
    # Fused matmul: (B,N,2K) @ (V,2K).T over sigma_q + mu_q^2 / -2 mu_q  vs  1/sigma_p, mu_p/sigma_p
    ...
    scale = torch.exp(self.decode_log_scale.clamp(-3.0, 3.0))
    logits = -0.5 * scale / tau * (combined + prior_bias.unsqueeze(0).unsqueeze(0))
    return logits
```

Diagonal-KL readout, fused single-matmul over the whole vocab. Returns `(B, N, V)` logits — this IS `log p(o | s)` up to a vocabulary-uniform additive constant (the `-K + log|Sigma_q|` terms that cancel in softmax are dropped at `transformer/vfe/prior_bank.py:448`).

The decode rescales canonical KL by `c = exp(decode_log_scale)` clamped to `[exp(-3), exp(3)]` (`prior_bank.py:25`). The published / canonical form `logits = -KL / tau` corresponds to `c = 1`; this is initialised at zero so untrained models match the canonical form bitwise. This is a documented novel construction (see `prior_bank.py:22-34` module-level "Implementation note").

**Reusable WITH EXTENSION:**

- For canonical EFE risk under the `current_belief` preference, the existing `compute_risk` (see §1.7) consumes the softmax of `decode` output directly — no change needed.
- For the **canonical** pragmatic value `E_q[-log p^*(o | C)]` with an exogenous preference `p^*`, the AIF module needs the *log-prob* readout (the un-normalized log-prior of each token) for KL/cross-entropy against `p^*`. The current `decode` returns negative-KL-scaled logits, not `log p(o | s)`. Either reuse the existing logits-into-softmax pipeline (compute `q(o|a) = softmax(decode(...))` and then `cross_entropy(q, p^*)`), or expose a new helper that returns the per-token KL matrix directly. Reuse the softmax path; do not add a new logit form.

### 1.4 `transformer/vfe/e_step.py:725 — VFEEStep.forward`

```python
def forward(
    self,
    beliefs: BeliefState,
    priors: BeliefState,
    mask: Optional[torch.Tensor] = None,
    active_inference_fn: "Optional[ActiveInferenceFn]" = None,
) -> BeliefState:
    # Law 1 enforced: No `targets` parameter exists.
```

The signature is explicit: there is no `targets` parameter. The docstring at `transformer/vfe/e_step.py:734` states "Law 1 enforced: No targets parameter exists. Target leakage is structurally impossible." `sigma_p` is detached at extraction (`transformer/vfe/e_step.py:790`) per CLAUDE.md "sigma_p is an M-step parameter — the E-step reads it but must not write gradients to it".

Per-iteration body (lines 833 onward): compute transport via `compute_gauge_transport`, compute KL attention `beta_ij`, compute VFE gradients `(grad_mu, grad_sigma)`, optionally add `active_inference_fn(mu, sigma)` contributions, apply natural-gradient projection, retract sigma on SPD, update phi.

**Reusable AS IS for canonical EFE belief-state-update mechanics.** This is the right primitive to call inside the AIF tree expansion: after committing to a candidate token `a`, append `a` to the context, encode, run the E-step on the extended context, and use the converged `(mu, sigma, phi)` at the new last position as the next-state belief.

Sandwich-product compliance: transport is computed via fused block-diagonal kernels (`_fused_attention_and_vfe_gradients_block_diag`, lines 1006-1028) which internally apply `Omega @ Sigma @ Omega.T`; the docstring at `e_step.py:10` states "Law 2 enforced: all transport goes through fused block-diagonal kernels which internally compute Omega @ Sigma @ Omega^T."

The `active_inference_fn` callback hook (`e_step.py:730`) is currently used by `VFEActiveInference` (the non-canonical surrogate flagged in §2). **The hook itself is a clean entry point.** A canonical EFE-augmented training pass (option 2 in the context's "Scope decisions deferred") could supply a canonical-EFE callback through the same slot — but only after the renaming in §2 cleanly separates the surrogate from any canonical path.

### 1.5 `transformer/vfe/stack.py:76 — VFEStack.forward`

```python
def forward(
    self,
    beliefs: BeliefState,
    initial_priors: BeliefState,
    mask: Optional[torch.Tensor] = None,
    active_inference_fn: Optional[ActiveInferenceFn] = None,
) -> BeliefState:
    priors = initial_priors
    for block in self.blocks:
        beliefs = block(beliefs, priors, mask, active_inference_fn)
        # cross-layer handoff: rho_mu * beliefs.mu + (1-rho_mu) * priors.mu;
        # sigma frozen at embedding when rho_sigma=0; phi frozen at embedding.
        ...
        priors = BeliefState(mu=new_prior_mu, sigma=new_prior_sigma, phi=new_prior_phi)
    return beliefs
```

Multi-layer wrapper with cross-layer prior handoff (`stack.py:107-160`). Defaults `rho_mu=1.0, rho_sigma=0.0`: the posterior mean of layer L flows into the prior of layer L+1; the prior sigma and prior phi reuse the embedding values. Per the `stack.py:19-34` docstring this is a "*point-estimate handoff*, not the full distributional handoff that canonical hierarchical variational inference (Friston 2017; Parr, Pezzulo, Friston 2022; Blei, Kucukelbir, Jordan 2017) prescribes" and is a documented mean-only cascade.

**Reusable AS IS for the AIF build-out's intra-context E-step.** Each tree-search node already calls `VFEModel.forward` which internally calls `VFEStack.forward`; nothing in the AIF generator needs to touch this directly except to be aware that the prior-cascade convention is mean-only at the default config.

For sophisticated-inference style recursion through multi-step futures `[Friston2021SophisticatedInference]`, the per-step belief update at the *generation level* is `s_{t+1} = E-step on context || a_t`. The intra-context multi-layer cascade in `VFEStack` is orthogonal to the inter-token state evolution — confirm at the architecture design step (Agent 3) that these two "layers" of nesting do not get conflated.

### 1.6 `transformer/vfe/model.py:126 — VFEModel.forward`

```python
def forward(
    self,
    token_ids: torch.Tensor,
    targets: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    # 1. encode → 2. positional → 2b. (omega_direct init) → 3. priors snapshot
    # → 4. causal mask → 5. stack → 6. final norm → 7. decode
```

Full pipeline: encode (`model.py:159`), positional BCH (`model.py:162-164`), optional omega-direct init (`model.py:172-177`), priors snapshot (`model.py:182-186`), causal mask (`model.py:189-190`), stack forward with optional `active_inference_fn` (`model.py:193-196`), final norm (`model.py:199-201`), decode (`model.py:204-215`).

Target-blindness: the `targets` parameter at line 129 is consumed ONLY at line 217 (`F.cross_entropy(logits, targets)`) AFTER all E-step computation has completed. Compliant with CLAUDE.md Law 1.

**Reusable AS IS as the AIF context-encoding primitive.** Each AIF tree-search node calls `model.forward(context_ids_for_this_node)` (with `targets=None`) to obtain the predictive distribution over the next token. This is exactly what the existing `VFEExpectedFreeEnergy.score_candidates` and `compute_efe` already do (see §1.7-1.8).

Sandwich product: enforced at every consumer of the returned `BeliefState` (E-step, decode); the model itself does not bypass it.

### 1.7 `transformer/vfe/model.py:272 — VFEModel.generate`

```python
@torch.no_grad()
def generate(
    self,
    prompt_ids: torch.Tensor,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int = 50,
    use_efe: bool = False,
    efe_gamma: float = 1.0,
) -> torch.Tensor:
    # Branches: use_efe=True → VFEExpectedFreeEnergy.select_action (depth-1 only)
    # Otherwise: temperature + top-k sampling on raw model logits.
```

Standard greedy/top-k generation by default. With `use_efe=True`, depth-1 EFE-weighted action selection per the canonical formulation in `VFEExpectedFreeEnergy`.

**Structural gap.** This is depth-1 only. Canonical sophisticated inference [Friston2021SophisticatedInference] requires recursive EFE evaluation over a planning horizon `D > 1`. The current generate loop does not maintain a belief-state cache across multi-token rollouts — it re-runs the full forward each step.

**Reusable AS the scaffolding for an AIFGenerator wrapper.** The wrapper either replaces `VFEModel.generate` entirely or subclasses it. Do not edit `generate` in place — keep the depth-1 EFE path for the user's current generation needs; add a new `aif_generate` that performs the tree-search.

### 1.8 `transformer/vfe/efe.py:28 — VFEExpectedFreeEnergy`

```python
class VFEExpectedFreeEnergy:
    def __init__(self, model, gamma=1.0, preference_mode='uniform',
                 epistemic_weight=0.0, epistemic_samples=4): ...
    @torch.no_grad()
    def score_candidates(self, context_ids, candidate_ids) -> Dict[str, torch.Tensor]: ...
    @torch.no_grad()
    def select_action(self, context_ids, top_k=50, temperature=1.0) -> int: ...
```

Depth-1 canonical EFE at generation time. `G(a) = risk + ambiguity - epistemic_weight * epistemic`. Per-candidate rollout: append candidate to context, re-encode via `prior_bank.encode`, re-apply positional via `self.model.pos_enc`, then compute BALD MI (`efe.py:73-134`) and risk through `compute_risk` (`efe.py:178-184`).

Ambiguity bug-fix: `efe.py:191-196` correctly returns `mean_H = E_q(z)[H[p(o|z)]]` as the canonical ambiguity per `[ParrPezzuloFriston2022]`, not the (incorrect) marginal-predictive entropy `H[\bar p]` which exactly cancels risk under uniform preferences (see `expected_free_energy.py:117-127` for the prose explanation of that pitfall).

**Reusable WITH EXTENSION for canonical AIF.** This is the canonical depth-1 case. The build-out needs depth-D generalisation:

- The per-candidate rollout pattern at `efe.py:164-194` is the right primitive to call recursively at each node. But it re-encodes the entire context on every candidate, which makes deep recursion intractable without belief caching (see §3.3).
- `preference_mode='uniform'` default reduces risk to `log V - H[q]` (`expected_free_energy.py:94-98`), constant offset; this is consistent with the no-exogenous-preference language-modeling setting that Agent 1 must justify.
- `preference_mode='current_belief'` ties the preference back to the model's own marginal, which is a self-evidencing reduction — flag for Agent 1 to determine if this counts as canonical under `[ParrPezzuloFriston2022]` or is a softer variant of the same surrogate identified in `04_verdict.md`.

### 1.9 `transformer/core/expected_free_energy.py — module-level helpers`

`compute_risk` (`expected_free_energy.py:48-109`): the canonical risk computer per `[ParrPezzuloFriston2022]`. Three preference modes:

- `'target'` — `risk = -log q(o=y|a)` at the target token. Used for teacher-forced EFE-proxy training.
- `'uniform'` — `risk = log V - H[q]`.
- `'current_belief'` — `risk = -sum_v q(v|a) log p^*(v)` with `p^* = current marginal`.

`compute_ambiguity` (`expected_free_energy.py:112-137`): returns `H[q_marginal]` and *documents in the docstring* (lines 117-126) that this is NOT the canonical ambiguity. Retained for back-compat; canonical consumers route through `compute_epistemic_value(..., return_mean_H=True)` (line 218) which returns the canonical `mean_H = E_q(z)[H[p(o|z)]]`.

`compute_epistemic_value` (`expected_free_energy.py:140-220`): BALD mutual information via Welford-style streaming MC. Now also returns `mean_H` (the canonical ambiguity term) when `return_mean_H=True`.

`rollout_candidates` (`expected_free_energy.py:227-296`): batched K-candidate rollout in a single forward pass. Used by `compute_efe`.

`compute_efe` (`expected_free_energy.py:303-404`) and `generate_active_inference` (`expected_free_energy.py:411-534`): the full depth-1 EFE pipeline for the *core* (legacy) model. The `vfe/efe.py` is the per-step equivalent for the `/vfe` model. Both share the same `compute_risk` primitive but live in different model wrappers.

**Reusable AS IS for canonical EFE primitives.** `compute_risk` is the canonical risk per `[ParrPezzuloFriston2022]`; `compute_epistemic_value` (with `return_mean_H=True`) supplies both BALD MI and canonical ambiguity in one MC pass. Both should be the foundation of the new AIF module's per-node EFE.

`compute_efe` and `generate_active_inference` are tied to the *core* model's `_compute_logits` API (`expected_free_energy.py:195`). The new AIF module should not reuse `compute_efe` directly because it depends on the legacy model interface — instead, build a new orchestrator that calls `compute_risk` / `compute_epistemic_value` against the `/vfe` model's PriorBank decode.

## §2. Existing primitives — REPLACE / DEPRECATE candidates

Per `04_verdict.md` action item 1, the surrogate self-evidencing-regularizer path is to be renamed and clearly separated from any canonical EFE path.

### 2.1 `transformer/vfe/active_inference.py:27 — VFEActiveInference`

```python
class VFEActiveInference(nn.Module):
    # F_AI = lambda_prag * H[p_pred(v | mu_i)]  - lambda_epi * MI(v; mu | q_i)
```

Pragmatic term is `H[p_pred(v | mu_i)]` — the entropy of the model's OWN predictive at the current `mu_i`. Per `04_verdict.md` decisive evidence and the in-code disclosure at `transformer/core/active_inference.py:28-37`, this is NOT the canonical EFE pragmatic value `E_q[-log p^*(o | C)]` from `[ParrPezzuloFriston2022]`. It is a self-evidencing surrogate.

**Action (per verdict item 1):**

- Rename class: `VFEActiveInference` → `VFESelfEvidencingRegularizer`.
- Rename config: `active_inference` → `self_evidencing_regularizer`; `pragmatic_weight` → `self_confidence_weight`. Retain a deprecation alias for one release.
- The epistemic / BALD MI half is canonical and keeps its name.
- Restrict the user-facing claim in `transformer/vfe/train_vfe.py:104-108` to "self-evidencing surrogate at the E-step plus canonical EFE at generation time."

After rename, this module continues to exist on the research track. The new canonical-AIF module is a parallel, separately-named addition.

### 2.2 `transformer/core/active_inference.py:94 — _compute_active_inference_gradient`

Same surrogate kernel as §2.1, but operating on the *core* legacy model. The in-file disclosure at lines 28-37 is the decisive evidence in `04_verdict.md`. Rename to `_compute_self_evidencing_gradient` (or similar) and restrict its use to the renamed surrogate path.

### 2.3 No other surrogate paths identified

`transformer/core/expected_free_energy.py` is the canonical pipeline and stays as is.

`transformer/vfe/efe.py:VFEExpectedFreeEnergy` is the canonical depth-1 pipeline for the `/vfe` model and stays as is (extension candidate, not replacement).

The `compute_ambiguity` helper at `expected_free_energy.py:112-137` is documented as non-canonical (line 117-126: "NOTE: Canonical EFE ambiguity is ..., NOT ..."); it is retained for back-compat. Mark it deprecated; new code should use `compute_epistemic_value(..., return_mean_H=True)`. This is not a critical replacement — the current callers already do so (`expected_free_energy.py:380-386`).

## §3. Structural gaps — must be freshly written

### 3.1 Policy data structure

There is no Policy class anywhere in `transformer/`. `grep -rn "class Policy\|class Plan\|class Trajectory" transformer/` returns nothing. A canonical AIF policy in the `[ParrPezzuloFriston2022]` sense is an ordered sequence of actions (token IDs) with a cumulative G score and a belief-state pointer for resumption. This must be a fresh class — minimal sketch:

```python
@dataclass
class Policy:
    token_ids: List[int]          # actions a_{T+1:T+D}
    cumulative_G: float            # sum_d G_d (with discount if used)
    leaf_belief: BeliefState       # state at the end of the policy
    parent: Optional['Policy']    # back-pointer for path reconstruction
```

The action item is to write this fresh — do not piggyback on `BeliefState` or any existing container.

### 3.2 Tree-search expansion logic

No tree-search infrastructure exists. Searched: `grep -rn "policy\|Policy\|beam\|MCTS\|tree.search\|sophisticated" transformer/vfe/` — the only matches are inside `efe.py` and `model.py` referring to the *EFE policy posterior* `q(a) ∝ exp(-γ G)`, not a tree-search policy class.

The build-out needs an expander: given a node (context + belief), pick top-K candidates (from the model's own next-token distribution or via a separate proposal), expand each as a new node, score the new node's G, push onto the frontier. Strategy choice (beam, MCTS-like, recursive depth-D for sophisticated inference) is deferred to Agent 3.

### 3.3 Belief-state caching for partial rollouts

`grep -rn "kv_cache\|KVCache\|past_key_values" transformer/` returns ZERO matches. There is no KV cache and no analogous belief-state cache. This is consistent with the gauge architecture: attention weights `beta_ij` are computed from pairwise KL divergences between transported beliefs (`vfe/attention.py:compute_kl_attention`), not from precomputable Q/K projections, so a literal KV cache is structurally inapplicable.

What IS cacheable for AIF tree search: the converged `BeliefState` at each tree node. When two sibling policies share the first D-1 tokens, their belief at depth D-1 is identical and can be reused. The cache key is the token sequence prefix; the cache value is the `BeliefState` at the prefix's last position (plus any auxiliary state the E-step needs to resume).

Implementation note: this is a *correctness-preserving* cache only if the E-step on `(context || a)` is deterministic given `(context, a)`, which holds under `@torch.no_grad()` generation as long as `epistemic_samples` MC draws use a deterministic RNG seed or are re-sampled per query (the BALD MI is a noisy estimator either way; caching it across queries on the same node is fine).

**This is fresh code.** Existing decode cache (`prior_bank.py:_decode_cache` at line 211) is a different object — it caches the all-V prior tensors, not belief states.

### 3.4 Multi-step EFE evaluation

The existing `score_candidates` (`vfe/efe.py:136`) evaluates depth-1 only. For depth D > 1, the canonical recursion per `[Friston2021SophisticatedInference]` is

`G(π_{1:D}) = G_1 + E_{q(o_1|a_1)}[G(π_{2:D} | o_1)]`

with `G_d` evaluated against the predicted next-token distribution at depth d. The recursion is over future observations as well as actions; in the LM setting the observation equals the emitted token (a == o), which collapses the policy expectation onto the action enumeration.

Fresh code. Either implement as an explicit recursion with depth-limit `D` or as an iterative search (frontier of policies, expand-and-score until horizon).

### 3.5 Preference distribution `p^*(o | C)`

No `Preference` class exists. `transformer/core/expected_free_energy.py:compute_risk` consumes a `preferences: torch.Tensor` directly (one of `(V,)` or `(K, V)`); the responsibility for constructing `p^*` is currently external.

Build-out needs a small class hierarchy:

- `EmpiricalMarginalPreference`: `p^*(v | C) = N_v / total_count` from the training corpus. Under this choice the pragmatic term reduces to expected NLL against the empirical marginal — this is the natural LM default per the user's intuition in `00_context.md` "the natural default is the empirical training-data marginal, in which case pragmatic value reduces to expected NLL."
- `LowEntropyPreference`: `p^*(v) ∝ exp(-β H_v)` parameterised by some entropy-shape function H_v. Useful for "prefer confident outputs" research-track experiments.
- `TaskConditionedPreference`: `p^*(v | C)` where C is a task / instruction embedding. RLHF-style; requires a task encoder that does not violate the no-NN constraint of CLAUDE.md (would be hard to add cleanly; defer or skip).

The class API should return a `(V,)` or `(B, V)` tensor consumable by `compute_risk(preference_mode='current_belief', preferences=...)`. The existing `compute_risk` already accepts this — fresh code is the *preference factory*, not the risk computer.

### 3.6 AIF-augmented training entry point

`transformer/vfe/train_vfe.py` is the current VFE trainer. It minimises CE + mass_phi + aux_hyperparameter loss (per `model.py:10-32` docstring and per `model.py:217-268`). Per `00_context.md`'s deferred scope decision: if Agent 3 recommends EFE-augmented training (option 2: loss = `-log q(π_observed)` over training trajectories), a thin parallel trainer is needed — call it `train_aif.py`.

This trainer reuses `VFEModel` for context encoding but adds a per-batch EFE-policy loss. It does NOT reuse `train_vfe.py` directly because the loss composition is structurally different. The CLAUDE.md hard constraint "NO CLI ARGUMENTS" applies: the new entry uses the click-to-run config dict pattern at the top of file, matching `train_vfe.py`.

## §4. Integration boundaries

### 4.1 Construction

```python
model = VFEModel(cfg)                   # existing
aif_generator = AIFGenerator(            # fresh
    model,
    preference=EmpiricalMarginalPreference.from_corpus(corpus),
    horizon=D,
    expansion_strategy='top_k_recursive',  # one of: top_k_recursive, beam, mcts
    gamma=1.0,
    epistemic_weight=0.5,
    epistemic_samples=8,
)
```

`AIFGenerator` wraps a trained (or training-time) `VFEModel`. It does not subclass `VFEModel` — composition over inheritance, so research-track experiments can swap models without touching the AIF layer.

### 4.2 Forward (context encoding)

When the AIF needs the predictive distribution at a node, it calls `self.model.forward(context_ids, targets=None)` — this returns `(B, N, V)` logits, internally exercising encode → positional → stack → norm → decode. No state to manage; reused as-is.

If the AIF needs the converged belief state (for use as the new node's `leaf_belief`), it must either call a new accessor (e.g., `model.forward_with_beliefs(context_ids)`) or, if reluctant to add the accessor, monkey-patch the final-norm output capture. Adding a clean accessor is the better path — `VFEModel.forward` already constructs the converged beliefs at `model.py:199-201`; the only change is to optionally return them. **Confirm with Agent 3 whether this accessor counts as "modification to `/vfe`" — the verdict's "files that must NOT be modified" applies to the surrogate active-inference modules, not to `model.py`.**

### 4.3 Belief update on action commit

To commit token `a` at position t+1: append `a` to context, run `self.model.forward(context || a)` — the internal stack already runs the E-step at every position including the new last. The converged belief at position `len(context)+1` is the next state `s_{t+1}`. The existing E-step natively handles context extension; no new transport code is needed — every position runs the full E-step with the new attention pattern that includes position t+1.

This satisfies the gauge-equivariance test in §5 below: position t+1 enters the E-step via `compute_kl_attention` and `compute_gauge_transport` in the standard way; sandwich product compliance is automatic because the existing fused kernel handles it.

### 4.4 Generation (the actual AIF loop)

```python
@torch.no_grad()
def aif_generate(self, prompt_ids, max_new_tokens):
    context = prompt_ids
    for _ in range(max_new_tokens):
        # Build the policy tree to depth D from `context`
        best_policy = self._tree_search(context)
        # Commit the first action of the best policy
        next_token = best_policy.token_ids[0]
        context = torch.cat([context, next_token.unsqueeze(0)], dim=1)
    return context
```

`_tree_search(context)` is the fresh tree-search routine using `Policy`, `BeliefStateCache`, depth-D recursive EFE, and the chosen expansion strategy.

### 4.5 Training (optional, deferred to Agent 3 design)

If the EFE-augmented training objective is chosen, the loss is `-log q(π = observed_sequence)` where `q(π) ∝ exp(-γ G(π))`. The training loop computes `G(π_observed)` against the dataset trajectory (treating it as the chosen policy) and against a sampled set of alternative policies to form the normalizer. Reuses `VFEModel.forward` for per-position context encoding; the loss is computed by the new trainer, not by `VFEModel`.

## §5. Gauge-equivariance audit

The new pieces of compute introduced by the AIF module are:

- Policy expansion: token append + re-encode + E-step. Re-encode delegates to `VFEPriorBank.encode` which applies the sandwich product at `prior_bank.py:321` (full-cov) or its diagonal approximation at `prior_bank.py:326`. E-step uses the fused kernel which enforces sandwich. **No new transport code.**
- Tree-search frontier: pure data-structure operations on `Policy` and `BeliefState`. **No transport.**
- Multi-step EFE recursion: invokes `compute_risk` / `compute_epistemic_value` / `decode`. All three operate on already-transported beliefs from the underlying `VFEModel`. **No transport.**
- Preference distribution: pure `(V,)` tensor over vocabulary. **Not a covariance — no sandwich needed.**

The state-update `p(s_{t+1} | s_t, a_t)` corresponds to: extend context with `a_t`, re-run `VFEModel.forward(context || a_t)`. The internal E-step at the new last position uses the new attention pattern (one more row + column in the KL-attention matrix). The transport `Omega_{t+1, j}` for j ≤ t+1 is built per the existing rules in `transformer/vfe/attention.py:compute_gauge_transport`; sandwich product is applied automatically by `_fused_attention_and_vfe_gradients_block_diag`. **No new transport hooks required by the AIF build-out.**

Open question (flag for Agent 3): if the AIF wants to cache the belief at depth d-1 and apply a *single new* token's transport rather than re-running the full E-step, that incremental update would be new transport code and would require a fresh sandwich-product audit. The simpler — and safer — pattern is to *not* attempt incremental belief updates and to instead re-run the full E-step at each node. The cost is borne by the depth/branching factor, not by adding transport code paths.

## §6. Test coverage gaps

### 6.1 Existing tests touching active inference / EFE

```
grep -rn "active_inference|efe|VFEActiveInference|VFEExpectedFreeEnergy" tests/
```

Results:
- `tests/transformer/test_vfe_package.py:266-280` — `TestVFEActiveInference.test_callback_produces_gradients`: asserts only that `grad_mu` and `grad_sigma` come back with the expected shape. Per `04_verdict.md` action item 4, this is "shape only and is insufficient under the canon §10 sign-convention pitfall."
- `tests/transformer/test_vfe_package.py:287-308` — `TestVFEExpectedFreeEnergy`: `test_score_candidates` asserts only the dict keys and shape of `scores['efe']`; `test_select_action` asserts only that the returned token ID is an int in `[0, V)`.

There is no test of: pragmatic-gradient sign, BALD MI agreement with finite difference, EFE ranking correctness (low-preference tokens get higher G), end-to-end EFE-vs-non-EFE generation quality.

### 6.2 New tests required by the build-out

**Sign of pragmatic gradient.** Construct a small `(V=10, K=4)` setup. Pick a preference `p^*` peaked at one token; pick a context whose decode predicts that token with high probability. The pragmatic term `risk = -sum_v q(v|a) log p^*(v)` should be LOWER for actions whose post-rollout `q(v|a)` matches `p^*`. Test: rank candidates by `compute_risk(...)`; verify that the candidate yielding the predictive closest to `p^*` (by KL) gets the lowest risk.

**BALD MI agreement: finite-difference vs analytic.** For small `(V, K, S=epistemic_samples)`, fix RNG, compute `compute_epistemic_value` directly. Separately, compute `MI(z; o) = H[E_z p(o|z)] - E_z[H[p(o|z)]]` by enumeration of a quantised sample grid. The two should agree within MC noise tolerance. (`04_verdict.md` action item 4 specifies the finite-difference test for the surrogate gradient; the analogous canonical test is needed for the EFE pipeline.)

**Sophisticated-inference depth-2 recursion correctness on a 2-token toy problem.** Construct a vocabulary `V = 3` with hand-designed preferences and a deterministic two-step transition where the optimal depth-2 policy is `(a_1, a_2)` with strictly lower cumulative G than every other length-2 sequence. Run depth-2 sophisticated inference; verify it selects the known-optimal policy. This is the smallest test that distinguishes depth-2 from depth-1 (depth-1 picks the locally best `a_1`, which differs from the depth-2 optimum by construction).

**End-to-end empirical falsification.** Train a small `VFEModel` to a fixed validation NLL on a small corpus. Run two generation conditions on a held-out validation prompt set: (i) `VFEModel.generate(use_efe=False)` (top-k sampling baseline), (ii) `AIFGenerator.aif_generate(...)` with the empirical-marginal preference. Compute validation NLL of each completion under a reference model (or under the same VFEModel — both work). The AIF generator should produce completions with strictly lower NLL on the validation distribution; if not, the build-out has not delivered on its empirical promise.

**Gauge-equivariance check for the AIF module.** Apply a gauge transform `g ∈ GL(K)` to all beliefs at the start of a tree search; run the search; verify that the chosen action sequence is invariant. This is the AIF-level analogue of the existing pure_vfe gauge-transport tests; it inherits sandwich-product correctness from `VFEPriorBank.encode` and `VFEEStep.forward`, so failure indicates a new transport bug in the AIF wrapper itself.

**E-step blindness regression test for AIFGenerator.** Construct an `AIFGenerator` and inspect (via reflection or by inserting a debug assertion) that no path from the targets argument (if any new training path) reaches `VFEEStep.forward`. The existing `VFEEStep.forward` signature has no `targets` parameter (`e_step.py:725-731`), so any leak would have to introduce a side channel — the test guards against that.

## Out-of-scope observations

- The `transformer/vfe/efe.py.tmp.*` files (search results above) suggest a previous editor session left temp files. They should be cleaned up before the AIF build-out begins, otherwise tooling will surface stale code. Not a blocker for this audit.
- `VFEStack`'s default mean-only cascade (`stack.py:19-34`) is a documented research-track gap from canonical hierarchical VI. The AIF build-out does not change this, but downstream calibration of multi-layer AIF results should account for the fact that cross-layer uncertainty is being dropped at every boundary. Flag for Agent 3 if it materially affects design.
