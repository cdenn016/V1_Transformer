# Verify A — EM-autograd (memo 02) + BR-1a routing hazard (memo 07) + equivalence-gate baseline

VERIFIER: independent adversarial confirmation, not restatement. Every claim
below is grounded in a real forward+CE-backward on a CPU model built from the
tiny LIVE-patterned config, or in a direct read of `optimizer.py`. Numbers are
mine (seed 0, deterministic inputs), NOT memo 02's — see the magnitude note.

Date: 2026-06-01. Env: torch 2.11 CPU-only, device forced to CPU.
Repo root `C:/Users/chris and christine/Desktop/VFE_1.0`.

Durable artifacts produced:
- `docs/audit_workspace/equivalence_harness.py` — the re-runnable gate (capture + `--gate`).
- `docs/audit_workspace/baseline_skip_attention.json` — loss + full grad-norm table + resolved flags.
- `docs/audit_workspace/baseline_skip_attention.weights.pt` — pinned weights for the gate.

## VERDICT — all three tasks CONFIRMED

| Task | Claim | Status |
|------|-------|--------|
| 1 | `mu_embed`, `log_sigma_diag`, `phi_embed` all get nonzero grad under `ift_phi`+`skip_attention`; no silent-freeze gap | CONFIRMED |
| 1 | `raw_sigma_lr` grad is None / `norm2.{w,b}` grad is None are decode+skip artifacts, NOT EM-boundary defects | CONFIRMED (by control) |
| 2 | `constant_omega` registered only on attention, FFN borrows same object via `__dict__` (invisible to `named_parameters`); BR-1a silent-LR-move | CONFIRMED + escalated (param is LIVE) |
| 3 | behavior-preserving gate baseline captured + byte-reproducible | CONFIRMED (gate PASS twice) |

The resolved path was hard-asserted before capture (`assert_live_path`), so the
baseline IS the live path, not a silent dataclass-default fallback. Resolved
flags actually wired: `block.skip_attention=True`, `ffn.em_phi_mode='amortized'`,
`amortized_inference=True`, `amortize_sigma=True`, `exact_phi_grad=True`,
`gauge_mode='learned'`, `gauge_param='phi'`, `diagonal_covariance=True`,
`n_iterations=1`, `use_prior_bank=False`.

## TASK 1 — EM-autograd (memo 02 confirmed)

Real forward (B=2, N=16, deterministic ids) + `F.cross_entropy` backward. The
three embedding parameters are ALL alive:

| param | grad norm (mine, seed 0) | alive |
|-------|--------------------------|-------|
| `token_embed.mu_embed.weight`  | 8.3305e-03 | yes |
| `token_embed.log_sigma_diag`   | 2.2818e-03 | yes |
| `token_embed.phi_embed.weight` | 1.1201e-02 | yes |

Central claim HOLDS: `ift_phi` + `skip_attention` has NO silent-freeze gap. The
detaching-mode pathology (`em_phi_p`/`em_phi_q` freeze σ_embed/φ_embed when the
attention sublayer was their only autograd path) does NOT recur here — the FFN
E-step is itself the autograd path (`em_phi_mode='amortized'` ⇒ no detach at the
EM boundary; priors stay attached).

Magnitude note (verifier honesty): my norms are ~100–176× SMALLER than memo 02's
(mu 0.0083 vs 1.4678; log_sigma 0.0023 vs 0.2363; phi 0.0112 vs 1.3064). This is
expected and is NOT a reproduction failure — different seed, and I use
deterministic `arange`-based inputs with mean-reduction CE rather than memo 02's
random inputs. TASK 1's requirement is qualitative aliveness (all three nonzero),
which is robustly confirmed and independently anchored by the asserted resolved
flags. I do NOT claim to reproduce memo 02's magnitudes; MY numbers are the
authoritative baseline going forward. Sanity check on the forward itself:
loss = 4.84829 ≈ ln(128) = 4.852, i.e. near-uniform logits at init — the
forward/decode graph is behaving correctly.

### Dead-weight findings — confirmed as artifacts by discriminating controls

`raw_sigma_lr.grad is None` and `norm2.{weight,bias}.grad is None` alone are
consistent with BOTH "benign artifact" AND "EM-boundary defect". Only the
controls separate them:

- CONTROL (raw_sigma_lr): `use_prior_bank=False` → `raw_sigma_lr.grad = None`;
  flip `use_prior_bank=True` (KL decode consumes σ_q) → grad flips to
  **1.339e-09** (nonzero). Proves the cause is "σ_q output cannot reach the
  mu-only linear decode", a DECODE artifact, not a stray detach at the EM
  boundary. Also scoped by `n_layers=1` (no later layer to consume σ_q).
- CONTROL (norm2): `norm1.{weight,bias}` are ALIVE (0.01636 / 0.01587) while
  `norm2.{weight,bias}` are None. Proves `norm2` is simply never invoked under
  `skip_attention=True` (the single VFE sublayer uses `norm1`), a SKIP artifact,
  not a broken norm. If `norm1` were also dead the story would collapse — it is
  not.

Both are "allocated-but-silently-ignored" dead weight in the state_dict, not
correctness bugs. Memo 02's EM-1 and EM-2 confirmed.

## TASK 2 — BR-1a silent-LR hazard (memo 07 confirmed + escalated)

Built `gauge_mode='constant'` (NOT live; the dangerous case), tiny config
otherwise identical.

(i) Ownership — CONFIRMED. `named_parameters()` contains ONLY
`transformer.blocks.0.attention.constant_omega.0` and `.1` (10×10 each,
`nn.ParameterList` len 2). `ffn.named_parameters()` contains NO `constant_omega`
key (invisible). `ffn.__dict__['constant_omega'] is attention.constant_omega`
is **True** — the FFN borrows the SAME object via `__dict__` to bypass
nn.Module registration (`variational_ffn.py:435`; created at
`attention.py:1623-1633`).

(ii) Optimizer routing — the exact rule:line (from a direct read of
`transformer/training/optimizer.py::create_param_groups`, the single first-match
`for name, param in model.named_parameters()` loop):

- `transformer.blocks.0.attention.constant_omega.0/.1`
  → first-matches `'attention' in name` at **optimizer.py:750**
  → `'attention'` group @ `config.M_attention_lr` = **0.013**.
  The `'omega_embed' in name` test at **optimizer.py:736** does NOT match —
  the substring `omega_embed` is absent from `constant_omega` (substring check
  fails). Verified two ways: the REAL `create_param_groups` call routes both to
  group `'attention'` lr=0.013, and a first-match replay agrees.
- Hypothetical `transformer.blocks.0.ffn.constant_omega.0` (the post-refactor
  re-home target) matches NONE of `omega_embed`/`phi_embed`/`log_kappa`/
  `attention`/`attn`/`out_proj`/`lm_head`/`norm`/`bias`/`raw_`/`gate`/
  `log_scale` → falls through to the final `else` at **optimizer.py:759**
  → `'ffn'` group @ `config.M_vfe_hyperparam_lr` = **0.095**.

So re-homing `constant_omega` from `attention.*` to `ffn.*` silently moves its
learning rate **0.013 → 0.095** (≈7.3×) with no error and no warning. BR-1a
holds exactly as memo 07 states (weight_decay stays `non_embed_wd` either way).

ESCALATION (load-bearing check the memo did not run): under `skip_attention=True`
+ `gauge_mode='constant'`, a real CE backward gives
`constant_omega.0.grad_norm = 7.754e-03` and `constant_omega.1.grad_norm =
5.215e-03` — both NONZERO, reached purely through the FFN E-step transport
(the attention sublayer is skipped). The param is therefore genuinely TRAINED in
this config, so the silent LR move is a REAL training-behavior change, not a
harmless routing curiosity. If `constant_omega` had come back dead, the LR move
would not matter; it does not, so BR-1a must be fixed deliberately (explicit
routing rule or rename) before any re-home.

## TASK 3 — equivalence gate baseline (captured + reproducible)

`baseline_skip_attention.json`: `loss = 4.848293781280518`, `param_count = 14`,
full `grad_norms{name→float|null}` table, and `config_used` (seed, device,
tiny_config, resolved_flags, weights_file, tolerances). `grad is None` is
recorded as JSON `null` (distinct from a real 0.0), so the gate distinguishes
dead from zero-grad.

`equivalence_harness.py` re-runs verbatim: `python equivalence_harness.py`
captures; `python equivalence_harness.py --gate` rebuilds, LOADS the pinned
weights (`strict=False`, reports key migrations), recomputes, and asserts
loss + every grad-norm match to `atol=1e-6, rtol=1e-5`. Ran the gate twice →
**PASS** both times (byte-reproducible).

Two design choices that make this a TRUSTWORTHY gate (not an init-order trap):
1. WEIGHTS ARE PINNED. Removing the `IrrepMultiHeadAttention` instantiation
   shifts the global-RNG init order, so a seed-0-init-only gate would
   spuriously FAIL a behavior-preserving refactor. Loading pinned weights makes
   the gate test "same math given same weights", isolating it from init order.
2. INPUTS are construction-RNG-independent (`arange`-based), so an init-order
   change cannot shift the inputs either.

### SCOPE BOUNDARY OF THIS GATE (must read before trusting it for the refactor)

This baseline is captured under `gauge_mode='learned'` + `skip_attention=True`.
In that path the attention sublayer is NEVER called — `blocks.py:795`
(`if not self.skip_attention:`) guards the entire dispatch, and `constant_omega`
is `None`. Therefore deleting the attention module **cannot change this gate's
loss or grads**: the gate will PASS trivially for the attention-removal refactor
whether or not the `constant_omega` re-home botches the LR. The gate's real
protective value is over the SHARED embed → FFN E-step → transport → decode graph
under learned gauge.

It does NOT exercise constant-gauge transport — which is exactly where BR-1a
bites. RECOMMENDATION: capture a COMPANION baseline under `gauge_mode='constant'`
(harness is trivially extended — set `cfg['gauge_mode']='constant'`) and assert
`constant_omega.{0,1}.grad` stay nonzero with the intended LR after the re-home.
That companion baseline, not this one, is the true BR-1a gate. Do not present
"learned-gauge gate PASS" as assurance that the constant_omega re-home preserved
training behavior.

## Cross-check against memos

- Memo 02 (EM-1, EM-2, central no-gap claim): CONFIRMED, with the magnitude
  caveat above.
- Memo 07 (BR-1a routing, `__dict__` borrow, sole-registered-owner): CONFIRMED;
  escalated with the live-grad demonstration.
- One nit: memo 07's prose cites the catch-all `'attention'` rule at
  `optimizer.py:750` and the `else` at `:759`; both line numbers match the code
  I read. (Memo 07 §BR-1a also writes `optimizer.py:711-760` / `:736` / `:747`
  for the loop, `omega_embed`, and `log_kappa` rules — all consistent with my
  read.)
