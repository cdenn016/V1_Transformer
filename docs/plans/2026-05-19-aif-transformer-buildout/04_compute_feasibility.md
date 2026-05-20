# Compute and Memory Feasibility — Canonical AIF Transformer Build-Out

**Date:** 2026-05-19
**Scope:** Cost and memory analysis for the canonical active-inference transformer where policies map to future token sequences over horizon D, evaluated via Expected Free Energy G(π) per [ParrPezzuloFriston2022 Ch. 7].

All estimates assume the active config from `transformer/vfe/train_vfe.py` unless stated otherwise:

- V = 50257 (vocabulary, WikiText tokenizer)
- K = 20 (embed_dim; `irrep_spec=[('fund', 2, 10)]` gives H=2 heads of d_head=10)
- n_gen = 2 × 10² = 200 (GLK generators for two 10×10 blocks)
- N = 128 (max_seq_len)
- B = 64 (batch_size)
- L = 1 (n_layers)
- T_E = 1 (n_e_steps)
- S = 4 (epistemic_samples)
- diagonal_covariance = True
- Data type: float32 throughout (4 bytes per element)

---

## §1. Cost of a Single Forward Pass

### 1.1 Per-Iteration E-Step Cost

Each E-step iteration (the body of `transformer/vfe/e_step.py:833` loop) executes the following dominant operations, traced through the fused active-config path.

**compute_gauge_transport** (`transformer/vfe/e_step.py:840-843`, `transformer/vfe/attention.py`). Computes block-diagonal matrix exponentials for φ of shape (B, N, n_gen). With irrep_dims=[10,10] and n_gen=200, per-block cost is dominated by expm of a 10×10 matrix per token. Using scaling-and-squaring, expm of a d×d matrix costs O(d³) FLOPs. Total: B × N × H × d_head³ = 64 × 128 × 2 × 10³ ≈ 164 MFLOPs. This is a minor term compared to the attention grid.

**KL attention grid** (`transformer/vfe/e_step.py:1006-1028`, `_fused_attention_and_vfe_gradients_block_diag`). Computes the (B, N, N) KL matrix and β = softmax(-KL/τ). For diagonal Gaussian KL between transported beliefs, cost scales as O(B × N² × K). At B=64, N=128, K=20: 64 × 128² × 20 × 5 ops-per-pair ≈ 1.05 GFLOPs (factor ~5 per pair accounts for subtraction, division, log, sum). Under the per-head dispatch (H=2 heads of d_head=10), this runs as two independent blocks, each at d_head=10 rather than K=20, but the loop doubles the calls — net cost is unchanged.

**VFE gradient computation** (`transformer/vfe/e_step.py:1054-1078`). The natural gradient requires forming the β-weighted gradient of KL with respect to μ and σ. Dominant term: O(B × N² × K) again to form the β-weighted aggregation. ≈ 1 GFLOPs, same order as the KL grid.

**Natural gradient projection** (`transformer/vfe/e_step.py:1173-1175`, `compute_natural_gradient_gpu`). Divides grad_mu by σ (Fisher metric preconditioning for diagonal covariance). O(B × N × K) = negligible.

**Retractions** (μ retraction at `e_step.py:1370`, σ retraction at `e_step.py:1373-1383`, φ update at `e_step.py:1400`). All O(B × N × K) or O(B × N × n_gen). φ update requires a second expm pass (same cost as compute_gauge_transport above). Negligible relative to the attention grid.

Per-iteration E-step cost summary (at B=64, N=128, K=20, fp32): approximately 2–3 GFLOPs dominated by the O(B × N² × K) attention-KL computation. This is roughly 100× cheaper than the decode step derived below.

### 1.2 Full Forward Pass Cost

One model forward = encode + (L × T_E × E-step) + decode.

**Encode** (`transformer/vfe/prior_bank.py:316-425`). Table lookup of (μ, σ, φ) from embedding tables: O(B × N × K + B × N × n_gen) = O(B × N × 220) ≈ negligible (pure memory read, no FLOPs).

**L × T_E E-step layers**: With L=1, T_E=1, cost = one E-step iteration ≈ 2–3 GFLOPs (above).

**Decode** (`transformer/vfe/prior_bank.py:427-508`). The dominant term and the feasibility bottleneck. The fused implementation at `prior_bank.py:498-500` computes:

```
lhs = (B, N, 2K)    rhs = (V, 2K)
combined = lhs @ rhs.T    # (B, N, V)
```

FLOP count: 2 × B × N × V × 2K = 4 × B × N × V × K. At B=64, N=128, V=50257, K=20:
4 × 64 × 128 × 50257 × 20 ≈ **33 GFLOPs**.

This single matmul is roughly 10–15× the cost of the entire E-step. The output tensor `(B, N, V)` = 64 × 128 × 50257 × 4 bytes ≈ **1.65 GB** for the logits alone. This tensor must be materialized in GPU memory during every forward pass that reads the full logits. The decode dominates both FLOP and memory budgets — all subsequent sections take this as the baseline cost.

**Total forward pass** at training dimensions (B=64, N=128, K=20): approximately 35–40 GFLOPs and 1.65 GB peak activation memory from logits. Under a40/A100 at ~300 TFLOPs fp32 effective, one forward pass takes roughly **0.1–0.15 ms** in pure FLOP terms, but memory bandwidth typically limits the realized throughput to 1–5 ms per forward at these dimensions (the matmul is memory-bandwidth-bound at V=50257, K=20 because the operand is thin and wide).

At generation shape (B=1, N=128), the decode costs 2 × 1 × 128 × 50257 × 2×20 ≈ 514 MFLOPs and produces a `(1, 128, V)` output tensor of **26 MB**. The generation-time forward is ~64× cheaper in FLOPs than the training-time forward. All tree-search and beam analysis in §3 assumes B=1.

---

## §2. Cost of a Depth-1 EFE Candidate Score

The existing implementation in `transformer/vfe/efe.py:VFEExpectedFreeEnergy.score_candidates` scores C candidate next tokens sequentially (`efe.py:164` loop over C). For each candidate:

1. **Model forward** (`efe.py:174`): appends candidate to context and calls `self.model(trial_ids)`. At B=1, N=129, this is one full forward pass including decode ≈ 514 MFLOPs + 26 MB logits.

2. **Second encode pass** (`efe.py:191`): calls `self.model.prior_bank.encode(trial_ids)` again on the same token sequence for BALD MI. This is a redundant embedding lookup — the belief state from the first forward is not returned by `VFEModel.forward` and cannot be reused without interface modification. Cost is O(B×N×K) memory reads: negligible in FLOPs, but a correctness-relevant design gap since the two passes may see different positional encoding state.

3. **BALD epistemic value** (`efe.py:73-134`, `_compute_epistemic_value`): S = 4 decode passes at B=1, N=1 (last position only). Each decode: 2 × 1 × 1 × V × 2K ≈ 4 MFLOPs. Total: 4 × 4 = 16 MFLOPs. Negligible.

Cost per candidate: approximately **1 forward pass** ≈ 514 MFLOPs at generation shape. The BALD and risk terms are both negligible relative to the full-sequence forward.

The `select_action` method (`efe.py:215-252`) pre-filters to top_k=50 candidates via one initial forward (`efe.py:237-241`) before scoring. The bottleneck is therefore **50 sequential forward passes** per generated token. On an A100, where a single B=1 forward might take 2–10 ms (memory-bandwidth bound), scoring 50 candidates costs **0.1–0.5 s per token**. This is acceptable for offline or research generation but not production streaming.

---

## §3. Cost of a Depth-D Sophisticated-Inference Rollout

### 3.1 Naive Exhaustive Tree

The full tree of all possible depth-D futures has V^D leaves. For V=50257 and D=2, this is 2.5 × 10⁹ leaf nodes. Even at 1 byte per node for bookkeeping, the tree requires 2.5 GB of memory before any computation. At D=3, V³ ≈ 1.27 × 10¹⁴ nodes. Exhaustive tree search is structurally intractable for D ≥ 2 at natural language vocabulary sizes. No further analysis is needed on this path; a filtering step at each level is mandatory.

### 3.2 Beam Tree

Beam search keeps the b highest-scoring candidates at each expansion. The tree has at most b nodes per depth level (retained) and b² forward passes evaluated per level (each retained node expands to b children, then top-b are kept). Total forward passes per generated output token: approximately b² × D.

At each depth d, the context length grows to N + d tokens. The forward pass cost at depth d ≈ (N + d) / N times the depth-0 cost (the (B,N) decode matmul scales linearly with N). For D ≤ 5, this overhead is at most 5/128 ≈ 4%, negligible.

Forward passes per generated token: b² × D. At B=1, each forward ≈ 514 MFLOPs (±50%).

| D | b  | Forwards/token | Approx FLOPs | Est. A100 time (±2×) |
|---|-----|----------------|--------------|----------------------|
| 1 | 50  | 50             | 25.7 GFLOPs  | 0.1–0.5 s            |
| 2 | 8   | 128            | 65.8 GFLOPs  | 0.5–2 s              |
| 2 | 16  | 512            | 263 GFLOPs   | 1–5 s                |
| 3 | 8   | 192            | 98.7 GFLOPs  | 0.5–3 s              |
| 3 | 16  | 768            | 395 GFLOPs   | 2–10 s               |
| 4 | 8   | 512            | 263 GFLOPs   | 1–5 s                |
| 4 | 16  | 4096           | 2.1 TFLOPs   | 7–30 s               |
| 5 | 32  | 25600          | 13.2 TFLOPs  | 45–180 s             |

The A100 time estimates assume 100–500 GFLOPs/s effective throughput at generation shape (memory-bandwidth bound at K=20, V=50257). The 5× range per row reflects variability in kernel dispatch overhead for small matmuls.

Practical feasibility frontier on a single 24 GB GPU:

- **(D=2, b≤8) and (D=3, b≤8)**: feasible for interactive generation at several seconds per token. Suitable as a research demo default.
- **(D=2, b=16) or (D=3, b=16)**: feasible for offline generation, ~5–10 s/token.
- **(D=4, b≥16) or (D=5, b≥8)**: demo-prohibitive (minutes per token). Research-only mode.
- **(D≥4, b≥32)**: hours per token at this vocabulary size; requires distributed inference or radical approximation.

The full-covariance Σ path (diagonal_covariance=False) multiplies K by K in the attention grid: (B, N², K²) = 128× the diagonal cost at K=20. Full-covariance tree search is intractable beyond D=1.

---

## §4. Belief-State Caching Strategy

The (μ, σ, φ) belief tuple replaces the standard KV-cache in this architecture. The cache must store the converged belief state after the E-step at each position, per tree node.

Memory per snapshot (diagonal Σ, fp32, B=1, sequence length N+d, K=20, n_gen=200):

- μ: (1, N+d, K) × 4 bytes = (N+d) × 80 bytes
- σ: (1, N+d, K) × 4 bytes = (N+d) × 80 bytes
- φ: (1, N+d, n_gen) × 4 bytes = (N+d) × 800 bytes

At d=0, N=128: φ dominates at 102.4 KB; μ and σ contribute 10.24 KB each. Total per node: **122.9 KB**.

φ is the dominant cache component because n_gen = K² / H = 200 at the active config, a factor of 10× larger than K. This ratio grows as d_head² per block, making n_gen scaling the main constraint for larger gauge groups.

Tree cache sizes at D=3, b=8, N=128 (full belief snapshot per node, not sharing prefix states):

- Number of tree nodes: b^D = 512 at the leaves; total nodes in the tree = b⁰ + b¹ + b² + b³ = 1 + 8 + 64 + 512 = 585.
- Memory per node: 122.9 KB (at depth 0) up to ~127.4 KB (at depth 3, N+d=131). Use 125 KB average.
- Total cache: 585 × 125 KB ≈ **72 MB**. This fits comfortably in 24 GB GPU memory.

At B=1 tree search the belief cache is negligible compared to the decode logits tensor. At B=64 training shape, a single forward's decode logit tensor (1.65 GB) dwarfs the entire belief cache of the tree.

**Prefix state sharing.** Nodes at depth d share the belief state of their common ancestor prefix through depth d−1. This sharing reduces total cache by (b^D − b^(D-1) × ... ) / b^D in the leaves, roughly a (D−1)/D fraction sharing for a balanced beam tree. In practice, the first encode (depth 0, the original context) is always shared: one copy of the (1, N, K) belief state serves all b^D leaves. Only the incremental states at each depth need separate storage per node, reducing total unique cache footprint by roughly b/(b−1) × (D−1) factor. At (D=3, b=8) this cuts the effective cache from 72 MB to approximately 20–30 MB.

The shared decode prior cache — `VFEPriorBank._decode_cache` materialized as (V, K) = 50257 × 20 × 4 bytes ≈ 4 MB — is a singleton that should be computed once and shared across all tree nodes. Under the current implementation (`prior_bank.py:464-471`), the cache is invalidated per forward via `model.py:invalidate_cache()`. The build-out must preserve this cache across tree-search siblings; each tree-node forward should NOT call `invalidate_cache()` unless the beam has exhausted that token's generation step.

Detailed memory budget for D=3, b=8, B=1, K=20, N=128:

| Component | Per-item size | Count | Total |
|---|---|---|---|
| Belief cache (μ+σ+φ) | 122.9 KB | 585 nodes | 72 MB |
| Decode prior (V,K) shared | 4 MB | 1 | 4 MB |
| Active forward logits (1,N,V) | 26 MB | 1 (reused) | 26 MB |
| BALD MC sample (S=4, single position) | ~0.8 MB | 1 | 0.8 MB |
| **Total** | | | **~103 MB** |

This is well within 24 GB GPU memory even with model parameters (embed_dim=20, V=50257: parameter budget ≈ 50 MB for phi_embed alone at 200 floats per token).

---

## §5. BALD MI Estimator Variance vs S

The BALD estimator is I(z; o | q) = H[p̄] − (1/S) Σ_s H[p_s], where p̄ = (1/S) Σ_s p_s and p_s = softmax(decode(μ + √σ · ε_s)). This estimator is unbiased (its expectation over ε is the true MI) and has Monte Carlo standard error O(1/√S).

At S=4, the relative standard error is 1/√4 = 50%. This is the standard error on the MI estimate as a scalar signal, not the standard error on the policy selection itself. The policy is selected by softmax(-γG(π)), which is a noisy ranking. The 50% MI variance maps to score noise of approximately epistemic_weight × 0.5 × E[I] per candidate. In the active config, epistemic_weight = 0.5 (`train_vfe.py:108`), so the epistemic contribution to total G is scaled by 0.5, reducing the effective MI variance in the EFE score to roughly 25% of the MI value.

For pure generation-time scoring without gradient, S=4 is acceptable when the epistemic term plays a secondary role (as configured: pragmatic_weight=1.0, epistemic_weight=0.5). The score noise from S=4 BALD is unlikely to change which candidate is selected when candidates differ substantially in risk; it may affect close-call decisions. S=8 would halve the MI variance (to ~12% of MI), and S=16 would reduce it to ~6%.

For training-time gradient signals through the BALD term, the two-pass streaming implementation (`transformer/core/active_inference.py:270-345`) is designed to avoid materializing S×(B,N,V) logits simultaneously. The streaming p̄ accumulation (`active_inference.py:277-296`) passes one sample at a time and accumulates via online mean, keeping peak memory at one (B,N,V) tensor ≈ 26 MB at generation shape. This memory pattern is the correct design for all depth-D recursion.

Recommended S budget for the build-out. For the first demo, S=4 matches the active config and keeps per-candidate BALD cost at 4 decode calls of (1,1,V) ≈ 16 MFLOPs. For higher-quality epistemic scoring in depth-D tree search, S=8 doubles this cost but is still negligible relative to the full-sequence forward. For training-time MI gradients, S=4 is the practical minimum; S=8 is preferred.

Total sample count at generation time: S × b × D per generated token. At S=4, b=8, D=3: 96 BALD samples per token. Each BALD sample involves one (1,1,V) decode ≈ 4 MFLOPs. Total BALD cost per generated token at this setting: 96 × 4 ≈ 384 MFLOPs. This is less than one full-sequence forward (514 MFLOPs) and is not the bottleneck.

---

## §6. Practical Default Configuration

The recommended AIFConfig for a click-to-run demo that exercises the canonical machinery and completes in minutes:

```python
aif_config = {
    # Horizon and search
    'horizon_depth':       2,       # D=2: one step ahead of depth-1 EFE
    'beam_width':          4,       # b=4: 16 forward passes per generation step
    'top_k_filter':        50,      # pre-filter V->50 before expansion (mandatory)
    
    # BALD
    'epistemic_samples':   4,       # S=4: matches active training config
    
    # Policy weighting
    'pragmatic_weight':    1.0,     # as in active config
    'epistemic_weight':    0.5,     # as in active config
    'gamma':               1.0,     # Gibbs inverse temperature for q(pi)
    
    # Covariance
    'diagonal_covariance': True,    # required for depth > 1 (hard constraint)
    
    # Generation
    'batch_size':          1,       # B=1 at generation time
    'max_seq_len':         128,     # as in active config
}
```

Expected per-token generation cost at D=2, b=4 on a single A100 (40 GB) or 24 GB consumer GPU:

- Forward passes per token: b² × D = 4² × 2 = 32 (each of b retained nodes expands to b children). Plus one initial forward for top-k filtering = 33 total.
- FLOP budget: 33 × 514 MFLOPs ≈ 17 GFLOPs.
- At 100–500 GFLOPs/s effective throughput (memory-bandwidth bound): **0.03–0.17 s per token**, plus Python/PyTorch dispatch overhead.
- With Python-level sequential loop overhead (efe.py:164 pattern): estimate **0.5–3 s per token** for 33 sequential forwards, dominated by Python-CUDA dispatch latency. Batching the candidates into a single forward of shape (b, N) would reduce this to 0.05–0.5 s per token at the cost of interface changes.

Wall-clock estimate for the current sequential implementation: approximately 1–5 s per token on a 24 GB consumer GPU (RTX 3090 / 4090 class), and 0.3–2 s per token on an A100. These estimates carry ±2× uncertainty due to variability in kernel dispatch latency for the thin-and-wide matmul (K=20 is small enough that the V×2K matmul may not fully saturate tensor cores).

---

## §7. Failure Modes and OOM Risks

The dominant failure modes follow directly from §1 and §3.

**Training-shape tree search.** The decode logit tensor scales as B × N × V × 4 bytes. At training shape B=64, one forward produces 1.65 GB of logits. Running tree search at B=64 (treating each training example as a separate generation context) would require 1.65 GB × b² forwards simultaneously, plus the E-step activation graph. At b=4, D=2 with gradient tracking: 16 forwards × 1.65 GB ≈ 26 GB logit memory alone, not counting gradients (which triple this via forward-backward graph). This causes OOM on any single GPU. Mitigation: AIF generation must run at B=1; training-time EFE objectives (§9) must either detach the logit tensor or accumulate gradients one sample at a time.

**Full-covariance Σ at depth > 1.** For diagonal_covariance=False, σ has shape (B, N, K, K) = (1, 128, 20, 20) per node = 409.6 KB per node. More critically, the attention grid cost scales as O(B × N² × K²) = 8 GFLOPs per iteration instead of 200 MFLOPs. At tree-search depth with b=8, D=3 this multiplies to ~1.5 TFLOPs per generated token for the E-step alone, separate from decode. The build-out must enforce diagonal_covariance=True at depth > 1 via a runtime guard in the AIF entry point. This is already the documented constraint in CLAUDE.md.

**Non-streaming BALD in recursive rollout.** A naive recursive implementation that stores S samples of (B, N+d, V) logits per tree node would accumulate S × b^D × 26 MB ≈ 4 GB at (S=4, D=3, b=8). The existing streaming pattern at `active_inference.py:277-296` (streaming p̄ without materialization of the S-stack) resolves this for depth-1 BALD. Any depth-D recursion must carry this streaming pattern through each level; a naive implementation that collects all S sample logits before computing H[p̄] reintroduces this allocation.

**Per-node decode cache materialization.** If `VFEPriorBank._decode_cache` is invalidated and recomputed at every tree node (as the current model.py forward does), the build-out pays V × K × 4 bytes = 4 MB to recompute per-node. Over 585 nodes at D=3, b=8, this adds 2.3 GB of redundant compute and memory pressure. Mitigation: compute the (V, K) prior once at the start of the generation step and pass it as a frozen context to all tree nodes, bypassing `invalidate_cache()`.

**Large beam width with gradient tracking.** If an EFE-augmented training objective requires gradients through the tree search, the activation graph grows as O(b^D) forwards × gradient overhead. At b=8, D=2 already materializes 64 full forward graphs simultaneously. Mitigation: gradient-free tree search (all tree evaluation under `torch.no_grad()`), with the policy gradient computed via REINFORCE or the surrogate `-log q(π_observed)` on the observed path only.

**Horizon D=4 or 5 on any hardware.** At (D=4, b=16), the forward pass budget per token is 4096 forwards × 514 MFLOPs = 2.1 TFLOPs. Even on an A100 with 80 TFLOPS fp32 dense, this requires ~26 ms just in FLOPs. With sequential dispatch and Python overhead, hours-scale wall time is likely. Mitigation: hard cap D ≤ 3 in the demo config; expose D ≥ 4 only under a research-mode flag with explicit warnings.

---

## §8. Comparison to Alternatives

**Depth-D beam vs depth-1 EFE.** The existing `vfe/efe.py` depth-1 EFE already provides risk + ambiguity + epistemic scoring at the cost of 50 forward passes per token (after top-k filtering). Depth-2 beam at b=4 adds 32 forwards on top of the depth-1 initial scan, totaling ~82 forwards per token but evaluating two-step consequences. The quality difference depends on whether the target task requires planning ahead: for fluent language modeling where the next-best token is usually unambiguous, depth-1 is sufficient and depth-2 adds noise from the MI estimator variance. For tasks requiring deliberate multi-step planning (code generation, math), depth-2 and beyond should help, but empirical evidence for language models is sparse [ParrPezzuloFriston2022 Ch. 7 discusses conditions under which deeper rollout adds value for discrete AIF agents]. The cost overhead for depth-2 is 1.6× vs depth-1 EFE, which is modest. Depth-3 is 3.8× more expensive than depth-1 — a reasonable research trade-off for controlled experiments.

**Beam vs top-k vs nucleus sampling at expansion.** Beam search with b=4–8 is the most principled expansion strategy for EFE: it maintains the b best prefix beliefs and evaluates their EFE-weighted posterior. Top-k with k=50 at a single depth (the current efe.py approach) is equivalent to depth-1 beam search. Nucleus sampling at expansion would reduce the candidate count but introduces non-determinism in the tree structure. For a canonical AIF implementation, beam search is preferred because the Gibbs policy posterior `q(π) ∝ exp(-γG(π))` is exactly the soft-max over the beam's EFE scores — the connection to [ParrPezzuloFriston2022 eq. 7.6] is exact for beam search and only approximate for nucleus sampling.

**Sophisticated inference vs flat leaf scoring.** Flat leaf scoring evaluates G only at the leaf nodes (depth D) and ignores intermediate nodes. Sophisticated inference [Friston2021SophisticatedInference] evaluates G recursively at every node: G(π) = G(a_t) + G(π_{t+1:T} | a_t chosen). This requires one EFE evaluation per tree node (all b^D nodes) rather than just the b^D leaves, roughly doubling the compute cost but producing a proper recursive policy that accounts for epistemic value at each step. For D=2 the difference is 16 vs 8 EFE evaluations (2×); for D=3 it is 72 vs 64 (1.1×). The marginal cost of proper sophisticated inference over flat leaf scoring decreases with D because the tree grows geometrically and the interior nodes become a smaller fraction. Recommend implementing recursive EFE evaluation as the default (canonical) and flat-leaf as a cost-reduction approximation.

---

## §9. Training-Time Cost

Two distinct training-time regimes exist.

**Regime A: EFE as policy prior, no tree expansion.** The observed training trajectory π_obs = (a_1, ..., a_T) is treated as the chosen policy. The training loss adds `-log q(π_obs) = γ · G(π_obs)` to the standard cross-entropy, where G is evaluated along the observed tokens only. This requires one additional EFE evaluation per training step: one forward pass on the observed sequence (already computed for CE) plus BALD scoring at the final position. Total added cost: approximately the BALD pass only ≈ S × 4 MFLOPs per position ≈ 16 MFLOPs per training step, a less-than-1% overhead on the 35 GFLOPs training forward. The E-step gradient still does not see targets (CLAUDE.md constraint is preserved since EFE scoring is separated from the E-step loop). This regime is feasible and recommended as the default training-time option.

**Regime B: EFE-augmented training with tree expansion.** Each training step runs a beam tree of b^D forward passes per example, computing the policy posterior gradient for the observed sequence. At b=8, D=3, B=64: 512 forwards per training step, each requiring 35 GFLOPs at training shape. Total per-step cost: 512 × 35 GFLOPs ≈ **18 TFLOPs**. On an A100 at ~80 TFLOPs effective training throughput (with gradients), this is approximately **225 ms per training step** versus approximately 0.5–1 ms for standard CE training. The overhead multiplier is of order 200–500×. Training on 15,000 steps (the active config max_steps) would require ~940 hours of A100 compute. This regime is infeasible for routine training and should be exposed only as an explicitly off-by-default research toggle with the wall-clock estimate printed as a warning before any run.

For the build-out, Regime A is the recommended training-time default. Regime B should exist in the codebase as a clearly documented but disabled path, consistent with CLAUDE.md's "computationally extreme paths should be opt-in toggles and clearly documented" policy.

---

*End of compute feasibility analysis.*
