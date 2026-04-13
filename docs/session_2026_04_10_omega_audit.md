# Deep Audit: `gauge_param='omega'` Path — 2026-04-10

Three parallel agents audited the direct omega parameterization path where `Ω_i ∈ GL(K)` are learned directly instead of via `Ω_i = exp(φ_i · G)`.

## Verdict

**All 4 prior audit fixes (#10, #16, #31, #32) are confirmed live in the current code.** The omega path is production-ready for the current config. No critical bugs found. Several improvement opportunities and one missing assertion identified.

## Prior Fixes Verified Live

| Fix | Description | Status |
|-----|-------------|--------|
| #10 | `_retract_omega` left-invariant pullback via `torch.linalg.solve` | LIVE (lines 986, 1007) |
| #16 | Safe inverse in `compute_transport_operators_direct` and `omega_to_block_exp_pairs` | LIVE (lines 460, 513) |
| #31 | Safe inverse in `_compute_omega_grad_direct` | LIVE (lines 856-866) |
| #32 | `lambda_softmax` split in `_compute_omega_grad_direct` | LIVE (lines 910-911) |

## New Findings

### Finding 1 (MEDIUM): Missing `sum(irrep_dims) == K` assertion in `omega_to_block_exp_pairs`

**File:** `transport_ops.py:486-520`

`fused_block_matrix_exp_pairs` has a `ValueError` guard (Fix #36 from round 6) asserting `sum(irrep_dims) == generators.shape[-1]`. The omega equivalent `omega_to_block_exp_pairs` has no such assertion. A caller passing mismatched `irrep_dims` would silently process only part of omega with no error.

### Finding 2 (LOW): Redundant omega inverse computation (31 sites)

The same omega block inverse is computed independently in:
- `model.py:forward()` — cached_head_transports construction
- `blocks.py:forward()` — cached_head_transports construction  
- `variational_ffn.py:_build_block_exp_pairs()` — per E-step iteration
- `variational_ffn.py:_compute_omega_grad_direct()` — gradient computation
- `transport_ops.py:compute_transport_operators_direct()` — full pairwise transport
- `transport_ops.py:omega_to_block_exp_pairs()` — block adapter
- `attention.py:compute_attention_weights()` — attention routing

With `n_iterations=1`, omega is inverted ~4-5 times per forward pass per head. Each is a `torch.linalg.inv` call on `(B, N, d_h, d_h)`. Could be cached once and shared.

### Finding 3 (LOW): P-flow / Hebbian path doesn't update omega_embed

The Hebbian learning path (`use_p_flow=True`, `use_delta_rule_w_out=True`) updates `mu_embed` and `phi_embed` but NOT `omega_embed`. If someone enables P-flow with `gauge_param='omega'`, omega embeddings remain at their initialization. Currently P-flow is off in EM_CONFIG, so this is latent.

### Finding 4 (LOW): RiemannianAdamW applies Killing form to omega gradients

**File:** `optimizer.py:103-126`

Line 103: `if 'phi' in name or 'omega' in name: self._precondition_phi(group)` applies `grad @ K_inv` to omega gradients. The Killing form `K_inv` is the metric on the Lie algebra generators. For omega (direct matrix entries), this is the correct metric ONLY near identity (where the Lie algebra and group tangent space coincide via the exponential map differential). As omega drifts from identity, this becomes an approximation. This is acceptable for a first-order optimizer and the shapes match (both are `n_gen = sum(d_h^2)` dimensional), but it's undocumented.

### Finding 5 (INFO): No multihead or long-training tests

Test coverage uses single-block configs (K=3,4,6). No test with multi-head `irrep_dims=[10,10]`. No integration test tracking omega conditioning over 1000+ steps to detect singular drift.

## What Was Verified Correct

- **Embedding initialization**: Near-identity with small perturbation (lines 289-303 of embeddings.py)
- **Forward pass 4-tuple return**: `(mu, sigma, phi, omega)` with dummy phi for compatibility
- **Block-diagonal reshaping**: Flat `(V, total_omega_params)` → block-diagonal `(B, N, K, K)`
- **Transport cocycle**: `Ω_ij = Ω_i · Ω_j⁻¹` satisfies `Ω_ij · Ω_jk = Ω_ik` (tested)
- **Non-flat transport**: `Ω_ij = Ω_i · exp(α·δ_ij·G) · Ω_j⁻¹` correctly sandwiches connection
- **Retraction geometry**: Left-invariant `ξ = Ω⁻¹ · grad` via `torch.linalg.solve` (Fix #10)
- **Trust region**: Clips `||ξ||_F` in the Lie algebra (left-invariant metric)
- **Multi-layer propagation**: `_last_omega` propagates evolved omega between blocks
- **Optimizer grouping**: omega_embed gets its own param group with `omega_lr` (defaults to `M_phi_lr`)
- **Test coverage**: 9 test suites in `test_omega_gradient.py` + 6 tests in `test_transport_ops.py`
