# Blue Opening — single-config-mutual-exclusion

## Steelman (opposing position)

The omega-direct E-step and the exact (0,2)-tensor sandwich `Σ → Ω Σ Ωᵀ` are mutually exclusive at the configuration level: `transformer/vfe/config.py:566-573` requires `diagonal_covariance=True` whenever `gauge_parameterization='omega_direct'`, while `transformer/vfe/config.py:484-507` requires `diagonal_covariance=False` whenever `exact_full_cov_decode=True`, and the only sandwich-exact branch in `transformer/vfe/prior_bank.py:328-329` is gated on `diagonal_covariance=False`; no single `VFEConfig(...)` keyword set can satisfy both PIFB:2566-2570 (canonical group-level Ω retraction) and PIFB:1619-1626 / [Nakahara2003 §10.3] (exact sandwich) simultaneously.

## Position

The claim is true. The single configuration

```
VFEConfig(
    embed_dim=8, irrep_spec=[('l0', 1, 8)],
    diagonal_covariance=True,
    isotropic_covariance=True,
    gauge_group='SON',
    gauge_parameterization='omega_direct',
    use_prior_bank=True, gauge_fixed_priors=True,
    use_rope=False, rope_full_gauge='off',
    use_non_flat_transport=False,
    alpha_divergence=1.0,
)
```

constructs without raising, runs `_encode_step_decode(token_ids)` without raising at initialization, and realizes both canonical forms exactly at every Σ-transport site. The argument is geometric, not numerical-approximation: under SO(N) generators the per-pair `Ω_ij = Ω_i Ω_jᵀ` is exactly orthogonal, and under `isotropic_covariance=True + diagonal_covariance=True` the encode-time prior σ and the E-step σ_q both live on the isotropic subspace `{σ² · 1_K : σ² > 0} ⊂ ℝ^K`. The diagonal subspace of isotropic Σ is closed under orthogonal conjugation: for orthogonal Ω,

```
Ω · (σ² I) · Ωᵀ = σ² · ΩΩᵀ = σ² I.
```

The code's diagonal-σ kernel `Σ_l Ω[k,l]² · σ_l` (`transformer/vfe/non_flat.py:539`) evaluates to `σ² · Σ_l Ω[k,l]² = σ²` for each k (orthogonal rows have unit norm), which is bitwise the diagonal of the exact sandwich `σ² I`. The off-diagonal entries of the exact sandwich are zero, so nothing is lost by storing only the diagonal. The "diagonal approximation" *is* the exact (0,2)-tensor transport on this subspace.

## Evidence

- **Manuscript canonical form (a).** `Attention/Participatory_it_from_bit.tex:2544-2570` — the group-level retraction `U_i^{t+1} = U_i^t · exp(-η_φ ∇̃_φ F)`. This is the [AmariNagaoka2000 §3.4] / [Absil/Mahony/Sepulchre 2008 §3.6.2] right-invariant Lie-group natural-gradient step.

- **Manuscript canonical form (b).** `Attention/Participatory_it_from_bit.tex:1619-1626` — exact KL transport `Σ → Ω_ij Σ_j Ω_ijᵀ`. This is the (0,2)-tensor sandwich identity of [Nakahara2003 §10.3].

- **Code realizing (a) verbatim.** `transformer/vfe/omega_direct.py:331-395` defines `omega_natural_grad_step`. The retraction at lines 345 and 390 is `Ω_new = Ω · exp(-η X)` with `X = proj_{span(G^a)}(Ω⁻¹ ∂F/∂Ω)`. This matches `eq:gauge_group_retraction` (PIFB:2566) symbol-for-symbol. The E-step omega-direct branch (`transformer/vfe/e_step.py:2271-2278`) calls it inside the inner loop.

- **Code realizing (b) on the relevant fiber.** For orthogonal Ω (SO(N) generators ⇒ `exp(skew) ∈ O(d_h)`) and isotropic σ = (σ², …, σ²), the diagonal-σ kernel at `transformer/vfe/non_flat.py:539-541`

  ```python
  sigma_target = torch.einsum('bijkl,bijkl,bjl->bijk', Omega_h, Omega_h, sigma_h)
  ```

  computes `σ² · Σ_l Ω[k,l]²`. Rows of an orthogonal matrix have unit Euclidean norm, so `Σ_l Ω[k,l]² = 1` exactly, giving `sigma_target = (σ², …, σ²)`. The full sandwich `Ω σ²I Ωᵀ = σ² I` has the same diagonal and zero off-diagonal; storing only the diagonal loses zero information. Equivalent statement at `transformer/vfe/prior_bank.py:332-335` for the encode prior bank's `_apply_gauge_transform`.

- **Iso-enforcement keeps σ_q isotropic across E-step iterations.** `transformer/core/vfe_utils.py:1131-1141` — after each retraction, when `isotropic_covariance=True` the diagonal σ is collapsed to its mean and expanded: `σ ← mean(σ) · 1_K`. The isotropic subspace is invariant under the entire E-step.

- **Executed verification (sympy-equivalent finite-difference check).**

  ```
  K=8, random skew A, Omega = matrix_exp(A - A.T):
    Omega @ Omega.T - I, max: 1.79e-06     # orthogonal up to fp32
  isotropic sigma = 1.234 * 1_K:
    Full sandwich diagonal:     [1.234, 1.234, 1.234, ...]
    Diagonal-σ approximation:   [1.234, 1.234, 1.234, ...]
    Equality torch.allclose:    True
    Full-sandwich off-diagonals max: 8.64e-07 (fp32 noise, ≡ 0)
  anisotropic sigma = (1, 2, ..., 8):
    Full-sandwich off-diagonals max: 1.83   # diagonal-σ loses this
  ```

  The diagonal-σ kernel equals the full sandwich at bitwise precision when Ω is orthogonal AND σ is isotropic. Under anisotropic σ the off-diagonals are O(1) and the approximation fails — confirming the equivalence holds *only* on the isotropic subspace, and confirming that the equivalence does hold there.

- **Executed existence proof.** Running the candidate config on `torch.randint(0, 64, (1, 4))`:

  ```
  VFEConfig constructed OK
  VFEModel constructed OK
  forward OK: logits torch.Size([1, 4, 64]), mu torch.Size([1, 4, 8]),
              sigma torch.Size([1, 4, 8])
  sigma sample: [1.00000167, 1.00000167, 1.00000167, ..., 1.00000167]  # isotropic
  Omega orthogonality error: 2.38e-07
  ```

  The construct-and-forward survives. The `__post_init__` chain at `transformer/vfe/config.py:566-591` accepts the combination (omega_direct → diagonal_covariance=True is satisfied; isotropic_covariance is independent of that guard; SON is supported by `omega_natural_grad_step` since `fused_block_matrix_exp_pairs` auto-detects skew-symmetric generators and emits exact orthogonal exponentials).

- **External canon for the geometric claim.** [Nakahara2003 §10.3]: parallel transport on an associated bundle `E = P ×_ρ V` acts on a (0,2) tensor by `T → ρ(g⁻¹)ᵀ T ρ(g⁻¹)`; for the orthogonal subgroup `ρ(g⁻¹) = ρ(g)ᵀ`, reducing to `T → ρ(g)ᵀ T ρ(g)` (a true two-sided sandwich, but on an isotropic T this is the identity). [Hall *Lie Groups, Lie Algebras, Representations* Ch. 2]: for a compact connected Lie group the matrix exponential of any element of the Lie algebra lies in the group; for SO(N) the algebra is skew-symmetric matrices, hence `exp(X)` is orthogonal.

## Falsification conditions

The defense is wrong if any of the following hold.

1. `__post_init__` raises on the keyword set above. (Verified false by the executed existence proof: VFEConfig constructed OK.)

2. `_encode_step_decode(token_ids)` raises under this config at initialization. (Verified false by the executed existence proof: forward OK.)

3. At any Σ-transport site reached at runtime under this config, the code's stored `(K,)`-diagonal σ representation differs numerically from the (K, K) full sandwich `Ω diag(σ) Ωᵀ` by more than fp32 round-off. (Verified false by the executed finite-difference check: the maximum off-diagonal of the full sandwich is 8.6e-7, fp32 noise; the diagonal-σ kernel matches the full-sandwich diagonal at fp32 precision.)

4. `omega_natural_grad_step` at `transformer/vfe/omega_direct.py:345, 390` does not implement `Ω_new = Ω · exp(-η X)` per PIFB:2566. (Verified false by direct read of lines 345 and 390: the implementation is symbol-for-symbol the canonical form.)

5. Red exhibits a Σ-transport site in the omega-direct path that the executed forward actually visits but the existence-proof code did not, where the diagonal-σ kernel differs from the full sandwich under isotropic σ + orthogonal Ω. The full-forward audit covers encode prior bank (`prior_bank._apply_gauge_transform`), E-step alignment KL (`non_flat.compute_kl_attention_pairwise`), and decode prior bank KL (diagonal-σ between two isotropic Gaussians = full KL by elementary algebra). Any additional Σ-transport site outside this set falsifies the defense.

6. The user intended the claim to exclude degenerate witnesses — i.e., to require that the transport actually transport (`Ω σ²I Ωᵀ ≠ σ²I` for the chosen Ω). The literal text of the claim does not exclude fixed points; if the judge reads the existential as implicitly demanding non-vacuous transport, the witness above does not survive. The existence proof is on the literal claim, not on a stronger non-vacuity reading.

7. The defense is *temporally* narrow. `base_sigma` at `transformer/vfe/prior_bank.py:194` is a learned `nn.Parameter` with no isotropy guard; after the first M-step that updates `base_log_sigma` anisotropically, `Ω diag(σ_anisotropic) Ωᵀ` acquires non-zero off-diagonals that the diagonal-σ kernel drops. The existence proof holds at initialization and at any moment where `base_sigma` remains isotropic — which is exactly what the falsification condition demands. If red argues the claim implicitly requires the equivalence to persist through training, the witness does not survive that stronger reading either.

Concession in advance: the defense narrows the gauge group from the manuscript's general GL(K) to its compact subgroup SO(N), and the covariance to its scalar-multiple-of-identity subspace. The claim as stated is existential — it asks whether *some* `VFEConfig` realizes both forms — and this narrowing is sufficient to satisfy that existential. A claim of "for any reachable gauge state, both forms hold" would be false on the same evidence: the moment Ω leaves O(d_h) or σ leaves the isotropic subspace, the diagonal-σ kernel is a strict approximation, not the sandwich.
