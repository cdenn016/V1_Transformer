# Red Opening — single-config-mutual-exclusion

## Steelman (opposing position)

A `VFEConfig(gauge_parameterization='omega_direct', diagonal_covariance=True, ...)` realizes the canonical group-level retraction `Ω^{t+1} = Ω^t · exp(-η ∇̃F)` on the gauge state and simultaneously realizes covariance transport `Σ → Ω Σ Ω^T` exactly, because under a block-diagonal-Σ ansatz the relevant transport is the pairwise-Ω-on-diagonal-σ kernel, which (the steelman argues) computes the sandwich exactly on the diagonal subspace without approximation.

## Position

The claim is false. The `VFEConfig` dataclass enforces two structurally mutually exclusive guards at construction time: `omega_direct` requires `diagonal_covariance=True` (`transformer/vfe/config.py:566-573`), and exact full-covariance transport in the prior bank requires `diagonal_covariance=False` (`transformer/vfe/config.py:484-507`, with `exact_full_cov_decode=True`; and the prior bank's full-cov sandwich branch at `prior_bank.py:326-329` is gated on `diagonal_covariance=False` per `prior_bank.py:310`). No single keyword set satisfies both. Worse, the steelman's "the diagonal subspace is invariant under Ω" defense fails as a matter of standard differential geometry: the (0,2)-tensor sandwich `Ω Σ Ω^T` does not preserve the diagonal subspace under non-orthogonal Ω ∈ GL+(K) (Nakahara 2003 §10.3; Frankel 2011 Ch. 17). The `diag(Ω diag(σ) Ω^T)` kernel used under `omega_direct` is the standard diagonal-truncation approximation, not the exact sandwich.

## Evidence

- **Config-level mutual exclusion (path:line).** `transformer/vfe/config.py:566-573`:
  ```python
  if self.gauge_parameterization == 'omega_direct':
      if not self.diagonal_covariance:
          raise ValueError(
              "gauge_parameterization='omega_direct' currently requires "
              "diagonal_covariance=True (the omega-direct path uses the "
              "same pairwise-Omega KL kernel as use_non_flat_transport, "
              "which is diagonal-σ only)."
          )
  ```
  Setting `gauge_parameterization='omega_direct'` raises unless `diagonal_covariance=True`.

- **The opposing constraint (path:line).** `transformer/vfe/config.py:491-495`:
  ```python
  if self.diagonal_covariance:
      _ex_problems.append(
          "diagonal_covariance=True (Σ is (K,) diagonal; no full-cov "
          "decode applies)"
      )
  ```
  Setting `exact_full_cov_decode=True` raises unless `diagonal_covariance=False`. The two guards intersect at the empty set in the `diagonal_covariance` dimension: `omega_direct` forces `diagonal_covariance=True`; exact full-cov decode forces `diagonal_covariance=False`.

- **The "exact sandwich" branch is gated off the omega-direct path (path:line).** `transformer/vfe/prior_bank.py:310, 326-329`:
  ```python
  full_cov = not self.diagonal_covariance
  ...
  if full_cov:
      # Exact: Sigma_h = A_h @ diag(s_h) @ A_h^T  (sandwich product)
      sigma_diag = torch.diag(base_sigma_h)
      sigma_h = exp_h_f32 @ sigma_diag @ exp_h_f32.transpose(-2, -1)
      sigma_parts.append(sigma_h)
  else:
      # Diagonal approx: diag(A diag(s) A^T) = sum_j A_{ij}^2 * s_j
      sigma_parts.append(
          (exp_h_f32 ** 2 @ base_sigma_h).clamp(min=self.eps)
      )
  ```
  When `diagonal_covariance=True` (which `omega_direct` mandates), the prior bank evaluates `Σ_v = diag(A diag(s) A^T) = (A**2) @ s`, a diagonal-truncation approximation that drops every off-diagonal entry of the true sandwich. The exact sandwich is only reached on the `full_cov` branch, which is unreachable from the `omega_direct` config.

- **E-step dispatch confirmation (path:line).** `transformer/vfe/e_step.py:774-775`:
  ```python
  if self.gauge_parameterization == 'omega_direct':
      return self._forward_omega_direct(beliefs, priors, mask)
  ```
  This branch routes through the pairwise-Ω-on-diagonal-σ KL kernel (per the docstring at `config.py:570-572`, "the same pairwise-Omega KL kernel as use_non_flat_transport, which is diagonal-σ only"). Under `omega_direct`, every E-step pairwise alignment KL uses the diagonal kernel; the exact sandwich is never evaluated in the inner loop.

- **Standard form of (0,2) tensor transport (canon citation).** Nakahara 2003 §10.3 (parallel transport on associated bundles): for a (0,2)-tensor under structure-group action `ρ(g)`, the canonical transport is the two-sided sandwich `T → ρ(g)^T T ρ(g)` (or `ρ(g^{-T}) T ρ(g^{-1})` depending on covariant/contravariant identification). `[Nakahara2003 §10.3]`. The vfe-knowledge canon at `external_canon_math.md:89-104` confirms: "Two-sided sandwich. ... The sandwich (two-sided conjugation) is the standard for bilinear forms."

- **Diagonal subspace is not preserved under non-orthogonal Ω.** For Ω ∈ GL+(K) generic (the actual support of `omega_direct`, see `omega_direct.py:7-8`: "Ω_i ∈ G (block-diagonal GL+(K) in /vfe)"), `Ω diag(σ) Ω^T` has off-diagonal entries `(Ω diag(σ) Ω^T)_{ij} = Σ_k Ω_{ik} σ_k Ω_{jk}` which are generically non-zero for i ≠ j. The pairwise-Ω kernel computing `(Ω**2) @ σ` retains only the diagonal of `Ω diag(σ) Ω^T` and discards the off-diagonal information. This is the classical "one-sided conjugation on a bilinear form" pitfall flagged at `external_canon_math.md:131` — a documented standard-literature mistake.

- **Canonical Lie-group retraction (canon citation).** Amari 1998 §3.4 and Absil/Mahony/Sepulchre 2008 §3.6.2: the right-invariant retraction `g ← g · exp(η X)` is the canonical Lie-group natural-gradient step. `omega_direct.py:24-27` implements this exactly. So the group-level retraction (a) is realized — but only in the configuration that mandates the diagonal-σ kernel and so forbids (b).

## Falsification conditions

This red position is wrong if blue exhibits any of the following.

1. A concrete `VFEConfig(...)` keyword set (literal Python dict) that constructs without raising AND uses `gauge_parameterization='omega_direct'` AND, at every Σ-transport site (E-step pairwise KL kernel, encode prior bank, decode prior bank), evaluates `Ω Σ Ω^T` as a (K,K) tensor without dropping off-diagonal entries. The two config guards (`config.py:566-573` and `config.py:491-495`) must both be satisfied simultaneously — including the case where some upstream override quietly flips `diagonal_covariance` after construction.

2. A primary-source citation (Nakahara, Kobayashi-Nomizu, Frankel, or equivalent textbook) showing that the diagonal-truncation `(Ω**2) @ σ` is the canonical (0,2)-tensor transport on the diagonal subspace under non-orthogonal Ω — i.e., that `diag(Ω diag(σ) Ω^T)` is the sandwich product on the appropriate associated bundle, not an approximation.

3. An executed runtime trace where `_encode_step_decode(token_ids)` is called under the candidate single config and the values at every Σ-transport site match what `Ω Σ Ω^T` would produce under the manuscript's full-covariance Σ, including off-diagonal entries — i.e., the diagonal kernel agrees with the sandwich numerically (which it cannot, in general, since `diag(Ω diag(σ) Ω^T)_{ii} = Σ_k Ω_{ik}^2 σ_k` discards `(Ω diag(σ) Ω^T)_{ij}` for i ≠ j).
