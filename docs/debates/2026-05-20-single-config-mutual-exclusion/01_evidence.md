# Evidence Pack — single-config-mutual-exclusion

## Code references

- `transformer/vfe/config.py:566-573` — `gauge_parameterization='omega_direct'` requires `diagonal_covariance=True`. Verbatim from `__post_init__`:
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
- `transformer/vfe/config.py:484-507` — `exact_full_cov_decode=True` requires `diagonal_covariance=False` (plus `gauge_fixed_priors=True` and `use_prior_bank=True`):
  ```python
  if self.exact_full_cov_decode:
      _ex_problems: List[str] = []
      ...
      if self.diagonal_covariance:
          _ex_problems.append(
              "diagonal_covariance=True (Σ is (K,) diagonal; no full-cov "
              "decode applies)"
          )
      ...
      if _ex_problems:
          raise ValueError(...)
  ```
- `transformer/vfe/omega_direct.py:1-50` — module docstring: omega-direct stores per-block `(Ω_h, Ω_h^{-1})` and updates via the right-invariant retraction on the group; explicitly the canonical group-level form of PIFB Eq:gauge_group_retraction.
- `transformer/vfe/prior_bank.py:291-372` — `_apply_gauge_transform`: diagonal-cov branch uses `diag(A diag(s) A^T)` approximation per block (line 333); full-cov branch uses the exact sandwich `A_h @ diag(s_h) @ A_h^T` per block (line 328-329).
- `transformer/vfe/prior_bank.py:_decode_exact_full_cov` (called from `prior_bank.py` decode dispatch under `exact_full_cov_decode=True`): exact per-block KL between block-diagonal Gaussians (the Law-3 pure full-cov decode).
- `transformer/vfe/e_step.py:774-775` — `if self.gauge_parameterization == 'omega_direct': return self._forward_omega_direct(...)`. The omega-direct E-step branch.
- `transformer/vfe/e_step.py` `_iter_nonflat` / omega-direct iteration — pairwise-Ω kernel asserted diagonal-σ only per the docstring and code paths.

## Manuscript references

- `Attention/Participatory_it_from_bit.tex:2544-2570` — Eq:gauge_natural_gradient `dU_i/dt = -η_φ U_i ∇̃_φ F` and discrete-time Eq:gauge_group_retraction `U_i^{t+1} = U_i^t exp(-η_φ ∇̃_φ F)`. Canonical group-level form.
- `Attention/Participatory_it_from_bit.tex:1619-1626` — exact KL under gauge transport in the deterministic-belief reduction: `Σ → Ω_ij Σ_j Ω_ij^T`, exact sandwich form.
- `Attention/Participatory_it_from_bit.tex:1252-1264` — Eq:free_energy_functional_final boxed F. The alignment term `β_ij · KL(q_i || Ω_ij q_j)` requires evaluating KL of `Ω_ij q_j`, which has covariance `Ω_ij Σ_j Ω_ij^T`. Exact = sandwich; diagonal approximation = `diag(Ω Σ Ω^T)`.
- `Attention/Participatory_it_from_bit.tex:2572-2576` — chart-coordinate `dφ_i/dt = -η_φ ∇̃_φ F + O(‖·‖²)` is the *first-order truncation* of the canonical group-level form. The canonical theoretical update is the group-level form; chart-coordinate is "implementation specialization."

## Canon excerpts (external sources of truth)

- **Nakahara 2003 §10.3** — parallel transport on associated bundles: the (0,2)-tensor (covariance) transforms by conjugation `Σ → ρ(g) Σ ρ(g)^T` under structure-group action; the sandwich product is the canonical form.
- **Amari 1998 §3.4** (natural gradient on Lie groups) / **Absil/Mahony/Sepulchre 2008 §3.6.2** (optimization on Lie groups) — the right-invariant retraction `g ← g · exp(η X)` is the canonical Lie-group natural-gradient step. The Lie-algebra chart-coordinate update `φ ← φ + η X` is the first-order BCH truncation.
- **Hall 2015 §3.3** — BCH series: `exp(X)exp(Y) = exp(X + Y + ½[X,Y] + ...)`. Truncation at first order is exact only for abelian Lie algebras.

## What this evidence does NOT settle

1. Whether the diagonal-σ approximation `diag(A diag(s) A^T)` is "transport without diagonal approximation" or "transport with diagonal approximation." Blue may argue that for `omega_direct` the relevant transport for the E-step is the pairwise-Ω-on-diagonal-σ kernel, which is *exact* under the diagonal-Gaussian fiber assumption (no sandwich product is *needed* because Σ is already in the diagonal subspace and the pairwise-Ω kernel computes `diag(Ω diag(σ) Ω^T)` as a different operator). Red will argue Nakahara §10.3 prescribes the sandwich whenever Σ is treated as a (0,2)-tensor under the group action, and the diagonal subspace is not invariant under non-orthogonal Ω.
2. Whether the existential is naturally "one config does both" (red's reading) or "for each canonical form, ∃ a config realizing it" (blue's reading). The first-debate verdict adopted the second; this debate explicitly forces the first.
3. Whether the user's "pure path" in `CLAUDE.md` ("There should ALWAYS exist a theoretically/mathematically 'pure' path under appropriate toggles") is consistent with mutual-exclusion gates. A purity that requires choosing between two canonical forms is itself a limitation of the codebase that the user may or may not endorse.
