# Verdict — single-config-mutual-exclusion

## Outcome

RED_WINS

## Decisive evidence

The two structural validation guards in `transformer/vfe/config.py` intersect at the empty set on the `diagonal_covariance` axis, and blue conceded this in the rebuttal:

- `transformer/vfe/config.py:566-573` — `gauge_parameterization='omega_direct'` raises `ValueError` unless `diagonal_covariance=True`.
- `transformer/vfe/config.py:484-507` — `exact_full_cov_decode=True` (the only path reaching the exact full-cov sandwich at `transformer/vfe/prior_bank.py:326-329`, gated on `not self.diagonal_covariance` at `prior_bank.py:310`) raises unless `diagonal_covariance=False`.
- `transformer/vfe/e_step.py:2042-2045` — defensive runtime guard raises `RuntimeError` if `diagonal_covariance=False` reaches the omega-direct dispatch.
- `transformer/vfe/non_flat.py:500-505` — pairwise-Ω KL kernel raises `NotImplementedError` on full-cov input.
- `transformer/vfe/non_flat.py:539-541` — the omega-direct kernel computes `einsum('bijkl,bijkl,bjl->bijk', Ω, Ω, σ)`, which is the diagonal-truncation `(Ω**2) @ σ`, not the (0,2)-tensor sandwich per `[Nakahara2003 §10.3]`.

Blue's rebuttal at `03_blue_rebuttal.md` lines 5-11 grants this exactly: "The two `__post_init__` guards in `transformer/vfe/config.py` intersect at the empty set on the `diagonal_covariance` axis. ... a single `VFEConfig` keyword set cannot simultaneously satisfy both clauses. ... The claim is not defensible on the evidence under its own falsification clause."

## Reasoning

Blue's opening offered a witness `VFEConfig(gauge_group='SON', isotropic_covariance=True, diagonal_covariance=True, gauge_parameterization='omega_direct', ...)` and argued that under the SO(N)+isotropic slice the algebraic identity `Ω σ²I Ω^T = σ²I` makes the diagonal-σ kernel coincide bitwise with the full sandwich. Red's rebuttal answered on three independent fronts: vacuity (the witness exhibits the sandwich only as the identity map on a one-dimensional fixed-point fiber, not as the transformation rule the canonical form encodes per `[Nakahara2003 §10.3]`); M-step drift (`base_log_sigma` at `transformer/vfe/prior_bank.py:194` carries no isotropy projector, so after one optimizer step `base_sigma` leaves the isotropic ray and the diagonal kernel becomes a strict approximation with O(1) off-diagonals as blue's own anisotropic finite-difference check confirmed); and gauge-group mismatch (PIFB:1785 fixes the language-modeling reduction's gauge group as `GL(d_head)^H`, not SO(N)).

Blue's rebuttal then conceded the structural mutual exclusion outright and stated three concrete reasons the strongest defense (the isotropic+orthogonal slice) fails: (1) `enforce_orthogonal` is not threaded into `_forward_omega_direct` at `transformer/vfe/e_step.py:1981`, so the right-invariant retraction on GL+(K) drifts off SO(K) after iteration one per `[Absil2008 §3.6.2]`; (2) the prior-bank computes the diagonal-truncation *algorithm* `(A**2) @ s` at `prior_bank.py:333` regardless of whether the numerical output happens to coincide with the sandwich on some input slice — and the claim asks for the sandwich algorithm at every site, not for numerical coincidence on an input subset; (3) the omega-direct path exposes no `exact_diagonal_transport` toggle, so there is no route from the omega-direct dispatch to the sandwich computation.

The blue-side falsification clause from `00_claim.md` reads: "Blue (defender) loses if red exhibits config.py validation code that rejects any single configuration combining `gauge_parameterization='omega_direct'` with the exact (0,2)-tensor sandwich transport on Σ at all sites where Σ is transported." Red exhibited exactly that, and blue agreed under the single-configuration reading the claim demands. This is not a split-difference case: red has cited code, manuscript line, and external canon; blue has cited the same evidence and stated the claim is indefensible on it.

## Action

Two paths forward, both acceptable, neither yet implemented:

1. Implement a fourth `VFEConfig` path that combines `gauge_parameterization='omega_direct'` with `diagonal_covariance=False` via a per-pair logdet-aware sandwich KL kernel — concretely, implement the open `NotImplementedError` at `transformer/vfe/non_flat.py:500-505` and remove the `__post_init__` guard at `config.py:566-573`. The encode prior bank's exact-sandwich branch at `prior_bank.py:326-329` already exists; the missing piece is the E-step pairwise-Ω full-cov KL kernel. This restores the user's "theoretically pure path under appropriate toggles" mandate (`CLAUDE.md`) for the joint canonical form.

2. Document the mutual exclusion explicitly in `CLAUDE.md` as a known structural limitation of the `transformer/vfe/` package — that the canonical group-level retraction `Ω^{t+1} = Ω^t · exp(-η ∇̃F)` of PIFB:2566-2570 and the canonical sandwich `Σ → Ω Σ Ω^T` of PIFB:1619-1626 do not coexist in a single `VFEConfig` and must be exercised via separate config instances. This is the per-construction reading adopted by the prior debate verdict at `docs/debates/2026-05-20-vfe-module-purity-for-pifb/04_verdict.md`; if the user endorses that reading, the present claim is OUT_OF_SCOPE for what the `/vfe` module is intended to deliver, and the action is to record the asymmetry where future readers will see it.

Recommended: option 1 if the user wants both canonical forms simultaneously reachable from one config (the natural reading of "pure path"); option 2 if the per-construction reading is acceptable and the asymmetry is intentional. Either way, the silent mutual exclusion at `config.py:484-507` and `config.py:566-573` is the artifact to either remove or surface.
