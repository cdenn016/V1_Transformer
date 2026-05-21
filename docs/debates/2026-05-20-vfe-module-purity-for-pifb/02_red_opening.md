# Red Opening — vfe-module-purity-for-pifb

## Steelman (opposing position)

Under the manuscript's own timescale-separation argument (PIFB §1502-1551), the language-modeling regime is the fast-belief subsystem in which `s_i` and `φ_i` are frozen, the surviving free energy collapses to the three-term `F_fast` of PIFB:1514-1518, and a configuration `(gauge_fixed_priors=True, use_prior_bank=True, exact_full_cov_decode=True, diagonal_covariance=False, include_attention_entropy=True, em_mode='ift_phi', use_non_flat_transport=False, alpha_divergence=1.0, e_phi_lr>0)` walks `transformer/vfe/` through encode (Law-3 gauge-orbit prior), E-step (analytic `α·KL(q‖p) + Σβ·KL(q‖Ωq) + τβ·log(β/π)` with envelope gradients), Ω-conjugation transport via `compute_gauge_transport`, and `-KL(q‖π_v)/τ` decode, exactly realizing the PIFB transformer-limit construction of §1571-1842 and Eq. \eqref{eq:gauge_qk}.

## Position

**There is no configuration of `transformer/vfe/` that realizes the canonical natural-gradient flow on the gauge frame `φ` prescribed by PIFB Eq. \eqref{eq:gauge_natural_gradient} together with the manuscript's own positive-definiteness requirement on the non-compact gauge group `GL(K)` of the transformer limit.** The pullback metric of Eq. \eqref{eq:pullback_metric} — labeled by PIFB:2558-2564 as the *only* positive-definite preconditioner option on non-compact `G` — is rejected at runtime inside the inner E-step at `transformer/vfe/config.py:597-609`, and every reachable alternative (`clip`, `cartan`, `killing`, `killing_per_block`) either has no Riemannian content (`clip`) or applies the manuscript-acknowledged indefinite Killing form on `gl(K)`. The claim therefore fails on the φ-sector of the canonical update equations.

## Evidence

### Manuscript — what is required

- PIFB:1574-1604 fixes the transformer limit as `GL(K) = GL(d_k)` (the per-head invertible matrix group), explicitly non-compact: "Note that we are embedding in $K = d_k$ dimensions in order to make contact with standard notation and, for now, the gauge group will be the fundamental $\mathrm{GL}(K = d_k)$." Multi-head lift is `GL(d_head)^H ⊂ GL(d_model)` (PIFB:1590).
- PIFB:2547-2570 (Eq. \eqref{eq:gauge_natural_gradient} and Eq. \eqref{eq:gauge_group_retraction}) is the canonical natural-gradient flow on `φ`:
  - `dU_i/dt = -η_φ U_i \tilde{\nabla}_{φ_i} F`, with `\tilde{\nabla}_{φ_i} F := G_κ^{-1}(φ_i) \nabla_{φ_i} F` (Eq. \eqref{eq:gauge_natural_gradient_def}).
- PIFB:2558-2564 states the manuscript's own positive-definiteness requirement:
  > "For non-compact $G$, the Killing form is indefinite and the pullback of a Gram inner product through $d\exp$, $\mathcal{G}_{ab}(\phi) = \langle \Psi(\mathrm{ad}_\phi) T_a, \Psi(\mathrm{ad}_\phi) T_b \rangle_G$, $\Psi(z) = (e^z - 1)/z$, [...] provides a position-dependent right-invariant metric that is positive definite for any choice of inner product $\langle\cdot,\cdot\rangle_G$ on $\mathfrak{g}$."

So `gl(K)` is non-compact ⇒ Killing form is indefinite ⇒ pullback metric Eq. \eqref{eq:pullback_metric} is the only positive-definite invariant metric option the manuscript supplies.

### Code — what is reachable in `transformer/vfe/`

- `transformer/vfe/config.py:593-609` — `VFEConfig.__post_init__` raises at runtime if `phi_preconditioner` is not in `('clip', 'cartan', 'killing', 'killing_per_block')`:
  > `phi_preconditioner=...'pullback'... is not supported. [...] 'pullback' is gated here because the inner E-step call chain in 'VFEEStep._update_phi' does not thread the structure_constants tensor that apply_pullback_natural_gradient requires.`

  Note line 606 explicitly acknowledges the metric kernel itself was corrected on 2026-05-20 and is reachable only via the *outer* optimizer `RiemannianAdamW(metric='pullback')` — i.e., on the M-step parameter manifold, not inside the E-step where `φ` actually evolves under the canonical Eq. \eqref{eq:gauge_natural_gradient} flow.
- `transformer/core/phi_evolution.py:21-79` — `precondition_phi_gradient` does accept `mode='pullback'` and routes to `apply_pullback_natural_gradient(grad_phi, phi, generators, structure_constants, gram)` at line 67-71. The kernel exists and works. The blockage is purely the config validator at `config.py:597-609` refusing to construct a `VFEConfig` with that mode.
- `transformer/vfe/e_step.py:1608-1614` — the actual inner-loop call site:
  ```
  grad_phi = precondition_phi_gradient(
      grad_phi, phi,
      mode=self.phi_preconditioner_mode,
      preconditioner=self._phi_preconditioner,
      generators=self.generators,
  )
  ```
  The `structure_constants` and `gram` arguments required by the pullback branch (`phi_evolution.py:51-53, 67-71`) are not passed. Even if the config validator were lifted, `mode='pullback'` would hit `phi_evolution.py:67` and call `apply_pullback_natural_gradient` with `structure_constants=None, gram=None`, which the inner kernel does not accept.
- `transformer/vfe/e_step.py:1611, 1964` — `phi_preconditioner_mode` is read from `cfg.phi_preconditioner` (set at `e_step.py:405`), so the validator at `config.py:597-609` is the unique gate.
- `transformer/vfe/omega_direct.py:22-23, 36-42` — the alleged "group-level" escape hatch (`gauge_parameterization='omega_direct'`) is documented as "Killing-preconditioned to match the existing φ-mode metric", and its own docstring acknowledges that "The Killing form on $\mathfrak{gl}(K) = \mathfrak{sl}(K) \oplus \mathbb{R} \cdot I$ is degenerate on the $\mathbb{R} \cdot I$ (trace/det) direction — the Riemannian metric provides no restoring force." The determinant has to be re-projected periodically via `project_omega_to_slk`. This is the manuscript's own indefinite-metric pathology surfacing in code.

### External canon — why this matters

- [AbsilMahonySepulchre2008] and [Amari1998 §3-4] require the Riemannian metric used for natural-gradient flow on a Lie group to be positive-definite for the steepest-descent property to hold. An indefinite metric does not give a descent direction in general.
- [Hall2015 §7.3] establishes that the Killing form $B(X,Y) = \mathrm{tr}(\mathrm{ad}_X \mathrm{ad}_Y)$ on $\mathfrak{gl}(K)$ is indefinite (sign-indefinite on the trace/scaling direction and on the symmetric/antisymmetric split). For semisimple $\mathfrak{so}(K)$ with $K \ge 3$ it is negative-definite (gives a positive metric after sign flip); for $\mathfrak{gl}(K) = \mathfrak{sl}(K) \oplus \mathbb{R}$ it is degenerate on the scalar direction.
- The standard preconditioned-Lie-group pitfall flagged in `external_canon_transformers.md §10 item 6` and `external_canon_math.md §3 item 5`: "Using natural gradient with the wrong Fisher" / "Using the Killing form on a non-compact Lie algebra without justification" — applies directly. The manuscript itself flags this pitfall (PIFB:2558) and supplies Eq. \eqref{eq:pullback_metric} as the remedy. The remedy is unreachable in `transformer/vfe/` under any toggle.

### Synthesis

The compound claim asks "for every PIFB construction X (in scope), there exists a toggle realizing X in `transformer/vfe/`". The φ-sector of the canonical update equation Eq. \eqref{eq:gauge_natural_gradient} requires `\tilde{\nabla}_{φ_i} F` computed with a positive-definite invariant metric on `gl(K)`. Per PIFB:2558-2564, the unique positive-definite option supplied by the manuscript is the pullback metric of Eq. \eqref{eq:pullback_metric}. That toggle is **rejected at config construction time** by `config.py:597-609`, with the explanatory docstring acknowledging the gating is intentional (the E-step call chain does not thread `structure_constants`). The compound claim therefore fails: there is one PIFB construction (the canonical `φ`-natural-gradient flow) with no realizing toggle inside `transformer/vfe/`.

This attack is restricted to language-modeling scope: it concerns the gauge group `GL(K)` of the transformer limit (PIFB:1574-1604), not any meta-agent / mass-analogy / Lorentzian-signature construction. It does not depend on the default `norm_type` or any other excluded default. It depends only on the existential question the user posed: is there *any* toggle setting that walks the canonical construction end-to-end?

## Falsification conditions

This position is wrong if blue produces *any one* of the following:

1. **A code path that threads `structure_constants` and `gram` into `precondition_phi_gradient` at `e_step.py:1608-1614` under some toggle setting**, bypassing the `config.py:597` validator, such that `mode='pullback'` actually runs inside the inner E-step. (The kernel exists at `phi_evolution.py:67-71`; the gate is the validator.)
2. **A primary-source citation showing that the Killing form on `gl(K)` is positive-definite in some sense relevant to the natural-gradient steepest-descent property**, contradicting [Hall2015 §7.3] and PIFB's own statement at 2558.
3. **A primary-source citation showing that the canonical natural-gradient flow Eq. \eqref{eq:gauge_natural_gradient} does not require positive-definiteness of the preconditioner** (e.g., a derivation that the flow is well-defined under an indefinite invariant form). [Absil/Mahony/Sepulchre 2008] and [Amari1998] are the relevant canon; a primary source going against them would falsify.
4. **A demonstration that with `e_phi_lr=0` across all toggles, the φ-sector simply does not evolve in the language-modeling regime under PIFB's adiabatic argument**, and therefore the preconditioner choice is *vacuous* — making the missing pullback option a non-defect. This would require a citation in PIFB §1502-1569 that the language-modeling timescale freezes `φ` *inside the inner E-step* (not merely on the slow learning timescale), and a verification that `e_phi_lr` is not a tunable knob the claim must cover. Note: the active config has `e_phi_lr=0.0`, but the user's claim is about whether *a configuration path exists* — toggles with `e_phi_lr > 0` are in scope.
5. **A toggle combination that yields some other manuscript-endorsed positive-definite invariant metric on `gl(K)`** (the manuscript supplies exactly two options at §2558-2564: Killing form, which is indefinite on `gl(K)`; and pullback, which is gated out). If blue locates a third option in the manuscript that is reachable, the position falls.
