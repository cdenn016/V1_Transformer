# Blue Opening — canonical-F-vs-surrogate

## Steelman (opposing position)

Red's strongest line: the manuscript writes the canonical `F_red` (Eq. `free_energy_final`, line 845) as the *variational objective being minimized*, derives the softmax-β stationarity from the entropy-augmented Lagrangian (line 768), and then at line 874 silently substitutes a different objective — the attention-weighted energy `⟨E⟩_{β*}` — for the implementation's E-step descent. Because the two gradients differ off-equilibrium by `-τ⁻¹ Cov_{β*}(E_{ij}, ∂E_{ij}/∂x)` (line 870, supp line 183), the framework's E-step iterates descend on a vector field whose stationary points coincide with `F_red`'s *only* at the joint fixed point or in the τ → ∞ limit. The stationary-point analysis (softmax-β derivation, envelope-form gradient) therefore describes `F_red`; the implementation descends on `⟨E⟩_{β*}`; the two are not the same objective off-equilibrium, and the framework's interpretive claims about minimizing `F_red` are accordingly weaker than they read.

## Position

The manuscript's autograd-convention adoption is mathematically clean, explicitly disclosed at the point of substitution, and consistent with the framework's broader architectural correspondence. Specifically:

(i) The covariance-gap formula at line 870–872 is correct as a theorem (verified by direct calculation and sympy, below).
(ii) The substitution is *announced* in plain language at line 874 — "The gradient expressions below are therefore gradients of the attention-weighted energy `Σ_j β_{ij}* E_{ij}` ... not of the reduced free energy `F_red`; the two differ by (eq:autograd_envelope_gap)." This is disclosure, not concealment.
(iii) Both objectives share the same critical-point set on the joint fixed-point manifold where `β = β*(x)` solves `∂F/∂β = 0` *simultaneously* with `∂F_red/∂x = 0`. The E-step's β recomputation at each iteration keeps β on the stationarity manifold for any current x; the gap is therefore the off-equilibrium-in-x discrepancy only, not an arbitrary objective swap.
(iv) The retained `∂β/∂x` terms are not an algorithmic accident — the manuscript at line 1937 identifies them with the softmax-gradient nonlinearity that constitutes the framework's GLU/FFN-style activation in the variational E-step. The autograd convention is therefore *load-bearing* for the FFN-as-VFE-iterations correspondence (§ffn_nonlinearity, line 1944–1946), not a corner cut.
(v) The supplementary §B.1 statement at line 183 is mathematically correct on its own terms: the attention-entropy term `τβ log(β/π)` does not depend on `Σ_i`, so the Σ_i gradient is identical under both forms when β is held at its softmax value. The supplementary derives only the Σ_i case; the μ_i and φ_i cases — where the gap is non-trivial — are handled in the main text §4.7 where the gap is explicitly written.

## Evidence

- **Primary-source citation, main text:** `Attention/GL(K)_attention.tex:874` — "The gradient expressions below are therefore gradients of the attention-weighted energy `Σ_j β_{ij}* E_{ij}` (equivalently, derivatives of F before the β-minimization is performed, evaluated at β = β*), not of the reduced free energy `F_red`; the two differ by (eq:autograd_envelope_gap)." This is an explicit, in-line declaration of which objective is being differentiated. The manuscript does not nominally define `F_red` and silently differentiate `⟨E⟩`; it names both, derives the gap, and announces the choice.

- **Primary-source citation, manuscript gap formula:** `Attention/GL(K)_attention.tex:870–872` — Eq. `eq:autograd_envelope_gap`:
  `∇_x⟨E⟩_{β*} − ∇_x F_red = -τ⁻¹ Cov_{β*}(E_{ij}, ∂E_{ij}/∂x)`.
  This is the manuscript's own derivation of the gap. It is not a critic's discovery the manuscript failed to disclose; it is a labeled, equation-numbered result of the manuscript.

- **Sympy verification of sub-claim α (gradient-gap covariance identity):**

  ```python
  import sympy as sp
  n = 3
  tau = sp.Symbol('tau', positive=True); x = sp.Symbol('x')
  E = [sp.Function(f'E{j}')(x) for j in range(n)]
  Z = sum(sp.exp(-Ej/tau) for Ej in E)
  beta = [sp.exp(-Ej/tau)/Z for Ej in E]
  F_red = -tau * sp.log(Z)
  grad_F_red = sp.diff(F_red, x)
  avg_E = sum(beta[j]*E[j] for j in range(n))
  grad_avg = sp.diff(avg_E, x)
  gap = sp.simplify(grad_avg - grad_F_red)
  dE = [sp.diff(Ej, x) for Ej in E]
  mean_E = sum(beta[j]*E[j] for j in range(n))
  mean_dE = sum(beta[j]*dE[j] for j in range(n))
  mean_E_dE = sum(beta[j]*E[j]*dE[j] for j in range(n))
  cov_form = -(1/tau)*(mean_E_dE - mean_E*mean_dE)
  print(sp.simplify(gap - cov_form))   # → 0
  ```

  Output: `gap - cov_form = 0`. The covariance form at line 871 is the exact algebraic difference. Sub-claim α is verified.

- **GLU correspondence, primary-source citation:** `Attention/GL(K)_attention.tex:1937` — "Beyond the GLU structure of the message-passing term itself, the autograd gradient of the attention-weighted energy `Σ_j β_{ij} E_{ij}` contains an additional nonlinear correction from differentiating `β_{ij}` with respect to `μ_i` (this term is absent from the gradient of the reduced free energy `F_red` by the envelope theorem, but present in standard autograd implementations; cf. §final_free_energy)." Lines 1944–1946 then state that "Each variational channel (μ, Σ, φ) thus contributes its own GLU-type activation and the composition of these channels within each VFE iteration produces a richer non-linear map than any single channel alone." The retained `∂β/∂x` term is the framework's *predicted* second-order nonlinearity; the autograd convention is the implementation route to it. The two objectives differ in a way the framework architecturally exploits, not a way it accidentally drifts.

- **Canonical-form citation, [external_canon_inference §1, Form 3]:** Standard variational F = `KL(q‖p) + E_q[-log p(o|s)]` (accuracy + complexity). The user's `F_red` reduces to this single-agent form when the multi-agent coupling `-τ log Z_i` is zero (one-agent or τ → ∞). The autograd-substitution issue does not arise for the single-agent terms (KL(q‖p), -E_q log p(o)); it arises only in the multi-agent coupling block where β is introduced. That block's stationary-point analysis (softmax-β at line 768) uses the entropy-augmented Lagrangian `Σ β E + τ Σ β log(β/π)`. The autograd form `Σ β*(x) E(x)` is the *post-minimization-in-β* energy at `β = β*(x)` — equivalently, the framework's "envelope form evaluated before the β-stationarity argument is taken." Both forms have the same value at β = β*(x); they differ only in how derivatives in x propagate.

- **Supplementary §B.1 citation:** `Attention/GL(K)_supplementary.tex:183` — "The covariance gradient is identical under both forms because the attention entropy does not depend on Σ_i." This is mathematically correct (the `τβ log(β/π)` term contains β but not Σ_i directly; ∂/∂Σ_i acts on `E_{ij}(Σ_i)` only). The supplementary's "for brevity" framing is therefore *justified* for the Σ_i derivation specifically — the entropy term contributes zero to ∂F/∂Σ_i regardless of whether β is held fixed. Sub-claim δ is true; the supplementary does not paper over the μ/φ case because it does not treat the μ/φ case — that is the main-text §4.7's job, and §4.7 derives the gap explicitly (line 870).

- **Magnitude of the gap, scale argument:** From line 868 and the manuscript's `τ = κ√K` convention at line 857, with K = 64 and κ ≈ 1, τ ≈ 8; with K = 512, τ ≈ 22. The gap scales as τ⁻¹ times a covariance of bounded-energy quantities, so the off-equilibrium gradient direction differs from `∇F_red` by an O(1/τ) correction whose magnitude is bounded by the std-deviation product `√Var_{β*}(E) · √Var_{β*}(∂E/∂x)`. At joint stationarity the covariance vanishes by sub-claim β; off-equilibrium the τ⁻¹ scaling makes it a perturbation of the canonical envelope gradient, not a wholesale substitution.

- **Standard transformer training practice:** Vaswani et al. 2017 §3.2.1 (scaled dot-product attention) differentiates through the softmax as standard practice. No transformer training procedure detaches β from x in backprop; the autograd convention is the field standard. The manuscript's adoption of it at line 874 ("Standard transformer training follows the autograd convention (differentiating through the softmax), and we adopt the same convention") is therefore consistent with the framework's broader correspondence with standard transformer practice — not a deviation that needs special justification.

## Falsification conditions

This position is wrong if any of the following hold:

1. **The gap formula at line 870 is algebraically wrong.** If a direct calculation or sympy verification yields anything other than `-τ⁻¹ Cov_{β*}(E_{ij}, ∂E_{ij}/∂x)` for `∇_x⟨E⟩ − ∇_x F_red`, the manuscript's own theorem is false and the disclosure is incoherent. Falsifier: produce a counter-example or sympy session showing a non-zero residual. (My sympy session above shows residual = 0; this falsifier is closed unless red produces a different one.)

2. **The disclosure at line 874 is hidden, ambiguous, or contradicted elsewhere.** If the manuscript elsewhere claims `F_red` is the objective being descended on by the E-step (not just the objective whose stationary points are studied), the in-line disclosure is undone. Falsifier: cite a manuscript line where the E-step is claimed to descend on `F_red` itself, not on `⟨E⟩_{β*}`. Lines 1483 ("the autograd objective") and 1937 ("the autograd gradient ... contains an additional nonlinear correction") are *consistent* with line 874, not contradictory.

3. **The supplementary §B.1 claim is wrong for Σ_i.** If holding β fixed and including the entropy term yield different Σ_i gradients (contrary to "the attention entropy does not depend on Σ_i"), the supplementary's brevity choice is false. Falsifier: show that `∂[τβ log(β/π)]/∂Σ_i ≠ 0` when β is at its softmax value. (Standard chain rule: when β = β*(Σ_i), the envelope theorem at β = β* makes the explicit ∂/∂Σ_i contribution from the entropy term vanish anyway, because `∂F/∂β = 0` at β = β*. The supplementary's stronger "β held fixed" route is even more directly zero.)

4. **The GLU/FFN identification at line 1937 is post-hoc.** If the manuscript's GLU correspondence claim (line 1937, 1944–1946) was introduced *after* the autograd-convention adoption was forced by implementation choice, it is rationalization, not derivation. Falsifier: a commit-history or draft-history showing the GLU claim was added after the autograd-vs-envelope discussion was forced by code. (Out of scope for this debate's evidence pack; absent such evidence, the manuscript's structural argument — autograd-retained `∂β/∂x` = softmax-gradient nonlinearity = GLU's second-order term — stands as a derivable correspondence, not post-hoc framing.)

5. **The E-step actually fails to converge** under the autograd / surrogate vector field at typical κ, K, training temperatures. If the framework's iterative belief updates demonstrably diverge or oscillate rather than converging to a useful fixed point — and if `F_red` itself would converge under canonical-gradient flow at the same hyperparameters — the autograd convention is empirically inferior, not just non-canonical. Falsifier: a training run showing E-step divergence under the implemented gradient with κ, K, τ in the manuscript's working regime. (Empirical: the framework's working models, including the user's best EM_CONFIG, train to convergence under this convention; this falsifier is empirically closed at the working regime.)

6. **The framework's analytical results (softmax-β derivation, envelope-theorem gradient form, residual-gradient flow at line 1478) are quoted as describing the implementation's dynamics off-equilibrium.** If any line of the manuscript says "the E-step descends on `F_red` along the envelope-form gradient at line 862" — without the line 874 caveat — the analytical-implementation correspondence is overclaimed. Falsifier: produce such a line. I have grepped lines 760–880, 1340–1490, and 1925–1948 and found only the line-874 caveat repeated (lines 1483, 1937 carry the same "the autograd objective" / "absent from F_red by the envelope theorem, but present in autograd implementations" framing).

If none of (1)–(6) hold under the evidence, the manuscript's autograd-convention adoption is what it announces itself to be — a deliberate, disclosed, derivation-consistent algorithmic choice whose off-equilibrium gradient gap is *the* second-order softmax nonlinearity the framework predicts as the FFN's GLU-style activation.
