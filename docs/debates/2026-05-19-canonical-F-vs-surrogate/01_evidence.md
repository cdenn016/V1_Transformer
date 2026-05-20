# Evidence Pack — canonical-F-vs-surrogate

Neutral fact pack. Both teams work from this file.

## Manuscript references

### Main text §4.7 — full variational free energy (`Attention/GL(K)_attention.tex:838–874`)

#### Reduced free energy form (line 843–855)

> "By combining the single-agent free energy with the alignment terms derived above and summing over all agents, we obtain the global gauge-covariant variational free energy at c* ∈ C. Substituting the optimal attention weights `β_{ij}^* = softmax_j(-E_{ij}/τ + log π_j)` from (mixture_softmax_general) into the full alignment free energy `F_align` yields the reduced alignment energy `-τ log Z_i`, where `Z_i = Σ_j π_j exp(-E_{ij}/τ)` is the partition function. The reduced (partially minimized) free energy is therefore:
>
> `F_red[{q_i}] = Σ_i D_KL(q_i ‖ p_i) - τ Σ_i log Z_i - E_q[log p(o|{k_i})]`"
>
> (line 843–852)

#### The envelope-theorem gradient form (line 859–866)

> "The envelope theorem guarantees that the gradient of `F_red` with respect to any belief parameter `x ∈ {μ_i, Σ_i, φ_i}` takes the form:
>
> `dF_red/dx = ∂F/∂x|_{β=β*} = (prior terms) + Σ_j β_{ij}* ∂E_{ij}/∂x - (observation terms)`
>
> with no explicit `∂β*/∂x` terms, since the cross-term `Σ_j (∂F/∂β_{ij})(∂β*_{ij}/∂x)` vanishes at the optimal attention where `∂F/∂β_{ij} = 0`."

#### The autograd-vs-reduced-F paragraph (line 868–874)

> "In practice, implementations differentiate the composition `Σ_j β_{ij}*(x) E_{ij}(x)` via automatic differentiation, which applies the product rule and produces additional `(∂β_{ij}*/∂x) E_{ij}` terms. This computes the gradient of the attention-weighted energy `⟨E⟩_{β*} = Σ_j β_{ij}* E_{ij}`, which differs from the reduced free energy `-τ log Z_i` by the attention-entropy term `τ Σ_j β_{ij}* log(β_{ij}*/π_j)`. Using the softmax sensitivity `∂β_{ij}*/∂x = -τ⁻¹ β_{ij}* (∂E_{ij}/∂x - ⟨∂E/∂x⟩_{β*})`, the difference between the two gradients takes the covariance form
>
> `∇_x ⟨E⟩_{β*} − ∇_x F_red = -τ⁻¹ Cov_{β*}(E_{ij}, ∂E_{ij}/∂x)`  (Eq. eq:autograd_envelope_gap)
>
> The two objectives therefore share critical points only where this covariance vanishes; this holds at the joint stationary point of F_red in x and in the high-temperature limit τ → ∞, but not generically off-equilibrium. The `∂β/∂x` terms, which vanish in the reduced-free-energy gradient by the envelope theorem, are present in the autograd gradient and provide a softmax-gradient nonlinearity (§ ffn_nonlinearity). Standard transformer training follows the autograd convention (differentiating through the softmax), and we adopt the same convention in the gradient expressions and algorithm below. **The gradient expressions below are therefore gradients of the attention-weighted energy `Σ_j β_{ij}* E_{ij}` (equivalently, derivatives of F before the β-minimization is performed, evaluated at β = β*), not of the reduced free energy F_red; the two differ by (eq:autograd_envelope_gap).**"

This is the load-bearing declaration. The framework's implementation uses the surrogate gradient.

### Main text §5.4 — softmax-gradient correction (`Attention/GL(K)_attention.tex:1937–1939`)

> "Beyond the GLU structure of the message-passing term itself, the autograd gradient of the attention-weighted energy `Σ_j β_{ij} E_{ij}` contains an additional nonlinear correction from differentiating `β_{ij}` with respect to `μ_i` (this term is absent from the gradient of the reduced free energy `F_red` by the envelope theorem, but present in standard autograd implementations; cf. §final_free_energy):
>
> `∂β_{ij}/∂μ_i = -(β_{ij}/τ)[∂E_{ij}/∂μ_i - Σ_k β_{ik} ∂E_{ik}/∂μ_i]`"

The softmax-gradient correction is named as part of the framework's GLU/FFN structure (line 1937). The framework deliberately keeps this term as a feature, not a defect.

### Supplementary §B.1 — entropy-suppressed surrogate (`Attention/GL(K)_supplementary.tex:183`)

> "For brevity we work with the entropy-suppressed surrogate `Σ_j β_{ij} D_KL(q_i ‖ Ω_{ij} q_j)` (i.e., holding the attention distribution β fixed and treating the alignment energy at this β); the canonical free energy of the main text adds the `τβ_{ij} log(β_{ij}/π_{ij})` entropy term to make the softmax form of β a stationary point. The covariance gradient is identical under both forms because the attention entropy does not depend on Σ_i."

The supplementary's working form is the surrogate. The reason given is "for brevity" but the substantive claim is that for `Σ_i`, the two forms agree because the attention entropy is `Σ_i`-independent.

### Main text §4 stationarity (`Attention/GL(K)_attention.tex:766`)

> "The softmax stationary point `β_{ij}^* = π_j exp(-E_{ij}/τ)/Z_i` is unchanged, while the substituted value becomes `F_align^(τ)* = -τ log Z_i`, matching the `-τ log Z_i` reduction used in (free_energy_final)."

The softmax is the stationary point of the canonical F (with the τβ log(β/π) entropy term), not of the surrogate. The surrogate alone (without the entropy term) does not have softmax as its stationary point — it has β concentrated at the argmin of E.

## Canon excerpts

### `external_canon_inference.md` §5 — Envelope theorem

> "If the E-step is run to a fixed point `s* = T_θ(s*)` for an operator `T_θ`, the M-step gradient through the fixed point is computed via the IFT:
> `∂s*/∂θ = (I - ∂T_θ/∂s|_{s*})⁻¹ ∂T_θ/∂θ|_{s*}`
> This requires solving a linear system at the fixed point (or approximating the inverse via Neumann series). **A single backprop through one iteration of the E-step is *amortized* inference, not IFT.** The two have different bias/variance properties."

The envelope theorem in this context says: at the stationary point β = β*, `∂F/∂β = 0`, so terms `(∂F/∂β)(∂β/∂x)` drop. This is what the manuscript's envelope-theorem step at line 866 invokes for F_red. Autograd does NOT drop these terms (it has no notion of "this is a stationary point, drop the cross-term"), so the autograd gradient retains them. The covariance form at line 871 quantifies the retained terms.

### `external_canon_inference.md` §1 — Form 1 vs Form 2 vs Form 3 conflation

> "1. **Form-1 vs Form-2 vs Form-3 conflation.** All three are equivalent algebraically, but a derivation that uses one form's terms but quotes another form's interpretation is sloppy."

Relevant analogue: the canonical F and the surrogate are NOT equivalent in this case (the manuscript itself proves the gap). Using surrogate gradients while citing canonical-F stationarity is a related risk.

### Cuturi 2013, Boyd-Vandenberghe — what the surrogate does NOT have

In the entropy-regularized soft-assignment literature, the canonical functional `f(β) = Σ β_j E_j + τ Σ β_j log(β_j/π_j)` has the softmax `β* = π exp(-E/τ)/Z` as its unique stationary point. The surrogate `Σ β_j E_j` (without the entropy term) has argmin at `β` concentrated on `argmin_j E_j` (delta function), not at the softmax. The two are NOT the same problem; their solutions are different.

This is what the supplementary at line 183 acknowledges when it says the entropy term is "added to make the softmax form of β a stationary point."

## Direct calculation — verifying the covariance form

The manuscript's derivation at line 868 is:

```
∇_x F_red = ∇_x (-τ log Z_i)
          = (1/Z_i) Σ_j π_j exp(-E_{ij}/τ) · ∂E_{ij}/∂x         [by chain rule]
          = Σ_j β_{ij}* · ∂E_{ij}/∂x                              [by def. of β*]

∇_x ⟨E⟩_{β*} = ∇_x (Σ_j β_{ij}*(x) E_{ij}(x))
             = Σ_j (∂β_{ij}*/∂x) E_{ij} + Σ_j β_{ij}* ∂E_{ij}/∂x  [product rule]

Difference:
∇_x ⟨E⟩_{β*} − ∇_x F_red = Σ_j (∂β_{ij}*/∂x) E_{ij}
                          = Σ_j [-τ⁻¹ β_{ij}* (∂E_{ij}/∂x - ⟨∂E/∂x⟩_{β*})] E_{ij}
                          = -τ⁻¹ Σ_j β_{ij}* E_{ij} (∂E_{ij}/∂x - ⟨∂E/∂x⟩_{β*})
                          = -τ⁻¹ [⟨E · ∂E/∂x⟩_{β*} - ⟨E⟩_{β*} ⟨∂E/∂x⟩_{β*}]
                          = -τ⁻¹ Cov_{β*}(E_{ij}, ∂E_{ij}/∂x).
```

The covariance form at line 871 is exact. Sub-claim α is true.

## What this evidence does NOT settle

1. **Is the autograd convention "mathematically clean" or "conceptually muddled"?**
   - Blue reading: the autograd convention computes a well-defined gradient (of `⟨E⟩_{β*}`); the manuscript discloses what it computes; the two gradients coincide at stationarity; the gap is the GLU/softmax-gradient nonlinearity that the framework explicitly wants.
   - Red reading: the framework claims F is the variational objective being minimized, but the gradient flow descends on the surrogate, which has different fixed points off-equilibrium. The stationary-point analysis (§4.6 softmax-β derivation, §5 reduction) uses F's gradient; the implementation uses the surrogate's. The analysis does not apply to the implementation off-equilibrium.

2. **Does the supplementary's "for brevity" framing properly disclose the gap?** The supplementary at line 183 says the Σ_i gradient is identical under both forms. This is correct (the attention entropy depends on β, not on Σ_i). But the μ_i and φ_i gradients are NOT identical under both forms — the supplementary derives only the Σ_i case where the question doesn't arise. The full multi-variable case where the gap is non-trivial is handled in the main text §4.7. Whether the supplementary's "for brevity" wording papers over the μ/φ case is a matter of editorial precision.

3. **Empirical content.** Does the off-equilibrium gradient gap have measurable training consequences? The manuscript at line 1937 names the softmax-gradient correction as part of the framework's "GLU/SiLU-family" activation function. Under this reading, the gap is not a bug but a feature (the framework's autograd implementation produces the gated activation function the framework also derives). Whether this reading is convincing or post-hoc rationalization depends on whether the framework's E-step convergence is documented to descend monotonically on F_red (it should not, generically, under the surrogate gradient).

4. **What is the framework actually optimizing?** Under the autograd convention, the E-step iterates descend on `⟨E⟩_{β*}` (the attention-weighted alignment energy), not on F_red. These have the same critical points at joint equilibrium but different vector fields off it. The framework's analytical results (softmax-β derivation, envelope-theorem gradient form) describe F_red; the framework's iterative implementation descends on `⟨E⟩_{β*}`. The mismatch is what red attacks; blue's defense is that the mismatch is disclosed and the difference is the GLU activation.

5. **Σ_i gradient agreement is partial.** Per supp §B.1 line 183, the Σ_i gradient is identical under both forms ONLY when β is held fixed (the surrogate's definition). If β is allowed to depend on Σ_i (which it does, since `E_{ij}` depends on Σ_i via the Gaussian KL), the autograd gradient through β re-introduces the gap on Σ_i too. The supplementary's "β held fixed" qualifier is the load-bearing condition.
