# Free Energy Entropy-Term Audit (Eq. 27 / 32 / 39)

## Verdict

**Refuted as a bug; partially confirmed as a presentation hazard.** The manuscript is internally consistent because Section 3.4 ("The Envelope Theorem and the Reduced Free Energy", line 867 onward) explicitly labels Eq. 32/39 as the *full (un-minimized) joint* free energy with `β` treated as a free variational parameter, and writes the *reduced* free energy with the `-τ log Z_i` term in a separate boxed equation (Eq. `eq:free_energy_reduced`, lines 873-882). The envelope-theorem paragraph also disambiguates "autograd convention" (`<E>_β*`) from "reduced FE gradient" (`-τ log Z_i`) and explicitly tells the reader they differ by the entropy/prior term. The user's algebra is correct, and the alleged inconsistency would be a bug in any manuscript that did not contain Section 3.4 — but this one does.

The remaining hazard is purely presentational: the boxed Eq. 39 visually shows `β_ij(c) KL(...)` *and* substitutes the softmax solution for `β_ij(c)` underneath, which on a casual read looks like the manuscript is claiming `<E>_β* = F_red`. Section 3.4 then immediately corrects this. A reader who stops at the box will misread the value. Recommended fix is small and editorial.

## Quoted text

### Eq. 27 (line ~642), `eq:mixture_energy_entropy`

> Therefore, we may define an alignment energy `E_ij ≡ D_KL[q_i || Ω_ij q_j]` such that:
>
> ```
> F_align = sum_j β_ij ( E_ij + log β_ij - log π_j )    (27)
> ```
>
> This has the usual "energy minus entropy" form `F_align = <E>_β - H(β) + const`, where `H(β) = -sum_j β_ij log β_ij` is the entropy of the attention distribution.

The temperature τ is introduced one paragraph later (line ~677) as `E_ik → E_ik/τ`, equivalent to the form `sum β_ij E_ij + τ sum β_ij log(β_ij/π_j)` the user quoted.

### Eq. 32 (line ~802), `eq:pointwise_free_energy`

> ```
> F(c) = sum_i KL(q_i(c) || p_i(c)) + sum_i KL(s_i(c) || r_i(c))
>      + sum_{i,j} β*_ij(c) KL(q_i(c) || Ω_ij(c) q_j(c))
>      + sum_{i,j} γ*_ij(c) KL(s_i(c) || Ω̃_ij(c) s_j(c))
>      - sum_i E_{q_i(c)}[log p(o(c) | k_i)]
> ```
>
> where `β*_ij(c)` and `γ*_ij(c)` are the optimal softmax attention weights from `eq:softmax_attention_general`, and the alignment terms emerge directly from the consensus-energy construction.

No entropy/prior term `τ Σ β log(β/π)` is shown here.

### Eq. 39 (line ~828), `eq:free_energy_functional_final`

The boxed continuum functional has the same structure with `β_ij(c) KL(...)` and `γ_ij(c) KL(...)`. Immediately below the box:

> `β_ij(c) = exp[-(1/τ) KL(...)] / Σ_k exp[-(1/τ) KL(...)]`

(uniform-prior softmax form).

### Disambiguation passage at lines 867-905, Section 3.4

> The boxed functional `eq:free_energy_functional_final` is the *full* (un-minimized) free energy: `β_ij` and `γ_ij` appear as variational attention parameters that are optimized jointly with the belief and model fields. Substituting the optimal weights `β*_ij = softmax_j(-E_ij/τ + log π_j)` from `eq:softmax_attention_general` into the alignment free energy `F_align` `eq:mixture_energy_entropy` yields the reduced alignment energy `-τ log Z_i` [...]

Reduced form (Eq. `eq:free_energy_reduced`):

> ```
> F_red[{q_i}] = sum_i KL(q_i || p_i) - τ sum_i log Z_i - E_q[log p(o | {k_i})]
> ```
>
> where `E_ij = D_KL(q_i || Ω_ij q_j)` and `Z_i = sum_j π_j exp(-E_ij/τ)`. The log-partition function `-τ log Z_i` is the result of substituting `β*_ij` into *both* the energy term `Σ_j β_ij E_ij` *and* the entropy term `τ Σ_j β_ij log(β_ij/π_j)` of `F_align`. The full (un-minimized) alignment energy includes the attention entropy/prior term, which is essential: it is this term that makes the β-optimization yield the softmax rather than an arg-min.

And lines 897-899 nail it down:

> In practice, implementations differentiate the composition `Σ_j β*_ij(x) E_ij(x)` via automatic differentiation [...]. This computes the gradient of the attention-weighted energy `<E>_β = sum_j β*_ij E_ij`, which differs from the reduced free energy `-τ log Z_i` by the entropy term `τ Σ_j β*_ij log(β*_ij/π_j)`. These two objectives share the same critical points but have different gradient fields away from equilibrium.

## Algebraic verification (sympy)

For `F_align = Σ_j β_j (E_j + τ log β_j - τ log π_j)` with `β*_j = π_j exp(-E_j/τ) / Z`, `Z = Σ_k π_k exp(-E_k/τ)`:

```
F_align(β*) - (-τ log Z) = 0          # exact
<E>_β* - F_align(β*) + τ KL(β* || π) = 0   # exact
```

Both identities are simplified to zero by SymPy. The user's two relations

- `F_align(β*) = -τ log Z`
- `<E>_β* = -τ log Z - τ KL(β* || π)`

are correct. (Note: `<E>_β* + τ KL(β* || π) = -τ log Z`; equivalently `<E>_β* = -τ log Z - τ KL(β* || π)`. Sign convention in the brief was correct.)

## Manuscript intent

Eq. 32 / 39 is the **joint free energy `F(q, β=β*)` with `β` treated as a free variational parameter and the softmax solution substituted in *as a definition of `β*`*, not as a substitution into the value of F**. The text at lines 870-871 calls this "the full (un-minimized) free energy" with `β_ij` "appear[ing] as variational attention parameters that are optimized jointly". So:

- It is **not** the reduced free energy `F_red` (which has `-τ log Z_i`, Eq. `eq:free_energy_reduced`).
- It is **not** strictly `<E>_β*` either, because in Eq. 39 the entropy term has been dropped, not subtracted.
- It is best described as **the autograd-convention training objective**, which the manuscript explicitly elects (line 902): "Standard transformer training follows the autograd convention (differentiating through the softmax), and we adopt the same convention in the gradient expressions below."

So the framing is option (c) from the brief — the envelope-theorem / autograd form valid because critical points coincide — and the manuscript *does* label it, just not at the boxed equation itself.

## Downstream impact

Looking for places where the value (not just the gradient) of F is quoted:

- **Line 1183** (second variations / mass matrix): explicitly says "treat the attention weights `β_ij` as fixed at their softmax-equilibrium values, equivalent to the envelope-theorem convention" and that "the convention switch here is deliberate". Convention is consistent with Section 3.4 — no bug.
- **Section "Autograd versus reduced-free-energy gradients"** (lines 897-905): explicitly states the two objectives share critical points but differ off-equilibrium. The training loss differs from `F_red` by `τ Σ β* log(β*/π)`; this is documented, not hidden.
- **Numerical reports of F** in figures/tables: I did not find a quoted scalar value of "F" that was claimed to be `F_red` while computed as `<E>_β*`. The empirical mass-precision validation (line 1183) uses the isolated-agent limit `β_ij = 0` where the entropy term collapses to a constant (`τ Σ β log β = 0` is moot — the term simply doesn't appear), so reported values are unambiguous there.
- **Conditional-uniqueness appendix** (line 3635, `eq:envelope_app`): the dual-variable interpretation of `β` relies on the envelope theorem applied to `F_align` *with* the entropy term, so the appendix is consistent with Section 3.4 and Eq. 27.

Net: no numerical claim is corrupted. The risk is a reader who reads only Eq. 32/39 and concludes `F = <E>_β*`. They would be off by `τ Σ β* log(β*/π)` at any non-equilibrium `β`, and would underestimate `F` by `τ KL(β* || π) ≥ 0` even at equilibrium (`F_red = <E>_β* - τ Σ β* log(β*/π) = <E>_β* + τ H(β*) - τ Σ β* log π_j ≤ <E>_β*` for the uniform prior since `H(β*) ≥ 0`; for general π the sign depends on whether β* concentrates on high-π or low-π entries).

## Recommended fix

Use option **(C)** with a small additive note. Concretely: amend the boxed Eq. 39 with a short clause inside the equation environment, and flag the convention at Eq. 32 as well. Replacing the box with the full `<E>_β + entropy` form (option B) would force the reader to carry an extra term that the manuscript will *immediately* eliminate via the envelope theorem; replacing it with `-τ log Z` (option A) breaks the visual symmetry with the well-known `Σ β KL(...)` transformer attention loss that the manuscript wants to recover. The deliberate choice is option (C), and only the labeling needs improvement.

### Suggested LaTeX patch (Eq. 32, line ~802)

Add, immediately after the `where β*_ij(c)` clause:

```latex
. The expression above is the full (un-minimized, autograd-convention) free energy in
which $\beta_{ij}(c)$ and $\gamma_{ij}(c)$ are treated as variational parameters and the
softmax solutions \eqref{eq:softmax_attention_general} are substituted as their values;
the corresponding \emph{reduced} free energy obtained by also folding in the attention
entropy/prior term is given in \eqref{eq:free_energy_reduced}, and the two objectives
share critical points but differ off-equilibrium by $\tau\sum_j \beta^*_{ij}\log(\beta^*_{ij}/\pi_j)$.
```

### Suggested LaTeX patch (Eq. 39 box, line ~828)

Inside the `\boxed{ \begin{aligned} ... \end{aligned} }`, add as a final commented row:

```latex
% F here is the joint (autograd-convention) free energy with beta, gamma free;
% the reduced (partially minimized) form is in Eq. \eqref{eq:free_energy_reduced}.
```

and add to the prose just below the box (before "where `χ_i(c)`..."):

> This is the *joint* free energy treating `β_ij(c)` and `γ_ij(c)` as variational parameters; substituting their softmax minimizers into both the energy and entropy terms of `F_align` (Eq. `eq:mixture_energy_entropy`) yields the reduced free energy `F_red` of Eq. `eq:free_energy_reduced`, in which each alignment block becomes `-τ log Z_i`. The autograd-convention objective and the reduced free energy share critical points; their gradients agree at the optimum but differ off-equilibrium by `τ Σ_j β*_ij log(β*_ij / π_j)`.

These two edits make the boxed equation self-disambiguating without disturbing the reduced form already in Section 3.4.

### Optional stronger fix

If the editors prefer a single canonical box, the cleanest variant is:

```latex
\boxed{
\mathcal{F}_{\mathrm{red}}[\{q_i\},\{p_i\},\{s_i\},\{r_i\},\{\phi_i\}]
= \sum_i \int \chi_i \mathrm{KL}(q_i\|p_i)\,dc
 + \sum_i \int \chi_i \mathrm{KL}(s_i\|r_i)\,dc
 - \tau \sum_i \int \chi_i \log Z_i^{(\beta)}\,dc
 - \tau \sum_i \int \chi_i \log Z_i^{(\gamma)}\,dc
 - \sum_i \int \chi_i \mathbb{E}_{q_i}[\log p(o|k_i,m_i)]\,dc
}
```

with `Z_i^{(β)} = Σ_j π_j^{(β)} exp(-KL(q_i || Ω_ij q_j)/τ)` and analogously for γ. This is mathematically the cleanest object (it *is* the value of the partially minimized free energy), and it leaves the joint form to a non-boxed equation as a stepping stone.

## Bottom line

The core math is correct and the manuscript carries the disambiguation in Section 3.4. The defect is presentational: the boxed Eq. 39 looks like `F = <E>_β*` to a reader who stops at the box. A two-sentence amendment to the prose around Eq. 32 and Eq. 39, or a re-boxing in terms of `-τ log Z_i`, fully resolves the hazard. No retraction or numerical correction is required.
