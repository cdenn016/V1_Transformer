# External Canon — Free Energy Principle, Active Inference, Variational Inference

**Status:** source of truth for both agents. The user's manuscripts and codebase are evaluated *against* these standard treatments.

Citations resolve to `external_bibliography.md`.

**Citation hygiene.** Section numbers and equation labels (e.g., `[Friston2010 Eq. 2.2]`) are best-effort pointers; verify before citing the specific number in a finding. When in doubt, cite only the source tag (`[Friston2010]`).

---

## 1. Variational free energy — standard form

The standard variational free energy [Friston2010, ParrPezzuloFriston2022 Ch. 2] for a generative model `p(o, s)` over observations `o` and latent states `s`, with recognition distribution `q(s)`, is
```
F[q]  =  E_q[ log q(s) - log p(o, s) ]
      =  KL( q(s) ‖ p(s | o) )  -  log p(o)
      =  E_q[ -log p(o | s) ]  +  KL( q(s) ‖ p(s) )
```

Three equivalent forms. Each reveals a different structural fact:
- **Form 1** (free energy = expected log-ratio): the original variational definition.
- **Form 2** (free energy = KL to true posterior − log evidence): shows F is an upper bound on −log p(o). Minimizing F over q minimizes the KL gap to the true posterior, since −log p(o) does not depend on q.
- **Form 3** (free energy = accuracy + complexity): the active-inference decomposition. The first term is expected inaccuracy (negative log-likelihood under q); the second is the complexity (departure of q from prior).

**Negative free energy = ELBO** in the variational inference / VAE literature [BleiKuckelbirgJordan2017, KingmaWelling2014]: `ELBO = -F = E_q[log p(o, s)] - E_q[log q(s)]`. Maximizing ELBO ≡ minimizing F.

### What is *not* in this standard form

- **Multi-agent coupling terms** of the form `Σ_ij β_ij KL(q_i ‖ Ω_ij q_j)`. These are user-introduced. The standard FEP is single-agent (or hierarchical with a single ancestral generative model). Multi-agent generalizations exist in the FEP literature (e.g., [Friston2017Graphical] for graphical brain, [Ramstead2020] for variational ecology), but the *specific* coupling form via gauge transport `Ω_ij` is a user construction, not the field standard. **The agent should label this as a novel construction requiring its own justification.**
- **Attention entropy term `τ β log(β/π)`**. The user's claim that this is required for softmax to be stationary is *internally consistent* (it's the standard maximum-entropy regularization), but it is not part of standard single-agent FEP. It belongs to the entropy-regularized optimal-transport / soft-assignment literature (Sinkhorn divergences, etc.) and to the user's specific functional. Standard if presented as a Lagrangian for the soft-assignment problem; novel if claimed to follow from FEP alone.

## 2. Active inference — standard form

[ParrPezzuloFriston2022, FristonEtAl2017]

For an agent choosing policies π, the **expected free energy** of a policy is
```
G(π)  =  E_{q(s,o|π)} [ log q(s|π) - log p(o, s|π) ]
      =  -E_{q(o|π)}[ log p(o|C) ]                                # pragmatic value (goal-directed)
         + E_{q(o|π)}[ KL( q(s|o,π) ‖ q(s|π) ) ]                   # epistemic value (information gain)
```
or equivalently
```
G(π)  =  E_q[ -log p(o|C) ]  -  E_{q(o|π)}[ KL(q(s|o,π) ‖ q(s|π)) ]
       =  expected cost (under preferred outcomes C)
        - expected information gain
```
Policy selection: `q(π) ∝ exp(-G(π))`.

The user's `expected_free_energy.py` / `efe.py` modules implement this; agent should verify the decomposition is the standard one.

**Pitfall:** sign conventions differ across papers. Some write G as energy-to-minimize, others as utility-to-maximize. Verify the user's sign convention is internally consistent.

## 3. Hierarchical / nested formulations

[Friston2017Graphical, ParrPezzuloFriston2022 Ch. 9]

For a hierarchical model `p(o, s₁, s₂, ..., s_L)` with `s_ℓ` the latent at level ℓ, the recognition distribution factorizes as `q(s₁, ..., s_L) = Π_ℓ q(s_ℓ)` (mean-field) or with more structure. F decomposes additively across levels under mean-field. Cross-level couplings come from the generative model `p(s_ℓ | s_{ℓ+1})`.

The user's multi-layer cascade (where the previous layer's posterior `μ_q` becomes the next layer's prior `μ_p`) is **not standard mean-field across the hierarchy**. It is a deterministic point-passing scheme that loses the variational uncertainty about `s_ℓ` when passed to ℓ+1. Standard variational hierarchical inference passes the full posterior `q(s_ℓ)` to inform `q(s_{ℓ+1})`. **The agent should flag this:** the user's scheme is a specific approximation (point estimate at each level), not the full variational hierarchical scheme.

## 4. KL bounds, mode-seeking vs mean-seeking

[BleiKuckelbirgJordan2017]

- `KL(q ‖ p)` (forward / "inclusive" KL): zero-forcing — q must be zero wherever p is zero. **Used in variational inference (ELBO).** Tends to be mode-seeking (q hugs one mode of p).
- `KL(p ‖ q)` (reverse / "exclusive" KL): mean-seeking — q must cover the support of p. Used in EP and some other methods.

The user's framework uses `KL(q ‖ Ω q')` — forward KL with the transported neighbor as the "prior." This is the variational direction. If a manuscript switches to `KL(Ω q' ‖ q)` for any equation, that is a different inference problem; flag and ask for justification.

## 5. EM algorithm and the IFT for fixed-point inference

### Standard EM
[DempsterLairdRubin1977 — classic; see also any modern reference]

E-step: compute `q(s) = p(s | o, θ)` (or its variational approximation `q(s) = argmin_q F[q; θ]`).

M-step: update `θ` to maximize `E_{q(s)}[log p(o, s | θ)]`.

In standard EM, the E-step is *complete* (or at least driven to a high-quality approximation) before the M-step uses gradients. Variational EM relaxes this — the E-step does a few iterations of variational updates.

### Implicit Function Theorem for fixed-point methods
[BaiKolterKoltun2019]

If the E-step is run to a fixed point `s* = T_θ(s*)` for an operator `T_θ`, the M-step gradient through the fixed point is computed via the IFT:
```
∂s*/∂θ = (I - ∂T_θ/∂s |_{s*})⁻¹ ∂T_θ/∂θ |_{s*}
```
This requires solving a linear system at the fixed point (or approximating the inverse via Neumann series). **A single backprop through one iteration of the E-step is *amortized* inference, not IFT.** The two have different bias/variance properties.

The user's `em_mode='ift_phi'` claims IFT. The auditor must verify the implementation actually solves the implicit linear system, not just backprops through one fixed-point iteration. If it's the latter, the label is misleading — flag.

## 6. Mean-field, structured, amortized

[BleiKuckelbirgJordan2017]

Three standard q-families:
- **Mean-field:** `q(s) = Π_i q_i(s_i)`. Factorizes; loses correlations.
- **Structured:** `q(s) = q(s_1) q(s_2 | s_1) ...` or similar. Keeps some correlations.
- **Amortized:** `q(s | o) = q_φ(s | o)` parameterized by a function (e.g., neural net) of o. Standard in VAEs.

The user's per-token Gaussian factorization `q(s) = Π_i N(s_i; μ_i, Σ_i)` is mean-field across tokens. Cross-token coupling enters only through the F functional (the attention/coupling terms), not through correlations in q. Standard pitfall: writing `q(s_1, s_2, ..., s_N)` and silently assuming factorization without stating it. Verify the user is explicit.

## 7. Markov blankets, NESS, the "physics-y" formulation

[Friston2010 §3, Ramstead2020]

Modern FEP statements ground free energy minimization in non-equilibrium steady-state (NESS) systems with Markov blankets. This is the most physics-flavored version of FEP. It is *contested* in the literature — e.g., debates around whether the Markov-blanket / NESS argument is tautological or has empirical content (van Es 2021, Aguilera et al. 2022, etc.). When a manuscript invokes this framing, the agent should not treat NESS-FEP as uncontested — it's an active research area.

## 8. Where variational inference and Bayesian inference diverge

Standard variational inference *approximates* Bayesian posteriors; it is not Bayesian inference proper. The approximation gap is the KL between q and the true posterior (Form 2 above). VAEs and friends inherit this gap. The user's framework, insofar as it uses ELBO/F-minimization with a parameterized q, is variational and inherits the gap. Claims of "exact Bayesian inference" should be flagged.

## 9. Predictive coding correspondence

[Bogacz2017, Millidge2021]

Predictive coding minimizes prediction errors at each cortical level. Under specific assumptions (Gaussian, linear hierarchical model), it implements gradient descent on F. The correspondence is exact under those assumptions; outside them, predictive coding is one of several variational algorithms.

When a manuscript claims attention "is" predictive coding, the agent should verify which assumptions are invoked. Bare equivalence claims are usually overstated.

## 10. Pitfalls the agents must check for

1. **Form-1 vs Form-2 vs Form-3 conflation.** All three are equivalent algebraically, but a derivation that uses one form's terms but quotes another form's interpretation is sloppy.
2. **Sign convention.** `F` vs `-F` vs `ELBO`. Verify consistency throughout.
3. **"FEP implies X."** FEP is a variational principle; specific implementations follow from specific generative-model choices. Claims that FEP alone implies an architectural choice (attention, layer-norm, dropout) require the explicit generative model that connects them.
4. **EM vs IFT vs amortized.** Different gradient profiles; cannot be interchanged.
5. **Hierarchical mean-field vs point-passing.** Passing a posterior *mean* (not the full posterior) between levels is a deterministic approximation, not variational inference proper.
6. **Single-agent FEP extended to multi-agent.** The user's coupled-agent F is a generalization. The standard FEP literature does not contain this specific functional with gauge-transport-coupled KL terms. Standard FEP literature with multi-agent extensions (e.g., Ramstead variational ecology) uses different couplings. Flag the user's coupling as novel.
7. **"Exact Bayesian."** Variational ≠ Bayesian. Claims of exactness are usually wrong.
8. **NESS-FEP as uncontested.** It's contested; treat as such.
9. **Predictive coding equivalence.** Holds under specific Gaussian/linear assumptions only.
10. **"Attention is variational inference."** Claim requires the explicit generative model + the explicit recognition q for which softmax-attention is the variational update. The user's framework provides this (mixture-of-sources generative model); other frameworks (kernel methods, Hopfield networks) provide different generative interpretations. None are uniquely "the" interpretation; the user's is one of several.

---

## How to use this file

When a finding involves FEP / VI / active inference:
- Cite the standard form (Form 1, 2, or 3 above) and verify the user's form reduces to it under the claimed limit.
- Cite [Friston2010] for the canonical statement, [BleiKuckelbirgJordan2017] for the VI framing, [ParrPezzuloFriston2022] for the active-inference textbook treatment.
- For multi-agent extensions, label the user's coupling as a novel construction (not in standard FEP) and require independent justification.
- For IFT claims, cite [BaiKolterKoltun2019] and require the implicit linear system to be solved, not just backprop through one iteration.
