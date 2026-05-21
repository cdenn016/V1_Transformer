# Action — vfe-module-purity-for-pifb

**From verdict:** RED_WINS

## Recommended action

The /vfe module fails the user's stipulated falsifiability test on **one** specific PIFB construction: the state-dependent prior precision Eq:state_dependent_alpha (`Attention/Participatory_it_from_bit.tex:1308-1313`).

PIFB derives a per-agent **scalar** `α_i* = c₀ / (b₀ + D_KL(q_i ‖ p_i))` with a scalar log-barrier `R(α_i) = b₀·α_i - c₀·log α_i`. `transformer/vfe/e_step.py:50-66` implements a per-K-dimension `α_k = c₀ / (b₀ + KL_k)` with `raw_c0`, `raw_b0` of shape `(K,)`, and the module docstring itself states "This is a stronger generalisation than the published scalar form and is not currently derived in the manuscript." For K > 1 the two forms are algebraically inequivalent, and no `VFEConfig` flag recovers the scalar form.

Two non-mutually-exclusive remediations close the gap:

1. **Code-side scalar α toggle.** Add a `VFEConfig` flag (e.g. `bayesian_alpha_scalar: bool = False`) that, when True, constrains `raw_c0` and `raw_b0` to shape `(1,)` and routes `get_bayesian_alpha` to return `c₀ / (b₀ + Σ_k KL_k)` instead of the per-dim form. This restores the canonical PIFB:1311 construction as a reachable configuration. The per-dim form remains the default; the toggle opts into the manuscript-derived form.

2. **Manuscript-side derivation of per-dim form.** Derive the per-dimension Gamma-Normal conjugacy and per-dimension log-barrier in `Attention/Participatory_it_from_bit.tex` so the implemented per-K form has manuscript support and the "not derived in the manuscript" caveat at `e_step.py:62-66` can be retired. This would change the manuscript's claim under evaluation, not the code.

Either remediation alone closes the decisive gap.

## Follow-up debates (if any)

Two threads surfaced during the debate that the verdict explicitly set aside on jurisdictional grounds. Both are real questions whose resolution depends on a reading the user must supply:

### Follow-up debate 1 — single-configuration vs per-construction reading of the existential

The verdict adopted the per-construction reading because the falsification clause's "under any toggle setting" phrasing licenses different toggles for different PIFB constructions. Under the alternative single-configuration reading ("one VFEConfig realizes the whole framework"), Red's mutual-exclusion attack lands as a second independent falsifier:

- `transformer/vfe/config.py:566-573` (omega_direct branch): `gauge_parameterization='omega_direct'` requires `diagonal_covariance=True`.
- `transformer/vfe/config.py:491-495` (exact full-cov decode branch): `exact_full_cov_decode=True` requires `diagonal_covariance=False`.

The canonical group-level retraction (PIFB:2566-2570, Amari 1998 §3.4) and the canonical (0,2)-tensor sandwich decode (Nakahara 2003 §10.3, PIFB:1619-1626) are therefore not simultaneously reachable in a single `VFEConfig`. If the user intended the single-configuration reading, the verdict should reopen on this point.

Suggested invocation:

```
/red-blue-debate "There exists a single VFEConfig instance that simultaneously realizes (a) the canonical group-level Lie-group natural-gradient retraction Ω^{t+1} = Ω^t · exp(-η·∇̃F) (PIFB:2566-2570) and (b) the exact (0,2)-tensor sandwich covariance transport / decode Σ → Ω Σ Ω^T (PIFB:1619-1626, Nakahara 2003 §10.3) without diagonal approximation." mode=implementation
```

### Follow-up debate 2 — cross-layer φ-handoff prescription

Red's rebuttal argued PIFB:1545 prescribes φ evolution and `stack.py:145` hard-wires the embedding-value handoff (`new_prior_phi = initial_priors.phi`) with no toggle. The verdict ruled the PIFB:1537 "may evolve" phrasing is modal (permissive) rather than prescriptive, and that posterior φ does cascade via `beliefs.phi` at `stack.py:142-144`. The judgment was that this gap is not load-bearing relative to the per-dim α gap that decided the debate.

If the user reads "may evolve" as prescriptive (rather than permissive), a separate debate is warranted:

```
/red-blue-debate "PIFB §1502-1551 (timescale separation) and §1535-1545 (gauge frame evolution) prescribe — not permit — that cross-layer prior φ in the L-block transformer carry the posterior φ from the previous layer, and transformer/vfe/stack.py:145's unilateral embedding-value handoff with no toggle is a deviation from that prescription." mode=implementation
```

### Follow-up debate 3 — Wilson observable / α-homotopy parameter scope

Red attacked the absence of the Wilson observable construction (PIFB:eq:wilson_observable around line 847) and the homotopy parameter α at PIFB:871. The verdict ruled this construction is in the framework's broader "Discrete Regime II" extension at PIFB:824-880, not in the language-modeling core, and therefore out of scope for this debate per the user's "language-modeling aspects only" restriction.

If the user disagrees and considers the edge-relaxed cocycle construction part of the language-modeling core (it underpins `transformer/vfe/non_flat.py`, which is reachable via `use_non_flat_transport=True`), a focused debate is warranted:

```
/red-blue-debate "The non-flat transport path in transformer/vfe/non_flat.py implements the edge-relaxed cocycle Ω_ij = exp(φ_i·G)·exp(δ_ij·G)·exp(-φ_j·G) (PIFB:eq:edge_relaxed_omega) but does not realize the Wilson observable (PIFB:eq:wilson_observable) or homotopy parameter α (PIFB:871); these are required for the construction's stated language-modeling use, not just the cross-scale extensions." mode=implementation
```
