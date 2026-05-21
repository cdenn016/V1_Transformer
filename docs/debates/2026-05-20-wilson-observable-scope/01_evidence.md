# Evidence Pack — wilson-observable-scope

## Code references

- `transformer/vfe/non_flat.py` — exists in the /vfe module. Implements `VFENonFlatConnection` and `compute_pairwise_omega_with_delta`. Per `config.py:677-705`, gated by `use_non_flat_transport=True`. Forces `use_autograd_mu_sigma=True` (warning); requires `diagonal_covariance=True` and `rope_full_gauge='off'`.
- `transformer/vfe/e_step.py:419-440` — non-flat E-step branch: builds δ_ij from `(μ_i, μ_j)` via a bilinear connection module (`VFENonFlatConnection`), composes `Ω_ij = exp(φ_i·G) · exp(δ_ij·G) · exp(-φ_j·G)`.
- `transformer/vfe/e_step.py:859-868` (`_iter_nonflat` call site) — pairwise-Ω with non-trivial δ active in the inner loop.
- **Grep result** for `cocycle_relaxation|holonomy_penalty|wilson|Wilson` in `transformer/vfe/`:
  ```
  No files found
  ```
  The construction PIFB:874 names `cocycle_relaxation`, PIFB:868 names `holonomy_penalty`, and PIFB:858 names `Wilson observable` are all absent from `/vfe`.
- **Grep result** in the wider codebase (`transformer/`): the identifiers do appear in `transformer/core/connection.py`, `transformer/core/transport_ops.py`, `transformer/analysis/holonomy.py`, `transformer/analysis/holonomy_metrics.py`, `transformer/visualization/holonomy_plots.py`, `transformer/pure_vfe/gauge.py`, `transformer/pure_vfe/inference.py`, etc. The constructions exist in OTHER parts of the repo but NOT in `/vfe`. The user's directive in `00_claim.md` of the main debate restricts the canonical "pure path" to `transformer/vfe/`.

## Manuscript references

- `Attention/Participatory_it_from_bit.tex:824` — PIFB explicitly labels §"Gauge Field Strength" as Regime II content: "Under Regime I, the connection inherits the pure-gauge form `A = U⁻¹dU`, the Maurer-Cartan identity forces `F_μν = 0` identically ..."
- `Attention/Participatory_it_from_bit.tex:826` — "None of this is dynamically active in the Regime I implementation, but the construction is retained because the framework's gravitational and signature-related extrapolations require it."
- `Attention/Participatory_it_from_bit.tex:828-880` — §"Discrete Regime II via an Edge-Relaxed Cocycle". The edge-relaxed cocycle Eq:edge_relaxed_omega (line 836-837): `Ω_ij = U_i exp(δ_ij · G) U_j^{-1}`.
- `Attention/Participatory_it_from_bit.tex:847-867` — Wilson holonomy Eq:discrete_holonomy and Wilson observable Eq:wilson_observable; Wilson holonomy-penalty regularizer Eq:wilson_action.
- `Attention/Participatory_it_from_bit.tex:868` — "The squared Frobenius variant `Σ_{ijk} ‖H_{ijk} - I‖_F²` is equivalent up to bounded reparameterization and is the form actually implemented as the optional `holonomy_penalty` regularizer." **Direct manuscript claim about the implementation.**
- `Attention/Participatory_it_from_bit.tex:870-874` — homotopy parameter α: "The implementation exposes α as the `cocycle_relaxation` configuration parameter; the default α = 0 together with δ_ij ≡ 0 recovers the vanishing-holonomy theorem." **Second direct manuscript claim about the implementation.**
- `Attention/Participatory_it_from_bit.tex:876` — DAG / cycle interpretation: in autoregressive (causal) attention, the attention DAG has no closed loops and the holonomy is purely formal; in bidirectional attention, the Wilson observable acquires direct dynamical content. **The cocycle holonomy is THE clearest experimental test of the gauge-curvature linguistic conjecture per PIFB:876.**
- `Attention/Participatory_it_from_bit.tex:880` — "In the Regime I implementation realized in this manuscript, the connection is pure-gauge and curvature vanishes identically ... the holonomy diagnostics above are degenerate in that limit." Re-affirms Regime I trivializes Wilson.

## Canon excerpts (external sources of truth)

- **Wilson 1974** (Confinement of quarks, original Wilson loop / link variable formulation). **Kogut-Susskind 1975** (Hamiltonian formulation of Wilson lattice gauge theory). **Creutz 1983** §5 (Quarks, gluons and lattices). Wilson observable is the canonical gauge-invariant observable of a lattice gauge theory; the cocycle relaxation parameter α is the Regge-Wilson interpolation parameter between flat (α=0) and curved (α=1) limits.
- **PIFB:838** itself cites WilsonConfinement1974, KogutSusskind1975, Creutz1983 as the canonical source for Eq:edge_relaxed_omega.

## What this evidence does NOT settle

1. Whether the Wilson construction is "language-modeling core" or "Regime-II extension." PIFB:824, PIFB:826, PIFB:880 explicitly label Regime II as the gravitational/signature extensions and Regime I as the language-modeling implementation. But PIFB:876 and PIFB:880 also say the cocycle holonomy is the clearest test of the *gauge-curvature linguistic conjecture* — which IS a language-modeling claim. The manuscript is ambiguous about whether Regime II is part of the language-modeling core or solely a geometric extension.
2. Whether `non_flat.py` is genuinely a Regime II realization or whether δ_ij ≠ 0 there breaks the Regime I pure-gauge condition silently. PIFB:870-874 says α=0 + δ_ij≡0 recovers Regime I; `non_flat.py` initializes the connection at zero (per `e_step.py` documentation: "All gates start at zero — fresh model with the flag on is bitwise-equivalent to the flag-off path at step 0"), so at init the path IS Regime I; under training, δ_ij can drift away from zero and become a non-trivial Regime II connection.
3. The two **direct manuscript claims of implementation** at PIFB:868 (`holonomy_penalty`) and PIFB:874 (`cocycle_relaxation`) are *false relative to /vfe* (grep returns nothing). They are *true relative to the wider transformer/ tree* (the identifiers exist in `transformer/core/connection.py` and `transformer/analysis/holonomy_metrics.py`). This is a manuscript-vs-code consistency gap: PIFB describes the wider codebase, not the /vfe pure path the user has nominated as canonical.
