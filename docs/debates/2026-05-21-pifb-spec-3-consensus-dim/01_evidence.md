# Evidence Pack — pifb-spec-3-consensus-dim

## Manuscript references (read the actual TeX, not this summary)

Source file: `C:/Users/chris and christine/Desktop/V13_Gauge_Transformer/Attention/Participatory_it_from_bit.tex`

### §Collective Geometry and Gauge Invariance (lines 2905-2943)

- **2907-2917 (`Consensus Metrics`)** — Naive average `bar G = (1/N) sum w_i(c) G_{i, mu nu}(c)` depends on each agent's `phi_i` choice → violates gauge invariance. Section explicitly motivates the gauge-averaged construction.
- **2919-2937 (`sec:consensus_metric` — Gauge-Invariant Metric Construction)** — Eq. `<G_i>_{mu nu}(c) = integral_G dg G_{i,munu}(c; U_i -> U_i g)` with Haar measure replaces the abelian Lie-algebra shorthand `phi -> phi + xi`. **2928 — Substantive disclosure paragraph:**
  - For **constant** g: connection transforms `A -> g^-1 A g`; `tr(A_mu A_nu)` is already invariant by cyclicity → constant-g Haar average is "trivial or unnecessary". Does NOT rescue non-invariance of horizontal block.
  - For **local** `g(c)`: connection picks up Maurer-Cartan piece `A -> g^-1 A g + g^-1 dg`. Honest gauge-orbit average would have to integrate over all maps `g: C -> G`, infinite-dimensional functional integral requiring a gauge fixing or regulator.
  - Non-compact `SO(1,3) ⊂ GL(K,C)`: Haar measure is **infinite** even for constant g.
  - Status: "retain the construction below as a heuristic for what gauge-invariant content the horizontal block could be reduced to under a chosen regulator, but explicitly do not claim it produces a finite, regulator-free gauge-invariant metric. Whether a regulated version of the construction yields a useful gauge-invariant ontology is open and is the natural follow-up to the present manuscript."
- **2930-2937 — Consensus metric Eq. `consensus_metric`** = `sum_i w_i(c) <G_i>_{mu nu}(c)`. **Explicit conditional status:** the consensus metric "remains a heuristic target rather than a completed observable". Structural tier `G^(s)` load-bearing. Transfers verbatim from state fiber to model fiber.
- **2939-2943 (`Connection to Physical Gauge Invariance`)** — Hypothesis: gauge invariance in fundamental physics arises as consistency requirement for multi-agent consensus (not imposed on nature). Labeled "speculative" and "metaphysical interpretation rather than a derivation". May not be falsifiable.

### §Dimensional Structure and Observable Sectors (lines 2945-3024)

- **2948-2958 (`Eigenvalue Decomposition`)** — Standard spectral decomposition `G_i(c) = sum lambda_a (e_a tensor e_a)` with `lambda_1 >= ... >= lambda_n >= 0`.
- **2960-2970 (`Observable vs Subthreshold vs Internal`)** — Three sectors by eigenvalue threshold. For K=768 Gaussian: `dim(B) = K(K+3)/2 ≈ 296,064`. **2970 — "What lives where" paragraph:** The induced metric `G_i(c) = sigma_i^* g_B|_{q_i(c)}` is a tensor on the BASE manifold C, NOT on the fiber B. At each point c it is `n x n` with `n = dim(C)` and **at most n eigenvalues**. "Vast majority of dimensions" in internal sector refers to fiber directions NOT sampled by `d sigma_i`, NOT to eigen-directions of the base pullback. **This is a careful distinction — verify the surrounding prose stays consistent.**
- **2974-3004** — Observable `D_obs = {e_a : lambda_a > Lambda_obs}`; subthreshold `D_subthresh = {e_a : Lambda_subthresh < lambda_a <= Lambda_obs}`; internal `D_internal = {e_a : lambda_a <= Lambda_subthresh}`. For human agents conjectured `|D_obs| ≈ 4` (1 temporal + 3 spatial).
- **3006-3016** — Hierarchy `|D_obs| << |D_subthresh| << |D_internal|`, `lambda_obs >> lambda_subthresh >> lambda_internal ≈ 0`. "We may suppose..."
- **3018-3024 (`Hypothesized (3+1) Structure - Speculative`)** — `|D_obs| = 4` hypothesized for human cognitive agents. Three possible mechanisms (neural architecture, sensory modalities, evolutionary optimization). Explicit: "This remains a toy model demonstration that dimensional structure can, in principle, emerge from information geometry eigenvalue hierarchies. The framework does not explain why exactly three spatial dimensions rather than two or four, nor does it connect to fundamental physics explanations such as the anthropic principle or string theory compactification scenarios. Above all, it makes no quantitative predictions about measured spacetime properties that could be tested experimentally."

### §Observer-Dependent Reality (lines 3026-3049)

- **3029-3045 (`Multiple Phenomenal Realities from One Noumenal Substrate`)** — `G_i != G_j` for `i != j`. Different perceived geometries, both equally valid. Agents informationally coupled through attention `beta_ij(x) propto exp(-KL(q_i(x) || Omega_ij[q_j](x)))`. "Not solipsism — agents inhabit different phenomenal spaces while coupled through a shared noumenal substrate."
- **3047-3049 (`No View From Nowhere`)** — Noumenal `C` exists as mathematical substrate but possesses no intrinsic metric. Kantian phenomena/noumena identification forward-referenced to `sec:phenomenological_interpretation` (Debate 4 territory).

## Canon excerpts and external sources to verify

- **Haar measure on non-compact Lie groups** — non-compact connected Lie groups have left-invariant Haar measure unique up to scale, but the total volume is generally infinite. Reference: Folland *A Course in Abstract Harmonic Analysis*, or Knapp *Lie Groups Beyond an Introduction*. **The 2928 claim that "the Haar measure is infinite even for constant g" on non-compact SO(1,3) is standard — verify.**
- **Gauge orbit averaging in physics** — functional integrals over `g: C -> G` for local gauge groups require Faddeev-Popov gauge fixing or BRST quantization. Reference: Peskin & Schroeder *Introduction to QFT* §16.2, or Pokorski *Gauge Field Theories* §3.3. **The 2928 claim that "an honest gauge-orbit average over local g(c) ... is an infinite-dimensional functional integral over a space of gauge-group-valued fields and requires a gauge fixing or a regulator to be well-defined" is standard — verify against Peskin/Schroeder.**
- **Spectral theorem for symmetric tensors** — standard linear algebra, every symmetric `n x n` matrix has `n` real eigenvalues and an orthonormal eigenbasis. Verify 2950.
- **Base-vs-fiber dimension** — for a smooth section `sigma: C -> E_q` with fiber B and base C, the differential `d sigma_c: T_c C -> T_{sigma(c)} E_q` has rank at most `dim(C)`. The pullback metric `sigma^* g_E` is on `T_c C` with at most `dim(C)` non-zero eigenvalues. **Verify the 2970 careful distinction.**
- **Cosmological/observer principle** — "no view from nowhere" — see Nagel *The View from Nowhere* (1986) and discussions in Rovelli's relational QM (2004). The text does not cite Nagel; check whether it should.
- **Friston 2017 Graphical Brain** for the consensus-via-coupling reading at 2939-2943.

## What this evidence does NOT settle

1. Whether the framework's claim at 2939-2943 that gauge invariance in physics ARISES from cognitive consensus is genuinely a hypothesis (as labeled) or whether it slips into a stronger claim elsewhere.
2. Whether the `|D_obs| = 4` conjecture at 2980 silently inherits the postulates from §`sec:signature_resolution` (Debate 2) — the "1 temporal + 3 spatial" split is contingent on the imaginary-frame postulate and real-part projection still being in force.
3. Whether the "no view from nowhere" claim at 3047-3049 is consistent with the framework's later use of within-species consensus as a candidate objective geometry (cross-section ref).
4. Whether the Haar-averaging construction's non-compact regulator caveat at 2928 is correctly stated in light of the Pontryagin-Lefschetz duality for non-compact groups (advanced check).
5. Whether the base-vs-fiber dimension distinction at 2970 is maintained throughout the §Dimensional Structure subsection or whether language elsewhere conflates the two.
