# Evidence Pack — pifb-spec-1-pullback

## Manuscript references (read the actual TeX, not this summary)

Source file: `C:/Users/chris and christine/Desktop/V13_Gauge_Transformer/Attention/Participatory_it_from_bit.tex`

### §Time as Information Flow (lines 2584-2611)

- **2584-2592** — Bit-counting time: `Delta tau_i = Delta I_i / (1 bit)` with `Delta I_i = KL(q_i^new || q_i^old)`. Explicit dimensional disclaimer (tau dimensionless, no SI seconds). Citations: Wheeler 1990, Lloyd 2002, Rovelli 2004, Page-Wootters 1983, Connes-Rovelli 1994, Jacobson 1995, Van Raamsdonk 2010.
- **2594-2598** — Minimum-time-from-minimum-information speculation. Explicit "highly speculative" labeling.
- **2600-2611 (`sec:fisher_arc_length`)** — Fisher arc length `Delta tau = integral sqrt(g_B(qdot, qdot)) dtau`. Explicit non-identification with relativistic proper time (Riemannian positive-definite vs indefinite Lorentzian; runs opposite direction with motion).

### §Natural Gradient Dynamics on Statistical Manifolds (lines 2613-2655)

- **2615-2617** — Block-Fisher form `~∇_mu F = Sigma (∇_mu F)`, `~∇_Sigma F = 2 Sigma (∇_Sigma F) Sigma`. Cross-reference to App. Fisher Gaussian.
- **2619-2645** — Canonical group-level retraction `U^{t+1} = U^t exp(-eta ~∇_phi F)` (Eq. `gauge_group_retraction`). Killing-form preconditioner discussion; pullback metric Eq. `pullback_metric` with `Psi(z) = (e^z - 1)/z`. Cites companion paper `Dennis2025trans`.
- **2647-2651** — Chart-coordinate truncation `dphi/dt = -eta ~∇_phi F + O(||eta ~∇||^2)` with BCH correction; explicit acknowledgement that the implementation is the first-order truncation, exact only in the abelian sector.
- **2653-2655** — Gauge covariance of updates under `U_i -> U_i g`, `Sigma_i -> rho(g) Sigma_i rho(g)^T`. Product manifold M = (R^K x S+_K x G)^N with G = SO(3) in simulations.

### §It From Bit: The Pullback Construction (lines 2657-2662)

- **2660** — Pullback of Fisher-Rao from fiber to base manifold via smooth sections. Within-agent gauge fixing flagged; the within-species consensus framing is forward-referenced to `sec:consensus_metric`.
- **2662** — "We emphasize this is a toy model demonstration that spacetime-like geometry can emerge from information, not a claim that this is how physical spacetime actually arises... We acknowledge that the foundation remains incomplete." Explicit honesty disclaimer.

### §The Pullback Mechanism: From Information to Geometry (lines 2664-2771)

- **2670-2676** — Fisher-Rao metric on Gaussian fiber: `g_B(dq, dq) = dmu^T Sigma^-1 dmu + (1/2) tr(Sigma^-1 dSigma Sigma^-1 dSigma)`.
- **2680-2701** — Bundle metric on associated bundle E_q = P x_G B_state. Horizontal-vertical decomposition `T_{(c,q)} E_q = H + V`. Bundle metric Eq. `bundle_metric` = `g_C^tw(pi_* X_H, pi_* Y_H) + g_B(X_V, Y_V)`. Frame-twist Eq. `horizontal_metric` = `kappa(A^(i)_mu, A^(i)_nu)`. Piecewise convention: `kappa(A,B) = -tr(AB)` for compact real forms (`so(N)`, `su(N)` -- Riemannian); `kappa(A,B) = +tr(AB)` for `gl(K,C)` or non-compact forms (indefinite). The "tw" label is deliberately distinguished from Yang-Mills `tr(F F)`.
- **2703-2720** — Pullback `G_i^(q) = (sigma_i^(q))* g_E` evaluated as Eq. `induced_metric_full` = `kappa(A_mu, A_nu) + E_q[(∇_mu log q)(∇_nu log q)]`.
- **2722-2723 — Gauge-invariance disclosure** (paragraph "Gauge-invariance disclosure for the horizontal block"). Under local `g(c)`: `A -> g^-1 A g + g^-1 dg`; `kappa(A_mu, A_nu)` acquires Maurer-Cartan cross terms. Explicit non-invariance acknowledgement; F_{mu nu} = 0 in Regime I (pure-gauge connection), so `tr(F F)` supplies no escape hatch. Routed to consensus metric instead.
- **2725-2736** — Three-tier parallel pullbacks G_i^(q), G_i^(p), G_i^(s), G_i^(r). The four tensors "coexistent rather than alternative".
- **2740-2753 (`sec:three_tiers`)** — Three Tiers of Induced Geometry: epistemic G^(q) (perceptual timescale), expectational G^(p) (inherits via cross-scale shadow `p_i = Omega_{i,I}[q_I^{(s+1)}]` per `cross_scale_shadow`), structural G^(s) on model fiber B_model (quasi-static). Identification of "perceived space" with the **structural tier G^(s)** rather than expectational or epistemic. Kant/Wheeler/Clark2016/Seth2021/Hoffman2019/Friston2017 cited.
- **2755-2771** — Gaussian R^2 worked example. Conformal-metric reduction `G^(q)_munu = (1/sigma^2)(∂_mu mu_i)(∂_nu mu_i)` when `Sigma = sigma^2 I` constant.

## Canon excerpts (read the full canon files for the relevant claim)

Canon files at `C:/Users/chris and christine/Desktop/V13_Gauge_Transformer/.claude/agents/vfe-knowledge/`:
- `external_canon_math.md` — Nakahara, Amari, do Carmo, Lee on bundles and Fisher-Rao
- `external_canon_inference.md` — Friston, Beal, Bishop on VFE and natural gradient
- `external_bibliography.md` — short tags (`[Amari1998]`, `[Friston2010]`, `[FristonEtAl2017]`, `[Nakahara2003]`, etc.)

Key canon items relevant here:
- **[Amari1998]** Natural gradient theorem: `~∇F = G^-1 ∇F` where G is Fisher information. For Gaussians, `~∇_mu = Sigma ∇_mu`, `~∇_Sigma = 2 Sigma (∇_Sigma) Sigma`. **This is what 2615-2617 invokes — verify the form is canonical.**
- **[AmariNagaoka2000]** §2.2 / §3 — Fisher-Rao as the unique invariant Riemannian metric on a statistical manifold (Cencov's theorem). **This is the basis for the pullback construction.**
- **[Nakahara2003]** §10.1-10.3 — Principal bundles, associated bundles, connection one-form, horizontal-vertical decomposition. **Compare to 2682-2701 — does the bundle metric decomposition match the standard?**
- **[Cencov1972]** — Fisher information's uniqueness under sufficient statistics. **Bears on whether the pullback construction "carries" the Fisher uniqueness up to C.**
- **[Wheeler1990]** Primary source: "It from bit" — Wheeler proposes physical reality derives from information. **Citation at 2660 invokes Wheeler's original formulation.**
- **[Lloyd2002]** Primary source: "Computational capacity of the universe" — Lloyd's claim about universal computation bounding evolution.
- **[Rovelli2004]** *Quantum Gravity* — relational interpretation; time only as relation between systems.
- **[PageWootters1983]** Phys Rev D 27, 2885 — clock subsystem entangled with rest of universe yields apparent time.
- **[Friston2017]** Hierarchical active inference: structural learning at parameter level vs state inference. **Supports the structural-tier G^(s) interpretation at 2749-2751.**
- **[Hoffman2019]** *The Case Against Reality* — interface theory of perception. Cited at 2751.

## What this evidence does NOT settle

1. Whether the bundle-metric construction at 2686-2701 is correctly identified as a fiber-respecting metric on E_q (as opposed to merely an L^2 pullback of densities). The agents should verify this against Nakahara §10.3 directly.
2. Whether the three-tier identification of "perceived space" with G^(s) (line 2749) is canonically defensible from Friston 2017 + Hoffman 2019, or whether it goes beyond what those sources support.
3. Whether the BCH error term `O(||eta ~∇||^2)` at line 2649 is sharp; whether the abelian-sector exactness claim is correctly stated.
4. Whether the Killing-form proportionality claim at 2633 (`(K-2) tr(XY)` for so(K), K > 2) is correctly tracked under the +tr vs -tr convention.
5. Whether any citation at 2592 (the Wheeler/Lloyd/Rovelli/Page-Wootters/Connes-Rovelli/Jacobson/Van Raamsdonk list) misrepresents the cited author's position.
