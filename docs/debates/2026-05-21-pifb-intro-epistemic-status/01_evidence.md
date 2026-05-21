# Evidence Pack — pifb-intro-epistemic-status

## Manuscript references — Introduction (claim under evaluation)

- `Attention/Participatory_it_from_bit.tex:121-131` — §1.6 Epistemic Status (three-level framing):
  - **Level 1 (Implemented and Empirically Probed):** "Standard scaled dot-product attention is recovered as a zero-dimensional isotropic-Gaussian limit of the KL-consensus construction up to a separately introduced learned bilinear compatibility $M$ and the standard normalization and bias assumptions … multi-seed embedding-dimension sweep on WikiText-103 (Section~\ref{sec:scaling_validation}) giving $b = -1.049$ (95\% bootstrap CI $[-1.103, -0.998]$), $R^2 \approx 0.9998$ on per-$K$ seed means with the floor parameter $c \approx 61$. The bootstrap CI brackets $-1$ at its upper edge while a nested $F$-test rejects $b = -1$ at $\alpha = 0.05$ ($F(1,8) = 9.73$, $p = 0.014$); the two finite-sample criteria disagree at $N = 11$ axis points, and we report $b \approx -1.05$ as the point estimate."
  - **Level 2 (Mathematical Implementation):** "The Ouroboros Tower simulation reported in Section~\ref{sec:results} runs the fast subsystem only ($\gamma_{ij} = 0$, slow subsystem frozen) on a single seed; in that operational regime the fast-channel agents form clusters whose threshold-detected aggregations propagate updates back to constituents through the cross-scale prior shadow. Multi-seed reproducibility, the slow-channel coupling, and release of the simulator code are deferred to follow-up work".
  - **Level 3 (Speculative Physical Interpretation):** "The natural $\mathrm{GL}(K, \mathbb{R})$ gauge symmetry on the real Gaussian fiber, together with a complexification $\mathrm{GL}(K, \mathbb{C})$ of the connection sector only, makes Lorentzian-signature subgroups available on the horizontal frame-twist block of the metric, and Section~\ref{sec:signature_resolution} exhibits a 2D worked example in which an indefinite bilinear form is obtained after additionally postulating an imaginary frame component along a designated temporal direction and a real-part projection."

- `Attention/Participatory_it_from_bit.tex:133-156` — §1.7 Scope and Limitations (the matching bounding statements). Categorical claims:
  - "Implemented and empirically probed": 0D recovery; b ≈ -1.05 fit.
  - "Demonstrated in working code (single illustrative run)": Ouroboros Tower meta-agent simulations, "conditional on the threshold-based consensus detector of Section~\ref{sec:meta_agent_threshold}".
  - "Worked example only": Lorentzian-signature via $\mathrm{GL}(K,\mathbb{C})$ frame-twist route AND via causal-cone route (both with disjoint postulate sets, both existence statements not derivations).
  - "No quantitative physics predictions."
  - "No quantum extension."
  - "Conceptual reframing, not derivation" for consciousness section.
  - "Scaling-study scope" disclaimer on PPL ≈ 73 at K=120 vs. published baselines PPL ~ 18–25.

## Manuscript references — referenced sections (for cross-checking the classification)

- `Attention/Participatory_it_from_bit.tex:1598+` (§sec:transformers) — read the actual derivation of 0D attention recovery; check that the "up to bilinear M + normalization + bias" qualifiers in §1.6 Level 1 match what the section actually establishes.

- `Attention/Participatory_it_from_bit.tex:2511-2529` (§sec:scaling_validation) — full multi-seed scaling fit:
  - Fitted parameters: $b = -1.049$ (95% bootstrap CI $[-1.103, -0.998]$), $c = 61.17$ ([59.01, 63.16]), $a = 1805.55$ ([1598.56, 2063.99]), $R^2 \approx 0.9998$ on per-K seed-mean PPL.
  - Two-criterion disagreement: bootstrap CI brackets $-1$ at upper edge; $F(1,8) = 9.73$, $p = 0.014$ rejects $b = -1$ at $\alpha = 0.05$.
  - Achieved test PPL ≈ 73 at K=120 vs. published WikiText-103 multi-layer baselines PPL ~ 18–25.
  - For this architecture $N$ is exactly linear in $K$ across the sweep ($N/K \approx 6.53 \times 10^5$); $b_K = b_N \approx -1.05$ on a parameter class restricted overwhelmingly to embedding parameters.
  - Direct comparison to Chinchilla cross-entropy exponent ≈ -0.34 (Hoffmann 2022) is "mis-framed in the natural direction".

- `Attention/Participatory_it_from_bit.tex:2725-2810` (§sec:signature_resolution and §sec:worked_signature) — read the actual Lorentzian construction:
  - Sector split: complex gauge frame on connection sector only; real Gaussian belief fiber preserved.
  - Worked example: 2D base manifold, $T = \mathrm{diag}(1,-1)$ as non-compact generator, separable ansatz $\psi_\tau(\tau)$ and $\psi_x(x)$, postulated imaginary temporal component $\phi(\tau,x) = i\psi_\tau T + \psi_x T$, real-part projection $G^{\mathrm{Lor}}_{\mu\nu} := \mathrm{Re}(G_{\mu\nu})$.
  - Explicit caveats in the section: "structural existence proof, not a dynamical derivation"; "the assignment of an imaginary component along $\tau$ … is a postulate, and the projection to the real part of $G_{\mu\nu}$ is a further postulate; neither follows from variational free-energy minimization"; "not a direct Wick analog … a Wick-like continuation in the Lie algebra plus an additional real-projection step that has no Wick counterpart"; "the construction does not currently distinguish 1+3 from 2+2 on dynamical grounds".

- `Attention/Participatory_it_from_bit.tex:2812-2835` (§sec:causal_cone_route) — second, complementary route to Lorentzian signature using finite-speed epistemic causality. Independent postulate set. Has "Tension with the framework's first-order dynamics" subsection (line 2832): "Naive continuum limits of such flows are parabolic and yield infinite signal speed, so the causal-cone route does not apply directly to the current implementation".

- `Attention/Participatory_it_from_bit.tex:2073-2086+` (§sec:participatory) and §sec:meta_agent_emergence (2277+) and §sec:meta_agent_threshold (2133+) — read what the Ouroboros Tower simulation actually does and whether the §1.6 Level 2 description ("single seed", "fast subsystem only", "slow subsystem frozen", "threshold-detected aggregations") is faithful.

## Canon references for evaluating the classification

- Vaswani et al. 2017 §3.2.1 — canonical scaled dot-product attention $\mathrm{softmax}(QK^T/\sqrt{d_k})V$, with learned $W_Q, W_K, W_V$ projections. The §1.6 Level 1 claim that the manuscript's 0D-limit recovery is "standard scaled dot-product attention" up to "a separately introduced learned bilinear compatibility $M$" must be checked: in Vaswani, $M$ is implicitly $K^{-1} W_K^T W_Q$ on the embedded inputs, i.e., a learned bilinear; verify whether the manuscript's "$M$" is the same object or a strictly weaker / stronger structure.
- Hoffmann et al. 2022 (Chinchilla scaling) — cross-entropy scaling exponent ≈ -0.34 for compute-optimal training; reported on a parameter class spanning $W_Q, W_K, W_V, W_O$ and MLP weights (not embedding-only). The §1.6 / §sec:scaling_validation claim that this is "mis-framed in the natural direction" must be checked.
- Bishop 2006 §10 (variational inference) for the standard treatment that level-1 "Implemented" should mean.
- Standard $F$-test reference (Fisher / textbook) — at $N=11$, $F(1,8) = 9.73, p = 0.014$ is the canonical test for "is the restricted model $\mathrm{PPL} = a/K + c$ adequate against the unrestricted three-parameter fit". A bootstrap CI is a different test (covers parameter location, not nested fits). The two answering different questions is itself a methodological point.

## What this evidence does NOT settle

- Whether the manuscript's actual scope-and-limitations disclaimers in §1.7 are *complete* — there may be in-scope claims elsewhere that should appear in §1.7 but do not (e.g., the consensus-metric regulator gap, the pan-agentic ontology applicability gap, the cross-language pre-trained baseline absence).
- Whether the "up to a separately introduced learned bilinear compatibility $M$" qualifier in §1.6 Level 1 is honest given that *the* substantive content of attention (the learned query-key inner product) is exactly what $M$ supplies; if $M$ carries all the learning, calling the rest a "recovery" of attention is rhetorically generous.
- Whether the "single seed" Ouroboros run framing at §1.6 Level 2 is honest given that the Section~\ref{sec:results} reference may show single-seed results being discussed in non-conditional terms in the body.
- Whether placing the entire Lorentzian-signature program at Level 3 is honest, given that §1.7 calls it "Worked example only" and §sec:signature_resolution further admits the construction is not a Wick analog, is dependent on a real-part projection without dynamical justification, does not select 1+3 vs. 2+2, and (in the causal-cone variant) is in tension with the framework's actual first-order dynamics.
