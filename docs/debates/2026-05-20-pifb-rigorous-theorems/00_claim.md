# Claim — pifb-rigorous-theorems

**Mode:** math (with theory support permitted)
**Rounds:** 2
**Judge:** on
**Evidence scope:** Attention/Participatory_it_from_bit.tex lines 4592-4767 (the entire `\subsection{Rigorous Theorem-Level Statements}` appendix)
**Canon location:** C:/Users/chris and christine/Desktop/V13_Gauge_Transformer/.claude/agents/vfe-knowledge/

## Claim

The `\subsection{Rigorous Theorem-Level Statements}` appendix at `Attention/Participatory_it_from_bit.tex:4592-4767` provides the rigorous mathematical backbone the body subsection sec:meta_agent_rg claims to invoke: it states and proves Theorem thm:rg_pushforward (partition function and observable preservation), Theorem thm:rg_semigroup (discrete semigroup composition), Theorem thm:rg_covariance (gauge covariance of the coarse-graining map under SO(K) bi-invariant metric), Theorem thm:rg_exact_closure (augmented-class closure with explicit log-determinant correction), Proposition thm:rg_residual (schematic closure-residual decomposition into six terms with constants $C_1, \ldots, C_6$), Theorem thm:rg_residual_explicit (explicit dispersion and anharmonic constants $C_1, C_5$ under three specifications), and Theorem thm:rg_detector_retention (detector implies positive retention gain under Lipschitz hypothesis). The appendix is honest about its scope: compact $G = \mathrm{SO}(K)$ throughout with the noncompact $\mathrm{GL}^+(K)$ case explicitly flagged as requiring a gauge slice or Radon-Nikodym correction; finite-dimensional throughout with the continuum limit on $\mathcal{C}$ following by mesh refinement under regularity assumptions; the four remaining constants $C_2, C_3, C_4, C_6$ in Proposition thm:rg_residual explicitly registered as proposition-grade rather than theorem-grade; finite-size scaling claims explicitly relegated to "future-work program rather than a delivered result."

## User context

This is a separate debate on the rigorous-theorem appendix that grounds the body-level RG analysis at sec:meta_agent_rg. The user has separately requested a debate on `\section{Critical Open Problems and Future Directions}` — that is a separate debate, queued after this one.

## Sub-claims

1. **Pushforward/semigroup correctness sub-claim:** Theorems thm:rg_pushforward and thm:rg_semigroup are standard measure-theoretic results correctly stated and proved; their content is the elementary change-of-variables and composition rule for pushforward measures.

2. **Gauge covariance sub-claim:** Theorem thm:rg_covariance assumes (i) compact $G$ with locally unique Karcher means, (ii) gauge-invariant weights $w_i^I$, (iii) forward-KL (M-projection) barycenter for belief and model parents. Under these assumptions the proof via bi-invariance is correct.

3. **Augmented-class closure sub-claim:** Theorem thm:rg_exact_closure correctly identifies the Schur complement $A_{\mathrm{eff}} = A_{YY} - A_{Y\xi}A_{\xi\xi}^{-1}A_{\xi Y}$ and the canonical Wilsonian fluctuation-determinant correction $V(Y) = \tfrac{\tau}{2}\log\det'A_{\xi\xi}(Y)$ as the augmented-class form. The Remark at 4684-4686 honestly registers that the strict-Gaussian-KL-closure special case is "empty in the operational regime of this manuscript."

4. **Schematic residual bound sub-claim:** Proposition thm:rg_residual decomposes the closure residual into six structural terms (dispersion, transport spread, edge-marginal mismatch, off-diagonal Hessian, anharmonic remainder, non-Gaussian cumulants) with the $V_I^{3/2}$ exponent registered as $L^\infty$ Edgeworth-derived. The "stated as a proposition rather than a theorem" framing at 4718 is the correct epistemic register.

5. **Explicit dispersion and anharmonic constants sub-claim:** Theorem thm:rg_residual_explicit gives $C_1 = (\sqrt 2/12) \|F(q_I)\|^{3/2}_{\mathrm{op}}/m_I^{1/2}$ via Pinsker-Edgeworth and $C_5 = (1/8) \|T_4\|_{\mathrm{op}}/m_I^2 + (5/24)\|T_3\|^2_{\mathrm{op}}/m_I^3$ via integrated Wick coefficients. The 5/24 = 1/12 + 1/8 decomposition into sunset (1/12) and double-bubble (1/8) two-loop topologies is mathematically standard. The density-vs-integrated reconciliation paragraph at 4737 correctly notes the relation to the Hermite-evaluation coefficients 1/24 and 1/72 fails as an $L^\infty$ bound at radius $> 1.126$.

6. **Detector retention sub-claim:** Theorem thm:rg_detector_retention with the Lipschitz hypothesis and the bounded-exponential detector $\Gamma_I$ correctly establishes that $\Gamma_I > \Gamma_{\min}$ implies $\Delta_I > 0$ when condition eq:rg_app_detector_retention holds. The Remark at 4742-4744 admits the Lipschitz bound is "locally derivable under the closure-ansatz conditions (i)-(v)" — circular if not carefully read but the prose acknowledges this.

7. **Scope-and-honesty sub-claim:** The compact-group restriction at 4595, the finite-size scaling deferral at 4763-4764, the C_2/C_3/C_4/C_6 deferral at 4734, the augmented-vs-strict closure clarification at 4684-4686, and the explicit "noncompact case requires a gauge slice or regulator that the present manuscript does not provide" closing at 4767 are correctly stated.

Red attacks: that Theorem thm:rg_exact_closure's "augmented-class closure" framing overclaims (the log-det term itself needs renormalization at each step, so the "class" doesn't close in a fixed-point sense); that the gauge-covariance theorem's bi-invariant-metric assumption is inconsistent with the body's GL+(K) framework; that the Lipschitz bound in thm:rg_detector_retention is circular; that the Wick coefficients 1/8 and 5/24 may not be the load-bearing integrated-free-energy values (a prior debate produced these — verify); that the C_1 Pinsker-Edgeworth derivation has a subtle issue.

Blue defends: that all theorems are correctly stated and proved relative to the standard mathematical literature; that scope qualifications are appropriately registered; that the "augmented-class" framing is honest (the augmentation IS iterable — Wilsonian RG canonically uses fluctuation determinants at each step).

The judge may rule:
- All theorems sound (BLUE_WINS)
- Specific theorems need tightening of hypotheses or statements (RED_WINS_NARROW with surgical edits)
- The augmented-class closure framing needs more careful articulation of the iteration (RED_WINS_NARROW with a remark addition)
- Multiple small edits (REMAND)
