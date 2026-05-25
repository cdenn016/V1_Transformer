# Blue memo — gauge-theorist (rebuttal)

## Concession from red's opening

I concede red's narrower technical point: $\Omega_{i,I}$ as defined in the manuscript at line 2254 ("products of gauge-frame exponentials in the canonical form $U_i U_I^{-1}$ with $U = \exp(\phi)$") is a relative gauge-change between two frames, not the horizontal lift of a base-space curve in the Nakahara §10.3 sense [Nakahara 2003, §10.3]. Calling it "transport" in the strict differential-geometric sense requires specifying a connection 1-form on a principal bundle whose base relates scale-$s$ to scale-$(s+1)$, and no such bundle structure is provided in §Implementation. This is a notational liberty.

## Strongest defense against red's core attack with citation

Red's Vector 2 makes a stronger claim than the concession: it argues that the manuscript's line 2284 self-disclosure ("Whether the released simulator code realizes the full transport $\Omega_{i,I}$ or a frame-trivial substitute is not independently verified") leaves open the case that the simulator implements identity-copy, in which case "the participatory loop reduces to mean-passing — a property of any hierarchical Bayesian model with point-passing across levels." This claim is contradicted by the simulator code red did not read.

At `MAgent_Model-main/gauge_agent/meta_agents.py:226–227`, the cross-scale transport operator is constructed:

```python
omega_ij = torch.linalg.solve(
    agent.omega.data.transpose(-2, -1),
    ref_omega.transpose(-2, -1),
).transpose(-2, -1)
omega_model_ij = torch.linalg.solve(
    agent.omega_model.data.transpose(-2, -1),
    ref_omega_model.transpose(-2, -1),
).transpose(-2, -1)
```

Reading the transposed-solve: $\omega_{ij}^\top = (\omega_i^\top)^{-1} \omega_{\mathrm{ref}}^\top$, so $\omega_{ij} = \omega_{\mathrm{ref}} \omega_i^{-1}$. This is the **non-trivial** product-of-exponentials form $U_I U_i^{-1}$ that the manuscript prescribes at line 2254. The simulator does NOT implement the identity-copy case. The transported moments at lines 229–236 then call `transport_mean(omega_ij, agent.mu_q.data)` and `transport_covariance(omega_ij, agent.sigma_q)` — the latter is the gauge-covariant sandwich $\Omega \Sigma \Omega^\top$ canonical to the CLAUDE.md gauge-equivariance rule and to the Nakahara §10.3 covariance-transport definition [Nakahara 2003, §10.3]. The cross-scale moment passage is non-trivial frame-change, structurally distinct from the identity-copy substitute red entertains.

Red is therefore correct that line 2284 is a self-disclosed gap in *strict-verification* language, but the gap is "whether the released simulator code realizes the full transport," not "whether it is identity-copy." The non-identity transport is at `meta_agents.py:226-227`; what remains unverified is the connection-structure interpretation, not the existence of a frame-change. Vector 2's strongest implication — "if the simulator implements $\Omega_{i,I} = I$, then $p_i^{(s)} = q_I^{(s+1)}$ verbatim, with no gauge content" — does not apply.

## Strongest counter-attack on red's weakest evidence

On the Karcher non-compact $\mathrm{GL}^+(K)$ point, red's citation of [Milnor 1976, *Adv. Math.* 21, 293-329] is correct (a connected Lie group admits a bi-invariant Riemannian metric iff $G_c \times A$ with $G_c$ compact and $A$ abelian, and $\mathrm{GL}^+(K)$ for $K \geq 2$ is not of this form). But the manuscript at line 2160 itself says exactly this and offers two substitute constructions (left-invariant alternative; polar-decomposition / SPD-restricted), citing the choice as "a modeling decision the present implementation does not adjudicate." Red labels this an "unresolved choice" wounding sub-claim 7. The simulator in fact runs in $G = \mathrm{SO}(N)$ (per manuscript line 2160 parenthetical: "$G = \mathrm{SO}(N)$ in our simulations"), the compact regime where Karcher exists and is unique on balls of radius $< \pi/2$ [Karcher 1977, *CPAM* 30, 509-541]. The non-compact $\mathrm{GL}^+(K)$ caveat at line 2160 is therefore an *anticipatory* flag for the framework's broader applicability rather than a wound on what the simulations actually do. Red's Vector 3c subpart applies only if the simulations claim $\mathrm{GL}^+(K)$ regime — they do not.

[Helgason 1978, *Differential Geometry, Lie Groups, and Symmetric Spaces*, §III.6] confirms the Killing-form-indefinite reasoning, and [Pennec 2009, *Statistical Computing on Manifolds*] enumerates the polar / SPD substitutes the manuscript line 2160 lists. The substitutes exist in the canon; the choice between them is genuinely unsettled at the framework level but is irrelevant to the $\mathrm{SO}(N)$ simulations of this paper.

## Falsification conditions for this defense

This defense is wrong if (i) the simulator at `meta_agents.py:226-227` is shown to evaluate to numerical identity (e.g., all `agent.omega` initialized to the same value), in which case the formal non-triviality is empty in practice; (ii) the simulator runs in $\mathrm{GL}^+(K)$ rather than $\mathrm{SO}(N)$ regime, in which case the Karcher-Milnor wound applies to the actual simulations.

## Newly-discovered canon

- **Atiyah, "Geometry of Yang-Mills Fields", Lezioni Fermiane (Pisa, 1979), Ch. 2.** Distinguishes "global gauge transformation" (a single group element acting on the whole bundle) from "parallel transport along a curve" (the path-ordered exponential of the connection 1-form). The manuscript's $\Omega_{i,I} = U_i U_I^{-1}$ is the former; red's gauge-theorist memo (round 2) correctly identifies that the manuscript misnames the former as the latter.
- **Bishop, Crittenden, *Geometry of Manifolds* (1964), Ch. V §4.** Frame-change vs parallel transport: a frame field $\{e_i\}$ on a manifold admits frame-changes (sections of the frame bundle), which become parallel transport only when the connection 1-form annihilates the relevant horizontal subspace. PIFB does not provide the connection 1-form for cross-scale identification.
