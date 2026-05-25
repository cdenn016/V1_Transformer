# Memo — debate-expert-gauge-theorist — blue — opening — pifb-implementation-rock-solid

## Lens

Gauge theory — Lie groups, principal bundles, irreps, holonomy, equivariance, gauge fixing, Karcher mean on compact and non-compact Lie groups, Baker-Campbell-Hausdorff approximation.

## Steelman of the opposing position

A section that displays the Karcher-frame barycenter $U_I^* = \arg\min_U \sum_i w_i d_G(U, U_i)^2$ (Eq. 2156-2158) and immediately acknowledges that the formulation requires a bi-invariant Riemannian metric the gauge group $G = \mathrm{GL}^+(K)$ does not admit (line 2160) is exhibiting a load-bearing mathematical object whose definition is incompatible with the section's chosen gauge group, and the two enumerated substitutes (left-invariant alternative or polar-decomposition / SPD-restricted construction) both break gauge symmetry partially in a manner the section explicitly declines to adjudicate.

## My position (in service of blue)

The Karcher-frame caveat at line 2160 is a textbook-correct disclosure of the well-known obstruction on non-compact semisimple Lie groups (the Killing form on $\mathfrak{gl}(K, \mathbb{R})$ is indefinite [Nakahara2003 §5.5]), and the manuscript bounds its empirical claim to the compact subgroup $\mathrm{SO}(N)$ at line 2160 ("$G = \mathrm{SO}(N)$ in our simulations") where the Karcher mean does exist and is unique on convex normal balls of radius $< \pi/2$ [Karcher 1977]. The Lie-algebra-additive working form $\phi_I = \sum w_i \phi_i$ at line 2191 is explicitly labeled the first-order BCH approximation, with the higher-order error $\mathcal{O}(\|\phi_i\|^2)$ stated; this is the standard treatment, not a hidden assumption.

The covariance-transport identity $\tilde\Sigma_i = \Omega_{Ii}\Sigma_i\Omega_{Ii}^\top$ at line 2145 is the standard two-sided sandwich for (2,0)-tensors under the defining $\mathrm{GL}(K)$ action [Nakahara2003 §10.3]; the working formula at Eq. 2184 uses it correctly.

## Evidence

- **[Nakahara2003 §10.3]** (Geometry, Topology and Physics, 2nd ed., 2003): for a tensor of type $(k, \ell)$ on an associated vector bundle, parallel transport by $g \in G$ acts as $T \to \rho(g)^{\otimes k} \otimes \rho(g^{-1})^{*\otimes \ell} T$. For a (2,0)-tensor like the covariance in the user's framework, this is the two-sided sandwich $\Sigma \to \Omega \Sigma \Omega^\top$. The manuscript's transported covariance at line 2145 ($\tilde\Sigma_i = \Omega_{Ii}\Sigma_i\Omega_{Ii}^\top$) and the implementation formula at line 2184 both apply the sandwich, satisfying the textbook standard.
- **[Karcher 1977]** "Riemannian center of mass and mollifier smoothing," Comm. Pure Appl. Math. 30: 509-541. Existence-uniqueness of the Karcher mean requires a sufficiently restricted convex ball; for compact Lie groups with bi-invariant metric, balls of radius $< \mathrm{inj}(G)/2$ suffice. For $\mathrm{SO}(N)$ the injectivity radius is $\pi$, so balls of radius $< \pi/2$ suffice — exactly the manuscript's statement at line 2160.
- **[Hall, *Lie Groups, Lie Algebras, and Representations* (2nd ed., 2015), §5.3]**: BCH expansion $\log(e^X e^Y) = X + Y + \frac{1}{2}[X,Y] + \frac{1}{12}([X,[X,Y]] - [Y,[X,Y]]) + \ldots$. The truncation $\phi_I = \sum w_i \phi_i$ is exact when all $\phi_i$ commute (abelian $G$) and is accurate to $\mathcal{O}(\|\phi_i\|^2)$ otherwise, matching the manuscript's claim at line 2191.
- **No bi-invariant Riemannian metric on $\mathrm{GL}^+(K, \mathbb{R})$**: the Killing form on $\mathfrak{gl}(K, \mathbb{R})$ is given by $B(X, Y) = 2K \mathrm{tr}(XY) - 2\mathrm{tr}(X)\mathrm{tr}(Y)$ and is indefinite for $K \geq 2$ (the trace direction is null) [Helgason, *Differential Geometry, Lie Groups, and Symmetric Spaces* (1978), §III.6]. This corroborates the manuscript's line 2160 statement that "no bi-invariant Riemannian metric exists" for non-compact $\mathrm{GL}^+(K)$.

## Newly-discovered canon (for 01b_extended_evidence.md)

- **Pennec, "Statistical Computing on Manifolds: From Riemannian Geometry to Computational Anatomy," in *Emerging Trends in Visual Computing* (LNCS 5416, 2009): 347-386.** Standard reference on the Karcher mean for matrix Lie groups, including the polar-decomposition substitute for non-compact $\mathrm{GL}^+(K)$ that the manuscript at line 2160 acknowledges as a candidate. The substitute family that breaks bi-invariance is explicitly enumerated; the manuscript's caveat aligns with Pennec's treatment.
- **Bonnabel & Sepulchre, "Riemannian metric and geometric mean for positive semidefinite matrices of fixed rank," *SIAM J. Matrix Anal. Appl.* 31(3) (2009): 1055-1070.** SPD-restricted construction for matrix-valued Karcher means; this is the second substitute the manuscript at line 2160 enumerates ("polar-decomposition / SPD-restricted construction").
- **Moakher, "Means and averaging in the group of rotations," *SIAM J. Matrix Anal. Appl.* 24(1) (2002): 1-16.** For the compact $\mathrm{SO}(N)$ case the manuscript actually uses in simulations (line 2160), the existence and uniqueness of the Frechet/Karcher mean on convex normal balls is standard; Moakher gives explicit closed-form computations for $\mathrm{SO}(3)$ that match the simulator's regime.

## Falsification conditions

This gauge-theoretic defense is wrong if:

1. The covariance transport in the simulator code at `MAgent_Model-main/gauge_agent/meta_agents.py:217-238` does not actually apply the two-sided sandwich (the implementation-engineer expert is asked to verify); if the simulator does one-sided conjugation, sub-claim 6 takes a second wound beyond the consensus-detector mismatch.
2. The simulator gauge group is not actually $\mathrm{SO}(N)$ but secretly extends to $\mathrm{GL}^+(K)$ in some code path, in which case the manuscript's compact-group scoping at line 2160 is not honored and the Karcher caveat becomes load-bearing rather than discharged.
3. The first-order BCH approximation at line 2191 is invoked in a regime where constituent frames are not close, i.e., $\|\phi_i\|$ is not small; in that regime the $\mathcal{O}(\|\phi_i\|^2)$ error becomes load-bearing and the substitution is no longer accurate.

The strongest blue argument is conditional: the gauge-theoretic disclosures at lines 2160 and 2191 are textbook-correct and reviewer-grade *under* the compact-$\mathrm{SO}(N)$ scoping the section adopts. They do not defend any extension to non-compact $\mathrm{GL}^+(K)$ that the section does not claim.

## Confidence

HIGH on the gauge-theoretic disclosures being correct and discharged; MEDIUM on whether they suffice for sub-claim 4 (faithful labeling) without independent verification that the simulator stays in $\mathrm{SO}(N)$ — would shift to LOW if the implementation-engineer reports the simulator escapes the compact regime.
