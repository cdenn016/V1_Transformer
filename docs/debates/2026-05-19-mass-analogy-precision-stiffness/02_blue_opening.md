# Blue Opening — mass-analogy-precision-stiffness

## Steelman (opposing position)

A red-team attack worth taking seriously is the following. The section's central terminological move — calling the Hessian a "mass matrix" — is decorative rather than derived. The Hessian of $\mathcal{F}$ is by definition a stiffness (a coefficient of a quadratic form on small displacements about an equilibrium). Reusing the same matrix as the kinetic-metric coefficient is a postulate the manuscript itself flags but does not derive, and the kinetic postulate is what carries the empirical scaling $\omega^2 \propto m_{\text{eff}}^{-1}$ — not the second-variation analysis. Under that reading, the section's algebra (which is correct) is being used to license a name (which is not derived), and the title "Mass Analogy" promotes a postulated identification into a result.

This is the strongest available critique because it does not require finding an algebra bug — there are none in the boxed equations — and instead targets the framing.

## Position

C1 (mathematical correctness of the boxed equations) holds under direct symbolic verification. C2 (framing as a precision-induced stiffness/mass analogy with explicit kinetic-metric postulate, isolated/symmetric limit restriction, and asymmetric non-conservativity caveat) holds under the section's own disclaimers, which are unusually explicit by manuscript standards. The compound claim is defensible. The decorative-naming critique is real, the section concedes it textually, and the concession is load-bearing rather than perfunctory.

## Evidence

**C1 — algebraic verification (sympy, all six identities pass).**

I executed sympy verification on the load-bearing identities under generic GL(d) (non-orthogonal $\Omega$):

1. Eq. `eq:precision_transport` ($\tilde\Lambda_{q_k} = (\Omega\Sigma_k\Omega^T)^{-1} = \Omega^{-T}\Lambda_{q_k}\Omega^{-1}$): matches by direct matrix inversion. Result: `True`. This is a textbook identity for inverse of a sandwich product [PetersenPedersen MatrixCookbook §2.2; MagnusNeudecker 1999].

2. Algebraic step at line 1947 ($\tilde\Lambda_{q_k}\Omega = \Omega^{-T}\Lambda_{q_k}$): the manuscript's reasoning is direct substitution from Eq. `eq:precision_transport`, cancelling $\Omega^{-1}\Omega = I$. Sympy confirms `True` for a generic non-orthogonal $\Omega$ in 3D.

3. Boxed sender identity ($\Omega_{ji}^T \tilde\Lambda_{q_i}^{(j)} \Omega_{ji} = \Lambda_{q_i}$): manuscript line 1937 invokes this in the mean-sector diagonal block decomposition. Sympy confirms `True`.

4. Mean-sector receiver diagonal ($\partial^2 D_{KL}(q_i\|\tilde q_k)/\partial\mu_i^2 = \tilde\Lambda_{q_k}$): matches by direct differentiation of the Mahalanobis term $\tfrac12 d^\top \tilde\Lambda_{q_k} d$ with $d = \mu_i - \Omega\mu_k$. Sympy: `True`.

5. Mean-sector off-diagonal block (Eq. `eq:mass_mu_offdiagonal`): both forms $-\tilde\Lambda_{q_k}\Omega_{ik}$ and $-\Omega_{ik}^{-T}\Lambda_{q_k}$ match the autograd Hessian on a 2D test. Sympy: `True`.

6. Block-transpose symmetry $[M_{\mu\mu}]_{ik} = [M_{\mu\mu}]_{ki}^\top$: holds algebraically as an identity on the summed boxed expression, with no assumption on $\Omega_{ki}$ being related to $\Omega_{ik}$. Sympy: `True`. This matches Schwarz's theorem on mixed partials [Rudin 1976 Theorem 9.41] as cited in §`sec:mass_block_caveats` line 1958.

7. At-consensus collapse of the sender $\Sigma$-sector contribution (line 1981): I verified directly in 1D that $d^2 \mathrm{KL}(q_j\|\tilde q_i)/d\Sigma_i^2$, evaluated at $\Sigma_j = \Omega_{ji}\Sigma_i\Omega_{ji}^T$, collapses to $+\tfrac12\Lambda_{q_i}\otimes\Lambda_{q_i}$ (in 1D, $+\tfrac12/\sigma_i^2$). Result: `True`. The off-consensus form carries the $\Sigma_j$-coupling that the manuscript explicitly retains.

8. Cross block (Eq. `eq:cross_block`) vanishing at consensus: receiver contribution $\partial^2 D_{KL}(q_i\|\tilde q_k)/\partial\mu_i \partial\Sigma_k$ at $\mu_i = \tilde\mu_k$ gives zero (sympy verified, 2D, $\partial(\mu_i)_a\partial(\Sigma_k)_{bc}$ all components). The sender contribution $\partial^2 D_{KL}(q_k\|\tilde q_i)/\partial\mu_i \partial\Sigma_k$ is identically zero (not just at consensus), because KL(q_k‖tilde q_i) does not couple $\mu_i$ and $\Sigma_k$: $\mu_i$ enters only through $\tilde\mu_i = \Omega_{ki}\mu_i$ in the Mahalanobis term, while $\Sigma_k$ enters only as the "first slot" covariance. Sympy: all six $\partial^2/\partial(\mu_i)_a\partial(\Sigma_k)_{bc}$ entries return `0`. The full cross block therefore vanishes at consensus.

The Fisher–Rao information-geometric metric on Gaussians factors as $g_F = \Sigma^{-1} \oplus \tfrac12(\Sigma^{-1}\otimes\Sigma^{-1})$ on the mean-covariance split [Amari & Nagaoka 2000 §3.5; Calvo & Oller 1990]. The isolated-agent limit of Eqs. `eq:mass_mu_diagonal` and `eq:mass_sigma_diagonal` ($\beta = 0$, $\Lambda_o = 0$, $q = p$) reduces exactly to $\bar\Lambda_{p_i} \oplus \tfrac12 \Lambda_{q_i}\otimes\Lambda_{q_i}$. This is canon-grounding for the second-variation form: it is the second-order expansion of KL [`external_canon_math.md` "KL is the second-order expansion of the Fisher metric"].

**C2 — framing/motivation disclaimers (manuscript lines).**

The section's framing of the mass analogy is unusually explicit by manuscript standards. Concrete textual hedges, by line:

- **Line 1846 (opening):** "The construction is best read as a precision-induced configuration-space metric and a Newtonian-shaped harmonic analogy under that postulate, *not as a derivation of physical inertial mass from statistical precision*; physical mass requires a kinetic term, time parameterization, units, and a physical action principle that we do not establish here."

- **Line 1848 (Hessian-vs-mass paragraph):** "The Hessian of $\mathcal{F}$ that we derive in this section is therefore, *in the first instance, a stiffness on belief configuration space, not a mass*." Plus: "We do *not* claim to derive the kinetic structure from $\mathcal{F}$ alone: no explicit Lagrangian $L = T - V$ is constructed from first principles, no Euler--Lagrange equations are derived, and the Newtonian-shaped equation of motion $M\ddot\mu = -\nabla V$ is obtained empirically once the kinetic-metric postulate is in place."

- **Line 2028 (kinetic postulate, §`sec:velocity_quadratic`):** "This is a postulate, not a consequence of $\mathcal{F}$." And: "The match is consistent but contingent on the kinetic postulate, and would be vacated by any other choice of kinetic metric."

- **Line 2014 (within-framework interpretation):** "Throughout this subsection, the identification of the Hessian sector $[M_{\mu\mu}]$ with an 'effective mass' is interpretive within the framework rather than a derivation of physical inertial mass."

- **Line 2023 (dimensional and quantum caveats):** "The effective mass we obtain has dimensions inherited from $\Sigma^{-1}$ (dimensionless on the framework's representational space) rather than the kilogram dimension of physical mass." And the explicit non-extension to quantum: "We do not extend this analogy to quantum-mechanical scenarios: spatial delocalization of a quantum particle does not imply lower inertial mass in standard quantum mechanics."

- **Lines 1957–1960 (`sec:mass_block_caveats`):** the asymmetric-attention caveat distinguishes block-transpose symmetry of the Hessian (which holds by Schwarz, line 1958) from existence of a global conservative Hamiltonian under asymmetric attention (which does not, line 1960). The latter is explicitly identified as a Lyapunov / dissipative-metric reading rather than an inertia-tensor reading, consistent with the canonical literature [`external_canon_math.md`: "For asymmetric / non-symmetric Hessian-like objects ... the matrix is a Lyapunov / contraction metric rather than an inertia tensor", citing Khalil 2002 §4; Lohmiller & Slotine 1998].

The reciprocal-attention sufficient condition $\beta_{ik}=\beta_{ki}$ and $\Omega_{ik}\Omega_{ki} = I$ is stated as the condition for the Newtonian reading to hold; outside that condition the section explicitly says "the Newtonian reading is recovered cleanly in the symmetric-attention or isolated-agent limits where the empirical validation of Section~\ref{sec:mass} operates."

**Canon comparison.**

Standard mechanics defines mass as the coefficient of the kinetic form and stiffness as the Hessian of the potential at a minimum [Arnold 1989 §1, §17; Goldstein 2002 §1.2; Marsden & Ratiu 1999 §1.4]. The section reuses the Hessian as both, *under a separately stated kinetic-metric postulate*. The manuscript is explicit (line 1848) that this dual identification is what licenses the $\omega^2 \propto k/m$ harmonic-oscillator reading. The construction is therefore a labeled novel postulate, in the sense of `external_canon_math.md` §4: "If it differs and ... claims standard-equivalence without proof: novel claim. The manuscript must label it as novel and provide independent justification." The label is present (line 2028: "This is a postulate, not a consequence of $\mathcal{F}$"); the independent justification is the empirical scaling validated in the isolated-agent limit.

Standard FEP literature treats precision as a Bayesian weight on evidence, not as a kinetic coefficient [Friston 2010; `external_canon_inference.md` §1: "active-inference 'precision-weighting' idea has $\Sigma_p^{-1}$ as a Bayesian-prior weight on observation evidence, not as a kinetic-energy coefficient. Identifying precision with mass requires an additional postulate."]. The manuscript adds that postulate, names it, and restricts the Newtonian reading to the regime where it applies.

## Falsification conditions

The defense fails if any of the following is shown:

1. **Algebraic falsification (C1).** A symbolic or finite-difference computation shows any boxed equation (`eq:precision_transport`, `eq:mass_mu_diagonal`, `eq:mass_mu_offdiagonal`, `eq:mass_sigma_diagonal` at consensus, `eq:mass_sigma_offdiagonal`, `eq:cross_block`) is wrong under generic GL(d). I have verified the contrary by sympy in 1D and 2D; a counterexample in dimension $\geq 2$ with non-orthogonal $\Omega$ would overturn the defense.

2. **Hidden contribution falsification (C1).** Demonstration that a load-bearing contribution has been omitted. Two candidates: (a) the manuscript's envelope-theorem convention (line 1933) silently drops a $\partial\beta/\partial\mu$ term that is non-vanishing in the regime where the empirical scaling is validated; (b) the cross-block claim relies on a contribution that does not actually vanish at consensus. I have verified the cross block: the receiver contribution vanishes at $\mu_i = \tilde\mu_k$, and the sender contribution from $D_{KL}(q_k\|\tilde q_i)$ is identically zero (not just at consensus) because that KL does not couple $\mu_i$ and $\Sigma_k$ — only $\Sigma_i$ and $\mu_k$. The envelope-theorem convention is restricted by the manuscript to the regime where $\beta_{ij}$ is at its softmax equilibrium and the empirical validation is in the $\beta_{ij}=0$ isolated limit (line 1933), so the dropped term is identically zero in the validation regime.

3. **Disclaimer-insufficiency falsification (C2).** Demonstration that the section's hedges in lines 1846, 1848, 2014, 2023, 2028 are belied by load-bearing prose elsewhere that *does* claim a derivation of physical inertial mass, or by use of the precision-mass identification outside the symmetric/isolated limits where the manuscript restricts it. I have not located such prose in lines 1843–2040; if a red-team finding identifies one elsewhere in the manuscript that operates inside §`sec:mass`'s domain, the defense fails.

4. **Envelope-theorem misapplication falsification.** Demonstration that the envelope-theorem convention (line 1933, citing Milgrom & Segal 2002 Theorem 3 by canon) fails to license the boxed forms because $\beta_{ij}$ is not the actual maximizer/minimizer of a sub-problem whose envelope is $\mathcal{F}$. The softmax-attention is the stationary point of the F-functional augmented by the entropy term $\tau \beta_{ij}\log(\beta_{ij}/\pi_{ij})$ (see CLAUDE.md "Free energy" reference), so the envelope-theorem reading is licensed when that entropy term is present in $\mathcal{F}$. The boxed form thus inherits this licensing.

5. **Schwarz / block-transpose failure.** Demonstration that the manuscript's claim "block-transpose symmetry holds; conservative-Hamiltonian reading does not, under asymmetric attention" is internally inconsistent. The two are distinct: block-transpose holds by Schwarz on a scalar $\mathcal{F}$; conservativity fails when $\beta_{ij}$ is treated as instantaneous and non-symmetric inside the *dynamics*, not when the Hessian is computed. The manuscript's distinction (lines 1957–1960) is correct under [Khalil 2002 §4] canon.

## Concessions, stated as part of the opening

Three points the defense does not contest:

- The "mass" terminology is decorative rather than derived. The section concedes this textually (line 2014: "interpretive within the framework rather than a derivation"). The defense rests on the explicit concession, not on a claim that the terminology is earned by derivation.

- The kinetic-metric postulate at line 2028 is ad hoc in the sense that no first-principles justification is given. It is justified empirically by matching the observed scaling. The manuscript labels it as a postulate. The defense rests on the labeling, not on a claim that the postulate is derived.

- The asymmetric-attention case lacks a conservative-Hamiltonian reading. The manuscript states this (line 1960). The defense rests on that statement, not on a claim of Hamiltonian structure under asymmetric attention.

These are not weaknesses of the section — they are appropriately scoped disclaimers. If the red team argues that the section *promotes* the mass terminology beyond what the hedges allow, the load-bearing question is whether the hedges are robust to a careful reading. They are: every key identification (line 1846, 1848, 2014, 2023, 2028) is labeled, and the empirical scaling is restricted to the isolated-agent / symmetric-attention regime where the kinetic postulate is consistent.
