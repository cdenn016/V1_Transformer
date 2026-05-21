# Red Rebuttal Memo — geometer

Lens: differential geometry — SPD manifolds, parallel transport, sandwich product, exponential map, second-variation Hessians.

## One concession from blue's opening

Blue's defense items 1 and 2 are both granted in full.

Item 1: The sandwich rule at manuscript line 614, $\rho_{\mathrm{state}}(g) \cdot \mathcal{N}(\mu, \Sigma) = \mathcal{N}(g\mu, g\Sigma g^\top)$, is the canonical action on the associated bundle for a Gaussian fiber under a linear $\mathrm{GL}(K)$ representation per `[Nakahara2003 §10.3]`. The user's identification of $\Sigma$ as a (2,0)-tensor under the $\mathrm{GL}(K)$ action with $\Omega = \rho(g)$ matches the canonical form. No flag.

Item 2: The GL-correct precision transport at manuscript line 1894, $\tilde{\Lambda}_{q_k} = (\Omega_{ik}\Sigma_k\Omega_{ik}^\top)^{-1} = \Omega_{ik}^{-T}\Lambda_{q_k}\Omega_{ik}^{-1}$, with the textual caveat at line 1897 that the $\Omega \mapsto \Omega^{-T}$ substitution is required outside O(d), is the canonical dual-tensor transport rule for inverse bilinear forms. The manuscript pre-empts the standard pitfall (`external_canon_math.md:131`) with an in-line warning. No flag.

Item 2 also grants blue's falsification trigger (a): the secondary identities $\tilde\Lambda_k \Omega_{ik} = \Omega_{ik}\Lambda_k$ and $\Omega_{ki}^\top \tilde\Lambda_i = \Lambda_i \Omega_{ki}^\top$ at line 1897 hold only under O(d). A red search of §Theory past line 1897 for sandwich-misuse turned up no such instance: lines 1971–1990 explicitly use the GL-correct $\Omega^{-T}$ form (Eq.~\eqref{eq:mass_mu_offdiagonal} uses $-\beta_{ik}\Omega_{ik}^{-T}\Lambda_{q_k} - \beta_{ki}\Lambda_{q_i}\Omega_{ki}^{-1}$, not the O(d) reduction), and Eq.~\eqref{eq:mass_sigma_offdiagonal} at line 2019 retains the dual-tensor structure explicitly. The geometric primitives of §1.19 are GL-correct. No falsification trigger (a) found.

## Strongest attack on blue's core defense

Blue's defense item 1 grants the sandwich form. Blue's evidence item 2 grants the GL-correct precision transport. Blue's evidence items 5–6 grant the Goldstone caveat and the notation/analogy discipline. Blue's evidence items 1–2 are therefore correct concessions, *and red grants them in turn*.

The geometric attack that survives lies elsewhere: at the Hessian-as-mass identification (lines 2046–2069) and the kinetic-metric postulate (line 2064).

Manuscript line 2064 states explicitly:

> "Under this identification the harmonic-oscillator scaling $\omega^2 \propto k/m$ is a definitional consequence of the postulate rather than an independent dynamical scaling: when $k$ and $m$ are both equal to $M_{\mu\mu}$ by construction, $\omega^2$ reduces to a per-direction unit relation and the analogy is structural, not empirical."

This is the manuscript flagging its own load-bearing scaling claim as definitional rather than dynamical. Per `[Arnold1989]` GTM 60 Ch. 5 §22–25 (the manuscript's own citation at line 1882, 2064), small-oscillations theory requires the inertia tensor $T$ and the potential Hessian $V$ to be operationally independent positive-definite quadratic forms at the equilibrium configuration; the generalized eigenvalue problem $V v = \omega^2 T v$ has empirical content only when $T$ and $V$ are independent measurements. The manuscript uses the same matrix $M_{\mu\mu}$ for both $T$ and $V$ by postulate — see line 1882: "the present construction reuses the same matrix $M_{\mu\mu}$ for both roles and therefore does not supply such an independent test." The scaling $\omega^2 \propto m_{\text{eff}}^{-1}$ is then a tautology (under the postulate), not a result. Per `[GoldsteinPooleSafko2002 Ch. 6]` *Classical Mechanics* (3rd ed., Addison-Wesley), the same independence requirement is in every standard mechanics text.

Blue's defense item 6 says "analogies are labeled at lines 1409, 1497, 1518, 1876, 2058." This is true but evades the geometric strike: the analogy *labeling* does not repair the *missing kinetic-metric postulate*. The manuscript labels the mass analogy as analogy *and then deploys the harmonic-oscillator scaling as if it had empirical content* via the "rocks-are-stiff-because-precise" reading at lines 2057–2059. The Newtonian-shaped reading at line 2055 ("incoming relational mass ... outgoing recoil ... sensory mass grounds the agent in observations") writes the *language* of inertia but supplies no independent inertia tensor. The asymmetric-attention regime at line 1994 explicitly *abandons* the conservative-Hamiltonian reading, leaving the section without a global Lyapunov function either (the manuscript admits this is "open and is not addressed here").

The strike: **§1.19 (lines 1877–2069) is the section where blue's notation/analogy discipline of evidence item 6 most needs to hold and where it does not hold.** The TODO at line 1880 is the manuscript's own admission that the empirical test of the section's central scaling claim is not in this manuscript. Under the conjunctive operationalization, this fires sub-claim 6 (no unresolved gaps) by manuscript-line construction and sub-claim 4 (falsifiability) by Arnold-independence construction.

## Strongest defense against blue's strongest attack

Blue does not attack red's opening geometric strikes directly — the geometer memo grants the sandwich and dual-precision-transport defenses and shifts the strike to §1.19. The defense to strengthen against blue's "analogies are labeled" framing (evidence item 6) is the manuscript's *own* statement at line 2064 that the scaling is "a definitional consequence of the postulate rather than an independent dynamical scaling." A definitional consequence under a postulate is unfalsifiable as stated per `[Popper1959 Ch. 4 §15]`; labeling it analogy does not change this. `[MarsdenRatiu2002]` *Introduction to Mechanics and Symmetry* (Springer TAM 17) and `[Bloch2003]` *Nonholonomic Mechanics and Control* (Springer IAM 24) are the canonical references for what a "Newtonian reading" of a variational principle would have to display — none of which §1.19 exhibits.

## Newly-discovered canon

None beyond what the Phase-2 red harvest recorded. `[Arnold1989]`, `[GoldsteinPooleSafko2002]`, `[MarsdenRatiu2002]`, `[Bloch2003]` are all in `01b_extended_evidence.md`.
