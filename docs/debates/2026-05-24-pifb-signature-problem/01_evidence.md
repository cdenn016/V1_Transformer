# Evidence Pack — pifb-signature-problem (Debate 2)

Neutral fact pack. Coordinators/experts read the manuscript and canon directly; verify the math with sympy.

## Manuscript references (Attention/Participatory_it_from_bit.tex)

- `:2822–2829` — sector split: gauge group enlarged to $\mathrm{GL}(K,\mathbb{C})$ on the connection sector ONLY; belief transport restricted to $\mathrm{GL}(K,\mathbb{R})$; claim that complex $\Omega$ gives non-Hermitian $\Omega\Sigma\Omega^\top$ and complex/negative Gaussian KL (numerically checked at $K=2,3$ per the text); real Fisher-Rao / KL≥0 / $\mathcal{F}\ge0$ live on the real sector. Regime I: $A_\mu=U^{-1}\partial_\mu U$, $F_{\mu\nu}\equiv0$; signature mechanism via $\mathrm{tr}(A_\mu A_\nu)$, gauge-noninvariant, NOT a YM invariant.
- `:2831–2833` — diagnosis: induced $G^{(q)},G^{(p)}$ positive semidefinite (score outer products); observed spacetime is Lorentzian $(-,+,+,+)$.
- `:2835–2837` — compact $\mathrm{SO}(N)$ preserves positive-definiteness (similarity transform preserves eigenvalues); real $\mathrm{GL}(K,\mathbb{R})$ still positive-definite by Sylvester's law (non-compactness necessary but not sufficient).
- `:2839–2847` — three consequences; complexification & Lorentz group; vector rep of $\mathrm{SO}^+(1,3)$ in $\mathrm{GL}(4,\mathbb{R})\subset\mathrm{GL}(4,\mathbb{C})$; $\mathrm{SL}(2,\mathbb{C})\cong\mathrm{Spin}^+(1,3)\subset\mathrm{GL}(2,\mathbb{C})$; Wick rotation between real forms of $\mathrm{SO}(4,\mathbb{C})$; imaginary frame component flips metric sign $G^{\mathrm{tw}}_{\tau\tau}=-(\partial_\tau\psi)^2\mathrm{tr}(T^2)<0$.
- `:2849–2861` — three postulates (non-compact real forms; complexified frames on real Gaussians; subgroup restriction to $\mathrm{SO}(1,3)$); explicit statement each step is well-defined but the DYNAMICAL selection is unresolved.
- `:2863–2902` (`sec:worked_signature`) — the $\mathrm{GL}(2,\mathbb{C})$ worked example. $T=\mathrm{diag}(1,-1)\in\mathfrak{sl}(2,\mathbb{R})$ (non-compact, $\mathrm{tr}(T^2)=2$); single-generator collapse $T_\tau=T_x=T$; imaginary temporal postulate $\phi=i\psi_\tau T + \psi_x T$ (Eq. complex_gauge_frame); separable ansatz; $A_\tau=i(\partial_\tau\psi_\tau)T$, $A_x=(\partial_x\psi_x)T$; $G_{\tau\tau}=-2(\partial\psi)^2$, $G_{xx}=+2(\partial\psi)^2$, $G_{\tau x}\in i\mathbb{R}$; rank-1 complex ($\det=0$) → $\mathrm{Re}(\cdot)$ → rank-2 real Lorentzian (Eq. lorentzian_metric, :2889); explicit "derivation gap" flag for the real-part projection; conformal-class statement (signature intrinsic, scale not).
- `:2894–2902` — $\mathrm{O}(1,1)/\mathrm{SO}^+(1,1)$ local frame group, $\Lambda^\top\eta\Lambda=\eta$ verified; $\mathrm{SO}^+(1,3)$ for 4D; $1+3$ vs $2+2$ not dynamically distinguished; four explicit caveats.
- `:2904–2907` — cross-section regulator obstruction (gauge-noninvariant, infinite Haar for non-compact); central open question = whether free energy selects imaginary $\phi_\tau$ and $1{+}3$.
- `:2909–2932` (`sec:causal_cone_route`) — alternative route: finite max epistemic speed $c_\mathcal{I}$ + positive-definite spatial $h$ → $g=-c_\mathcal{I}^2d\tau^2+h_{ab}dx^adx^b$ (Eq. causal_cone_metric), Lorentzian by Sylvester; conformal-class ambiguity; dimension not selected; first-order natural-gradient flow is parabolic → infinite signal speed → tension; three possible fixes (telegraph limit, hyperbolic dynamics, architectural constraint), none realized.
- `:2934–2952` — temporal direction from belief trajectories; Gram-Schmidt singles out a direction but not a sign; the minus sign in Eq. lor_belief_metric is by ansatz; same postulates as worked example; alternative real $\mathfrak{gl}(K,\mathbb{R})$ mixed-generator route also gives indefinite signature ($+\mathrm{tr}$ convention; Killing form on $\mathfrak{gl}(K,\mathbb{R})$ indefinite).

## Canon excerpts and external facts to verify (experts: WebFetch)

- Sylvester's law of inertia (signature invariant under congruence) — any linear-algebra text; the causal-cone signature count and the "$\mathrm{GL}(\mathbb{R})$ preserves positive-definiteness" claim both rest on it.
- Killing form of $\mathfrak{gl}(K,\mathbb{C})$ / $\mathfrak{sl}(2,\mathbb{R})$ signature; compact vs split real forms; that $-\mathrm{tr}(AB)$ is the positive-definite form on $\mathfrak{so}(N)$ and $\mathrm{tr}(AB)$ is indefinite on $\mathfrak{gl}$ — Knapp *Lie Groups Beyond an Introduction*; Hall *Lie Groups, Lie Algebras, and Representations*.
- $\mathrm{SL}(2,\mathbb{C})\cong\mathrm{Spin}^+(1,3)$ double cover; vector rep of $\mathrm{SO}^+(1,3)$ — standard QFT/group theory (Weinberg Vol. 1; Hall).
- Wick rotation as relating real forms of a complexified group — standard.
- Parabolic (first-order/heat-type) vs hyperbolic (wave-type) PDE signal speed — any PDE text; bears on the causal-cone first-order-dynamics tension.
- `external_canon_math.md` for Fisher-Rao positive-definiteness and the KL form.

## What this evidence does NOT settle

- Whether the real-part projection ($\mathrm{Re}(G_{\mu\nu})$, a rank-changing operation the manuscript admits has "no physical principle mandating it") is a benign existence-demonstration step or so unmotivated that even "structurally compatible with Lorentzian signature" overstates it.
- Whether confining complexification to the connection while the fiber stays real-positive-definite yields a signature genuinely connected to the framework's information content, or a free-floating $\mathrm{tr}(A_\mu A_\nu)$ sign game decoupled from the statistics.
- Whether the two routes' disclosed gap-lists are COMPLETE (any undisclosed assumption?), e.g. the separability ansatz suppressing cross-derivative terms, the single-generator collapse, or the global-vs-local isometry distinction.
- Whether the causal-cone route's "inner-product type not Finsler-norm" assumption is doing hidden work to force a quadratic (metric) rather than a general convex cone.
