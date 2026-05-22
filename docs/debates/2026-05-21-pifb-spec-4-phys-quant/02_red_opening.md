# Red Opening — pifb-spec-4-phys-quant

## Steelman (opposing position)

The four subsections (3051-3157) make their philosophical claims with explicit and matched disavowals — mass-Fisher is structural-not-empirical, the dimensionless-constants program imposes a symmetric success-failure criterion that forbids indefinite goalpost-shifting, structural realism is invoked rather than Kantian constitutive a priori, and the boxed list pins (i)-(iv) as postulated rather than derived — so the section is honest about what it does and does not establish and is therefore publication-ready.

## Position

The section contains two falsification-grade defects that the disavowals do not repair: (A) a direct internal contradiction at line 3076 about the relationship between δF=0 and the natural-gradient flow equations, and (B) a philosophy-of-science citation incoherence at line 3092 where "agnosticism among" Worrall, Ladyman & Ross, and Cassirer is asserted across positions that make *mutually incompatible* ontological commitments. Either defect is sufficient to refuse "rock-solid" status. Both are repairable; neither is repaired in the present draft.

## Evidence

### Defect A — δF=0 ⟶ natural-gradient flow is a category error AND contradicts the manuscript's own Discussion

Manuscript line 3076 (`Participatory_it_from_bit.tex`):

> "The Euler-Lagrange equations from $\delta \mathcal{F} = 0$ **become** our natural-gradient flow equations: Fisher-Rao natural gradient on the Gaussian sector $(\mu, \Sigma)$ and Lie-group natural gradient on the gauge-frame sector $G$..."

The natural-gradient flow is `dθ/dt = -G(θ)⁻¹ ∇F` — a *first-order dissipative ODE*. Steepest descent in the Fisher–Riemannian sense [Amari1998 §2-4]. It is the unique flow that is invariant under reparametrization of the statistical manifold; it is *not* the Euler-Lagrange equation of any action constructed from F alone. Hamilton's principle yields second-order Euler-Lagrange equations from `δ ∫ L dt = 0` where L has kinetic and potential pieces; the natural-gradient flow is first-order, with F playing the role of a *potential*, not a Lagrangian. There is no Lagrangian of the form `L(θ, θ̇) = ½ θ̇ᵀ G(θ) θ̇ − F(θ)` whose Euler-Lagrange equation reduces to `θ̇ = -G⁻¹∇F`; that EL equation would instead read `d/dt(G θ̇) − ½ θ̇ᵀ ∂G θ̇ + ∇F = 0`, which is second-order. The manuscript's own §sec:mass at line **1906** records this distinction in plain language:

> "We do *not* claim to derive the kinetic structure from $\mathcal{F}$ alone: no explicit Lagrangian $L = T - V$ is constructed from first principles, **no Euler-Lagrange equations are derived**, and the Newtonian-shaped equation of motion $M\ddot\mu = -\nabla V$ is a structural consequence of the kinetic-metric postulate rather than a derived dynamical law."

And the Discussion section at line **3172** states the same thing canon-style:

> "[the analogy] is not the derivation route by which standard active inference arrives at first-order natural-gradient descent, **which is an intrinsic Riemannian gradient flow on the statistical manifold in the sense of Amari (1998) §2, not the overdamped limit of a Newtonian SDE.**"

So three places in the manuscript disagree about whether the natural-gradient flow is an Euler-Lagrange equation: §3076 says **it is** ("become"); §1906 says **it is not** ("no Euler-Lagrange equations are derived"); §3172 says **it is not** ("an intrinsic Riemannian gradient flow ... not the overdamped limit"). Under the source-of-truth rule [Amari1998 §4] sides with §1906 and §3172. The §3076 claim is the outlier and is false as written: a first-order Riemannian gradient flow does not become the Euler-Lagrange equation of any standard variational principle in F.

Canon: [Amari1998 §2] — natural gradient is `g(θ)⁻¹ ∇L`, defined as steepest descent in the Fisher–Riemannian sense, with no claim of equivalence to Hamilton's principle. The "variational" character of variational *inference* refers to minimization over a function space (variational calculus on probability distributions), not Hamilton's principle on a configuration manifold.

### Defect B — "Agnostic among Worrall, Ladyman & Ross, and Cassirer" packs mutually incompatible positions

Manuscript line 3092:

> "...the structural-realism family — Cassirer's neo-Kantian variant, Worrall's epistemic structural realism [Worrall1989], and the ontic structural realism of Ladyman \& Ross [Ladyman2007] (**remaining agnostic among these variants**)..."

These three positions make *mutually exclusive ontological commitments*:

- **Worrall1989 (ESR):** the world contains unobservable objects with intrinsic natures; we have epistemic access only to the *relational structure* among them. Objects exist; only structure is knowable.
- **Ladyman & Ross 2007 (OSR, eliminative reading):** there are *no individuals or intrinsic natures*; structure is all that fundamentally exists. The motivation is precisely that quantum-mechanical permutation symmetries and spacetime-point identity puzzles eliminate the realist commitment to individual objects that ESR retains.
- **Cassirer (neo-Kantian):** a third, distinct position — structure constitutes the form of objects through the cognitive contribution of the knowing subject, with a transcendental rather than scientific-realist grounding.

The conflict between ESR and OSR is canonical in philosophy of science: ESR is committed to the existence of unobservable individuals (their intrinsic natures merely unknown), OSR (in its eliminative or "radical" form) denies that there are individuals to be ignorant about. "Remaining agnostic" between these is not a coherent epistemic stance — it is agnosticism between *p* and *not p* on what exists.

The manuscript's own substantive commitment at line 3090 — "The information geometry on principal bundles ... **constitutes** the noumenal realm. This structure exists independently but has no accessible content. Physical quantities ... constitute the phenomenal realm" — is closer to an *eliminative-OSR* reading (structure constitutes the noumenal, full stop) than to ESR (which preserves individuals behind the structure) and is incompatible with Cassirer's constitutive-a-priori reading (where the noumenal is in principle inaccessible because *we* constitute the phenomenal). Asserting agnosticism *while also* making the constitutive claim at 3090 is internally inconsistent: the constitutive claim has already selected an OSR-flavored variant.

Canon: Stanford Encyclopedia of Philosophy entry on Structural Realism (Ladyman, multiple editions) is the standard reference; the ESR/OSR distinction is the entry's organizing axis. Ladyman's own statement of OSR explicitly contrasts with Worrall's ESR on this point.

### Secondary observations (weaker but worth flagging)

- The "Cencov scale fixings" at 3084 is *coherent enough* — Cencov's theorem [Cencov1972] gives the Fisher metric up to a multiplicative scalar, and "scale fixing" is the natural language for picking that scalar. The defensive reading is sound; this is not a strike.
- Wheeler 1983 ("Law Without Law" in *Quantum Theory and Measurement*) does contain the "self-excited circuit" image. The citation at 3155 is correctly attributed.
- The boxed postulate (iv) at 3141 ("$T_x, T_y, T_z$ mutually trace-orthogonal") is correctly labeled as a 4D extension postulate, consistent with §sec:worked_signature (line 2818-2858) using only one generator $T = \mathrm{diag}(1,-1)$. No strike.
- The Pan-Agentic / Plural Time concession at 3149 ("Fisher arc length is Riemannian and positive-definite while proper time is Lorentzian, and the two run in opposite directions with motion") is internally consistent with §sec:fisher_arc_length at line 2609. No strike.
- The mass-Fisher disavowal at 3070 is consistent with the §sec:mass disavowal at 1904 ("no operationally independent measurement is reported in this manuscript"). The cross-reference is honest. No strike on the empirical-mass front.

## Falsification conditions

This Red position fails if:

1. The Euler-Lagrange claim at 3076 can be derived rigorously — i.e., there exists a Lagrangian `L[θ, θ̇]` constructed *only* from F (and possibly its Hessian/Fisher) whose Euler-Lagrange equation is exactly `θ̇ = -G⁻¹ ∇F`. The standard reference would be a derivation showing the natural-gradient flow as the EL equation of a *gradient-system Lagrangian* in the sense of Helmholtz / inverse problem of the calculus of variations. The standard literature does not produce such a Lagrangian for general (non-symplectic) gradient flows; if Blue produces one, Defect A is repaired.

2. Or if Blue produces a charitable reading under which "Euler-Lagrange equations from δF = 0" at 3076 means *the stationary conditions* of a variational *inference* problem (functional derivative of F over the space of distributions = 0), not the Euler-Lagrange equation of a *Hamilton's-principle* time integral. That reading is available — but then 3076 still mis-attributes by also calling these "extremize the action integral" in the immediately preceding sentence, conflating two distinct uses of "variational."

3. The structural-realism position at 3092 can be repaired by either (a) committing to one variant rather than agnosticism, or (b) carving out a region of philosophical space where the three positions agree (the *common-structural-content* claim). Blue must show that "agnosticism" is defensible across mutually-exclusive ontological positions, not that the three positions share *some* features.
