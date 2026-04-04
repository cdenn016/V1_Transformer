"""
Derivation: Second-Order Meta-Agent Dynamics from the Ouroboros Feedback Loop
=============================================================================

Shows that two coupled FIRST-ORDER ODEs (fast belief update + slow prior update)
produce an effective SECOND-ORDER equation for the meta-agent belief.

The inertial mass M_eff = 1/(epsilon * eta * Lambda_m) arises from the
sluggishness of the prior in tracking the belief -- NOT from the Hessian
of the free energy (which is the spring constant K, not the mass).

Result:
    M_eff * d^2μ_α/dt^2 + Γ_eff * dμ_α/dt + V'(μ_α) = 0

where:
    M_eff   = 1 / (ε η Λ_m)
    Γ_eff   = (ε η Λ_m + η α λ_p) / (ε η Λ_m)
    V'(μ_α) = gradient of meta-level potential

Author: Robert C. Dennis (framework), derivation worked with Claude
Date: 2026-03-31
"""

import sympy as sp


def derive_two_agent():
    """
    Two-agent Ouroboros system -> effective second-order meta-agent dynamics.

    SETUP
    -----
    Two scalar agents (μ₁, μ₂) form a meta-agent μ_α = (μ₁ + μ₂)/2.
    The Ouroboros loop sets each constituent's prior to the meta-agent belief,
    but the prior updates SLOWLY (rate ε η) while beliefs update FAST (rate η).

    Constituent first-order dynamics:
        dμ₁/dt = -η [α λ_p (μ₁ - μ̄) + β λ (μ₁ - μ₂) + ½ V'(μ_α)]
        dμ₂/dt = -η [α λ_p (μ₂ - μ̄) + β λ (μ₂ - μ₁) + ½ V'(μ_α)]

    Slow prior dynamics:
        dμ̄/dt = -ε η Λ_m (μ̄ - μ_α)

    where:
        μ̄ = slow prior (tracks μ_α with lag)
        μ_α = (μ₁ + μ₂)/2 (meta-agent belief, changes fast)
        ε = timescale separation parameter (ε << 1 for strong separation)
    """
    print("=" * 70)
    print("STEP 1: Change of variables")
    print("=" * 70)

    t = sp.Symbol('t')
    eta = sp.Symbol('eta', positive=True)       # fast learning rate
    alpha = sp.Symbol('alpha', positive=True)    # self-coupling weight
    lam_p = sp.Symbol('lambda_p', positive=True) # prior precision
    beta_s = sp.Symbol('beta', positive=True)    # alignment weight
    lam = sp.Symbol('lambda', positive=True)     # belief precision
    eps = sp.Symbol('epsilon', positive=True)    # timescale separation
    Lam_m = sp.Symbol('Lambda_m', positive=True) # meta-level precision

    delta = sp.Symbol('delta')      # relative coordinate μ₁ - μ₂
    V_prime = sp.Symbol("V'")       # external force V'(μ_α)

    print("""
    Define:
        μ_α = (μ₁ + μ₂)/2    (center of mass = meta-agent belief)
        δ   = μ₁ - μ₂         (relative coordinate = fast internal mode)
        ξ   = μ̄ - μ_α         (prior lag = slow mode)

    Then: μ₁ = μ_α + δ/2,  μ₂ = μ_α - δ/2
          μ₁ - μ̄ = δ/2 - ξ,  μ₂ - μ̄ = -δ/2 - ξ
    """)

    # =====================================================================
    # Derive the three decoupled ODEs
    # =====================================================================
    print("Substituting into constituent dynamics:")
    print()

    # dμ₁/dt = -η[α λ_p (μ₁ - μ̄) + β λ (μ₁ - μ₂) + V'/2]
    #        = -η[α λ_p (δ/2 - ξ) + β λ δ + V'/2]
    # dμ₂/dt = -η[α λ_p (-δ/2 - ξ) - β λ δ + V'/2]

    # dμ_α/dt = (dμ₁ + dμ₂)/2 = -η[-α λ_p ξ + V'/2]
    #         = η α λ_p ξ - η V'/2

    # dδ/dt = dμ₁ - dμ₂ = -η[α λ_p δ + 2β λ δ]
    #       = -η(α λ_p + 2β λ) δ

    omega_fast = eta * (alpha * lam_p + 2 * beta_s * lam)

    print(f"    (1)  dδ/dt   = -ω_f · δ           where ω_f = η(αλ_p + 2βλ)")
    print(f"         [Fast internal mode: decouples and decays exponentially]")
    print()
    print(f"    (2)  dμ_α/dt = η α λ_p · ξ - V'(μ_α)")
    print(f"         [Belief driven by prior lag ξ and external force]")
    print()
    print(f"    (3)  dξ/dt   = dμ̄/dt - dμ_α/dt")
    print(f"                 = -ε η Λ_m · (-ξ) - (η α λ_p ξ - V')")
    print(f"                 = -(ε η Λ_m + η α λ_p) ξ + V'(μ_α)")
    print(f"         [Prior lag: relaxes toward zero, driven by external force]")
    print()
    print(f"    δ decouples completely. The non-trivial dynamics is the")
    print(f"    coupled (μ_α, ξ) system: equations (2) and (3).")

    # =====================================================================
    # Eliminate ξ to get a single second-order ODE for μ_α
    # =====================================================================
    print()
    print("=" * 70)
    print("STEP 2: Eliminate ξ -> second-order ODE for μ_α")
    print("=" * 70)
    print()

    mu_dot = sp.Symbol(r'\dot{\mu}_\alpha')
    F = sp.Symbol("V'")
    dF = sp.Symbol("V''")

    # From (2): ξ = (μ̇_α + V') / (η α λ_p)
    xi_expr = (mu_dot + F) / (eta * alpha * lam_p)

    print(f"    From (2):  ξ = (μ̇_α + V') / (η α λ_p)")
    print()

    # Total relaxation rate for ξ
    gamma_tot = eps * eta * Lam_m + eta * alpha * lam_p

    # dξ/dt from (3):
    dxi_dt = -gamma_tot * xi_expr + F

    # Differentiate (2): μ̈_α = η α λ_p · dξ/dt - V'' · μ̇_α
    d2mu = eta * alpha * lam_p * dxi_dt - dF * mu_dot
    d2mu = sp.expand(d2mu)

    print(f"    Differentiate (2):  μ̈_α = η α λ_p · (dξ/dt) - V'' · μ̇_α")
    print()
    print(f"    Substitute (3) for dξ/dt and ξ = (μ̇ + V')/(ηαλ_p):")
    print()
    print(f"    μ̈_α = {d2mu}")
    print()

    # Extract coefficients
    c_mudot = d2mu.coeff(mu_dot)
    c_F = d2mu.coeff(F)

    # Standard form: μ̈ + γ_eff μ̇ + force_coeff V' = 0
    gamma_eff = sp.simplify(-c_mudot)
    force_coeff = sp.simplify(-c_F)

    print(f"    Rearranging:  μ̈_α + γ_eff · μ̇_α + f_coeff · V' = 0")
    print()
    print(f"    γ_eff   = {gamma_eff}")
    print(f"    f_coeff = {force_coeff}")
    print()

    # Divide by force_coeff -> M μ̈ + Γ μ̇ + V' = 0
    M_eff = sp.simplify(1 / force_coeff)
    Gamma_eff = sp.simplify(gamma_eff / force_coeff)

    # For the linear regime (V'' is part of the potential, not the damping)
    # set dF (V'') = 0 in γ_eff for the clean result
    M_lin = M_eff.subs(dF, 0)
    Gamma_lin = sp.simplify(Gamma_eff.subs(dF, 0))
    ratio_lin = sp.simplify(Gamma_lin / M_lin)

    print("=" * 70)
    print("RESULT: Effective second-order equation for meta-agent belief")
    print("=" * 70)
    print()
    print("    M_eff · μ̈_α + Γ_eff · μ̇_α + V'(μ_α) = 0")
    print()
    print(f"    M_eff   = {M_lin}")
    print(f"            = 1 / (ε η Λ_m)")
    print()
    print(f"    Γ_eff   = {Gamma_lin}")
    print(f"            = 1 + α λ_p / (ε Λ_m)    [dimensionless, times 1/M]")
    print()
    print(f"    Γ/M     = {ratio_lin}")
    print(f"            = ε η Λ_m + η α λ_p")
    print()
    print(f"    (With V'' correction:  γ_eff includes +V'' term in damping)")

    # =====================================================================
    # Physical interpretation
    # =====================================================================
    print()
    print("=" * 70)
    print("STEP 3: Physical interpretation")
    print("=" * 70)
    print("""
    MASS = 1/(ε η Λ_m): Inertia from prior sluggishness
    ─────────────────────────────────────────────────────
    M_eff increases when:
      • ε -> 0:   Stronger timescale separation (prior updates much slower
                  than beliefs). The prior barely moves -> large inertia.
      • η -> 0:   Slower overall dynamics -> everything more sluggish.
      • Λ_m -> 0: Weaker meta-level coupling -> prior weakly attracted to
                  belief -> drifts slowly -> large inertia.

    DAMPING: Γ/M = ε η Λ_m + η α λ_p
    ──────────────────────────────────
    Two contributions:
      • ε η Λ_m:   Meta-level coupling (prior tracking)
      • η α λ_p:   Self-coupling (belief anchoring to prior)

    ADIABATIC LIMIT (ε -> 0):
    ────────────────────────
    M -> ∞, Γ -> ∞, but Γ/M -> η α λ_p (FINITE).
    The M μ̈ term becomes negligible: M μ̈ ≪ Γ μ̇.
    Equation degenerates to FIRST ORDER:
        (η α λ_p)⁻¹ · μ̇_α + V'(μ_α) = 0
    This IS standard overdamped VFE gradient flow.
    Standard variational inference and transformers live here.

    FINITE ε (partial separation):
    ──────────────────────────────
    Second-order inertial term survives.
    -> Overshoot when beliefs are pushed by sudden evidence
    -> Oscillation around equilibrium (belief ringing)
    -> Momentum transfer between coupled meta-agents
    -> Resonance when driven at natural frequency
    """)

    # =====================================================================
    # Damping regimes
    # =====================================================================
    print("=" * 70)
    print("STEP 4: Underdamped vs overdamped")
    print("=" * 70)
    print()

    K = sp.Symbol('K', positive=True)  # V'' at equilibrium

    # Discriminant: Δ = Γ^2 - 4MK
    disc = sp.expand(Gamma_lin**2 - 4 * M_lin * K)
    print(f"    Discriminant Δ = Γ^2 - 4MK = {disc}")
    print()
    print(f"    Using ω_p = η α λ_p (self-coupling rate)")
    print(f"          ω_m = ε η Λ_m  (meta-coupling rate):")
    print()
    print(f"    Δ = (ω_p + ω_m)^2 / ω_m^2 - 4K/ω_m")
    print()
    print(f"    Underdamped when Δ < 0:")
    print(f"        4K · ω_m > (ω_p + ω_m)^2")
    print()
    print(f"    Natural frequency (underdamped case):")
    print(f"        ω^2 = K/M = K · ε η Λ_m")
    print()
    print(f"    Decay time:")
    print(f"        τ = 2M/Γ = 2/(Γ/M) = 2/(ε η Λ_m + η α λ_p)")

    # =====================================================================
    # Connection to Hessian mass
    # =====================================================================
    print()
    print("=" * 70)
    print("STEP 5: Dynamical mass vs Hessian mass")
    print("=" * 70)
    print("""
    The belief inertia manuscript identifies mass with the Hessian:
        M_Hessian = ∂^2S/∂μ^2 = Λ̄_p + Λ_o + Σβ Λ̃ + Σβ Λ = K

    This derivation shows:
        M_dynamical = 1/(ε η Λ_m)    (from timescale separation)
        K_spring    = M_Hessian       (curvature of VFE potential)

    These are DIFFERENT objects:
        • K (Hessian)  = spring constant = how STIFF the potential well is
        • M (dynamical) = inertial mass  = how SLUGGISHLY the system responds

    In real physics, K and M are independent. The Hessian-as-mass analogy
    conflated them. The Ouroboros derivation separates them:

        ω^2 = K/M = (Λ̄_p + ...) · ε η Λ_m

    The natural frequency depends on BOTH the precision of the well (Hessian)
    AND the timescale separation (Ouroboros coupling).
    """)

    return {
        'M_eff': M_lin,
        'Gamma_eff': Gamma_lin,
        'Gamma_over_M': ratio_lin,
        'omega_fast': omega_fast,
        'gamma_total': gamma_tot,
    }


def derive_n_agent():
    """
    N-agent generalization sketch.

    With N agents at scale ζ forming one meta-agent at scale ζ+1:
        μ_α = Λ_α⁻¹ Σᵢ wᵢ Λᵢ μᵢ     (precision-weighted mean)
        Λ_α = Σᵢ wᵢ Λᵢ                (pooled precision)

    The relative coordinates δᵢ = μᵢ - μ_α (i = 1..N-1) are fast modes
    that decouple from μ_α in the symmetric case (uniform weights/precisions).

    The (μ_α, ξ) system generalizes identically:
        dμ_α/dt = η α λ_p ξ - V'(μ_α)
        dξ/dt   = -(ε η Λ_m + η α λ_p) ξ + V'(μ_α)

    The effective equation is the SAME:
        M_eff μ̈_α + Γ_eff μ̇_α + V'(μ_α) = 0

    with M_eff = 1/(ε η Λ_m), independent of N.

    N enters through the pooled precision Λ_α = N · λ (for uniform case),
    which affects the meta-agent's OWN coupling to the scale above.
    If Λ_m depends on Λ_α (because the meta-agent is itself a constituent
    at the next scale), then M_eff depends on N indirectly:
        M_eff = 1/(ε η Λ_m(Λ_α)) = 1/(ε η Λ_m(N λ))

    For heterogeneous agents (different λᵢ), the δᵢ modes couple to μ_α
    through the attention weights βᵢⱼ (which depend on the δᵢ's).
    The effective equation acquires corrections of order δ^2:
        M_eff μ̈ + Γ_eff μ̇ + V' + O(⟨δ^2⟩) = 0

    The O(⟨δ^2⟩) corrections are the fluctuation contribution to the
    effective potential -- analogous to the Casimir effect or Coleman-Weinberg
    potential in QFT, where integrating out fast modes renormalizes the
    slow-mode potential.
    """
    print()
    print("=" * 70)
    print("N-AGENT GENERALIZATION")
    print("=" * 70)
    print("""
    For N symmetric agents, the result is identical:
        M_eff = 1/(ε η Λ_m),  same as N=2.

    N enters through the pooled precision Λ_α = Σᵢ wᵢ Λᵢ:
    - More constituents -> higher pooled precision -> stiffer meta-agent
    - This affects K (spring constant), not M (inertia)
    - Unless Λ_m depends on Λ_α via the next Ouroboros level

    For heterogeneous agents, fast-mode fluctuations ⟨δ^2⟩ renormalize
    the effective potential:
        V_eff(μ_α) = V(μ_α) + (1/2) Σᵢ ∂^2V/∂δᵢ^2 · ⟨δᵢ^2⟩ + ...

    This is the analog of the Coleman-Weinberg mechanism: integrating
    out fast internal modes generates corrections to the slow potential.
    The δᵢ fluctuation amplitude ⟨δᵢ^2⟩ ~ T/ω_fast^2 (thermal) or
    ~ 1/(2 ω_fast) (quantum/zero-point), depending on whether noise
    is included in the constituent dynamics.
    """)


def derive_multi_scale():
    """
    Multi-scale tower: recursive application.

    At each scale ζ, the effective equation is:
        M^(ζ) μ̈^(ζ) + Γ^(ζ) μ̇^(ζ) + V'^(ζ)(μ^(ζ)) = 0

    where V'^(ζ) includes the coupling to scale ζ+1, which itself
    has second-order dynamics. Substituting recursively:

    Scale 0 (elementary):     M₀ μ̈₀ + Γ₀ μ̇₀ + K₀(μ₀ - μ₁) = 0
    Scale 1 (first meta):     M₁ μ̈₁ + Γ₁ μ̇₁ + K₁(μ₁ - μ₂) = f₁(μ₀)
    Scale 2 (second meta):    M₂ μ̈₂ + Γ₂ μ̇₂ + K₂(μ₂ - μ₃) = f₂(μ₁)
    ...

    This is a CHAIN OF COUPLED OSCILLATORS across scales, with:
        Mᵢ = 1/(εᵢ ηᵢ Λ_m,ᵢ)   -- mass increases with scale (εᵢ decreases)
        Kᵢ = Hessian at scale i  -- spring constant from VFE curvature
        Γᵢ = damping at scale i  -- increases with scale

    The lowest scales are underdamped (fast, oscillatory).
    The highest scales are overdamped (slow, sluggish).
    Somewhere in the middle is critical damping -- the scale at which
    the system transitions from wave-like to diffusive behavior.
    """
    print()
    print("=" * 70)
    print("MULTI-SCALE TOWER: Chain of coupled oscillators")
    print("=" * 70)
    print("""
    Each Ouroboros level adds another oscillator to the chain:

    Scale 0:  M₀ μ̈₀ + Γ₀ μ̇₀ + K₀₁(μ₀ - μ₁) = F_ext
    Scale 1:  M₁ μ̈₁ + Γ₁ μ̇₁ + K₁₂(μ₁ - μ₂) = K₀₁(μ₀ - μ₁)  [bottom-up]
    Scale 2:  M₂ μ̈₂ + Γ₂ μ̇₂ + K₂₃(μ₂ - μ₃) = K₁₂(μ₁ - μ₂)
    ...

    With Mζ = 1/(εζ ηζ Λ_m,ζ) growing at each scale (εζ shrinks).

    Properties:
    • Low scales: light, fast, possibly underdamped -> oscillatory
    • High scales: heavy, slow, overdamped -> sluggish relaxation
    • Critical scale ζ* where Δ = Γ^2 - 4MK = 0 -> transition
    • Below ζ*: wave-like (phonon-like excitations of belief)
    • Above ζ*: diffusive (gradient-flow dynamics)

    This is a PREDICTION: there exists a critical scale in any
    hierarchical agent system where dynamics transitions from
    oscillatory to overdamped. The scale depends on the timescale
    separation ε and the coupling strengths.
    """)


if __name__ == '__main__':
    results = derive_two_agent()
    derive_n_agent()
    derive_multi_scale()

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
    Starting from:
        • N first-order agents coupled through VFE (no momentum, no mass)
        • Ouroboros loop: meta-agent belief -> constituent prior (slow update)

    Derived:
        • Effective SECOND-ORDER dynamics for meta-agent belief
        • M_eff = 1/(ε η Λ_m)  -- mass from prior sluggishness
        • K = Hessian of VFE   -- spring constant from precision (NOT mass)
        • Overdamped limit ε->0 recovers standard VFE / transformers
        • Multi-scale tower = chain of coupled oscillators

    Key distinction:
        • Hessian Λ_p = SPRING CONSTANT (how stiff the well is)
        • M_eff = 1/(εηΛ_m) = INERTIAL MASS (how sluggish the response)
        • These are independent, as in real physics
    """)
