# Blue Memo — gauge-theorist

## Steelman of the attack
Reading a "signature" off the gauge-noninvariant trace form tr(A_μ A_ν) — which is not a Yang-Mills invariant and has infinite Haar average over the non-compact orbit — is a sign game decoupled from any physical content; the single-generator collapse T_τ=T_x=T makes the result an artifact of one hand-picked matrix.

## Defense from canon
The construction is internally consistent with standard Lie-theory facts, each of which is correctly stated:

1. T = diag(1,−1) is symmetric, traceless, hence in the non-compact 𝔰𝔩(2,ℝ), *not* in the compact 𝔰𝔬(2) (skew-symmetric only). tr(T²)=2>0 is available precisely because the generator is non-compact. The compact-form inner product on 𝔰𝔬(N) is −tr(AB) (positive-definite), under which tr(T²)≤0; the construction's use of +tr(AB) and a non-compact generator is the *stated* postulate, not a hidden one [Knapp, *Lie Groups Beyond an Introduction*, ch. I: the Killing form is negative-definite on a compact real form and indefinite on a split/non-compact real form; for 𝔤𝔩 the trace form tr(AB) is indefinite].

2. The group facts are standard and correct: SL(2,ℂ) ≅ Spin⁺(1,3) is the spin double cover [Hall, *Lie Groups, Lie Algebras, and Representations*, ch. 5; confirmed Lorentz-group canon]; the vector representation of SO⁺(1,3) sits in GL(4,ℝ)⊂GL(4,ℂ); SO⁺(1,3) is the identity component of O(1,3) [Lorentz-group canon, confirmed]. The manuscript keeps the spinor and vector reps explicitly distinct (2900) — a place where many treatments conflate them, and the manuscript does not.

3. Wick rotation as a relation between real forms of the complexified SO(4,ℂ) is the standard QFT statement [Streater-Wightman / Folland, *Quantum Field Theory*: Euclidean and Minkowski signatures are different real forms of the complexified rotation group].

4. The gauge-noninvariance of tr(A_μ A_ν) and the infinite-Haar regulator obstruction are *disclosed by the manuscript itself* (2829, 2904) as a limitation, not claimed away — this is the honest move, not the overclaim.

## External primary-source citation
[Knapp, *Lie Groups Beyond an Introduction*, ch. I — Killing/trace form is indefinite on 𝔤𝔩 and on a split real form, negative-definite on a compact form]; [Hall, ch. 5 — SL(2,ℂ) ≅ Spin⁺(1,3)].

## Falsification condition (argued unmet)
The gauge-side defense fails if (i) the trace algebra is wrong — sympy confirms it is correct (tr(T²)=2, G_ττ=−2(∂ψ)², G_xx=+2(∂ψ)², G_τx=2i(∂ψ)(∂ψ)); or (ii) the manuscript claims tr(A_μ A_ν) is a gauge invariant or a YM kinetic term — it explicitly says the opposite (2829). Neither holds. The single-generator-collapse sensitivity is disclosed (2872, 2902): with a compact T_x the trace identities flip sign (sympy: tr(T_c²)=−2). Disclosed sensitivity is not an error.
