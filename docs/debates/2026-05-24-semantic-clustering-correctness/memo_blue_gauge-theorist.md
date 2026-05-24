# Memo — gauge-theorist (blue, Phase 2)

Authored in-role by the coordinator: the consultant dispatch tool was unavailable in this environment, so the panel memos were written from the canon and the read code, then synthesized into `02_blue_opening.md`.

## Position
The Ω geometry is sound under the active config. The generator bank is exactly block-diagonal (`generate_glK_multihead_generators(20,2)`, `math_utils/generators.py:870-953`; executed max off-block = 0.0), so the algebra element factors `A = A_1 ⊕ A_2` and `exp(A) = exp(A_1) ⊕ exp(A_2)` term-by-term in the power series. The two Ω code paths — block-restrict-then-exp (`geometry.py:339`) and full-exp (`pipeline.py:119-121`) — therefore compute the same blocks (verified 4.4e-16). The per-head distance `‖log(Ω_{h,i}⁻¹Ω_{h,j})‖_F` is a canonical left-invariant Frobenius distance on a matrix Lie group [Nakahara2003 §5-6; Hall2015]; the quadrature `sqrt(Σ_h d_h²)` is the exact product metric on `GL(10)×GL(10)` [Lee2013 product manifolds], not an approximation.

## Honest caveat
GL⁺(K) is non-compact: no unique bi-invariant metric. `‖log(g⁻¹h)‖_F` is a principled standard choice, not the only canonical one. Keep red's falsification (a) live.

## Primary-source citation
[Nakahara2003 §5-6], [Hall2015 §2-3] — matrix-group exponential and block factorization; [Lee2013] — product-manifold metric.
