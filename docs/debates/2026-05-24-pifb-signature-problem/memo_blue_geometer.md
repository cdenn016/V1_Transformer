# Blue Memo — geometer

## Steelman of the attack
The real-part projection is a rank-changing operation (rank-1 complex → rank-2 real) with "no physical principle mandating it" by the manuscript's own admission; a metric conjured by discarding the imaginary part of a degenerate form is not a geometric object the framework produces, so "Lorentzian metric" overstates a linear-algebra maneuver.

## Defense from canon
Two linear-algebra facts carry the geometric content, both correctly stated and verified:

1. **Sylvester's law of inertia.** The signature of a real symmetric bilinear form is invariant under congruence (change of basis) [Horn & Johnson, *Matrix Analysis*, §4.5, Sylvester's law of inertia]. This grounds two distinct manuscript claims, both correct: (a) the "no-go" that GL(K,ℝ) transport ΩΣΩ⊤ of a positive-definite Σ stays positive-definite — congruence cannot change inertia, so a real frame cannot manufacture a timelike direction (2837, 2950); (b) the causal-cone count: g = diag(−c², h) with c²>0 and h positive-definite has exactly one negative and dim Σ positive eigenvalues, signature (−,+,…,+) (2923). sympy confirms eigenvalues {−c², h₁, h₂, h₃}.

2. **The rank-1 → rank-2 projection is honestly characterized, not hidden.** sympy: the unprojected complex form has det = 0 (rank one); Re(·) gives diag(−2(∂ψ)², +2(∂ψ)²) with det = −4(∂ψ)⁴ < 0 (rank two, signature (−,+)). The manuscript states *exactly* this — "rank-changing operation on the bilinear form (rank one complex to rank two real), not merely a discard of an off-diagonal imaginary piece" (2887) — and flags it as a "derivation gap" with "no physical principle mandating it" (2892). A gap that is named, characterized to the correct rank count, and labelled a gap is the opposite of an overclaim.

The causal-cone route reaches Lorentzian signature with *no* complexification and *no* projection at all — purely Sylvester on a real form — so the projection objection does not touch the second, independent existence demonstration (2920–2923).

## External primary-source citation
[Horn & Johnson, *Matrix Analysis* (2nd ed.), §4.5 — Sylvester's law of inertia: signature is a congruence invariant]. Standard linear-algebra canon; also any treatment of quadratic forms (Lee, *Smooth Manifolds*, on non-degenerate symmetric 2-tensors).

## Falsification condition (argued unmet)
Fails if (i) the Sylvester count is wrong — sympy confirms it; (ii) the rank claim is wrong — sympy confirms det=0 complex, det<0 real; (iii) the manuscript mischaracterizes the projection as benign — it does the reverse, calling it a rank-changing derivation gap. None hold. Honest blue position: the *physical motivation* of the projection is genuinely absent, and the claim does not defend it; it defends only that the existence demonstration and its gap are correctly stated.
