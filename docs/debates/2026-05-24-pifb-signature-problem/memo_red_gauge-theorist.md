# Memo — debate-expert-gauge-theorist — red — opening — pifb-signature-problem

## Lens
Lie groups, real forms, Killing/trace form signature, generator and convention choices.

## Steelman of the opposing position
The group-theory facts the section invokes are all standard and stated correctly: $\mathfrak{sl}(2,\mathbb{R})$ is non-compact with $\mathrm{tr}(T^2)>0$ available, the compact form uses $-\mathrm{tr}$, $\mathrm{SL}(2,\mathbb{C})\cong\mathrm{Spin}^+(1,3)$, $\mathrm{SO}^+(1,3)$ sits in $\mathrm{GL}(4,\mathbb{R})$, and Wick rotation relates real forms of $\mathrm{SO}(4,\mathbb{C})$ — so the worked example correctly identifies the algebraic ingredients of an indefinite form.

## My position (in service of red)
Every standard group fact is stated correctly — concede that fully. But correctly identifying the ingredients of an indefinite form is exactly the problem: the signature is *manufactured by the choice of bilinear form on $\mathfrak{g}$*, and that choice is a free input, not a framework output. The section reads signature off $G_{\mu\nu}=\mathrm{tr}(A_\mu A_\nu)$ with the $+\mathrm{tr}$ convention. On a compact real form the canonical invariant inner product is $-\mathrm{tr}(AB)$, which is negative-definite; the trace form $+\mathrm{tr}$ is indefinite on a split real form. So the section is not discovering an indefinite signature — it is selecting the indefinite invariant form and then noting that it is indefinite.

The manuscript itself certifies this at `:2868`: "Under the standard positive-definite inner product on a compact form $\mathfrak{so}(N)$ (which is $-\mathrm{tr}(AB)$, not $+\mathrm{tr}(AB)$), the sign relation in the calculation below would be reversed, and the construction would not produce the desired Lorentzian split." And at `:2950` it admits the indefinite signature is obtainable with **real** generators (compact + non-compact mixed) under $+\mathrm{tr}$, no complexification at all. The Killing form on a non-compact semisimple real form is generically indefinite by Cartan's criterion; on $\mathfrak{sl}(2,\mathbb{R})$ it has signature $(2,1)$. So "indefinite signature on the connection sector" is the *generic* situation for the trace form on a non-compact algebra. The worked example's imaginary-$\phi_\tau$ postulate and real-part projection are an elaborate route to a sign that the convention choice already determines.

The single-generator collapse $T_\tau=T_x=T$ (`:2872`) compounds this: it is what forces the complex form to be rank-1 ($\det=0$), which is what *requires* the real-part projection to recover a non-degenerate form. Both are postulates the section admits change the answer. The gauge-theoretic content reduces to: pick the split real form and the $+\mathrm{tr}$ convention, and you get an indefinite signature — which is a tautology about real forms of $\mathfrak{gl}(K,\mathbb{C})$, not a derivation of Lorentzian spacetime.

## Evidence
- Knapp, *Lie Groups Beyond an Introduction* (2nd ed., 2002), Ch. VI: the Killing form of a real semisimple Lie algebra is negative-definite iff the algebra is compact; on a non-compact (split) form it is indefinite. The $+\mathrm{tr}$ trace form inherits this on $\mathfrak{sl}(2,\mathbb{R})$.
- Verified via WebFetch (Wikipedia, *Killing form*, for orientation; cite Knapp at source): split $\mathfrak{sl}(2,\mathbb{R})$ Killing-form signature $(2,1)$ (indefinite); compact $\mathfrak{su}(2)$ signature $(0,3)$ (negative-definite). This is precisely the sign reversal the manuscript flags at `:2868`.
- Hall, *Lie Groups, Lie Algebras, and Representations* (2nd ed., 2015), Ch. on $\mathrm{SU}(2)$/$\mathrm{SL}(2,\mathbb{C})$ and spin groups: $\mathrm{SL}(2,\mathbb{C})\cong\mathrm{Spin}^+(1,3)$ double cover — the manuscript states this correctly (concede).
- Manuscript `:2950`: indefinite signature obtainable from a real $\mathfrak{gl}(K,\mathbb{R})$ frame mixing compact and non-compact generators under $+\mathrm{tr}$ — the complexification is inessential to the signature mechanism.

## Newly-discovered canon (for 01b_extended_evidence.md)
- Knapp, A. W. (2002). *Lie Groups Beyond an Introduction*, 2nd ed., Birkhäuser, Ch. VI (Structure theory; compact and non-compact real forms). Cartan's criterion: a real semisimple Lie algebra is compact iff its Killing form is negative-definite; non-compact real forms have indefinite Killing form. URL: book, not freely fetchable — verify against print copy.
- Wikipedia, *Killing form* (orientation only): $\mathfrak{sl}(2,\mathbb{R})$ Killing-form signature $(2,1)$; $\mathfrak{su}(2)$ signature $(0,3)$. https://en.wikipedia.org/wiki/Killing_form

## Falsification conditions
My position is wrong if (a) the framework independently fixes the $+\mathrm{tr}$ convention and the split real form on dynamical grounds (it does not — `:2829`, `:2868` present both as choices); or (b) the indefinite signature were *unavailable* without complexification, contradicting the manuscript's own `:2950` real-generator route.

## Confidence
HIGH — the manuscript states the convention-dependence at `:2868` and the real-generator route at `:2950` itself; the Killing-form signature facts are textbook (Knapp).
