# Defect 10: "Yang-Mills kinetic metric" — naming and sign-convention conflict

**Manuscript:** `Attention/Participatory_it_from_bit.tex`
**Date:** 2026-05-06
**Reviewer task:** Verify defects 4a (naming/structure) and 4b (sign/convention conflict) for the object `tr(A_mu A_nu)`.

---

## Verdict

- **Defect 4a (naming):** **Partially confirmed.** The manuscript does call this object the "Yang-Mills kinetic metric" in two places (lines 1633 and 1648 of the active text, which are the section heading and worked-example heading, respectively), and as `g_C^YM` / `G^YM_munu` in equation labels. However, the manuscript already concedes the naming is shorthand (line 1614) and explicitly disowns the Yang-Mills invariant `tr(F F)` as an escape hatch (lines 1545–1546, 1614). The remaining defect is that the label "Yang-Mills" still propagates through equation tags (`g_{C,munu}^{YM}`, `G_{munu}^{YM}`) and several prose references after the disclaimer, which is misleading because the object is neither built from curvature nor gauge-invariant.

- **Defect 4b (sign/convention conflict):** **Confirmed.** The manuscript uses `T = diag(1,-1)` (lines 1640–1642), which is symmetric and traceless, hence in `sl(2,R)` but **not** in `so(2)`. `so(2)` consists of skew-symmetric `2x2` matrices, whose only nonzero element is the off-diagonal antisymmetric block. Under the standard compact convention `tr(A^2) <= 0` for `A in so(n)` or `su(n)`, so the kinetic form would be negative-definite, not positive. The manuscript reaches a positive `tr(T^2) = 2` only by selecting a non-compact generator. The text does not flag that this generator is outside the compact form, nor does it state which inner product on `g` is being used (Killing form vs. trace vs. negative trace). The Wick-rotation-via-`i` move on top of a non-compact generator then reads as a second sign flip whose mathematical status is not pinned down.

---

## Quoted text — definition and naming

**Lines 1521–1525 (definition):**

> the horizontal part is the gauge-connection kinetic form
> `g_{C,munu}^{YM}(c) := tr(A^{(i)}_mu(c) A^{(i)}_nu(c))`,
> already encountered in Sections~\ref{sec:cognitive_reference_frames} and~\ref{sec:signature_resolution}.

**Lines 1545–1546 (gauge-variance disclosure):**

> `tr(A_mu A_nu)` is therefore not invariant: it transforms with cross terms ... The horizontal block is consequently agent-frame-dependent ... We do not attempt to absorb this non-invariance by appealing to the Yang-Mills invariant `tr(F_{munu} F^{munu})`: under Regime~I ... `F_{munu} = 0` identically, so `tr(FF)` vanishes and supplies no escape hatch in the present implementation.

**Line 1614 (regime disclaimer, "shorthand" admission):**

> The signature mechanism operates not through `F_{munu}` but through the quantity `tr(A_mu A_nu)`, which is gauge-noninvariant ... and is therefore best understood as a frame-dependent quadratic form rather than as a Yang-Mills kinetic invariant. ... Calling the resulting object a "Yang-Mills kinetic metric" is shorthand; the construction is a frame-dependent metric whose gauge-invariant content is recovered only after the consensus / Haar-averaged construction.

**Line 1633 (sub-section heading "3. The Yang-Mills kinetic metric."):**

> On base manifolds of dimension >= 1, the gauge frame field `phi: C -> gl(K,C)` induces a connection `A_mu = U^{-1} partial_mu U` whose Yang-Mills kinetic term `G_{munu}^{YM} = tr(A_mu A_nu)` contributes to the effective metric on `C`.

**Line 1648 (worked example):**

> The Yang-Mills kinetic form `G_{munu} = tr(A_mu A_nu)` evaluates to ...

So: the "Yang-Mills kinetic metric" name is used after the disclaimer; relationship to the curvature `tr(F F)` is acknowledged (the connection is pure gauge in Regime I, so `F = 0` and the YM action vanishes), and the object is **not** claimed to be gauge-invariant. The naming is conceded to be shorthand but is not actually replaced.

## Quoted text — generator choice

**Lines 1640–1642 (worked example):**

> Consider a 2-dimensional base manifold `C` with coordinates `(tau, x)` and a `GL(2,C)` gauge bundle with Lie algebra generators `T in gl(2)`. For concreteness, take `T = diag(1, -1)` with `tr(T^2) = 2`.

The membership is correctly stated as `T in gl(2)`, **not** `so(2)`. But the manuscript does not say that this choice is what makes `tr(T^2) > 0` available, nor that the same construction with a compact generator would give `tr(T^2) <= 0` and require a different sign convention.

---

## Sign computation (sympy-verified)

```
A in so(2):  A = [[0,a],[-a,0]]   ->  tr(A^2) = -2 a^2  <= 0
A in so(3):  general skew         ->  tr(A^2) = -2(b^2+c^2+d^2) <= 0
A in su(2):  A = i*diag(1,-1)     ->  tr(A^2) = -2          <= 0
T = diag(1,-1) (symmetric, in sl(2,R), not in so(2)):  tr(T^2) = +2 > 0
T + T^T = diag(2,-2) != 0, so T is symmetric, not skew.
```

So `T = diag(1,-1)` is in `sl(2,R) ⊂ gl(2,R)` but **not** in `so(2)`. The sign claim is correct: under the compact convention, `tr(A^2)` is negative-semidefinite, and the standard positive-definite inner product is `-tr(A B)` (or, more invariantly, `-(1/2)tr(A B)` for `su(N)`, the negative Killing form up to normalization). The manuscript's `+tr(T^2) = +2` is only available because `T` is non-compact.

---

## Diagnosis: convention conflict

The manuscript wants `tr(A_mu A_nu)` to play two roles simultaneously:

1. A **positive-definite spatial kinetic term** (so that `G_{xx} = +2(partial_x psi_x)^2 > 0`).
2. A quantity that becomes **negative** in the temporal direction via `phi_tau -> i phi_tau`, giving `G_{tau tau} = -2(partial_tau psi_tau)^2 < 0`.

Under the compact-group reading (`A in so(N)` or `su(N)`), step 1 fails: `tr(A^2) <= 0` already, so the spatial sector would be negative, and the construction would have to use `-tr(A_mu A_nu)` (or equivalently the negative Killing form) as the inner product. The Wick rotation `phi_tau -> i phi_tau` would then make the temporal sector **positive**, giving signature `(+,-)` — the opposite of what the manuscript claims.

Under the non-compact reading (`T = diag(1,-1) in gl(2,R)`), step 1 succeeds because the generator is symmetric, and step 2 works as the manuscript writes it. But this means:

- The construction is **not** a Wick rotation of a compact gauge theory at all. A compact-group Wick rotation would replace `so(N)` with the non-compact real form `so(p,q)` or pass to the complexification `so(N,C)`. The manuscript is doing something different: it is taking a non-compact real generator and multiplying its scalar coefficient by `i`.
- The "Wick rotation in the Lie algebra" framing on line 1638 (`phi_tau -> i phi_tau`) is structurally the same as analytically continuing the scalar field; it is **not** a passage between real forms of a complex Lie algebra. Calling this Wick rotation needs more care.
- The manuscript never states which inner product on `g` is being used. For `gl(2,R)` the trace form `tr(A B)` is indefinite (signature `(3,1)` on the symmetric `2x2` matrices), so picking the trace form is a choice; for `so(N)` the standard choice is the negative trace (or negative Killing).

The manuscript's choices **are** internally consistent if read as: "`g = trace form on gl(2,R)`, generator `T` chosen symmetric so that `tr(T^2) > 0`, then continue `phi_tau` to imaginary values." But it does not say this. The "Yang-Mills kinetic" framing instead invites the compact-group reading under which the signs are wrong.

---

## Recommended LaTeX rewrites

### Fix 1 — Around line 1521 (definition of horizontal metric)

Replace the equation label `g_{C,munu}^{YM}` and the prose "gauge-connection kinetic form" with a name that is honest about what the object is. Suggested rewrite:

```latex
where $\pi: E_q \to \mathcal{C}$ is the bundle projection, $g_{\mathcal{B}}$ is the
Fisher-Rao metric on the fiber, and the horizontal part is the
\emph{frame-twist quadratic form}
\begin{equation}
g_{\mathcal{C},\mu\nu}^{\mathrm{tw}}(c) := \langle A^{(i)}_\mu(c), A^{(i)}_\nu(c)
\rangle_{\mathfrak{g}},
\label{eq:horizontal_metric}
\end{equation}
where $\langle \cdot, \cdot \rangle_{\mathfrak{g}}$ is a fixed bilinear form on
$\mathfrak{g}$. For $\mathfrak{g} = \mathfrak{so}(N)$ or $\mathfrak{su}(N)$ the
canonical positive-definite choice is $\langle A, B \rangle_{\mathfrak{g}} =
-\mathrm{tr}(AB)$ (the trace form is negative-semidefinite on these compact real
forms). For $\mathfrak{g} = \mathfrak{gl}(K)$ the choice
$\langle A, B \rangle_{\mathfrak{g}} = \mathrm{tr}(AB)$ is indefinite, and this
indefiniteness is what the signature construction of Section~\ref{sec:signature_resolution}
exploits. The object is \emph{not} the Yang-Mills kinetic invariant
$\mathrm{tr}(F_{\mu\nu}F^{\mu\nu})$: it depends on the connection $A$ rather than
on its curvature $F$, is gauge-variant (see disclosure below), and reduces to
zero only when the connection is locally trivial, not when the field strength
vanishes.
```

Then propagate `g_{C,munu}^{YM}` -> `g_{C,munu}^{tw}` (or `g_{C,munu}^{conn}`), and `G_{munu}^{YM}` -> `G_{munu}^{tw}` everywhere downstream (the equation labels at 1525, 1633, 1648, and the dual-metric box at 1556).

### Fix 2 — Around lines 1602–1620 (`SO(3)` -> `GL(K,C)` resolution)

Rename the subsubsection at line 1633 from "The Yang-Mills kinetic metric." to "The frame-twist quadratic form." and rewrite the opening:

```latex
\paragraph{3. The frame-twist quadratic form.} On base manifolds of dimension
$\geq 1$, the gauge frame field $\phi: \mathcal{C} \to \mathfrak{gl}(K,
\mathbb{C})$ induces a connection $A_\mu = U^{-1}\partial_\mu U$ whose pulled-back
quadratic form $G_{\mu\nu}^{\mathrm{tw}} = \mathrm{tr}(A_\mu A_\nu)$ contributes
to the effective metric on $\mathcal{C}$. We use the trace form on
$\mathfrak{gl}(K,\mathbb{C})$ rather than the negative trace appropriate to the
compact real forms $\mathfrak{su}(N)$ or $\mathfrak{so}(N)$; this choice is what
allows $\mathrm{tr}(A^2)$ to be positive on a real spatial direction and, via
the Lie-algebra continuation $\phi_\tau \to i\phi_\tau$, negative on a temporal
direction. Under the compact convention $\langle A, B \rangle = -\mathrm{tr}(AB)$
the same construction would produce signature $(+,-)$ rather than $(-,+)$ and
require the temporal continuation to be performed in the opposite direction.
```

### Fix 3 — Worked example (around lines 1640–1642)

Add an explicit statement that the generator is non-compact:

```latex
For concreteness, take $T = \mathrm{diag}(1, -1)$. The generator is symmetric,
hence $T \in \mathfrak{sl}(2,\mathbb{R}) \subset \mathfrak{gl}(2,\mathbb{R})$
but $T \notin \mathfrak{so}(2)$ (the Lie algebra of $\mathrm{SO}(2)$ consists of
$2\times 2$ skew-symmetric matrices, for which $\mathrm{tr}(A^2) \leq 0$). With
this non-compact choice, $\mathrm{tr}(T^2) = +2$, and $\mathrm{tr}(A_\mu A_\nu)$
gives a positive spatial component before the Lie-algebra Wick rotation. A
construction starting from a compact generator would have $\mathrm{tr}(T^2) < 0$,
and the conventional positive-definite inner product is $-\mathrm{tr}(\cdot,
\cdot)$; in that convention the worked example below would produce signature
$(+,-)$ and would require continuing the \emph{spatial} component to imaginary
values rather than the temporal one. The choice between conventions is not
physical, but stating it explicitly is necessary to compare the two routes.
```

### Fix 4 — Equation labels and global rename

Change `g_{C,munu}^{YM}` -> `g_{C,munu}^{tw}` and `G_{munu}^{YM}` -> `G_{munu}^{tw}` at every occurrence (lines 1525, 1556 (`G^{(p)}` box if applicable), 1633 heading, 1648, 1665 paragraph 3, 1726 (`G^{total} = G^{tw} + G^{Fisher}`)). The two `tr(F F)` references at 1545–1546 and 1614 are correctly distinguishing the YM invariant from this object and should remain unchanged.

---

## Notes for the editor

The disclaimer at line 1614 already concedes most of defect 4a in plain text: "Calling the resulting object a 'Yang-Mills kinetic metric' is shorthand." The remaining work is mechanical: rename the equation labels and the subsubsection heading so that the body text and the boxed expressions match the disclaimer. Defect 4b is more substantive — the choice of trace form (rather than negative trace) needs to be stated explicitly, and the non-compactness of `T = diag(1,-1)` should be acknowledged inline at the point of choice, not left implicit in the membership statement `T in gl(2)`.

The framework's actual logical situation is: pick `gl(K,C)` with the trace form (indefinite), write a connection that is pure-gauge so `F = 0`, then exploit the indefiniteness of the trace form together with a Lie-algebra analytic continuation to get a Lorentzian-signature pulled-back form. None of this is Yang-Mills theory; it is one bilinear form on `gl(K,C)` evaluated on a flat connection. The naming should reflect this.
