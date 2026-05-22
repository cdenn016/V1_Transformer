# Red Opening — pifb-spec-2-signature

## Steelman (opposing position)

The subsection 2773-2903 is a structurally honest existence demonstration that the framework's gauge geometry can host Lorentzian signature under a labelled postulate set; each substantive input (sector split, choice of bilinear form, imaginary temporal frame component, real-part projection, the 1+3 versus 2+2 input) is flagged as a postulate in the text, the central derivation gap is registered at 2846, and the alternative causal-cone route's tension with first-order natural-gradient dynamics is registered at 2879-2880, so the subsection is publication-ready as a worked example rather than as a derivation.

## Position

The subsection is **not** publication-ready as written. The worked example silently collapses two independent generators `T_tau`, `T_x` to a single `T` between equation (2824) and equation (2828) — that collapse is what forces the connection's image in the Lie algebra to lie in a one-dimensional subspace, which in turn forces the unprojected complex bilinear form `G_{mu nu} = tr(A_mu A_nu)` to be a rank-1 outer product `2 v v^T` with `det = 0`, not a rank-2 indefinite form. The "real-part projection" at 2841-2846 is therefore not a "discard the imaginary off-diagonal piece" operation, as the text frames it; it is a rank-changing operation from a degenerate rank-1 complex form to a non-degenerate rank-2 real form. The collapse `T_tau = T_x = T` is the unflagged postulate that creates the very problem the projection is then deployed to solve. Operational test 2 of `00_claim.md` ("each postulate explicitly flagged") fails on this point.

## Evidence

- **Manuscript line 2824 vs 2828 — the unflagged collapse.** Equation 2824 writes the frame field decomposition as `phi(tau, x) = psi_tau · T_tau + psi_x · T_x`, with the immediately following line declaring `psi_tau, psi_x` scalar fields *and* `T_tau, T_x` generators (plural, distinct). Equation 2828 then writes `phi(tau, x) = i psi_tau · T + psi_x · T`, with a single `T`. The postulate introduced at 2826 is explicitly only "that the temporal component is imaginary" (verbatim: "We now \emph{postulate} that the temporal component is imaginary"); the collapse `T_tau = T_x = T` is not labelled a postulate or simplification anywhere in 2822-2858.

- **Sympy verification — the unprojected complex form is rank-1.** With the single-generator assignment from equation 2828, the linearized connection is `A_tau = i (d_tau psi_tau) T`, `A_x = (d_x psi_x) T`, both proportional to the same `T`. The frame-twist form is therefore `G_{mu nu} = tr(A_mu A_nu) = c_mu c_nu tr(T^2)` with `c_tau = i d_tau psi_tau`, `c_x = d_x psi_x`, an outer product `2 v v^T` with `v = (c_tau, c_x)^T`. Sympy session:

  ```python
  v = Matrix([[I*dt],[dx]])
  G = 2 * v * v.T
  # G = [[-2 dt^2, 2 I dt dx], [2 I dt dx, 2 dx^2]]
  G.det()     # = 0
  G.rank()    # = 1
  G.eigenvals()  # {-2 dt^2 + 2 dx^2: 1, 0: 1}
  ```

  The full unprojected complex form has signature `(1, 0, 1)` (one nonzero eigenvalue plus a zero), not signature `(1, 1)`. Calling the real-part projection at 2841 a step that "discards the imaginary off-diagonal piece" understates the operation by a rank.

- **Sympy verification — two trace-orthogonal generators would not need any projection.** Replacing the single-generator assignment with `T_tau = diag(1,-1)`, `T_x = [[0,1],[1,0]]` (both in `sl(2, R)`, both with `tr(T_a^2) > 0`, mutually trace-orthogonal: `tr(T_tau T_x) = 0`), the same calculation gives `G_{tau tau} = -2 (d_tau psi_tau)^2`, `G_{xx} = 2 (d_x psi_x)^2`, **`G_{tau x} = 0` exactly**. The form is rank-2 with signature `(-, +)` directly, no projection required. So the projection step is a consequence of the single-generator simplification, not of the imaginary-direction postulate. Sympy session output:

  ```
  With distinct gens T1=diag(1,-1), T2=[[0,1],[1,0]]:
  G_tt =  -2*dpt**2
  G_xx =  2*dpx**2
  G_tx =  0
  ```

- **External canon — standard Wick rotation does not produce complex forms requiring real-part projection.** Standard Wick rotation continues a real coordinate `tau -> i tau` on the base manifold and substitutes back into a metric `ds^2 = -dt^2 + dx^2 + ...` to obtain a real Euclidean metric `dtau^2 + dx^2 + ...` directly, with no complex residue [nLab "Wick rotation"; standard treatment in any QFT textbook]. The tangent-space version of Wick rotation due to Samuel ([Samuel arXiv:1510.07365]) uses the substitution `e_0 -> i e_0` on a tetrad and likewise yields a real Euclidean metric directly. The Lie-group-level treatment of Helleland and Hervik ([Helleland-Hervik arXiv:1703.04576; arXiv:1810.12037]) treats Wick rotation as a comparison of distinct real forms of a complexified Lie group via Cartan involutions and similarly produces real metrics on each real slice. None of these standard constructions requires a real-part projection of a complex bilinear form. The manuscript at 2820 calling its operation "structurally analogous to a Wick-like continuation" therefore overreaches; the registered acknowledgement at 2846 that the construction "has no Wick counterpart" is correct, but the two statements are in tension within four lines of each other.

- **Manuscript 2856 vs 2858 — internal framing inconsistency.** Line 2856 states the conclusion as "the indefinite signature arises from the gauge connection (the frame-twist form `tr(A_mu A_nu)`)". Line 2858 immediately states "the central open question is whether a free-energy mechanism selects an imaginary `phi_tau` over a real one". The 2856 framing reads as a derived result; the 2858 framing reads as an undischarged postulate. Both readings cannot be simultaneously true in the "publication-ready and rock-solid" sense the claim asserts.

## Falsification conditions

This position is wrong if any of the following is established by blue:

1. The manuscript anywhere in 2773-2903 explicitly flags `T_tau = T_x = T` as a postulate, simplification, or input to the worked example (read 2822-2854 word-by-word). If such a flag exists, the unflagged-postulate strike collapses.
2. The unprojected complex form `G_{mu nu}` is in fact rank-2 (non-degenerate) under the manuscript's stated equation 2828 assignment with a single `T`. (Sympy verification above gives det = 0; counter-evidence would require pointing to a computational error in the verification.)
3. Standard mathematical or physics literature exists in which "Wick rotation" is applied to a Lie-algebra-valued connection and produces a complex bilinear form whose real part is then taken as the physical metric, with the rank-changing step accepted as a Wick-like operation. (Helleland-Hervik and Samuel were checked; neither does this.)
4. The framing tension between 2856 ("arises from the gauge connection") and 2858 ("central open question is whether free-energy mechanism selects imaginary phi_tau") is dissolved by a third reading I have missed.

Sources:
- [Wick rotation — nLab](https://ncatlab.org/nlab/show/Wick+rotation)
- [Samuel, "Wick Rotation in the Tangent Space" arXiv:1510.07365](https://arxiv.org/abs/1510.07365)
- [Helleland and Hervik, "Wick rotations and real GIT" arXiv:1703.04576](https://arxiv.org/abs/1703.04576)
- [Helleland and Hervik, "Wick-rotations of pseudo-Riemannian Lie groups" arXiv:1810.12037](https://arxiv.org/abs/1810.12037)
