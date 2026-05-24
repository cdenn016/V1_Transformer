# Memo — debate-expert-numerical-analyst — red — opening — pifb-signature-problem

## Lens
Conditioning, continuum-limit stability, discontinuity of rank-changing maps, scale indeterminacy.

## Steelman of the opposing position
The Sylvester count for the causal-cone metric is exact ($g=\mathrm{diag}(-c_\mathcal{I}^2,h)$ with $c_\mathcal{I}>0$, $h\succ0$ has signature $(-,+,\dots,+)$), the conformal-class ambiguity is disclosed, and the first-order/parabolic tension is named with three candidate fixes — so the causal-cone route is a correctly-stated conditional existence statement.

## My position (in service of red)
The causal-cone route is **vacuous for the framework actually under evaluation**, because its load-bearing Postulate 3 (finite maximal information speed) directly contradicts the framework's own dynamics. The manuscript states (`:2928`) that the E-step natural-gradient update $\mu_i\leftarrow\mu_i-\eta\Sigma_i\nabla_\mu\mathcal{F}$ "implemented in the working code" is the Euler scheme for a first-order natural-gradient flow, whose continuum limit is parabolic and gives *infinite* signal speed. Parabolic equations propagate disturbances instantaneously: the heat kernel is a Gaussian with support everywhere for any $t>0$, so a localized perturbation affects the entire domain immediately (Evans, *PDE*, §2.3). A framework whose dynamics have infinite propagation speed has no causal cone, hence no null cone to read a Lorentzian metric off. The route's central object does not exist in this framework.

The manuscript lists three fixes (`:2929`) — telegraph scaling limit, hyperbolic second-order dynamics, architectural finite-speed constraint — and concedes "none of these is currently realized." So the route is conditional on replacing or re-deriving the framework's dynamics from the ground up. An existence statement conditioned on dynamics the framework does not have is not an existence statement *about the framework*; it is an existence statement about a hypothetical different framework. The Sylvester algebra is correct (concede), but it operates on inputs ($c_\mathcal{I}<\infty$, a quadratic "inner-product-type" cone) that the present dynamics do not supply. The "inner-product type rather than Finsler-norm type" assumption (`:2915`) is additional hidden work: a generic finite-speed influence boundary is a convex cone, not necessarily a quadric, so even granting a finite cone, forcing it to be a *metric* cone is a further unjustified input.

On the $\mathrm{GL}(K,\mathbb{C})$ route, the numerical pathology is the rank-1$\to$rank-2 map itself. $\mathrm{Re}(\cdot)$ applied to the degenerate complex form is discontinuous as a rank operation: the genuine object sits exactly on the measure-zero degenerate locus ($\det=0$, verified), and the projection jumps it to rank 2. Any perturbation that gives the off-diagonal a real part would change the result; the construction is conditioned on the off-diagonal being *purely* imaginary, which the separability ansatz (`:2877`) enforces by hand. Drop separability and $\mathrm{Re}(G_{\tau\tau})=2((\partial_\tau\psi_x)^2-(\partial_\tau\psi_\tau)^2)$ (verified) — the temporal sign is no longer guaranteed negative. The signature is not robust to the ansatz the section calls a mere display simplification.

## Evidence
- Evans, L. C. (2010). *Partial Differential Equations*, 2nd ed., AMS, §2.3 (heat equation): the fundamental solution has support on all of $\mathbb{R}^n$ for every $t>0$ — infinite propagation speed; parabolic equations have no finite characteristic cone, in contrast to hyperbolic (wave) equations (§2.4). Verified via WebFetch (Wikipedia *Heat equation*, orientation): heat-equation solutions involve "instantaneous propagation of a disturbance."
- sympy (executed): causal-cone $g=\mathrm{diag}(-c^2,h_1,h_2,h_3)$ eigenvalues $\{-c^2,h_1,h_2,h_3\}$ — Sylvester count correct (concede). Non-separable $\mathrm{GL}(2,\mathbb{C})$ form: $\mathrm{Re}(G_{\tau\tau})=2((\partial_\tau\psi_x)^2-(\partial_\tau\psi_\tau)^2)$ — sign indefinite when cross-derivatives present.
- Manuscript `:2928`–`:2929`: first-order natural-gradient flow is parabolic, infinite signal speed, "none of these [fixes] is currently realized."

## Newly-discovered canon (for 01b_extended_evidence.md)
- Evans, L. C. (2010). *Partial Differential Equations*, 2nd ed., AMS, §2.3–2.4: parabolic (heat) equations exhibit infinite propagation speed via everywhere-supported fundamental solution; hyperbolic (wave) equations have finite-speed propagation governed by a characteristic cone. The causal-cone route requires the hyperbolic behavior the framework's parabolic flow lacks.
- John, F. (1982). *Partial Differential Equations*, 4th ed., Springer, Ch. 7: classification of second-order PDE and the domain-of-dependence / characteristic-cone distinction between parabolic and hyperbolic types.

## Falsification conditions
My position is wrong if (a) the framework's actual implemented dynamics admit a finite-speed (hyperbolic or telegraph) continuum limit — the section concedes none is realized (`:2929`); or (b) the worked-example signature is robust to dropping separability — sympy shows it is not (the temporal sign can flip).

## Confidence
HIGH on the causal-cone vacuity (manuscript concedes the unrealized dynamics; parabolic infinite-speed is textbook). HIGH on the separability gap: the `:2877` sentence frames the cross-derivative terms as ones "that the displayed result suppresses" — a display simplification, with no statement that their presence can change the sign of $\mathrm{Re}(G_{\tau\tau})$. The sympy shows it can. The omission of the cross-derivatives is disclosed; the consequence for the signature is not.
