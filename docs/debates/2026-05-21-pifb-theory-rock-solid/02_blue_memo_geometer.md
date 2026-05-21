# Blue Memo — geometer

Lens: differential geometry — principal bundles, associated bundles, parallel transport, sandwich product, exponential map.

## Steelman (opposing position)

The §Theory of `Participatory_it_from_bit.tex` is not publication ready because the principal-bundle / associated-bundle apparatus carries unstated structural assumptions: a manifold-section "$q_i: \mathcal{U}_i \to \mathcal{E}_{\mathrm{state}}$" is constructed only on local trivializing patches and the gauge group is taken non-compact $\mathrm{GL}^+(K)$ where the exponential map is not surjective. A reviewer who treats local-trivialization caveats as load-bearing should not accept a paper that asserts global frames informally then walks them back in a paragraph.

## Defense from the geometric lens

Three transport-related load-bearing equations of §Theory match the canonical differential-geometric form to the letter, and the local-trivialization caveats are stated where they belong rather than papered over. The geometric core is rock solid; the failure modes the steelman raises are stated explicitly in §Theory and are acceptable scope statements rather than gaps.

(1) **Sandwich product for covariance transport.** Line 614 of the manuscript states
$$\rho_{\mathrm{state}}(g)\cdot\mathcal{N}(\mu,\Sigma)=\mathcal{N}(g\mu,\,g\Sigma g^\top).$$
This is the canonical action on the associated bundle for a Gaussian fiber, matching `[Nakahara2003 §10.3]` (transport of a covariant 2-tensor / bilinear form under a linear representation) and `external_canon_math.md:90–102` ("The sandwich identity (THIS IS THE STANDARD)"). The user identifies $\Sigma$ as a (2,0)-tensor under the $\mathrm{GL}(K)$ action; the sandwich rule with $g$ on both sides (not $g^{-\top}\Sigma g^{-1}$) is internally consistent with that identification and is the form that appears in `KingmaWelling2014 App. B` for Gaussian KL closure under linear reparameterization. No flag.

(2) **GL-correct precision transport with explicit O(d) caveat.** Line 1894 states
$$\tilde{\Lambda}_{q_k} := (\Omega_{ik}\Sigma_k\Omega_{ik}^\top)^{-1} = \Omega_{ik}^{-\top}\Lambda_{q_k}\Omega_{ik}^{-1},$$
together with the textual statement at line 1897: *"for orthogonal transport $\Omega \in \mathrm{O}(d)$ the two laws coincide ($\Omega^{-\top} = \Omega$); for general $\mathrm{GL}(d)$ the dual transport is essential, and substituting the orthogonal form into the GL setting is a frequent source of error."* This is the dual-tensor transport rule for (0,2)-tensors / inverse bilinear forms — `external_canon_math.md:99` lists exactly this case ("(0,2)-tensor: $T \to \rho(g^{-\top}) T \rho(g^{-1})$ … For the orthogonal group … reduces to $T \to \rho(g)^\top T \rho(g)$"). The manuscript pre-empts standard pitfall #2 of `external_canon_math.md:131` (missing-transpose / orthogonal-form substitution into GL) with an in-line warning. No flag.

(3) **Exponential-map / local-trivialization caveats.** Lines 583–587 state that the exponential parameterization $U_i(c)=\exp[\phi_i(c)]$ is only locally bijective and that the global bundle is the Čech limit of the local patches with the cocycle $\Omega_{ij}\Omega_{jk}=\Omega_{ik}$ on triple overlaps. This matches `[Nakahara2003 Ch. 9–10]` and `external_canon_math.md:111–112`, which says explicitly that $\exp:\mathfrak{gl}^+(K)\to\mathrm{GL}^+(K)$ is *not* surjective in general (counterexample: a Jordan block with negative eigenvalue). The user's framework names this restriction, offers the polar/Cartan decomposition workaround, and notes that the empirical simulations use connected-compact $G=\mathrm{SO}(N)$ where the issue does not arise. This is correct scope discipline, not a gap.

## Falsification condition for the claim

The geometric defense breaks if (a) any §Theory equation invokes the sandwich product with the wrong-side conjugation on a quantity that is identified elsewhere in the manuscript as a (0,2)-tensor, or (b) any §Theory step uses $\exp(A+B)=\exp(A)\exp(B)$ to combine $\phi_i$ and $\phi_j$ without commutation. I have not found either. The closest near-miss is line 1897's secondary identities $\tilde\Lambda_k\Omega_{ik} = \Omega_{ik}\Lambda_k$, which the manuscript explicitly flags as O(d)-only. If a downstream §Theory derivation uses these O(d)-only identities outside the O(d) limit without re-flagging, the geometric defense fails — this is the specific falsification condition I can stake.

## Newly-discovered canon

None beyond what `external_canon_math.md` already records. `[Frankel2011 Ch. 17]` on Čech cocycles and the standard fact that "a globally defined Lie-algebra-valued frame field exists iff the bundle is trivializable" (line 583) is in `external_bibliography.md`; the user's wording at line 583 is the textbook statement.
