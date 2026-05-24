# Red Panel Choice — pifb-dimensional-sectors (Phase 2, opening)

Mode: theory + formal math (spectral / linear algebra, dimension counting). Panel=full, 5-of-10.
No `Agent` tool in this environment; the coordinator embodies each lens and writes the memos directly (per dispatch fallback instruction). Memos at `memo_red_<tag>.md`.

| Tag | Why this lens attacks this claim |
|---|---|
| `philosophy-of-science` (mandatory) | Owns the two frame-level failures: free-threshold structure-imposition (Λ_obs, Λ_subthresh tune any sector sizes) and importing the explanandum (identifying the large-λ sector with "perceived spacetime" assumes what is to be explained); also the scope-vacuity frame. |
| `geometer` | Pullback-rank degeneracy: $G_i(c)=\sigma_i^* g_{\mathcal B}$ has rank $\le\dim\mathcal C$; in the framework's only realized construction $\dim\mathcal C=0$, so the spectrum is empty and the three-sector decomposition is vacuous there. Verifies the bookkeeping concession (Lee pullback rank). |
| `info-geometer` | The eigenvalues of a Fisher-pullback are reparameterization-dependent in *magnitude*; "large vs small eigenvalue" is not a coordinate-free predicate, so a threshold on raw $\lambda_a$ does not carve a canonical sector. Verifies $\dim(\mathcal B)=K(K+3)/2$. |
| `numerical-analyst` | A two-threshold partition of a spectrum is well-defined only if there are genuine spectral *gaps*; the hierarchy $\lambda_{\rm obs}\gg\lambda_{\rm sub}\gg\lambda_{\rm int}$ is asserted (Eq. 3061), not produced by any gap theorem (Davis–Kahan / Weyl). Without a demonstrated gap, the three-set partition is arbitrary. |
| `gauge-theorist` | A PSD pullback gives $\lambda_a\ge 0$; a Lorentzian $(1+3)$ signature has a minus sign that cannot come from the spectrum. The temporal sign is imported from the signature section, which carries Debate 2's REMAND, so the $(1+3)$ identification rests on an unresolved postulate. |

Dropped relative to the `theory` default (geometer, info-geometer, variational, gauge-theorist, philosophy-of-science): `variational` — the claim is about the spectral/dimensional bookkeeping of an induced metric, not about ELBO/EM separation or factorization, so the variational lens adds the least here. `transformer-ml` considered for the $\dim\mathcal C=0$ point but that evidence routes through geometer (rank) and philosophy-of-science (scope); gauge-theorist is needed for the signature-import attack, which no other lens covers.
