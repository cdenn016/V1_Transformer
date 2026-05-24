# Evidence Pack — pifb-dimensional-sectors (Debate 4)

## Manuscript references (Attention/Participatory_it_from_bit.tex)

- `:3001–3007` — spectral decomposition $G_i(c)=\sum_a \lambda_a(e_a\otimes e_a)$, $\lambda_1\ge\dots\ge\lambda_n\ge0$; each $\lambda_a$ = information flux in direction $e_a$; large = observable, small = subthreshold/internal.
- `:3013–3017` — $\dim(\mathcal{B})=K+K(K+1)/2=K(K+3)/2$; $K=768\Rightarrow 296{,}064$.
- `:3019` — "What lives where: base vs fiber dimensions." $G_i(c)=\sigma_i^* g_{\mathcal{B}}|_{q_i(c)}$ is a tensor on $\mathcal{C}$, $n\times n$ with $n=\dim\mathcal{C}$, at most $n$ eigenvalues; $d\sigma_i:T_c\mathcal{C}\to T_{q_i}\mathcal{B}$ maps $n$-dim into the much larger fiber tangent; "internal sector / vast majority of dimensions" refers to UNsampled fiber directions, not eigen-directions of $G_i$.
- `:3023–3045` — observable $\mathcal{D}_{\text{obs}}=\{\lambda_a>\Lambda_{\text{obs}}\}$ (≈4 for human agents, $(1+3)$ Lorentzian under signature postulates :3029); subthreshold $\{\Lambda_{\text{subthresh}}<\lambda_a\le\Lambda_{\text{obs}}\}$; internal $\{\lambda_a\le\Lambda_{\text{subthresh}}\}$. :3029 — temporal sign NOT intrinsic to the PSD spectral decomposition; imported from the signature postulates; inherits that section's open-problem status.
- `:3047–3065` — three-sector sum; $|\mathcal{D}_{\text{obs}}|\ll|\mathcal{D}_{\text{subthresh}}|\ll|\mathcal{D}_{\text{internal}}|$; $\lambda_{\text{obs}}\gg\lambda_{\text{subthresh}}\gg\lambda_{\text{internal}}\approx0$; "we may suppose the observable sector corresponds to the $(1+3)$ we perceive."
- `:3067–3071` (Hypothesized $(3+1)$, Speculative) — $|\mathcal{D}_{\text{obs}}|=4$ hypothesized; three candidate mechanisms (neural-architecture, sensory bottleneck, evolutionary); "toy model demonstration that dimensional structure CAN in principle emerge"; "does not explain why exactly three spatial dimensions"; "makes no quantitative predictions about measured spacetime properties that could be tested experimentally."
- `:3073` — two distinct senses of "invisible" (base-direction subthreshold/internal eigenvalues vs unsampled fiber directions); explicitly says not to conflate them.

## Canon facts to verify

- Pullback of a (0,2)-tensor by a map $\sigma:\mathcal{C}\to\mathcal{B}$ has rank $\le\dim\mathcal{C}$, with at most $\dim\mathcal{C}$ nonzero eigenvalues; a pullback of a PSD form is PSD [Lee, *Introduction to Smooth Manifolds*, Ch. 11 (pullbacks) / Ch. 13 (Riemannian)].
- Gaussian statistical-manifold dimension $K+K(K+1)/2$ — standard (mean + symmetric covariance). Verify the arithmetic for $K=768$.
- Spectral theorem for symmetric PSD matrices ($\lambda_a\ge0$, orthonormal eigenbasis) — elementary.
- `external_canon_math.md` for Fisher-Rao PSD and the score-form metric.

## What this evidence does NOT settle

- Whether the two thresholds $\Lambda_{\text{obs}},\Lambda_{\text{subthresh}}$ are principled or free parameters that can be tuned to produce any sector sizes (does "$|\mathcal{D}_{\text{obs}}|\ll|\mathcal{D}_{\text{subthresh}}|\ll|\mathcal{D}_{\text{internal}}|$" follow from anything, or is it assumed?).
- Whether identifying the large-eigenvalue (observable) sector with "perceived spacetime" is a defensible reading or imports unargued the very thing to be explained.
- Whether "we may suppose the observable sector corresponds to the (1+3) we perceive" (:3065) is adequately hedged given the explicit "(Speculative)" / "no quantitative predictions" labels at :3067–3071, or whether the supposition leaks into later sections as if established.
- Whether the base manifold $\mathcal{C}$ even HAS a dimension $n$ large enough to "accommodate" three sectors in the framework's actual constructions (the transformer limit is zero-dimensional; what is $\dim\mathcal{C}$ in the multi-agent runs?) — a scope/consistency question.
