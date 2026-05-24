# Memo — info-geometer (RED, opening)

## Steelman of the claim
The dimension count is correct. A $K$-dimensional Gaussian family has $K$ mean parameters and $K(K+1)/2$ independent covariance parameters, so $\dim(\mathcal B) = K + K(K+1)/2 = K(K+3)/2$ [Amari & Nagaoka 2000, *Methods of Information Geometry*, Ch. 2, the Gaussian as an exponential family with this parameter count]. For $K=768$: $768\cdot 771/2 = 296{,}064$. The arithmetic (:3013–3017) checks out. I concede the dimension bookkeeping.

## Falsification target
The sector partition rests on the *magnitudes* of the eigenvalues $\lambda_a$ of $G_i(c)$ via two raw-magnitude thresholds (:3026, :3034, :3042). But the magnitude of an eigenvalue of an induced metric is not a coordinate-free quantity. Under a reparameterization of the base $c \mapsto \tilde c = \psi(c)$ with Jacobian $J$, the pullback metric transforms as a (0,2)-tensor, $\tilde G = J^{-\top} G\, J^{-1}$, so the eigenvalues rescale by the squared singular values of $J$. "$\lambda_a > \Lambda_{\rm obs}$" is therefore a statement about the chosen coordinates on $\mathcal C$, not an intrinsic property of the direction $e_a$. The Fisher metric earns its canonical status precisely because it is invariant under sufficient statistics [Čencov 1972; Amari–Nagaoka 2000, Ch. 2, Čencov uniqueness] — but Čencov-invariance is invariance of the *metric tensor*, not invariance of the *numerical eigenvalues* of its pullback to an arbitrary base parameterization. The section reads physical significance ("observable," "perceived spacetime") off a coordinate-dependent number.

What *is* coordinate-free is the *ordering* and *ratios* of eigenvalues at a fixed point, and the rank. A reparameterization-invariant sector criterion would have to be a statement about ratios or gaps (e.g., "$\lambda_a/\lambda_{a+1} > $ some intrinsic ratio"), not about absolute thresholds $\Lambda_{\rm obs}, \Lambda_{\rm subthresh}$ carrying units of the chosen $c$-coordinates. The manuscript uses absolute thresholds (:3026, :3034, :3042) and so does not deliver a coordinate-free partition. The asserted magnitude hierarchy $\lambda_{\rm obs}\gg\lambda_{\rm sub}\gg\lambda_{\rm int}$ (:3061) inherits the same defect: "$\gg$" between absolute eigenvalues is not invariant unless promoted to a ratio statement, which the manuscript does not do.

## Relation to the threshold attack
This sharpens the philosophy-of-science free-parameter point with an information-geometric reason the thresholds *cannot* be principled as written: any absolute-magnitude threshold is reparameterization-covariant, so there is no canonical value to fix it to. The only invariant content is rank and eigenvalue ratios — neither of which the section uses to define the sectors.

## Falsification condition (when RED is wrong here)
RED loses this vector if blue recasts the sector boundaries as reparameterization-invariant quantities (eigenvalue ratios, relative gaps, or a dimensionless criterion tied to an intrinsic scale of $\mathcal C$), and shows the three-sector partition is stable under base reparameterization. As written with absolute thresholds, it is not.

## Newly-discovered canon
- Amari & Nagaoka 2000, *Methods of Information Geometry* (AMS/Oxford, Translations of Mathematical Monographs 191), Ch. 2: Gaussian family parameter count $K(K+3)/2$; Fisher metric and Čencov's uniqueness theorem (invariance under sufficient statistics / Markov morphisms).
- Čencov (Chentsov) 1972, *Statistical Decision Rules and Optimal Inference* (AMS translation 1982): the Fisher metric is the unique (up to scale) monotone metric on a statistical manifold — invariance is of the *tensor*, not of pullback eigenvalue magnitudes under arbitrary base reparameterization.
