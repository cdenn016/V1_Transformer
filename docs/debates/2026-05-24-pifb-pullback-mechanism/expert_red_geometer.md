# Expert Memo — Red / Geometer — pifb-pullback-mechanism (opening)

## Steelman

A Riemannian metric on a manifold $E$ is any smooth, symmetric, positive-definite (0,2)-tensor field. Given the connection-induced splitting $T_{(c,q)}E = H\oplus V$ (:2730), one may declare $H$ and $V$ orthogonal and assign each block an arbitrary positive-definite form. The vertical block is the Fisher-Rao metric $g_{\mathcal B}$; the horizontal block is $\kappa(A_\mu,A_\nu)$, positive-definite for compact $\mathfrak{so}(N)$ under $\kappa=-\mathrm{tr}$ (:2744). The direct sum of two positive-definite blocks on complementary distributions is a positive-definite (0,2)-tensor on $E$. On that minimal reading $g_{E_q}$ is "a Riemannian metric on the total space," and the pullback $G^{(q)}=(\sigma^*)g_{E_q}$ is a well-defined symmetric (0,2)-tensor on $\mathcal C$. The score-only expression then is genuinely a special case ($A=0$), so the manuscript's claim that it furnishes "more than" the $L^2$ pullback is, narrowly, the observation that $g_{E_q}$ carries an extra additive horizontal term.

## Strongest falsification

The manuscript names $g_{E_q}$ a "bona-fide bundle metric" (:2726) and contrasts it with the "merely $L^2$" score pullback. The canonical Kaluza–Klein / Sasaki bundle metric on the total space of a bundle with connection is, in adapted coordinates,
$$ds^2 = \underbrace{g_{ab}(x)\,dx^a dx^b}_{\text{horizontal}} + \underbrace{k_{ij}\,(dy^i + A^i_a dx^a)(dy^j + A^j_a dx^a)}_{\text{vertical, connection enters as the shift}}.$$
The horizontal block is **the pullback of a base metric $g_{ab}(x)$**. The connection $A$ enters only inside the **vertical** block, as the shift that makes the fiber coordinate covariant. The connection-norm-squared $\kappa(A_\mu,A_\nu)$ appears *nowhere* in the canonical construction — it is not a metric block; it is the trace form evaluated on connection components.

The manuscript substitutes $\kappa(A_\mu,A_\nu)$ into the horizontal slot precisely because it has no base metric $g_{ab}$ on $\mathcal C$ to use — the stated purpose of the entire construction is to *manufacture* a metric on $\mathcal C$ by pullback (:2784, "a metric on the noumenal substrate is the pullback ..."). This is circular: the canonical Sasaki/KK horizontal block requires a base metric as input, and the manuscript wants the base metric as output. To break the circularity it inserts $\kappa(A,A)$, which is not the horizontal block of any standard bundle metric. So $g_{E_q}$ is not the Kaluza–Klein/Sasaki object the name "bona-fide bundle metric" invokes; it is a block-diagonal quadratic form whose horizontal block is an ad-hoc connection-dependent additive term. The substantive content the manuscript claims over the $L^2$ pullback reduces to: "add $\kappa(A,A)$ to the score outer product." Whether that added term is legitimate is the gauge-theorist's question; geometrically, it is not the horizontal block of a canonical bundle metric.

The cross term vanishing (sub-point 3) is tautological — $g_{E_q}$ is *defined* block-diagonal in $H\oplus V$ (:2734), so $g_{E_q}(X_H,Y_V)=0$ by fiat, not by a derived gauge-orthogonality theorem. The score-outer-product $=$ Fisher identity (:2759) is the standard Gaussian identity and is conceded.

## External citation

[Bundle metric, HandWiki / Kaluza–Klein theory, Wikipedia] The canonical principal-bundle metric is $\hat g = \pi^* g + k\,\omega\otimes\omega$ where $\pi^* g$ is the **pullback of the base metric** and $k$ is the Ad-invariant fiber metric applied to the connection one-form $\omega$; the 5D KK line element $ds^2 = g_{\mu\nu}dx^\mu dx^\nu + \phi^2(A_\nu dx^\nu + dx^5)^2$ has the base metric $g_{\mu\nu}$ as horizontal block and the connection only in the fiber shift. Standard textbook treatments: [Nakahara2003 Ch. 10–11] (connections, horizontal lift, associated bundles); [KobayashiNomizu Vol. I §II–III] (horizontal lift of a base metric); the Sasaki metric [Albuquerque, "Notes on the Sasaki metric," 2018] is the horizontal-lift extension of a base metric $g$ to $TM$, again with the base metric as the horizontal block.

## Falsification condition

My falsification is wrong if a standard reference exhibits a bundle metric on a total space whose horizontal block is the trace form $\kappa(A_\mu,A_\nu)$ of the connection components (rather than the pullback of a base metric), and calls that the canonical horizontal block. I found none — every canonical construction (KK, Sasaki, [Nakahara2003 Ch. 11]) uses a pulled-back base metric horizontally and puts the connection in the fiber shift. If blue produces such a reference, the "bona-fide" naming survives.

## Newly-discovered canon

- HandWiki, "Bundle metric": "this metric is $\pi^* g + k\omega$"; the principal-bundle metric requires a base metric $g$ on $M$, an Ad($G$)-invariant fiber metric $k$, and a connection one-form $\omega$; the horizontal piece is $\pi^* g$ (pullback of the base metric). URL: https://handwiki.org/wiki/Bundle_metric
- Wikipedia, "Kaluza–Klein theory": 5D ansatz $ds^2 = g_{\mu\nu}dx^\mu dx^\nu + \phi^2(A_\nu dx^\nu + dx^5)^2$; horizontal block is the spacetime metric $g_{\mu\nu}$, connection $A$ enters only in the fiber shift; "a metric on the total space invariant under the principal action reduces to a metric plus a gauge field on the base." URL: https://en.wikipedia.org/wiki/Kaluza%E2%80%93Klein_theory
- Albuquerque, R., "Notes on the Sasaki metric" (2018): the Sasaki metric on $TM$ is built from the Levi-Civita connection of a **given base metric $g$**, with horizontal and vertical subbundles orthogonal and the horizontal lift carrying the base metric; "invariant under isometries." URL: https://dspace.uevora.pt/rdpc/bitstream/10174/26894/1/NotesonSasakigeometryoftangentbundles.pdf
