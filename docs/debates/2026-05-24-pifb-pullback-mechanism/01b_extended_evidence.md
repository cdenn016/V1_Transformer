# Extended Evidence — pifb-pullback-mechanism

Canon harvested by the expert panels beyond the neutral evidence pack. Judges read this file.

## Round 2 (opening) — Red panel

### Canonical bundle-metric form (geometer, gauge-theorist)

- **HandWiki, "Bundle metric"** (https://handwiki.org/wiki/Bundle_metric): the canonical principal-bundle metric is $\hat g = \pi^* g + k\,\omega$, where $\pi^* g$ is the **pullback of the base metric** $g$ on $M$ (the horizontal block), $k$ is the **Ad($G$)-invariant fiber metric**, and $\omega$ is the connection one-form. Requires three inputs: a base metric on $M$, an Ad-invariant fiber metric $k$, and a connection $\omega$. The connection enters the fiber/vertical structure, not a horizontal trace form.
- **Wikipedia, "Kaluza–Klein theory"** (https://en.wikipedia.org/wiki/Kaluza%E2%80%93Klein_theory): 5D line element $ds^2 = g_{\mu\nu}dx^\mu dx^\nu + \phi^2(A_\nu dx^\nu + dx^5)^2$. Horizontal block = base spacetime metric $g_{\mu\nu}$; the gauge connection $A$ appears **only inside the fiber shift** $(A_\nu dx^\nu + dx^5)$, never as a standalone horizontal block. "A metric on the total space invariant under the principal action reduces to a metric together with a gauge field on the base" — the canonical total-space metric is gauge-invariant.
- **Albuquerque, R., "Notes on the Sasaki metric" (2018)** (https://dspace.uevora.pt/rdpc/bitstream/10174/26894/1/NotesonSasakigeometryoftangentbundles.pdf): the Sasaki metric on $TM$ extends a **given base metric $g$** to the tangent bundle via the Levi-Civita connection; horizontal and vertical subbundles orthogonal; horizontal lift carries the base metric; "invariant under isometries." Again: horizontal block = pulled-back base metric, not a connection norm.

### Connection-form transformation / non-tensoriality (gauge-theorist)

- [Nakahara2003 Ch. 10–11], [KobayashiNomizu Vol. I §II–III]: connection one-form satisfies $R_g^* A=\mathrm{Ad}(g^{-1})A$; between sections $A'=g^{-1}Ag+g^{-1}dg$ (inhomogeneous Maurer–Cartan term). Curvature $F=dA+\tfrac12[A,A]$ is the tensorial object; $A$ is not tensorial, so any scalar quadratic in $A$ alone (e.g. $\mathrm{tr}(A_\mu A_\nu)$) is gauge-noninvariant and not a tensor on the total space.

### Fisher metric uniqueness (info-geometer)

- **Wikipedia, "Fisher information metric"** (https://en.wikipedia.org/wiki/Fisher_information_metric): $g_{jk}(\theta)=\mathbb E_{x\sim p(x|\theta)}[\partial_j\log p\,\partial_k\log p]$; "By Chentsov's theorem, the Fisher information metric on statistical models is the only Riemannian metric (up to rescaling) that is invariant under sufficient statistics." Corroborates [Cencov1972] / [external_canon_math.md §1]: a metric failing sufficient-statistic invariance is not a valid information metric.

### Hierarchical active inference — structure localization (variational)

- [FristonEtAl2017, "Active inference: A process theory," *Neural Computation* 29:1–49] (record: https://www.researchgate.net/publication/310627938_Active_Inference_A_Process_Theory): structural/perceptual organization carried by hierarchical/deep generative models; parameter learning and state inference are complementary free-energy-minimizing processes — not a localization of "perceived space" specifically in the parameter-learning tier.
- Da Costa et al., "Active inference on discrete state-spaces: A synthesis," *J. Math. Psychol.* (2020) (https://www.sciencedirect.com/science/article/pii/S0022249620300857): structural learning via Bayesian model reduction / hierarchical depth, not a single slow-parameter tier.

### Falsifiability frame (philosophy-of-science)

- [Popper, *The Logic of Scientific Discovery*, 1959]: demarcation by falsifiability — a contentful claim must specify refutation conditions.

## Round 2 (opening) — Blue panel

### Bundle-metric canonical lineage (geometer)

- **Sasaki, S. (1958)**, "On the differential geometry of tangent bundles of Riemannian manifolds," *Tohoku Math. J.* 10. — Block-diagonal total-space metric: projection a Riemannian submersion, fiber gets induced metric, horizontal/vertical orthogonal, horizontal distribution connection-determined. (WebFetch of the three defining properties confirmed.)
- **Yano, K. & Ishihara, S. (1973)**, *Tangent and Cotangent Bundles*, Marcel Dekker. — Sasaki construction and the connection map $K$.
- **[Bleecker1981]**, **[Frankel2011 Ch. 20]** — Kaluza–Klein total-space metric; the base-tensor representative of the horizontal block is section-dependent (standard, not a defect).

### Note for judges — blue/red conflict on the canonical horizontal block

The red panel cites HandWiki/Wikipedia/Albuquerque for the canonical form $\hat g = \pi^* g + k\,\omega$, in which the connection $\omega$ enters the **vertical/fiber** structure and the horizontal block is the **pulled-back base metric** $\pi^* g$, NOT a connection norm $\kappa(A,A)$. Blue's position: the manuscript's $\mathcal{C}$ carries **no pre-existing base metric** to pull up (the entire program is to *induce* a metric on $\mathcal{C}$ from information — "it from bit"), so the standard $\pi^*g$ horizontal block is unavailable by hypothesis. The manuscript substitutes a connection-built horizontal form $\kappa(A,A)$. This is a **principled deviation** from the textbook bundle metric forced by the no-prior-base-metric setting, not the textbook form. It must be labeled as a novel construction (per external_canon_math.md §4), which the manuscript does ("tw not YM," connection-dependent, gauge-noninvariant, disclosed at :2768). The blue defense rests on this being a *disclosed, principled* substitution, not on it matching the textbook $\pi^*g$ form — which it does not.

### Connection gauge transformation (gauge-theorist)

- **[Nakahara2003 §10.4]**, **[KobayashiNomizu Vol. I §II.1]** — $A\to g^{-1}Ag+g^{-1}dg$; pure-gauge $A=U^{-1}dU\Rightarrow F\equiv 0$; connection-built base scalars generically non-invariant (only curvature invariants are invariant). Same sources as red, opposite emphasis: non-invariance of $\kappa(A,A)$ is expected behavior of a connection-built form, disclosed and routed to the consensus metric.

### Fisher-Rao uniqueness (info-geometer)

- **[AmariNagaoka2000 Ch. 2]**, **[Cencov1972]**, **Nielsen (2020)** *Entropy* 22(10):1100 — Fisher = score outer product; Cencov uniqueness forces the vertical block. The vertical block is the unassailable piece.

### Active inference fast/slow (variational)

- **[FristonEtAl2017]**, **Da Costa et al. (2020)** *J. Math. Psych.* 99:102447, **[Friston2017Graphical]** — fast state inference vs slow parameter (concentration) learning at trial boundaries; slow parameters carry learned structure. Anchors the slow-tier $s$ vs fast-tier $q$ split. Blue concedes (matching red's reading) that the canon supports *structure in slow parameters generically*, NOT a localization of "perceived space" specifically in $G^{(s)}$ over $G^{(p)}$/$G^{(q)}$.
