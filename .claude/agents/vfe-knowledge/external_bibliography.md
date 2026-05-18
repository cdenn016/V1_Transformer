# External Bibliography — Standard References

These are the load-bearing sources both agents treat as canon. Citations elsewhere in `external_canon_*.md` files reference this bibliography by short tag (e.g., `[Amari2016]`). For specific claims that need verification, agents should fetch the actual source via WebFetch or arxiv-database — these summaries are pointers, not substitutes.

## Free Energy Principle / Active Inference

- **[Friston2010]** Friston, K. (2010). "The free-energy principle: a unified brain theory?" *Nature Reviews Neuroscience* 11(2): 127–138. The canonical statement. F = E_q[log q(s) - log p(o,s)] = KL(q ‖ p(s|o)) - log p(o). Minimizing F bounds surprise.
- **[FristonEtAl2017]** Friston, K., FitzGerald, T., Rigoli, F., Schwartenbeck, P., Pezzulo, G. (2017). "Active inference: A process theory." *Neural Computation* 29: 1–49. Process-level formulation with policy selection via expected free energy G.
- **[Friston2017Graphical]** Friston, K. et al. (2017). "Graphical brain: Belief propagation and active inference." *Network Neuroscience* 1(4): 381–414. Hierarchical / nested formulation.
- **[ParrPezzuloFriston2022]** Parr, T., Pezzulo, G., Friston, K. (2022). *Active Inference: The Free Energy Principle in Mind, Brain, and Behavior*. MIT Press. The textbook. Has the standard expected-free-energy decomposition G = epistemic + pragmatic.
- **[Bogacz2017]** Bogacz, R. (2017). "A tutorial on the free-energy framework for modeling perception and learning." *J. Math. Psych.* 76: 198–211. Pedagogical introduction.
- **[Millidge2021]** Millidge, B., Tschantz, A., Buckley, C. (2021). "Predictive coding approximates backprop along arbitrary computation graphs." *Neural Comp.* 33(6). Connects predictive coding to backprop.
- **[Ramstead2020]** Ramstead, M. et al. (2020). "Variational ecology and the physics of sentient systems." *Phys. Life Rev.* 35.

## Information Geometry

- **[AmariNagaoka2000]** Amari, S. & Nagaoka, H. (2000). *Methods of Information Geometry*. AMS Translations of Mathematical Monographs 191. The standard textbook. Statistical manifolds, dual connections, Fisher metric.
- **[Amari2016]** Amari, S. (2016). *Information Geometry and Its Applications*. Springer. Modern treatment.
- **[Amari1998]** Amari, S. (1998). "Natural gradient works efficiently in learning." *Neural Computation* 10(2): 251–276. Foundational paper on natural gradient as steepest descent on the statistical manifold.
- **[Cencov1972]** Cencov, N. N. (1972; English transl. 1982). *Statistical Decision Rules and Optimal Inference*. AMS. **The uniqueness theorem:** the Fisher information metric is the unique Riemannian metric on a statistical manifold invariant under sufficient statistics.
- **[Nielsen2020]** Nielsen, F. (2020). "An elementary introduction to information geometry." *Entropy* 22(10): 1100.
- **[KullbackLeibler1951]** Kullback, S. & Leibler, R. A. (1951). "On information and sufficiency." *Annals of Math. Stat.* 22(1): 79–86. Original KL paper.

## Differential Geometry / Gauge Theory

- **[Nakahara2003]** Nakahara, M. (2003). *Geometry, Topology and Physics* (2nd ed.). IOP Publishing. Standard physics-oriented reference. Chapter 10: fiber bundles and gauge theory. Chapter 11: connections on fiber bundles.
- **[Frankel2011]** Frankel, T. (2011). *The Geometry of Physics: An Introduction* (3rd ed.). Cambridge. Comprehensive treatment of bundles, connections, gauge theories.
- **[KobayashiNomizu]** Kobayashi, S. & Nomizu, K. (1963, 1969). *Foundations of Differential Geometry* (Vols. I & II). Wiley. The mathematical reference; rigorous treatment of connections, holonomy, structure equations.
- **[Bleecker1981]** Bleecker, D. (1981). *Gauge Theory and Variational Principles*. Addison-Wesley. (Now Dover reprint.) Gauge theory specifically tied to variational principles.
- **[Lee2013]** Lee, J. M. (2013). *Introduction to Smooth Manifolds* (2nd ed.). Springer. Standard smooth-manifold reference.
- **[doCarmo1992]** do Carmo, M. P. (1992). *Riemannian Geometry*. Birkhäuser. Standard Riemannian reference.

## Variational Inference / VAEs

- **[BleiKuckelbirgJordan2017]** Blei, D. M., Kucukelbir, A., McAuliffe, J. D. (2017). "Variational inference: A review for statisticians." *JASA* 112(518): 859–877. Canonical modern survey.
- **[KingmaWelling2014]** Kingma, D. P. & Welling, M. (2014). "Auto-encoding variational Bayes." *ICLR*. The VAE paper. Reparameterization trick.
- **[JordanGhahramaniJaakkolaSaul1999]** Jordan, M. I., Ghahramani, Z., Jaakkola, T., Saul, L. (1999). "An introduction to variational methods for graphical models." *Machine Learning* 37: 183–233. Foundational tutorial.

## Attention / Transformers / Geometric Deep Learning

- **[Vaswani2017]** Vaswani, A. et al. (2017). "Attention is all you need." *NeurIPS*. The transformer paper. Scaled dot-product `softmax(QKᵀ/√d_k)V`, multi-head, positional encoding.
- **[Bronstein2021]** Bronstein, M., Bruna, J., Cohen, T., Veličković, P. (2021). "Geometric deep learning: Grids, groups, graphs, geodesics, and gauges." *arXiv:2104.13478*. Synthesis of GDL including gauge-equivariant networks.
- **[CohenGeigerKohlerWelling2018]** Cohen, T. et al. (2018). "Spherical CNNs." *ICLR*. Group-equivariant convolution.
- **[Tsai2019]** Tsai, Y.-H. et al. (2019). "Transformer dissection: An unified understanding for transformer's attention via the lens of kernel." *EMNLP*. Kernel-method view of attention.
- **[Katharopoulos2020]** Katharopoulos, A. et al. (2020). "Transformers are RNNs: Fast autoregressive transformers with linear attention." *ICML*.
- **[Ramsauer2021]** Ramsauer, H. et al. (2021). "Hopfield networks is all you need." *ICLR*. Attention as modern Hopfield update.
- **[Su2024RoPE]** Su, J. et al. (2024). "RoFormer: Enhanced transformer with rotary position embedding." *Neurocomputing* 568. RoPE original.

## Optimization on Manifolds / Retractions

- **[AbsilMahonySepulchre2008]** Absil, P.-A., Mahony, R., Sepulchre, R. (2008). *Optimization Algorithms on Matrix Manifolds*. Princeton. Standard reference for retractions, vector transports.
- **[Bonnabel2013]** Bonnabel, S. (2013). "Stochastic gradient descent on Riemannian manifolds." *IEEE Trans. Auto. Control* 58(9): 2217–2229.
- **[BhatiaJainLim2019]** Bhatia, R., Jain, T., Lim, Y. (2019). "On the Bures–Wasserstein distance between positive definite matrices." *Expo. Math.* 37(2): 165–191. SPD manifold geometry.

## Implicit Differentiation / Fixed-Point Methods

- **[BaiKolterKoltun2019]** Bai, S., Kolter, J. Z., Koltun, V. (2019). "Deep equilibrium models." *NeurIPS*. Implicit-function theorem for fixed-point networks. **This is the canonical reference for IFT-style gradients through fixed points; relevant for `em_mode='ift_phi'`.**
- **[Krantz2002]** Krantz, S. G. & Parks, H. R. (2002). *The Implicit Function Theorem: History, Theory, and Applications*. Birkhäuser.

## RoPE / Rotary Position Embeddings

- See [Su2024RoPE] above.

## Coverage gaps — extend on demand

The three external-canon files in this directory cover information geometry, differential geometry / gauge theory, FEP / active inference / variational inference, attention / transformers / GDL, and manifold optimization. They do **not** cover the following topics that appear in the user's manuscripts:

- **Renormalization group (RG).** `GL(K)_supplementary.tex` Appendix F discusses an RG universality conjecture. Standard refs to fetch on demand: Wilson 1971 RG papers; Kadanoff block-spin; Cardy *Scaling and Renormalization in Statistical Physics* (1996); Polchinski 1984 on exact RG.
- **Symmetry breaking and effective field theory.** `GL(K)_supplementary.tex` Appendix G. Standard refs: Weinberg *Quantum Theory of Fields* Vol. II; Peskin & Schroeder *Introduction to QFT* on spontaneous symmetry breaking; for finite-dim and condensed-matter analogues, Chaikin & Lubensky *Principles of Condensed Matter Physics*.
- **Topology of holonomy / monodromy.** Standard refs: Steenrod *Topology of Fibre Bundles*; Husemoller *Fibre Bundles*; for the holonomy group structure theorem, Kobayashi-Nomizu Vol. I.
- **Wheeler's "it from bit" tradition.** `Participatory_it_from_bit.tex` engages this directly. Standard refs: Wheeler 1990 "Information, Physics, Quantum" (in *Complexity, Entropy, and the Physics of Information*, ed. Zurek); Wheeler 1983 *Beyond the Black Hole*; subsequent: Lloyd *Programming the Universe* (2006); Verlinde "On the Origin of Gravity and the Laws of Newton" (2011); Carroll-Singh on it-from-qubit.
- **Lahav-Neemeh cognitive reference frames.** `Participatory_it_from_bit.tex` references this. Fetch their paper(s) directly when reviewing that section — not standard reading; verify the user's representation of their work.
- **Kant on space/time as forms of intuition.** Philosophical background only; not subject to mathematical review.
- **Sociological belief-dynamics models** (DeGroot, Friedkin-Johnsen, bounded confidence, Social Impact Theory, Hegselmann-Krause) — needed for `belief_inertia_unified.tex`. Standard refs: Castellano-Fortunato-Loreto 2009 RMP statistical-physics review of social dynamics; original papers DeGroot 1974, Friedkin-Johnsen 1990, Hegselmann-Krause 2002.

When reviewing manuscript sections that touch these topics, fetch the relevant standard reference via WebFetch / arxiv-database / WebSearch rather than guessing. Mark findings `[citation pending verification]` if the source can't be retrieved.

## How to use this bibliography

When making a finding:

- Tag at the source level: `[Nakahara2003]`, `[Friston2010]`, `[Vaswani2017]`.
- **Do not append specific equation or section numbers unless you have verified them** via WebFetch (for papers/preprints) or by reading the actual document. The canon files include section/equation pointers as starting points — these are best-effort and have not been independently verified. Citing them with false specificity in a finding makes the agent itself the source of citation drift.
- If you cannot retrieve a source, mark the finding with `[unverified — cite later]` rather than asserting a citation you cannot back up.
- WebFetch is appropriate for arXiv preprints and papers with public PDFs. Wikipedia is acceptable for orientation only, never as load-bearing citation.
- Books not freely available online (Nakahara, Amari & Nagaoka, Kobayashi-Nomizu, Frankel, Hall, etc.) cannot be verified by the agent at audit time — cite the book at source level (`[Nakahara2003]`), include the topic ("standard treatment of associated-bundle parallel transport"), and trust that the user can verify against their own copy.
