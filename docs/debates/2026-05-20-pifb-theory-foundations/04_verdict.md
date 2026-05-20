# Verdict — pifb-theory-foundations

## Outcome

RED_WINS_NARROW

The §Theory foundations at `Attention/Participatory_it_from_bit.tex:483-911` are mathematically rigorous on the load-bearing claims, but one citation at line 510 is scope-misaligned and must be patched. Blue's defense of the bundle constructions, the two-roles framing, and the Appendix A cross-scale shadow reduction holds on the evidence. Red lands a single, surgical strike on the Cencov citation at line 510 that blue explicitly concedes in its "Pre-empted attacks" section.

## Decisive evidence

Two converging primary sources establish that Cencov's 1972/1982 theorem covers finite sample spaces only, and that the continuous-parametric extension required by the manuscript's Gaussian families at `Participatory_it_from_bit.tex:512-518` was supplied by a separate body of work that the manuscript does not cite.

First, Lê (2017), "The uniqueness of the Fisher metric as information metric," *Ann. Inst. Statist. Math.* (arXiv:1306.1465) §1: "In 1972 Chentsov proved that on statistical models $(M, \Omega, \mu, p)$ associated with finite sample spaces $\Omega$ the Fisher metric $g^F$ is a unique metric, up to a multiplicative constant, that satisfies (2)." The 2013/2017 paper exists precisely to fill the continuous-sample-space gap.

Second, Ay, Jost, Lê, Schwachhöfer (2017), *Information Geometry*, *Ergebnisse der Mathematik und ihrer Grenzgebiete* Vol. 64, Springer, supplies the generalization to parametrized measure models including continuous-sample-space cases — the canonical reference required when applying the uniqueness theorem to Gaussian families.

The manuscript at `:510` cites only `\cite{Cencov1982}` for a uniqueness claim that is operationally applied to Gaussian families (continuous parametric models) at `:512-518`. Verification by grep over lines 495-534: Cencov1982 is the only uniqueness citation in §Statistical Manifolds; Ay-Jost-Lê-Schwachhöfer and Lê do not appear anywhere in the manuscript. Blue's opening explicitly concedes the scope mismatch ("Granted as a citation-tightening point") and confirms the AJLS 2017 extension is the appropriate co-citation. Blue offers no counter-citation showing Cencov 1982 covers continuous parametric families directly.

R2 (dual-role symbol) is defused by the explicit Two-Roles paragraph at `:557` and the residual-subgroup invariance computation at `:565`, where $\mathrm{tr}(g^{-1}A_\mu g \cdot g^{-1}A_\nu g) = \mathrm{tr}(A_\mu A_\nu)$ follows by trace cyclicity — a one-line proof under the constant-per-agent subgroup blue identifies. Red itself classifies this as "a secondary, weaker strike — not load-bearing for the verdict."

R3 (Appendix A cross-scale shadow reduction) is conceded by red after verification against Lemmas `aug_joint_welldefined` (`:4436-4443`) and `shadow_mf_optimum` (`:4481-4502`).

## Reasoning

The Cencov scope citation at `:510` is the only place where red lands a strike under the source-of-truth rule. Red provides primary-source citations naming the finite-sample-space restriction in Cencov's original theorem and naming the canonical continuous-parametric extension (AJLS 2017, Lê 2017). Blue defends every other foundation — base manifold, principal bundle, associated bundle, agent sections, two roles, exp-map non-compactness, associated-bundle convention, Appendix A reduction — with primary-source citations (Nakahara 2003, Frankel 2011, Kobayashi-Nomizu 1963, Amari-Nagaoka 2000, Lauritzen 1996, Wainwright 2008, Hall 2015, Donnelly-Freidel 2016, Bartlett-Rudolph-Spekkens 2007, Vanrietvelde 2020, Rovelli 1996), and these defenses survive. R2 and R3 do not land; R1 lands narrowly. Because R1 is conceded by blue and is correctable by a one-line citation patch rather than a structural revision, the verdict is RED_WINS_NARROW rather than RED_WINS — the underlying mathematical claim that the Fisher-Rao metric is unique up to scaling on the manuscript's Gaussian families is correct under the AJLS 2017 extension; only the source citation is misaligned with the operative scope.

This is not a case for REMAND: both sides agree on the substantive math, both sides agree on the remedy, and the action item is a single bibliographic addition at one manuscript line. It is not a case for OUT_OF_SCOPE: the §Theory foundations subsections are the right level of abstraction for the framework, and the rest of red's attack surface (R2, R3) has been adjudicated within the existing claim.

## Action

Add Ay-Jost-Lê-Schwachhöfer (2017), *Information Geometry*, *Ergebnisse der Mathematik und ihrer Grenzgebiete* Vol. 64, Springer, as a co-citation at `Participatory_it_from_bit.tex:510` alongside `\cite{Cencov1982}`. Optionally also cite Lê (2017) "The uniqueness of the Fisher metric as information metric," *Ann. Inst. Statist. Math.* (arXiv:1306.1465). Qualify the scope of each in a brief parenthetical or footnote: Cencov 1972/1982 for finite sample spaces, AJLS 2017 / Lê 2017 for the extension to continuous parametric models including Gaussian families. The replacement sentence in the same prose register as the existing line is:

> The metric is unique up to scaling as the only Riemannian metric on probability spaces invariant under sufficient statistics — Cencov's theorem for finite sample spaces~\cite{Cencov1982}, extended to continuous parametric models including Gaussian families by Ay-Jost-Lê-Schwachhöfer~\cite{AyJostLeSchwachhofer2017} and Lê~\cite{Le2017}.

Add the two bibliography entries (AJLS 2017 Springer, Lê 2017 *AISM*) to the manuscript's `.bib` file. The internal canon entry at `.claude/agents/vfe-knowledge/external_canon_math.md:24` and the corresponding bibliography entry at `external_bibliography.md:20` should be updated with the same scope distinction for consistency. No other manuscript line requires revision; the substantive information-geometric backbone of the framework is sound.

A secondary editorial note for the author's discretion: at `:561` the phrase "the discrete analog of three well-developed mechanisms" implies isomorphism with edge modes, quantum reference frames, and relational interpretation, while the actual technical move is the residual-subgroup invariance at `:565`. Replacing "the discrete analog of" with "consonant with" or "in the spirit of" would more accurately reflect that the appeal is to suggestive antecedents rather than to structurally identical constructions. This is style, not rigor, and does not alter the verdict.
