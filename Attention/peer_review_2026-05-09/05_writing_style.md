# Peer Review 05 — Writing Style, Structure, and Exposition

**Reviewer scope:** Flow, internal notation consistency, redundancy, banned words, structural pacing, reader navigability.
**Manuscript:** `Attention/Participatory_it_from_bit.tex` (4598 lines, 11 sections, ~80 subsections, ~80 subsubsections, 198 labels, ~147 unique cross-references, 91 citation calls).
**Mode:** Read-only review. No edits to manuscript.

---

## 0. Mechanical scans (banned-pattern grep results)

### 0.1 Project-banned words (case-insensitive grep)

| Pattern               | Count | Locations |
|----------------------|------:|-----------|
| `key insight`        | 0     | —         |
| `crucially`          | 0     | —         |
| `critically`         | 0     | —         |
| `notably`            | 0     | —         |
| `importantly`        | 0     | —         |
| `interestingly`      | 0     | —         |
| `fundamentally`      | 0     | —         |
| `in particular`      | 0     | —         |
| `leverages`          | 0     | —         |
| `underscores`        | 0     | —         |
| `worth noting`       | 0     | —         |
| `it's worth noting`  | 0     | —         |

**Verdict on banned-word policy:** Manuscript is clean. The previous round of style cleanup has held. (Note: the phrase "It is worth being explicit about ..." appears at lines 935 and 1525. This is *not* on the banned list, but it is functionally a hedge of the same family as "worth noting" and may deserve elimination on prose-economy grounds — see Finding F.10.)

### 0.2 LaTeX spacing macros (banned per CLAUDE.md)

| Macro | Count | Locations |
|------|------:|-----------|
| `\,` | 4     | Lines 399, 453, 454, 1181 |
| `\!` | 2     | Lines 602, 603 |
| `\;` | 1     | Line 2332 |

All seven instances are inside math mode and several are decorative spacing inside `\Omega_{ij}\,q_j` (lines 399, 453, 454) and `\Gamma\!\left(...\right)` (lines 602, 603). The line-2332 instance `G_{i,\mu\nu}\big(c;\; U_i \mapsto U_i \cdot g\big)` and the line-1181 inline math also use them. **All seven must be removed** per project policy.

### 0.3 Horizontal-rule (`---`) markdown ban

No raw `---` separators in the body. The `\toprule` / `\midrule` / `\bottomrule` calls (lines 2619, 2621) are booktabs LaTeX commands inside a `tabular` environment and are not the banned markdown horizontal rule. **Clean.**

### 0.4 Self-referential drafting history

Greps for `earlier draft`, `previous version`, `previously claimed`, `corrected reading`, `in the present version`, `we previously`, `our earlier`, `prior version`: **no matches**. Manuscript does not narrate its own revision history. **Clean.**

### 0.5 Itemize/enumerate audit

19 list environments total (13 itemize + 6 enumerate). Sizes range from 4 to 11 lines; none are pathologically long. The most prominent are:
- enumerate at 102–109 (Introduction, scoping the three "levels" of claims)
- itemize at 549–553 (caveats on the principal-bundle assumption)
- enumerate at 2214–2220 (postulates required for indefinite pullback)
- enumerate at 2806–2816 (Wheeler–framework correspondence; 11 lines, the longest)
- itemize/enumerate clusters at 2686–2778 and 3036–3040 (Results phase descriptions)

Most lists are defensible (postulates, numbered correspondences, structured caveats). Two flagged for possible conversion to prose: see F.21.

### 0.6 Citation style

91 `\cite{...}` calls; **0** `\citep` and **0** `\citet`. The manuscript uses bare `\cite` uniformly. This is internally consistent but means the bibliography style must be `unsrt`/`plain` style or that `\cite` has been redefined to behave parenthetically — verify with the journal class. **Consistent, but flag for journal compliance.**

### 0.7 Cross-reference health

198 `\label` definitions and ~147 unique `\ref` targets. No `??` (broken-ref) markers found in the source. Forward references are pervasive but the introductory roadmap (line 168) declares them up front, which is acceptable convention for a long theoretical paper.

---

## 1. Structural / pacing findings

### F.1 — Manuscript length and density (MAJOR)
**Location:** Whole document, 4598 lines (~80 subsections, ~80 subsubsections).
**Defect:** This is two papers wearing one cover. The structural spine — Theory (172–1517), Mass-as-Stiffness (1519–1717), Implementation/Participatory (1718–2076), Pullback/Spacetime (2077–2585), Results (2586–3151), Discussion (3152–3541), Open Problems (3542), Conclusion (3598), plus 5 appendices — covers gauge-theoretic active inference, an emergent-spacetime program, a transformer-architecture derivation, a consciousness theory, *and* a philosophy-of-science chapter. The reader has lost the thread by Section 6 because the "what is being claimed?" load is constantly rebased.
**Suggested fix:** Split. The natural cleavage is between (a) the gauge-VFE framework + transformer derivation + multi-agent simulation (publishable as a single ML/information-geometry paper), and (b) the spacetime-emergence + consciousness + Kantian-philosophy program (a companion theoretical paper). If splitting is non-negotiable, then at minimum cut the candidates listed in F.5–F.8.

### F.2 — Mass/stiffness section is structurally orphaned (MAJOR)
**Location:** §3 "Statistical Precision as Configuration-Space Stiffness" (1519–1717).
**Defect:** This 200-line section sits between Theory and Implementation, but it neither develops the framework's machinery nor advances the empirical program. Its content (Hessian = stiffness, plus a postulated kinetic metric) is referenced later by Section 6 (`sec:phenomenological_interpretation`) and Section 7 (Discussion: "Inertial Mass and Gravitational Effects"). Placing it before Implementation and Pullback breaks the reader's expectation of the bundle-theoretic narrative and loads them with caveats ("This is a postulate, not a consequence of $\mathcal{F}$" at line 1705) before the basic apparatus is fully in place.
**Suggested fix:** Move §3 to immediately after §6 (Pullback/Spacetime) or fold its empirical content into the mass-precision discussion of §7. The first-variations / second-variations machinery (1560–1689) could go to an appendix; only the kinetic-metric postulate and the Hessian-as-stiffness reading need to live in main text.

### F.3 — Implementation section is mis-titled (MAJOR)
**Location:** §4 "Implementation" (1718–2076).
**Defect:** "Implementation" suggests code/algorithm. The section actually contains theoretical content about meta-agent formation, RG construction, and bottom-up emergence (1779–1992). Real implementation details live in Methods (3625) and the Results section (2586). Naming is misleading and contributes to navigation confusion.
**Suggested fix:** Rename to "Multi-Scale Structure and Meta-Agent Formation" or absorb into §2 Theory as a final subsection, with a separate short "Computational Implementation" section preceding Results.

### F.4 — Two competing "It from Bit" sections (MAJOR)
**Location:** §5 "It From Bit: The Pullback Construction" header at 2077, then `\subsection{It From Bit: The Pullback Construction}` again at 2077 immediately after a `\section{}` at 2077.
**Defect:** The pullback construction is the manuscript's central technical move toward Wheeler's program, but it is buried under the umbrella section title without a transition that frames it as such. The `\subsection{The Pullback Mechanism}` at 2084 is the actual entry point. Reader sees "It From Bit" twice in three lines.
**Suggested fix:** Make the `\section{}` at 2077 the explicit pullback section ("The Pullback Construction: Geometry from Information") and drop the duplicate subsection title.

### F.5 — Discussion §7 is too discursive (MAJOR)
**Location:** §7 Discussion (3152–3541). 16 subsections covering: kinetic-term archaeology, gravity, macroscopic objects, quantum systems, Wheeler's program, transformers, language, gauge-curvature conjecture, model alignment, qualia, hard-problem-of-consciousness, meta-agent consciousness, Lahav-Neemeh convergence, philosophy of science, Kuhnian revolutions, scientific knowledge limits, future directions.
**Defect:** This is an essay collection appended to a research paper. Each subsection raises live questions deserving a paragraph, not a subsubsection.
**Suggested fix:** Cut by ~50%. Strong candidates for cutting wholesale or moving to a companion paper:
- §7.7 "Implications for Language and Cognition" (3284) — the Gauge Curvature Conjecture is interesting but not derived; orphan.
- §7.8 "Gauge Invariance as Cognitive Consensus Requirement" (3311) — repeats the consensus-metric argument from §6 in philosophical register.
- §7.9 "Consciousness and Hierarchical Information Integration" (3392) and §7.10 "Independent Convergence with Lahav and Neemeh" (3443) — the consciousness program is a separate paper.
- §7.11 "Philosophy of Science: Knowledge as Collective Free Energy Minimization" (3484) — Kuhnian revolutions and theory-choice are not the contribution of this paper.

### F.6 — Open Problems §8 duplicates Discussion §7 (MAJOR)
**Location:** §8 Critical Open Problems (3542–3597), seven subsections (Lorentzian signature, within-species pullback agreement, dimensional structure, scaling/phase transitions, quantum extension, experimental validation, computational optimization).
**Defect:** Each of these has been called out as a limitation or open question multiple times within Sections 2–7 already. §8 reads as a litany rather than a synthesis.
**Suggested fix:** Compress §8 into a single subsection of the Conclusion (~one paragraph per open problem), or remove and rely on the in-line "open" flags scattered throughout the body.

### F.7 — Roadmap (line 168) lists 8 sections; manuscript has 11 + appendices (MINOR)
**Location:** Line 168 ("Section~\ref{sec:framework} ... Section~\ref{sec:conclusion} concludes.").
**Defect:** Roadmap omits §3 (Mass), §10 (Methods), and the appendices. Reader's expectation set in Introduction is violated by §3 in particular.
**Suggested fix:** Either drop §3 from main text (per F.2), or update the roadmap to mention it. Methods and appendices need not be roadmapped.

### F.8 — Appendix C "Intuitive Examples" (MINOR)
**Location:** Appendix at 3901, single subsection "Macroscopic Objects as Meta-Agents: The Rock Example".
**Defect:** A 16-line appendix with one example. Either it earns enough space to be a section, or it folds into the Discussion of macroscopic objects (§7.3, line 3207).
**Suggested fix:** Inline this into §7.3, then delete the appendix.

---

## 2. Notation collisions and consistency

### F.9 — `M` overloaded across sections (MAJOR)
**Location:** Throughout, but pinch points at 637, 1598, 1607, 2075, 2273, 2479, 2481.
**Defect:** Same symbol carries at least four distinct meanings:
1. `\mathcal{M}` = multi-agent system (line 637, 639);
2. `M` = Hessian / mass matrix of the free energy (line 1598, 1607);
3. `\mathcal{M}` = product manifold $(\mathbb{R}^K \times \mathbb{S}^+_K \times G)^N$ (line 2075);
4. `\mathcal{M}` = Fisher information matrix $\mathcal{M}_{ij}[q]$ (line 2479, 2481);
5. `\mathcal{M}_{\text{geom}}` = the kinetic Lagrangian density (line 1707);
6. `M` = model-fiber dimension in `\mu_i^s, \mu_i^r: \mathcal{U}_i \to \mathbb{R}^M` (line 624) — this collides with the mass-matrix `M` two pages later.
The reader cannot tell at a glance whether `M` in §3 is the mass Hessian or the model-fiber dimension. Worse, line 2479 reintroduces $\mathcal{M}$ as Fisher information but trace-of-Fisher already appeared as `\mathrm{tr}(\mathcal{M})` at line 2465 without definition.
**Suggested fix:** Reserve `\mathcal{M}` for the multi-agent system collection and never reuse. Use `H` (or `H_{\mu\mu}`) for the Hessian/mass matrix. Use `I_F[q]` or `\mathcal{I}` for the Fisher information matrix. Use `D_s` (or similar) for the model-fiber dimension. State all of these in a notation table at the start of §2.

### F.10 — `K` overloaded (MAJOR)
**Location:** `K` appears as the mean-fiber dimension throughout (line 624, 1607), and as the spring stiffness `\omega^2 = K/m` at line 1691 (and `k` at 1525).
**Defect:** The mass section uses lowercase `k` (line 1525) but the within-framework recap at 1691 uses `K`, which collides with the fiber dimension that has been the only meaning of `K` for 1100 lines prior.
**Suggested fix:** Use lowercase `k` consistently for the harmonic-oscillator stiffness, or use `\kappa_{\text{stiff}}` to avoid all confusion. Keep `K` for fiber dimension only.

### F.11 — `\phi` does triple duty (MAJOR)
**Location:** §2.4 explicitly flags two roles for $\phi_i$ ("transport role" Role A, "state role" Role B; line 537). But $\phi$ is also used as the gauge-frame field appearing in $\Omega_{ij} = e^{\phi_i}e^{-\phi_j}$ throughout; as a scalar coefficient in the expansion $\phi_i^a$ over generators $\{G_a\}$ (line 624); and in the optional regularizer `\lambda_\phi \int \|\nabla \phi_i\|^2` (line 1241) where it functions as a smooth field.
**Defect:** The Role-A / Role-B disambiguation at 537 is excellent and should be retained, but the manuscript never warns the reader that $\phi$ later acquires a *third* sense as scalar generator-coefficients.
**Suggested fix:** In the same paragraph at 537, add a sentence: "Component-wise we write $\phi_i = \sum_a \phi_i^a G_a$ where $\{G_a\}$ are basis generators of $\mathfrak{g}$; the coefficient fields $\phi_i^a$ are scalars and inherit their gauge transformation from $\phi_i$." This makes the third sense explicit.

### F.12 — `\Omega` versus `\tilde{\Omega}` (MINOR)
**Location:** Line 603 ("$r_i = \tilde\Omega_{i,I}[s_I^{(s+1)}]$"), with $\Omega$ used for state-fiber transport and $\tilde\Omega$ for model-fiber transport.
**Defect:** The tilde convention is used consistently downstream (e.g., line 2671 "$\tilde\Omega_{ij}$ on the model fiber"), but it is introduced *implicitly* — the first appearance of $\tilde\Omega_{i,I}$ at line 603 has no defining sentence.
**Suggested fix:** Add one line at 603 along the lines of: "We write $\tilde\Omega$ for the transport on the model-parameter fiber $\mathcal{B}_{\mathrm{model}}$; the structure parallels the state-fiber transport $\Omega$ but acts in the $M$-dimensional model representation."

### F.13 — `\beta_{ij}` versus `\beta^{(p)}_{ij}` versus `\gamma_{ij}` (MINOR)
**Location:** Caption of Figure at line 2671 introduces `\beta^{(p)}` as a "state-fiber coupling distinct from the canonical model-channel weight `\gamma_{ij}`".
**Defect:** This is a *third* coupling weight introduced inside a figure caption, not in the body. The reader has no chance of finding the definition again.
**Suggested fix:** Define `\beta^{(p)}` once in §2.10 (the canonical free-energy functional) or §4 alongside meta-agent formation. The figure caption can then cite the body definition.

### F.14 — `\zeta` (scale index) versus other uses (MINOR)
**Location:** First appears in figure captions (line 2661) without prior body definition. Search of body text shows `\zeta` only enters the manuscript via Results captions.
**Defect:** A scale index that is central to the multi-scale meta-agent program should be defined in §4 (Implementation / multi-scale structure), not introduced in a results figure caption.
**Suggested fix:** Add an explicit "we index the hierarchy of scales by $\zeta = 0, 1, 2, \ldots$" sentence in the meta-agent variational principle at line 1736.

### F.15 — Notation table missing (MINOR)
**Location:** Whole document.
**Defect:** A 4598-line manuscript with this much notation overloading needs a one-page symbol glossary.
**Suggested fix:** Add a "Notation" subsection at the start of §2 listing: $\mathcal{C}, \mathcal{N}, \mathcal{B}, G, \mathfrak{g}, U_i, \phi_i, \Omega_{ij}, \tilde\Omega_{ij}, q_i, p_i, s_i, r_i, \alpha_i, \beta_{ij}, \beta^{(p)}_{ij}, \gamma_{ij}, \chi_i, \pi_{ij}, \kappa_\beta, \kappa_\gamma, K, M, \zeta, M_{\mu\mu}, m_{\text{eff}}, \mathcal{F}$. Two columns, ~25 rows.

---

## 3. Redundancy

### F.16 — Mass / precision / Fisher claim repeated three times (MAJOR)
**Location:** §3 (1519–1717), §6.5 lines 2479–2481, §7.2 lines 3169–3206.
**Defect:** The "precision = mass" reading is established once (§3), reasserted in the phenomenological-interpretation block (§6.5), and then re-litigated again in Discussion §7.2 ("Inertial Mass and Gravitational Effects"). Each pass adds caveats, but the substantive content is the same: the harmonic-oscillator analogy and the empirical $\omega^2 \propto 1/m_{\text{eff}}$ scaling. The manuscript would be stronger if §3 made the case once and §6.5 / §7.2 just cited it.
**Suggested fix:** Cut §6.5 lines 2479–2484 to a single sentence with a `\ref` to §3. Cut §7.2 from its current 38 lines to a single paragraph that *only* covers what is genuinely new (the gravitational coupling speculation).

### F.17 — Wheeler's program retold three times (MINOR)
**Location:** §1.1 (63), §2.10.5 (1394), §5 Pullback (2077), §6 Self-Excited Bootstrap (2796), §7.5 (3264).
**Defect:** The "It from Bit" slogan and the participatory-universe motivation are re-narrated in each of these locations. Once is sufficient as motivation; one of the later restatements should be a one-sentence callback.
**Suggested fix:** Keep §1.1 as motivation. Cut the §2.10.5 "Connection to Wheeler's 'It From Bit'" subsubsection (1394–1398) since §5 will revisit it formally. Compress §6 lines 2796–2853 ("Reality Participates", "From Bit to It to Bit Again", "Comparison to Wheeler's Vision") to half their length. Cut §7.5 entirely.

### F.18 — Caveat paragraphs repeat themselves (MAJOR)
**Location:** Lines 1523, 1525, 1607, 1610, 1635, 1637, 1658, 1689, 1691, 1705.
**Defect:** §3 carries roughly ten caveat paragraphs in a 200-line section, each restating that "the Hessian gives stiffness, identification with mass requires the kinetic-metric postulate". This is over-defended. Once stated clearly in the section opener, the same caveat does not need to be repeated in every sub-block. The repetition is a writing tic — likely an artifact of multiple round-trip edits — and reads as nervousness rather than rigor.
**Suggested fix:** State the kinetic-postulate caveat once at the start of §3 (a single paragraph), once in the §3.7 "Velocity-Quadratic Metric Form" subsection where the postulate is formally introduced, and nowhere else within the section. Internal restatements can be replaced with `(\ref{...})` callbacks.

### F.19 — Forward / backward references to `sec:pullback` and `sec:mass` (MINOR)
**Location:** ~17 references to `sec:pullback` and ~10 to `sec:mass` from elsewhere in the manuscript.
**Defect:** Both labels are referenced before they appear (`sec:pullback` first appears as a forward ref at line 121 in the Introduction). The roadmap mitigates this for `sec:pullback`, but `sec:mass` is referenced from 1691 forward without mention in the roadmap. See F.7.
**Suggested fix:** Mention the mass section in the roadmap, or move §3 after §5 and reorder the forward refs accordingly (preferred: see F.2).

---

## 4. Math display / punctuation

### F.20 — Equation punctuation is inconsistent (MINOR)
**Location:** Sample of display equations across the manuscript.
**Defect:** Many display equations end without terminal punctuation (comma or period), which is the standard convention in physics journals. Examples: line 639 (period), line 1487 (comma), line 1707 (no punctuation), line 1708 (no punctuation, despite continuing prose), line 2273 (no punctuation). The manuscript is not internally consistent.
**Suggested fix:** Apply the standard rule in a single pass: equations that end a sentence get a period; equations followed by "where ..." or other continuation get a comma; equations that are followed by a fresh sentence still get a period. CLAUDE.md explicitly calls this out as part of the doc-cleanup pass.

### F.21 — `\,` and `\!` instances must go (MAJOR)
**Location:** Lines 399, 453, 454, 602, 603, 1181, 2332.
**Defect:** Per CLAUDE.md "Scientific Writing Rules", `\;`, `\,`, `\!` are banned outright. There are 7 instances.
**Suggested fix:**
- Lines 399, 453, 454: replace `\Omega_{ij}\,q_j` with `\Omega_{ij} q_j` (the multiplicative-style spacing is unnecessary; the standard typesetting is acceptable).
- Lines 602, 603: replace `\Gamma\!\left(...\right)` with `\Gamma\left(...\right)` — the negative-thin-space removes a slight gap before the `\left`, which is cosmetic.
- Line 1181: `U_i(c)\, g(c)\, g(c)^{-1}\, U_j(c)^{-1}` → use plain spaces.
- Line 2332: `c;\; U_i \mapsto U_i \cdot g` → `c; U_i \mapsto U_i \cdot g`.

---

## 5. Figure / table captions

### F.22 — Caption at 1323 contains a derivation, not a description (MINOR)
**Location:** Figure caption line 1323 (Belief attention field $\beta_{ij}(c)$).
**Defect:** The caption embeds the full softmax formula `\beta_{ij}(c) = \pi_{ij}(c)\exp[-\kappa_\beta^{-1} \mathrm{KL}(...)] / \sum_k ...` inline. This is the *body equation* (Eq.~\ref{eq:softmax_attention_general} from line 1241 region). Captions should describe and reference, not re-derive.
**Suggested fix:** Replace inline formula with `\eqref{eq:softmax_attention_general}` and add a one-sentence interpretive gloss.

### F.23 — Caption at 2762 carries an apologia (MINOR)
**Location:** Figure caption line 2762 (hierarchical meta-agent structure).
**Defect:** "...the threshold-based consensus detector of Section~\ref{sec:meta_agent_threshold} acting on agent state variables driven by the gauge-theoretic dynamics; the energy functional and initialization do not hard-code a hierarchical depth, scale assignments, or a tree topology, but the detector that nucleates meta-agents is imposed. Whether a continuous-time evaluation of the variational criterion ... reproduces this structure is left for future work." This is a limitation paragraph hidden in a caption.
**Suggested fix:** Move the limitation discussion to body text; keep the caption descriptive (panel labels, scale color coding, what the figure shows).

### F.24 — Captions are otherwise good (PASS)
The captions at 1428, 1439, 2661, 2671, 2702, 2730, 3146 are well-constructed: they describe what is shown, identify what each panel represents, and provide enough quantitative detail (e.g., "$520$-fold spike", "$R^2 = 1.000$ on per-$K$ seed means") to be read independently. **No changes needed.**

---

## 6. Hedging and prose economy

### F.25 — High count of hedge / filler words (MINOR)
**Location:** 17 instances of `rigorously / essentially / truly / simply / clearly / obviously / of course`; 8 instances of `It is worth being / We note that / We emphasize / Note that`.
**Defect:** Below the dramatic count of a typical preprint, but a paper this long benefits from a final pass to remove "essentially", "simply", "of course", and "It is worth being explicit about ...". Each is an admission that the prose is doing the work the math should do.
**Suggested fix:** Single search-and-cut pass; aim to eliminate roughly half.

### F.26 — Use of "we suspect" / "we flag" (NIT)
**Location:** Lines 849 ("We flag this as a research direction"), 1414 ("We suspect that spacetime emerges as a derived structure").
**Defect:** Acceptable in a frankly speculative paper, but two of these in a single subsection start to read as hand-wave. Pair them with explicit cross-refs to the open-problems section.
**Suggested fix:** Replace "we flag" / "we suspect" with concrete pointers when an open-problems entry exists.

---

## 7. Section ordering recommendation

The current order is:
1 Intro → 2 Theory → 3 Mass → 4 Implementation → 5 Pullback → 6 Observer/Quantum → 7 Discussion → 8 Open Problems → 9 Conclusion.

A reordering that would substantially help the reader:
1 Intro → 2 Theory → 3 Pullback (current §5; this is the *technical* heart of the spacetime program) → 4 Mass-as-Stiffness (current §3; folded as §3.5 of the new Theory or kept as a separate short section that *follows* Pullback so the metric postulate has motivation) → 5 Implementation/Multi-Scale (current §4) → 6 Observer/Quantum (current §6) → 7 Results → 8 Discussion → 9 Conclusion.

The key move is putting Pullback before Mass: the mass section's appeal to "kinetic structure" and "spacetime" lands without grounding when the pullback construction has not yet been seen. Putting Implementation after Mass keeps the empirical narrative (validation simulation comes in the Results section anyway) close to the constructions it validates.

---

## 8. Summary of severities

| Severity   | Count |
|-----------|------:|
| BLOCKER    | 0     |
| MAJOR      | 11    |
| MINOR      | 14    |
| NIT        | 1     |

---

## 9. Verdict

**Conditional accept on writing-style grounds**, contingent on: (1) removing all 7 banned LaTeX spacing macros (F.21); (2) at least one major structural decision — either split the manuscript (F.1) or accept the cuts in F.5–F.8; (3) addressing the `M` and `\phi` notation overload with a notation table (F.9, F.11, F.15); (4) collapsing the threefold mass / precision / Fisher reiteration into a single canonical exposition with cross-references (F.16, F.18); (5) one consistency pass on equation punctuation (F.20).

The banned-words check passed cleanly, the citation style is internally consistent, and there are no broken cross-references or self-referential drafting history. The structural concerns are not about correctness but about a manuscript that is roughly twice as long as it needs to be for any single research community, with notation that has accumulated more meanings than the symbol space affords.
