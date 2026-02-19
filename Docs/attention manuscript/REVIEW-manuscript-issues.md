# Manuscript Review: JMLR Attention 11-3-25.tex
## Redundancies, Errors, and Repetition

---

## A. TYPOS AND GRAMMATICAL ERRORS

1. **Line 66**: `"based opn mixture Gaussians"` — should be **"based on mixture Gaussians"**

2. **Line 663**: `"In a straight forward manner"` — should be **"In a straightforward manner"** (one word)

3. **Line 775**: `"while small $b_0$ yield $\alpha_i$"` — **sentence is truncated/incomplete**. Should say something like "while small $b_0$ yield rapidly varying $\alpha_i$."

4. **Line 1352**: `"heads\citep{vaswani2017attention}"` — **missing space** before `\citep`.

5. **Line 1429**: `"Pactically speaking"` — should be **"Practically speaking"**

6. **Line 1493**: `"a intermediate position"` — should be **"an intermediate position"**

7. **Line 1637**: `"backpropogation-free"` — should be **"backpropagation-free"**

8. **Line 2486**: `"restricting the the matched-fiber"` — **doubled "the"**

---

## B. INCOMPLETE / PLACEHOLDER CONTENT

9. **Line 41**: `\editor{TBD}` — placeholder still present.

10. **Table 5 (lines 1911-1929)**: Clustering metrics table has **"(to be computed on trained model)"** for 5 of 7 metrics. Must be populated or removed before submission.

11. **Line 32**: `\jmlrheading{}{2025}` — verify intended year.

---

## C. FACTUAL INCONSISTENCIES

12. **Hardware discrepancy**: Main text (line 1579) says **"AMD Ryzen 9900x"** but Appendix E (line 2863) says **"AMD Ryzen 9 5950X"**. These are different processors.

13. **GL(20) vs GL(30) mismatch**: Table 4 (line 1870) reports **GL(20), K=80** but:
    - Section label is `sec:gl30`
    - Figure filenames are `gl30_*`
    - Figure captions (lines 1894, 1899, 1901) describe "GL(30) representations" with mu in R^30 and phi in R^900
    - These figures appear to be from a different model run than the reported GL(20) results

14. **Missing SO(N) results**: Section header (line 1613) reads "GL(K) and SO(N) Language Modeling" but no SO(N) language modeling results are presented.

15. **415x improvement arithmetic**: 50,000/121.1 ~ 413, not 415. Minor but worth standardizing.

16. **Variable collision**: Appendix E uses H x W for spatial grid but H = number of heads throughout the main text.

17. **Duplicate subsubsection title**: "Dual Relation via the Envelope Theorem" appears at both line 2573 and line 2658 in Appendix C.

---

## D. MAJOR REDUNDANCIES AND REPETITION

These are the most significant issues. The manuscript repeats key results verbatim far more than necessary, substantially inflating length.

### D1. BERT validation statistics (repeated 4-5x)
Grand mean r=0.804, 95% CI [0.771, 0.838], tau=19.0 appears at:
- Abstract (line 51)
- Results (line 1655)
- Temperature section (line 1710)
- Table caption (line 1731)
- Discussion (line 1988)

**Recommendation**: State fully in results; reference briefly in abstract/conclusion.

### D2. Cross-passage standard errors (repeated 5-6x)
"mean SE = 0.006" / "SE < 0.02 for all 144 heads" at:
- Abstract (line 51)
- Results subsection (line 1659)
- Figure caption (lines 1690-1691)
- Scope caveat (line 1706)
- Discussion (line 1988)
- Limitations (line 2038)

### D3. GL(K) perplexity 121 (stated 6x)
- Abstract, Results table, Results narrative, KN-5 comparison, Discussion, Conclusion

### D4. "No MLPs, activation functions, or learned projections" (stated 5x)
- Lines 1615, 1876, 1884, 1990, 2080

### D5. Uniform attention / PPL 377 (described 5x)
- Section 5.5 (full description)
- Discussion 6.1 (restated)
- Discussion 6.2 (substantially restated)
- Discussion 6.4.4 (restated)
- Conclusion (restated)

### D6. "Transport geometry over attention weighting" hypothesis (stated 5x)
- Lines 1933, 1976, 1992, 1996, ~2080

### D7. Gauge transport Omega_ij = exp(phi_i)exp(-phi_j) (defined 5x)
- Table 1 (line 281), line 324, Eq at line 483, Discussion (line 2016), Appendix A (line 2213)

### D8. W_Q W_K^T as gauge transport (explained 5-6x)
- Abstract, Section 4.2.1, Section 4.9, Section 4 Summary, Discussion 6.1, Conclusion

### D9. Temperature tau=19.0 vs 2*sqrt(d)=16 (repeated 4x)
- Lines 1655, 1710, 1731, 1988

### D10. Section summaries overlap extensively
- Section 3 Summary (lines 828-832) overlaps with Section 4 Summary (lines 1554-1572), which overlaps with Discussion 6.1, which overlaps with the Conclusion. Four "summaries" cover the same ground.

### D11. Representational capacity bottleneck
- "Representational Capacity vs. Output Capacity" (lines 1937-1942) substantially overlaps with Discussion 6.2 (lines 1994-2002).

### D12. Layer normalization explanation (repeated 4x)
- Section 4.2.1 (lines 984-1000), Section 4 Summary (line 1570), Section 5.2.3 (line 1795), Discussion 6.3 (line 2006)

---

## E. STRUCTURAL SUGGESTIONS

- **Discussion (Section 6)** is too long and mostly recapitulates Sections 4-5. Section 6.1 ("Summary of Contributions") re-presents every major result verbatim. Focus the discussion on interpretation, implications, and connections.
- **Conclusion** runs ~500 words and restates every result again. A 150-200 word conclusion would be more appropriate for JMLR.
- **Flat bundle discussion** appears implicitly in Section 4 limits and again as standalone Discussion subsection 6.3. Consolidate.

---

## F. MINOR NOTES

- **Line 2094**: Acknowledgment mentions "Claude Sonnet 4.5" — verify model name.
- **Line 2895**: Command uses escaped underscores in plain text rather than lstlisting/texttt.
