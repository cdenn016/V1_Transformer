# Extended Evidence — harvested canon (blue Phase 2)

Newly-surfaced canon from the blue panel, beyond the citations already in `01_evidence.md`. Deduplicated against the evidence pack (which already lists Pennec/Arsigny, Nakahara, Hall, Bishop, McInnes, Rousseeuw, Roy-Vetterli).

- **Product-manifold metric (justifies the per-head quadrature).** For a Riemannian product manifold `M = M_1 × M_2` with the product metric, the squared geodesic length decomposes as the sum of the per-factor squared lengths, `d(p,q)² = d_1(p_1,q_1)² + d_2(p_2,q_2)²` [Lee2013, *Introduction to Smooth Manifolds*, product-manifold construction]. Applied to the direct-product group `GL(10) × GL(10) ⊂ GL(20)` with the product of left-invariant metrics, the quadrature `d = sqrt(Σ_h d_h²)` in `geometry.py:355-356` is the exact product metric, not an approximation. This is the canonical backing for sub-proposition 1's quadrature step, which `01_evidence.md` listed under "what this evidence does NOT settle."

- **Block-diagonal exp factorization.** For `A = A_1 ⊕ A_2` block-diagonal, `exp(A) = exp(A_1) ⊕ exp(A_2)` exactly, because every term `A^k` in the power series is block-diagonal with blocks `A_1^k ⊕ A_2^k` [Hall2015 §2-3 on the matrix exponential; Nakahara2003 §5]. This is the analytic reason the two Ω code paths (`geometry.py:339` block-restrict-then-exp vs `pipeline.py:119-121` full-exp) agree to 4.4e-16 under a block-diagonal bank — the agreement is structural, not coincidental.

No other new canon. The remaining defense citations (Cencov1972, Bishop2006, McInnes2018, Rousseeuw1987, Nakahara2003, Hall2015) are already in `01_evidence.md` / `external_bibliography.md`.

# Extended Evidence — harvested canon (red Phase 2)

Canon harvested by the red expert panel beyond the neutral pack and the blue section above.

- **[Pennec, Fillard, Ayache 2006, IJCV 66(1):41-66, "A Riemannian Framework for Tensor Computing"]**
  — affine-invariant Riemannian metric on the SPD/PD cone; geodesic distance `||log(P^{-1/2} Q
  P^{-1/2})||_F`, invariant under the congruence action `P → A P Aᵀ`; unique geodesic and Fréchet
  mean. The defining property is invariance under anisotropic linear rescaling. Verified via Springer
  DOI 10.1007/s11263-005-3222-z and the CIS-UPenn hosted PDF. This is the intrinsic, *un-whitened*
  distance class the Ω geodesic (`||log(Ω_i⁻¹Ω_j)||_F`, geometry.py:297) belongs to — and the class
  the whitened-Euclidean φ distance does NOT belong to.

- **[McInnes, Healy, Melville 2018, arXiv:1802.03426, "UMAP"]** — embedding dimensions are not
  interpretable; the objective is topological/neighborhood cross-entropy, not exact distance
  preservation. Kobak & Linderman (bioRxiv 2019.12.19.877522) show UMAP global structure is
  initialization-driven, not distance-faithful. Consequence: two UMAP layouts from two different
  precomputed matrices are not related by a rigid transform, so a 0.96 matrix-level Spearman cannot
  rescue a side-by-side *visual* comparison.

- **[Amari 1998, Neural Computation 10(2):251-276]** — reparameterization invariance is the defining
  property of a principled metric; sample-covariance preconditioning (whitening, like Adam/RMSProp) is
  not the Fisher metric and is not invariant. Registered in external_bibliography.md.

- **Spearman is a rank-correlation summary** — it constrains the ordering of distance values, not the
  geometric structure of the underlying matrices, and does not predict 2D-layout similarity under a
  manifold-learning embedding (UMAP's objective is built from actual neighbor distances, not their
  ranks [McInnes2018]). A 0.96 matrix-level Spearman therefore does not rescue a side-by-side visual
  comparison of two UMAP layouts.

- **[Bishop, PRML 2006 §12.1]** — whitening `y = L^{-1/2} Uᵀ (x − x̄)` rotates to the PCA basis and
  rescales each axis to unit variance; the whitened-coordinate Euclidean distance differs from the
  original-coordinate distance and is not distance-preserving.

# Extended Evidence — harvested canon (red Phase 3 rebuttal)

Canon surfaced by the red panel in the rebuttal round, beyond all sections above. Deduplicated.

- **[Popper 1959, *The Logic of Scientific Discovery*, §§4, 6, 19–21]** — demarcation by
  falsifiability. A claim that reassigns every present inaccuracy to a non-correctness bucket
  ("completeness") is immunized against all observation and is therefore not an empirical claim about
  the artifact's correctness. Anchors the unfalsifiability attack on "correct-by-design."

- **[Tufte 1983, *The Visual Display of Quantitative Information*, ch. on graphical integrity]** — a
  graphic has integrity only if the visual conclusion matches the data; rendering non-data
  (byte-fallback U+FFFD / empty tokens) as semantic structure violates graphical integrity. The unit
  of correctness for a figure is whether a competent reader reaches a true conclusion, not whether the
  per-row arithmetic is right.

- **[Wilson et al. 2017, PLOS Computational Biology 13(6):e1005510, "Good enough practices in
  scientific computing"]** — analysis outputs (figures, tables) are products subject to the same
  correctness scrutiny as code; a misleading figure is a defective result, not a cosmetic limitation.

- **[Hall 2015, *Lie Groups, Lie Algebras, and Representations*, §3]** — the inner product inducing a
  left-invariant metric on a matrix Lie group is the Frobenius/Killing form at the identity. PCA-
  whitening rescales algebra coordinates by empirical per-sample singular values, which is not such an
  inner product and is not left-invariant — so the φ-panel metric and the Ω-panel left-invariant
  geodesic are different geometric objects.

- **Red concession (recorded for the judges):** the Pennec/Fillard/Ayache 2006 affine-invariant SPD
  geodesic is withdrawn as the canon for `omega_geodesic_distances`. The code (`geometry.py:349-352`)
  computes `‖log(Ωᵢ⁻¹Ωⱼ)‖_F`, a left-invariant distance on GL⁺(K) group elements (invariant under
  Ω → AΩ), not the SPD congruence distance (invariant under P → APAᵀ). The docstring at
  `geometry.py:290` carries the same wrong-domain "affine-invariant" label (drift; code is canonical).
