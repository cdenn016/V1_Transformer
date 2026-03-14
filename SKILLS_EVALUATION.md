# Claude Scientific Skills Evaluation for Gauge-Transformer

**Source:** [K-Dense-AI/claude-scientific-skills](https://github.com/K-Dense-AI/claude-scientific-skills)
**Date:** 2026-03-14
**Project:** Gauge-Transformer (GL(K) gauge-covariant VFE minimization for language modeling)

---

## Tier 1: Highly Relevant (Direct Match to Core Workflow)

### 1. `sympy` — Symbolic Mathematics
The project is built on Lie algebra generators, gauge transport operators (`exp(φ_i)·exp(-φ_j)`), KL divergence between Gaussians, and Baker-Campbell-Hausdorff expansions. SymPy can:
- Verify symbolic derivations of KL divergence invariance under GL(K) transformations
- Validate Lie algebra commutation relations and structure constants
- Symbolically derive gradients of the VFE objective
- Generate LaTeX from symbolic expressions for the manuscript

### 2. `matplotlib` + `scientific-visualization` — Publication Figures
The project already uses matplotlib extensively (`scripts/generate_publication_figures.py`, `transformer/visualization/`). The scientific-visualization skill adds:
- Journal-specific formatting (Nature, Science, NeurIPS, ICML, etc.)
- Colorblind-accessible palettes for attention heatmaps and RG flow diagrams
- Multi-panel figure composition for perplexity/entropy/clustering plots
- Correct DPI/format export for `Attention/figs/`

### 3. `scientific-writing` — Manuscript Development
Active manuscript files: `GL(K)_attention.tex`, `GL(K)_supplementary.tex`. This skill provides:
- IMRAD structure guidance for theoretical ML papers
- LaTeX scientific report styling
- Figure/table integration best practices
- Section-by-section writing support

### 4. `peer-review` — Manuscript Quality Assessment
Before submission, this skill can:
- Evaluate methodological rigor of gauge-theoretic claims
- Check statistical validity of BERT validation results (144 attention heads)
- Assess reproducibility (hyperparameters, seeds, configs)
- Review figure quality and data presentation

### 5. `pytorch-lightning` — Training Infrastructure
Training code (`train.py`, `train_publication.py`) totals ~125KB of hand-rolled loops. PyTorch Lightning could:
- Replace boilerplate with `LightningModule` for `GaugeTransformerLM`
- Add automatic checkpointing, mixed precision, and multi-GPU support
- Integrate W&B/TensorBoard logging natively
- Simplify the resume-training workflow

---

## Tier 2: Strong Supporting Value

### 6. `plotly` — Interactive Visualization
For exploring attention patterns, gauge frame evolution, and RG flow interactively:
- 3D visualization of learned gauge frames in GL(K) space
- Interactive attention heatmaps with hover details
- Animated belief evolution across VFE iterations

### 7. `networkx` — Graph/Network Analysis
Already imported for meta-agent detection and RG flow analysis (`transformer/analysis/rg_metrics.py`). Skill provides structured guidance for:
- Spectral clustering of attention graphs
- Community detection in learned token relationships
- Modularity analysis for emergent structure

### 8. `statistical-analysis` — Rigorous Statistics
For strengthening empirical claims:
- Proper effect size reporting for BERT correlation analysis
- Confidence intervals on perplexity measurements
- Bayesian alternatives for null hypothesis testing
- APA-formatted statistical reporting

### 9. `arxiv-database` — Literature Search
For tracking related work in gauge-equivariant networks, information geometry, and VFE:
- Search `cs.LG`, `cs.CL`, `stat.ML`, `hep-th` categories
- Monitor new submissions on geometric deep learning
- Build comprehensive related work sections

### 10. `literature-review` — Systematic Review
For building thorough related-work covering:
- Gauge equivariant neural networks (Cohen, Weiler, et al.)
- Information-geometric approaches to ML
- Active inference / free energy principle in ML
- Attention mechanism theory

### 11. `umap-learn` — Dimensionality Reduction
For visualizing learned representations:
- Embed token beliefs (μ, Σ) into 2D/3D
- Compare gauge frame structure across training
- Visualize semantic clustering in learned embedding space

---

## Tier 3: Useful for Specific Tasks

### 12. `shap` — Model Interpretability
Feature attribution for VFE components beyond attention weights.

### 13. `pymc` — Bayesian Modeling
The framework is fundamentally Bayesian (beliefs, priors, variational inference). PyMC could:
- Validate variational approximation quality
- Compare VFE posterior to true posterior on small-scale problems
- Bayesian hyperparameter sensitivity analysis

### 14. `hypothesis-generation` — Research Direction
Systematically generate testable predictions from the gauge-theoretic framework (scaling laws, cross-linguistic universality).

### 15. `latex-posters` — Conference Posters
For presenting at NeurIPS, ICML, ICLR, etc.

### 16. `scientific-schematics` — Diagram Generation
Architecture diagrams of the gauge-transformer pipeline, fiber bundle illustrations, information flow diagrams.

### 17. `citation-management` — Reference Management
Maintaining the bibliography in the LaTeX manuscript.

---

## Not Relevant

The majority of the repository (bioinformatics, drug discovery, proteomics, clinical, geospatial, lab automation, cheminformatics, etc.) does not apply to this project. The relevant subset is concentrated in: **math/physics, ML/AI, visualization, and scientific communication**.

---

## Recommended Top 5 to Install First

| Priority | Skill | Rationale |
|----------|-------|-----------|
| 1 | `sympy` | Core math verification for Lie algebra / gauge theory derivations |
| 2 | `scientific-visualization` | Publication-quality figures for the manuscript |
| 3 | `peer-review` | Pre-submission quality check of manuscript and methods |
| 4 | `scientific-writing` | Manuscript structure, polish, and LaTeX best practices |
| 5 | `statistical-analysis` | Rigorous empirical claims with proper effect sizes and CIs |
