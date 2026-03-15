# Flat Bundle Hypothesis — Testable Predictions

**Derived from:** Section 7.2, "The Flat Bundle Limit and the Geometry of Language"

**Core Conjecture:** *The compositional-semantic component of natural language is approximately path-independent in the sense of gauge transport. Transformers are architecturally matched to this regime.*

**Three logically distinct sub-claims:**
1. The bulk of semantic transport is path-independent (compositional semantics ↔ flat bundle)
2. Communicative systems face functional pressure toward path-independence (reliable communication requires Ω_ij·Ω_jk ≈ Ω_ik)
3. Transformers succeed in part because their flat bundle structure (W_Q W_K^⊤ = σ^{-2}Ω^{-⊤}) matches this property of language

---

## Category F1: Compositionality Benchmarks (Direct Tests)

### HF1.1: Flat Architecture Suffices for Compositional Generalization

**Derived from:** Sub-claim 1 (compositional semantics is path-independent)
**Prediction:** The Gauge-Transformer (flat bundle by construction) should match or exceed standard transformers on compositional generalization benchmarks (COGS, SCAN), because compositional semantics lives in the flat regime that both architectures capture.
**Null hypothesis:** The Gauge-Transformer performs significantly worse than standard transformers on COGS/SCAN, indicating that compositional generalization requires structure beyond flat gauge transport.
**Operationalization:**
- Train Gauge-Transformer and size-matched standard transformer on COGS train split
- Evaluate on COGS generalization split (novel structural combinations)
- Repeat for SCAN (length generalization, template-around-right splits)
- Primary metric: exact-match accuracy on held-out structural splits
**Expected:** Gauge-Transformer accuracy ≥ standard transformer accuracy (within 2%)
**Priority:** ★★★★★ — this is the most direct test of the conjecture
**Risk:** Both architectures may fail on COGS/SCAN for reasons unrelated to flatness (e.g., training distribution mismatch). Mitigate by including known-strong baselines (T5, BART with augmentation).

### HF1.2: Non-Flat Architecture Learns Holonomy on Pragmatic Tasks

**Derived from:** Sub-claim 1 (pragmatic inference is path-dependent)
**Prediction:** A non-flat gauge architecture (Ω_ij as independent GL(K) elements per edge, unconstrained by cocycle condition) trained on tasks requiring pragmatic/contextual inference should learn holonomy H_ijk = Ω_ij·Ω_jk·Ω_ki ≠ I specifically on token triples where meaning is context-dependent.
**Null hypothesis:** Learned holonomy is uniformly near identity even on pragmatic tasks, or holonomy is unstructured (no correlation with context-dependence).
**Operationalization:**
- **Architecture:** Replace Ω_ij = exp(φ_i)·exp(-φ_j) with independent learned Ω_ij ∈ GL(K) per edge (parameterized as MLP(h_i, h_j) → K×K matrix, or direct embedding lookup for small graphs)
- **Task:** Train on datasets requiring pragmatic inference:
  - Irony/sarcasm detection (SemEval-2018 Task 3)
  - Implicature resolution (GRICE dataset, Ruis et al. 2022)
  - Context-dependent word sense disambiguation (WiC, Pilehvar & Camacho-Collados 2019)
- **Measurement:** For each token triple (i,j,k), compute H_ijk = Ω_ij·Ω_jk·Ω_ki and measure ‖H_ijk − I‖_F
- **Analysis:** Correlate holonomy magnitude with human annotations of context-dependence
**Expected:** ‖H_ijk − I‖_F is significantly larger for context-dependent triples (effect size d > 0.5, p < 0.01)
**Priority:** ★★★★★ — the manuscript identifies this as the key prediction
**Risk:** The non-flat architecture may be too unconstrained to train stably. Mitigate by starting from a pre-trained flat model and relaxing the cocycle constraint gradually.

### HF1.3: Holonomy Magnitude Predicts Compositionality Score

**Derived from:** Sub-claims 1 and 3
**Prediction:** Across a range of sentences varying in compositional complexity, the average holonomy of the non-flat model should anti-correlate with compositionality: sentences with straightforward compositional semantics → low holonomy; sentences with pragmatic/non-compositional meaning → high holonomy.
**Null hypothesis:** No correlation between holonomy and compositionality ratings.
**Operationalization:**
- Use graded compositionality judgments (e.g., Reddy et al. 2011 for noun compounds, or construct a new dataset with Likert-scale compositionality ratings)
- For each sentence, compute mean holonomy: H̄ = (1/|T|) Σ_{(i,j,k)∈T} ‖H_ijk − I‖_F over token triples T
- Compute Spearman ρ between H̄ and compositionality rating
**Expected:** ρ < −0.3 (negative correlation, significant at p < 0.01)
**Priority:** ★★★★☆

---

## Category F2: Scaling Laws Under Flatness

### HF2.1: Flat Bundle Advantage Increases with Compositional Complexity

**Derived from:** Sub-claim 3 (transformers succeed because of flat bundle match)
**Prediction:** The Gauge-Transformer's advantage over a hypothetical non-flat architecture should grow with the compositional complexity of the data. On simple (low-compositionality) tasks, flat and non-flat should perform similarly. On highly compositional tasks, the flat inductive bias should provide a larger advantage.
**Null hypothesis:** Non-flat architecture matches or exceeds flat architecture at all compositionality levels.
**Operationalization:**
- Define a compositionality gradient: (a) bag-of-words tasks → (b) fixed-template generation → (c) recursive phrase composition → (d) deeply nested clauses
- Train both flat (standard Gauge-Transformer) and non-flat (edge-independent Ω_ij) at each level
- Plot performance gap as a function of compositionality depth
**Expected:** Performance gap (flat − non-flat) increases monotonically with compositionality level
**Priority:** ★★★★☆
**Risk:** Compositionality level conflates with task difficulty. Control by matching per-level baseline perplexity.

### HF2.2: Flat Bundle Data Efficiency on Compositional Tasks

**Derived from:** Sub-claim 3
**Prediction:** The flat-bundle Gauge-Transformer reaches a given accuracy on COGS/SCAN with fewer training examples than (a) a non-flat gauge architecture and (b) a standard transformer without gauge structure, because the flat bundle prior matches the compositional structure of the data.
**Null hypothesis:** All three architectures have similar sample efficiency on compositional tasks.
**Operationalization:**
- Train all three architectures on {1%, 5%, 10%, 25%, 50%, 100%} of COGS training data
- Plot learning curves (accuracy vs. training examples)
- Compute the "50% accuracy threshold" — number of examples needed to reach 50% generalization accuracy
**Expected:** threshold_flat < threshold_standard < threshold_non-flat
**Priority:** ★★★★☆

### HF2.3: Holonomy Penalty Scaling

**Derived from:** Sub-claim 2 (functional pressure toward flatness)
**Prediction:** Adding a soft holonomy penalty λ·Σ_T ‖H_ijk − I‖²_F to the loss of a non-flat architecture should improve performance on compositional tasks, with optimal λ > 0. This operationalizes the "functional pressure toward flatness."
**Null hypothesis:** Optimal λ = 0 (no benefit from penalizing holonomy) or holonomy penalty degrades performance.
**Operationalization:**
- Train non-flat architecture with λ ∈ {0, 0.01, 0.1, 1.0, 10.0}
- Evaluate on COGS generalization split
- Plot accuracy vs. λ
**Expected:** Inverted-U or monotonically increasing curve with peak at λ* > 0
**Priority:** ★★★★☆ — directly tests whether flatness is beneficial, not just architectural coincidence

---

## Category F3: Cross-Linguistic Predictions

### HF3.1: Holonomy Structure Varies with Pragmatic Complexity Across Languages

**Derived from:** Sub-claim 1 (compositional semantics is universally path-independent)
**Prediction:** Languages with higher pragmatic load (e.g., high-context cultures: Japanese, Korean, Arabic) should exhibit larger holonomy in non-flat models than languages with lower pragmatic load (e.g., low-context cultures: German, Finnish), because more meaning is carried by path-dependent pragmatic inference.
**Null hypothesis:** Holonomy magnitude does not correlate with Hall's high/low-context classification.
**Operationalization:**
- Train non-flat gauge models on parallel corpora across 6+ typologically diverse languages
- Compute average holonomy per language
- Correlate with pragmatic complexity proxies (Hall's high/low-context index, or % of meaning carried by implicature in annotated corpora)
**Expected:** Positive correlation between pragmatic load and average holonomy (ρ > 0.5)
**Priority:** ★★★☆☆ — high novelty but challenging operationalization of "pragmatic load"
**Risk:** The high/low-context distinction is coarse and contested. Use multiple proxies.

### HF3.2: Universal Flatness of Compositional Core

**Derived from:** Sub-claims 1 and 2
**Prediction:** When a non-flat model is trained on different languages, the subset of token interactions with near-zero holonomy (|H_ijk − I| < ε) should correspond to compositional syntactic structures (e.g., verb-argument, modifier-head) across all languages, while high-holonomy interactions should be language-specific.
**Null hypothesis:** The low-holonomy subset is language-specific and does not align with universal syntactic relations.
**Operationalization:**
- Train non-flat models on Universal Dependencies treebanks for 10+ languages
- For each language, identify the low-holonomy token triples
- Map these triples to UD relation types
- Test whether the same UD relations consistently appear in the low-holonomy set across languages
**Expected:** Core compositional relations (nsubj, obj, amod, det) are consistently flat; discourse/pragmatic relations (discourse, parataxis, vocative) show higher holonomy
**Priority:** ★★★★☆ — would be a strong result connecting gauge geometry to linguistic typology

### HF3.3: Word Order Freedom Correlates with Gauge Frame Diversity

**Derived from:** Sub-claim 3 (content-dependent path-independent transport)
**Prediction:** Languages with freer word order (e.g., Russian, Latin, Warlpiri) require more diverse gauge frames φ_i to maintain flat transport under permutation, because the transport must remain path-independent despite variable token ordering. This should manifest as higher effective rank of the gauge frame matrix.
**Null hypothesis:** Gauge frame diversity does not correlate with word order freedom.
**Operationalization:**
- Train Gauge-Transformer on languages spanning the word-order freedom spectrum
- Measure effective rank of {φ_i} per language (number of principal components explaining 95% variance)
- Correlate with word-order freedom index from WALS
**Expected:** Positive correlation (ρ > 0.4)
**Priority:** ★★★☆☆

---

## Category F4: Ablation Studies

### HF4.1: Cocycle Relaxation Ablation

**Derived from:** Sub-claim 3 (architectural match)
**Prediction:** Progressively relaxing the cocycle condition should degrade performance on compositional tasks but potentially improve performance on pragmatic tasks. Interpolate between flat and non-flat:
  Ω_ij = (1−α)·exp(φ_i)·exp(−φ_j) + α·MLP(h_i, h_j)
where α=0 is flat and α=1 is fully non-flat.
**Null hypothesis:** Performance is monotonic in α (always better or always worse with relaxation).
**Operationalization:**
- Train with α ∈ {0.0, 0.1, 0.25, 0.5, 0.75, 1.0}
- Evaluate separately on: (a) COGS generalization, (b) sarcasm detection, (c) standard LM perplexity
- Plot each metric vs. α
**Expected:**
  - COGS accuracy peaks at α ≈ 0 (flat is best for compositionality)
  - Sarcasm detection peaks at α > 0 (non-flat helps for pragmatics)
  - LM perplexity has intermediate optimum (language has both compositional and pragmatic components)
**Priority:** ★★★★★ — cleanest ablation, directly tests the core claim
**Risk:** The interpolation may not be well-defined (linear interpolation of matrix-valued objects). Alternative: use a per-head gating mechanism.

### HF4.2: Per-Head Flatness Specialization

**Derived from:** Sub-claim 1 (language has both compositional and pragmatic components)
**Prediction:** In a model where each attention head can independently choose flat or non-flat transport (via a learned gate), different heads should specialize: some heads learn flat transport (compositional role) and others learn non-flat transport (pragmatic/contextual role). The flat heads should attend to syntactic structure; the non-flat heads should attend to discourse/pragmatic cues.
**Null hypothesis:** All heads converge to the same flatness level, or flatness does not correlate with head function.
**Operationalization:**
- Add a per-head learnable scalar gate g_h ∈ [0,1] (sigmoid-activated):
  Ω_ij^{(h)} = g_h · exp(φ_i)·exp(−φ_j) + (1−g_h) · Ω_ij^{free}
- After training, classify heads by learned g_h value
- Probe flat vs. non-flat heads for syntactic vs. pragmatic sensitivity using diagnostic classifiers (Hewitt & Manning 2019 structural probes)
**Expected:** Strong correlation between g_h and head function: high g_h (flat) ↔ syntactic; low g_h (non-flat) ↔ discourse
**Priority:** ★★★★★ — most informative single experiment; could go in the main paper
**Risk:** The gating mechanism adds parameters. Control with a fixed-gate ablation.

### HF4.3: Holonomy Grows with Context Window on Ambiguous Inputs

**Derived from:** Sub-claim 1 (path-dependent meaning requires contextual relay)
**Prediction:** For ambiguous sentences (garden-path, structural ambiguity), the holonomy in a non-flat model should increase as the context window grows (more paths = more opportunity for path-dependence). For unambiguous sentences, holonomy should remain near zero regardless of context length.
**Null hypothesis:** Holonomy is independent of context length, or grows uniformly for all inputs.
**Operationalization:**
- Use garden-path sentences (e.g., "The horse raced past the barn fell") and matched unambiguous controls
- Train non-flat model, evaluate holonomy at context lengths {16, 32, 64, 128, 256}
- Measure interaction: holonomy × ambiguity × context_length
**Expected:** Significant three-way interaction: holonomy grows with context only for ambiguous inputs
**Priority:** ★★★★☆

---

## Category F5: Novel Experimental Designs

### HF5.1: Holonomy as a Pragmatic Inference Detector

**Derived from:** All three sub-claims
**Prediction:** Holonomy magnitude in a non-flat model can serve as an unsupervised detector of pragmatic inference — token triples with H_ijk ≠ I are precisely those where meaning cannot be composed path-independently.
**Operationalization:**
- Train a non-flat gauge architecture on a large LM corpus
- Compute holonomy for all token triples in a held-out set
- Use holonomy as a feature (without any pragmatics-specific supervision) to predict:
  - Sarcasm labels (SemEval 2018)
  - Implicature resolution (GRICE)
  - Metaphor detection (VU Amsterdam Metaphor Corpus)
- Compare with attention-weight-based and representation-based baselines
**Expected:** Holonomy-based features achieve non-trivial F1 (> 0.5) on pragmatic tasks, competitive with supervised attention probes
**Priority:** ★★★★☆ — novel contribution; connects geometry to pragmatics
**Risk:** Holonomy may capture non-pragmatic structure. Control by comparing holonomy patterns on literal vs. figurative uses of the same expressions.

### HF5.2: Flat Bundle Violation Predicts Transformer Failure Modes

**Derived from:** Sub-claim 3 (transformers matched to flat regime)
**Prediction:** Standard transformers should systematically fail on tasks where language violates path-independence. Specifically, transformer performance should degrade on inputs where a non-flat model learns high holonomy.
**Operationalization:**
- Train a non-flat model on a diverse corpus; identify high-holonomy inputs
- Evaluate a standard (flat) transformer on the same inputs
- Partition inputs by holonomy magnitude into deciles
- Test whether standard transformer accuracy anti-correlates with holonomy decile
**Expected:** Monotonic decrease in standard transformer accuracy with increasing holonomy (ρ < −0.5)
**Priority:** ★★★★★ — directly falsifiable, strong narrative for the paper
**Risk:** Holonomy may correlate with difficulty for reasons other than flatness. Control by matching perplexity within holonomy deciles.

### HF5.3: Synthetic Languages with Controlled Holonomy

**Derived from:** All three sub-claims
**Prediction:** On synthetic languages with explicitly controlled path-dependence, flat architectures should excel when the language is flat and non-flat architectures should excel when the language has non-trivial holonomy.
**Operationalization:**
- Design synthetic languages on a token graph:
  - **Flat language:** Meaning transport satisfies cocycle condition by construction. Token k's meaning relative to i is the same whether computed directly or via j.
  - **Non-flat language:** Inject controlled holonomy — meaning of token k relative to i depends on the relay path. E.g., word sense depends on which other words mediated the interpretation.
- Vary holonomy strength continuously: H_ijk = exp(ε · A_ijk) where ε controls departure from flatness
- Train flat and non-flat architectures at each ε
- Plot accuracy vs. ε for both architectures
**Expected:** Crossover point at ε* where non-flat surpasses flat. For ε < ε*, flat architecture wins; for ε > ε*, non-flat wins.
**Priority:** ★★★★★ — cleanest possible test, fully controls confounds
**Risk:** Synthetic language may not capture real language structure. Mitigate by designing the synthetic language to have naturalistic statistical properties (Zipfian distribution, hierarchical phrase structure).

### HF5.4: Developmental Trajectory — Flatness Emerges During Training

**Derived from:** Sub-claim 2 (functional pressure toward flatness)
**Prediction:** If a non-flat model is initialized with random (non-flat) transport and trained on natural language, the holonomy should decrease during training as the model discovers that language is approximately flat. The rate of holonomy decrease should correlate with the learning rate of compositional structure.
**Null hypothesis:** Holonomy does not systematically decrease during training, or decreases uniformly without correlation to compositionality learning.
**Operationalization:**
- Initialize non-flat model with random Ω_ij (average ‖H_ijk − I‖_F > 0)
- Track holonomy during training at checkpoints {1k, 5k, 10k, 50k, 100k, 500k steps}
- Simultaneously track COGS-style compositional generalization at each checkpoint
- Compute temporal correlation between holonomy decrease and compositionality increase
**Expected:** Strong negative temporal correlation (holonomy ↓ as compositionality ↑)
**Priority:** ★★★★☆ — beautiful result if confirmed; connects learning dynamics to geometry

---

## Prioritization Matrix

| ID | Hypothesis | Impact | Feasibility | Novelty | Score |
|----|-----------|--------|-------------|---------|-------|
| HF4.2 | Per-head flatness specialization | Very High | Medium | Very High | ★★★★★ |
| HF1.2 | Non-flat learns holonomy on pragmatic tasks | Very High | Medium | Very High | ★★★★★ |
| HF5.3 | Synthetic languages with controlled holonomy | Very High | Medium | Very High | ★★★★★ |
| HF4.1 | Cocycle relaxation ablation | High | Easy | High | ★★★★★ |
| HF5.2 | Flat violation predicts transformer failure | Very High | Medium | High | ★★★★★ |
| HF1.1 | Flat suffices for COGS/SCAN | High | Easy | Medium | ★★★★★ |
| HF2.3 | Holonomy penalty scaling | High | Easy | High | ★★★★☆ |
| HF1.3 | Holonomy ↔ compositionality correlation | High | Medium | High | ★★★★☆ |
| HF5.1 | Holonomy as pragmatic detector | High | Medium | Very High | ★★★★☆ |
| HF3.2 | Universal flatness of compositional core | Very High | Hard | Very High | ★★★★☆ |
| HF5.4 | Flatness emerges during training | High | Medium | High | ★★★★☆ |
| HF4.3 | Holonomy × ambiguity × context length | Medium | Medium | High | ★★★★☆ |
| HF2.1 | Flat advantage scales with compositionality | High | Medium | Medium | ★★★★☆ |
| HF2.2 | Flat bundle data efficiency | High | Medium | Medium | ★★★★☆ |
| HF3.1 | Holonomy varies with pragmatic load | High | Hard | High | ★★★☆☆ |
| HF3.3 | Word order freedom ↔ gauge frame diversity | Medium | Medium | Medium | ★★★☆☆ |

---

## Recommended Experimental Sequence

**Phase 1 — Foundation (single GPU, weeks):**
1. HF1.1: Evaluate existing Gauge-Transformer on COGS/SCAN (baseline establishment)
2. HF4.1: Cocycle relaxation ablation (α sweep, most informative for least compute)
3. HF2.3: Holonomy penalty λ sweep on non-flat architecture

**Phase 2 — Core Tests (moderate compute, months):**
4. HF1.2: Non-flat architecture on pragmatic tasks (the headline experiment)
5. HF4.2: Per-head flatness gating (cleanest architecture-level test)
6. HF5.3: Synthetic language with controlled holonomy (fully controlled experiment)

**Phase 3 — Extensions (multi-GPU, longer term):**
7. HF5.2: Transformer failure mode prediction via holonomy
8. HF5.4: Developmental trajectory of flatness
9. HF3.2: Cross-linguistic universal flatness (requires multilingual infrastructure)
10. HF5.1: Holonomy as unsupervised pragmatic detector

---

## Key Non-Flat Architecture Design

Several hypotheses require a "non-flat gauge architecture." Here is a concrete specification:

```python
class NonFlatGaugeTransport(nn.Module):
    """Edge-independent gauge transport with unconstrained holonomy.

    Unlike the flat Ω_ij = exp(φ_i)·exp(-φ_j), this parameterizes
    Ω_ij as an independent GL(K) element per edge, allowing H_ijk ≠ I.
    """
    def __init__(self, d_model, K, n_heads):
        super().__init__()
        self.K = K
        # Produce K×K transport matrix from token pair representations
        self.transport_net = nn.Sequential(
            nn.Linear(2 * d_model, 4 * K * K),
            nn.GELU(),
            nn.Linear(4 * K * K, K * K),
        )
        # Initialize near identity (start near flat)
        nn.init.zeros_(self.transport_net[-1].weight)
        nn.init.zeros_(self.transport_net[-1].bias)

    def forward(self, h_i, h_j):
        """Compute Ω_ij from token representations.

        Args:
            h_i: (B, N, d_model) query token representations
            h_j: (B, N, d_model) key token representations
        Returns:
            Omega: (B, N, N, K, K) transport operators
        """
        B, N, D = h_i.shape
        # Broadcast to all pairs
        h_i_exp = h_i.unsqueeze(2).expand(-1, -1, N, -1)  # (B, N, N, D)
        h_j_exp = h_j.unsqueeze(1).expand(-1, N, -1, -1)  # (B, N, N, D)
        pair_repr = torch.cat([h_i_exp, h_j_exp], dim=-1)  # (B, N, N, 2D)

        # Produce transport matrix (initialized near zero → Ω ≈ I)
        omega_flat = self.transport_net(pair_repr)  # (B, N, N, K²)
        Omega = omega_flat.reshape(B, N, N, self.K, self.K)
        Omega = Omega + torch.eye(self.K, device=Omega.device)  # Residual: Ω = I + learned

        return Omega

    def compute_holonomy(self, Omega, triples):
        """Compute holonomy H_ijk = Ω_ij · Ω_jk · Ω_ki for given triples."""
        i, j, k = triples  # index tensors
        H = Omega[:, i, j] @ Omega[:, j, k] @ Omega[:, k, i]
        return H  # (B, n_triples, K, K), should be ≈ I if flat
```

This design ensures:
- **Near-flat initialization** (residual connection to identity)
- **Content-dependent transport** (conditions on token representations)
- **Measurable holonomy** via `compute_holonomy()`
- **Smooth interpolation** to flat regime (if learned perturbation → 0)

---

## Category F6: RG Universality — Transformer as IR Fixed Point

**Derived from:** The three-limit hierarchy (Section 4.6) and the renormalization group analysis in `derivations/rg_universality_derivation.py`.

**Core Conjecture:** *The standard transformer is a stable infrared fixed point of a renormalization group flow on the space of gauge-theoretic VFE models. The gauge VFE and standard transformers belong to the same universality class, with the gauge VFE providing an O(√K) sample-efficiency advantage from geometric inductive bias.*

**Theoretical basis:** Under coarse-graining (clustering tokens into meta-agents), three coupling constants control the distance from the transformer limit:
- g₁ (anisotropy): scaling dimension y₁ = −1/2 → **irrelevant**
- g₂ (gauge variation): scaling dimension y₂ = −1 → **irrelevant**
- g₃ (holonomy): scaling dimension y₃ = −2 → **irrelevant**

All negative → the transformer limit is a **stable** IR fixed point. See `derivations/rg_universality_derivation.py` for the full derivation and Monte Carlo verification.

---

### HF6.1: Sample-Efficiency Advantage Grows with K

**Derived from:** RG universality theorem, Part (v) — efficiency gap scales as O(K²) absorbed degrees of freedom.
**Prediction:** At matched parameter count, the gauge VFE achieves a given perplexity with fewer training tokens than a standard transformer, and this advantage grows with the belief dimension K (embedding dimension per head).
**Null hypothesis:** Sample efficiency is independent of K, or the transformer is more efficient at all K.
**Operationalization:**
- Train gauge VFE and standard transformer at K ∈ {8, 16, 32, 64} on WikiText-103
- For each K, record tokens-to-reach-target-PPL (e.g., target PPL = 100)
- Compute ratio R(K) = tokens_transformer / tokens_VFE
- Fit R(K) = a · K^δ
**Expected:** δ > 0 (advantage grows with K); theoretical prediction δ ≈ 0.5
**Compute:** ~4 GPU-days per K value on RTX 5090 (16 runs total, ~2 weeks)
**Priority:** ★★★★★ — the most direct test of the universality efficiency gap

### HF6.2: Scaling Exponents Match (Same Universality Class)

**Derived from:** RG universality theorem, Part (iii) — both flow to same fixed point.
**Prediction:** The loss-vs-tokens scaling exponent β in PPL(D) = A·D^{−β} + PPL_∞ should be the **same** for gauge VFE and standard transformer (to within statistical error), because they share the same universality class. The prefactor A should differ (gauge VFE has smaller A due to geometric bias).
**Null hypothesis:** β_VFE ≠ β_TF (different universality classes).
**Operationalization:**
- Train both architectures on WikiText-103 with 10 checkpoints logarithmically spaced in tokens
- Fit power-law scaling curves using Bayesian regression (see `scripts/rg_universality_bayesian.py`)
- Compare β_VFE vs β_TF with 95% credible intervals
- Compare prefactors A_VFE vs A_TF
**Expected:** |β_VFE − β_TF| < 0.02 (same exponent); A_VFE < A_TF (better prefactor)
**Compute:** ~2 GPU-days on RTX 5090 (can reuse HF6.1 runs)
**Priority:** ★★★★★ — directly tests universality

### HF6.3: Attention Graph Coarse-Graining Exponents

**Derived from:** RG universality theorem, Parts (i)–(ii) — scaling dimensions at the fixed point.
**Prediction:** Under iterative spectral coarse-graining of the attention graph (see `scripts/rg_universality_networkx.py`), the coupling constants decay as:
  - g₁(ζ) ∝ b^{−ζ/2}  (anisotropy, y₁ = −1/2)
  - g₂(ζ) ∝ b^{−ζ}    (gauge variation, y₂ = −1)
  - g₃(ζ) ∝ b^{−2ζ}   (holonomy, y₃ = −2)
where ζ is the coarse-graining level and b is the scale factor.
**Null hypothesis:** Couplings do not decay as power laws, or exponents differ significantly from predictions.
**Operationalization:**
- Extract attention matrices from trained gauge VFE and standard transformer
- Run `scripts/rg_universality_networkx.py` at 3–5 coarse-graining levels
- Fit log–log slopes for each coupling constant
- Compare measured exponents to predicted y₁, y₂, y₃
**Expected:** Measured exponents within 30% of predictions (allowing for finite-size effects)
**Compute:** Negligible (post-hoc analysis of existing checkpoints)
**Priority:** ★★★★☆ — elegant but requires careful finite-size correction

### HF6.4: Emergent Anisotropy Under Coarse-Graining

**Derived from:** RG universality theorem, Part (iv) — coarse-graining generates anisotropy from within-cluster mean variance.
**Prediction:** Even when the microscopic theory is isotropic (Σ_i = σ²I), the meta-agent covariances Σ_A are anisotropic, with anisotropy magnitude proportional to the within-cluster variance of means. Standard transformers must absorb this emergent structure into W_Q, W_K; the gauge VFE tracks it explicitly in Σ_i.
**Null hypothesis:** Meta-agent covariances are approximately isotropic, or anisotropy is uncorrelated with cluster structure.
**Operationalization:**
- For a trained gauge VFE, extract per-token beliefs (μ_i, Σ_i)
- Cluster tokens by attention modularity
- Compute meta-agent covariances: Σ_A = avg(Σ_i∈A) + Var_A(μ)
- Measure anisotropy: g₁^{emergent} = ||Σ_A − (tr Σ_A / K)·I|| / (tr Σ_A / K)
- For the standard transformer, extract hidden states and compute analogous within-cluster variance
**Expected:** g₁^{emergent} > 0.1 (significant emergent anisotropy); correlates with cluster separation
**Compute:** Negligible (post-hoc analysis)
**Priority:** ★★★★☆ — tests the mechanism, not just the prediction

### HF6.5: Compute Crossover Point

**Derived from:** RG universality corollary — crossover at C* = O(K² · V).
**Prediction:** There exists a total compute budget C* below which the gauge VFE achieves better perplexity, and above which the standard transformer catches up. C* should scale as O(K²).
**Null hypothesis:** No crossover exists (one architecture dominates at all compute budgets).
**Operationalization:**
- Train both architectures with compute budgets spanning 3 orders of magnitude
- Plot iso-perplexity curves in (tokens, parameters) space
- Identify crossover point C* where curves intersect
- Repeat for K ∈ {16, 32, 64} and test C* ∝ K²
**Expected:** C* exists and scales roughly as K²
**Compute:** ~1–2 GPU-weeks on RTX 5090 (can be distributed across K values)
**Priority:** ★★★★☆ — strong result if confirmed, but compute-intensive

### HF6.6: LayerNorm as Isotropy Projector (Emergent RG Mechanism)

**Derived from:** Part 1g of the RG derivation — standard transformers use LayerNorm to project back to approximate isotropy.
**Prediction:** Removing LayerNorm from a standard transformer should degrade performance more on high-anisotropy inputs (where the transformer needs LayerNorm to project emergent anisotropy back to isotropy) than on low-anisotropy inputs. In contrast, the gauge VFE (which tracks Σ explicitly) should be robust to LayerNorm removal.
**Null hypothesis:** LayerNorm removal degrades performance uniformly, independent of input anisotropy.
**Operationalization:**
- Train standard transformer with and without LayerNorm
- Partition inputs by effective anisotropy of hidden states (measured as ||h − mean(h)·1|| / ||h||)
- Compare per-decile performance degradation
- Train gauge VFE as control (no LayerNorm needed)
**Expected:** Performance degradation anti-correlates with input isotropy for transformer; no correlation for gauge VFE
**Compute:** ~2 GPU-days on RTX 5090
**Priority:** ★★★☆☆ — mechanistic insight, moderate novelty

---

## Category F7: At-Scale Predictions (Future Work for Well-Resourced Labs)

These predictions require compute beyond a single workstation but represent the most impactful potential results. They are included as **testable predictions** for the manuscript's future-work section.

### HF7.1: Chinchilla-Law Correction from Gauge Geometry

**Prediction:** The Chinchilla scaling law PPL(N, D) = A·N^{−α}·D^{−β} + PPL_∞ should acquire a geometric correction for the gauge VFE:
  PPL_VFE(N, D, K) = A·N^{−α}·D^{−β}·(1 − γ/K^{1/2}) + PPL_∞
with γ > 0, meaning the gauge VFE achieves better PPL at matched (N, D).
**Required compute:** 100+ GPU-days (training at 5+ scales from 10M to 1B parameters)
**Testable by:** Google, Meta, Anthropic, DeepMind, or well-funded academic labs

### HF7.2: Gauge VFE Reaches GPT-2 Perplexity at 1/3 the Parameters

**Prediction:** A gauge VFE model with ~50M parameters and K=128 should match GPT-2 (117M parameters) on WikiText-103 perplexity, because the geometric inductive bias compensates for the parameter gap.
**Required compute:** ~50 GPU-days (single large-K training run with hyperparameter tuning)
**Testable by:** Well-funded academic groups or industry research labs

### HF7.3: Universality Class Extends to Vision Transformers

**Prediction:** The RG universality result is not specific to language — vision transformers (ViT) should also be in the same universality class as a gauge VFE over image patches. The sample-efficiency advantage should be even larger for images (higher intrinsic dimensionality → more anisotropy to absorb).
**Required compute:** 200+ GPU-days (training gauge VFE on ImageNet at multiple scales)
**Testable by:** Any group with sufficient compute and interest in geometric deep learning

### HF7.4: Multi-Modal Gauge VFE Has Universal Transport

**Prediction:** A gauge VFE trained on multi-modal data (text + images) should learn gauge frames φ_i that factorize into modality-specific and modality-universal components. The universal component should capture cross-modal compositional semantics (flat transport), while the modality-specific component captures domain-specific non-compositional structure.
**Required compute:** 500+ GPU-days
**Testable by:** Large-scale multi-modal research groups

---

## Updated Prioritization Matrix (Including F6–F7)

| ID | Hypothesis | Impact | Feasibility | Novelty | Score |
|----|-----------|--------|-------------|---------|-------|
| **HF6.1** | **Sample efficiency grows with K** | **Very High** | **Medium** | **Very High** | **★★★★★** |
| **HF6.2** | **Scaling exponents match** | **Very High** | **Medium** | **Very High** | **★★★★★** |
| HF4.2 | Per-head flatness specialization | Very High | Medium | Very High | ★★★★★ |
| HF1.2 | Non-flat learns holonomy on pragmatic tasks | Very High | Medium | Very High | ★★★★★ |
| HF5.3 | Synthetic languages with controlled holonomy | Very High | Medium | Very High | ★★★★★ |
| HF4.1 | Cocycle relaxation ablation | High | Easy | High | ★★★★★ |
| HF5.2 | Flat violation predicts transformer failure | Very High | Medium | High | ★★★★★ |
| **HF6.3** | **RG coarse-graining exponents** | **High** | **Easy** | **Very High** | **★★★★☆** |
| **HF6.4** | **Emergent anisotropy mechanism** | **High** | **Easy** | **High** | **★★★★☆** |
| **HF6.5** | **Compute crossover point** | **High** | **Hard** | **High** | **★★★★☆** |
| HF2.3 | Holonomy penalty scaling | High | Easy | High | ★★★★☆ |
| HF1.3 | Holonomy ↔ compositionality correlation | High | Medium | High | ★★★★☆ |
| **HF6.6** | **LayerNorm as isotropy projector** | **Medium** | **Easy** | **High** | **★★★☆☆** |
| HF3.2 | Universal flatness of compositional core | Very High | Hard | Very High | ★★★★☆ |
| *HF7.1* | *Chinchilla correction (at-scale)* | *Very High* | *Hard* | *Very High* | *Future* |
| *HF7.2* | *GPT-2 parity at 1/3 params (at-scale)* | *Very High* | *Hard* | *High* | *Future* |
| *HF7.3* | *ViT universality (at-scale)* | *Very High* | *Hard* | *Very High* | *Future* |

---

## Recommended Experimental Sequence (Updated)

**Phase 0 — Post-Hoc Analysis (no new training, days):**
1. HF6.3: RG coarse-graining exponents from existing checkpoints
2. HF6.4: Emergent anisotropy measurement from existing beliefs
3. HF6.6: LayerNorm ablation analysis on existing hidden states

**Phase 1 — Foundation (single GPU, 1–2 weeks):**
4. HF6.1: Sample-efficiency comparison at K ∈ {8, 16, 32, 64}
5. HF6.2: Scaling exponent comparison (reuse HF6.1 runs)
6. HF4.1: Cocycle relaxation ablation

**Phase 2 — Core Tests (single GPU, 1–2 months):**
7. HF1.2: Non-flat architecture on pragmatic tasks
8. HF4.2: Per-head flatness gating
9. HF5.3: Synthetic language with controlled holonomy
10. HF6.5: Compute crossover point (extended training runs)

**Phase 3 — Manuscript Predictions (future work):**
11. HF7.1–7.4: At-scale predictions for well-resourced labs
