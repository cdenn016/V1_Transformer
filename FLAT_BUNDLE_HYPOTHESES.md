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
