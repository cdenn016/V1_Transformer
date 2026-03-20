# Diachronic Language Evolution via Pure VFE Transformer

## Motivation

The Pure VFE Transformer's prior bank (`prior_mu[v]`, `prior_Sigma[v]`, `prior_Omega[v]`) evolves via M-step natural gradient updates — the slow s_i timescale in the FEP hierarchy. When trained on time-ordered historical text, the trajectory `prior_mu[v](t)` through belief space IS the semantic drift of word v.

No existing approach models diachronic language change as a continuous dynamical process. All prior work — HistWords (Stanford), HistBERT, DPLM — trains separate snapshot models per era and post-hoc aligns them via Procrustes or similar. The Gauge-Transformer offers something fundamentally different:

1. **Continuous s_i(t) trajectories** — not aligned snapshots, actual learned dynamics
2. **Gauge curvature as a signal** — periods of rapid language change should show increased curvature in the connection Ω
3. **Bayesian uncertainty** — Σ_v(t) tracks how confident the model is about each word's meaning, and should increase during semantic transitions
4. **Semantic field co-evolution** — the gauge transport Ω_ij encodes how relationships between words transform, not just that individual words moved

### The FEP hierarchy in language terms

| Variable | Timescale | Language analogy |
|----------|-----------|-----------------|
| **q_i** (beliefs) | Milliseconds | Parsing this sentence right now |
| **s_i = p_i** (models/priors) | Years/decades | What "cat" means in English |
| **h** (hyper-prior) | Centuries | Universal grammar / shared structure |

The q_i change every forward pass. The s_i — `prior_mu[v]` — change via M-step across training batches. Training chronologically on historical text lets us observe this slow evolution directly.

---

## Primary Dataset: CCOHA

**Clean Corpus of Historical American English** (Alatrash et al., 2020)
- 475M words, ~600M BPE tokens
- 1820–2019, decade-tagged
- Balanced across 4 genres: fiction, magazines, newspapers, academic
- Standard benchmark for diachronic computational linguistics
- Cleaned version of COHA with consistent lemmatization

### Supplementary datasets (future)
| Dataset | Period | Size | Use case |
|---------|--------|------|----------|
| Chronicling America | 1836–1922 | 21M pages | Daily resolution |
| Common Corpus (HF) | Mixed historical | 500B words | Scale experiments |
| DUKweb | 1996–2013 | 1.3T tokens | Modern web evolution |
| Google Books Ngrams | 1500s–2019 | Massive | Frequency validation |

---

## Architecture: What Changes and What Doesn't

### Unchanged (core VFE machinery)
- `transformer/pure_vfe/model.py` — PureVFETransformer
- `transformer/pure_vfe/inference.py` — E-step (VFE descent)
- `transformer/pure_vfe/learning.py` — M-step (prior bank updates)
- `transformer/pure_vfe/gaussians.py` — Analytic KL, gradients, SPD retraction
- `transformer/pure_vfe/gauge.py` — GL(K) transport, gauge natural gradient
- `transformer/pure_vfe/csrc/` — CUDA kernels

The key insight: **only the data pipeline and outer training loop change.** The VFE machinery is time-agnostic. We simply feed it data in temporal order and snapshot the prior bank.

### New: `transformer/pure_vfe/diachronic/` subpackage

```
transformer/pure_vfe/diachronic/
├── __init__.py
├── dataset.py              # CCOHA loader with decade metadata
├── snapshot.py             # Prior bank snapshotting
├── train_diachronic.py     # Chronological training loop
├── metrics.py              # Semantic drift metrics
├── evaluation.py           # Ground truth: known historical shifts
├── visualize.py            # UMAP trajectories, drift plots
└── tests/
    ├── __init__.py
    ├── test_snapshot.py    # Snapshot round-trip tests
    └── test_metrics.py     # Metric sanity tests
```

### Minor modification: `transformer/pure_vfe/config.py`
Add optional diachronic fields:
```python
# Diachronic training (optional)
diachronic_mode: bool = False
snapshot_dir: str = 'snapshots/'
snapshot_interval: int = 100  # steps between snapshots
```

---

## Data Pipeline: `diachronic/dataset.py`

### DiachronicConfig

```python
@dataclass
class DiachronicConfig:
    data_dir: str                          # Path to CCOHA text files by decade
    decade_range: Tuple[int, int] = (1820, 2019)
    genres: List[str] = field(default_factory=lambda: ['fiction', 'news', 'magazine', 'academic'])
    mode: str = 'epoch_per_decade'         # Training mode (see below)
    seq_len: int = 64
    batch_size: int = 8
    tokenizer: str = 'gpt2'               # tiktoken GPT-2 (50,257 vocab)
    cache_dir: str = '~/.cache/diachronic_cache/'
```

### Training Modes

| Mode | Description | Use case |
|------|-------------|----------|
| `epoch_per_decade` | Train N epochs on 1820s, then 1830s, ..., then 2010s | Primary diachronic mode |
| `chronological` | Serve ALL data in strict temporal order, single pass | Streaming / online learning |
| `shuffled` | Standard random sampling across all decades | Baseline comparison |

### Expected CCOHA directory layout

```
ccoha/
├── 1820/
│   ├── fiction/
│   ├── magazine/
│   ├── newspaper/
│   └── academic/
├── 1830/
│   └── ...
...
└── 2010/
    └── ...
```

### Tokenization

Reuse existing tiktoken infrastructure from `transformer/data/datasets.py`:
- GPT-2 tokenizer (50,257 vocab)
- Token caching at `~/.cache/diachronic_cache/{decade}_{genre}_tokens.pt`
- Consistent vocab across all decades (subword tokenization handles neologisms)

---

## Prior Bank Snapshotting: `diachronic/snapshot.py`

### PriorBankSnapshot

```python
@dataclass
class PriorBankSnapshot:
    prior_mu: torch.Tensor       # [V, K]
    prior_Sigma: torch.Tensor    # [V, K, K]
    prior_Omega: torch.Tensor    # [V, H, K_h, K_h]
    pos_Omega: torch.Tensor      # [N_max, H, K_h, K_h]
    decade: int                  # e.g. 1820
    step: int                    # global training step
    timestamp: float             # wall clock time
    ce_loss: float               # cross-entropy at snapshot time
    vfe: float                   # VFE at snapshot time
```

### SnapshotManager

```python
class SnapshotManager:
    def __init__(self, snapshot_dir: str, interval: int = 100):
        """
        interval: save every N steps AND at every decade boundary
        Storage: {snapshot_dir}/{decade}_{step:06d}.pt
        """

    def save_snapshot(self, model, decade, step, ce_loss=0.0, vfe=0.0):
        """Save current prior bank state."""

    def load_snapshot(self, path) -> PriorBankSnapshot:
        """Load single snapshot."""

    def load_trajectory(self) -> List[PriorBankSnapshot]:
        """Load all snapshots, sorted by (decade, step)."""

    def get_decade_boundaries(self) -> List[PriorBankSnapshot]:
        """Return only the first snapshot of each new decade."""
```

Snapshots are compact — just 4 tensors + metadata. At V=50257, K=32, H=4, K_h=8:
- `prior_mu`: 50257 × 32 × 4 bytes = 6.4 MB
- `prior_Sigma`: 50257 × 32 × 32 × 4 bytes = 206 MB
- `prior_Omega`: 50257 × 4 × 8 × 8 × 4 bytes = 41 MB
- Total per snapshot: ~254 MB
- 20 decade boundaries + every 1000 steps ≈ ~100 snapshots ≈ 25 GB

For lighter storage: option to save only `prior_mu` (6.4 MB per snapshot) for trajectory analysis, with full snapshots at decade boundaries only.

---

## Training Loop: `diachronic/train_diachronic.py`

```python
def train_diachronic(config: PureVFEConfig,
                      diachronic_config: DiachronicConfig,
                      n_epochs_per_decade: int = 5,
                      snapshot_interval: int = 1000,
                      save_dir: str = 'checkpoints/diachronic/'):
    """
    Train Pure VFE Transformer with chronological curriculum.

    Algorithm:
        model = PureVFETransformer(config)
        snapshot_mgr = SnapshotManager(save_dir)

        for decade in [1820, 1830, ..., 2010]:
            data = load_decade(decade, diachronic_config)
            snapshot_mgr.save_snapshot(model, decade, step)  # decade boundary

            for epoch in range(n_epochs_per_decade):
                for inputs, targets in make_batches(data, config.batch_size):
                    logits, ce_loss, vfe = model.update(inputs, targets)
                    step += 1

                    if step % snapshot_interval == 0:
                        snapshot_mgr.save_snapshot(model, decade, step)

            # Log per-decade summary
            log_decade_metrics(model, decade, ...)

        # Final analysis
        trajectory = snapshot_mgr.load_trajectory()
        results = evaluate_known_shifts(trajectory, tokenizer)
    """
```

### Key design: the training loop is the ONLY thing that changes

The `model.update(inputs, targets)` call is identical to standard training. The E-step runs VFE descent, the M-step updates priors. We just control WHICH data goes in and WHEN.

---

## Semantic Drift Metrics: `diachronic/metrics.py`

### Per-token metrics (computed from snapshot trajectories)

```python
def cosine_drift(trajectory: List[PriorBankSnapshot], token_id: int) -> List[float]:
    """
    Cosine distance of prior_mu[v] between consecutive snapshots.
    drift[t] = 1 - cos(μ_v(t), μ_v(t+1))
    """

def drift_velocity(trajectory, token_id) -> List[float]:
    """
    Rate of semantic change: ||Δμ_v|| / Δdecade
    Normalized by average token drift to control for global learning rate.
    """

def neighborhood_stability(trajectory, token_id, k=10) -> List[float]:
    """
    Jaccard overlap of k-nearest neighbors across consecutive snapshots.
    J(t) = |N_k(v,t) ∩ N_k(v,t+1)| / |N_k(v,t) ∪ N_k(v,t+1)|
    High stability = word's semantic neighborhood is preserved.
    """

def uncertainty_evolution(trajectory, token_id) -> List[float]:
    """
    Trace of prior_Sigma[v] over time.
    tr(Σ_v(t)) — does uncertainty increase during rapid change?
    """
```

### Pairwise / field-level metrics

```python
def gauge_curvature(trajectory, token_id_i, token_id_j) -> List[float]:
    """
    Frobenius norm of gauge frame change between two tokens.
    ||Ω_i(t+1) Ω_j(t+1)⁻¹ - Ω_i(t) Ω_j(t)⁻¹||_F
    High curvature = the RELATIONSHIP between words is transforming.
    """

def semantic_field_coherence(trajectory, field_tokens: List[int], k=5) -> List[float]:
    """
    Average pairwise cosine similarity within a semantic field over time.
    Tests Trier's word field theory: do fields co-evolve?
    """

def svd_spectrum(trajectory, token_id) -> List[np.ndarray]:
    """
    Singular values of prior_Omega[v] per head over time.
    Tracks gauge frame structure evolution.
    """
```

---

## Ground Truth Evaluation: `diachronic/evaluation.py`

### Known historical semantic shifts

```python
KNOWN_SHIFTS = {
    # word: (shift_type, (start_decade, end_decade), description)
    'awful':     ('pejoration',     (1800, 1900), 'awe-inspiring → terrible'),
    'nice':      ('amelioration',   (1700, 1900), 'ignorant → pleasant'),
    'gay':       ('semantic_shift', (1960, 2000), 'happy → homosexual'),
    'computer':  ('semantic_shift', (1940, 1970), 'human calculator → machine'),
    'mouse':     ('polysemy',       (1960, 1990), 'animal → device'),
    'broadcast': ('metaphor',       (1920, 1950), 'scatter seed → transmit'),
    'cloud':     ('metaphor',       (2000, 2015), 'weather → computing'),
    'web':       ('metaphor',       (1990, 2000), 'spider → internet'),
    'cell':      ('polysemy',       (1980, 2010), 'biology → phone'),
    'tweet':     ('semantic_shift', (2006, 2015), 'bird sound → social media post'),
    'stream':    ('metaphor',       (2005, 2015), 'water → media delivery'),
    'tablet':    ('polysemy',       (2010, 2015), 'stone slab → computing device'),
}
```

Note: Some shifts (meat, nice) predate CCOHA's 1820 start. These serve as controls — we expect NO drift signal for completed shifts.

### Evaluation functions

```python
def evaluate_known_shifts(trajectory, tokenizer) -> dict:
    """
    For each known shift:
    1. Compute drift velocity time series for the target word
    2. Check if peak drift occurs during the known shift period
    3. Return: {word: {peak_decade, shift_period, hit: bool, velocity_timeseries}}
    """

def shift_detection_accuracy(trajectory, tokenizer, threshold=None) -> float:
    """
    Fraction of known shifts detected (peak drift in correct period).
    If threshold=None, use 2σ above mean drift velocity.
    """

def compare_to_histwords(trajectory, histwords_path, tokenizer) -> dict:
    """
    Compare our prior_mu cosine distances to HistWords embeddings
    (Stanford, 1800-2000, per-decade Word2Vec).
    Spearman correlation of per-word drift magnitudes.
    """
```

---

## Visualization: `diachronic/visualize.py`

### 1. UMAP Trajectories

```python
def plot_mu_trajectories_umap(snapshots, token_ids, tokenizer, dim=2):
    """
    Project prior_mu[v] for selected words into 2D/3D via UMAP.
    Each word traces a colored path through the space.
    Points labeled by decade. Arrows show drift direction.

    Key visual: words that undergo semantic shifts should show
    sharp turns or long jumps during the shift period.
    """
```

### 2. Drift Velocity Time Series

```python
def plot_drift_velocity(snapshots, token_ids, tokenizer):
    """
    Time series plot: x = decade, y = ||Δμ_v||/Δt for each word.
    Annotate known shift periods with shaded regions.
    Expected: peaks align with known shifts.
    """
```

### 3. Semantic Field Networks

```python
def plot_semantic_field_evolution(snapshots, field_name, field_tokens):
    """
    Grid of network graphs (one per decade).
    Nodes = words in field, edges = cosine sim > threshold.
    Shows how a semantic field restructures over time.

    Example field: {awful, terrible, wonderful, magnificent, dreadful}
    """
```

### 4. Gauge Spectrum Evolution

```python
def plot_gauge_spectrum(snapshots, token_ids):
    """
    SVD spectrum of prior_Omega[v] per head, animated over decades.
    Shows how the gauge frame's structure evolves.
    """
```

### 5. Uncertainty During Change

```python
def plot_uncertainty_evolution(snapshots, token_ids):
    """
    Dual-axis plot: drift velocity (left) and tr(Σ_v) (right) vs. decade.
    Tests H2: does uncertainty increase during rapid change?
    """
```

---

## Testable Hypotheses

### H1: Prior μ trajectories recover known semantic shifts ★★★★★

**Derived from**: s_i trajectory = semantic drift
**Prediction**: For words in KNOWN_SHIFTS, cosine drift `d(μ_v(t), μ_v(t+Δ))` peaks during the known shift period.
**Metric**: Spearman ρ between drift velocity and known shift timing.
**Null**: Drift velocity is uniform across decades.
**Success**: ρ > 0.5 with p < 0.01 for ≥ 60% of test words.
**Test**: Run on CCOHA 1820-2019; check "gay" (1960s) as easiest case.

### H2: Semantic uncertainty increases during rapid change ★★★★☆

**Derived from**: Bayesian interpretation of Σ_v
**Prediction**: `tr(Σ_v(t))` rises when `||Δμ_v/Δt||` is high.
**Metric**: Cross-correlation between `tr(Σ)` and drift velocity time series.
**Null**: No correlation between uncertainty and drift.
**Success**: Significant positive cross-correlation (r > 0.3, p < 0.05).

### H3: Gauge curvature signals structural change ★★★★☆

**Derived from**: Gauge-theoretic structure (unique to this architecture)
**Prediction**: `||ΔΩ_ij||_F` increases when the semantic relationship between words i,j changes.
**Metric**: Gauge curvature time series for known field reorganizations.
**Null**: Gauge evolution is uniform/noise-like.
**Success**: Curvature peaks correlate with known relationship changes.

### H4: Neighborhood stability predicts shift type ★★★☆☆

**Derived from**: Word field theory (Trier)
**Prediction**: Narrowing → increasing k-NN stability; broadening → decreasing stability.
**Metric**: Slope of Jaccard overlap vs. decade, compared against shift type.
**Null**: No correlation between stability trend and shift type.

### H5: Chronological training outperforms shuffled ★★★★★

**Derived from**: Adiabatic separation of timescales
**Prediction**: Chronological model achieves lower perplexity on held-out decade-matched text.
**Metric**: Per-decade test perplexity, chronological vs. shuffled.
**Null**: No difference (temporal order doesn't matter).
**Success**: Statistically significant (paired t-test, p < 0.05) lower perplexity.

### H6: Predictive semantic change ★★★☆☆ (Phase 2)

**Derived from**: Momentum in s_i dynamics
**Prediction**: `Δμ_v(T→T+1)` is predictable from `Δμ_v(T-1→T)` + neighborhood.
**Metric**: Cosine similarity between predicted and actual drift vectors.
**Null**: Drift is a random walk.

---

## Phase 2 (Future): Model Coupling γ_ij KL(s_i || Ω_ij s_j)

The unimplemented model-coupling term from the full FEP hierarchy:

```
F_slow += γ Σ_{i,j∈N(i)} KL(s_i || Ω_ij s_j)
```

This enforces that semantically related tokens' generative models co-evolve coherently under gauge transport. Implementation plan:

1. **Define semantic neighborhoods**: k-nearest neighbors in current prior_mu space
2. **Compute model-coupling gradient**: `∂/∂μ_v [γ Σ_{j∈N(v)} KL(s_v || Ω_vj s_j)]`
   - Reuse `kl_divergence()` and `grad_kl_Omega_ij()` from `gaussians.py` and `gauge.py`
3. **Add to M-step** in `learning.py`: extra gradient term alongside prior + observation
4. **Ablation**: γ=0 (Phase 1 baseline) vs γ > 0
5. **Hypothesis**: Model coupling improves semantic field coherence (H3) and shift prediction (H6)

---

## Implementation Order

| Step | File | Description |
|------|------|-------------|
| 1 | `diachronic/__init__.py` | Package init |
| 2 | `diachronic/dataset.py` | CCOHA loader + DiachronicConfig |
| 3 | `diachronic/snapshot.py` | PriorBankSnapshot + SnapshotManager |
| 4 | `diachronic/train_diachronic.py` | Chronological training loop |
| 5 | `diachronic/metrics.py` | Drift metrics (cosine, velocity, neighborhood, gauge) |
| 6 | `diachronic/evaluation.py` | Ground truth + known shift comparison |
| 7 | `diachronic/visualize.py` | UMAP trajectories, drift plots, field graphs |
| 8 | `config.py` | Add diachronic fields to PureVFEConfig |
| 9 | `diachronic/tests/` | Snapshot round-trip, metric sanity, synthetic drift |

---

## Verification Plan

1. **Snapshot round-trip**: Save and load prior bank, verify exact bitwise equality
2. **Metric sanity**: Inject synthetic drift (manually shift `prior_mu["cat"]` by known Δ), verify all metrics detect it with expected magnitudes
3. **Baseline on WikiText-2**: Run full pipeline on existing WikiText-2 data (no temporal structure) to verify infrastructure works before obtaining CCOHA
4. **Known shift smoke test**: On CCOHA, check "gay" (1960s-2000s) — strongest, most recent signal
5. **VFE monotonicity**: Confirm E-step VFE still decreases normally (no architectural changes)
6. **Gauge health**: Monitor condition numbers of `prior_Omega` across decades — alert if > 100

---

## Compute Estimates

| Component | Estimate |
|-----------|----------|
| CCOHA tokenization | ~600M BPE tokens, ~10 min (cached) |
| Sequences (seq_len=64) | ~9.4M total, ~470K per decade |
| Steps per decade (batch=8) | ~59K |
| E-step per batch (12 iters, K=32) | ~50ms (GPU) |
| Full training (5 epochs × 20 decades) | 1–2 GPU-hours (A100) |
| Snapshot storage (mu only, 100 snapshots) | ~640 MB |
| Snapshot storage (full, 20 decade boundaries) | ~5 GB |

---

## Related Work

| Paper | Year | Method | Limitation |
|-------|------|--------|------------|
| Hamilton et al. (HistWords) | 2016 | Word2Vec per decade + Procrustes | No dynamics, post-hoc alignment |
| Hu et al. (HistBERT) | 2022 | BERT on COHA | Static model, compares representations |
| Fittschen et al. | 2025 | Pretrain LMs on time slices | Separate models, no coupling |
| DHPLT | 2025 | 41-language diachronic corpus | Static snapshots, no dynamics |
| **This work** | — | Pure VFE with chronological curriculum | Continuous s_i(t) + gauge geometry |

The key differentiator: every prior approach produces **snapshots**. We produce **trajectories** with geometric structure (gauge curvature, uncertainty evolution, semantic field coherence).
