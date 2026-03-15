# Flat Bundle Experiments — Implementation Plan

**Goal:** Implement the infrastructure to test hypotheses HF1.1–HF5.4 from `FLAT_BUNDLE_HYPOTHESES.md`.

**Architecture Principle:** Minimal, surgical changes to the existing codebase. The non-flat transport is a *drop-in replacement* for `compute_transport_operators()`, controlled by new `BlockConfig` fields. All existing functionality is preserved when the new fields are at their defaults.

---

## Part 0: The Core Mathematical Insight

The existing flat transport:
```
Ω_ij = exp(φ_i · G) · exp(-φ_j · G)
```

The non-flat generalization (multiplicative perturbation):
```
Ω_ij = exp(φ_i · G) · exp(δ_ij · G) · exp(-φ_j · G)
```

where `δ_ij = f(μ_i, μ_j) ∈ ℝ^{n_gen}` is an edge-local Lie algebra element — the **connection**.

**Key properties:**
- When `δ_ij = 0`: exactly recovers the flat case (no code path changes needed)
- Holonomy: `H_ijk = exp(φ_i) · [exp(δ_ij)·exp(δ_jk)·exp(δ_ki)] · exp(-φ_i)`
  - The "connection holonomy" `C_ijk = exp(δ_ij)·exp(δ_jk)·exp(δ_ki)` is gauge-conjugated by the local frame
  - `C_ijk = I ⟺` flat (cocycle condition satisfied)
- Stays on the group manifold (all factors are group elements)
- Gradients flow cleanly through `matrix_exp` (PyTorch autograd handles this)

**Why multiplicative, not additive?** Because `δ_ij = 0` *exactly* recovers the flat case. An additive perturbation `Ω_ij = exp(φ_i)·exp(-φ_j) + Δ_ij` breaks the group structure and doesn't reduce to identity at `Δ=0` in general.

**Computational cost:** One additional `matrix_exp` per edge per head. For the irrep-decomposed architecture with per-head K ∈ {1, 3, 5, 7}, this is cheap:
- K=1 (scalar heads): `exp(δ) = e^δ`, trivially O(1)
- K=3 (SO(3)): 3×3 matrix exp via Rodrigues formula, O(1)
- K=5,7: Padé approximation on small matrices, still fast
- Total: O(N² · Σ_h K_h³) additional compute — comparable to what we already spend on the KL computation

**Memory:** `δ_ij` is `(B, N, N, n_gen)` — same order as the KL matrix `(B, N, N)` that we already materialize. No new memory bottleneck.

---

## Part 1: Core Infrastructure (Non-Flat Transport)

### 1.1 Extend `BlockConfig` with Non-Flat Fields

**File:** `transformer/core/block_config.py`

```python
# === Non-flat gauge transport (flat bundle experiments) ===
non_flat_transport: bool = False       # Enable edge-dependent connection δ_ij
cocycle_relaxation: float = 0.0        # Scale factor for δ_ij: 0=flat, 1=fully non-flat
per_head_flatness_gate: bool = False   # Learnable per-head g_h ∈ [0,1] replacing fixed cocycle_relaxation
connection_type: str = 'bilinear'      # 'bilinear' | 'mlp' — how to produce δ_ij from (μ_i, μ_j)
connection_hidden_dim: int = 64        # Hidden dim for MLP connection (ignored for bilinear)
holonomy_penalty: float = 0.0          # λ_H · Σ ‖H_ijk - I‖²_F added to loss (for HF2.3)
```

Add to `from_config()` to read from the training config dict. All defaults preserve existing behavior (non_flat_transport=False → nothing changes).

### 1.2 Connection Network Module

**New file:** `transformer/core/connection.py`

```python
class GaugeConnection(nn.Module):
    """Edge-local connection δ_ij producing Lie algebra elements for non-flat transport.

    Parameterizes the "gauge connection" on the token graph: for each edge (i,j),
    produces δ_ij ∈ ℝ^{n_gen} such that the transport operator becomes:
        Ω_ij = exp(φ_i · G) · exp(δ_ij · G) · exp(-φ_j · G)

    When δ_ij = 0, this is identically the flat transport.

    The connection controls holonomy: H_ijk = exp(φ_i) · C_ijk · exp(-φ_i)
    where C_ijk = exp(δ_ij·G) · exp(δ_jk·G) · exp(δ_ki·G).
    """

    def __init__(self, d_head, n_gen, connection_type='bilinear', hidden_dim=64):
        super().__init__()
        self.n_gen = n_gen
        self.connection_type = connection_type

        if connection_type == 'bilinear':
            # δ_ij^a = μ_i^T W^a μ_j   (one bilinear form per generator)
            # Parameter-efficient: n_gen × d_head × d_head parameters
            self.W = nn.Parameter(torch.zeros(n_gen, d_head, d_head))
            # Zero init → δ_ij = 0 → flat at initialization
        elif connection_type == 'mlp':
            self.net = nn.Sequential(
                nn.Linear(2 * d_head, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, n_gen),
            )
            # Zero-init output layer → flat at initialization
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, mu_i, mu_j):
        """Compute edge-local connection δ_ij.

        Args:
            mu_i: (B, N, d_head) query belief means
            mu_j: (B, N, d_head) key belief means

        Returns:
            delta: (B, N, N, n_gen) Lie algebra elements per edge
        """
        if self.connection_type == 'bilinear':
            # δ_ij^a = μ_i^T W^a μ_j
            # mu_i: (B, N, d) → (B, N, 1, d)
            # W: (n_gen, d, d)
            # mu_j: (B, N, d) → (B, 1, N, d)
            # Result: (B, N, N, n_gen)
            delta = torch.einsum('bid,adg,bjg->bija', mu_i, self.W, mu_j)
            return delta
        elif self.connection_type == 'mlp':
            B, N, D = mu_i.shape
            # Expand to all pairs: (B, N, N, 2D)
            mu_i_exp = mu_i.unsqueeze(2).expand(-1, -1, N, -1)
            mu_j_exp = mu_j.unsqueeze(1).expand(-1, N, -1, -1)
            pair = torch.cat([mu_i_exp, mu_j_exp], dim=-1)
            delta = self.net(pair)  # (B, N, N, n_gen)
            return delta
```

**Design choices:**
- **Bilinear** (default): `δ_ij^a = μ_i^T W^a μ_j`. Most parameter-efficient (`n_gen × d_head²` params). Captures pairwise interactions. Antisymmetrizing `W` gives `δ_ij = -δ_ji` (natural for connections).
- **MLP**: More expressive but O(N²) forward passes through the MLP. Use for experiments where bilinear is insufficient.
- **Zero initialization**: Both types start at δ_ij = 0 (flat), so the model starts from the known-good flat regime and learns to deviate only if the data warrants it.

### 1.3 Modify `compute_transport_operators()`

**File:** `transformer/core/attention.py`, function `compute_transport_operators()`

Add a new branch after the existing `# LEARNED GAUGE` section:

```python
def compute_transport_operators(
    phi, generators, enforce_orthogonal=False, gauge_mode='learned',
    # NEW: non-flat transport parameters
    connection_delta=None,      # (B, N, N, n_gen) edge-local Lie algebra elements
    cocycle_relaxation=0.0,     # Scale factor: 0=flat, 1=fully non-flat
):
    # ... existing code for trivial/constant ...

    # LEARNED GAUGE (existing code)
    phi_matrix = torch.einsum('bna,aij->bnij', phi, generators)
    exp_phi, exp_neg_phi = stable_matrix_exp_pair(phi_matrix)
    # ... skew-symmetric optimization ...

    if connection_delta is not None and cocycle_relaxation > 0:
        # ===================================================================
        # NON-FLAT TRANSPORT: Ω_ij = exp(φ_i) · exp(α·δ_ij·G) · exp(-φ_j)
        # ===================================================================
        scaled_delta = cocycle_relaxation * connection_delta  # (B, N, N, n_gen)
        delta_matrix = torch.einsum('bija,akl->bijkl', scaled_delta, generators)  # (B, N, N, K, K)

        # Matrix exponential for each edge (this is the new cost)
        B, N, _, K, _ = delta_matrix.shape
        exp_delta = stable_matrix_exp(delta_matrix.reshape(-1, K, K)).reshape(B, N, N, K, K)

        # Ω_ij = exp(φ_i) · exp(α·δ_ij) · exp(-φ_j)
        temp = torch.einsum('bikl,bijlm->bijkm', exp_phi, exp_delta)  # exp(φ_i) · exp(δ_ij)
        Omega = torch.einsum('bijkl,bjlm->bijkm', temp, exp_neg_phi)   # ... · exp(-φ_j)
    else:
        # FLAT TRANSPORT (existing code, unchanged)
        Omega = torch.einsum('bikl,bjlm->bijkm', exp_phi, exp_neg_phi)

    return {
        'exp_phi': exp_phi,
        'exp_neg_phi': exp_neg_phi,
        'Omega': Omega,
        'exp_delta': exp_delta if connection_delta is not None else None,  # For holonomy computation
    }
```

**Critical detail:** The block-diagonal and diagonal KL paths also need updating. These optimized paths currently never materialize the full Ω. For the non-flat case, the delta per edge breaks the factored structure. Two options:
1. **Fall through to the full-Omega path** when `non_flat_transport=True` (simplest, correct, but loses memory optimization)
2. **Extend the block-diagonal path** to handle per-edge delta (more work, better performance)

**Recommendation:** Start with option 1 (force full-Omega path when non-flat). Optimize later if memory is a bottleneck. Add a guard in `compute_attention_weights()`:

```python
# Force full-Omega path when non-flat transport is active
if connection_delta is not None:
    irrep_dims = None        # Disable block-diagonal path
    diagonal_covariance = False  # Disable diagonal path
    chunk_size = None
```

### 1.4 Holonomy Computation Module

**New file:** `transformer/analysis/holonomy.py`

```python
def compute_holonomy(exp_delta, generators, triples=None, sample_size=1000):
    """Compute holonomy H_ijk = exp(δ_ij·G) · exp(δ_jk·G) · exp(δ_ki·G) for token triples.

    Note: We compute the CONNECTION holonomy C_ijk (without the gauge-frame conjugation),
    since ||C_ijk - I|| = ||H_ijk - I|| (conjugation preserves Frobenius norm for unitary,
    and we care about deviation from identity, not the specific direction).

    For non-unitary GL(K), ||H_ijk - I|| ≠ ||C_ijk - I|| in general, but the
    connection holonomy is the more physically meaningful quantity (it's gauge-invariant
    up to conjugation, and the deviation from identity is gauge-covariant).

    Args:
        exp_delta: (B, N, N, K, K) — exp(δ_ij · G) per edge (from compute_transport_operators)
        generators: (n_gen, K, K) — unused here, but passed for API consistency
        triples: Optional list of (i,j,k) tuples. If None, samples randomly.
        sample_size: Number of random triples to sample if triples is None.

    Returns:
        holonomy_matrices: (B, n_triples, K, K) — H_ijk per triple
        holonomy_norms: (B, n_triples) — ‖H_ijk - I‖_F per triple
        triple_indices: (n_triples, 3) — the (i,j,k) indices used
    """
    B, N, _, K, _ = exp_delta.shape

    if triples is None:
        # Sample random triples (i,j,k) with i≠j≠k
        all_triples = []
        for _ in range(sample_size):
            ijk = torch.randperm(N)[:3]
            all_triples.append(ijk)
        triple_indices = torch.stack(all_triples)  # (sample_size, 3)
    else:
        triple_indices = torch.tensor(triples)

    i, j, k = triple_indices[:, 0], triple_indices[:, 1], triple_indices[:, 2]

    # C_ijk = exp(δ_ij) · exp(δ_jk) · exp(δ_ki)
    C = exp_delta[:, i, j] @ exp_delta[:, j, k] @ exp_delta[:, k, i]  # (B, n_triples, K, K)

    # Deviation from identity
    I_K = torch.eye(K, device=C.device, dtype=C.dtype)
    holonomy_norms = torch.norm(C - I_K, dim=(-2, -1))  # (B, n_triples) Frobenius norm

    return C, holonomy_norms, triple_indices


def holonomy_penalty_loss(exp_delta, sample_size=500):
    """Compute the holonomy penalty: λ_H · E[‖H_ijk - I‖²_F].

    This is a regularizer that pushes the model toward flatness.
    Used for HF2.3 (holonomy penalty scaling) experiments.
    """
    _, holonomy_norms, _ = compute_holonomy(exp_delta, None, sample_size=sample_size)
    return (holonomy_norms ** 2).mean()


def holonomy_statistics(exp_delta, sample_size=2000):
    """Compute summary statistics of holonomy across sampled triples.

    Returns dict with:
        - mean_holonomy: E[‖H - I‖_F]
        - max_holonomy: max ‖H - I‖_F
        - std_holonomy: std of ‖H - I‖_F
        - fraction_nontrivial: fraction of triples with ‖H - I‖_F > threshold
    """
    _, norms, _ = compute_holonomy(exp_delta, None, sample_size=sample_size)
    return {
        'mean_holonomy': norms.mean().item(),
        'max_holonomy': norms.max().item(),
        'std_holonomy': norms.std().item(),
        'fraction_nontrivial_0.01': (norms > 0.01).float().mean().item(),
        'fraction_nontrivial_0.1': (norms > 0.1).float().mean().item(),
    }
```

### 1.5 Per-Head Flatness Gating

**File:** `transformer/core/attention.py`, class `IrrepMultiHeadAttention`

Add a learnable gate per head:

```python
class IrrepMultiHeadAttention(nn.Module):
    def __init__(self, ..., per_head_flatness_gate=False, ...):
        # ... existing init ...
        self.per_head_flatness_gate = per_head_flatness_gate
        if per_head_flatness_gate:
            # One gate per head, initialized at logit=−2 → sigmoid ≈ 0.12 (near-flat)
            self.flatness_gate_logits = nn.Parameter(torch.full((n_heads,), -2.0))

    def forward(self, ...):
        # ... existing head loop ...
        for h, (head_start, head_end, irrep_dim, ...) in enumerate(self.head_specs):
            if self.per_head_flatness_gate:
                g_h = torch.sigmoid(self.flatness_gate_logits[h])
                # Scale the connection delta for this head
                head_cocycle_relaxation = g_h
            else:
                head_cocycle_relaxation = self.cocycle_relaxation  # global

            # Pass to compute_transport_operators or compute_attention_weights
            # ...
```

This lets each head independently learn its flatness level. After training, inspect `sigmoid(flatness_gate_logits)`:
- Heads with g_h ≈ 0: flat (compositional)
- Heads with g_h ≈ 1: non-flat (pragmatic)

### 1.6 Integrate into the Loss Function

**File:** `transformer/training/metrics.py`, function `compute_free_energy_loss()`

Add holonomy penalty as a 7th loss term:

```python
# === Holonomy penalty (flat bundle experiments) ===
if holonomy_penalty > 0 and exp_delta is not None:
    h_penalty = holonomy_penalty_loss(exp_delta, sample_size=500)
    loss_total += holonomy_penalty * h_penalty
    metrics['holonomy/penalty'] = h_penalty.item()

# Always log holonomy statistics when non-flat transport is active
if exp_delta is not None:
    h_stats = holonomy_statistics(exp_delta, sample_size=1000)
    for k, v in h_stats.items():
        metrics[f'holonomy/{k}'] = v
```

### 1.7 Holonomy Tracking in `forward_with_attention()`

**File:** `transformer/core/model.py`

The existing `forward_with_attention()` returns `beta`, `kl`, `mu`, `sigma`, `phi`, `mu_prior`, etc. Extend to also return `exp_delta` per layer when non-flat transport is active:

```python
# In the attention_info dict:
attention_info['exp_delta'] = exp_delta_per_layer  # List[(B, N, N, K, K)] or None
```

This feeds into the loss function and analysis tools.

---

## Part 2: Experiment-Specific Components

### 2.1 Synthetic Language with Controlled Holonomy (HF5.3)

**New file:** `transformer/data/synthetic_gauge.py`

This is the cleanest falsification test. Design a language where:
- Each token v ∈ {1, ..., V} has a hidden "local frame" g_v ∈ GL(K)
- A "connection field" A_{v→w} ∈ gl(K) per directed edge
- **Flat language (ε=0):** `A_{v→w} = log(g_v · g_w^{-1})` (derived from frames, cocycle-consistent)
- **Non-flat language (ε>0):** `A_{v→w} = log(g_v · g_w^{-1}) + ε · noise_{v→w}`
- A sentence `[t_1, ..., t_n]` has a "transport value" `T = exp(A_{t_1→t_2}) · exp(A_{t_2→t_3}) · ... · exp(A_{t_{n-1}→t_n})`

**The prediction task:**
- Compute T for the sequence
- Map T to a discrete label: `label = quantize(‖T‖_F)` into C classes
- Frame as next-token prediction: input `[t_1, ..., t_n, SEP]`, predict `[label]`

**Key property:**
- When ε=0: `T = g_{t_1} · g_{t_n}^{-1}` — depends only on first and last tokens (path-independent!). A flat model can learn this with just endpoint information.
- When ε>0: `T` depends on the full path. Intermediate tokens change the result. A flat model *cannot* learn this, but a non-flat model can.

```python
class SyntheticGaugeLanguage:
    """Synthetic language with controllable holonomy for flat bundle experiments.

    Generates (sequence, label) pairs where the label depends on parallel transport
    along the sequence. The holonomy strength ε controls how path-dependent the
    transport is:
        ε = 0: completely flat (path-independent, endpoints determine label)
        ε > 0: non-flat (full path determines label)
    """

    def __init__(self, vocab_size=100, K=3, epsilon=0.0, n_classes=4, seq_len=16, seed=42):
        rng = np.random.RandomState(seed)

        # Fixed local frames g_v ∈ GL(K) for each token
        self.frames = [random_gl_element(K, rng) for _ in range(vocab_size)]

        # Connection field A_{v→w}
        # Flat part: A_flat_{v→w} = log(g_v · g_w^{-1})
        # Non-flat noise: A_noise_{v→w} ~ N(0, I) in gl(K)
        self.A_flat = {}
        self.A_noise = {}
        for v in range(vocab_size):
            for w in range(vocab_size):
                self.A_flat[(v, w)] = scipy.linalg.logm(self.frames[v] @ np.linalg.inv(self.frames[w]))
                self.A_noise[(v, w)] = rng.randn(K, K)

        self.epsilon = epsilon
        self.K = K
        self.n_classes = n_classes
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def get_connection(self, v, w):
        """A_{v→w} = A_flat + ε · A_noise"""
        return self.A_flat[(v, w)] + self.epsilon * self.A_noise[(v, w)]

    def compute_transport(self, sequence):
        """Parallel transport along a sequence: T = Π_{i} exp(A_{t_i → t_{i+1}})"""
        T = np.eye(self.K)
        for i in range(len(sequence) - 1):
            A = self.get_connection(sequence[i], sequence[i + 1])
            T = T @ scipy.linalg.expm(A)
        return T

    def generate_sample(self, rng):
        """Generate (sequence, label) pair."""
        seq = rng.randint(0, self.vocab_size, size=self.seq_len)
        T = self.compute_transport(seq)
        # Quantize transport norm into classes
        norm = np.linalg.norm(T, 'fro')
        label = min(int(norm * self.n_classes / (self.K * 2)), self.n_classes - 1)
        return seq, label
```

**Experimental protocol for HF5.3:**
1. Generate datasets at ε ∈ {0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0}
2. Train flat (standard) and non-flat Gauge-Transformer at each ε
3. Plot accuracy vs. ε for both architectures
4. Expected: crossover at ε* where non-flat surpasses flat

### 2.2 COGS/SCAN Data Loaders (HF1.1, HF2.2)

**New file:** `transformer/data/compositionality.py`

COGS and SCAN are sequence-to-sequence tasks. Frame as autoregressive LM:

```
Input:  "The cat sat on the mat [SEP] ∃x.cat(x)∧sit(x)∧on(x,mat)"
Target: shifted right for next-token prediction
```

```python
class COGSDataset(torch.utils.data.Dataset):
    """COGS compositional generalization benchmark.

    Kim & Linzen (2020): https://github.com/najoungkim/COGS

    Formats source→target pairs as single sequences for autoregressive LM training.
    Evaluates exact-match accuracy on the generalization split.
    """

    def __init__(self, split='train', tokenizer=None, max_len=256):
        # Download or load from cache
        self.data = load_cogs_split(split)  # list of (source, target) strings
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, idx):
        source, target = self.data[idx]
        # Concatenate with separator: "source [SEP] target [EOS]"
        combined = f"{source} [SEP] {target} [EOS]"
        token_ids = self.tokenizer.encode(combined)[:self.max_len]
        return {
            'input_ids': torch.tensor(token_ids[:-1]),
            'target_ids': torch.tensor(token_ids[1:]),
            'sep_position': token_ids.index(self.tokenizer.encode('[SEP]')[0]),
        }

    def evaluate_exact_match(self, model, tokenizer):
        """Generate target from source and compute exact-match accuracy."""
        correct = 0
        for source, target in self.data:
            prompt = f"{source} [SEP]"
            generated = generate_greedy(model, tokenizer, prompt, max_new_tokens=128)
            if generated.strip() == target.strip():
                correct += 1
        return correct / len(self.data)


class SCANDataset(torch.utils.data.Dataset):
    """SCAN compositional generalization benchmark.

    Lake & Baroni (2018): https://github.com/brendenlake/SCAN

    Same approach as COGS: frame as autoregressive seq2seq.
    """
    # ... same pattern as COGSDataset ...
```

### 2.3 Pragmatic Task Data Loaders (HF1.2, HF5.1)

**New file:** `transformer/data/pragmatic_tasks.py`

```python
class SarcasmDataset(torch.utils.data.Dataset):
    """SemEval-2018 Task 3: Irony Detection in English Tweets.

    Frame as binary classification via next-token prediction:
        "[text] [SEP] ironic" or "[text] [SEP] literal"
    """

class WiCDataset(torch.utils.data.Dataset):
    """Word-in-Context: binary word sense disambiguation.

    Input: two sentences with a shared target word.
    Label: same sense (True) or different sense (False).

    Frame as: "[sent1] [SEP] [sent2] [SEP] [target_word] [SEP] same/different"
    """

class GardenPathDataset(torch.utils.data.Dataset):
    """Garden-path sentences with matched unambiguous controls.

    For HF4.3: testing holonomy × ambiguity × context length interaction.

    Sources:
        - Bever (1970) classic garden-path sentences
        - Tabor et al. (2004) graded garden-path effects
        - Custom matched controls
    """
```

---

## Part 3: Experiment Configs and Training Scripts

### 3.1 Training Configs

**New file:** `transformer/training/flat_bundle_configs.py`

```python
def get_cocycle_relaxation_configs(alpha_values=None):
    """Generate configs for HF4.1: cocycle relaxation ablation.

    Sweeps cocycle_relaxation from 0 (flat) to 1 (fully non-flat).
    """
    if alpha_values is None:
        alpha_values = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]

    configs = {}
    for alpha in alpha_values:
        configs[f'cocycle_alpha_{alpha}'] = {
            **get_base_config(),
            'non_flat_transport': alpha > 0,
            'cocycle_relaxation': alpha,
            'connection_type': 'bilinear',
        }
    return configs


def get_per_head_gating_config():
    """Config for HF4.2: per-head flatness specialization."""
    return {
        **get_base_config(),
        'non_flat_transport': True,
        'cocycle_relaxation': 1.0,  # Let the gates control per-head
        'per_head_flatness_gate': True,
        'connection_type': 'bilinear',
    }


def get_holonomy_penalty_configs(lambda_values=None):
    """Configs for HF2.3: holonomy penalty scaling."""
    if lambda_values is None:
        lambda_values = [0.0, 0.01, 0.1, 1.0, 10.0]

    configs = {}
    for lam in lambda_values:
        configs[f'holonomy_penalty_{lam}'] = {
            **get_base_config(),
            'non_flat_transport': True,
            'cocycle_relaxation': 1.0,
            'holonomy_penalty': lam,
        }
    return configs


def get_synthetic_language_configs(epsilon_values=None):
    """Configs for HF5.3: synthetic languages with controlled holonomy."""
    if epsilon_values is None:
        epsilon_values = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0]

    configs = {}
    for eps in epsilon_values:
        for model_type in ['flat', 'non_flat']:
            configs[f'synthetic_eps{eps}_{model_type}'] = {
                **get_small_config(),   # Smaller model for synthetic task
                'non_flat_transport': model_type == 'non_flat',
                'cocycle_relaxation': 1.0 if model_type == 'non_flat' else 0.0,
                'dataset': 'synthetic_gauge',
                'synthetic_epsilon': eps,
            }
    return configs
```

### 3.2 Experiment Runner

**New file:** `scripts/run_flat_bundle_experiments.py`

```python
"""Run flat bundle hypothesis experiments.

Usage:
    python scripts/run_flat_bundle_experiments.py --experiment cocycle_relaxation --seeds 3
    python scripts/run_flat_bundle_experiments.py --experiment per_head_gating --dataset cogs
    python scripts/run_flat_bundle_experiments.py --experiment synthetic --epsilon 0.0 0.1 0.5 1.0
    python scripts/run_flat_bundle_experiments.py --experiment holonomy_penalty --dataset cogs
"""

EXPERIMENTS = {
    'cocycle_relaxation': {  # HF4.1
        'configs': get_cocycle_relaxation_configs,
        'datasets': ['cogs', 'sarcasm', 'wikitext-2'],
        'metrics': ['accuracy', 'perplexity', 'holonomy/mean'],
    },
    'per_head_gating': {     # HF4.2
        'configs': get_per_head_gating_config,
        'datasets': ['wikitext-103'],
        'metrics': ['perplexity', 'holonomy/mean', 'gate_values'],
        'analysis': ['head_specialization_probe'],  # Post-hoc structural probe
    },
    'holonomy_penalty': {    # HF2.3
        'configs': get_holonomy_penalty_configs,
        'datasets': ['cogs'],
        'metrics': ['accuracy', 'holonomy/mean'],
    },
    'synthetic': {           # HF5.3
        'configs': get_synthetic_language_configs,
        'datasets': ['synthetic_gauge'],
        'metrics': ['accuracy', 'holonomy/mean'],
    },
    'compositionality': {    # HF1.1
        'configs': [get_flat_config, get_standard_transformer_config],
        'datasets': ['cogs', 'scan'],
        'metrics': ['exact_match_accuracy'],
    },
    'pragmatic': {           # HF1.2
        'configs': [get_non_flat_config],
        'datasets': ['sarcasm', 'wic'],
        'metrics': ['accuracy', 'holonomy/mean', 'holonomy_by_label'],
    },
    'flatness_emerges': {    # HF5.4
        'configs': [get_non_flat_config],
        'datasets': ['wikitext-103'],
        'metrics': ['holonomy/mean_per_checkpoint', 'cogs_accuracy_per_checkpoint'],
        'checkpoints': [1000, 5000, 10000, 50000, 100000],
    },
}
```

### 3.3 Analysis Scripts

**New file:** `transformer/analysis/flat_bundle_analysis.py`

```python
def analyze_per_head_gating(model):
    """Extract and interpret per-head flatness gates.

    Returns: dict mapping head_index → {gate_value, classification}
    Classification: gate < 0.3 → 'flat', gate > 0.7 → 'non-flat', else → 'mixed'
    """
    results = []
    for layer_idx, block in enumerate(model.transformer.blocks):
        attn = block.attention
        if hasattr(attn, 'flatness_gate_logits'):
            gates = torch.sigmoid(attn.flatness_gate_logits).detach().cpu()
            for h, g in enumerate(gates):
                classification = 'flat' if g < 0.3 else ('non-flat' if g > 0.7 else 'mixed')
                results.append({
                    'layer': layer_idx, 'head': h,
                    'gate_value': g.item(),
                    'classification': classification,
                })
    return results


def holonomy_by_token_relation(model, dataset, ud_annotations=None):
    """Compute average holonomy grouped by syntactic relation type.

    For HF3.2: test whether compositional relations (nsubj, obj, amod)
    have lower holonomy than discourse relations (parataxis, discourse).
    """
    relation_holonomy = defaultdict(list)

    for batch in dataset:
        token_ids = batch['input_ids']
        attention_info = model.forward_with_attention(token_ids)
        exp_delta = attention_info.get('exp_delta')
        if exp_delta is None:
            continue

        # For each annotated dependency triple (head, dep, relation):
        for head_idx, dep_idx, rel_type in ud_annotations[batch['sentence_id']]:
            # Pick a third token to form a triple
            for k in range(len(token_ids)):
                if k != head_idx and k != dep_idx:
                    _, norms, _ = compute_holonomy(
                        exp_delta, None,
                        triples=[(head_idx, dep_idx, k)]
                    )
                    relation_holonomy[rel_type].append(norms.mean().item())

    # Summarize
    return {rel: {'mean': np.mean(norms), 'std': np.std(norms), 'n': len(norms)}
            for rel, norms in relation_holonomy.items()}


def holonomy_compositionality_correlation(model, dataset_with_ratings):
    """Compute correlation between holonomy and compositionality ratings.

    For HF1.3: test whether holonomy anti-correlates with compositionality.
    """
    holonomy_values = []
    compositionality_ratings = []

    for sample in dataset_with_ratings:
        token_ids = sample['input_ids']
        rating = sample['compositionality_rating']

        attention_info = model.forward_with_attention(token_ids.unsqueeze(0))
        exp_delta = attention_info.get('exp_delta')
        if exp_delta is None:
            continue

        stats = holonomy_statistics(exp_delta, sample_size=500)
        holonomy_values.append(stats['mean_holonomy'])
        compositionality_ratings.append(rating)

    rho, p_value = scipy.stats.spearmanr(holonomy_values, compositionality_ratings)
    return {'spearman_rho': rho, 'p_value': p_value, 'n': len(holonomy_values)}
```

---

## Part 4: Implementation Sequence

### Phase 1: Core Infrastructure (1–2 weeks)

**Goal:** Non-flat transport works end-to-end, holonomy is measurable.

| Step | File(s) | What | Test |
|------|---------|------|------|
| 1.1 | `block_config.py` | Add 6 new config fields | Unit test: config creation with new fields |
| 1.2 | `connection.py` (NEW) | `GaugeConnection` module (bilinear + MLP) | Unit test: output shape, zero-init → zero output |
| 1.3 | `attention.py` | Add `connection_delta` and `cocycle_relaxation` args to `compute_transport_operators` | Unit test: `delta=None` reproduces exact flat output; `delta=0` also reproduces flat |
| 1.4 | `attention.py` | Integrate `GaugeConnection` into `IrrepMultiHeadAttention` | Integration test: forward pass with non_flat_transport=True |
| 1.5 | `holonomy.py` (NEW) | `compute_holonomy()`, `holonomy_statistics()`, `holonomy_penalty_loss()` | Unit test: flat transport → ‖H−I‖ < ε; random transport → ‖H−I‖ >> 0 |
| 1.6 | `attention.py` | Add per-head flatness gating | Unit test: gate=0 → flat behavior; gate=1 → non-flat |
| 1.7 | `model.py` | Return `exp_delta` from `forward_with_attention()` | Integration test: exp_delta populated when non_flat=True |
| 1.8 | `metrics.py` | Add holonomy penalty to loss, log holonomy stats | Integration test: loss includes holonomy term; metrics logged |

**Validation checkpoint:** Train a small non-flat model on WikiText-2 for 1000 steps. Verify:
- [x] Loss decreases (model trains)
- [x] Holonomy stats are logged
- [x] With cocycle_relaxation=0, exact same loss curve as flat model
- [x] With cocycle_relaxation=1, holonomy is non-zero

### Phase 2: Synthetic Language (1 week)

**Goal:** The cleanest falsification test (HF5.3) is runnable.

| Step | File(s) | What |
|------|---------|------|
| 2.1 | `synthetic_gauge.py` (NEW) | `SyntheticGaugeLanguage` class |
| 2.2 | `synthetic_gauge.py` | PyTorch Dataset/DataLoader integration |
| 2.3 | `flat_bundle_configs.py` (NEW) | Configs for synthetic experiments |
| 2.4 | `run_flat_bundle_experiments.py` (NEW) | Runner for synthetic experiment |
| 2.5 | Run | Sweep ε ∈ {0, 0.01, 0.05, 0.1, 0.5, 1.0} × {flat, non-flat} |

**Expected result:** Clear crossover graph showing flat wins at ε≈0, non-flat wins at ε>>0.

### Phase 3: Compositionality Benchmarks (1 week)

**Goal:** Evaluate on COGS and SCAN (HF1.1, HF2.2).

| Step | File(s) | What |
|------|---------|------|
| 3.1 | `compositionality.py` (NEW) | COGS/SCAN data loaders |
| 3.2 | `compositionality.py` | Seq2seq framing for autoregressive LM |
| 3.3 | `compositionality.py` | Exact-match evaluation function |
| 3.4 | `flat_bundle_configs.py` | Configs for COGS/SCAN experiments |
| 3.5 | Run | Flat vs. standard transformer on COGS/SCAN |
| 3.6 | Run | Data efficiency curves (HF2.2): {1%, 5%, 10%, 25%, 50%, 100%} |

### Phase 4: Ablation Sweeps (1 week)

**Goal:** Run the key ablations (HF4.1, HF4.2, HF2.3).

| Step | What |
|------|------|
| 4.1 | **Cocycle relaxation (HF4.1):** α ∈ {0, 0.1, 0.25, 0.5, 0.75, 1.0} on COGS + sarcasm + WikiText-2 |
| 4.2 | **Per-head gating (HF4.2):** Train gated model on WikiText-103, analyze learned gates |
| 4.3 | **Holonomy penalty (HF2.3):** λ_H ∈ {0, 0.01, 0.1, 1.0, 10.0} on COGS |
| 4.4 | Structural probes on gated heads (syntactic vs. discourse sensitivity) |

### Phase 5: Pragmatic Tasks and Analysis (1–2 weeks)

**Goal:** Test the headline prediction — holonomy appears where meaning is context-dependent (HF1.2).

| Step | What |
|------|------|
| 5.1 | Sarcasm/WiC data loaders |
| 5.2 | Train non-flat model on pragmatic tasks |
| 5.3 | Measure holonomy per token triple, correlate with context-dependence labels |
| 5.4 | Holonomy as unsupervised pragmatic detector (HF5.1) |
| 5.5 | Garden-path experiments: holonomy × ambiguity × context length (HF4.3) |

### Phase 6: Cross-Linguistic and Developmental (2+ weeks)

**Goal:** Longer-term extensions (HF3.1, HF3.2, HF5.4).

| Step | What |
|------|------|
| 6.1 | Train non-flat model on UD treebanks for 10+ languages |
| 6.2 | Compute holonomy grouped by dependency relation type |
| 6.3 | Developmental trajectory: track holonomy during training from random init |
| 6.4 | Transformer failure mode prediction (HF5.2) |

---

## Part 5: File Manifest

```
transformer/
  core/
    attention.py              ← MODIFY: add connection_delta, cocycle_relaxation args
    block_config.py           ← MODIFY: add 6 non-flat config fields
    connection.py             ← NEW: GaugeConnection module
    model.py                  ← MODIFY: return exp_delta, support non-flat forward
  analysis/
    holonomy.py               ← NEW: compute_holonomy, holonomy_statistics, penalty
    flat_bundle_analysis.py   ← NEW: per-head gating analysis, relation-grouped holonomy
  data/
    synthetic_gauge.py        ← NEW: synthetic language with controlled holonomy
    compositionality.py       ← NEW: COGS, SCAN data loaders
    pragmatic_tasks.py        ← NEW: sarcasm, WiC data loaders
  training/
    config.py                 ← MODIFY: add non-flat training config fields
    metrics.py                ← MODIFY: holonomy penalty loss term, holonomy logging
    flat_bundle_configs.py    ← NEW: experiment-specific training configs
tests/
  transformer/
    test_non_flat_transport.py  ← NEW: tests for non-flat transport, holonomy
    test_synthetic_language.py  ← NEW: tests for synthetic data generation
scripts/
  run_flat_bundle_experiments.py  ← NEW: experiment runner
```

**Files modified:** 5 (attention.py, block_config.py, model.py, config.py, metrics.py)
**Files created:** 8 (connection.py, holonomy.py, flat_bundle_analysis.py, synthetic_gauge.py, compositionality.py, pragmatic_tasks.py, flat_bundle_configs.py, run_flat_bundle_experiments.py) + 2 test files

---

## Part 6: Critical Design Decisions

### Decision 1: Multiplicative vs. Additive Connection

**Chosen: Multiplicative** (`Ω_ij = exp(φ_i) · exp(δ_ij·G) · exp(-φ_j)`)

Reasons:
- `δ_ij = 0` *exactly* recovers the flat case (not approximately)
- Stays on the Lie group (all factors are group elements)
- Holonomy has a clean form: conjugation of connection holonomy by gauge frame
- Gradients through `matrix_exp` are well-conditioned (PyTorch autograd)

Rejected alternatives:
- **Additive Ω + ΔΩ**: breaks group structure, doesn't reduce to identity
- **Lie algebra sum `exp((φ_i − φ_j + δ_ij)·G)`**: doesn't exactly recover flat case for non-abelian groups (BCH corrections)
- **Independent Ω_ij per edge**: too many parameters, hard to train, loses gauge frame structure

### Decision 2: Bilinear vs. MLP Connection

**Default: Bilinear** (`δ_ij^a = μ_i^T W^a μ_j`)

Reasons:
- Parameter-efficient: `n_gen × d_head²` per head
- Captures pairwise interactions (the natural structure for a connection)
- Can be antisymmetrized: `W → (W − W^T)/2` gives `δ_ij = −δ_ji` (natural for connections on undirected graphs)
- Faster than MLP (no activation function, just a batched bilinear form)

MLP as fallback for experiments where bilinear is insufficient.

### Decision 3: Where Scalar Heads Fit

K=1 (scalar) heads have trivially commutative transport. For these heads:
- `exp(δ) = e^δ` (scalar exponential, trivially fast)
- All holonomy vanishes identically: `H_ijk = e^{δ_ij + δ_jk + δ_ki}`, which equals 1 iff `δ_ij + δ_jk + δ_ki = 0`
- For bilinear connection: `δ_ij = w · μ_i · μ_j` (scalar), and `δ_ij + δ_jk + δ_ki = w(μ_iμ_j + μ_jμ_k + μ_kμ_i)` which is generically ≠ 0

So even scalar heads can have non-trivial holonomy in the abelian (U(1)) sense! But the holonomy is commutative — it's a phase, not a matrix rotation.

**Decision:** Allow non-flat transport for all head dimensions, including K=1. The abelian vs. non-abelian distinction is itself interesting.

### Decision 4: Memory Strategy for Non-Flat Transport

The optimized block-diagonal and diagonal KL paths avoid materializing the full Ω matrix. Non-flat transport breaks this factorization.

**Strategy:**
1. When `non_flat_transport=True`, fall through to the full-Omega PyTorch path
2. Use gradient checkpointing to manage memory
3. For large N, use the chunked path with non-flat transport within chunks

This trades some memory efficiency for correctness. The synthetic language and COGS/SCAN experiments use short sequences (N ≤ 128), so memory is not a bottleneck.

### Decision 5: Initialization — Near-Flat Start

All connection parameters initialized to zero → δ_ij = 0 at init → model starts in the flat regime. This is critical because:
1. The flat regime is known to work (existing publication results)
2. The model must *learn* to deviate from flatness only where the data warrants it
3. Gradient-based learning from a known-good initialization is more stable than random init in a high-dimensional space

The per-head flatness gate is initialized at logit = −2 → sigmoid ≈ 0.12 (near-flat but not exactly flat, allowing gradient flow).

---

## Part 7: Expected Outcomes and Narrative

### If the conjecture is correct (flatness matches compositionality):

1. **HF5.3 (synthetic):** Clear crossover graph at ε* > 0
2. **HF4.1 (cocycle relaxation):** COGS peaks at α ≈ 0; sarcasm peaks at α > 0
3. **HF4.2 (per-head gating):** Bimodal gate distribution: some heads flat, some non-flat
4. **HF1.2 (pragmatic holonomy):** Holonomy correlates with context-dependence labels
5. **HF5.4 (developmental):** Holonomy decreases during training as model discovers flatness

### If the conjecture is wrong:

1. Non-flat model outperforms flat even on COGS/SCAN → compositionality needs path-dependence
2. Holonomy is uniformly zero even on pragmatic tasks → pragmatics doesn't need non-flat transport
3. Per-head gates all converge to the same value → no compositional/pragmatic specialization
4. Synthetic language shows no crossover → flatness advantage is not about path-independence

Either outcome is publishable. The framework provides clean falsification.
