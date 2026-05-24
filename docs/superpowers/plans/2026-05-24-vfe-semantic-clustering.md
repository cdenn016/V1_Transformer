# VFE Semantic-Clustering Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone `transformer/vfe/semantic_clustering/` package that clusters per-token belief geometry (mu, Sigma, phi, and Omega=exp(phi·G)) with geometry-faithful distances and emits separate publication-quality images plus a metrics sidecar, for both contextual and vocab-level views.

**Architecture:** A small package, one file per concern (bundle → extract → geometry → projection → clustering → metrics → plotting), driven by a click-to-run orchestrator. Distances are manifold-aware and fed as precomputed dissimilarity matrices into UMAP (primary) / sklearn fallback. No coupling to legacy `transformer/analysis/semantics.py`; no neural-network components; no CLI args.

**Tech Stack:** PyTorch (belief extraction), NumPy/SciPy (linear algebra: `expm`, `logm`, `slogdet`), scikit-learn (t-SNE/MDS/agglomerative/HDBSCAN/silhouette), umap-learn (primary projector, to be installed), matplotlib + `transformer/visualization/pub_style.py` (figures). Reference design: `docs/superpowers/specs/2026-05-24-vfe-semantic-clustering-design.md`.

---

## Conventions for every task
- Tests live under `tests/transformer/vfe/`. Run with `python -m pytest <path> -v` from repo root.
- Type hints on every signature; LaTeX-bearing docstrings on the math functions.
- No CLI args. No `nn.*` layers. Match `pub_style` house style for figures.
- Commit after each task with the message shown.

---

## Task 0: Environment + package skeleton + data contract

**Files:**
- Create: `transformer/vfe/semantic_clustering/__init__.py`
- Create: `transformer/vfe/semantic_clustering/bundle.py`
- Test: `tests/transformer/vfe/test_semantic_clustering_bundle.py`

- [ ] **Step 1: Install umap-learn**

Run: `python -m pip install "umap-learn>=0.5.0"`
Expected: success; `python -c "import umap; print(umap.__version__)"` prints a version.
(If install fails, the module must still import — projection.py try-guards the import. Record the failure in the audit md and proceed; fallback is t-SNE/MDS.)

- [ ] **Step 2: Write the failing test for BeliefBundle**

```python
# tests/transformer/vfe/test_semantic_clustering_bundle.py
import torch
from transformer.vfe.semantic_clustering.bundle import BeliefBundle

def test_bundle_holds_fields_and_n():
    n, K, n_gen = 5, 8, 4
    b = BeliefBundle(
        mu=torch.zeros(n, K), sigma=torch.ones(n, K), phi=torch.zeros(n, n_gen),
        token_ids=torch.arange(n), token_strings=None, generators=None,
        irrep_dims=[K], source="vocab", layer="final", diagonal=True,
    )
    assert b.n == n
    assert b.K == K
    assert b.diagonal is True
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/transformer/vfe/test_semantic_clustering_bundle.py -v`
Expected: FAIL (module not found).

- [ ] **Step 4: Implement bundle.py**

```python
# transformer/vfe/semantic_clustering/bundle.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Union
import torch


@dataclass
class BeliefBundle:
    r"""Per-token belief geometry extracted from a VFE model.

    Fields
    ------
    mu : (n, K) token means.
    sigma : (n, K) diagonal variances, or (n, K, K) full covariances.
    phi : (n, n_gen) Lie-algebra coefficients.
    token_ids : (n,) integer ids.
    token_strings : optional decoded strings, len n.
    generators : optional (n_gen, K, K) generator bank G; the algebra element is sum_c phi_c G_c.
    irrep_dims : per-head irrep dims (block structure of Omega), e.g. [10]*20.
    source : 'contextual' | 'vocab'.
    layer : 'final' | int.
    diagonal : True if sigma is (n, K).
    """
    mu: torch.Tensor
    sigma: torch.Tensor
    phi: torch.Tensor
    token_ids: torch.Tensor
    token_strings: Optional[list[str]]
    generators: Optional[torch.Tensor]
    irrep_dims: list[int]
    source: str
    layer: Union[str, int]
    diagonal: bool

    @property
    def n(self) -> int:
        return int(self.mu.shape[0])

    @property
    def K(self) -> int:
        return int(self.mu.shape[1])
```

```python
# transformer/vfe/semantic_clustering/__init__.py
from transformer.vfe.semantic_clustering.bundle import BeliefBundle

__all__ = ["BeliefBundle"]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/transformer/vfe/test_semantic_clustering_bundle.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add transformer/vfe/semantic_clustering/ tests/transformer/vfe/test_semantic_clustering_bundle.py
git commit -m "feat(vfe): semantic-clustering package skeleton + BeliefBundle contract"
```

---

## Task 1: geometry.py — geometry-faithful distance matrices

**Files:**
- Create: `transformer/vfe/semantic_clustering/geometry.py`
- Test: `tests/transformer/vfe/test_semantic_clustering_geometry.py`

Math reference (all return a symmetric `(n,n)` numpy array, zero diagonal, non-negative):
- **mu euclidean:** `d_ij = ||mu_i - mu_j||_2`.
- **mu mahalanobis:** `d_ij = sqrt((mu_i-mu_j)^T Sbar^{-1} (mu_i-mu_j))`, `Sbar = mean_k Sigma_k` (diagonal → elementwise mean; full → mean matrix). Use diagonal whitening when `diagonal`.
- **sigma bhattacharyya** between `N(mu_i,S_i), N(mu_j,S_j)`: with `S = (S_i+S_j)/2`,
  `D_B = (1/8)(mu_i-mu_j)^T S^{-1}(mu_i-mu_j) + (1/2) ln( det S / sqrt(det S_i det S_j) )`.
  Diagonal: fully vectorized with elementwise ops + `log`. Full: `slogdet` + solve. Dissimilarity = `D_B` (>=0, symmetric, zero diagonal).
- **sigma logeuclidean:** diagonal → `||log sigma_i - log sigma_j||_2`; full → `||logm(S_i) - logm(S_j)||_F` (precompute `logm(S_k)` once each).
- **phi vector:** PCA-whiten phi to min(n-1, 50) comps (guard n<=2), then Euclidean.
- **omega geodesic:** build per-head `Omega_{h,k} = expm( sum_c phi[k, head_slice_h, c] * G_{h,c} )`. Practically: form algebra matrix `A_{h,k}` of shape (d_h,d_h) from the generator bank restricted to head h, `Omega = expm(A)`. Geodesic `d_ij^2 = sum_h || logm(Omega_{h,i}^{-1} Omega_{h,j}) ||_F^2`; `d_ij = sqrt(...)`. Use `scipy.linalg.expm`/`logm`; wrap `logm` to take `.real` and fall back to Frobenius `||Omega_i - Omega_j||_F` on non-finite. Generators MUST come from the model (passed in the bundle); tests use a supplied synthetic bank.

- [ ] **Step 1: Write failing tests**

```python
# tests/transformer/vfe/test_semantic_clustering_geometry.py
import numpy as np
import torch
import pytest
from transformer.vfe.semantic_clustering import geometry as geo


def _props(D, n):
    assert D.shape == (n, n)
    assert np.allclose(D, D.T, atol=1e-6)          # symmetric
    assert np.allclose(np.diag(D), 0.0, atol=1e-6) # zero diagonal
    assert (D >= -1e-9).all()                       # non-negative


def test_mu_euclidean_props_and_value():
    mu = torch.tensor([[0.0, 0.0], [3.0, 4.0], [0.0, 1.0]])
    D = geo.mu_distances(mu, metric="euclidean")
    _props(D, 3)
    assert np.isclose(D[0, 1], 5.0)


def test_mu_mahalanobis_identity_cov_equals_euclidean():
    mu = torch.randn(6, 4)
    sigma = torch.ones(6, 4)  # diagonal, unit
    De = geo.mu_distances(mu, metric="euclidean")
    Dm = geo.mu_distances(mu, sigma=sigma, metric="mahalanobis", diagonal=True)
    assert np.allclose(De, Dm, atol=1e-5)


def test_sigma_bhattacharyya_zero_for_identical():
    mu = torch.randn(5, 3)
    sigma = torch.rand(5, 3) + 0.5
    D = geo.sigma_distances(sigma, mu=mu, metric="bhattacharyya", diagonal=True)
    _props(D, 5)


def test_sigma_logeuclidean_diag_matches_full_when_full_is_diagonal():
    n, K = 4, 3
    diag = torch.rand(n, K) + 0.5
    full = torch.stack([torch.diag(diag[i]) for i in range(n)])
    Dd = geo.sigma_distances(diag, metric="logeuclidean", diagonal=True)
    Df = geo.sigma_distances(full, metric="logeuclidean", diagonal=False)
    assert np.allclose(Dd, Df, atol=1e-4)


def test_phi_vector_zero_distance_identical_rows():
    phi = torch.randn(5, 12)
    phi[2] = phi[0]
    D = geo.phi_vector_distances(phi)
    _props(D, 5)
    assert np.isclose(D[0, 2], 0.0, atol=1e-5)


def test_omega_geodesic_equals_quadrature_of_per_head():
    # 2 heads of dim 2, full gl(2) generators per head (4 each) -> n_gen=8
    torch.manual_seed(0)
    d = 2
    eye = np.eye(d)
    basis = [np.zeros((d, d)) for _ in range(d * d)]
    for idx in range(d * d):
        basis[idx].flat[idx] = 1.0
    # block-diagonal generator bank (8, 4, 4)
    K = 2 * d
    G = np.zeros((8, K, K))
    for h in range(2):
        for c in range(4):
            G[h * 4 + c, h * d:(h + 1) * d, h * d:(h + 1) * d] = basis[c]
    G = torch.tensor(G, dtype=torch.float64)
    phi = torch.randn(3, 8, dtype=torch.float64) * 0.1
    D = geo.omega_geodesic_distances(phi, generators=G, irrep_dims=[d, d])
    _props(D, 3)
    # identical phi -> zero
    phi2 = phi.clone(); phi2[1] = phi2[0]
    D2 = geo.omega_geodesic_distances(phi2, generators=G, irrep_dims=[d, d])
    assert np.isclose(D2[0, 1], 0.0, atol=1e-6)
    # the total distance IS the quadrature of independent per-head geodesics
    from scipy.linalg import expm, logm
    def head_geo(k, l, h):
        A_k = sum(phi[k, h*4 + c].item() * G[h*4 + c, h*d:(h+1)*d, h*d:(h+1)*d].numpy() for c in range(4))
        A_l = sum(phi[l, h*4 + c].item() * G[h*4 + c, h*d:(h+1)*d, h*d:(h+1)*d].numpy() for c in range(4))
        Ok, Ol = expm(A_k), expm(A_l)
        return np.linalg.norm(logm(np.linalg.inv(Ok) @ Ol).real, "fro")
    expected_01 = np.sqrt(head_geo(0, 1, 0)**2 + head_geo(0, 1, 1)**2)
    assert np.isclose(D[0, 1], expected_01, atol=1e-5)
```

- [ ] **Step 2: Run to verify fail**

Run: `python -m pytest tests/transformer/vfe/test_semantic_clustering_geometry.py -v`
Expected: FAIL (module / functions not found).

- [ ] **Step 3: Implement geometry.py**

Implement `mu_distances`, `sigma_distances`, `phi_vector_distances`, `omega_geodesic_distances` per the math reference above. Required signatures:

```python
def mu_distances(mu, sigma=None, metric="euclidean", diagonal=True) -> np.ndarray: ...
def sigma_distances(sigma, mu=None, metric="bhattacharyya", diagonal=True) -> np.ndarray: ...
def phi_vector_distances(phi, whiten=True, max_comps=50) -> np.ndarray: ...
def omega_geodesic_distances(phi, generators, irrep_dims) -> np.ndarray: ...
```

Use `scipy.spatial.distance.squareform/pdist` where convenient for Euclidean paths; `scipy.linalg.expm`, `scipy.linalg.logm` for the Omega path (cast to float64, take `.real`, Frobenius fallback on non-finite). For bhattacharyya diagonal path, vectorize over pairs with broadcasting on `(n,1,K)` vs `(1,n,K)`.

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/transformer/vfe/test_semantic_clustering_geometry.py -v`
Expected: PASS (all 6).

- [ ] **Step 5: Commit**

```bash
git add transformer/vfe/semantic_clustering/geometry.py tests/transformer/vfe/test_semantic_clustering_geometry.py
git commit -m "feat(vfe): geometry-faithful distance matrices for mu/Sigma/phi/Omega"
```

---

## Task 2: extract.py — contextual + vocab-level extraction

**Files:**
- Create: `transformer/vfe/semantic_clustering/extract.py`
- Test: `tests/transformer/vfe/test_semantic_clustering_extract.py`

Behavior:
- `extract_contextual(model, token_ids, layer="final") -> BeliefBundle`. NEVER pass targets (Law 1). `layer="final"` → `logits, beliefs = model.forward_with_beliefs(token_ids)`; flatten `(B,N,*)` to `(B*N,*)`. `layer=int` → set every `block.e_step._capture_attention_state = True`, run `model.forward(token_ids)` (no targets), then read `model.stack.blocks[layer].e_step._last_attention_state` keys `mu_q`, `sigma_q`, `phi`. Warn (via `warnings.warn`) if `model.cfg.gauge_parameterization == "omega_direct"` that the per-layer cache is empty.
- `extract_vocab(model, token_ids=None) -> BeliefBundle`. If `token_ids` None, use `torch.arange(min(vocab_size, max_tokens))`. Read encode bank: `pb = model.prior_bank` (or wherever encode lives — confirm at impl time). mu from `pb.mu_embed(ids)`; sigma from `exp(pb.sigma_log_embed(ids))` if present else broadcast `exp(pb.base_log_sigma)`; phi from `pb.phi_embed(ids)`. Mark `source="vocab"`, `diagonal=model.cfg.diagonal_covariance`.
- Both populate `generators` (the exact model generator bank — locate it: candidates are `block.gauge`/`e_step` generator tensor, or reconstruct from `block_exp_pairs`; confirm by reading the code at impl time) and `irrep_dims` from `model.cfg`.
- Token strings: if a tokenizer/dataset is passed, decode; else None.

**Verification requirement (CLAUDE.md):** at implementation time, open `transformer/vfe/train_vfe.py`, trace the active config keys, and confirm the exact attribute names (`mu_embed`, `sigma_log_embed`/`base_log_sigma`, `phi_embed`, generator bank) against `transformer/vfe/prior_bank.py` and `e_step.py` BEFORE asserting the path works. Do not trust this plan's attribute names blindly.

**Generator-bank provenance (load-bearing — do this read-only pass FIRST):** the Ω geodesic in Task 1 is only correct if `BeliefBundle.generators` is the EXACT generator tensor the model uses to build transport. Before implementing extraction, grep the generator construction across `vfe/prior_bank.py`, `vfe/e_step.py`, `vfe/block.py`, and `vfe/attention.py::compute_gauge_transport`; produce ONE concrete `file:line` citation for where the `(n_gen, K, K)` (or per-head block) generator bank is created/stored, and store that exact tensor in the bundle. If φ→Ω is built via `block_exp_pairs` rather than an explicit bank, store whatever the model exponentiates so geometry can reproduce the model's own Ω. Do NOT invent a basis.

- [ ] **Step 1: Write failing smoke test (uses a tiny real model)**

```python
# tests/transformer/vfe/test_semantic_clustering_extract.py
import torch
from transformer.vfe.config import VFEConfig
from transformer.vfe.model import VFEModel
from transformer.vfe.semantic_clustering.extract import extract_contextual, extract_vocab


def _tiny_model():
    cfg = VFEConfig(
        vocab_size=64, embed_dim=8, irrep_spec=[("fund", 2, 4)],
        diagonal_covariance=True, n_layers=1,
    )
    return VFEModel(cfg), cfg


def test_contextual_final_shapes():
    model, cfg = _tiny_model()
    model.eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 5))
    b = extract_contextual(model, ids, layer="final")
    assert b.mu.shape == (10, cfg.embed_dim)
    assert b.sigma.shape == (10, cfg.embed_dim)   # diagonal active
    assert b.phi.shape[0] == 10
    assert b.source == "contextual"


def test_vocab_shapes():
    model, cfg = _tiny_model()
    model.eval()
    b = extract_vocab(model, token_ids=torch.arange(20))
    assert b.mu.shape == (20, cfg.embed_dim)
    assert b.phi.shape[0] == 20
    assert b.source == "vocab"
```

- [ ] **Step 2: Run to verify fail** — `python -m pytest tests/transformer/vfe/test_semantic_clustering_extract.py -v` → FAIL.

- [ ] **Step 3: Implement extract.py** after confirming the real attribute names (see verification requirement). Detach all tensors, move to CPU.

- [ ] **Step 4: Run to verify pass** → PASS (2).

- [ ] **Step 5: Commit**

```bash
git add transformer/vfe/semantic_clustering/extract.py tests/transformer/vfe/test_semantic_clustering_extract.py
git commit -m "feat(vfe): contextual + vocab-level belief extraction for semantic clustering"
```

---

## Task 3: projection.py — UMAP-primary 2D/3D projection

**Files:**
- Create: `transformer/vfe/semantic_clustering/projection.py`
- Test: `tests/transformer/vfe/test_semantic_clustering_projection.py`

Behavior: `project(matrix, method="umap", n_components=2, precomputed=True, random_state=0) -> np.ndarray (n, n_components)`. If `precomputed`, `matrix` is an `(n,n)` distance matrix. Order of attempts when `method="umap"`: try `import umap` → `umap.UMAP(metric="precomputed", n_components=...)`; on ImportError fall back to `sklearn.manifold.TSNE(metric="precomputed", init="random")`; then `sklearn.manifold.MDS(dissimilarity="precomputed")`. `method="pca"` only valid for feature matrices (`precomputed=False`). Guard tiny n (n<=n_components+1 → return zero-padded coords). Emit a `warnings.warn` naming the backend actually used.

**Non-metric dissimilarity caveat:** the Bhattacharyya Σ distance is NOT a metric (no triangle inequality). UMAP `metric="precomputed"`, `MDS(dissimilarity="precomputed")`, and average-linkage agglomerative clustering all accept dissimilarity matrices and work. `TSNE(metric="precomputed")` can warn/error on negative or non-metric entries in some sklearn versions — clamp the input to `>=0` (it already is) and, in the t-SNE fallback branch, wrap the call in try/except and degrade to MDS if it raises. umap-learn 0.5.12 is installed, so the primary path is UMAP and this caveat only bites the fallback.

- [ ] **Step 1: Write failing test**

```python
# tests/transformer/vfe/test_semantic_clustering_projection.py
import numpy as np
from transformer.vfe.semantic_clustering.projection import project

def test_project_precomputed_returns_2d():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(15, 5))
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    Y = project(D, method="umap", n_components=2, precomputed=True)
    assert Y.shape == (15, 2)
    assert np.isfinite(Y).all()

def test_project_tiny_n_safe():
    D = np.zeros((2, 2))
    Y = project(D, method="umap", n_components=2, precomputed=True)
    assert Y.shape == (2, 2)
```

- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement projection.py per behavior above.**
- [ ] **Step 4: Run → PASS (2).**
- [ ] **Step 5: Commit**

```bash
git add transformer/vfe/semantic_clustering/projection.py tests/transformer/vfe/test_semantic_clustering_projection.py
git commit -m "feat(vfe): UMAP-primary projection with sklearn precomputed fallback"
```

---

## Task 4: clustering.py — unsupervised labels with auto-k

**Files:**
- Create: `transformer/vfe/semantic_clustering/clustering.py`
- Test: `tests/transformer/vfe/test_semantic_clustering_clustering.py`

Behavior: `cluster(matrix, method="agglomerative", precomputed=True, k="auto", k_range=range(2,9), random_state=0) -> np.ndarray (n,) int labels`. `agglomerative` → `AgglomerativeClustering(metric="precomputed", linkage="average", n_clusters=k)`. For `k="auto"`, sweep `k_range`, score each with silhouette on the precomputed distances (`metric="precomputed"`), pick argmax; guard n small (n<4 → single cluster of zeros). Optional `method="hdbscan"` via `sklearn.cluster.HDBSCAN(metric="precomputed")`.

- [ ] **Step 1: Failing test**

```python
# tests/transformer/vfe/test_semantic_clustering_clustering.py
import numpy as np
from transformer.vfe.semantic_clustering.clustering import cluster

def test_recovers_two_obvious_blobs():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 0.05, size=(20, 2))
    b = rng.normal(5, 0.05, size=(20, 2))
    X = np.vstack([a, b])
    D = np.linalg.norm(X[:, None] - X[None], axis=-1)
    labels = cluster(D, method="agglomerative", precomputed=True, k="auto")
    assert labels.shape == (40,)
    # the two blobs end up in different clusters
    assert labels[0] != labels[-1]

def test_tiny_n_single_cluster():
    D = np.zeros((3, 3))
    labels = cluster(D, precomputed=True, k="auto")
    assert labels.shape == (3,)
```

- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement clustering.py.**
- [ ] **Step 4: Run → PASS (2).**
- [ ] **Step 5: Commit**

```bash
git add transformer/vfe/semantic_clustering/clustering.py tests/transformer/vfe/test_semantic_clustering_clustering.py
git commit -m "feat(vfe): unsupervised clustering with silhouette-swept auto-k"
```

---

## Task 5: metrics.py — common + per-quantity metrics

**Files:**
- Create: `transformer/vfe/semantic_clustering/metrics.py`
- Test: `tests/transformer/vfe/test_semantic_clustering_metrics.py`

Behavior (all return plain-keyed dicts, JSON-serializable floats/lists):
- `common_metrics(D, labels, precomputed=True) -> dict`: `silhouette` (`metric="precomputed"`), `calinski_harabasz` and `davies_bouldin` (need features — accept optional `X`; skip with `None` if only D given), `inter_intra_ratio` (mean between-cluster / mean within-cluster distance from D), `n_clusters`. Guard single-cluster → silhouette `None`.
- `sigma_metrics(sigma, diagonal=True) -> dict`: per-token effective rank `exp(H(p))`, `p=lam/sum(lam)` (diag → `lam=sigma`); report mean/std `effective_rank`, mean `logdet`, mean `trace`, mean `anisotropy` (max/min eigenvalue ratio). Effective rank ∈ [1, K].
- `phi_metrics(phi, irrep_dims, omega=None) -> dict`: energy partition — fraction of `||phi||^2` in diagonal generators vs off-diagonal (cross-coupling) generators per head; fractions sum to 1. If `omega` (per-token Omega available) → mean `||Omega - I||_F`, mean `det`. **Before deriving the diagonal/off-diagonal index split from `irrep_dims`, consult `transformer/vfe/cross_coupling_metrics.py::phi_energy_partition` (line ~90) and MATCH its index convention; reuse that function directly if its signature is compatible. It is a vfe-package function (not legacy semantics), so reusing it is standalone-correct and avoids silently disagreeing with the rest of the package.**
- `mu_metrics(mu) -> dict`: mean/std of `||mu||`, mean pairwise distance.

- [ ] **Step 1: Failing tests**

```python
# tests/transformer/vfe/test_semantic_clustering_metrics.py
import numpy as np, torch
from transformer.vfe.semantic_clustering import metrics as M

def test_silhouette_in_range():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0,0.1,(15,2)), rng.normal(4,0.1,(15,2))])
    D = np.linalg.norm(X[:,None]-X[None], axis=-1)
    labels = np.array([0]*15 + [1]*15)
    out = M.common_metrics(D, labels, precomputed=True)
    assert -1.0 <= out["silhouette"] <= 1.0
    assert out["n_clusters"] == 2

def test_effective_rank_bounds():
    sigma = torch.rand(10, 6) + 0.1
    out = M.sigma_metrics(sigma, diagonal=True)
    assert 1.0 <= out["effective_rank_mean"] <= 6.0

def test_phi_energy_partition_sums_to_one():
    phi = torch.randn(8, 8)          # 2 heads x gl(2) = 8 gens
    out = M.phi_metrics(phi, irrep_dims=[2, 2])
    assert np.isclose(out["energy_frac_diagonal"] + out["energy_frac_offdiag"], 1.0, atol=1e-6)
```

- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement metrics.py.** Map which generator indices are "diagonal" per head (for gl(d) with the standard E_{ab} basis, diagonal generators are indices where row==col within the d×d block — derive from `irrep_dims`).
- [ ] **Step 4: Run → PASS (3).**
- [ ] **Step 5: Commit**

```bash
git add transformer/vfe/semantic_clustering/metrics.py tests/transformer/vfe/test_semantic_clustering_metrics.py
git commit -m "feat(vfe): clustering + per-quantity (Sigma/phi/mu) metrics"
```

---

## Task 6: plotting.py — four separate publication figures

**Files:**
- Create: `transformer/vfe/semantic_clustering/plotting.py`
- Test: `tests/transformer/vfe/test_semantic_clustering_plotting.py`

Behavior: try `from transformer.visualization.pub_style import set_pub_style, PUB_COLORS, PUB_CYCLE`; on ImportError use a local minimal style (serif, dpi=300). Force `matplotlib.use("Agg")` lazily inside functions (not at import). Four functions, each saving `<outdir>/<name>.pdf` AND `<outdir>/<name>.png` (dpi=300, bbox_inches="tight"):
- `plot_mu_clustering(coords, labels, metrics, outdir, token_strings=None)`
- `plot_sigma_clustering(coords, labels, metrics, outdir, token_strings=None)`
- `plot_phi_vector_clustering(coords, labels, metrics, outdir, token_strings=None)`
- `plot_omega_clustering(coords, labels, metrics, outdir, token_strings=None)`

Each: 2D scatter colored by cluster label using `PUB_COLORS` cycle, light cluster-centroid markers, a small text box rendering the headline metrics (silhouette, n_clusters, plus the quantity-specific headline: Sigma→effective_rank_mean, phi→energy_frac_offdiag, Omega→mean ||Omega-I||_F). Return the two saved paths. Annotate at most ~30 token strings to avoid clutter.

- [ ] **Step 1: Failing test**

```python
# tests/transformer/vfe/test_semantic_clustering_plotting.py
import numpy as np
from pathlib import Path
from transformer.vfe.semantic_clustering.plotting import plot_mu_clustering

def test_plot_writes_pdf_and_png(tmp_path):
    coords = np.random.default_rng(0).normal(size=(20, 2))
    labels = np.array([0]*10 + [1]*10)
    metrics = {"silhouette": 0.5, "n_clusters": 2}
    pdf, png = plot_mu_clustering(coords, labels, metrics, outdir=tmp_path)
    assert Path(pdf).exists() and Path(png).exists()
```

- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement plotting.py** (write `plot_mu_clustering` fully; the other three share a private `_scatter_figure(...)` helper with a quantity-specific headline string).
- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Commit**

```bash
git add transformer/vfe/semantic_clustering/plotting.py tests/transformer/vfe/test_semantic_clustering_plotting.py
git commit -m "feat(vfe): four separate publication-quality clustering figures"
```

---

## Task 7: run_semantic_clustering.py — click-to-run orchestrator + end-to-end smoke

**Files:**
- Create: `transformer/vfe/run_semantic_clustering.py`
- Create: `transformer/vfe/semantic_clustering/pipeline.py` (importable orchestration so it is testable without the CONFIG side-effects)
- Test: `tests/transformer/vfe/test_semantic_clustering_smoke.py`

`pipeline.py`: `run_clustering(model, *, source, layer, token_ids, dataset, methods, outdir) -> dict` ties the units together for one view: extract → for each quantity build distances (mu, sigma, phi-vector, omega) → project → cluster → metrics → plots. Returns the metrics dict and writes `metrics.json` + `metrics.csv`. Handles `source in {"contextual","vocab"}`.

`run_semantic_clustering.py`: a top-level `CONFIG = {...}` dict (checkpoint_path, text sample params, layer, methods, output_dir, do_contextual, do_vocab) and a `main()` guarded by `if __name__ == "__main__":`. No argparse. Loads model from checkpoint (fresh tiny model fallback if `checkpoint_path is None`), resolves output root to `<checkpoint_dir>/semantic_clustering/` (fallback `./outputs/semantic_clustering/`), runs requested views into `contextual/` and `vocab/` subdirs.

- [ ] **Step 1: Failing end-to-end smoke test**

```python
# tests/transformer/vfe/test_semantic_clustering_smoke.py
import torch
from pathlib import Path
from transformer.vfe.config import VFEConfig
from transformer.vfe.model import VFEModel
from transformer.vfe.semantic_clustering.pipeline import run_clustering

def test_vocab_pipeline_end_to_end(tmp_path):
    cfg = VFEConfig(vocab_size=64, embed_dim=8, irrep_spec=[("fund", 2, 4)],
                    diagonal_covariance=True, n_layers=1)
    model = VFEModel(cfg); model.eval()
    out = run_clustering(model, source="vocab", layer="final",
                         token_ids=torch.arange(40), dataset=None,
                         methods={"projection": "umap", "clustering": "agglomerative"},
                         outdir=tmp_path)
    for name in ["mu_clustering", "sigma_clustering",
                 "phi_vector_clustering", "omega_clustering"]:
        assert (Path(tmp_path) / f"{name}.pdf").exists()
        assert (Path(tmp_path) / f"{name}.png").exists()
    assert (Path(tmp_path) / "metrics.json").exists()
    assert (Path(tmp_path) / "metrics.csv").exists()
    # no NaNs in headline metrics
    import json
    m = json.loads((Path(tmp_path) / "metrics.json").read_text())
    assert all(v == v for v in _flatten_floats(m))  # NaN != NaN

def _flatten_floats(d):
    for v in d.values() if isinstance(d, dict) else []:
        if isinstance(v, dict):
            yield from _flatten_floats(v)
        elif isinstance(v, (int, float)):
            yield float(v)
```

- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement pipeline.py then run_semantic_clustering.py.**
- [ ] **Step 4: Run → PASS.** Then run the full module test suite: `python -m pytest tests/transformer/vfe/test_semantic_clustering_*.py -v` → all PASS.
- [ ] **Step 5: Commit**

```bash
git add transformer/vfe/run_semantic_clustering.py transformer/vfe/semantic_clustering/pipeline.py tests/transformer/vfe/test_semantic_clustering_smoke.py
git commit -m "feat(vfe): semantic-clustering orchestrator + click-to-run entry + e2e smoke"
```

---

## Task 8: Public API exports + post-edit audit log

**Files:**
- Modify: `transformer/vfe/semantic_clustering/__init__.py`
- Create: `docs/audits/audit-2026-05-24-vfe-semantic-clustering.md`

- [ ] **Step 1: Export the public surface** from `__init__.py`: `BeliefBundle`, `extract_contextual`, `extract_vocab`, `run_clustering`, and the four plot functions.
- [ ] **Step 2: Write the post-edit audit md** documenting: every file created, the active-config trace performed in Task 2, the umap-learn install outcome, test counts, and any deviations from this plan.
- [ ] **Step 3: Full suite** `python -m pytest tests/transformer/vfe/test_semantic_clustering_*.py -v` → all PASS; paste the summary line into the audit md.
- [ ] **Step 4: Commit**

```bash
git add transformer/vfe/semantic_clustering/__init__.py docs/audits/audit-2026-05-24-vfe-semantic-clustering.md
git commit -m "docs(vfe): semantic-clustering public API + post-edit audit log"
```

---

## Self-review notes
- Spec coverage: bundle (T0), extraction both views (T2), four distance geometries (T1), projection UMAP+fallback (T3), unsupervised+auto-k (T4), common+per-quantity metrics (T5), four separate images (T6), both-views orchestrator + outputs layout + click-to-run (T7), exports + audit (T8). The "both phi views" requirement maps to `phi_vector_clustering` (T1 `phi_vector_distances`) + `omega_clustering` (T1 `omega_geodesic_distances`), both plotted in T6.
- Type consistency: `BeliefBundle` fields used in T2/T5/T7 match T0; distance functions' signatures in T1 match their callers in T7 pipeline.
- Known runtime check deferred to T2 (confirm encode-bank attribute names against live code) — explicitly flagged, not a placeholder.
- Generators provenance: tests use synthetic banks (T1); real bank sourced from the model in T2 — the one cross-task dependency, called out in T2.
```
