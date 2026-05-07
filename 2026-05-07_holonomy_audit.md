# 2026-05-07 — Code audit: gauge covariance and holonomy diagnostic

**Scope.** Read-only audit of the existing non-flat-transport implementation
against the formalized discrete Regime II structure
$\Omega_{ij} = U_i \exp(\delta_{ij} \cdot G) U_j^{-1}$.

Audited files:
- `transformer/core/connection.py`
- `transformer/core/transport_ops.py:257-402`
- `transformer/analysis/holonomy.py`
- `transformer/analysis/holonomy_metrics.py`
- `transformer/core/block_config.py:307-518`

## A. Gauge-covariance of the connection module — **FAIL**

**Bilinear mode** (`connection.py:138-150`): `delta = einsum('bid,adg,bjg->bija', mu_i, W, mu_j)`.
Under a vertex-local gauge transformation $U_i \to g_i U_i$ acting on the
inputs $\mu_i \to g_i \mu_i$, the output transforms as $\delta'_{ij}^a =
\mu_i^\top g_i^\top W^a g_j \mu_j$, which is **not** a covariant transformation
of $\delta_{ij}^a$. The bilinear weight $W^a$ does not transform; consequently
the predicted $\delta$ is gauge-invariant only under *global* frame
transformations, not under the per-token local gauge action.

**MLP mode** (`connection.py:151-157`): `delta = self.net(concat(mu_i, mu_j))`.
The MLP weights are fixed in the ambient $\mathbb{R}^{2d}$; the connection
sees $(\mu_i,\mu_j)$ in the per-token belief frame and produces $\delta$
without any gauge-equivariance constraint. **Same conclusion**: not
covariant under local gauge.

**Implication for the manuscript.** The data-dependent parameterization
$\delta_{ij} = f_W(\mu_i, \mu_j)$ implements a *fixed-frame* connection: it
treats the embedding ambient space as a global frame in which the connection
lives. The Regime II subsection must state this explicitly. The
"$\delta_{ij}$ is gauge-invariant in the internal frame" formalization in
the plan is consistent with this reading provided we identify the "internal
frame" with the embedding ambient space and not with a per-token-rotated
frame. If full per-token gauge covariance is desired, the connection module
must be modified to consume frame-relative inputs $U_i^{-1} \mu_i$ rather
than the raw $\mu_i$ — out of scope for this audit.

## B. Holonomy-diagnostic algebra — **PASS**

`holonomy.py:59-63` computes
```
C = torch.bmm(exp_delta[:, i, j], exp_delta[:, j, k])
C = torch.bmm(C, exp_delta[:, k, i])
```
This is exactly $C_{ijk} = \exp(\delta_{ij}\cdot G)\exp(\delta_{jk}\cdot G)\exp(\delta_{ki}\cdot G)$
as required by the discrete Regime II formalization.

The docstring at `holonomy.py:22-31` explicitly notes that the full
holonomy $H_{ijk} = U_i \cdot C_{ijk} \cdot U_i^{-1}$ differs from $C_{ijk}$
only by conjugation at the base point $i$, and that $\|C_{ijk} - I\|_F = 0
\iff \|H_{ijk} - I\|_F = 0$. The diagnostic returns $C_{ijk}$ (no
conjugation), which is the correct gauge-invariant content to track.

## C. Causal-mask interaction — **FAIL (with caveat)**

`holonomy_metrics.py:268-278` and `holonomy.py:37-70` sample triples
$(i,j,k)$ uniformly without imposing $i < j < k$ or any causal-DAG ordering.
The default sampler can produce triples that wrap around the sequence
(`(i, i+d, i+2d) mod N`) or that violate strict causal order.

This is **consistent with the "index-space invariant" reading** of the plan:
under autoregressive (causal) attention, the data-flow graph is a DAG and
has no closed loops, so the 3-cycles measured by the diagnostic are
index-space cycles, not data-flow cycles. The diagnostic is therefore a
*formal* invariant of the connection field $\delta_{ij}$ as a tensor on the
complete index graph $K_N$, not a property of the actual attention pattern
in inference.

**Caveat for the manuscript.** This must be stated explicitly: the
manuscript's Regime II subsection should distinguish (i) the index-space
holonomy that is well-defined for *any* learned $\delta$ tensor, regardless
of attention masking, from (ii) the data-flow holonomy that has dynamical
content only in bidirectional/encoder attention.

## D. `cocycle_relaxation` and `non_flat_transport` defaults — **PASS**

`block_config.py:313-318`:
- `non_flat_transport: bool = False`
- `cocycle_relaxation: float = 0.5`
- `connection_type: str = 'bilinear'`
- `connection_hidden_dim: int = 64`
- `connection_init_scale: float = 0.01`
- `holonomy_penalty: float = 0.0`

No `__post_init__` validation is required (the field types and defaults
permit all combinations). The `from_config()` factory at lines 511-516
applies the same defaults, so user-supplied configs that omit these fields
land in Regime I (flat) by default. No hidden overrides in
`transformer/train_publication.py` were found that flip these defaults.

## E. Multi-head holonomy — **PASS for δ, AMBIGUOUS for diagnostic**

**Per-head δ generation.** `PerHeadGaugeConnection` (`connection.py:298-318`)
slices $\mu_i, \mu_j$ by head and runs an independent `GaugeConnection` on
each, producing per-head $\delta_{ij}^{(h)}$ that is then concatenated
block-diagonally into a $(B,N,N,n_\mathrm{gen,total})$ tensor. **PASS**:
each head carries its own connection.

**Per-head holonomy diagnostic.** `holonomy_metrics.compute_holonomy_snapshot`
(`holonomy_metrics.py:134-237`) takes a single $(B,N,N,K,K)$ exp-delta tensor
and runs `compute_holonomy` on the full $K \times K$ representation,
producing **one** holonomy snapshot per call. The `head: int = 0` field on
the snapshot record is a placeholder; no per-head loop is executed by
default.

**AMBIGUOUS.** Per-head holonomy *can* be obtained by slicing $\delta$ by
head and calling the diagnostic separately for each head, but the default
diagnostic aggregates the block-diagonal $K \times K$ matrix and reports a
single $\|C_{ijk} - I\|_F$ that mixes contributions from all heads.
Manuscript should reflect this: the diagnostic is "block-diagonal-aggregate"
in the multi-head setting, with per-head decomposition available but not
default.

## Summary

| Item | Verdict | Manuscript implication |
|---|---|---|
| A. Gauge covariance of connection | **FAIL** | Must qualify $\delta$ as living in a fixed ambient frame, not in per-token gauge |
| B. Holonomy algebra | **PASS** | Manuscript formula matches code |
| C. Causal-mask interaction | **FAIL** with caveat | DAG-vs-cycle distinction must be made explicit |
| D. Config defaults | **PASS** | Default $\delta=0$ Regime I, opt-in to Regime II |
| E. Multi-head: per-head δ | **PASS** | Each head carries own connection |
| E. Multi-head: diagnostic | **AMBIGUOUS** | Aggregate by default; per-head decomposition needs explicit slicing |

No code changes were made. This document is the read-only audit deliverable
referenced in `~/.claude/plans/foamy-yawning-gadget.md`.
