# K-Scaling Experimental Plan: Irrep Structure Study

## Overview

This plan outlines experiments to systematically study how **embedding dimension K** (controlled by irrep choice) affects gauge VFE performance, independent of gauge group dimension N.

**Key insight from current results:**
- SO(3) with multi-irrep (ℓ=1,4,5,6,7) outperforms SO(50) with fundamental only
- This suggests irrep structure > gauge dimension for performance

## Current Data Status

All current experiments use only **fundamental irreps**:
- SO(2): K=2 (fund), 0.35M params
- SO(5): K=5 (fund), 1.26M params
- SO(10): K=10 (fund), 3.77M params
- SO(20): K=20 (fund), 12.56M params
- SO(30): K=30 (fund), 26.39M params
- SO(40): K=40 (fund), 45.23M params

**Data budget**: WikiText-103 has 117M tokens
**Chinchilla threshold**: ~20 tokens/param → models up to ~6M params are data-rich

---

## Experimental Design

### Experiment 1: SO(3) K-Scaling (Recommended - Highest Priority)

Fixed gauge group SO(3), vary K via different irrep combinations.

| Config | Irreps | K (embed dim) | Est. Params | Tokens/Param | Priority |
|--------|--------|---------------|-------------|--------------|----------|
| SO(3)-K3 | ℓ=1 (fund) | 3 | ~0.5M | 234 | HIGH |
| SO(3)-K5 | ℓ=2 | 5 | ~0.8M | 146 | HIGH |
| SO(3)-K7 | ℓ=3 | 7 | ~1.0M | 117 | HIGH |
| SO(3)-K8 | ℓ=1,2 | 3+5=8 | ~1.2M | 97 | HIGH |
| SO(3)-K12 | ℓ=1,3 | 3+7=10 | ~1.5M | 78 | MED |
| SO(3)-K15 | ℓ=1,2,3 | 3+5+7=15 | ~2.2M | 53 | HIGH |
| SO(3)-K24 | ℓ=1,2,3,4 | 3+5+7+9=24 | ~3.6M | 32 | MED |
| SO(3)-K35 | ℓ=1,2,3,4,5 | 3+5+7+9+11=35 | ~5.3M | 22 | HIGH |
| SO(3)-K48 | ℓ=1,2,3,4,5,6 | +13=48 | ~7.2M | 16 | MED |
| SO(3)-K63 | ℓ=1,2,3,4,5,6,7 | +15=63 | ~9.4M | 12 | LOW |

**Note**: SO(3) irrep ℓ has dimension 2ℓ+1, so ℓ=1→3, ℓ=2→5, ℓ=3→7, etc.

**Why SO(3)?**
- Best understood theoretically (rotation group)
- Rich irrep structure with clear physical interpretation
- Your best model already uses SO(3) multi-irrep
- Can compare single vs multi-irrep at same K


### Experiment 2: SO(N) at Fixed K (Secondary)

Fix K ≈ 20, vary gauge group to isolate effect of N.

| Config | Gauge | Irrep | K | Est. Params | Notes |
|--------|-------|-------|---|-------------|-------|
| SO(2)-K20 | SO(2) | 10× fund | 20 | ~3M | Max copies |
| SO(3)-K21 | SO(3) | ℓ=1,2,3 | 15 | ~2.2M | 3 irreps |
| SO(5)-K20 | SO(5) | 4× fund | 20 | ~3M | 4 copies |
| SO(10)-K20 | SO(10) | 2× fund | 20 | ~3M | 2 copies |
| SO(20)-K20 | SO(20) | 1× fund | 20 | ~3M | Baseline |

This tests: **Does gauge group matter when K is controlled?**


### Experiment 3: Multiplicity vs Irrep Diversity (Tertiary)

At fixed K ≈ 15, compare:

| Config | Irrep Spec | K | Heads | Question |
|--------|-----------|---|-------|----------|
| SO(3)-5×fund | 5× ℓ=1 | 15 | 5 identical | Multiple copies of same irrep |
| SO(3)-1,2,3 | ℓ=1,2,3 | 15 | 3 diverse | Different irreps |
| SO(5)-3×fund | 3× fund | 15 | 3 identical | SO(5) comparison |

This tests: **Is irrep diversity better than multiplicity?**

---

## Recommended Run Order

### Phase 1: Core K-scaling (6 runs, ~2-3 days)
1. SO(3)-K3 (ℓ=1 only) - baseline
2. SO(3)-K7 (ℓ=3 only) - single higher irrep
3. SO(3)-K8 (ℓ=1,2) - 2 irreps
4. SO(3)-K15 (ℓ=1,2,3) - 3 irreps
5. SO(3)-K35 (ℓ=1-5) - 5 irreps, near Chinchilla optimal
6. SO(3)-K5 (ℓ=2 only) - isolated ℓ=2

### Phase 2: Fixed-K comparison (4 runs, ~1-2 days)
7. SO(5)-K20 (4× fund)
8. SO(10)-K20 (2× fund)
9. SO(3)-K21 (ℓ=1,2,3,4) - match K≈20
10. SO(2)-K20 (10× fund)

### Phase 3: Refinement (optional)
- Fill in gaps based on Phase 1-2 results
- Test irrep diversity vs multiplicity

---

## Training Protocol

For fair comparison, all K-scaling experiments should use:

```
tokens_seen: 102.4M (same as N-scaling experiments)
batch_size: 128 (or adjust to fit GPU, keep tokens_seen constant)
seq_len: 64
eval_interval: 500 steps
seed: 6 (same as N-scaling for reproducibility)
```

**Steps calculation**: steps = 102.4M / (batch_size × seq_len)
- batch_size=128, seq_len=64 → 12,500 steps
- batch_size=64, seq_len=64 → 25,000 steps
- batch_size=32, seq_len=64 → 50,000 steps

---

## Expected Outcomes

1. **K-scaling exponent**: Expect PPL ∝ K^α with α similar to N-scaling (-0.4 to -0.5)

2. **Irrep efficiency**: Higher irreps (ℓ>1) may be more parameter-efficient than fundamental

3. **Multi-irrep advantage**: Combining irreps likely outperforms single irrep at same K

4. **Gauge group independence**: At fixed K, gauge group (N) may have minimal effect

---

## Analysis Plan

After experiments complete:

1. **Plot K vs Test PPL** (log-log) - fit power law
2. **Plot K vs Test PPL colored by gauge group** - test N-independence
3. **Compare multi-irrep vs single-irrep at matched K**
4. **Parameter efficiency**: PPL per million params across all configs
5. **Update manuscript Section on irrep scaling**

---

## Directory Structure

```
data/
├── seed=6/                    # Existing N-scaling data
│   ├── 1503_SO(2)_N=64_seed=6/
│   ├── ...
│   └── scaling_figures/
├── K_scaling/                  # New K-scaling experiments
│   ├── SO3_K3_ell1/
│   ├── SO3_K7_ell3/
│   ├── SO3_K8_ell12/
│   ├── SO3_K15_ell123/
│   ├── SO3_K35_ell12345/
│   └── ...
└── K_scaling_figures/          # K-scaling analysis plots
```

---

## Summary

| Priority | Experiments | Purpose |
|----------|-------------|---------|
| **HIGH** | SO(3) K-scaling (6 configs) | Core irrep scaling law |
| **MED** | Fixed-K across SO(N) | Isolate gauge group effect |
| **LOW** | Multiplicity vs diversity | Fine-grained irrep analysis |

**Total new runs**: ~10-14 experiments
**Estimated time**: 3-5 days on single RTX 5090
**Expected manuscript contribution**: New subsection on K-scaling / irrep efficiency
