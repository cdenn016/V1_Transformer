# PyTorch Lightning Migration Plan for VFE Gauge Transformer

## Executive Summary

**Verdict: Yes, PyTorch Lightning is highly beneficial for this project.**

The current codebase has **two separate raw-PyTorch trainers** (`FastTrainer` in `train_fast.py` and `PublicationTrainer` in `train_publication.py`) with ~400 lines of duplicated boilerplate for AMP, gradient accumulation, checkpointing, W&B logging, early stopping, and LR scheduling. PyTorch Lightning would eliminate this duplication, add multi-GPU scaling (critical for WikiText-103/OpenWebText experiments), and provide a clean callback system for the project's many custom training hooks (RG metrics, P-flow EMA, delta rule, gauge geometry monitoring).

---

## Why Lightning Is Beneficial — Specific to This Project

### 1. Eliminate Trainer Duplication (HIGH IMPACT)
**Current problem:** `FastTrainer` (850 lines) and `PublicationTrainer` (800+ lines) both implement:
- Manual AMP with `torch.amp.autocast` + `GradScaler`
- Manual gradient accumulation with modulo checks
- Manual checkpoint save/load with old-checkpoint cleanup
- Manual W&B integration
- Manual early stopping with patience counters
- Manual LR scheduling with warmup + cosine decay
- Manual validation loops with token-weighted averaging

**Lightning solution:** All of this is built-in. One `LightningModule` replaces both trainers.

### 2. Multi-GPU / Multi-Node Scaling (HIGH IMPACT)
**Current problem:** Training is single-GPU only. The gauge transformer's `matrix_exp` operations, iterative VFE E-steps (`ffn_n_iterations`), and Newton-Schulz orthogonalization are computationally expensive. Scaling to WikiText-103 (103M tokens) or OpenWebText (8B tokens) on a single GPU is impractical.

**Lightning solution:** `Trainer(devices=4, strategy='ddp')` — zero code changes needed. FSDP for model-parallel if the model grows. DeepSpeed integration for memory efficiency.

### 3. Custom Parameter Groups Work Natively (NO RISK)
**Current concern:** The 6-group natural gradient optimizer (mu, sigma, phi, attention, ffn, output) is critical to the project's theory.

**Lightning solution:** `configure_optimizers()` returns the exact same `AdamW` with parameter groups. No changes to the optimization logic. Example:
```python
def configure_optimizers(self):
    param_groups = create_param_groups(self, self.config)
    optimizer = AdamW(param_groups, ...)
    scheduler = CosineAnnealingLR(...)
    return [optimizer], [scheduler]
```

### 4. Callbacks for Custom Training Hooks (MEDIUM IMPACT)
**Current problem:** RG metrics, P-flow EMA updates, delta rule for W_out, Killing form preconditioning, and numerical monitor flushing are all interleaved in the main training loop, making it hard to enable/disable features and hard to read.

**Lightning solution:** Each becomes a clean callback:
- `RGFlowCallback` — computes modularity, effective rank, meta-agent detection at intervals
- `PFlowCallback` — EMA update of token embeddings toward successful beliefs
- `DeltaRuleCallback` — backprop-free W_out learning
- `GaugeGeometryCallback` — SL(K) projection, Killing form monitoring
- `NumericalMonitorCallback` — flushes fallback counters at log intervals

### 5. Built-in Profiling (MEDIUM IMPACT)
**Current problem:** No profiling infrastructure. It's unclear where the bottleneck is: `matrix_exp`? KL computation? Newton-Schulz? VFE iterations?

**Lightning solution:** `Trainer(profiler='advanced')` gives per-function timing breakdown. Critical for optimizing the VFE E-step.

### 6. Precision Plugins (LOW-MEDIUM IMPACT)
**Current problem:** Manual AMP can be fragile with the VFE loss (KL divergences involve `log_det` computations that can underflow in fp16).

**Lightning solution:** Precision plugins with bf16 (better dynamic range for log-det), mixed precision with custom autocast regions, and automatic loss scaling.

### 7. Logging Abstraction (LOW IMPACT)
**Current problem:** Manual W&B integration with `if self.config.use_wandb and WANDB_AVAILABLE` checks scattered throughout.

**Lightning solution:** `self.log('train/loss', loss)` works with any logger (W&B, TensorBoard, CSV, MLflow) — configured once at trainer level.

---

## What Would NOT Change

- **Model architecture** (`GaugeTransformerLM`, `IrrepMultiHeadAttention`, `VariationalFFNDynamic`) — untouched
- **Loss function** (`compute_free_energy_loss`) — called from `training_step()` exactly as before
- **Math utilities** (`math_utils/`) — completely independent of training framework
- **Data loading** (`create_dataloaders`) — Lightning's `LightningDataModule` wraps this cleanly
- **Analysis/visualization** — completely independent
- **Test suite** — model/attention/FFN tests are framework-independent

---

## Implementation Plan

### Phase 1: LightningModule (Core)
**File:** `transformer/training/lightning_module.py`

```python
class GaugeTransformerLightning(pl.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config
        self.save_hyperparameters(ignore=['model'])

    def forward(self, input_ids):
        return self.model(input_ids)

    def training_step(self, batch, batch_idx):
        input_ids, target_ids = batch
        loss, metrics = compute_free_energy_loss(
            self.model, input_ids, target_ids,
            alpha=self.config.alpha,
            lambda_beta=self.config.lambda_beta,
            ...
        )
        # Log all metrics (works with ANY logger)
        self.log_dict({f'train/{k}': v for k, v in metrics.items()})
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, target_ids = batch
        _, metrics = compute_free_energy_loss(
            self.model, input_ids, target_ids,
            alpha=0.0, lambda_beta=0.0,  # Pure CE for validation
        )
        self.log('val/ce_loss', metrics['loss/ce'], sync_dist=True)
        self.log('val/perplexity', math.exp(metrics['loss/ce']), sync_dist=True)

    def configure_optimizers(self):
        param_groups = create_param_groups(self.model, self.config)
        optimizer = torch.optim.AdamW(param_groups, ...)
        scheduler = self._create_scheduler(optimizer)
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]
```

### Phase 2: LightningDataModule
**File:** `transformer/training/lightning_data.py`

```python
class GaugeTransformerDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        self.train_loader, self.val_loader, self.vocab_size = create_dataloaders(
            dataset=self.config.dataset,
            batch_size=self.config.batch_size,
            max_seq_len=self.config.max_seq_len,
        )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader
```

### Phase 3: Custom Callbacks
**File:** `transformer/training/callbacks.py`

```python
class RGFlowCallback(pl.Callback):
    """Compute RG flow metrics at configurable intervals."""

class PFlowCallback(pl.Callback):
    """EMA update of token embeddings toward successful beliefs."""

class GaugeGeometryCallback(pl.Callback):
    """SL(K) projection and Killing form monitoring after optimizer step."""

class VFEDiagnosticsCallback(pl.Callback):
    """Log VFE component breakdown (CE, alpha, beta, gamma terms)."""
```

### Phase 4: Training Script
**File:** `transformer/training/train_lightning.py`

```python
# Simple entry point
config = get_vfe_dynamic_config()
model = GaugeTransformerLM(block_config)
lit_model = GaugeTransformerLightning(model, config)
datamodule = GaugeTransformerDataModule(config)

trainer = pl.Trainer(
    max_steps=config.max_steps,
    precision='bf16-mixed',          # Better than fp16 for log-det ops
    gradient_clip_val=config.grad_clip,
    accumulate_grad_batches=config.accumulation_steps,
    callbacks=[
        ModelCheckpoint(monitor='val/ce_loss', save_top_k=3),
        EarlyStopping(monitor='val/ce_loss', patience=config.patience),
        LearningRateMonitor(),
        RGFlowCallback(interval=config.rg_metrics_interval),
        GaugeGeometryCallback(),
    ],
    logger=WandbLogger(project='gauge-transformer') if config.use_wandb else CSVLogger('logs/'),
    devices='auto',
    strategy='auto',  # DDP if multiple GPUs, else single
)

trainer.fit(lit_model, datamodule=datamodule)
```

### Phase 5: Backward Compatibility
- Keep `FastTrainer` and `PublicationTrainer` as-is (deprecated but functional)
- New code uses Lightning exclusively
- Tests for the Lightning module alongside existing tests

---

## Lines of Code Comparison

| Component | Current (Raw PyTorch) | Lightning |
|---|---|---|
| Training loop boilerplate | ~400 lines (duplicated x2) | 0 (built-in) |
| AMP handling | ~30 lines | 1 line (`precision='bf16-mixed'`) |
| Gradient accumulation | ~15 lines | 1 line (`accumulate_grad_batches=N`) |
| Checkpointing | ~80 lines | 1 callback (`ModelCheckpoint`) |
| Early stopping | ~20 lines | 1 callback (`EarlyStopping`) |
| W&B integration | ~40 lines | 1 line (`logger=WandbLogger(...)`) |
| Multi-GPU | Not supported | 1 line (`devices='auto'`) |
| **LightningModule** | N/A | ~120 lines (clean, focused) |
| **Callbacks** | Interleaved in loop | ~150 lines (modular) |
| **Total new code** | ~1600 lines across 2 trainers | ~300 lines |

---

## Risk Assessment

| Risk | Severity | Mitigation |
|---|---|---|
| Custom param groups break | LOW | `configure_optimizers()` supports arbitrary groups natively |
| VFE loss incompatible | NONE | Called identically from `training_step()` |
| P-flow/delta rule timing | LOW | `on_train_batch_end` callback fires at exact right time |
| Existing tests break | NONE | Model tests are framework-independent; keep old trainers |
| Performance overhead | NEGLIGIBLE | Lightning adds <1% overhead; gains from DDP dwarf this |
| Learning curve | LOW | Lightning API is well-documented; core is 3 methods |

---

## Recommended Approach

**Incremental migration, not big-bang rewrite:**

1. Add `pytorch-lightning` to dependencies
2. Create `lightning_module.py` with `GaugeTransformerLightning`
3. Create `lightning_data.py` with `GaugeTransformerDataModule`
4. Create `callbacks.py` with RG flow, P-flow, gauge geometry callbacks
5. Create `train_lightning.py` as the new entry point
6. Add tests for the Lightning module
7. Deprecate (but don't delete) `FastTrainer` and `PublicationTrainer`

This preserves all existing functionality while unlocking multi-GPU training, eliminating boilerplate, and making the codebase more maintainable.
