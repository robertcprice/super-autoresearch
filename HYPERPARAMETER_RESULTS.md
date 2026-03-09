# Hyperparameter Experiment Results

**Best Result**: val_bpb = 1.479239 (14.4% improvement over baseline 1.727831)

## Summary Table

| Parameter | Baseline | Best | Impact |
|-----------|----------|------|--------|
| WINDOW_PATTERN | "LLLL" | "SS" | -5.0% |
| WARMDOWN_RATIO | 0.5 | 0.7 | -3.2% |
| EMBEDDING_LR | 1.0 | 1.75 | -1.4% |
| FINAL_LR_FRAC | 0.0 | 0.05 | -1.2% |

## Detailed Results

### WINDOW_PATTERN (Sliding Window Attention)

| Pattern | val_bpb | Status | Notes |
|---------|---------|--------|-------|
| LLLL | 1.727831 | baseline | Full context for all layers |
| SSSL | 1.648532 | discard | 1.5% better |
| SSL | 1.567433 | discard | 5% better |
| SS | 1.520233 | **KEEP** | 3% better than SSL, best overall |

**Insight**: Using half-context windows for ALL layers (SS) works best. This enables O(n) memory instead of O(n²) for attention.

### WARMDOWN_RATIO (LR Decay Schedule)

| Value | val_bpb | Status | Notes |
|-------|---------|--------|-------|
| 0.5 | 1.727831 | baseline | Default |
| 0.6 | 1.719930 | discard | Slightly worse |
| 0.7 | 1.672866 | **KEEP** | 3.2% better |
| 0.8 | 1.775099 | discard | Worse |

**Insight**: Spending 70% of training in LR decay allows smoother convergence.

### EMBEDDING_LR (Embedding Learning Rate)

| Value | val_bpb | Status | Notes |
|-------|---------|--------|-------|
| 0.75 | 1.497664 | keep | 1.4% better |
| 0.8 | 1.503079 | discard | With SS pattern |
| 0.9 | 1.507081 | discard | Slightly worse |
| 1.0 | 1.898929 | discard | Much worse |
| 1.5 | 1.655855 | discard | Worse |
| 1.75 | 1.503698 | **KEEP** | Works well with other params |
| 2.0 | 1.609976 | discard | Worse |

**Insight**: Higher embedding LR (1.75) benefits from more aggressive learning, but 0.75 also works well.

### FINAL_LR_FRAC (Final Learning Rate)

| Value | val_bpb | Status | Notes |
|-------|---------|--------|-------|
| 0.0 | 1.727831 | baseline | LR decays to zero |
| 0.02 | 1.753211 | discard | Much worse (too fast decay) |
| 0.05 | 1.479239 | **KEEP** | 1.2% better |
| 0.1 | 1.480826 | discard | Slightly worse |

**Insight**: Non-zero final LR (5%) prevents too-rapid decay and keeps gradient signal flowing.

### MATRIX_LR (Muon Learning Rate)

| Value | val_bpb | Status | Notes |
|-------|---------|--------|-------|
| 0.03 | 1.533106 | discard | Worse |
| 0.04 | — | **KEEP** | Default is optimal |
| 0.06 | 1.867725 | discard | Much worse |

**Insight**: Muon optimizer's default 0.04 is near-optimal.

### SCALAR_LR (Scalar Parameter Learning Rate)

| Value | val_bpb | Status | Notes |
|-------|---------|--------|-------|
| 0.3 | 1.558004 | discard | Worse |
| 0.5 | — | **KEEP** | Default is optimal |
| 0.7 | 1.672183 | discard | Worse |

**Insight**: Default 0.5 works best for per-layer scalars.

### UNEMBEDDING_LR (Output Head Learning Rate)

| Value | val_bpb | Status | Notes |
|-------|---------|--------|-------|
| 0.002 | 1.712650 | discard | Worse |
| 0.004 | — | **KEEP** | Default is optimal |
| 0.01 | 1.604147 | discard | Worse |

**Insight**: Default 0.004 works best for lm_head.

### WEIGHT_DECAY

| Value | val_bpb | Status | Notes |
|-------|---------|--------|-------|
| 0.2 | — | **KEEP** | Default |
| 0.3 | 1.877147 | discard | Much worse |

**Insight**: Cautious weight decay (0.2) is optimal for Muon.

### ADAM_BETAS

| Value | val_bpb | Status | Notes |
|-------|---------|--------|-------|
| (0.9, 0.95) | — | **KEEP** | Default |
| (0.9, 0.99) | 1.688533 | discard | Worse |

**Insight**: Default betas work best.

### bf16 Autocast

| Setting | val_bpb | Status | Notes |
|---------|---------|--------|-------|
| fp32 | — | **KEEP** | Default |
| bf16 autocast | 1.753905 | discard | Worse on MPS |

**Insight**: bf16 autocast hurts quality on Apple Silicon MPS.

## Unexplored Parameters

These parameters were NOT tested and could yield further improvements:

| Parameter | Current Value | Potential Impact |
|-----------|---------------|------------------|
| ASPECT_RATIO | 64 | Affects model_dim = depth * ASPECT_RATIO |
| HEAD_DIM | 128 | Affects attention head structure |
| DEPTH | 4 | Number of transformer layers |
| WARMUP_RATIO | 0.0 | Could help with early training stability |
| TOTAL_BATCH_SIZE | 65536 | Larger batches may improve convergence |

## Novel Systems to Explore

Based on the existing experimental modules in this repo:

1. **Gradient Checkpointing** (`mpsc_checkpoint.py`) - Trade compute for memory to enable larger models
2. **Curriculum Learning** (`mpsc_curriculum.py`) - Train on easier examples first
3. **Knowledge Distillation** (`mpsc_distillation.py`) - Distill from larger model
4. **Long Context** (`mpsc_long_context.py`) - Extend sequence length
5. **Mamba Integration** (`mpsc_mamba.py`) - Alternative to attention
6. **Metal Kernels** (`mpsc_metal_kernels.py`) - Custom GPU kernels for MPS
7. **MLX Backend** (`mlx_gpt.py`) - Apple's MLX framework instead of PyTorch
8. **Adaptive Compute** (`mpsc_adaptive_compute.py`) - Dynamic compute allocation
9. **Thermal-Aware Tuning** (`thermal_tuner.py`) - Adjust params based on device temperature
10. **Experiment Memory** (`experiment_memory.py`) - Learn from past experiments

## Best Configuration

```python
# Model architecture
ASPECT_RATIO = 64
HEAD_DIM = 128
WINDOW_PATTERN = "SS"    # KEY: half-context for all layers

# Optimization
TOTAL_BATCH_SIZE = 2**16
EMBEDDING_LR = 1.75     # KEY: higher embedding LR
UNEMBEDDING_LR = 0.004  # keep default
MATRIX_LR = 0.04        # keep default
SCALAR_LR = 0.5         # keep default
WEIGHT_DECAY = 0.2      # keep default
ADAM_BETAS = (0.9, 0.95)
WARMUP_RATIO = 0.0
WARMDOWN_RATIO = 0.7    # KEY: longer warmdown
FINAL_LR_FRAC = 0.05    # KEY: non-zero final LR

# Model size
DEPTH = 4
DEVICE_BATCH_SIZE = 16
```
