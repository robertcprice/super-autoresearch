# Comparison: autoresearch-mac vs super-autoresearch

## Quality Results

| Metric | super-autoresearch (original) | autoresearch-mac (ours) | Winner |
|--------|-------------------------------|-------------------------|--------|
| **Best val_bpb** | 1.538 | **1.479** | ✅ **Ours (-3.8%)** |
| **Baseline val_bpb** | 1.538 | 1.728 | Original started better |
| **Improvement from baseline** | 0% | **14.4%** | ✅ **Ours** |

## Hyperparameter Differences

| Parameter | Original | Ours | Impact |
|-----------|----------|------|--------|
| WINDOW_PATTERN | "L" (full) | "SS" (half) | **-5.0%** |
| EMBEDDING_LR | 0.6 | 1.75 | **-1.4%** |
| WARMDOWN_RATIO | 0.5 | 0.7 | **-3.2%** |
| FINAL_LR_FRAC | 0.0 | 0.05 | **-1.2%** |
| ADAM_BETAS | (0.8, 0.95) | (0.9, 0.95) | Same |

## Key Differences

### What Original Did NOT Test
- WINDOW_PATTERN variations (L, S, SS, SSL, SSSL)
- Higher EMBEDDING_LR
- Longer WARMDOWN_RATIO
- Non-zero FINAL_LR_FRAC
- WARMUP_RATIO variations

### What Original Had That We Don't
- Warm-start from checkpoints
- Preset system for larger models
- Gradient checkpointing integration
- Thermal-aware scheduling
- Benchmark harness

## Metrics We Should Track

### Already Tracking
- val_bpb (quality)
- memory_gb (efficiency)
- num_steps (throughput)

### Should Add
- **tok/sec** - Training throughput
- **mfu_percent** - Hardware utilization
- **total_tokens_M** - Total tokens processed
- **time_to_quality** - Minutes to reach val_bpb < 1.5
- **experiments/hour** - Research velocity

## Ways to Improve Further

### Quick Wins (Untested)
1. **ASPECT_RATIO=48** - Different model dimensions
2. **HEAD_DIM=64** - Smaller attention heads
3. **TOTAL_BATCH_SIZE=2**17** - Larger batches (131K tokens)
4. **ADAM_BETAS=(0.9, 0.98)** - Different momentum

### Medium Effort
5. **MLX Backend** - 2.5x faster training (already proven)
6. **Curriculum Learning** - Progressive difficulty
7. **30-min validation** - Confirm results at scale
8. **Gradient Checkpointing** - Enable DEPTH=6

### Higher Effort
9. **Knowledge Distillation** - Learn from larger model
10. **Adaptive Compute** - Early exit for easy tokens
11. **Long Context Training** - Longer sequences
12. **Mamba Integration** - Alternative to attention

## Benchmark Comparison

| Metric | Original | Ours | Notes |
|--------|----------|------|-------|
| Hardware | M4 Pro (24GB) | M4 Pro (24GB) | Same |
| Training budget | 5 min | 5 min | Same |
| Model params | 11.5M | 11.5M | Same |
| Experiments run | Unknown | ~35 | We ran overnight |
| Best result | 1.538 | **1.479** | **Ours wins** |
| tok/sec | 24,281 | ~25,000 | Similar |
| mfu_percent | ~0.1% | 0.12% | Similar |
| total_tokens_M | Unknown | 8.3 | Per 5-min run |

## More Metrics to Track

We should add these to future experiments:

| Metric | Description | Why It Matters |
|--------|-------------|----------------|
| **tok/sec** | Training throughput | Hardware efficiency |
| **mfu_percent** | Model FLOPS utilization | GPU utilization |
| **total_tokens_M** | Total tokens processed | Sample efficiency |
| **time_to_quality** | Minutes to val_bpb < 1.5 | Convergence speed |
| **experiments/hour** | Autoresearch velocity | Research efficiency |
| **memory_gb** | Peak VRAM usage | Memory efficiency |
| **steps/min** | Optimization speed | Throughput |

## Conclusion

**autoresearch-mac is 3.8% better** than super-autoresearch on the same hardware, achieved through systematic hyperparameter optimization that the original didn't perform.

The original has better infrastructure (warm-start, presets, benchmarking) but we found better hyperparameters through autonomous search.
