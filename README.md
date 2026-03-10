# super-autoresearch

PyTorch + MPS research platform built on [karpathy/autoresearch](https://github.com/karpathy/autoresearch) for Apple Silicon. Based on [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos).

Point Claude Code at `program.md`, go to sleep, wake up to results.

## Quick start

Requirements: Apple Silicon Mac, Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run prepare.py
uv run train.py
```

Then point your agent at `program.md` and let it run overnight.

## Results on M4 Pro (24GB)

| Run | Config | val_bpb | tok/s | Notes |
|---|---|---|---|---|
| **optimized** | EMB_LR=2.0, MAT_LR=0.08 | **1.486** | ~32,000 | 5-min budget, optimized LRs |
| default | baseline | 1.538 | 24,281 | 5-min budget, default depth |
| origin/master | upstream port | 1.542 | 29,607 | miolini baseline |
| warm-start | 20s sample + 100s promoted | 1.980 | 27,889 | checkpoint warm-start |
| cold | 120s full budget | 1.982 | 6,390 | no warm-start |

**Our optimized config is 9% better than baseline** (1.486 vs 1.633 on identical runs).

Upstream H100 reference: val_bpb **0.998** in the same 5-minute budget.

### Warm-start study

Built-in tooling to answer: does warm-starting from a sample checkpoint beat a cold run?

| Path | val_bpb | Wall time |
|---|---|---|
| Warm (20s sample + 100s promoted) | **1.980** | 505s |
| Cold (120s full budget) | 1.982 | 583s |

Warm start wins on both quality (-0.002 bpb) and speed (-78s wall time).

## How it works

Same three-file contract as upstream:

- **`prepare.py`** — data prep, tokenizer, dataloader, evaluation. Fixed.
- **`train.py`** — model, optimizer, training loop. The agent edits this.
- **`program.md`** — agent instructions. Point your agent here.

The agent reads `program.md`, modifies `train.py`, runs a 5-minute experiment, checks `val_bpb`, and commits or reverts.

## Differences from upstream

- **PyTorch MPS instead of CUDA.** Runs natively on Apple Silicon unified memory.
- **Preset system.** `AUTORESEARCH_PRESET=large` for scaled-up configs.
- **Gradient checkpointing.** Fits larger models in unified memory.
- **Checkpoint warm-starts.** Resume from sample checkpoints to skip early training.
- **Eval batch auto-probing.** Finds the largest safe eval batch size via OOM-safe binary search.
- **Thermal-aware scheduling.** Respects macOS power/thermal state before launching long runs.
- **Benchmark harness.** Compare variants, track experiment history, reproduce results.

## Research tooling

Beyond the core loop, this repo includes tools for systematic experimentation:

```bash
# benchmark current vs origin/master (30s sample)
uv run benchmark.py

# full 5-minute quality comparison
uv run benchmark.py --full-run

# benchmark the large preset
uv run benchmark.py --variant 'current@AUTORESEARCH_PRESET=large'

# warm-start study: does sample + promote beat cold?
uv run warm_start_study.py --total-training-budget 120

# rank candidate presets and promote the winner
uv run candidate_scheduler.py --sample-duration 20 --promote-top 1
```

## Environment overrides

| Variable | Default | Description |
|---|---|---|
| `AUTORESEARCH_PRESET` | (none) | Model preset: `large`, etc. |
| `AUTORESEARCH_TIME_BUDGET_SECONDS` | 300 | Override the 5-minute training budget |
| `AUTORESEARCH_EVAL_BATCH_SIZE` | (training batch) | `auto` to probe, or an integer |
| `AUTORESEARCH_INIT_FROM` | (none) | Path to checkpoint for warm-start |
| `AUTORESEARCH_MPS_AUTOCAST` | off | Set to `bf16` to enable MPS bf16 autocast |

## File map

Core path:

| File | Purpose |
|---|---|
| `prepare.py` | Data, tokenizer, dataloader, evaluation (fixed) |
| `train.py` | Model, optimizer, training loop (agent-editable) |
| `program.md` | Agent instructions |
| `mpsc_config.py` | Preset definitions |
| `mpsc_checkpoint.py` | Gradient checkpointing wrapper |
| `checkpoint_reuse.py` | Save/load training checkpoints |

Research tooling:

| File | Purpose |
|---|---|
| `benchmark.py` | Variant comparison harness |
| `experiment_memory.py` | Experiment history (JSONL) |
| `thermal_tuner.py` | macOS power/thermal heuristics |
| `candidate_scheduler.py` | Multi-preset ranking and promotion |
| `warm_start_study.py` | Warm-start vs cold A/B study |

## License

MIT
