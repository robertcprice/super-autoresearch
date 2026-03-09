# autoresearch

This repo is an experiment in having the LLM run its own research loop on Apple Silicon.

## Setup

To set up a new experiment, work with the user to:

1. Agree on a run tag. Propose a tag based on today's date, such as `mar8`.
2. Create a branch `autoresearch/<tag>` from the current default branch.
3. Read the in-scope files:
   - `README.md`
   - `prepare.py`
   - `train.py`
4. Verify data exists in `~/.cache/autoresearch/`. If not, tell the human to run `uv run prepare.py`.
5. Initialize `results.tsv` with the header row only. Do not seed a fake baseline entry.
6. Confirm the setup and then start experimenting.

## Experimentation

Each experiment runs through the canonical entrypoint:

```bash
uv run train.py
```

The script runs for a fixed 5-minute training budget and then performs evaluation. The main metric is `val_bpb`, where lower is better.

What you can do:

- Modify `train.py`

What you cannot do:

- Modify `prepare.py`
- Change the evaluation harness
- Add new dependencies

The first actual run establishes the baseline. Record the real output from that run in `results.tsv`; do not assume a precomputed baseline from another machine.

## Output Format

The script finishes with a summary shaped like this:

```text
---
val_bpb:          1.234567
training_seconds: 300.0
total_seconds:    324.0
peak_vram_mb:     4096.0
mfu_percent:      0.20
total_tokens_M:   10.5
num_steps:        160
num_params_M:     11.5
depth:            4
```

These numbers vary by machine, defaults, and runtime configuration. Always log the numbers from the run you just executed.

## Logging Results

Keep `results.tsv` as tab-separated values with this schema:

```text
commit	val_bpb	memory_gb	status	description
```

Use:

1. short git commit hash
2. `val_bpb`
3. peak memory in GB (`peak_vram_mb / 1024`, rounded to one decimal)
4. `keep`, `discard`, or `crash`
5. short experiment description

## Loop

Repeat:

1. Check the current branch and commit.
2. Edit `train.py` with one focused idea.
3. Commit the change.
4. Run `uv run train.py > run.log 2>&1`.
5. Read `val_bpb` and `peak_vram_mb` from `run.log`.
6. If the run crashed, inspect the traceback, decide whether it is worth a quick fix, and otherwise discard it.
7. Log the result in `results.tsv`.
8. Keep commits that improve `val_bpb`; discard or rewind changes that do not.

Do not stop after a single successful experiment. Keep iterating until the human redirects you.
