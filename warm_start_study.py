#!/usr/bin/env python3
"""
Study whether sample-stage warm starts help relative to a cold full run.

The warm path spends part of the total training budget in the sample stage, then
uses the remaining budget for the promoted full run. This keeps the comparison
closer to equal total wall time than simply giving the warm run a fresh 5
minutes after sampling.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from benchmark import BenchmarkResult, Variant, machine_info, make_variant, run_variant
from experiment_memory import ExperimentMemory, benchmark_payload_to_entries, default_memory_path
from thermal_tuner import recommend_runtime_profile


REPO_ROOT = Path(__file__).resolve().parent


def _result_dict(result: BenchmarkResult, label_suffix: str, note_suffix: str) -> dict:
    payload = asdict(result)
    payload["label"] = f"{result.label}__{label_suffix}"
    payload["note"] = f"{result.note}; {note_suffix}"
    return payload


def _metric(result: BenchmarkResult, key: str) -> float | None:
    value = result.summary.get(key)
    if value is None:
        return None
    return float(value)


def _comparison(sample_result: BenchmarkResult, warm_result: BenchmarkResult, cold_result: BenchmarkResult) -> dict[str, float | None]:
    sample_wall_seconds = float(sample_result.elapsed_seconds or 0.0)
    warm_total_seconds = _metric(warm_result, "total_seconds")
    cold_total_seconds = _metric(cold_result, "total_seconds")
    warm_val_bpb = _metric(warm_result, "val_bpb")
    cold_val_bpb = _metric(cold_result, "val_bpb")
    warm_eval_seconds = _metric(warm_result, "eval_seconds")
    cold_eval_seconds = _metric(cold_result, "eval_seconds")

    return {
        "sample_wall_seconds": sample_wall_seconds,
        "warm_pipeline_wall_seconds": sample_wall_seconds + warm_total_seconds if warm_total_seconds is not None else None,
        "cold_pipeline_wall_seconds": cold_total_seconds,
        "warm_minus_cold_val_bpb": warm_val_bpb - cold_val_bpb if warm_val_bpb is not None and cold_val_bpb is not None else None,
        "warm_minus_cold_total_seconds": warm_total_seconds - cold_total_seconds if warm_total_seconds is not None and cold_total_seconds is not None else None,
        "warm_minus_cold_eval_seconds": warm_eval_seconds - cold_eval_seconds if warm_eval_seconds is not None and cold_eval_seconds is not None else None,
        "warm_minus_cold_avg_last_five_tok_per_sec": (
            float(warm_result.avg_last_five_tok_per_sec) - float(cold_result.avg_last_five_tok_per_sec)
            if warm_result.avg_last_five_tok_per_sec is not None and cold_result.avg_last_five_tok_per_sec is not None
            else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare warm-started promotion against a cold full run.")
    parser.add_argument("--variant", default="current", help="Variant spec to study.")
    parser.add_argument("--sample-duration", type=int, help="Wall-clock seconds for the sample stage.")
    parser.add_argument("--total-training-budget", type=int, default=300, help="Target total training-budget seconds across sample + warm run.")
    parser.add_argument("--force-full-run", action="store_true", help="Ignore the thermal gate and run the full comparison anyway.")
    parser.add_argument("--output-dir", type=Path, help="Directory for artifacts.")
    parser.add_argument("--memory-path", type=Path, default=default_memory_path())
    args = parser.parse_args()

    profile = recommend_runtime_profile()
    if not profile.allow_full_run and not args.force_full_run:
        print("Thermal profile does not recommend full runs right now; rerun with --force-full-run to continue.")
        return 0

    sample_duration = args.sample_duration or profile.sample_duration_seconds
    if sample_duration >= args.total_training_budget:
        raise ValueError("sample duration must be smaller than total training budget")

    output_dir = args.output_dir or (REPO_ROOT / "warm_start_runs" / datetime.now().strftime("%Y%m%d_%H%M%S"))
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = output_dir / "sample"
    warm_dir = output_dir / "warm"
    cold_dir = output_dir / "cold"
    sample_dir.mkdir(parents=True, exist_ok=True)
    warm_dir.mkdir(parents=True, exist_ok=True)
    cold_dir.mkdir(parents=True, exist_ok=True)

    base_variant = make_variant(args.variant)
    checkpoint_path = sample_dir / f"{base_variant.label}.ckpt.pt"

    sample_env = dict(base_variant.env_overrides)
    sample_env["AUTORESEARCH_PERIODIC_CHECKPOINT"] = str(checkpoint_path)
    sample_env["AUTORESEARCH_CHECKPOINT_EVERY_STEPS"] = "1"
    sample_variant = Variant(
        spec=base_variant.spec,
        label=base_variant.label,
        script_path=base_variant.script_path,
        workdir=base_variant.workdir,
        note=base_variant.note,
        env_overrides=sample_env,
        tempdir=base_variant.tempdir,
    )
    sample_result = run_variant(sample_variant, sample_duration, sample_dir)

    if not checkpoint_path.exists():
        raise RuntimeError(f"sample checkpoint was not created: {checkpoint_path}")

    remaining_budget = max(1, args.total_training_budget - sample_duration)

    warm_env = dict(base_variant.env_overrides)
    warm_env["AUTORESEARCH_INIT_FROM"] = str(checkpoint_path)
    warm_env["AUTORESEARCH_TIME_BUDGET_SECONDS"] = str(remaining_budget)
    warm_variant = Variant(
        spec=base_variant.spec,
        label=base_variant.label,
        script_path=base_variant.script_path,
        workdir=base_variant.workdir,
        note=base_variant.note,
        env_overrides=warm_env,
        tempdir=base_variant.tempdir,
    )
    warm_result = run_variant(warm_variant, None, warm_dir)

    cold_env = dict(base_variant.env_overrides)
    cold_env["AUTORESEARCH_TIME_BUDGET_SECONDS"] = str(args.total_training_budget)
    cold_variant = Variant(
        spec=base_variant.spec,
        label=base_variant.label,
        script_path=base_variant.script_path,
        workdir=base_variant.workdir,
        note=base_variant.note,
        env_overrides=cold_env,
        tempdir=base_variant.tempdir,
    )
    cold_result = run_variant(cold_variant, None, cold_dir)

    comparison = _comparison(sample_result, warm_result, cold_result)
    payload = {
        "generated_at": datetime.now().isoformat(),
        "mode": "warm-start-study",
        "machine": machine_info(),
        "runtime_profile": asdict(profile),
        "study": {
            "variant": args.variant,
            "sample_duration_seconds": sample_duration,
            "total_training_budget_seconds": args.total_training_budget,
            "warm_remaining_budget_seconds": remaining_budget,
        },
        "results": [
            _result_dict(sample_result, "sample", "sample stage"),
            _result_dict(warm_result, "warm", f"warm-start from {checkpoint_path.name}"),
            _result_dict(cold_result, "cold", "cold full run"),
        ],
        "comparison": comparison,
    }
    output_path = output_dir / "results.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    memory = ExperimentMemory(args.memory_path)
    added = memory.append_many(benchmark_payload_to_entries(payload, output_path))

    print(f"artifacts: {output_dir}")
    print(f"sample_checkpoint: {checkpoint_path}")
    print(f"warm_val_bpb: {warm_result.summary.get('val_bpb')}")
    print(f"cold_val_bpb: {cold_result.summary.get('val_bpb')}")
    print(f"warm_minus_cold_val_bpb: {comparison['warm_minus_cold_val_bpb']}")
    print(f"warm_pipeline_wall_seconds: {comparison['warm_pipeline_wall_seconds']}")
    print(f"cold_pipeline_wall_seconds: {comparison['cold_pipeline_wall_seconds']}")
    print(f"memory_entries_added: {added}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
