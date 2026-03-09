"""
Short-run ranking and promotion scheduler built on top of benchmark.py.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from benchmark import Variant, machine_info, make_variant, run_variant
from experiment_memory import ExperimentMemory, benchmark_payload_to_entries, default_memory_path
from mpsc_config import list_presets
from thermal_tuner import recommend_runtime_profile


REPO_ROOT = Path(__file__).resolve().parent


def sample_rank_key(result):
    if result.avg_last_five_tok_per_sec is None:
        return float("-inf")
    return float(result.avg_last_five_tok_per_sec)


def full_rank_key(result):
    if "val_bpb" in result.summary:
        return -float(result.summary["val_bpb"])
    if result.avg_last_five_tok_per_sec is None:
        return float("-inf")
    return float(result.avg_last_five_tok_per_sec)


def _write_payload(path: Path, mode: str, profile, results):
    payload = {
        "generated_at": datetime.now().isoformat(),
        "mode": mode,
        "runtime_profile": asdict(profile),
        "machine": machine_info(),
        "results": [asdict(result) for result in results],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _default_variants() -> list[str]:
    variants = ["current"]
    for preset in list_presets():
        if preset == "tiny":
            continue
        variants.append(f"current@AUTORESEARCH_PRESET={preset}")
    if (REPO_ROOT / "train_optimized.py").exists():
        variants.append("path:train_optimized.py")
    return variants


def main() -> int:
    parser = argparse.ArgumentParser(description="Rank candidate training variants with short-run samples.")
    parser.add_argument("--variant", action="append", dest="variants", help="Variant spec: current, ref:<git-ref>, or path:<script.py>.")
    parser.add_argument("--sample-duration", type=int, help="Override the thermal-aware sample duration.")
    parser.add_argument("--promote-top", type=int, default=0, help="Run full benchmarks for the top N short-run candidates.")
    parser.add_argument("--force-full-run", action="store_true", help="Ignore the thermal gate and run the promotion stage anyway.")
    parser.add_argument("--output-dir", type=Path, help="Directory for scheduler artifacts.")
    parser.add_argument("--memory-path", type=Path, default=default_memory_path())
    args = parser.parse_args()

    profile = recommend_runtime_profile()
    sample_duration = args.sample_duration or profile.sample_duration_seconds
    output_dir = args.output_dir or (REPO_ROOT / "scheduler_runs" / datetime.now().strftime("%Y%m%d_%H%M%S"))
    output_dir.mkdir(parents=True, exist_ok=True)

    variant_specs = args.variants or _default_variants()
    variants = [make_variant(spec) for spec in variant_specs]
    sample_dir = output_dir / "sample"
    full_dir = output_dir / "full"
    sample_dir.mkdir(parents=True, exist_ok=True)
    full_dir.mkdir(parents=True, exist_ok=True)

    sample_variants = []
    sample_checkpoint_paths: dict[str, Path] = {}
    for variant in variants:
        checkpoint_path = sample_dir / f"{variant.label}.ckpt.pt"
        sample_checkpoint_paths[variant.spec] = checkpoint_path
        env_overrides = dict(variant.env_overrides)
        env_overrides.setdefault("AUTORESEARCH_PERIODIC_CHECKPOINT", str(checkpoint_path))
        env_overrides.setdefault("AUTORESEARCH_CHECKPOINT_EVERY_STEPS", "1")
        sample_variants.append(
            Variant(
                spec=variant.spec,
                label=variant.label,
                script_path=variant.script_path,
                workdir=variant.workdir,
                note=variant.note,
                env_overrides=env_overrides,
                tempdir=variant.tempdir,
            )
        )

    sample_results = [run_variant(variant, sample_duration, sample_dir) for variant in sample_variants]
    ranked_samples = sorted(sample_results, key=sample_rank_key, reverse=True)
    sample_payload = _write_payload(output_dir / "sample_results.json", f"{sample_duration}s sample", profile, sample_results)

    memory = ExperimentMemory(args.memory_path)
    memory.append_many(benchmark_payload_to_entries(sample_payload, output_dir / "sample_results.json"))

    print("Sample ranking:")
    for idx, result in enumerate(ranked_samples, start=1):
        print(f"{idx}. {result.label}: avg_last_5_tok_per_sec={result.avg_last_five_tok_per_sec}")

    if args.promote_top <= 0:
        print(f"artifacts: {output_dir}")
        return 0

    if not profile.allow_full_run and not args.force_full_run:
        print("Thermal profile does not recommend full runs right now; skipping promotion stage.")
        print(f"artifacts: {output_dir}")
        return 0

    top_results = ranked_samples[: args.promote_top]
    full_variants = []
    for result in top_results:
        variant = make_variant(result.spec)
        checkpoint_path = sample_checkpoint_paths.get(result.spec)
        env_overrides = dict(variant.env_overrides)
        if checkpoint_path is not None and checkpoint_path.exists():
            env_overrides["AUTORESEARCH_INIT_FROM"] = str(checkpoint_path)
        full_variants.append(
            Variant(
                spec=variant.spec,
                label=variant.label,
                script_path=variant.script_path,
                workdir=variant.workdir,
                note=variant.note,
                env_overrides=env_overrides,
                tempdir=variant.tempdir,
            )
        )

    full_results = [run_variant(variant, None, full_dir) for variant in full_variants]
    ranked_full = sorted(full_results, key=full_rank_key, reverse=True)
    full_payload = _write_payload(output_dir / "full_results.json", "full-run", profile, full_results)
    memory.append_many(benchmark_payload_to_entries(full_payload, output_dir / "full_results.json"))

    print("\nPromoted full-run ranking:")
    for idx, result in enumerate(ranked_full, start=1):
        print(f"{idx}. {result.label}: val_bpb={result.summary.get('val_bpb')} avg_last_5_tok_per_sec={result.avg_last_five_tok_per_sec}")

    print(f"artifacts: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
