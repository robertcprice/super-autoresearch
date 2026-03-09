#!/usr/bin/env python3
"""
Benchmark the real training entrypoint against alternate variants.

By default this compares the current working tree against `origin/master`
for a 30-second throughput sample. Use `--full-run` to let each variant
complete its full 5-minute training + eval cycle.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from experiment_memory import ExperimentMemory, benchmark_payload_to_entries, default_memory_path


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_VARIANTS = ("current", "ref:origin/master")
TOK_PER_SEC_RE = re.compile(r"tok/sec:\s*([0-9,]+)")
STEP_DT_RE = re.compile(r"dt:\s*([0-9]+)ms")
SUMMARY_KEYS = {
    "checkpointing",
    "val_bpb",
    "time_budget_seconds",
    "training_seconds",
    "eval_seconds",
    "total_seconds",
    "peak_vram_mb",
    "mfu_percent",
    "total_tokens_M",
    "num_steps",
    "num_params_M",
    "depth",
    "device_batch_size",
    "eval_batch_size",
}


@dataclass
class Variant:
    spec: str
    label: str
    script_path: Path
    workdir: Path
    note: str
    env_overrides: dict[str, str] = field(default_factory=dict)
    tempdir: tempfile.TemporaryDirectory[str] | None = None


@dataclass
class BenchmarkResult:
    label: str
    spec: str
    script_path: str
    note: str
    launched: bool
    completed: bool
    returncode: int | None
    elapsed_seconds: float | None
    log_path: str
    tok_per_sec_samples: list[int]
    last_five_tok_per_sec: list[int]
    avg_last_five_tok_per_sec: float | None
    first_five_step_dt_ms: list[int]
    summary: dict[str, float | str]


def make_variant(spec: str) -> Variant:
    base_spec, env_overrides = split_spec_and_env(spec)

    if base_spec == "current":
        return Variant(
            spec=spec,
            label=label_with_env_suffix("current", env_overrides),
            script_path=REPO_ROOT / "train.py",
            workdir=REPO_ROOT,
            note=build_note("working tree", env_overrides),
            env_overrides=env_overrides,
        )

    if base_spec.startswith("path:"):
        rel_path = base_spec.split(":", 1)[1]
        script_path = (REPO_ROOT / rel_path).resolve()
        return Variant(
            spec=spec,
            label=label_with_env_suffix(script_path.stem, env_overrides),
            script_path=script_path,
            workdir=script_path.parent,
            note=build_note(f"path {rel_path}", env_overrides),
            env_overrides=env_overrides,
        )

    if base_spec.startswith("ref:"):
        ref = base_spec.split(":", 1)[1]
        tempdir = tempfile.TemporaryDirectory(prefix="autoresearch-bench-")
        workdir = Path(tempdir.name)
        script_path = workdir / "train.py"
        prepare_path = workdir / "prepare.py"
        for rel_path, destination in (("train.py", script_path), ("prepare.py", prepare_path)):
            result = subprocess.run(
                ["git", "show", f"{ref}:{rel_path}"],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip() or f"Unable to materialize {ref}:{rel_path}")
            destination.write_text(result.stdout, encoding="utf-8")
        return Variant(
            spec=spec,
            label=label_with_env_suffix(ref.replace("/", "_"), env_overrides),
            script_path=script_path,
            workdir=workdir,
            note=build_note(f"git ref {ref}", env_overrides),
            env_overrides=env_overrides,
            tempdir=tempdir,
        )

    raise ValueError(f"Unsupported variant spec: {spec}")


def split_spec_and_env(spec: str) -> tuple[str, dict[str, str]]:
    if "@" not in spec:
        return spec, {}
    base_spec, env_text = spec.split("@", 1)
    env_overrides = {}
    for item in env_text.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid env override `{item}` in variant spec `{spec}`")
        key, value = item.split("=", 1)
        env_overrides[key.strip()] = value.strip()
    return base_spec, env_overrides


def label_with_env_suffix(label: str, env_overrides: dict[str, str]) -> str:
    if not env_overrides:
        return label
    suffix_parts = []
    for key, value in sorted(env_overrides.items()):
        key_part = key.lower().replace("autoresearch_", "")
        key_part = re.sub(r"[^a-z0-9]+", "_", key_part).strip("_")
        value_part = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
        suffix_parts.append(f"{key_part}_{value_part}")
    return f"{label}__{'__'.join(suffix_parts)}"


def build_note(base_note: str, env_overrides: dict[str, str]) -> str:
    if not env_overrides:
        return base_note
    env_note = ", ".join(f"{key}={value}" for key, value in sorted(env_overrides.items()))
    return f"{base_note}; env: {env_note}"


def parse_summary(output: str) -> dict[str, float | str]:
    summary: dict[str, float | str] = {}
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key not in SUMMARY_KEYS:
            continue
        value = value.strip()
        try:
            summary[key] = float(value)
        except ValueError:
            summary[key] = value
    return summary


def build_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("AUTORESEARCH_BENCHMARK", "1")
    return env


def normalize_stream(stream: str | bytes | None) -> str:
    if stream is None:
        return ""
    if isinstance(stream, bytes):
        return stream.decode("utf-8", errors="replace")
    return stream


def run_variant(variant: Variant, duration: int | None, output_dir: Path) -> BenchmarkResult:
    env = build_env()
    env.update(variant.env_overrides)
    log_path = output_dir / f"{variant.label}.log"
    cmd = [sys.executable, str(variant.script_path)]
    launched = False
    completed = True
    returncode: int | None = None
    elapsed_seconds: float | None = None

    try:
        launched = True
        proc = subprocess.run(
            cmd,
            cwd=variant.workdir,
            env=env,
            capture_output=True,
            text=True,
            timeout=duration,
            check=False,
        )
        output = normalize_stream(proc.stdout) + normalize_stream(proc.stderr)
        returncode = proc.returncode
    except subprocess.TimeoutExpired as exc:
        completed = False
        output = normalize_stream(exc.stdout) + normalize_stream(exc.stderr)

    summary = parse_summary(output)
    if "total_seconds" in summary:
        try:
            elapsed_seconds = float(summary["total_seconds"])
        except (TypeError, ValueError):
            elapsed_seconds = None
    elif duration is not None:
        elapsed_seconds = float(duration)

    log_path.write_text(output, encoding="utf-8")
    tok_samples = [int(value.replace(",", "")) for value in TOK_PER_SEC_RE.findall(output)]
    dt_samples = [int(value) for value in STEP_DT_RE.findall(output)]
    last_five = tok_samples[-5:]
    avg_last_five = round(sum(last_five) / len(last_five), 2) if last_five else None

    return BenchmarkResult(
        label=variant.label,
        spec=variant.spec,
        script_path=str(variant.script_path),
        note=variant.note,
        launched=launched,
        completed=completed,
        returncode=returncode,
        elapsed_seconds=elapsed_seconds,
        log_path=str(log_path),
        tok_per_sec_samples=tok_samples,
        last_five_tok_per_sec=last_five,
        avg_last_five_tok_per_sec=avg_last_five,
        first_five_step_dt_ms=dt_samples[:5],
        summary=summary,
    )


def machine_info() -> dict[str, str]:
    info = {
        "python": sys.version.split()[0],
        "platform": sys.platform,
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["mps_available"] = str(torch.backends.mps.is_available())
        info["mps_built"] = str(torch.backends.mps.is_built())
    except Exception as exc:
        info["torch_error"] = str(exc)
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            info["cpu"] = result.stdout.strip()
    except Exception:
        pass
    return info


def print_result(result: BenchmarkResult) -> None:
    print(f"\n== {result.label} ==")
    print(f"spec: {result.spec}")
    print(f"note: {result.note}")
    print(f"launched: {result.launched}")
    print(f"completed: {result.completed}")
    print(f"returncode: {result.returncode}")
    print(f"log: {result.log_path}")
    if result.last_five_tok_per_sec:
        values = ", ".join(f"{value:,}" for value in result.last_five_tok_per_sec)
        print(f"last_5_tok_per_sec: {values}")
        print(f"avg_last_5_tok_per_sec: {result.avg_last_five_tok_per_sec:,.2f}")
    if result.first_five_step_dt_ms:
        values = ", ".join(str(value) for value in result.first_five_step_dt_ms)
        print(f"first_5_step_dt_ms: {values}")
    if result.summary:
        print("summary:")
        for key in sorted(result.summary):
            print(f"  {key}: {result.summary[key]}")


def compare(results: list[BenchmarkResult]) -> None:
    if len(results) < 2:
        return
    candidate = results[0]
    baseline = results[1]
    if candidate.avg_last_five_tok_per_sec is not None and baseline.avg_last_five_tok_per_sec is not None:
        delta = candidate.avg_last_five_tok_per_sec - baseline.avg_last_five_tok_per_sec
        pct = (delta / baseline.avg_last_five_tok_per_sec) * 100 if baseline.avg_last_five_tok_per_sec else 0.0
        print("\n== Comparison ==")
        print(
            f"{candidate.label} vs {baseline.label}: "
            f"{delta:+,.2f} tok/sec over the last-5 average ({pct:+.2f}%)"
        )
    if "val_bpb" in candidate.summary and "val_bpb" in baseline.summary:
        delta = float(candidate.summary["val_bpb"]) - float(baseline.summary["val_bpb"])
        print(
            f"{candidate.label} vs {baseline.label}: "
            f"{delta:+.6f} val_bpb (lower is better)"
        )
    if "total_tokens_M" in candidate.summary and "total_tokens_M" in baseline.summary:
        delta = float(candidate.summary["total_tokens_M"]) - float(baseline.summary["total_tokens_M"])
        print(
            f"{candidate.label} vs {baseline.label}: "
            f"{delta:+.2f}M total tokens"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark autoresearch training variants.")
    parser.add_argument(
        "--variant",
        action="append",
        dest="variants",
        help="Variant to run: `current`, `ref:<git-ref>`, or `path:<script.py>`.",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=30,
        help="Wall-clock timeout per variant in seconds. Ignored by --full-run.",
    )
    parser.add_argument(
        "--full-run",
        action="store_true",
        help="Let each variant finish naturally and collect final summary metrics.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for logs and JSON output. Defaults to benchmark_runs/<timestamp>/.",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (REPO_ROOT / "benchmark_runs" / timestamp)
    output_dir.mkdir(parents=True, exist_ok=True)

    variants = [make_variant(spec) for spec in (args.variants or list(DEFAULT_VARIANTS))]
    timeout = None if args.full_run else args.duration

    print("AUTORESEARCH BENCHMARK")
    print(f"output_dir: {output_dir}")
    print(f"mode: {'full-run' if args.full_run else f'{args.duration}s sample'}")
    for key, value in machine_info().items():
        print(f"{key}: {value}")

    results = [run_variant(variant, timeout, output_dir) for variant in variants]
    payload = {
        "generated_at": datetime.now().isoformat(),
        "mode": "full-run" if args.full_run else f"{args.duration}s sample",
        "machine": machine_info(),
        "results": [asdict(result) for result in results],
    }
    json_path = output_dir / "results.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    memory = ExperimentMemory(default_memory_path())
    added = memory.append_many(benchmark_payload_to_entries(payload, json_path))

    for result in results:
        print_result(result)
    compare(results)
    print(f"\njson: {json_path}")
    print(f"memory_path: {memory.path}")
    print(f"memory_entries_added: {added}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
