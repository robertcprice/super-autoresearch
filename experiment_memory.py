"""
Simple experiment memory store for benchmark and scheduler results.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


TOKEN_RE = re.compile(r"[a-z0-9_]+")
REPO_ROOT = Path(__file__).resolve().parent


@dataclass
class ExperimentMemoryEntry:
    dedupe_key: str
    created_at: str
    source: str
    label: str
    spec: str
    note: str
    mode: str
    artifacts: dict[str, str]
    metrics: dict[str, Any]
    text: str


def default_memory_path() -> Path:
    return REPO_ROOT / "experiment_memory.jsonl"


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def benchmark_payload_to_entries(payload: dict[str, Any], payload_path: str | Path) -> list[ExperimentMemoryEntry]:
    payload_path = Path(payload_path)
    created_at = payload.get("generated_at", "")
    mode = payload.get("mode", "unknown")
    machine = payload.get("machine", {})
    entries: list[ExperimentMemoryEntry] = []

    for result in payload.get("results", []):
        metrics = dict(result.get("summary", {}))
        if result.get("avg_last_five_tok_per_sec") is not None:
            metrics["avg_last_five_tok_per_sec"] = result["avg_last_five_tok_per_sec"]
        text_parts = [
            result.get("label", ""),
            result.get("spec", ""),
            result.get("note", ""),
            mode,
            machine.get("cpu", ""),
            " ".join(f"{key} {value}" for key, value in metrics.items()),
        ]
        entries.append(
            ExperimentMemoryEntry(
                dedupe_key=f"{payload_path.resolve()}::{result.get('label', '')}",
                created_at=created_at,
                source="benchmark",
                label=result.get("label", ""),
                spec=result.get("spec", ""),
                note=result.get("note", ""),
                mode=mode,
                artifacts={
                    "payload": str(payload_path.resolve()),
                    "log": result.get("log_path", ""),
                },
                metrics=metrics,
                text=" ".join(part for part in text_parts if part),
            )
        )
    return entries


class ExperimentMemory:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else default_memory_path()

    def load(self) -> list[ExperimentMemoryEntry]:
        if not self.path.exists():
            return []
        entries = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entries.append(ExperimentMemoryEntry(**json.loads(line)))
        return entries

    def append_many(self, entries: list[ExperimentMemoryEntry]) -> int:
        existing_keys = {entry.dedupe_key for entry in self.load()}
        new_entries = [entry for entry in entries if entry.dedupe_key not in existing_keys]
        if not new_entries:
            return 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            for entry in new_entries:
                handle.write(json.dumps(asdict(entry)) + "\n")
        return len(new_entries)

    def query(self, text: str, limit: int = 5) -> list[tuple[float, ExperimentMemoryEntry]]:
        query_tokens = set(tokenize(text))
        scored: list[tuple[float, ExperimentMemoryEntry]] = []
        for entry in self.load():
            entry_tokens = set(tokenize(entry.text))
            if not entry_tokens:
                continue
            overlap = len(query_tokens & entry_tokens)
            if overlap == 0:
                continue
            score = overlap / len(query_tokens | entry_tokens)
            scored.append((score, entry))
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[:limit]


def _main() -> int:
    parser = argparse.ArgumentParser(description="Experiment memory for benchmark results.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Ingest a benchmark results.json file.")
    ingest_parser.add_argument("payload", type=Path)
    ingest_parser.add_argument("--memory-path", type=Path, default=default_memory_path())

    query_parser = subparsers.add_parser("query", help="Query the memory store.")
    query_parser.add_argument("text")
    query_parser.add_argument("--limit", type=int, default=5)
    query_parser.add_argument("--memory-path", type=Path, default=default_memory_path())

    args = parser.parse_args()

    if args.command == "ingest":
        payload = json.loads(args.payload.read_text(encoding="utf-8"))
        store = ExperimentMemory(args.memory_path)
        added = store.append_many(benchmark_payload_to_entries(payload, args.payload))
        print(f"memory_path: {store.path}")
        print(f"entries_added: {added}")
        return 0

    store = ExperimentMemory(args.memory_path)
    for score, entry in store.query(args.text, limit=args.limit):
        print(f"{score:.3f} {entry.label} {entry.spec}")
        if entry.metrics:
            print(f"  metrics: {entry.metrics}")
        if entry.artifacts:
            print(f"  artifacts: {entry.artifacts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
