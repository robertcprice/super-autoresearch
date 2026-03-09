"""
Checkpoint save/load helpers for experiment reuse.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import torch


@dataclass
class LoadReport:
    path: str
    loaded_tensors: int
    skipped_tensors: int
    total_candidate_tensors: int


def _load_payload(path: str | Path) -> Any:
    return torch.load(Path(path), map_location="cpu")


def _extract_model_state(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict) and "model" in payload:
        return payload["model"]
    if isinstance(payload, dict):
        return payload
    raise TypeError("Unsupported checkpoint payload")


def warm_start_model(model: torch.nn.Module, checkpoint_path: str | Path) -> LoadReport:
    payload = _load_payload(checkpoint_path)
    source_state = _extract_model_state(payload)
    target_state = model.state_dict()

    matched: dict[str, torch.Tensor] = {}
    skipped = 0
    for name, tensor in source_state.items():
        if name not in target_state:
            skipped += 1
            continue
        if tuple(target_state[name].shape) != tuple(tensor.shape):
            skipped += 1
            continue
        matched[name] = tensor

    target_state.update(matched)
    model.load_state_dict(target_state, strict=False)
    return LoadReport(
        path=str(checkpoint_path),
        loaded_tensors=len(matched),
        skipped_tensors=skipped,
        total_candidate_tensors=len(source_state),
    )


def save_training_checkpoint(
    checkpoint_path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    config: dict[str, Any] | None = None,
    summary: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> str:
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "saved_at": datetime.utcnow().isoformat() + "Z",
        "model": model.state_dict(),
        "config": config or {},
        "summary": summary or {},
        "extra": extra or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()

    with NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
        temp_path = Path(handle.name)
    try:
        torch.save(payload, temp_path)
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
    return str(path)
