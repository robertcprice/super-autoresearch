"""
Model presets for M4 Pro with 24GB unified memory.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any

# Model presets optimized for M4 Pro
MODEL_PRESETS: Dict[str, Dict[str, Any]] = {
    "tiny": {
        "depth": 4,
        "batch_size": 16,
        "aspect_ratio": 64,
        "use_checkpointing": False,
        "description": "Safe baseline, fast iteration"
    },
    "small": {
        "depth": 4,
        "batch_size": 24,
        "aspect_ratio": 96,
        "use_checkpointing": False,
        "description": "Wider model for more capacity"
    },
    "medium": {
        "depth": 6,
        "batch_size": 12,
        "aspect_ratio": 64,
        "use_checkpointing": False,
        "description": "Deeper model for better quality"
    },
    "large": {
        "depth": 8,
        "batch_size": 8,
        "aspect_ratio": 64,
        "use_checkpointing": True,
        "description": "Deep model with memory optimization"
    },
}


def get_preset(name: str) -> Dict[str, Any]:
    """Get a preset by name."""
    if name not in MODEL_PRESETS:
        raise ValueError(f"Unknown preset: {name}. Available: {list(MODEL_PRESETS.keys())}")
    return MODEL_PRESETS[name].copy()


def list_presets() -> list:
    """List available presets."""
    return list(MODEL_PRESETS.keys())
