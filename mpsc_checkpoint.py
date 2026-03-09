"""
Gradient checkpointing for deeper models on MPS.
"""

import torch
from torch.utils.checkpoint import checkpoint
from typing import Optional

def checkpoint_wrapper(module, callable, *args, **kwargs):
    """Checkpoint wrapper for gradient memory savings."""
    return checkpoint(callable, *args, use_reentrant=False, **kwargs)


