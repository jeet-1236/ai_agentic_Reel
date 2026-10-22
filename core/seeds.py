"""Deterministic per-stage seeds: same (seed, stage, item, loop) -> same seed, different loop -> new seed."""
from __future__ import annotations

import hashlib
import random


def stage_seed(seed: int, stage: str, item: int | str = 0, loop: int = 0) -> int:
    h = hashlib.sha256(f"{seed}|{stage}|{item}|{loop}".encode()).hexdigest()
    return int(h[:8], 16)  # fits int32-unsigned; llama.cpp/torch accept it


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32))
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass
