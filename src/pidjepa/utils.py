"""Reproducibility, formatting, and serialization helpers.

The modules in this package are intentionally small and single-purpose; this
one contains the few helpers shared by training, evaluation, and reporting.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import torch

#: Natural logarithm of 2, used to convert nats to bits.
LN_2 = math.log(2.0)


def set_seed(seed: int) -> None:
    """Seed every random number generator used by the benchmark."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_mean_std(values: Iterable[float | None], digits: int = 2) -> str:
    """Format a sequence as ``mean ± sample_std``, ignoring missing values."""
    clean = [float(value) for value in values if value is not None and np.isfinite(value)]
    if not clean:
        return "n/a"
    array = np.asarray(clean, dtype=float)
    std = array.std(ddof=1) if len(array) > 1 else 0.0
    return f"{array.mean():.{digits}f} ± {std:.{digits}f}"


def make_json_serializable(value: Any) -> Any:
    """Recursively convert numpy/torch values into JSON-compatible objects."""
    if isinstance(value, Mapping):
        return {str(key): make_json_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_serializable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.tolist()
    return value
