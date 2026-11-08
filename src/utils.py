"""Small runtime helpers shared across the package."""

from __future__ import annotations

import logging
import random
import time
from contextlib import nullcontext
from typing import Iterable, Iterator, Sequence, TypeVar

import numpy as np
import torch
from torch import nn

T = TypeVar("T")


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch with the same value."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def cycle(loader: Iterable[T]) -> Iterator[T]:
    """Yield batches from ``loader`` forever."""

    while True:
        yield from loader


def resolve_device(requested: str) -> torch.device:
    """Resolve ``auto`` to CUDA when available, otherwise CPU."""

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def amp_enabled(device: torch.device, requested: bool) -> bool:
    """Autocast/AMP is only active on CUDA and when explicitly requested."""

    return requested and device.type == "cuda"


def autocast_context(device: torch.device, enabled: bool):
    """Return an autocast context, or a no-op context when disabled."""

    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, enabled=True)


def make_grad_scaler(enabled: bool) -> torch.amp.GradScaler:
    """Create the GradScaler used by the original two-model training loop."""

    return torch.amp.GradScaler("cuda", enabled=enabled)


def trainable_parameters(
    parameters: Iterable[nn.Parameter],
) -> list[nn.Parameter]:
    """Filter out frozen parameters before constructing an optimiser."""

    return [parameter for parameter in parameters if parameter.requires_grad]


def configure_logging(level: str = "INFO") -> None:
    """Configure a simple, timestamped console logger."""

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )


def format_duration(seconds: float) -> str:
    """Human-readable duration such as ``1m 05s``."""

    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def progress_suffix(step: int, total: int, started: float) -> str:
    """Standard ``step/total | % | elapsed | ETA`` progress fragment."""

    elapsed = max(1e-9, time.perf_counter() - started)
    total = max(1, total)
    fraction = min(max(step / total, 0.0), 1.0)
    rate = step / elapsed
    remaining = (total - step) / rate if rate > 0 else 0.0
    return (
        f"{step:>5d}/{total:<5d} {fraction * 100:5.1f}% | "
        f"elapsed {format_duration(elapsed)} | ETA {format_duration(remaining)}"
    )


def as_tuple(values: Sequence[int] | int) -> tuple[int, ...]:
    """Normalise a scalar or sequence of seeds to a tuple of ints."""

    if isinstance(values, int):
        return (values,)
    return tuple(int(value) for value in values)
