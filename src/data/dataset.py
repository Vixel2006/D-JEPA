"""Small in-memory dataset format for multimodal DJEPA training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class PreparedDataset:
    """Context/target views, continuous factors, and held-out row indices."""

    context: dict[str, np.ndarray]
    target: dict[str, np.ndarray]
    factors: dict[str, np.ndarray]
    train_indices: np.ndarray
    test_indices: np.ndarray

    @property
    def modalities(self) -> tuple[str, ...]:
        return tuple(self.context)


def make_split(n_samples: int, seed: int, train_fraction: float = 0.8):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)
    split = int(n_samples * train_fraction)
    return indices[:split], indices[split:]


def load_npz_dataset(
    path: str | Path,
    modalities: tuple[str, ...] | None = None,
    seed: int = 0,
    train_fraction: float = 0.8,
) -> PreparedDataset:
    """Load ``context_<modality>``, ``target_<modality>``, and ``factor_*`` arrays.

    Optional ``train_indices`` and ``test_indices`` arrays in the archive take
    precedence over a generated split.
    """
    with np.load(path, allow_pickle=False) as archive:
        names = modalities or tuple(
            key.removeprefix("context_")
            for key in archive.files
            if key.startswith("context_")
        )
        if not names:
            raise ValueError("NPZ must include context_<modality> arrays")
        context, target = {}, {}
        for name in names:
            context_key, target_key = f"context_{name}", f"target_{name}"
            if context_key not in archive or target_key not in archive:
                raise ValueError(f"NPZ is missing {context_key!r} or {target_key!r}")
            context[name] = np.asarray(archive[context_key])
            target[name] = np.asarray(archive[target_key])
        n_samples = len(next(iter(context.values())))
        for name in names:
            if len(context[name]) != n_samples or len(target[name]) != n_samples:
                raise ValueError("All modality views must have the same number of rows")
        factors = {
            key.removeprefix("factor_"): np.asarray(archive[key])
            for key in archive.files
            if key.startswith("factor_")
        }
        if "train_indices" in archive and "test_indices" in archive:
            train_indices = np.asarray(archive["train_indices"], dtype=np.int32)
            test_indices = np.asarray(archive["test_indices"], dtype=np.int32)
        else:
            train_indices, test_indices = make_split(n_samples, seed, train_fraction)
    return PreparedDataset(context, target, factors, train_indices, test_indices)
