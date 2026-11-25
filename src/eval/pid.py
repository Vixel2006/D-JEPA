"""Evaluation wrappers for FlowPID across source pairs and target factors."""

from __future__ import annotations

import numpy as np

from ..utils.pid.flow import estimate_flow_pid_pair


def evaluate_flow_pid(
    blocks: dict[str, np.ndarray],
    source_pairs: dict[str, tuple[list[str], list[str]]],
    targets: dict[str, np.ndarray],
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    *,
    seed: int = 0,
    pid_dim: int = 32,
    **flow_options,
) -> dict[str, dict[str, dict[str, float] | None]]:
    """Estimate FlowPID for each named source-pair and scalar target."""
    results: dict[str, dict[str, dict[str, float] | None]] = {}
    for pair_index, (pair_name, (first_keys, second_keys)) in enumerate(source_pairs.items()):
        results[pair_name] = {}
        for target_name, target in targets.items():
            target = np.asarray(target)
            columns = target[:, None] if target.ndim == 1 else target
            for column in range(columns.shape[1]):
                name = target_name if columns.shape[1] == 1 else f"{target_name}_{column}"
                results[pair_name][name] = estimate_flow_pid_pair(
                    blocks,
                    first_keys,
                    second_keys,
                    columns[:, column],
                    train_indices,
                    test_indices,
                    pid_dim=pid_dim,
                    seed=seed + pair_index + column,
                    **flow_options,
                )
    return results
