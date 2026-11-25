"""Factor probes and representation diagnostics for held-out evaluation."""

from __future__ import annotations

import numpy as np

from ..utils.metrics import factor_r2_probes, representation_diagnostics


def evaluate_representations(
    representations: dict[str, np.ndarray],
    factors: dict[str, np.ndarray],
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    *,
    regularization: float = 1.0,
) -> dict[str, dict]:
    """Compute held-out factor R² probes and collapse diagnostics per block."""
    r2 = factor_r2_probes(
        representations,
        factors,
        train_indices,
        test_indices,
        regularization=regularization,
    )
    diagnostics = {
        name: representation_diagnostics(values[test_indices])
        for name, values in representations.items()
    }
    return {"r2": r2, "diagnostics": diagnostics}
