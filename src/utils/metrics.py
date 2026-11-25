"""Held-out probe scores and representation diagnostics."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge


def ridge_r2(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    regularization: float = 1.0,
) -> float:
    """Fit a ridge probe on training data and return held-out R²."""
    x_train, x_test = np.asarray(x_train), np.asarray(x_test)
    y_train, y_test = np.asarray(y_train), np.asarray(y_test)
    if y_train.ndim == 1:
        y_train = y_train[:, None]
        y_test = y_test[:, None]
    mean = x_train.mean(axis=0)
    scale = np.maximum(x_train.std(axis=0), 1e-6)
    model = Ridge(alpha=regularization).fit((x_train - mean) / scale, y_train)
    prediction = model.predict((x_test - mean) / scale)
    total = np.sum(np.square(y_test - y_test.mean(axis=0)), axis=0)
    valid = total > 1e-12
    if not np.any(valid):
        return float("nan")
    residual = np.sum(np.square(y_test - prediction), axis=0)
    return float(np.mean(1.0 - residual[valid] / total[valid]))


def factor_r2_probes(
    representations: dict[str, np.ndarray],
    factors: dict[str, np.ndarray],
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    regularization: float = 1.0,
) -> dict[str, dict[str, float]]:
    """Return held-out representation-by-factor R² scores."""
    scores = {}
    for representation_name, representation in representations.items():
        scores[representation_name] = {}
        for factor_name, factor in factors.items():
            factor = np.asarray(factor)
            is_valid = (
                np.isfinite(factor).all(axis=-1)
                if factor.ndim > 1
                else np.isfinite(factor)
            )
            valid_train = train_indices[is_valid[train_indices]]
            valid_test = test_indices[is_valid[test_indices]]
            scores[representation_name][factor_name] = ridge_r2(
                representation[valid_train],
                factor[valid_train],
                representation[valid_test],
                factor[valid_test],
                regularization,
            )
    return scores


def linear_cka(first: np.ndarray, second: np.ndarray) -> float:
    """Linear centered kernel alignment between two representation matrices."""
    first, second = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    first = first - first.mean(axis=0)
    second = second - second.mean(axis=0)
    numerator = np.linalg.norm(first.T @ second) ** 2
    denominator = np.linalg.norm(first.T @ first) * np.linalg.norm(second.T @ second)
    return float(numerator / max(denominator, 1e-12))


def mutual_knn(first: np.ndarray, second: np.ndarray, k: int = 10) -> float:
    """Fraction of neighbours shared by two representations."""
    first, second = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    if len(first) != len(second) or len(first) < 2:
        raise ValueError("Inputs must have the same number of at least two examples")
    k = min(k, len(first) - 1)

    def neighbors(values: np.ndarray) -> np.ndarray:
        values = values - values.mean(axis=0)
        values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
        similarities = values @ values.T
        np.fill_diagonal(similarities, -np.inf)
        return np.argsort(-similarities, axis=1)[:, :k]

    first_neighbors, second_neighbors = neighbors(first), neighbors(second)
    return float(
        np.mean(
            [
                len(set(a).intersection(b)) / k
                for a, b in zip(first_neighbors, second_neighbors, strict=True)
            ]
        )
    )


def representation_diagnostics(values: np.ndarray) -> dict[str, float]:
    """Summarize feature spread and covariance effective rank."""
    values = np.asarray(values, dtype=np.float64)
    covariance = np.atleast_2d(np.cov(values, rowvar=False))
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
    squared_sum = np.square(eigenvalues).sum()
    return {
        "mean_std": float(values.std(axis=0).mean()),
        "effective_rank": float(eigenvalues.sum() ** 2 / max(squared_sum, 1e-18)),
        "dead_feature_fraction": float(np.mean(values.std(axis=0) < 1e-6)),
    }


def heldout_procrustes(
    first_train: np.ndarray,
    second_train: np.ndarray,
    first_test: np.ndarray,
    second_test: np.ndarray,
) -> float:
    """Fit an orthogonal map on train examples and return test cosine similarity."""
    first_train, second_train = np.asarray(first_train), np.asarray(second_train)
    first_test, second_test = np.asarray(first_test), np.asarray(second_test)
    first_mean, second_mean = first_train.mean(axis=0), second_train.mean(axis=0)
    u, _, vh = np.linalg.svd((second_train - second_mean).T @ (first_train - first_mean))
    rotation = u @ vh
    first_test = first_test - first_mean
    second_test = (second_test - second_mean) @ rotation
    numerator = np.sum(first_test * second_test, axis=-1)
    denominator = np.linalg.norm(first_test, axis=-1) * np.linalg.norm(second_test, axis=-1)
    return float(np.mean(numerator / np.maximum(denominator, 1e-12)))
