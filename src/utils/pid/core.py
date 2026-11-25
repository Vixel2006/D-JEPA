"""Gaussian information and preprocessing helpers for FlowPID."""

from __future__ import annotations

import jax
import numpy as np


def minimum_mutual_information_pid(i_first: float, i_second: float, i_joint: float) -> dict[str, float]:
    """Decompose two source-target mutual informations into MMI PID atoms."""
    i_first, i_second = max(i_first, 0.0), max(i_second, 0.0)
    i_joint = max(i_joint, i_first, i_second)
    redundancy = min(i_first, i_second)
    return {
        "I1": i_first,
        "I2": i_second,
        "I12": i_joint,
        "R": redundancy,
        "U1": i_first - redundancy,
        "U2": i_second - redundancy,
        "S": i_joint - max(i_first, i_second),
    }


def whiten_and_reduce(
    train: np.ndarray, test: np.ndarray, n_components: int
) -> tuple[np.ndarray, np.ndarray]:
    """Whiten using train statistics and retain the leading principal components."""
    train, test = np.asarray(train, np.float64), np.asarray(test, np.float64)
    if train.ndim != 2 or test.ndim != 2 or train.shape[1] != test.shape[1]:
        raise ValueError("Train and test features must be 2D arrays with matching dimensions")
    mean = train.mean(axis=0)
    centered = train - mean
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1][:n_components]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    keep = eigenvalues > 1e-8
    projection = eigenvectors[:, keep] / np.sqrt(eigenvalues[keep])
    return ((train - mean) @ projection).astype(np.float32), ((test - mean) @ projection).astype(
        np.float32
    )


def quantile_gaussianize(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Map values through the reference empirical CDF to normal scores."""
    reference = np.asarray(reference, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if reference.ndim == 1:
        reference = reference[:, None]
        values = values[:, None]
    transformed = np.empty_like(values, dtype=np.float32)
    for column in range(reference.shape[1]):
        sorted_reference = np.sort(reference[:, column])
        ranks = np.searchsorted(sorted_reference, values[:, column], side="left")
        quantiles = np.clip(
            (ranks + 0.5) / (len(sorted_reference) + 1.0), 1e-5, 1.0 - 1e-5
        )
        transformed[:, column] = np.asarray(jax.scipy.special.ndtri(quantiles))
    return transformed[:, 0] if transformed.shape[1] == 1 else transformed


def gaussian_mutual_information(
    covariance: np.ndarray, indices_a: list[int], indices_b: list[int]
) -> float:
    """Gaussian mutual information in bits, calculated from covariance blocks."""
    covariance = np.asarray(covariance, dtype=np.float64)

    def logdet(indices: list[int]) -> float:
        sign, value = np.linalg.slogdet(covariance[np.ix_(indices, indices)])
        return float(value) if sign > 0 else float("-inf")

    return 0.5 * (logdet(indices_a) + logdet(indices_b) - logdet(indices_a + indices_b)) / np.log(2)
