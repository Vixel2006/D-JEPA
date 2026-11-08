"""Representation diagnostics used by the paper figures.

Everything here is framework-light: the functions take CPU tensors and return
plain Python values so the results are easy to serialise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SpectralMetrics:
    name: str
    dim: int
    effective_rank: float
    participation_ratio: float
    stable_rank: float
    top1_variance: float
    condition_number: float


@dataclass(frozen=True)
class CKAMetrics:
    a: str
    b: str
    cka: float


@dataclass(frozen=True)
class AlignmentMetrics:
    a: str
    b: str
    centered_cosine: float


@dataclass(frozen=True)
class CalibrationMetrics:
    condition: str
    accuracy: float
    macro_f1: float
    ece: float
    nll: float


@dataclass
class Diagnostics:
    spectral: list[SpectralMetrics] = field(default_factory=list)
    cka: list[CKAMetrics] = field(default_factory=list)
    alignment: list[AlignmentMetrics] = field(default_factory=list)
    calibration: list[CalibrationMetrics] = field(default_factory=list)
    swaps: dict[str, dict[str, float]] = field(default_factory=dict)
    confusion: dict[str, list[list[int]]] = field(default_factory=dict)
    extras: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Spectra
# ---------------------------------------------------------------------------
def _covariance(z: torch.Tensor) -> torch.Tensor:
    z = z.double()
    centered = z - z.mean(dim=0, keepdim=True)
    return centered.T @ centered / max(1, z.size(0) - 1)


def covariance_eigenvalues(z: torch.Tensor) -> np.ndarray:
    """Eigenvalues of the feature covariance, descending."""

    eigenvalues = torch.linalg.eigvalsh(_covariance(z))
    return torch.flip(eigenvalues, dims=(0,)).clamp_min(0.0).numpy()


def _entropy_effective_rank(eigenvalues: np.ndarray) -> float:
    positive = eigenvalues[eigenvalues > 1e-12]
    if positive.size == 0:
        return 0.0
    probabilities = positive / positive.sum()
    return float(np.exp(-(probabilities * np.log(probabilities)).sum()))


def spectral_metrics(name: str, z: torch.Tensor) -> SpectralMetrics:
    eigenvalues = covariance_eigenvalues(z)
    positive = eigenvalues[eigenvalues > 1e-12]
    if positive.size == 0:
        return SpectralMetrics(
            name=name,
            dim=int(z.size(1)),
            effective_rank=0.0,
            participation_ratio=0.0,
            stable_rank=0.0,
            top1_variance=0.0,
            condition_number=float("inf"),
        )

    total = positive.sum()
    participation = (total**2) / float((positive**2).sum())
    stable_rank = total / float(positive.max())
    return SpectralMetrics(
        name=name,
        dim=int(z.size(1)),
        effective_rank=_entropy_effective_rank(eigenvalues),
        participation_ratio=float(participation),
        stable_rank=float(stable_rank),
        top1_variance=float(positive[0] / total),
        condition_number=float(positive[0] / positive[-1]),
    )


def compute_spectral_diagnostics(
    representations: Mapping[str, torch.Tensor],
    names: Optional[Sequence[str]] = None,
) -> list[SpectralMetrics]:
    order = names or tuple(representations)
    return [
        spectral_metrics(name, representations[name])
        for name in order
        if name in representations
    ]


# ---------------------------------------------------------------------------
# CKA / alignment
# ---------------------------------------------------------------------------
def linear_cka(a: torch.Tensor, b: torch.Tensor) -> float:
    """Centered linear kernel alignment between two representations."""

    a = a.double()
    b = b.double()
    a = a - a.mean(dim=0, keepdim=True)
    b = b - b.mean(dim=0, keepdim=True)
    cross = (a.T @ b).pow(2).sum()
    norm_a = (a.T @ a).pow(2).sum()
    norm_b = (b.T @ b).pow(2).sum()
    denominator = torch.sqrt(norm_a * norm_b)
    if denominator <= 1e-12:
        return 0.0
    return float((cross / denominator).clamp(0.0, 1.0))


def centered_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean per-sample cosine after removing each feature's mean."""

    a = a.double()
    b = b.double()
    a = a - a.mean(dim=0, keepdim=True)
    b = b - b.mean(dim=0, keepdim=True)
    return float(
        torch.nn.functional.cosine_similarity(a, b, dim=-1).mean()
    )


def compute_cka_pairs(
    representations: Mapping[str, torch.Tensor],
    pairs: Sequence[tuple[str, str]],
) -> list[CKAMetrics]:
    results: list[CKAMetrics] = []
    for a, b in pairs:
        if a not in representations or b not in representations:
            continue
        results.append(
            CKAMetrics(a=a, b=b, cka=linear_cka(representations[a], representations[b]))
        )
    return results


def compute_alignment_pairs(
    representations: Mapping[str, torch.Tensor],
    pairs: Sequence[tuple[str, str]],
) -> list[AlignmentMetrics]:
    results: list[AlignmentMetrics] = []
    for a, b in pairs:
        if a not in representations or b not in representations:
            continue
        results.append(
            AlignmentMetrics(
                a=a,
                b=b,
                centered_cosine=centered_cosine(
                    representations[a], representations[b]
                ),
            )
        )
    return results


# ---------------------------------------------------------------------------
# Classification diagnostics
# ---------------------------------------------------------------------------
def expected_calibration_error(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_bins: int = 15,
) -> float:
    probabilities = torch.softmax(logits.float(), dim=-1)
    confidence, predictions = probabilities.max(dim=-1)
    accuracy = (predictions == labels).float()

    boundaries = torch.linspace(0.0, 1.0, num_bins + 1)
    error = torch.tensor(0.0)
    total = labels.numel()
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        in_bin = (confidence > lower) & (confidence <= upper)
        count = int(in_bin.sum())
        if count == 0:
            continue
        bin_accuracy = accuracy[in_bin].mean()
        bin_confidence = confidence[in_bin].mean()
        error += (count / total) * (bin_confidence - bin_accuracy).abs()
    return float(error)


def macro_f1(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> float:
    f1_scores: list[float] = []
    for cls in range(num_classes):
        true_positive = int(((predictions == cls) & (labels == cls)).sum())
        false_positive = int(((predictions == cls) & (labels != cls)).sum())
        false_negative = int(((predictions != cls) & (labels == cls)).sum())
        denominator = 2 * true_positive + false_positive + false_negative
        if denominator == 0:
            continue
        f1_scores.append(2 * true_positive / denominator)
    return float(np.mean(f1_scores)) if f1_scores else 0.0


def confusion_matrix(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> list[list[int]]:
    matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for label, prediction in zip(labels.tolist(), predictions.tolist()):
        if 0 <= label < num_classes and 0 <= prediction < num_classes:
            matrix[label, prediction] += 1
    return matrix.tolist()


def compute_calibration(
    logits: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
    num_classes: int,
    conditions: Optional[Sequence[str]] = None,
) -> list[CalibrationMetrics]:
    names = conditions or tuple(logits)
    results: list[CalibrationMetrics] = []
    for name in names:
        if name not in logits:
            continue
        condition_logits = logits[name]
        predictions = condition_logits.argmax(dim=-1)
        accuracy = float((predictions == labels).float().mean() * 100.0)
        results.append(
            CalibrationMetrics(
                condition=name,
                accuracy=accuracy,
                macro_f1=macro_f1(predictions, labels, num_classes),
                ece=expected_calibration_error(condition_logits, labels),
                nll=float(
                    torch.nn.functional.cross_entropy(
                        condition_logits.float(), labels
                    )
                ),
            )
        )
    return results
