"""Typed result containers shared by evaluation and reporting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .diagnostics import Diagnostics


@dataclass(frozen=True)
class SeedMetrics:
    accuracy: dict[str, dict[str, float]]
    flip_rate: float
    procrustes_lejepa: float
    procrustes_djepa: float


@dataclass(frozen=True)
class PIDRow:
    code: str
    r2_x1: float
    r2_x2: float
    r2_joint: float
    unique_x1: float
    unique_x2: float
    redundancy: float
    synergy: float
    dominant: str


@dataclass(frozen=True)
class CaseStudy:
    category: str
    kind: str
    question: str
    ground_truth: str
    lejepa_prediction: str
    djepa_prediction: str
    scene_id: int
    caption: str


@dataclass
class PipelineResults:
    seeds: tuple[int, ...]
    subspace_names: list[str]
    categories: list[str]
    seed_metrics: list[SeedMetrics] = field(default_factory=list)
    pid_rows: list[PIDRow] = field(default_factory=list)
    case_studies: dict[str, dict[str, Optional[CaseStudy]]] = field(
        default_factory=dict
    )
    diagnostics: Optional[Diagnostics] = None
    embeddings: dict[str, np.ndarray] = field(default_factory=dict)
    records: list[dict[str, Any]] = field(default_factory=list)
    loss_history: dict[str, list[float]] = field(default_factory=dict)
    sample_images: dict[int, Any] = field(default_factory=dict)
    figure_paths: list[str] = field(default_factory=list)
