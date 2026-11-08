"""Publication-quality figure generation for the D-JEPA benchmark."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import torch

# Matplotlib is optional at import time so that the package remains usable in
# a minimal training environment.
try:  # pragma: no cover - import guard
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
except ImportError as exc:  # pragma: no cover - depends on environment
    plt = None
    GridSpec = None
    _MATPLOTLIB_IMPORT_ERROR = exc
else:
    _MATPLOTLIB_IMPORT_ERROR = None

from ..config import FigureConfig
from ..diagnostics import Diagnostics, covariance_eigenvalues
from ..evaluation.probes import (
    CATEGORIES,
    DJEPA_C,
    DJEPA_C_U1,
    DJEPA_C_U2,
    DJEPA_C_U1_U2,
    DJEPA_FULL,
    LEJEPA,
)

CATEGORY_LABELS = {
    "counting": "Counting",
    "attribute": "Attribute",
    "spatial_relation": "Spatial relation",
}
CATEGORY_COLORS = {
    "counting": "#2E5EAA",
    "attribute": "#C56A12",
    "spatial_relation": "#6D4AA5",
}
CONDITION_COLORS = {
    LEJEPA: "#5D6673",
    DJEPA_C: "#5D6673",
    DJEPA_C_U1: "#C56A12",
    DJEPA_C_U2: "#C56A12",
    DJEPA_C_U1_U2: "#6D4AA5",
    DJEPA_FULL: "#6D4AA5",
}


@dataclass
class FigureData:
    """Everything the figure suite needs, already aggregated."""

    accuracy: Mapping[str, Mapping[str, tuple[float, float]]]
    categories: Sequence[str]
    subspace_names: Sequence[str]
    embeddings: Mapping[str, np.ndarray]
    representations: Mapping[str, torch.Tensor]
    records: Sequence[dict]
    sample_images: Mapping[int, np.ndarray]
    diagnostics: Optional[Diagnostics]
    loss_history: Mapping[str, Sequence[float]]
    seeds: Sequence[int]
    answer_vocabulary: Sequence[str]


def shorten_condition(name: str) -> str:
    replacements = {
        "D-JEPA ": "",
        "D-JEPA": "D-JEPA",
    }
    for source, target in replacements.items():
        if name.startswith(source):
            return target + name[len(source):]
    return name


def _require_matplotlib() -> None:
    if plt is None:
        raise ImportError(
            "Figure generation requires matplotlib. Install it with "
            "`pip install matplotlib`."
        ) from _MATPLOTLIB_IMPORT_ERROR


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.linewidth": 0.6,
            "figure.dpi": 150,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )


def _save(fig, output_dir: Path, stem: str, config: FigureConfig) -> list[str]:
    paths = []
    png_path = output_dir / f"{stem}.png"
    fig.savefig(png_path, dpi=config.dpi)
    paths.append(str(png_path))
    if config.save_pdf:
        pdf_path = output_dir / f"{stem}.pdf"
        fig.savefig(pdf_path)
        paths.append(str(pdf_path))
    plt.close(fig)
    print(f"  -> {png_path}")
    return paths


def _ordered_conditions(data: FigureData) -> list[str]:
    preferred = [
        LEJEPA,
        DJEPA_C,
        DJEPA_C_U1,
        DJEPA_C_U2,
        DJEPA_C_U1_U2,
        DJEPA_FULL,
    ]
    ordered = [name for name in preferred if name in data.accuracy]
    ordered.extend(
        name for name in data.subspace_names if name not in ordered
    )
    return ordered


# ---------------------------------------------------------------------------
# 1. Accuracy heatmap
# ---------------------------------------------------------------------------
def figure_accuracy_heatmap(
    data: FigureData,
    output_dir: Path,
    config: FigureConfig,
) -> list[str]:
    conditions = _ordered_conditions(data)
    categories = list(data.categories) + ["overall"]
    labels = [CATEGORY_LABELS.get(c, c.title()) for c in data.categories]
    labels.append("Overall")

    matrix = np.zeros((len(conditions), len(categories)))
    for row, condition in enumerate(conditions):
        for column, category in enumerate(categories):
            matrix[row, column] = data.accuracy[condition][category][0]

    fig, ax = plt.subplots(figsize=(8.2, 0.42 * len(conditions) + 1.6))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(len(conditions)))
    ax.set_yticklabels([shorten_condition(c) for c in conditions])
    ax.set_title(
        f"Per-category VQA accuracy "
        f"({len(data.seeds)}-seed mean, %)",
        pad=10,
    )

    threshold = (matrix.min() + matrix.max()) / 2
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            std = data.accuracy[conditions[row]][categories[column]][1]
            text = f"{matrix[row, column]:.1f}"
            if std > 0 and categories[column] == "overall":
                text += f"\n±{std:.1f}"
            ax.text(
                column,
                row,
                text,
                ha="center",
                va="center",
                fontsize=7.5,
                fontweight="bold",
                color="white" if matrix[row, column] < threshold else "black",
            )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    colorbar.set_label("Accuracy (%)")
    fig.tight_layout()
    return _save(
        fig, output_dir, "fig01_probe_accuracy_heatmap", config
    )


# ---------------------------------------------------------------------------
# 2. Per-category bars
# ---------------------------------------------------------------------------
def figure_per_category_bars(
    data: FigureData,
    output_dir: Path,
    config: FigureConfig,
) -> list[str]:
    conditions = _ordered_conditions(data)
    x = np.arange(len(data.categories))
    width = min(0.8 / max(1, len(conditions)), 0.16)

    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    for index, condition in enumerate(conditions):
        values = [
            data.accuracy[condition][category][0]
            for category in data.categories
        ]
        errors = [
            data.accuracy[condition][category][1]
            for category in data.categories
        ]
        positions = x + (index - (len(conditions) - 1) / 2) * width
        ax.bar(
            positions,
            values,
            width,
            label=shorten_condition(condition),
            color=CONDITION_COLORS.get(condition, "#4C566A"),
            edgecolor="black",
            linewidth=0.35,
        )
        ax.errorbar(
            positions,
            values,
            yerr=errors,
            fmt="none",
            color="black",
            capsize=2,
            linewidth=0.6,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [CATEGORY_LABELS.get(c, c.title()) for c in data.categories]
    )
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(
        f"Per-category accuracy by condition "
        f"({len(data.seeds)}-seed mean ± std)"
    )
    ax.legend(loc="upper left", fontsize=7, ncol=3, framealpha=0.9)
    ax.grid(True, alpha=0.25, axis="y")
    fig.tight_layout()
    return _save(fig, output_dir, "fig02_per_category_bars", config)


# ---------------------------------------------------------------------------
# 3. Fibre contribution
# ---------------------------------------------------------------------------
def figure_fiber_contribution(
    data: FigureData,
    output_dir: Path,
    config: FigureConfig,
) -> list[str]:
    if DJEPA_C not in data.accuracy:
        return []
    additions = [
        ("+ u1", DJEPA_C_U1),
        ("+ u2", DJEPA_C_U2),
        ("+ u1,u2", DJEPA_C_U1_U2),
        ("+ u1,u2,s", DJEPA_FULL),
    ]
    additions = [
        (label, condition)
        for label, condition in additions
        if condition in data.accuracy
    ]
    if not additions:
        return []

    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    for category in data.categories:
        base = data.accuracy[DJEPA_C][category][0]
        deltas = [
            data.accuracy[condition][category][0] - base
            for _, condition in additions
        ]
        ax.plot(
            range(len(deltas)),
            deltas,
            marker="o",
            label=CATEGORY_LABELS.get(category, category.title()),
            color=CATEGORY_COLORS.get(category, "#333333"),
            linewidth=2,
            markersize=7,
        )
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(range(len(additions)))
    ax.set_xticklabels([label for label, _ in additions])
    ax.set_ylabel("Δ accuracy vs [c1, c2] (pp)")
    ax.set_title("Contribution of each private fibre by category")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25, axis="y")
    fig.tight_layout()
    return _save(fig, output_dir, "fig03_fiber_contribution", config)


# ---------------------------------------------------------------------------
# 4. t-SNE category separation
# ---------------------------------------------------------------------------
def _pca_2d(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T


def _embed_2d(values: np.ndarray, config: FigureConfig) -> Optional[np.ndarray]:
    if values is None or len(values) < 4:
        return None
    # t-SNE needs more points than the perplexity bound; use a deterministic
    # PCA view for tiny smoke runs so the figure suite stays complete.
    if len(values) < 12:
        return _pca_2d(values)
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        TSNE = None

    if TSNE is not None:
        perplexity = min(
            config.tsne_perplexity, max(5, (len(values) - 1) // 3)
        )
        try:
            return TSNE(
                n_components=2,
                random_state=config.seed,
                perplexity=perplexity,
                init="pca",
                max_iter=config.tsne_max_iter,
            ).fit_transform(values)
        except TypeError:  # older scikit-learn
            return TSNE(
                n_components=2,
                random_state=config.seed,
                perplexity=perplexity,
                init="pca",
                n_iter=config.tsne_max_iter,
            ).fit_transform(values)

    # Deterministic PCA fallback.
    return _pca_2d(values)


def figure_tsne(
    data: FigureData,
    output_dir: Path,
    config: FigureConfig,
) -> list[str]:
    keys = [
        key
        for key in ("c", "c+u1+u2", "full")
        if key in data.embeddings and len(data.embeddings[key])
    ]
    if not keys:
        return []

    categories = np.array(
        [record["category"] for record in data.records]
    )
    if len(categories) != len(data.embeddings[keys[0]]):
        categories = categories[: len(data.embeddings[keys[0]])]

    small_sample = any(
        len(np.asarray(data.embeddings[key])) < 12 for key in keys
    )
    projections = {
        key: _embed_2d(np.asarray(data.embeddings[key]), config)
        for key in keys
    }
    projections = {
        key: value for key, value in projections.items() if value is not None
    }
    if not projections:
        return []

    fig, axes = plt.subplots(
        1, len(projections), figsize=(5.6 * len(projections), 4.6)
    )
    if len(projections) == 1:
        axes = [axes]
    titles = {
        "c": "c only (shared code)",
        "c+u1+u2": "[c, u1, u2] (with private fibres)",
        "full": "[c, u1, u2, s] (full model)",
    }
    for axis, (key, values) in zip(axes, projections.items()):
        for category in data.categories:
            mask = categories == category
            if not mask.any():
                continue
            axis.scatter(
                values[mask, 0],
                values[mask, 1],
                label=CATEGORY_LABELS.get(category, category.title()),
                color=CATEGORY_COLORS.get(category, "#333333"),
                s=10,
                alpha=0.65,
                edgecolors="none",
            )
        axis.set_title(titles.get(key, key))
        axis.set_xticks([])
        axis.set_yticks([])
        axis.legend(loc="best", fontsize=7, framealpha=0.9)
    fig.suptitle(
        "Held-out CLEVR embeddings, colored by question type", fontsize=11
    )
    if small_sample:
        fig.text(
            0.5,
            0.005,
            "Small held-out set (<12 items): deterministic PCA fallback",
            ha="center",
            fontsize=8,
            color="#555555",
        )
    fig.tight_layout()
    return _save(fig, output_dir, "fig04_tsne_category_separation", config)


# ---------------------------------------------------------------------------
# 5. Prediction distribution / mode collapse
# ---------------------------------------------------------------------------
def figure_prediction_distribution(
    data: FigureData,
    output_dir: Path,
    config: FigureConfig,
) -> list[str]:
    from collections import Counter

    fig, axes = plt.subplots(
        1, len(data.categories), figsize=(4.6 * len(data.categories), 3.8)
    )
    if len(data.categories) == 1:
        axes = [axes]

    for axis, category in zip(axes, data.categories):
        records = [
            record for record in data.records
            if record["category"] == category
        ]
        if not records:
            axis.axis("off")
            continue
        ground_truth = Counter(
            record["ground_truth"] for record in records
        )
        lejepa = Counter(
            record.get("lejepa", "") for record in records
        )
        full = Counter(record.get("full", "") for record in records)

        candidates = set(ground_truth) | set(lejepa) | set(full)
        top = sorted(
            candidates, key=lambda answer: ground_truth.get(answer, 0),
            reverse=True,
        )[:8]
        x = np.arange(len(top))
        width = 0.27
        axis.bar(
            x - width,
            [ground_truth.get(answer, 0) for answer in top],
            width,
            label="Ground truth",
            color="#5D6673",
            edgecolor="black",
            linewidth=0.35,
        )
        axis.bar(
            x,
            [lejepa.get(answer, 0) for answer in top],
            width,
            label="LeJEPA",
            color="#2E5EAA",
            edgecolor="black",
            linewidth=0.35,
        )
        axis.bar(
            x + width,
            [full.get(answer, 0) for answer in top],
            width,
            label="D-JEPA Full",
            color="#6D4AA5",
            edgecolor="black",
            linewidth=0.35,
        )
        axis.set_xticks(x)
        axis.set_xticklabels(top, rotation=45, ha="right", fontsize=7)
        axis.set_title(
            CATEGORY_LABELS.get(category, category.title()),
            fontweight="bold",
        )
        axis.set_ylabel("Prediction count")
        axis.legend(fontsize=7, loc="best")
        axis.grid(True, alpha=0.25, axis="y")

    fig.suptitle("LeJEPA mode collapse vs. decomposition recovery")
    fig.tight_layout()
    return _save(
        fig, output_dir, "fig05_prediction_distribution", config
    )


# ---------------------------------------------------------------------------
# 6. Qualitative grid
# ---------------------------------------------------------------------------
def figure_qualitative_grid(
    data: FigureData,
    output_dir: Path,
    config: FigureConfig,
) -> list[str]:
    rows = []
    for category in data.categories:
        category_records = [
            record for record in data.records
            if record["category"] == category
        ]
        wins = [
            record for record in category_records if record.get("is_win")
        ]
        losses = [
            record for record in category_records if record.get("is_loss")
        ]
        fallback = category_records[:1]
        rows.append(
            (
                category,
                wins[0] if wins else (fallback[0] if fallback else None),
                losses[0] if losses else None,
            )
        )

    fig = plt.figure(figsize=(16.0, 2.6 * len(rows) + 0.6))
    grid = GridSpec(
        len(rows), 4, figure=fig, hspace=0.35, wspace=0.08,
        width_ratios=[1.0, 1.7, 1.0, 1.7],
    )
    for row, (category, win, loss) in enumerate(rows):
        for column, (record, headline) in enumerate(
            ((win, "Win"), (loss, "Failure mode"))
        ):
            if record is None:
                continue
            image_axis = fig.add_subplot(grid[row, column * 2])
            if record["idx"] in data.sample_images:
                image_axis.imshow(data.sample_images[record["idx"]])
            image_axis.axis("off")
            image_axis.set_title(
                f"{CATEGORY_LABELS.get(category, category.title())} · "
                f"{headline}",
                loc="left",
                fontsize=9,
                fontweight="bold",
            )

            text_axis = fig.add_subplot(grid[row, column * 2 + 1])
            text_axis.axis("off")
            ground_truth = record["ground_truth"]
            lines = [
                f"Q: {record['question']}",
                "",
                f"Ground truth: {ground_truth}",
                "",
                f"LeJEPA:    {record.get('lejepa', '—')}",
                f"c only:      {record.get('c_only', '—')}",
                f"c + u1:      {record.get('u1_only', '—')}",
                f"c + u2:      {record.get('u2_only', '—')}",
                f"c + u1 + u2: {record.get('both_u1_u2', '—')}",
                f"Full:        {record.get('full', '—')}",
            ]
            text_axis.text(
                0.01,
                0.95,
                "\n".join(lines),
                ha="left",
                va="top",
                fontsize=7.5,
                family="monospace",
                bbox=dict(
                    boxstyle="round,pad=0.5",
                    facecolor="#F4F6F8",
                    edgecolor="#5D6673",
                    alpha=0.95,
                ),
            )
    fig.suptitle("Qualitative CLEVR predictions per condition", fontsize=12)
    return _save(fig, output_dir, "fig06_qualitative_grid", config)


# ---------------------------------------------------------------------------
# 7. Spectral diagnostics
# ---------------------------------------------------------------------------
def figure_spectral_diagnostics(
    data: FigureData,
    output_dir: Path,
    config: FigureConfig,
) -> list[str]:
    names = [
        name for name in ("z1", "z2", "c1", "c2", "u1", "u2", "s")
        if name in data.representations
    ]
    if not names:
        return []

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for name in names:
        eigenvalues = covariance_eigenvalues(data.representations[name])
        positive = eigenvalues[eigenvalues > 1e-12]
        axes[0].loglog(
            np.arange(1, len(positive) + 1),
            positive,
            linewidth=1.3,
            label=name,
            alpha=0.85,
        )
    axes[0].set_xlabel("Component rank")
    axes[0].set_ylabel("Covariance eigenvalue")
    axes[0].set_title("Representation eigenspectrum")
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].grid(True, alpha=0.2, which="both")

    if data.diagnostics is not None:
        spectral = [
            metrics for metrics in data.diagnostics.spectral
            if metrics.name in names
        ]
        labels = [metrics.name for metrics in spectral]
        values = [metrics.effective_rank for metrics in spectral]
        dims = [metrics.dim for metrics in spectral]
        axes[1].bar(
            range(len(labels)),
            values,
            color="#2E5EAA",
            edgecolor="black",
            linewidth=0.35,
        )
        axes[1].set_xticks(range(len(labels)))
        axes[1].set_xticklabels(labels)
        axes[1].set_ylabel("Effective rank")
        axes[1].set_title("Effective rank (entropy of spectrum)")
        for index, (value, dim) in enumerate(zip(values, dims)):
            axes[1].text(
                index,
                value,
                f"{value:.0f}/{dim}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
        axes[1].grid(True, alpha=0.25, axis="y")
    else:
        axes[1].axis("off")

    fig.tight_layout()
    return _save(fig, output_dir, "fig07_spectral_diagnostics", config)


# ---------------------------------------------------------------------------
# 8. CKA matrix
# ---------------------------------------------------------------------------
def figure_cka_matrix(
    data: FigureData,
    output_dir: Path,
    config: FigureConfig,
) -> list[str]:
    if data.diagnostics is None or not data.diagnostics.cka:
        return []

    names = []
    for metrics in data.diagnostics.cka:
        if metrics.a not in names:
            names.append(metrics.a)
        if metrics.b not in names:
            names.append(metrics.b)
    index = {name: position for position, name in enumerate(names)}
    matrix = np.eye(len(names))
    for metrics in data.diagnostics.cka:
        matrix[index[metrics.a], index[metrics.b]] = metrics.cka
        matrix[index[metrics.b], index[metrics.a]] = metrics.cka

    fig, ax = plt.subplots(
        figsize=(0.65 * len(names) + 2.4, 0.65 * len(names) + 1.8)
    )
    image = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="magma")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    for row in range(len(names)):
        for column in range(len(names)):
            ax.text(
                column,
                row,
                f"{matrix[row, column]:.2f}",
                ha="center",
                va="center",
                fontsize=6.5,
                color="white" if matrix[row, column] < 0.55 else "black",
            )
    ax.set_title("Linear CKA between representation codes")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    colorbar.set_label("CKA")
    fig.tight_layout()
    return _save(fig, output_dir, "fig08_cka_matrix", config)


# ---------------------------------------------------------------------------
# 9. Causal swap
# ---------------------------------------------------------------------------
def figure_causal_swap(
    data: FigureData,
    output_dir: Path,
    config: FigureConfig,
) -> list[str]:
    if data.diagnostics is None or not data.diagnostics.swaps:
        return []

    fibres = list(data.diagnostics.swaps)
    categories = list(data.categories) + ["overall"]
    x = np.arange(len(categories))
    width = min(0.8 / max(1, len(fibres)), 0.24)

    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    for index, fibre in enumerate(fibres):
        values = [
            data.diagnostics.swaps[fibre].get(category, 0.0)
            for category in categories
        ]
        ax.bar(
            x + (index - (len(fibres) - 1) / 2) * width,
            values,
            width,
            label=f"swap {fibre}",
            edgecolor="black",
            linewidth=0.35,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [CATEGORY_LABELS.get(c, c.title()) for c in data.categories]
        + ["Overall"]
    )
    ax.set_ylabel("Prediction flip rate (%)")
    ax.set_title("Causal fibre-swap intervention")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25, axis="y")
    fig.tight_layout()
    return _save(fig, output_dir, "fig09_causal_swap", config)


# ---------------------------------------------------------------------------
# 10. PID decomposition
# ---------------------------------------------------------------------------
def figure_pid_decomposition(
    data: FigureData,
    output_dir: Path,
    config: FigureConfig,
    pid_rows,
) -> list[str]:
    if not pid_rows:
        return []

    codes = [row.code for row in pid_rows]
    unique_x1 = np.array([row.unique_x1 for row in pid_rows])
    unique_x2 = np.array([row.unique_x2 for row in pid_rows])
    redundancy = np.array([row.redundancy for row in pid_rows])
    synergy = np.array([row.synergy for row in pid_rows])
    x = np.arange(len(codes))

    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    ax.bar(
        x, redundancy, 0.55, label="Redundancy",
        color="#5D6673", edgecolor="black", linewidth=0.35,
    )
    ax.bar(
        x, unique_x1, 0.55, bottom=redundancy, label="Unique to vision (u1)",
        color="#2E5EAA", edgecolor="black", linewidth=0.35,
    )
    ax.bar(
        x, unique_x2, 0.55, bottom=redundancy + unique_x1,
        label="Unique to text (u2)", color="#C56A12",
        edgecolor="black", linewidth=0.35,
    )
    ax.bar(
        x, synergy, 0.55, bottom=redundancy + unique_x1 + unique_x2,
        label="Synergy", color="#6D4AA5",
        edgecolor="black", linewidth=0.35,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(codes, rotation=15, ha="right")
    ax.set_ylabel("Kernel-ridge R² contribution")
    ax.set_title("Flow-PID decomposition of every D-JEPA code")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.25, axis="y")
    fig.tight_layout()
    return _save(fig, output_dir, "fig10_pid_decomposition", config)


# ---------------------------------------------------------------------------
# 11. Training curves
# ---------------------------------------------------------------------------
def figure_training_curves(
    data: FigureData,
    output_dir: Path,
    config: FigureConfig,
) -> list[str]:
    history = data.loss_history
    if not history or not history.get("step"):
        return []

    steps = history["step"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    axes[0].plot(steps, history["loss_lejepa"], label="LeJEPA total")
    axes[0].plot(steps, history["loss_djepa"], label="D-JEPA total")
    axes[0].set_xlabel("Pretraining step")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Optimisation curves")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.25)

    for key, label in (
        ("djepa_cross", "cross-modal alignment"),
        ("djepa_decorrelation", "fibre decorrelation"),
        ("djepa_vision_recon", "masked vision reconstruction"),
        ("djepa_text_recon", "masked text reconstruction"),
        ("djepa_synergy", "synergy SIGReg"),
    ):
        if key in history:
            axes[1].plot(steps, history[key], label=label)
    axes[1].set_xlabel("Pretraining step")
    axes[1].set_ylabel("Loss term")
    axes[1].set_title("D-JEPA loss decomposition")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    return _save(fig, output_dir, "fig11_training_curves", config)


# ---------------------------------------------------------------------------
# 12. Calibration and confusion
# ---------------------------------------------------------------------------
def figure_calibration_confusion(
    data: FigureData,
    output_dir: Path,
    config: FigureConfig,
) -> list[str]:
    if data.diagnostics is None or not data.diagnostics.calibration:
        return []

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    rows = data.diagnostics.calibration
    labels = [shorten_condition(row.condition) for row in rows]
    positions = np.arange(len(rows))
    axes[0].bar(
        positions,
        [row.ece for row in rows],
        color="#2E5EAA",
        edgecolor="black",
        linewidth=0.35,
    )
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    axes[0].set_ylabel("Expected calibration error")
    axes[0].set_title("Calibration error by condition")
    axes[0].grid(True, alpha=0.25, axis="y")
    for position, row in zip(positions, rows):
        axes[0].text(
            position,
            row.ece,
            f"{row.ece:.2f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    for axis, condition in zip(axes[1:], (LEJEPA, DJEPA_FULL)):
        if condition not in data.diagnostics.confusion:
            axis.axis("off")
            continue
        matrix = np.asarray(data.diagnostics.confusion[condition], dtype=float)
        if matrix.sum() == 0:
            axis.axis("off")
            continue
        matrix = matrix / np.clip(matrix.sum(axis=1, keepdims=True), 1, None)
        axis.imshow(matrix, cmap="Blues", vmin=0.0, vmax=1.0)
        axis.set_title(shorten_condition(condition))
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("True class")
        axis.set_xticks([])
        axis.set_yticks([])

    fig.suptitle("Calibration and confusion structure")
    fig.tight_layout()
    return _save(fig, output_dir, "fig12_calibration_confusion", config)


# ---------------------------------------------------------------------------
# 13. Alignment summary
# ---------------------------------------------------------------------------
def figure_alignment_summary(
    data: FigureData,
    output_dir: Path,
    config: FigureConfig,
) -> list[str]:
    if data.diagnostics is None or not data.diagnostics.alignment:
        return []

    labels = [
        f"{metrics.a}–{metrics.b}"
        for metrics in data.diagnostics.alignment
    ]
    values = [metrics.centered_cosine for metrics in data.diagnostics.alignment]
    order = np.argsort(values)
    labels = [labels[index] for index in order]
    values = [values[index] for index in order]

    fig, ax = plt.subplots(figsize=(8.0, 0.4 * len(labels) + 1.8))
    colors = ["#2E5EAA" if value >= 0 else "#C56A12" for value in values]
    ax.barh(range(len(labels)), values, color=colors, edgecolor="black",
            linewidth=0.35)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Centered cosine similarity")
    ax.set_title("Cross-modal alignment after mean removal")
    for index, value in enumerate(values):
        ax.text(
            value,
            index,
            f" {value:.3f}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=7,
        )
    ax.grid(True, alpha=0.25, axis="x")
    fig.tight_layout()
    return _save(fig, output_dir, "fig13_alignment_summary", config)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def generate_all_figures(
    data: FigureData,
    output_dir: Path,
    config: FigureConfig = FigureConfig(),
    pid_rows: Optional[Sequence] = None,
) -> list[str]:
    """Generate the complete figure suite and return the written paths."""

    if not config.enabled:
        return []
    _require_matplotlib()
    _setup_style()
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("\n[*] Generating publication figures...")
    paths: list[str] = []
    generators = [
        lambda: figure_accuracy_heatmap(data, output_dir, config),
        lambda: figure_per_category_bars(data, output_dir, config),
        lambda: figure_fiber_contribution(data, output_dir, config),
        lambda: figure_tsne(data, output_dir, config),
        lambda: figure_prediction_distribution(data, output_dir, config),
        lambda: figure_qualitative_grid(data, output_dir, config),
        lambda: figure_spectral_diagnostics(data, output_dir, config),
        lambda: figure_cka_matrix(data, output_dir, config),
        lambda: figure_causal_swap(data, output_dir, config),
        lambda: figure_pid_decomposition(
            data, output_dir, config, pid_rows
        ),
        lambda: figure_training_curves(data, output_dir, config),
        lambda: figure_calibration_confusion(data, output_dir, config),
        lambda: figure_alignment_summary(data, output_dir, config),
    ]
    for generator in generators:
        try:
            paths.extend(generator())
        except Exception as exc:  # pragma: no cover - defensive
            print(f"  [!] Figure skipped: {exc}")
    return paths
