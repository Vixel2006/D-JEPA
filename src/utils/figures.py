"""Small reusable plotting helpers for representation analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


def plot_tsne(
    representations: np.ndarray,
    labels: Sequence,
    path: str | Path,
    *,
    title: str = "t-SNE of representations",
    perplexity: float = 30.0,
    seed: int = 0,
) -> np.ndarray:
    """Save a 2D t-SNE scatter plot and return the projected coordinates."""
    representations = np.asarray(representations)
    labels = np.asarray(labels)
    if representations.ndim != 2 or len(representations) != len(labels):
        raise ValueError("Representations must be 2D and have one label per row")
    if len(representations) < 3:
        raise ValueError("t-SNE requires at least three examples")
    projection = TSNE(
        n_components=2,
        perplexity=min(float(perplexity), len(representations) - 1.0),
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(representations)

    fig, axis = plt.subplots(figsize=(7, 5.5))
    categories, encoded = np.unique(labels.astype(str), return_inverse=True)
    scatter = axis.scatter(
        projection[:, 0], projection[:, 1], c=encoded, cmap="tab10", s=14, alpha=0.8
    )
    handles, _ = scatter.legend_elements()
    axis.legend(handles, categories, title="Group", frameon=False, bbox_to_anchor=(1.02, 1))
    axis.set(title=title, xlabel="t-SNE 1", ylabel="t-SNE 2")
    _save(fig, path)
    return projection


def plot_heatmap(
    values: np.ndarray,
    row_labels: Sequence[str],
    column_labels: Sequence[str],
    path: str | Path,
    *,
    title: str = "",
    value_label: str = "Score",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """Save a labeled matrix heatmap, suitable for factor R² scores."""
    values = np.asarray(values)
    if values.shape != (len(row_labels), len(column_labels)):
        raise ValueError("Heatmap dimensions must match row and column labels")
    width = max(5.0, 0.8 * len(column_labels) + 2.0)
    height = max(3.5, 0.55 * len(row_labels) + 1.5)
    fig, axis = plt.subplots(figsize=(width, height))
    image = axis.imshow(values, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    axis.set_xticks(np.arange(len(column_labels)), labels=column_labels, rotation=35, ha="right")
    axis.set_yticks(np.arange(len(row_labels)), labels=row_labels)
    axis.set_title(title)
    colorbar = fig.colorbar(image, ax=axis)
    colorbar.set_label(value_label)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            label = "nan" if not np.isfinite(value) else f"{value:.2f}"
            axis.text(column, row, label, ha="center", va="center", color="white", fontsize=8)
    fig.tight_layout()
    _save(fig, path)


def plot_factor_r2_heatmap(
    scores: dict[str, dict[str, float]], path: str | Path, *, title: str = "Factor probe R²"
) -> None:
    """Plot the nested mapping returned by ``factor_r2_probes``."""
    rows = list(scores)
    columns = list(dict.fromkeys(factor for row in scores.values() for factor in row))
    values = np.asarray(
        [[scores[row].get(column, np.nan) for column in columns] for row in rows], dtype=float
    )
    plot_heatmap(values, rows, columns, path, title=title, value_label="Held-out R²")


def _save(figure, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)
