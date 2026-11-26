from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_factor_probes(scores, path, metric="r2"):
    """Plot factor-probe scores as a representation-by-factor heatmap."""
    rows = {}
    for name, value in scores.items():
        if not name.startswith("probe/") or not name.endswith(f"_{metric}"):
            continue
        representation, factor = name.removeprefix("probe/").rsplit("/", 1)
        factor = factor.removesuffix(f"_{metric}")
        rows.setdefault(representation, {})[factor] = value

    if not rows:
        raise ValueError(f"No probe scores found for metric {metric!r}.")

    representations = list(rows)
    factors = list(dict.fromkeys(factor for row in rows.values() for factor in row))
    values = np.array(
        [[rows[representation].get(factor, np.nan) for factor in factors]
         for representation in representations]
    )
    width = max(8, 1.2 * len(factors))
    height = max(4, 0.55 * len(representations))
    fig, ax = plt.subplots(figsize=(width, height), layout="constrained")
    sns.heatmap(
        values,
        xticklabels=factors,
        yticklabels=representations,
        annot=True,
        fmt=".2f",
        cmap="vlag" if metric == "r2" else "viridis",
        center=0 if metric == "r2" else None,
        vmin=-1 if metric == "r2" else 0,
        vmax=1,
        ax=ax,
    )
    ax.set_xlabel("Ground-truth factor")
    ax.set_ylabel("Representation")
    ax.set_title(f"Linear probe {metric.upper()}")
    _save(fig, path)


def plot_pid_atoms(pid_results, path):
    """Plot PID atoms as stacked bars, one bar per target variable.

    ``pid_results`` maps targets to dictionaries with keys ``redundant``,
    ``unique_vision``, ``unique_text``, and ``synergy``.
    """
    atoms = ("redundant", "unique_vision", "unique_text", "synergy")
    targets = list(pid_results)
    bottom = [0.0] * len(targets)
    fig, ax = plt.subplots(figsize=(max(7, len(targets) * 1.2), 5), layout="constrained")

    for atom in atoms:
        values = [pid_results[target].get(atom, 0.0) for target in targets]
        ax.bar(targets, values, bottom=bottom, label=atom.replace("_", " "))
        bottom = [base + value for base, value in zip(bottom, values)]

    ax.set_ylabel("Information (bits)")
    ax.set_xlabel("Target variable")
    ax.set_title("Partial information decomposition")
    ax.legend()
    _save(fig, path)


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
