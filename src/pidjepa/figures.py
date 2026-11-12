"""Publication figure suite."""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING

# Safe headless plotting configuration.
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

from pidjepa.models import ARM_CONFIGS

if TYPE_CHECKING:
    from pidjepa.data import Bundle


# The arm that trains cross-modal invariance only measures alignment between
# modalities; it is not a test of the Platonic Representation Hypothesis.
ARM_LABELS = {
    "raw": "Raw",
    "alignment": "Alignment",
    "alignment_rec": "Alignment + Rec",
    "djepa": "D-JEPA",
    "djepa_noindep": "D-JEPA (No-Indep)",
    "djepa_norec": "D-JEPA (No-Rec)",
}

# Reports produced before the rename still use these arm names. Normalizing
# them on load keeps old ``report.json`` files reproducible.
LEGACY_ARM_NAMES = {
    "platonic": "alignment",
    "platonic_rec": "alignment_rec",
}


def canonical_arm(name: str) -> str:
    return LEGACY_ARM_NAMES.get(name, name)


def arm_label(name: str) -> str:
    name = canonical_arm(name)
    return ARM_LABELS.get(name, name.replace("_", " ").title())


def canonical_report_key(key: str) -> str:
    """Rename arm tokens inside a report key (``platonic|z1``, ``djepa-platonic|x``)."""
    for old, new in LEGACY_ARM_NAMES.items():
        key = re.sub(rf"(?<![A-Za-z0-9_]){re.escape(old)}(?![A-Za-z0-9_])", new, key)
    return key


def normalize_report(summary: dict) -> dict:
    """Return a shallow copy of ``summary`` with legacy arm names canonicalised."""
    return {
        section: {canonical_report_key(key): value for key, value in value.items()}
        if isinstance(value, dict)
        else value
        for section, value in summary.items()
    }


def _ordered_arms(available: Sequence[str]) -> list[str]:
    available = set(available)
    known = ["raw"] + list(ARM_CONFIGS)
    ordered = [arm for arm in known if arm in available]
    return ordered + sorted(available - set(ordered))


def _infer_factors(summary: dict) -> list[str]:
    for value in summary.get("idmat", {}).values():
        if isinstance(value, dict) and value:
            return list(value.keys())
    return ["shared", "priv_v", "priv_t"]


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "figure.dpi": 200,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def _save_figure(fig, out_dir: str, filename: str) -> None:
    fig.savefig(os.path.join(out_dir, filename))
    plt.close(fig)
    print(f"  [*] Saved: {out_dir}/{filename}")


# =============================================================================
# Report-only figures (reproducible from report.json)
# =============================================================================
def plot_modality_gap(summary: dict, out_dir: str) -> None:
    """FIG 2: modality gap (synergy of the joint representation)."""
    if "gap" not in summary:
        return

    tasks = sorted({key.split("|")[1] for key in summary["gap"] if key.count("|") == 1})
    if not tasks:
        return
    task_name = "answer" if "answer" in tasks else ("shared" if "shared" in tasks else tasks[0])

    available = {key.split("|")[0] for key in summary["gap"] if key.endswith(f"|{task_name}")}
    gap_arms = _ordered_arms(available)
    if not gap_arms:
        return

    gaps = [np.mean(summary["gap"][f"{arm}|{task_name}"]) for arm in gap_arms]
    errors = [
        np.std(summary["gap"][f"{arm}|{task_name}"], ddof=1)
        if len(summary["gap"][f"{arm}|{task_name}"]) > 1
        else 0.0
        for arm in gap_arms
    ]

    fig, ax = plt.subplots(figsize=(6, 3.8))
    colors = ["#2E7D32" if gap >= 0 else "#C62828" for gap in gaps]
    ax.bar(
        range(len(gap_arms)),
        gaps,
        yerr=errors,
        capsize=4,
        color=colors,
        edgecolor="black",
        linewidth=0.6,
        width=0.55,
    )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Modality Gap (pp)\n[Joint - max(Vision, Text)]")
    ax.set_title("Multimodal Synergy vs. Modality Collapse", pad=10)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_xticks(range(len(gap_arms)))
    ax.set_xticklabels([arm_label(arm) for arm in gap_arms], rotation=25, ha="right")
    fig.tight_layout()
    _save_figure(fig, out_dir, "fig2_modality_gap.png")


def plot_factor_heatmap(summary: dict, out_dir: str, factors: Sequence[str] | None = None) -> None:
    """FIG 3: feature dissection heatmap (what did each block learn?)."""
    if "idmat" not in summary:
        return

    factors = list(factors) if factors is not None else _infer_factors(summary)
    panels = [
        ("alignment", ["z1", "z2"]),
        ("djepa", ["c1", "c2", "u1", "u2"]),
    ]
    panels = [
        (arm, blocks)
        for arm, blocks in panels
        if any(f"{arm}|{block}" in summary["idmat"] for block in blocks)
    ]
    if not panels:
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 4.2), squeeze=False)
    axes = axes[0]
    for ax, (arm, blocks) in zip(axes, panels):
        matrix = np.array(
            [
                [summary["idmat"].get(f"{arm}|{block}", {}).get(factor, 0.0) for factor in factors]
                for block in blocks
            ]
        )
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".2f",
            cmap="YlGnBu",
            cbar=(ax is axes[-1]),
            xticklabels=factors,
            yticklabels=blocks,
            vmin=-0.05,
            vmax=1.0,
            ax=ax,
            annot_kws={"fontsize": 9, "fontweight": "bold"},
        )
        ax.set_title(f"{arm_label(arm)} Subspaces $\\to$ Factors ($R^2$)", pad=8)
        ax.set_xlabel("Target Factors")

    fig.tight_layout()
    _save_figure(fig, out_dir, "fig3_factor_dissection_heatmap.png")


def plot_pid(summary: dict, field_name: str, filename: str, title: str, out_dir: str) -> None:
    """FIG 6/7: stacked PID information allocation (Probe-PID / Flow-PID)."""
    if not summary.get(field_name):
        return

    target_keys = sorted(
        {key.split("|")[2] for key in summary[field_name] if len(key.split("|")) >= 3}
    )
    if not target_keys:
        return

    probe_key = f"|stream|{target_keys[0]}"
    available = {
        key.split("|")[0]
        for key in summary[field_name]
        if probe_key in key or f"{probe_key}|all" in key
    }
    plotted_arms = _ordered_arms(available)
    if not plotted_arms:
        return

    fig, axes = plt.subplots(
        1, len(target_keys), figsize=(3.8 * len(target_keys), 3.8), sharey=True
    )
    axes = np.atleast_1d(axes)

    component_colors = [
        ("R", "Redundancy", "#5D6673"),
        ("U1", "Unq1 (Vision)", "#2E5EAA"),
        ("U2", "Unq2 (Text)", "#C56A12"),
        ("S", "Synergy", "#6D4AA5"),
    ]

    for ax, target_name in zip(axes, target_keys):
        bottoms = np.zeros(len(plotted_arms))
        for component_code, component_label, color in component_colors:
            values = []
            for arm in plotted_arms:
                data = (
                    summary[field_name].get(f"{arm}|stream|{target_name}")
                    or summary[field_name].get(f"{arm}|stream|{target_name}|all")
                    or {}
                )
                values.append(data.get(component_code, 0.0))
            ax.bar(
                range(len(plotted_arms)),
                values,
                bottom=bottoms,
                color=color,
                label=component_label,
                edgecolor="black",
                linewidth=0.4,
                width=0.6,
            )
            bottoms += np.array(values)

        ax.set_title(f"Target: {target_name.title()}", fontsize=10, pad=8)
        ax.set_xticks(range(len(plotted_arms)))
        ax.set_xticklabels(
            [arm_label(arm) for arm in plotted_arms], rotation=25, ha="right", fontsize=8
        )
        ax.grid(axis="y", linestyle=":", alpha=0.5)

    axes[0].set_ylabel("Mutual Information (bits)")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[-1].legend(handles, labels, fontsize=7.5, loc="upper right")
    fig.suptitle(title, fontsize=11, y=1.02)
    fig.tight_layout()
    _save_figure(fig, out_dir, filename)


def make_report_plots(summary: dict, out_dir: str, factors: Sequence[str] | None = None) -> None:
    """Regenerate every figure whose data is stored in ``report.json``.

    The latent-space figures (1/4/5) need per-example model outputs and are
    therefore produced by :func:`make_all_figures` during a run.
    """
    os.makedirs(out_dir, exist_ok=True)
    summary = normalize_report(summary)
    _configure_style()

    plot_modality_gap(summary, out_dir)
    plot_factor_heatmap(summary, out_dir, factors=factors)

    plot_pid(
        summary,
        "ppid",
        "fig6_probe_pid.png",
        "Probe-PID: Information Allocation Across Modality Streams",
        out_dir,
    )
    if summary.get("fpid"):
        plot_pid(
            summary,
            "fpid",
            "fig7_flow_pid.png",
            "Flow-PID: Continuous Latent Density Decomposition",
            out_dir,
        )


# =============================================================================
# Full figure suite (run-time figures)
# =============================================================================
def make_all_figures(summary: dict, bundle: Bundle, results: list[dict], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    summary = normalize_report(summary)
    _configure_style()

    arms = ["raw"] + list(ARM_CONFIGS)
    palette = sns.color_palette("muted", len(arms))
    arm_colors = dict(zip(arms, palette))

    # FIG 1: per-category accuracy breakdown with confidence intervals.
    if bundle.categories is not None and "per_cat_acc" in summary:
        categories = sorted(set(bundle.categories[bundle.split["test"].numpy()]))
        fig, ax = plt.subplots(figsize=(8, 4))
        bar_width = 0.8 / len(arms)
        positions = np.arange(len(categories))
        for index, arm in enumerate(arms):
            means = [
                np.mean(summary["per_cat_acc"][f"{arm}|{category}"]) for category in categories
            ]
            errors = [
                np.std(summary["per_cat_acc"][f"{arm}|{category}"], ddof=1)
                if len(summary["per_cat_acc"][f"{arm}|{category}"]) > 1
                else 0.0
                for category in categories
            ]
            ax.bar(
                positions + index * bar_width,
                means,
                bar_width,
                yerr=errors,
                capsize=3,
                label=arm_label(arm),
                color=arm_colors[arm],
                edgecolor="black",
                linewidth=0.5,
            )

        ax.set_xticks(positions + (len(arms) - 1) * bar_width / 2)
        ax.set_xticklabels([category.replace("_", " ").title() for category in categories])
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"CLEVR Reasoning Performance Across Arms ({len(results)}-Seed Mean ± Std)")
        ax.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=8, ncol=3)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        fig.tight_layout()
        _save_figure(fig, out_dir, "fig1_per_category_bars.png")

    # FIG 2: modality gap (synergy of the joint representation).
    plot_modality_gap(summary, out_dir)

    # FIG 3: feature dissection heatmap (what did each block learn?).
    plot_factor_heatmap(summary, out_dir, factors=list(bundle.factors))

    # FIG 4: probe block attribution.
    try:
        first_result = results[0]
        if "probes" in first_result and ("djepa", "joint", "answer") in first_result["probes"]:
            probe_network = first_result["probes"][("djepa", "joint", "answer")].network
            first_layer_weights = probe_network[0].weight.data.abs().cpu()

            block_dim = first_result["models"]["djepa"].block_dim
            block_norms = [
                first_layer_weights[:, start : start + block_dim].norm(p=2).item()
                for start in range(0, 4 * block_dim, block_dim)
            ]
            block_names = [
                "$c_1$ (Shared Vis)",
                "$c_2$ (Shared Txt)",
                "$u_1$ (Unique Vis)",
                "$u_2$ (Unique Txt)",
            ]
            total = sum(block_norms)
            percentages = [norm / total * 100 for norm in block_norms]

            fig, ax = plt.subplots(figsize=(6.5, 3.6))
            bar_colors = ["#4A90E2", "#50E3C2", "#F5A623", "#E94E77"]
            bars = ax.bar(
                block_names,
                percentages,
                color=bar_colors,
                edgecolor="black",
                linewidth=0.6,
                width=0.55,
            )
            for bar, percentage in zip(bars, percentages):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.0,
                    f"{percentage:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

            ax.set_ylabel("Norm Weight Attribution (%)")
            ax.set_ylim(0, max(percentages) + 8)
            ax.set_title("Downstream Probe Saliency per D-JEPA Block", pad=10)
            ax.grid(axis="y", linestyle=":", alpha=0.5)
            fig.tight_layout()
            _save_figure(fig, out_dir, "fig4_probe_block_attribution.png")
    except Exception as error:
        print(f"  [!] Skipped Fig 4: {error}")

    # FIG 5: latent-space geometry and disentanglement (t-SNE).
    try:
        first_result = results[0]
        if "blocks" in first_result and "djepa" in first_result["blocks"]:
            blocks = first_result["blocks"]["djepa"]
            n_points = min(1000, blocks["c1"].shape[0])

            representations = [
                blocks["c1"][:n_points].numpy(),
                blocks["c2"][:n_points].numpy(),
                blocks["u1"][:n_points].numpy(),
                blocks["u2"][:n_points].numpy(),
            ]
            group_names = [
                "Shared $c_1$ (Vision)",
                "Shared $c_2$ (Text)",
                "Unique $u_1$ (Vision)",
                "Unique $u_2$ (Text)",
            ]
            all_representations = np.concatenate(representations, axis=0)
            labels = [name for name in group_names for _ in range(n_points)]

            tsne = TSNE(n_components=2, perplexity=30, random_state=42, init="pca", max_iter=1000)
            embedded = tsne.fit_transform(all_representations)

            fig, ax = plt.subplots(figsize=(6.8, 5.2))
            colors = {
                "Shared $c_1$ (Vision)": "#2E5EAA",
                "Shared $c_2$ (Text)": "#50E3C2",
                "Unique $u_1$ (Vision)": "#C56A12",
                "Unique $u_2$ (Text)": "#6D4AA5",
            }

            for group_name in colors:
                indices = [index for index, label in enumerate(labels) if label == group_name]
                ax.scatter(
                    embedded[indices, 0],
                    embedded[indices, 1],
                    label=group_name,
                    c=colors[group_name],
                    s=16,
                    alpha=0.6,
                    edgecolors="none",
                )

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title("t-SNE Geometry: Shared Alignment vs. Private Isolation", pad=10)
            ax.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=8, loc="best")
            fig.tight_layout()
            _save_figure(fig, out_dir, "fig5_tsne_latent_dissection.png")
    except Exception as error:
        print(f"  [!] Skipped Fig 5: {error}")

    # FIG 6 & 7: PID information allocations.
    plot_pid(
        summary,
        "ppid",
        "fig6_probe_pid.png",
        "Probe-PID: Information Allocation Across Modality Streams",
        out_dir,
    )
    if summary.get("fpid"):
        plot_pid(
            summary,
            "fpid",
            "fig7_flow_pid.png",
            "Flow-PID: Continuous Latent Density Decomposition",
            out_dir,
        )
