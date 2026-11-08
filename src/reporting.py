"""Console tables and machine-readable artefacts for a completed run."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .config import FigureConfig
from .evaluation.probes import (
    CATEGORIES,
    DJEPA_C,
    DJEPA_FULL,
    LEJEPA,
)
from .figures import FigureData, generate_all_figures
from .results import CaseStudy, PipelineResults


def _mean_std(values: list[float]) -> tuple[float, float]:
    return float(np.mean(values)), float(np.std(values))


def aggregate_accuracy(
    results: PipelineResults,
) -> dict[str, dict[str, tuple[float, float]]]:
    aggregated: dict[str, dict[str, tuple[float, float]]] = {}
    for subspace in results.subspace_names:
        aggregated[subspace] = {}
        for category in (*results.categories, "overall"):
            values = [
                metrics.accuracy[subspace][category]
                for metrics in results.seed_metrics
            ]
            aggregated[subspace][category] = _mean_std(values)
    return aggregated


def print_accuracy_table(results: PipelineResults) -> None:
    aggregated = aggregate_accuracy(results)
    header = (
        f"{'Model / Subspace Configuration':<36} | "
        f"{'Overall':^13} | {'Counting':^13} | "
        f"{'Attribute':^13} | {'Spatial Rel':^13}"
    )
    separator = "-" * 96
    print("\n" + "=" * 96)
    print(
        "1. MULTI-SEED VQA REASONING TABLE "
        f"(MEAN ± STD ACROSS {len(results.seeds)} SEEDS)"
    )
    print("=" * 96)
    print(header)
    print(separator)
    for subspace in results.subspace_names:
        overall_mean, overall_std = aggregated[subspace]["overall"]
        counting_mean, counting_std = aggregated[subspace]["counting"]
        attribute_mean, attribute_std = aggregated[subspace]["attribute"]
        spatial_mean, spatial_std = aggregated[subspace]["spatial_relation"]
        print(
            f"{subspace:<36} | "
            f"{overall_mean:5.2f} ± {overall_std:4.2f} | "
            f"{counting_mean:5.2f} ± {counting_std:4.2f} | "
            f"{attribute_mean:5.2f} ± {attribute_std:4.2f} | "
            f"{spatial_mean:5.2f} ± {spatial_std:4.2f}"
        )
    print(separator)


def print_flip_rate(results: PipelineResults) -> None:
    if results.diagnostics and results.diagnostics.swaps:
        print("\n" + "=" * 80)
        print("2. CAUSAL FIBRE-SWAP INTERVENTION RATES")
        print("=" * 80)
        header = f"{'Fibre':<8} | {'Overall':>9} | {'Counting':>9} | {'Attribute':>10} | {'Spatial':>9}"
        print(header)
        print("-" * 62)
        for fibre, values in results.diagnostics.swaps.items():
            print(
                f"{fibre:<8} | "
                f"{values.get('overall', 0.0):8.2f}% | "
                f"{values.get('counting', 0.0):8.2f}% | "
                f"{values.get('attribute', 0.0):9.2f}% | "
                f"{values.get('spatial_relation', 0.0):8.2f}%"
            )
        print("-" * 62)
        print("u1 uses the [c1,c2,u1] probe, u2 the [c1,c2,u2] probe,")
        print("and s the full probe. Values are prediction flip rates.")
        print("=" * 80)
        return

    mean, std = _mean_std([m.flip_rate for m in results.seed_metrics])
    print("\n" + "=" * 80)
    print(
        "2. CAUSAL FIBER SWAP INTERVENTION RATE "
        f"(MEAN ± STD ACROSS {len(results.seeds)} SEEDS)"
    )
    print("=" * 80)
    print(
        "Causal Prediction Flip Rate: "
        f"{mean:.2f}% ± {std:.2f}% "
        "(Intervention on u1 holding [c1, c2] fixed)"
    )
    print("=" * 80)


def print_procrustes(results: PipelineResults) -> None:
    lejepa_mean, lejepa_std = _mean_std(
        [m.procrustes_lejepa for m in results.seed_metrics]
    )
    djepa_mean, djepa_std = _mean_std(
        [m.procrustes_djepa for m in results.seed_metrics]
    )
    print("\n" + "=" * 80)
    print(
        "3. PROCRUSTES ROTATION ALIGNMENT COSINE "
        f"(MEAN ± STD ACROSS {len(results.seeds)} SEEDS)"
    )
    print("=" * 80)
    print(
        "LeJEPA Baseline (Balestriero & LeCun, 2026): "
        f"{lejepa_mean:.4f} ± {lejepa_std:.4f}"
    )
    print(
        "D-JEPA shared base [c]:                     "
        f"{djepa_mean:.4f} ± {djepa_std:.4f}"
    )
    print("=" * 80)


def print_pid_table(results: PipelineResults) -> None:
    print("\n" + "=" * 80)
    print("4. FLOW-PID INFORMATION ALLOCATION TABLE (DEGREE-2 RKHS R^2)")
    print("=" * 80)
    print(
        f"{'Code':<12} | {'R2(X1)':>7} {'R2(X2)':>7} {'R2(Joint)':>9} | "
        f"{'Unq1':>6} {'Unq2':>6} {'Red':>6} {'Syn':>6} | Dominant"
    )
    print("-" * 76)
    for row in results.pid_rows:
        print(
            f"{row.code:<12} | "
            f"{row.r2_x1:7.2f} {row.r2_x2:7.2f} {row.r2_joint:9.2f} | "
            f"{row.unique_x1:6.2f} {row.unique_x2:6.2f} "
            f"{row.redundancy:6.2f} {row.synergy:6.2f} | {row.dominant}"
        )
    print("-" * 76)


def print_spectral_table(results: PipelineResults) -> None:
    if not results.diagnostics or not results.diagnostics.spectral:
        return
    print("\n" + "=" * 88)
    print("5. REPRESENTATION GEOMETRY (HELD-OUT TEST SET)")
    print("=" * 88)
    print(
        f"{'Code':<8} | {'Dim':>4} | {'Eff. rank':>9} | "
        f"{'Participation':>13} | {'Stable rank':>11} | "
        f"{'Top-1 var':>9} | {'Cond.':>8}"
    )
    print("-" * 88)
    for metrics in results.diagnostics.spectral:
        condition = (
            f"{metrics.condition_number:.1e}"
            if metrics.condition_number != float("inf")
            else "inf"
        )
        print(
            f"{metrics.name:<8} | {metrics.dim:4d} | "
            f"{metrics.effective_rank:9.2f} | "
            f"{metrics.participation_ratio:13.2f} | "
            f"{metrics.stable_rank:11.2f} | "
            f"{metrics.top1_variance:9.3f} | {condition:>8}"
        )
    print("-" * 88)


def print_calibration_table(results: PipelineResults) -> None:
    if not results.diagnostics or not results.diagnostics.calibration:
        return
    print("\n" + "=" * 80)
    print("6. PROBE CALIBRATION (HELD-OUT TEST SET)")
    print("=" * 80)
    print(
        f"{'Condition':<36} | {'Acc.':>7} | {'Macro F1':>8} | "
        f"{'ECE':>7} | {'NLL':>7}"
    )
    print("-" * 80)
    for metrics in results.diagnostics.calibration:
        print(
            f"{metrics.condition:<36} | "
            f"{metrics.accuracy:6.2f}% | {metrics.macro_f1:8.3f} | "
            f"{metrics.ece:7.3f} | {metrics.nll:7.3f}"
        )
    print("-" * 80)


def print_case_studies(results: PipelineResults) -> None:
    print("\n" + "=" * 80)
    print("7. CASE STUDIES (WINS AND LOSSES PER CATEGORY)")
    print("=" * 80)

    for category in CATEGORIES:
        print(f"\n--- Category: {category.upper()} ---")
        for kind, label in (("wins", "CONFIRMED WIN"), ("losses", "FAILURE MODE")):
            study = results.case_studies.get(kind, {}).get(category)
            if study is None:
                continue
            print(
                f"{label}:  Q: \"{study.question}\" | "
                f"GT: '{study.ground_truth}' | "
                f"LeJEPA: '{study.lejepa_prediction}' | "
                f"D-JEPA: '{study.djepa_prediction}'"
            )
            print(f"Caption: {study.caption}")


def print_summary(results: PipelineResults) -> None:
    aggregated = aggregate_accuracy(results)
    lejepa_overall = aggregated[LEJEPA]["overall"]
    djepa_overall = aggregated[DJEPA_FULL]["overall"]
    gap_mean = djepa_overall[0] - lejepa_overall[0]
    gap_std = math.sqrt(lejepa_overall[1] ** 2 + djepa_overall[1] ** 2)

    lejepa_attribute = aggregated[LEJEPA]["attribute"]
    djepa_attribute = aggregated[DJEPA_FULL]["attribute"]
    lejepa_spatial = aggregated[LEJEPA]["spatial_relation"]
    djepa_spatial = aggregated[DJEPA_FULL]["spatial_relation"]

    flip_mean, flip_std = _mean_std([m.flip_rate for m in results.seed_metrics])
    proc_lejepa = _mean_std(
        [m.procrustes_lejepa for m in results.seed_metrics]
    )
    proc_djepa = _mean_std(
        [m.procrustes_djepa for m in results.seed_metrics]
    )

    print("\n" + "=" * 80)
    print("8. SUMMARY FOR PAPER ABSTRACT:")
    print("=" * 80)
    print(f"  LeJEPA overall: {lejepa_overall[0]:.2f} ± {lejepa_overall[1]:.2f}")
    print(f"  D-JEPA Full overall:      {djepa_overall[0]:.2f} ± {djepa_overall[1]:.2f}")
    print(f"  Gap:                      {gap_mean:.2f} ± {gap_std:.2f}")
    print(
        f"  Attribute baseline:       "
        f"{lejepa_attribute[0]:.2f} ± {lejepa_attribute[1]:.2f}"
    )
    print(
        f"  Attribute D-JEPA:         "
        f"{djepa_attribute[0]:.2f} ± {djepa_attribute[1]:.2f}"
    )
    print(
        f"  Spatial-rel baseline:     "
        f"{lejepa_spatial[0]:.2f} ± {lejepa_spatial[1]:.2f}"
    )
    print(
        f"  Spatial-rel D-JEPA:       "
        f"{djepa_spatial[0]:.2f} ± {djepa_spatial[1]:.2f}"
    )
    print(f"  Causal flip (u1):         {flip_mean:.2f} ± {flip_std:.2f}")
    print(
        f"  Procrustes (LeJEPA):    "
        f"{proc_lejepa[0]:.4f} ± {proc_lejepa[1]:.4f}"
    )
    print(
        f"  Procrustes (D-JEPA c):    "
        f"{proc_djepa[0]:.4f} ± {proc_djepa[1]:.4f}"
    )
    print("=" * 80 + "\n")


def _save_case_study_images(
    case_studies: dict[str, dict[str, Optional[CaseStudy]]],
    images: dict[int, object],
    output_dir: Path,
) -> None:
    case_dir = output_dir / "case_studies"
    case_dir.mkdir(parents=True, exist_ok=True)
    for kind, entries in case_studies.items():
        for category, study in entries.items():
            if study is None:
                continue
            image = images.get(study.scene_id)
            if image is None:
                continue
            category_name = (
                "spatial" if category == "spatial_relation" else category
            )
            kind_name = "win" if kind == "wins" else "loss"
            filename = f"clevr_fig_{category_name}_{kind_name}.jpg"
            image.convert("RGB").save(case_dir / filename, quality=95)


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def save_results(
    results: PipelineResults,
    output_dir: Path,
    images: Optional[dict[int, object]] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregated = aggregate_accuracy(results)

    accuracy_payload = {
        subspace: {
            category: {"mean": mean, "std": std}
            for category, (mean, std) in categories.items()
        }
        for subspace, categories in aggregated.items()
    }
    flip_mean, flip_std = _mean_std(
        [metrics.flip_rate for metrics in results.seed_metrics]
    )
    proc_lejepa = _mean_std(
        [metrics.procrustes_lejepa for metrics in results.seed_metrics]
    )
    proc_djepa = _mean_std(
        [metrics.procrustes_djepa for metrics in results.seed_metrics]
    )

    payload = {
        "seeds": list(results.seeds),
        "accuracy": accuracy_payload,
        "flip_rate": {"mean": flip_mean, "std": flip_std},
        "procrustes": {
            "lejepa": {"mean": proc_lejepa[0], "std": proc_lejepa[1]},
            "djepa": {"mean": proc_djepa[0], "std": proc_djepa[1]},
        },
        "pid": [asdict(row) for row in results.pid_rows],
        "case_studies": {
            kind: {
                category: None if study is None else asdict(study)
                for category, study in entries.items()
            }
            for kind, entries in results.case_studies.items()
        },
        "diagnostics": (
            None
            if results.diagnostics is None
            else {
                "spectral": [asdict(row) for row in results.diagnostics.spectral],
                "cka": [asdict(row) for row in results.diagnostics.cka],
                "alignment": [
                    asdict(row) for row in results.diagnostics.alignment
                ],
                "calibration": [
                    asdict(row) for row in results.diagnostics.calibration
                ],
                "swaps": results.diagnostics.swaps,
                "confusion": results.diagnostics.confusion,
            }
        ),
        "loss_history": results.loss_history,
        "figures": results.figure_paths,
        "num_test_records": len(results.records),
    }
    with (output_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_jsonable)

    with (output_dir / "accuracy.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["subspace", "category", "mean", "std"])
        for subspace, categories in aggregated.items():
            for category, (mean, std) in categories.items():
                writer.writerow([subspace, category, mean, std])

    with (output_dir / "pid.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "code",
                "r2_x1",
                "r2_x2",
                "r2_joint",
                "unique_x1",
                "unique_x2",
                "redundancy",
                "synergy",
                "dominant",
            ]
        )
        for row in results.pid_rows:
            writer.writerow(asdict(row).values())

    if results.diagnostics is not None:
        with (output_dir / "diagnostics.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(
                {
                    "spectral": [
                        asdict(row) for row in results.diagnostics.spectral
                    ],
                    "cka": [asdict(row) for row in results.diagnostics.cka],
                    "alignment": [
                        asdict(row) for row in results.diagnostics.alignment
                    ],
                    "calibration": [
                        asdict(row) for row in results.diagnostics.calibration
                    ],
                },
                handle,
                indent=2,
            )

        with (output_dir / "spectral.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(list(SpectralFields))
            for row in results.diagnostics.spectral:
                writer.writerow(
                    [getattr(row, field) for field in SpectralFields]
                )

        with (output_dir / "calibration.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(list(CalibrationFields))
            for row in results.diagnostics.calibration:
                writer.writerow(
                    [getattr(row, field) for field in CalibrationFields]
                )

    if results.records:
        with (output_dir / "test_records.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(results.records, handle, indent=2, default=_jsonable)

    if results.embeddings:
        np.savez_compressed(
            output_dir / "embeddings.npz",
            **{
                name: np.asarray(values)
                for name, values in results.embeddings.items()
            },
        )

    if images is not None:
        _save_case_study_images(results.case_studies, images, output_dir)


SpectralFields = (
    "name",
    "dim",
    "effective_rank",
    "participation_ratio",
    "stable_rank",
    "top1_variance",
    "condition_number",
)
CalibrationFields = ("condition", "accuracy", "macro_f1", "ece", "nll")


def report_results(
    results: PipelineResults,
    output_dir: Path,
    save: bool = True,
    images: Optional[dict[int, object]] = None,
    figure_data: Optional[FigureData] = None,
    figure_config: FigureConfig = FigureConfig(),
) -> None:
    print_accuracy_table(results)
    print_flip_rate(results)
    print_procrustes(results)
    print_pid_table(results)
    print_spectral_table(results)
    print_calibration_table(results)
    if results.case_studies:
        print_case_studies(results)
    print_summary(results)

    if save and figure_data is not None and figure_config.enabled:
        try:
            results.figure_paths = generate_all_figures(
                figure_data,
                output_dir=Path(output_dir) / "figures",
                config=figure_config,
                pid_rows=results.pid_rows,
            )
        except Exception:
            # Figures should never invalidate an otherwise successful run.
            import logging

            logging.getLogger(__name__).exception(
                "Figure generation failed; continuing."
            )

    if save:
        save_results(results, output_dir, images=images)
