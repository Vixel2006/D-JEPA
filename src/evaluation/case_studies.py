"""Qualitative win/loss case studies for the manuscript."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from ..results import CaseStudy
from ..utils import autocast_context
from .probes import (
    CATEGORIES,
    DJEPA_FULL,
    LEJEPA,
    CapacityMatchedTaskHead,
)

WIN_CAPTION = (
    "D-JEPA successfully leverages grounded coordinates to resolve the "
    "{category} query where LeJEPA alignment defaults to modal collapse."
)
LOSS_CAPTION = (
    "Failure case: LeJEPA correctly answers '{question}' as "
    "'{lejepa}', while D-JEPA Full answers '{djepa}'. We hypothesize that "
    "the private fibers introduce object-binding confusion when multiple "
    "same-shape objects are present; this is a target for the next iteration."
)


def case_studies_from_records(
    records,
    categories=CATEGORIES,
) -> dict[str, dict[str, Optional[CaseStudy]]]:
    """Build win/loss case studies directly from prediction records.

    This avoids re-running the encoders: ``records`` already contains every
    probe's prediction for each held-out example.
    """

    wins: dict[str, Optional[CaseStudy]] = {
        category: None for category in categories
    }
    losses: dict[str, Optional[CaseStudy]] = {
        category: None for category in categories
    }

    for record in records:
        category = record["category"]
        if category not in wins:
            continue
        if wins[category] is None and record.get("is_win"):
            wins[category] = CaseStudy(
                category=category,
                kind="win",
                question=record["question"],
                ground_truth=record["ground_truth"],
                lejepa_prediction=record.get("lejepa", ""),
                djepa_prediction=record.get("full", ""),
                scene_id=int(record["idx"]),
                caption=WIN_CAPTION.format(category=category),
            )
        if losses[category] is None and record.get("is_loss"):
            losses[category] = CaseStudy(
                category=category,
                kind="loss",
                question=record["question"],
                ground_truth=record["ground_truth"],
                lejepa_prediction=record.get("lejepa", ""),
                djepa_prediction=record.get("full", ""),
                scene_id=int(record["idx"]),
                caption=LOSS_CAPTION.format(
                    question=record["question"],
                    lejepa=record.get("lejepa", ""),
                    djepa=record.get("full", ""),
                ),
            )

    return {"wins": wins, "losses": losses}


def find_case_studies(
    dataset,
    test_indices: list[int],
    lejepa: nn.Module,
    decomposed: nn.Module,
    probes: dict[str, CapacityMatchedTaskHead],
    answers: list[str],
    device: torch.device,
    amp: bool,
) -> dict[str, dict[str, Optional[CaseStudy]]]:
    """Find the first win and loss per category, matching the original loop."""

    wins: dict[str, Optional[CaseStudy]] = {
        category: None for category in CATEGORIES
    }
    losses: dict[str, Optional[CaseStudy]] = {
        category: None for category in CATEGORIES
    }

    for probe in probes.values():
        probe.eval()

    with torch.no_grad():
        for index in test_indices:
            item = dataset[index]
            eval_view = item["eval_view"].unsqueeze(0).to(device)
            question_ids = item["question_ids"].unsqueeze(0).to(device)
            question_mask = item["question_mask"].unsqueeze(0).to(device)

            with autocast_context(device, amp):
                lejepa_out = lejepa.extract_features(
                    eval_view, question_ids, question_mask
                )
                decomposed_out = decomposed.extract_features(
                    eval_view, question_ids, question_mask
                )

            lejepa_features = lejepa_out["lejepa"].float()
            full_features = torch.cat(
                [
                    decomposed_out["c1"],
                    decomposed_out["c2"],
                    decomposed_out["u1"],
                    decomposed_out["u2"],
                    decomposed_out["s"],
                ],
                dim=-1,
            ).float()

            lejepa_prediction = answers[
                probes[LEJEPA](lejepa_features).argmax(-1).item()
            ]
            djepa_prediction = answers[
                probes[DJEPA_FULL](full_features).argmax(-1).item()
            ]

            category = item["category"]
            ground_truth = item["answer"]

            if (
                wins[category] is None
                and djepa_prediction == ground_truth
                and lejepa_prediction != ground_truth
            ):
                wins[category] = CaseStudy(
                    category=category,
                    kind="win",
                    question=item["question"],
                    ground_truth=ground_truth,
                    lejepa_prediction=lejepa_prediction,
                    djepa_prediction=djepa_prediction,
                    scene_id=item["scene_id"],
                    caption=WIN_CAPTION.format(category=category),
                )

            if (
                losses[category] is None
                and lejepa_prediction == ground_truth
                and djepa_prediction != ground_truth
            ):
                losses[category] = CaseStudy(
                    category=category,
                    kind="loss",
                    question=item["question"],
                    ground_truth=ground_truth,
                    lejepa_prediction=lejepa_prediction,
                    djepa_prediction=djepa_prediction,
                    scene_id=item["scene_id"],
                    caption=LOSS_CAPTION.format(
                        question=item["question"],
                        lejepa=lejepa_prediction,
                        djepa=djepa_prediction,
                    ),
                )

            if all(study is not None for study in wins.values()) and all(
                study is not None for study in losses.values()
            ):
                break

    return {"wins": wins, "losses": losses}
