"""Collect held-out features once for diagnostics, figures, and case studies."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..utils import (
    autocast_context,
    format_duration,
    progress_suffix,
)
from .probes import (
    CATEGORIES,
    DJEPA_C,
    DJEPA_C_U1,
    DJEPA_C_U2,
    DJEPA_C_U1_U2,
    DJEPA_FULL,
    LEJEPA,
    build_probe_features,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class EvaluationFeatures:
    """Held-out representations for every probe and diagnostic."""

    condition_features: dict[str, torch.Tensor]
    representations: dict[str, torch.Tensor]
    labels: torch.Tensor
    categories: list[str]
    questions: list[str]
    answers: list[str]
    scene_ids: list[int]
    logits: dict[str, torch.Tensor] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.labels.size(0))

    @property
    def predictions(self) -> dict[str, torch.Tensor]:
        return {
            name: values.argmax(dim=-1)
            for name, values in self.logits.items()
        }


def collect_evaluation_features(
    lejepa: nn.Module,
    decomposed: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    component_dim: int,
    max_samples: Optional[int] = None,
) -> EvaluationFeatures:
    """Run both encoders once over the held-out loader."""

    condition_chunks: dict[str, list[torch.Tensor]] = {}
    representation_chunks: dict[str, list[torch.Tensor]] = {}
    labels: list[torch.Tensor] = []
    categories: list[str] = []
    questions: list[str] = []
    answers: list[str] = []
    scene_ids: list[int] = []
    collected = 0
    started = time.perf_counter()
    total_batches = max(1, len(loader))
    log_every = max(1, total_batches // 5)
    LOGGER.info(
        "Collecting held-out features on %d batches", total_batches
    )

    with torch.no_grad():
        for batch_index, batch in enumerate(loader, 1):
            eval_view = batch["eval_view"].to(device)
            question_ids = batch["question_ids"].to(device)
            question_mask = batch["question_mask"].to(device)

            with autocast_context(device, amp):
                lejepa_out = lejepa.extract_features(
                    eval_view, question_ids, question_mask
                )
                decomposed_out = decomposed.extract_features(
                    eval_view, question_ids, question_mask
                )

            random_vector = torch.randn(
                eval_view.size(0), component_dim, device=device
            )
            features = build_probe_features(
                lejepa_out, decomposed_out, random_vector
            )
            for name, values in features.items():
                condition_chunks.setdefault(name, []).append(
                    values.detach().cpu().float()
                )

            raw = {
                "z1": lejepa_out["z1"],
                "z2": lejepa_out["z2"],
                "c1": decomposed_out["c1"],
                "c2": decomposed_out["c2"],
                "c": decomposed_out["c"],
                "u1": decomposed_out["u1"],
                "u2": decomposed_out["u2"],
                "s": decomposed_out["s"],
            }
            for name, values in raw.items():
                representation_chunks.setdefault(name, []).append(
                    values.detach().cpu().float()
                )

            labels.append(batch["label"].detach().cpu())
            categories.extend(list(batch["category"]))
            questions.extend(list(batch["question"]))
            answers.extend(list(batch["answer"]))
            scene_ids.extend(int(value) for value in batch["scene_id"])

            collected += eval_view.size(0)
            if batch_index == total_batches or batch_index % log_every == 0:
                LOGGER.info(
                    "feature collection %s | %d samples",
                    progress_suffix(batch_index, total_batches, started),
                    collected,
                )
            if max_samples is not None and collected >= max_samples:
                break

    LOGGER.info(
        "Collected %d held-out features in %s",
        len(scene_ids),
        format_duration(time.perf_counter() - started),
    )

    def _concat(chunks: dict[str, list[torch.Tensor]]) -> dict[str, torch.Tensor]:
        return {
            name: torch.cat(values, dim=0)[:max_samples]
            if max_samples is not None
            else torch.cat(values, dim=0)
            for name, values in chunks.items()
        }

    limit = len(scene_ids) if max_samples is None else min(
        max_samples, len(scene_ids)
    )
    return EvaluationFeatures(
        condition_features=_concat(condition_chunks),
        representations=_concat(representation_chunks),
        labels=torch.cat(labels, dim=0)[:limit],
        categories=categories[:limit],
        questions=questions[:limit],
        answers=answers[:limit],
        scene_ids=scene_ids[:limit],
    )


def compute_logits(
    probes: dict[str, nn.Module],
    features: EvaluationFeatures,
) -> None:
    """Populate ``features.logits`` in place for every probe."""

    features.logits = {}
    with torch.no_grad():
        for name, probe in probes.items():
            if name not in features.condition_features:
                continue
            probe.eval()
            try:
                probe_device = next(probe.parameters()).device
            except StopIteration:  # parameter-free probe
                probe_device = torch.device("cpu")
            inputs = features.condition_features[name].to(probe_device)
            features.logits[name] = probe(inputs).cpu()


def build_test_records(
    features: EvaluationFeatures,
    answer_vocabulary: list[str],
    conditions: tuple[str, ...] = (
        LEJEPA,
        DJEPA_C,
        DJEPA_C_U1,
        DJEPA_C_U2,
        DJEPA_C_U1_U2,
        DJEPA_FULL,
    ),
) -> list[dict]:
    """Build per-example prediction records for figures and case studies."""

    predictions = features.predictions
    records: list[dict] = []
    for index in range(len(features)):
        ground_truth = features.answers[index]
        record = {
            "idx": features.scene_ids[index],
            "row": index,
            "category": features.categories[index],
            "question": features.questions[index],
            "ground_truth": ground_truth,
        }
        for condition in conditions:
            if condition not in predictions:
                continue
            predicted_index = int(predictions[condition][index])
            predicted = answer_vocabulary[predicted_index]
            record[_short_name(condition)] = predicted
        record["is_win"] = bool(
            record.get("full") == ground_truth
            and record.get("lejepa") != ground_truth
        )
        record["is_loss"] = bool(
            record.get("lejepa") == ground_truth
            and record.get("full") != ground_truth
        )
        records.append(record)
    return records


def _short_name(condition: str) -> str:
    mapping = {
        LEJEPA: "lejepa",
        DJEPA_C: "c_only",
        DJEPA_C_U1: "u1_only",
        DJEPA_C_U2: "u2_only",
        DJEPA_C_U1_U2: "both_u1_u2",
        DJEPA_FULL: "full",
    }
    return mapping.get(condition, condition)
