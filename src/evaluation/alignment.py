"""Orthogonal Procrustes alignment between vision and text subspaces."""

from __future__ import annotations

import logging
import time

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from ..utils import (
    autocast_context,
    format_duration,
    progress_suffix,
)

LOGGER = logging.getLogger(__name__)


def solve_procrustes_centered(
    vision: torch.Tensor,
    text: torch.Tensor,
) -> float:
    """Mean cosine similarity after centering and an optimal rotation."""

    vision_centered = vision - vision.mean(dim=0, keepdim=True)
    text_centered = text - text.mean(dim=0, keepdim=True)

    cross_covariance = text_centered.T @ vision_centered
    u, _, vt = torch.linalg.svd(cross_covariance)
    rotation = u @ vt

    aligned_text = text_centered @ rotation
    return F.cosine_similarity(
        vision_centered, aligned_text, dim=-1
    ).mean().item()


def procrustes_alignment(
    lejepa: nn.Module,
    decomposed: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    max_samples: int = 800,
    batch_size: int = 64,
) -> tuple[float, float]:
    """Return (LeJEPA alignment, D-JEPA shared-code alignment)."""

    lejepa_vision: list[torch.Tensor] = []
    lejepa_text: list[torch.Tensor] = []
    djepa_vision: list[torch.Tensor] = []
    djepa_text: list[torch.Tensor] = []

    started = time.perf_counter()
    LOGGER.info("Computing Procrustes alignment (up to %d samples)", max_samples)

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

            lejepa_vision.append(lejepa_out["z1"].cpu().float())
            lejepa_text.append(lejepa_out["z2"].cpu().float())
            djepa_vision.append(decomposed_out["c1"].cpu().float())
            djepa_text.append(decomposed_out["c2"].cpu().float())

            if batch_index % 5 == 0:
                LOGGER.info(
                    "procrustes %5d samples | %.1fs elapsed",
                    len(lejepa_vision) * batch_size,
                    time.perf_counter() - started,
                )

            if len(lejepa_vision) * batch_size >= max_samples:
                break

    lejepa_v = torch.cat(lejepa_vision)[:max_samples]
    lejepa_t = torch.cat(lejepa_text)[:max_samples]
    djepa_v = torch.cat(djepa_vision)[:max_samples]
    djepa_t = torch.cat(djepa_text)[:max_samples]

    lejepa_score = solve_procrustes_centered(lejepa_v, lejepa_t)
    djepa_score = solve_procrustes_centered(djepa_v, djepa_t)
    LOGGER.info(
        "Procrustes finished in %s | LeJEPA %.4f | D-JEPA %.4f",
        format_duration(time.perf_counter() - started),
        lejepa_score,
        djepa_score,
    )
    return lejepa_score, djepa_score
