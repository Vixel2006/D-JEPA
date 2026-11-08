"""Regression tests for the held-out feature pass."""

from __future__ import annotations

import torch
from torch import nn

from djepa.evaluation.features import EvaluationFeatures, compute_logits


class _DeviceSpyProbe(nn.Module):
    """Probe whose parameter lives on a non-CPU device (the ``meta`` device)."""

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1, device="meta"))
        self.seen_device: str | None = None

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        self.seen_device = features.device.type
        # Return a real CPU tensor so ``compute_logits`` can stash it.
        return torch.zeros(features.shape[0], 5)


def test_compute_logits_moves_features_to_probe_device() -> None:
    features = EvaluationFeatures(
        condition_features={"probe": torch.randn(7, 3)},
        representations={},
        labels=torch.zeros(7, dtype=torch.long),
        categories=[],
        questions=[],
        answers=[],
        scene_ids=[],
    )
    probe = _DeviceSpyProbe()

    compute_logits({"probe": probe}, features)

    # The input must follow the probe, not stay on the collection device.
    assert probe.seen_device == "meta"
    assert features.logits["probe"].shape == (7, 5)
    assert features.logits["probe"].device.type == "cpu"
