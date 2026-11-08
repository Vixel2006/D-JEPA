"""Projection, prediction, and fusion heads."""

from __future__ import annotations

from torch import nn


class LeJEPAProjector(nn.Module):
    """The three-layer BatchNorm projector used by the original experiment."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Linear(1024, out_dim),
            nn.BatchNorm1d(out_dim),
        )

    def forward(self, features):
        return self.net(features)


# Backward-compatible alias for readers of the original script.
OfficialLeJEPAProjector = LeJEPAProjector


def make_predictor(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    """Contextual JEPA predictor used to ground each private fibre."""

    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, output_dim),
        nn.LayerNorm(output_dim),
    )


class SynergyHead(nn.Module):
    """Centered fusion head that predicts a shared synergy code."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, features):
        return self.net(features)
