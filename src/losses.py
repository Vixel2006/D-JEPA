"""Loss functions used by LeJEPA and D-JEPA."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def sigreg_loss(z: torch.Tensor, num_slices: int = 512) -> torch.Tensor:
    """Sketched Isotropic Gaussian Regulariser (SIGReg).

    The characteristic-function matching objective is evaluated on random
    one-dimensional projections of ``z``. Directions are resampled on every
    call, matching the original experiment.
    """

    batch_size, dim = z.shape
    if batch_size <= 2:
        return torch.tensor(0.0, device=z.device)

    directions = torch.randn(dim, num_slices, device=z.device)
    directions = F.normalize(directions, dim=0)

    projections = z @ directions
    t = torch.linspace(0.2, 4.0, 32, device=z.device).view(1, 1, -1)
    expanded = projections.unsqueeze(-1)

    cos_t = torch.cos(expanded * t).mean(dim=0)
    sin_t = torch.sin(expanded * t).mean(dim=0)
    target = torch.exp(-0.5 * t.squeeze(0) ** 2)

    return ((cos_t - target).pow(2) + sin_t.pow(2)).mean()


def lejepa_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    sigreg_weight: float = 0.02,
    num_slices: int = 512,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """LeJEPA invariance + SIGReg loss.

    Returns ``(total, invariance, regulariser)`` so callers can log the
    individual terms.
    """

    invariance = 0.5 * F.mse_loss(z1, z2)
    if sigreg_weight > 0.0:
        regulariser = 0.5 * (
            sigreg_loss(z1, num_slices=num_slices)
            + sigreg_loss(z2, num_slices=num_slices)
        )
    else:
        regulariser = torch.tensor(0.0, device=z1.device)
    return invariance + sigreg_weight * regulariser, invariance, regulariser


def covariance_decorrelation_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Penalise linear covariance between two representation fibres."""

    a_centered = a - a.mean(dim=0, keepdim=True)
    b_centered = b - b.mean(dim=0, keepdim=True)
    covariance = (a_centered.T @ b_centered) / (a.size(0) - 1.0)
    return covariance.pow(2).sum() / (a.size(1) * b.size(1))
