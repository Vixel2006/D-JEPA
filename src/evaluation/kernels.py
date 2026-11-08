"""Kernel ridge regression utilities used by the Flow-PID analysis."""

from __future__ import annotations

import torch


def evaluate_kernel_ridge_r2(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    kernel_type: str = "poly2",
    lam: float = 0.5,
) -> float:
    """Degree-2 polynomial kernel ridge R^2, matching the original script."""

    x_train = x_train.double()
    y_train = y_train.double()
    x_test = x_test.double()
    y_test = y_test.double()

    x_mean = x_train.mean(dim=0)
    x_std = x_train.std(dim=0).clamp_min(1e-6)
    x_train = (x_train - x_mean) / x_std
    x_test = (x_test - x_mean) / x_std

    y_mean = y_train.mean(dim=0)
    y_train = y_train - y_mean
    y_test = y_test - y_mean

    dim = x_train.size(1)

    if kernel_type == "linear":
        kernel_train = x_train @ x_train.T / dim + 1.0
        kernel_test = x_test @ x_train.T / dim + 1.0
    elif kernel_type == "poly2":
        kernel_train = (x_train @ x_train.T / dim + 1.0).pow(2)
        kernel_test = (x_test @ x_train.T / dim + 1.0).pow(2)
    elif kernel_type == "rbf":
        gamma = 1.0 / dim
        kernel_train = torch.exp(
            -gamma * torch.cdist(x_train, x_train).pow(2)
        )
        kernel_test = torch.exp(
            -gamma * torch.cdist(x_test, x_train).pow(2)
        )
    else:
        raise ValueError(f"Unsupported kernel type: {kernel_type}")

    identity = torch.eye(
        kernel_train.size(0), dtype=torch.double, device=x_train.device
    )
    alpha = torch.linalg.solve(kernel_train + lam * identity, y_train)
    predictions = kernel_test @ alpha

    total_variance = y_test.pow(2).sum(dim=0).clamp_min(1e-8)
    residual = (y_test - predictions).pow(2).sum(dim=0)
    return (1.0 - residual / total_variance).mean().item()
