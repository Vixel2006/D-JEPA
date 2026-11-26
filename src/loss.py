import torch
import torch.nn.functional as F


def jepa_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError(
            f"Prediction and target shapes must match, got {prediction.shape} and {target.shape}."
        )
    return F.mse_loss(prediction, target)


def alignment_loss(c1: torch.Tensor, c2: torch.Tensor) -> torch.Tensor:
    if c1.shape != c2.shape:
        raise ValueError(
            f"Aligned representations must match, got {c1.shape} and {c2.shape}."
        )
    return F.mse_loss(c1, c2)


def independence_loss(
    c: torch.Tensor, u: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    if c.ndim < 2 or u.ndim < 2:
        raise ValueError(
            "Representations must have at least batch and feature dimensions."
        )
    if c.shape[:-1] != u.shape[:-1]:
        raise ValueError(
            f"Representations must have matching sample shapes, got {c.shape} and {u.shape}."
        )

    c = c.reshape(-1, c.shape[-1])
    u = u.reshape(-1, u.shape[-1])
    if c.shape[0] < 2:
        raise ValueError(
            "At least two samples are needed to compute cross-correlation."
        )

    c = c - c.mean(dim=0, keepdim=True)
    u = u - u.mean(dim=0, keepdim=True)
    c = c / c.square().mean(dim=0, keepdim=True).sqrt().clamp_min(eps)
    u = u / u.square().mean(dim=0, keepdim=True).sqrt().clamp_min(eps)

    cross_correlation = c.T @ u / c.shape[0]
    return cross_correlation.square().mean()
