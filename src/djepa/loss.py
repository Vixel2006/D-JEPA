import torch
import torch.nn.functional as F


def jepa_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(prediction, target)


def alignment_loss(c1: torch.Tensor, c2: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(c1, c2)


def independence_loss(
    c: torch.Tensor, u: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    c = c.reshape(-1, c.shape[-1])
    u = u.reshape(-1, u.shape[-1])

    c = c - c.mean(dim=0, keepdim=True)
    u = u - u.mean(dim=0, keepdim=True)
    c = c / c.square().mean(dim=0, keepdim=True).sqrt().clamp_min(eps)
    u = u / u.square().mean(dim=0, keepdim=True).sqrt().clamp_min(eps)

    cross_correlation = c.T @ u / c.shape[0]
    return cross_correlation.square().mean()
