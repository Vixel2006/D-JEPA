import torch
import torch.nn as nn


class Projector(nn.Module):
    def __init__(self, proj_dim: int, target_dim: int):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(2 * proj_dim, 2 * proj_dim),
            nn.GELU(),
        )
        self.predictor = nn.Linear(2 * proj_dim, target_dim)

    def forward(self, c: torch.Tensor, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.projector(torch.cat((c, u), dim=-1))
        return self.predictor(z), z
