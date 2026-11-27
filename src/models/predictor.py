import torch.nn as nn


class PredictorHead(nn.Module):
    """Small task head applied to projected D-JEPA representations."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, values):
        return self.network(values)
