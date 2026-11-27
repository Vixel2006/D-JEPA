import torch
import torch.nn as nn


def _critic_mlp(input_dim: int, hidden_dim: int, output_dim: int, layers: int = 1):
    modules = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
    for _ in range(layers):
        modules.extend((nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))
    modules.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*modules)


class InfoNCECritic(nn.Module):
    """Pairwise MLP critic for the FactorCL InfoNCE lower bound."""

    def __init__(self, x_dim: int, y_dim: int, hidden_dim: int, layers: int = 1):
        super().__init__()
        self.score = _critic_mlp(x_dim + y_dim, hidden_dim, 1, layers)

    def _scores(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x_pairs = x.unsqueeze(0).expand(batch_size, -1, -1)
        y_pairs = y.unsqueeze(1).expand(-1, batch_size, -1)
        return self.score(torch.cat((x_pairs, y_pairs), dim=-1)).squeeze(-1)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        scores = self._scores(x, y)
        lower_bound = scores.diagonal().mean() - scores.logsumexp(dim=1).mean()
        lower_bound = lower_bound + torch.log(
            torch.tensor(float(scores.shape[0]), device=scores.device, dtype=scores.dtype)
        )
        return -lower_bound


class CLUBInfoNCECritic(InfoNCECritic):
    """NCE-CLUB upper-bound critic and its critic-fitting objective."""

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        scores = self._scores(x, y)
        return scores.diagonal().mean() - scores.mean()

    def learning_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        scores = self._scores(x, y)
        lower_bound = scores.diagonal().mean() - scores.logsumexp(dim=1).mean()
        lower_bound = lower_bound + torch.log(
            torch.tensor(float(scores.shape[0]), device=scores.device, dtype=scores.dtype)
        )
        return -lower_bound
