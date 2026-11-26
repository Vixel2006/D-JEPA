import torch
from torch.utils.data import Dataset


class SyntheticDataset(Dataset):
    """Paired image/text views with known shared and private factors."""

    def __init__(
        self,
        n_samples: int = 1024,
        seed: int = 0,
        split: str = "train",
        train_fraction: float = 0.8,
    ):
        if split not in {"train", "validation", "all"}:
            raise ValueError("split must be 'train', 'validation', or 'all'.")

        generator = torch.Generator().manual_seed(seed)
        shared = torch.randn(n_samples, 4, generator=generator)
        private_vision = torch.randn(n_samples, 4, generator=generator)
        private_text = torch.randn(n_samples, 4, generator=generator)
        nuisance_vision = torch.randn(n_samples, 1, generator=generator)
        nuisance_text = torch.randn(n_samples, 1, generator=generator)

        vision_latent = torch.cat((shared, private_vision, nuisance_vision), dim=-1)
        text_latent = torch.cat((shared, private_text, nuisance_text), dim=-1)

        self.context = {
            "vision": self._make_images(vision_latent, generator),
            "text": self._make_text(text_latent, generator),
        }
        self.target = {
            "vision": self._make_images(vision_latent, generator),
            "text": self._make_text(text_latent, generator),
        }
        self.factors = {
            "shared": shared,
            "private_vision": private_vision,
            "private_text": private_text,
            "nuisance_vision": nuisance_vision,
            "nuisance_text": nuisance_text,
            "synergy": torch.sign(private_vision[:, 0])
            * torch.sign(private_text[:, 0])
            + 0.05 * torch.randn(n_samples, generator=generator),
        }

        permutation = torch.randperm(n_samples, generator=generator)
        split_at = int(n_samples * train_fraction)
        if split == "train":
            self.indices = permutation[:split_at]
        elif split == "validation":
            self.indices = permutation[split_at:]
        else:
            self.indices = torch.arange(n_samples)

    @staticmethod
    def _make_images(latent, generator):
        values = torch.tanh(latent)
        image = torch.zeros(len(latent), 32, 32, 3)
        for coordinate in range(values.shape[1]):
            row, column = divmod(coordinate, 5)
            channel = coordinate % 3
            image[:, row * 16 : (row + 1) * 16, column * 6 : (column + 1) * 6, channel] = (
                values[:, coordinate, None, None]
            )
        noise = 0.05 * torch.randn(image.shape, generator=generator)
        return (image + noise).clamp(-1, 1)

    @staticmethod
    def _make_text(latent, generator):
        noisy_latent = latent + 0.03 * torch.randn(latent.shape, generator=generator)
        coordinates = torch.arange(latent.shape[1]).repeat(2)
        values = noisy_latent[:, coordinates]
        bins = ((torch.tanh(values) + 1) * 16).long().clamp(0, 31)
        return bins + coordinates * 32

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        row = self.indices[index].item()
        return {
            "context": {name: value[row] for name, value in self.context.items()},
            "target": {name: value[row] for name, value in self.target.items()},
            "factors": {name: value[row] for name, value in self.factors.items()},
        }
