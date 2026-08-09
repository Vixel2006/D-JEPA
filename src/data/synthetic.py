import torch
from torch.utils.data import Dataset
import numpy as np


class SyntheticPairedData(Dataset):
    """
    Generates correlated synthetic image and text features from a shared latent variable s.

    x_img = MLP_img(s) + noise
    x_txt = MLP_txt(s) + noise
    """

    def __init__(
        self,
        num_samples: int = 2000,
        d_z: int = 128,
        img_dim: int = 512,
        txt_dim: int = 384,
        noise_std: float = 0.1,
        seed: int = 42,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.d_z = d_z
        self.img_dim = img_dim
        self.txt_dim = txt_dim
        self.noise_std = noise_std
        self.seed = seed

        # Save previous torch/numpy seed and set local seed
        prev_torch_seed = torch.initial_seed()
        prev_numpy_state = np.random.get_state()

        torch.manual_seed(seed)
        np.random.seed(seed)

        # Create fixed random projection MLPs (no training)
        self.mlp_img = torch.nn.Sequential(
            torch.nn.Linear(d_z, 256), torch.nn.ReLU(), torch.nn.Linear(256, img_dim)
        )
        self.mlp_txt = torch.nn.Sequential(
            torch.nn.Linear(d_z, 256), torch.nn.ReLU(), torch.nn.Linear(256, txt_dim)
        )

        # Freeze parameters
        for p in self.mlp_img.parameters():
            p.requires_grad = False
        for p in self.mlp_txt.parameters():
            p.requires_grad = False

        # Generate shared latents s ~ N(0, I)
        self.s = torch.randn(num_samples, d_z)

        # Generate features with noise
        with torch.no_grad():
            self.image_feat = self.mlp_img(self.s) + noise_std * torch.randn(
                num_samples, img_dim
            )
            self.text_feat = self.mlp_txt(self.s) + noise_std * torch.randn(
                num_samples, txt_dim
            )

        # Restore seed/state
        torch.manual_seed(prev_torch_seed)
        np.random.set_state(prev_numpy_state)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        return self.image_feat[idx], self.text_feat[idx], self.s[idx]
