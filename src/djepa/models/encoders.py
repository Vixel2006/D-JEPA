import torch
import torch.nn as nn
from torchvision.ops import MLP

from ...models.backbones import BertBackbone, ViTBackbone


class ViTEncoder(ViTBackbone):
    """D-JEPA vision encoder with shared and unique projection heads."""

    def __init__(
        self,
        model_name: str = "vit_small_patch8_224",
        img_size: int = 128,
        proj_dim: int = 128,
    ):
        super().__init__(model_name=model_name, img_size=img_size)
        self.c_proj = MLP(512, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)
        self.u_proj = MLP(512, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)

    def forward(self, x: torch.Tensor):
        batch_size, num_views = x.shape[:2]
        embedding = self.encode_backbone(x)
        shared = self.c_proj(embedding).reshape(batch_size, num_views, -1)
        unique = self.u_proj(embedding).reshape(batch_size, num_views, -1)
        return embedding, shared.transpose(0, 1), unique.transpose(0, 1)


class BertEncoder(BertBackbone):
    """D-JEPA text encoder with shared and unique projection heads."""

    def __init__(
        self,
        proj_dim: int = 128,
        vocab_size: int = 30_522,
        hidden_size: int = 256,
        num_hidden_layers: int = 4,
        num_attention_heads: int = 4,
        intermediate_size: int = 1_024,
        max_position_embeddings: int = 512,
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
        )
        self.c_proj = MLP(
            hidden_size, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d
        )
        self.u_proj = MLP(
            hidden_size, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d
        )

    def forward(self, x: torch.Tensor | dict[str, torch.Tensor]):
        embedding = self.encode_backbone(x)
        return embedding, self.c_proj(embedding), self.u_proj(embedding)
