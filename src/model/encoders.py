import timm
import torch
import torch.nn as nn
from torchvision.ops import MLP
from transformers import BertConfig, BertModel


class ViTEncoder(nn.Module):
    def __init__(
        self, model_name: str = "vit_small_patch8_224", img_size: int = 128, proj_dim: int = 128
    ):
        super().__init__()

        self.backbone = timm.create_model(
            model_name, pretrained=False, num_classes=512, drop_path_rate=0.1, img_size=img_size
        )
        self.c_proj = MLP(512, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)
        self.u_proj = MLP(512, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        N, V = x.shape[:2]
        emb = self.backbone(x.flatten(0, 1))
        return (
            emb,
            self.c_proj(emb).reshape(N, V, -1).transpose(0, 1),
            self.u_proj(emb).reshape(N, V, -1).transpose(0, 1),
        )


class BertEncoder(nn.Module):
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
        super().__init__()

        config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
        )
        self.backbone = BertModel(config)
        latent_dim = hidden_size

        self.c_proj = MLP(latent_dim, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)
        self.u_proj = MLP(latent_dim, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)

    def forward(self, x: torch.Tensor | dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
        if isinstance(x, torch.Tensor):
            input_ids = x
            attention_mask = torch.ones_like(input_ids)
        else:
            input_ids = x["input_ids"]
            attention_mask = x.get("attention_mask", torch.ones_like(input_ids))

        token_emb = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(dtype=token_emb.dtype)
        emb = (token_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return emb, self.c_proj(emb), self.u_proj(emb)


if __name__ == "__main__":
    batch_size, num_views, image_size = 2, 2, 128
    vit = ViTEncoder(img_size=image_size, proj_dim=64)
    images = torch.randn(batch_size, num_views, 3, image_size, image_size)
    image_outputs = vit(images)
    expected_image_shapes = ((batch_size * num_views, 512), (num_views, batch_size, 64), (num_views, batch_size, 64))
    assert tuple(tuple(output.shape) for output in image_outputs) == expected_image_shapes
    print("ViTEncoder:", [tuple(output.shape) for output in image_outputs])

    bert = BertEncoder(proj_dim=64)
    tokens = {
        "input_ids": torch.randint(0, 30_522, (batch_size, 16)),
        "attention_mask": torch.ones(batch_size, 16, dtype=torch.long),
    }
    text_outputs = bert(tokens)
    expected_text_shapes = ((batch_size, 256), (batch_size, 64), (batch_size, 64))
    assert tuple(tuple(output.shape) for output in text_outputs) == expected_text_shapes
    print("BertEncoder:", [tuple(output.shape) for output in text_outputs])
