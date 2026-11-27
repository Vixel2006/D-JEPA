import timm
import torch
import torch.nn as nn
from transformers import BertConfig, BertModel


class ViTBackbone(nn.Module):
    def __init__(
        self,
        model_name: str = "vit_small_patch8_224",
        img_size: int = 128,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=512,
            drop_path_rate=0.1,
            img_size=img_size,
        )

    def encode_backbone(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x.flatten(0, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode_backbone(x)


class BertBackbone(nn.Module):
    def __init__(
        self,
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

    def encode_backbone(
        self, x: torch.Tensor | dict[str, torch.Tensor]
    ) -> torch.Tensor:
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
        return (token_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)

    def forward(self, x: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        return self.encode_backbone(x)
