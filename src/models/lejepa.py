from __future__ import annotations

import torch
from torch import nn

from ..config import ModelConfig
from .backbones import TextEncoderBERT, VisionEncoderDINO
from .projectors import LeJEPAProjector


class LeJEPA(nn.Module):
    def __init__(
        self,
        vision_dim: int = 384,
        text_dim: int = 384,
        projection_dim: int = 512,
        vision_backbone: str = "dinov2_vits14",
        text_backbone: str = "distilbert-base-uncased",
    ) -> None:
        super().__init__()
        self.vision_encoder = VisionEncoderDINO(vision_backbone)
        self.text_encoder = TextEncoderBERT(text_backbone, output_dim=text_dim)
        self.vision_projector = LeJEPAProjector(vision_dim, projection_dim)
        self.text_projector = LeJEPAProjector(text_dim, projection_dim)

    def forward_vision(
        self,
        view1: torch.Tensor,
        view2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.vision_projector(self.vision_encoder(view1)),
            self.vision_projector(self.vision_encoder(view2)),
        )

    def forward_text(
        self,
        ids1: torch.Tensor,
        mask1: torch.Tensor,
        ids2: torch.Tensor,
        mask2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.text_projector(self.text_encoder(ids1, mask1)),
            self.text_projector(self.text_encoder(ids2, mask2)),
        )

    def forward_text_clean(
        self,
        ids: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.text_projector(self.text_encoder(ids, mask))

    @torch.no_grad()
    def extract_features(
        self,
        image: torch.Tensor,
        question_ids: torch.Tensor,
        question_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        z1 = self.vision_projector(self.vision_encoder(image))
        z2 = self.text_projector(self.text_encoder(question_ids, question_mask))
        return {
            "lejepa": torch.cat([z1, z2], dim=-1),
            "z1": z1,
            "z2": z2,
        }


def build_lejepa_model(config: ModelConfig) -> LeJEPA:
    return LeJEPA(
        vision_dim=config.vision_dim,
        text_dim=config.text_dim,
        projection_dim=config.projection_dim,
        vision_backbone=config.vision_backbone,
        text_backbone=config.text_backbone,
    )
