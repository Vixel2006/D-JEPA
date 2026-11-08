"""D-JEPA: shared components, private fibres, and a synergy head."""

from __future__ import annotations

import torch
from torch import nn

from ..config import ModelConfig
from .backbones import TextEncoderBERT, VisionEncoderDINO
from .projectors import LeJEPAProjector, SynergyHead, make_predictor


class DecomposedLeJEPA(nn.Module):
    def __init__(
        self,
        vision_dim: int = 384,
        text_dim: int = 384,
        component_dim: int = 256,
        synergy_dim: int = 128,
        vision_backbone: str = "dinov2_vits14",
        text_backbone: str = "distilbert-base-uncased",
    ) -> None:
        super().__init__()
        self.component_dim = component_dim
        self.synergy_dim = synergy_dim

        self.vision_encoder = VisionEncoderDINO(vision_backbone)
        self.text_encoder = TextEncoderBERT(text_backbone, output_dim=text_dim)

        self.vision_c = LeJEPAProjector(vision_dim, component_dim)
        self.vision_u = LeJEPAProjector(vision_dim, component_dim)
        self.text_c = LeJEPAProjector(text_dim, component_dim)
        self.text_u = LeJEPAProjector(text_dim, component_dim)

        self.vision_predictor = make_predictor(
            input_dim=component_dim * 2,
            hidden_dim=component_dim,
            output_dim=vision_dim,
        )
        self.text_predictor = make_predictor(
            input_dim=component_dim * 2,
            hidden_dim=component_dim,
            output_dim=text_dim,
        )

        self.fusion = SynergyHead(
            input_dim=component_dim * 2,
            hidden_dim=component_dim * 2,
            output_dim=synergy_dim,
        )

    def forward_vision(
        self,
        view1: torch.Tensor,
        view2: torch.Tensor,
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
    ]:
        hidden1 = self.vision_encoder(view1)
        hidden2 = self.vision_encoder(view2)
        return (
            (self.vision_c(hidden1), self.vision_u(hidden1)),
            (self.vision_c(hidden2), self.vision_u(hidden2)),
            (hidden1, hidden2),
        )

    def forward_text(
        self,
        ids1: torch.Tensor,
        mask1: torch.Tensor,
        ids2: torch.Tensor,
        mask2: torch.Tensor,
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
    ]:
        hidden1 = self.text_encoder(ids1, mask1)
        hidden2 = self.text_encoder(ids2, mask2)
        return (
            (self.text_c(hidden1), self.text_u(hidden1)),
            (self.text_c(hidden2), self.text_u(hidden2)),
            (hidden1, hidden2),
        )

    def forward_vision_single(
        self,
        image: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.vision_encoder(image)
        return self.vision_c(hidden), self.vision_u(hidden), hidden

    def forward_text_single(
        self,
        ids: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.text_encoder(ids, mask)
        return self.text_c(hidden), self.text_u(hidden), hidden

    def extract_synergy(
        self,
        vision_components: torch.Tensor,
        text_components: torch.Tensor,
    ) -> torch.Tensor:
        centered_vision = vision_components - vision_components.mean(
            dim=0, keepdim=True
        )
        centered_text = text_components - text_components.mean(
            dim=0, keepdim=True
        )
        return self.fusion(torch.cat([centered_vision, centered_text], dim=-1))

    @torch.no_grad()
    def extract_features(
        self,
        image: torch.Tensor,
        question_ids: torch.Tensor,
        question_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        vision_hidden = self.vision_encoder(image)
        text_hidden = self.text_encoder(question_ids, question_mask)

        vision_c = self.vision_c(vision_hidden)
        vision_u = self.vision_u(vision_hidden)
        text_c = self.text_c(text_hidden)
        text_u = self.text_u(text_hidden)

        return {
            "c1": vision_c,
            "c2": text_c,
            "c": 0.5 * (vision_c + text_c),
            "u1": vision_u,
            "u2": text_u,
            "s": self.extract_synergy(vision_c, text_c),
        }


def build_decomposed_model(config: ModelConfig) -> DecomposedLeJEPA:
    return DecomposedLeJEPA(
        vision_dim=config.vision_dim,
        text_dim=config.text_dim,
        component_dim=config.component_dim,
        synergy_dim=config.synergy_dim,
        vision_backbone=config.vision_backbone,
        text_backbone=config.text_backbone,
    )
