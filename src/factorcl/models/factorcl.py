from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...models import BertBackbone, ViTBackbone
from ..loss import CLUBInfoNCECritic, InfoNCECritic


def _projection(input_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, input_dim),
        nn.ReLU(),
        nn.Linear(input_dim, input_dim),
        nn.ReLU(),
        nn.Linear(input_dim, input_dim),
    )


class FactorCLModel(nn.Module):
    """FactorCLSSL model with vision and text Transformer backbones."""

    def __init__(
        self,
        image_size: int = 128,
        vision_dim: int = 512,
        text_dim: int = 256,
        critic_hidden_dim: int = 512,
    ):
        super().__init__()
        self.image_size = image_size
        self.vision_backbone = ViTBackbone(img_size=image_size)
        self.text_backbone = BertBackbone(hidden_size=text_dim)

        self.shared_projection = nn.ModuleList(
            [_projection(vision_dim), _projection(text_dim)]
        )
        self.unique_projection = nn.ModuleList(
            [_projection(vision_dim), _projection(text_dim)]
        )
        self.conditional_nce_projection = nn.ModuleList(
            [_projection(vision_dim), _projection(text_dim)]
        )
        self.conditional_club_projection = nn.ModuleList(
            [_projection(vision_dim), _projection(text_dim)]
        )
        self.unconditional_club_projection = nn.ModuleList(
            [_projection(vision_dim), _projection(text_dim)]
        )

        self.nce_critics = nn.ModuleDict(
            {
                "shared": InfoNCECritic(vision_dim, text_dim, critic_hidden_dim),
                "unique_vision": InfoNCECritic(
                    vision_dim, vision_dim, critic_hidden_dim
                ),
                "unique_text": InfoNCECritic(text_dim, text_dim, critic_hidden_dim),
                "conditional": InfoNCECritic(
                    2 * vision_dim, 2 * text_dim, critic_hidden_dim
                ),
            }
        )
        self.club_critics = nn.ModuleDict(
            {
                "unconditional": CLUBInfoNCECritic(
                    vision_dim, text_dim, critic_hidden_dim
                ),
                "conditional": CLUBInfoNCECritic(
                    2 * vision_dim, 2 * text_dim, critic_hidden_dim
                ),
            }
        )

    def encode_vision(self, images: torch.Tensor) -> torch.Tensor:
        images = images.permute(0, 3, 1, 2)
        images = F.interpolate(
            images, size=(self.image_size, self.image_size), mode="bilinear"
        )
        return self.vision_backbone.encode_backbone(images.unsqueeze(1))

    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        return self.text_backbone.encode_backbone(text)

    def forward(self, images, text, augmented_images, augmented_text):
        vision = self.encode_vision(images)
        language = self.encode_text(text)
        vision_aug = self.encode_vision(augmented_images)
        text_aug = self.encode_text(augmented_text)

        shared_nce = self.nce_critics["shared"](
            self.shared_projection[0](vision), self.shared_projection[1](language)
        )
        unconditional_club = self.club_critics["unconditional"](
            self.unconditional_club_projection[0](vision),
            self.unconditional_club_projection[1](language),
        )
        unique_vision_nce = self.nce_critics["unique_vision"](
            self.unique_projection[0](vision),
            self.unique_projection[0](vision_aug),
        )
        unique_text_nce = self.nce_critics["unique_text"](
            self.unique_projection[1](language),
            self.unique_projection[1](text_aug),
        )
        conditional_nce = self.nce_critics["conditional"](
            self._concatenate_views(
                self.conditional_nce_projection[0], vision, vision_aug
            ),
            self._concatenate_views(
                self.conditional_nce_projection[1], language, text_aug
            ),
        )
        conditional_club = self.club_critics["conditional"](
            self._concatenate_views(
                self.conditional_club_projection[0], vision, vision_aug
            ),
            self._concatenate_views(
                self.conditional_club_projection[1], language, text_aug
            ),
        )

        losses = {
            "shared_nce": shared_nce,
            "unconditional_club": unconditional_club,
            "unique_vision_nce": unique_vision_nce,
            "unique_text_nce": unique_text_nce,
            "conditional_nce": conditional_nce,
            "conditional_club": conditional_club,
        }
        return sum(losses.values()), losses

    @staticmethod
    def _concatenate_views(projection, first, second):
        return torch.cat((projection(first), projection(second)), dim=-1)

    def critic_learning_loss(self, images, text, augmented_images, augmented_text):
        with torch.no_grad():
            vision = self.encode_vision(images)
            language = self.encode_text(text)
            vision_aug = self.encode_vision(augmented_images)
            text_aug = self.encode_text(augmented_text)

        unconditional = self.club_critics["unconditional"].learning_loss(
            self.unconditional_club_projection[0](vision).detach(),
            self.unconditional_club_projection[1](language).detach(),
        )
        conditional = self.club_critics["conditional"].learning_loss(
            self._concatenate_views(
                self.conditional_club_projection[0], vision, vision_aug
            ).detach(),
            self._concatenate_views(
                self.conditional_club_projection[1], language, text_aug
            ).detach(),
        )
        return unconditional + conditional

    def representations(self, images, text):
        vision = self.encode_vision(images)
        language = self.encode_text(text)
        shared_vision = self.shared_projection[0](vision)
        shared_text = self.shared_projection[1](language)
        unique_vision = self.unique_projection[0](vision)
        unique_text = self.unique_projection[1](language)
        return vision, language, shared_vision, unique_vision, shared_text, unique_text

    def non_club_parameters(self):
        modules = (
            self.vision_backbone,
            self.text_backbone,
            self.shared_projection,
            self.unique_projection,
            self.conditional_nce_projection,
            self.conditional_club_projection,
            self.unconditional_club_projection,
            self.nce_critics,
        )
        return chain.from_iterable(module.parameters() for module in modules)

    def club_parameters(self):
        return self.club_critics.parameters()
