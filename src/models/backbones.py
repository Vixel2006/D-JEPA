from __future__ import annotations

import logging

import torch
from torch import nn

LOGGER = logging.getLogger(__name__)


class VisionEncoderDINO(nn.Module):
    """DINOv2 ViT-S/14 with layered fallbacks.

    Loading order:

    1. ``torch.hub`` (retried once to absorb transient GitHub/network errors)
    2. Hugging Face ``facebook/dinov2-small`` via ``transformers``
    3. A frozen ImageNet ResNet-18

    All branches are frozen and produce a 384-dimensional CLS feature.
    """

    def __init__(self, model_name: str = "dinov2_vits14") -> None:
        super().__init__()
        self._hf_model: nn.Module | None = None
        self.model: nn.Module | None = None

        for attempt in range(2):
            try:
                LOGGER.info(
                    "Loading DINOv2 %s from torch.hub (attempt %d)...",
                    model_name,
                    attempt + 1,
                )
                self.model = torch.hub.load(
                    "facebookresearch/dinov2",
                    model_name,
                    trust_repo=True,
                )
                break
            except Exception as exc:  # pragma: no cover - network dependent
                LOGGER.warning(
                    "DINOv2 torch.hub load failed (attempt %d/2): %s",
                    attempt + 1,
                    exc,
                )

        if self.model is None:
            try:
                from transformers import AutoModel

                LOGGER.info("Trying Hugging Face DINOv2 fallback...")
                self._hf_model = AutoModel.from_pretrained("facebook/dinov2-small")
            except Exception as exc:  # pragma: no cover - network dependent
                LOGGER.warning(
                    "Hugging Face DINOv2 fallback failed (%s); "
                    "falling back to ResNet-18.",
                    exc,
                )
                from torchvision.models import ResNet18_Weights, resnet18

                backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
                self.model = nn.Sequential(
                    backbone.conv1,
                    backbone.bn1,
                    backbone.relu,
                    backbone.maxpool,
                    backbone.layer1,
                    backbone.layer2,
                    backbone.layer3,
                    backbone.layer4,
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(512, 384),
                )

        active = self.model if self.model is not None else self._hf_model
        assert active is not None
        for parameter in active.parameters():
            parameter.requires_grad = False
        self.eval()

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self._hf_model is not None:
            outputs = self._hf_model(pixel_values=images)
            return outputs.last_hidden_state[:, 0]
        assert self.model is not None
        return self.model(images)


class TextEncoderBERT(nn.Module):
    """Frozen DistilBERT CLS encoder with a trainable projection."""

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        output_dim: int = 384,
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "The CLEVR experiment requires `transformers`. "
                "Install it with `pip install transformers`."
            ) from exc

        self.bert = AutoModel.from_pretrained(model_name)
        for parameter in self.bert.parameters():
            parameter.requires_grad = False

        hidden_size = int(
            getattr(
                self.bert.config,
                "hidden_size",
                getattr(self.bert.config, "dim", 768),
            )
        )
        self.proj = nn.Linear(hidden_size, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        cls_token = outputs.last_hidden_state[:, 0]
        return self.norm(self.proj(cls_token))
