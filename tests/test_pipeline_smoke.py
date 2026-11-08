"""End-to-end smoke test with tiny fake backbones.

This exercises dataset construction, pretraining, probe training, all four
evaluations, case-study search, and result serialization without downloading
DINOv2 or DistilBERT.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

import djepa.pipeline as pipeline
from djepa.config import DataConfig, ModelConfig, OptimizationConfig, RunConfig


class FakeTokenizer:
    mask_token_id = 103
    pad_token_id = 0
    cls_token_id = 101
    sep_token_id = 102

    def __call__(
        self,
        text: str,
        padding: str = "max_length",
        truncation: bool = True,
        max_length: int = 24,
        return_tensors: str = "pt",
    ):
        tokens = [self.cls_token_id]
        tokens.extend(1 + (ord(character) % 100) for character in text)
        tokens.append(self.sep_token_id)
        tokens = tokens[:max_length]
        attention = [1] * len(tokens)
        while len(tokens) < max_length:
            tokens.append(self.pad_token_id)
            attention.append(0)
        return {
            "input_ids": torch.tensor([tokens], dtype=torch.long),
            "attention_mask": torch.tensor([attention], dtype=torch.long),
        }


class FakeLeJEPA(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.vision_encoder = nn.Linear(3, config.vision_dim)
        self.text_encoder = nn.Linear(4, config.text_dim)
        self.vision_projector = nn.Linear(
            config.vision_dim, config.projection_dim
        )
        self.text_projector = nn.Linear(
            config.text_dim, config.projection_dim
        )

    def _image(self, images: torch.Tensor) -> torch.Tensor:
        pooled = images.mean(dim=(2, 3))
        return self.vision_projector(self.vision_encoder(pooled))

    def _text(self, ids: torch.Tensor) -> torch.Tensor:
        return self.text_projector(self.text_encoder(ids.float()[:, :4]))

    def forward_vision(self, view1, view2):
        return self._image(view1), self._image(view2)

    def forward_text(self, ids1, mask1, ids2, mask2):
        return self._text(ids1), self._text(ids2)

    @torch.no_grad()
    def extract_features(self, image, question_ids, question_mask):
        z1 = self._image(image)
        z2 = self._text(question_ids)
        return {
            "lejepa": torch.cat([z1, z2], dim=-1),
            "z1": z1,
            "z2": z2,
        }


class FakeDecomposed(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        comp = config.component_dim
        self.vision_encoder = nn.Linear(3, config.vision_dim)
        self.text_encoder = nn.Linear(4, config.text_dim)
        self.vision_c = nn.Linear(config.vision_dim, comp)
        self.vision_u = nn.Linear(config.vision_dim, comp)
        self.text_c = nn.Linear(config.text_dim, comp)
        self.text_u = nn.Linear(config.text_dim, comp)
        self.vision_predictor = nn.Linear(comp * 2, config.vision_dim)
        self.text_predictor = nn.Linear(comp * 2, config.text_dim)
        self.fusion = nn.Linear(comp * 2, config.synergy_dim)

    def _image(self, images: torch.Tensor) -> torch.Tensor:
        return self.vision_encoder(images.mean(dim=(2, 3)))

    def _text(self, ids: torch.Tensor) -> torch.Tensor:
        return self.text_encoder(ids.float()[:, :4])

    def forward_vision(self, view1, view2):
        hidden1 = self._image(view1)
        hidden2 = self._image(view2)
        return (
            (self.vision_c(hidden1), self.vision_u(hidden1)),
            (self.vision_c(hidden2), self.vision_u(hidden2)),
            (hidden1, hidden2),
        )

    def forward_text(self, ids1, mask1, ids2, mask2):
        hidden1 = self._text(ids1)
        hidden2 = self._text(ids2)
        return (
            (self.text_c(hidden1), self.text_u(hidden1)),
            (self.text_c(hidden2), self.text_u(hidden2)),
            (hidden1, hidden2),
        )

    def extract_synergy(self, vision_components, text_components):
        vision_centered = vision_components - vision_components.mean(
            dim=0, keepdim=True
        )
        text_centered = text_components - text_components.mean(
            dim=0, keepdim=True
        )
        return self.fusion(torch.cat([vision_centered, text_centered], dim=-1))

    @torch.no_grad()
    def extract_features(self, image, question_ids, question_mask):
        vision_hidden = self._image(image)
        text_hidden = self._text(question_ids)
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


def test_pipeline_smoke(monkeypatch, tmp_path: Path):
    model_config = ModelConfig(
        vision_dim=4,
        text_dim=4,
        projection_dim=4,
        component_dim=4,
        synergy_dim=2,
        probe_bottleneck_dim=2,
    )
    monkeypatch.setattr(
        pipeline, "load_tokenizer", lambda name: FakeTokenizer()
    )
    monkeypatch.setattr(
        pipeline,
        "build_lejepa_model",
        lambda config: FakeLeJEPA(config),
    )
    monkeypatch.setattr(
        pipeline,
        "build_decomposed_model",
        lambda config: FakeDecomposed(config),
    )

    config = RunConfig(
        data=DataConfig(
            num_scenes=12,
            batch_size=4,
            test_size=4,
        ),
        model=model_config,
        optim=OptimizationConfig(
            pretrain_steps=2,
            probe_steps=2,
            amp=False,
        ),
        seeds=(42,),
        device="cpu",
        output_dir=tmp_path / "clevr",
        save_results=True,
        run_case_studies=True,
    )

    results = pipeline.run(config)

    assert len(results.seed_metrics) == 1
    assert len(results.pid_rows) == 4
    assert (tmp_path / "clevr" / "results.json").exists()
    assert (tmp_path / "clevr" / "accuracy.csv").exists()
    assert (tmp_path / "clevr" / "pid.csv").exists()
