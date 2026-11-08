"""Typed configuration for the CLEVR / D-JEPA benchmark."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class DataConfig:
    """Dataset and dataloader settings.

    Two data sources are supported:

    ``procedural``
        The original synthetic scene generator.  Fast, fully offline, and
        useful for tests and ablations.
    ``clevr``
        The real Blender-rendered CLEVR split distributed through
        ``HuggingFaceM4/the_cauldron``.  Images are streamed so that the
        benchmark can be run without downloading the full dataset first.
    """

    source: str = "procedural"
    num_scenes: int = 1500
    max_samples: int = 10_000
    hf_dataset: str = "HuggingFaceM4/the_cauldron"
    hf_config: Optional[str] = "clevr"
    hf_split: str = "train"
    hf_streaming: bool = True
    fallback_to_procedural: bool = True

    batch_size: int = 64
    num_workers: int = 0
    max_question_length: int = 64
    answer_vocab_size: int = 40
    mask_ratio: float = 0.15
    image_size: int = 224
    patch_size: int = 32
    patch_drop_ratio: float = 0.30
    aug_scale_min: float = 0.7

    # ``None`` reproduces the original split: min(800, len(dataset) // 4).
    test_size: Optional[int] = None
    seed: int = 42


@dataclass(frozen=True)
class ModelConfig:
    """Architecture settings shared by the two models."""

    vision_dim: int = 384
    text_dim: int = 384
    projection_dim: int = 512
    component_dim: int = 256
    synergy_dim: int = 128
    probe_bottleneck_dim: int = 128
    vision_backbone: str = "dinov2_vits14"
    text_backbone: str = "distilbert-base-uncased"


@dataclass(frozen=True)
class OptimizationConfig:
    """Optimisation and loss settings."""

    pretrain_lr: float = 5e-4
    pretrain_weight_decay: float = 0.05
    probe_lr: float = 1e-3
    probe_weight_decay: float = 1e-4
    pretrain_steps: int = 3000
    probe_steps: int = 600
    sigreg_weight: float = 0.02
    sigreg_slices: int = 512
    synergy_sigreg_slices: int = 256
    amp: bool = True
    log_every: int = 50


@dataclass(frozen=True)
class FigureConfig:
    """Publication-figure settings."""

    enabled: bool = True
    dpi: int = 200
    save_pdf: bool = True
    tsne_perplexity: int = 30
    tsne_max_iter: int = 1000
    max_qualitative: int = 6
    seed: int = 42


@dataclass(frozen=True)
class RunConfig:
    """Top-level run configuration."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimizationConfig = field(default_factory=OptimizationConfig)
    figures: FigureConfig = field(default_factory=FigureConfig)
    seeds: tuple[int, ...] = (42, 43, 44)
    device: str = "auto"
    output_dir: Path = Path("outputs/clevr")
    save_results: bool = True
    run_case_studies: bool = True
    collect_diagnostics: bool = True
    log_level: str = "INFO"

    @property
    def figure_dir(self) -> Path:
        return Path(self.output_dir) / "figures"
