from __future__ import annotations

import os
import random
from pathlib import Path
from collections.abc import Mapping
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from .huggingface import (
    flickr_test_pairs,
    load_cmu_mosi,
    load_flickr30k,
    load_mm_imdb,
    load_mustard,
    load_ur_funny,
)

_TEXT_COLUMNS = (
    "text",
    "utterance",
    "sentence",
    "transcript",
    "caption",
    "captions",
    "plot",
    "synopsis",
    "premise",
)
_LABEL_COLUMNS = ("label", "labels", "genre", "genres", "sentiment", "sarcasm", "humor", "target")
_ID_COLUMNS = ("image_id", "video_id", "id", "clip_id", "movie_id")


class PairedDataset(Dataset):
    """Convert one Hugging Face split to augmented PyTorch training samples.

    Each item contains ``context`` and ``target`` dictionaries for the
    modalities present in the source, along with its source label and ID.
    Images use independent random crops, flips, and color jitter for training;
    text uses independent token masking. Evaluation transforms are fixed.
    """
    def __init__(
        self,
        dataset,
        *,
        training: bool = True,
        image_size: int = 128,
        max_text_length: int = 128,
        tokenizer: PreTrainedTokenizerBase | None = None,
        tokenizer_name: str = "bert-base-uncased",
        text_mask_probability: float = 0.08,
        image_cache_path: str | Path | None = None,
    ):
        if not 0 <= text_mask_probability <= 1:
            raise ValueError("text_mask_probability must be between 0 and 1")
        self.dataset = dataset
        self.training = training
        self.image_size = image_size
        self.max_text_length = max_text_length
        self.text_mask_probability = text_mask_probability
        self.image_cache_path = Path(image_cache_path) if image_cache_path else None
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(tokenizer_name)
        self.text_column = self._find_column(_TEXT_COLUMNS)
        self.image_column = "image" if "image" in dataset.column_names else None
        self.label_column = self._find_column(_LABEL_COLUMNS)
        self.id_column = self._find_column(_ID_COLUMNS)

        if self.text_column is None and self.image_column is None:
            raise ValueError(
                "Dataset must contain a recognized text or image column; "
                f"found {dataset.column_names}."
            )

        if training:
            self.image_transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        image_size,
                        scale=(0.75, 1.0),
                        ratio=(0.85, 1.15),
                        interpolation=InterpolationMode.BICUBIC,
                    ),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(
                        brightness=0.15, contrast=0.15, saturation=0.12, hue=0.02
                    ),
                    transforms.ToTensor(),
                ]
            )
        else:
            self.image_transform = transforms.Compose(
                [
                    transforms.Resize(
                        image_size,
                        interpolation=InterpolationMode.BICUBIC,
                    ),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                ]
            )
        self._cached_images = self._open_or_create_image_cache()

    def _find_column(self, candidates: tuple[str, ...]) -> str | None:
        columns = set(self.dataset.column_names)
        return next((column for column in candidates if column in columns), None)

    def __len__(self) -> int:
        return len(self.dataset)

    def _text(self, value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            # Flickr30k has five references per image. Sample a caption for
            # training and use the first reference for repeatable evaluation.
            captions = [str(caption) for caption in value if caption]
            if not captions:
                return ""
            return random.choice(captions) if self.training else captions[0]
        return str(value)

    def _token_ids(self, text: str) -> torch.Tensor:
        encoded = self.tokenizer(
            text,
            max_length=self.max_text_length,
            truncation=True,
            padding="max_length",
            return_attention_mask=False,
            return_tensors="pt",
        )
        tokens = encoded["input_ids"].squeeze(0)
        if (
            self.training
            and self.text_mask_probability
            and self.tokenizer.mask_token_id is not None
        ):
            special = torch.zeros_like(tokens, dtype=torch.bool)
            for special_id in self.tokenizer.all_special_ids:
                special |= tokens.eq(special_id)
            selected = (
                torch.rand(tokens.shape) < self.text_mask_probability
            ) & ~special
            tokens[selected] = self.tokenizer.mask_token_id
        return tokens

    def _image_view(self, image: Any) -> torch.Tensor:
        if not isinstance(image, Image.Image):
            # Hugging Face image columns can be configured to decode lazily.
            image = Image.open(image["path"] if isinstance(image, Mapping) else image)
        image = image.convert("RGB")
        return self.image_transform(image).permute(1, 2, 0).contiguous()

    def _open_or_create_image_cache(self):
        if self.image_cache_path is None or self.image_column is None:
            return None

        import numpy as np

        path = self.image_cache_path
        expected_shape = (len(self), 2, self.image_size, self.image_size, 3)
        if path.exists():
            try:
                cache = np.load(path, mmap_mode="r")
                if cache.shape == expected_shape and cache.dtype == np.uint8:
                    print(f"Using cached images: {path}")
                    return cache
            except (OSError, ValueError):
                pass
            print(f"Image cache shape mismatch; rebuilding: {path}")

        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        cache = np.lib.format.open_memmap(
            temporary_path, mode="w+", dtype=np.uint8, shape=expected_shape
        )
        for index in range(len(self)):
            image = self.dataset[index][self.image_column]
            for view in range(2):
                tensor = self._image_view(image)
                cache[index, view] = (
                    tensor.numpy().clip(0, 1) * 255
                ).round().astype(np.uint8)
        cache.flush()
        del cache
        os.replace(temporary_path, path)
        print(f"Cached {len(self)} image pairs: {path}")
        return np.load(path, mmap_mode="r")

    @staticmethod
    def _label(value: Any) -> Any:
        if isinstance(value, bool):
            return torch.tensor(value, dtype=torch.long)
        if isinstance(value, (int, float)):
            return torch.tensor(value, dtype=torch.float32)
        if isinstance(value, str):
            try:
                return torch.tensor(float(value), dtype=torch.float32)
            except ValueError:
                return value
        if isinstance(value, (list, tuple)):
            # Multi-label genre lists have variable length; joining keeps
            # default PyTorch collation valid for batches of unlike lengths.
            return "|".join(str(label) for label in value)
        return value

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self._cached_images is None:
            row = self.dataset[index]
        else:
            columns = [self.text_column, self.label_column, self.id_column]
            row = {
                column: self.dataset[column][index]
                for column in columns if column is not None
            }
        context: dict[str, torch.Tensor] = {}
        target: dict[str, torch.Tensor] = {}

        if self.image_column:
            if self._cached_images is None:
                image = row[self.image_column]
                context["vision"] = self._image_view(image)
                target["vision"] = self._image_view(image)
            else:
                cached = torch.from_numpy(self._cached_images[index].copy()).float() / 255
                context["vision"], target["vision"] = cached[0], cached[1]

        if self.text_column:
            text = self._text(row[self.text_column])
            if text is not None:
                context["text"] = self._token_ids(text)
                target["text"] = self._token_ids(text)

        result: dict[str, Any] = {
            "context": context,
            "target": target,
            "modalities": tuple(context.keys()),
        }
        if self.label_column:
            result["label"] = self._label(row[self.label_column])
        if self.id_column:
            result["id"] = row[self.id_column]
        return result


class CMUMOSIDataset(PairedDataset):
    """PyTorch view of a CMU-MOSI split (text and sentiment label)."""

    def __init__(self, split: str = "train", **kwargs):
        super().__init__(load_cmu_mosi(split), **kwargs)


class MUSTARDDataset(PairedDataset):
    """PyTorch view of the MUStARD sarcasm benchmark split."""

    def __init__(self, split: str = "train", **kwargs):
        super().__init__(load_mustard(split), **kwargs)


class URFunnyDataset(PairedDataset):
    """PyTorch view of a UR-FUNNY v2 split."""

    def __init__(self, split: str = "train", **kwargs):
        super().__init__(load_ur_funny(split), **kwargs)


class MMIMDbDataset(PairedDataset):
    """PyTorch view of MM-IMDb poster, plot, and genre samples."""

    def __init__(self, split: str = "train", **kwargs):
        if split in {"train", "validation", "val"}:
            source = load_mm_imdb("train")
            partition = source.train_test_split(test_size=0.1, seed=42)
            source = partition["train" if split == "train" else "test"]
        else:
            source = load_mm_imdb(split)
        super().__init__(source, **kwargs)


class Flickr30kDataset(PairedDataset):
    """PyTorch view of Flickr30k's Karpathy image/caption split.

    Set ``all_test_captions=True`` for the 5,000 Karpathy test pairs rather
    than one image row per item. Training rows continue to sample among their
    five captions in :class:`PairedDataset`.
    """

    def __init__(
        self,
        split: str = "train",
        *,
        all_test_captions: bool = False,
        **kwargs,
    ):
        source = (
            flickr_test_pairs()
            if split == "test" and all_test_captions
            else load_flickr30k(split)
        )
        super().__init__(source, **kwargs)


class SyntheticDataset(Dataset):
    """Paired image/text views with known shared and private factors."""

    def __init__(
        self,
        n_samples: int = 1024,
        seed: int = 0,
        split: str = "train",
        train_fraction: float = 0.8,
    ):
        if split not in {"train", "validation", "all"}:
            raise ValueError("split must be 'train', 'validation', or 'all'.")

        generator = torch.Generator().manual_seed(seed)
        shared = torch.randn(n_samples, 4, generator=generator)
        private_vision = torch.randn(n_samples, 4, generator=generator)
        private_text = torch.randn(n_samples, 4, generator=generator)
        nuisance_vision = torch.randn(n_samples, 1, generator=generator)
        nuisance_text = torch.randn(n_samples, 1, generator=generator)

        vision_latent = torch.cat((shared, private_vision, nuisance_vision), dim=-1)
        text_latent = torch.cat((shared, private_text, nuisance_text), dim=-1)

        self.context = {
            "vision": self._make_images(vision_latent, generator),
            "text": self._make_text(text_latent, generator),
        }
        self.target = {
            "vision": self._make_images(vision_latent, generator),
            "text": self._make_text(text_latent, generator),
        }
        self.factors = {
            "shared": shared,
            "private_vision": private_vision,
            "private_text": private_text,
            "nuisance_vision": nuisance_vision,
            "nuisance_text": nuisance_text,
            "synergy": torch.sign(private_vision[:, 0])
            * torch.sign(private_text[:, 0])
            + 0.05 * torch.randn(n_samples, generator=generator),
        }

        permutation = torch.randperm(n_samples, generator=generator)
        split_at = int(n_samples * train_fraction)
        if split == "train":
            self.indices = permutation[:split_at]
        elif split == "validation":
            self.indices = permutation[split_at:]
        else:
            self.indices = torch.arange(n_samples)

    @staticmethod
    def _make_images(latent, generator):
        values = torch.tanh(latent)
        image = torch.zeros(len(latent), 32, 32, 3)
        for coordinate in range(values.shape[1]):
            row, column = divmod(coordinate, 5)
            channel = coordinate % 3
            image[:, row * 16 : (row + 1) * 16, column * 6 : (column + 1) * 6, channel] = (
                values[:, coordinate, None, None]
            )
        noise = 0.05 * torch.randn(image.shape, generator=generator)
        return (image + noise).clamp(-1, 1)

    @staticmethod
    def _make_text(latent, generator):
        noisy_latent = latent + 0.03 * torch.randn(latent.shape, generator=generator)
        coordinates = torch.arange(latent.shape[1]).repeat(2)
        values = noisy_latent[:, coordinates]
        bins = ((torch.tanh(values) + 1) * 16).long().clamp(0, 31)
        return bins + coordinates * 32

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        row = self.indices[index].item()
        return {
            "context": {name: value[row] for name, value in self.context.items()},
            "target": {name: value[row] for name, value in self.target.items()},
            "factors": {name: value[row] for name, value in self.factors.items()},
        }


def make_dataset_pair(
    name: str,
    *,
    seed: int,
    n_samples: int = 1024,
    image_size: int = 128,
    max_text_length: int = 128,
    tokenizer_name: str = "bert-base-uncased",
    text_mask_probability: float = 0.08,
    image_cache_dir: str | Path | None = None,
):
    """Build train/validation datasets for one paired-modality experiment.

    The current D-JEPA and FactorCL trainers require both image and text
    tensors. MOSI, MUStARD, and UR-FUNNY Hub exports currently provide text and
    labels only, so those datasets need video frame extraction before they can
    be used by these two methods.
    """
    normalized = name.lower().replace("-", "_")
    if normalized == "synthetic":
        return (
            SyntheticDataset(n_samples=n_samples, seed=seed, split="train"),
            SyntheticDataset(n_samples=n_samples, seed=seed, split="validation"),
        )

    supported = {
        "flickr30k": Flickr30kDataset,
        "mm_imdb": MMIMDbDataset,
    }
    if normalized not in supported:
        if normalized in {"cmu_mosi", "mosi", "mustard", "ur_funny"}:
            raise ValueError(
                f"{name} is currently text-only in the Hugging Face loader. "
                "D-JEPA and FactorCL require paired vision and text inputs; "
                "extract/load video frames before training on this dataset."
            )
        raise ValueError(
            f"Unknown or unsupported paired dataset {name!r}. "
            f"Choose one of: {', '.join(supported)} or 'synthetic'."
        )

    options = {
        "training": True,
        "image_size": image_size,
        "max_text_length": max_text_length,
        "tokenizer_name": tokenizer_name,
        "text_mask_probability": text_mask_probability,
    }
    if image_cache_dir is not None:
        cache_dir = Path(image_cache_dir)
        cache_tag = f"{normalized}_{image_size}"
        options["image_cache_path"] = cache_dir / f"{cache_tag}_train.npy"
    validation_options = {**options, "training": False}
    if image_cache_dir is not None:
        validation_options["image_cache_path"] = (
            Path(image_cache_dir) / f"{cache_tag}_validation.npy"
        )
    dataset_type = supported[normalized]
    train = dataset_type("train", **options)
    validation = dataset_type("validation", **validation_options)
    return train, validation


__all__ = [
    "CMUMOSIDataset",
    "Flickr30kDataset",
    "MMIMDbDataset",
    "MUSTARDDataset",
    "make_dataset_pair",
    "PairedDataset",
    "SyntheticDataset",
    "URFunnyDataset",
]
