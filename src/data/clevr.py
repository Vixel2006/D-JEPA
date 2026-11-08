"""CLEVR data sources: real Blender scenes and a procedural fallback.

The real dataset is streamed from ``HuggingFaceM4/the_cauldron`` (config
``clevr``).  Every row contains a rendered image, a natural-language question,
and the CLEVR answer.  Questions are grouped into the three reasoning
categories used by the paper: counting, attribute, and spatial relation.
"""

from __future__ import annotations

import logging
import random
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Optional

import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision import transforms

from ..augment import random_token_mask
from ..config import DataConfig
from ..utils import format_duration, progress_suffix

LOGGER = logging.getLogger(__name__)

COLORS = {
    "red": (220, 40, 40),
    "blue": (40, 80, 220),
    "green": (40, 180, 40),
    "yellow": (230, 210, 40),
    "cyan": (40, 220, 220),
    "purple": (180, 40, 220),
    "gray": (150, 150, 150),
    "brown": (140, 80, 40),
}
SHAPES = ("cube", "sphere", "cylinder")
GRID_POSITIONS = [
    (x, y) for x in range(35, 190, 45) for y in range(45, 185, 45)
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CATEGORY_COUNTING = "counting"
CATEGORY_ATTRIBUTE = "attribute"
CATEGORY_SPATIAL = "spatial_relation"


@dataclass(frozen=True)
class SceneObject:
    color: str
    shape: str
    x: int
    y: int


@dataclass(frozen=True)
class CLEVRItem:
    scene_id: int
    question: str
    answer: str
    category: str
    image: Optional[Any] = None


@dataclass(frozen=True)
class CLEVRData:
    dataset: "CLEVRDataset"
    answers: list[str]
    answer_to_index: dict[str, int]
    images: dict[int, Image.Image]
    items: list[CLEVRItem]
    source: str = "procedural"


def load_tokenizer(name: str) -> Any:
    """Load a Hugging Face tokenizer, with a helpful dependency error."""

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "The CLEVR experiment requires `transformers`. "
            "Install it with `pip install transformers`."
        ) from exc
    return AutoTokenizer.from_pretrained(name)


def make_train_transform(
    image_size: int = 224,
    scale_min: float = 0.7,
) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size, scale=(scale_min, 1.0)
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def make_eval_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


# ---------------------------------------------------------------------------
# Procedural CLEVR
# ---------------------------------------------------------------------------
def generate_procedural_clevr_scene(
    rng: Optional[random.Random] = None,
) -> tuple[Image.Image, list[dict[str, str]]]:
    """Draw one synthetic CLEVR scene and return three QA pairs."""

    rng = rng or random
    image = Image.new("RGB", (224, 224), color=(205, 205, 205))
    draw = ImageDraw.Draw(image)

    num_objects = rng.randint(3, 6)
    positions = rng.sample(GRID_POSITIONS, num_objects)
    objects: list[SceneObject] = []

    for index in range(num_objects):
        color_name = rng.choice(list(COLORS.keys()))
        shape_name = rng.choice(SHAPES)
        px, py = positions[index]
        radius = rng.randint(14, 20)

        draw.ellipse(
            [px - radius + 3, py + radius - 3,
             px + radius + 5, py + radius + 3],
            fill=(130, 130, 130),
        )
        if shape_name == "cube":
            draw.rectangle(
                [px - radius, py - radius, px + radius, py + radius],
                fill=COLORS[color_name],
                outline=(30, 30, 30),
            )
        elif shape_name == "sphere":
            draw.ellipse(
                [px - radius, py - radius, px + radius, py + radius],
                fill=COLORS[color_name],
                outline=(30, 30, 30),
            )
        else:
            draw.rectangle(
                [px - radius + 3, py - radius, px + radius - 3, py + radius],
                fill=COLORS[color_name],
                outline=(30, 30, 30),
            )
            draw.ellipse(
                [
                    px - radius + 3,
                    py - radius - 4,
                    px + radius - 3,
                    py - radius + 4,
                ],
                fill=COLORS[color_name],
                outline=(30, 30, 30),
            )

        objects.append(
            SceneObject(color=color_name, shape=shape_name, x=px, y=py)
        )

    objects.sort(key=lambda item: item.x)
    qa_list: list[dict[str, str]] = []

    target = rng.choice(objects)
    count = sum(
        1
        for item in objects
        if item.color == target.color and item.shape == target.shape
    )
    qa_list.append(
        {
            "question": (
                f"how many {target.color} {target.shape}s are there?"
            ),
            "answer": str(count),
            "category": CATEGORY_COUNTING,
        }
    )

    qa_list.append(
        {
            "question": f"what color is the {objects[0].shape} on the left?",
            "answer": objects[0].color,
            "category": CATEGORY_ATTRIBUTE,
        }
    )

    if len(objects) >= 2:
        qa_list.append(
            {
                "question": (
                    "what shape is immediately to the right of the "
                    f"{objects[0].color} {objects[0].shape}?"
                ),
                "answer": objects[1].shape,
                "category": CATEGORY_SPATIAL,
            }
        )

    return image, qa_list


# ---------------------------------------------------------------------------
# Real CLEVR from the Hugging Face Hub
# ---------------------------------------------------------------------------
def classify_clevr_question(question: str) -> str:
    """Map a CLEVR question onto one of the three paper categories."""

    lowered = question.lower()
    if "how many" in lowered:
        return CATEGORY_COUNTING
    if any(word in lowered for word in ("color", "material", "size")):
        return CATEGORY_ATTRIBUTE
    if any(
        word in lowered
        for word in ("left", "right", "behind", "front", "closer", "farther")
    ):
        return CATEGORY_SPATIAL
    # CLEVR's remaining questions are almost all spatial comparisons.
    return CATEGORY_SPATIAL


def normalize_answer(answer: str) -> str:
    return answer.replace("Answer:", "").strip().lower().rstrip(".")


def _load_hub_rows(config: DataConfig):
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "Loading the real CLEVR split requires `datasets`. "
            "Install it with `pip install datasets`."
        ) from exc

    return load_dataset(
        config.hf_dataset,
        config.hf_config,
        split=config.hf_split,
        streaming=config.hf_streaming,
    )


def build_hub_clevr_items(
    config: DataConfig,
) -> tuple[list[CLEVRItem], list[str], dict[int, Image.Image]]:
    """Stream CLEVR QA pairs from the Hugging Face Hub."""

    LOGGER.info(
        "Streaming up to %d CLEVR samples from %s/%s...",
        config.max_samples,
        config.hf_dataset,
        config.hf_config,
    )

    rows = _load_hub_rows(config)
    items: list[CLEVRItem] = []
    answer_counts: Counter[str] = Counter()
    images: dict[int, Image.Image] = {}
    started = time.perf_counter()

    for index, row in enumerate(rows):
        if index >= config.max_samples:
            break

        text_pair = row["texts"][0]
        question = (
            text_pair["user"]
            .replace("<image>", "")
            .replace("Question:", "")
            .strip()
        )
        answer = normalize_answer(text_pair["assistant"])
        pil_image = row["images"][0]
        if hasattr(pil_image, "convert"):
            pil_image = pil_image.convert("RGB")

        items.append(
            CLEVRItem(
                scene_id=index,
                question=question,
                answer=answer,
                category=classify_clevr_question(question),
                image=pil_image,
            )
        )
        images[index] = pil_image
        answer_counts[answer] += 1

        if index and index % 500 == 0:
            LOGGER.info(
                "  streamed CLEVR %s",
                progress_suffix(
                    min(index + 1, config.max_samples),
                    config.max_samples,
                    started,
                ),
            )

    if not items:
        raise RuntimeError("The CLEVR stream returned no samples.")

    LOGGER.info(
        "Streamed %d CLEVR samples in %s",
        len(items),
        format_duration(time.perf_counter() - started),
    )

    top_answers = [
        answer
        for answer, _ in answer_counts.most_common(config.answer_vocab_size)
    ]
    answer_to_index = {
        answer: index for index, answer in enumerate(top_answers)
    }
    filtered = [item for item in items if item.answer in answer_to_index]

    LOGGER.info(
        "Loaded %d CLEVR items (%d retained across %d answers). "
        "Category counts: %s",
        len(items),
        len(filtered),
        len(answer_to_index),
        dict(Counter(item.category for item in filtered)),
    )
    return filtered, top_answers, images


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------
class CLEVRDataset(Dataset):
    """Dataset returning augmented views, token ids, labels, and metadata."""

    def __init__(
        self,
        items: list[CLEVRItem],
        tokenizer: Any,
        images: dict[int, Image.Image],
        answer_to_index: dict[str, int],
        config: DataConfig,
    ) -> None:
        self.items = items
        self.tokenizer = tokenizer
        self.images = images
        self.answer_to_index = answer_to_index
        self.config = config

        self.transform_view1 = make_train_transform(
            config.image_size, config.aug_scale_min
        )
        self.transform_view2 = make_train_transform(
            config.image_size, config.aug_scale_min
        )
        self.eval_transform = make_eval_transform(config.image_size)

        self.mask_token_id = int(getattr(tokenizer, "mask_token_id", 103))
        self.special_token_ids = tuple(
            token_id
            for token_id in (
                getattr(tokenizer, "pad_token_id", 0),
                getattr(tokenizer, "cls_token_id", 101),
                getattr(tokenizer, "sep_token_id", 102),
            )
            if token_id is not None
        )

    def __len__(self) -> int:
        return len(self.items)

    def _image_for(self, item: CLEVRItem) -> Image.Image:
        if item.image is not None:
            return item.image
        return self.images[item.scene_id]

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.items[index]
        image = self._image_for(item)

        view1 = self.transform_view1(image)
        view2 = self.transform_view2(image)
        eval_view = self.eval_transform(image)

        encoded = self.tokenizer(
            item.question,
            padding="max_length",
            truncation=True,
            max_length=self.config.max_question_length,
            return_tensors="pt",
        )
        question_ids = encoded["input_ids"].squeeze(0)
        question_mask = encoded["attention_mask"].squeeze(0)

        masked_ids1, masked_mask1 = random_token_mask(
            question_ids,
            question_mask,
            mask_token_id=self.mask_token_id,
            special_token_ids=self.special_token_ids,
            ratio=self.config.mask_ratio,
        )
        masked_ids2, masked_mask2 = random_token_mask(
            question_ids,
            question_mask,
            mask_token_id=self.mask_token_id,
            special_token_ids=self.special_token_ids,
            ratio=self.config.mask_ratio,
        )

        return {
            "view1": view1,
            "view2": view2,
            "eval_view": eval_view,
            "masked_ids1": masked_ids1,
            "masked_mask1": masked_mask1,
            "masked_ids2": masked_ids2,
            "masked_mask2": masked_mask2,
            "question_ids": question_ids,
            "question_mask": question_mask,
            "label": torch.tensor(
                self.answer_to_index[item.answer], dtype=torch.long
            ),
            "category": item.category,
            "question": item.question,
            "answer": item.answer,
            "scene_id": item.scene_id,
        }


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------
def build_procedural_clevr_dataset(
    config: DataConfig,
    tokenizer: Any,
) -> CLEVRData:
    """Generate the procedural dataset and its answer vocabulary."""

    LOGGER.info(
        "Generating procedural CLEVR benchmark (%d scenes)...",
        config.num_scenes,
    )

    items: list[CLEVRItem] = []
    answer_counts: Counter[str] = Counter()
    images: dict[int, Image.Image] = {}
    started = time.perf_counter()

    for scene_id in range(config.num_scenes):
        image, qa_list = generate_procedural_clevr_scene()
        images[scene_id] = image
        for qa in qa_list:
            items.append(
                CLEVRItem(
                    scene_id=scene_id,
                    question=qa["question"],
                    answer=qa["answer"],
                    category=qa["category"],
                    image=image,
                )
            )
            answer_counts[qa["answer"]] += 1

        if scene_id and (scene_id + 1) % 200 == 0:
            LOGGER.info(
                "  generated procedural CLEVR %s",
                progress_suffix(
                    scene_id + 1, config.num_scenes, started
                ),
            )

    LOGGER.info(
        "Generated %d procedural scenes in %s",
        len(images),
        format_duration(time.perf_counter() - started),
    )
    return _assemble_clevr_data(
        items, images, answer_counts, tokenizer, config, source="procedural"
    )


def build_hub_clevr_dataset(
    config: DataConfig,
    tokenizer: Any,
) -> CLEVRData:
    items, top_answers, images = build_hub_clevr_items(config)
    answer_counts = Counter(item.answer for item in items)
    return _assemble_clevr_data(
        items,
        images,
        answer_counts,
        tokenizer,
        config,
        source="clevr",
        top_answers=top_answers,
    )


def _assemble_clevr_data(
    items: list[CLEVRItem],
    images: dict[int, Image.Image],
    answer_counts: Counter[str],
    tokenizer: Any,
    config: DataConfig,
    source: str,
    top_answers: Optional[list[str]] = None,
) -> CLEVRData:
    if top_answers is None:
        top_answers = [
            answer
            for answer, _ in answer_counts.most_common(
                config.answer_vocab_size
            )
        ]
    answer_to_index = {
        answer: index for index, answer in enumerate(top_answers)
    }
    filtered_items = [
        item for item in items if item.answer in answer_to_index
    ]

    LOGGER.info(
        "Verified %d CLEVR items across %d answer classes (%s).",
        len(filtered_items),
        len(answer_to_index),
        source,
    )

    dataset = CLEVRDataset(
        items=filtered_items,
        tokenizer=tokenizer,
        images=images,
        answer_to_index=answer_to_index,
        config=config,
    )
    return CLEVRData(
        dataset=dataset,
        answers=top_answers,
        answer_to_index=answer_to_index,
        images=images,
        items=filtered_items,
        source=source,
    )


def build_clevr_dataset(config: DataConfig, tokenizer: Any) -> CLEVRData:
    """Build the configured CLEVR source, falling back when requested."""

    source = config.source.lower()
    if source in {"procedural", "synthetic", "offline"}:
        return build_procedural_clevr_dataset(config, tokenizer)
    if source in {"clevr", "hub", "real", "cauldron"}:
        try:
            return build_hub_clevr_dataset(config, tokenizer)
        except Exception:
            if not config.fallback_to_procedural:
                raise
            LOGGER.exception(
                "Real CLEVR loading failed; falling back to the procedural "
                "generator."
            )
            return build_procedural_clevr_dataset(config, tokenizer)
    raise ValueError(f"Unknown data source: {config.source!r}")


def split_indices(
    dataset: Dataset,
    test_size: Optional[int] = None,
) -> tuple[list[int], list[int]]:
    """Split by item index, matching the original benchmark protocol.

    Note: this is an item-level split, so scenes can appear in both train and
    test.  It is preserved for comparability with the reported numbers.
    """

    total = len(dataset)
    if test_size is None:
        num_test = min(800, total // 4)
    else:
        num_test = min(max(0, test_size), total)
    train_indices = list(range(total - num_test))
    test_indices = list(range(total - num_test, total))
    return train_indices, test_indices
