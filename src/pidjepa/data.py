"""Dataset bundles: a synthetic ground-truth benchmark and real CLEVR."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class Bundle:
    """A paired multimodal dataset plus task labels and metadata.

    ``vision_views`` and ``text_views`` have shape ``[N, V, ...]`` where ``V``
    is the number of augmented views.  Vector features and token features share
    this layout; token features add a token dimension.
    """

    vision_views: torch.Tensor
    text_views: torch.Tensor
    tasks: dict[str, torch.Tensor]
    n_classes: dict[str, int]
    factors: dict[str, torch.Tensor]
    split: dict[str, torch.Tensor]
    text_mask: torch.Tensor | None = None
    kind: str = "vec"
    has_ground_truth: bool = False
    categories: np.ndarray | None = None
    task_descriptions: dict[str, str] = field(default_factory=dict)
    pid_targets: dict[str, torch.Tensor] = field(default_factory=dict)


def standardize_features(features: torch.Tensor, train_indices: torch.Tensor) -> torch.Tensor:
    """Standardize features using statistics from the training split only."""
    flat = features[train_indices].reshape(-1, features.size(-1))
    mean, std = flat.mean(0), flat.std(0).clamp_min(1e-6)
    return (features - mean) / std


def grouped_split(
    groups: np.ndarray,
    seed: int,
    fractions: tuple[float, float, float] = (0.6, 0.15, 0.25),
) -> dict[str, torch.Tensor]:
    """Split indices by group so no group appears in more than one split."""
    rng = np.random.RandomState(seed)
    unique_groups = np.unique(groups)
    rng.shuffle(unique_groups)
    first_cut = int(fractions[0] * len(unique_groups))
    second_cut = int((fractions[0] + fractions[1]) * len(unique_groups))

    group_to_split = {group: 0 for group in unique_groups[:first_cut]}
    group_to_split.update({group: 1 for group in unique_groups[first_cut:second_cut]})
    group_to_split.update({group: 2 for group in unique_groups[second_cut:]})
    labels = np.array([group_to_split[group] for group in groups])
    names = ("train", "val", "test")
    return {
        name: torch.from_numpy(np.where(labels == index)[0]) for index, name in enumerate(names)
    }


def make_synthetic_bundle(
    n_samples: int,
    seed: int,
    n_views: int = 3,
    n_shared: int = 4,
    n_private: int = 4,
    n_nuisance: int = 6,
    vision_dim: int = 64,
    text_dim: int = 48,
    noise: float = 0.1,
) -> Bundle:
    """Create a synthetic multimodal dataset with known information atoms.

    Each stream is a nonlinear mixture of a shared latent factor, a
    modality-private factor, and independent nuisance noise.  The task labels
    isolate redundancy (``shared``), unique information (``priv_v``/``priv_t``),
    and synergy (the XOR of the first private coordinates).
    """
    generator = torch.Generator().manual_seed(seed)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator)

    shared = randn(n_samples, n_shared)
    vision_private = randn(n_samples, n_private)
    text_private = randn(n_samples, n_private)

    def make_mixer(input_dim: int, output_dim: int, hidden_dim: int = 128):
        weight_1 = randn(input_dim, hidden_dim) / math.sqrt(input_dim)
        weight_2 = randn(hidden_dim, output_dim) / math.sqrt(hidden_dim)
        return lambda inputs: torch.tanh(inputs @ weight_1) @ weight_2

    vision_mixer = make_mixer(n_shared + n_private + n_nuisance, vision_dim)
    text_mixer = make_mixer(n_shared + n_private + n_nuisance, text_dim)

    vision_views = torch.stack(
        [
            vision_mixer(torch.cat([shared, vision_private, randn(n_samples, n_nuisance)], 1))
            + noise * randn(n_samples, vision_dim)
            for _ in range(n_views)
        ],
        dim=1,
    )
    text_views = torch.stack(
        [
            text_mixer(torch.cat([shared, text_private, randn(n_samples, n_nuisance)], 1))
            + noise * randn(n_samples, text_dim)
            for _ in range(n_views)
        ],
        dim=1,
    )

    def quantile_bins(values: torch.Tensor, n_bins: int = 4) -> torch.Tensor:
        ranks = values.argsort().argsort()
        return (ranks * n_bins // len(values)).long()

    shared_scores = shared @ randn(n_shared)
    vision_private_scores = vision_private[:, 1:] @ randn(n_private - 1)
    text_private_scores = text_private[:, 1:] @ randn(n_private - 1)
    tasks = {
        "shared": quantile_bins(shared_scores),
        "priv_v": quantile_bins(vision_private_scores),
        "priv_t": quantile_bins(text_private_scores),
        "synergy": ((vision_private[:, 0] > 0) ^ (text_private[:, 0] > 0)).long(),
    }
    n_classes = {"shared": 4, "priv_v": 4, "priv_t": 4, "synergy": 2}
    split = grouped_split(np.arange(n_samples), seed)

    return Bundle(
        vision_views=standardize_features(vision_views, split["train"]),
        text_views=standardize_features(text_views, split["train"]),
        tasks=tasks,
        n_classes=n_classes,
        factors={"shared": shared, "priv_v": vision_private, "priv_t": text_private},
        split=split,
        kind="vec",
        has_ground_truth=True,
        task_descriptions={
            "shared": "shared factor",
            "priv_v": "vision-private",
            "priv_t": "text-private",
            "synergy": "XOR synergy",
        },
        pid_targets={
            "shared": shared_scores,
            "priv_v": vision_private_scores,
            "priv_t": text_private_scores,
            "product": vision_private[:, 0] * text_private[:, 0],
        },
    )


# =============================================================================
# CLEVR
# =============================================================================
COLORS = ["red", "blue", "green", "yellow", "cyan", "purple", "brown", "gray"]
SHAPES = ["cube", "sphere", "cylinder", "cubes", "spheres", "cylinders", "block", "ball"]
SIZES = ["small", "large", "big", "tiny"]
MATERIALS = ["rubber", "metal", "matte", "shiny"]
RELATIONS = ["left", "right", "front", "behind"]
ATTRIBUTE_ANSWERS = set(
    COLORS + ["cube", "sphere", "cylinder", "small", "large", "rubber", "metal"]
)
MENTION_VOCABULARY = COLORS + SHAPES + SIZES + MATERIALS + RELATIONS
QUESTION_CATEGORIES = ["counting", "exist_compare", "attribute", "other"]


def question_category(question: str, answer: str) -> str:
    """Map a CLEVR question/answer pair to one of four reasoning categories."""
    question = question.lower()
    if question.startswith("how many"):
        return "counting"
    if answer in ("yes", "no"):
        return "exist_compare"
    if answer in ATTRIBUTE_ANSWERS:
        return "attribute"
    return "other"


def load_clevr_bundle(args, device: str) -> Bundle:
    """Load CLEVR from the Cauldron stream and encode it with frozen towers."""
    from datasets import load_dataset
    from torchvision import transforms
    from transformers import AutoModel, AutoTokenizer

    n_views, max_length = args.views, args.max_text_length
    use_cuda = str(device).startswith("cuda")
    dataset = load_dataset("HuggingFaceM4/the_cauldron", "clevr", split="train", streaming=True)
    images, questions, answers, image_ids = [], [], [], []
    for image_index, row in enumerate(dataset):
        if image_index >= args.clevr_images:
            break
        images.append(row["images"][0].convert("RGB").resize((224, 224)))
        for text_pair in row["texts"][: args.questions_per_image]:
            questions.append(
                text_pair["user"].replace("<image>", "").replace("Question:", "").strip()
            )
            answers.append(
                text_pair["assistant"].replace("Answer:", "").replace(".", "").strip().lower()
            )
            image_ids.append(image_index)
    image_ids = np.array(image_ids)
    print(f"[data] {len(images)} images, {len(questions)} QA pairs")

    top_answers = [answer for answer, _ in Counter(answers).most_common(args.max_answer_classes)]
    answer_to_index = {answer: index for index, answer in enumerate(top_answers)}
    keep = np.array([answer in answer_to_index for answer in answers])
    questions = [question for question, keep_row in zip(questions, keep) if keep_row]
    answers = [answer for answer, keep_row in zip(answers, keep) if keep_row]
    image_ids = image_ids[keep]
    categories = np.array(
        [question_category(question, answer) for question, answer in zip(questions, answers)]
    )
    n_samples = len(questions)

    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    clean_transform = transforms.Compose([transforms.ToTensor(), normalize])
    augmented_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    vision_tower = AutoModel.from_pretrained("facebook/dinov2-small").to(device).eval()

    @torch.no_grad()
    def encode_images(transform, tower) -> torch.Tensor:
        encoded_batches = []
        for start in range(0, len(images), 64):
            batch = torch.stack([transform(image) for image in images[start : start + 64]])
            with torch.autocast("cuda", enabled=use_cuda):
                hidden = tower(pixel_values=batch.to(device)).last_hidden_state
            cls_token, patch_tokens = hidden[:, :1].float(), hidden[:, 1:].float()
            side = int(round(patch_tokens.size(1) ** 0.5))
            patch_tokens = patch_tokens.reshape(patch_tokens.size(0), side, side, -1)
            patch_tokens = patch_tokens.permute(0, 3, 1, 2)
            patch_tokens = F.adaptive_avg_pool2d(patch_tokens, args.vision_grid)
            patch_tokens = patch_tokens.flatten(2).transpose(1, 2)
            encoded_batches.append(torch.cat([cls_token, patch_tokens], 1).half().cpu())
        return torch.cat(encoded_batches)

    image_views = torch.stack(
        [encode_images(clean_transform, vision_tower)]
        + [encode_images(augmented_transform, vision_tower) for _ in range(n_views - 1)],
        dim=1,
    )
    del vision_tower
    vision_views = image_views[torch.from_numpy(image_ids)]
    del image_views

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    text_tower = AutoModel.from_pretrained("bert-base-uncased").to(device).eval()
    tokenized = tokenizer(
        questions,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids, attention_mask = tokenized["input_ids"], tokenized["attention_mask"]
    special_tokens = torch.zeros_like(input_ids, dtype=torch.bool)
    for special_id in tokenizer.all_special_ids:
        special_tokens |= input_ids == special_id

    text_hidden_dim = text_tower.config.hidden_size
    text_views = torch.empty(n_samples, n_views, max_length, text_hidden_dim, dtype=torch.half)
    with torch.no_grad():
        for view_index in range(n_views):
            mask_ratio = 0.0 if view_index == 0 else 0.15
            for start in range(0, n_samples, 256):
                batch_ids = input_ids[start : start + 256].clone()
                batch_mask = attention_mask[start : start + 256]
                if mask_ratio > 0:
                    random_values = torch.rand(batch_ids.shape)
                    mask_positions = (
                        (random_values < mask_ratio)
                        & ~special_tokens[start : start + 256]
                        & (batch_mask == 1)
                    )
                    batch_ids[mask_positions] = tokenizer.mask_token_id
                with torch.autocast("cuda", enabled=use_cuda):
                    hidden = text_tower(
                        input_ids=batch_ids.to(device),
                        attention_mask=batch_mask.to(device),
                    ).last_hidden_state
                text_views[start : start + 256, view_index] = hidden.half().cpu()
    del text_tower

    question_type = torch.zeros(n_samples, len(QUESTION_CATEGORIES))
    for index, category in enumerate(categories):
        question_type[index, QUESTION_CATEGORIES.index(category)] = 1

    mentions = torch.zeros(n_samples, len(MENTION_VOCABULARY))
    for index, question in enumerate(questions):
        words = set(question.lower().replace("?", "").split())
        for mention_index, mention in enumerate(MENTION_VOCABULARY):
            mentions[index, mention_index] = float(mention in words)

    counting_target = torch.full((n_samples,), float("nan"))
    for index, (category, answer) in enumerate(zip(categories, answers)):
        if category == "counting" and answer.isdigit():
            counting_target[index] = float(answer)

    labels = torch.tensor([answer_to_index[answer] for answer in answers])
    split = grouped_split(image_ids, seed=1234)
    return Bundle(
        vision_views=vision_views,
        text_views=text_views,
        tasks={"answer": labels},
        n_classes={"answer": len(top_answers)},
        factors={"qtype": question_type, "mentions": mentions},
        split=split,
        text_mask=attention_mask.bool(),
        kind="tokens",
        has_ground_truth=False,
        categories=categories,
        task_descriptions={"answer": "CLEVR answer"},
        pid_targets={"count": counting_target},
    )
