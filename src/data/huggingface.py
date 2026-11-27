from __future__ import annotations

from typing import Literal

from datasets import Dataset, DatasetDict, load_dataset

_MMLA_DATASET = "THUIAR/MMLA-Datasets"
_MMLA_SUBDIRECTORIES = {
    "mosi": "MOSI",
    "mustard": "MUStARD",
    "ur_funny": "UR-FUNNY-v2",
}


def _load_mmla(name: str, split: str | None) -> Dataset | DatasetDict:
    directory = _MMLA_SUBDIRECTORIES[name]
    # Keep the conventional `validation` name while the Hub stores `dev.tsv`.
    data_files = {
        "train": f"https://huggingface.co/datasets/{_MMLA_DATASET}/resolve/main/MMLA-Datasets/{directory}/train.tsv",
        "validation": f"https://huggingface.co/datasets/{_MMLA_DATASET}/resolve/main/MMLA-Datasets/{directory}/dev.tsv",
        "test": f"https://huggingface.co/datasets/{_MMLA_DATASET}/resolve/main/MMLA-Datasets/{directory}/test.tsv",
    }
    dataset = load_dataset("csv", data_files=data_files, delimiter="\t")
    if split is None:
        return dataset
    requested = "validation" if split == "dev" else split
    if requested not in dataset:
        raise ValueError(f"Unknown split {split!r}; available splits: {list(dataset)}")
    return dataset[requested]


def load_cmu_mosi(split: str | None = None) -> Dataset | DatasetDict:
    """Load CMU-MOSI split metadata (train, validation/dev, or test)."""
    return _load_mmla("mosi", split)


def load_mustard(split: str | None = None) -> Dataset | DatasetDict:
    """Load the sarcasm dataset MUStARD split metadata (train/dev/test)."""
    return _load_mmla("mustard", split)


def load_ur_funny(split: str | None = None) -> Dataset | DatasetDict:
    """Load UR-FUNNY v2 split metadata (train/dev/test)."""
    return _load_mmla("ur_funny", split)


def load_mm_imdb(split: str = "train") -> Dataset:
    """Load MM-IMDb posters, plot text, and genre labels from the Hub.

    The selected Hub mirror currently publishes a single ``train`` split.
    """
    return load_dataset("pranavmr/MM-IMDb", split=split)


def load_flickr30k(split: Literal["train", "validation", "test"] | None = None):
    """Load Flickr30k with the Karpathy train/validation/test partition.

    Each image row has five captions. The Karpathy test partition contains
    1,000 images (5,000 image-caption pairs); use ``flickr_test_pairs`` to
    flatten those rows when an evaluation expects individual pairs.
    """
    dataset = load_dataset("noonamkha/flickr30k-karpathy")
    return dataset if split is None else dataset[split]


def flickr_test_pairs():
    """Return the 5,000 Karpathy test image-caption pairs as a Dataset."""
    test = load_flickr30k("test")
    return test.map(
        lambda batch: {
            "image_id": [
                image_id
                for image_id, captions in zip(batch["image_id"], batch["captions"])
                for _ in captions
            ],
            "image": [
                image
                for image, captions in zip(batch["image"], batch["captions"])
                for _ in captions
            ],
            "caption": [
                caption for captions in batch["captions"] for caption in captions
            ],
        },
        batched=True,
        remove_columns=test.column_names,
    )


__all__ = [
    "flickr_test_pairs",
    "load_cmu_mosi",
    "load_flickr30k",
    "load_mm_imdb",
    "load_mustard",
    "load_ur_funny",
]
