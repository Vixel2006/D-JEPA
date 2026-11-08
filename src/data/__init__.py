"""Data loading for the CLEVR benchmark."""

from .clevr import (
    CATEGORY_ATTRIBUTE,
    CATEGORY_COUNTING,
    CATEGORY_SPATIAL,
    CLEVRData,
    CLEVRDataset,
    CLEVRItem,
    build_clevr_dataset,
    build_hub_clevr_dataset,
    build_procedural_clevr_dataset,
    classify_clevr_question,
    load_tokenizer,
    split_indices,
)

__all__ = [
    "CATEGORY_ATTRIBUTE",
    "CATEGORY_COUNTING",
    "CATEGORY_SPATIAL",
    "CLEVRData",
    "CLEVRDataset",
    "CLEVRItem",
    "build_clevr_dataset",
    "build_hub_clevr_dataset",
    "build_procedural_clevr_dataset",
    "classify_clevr_question",
    "load_tokenizer",
    "split_indices",
]
