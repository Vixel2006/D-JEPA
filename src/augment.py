"""Masking utilities shared by pretraining and dataset construction."""

from __future__ import annotations

from typing import Sequence, Tuple

import torch


def patch_drop_vision(
    images: torch.Tensor,
    patch_size: int = 32,
    drop_ratio: float = 0.30,
) -> torch.Tensor:
    """Zero out a random fraction of non-overlapping image patches.

    The per-sample ``randperm`` is deliberately kept in the loop: it matches
    the random-number consumption of the original experiment.
    """

    batch_size, channels, height, width = images.shape
    output = images.clone()
    patches_h, patches_w = height // patch_size, width // patch_size
    num_patches = patches_h * patches_w
    num_drop = int(num_patches * drop_ratio)

    for batch_index in range(batch_size):
        drop_indices = torch.randperm(num_patches)[:num_drop]
        for flat_index in drop_indices:
            index = int(flat_index)
            row = (index // patches_w) * patch_size
            col = (index % patches_w) * patch_size
            output[
                batch_index,
                :,
                row : row + patch_size,
                col : col + patch_size,
            ] = 0.0
    return output


def random_token_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    mask_token_id: int = 103,
    special_token_ids: Sequence[int] = (0, 101, 102),
    ratio: float = 0.15,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Randomly replace non-special tokens with ``mask_token_id``."""

    masked_ids = input_ids.clone()
    valid_mask = attention_mask == 1
    for token_id in special_token_ids:
        valid_mask = valid_mask & (input_ids != token_id)

    random_values = torch.rand(input_ids.shape, device=input_ids.device)
    mask_indices = valid_mask & (random_values < ratio)
    masked_ids[mask_indices] = mask_token_id
    return masked_ids, attention_mask
