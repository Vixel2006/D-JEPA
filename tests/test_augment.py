import torch

from djepa.augment import patch_drop_vision, random_token_mask


def test_patch_drop_preserves_shape_and_zeroes_patches():
    torch.manual_seed(0)
    images = torch.ones(2, 3, 32, 32)
    dropped = patch_drop_vision(images, patch_size=8, drop_ratio=0.5)
    assert dropped.shape == images.shape
    assert dropped.min() == 0.0
    assert dropped.max() == 1.0
    assert not torch.equal(images, dropped)


def test_random_token_mask_preserves_special_tokens():
    torch.manual_seed(0)
    input_ids = torch.tensor([[101, 5, 6, 7, 102, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 0]])
    masked_ids, masked_mask = random_token_mask(
        input_ids,
        attention_mask,
        mask_token_id=103,
        special_token_ids=(0, 101, 102),
        ratio=1.0,
    )
    assert masked_ids.shape == input_ids.shape
    assert masked_mask is attention_mask
    assert masked_ids[0, 0] == 101
    assert masked_ids[0, 4] == 102
    assert masked_ids[0, 5] == 0
    assert torch.equal(masked_ids[0, 1:4], torch.tensor([103, 103, 103]))
