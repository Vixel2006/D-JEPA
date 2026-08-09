import pathlib
from typing import List, Tuple

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch

IMAGES_DIR = "Flicker8k_Dataset"
TOKEN_FILE = "Flickr8k.token.txt"
SPLIT_FILES = {
    "train": "Flickr_8k.trainImages.txt",
    "dev": "Flickr_8k.devImages.txt",
    "test": "Flickr_8k.testImages.txt",
}

DEFAULT_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def _validate_split(split: str) -> str:
    if split == "all" or split in SPLIT_FILES:
        return split
    raise ValueError(
        f"Unknown split '{split}'. Choose from {['all', *SPLIT_FILES]}"
    )


class Flickr8K(Dataset):
    """
    Flickr8k image-caption pairs.

    Each ``__getitem__`` returns ``(image, caption, image_index)``:
        - image: transformed image tensor (C, H, W)
        - caption: caption text associated with that image
        - image_index: integer index of the image within this split,
          usable as an identity proxy for the shared-latent slot.
    Flattens so that every (image, caption) pair is its own sample.
    """

    def __init__(
        self,
        root: str | pathlib.Path = "data/flickr8k",
        split: str = "all",
        transform=None,
    ):
        super().__init__()
        self.root = pathlib.Path(root)
        self.split = _validate_split(split)
        self.transform = transform if transform is not None else DEFAULT_TRANSFORM

        images_dir = self.root / IMAGES_DIR
        token_path = self.root / TOKEN_FILE
        if not images_dir.is_dir() or not token_path.is_file():
            raise FileNotFoundError(
                f"Flickr8k data not found under {self.root}. "
                f"Run `python scripts/download_flickr8k.py` first."
            )

        self.image_dir = images_dir
        self.image_to_captions = self._load_captions(token_path)
        if not self.image_to_captions:
            raise ValueError(f"No captions parsed from {token_path}")

        image_ids = self._select_image_ids(images_dir)

        self.samples: List[Tuple[str, str, int]] = []
        for image_index, image_id in enumerate(image_ids):
            image_path = str(images_dir / image_id)
            for caption in self.image_to_captions[image_id]:
                self.samples.append((image_path, caption, image_index))

    @staticmethod
    def _load_captions(token_path: pathlib.Path) -> dict[str, list[str]]:
        captions = {}
        with open(token_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                image_part, sep, caption = line.partition("\t")
                if not sep or not caption:
                    continue
                image_id = image_part.split("#")[0]
                captions.setdefault(image_id, []).append(caption)
        return captions

    def _select_image_ids(self, images_dir: pathlib.Path) -> list[str]:
        if self.split == "all":
            return list(self.image_to_captions.keys())

        image_ids = []
        split_file = self.root / SPLIT_FILES[self.split]
        with open(split_file, encoding="utf-8") as fh:
            for line in fh:
                image_id = line.strip()
                if image_id:
                    image_ids.append(image_id)
        missing = [name for name in image_ids if not (images_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Split '{self.split}' references {len(missing)} images "
                f"missing from {images_dir}, e.g. {missing[:3]}"
            )
        return image_ids

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, int]:
        image_path, caption, image_index = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, caption, image_index


def flickr8k_collate(batch):
    images = torch.stack([item[0] for item in batch])
    captions = [item[1] for item in batch]
    image_indices = [item[2] for item in batch]
    return images, captions, image_indices


def make_flickr8k_dataloader(
    data_dir: str | pathlib.Path = "data/flickr8k",
    split: str = "train",
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    transform=None,
) -> DataLoader:
    dataset = Flickr8K(root=data_dir, split=split, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=flickr8k_collate,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


class Flickr8kFeatures(Dataset):
    """
    Precomputed CLIP image/text features for a Flickr8k split.

    Each ``__getitem__`` returns ``(img_feat, txt_feat, img_feat)``:
        - img_feat, txt_feat: flat feature vectors (512,) consumed by the SSD
          and CLIP encoders (which reshape them into ``seq_len`` tokens).
        - the image embedding is passed along as the proxy "shared latent"
          slot expected by the training/evaluation loops.
    """

    def __init__(
        self,
        features_dir: str | pathlib.Path = "data/flickr8k/features",
        split: str = "train",
    ):
        super().__init__()
        self.split = _validate_split(split)
        if self.split == "all":
            raise ValueError("Flickr8kFeatures requires a concrete split: train/dev/test.")

        features_dir = pathlib.Path(features_dir)
        split_dir = features_dir / self.split
        img_path = split_dir / "img_feat.pt"
        txt_path = split_dir / "txt_feat.pt"
        if not img_path.is_file() or not txt_path.is_file():
            raise FileNotFoundError(
                f"Features for split '{self.split}' not found in {split_dir}. "
                f"Run `python scripts/prepare_flickr8k_features.py` first."
            )

        self.img_feat = torch.load(img_path, weights_only=True)
        self.txt_feat = torch.load(txt_path, weights_only=True)
        if self.img_feat.size(0) != self.txt_feat.size(0):
            raise ValueError(
                f"Mismatched feature counts in {split_dir}: "
                f"img={self.img_feat.size(0)} txt={self.txt_feat.size(0)}"
            )

    def __len__(self) -> int:
        return self.img_feat.size(0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img = self.img_feat[idx]
        txt = self.txt_feat[idx]
        return img, txt, img


def make_flickr8k_features_dataloader(
    features_dir: str | pathlib.Path = "data/flickr8k/features",
    split: str = "train",
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
) -> DataLoader:
    dataset = Flickr8kFeatures(features_dir=features_dir, split=split)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )