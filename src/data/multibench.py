"""Download and load the MM-IMDb and UR-FUNNY data distributed by MultiBench.

The dataset files are large and remain outside the source tree.  Downloads are
opt-in through :func:`download_multibench_dataset`; PyTorch and h5py are only
needed when constructing DataLoaders.
"""

from __future__ import annotations

import importlib.util
import pickle
import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .dataset import PreparedDataset

MMIMDB_URL = "https://archive.org/download/mmimdb/multimodal_imdb.hdf5"
UR_FUNNY_DRIVE_URL = "https://drive.google.com/drive/folders/1Agzm157lciMONHOHemHRSySmjn1ahHX1?usp=sharing"
MMIMDB_FILENAME = "multimodal_imdb.hdf5"
UR_FUNNY_FILENAME = "humor.pkl"


def download_multibench_dataset(
    name: str, data_dir: str | Path = "data/multibench", *, overwrite: bool = False
) -> Path:
    """Download a MultiBench dataset and return its local file path.

    MM-IMDb is a direct Archive.org download.  MultiBench distributes its
    processed UR-FUNNY ``humor.pkl`` through a Google Drive folder, downloaded
    with ``gdown`` (``pip install gdown``).  If Drive changes the folder
    contents, pass the downloaded pickle path directly to
    :func:`get_ur_funny_dataloaders`.
    """
    key = name.lower().replace("_", "-")
    root = Path(data_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    if key in {"mm-imdb", "mmimdb"}:
        destination = root / MMIMDB_FILENAME
        if overwrite or not destination.is_file():
            _download_file(MMIMDB_URL, destination)
        return destination
    if key in {"ur-funny", "urfunny", "humor"}:
        destination = root / UR_FUNNY_FILENAME
        if destination.is_file() and not overwrite:
            return destination
        if importlib.util.find_spec("gdown") is None:
            raise ImportError(
                "UR-FUNNY download requires gdown; install it with `pip install gdown`."
            )
        result = subprocess.run(
            [
                shutil.which("gdown") or "gdown",
                "--folder",
                UR_FUNNY_DRIVE_URL,
                "--output",
                str(root),
            ],
            check=False,
        )
        if result.returncode:
            raise RuntimeError(f"gdown could not download the UR-FUNNY folder (exit {result.returncode})")
        matches = sorted(root.rglob(UR_FUNNY_FILENAME))
        if not matches:
            raise FileNotFoundError(
                f"Downloaded the MultiBench UR-FUNNY folder to {root}, but did not find {UR_FUNNY_FILENAME}."
            )
        if matches[0] != destination:
            shutil.copy2(matches[0], destination)
        return destination
    raise ValueError("name must be 'mm-imdb' or 'ur-funny'")


def _download_file(url: str, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        urllib.request.urlretrieve(url, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _torch_data(*, require_h5py: bool = True):
    try:
        import numpy as np
        import torch
        from torch.utils.data import DataLoader, Dataset
    except ImportError as exc:
        raise ImportError("These dataloaders require the `data` extra: pip install -e '.[data]'") from exc
    try:
        import h5py
    except ImportError as exc:
        if require_h5py:
            raise ImportError(
                "MM-IMDb loading requires h5py; install it with `pip install h5py`."
            ) from exc
        h5py = None
    return h5py, np, torch, DataLoader, Dataset


def get_mmimdb_dataloaders(
    path: str | Path,
    *,
    batch_size: int = 32,
    num_workers: int = 0,
    train_shuffle: bool = True,
    vgg_features: bool = False,
):
    """Return train, validation, and test DataLoaders for MultiBench MM-IMDb.

    Samples are ``(text_features, image_or_vgg_features, genre_labels)``.
    MultiBench's official fixed split boundaries are used (15,552 / 2,608 /
    7,799 examples).
    """
    h5py, _, _, DataLoader, Dataset = _torch_data()
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as archive:
        image_key = "vgg_features" if vgg_features else "images"
        required = ("features", image_key, "genres")
        missing = [key for key in required if key not in archive]
        if missing:
            raise ValueError(f"MM-IMDb HDF5 is missing expected datasets: {', '.join(missing)}")
        lengths = [len(archive[key]) for key in required]
        if len(set(lengths)) != 1 or lengths[0] < 25959:
            raise ValueError(f"Unexpected MM-IMDb array lengths: {dict(zip(required, lengths))}")

    class MMIMDbDataset(Dataset):
        def __init__(self, start: int, end: int):
            self.start, self.end = start, end
            self._archive = None

        def __len__(self):
            return self.end - self.start

        def __getitem__(self, index):
            import torch

            if self._archive is None:
                self._archive = h5py.File(path, "r")
            row = self.start + index
            return tuple(
                torch.as_tensor(self._archive[key][row])
                for key in ("features", image_key, "genres")
            )

    boundaries = ((0, 15552), (15552, 18160), (18160, 25959))
    return tuple(
        DataLoader(
            MMIMDbDataset(start, end),
            batch_size=batch_size,
            shuffle=train_shuffle if split == 0 else False,
            num_workers=num_workers,
        )
        for split, (start, end) in enumerate(boundaries)
    )


def get_ur_funny_dataloaders(
    path: str | Path,
    *,
    batch_size: int = 32,
    num_workers: int = 0,
    train_shuffle: bool = True,
    max_seq_len: int | None = None,
):
    """Load MultiBench's processed UR-FUNNY ``humor.pkl`` into three loaders.

    Items are dictionaries with ``text``, ``vision``, ``audio``, and
    ``label`` tensors. Variable length sequences are padded per batch; optional
    ``max_seq_len`` truncates/pads every stream to a fixed length.
    """
    _, np, torch, DataLoader, Dataset = _torch_data(require_h5py=False)
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as stream:
        raw = pickle.load(stream)
    if not isinstance(raw, dict) or not {"train", "valid", "test"}.issubset(raw):
        raise ValueError("UR-FUNNY pickle must contain train, valid, and test splits")

    class URFunnyDataset(Dataset):
        def __init__(self, split: dict[str, Any]):
            self.split = split
            self.modalities = {
                name: [np.asarray(value) for value in split[name]]
                for name in ("text", "vision", "audio")
            }
            self.labels = np.asarray(split["labels"])
            sizes = {len(value) for value in (*self.modalities.values(), self.labels)}
            if len(sizes) != 1:
                raise ValueError("UR-FUNNY modality and label row counts do not match")

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, index):
            item = {name: torch.as_tensor(value[index], dtype=torch.float32) for name, value in self.modalities.items()}
            item["label"] = torch.as_tensor(self.labels[index]).squeeze().long()
            return item

    def collate(rows):
        from torch.nn.utils.rnn import pad_sequence

        batch: dict[str, Any] = {}
        for name in ("text", "vision", "audio"):
            values = [row[name] for row in rows]
            if values[0].ndim == 0:
                batch[name] = torch.stack(values)
                continue
            if max_seq_len is not None:
                values = [value[:max_seq_len] for value in values]
            padded = pad_sequence(values, batch_first=True)
            if max_seq_len is not None and padded.shape[1] < max_seq_len:
                padding = padded.new_zeros((len(values), max_seq_len - padded.shape[1], *padded.shape[2:]))
                padded = torch.cat((padded, padding), dim=1)
            batch[name] = padded
        batch["label"] = torch.stack([row["label"] for row in rows])
        return batch

    output = []
    for split_name in ("train", "valid", "test"):
        if not isinstance(raw[split_name], dict):
            raise ValueError(f"UR-FUNNY {split_name!r} split must be a mapping")
        dataset = URFunnyDataset(raw[split_name])
        output.append(
            DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=train_shuffle if split_name == "train" else False,
                num_workers=num_workers,
                collate_fn=collate,
            )
        )
    return tuple(output)


def load_multibench_prepared_dataset(
    name: str,
    path: str | Path,
    *,
    batch_size: int = 256,
) -> PreparedDataset:
    """Materialize a MultiBench dataset in the in-memory format used by main.py."""
    from .dataset import PreparedDataset

    key = name.lower().replace("_", "-")
    if key in {"mmimdb", "mm-imdb"}:
        train_loader, valid_loader, test_loader = get_mmimdb_dataloaders(
            path, batch_size=batch_size, num_workers=0, train_shuffle=False
        )
        modality_names = ("text", "vision")

        def unpack(loader):
            rows, labels = [], []
            for text, image, target in loader:
                rows.append((text.numpy(), image.numpy()))
                labels.append(target.numpy())
            return [np.concatenate([row[index] for row in rows]) for index in range(2)], np.concatenate(labels)

    elif key in {"urfunny", "ur-funny", "humor"}:
        train_loader, valid_loader, test_loader = get_ur_funny_dataloaders(
            path, batch_size=batch_size, num_workers=0, train_shuffle=False
        )
        modality_names = ("text", "vision", "audio")

        def unpack(loader):
            rows = {name: [] for name in modality_names}
            labels = []
            for batch in loader:
                for name in modality_names:
                    rows[name].append(batch[name].numpy())
                labels.append(batch["label"].numpy())
            return [np.concatenate(rows[name]) for name in modality_names], np.concatenate(labels)

    else:
        raise ValueError("name must be 'mmimdb' or 'ur-funny'")

    train_rows, train_labels = unpack(train_loader)
    valid_rows, valid_labels = unpack(valid_loader)
    test_rows, test_labels = unpack(test_loader)
    context = {
        name: np.concatenate((train_rows[index], valid_rows[index], test_rows[index])).astype(
            np.float32
        )
        for index, name in enumerate(modality_names)
    }
    labels = np.concatenate((train_labels, valid_labels, test_labels))
    n_train = len(train_labels) + len(valid_labels)
    train_indices = np.arange(n_train, dtype=np.int32)
    test_indices = np.arange(n_train, len(labels), dtype=np.int32)
    return PreparedDataset(
        context=context,
        target={name: values.copy() for name, values in context.items()},
        factors={"labels": labels.astype(np.float32)},
        train_indices=train_indices,
        test_indices=test_indices,
    )
