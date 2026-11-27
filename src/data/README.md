# Dataset API

Install the project dependencies, including Hugging Face `datasets`, with:

```bash
pip install -r requirements.txt
```

`src/data/datasets.py` contains a PyTorch dataset class for each benchmark and
the synthetic dataset. Dataset constructors fetch their Hugging Face split on
first use and the Hub cache handles subsequent access.

```python
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from src.data.datasets import (
    CMUMOSIDataset,
    Flickr30kDataset,
    MMIMDbDataset,
    MUSTARDDataset,
    SyntheticDataset,
    URFunnyDataset,
)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
train = Flickr30kDataset("train", tokenizer=tokenizer, training=True)
loader = DataLoader(train, batch_size=32, shuffle=True, num_workers=4)
batch = next(iter(loader))

mosi = CMUMOSIDataset("train", tokenizer=tokenizer, training=True)
mustard = MUSTARDDataset("train", tokenizer=tokenizer, training=True)
ur_funny = URFunnyDataset("train", tokenizer=tokenizer, training=True)
mm_imdb = MMIMDbDataset("train", tokenizer=tokenizer, training=True)
flickr_test = Flickr30kDataset(
    "test", all_test_captions=True, tokenizer=tokenizer, training=False
)  # Karpathy 5,000 image-caption pairs
synthetic = SyntheticDataset(split="train")
```

Hugging Face dataset classes return `context` and `target` views containing
the modalities present in that dataset, plus labels and IDs where available.
Image views use random crop, flip, and color jitter during training; text views
use independent token masking. Set `training=False` for deterministic
validation and test transforms. MOSI, MUStARD, and UR-FUNNY currently provide
text and labels via `THUIAR/MMLA-Datasets`; their video archives are not
downloaded by these classes. MM-IMDb and Flickr30k provide image-text pairs.

Runnable D-JEPA and FactorCL configs are in `configs/djepa/` and
`configs/factorcl/` for Flickr30k and MM-IMDb. The root `configs/train.yaml`
and `configs/factorcl.yaml` remain the synthetic baselines. Start an experiment
with:

```bash
python main.py --config configs/djepa/flickr30k.yaml
python main.py --config configs/factorcl/mm_imdb.yaml
```

Both methods require image-text pairs. MOSI, MUStARD, and UR-FUNNY text-only
Hub exports are rejected with a clear error until video frames are added to
their dataset rows.
