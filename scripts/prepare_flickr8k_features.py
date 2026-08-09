import argparse
import pathlib
import sys
import time

import torch
from transformers import CLIPModel, CLIPProcessor

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.flickr8k import Flickr8K

SPLITS = ["train", "dev", "test"]


def pick_device(requested):
    if requested:
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def encode_image_paths(paths, model, processor, device, batch_size, normalize):
    feats = {}
    paths = list(dict.fromkeys(paths))
    for start in range(0, len(paths), batch_size):
        chunk = paths[start : start + batch_size]
        inputs = processor(images=chunk, return_tensors="pt").to(device)
        with torch.no_grad():
            emb = model.get_image_features(pixel_values=inputs["pixel_values"])
        if normalize:
            emb = emb / emb.norm(dim=-1, keepdim=True)
        for path, e in zip(chunk, emb):
            feats[path] = e.detach().cpu()
        print(f"    embedded images {start + len(chunk)}/{len(paths)}", flush=True)
    return feats


def encode_texts(texts, model, processor, device, batch_size, normalize):
    embs = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        inputs = processor(
            text=chunk, return_tensors="pt", padding=True, truncation=True
        ).to(device)
        with torch.no_grad():
            emb = model.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
        if normalize:
            emb = emb / emb.norm(dim=-1, keepdim=True)
        embs.append(emb.detach().cpu())
        print(f"    embedded captions {start + len(chunk)}/{len(texts)}", flush=True)
    return torch.cat(embs, dim=0)


def prepare_split(
    split, model, processor, device, data_dir, features_dir, batch_size, normalize
):
    ds = Flickr8K(root=data_dir, split=split)
    paths = [p for p, _, _ in ds.samples]
    captions = [c for _, c, _ in ds.samples]
    image_ids = [i for _, _, i in ds.samples]

    print(f"Building '{split}' split: {len(ds)} samples")
    image_feats = encode_image_paths(paths, model, processor, device, batch_size, normalize)
    txt_feats = encode_texts(captions, model, processor, device, batch_size, normalize)

    img_feats_all = torch.stack([image_feats[p] for p in paths])
    img_index_t = torch.tensor(image_ids)

    out_dir = features_dir / split
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(img_feats_all, out_dir / "img_feat.pt")
    torch.save(txt_feats, out_dir / "txt_feat.pt")
    torch.save(img_index_t, out_dir / "img_index.pt")
    print(f"  saved '{split}' features to {out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Precompute CLIP image/text features for Flickr8k splits."
    )
    parser.add_argument("--data-dir", default="data/flickr8k", help="Flickr8k root directory")
    parser.add_argument(
        "--features-dir", default=None, help="Cache dir (default: <data-dir>/features)"
    )
    parser.add_argument(
        "--model", default="openai/clip-vit-base-patch32", help="HuggingFace CLIP model id"
    )
    parser.add_argument(
        "--split", nargs="+", default=["all"], choices=["all", *SPLITS],
        help="Splits to precompute (default: all)",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None, help="cpu or cuda (default: auto)")
    parser.add_argument(
        "--no-normalize", action="store_true", help="Skip L2 normalization of features"
    )
    args = parser.parse_args()

    data_dir = pathlib.Path(args.data_dir)
    if not (data_dir / "Flicker8k_Dataset").is_dir():
        sys.exit(
            f"Flickr8k images not found at {data_dir}. "
            f"Run `python scripts/download_flickr8k.py` first."
        )

    features_dir = pathlib.Path(args.features_dir or (data_dir / "features"))
    features_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(pick_device(args.device))
    print(f"Loading CLIP model {args.model} on {device} ...")
    model = CLIPModel.from_pretrained(args.model)
    processor = CLIPProcessor.from_pretrained(args.model)
    model.to(device).eval()

    splits = list(SPLITS) if "all" in args.split else args.split
    for split in splits:
        start = time.time()
        prepare_split(
            split, model, processor, device, data_dir, features_dir,
            args.batch_size, not args.no_normalize,
        )
        print(f"  ({time.time() - start:.1f}s)")

    print("Done. Features cached; now run train_teacher / distill / eval with --dataset flickr8k.")


if __name__ == "__main__":
    main()