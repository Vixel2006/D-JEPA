# Sufficient Statistic Distillation (SSD) for Multimodal Representation Learning

Paired image–caption representation learning that compares two approaches to building a
shared latent space between modalities, with the goal of **eliminating the modality gap**
(the distance between image and text embeddings in the shared space):

1. **CLIP baseline** — a symmetric InfoNCE (contrastive) loss over paired image/text embeddings.
2. **Sufficient Statistic Distillation (SSD)** — a two-stage pipeline:

   - **Teacher** (cross-modal beta-VAE, `src/models/ssd_teacher.py`): learns a joint
     posterior `q(z | x_img, x_txt)` by reconstructing both modalities from a shared
     latent `z`.
   - **Students** (`src/models/ssd_students.py`): a unimodal image encoder and a
     unimodal text encoder, distilled from the frozen teacher so that each modality alone
     recovers the same shared latent.

The core question: does SSD produce representations with a smaller (or eliminated)
modality gap while preserving cross-modal retrieval?

## Project layout

```
main.py                        CLI entry point (train/eval)
src/
  config.py                    default hyper-parameters
  data/
    synthetic.py               synthetic paired features
    flickr8k.py                raw Flickr8k dataset + precomputed-features dataset
  models/
    clip_baseline.py           CLIP/ImageEncoder + TextEncoder + CLIPModel
    ssd_teacher.py             cross-modal beta-VAE teacher
    ssd_students.py            ViT image / transformer text encoders(+decoders), students
  training/
    train_clip.py              InfoNCE contrastive training
    train_ssd_teacher.py       beta-VAE reconstruction training
    train_ssd_students.py      distillation training
  evaluation/
    metrics.py                 modality gap, recall@k, variance, retrieval examples
    visualize.py               joint t-SNE plots
scripts/
  download_flickr8k.py         fetches + extracts the Flickr8k dataset
  prepare_flickr8k_features.py precomputes CLIP features into a cache
```

## Getting started

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install transformers   # only needed for the Flickr8k feature precompute
```

## Data

### Synthetic (default)

`SyntheticPairedData` generates correlated image (512-d) and text (384-d) feature vectors
from a shared latent. Used by the default commands and by `test_synthetic`; no setup needed.

### Flickr8k

1. Download and extract the dataset (~1.1 GB, then deletes the archives):

   ```bash
   python scripts/download_flickr8k.py
   ```

   Produces `data/flickr8k/`:
   `Flicker8k_Dataset/` (8,091 JPEGs), `Flickr8k.token.txt` (5 captions per image),
   plus `Flickr_8k.{train,dev,test}Images.txt` (6,000 / 1,000 / 1,000 image ids).

2. Precompute CLIP features once into `data/flickr8k/features/{train,dev,test}/`:

   ```bash
   python scripts/prepare_flickr8k_features.py
   ```

   Uses HuggingFace `openai/clip-vit-base-patch32` (installed via `transformers`) to embed
   each image and each caption as an L2-normalized 512-d vector, so data matches the flat
   feature format the models consume.

## Usage

All commands go through the single entry point:

```bash
python main.py <command> [options]
```

Every training command shares: `--epochs`, `--lr`, `--batch-size`,
`--seed` (and `--dataset {synthetic,flickr8k}`).

### Quick verification (synthetic, end-to-end on CPU)

```bash
python main.py test_synthetic
```

Trains CLIP, the SSD teacher, and the SSD students for 2 epochs on synthetic data and
prints a comparison summary.

### Train

**CLIP baseline**

```bash
# synthetic
python main.py train_clip --epochs 5 --batch_size 64 --save_path checkpoints/clip_model.pt

# flickr8k
python main.py train_clip --dataset flickr8k --epochs 30 --batch_size 64 \
  --save_path checkpoints/clip_flickr8k.pt
```

**SSD teacher**

```bash
# synthetic
python main.py train_teacher --epochs 5 --batch_size 64 --beta_teacher 0.5 \
  --save_path checkpoints/ssd_teacher.pt

# flickr8k
python main.py train_teacher --dataset flickr8k --epochs 30 --batch_size 64 \
  --save_path checkpoints/ssd_teacher_flickr8k.pt
```

**SSD students (from a trained teacher)**

```bash
# synthetic
python main.py distill --teacher_checkpoint checkpoints/ssd_teacher.pt --epochs 5 \
  --save_path_img checkpoints/ssd_img.pt --save_path_txt checkpoints/ssd_txt.pt

# flickr8k
python main.py distill --dataset flickr8k \
  --teacher_checkpoint checkpoints/ssd_teacher_flickr8k.pt --epochs 30 \
  --save_path_img checkpoints/ssd_img_flickr8k.pt \
  --save_path_txt checkpoints/ssd_txt_flickr8k.pt
```

### Evaluation

```bash
# CLIP
python main.py eval --model clip --checkpoint checkpoints/clip_model.pt
python main.py eval --dataset flickr8k --model clip --checkpoint checkpoints/clip_flickr8k.pt --split test

# SSD students
python main.py eval --model ssd \
  --student_checkpoint_img checkpoints/ssd_img.pt \
  --student_checkpoint_txt checkpoints/ssd_txt.pt
python main.py eval --dataset flickr8k --model ssd --split test \
  --student_checkpoint_img checkpoints/ssd_img_flickr8k.pt \
  --student_checkpoint_txt checkpoints/ssd_txt_flickr8k.pt
```

`--split` (flickr8k only) selects the evaluation split: `train`, `dev`, or `test` (default `test`).

## Outputs

- **Checkpoints** are saved by `--save_path` (defaults under `checkpoints/`). The teacher's
  config, including feature dims, is stored in its checkpoint and reused by `distill`.
- **Evaluation** saves `tsne_clip.png` / `tsne_ssd.png` and prints:
  - modality gap (Euclidean distance between modality means),
  - latent variance,
  - image-to-text and text-to-image R@1/R@5/R@10,
  - example retrievals.

## Notes/caveats

- Retrieval metrics treat each batch row as a 1:1 image/caption pair. On Flickr8k (5
  captions per image) a row’s own caption is the only “correct” match, so R@K is
  undercounted relative to the standard Flickr8k protocol.
- The precompute step downloads the CLIP weights (~600 MB) on first use.