# D-JEPA

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

Partial-information decomposition of multimodal joint-embedding pretraining.
The benchmark compares **D-JEPA** against **alignment-only** representations on
a synthetic ground-truth dataset and on real CLEVR.

All learned components — token encoders, MLPs, probes, normalizing flows, and
every training/evaluation step — are implemented in **JAX, Flax, and Optax**.
PyTorch/Transformers are used only as optional frozen feature extractors for
the real CLEVR images and captions. Grain streams deterministic pretraining
batches, Orbax stores checkpoints, and every run is driven by one typed config.

The code is deliberately small: one module per concern, one config-driven CLI,
and a `report.json` schema that can regenerate most figures without retraining.

## What it does

1. **Data** — a synthetic multimodal dataset with known shared, vision-private,
   text-private, and synergistic factors, or real CLEVR from
   `HuggingFaceM4/the_cauldron` encoded with frozen DINOv2 and BERT towers.
2. **Pretraining** — paired context and target views share one encoder and
   projection heads. The objective combines LeJEPA view prediction and
   per-view SIGReg (`--sigreg-weight`, default `0.05`) with redundant
   cross-modal alignment and within-modality unique/redundant cross-covariance
   (`--alignment-weight` and `--covariance-weight`, default `1.0`). It has no
   EMA target tower or stop-gradient branch.
3. **Evaluation** — capacity-matched Flax probes per feature group,
   paired-bootstrap differences, modality-gap analysis, collapse diagnostics,
   ridge factor identification, Procrustes/CKA/mutual-kNN alignment, causal
   block swaps, and two PID estimators (probe-PID for discrete tasks, flow-PID
   with Flax affine coupling flows for continuous targets).
4. **Observability** — run metadata, resumable Orbax checkpoints,
   tidy CSV tables, and eight publication figures.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Optional TensorBoard logging
pip install -e ".[dev]"

# Optional frozen extractors for the real CLEVR benchmark
pip install -e ".[dev,clevr]"

# Optional MultiBench MM-IMDb and UR-FUNNY downloaders / PyTorch DataLoaders
pip install -e ".[data]"
```

### MM-IMDb and UR-FUNNY

The MultiBench helpers download the processed MM-IMDb HDF5 archive from
Archive.org and the processed UR-FUNNY `humor.pkl` from MultiBench's Google
Drive folder. The latter uses `gdown`, included in the `data` extra. Files are
stored under `data/multibench/` by default and are not included in the source
package.

```python
from pidjepa.data import (
    download_multibench_dataset,
    get_mmimdb_dataloaders,
    get_ur_funny_dataloaders,
)

mmimdb_path = download_multibench_dataset("mm-imdb")
mmimdb_train, mmimdb_valid, mmimdb_test = get_mmimdb_dataloaders(
    mmimdb_path, batch_size=32
)

ur_funny_path = download_multibench_dataset("ur-funny")
funny_train, funny_valid, funny_test = get_ur_funny_dataloaders(
    ur_funny_path, batch_size=32, max_seq_len=50
)
```

MM-IMDb batches contain `(text_features, image_features, genre_labels)` and
use MultiBench's fixed train/validation/test boundaries. UR-FUNNY batches are
dictionaries with `text`, `vision`, `audio`, and `label` tensors. If the Drive
folder layout changes, download its processed `humor.pkl` manually and pass
that file path to `get_ur_funny_dataloaders`.

## Run

Every run is described by a JSON/YAML config. Example configs live in
[`configs/`](configs):

```bash
# Tiny end-to-end smoke test
python main.py --config configs/smoke_synthetic.yaml

# Synthetic ground-truth benchmark (3 seeds)
python main.py --config configs/synthetic_paper.yaml

# CLEVR benchmark (downloads DINOv2, BERT, and the Cauldron stream)
python main.py --config configs/clevr_paper.yaml
```

Override any setting with a dotted `--set` without editing the file:

```bash
python main.py --config configs/synthetic_paper.yaml \
  --set run.seeds=5 \
  --set optim.pretrain_steps=500 \
  --set run.output_dir=results/pilot

# The quick profile is a structurally complete smoke run
python main.py --config configs/synthetic_paper.yaml --quick
```

`python main.py` is the entry point (or `python -m pidjepa`). Use
`--print-config` to inspect the fully resolved configuration before training.

Training and evaluation are decoupled: pretrain once, then evaluate later
against the saved checkpoints.

```bash
# 1. Pretrain every variant (writes checkpoints under <output-dir>/checkpoints)
python main.py --config configs/synthetic_paper.yaml --mode train

# 2. Later, evaluate from those checkpoints (probes, PID, figures, report)
python main.py --config configs/synthetic_paper.yaml --mode eval \
    --output-dir results/synthetic_paper

# Or do both in one run (default: --mode train+eval)
python main.py --config configs/synthetic_paper.yaml
```

### CLI reference

| Flag                | Meaning                                                       |
| ------------------- | ------------------------------------------------------------- |
| `--config PATH`     | JSON or YAML experiment config (defaults are used if omitted) |
| `--set KEY=VALUE`   | Dotted override, repeatable (`optim.pretrain_steps=500`)      |
| `--mode MODE`       | `train`, `eval` (pretrained checkpoints), or `train+eval`      |
| `--set KEY=VALUE`   | Dotted override, repeatable (`optim.pretrain_steps=500`)      |
| `--quick`           | Small but structurally complete smoke profile                 |
| `--seeds N`         | Override `run.seeds` and clear any explicit `run.seed_list`   |
| `--output-dir PATH` | Override `run.output_dir`                                     |
| `--device DEVICE`   | Device for optional CLEVR feature extraction                  |
| `--resume`          | Resume every variant from its latest checkpoint               |
| `--overwrite`       | Allow writing into a directory that already has `report.json` |
| `--no-figures`      | Skip figure generation                                        |
| `--no-checkpoints`  | Disable checkpointing                                         |
| `--profile`         | Write a JAX profiler trace under `<output-dir>/profile`       |
| `--debug-nans`      | Abort if a training step produces non-finite values           |
| `--print-config`    | Print the resolved config and exit                            |

### Config sections

| Section      | Controls                                                  |
| ------------ | --------------------------------------------------------- |
| `data`       | Synthetic/CLEVR mode, sample counts, views, token limits  |
| `model`      | Block width, hidden width, transformer depth/heads        |
| `optim`      | Steps, batch size, AdamW schedule, checkpoint cadence     |
| `objective`  | Loss weights and SIGReg quadrature                        |
| `evaluation` | Probe budget, PID dimensions/steps, flow architecture     |
| `run`        | Seeds, output directory, checkpoints, figures, logging |

The config is validated on load. Unknown keys and invalid values fail fast, and
every run records a `config_fingerprint` that excludes non-result settings such
as output paths and devices.

## Outputs

```text
results/
  config.json                          fully resolved config
  run.json                             command, git state, seed and dataset fingerprints
  report.json                          aggregated metrics (accuracy, PID, alignment, ...)
  tables/                              one tidy CSV per report section
  checkpoints/seed_<seed>/<variant>/   latest Orbax checkpoints (step_NNNNNNNN)
  fig1_per_category_bars.png           per-question-type accuracy (CLEVR)
  fig2_modality_gap.png                joint - max(single-stream) synergy
  fig3_factor_dissection_heatmap.png   feature groups -> factors (R^2)
  fig4_probe_block_attribution.png     downstream probe weight per D-JEPA block
  fig5_tsne_latent_dissection.png      t-SNE of c/u blocks
  fig6_probe_pid.png                   probe-PID information allocation
  fig7_flow_pid.png                    flow-PID information allocation
  fig8_training_curves.png             loss components over pretraining
```

Figures 2, 3, 6, and 7 are pure functions of `report.json` and can be
regenerated without retraining:

```bash
python scripts/regen_figures.py results/report.json
```

## Reproducing and comparing runs

- The resolved config is saved as `config.json` and its hash is stored in
  `run.json` and `report.json`.
- Pretraining batches are addressed by global step, so resuming from a
  checkpoint continues the exact same Grain stream.
- `--resume` loads the latest Orbax checkpoint for each variant (written and
  restored directly with `orbax.checkpoint.StandardCheckpointer`).
- Synthetic data generation and all evaluation permutations are deterministic
  given the seed. The CLEVR split is grouped by image so no scene leaks between
  train and test.
- The synthetic split is by sample; every variant shares the same optimizer,
  schedule, and capacity-matched probes.

## Run on Google Colab

`scripts/colab_run.sh` packs the working tree and configs, creates a fresh
`colab-cli` session, installs dependencies, runs `python main.py`, downloads
the outputs, and stops the VM:

```bash
# Quick synthetic verification
GPU=T4 scripts/colab_run.sh --config configs/smoke_synthetic.yaml

# Full CLEVR paper run
GPU=A100 TIMEOUT=14400 scripts/colab_run.sh --config configs/clevr_paper.yaml
```

Pass any CLI arguments after the script; they are forwarded to `pidjepa`:

```bash
GPU="" scripts/colab_run.sh \
  --config configs/synthetic_paper.yaml \
  --set optim.pretrain_steps=100 \
  --set run.output_dir=results/colab_pilot
```

Environment overrides: `SESSION` (session name), `GPU` (`T4`, `L4`, `A100`,
or empty for CPU), `TIMEOUT` (remote timeout), and `KEEP=1` to keep the VM
alive for debugging. The archive lands in `outputs/colab_<timestamp>.tar.gz`
and contains the full results tree plus `run.log`.

## Citation

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). Update the
author list and repository URL before publishing a paper release.

## License

Released under the [MIT License](LICENSE).

## Acknowledgements

This benchmark builds on DINOv2, BERT, CLEVR, the Cauldron dataset, and the
SIGReg objective from the LeJEPA line of work.
