# PID-JEPA / D-JEPA CLEVR

A modular CLEVR benchmark for **LeJEPA** vs. **Decomposed LeJEPA
(D-JEPA)**. It runs on the real Blender-rendered CLEVR split (streamed from
Hugging Face) or on an offline procedural fallback, and produces the
multi-seed tables plus a full publication figure suite.

## What the pipeline does

1. **Data** — real CLEVR questions from
   `HuggingFaceM4/the_cauldron` (`clevr` configuration), or a deterministic
   procedural scene generator. Questions are grouped into `counting`,
   `attribute`, and `spatial_relation`.
2. **Backbones** — frozen DINOv2 ViT-S/14 vision encoder (ResNet-18 fallback)
   and frozen DistilBERT text encoder.
3. **Pretraining** — LeJEPA (one shared latent) and D-JEPA (shared
   code `c`, private fibres `u1`, `u2`, and synergy `s`).
4. **Multi-seed probing** — capacity-matched probes for every subspace,
   including `+u1`, `+u2`, `+u1+u2`, full, projected-capacity controls, and a
   random-feature control.
5. **Diagnostics** — causal fibre swaps, Procrustes alignment, Flow-PID,
   covariance eigenspectra, effective rank, linear CKA, calibration (ECE /
   NLL / macro-F1), and confusion matrices.
6. **Figures** — 13 publication-ready PNG and PDF figures.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

The first run downloads DINOv2 through `torch.hub` (retried once) and
DistilBERT through `transformers`; real CLEVR is streamed from the Hugging
Face Hub. If the `torch.hub` load fails, the vision tower tries Hugging Face
`facebook/dinov2-small`, then falls back to a frozen ImageNet ResNet-18. If
the real CLEVR stream fails and `--no-fallback` is not set, the procedural
generator is used instead.

## Run

Real CLEVR (matches the reference script):

```bash
python -m djepa \
  --dataset clevr \
  --max-samples 10000 \
  --output-dir outputs/clevr \
  --seeds 42,43,44
```

Offline / procedural:

```bash
python -m djepa \
  --dataset procedural \
  --scenes 1500 \
  --output-dir outputs/clevr_procedural
```

Quick smoke run:

```bash
python -m djepa \
  --dataset procedural \
  --scenes 20 \
  --batch-size 8 \
  --test-size 8 \
  --pretrain-steps 2 \
  --probe-steps 2 \
  --seeds 42 \
  --device cpu \
  --no-amp
```

Useful options:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--dataset` | `clevr` | `clevr` (Hub) or `procedural` (offline) |
| `--max-samples` | `10000` | Maximum streamed real CLEVR items |
| `--scenes` | `1500` | Procedural scenes |
| `--batch-size` | `64` | Pretraining and probe batch size |
| `--test-size` | auto | Default is `min(800, len(dataset) // 4)` |
| `--max-question-length` | `64` | Tokenizer maximum length |
| `--aug-scale-min` | `0.7` | Minimum `RandomResizedCrop` scale |
| `--pretrain-steps` | `3000` | Joint pretraining steps |
| `--probe-steps` | `600` | Probe steps per seed |
| `--log-every` | `50` | Log training/probe progress every N steps |
| `--seeds` | `42,43,44` | Evaluation seeds |
| `--device` | `auto` | `cpu`, `cuda`, `cuda:0`, or `auto` |
| `--no-figures` | off | Skip the figure suite |
| `--no-pdf` | off | Write PNG figures only |
| `--no-diagnostics` | off | Skip spectral/CKA/calibration analysis |
| `--no-amp` | off | Disable CUDA mixed precision |
| `--no-case-studies` | off | Skip qualitative win/loss search |
| `--no-save` | off | Print tables without writing files |
| `--num-workers` | `0` | Keep `0` for bitwise reproducibility |

## Outputs

```text
outputs/clevr/
  results.json            aggregated metrics, diagnostics, loss history
  accuracy.csv            per-subspace, per-category mean/std
  pid.csv                 Flow-PID rows
  diagnostics.json        spectral, CKA, alignment, calibration
  spectral.csv            effective rank and covariance geometry
  calibration.csv         accuracy, macro-F1, ECE, NLL
  test_records.json       every held-out prediction record
  embeddings.npz          held-out embeddings (z1, z2, c1, c2, u1, u2, s)
  case_studies/           clevr_fig_*.jpg win/loss images
  figures/
    fig01_probe_accuracy_heatmap.{png,pdf}
    fig02_per_category_bars.{png,pdf}
    fig03_fiber_contribution.{png,pdf}
    fig04_tsne_category_separation.{png,pdf}
    fig05_prediction_distribution.{png,pdf}
    fig06_qualitative_grid.{png,pdf}
    fig07_spectral_diagnostics.{png,pdf}
    fig08_cka_matrix.{png,pdf}
    fig09_causal_swap.{png,pdf}
    fig10_pid_decomposition.{png,pdf}
    fig11_training_curves.{png,pdf}
    fig12_calibration_confusion.{png,pdf}
    fig13_alignment_summary.{png,pdf}
```

### Figure guide

- **01 accuracy heatmap** — per-category accuracy for every probe condition.
- **02 per-category bars** — same numbers with seed error bars.
- **03 fibre contribution** — Δ accuracy from adding `u1`, `u2`, `s`.
- **04 t-SNE** — shared code vs. shared + private fibres (PCA fallback when
  scikit-learn is unavailable).
- **05 prediction distribution** — LeJEPA mode collapse vs. D-JEPA.
- **06 qualitative grid** — wins and failure modes per category.
- **07 spectral diagnostics** — eigenspectrum decay and effective rank.
- **08 CKA matrix** — linear CKA between all representation codes.
- **09 causal swap** — prediction flip rates for `u1`, `u2`, and `s`.
- **10 PID decomposition** — stacked unique/redundant/synergy R² terms.
- **11 training curves** — total losses and the D-JEPA loss decomposition.
- **12 calibration & confusion** — ECE by condition and confusion matrices.
- **13 alignment summary** — centered cosine between code pairs.

## Run on Google Colab

`scripts/colab_run.sh` packs the current working tree, creates a fresh
`colab-cli` session, installs dependencies, runs the benchmark, downloads the
outputs, and stops the VM:

```bash
# Tiny end-to-end smoke run (real CLEVR stream, T4 GPU)
scripts/colab_run.sh \
  --dataset clevr --max-samples 40 \
  --batch-size 8 --test-size 8 \
  --pretrain-steps 2 --probe-steps 2 \
  --seeds 42 --device cuda

# Full paper run
GPU=A100 TIMEOUT=14400 scripts/colab_run.sh \
  --dataset clevr --max-samples 10000 \
  --output-dir outputs/clevr \
  --seeds 42,43,44
```

Environment overrides: `SESSION` (session name), `GPU` (`T4`, `L4`, `A100`,
or empty for CPU), `TIMEOUT` (remote timeout), and `KEEP=1` to keep the VM
alive for debugging. Training, probing, inference, PID, and figure progress are
streamed live to the Colab cell (with elapsed time and ETA), and the same
transcript is saved to `outputs/run.log` inside the downloaded archive. The
archive lands in `outputs/colab_<timestamp>.tar.gz` and contains the full
`outputs/` tree. On failure the runner prints the log tail again, preserves the
nonzero exit code, and still downloads any partial outputs.

Use `--log-every N` to control pretraining/probe progress frequency (for
example `--log-every 10`).

## Package layout

```text
src/                        installed as the `djepa` package
  config.py                 typed data/model/optimisation/figure/run configs
  cli.py                    argparse entry point
  pipeline.py               end-to-end orchestration
  diagnostics.py            spectra, effective rank, CKA, calibration
  reporting.py              console tables, JSON/CSV/NPZ artefacts
  results.py                typed result containers
  augment.py                image-patch and token masking
  losses.py                 SIGReg, LeJEPA, covariance decorrelation
  data/
    clevr.py                real CLEVR streaming, procedural fallback
  evaluation/
    probes.py               capacity-matched/ projected probes, causal swaps
    features.py             single held-out feature pass
    alignment.py            Procrustes alignment
    kernels.py              degree-2 kernel ridge R²
    pid.py                  Flow-PID decomposition
    case_studies.py         qualitative win/loss extraction
  figures/
    paper.py                13-figure publication suite
  models/
    backbones.py            frozen DINOv2 and DistilBERT
    projectors.py           projectors, predictor, synergy head
    lejepa.py               LeJEPA baseline
    decomposed.py           D-JEPA
  training/
    pretrain.py             joint pretraining objective + loss history
  utils.py                  seeding, device, AMP, cycle helpers
```

## Reproducibility notes

- The default seeds, step counts, augmentation scale, and answer vocabulary
  match the reference script.
- `--num-workers 0` is required for bitwise reproducibility because the
  dataset consumes the global PyTorch RNG when generating masked views.
- The test split is item-level, exactly as in the reference script. Scenes can
  therefore appear in both train and test; this is preserved for comparability
  and should be changed before using the benchmark for new claims.
- The `RANDOM-*` probe uses a fresh Gaussian vector per batch, matching the
  reference random-feature control.
- Figure generation is defensive: an optional dependency or a degenerate
  small-sample run cannot invalidate an otherwise successful benchmark.
