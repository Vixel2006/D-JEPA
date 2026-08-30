# Multimodal Model (ViT + BERT + Fusion + Decoders)

## Training

### Quick Start

```bash
# Activate virtual environment
source .venv/bin/activate

# Run with defaults
python main.py
```

### Custom Training

```bash
python main.py \
  --epochs 10 \
  --batch-size 32 \
  --lr 1e-4 \
  --steps 10000 \
  --log-every 10 \
  --eval-every 100 \
  --save-every 500 \
  --checkpoint-dir ./checkpoints \
  --tensorboard-dir ./tensorboard_logs
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 2 | Number of epochs |
| `--steps` | None | Max steps (overrides epochs) |
| `--batch-size` | 4 | Batch size |
| `--lr` | 1e-4 | Learning rate |
| `--log-every` | 5 | Log metrics every N steps |
| `--eval-every` | 20 | Evaluate every N steps |
| `--save-every` | 500 | Save checkpoint every N steps |
| `--checkpoint-dir` | ./checkpoints | Checkpoint directory |
| `--tensorboard-dir` | ./tensorboard_logs | TensorBoard log directory |
| `--checkpoint N` | None | Resume from step N (or latest if omitted) |
| `--no-reconstruction` | False | Disable reconstruction losses |

### Resume Training

```bash
# Resume from latest checkpoint
python main.py --checkpoint-dir ./checkpoints

# Resume from specific step
python main.py --checkpoint 5000 --checkpoint-dir ./checkpoints
```

## TensorBoard

```bash
tensorboard --logdir ./tensorboard_logs
```

Then open **http://localhost:6006** in your browser.

## Project Structure

```
src/
├── models/              # Model definitions only
│   ├── vit.py              # Vision Transformer
│   ├── bert.py             # BERT encoder
│   ├── fusion.py           # Multimodal fusion
│   ├── image_decoder.py    # Image reconstruction decoder
│   ├── text_decoder.py     # Text reconstruction decoder
│   ├── multimodal_model.py # Full model composition
│   └── components/         # Shared components (attention, blocks, RoPE)
├── training/          # Training logic
│   ├── trainer.py          # Trainer class, train/eval steps
│   ├── train_config.py     # Training hyperparameters
│   └── optimizer.py        # Optimizer creation
├── losses/            # Loss functions
│   └── loss.py             # KL divergence, reconstruction losses
└── main.py              # Training entry point
```

## Notes

- **Dummy data** is used by default (`create_dummy_dataloader` in `main.py`). Replace with real data for actual training.
- Checkpoints saved via **Orbax** to `./checkpoints`
- TensorBoard logs written to `./tensorboard_logs`
- Requires: `flax`, `optax`, `chex`, `orbax-checkpoint`, `tensorboard`, `grain`