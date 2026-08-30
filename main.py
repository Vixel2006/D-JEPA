#!/usr/bin/env python
"""Main training entry point for multimodal model (ViT + BERT + Fusion + Decoders)."""

import argparse
import jax
import jax.numpy as jnp
import flax.nnx as nnx
from src.models import (
    ViTConfig,
    BERTConfig,
    MultimodalFusionConfig,
    ImageDecoderConfig,
    TextDecoderConfig,
    MultimodalModel,
)
from src.training import (
    TrainConfig,
    Trainer,
    eval_step,
)


def create_dummy_dataloader(config: TrainConfig, split: str = "train"):
    """Create dummy data loader for testing."""
    rng = jax.random.PRNGKey(config.seed + (0 if split == "train" else 1))

    def gen():
        nonlocal rng
        while True:
            rng, key1, key2, key3 = jax.random.split(rng, 4)
            images = jax.random.normal(key1, (config.batch_size, 224, 224, 3))
            input_ids = jax.random.randint(key2, (config.batch_size, 128), 0, 30522)
            yield (images, input_ids)

    return gen()


def main():
    parser = argparse.ArgumentParser(
        description="Train multimodal model with reconstruction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--steps", type=int, default=None, help="Max steps (overrides epochs)")
    parser.add_argument("--log-every", type=int, default=5, help="Log every N steps")
    parser.add_argument("--eval-every", type=int, default=20, help="Evaluate every N steps")
    parser.add_argument("--save-every", type=int, default=500, help="Save checkpoint every N steps")
    parser.add_argument("--kl-weight", type=float, default=1.0, help="Weight for KL divergence loss")
    parser.add_argument("--image-recon-weight", type=float, default=1.0, help="Weight for image reconstruction loss")
    parser.add_argument("--text-recon-weight", type=float, default=1.0, help="Weight for text reconstruction loss")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--checkpoint", type=int, default=None, help="Checkpoint step to resume from (default: latest)")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints", help="Checkpoint directory")
    parser.add_argument("--tensorboard-dir", type=str, default="./tensorboard_logs", help="TensorBoard log directory")
    parser.add_argument("--use-reconstruction", action="store_true", default=True, help="Use reconstruction losses")
    parser.add_argument("--no-reconstruction", dest="use_reconstruction", action="store_false", help="Disable reconstruction losses")
    args = parser.parse_args()

    train_config = TrainConfig(
        seed=args.seed,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        log_every=args.log_every,
        eval_every=args.eval_every,
        save_every=args.save_every,
        kl_weight=args.kl_weight,
        image_recon_weight=args.image_recon_weight,
        text_recon_weight=args.text_recon_weight,
        grad_clip=args.grad_clip,
        weight_decay=args.weight_decay,
        checkpoint_dir=args.checkpoint_dir,
        tensorboard_dir=args.tensorboard_dir,
    )

    print("=" * 60)
    print("Training Configuration")
    print("=" * 60)
    for k, v in train_config.__dict__.items():
        print(f"  {k}: {v}")
    print("=" * 60)

    rngs = nnx.Rngs(train_config.seed)

    print("\nCreating models...")
    model = MultimodalModel(rngs,
        vit_config=ViTConfig(
            img_size=224, patch_size=16, in_channels=3,
            num_classes=1000, d_model=512, num_layers=4, num_heads=8,
            use_slots=False,
        ),
        bert_config=BERTConfig(
            vocab_size=30522, max_seq_len=512, num_classes=2,
            d_model=512, num_layers=4, num_heads=8,
            use_slots=False,
        ),
        fusion_config=MultimodalFusionConfig(
            vit_d_model=512, bert_d_model=512, joint_d_model=512,
            num_heads=8, num_layers=2,
        ),
        image_decoder_config=ImageDecoderConfig(
            joint_d_model=512, img_size=224, patch_size=16, in_channels=3,
            d_model=768, num_layers=4, num_heads=12,
        ),
        text_decoder_config=TextDecoderConfig(
            joint_d_model=512, vocab_size=30522, max_seq_len=512,
            d_model=768, num_layers=4, num_heads=12,
        ),
    )

    def count_params(model):
        return sum(x.size for x in jax.tree.leaves(nnx.state(model)))

    print(f"  Total: {count_params(model):,} params")

    trainer = Trainer(train_config, model)

    if args.checkpoint is not None:
        trainer.load_checkpoint(step=args.checkpoint)
    elif args.checkpoint is None:
        trainer.load_checkpoint()  # Load latest

    print("\nCreating data loaders...")
    train_loader = create_dummy_dataloader(train_config, "train")
    eval_loader = create_dummy_dataloader(train_config, "eval")

    print("\nStarting training...")
    max_steps = args.steps if args.steps else float('inf')
    step = 0

    for epoch in range(train_config.num_epochs):
        trainer.epoch = epoch
        print(f"\n--- Epoch {epoch + 1}/{train_config.num_epochs} ---")

        for batch in train_loader:
            step += 1
            trainer.step = step

            loss, metrics = trainer.train_step(batch, use_reconstruction=args.use_reconstruction)

            if step % train_config.eval_every == 0:
                print("\nEvaluating...")
                total_loss = 0.0
                total_metrics = {}
                num_batches = 0
                for _ in range(5):  # 5 eval batches
                    eval_batch = next(eval_loader)
                    loss, metrics = eval_step(trainer.model, eval_batch, 
                        kl_weight=train_config.kl_weight,
                        image_recon_weight=train_config.image_recon_weight,
                        text_recon_weight=train_config.text_recon_weight,
                        use_reconstruction=args.use_reconstruction,
                    )
                    total_loss += loss
                    for k, v in metrics.items():
                        total_metrics[k] = total_metrics.get(k, 0.0) + v
                    num_batches += 1
                avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
                avg_metrics["eval_loss"] = total_loss / num_batches
                prefix_str = "eval/"
                metric_str = " | ".join(f"{prefix_str}{k}: {float(v):.4f}" for k, v in avg_metrics.items())
                print(f"Step {step} | {metric_str}")

            if step % train_config.save_every == 0:
                trainer.save_checkpoint(step=step)

            if step >= max_steps:
                break

        if step >= max_steps:
            break

    print("\nTraining complete!")
    trainer.save_checkpoint(step=step)
    trainer.close()


if __name__ == "__main__":
    main()