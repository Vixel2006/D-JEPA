from pathlib import Path

import torch
import tqdm
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, default_collate

import wandb

from ..data.datasets import make_dataset_pair
from ..utils.evaluate import evaluate_representation
from .models import FactorCLModel


def train(cfg: DictConfig, *, output_dir=".", run_name=None):
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run = wandb.init(
        project=cfg.wandb.project,
        mode=cfg.wandb.mode,
        name=run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    train_dataset, validation_dataset = make_dataset_pair(
        cfg.data.get("dataset", "synthetic"),
        seed=cfg.seed,
        n_samples=cfg.data.get("n_samples", 1024),
        image_size=cfg.model.image_size,
        max_text_length=cfg.data.get("max_text_length", 128),
        tokenizer_name=cfg.data.get("tokenizer", "bert-base-uncased"),
        text_mask_probability=cfg.data.get("text_mask_probability", 0.08),
        image_cache_dir=(
            cfg.data.get("cache_dir", "cache")
            if cfg.data.get("cache_images", False) else None
        ),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.training.num_workers,
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
    )
    if not len(train_loader):
        raise ValueError(
            "Training dataset is smaller than one full batch; lower the batch size."
        )
    if cfg.training.batch_size < 2:
        raise ValueError("FactorCL contrastive objectives require batch_size >= 2.")
    if cfg.factorcl.num_critic_steps < 1:
        raise ValueError("factorcl.num_critic_steps must be positive.")
    # Keep corresponding modality backbones identically initialized across objectives.
    torch.manual_seed(cfg.seed)
    model = FactorCLModel(
        image_size=cfg.model.image_size,
        critic_hidden_dim=cfg.factorcl.critic_hidden_dim,
    ).to(device)
    critic_batch_generator = torch.Generator().manual_seed(cfg.seed + 2)
    optimizer = torch.optim.Adam(
        model.non_club_parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.factorcl.weight_decay,
    )
    club_optimizer = torch.optim.Adam(
        model.club_parameters(), lr=cfg.training.learning_rate
    )

    for epoch in range(cfg.training.epochs):
        model.train()
        totals = {
            "factorcl_total": 0.0,
            "factorcl_shared_nce": 0.0,
            "factorcl_unconditional_club": 0.0,
            "factorcl_unique_vision_nce": 0.0,
            "factorcl_unique_text_nce": 0.0,
            "factorcl_conditional_nce": 0.0,
            "factorcl_conditional_club": 0.0,
            "factorcl_critic": 0.0,
        }

        progress = tqdm.tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{cfg.training.epochs}"
        )
        for batch in progress:
            context = batch["context"]
            target = batch["target"]
            images = context["vision"].to(device, non_blocking=True)
            text = context["text"].to(device, non_blocking=True)
            target_images = target["vision"].to(device, non_blocking=True)
            target_text = target["text"].to(device, non_blocking=True)
            loss, components = model(images, text, target_images, target_text)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            critic_loss_value = 0.0
            for _ in range(cfg.factorcl.num_critic_steps):
                critic_indices = torch.randint(
                    len(train_dataset),
                    (cfg.training.batch_size,),
                    generator=critic_batch_generator,
                ).tolist()
                critic_batch = default_collate(
                    [train_dataset[index] for index in critic_indices]
                )
                critic_context = critic_batch["context"]
                critic_target = critic_batch["target"]
                critic_images = critic_context["vision"].to(device)
                critic_tokens = critic_context["text"].to(device)
                critic_target_images = critic_target["vision"].to(device)
                critic_target_tokens = critic_target["text"].to(device)
                critic_loss = model.critic_learning_loss(
                    critic_images,
                    critic_tokens,
                    critic_target_images,
                    critic_target_tokens,
                )
                club_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward()
                club_optimizer.step()
                critic_loss_value += critic_loss.item()

            totals["factorcl_total"] += loss.item()
            totals["factorcl_critic"] += critic_loss_value / cfg.factorcl.num_critic_steps
            for name, component in components.items():
                totals[f"factorcl_{name}"] += component.item()

            progress.set_postfix(loss=f"{loss.item():.3f}")

        metrics = {
            f"train/{name}": value / len(train_loader) for name, value in totals.items()
        }
        metrics["train/loss"] = metrics["train/factorcl_total"]
        metrics["epoch"] = epoch
        wandb.log(metrics)

    evaluate_representation(
        train_eval_loader,
        validation_loader,
        factorcl_model=model,
        objective="factorcl",
        device=device,
        image_size=cfg.model.image_size,
        output_dir=output_dir,
        wandb_run=run,
    )
    checkpoint = {
        "factorcl_model": model.state_dict(),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }

    torch.save(
        checkpoint,
        output_dir / "checkpoint.pt",
    )
    run.finish()
