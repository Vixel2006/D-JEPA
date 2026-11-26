import torch
import tqdm
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

if __package__:
    from .data.synthetic import SyntheticDataset
    from .evaluate import evaluate_representation
    from .loss import alignment_loss, independence_loss, jepa_loss
    from .model.encoders import BertEncoder, ViTEncoder
    from .model.projectors import Projector
    from .model.sigreg import SIGReg
    from .utils.evaluation import encode_vision
else:
    from data.synthetic import SyntheticDataset
    from evaluate import evaluate_representation
    from loss import alignment_loss, independence_loss, jepa_loss
    from model.encoders import BertEncoder, ViTEncoder
    from model.projectors import Projector
    from model.sigreg import SIGReg
    from utils.evaluation import encode_vision


def train(cfg: DictConfig):
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run = wandb.init(
        project=cfg.wandb.project,
        mode=cfg.wandb.mode,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    train_dataset = SyntheticDataset(
        n_samples=cfg.data.n_samples,
        seed=cfg.seed,
        split="train",
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.training.num_workers,
    )
    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
    )
    validation_dataset = SyntheticDataset(
        n_samples=cfg.data.n_samples,
        seed=cfg.seed,
        split="validation",
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

    vision_encoder = ViTEncoder(
        img_size=cfg.model.image_size, proj_dim=cfg.model.proj_dim
    ).to(device)
    text_encoder = BertEncoder(proj_dim=cfg.model.proj_dim).to(device)

    vision_projector = Projector(cfg.model.proj_dim, 512).to(device)
    text_projector = Projector(cfg.model.proj_dim, 256).to(device)
    sigreg = SIGReg().to(device)

    optimizer = torch.optim.AdamW(
        [
            *vision_encoder.parameters(),
            *text_encoder.parameters(),
            *vision_projector.parameters(),
            *text_projector.parameters(),
        ],
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * cfg.training.epochs
    warmup_steps = min(steps_per_epoch, max(total_steps - 1, 0))
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=max(warmup_steps, 1)
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_steps - warmup_steps, 1)
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup, cosine], milestones=[warmup_steps]
    )

    for epoch in range(cfg.training.epochs):
        vision_encoder.train()
        text_encoder.train()
        vision_projector.train()
        text_projector.train()
        totals = {"jepa": 0.0, "alignment": 0.0, "independence": 0.0, "sigreg": 0.0}

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

            _, c_vision, u_vision = encode_vision(
                vision_encoder, images, cfg.model.image_size
            )
            _, c_text, u_text = text_encoder(text)
            prediction_vision, z_vision = vision_projector(c_vision, u_vision)
            prediction_text, z_text = text_projector(c_text, u_text)

            target_vision, _, _ = encode_vision(
                vision_encoder, target_images, cfg.model.image_size
            )
            target_text, _, _ = text_encoder(target_text)

            loss_jepa = jepa_loss(prediction_vision, target_vision) + jepa_loss(
                prediction_text, target_text
            )
            loss_alignment = alignment_loss(c_vision, c_text)
            loss_independence = independence_loss(
                c_vision, u_vision
            ) + independence_loss(c_text, u_text)
            loss_sigreg = (
                sigreg(c_vision) + sigreg(c_text) + sigreg(z_vision) + sigreg(z_text)
            )
            loss = (
                loss_jepa
                + cfg.loss.alignment * loss_alignment
                + cfg.loss.independence * loss_independence
                + cfg.loss.sigreg * loss_sigreg
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            totals["jepa"] += loss_jepa.item()
            totals["alignment"] += loss_alignment.item()
            totals["independence"] += loss_independence.item()
            totals["sigreg"] += loss_sigreg.item()
            progress.set_postfix(loss=f"{loss.item():.3f}")

        metrics = {
            f"train/{name}": value / len(train_loader) for name, value in totals.items()
        }
        metrics["train/loss"] = (
            metrics["train/jepa"]
            + cfg.loss.alignment * metrics["train/alignment"]
            + cfg.loss.independence * metrics["train/independence"]
            + cfg.loss.sigreg * metrics["train/sigreg"]
        )
        metrics["epoch"] = epoch
        wandb.log(metrics)

    evaluate_representation(
        train_eval_loader,
        validation_loader,
        vision_encoder,
        text_encoder,
        vision_projector,
        text_projector,
        device=device,
        image_size=cfg.model.image_size,
        wandb_run=run,
    )

    torch.save(
        {
            "vision_encoder": vision_encoder.state_dict(),
            "text_encoder": text_encoder.state_dict(),
            "vision_projector": vision_projector.state_dict(),
            "text_projector": text_projector.state_dict(),
            "config": OmegaConf.to_container(cfg, resolve=True),
        },
        "checkpoint.pt",
    )
    run.finish()


if __name__ == "__main__":
    train()
