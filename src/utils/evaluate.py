from pathlib import Path

import torch

from .evaluation import (
    encode_vision,
    factor_probe_metrics,
    joint_synergy_metrics,
    representation_health,
    save_pid_inputs,
)
from .figures import plot_factor_probes


@torch.inference_mode()
def _collect_features(
    loader,
    vision_encoder,
    text_encoder,
    device,
    image_size,
    objective,
    factorcl_model,
):
    names = (
        "vision_embedding", "text_embedding", "embedding_joint", "c_vision",
        "u_vision", "c_text", "u_text", "vision_joint", "text_joint",
        "c_joint", "u_joint", "all_joint",
    )
    features = {name: [] for name in names}
    factors = {}

    for batch in loader:
        images = batch["context"]["vision"].to(device)
        text = batch["context"]["text"].to(device)
        if objective == "factorcl":
            (
                embedding_vision,
                embedding_text,
                c_vision,
                u_vision,
                c_text,
                u_text,
            ) = factorcl_model.representations(images, text)
        else:
            embedding_vision, c_vision, u_vision = encode_vision(
                vision_encoder, images, image_size
            )
            embedding_text, c_text, u_text = text_encoder(text)
        current = {
            "vision_embedding": embedding_vision,
            "text_embedding": embedding_text,
            "embedding_joint": torch.cat((embedding_vision, embedding_text), dim=-1),
        }
        current.update(
            {
                "c_vision": c_vision,
                "u_vision": u_vision,
                "c_text": c_text,
                "u_text": u_text,
                "vision_joint": torch.cat((c_vision, u_vision), dim=-1),
                "text_joint": torch.cat((c_text, u_text), dim=-1),
                "c_joint": torch.cat((c_vision, c_text), dim=-1),
                "u_joint": torch.cat((u_vision, u_text), dim=-1),
                "all_joint": torch.cat(
                    (c_vision, u_vision, c_text, u_text), dim=-1
                ),
            }
        )
        for name, value in current.items():
            features[name].append(value.cpu())
        for name, value in batch["factors"].items():
            factors.setdefault(name, []).append(value)

    features = {name: torch.cat(values).numpy() for name, values in features.items()}
    factors = {name: torch.cat(values).numpy() for name, values in factors.items()}
    return features, factors


def evaluate_representation(
    train_loader,
    validation_loader,
    vision_encoder=None,
    text_encoder=None,
    *,
    device,
    image_size,
    objective="decomposition",
    factorcl_model=None,
    output_dir=".",
    wandb_run=None,
):
    """Probe and save the learned representation after pretraining."""
    if vision_encoder is not None:
        vision_encoder.eval()
    if text_encoder is not None:
        text_encoder.eval()
    if factorcl_model is not None:
        factorcl_model.eval()
    train_features, train_factors = _collect_features(
        train_loader,
        vision_encoder,
        text_encoder,
        device,
        image_size,
        objective,
        factorcl_model,
    )
    validation_features, validation_factors = _collect_features(
        validation_loader,
        vision_encoder,
        text_encoder,
        device,
        image_size,
        objective,
        factorcl_model,
    )

    metrics = factor_probe_metrics(
        train_features, train_factors, validation_features, validation_factors
    )
    metrics.update(
        joint_synergy_metrics(
            train_features, train_factors, validation_features, validation_factors
        )
    )
    metrics.update(representation_health(validation_features))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pid_path = output_dir / "pid_inputs.npz"
    figure_path = output_dir / "factor_probes.png"
    save_pid_inputs(pid_path, validation_features, validation_factors)
    plot_factor_probes(metrics, figure_path)

    if wandb_run is not None:
        wandb_run.log(metrics)
        import wandb

        wandb_run.log({"figures/factor_probes": wandb.Image(str(figure_path))})

    print("Evaluation:", metrics)
    return metrics
