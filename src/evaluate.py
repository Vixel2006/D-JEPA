from pathlib import Path

import torch

if __package__:
    from .utils.evaluation import (
        encode_vision,
        factor_probe_metrics,
        representation_health,
        save_pid_inputs,
    )
    from .utils.figures import plot_factor_probes
else:
    from utils.evaluation import (
        encode_vision,
        factor_probe_metrics,
        representation_health,
        save_pid_inputs,
    )
    from utils.figures import plot_factor_probes


@torch.inference_mode()
def _collect_features(
    loader,
    vision_encoder,
    text_encoder,
    vision_projector,
    text_projector,
    device,
    image_size,
):
    features = {
        name: []
        for name in ("c_vision", "u_vision", "z_vision", "c_text", "u_text", "z_text")
    }
    factors = {}

    for batch in loader:
        images = batch["context"]["vision"].to(device)
        text = batch["context"]["text"].to(device)
        _, c_vision, u_vision = encode_vision(vision_encoder, images, image_size)
        _, c_text, u_text = text_encoder(text)
        _, z_vision = vision_projector(c_vision, u_vision)
        _, z_text = text_projector(c_text, u_text)

        current = {
            "c_vision": c_vision,
            "u_vision": u_vision,
            "z_vision": z_vision,
            "c_text": c_text,
            "u_text": u_text,
            "z_text": z_text,
        }
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
    vision_encoder,
    text_encoder,
    vision_projector,
    text_projector,
    *,
    device,
    image_size,
    output_dir=".",
    wandb_run=None,
):
    """Probe and save the learned representation after pretraining."""
    for model in (vision_encoder, text_encoder, vision_projector, text_projector):
        model.eval()

    train_features, train_factors = _collect_features(
        train_loader,
        vision_encoder,
        text_encoder,
        vision_projector,
        text_projector,
        device,
        image_size,
    )
    validation_features, validation_factors = _collect_features(
        validation_loader,
        vision_encoder,
        text_encoder,
        vision_projector,
        text_projector,
        device,
        image_size,
    )

    metrics = factor_probe_metrics(
        train_features, train_factors, validation_features, validation_factors
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
