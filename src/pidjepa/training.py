"""Joint pretraining of the experimental arms and feature extraction."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from pidjepa.models import SIGReg, cross_correlation_loss, pool_tokens

if TYPE_CHECKING:
    from pidjepa.data import Bundle
    from pidjepa.models import MultimodalModel


def pretrain_arm(
    model: MultimodalModel,
    config: dict,
    bundle: Bundle,
    args,
    device: str,
    sigreg: SIGReg,
    name: str,
) -> None:
    """Optimize one arm with invariance, cross-modal, reconstruction, and SIGReg terms."""
    vision_views = bundle.vision_views.to(device)
    text_views = bundle.text_views.to(device)
    text_mask = bundle.text_mask.to(device) if bundle.text_mask is not None else None
    train_indices = bundle.split["train"].to(device)
    n_views = vision_views.size(1)
    if n_views < 3:
        raise ValueError("pretraining requires at least three augmented views")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.05)
    warmup_steps = max(1, int(0.05 * args.pretrain_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: (
            (step + 1) / warmup_steps
            if step < warmup_steps
            else 0.5
            * (
                1
                + math.cos(
                    math.pi * (step - warmup_steps) / max(1, args.pretrain_steps - warmup_steps)
                )
            )
        ),
    )
    model.train()
    sigreg_weight = args.sigreg_weight

    for step in range(1, args.pretrain_steps + 1):
        indices = train_indices[
            torch.randint(len(train_indices), (args.batch_size,), device=device)
        ]
        view_a = torch.randint(1, n_views, (args.batch_size,), device=device)
        view_b = (view_a - 1 + torch.randint(1, n_views - 1, (args.batch_size,), device=device)) % (
            n_views - 1
        ) + 1
        vision_a, vision_b = vision_views[indices, view_a], vision_views[indices, view_b]
        text_a, text_b = text_views[indices, view_a], text_views[indices, view_b]
        mask = text_mask[indices] if text_mask is not None else None

        vision_a_repr = model.vision_encoder(vision_a)
        vision_b_repr = model.vision_encoder(vision_b)
        text_a_repr = model.text_encoder(text_a, mask)
        text_b_repr = model.text_encoder(text_b, mask)

        invariance_loss = F.mse_loss(vision_a_repr, vision_b_repr) + F.mse_loss(
            text_a_repr, text_b_repr
        )

        if config["architecture"] == "alignment":
            cross_modal_loss = F.mse_loss(vision_a_repr, text_a_repr) + F.mse_loss(
                vision_b_repr, text_b_repr
            )
            sigreg_loss = 0.25 * (
                sigreg(vision_a_repr)
                + sigreg(vision_b_repr)
                + sigreg(text_a_repr)
                + sigreg(text_b_repr)
            )
        else:
            vision_shared_a, vision_private_a = model.split_representation(vision_a_repr)
            text_shared_a, text_private_a = model.split_representation(text_a_repr)
            vision_shared_b, vision_private_b = model.split_representation(vision_b_repr)
            text_shared_b, text_private_b = model.split_representation(text_b_repr)

            vision_shared_a_n = F.normalize(vision_shared_a, dim=-1)
            text_shared_a_n = F.normalize(text_shared_a, dim=-1)
            vision_shared_b_n = F.normalize(vision_shared_b, dim=-1)
            text_shared_b_n = F.normalize(text_shared_b, dim=-1)
            cross_modal_loss = 0.5 * (
                (2.0 - 2.0 * (vision_shared_a_n * text_shared_a_n).sum(-1)).mean()
                + (2.0 - 2.0 * (vision_shared_b_n * text_shared_b_n).sum(-1)).mean()
            )

            shared_sigreg = 0.25 * (
                sigreg(vision_shared_a)
                + sigreg(vision_shared_b)
                + sigreg(text_shared_a)
                + sigreg(text_shared_b)
            )
            private_sigreg = 0.25 * (
                sigreg(vision_private_a)
                + sigreg(vision_private_b)
                + sigreg(text_private_a)
                + sigreg(text_private_b)
            )
            sigreg_loss = 0.5 * (shared_sigreg + private_sigreg)

        loss = (1 - sigreg_weight) * (
            args.invariance_weight * invariance_loss + args.cross_modal_weight * cross_modal_loss
        ) + sigreg_weight * sigreg_loss
        loss_terms = {
            "inv": invariance_loss.item(),
            "cross": cross_modal_loss.item(),
            "sig": sigreg_loss.item(),
        }

        if config["reconstruct"]:
            reconstruction_loss = F.mse_loss(
                model.vision_predictor(vision_a_repr),
                model.normalized_target(vision_b, None),
            ) + F.mse_loss(
                model.text_predictor(text_a_repr),
                model.normalized_target(text_b, mask),
            )
            loss = loss + args.reconstruction_weight * reconstruction_loss
            loss_terms["rec"] = reconstruction_loss.item()

        if config["independence"]:
            shared_vision, private_vision = model.split_representation(vision_a_repr)
            shared_text, private_text = model.split_representation(text_a_repr)
            independence_loss = (
                cross_correlation_loss(private_vision, shared_vision)
                + cross_correlation_loss(private_text, shared_text)
                + cross_correlation_loss(private_vision, shared_text)
                + cross_correlation_loss(private_text, shared_vision)
            )
            loss = loss + args.independence_weight * independence_loss
            loss_terms["ind"] = independence_loss.item()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step == 1 or step % max(1, args.pretrain_steps // 4) == 0:
            term_text = " ".join(f"{key}={value:.4f}" for key, value in loss_terms.items())
            print(
                f"    [{name}] step {step:5d}/{args.pretrain_steps} "
                f"loss={loss.item():.4f} {term_text}",
                flush=True,
            )

    model.eval()
    del vision_views, text_views


@torch.no_grad()
def encode_dataset(
    model: MultimodalModel, bundle: Bundle, device: str, batch_size: int = 1024
) -> dict[str, torch.Tensor]:
    """Encode every sample once and return the concatenated feature blocks."""
    features: dict[str, list[torch.Tensor]] = defaultdict(list)
    n_samples = bundle.vision_views.size(0)
    for start in range(0, n_samples, batch_size):
        batch_mask = (
            bundle.text_mask[start : start + batch_size].to(device)
            if bundle.text_mask is not None
            else None
        )
        encoded = model.encode_features(
            bundle.vision_views[start : start + batch_size, 0].to(device),
            bundle.text_views[start : start + batch_size, 0].to(device),
            batch_mask,
        )
        for key, value in encoded.items():
            features[key].append(value.cpu())
    return {key: torch.cat(values) for key, values in features.items()}


def extract_raw_features(bundle: Bundle) -> dict[str, torch.Tensor]:
    """Build the frozen-backbone ``raw`` baseline features."""

    def concatenate_views(views: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        outputs = []
        for start in range(0, views.size(0), 2048):
            view = views[start : start + 2048, 0]
            if view.dim() == 2:
                outputs.append(view.float())
            else:
                batch_mask = None if mask is None else mask[start : start + 2048]
                outputs.append(torch.cat([view[:, 0].float(), pool_tokens(view, batch_mask)], -1))
        return torch.cat(outputs)

    return {
        "h1": concatenate_views(bundle.vision_views, None),
        "h2": concatenate_views(bundle.text_views, bundle.text_mask),
    }
