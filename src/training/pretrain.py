"""Joint pretraining of LeJEPA and D-JEPA."""

from __future__ import annotations

import logging
import time
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from ..augment import patch_drop_vision
from ..config import DataConfig, OptimizationConfig
from ..losses import (
    covariance_decorrelation_loss,
    lejepa_loss,
    sigreg_loss,
)
from ..utils import (
    autocast_context,
    cycle,
    format_duration,
    progress_suffix,
)

LOGGER = logging.getLogger(__name__)


def _forward_text_clean(model: nn.Module, ids: torch.Tensor, mask: torch.Tensor):
    if hasattr(model, "forward_text_clean"):
        return model.forward_text_clean(ids, mask)
    return model.forward_text(ids, mask, ids, mask)[0]


def _forward_vision_single(model: nn.Module, image: torch.Tensor):
    if hasattr(model, "forward_vision_single"):
        return model.forward_vision_single(image)
    (component, unique), _, (hidden, _) = model.forward_vision(image, image)
    return component, unique, hidden


def _forward_text_single(
    model: nn.Module, ids: torch.Tensor, mask: torch.Tensor
):
    if hasattr(model, "forward_text_single"):
        return model.forward_text_single(ids, mask)
    (component, unique), _, (hidden, _) = model.forward_text(
        ids, mask, ids, mask
    )
    return component, unique, hidden


def pretrain_models(
    lejepa: nn.Module,
    decomposed: nn.Module,
    lejepa_optimizer: torch.optim.Optimizer,
    decomposed_optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    loader: DataLoader,
    data_config: DataConfig,
    optim_config: OptimizationConfig,
    device: torch.device,
    amp: bool,
) -> dict[str, list[float]]:
    """Run the four-term D-JEPA objective and the LeJEPA baseline.

    Returns a scalar history that is used for the training-curve figure and
    for diagnosing representation collapse.
    """

    iterator = cycle(loader)
    history: dict[str, list[float]] = {
        "step": [],
        "loss_lejepa": [],
        "loss_djepa": [],
        "lejepa_cross": [],
        "djepa_shared": [],
        "djepa_cross": [],
        "djepa_decorrelation": [],
        "djepa_vision_recon": [],
        "djepa_text_recon": [],
        "djepa_synergy": [],
    }

    total_steps = optim_config.pretrain_steps
    log_every = max(1, optim_config.log_every)
    started = time.perf_counter()
    LOGGER.info(
        "Pretraining %d steps (batch size %d, log every %d)",
        total_steps,
        data_config.batch_size,
        log_every,
    )

    for step in range(1, total_steps + 1):
        batch = next(iterator)
        view1 = batch["view1"].to(device)
        view2 = batch["view2"].to(device)
        masked_ids1 = batch["masked_ids1"].to(device)
        masked_mask1 = batch["masked_mask1"].to(device)
        question_ids = batch["question_ids"].to(device)
        question_mask = batch["question_mask"].to(device)

        with autocast_context(device, amp):
            # -------------------------------------------------------------
            # 1. LeJEPA baseline: one shared latent for both modalities.
            # -------------------------------------------------------------
            lejepa_vision1, lejepa_vision2 = lejepa.forward_vision(
                view1, view2
            )
            lejepa_text = _forward_text_clean(
                lejepa, question_ids, question_mask
            )

            loss_pv, _, _ = lejepa_loss(
                lejepa_vision1,
                lejepa_vision2,
                sigreg_weight=optim_config.sigreg_weight,
                num_slices=optim_config.sigreg_slices,
            )
            loss_pt = optim_config.sigreg_weight * sigreg_loss(
                lejepa_text, num_slices=optim_config.sigreg_slices
            )
            loss_pcross = 0.5 * (
                F.mse_loss(lejepa_vision1, lejepa_text)
                + F.mse_loss(lejepa_vision2, lejepa_text)
            )
            loss_lejepa = loss_pv + loss_pt + loss_pcross

            # -------------------------------------------------------------
            # 2. D-JEPA shared components.
            # -------------------------------------------------------------
            (vision_c1, vision_u1), (vision_c2, vision_u2), (
                vision_hidden1,
                _,
            ) = decomposed.forward_vision(view1, view2)
            text_c, text_u, text_hidden = _forward_text_single(
                decomposed, question_ids, question_mask
            )

            loss_dvc, _, _ = lejepa_loss(
                vision_c1,
                vision_c2,
                sigreg_weight=optim_config.sigreg_weight,
                num_slices=optim_config.sigreg_slices,
            )
            loss_dtc = optim_config.sigreg_weight * sigreg_loss(
                text_c, num_slices=optim_config.sigreg_slices
            )
            loss_dcross = 0.5 * (
                F.mse_loss(vision_c1, text_c)
                + F.mse_loss(vision_c2, text_c)
            )
            loss_dc = loss_dvc + loss_dtc + loss_dcross

            # -------------------------------------------------------------
            # 3. Private fibres are decorrelated from the shared code.
            # -------------------------------------------------------------
            loss_dvu = covariance_decorrelation_loss(vision_u1, vision_c1)
            loss_dtu = covariance_decorrelation_loss(text_u, text_c)

            # -------------------------------------------------------------
            # 4. Masked JEPA grounding for each private fibre.
            # -------------------------------------------------------------
            masked_view1 = patch_drop_vision(
                view1,
                patch_size=data_config.patch_size,
                drop_ratio=data_config.patch_drop_ratio,
            )
            masked_cv, masked_uv, _ = _forward_vision_single(
                decomposed, masked_view1
            )
            predicted_vision_hidden = decomposed.vision_predictor(
                torch.cat([masked_cv, masked_uv], dim=-1)
            )
            loss_duv_info = F.mse_loss(
                predicted_vision_hidden, vision_hidden1.detach()
            )

            # ``masked_ids1`` already contains the 15% token masking applied
            # by the dataset, matching the reference implementation.
            masked_ct, masked_ut, _ = _forward_text_single(
                decomposed, masked_ids1, masked_mask1
            )
            predicted_text_hidden = decomposed.text_predictor(
                torch.cat([masked_ct, masked_ut], dim=-1)
            )
            loss_dut_info = F.mse_loss(
                predicted_text_hidden, text_hidden.detach()
            )

            # -------------------------------------------------------------
            # 5. Synergy regulariser (the head is evaluated separately).
            # -------------------------------------------------------------
            synergy = decomposed.extract_synergy(vision_c1, text_c)
            loss_synergy = sigreg_loss(
                synergy, num_slices=optim_config.synergy_sigreg_slices
            )

            loss_djepa = (
                loss_dc
                + 0.5 * (loss_dvu + loss_dtu)
                + 1.0 * (loss_duv_info + loss_dut_info)
                + 0.5 * loss_synergy
            )

        lejepa_optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss_lejepa).backward()
        scaler.step(lejepa_optimizer)

        decomposed_optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss_djepa).backward()
        scaler.step(decomposed_optimizer)

        scaler.update()

        if step == 1 or step % log_every == 0 or step == total_steps:
            history["step"].append(float(step))
            history["loss_lejepa"].append(float(loss_lejepa.item()))
            history["loss_djepa"].append(float(loss_djepa.item()))
            history["lejepa_cross"].append(float(loss_pcross.item()))
            history["djepa_shared"].append(float(loss_dc.item()))
            history["djepa_cross"].append(float(loss_dcross.item()))
            history["djepa_decorrelation"].append(
                float(0.5 * (loss_dvu + loss_dtu).item())
            )
            history["djepa_vision_recon"].append(float(loss_duv_info.item()))
            history["djepa_text_recon"].append(float(loss_dut_info.item()))
            history["djepa_synergy"].append(float(loss_synergy.item()))

            LOGGER.info(
                "pretrain %s | lejepa %.4f | djepa %.4f | "
                "cross %.4f | priv-vis %.4f | decorr %.4f",
                progress_suffix(step, total_steps, started),
                loss_lejepa.item(),
                loss_djepa.item(),
                loss_dcross.item(),
                loss_duv_info.item(),
                loss_dvu.item(),
            )

    LOGGER.info(
        "Pretraining finished in %s (%d steps)",
        format_duration(time.perf_counter() - started),
        total_steps,
    )
    return history
