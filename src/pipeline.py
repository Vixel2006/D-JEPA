"""End-to-end CLEVR D-JEPA pipeline."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .config import RunConfig
from .data import build_clevr_dataset, load_tokenizer, split_indices
from .diagnostics import (
    Diagnostics,
    compute_alignment_pairs,
    compute_calibration,
    compute_cka_pairs,
    compute_spectral_diagnostics,
    confusion_matrix,
)
from .evaluation import (
    CATEGORIES,
    DJEPA_C,
    DJEPA_C_U1,
    DJEPA_C_U1_U2,
    DJEPA_FULL,
    LEJEPA,
    build_subspace_specs,
    build_test_records,
    case_studies_from_records,
    causal_fiber_swap_analysis,
    collect_evaluation_features,
    compute_logits,
    compute_pid_rows,
    create_probe_optimizers,
    create_probes,
    evaluate_probes,
    pid_inputs_from_representations,
    procrustes_alignment,
    random_subspace_name,
    train_probes,
)
from .figures import FigureData
from .models import build_decomposed_model, build_lejepa_model
from .reporting import aggregate_accuracy, report_results
from .results import PipelineResults, SeedMetrics
from .training import pretrain_models
from .utils import (
    amp_enabled,
    format_duration,
    make_grad_scaler,
    resolve_device,
    seed_everything,
    trainable_parameters,
)

LOGGER = logging.getLogger(__name__)

SPECTRAL_NAMES = ("z1", "z2", "c1", "c2", "c", "u1", "u2", "s")
CKA_NAMES = ("z1", "z2", "c1", "c2", "c", "u1", "u2", "s")
ALIGNMENT_PAIRS = (
    ("z1", "z2"),
    ("c1", "c2"),
    ("u1", "u2"),
    ("c1", "u1"),
    ("c2", "u2"),
    ("c", "c1"),
    ("c", "c2"),
)
DIAGNOSTIC_CONDITIONS = (
    LEJEPA,
    DJEPA_C,
    DJEPA_C_U1_U2,
    DJEPA_FULL,
)


def _make_dataloaders(config: RunConfig, dataset):
    train_indices, test_indices = split_indices(dataset, config.data.test_size)
    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=config.data.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.data.num_workers,
    )
    test_loader = DataLoader(
        Subset(dataset, test_indices),
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
    )
    return train_loader, test_loader, test_indices


def _make_lejepa_optimizer(
    model: torch.nn.Module, config: RunConfig
) -> torch.optim.Optimizer:
    parameters = trainable_parameters(
        list(model.vision_projector.parameters())
        + list(model.text_encoder.parameters())
        + list(model.text_projector.parameters())
    )
    return torch.optim.AdamW(
        parameters,
        lr=config.optim.pretrain_lr,
        weight_decay=config.optim.pretrain_weight_decay,
    )


def _make_decomposed_optimizer(
    model: torch.nn.Module, config: RunConfig
) -> torch.optim.Optimizer:
    parameters = trainable_parameters(
        list(model.vision_c.parameters())
        + list(model.vision_u.parameters())
        + list(model.vision_predictor.parameters())
        + list(model.text_encoder.parameters())
        + list(model.text_c.parameters())
        + list(model.text_u.parameters())
        + list(model.text_predictor.parameters())
        + list(model.fusion.parameters())
    )
    return torch.optim.AdamW(
        parameters,
        lr=config.optim.pretrain_lr,
        weight_decay=config.optim.pretrain_weight_decay,
    )


def _collect_sample_images(
    dataset,
    records: list[dict[str, Any]],
    limit: int = 30,
) -> dict[int, np.ndarray]:
    images: dict[int, np.ndarray] = {}
    items = getattr(dataset, "items", [])
    scene_images = getattr(dataset, "images", {})
    for record in records:
        if len(images) >= limit:
            break
        row = record.get("row")
        scene_id = record.get("idx")
        if row is None or scene_id is None or scene_id in images:
            continue
        if row >= len(items):
            continue
        item = items[row]
        image = item.image
        if image is None:
            image = scene_images.get(item.scene_id)
        if image is None:
            continue
        images[int(scene_id)] = np.asarray(image.convert("RGB")) / 255.0
    return images


def _build_diagnostics(
    features,
    swap_analysis: dict[str, dict[str, float]],
    num_classes: int,
    component_dim: int,
) -> Diagnostics:
    representations = features.representations
    spectral = compute_spectral_diagnostics(representations, SPECTRAL_NAMES)

    cka_pairs = [
        (a, b) for index, a in enumerate(CKA_NAMES) for b in CKA_NAMES[index + 1 :]
    ]
    cka = compute_cka_pairs(representations, cka_pairs)
    alignment = compute_alignment_pairs(representations, ALIGNMENT_PAIRS)

    random_condition = random_subspace_name(component_dim)
    calibration = compute_calibration(
        features.logits,
        features.labels,
        num_classes=num_classes,
        conditions=(
            *DIAGNOSTIC_CONDITIONS,
            random_condition,
        ),
    )

    confusion: dict[str, list[list[int]]] = {}
    predictions = features.predictions
    for condition in (LEJEPA, DJEPA_C, DJEPA_FULL):
        if condition in predictions:
            confusion[condition] = confusion_matrix(
                predictions[condition],
                features.labels,
                num_classes=num_classes,
            )

    return Diagnostics(
        spectral=spectral,
        cka=cka,
        alignment=alignment,
        calibration=calibration,
        swaps=swap_analysis,
        confusion=confusion,
    )


def run(config: RunConfig) -> PipelineResults:
    """Run pretraining, multi-seed probing, PID analysis, and diagnostics."""

    if not config.seeds:
        raise ValueError("At least one evaluation seed is required.")

    device = resolve_device(config.device)
    amp = amp_enabled(device, config.optim.amp)
    LOGGER.info("Executing multi-seed CLEVR benchmark on: %s", device)
    LOGGER.info(
        "Plan: %d pretrain steps, %d probe steps x %d seeds, amp=%s",
        config.optim.pretrain_steps,
        config.optim.probe_steps,
        len(config.seeds),
        amp,
    )

    seed_everything(config.data.seed)
    tokenizer = load_tokenizer(config.model.text_backbone)
    data = build_clevr_dataset(config.data, tokenizer)
    train_loader, test_loader, test_indices = _make_dataloaders(config, data.dataset)
    LOGGER.info(
        "Dataset ready: %d train / %d test items | %d answer classes",
        len(train_loader.dataset),
        len(test_loader.dataset),
        len(data.answers),
    )

    LOGGER.info(
        "Building LeJEPA and D-JEPA (vision=%s, text=%s)",
        config.model.vision_backbone,
        config.model.text_backbone,
    )
    lejepa = build_lejepa_model(config.model).to(device)
    decomposed = build_decomposed_model(config.model).to(device)
    LOGGER.info(
        "Trainable parameters: LeJEPA %s | D-JEPA %s",
        f"{sum(p.numel() for p in trainable_parameters(lejepa.parameters())):,}",
        f"{sum(p.numel() for p in trainable_parameters(decomposed.parameters())):,}",
    )

    lejepa_optimizer = _make_lejepa_optimizer(lejepa, config)
    decomposed_optimizer = _make_decomposed_optimizer(decomposed, config)
    scaler = make_grad_scaler(amp)

    lejepa.train()
    decomposed.train()

    loss_history = pretrain_models(
        lejepa=lejepa,
        decomposed=decomposed,
        lejepa_optimizer=lejepa_optimizer,
        decomposed_optimizer=decomposed_optimizer,
        scaler=scaler,
        loader=train_loader,
        data_config=config.data,
        optim_config=config.optim,
        device=device,
        amp=amp,
    )

    lejepa.eval()
    decomposed.eval()

    specs = build_subspace_specs(config.model)
    subspace_names = [spec.name for spec in specs]
    categories = list(CATEGORIES)
    results = PipelineResults(
        seeds=config.seeds,
        subspace_names=subspace_names,
        categories=categories,
        loss_history=loss_history,
    )

    probes: dict[str, torch.nn.Module] = {}
    swap_analysis: dict[str, dict[str, float]] = {}
    probing_started = time.perf_counter()

    for seed_index, run_seed in enumerate(config.seeds, 1):
        seed_started = time.perf_counter()
        seed_everything(run_seed)
        LOGGER.info(
            "Seed %d/%d (%d): training %d probes for %d steps",
            seed_index,
            len(config.seeds),
            run_seed,
            len(specs),
            config.optim.probe_steps,
        )
        probes = create_probes(
            specs,
            num_classes=len(data.answers),
            bottleneck=config.model.probe_bottleneck_dim,
            device=device,
        )
        probe_optimizers = create_probe_optimizers(probes, config.optim)

        train_probes(
            probes=probes,
            optimizers=probe_optimizers,
            lejepa=lejepa,
            decomposed=decomposed,
            loader=train_loader,
            specs=specs,
            config=config.optim,
            component_dim=config.model.component_dim,
            device=device,
            amp=amp,
        )

        for probe in probes.values():
            probe.eval()

        accuracy = evaluate_probes(
            probes=probes,
            lejepa=lejepa,
            decomposed=decomposed,
            loader=test_loader,
            specs=specs,
            device=device,
            amp=amp,
            component_dim=config.model.component_dim,
        )
        swap_analysis = causal_fiber_swap_analysis(
            probes=probes,
            decomposed=decomposed,
            loader=test_loader,
            device=device,
            amp=amp,
            component_dim=config.model.component_dim,
        )
        flip_rate = swap_analysis["u1"]["counting"]
        procrustes_lejepa, procrustes_djepa = procrustes_alignment(
            lejepa=lejepa,
            decomposed=decomposed,
            loader=test_loader,
            device=device,
            amp=amp,
            max_samples=800,
            batch_size=config.data.batch_size,
        )

        results.seed_metrics.append(
            SeedMetrics(
                accuracy=accuracy,
                flip_rate=flip_rate,
                procrustes_lejepa=procrustes_lejepa,
                procrustes_djepa=procrustes_djepa,
            )
        )
        LOGGER.info(
            "seed %d | overall LeJEPA %.2f | D-JEPA Full %.2f | "
            "u1 flip %.2f%% | Procrustes c %.4f",
            run_seed,
            accuracy[LEJEPA]["overall"],
            accuracy[DJEPA_FULL]["overall"],
            flip_rate,
            procrustes_djepa,
        )
        LOGGER.info(
            "Seed %d complete in %s",
            run_seed,
            format_duration(time.perf_counter() - seed_started),
        )

    LOGGER.info(
        "All %d seeds finished in %s",
        len(config.seeds),
        format_duration(time.perf_counter() - probing_started),
    )

    # ------------------------------------------------------------------
    # Single held-out pass shared by diagnostics, figures, and case studies.
    # ------------------------------------------------------------------
    LOGGER.info("Collecting held-out features for diagnostics and figures...")
    features = collect_evaluation_features(
        lejepa=lejepa,
        decomposed=decomposed,
        loader=test_loader,
        device=device,
        amp=amp,
        component_dim=config.model.component_dim,
    )
    compute_logits(probes, features)
    LOGGER.info("Scoring probes on held-out features...")
    records = build_test_records(features, answer_vocabulary=data.answers)

    LOGGER.info("Computing Flow-PID decomposition...")
    pid_inputs = pid_inputs_from_representations(
        features.representations, n_samples=800
    )
    results.pid_rows = compute_pid_rows(
        pid_inputs, n_samples=800, train_fraction=0.7, seed=42
    )

    LOGGER.info("Computing diagnostics (spectral, CKA, calibration)...")
    results.diagnostics = (
        _build_diagnostics(
            features,
            swap_analysis=swap_analysis,
            num_classes=len(data.answers),
            component_dim=config.model.component_dim,
        )
        if config.collect_diagnostics
        else None
    )
    results.records = records

    c1 = features.representations["c1"]
    c2 = features.representations["c2"]
    u1 = features.representations["u1"]
    u2 = features.representations["u2"]
    s = features.representations["s"]
    results.embeddings = {
        "z1": features.representations["z1"].numpy(),
        "z2": features.representations["z2"].numpy(),
        "c1": c1.numpy(),
        "c2": c2.numpy(),
        "shared_c": features.representations["c"].numpy(),
        "u1": u1.numpy(),
        "u2": u2.numpy(),
        "s": s.numpy(),
        "c": torch.cat([c1, c2], dim=-1).numpy(),
        "c+u1+u2": torch.cat([c1, c2, u1, u2], dim=-1).numpy(),
        "full": torch.cat([c1, c2, u1, u2, s], dim=-1).numpy(),
    }

    if config.run_case_studies:
        LOGGER.info("Extracting qualitative win/loss case studies...")
        results.case_studies = case_studies_from_records(records)
    results.sample_images = _collect_sample_images(
        data.dataset, records, limit=config.figures.max_qualitative * 6
    )

    figure_data = FigureData(
        accuracy=aggregate_accuracy(results),
        categories=results.categories,
        subspace_names=results.subspace_names,
        embeddings=results.embeddings,
        representations=features.representations,
        records=results.records,
        sample_images=results.sample_images,
        diagnostics=results.diagnostics,
        loss_history=results.loss_history,
        seeds=results.seeds,
        answer_vocabulary=data.answers,
    )

    LOGGER.info(
        "Writing results, diagnostics, and figures to %s", config.output_dir
    )
    report_results(
        results,
        output_dir=Path(config.output_dir),
        save=config.save_results,
        images=data.images,
        figure_data=figure_data,
        figure_config=config.figures,
    )
    LOGGER.info("Benchmark complete.")
    return results
