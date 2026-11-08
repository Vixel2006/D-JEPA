"""Capacity-matched probes and downstream CLEVR evaluation."""

from __future__ import annotations

import logging
import time
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from ..config import ModelConfig, OptimizationConfig
from ..utils import (
    autocast_context,
    cycle,
    format_duration,
    progress_suffix,
)

LOGGER = logging.getLogger(__name__)

CATEGORIES = ("counting", "attribute", "spatial_relation")
OVERALL = "overall"

LEJEPA = "LeJEPA [z1*, z2*]"
DJEPA_C = "D-JEPA [c1, c2]"
DJEPA_C_U1 = "D-JEPA [c1, c2, u1]"
DJEPA_C_U2 = "D-JEPA [c1, c2, u2]"
DJEPA_C_U1_U2 = "D-JEPA [c1, c2, u1, u2]"
DJEPA_FULL = "D-JEPA Full [c1, c2, u1, u2, s]"
DJEPA_PROJ_1024 = "D-JEPA Proj Full->1024"
DJEPA_PROJ_512 = "D-JEPA Proj Full->512"

FIBER_TO_JEPA_KEY = {"u1": "u1", "u2": "u2", "s": "s"}


def random_subspace_name(component_dim: int) -> str:
    return f"D-JEPA [c1, c2, RANDOM-{component_dim}]"


@dataclass(frozen=True)
class SubspaceSpec:
    name: str
    dim: int
    projection_dim: Optional[int] = None


def build_subspace_specs(config: ModelConfig) -> tuple[SubspaceSpec, ...]:
    """Return probe dimensions in the reference table order."""

    comp = config.component_dim
    specs = (
        SubspaceSpec(LEJEPA, 2 * config.projection_dim),
        SubspaceSpec(DJEPA_C, 2 * comp),
        SubspaceSpec(DJEPA_C_U1, 3 * comp),
        SubspaceSpec(DJEPA_C_U2, 3 * comp),
        SubspaceSpec(DJEPA_C_U1_U2, 4 * comp),
        SubspaceSpec(DJEPA_FULL, 4 * comp + config.synergy_dim),
        SubspaceSpec(
            DJEPA_PROJ_1024,
            4 * comp + config.synergy_dim,
            projection_dim=1024,
        ),
        SubspaceSpec(
            DJEPA_PROJ_512,
            4 * comp + config.synergy_dim,
            projection_dim=512,
        ),
        SubspaceSpec(random_subspace_name(comp), 3 * comp),
    )
    return specs


# ``SUBSPACE_SPECS`` is convenient for imports and tests; the pipeline uses
# ``build_subspace_specs`` so dimensions follow the model config.
SUBSPACE_SPECS = build_subspace_specs(ModelConfig())


class CapacityMatchedTaskHead(nn.Module):
    """Shared probe head: input -> 512 -> bottleneck -> answer classes."""

    def __init__(
        self,
        in_dim: int,
        bottleneck: int = 128,
        num_classes: int = 40,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(512, bottleneck),
            nn.LayerNorm(bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class ProjectedProbe(nn.Module):
    """Probe with an explicit linear projection before the shared head.

    This is the capacity control used in the reference script: it asks whether
    a wider probe can close the gap without changing the information available
    in the representation.
    """

    def __init__(
        self,
        in_dim: int,
        proj_dim: int,
        bottleneck: int = 128,
        num_classes: int = 40,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, proj_dim)
        self.probe = CapacityMatchedTaskHead(
            in_dim=proj_dim,
            bottleneck=bottleneck,
            num_classes=num_classes,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.probe(self.proj(features))


def build_probe_features(
    lejepa_out: dict[str, torch.Tensor],
    decomposed_out: dict[str, torch.Tensor],
    random_vector: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Assemble every probe input exactly once per batch."""

    c1 = decomposed_out["c1"]
    c2 = decomposed_out["c2"]
    u1 = decomposed_out["u1"]
    u2 = decomposed_out["u2"]
    s = decomposed_out["s"]
    full = torch.cat([c1, c2, u1, u2, s], dim=-1).float()

    return {
        LEJEPA: lejepa_out["lejepa"].float(),
        DJEPA_C: torch.cat([c1, c2], dim=-1).float(),
        DJEPA_C_U1: torch.cat([c1, c2, u1], dim=-1).float(),
        DJEPA_C_U2: torch.cat([c1, c2, u2], dim=-1).float(),
        DJEPA_C_U1_U2: torch.cat([c1, c2, u1, u2], dim=-1).float(),
        DJEPA_FULL: full,
        DJEPA_PROJ_1024: full,
        DJEPA_PROJ_512: full,
        random_subspace_name(random_vector.size(1)): torch.cat(
            [c1, c2, random_vector], dim=-1
        ).float(),
    }


def create_probes(
    specs: Iterable[SubspaceSpec],
    num_classes: int,
    bottleneck: int,
    device: torch.device,
) -> dict[str, nn.Module]:
    probes: dict[str, nn.Module] = {}
    for spec in specs:
        if spec.projection_dim is None:
            probe = CapacityMatchedTaskHead(
                in_dim=spec.dim,
                bottleneck=bottleneck,
                num_classes=num_classes,
            )
        else:
            probe = ProjectedProbe(
                in_dim=spec.dim,
                proj_dim=spec.projection_dim,
                bottleneck=bottleneck,
                num_classes=num_classes,
            )
        probes[spec.name] = probe.to(device)
    return probes


def create_probe_optimizers(
    probes: dict[str, nn.Module],
    config: OptimizationConfig,
) -> dict[str, torch.optim.Optimizer]:
    return {
        name: torch.optim.AdamW(
            probe.parameters(),
            lr=config.probe_lr,
            weight_decay=config.probe_weight_decay,
        )
        for name, probe in probes.items()
    }


def train_probes(
    probes: dict[str, nn.Module],
    optimizers: dict[str, torch.optim.Optimizer],
    lejepa: nn.Module,
    decomposed: nn.Module,
    loader: DataLoader,
    specs: tuple[SubspaceSpec, ...],
    config: OptimizationConfig,
    component_dim: int,
    device: torch.device,
    amp: bool,
) -> None:
    """Train one probe per subspace with a shared feature pass."""

    iterator = cycle(loader)
    total_steps = config.probe_steps
    log_every = max(1, config.log_every)
    started = time.perf_counter()
    LOGGER.info(
        "Training %d probes for %d steps (log every %d)",
        len(probes),
        total_steps,
        log_every,
    )

    for step in range(1, total_steps + 1):
        batch = next(iterator)
        eval_view = batch["eval_view"].to(device)
        question_ids = batch["question_ids"].to(device)
        question_mask = batch["question_mask"].to(device)
        labels = batch["label"].to(device)

        with torch.no_grad():
            with autocast_context(device, amp):
                lejepa_out = lejepa.extract_features(
                    eval_view, question_ids, question_mask
                )
                decomposed_out = decomposed.extract_features(
                    eval_view, question_ids, question_mask
                )

        random_vector = torch.randn(
            eval_view.size(0), component_dim, device=device
        )
        features = build_probe_features(
            lejepa_out, decomposed_out, random_vector
        )

        log_now = (
            step == 1 or step % log_every == 0 or step == total_steps
        )
        step_losses: dict[str, float] = {}
        for spec in specs:
            probe = probes[spec.name]
            probe.train()
            logits = probe(features[spec.name])
            loss = F.cross_entropy(
                logits, labels, label_smoothing=0.05
            )
            optimizers[spec.name].zero_grad()
            loss.backward()
            optimizers[spec.name].step()
            if log_now:
                step_losses[spec.name] = float(loss.item())

        if log_now:
            mean_loss = sum(step_losses.values()) / max(1, len(step_losses))
            full_loss = step_losses.get(DJEPA_FULL, float("nan"))
            LOGGER.info(
                "probe train %s | mean loss %.4f | full loss %.4f",
                progress_suffix(step, total_steps, started),
                mean_loss,
                full_loss,
            )

    LOGGER.info(
        "Probe training finished in %s",
        format_duration(time.perf_counter() - started),
    )


def evaluate_probes(
    probes: dict[str, nn.Module],
    lejepa: nn.Module,
    decomposed: nn.Module,
    loader: DataLoader,
    specs: tuple[SubspaceSpec, ...],
    device: torch.device,
    amp: bool,
    component_dim: int,
) -> dict[str, dict[str, float]]:
    """Return per-subspace accuracy for every category and overall."""

    for probe in probes.values():
        probe.eval()

    counts = {spec.name: Counter() for spec in specs}
    totals: Counter = Counter()
    started = time.perf_counter()
    total_batches = max(1, len(loader))
    log_every = max(1, total_batches // 5)
    LOGGER.info(
        "Evaluating %d probes on %d held-out batches",
        len(specs),
        total_batches,
    )

    with torch.no_grad():
        for batch_index, batch in enumerate(loader, 1):
            eval_view = batch["eval_view"].to(device)
            question_ids = batch["question_ids"].to(device)
            question_mask = batch["question_mask"].to(device)
            labels = batch["label"].to(device)

            with autocast_context(device, amp):
                lejepa_out = lejepa.extract_features(
                    eval_view, question_ids, question_mask
                )
                decomposed_out = decomposed.extract_features(
                    eval_view, question_ids, question_mask
                )

            random_vector = torch.randn(
                eval_view.size(0), component_dim, device=device
            )
            features = build_probe_features(
                lejepa_out, decomposed_out, random_vector
            )

            categories = batch["category"]
            for category in categories:
                totals[category] += 1
            totals[OVERALL] += labels.size(0)

            for spec in specs:
                predictions = probes[spec.name](features[spec.name]).argmax(
                    dim=-1
                )
                matches = (predictions == labels).cpu()
                for index, category in enumerate(categories):
                    if matches[index]:
                        counts[spec.name][category] += 1
                counts[spec.name][OVERALL] += int(matches.sum())

            if batch_index == total_batches or batch_index % log_every == 0:
                LOGGER.info(
                    "probe eval %s",
                    progress_suffix(batch_index, total_batches, started),
                )

    LOGGER.info(
        "Probe evaluation finished in %s",
        format_duration(time.perf_counter() - started),
    )
    return {
        spec.name: {
            category: (
                counts[spec.name][category]
                / max(1, totals[category])
                * 100.0
            )
            for category in (*CATEGORIES, OVERALL)
        }
        for spec in specs
    }


def _swap_flip_rate(
    probe: nn.Module,
    features: torch.Tensor,
    fiber_slice: slice,
    categories: list[str],
) -> dict[str, float]:
    """Replace one fibre across a batch and measure prediction flips."""

    batch_size = features.size(0)
    if batch_size < 2:
        return {category: 0.0 for category in (*CATEGORIES, OVERALL)}

    with torch.no_grad():
        original = probe(features).argmax(dim=-1)
        swapped_features = features.clone()
        permutation = torch.randperm(batch_size, device=features.device)
        swapped_features[:, fiber_slice] = features[permutation, fiber_slice]
        swapped = probe(swapped_features).argmax(dim=-1)

    counts: Counter = Counter()
    totals: Counter = Counter()
    for index, category in enumerate(categories):
        totals[category] += 1
        totals[OVERALL] += 1
        if original[index] != swapped[index]:
            counts[category] += 1
            counts[OVERALL] += 1

    return {
        category: counts[category] / max(1, totals[category]) * 100.0
        for category in (*CATEGORIES, OVERALL)
    }


def causal_fiber_swap_analysis(
    probes: dict[str, nn.Module],
    decomposed: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    component_dim: int,
) -> dict[str, dict[str, float]]:
    """Measure causal prediction flips for u1, u2, and s.

    Each fibre is swapped only inside the probe that isolates it:
    ``[c1,c2,u1]`` for u1, ``[c1,c2,u2]`` for u2, and the full probe for s.
    This keeps the intervention well defined.
    """

    component = component_dim
    slices = {
        "u1": (DJEPA_C_U1, slice(2 * component, 3 * component)),
        "u2": (DJEPA_C_U2, slice(2 * component, 3 * component)),
        "s": (DJEPA_FULL, slice(4 * component, 5 * component)),
    }

    accumulators = {
        fibre: {category: [0.0, 0] for category in (*CATEGORIES, OVERALL)}
        for fibre in slices
    }

    for probe in probes.values():
        probe.eval()

    started = time.perf_counter()
    total_batches = max(1, len(loader))
    log_every = max(1, total_batches // 5)
    LOGGER.info(
        "Running causal fibre swaps (u1, u2, s) on %d held-out batches",
        total_batches,
    )

    with torch.no_grad():
        for batch_index, batch in enumerate(loader, 1):
            eval_view = batch["eval_view"].to(device)
            question_ids = batch["question_ids"].to(device)
            question_mask = batch["question_mask"].to(device)
            categories = batch["category"]

            with autocast_context(device, amp):
                decomposed_out = decomposed.extract_features(
                    eval_view, question_ids, question_mask
                )

            random_vector = torch.randn(
                eval_view.size(0), component_dim, device=device
            )
            # ``build_probe_features`` needs a lejepa dict; the values are
            # unused for the D-JEPA probes, so pass an empty tensor.
            features = build_probe_features(
                {"lejepa": torch.empty(0, device=device)},
                decomposed_out,
                random_vector,
            )

            for fibre, (probe_name, fiber_slice) in slices.items():
                if probe_name not in probes:
                    continue
                flips = _swap_flip_rate(
                    probes[probe_name],
                    features[probe_name],
                    fiber_slice,
                    list(categories),
                )
                for category, value in flips.items():
                    accumulators[fibre][category][0] += value
                    accumulators[fibre][category][1] += 1

            if batch_index == total_batches or batch_index % log_every == 0:
                LOGGER.info(
                    "causal swap %s",
                    progress_suffix(batch_index, total_batches, started),
                )

    LOGGER.info(
        "Causal fibre swaps finished in %s",
        format_duration(time.perf_counter() - started),
    )
    return {
        fibre: {
            category: total / max(1, count)
            for category, (total, count) in values.items()
        }
        for fibre, values in accumulators.items()
    }


def causal_fiber_swap(
    probe: nn.Module,
    decomposed: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    component_dim: int = 256,
) -> float:
    """Backward-compatible counting flip rate for the ``u1`` fibre."""

    analysis = causal_fiber_swap_analysis(
        probes={DJEPA_C_U1: probe},
        decomposed=decomposed,
        loader=loader,
        device=device,
        amp=amp,
        component_dim=component_dim,
    )
    return analysis["u1"]["counting"]
