"""Flow-PID information decomposition over D-JEPA fibre codes."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..results import PIDRow
from ..utils import autocast_context
from .kernels import evaluate_kernel_ridge_r2

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PIDInputs:
    x1: torch.Tensor
    x2: torch.Tensor
    joint: torch.Tensor
    representations: Mapping[str, torch.Tensor]


def collect_pid_inputs(
    decomposed: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    n_samples: int = 800,
    batch_size: int = 64,
) -> PIDInputs:
    """Collect up to ``n_samples`` D-JEPA representations."""

    all_x1: list[torch.Tensor] = []
    all_x2: list[torch.Tensor] = []
    all_u1: list[torch.Tensor] = []
    all_u2: list[torch.Tensor] = []
    all_c: list[torch.Tensor] = []
    all_s: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in loader:
            eval_view = batch["eval_view"].to(device)
            question_ids = batch["question_ids"].to(device)
            question_mask = batch["question_mask"].to(device)

            with autocast_context(device, amp):
                decomposed_out = decomposed.extract_features(
                    eval_view, question_ids, question_mask
                )

            all_x1.append(decomposed_out["c1"].cpu())
            all_x2.append(decomposed_out["c2"].cpu())
            all_u1.append(decomposed_out["u1"].cpu())
            all_u2.append(decomposed_out["u2"].cpu())
            all_c.append(decomposed_out["c"].cpu())
            all_s.append(decomposed_out["s"].cpu())

            if len(all_x1) * batch_size >= n_samples:
                break

    x1 = torch.cat(all_x1)[:n_samples]
    x2 = torch.cat(all_x2)[:n_samples]
    joint = torch.cat([x1, x2], dim=-1)

    representations = {
        "D-JEPA u1": torch.cat(all_u1)[:n_samples],
        "D-JEPA u2": torch.cat(all_u2)[:n_samples],
        "D-JEPA c": torch.cat(all_c)[:n_samples],
        "D-JEPA s": torch.cat(all_s)[:n_samples],
    }
    return PIDInputs(x1=x1, x2=x2, joint=joint, representations=representations)


def pid_inputs_from_representations(
    representations: Mapping[str, torch.Tensor],
    n_samples: int = 800,
) -> PIDInputs:
    """Build PID inputs from already-collected representation tensors."""

    def take(name: str) -> torch.Tensor:
        return representations[name][:n_samples]

    x1 = take("c1")
    x2 = take("c2")
    joint = torch.cat([x1, x2], dim=-1)
    return PIDInputs(
        x1=x1,
        x2=x2,
        joint=joint,
        representations={
            "D-JEPA u1": take("u1"),
            "D-JEPA u2": take("u2"),
            "D-JEPA c": take("c"),
            "D-JEPA s": take("s"),
        },
    )


def compute_pid_rows(
    inputs: PIDInputs,
    n_samples: int = 800,
    train_fraction: float = 0.7,
    seed: int = 42,
) -> list[PIDRow]:
    """Compute unique, redundant, and synergistic R^2 terms."""

    n_pid = min(n_samples, inputs.x1.size(0))
    permutation = torch.randperm(
        n_pid, generator=torch.Generator().manual_seed(seed)
    )
    split = int(train_fraction * n_pid)
    train_indices = permutation[:split]
    test_indices = permutation[split:]

    sources = {
        "X1": inputs.x1,
        "X2": inputs.x2,
        "Joint": inputs.joint,
    }

    rows: list[PIDRow] = []
    total_codes = len(inputs.representations)
    started = time.perf_counter()
    LOGGER.info(
        "Computing Flow-PID for %d codes on %d samples",
        total_codes,
        n_pid,
    )
    for code_index, (code_name, target) in enumerate(
        inputs.representations.items(), 1
    ):
        code_started = time.perf_counter()
        r2_x1 = evaluate_kernel_ridge_r2(
            sources["X1"][train_indices],
            target[train_indices],
            sources["X1"][test_indices],
            target[test_indices],
            kernel_type="poly2",
        )
        r2_x2 = evaluate_kernel_ridge_r2(
            sources["X2"][train_indices],
            target[train_indices],
            sources["X2"][test_indices],
            target[test_indices],
            kernel_type="poly2",
        )
        r2_joint = evaluate_kernel_ridge_r2(
            sources["Joint"][train_indices],
            target[train_indices],
            sources["Joint"][test_indices],
            target[test_indices],
            kernel_type="poly2",
        )

        r2_x1_clamped = max(r2_x1, 0.0)
        r2_x2_clamped = max(r2_x2, 0.0)
        r2_joint_clamped = max(r2_joint, 0.0)

        redundancy = min(r2_x1_clamped, r2_x2_clamped)
        unique_x1 = max(r2_x1_clamped - redundancy, 0.0)
        unique_x2 = max(r2_x2_clamped - redundancy, 0.0)
        synergy = max(
            r2_joint_clamped - max(r2_x1_clamped, r2_x2_clamped), 0.0
        )

        terms = {
            "Unq1": unique_x1,
            "Unq2": unique_x2,
            "Red": redundancy,
            "Syn": synergy,
        }
        rows.append(
            PIDRow(
                code=code_name,
                r2_x1=r2_x1,
                r2_x2=r2_x2,
                r2_joint=r2_joint,
                unique_x1=unique_x1,
                unique_x2=unique_x2,
                redundancy=redundancy,
                synergy=synergy,
                dominant=max(terms, key=terms.get),
            )
        )
        LOGGER.info(
            "PID %d/%d %-10s | R2 x1=%+.3f x2=%+.3f joint=%+.3f | %.1fs",
            code_index,
            total_codes,
            code_name,
            r2_x1,
            r2_x2,
            r2_joint,
            time.perf_counter() - code_started,
        )

    LOGGER.info(
        "Flow-PID finished in %.1fs",
        time.perf_counter() - started,
    )
    return rows
