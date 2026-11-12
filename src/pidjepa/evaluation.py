"""Probes, representation metrics, PID estimators, and the per-seed experiment."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pidjepa.models import (
    ARM_CONFIGS,
    EXPECTED_COMPONENTS,
    FEATURE_SETS,
    PID_STREAM_PAIRS,
    SIGReg,
    arm_family,
    build_model,
)
from pidjepa.training import encode_dataset, extract_raw_features, pretrain_arm
from pidjepa.utils import LN_2, set_seed

if TYPE_CHECKING:
    from pidjepa.data import Bundle


class LinearProbe:
    """A capacity-matched MLP probe with early stopping on a validation split."""

    def __init__(
        self,
        in_dim: int,
        n_classes: int,
        device: str,
        hidden_dim: int = 256,
        seed: int = 0,
    ):
        torch.manual_seed(seed)
        self.device = device
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_classes),
        ).to(device)
        self.mean: torch.Tensor
        self.std: torch.Tensor

    def _normalize(self, inputs: torch.Tensor) -> torch.Tensor:
        return (inputs.to(self.device) - self.mean) / self.std

    def fit(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_val: torch.Tensor,
        y_val: torch.Tensor,
        epochs: int = 60,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-2,
        batch_size: int = 256,
        patience: int = 8,
    ) -> float:
        self.mean = x_train.to(self.device).mean(0)
        self.std = x_train.to(self.device).std(0).clamp_min(1e-6)
        x_train_n = self._normalize(x_train)
        y_train_n = y_train.to(self.device)
        x_val_n = self._normalize(x_val)
        y_val_n = y_val.to(self.device)

        optimizer = torch.optim.AdamW(
            self.network.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        best_loss, best_state, epochs_without_improvement = float("inf"), None, 0
        for _ in range(epochs):
            self.network.train()
            permutation = torch.randperm(len(x_train_n), device=self.device)
            for start in range(0, len(permutation), batch_size):
                batch_indices = permutation[start : start + batch_size]
                loss = F.cross_entropy(
                    self.network(x_train_n[batch_indices]), y_train_n[batch_indices]
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            self.network.eval()
            with torch.no_grad():
                val_loss = F.cross_entropy(self.network(x_val_n), y_val_n).item()
            if val_loss < best_loss - 1e-4:
                best_loss = val_loss
                epochs_without_improvement = 0
                best_state = copy.deepcopy(self.network.state_dict())
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    break

        if best_state is not None:
            self.network.load_state_dict(best_state)
        self.network.eval()
        return best_loss

    @torch.no_grad()
    def logits(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(self._normalize(inputs))

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.logits(inputs).argmax(-1).cpu()

    def per_example_nll(self, inputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(self.logits(inputs), labels.to(self.device), reduction="none").cpu()


def ridge_r2(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    regularization: float = 1.0,
) -> float:
    """Held-out ridge regression R², averaged over output dimensions."""
    x_train, y_train = x_train.double(), y_train.double()
    x_test, y_test = x_test.double(), y_test.double()
    mean, std = x_train.mean(0), x_train.std(0).clamp_min(1e-6)
    x_train, x_test = (x_train - mean) / std, (x_test - mean) / std
    y_mean = y_train.mean(0)
    weights = torch.linalg.solve(
        x_train.T @ x_train + regularization * torch.eye(x_train.size(1), dtype=torch.double),
        x_train.T @ (y_train - y_mean),
    )
    predictions = x_test @ weights + y_mean
    total_variance = (y_test - y_test.mean(0)).pow(2).sum(0)
    valid = total_variance > 1e-8
    if valid.sum() == 0:
        return float("nan")
    residual_variance = (y_test - predictions).pow(2).sum(0)[valid]
    return (1 - residual_variance / total_variance[valid]).mean().item()


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    """Linear centered kernel alignment between two representation matrices."""
    x, y = x.double(), y.double()
    x, y = x - x.mean(0), y - y.mean(0)
    numerator = (x.T @ y).norm() ** 2
    denominator = (x.T @ x).norm() * (y.T @ y).norm()
    return (numerator / denominator.clamp_min(1e-12)).item()


def mutual_knn(x: torch.Tensor, y: torch.Tensor, k: int = 10) -> float:
    """Fraction of k-nearest neighbours shared between two representations."""
    x = F.normalize(x - x.mean(0), dim=1)
    y = F.normalize(y - y.mean(0), dim=1)
    similarity_x, similarity_y = x @ x.T, y @ y.T
    similarity_x.fill_diagonal_(-1e9)
    similarity_y.fill_diagonal_(-1e9)
    neighbors_x = similarity_x.topk(k, 1).indices
    neighbors_y = similarity_y.topk(k, 1).indices
    overlap = (neighbors_x.unsqueeze(2) == neighbors_y.unsqueeze(1)).any(2).float().sum(1)
    return overlap.div(k).mean().item()


def heldout_procrustes(
    vision_train: torch.Tensor,
    text_train: torch.Tensor,
    vision_test: torch.Tensor,
    text_test: torch.Tensor,
) -> tuple[float, float]:
    """Fit an orthogonal map on train data and compare it against a shuffled null."""
    vision_mean, text_mean = vision_train.mean(0), text_train.mean(0)
    u, _, vh = torch.linalg.svd(
        (text_train - text_mean).double().T @ (vision_train - vision_mean).double()
    )
    rotation = (u @ vh).float()
    real = (
        F.cosine_similarity(vision_test - vision_mean, (text_test - text_mean) @ rotation, dim=-1)
        .mean()
        .item()
    )
    permutation = torch.randperm(len(text_test))
    null = (
        F.cosine_similarity(
            vision_test - vision_mean, (text_test[permutation] - text_mean) @ rotation, dim=-1
        )
        .mean()
        .item()
    )
    return real, null


def representation_diagnostics(x: torch.Tensor) -> tuple[float, float]:
    """Return per-dimension standard deviation and effective rank."""
    x = x.double()
    std = x.std(0).mean().item()
    covariance = torch.cov(x.T) + 1e-12 * torch.eye(x.size(1), dtype=torch.double)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
    effective_rank = (eigenvalues.sum() ** 2 / eigenvalues.pow(2).sum().clamp_min(1e-18)).item()
    return std, effective_rank


def concatenate_features(blocks: dict[str, torch.Tensor], keys: list[str], indices) -> torch.Tensor:
    """Concatenate the selected feature blocks at the selected row indices."""
    return torch.cat([blocks[key][indices] for key in keys], dim=-1)


def evaluate_block_swap(
    probe: LinearProbe,
    blocks: dict[str, torch.Tensor],
    keys: list[str],
    test_indices: torch.Tensor,
    labels: torch.Tensor,
    swap_key: str,
    generator: torch.Generator,
) -> tuple[float, float]:
    """Flip rate and donor-transfer rate after swapping one latent block."""
    n_test = len(test_indices)
    test_labels = labels[test_indices]
    permutation = torch.randperm(n_test, generator=generator)
    for _ in range(20):
        collisions = test_labels[permutation] == test_labels
        if not collisions.any():
            break
        permutation[collisions] = torch.randint(
            0, n_test, (int(collisions.sum()),), generator=generator
        )
    valid = test_labels[permutation] != test_labels
    if valid.sum() == 0:
        return float("nan"), float("nan")

    original = probe.predict(concatenate_features(blocks, keys, test_indices))
    swapped_inputs = torch.cat(
        [
            blocks[key][test_indices][permutation] if key == swap_key else blocks[key][test_indices]
            for key in keys
        ],
        dim=-1,
    )
    swapped = probe.predict(swapped_inputs)
    flip_rate = (original != swapped)[valid].float().mean().item()
    transfer_rate = (swapped == test_labels[permutation])[valid].float().mean().item()
    return flip_rate, transfer_rate


# =============================================================================
# PID: minimum mutual information decomposition and estimators
# =============================================================================
def minimum_mutual_information_pid(i_1: float, i_2: float, i_12: float) -> dict[str, float]:
    """Decompose two source-target mutual informations into PID atoms."""
    i_1, i_2 = max(i_1, 0.0), max(i_2, 0.0)
    i_12 = max(i_12, i_1, i_2)
    redundancy = min(i_1, i_2)
    return {
        "I1": i_1,
        "I2": i_2,
        "I12": i_12,
        "R": redundancy,
        "U1": i_1 - redundancy,
        "U2": i_2 - redundancy,
        "S": i_12 - max(i_1, i_2),
    }


def evaluate_pid_match(
    pid: dict[str, float],
    expected: str | None,
    reference_total: float | None,
    min_bits: float = 0.05,
) -> tuple[bool | None, float | None]:
    """Compare an estimated PID against the synthetic ground-truth expectation."""
    if expected is None:
        return None, None
    total = pid["R"] + pid["U1"] + pid["U2"] + pid["S"]
    if expected == "none":
        if reference_total is None or reference_total < min_bits:
            return None, None
        fraction = total / reference_total
        return bool(fraction < 0.15), fraction
    if total < min_bits:
        return False, 0.0
    components = {"R": pid["R"], "U1": pid["U1"], "U2": pid["U2"], "S": pid["S"]}
    fraction = components[expected] / total
    dominant = max(components, key=components.get) == expected
    return bool(dominant and fraction >= 0.5), fraction


class AffineCoupling(nn.Module):
    """An affine coupling layer for a normalizing flow."""

    def __init__(self, dim: int, hidden_dim: int, mask: torch.Tensor):
        super().__init__()
        self.register_buffer("mask", mask.float())
        self.network = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * dim),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = self.mask
        scale, shift = self.network(x * mask).chunk(2, -1)
        scale = torch.tanh(scale) * (1 - mask)
        shift = shift * (1 - mask)
        transformed = mask * x + (1 - mask) * (x * torch.exp(scale)) + shift
        return transformed, scale.sum(-1)


class NormalizingFlow(nn.Module):
    """A small affine normalizing flow with a learned diagonal output transform."""

    def __init__(self, dim: int, hidden_dim: int = 128, n_layers: int = 4):
        super().__init__()
        self.layers = nn.ModuleList()
        if dim >= 2:
            for _ in range(n_layers):
                mask = torch.zeros(dim)
                mask[torch.randperm(dim)[: max(1, dim // 2)]] = 1
                self.layers.append(AffineCoupling(dim, hidden_dim, mask))
        self.log_scale = nn.Parameter(torch.zeros(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        log_determinant = torch.zeros(x.size(0), device=x.device)
        for layer in self.layers:
            x, layer_log_determinant = layer(x)
            log_determinant = log_determinant + layer_log_determinant
        return x * self.log_scale.exp() + self.bias, log_determinant + self.log_scale.sum()


def whiten_and_reduce(
    x_train: torch.Tensor, x_test: torch.Tensor, n_components: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Whiten using training statistics and keep the top principal components."""
    mean = x_train.mean(0)
    centered = (x_train - mean).double()
    eigenvalues, eigenvectors = torch.linalg.eigh(centered.T @ centered / (len(centered) - 1))
    order = eigenvalues.argsort(descending=True)[:n_components]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    keep = eigenvalues > 1e-8
    projection = (eigenvectors[:, keep] / eigenvalues[keep].sqrt()).float()
    return (x_train - mean) @ projection, (x_test - mean) @ projection


def quantile_gaussianize(reference: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Map values to Gaussian scores using the reference empirical distribution."""
    sorted_reference = reference.sort().values
    ranks = torch.searchsorted(sorted_reference, values.contiguous())
    quantiles = ((ranks.float() + 0.5) / (len(sorted_reference) + 1)).clamp(1e-4, 1 - 1e-4)
    return (2.0**0.5) * torch.erfinv(2 * quantiles - 1)


def gaussian_mutual_information(
    covariance: torch.Tensor, indices_a: list[int], indices_b: list[int]
) -> float:
    """Mutual information of two Gaussian groups from their joint covariance."""

    def log_determinant(indices: list[int]) -> torch.Tensor:
        return torch.logdet(covariance[indices][:, indices])

    return (
        0.5
        * (
            log_determinant(indices_a)
            + log_determinant(indices_b)
            - log_determinant(indices_a + indices_b)
        )
        / LN_2
    ).item()


def estimate_flow_pid(
    a_train: torch.Tensor,
    b_train: torch.Tensor,
    target_train: torch.Tensor,
    a_test: torch.Tensor,
    b_test: torch.Tensor,
    target_test: torch.Tensor,
    steps: int,
    device: str,
    seed: int,
    hidden_dim: int = 128,
    n_layers: int = 4,
) -> dict[str, float]:
    """Estimate PID atoms by fitting Gaussianizing flows to both sources."""
    torch.manual_seed(seed)
    target_train_raw, target_test_raw = target_train, target_test
    a_train, b_train, a_test, b_test = [
        tensor.to(device).float() for tensor in (a_train, b_train, a_test, b_test)
    ]
    z_target_train = quantile_gaussianize(target_train_raw, target_train_raw).to(device)
    z_target_test = quantile_gaussianize(target_train_raw, target_test_raw).to(device)
    dim_a, dim_b = a_train.size(1), b_train.size(1)
    flow_a = NormalizingFlow(dim_a, hidden_dim, n_layers).to(device)
    flow_b = NormalizingFlow(dim_b, hidden_dim, n_layers).to(device)
    optimizer = torch.optim.Adam(list(flow_a.parameters()) + list(flow_b.parameters()), lr=2e-3)
    identity = torch.eye(dim_a + dim_b + 1, device=device)

    for _ in range(steps):
        indices = torch.randint(len(a_train), (min(512, len(a_train)),), device=device)
        z_a, log_determinant_a = flow_a(a_train[indices])
        z_b, log_determinant_b = flow_b(b_train[indices])
        latent = torch.cat([z_a, z_b, z_target_train[indices, None]], 1)
        centered = latent - latent.mean(0)
        covariance = centered.T @ centered / (len(latent) - 1) + 1e-3 * identity
        negative_log_likelihood = (
            0.5 * torch.logdet(covariance) - (log_determinant_a + log_determinant_b).mean()
        )
        optimizer.zero_grad()
        negative_log_likelihood.backward()
        torch.nn.utils.clip_grad_norm_(list(flow_a.parameters()) + list(flow_b.parameters()), 5.0)
        optimizer.step()

    with torch.no_grad():
        z_a, _ = flow_a(a_test)
        z_b, _ = flow_b(b_test)

        def mutual_information_scores(
            gaussianized_target: torch.Tensor,
        ) -> tuple[float, float, float]:
            latent = torch.cat([z_a, z_b, gaussianized_target[:, None]], 1).double()
            centered = latent - latent.mean(0)
            covariance = centered.T @ centered / (len(latent) - 1)
            covariance = covariance + 1e-5 * torch.eye(
                latent.size(1), dtype=torch.double, device=device
            )
            indices_a = list(range(dim_a))
            indices_b = list(range(dim_a, dim_a + dim_b))
            target_index = [dim_a + dim_b]
            return (
                gaussian_mutual_information(covariance, indices_a, target_index),
                gaussian_mutual_information(covariance, indices_b, target_index),
                gaussian_mutual_information(covariance, indices_a + indices_b, target_index),
            )

        pid = minimum_mutual_information_pid(*mutual_information_scores(z_target_test))
        shuffled_target = z_target_test[torch.randperm(len(z_target_test), device=device)]
        pid["null_I12"] = max(mutual_information_scores(shuffled_target)[2], 0.0)
    return pid


def estimate_flow_pid_pair(
    blocks: dict[str, torch.Tensor],
    keys_a: list[str],
    keys_b: list[str],
    target: torch.Tensor,
    train_indices: torch.Tensor,
    test_indices: torch.Tensor,
    args,
    device: str,
    seed: int,
) -> dict[str, float] | None:
    """Prepare two feature groups and estimate flow-PID for one target."""
    valid = ~torch.isnan(target)
    valid_train = train_indices[valid[train_indices]]
    valid_test = test_indices[valid[test_indices]]
    if len(valid_train) < 300 or len(valid_test) < 200:
        return None
    features_a = concatenate_features(blocks, keys_a, slice(None)).float()
    features_b = concatenate_features(blocks, keys_b, slice(None)).float()
    a_train, a_test = whiten_and_reduce(
        features_a[valid_train], features_a[valid_test], args.pid_dim
    )
    b_train, b_test = whiten_and_reduce(
        features_b[valid_train], features_b[valid_test], args.pid_dim
    )
    return estimate_flow_pid(
        a_train,
        b_train,
        target[valid_train],
        a_test,
        b_test,
        target[valid_test],
        args.pid_steps,
        device,
        seed,
    )


def estimate_probe_pid(
    cross_entropy_a: torch.Tensor,
    cross_entropy_b: torch.Tensor,
    cross_entropy_joint: torch.Tensor,
    prior_cross_entropy: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> dict[str, float]:
    """Estimate PID atoms from probe cross-entropies (probe-PID)."""
    selector = slice(None) if mask is None else mask
    entropy = prior_cross_entropy[selector].mean()

    def mutual_information(cross_entropy: torch.Tensor) -> float:
        bits = (entropy - cross_entropy[selector].mean()).item() / LN_2
        return max(bits, 0.0)

    return minimum_mutual_information_pid(
        mutual_information(cross_entropy_a),
        mutual_information(cross_entropy_b),
        mutual_information(cross_entropy_joint),
    )


# =============================================================================
# Per-seed execution
# =============================================================================
def run_seed_experiment(seed: int, args, bundle: Bundle, device: str) -> dict:
    """Train every arm and run the complete evaluation suite for one seed."""
    set_seed(seed)
    train_indices = bundle.split["train"]
    val_indices = bundle.split["val"]
    test_indices = bundle.split["test"]
    sigreg = SIGReg(n_slices=args.sigreg_slices).to(device)

    arm_features = {"raw": extract_raw_features(bundle)}
    arm_models = {}
    for arm_name, arm_config in ARM_CONFIGS.items():
        print(f"  -- pretraining arm: {arm_name}", flush=True)
        set_seed(seed)
        model = build_model(bundle, arm_config, args).to(device)
        pretrain_arm(model, arm_config, bundle, args, device, sigreg, arm_name)
        arm_features[arm_name] = encode_dataset(model, bundle, device)
        arm_models[arm_name] = model
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    output = dict(
        acc={},
        corr={},
        ce={},
        prior_ce={},
        idmat={},
        align={},
        swap={},
        scalars={},
        majority={},
        diag={},
        fpid={},
        ppid={},
        probes={},
        models=arm_models,
        blocks=arm_features,
    )
    generator = torch.Generator().manual_seed(seed)

    # ---- Collapse diagnostics
    for arm, blocks in arm_features.items():
        for block_name, values in blocks.items():
            output["diag"][(arm, block_name)] = representation_diagnostics(
                values[test_indices][:4000]
            )

    # ---- Probes: accuracy and per-example cross-entropy
    for task, labels in bundle.tasks.items():
        class_counts = (
            torch.bincount(labels[train_indices], minlength=bundle.n_classes[task]).float() + 1.0
        )
        log_priors = (class_counts / class_counts.sum()).log()
        output["prior_ce"][task] = -log_priors[labels[test_indices]]
        output["majority"][task] = (
            labels[test_indices] == class_counts.argmax()
        ).float().mean().item() * 100

    probe_seed = seed * 10_000
    for arm, blocks in arm_features.items():
        feature_sets = FEATURE_SETS[arm_family(arm)]
        for task, labels in bundle.tasks.items():
            for feature_set_name, keys in feature_sets.items():
                probe_seed += 1
                x_train, x_val, x_test = (
                    concatenate_features(blocks, keys, indices)
                    for indices in (train_indices, val_indices, test_indices)
                )
                probe = LinearProbe(
                    x_train.size(1), bundle.n_classes[task], device, seed=probe_seed
                )
                probe.fit(
                    x_train,
                    labels[train_indices],
                    x_val,
                    labels[val_indices],
                    epochs=args.probe_epochs,
                )
                correct = (probe.predict(x_test) == labels[test_indices]).numpy().astype(np.float32)
                output["acc"][(arm, feature_set_name, task)] = correct.mean() * 100
                output["corr"][(arm, feature_set_name, task)] = correct
                output["ce"][(arm, feature_set_name, task)] = probe.per_example_nll(
                    x_test, labels[test_indices]
                )
                output["probes"][(arm, feature_set_name, task)] = probe
                if feature_set_name == "joint" and arm != "raw":
                    for block_name in keys:
                        output["swap"][(arm, task, block_name)] = evaluate_block_swap(
                            probe,
                            blocks,
                            keys,
                            test_indices,
                            labels,
                            block_name,
                            generator,
                        )

    # ---- Factor identification
    for arm, blocks in arm_features.items():
        feature_sets = FEATURE_SETS[arm_family(arm)]
        rows = dict(blocks)
        rows["stream_v"] = concatenate_features(blocks, feature_sets["v"], slice(None))
        rows["stream_t"] = concatenate_features(blocks, feature_sets["t"], slice(None))
        for row_name, features in rows.items():
            for factor_name, factor_values in bundle.factors.items():
                output["idmat"][(arm, row_name, factor_name)] = ridge_r2(
                    features[train_indices],
                    factor_values[train_indices],
                    features[test_indices],
                    factor_values[test_indices],
                )

    # ---- Cross-modal alignment
    subset = test_indices[: min(len(test_indices), 2000)]
    alignment_pairs = {
        "raw": [("h1", "h2")],
        "alignment": [("z1", "z2")],
        "djepa": [("c1", "c2"), ("u1", "u2")],
    }
    for arm, blocks in arm_features.items():
        for key_a, key_b in alignment_pairs[arm_family(arm)]:
            features_a, features_b = blocks[key_a], blocks[key_b]
            key = (arm, f"{key_a}~{key_b}")
            permutation = torch.randperm(len(subset))
            output["align"][key] = {
                "cka": linear_cka(features_a[subset], features_b[subset]),
                "cka_null": linear_cka(features_a[subset], features_b[subset][permutation]),
                "mknn": mutual_knn(features_a[subset], features_b[subset]),
                "mknn_null": mutual_knn(features_a[subset], features_b[subset][permutation]),
            }
            if features_a.size(1) == features_b.size(1):
                real, null = heldout_procrustes(
                    features_a[train_indices][:5000],
                    features_b[train_indices][:5000],
                    features_a[subset],
                    features_b[subset],
                )
                output["align"][key].update(procrustes=real, procrustes_null=null)

    # ---- Synthetic decomposition scalars
    if bundle.has_ground_truth:
        idmat = output["idmat"]
        for arm in arm_features:
            scores = {
                "shared_captured": np.nanmean(
                    [idmat[(arm, "stream_v", "shared")], idmat[(arm, "stream_t", "shared")]]
                ),
                "private_retained": np.nanmean(
                    [idmat[(arm, "stream_v", "priv_v")], idmat[(arm, "stream_t", "priv_t")]]
                ),
            }
            if arm_family(arm) == "djepa":
                good = np.mean(
                    [
                        idmat[(arm, "c1", "shared")],
                        idmat[(arm, "c2", "shared")],
                        idmat[(arm, "u1", "priv_v")],
                        idmat[(arm, "u2", "priv_t")],
                    ]
                )
                leakage = np.mean(
                    [
                        idmat[(arm, "c1", "priv_v")],
                        idmat[(arm, "c2", "priv_t")],
                        idmat[(arm, "u1", "shared")],
                        idmat[(arm, "u2", "shared")],
                    ]
                )
                scores.update(
                    c_to_shared=np.mean(
                        [idmat[(arm, "c1", "shared")], idmat[(arm, "c2", "shared")]]
                    ),
                    u_to_private=np.mean(
                        [idmat[(arm, "u1", "priv_v")], idmat[(arm, "u2", "priv_t")]]
                    ),
                    leakage=leakage,
                    DS=good - leakage,
                )
            output["scalars"][arm] = scores

    # ---- PID estimators
    subsets: dict[str, torch.Tensor | None] = {"all": None}
    if bundle.categories is not None:
        test_categories = bundle.categories[test_indices.numpy()]
        for category in sorted(set(test_categories)):
            subsets[category] = torch.from_numpy(test_categories == category)
    for arm, blocks in arm_features.items():
        for pair_name, feature_a, feature_b, feature_joint in PID_STREAM_PAIRS[arm_family(arm)]:
            for task in bundle.tasks:
                for subset_name, subset_mask in subsets.items():
                    output["ppid"][(arm, pair_name, task, subset_name)] = estimate_probe_pid(
                        output["ce"][(arm, feature_a, task)],
                        output["ce"][(arm, feature_b, task)],
                        output["ce"][(arm, feature_joint, task)],
                        output["prior_ce"][task],
                        subset_mask,
                    )
            if not args.no_flow_pid:
                feature_sets = FEATURE_SETS[arm_family(arm)]
                for target_name, target_values in bundle.pid_targets.items():
                    pid = estimate_flow_pid_pair(
                        blocks,
                        feature_sets[feature_a],
                        feature_sets[feature_b],
                        target_values,
                        train_indices,
                        test_indices,
                        args,
                        device,
                        seed,
                    )
                    if pid is not None:
                        output["fpid"][(arm, pair_name, target_name)] = pid
        print(f"    PID done: {arm}", flush=True)

    if bundle.has_ground_truth:
        for field_name in ("ppid", "fpid"):
            pid_table = output[field_name]
            for key, pid in pid_table.items():
                arm, pair_name, target = key[:3]
                if field_name == "ppid" and key[3] != "all":
                    continue
                expected = EXPECTED_COMPONENTS.get(target, {}).get(pair_name)
                reference = pid_table.get((arm, "stream", target) + key[3:])
                reference_total = (
                    None
                    if reference is None
                    else reference["R"] + reference["U1"] + reference["U2"] + reference["S"]
                )
                pid["expected"] = expected
                pid["ok"], pid["frac"] = evaluate_pid_match(pid, expected, reference_total)
    return output
