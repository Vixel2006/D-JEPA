"""Train and evaluate DJEPA models from a small command-line interface."""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import time
from pathlib import Path

import jax
import numpy as np
import optax
import orbax.checkpoint as ocp

if __package__:
    from .data import (
        download_multibench_dataset,
        load_multibench_prepared_dataset,
        load_npz_dataset,
        make_synthetic_dataset,
    )
    from .eval import evaluate_flow_pid, evaluate_representations
    from .models import DJEPA, AudioEncoder, FeatureEncoder, TextEncoder, VisionEncoder
    from .train import create_pretrain_state, make_train_step
    from .utils.figures import plot_factor_r2_heatmap, plot_tsne
else:
    from src.data import (
        download_multibench_dataset,
        load_multibench_prepared_dataset,
        load_npz_dataset,
        make_synthetic_dataset,
    )
    from src.eval import evaluate_flow_pid, evaluate_representations
    from src.models import DJEPA, AudioEncoder, FeatureEncoder, TextEncoder, VisionEncoder
    from src.train import create_pretrain_state, make_train_step
    from src.utils.figures import plot_factor_r2_heatmap, plot_tsne


VARIANTS = {"lejepa": None}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "eval", "train+eval"), default="train+eval")
    parser.add_argument(
        "--dataset", choices=("synthetic", "npz", "mmimdb", "ur-funny"), default="synthetic"
    )
    parser.add_argument("--dataset-path", type=Path, help="Dataset file; downloaded if omitted for MultiBench")
    parser.add_argument("--modalities", help="Comma-separated subset, e.g. vision,text,audio")
    parser.add_argument("--variant", choices=(*VARIANTS, "all"), default="lejepa")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-samples", type=int, default=1024)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--projection-dim", type=int, default=32)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=1, help="Number of consecutive seeds to run")
    parser.add_argument("--sigreg-weight", type=float, default=0.05)
    parser.add_argument("--alignment-weight", type=float, default=1.0)
    parser.add_argument("--covariance-weight", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/lejepa"))
    parser.add_argument("--flow-pid", action="store_true", help="Run FlowPID during evaluation")
    parser.add_argument("--flow-steps", type=int, default=200)
    parser.add_argument("--no-plots", action="store_true", help="Skip t-SNE and heatmap files")
    return parser.parse_args()


def _load_dataset(args: argparse.Namespace):
    modalities = tuple(args.modalities.split(",")) if args.modalities else None
    if args.dataset == "synthetic":
        names = modalities or ("vision", "text", "audio")
        return make_synthetic_dataset(args.n_samples, args.seed, names)
    if args.dataset in {"mmimdb", "ur-funny"}:
        path = args.dataset_path or download_multibench_dataset(args.dataset)
        return load_multibench_prepared_dataset(args.dataset, path)
    if args.dataset_path is None:
        raise ValueError("--dataset-path is required when --dataset is npz")
    return load_npz_dataset(args.dataset_path, modalities, seed=args.seed)


def _make_model(modalities: tuple[str, ...], args: argparse.Namespace) -> DJEPA:
    encoders = {}
    for name in modalities:
        if args.dataset in {"mmimdb", "ur-funny"}:
            encoders[name] = FeatureEncoder(
                embedding_dim=args.embedding_dim,
                num_heads=4,
                num_layers=1,
                mlp_dim=2 * args.embedding_dim,
            )
            continue
        if name == "vision":
            encoders[name] = VisionEncoder(
                embedding_dim=args.embedding_dim,
                patch_size=8,
                num_heads=4,
                num_layers=1,
                mlp_dim=2 * args.embedding_dim,
            )
        elif name == "text":
            encoders[name] = TextEncoder(
                vocab_size=args.vocab_size,
                embedding_dim=args.embedding_dim,
                num_heads=4,
                num_layers=1,
                mlp_dim=2 * args.embedding_dim,
            )
        elif name == "audio":
            encoders[name] = AudioEncoder(
                embedding_dim=args.embedding_dim,
                num_heads=4,
                num_layers=1,
                mlp_dim=2 * args.embedding_dim,
            )
        else:
            raise ValueError(f"Unknown modality {name!r}; choose vision, text, or audio")
    return DJEPA(encoders=encoders, projection_dim=args.projection_dim)


def _optimizer(learning_rate: float):
    return optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(learning_rate))


def _checkpoint_payload(state):
    return {
        "params": state.params,
        "opt_state": state.opt_state,
        "step": state.step,
        "rng": state.rng,
    }


def _save_checkpoint(path: Path, state) -> None:
    if path.exists():
        path = path.with_name(f"{path.name}_{time.time_ns()}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with ocp.StandardCheckpointer() as checkpointer:
        checkpointer.save(str(path), _checkpoint_payload(state))
        checkpointer.wait_until_finished()


def _restore_checkpoint(directory: Path, state):
    checkpoints = sorted(directory.glob("step_*"), key=lambda path: path.stat().st_mtime_ns)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint found in {directory}")
    with ocp.StandardCheckpointer() as checkpointer:
        payload = checkpointer.restore(str(checkpoints[-1]), _checkpoint_payload(state))
    return state.replace(**payload)


def _train(args, model: DJEPA, dataset, variant: str, state, optimizer):
    train_step = make_train_step(
        model,
        optimizer,
        sigreg_weight=args.sigreg_weight,
        alignment_weight=args.alignment_weight,
        covariance_weight=args.covariance_weight,
    )
    rng = np.random.default_rng(args.seed)
    history = []
    for epoch in range(1, args.epochs + 1):
        indices = rng.permutation(dataset.train_indices)
        epoch_logs = []
        for start in range(0, len(indices), args.batch_size):
            batch = indices[start : start + args.batch_size]
            context = {name: values[batch] for name, values in dataset.context.items()}
            target = {name: values[batch] for name, values in dataset.target.items()}
            state, logs = train_step(state, context, target)
            epoch_logs.append({name: float(value) for name, value in logs.items()})
        means = {name: float(np.mean([row[name] for row in epoch_logs])) for name in epoch_logs[0]}
        history.append(means)
        print(f"{variant} epoch {epoch:03d}/{args.epochs}: loss={means['loss']:.4f}")

    checkpoint_dir = args.output_dir / "checkpoints" / variant
    checkpoint_path = checkpoint_dir / f"step_{int(state.step):08d}"
    _save_checkpoint(checkpoint_path, state)
    return state, history


def _evaluate(args, model: DJEPA, dataset, variant: str, state):
    params = state.params
    indices = dataset.test_indices
    context = dataset.context
    target = dataset.target
    outputs = model.apply({"params": params}, context, target)
    representations = {
        **{f"{name}_unique": value for name, value in outputs["context_unique"].items()},
        **{f"{name}_redundant": value for name, value in outputs["context_redundant"].items()},
    }
    report = evaluate_representations(
        representations,
        dataset.factors,
        dataset.train_indices,
        dataset.test_indices,
    )

    if args.flow_pid and len(dataset.modalities) >= 2 and dataset.factors:
        source_pairs = {
            f"{first}_vs_{second}": ([f"{first}_unique"], [f"{second}_unique"])
            for first, second in itertools.combinations(dataset.modalities, 2)
        }
        report["flow_pid"] = evaluate_flow_pid(
            representations,
            source_pairs,
            dataset.factors,
            dataset.train_indices,
            dataset.test_indices,
            seed=args.seed,
            pid_dim=args.projection_dim,
            steps=args.flow_steps,
        )

    output_dir = args.output_dir / "eval" / variant
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=True)
    if not args.no_plots:
        if report["r2"]:
            plot_factor_r2_heatmap(report["r2"], output_dir / "factor_r2.png")
        if len(indices) >= 3:
            tsne_values = np.concatenate(
                [outputs["context_unique"][name][indices] for name in dataset.modalities], axis=0
            )
            tsne_labels = np.concatenate(
                [np.repeat(name, len(indices)) for name in dataset.modalities]
            )
            plot_tsne(tsne_values, tsne_labels, output_dir / "unique_tsne.png")
    print(f"Saved evaluation for {variant} to {output_dir}")
    return report


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.n_samples < 20 or args.seeds < 1:
        raise ValueError("epochs, batch size, and seeds must be positive; n-samples must be at least 20")
    all_runs = []
    for seed_offset in range(args.seeds):
        run_args = copy.copy(args)
        run_args.seed = args.seed + seed_offset
        if args.seeds > 1:
            run_args.output_dir = args.output_dir / f"seed_{run_args.seed}"
        dataset = _load_dataset(run_args)
        variants = list(VARIANTS) if run_args.variant == "all" else [run_args.variant]
        run_args.output_dir.mkdir(parents=True, exist_ok=True)
        run_summary = {
            "dataset": run_args.dataset,
            "seed": run_args.seed,
            "modalities": dataset.modalities,
            "variants": {},
        }

        for variant in variants:
            model = _make_model(dataset.modalities, run_args)
            sample = {name: values[:2] for name, values in dataset.context.items()}
            target_sample = {name: values[:2] for name, values in dataset.target.items()}
            optimizer = _optimizer(run_args.learning_rate)
            state = create_pretrain_state(
                model, jax.random.PRNGKey(run_args.seed), sample, target_sample, optimizer
            )
            history = []
            if run_args.mode in ("train", "train+eval"):
                state, history = _train(run_args, model, dataset, variant, state, optimizer)
            elif run_args.mode == "eval":
                state = _restore_checkpoint(run_args.output_dir / "checkpoints" / variant, state)
            run_summary["variants"][variant] = {"history": history}
            if run_args.mode in ("eval", "train+eval"):
                run_summary["variants"][variant]["metrics"] = _evaluate(
                    run_args, model, dataset, variant, state
                )

        with (run_args.output_dir / "run.json").open("w", encoding="utf-8") as stream:
            json.dump(run_summary, stream, indent=2, allow_nan=True)
        all_runs.append(run_summary)
        print(f"Completed seed {run_args.seed}; outputs: {run_args.output_dir}")

    if args.seeds > 1:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with (args.output_dir / "runs.json").open("w", encoding="utf-8") as stream:
            json.dump(all_runs, stream, indent=2, allow_nan=True)


if __name__ == "__main__":
    main()
