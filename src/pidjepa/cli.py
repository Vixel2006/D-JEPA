"""Command-line entry point for the PID-JEPA benchmark."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Sequence

import torch

from pidjepa.data import load_clevr_bundle, make_synthetic_bundle
from pidjepa.evaluation import run_seed_experiment
from pidjepa.figures import make_all_figures
from pidjepa.reporting import aggregate_results
from pidjepa.utils import make_json_serializable


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the benchmark."""
    parser = argparse.ArgumentParser(
        prog="pidjepa",
        description=(
            "Partial-information decomposition of multimodal joint-embedding "
            "pretraining on synthetic data or CLEVR."
        ),
    )
    parser.add_argument("--mode", choices=["synthetic", "clevr"], default="clevr")
    parser.add_argument("--seeds", type=int, default=3, help="number of seeds (42, 43, ...)")
    parser.add_argument("--pretrain-steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--block-dim",
        type=int,
        default=64,
        help="dimensions per shared/private block (default: 64)",
    )
    parser.add_argument("--hidden-dim", type=int, default=512, help="MLP/encoder hidden width")
    parser.add_argument("--transformer-dim", type=int, default=256)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--sigreg-weight", type=float, default=0.05)
    parser.add_argument("--invariance-weight", type=float, default=1.0)
    parser.add_argument("--cross-modal-weight", type=float, default=1.0)
    parser.add_argument("--reconstruction-weight", type=float, default=1.0)
    parser.add_argument(
        "--independence-weight",
        type=float,
        default=0.2,
        help="independence penalty (scaled down from 1.0)",
    )
    parser.add_argument("--sigreg-slices", type=int, default=256)
    parser.add_argument(
        "--views",
        type=int,
        default=3,
        help="augmented views per sample (must be at least 3)",
    )
    parser.add_argument(
        "--synthetic-samples", type=int, default=12000, help="synthetic sample count"
    )
    parser.add_argument("--probe-epochs", type=int, default=60)
    parser.add_argument(
        "--pid-dim", type=int, default=64, help="max dims per source for flow-PID (whitened PCA)"
    )
    parser.add_argument("--pid-steps", type=int, default=400)
    parser.add_argument("--no-flow-pid", action="store_true")
    parser.add_argument("--clevr-images", type=int, default=3000)
    parser.add_argument("--questions-per-image", type=int, default=6)
    parser.add_argument(
        "--vision-grid", type=int, default=4, help="DINOv2 patch tokens pooled to grid x grid"
    )
    parser.add_argument("--max-answer-classes", type=int, default=40)
    parser.add_argument("--max-text-length", type=int, default=32)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--device", default=None)
    parser.add_argument("--quick", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark, write ``report.json``, and generate all figures."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.seeds < 1:
        parser.error("--seeds must be at least 1")
    if args.views < 3:
        parser.error("--views must be at least 3")

    if args.quick:
        args.pretrain_steps = 300
        args.synthetic_samples = 3000
        args.probe_epochs = 15
        args.seeds = 2
        args.hidden_dim = 128
        args.pid_steps = 100
        args.transformer_dim = 128

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    print(
        f"[run] mode={args.mode} device={device} seeds={args.seeds} "
        f"steps={args.pretrain_steps} block_dim={args.block_dim}"
    )

    clevr_bundle = load_clevr_bundle(args, device) if args.mode == "clevr" else None
    results = []
    bundle = clevr_bundle
    for seed_index in range(args.seeds):
        seed = 42 + seed_index
        start = time.time()
        print(f"\n===== SEED {seed} =====")
        if clevr_bundle is None:
            bundle = make_synthetic_bundle(args.synthetic_samples, seed, args.views)
        results.append(run_seed_experiment(seed, args, bundle, device))
        print(f"  seed {seed} done in {time.time() - start:.0f}s")

    summary = aggregate_results(results, bundle)

    report_path = os.path.join(args.output_dir, "report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(
            make_json_serializable({"args": vars(args), "report": summary}),
            handle,
            indent=1,
        )
    print(f"\n[saved] {report_path}")

    print("\n[*] Generating complete publication figure suite...")
    make_all_figures(summary, bundle, results, args.output_dir)
    print(f"[*] All figures successfully generated and saved to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
