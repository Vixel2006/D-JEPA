"""Command-line entry point for the CLEVR D-JEPA benchmark."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config import (
    DataConfig,
    FigureConfig,
    ModelConfig,
    OptimizationConfig,
    RunConfig,
)
from .pipeline import run
from .utils import configure_logging

LOGGER = logging.getLogger(__name__)


def parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not seeds:
        raise argparse.ArgumentTypeError("At least one seed is required.")
    return seeds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train LeJEPA and D-JEPA on CLEVR, then run the "
            "multi-seed probe, causal-swap, Procrustes, Flow-PID, and "
            "representation-geometry evaluations, and generate the paper "
            "figures."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=("clevr", "procedural"),
        default="clevr",
        help=(
            "Data source. 'clevr' streams the real Blender-rendered CLEVR "
            "split from HuggingFaceM4/the_cauldron; 'procedural' uses the "
            "offline synthetic generator (default: clevr)."
        ),
    )
    parser.add_argument(
        "--scenes",
        type=int,
        default=1500,
        help="Number of procedural scenes (default: 1500).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10_000,
        help="Maximum streamed CLEVR samples (default: 10000).",
    )
    parser.add_argument(
        "--hf-dataset",
        default="HuggingFaceM4/the_cauldron",
        help="Hugging Face dataset id for the real CLEVR split.",
    )
    parser.add_argument(
        "--hf-config",
        default="clevr",
        help="Hugging Face dataset configuration (default: clevr).",
    )
    parser.add_argument(
        "--hf-split",
        default="train",
        help="Hugging Face split (default: train).",
    )
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Download the Hugging Face split instead of streaming it.",
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Fail instead of falling back to procedural data.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for pretraining and probing (default: 64).",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=None,
        help=(
            "Test items. Default reproduces the original split: "
            "min(800, len(dataset) // 4)."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers. Keep 0 for bitwise reproducibility.",
    )
    parser.add_argument(
        "--max-question-length",
        type=int,
        default=64,
        help="Tokenizer maximum length (default: 64).",
    )
    parser.add_argument(
        "--aug-scale-min",
        type=float,
        default=0.7,
        help="Minimum RandomResizedCrop scale (default: 0.7).",
    )
    parser.add_argument(
        "--pretrain-steps",
        type=int,
        default=3000,
        help="Joint pretraining steps (default: 3000).",
    )
    parser.add_argument(
        "--probe-steps",
        type=int,
        default=600,
        help="Downstream probe steps per seed (default: 600).",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="Log pretraining/probe progress every N steps (default: 50).",
    )
    parser.add_argument(
        "--seeds",
        type=parse_seeds,
        default=(42, 43, 44),
        help="Comma-separated evaluation seeds (default: 42,43,44).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device specifier: auto, cpu, cuda, or cuda:0 (default: auto).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/clevr"),
        help="Directory for results, diagnostics, and figures.",
    )
    parser.add_argument(
        "--figures",
        dest="figures",
        action="store_true",
        default=True,
        help="Generate the full paper figure suite (default: on).",
    )
    parser.add_argument(
        "--no-figures",
        dest="figures",
        action="store_false",
        help="Skip figure generation.",
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Write PNG figures only, not PDF.",
    )
    parser.add_argument(
        "--figure-dpi",
        type=int,
        default=200,
        help="Raster figure DPI (default: 200).",
    )
    parser.add_argument(
        "--no-diagnostics",
        action="store_true",
        help="Skip spectral/CKA/calibration diagnostics.",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable CUDA automatic mixed precision.",
    )
    parser.add_argument(
        "--no-case-studies",
        action="store_true",
        help="Skip qualitative win/loss case studies.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Print tables without writing result files.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Console logging level (default: INFO).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    configure_logging(args.log_level)

    config = RunConfig(
        data=DataConfig(
            source=args.dataset,
            num_scenes=args.scenes,
            max_samples=args.max_samples,
            hf_dataset=args.hf_dataset,
            hf_config=args.hf_config,
            hf_split=args.hf_split,
            hf_streaming=not args.no_streaming,
            fallback_to_procedural=not args.no_fallback,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_question_length=args.max_question_length,
            aug_scale_min=args.aug_scale_min,
            test_size=args.test_size,
        ),
        model=ModelConfig(),
        optim=OptimizationConfig(
            pretrain_steps=args.pretrain_steps,
            probe_steps=args.probe_steps,
            log_every=args.log_every,
            amp=not args.no_amp,
        ),
        figures=FigureConfig(
            enabled=args.figures,
            dpi=args.figure_dpi,
            save_pdf=not args.no_pdf,
        ),
        seeds=args.seeds,
        device=args.device,
        output_dir=args.output_dir,
        save_results=not args.no_save,
        run_case_studies=not args.no_case_studies,
        collect_diagnostics=not args.no_diagnostics,
        log_level=args.log_level,
    )

    try:
        run(config)
    except Exception:
        LOGGER.exception("CLEVR benchmark failed")
        return 1
    return 0
