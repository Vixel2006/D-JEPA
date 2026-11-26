import argparse
from pathlib import Path

from omegaconf import OmegaConf

from src.train import train


def parse_args():
    parser = argparse.ArgumentParser(description="Run a PID-JEPA pretraining experiment.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train.yaml"),
        help="Path to the YAML experiment config (default: configs/train.yaml).",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if not args.config.is_file():
        raise FileNotFoundError(f"Config file not found: {args.config}")
    train(OmegaConf.load(args.config))
