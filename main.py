import argparse
from pathlib import Path

from omegaconf import OmegaConf

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a PID-JEPA pretraining experiment."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train.yaml"),
        help="Path to the YAML experiment config (default: configs/train.yaml).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="Seeds to run. Overrides the seeds list in the config.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.config.is_file():
        raise FileNotFoundError(f"Config file not found: {args.config}")

    config = OmegaConf.load(args.config)
    objective = config.get("objective", "decomposition")
    if objective == "factorcl":
        from src.factorcl.train import train
    elif objective in {"decomposition", "djepa"}:
        from src.djepa.train import train
    else:
        raise ValueError(
            "objective must be 'decomposition', 'djepa', or 'factorcl'."
        )

    seeds = args.seeds or config.get("seeds", [config.get("seed", 0)])
    experiment_name = config.get("experiment_name", args.config.stem)
    output_root = Path(config.get("output_dir", "runs")) / experiment_name

    for seed in seeds:
        run_config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
        run_config.seed = seed
        train(
            run_config,
            output_dir=output_root / f"seed_{seed}",
            run_name=f"{experiment_name}-seed-{seed}",
        )
