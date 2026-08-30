from dataclasses import dataclass


@dataclass
class TrainConfig:
    seed: int = 42
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    grad_clip: float = 1.0

    num_epochs: int = 10
    batch_size: int = 32

    kl_weight: float = 1.0
    image_recon_weight: float = 1.0
    text_recon_weight: float = 1.0

    log_every: int = 10
    eval_every: int = 100
    save_every: int = 500

    num_workers: int = 4
    prefetch: int = 2

    mixed_precision: bool = False
    compile: bool = True

    checkpoint_dir: str = "./checkpoints"
    tensorboard_dir: str = "./tensorboard_logs"
    max_checkpoints: int = 5