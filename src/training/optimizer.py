import optax
import flax.nnx as nnx


def create_optimizer(config: "TrainConfig", model: nnx.Module) -> optax.GradientTransformation:
    """Create optimizer with weight decay."""
    tx = optax.chain(
        optax.clip_by_global_norm(config.grad_clip),
        optax.adamw(
            learning_rate=config.learning_rate,
            b1=config.beta1,
            b2=config.beta2,
            eps=config.eps,
            weight_decay=config.weight_decay,
        ),
    )
    return tx