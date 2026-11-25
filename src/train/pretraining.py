"""LeJEPA optimization with shared parameters and SIGReg."""

from collections.abc import Callable

import flax
import jax
import jax.numpy as jnp
import optax
from flax import struct

from ..losses import loss_alignment, loss_cross_covariance, loss_sigreg
from ..models.djepa import DJEPA


@struct.dataclass
class PretrainState:
    """Shared encoder parameters, optimizer state, step, and PRNG state."""

    params: flax.core.FrozenDict
    opt_state: optax.OptState
    step: jax.Array
    rng: jax.Array


def create_pretrain_state(
    model: DJEPA,
    key: jax.Array,
    context: dict[str, jax.Array],
    target: dict[str, jax.Array],
    optimizer: optax.GradientTransformation,
) -> PretrainState:
    """Initialize the single shared parameter set used by both views."""
    init_key, rng = jax.random.split(key)
    params = flax.core.freeze(model.init(init_key, context, target)["params"])
    return PretrainState(params, optimizer.init(params), jnp.array(0), rng)


def _losses(
    outputs: dict[str, dict[str, jax.Array]],
    modalities: tuple[str, ...],
    key: jax.Array,
) -> dict[str, jax.Array]:
    """Compute LeJEPA prediction and per-view SIGReg losses."""
    predictions = []
    regularizers = []
    redundant_alignments = []
    cross_covariances = []
    view_count = 2 * len(modalities)
    keys = jax.random.split(key, view_count)
    index = 0
    for name in modalities:
        context_embedding = jnp.concatenate(
            (outputs["context_unique"][name], outputs["context_redundant"][name]),
            axis=-1,
        )
        target_embedding = jnp.concatenate(
            (outputs["target_unique"][name], outputs["target_redundant"][name]),
            axis=-1,
        )
        center = 0.5 * (context_embedding + target_embedding)
        predictions.append(
            0.5
            * (
                loss_alignment(center, context_embedding)
                + loss_alignment(center, target_embedding)
            )
        )
        regularizers.extend(
            (
                loss_sigreg(context_embedding, keys[index]),
                loss_sigreg(target_embedding, keys[index + 1]),
            )
        )
        cross_covariances.append(
            loss_cross_covariance(
                outputs["context_unique"][name],
                outputs["context_redundant"][name],
            )
        )
        index += 2
    redundant_alignments = [
        loss_alignment(
            outputs["context_redundant"][first],
            outputs["context_redundant"][second],
        )
        for index, first in enumerate(modalities)
        for second in modalities[index + 1 :]
    ]
    return {
        "prediction": jnp.mean(jnp.stack(predictions)),
        "sigreg": jnp.mean(jnp.stack(regularizers)),
        "alignment": (
            jnp.mean(jnp.stack(redundant_alignments))
            if redundant_alignments
            else jnp.array(0.0)
        ),
        "covariance": jnp.mean(jnp.stack(cross_covariances)),
    }


def make_train_step(
    model: DJEPA,
    optimizer: optax.GradientTransformation,
    *,
    sigreg_weight: float = 0.05,
    alignment_weight: float = 1.0,
    covariance_weight: float = 1.0,
) -> Callable:
    """Create a JIT LeJEPA step; ``sigreg_weight`` is the mixture λ."""
    if not 0.0 <= sigreg_weight <= 1.0:
        raise ValueError("sigreg_weight must be between 0 and 1")
    modalities = tuple(model.encoders)

    @jax.jit
    def train_step(
        state: PretrainState,
        context: dict[str, jax.Array],
        target: dict[str, jax.Array],
    ) -> tuple[PretrainState, dict[str, jax.Array]]:
        next_rng, sigreg_key = jax.random.split(state.rng)

        def objective(params: flax.core.FrozenDict):
            outputs = model.apply({"params": params}, context, target)
            terms = _losses(outputs, modalities, sigreg_key)
            total = (
                (1.0 - sigreg_weight) * terms["prediction"]
                + sigreg_weight * terms["sigreg"]
                + alignment_weight * terms["alignment"]
                + covariance_weight * terms["covariance"]
            )
            return total, terms

        (total, terms), grads = jax.value_and_grad(objective, has_aux=True)(state.params)
        updates, opt_state = optimizer.update(grads, state.opt_state, state.params)
        params = optax.apply_updates(state.params, updates)
        new_state = state.replace(
            params=params,
            opt_state=opt_state,
            step=state.step + 1,
            rng=next_rng,
        )
        return new_state, {"loss": total, **terms}

    return train_step
