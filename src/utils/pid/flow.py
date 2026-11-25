"""FlowPID: estimate PID atoms with Gaussianizing affine coupling flows."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.training import train_state

from .core import (
    gaussian_mutual_information,
    minimum_mutual_information_pid,
    quantile_gaussianize,
    whiten_and_reduce,
)


def make_flow_masks(dim: int, n_layers: int, seed: int) -> tuple[tuple[int, ...], ...]:
    """Create deterministic masks for affine coupling layers."""
    rng = np.random.default_rng(seed)
    masks = []
    for _ in range(n_layers):
        mask = np.zeros(dim, dtype=np.float32)
        if dim > 1:
            mask[rng.permutation(dim)[: max(1, dim // 2)]] = 1.0
        masks.append(tuple(mask.astype(int)))
    return tuple(masks)


class AffineFlow(nn.Module):
    dim: int
    hidden_dim: int
    masks: tuple[tuple[int, ...], ...]

    @nn.compact
    def __call__(self, inputs: jax.Array) -> tuple[jax.Array, jax.Array]:
        values = inputs
        logdet = jnp.zeros((inputs.shape[0],), dtype=inputs.dtype)
        for layer, mask_tuple in enumerate(self.masks):
            mask = jnp.asarray(mask_tuple, dtype=inputs.dtype)
            hidden = nn.Dense(self.hidden_dim, name=f"dense_{layer}_1")(values * mask)
            hidden = nn.gelu(hidden)
            hidden = nn.Dense(self.hidden_dim, name=f"dense_{layer}_2")(hidden)
            hidden = nn.gelu(hidden)
            shift_scale = nn.Dense(
                2 * self.dim,
                kernel_init=nn.initializers.zeros,
                bias_init=nn.initializers.zeros,
                name=f"shift_scale_{layer}",
            )(hidden)
            scale, shift = jnp.split(shift_scale, 2, axis=-1)
            changed = 1.0 - mask
            scale = jnp.tanh(scale) * changed
            shift = shift * changed
            values = mask * values + changed * (values * jnp.exp(scale) + shift)
            logdet += jnp.sum(scale, axis=-1)

        log_scale = self.param("log_scale", nn.initializers.zeros, (self.dim,))
        bias = self.param("bias", nn.initializers.zeros, (self.dim,))
        return values * jnp.exp(log_scale) + bias, logdet + jnp.sum(log_scale)


def estimate_flow_pid(
    first_train: np.ndarray,
    second_train: np.ndarray,
    target_train: np.ndarray,
    first_test: np.ndarray,
    second_test: np.ndarray,
    target_test: np.ndarray,
    *,
    steps: int = 1000,
    seed: int = 0,
    batch_size: int = 512,
    hidden_dim: int = 128,
    n_layers: int = 4,
    learning_rate: float = 2e-3,
) -> dict[str, float]:
    """Fit source flows on train data and estimate held-out MMI FlowPID."""
    first_train, second_train = np.asarray(first_train, np.float32), np.asarray(second_train, np.float32)
    first_test, second_test = np.asarray(first_test, np.float32), np.asarray(second_test, np.float32)
    target_train = np.asarray(target_train, np.float32).reshape(-1)
    target_test = np.asarray(target_test, np.float32).reshape(-1)
    if not (len(first_train) == len(second_train) == len(target_train)):
        raise ValueError("All training arrays must contain the same examples")
    if not (len(first_test) == len(second_test) == len(target_test)):
        raise ValueError("All test arrays must contain the same examples")
    if len(first_train) < 2 or len(first_test) < 2:
        raise ValueError("FlowPID requires at least two train and two test examples")
    if first_train.shape[1] == 0 or second_train.shape[1] == 0:
        raise ValueError("Each source must have at least one feature")

    y_train = quantile_gaussianize(target_train, target_train)
    y_test = quantile_gaussianize(target_train, target_test)
    dim_a, dim_b = first_train.shape[1], second_train.shape[1]
    flow_a = AffineFlow(dim_a, hidden_dim, make_flow_masks(dim_a, n_layers, seed))
    flow_b = AffineFlow(dim_b, hidden_dim, make_flow_masks(dim_b, n_layers, seed + 1))
    key_a, key_b = jax.random.split(jax.random.PRNGKey(seed))
    params_a = flow_a.init(key_a, jnp.asarray(first_train[:1]))["params"]
    params_b = flow_b.init(key_b, jnp.asarray(second_train[:1]))["params"]
    optimizer = optax.chain(optax.clip_by_global_norm(5.0), optax.adam(learning_rate))
    state_a = train_state.TrainState.create(apply_fn=flow_a.apply, params=params_a, tx=optimizer)
    state_b = train_state.TrainState.create(apply_fn=flow_b.apply, params=params_b, tx=optimizer)
    identity = jnp.eye(dim_a + dim_b + 1, dtype=jnp.float32)

    @jax.jit
    def train_step(state_a, state_b, batch_a, batch_b, batch_y):
        def objective(params_a, params_b):
            z_a, logdet_a = flow_a.apply({"params": params_a}, batch_a)
            z_b, logdet_b = flow_b.apply({"params": params_b}, batch_b)
            joint = jnp.concatenate((z_a, z_b, batch_y[:, None]), axis=-1)
            centered = joint - jnp.mean(joint, axis=0, keepdims=True)
            covariance = centered.T @ centered / (joint.shape[0] - 1.0) + 1e-4 * identity
            return 0.5 * jnp.linalg.slogdet(covariance)[1] - jnp.mean(logdet_a + logdet_b)

        _, (grads_a, grads_b) = jax.value_and_grad(objective, argnums=(0, 1))(
            state_a.params, state_b.params
        )
        return state_a.apply_gradients(grads=grads_a), state_b.apply_gradients(grads=grads_b)

    rng = np.random.default_rng(seed)
    for _ in range(steps):
        indices = rng.integers(0, len(first_train), size=min(batch_size, len(first_train)))
        state_a, state_b = train_step(
            state_a,
            state_b,
            jnp.asarray(first_train[indices]),
            jnp.asarray(second_train[indices]),
            jnp.asarray(y_train[indices]),
        )

    z_a = np.asarray(flow_a.apply({"params": state_a.params}, jnp.asarray(first_test))[0])
    z_b = np.asarray(flow_b.apply({"params": state_b.params}, jnp.asarray(second_test))[0])
    y_test = np.asarray(y_test).reshape(-1, 1)
    indices_a = list(range(dim_a))
    indices_b = list(range(dim_a, dim_a + dim_b))
    target_index = [dim_a + dim_b]

    def information(target_values: np.ndarray) -> tuple[float, float, float]:
        combined = np.concatenate((z_a, z_b, target_values.reshape(-1, 1)), axis=-1)
        centered_values = combined - combined.mean(axis=0)
        covariance_values = centered_values.T @ centered_values / max(len(combined) - 1, 1)
        covariance_values += 1e-5 * np.eye(covariance_values.shape[0])
        return (
            gaussian_mutual_information(covariance_values, indices_a, target_index),
            gaussian_mutual_information(covariance_values, indices_b, target_index),
            gaussian_mutual_information(covariance_values, indices_a + indices_b, target_index),
        )

    result = minimum_mutual_information_pid(*information(y_test[:, 0]))
    shuffled_target = y_test[rng.permutation(len(y_test)), 0]
    result["null_I12"] = max(information(shuffled_target)[2], 0.0)
    return result


def estimate_flow_pid_pair(
    blocks: dict[str, np.ndarray],
    first_keys: list[str],
    second_keys: list[str],
    target: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    *,
    pid_dim: int = 32,
    **flow_options,
) -> dict[str, float] | None:
    """Select feature groups, reduce dimensions, then estimate FlowPID."""
    target = np.asarray(target)
    valid = np.isfinite(target).all(axis=-1) if target.ndim > 1 else np.isfinite(target)
    train = train_indices[valid[train_indices]]
    test = test_indices[valid[test_indices]]
    if len(train) < 20 or len(test) < 10:
        return None
    first = np.concatenate([blocks[key] for key in first_keys], axis=-1)
    second = np.concatenate([blocks[key] for key in second_keys], axis=-1)
    a_train, a_test = whiten_and_reduce(first[train], first[test], pid_dim)
    b_train, b_test = whiten_and_reduce(second[train], second[test], pid_dim)
    if a_train.shape[1] == 0 or b_train.shape[1] == 0:
        return None
    if target.ndim > 1:
        # The current PID estimator treats the target as one scalar; callers can
        # evaluate each target coordinate separately to retain interpretable atoms.
        raise ValueError("Pass one scalar target per FlowPID estimate")
    return estimate_flow_pid(
        a_train,
        b_train,
        target[train],
        a_test,
        b_test,
        target[test],
        **flow_options,
    )
