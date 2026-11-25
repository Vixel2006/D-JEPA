"""Alignment, covariance, and sliced Gaussian regularization losses."""

import jax
import jax.numpy as jnp


def loss_alignment(first: jnp.ndarray, second: jnp.ndarray) -> jnp.ndarray:
    """Mean squared distance between corresponding embeddings from two views."""
    return jnp.mean(jnp.square(first - second))


def loss_covariance(embeddings: jnp.ndarray) -> jnp.ndarray:
    """Penalize correlations between different embedding dimensions.

    ``embeddings`` has shape ``(batch, features)``. The diagonal variance is
    left unconstrained; only off-diagonal covariance entries contribute.
    """
    centered = embeddings - jnp.mean(embeddings, axis=0, keepdims=True)
    covariance = centered.T @ centered / jnp.maximum(embeddings.shape[0] - 1, 1)
    off_diagonal = covariance - jnp.diag(jnp.diag(covariance))
    return jnp.sum(jnp.square(off_diagonal)) / embeddings.shape[-1]


def loss_cross_covariance(first: jnp.ndarray, second: jnp.ndarray) -> jnp.ndarray:
    """Penalize dependence between two representation spaces of one modality."""
    first = first - jnp.mean(first, axis=0, keepdims=True)
    second = second - jnp.mean(second, axis=0, keepdims=True)
    covariance = first.T @ second / jnp.maximum(first.shape[0] - 1, 1)
    return jnp.mean(jnp.square(covariance))


def loss_sigreg(
    embeddings: jnp.ndarray,
    key: jax.Array,
    num_projections: int = 256,
    num_frequencies: int = 17,
    max_frequency: float = 3.0,
) -> jnp.ndarray:
    """Epps–Pulley SIGReg over random directions and quadrature frequencies."""
    directions = jax.random.normal(
        key, (embeddings.shape[-1], num_projections)
    )
    directions = directions / jnp.maximum(
        jnp.linalg.norm(directions, axis=0, keepdims=True), 1e-8
    )
    frequencies = jnp.linspace(-max_frequency, max_frequency, num_frequencies)

    phase = (embeddings @ directions)[..., None] * frequencies
    empirical_real = jnp.mean(jnp.cos(phase), axis=0)
    empirical_imag = jnp.mean(jnp.sin(phase), axis=0)
    target_real = jnp.exp(-0.5 * jnp.square(frequencies))
    error = jnp.square(empirical_real - target_real) + jnp.square(empirical_imag)
    weighted_error = error * jnp.exp(-0.5 * jnp.square(frequencies))
    intervals = jnp.diff(frequencies)
    integral = jnp.sum(
        0.5 * (weighted_error[..., 1:] + weighted_error[..., :-1]) * intervals,
        axis=-1,
    )
    return jnp.mean(integral)
