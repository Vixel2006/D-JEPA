"""Sliced characteristic-function regularization for representation batches."""

import jax
import jax.numpy as jnp
from flax import linen as nn

from ..losses import loss_sigreg


class SIGReg(nn.Module):
    """Penalize projected embeddings whose marginals differ from a standard normal.

    The loss compares empirical and standard-normal characteristic functions over
    random unit directions and Epps–Pulley quadrature frequencies. Pass a PRNG key to
    ``__call__`` so sampling stays explicit and works with JAX transformations.
    """

    num_projections: int = 256
    num_frequencies: int = 17
    max_frequency: float = 3.0

    @nn.compact
    def __call__(self, embeddings: jnp.ndarray, key: jax.Array) -> jnp.ndarray:
        return loss_sigreg(
            embeddings,
            key,
            num_projections=self.num_projections,
            num_frequencies=self.num_frequencies,
            max_frequency=self.max_frequency,
        )
