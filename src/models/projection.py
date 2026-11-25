"""A feed-forward projection head for modality embeddings."""

import jax.numpy as jnp
from flax import linen as nn


class ProjectionHead(nn.Module):
    """Map an embedding to a shared representation space."""

    output_dim: int
    hidden_dim: int | None = None

    @nn.compact
    def __call__(self, embedding: jnp.ndarray) -> jnp.ndarray:
        hidden_dim = self.hidden_dim or embedding.shape[-1]
        x = nn.Dense(hidden_dim)(embedding)
        x = nn.gelu(x)
        return nn.Dense(self.output_dim)(x)
