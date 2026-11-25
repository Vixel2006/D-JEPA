"""Encoder for precomputed modality features distributed by MultiBench."""

import jax.numpy as jnp
from flax import linen as nn

from .transformer import TransformerEncoder, positional_embeddings


class FeatureEncoder(nn.Module):
    """Encode floating point vectors or sequences shaped ``(B,D)`` / ``(B,T,D)``."""

    embedding_dim: int = 256
    num_heads: int = 4
    num_layers: int = 1
    mlp_dim: int = 512

    @nn.compact
    def __call__(self, features: jnp.ndarray) -> jnp.ndarray:
        values = features.astype(jnp.float32)
        if values.ndim == 2:
            values = values[:, None, :]
        if values.ndim != 3:
            raise ValueError(
                f"Expected (batch, features) or (batch, time, features), got {values.shape}"
            )
        values = nn.Dense(self.embedding_dim, name="feature_embedding")(values)
        values += positional_embeddings(values.shape[1], self.embedding_dim, values.dtype)[None]
        values = TransformerEncoder(
            self.embedding_dim, self.num_heads, self.num_layers, self.mlp_dim
        )(values)
        return jnp.mean(values, axis=1)
