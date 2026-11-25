"""A feature-embedding Transformer encoder for audio sequences."""

import jax.numpy as jnp
from flax import linen as nn

from .transformer import TransformerEncoder, positional_embeddings


class AudioEncoder(nn.Module):
    """Encode audio features shaped ``(batch, time, features)``."""

    embedding_dim: int = 256
    num_heads: int = 4
    num_layers: int = 2
    mlp_dim: int = 512

    @nn.compact
    def __call__(self, audio: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.embedding_dim, name="feature_embedding")(audio.astype(jnp.float32))
        x += positional_embeddings(x.shape[1], self.embedding_dim, x.dtype)[None]
        x = TransformerEncoder(self.embedding_dim, self.num_heads, self.num_layers, self.mlp_dim)(x)
        return jnp.mean(x, axis=1)
