"""A compact patch-embedding Transformer for NHWC images."""

import jax.numpy as jnp
from flax import linen as nn

from .transformer import TransformerEncoder, positional_embeddings


class VisionEncoder(nn.Module):
    """Encode an image batch into one vector per image."""

    embedding_dim: int = 256
    patch_size: int = 16
    num_heads: int = 4
    num_layers: int = 2
    mlp_dim: int = 512

    @nn.compact
    def __call__(self, images: jnp.ndarray) -> jnp.ndarray:
        x = nn.Conv(
            self.embedding_dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding="SAME",
            name="patch_embedding",
        )(images.astype(jnp.float32))
        batch, height, width, channels = x.shape
        tokens = x.reshape(batch, height * width, channels)
        tokens += positional_embeddings(tokens.shape[1], self.embedding_dim, tokens.dtype)[None]
        tokens = TransformerEncoder(
            self.embedding_dim, self.num_heads, self.num_layers, self.mlp_dim
        )(tokens)
        return jnp.mean(tokens, axis=1)
