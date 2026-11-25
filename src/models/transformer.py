"""Small pre-normalized Transformer blocks shared by modality encoders."""

import jax.numpy as jnp
from flax import linen as nn


class TransformerBlock(nn.Module):
    embedding_dim: int
    num_heads: int
    mlp_dim: int

    @nn.compact
    def __call__(
        self, tokens: jnp.ndarray, attention_mask: jnp.ndarray | None = None
    ) -> jnp.ndarray:
        x = nn.LayerNorm()(tokens)
        x = nn.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.embedding_dim,
            out_features=self.embedding_dim,
        )(x, mask=attention_mask)
        tokens = tokens + x

        x = nn.LayerNorm()(tokens)
        x = nn.Dense(self.mlp_dim)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.embedding_dim)(x)
        return tokens + x


class TransformerEncoder(nn.Module):
    """Apply a stack of Transformer blocks to a batch of token sequences."""

    embedding_dim: int = 256
    num_heads: int = 4
    num_layers: int = 2
    mlp_dim: int = 512

    @nn.compact
    def __call__(
        self, tokens: jnp.ndarray, attention_mask: jnp.ndarray | None = None
    ) -> jnp.ndarray:
        if attention_mask is not None:
            attention_mask = nn.make_attention_mask(attention_mask, attention_mask, dtype=jnp.bool_)
        for layer in range(self.num_layers):
            tokens = TransformerBlock(
                self.embedding_dim,
                self.num_heads,
                self.mlp_dim,
                name=f"block_{layer}",
            )(tokens, attention_mask)
        return nn.LayerNorm()(tokens)


def positional_embeddings(length: int, embedding_dim: int, dtype: jnp.dtype) -> jnp.ndarray:
    """Build sinusoidal positions for a sequence without length-specific parameters."""
    half_dim = (embedding_dim + 1) // 2
    positions = jnp.arange(length, dtype=jnp.float32)[:, None]
    scales = jnp.exp(-jnp.log(10000.0) * jnp.arange(half_dim) / max(half_dim - 1, 1))
    angles = positions * scales[None, :]
    return jnp.concatenate((jnp.sin(angles), jnp.cos(angles)), axis=-1)[:, :embedding_dim].astype(
        dtype
    )
