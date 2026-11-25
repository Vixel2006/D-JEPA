"""A token embedding Transformer encoder."""

import jax.numpy as jnp
from flax import linen as nn

from .transformer import TransformerEncoder, positional_embeddings


class TextEncoder(nn.Module):
    """Encode integer token IDs shaped ``(batch, length)``."""

    vocab_size: int
    embedding_dim: int = 256
    num_heads: int = 4
    num_layers: int = 2
    mlp_dim: int = 512

    @nn.compact
    def __call__(
        self, token_ids: jnp.ndarray, attention_mask: jnp.ndarray | None = None
    ) -> jnp.ndarray:
        x = nn.Embed(self.vocab_size, self.embedding_dim)(token_ids)
        if attention_mask is None:
            attention_mask = jnp.ones(token_ids.shape, dtype=x.dtype)
        x += positional_embeddings(token_ids.shape[1], self.embedding_dim, x.dtype)[None]
        x = TransformerEncoder(self.embedding_dim, self.num_heads, self.num_layers, self.mlp_dim)(
            x, attention_mask
        )
        mask = attention_mask.astype(x.dtype)[..., None]
        return jnp.sum(x * mask, axis=1) / jnp.maximum(jnp.sum(mask, axis=1), 1.0)
