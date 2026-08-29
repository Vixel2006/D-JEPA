import jax.numpy as jnp
import flax.nnx as nnx
import chex


class RoPE(nnx.Module):
    def __init__(
        self,
        d_model: int,
        base: float = 10000.0,
        max_seq_len: int = 2048,
    ):
        assert d_model % 2 == 0, "RoPE requires even number of dimensions"
        
        self.d_model = d_model
        self.base = base
        self.max_seq_len = max_seq_len
        
        half_dim = d_model // 2
        freqs = 1.0 / (base ** (jnp.arange(0, half_dim, dtype=jnp.float32) * 2.0 / d_model))
        
        t = jnp.arange(max_seq_len, dtype=jnp.float32)
        freqs = jnp.outer(t, freqs)
        
        self.cos_cache = nnx.Variable(jnp.cos(freqs))
        self.sin_cache = nnx.Variable(jnp.sin(freqs))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        seq_len = x.shape[-2]
        half_dim = self.d_model // 2
        
        cos = self.cos_cache.value[:seq_len]
        sin = self.sin_cache.value[:seq_len]
        
        x_reshaped = x.reshape(*x.shape[:-1], half_dim, 2)
        x1 = x_reshaped[..., 0]
        x2 = x_reshaped[..., 1]
        
        cos = cos.reshape((1,) * (x.ndim - 2) + (seq_len, half_dim))
        sin = sin.reshape((1,) * (x.ndim - 2) + (seq_len, half_dim))
        
        x1_rot = x1 * cos - x2 * sin
        x2_rot = x1 * sin + x2 * cos
        
        x_rot = jnp.stack([x1_rot, x2_rot], axis=-1)
        return x_rot.reshape(*x.shape[:-1], self.d_model)

    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.__call__(x)