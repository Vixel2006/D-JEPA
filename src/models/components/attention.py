import jax
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
        freqs = 1.0 / (
            base ** (jnp.arange(0, half_dim, dtype=jnp.float32) * 2.0 / d_model)
        )

        t = jnp.arange(max_seq_len, dtype=jnp.float32)
        freqs = jnp.outer(t, freqs)

        chex.assert_shape(freqs, (max_seq_len, half_dim))

        self.cos_cache = nnx.Variable(jnp.cos(freqs))
        self.sin_cache = nnx.Variable(jnp.sin(freqs))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        assert x.ndim in (2, 3), f"RoPE expects 2D or 3D input, got {x.ndim}D"
        *batch_dims, seq_len, d_model = x.shape
        chex.assert_equal(d_model, self.d_model)
        chex.assert_scalar_positive(seq_len)
        assert (
            seq_len <= self.max_seq_len
        ), f"seq_len {seq_len} > max_seq_len {self.max_seq_len}"

        half_dim = self.d_model // 2

        cos = self.cos_cache.value[:seq_len]
        sin = self.sin_cache.value[:seq_len]

        chex.assert_shape(cos, (seq_len, half_dim))
        chex.assert_shape(sin, (seq_len, half_dim))

        x_reshaped = x.reshape(*batch_dims, seq_len, half_dim, 2)
        x1 = x_reshaped[..., 0]
        x2 = x_reshaped[..., 1]

        chex.assert_shape(x1, (*batch_dims, seq_len, half_dim))
        chex.assert_shape(x2, (*batch_dims, seq_len, half_dim))

        cos = cos.reshape((1,) * len(batch_dims) + (seq_len, half_dim))
        sin = sin.reshape((1,) * len(batch_dims) + (seq_len, half_dim))

        x1_rot = x1 * cos - x2 * sin
        x2_rot = x1 * sin + x2 * cos

        x_rot = jnp.stack([x1_rot, x2_rot], axis=-1)
        out = x_rot.reshape(*batch_dims, seq_len, self.d_model)

        chex.assert_shape(out, (*batch_dims, seq_len, self.d_model))
        return out


class MultiHeadAttention(nnx.Module):
    def __init__(
        self,
        rngs: nnx.Rngs,
        d_model: int,
        num_heads: int,
        use_rope: bool = True,
        max_seq_len: int = 2048,
        rope_base: float = 10000.0,
    ):
        chex.assert_scalar_positive(d_model)
        chex.assert_scalar_positive(num_heads)
        assert (
            d_model % num_heads == 0
        ), f"d_model {d_model} must be divisible by num_heads {num_heads}"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.use_rope = use_rope

        self.qkv = nnx.Linear(d_model, d_model * 3, rngs=rngs, use_bias=False)
        self.out_proj = nnx.Linear(d_model, d_model, rngs=rngs, use_bias=False)

        if use_rope:
            self.rope = RoPE(
                d_model=self.head_dim, base=rope_base, max_seq_len=max_seq_len
            )

    def __call__(self, x: jax.Array) -> jax.Array:
        chex.assert_rank(x, 3)
        batch_size, seq_len, d_model = x.shape
        chex.assert_equal(d_model, self.d_model)

        qkv = self.qkv(x)
        chex.assert_shape(qkv, (batch_size, seq_len, self.d_model * 3))

        q, k, v = jnp.split(qkv, 3, axis=-1)

        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )

        chex.assert_shape(q, (batch_size, self.num_heads, seq_len, self.head_dim))
        chex.assert_shape(k, (batch_size, self.num_heads, seq_len, self.head_dim))
        chex.assert_shape(v, (batch_size, self.num_heads, seq_len, self.head_dim))

        if self.use_rope:
            q = self.rope(
                q.reshape(batch_size * self.num_heads, seq_len, self.head_dim)
            )
            k = self.rope(
                k.reshape(batch_size * self.num_heads, seq_len, self.head_dim)
            )
            q = q.reshape(batch_size, self.num_heads, seq_len, self.head_dim)
            k = k.reshape(batch_size, self.num_heads, seq_len, self.head_dim)

        attn_logits = jnp.einsum("bhnd,bhmd->bhnm", q, k) / jnp.sqrt(self.head_dim)
        chex.assert_shape(attn_logits, (batch_size, self.num_heads, seq_len, seq_len))

        attn = jax.nn.softmax(attn_logits, axis=-1)
        chex.assert_shape(attn, (batch_size, self.num_heads, seq_len, seq_len))

        out = jnp.einsum("bhnm,bhmd->bhnd", attn, v)
        chex.assert_shape(out, (batch_size, self.num_heads, seq_len, self.head_dim))

        out = out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        chex.assert_shape(out, (batch_size, seq_len, self.d_model))

        out = self.out_proj(out)
        chex.assert_shape(out, (batch_size, seq_len, self.d_model))
        return out