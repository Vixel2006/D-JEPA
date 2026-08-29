import jax
import jax.numpy as jnp
import flax.nnx as nnx
import chex

rng = jax.random.PRNGKey(42)


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
        assert seq_len <= self.max_seq_len, f"seq_len {seq_len} > max_seq_len {self.max_seq_len}"

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


class SlotAttention(nnx.Module):
    def __init__(
        self,
        rngs: nnx.Rngs,
        num_slots: int,
        d_slot: int,
        d_model: int,
        iterations: int = 3,
    ):
        self.rngs = rngs
        self.num_slots = num_slots
        self.d_slot = d_slot
        self.iterations = iterations

        self.q = nnx.Linear(d_slot, d_slot, rngs=rngs, use_bias=False)
        self.k = nnx.Linear(d_model, d_slot, rngs=rngs, use_bias=False)
        self.v = nnx.Linear(d_model, d_slot, rngs=rngs, use_bias=False)

        self.slots_mu = nnx.Variable(jax.random.normal(rngs.params(), (1, 1, d_slot)))
        self.slots_log_sigma = nnx.Variable(jnp.zeros((1, 1, d_slot)))

        self.norm_inputs = nnx.BatchNorm(d_model, rngs=rngs)
        self.norm_slots = nnx.BatchNorm(d_slot, rngs=rngs)
        self.norm_mlp = nnx.BatchNorm(d_slot, rngs=rngs)

        self.gru = nnx.GRUCell(d_slot, d_slot, rngs=rngs)

        self.mlp = nnx.Sequential(
            nnx.Linear(d_slot, d_slot * 4, rngs=rngs),
            nnx.relu,
            nnx.Linear(d_slot * 4, d_slot, rngs=rngs),
        )

    def init_first_slot(self, batch_size: int):
        mu = self.slots_mu.value
        log_sigma = self.slots_log_sigma.value
        sigma = jnp.exp(log_sigma)
        slots = mu + sigma * jax.random.normal(self.rngs.params(), (batch_size, self.num_slots, self.d_slot))
        return slots

    def __call__(self, x: jax.Array) -> jax.Array:
        chex.assert_rank(x, 3)
        batch_size, seq_len, d_model = x.shape
        chex.assert_shape(x, (batch_size, seq_len, d_model))

        x = self.norm_inputs(x)
        k = self.k(x)
        v = self.v(x)

        chex.assert_shape(k, (batch_size, seq_len, self.d_slot))
        chex.assert_shape(v, (batch_size, seq_len, self.d_slot))

        slots = self.init_first_slot(batch_size)
        chex.assert_shape(slots, (batch_size, self.num_slots, self.d_slot))

        for _ in range(self.iterations):
            slots_prev = slots
            slots = self.norm_slots(slots)

            q = self.q(slots)
            chex.assert_shape(q, (batch_size, self.num_slots, self.d_slot))

            attn_logits = jnp.einsum('bnd,bmd->bnm', q, k) / jnp.sqrt(self.d_slot)
            chex.assert_shape(attn_logits, (batch_size, self.num_slots, seq_len))

            attn = jax.nn.softmax(attn_logits, axis=-1)
            chex.assert_shape(attn, (batch_size, self.num_slots, seq_len))

            updates = jnp.einsum('bnm,bmd->bnd', attn, v)
            chex.assert_shape(updates, (batch_size, self.num_slots, self.d_slot))

            updates = self.norm_mlp(updates)

            slots = self.gru(slots_prev.reshape(-1, self.d_slot), updates.reshape(-1, self.d_slot))[0]
            slots = slots.reshape(batch_size, self.num_slots, self.d_slot)
            chex.assert_shape(slots, (batch_size, self.num_slots, self.d_slot))

            mlp_out = self.mlp(self.norm_mlp(slots))
            chex.assert_shape(mlp_out, (batch_size, self.num_slots, self.d_slot))

            slots = slots + mlp_out

        return slots

