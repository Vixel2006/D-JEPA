import jax
import jax.numpy as jnp
import flax.nnx as nnx
import chex
from .attention import MultiHeadAttention


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
        slots = mu + sigma * jax.random.normal(
            self.rngs.params(), (batch_size, self.num_slots, self.d_slot)
        )
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

            attn_logits = jnp.einsum("bnd,bmd->bnm", q, k) / jnp.sqrt(self.d_slot)
            chex.assert_shape(attn_logits, (batch_size, self.num_slots, seq_len))

            attn = jax.nn.softmax(attn_logits, axis=-1)
            chex.assert_shape(attn, (batch_size, self.num_slots, seq_len))

            updates = jnp.einsum("bnm,bmd->bnd", attn, v)
            chex.assert_shape(updates, (batch_size, self.num_slots, self.d_slot))

            updates = self.norm_mlp(updates)

            slots = self.gru(
                slots_prev.reshape(-1, self.d_slot), updates.reshape(-1, self.d_slot)
            )[0]
            slots = slots.reshape(batch_size, self.num_slots, self.d_slot)
            chex.assert_shape(slots, (batch_size, self.num_slots, self.d_slot))

            mlp_out = self.mlp(self.norm_mlp(slots))
            chex.assert_shape(mlp_out, (batch_size, self.num_slots, self.d_slot))

            slots = slots + mlp_out

        return slots


class TransformerBlock(nnx.Module):
    def __init__(
        self,
        rngs: nnx.Rngs,
        d_model: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_rope: bool = True,
        max_seq_len: int = 2048,
        rope_base: float = 10000.0,
        use_slots: bool = False,
        num_slots: int = 7,
        d_slot: int = 64,
        slot_iterations: int = 3,
    ):
        self.use_slots = use_slots
        self.d_model = d_model

        self.norm1 = nnx.BatchNorm(d_model, rngs=rngs)

        if use_slots:
            self.attn = SlotAttention(
                rngs=rngs,
                num_slots=num_slots,
                d_slot=d_slot,
                d_model=d_model,
                iterations=slot_iterations,
            )
            self.slot_proj = nnx.Linear(d_slot, d_model, rngs=rngs)
        else:
            self.attn = MultiHeadAttention(
                rngs=rngs,
                d_model=d_model,
                num_heads=num_heads,
                use_rope=use_rope,
                max_seq_len=max_seq_len,
                rope_base=rope_base,
            )

        self.norm2 = nnx.BatchNorm(d_model, rngs=rngs)
        mlp_dim = int(d_model * mlp_ratio)
        self.mlp = nnx.Sequential(
            nnx.Linear(d_model, mlp_dim, rngs=rngs),
            nnx.gelu,
            nnx.Linear(mlp_dim, d_model, rngs=rngs),
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        chex.assert_rank(x, 3)
        batch_size, seq_len, d_model = x.shape
        chex.assert_equal(d_model, self.d_model)

        if self.use_slots:
            x = self.norm1(x)
            x = self.attn(x)
            x = self.slot_proj(x)
        else:
            residual = x
            x = self.norm1(x)
            x = self.attn(x)
            x = residual + x

        x = self.norm2(x)
        x = x + self.mlp(x)
        return x