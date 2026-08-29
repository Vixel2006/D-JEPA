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


class ViT(nnx.Module):
    def __init__(
        self,
        rngs: nnx.Rngs,
        *,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 1000,
        d_model: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        use_rope: bool = True,
        rope_base: float = 10000.0,
        use_slots: bool = False,
        num_slots: int = 7,
        d_slot: int = 64,
        slot_iterations: int = 3,
        dropout: float = 0.0,
    ):
        chex.assert_scalar_positive(img_size)
        chex.assert_scalar_positive(patch_size)
        chex.assert_scalar_positive(in_channels)
        chex.assert_scalar_positive(num_classes)
        chex.assert_scalar_positive(d_model)
        chex.assert_scalar_positive(num_layers)
        chex.assert_scalar_positive(num_heads)
        assert (
            img_size % patch_size == 0
        ), f"img_size {img_size} must be divisible by patch_size {patch_size}"

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.use_slots = use_slots

        self.patch_embed = nnx.Conv(
            in_channels,
            d_model,
            kernel_size=(patch_size, patch_size),
            strides=(patch_size, patch_size),
            rngs=rngs,
        )

        if use_slots:
            seq_len = num_slots
        else:
            seq_len = self.num_patches + 1

        self.cls_token = nnx.Variable(jax.random.normal(rngs.params(), (1, 1, d_model)))

        max_seq_len = max(seq_len, 2048)
        self.pos_embed = nnx.Variable(
            jax.random.normal(rngs.params(), (1, max_seq_len, d_model))
        )

        self.blocks = nnx.List(
            [
                TransformerBlock(
                    rngs=rngs,
                    d_model=d_model,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    use_rope=use_rope,
                    max_seq_len=max_seq_len,
                    rope_base=rope_base,
                    use_slots=use_slots,
                    num_slots=num_slots,
                    d_slot=d_slot,
                    slot_iterations=slot_iterations,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm = nnx.BatchNorm(d_model, rngs=rngs)
        self.head = nnx.Linear(d_model, num_classes, rngs=rngs)
        self.dropout = dropout

    def __call__(self, x: jax.Array) -> jax.Array:
        chex.assert_rank(x, 4)
        batch_size, h, w, c = x.shape
        chex.assert_equal(h, self.img_size)
        chex.assert_equal(w, self.img_size)

        x = self.patch_embed(x)
        chex.assert_shape(
            x,
            (
                batch_size,
                self.img_size // self.patch_size,
                self.img_size // self.patch_size,
                self.d_model,
            ),
        )

        x = x.reshape(batch_size, self.num_patches, self.d_model)
        chex.assert_shape(x, (batch_size, self.num_patches, self.d_model))

        cls_token = jnp.broadcast_to(
            self.cls_token.value, (batch_size, 1, self.d_model)
        )
        x = jnp.concatenate([cls_token, x], axis=1)
        chex.assert_shape(x, (batch_size, self.num_patches + 1, self.d_model))

        seq_len = x.shape[1]
        x = x + self.pos_embed.value[:, :seq_len]
        chex.assert_shape(x, (batch_size, seq_len, self.d_model))

        for block in self.blocks:
            x = block(x)

        if self.use_slots:
            x = x.mean(
                axis=1
            )  # pool slots: (batch, num_slots, d_model) -> (batch, d_model)
        else:
            x = x[:, 0]  # cls token

        x = self.norm(x)
        x = self.head(x)
        chex.assert_shape(x, (batch_size, self.num_classes))
        return x


class BERT(nnx.Module):
    def __init__(
        self,
        rngs: nnx.Rngs,
        *,
        vocab_size: int = 30522,
        max_seq_len: int = 512,
        num_classes: int = 2,
        d_model: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        use_rope: bool = True,
        rope_base: float = 10000.0,
        use_slots: bool = False,
        num_slots: int = 7,
        d_slot: int = 64,
        slot_iterations: int = 3,
        dropout: float = 0.0,
    ):
        assert (
            d_model % num_heads == 0
        ), f"d_model {d_model} must be divisible by num_heads {num_heads}"

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.num_classes = num_classes
        self.d_model = d_model
        self.num_layers = num_layers
        self.use_slots = use_slots

        self.token_embedding = nnx.Embed(vocab_size, d_model, rngs=rngs)
        self.pos_embedding = nnx.Variable(
            jax.random.normal(rngs.params(), (1, max_seq_len, d_model))
        )

        if use_slots:
            seq_len = num_slots
        else:
            seq_len = max_seq_len

        self.blocks = nnx.List(
            [
                TransformerBlock(
                    rngs=rngs,
                    d_model=d_model,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    use_rope=use_rope,
                    max_seq_len=max_seq_len,
                    rope_base=rope_base,
                    use_slots=use_slots,
                    num_slots=num_slots,
                    d_slot=d_slot,
                    slot_iterations=slot_iterations,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm = nnx.LayerNorm(d_model, rngs=rngs)
        self.pooler = nnx.Linear(d_model, d_model, rngs=rngs)
        self.classifier = nnx.Linear(d_model, num_classes, rngs=rngs)
        self.dropout = dropout

    def __call__(
        self, input_ids: jax.Array, attention_mask: jax.Array | None = None
    ) -> jax.Array:
        chex.assert_rank(input_ids, 2)
        batch_size, seq_len = input_ids.shape
        assert (
            seq_len <= self.max_seq_len
        ), f"seq_len {seq_len} > max_seq_len {self.max_seq_len}"

        if attention_mask is not None:
            chex.assert_shape(attention_mask, (batch_size, seq_len))

        x = self.token_embedding(input_ids)
        chex.assert_shape(x, (batch_size, seq_len, self.d_model))

        x = x + self.pos_embedding.value[:, :seq_len]
        chex.assert_shape(x, (batch_size, seq_len, self.d_model))

        for block in self.blocks:
            x = block(x)

        if self.use_slots:
            x = x.mean(axis=1)
            chex.assert_shape(x, (batch_size, self.d_model))
        else:
            x = x[:, 0]
            chex.assert_shape(x, (batch_size, self.d_model))

        x = self.norm(x)
        x = self.pooler(x)
        x = jax.nn.gelu(x)
        x = self.classifier(x)
        chex.assert_shape(x, (batch_size, self.num_classes))
        return x
